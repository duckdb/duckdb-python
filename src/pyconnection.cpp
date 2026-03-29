#include "duckdb_python/pyconnection/pyconnection.hpp"

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table/read_csv.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/db_instance_cache.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/main/relation/read_csv_relation.hpp"
#include "duckdb/main/relation/read_json_relation.hpp"
#include "duckdb/main/relation/value_relation.hpp"
#include "duckdb/main/relation/view_relation.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb_python/map.hpp"
#include "duckdb_python/pandas/pandas_scan.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pystatement.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/python_conversion.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/numpy/numpy_array.hpp"
#include "duckdb_python/jupyter_progress_bar_display.hpp"
#include "duckdb_python/pyfilesystem.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb_python/nb/conversions/exception_handling_enum.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "duckdb_python/python_replacement_scan.hpp"
#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/stream_query_result.hpp"
#include "duckdb/main/relation/materialized_relation.hpp"
#include "duckdb/parser/statement/load_statement.hpp"
#include "duckdb_python/expression/pyexpression.hpp"
#include "duckdb_python/nb/conversions/python_csv_line_terminator_enum.hpp"

namespace duckdb {

// All process-global module state lives in one struct, reached only through GetModuleState().
// This is the single seam to retarget for PEP 489 multi-phase init (per-module state via
// PyModule_GetState); call sites never touch the storage directly.
struct DuckDBPyModuleState {
	DefaultConnectionHolder default_connection;
	DBInstanceCache instance_cache;
	std::shared_ptr<PythonImportCache> import_cache;
	PythonEnvironmentType environment = PythonEnvironmentType::NORMAL;
	std::string formatted_python_version;
};

static DuckDBPyModuleState &GetModuleState() {
	static DuckDBPyModuleState state; // NOLINT: allow global - sole module-state seam (future: PyModule_GetState)
	return state;
}

DuckDBPyConnection::~DuckDBPyConnection() {
	try {
		// The native Connection / DuckDB teardown is pure C++ work — release
		// the GIL for it so other Python threads can run. The implicit member
		// destructors that fire after this scope (notably
		// `registered_functions`, a `case_insensitive_map_t<unique_ptr<ExternalDependency>>`
		// whose entries transitively own Python references)
		// run with the GIL reacquired because `gil` is destroyed at the end
		// of the inner block.
		{
			nb::gil_scoped_release gil;
			con.SetDatabase(nullptr);
			con.SetConnection(nullptr);
		}
	} catch (...) { // NOLINT
	}
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::CreateRelation(shared_ptr<Relation> rel) {
	auto py_rel = std::make_unique<DuckDBPyRelation>(std::move(rel));
	nb::gil_scoped_acquire gil;
	py_rel->SetConnectionOwner(nb::cast(shared_from_this()));
	return py_rel;
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::CreateRelation(std::shared_ptr<DuckDBPyResult> result) {
	auto py_rel = std::make_unique<DuckDBPyRelation>(std::move(result));
	nb::gil_scoped_acquire gil;
	py_rel->SetConnectionOwner(nb::cast(shared_from_this()));
	return py_rel;
}

void DuckDBPyConnection::DetectEnvironment() {
	// Get the formatted Python version
	nb::module_ sys = nb::module_::import_("sys");
	nb::object version_info = sys.attr("version_info");
	int major = nb::cast<int>(version_info.attr("major"));
	int minor = nb::cast<int>(version_info.attr("minor"));
	GetModuleState().formatted_python_version = std::to_string(major) + "." + std::to_string(minor);

	// If __main__ does not have a __file__ attribute, we are in interactive mode
	auto main_module = nb::module_::import_("__main__");
	if (nb::hasattr(main_module, "__file__")) {
		return;
	}
	GetModuleState().environment = PythonEnvironmentType::INTERACTIVE;
	if (!ModuleIsLoaded<IpythonCacheItem>()) {
		return;
	}

	// Check to see if we are in a Jupyter Notebook
	auto &import_cache_py = *DuckDBPyConnection::ImportCache();
	auto get_ipython = import_cache_py.IPython.get_ipython();
	if (get_ipython.ptr() == nullptr) {
		// Could either not load the IPython module, or it has no 'get_ipython' attribute
		return;
	}
	auto ipython = get_ipython();
	if (!nb::hasattr(ipython, "config")) {
		return;
	}
	nb::dict ipython_config = ipython.attr("config");
	if (ipython_config.contains("IPKernelApp")) {
		GetModuleState().environment = PythonEnvironmentType::JUPYTER;
	}
	return;
}

bool DuckDBPyConnection::DetectAndGetEnvironment() {
	DuckDBPyConnection::DetectEnvironment();
	return DuckDBPyConnection::IsInteractive();
}

bool DuckDBPyConnection::IsJupyter() {
	return GetModuleState().environment == PythonEnvironmentType::JUPYTER;
}

std::string DuckDBPyConnection::FormattedPythonVersion() {
	return GetModuleState().formatted_python_version;
}

// NOTE: this function is generated by tools/pythonpkg/scripts/generate_connection_methods.py.
// Do not edit this function manually, your changes will be overwritten!

static void InitializeConnectionMethods(nb::class_<DuckDBPyConnection> &m) {
	m.def("cursor", &DuckDBPyConnection::Cursor, "Create a duplicate of the current connection");
	// .none() lets None reach RegisterFilesystem's body, which imports fsspec explicitly (surfacing
	// ModuleNotFoundError when fsspec is absent) before validating the instance.
	m.def("register_filesystem", &DuckDBPyConnection::RegisterFilesystem, "Register a fsspec compliant filesystem",
	      nb::arg("filesystem").none());
	m.def("unregister_filesystem", &DuckDBPyConnection::UnregisterFilesystem, "Unregister a filesystem",
	      nb::arg("name"));
	m.def("list_filesystems", &DuckDBPyConnection::ListFilesystems,
	      "List registered filesystems, including builtin ones");
	m.def("filesystem_is_registered", &DuckDBPyConnection::FileSystemIsRegistered,
	      "Check if a filesystem with the provided name is currently registered", nb::arg("name"));
	m.def("create_function", &DuckDBPyConnection::RegisterScalarUDF,
	      "Create a DuckDB function out of the passing in Python function so it can be used in queries",
	      nb::arg("name"), nb::arg("function"), nb::arg("parameters") = nb::none(),
	      nb::arg("return_type").none() = nb::none(), nb::kw_only(), nb::arg("type") = PythonUDFType::NATIVE,
	      nb::arg("null_handling") = FunctionNullHandling::DEFAULT_NULL_HANDLING,
	      nb::arg("exception_handling") = PythonExceptionHandling::FORWARD_ERROR, nb::arg("side_effects") = false);
	m.def("remove_function", &DuckDBPyConnection::UnregisterUDF, "Remove a previously created function",
	      nb::arg("name"));
	m.def("sqltype", &DuckDBPyConnection::Type, "Create a type object by parsing the 'type_str' string",
	      nb::arg("type_str"));
	m.def("dtype", &DuckDBPyConnection::Type, "Create a type object by parsing the 'type_str' string",
	      nb::arg("type_str"));
	m.def("type", &DuckDBPyConnection::Type, "Create a type object by parsing the 'type_str' string",
	      nb::arg("type_str"));
	m.def("array_type", &DuckDBPyConnection::ArrayType, "Create an array type object of 'type'",
	      nb::arg("type").none(false), nb::arg("size"));
	m.def("list_type", &DuckDBPyConnection::ListType, "Create a list type object of 'type'",
	      nb::arg("type").none(false));
	m.def("union_type", &DuckDBPyConnection::UnionType, "Create a union type object from 'members'",
	      nb::arg("members").none(false));
	m.def("string_type", &DuckDBPyConnection::StringType, "Create a string type with an optional collation",
	      nb::arg("collation") = "");
	m.def("enum_type", &DuckDBPyConnection::EnumType,
	      "Create an enum type of underlying 'type', consisting of the list of 'values'", nb::arg("name"),
	      nb::arg("type"), nb::arg("values"));
	m.def("decimal_type", &DuckDBPyConnection::DecimalType, "Create a decimal type with 'width' and 'scale'",
	      nb::arg("width"), nb::arg("scale"));
	m.def("struct_type", &DuckDBPyConnection::StructType, "Create a struct type object from 'fields'",
	      nb::arg("fields"));
	m.def("row_type", &DuckDBPyConnection::StructType, "Create a struct type object from 'fields'", nb::arg("fields"));
	m.def("map_type", &DuckDBPyConnection::MapType, "Create a map type object from 'key_type' and 'value_type'",
	      nb::arg("key").none(false), nb::arg("value").none(false));
	m.def("duplicate", &DuckDBPyConnection::Cursor, "Create a duplicate of the current connection");
	m.def("execute", &DuckDBPyConnection::Execute,
	      "Execute the given SQL query, optionally using prepared statements with parameters set", nb::arg("query"),
	      nb::arg("parameters") = nb::none());
	m.def("executemany", &DuckDBPyConnection::ExecuteMany,
	      "Execute the given prepared statement multiple times using the list of parameter sets in parameters",
	      nb::arg("query"), nb::arg("parameters") = nb::none());
	m.def("close", &DuckDBPyConnection::Close, "Close the connection");
	m.def("interrupt", &DuckDBPyConnection::Interrupt, "Interrupt pending operations");
	m.def("query_progress", &DuckDBPyConnection::QueryProgress, "Query progress of pending operation");
	m.def("fetchone", &DuckDBPyConnection::FetchOne, "Fetch a single row from a result following execute");
	m.def("fetchmany", &DuckDBPyConnection::FetchMany, "Fetch the next set of rows from a result following execute",
	      nb::arg("size") = 1);
	m.def("fetchall", &DuckDBPyConnection::FetchAll, "Fetch all rows from a result following execute");
	m.def("fetchnumpy", &DuckDBPyConnection::FetchNumpy, "Fetch a result as list of NumPy arrays following execute");
	m.def("fetchdf", &DuckDBPyConnection::FetchDF, "Fetch a result as DataFrame following execute()", nb::kw_only(),
	      nb::arg("date_as_object") = false);
	m.def("fetch_df", &DuckDBPyConnection::FetchDF, "Fetch a result as DataFrame following execute()", nb::kw_only(),
	      nb::arg("date_as_object") = false);
	m.def("df", &DuckDBPyConnection::FetchDF, "Fetch a result as DataFrame following execute()", nb::kw_only(),
	      nb::arg("date_as_object") = false);
	m.def("fetch_df_chunk", &DuckDBPyConnection::FetchDFChunk,
	      "Fetch a chunk of the result as DataFrame following execute()", nb::arg("vectors_per_chunk") = 1,
	      nb::kw_only(), nb::arg("date_as_object") = false);
	m.def("pl", &DuckDBPyConnection::FetchPolars, "Fetch a result as Polars DataFrame following execute()",
	      nb::arg("rows_per_batch") = 1000000, nb::kw_only(), nb::arg("lazy") = false);
	m.def("to_arrow_table", &DuckDBPyConnection::FetchArrow, "Fetch a result as Arrow table following execute()",
	      nb::arg("batch_size") = 1000000);
	m.def("to_arrow_reader", &DuckDBPyConnection::FetchRecordBatchReader,
	      "Fetch an Arrow RecordBatchReader following execute()", nb::arg("batch_size") = 1000000);
	m.def(
	    "fetch_arrow_table",
	    [](DuckDBPyConnection &self, idx_t rows_per_batch) {
		    PyErr_WarnEx(PyExc_DeprecationWarning, "fetch_arrow_table() is deprecated, use to_arrow_table() instead.",
		                 0);
		    return self.FetchArrow(rows_per_batch);
	    },
	    "Fetch a result as Arrow table following execute()", nb::arg("rows_per_batch") = 1000000);
	m.def(
	    "fetch_record_batch",
	    [](DuckDBPyConnection &self, idx_t rows_per_batch) {
		    PyErr_WarnEx(PyExc_DeprecationWarning, "fetch_record_batch() is deprecated, use to_arrow_reader() instead.",
		                 0);
		    return self.FetchRecordBatchReader(rows_per_batch);
	    },
	    "Fetch an Arrow RecordBatchReader following execute()", nb::arg("rows_per_batch") = 1000000);
	m.def("arrow", &DuckDBPyConnection::FetchRecordBatchReader,
	      "Alias of to_arrow_reader(). We recommend using to_arrow_reader() instead.",
	      nb::arg("rows_per_batch") = 1000000);
	m.def("torch", &DuckDBPyConnection::FetchPyTorch, "Fetch a result as dict of PyTorch Tensors following execute()");
	m.def("tf", &DuckDBPyConnection::FetchTF, "Fetch a result as dict of TensorFlow Tensors following execute()");
	m.def("begin", &DuckDBPyConnection::Begin, "Start a new transaction");
	m.def("commit", &DuckDBPyConnection::Commit, "Commit changes performed within a transaction");
	m.def("rollback", &DuckDBPyConnection::Rollback, "Roll back changes performed within a transaction");
	m.def("checkpoint", &DuckDBPyConnection::Checkpoint,
	      "Synchronizes data in the write-ahead log (WAL) to the database data file (no-op for in-memory connections)");
	m.def("append", &DuckDBPyConnection::Append, "Append the passed DataFrame to the named table",
	      nb::arg("table_name"), nb::arg("df"), nb::kw_only(), nb::arg("by_name") = false);
	m.def("register", &DuckDBPyConnection::RegisterPythonObject,
	      "Register the passed Python Object value for querying with a view", nb::arg("view_name"),
	      nb::arg("python_object"));
	m.def("unregister", &DuckDBPyConnection::UnregisterPythonObject, "Unregister the view name", nb::arg("view_name"));
	m.def("table", &DuckDBPyConnection::Table, "Create a relation object for the named table", nb::arg("table_name"));
	m.def("view", &DuckDBPyConnection::View, "Create a relation object for the named view", nb::arg("view_name"));
	m.def("values", &DuckDBPyConnection::Values, "Create a relation object from the passed values");
	m.def("table_function", &DuckDBPyConnection::TableFunction,
	      "Create a relation object from the named table function with given parameters", nb::arg("name"),
	      nb::arg("parameters") = nb::none());
	m.def("read_json", &DuckDBPyConnection::ReadJSON, "Create a relation object from the JSON file in 'name'",
	      nb::arg("path_or_buffer"), nb::kw_only(), nb::arg("columns") = nb::none(),
	      nb::arg("sample_size") = nb::none(), nb::arg("maximum_depth") = nb::none(), nb::arg("records") = nb::none(),
	      nb::arg("format") = nb::none(), nb::arg("date_format") = nb::none(), nb::arg("timestamp_format") = nb::none(),
	      nb::arg("compression") = nb::none(), nb::arg("maximum_object_size") = nb::none(),
	      nb::arg("ignore_errors") = nb::none(), nb::arg("convert_strings_to_integers") = nb::none(),
	      nb::arg("field_appearance_threshold") = nb::none(), nb::arg("map_inference_threshold") = nb::none(),
	      nb::arg("maximum_sample_files") = nb::none(), nb::arg("filename") = nb::none(),
	      nb::arg("hive_partitioning") = nb::none(), nb::arg("union_by_name") = nb::none(),
	      nb::arg("hive_types") = nb::none(), nb::arg("hive_types_autocast") = nb::none());
	m.def("extract_statements", &DuckDBPyConnection::ExtractStatements,
	      "Parse the query string and extract the Statement object(s) produced", nb::arg("query"));
	m.def("sql", &DuckDBPyConnection::RunQuery,
	      "Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise "
	      "run the query as-is.",
	      nb::arg("query"), nb::kw_only(), nb::arg("alias") = "", nb::arg("params") = nb::none());
	m.def("query", &DuckDBPyConnection::RunQuery,
	      "Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise "
	      "run the query as-is.",
	      nb::arg("query"), nb::kw_only(), nb::arg("alias") = "", nb::arg("params") = nb::none());
	m.def("from_query", &DuckDBPyConnection::RunQuery,
	      "Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise "
	      "run the query as-is.",
	      nb::arg("query"), nb::kw_only(), nb::arg("alias") = "", nb::arg("params") = nb::none());
	// read_csv takes a positional source plus **kwargs of options. Bind via a nb::args lambda so None is accepted as
	// the source: a typed nb::object param would be rejected by nanobind before ReadCSV's body runs (and .none()
	// can't combine with nb::kwargs), whereas a nb::args tuple element may be None. ReadCSV itself raises the
	// "non file-like object" error for a None/invalid source.
	//
	// The pre-nanobind binding also advertised `path_or_buffer` as a positional-or-keyword parameter (the stubs
	// still do). nanobind's all-or-nothing nb::arg rule forbids naming just the source alongside **kwargs, so we
	// honor the keyword by pulling `path_or_buffer` out of kwargs when no positional source was given, and pop it
	// so ReadCSV's unknown-parameter check doesn't reject it.
	auto read_csv_fn = [](DuckDBPyConnection &self, nb::args args, nb::kwargs kwargs) {
		nb::object name = nb::none();
		if (args.size() >= 1) {
			name = nb::object(args[0]);
		} else if (kwargs.contains("path_or_buffer")) {
			name = kwargs["path_or_buffer"];
			PyDict_DelItemString(kwargs.ptr(), "path_or_buffer");
		}
		return self.ReadCSV(name, kwargs);
	};
	m.def("read_csv", read_csv_fn, "Create a relation object from the CSV file in 'name'");
	m.def("from_csv_auto", read_csv_fn, "Create a relation object from the CSV file in 'name'");
	m.def("from_df", &DuckDBPyConnection::FromDF, "Create a relation object from the DataFrame in df", nb::arg("df"));
	m.def("from_arrow", &DuckDBPyConnection::FromArrow, "Create a relation object from an Arrow object",
	      nb::arg("arrow_object"));
	m.def("from_parquet", &DuckDBPyConnection::FromParquet,
	      "Create a relation object from the Parquet path(s) or file-like object(s) in 'path_or_buffer'",
	      nb::arg("path_or_buffer"), nb::arg("binary_as_string") = false, nb::kw_only(),
	      nb::arg("file_row_number") = false, nb::arg("filename") = false, nb::arg("hive_partitioning") = false,
	      nb::arg("union_by_name") = false, nb::arg("compression") = nb::none());
	m.def("read_parquet", &DuckDBPyConnection::FromParquet,
	      "Create a relation object from the Parquet path(s) or file-like object(s) in 'path_or_buffer'",
	      nb::arg("path_or_buffer"), nb::arg("binary_as_string") = false, nb::kw_only(),
	      nb::arg("file_row_number") = false, nb::arg("filename") = false, nb::arg("hive_partitioning") = false,
	      nb::arg("union_by_name") = false, nb::arg("compression") = nb::none());
	m.def("get_table_names", &DuckDBPyConnection::GetTableNames, "Extract the required table names from a query",
	      nb::arg("query"), nb::kw_only(), nb::arg("qualified") = false);
	m.def("install_extension", &DuckDBPyConnection::InstallExtension,
	      "Install an extension by name, with an optional version and/or repository to get the extension from",
	      nb::arg("extension"), nb::kw_only(), nb::arg("force_install") = false, nb::arg("repository") = nb::none(),
	      nb::arg("repository_url") = nb::none(), nb::arg("version") = nb::none());
	m.def("load_extension", &DuckDBPyConnection::LoadExtension, "Load an installed extension", nb::arg("extension"));
	m.def("get_profiling_information", &DuckDBPyConnection::GetProfilingInformation,
	      "Get profiling information for a query", nb::arg("format") = "json");
	m.def("enable_profiling", &DuckDBPyConnection::EnableProfiling, "Enable profiling for subsequent queries");
	m.def("disable_profiling", &DuckDBPyConnection::DisableProfiling, "Disable profiling for subsequent queries");
} // END_OF_CONNECTION_METHODS

void DuckDBPyConnection::UnregisterFilesystem(const nb::str &name) {
	auto &database = con.GetDatabase();
	auto &fs = database.GetFileSystem();

	fs.ExtractSubSystem(nb::cast<std::string>(name));
}

void DuckDBPyConnection::RegisterFilesystem(nb::object filesystem) {
	nb::gil_scoped_acquire gil;

	auto &database = con.GetDatabase();
	// Import fsspec here (a normal, throwing context) so a missing install surfaces as ModuleNotFoundError, rather
	// than terminating inside the noexcept AbstractFileSystem type check (which nanobind cannot let throw).
	auto abstract_filesystem = nb::module_::import_("fsspec").attr("AbstractFileSystem");
	if (filesystem.is_none() || !duckdb::PyUtil::IsInstance(filesystem, abstract_filesystem)) {
		throw InvalidInputException("Bad filesystem instance");
	}

	auto &fs = database.GetFileSystem();

	// nb::object (not auto, which deduces an accessor): nb::str(protocol) below is an ambiguous overload on MSVC.
	nb::object protocol = filesystem.attr("protocol");
	if (protocol.is_none() || nb::str("abstract").equal(protocol)) {
		throw InvalidInputException("Must provide concrete fsspec implementation");
	}

	vector<string> protocols;
	if (nb::isinstance<nb::str>(protocol)) {
		protocols.push_back(nb::cast<std::string>(nb::str(protocol)));
	} else {
		for (const auto &sub_protocol : protocol) {
			protocols.push_back(nb::cast<std::string>(nb::str(sub_protocol)));
		}
	}

	fs.RegisterSubSystem(make_uniq<PythonFilesystem>(std::move(protocols), nb::borrow<AbstractFileSystem>(filesystem)));
}

nb::list DuckDBPyConnection::ListFilesystems() {
	auto &database = con.GetDatabase();
	auto subsystems = database.GetFileSystem().ListSubSystems();
	nb::list names;
	for (auto &name : subsystems) {
		names.append(nb::str(name.c_str(), name.size()));
	}
	return names;
}

nb::str DuckDBPyConnection::GetProfilingInformation(const string &format) {
	// We want to expose ProfilerPrintFormat as a string to Python users
	ProfilerPrintFormat format_enum;
	if (format == "html") {
		format_enum = ProfilerPrintFormat::HTML();
	} else if (format == "json") {
		format_enum = ProfilerPrintFormat::JSON();
	} else if (format == "graphviz") {
		format_enum = ProfilerPrintFormat::Graphviz();
	} else if (format == "default") {
		format_enum = ProfilerPrintFormat::Default();
	} else if (format == "mermaid") {
		format_enum = ProfilerPrintFormat::Mermaid();
	} else if (format == "text") {
		format_enum = ProfilerPrintFormat::Text();
	} else if (format == "yaml") {
		format_enum = ProfilerPrintFormat::YAML();
	} else {
		throw InvalidInputException(
		    "Invalid ProfilerPrintFormat string: " + std::string(format) +
		    ". Valid options are: query_tree, json, query_tree_optimizer, no_output, html, graphviz.");
	}
	auto &connection = con.GetConnection();
	auto profiling_info = connection.GetProfilingInformation(format_enum);
	return nb::str(profiling_info.c_str(), profiling_info.size());
}

void DuckDBPyConnection::EnableProfiling() {
	auto &connection = con.GetConnection();
	connection.EnableProfiling();
}

void DuckDBPyConnection::DisableProfiling() {
	auto &connection = con.GetConnection();
	connection.DisableProfiling();
}

nb::list DuckDBPyConnection::ExtractStatements(const string &query) {
	nb::list result;
	auto &connection = con.GetConnection();
	auto statements = connection.ExtractStatements(query);
	for (auto &statement : statements) {
		result.append(std::make_unique<DuckDBPyStatement>(std::move(statement)));
	}
	return result;
}

bool DuckDBPyConnection::FileSystemIsRegistered(const string &name) {
	auto &database = con.GetDatabase();
	auto subsystems = database.GetFileSystem().ListSubSystems();
	return std::find(subsystems.begin(), subsystems.end(), name) != subsystems.end();
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::UnregisterUDF(const string &name) {
	auto entry = registered_functions.find(name);
	if (entry == registered_functions.end()) {
		// Not registered or already unregistered
		throw InvalidInputException("No function by the name of '%s' was found in the list of registered functions",
		                            name);
	}

	auto &connection = con.GetConnection();
	auto &context = *connection.context;

	context.RunFunctionInTransaction([&]() {
		// create function
		auto &catalog = Catalog::GetCatalog(context, SYSTEM_CATALOG);
		DropInfo info;
		info.type = CatalogType::SCALAR_FUNCTION_ENTRY;
		info.SetName(Identifier(name));
		info.allow_drop_internal = true;
		info.cascade = false;
		info.if_not_found = OnEntryNotFound::THROW_EXCEPTION;
		catalog.DropEntry(context, info);
	});
	registered_functions.erase(entry);

	return shared_from_this();
}

std::shared_ptr<DuckDBPyConnection>
DuckDBPyConnection::RegisterScalarUDF(const string &name, const nb::callable &udf, const nb::object &parameters_p,
                                      const nb::object &return_type_p, PythonUDFType type,
                                      FunctionNullHandling null_handling, PythonExceptionHandling exception_handling,
                                      bool side_effects) {
	auto &connection = con.GetConnection();
	auto &context = *connection.context;

	if (context.transaction.HasActiveTransaction()) {
		context.CancelTransaction();
	}
	if (registered_functions.find(name) != registered_functions.end()) {
		throw NotImplementedException("A function by the name of '%s' is already created, creating multiple "
		                              "functions with the same name is not supported yet, please remove it first",
		                              name);
	}
	auto scalar_function = CreateScalarUDF(name, udf, parameters_p, return_type_p, type == PythonUDFType::ARROW,
	                                       null_handling, exception_handling, side_effects);
	CreateScalarFunctionInfo info(scalar_function);

	context.RegisterFunction(info);

	auto dependency = make_uniq<ExternalDependency>();
	dependency->AddDependency("function", PythonDependencyItem::Create(udf));
	registered_functions[name] = std::move(dependency);

	return shared_from_this();
}

void DuckDBPyConnection::Initialize(nb::handle &m) {
	// nanobind types aren't weak-referenceable by default;
	// otherwise weakref.ref/proxy/finalize on a connection raises TypeError.
	auto connection_module = nb::class_<DuckDBPyConnection>(m, "DuckDBPyConnection", nb::is_weak_referenceable());

	connection_module.def("__enter__", &DuckDBPyConnection::Enter)
	    .def(
	        "__exit__",
	        [](DuckDBPyConnection *self, const nb::object &exc_type, const nb::object &exc,
	           const nb::object &traceback) { DuckDBPyConnection::Exit(*self, exc_type, exc, traceback); },
	        nb::arg("exc_type").none(), nb::arg("exc").none(), nb::arg("traceback").none());
	connection_module.def("__del__", &DuckDBPyConnection::Close);

	InitializeConnectionMethods(connection_module);
	connection_module.def_prop_ro("description", &DuckDBPyConnection::GetDescription,
	                              "Get result set attributes, mainly column names");
	connection_module.def_prop_ro("rowcount", &DuckDBPyConnection::GetRowcount, "Get result set row count");
	PyDateTime_IMPORT; // NOLINT
	DuckDBPyConnection::ImportCache();
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::ExecuteMany(const nb::object &query, nb::object params_p) {
	nb::gil_scoped_acquire gil;
	ConnectionLockGuard conn_lock(*this);
	con.SetResult(nullptr);
	if (params_p.is_none()) {
		params_p = nb::list();
	}

	auto statements = GetStatements(query);
	if (statements.empty()) {
		// TODO: should we throw?
		return nullptr;
	}

	auto last_statement = std::move(statements.back());
	statements.pop_back();
	// First immediately execute any preceding statements (if any)
	// FIXME: DBAPI says to not accept an 'executemany' call with multiple statements
	ExecuteImmediately(std::move(statements));

	auto prep = PrepareQuery(std::move(last_statement));

	if (!duckdb::PyUtil::IsListLike(params_p)) {
		throw InvalidInputException("executemany requires a list of parameter sets to be provided");
	}
	auto outer_list = nb::list(params_p);
	if (outer_list.empty()) {
		throw InvalidInputException("executemany requires a non-empty list of parameter sets to be provided");
	}

	unique_ptr<QueryResult> query_result;
	// Execute once for every set of parameters that are provided
	for (auto parameters : outer_list) {
		auto params = nb::borrow<nb::object>(parameters);
		query_result = ExecuteInternal(*prep, std::move(params));
	}
	// Set the internal 'result' object
	if (query_result) {
		// Don't use CreateRelation here — the result is stored inside the connection,
		// so setting connection_owner would create a ref cycle (connection → result → connection).
		con.SetResult(std::make_unique<DuckDBPyRelation>(std::make_shared<DuckDBPyResult>(std::move(query_result))));
	}

	return shared_from_this();
}

unique_ptr<QueryResult> DuckDBPyConnection::CompletePendingQuery(PendingQueryResult &pending_query) {
	PendingExecutionResult execution_result;
	if (pending_query.HasError()) {
		pending_query.ThrowError();
	}
	while (!PendingQueryResult::IsResultReady(execution_result = pending_query.ExecuteTask())) {
		{
			nb::gil_scoped_acquire gil;
			if (PyErr_CheckSignals() != 0) {
				throw std::runtime_error("Query interrupted");
			}
		}
		if (execution_result == PendingExecutionResult::BLOCKED) {
			pending_query.WaitForTask();
		}
	}
	if (execution_result == PendingExecutionResult::EXECUTION_ERROR) {
		pending_query.ThrowError();
	}
	return pending_query.Execute();
}

nb::list TransformNamedParameters(const case_insensitive_map_t<idx_t> &named_param_map, const nb::dict &params) {
	// nanobind nb::list has no pre-sized constructor; pre-fill with None so indexed assignment below works
	nb::list new_params;
	for (idx_t i = 0; i < params.size(); i++) {
		new_params.append(nb::none());
	}

	for (auto item : params) {
		const std::string &item_name = duckdb::PyUtil::CastToString(item.first);
		auto entry = named_param_map.find(item_name);
		if (entry == named_param_map.end()) {
			throw InvalidInputException(
			    "Named parameters could not be transformed, because query string is missing named parameter '%s'",
			    item_name);
		}
		auto param_idx = entry->second;
		// Add the value of the named parameter to the list
		new_params[param_idx - 1] = item.second;
	}

	if (named_param_map.size() != params.size()) {
		// One or more named parameters were expected, but not found
		vector<string> missing_params;
		missing_params.reserve(named_param_map.size());
		for (auto &entry : named_param_map) {
			auto &name = entry.first;
			if (!params.contains(name)) {
				missing_params.push_back(name);
			}
		}
		auto message = StringUtil::Join(missing_params, ", ");
		throw InvalidInputException("Not all named parameters have been located, missing: %s", message);
	}

	return new_params;
}

identifier_map_t<BoundParameterData> TransformPreparedParameters(ClientContext &context, const nb::object &params,
                                                                 optional_ptr<PreparedStatement> prep = {}) {
	identifier_map_t<BoundParameterData> named_values;
	if (duckdb::PyUtil::IsListLike(params)) {
		if (prep && prep->named_param_map.size() != nb::len(params)) {
			if (nb::len(params) == 0) {
				throw InvalidInputException("Expected %d parameters, but none were supplied",
				                            prep->named_param_map.size());
			}
			throw InvalidInputException("Prepared statement needs %d parameters, %d given",
			                            prep->named_param_map.size(), nb::len(params));
		}
		auto unnamed_values = DuckDBPyConnection::TransformPythonParamList(context, params);
		for (idx_t i = 0; i < unnamed_values.size(); i++) {
			auto &value = unnamed_values[i];
			auto identifier = Identifier(std::to_string(i + 1));
			named_values[identifier] = BoundParameterData(std::move(value));
		}
	} else if (duckdb::PyUtil::IsDictLike(params)) {
		auto dict = nb::cast<nb::dict>(params);
		named_values = DuckDBPyConnection::TransformPythonParamDict(context, dict);
	} else {
		throw InvalidInputException("Prepared parameters can only be passed as a list or a dictionary");
	}
	return named_values;
}

unique_ptr<PreparedStatement> DuckDBPyConnection::PrepareQuery(unique_ptr<SQLStatement> statement) {
	auto &connection = con.GetConnection();
	unique_ptr<PreparedStatement> prep;
	{
		D_ASSERT(duckdb::PyUtil::GilCheck());
		nb::gil_scoped_release release;
		unique_lock<std::recursive_mutex> lock(py_connection_lock);

		prep = connection.Prepare(std::move(statement));
		if (prep->HasError()) {
			prep->error.Throw();
		}
	}
	return prep;
}

unique_ptr<QueryResult> DuckDBPyConnection::ExecuteInternal(PreparedStatement &prep, nb::object params) {
	if (params.is_none()) {
		params = nb::list();
	}
	auto &context = *con.GetConnection().context;

	// Execute the prepared statement with the prepared parameters
	auto named_values = TransformPreparedParameters(context, params, prep);
	unique_ptr<QueryResult> res;
	{
		D_ASSERT(duckdb::PyUtil::GilCheck());
		nb::gil_scoped_release release;
		unique_lock<std::recursive_mutex> lock(py_connection_lock);

		auto pending_query = prep.PendingQuery(named_values);
		if (pending_query->HasError()) {
			pending_query->ThrowError();
		}
		res = CompletePendingQuery(*pending_query);

		if (res->HasError()) {
			res->ThrowError();
		}
	}
	return res;
}

unique_ptr<QueryResult> DuckDBPyConnection::PrepareAndExecuteInternal(unique_ptr<SQLStatement> statement,
                                                                      nb::object params) {
	if (params.is_none()) {
		params = nb::list();
	}
	auto &context = *con.GetConnection().context;

	auto named_values = TransformPreparedParameters(context, params);

	unique_ptr<QueryResult> res;
	{
		D_ASSERT(duckdb::PyUtil::GilCheck());
		nb::gil_scoped_release release;
		unique_lock<std::recursive_mutex> lock(py_connection_lock);

		auto pending_query = con.GetConnection().PendingQuery(std::move(statement), named_values, true);

		if (pending_query->HasError()) {
			pending_query->ThrowError();
		}

		res = CompletePendingQuery(*pending_query);

		if (res->HasError()) {
			res->ThrowError();
		}
	}
	return res;
}

vector<unique_ptr<SQLStatement>> DuckDBPyConnection::GetStatements(const nb::object &query) {
	if (nb::isinstance<DuckDBPyStatement>(query)) {
		auto &statement_obj = nb::cast<DuckDBPyStatement &>(query);
		vector<unique_ptr<SQLStatement>> result;
		result.push_back(statement_obj.GetStatement());
		return result;
	}
	if (nb::isinstance<nb::str>(query)) {
		auto &connection = con.GetConnection();
		auto sql_query = nb::cast<std::string>(nb::str(query));
		auto statements = connection.ExtractStatements(sql_query);
		return std::move(statements);
	}
	throw InvalidInputException("Please provide either a DuckDBPyStatement or a string representing the query");
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::ExecuteFromString(const string &query) {
	return Execute(nb::str(query.c_str(), query.size()));
}

pair<string, nb::object> DuckDBPyConnection::ExtractCompiledSqlAndParams(const nb::object &query, nb::object params) {
	nb::object compiled = query;
	if (!nb::hasattr(compiled, "sql") || !nb::hasattr(compiled, "params")) {
		if (!nb::hasattr(query, "compile")) {
			return {string(), params};
		}
		compiled = query.attr("compile")();
	}

	if (!nb::hasattr(compiled, "sql") || !nb::hasattr(compiled, "params")) {
		return {string(), params};
	}

	auto compiled_sql = nb::cast<std::string>(compiled.attr("sql"));
	auto compiled_params_obj = compiled.attr("params");
	if (!duckdb::PyUtil::IsDictLike(compiled_params_obj)) {
		throw InvalidInputException("Compiled SQL parameters must be a dictionary");
	}

	auto compiled_params = nb::cast<nb::dict>(compiled_params_obj);
	if (compiled_params.size() == 0) {
		return {compiled_sql, params};
	}

	if (params.is_none()) {
		return {compiled_sql, compiled_params};
	}

	if (duckdb::PyUtil::IsDictLike(params)) {
		auto merged_params = nb::dict();
		for (auto item : compiled_params) { // nanobind dict iteration yields std::pair<handle,handle> by value
			merged_params[item.first] = item.second;
		}
		auto provided_params = nb::cast<nb::dict>(params);
		for (auto item : provided_params) { // nanobind dict iteration yields std::pair<handle,handle> by value
			if (merged_params.contains(item.first)) {
				throw nb::value_error("Cannot merge compiled SQL parameters with duplicate parameter names");
			}
			merged_params[item.first] = item.second;
		}
		return {compiled_sql, merged_params};
	}

	if (duckdb::PyUtil::IsListLike(params)) {
		if (nb::len(params) == 0) {
			return {compiled_sql, compiled_params};
		}
		throw nb::value_error("Cannot merge compiled SQL named parameters with positional parameters");
	}

	throw InvalidInputException("Prepared parameters can only be passed as a list or a dictionary");
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Execute(const nb::object &query, nb::object params) {
	nb::gil_scoped_acquire gil;
	ConnectionLockGuard conn_lock(*this);
	con.SetResult(nullptr);

	auto normalized_query = ExtractCompiledSqlAndParams(query, params);
	auto &compiled_sql = normalized_query.first;
	auto &merged_params = normalized_query.second;
	vector<unique_ptr<SQLStatement>> statements;
	if (!compiled_sql.empty()) {
		statements = GetStatements(nb::str(compiled_sql.c_str(), compiled_sql.size()));
		params = merged_params;
	} else {
		statements = GetStatements(query);
	}

	if (statements.empty()) {
		// TODO: should we throw?
		return nullptr;
	}

	auto last_statement = std::move(statements.back());
	statements.pop_back();
	// First immediately execute any preceding statements (if any)
	// FIXME: SQLites implementation says to not accept an 'execute' call with multiple statements
	ExecuteImmediately(std::move(statements));

	auto res = PrepareAndExecuteInternal(std::move(last_statement), std::move(params));

	// Set the internal 'result' object
	if (res) {
		// Don't use CreateRelation here — the result is stored inside the connection,
		// so setting connection_owner would create a ref cycle (connection → result → connection).
		con.SetResult(std::make_unique<DuckDBPyRelation>(std::make_shared<DuckDBPyResult>(std::move(res))));
	}
	return shared_from_this();
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Append(const string &name, const PandasDataFrame &value,
                                                               bool by_name) {
	RegisterPythonObject("__append_df", value);
	string columns = "";
	if (by_name) {
		auto df_columns = value.attr("columns");
		vector<string> column_names;
		for (auto column : df_columns) {
			column_names.push_back(nb::cast<std::string>(nb::str(column)));
		}
		columns += "(";
		for (idx_t i = 0; i < column_names.size(); i++) {
			auto &column = column_names[i];
			if (i != 0) {
				columns += ", ";
			}
			columns += StringUtil::Format("%s", SQLIdentifier(column));
		}
		columns += ")";
	}

	auto sql_query = StringUtil::Format("INSERT INTO %s %s SELECT * FROM __append_df", SQLIdentifier(name), columns);
	return Execute(nb::str(sql_query.c_str(), sql_query.size()));
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::RegisterPythonObject(const string &name,
                                                                             const nb::object &python_object) {
	auto &connection = con.GetConnection();
	auto &client = *connection.context;
	auto object = PythonReplacementScan::ReplacementObject(python_object, name, client);
	auto view_rel = make_shared_ptr<ViewRelation>(connection.context, std::move(object), name);
	bool replace = registered_objects.count(name);
	view_rel->CreateView(Identifier(name), replace, true);
	registered_objects.insert(name);
	return shared_from_this();
}

static void ParseMultiFileOptions(ClientContext &context, named_parameter_map_t &options,
                                  const Optional<nb::object> &filename, const Optional<nb::object> &hive_partitioning,
                                  const Optional<nb::object> &union_by_name, const Optional<nb::object> &hive_types,
                                  const Optional<nb::object> &hive_types_autocast) {
	if (!nb::none().is(filename)) {
		auto val = TransformPythonValue(context, filename);
		options["filename"] = val;
	}

	if (!nb::none().is(hive_types)) {
		auto val = TransformPythonValue(context, hive_types);
		options["hive_types"] = val;
	}

	if (!nb::none().is(hive_partitioning)) {
		if (!nb::isinstance<nb::bool_>(hive_partitioning)) {
			string actual_type = nb::cast<std::string>(nb::str((hive_partitioning).type()));
			throw BinderException("read_json only accepts 'hive_partitioning' as a boolean, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, hive_partitioning, LogicalTypeId::BOOLEAN);
		options["hive_partitioning"] = val;
	}

	if (!nb::none().is(union_by_name)) {
		if (!nb::isinstance<nb::bool_>(union_by_name)) {
			string actual_type = nb::cast<std::string>(nb::str((union_by_name).type()));
			throw BinderException("read_json only accepts 'union_by_name' as a boolean, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, union_by_name, LogicalTypeId::BOOLEAN);
		options["union_by_name"] = val;
	}

	if (!nb::none().is(hive_types_autocast)) {
		if (!nb::isinstance<nb::bool_>(hive_types_autocast)) {
			string actual_type = nb::cast<std::string>(nb::str((hive_types_autocast).type()));
			throw BinderException("read_json only accepts 'hive_types_autocast' as a boolean, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, hive_types_autocast, LogicalTypeId::BOOLEAN);
		options["hive_types_autocast"] = val;
	}
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::ReadJSON(
    const nb::object &name_p, const Optional<nb::object> &columns, const Optional<nb::object> &sample_size,
    const Optional<nb::object> &maximum_depth, const Optional<nb::str> &records, const Optional<nb::str> &format,
    const Optional<nb::object> &date_format, const Optional<nb::object> &timestamp_format,
    const Optional<nb::object> &compression, const Optional<nb::object> &maximum_object_size,
    const Optional<nb::object> &ignore_errors, const Optional<nb::object> &convert_strings_to_integers,
    const Optional<nb::object> &field_appearance_threshold, const Optional<nb::object> &map_inference_threshold,
    const Optional<nb::object> &maximum_sample_files, const Optional<nb::object> &filename,
    const Optional<nb::object> &hive_partitioning, const Optional<nb::object> &union_by_name,
    const Optional<nb::object> &hive_types, const Optional<nb::object> &hive_types_autocast) {

	named_parameter_map_t options;

	auto &connection = con.GetConnection();
	auto &context = *connection.context;
	auto path_like = GetPathLike(name_p);
	auto &name = path_like.files;
	auto file_like_object_wrapper = std::move(path_like.dependency);

	ParseMultiFileOptions(context, options, filename, hive_partitioning, union_by_name, hive_types,
	                      hive_types_autocast);

	if (!nb::none().is(columns)) {
		if (!duckdb::PyUtil::IsDictLike(columns)) {
			throw BinderException("read_json only accepts 'columns' as a dict[str, str]");
		}
		nb::dict columns_dict = nb::cast<nb::dict>(columns);
		child_list_t<Value> struct_fields;

		for (auto kv : columns_dict) { // nanobind dict iteration yields std::pair<handle,handle> by value
			auto column_name = kv.first;
			auto type = kv.second;
			if (!nb::isinstance<nb::str>(column_name)) {
				string actual_type = nb::cast<std::string>(nb::str((column_name).type()));
				throw BinderException("The provided column name must be a str, not of type '%s'", actual_type);
			}
			if (!nb::isinstance<nb::str>(type)) {
				string actual_type = nb::cast<std::string>(nb::str((column_name).type()));
				throw BinderException("The provided column type must be a str, not of type '%s'", actual_type);
			}
			struct_fields.emplace_back(nb::cast<std::string>(nb::str(column_name)), Value(nb::cast<std::string>(type)));
		}
		auto dtype_struct = Value::STRUCT(std::move(struct_fields));
		options["columns"] = std::move(dtype_struct);
	}

	if (!nb::none().is(records)) {
		if (!nb::isinstance<nb::str>(records)) {
			string actual_type = nb::cast<std::string>(nb::str((records).type()));
			throw BinderException("read_json only accepts 'records' as a string, not '%s'", actual_type);
		}
		auto records_s = nb::borrow<nb::str>(records);
		auto records_option = nb::cast<std::string>(nb::str(records_s));
		options["records"] = Value(records_option);
	}

	if (!nb::none().is(format)) {
		if (!nb::isinstance<nb::str>(format)) {
			string actual_type = nb::cast<std::string>(nb::str((format).type()));
			throw BinderException("read_json only accepts 'format' as a string, not '%s'", actual_type);
		}
		auto format_s = nb::borrow<nb::str>(format);
		auto format_option = nb::cast<std::string>(nb::str(format_s));
		options["format"] = Value(format_option);
	}

	if (!nb::none().is(date_format)) {
		if (!nb::isinstance<nb::str>(date_format)) {
			string actual_type = nb::cast<std::string>(nb::str((date_format).type()));
			throw BinderException("read_json only accepts 'date_format' as a string, not '%s'", actual_type);
		}
		auto date_format_s = nb::borrow<nb::str>(date_format);
		auto date_format_option = nb::cast<std::string>(nb::str(date_format_s));
		options["date_format"] = Value(date_format_option);
	}

	if (!nb::none().is(timestamp_format)) {
		if (!nb::isinstance<nb::str>(timestamp_format)) {
			string actual_type = nb::cast<std::string>(nb::str((timestamp_format).type()));
			throw BinderException("read_json only accepts 'timestamp_format' as a string, not '%s'", actual_type);
		}
		auto timestamp_format_s = nb::borrow<nb::str>(timestamp_format);
		auto timestamp_format_option = nb::cast<std::string>(nb::str(timestamp_format_s));
		options["timestamp_format"] = Value(timestamp_format_option);
	}

	if (!nb::none().is(compression)) {
		if (!nb::isinstance<nb::str>(compression)) {
			string actual_type = nb::cast<std::string>(nb::str((compression).type()));
			throw BinderException("read_json only accepts 'compression' as a string, not '%s'", actual_type);
		}
		auto compression_s = nb::borrow<nb::str>(compression);
		auto compression_option = nb::cast<std::string>(nb::str(compression_s));
		options["compression"] = Value(compression_option);
	}

	if (!nb::none().is(sample_size)) {
		if (!nb::isinstance<nb::int_>(sample_size)) {
			string actual_type = nb::cast<std::string>(nb::str((sample_size).type()));
			throw BinderException("read_json only accepts 'sample_size' as an integer, not '%s'", actual_type);
		}
		options["sample_size"] = Value::INTEGER((int32_t)nb::int_(sample_size));
	}

	if (!nb::none().is(maximum_depth)) {
		if (!nb::isinstance<nb::int_>(maximum_depth)) {
			string actual_type = nb::cast<std::string>(nb::str((maximum_depth).type()));
			throw BinderException("read_json only accepts 'maximum_depth' as an integer, not '%s'", actual_type);
		}
		options["maximum_depth"] = Value::INTEGER((int32_t)nb::int_(maximum_depth));
	}

	if (!nb::none().is(maximum_object_size)) {
		if (!nb::isinstance<nb::int_>(maximum_object_size)) {
			string actual_type = nb::cast<std::string>(nb::str((maximum_object_size).type()));
			throw BinderException("read_json only accepts 'maximum_object_size' as an unsigned integer, not '%s'",
			                      actual_type);
		}
		auto val = TransformPythonValue(context, maximum_object_size, LogicalTypeId::UINTEGER);
		options["maximum_object_size"] = val;
	}

	if (!nb::none().is(ignore_errors)) {
		if (!nb::isinstance<nb::bool_>(ignore_errors)) {
			string actual_type = nb::cast<std::string>(nb::str((ignore_errors).type()));
			throw BinderException("read_json only accepts 'ignore_errors' as a boolean, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, ignore_errors, LogicalTypeId::BOOLEAN);
		options["ignore_errors"] = val;
	}

	if (!nb::none().is(convert_strings_to_integers)) {
		if (!nb::isinstance<nb::bool_>(convert_strings_to_integers)) {
			string actual_type = nb::cast<std::string>(nb::str((convert_strings_to_integers).type()));
			throw BinderException("read_json only accepts 'convert_strings_to_integers' as a boolean, not '%s'",
			                      actual_type);
		}
		auto val = TransformPythonValue(context, convert_strings_to_integers, LogicalTypeId::BOOLEAN);
		options["convert_strings_to_integers"] = val;
	}

	if (!nb::none().is(field_appearance_threshold)) {
		if (!nb::isinstance<nb::float_>(field_appearance_threshold)) {
			string actual_type = nb::cast<std::string>(nb::str((field_appearance_threshold).type()));
			throw BinderException("read_json only accepts 'field_appearance_threshold' as a float, not '%s'",
			                      actual_type);
		}
		auto val = TransformPythonValue(context, field_appearance_threshold, LogicalTypeId::DOUBLE);
		options["field_appearance_threshold"] = val;
	}

	if (!nb::none().is(map_inference_threshold)) {
		if (!nb::isinstance<nb::int_>(map_inference_threshold)) {
			string actual_type = nb::cast<std::string>(nb::str((map_inference_threshold).type()));
			throw BinderException("read_json only accepts 'map_inference_threshold' as an integer, not '%s'",
			                      actual_type);
		}
		auto val = TransformPythonValue(context, map_inference_threshold, LogicalTypeId::BIGINT);
		options["map_inference_threshold"] = val;
	}

	if (!nb::none().is(maximum_sample_files)) {
		if (!nb::isinstance<nb::int_>(maximum_sample_files)) {
			string actual_type = nb::cast<std::string>(nb::str((maximum_sample_files).type()));
			throw BinderException("read_json only accepts 'maximum_sample_files' as an integer, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, maximum_sample_files, LogicalTypeId::BIGINT);
		options["maximum_sample_files"] = val;
	}

	bool auto_detect = false;
	if (!options.count("columns")) {
		options["auto_detect"] = Value::BOOLEAN(true);
		auto_detect = true;
	}

	D_ASSERT(duckdb::PyUtil::GilCheck());
	nb::gil_scoped_release gil;
	auto read_json_relation =
	    make_shared_ptr<ReadJSONRelation>(connection.context, name, std::move(options), auto_detect);
	if (read_json_relation == nullptr) {
		throw BinderException("read_json can only be used when the JSON extension is (statically) loaded");
	}
	if (file_like_object_wrapper) {
		read_json_relation->AddExternalDependency(std::move(file_like_object_wrapper));
	}
	return CreateRelation(std::move(read_json_relation));
}

PathLike DuckDBPyConnection::GetPathLike(const nb::object &object) {
	return PathLike::Create(object, *this);
}

static void AcceptableCSVOptions(const string &unkown_parameter) {
	// List of strings to match against
	const unordered_set<string> valid_parameters = {"header",
	                                                "strict_mode",
	                                                "compression",
	                                                "comment"
	                                                "sep",
	                                                "delimiter",
	                                                "files_to_sniff",
	                                                "dtype",
	                                                "na_values",
	                                                "skiprows",
	                                                "quotechar",
	                                                "escapechar",
	                                                "encoding",
	                                                "parallel",
	                                                "date_format",
	                                                "timestamp_format",
	                                                "sample_size",
	                                                "all_varchar",
	                                                "normalize_names",
	                                                "null_padding",
	                                                "names",
	                                                "lineterminator",
	                                                "columns",
	                                                "auto_type_candidates",
	                                                "max_line_size",
	                                                "ignore_errors",
	                                                "store_rejects",
	                                                "rejects_table",
	                                                "rejects_scan",
	                                                "rejects_limit",
	                                                "force_not_null",
	                                                "buffer_size",
	                                                "decimal",
	                                                "allow_quoted_nulls",
	                                                "filename",
	                                                "hive_partitioning",
	                                                "union_by_name",
	                                                "hive_types",
	                                                "hive_types_autocast",
	                                                "thousands"};

	std::ostringstream error;
	error << "The methods read_csv and read_csv_auto do not have the \"" << unkown_parameter << "\" argument." << '\n';
	error << "Possible arguments as suggestions: " << '\n';
	vector<string> parameters(valid_parameters.begin(), valid_parameters.end());
	auto suggestions = StringUtil::TopNJaroWinkler(parameters, unkown_parameter, 3);
	for (auto &suggestion : suggestions) {
		error << "* " << suggestion << '\n';
	}
	throw InvalidInputException(error.str());
}
void ConvertBooleanValue(const nb::object &value, string param_name, named_parameter_map_t &bind_parameters) {
	if (!nb::none().is(value)) {

		bool value_as_int = nb::isinstance<nb::int_>(value);
		bool value_as_bool = nb::isinstance<nb::bool_>(value);

		bool converted_value;
		if (value_as_bool) {
			converted_value = (bool)nb::bool_(value);
		} else if (value_as_int) {
			if (static_cast<int>(nb::int_(value)) != 0) {
				throw InvalidInputException("read_csv only accepts 0 if '%s' is given as an integer", param_name);
			}
			converted_value = true;
		} else {
			throw InvalidInputException("read_csv only accepts '%s' as an integer, or a boolean", param_name);
		}
		bind_parameters[Identifier(param_name)] = Value::BOOLEAN(converted_value);
	}
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::ReadCSV(const nb::object &name_p, nb::kwargs &kwargs) {
	nb::object header = nb::none();
	nb::object strict_mode = nb::none();
	nb::object auto_detect = nb::none();
	nb::object compression = nb::none();
	nb::object sep = nb::none();
	nb::object delimiter = nb::none();
	nb::object files_to_sniff = nb::none();
	nb::object dtype = nb::none();
	nb::object na_values = nb::none();
	nb::object skiprows = nb::none();
	nb::object quotechar = nb::none();
	nb::object escapechar = nb::none();
	nb::object encoding = nb::none();
	nb::object parallel = nb::none();
	nb::object date_format = nb::none();
	nb::object timestamp_format = nb::none();
	nb::object sample_size = nb::none();
	nb::object all_varchar = nb::none();
	nb::object normalize_names = nb::none();
	nb::object null_padding = nb::none();
	nb::object names_p = nb::none();
	nb::object lineterminator = nb::none();
	nb::object columns = nb::none();
	nb::object auto_type_candidates = nb::none();
	nb::object max_line_size = nb::none();
	nb::object ignore_errors = nb::none();
	nb::object store_rejects = nb::none();
	nb::object rejects_table = nb::none();
	nb::object rejects_scan = nb::none();
	nb::object rejects_limit = nb::none();
	nb::object force_not_null = nb::none();
	nb::object buffer_size = nb::none();
	nb::object decimal = nb::none();
	nb::object allow_quoted_nulls = nb::none();
	nb::object filename = nb::none();
	nb::object hive_partitioning = nb::none();
	nb::object union_by_name = nb::none();
	nb::object hive_types = nb::none();
	nb::object hive_types_autocast = nb::none();
	nb::object comment = nb::none();
	nb::object thousands_separator = nb::none();

	for (auto arg : kwargs) { // nanobind dict iteration yields std::pair<handle,handle> by value
		const auto &arg_name = nb::cast<std::string>(nb::str(arg.first));
		if (arg_name == "header") {
			header = kwargs[arg_name.c_str()];
		} else if (arg_name == "compression") {
			compression = kwargs[arg_name.c_str()];
		} else if (arg_name == "sep") {
			sep = kwargs[arg_name.c_str()];
		} else if (arg_name == "delimiter") {
			delimiter = kwargs[arg_name.c_str()];
		} else if (arg_name == "files_to_sniff") {
			files_to_sniff = kwargs[arg_name.c_str()];
		} else if (arg_name == "comment") {
			comment = kwargs[arg_name.c_str()];
		} else if (arg_name == "thousands") {
			thousands_separator = kwargs[arg_name.c_str()];
		} else if (arg_name == "dtype") {
			dtype = kwargs[arg_name.c_str()];
		} else if (arg_name == "na_values") {
			na_values = kwargs[arg_name.c_str()];
		} else if (arg_name == "skiprows") {
			skiprows = kwargs[arg_name.c_str()];
		} else if (arg_name == "quotechar") {
			quotechar = kwargs[arg_name.c_str()];
		} else if (arg_name == "escapechar") {
			escapechar = kwargs[arg_name.c_str()];
		} else if (arg_name == "encoding") {
			encoding = kwargs[arg_name.c_str()];
		} else if (arg_name == "parallel") {
			parallel = kwargs[arg_name.c_str()];
		} else if (arg_name == "date_format") {
			date_format = kwargs[arg_name.c_str()];
		} else if (arg_name == "timestamp_format") {
			timestamp_format = kwargs[arg_name.c_str()];
		} else if (arg_name == "sample_size") {
			sample_size = kwargs[arg_name.c_str()];
		} else if (arg_name == "auto_detect") {
			auto_detect = kwargs[arg_name.c_str()];
		} else if (arg_name == "all_varchar") {
			all_varchar = kwargs[arg_name.c_str()];
		} else if (arg_name == "normalize_names") {
			normalize_names = kwargs[arg_name.c_str()];
		} else if (arg_name == "null_padding") {
			null_padding = kwargs[arg_name.c_str()];
		} else if (arg_name == "names") {
			names_p = kwargs[arg_name.c_str()];
		} else if (arg_name == "lineterminator") {
			lineterminator = kwargs[arg_name.c_str()];
		} else if (arg_name == "columns") {
			columns = kwargs[arg_name.c_str()];
		} else if (arg_name == "auto_type_candidates") {
			auto_type_candidates = kwargs[arg_name.c_str()];
		} else if (arg_name == "max_line_size") {
			max_line_size = kwargs[arg_name.c_str()];
		} else if (arg_name == "ignore_errors") {
			ignore_errors = kwargs[arg_name.c_str()];
		} else if (arg_name == "store_rejects") {
			store_rejects = kwargs[arg_name.c_str()];
		} else if (arg_name == "rejects_table") {
			rejects_table = kwargs[arg_name.c_str()];
		} else if (arg_name == "rejects_scan") {
			rejects_scan = kwargs[arg_name.c_str()];
		} else if (arg_name == "rejects_limit") {
			rejects_limit = kwargs[arg_name.c_str()];
		} else if (arg_name == "force_not_null") {
			force_not_null = kwargs[arg_name.c_str()];
		} else if (arg_name == "buffer_size") {
			buffer_size = kwargs[arg_name.c_str()];
		} else if (arg_name == "decimal") {
			decimal = kwargs[arg_name.c_str()];
		} else if (arg_name == "allow_quoted_nulls") {
			allow_quoted_nulls = kwargs[arg_name.c_str()];
		} else if (arg_name == "filename") {
			filename = kwargs[arg_name.c_str()];
		} else if (arg_name == "hive_partitioning") {
			hive_partitioning = kwargs[arg_name.c_str()];
		} else if (arg_name == "union_by_name") {
			union_by_name = kwargs[arg_name.c_str()];
		} else if (arg_name == "hive_types") {
			hive_types = kwargs[arg_name.c_str()];
		} else if (arg_name == "hive_types_autocast") {
			hive_types_autocast = kwargs[arg_name.c_str()];
		} else if (arg_name == "strict_mode") {
			strict_mode = kwargs[arg_name.c_str()];
		} else {
			AcceptableCSVOptions(arg_name);
		}
	}

	auto &connection = con.GetConnection();
	auto &context = *connection.context;
	CSVReaderOptions options;
	auto path_like = GetPathLike(name_p);
	auto &name = path_like.files;
	auto file_like_object_wrapper = std::move(path_like.dependency);
	named_parameter_map_t bind_parameters;

	ParseMultiFileOptions(context, bind_parameters, filename, hive_partitioning, union_by_name, hive_types,
	                      hive_types_autocast);

	// First check if the header is explicitly set
	// when false this affects the returned types, so it needs to be known at initialization of the relation
	ConvertBooleanValue(header, "header", bind_parameters);
	ConvertBooleanValue(strict_mode, "strict_mode", bind_parameters);

	if (!nb::none().is(compression)) {
		if (!nb::isinstance<nb::str>(compression)) {
			throw InvalidInputException("read_csv only accepts 'compression' as a string");
		}
		bind_parameters["compression"] = Value(nb::cast<std::string>(nb::str(compression)));
	}

	if (!nb::none().is(dtype)) {
		if (duckdb::PyUtil::IsDictLike(dtype)) {
			child_list_t<Value> struct_fields;
			nb::dict dtype_dict = nb::cast<nb::dict>(dtype);
			for (auto kv : dtype_dict) { // nanobind dict iteration yields std::pair<handle,handle> by value
				auto key = nb::cast<std::string>(nb::str(kv.first));
				auto value_obj = nb::borrow<nb::object>(kv.second);
				if (nb::isinstance<nb::str>(value_obj)) {
					// A type string -- pass through for DuckDB to parse.
					struct_fields.emplace_back(key, Value(nb::cast<std::string>(value_obj)));
				} else {
					// A DuckDBPyType instance, or a Python type object (int/str/...). Build the DuckDBPyType via its
					// registered constructor, then borrow a const ref (no ownership extraction) to read it.
					if (!nb::isinstance<DuckDBPyType>(value_obj)) {
						value_obj = nb::type<DuckDBPyType>()(value_obj);
					}
					auto &sql_type = nb::cast<const DuckDBPyType &>(value_obj);
					struct_fields.emplace_back(key, Value(sql_type.ToString()));
				}
			}
			auto dtype_struct = Value::STRUCT(std::move(struct_fields));
			bind_parameters["dtypes"] = std::move(dtype_struct);
		} else if (duckdb::PyUtil::IsListLike(dtype)) {
			vector<Value> list_values;
			nb::list dtype_list = nb::cast<nb::list>(dtype);
			for (auto child : dtype_list) {
				auto child_obj = nb::borrow<nb::object>(child);
				std::unique_ptr<DuckDBPyType> sql_type;
				if (!nb::isinstance<nb::str>(child_obj) && DuckDBPyType::TryConvert(child_obj, sql_type)) {
					list_values.push_back(sql_type->ToString());
				} else {
					list_values.push_back(Value(nb::cast<std::string>(nb::str(child_obj))));
				}
			}
			bind_parameters["dtypes"] = Value::LIST(LogicalType::VARCHAR, std::move(list_values));
		} else {
			throw InvalidInputException("read_csv only accepts 'dtype' as a dictionary or a list of strings");
		}
	}

	bool has_sep = !nb::none().is(sep);
	bool has_delimiter = !nb::none().is(delimiter);
	if (has_sep && has_delimiter) {
		throw InvalidInputException("read_csv takes either 'delimiter' or 'sep', not both");
	}
	if (has_sep) {
		bind_parameters["delim"] = Value(duckdb::PyUtil::CastToString(sep));
	} else if (has_delimiter) {
		bind_parameters["delim"] = Value(duckdb::PyUtil::CastToString(delimiter));
	}

	if (!nb::none().is(files_to_sniff)) {
		if (!nb::isinstance<nb::int_>(files_to_sniff)) {
			throw InvalidInputException("read_csv only accepts 'files_to_sniff' as an integer");
		}
		bind_parameters["files_to_sniff"] = Value::INTEGER((int32_t)nb::int_(files_to_sniff));
	}

	if (!nb::none().is(names_p)) {
		if (!duckdb::PyUtil::IsListLike(names_p)) {
			throw InvalidInputException("read_csv only accepts 'names' as a list of strings");
		}
		vector<Value> names;
		nb::list names_list = nb::cast<nb::list>(names_p);
		for (auto elem : names_list) {
			if (!nb::isinstance<nb::str>(elem)) {
				throw InvalidInputException("read_csv 'names' list has to consist of only strings");
			}
			names.push_back(Value(nb::cast<std::string>(nb::str(elem))));
		}
		bind_parameters["names"] = Value::LIST(LogicalType::VARCHAR, std::move(names));
	}

	if (!nb::none().is(na_values)) {
		vector<Value> null_values;
		if (!nb::isinstance<nb::str>(na_values) && !duckdb::PyUtil::IsListLike(na_values)) {
			throw InvalidInputException("read_csv only accepts 'na_values' as a string or a list of strings");
		} else if (nb::isinstance<nb::str>(na_values)) {
			null_values.push_back(Value(nb::cast<std::string>(na_values)));
		} else {
			nb::list null_list = nb::cast<nb::list>(na_values);
			for (auto elem : null_list) {
				if (!nb::isinstance<nb::str>(elem)) {
					throw InvalidInputException("read_csv 'na_values' list has to consist of only strings");
				}
				null_values.push_back(Value(nb::cast<std::string>(nb::str(elem))));
			}
		}
		bind_parameters["nullstr"] = Value::LIST(LogicalType::VARCHAR, std::move(null_values));
	}

	if (!nb::none().is(skiprows)) {
		if (!nb::isinstance<nb::int_>(skiprows)) {
			throw InvalidInputException("read_csv only accepts 'skiprows' as an integer");
		}
		bind_parameters["skip"] = Value::INTEGER((int32_t)nb::int_(skiprows));
	}

	if (!nb::none().is(parallel)) {
		if (!nb::isinstance<nb::bool_>(parallel)) {
			throw InvalidInputException("read_csv only accepts 'parallel' as a boolean");
		}
		bind_parameters["parallel"] = Value::BOOLEAN((bool)nb::bool_(parallel));
	}

	if (!nb::none().is(quotechar)) {
		if (!nb::isinstance<nb::str>(quotechar)) {
			throw InvalidInputException("read_csv only accepts 'quotechar' as a string");
		}
		bind_parameters["quote"] = Value(nb::cast<std::string>(quotechar));
	}

	if (!nb::none().is(comment)) {
		if (!nb::isinstance<nb::str>(comment)) {
			throw InvalidInputException("read_csv only accepts 'comment' as a string");
		}
		bind_parameters["comment"] = Value(nb::cast<std::string>(comment));
	}

	if (!nb::none().is(thousands_separator)) {
		if (!nb::isinstance<nb::str>(thousands_separator)) {
			throw InvalidInputException("read_csv only accepts 'thousands' as a string");
		}
		bind_parameters["thousands"] = Value(nb::cast<std::string>(thousands_separator));
	}

	if (!nb::none().is(escapechar)) {
		if (!nb::isinstance<nb::str>(escapechar)) {
			throw InvalidInputException("read_csv only accepts 'escapechar' as a string");
		}
		bind_parameters["escape"] = Value(nb::cast<std::string>(escapechar));
	}

	if (!nb::none().is(encoding)) {
		if (!nb::isinstance<nb::str>(encoding)) {
			throw InvalidInputException("read_csv only accepts 'encoding' as a string");
		}
		string encoding_str = StringUtil::Lower(nb::cast<std::string>(encoding));
		if (encoding_str != "utf8" && encoding_str != "utf-8") {
			throw BinderException("Copy is only supported for UTF-8 encoded files, ENCODING 'UTF-8'");
		}
	}

	if (!nb::none().is(date_format)) {
		if (!nb::isinstance<nb::str>(date_format)) {
			throw InvalidInputException("read_csv only accepts 'date_format' as a string");
		}
		bind_parameters["dateformat"] = Value(nb::cast<std::string>(date_format));
	}

	if (!nb::none().is(auto_detect)) {
		bool auto_detect_as_int = nb::isinstance<nb::int_>(auto_detect);
		bool auto_detect_as_bool = nb::isinstance<nb::bool_>(auto_detect);
		bool auto_detect_value;
		if (auto_detect_as_bool) {
			auto_detect_value = (bool)nb::bool_(auto_detect);
		} else if (auto_detect_as_int) {
			if ((int)nb::int_(auto_detect) != 0) {
				throw InvalidInputException("read_csv only accepts 0 if 'auto_detect' is given as an integer");
			}
			auto_detect_value = true;
		} else {
			throw InvalidInputException("read_csv only accepts 'auto_detect' as an integer, or a boolean");
		}
		bind_parameters["auto_detect"] = Value::BOOLEAN(auto_detect_value);
	}

	if (!nb::none().is(timestamp_format)) {
		if (!nb::isinstance<nb::str>(timestamp_format)) {
			throw InvalidInputException("read_csv only accepts 'timestamp_format' as a string");
		}
		bind_parameters["timestampformat"] = Value(nb::cast<std::string>(timestamp_format));
	}

	if (!nb::none().is(sample_size)) {
		if (!nb::isinstance<nb::int_>(sample_size)) {
			throw InvalidInputException("read_csv only accepts 'sample_size' as an integer");
		}
		bind_parameters["sample_size"] = Value::INTEGER((int32_t)nb::int_(sample_size));
	}

	if (!nb::none().is(all_varchar)) {
		if (!nb::isinstance<nb::bool_>(all_varchar)) {
			throw InvalidInputException("read_csv only accepts 'all_varchar' as a boolean");
		}
		bind_parameters["all_varchar"] = Value::BOOLEAN((bool)nb::bool_(all_varchar));
	}

	if (!nb::none().is(normalize_names)) {
		if (!nb::isinstance<nb::bool_>(normalize_names)) {
			throw InvalidInputException("read_csv only accepts 'normalize_names' as a boolean");
		}
		bind_parameters["normalize_names"] = Value::BOOLEAN((bool)nb::bool_(normalize_names));
	}

	if (!nb::none().is(null_padding)) {
		if (!nb::isinstance<nb::bool_>(null_padding)) {
			throw InvalidInputException("read_csv only accepts 'null_padding' as a boolean");
		}
		bind_parameters["null_padding"] = Value::BOOLEAN((bool)nb::bool_(null_padding));
	}

	if (!nb::none().is(lineterminator)) {
		PythonCSVLineTerminator::Type new_line_type;
		if (!nb::try_cast<PythonCSVLineTerminator::Type>(lineterminator, new_line_type)) {
			string actual_type = nb::cast<std::string>(nb::str((lineterminator).type()));
			throw BinderException("read_csv only accepts 'lineterminator' as a string or CSVLineTerminator, not '%s'",
			                      actual_type);
		}
		bind_parameters["new_line"] = Value(PythonCSVLineTerminator::ToString(new_line_type));
	}

	if (!nb::none().is(max_line_size)) {
		if (!nb::isinstance<nb::str>(max_line_size) && !nb::isinstance<nb::int_>(max_line_size)) {
			string actual_type = nb::cast<std::string>(nb::str((max_line_size).type()));
			throw BinderException("read_csv only accepts 'max_line_size' as a string or an integer, not '%s'",
			                      actual_type);
		}
		auto val = TransformPythonValue(context, max_line_size, LogicalTypeId::VARCHAR);
		bind_parameters["max_line_size"] = val;
	}

	if (!nb::none().is(auto_type_candidates)) {
		if (!nb::isinstance<nb::list>(auto_type_candidates)) {
			string actual_type = nb::cast<std::string>(nb::str((auto_type_candidates).type()));
			throw BinderException("read_csv only accepts 'auto_type_candidates' as a list[str], not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, auto_type_candidates, LogicalType::LIST(LogicalTypeId::VARCHAR));
		bind_parameters["auto_type_candidates"] = val;
	}

	if (!nb::none().is(ignore_errors)) {
		if (!nb::isinstance<nb::bool_>(ignore_errors)) {
			string actual_type = nb::cast<std::string>(nb::str((ignore_errors).type()));
			throw BinderException("read_csv only accepts 'ignore_errors' as a bool, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, ignore_errors, LogicalTypeId::BOOLEAN);
		bind_parameters["ignore_errors"] = val;
	}

	if (!nb::none().is(store_rejects)) {
		if (!nb::isinstance<nb::bool_>(store_rejects)) {
			string actual_type = nb::cast<std::string>(nb::str((store_rejects).type()));
			throw BinderException("read_csv only accepts 'store_rejects' as a bool, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, store_rejects, LogicalTypeId::BOOLEAN);
		bind_parameters["store_rejects"] = val;
	}

	if (!nb::none().is(rejects_table)) {
		if (!nb::isinstance<nb::str>(rejects_table)) {
			string actual_type = nb::cast<std::string>(nb::str((rejects_table).type()));
			throw BinderException("read_csv only accepts 'rejects_table' as a string, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, rejects_table, LogicalTypeId::VARCHAR);
		bind_parameters["rejects_table"] = val;
	}

	if (!nb::none().is(rejects_scan)) {
		if (!nb::isinstance<nb::str>(rejects_scan)) {
			string actual_type = nb::cast<std::string>(nb::str((rejects_scan).type()));
			throw BinderException("read_csv only accepts 'rejects_scan' as a string, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, rejects_scan, LogicalTypeId::VARCHAR);
		bind_parameters["rejects_scan"] = val;
	}

	if (!nb::none().is(rejects_limit)) {
		if (!nb::isinstance<nb::int_>(rejects_limit)) {
			string actual_type = nb::cast<std::string>(nb::str((rejects_limit).type()));
			throw BinderException("read_csv only accepts 'rejects_limit' as an int, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, rejects_limit, LogicalTypeId::BIGINT);
		bind_parameters["rejects_limit"] = val;
	}

	if (!nb::none().is(force_not_null)) {
		if (!nb::isinstance<nb::list>(force_not_null)) {
			string actual_type = nb::cast<std::string>(nb::str((force_not_null).type()));
			throw BinderException("read_csv only accepts 'force_not_null' as a list[str], not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, force_not_null, LogicalType::LIST(LogicalTypeId::VARCHAR));
		bind_parameters["force_not_null"] = val;
	}

	if (!nb::none().is(buffer_size)) {
		if (!nb::isinstance<nb::int_>(buffer_size)) {
			string actual_type = nb::cast<std::string>(nb::str((buffer_size).type()));
			throw BinderException("read_csv only accepts 'buffer_size' as a list[str], not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, buffer_size, LogicalTypeId::UBIGINT);
		bind_parameters["buffer_size"] = val;
	}

	if (!nb::none().is(decimal)) {
		if (!nb::isinstance<nb::str>(decimal)) {
			string actual_type = nb::cast<std::string>(nb::str((decimal).type()));
			throw BinderException("read_csv only accepts 'decimal' as a string, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, decimal, LogicalTypeId::VARCHAR);
		bind_parameters["decimal_separator"] = val;
	}

	if (!nb::none().is(allow_quoted_nulls)) {
		if (!nb::isinstance<nb::bool_>(allow_quoted_nulls)) {
			string actual_type = nb::cast<std::string>(nb::str((allow_quoted_nulls).type()));
			throw BinderException("read_csv only accepts 'allow_quoted_nulls' as a bool, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(context, allow_quoted_nulls, LogicalTypeId::BOOLEAN);
		bind_parameters["allow_quoted_nulls"] = val;
	}

	if (!nb::none().is(columns)) {
		if (!duckdb::PyUtil::IsDictLike(columns)) {
			throw BinderException("read_csv only accepts 'columns' as a dict[str, str]");
		}
		nb::dict columns_dict = nb::cast<nb::dict>(columns);
		child_list_t<Value> struct_fields;

		for (auto kv : columns_dict) { // nanobind dict iteration yields std::pair<handle,handle> by value
			auto column_name = kv.first;
			auto type = kv.second;
			if (!nb::isinstance<nb::str>(column_name)) {
				string actual_type = nb::cast<std::string>(nb::str((column_name).type()));
				throw BinderException("The provided column name must be a str, not of type '%s'", actual_type);
			}
			if (!nb::isinstance<nb::str>(type)) {
				string actual_type = nb::cast<std::string>(nb::str((column_name).type()));
				throw BinderException("The provided column type must be a str, not of type '%s'", actual_type);
			}
			struct_fields.emplace_back(nb::cast<std::string>(nb::str(column_name)), Value(nb::cast<std::string>(type)));
		}
		auto dtype_struct = Value::STRUCT(std::move(struct_fields));
		bind_parameters["columns"] = std::move(dtype_struct);
	}

	// Create the ReadCSV Relation using the 'options'

	D_ASSERT(duckdb::PyUtil::GilCheck());
	nb::gil_scoped_release gil;
	auto read_csv_p = connection.ReadCSV(name, std::move(bind_parameters));
	auto &read_csv = read_csv_p->Cast<ReadCSVRelation>();
	if (file_like_object_wrapper) {
		read_csv.AddExternalDependency(std::move(file_like_object_wrapper));
	}

	return CreateRelation(read_csv_p->Alias(read_csv.alias));
}

void DuckDBPyConnection::ExecuteImmediately(vector<unique_ptr<SQLStatement>> statements) {
	auto &connection = con.GetConnection();
	D_ASSERT(duckdb::PyUtil::GilCheck());
	nb::gil_scoped_release release;
	if (statements.empty()) {
		return;
	}
	for (auto &stmt : statements) {
		if (!stmt->named_param_map.empty()) {
			throw NotImplementedException(
			    "Prepared parameters are only supported for the last statement, please split your query up into "
			    "separate 'execute' calls if you want to use prepared parameters");
		}
		auto pending_query = connection.PendingQuery(std::move(stmt), false);
		if (pending_query->HasError()) {
			pending_query->ThrowError();
		}
		auto res = CompletePendingQuery(*pending_query);

		if (res->HasError()) {
			res->ThrowError();
		}
	}
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::RunQuery(const nb::object &query, string alias,
                                                               nb::object params) {
	auto &connection = con.GetConnection();
	if (alias.empty()) {
		alias = "unnamed_relation_" + StringUtil::GenerateRandomName(16);
	}

	auto normalized_query = ExtractCompiledSqlAndParams(query, params);
	auto &compiled_sql = normalized_query.first;
	auto &merged_params = normalized_query.second;
	vector<unique_ptr<SQLStatement>> statements;
	if (!compiled_sql.empty()) {
		statements = GetStatements(nb::str(compiled_sql.c_str(), compiled_sql.size()));
		params = merged_params;
	} else {
		statements = GetStatements(query);
	}

	if (statements.empty()) {
		// TODO: should we throw?
		return nullptr;
	}

	auto last_statement = std::move(statements.back());
	statements.pop_back();
	// First immediately execute any preceding statements (if any)
	ExecuteImmediately(std::move(statements));

	// Attempt to create a Relation for lazy execution if possible
	shared_ptr<Relation> relation;
	bool has_params = !params.is_none() && nb::len(params) > 0;
	if (!has_params) {
		// No params (or empty params) — use lazy QueryRelation path
		{
			D_ASSERT(duckdb::PyUtil::GilCheck());
			nb::gil_scoped_release gil;
			auto statement_type = last_statement->type;
			switch (statement_type) {
			case StatementType::SELECT_STATEMENT: {
				auto select_statement = unique_ptr_cast<SQLStatement, SelectStatement>(std::move(last_statement));
				relation = connection.RelationFromQuery(std::move(select_statement), alias);
				break;
			}
			default:
				break;
			}
		}
	}

	if (!relation) {
		// Could not create a relation, resort to direct execution
		unique_ptr<QueryResult> res;

		res = PrepareAndExecuteInternal(std::move(last_statement), std::move(params));

		if (!res) {
			return nullptr;
		}
		if (res->properties.return_type != StatementReturnType::QUERY_RESULT) {
			return nullptr;
		}
		if (res->type == QueryResultType::STREAM_RESULT) {
			auto &stream_result = res->Cast<StreamQueryResult>();
			res = stream_result.Materialize();
		}
		auto &materialized_result = res->Cast<MaterializedQueryResult>();
		vector<Identifier> col_names(res->names.size());
		std::transform(res->names.begin(), res->names.end(), col_names.begin(),
		               [](string &name) { return Identifier(name); });
		relation = make_shared_ptr<MaterializedRelation>(connection.context, materialized_result.TakeCollection(),
		                                                 col_names, Identifier(alias));
	}
	return CreateRelation(std::move(relation));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::Table(const string &tname) {
	auto &connection = con.GetConnection();
	auto qualified_name = QualifiedName::Parse(tname);
	if (qualified_name.Schema().empty()) {
		qualified_name = QualifiedName(qualified_name.Catalog(), DEFAULT_SCHEMA, qualified_name.Name());
	}
	try {
		return CreateRelation(
		    connection.Table(qualified_name.Catalog(), qualified_name.Schema(), qualified_name.Name()));
	} catch (const CatalogException &) {
		// CatalogException will be of the type '... is not a table'
		// Not a table in the database, make a query relation that can perform replacement scans
		auto sql_query = StringUtil::Format("from %s", SQLIdentifier::ToString(tname));
		return RunQuery(nb::str(sql_query.c_str(), sql_query.size()), tname);
	}
}

static vector<unique_ptr<ParsedExpression>> ValueListFromExpressions(const nb::args &expressions) {
	vector<unique_ptr<ParsedExpression>> result;
	auto arg_count = expressions.size();
	if (arg_count == 0) {
		throw InvalidInputException("Please provide a non-empty tuple");
	}

	for (idx_t i = 0; i < arg_count; i++) {
		nb::handle arg = expressions[i];
		auto py_expr = DuckDBPyExpression::ToExpression(arg);
		result.push_back(py_expr->GetExpression().Copy());
	}
	return result;
}

static vector<vector<unique_ptr<ParsedExpression>>> ValueListsFromTuples(const nb::args &tuples) {
	auto arg_count = tuples.size();
	if (arg_count == 0) {
		throw InvalidInputException("Please provide a non-empty tuple");
	}

	idx_t expected_length = 0;
	vector<vector<unique_ptr<ParsedExpression>>> result;
	for (idx_t i = 0; i < arg_count; i++) {
		nb::handle arg = tuples[i];
		if (!nb::isinstance<nb::tuple>(arg)) {
			string actual_type = nb::cast<std::string>(nb::str((arg).type()));
			throw InvalidInputException("Expected objects of type tuple, not %s", actual_type);
		}
		auto expressions = nb::cast<nb::args>(arg);
		auto value_list = ValueListFromExpressions(expressions);
		if (i && value_list.size() != expected_length) {
			throw InvalidInputException("Mismatch between length of tuples in input, expected %d but found %d",
			                            expected_length, value_list.size());
		}
		expected_length = value_list.size();
		result.push_back(std::move(value_list));
	}
	return result;
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::Values(const nb::args &args) {
	auto &connection = con.GetConnection();
	auto &context = *connection.context;

	auto arg_count = args.size();
	if (arg_count == 0) {
		throw InvalidInputException("Could not create a ValueRelation without any inputs");
	}

	D_ASSERT(duckdb::PyUtil::GilCheck());
	nb::handle first_arg = args[0];
	if (arg_count == 1 && nb::isinstance<nb::list>(first_arg)) {
		vector<vector<Value>> values {DuckDBPyConnection::TransformPythonParamList(context, first_arg)};
		return CreateRelation(connection.Values(values));
	} else {
		vector<vector<unique_ptr<ParsedExpression>>> expressions;
		if (nb::isinstance<nb::tuple>(first_arg)) {
			expressions = ValueListsFromTuples(args);
		} else {
			auto values = ValueListFromExpressions(args);
			expressions.push_back(std::move(values));
		}
		return CreateRelation(connection.Values(std::move(expressions)));
	}
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::View(const string &vname) {
	auto &connection = con.GetConnection();
	return CreateRelation(connection.View(Identifier(vname)));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::TableFunction(const string &fname, nb::object params) {
	auto &connection = con.GetConnection();
	auto &context = *connection.context;
	if (params.is_none()) {
		params = nb::list();
	}
	if (!duckdb::PyUtil::IsListLike(params)) {
		throw InvalidInputException("'params' has to be a list of parameters");
	}

	return CreateRelation(
	    connection.TableFunction(fname, DuckDBPyConnection::TransformPythonParamList(context, params)));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::FromDF(const PandasDataFrame &value) {
	auto &connection = con.GetConnection();
	string name = "df_" + StringUtil::GenerateRandomName();
	if (PandasDataFrame::IsPyArrowBacked(value)) {
		auto table = PandasDataFrame::ToArrowTable(value);
		return DuckDBPyConnection::FromArrow(table);
	}
	auto tableref = PythonReplacementScan::ReplacementObject(value, name, *connection.context);
	D_ASSERT(tableref);
	auto rel = make_shared_ptr<ViewRelation>(connection.context, std::move(tableref), name);
	return CreateRelation(std::move(rel));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::FromParquet(const nb::object &path_or_buffer,
                                                                  bool binary_as_string, bool file_row_number,
                                                                  bool filename, bool hive_partitioning,
                                                                  bool union_by_name, const nb::object &compression) {
	auto &connection = con.GetConnection();
	auto path_like = GetPathLike(path_or_buffer);
	auto file_like_object_wrapper = std::move(path_like.dependency);

	string name = "parquet_" + StringUtil::GenerateRandomName();
	vector<Value> file_values;
	for (auto &file : path_like.files) {
		file_values.emplace_back(std::move(file));
	}
	vector<Value> params;
	params.emplace_back(Value::LIST(LogicalType::VARCHAR, std::move(file_values)));
	named_parameter_map_t named_parameters({{"binary_as_string", Value::BOOLEAN(binary_as_string)},
	                                        {"file_row_number", Value::BOOLEAN(file_row_number)},
	                                        {"filename", Value::BOOLEAN(filename)},
	                                        {"hive_partitioning", Value::BOOLEAN(hive_partitioning)},
	                                        {"union_by_name", Value::BOOLEAN(union_by_name)}});

	if (!nb::none().is(compression)) {
		if (!nb::isinstance<nb::str>(compression)) {
			throw InvalidInputException("from_parquet only accepts 'compression' as a string");
		}
		named_parameters["compression"] = Value(nb::cast<std::string>(compression));
	}
	D_ASSERT(duckdb::PyUtil::GilCheck());
	nb::gil_scoped_release gil;
	auto parquet_relation = connection.TableFunction("parquet_scan", params, named_parameters);
	if (file_like_object_wrapper) {
		parquet_relation->AddExternalDependency(std::move(file_like_object_wrapper));
	}
	return CreateRelation(parquet_relation->Alias(name));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyConnection::FromArrow(nb::object &arrow_object) {
	auto &connection = con.GetConnection();
	string name = "arrow_object_" + StringUtil::GenerateRandomName();
	if (!IsAcceptedArrowObject(arrow_object)) {
		// nb::object wrap: nb::str() of a bare .attr() accessor is an ambiguous overload on MSVC.
		auto py_object_type = nb::cast<std::string>(nb::str(nb::object((arrow_object).type().attr("__name__"))));
		throw InvalidInputException("Python Object Type %s is not an accepted Arrow Object.", py_object_type);
	}
	auto tableref = PythonReplacementScan::ReplacementObject(arrow_object, name, *connection.context, true);
	D_ASSERT(tableref);
	auto rel = make_shared_ptr<ViewRelation>(connection.context, std::move(tableref), name);
	return CreateRelation(std::move(rel));
}

unordered_set<string> DuckDBPyConnection::GetTableNames(const string &query, bool qualified) {
	auto &connection = con.GetConnection();
	return connection.GetTableNames(query, qualified);
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::UnregisterPythonObject(const string &name) {
	auto &connection = con.GetConnection();
	if (!registered_objects.count(name)) {
		return shared_from_this();
	}
	D_ASSERT(duckdb::PyUtil::GilCheck());
	nb::gil_scoped_release release;
	// FIXME: DROP TEMPORARY VIEW? doesn't exist?
	const auto quoted_name = SQLQuotedIdentifier::ToString(name);
	connection.Query("DROP VIEW " + quoted_name + "");
	registered_objects.erase(name);
	return shared_from_this();
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Begin() {
	ExecuteFromString("BEGIN TRANSACTION");
	return shared_from_this();
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Commit() {
	auto &connection = con.GetConnection();
	if (connection.context->transaction.IsAutoCommit()) {
		return shared_from_this();
	}
	ExecuteFromString("COMMIT");
	return shared_from_this();
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Rollback() {
	ExecuteFromString("ROLLBACK");
	return shared_from_this();
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Checkpoint() {
	ExecuteFromString("CHECKPOINT");
	return shared_from_this();
}

Optional<nb::list> DuckDBPyConnection::GetDescription() {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		return nb::none();
	}
	auto &result = con.GetResult();
	return result.Description();
}

int DuckDBPyConnection::GetRowcount() {
	return -1;
}

void DuckDBPyConnection::Close() {
	ConnectionLockGuard conn_lock(*this);
	con.SetResult(nullptr);
	D_ASSERT(duckdb::PyUtil::GilCheck());
	// Release the GIL only for the native Connection / DuckDB teardown, which
	// is pure C++ work and can take noticeable time. Hold the GIL back for
	// `registered_functions.clear()` because the
	// `case_insensitive_map_t<unique_ptr<ExternalDependency>>` it destroys
	// transitively owns Python references (Python UDF
	// callables, registered Python objects, …). Decrementing those
	// references with the GIL released is undefined behaviour — see
	// duckdb-python#456.
	{
		nb::gil_scoped_release release;
		con.SetConnection(nullptr);
		con.SetDatabase(nullptr);
	}
	// https://peps.python.org/pep-0249/#Connection.close
	cursors.ClearCursors();
	registered_functions.clear();
}

void DuckDBPyConnection::Interrupt() {
	auto &connection = con.GetConnection();
	connection.Interrupt();
}

double DuckDBPyConnection::QueryProgress() {
	auto &connection = con.GetConnection();
	return connection.GetQueryProgress();
}

void DuckDBPyConnection::InstallExtension(const string &extension, bool force_install, const nb::object &repository,
                                          const nb::object &repository_url, const nb::object &version) {
	auto &connection = con.GetConnection();

	auto install_statement = make_uniq<LoadStatement>();
	install_statement->info = make_uniq<LoadInfo>();
	auto &info = *install_statement->info;

	info.filename = extension;

	const bool has_repository = !nb::none().is(repository);
	const bool has_repository_url = !nb::none().is(repository_url);
	if (has_repository && has_repository_url) {
		throw InvalidInputException(
		    "Both 'repository' and 'repository_url' are set which is not allowed, please pick one or the other");
	}
	string repository_string;
	if (has_repository) {
		repository_string = nb::cast<std::string>(nb::str(repository));
	} else if (has_repository_url) {
		repository_string = nb::cast<std::string>(nb::str(repository_url));
	}

	if ((has_repository || has_repository_url) && repository_string.empty()) {
		throw InvalidInputException("The provided 'repository' or 'repository_url' can not be empty!");
	}

	string version_string;
	if (!nb::none().is(version)) {
		version_string = nb::cast<std::string>(nb::str(version));
		if (version_string.empty()) {
			throw InvalidInputException("The provided 'version' can not be empty!");
		}
	}

	info.repository = repository_string;
	info.repo_is_alias = repository_string.empty() ? false : has_repository;
	info.version = version_string;
	info.load_type = force_install ? LoadType::FORCE_INSTALL : LoadType::INSTALL;
	auto res = connection.Query(std::move(install_statement));
	if (res->HasError()) {
		res->ThrowError();
	}
}

void DuckDBPyConnection::LoadExtension(const string &extension) {
	auto &connection = con.GetConnection();
	const ExtensionLoadOptions extension_opts = {extension};
	ExtensionHelper::LoadExternalExtension(*connection.context, extension_opts);
}

std::shared_ptr<DuckDBPyConnection> DefaultConnectionHolder::Get() {
	lock_guard<mutex> guard(l);
	if (!connection || connection->con.ConnectionIsClosed()) {
		nb::dict config_dict;
		connection = DuckDBPyConnection::Connect(nb::str(":memory:"), false, config_dict);
	}
	return connection;
}

void DefaultConnectionHolder::Set(std::shared_ptr<DuckDBPyConnection> conn) {
	lock_guard<mutex> guard(l);
	connection = conn;
}

void DuckDBPyConnection::Cursors::AddCursor(std::shared_ptr<DuckDBPyConnection> conn) {
	lock_guard<mutex> l(lock);

	// Clean up previously created cursors
	vector<std::weak_ptr<DuckDBPyConnection>> compacted_cursors;
	bool needs_compaction = false;
	for (auto &cur_p : cursors) {
		auto cur = cur_p.lock();
		if (!cur) {
			needs_compaction = true;
			continue;
		}
		compacted_cursors.push_back(cur_p);
	}
	if (needs_compaction) {
		cursors = std::move(compacted_cursors);
	}

	cursors.push_back(conn);
}

void DuckDBPyConnection::Cursors::ClearCursors() {
	lock_guard<mutex> l(lock);

	for (auto &cur : cursors) {
		auto cursor = cur.lock();
		if (!cursor) {
			// The cursor has already been closed
			continue;
		}
		// This is *only* needed because we have a nb::gil_scoped_release in Close, so it *needs* the GIL in order to
		// release it don't ask me why it can't just realize there is no GIL and move on
		nb::gil_scoped_acquire gil;
		cursor->Close();
		// Ensure destructor runs with gil if triggered.
		cursor.reset();
	}

	cursors.clear();
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Cursor() {
	auto res = std::make_shared<DuckDBPyConnection>();
	res->con.SetDatabase(con);
	res->con.SetConnection(make_uniq<Connection>(res->con.GetDatabase()));
	cursors.AddCursor(res);
	return res;
}

// these should be functions on the result but well
//
// All of the connection-level fetch methods below take `py_connection_lock`
// before touching `con.GetResult()`, so that another thread cannot replace
// or destroy the connection's current result while we are mid-fetch — see
// duckdb-python#435.
Optional<nb::tuple> DuckDBPyConnection::FetchOne() {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchOne();
}

nb::list DuckDBPyConnection::FetchMany(idx_t size) {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchMany(size);
}

nb::list DuckDBPyConnection::FetchAll() {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchAll();
}

nb::dict DuckDBPyConnection::FetchNumpy() {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchNumpyInternal();
}

PandasDataFrame DuckDBPyConnection::FetchDF(bool date_as_object) {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchDF(date_as_object);
}

PandasDataFrame DuckDBPyConnection::FetchDFChunk(const idx_t vectors_per_chunk, bool date_as_object) {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchDFChunk(vectors_per_chunk, date_as_object);
}

duckdb::pyarrow::Table DuckDBPyConnection::FetchArrow(idx_t rows_per_batch) {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.ToArrowTable(rows_per_batch);
}

nb::dict DuckDBPyConnection::FetchPyTorch() {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchPyTorch();
}

nb::dict DuckDBPyConnection::FetchTF() {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchTF();
}

PolarsDataFrame DuckDBPyConnection::FetchPolars(idx_t rows_per_batch, bool lazy) {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.ToPolars(rows_per_batch, lazy);
}

duckdb::pyarrow::RecordBatchReader DuckDBPyConnection::FetchRecordBatchReader(const idx_t rows_per_batch) {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchRecordBatchReader(rows_per_batch);
}

case_insensitive_map_t<Value> TransformPyConfigDict(const nb::dict &py_config_dict) {
	case_insensitive_map_t<Value> config_dict;
	for (auto kv : py_config_dict) {
		// Config values may be int/bool/str; str-ify them rather than
		// requiring an actual Python str (nb::cast<std::string> would throw on a non-str like 0 or False).
		auto key = nb::cast<std::string>(nb::str(kv.first));
		auto val = nb::cast<std::string>(nb::str(kv.second));
		config_dict[key] = Value(val);
	}
	return config_dict;
}

static bool HasJupyterProgressBarDependencies() {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (!import_cache.ipywidgets()) {
		// ipywidgets not installed, needed to support the progress bar
		return false;
	}
	return true;
}

static void SetDefaultConfigArguments(ClientContext &context) {
	if (!DuckDBPyConnection::IsInteractive()) {
		// Don't need to set any special default arguments
		return;
	}

	auto &config = ClientConfig::GetConfig(context);
	config.enable_progress_bar = true;

	if (!DuckDBPyConnection::IsJupyter()) {
		return;
	}
	if (!HasJupyterProgressBarDependencies()) {
		// Disable progress bar altogether
		config.system_progress_bar_disable_reason =
		    "required package 'ipywidgets' is missing, which is needed to render progress bars in Jupyter";
		config.enable_progress_bar = false;
		return;
	}

	// Set the function used to create the display for the progress bar
	context.config.display_create_func = JupyterProgressBarDisplay::Create;
}

void InstantiateNewInstance(DuckDB &db) {
	auto &db_instance = *db.instance;
	PandasScanFunction scan_fun;
	MapFunction map_fun;

	TableFunctionSet map_set(map_fun.name);
	map_set.AddFunction(static_cast<TableFunction>(std::move(map_fun)));
	CreateTableFunctionInfo map_info(std::move(map_set));
	map_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

	TableFunctionSet scan_set(scan_fun.name);
	scan_set.AddFunction(static_cast<TableFunction>(std::move(scan_fun)));
	CreateTableFunctionInfo scan_info(std::move(scan_set));
	scan_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

	auto &system_catalog = Catalog::GetSystemCatalog(db_instance);
	auto transaction = CatalogTransaction::GetSystemTransaction(db_instance);

	system_catalog.CreateFunction(transaction, map_info);
	system_catalog.CreateFunction(transaction, scan_info);
}

static std::shared_ptr<DuckDBPyConnection> FetchOrCreateInstance(const string &database_path, DBConfig &config) {
	auto res = std::make_shared<DuckDBPyConnection>();
	bool cache_instance = database_path != ":memory:" && !database_path.empty();
	config.replacement_scans.emplace_back(PythonReplacementScan::Replace);
	{
		D_ASSERT(duckdb::PyUtil::GilCheck());
		nb::gil_scoped_release release;
		unique_lock<std::recursive_mutex> lock(res->py_connection_lock);
		auto database = GetModuleState().instance_cache.GetOrCreateInstance(database_path, config, cache_instance,
		                                                                    InstantiateNewInstance);
		res->con.SetDatabase(std::move(database));
		res->con.SetConnection(make_uniq<Connection>(res->con.GetDatabase()));
	}
	return res;
}

bool IsDefaultConnectionString(const string &database, bool read_only, case_insensitive_map_t<Value> &config) {
	bool is_default = StringUtil::CIEquals(database, ":default:");
	if (!is_default) {
		return false;
	}
	// Only allow fetching the default connection when no options are passed
	if (read_only == true || !config.empty()) {
		throw InvalidInputException("Default connection fetching is only allowed without additional options");
	}
	return true;
}

static string GetPathString(const nb::object &path) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	const bool is_path = duckdb::PyUtil::IsInstance(path, import_cache.pathlib.Path());
	if (is_path || nb::isinstance<nb::str>(path)) {
		return nb::cast<std::string>(nb::str(path));
	}
	string actual_type = nb::cast<std::string>(nb::str((path).type()));
	throw InvalidInputException("Please provide either a str or a pathlib.Path, not %s", actual_type);
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Connect(const nb::object &database_p, bool read_only,
                                                                const nb::dict &config_options) {
	auto config_dict = TransformPyConfigDict(config_options);
	auto database = GetPathString(database_p);
	if (IsDefaultConnectionString(database, read_only, config_dict)) {
		return DuckDBPyConnection::DefaultConnection();
	}

	DBConfig config(read_only);
	config.AddExtensionOption("pandas_analyze_sample",
	                          "The maximum number of rows to sample when analyzing a pandas object column.",
	                          LogicalType::UBIGINT, Value::UBIGINT(1000));
	config.AddExtensionOption("python_enable_replacements",
	                          "Whether variables visible to the current stack should be used for replacement scans.",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true));
	config.AddExtensionOption(
	    "python_scan_all_frames",
	    "If set, restores the old behavior of scanning all preceding frames to locate the referenced variable.",
	    LogicalType::BOOLEAN, Value::BOOLEAN(false));
	if (!DuckDBPyConnection::IsJupyter()) {
		config_dict["duckdb_api"] = Value("python/" + DuckDBPyConnection::FormattedPythonVersion());
	} else {
		config_dict["duckdb_api"] = Value("python/" + DuckDBPyConnection::FormattedPythonVersion() + " jupyter");
	}
	config.SetOptionsByName(config_dict);

	auto res = FetchOrCreateInstance(database, config);
	auto &client_context = *res->con.GetConnection().context;
	SetDefaultConfigArguments(client_context);
	return res;
}

vector<Value> DuckDBPyConnection::TransformPythonParamList(ClientContext &context, const nb::handle &params) {
	vector<Value> args;
	args.reserve(nb::len(params));

	for (auto param : params) {
		args.emplace_back(TransformPythonValue(context, param, LogicalType::UNKNOWN, false));
	}
	return args;
}

identifier_map_t<BoundParameterData> DuckDBPyConnection::TransformPythonParamDict(ClientContext &context,
                                                                                  const nb::dict &params) {
	identifier_map_t<BoundParameterData> args;

	for (auto pair : params) {
		auto &key = pair.first;
		auto &value = pair.second;
		args[Identifier(duckdb::PyUtil::CastToString(key))] =
		    BoundParameterData(TransformPythonValue(context, value, LogicalType::UNKNOWN, false));
	}
	return args;
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::DefaultConnection() {
	return GetModuleState().default_connection.Get();
}

void DuckDBPyConnection::SetDefaultConnection(std::shared_ptr<DuckDBPyConnection> connection) {
	return GetModuleState().default_connection.Set(std::move(connection));
}

PythonImportCache *DuckDBPyConnection::ImportCache() {
	auto &import_cache = GetModuleState().import_cache;
	if (!import_cache) {
		import_cache = std::make_shared<PythonImportCache>();
	}
	return import_cache.get();
}

ModifiedMemoryFileSystem &DuckDBPyConnection::GetObjectFileSystem() {
	if (!internal_object_filesystem) {
		D_ASSERT(!FileSystemIsRegistered("DUCKDB_INTERNAL_OBJECTSTORE"));
		auto &import_cache_py = *ImportCache();
		auto modified_memory_fs = import_cache_py.duckdb.filesystem.ModifiedMemoryFileSystem();
		if (modified_memory_fs.ptr() == nullptr) {
			throw InvalidInputException(
			    "This operation could not be completed because required module 'fsspec' is not installed");
		}
		internal_object_filesystem = std::make_shared<ModifiedMemoryFileSystem>(modified_memory_fs());
		auto &abstract_fs = reinterpret_cast<AbstractFileSystem &>(*internal_object_filesystem);
		RegisterFilesystem(abstract_fs);
	}
	return *internal_object_filesystem;
}

bool DuckDBPyConnection::IsInteractive() {
	return GetModuleState().environment != PythonEnvironmentType::NORMAL;
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Enter() {
	return shared_from_this();
}

void DuckDBPyConnection::Exit(DuckDBPyConnection &self, const nb::object &exc_type, const nb::object &exc,
                              const nb::object &traceback) {
	self.Close();
	if (exc_type.ptr() != Py_None) {
		// Propagate the exception if any occurred
		PyErr_SetObject(exc_type.ptr(), exc.ptr());
		throw nb::python_error();
	}
}

void DuckDBPyConnection::Cleanup() {
	GetModuleState().default_connection.Set(nullptr);
	GetModuleState().import_cache.reset();
}

bool DuckDBPyConnection::IsPandasDataframe(const nb::object &object) {
	if (!ModuleIsLoaded<PandasCacheItem>()) {
		return false;
	}
	auto &import_cache_py = *DuckDBPyConnection::ImportCache();
	return duckdb::PyUtil::IsInstance(object, import_cache_py.pandas.DataFrame());
}

bool IsValidNumpyDimensions(const nb::handle &object, int &dim) {
	// check the dimensions of numpy arrays
	// should only be called by IsAcceptedNumpyObject
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (!duckdb::PyUtil::IsInstance(object, import_cache.numpy.ndarray())) {
		return false;
	}
	nb::object shape = NumpyArray(nb::borrow<nb::object>(object)).GetArray().attr("shape");
	if (nb::len(shape) != 1) {
		return false;
	}
	int cur_dim = nb::cast<int>((shape.attr("__getitem__")(0)));
	dim = dim == -1 ? cur_dim : dim;
	return dim == cur_dim;
}
NumpyObjectType DuckDBPyConnection::IsAcceptedNumpyObject(const nb::object &object) {
	if (!ModuleIsLoaded<NumpyCacheItem>()) {
		return NumpyObjectType::INVALID;
	}
	auto import_cache_ = ImportCache();
	if (duckdb::PyUtil::IsInstance(object, import_cache_->numpy.ndarray())) {
		auto len = nb::len(nb::object(NumpyArray(object).GetArray().attr("shape")));
		switch (len) {
		case 1:
			return NumpyObjectType::NDARRAY1D;
		case 2:
			return NumpyObjectType::NDARRAY2D;
		default:
			return NumpyObjectType::INVALID;
		}
	} else if (duckdb::PyUtil::IsDictLike(object)) {
		int dim = -1;
		for (auto item : nb::cast<nb::dict>(object)) {
			if (!IsValidNumpyDimensions(item.second, dim)) {
				return NumpyObjectType::INVALID;
			}
		}
		return NumpyObjectType::DICT;
	} else if (duckdb::PyUtil::IsListLike(object)) {
		int dim = -1;
		for (auto item : nb::cast<nb::list>(object)) {
			if (!IsValidNumpyDimensions(item, dim)) {
				return NumpyObjectType::INVALID;
			}
		}
		return NumpyObjectType::LIST;
	}
	return NumpyObjectType::INVALID;
}

PyArrowObjectType DuckDBPyConnection::GetArrowType(const nb::handle &obj) {
	D_ASSERT(duckdb::PyUtil::GilCheck());

	if (nb::isinstance<nb::capsule>(obj)) {
		auto capsule = nb::borrow<nb::capsule>(obj);
		if (string(capsule.name()) != "arrow_array_stream") {
			throw InvalidInputException("Expected a 'arrow_array_stream' PyCapsule, got: %s", string(capsule.name()));
		}
		auto stream = reinterpret_cast<ArrowArrayStream *>(capsule.data("arrow_array_stream"));
		if (!stream->release) {
			throw InvalidInputException("The ArrowArrayStream was already released");
		}
		return PyArrowObjectType::PyCapsule;
	}

	if (ModuleIsLoaded<PyarrowCacheItem>()) {
		auto import_cache_ = ImportCache();
		// MessageReader requires nanoarrow, separate scan function
		if (duckdb::PyUtil::IsInstance(obj, import_cache_->pyarrow.ipc.MessageReader())) {
			return PyArrowObjectType::MessageReader;
		}

		if (ModuleIsLoaded<PyarrowDatasetCacheItem>()) {
			// Scanner/Dataset don't have __arrow_c_stream__, need dedicated handling
			if (duckdb::PyUtil::IsInstance(obj, import_cache_->pyarrow.dataset.Scanner())) {
				return PyArrowObjectType::Scanner;
			} else if (duckdb::PyUtil::IsInstance(obj, import_cache_->pyarrow.dataset.Dataset())) {
				return PyArrowObjectType::Dataset;
			}
		}
	}

	if (nb::hasattr(obj, "__arrow_c_stream__")) {
		return PyArrowObjectType::PyCapsuleInterface;
	}

	return PyArrowObjectType::Invalid;
}

bool DuckDBPyConnection::IsAcceptedArrowObject(const nb::object &object) {
	return DuckDBPyConnection::GetArrowType(object) != PyArrowObjectType::Invalid;
}

} // namespace duckdb

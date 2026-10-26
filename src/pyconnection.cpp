#include "duckdb_python/pyconnection/pyconnection.hpp"

#include "duckdb/catalog/catalog.hpp"
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
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/statement/transaction_statement.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/map.hpp"
#include "duckdb_python/pandas/pandas_scan.hpp"
#include "duckdb_python/arrow/python_arrow_scan.hpp"
#include "duckdb_python/pyconnection/seam.hpp"
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
#include "duckdb/main/query_result.hpp"
#include "duckdb_python/python_replacement_scan.hpp"
#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/parser/statement/load_statement.hpp"
#include "duckdb_python/nb/conversions/python_csv_line_terminator_enum.hpp"

namespace duckdb {

// All process-global module state lives in one struct, reached only through GetModuleState().
// This is the single seam to retarget for PEP 489 multi-phase init (per-module state via
// PyModule_GetState); call sites never touch the storage directly.
struct DuckDBPyModuleState {
	DefaultConnectionHolder default_connection;
	DBInstanceCache instance_cache;
	std::shared_ptr<PythonImportCache> import_cache;
	//! The single-use Arrow readers that have handed out their stream, by identity and held weakly
	unique_ptr<PyObjectHolder> consumed_arrow_readers;
	//! Held rather than imported on use, since an import fails once the package has left sys.modules
	unique_ptr<PyObjectHolder> arrow_sources_adapt;
	unique_ptr<PyObjectHolder> arrow_sources_nodes;
	PythonEnvironmentType environment = PythonEnvironmentType::NORMAL;
	std::string formatted_python_version;
};

static DuckDBPyModuleState &GetModuleState() {
	// Never destroyed: static destructors run after Py_Finalize, too late to release Python references
	static auto &state = *new DuckDBPyModuleState(); // NOLINT: allow global - sole module-state seam
	return state;
}

DuckDBPyConnection::~DuckDBPyConnection() {
	try {
		// Released with the GIL held, since the registry entries they let go own Python references
		ReleaseOwnedSources();
		auto released = con.Release();
		{
			nb::gil_scoped_release gil;
			released.Destroy();
		}
	} catch (...) { // NOLINT
	}
}

DbapiResult::DbapiResult(std::shared_ptr<DuckDBPyResult> result_p) : result(std::move(result_p)) {
	if (!result) {
		throw InternalException("DbapiResult created without a result");
	}
	types = result->GetTypes();
	names = result->GetNames();
	changed_rows = result->Rowcount();
}

nb::list DbapiResult::Description() const {
	return DuckDBPyResult::GetDescription(names, types);
}

Optional<nb::tuple> DbapiResult::FetchOne() {
	if (!result) {
		return nb::none();
	}
	return result->Fetchone();
}

nb::list DbapiResult::FetchMany(idx_t size) {
	if (!result) {
		return nb::list();
	}
	return result->Fetchmany(size);
}

nb::list DbapiResult::FetchAll() {
	if (!result) {
		return nb::list();
	}
	auto rows = result->Fetchall();
	result = nullptr;
	return rows;
}

nb::dict DbapiResult::FetchNumpy() {
	if (!result) {
		return nb::borrow<nb::dict>(nb::none());
	}
	auto arrays = result->FetchNumpyInternal();
	result = nullptr;
	return arrays;
}

PandasDataFrame DbapiResult::FetchDF(bool date_as_object) {
	if (!result) {
		return nb::none();
	}
	auto df = result->FetchDF(date_as_object);
	result = nullptr;
	return df;
}

PandasDataFrame DbapiResult::FetchDFChunk(idx_t vectors_per_chunk, bool date_as_object) {
	if (!result) {
		return nb::none();
	}
	return result->FetchDFChunk(vectors_per_chunk, date_as_object);
}

nb::dict DbapiResult::FetchPyTorch() {
	if (!result) {
		return nb::borrow<nb::dict>(nb::none());
	}
	auto tensors = result->FetchPyTorch();
	result = nullptr;
	return tensors;
}

nb::dict DbapiResult::FetchTF() {
	if (!result) {
		return nb::borrow<nb::dict>(nb::none());
	}
	auto tensors = result->FetchTF();
	result = nullptr;
	return tensors;
}

void DbapiResult::Complete() {
	if (result) {
		result->Complete();
		result = nullptr;
	}
}

void DbapiResult::Supersede() {
	if (!result) {
		return;
	}
	auto superseded = std::move(result);
	if (superseded->HasWritesToRun()) {
		superseded->Complete();
	} else {
		superseded->Close();
	}
}

unique_ptr<ColumnDataCollection> DbapiResult::TakeRows(vector<Identifier> &names) {
	if (!result) {
		return nullptr;
	}
	// Moved out first: the take runs the statement to its end or ends it, so the rows never come back, and a
	// failure must leave the result read out rather than reporting its error at every later fetch
	auto taken = std::move(result);
	names = taken->ResultNames();
	return taken->TakeRows(DuckDBPyConnection::CheckSignals);
}

void DbapiResult::Abort() {
	if (result) {
		result->Close();
		result = nullptr;
	}
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
	m.def("array_type", &DuckDBPyConnection::ArrayType, "Create an array type object of 'type'", nb::arg("type"),
	      nb::arg("size"));
	m.def("list_type", &DuckDBPyConnection::ListType, "Create a list type object of 'type'", nb::arg("type"));
	m.def("union_type", &DuckDBPyConnection::UnionType, "Create a union type object from 'members'",
	      nb::arg("members"));
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
	      nb::arg("key"), nb::arg("value"));
	m.def("duplicate", &DuckDBPyConnection::Cursor, "Create a duplicate of the current connection");
	m.def("execute", &DuckDBPyConnection::Execute,
	      "Execute the given SQL query, optionally using prepared statements with parameters set. A query runs until "
	      "its first rows are ready, so an error there raises here; every other statement runs to its end",
	      nb::arg("query"), nb::arg("parameters") = nb::none());
	m.def("executemany", &DuckDBPyConnection::ExecuteMany,
	      "Execute the given prepared statement multiple times using the list of parameter sets in parameters",
	      nb::arg("query"), nb::arg("parameters") = nb::none());
	m.def("close", &DuckDBPyConnection::Close, "Close the connection");
	m.def("interrupt", &DuckDBPyConnection::Interrupt, "Interrupt pending operations");
	m.def("complete", &DuckDBPyConnection::Complete, "Run the result of execute() to its end without reading its rows");
	m.def("abort", &DuckDBPyConnection::Abort,
	      "Cancel the rest of the result of execute(); a statement that may write fails an open transaction");
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
	m.def("torch", &DuckDBPyConnection::FetchPyTorch, "Fetch a result as dict of PyTorch Tensors following execute()");
	m.def("tf", &DuckDBPyConnection::FetchTF, "Fetch a result as dict of TensorFlow Tensors following execute()");
	m.def("begin", &DuckDBPyConnection::Begin, "Start a new transaction");
	m.def("commit", &DuckDBPyConnection::Commit, "Commit changes performed within a transaction");
	m.def("rollback", &DuckDBPyConnection::Rollback, "Roll back changes performed within a transaction");
	m.def("checkpoint", &DuckDBPyConnection::Checkpoint,
	      "Synchronizes data in the write-ahead log (WAL) to the database data file (no-op for in-memory connections)");
	m.def("append", &DuckDBPyConnection::Append, "Append the passed DataFrame to the named table",
	      nb::arg("table_name"), nb::arg("df"), nb::kw_only(), nb::arg("by_name") = false);
	m.def(
	    "appender", &DuckDBPyConnection::CreateAppender,
	    "Create an appender that buffers rows and inserts, upserts, updates or deletes them in the named table at each "
	    "flush",
	    nb::arg("table"), nb::kw_only(), nb::arg("columns").none() = nb::none(), nb::arg("action") = "insert",
	    nb::arg("key").none() = nb::none());
	m.def("query_appender", &DuckDBPyConnection::CreateQueryAppender,
	      "Create an appender that buffers rows of the declared columns and runs the DML over them at each flush",
	      nb::arg("dml"), nb::arg("columns"), nb::kw_only(), nb::arg("source").none() = nb::none());
	m.def("register", &DuckDBPyConnection::RegisterPythonObject,
	      "Register the passed Python Object value for querying with a view", nb::arg("view_name"),
	      nb::arg("python_object"));
	m.def("_register_arrow_source", &DuckDBPyConnection::RegisterArrowSource,
	      "Register a source from duckdb._sources for querying with a view", nb::arg("view_name"), nb::arg("source"));
	m.def("unregister", &DuckDBPyConnection::UnregisterPythonObject, "Unregister the view name", nb::arg("view_name"));
	m.def("extract_statements", &DuckDBPyConnection::ExtractStatements,
	      "Parse the query string and extract the Statement object(s) produced", nb::arg("query"));
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
	engine::StatementGroup statements(EngineSession(), query, false);
	while (auto statement = statements.Next()) {
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
	// Held from the check to the change, so no result can open in between and be ended by it
	engine::ConnectionClaim claim(context);
	engine::VerifyNoOpenResult(context);

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
	// Refused before any user code or type lookup runs, so neither works on a busy connection or on its open result
	engine::VerifyNoOpenResult(*con.GetConnection().context);
	// Converted before the claim, because a type name is parsed on this connection, which refuses while claimed
	auto scalar_function = CreateScalarUDF(name, udf, parameters_p, return_type_p, type == PythonUDFType::ARROW,
	                                       null_handling, exception_handling, side_effects);
	auto &connection = con.GetConnection();
	auto &context = *connection.context;
	// Held from the check to the change, so no result can open in between and be ended by it, and checked again
	// because converting runs user code that can open a result
	engine::ConnectionClaim claim(context);
	engine::VerifyNoOpenResult(context);

	if (context.transaction.HasActiveTransaction()) {
		context.CancelTransaction();
	}
	if (registered_functions.find(name) != registered_functions.end()) {
		throw NotImplementedException("A function by the name of '%s' is already created, creating multiple "
		                              "functions with the same name is not supported yet, please remove it first",
		                              name);
	}
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
	// No __del__: the cyclic collector finalizes a wrapper that only garbage reaches while the connection can have
	// other owners, the default connection's holder for one, so a connection ends at close() or when its last owner
	// goes.

	InitializeConnectionMethods(connection_module);
	DuckDBPySeam::InitializeConnectionMethods(connection_module);
	connection_module.def_prop_ro("description", &DuckDBPyConnection::GetDescription,
	                              "Get result set attributes, mainly column names");
	connection_module.def_prop_ro("rowcount", &DuckDBPyConnection::GetRowcount, "Get result set row count");
	DuckDBPyConnection::ImportCache();
}

static bool IsRollback(optional_ptr<const SQLStatement> statement) {
	if (!statement || statement->type != StatementType::TRANSACTION_STATEMENT) {
		return false;
	}
	return statement->Cast<TransactionStatement>().info->type == TransactionType::ROLLBACK;
}

static bool HasParameters(nb::handle params) {
	if (params.is_none()) {
		return false;
	}
	auto type = Py_TYPE(params.ptr());
	auto has_length = PyType_GetSlot(type, Py_sq_length) || PyType_GetSlot(type, Py_mp_length);
	if (!has_length) {
		// An object without a length, such as a generator, cannot be checked without consuming it
		return true;
	}
	auto size = PyObject_Size(params.ptr());
	if (size < 0) {
		throw nb::python_error();
	}
	return size > 0;
}

static bool HasParameterSets(const nb::list &sets) {
	// Every set is read, so a length that raises does so before anything runs
	bool given = false;
	for (auto parameters : sets) {
		given = HasParameters(parameters) || given;
	}
	return given;
}

[[noreturn]] static void RefuseParameters(engine::StatementGroup &statements) {
	DUCKDB_PY_ASSERT_GIL_HELD();
	nb::gil_scoped_release release;
	statements.RefuseParameters();
}

//! The engine lets parameters through only to a statement that expands to no statements, which takes none of them;
//! its empty result returns nothing and has no columns
static bool ExpandedToNothing(const engine::Result &result) {
	return result.MetadataAvailable() && result.Properties().return_type == StatementReturnType::NOTHING &&
	       result.Names().empty();
}

void DuckDBPyConnection::SupersedeResult(bool cancel) {
	if (!con.HasResult()) {
		return;
	}
	// The result is dropped even when draining it raises, so the error is reported once
	auto &result = con.GetResult();
	try {
		if (cancel) {
			result.Abort();
		} else {
			result.Supersede();
		}
	} catch (...) {
		con.SetResult(nullptr);
		throw;
	}
	con.SetResult(nullptr);
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::ExecuteMany(const nb::object &query, nb::object params_p) {
	nb::gil_scoped_acquire gil;
	ConnectionLockGuard conn_lock(*this);
	if (params_p.is_none()) {
		params_p = nb::list();
	}

	auto statements = GetStatements(query);
	bool has_statement;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		has_statement = statements.First() != nullptr;
	}
	auto list_like = duckdb::PyUtil::IsListLike(params_p);
	nb::list outer_list;
	if (list_like) {
		outer_list = nb::list(params_p);
	}
	// Input without a statement returns nothing, as execute() does, so only a query has parameter sets to check
	if (has_statement && !list_like) {
		throw InvalidInputException("executemany requires a list of parameter sets to be provided");
	}
	if (has_statement && outer_list.empty()) {
		throw InvalidInputException("executemany requires a non-empty list of parameter sets to be provided");
	}
	auto parameters_given = !list_like || HasParameterSets(outer_list);
	// FIXME: DBAPI says to not accept an 'executemany' call with multiple statements
	auto last_statement = RunAllButLast(statements, true);
	if (!last_statement && parameters_given) {
		RefuseParameters(statements);
	}

	shared_ptr<engine::Result> query_result;
	int64_t total_changed_rows = -1;
	if (last_statement) {
		auto prep = PrepareQuery(std::move(last_statement));

		// Execute once for every set of parameters that are provided
		for (auto parameters : outer_list) {
			auto params = nb::borrow<nb::object>(parameters);
			query_result = ExecuteInternal(prep, std::move(params), statements);
			auto changed_rows = query_result->ChangedRows();
			if (changed_rows >= 0) {
				total_changed_rows = (total_changed_rows < 0 ? 0 : total_changed_rows) + changed_rows;
			}
		}
	}
	shared_ptr<engine::Result> result;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		result = statements.Choose(query_result);
	}
	if (!result) {
		// TODO: should we throw?
		return nullptr;
	}
	auto py_result = std::make_shared<DuckDBPyResult>(std::move(result));
	// The rowcount counts what the parameterized statement changed, whichever statement's rows the result holds
	if (query_result) {
		py_result->OverrideRowcount(total_changed_rows);
	}
	con.SetResult(std::make_unique<DbapiResult>(std::move(py_result)));

	return shared_from_this();
}

void DuckDBPyConnection::CheckSignals() {
	nb::gil_scoped_acquire gil;
	if (PyErr_CheckSignals() != 0) {
		throw std::runtime_error("Query interrupted");
	}
}

engine::Session DuckDBPyConnection::EngineSession() {
	return engine::Session(con.GetConnection().context);
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
                                                                 optional_ptr<PreparedStatement> prep) {
	identifier_map_t<BoundParameterData> named_values;
	if (duckdb::PyUtil::IsListLike(params)) {
		if (prep && prep->GetParameterCount() != nb::len(params)) {
			if (nb::len(params) == 0) {
				throw InvalidInputException("Expected %d parameters, but none were supplied",
				                            prep->GetParameterCount());
			}
			throw InvalidInputException("Prepared statement needs %d parameters, %d given", prep->GetParameterCount(),
			                            nb::len(params));
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

engine::Prepared DuckDBPyConnection::PrepareQuery(unique_ptr<SQLStatement> statement) {
	auto session = EngineSession();
	engine::Prepared prep;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		unique_lock<std::recursive_mutex> lock(py_connection_lock);
		prep = session.Prepare(std::move(statement));
	}
	return prep;
}

shared_ptr<engine::Result> DuckDBPyConnection::ExecuteInternal(engine::Prepared &prep, nb::object params,
                                                               engine::StatementGroup &statements) {
	if (params.is_none()) {
		params = nb::list();
	}
	auto session = EngineSession();
	auto parameters_given = HasParameters(params);

	// A statement the engine expands takes no parameters of its own: the engine refuses them when it expands into
	// several, and the client below when it expands to none, so the count is not checked for it here
	optional_ptr<PreparedStatement> counted = prep.statement.get();
	if (prep.statement->GetParameterCount() == 0 && !prep.statement->MetadataAvailable()) {
		counted = nullptr;
	}
	// Execute the prepared statement with the prepared parameters
	auto named_values = TransformPreparedParameters(session.Context(), params, counted);
	shared_ptr<engine::Result> res;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		unique_lock<std::recursive_mutex> lock(py_connection_lock);

		res = session.Submit(prep, named_values, engine::Format::Chunks());
		if (parameters_given && ExpandedToNothing(*res)) {
			statements.RefuseParameters();
		}
		statements.ReleaseIfReplaced(*res);
		res->Retain(CheckSignals);
	}
	return res;
}

shared_ptr<engine::Result> DuckDBPyConnection::PrepareAndSubmitInternal(unique_ptr<SQLStatement> statement,
                                                                        nb::object params) {
	if (params.is_none()) {
		params = nb::list();
	}
	auto session = EngineSession();

	auto named_values = TransformPreparedParameters(session.Context(), params);

	shared_ptr<engine::Result> res;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		unique_lock<std::recursive_mutex> lock(py_connection_lock);
		res = session.Submit(std::move(statement), named_values, engine::Format::Chunks());
	}
	return res;
}

engine::StatementGroup DuckDBPyConnection::GetStatements(const nb::object &query) {
	if (nb::isinstance<DuckDBPyStatement>(query)) {
		auto &statement_obj = nb::cast<DuckDBPyStatement &>(query);
		return engine::StatementGroup(EngineSession(), statement_obj.GetStatement());
	}
	if (nb::isinstance<nb::str>(query)) {
		auto sql_query = nb::cast<std::string>(nb::str(query));
		return engine::StatementGroup(EngineSession(), sql_query);
	}
	throw InvalidInputException("Please provide either a DuckDBPyStatement or a string representing the query");
}

unique_ptr<SQLStatement> DuckDBPyConnection::RunAllButLast(engine::StatementGroup &statements, bool supersede) {
	DUCKDB_PY_ASSERT_GIL_HELD();
	if (supersede) {
		bool rollback;
		{
			nb::gil_scoped_release release;
			rollback = IsRollback(statements.First());
		}
		// A rollback discards what the open result would still write, so draining it first only risks failing the
		// rollback
		SupersedeResult(rollback);
	}
	nb::gil_scoped_release release;
	auto last_statement = statements.Next();
	if (last_statement) {
		last_statement = statements.RunAllButLast(std::move(last_statement), CheckSignals);
	}
	return last_statement;
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::ExecuteFromString(const string &query) {
	return Execute(nb::str(query.c_str(), query.size()));
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Execute(const nb::object &query, nb::object params) {
	nb::gil_scoped_acquire gil;
	ConnectionLockGuard conn_lock(*this);

	auto statements = GetStatements(query);
	auto parameters_given = HasParameters(params);
	// FIXME: SQLites implementation says to not accept an 'execute' call with multiple statements
	auto last_statement = RunAllButLast(statements, true);
	if (!last_statement && parameters_given) {
		RefuseParameters(statements);
	}
	shared_ptr<engine::Result> res;
	if (last_statement) {
		res = PrepareAndSubmitInternal(std::move(last_statement), std::move(params));
		if (parameters_given && ExpandedToNothing(*res)) {
			RefuseParameters(statements);
		}
	}
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		unique_lock<std::recursive_mutex> lock(py_connection_lock);
		if (res) {
			statements.ReleaseIfReplaced(*res);
			res->Settle(CheckSignals);
		}
		res = statements.Choose(std::move(res));
	}
	if (!res) {
		// TODO: should we throw?
		return nullptr;
	}
	auto py_result = std::make_shared<DuckDBPyResult>(std::move(res));
	con.SetResult(std::make_unique<DbapiResult>(std::move(py_result)));
	return shared_from_this();
}

std::unique_ptr<DuckDBPyAppender> DuckDBPyConnection::CreateAppender(const string &table,
                                                                     const std::optional<nb::list> &columns,
                                                                     const string &action,
                                                                     const std::optional<nb::list> &key) {
	return DuckDBPyAppender::ForTable(shared_from_this(), table, columns, action, key);
}

std::unique_ptr<DuckDBPyAppender> DuckDBPyConnection::CreateQueryAppender(const string &dml, const nb::dict &columns,
                                                                          const std::optional<string> &source) {
	return DuckDBPyAppender::ForQuery(shared_from_this(), dml, columns, source);
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

void DuckDBPyConnection::RegisterSourceView(const string &name, const SourceHandle &source) {
	// The session owns the context for the call, so a close() on another thread cannot free it under the call
	auto session = EngineSession();
	auto &client = session.Context();
	auto sql = "SELECT * FROM " + SQLQuotedIdentifier::ToString(source.Name());
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		// Held from the check to the change, so no result can open in between and be ended by it
		auto claim = session.Claim();
		engine::VerifyNoOpenResult(client);
		// Only a view register() made is replaced; a temporary view of the user's own makes the create fail, as
		// before
		auto replace = DuckDBPySeam::IsRegisteredView(client, name);
		// The view holds the source's table reference itself, so the catalog owns the object: a drop that commits
		// frees it, and one that rolls back brings it back with the view
		auto statement = DuckDBPySeam::ViewStatement(client, QualifiedName(Identifier(name)), sql, true, replace, true);
		session.RunClaimed(std::move(statement), CheckSignals);
	}
	// Recorded once the view exists, so a register() that failed leaves unregister() of the name a no-op
	RememberRegistered(name);
}

void DuckDBPyConnection::RememberRegistered(const string &name) {
	if (!ever_registered.insert(name).second) {
		return;
	}
	ever_registered_order.push_back(name);
	if (ever_registered_order.size() > EVER_REGISTERED_MEMORY) {
		ever_registered.erase(ever_registered_order.front());
		ever_registered_order.pop_front();
	}
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::RegisterPythonObject(const string &name,
                                                                             const nb::object &python_object) {
	con.GetConnection();
	auto source = DuckDBPySeam::RegisterSource(*this, python_object, name);
	RegisterSourceView(name, *source);
	return shared_from_this();
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::RegisterArrowSource(const string &name,
                                                                            const nb::object &source) {
	auto &client = *con.GetConnection().context;
	auto handle = DuckDBPySeam::RegisterTableRef(client, CreatePythonArrowScanRef(source, name), name);
	RegisterSourceView(name, *handle);
	return shared_from_this();
}

PathLike DuckDBPyConnection::GetPathLike(const nb::object &object) {
	return PathLike::Create(object, *this);
}

nb::tuple DuckDBPyConnection::PrepareScript(const nb::object &query, const nb::object &params, bool with_statement) {
	auto statements = GetStatements(query);
	auto last_statement = RunAllButLast(statements, false);
	if (!last_statement && HasParameters(params)) {
		RefuseParameters(statements);
	}
	shared_ptr<engine::Result> kept;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		kept = statements.Choose(nullptr);
	}
	// The group's parser keeps the settings it started with, so after an earlier statement such as a SET the parsed
	// last statement can differ from what its text parses to now
	auto earlier_ran = kept != nullptr;
	nb::object kept_result = nb::none();
	// A kept result that returns no rows only stands in for a last statement that returns none, which the caller runs
	if (kept && kept->Properties().return_type == StatementReturnType::QUERY_RESULT) {
		kept_result = nb::cast(std::make_shared<DuckDBPyResult>(std::move(kept)));
	}
	if (!last_statement) {
		if (with_statement) {
			return nb::make_tuple(kept_result, nb::none(), nb::none(), nb::none());
		}
		return nb::make_tuple(kept_result, nb::none(), nb::none());
	}
	auto &location = last_statement->stmt_location;
	auto text = last_statement->query;
	if (location.offset != QueryLocation::INVALID_OFFSET && location.offset + location.length <= text.size()) {
		text = text.substr(location.offset, location.length);
	}
	// The text gets embedded in a larger statement, where a terminating semicolon would end it early
	while (!text.empty() && (StringUtil::CharacterIsSpace(text.back()) || text.back() == ';')) {
		text.pop_back();
	}
	auto type_name = StringUtil::Lower(StatementTypeToString(last_statement->type));
	if (with_statement) {
		// The parsed statement too, so a caller binds it without parsing the text again
		nb::object statement = nb::none();
		if (!earlier_ran) {
			statement = nb::cast(std::make_shared<DuckDBPyStatement>(std::move(last_statement)));
		}
		return nb::make_tuple(kept_result, nb::str(text.c_str(), text.size()),
		                      nb::str(type_name.c_str(), type_name.size()), statement);
	}
	return nb::make_tuple(kept_result, nb::str(text.c_str(), text.size()),
	                      nb::str(type_name.c_str(), type_name.size()));
}

unordered_set<string> DuckDBPyConnection::GetTableNames(const string &query, bool qualified) {
	auto &connection = con.GetConnection();
	return connection.GetTableNames(query, qualified);
}

std::shared_ptr<DuckDBPyConnection> DuckDBPyConnection::UnregisterPythonObject(const string &name) {
	// The session owns the context for the call, as in RegisterSourceView
	auto session = EngineSession();
	auto &client = session.Context();
	auto owned = owned_sources.find(name) != owned_sources.end();
	auto known = owned || ever_registered.find(name) != ever_registered.end();
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		// Held from the check to the change, so no result can open in between and be ended by it
		auto claim = session.Claim();
		if (!known) {
			// A name this connection did not register here, a view renamed in SQL for one, is asked of the catalog
			// only when that disturbs nothing: beside an open DB-API result it stays the no-op it always was
			try {
				client.VerifyNoOpenResult();
			} catch (ResourceInUseException &) {
				return shared_from_this();
			}
		}
		engine::VerifyNoOpenResult(client);
		const auto quoted_name = SQLQuotedIdentifier::ToString(name);
		// The catalog says whether register() made the view, so a view the user dropped, or one a rolled back
		// transaction took away, is not an error, and an aborted transaction refuses the lookup as it would the drop
		string drop;
		if (DuckDBPySeam::IsRegisteredView(client, name)) {
			// Named in the temporary schema, so a persistent view of the same name is never the one dropped
			drop = "DROP VIEW IF EXISTS temp.main." + quoted_name;
		} else if (owned) {
			// A view whose sources the connection owns, by the name it was given
			drop = "DROP VIEW IF EXISTS " + quoted_name;
		}
		if (!drop.empty()) {
			Parser parser(client);
			parser.ParseQuery(drop);
			session.RunClaimed(std::move(parser.statements[0]), CheckSignals);
		}
	}
	if (owned) {
		// Moved out before it is destroyed: dropping a handle can run a finalizer that writes this map
		auto released = std::move(owned_sources[name]);
		owned_sources.erase(name);
	}
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

int64_t DuckDBPyConnection::GetRowcount() {
	ConnectionLockGuard conn_lock(*this);
	if (!con.HasResult()) {
		return -1;
	}
	return con.GetResult().Rowcount();
}

void DuckDBPyConnection::Complete() {
	ConnectionLockGuard conn_lock(*this);
	if (con.HasResult()) {
		con.GetResult().Complete();
	}
}

void DuckDBPyConnection::Abort() {
	ConnectionLockGuard conn_lock(*this);
	if (con.HasResult()) {
		con.GetResult().Abort();
	}
}

void DuckDBPyConnection::Close() {
	ConnectionLockGuard conn_lock(*this);
	con.SetResult(nullptr);
	DUCKDB_PY_ASSERT_GIL_HELD();
	// Released with the GIL held, since the registry entries they let go own Python references
	ReleaseOwnedSources();
	// Taken out under the GIL, so a seam call reading the guard under the GIL sees all of it or a closed connection.
	auto released = con.Release();
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
		released.Destroy();
	}
	// https://peps.python.org/pep-0249/#Connection.close
	cursors.ClearCursors();
	registered_functions.clear();
}

void DuckDBPyConnection::ReleaseOwnedSources() {
	// Moved out before anything is destroyed: dropping a handle can run a finalizer, which can hand the GIL to a
	// thread that writes these members, so they must already be in their final state
	auto owned = std::move(owned_sources);
	owned_sources.clear();
	auto identity = std::move(source_identity);
	source_identity.reset();
}

void DuckDBPyConnection::Interrupt() {
	auto &connection = con.GetConnectionWhileBusy();
	connection.Interrupt();
}

double DuckDBPyConnection::QueryProgress() {
	auto &connection = con.GetConnectionWhileBusy();
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
	// Held from the check to the change, so no result can open in between and be ended by it
	engine::ConnectionClaim claim(*connection.context);
	engine::VerifyNoOpenResult(*connection.context);
	const ExtensionLoadOptions extension_opts = {extension};
	ExtensionHelper::LoadExternalExtension(*connection.context, extension_opts);
}

// sys.is_finalizing is asked because Py_IsFinalizing is public only from Python 3.13; without sys the interpreter is
// finalizing.
static bool InterpreterIsFinalizing() {
	PyObject *is_finalizing = PySys_GetObject("is_finalizing");
	if (!is_finalizing) {
		return true;
	}
	PyObject *result = PyObject_CallNoArgs(is_finalizing);
	if (!result) {
		PyErr_Clear();
		return true;
	}
	const int truth = PyObject_IsTrue(result);
	Py_DECREF(result);
	if (truth < 0) {
		PyErr_Clear();
		return true;
	}
	return truth != 0;
}

// The mutex guards only reads and swaps of the pointer: opening or destroying a connection releases the GIL and can run
// Python code that fetches the default connection again, so both happen after the mutex is released.
std::shared_ptr<DuckDBPyConnection> DefaultConnectionHolder::Get() {
	bool was_released;
	{
		lock_guard<mutex> guard(l);
		if (connection && !connection->con.ConnectionIsClosed()) {
			return connection;
		}
		was_released = released;
	}
	if (was_released && InterpreterIsFinalizing()) {
		throw ConnectionException("the default connection is not available during interpreter shutdown");
	}
	nb::dict config_dict;
	auto opened = DuckDBPyConnection::Connect(nb::str(":memory:"), false, config_dict);
	std::shared_ptr<DuckDBPyConnection> replaced;
	{
		lock_guard<mutex> guard(l);
		if (connection && !connection->con.ConnectionIsClosed()) {
			return connection;
		}
		replaced = std::move(connection);
		connection = opened;
	}
	return opened;
}

void DefaultConnectionHolder::Set(std::shared_ptr<DuckDBPyConnection> conn) {
	bool was_released;
	{
		lock_guard<mutex> guard(l);
		was_released = released;
	}
	if (was_released && InterpreterIsFinalizing()) {
		return;
	}
	{
		lock_guard<mutex> guard(l);
		connection.swap(conn);
	}
}

void DefaultConnectionHolder::Release() {
	std::shared_ptr<DuckDBPyConnection> released_connection;
	{
		lock_guard<mutex> guard(l);
		released = true;
		connection.swap(released_connection);
	}
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
	return result.FetchNumpy();
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

nb::object DuckDBPyConnection::ResultAsArrow(idx_t batch_size, bool reader) {
	ConnectionLockGuard conn_lock(*this);
	// Asked before the result, so a closed connection reports itself and not the slot its close emptied
	auto session = EngineSession();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	// While CONNECT-ed the engine sends every statement to the remote as text, and a statement over rows held
	// here has none, so the rows would be taken for nothing
	if (session.Context().IsConnected()) {
		throw InvalidInputException("Arrow output on a DB-API result is not available while CONNECT-ed; DISCONNECT "
		                            "first, or use con.sql(query).to_arrow_table() or to_arrow_reader()");
	}
	// Everything that can refuse before the rows are taken does so, so a refused call costs no rows
	if (reader) {
		DuckDBPyResult::CheckBatchSize(batch_size);
	}
	auto format = engine::Format::Arrow(batch_size);
	// Imported before the engine checks, so no Python code runs between what they find and the take
	nb::module_::import_("pyarrow");
	auto held = con.GetResult().EngineResult();
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		auto claim = session.Claim();
		// The engine refuses an aborted transaction here under its own lock, and a running result keeps its
		// transaction active, so the empty function neither begins nor commits anything the result depends on
		session.Context().RunFunctionInTransaction([] {});
		// Another open result, a relation's reader for one, would refuse the statement over the rows after the take;
		// a running DB-API result is itself the one open result and the take drains it
		if (held && !held->Running()) {
			engine::VerifyNoOpenResult(session.Context());
		}
	}

	vector<Identifier> names;
	auto rows = con.GetResult().TakeRows(names);
	if (!rows) {
		return nb::none();
	}
	shared_ptr<engine::Result> arrow;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		arrow = session.Submit(std::move(rows), names, format);
		if (reader) {
			arrow->Settle(DuckDBPyConnection::CheckSignals);
		} else {
			arrow->Retain(DuckDBPyConnection::CheckSignals);
		}
	}
	DuckDBPyResult converted(std::move(arrow), std::move(names));
	if (reader) {
		return nb::cast(converted.FetchRecordBatchReader(batch_size));
	}
	return nb::cast(converted.FetchArrowTable(batch_size, false));
}

identifier_map_t<Value> TransformPyConfigDict(const nb::dict &py_config_dict) {
	identifier_map_t<Value> config_dict;
	for (auto kv : py_config_dict) {
		// Config values may be int/bool/str; str-ify them rather than
		// requiring an actual Python str (nb::cast<std::string> would throw on a non-str like 0 or False).
		auto key = nb::cast<std::string>(nb::str(kv.first));
		auto val = nb::cast<std::string>(nb::str(kv.second));
		config_dict[Identifier(key)] = Value(val);
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
	RegisterPythonArrowScan(db_instance);
}

static std::shared_ptr<DuckDBPyConnection> FetchOrCreateInstance(const string &database_path, DBConfig &config) {
	auto res = std::make_shared<DuckDBPyConnection>();
	bool cache_instance = database_path != ":memory:" && !database_path.empty();
	config.replacement_scans.emplace_back(PythonReplacementScan::Replace);
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		unique_lock<std::recursive_mutex> lock(res->py_connection_lock);
		auto database = GetModuleState().instance_cache.GetOrCreateInstance(database_path, config, cache_instance,
		                                                                    InstantiateNewInstance);
		res->con.SetDatabase(std::move(database));
		res->con.SetConnection(make_uniq<Connection>(res->con.GetDatabase()));
	}
	return res;
}

bool IsDefaultConnectionString(const string &database, bool read_only, identifier_map_t<Value> &config) {
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

nb::object DuckDBPyConnection::ConsumedArrowReaders() {
	auto &readers = GetModuleState().consumed_arrow_readers;
	if (!readers) {
		nb::dict record;
		// Creating the record can run Python, so another thread may have published one in the meantime.
		if (!readers) {
			readers = make_uniq<PyObjectHolder>(std::move(record));
		}
	}
	return readers->obj;
}

void DuckDBPyConnection::SetArrowSources(const nb::object &sources) {
	auto adapt = sources.attr("adapt");
	auto nodes = sources.attr("nodes");
	auto &state = GetModuleState();
	state.arrow_sources_adapt = make_uniq<PyObjectHolder>(std::move(adapt));
	state.arrow_sources_nodes = make_uniq<PyObjectHolder>(std::move(nodes));
}

static nb::object ArrowSourcesPart(const unique_ptr<PyObjectHolder> &part) {
	if (!part) {
		// A program can outlive Cleanup; the package is then taken from sys.modules again, never imported anew.
		auto modules = nb::borrow<nb::dict>(PyImport_GetModuleDict());
		auto sources = PyDict_GetItemString(modules.ptr(), "duckdb._sources");
		if (sources) {
			DuckDBPyConnection::SetArrowSources(nb::borrow(sources));
		}
	}
	if (!part) {
		throw InvalidInputException(
		    "Python Arrow objects cannot be scanned: the duckdb package did not set up duckdb._sources");
	}
	return part->obj;
}

nb::object DuckDBPyConnection::ArrowSourcesAdapt() {
	return ArrowSourcesPart(GetModuleState().arrow_sources_adapt);
}

nb::object DuckDBPyConnection::ArrowSourcesNodes() {
	return ArrowSourcesPart(GetModuleState().arrow_sources_nodes);
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
	GetModuleState().default_connection.Release();
	GetModuleState().import_cache.reset();
	GetModuleState().consumed_arrow_readers.reset();
	GetModuleState().arrow_sources_adapt.reset();
	GetModuleState().arrow_sources_nodes.reset();
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

} // namespace duckdb

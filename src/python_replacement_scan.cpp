#include "duckdb_python/python_replacement_scan.hpp"
#include "duckdb_python/engine/connection.hpp"
#include "duckdb/main/db_instance_cache.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb/main/client_properties.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/numpy/numpy_array.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/dataframe.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb_python/pandas/pandas_scan.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb_python/pyrelation.hpp"
#include <duckdb/main/settings.hpp>

namespace duckdb {

static void CreateArrowScan(const string &name, nb::object entry, TableFunctionRef &table_function,
                            vector<unique_ptr<ParsedExpression>> &children, ClientProperties &client_properties,
                            PyArrowObjectType type, DatabaseInstance &db) {
	shared_ptr<ExternalDependency> external_dependency = make_shared_ptr<ExternalDependency>();
	if (type == PyArrowObjectType::MessageReader) {
		if (!db.ExtensionIsLoaded("nanoarrow")) {
			throw MissingExtensionException(
			    "The nanoarrow community extension is needed to read the Arrow IPC protocol. \n You can install it "
			    "with \"INSTALL nanoarrow FROM community;\". \n Then you can load it with \"LOAD nanoarrow;\"");
		}
		vector<Value> values;
		nb::list stream_messages;
		while (true) {
			try {
				nb::object message = entry.attr("read_next_message")();
				if (message.is_none()) {
					break;
				}
				stream_messages.append(message.attr("serialize")());
				const auto buffer_address =
				    nb::cast<int64_t>(stream_messages[stream_messages.size() - 1].attr("address"));
				const auto buffer_size = nb::cast<uint32_t>(stream_messages[stream_messages.size() - 1].attr("size"));
				child_list_t<Value> buffer_values;
				buffer_values.push_back({"ptr", Value::POINTER(buffer_address)});
				buffer_values.push_back({"size", Value::UBIGINT(buffer_size)});
				values.push_back(Value::STRUCT(buffer_values));
			} catch (const nb::python_error &e) {
				break;
			}
		}
		auto list_value = Value::LIST(values);
		children.push_back(ConstantExpression::FromValue(list_value));
		table_function.function = make_uniq<FunctionExpression>("scan_arrow_ipc", std::move(children));
		auto dependency_item = PythonDependencyItem::Create(stream_messages);
		external_dependency->AddDependency("replacement_cache", std::move(dependency_item));
	} else {
		table_function.bind_info = make_shared_ptr<PythonTableArrowArrayStreamFactory>(entry, client_properties, type);

		if (type == PyArrowObjectType::PyCapsule) {
			// Disable projection+filter pushdown for bare capsules (single-use, no PyArrow wrapper)
			table_function.function = make_uniq<FunctionExpression>("arrow_scan_dumb", std::move(children));
		} else if (type == PyArrowObjectType::PyCapsuleInterface) {
			// Try to load pyarrow.dataset for pushdown support
			auto &cache = *DuckDBPyConnection::ImportCache();
			if (!cache.pyarrow.dataset()) {
				// No pyarrow.dataset: scan without pushdown, DuckDB handles projection/filter post-scan
				table_function.function = make_uniq<FunctionExpression>("arrow_scan_dumb", std::move(children));
			} else {
				table_function.function = make_uniq<FunctionExpression>("arrow_scan", std::move(children));
			}
		} else {
			table_function.function = make_uniq<FunctionExpression>("arrow_scan", std::move(children));
		}
		external_dependency->AddDependency("replacement_cache", PythonDependencyItem::Create(entry));
	}
	table_function.external_dependency = std::move(external_dependency);
}

static void ThrowScanFailureError(const nb::object &entry, const string &name, const string &location = "") {
	string error;
	// nb::object wrap: nb::str() of a bare .attr() accessor is an ambiguous overload on MSVC.
	auto py_object_type = nb::cast<std::string>(nb::str(nb::object((entry).type().attr("__name__"))));
	error += StringUtil::Format("Python Object \"%s\" of type \"%s\"", name, py_object_type);
	if (!location.empty()) {
		error += StringUtil::Format(" found on line \"%s\"", location);
	}
	error +=
	    StringUtil::Format(" not suitable for replacement scans.\nMake sure "
	                       "that \"%s\" is either a pandas.DataFrame, duckdb.DuckDBPyRelation, pyarrow Table, Dataset, "
	                       "RecordBatchReader, Scanner, or NumPy ndarrays with supported format",
	                       name);
	throw InvalidInputException(error);
}

unique_ptr<TableRef> PythonReplacementScan::ReplacementObject(const nb::object &entry, const string &name,
                                                              ClientContext &context, bool relation) {
	auto replacement = TryReplacementObject(entry, name, context, relation);
	if (!replacement) {
		ThrowScanFailureError(entry, name);
	}
	return replacement;
}

namespace {

struct RelationDependencyItem : DependencyItem {
	explicit RelationDependencyItem(shared_ptr<Relation> rel_p) : rel(std::move(rel_p)) {
	}
	static shared_ptr<DependencyItem> Create(shared_ptr<Relation> rel) {
		return make_shared_ptr<RelationDependencyItem>(std::move(rel));
	}

	shared_ptr<Relation> rel;
};

} // namespace

unique_ptr<TableRef> PythonReplacementScan::TryReplacementObject(const nb::object &entry, const string &name,
                                                                 ClientContext &context, bool relation) {
	auto client_properties = context.GetClientProperties();
	auto table_function = make_uniq<TableFunctionRef>();
	vector<unique_ptr<ParsedExpression>> children;
	NumpyObjectType numpytype;
	PyArrowObjectType arrow_type;
	if (DuckDBPyConnection::IsPandasDataframe(entry)) {
		if (PandasDataFrame::IsPyArrowBacked(entry)) {
			auto table = PandasDataFrame::ToArrowTable(entry);
			CreateArrowScan(name, table, *table_function, children, client_properties, PyArrowObjectType::Table,
			                *context.db);
		} else {
			string name = "df_" + StringUtil::GenerateRandomName();
			auto new_df = PandasScanFunction::PandasReplaceCopiedNames(entry);
			table_function->function = make_uniq<FunctionExpression>("pandas_scan", std::move(children));
			table_function->bind_info = make_shared_ptr<PandasScanInfo>(std::move(new_df));
			auto dependency = make_uniq<ExternalDependency>();
			dependency->AddDependency("replacement_cache", PythonDependencyItem::Create(entry));
			table_function->external_dependency = std::move(dependency);
		}
	} else if (DuckDBPyRelation::IsRelation(entry)) {
		auto pyrel = nb::cast<DuckDBPyRelation *>(entry);
		if (!pyrel->CanBeRegisteredBy(context)) {
			throw InvalidInputException(
			    "Python Object \"%s\" of type \"DuckDBPyRelation\" not suitable for replacement scan.\nThe object was "
			    "created by another Connection and can therefore not be used by this Connection.",
			    name);
		}
		// create a subquery from the underlying relation object
		auto select = make_uniq<SelectStatement>();
		select->node = pyrel->GetRel().GetQueryNode();
		auto subquery = make_uniq<SubqueryRef>(std::move(select));
		auto dependency = make_uniq<ExternalDependency>();
		// The native relation carries the Python data the scan reads. Owning the wrapper instead would
		// own its connection, and an open result holding that closes a cycle the garbage
		// collector cannot see: connection to result to wrapper to connection.
		dependency->AddDependency("replacement_cache", RelationDependencyItem::Create(pyrel->GetSharedRelation()));
		subquery->external_dependency = std::move(dependency);
		return std::move(subquery);
	} else if (PolarsDataFrame::IsDataFrame(entry)) {
		// Polars DataFrames always go through one-time .to_arrow() materialization.
		// Polars's __arrow_c_stream__() serializes from its internal layout on every call,
		// which is expensive for repeated scans. The .to_arrow() path converts once.
		auto arrow_dataset = entry.attr("to_arrow")();
		CreateArrowScan(name, arrow_dataset, *table_function, children, client_properties, PyArrowObjectType::Table,
		                *context.db);
	} else if (PolarsDataFrame::IsLazyFrame(entry)) {
		CreateArrowScan(name, entry, *table_function, children, client_properties, PyArrowObjectType::PolarsLazyFrame,
		                *context.db);
	} else if ((arrow_type = DuckDBPyConnection::GetArrowType(entry)) != PyArrowObjectType::Invalid &&
	           !(arrow_type == PyArrowObjectType::MessageReader && !relation)) {
		CreateArrowScan(name, entry, *table_function, children, client_properties, arrow_type, *context.db);
	} else if (DuckDBPyConnection::IsAcceptedNumpyObject(entry) != NumpyObjectType::INVALID) {
		numpytype = DuckDBPyConnection::IsAcceptedNumpyObject(entry);
		string np_name = "np_" + StringUtil::GenerateRandomName();
		nb::dict data; // we will convert all the supported format to dict{"key": np.array(value)}.
		size_t idx = 0;
		switch (numpytype) {
		case NumpyObjectType::NDARRAY1D:
			data["column0"] = entry;
			break;
		case NumpyObjectType::NDARRAY2D: {
			idx = 0;
			NumpyArray ndarray(entry);
			for (auto item : ndarray.GetArray()) {
				data[("column" + std::to_string(idx)).c_str()] = item;
				idx++;
			}
			break;
		}
		case NumpyObjectType::LIST:
			idx = 0;
			for (auto item : nb::cast<nb::list>(entry)) {
				data[("column" + std::to_string(idx)).c_str()] = item;
				idx++;
			}
			break;
		case NumpyObjectType::DICT:
			data = nb::cast<nb::dict>(entry);
			break;
		default:
			throw NotImplementedException("Unsupported Numpy object");
			break;
		}
		table_function->function = make_uniq<FunctionExpression>("pandas_scan", std::move(children));
		table_function->bind_info = make_shared_ptr<PandasScanInfo>(std::move(data));
		auto dependency = make_uniq<ExternalDependency>();
		dependency->AddDependency("replacement_cache", PythonDependencyItem::Create(entry));
		table_function->external_dependency = std::move(dependency);
	} else {
		// This throws an error later on!
		return nullptr;
	}
	return std::move(table_function);
}

static bool IsBuiltinFunction(const nb::object &object) {
	auto &import_cache_py = *DuckDBPyConnection::ImportCache();
	return duckdb::PyUtil::IsInstance(object, import_cache_py.types.BuiltinFunctionType());
}

static unique_ptr<TableRef> TryReplacement(nb::dict &dict, const string &name, ClientContext &context,
                                           nb::object &current_frame) {
	auto table_name = nb::str(name.c_str(), name.size());
	if (!dict.contains(table_name)) {
		// not present in the globals
		return nullptr;
	}
	const nb::object &entry = dict[table_name];

	if (IsBuiltinFunction(entry)) {
		return nullptr;
	}

	auto result = PythonReplacementScan::TryReplacementObject(entry, name, context);
	if (!result) {
		std::string location = nb::cast<std::string>(current_frame.attr("f_code").attr("co_filename"));
		location += ":";
		location += nb::cast<std::string>(nb::str(nb::object(current_frame.attr("f_lineno"))));
		ThrowScanFailureError(entry, name, location);
	}
	return result;
}

// Materialize a real nb::dict from a frame's f_locals/f_globals. f_globals is already a dict (borrow it);
// f_locals can be a FrameLocalsProxy on Python 3.13+ (PEP 667), which is a mapping but not a dict -- copy it.
static nb::dict FrameDictToDict(const nb::object &frame_dict) {
	if (PyDict_Check(frame_dict.ptr())) {
		return nb::borrow<nb::dict>(frame_dict);
	}
	nb::dict materialized;
	materialized.update(frame_dict);
	return materialized;
}

static unique_ptr<TableRef> ReplaceInternal(ClientContext &context, const string &table_name) {
	Value result;
	auto lookup_result = context.TryGetCurrentSetting("python_enable_replacements", result);
	D_ASSERT((bool)lookup_result);
	auto enabled = result.GetValue<bool>();

	if (!enabled) {
		return nullptr;
	}

	lookup_result = context.TryGetCurrentSetting("python_scan_all_frames", result);
	D_ASSERT((bool)lookup_result);
	auto scan_all_frames = result.GetValue<bool>();

	nb::gil_scoped_acquire acquire;
	nb::object current_frame;
	try {
		current_frame = nb::module_::import_("inspect").attr("currentframe")();
	} catch (nb::python_error &e) {
		//! Likely no call stack exists, just safely return
		return nullptr;
	}

	bool has_locals = false;
	bool has_globals = false;
	do {
		if (nb::none().is(current_frame)) {
			break;
		}

		nb::object local_dict_p;
		try {
			local_dict_p = current_frame.attr("f_locals");
		} catch (nb::python_error &e) {
			return nullptr;
		}
		has_locals = !nb::none().is(local_dict_p);
		if (has_locals) {
			// search local dictionary. On Python 3.13+ (PEP 667) frame.f_locals is a FrameLocalsProxy, not a
			// dict, so cast<nb::dict> would fail; materialize a real dict from the mapping.
			auto local_dict = FrameDictToDict(local_dict_p);
			auto result = TryReplacement(local_dict, table_name, context, current_frame);
			if (result) {
				return result;
			}
		}
		nb::object global_dict_p;
		try {
			global_dict_p = current_frame.attr("f_globals");
		} catch (nb::python_error &e) {
			return nullptr;
		}
		has_globals = !nb::none().is(global_dict_p);
		if (has_globals) {
			auto global_dict = FrameDictToDict(global_dict_p);
			// search global dictionary
			auto result = TryReplacement(global_dict, table_name, context, current_frame);
			if (result) {
				return result;
			}
		}
		try {
			current_frame = current_frame.attr("f_back");
		} catch (nb::python_error &e) {
			return nullptr;
		}
	} while (scan_all_frames && (has_locals || has_globals));
	return nullptr;
}

unique_ptr<TableRef> PythonReplacementScan::Replace(ClientContext &context, ReplacementScanInput &input,
                                                    optional_ptr<ReplacementScanData> data) {
	auto &table_name = input.table_name;
	auto &config = DBConfig::GetConfig(context);
	if (!Settings::Get<EnableExternalAccessSetting>(config)) {
		return nullptr;
	}

	auto state = engine::ConnectionState::Get(context);
	if (auto resolved = state->RecallResolved(table_name)) {
		return resolved;
	}
	auto result = ReplaceInternal(context, table_name);
	if (result) {
		state->RecordResolved(table_name, *result);
	}
	return result;
}

} // namespace duckdb

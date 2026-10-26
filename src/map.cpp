#include "duckdb_python/map.hpp"
#include "duckdb_python/engine/connection.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/numpy/numpy_scan.hpp"
#include "duckdb_python/pandas/pandas_bind.hpp"
#include "duckdb_python/numpy/numpy_result_conversion.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/pandas/column/pandas_numpy_column.hpp"
#include "duckdb_python/pandas/pandas_scan.hpp"
#include "duckdb_python/dataframe.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb_python/dataframe.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

namespace duckdb {

MapFunction::MapFunction() : TableFunction("python_map_function", {LogicalType::TABLE}, nullptr, MapFunctionBind) {
	in_out_function = MapFunctionExec;
}

struct MapFunctionData : public TableFunctionData {
	explicit MapFunctionData(shared_ptr<MapFunctionInfo> info_p) : info(std::move(info_p)) {
	}
	shared_ptr<MapFunctionInfo> info;
	vector<LogicalType> in_types, out_types;
	vector<Identifier> in_names, out_names;
};

static nb::object FunctionCall(NumpyResultConversion &conversion, const vector<Identifier> &names, PyObject *function) {
	nb::dict in_numpy_dict;
	for (idx_t col_idx = 0; col_idx < names.size(); col_idx++) {
		in_numpy_dict[names[col_idx].c_str()] = conversion.ToArray(col_idx);
	}

	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto pandas_df = import_cache.pandas.DataFrame();
	auto in_df = pandas_df(in_numpy_dict);
	D_ASSERT(in_df.ptr());

	D_ASSERT(function);
	// Not PyObject_CallOneArg: nanobind links macOS extensions against a fixed list of CPython symbols that lacks it.
	auto *df_obj = PyObject_CallFunctionObjArgs(function, in_df.ptr(), nullptr);
	if (!df_obj) {
		PyErr_PrintEx(1);
		throw InvalidInputException("Python error. See above for a stack trace.");
	}

	auto df = nb::steal<nb::object>(df_obj);
	if (df.is_none()) { // no return, probably modified in place
		throw InvalidInputException("No return value from Python function");
	}

	if (!nb::isinstance<PandasDataFrame>(df)) {
		throw InvalidInputException(
		    "Expected the UDF to return an object of type 'pandas.DataFrame', found '%s' instead",
		    nb::cast<std::string>(nb::str(nb::object(df.attr("__class__")))));
	}
	if (PandasDataFrame::IsPyArrowBacked(df)) {
		throw InvalidInputException(
		    "Produced DataFrame has columns that are backed by PyArrow, which is not supported yet in 'map'");
	}

	return df;
}

//! Converting here releases the traceback, and any DataFrames it holds, under the GIL and the callback scope.
static Exception AsDuckDBError(const nb::python_error &error) {
	return Exception(ExceptionType::INVALID, error.what());
}

static bool ContainsNullType(const vector<LogicalType> &types) {
	for (auto &type : types) {
		if (type.id() == LogicalTypeId::SQLNULL) {
			return true;
		}
	}
	return false;
}

static void OverrideNullType(vector<LogicalType> &return_types, const vector<Identifier> &return_names,
                             const vector<LogicalType> &original_types, const vector<Identifier> &original_names) {
	if (!ContainsNullType(return_types)) {
		// Nothing to override, none of the returned types are NULL
		return;
	}
	if (return_types.size() != original_types.size()) {
		// FIXME: we can probably infer from the names in this case
		//  Cant infer what the type should be
		return;
	}
	for (idx_t i = 0; i < return_types.size(); i++) {
		auto &return_type = return_types[i];
		auto &original_type = original_types[i];

		if (return_type != LogicalTypeId::SQLNULL) {
			continue;
		}
		if (return_names[i] != original_names[i]) {
			throw InvalidInputException(
			    "Returned dataframe contains NULL type, and we could not infer the desired type");
		}
		// Override the NULL with the original type
		return_type = original_type;
	}
}

unique_ptr<FunctionData> BindExplicitSchema(unique_ptr<MapFunctionData> function_data, PyObject *schema_p,
                                            vector<LogicalType> &types, vector<Identifier> &names) {
	D_ASSERT(schema_p != Py_None);

	auto schema_object = nb::borrow<nb::dict>(schema_p);
	if (!nb::isinstance<nb::dict>(schema_object)) {
		throw InvalidInputException("'schema' should be given as a Dict[str, DuckDBType]");
	}
	auto schema = nb::cast<nb::dict>(schema_object);

	auto column_count = schema.size();

	types.reserve(column_count);
	names.reserve(column_count);
	nb::list labels;
	for (auto item : schema) { // nanobind dict iteration yields std::pair<handle,handle> by value
		labels.append(item.first);
		auto type_p = item.second;
		// TryConvert applies the same implicit conversions a DuckDBPyType parameter would (DuckDBPyType instance,
		// a type string, or a Python type object), and reports a clear error instead of a raw cast failure.
		std::unique_ptr<DuckDBPyType> type;
		if (!DuckDBPyType::TryConvert(nb::borrow<nb::object>(type_p), type)) {
			string actual_type = nb::cast<std::string>(nb::str((type_p).type()));
			throw InvalidInputException("'schema' value could not be converted to a DuckDBPyType, got '%s'",
			                            actual_type);
		}
		types.push_back(type->Type());
	}
	for (auto &name : ColumnNamesFromLabels(labels)) {
		names.emplace_back(name);
	}

	function_data->out_names = names;
	function_data->out_types = types;

	return std::move(function_data);
}

// we call the passed function with a zero-row data frame to infer the output columns and their names.
// they better not change in the actual execution ^^
static unique_ptr<FunctionData> BindMap(ClientContext &context, shared_ptr<MapFunctionInfo> info,
                                        TableFunctionBindInput &input, vector<LogicalType> &return_types,
                                        vector<Identifier> &names) {
	auto data_uptr = make_uniq<MapFunctionData>(std::move(info));
	auto &data = *data_uptr;
	auto function = data.info->function.obj.ptr();
	auto explicit_schema = data.info->schema.obj.ptr();

	data.in_names = input.input_table_names;
	data.in_types = input.input_table_types;

	if (explicit_schema != Py_None) {
		return BindExplicitSchema(std::move(data_uptr), explicit_schema, return_types, names);
	}
	NumpyResultConversion conversion(data.in_types, 0, context.GetClientProperties());
	auto df = PandasScanFunction::PandasWithBoundNames(FunctionCall(conversion, data.in_names, function));
	vector<PandasColumnBindData> pandas_bind_data; // unused
	Pandas::Bind(context, df, pandas_bind_data, return_types, names);

	// output types are potentially NULL, this happens for types that map to 'object' dtype
	OverrideNullType(return_types, names, data.in_types, data.in_names);

	data.out_names = names;
	data.out_types = return_types;
	return std::move(data_uptr);
}

unique_ptr<FunctionData> MapFunction::MapFunctionBind(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<Identifier> &names) {
	if (!input.ref.bind_info) {
		throw BinderException("python_map_function requires a callable bind input");
	}
	DynamicCastCheck<MapFunctionInfo>(input.ref.bind_info.get());
	auto info = shared_ptr_cast<TableFunctionInfo, MapFunctionInfo>(input.ref.bind_info);
	nb::gil_scoped_acquire acquire;
	engine::CallbackScope callback(context);
	try {
		return BindMap(context, std::move(info), input, return_types, names);
	} catch (nb::python_error &error) {
		throw AsDuckDBError(error);
	}
}

static string TypeVectorToString(const vector<LogicalType> &types) {
	return StringUtil::Join(types, types.size(), ", ", [](const LogicalType &argument) { return argument.ToString(); });
}

static OperatorResultType ExecuteMap(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                     DataChunk &output) {
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	auto &data = data_p.bind_data->Cast<MapFunctionData>();

	D_ASSERT(input.GetTypes() == data.in_types);
	NumpyResultConversion conversion(data.in_types, input.size(), context.client.GetClientProperties());
	conversion.Append(input);

	auto df = PandasScanFunction::PandasWithBoundNames(
	    FunctionCall(conversion, data.in_names, data.info->function.obj.ptr()));

	vector<PandasColumnBindData> pandas_bind_data;
	vector<LogicalType> pandas_return_types;
	vector<Identifier> pandas_names;

	auto row_count = Pandas::Bind(context.client, df, pandas_bind_data, pandas_return_types, pandas_names);
	if (pandas_return_types.size() != output.ColumnCount()) {
		throw InvalidInputException("Expected %llu columns from UDF, got %llu", output.ColumnCount(),
		                            pandas_return_types.size());
	}
	D_ASSERT(output.GetTypes() == data.out_types);
	if (pandas_return_types != output.GetTypes()) {
		throw InvalidInputException("UDF column type mismatch, expected [%s], got [%s]",
		                            TypeVectorToString(data.out_types), TypeVectorToString(pandas_return_types));
	}
	if (pandas_names != data.out_names) {
		throw InvalidInputException("UDF column name mismatch, expected [%s], got [%s]",
		                            StringUtil::Join(data.out_names, ", "), StringUtil::Join(pandas_names, ", "));
	}

	if (row_count > STANDARD_VECTOR_SIZE) {
		throw InvalidInputException("UDF returned more than %llu rows, which is not allowed.", STANDARD_VECTOR_SIZE);
	}

	for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
		auto &bind_data = pandas_bind_data[col_idx];
		Vector temp(output.data[col_idx].GetType(), row_count);
		PandasScanFunction::PandasBackendScanSwitch(context.client, bind_data, row_count, 0, temp);
		VectorOperations::Copy(temp, output.data[col_idx], row_count, 0, 0);
	}
	output.SetChildCardinality(row_count);
	return OperatorResultType::NEED_MORE_INPUT;
}

OperatorResultType MapFunction::MapFunctionExec(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                                DataChunk &output) {
	nb::gil_scoped_acquire acquire;
	engine::CallbackScope callback(context.client);
	try {
		return ExecuteMap(context, data_p, input, output);
	} catch (nb::python_error &error) {
		throw AsDuckDBError(error);
	}
}

} // namespace duckdb

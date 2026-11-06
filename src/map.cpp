#include "duckdb_python/map.hpp"
#include "duckdb_python/engine/connection.hpp"
#include "duckdb_python/exceptions.hpp"
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
	//! Per output column, whether its type is only a guess from the typed input column of its name, which each chunk
	//! must confirm, rather than imposed by the schema or an object input
	vector<bool> guessed;
	//! Per input column, the CategoricalDtype an ENUM column is handed over in, built once since it lists every value
	vector<PyObjectHolder> enum_dtypes;
	//! The categories of those dtypes with their ENUM types, by which a returned column of them reads its codes
	vector<PyObjectHolder> enum_categories;
	vector<PandasEnumDtype> known_enums;
};

static nb::object FunctionCall(const MapFunctionData &data, NumpyResultConversion &conversion) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	nb::dict in_numpy_dict;
	for (idx_t col_idx = 0; col_idx < data.in_names.size(); col_idx++) {
		auto column = conversion.ToArray(col_idx);
		// As df() gives it: the codes alone bind back as TINYINT, a NULL as the code -1.
		if (data.in_types[col_idx].id() == LogicalTypeId::ENUM) {
			column = import_cache.pandas.Categorical().attr("from_codes")(column, nb::arg("dtype") =
			                                                                          data.enum_dtypes[col_idx].obj);
		}
		in_numpy_dict[data.in_names[col_idx].c_str()] = column;
	}
	auto function = data.info->function.obj.ptr();

	auto pandas_df = import_cache.pandas.DataFrame();
	auto in_df = pandas_df(in_numpy_dict);
	D_ASSERT(in_df.ptr());

	D_ASSERT(function);
	// Not PyObject_CallOneArg: nanobind links macOS extensions against a fixed list of CPython symbols that lacks it.
	auto *df_obj = PyObject_CallFunctionObjArgs(function, in_df.ptr(), nullptr);
	if (!df_obj) {
		// Never PyErr_Print: for SystemExit it ends the interpreter, from inside an engine task.
		nb::python_error error;
		throw InvalidInputException("Python error in the map function: %s", FormatUDFPythonError(error));
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

//! The input column of that name, the one at the same position first; the input names are deduplicated at bind
static optional_idx InputColumnNamed(const vector<Identifier> &input_names, const Identifier &name, idx_t position) {
	if (position < input_names.size() && input_names[position] == name) {
		return position;
	}
	for (idx_t col_idx = 0; col_idx < input_names.size(); col_idx++) {
		if (input_names[col_idx] == name) {
			return col_idx;
		}
	}
	return optional_idx();
}

//! An object column of the empty bind frame has no values to type it, so it takes the type of the input column of its
//! name. That type is imposed when the function got the input as an object column too; from a typed input it is only
//! a guess, since the function may have moved the column to another type, and casting back would lose values.
static void OverrideNullType(vector<LogicalType> &return_types, const vector<Identifier> &return_names,
                             const vector<LogicalType> &input_types, const vector<Identifier> &input_names,
                             vector<bool> &guessed) {
	guessed.assign(return_types.size(), false);
	for (idx_t col_idx = 0; col_idx < return_types.size(); col_idx++) {
		if (return_types[col_idx].id() != LogicalTypeId::SQLNULL) {
			continue;
		}
		auto input = InputColumnNamed(input_names, return_names[col_idx], col_idx);
		if (!input.IsValid()) {
			continue;
		}
		auto &input_type = input_types[input.GetIndex()];
		return_types[col_idx] = input_type;
		guessed[col_idx] = RawArrayWrapper::DuckDBToNumpyDtype(input_type) != "object";
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
	function_data->guessed.assign(types.size(), false);

	return std::move(function_data);
}

// we call the passed function with a zero-row data frame to infer the output columns and their names.
// they better not change in the actual execution ^^
static unique_ptr<FunctionData> BindMap(ClientContext &context, shared_ptr<MapFunctionInfo> info,
                                        TableFunctionBindInput &input, vector<LogicalType> &return_types,
                                        vector<Identifier> &names) {
	auto data_uptr = make_uniq<MapFunctionData>(std::move(info));
	auto &data = *data_uptr;
	auto explicit_schema = data.info->schema.obj.ptr();

	data.in_names = input.input_table_names;
	// The frame is keyed by name, so two input columns of one name would leave the function one of them
	QueryResult::DeduplicateColumns(data.in_names);
	data.in_types = input.input_table_types;
	for (auto &type : data.in_types) {
		if (type.id() != LogicalTypeId::ENUM) {
			data.enum_dtypes.emplace_back();
			continue;
		}
		data.enum_dtypes.emplace_back(EnumCategoricalDtype(type));
		data.enum_categories.emplace_back(data.enum_dtypes.back().obj.attr("categories"));
		data.known_enums.push_back({data.enum_categories.back().obj, type});
	}

	if (explicit_schema != Py_None) {
		return BindExplicitSchema(std::move(data_uptr), explicit_schema, return_types, names);
	}
	NumpyResultConversion conversion(data.in_types, 0, context.GetClientProperties());
	auto df = PandasScanFunction::PandasWithBoundNames(FunctionCall(data, conversion));
	vector<PandasColumnBindData> pandas_bind_data; // unused
	Pandas::Bind(context, df, pandas_bind_data, return_types, names, {}, data.known_enums);

	OverrideNullType(return_types, names, data.in_types, data.in_names, data.guessed);

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
	PyUtil::QueryGil acquire(context);
	engine::CallbackScope callback(context);
	try {
		return BindMap(context, std::move(info), input, return_types, names);
	} catch (nb::python_error &error) {
		throw AsDuckDBError(error);
	}
}

static InvalidInputException UntypedColumnError(const Identifier &name) {
	return InvalidInputException(
	    "UDF column '%s' has no type: the function returned no values for it at bind time; pass its type with schema=",
	    name.c_str());
}

static string TypeVectorToString(const vector<LogicalType> &types) {
	return StringUtil::Join(types, types.size(), ", ", [](const LogicalType &argument) { return argument.ToString(); });
}

//! A returned column the analyzer reads as all NULL binds as NULL, which holds for any type, as OverrideNullType
//! assumes at bind
static bool MatchesBoundTypes(const vector<LogicalType> &returned, const vector<LogicalType> &bound) {
	D_ASSERT(returned.size() == bound.size());
	for (idx_t col_idx = 0; col_idx < returned.size(); col_idx++) {
		if (returned[col_idx] != bound[col_idx] && returned[col_idx].id() != LogicalTypeId::SQLNULL) {
			return false;
		}
	}
	return true;
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

	auto df = PandasScanFunction::PandasWithBoundNames(FunctionCall(data, conversion));

	auto column_count = nb::len(nb::object(df.attr("columns")));
	if (column_count != output.ColumnCount()) {
		throw InvalidInputException("Expected %llu columns from UDF, got %llu", output.ColumnCount(), column_count);
	}
	// Before anything types the frame, which could fail on a value it cannot read
	for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
		if (data.out_types[col_idx].id() != LogicalTypeId::SQLNULL) {
			continue;
		}
		auto column = df.attr("__getitem__")(nb::object(df.attr("columns"))[nb::int_(col_idx)]);
		auto has_value = PyObject_IsTrue(nb::object(column.attr("notna")().attr("any")()).ptr());
		if (has_value < 0) {
			throw nb::python_error();
		}
		if (has_value) {
			throw UntypedColumnError(data.out_names[col_idx]);
		}
	}
	// An object column has no type of its own, only values, so it is read into the type the schema or an object input
	// imposed, whatever its values. A guessed type and a column bind could not type, which holds only missing values
	// here, are left to the analyzer.
	vector<LogicalType> declared_types;
	for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
		auto &type = data.out_types[col_idx];
		auto imposed = type.id() != LogicalTypeId::SQLNULL && !data.guessed[col_idx];
		declared_types.push_back(imposed ? type : LogicalType::INVALID);
	}
	vector<PandasColumnBindData> pandas_bind_data;
	vector<LogicalType> pandas_return_types;
	vector<Identifier> pandas_names;
	auto row_count = Pandas::Bind(context.client, df, pandas_bind_data, pandas_return_types, pandas_names,
	                              declared_types, data.known_enums);

	// A guess is only checked: a chunk of only missing values confirms nothing and reads NULL, any other type refutes
	// it, and the analyzer's sample decides which, as it did before for every column.
	vector<bool> missing_values(output.ColumnCount(), false);
	for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
		if (!data.guessed[col_idx] || pandas_return_types[col_idx] == data.out_types[col_idx]) {
			continue;
		}
		if (pandas_return_types[col_idx].id() != LogicalTypeId::SQLNULL) {
			throw UntypedColumnError(data.out_names[col_idx]);
		}
		pandas_return_types[col_idx] = data.out_types[col_idx];
		missing_values[col_idx] = true;
	}

	D_ASSERT(output.GetTypes() == data.out_types);
	if (!MatchesBoundTypes(pandas_return_types, output.GetTypes())) {
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
		// A guessed column whose chunk holds only missing values, and a column bind left untyped, the only one still
		// typed NULL here; the bound type's scanner would read some missing values as data, such as NaT as 'NaT'.
		if (missing_values[col_idx] || pandas_return_types[col_idx].id() == LogicalTypeId::SQLNULL) {
			ConstantVector::SetNull(output.data[col_idx], count_t(row_count));
			continue;
		}
		auto &bind_data = pandas_bind_data[col_idx];
		Vector temp(output.data[col_idx].GetType(), row_count);
		try {
			PandasScanFunction::PandasBackendScanSwitch(context.client, bind_data, row_count, 0, temp);
		} catch (std::exception &ex) {
			ErrorData error(ex);
			// The cast's own message names no column; a Python error keeps the traceback its message starts with
			if (!bind_data.declared || error.Type() == ExceptionType::INVALID ||
			    error.Type() == ExceptionType::INTERRUPT) {
				throw;
			}
			error.Throw(StringUtil::Format("UDF column '%s': ", data.out_names[col_idx].c_str()));
		}
		VectorOperations::Copy(temp, output.data[col_idx], row_count, 0, 0);
	}
	output.SetChildCardinality(row_count);
	return OperatorResultType::NEED_MORE_INPUT;
}

OperatorResultType MapFunction::MapFunctionExec(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                                DataChunk &output) {
	PyUtil::QueryGil acquire(context.client);
	engine::CallbackScope callback(context.client);
	try {
		return ExecuteMap(context, data_p, input, output);
	} catch (nb::python_error &error) {
		throw AsDuckDBError(error);
	}
}

} // namespace duckdb

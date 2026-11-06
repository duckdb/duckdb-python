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
#include "duckdb/common/type_visitor.hpp"
#include "duckdb_python/python_conversion.hpp"

namespace duckdb {

MapFunction::MapFunction() : TableFunction("python_map_function", {LogicalType::TABLE}, nullptr, MapFunctionBind) {
	in_out_function = MapFunctionExec;
}

//! Where an output column's type came from, which decides how a chunk's column is read into it
enum class TypeSource : uint8_t {
	//! The schema stated it, so an object column's values are cast into it whatever they hold
	SCHEMA,
	//! The empty bind frame typed the column, or the object input column of its name lent its type: an object
	//! column's cells are read into it only when they are of its kind, since a function that changed what they hold
	//! would otherwise see them cast back
	INFERRED,
	//! The input column of its name lent its type, but the function got that input typed, or as the member values of
	//! a UNION, whose tags cannot travel: an object column is read into it only from cells of its kind, as an
	//! inferred one is, and a typed column must have the type, or one that widens into it losslessly
	GUESSED,
	//! Nothing typed the column: it may hold only missing values
	UNTYPED
};

struct MapFunctionData : public TableFunctionData {
	explicit MapFunctionData(shared_ptr<MapFunctionInfo> info_p) : info(std::move(info_p)) {
	}
	shared_ptr<MapFunctionInfo> info;
	vector<LogicalType> in_types, out_types;
	vector<Identifier> in_names, out_names;
	vector<TypeSource> sources;
	//! Per output column, the type a returned object column is read into, INVALID where the analyzer types it
	vector<LogicalType> declared_types;
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
//! name: inferred when the function got that input as an object column too, a guess otherwise (TypeSource).
static void OverrideNullType(vector<LogicalType> &return_types, const vector<Identifier> &return_names,
                             const vector<LogicalType> &input_types, const vector<Identifier> &input_names,
                             vector<TypeSource> &sources) {
	sources.clear();
	for (idx_t col_idx = 0; col_idx < return_types.size(); col_idx++) {
		if (return_types[col_idx].id() != LogicalTypeId::SQLNULL) {
			sources.push_back(TypeSource::INFERRED);
			continue;
		}
		auto input = InputColumnNamed(input_names, return_names[col_idx], col_idx);
		if (!input.IsValid()) {
			sources.push_back(TypeSource::UNTYPED);
			continue;
		}
		auto &input_type = input_types[input.GetIndex()];
		return_types[col_idx] = input_type;
		auto object_input = RawArrayWrapper::DuckDBToNumpyDtype(input_type) == "object";
		auto lends_type = object_input && !TypeVisitor::Contains(input_type, LogicalTypeId::UNION);
		sources.push_back(lends_type ? TypeSource::INFERRED : TypeSource::GUESSED);
	}
}

//! Every object column is declared, so the analyzer never reads one and no sample decides how a cell reads; an untyped
//! column is declared NULL, its check that it holds no value comes after the frame's own checks
static void DeclareTypes(MapFunctionData &data) {
	data.declared_types.clear();
	for (idx_t col_idx = 0; col_idx < data.out_types.size(); col_idx++) {
		switch (data.sources[col_idx]) {
		case TypeSource::SCHEMA:
		case TypeSource::INFERRED:
		case TypeSource::GUESSED:
			data.declared_types.push_back(data.out_types[col_idx]);
			break;
		case TypeSource::UNTYPED:
			data.declared_types.push_back(LogicalType::SQLNULL);
			break;
		}
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
	function_data->sources.assign(types.size(), TypeSource::SCHEMA);
	DeclareTypes(*function_data);

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

	OverrideNullType(return_types, names, data.in_types, data.in_names, data.sources);

	data.out_names = names;
	data.out_types = return_types;
	DeclareTypes(data);
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

static InvalidInputException TypeMismatch(const MapFunctionData &data, const vector<LogicalType> &returned) {
	return InvalidInputException("UDF column type mismatch, expected [%s], got [%s]",
	                             TypeVectorToString(data.out_types), TypeVectorToString(returned));
}

static bool IsTrue(const nb::object &object) {
	auto truth = PyObject_IsTrue(object.ptr());
	if (truth < 0) {
		throw nb::python_error();
	}
	return truth;
}

//! Whether a column bind could not type holds a value: pandas' missing values, numpy's masked constant and DuckDB's
//! NULL Values are none, as the conversion reads them all as NULL
static bool HoldsValue(const nb::object &column) {
	auto present = column.attr("notna")();
	if (!IsTrue(present.attr("any")())) {
		return false;
	}
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto cells = nb::list(column.attr("__getitem__")(present).attr("tolist")());
	for (auto cell : cells) {
		if (cell.is(import_cache.numpy.ma.masked())) {
			continue;
		}
		if (!duckdb::PyUtil::IsInstance(cell, import_cache.duckdb.Value()) ||
		    !TransformPythonValue(nullptr, cell).IsNull()) {
			return true;
		}
	}
	return false;
}

static bool IntegerRange(const LogicalType &type, bool &is_signed, idx_t &width) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
		is_signed = true;
		break;
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::UHUGEINT:
		is_signed = false;
		break;
	default:
		return false;
	}
	width = GetTypeIdSize(type.InternalType());
	return true;
}

//! Whether every value of one type is a value of the other: an integer into a wider one, a FLOAT into a DOUBLE
static bool WidensLosslessly(const LogicalType &from, const LogicalType &to) {
	if (from.id() == LogicalTypeId::FLOAT && to.id() == LogicalTypeId::DOUBLE) {
		return true;
	}
	bool from_signed, to_signed;
	idx_t from_width, to_width;
	if (!IntegerRange(from, from_signed, from_width) || !IntegerRange(to, to_signed, to_width)) {
		return false;
	}
	if (from_signed) {
		return to_signed && to_width >= from_width;
	}
	return to_signed ? to_width > from_width : to_width >= from_width;
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

	auto labels = nb::list(nb::object(df.attr("columns")));
	if (labels.size() != output.ColumnCount()) {
		throw InvalidInputException("Expected %llu columns from UDF, got %llu", output.ColumnCount(), labels.size());
	}
	vector<PandasColumnBindData> pandas_bind_data;
	vector<LogicalType> pandas_return_types;
	vector<Identifier> pandas_names;
	auto row_count = Pandas::Bind(context.client, df, pandas_bind_data, pandas_return_types, pandas_names,
	                              data.declared_types, data.known_enums);
	for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
		pandas_bind_data[col_idx].implied_type =
		    data.sources[col_idx] == TypeSource::INFERRED || data.sources[col_idx] == TypeSource::GUESSED;
		pandas_bind_data[col_idx].bit_storage_refused = true;
	}

	// A typed column for a guess must have the guessed type, or one that widens losslessly into it and is cast after
	// the read; an object column was declared the guess and reads only cells of its kind
	vector<bool> widened(output.ColumnCount(), false);
	auto checked_types = pandas_return_types;
	for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
		auto &returned = pandas_return_types[col_idx];
		auto &bound = data.out_types[col_idx];
		if (data.sources[col_idx] != TypeSource::GUESSED || returned == bound || !WidensLosslessly(returned, bound)) {
			continue;
		}
		widened[col_idx] = true;
		checked_types[col_idx] = bound;
	}

	D_ASSERT(output.GetTypes() == data.out_types);
	if (checked_types != output.GetTypes()) {
		throw TypeMismatch(data, pandas_return_types);
	}
	if (pandas_names != data.out_names) {
		throw InvalidInputException("UDF column name mismatch, expected [%s], got [%s]",
		                            StringUtil::Join(data.out_names, ", "), StringUtil::Join(pandas_names, ", "));
	}

	if (row_count > STANDARD_VECTOR_SIZE) {
		throw InvalidInputException("UDF returned more than %llu rows, which is not allowed.", STANDARD_VECTOR_SIZE);
	}
	for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
		if (data.sources[col_idx] == TypeSource::UNTYPED && HoldsValue(df.attr("__getitem__")(labels[col_idx]))) {
			throw UntypedColumnError(data.out_names[col_idx]);
		}
	}

	for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
		if (pandas_return_types[col_idx].id() == LogicalTypeId::SQLNULL) {
			ConstantVector::SetNull(output.data[col_idx], count_t(row_count));
			continue;
		}
		auto &bind_data = pandas_bind_data[col_idx];
		Vector temp(pandas_return_types[col_idx], row_count);
		try {
			PandasScanFunction::PandasBackendScanSwitch(context.client, bind_data, row_count, 0, temp);
		} catch (ImpliedTypeRefused &refused) {
			auto returned = pandas_return_types;
			returned[col_idx] = refused.source;
			throw TypeMismatch(data, returned);
		} catch (std::exception &ex) {
			ErrorData error(ex);
			// The cast's own message names no column; a Python error keeps the traceback its message starts with
			if (!bind_data.declared || error.Type() == ExceptionType::INVALID ||
			    error.Type() == ExceptionType::INTERRUPT) {
				throw;
			}
			error.Throw(StringUtil::Format("UDF column '%s': ", data.out_names[col_idx].c_str()));
		}
		if (widened[col_idx]) {
			VectorOperations::Cast(context.client, temp, output.data[col_idx], row_count);
		} else {
			VectorOperations::Copy(temp, output.data[col_idx], row_count, 0, 0);
		}
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

#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/python_conversion.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/pybind11/python_object_container.hpp"

namespace duckdb {

struct PyTableUDFInfo : public TableFunctionInfo {
	py::function callable;
	vector<LogicalType> return_types;
	vector<string> return_names;
	PythonTableUDFType return_type;

	PyTableUDFInfo(py::function callable_p, vector<LogicalType> types_p, vector<string> names_p,
	               PythonTableUDFType return_type_p)
	    : callable(std::move(callable_p)), return_types(std::move(types_p)), return_names(std::move(names_p)),
	      return_type(return_type_p) {
	}

	~PyTableUDFInfo() override {
		py::gil_scoped_acquire acquire;
		callable = py::function();
	}
};

struct PyTableUDFBindData : public TableFunctionData {
	string func_name;
	vector<Value> args;
	named_parameter_map_t kwargs;
	vector<LogicalType> return_types;
	vector<string> return_names;
	PythonObjectContainer python_objects; // Holds the callable

	PyTableUDFBindData(string func_name, vector<Value> args, named_parameter_map_t kwargs,
	                   vector<LogicalType> return_types, vector<string> return_names, py::function callable)
	    : func_name(std::move(func_name)), args(std::move(args)), kwargs(std::move(kwargs)),
	      return_types(std::move(return_types)), return_names(std::move(return_names)) {
		// gil acquired inside push
		python_objects.Push(std::move(callable));
	}
};

struct PyTableUDFTuplesGlobalState : public GlobalTableFunctionState {
	PythonObjectContainer python_objects;
	bool iterator_exhausted = false;

	PyTableUDFTuplesGlobalState() : iterator_exhausted(false) {
	}
};

struct PyTableUDFArrowGlobalState : public GlobalTableFunctionState {
	unique_ptr<PythonTableArrowArrayStreamFactory> arrow_factory;
	unique_ptr<FunctionData> arrow_bind_data;
	unique_ptr<GlobalTableFunctionState> arrow_global_state;
	PythonObjectContainer python_objects;
	idx_t num_columns;

	PyTableUDFArrowGlobalState() {
	}
};

static void PyTableUDFTuplesScanFunction(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &gs = input.global_state->Cast<PyTableUDFTuplesGlobalState>();
	auto &bd = input.bind_data->Cast<PyTableUDFBindData>();

	if (gs.iterator_exhausted) {
		output.SetCardinality(0);
		return;
	}

	py::gil_scoped_acquire gil;
	auto &it = gs.python_objects.LastAddedObject();

	idx_t row_idx = 0;
	for (idx_t i = 0; i < STANDARD_VECTOR_SIZE; i++) {
		py::object next_item;
		try {
			next_item = it.attr("__next__")();
		} catch (py::error_already_set &e) {
			if (e.matches(PyExc_StopIteration)) {
				gs.iterator_exhausted = true;
				PyErr_Clear();
				break;
			}
			throw;
		}

		try {
			// Extract each column from the tuple/list
			for (idx_t col_idx = 0; col_idx < bd.return_types.size(); col_idx++) {
				auto py_val = next_item[py::int_(col_idx)];
				Value duck_val = TransformPythonValue(py_val, bd.return_types[col_idx]);
				output.SetValue(col_idx, row_idx, duck_val);
			}
		} catch (py::error_already_set &e) {
			throw InvalidInputException("Table function '%s' returned invalid data: %s", bd.func_name, e.what());
		}
		row_idx++;
	}
	output.SetCardinality(row_idx);
}

struct PyTableUDFArrowLocalState : public LocalTableFunctionState {
	unique_ptr<LocalTableFunctionState> arrow_local_state;

	explicit PyTableUDFArrowLocalState(unique_ptr<LocalTableFunctionState> arrow_local)
	    : arrow_local_state(std::move(arrow_local)) {
	}
};

static void PyTableUDFArrowScanFunction(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	// Delegates to ArrowScanFunction
	auto &gs = input.global_state->Cast<PyTableUDFArrowGlobalState>();
	auto &ls = input.local_state->Cast<PyTableUDFArrowLocalState>();

	TableFunctionInput arrow_input(gs.arrow_bind_data.get(), ls.arrow_local_state.get(), gs.arrow_global_state.get());
	ArrowTableFunction::ArrowScanFunction(context, arrow_input, output);
}

static unique_ptr<PyTableUDFBindData> PyTableUDFBindInternal(ClientContext &context, TableFunctionBindInput &in,
                                                             vector<LogicalType> &return_types,
                                                             vector<string> &return_names) {
	if (!in.info) {
		throw InvalidInputException("Table function '%s' missing function info", in.table_function.name);
	}

	auto &tableudf_info = in.info->Cast<PyTableUDFInfo>();
	return_types = tableudf_info.return_types;
	return_names = tableudf_info.return_names;

	// Acquire gil before copying py::function
	py::gil_scoped_acquire gil;
	return make_uniq<PyTableUDFBindData>(in.table_function.name, in.inputs, in.named_parameters, return_types,
	                                     return_names, tableudf_info.callable);
}

static unique_ptr<FunctionData> PyTableUDFTuplesBindFunction(ClientContext &context, TableFunctionBindInput &in,
                                                             vector<LogicalType> &return_types,
                                                             vector<string> &return_names) {
	auto bd = PyTableUDFBindInternal(context, in, return_types, return_names);
	return std::move(bd);
}

static unique_ptr<FunctionData> PyTableUDFArrowBindFunction(ClientContext &context, TableFunctionBindInput &in,
                                                            vector<LogicalType> &return_types,
                                                            vector<string> &return_names) {
	auto bd = PyTableUDFBindInternal(context, in, return_types, return_names);
	return std::move(bd);
}

static py::object CallPythonTableUDF(ClientContext &context, PyTableUDFBindData &bd) {
	py::gil_scoped_acquire gil;

	// positional arguments
	py::tuple args(bd.args.size());
	for (idx_t i = 0; i < bd.args.size(); i++) {
		args[i] = PythonObject::FromValue(bd.args[i], bd.args[i].type(), context.GetClientProperties());
	}

	// keyword arguments
	py::dict kwargs;
	for (auto &kv : bd.kwargs) {
		kwargs[py::str(kv.first)] = PythonObject::FromValue(kv.second, kv.second.type(), context.GetClientProperties());
	}

	// Call Python function
	auto &callable = bd.python_objects.LastAddedObject();
	py::object result = callable(*args, **kwargs);

	if (result.is_none()) {
		throw InvalidInputException("Table function '%s' returned None, expected iterable or Arrow table",
		                            bd.func_name);
	}

	return result;
}

static unique_ptr<GlobalTableFunctionState> PyTableUDFTuplesInitGlobal(ClientContext &context,
                                                                       TableFunctionInitInput &in) {
	auto &bd = in.bind_data->Cast<PyTableUDFBindData>();
	auto gs = make_uniq<PyTableUDFTuplesGlobalState>();

	{
		py::gil_scoped_acquire gil;
		// const_cast is safe here - we only read from python_objects, not modify bind_data structure
		py::object result = CallPythonTableUDF(context, const_cast<PyTableUDFBindData &>(bd));
		try {
			py::iterator it = py::iter(result);
			gs->python_objects.Push(std::move(it));
			gs->iterator_exhausted = false;
		} catch (const py::error_already_set &e) {
			throw InvalidInputException("Table function '%s' returned non-iterable result: %s", bd.func_name, e.what());
		}
	}

	return std::move(gs);
}

static unique_ptr<GlobalTableFunctionState> PyTableUDFArrowInitGlobal(ClientContext &context,
                                                                      TableFunctionInitInput &in) {
	auto &bd = in.bind_data->Cast<PyTableUDFBindData>();
	auto gs = make_uniq<PyTableUDFArrowGlobalState>();

	{
		py::gil_scoped_acquire gil;

		py::object result = CallPythonTableUDF(context, const_cast<PyTableUDFBindData &>(bd));
		PyObject *ptr = result.ptr();

		gs->python_objects.Push(std::move(result));

		gs->arrow_factory = make_uniq<PythonTableArrowArrayStreamFactory>(ptr, context.GetClientProperties(),
		                                                                  DBConfig::GetConfig(context));
	}

	// Build bind input for Arrow scan
	vector<Value> children;
	children.push_back(Value::POINTER(CastPointerToValue(gs->arrow_factory.get())));
	children.push_back(Value::POINTER(CastPointerToValue(PythonTableArrowArrayStreamFactory::Produce)));
	children.push_back(Value::POINTER(CastPointerToValue(PythonTableArrowArrayStreamFactory::GetSchema)));

	TableFunctionRef empty_ref;
	duckdb::TableFunction dummy_tf;
	dummy_tf.name = "PyTableUDFArrowWrapper";

	named_parameter_map_t named_params;
	vector<LogicalType> input_types;
	vector<string> input_names;

	TableFunctionBindInput bind_input(children, named_params, input_types, input_names, nullptr, nullptr, dummy_tf,
	                                  empty_ref);

	vector<LogicalType> return_types;
	vector<string> return_names;
	gs->arrow_bind_data = ArrowTableFunction::ArrowScanBind(context, bind_input, return_types, return_names);

	// Validate Arrow schema matches declared
	if (return_types.size() != bd.return_types.size()) {
		throw InvalidInputException("Schema mismatch in table function '%s': "
		                            "Arrow table has %lu columns but %lu were declared",
		                            bd.func_name, return_types.size(), bd.return_types.size());
	}

	// Check column types match
	for (idx_t i = 0; i < return_types.size(); i++) {
		if (return_types[i] != bd.return_types[i]) {
			throw InvalidInputException("Schema mismatch in table function '%s' at column %lu: "
			                            "Arrow table has type %s but %s was declared",
			                            bd.func_name, i, return_types[i].ToString().c_str(),
			                            bd.return_types[i].ToString().c_str());
		}
	}

	gs->num_columns = return_types.size();
	vector<column_t> all_columns;
	for (idx_t i = 0; i < gs->num_columns; i++) {
		all_columns.push_back(i);
	}

	TableFunctionInitInput init_input(gs->arrow_bind_data.get(), all_columns, all_columns, in.filters.get());
	gs->arrow_global_state = ArrowTableFunction::ArrowScanInitGlobal(context, init_input);

	return std::move(gs);
}

static unique_ptr<LocalTableFunctionState>
PyTableUDFArrowInitLocal(ExecutionContext &context, TableFunctionInitInput &in, GlobalTableFunctionState *gstate) {
	auto &gs = gstate->Cast<PyTableUDFArrowGlobalState>();

	vector<column_t> all_columns;
	for (idx_t i = 0; i < gs.num_columns; i++) {
		all_columns.push_back(i);
	}

	TableFunctionInitInput arrow_init(gs.arrow_bind_data.get(), all_columns, all_columns, in.filters.get());
	auto arrow_local_state =
	    ArrowTableFunction::ArrowScanInitLocalInternal(context.client, arrow_init, gs.arrow_global_state.get());

	return make_uniq<PyTableUDFArrowLocalState>(std::move(arrow_local_state));
}

duckdb::TableFunction DuckDBPyConnection::CreateTableFunctionFromCallable(const std::string &name,
                                                                          const py::function &callable,
                                                                          const py::object &parameters,
                                                                          const py::object &schema,
                                                                          PythonTableUDFType type) {

	// Schema
	if (schema.is_none()) {
		throw InvalidInputException("Table functions require a schema.");
	}

	vector<LogicalType> types;
	vector<string> names;

	// Schema must be dict format: {"col1": DuckDBPyType, "col2": DuckDBPyType}
	if (!py::isinstance<py::dict>(schema)) {
		throw InvalidInputException("Table function '%s' schema must be a dict mapping column names to duckdb.sqltypes "
		                            "(e.g., {\"col1\": INTEGER, \"col2\": VARCHAR})",
		                            name);
	}

	auto schema_dict = py::cast<py::dict>(schema);
	for (auto item : schema_dict) {
		// schema is a dict of str => DuckDBPyType

		string col_name = py::str(item.first);
		names.emplace_back(col_name);

		auto type_obj = py::cast<py::object>(item.second);

		// Check for string BEFORE DuckDBPyType because pybind11 has implicit conversion from str to DuckDBPyType
		if (py::isinstance<py::str>(type_obj)) {
			throw InvalidInputException("Invalid schema format: type for column '%s' must be a duckdb.sqltype (e.g., "
			                            "INTEGER, VARCHAR), not a string. "
			                            "Use sqltypes.%s instead of \"%s\"",
			                            col_name, py::str(type_obj).cast<std::string>().c_str(),
			                            py::str(type_obj).cast<std::string>().c_str());
		}

		if (!py::isinstance<DuckDBPyType>(type_obj)) {
			throw InvalidInputException(
			    "Invalid schema format: type for column '%s' must be a duckdb.sqltype (e.g., INTEGER, VARCHAR), got %s",
			    col_name, py::str(type_obj.get_type()).cast<std::string>());
		}
		auto pytype = py::cast<shared_ptr<DuckDBPyType>>(type_obj);
		types.emplace_back(pytype->Type());
	}

	if (types.empty()) {
		throw InvalidInputException("Table function '%s' schema cannot be empty", name);
	}

	duckdb::TableFunction tf;
	switch (type) {
	case PythonTableUDFType::TUPLES:
		tf = duckdb::TableFunction(name, {}, PyTableUDFTuplesScanFunction, PyTableUDFTuplesBindFunction,
		                           PyTableUDFTuplesInitGlobal);
		break;
	case PythonTableUDFType::ARROW_TABLE:
		tf = duckdb::TableFunction(name, {}, PyTableUDFArrowScanFunction, PyTableUDFArrowBindFunction,
		                           PyTableUDFArrowInitGlobal, PyTableUDFArrowInitLocal);
		break;
	default:
		throw InvalidInputException("Unknown return type for table function '%s'", name);
	}

	// Store the Python callable and schema
	tf.function_info = make_shared_ptr<PyTableUDFInfo>(callable, types, names, type);

	// args
	tf.varargs = LogicalType::ANY;
	tf.named_parameters["args"] = LogicalType::ANY;

	// kwargs
	if (!parameters.is_none()) {
		for (auto &param : py::cast<py::list>(parameters)) {
			string param_name = py::str(param);
			tf.named_parameters[param_name] = LogicalType::ANY;
		}
	}

	return tf;
}

} // namespace duckdb

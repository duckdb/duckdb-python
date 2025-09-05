#include "duckdb/main/query_result.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pandas/pandas_scan.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/arrow/arrow_appender.hpp"
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"
#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb_python/numpy/numpy_scan.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb/common/types/arrow_aux_data.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "duckdb_python/python_conversion.hpp"

namespace duckdb {

// Global registry for TVF connection lookup
case_insensitive_map_t<weak_ptr<DuckDBPyConnection>> &GetTVFConnectionRegistry() {
	static case_insensitive_map_t<weak_ptr<DuckDBPyConnection>> registry;
	return registry;
}

// Forward declarations for table function return types
enum class PyTVFReturnType {
	RECORDS,     // Python tuples/lists converted to DuckDB records with proper types
	ARROW_TABLE, // Arrow tables with proper schema and types
	ARROW_BATCH  // Arrow record batches
};

struct PyTVFGlobalState : public GlobalTableFunctionState {
	vector<vector<Value>> rows;
	idx_t idx = 0;

	// For Arrow support
	unique_ptr<ArrowArrayWrapper> arrow_array;
	unique_ptr<ArrowSchemaWrapper> arrow_schema;
	idx_t arrow_batch_idx = 0;
	PyTVFReturnType return_type;
};

// Helper function to convert DuckDB Values to Python objects
static py::object ValueToPy(const Value &v) {
	if (v.IsNull())
		return py::none();
	switch (v.type().id()) {
	case LogicalTypeId::BOOLEAN:
		return py::bool_(v.GetValue<bool>());
	case LogicalTypeId::TINYINT:
		return py::int_(v.GetValue<int8_t>());
	case LogicalTypeId::SMALLINT:
		return py::int_(v.GetValue<int16_t>());
	case LogicalTypeId::INTEGER:
		return py::int_(v.GetValue<int32_t>());
	case LogicalTypeId::BIGINT:
		return py::int_(v.GetValue<int64_t>());
	case LogicalTypeId::UINTEGER:
		return py::int_(v.GetValue<uint32_t>());
	case LogicalTypeId::UBIGINT:
		return py::int_(v.GetValue<uint64_t>());
	case LogicalTypeId::DOUBLE:
		return py::float_(v.GetValue<double>());
	case LogicalTypeId::FLOAT:
		return py::float_(v.GetValue<float>());
	case LogicalTypeId::VARCHAR:
		return py::str(v.GetValue<string>());
	case LogicalTypeId::DECIMAL:
		return py::str(v.ToString());
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIME:
	default:
		return py::str(v.ToString());
	}
}

struct PyTVFBindData : public TableFunctionData {
	string func_name;
	vector<Value> args;
	named_parameter_map_t kwargs;
	vector<LogicalType> return_types;
	vector<string> return_names;
	PyTVFReturnType return_type;
	weak_ptr<DuckDBPyConnection> connection;

	PyTVFBindData(string func_name, vector<Value> args, named_parameter_map_t kwargs, vector<LogicalType> return_types,
	              vector<string> return_names, PyTVFReturnType return_type, weak_ptr<DuckDBPyConnection> connection)
	    : func_name(std::move(func_name)), args(std::move(args)), kwargs(std::move(kwargs)),
	      return_types(std::move(return_types)), return_names(std::move(return_names)), return_type(return_type),
	      connection(connection) {
	}
};

duckdb::TableFunction DuckDBPyConnection::CreateTableFunctionFromCallable(const std::string &name,
                                                                          const py::function &callable,
                                                                          const py::object &parameters,
                                                                          const py::object &schema,
                                                                          const std::string &return_type_str) {
	// Parse schema immediately
	vector<LogicalType> types;
	vector<string> names;
	if (!schema.is_none()) {
		for (auto &c : py::cast<py::list>(schema)) {
			py::tuple tup = c.cast<py::tuple>();
			names.emplace_back(py::str(tup[0]));
			types.emplace_back(TransformStringToLogicalType(py::str(tup[1])));
		}
	}

	// Default schema if none provided
	if (types.empty()) {
		types.push_back(LogicalType::VARCHAR);
		names.push_back("result");
	}

	// Store callable and info in per-connection storage
	TVFInfo tvf_info;
	tvf_info.callable = callable;
	tvf_info.return_types = types;
	tvf_info.return_names = names;
	tvf_info.return_type_str = return_type_str;
	table_function_callables[name] = std::move(tvf_info);

	// Create connection weak pointer for bind data
	weak_ptr<DuckDBPyConnection> weak_conn = shared_from_this();

	// Create TableFunction using the working pattern
	vector<LogicalType> function_args; // No arguments for now
	duckdb::TableFunction tf(
	    name, function_args,
	    /* main execution function */ [](ClientContext &, TableFunctionInput &input, DataChunk &output) {
		    auto &gs = input.global_state->Cast<PyTVFGlobalState>();

		    // Handle record-based data (Arrow support to be added later)
		    if (gs.idx >= gs.rows.size()) {
			    output.SetCardinality(0);
			    return;
		    }
		    auto to_emit = MinValue<idx_t>(STANDARD_VECTOR_SIZE, gs.rows.size() - gs.idx);
		    output.SetCardinality(to_emit);

		    for (idx_t i = 0; i < to_emit; i++) {
			    for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
				    if (col_idx < gs.rows[gs.idx + i].size()) {
					    output.SetValue(col_idx, i, gs.rows[gs.idx + i][col_idx]);
				    }
			    }
		    }
		    gs.idx += to_emit;
	    });

	// Set up varargs and named parameters
	tf.varargs = LogicalType::ANY;
	tf.named_parameters["args"] = LogicalType::ANY;

	// Store connection in global registry for lookup by function name
	GetTVFConnectionRegistry()[name] = weak_conn;

	tf.bind = +[](ClientContext &context, TableFunctionBindInput &in, vector<LogicalType> &return_types,
	              vector<string> &return_names) -> unique_ptr<FunctionData> {
		// Get connection from global registry
		auto &connection_registry = GetTVFConnectionRegistry();
		auto conn_it = connection_registry.find(in.table_function.name);
		if (conn_it == connection_registry.end()) {
			throw InvalidInputException("Connection not found for table function '%s'", in.table_function.name);
		}
		auto conn = conn_it->second.lock();
		if (!conn) {
			throw InvalidInputException("Connection no longer available for table function '%s'",
			                            in.table_function.name);
		}

		auto it = conn->table_function_callables.find(in.table_function.name);
		if (it == conn->table_function_callables.end()) {
			throw InvalidInputException("Table function '%s' not found", in.table_function.name);
		}

		const auto &tvf_info = it->second;
		return_types = tvf_info.return_types;
		return_names = tvf_info.return_names;

		// Parse return type with error handling
		PyTVFReturnType ret_type = PyTVFReturnType::RECORDS;
		if (tvf_info.return_type_str == "records") {
			// Returns a list of iterables
			ret_type = PyTVFReturnType::RECORDS;
		} else if (tvf_info.return_type_str == "arrow_table" || tvf_info.return_type_str == "arrow") {
			ret_type = PyTVFReturnType::ARROW_TABLE;
		} else if (tvf_info.return_type_str == "arrow_batch") {
			ret_type = PyTVFReturnType::ARROW_BATCH;
		} else {
			throw InvalidInputException("Unknown return type '%s' for table function '%s'. Valid types are: 'records', "
			                            "'strings', 'arrow_table', 'arrow_batch'",
			                            tvf_info.return_type_str, in.table_function.name);
		}

		// Create bind data
		auto bd = make_uniq<PyTVFBindData>(in.table_function.name, in.inputs, in.named_parameters, return_types,
		                                   return_names, ret_type, conn_it->second);

		return unique_ptr<FunctionData>(static_cast<FunctionData *>(bd.release()));
	};

	tf.init_global = +[](ClientContext &context, TableFunctionInitInput &in) -> unique_ptr<GlobalTableFunctionState> {
		auto &bd = in.bind_data->Cast<PyTVFBindData>();
		auto gs = make_uniq<PyTVFGlobalState>();
		gs->return_type = bd.return_type;

		py::gil_scoped_acquire gil;

		// Get connection and look up callable
		auto conn = bd.connection.lock();
		if (!conn) {
			throw InvalidInputException("Connection no longer available for table function '%s'", bd.func_name);
		}

		auto it = conn->table_function_callables.find(bd.func_name);
		if (it == conn->table_function_callables.end()) {
			throw InvalidInputException("Table function '%s' not found", bd.func_name);
		}

		const py::function &fn = it->second.callable;

		// Build arguments with improved type conversion
		py::tuple args(bd.args.size());
		for (idx_t i = 0; i < bd.args.size(); i++) {
			const auto &val = bd.args[i];
			// Special handling for DECIMAL values to convert them to Python float instead of string
			if (val.type().id() == LogicalTypeId::DECIMAL && !val.IsNull()) {
				try {
					double d = val.GetValue<double>();
					args[i] = py::float_(d);
				} catch (...) {
					// Fallback to string if conversion fails
					args[i] = py::str(val.ToString());
				}
			} else {
				args[i] = ValueToPy(val);
			}
		}

		// Build keyword arguments with improved type conversion
		py::dict kwargs;
		for (auto &kv : bd.kwargs) {
			const auto &val = kv.second;
			// Special handling for DECIMAL values
			if (val.type().id() == LogicalTypeId::DECIMAL && !val.IsNull()) {
				try {
					double d = val.GetValue<double>();
					kwargs[py::str(kv.first)] = py::float_(d);
				} catch (...) {
					kwargs[py::str(kv.first)] = py::str(val.ToString());
				}
			} else {
				kwargs[py::str(kv.first)] = ValueToPy(val);
			}
		}

		// Call Python function
		py::object result = fn(*args, **kwargs);

		// Process result based on return type
		if (bd.return_type == PyTVFReturnType::RECORDS) {
			if (py::isinstance<py::list>(result)) {
				for (auto item : result.cast<py::list>()) {
					if (py::isinstance<py::tuple>(item)) {
						auto tup = item.cast<py::tuple>();
						if (tup.size() != bd.return_types.size()) {
							throw InvalidInputException(
							    "Table function '%s' returned tuple of size %d but schema expects %d columns",
							    bd.func_name, static_cast<int>(tup.size()), static_cast<int>(bd.return_types.size()));
						}
						vector<Value> row;
						for (idx_t i = 0; i < tup.size(); i++) {
							// Convert Python value to DuckDB Value with proper type
							auto py_val = tup[i];
							Value duck_val = TransformPythonValue(py_val, bd.return_types[i]);
							row.push_back(duck_val);
						}
						gs->rows.push_back(std::move(row));
					} else {
						if (bd.return_types.size() != 1) {
							throw InvalidInputException(
							    "Table function '%s' returned a single value but schema expects %d columns",
							    bd.func_name, static_cast<int>(bd.return_types.size()));
						}
						Value duck_val = TransformPythonValue(item, bd.return_types[0]);
						gs->rows.push_back({duck_val});
					}
				}
			}
		} else if (bd.return_type == PyTVFReturnType::ARROW_TABLE || bd.return_type == PyTVFReturnType::ARROW_BATCH) {
			// Handle Arrow objects - for now, convert to regular rows
			// TODO: Implement proper Arrow support
			throw InvalidInputException("Arrow return types not yet implemented for table function '%s'", bd.func_name);
		}

		return unique_ptr<GlobalTableFunctionState>(static_cast<GlobalTableFunctionState *>(gs.release()));
	};

	return tf;
}

} // namespace duckdb

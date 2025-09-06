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
#include <thread>
#include <chrono>
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

	// For large datasets - batch processing  
	vector<vector<vector<Value>>> batches;
	idx_t current_batch_idx = 0;
	idx_t current_row_in_batch = 0;
	bool is_batched = false;
	
	// For generator/iterator support - streaming processing
	py::object py_generator_object;  // Keep the generator alive
	py::iterator py_iterator;
	bool has_iterator = false;
	bool iterator_exhausted = false;

	// For Arrow support
	unique_ptr<ArrowArrayStreamWrapper> arrow_stream;
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
	    /* main execution function */ [](ClientContext &context, TableFunctionInput &input, DataChunk &output) {
		    auto &gs = input.global_state->Cast<PyTVFGlobalState>();

		    // Handle generator/iterator streaming
		    if (gs.has_iterator) {
		        if (gs.iterator_exhausted) {
		            output.SetCardinality(0);
		            return;
		        }
		        
		        // Process generator items one chunk at a time
		        idx_t items_processed = 0;
		        const idx_t max_items = STANDARD_VECTOR_SIZE;
		        
		        // Get bind data to access schema info
		        auto &bind_data = input.bind_data->Cast<PyTVFBindData>();
		        
		        while (items_processed < max_items) {
		            try {
		                // GIL release every 50 items for notebook responsiveness  
		                if (items_processed > 0 && items_processed % 50 == 0) {
		                    py::gil_scoped_release release;
		                }
		                
		                py::handle item = *gs.py_iterator;
		                ++gs.py_iterator;
		                
		                // Process the item (similar to existing tuple processing logic)
		                vector<Value> row_values;
		                if (py::isinstance<py::tuple>(item)) {
		                    auto tup = item.cast<py::tuple>();
		                    for (idx_t col_idx = 0; col_idx < tup.size() && col_idx < bind_data.return_types.size(); col_idx++) {
		                        Value duck_val = TransformPythonValue(tup[col_idx], bind_data.return_types[col_idx]);
		                        row_values.push_back(duck_val);
		                    }
		                } else {
		                    // Single value case
		                    Value duck_val = TransformPythonValue(item, bind_data.return_types[0]);
		                    row_values.push_back(duck_val);
		                }
		                
		                // Set values in output chunk
		                for (idx_t col_idx = 0; col_idx < row_values.size() && col_idx < output.ColumnCount(); col_idx++) {
		                    output.SetValue(col_idx, items_processed, row_values[col_idx]);
		                }
		                
		                items_processed++;
		                
		            } catch (py::stop_iteration &) {
		                gs.iterator_exhausted = true;
		                break;
		            } catch (...) {
		                gs.iterator_exhausted = true;
		                break;
		            }
		        }
		        
		        output.SetCardinality(items_processed);
		        return;
		    }

		    if (gs.is_batched) {
			    // Batched dataset path - for large datasets
			    if (gs.current_batch_idx >= gs.batches.size()) {
				    output.SetCardinality(0);
				    return;
			    }

			    auto &current_batch = gs.batches[gs.current_batch_idx];
			    if (gs.current_row_in_batch >= current_batch.size()) {
				    // Move to next batch
				    gs.current_batch_idx++;
				    gs.current_row_in_batch = 0;
				    if (gs.current_batch_idx >= gs.batches.size()) {
					    output.SetCardinality(0);
					    return;
				    }
				    current_batch = gs.batches[gs.current_batch_idx];
			    }

			    // Emit from current batch
			    auto to_emit = MinValue<idx_t>(STANDARD_VECTOR_SIZE, current_batch.size() - gs.current_row_in_batch);
			    output.SetCardinality(to_emit);

			    for (idx_t i = 0; i < to_emit; i++) {
				    for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
					    if (col_idx < current_batch[gs.current_row_in_batch + i].size()) {
						    output.SetValue(col_idx, i, current_batch[gs.current_row_in_batch + i][col_idx]);
					    }
				    }
			    }
			    gs.current_row_in_batch += to_emit;
		    } else if (!gs.rows.empty()) {
			    // Small dataset path - use existing logic
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
		    } else {
			    // No data available
			    output.SetCardinality(0);
		    }
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
			// Check if result is a generator/iterator first
			if (py::hasattr(result, "__iter__") && !py::isinstance<py::list>(result) && !py::isinstance<py::str>(result)) {
				// Handle generator/iterator - store both generator and iterator for lazy processing
				gs->py_generator_object = result;  // Keep generator alive
				gs->py_iterator = py::iter(result);
				gs->has_iterator = true;
				gs->iterator_exhausted = false;
				// No processing here - will be done lazily in the main execution function
			} else if (py::isinstance<py::list>(result)) {
				auto py_list = result.cast<py::list>();
				
				// For small lists (< 50k items), process immediately for performance
				// For large lists, process in batches to avoid memory issues
				if (py_list.size() < 5000) {
					// Small dataset - process immediately (existing behavior)
					for (auto item : py_list) {
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
				} else {
					// Large dataset - process in batches to manage memory
					gs->is_batched = true;
					gs->current_batch_idx = 0;
					gs->current_row_in_batch = 0;
					
					// Process data in chunks smaller than STANDARD_VECTOR_SIZE for notebooks
					// Release GIL extremely frequently to prevent asyncio notebook freezing
					const idx_t batch_size = STANDARD_VECTOR_SIZE / 2; // Smaller batches for notebooks
					const idx_t batches_per_gil_release = 1; // Release GIL every batch for notebooks
					idx_t batch_count = 0;
					
					for (idx_t start_idx = 0; start_idx < py_list.size(); start_idx += batch_size) {
						idx_t end_idx = MinValue<idx_t>(start_idx + batch_size, py_list.size());
						vector<vector<Value>> batch;
						
						// Release GIL before processing each batch
						{
							py::gil_scoped_release release;
						}
						
						for (idx_t i = start_idx; i < end_idx; i++) {
							// Extremely aggressive GIL release for notebook environments
							// Release GIL every 50 items within a batch for asyncio compatibility
							if (i > 0 && (i - start_idx) % 50 == 0) {
								// Try to yield to Python asyncio event loop for VS Code notebooks
								try {
									py::module asyncio = py::module::import("asyncio");
									if (asyncio.attr("current_task")(py::none()).is_none() == false) {
										// We're in an async context, try to yield
										py::gil_scoped_release inner_release;
										std::this_thread::sleep_for(std::chrono::microseconds(10));
									} else {
										py::gil_scoped_release inner_release;
									}
								} catch (...) {
									// Fallback to regular GIL release
									py::gil_scoped_release inner_release;
								}
							}
							auto item = py_list[i];
							if (py::isinstance<py::tuple>(item)) {
								auto tup = item.cast<py::tuple>();
								if (tup.size() != bd.return_types.size()) {
									throw InvalidInputException(
									    "Table function '%s' returned tuple of size %d but schema expects %d columns",
									    bd.func_name, static_cast<int>(tup.size()), static_cast<int>(bd.return_types.size()));
								}
								vector<Value> row;
								for (idx_t j = 0; j < tup.size(); j++) {
									auto py_val = tup[j];
									Value duck_val = TransformPythonValue(py_val, bd.return_types[j]);
									row.push_back(duck_val);
								}
								batch.push_back(std::move(row));
							} else {
								if (bd.return_types.size() != 1) {
									throw InvalidInputException(
									    "Table function '%s' returned a single value but schema expects %d columns",
									    bd.func_name, static_cast<int>(bd.return_types.size()));
								}
								Value duck_val = TransformPythonValue(item, bd.return_types[0]);
								batch.push_back({duck_val});
							}
						}
						gs->batches.push_back(std::move(batch));
						
						// Release GIL after each batch for notebook responsiveness
						// Also yield to asyncio event loop for VS Code notebooks
						{
							py::gil_scoped_release release;
							// Yield control to allow asyncio event loop processing
							std::this_thread::sleep_for(std::chrono::microseconds(1));
						}
						
						batch_count++;
						// Periodically release and reacquire GIL to prevent notebook freezing
						if (batch_count % batches_per_gil_release == 0) {
							py::gil_scoped_release release;
							// Brief yield to let other threads (including notebook UI) run
							// GIL will be reacquired when release goes out of scope
						}
					}
				}
			}
		} else if (bd.return_type == PyTVFReturnType::ARROW_TABLE || bd.return_type == PyTVFReturnType::ARROW_BATCH) {
			// Create Arrow stream factory from the Python result
			auto factory = make_uniq<PythonTableArrowArrayStreamFactory>(result.ptr(), context.GetClientProperties(),
			                                                             DBConfig::GetConfig(context));

			// Build fake TableFunction input for Arrow scan
			vector<Value> children;
			children.push_back(Value::POINTER(CastPointerToValue(factory.get())));
			children.push_back(Value::POINTER(CastPointerToValue(PythonTableArrowArrayStreamFactory::Produce)));
			children.push_back(Value::POINTER(CastPointerToValue(PythonTableArrowArrayStreamFactory::GetSchema)));

			TableFunctionRef empty_ref;
			duckdb::TableFunction dummy_tf;
			dummy_tf.name = "PyTVFArrowWrapper";

			named_parameter_map_t named_params;
			vector<LogicalType> input_types;
			vector<string> input_names;

			TableFunctionBindInput bind_input(children, named_params, input_types, input_names, nullptr, nullptr,
			                                  dummy_tf, empty_ref);

			vector<LogicalType> return_types;
			vector<string> return_names;
			auto bind_data = ArrowTableFunction::ArrowScanBind(context, bind_input, return_types, return_names);

			// Prepare scan state
			vector<column_t> column_ids;
			for (idx_t i = 0; i < return_types.size(); i++) {
				column_ids.push_back(i);
			}
			vector<idx_t> projection_ids;
			TableFunctionInitInput init_input(bind_data.get(), column_ids, projection_ids, nullptr);
			auto global_state = ArrowTableFunction::ArrowScanInitGlobal(context, init_input);
			auto local_state = ArrowTableFunction::ArrowScanInitLocalInternal(context, init_input, global_state.get());

			// Scan and copy rows into gs->rows
			DataChunk chunk;
			chunk.Initialize(context, return_types, STANDARD_VECTOR_SIZE);
			while (true) {
				chunk.Reset();
				TableFunctionInput tf_input(bind_data.get(), local_state.get(), global_state.get());
				ArrowTableFunction::ArrowScanFunction(context, tf_input, chunk);
				if (chunk.size() == 0) {
					break;
				}

				for (idx_t r = 0; r < chunk.size(); r++) {
					vector<Value> row;
					for (idx_t c = 0; c < chunk.ColumnCount(); c++) {
						row.push_back(chunk.GetValue(c, r));
					}
					gs->rows.push_back(std::move(row));
				}
			}
		}

		return unique_ptr<GlobalTableFunctionState>(static_cast<GlobalTableFunctionState *>(gs.release()));
	};

	return tf;
}

} // namespace duckdb

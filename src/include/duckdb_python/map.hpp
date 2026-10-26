//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pandas/pandas_scan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/registered_py_object.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/execution/execution_context.hpp"

namespace duckdb {

//! Carried through the bind input so that SQL text can never forge a reference to the callable
struct MapFunctionInfo : public TableFunctionInfo {
	MapFunctionInfo(nb::object function_p, nb::object schema_p)
	    : function(std::move(function_p)), schema(std::move(schema_p)) {
	}
	PyObjectHolder function;
	PyObjectHolder schema;
};

struct MapFunction : public TableFunction {

public:
	MapFunction();

	static unique_ptr<FunctionData> MapFunctionBind(ClientContext &context, TableFunctionBindInput &input,
	                                                vector<LogicalType> &return_types, vector<Identifier> &names);

	static OperatorResultType MapFunctionExec(ExecutionContext &context, TableFunctionInput &data, DataChunk &input,
	                                          DataChunk &output);
};

} // namespace duckdb

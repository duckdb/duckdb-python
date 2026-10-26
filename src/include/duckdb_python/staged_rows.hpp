//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/staged_rows.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/cast/cast_function_set.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

//! A converted value cast to a column's type with the engine's cast; without an error string the cast raises its own
//! exception, as the cast of an executemany parameter does
inline Value CastToColumn(ClientContext &context, const Value &value, const LogicalType &type) {
	if (value.IsNull() || value.type() == type) {
		return value;
	}
	Vector input(value, count_t(1));
	Vector result(type, 1);
	GetCastFunctionInput get_input(context);
	VectorOperations::TryCast(CastFunctionSet::Get(context), get_input, input, result, 1, nullptr);
	return result.GetValue(0);
}

//! Puts one row of values, one per column, at the end of a chunk; whether the chunk is full and must be handed on
inline bool StageRow(DataChunk &staged, const vector<Value> &row) {
	auto index = staged.size();
	for (idx_t i = 0; i < row.size(); i++) {
		staged.data[i].SetValue(index, row[i]);
	}
	staged.SetChildCardinality(index + 1);
	return staged.size() == STANDARD_VECTOR_SIZE;
}

} // namespace duckdb

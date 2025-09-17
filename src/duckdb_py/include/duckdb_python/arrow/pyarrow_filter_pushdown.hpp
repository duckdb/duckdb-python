//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/arrow/pyarrow_filter_pushdown.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/main/client_properties.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

struct PyArrowFilterPushdown {
	static py::object TransformFilter(TableFilterSet &filter_collection, unordered_map<idx_t, string> &columns,
	                                  unordered_map<idx_t, idx_t> filter_to_col, const ClientProperties &config,
	                                  const ArrowTableSchema &arrow_table);
};

} // namespace duckdb

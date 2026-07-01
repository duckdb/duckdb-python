//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/arrow/polars_filter_pushdown.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/unordered_map.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/main/client_properties.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

struct PolarsFilterPushdown {
	static py::object TransformFilter(const TableFilterSet &filter_collection, unordered_map<idx_t, string> &columns,
	                                  const unordered_map<idx_t, idx_t> &filter_to_col,
	                                  const ClientProperties &client_properties);
};

} // namespace duckdb

#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb/common/common.hpp"

namespace duckdb {

struct PandasColumnBindData;

struct NumpyScan {
	static void Scan(ClientContext &context, PandasColumnBindData &bind_data, idx_t count, idx_t offset, Vector &out);
	//! masked holds one numpy bool per row of the whole column, read as a byte, or is null when the column has no mask
	static void ScanObjectColumn(ClientContext &context, PyObject **col, idx_t stride, const uint8_t *masked,
	                             idx_t count, idx_t offset, Vector &out);
};

} // namespace duckdb

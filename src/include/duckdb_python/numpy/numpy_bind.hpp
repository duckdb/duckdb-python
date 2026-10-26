#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb/common/common.hpp"

namespace duckdb {

struct PandasColumnBindData;
class ClientContext;

struct NumpyBind {
	//! Returns the length of the first array, which every other array must match
	static idx_t Bind(ClientContext &config, nb::handle df, vector<PandasColumnBindData> &out,
	                  vector<LogicalType> &return_types, vector<Identifier> &names);
};

} // namespace duckdb

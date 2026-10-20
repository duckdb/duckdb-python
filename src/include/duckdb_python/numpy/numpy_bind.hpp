#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb/common/common.hpp"

namespace duckdb {

struct PandasColumnBindData;
class ClientContext;

struct NumpyBind {
	static void Bind(ClientContext &config, nb::handle df, vector<PandasColumnBindData> &out,
	                 vector<LogicalType> &return_types, vector<Identifier> &names);
};

} // namespace duckdb

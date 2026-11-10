#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/types.hpp"

namespace duckdb {

struct PandasColumnBindData;

//! A cell of a declared column whose type was implied is not of that type's kind; source is the type the cell
//! converted to, and the caller reports it in the words of its own type check
struct ImpliedTypeRefused : public std::exception {
	explicit ImpliedTypeRefused(LogicalType source_p) : source(std::move(source_p)) {
	}
	const char *what() const noexcept override {
		return "a value is not of the kind of its column's implied type";
	}
	LogicalType source;
};

struct NumpyScan {
	static void Scan(ClientContext &context, PandasColumnBindData &bind_data, idx_t count, idx_t offset, Vector &out);
	//! masked holds one numpy bool per row of the whole column, read as a byte, or is null when the column has no mask
	static void ScanObjectColumn(ClientContext &context, PyObject **col, idx_t stride, const uint8_t *masked,
	                             idx_t count, idx_t offset, Vector &out);
	//! Converts each cell into the declared type of the column as the appender converts a value; with implied_type,
	//! a cell that is not of the type's kind raises ImpliedTypeRefused, and with bit_storage_refused so does one
	//! holding bytes where the type holds a BIT
	static void ScanDeclaredColumn(ClientContext &context, PyObject **col, idx_t stride, const uint8_t *masked,
	                               idx_t count, idx_t offset, Vector &out, bool implied_type, bool bit_storage_refused);
};

} // namespace duckdb

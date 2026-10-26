#pragma once

#include "duckdb/main/client_context_state.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb_python/arrow/python_arrow_scan.hpp"
#include "duckdb_python/nb/casters.hpp"

namespace duckdb {

enum class ArrowObjectKind : uint8_t { NONE, SOURCE, MESSAGE_READER };

struct PythonReplacementScan {
public:
	//! Whether an object is read as Arrow, and how; a capsule that is not an unreleased Arrow stream throws, and so
	//! does a probed attribute that raises, with an error naming the object by `name` and `kind`
	static ArrowObjectKind ClassifyArrowObject(const nb::handle &entry, const string &name, ArrowSourceKind kind);
	static unique_ptr<TableRef> Replace(ClientContext &context, ReplacementScanInput &input,
	                                    optional_ptr<ReplacementScanData> data);
	//! Try to perform a replacement, returns NULL on error; `kind` says where the object came from, and only
	//! from_arrow (NAMELESS) reads a pyarrow MessageReader
	static unique_ptr<TableRef> TryReplacementObject(const nb::object &entry, const string &name,
	                                                 ClientContext &context,
	                                                 ArrowSourceKind kind = ArrowSourceKind::REGISTERED);
	//! Perform a replacement or throw if it failed
	static unique_ptr<TableRef> ReplacementObject(const nb::object &entry, const string &name, ClientContext &context,
	                                              ArrowSourceKind kind = ArrowSourceKind::REGISTERED);
};

} // namespace duckdb

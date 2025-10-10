
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/import_cache/modules/pyarrow_module.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/import_cache/python_import_cache_item.hpp"

//! Note: This class is generated using scripts.
//! If you need to add a new object to the cache you must:
//! 1. adjust scripts/imports.py
//! 2. run python scripts/generate_import_cache_json.py
//! 3. run python scripts/generate_import_cache_cpp.py
//! 4. run pre-commit to fix formatting errors

namespace duckdb {

struct PyarrowIpcCacheItem : public PythonImportCacheItem {

public:
	PyarrowIpcCacheItem(optional_ptr<PythonImportCacheItem> parent)
	    : PythonImportCacheItem("ipc", parent), MessageReader("MessageReader", this) {
	}
	~PyarrowIpcCacheItem() override {
	}

	PythonImportCacheItem MessageReader;
};

struct PyarrowDatasetCacheItem : public PythonImportCacheItem {

public:
	static constexpr const char *Name = "pyarrow.dataset";

public:
	PyarrowDatasetCacheItem()
	    : PythonImportCacheItem("pyarrow.dataset"), Scanner("Scanner", this), Dataset("Dataset", this) {
	}
	~PyarrowDatasetCacheItem() override {
	}

	PythonImportCacheItem Scanner;
	PythonImportCacheItem Dataset;
};

struct PyarrowCacheItem : public PythonImportCacheItem {

public:
	static constexpr const char *Name = "pyarrow";

public:
	PyarrowCacheItem()
	    : PythonImportCacheItem("pyarrow"), dataset(), Table("Table", this),
	      RecordBatchReader("RecordBatchReader", this), ipc(this), scalar("scalar", this), date32("date32", this),
	      time64("time64", this), timestamp("timestamp", this), uint8("uint8", this), uint16("uint16", this),
	      uint32("uint32", this), uint64("uint64", this), binary_view("binary_view", this),
	      decimal32("decimal32", this), decimal64("decimal64", this), decimal128("decimal128", this) {
	}
	~PyarrowCacheItem() override {
	}

	PyarrowDatasetCacheItem dataset;
	PythonImportCacheItem Table;
	PythonImportCacheItem RecordBatchReader;
	PyarrowIpcCacheItem ipc;
	PythonImportCacheItem scalar;
	PythonImportCacheItem date32;
	PythonImportCacheItem time64;
	PythonImportCacheItem timestamp;
	PythonImportCacheItem uint8;
	PythonImportCacheItem uint16;
	PythonImportCacheItem uint32;
	PythonImportCacheItem uint64;
	PythonImportCacheItem binary_view;
	PythonImportCacheItem decimal32;
	PythonImportCacheItem decimal64;
	PythonImportCacheItem decimal128;
};

} // namespace duckdb

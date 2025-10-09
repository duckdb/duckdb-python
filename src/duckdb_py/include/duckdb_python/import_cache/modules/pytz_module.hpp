
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/import_cache/modules/pytz_module.hpp
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

struct PytzCacheItem : public PythonImportCacheItem {

public:
	static constexpr const char *Name = "pytz";

public:
	PytzCacheItem() : PythonImportCacheItem("pytz"), timezone("timezone", this) {
	}
	~PytzCacheItem() override {
	}

	PythonImportCacheItem timezone;
};

} // namespace duckdb

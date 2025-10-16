
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/import_cache/modules/ipywidgets_module.hpp
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

struct IpywidgetsCacheItem : public PythonImportCacheItem {

public:
	static constexpr const char *Name = "ipywidgets";

public:
	IpywidgetsCacheItem() : PythonImportCacheItem("ipywidgets"), FloatProgress("FloatProgress", this) {
	}
	~IpywidgetsCacheItem() override {
	}

	PythonImportCacheItem FloatProgress;

protected:
	bool IsRequired() const override final {
		return false;
	}
};

} // namespace duckdb

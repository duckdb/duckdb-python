#include "duckdb_python/import_cache/python_import_cache.hpp"
#include "duckdb_python/import_cache/python_import_cache_item.hpp"
#include "duckdb/common/stack.hpp"
#include "duckdb_python/import_cache/importer.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// PythonImportCacheItem (SUPER CLASS)
//===--------------------------------------------------------------------===//

nb::handle PythonImportCacheItem::operator()(bool load) {
	if (IsLoaded()) {
		return object;
	}
	stack<reference<PythonImportCacheItem>> hierarchy;

	optional_ptr<PythonImportCacheItem> item = this;
	while (item) {
		hierarchy.emplace(*item);
		item = item->parent;
	}
	return PythonImporter::Import(hierarchy, load);
}

bool PythonImportCacheItem::LoadSucceeded() const {
	return load_succeeded;
}

inline bool PythonImportCacheItem::IsLoaded() const {
	return object.ptr() != nullptr;
}

nb::handle PythonImportCacheItem::AddCache(PythonImportCache &cache, nb::object object) {
	return cache.AddCache(std::move(object));
}

void PythonImportCacheItem::LoadModule(PythonImportCache &cache) {
	try {
		duckdb::PyUtil::GilAssert();
		object = AddCache(cache, std::move(nb::module_::import_(name.c_str())));
		load_succeeded = true;
	} catch (nb::python_error &e) {
		if (IsRequired()) {
			throw InvalidInputException(
			    "Required module '%s' failed to import, due to the following Python exception:\n%s", name, e.what());
		}
		object = nullptr;
		return;
	}
}

void PythonImportCacheItem::LoadAttribute(PythonImportCache &cache, nb::handle source) {
	if (nb::hasattr(source, name.c_str())) {
		object = AddCache(cache, std::move(source.attr(name.c_str())));
	} else {
		object = nullptr;
	}
}

nb::handle PythonImportCacheItem::Load(PythonImportCache &cache, nb::handle source, bool load) {
	if (IsLoaded()) {
		return object;
	}
	if (!load) {
		// Don't load the item if it's not already loaded
		return object;
	}
	if (is_module) {
		LoadModule(cache);
	} else {
		LoadAttribute(cache, source);
	}
	return object;
}

//===--------------------------------------------------------------------===//
// PythonImportCache (CONTAINER)
//===--------------------------------------------------------------------===//

PythonImportCache::~PythonImportCache() {
	if (!nb::is_alive()) {
		// Process-global state, so this can run from static destruction: acquiring the GIL there
		// is fatal and dropping the references without it is undefined behaviour, so leak them.
		for (auto &object : owned_objects) {
			object.release();
		}
		return;
	}
	try {
		nb::gil_scoped_acquire acquire;
		owned_objects.clear();
	} catch (...) { // NOLINT
	}
}

nb::handle PythonImportCache::AddCache(nb::object item) {
	auto object_ptr = item.ptr();
	owned_objects.push_back(std::move(item));
	return object_ptr;
}

} // namespace duckdb

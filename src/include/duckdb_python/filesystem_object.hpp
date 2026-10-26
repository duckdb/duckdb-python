//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/filesystem_object.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb_python/registered_py_object.hpp"
#include "duckdb_python/pyfilesystem.hpp"

namespace duckdb {

class FileSystemObject : public RegisteredObject {
public:
	explicit FileSystemObject(nb::object fs, vector<string> filenames_p)
	    : RegisteredObject(std::move(fs)), filenames(std::move(filenames_p)) {
	}
	~FileSystemObject() override {
		nb::detail::cleanup_guard guard {};
		if (!guard) {
			return;
		}
		D_ASSERT(duckdb::PyUtil::IsInstance(
		    obj, DuckDBPyConnection::ImportCache()->duckdb.filesystem.ModifiedMemoryFileSystem()));
		// Destructors are implicitly noexcept: a Python exception escaping here (fsspec `_rm` raises
		// KeyError for a missing entry) would std::terminate the process. Swallow it, mirroring
		// ~PythonFileHandle / ~PythonFilesystem.
		try {
			for (auto &file : filenames) {
				obj.attr("delete")(file);
			}
		} catch (...) { // NOLINT: intentional catch-all in a destructor
		}
	}

	vector<string> filenames;
};

} // namespace duckdb

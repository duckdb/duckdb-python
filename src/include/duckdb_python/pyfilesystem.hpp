#pragma once

#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/storage/object_cache.hpp"

namespace duckdb {

class ModifiedMemoryFileSystem : public nb::object {
public:
	using nb::object::object;
	ModifiedMemoryFileSystem(nb::object object) : nb::object(object) {
	}

public:
	static bool check_(const nb::handle &object) {
		// Non-throwing: nanobind can invoke check_ from noexcept caster / isinstance contexts, where a
		// thrown import error or an IsInstance failure (PyObject_IsInstance == -1) would std::terminate.
		// Mirror AbstractFileSystem::check_ and report "not an instance" on any error.
		try {
			return duckdb::PyUtil::IsInstance(
			    object, nb::module_::import_("duckdb.filesystem").attr("ModifiedMemoryFileSystem"));
		} catch (...) {
			return false;
		}
	}
};

class AbstractFileSystem : public nb::object {
public:
	using nb::object::object;

public:
	static bool check_(const nb::handle &object) {
		// Non-throwing: if fsspec isn't installed, nothing is an AbstractFileSystem. nanobind invokes check_ from
		// noexcept contexts (argument casters, isinstance), so a thrown import error would std::terminate rather
		// than propagate. register_filesystem() re-imports fsspec in a throwing context to surface ModuleNotFoundError.
		try {
			return duckdb::PyUtil::IsInstance(object, nb::module_::import_("fsspec").attr("AbstractFileSystem"));
		} catch (...) {
			return false;
		}
	}
};

class PythonFileHandle : public FileHandle {
public:
	//! opened_during_exit is the opening QueryGil's AdmittedDuringExit, which later calls on the handle carry
	PythonFileHandle(FileSystem &file_system, const string &path, const nb::object &handle, FileOpenFlags flags,
	                 bool opened_during_exit);
	~PythonFileHandle() override;
	void Close() override;

	static const nb::object &GetHandle(const FileHandle &handle);
	static bool OpenedDuringExit(const FileHandle &handle);

private:
	nb::object handle;
	bool opened_during_exit;
};

class PythonFilesystem : public FileSystem {
private:
	const vector<string> protocols;
	AbstractFileSystem filesystem;
	std::string DecodeFlags(FileOpenFlags flags);
	bool Exists(const string &filename, const char *func_name, optional_ptr<FileOpener> opener) const;

public:
	explicit PythonFilesystem(vector<string> protocols, AbstractFileSystem filesystem)
	    : protocols(std::move(protocols)), filesystem(std::move(filesystem)) {
	}
	~PythonFilesystem() override;

protected:
	string GetName() const override {
		return protocols[0];
	}

public:
	unique_ptr<FileHandle> OpenFile(const string &path, FileOpenFlags flags, optional_ptr<FileOpener> opener) override;
	void Seek(duckdb::FileHandle &handle, uint64_t location) override;
	FileType GetFileType(FileHandle &handle) override {
		return FileType::FILE_TYPE_REGULAR;
	}
	int64_t Read(FileHandle &handle, void *buffer, int64_t nr_bytes) override;
	void Read(duckdb::FileHandle &handle, void *buffer, int64_t nr_bytes, uint64_t location) override;

	void Write(FileHandle &handle, void *buffer, int64_t nr_bytes, idx_t location) override;
	int64_t Write(FileHandle &handle, void *buffer, int64_t nr_bytes) override;

	bool FileExists(const string &filename, optional_ptr<FileOpener> opener = nullptr) override;
	vector<OpenFileInfo> Glob(const string &path, FileOpener *opener) override;
	bool CanHandleFile(const string &fpath) override;
	bool CanSeek() override {
		return true;
	}

	bool IsManuallySet() override {
		return true;
	}

	bool OnDiskFile(FileHandle &handle) override {
		return false;
	}
	string PathSeparator(const string &path) override;
	int64_t GetFileSize(FileHandle &handle) override;
	void RemoveFile(const string &filename, optional_ptr<FileOpener> opener = nullptr) override;
	void MoveFile(const string &source, const string &dest, optional_ptr<FileOpener> opener = nullptr) override;
	timestamp_t GetLastModifiedTime(FileHandle &handle) override;
	void FileSync(FileHandle &handle) override;
	bool DirectoryExists(const string &directory, optional_ptr<FileOpener> opener = nullptr) override;
	void CreateDirectory(const string &directory, optional_ptr<FileOpener> opener = nullptr) override;
	void RemoveDirectory(const string &directory, optional_ptr<FileOpener> opener = nullptr) override;
	bool ListFiles(const string &directory, const std::function<void(const string &, bool)> &callback,
	               FileOpener *opener = nullptr) override;
	void Truncate(FileHandle &handle, int64_t new_size) override;
	bool IsPipe(const string &filename, optional_ptr<FileOpener> opener = nullptr) override;
	idx_t SeekPosition(FileHandle &handle) override;
};

//! The in-memory file system that holds file-like objects, one per database: a cursor registers nothing of its own,
//! since the database's virtual file system takes the name once. Found through the database's object cache, so every
//! connection and cursor of the database reads the same entries.
class ObjectStore : public ObjectCacheEntry {
public:
	~ObjectStore() override;

	static string ObjectType() {
		return "python_object_store";
	}
	string GetObjectType() override {
		return ObjectType();
	}
	optional_idx GetEstimatedCacheMemory() const override {
		return optional_idx();
	}

public:
	//! The one owner of the Python file system's reference. Read and written with the GIL held, which is what
	//! serialises the connections of a database here
	unique_ptr<ModifiedMemoryFileSystem> filesystem;
};

} // namespace duckdb

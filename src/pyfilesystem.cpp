#include "duckdb_python/pyfilesystem.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/nb/casters.hpp"

namespace duckdb {

PythonFileHandle::PythonFileHandle(FileSystem &file_system, const string &path, const nb::object &handle,
                                   FileOpenFlags flags)
    : FileHandle(file_system, path, flags), handle(handle) {
}
PythonFileHandle::~PythonFileHandle() {
	try {
		nb::gil_scoped_acquire gil;
		handle.dec_ref();
		handle.release();
	} catch (...) { // NOLINT
	}
}

const nb::object &PythonFileHandle::GetHandle(const FileHandle &handle) {
	return handle.Cast<PythonFileHandle>().handle;
}

void PythonFileHandle::Close() {
	nb::gil_scoped_acquire gil;
	handle.attr("close")();
}

PythonFilesystem::~PythonFilesystem() {
	try {
		nb::gil_scoped_acquire gil;
		filesystem.dec_ref();
		filesystem.release();
	} catch (...) { // NOLINT
	}
}

string PythonFilesystem::DecodeFlags(FileOpenFlags flags) {
	// see https://stackoverflow.com/a/58925279 for truth table of python file modes
	bool read = flags.OpenForReading();
	bool write = flags.OpenForWriting();
	bool append = flags.OpenForAppending();
	bool truncate = flags.OverwriteExistingFile();

	string flags_s;
	if (read && write && truncate) {
		flags_s = "w+";
	} else if (read && write && append) {
		flags_s = "a+";
	} else if (read && write) {
		flags_s = "r+";
	} else if (read) {
		flags_s = "r";
	} else if (write) {
		flags_s = "w";
	} else if (append) {
		flags_s = "a";
	} else {
		throw InvalidInputException("%s: unsupported file flags", GetName());
	}

	flags_s.insert(1, "b"); // always read in binary mode

	return flags_s;
}

unique_ptr<FileHandle> PythonFilesystem::OpenFile(const string &path, FileOpenFlags flags,
                                                  optional_ptr<FileOpener> opener) {
	nb::gil_scoped_acquire gil;

	if (flags.Compression() != FileCompressionType::UNCOMPRESSED) {
		throw IOException("Compression not supported");
	}
	// maybe this can be implemented in a better way?
	if (flags.ReturnNullIfNotExists()) {
		if (!FileExists(path)) {
			return nullptr;
		}
	}

	// TODO: lock support?

	string flags_s = DecodeFlags(flags);

	const auto &handle = filesystem.attr("open")(path, nb::str(flags_s.c_str(), flags_s.size()));
	return make_uniq<PythonFileHandle>(*this, path, handle, flags);
}

int64_t PythonFilesystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	nb::gil_scoped_acquire gil;

	const auto &write = PythonFileHandle::GetHandle(handle).attr("write");

	auto data = nb::bytes(const_char_ptr_cast(buffer), nr_bytes);

	return nb::cast<int64_t>(write(data));
}
void PythonFilesystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes, idx_t location) {
	nb::gil_scoped_acquire gil;
	auto &py_handle = PythonFileHandle::GetHandle(handle);
	py_handle.attr("seek")(location);
	auto data = nb::bytes(const_char_ptr_cast(buffer), nr_bytes);
	py_handle.attr("write")(data);
}

int64_t PythonFilesystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	nb::gil_scoped_acquire gil;

	const auto &read = PythonFileHandle::GetHandle(handle).attr("read");

	nb::bytes data = nb::bytes(read(nr_bytes));

	// `buffer` is sized for nr_bytes. A misbehaving fsspec read(n) may return MORE than n bytes; clamp so
	// the copy can never overflow `buffer`. Returning fewer than nr_bytes is a legal short read (EOF).
	int64_t data_size = static_cast<int64_t>(data.size());
	int64_t bytes_to_copy = data_size < nr_bytes ? data_size : nr_bytes;
	memcpy(buffer, data.c_str(), static_cast<size_t>(bytes_to_copy));

	return bytes_to_copy;
}

void PythonFilesystem::Read(duckdb::FileHandle &handle, void *buffer, int64_t nr_bytes, uint64_t location) {
	nb::gil_scoped_acquire gil;
	auto &py_handle = PythonFileHandle::GetHandle(handle);
	py_handle.attr("seek")(location);
	nb::bytes data = nb::bytes(py_handle.attr("read")(nr_bytes));
	// This overload must populate exactly nr_bytes: DuckDB assumes the whole buffer is filled. A short read
	// would leave the tail uninitialized (garbage handed back to the engine), so surface it as an error.
	// A read returning more than nr_bytes is clamped so it can never overflow `buffer`.
	int64_t data_size = static_cast<int64_t>(data.size());
	if (data_size < nr_bytes) {
		throw IOException("Failed to read " + std::to_string(nr_bytes) + " bytes from Python file at offset " +
		                  std::to_string(location) + ": only " + std::to_string(data_size) + " bytes returned");
	}
	memcpy(buffer, data.c_str(), static_cast<size_t>(nr_bytes));
}
bool PythonFilesystem::FileExists(const string &filename, optional_ptr<FileOpener> opener) {
	return Exists(filename, "isfile");
}
bool PythonFilesystem::Exists(const string &filename, const char *func_name) const {
	nb::gil_scoped_acquire gil;

	return nb::cast<bool>(filesystem.attr(func_name)(filename));
}
vector<OpenFileInfo> PythonFilesystem::Glob(const string &path, FileOpener *opener) {
	nb::gil_scoped_acquire gil;

	if (path.empty()) {
		return {path};
	}
	auto returner = nb::list(filesystem.attr("glob")(path));

	vector<OpenFileInfo> results;
	auto unstrip_protocol = filesystem.attr("unstrip_protocol");
	for (auto item : returner) {
		string file_path = nb::cast<std::string>(unstrip_protocol(nb::str(item)));
		results.emplace_back(file_path);
	}
	return results;
}
string PythonFilesystem::PathSeparator(const string &path) {
	return "/";
}
int64_t PythonFilesystem::GetFileSize(FileHandle &handle) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	// TODO: this value should be cached on the PythonFileHandle
	nb::gil_scoped_acquire gil;

	return nb::cast<int64_t>(filesystem.attr("size")(handle.path));
}
void PythonFilesystem::Seek(duckdb::FileHandle &handle, uint64_t location) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	nb::gil_scoped_acquire gil;

	auto seek = PythonFileHandle::GetHandle(handle).attr("seek");
	seek(location);
	if (PyErr_Occurred()) {
		PyErr_PrintEx(1);
		throw InvalidInputException("Python exception occurred!");
	}
}
bool PythonFilesystem::CanHandleFile(const string &fpath) {
	for (const auto &protocol : protocols) {
		if (StringUtil::StartsWith(fpath, protocol + "://")) {
			return true;
		}
	}
	return false;
}
void PythonFilesystem::MoveFile(const string &source, const string &dest, optional_ptr<FileOpener> opener) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	nb::gil_scoped_acquire gil;

	auto move = filesystem.attr("mv");
	move(nb::str(source.c_str(), source.size()), nb::str(dest.c_str(), dest.size()));
}
void PythonFilesystem::RemoveFile(const string &filename, optional_ptr<FileOpener> opener) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	nb::gil_scoped_acquire gil;

	auto remove = filesystem.attr("rm");
	remove(nb::str(filename.c_str(), filename.size()));
}

static bool IsUnsupportedModificationTimeError(const nb::python_error &error) {
	if (error.matches(PyExc_NotImplementedError)) {
		return true;
	}
	if (!error.matches(PyExc_KeyError)) {
		return false;
	}
	// gcsfs: GCSFileSystem.modified() indexes info(path)["mtime"] and raises KeyError('mtime') when
	// object metadata (or synthesized directory entries) has no mtime.
	try {
		nb::tuple args = nb::cast<nb::tuple>(error.value().attr("args"));
		return nb::len(args) == 1 && nb::cast<string>(nb::str(args[0])) == "mtime";
	} catch (...) {
		return false;
	}
}

timestamp_t PythonFilesystem::GetLastModifiedTime(FileHandle &handle) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	// TODO: this value should be cached on the PythonFileHandle
	nb::gil_scoped_acquire gil;

	nb::object last_mod;
	try {
		last_mod = filesystem.attr("modified")(handle.path);
	} catch (nb::python_error &e) {
		if (IsUnsupportedModificationTimeError(e)) {
			throw NotImplementedException("%s: GetLastModifiedTime is not implemented", GetName());
		}
		throw;
	}

	// datetime.timestamp() returns a float; truncate to int64 seconds (nb::cast<int64_t> would reject a float)
	return Timestamp::FromEpochSeconds((int64_t)nb::cast<double>(last_mod.attr("timestamp")()));
}
void PythonFilesystem::FileSync(FileHandle &handle) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	nb::gil_scoped_acquire gil;

	PythonFileHandle::GetHandle(handle).attr("flush")();
}
bool PythonFilesystem::DirectoryExists(const string &directory, optional_ptr<FileOpener> opener) {
	return Exists(directory, "isdir");
}
void PythonFilesystem::RemoveDirectory(const string &directory, optional_ptr<FileOpener> opener) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	nb::gil_scoped_acquire gil;

	filesystem.attr("rm")(directory, nb::arg("recursive") = true);
}
void PythonFilesystem::CreateDirectory(const string &directory, optional_ptr<FileOpener> opener) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	nb::gil_scoped_acquire gil;

	filesystem.attr("mkdir")(nb::str(directory.c_str(), directory.size()));
}
bool PythonFilesystem::ListFiles(const string &directory, const std::function<void(const string &, bool)> &callback,
                                 FileOpener *opener) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	nb::gil_scoped_acquire gil;
	bool nonempty = false;

	for (auto item : filesystem.attr("ls")(nb::str(directory.c_str(), directory.size()))) {
		bool is_dir = nb::cast<std::string>(item["type"]) == "directory";
		callback(nb::cast<std::string>(item["name"]), is_dir);
		nonempty = true;
	}

	return nonempty;
}
void PythonFilesystem::Truncate(FileHandle &handle, int64_t new_size) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	nb::gil_scoped_acquire gil;

	filesystem.attr("touch")(handle.path, nb::arg("truncate") = true);
}
bool PythonFilesystem::IsPipe(const string &filename, optional_ptr<FileOpener> opener) {
	return false;
}
idx_t PythonFilesystem::SeekPosition(FileHandle &handle) {
	D_ASSERT(!duckdb::PyUtil::GilCheck());
	nb::gil_scoped_acquire gil;

	return nb::cast<idx_t>(PythonFileHandle::GetHandle(handle).attr("tell")());
}
} // namespace duckdb

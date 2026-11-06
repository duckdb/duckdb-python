#include "duckdb_python/pyfilesystem.hpp"

#include "duckdb/common/file_opener.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/exceptions.hpp"
#include "duckdb_python/nb/casters.hpp"

namespace duckdb {

//! Runs fun, which calls into Python under the caller's GIL; a Python error it raises leaves as an engine error
template <class FUN>
static auto CallPython(FUN &&fun) -> decltype(fun()) {
	try {
		return fun();
	} catch (nb::python_error &error) {
		throw AsDuckDBError(error);
	}
}

PythonFileHandle::PythonFileHandle(FileSystem &file_system, const string &path, const nb::object &handle,
                                   FileOpenFlags flags, bool opened_during_exit)
    : FileHandle(file_system, path, flags), handle(handle), opened_during_exit(opened_during_exit) {
}
PythonFileHandle::~PythonFileHandle() {
	try {
		PyUtil::QueryGil gil(nullptr, opened_during_exit);
		handle.dec_ref();
	} catch (...) { // NOLINT
	}
	// Leaked when the gate refused, as the reference can then no longer be dropped
	handle.release();
}

const nb::object &PythonFileHandle::GetHandle(const FileHandle &handle) {
	return handle.Cast<PythonFileHandle>().handle;
}

bool PythonFileHandle::OpenedDuringExit(const FileHandle &handle) {
	return handle.Cast<PythonFileHandle>().opened_during_exit;
}

void PythonFileHandle::Close() {
	PyUtil::QueryGil gil(nullptr, opened_during_exit);
	CallPython([&]() { handle.attr("close")(); });
}

PythonFilesystem::~PythonFilesystem() {
	try {
		PyUtil::QueryGil gil(nullptr);
		filesystem.dec_ref();
	} catch (...) { // NOLINT
	}
	// Leaked when the gate refused, as the reference can then no longer be dropped
	filesystem.release();
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
	PyUtil::QueryGil gil(FileOpener::TryGetClientContext(opener));
	return CallPython([&]() -> unique_ptr<FileHandle> {
		if (flags.Compression() != FileCompressionType::UNCOMPRESSED) {
			throw IOException("Compression not supported");
		}
		// maybe this can be implemented in a better way?
		if (flags.ReturnNullIfNotExists()) {
			if (!FileExists(path, opener)) {
				return nullptr;
			}
		}

		// TODO: lock support?

		string flags_s = DecodeFlags(flags);

		const auto &handle = filesystem.attr("open")(path, nb::str(flags_s.c_str(), flags_s.size()));
		return make_uniq<PythonFileHandle>(*this, path, handle, flags, gil.AdmittedDuringExit());
	});
}

int64_t PythonFilesystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	PyUtil::QueryGil gil(nullptr, PythonFileHandle::OpenedDuringExit(handle));
	return CallPython([&]() -> int64_t {
		const auto &write = PythonFileHandle::GetHandle(handle).attr("write");

		auto data = nb::bytes(const_char_ptr_cast(buffer), nr_bytes);

		return nb::cast<int64_t>(write(data));
	});
}
void PythonFilesystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes, idx_t location) {
	PyUtil::QueryGil gil(nullptr, PythonFileHandle::OpenedDuringExit(handle));
	CallPython([&]() -> void {
		auto &py_handle = PythonFileHandle::GetHandle(handle);
		py_handle.attr("seek")(location);
		auto data = nb::bytes(const_char_ptr_cast(buffer), nr_bytes);
		py_handle.attr("write")(data);
	});
}

int64_t PythonFilesystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	PyUtil::QueryGil gil(nullptr, PythonFileHandle::OpenedDuringExit(handle));
	return CallPython([&]() -> int64_t {
		const auto &read = PythonFileHandle::GetHandle(handle).attr("read");

		nb::bytes data = nb::bytes(read(nr_bytes));

		// `buffer` is sized for nr_bytes. A misbehaving fsspec read(n) may return MORE than n bytes; clamp so
		// the copy can never overflow `buffer`. Returning fewer than nr_bytes is a legal short read (EOF).
		int64_t data_size = static_cast<int64_t>(data.size());
		int64_t bytes_to_copy = data_size < nr_bytes ? data_size : nr_bytes;
		memcpy(buffer, data.c_str(), static_cast<size_t>(bytes_to_copy));

		return bytes_to_copy;
	});
}

void PythonFilesystem::Read(duckdb::FileHandle &handle, void *buffer, int64_t nr_bytes, uint64_t location) {
	PyUtil::QueryGil gil(nullptr, PythonFileHandle::OpenedDuringExit(handle));
	CallPython([&]() -> void {
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
	});
}
bool PythonFilesystem::FileExists(const string &filename, optional_ptr<FileOpener> opener) {
	return Exists(filename, "isfile", opener);
}
bool PythonFilesystem::Exists(const string &filename, const char *func_name, optional_ptr<FileOpener> opener) const {
	PyUtil::QueryGil gil(FileOpener::TryGetClientContext(opener));
	return CallPython([&]() -> bool { return nb::cast<bool>(filesystem.attr(func_name)(filename)); });
}
// fsspec writes a Windows path as file://C:/..., with the drive where a URI's host goes, and the engine reads only the
// standard file:///C:/...
static string WithDriveLetterAsPath(string uri) {
	if (StringUtil::StartsWith(uri, "file://") && uri.size() >= 9 && StringUtil::CharacterIsAlpha(uri[7]) &&
	    uri[8] == ':') {
		uri.insert(7, "/");
	}
	return uri;
}

vector<OpenFileInfo> PythonFilesystem::Glob(const string &path, FileOpener *opener) {
	PyUtil::QueryGil gil(FileOpener::TryGetClientContext(opener));
	return CallPython([&]() -> vector<OpenFileInfo> {
		if (path.empty()) {
			return {path};
		}
		auto returner = nb::list(filesystem.attr("glob")(path));

		vector<OpenFileInfo> results;
		auto unstrip_protocol = filesystem.attr("unstrip_protocol");
		for (auto item : returner) {
			results.emplace_back(WithDriveLetterAsPath(nb::cast<std::string>(unstrip_protocol(nb::str(item)))));
		}
		return results;
	});
}
string PythonFilesystem::PathSeparator(const string &path) {
	return "/";
}
int64_t PythonFilesystem::GetFileSize(FileHandle &handle) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	// TODO: this value should be cached on the PythonFileHandle
	PyUtil::QueryGil gil(nullptr, PythonFileHandle::OpenedDuringExit(handle));
	return CallPython([&]() -> int64_t { return nb::cast<int64_t>(filesystem.attr("size")(handle.path)); });
}
void PythonFilesystem::Seek(duckdb::FileHandle &handle, uint64_t location) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	PyUtil::QueryGil gil(nullptr, PythonFileHandle::OpenedDuringExit(handle));
	CallPython([&]() -> void {
		auto seek = PythonFileHandle::GetHandle(handle).attr("seek");
		seek(location);
	});
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
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	PyUtil::QueryGil gil(FileOpener::TryGetClientContext(opener));
	CallPython([&]() -> void {
		auto move = filesystem.attr("mv");
		move(nb::str(source.c_str(), source.size()), nb::str(dest.c_str(), dest.size()));
	});
}
void PythonFilesystem::RemoveFile(const string &filename, optional_ptr<FileOpener> opener) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	PyUtil::QueryGil gil(FileOpener::TryGetClientContext(opener));
	CallPython([&]() -> void {
		auto remove = filesystem.attr("rm");
		remove(nb::str(filename.c_str(), filename.size()));
	});
}
timestamp_t PythonFilesystem::GetLastModifiedTime(FileHandle &handle) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	// TODO: this value should be cached on the PythonFileHandle
	PyUtil::QueryGil gil(nullptr, PythonFileHandle::OpenedDuringExit(handle));
	return CallPython([&]() -> timestamp_t {
		nb::object last_mod;
		try {
			last_mod = filesystem.attr("modified")(handle.path);
		} catch (nb::python_error &error) {
			// fsspec's default modified() raises NotImplementedError, and DirectFileReader turns a
			// NotImplementedException from the metadata it reads into NULL for that column.
			if (!error.matches(PyExc_NotImplementedError)) {
				throw;
			}
			auto message = GetName() + ": modified() is not implemented";
			// str() of the exception can itself raise, and the error is still a NotImplementedException without its
			// text.
			try {
				auto detail = nb::cast<std::string>(nb::str(error.value()));
				if (!detail.empty()) {
					message += ": " + detail;
				}
			} catch (nb::python_error &) {
			} catch (nb::cast_error &) {
			}
			throw NotImplementedException(message);
		}

		// datetime.timestamp() returns a float; truncate to int64 seconds (nb::cast<int64_t> would reject a float)
		return Timestamp::FromEpochSeconds((int64_t)nb::cast<double>(last_mod.attr("timestamp")()));
	});
}
void PythonFilesystem::FileSync(FileHandle &handle) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	PyUtil::QueryGil gil(nullptr, PythonFileHandle::OpenedDuringExit(handle));
	CallPython([&]() -> void { PythonFileHandle::GetHandle(handle).attr("flush")(); });
}
bool PythonFilesystem::DirectoryExists(const string &directory, optional_ptr<FileOpener> opener) {
	return Exists(directory, "isdir", opener);
}
void PythonFilesystem::RemoveDirectory(const string &directory, optional_ptr<FileOpener> opener) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	PyUtil::QueryGil gil(FileOpener::TryGetClientContext(opener));
	CallPython([&]() -> void { filesystem.attr("rm")(directory, nb::arg("recursive") = true); });
}
void PythonFilesystem::CreateDirectory(const string &directory, optional_ptr<FileOpener> opener) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	PyUtil::QueryGil gil(FileOpener::TryGetClientContext(opener));
	CallPython([&]() -> void { filesystem.attr("mkdir")(nb::str(directory.c_str(), directory.size())); });
}
bool PythonFilesystem::ListFiles(const string &directory, const std::function<void(const string &, bool)> &callback,
                                 FileOpener *opener) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	PyUtil::QueryGil gil(FileOpener::TryGetClientContext(opener));
	return CallPython([&]() -> bool {
		bool nonempty = false;

		for (auto item : filesystem.attr("ls")(nb::str(directory.c_str(), directory.size()))) {
			bool is_dir = nb::cast<std::string>(item["type"]) == "directory";
			// fsspec names an entry by its full path, and the engine joins the name it gets onto the directory
			auto path = nb::cast<std::string>(item["name"]);
			auto end = path.find_last_not_of('/');
			if (end == string::npos) {
				continue;
			}
			auto separator = path.find_last_of('/', end);
			auto begin = separator == string::npos ? 0 : separator + 1;
			callback(path.substr(begin, end - begin + 1), is_dir);
			nonempty = true;
		}

		return nonempty;
	});
}
void PythonFilesystem::Truncate(FileHandle &handle, int64_t new_size) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	PyUtil::QueryGil gil(nullptr, PythonFileHandle::OpenedDuringExit(handle));
	CallPython([&]() -> void { filesystem.attr("touch")(handle.path, nb::arg("truncate") = true); });
}
bool PythonFilesystem::IsPipe(const string &filename, optional_ptr<FileOpener> opener) {
	return false;
}
idx_t PythonFilesystem::SeekPosition(FileHandle &handle) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	PyUtil::QueryGil gil(nullptr, PythonFileHandle::OpenedDuringExit(handle));
	return CallPython([&]() -> idx_t { return nb::cast<idx_t>(PythonFileHandle::GetHandle(handle).attr("tell")()); });
}
} // namespace duckdb

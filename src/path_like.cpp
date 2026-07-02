#include "duckdb_python/path_like.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/pyfilesystem.hpp"
#include "duckdb_python/filesystem_object.hpp"
#include "duckdb/common/optional_ptr.hpp"

namespace duckdb {

struct PathLikeProcessor {
public:
	explicit PathLikeProcessor(DuckDBPyConnection &connection) : connection(connection) {
	}

public:
	void AddFile(const nb::object &object);
	PathLike Finalize();

protected:
	ModifiedMemoryFileSystem &GetFS() {
		if (!object_store) {
			object_store = &connection.GetObjectFileSystem();
		}
		return *object_store;
	}

public:
	DuckDBPyConnection &connection;
	optional_ptr<ModifiedMemoryFileSystem> object_store;
	// The list containing every file
	vector<string> all_files;
	// The list of files that are registered in the object_store;
	vector<string> fs_files;
};

bool TryDecodePath(const nb::object &object, string &result) {
	if (nb::isinstance<nb::str>(object)) {
		result = nb::cast<string>(object);
		return true;
	}
	if (nb::isinstance<nb::bytes>(object) || nb::hasattr(object, "__fspath__")) {
		// A bytes path or an os.PathLike object (e.g. pathlib.Path) - decode it to a string
		auto fsdecode = nb::module_::import_("os").attr("fsdecode");
		result = nb::cast<string>(fsdecode(object));
		return true;
	}
	return false;
}

string PathToString(const nb::object &object) {
	string result;
	if (!TryDecodePath(object, result)) {
		throw InvalidInputException("Expected a str, bytes, or os.PathLike object for the file path, not '%s'",
		                            Py_TYPE(object.ptr())->tp_name);
	}
	return result;
}

void PathLikeProcessor::AddFile(const nb::object &object) {
	string decoded;
	if (TryDecodePath(object, decoded)) {
		all_files.push_back(std::move(decoded));
		return;
	}
	// This is (assumed to be) a file-like object
	auto generated_name =
	    StringUtil::Format("%s://%s", "DUCKDB_INTERNAL_OBJECTSTORE", StringUtil::GenerateRandomName());
	all_files.push_back(generated_name);
	fs_files.push_back(generated_name);

	auto &fs = GetFS();
	fs.attr("add_file")(object, generated_name);
}

PathLike PathLikeProcessor::Finalize() {
	PathLike result;

	if (all_files.empty()) {
		throw InvalidInputException("Please provide a non-empty list of paths or file-like objects");
	}
	result.files = std::move(all_files);

	if (fs_files.empty()) {
		// No file-like objects were registered in the filesystem
		// no need to make a dependency
		return result;
	}

	// Create the dependency, which contains the logic to clean up the files in its destructor
	auto &fs = GetFS();
	auto dependency = make_uniq<ExternalDependency>();
	auto dependency_item = PythonDependencyItem::Create(make_uniq<FileSystemObject>(fs, std::move(fs_files)));
	dependency->AddDependency("file_handles", std::move(dependency_item));
	result.dependency = std::move(dependency);
	return result;
}

PathLike PathLike::Create(const nb::object &object, DuckDBPyConnection &connection) {
	PathLikeProcessor processor(connection);
	if (nb::isinstance<nb::list>(object)) {
		auto list = nb::list(object);
		for (auto item : list) { // nanobind list iteration yields temporary handles; bind by value (cheap handle)
			processor.AddFile(nb::borrow<nb::object>(item));
		}
	} else {
		// Single object
		processor.AddFile(object);
	}

	return processor.Finalize();
}

} // namespace duckdb

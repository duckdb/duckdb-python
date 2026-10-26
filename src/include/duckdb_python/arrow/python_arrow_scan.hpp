#pragma once

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/registered_py_object.hpp"

#include <type_traits>

namespace duckdb {

struct CachedArrowSchema;

//! How a source reached the scan, which decides how error texts name it
enum class ArrowSourceKind : uint8_t {
	//! Through register(), under its view name
	REGISTERED,
	//! Through the replacement scan, under the name of the Python variable
	VARIABLE,
	//! Through from_arrow, with no name
	NAMELESS
};

//! The words error texts use for a source: "object", "source" or "stream" as `noun` says, by its name and kind
string Subject(const string &name, ArrowSourceKind kind, const char *noun);

//! Carries a source from duckdb._sources to python_arrow_scan through the bind input, so SQL text can never name it
struct PythonArrowSourceInfo : public TableFunctionInfo {
	//! Call with the GIL held
	PythonArrowSourceInfo(nb::object source_p, string name_p, ArrowSourceKind kind_p)
	    : source(std::move(source_p)), name(std::move(name_p)), kind(kind_p) {
	}
	PyObjectHolder source;
	//! The name the source was registered or found under, for error texts; empty when it has none
	const string name;
	const ArrowSourceKind kind;
	//! The schema a schema hook gave, kept for the life of a source whose schema_is_stable is True; set and read with
	//! the GIL held
	shared_ptr<CachedArrowSchema> schema;
};

void RegisterPythonArrowScan(DatabaseInstance &db);

//! Registers the atexit hook after which no scan calls into a source, and a fork handler that drops from a child the
//! calls of threads the child does not have; call once, with the GIL held, as the module initializes, since exit
//! handlers run in reverse order and every handler registered after that may still scan
void WatchInterpreterExit();

//! A scan of `source` through python_arrow_scan, keeping the source alive with the reference; call with the GIL held
unique_ptr<TableFunctionRef> CreatePythonArrowScanRef(nb::object source, const string &name,
                                                      ArrowSourceKind kind = ArrowSourceKind::REGISTERED);

//! The struct a capsule named `kind` carries, still owned by the capsule; refuses a capsule of another kind and a
//! released one; call with the GIL held
template <class T>
T &InCapsule(nb::handle capsule, const char *kind, const string &subject) {
	if (!PyCapsule_IsValid(capsule.ptr(), kind)) {
		string found = "something else";
		if (PyCapsule_CheckExact(capsule.ptr())) {
			auto name = PyCapsule_GetName(capsule.ptr());
			found = name ? StringUtil::Format("a '%s' capsule", name) : "an unnamed capsule";
		}
		PyErr_Clear();
		throw InvalidInputException("%s did not export an '%s' capsule but %s", subject, kind, found);
	}
	auto held = static_cast<T *>(PyCapsule_GetPointer(capsule.ptr(), kind));
	if (!held || !held->release) {
		PyErr_Clear();
		if (std::is_same<T, ArrowArrayStream>::value) {
			throw InvalidInputException("This ArrowArrayStream has already been consumed and cannot be scanned again.");
		}
		throw InvalidInputException("the '%s' capsule of %s is released already", kind, subject);
	}
	return *held;
}

} // namespace duckdb

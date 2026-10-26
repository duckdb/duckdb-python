#include "duckdb_python/pyutil.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"

namespace duckdb {

#ifndef Py_LIMITED_API
bool PyUtil::GilCheck() {
	// Not PyGILState_Check, which answers true on every thread once any subinterpreter has been created.
#if PY_VERSION_HEX >= 0x030D0000
	return PyThreadState_GetUnchecked() != nullptr;
#elif PY_VERSION_HEX >= 0x030C0000
	return _PyThreadState_UncheckedGet() != nullptr;
#else
	// Before 3.12 the current thread state is process wide, so it must also be this thread's own.
	auto current = _PyThreadState_UncheckedGet();
	return current && current == PyGILState_GetThisThreadState();
#endif
}
#endif

bool PyUtil::IsListLike(nb::handle obj) {
	if (nb::isinstance<nb::str>(obj) || nb::isinstance<nb::bytes>(obj)) {
		return false;
	}
	if (IsDictLike(obj)) {
		return false;
	}
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto iterable = import_cache.collections.abc.Iterable();
	return IsInstance(obj, iterable);
}

bool PyUtil::IsDictLike(nb::handle obj) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto mapping = import_cache.collections.abc.Mapping();
	return IsInstance(obj, mapping);
}

} // namespace duckdb

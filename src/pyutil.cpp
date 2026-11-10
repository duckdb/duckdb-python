#include "duckdb_python/pyutil.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb_python/engine/connection.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_conversion.hpp"

namespace duckdb {

bool PyUtil::QueryGil::Admit(optional_ptr<ClientContext> context, bool opened_during_exit) {
	// CPython clears this in the statement after it marks the interpreter finalizing, and unlike Py_IsFinalizing it is
	// in the stable ABI.
	const bool finalizing = !Py_IsInitialized();
	if (!engine::Exiting()) {
		// Without the exit hook no thread is known to be the finalizing one, so none is let in.
		if (finalizing) {
			ThrowShuttingDown();
		}
		return false;
	}
	// The thread that ran the exit hook runs the later exit handlers and then finalizes, and CPython still lets it in;
	// what it opens while finalizing must not let other threads in.
	if (engine::OnExitThread()) {
		return !finalizing;
	}
	if (finalizing ||
	    !(opened_during_exit || (context && engine::ConnectionState::Get(*context)->AdmittedDuringExit()))) {
		ThrowShuttingDown();
	}
	return true;
}

void PyUtil::QueryGil::ThrowShuttingDown() {
	throw InvalidInputException("this query cannot run Python code: the Python interpreter is shutting down");
}

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

std::string PyUtil::CastToString(nb::handle obj) {
	if (nb::bytes::check_(obj)) {
		return nb::cast<std::string>(obj.attr("decode")("utf-8"));
	}
	if (nb::str::check_(obj)) {
		return TransformPythonString(obj);
	}
	return TransformPythonString(nb::str(obj));
}

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

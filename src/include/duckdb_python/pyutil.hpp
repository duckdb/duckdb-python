#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "duckdb/common/types.hpp"
#include "duckdb/common/helper.hpp"
#include <cassert>
#include <string>

namespace nb = nanobind;

namespace duckdb {

// Python interop helpers: raw CPython accessors plus duckdb extensions over nanobind (guarded isinstance,
// lenient string coercion, immutable-tuple builder, GIL and collection predicates). Self-contained on
// nanobind so the umbrella can include it; do not pull the umbrella back in here.
struct PyUtil {
	// isinstance(obj, type) with a null-type guard: an un-imported optional module yields a null type handle,
	// for which we return false. nanobind's isinstance(obj, type) would raise instead.
	static bool IsInstance(nb::handle obj, nb::handle type) {
		if (type.ptr() == nullptr) {
			return false;
		}
		const auto result = PyObject_IsInstance(obj.ptr(), type.ptr());
		if (result == -1) {
			throw nb::python_error();
		}
		return result != 0;
	}

	// Whether obj has the attribute. Unlike PyObject_HasAttr, which takes any exception for a missing attribute, only
	// an AttributeError means missing; any other exception propagates as nb::python_error.
	static bool HasAttribute(nb::handle obj, const char *name) {
		auto value = nb::steal(PyObject_GetAttrString(obj.ptr(), name));
		if (value.is_valid()) {
			return true;
		}
		if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
			PyErr_Clear();
			return false;
		}
		throw nb::python_error();
	}

	// Lenient string conversion: str as is, bytes UTF-8 decoded, anything else via str().
	// nanobind's cast<std::string> rejects bytes/scalars. For identifier/param-key/separator sites.
	static std::string CastToString(nb::handle obj) {
		if (nb::bytes::check_(obj)) {
			return nb::cast<std::string>(obj.attr("decode")("utf-8"));
		}
		if (nb::str::check_(obj)) {
			return nb::cast<std::string>(obj);
		}
		return nb::cast<std::string>(nb::str(obj));
	}

#ifndef Py_LIMITED_API
	// Whether this thread holds the GIL. The stable ABI has no way to ask, so the asserts below exist only in
	// full-API builds, where debugging happens.
	static bool GilCheck();
#endif

	// Releases the GIL for the scope when this thread holds it, and leaves a thread without it alone. Stream
	// callbacks arrive either way, and ending a query waits for tasks that may need the GIL.
	class GilReleaseIfHeld {
	public:
		GilReleaseIfHeld() {
#if PY_VERSION_HEX >= 0x030C0000
			// From 3.12 this detaches the thread's state and releases the GIL, or returns null for a thread without
			// one; before 3.12 it only swapped a process-wide pointer
			saved = PyThreadState_Swap(nullptr);
#elif defined(Py_LIMITED_API)
#error "the stable ABI build starts at CPython 3.12, where PyThreadState_Swap releases the GIL"
#else
			saved = GilCheck() ? PyEval_SaveThread() : nullptr;
#endif
		}
		~GilReleaseIfHeld() {
			if (!saved) {
				return;
			}
#if PY_VERSION_HEX >= 0x030C0000
			PyThreadState_Swap(saved);
#else
			PyEval_RestoreThread(saved);
#endif
		}
		GilReleaseIfHeld(const GilReleaseIfHeld &) = delete;
		GilReleaseIfHeld &operator=(const GilReleaseIfHeld &) = delete;

	private:
		PyThreadState *saved;
	};

	// Collection predicates consulting the connection's ImportCache (collections.abc Iterable/Mapping).
	static bool IsListLike(nb::handle obj);
	static bool IsDictLike(nb::handle obj);

	// Fills a fixed-size immutable nb::tuple via PyTuple_SetItem (cheaper than a list then a copy).
	// Fill every slot with append()/set_item(), then take().
	class TupleBuilder {
	public:
		explicit TupleBuilder(size_t size)
		    : tuple_(nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(size)))), size_(size) {
		}
		void append(nb::object item) {
			assert(index_ < size_);
			set_item(index_++, std::move(item));
		}
		void set_item(size_t index, nb::object item) {
			assert(index < size_);
			if (PyTuple_SetItem(tuple_.ptr(), static_cast<Py_ssize_t>(index), item.release().ptr()) != 0) {
				throw nb::python_error();
			}
		}
		size_t size() const {
			return size_;
		}
		nb::tuple take() {
			return std::move(tuple_);
		}

	private:
		nb::tuple tuple_;
		size_t size_;
		size_t index_ = 0;
	};
};

} // namespace duckdb

#ifdef Py_LIMITED_API
#define DUCKDB_PY_ASSERT_GIL_HELD()     ((void)0)
#define DUCKDB_PY_ASSERT_GIL_RELEASED() ((void)0)
#else
#define DUCKDB_PY_ASSERT_GIL_HELD()     D_ASSERT(duckdb::PyUtil::GilCheck())
#define DUCKDB_PY_ASSERT_GIL_RELEASED() D_ASSERT(!duckdb::PyUtil::GilCheck())
#endif

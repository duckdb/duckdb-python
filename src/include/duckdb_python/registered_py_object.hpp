//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/registered_py_object.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb_python/nb/casters.hpp"

namespace duckdb {

//! Any engine thread may drop the reference, possibly while the interpreter is tearing down, in
//! which case it is leaked rather than touched.
class PyObjectHolder {
public:
	PyObjectHolder() = default;
	//! Call with the GIL held
	explicit PyObjectHolder(nb::object obj_p) : obj(std::move(obj_p)) {
	}
	PyObjectHolder(const PyObjectHolder &) = delete;
	PyObjectHolder &operator=(const PyObjectHolder &) = delete;
	PyObjectHolder(PyObjectHolder &&) = default;
	//! Assignment would drop the previous reference outside the destructor's guard
	PyObjectHolder &operator=(PyObjectHolder &&) = delete;
	~PyObjectHolder() {
		if (nb::detail::cleanup_guard guard {}) {
			obj = nb::object();
		} else {
			obj.release();
		}
	}

	nb::object obj;
};

class RegisteredObject : public PyObjectHolder {
public:
	explicit RegisteredObject(nb::object obj_p) : PyObjectHolder(std::move(obj_p)) {
	}
	virtual ~RegisteredObject() = default;
};

} // namespace duckdb

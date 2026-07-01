//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/numpy/numpy_array.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb.hpp"

namespace duckdb {

//! Thin façade over pybind11's `py::array`.
//!
//! This class is the SINGLE place in the codebase that names `py::array` as the
//! underlying numpy-array representation. A future migration to nanobind's
//! `nb::ndarray` should only require changing the member type and the handful of
//! small methods defined here -- every call site goes through this wrapper
//! instead of touching `py::array` directly.
//!
//! For operations that don't (yet) have a first-class method on the façade
//! (Python attribute access via `.attr(...)`, iteration, resizing, handing the
//! array back to Python, ...) use `GetArray()` to reach the underlying object.
class NumpyArray {
public:
	NumpyArray() = default;
	//! Wrap an existing numpy array. A `py::object` argument is implicitly
	//! converted to a `py::array` (np.asarray semantics), matching the behaviour
	//! the call sites relied on before this façade existed.
	explicit NumpyArray(py::array arr) : array(std::move(arr)) {
	}

	NumpyArray(NumpyArray &&) = default;
	NumpyArray &operator=(NumpyArray &&) = default;
	NumpyArray(const NumpyArray &) = default;
	NumpyArray &operator=(const NumpyArray &) = default;

public:
	//! Allocate a fresh, contiguous 1-D numpy array of `count` elements with the
	//! given dtype.
	static NumpyArray Allocate(const py::dtype &dtype, idx_t count) {
		return NumpyArray(py::array(py::dtype(dtype), count));
	}

	//! Produce a numpy array from an arbitrary Python object (np.asarray semantics).
	static NumpyArray FromObject(py::object obj) {
		return NumpyArray(py::array(std::move(obj)));
	}

	//! Read-only pointer to the underlying data buffer (wraps `py::array::data()`).
	const void *Data() const {
		return array.data();
	}

	//! Mutable pointer to the underlying data buffer (wraps `py::array::mutable_data()`).
	void *MutableData() {
		return array.mutable_data();
	}

	//! Access the underlying array, e.g. for `.attr(...)` calls, iteration, or to
	//! hand it back to Python.
	py::array &GetArray() {
		return array;
	}
	const py::array &GetArray() const {
		return array;
	}

private:
	//! The single data member -- the one spot that later becomes `nb::ndarray`.
	py::array array;
};

} // namespace duckdb

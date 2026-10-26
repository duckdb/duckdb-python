//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/python_object_container.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/helper.hpp"

namespace duckdb {

//! Every Python Object Must be created through our container
//! The Container ensures that the GIL is HOLD on Python Object Construction/Destruction/Modification
class PythonObjectContainer {
public:
	PythonObjectContainer() {
	}

	~PythonObjectContainer() {
		if (nb::detail::cleanup_guard guard {}) {
			py_obj.clear();
		} else {
			for (auto &obj : py_obj) {
				obj.release();
			}
		}
	}

	void Push(nb::object &&obj) {
		nb::gil_scoped_acquire gil;
		PushInternal(std::move(obj));
	}

	const nb::object &LastAddedObject() {
		D_ASSERT(!py_obj.empty());
		return py_obj.back();
	}

private:
	void PushInternal(nb::object &&obj) {
		py_obj.emplace_back(obj);
	}

	vector<nb::object> py_obj;
};
} // namespace duckdb

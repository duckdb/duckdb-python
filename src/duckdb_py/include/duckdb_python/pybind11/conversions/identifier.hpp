#pragma once
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/common/identifier.hpp"

namespace py = pybind11;

namespace PYBIND11_NAMESPACE {
namespace detail {
template <>
class type_caster<duckdb::Identifier> {
	PYBIND11_TYPE_CASTER(duckdb::Identifier, const_name("str"));

	// Python str -> Identifier
	bool load(handle src, bool) {
		if (!PyUnicode_Check(src.ptr())) {
			return false;
		}
		value = duckdb::Identifier(src.cast<std::string>());
		return true;
	}

	// Identifier -> Python str
	static handle cast(const duckdb::Identifier &id, return_value_policy, handle) {
		auto &str_value = id.GetIdentifierName();
		return PyUnicode_FromStringAndSize(str_value.data(), py::ssize_t(str_value.size()));
	}
};
} // namespace detail
} // namespace PYBIND11_NAMESPACE
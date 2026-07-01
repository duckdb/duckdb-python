#pragma once

#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb/common/helper.hpp"

using duckdb::DuckDBPyConnection;

namespace py = pybind11;

namespace PYBIND11_NAMESPACE {
namespace detail {

// NANOBIND PORTING NOTE (None handling):
// This caster maps a Python None (or an omitted `connection=None` argument) to the module-level default
// connection. It works under pybind11 because pybind11 forwards None into a holder/pointer argument's caster
// `load()` by default (argument_record.none defaults to true). nanobind inverts this: it REJECTS None for
// bound-type (shared_ptr / pointer) arguments BEFORE the caster runs, unless the binding annotates the argument
// with `.none()`. So the eventual nanobind port must (1) keep this None -> DefaultConnection() branch AND
// (2) add `.none()` to every `connection` argument that currently defaults to `py::none()` (see
// NANOBIND_NONE_AUDIT.md -- 81 sites in duckdb_python.cpp). Object-family arguments (py::object / Optional<T>)
// do not need this annotation; their value casters accept None directly.
template <>
class type_caster<std::shared_ptr<DuckDBPyConnection>>
    : public copyable_holder_caster<DuckDBPyConnection, std::shared_ptr<DuckDBPyConnection>> {
	using type = DuckDBPyConnection;
	using holder_caster = copyable_holder_caster<DuckDBPyConnection, std::shared_ptr<DuckDBPyConnection>>;
	// This is used to generate documentation on duckdb-web
	PYBIND11_TYPE_CASTER(std::shared_ptr<type>, const_name("duckdb.DuckDBPyConnection"));

	bool load(handle src, bool convert) {
		if (py::none().is(src)) {
			value = DuckDBPyConnection::DefaultConnection();
			return true;
		}
		if (!holder_caster::load(src, convert)) {
			return false;
		}
		// pybind11's std::shared_ptr holder_caster (smart_holder bakein) has no `holder` member like the
		// generic template did for duckdb::shared_ptr; extract the loaded pointer via its conversion operator.
		value = static_cast<std::shared_ptr<type> &>(static_cast<holder_caster &>(*this));
		return true;
	}

	static handle cast(std::shared_ptr<type> base, return_value_policy rvp, handle h) {
		return holder_caster::cast(base, rvp, h);
	}
};

template <>
struct is_holder_type<DuckDBPyConnection, std::shared_ptr<DuckDBPyConnection>> : std::true_type {};

} // namespace detail
} // namespace PYBIND11_NAMESPACE

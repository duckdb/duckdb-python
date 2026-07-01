#pragma once

#include <pybind11/pybind11.h>
#include <cstdint>
#include <string>

//===----------------------------------------------------------------------===//
// Reusable pybind11 type_caster macros for "string / integer or enum" arguments
//===----------------------------------------------------------------------===//
//
// Several DuckDB enums are exposed to Python so that a binding parameter typed as
// the enum also accepts a string (and, for most, an integer) naming one of its
// values, while still accepting an actual registered enum instance. Every one of
// these casters had an identical shape:
//
//   - if the source is a Python str  -> value = <Enum>FromString(...)
//   - if the source is a Python int  -> value = <Enum>FromInteger(...)   (optional)
//   - otherwise delegate to a *local* type_caster_base<Enum> for the registered
//     enum instance.
//
// The macros below collapse that boilerplate into a single invocation per enum so
// the eventual nanobind port is a one-place change. Behavior is intentionally
// identical to the hand-written casters they replace.
//
// IMPORTANT (matches the original per-file notes): these casters own their value
// via PYBIND11_TYPE_CASTER and delegate ONLY the registered-instance case to a
// local base caster -- they do NOT inherit type_caster_base. Inheriting the base
// while also writing custom branches is what historically made a caster accept
// str XOR the enum depending on include visibility. Each specialization must be
// visible in every TU that converts the type (they live under the universally
// included pybind_wrapper.hpp umbrella), otherwise it is UB.
//
// Invoke these macros at GLOBAL scope (outside any namespace); each expands to a
// full `namespace pybind11 { namespace detail { ... } }` specialization. Pass
// fully-qualified names (e.g. duckdb::ExplainTypeFromString) for the conversion
// functions and the enum type.

//! str + int + registered-enum form.
#define DUCKDB_PY_ENUM_STRING_INT_CASTER(EnumType, FromStringFn, FromIntegerFn, NameLiteral)                           \
	namespace PYBIND11_NAMESPACE {                                                                                     \
	namespace detail {                                                                                                 \
	template <>                                                                                                        \
	struct type_caster<EnumType> {                                                                                     \
		PYBIND11_TYPE_CASTER(EnumType, const_name(NameLiteral));                                                       \
                                                                                                                       \
		bool load(handle src, bool convert) {                                                                          \
			if (isinstance<str>(src)) {                                                                                \
				value = FromStringFn(src.cast<std::string>());                                                         \
				return true;                                                                                           \
			}                                                                                                          \
			if (isinstance<int_>(src)) {                                                                               \
				value = FromIntegerFn(src.cast<int64_t>());                                                            \
				return true;                                                                                           \
			}                                                                                                          \
			type_caster_base<EnumType> base;                                                                           \
			if (!base.load(src, convert)) {                                                                            \
				return false;                                                                                          \
			}                                                                                                          \
			value = *static_cast<EnumType *>(base);                                                                    \
			return true;                                                                                               \
		}                                                                                                              \
                                                                                                                       \
		static handle cast(EnumType src, return_value_policy policy, handle parent) {                                  \
			return type_caster_base<EnumType>::cast(src, policy, parent);                                              \
		}                                                                                                              \
	};                                                                                                                 \
	} /* namespace detail */                                                                                           \
	} /* namespace PYBIND11_NAMESPACE */

//! str + registered-enum form (no integer accepted).
#define DUCKDB_PY_ENUM_STRING_CASTER(EnumType, FromStringFn, NameLiteral)                                              \
	namespace PYBIND11_NAMESPACE {                                                                                     \
	namespace detail {                                                                                                 \
	template <>                                                                                                        \
	struct type_caster<EnumType> {                                                                                     \
		PYBIND11_TYPE_CASTER(EnumType, const_name(NameLiteral));                                                       \
                                                                                                                       \
		bool load(handle src, bool convert) {                                                                          \
			if (isinstance<str>(src)) {                                                                                \
				value = FromStringFn(src.cast<std::string>());                                                         \
				return true;                                                                                           \
			}                                                                                                          \
			type_caster_base<EnumType> base;                                                                           \
			if (!base.load(src, convert)) {                                                                            \
				return false;                                                                                          \
			}                                                                                                          \
			value = *static_cast<EnumType *>(base);                                                                    \
			return true;                                                                                               \
		}                                                                                                              \
                                                                                                                       \
		static handle cast(EnumType src, return_value_policy policy, handle parent) {                                  \
			return type_caster_base<EnumType>::cast(src, policy, parent);                                              \
		}                                                                                                              \
	};                                                                                                                 \
	} /* namespace detail */                                                                                           \
	} /* namespace PYBIND11_NAMESPACE */

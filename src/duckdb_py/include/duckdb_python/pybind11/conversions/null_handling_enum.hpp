#pragma once

#include "duckdb/function/function.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/pybind11/conversions/enum_string_caster.hpp"

namespace duckdb {

inline FunctionNullHandling FunctionNullHandlingFromString(const string &type) {
	auto ltype = StringUtil::Lower(type);
	if (ltype.empty() || ltype == "default") {
		return FunctionNullHandling::DEFAULT_NULL_HANDLING;
	}
	if (ltype == "special") {
		return FunctionNullHandling::SPECIAL_HANDLING;
	}
	throw InvalidInputException("'%s' is not a recognized type for 'null_handling'", type);
}

inline FunctionNullHandling FunctionNullHandlingFromInteger(int64_t value) {
	if (value == 0) {
		return FunctionNullHandling::DEFAULT_NULL_HANDLING;
	}
	if (value == 1) {
		return FunctionNullHandling::SPECIAL_HANDLING;
	}
	throw InvalidInputException("'%d' is not a recognized type for 'null_handling'", value);
}

} // namespace duckdb

//! See enum_string_caster.hpp for why this owns its value and delegates the enum case to a local base caster
//! instead of inheriting type_caster_base. Must stay visible in every TU (included from pybind_wrapper.hpp).
DUCKDB_PY_ENUM_STRING_INT_CASTER(duckdb::FunctionNullHandling, duckdb::FunctionNullHandlingFromString,
                                 duckdb::FunctionNullHandlingFromInteger, "FunctionNullHandling")

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/pybind11/conversions/enum_string_caster.hpp"

namespace duckdb {

enum class PythonExceptionHandling : uint8_t { FORWARD_ERROR, RETURN_NULL };

inline PythonExceptionHandling PythonExceptionHandlingFromString(const string &type) {
	auto ltype = StringUtil::Lower(type);
	if (ltype.empty() || ltype == "default") {
		return PythonExceptionHandling::FORWARD_ERROR;
	}
	if (ltype == "return_null") {
		return PythonExceptionHandling::RETURN_NULL;
	}
	throw InvalidInputException("'%s' is not a recognized type for 'exception_handling'", type);
}

inline PythonExceptionHandling PythonExceptionHandlingFromInteger(int64_t value) {
	if (value == 0) {
		return PythonExceptionHandling::FORWARD_ERROR;
	}
	if (value == 1) {
		return PythonExceptionHandling::RETURN_NULL;
	}
	throw InvalidInputException("'%d' is not a recognized type for 'exception_handling'", value);
}

} // namespace duckdb

//! See enum_string_caster.hpp for the rationale (composition over inheritance, umbrella visibility).
DUCKDB_PY_ENUM_STRING_INT_CASTER(duckdb::PythonExceptionHandling, duckdb::PythonExceptionHandlingFromString,
                                 duckdb::PythonExceptionHandlingFromInteger, "PythonExceptionHandling")

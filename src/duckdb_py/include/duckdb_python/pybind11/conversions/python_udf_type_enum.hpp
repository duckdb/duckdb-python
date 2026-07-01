#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/pybind11/conversions/enum_string_caster.hpp"

namespace duckdb {

enum class PythonUDFType : uint8_t { NATIVE, ARROW };

inline PythonUDFType PythonUDFTypeFromString(const string &type) {
	auto ltype = StringUtil::Lower(type);
	if (ltype.empty() || ltype == "default" || ltype == "native") {
		return PythonUDFType::NATIVE;
	}
	if (ltype == "arrow") {
		return PythonUDFType::ARROW;
	}
	throw InvalidInputException("'%s' is not a recognized type for 'udf_type'", type);
}

inline PythonUDFType PythonUDFTypeFromInteger(int64_t value) {
	if (value == 0) {
		return PythonUDFType::NATIVE;
	}
	if (value == 1) {
		return PythonUDFType::ARROW;
	}
	throw InvalidInputException("'%d' is not a recognized type for 'udf_type'", value);
}

} // namespace duckdb

//! Accepts the registered PythonUDFType enum, or a string / integer naming one. See enum_string_caster.hpp for
//! the rationale (this owns its value via PYBIND11_TYPE_CASTER and delegates only the registered-enum case to a
//! local base caster instead of inheriting type_caster_base). Keeping the binding parameter typed as the enum
//! preserves the type + default in help()/stubs.
DUCKDB_PY_ENUM_STRING_INT_CASTER(duckdb::PythonUDFType, duckdb::PythonUDFTypeFromString,
                                 duckdb::PythonUDFTypeFromInteger, "PythonUDFType")

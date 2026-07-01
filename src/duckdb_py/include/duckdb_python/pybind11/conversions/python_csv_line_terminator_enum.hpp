#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/pybind11/conversions/enum_string_caster.hpp"

namespace duckdb {

struct PythonCSVLineTerminator {
public:
	PythonCSVLineTerminator() = delete;
	enum class Type { LINE_FEED, CARRIAGE_RETURN_LINE_FEED };

public:
	static Type FromString(const string &str) {
		if (str == "\\n") {
			return Type::LINE_FEED;
		}
		if (str == "\\r\\n") {
			return Type::CARRIAGE_RETURN_LINE_FEED;
		}
		if (str == "LINE_FEED") {
			return Type::LINE_FEED;
		}
		if (str == "CARRIAGE_RETURN_LINE_FEED") {
			return Type::CARRIAGE_RETURN_LINE_FEED;
		}
		throw InvalidInputException("'%s' is not a recognized type for 'lineterminator'", str);
	}
	static string ToString(Type type) {
		switch (type) {
		case Type::LINE_FEED:
			return "\\n";
		case Type::CARRIAGE_RETURN_LINE_FEED:
			return "\\r\\n";
		default:
			throw NotImplementedException("No conversion for PythonCSVLineTerminator enum to string");
		}
	}
};

} // namespace duckdb

//! See enum_string_caster.hpp for the rationale (composition over inheritance, umbrella visibility).
//! Only a string or the enum itself are accepted (no integer form).
DUCKDB_PY_ENUM_STRING_CASTER(duckdb::PythonCSVLineTerminator::Type, duckdb::PythonCSVLineTerminator::FromString,
                             "CSVLineTerminator")

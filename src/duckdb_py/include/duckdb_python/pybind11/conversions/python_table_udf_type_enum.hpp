#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

using duckdb::InvalidInputException;
using duckdb::string;
using duckdb::StringUtil;

namespace duckdb {

enum class PythonTableUDFType : uint8_t { TUPLES, ARROW_TABLE };

} // namespace duckdb

using duckdb::PythonTableUDFType;

namespace py = pybind11;

static PythonTableUDFType PythonTableUDFTypeFromString(const string &type) {
	auto ltype = StringUtil::Lower(type);
	if (ltype.empty() || ltype == "tuples") {
		return PythonTableUDFType::TUPLES;
	} else if (ltype == "arrow_table") {
		return PythonTableUDFType::ARROW_TABLE;
	} else {
		throw InvalidInputException("'%s' is not a recognized type for 'tvf_type'", type);
	}
}

static PythonTableUDFType PythonTableUDFTypeFromInteger(int64_t value) {
	if (value == 0) {
		return PythonTableUDFType::TUPLES;
	} else if (value == 1) {
		return PythonTableUDFType::ARROW_TABLE;
	} else {
		throw InvalidInputException("'%d' is not a recognized type for 'tvf_type'", value);
	}
}

namespace PYBIND11_NAMESPACE {
namespace detail {

template <>
struct type_caster<PythonTableUDFType> : public type_caster_base<PythonTableUDFType> {
	using base = type_caster_base<PythonTableUDFType>;
	PythonTableUDFType tmp;

public:
	bool load(handle src, bool convert) {
		if (base::load(src, convert)) {
			return true;
		} else if (py::isinstance<py::str>(src)) {
			tmp = PythonTableUDFTypeFromString(py::str(src));
			value = &tmp;
			return true;
		} else if (py::isinstance<py::int_>(src)) {
			tmp = PythonTableUDFTypeFromInteger(src.cast<int64_t>());
			value = &tmp;
			return true;
		}
		return false;
	}

	static handle cast(PythonTableUDFType src, return_value_policy policy, handle parent) {
		return base::cast(src, policy, parent);
	}
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE
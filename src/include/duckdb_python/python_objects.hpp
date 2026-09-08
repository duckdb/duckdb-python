#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/pyutil.hpp"
#include "duckdb/common/types/time.hpp"
#include "duckdb/common/types/date.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/common/types/interval.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/main/client_properties.hpp"

#include "datetime.h" //from python

/* Backport for Python < 3.10 */
#if PY_VERSION_HEX < 0x030a00a1
#ifndef PyDateTime_TIME_GET_TZINFO
#define PyDateTime_TIME_GET_TZINFO(o) ((((PyDateTime_Time *)o)->hastzinfo) ? ((PyDateTime_Time *)o)->tzinfo : Py_None)
#endif
#ifndef PyDateTime_DATE_GET_TZINFO
#define PyDateTime_DATE_GET_TZINFO(o)                                                                                  \
	((((PyDateTime_DateTime *)o)->hastzinfo) ? ((PyDateTime_DateTime *)o)->tzinfo : Py_None)
#endif
#endif

namespace duckdb {

struct PyDictionary {
public:
	PyDictionary(nb::object dict);
	// These are cached so we don't have to create new objects all the time
	// The CPython API offers PyDict_Keys but that creates a new reference every time, same for values
	nb::object keys;
	nb::object values;
	idx_t len;

public:
	nb::handle operator[](const nb::object &obj) const {
		return PyDict_GetItem(dict.ptr(), obj.ptr());
	}

public:
	string ToString() const {
		return nb::cast<std::string>(nb::str(dict));
	}

private:
	nb::object dict;
};

enum class PyDecimalExponentType {
	EXPONENT_SCALE,    //! Amount of digits after the decimal point
	EXPONENT_POWER,    //! Trailing zeros on the mantissa (positive Decimal exponent)
	EXPONENT_INFINITY, //! Decimal is INFINITY
	EXPONENT_NAN       //! Decimal is NAN
};

struct PyDecimal {

	struct PyDecimalScaleConverter {
		template <typename T, typename = std::enable_if<std::numeric_limits<T>::is_integer, T>>
		static Value Operation(bool signed_value, vector<uint8_t> &digits, uint8_t width, uint8_t scale) {
			T value = 0;
			for (auto it = digits.begin(); it != digits.end(); it++) {
				value = value * 10 + *it;
			}
			if (signed_value) {
				value = -value;
			}
			return Value::DECIMAL(value, width, scale);
		}
	};

public:
	PyDecimal(nb::handle &obj);
	vector<uint8_t> digits;
	bool signed_value = false;

	PyDecimalExponentType exponent_type;
	int32_t exponent_value;

public:
	bool TryGetType(LogicalType &type);
	Value ToDuckValue();

private:
	void SetExponent(nb::handle &exponent);
	nb::handle &obj;
};

struct PyTimeDelta {
public:
	PyTimeDelta(nb::handle &obj);
	int32_t days;
	int32_t seconds;
	int64_t microseconds;

public:
	interval_t ToInterval();

private:
	static int64_t GetDays(nb::handle &obj);
	static int64_t GetSeconds(nb::handle &obj);
	static int64_t GetMicros(nb::handle &obj);
};

struct PyTime {
public:
	PyTime(nb::handle &obj);
	nb::handle &obj;
	int32_t hour;
	int32_t minute;
	int32_t second;
	int32_t microsecond;
	nb::object timezone_obj;

public:
	dtime_t ToDuckTime();
	Value ToDuckValue();

private:
	static int32_t GetHours(nb::handle &obj);
	static int32_t GetMinutes(nb::handle &obj);
	static int32_t GetSeconds(nb::handle &obj);
	static int32_t GetMicros(nb::handle &obj);
	static nb::object GetTZInfo(nb::handle &obj);
};

struct PyDateTime {
public:
	PyDateTime(nb::handle &obj);
	nb::handle &obj;
	int32_t year;
	int32_t month;
	int32_t day;
	int32_t hour;
	int32_t minute;
	int32_t second;
	int32_t micros;
	nb::object tzone_obj;

public:
	timestamp_t ToTimestamp();
	date_t ToDate();
	dtime_t ToDuckTime();
	Value ToDuckValue(const LogicalType &target_type);

public:
	static int32_t GetYears(nb::handle &obj);
	static int32_t GetMonths(nb::handle &obj);
	static int32_t GetDays(nb::handle &obj);
	static int32_t GetHours(nb::handle &obj);
	static int32_t GetMinutes(nb::handle &obj);
	static int32_t GetSeconds(nb::handle &obj);
	static int32_t GetMicros(nb::handle &obj);
	static nb::object GetTZInfo(nb::handle &obj);
};

struct PyDate {
public:
	PyDate(nb::handle &ele);
	int32_t year;
	int32_t month;
	int32_t day;

public:
	date_t ToDate();
	Value ToDuckValue();
};

struct PyTimezone {
public:
	PyTimezone() = delete;

public:
	DUCKDB_API static int32_t GetUTCOffsetSeconds(nb::handle &tzone_obj);
	DUCKDB_API static interval_t GetUTCOffset(nb::handle &datetime, nb::handle &tzone_obj);
};

struct PythonObject {
	static void Initialize();
	static nb::object FromStruct(const Value &value, const LogicalType &id, const ClientProperties &client_properties);
	static nb::object FromValue(const Value &value, const LogicalType &id, const ClientProperties &client_properties);
};

template <class T>
class Optional : public nb::object {
public:
	Optional(const nb::object &o) : nb::object(o, nb::detail::borrow_t {}) {
	}
	using nb::object::object;

public:
	static bool check_(const nb::handle &object) {
		return object.is_none() || nb::isinstance<T>(object);
	}
};

class FileLikeObject : public nb::object {
public:
	FileLikeObject(const nb::object &o) : nb::object(o, nb::detail::borrow_t {}) {
	}
	using nb::object::object;

public:
	static bool check_(const nb::handle &object) {
		return duckdb::PyUtil::IsInstance(object, nb::module_::import_("io").attr("IOBase"));
	}
};

} // namespace duckdb

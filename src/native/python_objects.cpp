#include "duckdb_python/python_objects.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/decimal.hpp"
#include "duckdb/common/types/bit.hpp"
#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/types/bignum.hpp"
#include "duckdb/function/to_interval.hpp"

#include "datetime.h" // Python datetime initialize #1

#include <duckdb/common/types/variant.hpp>
#include <duckdb/function/scalar/variant_utils.hpp>

namespace duckdb {

PyDictionary::PyDictionary(nb::object dict) {
	keys = nb::list(dict.attr("keys")());
	values = nb::list(dict.attr("values")());
	len = nb::len(keys);
	this->dict = std::move(dict);
}

PyTimeDelta::PyTimeDelta(nb::handle &obj) {
	days = PyTimeDelta::GetDays(obj);
	seconds = PyTimeDelta::GetSeconds(obj);
	microseconds = PyTimeDelta::GetMicros(obj);
}

interval_t PyTimeDelta::ToInterval() {
	interval_t result;

	auto micros_interval = Interval::FromMicro(microseconds);
	auto days_interval = interval_t {/*months = */ 0,
	                                 /*days = */ days,
	                                 /*micros = */ 0};
	auto seconds_interval = ToSecondsOperator::Operation<int64_t, interval_t>(seconds);

	result = AddOperator::Operation<interval_t, interval_t, interval_t>(micros_interval, days_interval);
	result = AddOperator::Operation<interval_t, interval_t, interval_t>(result, seconds_interval);
	return result;
}

int64_t PyTimeDelta::GetDays(nb::handle &obj) {
	// nb::object wrap: nb::int_() of a bare .attr() accessor is an ambiguous overload on MSVC.
	return nb::cast<int64_t>(nb::int_(nb::object(obj.attr("days"))));
}

int64_t PyTimeDelta::GetSeconds(nb::handle &obj) {
	return nb::cast<int64_t>(nb::int_(nb::object(obj.attr("seconds"))));
}

int64_t PyTimeDelta::GetMicros(nb::handle &obj) {
	return nb::cast<int64_t>(nb::int_(nb::object(obj.attr("microseconds"))));
}

PyDecimal::PyDecimal(nb::handle &obj) : obj(obj) {
	auto as_tuple = obj.attr("as_tuple")();

	nb::object exponent = as_tuple.attr("exponent");
	SetExponent(exponent);

	auto sign = nb::cast<int8_t>(as_tuple.attr("sign"));
	signed_value = sign != 0;

	nb::object decimal_digits = as_tuple.attr("digits");
	auto width = nb::len(decimal_digits);
	digits.reserve(width);
	for (auto digit : decimal_digits) {
		digits.push_back(nb::cast<uint8_t>(digit));
	}
}

bool PyDecimal::TryGetType(LogicalType &type) {
	int32_t width = digits.size();

	switch (exponent_type) {
	case PyDecimalExponentType::EXPONENT_SCALE: {
		auto scale = exponent_value;
		if (scale > width) {
			// The value starts with 1 or more zeros, which are optimized out of the 'digits' array
			// 0.001; width=1, exponent=-3
			width = scale + 1; // DECIMAL(4,3) - add 1 for the non-decimal values
		}
		if (width > Decimal::MAX_WIDTH_INT128) {
			type = LogicalType::DOUBLE;
			return true;
		}
		type = LogicalType::DECIMAL(width, scale);
		return true;
	}
	case PyDecimalExponentType::EXPONENT_POWER: {
		// Positive exponent: integer with extra trailing zeros, scale 0.
		if (exponent_value > Decimal::MAX_WIDTH_INT128 || width > Decimal::MAX_WIDTH_INT128 - exponent_value) {
			type = LogicalType::DOUBLE;
			return true;
		}
		width += exponent_value;
		type = LogicalType::DECIMAL(width, 0);
		return true;
	}
	case PyDecimalExponentType::EXPONENT_INFINITY: {
		type = LogicalType::FLOAT;
		return true;
	}
	case PyDecimalExponentType::EXPONENT_NAN: {
		type = LogicalType::FLOAT;
		return true;
	}
	default: // LCOV_EXCL_START
		throw NotImplementedException("case not implemented for type PyDecimalExponentType");
	} // LCOV_EXCL_STOP
}
// LCOV_EXCL_START
static void ExponentNotRecognized() {
	throw NotImplementedException("Failed to convert decimal.Decimal value, exponent type is unknown");
}
// LCOV_EXCL_STOP

void PyDecimal::SetExponent(nb::handle &exponent) {
	if (nb::isinstance<nb::int_>(exponent)) {
		this->exponent_value = nb::cast<int32_t>(exponent);
		if (this->exponent_value >= 0) {
			exponent_type = PyDecimalExponentType::EXPONENT_POWER;
			return;
		}
		exponent_value *= -1;
		exponent_type = PyDecimalExponentType::EXPONENT_SCALE;
		return;
	}
	if (nb::isinstance<nb::str>(exponent)) {
		string exponent_string = nb::cast<std::string>(nb::str(exponent));
		if (exponent_string == "n") {
			exponent_type = PyDecimalExponentType::EXPONENT_NAN;
			return;
		}
		if (exponent_string == "F") {
			exponent_type = PyDecimalExponentType::EXPONENT_INFINITY;
			return;
		}
	}
	// LCOV_EXCL_START
	ExponentNotRecognized();
	// LCOV_EXCL_STOP
}

static bool WidthFitsInDecimal(int32_t width) {
	return width >= 0 && width <= Decimal::MAX_WIDTH_DECIMAL;
}

template <class OP>
Value PyDecimalCastSwitch(PyDecimal &decimal, uint8_t width, uint8_t scale) {
	if (width > DecimalWidth<int64_t>::max) {
		return OP::template Operation<hugeint_t>(decimal.signed_value, decimal.digits, width, scale);
	}
	if (width > DecimalWidth<int32_t>::max) {
		return OP::template Operation<int64_t>(decimal.signed_value, decimal.digits, width, scale);
	}
	if (width > DecimalWidth<int16_t>::max) {
		return OP::template Operation<int32_t>(decimal.signed_value, decimal.digits, width, scale);
	}
	return OP::template Operation<int16_t>(decimal.signed_value, decimal.digits, width, scale);
}

// Wont fit in a DECIMAL, fall back to DOUBLE
static Value CastToDouble(nb::handle &obj) {
	string converted = nb::cast<std::string>(nb::str(obj));
	string_t decimal_string(converted);
	double double_val;
	bool try_cast = TryCast::Operation<string_t, double>(decimal_string, double_val, true);
	(void)try_cast;
	D_ASSERT(try_cast);
	return Value::DOUBLE(double_val);
}

Value PyDecimal::ToDuckValue() {
	int32_t width = digits.size();
	if (!WidthFitsInDecimal(width)) {
		return CastToDouble(obj);
	}
	switch (exponent_type) {
	case PyDecimalExponentType::EXPONENT_SCALE: {
		uint8_t scale = exponent_value;
		D_ASSERT(WidthFitsInDecimal(width));
		if (scale > width) {
			// Values like '0.001'
			width = scale + 1; // leave 1 room for the non-decimal value
		}
		if (!WidthFitsInDecimal(width)) {
			return CastToDouble(obj);
		}
		return PyDecimalCastSwitch<PyDecimalScaleConverter>(*this, width, scale);
	}
	case PyDecimalExponentType::EXPONENT_POWER: {
		// Fold 10^exponent into the mantissa and store scale 0. Using the exponent as a
		// DECIMAL scale cancelled the 10^n multiply and stored only the mantissa (1E+2 -> 1).
		if (exponent_value > Decimal::MAX_WIDTH_DECIMAL || width > Decimal::MAX_WIDTH_DECIMAL - exponent_value) {
			return CastToDouble(obj);
		}
		digits.insert(digits.end(), static_cast<size_t>(exponent_value), static_cast<uint8_t>(0));
		width += exponent_value;
		D_ASSERT(WidthFitsInDecimal(width));
		return PyDecimalCastSwitch<PyDecimalScaleConverter>(*this, width, 0);
	}
	case PyDecimalExponentType::EXPONENT_NAN: {
		return Value::FLOAT(NAN);
	}
	case PyDecimalExponentType::EXPONENT_INFINITY: {
		return Value::FLOAT(INFINITY);
	}
	// LCOV_EXCL_START
	default: {
		throw NotImplementedException("case not implemented for type PyDecimalExponentType");
	} // LCOV_EXCL_STOP
	}
}

PyTime::PyTime(nb::handle &obj) : obj(obj) {
	hour = PyTime::GetHours(obj);          // NOLINT
	minute = PyTime::GetMinutes(obj);      // NOLINT
	second = PyTime::GetSeconds(obj);      // NOLINT
	microsecond = PyTime::GetMicros(obj);  // NOLINT
	timezone_obj = PyTime::GetTZInfo(obj); // NOLINT
}
dtime_t PyTime::ToDuckTime() {
	return Time::FromTime(hour, minute, second, microsecond);
}

Value PyTime::ToDuckValue() {
	auto duckdb_time = this->ToDuckTime();
	if (!nb::none().is(this->timezone_obj)) {
		auto seconds = PyTimezone::GetUTCOffsetSeconds(this->timezone_obj);
		return Value::TIMETZ(dtime_tz_t(duckdb_time, seconds));
	}
	return Value::TIME(duckdb_time);
}

int32_t PyTime::GetHours(nb::handle &obj) {
	return PyDateTime_TIME_GET_HOUR(obj.ptr()); // NOLINT
}

int32_t PyTime::GetMinutes(nb::handle &obj) {
	return PyDateTime_TIME_GET_MINUTE(obj.ptr()); // NOLINT
}

int32_t PyTime::GetSeconds(nb::handle &obj) {
	return PyDateTime_TIME_GET_SECOND(obj.ptr()); // NOLINT
}

int32_t PyTime::GetMicros(nb::handle &obj) {
	return PyDateTime_TIME_GET_MICROSECOND(obj.ptr()); // NOLINT
}

nb::object PyTime::GetTZInfo(nb::handle &obj) {
	// The object returned is borrowed, there is no reference to steal
	return nb::borrow<nb::object>(PyDateTime_TIME_GET_TZINFO(obj.ptr())); // NOLINT
}

interval_t PyTimezone::GetUTCOffset(nb::handle &datetime, nb::handle &tzone_obj) {
	// The datetime object is provided because the utcoffset could be ambiguous
	auto res = tzone_obj.attr("utcoffset")(datetime);
	auto timedelta = PyTimeDelta(res);
	return timedelta.ToInterval();
}

int32_t PyTimezone::GetUTCOffsetSeconds(nb::handle &tzone_obj) {
	// We should be able to use None here, the tzone_obj of a datetime.time should never be ambiguous
	auto res = tzone_obj.attr("utcoffset")(nb::none());
	auto timedelta = PyTimeDelta(res);
	if (timedelta.days != 0) {
		throw InvalidInputException(
		    "Failed to convert 'tzinfo' object, utcoffset returned an invalid timedelta (days)");
	}
	if (timedelta.microseconds != 0) {
		throw InvalidInputException(
		    "Failed to convert 'tzinfo' object, utcoffset returned an invalid timedelta (microseconds)");
	}
	return timedelta.seconds;
}

PyDateTime::PyDateTime(nb::handle &obj) : obj(obj) {
	year = PyDateTime::GetYears(obj);
	month = PyDateTime::GetMonths(obj);
	day = PyDateTime::GetDays(obj);
	hour = PyDateTime::GetHours(obj);
	minute = PyDateTime::GetMinutes(obj);
	second = PyDateTime::GetSeconds(obj);
	micros = PyDateTime::GetMicros(obj);
	tzone_obj = PyDateTime::GetTZInfo(obj);
}

timestamp_t PyDateTime::ToTimestamp() {
	auto date = ToDate();
	auto time = ToDuckTime();
	return Timestamp::FromDatetime(date, time);
}

Value PyDateTime::ToDuckValue(const LogicalType &target_type) {
	auto timestamp = ToTimestamp();
	if (!nb::none().is(tzone_obj)) {
		auto utc_offset = PyTimezone::GetUTCOffset(obj, tzone_obj);
		// Need to subtract the UTC offset, so we invert the interval
		utc_offset = Interval::Invert(utc_offset);
		timestamp = Interval::Add(timestamp, utc_offset);
		return Value::TIMESTAMPTZ(timestamp_tz_t(timestamp));
	}
	switch (target_type.id()) {
	case LogicalTypeId::UNKNOWN:
	case LogicalTypeId::TIMESTAMP: {
		return Value::TIMESTAMP(timestamp);
	}
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
		// Because the 'Time::FromTime' method constructs a regular (usecond) timestamp, this is not compatible with
		// creating sec/ms/ns timestamps
		throw NotImplementedException("Conversion from 'datetime' to type %s is not implemented yet",
		                              target_type.ToString());
	default:
		throw ConversionException("Could not convert 'datetime' to type %s", target_type.ToString());
	}
}

date_t PyDateTime::ToDate() {
	return Date::FromDate(year, month, day);
}
dtime_t PyDateTime::ToDuckTime() {
	return Time::FromTime(hour, minute, second, micros);
}

int32_t PyDateTime::GetYears(nb::handle &obj) {
	return PyDateTime_GET_YEAR(obj.ptr()); // NOLINT
}

int32_t PyDateTime::GetMonths(nb::handle &obj) {
	return PyDateTime_GET_MONTH(obj.ptr()); // NOLINT
}

int32_t PyDateTime::GetDays(nb::handle &obj) {
	return PyDateTime_GET_DAY(obj.ptr()); // NOLINT
}

int32_t PyDateTime::GetHours(nb::handle &obj) {
	return PyDateTime_DATE_GET_HOUR(obj.ptr()); // NOLINT
}

int32_t PyDateTime::GetMinutes(nb::handle &obj) {
	return PyDateTime_DATE_GET_MINUTE(obj.ptr()); // NOLINT
}

int32_t PyDateTime::GetSeconds(nb::handle &obj) {
	return PyDateTime_DATE_GET_SECOND(obj.ptr()); // NOLINT
}

int32_t PyDateTime::GetMicros(nb::handle &obj) {
	return PyDateTime_DATE_GET_MICROSECOND(obj.ptr()); // NOLINT
}

nb::object PyDateTime::GetTZInfo(nb::handle &obj) {
	// The object returned is borrowed, there is no reference to steal
	return nb::borrow<nb::object>(PyDateTime_DATE_GET_TZINFO(obj.ptr())); // NOLINT
}

PyDate::PyDate(nb::handle &ele) {
	year = PyDateTime::GetYears(ele);
	month = PyDateTime::GetMonths(ele);
	day = PyDateTime::GetDays(ele);
}

date_t PyDate::ToDate() {
	return Date::FromDate(year, month, day);
}

Value PyDate::ToDuckValue() {
	auto value = Value::DATE(year, month, day);
	return value;
}

void PythonObject::Initialize() {
	PyDateTime_IMPORT; // NOLINT: Python datetime initialize #2
}

enum class InfinityType : uint8_t { NONE, POSITIVE, NEGATIVE };

InfinityType GetTimestampInfinityType(timestamp_t &timestamp) {
	if (timestamp == timestamp_t::infinity()) {
		return InfinityType::POSITIVE;
	}
	if (timestamp == timestamp_t::ninfinity()) {
		return InfinityType::NEGATIVE;
	}
	return InfinityType::NONE;
}

nb::object PythonObject::FromStruct(const Value &val, const LogicalType &type,
                                    const ClientProperties &client_properties) {
	auto &struct_values = StructValue::GetChildren(val);

	auto &child_types = StructType::GetChildTypes(type);
	if (StructType::IsUnnamed(type)) {
		duckdb::PyUtil::TupleBuilder py_tuple(struct_values.size());
		for (idx_t i = 0; i < struct_values.size(); i++) {
			auto &child_entry = child_types[i];
			D_ASSERT(child_entry.first.empty());
			auto &child_type = child_entry.second;
			py_tuple.append(FromValue(struct_values[i], child_type, client_properties));
		}
		return py_tuple.take();
	} else {
		nb::dict py_struct;
		for (idx_t i = 0; i < struct_values.size(); i++) {
			auto &child_entry = child_types[i];
			auto &child_name = child_entry.first;
			auto &child_type = child_entry.second;
			py_struct[child_name.c_str()] = FromValue(struct_values[i], child_type, client_properties);
		}
		return std::move(py_struct);
	}
}

static bool KeyIsHashable(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UHUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
	case LogicalTypeId::ENUM:
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::BLOB:
	case LogicalTypeId::BIT:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIMESTAMP_TZ_NS:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIME:
	case LogicalTypeId::DATE:
	case LogicalTypeId::UUID:
	case LogicalTypeId::INTERVAL:
	case LogicalTypeId::GEOMETRY:
		return true;
	case LogicalTypeId::LIST:
	case LogicalTypeId::ARRAY:
	case LogicalTypeId::MAP:
	case LogicalTypeId::VARIANT:
		return false;
	case LogicalTypeId::UNION: {
		idx_t count = UnionType::GetMemberCount(type);
		for (idx_t i = 0; i < count; i++) {
			if (!KeyIsHashable(UnionType::GetMemberType(type, i))) {
				return false;
			}
		}
		// Only if all the member types are hashable do we say the entire UNION is hashable
		return true;
	}
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::TUPLE:
		return false;
	case LogicalTypeId::SQLNULL:
		// A SQLNULL key is always NULL, and Python's None is hashable.
		return true;
	default:
		throw NotImplementedException("Unsupported type: \"%s\"", type.ToString());
	}
}

nb::object PythonObject::FromValue(const Value &val, const LogicalType &type,
                                   const ClientProperties &client_properties) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (val.IsNull()) {
		return nb::none();
	}
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return nb::cast(val.GetValue<bool>());
	case LogicalTypeId::TINYINT:
		return nb::cast(val.GetValue<int8_t>());
	case LogicalTypeId::SMALLINT:
		return nb::cast(val.GetValue<int16_t>());
	case LogicalTypeId::INTEGER:
		return nb::cast(val.GetValue<int32_t>());
	case LogicalTypeId::BIGINT:
		return nb::cast(val.GetValue<int64_t>());
	case LogicalTypeId::UTINYINT:
		return nb::cast(val.GetValue<uint8_t>());
	case LogicalTypeId::USMALLINT:
		return nb::cast(val.GetValue<uint16_t>());
	case LogicalTypeId::UINTEGER:
		return nb::cast(val.GetValue<uint32_t>());
	case LogicalTypeId::UBIGINT:
		return nb::cast(val.GetValue<uint64_t>());
	case LogicalTypeId::HUGEINT:
		return nb::steal<nb::object>(PyLong_FromString(val.GetValue<string>().c_str(), nullptr, 10));
	case LogicalTypeId::UHUGEINT:
		return nb::steal<nb::object>(PyLong_FromString(val.GetValue<string>().c_str(), nullptr, 10));
	case LogicalTypeId::FLOAT:
		return nb::cast(val.GetValue<float>());
	case LogicalTypeId::DOUBLE:
		return nb::cast(val.GetValue<double>());
	case LogicalTypeId::DECIMAL: {
		return import_cache.decimal.Decimal()(val.ToString());
	}
	case LogicalTypeId::ENUM:
		return nb::cast(EnumType::GetValue(val));
	case LogicalTypeId::UNION: {
		return PythonObject::FromValue(UnionValue::GetValue(val), UnionValue::GetType(val), client_properties);
	}
	case LogicalTypeId::VARCHAR:
		return nb::cast(StringValue::Get(val));
	case LogicalTypeId::BLOB:
	case LogicalTypeId::GEOMETRY: {
		auto &blob = StringValue::Get(val);
		return nb::bytes(blob.data(), blob.size());
	}
	case LogicalTypeId::BIT:
		return nb::cast(Bit::ToString(StringValue::Get(val)));
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIMESTAMP_TZ_NS: {
		D_ASSERT(type.InternalType() == PhysicalType::INT64);
		auto timestamp = val.GetValueUnsafe<timestamp_t>();

		InfinityType infinity = GetTimestampInfinityType(timestamp);
		if (infinity == InfinityType::POSITIVE) {
			return nb::borrow<nb::object>(import_cache.datetime.datetime.max());
		}
		if (infinity == InfinityType::NEGATIVE) {
			return nb::borrow<nb::object>(import_cache.datetime.datetime.min());
		}

		if (type.id() == LogicalTypeId::TIMESTAMP_MS) {
			timestamp = Timestamp::FromEpochMs(timestamp.value);
		} else if (type.id() == LogicalTypeId::TIMESTAMP_NS || type.id() == LogicalTypeId::TIMESTAMP_TZ_NS) {
			timestamp = Timestamp::FromEpochNanoSeconds(timestamp.value);
		} else if (type.id() == LogicalTypeId::TIMESTAMP_SEC) {
			timestamp = Timestamp::FromEpochSeconds(timestamp.value);
		}

		int32_t year, month, day, hour, min, sec, micros;
		date_t date;
		dtime_t time;
		Timestamp::Convert(timestamp, date, time);
		Date::Convert(date, year, month, day);
		Time::Convert(time, hour, min, sec, micros);
		nb::object py_timestamp;
		try {
			auto python_conversion = PyDateTime_FromDateAndTime(year, month, day, hour, min, sec, micros);
			if (!python_conversion) {
				throw nb::python_error();
			}
			py_timestamp = nb::steal<nb::object>(python_conversion);
		} catch (nb::python_error &e) {
			// Failed to convert, fall back to str
			auto fallback_str = val.ToString();
			return nb::str(fallback_str.c_str(), fallback_str.size());
		}
		if (type.id() == LogicalTypeId::TIMESTAMP_TZ || type.id() == LogicalTypeId::TIMESTAMP_TZ_NS) {
			// We have to add the timezone info
			auto tz_utc = import_cache.pytz.timezone()("UTC");
			auto timestamp_utc = tz_utc.attr("localize")(py_timestamp);
			auto tz_info = import_cache.pytz.timezone()(client_properties.time_zone);
			return timestamp_utc.attr("astimezone")(tz_info);
		}
		return py_timestamp;
	}
	case LogicalTypeId::TIME_TZ: {
		D_ASSERT(type.InternalType() == PhysicalType::INT64);
		int32_t hour, min, sec, microsec;
		auto time_tz = val.GetValueUnsafe<dtime_tz_t>();
		auto time = time_tz.time();
		auto offset = time_tz.offset();
		duckdb::Time::Convert(time, hour, min, sec, microsec);
		nb::object py_time;
		try {
			auto python_conversion = PyTime_FromTime(hour, min, sec, microsec);
			if (!python_conversion) {
				throw nb::python_error();
			}
			py_time = nb::steal<nb::object>(python_conversion);
		} catch (nb::python_error &e) {
			// Failed to convert, fall back to str
			auto fallback_str = val.ToString();
			return nb::str(fallback_str.c_str(), fallback_str.size());
		}
		// We have to add the timezone info
		auto timedelta = import_cache.datetime.timedelta()(nb::arg("seconds") = offset);
		auto timezone_offset = import_cache.datetime.timezone()(timedelta);
		auto tmp_datetime = import_cache.datetime.datetime.min();
		auto tmp_datetime_with_tz = import_cache.datetime.datetime.combine()(tmp_datetime, py_time, timezone_offset);
		return tmp_datetime_with_tz.attr("timetz")();
	}
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_NS: {
		D_ASSERT(type.InternalType() == PhysicalType::INT64);
		int32_t hour, min, sec, usec;
		dtime_t time;
		if (type.id() == LogicalTypeId::TIME) {
			time = val.GetValueUnsafe<dtime_t>();
		} else {
			// Python's datetime doesn't support nanoseconds, we convert to micros.
			time = dtime_t(val.GetValueUnsafe<dtime_ns_t>().value / 1000);
		}
		duckdb::Time::Convert(time, hour, min, sec, usec);
		try {
			auto pytime = PyTime_FromTime(hour, min, sec, usec);
			if (!pytime) {
				throw nb::python_error();
			}
			return nb::steal<nb::object>(pytime);
		} catch (nb::python_error &e) {
			{
				auto fallback = val.ToString();
				return nb::str(fallback.c_str(), fallback.size());
			}
		}
	}
	case LogicalTypeId::DATE: {
		D_ASSERT(type.InternalType() == PhysicalType::INT32);
		auto date = val.GetValueUnsafe<date_t>();
		int32_t year, month, day;
		if (!Value::IsFinite(date)) {
			if (date == date_t::infinity()) {
				return nb::borrow<nb::object>(import_cache.datetime.date.max());
			}
			return nb::borrow<nb::object>(import_cache.datetime.date.min());
		}
		duckdb::Date::Convert(date, year, month, day);
		try {
			auto pydate = PyDate_FromDate(year, month, day);
			if (!pydate) {
				throw nb::python_error();
			}
			return nb::steal<nb::object>(pydate);
		} catch (nb::python_error &e) {
			{
				auto fallback = val.ToString();
				return nb::str(fallback.c_str(), fallback.size());
			}
		}
	}
	case LogicalTypeId::LIST: {
		auto &list_values = ListValue::GetChildren(val);

		nb::list list;
		for (auto &list_elem : list_values) {
			list.append(FromValue(list_elem, ListType::GetChildType(type), client_properties));
		}
		return std::move(list);
	}
	case LogicalTypeId::ARRAY: {
		auto &array_values = ArrayValue::GetChildren(val);
		auto array_size = ArrayType::GetSize(type);
		auto &child_type = ArrayType::GetChildType(type);

		duckdb::PyUtil::TupleBuilder arr(array_size);
		for (idx_t elem_idx = 0; elem_idx < array_size; elem_idx++) {
			arr.append(FromValue(array_values[elem_idx], child_type, client_properties));
		}
		return arr.take();
	}
	case LogicalTypeId::MAP: {
		auto &list_values = ListValue::GetChildren(val);

		auto &key_type = MapType::KeyType(type);
		auto &val_type = MapType::ValueType(type);

		nb::dict py_struct;
		if (KeyIsHashable(key_type)) {
			for (auto &list_elem : list_values) {
				auto &struct_children = StructValue::GetChildren(list_elem);
				auto key = PythonObject::FromValue(struct_children[0], key_type, client_properties);
				auto value = PythonObject::FromValue(struct_children[1], val_type, client_properties);
				py_struct[std::move(key)] = std::move(value);
			}
		} else {
			nb::list keys;
			nb::list values;
			for (auto &list_elem : list_values) {
				auto &struct_children = StructValue::GetChildren(list_elem);
				keys.append(PythonObject::FromValue(struct_children[0], key_type, client_properties));
				values.append(PythonObject::FromValue(struct_children[1], val_type, client_properties));
			}
			py_struct["key"] = std::move(keys);
			py_struct["value"] = std::move(values);
		}
		return std::move(py_struct);
	}
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::TUPLE: {
		return FromStruct(val, type, client_properties);
	}
	case LogicalTypeId::UUID: {
		auto uuid_value = val.GetValueUnsafe<hugeint_t>();
		return import_cache.uuid.UUID()(UUID::ToString(uuid_value));
	}
	case LogicalTypeId::BIGNUM: {
		auto bignum_value = val.GetValueUnsafe<bignum_t>();
		auto bignum_str = Bignum::BignumToVarchar(bignum_value);
		return nb::str(bignum_str.c_str(), bignum_str.size());
	}
	case LogicalTypeId::INTERVAL: {
		auto interval_value = val.GetValueUnsafe<interval_t>();
		int64_t days = duckdb::Interval::DAYS_PER_MONTH * interval_value.months + interval_value.days;
		return import_cache.datetime.timedelta()(nb::arg("days") = days,
		                                         nb::arg("microseconds") = interval_value.micros);
	}
	case LogicalTypeId::VARIANT: {
		Vector tmp(val, count_t(1));
		RecursiveUnifiedVectorFormat format;
		Vector::RecursiveToUnifiedFormat(tmp, format);
		UnifiedVariantVectorData vector_data(format);
		auto variant_val = VariantUtils::ConvertVariantToValue(vector_data, 0, 0);
		return FromValue(variant_val, variant_val.type(), client_properties);
	}

	default:
		throw NotImplementedException("Unsupported type: \"%s\"", type.ToString());
	}
}

} // namespace duckdb

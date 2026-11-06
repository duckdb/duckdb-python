//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pyresult.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/numpy/array_wrapper.hpp"
#include "duckdb.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/hugeint.hpp"

namespace duckdb {

enum class PythonObjectType {
	Other,
	None,
	Integer,
	Float,
	Bool,
	Decimal,
	Uuid,
	Datetime,
	Date,
	Time,
	Timedelta,
	String,
	ByteArray,
	MemoryView,
	Bytes,
	List,
	Tuple,
	Dict,
	NdArray,
	NdDatetime,
	Value
};

class PythonImportCache;

PythonObjectType GetPythonObjectType(nb::handle &ele);
//! Whether a value can be pandas.NaT or pandas.NA: only an imported pandas can have produced one
bool PandasIsImported(PythonImportCache &import_cache);
//! Whether a value can be a numpy masked array or numpy.ma.masked; numpy 2 imports numpy.ma only on first use
bool NumpyMaIsImported();

LogicalType SniffPythonIntegerType(nb::handle ele);
//! The type of a duckdb.Value, or InvalidInputException when its 'type' attribute is not a DuckDB type
LogicalType PythonValueType(nb::handle value);
bool DictionaryHasMapFormat(const PyDictionary &dict);
//! A target the declared scan builds from a column's type (a shape), as opposed to a column's own type: a dict for a
//! shaped STRUCT takes its own keys, as without a target; a shaped MAP casts each key and value whose type has no
//! children with the connection's settings; a shaped naive timestamp reads a cell as TransformTimestamp does
LogicalType AsShape(const LogicalType &type);
bool IsShape(const LogicalType &type);
//! Whether a cell is a naive pandas Timestamp or a numpy datetime64 finer than a microsecond, whose nanoseconds no
//! Python datetime holds; they are then in nanoseconds
bool HoldsNanoseconds(nb::handle cell, timestamp_ns_t &nanoseconds);
//! A numpy scalar's Python value; a datetime64 or timedelta64 at nanoseconds would read as an integer, so those take
//! the microsecond unit first, as a frame's own datetime columns do
nb::object NumpyScalarItem(nb::handle scalar);
//! A value for a naive timestamp type converted without a target, but keeping the nanoseconds of a pandas Timestamp
//! or a fine datetime64 for a TIMESTAMP_NS, and reading a datetime64 at microseconds for a coarser unit
Value TransformTimestamp(optional_ptr<ClientContext> context, nb::handle ele, const LogicalType &target_type,
                         bool nan_as_null);
void TransformPythonObject(optional_ptr<ClientContext> context, nb::handle ele, Vector &vector, idx_t result_offset,
                           bool nan_as_null = true);
Value TransformPythonValue(optional_ptr<ClientContext> context, nb::handle ele,
                           const LogicalType &target_type = LogicalType::UNKNOWN, bool nan_as_null = true);
string TransformPythonString(nb::handle str);
//! Needs no GIL
InvalidInputException UnencodableStrError(uint32_t codepoint, idx_t position);
//! Raises the pending UnicodeEncodeError of a failed UTF-8 conversion of str as the typed error above
[[noreturn]] void ThrowUnencodableStr(nb::handle str);

} // namespace duckdb

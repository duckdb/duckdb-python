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

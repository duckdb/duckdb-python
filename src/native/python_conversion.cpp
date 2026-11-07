#include "duckdb_python/python_conversion.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/pytype.hpp"

#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/common/exception/conversion_exception.hpp"

#include "duckdb/common/limits.hpp"

namespace duckdb {

// Proxy to LogicalType::ForceMaxLogicalType that switches based on the presence of a client context.
static LogicalType ProxiedForceMaxLogicalType(optional_ptr<ClientContext> context, const LogicalType &left,
                                              const LogicalType &right) {
	return context ? LogicalType::ForceMaxLogicalType(*context, left, right)
	               : LogicalType::DefaultForceMaxLogicalType(left, right);
}

// Like DefaultCastAs, but handles UNION targets by finding the first compatible member. DefaultCastAs raises a
// Conversion Error when multiple UNION members have the same type (e.g. UNION(u1 DOUBLE, u2 DOUBLE)), so for UNION
// targets we resolve the member ourselves.
static Value CastToTarget(Value val, const LogicalType &target_type) {
	if (target_type.id() != LogicalTypeId::UNION) {
		return val.DefaultCastAs(target_type);
	}

	auto member_count = UnionType::GetMemberCount(target_type);
	auto &source_type = val.type();

	// First pass: if there's an exact type match we use that
	for (idx_t i = 0; i < member_count; i++) {
		if (UnionType::GetMemberType(target_type, i) == source_type) {
			return Value::UNION(UnionType::CopyMemberTypes(target_type), NumericCast<uint8_t>(i), std::move(val));
		}
	}

	// Second pass: if there's a type we can implicitly cast to, we do that
	for (idx_t i = 0; i < member_count; i++) {
		auto member_type = UnionType::GetMemberType(target_type, i);
		// DefaultTryCastAs returns the converted value and leaves `val` alone. It used to cast in
		// place and return bool, and because std::optional is contextually convertible to bool the
		// old call still compiled while silently tagging the union with an unconverted value.
		auto candidate = val.DefaultTryCastAs(member_type);
		if (candidate) {
			return Value::UNION(UnionType::CopyMemberTypes(target_type), NumericCast<uint8_t>(i),
			                    std::move(*candidate));
		}
	}
	throw ConversionException("Could not convert value of type %s to %s", source_type.ToString(),
	                          target_type.ToString());
}

InvalidInputException UnencodableStrError(uint32_t codepoint, idx_t position) {
	return InvalidInputException(
	    "Python Conversion Failure: cannot encode str as UTF-8, surrogate '\\u%04x' at position %d is not allowed",
	    codepoint, position);
}

void ThrowUnencodableStr(nb::handle str) {
	nb::python_error error;
	Py_ssize_t start;
	if (!error.matches(PyExc_UnicodeEncodeError) || PyUnicodeEncodeError_GetStart(error.value().ptr(), &start) != 0) {
		throw error;
	}
	throw UnencodableStrError(PyUnicode_ReadChar(str.ptr(), start), NumericCast<idx_t>(start));
}

// Importing the module here instead would run the import on whichever thread of the query converts the value, and a
// None entry in sys.modules is how a program blocks the import
static bool ModuleIsImported(PythonImportCacheItem &module, PyObject *name) {
	if (module.LoadSucceeded()) {
		return true;
	}
	if (!name) {
		PyErr_Clear();
		return false;
	}
	auto entry = PyDict_GetItemWithError(PyImport_GetModuleDict(), name);
	if (!entry && PyErr_Occurred()) {
		PyErr_Clear();
	}
	return entry && entry != Py_None;
}

bool PandasIsImported(PythonImportCache &import_cache) {
	// Interned once: building the key on every call, or asking the import cache about a module it has not loaded,
	// costs an allocation for every converted value
	static PyObject *const name = PyUnicode_InternFromString("pandas");
	return ModuleIsImported(import_cache.pandas, name);
}

//! A numpy scalar cannot exist before numpy is imported
static bool NumpyIsImported(PythonImportCache &import_cache) {
	static PyObject *const name = PyUnicode_InternFromString("numpy");
	return ModuleIsImported(import_cache.numpy, name);
}

//! Decided from the object's type, which runs no code of the object's; numpy is optional and loaded by nothing here,
//! so an ordinary value costs no import and no retried import of a missing numpy
static bool IsNumpyScalar(nb::handle cell) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (!NumpyIsImported(import_cache)) {
		return false;
	}
	auto generic = import_cache.numpy.generic();
	auto type = Py_TYPE(cell.ptr());
	if (generic.ptr() != nullptr && PyType_IsSubtype(type, reinterpret_cast<PyTypeObject *>(generic.ptr()))) {
		return true;
	}
	// A subclass is not taken: numpy.ma.masked and a masked array of no dimensions carry a payload under their mask
	auto ndarray = import_cache.numpy.ndarray();
	return ndarray.ptr() != nullptr && reinterpret_cast<PyObject *>(type) == ndarray.ptr() &&
	       nb::cast<idx_t>(cell.attr("ndim")) == 0;
}

//! A datetime or a timedelta holds no unit finer than a microsecond, so item() gives an integer at these
static bool IsFinerThanAMicrosecond(nb::handle dtype) {
	auto code = nb::cast<string>(dtype.attr("str"));
	return StringUtil::EndsWith(code, "ns]") || StringUtil::EndsWith(code, "ps]") ||
	       StringUtil::EndsWith(code, "fs]") || StringUtil::EndsWith(code, "as]");
}

nb::object PythonValueOfNumpyScalar(nb::handle cell, timestamp_ns_t &timestamp) {
	auto object = cell.ptr();
	// numpy's float, str and bytes scalars are a float, str and bytes the conversion reads as they are, where item()
	// would drop the trailing NUL characters of a str or bytes; an int or None is never a numpy scalar
	if (object == Py_None || PyLong_Check(object) || PyFloat_Check(object) || PyUnicode_Check(object) ||
	    PyBytes_Check(object) || !IsNumpyScalar(cell)) {
		return nb::borrow(cell);
	}
	auto dtype = cell.attr("dtype");
	auto kind = nb::cast<string>(dtype.attr("kind"));
	if (kind == "M" && IsFinerThanAMicrosecond(dtype)) {
		// No Python value holds the nanoseconds; the engine's cast into the column then treats them as it treats a
		// datetime64[ns] column of a frame
		auto nanoseconds = cell.attr("astype")("datetime64[ns]").attr("item")();
		if (nanoseconds.is_none()) {
			return nanoseconds;
		}
		timestamp = timestamp_ns_t(nb::cast<int64_t>(nanoseconds));
		return nb::object();
	}
	nb::object value;
	if (kind == "m" && IsFinerThanAMicrosecond(dtype)) {
		value = cell.attr("astype")("timedelta64[us]").attr("item")();
	} else {
		value = cell.attr("item")();
	}
	return IsNumpyScalar(value) ? nb::borrow(cell) : value;
}

string TransformPythonString(nb::handle str) {
	Py_ssize_t size;
	const char *data = PyUnicode_AsUTF8AndSize(str.ptr(), &size);
	if (data) {
		return string(data, NumericCast<idx_t>(size));
	}
	ThrowUnencodableStr(str);
}

static Value EmptyMapValue() {
	auto map_type = LogicalType::MAP(LogicalType::SQLNULL, LogicalType::SQLNULL);
	return Value::MAP(ListType::GetChildType(map_type), vector<Value>());
}

static vector<Identifier> TransformStructKeys(nb::handle keys, idx_t size) {
	vector<Identifier> res;
	res.reserve(size);
	for (idx_t i = 0; i < size; i++) {
		// str() accepts non-string keys, such as the integer keys of a MAP that DuckDB returns as a plain {1: 10}
		// dict, which nanobind's nb::cast<std::string> rejects.
		res.emplace_back(Identifier(nb::cast<std::string>(nb::str(keys.attr("__getitem__")(i)))));
	}
	return res;
}

//! Whether an object is a list the conversion reads as a LIST: a list, a tuple or a numpy array of at least one
//! dimension, since an array of none holds a single value
static bool IsListLike(nb::handle ele) {
	if (PyList_Check(ele.ptr()) || PyTuple_Check(ele.ptr())) {
		return true;
	}
	return duckdb::PyUtil::IsInstance(ele, DuckDBPyConnection::ImportCache()->numpy.ndarray()) &&
	       nb::cast<idx_t>(ele.attr("ndim")) > 0;
}

bool MixesAListAndAValue(const PyDictionary &dict) {
	if (dict.len != 2) {
		return false;
	}
	auto keys = dict[nb::str("key")];
	auto values = dict[nb::str("value")];
	if (!keys || !values || nb::none().is(keys) || nb::none().is(values)) {
		return false;
	}
	return IsListLike(keys) != IsListLike(values);
}

bool DictionaryHasMapFormat(const PyDictionary &dict) {
	if (dict.len != 2) {
		return false;
	}

	//{ 'key': [ .. keys .. ], 'value': [ .. values .. ]}
	auto keys_key = nb::str("key");
	auto values_key = nb::str("value");
	auto keys = dict[keys_key];
	auto values = dict[values_key];
	if (!keys || !values) {
		return false;
	}

	// Two lists of one length are a MAP's keys and values, and None on both sides, or beside a list, a NULL MAP; any
	// other dict is a plain one, so a NULL entry or a list beside a plain value stays an entry or a field
	auto key_none = nb::none().is(keys);
	auto value_none = nb::none().is(values);
	if (key_none || value_none) {
		return (key_none || IsListLike(keys)) && (value_none || IsListLike(values));
	}
	return IsListLike(keys) && IsListLike(values) && nb::len(keys) == nb::len(values);
}

namespace {
//! The field index of each shaped STRUCT a declared scan meets, kept per thread until the scan of the vector ends,
//! since each takes a pass over the fields to build and every dict of the vector asks the same one; an entry holds
//! its type, so the type information it is found by cannot be freed and given to another type while it is kept, and
//! an index stays where it is while others are added
struct FieldIndex {
	LogicalType type;
	identifier_map_t<idx_t> fields;
};

unordered_map<const LogicalTypeInfo *, FieldIndex> &FieldIndexes() {
	thread_local unordered_map<const LogicalTypeInfo *, FieldIndex> indexes;
	return indexes;
}

const identifier_map_t<idx_t> &FieldIndexOf(const LogicalType &struct_type) {
	auto &indexes = FieldIndexes();
	auto entry = indexes.find(&struct_type.GetTypeInfo());
	if (entry != indexes.end()) {
		return entry->second.fields;
	}
	auto &index = indexes[&struct_type.GetTypeInfo()];
	index.type = struct_type;
	for (idx_t child = 0; child < StructType::GetChildCount(struct_type); child++) {
		index.fields.emplace(StructType::GetChildName(struct_type, child), child);
	}
	return index.fields;
}
} // namespace

// Python a conversion runs, as a tzinfo's utcoffset, can start another declared scan on this thread while an outer
// conversion holds an index, so only the outermost scope drops them
static thread_local idx_t struct_field_index_scopes = 0;

StructFieldIndexScope::StructFieldIndexScope() {
	struct_field_index_scopes++;
}

StructFieldIndexScope::~StructFieldIndexScope() {
	if (--struct_field_index_scopes == 0) {
		FieldIndexes().clear();
	}
}

Value TransformDictionaryToStruct(optional_ptr<ClientContext> context, const PyDictionary &dict,
                                  const LogicalType &target_type, bool nan_as_null) {
	auto conversion_error = [&]() {
		return InvalidInputException("We could not convert the object %s to the desired target type (%s)",
		                             dict.ToString(), target_type.ToString());
	};
	bool struct_target = StructType::IsStruct(target_type);
	// A shape gives only the containers of the values below it: the dict's own keys name the fields, as without a
	// target, and each value converts with the shape of the field of its name
	bool shape_only = struct_target && IsShape(target_type);
	if (struct_target && !shape_only && dict.len != StructType::GetChildCount(target_type)) {
		throw conversion_error();
	}

	child_list_t<Value> struct_values;
	if (!shape_only && StructType::IsUnnamed(target_type)) {
		// A TUPLE has no member names, so its members take the dict's values in dict order.
		for (idx_t i = 0; i < dict.len; i++) {
			auto val = TransformPythonValue(context, dict.values.attr("__getitem__")(i),
			                                StructType::GetChildType(target_type, i), nan_as_null);
			struct_values.emplace_back(make_pair(StructType::GetChildName(target_type, i), std::move(val)));
		}
		return Value::STRUCT(std::move(struct_values));
	}

	auto struct_keys = TransformStructKeys(dict.keys, dict.len);
	if (!struct_target || shape_only) {
		const LogicalType unknown(LogicalTypeId::UNKNOWN);
		auto field_count = shape_only ? StructType::GetChildCount(target_type) : 0;
		// The field at a key's own position is tried first, as df() hands a STRUCT over in field order; a key found
		// elsewhere is looked up in an index of the field names, so a wide STRUCT costs no search for every key
		optional_ptr<const identifier_map_t<idx_t>> fields;
		for (idx_t i = 0; i < dict.len; i++) {
			optional_idx field;
			if (i < field_count && StructType::GetChildName(target_type, i) == struct_keys[i]) {
				field = i;
			} else if (field_count > 0) {
				if (!fields) {
					fields = &FieldIndexOf(target_type);
				}
				auto entry = fields->find(struct_keys[i]);
				if (entry != fields->end()) {
					field = entry->second;
				}
			}
			auto &child_type = field.IsValid() ? StructType::GetChildType(target_type, field.GetIndex()) : unknown;
			auto value = nb::steal(PySequence_GetItem(dict.values.ptr(), NumericCast<Py_ssize_t>(i)));
			if (!value.is_valid()) {
				throw nb::python_error();
			}
			auto val = TransformPythonValue(context, value, child_type, nan_as_null);
			struct_values.emplace_back(make_pair(std::move(struct_keys[i]), std::move(val)));
		}
		return Value::STRUCT(std::move(struct_values));
	}

	// Keys that are distinct in Python but equal as identifiers would both claim one STRUCT field.
	identifier_map_t<idx_t> key_mapping;
	for (idx_t i = 0; i < struct_keys.size(); i++) {
		if (!key_mapping.emplace(struct_keys[i], i).second) {
			throw conversion_error();
		}
	}
	// Every field finds its key before any value converts, so a key mismatch never surfaces as a value cast error.
	vector<idx_t> value_indexes;
	value_indexes.reserve(dict.len);
	for (idx_t i = 0; i < dict.len; i++) {
		auto entry = key_mapping.find(StructType::GetChildName(target_type, i));
		if (entry == key_mapping.end()) {
			throw conversion_error();
		}
		value_indexes.push_back(entry->second);
		// Fields equal ignoring case would otherwise all read one key.
		key_mapping.erase(entry);
	}
	for (idx_t i = 0; i < dict.len; i++) {
		auto val = TransformPythonValue(context, dict.values.attr("__getitem__")(value_indexes[i]),
		                                StructType::GetChildType(target_type, i), nan_as_null);
		struct_values.emplace_back(make_pair(StructType::GetChildName(target_type, i), std::move(val)));
	}
	return Value::STRUCT(std::move(struct_values));
}

// Short enough to stay in a string's own buffer, since every shaped conversion compares it
static constexpr const char *SHAPE_ALIAS = "duckdb_py_shape";

LogicalType AsShape(const LogicalType &type) {
	return type.WithAlias(SHAPE_ALIAS);
}

bool IsShape(const LogicalType &type) {
	return type.HasAlias() && type.GetAlias() == SHAPE_ALIAS;
}

int64_t NanosecondOf(nb::handle cell) {
	// Interned once, since every cell of a timestamp column asks
	static PyObject *const name = PyUnicode_InternFromString("nanosecond");
	if (!name) {
		PyErr_Clear();
		return -1;
	}
	auto nanosecond = nb::steal(PyObject_GetAttr(cell.ptr(), name));
	if (!nanosecond.is_valid()) {
		PyErr_Clear();
		return -1;
	}
	// NaT's nanosecond is NaN
	return PyLong_Check(nanosecond.ptr()) ? PyLong_AsLongLong(nanosecond.ptr()) : -1;
}

bool HoldsNanoseconds(nb::handle cell, timestamp_ns_t &nanoseconds, bool zoned) {
	auto datetime_type = DuckDBPyConnection::ImportCache()->datetime.datetime();
	if (Py_TYPE(cell.ptr()) == reinterpret_cast<PyTypeObject *>(datetime_type.ptr()) ||
	    !duckdb::PyUtil::IsInstance(cell, datetime_type)) {
		return false;
	}
	// A Timestamp without nanoseconds reads as the datetime it is: its value in nanoseconds overflows outside the
	// years 1677 to 2262, and a coarser column would judge that value by a cast where the datetime fits outright
	if (NanosecondOf(cell) <= 0 || nb::getattr(cell, "tzinfo").is_none() == zoned) {
		return false;
	}
	// A zone aware Timestamp's value counts the nanoseconds since the epoch in UTC, as a TIMESTAMPTZ_NS does
	nanoseconds = timestamp_ns_t(nb::cast<int64_t>(cell.attr("value")));
	return true;
}

Value TransformTimestamp(optional_ptr<ClientContext> context, nb::handle ele, const LogicalType &target_type,
                         bool nan_as_null) {
	timestamp_ns_t nanoseconds;
	if (target_type.id() == LogicalTypeId::TIMESTAMP_NS && HoldsNanoseconds(ele, nanoseconds, false)) {
		return Value::TIMESTAMPNS(nanoseconds);
	}
	if (target_type.id() == LogicalTypeId::TIMESTAMP_TZ_NS && HoldsNanoseconds(ele, nanoseconds, true)) {
		return Value::TIMESTAMPTZNS(timestamp_tz_ns_t(nanoseconds.value));
	}
	auto plain = PythonValueOfNumpyScalar(ele, nanoseconds);
	if (!plain.is_valid()) {
		return Value::TIMESTAMPNS(nanoseconds);
	}
	return TransformPythonValue(context, plain, LogicalType::UNKNOWN, nan_as_null);
}

//! Whether an object is what a shape is for: a dict for a MAP or a STRUCT, a list for a LIST or an ARRAY; any other
//! object converts without a target and meets the engine's cast or the kind check, so a tuple reads as a LIST at
//! every depth, as the appender reads a value of a row
static bool ShapeTakes(nb::handle ele, const LogicalType &shape) {
	switch (shape.id()) {
	case LogicalTypeId::MAP:
	case LogicalTypeId::STRUCT:
		return PyDict_Check(ele.ptr());
	case LogicalTypeId::LIST:
	case LogicalTypeId::ARRAY:
		return IsListLike(ele);
	default:
		return false;
	}
}

Value TransformMapEntry(optional_ptr<ClientContext> context, nb::handle entry, const LogicalType &target_type,
                        bool nan_as_null) {
	if (IsShape(target_type) || target_type.IsNested() || target_type.id() == LogicalTypeId::UNKNOWN ||
	    target_type.id() == LogicalTypeId::UNION) {
		return TransformPythonValue(context, entry, target_type, nan_as_null);
	}
	auto value = TransformTimestamp(context, entry, target_type, nan_as_null);
	if (!context) {
		return value.DefaultCastAs(target_type);
	}
	string error;
	auto cast = value.TryCastAs(*context, target_type, &error);
	if (!cast) {
		throw ConversionException(error);
	}
	return std::move(*cast);
}

//! A MAP of the entries of a sequence of keys and one of values, both of the given size; under a shape each key and
//! value converts as TransformMapEntry reads it
static Value MapFromEntries(optional_ptr<ClientContext> context, nb::handle keys, nb::handle values, idx_t size,
                            const LogicalType &target_type, bool nan_as_null) {
	auto key_target = MapType::KeyType(target_type);
	auto value_target = MapType::ValueType(target_type);
	auto shape = IsShape(target_type);

	LogicalType key_type = LogicalType::SQLNULL;
	LogicalType value_type = LogicalType::SQLNULL;

	vector<Value> map_keys;
	vector<Value> map_values;
	map_keys.reserve(size);
	map_values.reserve(size);
	for (idx_t i = 0; i < size; i++) {
		auto key = nb::steal(PySequence_GetItem(keys.ptr(), NumericCast<Py_ssize_t>(i)));
		auto value = nb::steal(PySequence_GetItem(values.ptr(), NumericCast<Py_ssize_t>(i)));
		if (!key.is_valid() || !value.is_valid()) {
			throw nb::python_error();
		}
		Value new_key = shape ? TransformMapEntry(context, key, key_target, nan_as_null)
		                      : TransformPythonValue(context, key, key_target, nan_as_null);
		Value new_value = shape ? TransformMapEntry(context, value, value_target, nan_as_null)
		                        : TransformPythonValue(context, value, value_target, nan_as_null);

		key_type = ProxiedForceMaxLogicalType(context, key_type, new_key.type());
		value_type = ProxiedForceMaxLogicalType(context, value_type, new_value.type());
		map_keys.push_back(std::move(new_key));
		map_values.push_back(std::move(new_value));
	}
	// A target holding UNKNOWN asks for no type there, and a MAP of such a type is no value the engine can hold
	if (key_type.id() == LogicalTypeId::SQLNULL && !TypeVisitor::Contains(key_target, LogicalTypeId::UNKNOWN) &&
	    !IsShape(key_target)) {
		key_type = key_target;
	}
	if (value_type.id() == LogicalTypeId::SQLNULL && !TypeVisitor::Contains(value_target, LogicalTypeId::UNKNOWN) &&
	    !IsShape(value_target)) {
		value_type = value_target;
	}

	// Built from the keys and values themselves, as the STRUCT of each entry would only be taken apart again
	return Value::MAP(key_type, value_type, std::move(map_keys), std::move(map_values));
}

Value TransformStructFormatDictionaryToMap(optional_ptr<ClientContext> context, const PyDictionary &dict,
                                           const LogicalType &target_type, bool nan_as_null) {
	if (dict.len == 0) {
		return EmptyMapValue();
	}

	if (target_type.id() != LogicalTypeId::MAP) {
		throw InvalidInputException("Please provide a valid target type for transform from Python to Value");
	}

	if (nb::none().is(dict.keys) || nb::none().is(dict.values)) {
		return Value(LogicalType::MAP(LogicalTypeId::SQLNULL, LogicalTypeId::SQLNULL));
	}

	auto size = nb::len(dict.keys);
	D_ASSERT(size == nb::len(dict.values));
	return MapFromEntries(context, dict.keys, dict.values, size, target_type, nan_as_null);
}

Value TransformDictionaryToMap(optional_ptr<ClientContext> context, const PyDictionary &dict,
                               const LogicalType &target_type, bool nan_as_null) {
	if (target_type.id() != LogicalTypeId::UNKNOWN && !DictionaryHasMapFormat(dict)) {
		if (MixesAListAndAValue(dict)) {
			auto keys = TransformPythonValue(context, dict[nb::str("key")], LogicalType::UNKNOWN, nan_as_null);
			auto values = TransformPythonValue(context, dict[nb::str("value")], LogicalType::UNKNOWN, nan_as_null);
			throw InvalidInputException("A dict of 'key' and 'value' for a MAP takes a list of keys and a list of "
			                            "values, not %s and %s",
			                            keys.type().ToString(), values.type().ToString());
		}
		// dict == { 'k1': v1, 'k2': v2, ..., 'kn': vn }
		return TransformStructFormatDictionaryToMap(context, dict, target_type, nan_as_null);
	}

	auto keys = dict.values.attr("__getitem__")(0);
	auto values = dict.values.attr("__getitem__")(1);

	if (nb::none().is(keys) || nb::none().is(values)) {
		// Either 'key' or 'value' is None, return early with a NULL value
		return Value(LogicalType::MAP(LogicalTypeId::SQLNULL, LogicalTypeId::SQLNULL));
	}

	auto key_size = nb::len(keys);
	if (key_size == 0) {
		// dict == { 'key': [], 'value': [] }
		return EmptyMapValue();
	}

	// dict == { 'key': [ ... ], 'value' : [ ... ] }
	LogicalType key_target = LogicalTypeId::UNKNOWN;
	LogicalType value_target = LogicalTypeId::UNKNOWN;

	if (IsShape(target_type)) {
		// A shape reads the two lists entry by entry, as it reads a dict's own entries
		auto items = [](nb::handle list) {
			return PyList_Check(list.ptr()) || PyTuple_Check(list.ptr()) ? nb::borrow<nb::object>(list)
			                                                             : list.attr("tolist")();
		};
		return MapFromEntries(context, items(keys), items(values), key_size, target_type, nan_as_null);
	} else if (target_type.id() != LogicalTypeId::UNKNOWN) {
		key_target = LogicalType::LIST(MapType::KeyType(target_type));
		value_target = LogicalType::LIST(MapType::ValueType(target_type));
	}

	auto key_list = TransformPythonValue(context, keys, key_target, nan_as_null);
	auto value_list = TransformPythonValue(context, values, value_target, nan_as_null);
	if (key_list.type().id() != LogicalTypeId::LIST || value_list.type().id() != LogicalTypeId::LIST ||
	    ListValue::GetChildren(key_list).size() != ListValue::GetChildren(value_list).size()) {
		throw InvalidInputException("A dict of 'key' and 'value' for a MAP takes a list of keys and a list of values, "
		                            "not %s and %s",
		                            key_list.type().ToString(), value_list.type().ToString());
	}

	LogicalType key_type = LogicalType::SQLNULL;
	LogicalType value_type = LogicalType::SQLNULL;

	vector<Value> elements;
	for (idx_t i = 0; i < ListValue::GetChildren(key_list).size(); i++) {

		Value new_key = ListValue::GetChildren(key_list)[i];
		Value new_value = ListValue::GetChildren(value_list)[i];

		key_type = ProxiedForceMaxLogicalType(context, key_type, new_key.type());
		value_type = ProxiedForceMaxLogicalType(context, value_type, new_value.type());

		child_list_t<Value> struct_values;
		struct_values.emplace_back(make_pair("key", std::move(new_key)));
		struct_values.emplace_back(make_pair("value", std::move(new_value)));

		elements.push_back(Value::STRUCT(std::move(struct_values)));
	}

	LogicalType map_type = LogicalType::MAP(key_type, value_type);

	return Value::MAP(ListType::GetChildType(map_type), std::move(elements));
}

Value TransformTupleToStruct(optional_ptr<ClientContext> context, nb::handle ele, const LogicalType &target_type,
                             bool nan_as_null) {
	auto tuple = nb::cast<nb::tuple>(ele);
	auto size = nb::len(tuple);

	D_ASSERT(target_type.id() == LogicalTypeId::STRUCT || target_type.id() == LogicalTypeId::TUPLE);
	auto child_types = StructType::GetChildTypes(target_type);
	auto child_count = child_types.size();
	if (size != child_count) {
		throw InvalidInputException("Tried to create a STRUCT value from a tuple containing %d elements, but the "
		                            "STRUCT consists of %d children",
		                            size, child_count);
	}
	child_list_t<Value> children;
	for (idx_t i = 0; i < child_count; i++) {
		auto &type = child_types[i].second;
		auto &name = StructType::GetChildName(target_type, i);
		auto element = nb::handle(tuple[i]);
		auto converted_value = TransformPythonValue(context, element, type, nan_as_null);
		children.emplace_back(make_pair(name, std::move(converted_value)));
	}
	auto result = Value::STRUCT(std::move(children));
	return result;
}

// Tries to convert a Python integer that overflows int64/uint64 into a HUGEINT or UHUGEINT Value
// by decomposing it into upper and lower 64-bit components. Tries HUGEINT first; falls back to
// UHUGEINT for large positive values. Returns false if the value doesn't fit in 128 bits, and throws
// nb::python_error only when CPython itself fails (out of memory).
static bool TryTransformPythonLongToHugeInt(nb::handle ele, const LogicalType &target_type, Value &result) {
	auto ptr = ele.ptr();

	// Extract lower 64 bits (two's complement, works for negative values too)
	uint64_t lower = PyLong_AsUnsignedLongLongMask(ptr);
	if (lower == static_cast<uint64_t>(-1) && PyErr_Occurred()) {
		PyErr_Clear();
		return false;
	}

	// int's own shift, because an int subclass may override __rshift__ and its stored value is what converts
	static const auto int_rshift = reinterpret_cast<binaryfunc>(PyType_GetSlot(&PyLong_Type, Py_nb_rshift));
	D_ASSERT(int_rshift);
	nb::int_ shift_amount(64);
	auto upper_obj = nb::steal(int_rshift(ptr, shift_amount.ptr()));
	if (!upper_obj.is_valid()) {
		throw nb::python_error();
	}

	// Try signed 128-bit (hugeint) first
	int overflow;
	int64_t upper_signed = PyLong_AsLongLongAndOverflow(upper_obj.ptr(), &overflow);
	if (overflow == 0 && !(upper_signed == -1 && PyErr_Occurred())) {
		auto val = Value::HUGEINT(hugeint_t {upper_signed, lower});
		if (target_type.id() == LogicalTypeId::UNKNOWN || target_type.id() == LogicalTypeId::HUGEINT) {
			result = val;
		} else {
			result = CastToTarget(std::move(val), target_type);
		}
		return true;
	}
	PyErr_Clear();

	// Try unsigned 128-bit (uhugeint)
	uint64_t upper_unsigned = PyLong_AsUnsignedLongLong(upper_obj.ptr());
	if (PyErr_Occurred()) {
		PyErr_Clear();
		return false;
	}

	auto val = Value::UHUGEINT(uhugeint_t {upper_unsigned, lower});
	if (target_type.id() == LogicalTypeId::UNKNOWN || target_type.id() == LogicalTypeId::UHUGEINT) {
		result = val;
	} else {
		result = CastToTarget(std::move(val), target_type);
	}
	return true;
}

// Throwing wrapper for contexts that require a result (e.g. prepared statement parameters).
static Value TransformPythonLongToHugeInt(nb::handle ele, const LogicalType &target_type) {
	Value result;
	if (TryTransformPythonLongToHugeInt(ele, target_type, result)) {
		return result;
	}
	// int's own repr of an exact int copy, because printing a large int calls the value's own bit_length
	static const auto int_copy = reinterpret_cast<unaryfunc>(PyType_GetSlot(&PyLong_Type, Py_nb_int));
	static const auto int_repr = reinterpret_cast<reprfunc>(PyType_GetSlot(&PyLong_Type, Py_tp_repr));
	D_ASSERT(int_copy && int_repr);
	auto exact = nb::steal(int_copy(ele.ptr()));
	if (!exact.is_valid()) {
		throw nb::python_error();
	}
	auto digits = nb::steal<nb::str>(int_repr(exact.ptr()));
	if (!digits.is_valid()) {
		// A ValueError is Python's limit on the number of digits it prints.
		if (!PyErr_ExceptionMatches(PyExc_ValueError)) {
			throw nb::python_error();
		}
		PyErr_Clear();
		throw InvalidInputException("Python integer too large for 128-bit integer type");
	}
	throw InvalidInputException("Python integer too large for 128-bit integer type: %s", nb::cast<std::string>(digits));
}

// Picks the tightest DuckDB integer type (>=INT32) for an int64 value when no target type is specified.
static Value SniffIntegerValue(int64_t value) {
	if (value < (int64_t)std::numeric_limits<int32_t>::min() || value > (int64_t)std::numeric_limits<int32_t>::max()) {
		return Value::BIGINT(value);
	}
	return Value::INTEGER(value);
}

// Sniffs the tightest DuckDB integer type for a Python integer.
// Progressively widens: int64 → uint64 → hugeint → uhugeint.
// Returns SQLNULL if the value doesn't fit in any DuckDB integer type (> 128-bit), and throws nb::python_error only
// when CPython itself fails (out of memory).
LogicalType SniffPythonIntegerType(nb::handle ele) {
	auto ptr = ele.ptr();

	// Step 1: Try int64
	int overflow;
	const int64_t value = PyLong_AsLongLongAndOverflow(ptr, &overflow);

	if (overflow == 0) {
		if (value == -1 && PyErr_Occurred()) {
			PyErr_Clear();
			return LogicalType::SQLNULL;
		}
		if (value < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
		    value > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
			return LogicalType::BIGINT;
		}
		return LogicalType::INTEGER;
	}
	PyErr_Clear();

	// Step 2: For positive overflow, try uint64
	if (overflow == 1) {
		(void)PyLong_AsUnsignedLongLong(ptr);
		if (!PyErr_Occurred()) {
			return LogicalType::UBIGINT;
		}
		PyErr_Clear();
	}

	// Step 3: Try 128-bit (hugeint/uhugeint)
	Value res;
	if (!TryTransformPythonLongToHugeInt(ele, LogicalType::UNKNOWN, res)) {
		return LogicalType::SQLNULL;
	}
	return res.type();
}

LogicalType PythonValueType(nb::handle value) {
	nb::object type = value.attr("type");
	std::unique_ptr<DuckDBPyType> value_type;
	if (!DuckDBPyType::TryConvert(type, value_type)) {
		string actual_type = nb::cast<std::string>(nb::str(type.type()));
		throw InvalidInputException("The 'type' of a Value should be of type DuckDBPyType, not '%s'", actual_type);
	}
	return value_type->Type();
}

Value TransformDictionary(optional_ptr<ClientContext> context, const PyDictionary &dict, bool nan_as_null) {
	//! DICT -> MAP FORMAT
	// keys() = [key, value]
	// values() = [ [n keys] ], [ [n values] ]

	//! DICT -> STRUCT FORMAT
	// keys() = ['a', .., 'n']
	// values() = [ val1, .., valn]
	if (dict.len == 0) {
		// dict == {}
		return EmptyMapValue();
	}

	if (DictionaryHasMapFormat(dict)) {
		return TransformDictionaryToMap(context, dict, LogicalType::UNKNOWN, nan_as_null);
	}
	return TransformDictionaryToStruct(context, dict, LogicalType::UNKNOWN, nan_as_null);
}

struct ContiguousBuffer {
	explicit ContiguousBuffer(nb::handle obj) {
		if (PyObject_GetBuffer(obj.ptr(), &view, PyBUF_SIMPLE) != 0) {
			throw nb::python_error();
		}
	}
	~ContiguousBuffer() {
		PyBuffer_Release(&view);
	}
	ContiguousBuffer(const ContiguousBuffer &) = delete;
	ContiguousBuffer &operator=(const ContiguousBuffer &) = delete;

	Py_buffer view;
};

bool NumpyMaIsImported() {
	nb::str name("numpy.ma");
	auto module = nb::steal(PyImport_GetModule(name.ptr()));
	if (!module.is_valid()) {
		PyErr_Clear();
		return false;
	}
	return !module.is_none();
}

PythonObjectType GetPythonObjectType(nb::handle &ele) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();

	// pandas.NaT and pandas.NA are neither bool, int, float nor str, but NaT is a datetime, so they are matched in
	// between; no str can also be a Decimal, UUID, date, time or timedelta, which the str test now precedes
	if (ele.is_none()) {
		return PythonObjectType::None;
	} else if (nb::isinstance<nb::bool_>(ele)) {
		return PythonObjectType::Bool;
	} else if (nb::isinstance<nb::int_>(ele)) {
		return PythonObjectType::Integer;
	} else if (nb::isinstance<nb::float_>(ele)) {
		return PythonObjectType::Float;
	} else if (nb::isinstance<nb::str>(ele)) {
		return PythonObjectType::String;
	} else if (PandasIsImported(import_cache) &&
	           (ele.is(import_cache.pandas.NaT()) || ele.is(import_cache.pandas.NA()))) {
		return PythonObjectType::None;
	} else if (duckdb::PyUtil::IsInstance(ele, import_cache.decimal.Decimal())) {
		return PythonObjectType::Decimal;
	} else if (duckdb::PyUtil::IsInstance(ele, import_cache.uuid.UUID())) {
		return PythonObjectType::Uuid;
	} else if (duckdb::PyUtil::IsInstance(ele, import_cache.datetime.datetime())) {
		return PythonObjectType::Datetime;
	} else if (duckdb::PyUtil::IsInstance(ele, import_cache.datetime.time())) {
		return PythonObjectType::Time;
	} else if (duckdb::PyUtil::IsInstance(ele, import_cache.datetime.date())) {
		return PythonObjectType::Date;
	} else if (duckdb::PyUtil::IsInstance(ele, import_cache.datetime.timedelta())) {
		return PythonObjectType::Timedelta;
	} else if (nb::isinstance<nb::bytearray>(ele)) {
		return PythonObjectType::ByteArray;
	} else if (nb::isinstance<nb::memoryview>(ele)) {
		return PythonObjectType::MemoryView;
	} else if (nb::isinstance<nb::bytes>(ele)) {
		return PythonObjectType::Bytes;
	} else if (nb::isinstance<nb::list>(ele)) {
		return PythonObjectType::List;
	} else if (nb::isinstance<nb::tuple>(ele)) {
		return PythonObjectType::Tuple;
	} else if (nb::isinstance<nb::dict>(ele)) {
		return PythonObjectType::Dict;
	} else if (ele.is(import_cache.numpy.ma.masked())) {
		return PythonObjectType::None;
	} else if (duckdb::PyUtil::IsInstance(ele, import_cache.numpy.ndarray())) {
		return PythonObjectType::NdArray;
	} else if (duckdb::PyUtil::IsInstance(ele, import_cache.numpy.datetime64())) {
		return PythonObjectType::NdDatetime;
	} else if (duckdb::PyUtil::IsInstance(ele, import_cache.duckdb.Value())) {
		return PythonObjectType::Value;
	} else {
		return PythonObjectType::Other;
	}
}

struct PythonValueConversion {
	static const LogicalType &ConversionTarget(Value &result, const LogicalType &target_type) {
		return target_type;
	}

	static void HandleNull(Value &result, const LogicalType &target_type) {
		result = Value();
	}
	static void HandleBoolean(Value &result, const LogicalType &target_type, bool val) {
		result = Value::BOOLEAN(val);
	}
	static void HandleDouble(Value &result, const LogicalType &target_type, double val) {
		switch (target_type.id()) {
		case LogicalTypeId::UNKNOWN:
		case LogicalTypeId::DOUBLE: {
			result = Value::DOUBLE(val);
			break;
		}
		case LogicalTypeId::FLOAT: {
			result = Value::FLOAT(static_cast<float>(val));
			break;
		}
		case LogicalTypeId::DECIMAL: {
			throw ConversionException("Can't losslessly convert from object of float to type %s",
			                          target_type.ToString());
		}
		default:
			result = CastToTarget(Value::DOUBLE(val), target_type);
			break;
		}
	}
	static void HandleLongOverflow(Value &result, const LogicalType &target_type, nb::handle ele) {
		result = TransformPythonLongToHugeInt(ele, target_type);
	}
	static void HandleUnsignedBigint(Value &result, const LogicalType &target_type, uint64_t val) {
		auto cast_as = target_type.id() == LogicalTypeId::UNKNOWN ? LogicalType::UBIGINT : target_type;
		result = CastToTarget(Value::UBIGINT(val), cast_as);
	}
	static void HandleBigint(Value &res, const LogicalType &target_type, int64_t value) {
		if (target_type.id() == LogicalTypeId::UNKNOWN) {
			res = SniffIntegerValue(value);
		} else {
			res = CastToTarget(SniffIntegerValue(value), target_type);
		}
	}

	static void HandleString(Value &result, const LogicalType &target_type, const string &value) {
		if (target_type.id() == LogicalTypeId::UNKNOWN ||
		    (target_type.id() == LogicalTypeId::VARCHAR && !target_type.HasAlias())) {
			result = Value(value);
		} else {
			result = CastToTarget(Value(value), target_type);
		}
	}

	static void HandleTime(Value &result, const LogicalType &target_type, PyTime &time) {
		result = time.ToDuckValue();
	}

	static void HandleDate(Value &result, const LogicalType &target_type, PyDate &date) {
		result = date.ToDuckValue();
	}

	static void HandleDateTime(Value &result, const LogicalType &target_type, PyDateTime &datetime) {
		result = datetime.ToDuckValue(target_type);
	}

	static void HandleBlob(Value &result, const LogicalType &target_type, const_data_ptr_t blob, idx_t blob_size) {
		switch (target_type.id()) {
		case LogicalTypeId::UNKNOWN:
		case LogicalTypeId::BLOB:
			result = Value::BLOB(blob, blob_size);
			break;
		case LogicalTypeId::BIT:
			result = Value::BIT(string(const_char_ptr_cast(blob), blob_size));
			break;
		default:
			throw ConversionException("Could not convert 'bytes' to type %s", target_type.ToString());
		}
	}

	static void HandleList(optional_ptr<ClientContext> context, Value &result, const LogicalType &target_type,
	                       nb::handle ele, idx_t list_size, bool nan_as_null) {
		vector<Value> values;
		values.reserve(list_size);

		LogicalType child_type = LogicalType::UNKNOWN;
		bool is_array = false;
		if (target_type.id() == LogicalTypeId::ARRAY) {
			child_type = ArrayType::GetChildType(target_type);
			is_array = true;
		} else if (target_type.id() == LogicalTypeId::LIST) {
			child_type = ListType::GetChildType(target_type);
		}
		LogicalType element_type = LogicalType::SQLNULL;
		for (idx_t i = 0; i < list_size; i++) {
			Value new_value = TransformPythonValue(context, ele.attr("__getitem__")(i), child_type, nan_as_null);
			element_type = ProxiedForceMaxLogicalType(context, element_type, new_value.type());
			values.push_back(std::move(new_value));
		}
		if (is_array) {
			result = Value::ARRAY(element_type, std::move(values));
		} else {
			result = Value::LIST(element_type, std::move(values));
		}
	}

	static void HandleTuple(optional_ptr<ClientContext> context, Value &result, const LogicalType &target_type,
	                        nb::handle ele, idx_t list_size, bool nan_as_null) {
		if (target_type.id() == LogicalTypeId::STRUCT || target_type.id() == LogicalTypeId::TUPLE) {
			result = TransformTupleToStruct(context, ele, target_type, nan_as_null);
			return;
		}
		HandleList(context, result, target_type, ele, list_size, nan_as_null);
	}

	static Value HandleObjectInternal(optional_ptr<ClientContext> context, nb::handle ele, PythonObjectType object_type,
	                                  const LogicalType &target_type, bool nan_as_null) {
		switch (object_type) {
		case PythonObjectType::Decimal: {
			PyDecimal decimal(ele);
			return decimal.ToDuckValue();
		}
		case PythonObjectType::Uuid: {
			auto string_val = nb::cast<string>(nb::str(ele));
			return Value::UUID(string_val);
		}
		case PythonObjectType::Timedelta: {
			auto timedelta = PyTimeDelta(ele);
			return Value::INTERVAL(timedelta.ToInterval());
		}
		case PythonObjectType::Dict: {
			PyDictionary dict = PyDictionary(nb::borrow<nb::object>(ele));
			switch (target_type.id()) {
			case LogicalTypeId::STRUCT:
			case LogicalTypeId::TUPLE:
				return TransformDictionaryToStruct(context, dict, target_type, nan_as_null);
			case LogicalTypeId::MAP:
				return TransformDictionaryToMap(context, dict, target_type, nan_as_null);
			default:
				return TransformDictionary(context, dict, nan_as_null);
			}
		}
		case PythonObjectType::Value: {
			nb::object object = ele.attr("object");
			auto value_type = PythonValueType(ele);
			return TransformPythonValue(context, object, value_type, nan_as_null);
		}
		default:
			throw InternalException("Unsupported fallback");
		}
	}
	static void HandleObject(optional_ptr<ClientContext> context, nb::handle ele, PythonObjectType object_type,
	                         Value &result, const LogicalType &target_type, bool nan_as_null) {
		result = HandleObjectInternal(context, ele, object_type, target_type, nan_as_null);
	}
};

struct PythonVectorConversion {
	static const LogicalType &ConversionTarget(Vector &result, const idx_t &result_offset) {
		return result.GetType();
	}
	static void HandleNull(Vector &result, const idx_t &result_offset) {
		FlatVector::SetNull(result, result_offset, true);
	}
	static void HandleBoolean(Vector &result, const idx_t &result_offset, bool val) {
		if (result.GetType().id() != LogicalTypeId::BOOLEAN) {
			throw TypeMismatchException(
			    LogicalType::BOOLEAN, result.GetType(),
			    StringUtil::Format(
			        "Python Conversion Failure: Expected a value of type %s, but got a value of type boolean",
			        result.GetType().ToString()));
		}
		FlatVector::GetDataMutable<bool>(result)[result_offset] = val;
	}
	static void HandleDouble(Vector &result, const idx_t &result_offset, double val) {
		switch (result.GetType().id()) {
		case LogicalTypeId::DOUBLE: {
			FlatVector::GetDataMutable<double>(result)[result_offset] = val;
			break;
		}
		case LogicalTypeId::FLOAT: {
			FlatVector::GetDataMutable<float>(result)[result_offset] = static_cast<float>(val);
			break;
		}
		default:
			FallbackValueConversion(result, result_offset, CastToTarget(Value::DOUBLE(val), result.GetType()));
			break;
		}
	}
	static void HandleLongOverflow(Vector &result, const idx_t &result_offset, nb::handle ele) {
		Value result_val = TransformPythonLongToHugeInt(ele, result.GetType());
		FallbackValueConversion(result, result_offset, std::move(result_val));
	}
	static void HandleUnsignedBigint(Vector &result, const idx_t &result_offset, uint64_t value) {
		// this code path is only called for values in the range of [INT64_MAX...UINT64_MAX]
		switch (result.GetType().id()) {
		case LogicalTypeId::HUGEINT:
			FlatVector::GetDataMutable<hugeint_t>(result)[result_offset] = Hugeint::Convert(value);
			break;
		case LogicalTypeId::UHUGEINT:
			FlatVector::GetDataMutable<uhugeint_t>(result)[result_offset] = Uhugeint::Convert(value);
			break;
		case LogicalTypeId::UBIGINT:
			FlatVector::GetDataMutable<uint64_t>(result)[result_offset] = value;
			break;
		default:
			FallbackValueConversion(result, result_offset, CastToTarget(Value::UBIGINT(value), result.GetType()));
			break;
		}
	}
	static void HandleBigint(Vector &result, const idx_t &result_offset, int64_t value) {
		switch (result.GetType().id()) {
		case LogicalTypeId::HUGEINT: {
			FlatVector::GetDataMutable<hugeint_t>(result)[result_offset] = Hugeint::Convert(value);
			break;
		}
		case LogicalTypeId::UHUGEINT: {
			if (value < 0) {
				throw InvalidInputException("Python Conversion Failure: Value out of range for type UHUGEINT");
			}
			FlatVector::GetDataMutable<uhugeint_t>(result)[result_offset] = Uhugeint::Convert(value);
			break;
		}
		case LogicalTypeId::BIGINT: {
			FlatVector::GetDataMutable<int64_t>(result)[result_offset] = value;
			break;
		}
		case LogicalTypeId::INTEGER: {
			if (value < NumericLimits<int32_t>::Minimum() || value > NumericLimits<int32_t>::Maximum()) {
				throw InvalidInputException("Python Conversion Failure: Value out of range for type INT");
			}
			FlatVector::GetDataMutable<int32_t>(result)[result_offset] = static_cast<int32_t>(value);
			break;
		}
		case LogicalTypeId::SMALLINT: {
			if (value < NumericLimits<int16_t>::Minimum() || value > NumericLimits<int16_t>::Maximum()) {
				throw InvalidInputException("Python Conversion Failure: Value out of range for type SMALLINT");
			}
			FlatVector::GetDataMutable<int16_t>(result)[result_offset] = static_cast<int16_t>(value);
			break;
		}
		case LogicalTypeId::TINYINT: {
			if (value < NumericLimits<int8_t>::Minimum() || value > NumericLimits<int8_t>::Maximum()) {
				throw InvalidInputException("Python Conversion Failure: Value out of range for type TINYINT");
			}
			FlatVector::GetDataMutable<int8_t>(result)[result_offset] = static_cast<int8_t>(value);
			break;
		}
		case LogicalTypeId::UBIGINT: {
			if (value < 0) {
				throw InvalidInputException("Python Conversion Failure: Value out of range for type UBIGINT");
			}
			FlatVector::GetDataMutable<uint64_t>(result)[result_offset] = static_cast<uint64_t>(value);
			break;
		}
		case LogicalTypeId::UINTEGER: {
			if (value < 0 || value > (int64_t)NumericLimits<uint32_t>::Maximum()) {
				throw InvalidInputException("Python Conversion Failure: Value out of range for type UINTEGER");
			}
			FlatVector::GetDataMutable<uint32_t>(result)[result_offset] = static_cast<uint32_t>(value);
			break;
		}
		case LogicalTypeId::USMALLINT: {
			if (value < 0 || value > (int64_t)NumericLimits<uint16_t>::Maximum()) {
				throw InvalidInputException("Python Conversion Failure: Value out of range for type USMALLINT");
			}
			FlatVector::GetDataMutable<uint16_t>(result)[result_offset] = static_cast<uint16_t>(value);
			break;
		}
		case LogicalTypeId::UTINYINT: {
			if (value < 0 || value > (int64_t)NumericLimits<uint8_t>::Maximum()) {
				throw InvalidInputException("Python Conversion Failure: Value out of range for type UTINYINT");
			}
			FlatVector::GetDataMutable<uint8_t>(result)[result_offset] = static_cast<uint8_t>(value);
			break;
		}
		default:
			FallbackValueConversion(result, result_offset, CastToTarget(Value::BIGINT(value), result.GetType()));
			break;
		}
	}

	static void HandleString(Vector &result, const idx_t &result_offset, const string &value) {
		auto &result_type = result.GetType();
		if (result_type.id() == LogicalTypeId::VARCHAR) {
			FlatVector::GetDataMutable<string_t>(result)[result_offset] = StringVector::AddString(result, value);
			return;
		}
		Value result_val;
		PythonValueConversion::HandleString(result_val, result_type, value);
		FallbackValueConversion(result, result_offset, std::move(result_val));
	}

	static void HandleDate(Vector &result, const idx_t &result_offset, PyDate &date) {
		auto &result_type = result.GetType();
		switch (result_type.id()) {
		case LogicalTypeId::DATE:
			FlatVector::GetDataMutable<date_t>(result)[result_offset] = date.ToDate();
			break;
		default: {
			auto value = date.ToDuckValue();
			result.SetValue(result_offset, value);
			break;
		}
		}
	}

	static void HandleTime(Vector &result, const idx_t &result_offset, PyTime &time) {
		auto &result_type = result.GetType();
		switch (result_type.id()) {
		case LogicalTypeId::TIME:
			FlatVector::GetDataMutable<dtime_t>(result)[result_offset] = time.ToDuckTime();
			break;
		default: {
			auto value = time.ToDuckValue();
			result.SetValue(result_offset, value);
			break;
		}
		}
	}

	static void HandleBlob(Vector &result, const idx_t &result_offset, const_data_ptr_t blob, idx_t blob_size) {
		auto &result_type = result.GetType();
		switch (result_type.id()) {
		case LogicalTypeId::BLOB:
			FlatVector::GetDataMutable<string_t>(result)[result_offset] =
			    StringVector::AddStringOrBlob(result, const_char_ptr_cast(blob), blob_size);
			break;
		default: {
			auto value = Value::BLOB(blob, blob_size);
			result.SetValue(result_offset, value);
			break;
		}
		}
	}

	static void HandleDateTime(Vector &result, const idx_t &result_offset, PyDateTime &datetime) {
		auto &result_type = result.GetType();
		switch (result_type.id()) {
		case LogicalTypeId::TIMESTAMP:
			FlatVector::GetDataMutable<timestamp_t>(result)[result_offset] = datetime.ToTimestamp();
			break;
		case LogicalTypeId::TIME:
			FlatVector::GetDataMutable<dtime_t>(result)[result_offset] = datetime.ToDuckTime();
			break;
		case LogicalTypeId::DATE:
			FlatVector::GetDataMutable<date_t>(result)[result_offset] = datetime.ToDate();
			break;
		default: {
			auto value = datetime.ToDuckValue(result_type);
			result.SetValue(result_offset, value);
			break;
		}
		}
	}

	template <bool IS_LIST>
	static void HandleListFast(optional_ptr<ClientContext> context, Vector &result, const idx_t &result_offset,
	                           nb::handle ele, idx_t list_size, bool nan_as_null) {
		auto &result_type = result.GetType();
		if (result_type.id() == LogicalTypeId::ARRAY) {
			idx_t array_size = ArrayType::GetSize(result_type);
			if (list_size != array_size) {
				throw InvalidInputException("Python Conversion Failure: Array size mismatch - expected an array of "
				                            "size %d, but got a list of size %d",
				                            array_size, list_size);
			}
			auto &child_array = ArrayVector::GetChildMutable(result);
			idx_t start_offset = result_offset * array_size;
			for (idx_t i = 0; i < list_size; i++) {
				auto child_ele = IS_LIST ? PyList_GetItem(ele.ptr(), i) : PyTuple_GetItem(ele.ptr(), i);
				TransformPythonObject(context, child_ele, child_array, start_offset + i, nan_as_null);
			}
			return;
		}
		if (result_type.id() == LogicalTypeId::LIST) {
			// reserve space for the objects we are about to convert
			auto start_offset = ListVector::GetListSize(result);
			ListVector::Reserve(result, start_offset + list_size);

			// set up the list entry
			auto &list_entry = FlatVector::GetDataMutable<list_entry_t>(result)[result_offset];
			list_entry.offset = start_offset;
			list_entry.length = list_size;

			// convert the child elements
			auto &child_vector = ListVector::GetChildMutable(result);
			for (idx_t i = 0; i < list_size; i++) {
				auto child_ele = IS_LIST ? PyList_GetItem(ele.ptr(), i) : PyTuple_GetItem(ele.ptr(), i);
				TransformPythonObject(context, child_ele, child_vector, start_offset + i, nan_as_null);
			}
			ListVector::SetListSize(result, start_offset + list_size);
			return;
		}
		throw InternalException("Unsupported type for HandleListFast");
	}

	static void HandleList(optional_ptr<ClientContext> context, Vector &result, const idx_t &result_offset,
	                       nb::handle ele, idx_t list_size, bool nan_as_null) {
		auto &result_type = result.GetType();
		if (result_type.id() == LogicalTypeId::ARRAY || result_type.id() == LogicalTypeId::LIST) {
			HandleListFast<true>(context, result, result_offset, ele, list_size, nan_as_null);
			return;
		}
		// fallback to value conversion
		Value result_val;
		PythonValueConversion::HandleList(context, result_val, result_type, ele, list_size, nan_as_null);
		FallbackValueConversion(result, result_offset, std::move(result_val));
	}

	static void ConvertTupleToStruct(optional_ptr<ClientContext> context, Vector &result, const idx_t &result_offset,
	                                 nb::handle ele, idx_t size, bool nan_as_null) {
		auto &child_types = StructType::GetChildTypes(result.GetType());
		auto child_count = child_types.size();
		if (size != child_count) {
			throw InvalidInputException("Tried to create a STRUCT value from a tuple containing %d elements, but the "
			                            "STRUCT consists of %d children",
			                            size, child_count);
		}

		auto &struct_children = StructVector::GetEntries(result);
		for (idx_t i = 0; i < child_count; i++) {
			auto child_ele = PyTuple_GetItem(ele.ptr(), i);
			TransformPythonObject(context, child_ele, struct_children[i], result_offset, nan_as_null);
		}
	}

	static void HandleTuple(optional_ptr<ClientContext> context, Vector &result, const idx_t &result_offset,
	                        nb::handle ele, idx_t tuple_size, bool nan_as_null) {
		auto &result_type = result.GetType();
		switch (result_type.id()) {
		case LogicalTypeId::STRUCT:
		case LogicalTypeId::TUPLE:
			ConvertTupleToStruct(context, result, result_offset, ele, tuple_size, nan_as_null);
			break;
		case LogicalTypeId::ARRAY:
		case LogicalTypeId::LIST:
			HandleListFast<false>(context, result, result_offset, ele, tuple_size, nan_as_null);
			break;
		default:
			throw InternalException("Unsupported type for HandleTuple");
		}
	}

	static void FallbackValueConversion(Vector &result, const idx_t &result_offset, Value val) {
		result.SetValue(result_offset, val);
	}
	static void HandleObject(optional_ptr<ClientContext> context, nb::handle ele, PythonObjectType object_type,
	                         Vector &result, const idx_t &result_offset, bool nan_as_null) {
		Value result_val;
		PythonValueConversion::HandleObject(context, ele, object_type, result_val, result.GetType(), nan_as_null);
		result.SetValue(result_offset, result_val);
	}
};

template <class OP, class A, class B>
void TransformPythonObjectInternal(optional_ptr<ClientContext> context, nb::handle ele, A &result, const B &param,
                                   bool nan_as_null) {
	auto object_type = GetPythonObjectType(ele);

	switch (object_type) {
	case PythonObjectType::None:
		OP::HandleNull(result, param);
		break;
	case PythonObjectType::Bool:
		OP::HandleBoolean(result, param, nb::cast<bool>(ele));
		break;
	case PythonObjectType::Float:
		if (nan_as_null && std::isnan(PyFloat_AsDouble(ele.ptr()))) {
			OP::HandleNull(result, param);
			break;
		}
		OP::HandleDouble(result, param, nb::cast<double>(ele));
		break;
	case PythonObjectType::Integer: {
		auto ptr = ele.ptr();

		int overflow;
		int64_t value = PyLong_AsLongLongAndOverflow(ptr, &overflow);

		if (overflow != 0) {
			PyErr_Clear();
			auto &conversion_target = OP::ConversionTarget(result, param);
			switch (conversion_target.id()) {
			case LogicalTypeId::BIGINT:
			case LogicalTypeId::INTEGER:
			case LogicalTypeId::SMALLINT:
			case LogicalTypeId::TINYINT:
			case LogicalTypeId::UINTEGER:
			case LogicalTypeId::USMALLINT:
			case LogicalTypeId::UTINYINT:
				throw InvalidInputException("Python Conversion Failure: Value out of range for type %s",
				                            conversion_target);
			default:
				break;
			}
			if (overflow == 1) { // value is > LLONG_MAX
				uint64_t unsigned_value = PyLong_AsUnsignedLongLong(ptr);
				if (!PyErr_Occurred()) {
					// value does not fit within an int64, but it fits within a uint64
					OP::HandleUnsignedBigint(result, param, unsigned_value);
					break;
				}
				PyErr_Clear();
				if (conversion_target.id() == LogicalTypeId::UBIGINT) {
					throw InvalidInputException("Python Conversion Failure: Value out of range for type %s",
					                            conversion_target);
				}
			}
			OP::HandleLongOverflow(result, param, ele);
		} else if (value == -1 && PyErr_Occurred()) {
			throw InvalidInputException("An error occurred attempting to convert a python integer");
		} else {
			// value fits within an int64
			OP::HandleBigint(result, param, value);
		}
		break;
	}
	case PythonObjectType::List: {
		auto list_size = nb::len(ele);
		OP::HandleList(context, result, param, ele, list_size, nan_as_null);
		break;
	}
	case PythonObjectType::Tuple: {
		auto list_size = nb::len(ele);
		auto &conversion_target = OP::ConversionTarget(result, param);
		switch (conversion_target.id()) {
		case LogicalTypeId::STRUCT:
		case LogicalTypeId::TUPLE:
		case LogicalTypeId::UNKNOWN:
		case LogicalTypeId::LIST:
		case LogicalTypeId::ARRAY:
			OP::HandleTuple(context, result, param, ele, list_size, nan_as_null);
			break;
		default:
			throw InvalidInputException("Can't convert tuple to a Value of type %s", conversion_target);
		}
		break;
	}
	case PythonObjectType::String: {
		auto stringified = TransformPythonString(ele);
		OP::HandleString(result, param, stringified);
		break;
	}
	case PythonObjectType::Datetime: {
		auto &import_cache = *DuckDBPyConnection::ImportCache();
		bool is_nat = false;
		if (import_cache.pandas.isnull(false)) {
			auto isnull_result = import_cache.pandas.isnull()(ele);
			is_nat = nb::cast<std::string>(nb::str(isnull_result)) == "True";
		}
		if (is_nat) {
			OP::HandleNull(result, param);
			break;
		}
		PyDateTime datetime(ele);
		OP::HandleDateTime(result, param, datetime);
		break;
	}
	case PythonObjectType::Time: {
		PyTime time(ele);
		OP::HandleTime(result, param, time);
		break;
	}
	case PythonObjectType::Date: {
		PyDate date(ele);
		OP::HandleDate(result, param, date);
		break;
	}
	case PythonObjectType::ByteArray: {
		auto byte_array = ele;
		const_data_ptr_t bytes = const_data_ptr_cast(PyByteArray_AsString(byte_array.ptr())); // NOLINT
		idx_t byte_length = PyByteArray_Size(byte_array.ptr());
		OP::HandleBlob(result, param, bytes, byte_length);
		break;
	}
	case PythonObjectType::MemoryView: {
		ContiguousBuffer buffer(ele);
		OP::HandleBlob(result, param, const_data_ptr_t(buffer.view.buf), idx_t(buffer.view.len));
		break;
	}
	case PythonObjectType::Bytes: {
		// Read the buffer directly (mirrors the ByteArray branch above): nanobind's nb::cast<std::string> rejects
		// a bytes object, so go through the CPython API instead.
		char *bytes_buffer;
		Py_ssize_t bytes_length;
		PyBytes_AsStringAndSize(ele.ptr(), &bytes_buffer, &bytes_length); // NOLINT
		OP::HandleBlob(result, param, const_data_ptr_cast(bytes_buffer), idx_t(bytes_length));
		break;
	}
	case PythonObjectType::NdArray:
	case PythonObjectType::NdDatetime:
		TransformPythonObjectInternal<OP>(context, ele.attr("tolist")(), result, param, nan_as_null);
		break;
	case PythonObjectType::Uuid:
	case PythonObjectType::Timedelta:
	case PythonObjectType::Dict:
	case PythonObjectType::Value:
	case PythonObjectType::Decimal: {
		OP::HandleObject(context, ele, object_type, result, param, nan_as_null);
		break;
	}
	case PythonObjectType::Other:
		throw NotImplementedException("Unable to transform python value of type '%s' to DuckDB LogicalType",
		                              nb::cast<string>(nb::str((ele).type())));
	default:
		throw InternalException("Object type recognized but not implemented!");
	}
}

void TransformPythonObject(optional_ptr<ClientContext> context, nb::handle ele, Vector &vector, idx_t result_offset,
                           bool nan_as_null) {
	TransformPythonObjectInternal<PythonVectorConversion>(context, ele, vector, result_offset, nan_as_null);
}

//! A shape of no children marks a timestamp, read as TransformTimestamp reads it; a container shape takes only the
//! objects it is for, so a scalar inside a container meets the engine's cast as at the top, never a type holding
//! UNKNOWN
static Value TransformShapedValue(optional_ptr<ClientContext> context, nb::handle ele, const LogicalType &shape,
                                  bool nan_as_null) {
	if (!shape.IsNested()) {
		return TransformTimestamp(context, ele, shape, nan_as_null);
	}
	if (!ShapeTakes(ele, shape)) {
		return TransformPythonValue(context, ele, LogicalType::UNKNOWN, nan_as_null);
	}
	Value result;
	TransformPythonObjectInternal<PythonValueConversion>(context, ele, result, shape, nan_as_null);
	return result;
}

Value TransformPythonValue(optional_ptr<ClientContext> context, nb::handle ele, const LogicalType &target_type,
                           bool nan_as_null) {
	// One named result on every path, so it is built in place rather than moved out for every value converted; a
	// conversion without a target, the most common, asks for no alias
	Value result;
	if (target_type.id() != LogicalTypeId::UNKNOWN && IsShape(target_type)) {
		result = TransformShapedValue(context, ele, target_type, nan_as_null);
	} else {
		TransformPythonObjectInternal<PythonValueConversion>(context, ele, result, target_type, nan_as_null);
	}
	return result;
}

} // namespace duckdb

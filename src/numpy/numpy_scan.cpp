#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/python_conversion.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/swar.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/common/vector/map_vector.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb_python/pandas/pandas_bind.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/pandas/pandas_analyzer.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb_python/numpy/numpy_scan.hpp"
#include "duckdb_python/exceptions.hpp"
#include "duckdb_python/numpy/numpy_array.hpp"
#include "duckdb_python/pandas/column/pandas_numpy_column.hpp"
#include "duckdb/function/cast/cast_function_set.hpp"
#include "duckdb/function/cast/default_casts.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb_python/staged_rows.hpp"

namespace duckdb {

template <class T>
void ScanNumpyColumn(NumpyArray &numpy_col, idx_t stride, idx_t offset, Vector &out, idx_t count) {
	auto src_ptr = (T *)numpy_col.Data();
	if (stride == sizeof(T)) {
		FlatVector::SetData(out, data_ptr_cast(src_ptr + offset), count_t(count));
	} else {
		auto tgt_ptr = FlatVector::GetDataMutable<T>(out);
		for (idx_t i = 0; i < count; i++) {
			tgt_ptr[i] = src_ptr[stride / sizeof(T) * (i + offset)];
		}
	}
}

template <class T, class V>
void ScanNumpyCategoryTemplated(NumpyArray &column, idx_t stride, idx_t offset, Vector &out, idx_t count) {
	auto src_ptr = (T *)column.Data();
	auto tgt_ptr = FlatVector::GetDataMutable<V>(out);
	auto &tgt_mask = FlatVector::ValidityMutable(out);
	for (idx_t i = 0; i < count; i++) {
		auto code = src_ptr[stride / sizeof(T) * (i + offset)];
		if (code == -1) {
			// Null value
			tgt_mask.SetInvalid(i);
		} else {
			tgt_ptr[i] = code;
		}
	}
}

template <class T>
void ScanNumpyCategory(NumpyArray &column, idx_t stride, idx_t count, idx_t offset, Vector &out, string &src_type) {
	if (src_type == "int8") {
		ScanNumpyCategoryTemplated<int8_t, T>(column, stride, offset, out, count);
	} else if (src_type == "int16") {
		ScanNumpyCategoryTemplated<int16_t, T>(column, stride, offset, out, count);
	} else if (src_type == "int32") {
		ScanNumpyCategoryTemplated<int32_t, T>(column, stride, offset, out, count);
	} else if (src_type == "int64") {
		ScanNumpyCategoryTemplated<int64_t, T>(column, stride, offset, out, count);
	} else {
		throw NotImplementedException("The Pandas type " + src_type + " for categorical types is not implemented yet");
	}
}

//! One numpy bool per row of the whole column, or null when the column has no mask. They are bytes, because numpy takes
//! any nonzero byte for true and a C++ bool may hold only 0 or 1.
static const uint8_t *MaskedRows(const PandasColumnBindData &bind_data) {
	return bind_data.mask ? static_cast<const uint8_t *>(bind_data.mask->numpy_array.Data()) : nullptr;
}

//! The number of mask bytes read at once, one validity entry's worth of rows
static constexpr idx_t MASK_BLOCK_ROWS = SwarBlock::SIZE;
static_assert(MASK_BLOCK_ROWS == ValidityMask::BITS_PER_VALUE, "a block of mask bytes maps to one validity entry");

//! Bit i is set when byte i of the block is nonzero
static uint64_t MaskedBlockFlags(const_data_ptr_t block) {
	uint64_t words[SwarBlock::WORDS];
	uint64_t any = 0;
	for (idx_t i = 0; i < SwarBlock::WORDS; i++) {
		words[i] = Load<uint64_t>(block + i * SwarWord::SIZE);
		any |= words[i];
	}
	if (any == 0) {
		return 0;
	}
	uint64_t flags = 0;
	for (idx_t i = 0; i < SwarBlock::WORDS; i++) {
		flags |= SwarWord::PackFlags(SwarWord::ZeroBytes(words[i]) ^ SwarWord::MSB) << (i * SwarWord::SIZE);
	}
	return flags;
}

//! Bit i is set when the mask byte of row i of the row_count rows from mask is nonzero
static uint64_t MaskedRowFlags(const uint8_t *mask, idx_t row_count) {
	if (row_count >= MASK_BLOCK_ROWS) {
		return MaskedBlockFlags(mask);
	}
	// The bytes past the last row stay zero, so they mask nothing
	data_t tail[MASK_BLOCK_ROWS] = {};
	memcpy(tail, mask, row_count);
	return MaskedBlockFlags(tail);
}

//! Sets invalid the rows of the block that starts at block_row whose bit is set in rows
static void SetInvalidRows(ValidityMask &validity, idx_t block_row, uint64_t rows) {
	D_ASSERT(block_row % ValidityMask::BITS_PER_VALUE == 0);
	D_ASSERT(block_row < validity.Capacity());
	validity.EnsureWritable();
	validity.GetData()[block_row / ValidityMask::BITS_PER_VALUE] &= ~rows;
}

static void SetInvalidRowsRecursive(Vector &out, idx_t block_row, uint64_t rows) {
	SetInvalidRows(FlatVector::ValidityMutable(out), block_row, rows);
	if (out.GetType().InternalType() == PhysicalType::STRUCT) {
		auto &children = StructVector::GetEntries(out);
		for (idx_t i = 0; i < children.size(); i++) {
			SetInvalidRowsRecursive(children[i], block_row, rows);
		}
	}
}

static void ApplyMask(PandasColumnBindData &bind_data, ValidityMask &validity, idx_t count, idx_t offset) {
	D_ASSERT(bind_data.mask);
	auto mask = MaskedRows(bind_data) + offset;
	for (idx_t block_row = 0; block_row < count; block_row += MASK_BLOCK_ROWS) {
		auto masked = MaskedRowFlags(mask + block_row, count - block_row);
		if (masked != 0) {
			SetInvalidRows(validity, block_row, masked);
		}
	}
}

//! Sets invalid the rows of the block from block_row of the vector of count rows whose column mask byte is nonzero, and
//! returns the other rows of the block, bit i for row block_row + i, so their reader can skip the masked ones
static uint64_t UnmaskedBlockRows(const uint8_t *column_mask, idx_t count, idx_t offset, idx_t block_row, Vector &out) {
	D_ASSERT(column_mask);
	auto block_rows = MinValue<idx_t>(MASK_BLOCK_ROWS, count - block_row);
	auto all_rows = block_rows == MASK_BLOCK_ROWS ? ~uint64_t(0) : (uint64_t(1) << block_rows) - 1;
	auto masked = MaskedRowFlags(column_mask + offset + block_row, block_rows);
	if (masked != 0) {
		SetInvalidRowsRecursive(out, block_row, masked);
	}
	return all_rows & ~masked;
}

template <class T>
void ScanNumpyMasked(PandasColumnBindData &bind_data, idx_t count, idx_t offset, Vector &out) {
	D_ASSERT(bind_data.pandas_col->Backend() == PandasColumnBackend::NUMPY);
	auto &numpy_col = reinterpret_cast<PandasNumpyColumn &>(*bind_data.pandas_col);
	ScanNumpyColumn<T>(numpy_col.array, numpy_col.stride, offset, out, count);
	if (bind_data.mask) {
		auto &result_mask = FlatVector::ValidityMutable(out);
		ApplyMask(bind_data, result_mask, count, offset);
	}
}

template <class T>
void ScanNumpyFpColumn(PandasColumnBindData &bind_data, const T *src_ptr, idx_t stride, idx_t count, idx_t offset,
                       Vector &out) {
	if (stride == sizeof(T)) {
		FlatVector::SetData(out, (data_ptr_t)(src_ptr + offset), count_t(count)); // NOLINT
		// Turn NaN values into NULL
		auto tgt_ptr = FlatVector::GetData<T>(out);
		for (idx_t i = 0; i < count; i++) {
			if (Value::IsNan<T>(tgt_ptr[i])) {
				FlatVector::ValidityMutable(out).SetInvalid(i);
			}
		}
	} else {
		auto tgt_ptr = FlatVector::GetDataMutable<T>(out);
		for (idx_t i = 0; i < count; i++) {
			tgt_ptr[i] = src_ptr[stride / sizeof(T) * (i + offset)];
			if (Value::IsNan<T>(tgt_ptr[i])) {
				FlatVector::ValidityMutable(out).SetInvalid(i);
			}
		}
	}
	if (bind_data.mask) {
		auto &result_mask = FlatVector::ValidityMutable(out);
		ApplyMask(bind_data, result_mask, count, offset);
	}
}

static void SetInvalidRecursive(Vector &out, idx_t index) {
	auto row_in_block = index % ValidityMask::BITS_PER_VALUE;
	SetInvalidRowsRecursive(out, index - row_in_block, uint64_t(1) << row_in_block);
}

//! 'count' is the amount of rows in the 'out' vector
//! 'offset' is the current row number within this vector
void ScanNumpyObject(optional_ptr<ClientContext> context, PyObject *object, idx_t offset, Vector &out) {
	// handle None
	if (object == Py_None) {
		SetInvalidRecursive(out, offset);
		return;
	}

	TransformPythonObject(context, object, out, offset);
}

static void VerifyMapConstraints(Vector &vec, idx_t count) {
	auto invalid_reason = MapVector::CheckMapValidity(vec, count);
	switch (invalid_reason) {
	case MapInvalidReason::VALID:
		return;
	case MapInvalidReason::DUPLICATE_KEY:
		throw InvalidInputException("Dict->Map conversion failed because 'key' list contains duplicates");
	case MapInvalidReason::NULL_KEY:
		throw InvalidInputException("Dict->Map conversion failed because 'key' list contains None");
	default:
		throw InvalidInputException("Option not implemented for MapInvalidReason");
	}
}

//! A MAP's keys must be unique at every depth; a cast of a STRUCT into a MAP, as of a dict whose keys collapse into
//! one key, does not check them
void VerifyTypeConstraints(Vector &vec, idx_t count) {
	auto &type = vec.GetType();
	if (type.id() == LogicalTypeId::MAP) {
		VerifyMapConstraints(vec, count);
	}
	switch (type.InternalType()) {
	case PhysicalType::LIST:
		VerifyTypeConstraints(ListVector::GetEntry(vec), ListVector::GetListSize(vec));
		break;
	case PhysicalType::ARRAY:
		VerifyTypeConstraints(ArrayVector::GetEntry(vec), count * ArrayType::GetSize(type));
		break;
	case PhysicalType::STRUCT:
		for (auto &child : StructVector::GetEntries(vec)) {
			VerifyTypeConstraints(child, count);
		}
		break;
	default:
		break;
	}
}

//! The casts one declared column needs, looked up once per source type for the vector, since the lookup costs more
//! than casting one value. The cache is bounded: dicts give a STRUCT type per key set, and past the bound a lookup
//! per value beats a search through them.
class DeclaredCasts {
public:
	DeclaredCasts(ClientContext &context, const LogicalType &target)
	    : context(context), target(target), set(CastFunctionSet::Get(context)), get_input(context) {
	}

	Value Cast(const Value &value) {
		if (value.type() == target) {
			return value;
		}
		auto cast = For(value.type());
		if (!cast) {
			return CastToColumn(context, value, target);
		}
		Vector input(value, count_t(1));
		Vector result(target, 1);
		// Without an error string the cast raises its own exception, as the appender's cast does
		CastParameters parameters(cast->function.GetCastData().get(), false, nullptr, cast->local_state.get());
		cast->function.Cast(input, result, 1, parameters);
		return result.GetValue(0);
	}

private:
	static constexpr idx_t CACHED_SOURCE_TYPES = 16;

	struct Entry {
		LogicalType source;
		BoundCastInfo function;
		unique_ptr<FunctionLocalState> local_state;
	};

	optional_ptr<Entry> For(const LogicalType &source) {
		for (auto &entry : casts) {
			if (entry.source == source) {
				return &entry;
			}
		}
		if (casts.size() >= CACHED_SOURCE_TYPES) {
			return nullptr;
		}
		Entry entry {source, set.GetCastFunction(source, target, get_input), nullptr};
		if (entry.function.HasInitLocalState()) {
			CastLocalStateParameters parameters(get_input.context, entry.function.GetCastData());
			entry.local_state = entry.function.InitLocalState(parameters);
		}
		casts.push_back(std::move(entry));
		return &casts.back();
	}

	ClientContext &context;
	const LogicalType &target;
	CastFunctionSet &set;
	GetCastFunctionInput get_input;
	vector<Entry> casts;
};

//! A declared column converts a cell as the appender converts a value of a row: with the type the value has in
//! Python, then the engine's cast into the column's type, which reads the connection's settings; NaN reads NULL at
//! every depth, and a value without a DuckDB type of its own reads as its text in a text column
//! A numpy scalar, or an array of no dimensions, reads as its Python value; a datetime64 or timedelta64 at
//! nanoseconds would read as an integer, so those take the microsecond unit first, as the frame's own datetime
//! columns do
static bool IsNumpyScalar(nb::handle cell) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (nb::isinstance(cell, import_cache.numpy.generic())) {
		return true;
	}
	// A subclass is not taken: numpy.ma.masked and a masked array of no dimensions carry a payload under their mask
	return cell.type().is(import_cache.numpy.ndarray()) && nb::cast<idx_t>(cell.attr("ndim")) == 0;
}

static nb::object NumpyScalarValue(nb::handle scalar) {
	auto kind = nb::cast<string>(scalar.attr("dtype").attr("kind"));
	if (kind == "M") {
		return scalar.attr("astype")("datetime64[us]").attr("item")();
	}
	if (kind == "m") {
		return scalar.attr("astype")("timedelta64[us]").attr("item")();
	}
	return scalar.attr("item")();
}

static Value DeclaredValue(ClientContext &context, const LogicalType &target, nb::handle cell) {
	if (cell.ptr() == Py_None) {
		return Value();
	}
	if (IsNumpyScalar(cell)) {
		auto value = NumpyScalarValue(cell);
		// A long double's item() is a numpy scalar again; one step, then the conversion refuses what is left
		if (!IsNumpyScalar(value)) {
			return DeclaredValue(context, target, value);
		}
	}
	if (target.id() == LogicalTypeId::VARCHAR && GetPythonObjectType(cell) == PythonObjectType::Other) {
		return Value(TransformPythonString(nb::str(cell)));
	}
	return TransformPythonValue(context, cell, LogicalType::UNKNOWN, true);
}

//! The values of one vector of a declared column, written once the cells are read: a vector whose values share one
//! type is cast by the engine in one call, which costs a fraction of a cast per value. Only the values present
//! enter the cast and its result is copied into the output, so a NULL row never meets a cast that would give it a
//! shape of its own (a UNION member that is NULL), and the output owns every string the cast produced.
class DeclaredVector {
public:
	explicit DeclaredVector(idx_t count) : values(count), source(LogicalTypeId::SQLNULL) {
	}

	void Set(idx_t row, Value value) {
		if (value.IsNull()) {
			return;
		}
		if (source.id() == LogicalTypeId::SQLNULL) {
			source = value.type();
		} else if (uniform && value.type() != source) {
			uniform = false;
		}
		values[row] = std::move(value);
	}

	void Write(ClientContext &context, Vector &out, idx_t count) {
		auto &target = out.GetType();
		if (uniform && source.id() != LogicalTypeId::SQLNULL && source != target) {
			idx_t present = 0;
			for (idx_t row = 0; row < count; row++) {
				present += !values[row].IsNull();
			}
			Vector staged(source, present);
			idx_t staged_row = 0;
			for (idx_t row = 0; row < count; row++) {
				if (!values[row].IsNull()) {
					staged.SetValue(staged_row++, values[row]);
				}
			}
			// A vector built with a capacity has no rows, and the fallback cast reads the row count to decide whether
			// any value is present
			FlatVector::SetSize(staged, present);
			Vector casted(target, present);
			// Without an error string the cast raises its own exception, as the appender's cast does
			VectorOperations::TryCast(context, staged, casted, present, nullptr);
			idx_t copied = 0;
			for (idx_t row = 0; row < count;) {
				if (values[row].IsNull()) {
					FlatVector::SetNull(out, row, true);
					row++;
					continue;
				}
				idx_t run = 1;
				while (row + run < count && !values[row + run].IsNull()) {
					run++;
				}
				VectorOperations::Copy(casted, out, copied + run, copied, row);
				copied += run;
				row += run;
			}
			return;
		}
		DeclaredCasts casts(context, target);
		for (idx_t row = 0; row < count; row++) {
			if (values[row].IsNull()) {
				FlatVector::SetNull(out, row, true);
			} else {
				out.SetValue(row, casts.Cast(values[row]));
			}
		}
	}

private:
	vector<Value> values;
	LogicalType source;
	bool uniform = true;
};

// A column without a mask takes the instantiation that tests no mask bits, so it reads like the plain array
template <bool HAS_MASK, class CONVERT>
static void ScanObjectRows(PyObject **col, idx_t stride, const uint8_t *masked, idx_t count, idx_t offset, Vector &out,
                           CONVERT &&convert) {
	for (idx_t block_row = 0; block_row < count; block_row += MASK_BLOCK_ROWS) {
		auto block_end = MinValue<idx_t>(block_row + MASK_BLOCK_ROWS, count);
		// A masked cell can hold a value of another type, or no object at all
		uint64_t unmasked = HAS_MASK ? UnmaskedBlockRows(masked, count, offset, block_row, out) : 0;
		for (idx_t i = block_row; i < block_end; i++, unmasked >>= 1) {
			if (HAS_MASK && !(unmasked & 1)) {
				continue;
			}
			// Converting can run Python, which lets another thread replace the cell and free the object it held
			auto cell = nb::borrow<nb::object>(col[stride / sizeof(PyObject *) * (i + offset)]);
			convert(cell, i);
		}
	}
}

template <class CONVERT>
static void ScanObjectCells(ClientContext &context, PyObject **col, idx_t stride, const uint8_t *masked, idx_t count,
                            idx_t offset, Vector &out, CONVERT &&convert) {
	// numpy_col is a sequential list of objects, that make up one "column" (Vector)
	out.SetVectorType(VectorType::FLAT_VECTOR);
	PyUtil::QueryGil gil(context); // We're creating python objects here, so we need the GIL

	try {
		if (masked) {
			ScanObjectRows<true>(col, stride, masked, count, offset, out, convert);
		} else {
			ScanObjectRows<false>(col, stride, masked, count, offset, out, convert);
		}
	} catch (nb::python_error &error) {
		throw AsDuckDBError(error);
	}
}

void NumpyScan::ScanDeclaredColumn(ClientContext &context, PyObject **col, idx_t stride, const uint8_t *masked,
                                   idx_t count, idx_t offset, Vector &out) {
	DeclaredVector staged(count);
	auto &target = out.GetType();
	ScanObjectCells(context, col, stride, masked, count, offset, out,
	                [&](nb::handle cell, idx_t row) { staged.Set(row, DeclaredValue(context, target, cell)); });
	staged.Write(context, out, count);
	VerifyTypeConstraints(out, count);
}

void NumpyScan::ScanObjectColumn(ClientContext &context, PyObject **col, idx_t stride, const uint8_t *masked,
                                 idx_t count, idx_t offset, Vector &out) {
	ScanObjectCells(context, col, stride, masked, count, offset, out,
	                [&](nb::handle cell, idx_t row) { ScanNumpyObject(context, cell.ptr(), row, out); });
	VerifyTypeConstraints(out, count);
}

template <bool HAS_MASK>
static void ScanNumpyTimedelta(PandasColumnBindData &bind_data, NumpyArray &array, idx_t stride, idx_t count,
                               idx_t offset, Vector &out) {
	auto src_ptr = reinterpret_cast<const int64_t *>(array.Data());
	auto tgt_ptr = FlatVector::GetDataMutable<interval_t>(out);
	auto &mask = FlatVector::ValidityMutable(out);

	for (idx_t block_row = 0; block_row < count; block_row += MASK_BLOCK_ROWS) {
		auto block_end = MinValue<idx_t>(block_row + MASK_BLOCK_ROWS, count);
		// The hidden value of a masked row can overflow in the conversion
		uint64_t unmasked = HAS_MASK ? UnmaskedBlockRows(MaskedRows(bind_data), count, offset, block_row, out) : 0;
		// NaT rows are set invalid after the block, so the row loop calls nothing and keeps its constants in registers
		uint64_t not_a_time = 0;
		for (idx_t row = block_row; row < block_end; row++, unmasked >>= 1) {
			if (HAS_MASK && !(unmasked & 1)) {
				continue;
			}
			auto source_idx = stride / sizeof(int64_t) * (row + offset);
			if (src_ptr[source_idx] <= NumericLimits<int64_t>::Minimum()) {
				// pandas Not a Time (NaT)
				not_a_time |= uint64_t(1) << (row - block_row);
				continue;
			}

			int64_t micro;
			switch (bind_data.numpy_type.type) {
			case NumpyNullableType::TIMEDELTA_NS:
				micro = src_ptr[source_idx] / 1000; // ns -> us
				break;
			case NumpyNullableType::TIMEDELTA_US:
				micro = src_ptr[source_idx]; // already us
				break;
			case NumpyNullableType::TIMEDELTA_MS:
				micro = src_ptr[source_idx] * 1000; // ms -> us
				break;
			case NumpyNullableType::TIMEDELTA_S:
				micro = src_ptr[source_idx] * 1000000; // s -> us
				break;
			default:
				throw InternalException("Unexpected timedelta type");
			}

			int64_t days = micro / Interval::MICROS_PER_DAY;
			micro = micro % Interval::MICROS_PER_DAY;
			int64_t months = days / Interval::DAYS_PER_MONTH;
			days = days % Interval::DAYS_PER_MONTH;
			interval_t interval;
			interval.months = months;
			interval.days = days;
			interval.micros = micro;
			tgt_ptr[row] = interval;
		}
		if (not_a_time != 0) {
			SetInvalidRows(mask, block_row, not_a_time);
		}
	}
}

//! Call with the GIL held
template <bool HAS_MASK>
static void ScanNumpyStringRows(PandasColumnBindData &bind_data, PyObject **src_ptr, idx_t stride, idx_t count,
                                idx_t offset, Vector &out) {
	// Get the data pointer and the validity mask of the result vector
	auto tgt_ptr = FlatVector::GetDataMutable<string_t>(out);
	auto &out_mask = FlatVector::ValidityMutable(out);
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto masked_constant = import_cache.numpy.ma.masked(false);

	// Loop over every row of the arrays contents
	for (idx_t block_row = 0; block_row < count; block_row += MASK_BLOCK_ROWS) {
		auto block_end = MinValue<idx_t>(block_row + MASK_BLOCK_ROWS, count);
		uint64_t unmasked = HAS_MASK ? UnmaskedBlockRows(MaskedRows(bind_data), count, offset, block_row, out) : 0;
		for (idx_t row = block_row; row < block_end; row++, unmasked >>= 1) {
			if (HAS_MASK && !(unmasked & 1)) {
				continue;
			}
			auto source_idx = stride / sizeof(PyObject *) * (row + offset);

			// Get the pointer to the object
			PyObject *val = src_ptr[source_idx];
			if (!nb::isinstance<nb::str>(val)) {
				if (val == Py_None || (masked_constant && val == masked_constant.ptr())) {
					out_mask.SetInvalid(row);
					continue;
				}
				if (import_cache.pandas.NaT(false)) {
					// If pandas is imported, check if this is pandas.NaT
					nb::handle value(val);
					if (value.is(import_cache.pandas.NaT())) {
						out_mask.SetInvalid(row);
						continue;
					}
				}
				if (import_cache.pandas.NA(false)) {
					// If pandas is imported, check if this is pandas.NA
					nb::handle value(val);
					if (value.is(import_cache.pandas.NA())) {
						out_mask.SetInvalid(row);
						continue;
					}
				}
				if (nb::isinstance<nb::float_>(val) && std::isnan(PyFloat_AsDouble(val))) {
					out_mask.SetInvalid(row);
					continue;
				}
				if (!nb::isinstance<nb::str>(val)) {
					bind_data.object_str_val.Push(std::move(nb::str(val)));
					val = reinterpret_cast<PyObject *>(bind_data.object_str_val.LastAddedObject().ptr());
				}
			}
			nb::handle val_handle(val);
			if (!nb::isinstance<nb::str>(val_handle)) {
				out_mask.SetInvalid(row);
				continue;
			}
			Py_ssize_t size;
			// Zero copy: the bytes are the str's own or its cached UTF-8, both alive as long as the str
			auto utf8 = PyUnicode_AsUTF8AndSize(val, &size);
			if (!utf8) {
				ThrowUnencodableStr(val_handle);
			}
			tgt_ptr[row] = string_t(utf8, UnsafeNumericCast<uint32_t>(size));
		}
	}
}

static void ScanNumpyStrings(ClientContext &context, PandasColumnBindData &bind_data, PyObject **src_ptr, idx_t stride,
                             idx_t count, idx_t offset, Vector &out) {
	// A str's UTF-8 is reachable only through the C API, which needs the GIL
	PyUtil::QueryGil gil(context);
	try {
		if (bind_data.mask) {
			ScanNumpyStringRows<true>(bind_data, src_ptr, stride, count, offset, out);
		} else {
			ScanNumpyStringRows<false>(bind_data, src_ptr, stride, count, offset, out);
		}
	} catch (nb::python_error &error) {
		throw AsDuckDBError(error);
	}
}

//! 'offset' is the offset within the column
//! 'count' is the amount of values we will convert in this batch
void NumpyScan::Scan(ClientContext &context, PandasColumnBindData &bind_data, idx_t count, idx_t offset, Vector &out) {
	D_ASSERT(bind_data.pandas_col->Backend() == PandasColumnBackend::NUMPY);
	auto &numpy_col = reinterpret_cast<PandasNumpyColumn &>(*bind_data.pandas_col);
	auto &array = numpy_col.array;
	auto stride = numpy_col.stride;

	switch (bind_data.numpy_type.type) {
	case NumpyNullableType::BOOL:
		ScanNumpyMasked<bool>(bind_data, count, offset, out);
		break;
	case NumpyNullableType::UINT_8:
		ScanNumpyMasked<uint8_t>(bind_data, count, offset, out);
		break;
	case NumpyNullableType::UINT_16:
		ScanNumpyMasked<uint16_t>(bind_data, count, offset, out);
		break;
	case NumpyNullableType::UINT_32:
		ScanNumpyMasked<uint32_t>(bind_data, count, offset, out);
		break;
	case NumpyNullableType::UINT_64:
		ScanNumpyMasked<uint64_t>(bind_data, count, offset, out);
		break;
	case NumpyNullableType::INT_8:
		ScanNumpyMasked<int8_t>(bind_data, count, offset, out);
		break;
	case NumpyNullableType::INT_16:
		ScanNumpyMasked<int16_t>(bind_data, count, offset, out);
		break;
	case NumpyNullableType::INT_32:
		ScanNumpyMasked<int32_t>(bind_data, count, offset, out);
		break;
	case NumpyNullableType::INT_64:
		ScanNumpyMasked<int64_t>(bind_data, count, offset, out);
		break;
	case NumpyNullableType::FLOAT_32:
		ScanNumpyFpColumn<float>(bind_data, reinterpret_cast<const float *>(array.Data()), numpy_col.stride, count,
		                         offset, out);
		break;
	case NumpyNullableType::FLOAT_64:
		ScanNumpyFpColumn<double>(bind_data, reinterpret_cast<const double *>(array.Data()), numpy_col.stride, count,
		                          offset, out);
		break;
	case NumpyNullableType::DATETIME_NS:
	case NumpyNullableType::DATETIME_MS:
	case NumpyNullableType::DATETIME_US:
	case NumpyNullableType::DATETIME_S: {
		auto src_ptr = reinterpret_cast<const int64_t *>(array.Data());
		auto tgt_ptr = FlatVector::GetDataMutable<timestamp_t>(out);

		using timestamp_convert_func = std::function<timestamp_t(int64_t)>;
		timestamp_convert_func convert_func;

		switch (bind_data.numpy_type.type) {
		case NumpyNullableType::DATETIME_NS:
			if (bind_data.numpy_type.has_timezone) {
				// Our timezone type is US, so we need to convert from NS to US
				convert_func = Timestamp::FromEpochNanoSeconds;
			} else {
				convert_func = Timestamp::FromEpochMicroSeconds;
			}
			break;
		case NumpyNullableType::DATETIME_MS:
			if (bind_data.numpy_type.has_timezone) {
				// Our timezone type is US, so we need to convert from MS to US
				convert_func = Timestamp::FromEpochMs;
			} else {
				convert_func = Timestamp::FromEpochMicroSeconds;
			}
			break;
		case NumpyNullableType::DATETIME_US:
			convert_func = Timestamp::FromEpochMicroSeconds;
			break;
		case NumpyNullableType::DATETIME_S:
			if (bind_data.numpy_type.has_timezone) {
				// Our timezone type is US, so we need to convert from S to US
				convert_func = Timestamp::FromEpochSeconds;
			} else {
				convert_func = Timestamp::FromEpochMicroSeconds;
			}
			break;
		default:
			throw NotImplementedException("Scan for datetime of this type is not supported yet");
		};

		for (idx_t row = 0; row < count; row++) {
			auto source_idx = stride / sizeof(int64_t) * (row + offset);
			if (src_ptr[source_idx] <= NumericLimits<int64_t>::Minimum()) {
				// pandas Not a Time (NaT)
				FlatVector::ValidityMutable(out).SetInvalid(row);
				continue;
			}

			// Direct conversion, we've already matched the numpy type with the equivalent duckdb type
			auto input = timestamp_t(src_ptr[source_idx]);
			if (Value::IsFinite(input)) {
				tgt_ptr[row] = convert_func(src_ptr[source_idx]);
			} else {
				tgt_ptr[row] = input;
			}
		}
		if (bind_data.mask) {
			ApplyMask(bind_data, FlatVector::ValidityMutable(out), count, offset);
		}
		break;
	}
	case NumpyNullableType::TIMEDELTA_NS:
	case NumpyNullableType::TIMEDELTA_US:
	case NumpyNullableType::TIMEDELTA_MS:
	case NumpyNullableType::TIMEDELTA_S: {
		if (bind_data.mask) {
			ScanNumpyTimedelta<true>(bind_data, array, stride, count, offset, out);
		} else {
			ScanNumpyTimedelta<false>(bind_data, array, stride, count, offset, out);
		}
		break;
	}
	case NumpyNullableType::STRING:
	case NumpyNullableType::OBJECT: {
		// Get the source pointer of the numpy array
		auto src_ptr = (PyObject **)array.Data(); // NOLINT
		const bool is_object_col = bind_data.numpy_type.type == NumpyNullableType::OBJECT;
		if (is_object_col && bind_data.declared) {
			return NumpyScan::ScanDeclaredColumn(context, src_ptr, numpy_col.stride, MaskedRows(bind_data), count,
			                                     offset, out);
		}
		if (is_object_col && out.GetType().id() != LogicalTypeId::VARCHAR) {
			//! We have determined the underlying logical type of this object column
			return NumpyScan::ScanObjectColumn(context, src_ptr, numpy_col.stride, MaskedRows(bind_data), count, offset,
			                                   out);
		}

		ScanNumpyStrings(context, bind_data, src_ptr, numpy_col.stride, count, offset, out);
		break;
	}
	case NumpyNullableType::CATEGORY: {
		switch (out.GetType().InternalType()) {
		case PhysicalType::UINT8:
			ScanNumpyCategory<uint8_t>(array, stride, count, offset, out, bind_data.internal_categorical_type);
			break;
		case PhysicalType::UINT16:
			ScanNumpyCategory<uint16_t>(array, stride, count, offset, out, bind_data.internal_categorical_type);
			break;
		case PhysicalType::UINT32:
			ScanNumpyCategory<uint32_t>(array, stride, count, offset, out, bind_data.internal_categorical_type);
			break;
		default:
			throw InternalException("Invalid Physical Type for ENUMs");
		}
		break;
	}

	default:
		throw NotImplementedException("Unsupported pandas type");
	}
}

} // namespace duckdb

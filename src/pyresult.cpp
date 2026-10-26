#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"

#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/arrow/nanoarrow/nanoarrow.hpp"

#include <cerrno>

using namespace nanobind::literals;

namespace duckdb {

DuckDBPyResult::DuckDBPyResult(shared_ptr<engine::Result> result_p, vector<Identifier> names_override_p)
    : result(std::move(result_p)), names_override(std::move(names_override_p)) {
	if (!result) {
		throw InternalException("PyResult created without a result object");
	}
}

DuckDBPyResult::~DuckDBPyResult() {
	// The destructor must run with the GIL held: `result` and `current_chunk`
	// can transitively own Python references (registered
	// objects, arrow release callbacks, PYTHON_OBJECT vector values, etc.),
	// whose teardown calls into the Python C API. Releasing the GIL here
	// (as the previous implementation did) causes Py_DECREF / PyObject_Free
	// to run without a valid PyThreadState — see duckdb-python#456.
	try {
		DUCKDB_PY_ASSERT_GIL_HELD();
		Close();
	} catch (...) { // NOLINT
	}
}

DuckDBPyResult::Use::Use(DuckDBPyResult &result_p) : result(result_p) {
	if (result.in_use.exchange(true)) {
		throw ResourceInUseException(
		    "result is busy with another call; it refuses instead of waiting, so read one result from one thread at "
		    "a time");
	}
	if (!result.result) {
		result.in_use = false;
		throw InvalidInputException("The result is closed");
	}
	pinned = result.result;
}

DuckDBPyResult::Use::~Use() {
	result.in_use = false;
}

const vector<Identifier> &DuckDBPyResult::ResultNames() const {
	if (!names_override.empty()) {
		return names_override;
	}
	return result->Names();
}

const ClientProperties &DuckDBPyResult::GetClientProperties() const {
	return result->GetClientProperties();
}

vector<string> DuckDBPyResult::GetNames() {
	if (Empty()) {
		throw InternalException("Calling GetNames without a result object");
	}
	return IdentifiersToStrings(ResultNames());
}

const vector<LogicalType> &DuckDBPyResult::GetTypes() const {
	if (Empty()) {
		throw InternalException("Calling GetTypes without a result object");
	}
	return result->Types();
}

unique_ptr<DataChunk> DuckDBPyResult::FetchChunk() {
	if (Empty()) {
		throw InternalException("FetchChunk called without a result object");
	}
	return FetchNext();
}

unique_ptr<DataChunk> DuckDBPyResult::FetchNext() {
	auto chunk = FetchNextRaw();
	if (chunk) {
		chunk->Flatten();
	}
	return chunk;
}

unique_ptr<DataChunk> DuckDBPyResult::FetchNextRaw() {
	if (result->GetFormat().IsArrow()) {
		throw InternalException("Rows fetched from a result submitted in the Arrow format");
	}
	return result->FetchChunk(DuckDBPyConnection::CheckSignals);
}

unique_ptr<DataChunk> DuckDBPyResult::TakeBufferedRows() {
	unique_ptr<DataChunk> remainder;
	if (current_chunk && chunk_offset < current_chunk->size()) {
		remainder = make_uniq<DataChunk>();
		remainder->Initialize(Allocator::DefaultAllocator(), current_chunk->GetTypes());
		current_chunk->Copy(*remainder, chunk_offset);
	}
	current_chunk.reset();
	chunk_offset = 0;
	return remainder;
}

unique_ptr<ColumnDataCollection> DuckDBPyResult::TakeRows(const engine::InterruptCheck &check) {
	if (Empty()) {
		throw InternalException("TakeRows called without a result object");
	}
	auto buffered = TakeBufferedRows();
	unique_ptr<ColumnDataCollection> rows;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		auto taken = result->TakeChunks(check);
		if (!buffered) {
			return taken;
		}
		// Combine only appends, so the buffered rows, which come first, go into a new collection and the taken rows
		// follow
		rows = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), result->Types());
		rows->Append(*buffered);
		rows->Combine(*taken);
	}
	return rows;
}

Optional<nb::tuple> DuckDBPyResult::Fetchone() {
	if (Empty()) {
		throw InvalidInputException("result closed");
	}
	// Asking the engine on every row costs two locks and a GIL switch, so an interrupt or a failure is reported once
	// the rows of the current chunk run out
	if (!current_chunk || chunk_offset >= current_chunk->size()) {
		nb::gil_scoped_release release;
		current_chunk = FetchNext();
		chunk_offset = 0;
	}

	if (!current_chunk || current_chunk->size() == 0) {
		return nb::none();
	}
	auto &types = GetTypes();
	auto &client_properties = GetClientProperties();
	duckdb::PyUtil::TupleBuilder row(types.size());
	for (idx_t col_idx = 0; col_idx < types.size(); col_idx++) {
		auto &mask = FlatVector::Validity(current_chunk->data[col_idx]);
		if (!mask.RowIsValid(chunk_offset)) {
			row.append(nb::none());
		} else {
			auto val = current_chunk->data[col_idx].GetValue(chunk_offset);
			row.append(PythonObject::FromValue(val, types[col_idx], client_properties));
		}
	}
	chunk_offset++;
	return row.take();
}

nb::list DuckDBPyResult::Fetchmany(idx_t size) {
	nb::list res;
	for (idx_t i = 0; i < size; i++) {
		auto fres = Fetchone();
		if (fres.is_none()) {
			break;
		}
		res.append(fres);
	}
	return res;
}

nb::list DuckDBPyResult::Fetchall() {
	nb::list res;
	while (true) {
		auto fres = Fetchone();
		if (fres.is_none()) {
			break;
		}
		res.append(fres);
	}
	return res;
}

nb::dict DuckDBPyResult::FetchNumpy() {
	return FetchNumpyInternal();
}

void DuckDBPyResult::FillNumpy(nb::dict &res, idx_t col_idx, NumpyResultConversion &conversion, const char *name) {
	if (GetTypes()[col_idx].id() == LogicalTypeId::ENUM) {
		auto &import_cache = *DuckDBPyConnection::ImportCache();
		auto pandas_categorical = import_cache.pandas.Categorical();
		auto categorical_dtype = import_cache.pandas.CategoricalDtype();
		if (!pandas_categorical || !categorical_dtype) {
			throw InvalidInputException("'pandas' is required for this operation but it was not installed");
		}

		// first we (might) need to create the categorical type
		if (categories_type.find(col_idx) == categories_type.end()) {
			// Equivalent to: pandas.CategoricalDtype(['a', 'b'], ordered=True)
			categories_type[col_idx] = categorical_dtype(categories[col_idx], true);
		}
		// Equivalent to: pandas.Categorical.from_codes(codes=[0, 1, 0, 1], dtype=dtype)
		res[name] = pandas_categorical.attr("from_codes")(conversion.ToArray(col_idx),
		                                                  nb::arg("dtype") = categories_type[col_idx]);
		if (!conversion.ToPandas()) {
			res[name] = res[name].attr("to_numpy")();
		}
	} else {
		res[name] = conversion.ToArray(col_idx);
	}
}

void InsertCategory(const vector<LogicalType> &types, unordered_map<idx_t, nb::list> &categories) {
	for (idx_t col_idx = 0; col_idx < types.size(); col_idx++) {
		auto &type = types[col_idx];
		if (type.id() == LogicalTypeId::ENUM) {
			// It's an ENUM type, in addition to converting the codes we must convert the categories
			if (categories.find(col_idx) == categories.end()) {
				auto &categories_list = EnumType::GetValuesInsertOrder(type);
				auto categories_size = EnumType::GetSize(type);
				for (idx_t i = 0; i < categories_size; i++) {
					categories[col_idx].append(nb::cast(categories_list.GetValue(i).ToString()));
				}
			}
		}
	}
}

std::unique_ptr<NumpyResultConversion> DuckDBPyResult::InitializeNumpyConversion(bool pandas) {
	if (Empty()) {
		throw InvalidInputException("result closed");
	}

	idx_t initial_capacity = STANDARD_VECTOR_SIZE * 2ULL;
	if (result->IsRetained()) {
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		initial_capacity = result->RowCount(DuckDBPyConnection::CheckSignals);
	}

	auto conversion =
	    std::make_unique<NumpyResultConversion>(GetTypes(), initial_capacity, GetClientProperties(), pandas);
	return conversion;
}

nb::dict DuckDBPyResult::FetchNumpyInternal(bool chunked, idx_t vectors_per_chunk,
                                            std::unique_ptr<NumpyResultConversion> conversion_p) {
	if (Empty()) {
		throw InvalidInputException("result closed");
	}
	if (!conversion_p) {
		conversion_p = InitializeNumpyConversion();
	}
	auto &conversion = *conversion_p;
	if (!chunked) {
		vectors_per_chunk = NumericLimits<idx_t>::Maximum();
	}

	idx_t count_vec = 0;
	if (vectors_per_chunk > 0) {
		auto buffered = TakeBufferedRows();
		if (buffered) {
			conversion.Append(*buffered);
			count_vec++;
		}
	}
	for (; count_vec < vectors_per_chunk; count_vec++) {
		unique_ptr<DataChunk> chunk;
		{
			DUCKDB_PY_ASSERT_GIL_HELD();
			nb::gil_scoped_release release;
			chunk = FetchNextRaw();
		}
		if (!chunk || chunk->size() == 0) {
			break;
		}
		conversion.Append(*chunk);
	}
	InsertCategory(GetTypes(), categories);

	// now that we have materialized the result in contiguous arrays, construct the actual NumPy arrays or categorical
	// types
	nb::dict res;
	auto names = ResultNames();
	QueryResult::DeduplicateColumns(names);
	for (idx_t col_idx = 0; col_idx < names.size(); col_idx++) {
		auto &name = names[col_idx];
		FillNumpy(res, col_idx, conversion, name.c_str());
	}
	return res;
}

static void ReplaceDFColumn(PandasDataFrame &df, const char *col_name, idx_t idx, const nb::handle &new_value) {
	df.attr("drop")("columns"_a = col_name, "inplace"_a = true);
	df.attr("insert")(idx, col_name, new_value, "allow_duplicates"_a = false);
}

// TODO: unify these with an enum/flag to indicate which conversions to do
void DuckDBPyResult::ConvertDateTimeTypes(PandasDataFrame &df, bool date_as_object) const {
	auto names = nb::cast<vector<string>>(df.attr("columns"));

	auto &types = GetTypes();
	for (idx_t i = 0; i < types.size(); i++) {
		if (types[i] == LogicalType::TIMESTAMP_TZ) {
			// first localize to UTC then convert to timezone_config
			auto utc_local = df[names[i].c_str()].attr("dt").attr("tz_localize")("UTC");
			auto new_value = utc_local.attr("dt").attr("tz_convert")(GetClientProperties().time_zone);
			// We need to create the column anew because the exact dt changed to a new timezone
			ReplaceDFColumn(df, names[i].c_str(), i, new_value);
		} else if (date_as_object && types[i] == LogicalType::DATE) {
			nb::object new_value = df[names[i].c_str()].attr("dt").attr("date");
			ReplaceDFColumn(df, names[i].c_str(), i, new_value);
		}
	}
}

static nb::object ConvertNumpyDtype(nb::handle numpy_array) {
	DUCKDB_PY_ASSERT_GIL_HELD();
	auto &import_cache = *DuckDBPyConnection::ImportCache();

	auto dtype = numpy_array.attr("dtype");
	if (!duckdb::PyUtil::IsInstance(numpy_array, import_cache.numpy.ma.masked_array())) {
		return dtype;
	}

	auto numpy_type = ConvertNumpyType(dtype);
	switch (numpy_type.type) {
	case NumpyNullableType::BOOL: {
		return import_cache.pandas.BooleanDtype()();
	}
	case NumpyNullableType::UINT_8: {
		return import_cache.pandas.UInt8Dtype()();
	}
	case NumpyNullableType::UINT_16: {
		return import_cache.pandas.UInt16Dtype()();
	}
	case NumpyNullableType::UINT_32: {
		return import_cache.pandas.UInt32Dtype()();
	}
	case NumpyNullableType::UINT_64: {
		return import_cache.pandas.UInt64Dtype()();
	}
	case NumpyNullableType::INT_8: {
		return import_cache.pandas.Int8Dtype()();
	}
	case NumpyNullableType::INT_16: {
		return import_cache.pandas.Int16Dtype()();
	}
	case NumpyNullableType::INT_32: {
		return import_cache.pandas.Int32Dtype()();
	}
	case NumpyNullableType::INT_64: {
		return import_cache.pandas.Int64Dtype()();
	}
	case NumpyNullableType::FLOAT_32:
	case NumpyNullableType::FLOAT_64:
	case NumpyNullableType::FLOAT_16: // there is no pandas.Float16Dtype
	default:
		return dtype;
	}
}

PandasDataFrame DuckDBPyResult::FrameFromNumpy(bool date_as_object, const nb::handle &o) {
	DUCKDB_PY_ASSERT_GIL_HELD();
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto pandas = import_cache.pandas();
	if (!pandas) {
		throw InvalidInputException("'pandas' is required for this operation but it was not installed");
	}

	nb::object items = o.attr("items")();
	for (const nb::handle &item : items) {
		// Each item is a tuple of (key, value)
		auto key_value = nb::cast<nb::tuple>(item);
		nb::handle key = key_value[0];   // Access the first element (key)
		nb::handle value = key_value[1]; // Access the second element (value)

		auto dtype = ConvertNumpyDtype(value);
		if (duckdb::PyUtil::IsInstance(value, import_cache.numpy.ma.masked_array())) {
			// o[key] = pd.Series(value.filled(pd.NA), dtype=dtype)
			auto series = pandas.attr("Series")(value.attr("data"), nb::arg("dtype") = dtype);
			series.attr("__setitem__")(value.attr("mask"), import_cache.pandas.NA());
			o.attr("__setitem__")(key, series);
		}
	}

	PandasDataFrame df = nb::cast<PandasDataFrame>(pandas.attr("DataFrame").attr("from_dict")(o));
	// Convert TZ and (optionally) Date types
	ConvertDateTimeTypes(df, date_as_object);

	auto names = nb::cast<vector<string>>(df.attr("columns"));
	D_ASSERT(GetTypes().size() == names.size());
	return df;
}

PandasDataFrame DuckDBPyResult::FetchDF(bool date_as_object) {
	auto conversion = InitializeNumpyConversion(true);
	return FrameFromNumpy(date_as_object, FetchNumpyInternal(false, 1, std::move(conversion)));
}

PandasDataFrame DuckDBPyResult::FetchDFChunk(idx_t num_of_vectors, bool date_as_object) {
	auto conversion = InitializeNumpyConversion(true);
	return FrameFromNumpy(date_as_object, FetchNumpyInternal(true, num_of_vectors, std::move(conversion)));
}

nb::dict DuckDBPyResult::FetchPyTorch() {
	auto result_dict = FetchNumpyInternal();
	auto from_numpy = nb::module_::import_("torch").attr("from_numpy");
	for (auto item : result_dict) { // nanobind dict iteration yields std::pair<handle,handle> by value
		result_dict[item.first] = from_numpy(item.second);
	}
	return result_dict;
}

nb::dict DuckDBPyResult::FetchTF() {
	auto result_dict = FetchNumpyInternal();
	auto convert_to_tensor = nb::module_::import_("tensorflow").attr("convert_to_tensor");
	for (auto item : result_dict) { // nanobind dict iteration yields std::pair<handle,handle> by value
		result_dict[item.first] = convert_to_tensor(item.second);
	}
	return result_dict;
}

//! Arrow is produced only by results submitted in the Arrow format
static void RequireArrowFormat(const engine::Result &result) {
	if (!result.GetFormat().IsArrow()) {
		throw InternalException("Arrow fetched from a result not submitted in the Arrow format");
	}
}

template <typename T>
T DuckDBPyResult::RunWithArrowSchema(const std::function<T(const ArrowSchema &)> &fun, bool dedup_col_names) {
	D_ASSERT(!Empty());
	auto identifiers = ResultNames();
	if (dedup_col_names) {
		QueryResult::DeduplicateColumns(identifiers);
	}
	auto names = IdentifiersToStrings(identifiers);

	ArrowSchema arrow_schema;
	result->BuildArrowSchema(names, arrow_schema);
	return fun(arrow_schema);
}

duckdb::pyarrow::Table DuckDBPyResult::FetchArrowTable(const idx_t rows_per_batch, const bool to_polars) {
	if (Empty()) {
		throw InvalidInputException("There is no query result");
	}
	RequireArrowFormat(*result);
	return RunWithArrowSchema<duckdb::pyarrow::Table>(
	    [&](const ArrowSchema &schema) -> duckdb::pyarrow::Table {
		    auto pyarrow_schema = pyarrow::ToPyArrowSchema(schema);
		    vector<unique_ptr<ArrowArrayWrapper>> arrays;
		    {
			    DUCKDB_PY_ASSERT_GIL_HELD();
			    nb::gil_scoped_release release;
			    arrays = result->TakeArrays(DuckDBPyConnection::CheckSignals);
		    }
		    nb::list batches;
		    for (auto &array : arrays) {
			    ArrowArray data;
			    array->MoveTo(data);
			    TransformDuckToArrowChunk(pyarrow_schema, data, batches);
		    }
		    return pyarrow::ToArrowTable(std::move(batches), pyarrow_schema);
	    },
	    to_polars);
}

void DuckDBPyResult::CheckBatchSize(idx_t rows_per_batch) {
	if (rows_per_batch == 0) {
		throw std::runtime_error("Approximate Batch Size of Record Batch MUST be higher than 0");
	}
}

namespace {

//! pyarrow releases an imported stream with the GIL held, and releasing the engine's stream ends
//! the query and joins its tasks, one of which may be inside a Python UDF waiting for the GIL.
//! Pulling batches has the same shape when a caller holds the GIL, so every callback drops it.
template <class FUN>
auto WithoutGil(FUN &&fun) -> decltype(fun()) {
	duckdb::PyUtil::GilReleaseIfHeld release;
	return fun();
}

//! The result the arrays are popped from, and its schema, copied while the producing transaction was live
//! because get_schema cannot read the catalog again.
struct ArrowStreamState {
	shared_ptr<engine::Result> result;
	ArrowSchema cached_schema {};
	ErrorData last_error;

	~ArrowStreamState() {
		if (cached_schema.release) {
			cached_schema.release(&cached_schema);
		}
	}
};

ArrowStreamState &StreamState(ArrowArrayStream *stream) {
	return *static_cast<ArrowStreamState *>(stream->private_data);
}

//! No exception may cross a C callback, so each one reports through the errno-style return code.
int ArrowStreamGetSchema(ArrowArrayStream *stream, ArrowSchema *out) {
	if (!stream->release || !stream->private_data) {
		return EINVAL;
	}
	auto &state = StreamState(stream);
	return WithoutGil([&]() -> int {
		try {
			if (!state.cached_schema.release) {
				state.last_error = ErrorData("arrow stream: the schema is unavailable");
				return EINVAL;
			}
			// The consumer owns and releases what get_schema returns, independently of the stream
			if (duckdb_nanoarrow::ArrowSchemaDeepCopy(&state.cached_schema, out) != NANOARROW_OK) {
				state.last_error = ErrorData("arrow stream: failed to copy the schema");
				return ENOMEM;
			}
			return 0;
		} catch (std::exception &ex) {
			try {
				state.last_error = ErrorData(ex);
			} catch (...) { // NOLINT: best-effort
			}
			return EIO;
		} catch (...) {
			return EIO;
		}
	});
}

int ArrowStreamGetNext(ArrowArrayStream *stream, ArrowArray *out) {
	if (!stream->release || !stream->private_data) {
		return EINVAL;
	}
	auto &state = StreamState(stream);
	return WithoutGil([&]() -> int {
		try {
			// The consumer drives the stream and owns interruption, so no signal check runs here
			auto array = state.result->FetchArray([]() {});
			if (!array) {
				// The end of the stream, which the interface spells as a released array
				out->release = nullptr;
				return 0;
			}
			array->MoveTo(*out);
			return 0;
		} catch (std::exception &ex) {
			try {
				state.last_error = ErrorData(ex);
			} catch (...) { // NOLINT: best-effort
			}
			return EIO;
		} catch (...) {
			return EIO;
		}
	});
}

const char *ArrowStreamGetLastError(ArrowArrayStream *stream) {
	if (!stream->release || !stream->private_data) {
		return "arrow stream was released";
	}
	auto &error = StreamState(stream).last_error;
	return error.HasError() ? error.Message().c_str() : nullptr;
}

void ArrowStreamRelease(ArrowArrayStream *stream) {
	if (!stream || !stream->release) {
		return;
	}
	stream->release = nullptr;
	auto state = static_cast<ArrowStreamState *>(stream->private_data);
	stream->private_data = nullptr;
	if (!state) {
		return;
	}
	WithoutGil([&]() {
		try {
			state->result->Close();
		} catch (...) { // NOLINT: best-effort cleanup
		}
	});
	delete state;
}

//! Releases a stream that was never handed over, for example when the import into pyarrow throws
struct ArrowArrayStreamGuard {
	ArrowArrayStream stream;
	~ArrowArrayStreamGuard() {
		if (stream.release) {
			stream.release(&stream);
		}
	}
};

} // namespace

ArrowArrayStream DuckDBPyResult::FetchArrowArrayStream(idx_t rows_per_batch) {
	RequireArrowFormat(*result);
	auto state = make_uniq<ArrowStreamState>();
	if (names_override.empty()) {
		result->CopyArrowSchema(state->cached_schema);
	} else {
		result->BuildArrowSchema(IdentifiersToStrings(names_override), state->cached_schema);
	}
	state->result = std::move(result);
	current_chunk.reset();
	chunk_offset = 0;

	ArrowArrayStream result_stream;
	result_stream.get_schema = ArrowStreamGetSchema;
	result_stream.get_next = ArrowStreamGetNext;
	result_stream.get_last_error = ArrowStreamGetLastError;
	result_stream.release = ArrowStreamRelease;
	result_stream.private_data = state.release();
	return result_stream;
}

duckdb::pyarrow::RecordBatchReader DuckDBPyResult::FetchRecordBatchReader(idx_t rows_per_batch) {
	if (Empty()) {
		throw InvalidInputException("There is no query result");
	}
	CheckBatchSize(rows_per_batch);
	auto pyarrow_lib_module = nb::module_::import_("pyarrow").attr("lib");
	auto record_batch_reader_func = pyarrow_lib_module.attr("RecordBatchReader").attr("_import_from_c");
	ArrowArrayStreamGuard guard {FetchArrowArrayStream(rows_per_batch)};
	nb::object record_batch_reader = record_batch_reader_func((uint64_t)&guard.stream); // NOLINT
	return nb::cast<duckdb::pyarrow::RecordBatchReader>(record_batch_reader);
}

static void ArrowArrayStreamPyCapsuleDestructor(void *data) noexcept {
	if (!data) {
		return;
	}
	auto arrow_stream = reinterpret_cast<ArrowArrayStream *>(data);
	if (arrow_stream->release) {
		arrow_stream->release(arrow_stream);
	}
	delete arrow_stream;
}

nb::object DuckDBPyResult::FetchArrowCapsule(const idx_t rows_per_batch) {
	if (Empty()) {
		throw InvalidInputException("There is no query result");
	}
	CheckBatchSize(rows_per_batch);
	auto inner_stream = FetchArrowArrayStream(rows_per_batch);
	auto arrow_stream = new ArrowArrayStream();
	*arrow_stream = inner_stream;
	return nb::capsule(arrow_stream, "arrow_array_stream", ArrowArrayStreamPyCapsuleDestructor);
}

nb::list DuckDBPyResult::GetDescription(const vector<string> &names, const vector<LogicalType> &types) {
	nb::list desc;

	for (idx_t col_idx = 0; col_idx < names.size(); col_idx++) {
		auto py_name = nb::str(names[col_idx].c_str(), names[col_idx].size());
		auto py_type = DuckDBPyType(types[col_idx]);
		desc.append(nb::make_tuple(py_name, py_type, nb::none(), nb::none(), nb::none(), nb::none(), nb::none()));
	}
	return desc;
}

void DuckDBPyResult::Close() {
	if (result) {
		// Ending the query waits for its running tasks, and a task inside a Python UDF cannot finish
		// until the GIL is free.
		duckdb::PyUtil::GilReleaseIfHeld release;
		result->Close();
	}
	result.reset();
	current_chunk.reset();
	chunk_offset = 0;
}

void DuckDBPyResult::Complete() {
	if (Empty()) {
		return;
	}
	current_chunk.reset();
	chunk_offset = 0;
	const bool arrow = result->GetFormat().IsArrow();
	// Each unit is dropped with the GIL held, since a chunk can hold Python objects
	while (true) {
		unique_ptr<DataChunk> chunk;
		unique_ptr<ArrowArrayWrapper> array;
		{
			DUCKDB_PY_ASSERT_GIL_HELD();
			nb::gil_scoped_release release;
			if (arrow) {
				array = result->FetchArray(DuckDBPyConnection::CheckSignals);
			} else {
				chunk = result->FetchChunk(DuckDBPyConnection::CheckSignals);
			}
		}
		if (!chunk && !array) {
			break;
		}
	}
	Close();
}

bool DuckDBPyResult::HasWritesToRun() const {
	// A result that stopped running, a failed one included, has nothing left to write
	if (!result || !result->Running()) {
		return false;
	}
	// A statement that expands into several is treated as writing until it ran and its properties are known
	return !result->MetadataAvailable() || !result->Properties().modified_databases.empty();
}

int64_t DuckDBPyResult::Rowcount() {
	if (rowcount_override >= 0) {
		return rowcount_override;
	}
	return result ? result->ChangedRows() : -1;
}

void DuckDBPyResult::OverrideRowcount(int64_t rowcount) {
	rowcount_override = rowcount;
}

} // namespace duckdb

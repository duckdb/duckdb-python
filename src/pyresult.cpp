#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"

#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_util.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/enums/stream_execution_result.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb/common/arrow/arrow_query_result.hpp"
#include "duckdb/common/arrow/physical_arrow_collector.hpp"
#include "duckdb/main/chunk_scan_state/query_result.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/stream_query_result.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/column_data_ref.hpp"

using namespace nanobind::literals;

namespace duckdb {

DuckDBPyResult::DuckDBPyResult(unique_ptr<QueryResult> result_p) : result(std::move(result_p)) {
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
		D_ASSERT(duckdb::PyUtil::GilCheck());
		result.reset();
		current_chunk.reset();
	} catch (...) { // NOLINT
	}
}

ClientProperties DuckDBPyResult::GetClientProperties() {
	return result->client_properties;
}

const vector<string> &DuckDBPyResult::GetNames() {
	if (!result) {
		throw InternalException("Calling GetNames without a result object");
	}
	return result->names;
}

const vector<LogicalType> &DuckDBPyResult::GetTypes() {
	if (!result) {
		throw InternalException("Calling GetTypes without a result object");
	}
	return result->types;
}

int64_t DuckDBPyResult::GetRowcount() {
	if (!result || result->HasError()) {
		return -1;
	}
	if (result->properties.return_type != StatementReturnType::CHANGED_ROWS) {
		// The row count of a SELECT (or a statement that returns nothing) is not known without fully
		// consuming the result - report -1, as permitted by the DB-API 2.0 spec for 'rowcount'.
		return -1;
	}
	if (result->type == QueryResultType::STREAM_RESULT) {
		// CHANGED_ROWS statements always produce a single already-computed row, so materializing here
		// does not trigger any additional query execution - it just changes the in-memory representation.
		auto &stream_result = result->Cast<StreamQueryResult>();
		auto materialized = stream_result.Materialize();
		if (!materialized || materialized->HasError()) {
			return -1;
		}
		result = std::move(materialized);
	}
	if (result->type != QueryResultType::MATERIALIZED_RESULT) {
		// e.g. an Arrow result - can't peek at the value without disturbing it.
		return -1;
	}
	auto &materialized_result = result->Cast<MaterializedQueryResult>();
	if (materialized_result.RowCount() != 1 || materialized_result.ColumnCount() != 1) {
		return -1;
	}
	return materialized_result.GetValue(0, 0).GetValue<int64_t>();
}

unique_ptr<DataChunk> DuckDBPyResult::FetchChunk() {
	if (!result) {
		throw InternalException("FetchChunk called without a result object");
	}
	return FetchNext(*result);
}

unique_ptr<DataChunk> DuckDBPyResult::FetchNext(QueryResult &query_result) {
	if (!result_closed && query_result.type == QueryResultType::STREAM_RESULT &&
	    !query_result.Cast<StreamQueryResult>().IsOpen()) {
		result_closed = true;
		return nullptr;
	}
	if (query_result.type == QueryResultType::STREAM_RESULT) {
		auto &stream_result = query_result.Cast<StreamQueryResult>();
		StreamExecutionResult execution_result;
		while (!StreamQueryResult::IsChunkReady(execution_result = stream_result.ExecuteTask())) {
			{
				nb::gil_scoped_acquire gil;
				if (PyErr_CheckSignals() != 0) {
					throw std::runtime_error("Query interrupted");
				}
			}
			if (execution_result == StreamExecutionResult::BLOCKED) {
				stream_result.WaitForTask();
			}
		}
		if (execution_result == StreamExecutionResult::EXECUTION_CANCELLED) {
			throw InvalidInputException("The execution of the query was cancelled before it could finish, likely "
			                            "caused by executing a different query");
		}
		if (execution_result == StreamExecutionResult::EXECUTION_ERROR) {
			stream_result.ThrowError();
		}
	}
	auto chunk = query_result.Fetch();
	if (query_result.HasError()) {
		query_result.ThrowError();
	}
	return chunk;
}

unique_ptr<DataChunk> DuckDBPyResult::FetchNextRaw(QueryResult &query_result) {
	if (!result_closed && query_result.type == QueryResultType::STREAM_RESULT &&
	    !query_result.Cast<StreamQueryResult>().IsOpen()) {
		result_closed = true;
		return nullptr;
	}
	auto chunk = query_result.FetchRaw();
	if (query_result.HasError()) {
		query_result.ThrowError();
	}
	return chunk;
}

Optional<nb::tuple> DuckDBPyResult::Fetchone() {
	if (!result) {
		throw InvalidInputException("result closed");
	}
	if (!current_chunk || chunk_offset >= current_chunk->size()) {
		nb::gil_scoped_release release;
		current_chunk = FetchNext(*result);
		chunk_offset = 0;
	}

	if (!current_chunk || current_chunk->size() == 0) {
		return nb::none();
	}
	duckdb::PyUtil::TupleBuilder row(result->types.size());
	for (idx_t col_idx = 0; col_idx < result->types.size(); col_idx++) {
		auto &mask = FlatVector::Validity(current_chunk->data[col_idx]);
		if (!mask.RowIsValid(chunk_offset)) {
			row.append(nb::none());
		} else {
			auto val = current_chunk->data[col_idx].GetValue(chunk_offset);
			row.append(PythonObject::FromValue(val, result->types[col_idx], result->client_properties));
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
	if (result->types[col_idx].id() == LogicalTypeId::ENUM) {
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

void InsertCategory(QueryResult &result, unordered_map<idx_t, nb::list> &categories) {
	for (idx_t col_idx = 0; col_idx < result.types.size(); col_idx++) {
		auto &type = result.types[col_idx];
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
	if (!result) {
		throw InvalidInputException("result closed");
	}

	idx_t initial_capacity = STANDARD_VECTOR_SIZE * 2ULL;
	if (result->type == QueryResultType::MATERIALIZED_RESULT) {
		// materialized query result: we know exactly how much space we need
		auto &materialized = result->Cast<MaterializedQueryResult>();
		initial_capacity = materialized.RowCount();
	}

	auto conversion =
	    std::make_unique<NumpyResultConversion>(result->types, initial_capacity, result->client_properties, pandas);
	return conversion;
}

nb::dict DuckDBPyResult::FetchNumpyInternal(bool stream, idx_t vectors_per_chunk,
                                            std::unique_ptr<NumpyResultConversion> conversion_p) {
	if (!result) {
		throw InvalidInputException("result closed");
	}
	if (!conversion_p) {
		conversion_p = InitializeNumpyConversion();
	}
	auto &conversion = *conversion_p;

	if (result->type == QueryResultType::MATERIALIZED_RESULT) {
		auto &materialized = result->Cast<MaterializedQueryResult>();
		for (auto &chunk : materialized.Collection().Chunks()) {
			conversion.Append(chunk);
		}
		InsertCategory(materialized, categories);
		materialized.Collection().Reset();
	} else {
		D_ASSERT(result->type == QueryResultType::STREAM_RESULT);
		if (!stream) {
			vectors_per_chunk = NumericLimits<idx_t>::Maximum();
		}
		auto &stream_result = result->Cast<StreamQueryResult>();
		for (idx_t count_vec = 0; count_vec < vectors_per_chunk; count_vec++) {
			if (!stream_result.IsOpen()) {
				break;
			}
			unique_ptr<DataChunk> chunk;
			{
				D_ASSERT(duckdb::PyUtil::GilCheck());
				nb::gil_scoped_release release;
				chunk = FetchNextRaw(stream_result);
			}
			if (!chunk || chunk->size() == 0) {
				//! finished
				break;
			}
			conversion.Append(*chunk);
			InsertCategory(stream_result, categories);
		}
	}

	// now that we have materialized the result in contiguous arrays, construct the actual NumPy arrays or categorical
	// types
	nb::dict res;
	auto names = result->names;
	QueryResult::DeduplicateColumns(names);
	for (idx_t col_idx = 0; col_idx < result->names.size(); col_idx++) {
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

	for (idx_t i = 0; i < result->ColumnCount(); i++) {
		if (result->types[i] == LogicalType::TIMESTAMP_TZ) {
			// first localize to UTC then convert to timezone_config
			auto utc_local = df[names[i].c_str()].attr("dt").attr("tz_localize")("UTC");
			auto new_value = utc_local.attr("dt").attr("tz_convert")(result->client_properties.time_zone);
			// We need to create the column anew because the exact dt changed to a new timezone
			ReplaceDFColumn(df, names[i].c_str(), i, new_value);
		} else if (date_as_object && result->types[i] == LogicalType::DATE) {
			nb::object new_value = df[names[i].c_str()].attr("dt").attr("date");
			ReplaceDFColumn(df, names[i].c_str(), i, new_value);
		}
	}
}

static nb::object ConvertNumpyDtype(nb::handle numpy_array) {
	D_ASSERT(duckdb::PyUtil::GilCheck());
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
	D_ASSERT(duckdb::PyUtil::GilCheck());
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
	D_ASSERT(result->ColumnCount() == names.size());
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

// `SELECT * FROM <column data>` over `collection`, executed as a SelectStatement rather
// than via PendingQuery(relation) - the latter's RelationStatement stringifies the whole
// collection (O(rows)). The ColumnDataRef owns the collection, so it outlives the result
// (needed by the lazy stream path).
static unique_ptr<SelectStatement> MakeColumnDataScanStatement(unique_ptr<ColumnDataCollection> collection,
                                                               const vector<string> &names) {
	// The binder rejects duplicate column names; callers restore the originals afterwards.
	auto deduplicated_names = names;
	QueryResult::DeduplicateColumns(deduplicated_names);
	// Core's ColumnDataRef now takes case-insensitive Identifiers; promote the runtime names explicitly.
	vector<Identifier> expected_names;
	expected_names.reserve(deduplicated_names.size());
	for (auto &name : deduplicated_names) {
		expected_names.emplace_back(std::move(name));
	}
	auto table_ref = make_uniq<ColumnDataRef>(std::move(collection), std::move(expected_names));
	table_ref->alias = "materialized"; // binding asserts on an unset alias
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	select_node->from_table = std::move(table_ref);
	auto select = make_uniq<SelectStatement>();
	select->node = std::move(select_node);
	return select;
}

// Re-feed a materialized result through a PhysicalArrowCollector on the user's own context
// (parallel conversion, correct Arrow settings) -> ArrowQueryResult.
void DuckDBPyResult::PromoteMaterializedToArrow(idx_t batch_size) {
	D_ASSERT(result->type == QueryResultType::MATERIALIZED_RESULT);
	auto client_context = result->client_properties.client_context;
	if (!client_context) {
		throw InternalException("Cannot promote result to Arrow: the originating client context is gone");
	}
	auto context = client_context->shared_from_this();
	auto &materialized = result->Cast<MaterializedQueryResult>();
	auto names = result->names;
	auto select = MakeColumnDataScanStatement(materialized.TakeCollection(), names);

	auto &config = ClientConfig::GetConfig(*context);
	ScopedConfigSetting scoped_setting(
	    config,
	    [batch_size](ClientConfig &config) {
		    config.get_result_collector = [batch_size](ClientContext &context, PreparedStatementData &data) {
			    return PhysicalArrowCollector::Create(context, data, batch_size);
		    };
	    },
	    [](ClientConfig &config) { config.get_result_collector = nullptr; });

	unique_ptr<QueryResult> new_result;
	{
		D_ASSERT(duckdb::PyUtil::GilCheck());
		nb::gil_scoped_release release;
		auto pending_query = context->PendingQuery(std::move(select), QueryParameters(false));
		new_result = DuckDBPyConnection::CompletePendingQuery(*pending_query);
	}
	if (new_result->HasError()) {
		new_result->ThrowError();
	}
	new_result->names = std::move(names); // restore names de-duplicated by re-binding
	result = std::move(new_result);
}

template <typename T>
T DuckDBPyResult::RunWithArrowSchema(const std::function<T(const ArrowSchema &)> &fun, bool dedup_col_names) {
	D_ASSERT(result);
	if (!result->client_properties.client_context) {
		throw ConnectionException("Cannot fetch arrow schema without a valid connection");
	}
	auto ctx = result->client_properties.client_context->shared_from_this();

	auto names = result->names;
	if (dedup_col_names) {
		QueryResult::DeduplicateColumns(names);
	}

	ArrowSchema arrow_schema;
	ctx->RunFunctionInTransaction(
	    [&] { ArrowConverter::ToArrowSchema(&arrow_schema, result->types, names, result->client_properties); });

	return fun(arrow_schema);
}

duckdb::pyarrow::Table DuckDBPyResult::MaterializedResultToArrowTable(const ArrowSchema &arrow_schema,
                                                                      const idx_t rows_per_batch) {
	D_ASSERT(result);
	D_ASSERT(result->type == QueryResultType::MATERIALIZED_RESULT || result->type == QueryResultType::ARROW_RESULT);

	auto pyarrow_schema = pyarrow::ToPyArrowSchema(arrow_schema);
	if (result->type == QueryResultType::MATERIALIZED_RESULT) {
		PromoteMaterializedToArrow(rows_per_batch);
	}
	nb::list batches;
	auto &arrow_result = result->Cast<ArrowQueryResult>();
	auto arrays = arrow_result.ConsumeArrays();
	for (auto &array : arrays) {
		ArrowArray data = array->arrow_array;
		array->arrow_array.release = nullptr;
		TransformDuckToArrowChunk(pyarrow_schema, data, batches);
	}
	return pyarrow::ToArrowTable(std::move(batches), pyarrow_schema);
}

duckdb::pyarrow::Table DuckDBPyResult::FetchArrowTable(const idx_t rows_per_batch, const bool to_polars) {
	if (!result) {
		throw InvalidInputException("There is no query result");
	}

	return RunWithArrowSchema<duckdb::pyarrow::Table>(
	    [&](const ArrowSchema &schema) -> duckdb::pyarrow::Table {
		    if (result->type == QueryResultType::MATERIALIZED_RESULT || result->type == QueryResultType::ARROW_RESULT) {
			    return MaterializedResultToArrowTable(schema, rows_per_batch);
		    }
		    if (result->type != QueryResultType::STREAM_RESULT) {
			    throw InternalException("FetchArrowTable called with unsupported query result: %d", result->type);
		    }
		    auto pyarrow_schema = pyarrow::ToPyArrowSchema(schema);
		    nb::list batches;
		    QueryResultChunkScanState scan_state(*result);
		    while (true) {
			    ArrowArray data;
			    idx_t count;
			    {
				    D_ASSERT(duckdb::PyUtil::GilCheck());
				    nb::gil_scoped_release release;
				    count = ArrowUtil::FetchChunk(scan_state, result->client_properties, rows_per_batch, &data,
				                                  ArrowTypeExtensionData::GetExtensionTypes(
				                                      *result->client_properties.client_context, result->types));
			    }
			    if (count == 0) {
				    break;
			    }
			    TransformDuckToArrowChunk(pyarrow_schema, data, batches);
		    }
		    return pyarrow::ToArrowTable(std::move(batches), pyarrow_schema);
	    },
	    to_polars);
}

ArrowArrayStream DuckDBPyResult::FetchArrowArrayStream(idx_t rows_per_batch) {
	if (!result) {
		throw InvalidInputException("There is no query result");
	}
	if (result->type != QueryResultType::STREAM_RESULT) {
		throw InternalException("FetchArrowArrayStream called with unsupported query result: %d", result->type);
	}
	// The wrapper is owned by the ArrowArrayStream's private_data (released with the stream).
	const auto result_stream = new ResultArrowArrayStreamWrapper(std::move(result), rows_per_batch);
	return result_stream->stream;
}

duckdb::pyarrow::RecordBatchReader DuckDBPyResult::FetchRecordBatchReader(idx_t rows_per_batch) {
	if (!result) {
		throw InvalidInputException("There is no query result");
	}

	if (result->type == QueryResultType::MATERIALIZED_RESULT || result->type == QueryResultType::ARROW_RESULT) {
		constexpr bool dedup_column_names = false;
		return RunWithArrowSchema<duckdb::pyarrow::RecordBatchReader>(
		    [&](const ArrowSchema &schema) -> duckdb::pyarrow::RecordBatchReader {
			    const auto table = MaterializedResultToArrowTable(schema, rows_per_batch);
			    return nb::cast<duckdb::pyarrow::RecordBatchReader>(
			        table.attr("to_reader")(nb::arg("max_chunksize") = rows_per_batch));
		    },
		    dedup_column_names);
	}
	if (result->type != QueryResultType::STREAM_RESULT) {
		throw InternalException("FetchRecordBatchReader called with unsupported query result: %d", result->type);
	}
	nb::gil_scoped_acquire acquire;
	auto pyarrow_lib_module = nb::module_::import_("pyarrow").attr("lib");
	auto record_batch_reader_func = pyarrow_lib_module.attr("RecordBatchReader").attr("_import_from_c");
	auto stream = FetchArrowArrayStream(rows_per_batch);
	nb::object record_batch_reader = record_batch_reader_func((uint64_t)&stream); // NOLINT
	return nb::cast<duckdb::pyarrow::RecordBatchReader>(record_batch_reader);
}

static void ArrowArrayStreamPyCapsuleDestructor(void *data) noexcept {
	// nanobind capsule cleanup receives the raw pointer (via PyCapsule_GetPointer using the capsule's name)
	if (!data) {
		return;
	}
	auto stream = reinterpret_cast<ArrowArrayStream *>(data);
	if (stream->release) {
		stream->release(stream);
	}
	delete stream;
}

nb::object DuckDBPyResult::FetchArrowCapsule(const idx_t rows_per_batch) {
	if (!result) {
		throw InvalidInputException("There is no query result");
	}

	constexpr bool dedup_column_names = false;
	if (result->type == QueryResultType::MATERIALIZED_RESULT || result->type == QueryResultType::ARROW_RESULT) {
		return RunWithArrowSchema<nb::object>(
		    [&](const ArrowSchema &schema) -> nb::object {
			    const auto table = MaterializedResultToArrowTable(schema, rows_per_batch);
			    return table.attr("__arrow_c_stream__")();
		    },
		    dedup_column_names);
	}
	if (result->type != QueryResultType::STREAM_RESULT) {
		throw InternalException("FetchArrowCapsule called with unsupported query result: %d", result->type);
	}
	auto inner_stream = FetchArrowArrayStream(rows_per_batch);
	auto stream = new ArrowArrayStream();
	*stream = inner_stream;
	return nb::capsule(stream, "arrow_array_stream", ArrowArrayStreamPyCapsuleDestructor);
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
	result = nullptr;
}

bool DuckDBPyResult::IsClosed() const {
	return result_closed;
}

} // namespace duckdb

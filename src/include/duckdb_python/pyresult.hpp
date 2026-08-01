//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pyresult.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/numpy/numpy_result_conversion.hpp"
#include "duckdb.hpp"
#include "duckdb/main/chunk_scan_state.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/dataframe.hpp"

namespace duckdb {

struct DuckDBPyResult {
public:
	explicit DuckDBPyResult(unique_ptr<QueryResult> result);
	~DuckDBPyResult();

public:
	Optional<nb::tuple> Fetchone();

	nb::list Fetchmany(idx_t size);

	nb::list Fetchall();

	nb::dict FetchNumpy();

	nb::dict FetchNumpyInternal(bool stream = false, idx_t vectors_per_chunk = 1,
	                            std::unique_ptr<NumpyResultConversion> conversion = nullptr);

	PandasDataFrame FetchDF(bool date_as_object);

	PandasDataFrame FetchDFChunk(const idx_t vectors_per_chunk = 1, bool date_as_object = false);

	nb::dict FetchPyTorch();

	nb::dict FetchTF();

	duckdb::pyarrow::Table FetchArrowTable(idx_t rows_per_batch, bool to_polars);
	duckdb::pyarrow::RecordBatchReader FetchRecordBatchReader(idx_t rows_per_batch = 1000000);
	nb::object FetchArrowCapsule(idx_t rows_per_batch = 1000000);

	static nb::list GetDescription(const vector<string> &names, const vector<LogicalType> &types);

	void Close();

	bool IsClosed() const;

	unique_ptr<DataChunk> FetchChunk();

	const vector<string> &GetNames();
	const vector<LogicalType> &GetTypes();

	ClientProperties GetClientProperties();

	//! Number of rows changed by the last CHANGED_ROWS-returning statement (INSERT/UPDATE/DELETE/...).
	//! Returns -1 when not applicable/unknown, as permitted by the DB-API 2.0 spec for 'rowcount'.
	int64_t GetRowcount() const {
		return row_changes;
	}

private:
	void FillNumpy(nb::dict &res, idx_t col_idx, NumpyResultConversion &conversion, const char *name);

	PandasDataFrame FrameFromNumpy(bool date_as_object, const nb::handle &o);

	void ConvertDateTimeTypes(PandasDataFrame &df, bool date_as_object) const;
	unique_ptr<DataChunk> FetchNext(QueryResult &result);
	unique_ptr<DataChunk> FetchNextRaw(QueryResult &result);
	std::unique_ptr<NumpyResultConversion> InitializeNumpyConversion(bool pandas = false);

	//! Re-feed an already-MATERIALIZED result (a ColumnDataCollection, e.g. from
	//! rel.execute()) back through the engine on the user's own context. The eager
	//! variant installs a PhysicalArrowCollector to produce an ArrowQueryResult
	//! (parallel); the stream variant produces a lazy StreamQueryResult that co-owns
	//! the context (so it survives `del conn`). Never call these on a StreamQueryResult:
	//! a lazy result already has a live context and is converted/wrapped directly.
	void PromoteMaterializedToArrow(idx_t batch_size);

	template <typename T>
	T RunWithArrowSchema(const std::function<T(const ArrowSchema &)> &fun, bool dedup_col_names);
	duckdb::pyarrow::Table MaterializedResultToArrowTable(const ArrowSchema &arrow_schema, idx_t rows_per_batch);
	ArrowArrayStream FetchArrowArrayStream(idx_t rows_per_batch);

	//! Computes the CHANGED_ROWS value (if any) up front, before any Fetch call has had a chance to
	//! consume it, so that later fetch calls do not affect what GetRowcount() reports. Materializes a
	//! streaming result if necessary; this is cheap since CHANGED_ROWS results are always exactly one
	//! already-computed row.
	int64_t ComputeRowChanges();

private:
	idx_t chunk_offset = 0;

	unique_ptr<QueryResult> result;
	unique_ptr<DataChunk> current_chunk;
	// Holds the categories of Categorical/ENUM types
	unordered_map<idx_t, nb::list> categories;
	// Holds the categorical type of Categorical/ENUM types
	unordered_map<idx_t, nb::object> categories_type;
	bool result_closed = false;
	//! Cached by ComputeRowChanges() at construction time - see GetRowcount().
	int64_t row_changes = -1;
};

} // namespace duckdb

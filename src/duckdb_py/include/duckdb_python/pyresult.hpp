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
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/pybind11/dataframe.hpp"

namespace duckdb {

struct DuckDBPyResult {
public:
	explicit DuckDBPyResult(unique_ptr<QueryResult> result);
	~DuckDBPyResult();

public:
	Optional<py::tuple> Fetchone();

	py::list Fetchmany(idx_t size);

	py::list Fetchall();

	py::dict FetchNumpy();

	py::dict FetchNumpyInternal(bool stream = false, idx_t vectors_per_chunk = 1,
	                            std::unique_ptr<NumpyResultConversion> conversion = nullptr);

	PandasDataFrame FetchDF(bool date_as_object);

	PandasDataFrame FetchDFChunk(const idx_t vectors_per_chunk = 1, bool date_as_object = false);

	py::dict FetchPyTorch();

	py::dict FetchTF();

	duckdb::pyarrow::Table FetchArrowTable(idx_t rows_per_batch, bool to_polars);
	duckdb::pyarrow::RecordBatchReader FetchRecordBatchReader(idx_t rows_per_batch = 1000000);
	py::object FetchArrowCapsule(idx_t rows_per_batch = 1000000);

	static py::list GetDescription(const vector<string> &names, const vector<LogicalType> &types);

	void Close();

	bool IsClosed() const;

	unique_ptr<DataChunk> FetchChunk();

	const vector<string> &GetNames();
	const vector<LogicalType> &GetTypes();

	ClientProperties GetClientProperties();

private:
	void FillNumpy(py::dict &res, idx_t col_idx, NumpyResultConversion &conversion, const char *name);

	PandasDataFrame FrameFromNumpy(bool date_as_object, const py::handle &o);

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

private:
	idx_t chunk_offset = 0;

	unique_ptr<QueryResult> result;
	unique_ptr<DataChunk> current_chunk;
	// Holds the categories of Categorical/ENUM types
	unordered_map<idx_t, py::list> categories;
	// Holds the categorical type of Categorical/ENUM types
	unordered_map<idx_t, py::object> categories_type;
	bool result_closed = false;
};

} // namespace duckdb

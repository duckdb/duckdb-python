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
#include "duckdb_python/engine/connection.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/dataframe.hpp"

namespace duckdb {

struct DuckDBPyResult {
public:
	static constexpr idx_t DEFAULT_ARROW_BATCH_SIZE = 131072;

public:
	explicit DuckDBPyResult(shared_ptr<engine::Result> result);
	~DuckDBPyResult();

public:
	const shared_ptr<engine::Result> &EngineResult() const {
		return result;
	}
	Optional<nb::tuple> Fetchone();

	nb::list Fetchmany(idx_t size);

	nb::list Fetchall();

	nb::dict FetchNumpy();

	nb::dict FetchNumpyInternal(bool chunked = false, idx_t vectors_per_chunk = 1,
	                            std::unique_ptr<NumpyResultConversion> conversion = nullptr);

	PandasDataFrame FetchDF(bool date_as_object);

	PandasDataFrame FetchDFChunk(const idx_t vectors_per_chunk = 1, bool date_as_object = false);

	nb::dict FetchPyTorch();

	nb::dict FetchTF();

	duckdb::pyarrow::Table FetchArrowTable(idx_t rows_per_batch, bool to_polars);
	duckdb::pyarrow::RecordBatchReader FetchRecordBatchReader(idx_t rows_per_batch = DEFAULT_ARROW_BATCH_SIZE);
	nb::object FetchArrowCapsule(idx_t rows_per_batch = DEFAULT_ARROW_BATCH_SIZE);
	static void CheckBatchSize(idx_t rows_per_batch);

	static nb::list GetDescription(const vector<string> &names, const vector<LogicalType> &types);

	void Close();
	//! Runs the statement to its end without reading its rows, then closes the result
	void Complete();
	//! Whether the statement may change a database, nextval included, and still has work left to run
	bool HasWritesToRun() const;

	unique_ptr<DataChunk> FetchChunk();
	//! The rows a completed DML statement changed, -1 otherwise
	int64_t Rowcount();
	//! The accumulated count of an executemany, which spans several results
	void OverrideRowcount(int64_t rowcount);

	vector<string> GetNames();
	const vector<LogicalType> &GetTypes() const;

	const ClientProperties &GetClientProperties() const;

private:
	void FillNumpy(nb::dict &res, idx_t col_idx, NumpyResultConversion &conversion, const char *name);

	PandasDataFrame FrameFromNumpy(bool date_as_object, const nb::handle &o);

	void ConvertDateTimeTypes(PandasDataFrame &df, bool date_as_object) const;
	//! The names the Python layer reports, see the definition for why this is not always the result's own.
	const vector<Identifier> &ResultNames() const;
	bool Empty() const {
		return !result;
	}
	//! Flat vectors, for the row fetch
	unique_ptr<DataChunk> FetchNext();
	unique_ptr<DataChunk> FetchNextRaw();
	unique_ptr<DataChunk> TakeBufferedRows();
	std::unique_ptr<NumpyResultConversion> InitializeNumpyConversion(bool pandas = false);

	template <typename T>
	T RunWithArrowSchema(const std::function<T(const ArrowSchema &)> &fun, bool dedup_col_names);
	//! The stream's private data owns the result, and its callbacks run without the GIL.
	ArrowArrayStream FetchArrowArrayStream(idx_t rows_per_batch);

private:
	shared_ptr<engine::Result> result;
	int64_t rowcount_override = -1;
	idx_t chunk_offset = 0;
	unique_ptr<DataChunk> current_chunk;
	// Holds the categories of Categorical/ENUM types
	unordered_map<idx_t, nb::list> categories;
	// Holds the categorical type of Categorical/ENUM types
	unordered_map<idx_t, nb::object> categories_type;
};

} // namespace duckdb

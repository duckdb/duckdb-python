//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pyresult.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/atomic.hpp"
#include "duckdb_python/numpy/numpy_result_conversion.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
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
	//! names_override is the result's own names where they differ from the submitted statement's, which they do
	//! for a statement the client wrote over rows it already held
	explicit DuckDBPyResult(shared_ptr<engine::Result> result, vector<Identifier> names_override = {});
	~DuckDBPyResult();

public:
	const shared_ptr<engine::Result> &EngineResult() const {
		return result;
	}
	//! Pins the result for one call of the relational surface and refuses a second call that overlaps it, as the
	//! connection refuses a second call of its own. The DB-API surface is serialised by the connection lock instead.
	class Use {
	public:
		explicit Use(DuckDBPyResult &result);
		~Use();

		Use(const Use &) = delete;
		Use &operator=(const Use &) = delete;

		engine::Result &operator*() const {
			return *pinned;
		}
		engine::Result *operator->() const {
			return pinned.get();
		}

	private:
		DuckDBPyResult &result;
		//! Held for the call, so a close from another call cannot free the result under this one
		shared_ptr<engine::Result> pinned;
	};
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
	//! Every row the result still holds, the rows buffered from a partial fetch first; the result is empty
	//! afterwards. Call with the GIL held.
	unique_ptr<ColumnDataCollection> TakeRows(const engine::InterruptCheck &check);
	//! The names the Python layer reports, which are names_override when that is set.
	const vector<Identifier> &ResultNames() const;
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
	//! Set while a call of the relational surface runs on this result
	atomic<bool> in_use {false};
	int64_t rowcount_override = -1;
	idx_t chunk_offset = 0;
	//! Set only for a result the client submitted over rows of an earlier one, whose names the scan deduplicated
	//! and whose originals the caller carries here. Empty means the result's own names.
	vector<Identifier> names_override;
	unique_ptr<DataChunk> current_chunk;
	// Holds the categories of Categorical/ENUM types
	unordered_map<idx_t, nb::list> categories;
	// Holds the categorical type of Categorical/ENUM types
	unordered_map<idx_t, nb::object> categories_type;
};

} // namespace duckdb

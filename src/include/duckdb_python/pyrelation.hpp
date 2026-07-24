//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pyrelation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb.hpp"
#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/nb/conversions/render_mode_enum.hpp"
#include "duckdb_python/dataframe.hpp"
#include "duckdb_python/python_objects.hpp"

namespace duckdb {

struct DuckDBPyRelation {
public:
	explicit DuckDBPyRelation(shared_ptr<Relation> rel);
	explicit DuckDBPyRelation(std::shared_ptr<DuckDBPyResult> result);
	~DuckDBPyRelation();

public:
	static void Initialize(nb::handle &m);

	nb::list Description();

	void Close();

	std::unique_ptr<DuckDBPyRelation> GetAttribute(const string &name);

	nb::str GetAlias();

	static std::unique_ptr<DuckDBPyRelation> EmptyResult(const shared_ptr<ClientContext> &context,
	                                                     const vector<LogicalType> &types, vector<string> names);

	std::unique_ptr<DuckDBPyRelation> SetAlias(const string &expr);

	std::unique_ptr<DuckDBPyRelation> ProjectFromExpression(const string &expr);
	std::unique_ptr<DuckDBPyRelation> ProjectFromTypes(const nb::object &types);
	std::unique_ptr<DuckDBPyRelation> Project(const nb::args &args, const string &groups = "");
	std::unique_ptr<DuckDBPyRelation> Filter(const nb::object &expr);
	std::unique_ptr<DuckDBPyRelation> FilterFromExpression(const string &expr);
	std::unique_ptr<DuckDBPyRelation> Limit(int64_t n, int64_t offset = 0);
	std::unique_ptr<DuckDBPyRelation> Order(const string &expr);
	std::unique_ptr<DuckDBPyRelation> Sort(const nb::args &args);

	std::unique_ptr<DuckDBPyRelation> Aggregate(const nb::object &expr, const string &groups = "");

	std::unique_ptr<DuckDBPyRelation> GenericAggregator(const string &function_name, const string &aggregated_columns,
	                                                    const string &groups = "",
	                                                    const string &function_parameter = "",
	                                                    const string &projected_columns = "");

	/* General aggregate functions */
	std::unique_ptr<DuckDBPyRelation> AnyValue(const string &column, const string &groups = "",
	                                           const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> ArgMax(const string &arg_column, const string &value_column,
	                                         const string &groups = "", const string &window_spec = "",
	                                         const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> ArgMin(const string &arg_column, const string &value_column,
	                                         const string &groups = "", const string &window_spec = "",
	                                         const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> Avg(const string &column, const string &groups = "",
	                                      const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> BitAnd(const string &column, const string &groups = "",
	                                         const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> BitOr(const string &column, const string &groups = "",
	                                        const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> BitXor(const string &column, const string &groups = "",
	                                         const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> BitStringAgg(const string &column, const Optional<nb::object> &min,
	                                               const Optional<nb::object> &max, const string &groups = "",
	                                               const string &window_spec = "",
	                                               const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> BoolAnd(const string &column, const string &groups = "",
	                                          const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> BoolOr(const string &column, const string &groups = "",
	                                         const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> ValueCounts(const string &column, const string &groups = "");
	std::unique_ptr<DuckDBPyRelation> Count(const string &column, const string &groups = "",
	                                        const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> FAvg(const string &column, const string &groups = "",
	                                       const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> First(const string &column, const string &groups = "",
	                                        const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> FSum(const string &column, const string &groups = "",
	                                       const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> GeoMean(const string &column, const string &groups = "",
	                                          const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> Histogram(const string &column, const string &groups = "",
	                                            const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> Last(const string &column, const string &groups = "",
	                                       const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> List(const string &column, const string &groups = "",
	                                       const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> Max(const string &column, const string &groups = "",
	                                      const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> Min(const string &column, const string &groups = "",
	                                      const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> Product(const string &column, const string &groups = "",
	                                          const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> StringAgg(const string &column, const string &sep = ",",
	                                            const string &groups = "", const string &window_spec = "",
	                                            const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> Sum(const string &column, const string &groups = "",
	                                      const string &window_spec = "", const string &projected_columns = "");
	/* TODO: Approximate aggregate functions */
	/* TODO: Statistical aggregate functions */
	std::unique_ptr<DuckDBPyRelation> Median(const string &column, const string &groups = "",
	                                         const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> Mode(const string &column, const string &groups = "",
	                                       const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> QuantileCont(const string &column, const nb::object &q, const string &groups = "",
	                                               const string &window_spec = "",
	                                               const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> QuantileDisc(const string &column, const nb::object &q, const string &groups = "",
	                                               const string &window_spec = "",
	                                               const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> StdPop(const string &column, const string &groups = "",
	                                         const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> StdSamp(const string &column, const string &groups = "",
	                                          const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> VarPop(const string &column, const string &groups = "",
	                                         const string &window_spec = "", const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> VarSamp(const string &column, const string &groups = "",
	                                          const string &window_spec = "", const string &projected_columns = "");

	std::unique_ptr<DuckDBPyRelation> Describe();

	string ToSQL();

	duckdb::pyarrow::RecordBatchReader FetchRecordBatchReader(idx_t rows_per_batch);

	idx_t Length();

	nb::tuple Shape();

	std::unique_ptr<DuckDBPyRelation> Unique(const string &aggr_columns);

	std::unique_ptr<DuckDBPyRelation> GenericWindowFunction(const string &function_name,
	                                                        const string &function_parameters,
	                                                        const string &aggr_columns, const string &window_spec,
	                                                        const bool &ignore_nulls, const string &projected_columns);

	/* General purpose window functions */
	std::unique_ptr<DuckDBPyRelation> RowNumber(const string &window_spec, const string &projected_columns);
	std::unique_ptr<DuckDBPyRelation> Rank(const string &window_spec, const string &projected_columns);
	std::unique_ptr<DuckDBPyRelation> DenseRank(const string &window_spec, const string &projected_columns);
	std::unique_ptr<DuckDBPyRelation> PercentRank(const string &window_spec, const string &projected_columns);
	std::unique_ptr<DuckDBPyRelation> CumeDist(const string &window_spec, const string &projected_columns);
	std::unique_ptr<DuckDBPyRelation> FirstValue(const string &column, const string &window_spec = "",
	                                             const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> NTile(const string &window_spec, const int &num_buckets,
	                                        const string &projected_columns);
	std::unique_ptr<DuckDBPyRelation> Lag(const string &column, const string &window_spec, const int &offset,
	                                      const string &default_value, const bool &ignore_nulls,
	                                      const string &projected_columns);
	std::unique_ptr<DuckDBPyRelation> LastValue(const string &column, const string &window_spec = "",
	                                            const string &projected_columns = "");
	std::unique_ptr<DuckDBPyRelation> Lead(const string &column, const string &window_spec, const int &offset,
	                                       const string &default_value, const bool &ignore_nulls,
	                                       const string &projected_columns);

	std::unique_ptr<DuckDBPyRelation> NthValue(const string &column, const string &window_spec, const int &offset,
	                                           const bool &ignore_nulls, const string &projected_columns);

	std::unique_ptr<DuckDBPyRelation> Distinct();

	PandasDataFrame FetchDF(bool date_as_object);

	Optional<nb::tuple> FetchOne();

	nb::list FetchAll();

	nb::list FetchMany(idx_t size);

	nb::dict FetchNumpy();

	nb::dict FetchPyTorch();

	nb::dict FetchTF();

	nb::dict FetchNumpyInternal(bool stream = false, idx_t vectors_per_chunk = 1);

	PandasDataFrame FetchDFChunk(const idx_t vectors_per_chunk = 1, bool date_as_object = false);

	duckdb::pyarrow::Table ToArrowTable(idx_t batch_size);

	duckdb::pyarrow::Table ToArrowTableInternal(idx_t batch_size, bool to_polars);

	PolarsDataFrame ToPolars(idx_t batch_size, bool lazy);

	nb::object ToArrowCapsule(const nb::object &requested_schema = nb::none());

	duckdb::pyarrow::RecordBatchReader ToRecordBatch(idx_t batch_size);

	std::unique_ptr<DuckDBPyRelation> Union(DuckDBPyRelation *other);

	std::unique_ptr<DuckDBPyRelation> Except(DuckDBPyRelation *other);

	std::unique_ptr<DuckDBPyRelation> Intersect(DuckDBPyRelation *other);

	std::unique_ptr<DuckDBPyRelation> Map(nb::callable fun, Optional<nb::object> schema);

	std::unique_ptr<DuckDBPyRelation> Join(DuckDBPyRelation *other, const nb::object &condition, const string &type);
	std::unique_ptr<DuckDBPyRelation> Cross(DuckDBPyRelation *other);

	void ToParquet(const nb::object &file_name, const nb::object &compression = nb::none(),
	               const nb::object &field_ids = nb::none(), const nb::object &row_group_size_bytes = nb::none(),
	               const nb::object &row_group_size = nb::none(), const nb::object &overwrite = nb::none(),
	               const nb::object &per_thread_output = nb::none(), const nb::object &use_tmp_file = nb::none(),
	               const nb::object &partition_by = nb::none(), const nb::object &write_partition_columns = nb::none(),
	               const nb::object &append = nb::none(), const nb::object &filename_pattern = nb::none(),
	               const nb::object &file_size_bytes = nb::none());

	void ToCSV(const nb::object &file_name, const nb::object &sep = nb::none(), const nb::object &na_rep = nb::none(),
	           const nb::object &header = nb::none(), const nb::object &quotechar = nb::none(),
	           const nb::object &escapechar = nb::none(), const nb::object &date_format = nb::none(),
	           const nb::object &timestamp_format = nb::none(), const nb::object &quoting = nb::none(),
	           const nb::object &encoding = nb::none(), const nb::object &compression = nb::none(),
	           const nb::object &overwrite = nb::none(), const nb::object &per_thread_output = nb::none(),
	           const nb::object &use_tmp_file = nb::none(), const nb::object &partition_by = nb::none(),
	           const nb::object &write_partition_columns = nb::none());

	// should this return a rel with the new view?
	std::unique_ptr<DuckDBPyRelation> CreateView(const string &view_name, bool replace = true);

	std::unique_ptr<DuckDBPyRelation> Query(const string &view_name, const string &sql_query);

	// Update the internal result of the relation
	DuckDBPyRelation &Execute();

	void InsertInto(const string &table);

	void Insert(const nb::object &params = nb::list()) const;
	void Update(const nb::object &set, const nb::object &where = nb::none());

	void Create(const string &table);

	nb::str Type();
	nb::list Columns();
	nb::list ColumnTypes();

	string ToString();
	void Print(const Optional<nb::int_> &max_width, const Optional<nb::int_> &max_rows,
	           const Optional<nb::int_> &max_col_width, const Optional<nb::str> &null_value,
	           const nb::object &render_mode);

	string Explain(ExplainType type, const string &format = "");

	static bool IsRelation(const nb::object &object);

	bool CanBeRegisteredBy(Connection &con);
	bool CanBeRegisteredBy(ClientContext &context);
	bool CanBeRegisteredBy(shared_ptr<ClientContext> &context);

	Relation &GetRel();

	bool ContainsColumnByName(const string &name) const;

	void SetConnectionOwner(nb::object owner);
	std::unique_ptr<DuckDBPyRelation> DeriveRelation(shared_ptr<Relation> new_rel);
	std::unique_ptr<DuckDBPyRelation> DeriveRelation(std::shared_ptr<DuckDBPyResult> result);

private:
	string ToStringInternal(const BoxRendererConfig &config, bool invalidate_cache = false);
	string GenerateExpressionList(const string &function_name, const string &aggregated_columns,
	                              const string &groups = "", const string &function_parameter = "",
	                              bool ignore_nulls = false, const string &projected_columns = "",
	                              const string &window_spec = "");
	string GenerateExpressionList(const string &function_name, vector<string> aggregated_columns,
	                              const string &groups = "", const string &function_parameter = "",
	                              bool ignore_nulls = false, const string &projected_columns = "",
	                              const string &window_spec = "");
	std::unique_ptr<DuckDBPyRelation> ApplyAggOrWin(const string &function_name, const string &agg_columns,
	                                                const string &function_parameters = "", const string &groups = "",
	                                                const string &window_spec = "",
	                                                const string &projected_columns = "", bool ignore_nulls = false);

	void AssertResult() const;
	void AssertResultOpen() const;
	void AssertRelation() const;
	void ExecuteOrThrow(bool stream_result = false);
	unique_ptr<QueryResult> ExecuteInternal(bool stream_result = false);

private:
	//! Prevents GC of the parent DuckDBPyConnection.
	//! Declared first so it is destroyed last (reverse declaration order).
	nb::object connection_owner;
	//! Whether the relation has been executed at least once
	bool executed;
	shared_ptr<Relation> rel;
	vector<LogicalType> types;
	vector<string> names;
	std::shared_ptr<DuckDBPyResult> result;
	std::string rendered_result;
};

} // namespace duckdb

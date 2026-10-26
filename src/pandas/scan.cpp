#include "duckdb_python/pandas/pandas_scan.hpp"
#include "duckdb_python/pandas/pandas_bind.hpp"
#include "duckdb_python/engine/connection.hpp"
#include "duckdb_python/numpy/array_wrapper.hpp"
#include "utf8proc_wrapper.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb_python/numpy/numpy_scan.hpp"
#include "duckdb_python/numpy/numpy_bind.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb_python/pandas/column/pandas_numpy_column.hpp"
#include "duckdb_python/pandas/column/pandas_arrow_column.hpp"
#include "duckdb_python/arrow/arrow_importer.hpp"
#include "duckdb_python/pyutil.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

#include "duckdb/common/atomic.hpp"

namespace duckdb {

struct PandasScanFunctionData : public TableFunctionData {
	PandasScanFunctionData(shared_ptr<PandasScanInfo> info, idx_t row_count,
	                       vector<PandasColumnBindData> pandas_bind_data, vector<LogicalType> sql_types)
	    : info(std::move(info)), row_count(row_count), lines_read(0), pandas_bind_data(std::move(pandas_bind_data)),
	      sql_types(std::move(sql_types)) {
	}
	shared_ptr<PandasScanInfo> info;
	idx_t row_count;
	atomic<idx_t> lines_read;
	vector<PandasColumnBindData> pandas_bind_data;
	vector<LogicalType> sql_types;

	~PandasScanFunctionData() override {
		try {
			if (nb::detail::cleanup_guard guard {}) {
				pandas_bind_data.clear();
			} else {
				// Leaked on purpose: the Python references can no longer be dropped, and a static owner
				// keeps a leak checker from reporting them
				static auto *parked = new vector<vector<PandasColumnBindData>>();
				parked->push_back(std::move(pandas_bind_data));
			}
		} catch (...) { // NOLINT
		}
	}
};

struct PandasScanLocalState : public LocalTableFunctionState {
	PandasScanLocalState(idx_t start, idx_t end) : start(start), end(end), batch_index(0) {
	}

	idx_t start;
	idx_t end;
	idx_t batch_index;
	vector<column_t> column_ids;
};

struct PandasScanGlobalState : public GlobalTableFunctionState {
	explicit PandasScanGlobalState(idx_t max_threads) : position(0), batch_index(0), max_threads(max_threads) {
	}

	std::mutex lock;
	idx_t position;
	idx_t batch_index;
	idx_t max_threads;

	idx_t MaxThreads() const override {
		return max_threads;
	}
};

PandasScanFunction::PandasScanFunction()
    : TableFunction("pandas_scan", {}, PandasScanFunc, PandasScanBind, PandasScanInitGlobal, PandasScanInitLocal) {
	get_partition_data = PandasScanGetPartitionData;
	cardinality = PandasScanCardinality;
	table_scan_progress = PandasProgress;
	serialize = PandasSerialize;
	deserialize = PandasDeserialize;
	projection_pushdown = true;
	get_virtual_columns = PandasScanGetVirtualColumns;
}

virtual_column_map_t PandasScanFunction::PandasScanGetVirtualColumns(ClientContext &context,
                                                                     optional_ptr<FunctionData> bind_data) {
	virtual_column_map_t result;
	result.insert(make_pair(COLUMN_IDENTIFIER_EMPTY, TableColumn("", LogicalType::BOOLEAN)));
	return result;
}

OperatorPartitionData PandasScanFunction::PandasScanGetPartitionData(ClientContext &context,
                                                                     TableFunctionGetPartitionInput &input) {
	if (input.partition_info.RequiresPartitionColumns()) {
		throw InternalException("PandasScan::GetPartitionData: partition columns not supported");
	}
	auto &data = input.local_state->Cast<PandasScanLocalState>();
	return OperatorPartitionData(data.batch_index);
}

unique_ptr<FunctionData> PandasScanFunction::PandasScanBind(ClientContext &context, TableFunctionBindInput &input,
                                                            vector<LogicalType> &return_types,
                                                            vector<Identifier> &names) {
	if (!input.ref.bind_info) {
		throw BinderException("pandas_scan requires a dataframe bind input");
	}
	DynamicCastCheck<PandasScanInfo>(input.ref.bind_info.get());
	auto info = shared_ptr_cast<TableFunctionInfo, PandasScanInfo>(input.ref.bind_info);
	nb::gil_scoped_acquire acquire;
	engine::CallbackScope callback(context);
	nb::handle df(info->df.obj);

	vector<PandasColumnBindData> pandas_bind_data;

	idx_t row_count;
	if (nb::isinstance<nb::dict>(df)) {
		row_count = NumpyBind::Bind(context, df, pandas_bind_data, return_types, names);
	} else {
		row_count = Pandas::Bind(context, df, pandas_bind_data, return_types, names);
	}
	return make_uniq<PandasScanFunctionData>(std::move(info), row_count, std::move(pandas_bind_data), return_types);
}

unique_ptr<GlobalTableFunctionState> PandasScanFunction::PandasScanInitGlobal(ClientContext &context,
                                                                              TableFunctionInitInput &input) {
	DUCKDB_PY_ASSERT_GIL_RELEASED();
	return make_uniq<PandasScanGlobalState>(PandasScanMaxThreads(context, input.bind_data.get()));
}

unique_ptr<LocalTableFunctionState> PandasScanFunction::PandasScanInitLocal(ExecutionContext &context,
                                                                            TableFunctionInitInput &input,
                                                                            GlobalTableFunctionState *gstate) {
	auto result = make_uniq<PandasScanLocalState>(0, 0);
	result->column_ids = input.column_ids;
	PandasScanParallelStateNext(context.client, input.bind_data.get(), result.get(), gstate);
	return std::move(result);
}

idx_t PandasScanFunction::PandasScanMaxThreads(ClientContext &context, const FunctionData *bind_data_p) {
	if (ClientConfig::GetConfig(context).verify_parallelism) {
		return context.db->NumberOfThreads();
	}
	auto &bind_data = bind_data_p->Cast<PandasScanFunctionData>();
	return bind_data.row_count / PANDAS_PARTITION_COUNT + 1;
}

bool PandasScanFunction::PandasScanParallelStateNext(ClientContext &context, const FunctionData *bind_data_p,
                                                     LocalTableFunctionState *lstate,
                                                     GlobalTableFunctionState *gstate) {
	auto &bind_data = bind_data_p->Cast<PandasScanFunctionData>();
	auto &parallel_state = gstate->Cast<PandasScanGlobalState>();
	auto &state = lstate->Cast<PandasScanLocalState>();

	lock_guard<mutex> parallel_lock(parallel_state.lock);
	if (parallel_state.position >= bind_data.row_count) {
		return false;
	}
	state.start = parallel_state.position;
	parallel_state.position += PANDAS_PARTITION_COUNT;
	if (parallel_state.position > bind_data.row_count) {
		parallel_state.position = bind_data.row_count;
	}
	state.end = parallel_state.position;
	state.batch_index = parallel_state.batch_index++;
	return true;
}

double PandasScanFunction::PandasProgress(ClientContext &context, const FunctionData *bind_data_p,
                                          const GlobalTableFunctionState *gstate) {
	auto &bind_data = bind_data_p->Cast<PandasScanFunctionData>();
	if (bind_data.row_count == 0) {
		return 100;
	}
	auto percentage = (bind_data.lines_read * 100.0) / bind_data.row_count;
	return percentage;
}

static void ScanArrowColumn(ClientContext &context, PandasArrowColumn &column, idx_t count, idx_t offset, Vector &out) {
	if (count == 0) {
		return;
	}
	auto &starts = column.starts;
	auto chunk_idx = NumericCast<idx_t>(std::upper_bound(starts.begin(), starts.end(), offset) - starts.begin()) - 1;
	auto position = offset - starts[chunk_idx];
	if (position + count <= starts[chunk_idx + 1] - starts[chunk_idx]) {
		ArrowArrayScanState state(context);
		ReadArrowColumn(out, column.chunks[chunk_idx].arrow_array, *column.arrow_type, position, count, state);
		return;
	}
	// Only the entries are copied: the strings stay in the chunks, which the bind data keeps alive.
	Vector piece(out.GetType());
	auto target = FlatVector::GetDataMutable<string_t>(out);
	auto &target_validity = FlatVector::ValidityMutable(out);
	for (idx_t written = 0; written < count; chunk_idx++, position = 0) {
		auto rows = MinValue(count - written, starts[chunk_idx + 1] - starts[chunk_idx] - position);
		FlatVector::ValidityMutable(piece).Reset();
		ArrowArrayScanState state(context);
		ReadArrowColumn(piece, column.chunks[chunk_idx].arrow_array, *column.arrow_type, position, rows, state);
		auto source = FlatVector::GetData<string_t>(piece);
		auto &source_validity = FlatVector::Validity(piece);
		for (idx_t row = 0; row < rows; row++) {
			if (source_validity.RowIsValid(row)) {
				target[written + row] = source[row];
			} else {
				target_validity.SetInvalid(written + row);
			}
		}
		written += rows;
	}
	FlatVector::SetSize(out, count);
}

void PandasScanFunction::PandasBackendScanSwitch(ClientContext &context, PandasColumnBindData &bind_data, idx_t count,
                                                 idx_t offset, Vector &out) {
	auto backend = bind_data.pandas_col->Backend();
	switch (backend) {
	case PandasColumnBackend::NUMPY: {
		NumpyScan::Scan(context, bind_data, count, offset, out);
		break;
	}
	case PandasColumnBackend::ARROW: {
		ScanArrowColumn(context, static_cast<PandasArrowColumn &>(*bind_data.pandas_col), count, offset, out);
		break;
	}
	default: {
		throw NotImplementedException("Type not implemented for PandasColumnBackend");
	}
	}
}

//! The main pandas scan function: note that this can be called in parallel without the GIL
//! hence this needs to be GIL-safe, i.e. no methods that create Python objects are allowed
void PandasScanFunction::PandasScanFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.bind_data->CastNoConst<PandasScanFunctionData>();
	auto &state = data_p.local_state->Cast<PandasScanLocalState>();

	if (state.start >= state.end) {
		if (!PandasScanParallelStateNext(context, data_p.bind_data.get(), data_p.local_state.get(),
		                                 data_p.global_state.get())) {
			return;
		}
	}
	idx_t this_count = std::min((idx_t)STANDARD_VECTOR_SIZE, state.end - state.start);
	output.SetChildCardinality(this_count);
	for (idx_t idx = 0; idx < state.column_ids.size(); idx++) {
		auto col_idx = state.column_ids[idx];
		if (col_idx == COLUMN_IDENTIFIER_ROW_ID) {
			output.data[idx].Sequence(state.start, 1, this_count);
		} else if (col_idx == COLUMN_IDENTIFIER_EMPTY) {
			ConstantVector::SetNull(output.data[idx], count_t(this_count));
		} else {
			PandasBackendScanSwitch(context, data.pandas_bind_data[col_idx], this_count, state.start, output.data[idx]);
		}
	}
	state.start += this_count;
	data.lines_read += this_count;
}

unique_ptr<NodeStatistics> PandasScanFunction::PandasScanCardinality(ClientContext &context,
                                                                     const FunctionData *bind_data) {
	auto &data = bind_data->Cast<PandasScanFunctionData>();
	return make_uniq<NodeStatistics>(data.row_count, data.row_count);
}

static nb::object CopyWithColumnNames(const nb::object &df, const vector<string> &names) {
	nb::object copy_df = df.attr("copy")(false);
	nb::list new_columns;
	for (auto &name : names) {
		new_columns.append(nb::str(name.c_str(), name.size()));
	}
	copy_df.attr("columns") = std::move(new_columns);
	return copy_df;
}

nb::object PandasScanFunction::PandasReplaceCopiedNames(const nb::object &original_df) {
	return CopyWithColumnNames(original_df, ColumnNamesFromLabels(original_df.attr("columns")));
}

nb::object PandasScanFunction::PandasWithBoundNames(const nb::object &df) {
	auto labels = nb::list(nb::object(df.attr("columns")));
	auto names = ColumnNamesFromLabels(labels);
	for (idx_t col_idx = 0; col_idx < names.size(); col_idx++) {
		nb::handle label = labels[col_idx];
		if (!nb::isinstance<nb::str>(label) || nb::cast<std::string>(label) != names[col_idx]) {
			return CopyWithColumnNames(df, names);
		}
	}
	return df;
}

void PandasScanFunction::PandasSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data,
                                         const BoundTableFunction &function) {
	throw NotImplementedException("PandasScan function cannot be serialized");
}

unique_ptr<FunctionData> PandasScanFunction::PandasDeserialize(Deserializer &deserializer,
                                                               BoundTableFunction &function) {
	throw NotImplementedException("PandasScan function cannot be deserialized");
}

} // namespace duckdb

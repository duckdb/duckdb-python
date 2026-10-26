//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pandas/pandas_analyzer.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/python_conversion.hpp"

namespace duckdb {

class PandasNumpyColumn;
struct RegisteredArray;

//! How the cells that decided an object column's type were chosen
struct PandasSample {
	//! Every increment-th row from row 0 was sampled, masked rows aside
	idx_t increment = 1;
	//! When no sampled row held a value: the first row that did, which typed the column
	optional_idx fallback_row;
};

class PandasAnalyzer {
public:
	explicit PandasAnalyzer(ClientContext &context) : context(context) {
		analyzed_type = LogicalType::SQLNULL;

		Value result;
		auto lookup_result = context.TryGetCurrentSetting("pandas_analyze_sample", result);
		D_ASSERT((bool)lookup_result);
		sample_size = result.GetValue<uint64_t>();
	}

public:
	LogicalType GetListType(nb::object &ele, bool &can_convert);
	LogicalType DictToMap(const PyDictionary &dict, bool &can_convert);
	LogicalType DictToStruct(const PyDictionary &dict, bool &can_convert);
	LogicalType GetItemType(nb::object ele, bool &can_convert);
	//! Types an object column from the cells of its buffer; a row the mask marks holds no value
	bool Analyze(const PandasNumpyColumn &column, idx_t rows, optional_ptr<const RegisteredArray> mask);
	LogicalType AnalyzedType() {
		return analyzed_type;
	}
	const PandasSample &Sample() const {
		return sample;
	}

private:
	LogicalType InnerAnalyze(nb::object array, bool &can_convert);
	template <class GET_CELL>
	LogicalType SampleType(idx_t rows, idx_t increment, bool &can_convert, GET_CELL &&get_cell);
	uint64_t GetSampleIncrement(idx_t rows);

private:
	uint64_t sample_size;
	//! Holds the gil to allow python object creation/destruction
	nb::gil_scoped_acquire gil;
	//! The resulting analyzed type
	LogicalType analyzed_type;
	PandasSample sample;
	ClientContext &context;
};

} // namespace duckdb

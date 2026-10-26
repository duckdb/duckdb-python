#pragma once

#include "duckdb_python/pandas/pandas_column.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"

namespace duckdb {

//! A pyarrow backed str column, read from its exported chunks. The vectors the scan fills point into the chunks'
//! buffers, so the chunks live as long as the bind data, and are released or parked with it.
class PandasArrowColumn : public PandasColumn {
public:
	explicit PandasArrowColumn(unique_ptr<ArrowType> arrow_type_p)
	    : PandasColumn(PandasColumnBackend::ARROW), arrow_type(std::move(arrow_type_p)), starts {0} {
	}

	//! Takes the exported chunk over, releasing it at once when it holds no rows
	void AddChunk(ArrowArray &exported) {
		ArrowArrayWrapper chunk;
		chunk.arrow_array = exported;
		exported.release = nullptr;
		if (chunk.arrow_array.length == 0) {
			return;
		}
		starts.push_back(starts.back() + NumericCast<idx_t>(chunk.arrow_array.length));
		chunks.push_back(std::move(chunk));
	}

public:
	unique_ptr<ArrowType> arrow_type;
	//! Chunk i holds rows starts[i] up to starts[i + 1]; no chunk is empty
	vector<ArrowArrayWrapper> chunks;
	vector<idx_t> starts;
};

} // namespace duckdb

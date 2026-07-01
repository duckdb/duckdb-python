#pragma once

#include "duckdb_python/pandas/pandas_column.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb_python/numpy/numpy_array.hpp"

namespace duckdb {

class PandasNumpyColumn : public PandasColumn {
public:
	PandasNumpyColumn(NumpyArray array_p) : PandasColumn(PandasColumnBackend::NUMPY), array(std::move(array_p)) {
		auto &arr = array.GetArray();
		D_ASSERT(py::hasattr(arr, "strides"));
		stride = arr.attr("strides").attr("__getitem__")(0).cast<idx_t>();
	}

public:
	NumpyArray array;
	idx_t stride;
};

} // namespace duckdb

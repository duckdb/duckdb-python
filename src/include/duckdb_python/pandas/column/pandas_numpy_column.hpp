#pragma once

#include "duckdb_python/pandas/pandas_column.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/numpy/numpy_array.hpp"

namespace duckdb {

class PandasNumpyColumn : public PandasColumn {
public:
	PandasNumpyColumn(NumpyArray array_p) : PandasColumn(PandasColumnBackend::NUMPY), array(std::move(array_p)) {
		auto &arr = array.GetArray();
		D_ASSERT(nb::hasattr(arr, "strides"));
		auto first_stride = nb::cast<int64_t>(arr.attr("strides").attr("__getitem__")(0));
		auto item_size = nb::cast<int64_t>(arr.attr("dtype").attr("itemsize"));
		// The scan steps forward through whole aligned items, which a reversed view or a packed structured field lacks
		if (item_size <= 0 || first_stride < 0 || first_stride % item_size != 0 ||
		    !nb::cast<bool>(arr.attr("flags").attr("aligned"))) {
			array = NumpyArray(nb::module_::import_("numpy").attr("require")(arr, nb::arg("requirements") = "CA"));
		}
		auto copied_stride = nb::cast<int64_t>(array.GetArray().attr("strides").attr("__getitem__")(0));
		// numpy calls an array of one item contiguous whatever its stride, so the copy can leave it reversed
		stride = UnsafeNumericCast<idx_t>(copied_stride < 0 ? item_size : copied_stride);
	}

public:
	NumpyArray array;
	idx_t stride;
};

} // namespace duckdb

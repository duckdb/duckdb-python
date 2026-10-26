#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/python_object_container.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/numpy/numpy_array.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb_python/pandas/pandas_column.hpp"

namespace duckdb {

class ClientContext;

struct RegisteredArray {
	explicit RegisteredArray(NumpyArray numpy_array) : numpy_array(std::move(numpy_array)) {
	}
	NumpyArray numpy_array;
};

//! The scan reads a mask as one bool byte per row from its first element, so any other layout or dtype is copied.
std::unique_ptr<RegisteredArray> RegisterMask(nb::handle mask);

//! str() of each label, made unique the way the engine names result columns.
vector<string> ColumnNamesFromLabels(nb::handle labels);

//! The row count of an array for a column slot, which must be one dimensional.
idx_t ColumnArrayRows(const string &column, nb::handle array);
//! Every array a column slot holds must be one dimensional with the row count of the frame or dict it came from.
void CheckColumnArray(const string &column, nb::handle array, idx_t row_count, const string &counted_from);

struct PandasColumnBindData {
	//! The bound column name, as errors about the column show it
	string name;
	NumpyType numpy_type;
	std::unique_ptr<PandasColumn> pandas_col;
	std::unique_ptr<RegisteredArray> mask;
	//! Only for categorical types
	string internal_categorical_type;
	//! Hold ownership of objects created during scanning
	PythonObjectContainer object_str_val;
};

struct Pandas {
	//! Returns the frame's row count
	static idx_t Bind(ClientContext &config, nb::handle df, vector<PandasColumnBindData> &out,
	                  vector<LogicalType> &return_types, vector<Identifier> &names);
};

} // namespace duckdb

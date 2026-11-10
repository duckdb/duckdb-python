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
	//! The caller declared the column's type: each value converts into it as the appender converts a row
	bool declared = false;
	//! Of a declared column: the caller implied the type rather than stating it, so a cell is read only when it is of
	//! the type's kind, since a cast could change what it was
	bool implied_type = false;
	//! Of a declared column map() reads: a cell holding bytes where the type holds a BIT is refused, since df() hands
	//! a BIT value to the function as its storage bytes
	bool bit_storage_refused = false;
};

//! The categories of a CategoricalDtype the caller built from an ENUM's values, in order
struct PandasEnumDtype {
	nb::handle categories;
	LogicalType type;
};

struct Pandas {
	//! Returns the frame's row count; declared_types, when given, hold one type per column, INVALID where none; a
	//! categorical column whose categories are those of one of enum_dtypes, that very object, reads its codes as that
	//! ENUM
	static idx_t Bind(ClientContext &config, nb::handle df, vector<PandasColumnBindData> &out,
	                  vector<LogicalType> &return_types, vector<Identifier> &names,
	                  const vector<LogicalType> &declared_types = {}, const vector<PandasEnumDtype> &enum_dtypes = {});
};

} // namespace duckdb

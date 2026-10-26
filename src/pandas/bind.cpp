#include "duckdb_python/pandas/pandas_bind.hpp"
#include "duckdb_python/pandas/pandas_analyzer.hpp"
#include "duckdb_python/pandas/column/pandas_numpy_column.hpp"
#include "duckdb_python/pandas/column/pandas_arrow_column.hpp"
#include "duckdb_python/arrow/python_arrow_scan.hpp"
#include "duckdb_python/numpy/numpy_array.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_conversion.hpp"
#include "duckdb_python/pyutil.hpp"
#include "duckdb/main/query_result.hpp"

namespace duckdb {

namespace {

struct PandasBindColumn {
public:
	PandasBindColumn(nb::object type, nb::object column) : type(std::move(type)), handle(std::move(column)) {
	}

public:
	nb::object type;
	nb::object handle;
};

struct PandasDataFrameBind {
public:
	explicit PandasDataFrameBind(nb::handle &df) {
		names = nb::list(nb::object(df.attr("columns")));
		getter = df.attr("__getitem__");
	}
	PandasBindColumn operator[](idx_t index) const {
		D_ASSERT(index < names.size());
		auto column = nb::borrow<nb::object>(getter(names[index]));
		nb::object type = column.attr("dtype");
		return PandasBindColumn(std::move(type), std::move(column));
	}

public:
	nb::list names;

private:
	nb::object getter;
};

}; // namespace

vector<string> ColumnNamesFromLabels(nb::handle labels) {
	vector<string> names;
	for (auto label : labels) {
		names.push_back(nb::cast<std::string>(nb::str(label)));
	}
	QueryResult::DeduplicateColumns(names);
	return names;
}

idx_t ColumnArrayRows(const string &column, nb::handle array) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (!duckdb::PyUtil::IsInstance(array, import_cache.numpy.ndarray())) {
		auto type_name = nb::cast<std::string>(nb::str(nb::object(array.type().attr("__name__"))));
		throw InvalidInputException("Column '%s' holds a '%s' where a column holds a numpy.ndarray", column, type_name);
	}
	auto shape = nb::cast<nb::tuple>(array.attr("shape"));
	if (shape.size() != 1) {
		throw InvalidInputException("Column '%s' holds an array of %llu dimensions where a column has one", column,
		                            shape.size());
	}
	return nb::cast<idx_t>(shape[0]);
}

static void CheckColumnRows(const string &column, idx_t rows, idx_t row_count, const string &counted_from) {
	if (rows != row_count) {
		throw InvalidInputException("Column '%s' has %llu rows where %s has %llu", column, rows, counted_from,
		                            row_count);
	}
}

void CheckColumnArray(const string &column, nb::handle array, idx_t row_count, const string &counted_from) {
	CheckColumnRows(column, ColumnArrayRows(column, array), row_count, counted_from);
}

//! Reading the chunks costs an export of each chunk at every bind, where to_numpy costs a conversion of each row, so
//! a column whose chunks are shorter than this on average is cheaper to convert.
static constexpr idx_t ARROW_ROWS_PER_CHUNK = 50;

//! The column read from its own chunks, or null when it is to be converted with to_numpy instead
static unique_ptr<PandasArrowColumn> BindArrowStringColumn(ClientContext &context, const string &name,
                                                           nb::handle pandas_array, idx_t row_count,
                                                           const string &counted_from) {
	nb::object chunked_array = pandas_array.attr("__arrow_array__")();
	nb::object pyarrow_type = chunked_array.attr("type");
	// Before pyarrow 14 nothing exports through capsules.
	if (!PyUtil::HasAttribute(pyarrow_type, "__arrow_c_schema__")) {
		return nullptr;
	}
	if (row_count < ARROW_ROWS_PER_CHUNK * nb::cast<idx_t>(chunked_array.attr("num_chunks"))) {
		return nullptr;
	}
	auto subject = StringUtil::Format("column '%s'", name);
	nb::object schema = pyarrow_type.attr("__arrow_c_schema__")();
	auto arrow_type = ArrowType::GetArrowLogicalType(context, InCapsule<ArrowSchema>(schema, "arrow_schema", subject));
	// Only a plain layout may be read without the engine adopting the chunks.
	if (arrow_type->GetDuckType().id() != LogicalTypeId::VARCHAR ||
	    arrow_type->GetPhysicalType() != ArrowArrayPhysicalType::DEFAULT) {
		return nullptr;
	}
	auto result = make_uniq<PandasArrowColumn>(std::move(arrow_type));
	nb::object chunks = chunked_array.attr("chunks");
	for (auto chunk : chunks) {
		auto exported = nb::cast<nb::tuple>(chunk.attr("__arrow_c_array__")());
		result->AddChunk(InCapsule<ArrowArray>(exported[1], "arrow_array", subject));
	}
	CheckColumnRows(name, result->starts.back(), row_count, counted_from);
	return result;
}

std::unique_ptr<RegisteredArray> RegisterMask(nb::handle mask) {
	auto contiguous = nb::module_::import_("numpy").attr("ascontiguousarray")(mask, nb::arg("dtype") = "bool");
	return std::make_unique<RegisteredArray>(NumpyArray(std::move(contiguous)));
}

static LogicalType BindColumn(ClientContext &context, PandasBindColumn &column_p, PandasColumnBindData &bind_data,
                              idx_t row_count) {
	const string counted_from = "the frame";
	LogicalType column_type;
	auto &column = column_p.handle;

	bind_data.numpy_type = ConvertNumpyType(column_p.type);
	bool column_has_mask = nb::hasattr(column.attr("array"), "_mask");

	if (column_has_mask) {
		// masked object, fetch the internal data and mask array
		bind_data.mask = RegisterMask(column.attr("array").attr("_mask"));
		CheckColumnArray(bind_data.name, bind_data.mask->numpy_array.GetArray(), row_count, counted_from);
	}

	nb::object data;

	if (bind_data.numpy_type.type == NumpyNullableType::CATEGORY) {
		// for category types, we create an ENUM type for string or use the converted numpy type for the rest
		D_ASSERT(nb::hasattr(column, "cat"));
		D_ASSERT(nb::hasattr(column.attr("cat"), "categories"));
		NumpyArray categories(column.attr("cat").attr("categories"));
		auto categories_pd_type = ConvertNumpyType(categories.GetArray().attr("dtype"));
		// Legacy categories are backed by an `object` dtype; pandas >= 3.0 backs string categories with the new
		// StringDtype (reported as "str"), so treat both as string categories -> ENUM.
		if (categories_pd_type.type == NumpyNullableType::OBJECT ||
		    categories_pd_type.type == NumpyNullableType::STRING) {
			// Let's hope the object type is a string.
			bind_data.numpy_type.type = NumpyNullableType::CATEGORY;
			// str()-ify each category individually: pandas >= 3.0 categories are a StringArray whose elements are
			// numpy str scalars, which nanobind's vector<string>/string casters reject (nb::cast<vector<string>>
			// on the array throws). Iterating + nb::str handles both that and the legacy object[str] case.
			vector<string> enum_entries;
			for (auto category : categories.GetArray()) {
				enum_entries.push_back(TransformPythonString(nb::str(category)));
			}
			idx_t size = enum_entries.size();
			Vector enum_entries_vec(LogicalType::VARCHAR, size);
			auto enum_entries_ptr = FlatVector::GetDataMutable<string_t>(enum_entries_vec);
			for (idx_t i = 0; i < size; i++) {
				enum_entries_ptr[i] = StringVector::AddStringOrBlob(enum_entries_vec, enum_entries[i]);
			}
			D_ASSERT(nb::hasattr(column.attr("cat"), "codes"));
			column_type = LogicalType::ENUM(enum_entries_vec, size);
			// .to_numpy(): pandas >= 3.0 returns cat.codes as a Series (no .strides/.ctypes), but the scan needs a
			// real ndarray backing buffer; materialize it. (Older pandas returned an ndarray here directly.)
			data = column.attr("cat").attr("codes").attr("to_numpy")();
			bind_data.internal_categorical_type = nb::cast<std::string>(nb::str(nb::object(data.attr("dtype"))));
		} else {
			data = column.attr("to_numpy")();
			// for category types (non-strings), we use the converted numpy type
			bind_data.numpy_type = ConvertNumpyType(data.attr("dtype"));
			column_type = NumpyToLogicalType(bind_data.numpy_type);
		}
	} else if (bind_data.numpy_type.type == NumpyNullableType::FLOAT_16) {
		data = column.attr("to_numpy")("float32");
		bind_data.numpy_type.type = NumpyNullableType::FLOAT_32;
		column_type = NumpyToLogicalType(bind_data.numpy_type);
	} else {
		auto pandas_array = column.attr("array");
		auto &import_cache = *DuckDBPyConnection::ImportCache();
		if (duckdb::PyUtil::IsInstance(pandas_array, import_cache.pandas.arrays.ArrowStringArray())) {
			auto arrow_column = BindArrowStringColumn(context, bind_data.name, pandas_array, row_count, counted_from);
			if (arrow_column) {
				column_type = arrow_column->arrow_type->GetDuckType();
				bind_data.pandas_col = std::move(arrow_column);
				return column_type;
			}
			// On pandas 2 its _data is the pyarrow ChunkedArray, not an ndarray.
			data = column.attr("to_numpy")();
		} else if (duckdb::PyUtil::IsInstance(pandas_array, import_cache.pandas.arrays.StringArray())) {
			// to_numpy returns this same object array, but only after a pass over every value for missing ones.
			data = pandas_array.attr("_ndarray");
		} else if (nb::hasattr(pandas_array, "_data")) {
			// This means we can access the numpy array directly
			data = pandas_array.attr("_data");
		} else if (nb::hasattr(pandas_array, "asi8")) {
			// This is a datetime object, has the option to get the array as int64_t's
			data = pandas_array.attr("asi8");
		} else {
			// Otherwise we have to get it through 'to_numpy()'
			data = column.attr("to_numpy")();
		}
		column_type = NumpyToLogicalType(bind_data.numpy_type);
	}
	CheckColumnArray(bind_data.name, data, row_count, counted_from);
	auto numpy_column = std::make_unique<PandasNumpyColumn>(NumpyArray(std::move(data)));
	// Analyze the inner data type of the 'object' column
	if (bind_data.numpy_type.type == NumpyNullableType::OBJECT) {
		PandasAnalyzer analyzer(context);
		if (analyzer.Analyze(*numpy_column, row_count, bind_data.mask.get())) {
			column_type = analyzer.AnalyzedType();
		}
	}
	bind_data.pandas_col = std::move(numpy_column);
	return column_type;
}

idx_t Pandas::Bind(ClientContext &context, nb::handle df_p, vector<PandasColumnBindData> &bind_columns,
                   vector<LogicalType> &return_types, vector<Identifier> &names) {

	PandasDataFrameBind df(df_p);
	idx_t column_count = nb::len(df.names);
	if (column_count == 0) {
		throw InvalidInputException("Need a DataFrame with at least one column");
	}
	idx_t row_count = nb::len(nb::object(df_p.attr("index")));
	auto column_names = ColumnNamesFromLabels(df.names);

	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto pandas = import_cache.pandas();
	if (!pandas) {
		throw InvalidInputException("'pandas' is required for this operation, but it wasn't installed");
	}
	(void)import_cache.pandas.NA();
	(void)import_cache.pandas.NaT();
	if (NumpyMaIsImported()) {
		(void)import_cache.numpy.ma.masked();
	}

	return_types.reserve(column_count);
	names.reserve(column_count);
	// loop over every column
	for (idx_t col_idx = 0; col_idx < column_count; col_idx++) {
		PandasColumnBindData bind_data;
		bind_data.name = column_names[col_idx];
		names.emplace_back(bind_data.name);
		auto column = df[col_idx];
		auto column_type = BindColumn(context, column, bind_data, row_count);

		return_types.push_back(column_type);
		bind_columns.push_back(std::move(bind_data));
	}
	return row_count;
}

} // namespace duckdb

#include "duckdb_python/pandas/pandas_bind.hpp"
#include "duckdb_python/pandas/pandas_analyzer.hpp"
#include "duckdb_python/pandas/column/pandas_numpy_column.hpp"
#include "duckdb_python/numpy/numpy_array.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyutil.hpp"

namespace duckdb {

namespace {

struct PandasBindColumn {
public:
	PandasBindColumn(nb::handle name, nb::handle type, nb::object column)
	    : name(name), type(type), handle(std::move(column)) {
	}

public:
	nb::handle name;
	nb::handle type;
	nb::object handle;
};

struct PandasDataFrameBind {
public:
	explicit PandasDataFrameBind(nb::handle &df) {
		names = nb::list(nb::object(df.attr("columns")));
		types = nb::list(nb::object(df.attr("dtypes")));
		getter = df.attr("__getitem__");
	}
	PandasBindColumn operator[](idx_t index) const {
		D_ASSERT(index < names.size());
		auto column = nb::borrow<nb::object>(getter(names[index]));
		auto type = types[index];
		auto name = names[index];
		return PandasBindColumn(name, type, column);
	}

public:
	nb::list names;
	nb::list types;

private:
	nb::object getter;
};

}; // namespace

static LogicalType BindColumn(ClientContext &context, PandasBindColumn &column_p, PandasColumnBindData &bind_data) {
	LogicalType column_type;
	auto &column = column_p.handle;

	bind_data.numpy_type = ConvertNumpyType(column_p.type);
	bool column_has_mask = nb::hasattr(column.attr("array"), "_mask");

	if (column_has_mask) {
		// masked object, fetch the internal data and mask array
		bind_data.mask = std::make_unique<RegisteredArray>(NumpyArray(column.attr("array").attr("_mask")));
	}

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
				enum_entries.push_back(nb::cast<std::string>(nb::str(category)));
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
			NumpyArray pandas_col(column.attr("cat").attr("codes").attr("to_numpy")());
			bind_data.internal_categorical_type =
			    nb::cast<std::string>(nb::str(nb::object(pandas_col.GetArray().attr("dtype"))));
			bind_data.pandas_col = std::make_unique<PandasNumpyColumn>(std::move(pandas_col));
		} else {
			NumpyArray pandas_col(column.attr("to_numpy")());
			auto numpy_type = pandas_col.GetArray().attr("dtype");
			bind_data.pandas_col = std::make_unique<PandasNumpyColumn>(std::move(pandas_col));
			// for category types (non-strings), we use the converted numpy type
			bind_data.numpy_type = ConvertNumpyType(numpy_type);
			column_type = NumpyToLogicalType(bind_data.numpy_type);
		}
	} else if (bind_data.numpy_type.type == NumpyNullableType::FLOAT_16) {
		auto pandas_array = column.attr("array");
		bind_data.pandas_col = std::make_unique<PandasNumpyColumn>(NumpyArray(column.attr("to_numpy")("float32")));
		bind_data.numpy_type.type = NumpyNullableType::FLOAT_32;
		column_type = NumpyToLogicalType(bind_data.numpy_type);
	} else {
		auto pandas_array = column.attr("array");
		if (nb::hasattr(pandas_array, "_data")) {
			// This means we can access the numpy array directly
			bind_data.pandas_col = std::make_unique<PandasNumpyColumn>(NumpyArray(column.attr("array").attr("_data")));
		} else if (nb::hasattr(pandas_array, "asi8")) {
			// This is a datetime object, has the option to get the array as int64_t's
			bind_data.pandas_col = std::make_unique<PandasNumpyColumn>(NumpyArray(pandas_array.attr("asi8")));
		} else {
			// Otherwise we have to get it through 'to_numpy()'
			bind_data.pandas_col = std::make_unique<PandasNumpyColumn>(NumpyArray(column.attr("to_numpy")()));
		}
		column_type = NumpyToLogicalType(bind_data.numpy_type);
	}
	// Analyze the inner data type of the 'object' column
	if (bind_data.numpy_type.type == NumpyNullableType::OBJECT) {
		PandasAnalyzer analyzer(context);
		if (analyzer.Analyze(column)) {
			column_type = analyzer.AnalyzedType();
		}
	}
	return column_type;
}

void Pandas::Bind(ClientContext &context, nb::handle df_p, vector<PandasColumnBindData> &bind_columns,
                  vector<LogicalType> &return_types, vector<Identifier> &names) {

	PandasDataFrameBind df(df_p);
	idx_t column_count = nb::len(df.names);
	if (column_count == 0 || nb::len(df.types) == 0 || column_count != nb::len(df.types)) {
		throw InvalidInputException("Need a DataFrame with at least one column");
	}

	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto pandas = import_cache.pandas();
	if (!pandas) {
		throw InvalidInputException("'pandas' is required for this operation, but it wasn't installed");
	}
	(void)import_cache.pandas.NA();
	(void)import_cache.pandas.NaT();

	return_types.reserve(column_count);
	names.reserve(column_count);
	// loop over every column
	for (idx_t col_idx = 0; col_idx < column_count; col_idx++) {
		PandasColumnBindData bind_data;

		// Stringify any label (int/float/tuple/MultiIndex/datetime column names are all valid in pandas).
		// nb::cast<std::string> only accepts PyUnicode and would throw on a non-str label; CastToString runs
		// PyObject_Str like the pre-nanobind py::str(...) path did.
		names.emplace_back(duckdb::PyUtil::CastToString(df.names[col_idx]));
		auto column = df[col_idx];
		auto column_type = BindColumn(context, column, bind_data);

		return_types.push_back(column_type);
		bind_columns.push_back(std::move(bind_data));
	}
}

} // namespace duckdb

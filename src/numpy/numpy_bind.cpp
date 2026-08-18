#include "duckdb_python/numpy/numpy_bind.hpp"
#include "duckdb_python/numpy/array_wrapper.hpp"
#include "duckdb_python/numpy/numpy_array.hpp"
#include "duckdb_python/pandas/pandas_analyzer.hpp"
#include "duckdb_python/pandas/column/pandas_numpy_column.hpp"
#include "duckdb_python/pandas/pandas_bind.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"

namespace duckdb {

void NumpyBind::Bind(ClientContext &context, nb::handle df, vector<PandasColumnBindData> &bind_columns,
                     vector<LogicalType> &return_types, vector<Identifier> &names) {

	auto df_columns = nb::list(df.attr("keys")());
	auto df_types = nb::list();
	for (auto item : nb::cast<nb::dict>(df)) {
		if (nb::cast<std::string>(nb::str(nb::object(item.second.attr("dtype").attr("char")))) == "U") {
			df_types.attr("append")(nb::str("string"));
			continue;
		}
		df_types.attr("append")(nb::str(nb::object(item.second.attr("dtype"))));
	}
	auto get_fun = df.attr("__getitem__");
	if (nb::len(df_columns) == 0 || nb::len(df_types) == 0 || nb::len(df_columns) != nb::len(df_types)) {
		throw InvalidInputException("Need a DataFrame with at least one column");
	}
	for (idx_t col_idx = 0; col_idx < nb::len(df_columns); col_idx++) {
		LogicalType duckdb_col_type;
		PandasColumnBindData bind_data;

		names.emplace_back(nb::cast<std::string>(df_columns[col_idx]));
		bind_data.numpy_type = ConvertNumpyType(df_types[col_idx]);

		auto column = get_fun(df_columns[col_idx]);

		if (bind_data.numpy_type.type == NumpyNullableType::FLOAT_16) {
			bind_data.pandas_col = std::make_unique<PandasNumpyColumn>(NumpyArray(column.attr("astype")("float32")));
			bind_data.numpy_type.type = NumpyNullableType::FLOAT_32;
			duckdb_col_type = NumpyToLogicalType(bind_data.numpy_type);
		} else if (bind_data.numpy_type.type == NumpyNullableType::STRING) {
			bind_data.numpy_type.type = NumpyNullableType::CATEGORY;
			// here we call numpy.unique
			// this function call will return the unique values of a given array
			// together with the indices to reconstruct the given array
			auto uniq = nb::cast<nb::tuple>(nb::module_::import_("numpy").attr("unique")(column, false, true));
			vector<string> enum_entries = nb::cast<vector<string>>(uniq.attr("__getitem__")(0));
			idx_t size = enum_entries.size();
			Vector enum_entries_vec(LogicalType::VARCHAR, size);
			auto enum_entries_ptr = FlatVector::GetDataMutable<string_t>(enum_entries_vec);
			for (idx_t i = 0; i < size; i++) {
				enum_entries_ptr[i] = StringVector::AddStringOrBlob(enum_entries_vec, enum_entries[i]);
			}
			duckdb_col_type = LogicalType::ENUM(enum_entries_vec, size);
			auto pandas_col = uniq.attr("__getitem__")(1);
			bind_data.internal_categorical_type = nb::cast<std::string>(nb::str(nb::object(pandas_col.attr("dtype"))));
			bind_data.pandas_col = std::make_unique<PandasNumpyColumn>(NumpyArray(pandas_col));
		} else {
			bind_data.pandas_col = std::make_unique<PandasNumpyColumn>(NumpyArray(column));
			duckdb_col_type = NumpyToLogicalType(bind_data.numpy_type);
		}

		if (bind_data.numpy_type.type == NumpyNullableType::OBJECT) {
			PandasAnalyzer analyzer(context);
			if (analyzer.Analyze(get_fun(df_columns[col_idx]))) {
				duckdb_col_type = analyzer.AnalyzedType();
			}
		}

		return_types.push_back(duckdb_col_type);
		bind_columns.push_back(std::move(bind_data));
	}
}

} // namespace duckdb

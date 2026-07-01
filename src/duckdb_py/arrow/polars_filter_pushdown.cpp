#include "duckdb_python/arrow/polars_filter_pushdown.hpp"

#include "duckdb_python/arrow/filter_pushdown_visitor.hpp"
#include "duckdb_python/import_cache/python_import_cache.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_objects.hpp"

namespace duckdb {

namespace {

struct PolarsBackend : public FilterBackend {
	explicit PolarsBackend(const ClientProperties &client_properties_p)
	    : client_properties(client_properties_p), import_cache(*DuckDBPyConnection::ImportCache()) {
	}

	py::object MakeColumnRef(const vector<Identifier> &path) override {
		// pl.col(path[0]).struct.field(path[1]).struct.field(...) — polars supports arbitrary
		// chaining for nested struct access, verified empirically up to 3 levels.
		py::object col = import_cache.polars.col()(path[0]);
		for (idx_t i = 1; i < path.size(); i++) {
			col = col.attr("struct").attr("field")(path[i].GetIdentifierName());
		}
		return col;
	}

	py::object MakeScalar(const Value &v, const ArrowType *arrow_type, const string &timezone_config) override {
		// Polars handles type coercion for primitives; no ArrowType lookup is needed.
		(void)arrow_type;
		(void)timezone_config;
		return PythonObject::FromValue(v, v.type(), client_properties);
	}

	py::object Compare(ExpressionType op, py::object col, py::object scalar) override {
		switch (op) {
		case ExpressionType::COMPARE_EQUAL:
			return col.attr("__eq__")(scalar);
		case ExpressionType::COMPARE_NOTEQUAL:
			return col.attr("__ne__")(scalar);
		case ExpressionType::COMPARE_LESSTHAN:
			return col.attr("__lt__")(scalar);
		case ExpressionType::COMPARE_GREATERTHAN:
			return col.attr("__gt__")(scalar);
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			return col.attr("__le__")(scalar);
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return col.attr("__ge__")(scalar);
		default:
			throw NotImplementedException("Comparison Type %s can't be a polars pushdown filter",
			                              ExpressionTypeToString(op));
		}
	}

	py::object NaNCompare(ExpressionType op, py::object col) override {
		switch (op) {
		case ExpressionType::COMPARE_EQUAL:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return col.attr("is_nan")();
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_NOTEQUAL:
			return col.attr("is_nan")().attr("__invert__")();
		case ExpressionType::COMPARE_GREATERTHAN:
			// Nothing is greater than NaN.
			return import_cache.polars.lit()(false);
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			// Everything is less than or equal to NaN.
			return import_cache.polars.lit()(true);
		default:
			throw NotImplementedException("Unsupported comparison type (%s) for NaN values",
			                              ExpressionTypeToString(op));
		}
	}

	py::object IsNull(py::object col) override {
		return col.attr("is_null")();
	}

	py::object IsNotNull(py::object col) override {
		return col.attr("is_not_null")();
	}

	py::object IsIn(py::object col, const vector<Value> &values, const LogicalType &col_logical_type,
	                const string &timezone_config) override {
		(void)timezone_config;
		py::list py_values;
		for (auto &val : values) {
			py_values.append(PythonObject::FromValue(val, val.type(), client_properties));
		}
		if (col_logical_type.id() == LogicalTypeId::DECIMAL) {
			// Polars infers Decimal(38, scale) for a plain list of Python Decimal values,
			// which doesn't match the column's declared Decimal(precision, scale) — the call
			// then fails with `'is_in' cannot check for List(Decimal(38, _)) values in
			// Decimal(p, s) data`. Build a typed Series matching the column to side-step
			// that, and wrap it with `.implode()` to silence the
			// `is_in`-with-same-dtype-Series deprecation (issue 22149).
			uint8_t width;
			uint8_t scale;
			col_logical_type.GetDecimalProperties(width, scale);
			py::object dtype = import_cache.polars.Decimal()(py::arg("precision") = width, py::arg("scale") = scale);
			py::object typed_series =
			    import_cache.polars.Series()(py::arg("values") = py_values, py::arg("dtype") = dtype);
			return col.attr("is_in")(typed_series.attr("implode")());
		}
		return col.attr("is_in")(py_values);
	}

	py::object And(py::object a, py::object b) override {
		return a.attr("__and__")(b);
	}

	py::object Or(py::object a, py::object b) override {
		return a.attr("__or__")(b);
	}

private:
	const ClientProperties &client_properties;
	PythonImportCache &import_cache;
};

} // anonymous namespace

py::object PolarsFilterPushdown::TransformFilter(const TableFilterSet &filter_collection,
                                                 unordered_map<idx_t, string> &columns,
                                                 const unordered_map<idx_t, idx_t> &filter_to_col,
                                                 const ClientProperties &client_properties) {
	(void)filter_to_col;
	PolarsBackend backend(client_properties);
	py::object expression = py::none();
	for (auto &entry : filter_collection) {
		auto column_idx = entry.GetIndex();
		auto &column_name = columns[column_idx];
		D_ASSERT(columns.find(column_idx) != columns.end());

		vector<Identifier> column_path = {Identifier(column_name)};
		// Polars does not need ArrowType information — `nullptr` here propagates through the
		// shared walker; the PolarsBackend ignores the parameter in MakeScalar.
		py::object child_expression = duckdb::TransformFilter(entry.Filter(), std::move(column_path), backend, nullptr,
		                                                      client_properties.time_zone);
		if (child_expression.is(py::none())) {
			continue;
		}
		if (expression.is(py::none())) {
			expression = std::move(child_expression);
		} else {
			expression = expression.attr("__and__")(child_expression);
		}
	}
	return expression;
}

} // namespace duckdb

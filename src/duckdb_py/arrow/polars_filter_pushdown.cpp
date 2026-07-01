#include "duckdb_python/arrow/polars_filter_pushdown.hpp"

#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/struct_filter.hpp"
#include "duckdb/planner/table_filter.hpp"

#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_objects.hpp"

namespace duckdb {

static py::object TransformFilterRecursive(TableFilter &filter, py::object col_expr,
                                           const ClientProperties &client_properties) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();

	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		auto &constant = constant_filter.constant;
		auto &constant_type = constant.type();

		// Check for NaN
		bool is_nan = false;
		if (constant_type.id() == LogicalTypeId::FLOAT) {
			is_nan = Value::IsNan(constant.GetValue<float>());
		} else if (constant_type.id() == LogicalTypeId::DOUBLE) {
			is_nan = Value::IsNan(constant.GetValue<double>());
		}

		if (is_nan) {
			switch (constant_filter.comparison_type) {
			case ExpressionType::COMPARE_EQUAL:
			case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
				return col_expr.attr("is_nan")();
			case ExpressionType::COMPARE_LESSTHAN:
			case ExpressionType::COMPARE_NOTEQUAL:
				return col_expr.attr("is_nan")().attr("__invert__")();
			case ExpressionType::COMPARE_GREATERTHAN:
				return import_cache.polars.lit()(false);
			case ExpressionType::COMPARE_LESSTHANOREQUALTO:
				return import_cache.polars.lit()(true);
			default:
				return py::none();
			}
		}

		// Convert DuckDB Value to Python object
		auto py_value = PythonObject::FromValue(constant, constant_type, client_properties);

		switch (constant_filter.comparison_type) {
		case ExpressionType::COMPARE_EQUAL:
			return col_expr.attr("__eq__")(py_value);
		case ExpressionType::COMPARE_LESSTHAN:
			return col_expr.attr("__lt__")(py_value);
		case ExpressionType::COMPARE_GREATERTHAN:
			return col_expr.attr("__gt__")(py_value);
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			return col_expr.attr("__le__")(py_value);
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return col_expr.attr("__ge__")(py_value);
		case ExpressionType::COMPARE_NOTEQUAL:
			return col_expr.attr("__ne__")(py_value);
		default:
			return py::none();
		}
	}
	case TableFilterType::IS_NULL: {
		return col_expr.attr("is_null")();
	}
	case TableFilterType::IS_NOT_NULL: {
		return col_expr.attr("is_not_null")();
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &and_filter = filter.Cast<ConjunctionAndFilter>();
		py::object expression = py::none();
		for (idx_t i = 0; i < and_filter.child_filters.size(); i++) {
			auto child_expression = TransformFilterRecursive(*and_filter.child_filters[i], col_expr, client_properties);
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
	case TableFilterType::CONJUNCTION_OR: {
		auto &or_filter = filter.Cast<ConjunctionOrFilter>();
		py::object expression = py::none();
		for (idx_t i = 0; i < or_filter.child_filters.size(); i++) {
			auto child_expression = TransformFilterRecursive(*or_filter.child_filters[i], col_expr, client_properties);
			if (child_expression.is(py::none())) {
				// Can't skip children in OR
				return py::none();
			}
			if (expression.is(py::none())) {
				expression = std::move(child_expression);
			} else {
				expression = expression.attr("__or__")(child_expression);
			}
		}
		return expression;
	}
	case TableFilterType::STRUCT_EXTRACT: {
		auto &struct_filter = filter.Cast<StructFilter>();
		auto child_col = col_expr.attr("struct").attr("field")(struct_filter.child_name);
		return TransformFilterRecursive(*struct_filter.child_filter, child_col, client_properties);
	}
	case TableFilterType::IN_FILTER: {
		auto &in_filter = filter.Cast<InFilter>();
		py::list py_values;
		for (const auto &value : in_filter.values) {
			py_values.append(PythonObject::FromValue(value, value.type(), client_properties));
		}
		return col_expr.attr("is_in")(py_values);
	}
	case TableFilterType::OPTIONAL_FILTER: {
		auto &optional_filter = filter.Cast<OptionalFilter>();
		if (!optional_filter.child_filter) {
			return py::none();
		}
		return TransformFilterRecursive(*optional_filter.child_filter, col_expr, client_properties);
	}
	default:
		// We skip DYNAMIC_FILTER, EXPRESSION_FILTER, BLOOM_FILTER
		return py::none();
	}
}

py::object PolarsFilterPushdown::TransformFilter(const TableFilterSet &filter_collection,
                                                 unordered_map<idx_t, string> &columns,
                                                 const unordered_map<idx_t, idx_t> &filter_to_col,
                                                 const ClientProperties &client_properties) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto &filters_map = filter_collection.filters;

	py::object expression = py::none();
	for (auto &it : filters_map) {
		auto column_idx = it.first;
		auto &column_name = columns[column_idx];
		auto col_expr = import_cache.polars.col()(column_name);

		auto child_expression = TransformFilterRecursive(*it.second, col_expr, client_properties);
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

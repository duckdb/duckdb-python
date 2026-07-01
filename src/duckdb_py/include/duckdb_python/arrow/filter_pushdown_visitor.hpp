//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/arrow/filter_pushdown_visitor.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/value.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

// A FilterBackend abstracts the Python side of an `ExpressionFilter` →
// expression translation. The shared walker in this file handles the
// structural recursion (CONJUNCTION_AND/OR, struct_extract column paths, the
// optional / selectivity-optional filter wrappers, and the internal runtime
// filter functions) and dispatches leaf operations to the backend.
//
// Two backends exist today: PyArrowBackend (emits pyarrow.dataset.Expression)
// and PolarsBackend (emits polars.Expr). Adding a new backend is purely a
// matter of implementing this interface; the walker itself is reused.
//
// Convention: a backend method that cannot push the given filter must throw
// `NotImplementedException`. The walker swallows it at optional-filter
// boundaries (an optional filter is not required for correctness) and the
// top-level entry points catch it too, returning `py::none()` for the affected
// column. Throwing keeps the "I can't push this" path uniform across backends,
// replacing the old polars walker's ad hoc `return py::none()` style.
struct FilterBackend {
	virtual ~FilterBackend() = default;

	// Build a column expression from an accumulated path. `path` always has
	// at least one element (the top-level column). For nested struct
	// references the path accumulates one entry per `struct_extract`.
	virtual py::object MakeColumnRef(const vector<Identifier> &path) = 0;

	// Convert a DuckDB Value to a backend-native Python scalar. `arrow_type`
	// may be nullptr for backends that don't need Arrow type information
	// (polars relies on DuckDB LogicalType only). `timezone_config` is the
	// active session's time zone for `TIMESTAMP_TZ` handling.
	virtual py::object MakeScalar(const Value &v, const ArrowType *arrow_type, const string &timezone_config) = 0;

	// Apply a comparison operator. `op` is one of the COMPARE_* ExpressionTypes.
	// `scalar` is what MakeScalar returned. NaN special cases go through
	// NaNCompare instead.
	virtual py::object Compare(ExpressionType op, py::object col, py::object scalar) = 0;

	// NaN-specific comparison. DuckDB treats NaN as the greatest value, so
	// each operator decomposes into is_nan / ~is_nan / lit(true|false).
	virtual py::object NaNCompare(ExpressionType op, py::object col) = 0;

	virtual py::object IsNull(py::object col) = 0;
	virtual py::object IsNotNull(py::object col) = 0;

	// IN list. `col_logical_type` is the column's DuckDB logical type — needed
	// by polars to construct a typed Series with matching precision/scale for
	// decimal columns. PyArrow ignores this parameter and uses MakeScalar
	// per-element.
	virtual py::object IsIn(py::object col, const vector<Value> &values, const LogicalType &col_logical_type,
	                        const string &timezone_config) = 0;

	virtual py::object And(py::object a, py::object b) = 0;
	virtual py::object Or(py::object a, py::object b) = 0;
};

// Walk a TableFilter and emit a backend-specific expression. Since the
// table-filter -> expression-filter migration in core, the only runtime filter
// type is `EXPRESSION_FILTER`; this unwraps it and walks the expression tree.
// - `column_path` is the top-level column name; struct paths are accumulated
//    inside the expression walk via struct_extract.
// - `arrow_type` is the ArrowType for the current path leaf (nullable for
//    backends that don't track Arrow types).
// - Returns `py::none()` if no part of the filter could be pushed.
py::object TransformFilter(const TableFilter &filter, const vector<Identifier> &column_path, FilterBackend &backend,
                           const ArrowType *arrow_type, const string &timezone_config);

// Walk a bound Expression tree (the contents of an `ExpressionFilter`) and emit
// a backend-specific expression. Handles BOUND_FUNCTION comparisons,
// BOUND_OPERATOR (IS_NULL / IS_NOT_NULL / COMPARE_IN), BOUND_CONJUNCTION
// (AND/OR), struct_extract column chains, the optional / selectivity-optional
// wrappers (unwrapped from `bind_info`; an untranslatable child is swallowed),
// and the internal runtime filter functions (dynamic / bloom / perfect-hash-join
// / prefix-range, which are skipped). Returns `py::none()` for an optional or
// runtime filter that can't be pushed.
py::object TransformExpression(const Expression &expression, const vector<Identifier> &column_path,
                               FilterBackend &backend, const ArrowType *arrow_type, const string &timezone_config);

} // namespace duckdb

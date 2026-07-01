#include "duckdb_python/arrow/filter_pushdown_visitor.hpp"

#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"

namespace duckdb {

namespace {

bool ValueIsNan(const Value &value) {
	if (value.type().id() == LogicalTypeId::FLOAT) {
		return Value::IsNan(value.GetValue<float>());
	}
	if (value.type().id() == LogicalTypeId::DOUBLE) {
		return Value::IsNan(value.GetValue<double>());
	}
	return false;
}

// ResolveColumn walks a column-side expression to extract the (full path, leaf
// ArrowType) pair. Accepts a bare BoundReferenceExpression and (nested)
// `struct_extract` chains. Anything else throws NotImplementedException —
// that gives the OPTIONAL_FILTER catch point a chance to swallow it.
struct ResolvedColumn {
	vector<Identifier> path;
	const ArrowType *leaf_type;
};

ResolvedColumn ResolveColumn(const Expression &expr, const vector<Identifier> &root_path, const ArrowType *root_type) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		return {root_path, root_type};
	}
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		throw NotImplementedException("Cannot push down arrow scan filter on column-side expression: %s",
		                              ExpressionClassToString(expr.GetExpressionClass()));
	}
	auto &func = expr.Cast<BoundFunctionExpression>();
	idx_t child_idx;
	if (!TryGetStructExtractChildIndex(func, child_idx)) {
		throw NotImplementedException("Cannot push down arrow scan filter on column-side function: %s",
		                              ExpressionTypeToString(expr.GetExpressionType()));
	}
	// Recurse innermost-first so names accumulate root → leaf.
	auto inner = ResolveColumn(*func.GetChildren()[0], root_path, root_type);
	inner.path.push_back(StructType::GetChildName(func.GetChildren()[0]->GetReturnType(), child_idx));
	if (inner.leaf_type) {
		inner.leaf_type = &inner.leaf_type->GetTypeInfo<ArrowStructInfo>().GetChild(child_idx);
	}
	return inner;
}

py::object EmitCompare(FilterBackend &backend, ExpressionType op, py::object col, const Value &constant,
                       const ArrowType *arrow_type, const string &timezone_config) {
	if (ValueIsNan(constant)) {
		return backend.NaNCompare(op, std::move(col));
	}
	auto scalar = backend.MakeScalar(constant, arrow_type, timezone_config);
	return backend.Compare(op, std::move(col), std::move(scalar));
}

} // anonymous namespace

py::object TransformExpression(const Expression &expression, const vector<Identifier> &column_path,
                               FilterBackend &backend, const ArrowType *arrow_type, const string &timezone_config) {
	auto expression_class = expression.GetExpressionClass();
	auto expression_type = expression.GetExpressionType();

	if (expression_class == ExpressionClass::BOUND_FUNCTION) {
		auto &bound_function_expression = expression.Cast<BoundFunctionExpression>();
		if (BoundComparisonExpression::IsComparison(expression_type)) {
			auto &left = BoundComparisonExpression::Left(bound_function_expression);
			auto &right = BoundComparisonExpression::Right(bound_function_expression);

			optional_ptr<const Expression> column_side;
			optional_ptr<const BoundConstantExpression> constant_side;

			if (right.GetExpressionType() == ExpressionType::VALUE_CONSTANT) {
				column_side = &left;
				constant_side = &right.Cast<BoundConstantExpression>();
			} else if (left.GetExpressionType() == ExpressionType::VALUE_CONSTANT) {
				column_side = &right;
				constant_side = &left.Cast<BoundConstantExpression>();
				expression_type = FlipComparisonExpression(expression_type);
			} else {
				throw NotImplementedException("Can only push down constant comparisons.");
			}

			auto resolved = ResolveColumn(*column_side, column_path, arrow_type);
			auto col = backend.MakeColumnRef(resolved.path);
			return EmitCompare(backend, expression_type, std::move(col), constant_side->GetValue(), resolved.leaf_type,
			                   timezone_config);
		}

		// Internal table-filter functions. Since the table-filter -> expression-filter
		// migration in core, optional / dynamic / bloom / perfect-hash-join / prefix-range
		// filters no longer have dedicated TableFilter subtypes. They arrive as scalar
		// function wrappers inside the ExpressionFilter expression tree (see
		// table_filter_functions.hpp).
		const auto &func_name = bound_function_expression.Function().GetName();

		// OPTIONAL / SELECTIVITY_OPTIONAL wrap a child predicate that lives in `bind_info`
		// (their `children` hold only a placeholder column ref). An optional filter is never
		// required for correctness, so if its child can't be translated we push nothing for
		// it rather than failing the whole scan.
		if (func_name == OptionalFilterScalarFun::NAME || func_name == SelectivityOptionalFilterScalarFun::NAME) {
			optional_ptr<const Expression> child;
			if (bound_function_expression.BindInfo()) {
				if (func_name == OptionalFilterScalarFun::NAME) {
					child = bound_function_expression.BindInfo()
					            ->Cast<OptionalFilterFunctionData>()
					            .child_filter_expr.get();
				} else {
					child = bound_function_expression.BindInfo()
					            ->Cast<SelectivityOptionalFilterFunctionData>()
					            .child_filter_expr.get();
				}
			}
			if (!child) {
				return py::none();
			}
			try {
				return TransformExpression(*child, column_path, backend, arrow_type, timezone_config);
			} catch (const NotImplementedException &) {
				return py::none();
			}
		}

		// DYNAMIC / BLOOM / PERFECT_HASH_JOIN / PREFIX_RANGE are runtime filters with no
		// static pyarrow/polars equivalent. They are not required for correctness (the
		// engine applies them above the scan), so skip them.
		if (TableFilterFunctions::IsTableFilterFunction(func_name)) {
			return py::none();
		}
	}

	if (expression_class == ExpressionClass::BOUND_OPERATOR) {
		auto &op_expr = expression.Cast<BoundOperatorExpression>();
		if (expression_type == ExpressionType::OPERATOR_IS_NULL) {
			auto resolved = ResolveColumn(*op_expr.GetChildren()[0], column_path, arrow_type);
			auto col = backend.MakeColumnRef(resolved.path);
			return backend.IsNull(std::move(col));
		}
		if (expression_type == ExpressionType::OPERATOR_IS_NOT_NULL) {
			auto resolved = ResolveColumn(*op_expr.GetChildren()[0], column_path, arrow_type);
			auto col = backend.MakeColumnRef(resolved.path);
			return backend.IsNotNull(std::move(col));
		}
		if (expression_type == ExpressionType::COMPARE_IN) {
			auto resolved = ResolveColumn(*op_expr.GetChildren()[0], column_path, arrow_type);
			auto col = backend.MakeColumnRef(resolved.path);
			vector<Value> values;
			for (idx_t i = 1; i < op_expr.GetChildren().size(); i++) {
				auto &const_expr = op_expr.GetChildren()[i]->Cast<BoundConstantExpression>();
				values.push_back(const_expr.GetValue());
			}
			auto col_type = op_expr.GetChildren()[0]->GetReturnType();
			return backend.IsIn(std::move(col), values, col_type, timezone_config);
		}
	}

	if (expression_class == ExpressionClass::BOUND_CONJUNCTION) {
		if (expression_type == ExpressionType::CONJUNCTION_OR || expression_type == ExpressionType::CONJUNCTION_AND) {
			const bool is_and = expression_type == ExpressionType::CONJUNCTION_AND;
			auto &conj_expr = expression.Cast<BoundConjunctionExpression>();
			py::object result = py::none();
			for (idx_t i = 0; i < conj_expr.GetChildren().size(); i++) {
				py::object child_expression =
				    TransformExpression(*conj_expr.GetChildren()[i], column_path, backend, arrow_type, timezone_config);
				if (child_expression.is(py::none())) {
					if (is_and) {
						// A conjunct we can't push can simply be dropped: the remaining AND
						// terms still form a correct (if weaker) filter, and the engine
						// re-applies the rest above the scan.
						continue;
					}
					// An OR branch that can't be translated (e.g. a dynamic filter) would
					// make the pushed-down predicate stricter than the engine intends —
					// fall back to no pushdown for the whole disjunction.
					return py::none();
				}
				if (result.is(py::none())) {
					result = std::move(child_expression);
				} else if (is_and) {
					result = backend.And(std::move(result), std::move(child_expression));
				} else {
					result = backend.Or(std::move(result), std::move(child_expression));
				}
			}
			return result;
		}
	}

	throw NotImplementedException("Pushdown Filter Type %s is not currently supported in arrow scans",
	                              ExpressionClassToString(expression_class));
}

py::object TransformFilter(const TableFilter &filter, const vector<Identifier> &column_path, FilterBackend &backend,
                           const ArrowType *arrow_type, const string &timezone_config) {
	switch (filter.filter_type) {
	case TableFilterType::EXPRESSION_FILTER: {
		auto &expression_filter = filter.Cast<ExpressionFilter>();
		return TransformExpression(*expression_filter.expr, column_path, backend, arrow_type, timezone_config);
	}
	default:
		throw NotImplementedException("Pushdown Filter Type %s is not currently supported in arrow scans",
		                              EnumUtil::ToString(filter.filter_type));
	}
}

} // namespace duckdb

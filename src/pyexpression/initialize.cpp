#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/expression/pyexpression.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb_python/python_conversion.hpp"

namespace duckdb {

namespace {

// Binary operators take their operand as nb::object (not Expression) so that None can bind: nanobind rejects None for a
// bound-type parameter before the registered implicit conversion runs, so `expr == None` / `expr + None` would never
// reach the None -> SQL NULL conversion otherwise. We convert explicitly via TryToExpression (an existing Expression is
// copied, a str becomes a column reference, any other value (including None) becomes a constant). On a genuinely
// unconvertible operand we return Py_NotImplemented so Python falls back to the reflected operator / identity
// comparison, keeping e.g. `expr == object()` returning False
// instead of raising).
template <typename Build>
nb::object ExpressionBinaryOp(const nb::object &other, Build &&build) {
	std::unique_ptr<DuckDBPyExpression> converted;
	if (!DuckDBPyExpression::TryToExpression(other, converted)) {
		return nb::borrow(nb::handle(Py_NotImplemented));
	}
	return nb::cast(build(*converted));
}

} // namespace

// Forward binary operator __op__: self <op> other (other converted via ExpressionBinaryOp, so None -> SQL NULL).
#define DUCKDB_EXPR_BINARY_OP(PYNAME, METHOD)                                                                          \
	m.def(                                                                                                             \
	    PYNAME,                                                                                                        \
	    [](DuckDBPyExpression &self, const nb::object &other) {                                                        \
		    return ExpressionBinaryOp(other, [&](const DuckDBPyExpression &rhs) { return self.METHOD(rhs); });         \
	    },                                                                                                             \
	    nb::arg("expr").none(), docs, nb::is_operator())

// Reflected binary operator __rop__: other <op> self (other is the left operand, also accepts None).
#define DUCKDB_EXPR_REFLECTED_OP(PYNAME, METHOD)                                                                       \
	m.def(                                                                                                             \
	    PYNAME,                                                                                                        \
	    [](DuckDBPyExpression &self, const nb::object &other) {                                                        \
		    return ExpressionBinaryOp(other, [&](const DuckDBPyExpression &lhs) { return lhs.METHOD(self); });         \
	    },                                                                                                             \
	    nb::arg("expr").none(), docs, nb::is_operator())

void InitializeStaticMethods(nb::module_ &m) {
	const char *docs;

	// Constant Expression
	docs = "Create a constant expression from the provided value";
	m.def("ConstantExpression", &DuckDBPyExpression::ConstantExpression, nb::arg("value").none(),
	      docs); // None accepted (lit(None))

	// ColumnRef Expression
	docs = "Create a column reference from the provided column name";
	m.def("ColumnExpression", &DuckDBPyExpression::ColumnExpression, docs);

	// Default Expression
	docs = "";
	m.def("DefaultExpression", &DuckDBPyExpression::DefaultExpression, docs);

	// Case Expression
	docs = "";
	m.def("CaseExpression", &DuckDBPyExpression::CaseExpression, nb::arg("condition"), nb::arg("value").none(), docs);

	// Star Expression
	docs = "";
	m.def("StarExpression", &DuckDBPyExpression::StarExpression, nb::kw_only(), nb::arg("exclude") = nb::none(), docs);
	m.def("StarExpression", []() { return DuckDBPyExpression::StarExpression(); }, docs);

	// Function Expression
	docs = "";
	m.def("FunctionExpression", &DuckDBPyExpression::FunctionExpression,
	      docs); // nanobind: cannot name a positional before nb::args

	// Coalesce Operator
	docs = "";
	m.def("CoalesceOperator", &DuckDBPyExpression::Coalesce, docs);

	// Lambda Expression
	docs = "";
	m.def("LambdaExpression", &DuckDBPyExpression::LambdaExpression, nb::arg("lhs"), nb::arg("rhs"), docs);

	// SQL Expression
	docs = "";
	m.def("SQLExpression", &DuckDBPyExpression::SQLExpression, docs, nb::arg("expression"));
}

static void InitializeDunderMethods(nb::class_<DuckDBPyExpression> &m) {
	const char *docs;

	docs = R"(
		Add expr to self

		Parameters:
			expr: The expression to add together with

		Returns:
			FunctionExpression: self '+' expr
	)";

	DUCKDB_EXPR_BINARY_OP("__add__", Add);
	DUCKDB_EXPR_REFLECTED_OP("__radd__", Add);

	docs = R"(
		Negate the expression.

		Returns:
			FunctionExpression: -self
	)";
	m.def("__neg__", &DuckDBPyExpression::Negate, docs, nb::is_operator());

	docs = R"(
		Subtract expr from self

		Parameters:
			expr: The expression to subtract from

		Returns:
			FunctionExpression: self '-' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__sub__", Subtract);
	DUCKDB_EXPR_REFLECTED_OP("__rsub__", Subtract);

	docs = R"(
		Multiply self by expr

		Parameters:
			expr: The expression to multiply by

		Returns:
			FunctionExpression: self '*' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__mul__", Multiply);
	DUCKDB_EXPR_REFLECTED_OP("__rmul__", Multiply);

	docs = R"(
		Divide self by expr

		Parameters:
			expr: The expression to divide by

		Returns:
			FunctionExpression: self '/' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__div__", Division);
	DUCKDB_EXPR_REFLECTED_OP("__rdiv__", Division);

	DUCKDB_EXPR_BINARY_OP("__truediv__", Division);
	DUCKDB_EXPR_REFLECTED_OP("__rtruediv__", Division);

	docs = R"(
		(Floor) Divide self by expr

		Parameters:
			expr: The expression to (floor) divide by

		Returns:
			FunctionExpression: self '//' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__floordiv__", FloorDivision);
	DUCKDB_EXPR_REFLECTED_OP("__rfloordiv__", FloorDivision);

	docs = R"(
		Modulo self by expr

		Parameters:
			expr: The expression to modulo by

		Returns:
			FunctionExpression: self '%' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__mod__", Modulo);
	DUCKDB_EXPR_REFLECTED_OP("__rmod__", Modulo);

	docs = R"(
		Power self by expr

		Parameters:
			expr: The expression to power by

		Returns:
			FunctionExpression: self '**' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__pow__", Power);
	DUCKDB_EXPR_REFLECTED_OP("__rpow__", Power);

	docs = R"(
		Create an equality expression between two expressions

		Parameters:
			expr: The expression to check equality with

		Returns:
			FunctionExpression: self '=' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__eq__", Equality);

	docs = R"(
		Create an inequality expression between two expressions

		Parameters:
			expr: The expression to check inequality with

		Returns:
			FunctionExpression: self '!=' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__ne__", Inequality);

	docs = R"(
		Create a greater than expression between two expressions

		Parameters:
			expr: The expression to check

		Returns:
			FunctionExpression: self '>' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__gt__", GreaterThan);

	docs = R"(
		Create a greater than or equal expression between two expressions

		Parameters:
			expr: The expression to check

		Returns:
			FunctionExpression: self '>=' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__ge__", GreaterThanOrEqual);

	docs = R"(
		Create a less than expression between two expressions

		Parameters:
			expr: The expression to check

		Returns:
			FunctionExpression: self '<' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__lt__", LessThan);

	docs = R"(
		Create a less than or equal expression between two expressions

		Parameters:
			expr: The expression to check

		Returns:
			FunctionExpression: self '<=' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__le__", LessThanOrEqual);

	// AND, NOT and OR

	docs = R"(
		Binary-and self together with expr

		Parameters:
			expr: The expression to AND together with self

		Returns:
			FunctionExpression: self '&' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__and__", And);

	docs = R"(
		Binary-or self together with expr

		Parameters:
			expr: The expression to OR together with self

		Returns:
			FunctionExpression: self '|' expr
	)";
	DUCKDB_EXPR_BINARY_OP("__or__", Or);

	docs = R"(
		Create a binary-not expression from self

		Returns:
			FunctionExpression: ~self
	)";
	m.def("__invert__", &DuckDBPyExpression::Not, docs, nb::is_operator());

	docs = R"(
		Binary-and self together with expr

		Parameters:
			expr: The expression to AND together with self

		Returns:
			FunctionExpression: expr '&' self
	)";
	DUCKDB_EXPR_REFLECTED_OP("__rand__", And);

	docs = R"(
		Binary-or self together with expr

		Parameters:
			expr: The expression to OR together with self

		Returns:
			FunctionExpression: expr '|' self
	)";
	DUCKDB_EXPR_REFLECTED_OP("__ror__", Or);
}

#undef DUCKDB_EXPR_BINARY_OP
#undef DUCKDB_EXPR_REFLECTED_OP

static void InitializeImplicitConversion(nb::class_<DuckDBPyExpression> &m) {
	m.def(nb::new_([](const string &name) {
		auto names = nb::cast<nb::args>(nb::make_tuple(nb::str(name.c_str(), name.size())));
		return DuckDBPyExpression::ColumnExpression(names);
	}));
	m.def(nb::new_([](const nb::object &obj) {
		      auto val = TransformPythonValue(nullptr, obj);
		      return DuckDBPyExpression::InternalConstantExpression(std::move(val));
	      }),
	      nb::arg("value").none()); // accept None -> NULL constant (nanobind rejects None for nb::object otherwise)
	nb::implicitly_convertible<nb::str, DuckDBPyExpression>();
	nb::implicitly_convertible<nb::object, DuckDBPyExpression>();
}

void DuckDBPyExpression::Initialize(nb::module_ &m) {
	// nanobind types aren't weak-referenceable by default.
	auto expression = nb::class_<DuckDBPyExpression>(m, "Expression", nb::is_weak_referenceable());

	InitializeStaticMethods(m);
	InitializeDunderMethods(expression);
	InitializeImplicitConversion(expression);

	const char *docs;

	docs = R"(
		Print the stringified version of the expression.
	)";
	expression.def("show", &DuckDBPyExpression::Print, docs);
	expression.def("__duckdb_template__", &DuckDBPyExpression::ToString);

	docs = R"(
		Set the order by modifier to ASCENDING.
	)";
	expression.def("asc", &DuckDBPyExpression::Ascending, docs);

	docs = R"(
		Set the order by modifier to DESCENDING.
	)";
	expression.def("desc", &DuckDBPyExpression::Descending, docs);

	docs = R"(
		Set the NULL order by modifier to NULLS FIRST.
	)";
	expression.def("nulls_first", &DuckDBPyExpression::NullsFirst, docs);

	docs = R"(
		Set the NULL order by modifier to NULLS LAST.
	)";
	expression.def("nulls_last", &DuckDBPyExpression::NullsLast, docs);

	docs = R"(
		Create a binary IS NULL expression from self

		Returns:
			DuckDBPyExpression: self IS NULL
	)";
	expression.def("isnull", &DuckDBPyExpression::IsNull, docs);

	docs = R"(
		Create a binary IS NOT NULL expression from self

		Returns:
			DuckDBPyExpression: self IS NOT NULL
	)";
	expression.def("isnotnull", &DuckDBPyExpression::IsNotNull, docs);

	docs = R"(
		Return an IN expression comparing self to the input arguments.

		Returns:
			DuckDBPyExpression: The compare IN expression
	)";
	expression.def("isin", &DuckDBPyExpression::In, docs);

	docs = R"(
		Return a NOT IN expression comparing self to the input arguments.

		Returns:
			DuckDBPyExpression: The compare NOT IN expression
	)";
	expression.def("isnotin", &DuckDBPyExpression::NotIn, docs);

	docs = R"(
		Return the stringified version of the expression.

		Returns:
			str: The string representation.
	)";
	expression.def("__repr__", &DuckDBPyExpression::ToString, docs);

	expression.def("get_name", &DuckDBPyExpression::GetName, docs);

	docs = R"(
		Create a copy of this expression with the given alias.

		Parameters:
			name: The alias to use for the expression, this will affect how it can be referenced.

		Returns:
			Expression: self with an alias.
	)";
	expression.def("alias", &DuckDBPyExpression::SetAlias, docs);

	docs = R"(
		Add an additional WHEN <condition> THEN <value> clause to the CaseExpression.

		Parameters:
			condition: The condition that must be met.
			value: The value to use if the condition is met.

		Returns:
			CaseExpression: self with an additional WHEN clause.
	)";
	expression.def("when", &DuckDBPyExpression::When, nb::arg("condition"), nb::arg("value").none(), docs);

	docs = R"(
		Add an ELSE <value> clause to the CaseExpression.

		Parameters:
			value: The value to use if none of the WHEN conditions are met.

		Returns:
			CaseExpression: self with an ELSE clause.
	)";
	expression.def("otherwise", &DuckDBPyExpression::Else, nb::arg("value").none(), docs);

	docs = R"(
		Create a CastExpression to type from self

		Parameters:
			type: The type to cast to

		Returns:
			CastExpression: self::type
	)";
	expression.def("cast", &DuckDBPyExpression::Cast, nb::arg("type"), docs);

	docs = "";
	expression.def(
	    "between",
	    [](DuckDBPyExpression &self, const nb::object &lower, const nb::object &upper) {
		    return self.Between(*DuckDBPyExpression::ToExpression(lower), *DuckDBPyExpression::ToExpression(upper));
	    },
	    nb::arg("lower").none(), nb::arg("upper").none(), docs);

	docs = "";
	expression.def("collate", &DuckDBPyExpression::Collate, nb::arg("collation"), docs);
}

} // namespace duckdb

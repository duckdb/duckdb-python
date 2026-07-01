#include "duckdb_python/expression/pyexpression.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/case_expression.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/expression/between_expression.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression/lambda_expression.hpp"
#include "duckdb/parser/expression/operator_expression.hpp"
#include "duckdb/parser/expression/default_expression.hpp"
#include "duckdb/parser/expression/collate_expression.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/qualified_name.hpp"

namespace duckdb {

DuckDBPyExpression::DuckDBPyExpression(unique_ptr<ParsedExpression> expr_p, OrderType order_type,
                                       OrderByNullType null_order)
    : expression(std::move(expr_p)), null_order(null_order), order_type(order_type) {
	if (!expression) {
		throw InternalException("DuckDBPyExpression created without an expression");
	}
}

string DuckDBPyExpression::Type() const {
	return ExpressionTypeToString(expression->GetExpressionType());
}

string DuckDBPyExpression::ToString() const {
	return expression->ToString();
}

string DuckDBPyExpression::GetName() const {
	return expression->GetName().GetIdentifierName();
}

void DuckDBPyExpression::Print() const {
	Printer::Print(expression->ToString());
}

const ParsedExpression &DuckDBPyExpression::GetExpression() const {
	return *expression;
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Copy() const {
	auto expr = GetExpression().Copy();
	return std::make_shared<DuckDBPyExpression>(std::move(expr), order_type, null_order);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::SetAlias(const string &name) const {
	auto copied_expression = GetExpression().Copy();
	copied_expression->SetAlias(Identifier(name));
	return std::make_shared<DuckDBPyExpression>(std::move(copied_expression));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Cast(const DuckDBPyType &type) const {
	auto copied_expression = GetExpression().Copy();
	auto case_expr = make_uniq<duckdb::CastExpression>(type.Type(), std::move(copied_expression));
	return std::make_shared<DuckDBPyExpression>(std::move(case_expr));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Between(const DuckDBPyExpression &lower,
                                                                const DuckDBPyExpression &upper) {
	auto copied_expression = GetExpression().Copy();
	auto between_expr = make_uniq<BetweenExpression>(std::move(copied_expression), lower.GetExpression().Copy(),
	                                                 upper.GetExpression().Copy());
	return std::make_shared<DuckDBPyExpression>(std::move(between_expr));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Collate(const string &collation) {
	auto copied_expression = GetExpression().Copy();
	auto collation_expression = make_uniq<CollateExpression>(collation, std::move(copied_expression));
	return std::make_shared<DuckDBPyExpression>(std::move(collation_expression));
}

// Case Expression modifiers

void DuckDBPyExpression::AssertCaseExpression() const {
	if (expression->GetExpressionType() != ExpressionType::CASE_EXPR) {
		throw py::value_error("This method can only be used on a Expression resulting from CaseExpression or When");
	}
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::InternalWhen(unique_ptr<duckdb::CaseExpression> expr,
                                                                     const DuckDBPyExpression &condition,
                                                                     const DuckDBPyExpression &value) {
	CaseCheck check;
	check.when_expr = condition.GetExpression().Copy();
	check.then_expr = value.GetExpression().Copy();
	expr->CaseChecksMutable().push_back(std::move(check));
	return std::make_shared<DuckDBPyExpression>(std::move(expr));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::When(const DuckDBPyExpression &condition,
                                                             const DuckDBPyExpression &value) {
	AssertCaseExpression();
	auto expr_p = expression->Copy();
	auto expr = unique_ptr_cast<ParsedExpression, duckdb::CaseExpression>(std::move(expr_p));

	return InternalWhen(std::move(expr), condition, value);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Else(const DuckDBPyExpression &value) {
	AssertCaseExpression();
	auto expr_p = expression->Copy();
	auto expr = unique_ptr_cast<ParsedExpression, duckdb::CaseExpression>(std::move(expr_p));

	expr->ElseMutable() = value.GetExpression().Copy();
	return std::make_shared<DuckDBPyExpression>(std::move(expr));
}

// Binary operators

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Add(const DuckDBPyExpression &other) const {
	return DuckDBPyExpression::BinaryOperator("+", *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Subtract(const DuckDBPyExpression &other) const {
	return DuckDBPyExpression::BinaryOperator("-", *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Multiply(const DuckDBPyExpression &other) const {
	return DuckDBPyExpression::BinaryOperator("*", *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Division(const DuckDBPyExpression &other) const {
	return DuckDBPyExpression::BinaryOperator("/", *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::FloorDivision(const DuckDBPyExpression &other) const {
	return DuckDBPyExpression::BinaryOperator("//", *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Modulo(const DuckDBPyExpression &other) const {
	return DuckDBPyExpression::BinaryOperator("%", *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Power(const DuckDBPyExpression &other) const {
	return DuckDBPyExpression::BinaryOperator("**", *this, other);
}

// Comparison expressions

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Equality(const DuckDBPyExpression &other) {
	return ComparisonExpression(ExpressionType::COMPARE_EQUAL, *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Inequality(const DuckDBPyExpression &other) {
	return ComparisonExpression(ExpressionType::COMPARE_NOTEQUAL, *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::GreaterThan(const DuckDBPyExpression &other) {
	return ComparisonExpression(ExpressionType::COMPARE_GREATERTHAN, *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::GreaterThanOrEqual(const DuckDBPyExpression &other) {
	return ComparisonExpression(ExpressionType::COMPARE_GREATERTHANOREQUALTO, *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::LessThan(const DuckDBPyExpression &other) {
	return ComparisonExpression(ExpressionType::COMPARE_LESSTHAN, *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::LessThanOrEqual(const DuckDBPyExpression &other) {
	return ComparisonExpression(ExpressionType::COMPARE_LESSTHANOREQUALTO, *this, other);
}

// AND, OR and NOT

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Not() {
	return DuckDBPyExpression::InternalUnaryOperator(ExpressionType::OPERATOR_NOT, *this);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::And(const DuckDBPyExpression &other) const {
	return DuckDBPyExpression::InternalConjunction(ExpressionType::CONJUNCTION_AND, *this, other);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Or(const DuckDBPyExpression &other) const {
	return DuckDBPyExpression::InternalConjunction(ExpressionType::CONJUNCTION_OR, *this, other);
}

// NULL

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::IsNull() {
	return DuckDBPyExpression::InternalUnaryOperator(ExpressionType::OPERATOR_IS_NULL, *this);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::IsNotNull() {
	return DuckDBPyExpression::InternalUnaryOperator(ExpressionType::OPERATOR_IS_NOT_NULL, *this);
}

// IN / NOT IN

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::CreateCompareExpression(ExpressionType compare_type,
                                                                                const py::args &args) {
	D_ASSERT(args.size() >= 1);

	vector<unique_ptr<ParsedExpression>> expressions;
	expressions.reserve(args.size() + 1);
	expressions.push_back(GetExpression().Copy());

	for (auto arg : args) {
		std::shared_ptr<DuckDBPyExpression> py_expr;
		if (!py::try_cast<std::shared_ptr<DuckDBPyExpression>>(arg, py_expr)) {
			throw InvalidInputException("Please provide arguments of type Expression!");
		}
		auto expr = py_expr->GetExpression().Copy();
		expressions.push_back(std::move(expr));
	}
	auto operator_expr = make_uniq<OperatorExpression>(compare_type, std::move(expressions));
	return std::make_shared<DuckDBPyExpression>(std::move(operator_expr));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::In(const py::args &args) {
	if (args.size() == 0) {
		throw InvalidInputException("Incorrect amount of parameters to 'isin', needs at least 1 parameter");
	}
	return CreateCompareExpression(ExpressionType::COMPARE_IN, args);
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::NotIn(const py::args &args) {
	if (args.size() == 0) {
		throw InvalidInputException("Incorrect amount of parameters to 'isnotin', needs at least 1 parameter");
	}
	return CreateCompareExpression(ExpressionType::COMPARE_NOT_IN, args);
}

// COALESCE

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Coalesce(const py::args &args) {
	vector<unique_ptr<ParsedExpression>> expressions;
	expressions.reserve(args.size());

	for (auto arg : args) {
		std::shared_ptr<DuckDBPyExpression> py_expr;
		if (!py::try_cast<std::shared_ptr<DuckDBPyExpression>>(arg, py_expr)) {
			throw InvalidInputException("Please provide arguments of type Expression!");
		}
		auto expr = py_expr->GetExpression().Copy();
		expressions.push_back(std::move(expr));
	}
	if (expressions.empty()) {
		throw InvalidInputException("Please provide at least one argument");
	}
	auto operator_expr = make_uniq<OperatorExpression>(ExpressionType::OPERATOR_COALESCE, std::move(expressions));
	return std::make_shared<DuckDBPyExpression>(std::move(operator_expr));
}

// Order modifiers

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Ascending() {
	auto py_expr = Copy();
	py_expr->order_type = OrderType::ASCENDING;
	return py_expr;
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Descending() {
	auto py_expr = Copy();
	py_expr->order_type = OrderType::DESCENDING;
	return py_expr;
}

// Null order modifiers

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::NullsFirst() {
	auto py_expr = Copy();
	py_expr->null_order = OrderByNullType::NULLS_FIRST;
	return py_expr;
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::NullsLast() {
	auto py_expr = Copy();
	py_expr->null_order = OrderByNullType::NULLS_LAST;
	return py_expr;
}

// Unary operators

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::Negate() {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(GetExpression().Copy());
	return DuckDBPyExpression::InternalFunctionExpression("-", std::move(children), true);
}

// Static creation methods

static void PopulateExcludeList(qualified_column_set_t &exclude, py::object list_p) {
	if (py::none().is(list_p)) {
		list_p = py::list();
	}
	py::list list = py::cast<py::list>(list_p);
	for (auto item : list) {
		if (py::isinstance<py::str>(item)) {
			string col_str = std::string(py::str(item));
			QualifiedColumnName qname = QualifiedColumnName::Parse(col_str);
			exclude.insert(qname);
			continue;
		}
		std::shared_ptr<DuckDBPyExpression> expr;
		if (!py::try_cast(item, expr)) {
			throw py::value_error("Items in the exclude list should either be 'str' or Expression");
		}
		if (expr->GetExpression().GetExpressionType() != ExpressionType::COLUMN_REF) {
			throw py::value_error("Only ColumnExpressions are accepted Expression types here");
		}
		auto &column = expr->GetExpression().Cast<ColumnRefExpression>();
		exclude.insert(QualifiedColumnName(column.GetColumnName()));
	}
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::StarExpression(py::object exclude_list) {
	case_insensitive_set_t exclude;
	auto star = make_uniq<duckdb::StarExpression>();
	PopulateExcludeList(star->ExcludeListMutable(), std::move(exclude_list));
	return std::make_shared<DuckDBPyExpression>(std::move(star));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::ColumnExpression(const py::args &names) {
	vector<Identifier> column_names;
	if (names.size() == 1) {
		string column_name = std::string(py::str(names[0]));
		if (column_name == "*") {
			return StarExpression();
		}

		auto qualified_name = QualifiedName::Parse(column_name);
		if (!qualified_name.Catalog().empty()) {
			column_names.push_back(qualified_name.Catalog());
		}
		if (!qualified_name.Schema().empty()) {
			column_names.push_back(qualified_name.Schema());
		}
		column_names.push_back(qualified_name.Name());
	} else {
		for (auto &part : names) {
			column_names.push_back(Identifier(py::str(part)));
		}
	}
	auto column_ref = make_uniq<duckdb::ColumnRefExpression>(std::move(column_names));
	return std::make_shared<DuckDBPyExpression>(std::move(column_ref));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::DefaultExpression() {
	return std::make_shared<DuckDBPyExpression>(make_uniq<duckdb::DefaultExpression>());
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::ConstantExpression(const py::object &value) {
	auto val = TransformPythonValue(nullptr, value);
	return InternalConstantExpression(std::move(val));
}

static py::args CreateArgsFromItem(py::handle item) {
	if (py::isinstance<py::tuple>(item)) {
		return py::cast<py::args>(item);
	} else {
		return py::make_tuple(item);
	}
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::LambdaExpression(const py::object &lhs_p,
                                                                         const DuckDBPyExpression &rhs) {
	unique_ptr<ParsedExpression> lhs;
	if (py::isinstance<py::tuple>(lhs_p)) {
		// LambdaExpression(lhs=(<item>, <item>, <item>))
		auto lhs_tuple = py::cast<py::tuple>(lhs_p);
		vector<unique_ptr<ParsedExpression>> children;
		for (auto &item : lhs_tuple) {
			unique_ptr<ParsedExpression> column;
			if (py::isinstance<DuckDBPyExpression>(item)) {
				// 'item' is already an Expression, check its type and use it
				auto column_expr = py::cast<std::shared_ptr<DuckDBPyExpression>>(item);
				if (column_expr->GetExpression().GetExpressionType() != ExpressionType::COLUMN_REF) {
					throw py::value_error("'lhs' was provided as a tuple of columns, but one of the columns is not of "
					                      "type ColumnExpression");
				}
				column = column_expr->GetExpression().Copy();
			} else {
				// 'item' is a tuple[str, ...] or str, construct a ColumnExpression from it
				auto args = CreateArgsFromItem(item);
				auto column_expr = ColumnExpression(args);
				if (column_expr->GetExpression().GetExpressionType() != ExpressionType::COLUMN_REF) {
					throw py::value_error("'lhs' was provided as a tuple of columns, but one of the columns is not of "
					                      "type ColumnExpression");
				}
				column = std::move(column_expr->expression);
			}
			children.push_back(std::move(column));
		}
		auto row_function = InternalFunctionExpression("row", std::move(children), false);
		lhs = std::move(row_function->expression);
	} else if (py::isinstance<py::str>(lhs_p)) {
		// LambdaExpression(lhs=str)
		auto args = CreateArgsFromItem(lhs_p);
		auto column_expr = ColumnExpression(args);
		if (column_expr->GetExpression().GetExpressionType() != ExpressionType::COLUMN_REF) {
			throw py::value_error("'lhs' should be a valid ColumnExpression (or be used to create one)");
		}
		lhs = std::move(column_expr->expression);
	} else if (py::isinstance<DuckDBPyExpression>(lhs_p)) {
		// LambdaExpression(lhs=Expression)
		// 'lhs_p' is already an Expression, check its type and use it
		auto column_expr = py::cast<std::shared_ptr<DuckDBPyExpression>>(lhs_p);
		if (column_expr->GetExpression().GetExpressionType() != ExpressionType::COLUMN_REF) {
			throw py::value_error("'lhs' was an Expression, but is not of type ColumnExpression");
		}
		lhs = column_expr->GetExpression().Copy();
	} else {
		throw py::value_error("Please provide 'lhs' as either a tuple containing strings, or a single string");
	}
	auto lambda_expression = make_uniq<duckdb::LambdaExpression>(std::move(lhs), rhs.GetExpression().Copy());
	// Use the modern `lambda x, y: ...` syntax. The lhs we built (a column ref, or a `row` function for multiple
	// parameters) is identical to what the named-parameter constructor produces; only the syntax type differs, and
	// the single-arrow form is now deprecated and errors by default.
	lambda_expression->GetLambdaSyntaxTypeMutable() = LambdaSyntaxType::LAMBDA_KEYWORD;
	return std::make_shared<DuckDBPyExpression>(std::move(lambda_expression));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::SQLExpression(string sql) {
	auto conn = DuckDBPyConnection::DefaultConnection();
	auto &context = *conn->con.GetConnection().context;
	vector<unique_ptr<ParsedExpression>> expressions;
	try {
		expressions = Parser::ParseExpressionList(sql, context.GetParserOptions());
	} catch (std::runtime_error &e) {
		throw;
	}

	if (expressions.size() != 1) {
		throw InvalidInputException(
		    "Please provide only a single expression to SQLExpression, found %d expressions in the parsed string",
		    expressions.size());
	}

	return std::make_shared<DuckDBPyExpression>(std::move(expressions[0]));
}

// Private methods

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::BinaryOperator(const string &function_name,
                                                                       const DuckDBPyExpression &arg_one,
                                                                       const DuckDBPyExpression &arg_two) {
	vector<unique_ptr<ParsedExpression>> children;

	children.push_back(arg_one.GetExpression().Copy());
	children.push_back(arg_two.GetExpression().Copy());
	return InternalFunctionExpression(function_name, std::move(children), true);
}

std::shared_ptr<DuckDBPyExpression>
DuckDBPyExpression::InternalFunctionExpression(const string &function_name,
                                               vector<unique_ptr<ParsedExpression>> children, bool is_operator) {
	auto function_expression = make_uniq<duckdb::FunctionExpression>(Identifier(function_name), std::move(children),
	                                                                 nullptr, nullptr, false, is_operator);
	return std::make_shared<DuckDBPyExpression>(std::move(function_expression));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::InternalUnaryOperator(ExpressionType type,
                                                                              const DuckDBPyExpression &arg) {
	auto expr = arg.GetExpression().Copy();
	auto operator_expression = make_uniq<OperatorExpression>(type, std::move(expr));
	return std::make_shared<DuckDBPyExpression>(std::move(operator_expression));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::InternalConjunction(ExpressionType type,
                                                                            const DuckDBPyExpression &arg,
                                                                            const DuckDBPyExpression &other) {
	vector<unique_ptr<ParsedExpression>> children;
	children.reserve(2);
	children.push_back(arg.GetExpression().Copy());
	children.push_back(other.GetExpression().Copy());

	auto operator_expression = make_uniq<ConjunctionExpression>(type, std::move(children));
	return std::make_shared<DuckDBPyExpression>(std::move(operator_expression));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::InternalConstantExpression(Value val) {
	return std::make_shared<DuckDBPyExpression>(make_uniq<duckdb::ConstantExpression>(std::move(val)));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::ComparisonExpression(ExpressionType type,
                                                                             const DuckDBPyExpression &left_p,
                                                                             const DuckDBPyExpression &right_p) {
	auto left = left_p.GetExpression().Copy();
	auto right = right_p.GetExpression().Copy();
	return std::make_shared<DuckDBPyExpression>(
	    make_uniq<duckdb::ComparisonExpression>(type, std::move(left), std::move(right)));
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::CaseExpression(const DuckDBPyExpression &condition,
                                                                       const DuckDBPyExpression &value) {
	auto expr = make_uniq<duckdb::CaseExpression>();
	auto case_expr = InternalWhen(std::move(expr), condition, value);

	// Add NULL as default Else expression
	auto &internal_expression = reinterpret_cast<duckdb::CaseExpression &>(*case_expr->expression);
	internal_expression.ElseMutable() = make_uniq<duckdb::ConstantExpression>(Value(LogicalTypeId::SQLNULL));
	return case_expr;
}

std::shared_ptr<DuckDBPyExpression> DuckDBPyExpression::FunctionExpression(const string &function_name,
                                                                           const py::args &args) {
	vector<unique_ptr<ParsedExpression>> expressions;
	for (auto arg : args) {
		std::shared_ptr<DuckDBPyExpression> py_expr;
		if (!py::try_cast<std::shared_ptr<DuckDBPyExpression>>(arg, py_expr)) {
			string actual_type = py::str(py::type::of(arg));
			throw InvalidInputException("Expected argument of type Expression, received '%s' instead", actual_type);
		}
		auto expr = py_expr->GetExpression().Copy();
		expressions.push_back(std::move(expr));
	}
	return InternalFunctionExpression(function_name, std::move(expressions));
}

} // namespace duckdb

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/expression/pyexpression.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/parser/parsed_expression.hpp"
#include "duckdb/parser/expression/case_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb_python/python_conversion.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb/common/enums/order_type.hpp"

namespace duckdb {

struct DuckDBPyExpression : public std::enable_shared_from_this<DuckDBPyExpression> {
public:
	explicit DuckDBPyExpression(unique_ptr<ParsedExpression> expr, OrderType order_type = OrderType::ORDER_DEFAULT,
	                            OrderByNullType null_order = OrderByNullType::ORDER_DEFAULT);

public:
	std::shared_ptr<DuckDBPyExpression> shared_from_this() {
		return std::enable_shared_from_this<DuckDBPyExpression>::shared_from_this();
	}

public:
	static void Initialize(py::module_ &m);

	string Type() const;

	string ToString() const;
	string GetName() const;
	void Print() const;
	std::shared_ptr<DuckDBPyExpression> Add(const DuckDBPyExpression &other) const;
	std::shared_ptr<DuckDBPyExpression> Subtract(const DuckDBPyExpression &other) const;
	std::shared_ptr<DuckDBPyExpression> Multiply(const DuckDBPyExpression &other) const;
	std::shared_ptr<DuckDBPyExpression> Division(const DuckDBPyExpression &other) const;
	std::shared_ptr<DuckDBPyExpression> FloorDivision(const DuckDBPyExpression &other) const;
	std::shared_ptr<DuckDBPyExpression> Modulo(const DuckDBPyExpression &other) const;
	std::shared_ptr<DuckDBPyExpression> Power(const DuckDBPyExpression &other) const;
	std::shared_ptr<DuckDBPyExpression> Negate();

	// Equality operations

	std::shared_ptr<DuckDBPyExpression> Equality(const DuckDBPyExpression &other);
	std::shared_ptr<DuckDBPyExpression> Inequality(const DuckDBPyExpression &other);
	std::shared_ptr<DuckDBPyExpression> GreaterThan(const DuckDBPyExpression &other);
	std::shared_ptr<DuckDBPyExpression> GreaterThanOrEqual(const DuckDBPyExpression &other);
	std::shared_ptr<DuckDBPyExpression> LessThan(const DuckDBPyExpression &other);
	std::shared_ptr<DuckDBPyExpression> LessThanOrEqual(const DuckDBPyExpression &other);

	std::shared_ptr<DuckDBPyExpression> SetAlias(const string &alias) const;
	std::shared_ptr<DuckDBPyExpression> When(const DuckDBPyExpression &condition, const DuckDBPyExpression &value);
	std::shared_ptr<DuckDBPyExpression> Else(const DuckDBPyExpression &value);

	std::shared_ptr<DuckDBPyExpression> Cast(const DuckDBPyType &type) const;
	std::shared_ptr<DuckDBPyExpression> Between(const DuckDBPyExpression &lower, const DuckDBPyExpression &upper);
	std::shared_ptr<DuckDBPyExpression> Collate(const string &collation);

	// AND, OR and NOT

	std::shared_ptr<DuckDBPyExpression> Not();
	std::shared_ptr<DuckDBPyExpression> And(const DuckDBPyExpression &other) const;
	std::shared_ptr<DuckDBPyExpression> Or(const DuckDBPyExpression &other) const;

	// IS NULL / IS NOT NULL

	std::shared_ptr<DuckDBPyExpression> IsNull();
	std::shared_ptr<DuckDBPyExpression> IsNotNull();

	// IN / NOT IN

	std::shared_ptr<DuckDBPyExpression> CreateCompareExpression(ExpressionType compare_type, const py::args &args);
	std::shared_ptr<DuckDBPyExpression> In(const py::args &args);
	std::shared_ptr<DuckDBPyExpression> NotIn(const py::args &args);

	// Order modifiers

	std::shared_ptr<DuckDBPyExpression> Ascending();
	std::shared_ptr<DuckDBPyExpression> Descending();

	// Null order modifiers

	std::shared_ptr<DuckDBPyExpression> NullsFirst();
	std::shared_ptr<DuckDBPyExpression> NullsLast();

public:
	const ParsedExpression &GetExpression() const;
	std::shared_ptr<DuckDBPyExpression> Copy() const;

public:
	static std::shared_ptr<DuckDBPyExpression> StarExpression(py::object exclude = py::none());
	static std::shared_ptr<DuckDBPyExpression> ColumnExpression(const py::args &column_name);
	static std::shared_ptr<DuckDBPyExpression> DefaultExpression();
	static std::shared_ptr<DuckDBPyExpression> ConstantExpression(const py::object &value);
	static std::shared_ptr<DuckDBPyExpression> LambdaExpression(const py::object &lhs, const DuckDBPyExpression &rhs);
	static std::shared_ptr<DuckDBPyExpression> CaseExpression(const DuckDBPyExpression &condition,
	                                                          const DuckDBPyExpression &value);
	static std::shared_ptr<DuckDBPyExpression> FunctionExpression(const string &function_name, const py::args &args);
	static std::shared_ptr<DuckDBPyExpression> Coalesce(const py::args &args);
	static std::shared_ptr<DuckDBPyExpression> SQLExpression(string sql);

public:
	// Internal functions (not exposed to Python)
	static std::shared_ptr<DuckDBPyExpression> InternalFunctionExpression(const string &function_name,
	                                                                      vector<unique_ptr<ParsedExpression>> children,
	                                                                      bool is_operator = false);

	static std::shared_ptr<DuckDBPyExpression> InternalUnaryOperator(ExpressionType type,
	                                                                 const DuckDBPyExpression &arg);
	static std::shared_ptr<DuckDBPyExpression> InternalConjunction(ExpressionType type, const DuckDBPyExpression &arg,
	                                                               const DuckDBPyExpression &other);
	static std::shared_ptr<DuckDBPyExpression> InternalConstantExpression(Value value);
	static std::shared_ptr<DuckDBPyExpression>
	BinaryOperator(const string &function_name, const DuckDBPyExpression &arg_one, const DuckDBPyExpression &arg_two);
	static std::shared_ptr<DuckDBPyExpression> ComparisonExpression(ExpressionType type, const DuckDBPyExpression &left,
	                                                                const DuckDBPyExpression &right);
	static std::shared_ptr<DuckDBPyExpression> InternalWhen(unique_ptr<duckdb::CaseExpression> expr,
	                                                        const DuckDBPyExpression &condition,
	                                                        const DuckDBPyExpression &value);
	void AssertCaseExpression() const;

private:
	unique_ptr<ParsedExpression> expression;

public:
	OrderByNullType null_order = OrderByNullType::ORDER_DEFAULT;
	OrderType order_type = OrderType::ORDER_DEFAULT;
};

} // namespace duckdb

#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/relation/query_relation.hpp"
#include "duckdb/main/relation/join_relation.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/function/pragma/pragma_functions.hpp"
#include "duckdb/parser/statement/pragma_statement.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/parser/statement/explain_statement.hpp"
#include "duckdb/catalog/default/default_types.hpp"
#include "duckdb/main/relation/value_relation.hpp"
#include "duckdb/main/relation/table_function_relation.hpp"
#include "duckdb/main/relation/materialized_relation.hpp"
#include "duckdb/main/relation/table_relation.hpp"
#include "duckdb/main/relation/explain_relation.hpp"
#include "duckdb/main/relation/update_relation.hpp"
#include "duckdb_python/map.hpp"
#include "duckdb_python/expression/pyexpression.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"

namespace duckdb {

DuckDBPyRelation::DuckDBPyRelation(shared_ptr<Relation> rel_p) : rel(std::move(rel_p)) {
	if (!rel) {
		throw InternalException("DuckDBPyRelation created without a relation");
	}
	engine::ClaimingContextWrapper::Install(*rel);
	auto &columns = rel->Columns();
	for (auto &col : columns) {
		names.push_back(col.Name().GetIdentifierName());
		types.push_back(col.GetType());
	}
}

bool DuckDBPyRelation::CanBeRegisteredBy(Connection &con) {
	return CanBeRegisteredBy(con.context);
}

bool DuckDBPyRelation::CanBeRegisteredBy(ClientContext &context) {
	auto this_context = rel->context->TryGetContext();
	if (!this_context) {
		return false;
	}
	return &context == this_context.get();
}

bool DuckDBPyRelation::CanBeRegisteredBy(shared_ptr<ClientContext> &con) {
	if (!con) {
		return false;
	}
	return CanBeRegisteredBy(*con);
}

DuckDBPyRelation::~DuckDBPyRelation() {
	D_ASSERT(duckdb::PyUtil::GilCheck());
	nb::gil_scoped_release gil;
	rel.reset();
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ProjectFromExpression(const string &expression) {
	auto projected_relation = DeriveRelation(rel->Project(expression));
	for (auto &dep : this->rel->external_dependencies) {
		projected_relation->rel->AddExternalDependency(dep);
	}
	return projected_relation;
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Project(const nb::args &args, const string &groups) {
	auto arg_count = args.size();
	if (arg_count == 0) {
		return nullptr;
	}
	nb::handle first_arg = args[0];
	if (arg_count == 1 && nb::isinstance<nb::str>(first_arg)) {
		string expr_string = nb::cast<std::string>(nb::str(first_arg));
		return ProjectFromExpression(expr_string);
	} else {
		vector<unique_ptr<ParsedExpression>> expressions;
		for (auto arg : args) {
			auto py_expr = DuckDBPyExpression::ToExpression(arg);
			expressions.push_back(py_expr->GetExpression().Copy());
		}
		vector<string> empty_aliases;
		if (groups.empty()) {
			// No groups provided
			return DeriveRelation(rel->Project(std::move(expressions), empty_aliases));
		}
		return DeriveRelation(rel->Aggregate(std::move(expressions), groups));
	}
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ProjectFromTypes(const nb::object &obj) {
	if (!nb::isinstance<nb::list>(obj)) {
		throw InvalidInputException("'columns_by_type' expects a list containing types");
	}
	auto list = nb::list(obj);
	vector<LogicalType> types_filter;
	// Collect the list of types specified that will be our filter
	for (auto item : list) { // nanobind list iteration yields temporary handles; bind by value
		LogicalType type;
		if (nb::isinstance<nb::str>(item)) {
			string type_str = nb::cast<std::string>(nb::str(item));
			auto context = rel->context->GetContext();
			engine::ConnectionClaim claim(*context);
			context->RunFunctionInTransaction([&]() { type = TransformStringToLogicalType(type_str, *context); });
		} else if (nb::isinstance<DuckDBPyType>(item)) {
			auto *type_p = nb::cast<DuckDBPyType *>(item);
			type = type_p->Type();
		} else {
			string actual_type = nb::cast<std::string>(nb::str((item).type()));
			throw InvalidInputException("Can only project on objects of type DuckDBPyType or str, not '%s'",
			                            actual_type);
		}
		types_filter.push_back(std::move(type));
	}

	if (types_filter.empty()) {
		throw InvalidInputException("List of types can not be empty!");
	}

	string projection = "";
	for (idx_t i = 0; i < types.size(); i++) {
		auto &type = types[i];
		// Check if any of the types in the filter match the current type
		if (std::find_if(types_filter.begin(), types_filter.end(),
		                 [&](const LogicalType &filter) { return filter == type; }) != types_filter.end()) {
			if (!projection.empty()) {
				projection += ", ";
			}
			projection += SQLIdentifier(names[i]);
		}
	}
	if (projection.empty()) {
		throw InvalidInputException("None of the columns matched the provided type filter!");
	}
	return ProjectFromExpression(projection);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::SetAlias(const string &expr) {
	return DeriveRelation(rel->Alias(expr));
}

nb::str DuckDBPyRelation::GetAlias() {
	auto alias_str = rel->GetAlias();
	return nb::str(alias_str.c_str(), alias_str.size());
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Filter(const nb::object &expr) {
	if (nb::isinstance<nb::str>(expr)) {
		string expression = nb::cast<std::string>(expr);
		return FilterFromExpression(expression);
	}
	auto expression = DuckDBPyExpression::ToExpression(expr);
	auto expr_p = expression->GetExpression().Copy();
	return DeriveRelation(rel->Filter(std::move(expr_p)));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::FilterFromExpression(const string &expr) {
	return DeriveRelation(rel->Filter(expr));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Limit(int64_t n, int64_t offset) {
	return DeriveRelation(rel->Limit(n, offset));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Order(const string &expr) {
	return DeriveRelation(rel->Order(expr));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Sort(const nb::args &args) {
	vector<OrderByNode> order_nodes;
	order_nodes.reserve(args.size());

	for (auto arg : args) {
		auto py_expr = DuckDBPyExpression::ToExpression(arg);
		auto expr = py_expr->GetExpression().Copy();
		order_nodes.emplace_back(py_expr->order_type, py_expr->null_order, std::move(expr));
	}
	if (order_nodes.empty()) {
		throw InvalidInputException("Please provide at least one expression to sort on");
	}
	return DeriveRelation(rel->Order(std::move(order_nodes)));
}

vector<unique_ptr<ParsedExpression>> GetExpressions(ClientContext &context, const nb::object &expr) {
	if (duckdb::PyUtil::IsListLike(expr)) {
		vector<unique_ptr<ParsedExpression>> expressions;
		auto aggregate_list = nb::list(expr);
		for (auto item : aggregate_list) {
			auto py_expr = DuckDBPyExpression::ToExpression(item);
			expressions.push_back(py_expr->GetExpression().Copy());
		}
		return expressions;
	} else if (nb::isinstance<nb::str>(expr)) {
		auto aggregate_list = nb::cast<std::string>(nb::str(expr));
		return Parser(context).ParseExpressionList(aggregate_list);
	} else {
		// A single Expression could be supported here by wrapping it in a vector
		string actual_type = nb::cast<std::string>(nb::str((expr).type()));
		throw InvalidInputException("Please provide either a string or list of Expression objects, not %s",
		                            actual_type);
	}
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Aggregate(const nb::object &expr, const string &groups) {
	auto expressions = GetExpressions(*rel->context->GetContext(), expr);
	if (!groups.empty()) {
		return DeriveRelation(rel->Aggregate(std::move(expressions), groups));
	}
	return DeriveRelation(rel->Aggregate(std::move(expressions)));
}

nb::list DuckDBPyRelation::Description() {
	return DuckDBPyResult::GetDescription(names, types);
}

Relation &DuckDBPyRelation::GetRel() {
	if (!rel) {
		throw InternalException("DuckDBPyRelation - calling GetRel, but no rel was present");
	}
	return *rel;
}

struct DescribeAggregateInfo {
	explicit DescribeAggregateInfo(string name_p, bool numeric_only = false)
	    : name(std::move(name_p)), numeric_only(numeric_only) {
	}

	string name;
	bool numeric_only;
};

// Build the Describe query as two levels. The inner single-group aggregate computes one
// scalar per (column, aggregate); the outer projection pivots them into rows via parallel
// UNNESTs. This avoids GROUP BY ALL, which core rejects for UNNEST/UNLIST expressions.
void CreateDescribeExpressions(const vector<ColumnDefinition> &columns, const vector<DescribeAggregateInfo> &aggregates,
                               vector<string> &inner_aggregates, vector<string> &outer_projection) {
	outer_projection.reserve(columns.size() + 1);

	string aggr_names = "UNNEST([";
	for (idx_t i = 0; i < aggregates.size(); i++) {
		if (i > 0) {
			aggr_names += ", ";
		}
		aggr_names += "'";
		aggr_names += aggregates[i].name;
		aggr_names += "'";
	}
	aggr_names += "])";
	aggr_names += " AS aggr";
	outer_projection.push_back(aggr_names);

	for (idx_t c = 0; c < columns.size(); c++) {
		auto &col = columns[c];
		bool numeric = col.GetType().IsNumeric();
		string unnest = "UNNEST([";
		for (idx_t i = 0; i < aggregates.size(); i++) {
			auto alias = "__describe_" + std::to_string(c) + "_" + std::to_string(i);
			if (i > 0) {
				unnest += ", ";
			}
			unnest += SQLIdentifier(alias);
			string stat;
			if (aggregates[i].numeric_only && !numeric) {
				stat = "NULL";
			} else {
				stat = aggregates[i].name;
				stat += "(";
				stat += SQLIdentifier(col.GetName());
				stat += ")";
				stat += numeric ? "::DOUBLE" : "::VARCHAR";
			}
			stat += " AS " + SQLIdentifier(alias);
			inner_aggregates.push_back(stat);
		}
		unnest += "])";
		unnest += " AS " + SQLIdentifier(col.GetName());
		outer_projection.push_back(unnest);
	}
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Describe() {
	auto &columns = rel->Columns();
	vector<DescribeAggregateInfo> aggregates;
	aggregates = {DescribeAggregateInfo("count"),        DescribeAggregateInfo("mean", true),
	              DescribeAggregateInfo("stddev", true), DescribeAggregateInfo("min"),
	              DescribeAggregateInfo("max"),          DescribeAggregateInfo("median", true)};
	vector<string> inner_aggregates;
	vector<string> outer_projection;
	CreateDescribeExpressions(columns, aggregates, inner_aggregates, outer_projection);
	auto stats = rel->Aggregate(inner_aggregates);
	return DeriveRelation(stats->Project(outer_projection));
}

string DuckDBPyRelation::ToSQL() {
	try {
		return rel->GetQueryNode()->ToString();
	} catch (const std::exception &) {
		return "";
	}
}

string DuckDBPyRelation::GenerateExpressionList(const string &function_name, const string &aggregated_columns,
                                                const string &groups, const string &function_parameter,
                                                bool ignore_nulls, const string &projected_columns,
                                                const string &window_spec) {
	auto input = StringUtil::Split(aggregated_columns, ',');
	return GenerateExpressionList(function_name, std::move(input), groups, function_parameter, ignore_nulls,
	                              projected_columns, window_spec);
}

string DuckDBPyRelation::GenerateExpressionList(const string &function_name, vector<string> input, const string &groups,
                                                const string &function_parameter, bool ignore_nulls,
                                                const string &projected_columns, const string &window_spec) {
	string expr;

	if (StringUtil::CIEquals("count", function_name) && input.empty()) {
		// Insert an artificial '*'
		input.push_back("*");
	}

	if (!projected_columns.empty()) {
		expr = projected_columns + ", ";
	}

	if (input.empty() && !function_parameter.empty()) {
		return expr +=
		       function_name + "(" + function_parameter + ((ignore_nulls) ? " ignore nulls) " : ") ") + window_spec;
	}
	for (idx_t i = 0; i < input.size(); i++) {
		// We parse the input as an expression to validate it.
		auto trimmed_input = input[i];
		StringUtil::Trim(trimmed_input);
		if (trimmed_input.empty()) {
			throw ParserException("Invalid column expression: '%s'", input[i]);
		}

		unique_ptr<ParsedExpression> expression;
		try {
			auto expressions = Parser::GetBuiltinParser().ParseExpressionList(trimmed_input);
			if (expressions.size() == 1) {
				expression = std::move(expressions[0]);
			}
		} catch (const ParserException &) {
			// First attempt at parsing failed, the input might be a column name that needs quoting.
			auto quoted_input = SQLQuotedIdentifier::ToString(trimmed_input);
			auto expressions = Parser::GetBuiltinParser().ParseExpressionList(quoted_input);
			if (expressions.size() == 1 && expressions[0]->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
				expression = std::move(expressions[0]);
			}
		}

		if (!expression) {
			throw ParserException("Invalid column expression: %s", trimmed_input);
		}

		// ToString() handles escaping for all expression types
		auto escaped_input = expression->ToString();

		if (function_parameter.empty()) {
			expr += function_name + "(" + escaped_input + ((ignore_nulls) ? " ignore nulls) " : ") ") + window_spec;
		} else {
			expr += function_name + "(" + escaped_input + "," + function_parameter +
			        ((ignore_nulls) ? " ignore nulls) " : ") ") + window_spec;
		}

		if (i < input.size() - 1) {
			expr += ",";
		}
	}
	return expr;
}

/* General aggregate functions */

std::unique_ptr<DuckDBPyRelation>
DuckDBPyRelation::GenericAggregator(const string &function_name, const string &aggregated_columns, const string &groups,
                                    const string &function_parameter, const string &projected_columns) {

	//! Construct Aggregation Expression
	auto expr = GenerateExpressionList(function_name, aggregated_columns, groups, function_parameter, false,
	                                   projected_columns, "");
	return Aggregate(nb::str(expr.c_str(), expr.size()), groups);
}

std::unique_ptr<DuckDBPyRelation>
DuckDBPyRelation::GenericWindowFunction(const string &function_name, const string &function_parameters,
                                        const string &aggr_columns, const string &window_spec, const bool &ignore_nulls,
                                        const string &projected_columns) {
	auto expr = GenerateExpressionList(function_name, aggr_columns, "", function_parameters, ignore_nulls,
	                                   projected_columns, window_spec);
	return DeriveRelation(rel->Project(expr));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ApplyAggOrWin(const string &function_name,
                                                                  const string &agg_columns,
                                                                  const string &function_parameters,
                                                                  const string &groups, const string &window_spec,
                                                                  const string &projected_columns, bool ignore_nulls) {
	if (!groups.empty() && !window_spec.empty()) {
		throw InvalidInputException("Either groups or window must be set (can't be both at the same time)");
	}
	if (!window_spec.empty()) {
		return GenericWindowFunction(function_name, function_parameters, agg_columns, window_spec, ignore_nulls,
		                             projected_columns);
	} else {
		return GenericAggregator(function_name, agg_columns, groups, function_parameters, projected_columns);
	}
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::AnyValue(const std::string &column, const std::string &groups,
                                                             const std::string &window_spec,
                                                             const std::string &projected_columns) {
	return ApplyAggOrWin("any_value", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ArgMax(const std::string &arg_column,
                                                           const std::string &value_column, const std::string &groups,
                                                           const std::string &window_spec,
                                                           const std::string &projected_columns) {
	return ApplyAggOrWin("arg_max", arg_column, value_column, groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ArgMin(const std::string &arg_column,
                                                           const std::string &value_column, const std::string &groups,
                                                           const std::string &window_spec,
                                                           const std::string &projected_columns) {
	return ApplyAggOrWin("arg_min", arg_column, value_column, groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Avg(const std::string &column, const std::string &groups,
                                                        const std::string &window_spec,
                                                        const std::string &projected_columns) {
	return ApplyAggOrWin("avg", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BitAnd(const std::string &column, const std::string &groups,
                                                           const std::string &window_spec,
                                                           const std::string &projected_columns) {
	return ApplyAggOrWin("bit_and", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BitOr(const std::string &column, const std::string &groups,
                                                          const std::string &window_spec,
                                                          const std::string &projected_columns) {
	return ApplyAggOrWin("bit_or", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BitXor(const std::string &column, const std::string &groups,
                                                           const std::string &window_spec,
                                                           const std::string &projected_columns) {
	return ApplyAggOrWin("bit_xor", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation>
DuckDBPyRelation::BitStringAgg(const std::string &column, const Optional<nb::object> &min,
                               const Optional<nb::object> &max, const std::string &groups,
                               const std::string &window_spec, const std::string &projected_columns) {
	if ((min.is_none() && !max.is_none()) || (!min.is_none() && max.is_none())) {
		throw InvalidInputException("Both min and max values must be set");
	}
	if (!min.is_none()) {
		if (!nb::isinstance<nb::int_>(min) || !nb::isinstance<nb::int_>(max)) {
			throw InvalidTypeException("min and max must be of type int");
		}
	}
	auto bitstring_agg_params =
	    min.is_none() ? "" : (std::to_string(nb::cast<int>(min)) + "," + std::to_string(nb::cast<int>(max)));
	return ApplyAggOrWin("bitstring_agg", column, bitstring_agg_params, groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BoolAnd(const std::string &column, const std::string &groups,
                                                            const std::string &window_spec,
                                                            const std::string &projected_columns) {
	return ApplyAggOrWin("bool_and", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BoolOr(const std::string &column, const std::string &groups,
                                                           const std::string &window_spec,
                                                           const std::string &projected_columns) {
	return ApplyAggOrWin("bool_or", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ValueCounts(const std::string &column, const std::string &groups) {
	return Count(column, groups, "", column);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Count(const std::string &column, const std::string &groups,
                                                          const std::string &window_spec,
                                                          const std::string &projected_columns) {
	return ApplyAggOrWin("count", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::FAvg(const std::string &column, const std::string &groups,
                                                         const std::string &window_spec,
                                                         const std::string &projected_columns) {
	return ApplyAggOrWin("favg", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::First(const string &column, const std::string &groups,
                                                          const string &projected_columns) {
	return GenericAggregator("first", column, groups, "", projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::FSum(const std::string &column, const std::string &groups,
                                                         const std::string &window_spec,
                                                         const std::string &projected_columns) {
	return ApplyAggOrWin("fsum", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::GeoMean(const std::string &column, const std::string &groups,
                                                            const std::string &projected_columns) {
	return GenericAggregator("geomean", column, groups, "", projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Histogram(const std::string &column, const std::string &groups,
                                                              const std::string &window_spec,
                                                              const std::string &projected_columns) {
	return ApplyAggOrWin("histogram", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::List(const std::string &column, const std::string &groups,
                                                         const std::string &window_spec,
                                                         const std::string &projected_columns) {
	return ApplyAggOrWin("list", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Last(const std::string &column, const std::string &groups,
                                                         const std::string &projected_columns) {
	return GenericAggregator("last", column, groups, "", projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Max(const std::string &column, const std::string &groups,
                                                        const std::string &window_spec,
                                                        const std::string &projected_columns) {
	return ApplyAggOrWin("max", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Min(const std::string &column, const std::string &groups,
                                                        const std::string &window_spec,
                                                        const std::string &projected_columns) {
	return ApplyAggOrWin("min", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Product(const std::string &column, const std::string &groups,
                                                            const std::string &window_spec,
                                                            const std::string &projected_columns) {
	return ApplyAggOrWin("product", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::StringAgg(const std::string &column, const std::string &sep,
                                                              const std::string &groups, const std::string &window_spec,
                                                              const std::string &projected_columns) {
	auto string_agg_params = SQLString::ToString(sep);
	return ApplyAggOrWin("string_agg", column, string_agg_params, groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Sum(const std::string &column, const std::string &groups,
                                                        const std::string &window_spec,
                                                        const std::string &projected_columns) {
	return ApplyAggOrWin("sum", column, "", groups, window_spec, projected_columns);
}

/* TODO: Approximate aggregate functions */

/* TODO: Statistical aggregate functions */
std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Median(const std::string &column, const std::string &groups,
                                                           const std::string &window_spec,
                                                           const std::string &projected_columns) {
	return ApplyAggOrWin("median", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Mode(const std::string &column, const std::string &groups,
                                                         const std::string &window_spec,
                                                         const std::string &projected_columns) {
	return ApplyAggOrWin("mode", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::QuantileCont(const std::string &column, const nb::object &q,
                                                                 const std::string &groups,
                                                                 const std::string &window_spec,
                                                                 const std::string &projected_columns) {
	string quantile_params = "";
	if (nb::isinstance<nb::float_>(q)) {
		quantile_params = std::to_string(nb::cast<float>(q));
	} else if (nb::isinstance<nb::list>(q)) {
		auto aux = nb::cast<std::vector<double>>(q);
		quantile_params += "[";
		for (idx_t i = 0; i < aux.size(); i++) {
			quantile_params += std::to_string(aux[i]);
			if (i < aux.size() - 1) {
				quantile_params += ",";
			}
		}
		quantile_params += "]";
	} else {
		throw InvalidTypeException("Unsupported type for quantile");
	}
	return ApplyAggOrWin("quantile_cont", column, quantile_params, groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::QuantileDisc(const std::string &column, const nb::object &q,
                                                                 const std::string &groups,
                                                                 const std::string &window_spec,
                                                                 const std::string &projected_columns) {
	string quantile_params = "";
	if (nb::isinstance<nb::float_>(q)) {
		quantile_params = std::to_string(nb::cast<float>(q));
	} else if (nb::isinstance<nb::list>(q)) {
		auto aux = nb::cast<std::vector<double>>(q);
		quantile_params += "[";
		for (idx_t i = 0; i < aux.size(); i++) {
			quantile_params += std::to_string(aux[i]);
			if (i < aux.size() - 1) {
				quantile_params += ",";
			}
		}
		quantile_params += "]";
	} else {
		throw InvalidTypeException("Unsupported type for quantile");
	}
	return ApplyAggOrWin("quantile_disc", column, quantile_params, groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::StdPop(const std::string &column, const std::string &groups,
                                                           const std::string &window_spec,
                                                           const std::string &projected_columns) {
	return ApplyAggOrWin("stddev_pop", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::StdSamp(const std::string &column, const std::string &groups,
                                                            const std::string &window_spec,
                                                            const std::string &projected_columns) {
	return ApplyAggOrWin("stddev_samp", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::VarPop(const std::string &column, const std::string &groups,
                                                           const std::string &window_spec,
                                                           const std::string &projected_columns) {
	return ApplyAggOrWin("var_pop", column, "", groups, window_spec, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::VarSamp(const std::string &column, const std::string &groups,
                                                            const std::string &window_spec,
                                                            const std::string &projected_columns) {
	return ApplyAggOrWin("var_samp", column, "", groups, window_spec, projected_columns);
}

idx_t DuckDBPyRelation::Length() {
	auto aggregate_rel = GenericAggregator("count", "*");
	return aggregate_rel->ExecuteOrThrow()->FetchChunk()->GetValue(0, 0).GetValue<idx_t>();
}

nb::tuple DuckDBPyRelation::Shape() {
	auto length = Length();
	return nb::make_tuple(length, rel->Columns().size());
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Unique(const string &std_columns) {
	return DeriveRelation(rel->Project(std_columns)->Distinct());
}

/* General-purpose window functions */

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::RowNumber(const string &window_spec,
                                                              const string &projected_columns) {
	return GenericWindowFunction("row_number", "", "*", window_spec, false, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Rank(const string &window_spec, const string &projected_columns) {
	return GenericWindowFunction("rank", "", "*", window_spec, false, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::DenseRank(const string &window_spec,
                                                              const string &projected_columns) {
	return GenericWindowFunction("dense_rank", "", "*", window_spec, false, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::PercentRank(const string &window_spec,
                                                                const string &projected_columns) {
	return GenericWindowFunction("percent_rank", "", "*", window_spec, false, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::CumeDist(const string &window_spec,
                                                             const string &projected_columns) {
	return GenericWindowFunction("cume_dist", "", "*", window_spec, false, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::FirstValue(const string &column, const string &window_spec,
                                                               const string &projected_columns) {
	return GenericWindowFunction("first_value", "", column, window_spec, false, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::NTile(const string &window_spec, const int &num_buckets,
                                                          const string &projected_columns) {
	return GenericWindowFunction("ntile", std::to_string(num_buckets), "", window_spec, false, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Lag(const string &column, const string &window_spec,
                                                        const int &offset, const string &default_value,
                                                        const bool &ignore_nulls, const string &projected_columns) {
	string lag_params = "";
	if (offset != 0) {
		lag_params += std::to_string(offset);
	}
	if (!default_value.empty()) {
		lag_params += "," + default_value;
	}
	return GenericWindowFunction("lag", lag_params, column, window_spec, ignore_nulls, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::LastValue(const std::string &column, const std::string &window_spec,
                                                              const std::string &projected_columns) {
	return GenericWindowFunction("last_value", "", column, window_spec, false, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Lead(const string &column, const string &window_spec,
                                                         const int &offset, const string &default_value,
                                                         const bool &ignore_nulls, const string &projected_columns) {
	string lead_params = "";
	if (offset != 0) {
		lead_params += std::to_string(offset);
	}
	if (!default_value.empty()) {
		lead_params += "," + default_value;
	}
	return GenericWindowFunction("lead", lead_params, column, window_spec, ignore_nulls, projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::NthValue(const string &column, const string &window_spec,
                                                             const int &offset, const bool &ignore_nulls,
                                                             const string &projected_columns) {
	return GenericWindowFunction("nth_value", std::to_string(offset), column, window_spec, ignore_nulls,
	                             projected_columns);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Distinct() {
	return DeriveRelation(rel->Distinct());
}

//! Submits the relation and returns the undriven result. Call with the GIL released.
static shared_ptr<engine::Result> PySubmitRelation(const shared_ptr<Relation> &rel, const engine::Format &format) {
	return engine::Session(rel->context->GetContext()).Submit(rel, format);
}

static shared_ptr<engine::Result> PyExecuteRelation(const shared_ptr<Relation> &rel) {
	if (!rel) {
		return nullptr;
	}
	D_ASSERT(duckdb::PyUtil::GilCheck());
	nb::gil_scoped_release release;
	auto result = PySubmitRelation(rel, engine::Format::Chunks());
	result->Retain(DuckDBPyConnection::CheckSignals);
	return result;
}

shared_ptr<engine::Result> DuckDBPyRelation::ExecuteInternal() {
	return PyExecuteRelation(rel);
}

std::shared_ptr<DuckDBPyResult> DuckDBPyRelation::ExecuteOrThrow(bool stream_result, const engine::Format &format) {
	nb::gil_scoped_acquire gil;
	std::shared_ptr<DuckDBPyResult> py_result;
	{
		nb::gil_scoped_release release;
		auto submitted = PySubmitRelation(rel, format);
		if (!stream_result) {
			submitted->Retain(DuckDBPyConnection::CheckSignals);
		}
		py_result = std::make_shared<DuckDBPyResult>(std::move(submitted));
	}
	return py_result;
}

PandasDataFrame DuckDBPyRelation::FetchDF(bool date_as_object) {
	return ExecuteOrThrow()->FetchDF(date_as_object);
}

nb::list DuckDBPyRelation::FetchAll() {
	return ExecuteOrThrow()->Fetchall();
}

nb::dict DuckDBPyRelation::FetchNumpy() {
	return ExecuteOrThrow()->FetchNumpy();
}

nb::dict DuckDBPyRelation::FetchPyTorch() {
	return ExecuteOrThrow()->FetchPyTorch();
}

nb::dict DuckDBPyRelation::FetchTF() {
	return ExecuteOrThrow()->FetchTF();
}

nb::dict DuckDBPyRelation::FetchNumpyInternal(bool chunked, idx_t vectors_per_chunk) {
	return ExecuteOrThrow()->FetchNumpyInternal(chunked, vectors_per_chunk);
}

pyarrow::Table DuckDBPyRelation::ToArrowTableInternal(idx_t batch_size, bool to_polars) {
	return ExecuteOrThrow(false, engine::Format::Arrow(batch_size))->FetchArrowTable(batch_size, to_polars);
}

duckdb::pyarrow::Table DuckDBPyRelation::ToArrowTable(idx_t batch_size) {
	return ToArrowTableInternal(batch_size, false);
}

nb::object DuckDBPyRelation::ToArrowCapsule(const nb::object &requested_schema) {
	return ExecuteOrThrow(true, engine::Format::Arrow(DuckDBPyResult::DEFAULT_ARROW_BATCH_SIZE))->FetchArrowCapsule();
}

PolarsDataFrame DuckDBPyRelation::ToPolars(idx_t batch_size, bool lazy) {
	if (!lazy) {
		auto arrow = ToArrowTableInternal(batch_size, true);
		return nb::cast<PolarsDataFrame>(
		    nb::module_::import_("polars").attr("from_arrow")(arrow, nb::arg("rechunk") = false));
	}
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto lazy_frame_produce = import_cache.duckdb.polars_io.duckdb_source();
	auto result_names = names;
	QueryResult::DeduplicateColumns(result_names);
	auto client_properties = rel->context->GetContext()->GetClientProperties();
	nb::list batches;
	auto empty_table = pyarrow::ToArrowTable(types, result_names, batches, client_properties);

	auto polars_df = nb::cast<PolarsDataFrame>(nb::module_::import_("polars").attr("DataFrame")(empty_table));
	auto polars_schema = polars_df.attr("schema");

	return lazy_frame_produce(*this, polars_schema);
}

duckdb::pyarrow::RecordBatchReader DuckDBPyRelation::ToRecordBatch(idx_t batch_size) {
	DuckDBPyResult::CheckBatchSize(batch_size);
	return ExecuteOrThrow(true, engine::Format::Arrow(batch_size))->FetchRecordBatchReader(batch_size);
}

void DuckDBPyRelation::Close() {
	// A relation keeps no result, so there is nothing to close
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Materialize() {
	unique_ptr<ColumnDataCollection> collection;
	vector<Identifier> result_names;
	{
		D_ASSERT(duckdb::PyUtil::GilCheck());
		// Ending a query waits for its tasks, and a Python UDF among them needs the GIL, so the result must go
		// out of scope, also on an error, before the GIL is taken back
		nb::gil_scoped_release release;
		auto result = PySubmitRelation(rel, engine::Format::Chunks());
		result_names = result->Names();
		collection = result->TakeChunks(DuckDBPyConnection::CheckSignals);
	}
	auto materialized = make_shared_ptr<MaterializedRelation>(rel->context->GetContext(), std::move(collection),
	                                                          std::move(result_names), Identifier(rel->GetAlias()));
	return DeriveRelation(std::move(materialized));
}

bool DuckDBPyRelation::ContainsColumnByName(const string &name) const {
	return std::find_if(names.begin(), names.end(),
	                    [&](const string &item) { return StringUtil::CIEquals(name, item); }) != names.end();
}

void DuckDBPyRelation::SetConnectionOwner(nb::object owner) {
	connection_owner = std::move(owner);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::DeriveRelation(shared_ptr<Relation> new_rel) {
	auto result_ = std::make_unique<DuckDBPyRelation>(std::move(new_rel));
	result_->connection_owner = connection_owner;
	return result_;
}

static bool ContainsStructFieldByName(LogicalType &type, const string &name) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	const auto name_identifier = Identifier(name);
	const auto count = StructType::GetChildCount(type);
	for (idx_t i = 0; i < count; i++) {
		if (StructType::GetChildName(type, i) == name) {
			return true;
		}
	}
	return false;
}

//! Methods a relation does not have because it keeps no result between calls, with what replaces each
static const char *RemovedMethodReplacement(const string &name) {
	if (name == "fetchone") {
		return "rel.limit(1).fetchall()";
	}
	if (name == "fetchmany") {
		return "rel.limit(n).fetchall()";
	}
	if (name == "fetch_df_chunk") {
		return "rel.to_arrow_reader() to read in batches, or rel.limit(n).df()";
	}
	if (name == "execute") {
		return "rel.materialize() to run it once and keep its rows, or a method such as fetchall()";
	}
	return nullptr;
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::GetAttribute(const string &name) {
	// TODO: support fetching a result containing only column 'name' from a value_relation
	vector<Identifier> column_names;
	if (names.size() == 1 && ContainsStructFieldByName(types[0], name)) {
		// e.g 'rel['my_struct']['my_field']:
		// first 'my_struct' is selected by the bottom condition
		// then 'my_field' is accessed on the result of this
		column_names.push_back(Identifier(names[0]));
		column_names.push_back(Identifier(name));
	} else if (ContainsColumnByName(name)) {
		column_names.push_back(Identifier(name));
	}

	if (column_names.empty()) {
		if (auto replacement = RemovedMethodReplacement(name)) {
			throw nb::attribute_error(StringUtil::Format("Relations no longer have %s(): a relation keeps no result "
			                                             "between calls, so every call runs it again. Use %s instead.",
			                                             name, replacement)
			                              .c_str());
		}
		throw nb::attribute_error(
		    StringUtil::Format("This relation does not contain a column by the name of '%s'", name).c_str());
	}

	vector<unique_ptr<ParsedExpression>> expressions;
	expressions.push_back(std::move(make_uniq<ColumnRefExpression>(column_names)));
	vector<string> aliases;
	aliases.push_back(name);
	return DeriveRelation(rel->Project(std::move(expressions), aliases));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Union(DuckDBPyRelation *other) {
	return DeriveRelation(rel->Union(other->rel));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Except(DuckDBPyRelation *other) {
	return DeriveRelation(rel->Except(other->rel));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Intersect(DuckDBPyRelation *other) {
	return DeriveRelation(rel->Intersect(other->rel));
}

namespace {
struct SupportedPythonJoinType {
	string name;
	JoinType type;
};
} // namespace

static const SupportedPythonJoinType *GetSupportedJoinTypes(idx_t &length) {
	static const SupportedPythonJoinType SUPPORTED_TYPES[] = {{"left", JoinType::LEFT},   {"right", JoinType::RIGHT},
	                                                          {"outer", JoinType::OUTER}, {"semi", JoinType::SEMI},
	                                                          {"inner", JoinType::INNER}, {"anti", JoinType::ANTI}};
	static const auto SUPPORTED_TYPES_COUNT = sizeof(SUPPORTED_TYPES) / sizeof(SupportedPythonJoinType);
	length = SUPPORTED_TYPES_COUNT;
	return reinterpret_cast<const SupportedPythonJoinType *>(SUPPORTED_TYPES);
}

static JoinType ParseJoinType(const string &type) {
	idx_t supported_types_count;
	auto supported_types = GetSupportedJoinTypes(supported_types_count);
	for (idx_t i = 0; i < supported_types_count; i++) {
		auto &supported_type = supported_types[i];
		if (supported_type.name == type) {
			return supported_type.type;
		}
	}
	return JoinType::INVALID;
}

[[noreturn]] void ThrowUnsupportedJoinTypeError(const string &provided) {
	vector<string> supported_options;
	idx_t length;
	auto supported_types = GetSupportedJoinTypes(length);
	for (idx_t i = 0; i < length; i++) {
		supported_options.push_back(StringUtil::Format("'%s'", supported_types[i].name));
	}
	auto options = StringUtil::Join(supported_options, ", ");
	throw InvalidInputException("Unsupported join type %s, try one of: %s", provided, options);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Join(DuckDBPyRelation *other, const nb::object &condition,
                                                         const string &type) {
	if (!other) {
		throw InvalidInputException("No relation provided for join");
	}

	JoinType join_type;
	string type_string = StringUtil::Lower(type);
	StringUtil::Trim(type_string);

	join_type = ParseJoinType(type_string);
	if (join_type == JoinType::INVALID) {
		ThrowUnsupportedJoinTypeError(type);
	}
	auto alias = nb::cast<std::string>(GetAlias());
	auto other_alias = nb::cast<std::string>(other->GetAlias());
	if (StringUtil::CIEquals(alias, other_alias)) {
		throw InvalidInputException("Both relations have the same alias, please change the alias of one or both "
		                            "relations using 'rel = rel.set_alias(<new alias>)'");
	}
	if (nb::isinstance<nb::str>(condition)) {
		auto condition_string = nb::cast<std::string>(condition);
		return DeriveRelation(rel->Join(other->rel, condition_string, join_type));
	}
	vector<Identifier> using_list;
	if (duckdb::PyUtil::IsListLike(condition)) {
		for (auto item : nb::list(condition)) {
			if (!nb::isinstance<nb::str>(item)) {
				string actual_type = nb::cast<std::string>(nb::str((item).type()));
				throw InvalidInputException("Using clause should be a list of strings, not %s", actual_type);
			}
			using_list.push_back(Identifier(nb::cast<std::string>(nb::str(item))));
		}
		if (using_list.empty()) {
			throw InvalidInputException("Please provide at least one string in the condition to create a USING clause");
		}
		auto join_relation = make_shared_ptr<JoinRelation>(rel, other->rel, std::move(using_list), join_type);
		return DeriveRelation(std::move(join_relation));
	}
	// Strings (SQL condition) and lists (USING clause) are handled above; anything else is converted here.
	auto condition_expr = DuckDBPyExpression::ToExpression(condition);
	vector<unique_ptr<ParsedExpression>> conditions;
	conditions.push_back(condition_expr->GetExpression().Copy());
	return DeriveRelation(rel->Join(other->rel, std::move(conditions), join_type));
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Cross(DuckDBPyRelation *other) {
	return DeriveRelation(rel->CrossProduct(other->rel));
}

static Value NestedDictToStruct(const nb::object &dictionary) {
	if (!nb::isinstance<nb::dict>(dictionary)) {
		throw InvalidInputException("NestedDictToStruct only accepts a dictionary as input");
	}
	nb::dict dict_casted = nb::cast<nb::dict>(dictionary);

	child_list_t<Value> children;
	for (auto item : dict_casted) {
		nb::object item_key = nb::cast<nb::object>(item.first);
		nb::object item_value = nb::cast<nb::object>(item.second);

		if (!nb::isinstance<nb::str>(item_key)) {
			throw InvalidInputException("NestedDictToStruct only accepts a dictionary with string keys");
		}

		auto item_key_str = nb::cast<std::string>(nb::str(item_key));

		if (nb::isinstance<nb::int_>(item_value)) {
			int32_t item_value_int = (int32_t)nb::int_(item_value);
			children.push_back(std::make_pair(Identifier(item_key_str), Value(item_value_int)));
		} else if (nb::isinstance<nb::dict>(item_value)) {
			children.push_back(std::make_pair(Identifier(item_key_str), NestedDictToStruct(item_value)));
		} else {
			throw InvalidInputException(
			    "NestedDictToStruct only accepts a dictionary with integer values or nested dictionaries");
		}
	}
	return Value::STRUCT(std::move(children));
}

static void ApplyOverwriteOption(identifier_map_t<vector<Value>> &options, const string &function_name,
                                 const string &filename, Relation &relation, const nb::object &overwrite) {
	if (nb::none().is(overwrite)) {
		return;
	}
	if (!nb::isinstance<nb::bool_>(overwrite)) {
		throw InvalidInputException(function_name + " only accepts 'overwrite' as a boolean");
	}
	const bool overwrite_val = static_cast<bool>(nb::bool_(overwrite));
	options["overwrite_or_ignore"] = {Value::BOOLEAN(overwrite_val)};
	if (overwrite_val) {
		return;
	}
	// Single-file COPY replaces via a tmp rename, so overwrite_or_ignore=false is a no-op.
	auto context = relation.context->TryGetContext();
	if (!context) {
		throw InvalidInputException(function_name + " cannot run after the connection has been closed");
	}
	auto &fs = FileSystem::GetFileSystem(*context);
	if (fs.FileExists(filename)) {
		throw IOException("Cannot write to \"%s\" - it exists and is a file, not a directory! Enable OVERWRITE option "
		                  "to overwrite the file",
		                  filename);
	}
}

void DuckDBPyRelation::ToParquet(const string &filename, const nb::object &compression, const nb::object &field_ids,
                                 const nb::object &row_group_size_bytes, const nb::object &row_group_size,
                                 const nb::object &overwrite, const nb::object &per_thread_output,
                                 const nb::object &use_tmp_file, const nb::object &partition_by,
                                 const nb::object &write_partition_columns, const nb::object &append,
                                 const nb::object &filename_pattern, const nb::object &file_size_bytes) {
	identifier_map_t<vector<Value>> options;

	if (!nb::none().is(compression)) {
		if (!nb::isinstance<nb::str>(compression)) {
			throw InvalidInputException("to_parquet only accepts 'compression' as a string");
		}
		options["compression"] = {Value(nb::cast<std::string>(compression))};
	}

	if (!nb::none().is(field_ids)) {
		if (nb::isinstance<nb::dict>(field_ids)) {
			Value field_ids_value = NestedDictToStruct(field_ids);
			options["field_ids"] = {field_ids_value};
		} else if (nb::isinstance<nb::str>(field_ids)) {
			options["field_ids"] = {Value(nb::cast<std::string>(field_ids))};
		} else {
			throw InvalidInputException("to_parquet only accepts 'field_ids' as a dictionary or 'auto'");
		}
	}

	if (!nb::none().is(row_group_size_bytes)) {
		if (nb::isinstance<nb::int_>(row_group_size_bytes)) {
			int64_t row_group_size_bytes_int = (int64_t)nb::int_(row_group_size_bytes);
			options["row_group_size_bytes"] = {Value(row_group_size_bytes_int)};
		} else if (nb::isinstance<nb::str>(row_group_size_bytes)) {
			options["row_group_size_bytes"] = {Value(nb::cast<std::string>(row_group_size_bytes))};
		} else {
			throw InvalidInputException(
			    "to_parquet only accepts 'row_group_size_bytes' as an integer or 'auto' string");
		}
	}

	if (!nb::none().is(row_group_size)) {
		if (!nb::isinstance<nb::int_>(row_group_size)) {
			throw InvalidInputException("to_parquet only accepts 'row_group_size' as an integer");
		}
		int64_t row_group_size_int = (int64_t)nb::int_(row_group_size);
		options["row_group_size"] = {Value(row_group_size_int)};
	}

	if (!nb::none().is(partition_by)) {
		if (!nb::isinstance<nb::list>(partition_by)) {
			throw InvalidInputException("to_parquet only accepts 'partition_by' as a list of strings");
		}
		vector<Value> partition_by_values;
		nb::list partition_fields = nb::cast<nb::list>(partition_by);
		for (auto field : partition_fields) {
			if (!nb::isinstance<nb::str>(field)) {
				throw InvalidInputException("to_parquet only accepts 'partition_by' as a list of strings");
			}
			partition_by_values.emplace_back(nb::cast<std::string>(nb::str(field)));
		}
		options["partition_by"] = {partition_by_values};
	}

	if (!nb::none().is(write_partition_columns)) {
		if (!nb::isinstance<nb::bool_>(write_partition_columns)) {
			throw InvalidInputException("to_parquet only accepts 'write_partition_columns' as a boolean");
		}
		options["write_partition_columns"] = {Value::BOOLEAN((bool)nb::bool_(write_partition_columns))};
	}

	if (!nb::none().is(append)) {
		if (!nb::isinstance<nb::bool_>(append)) {
			throw InvalidInputException("to_parquet only accepts 'append' as a boolean");
		}
		options["append"] = {Value::BOOLEAN((bool)nb::bool_(append))};
	}

	ApplyOverwriteOption(options, "to_parquet", filename, *rel, overwrite);

	if (!nb::none().is(per_thread_output)) {
		if (!nb::isinstance<nb::bool_>(per_thread_output)) {
			throw InvalidInputException("to_parquet only accepts 'per_thread_output' as a boolean");
		}
		options["per_thread_output"] = {Value::BOOLEAN((bool)nb::bool_(per_thread_output))};
	}

	if (!nb::none().is(use_tmp_file)) {
		if (!nb::isinstance<nb::bool_>(use_tmp_file)) {
			throw InvalidInputException("to_parquet only accepts 'use_tmp_file' as a boolean");
		}
		options["use_tmp_file"] = {Value::BOOLEAN((bool)nb::bool_(use_tmp_file))};
	}

	if (!nb::none().is(filename_pattern)) {
		if (!nb::isinstance<nb::str>(filename_pattern)) {
			throw InvalidInputException("to_parquet only accepts 'filename_pattern' as a string");
		}
		options["filename_pattern"] = {Value(nb::cast<std::string>(filename_pattern))};
	}

	if (!nb::none().is(file_size_bytes)) {
		if (nb::isinstance<nb::int_>(file_size_bytes)) {
			int64_t file_size_bytes_int = (int64_t)nb::int_(file_size_bytes);
			options["file_size_bytes"] = {Value(file_size_bytes_int)};
		} else if (nb::isinstance<nb::str>(file_size_bytes)) {
			options["file_size_bytes"] = {Value(nb::cast<std::string>(file_size_bytes))};
		} else {
			throw InvalidInputException("to_parquet only accepts 'file_size_bytes' as an integer or string");
		}
	}

	auto write_parquet = rel->WriteParquetRel(filename, std::move(options));
	PyExecuteRelation(write_parquet);
}

void DuckDBPyRelation::ToCSV(const string &filename, const nb::object &sep, const nb::object &na_rep,
                             const nb::object &header, const nb::object &quotechar, const nb::object &escapechar,
                             const nb::object &date_format, const nb::object &timestamp_format,
                             const nb::object &quoting, const nb::object &encoding, const nb::object &compression,
                             const nb::object &overwrite, const nb::object &per_thread_output,
                             const nb::object &use_tmp_file, const nb::object &partition_by,
                             const nb::object &write_partition_columns) {
	identifier_map_t<vector<Value>> options;

	if (!nb::none().is(sep)) {
		if (!nb::isinstance<nb::str>(sep)) {
			throw InvalidInputException("to_csv only accepts 'sep' as a string");
		}
		options["delimiter"] = {Value(nb::cast<std::string>(sep))};
	}

	if (!nb::none().is(na_rep)) {
		if (!nb::isinstance<nb::str>(na_rep)) {
			throw InvalidInputException("to_csv only accepts 'na_rep' as a string");
		}
		options["null"] = {Value(nb::cast<std::string>(na_rep))};
	}

	if (!nb::none().is(header)) {
		if (!nb::isinstance<nb::bool_>(header)) {
			throw InvalidInputException("to_csv only accepts 'header' as a boolean");
		}
		options["header"] = {Value::BOOLEAN((bool)nb::bool_(header))};
	}

	if (!nb::none().is(quotechar)) {
		if (!nb::isinstance<nb::str>(quotechar)) {
			throw InvalidInputException("to_csv only accepts 'quotechar' as a string");
		}
		options["quote"] = {Value(nb::cast<std::string>(quotechar))};
	}

	if (!nb::none().is(escapechar)) {
		if (!nb::isinstance<nb::str>(escapechar)) {
			throw InvalidInputException("to_csv only accepts 'escapechar' as a string");
		}
		options["escape"] = {Value(nb::cast<std::string>(escapechar))};
	}

	if (!nb::none().is(date_format)) {
		if (!nb::isinstance<nb::str>(date_format)) {
			throw InvalidInputException("to_csv only accepts 'date_format' as a string");
		}
		options["dateformat"] = {Value(nb::cast<std::string>(date_format))};
	}

	if (!nb::none().is(timestamp_format)) {
		if (!nb::isinstance<nb::str>(timestamp_format)) {
			throw InvalidInputException("to_csv only accepts 'timestamp_format' as a string");
		}
		options["timestampformat"] = {Value(nb::cast<std::string>(timestamp_format))};
	}

	if (!nb::none().is(quoting)) {
		// TODO: add list of strings as valid option
		if (nb::isinstance<nb::str>(quoting)) {
			string quoting_option = StringUtil::Lower(nb::cast<std::string>(nb::str(quoting)));
			if (quoting_option != "force" && quoting_option != "all") {
				throw InvalidInputException(
				    "to_csv 'quoting' supported options are ALL or FORCE (both set FORCE_QUOTE=True)");
			}
		} else if (nb::isinstance<nb::int_>(quoting)) {
			int64_t quoting_value = (int64_t)nb::int_(quoting);
			// csv.QUOTE_ALL expands to 1
			static constexpr int64_t QUOTE_ALL = 1;
			if (quoting_value != QUOTE_ALL) {
				throw InvalidInputException("Only csv.QUOTE_ALL is a supported option for 'quoting' currently");
			}
		} else {
			throw InvalidInputException(
			    "to_csv only accepts 'quoting' as a string or a constant from the 'csv' package");
		}
		options["force_quote"] = {Value("*")};
	}

	if (!nb::none().is(encoding)) {
		if (!nb::isinstance<nb::str>(encoding)) {
			throw InvalidInputException("to_csv only accepts 'encoding' as a string");
		}
		string encoding_option = StringUtil::Lower(nb::cast<std::string>(nb::str(encoding)));
		if (encoding_option != "utf-8" && encoding_option != "utf8") {
			throw InvalidInputException("The only supported encoding option is 'UTF8");
		}
	}

	if (!nb::none().is(compression)) {
		if (!nb::isinstance<nb::str>(compression)) {
			throw InvalidInputException("to_csv only accepts 'compression' as a string");
		}
		options["compression"] = {Value(nb::cast<std::string>(compression))};
	}

	ApplyOverwriteOption(options, "to_csv", filename, *rel, overwrite);

	if (!nb::none().is(per_thread_output)) {
		if (!nb::isinstance<nb::bool_>(per_thread_output)) {
			throw InvalidInputException("to_csv only accepts 'per_thread_output' as a boolean");
		}
		options["per_thread_output"] = {Value::BOOLEAN((bool)nb::bool_(per_thread_output))};
	}

	if (!nb::none().is(use_tmp_file)) {
		if (!nb::isinstance<nb::bool_>(use_tmp_file)) {
			throw InvalidInputException("to_csv only accepts 'use_tmp_file' as a boolean");
		}
		options["use_tmp_file"] = {Value::BOOLEAN((bool)nb::bool_(use_tmp_file))};
	}

	if (!nb::none().is(partition_by)) {
		if (!nb::isinstance<nb::list>(partition_by)) {
			throw InvalidInputException("to_csv only accepts 'partition_by' as a list of strings");
		}
		vector<Value> partition_by_values;
		nb::list partition_fields = nb::cast<nb::list>(partition_by);
		for (auto field : partition_fields) {
			if (!nb::isinstance<nb::str>(field)) {
				throw InvalidInputException("to_csv only accepts 'partition_by' as a list of strings");
			}
			partition_by_values.emplace_back(nb::cast<std::string>(nb::str(field)));
		}
		options["partition_by"] = {partition_by_values};
	}

	if (!nb::none().is(write_partition_columns)) {
		if (!nb::isinstance<nb::bool_>(write_partition_columns)) {
			throw InvalidInputException("to_csv only accepts 'write_partition_columns' as a boolean");
		}
		options["write_partition_columns"] = {Value::BOOLEAN((bool)nb::bool_(write_partition_columns))};
	}

	auto write_csv = rel->WriteCSVRel(filename, std::move(options));
	PyExecuteRelation(write_csv);
}

// should this return a rel with the new view?
std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::CreateView(const string &view_name, bool replace) {
	try {
		rel->CreateView(Identifier(view_name), replace);
	} catch (std::exception &error) {
		engine::NameRefusal(*rel->context->GetContext(), error);
		throw;
	}
	return DeriveRelation(rel);
}

static bool IsDescribeStatement(SQLStatement &statement) {
	if (statement.type != StatementType::PRAGMA_STATEMENT) {
		return false;
	}
	auto &pragma_statement = statement.Cast<PragmaStatement>();
	if (pragma_statement.info->name != "show") {
		return false;
	}
	return true;
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Query(const string &view_name, const string &sql_query) {
	try {
		rel->CreateView(Identifier(view_name), /*replace=*/true, /*temporary=*/true);
	} catch (std::exception &error) {
		engine::NameRefusal(*rel->context->GetContext(), error);
		throw;
	}
	auto all_dependencies = rel->GetAllDependencies();

	Parser parser(*rel->context->GetContext());
	parser.ParseQuery(sql_query);
	if (parser.statements.size() != 1) {
		throw InvalidInputException("'DuckDBPyRelation.query' only accepts a single statement");
	}
	auto &statement = *parser.statements[0];
	if (statement.type == StatementType::SELECT_STATEMENT) {
		auto select_statement = unique_ptr_cast<SQLStatement, SelectStatement>(std::move(parser.statements[0]));
		auto query_relation =
		    make_shared_ptr<QueryRelation>(rel->context->GetContext(), std::move(select_statement),
		                                   "query_relation_" + StringUtil::GenerateRandomName(16), sql_query);
		return DeriveRelation(std::move(query_relation));
	} else if (IsDescribeStatement(statement)) {
		auto query = PragmaShow(view_name);
		return Query(view_name, query);
	}
	{
		D_ASSERT(duckdb::PyUtil::GilCheck());
		nb::gil_scoped_release release;
		// Execute it anyways, for creation/altering statements
		// We only care that it succeeds, we can't store the result
		engine::Session(rel->context->GetContext())
		    .Run(std::move(parser.statements[0]), DuckDBPyConnection::CheckSignals);
	}
	return nullptr;
}

void DuckDBPyRelation::InsertInto(const string &table) {
	auto parsed_info = QualifiedName::Parse(table);
	auto insert = rel->InsertRel(parsed_info.Catalog(), parsed_info.Schema(), parsed_info.Name());
	PyExecuteRelation(insert);
}

void DuckDBPyRelation::Update(const nb::object &set_p, const nb::object &where) {
	unique_ptr<ParsedExpression> condition;
	if (!nb::none().is(where)) {
		auto py_expr = DuckDBPyExpression::ToExpression(where);
		condition = py_expr->GetExpression().Copy();
	}

	if (!duckdb::PyUtil::IsDictLike(set_p)) {
		throw InvalidInputException("Please provide 'set' as a dictionary of column name to Expression");
	}

	vector<Identifier> names_;
	vector<unique_ptr<ParsedExpression>> expressions;

	nb::dict set = nb::cast<nb::dict>(set_p);
	auto arg_count = set.size();
	if (arg_count == 0) {
		throw InvalidInputException("Please provide at least one set expression");
	}

	for (auto item : set) {
		nb::object item_key = nb::cast<nb::object>(item.first);
		nb::object item_value = nb::cast<nb::object>(item.second);

		if (!nb::isinstance<nb::str>(item_key)) {
			throw InvalidInputException("Please provide the column name as the key of the dictionary");
		}
		std::unique_ptr<DuckDBPyExpression> py_expr;
		if (!DuckDBPyExpression::TryToExpression(item_value, py_expr)) {
			string actual_type = nb::cast<std::string>(nb::str((item_value).type()));
			throw InvalidInputException("Please provide an object of type Expression as the value, not %s",
			                            actual_type);
		}
		names_.push_back(Identifier(nb::cast<std::string>(nb::str(item_key))));
		expressions.push_back(py_expr->GetExpression().Copy());
	}

	if (rel->type != RelationType::TABLE_RELATION) {
		throw InvalidInputException("UPDATE can only be used on base tables");
	}
	auto &qualified_name = rel->Cast<TableRelation>().description->qualified_name;
	auto update = make_shared_ptr<UpdateRelation>(rel->context, std::move(condition), qualified_name.Catalog(),
	                                              qualified_name.Schema(), qualified_name.Name(), std::move(names_),
	                                              std::move(expressions));
	PyExecuteRelation(update);
}

void DuckDBPyRelation::Insert(const nb::object &params) const {
	if (this->rel->type != RelationType::TABLE_RELATION) {
		throw InvalidInputException("'DuckDBPyRelation.insert' can only be used on a table relation");
	}
	vector<vector<Value>> values {
	    DuckDBPyConnection::TransformPythonParamList(*this->rel->context->GetContext(), params)};

	D_ASSERT(duckdb::PyUtil::GilCheck());
	nb::gil_scoped_release release;
	engine::ConnectionClaim claim(*rel->context->GetContext());
	try {
		rel->Insert(values);
	} catch (std::exception &error) {
		engine::NameRefusal(*rel->context->GetContext(), error);
		throw;
	}
}

void DuckDBPyRelation::Create(const string &table) {
	auto parsed_info = QualifiedName::Parse(table);
	auto create = rel->CreateRel(parsed_info.Schema(), parsed_info.Name(), false);
	PyExecuteRelation(create);
}

std::unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Map(nb::callable fun, Optional<nb::object> schema) {
	auto info = make_shared_ptr<MapFunctionInfo>(std::move(fun), std::move(schema));
	auto context = rel->context->GetContext();
	shared_ptr<Relation> mapped;
	{
		engine::ConnectionClaim claim(*context);
		mapped = make_shared_ptr<TableFunctionRelation>(context, "python_map_function", vector<Value>(),
		                                                named_parameter_map_t(), rel, true, std::move(info));
	}
	return DeriveRelation(std::move(mapped));
}

string DuckDBPyRelation::ToStringInternal(const BoxRendererConfig &config) {
	auto limit = Limit(config.limit, 0);
	auto res = limit->ExecuteInternal();
	auto context = ClientBoxRendererContext(*rel->context->GetContext());
	return res->ToBox(context, config);
}

string DuckDBPyRelation::ToString() {
	BoxRendererConfig config;
	config.limit = 10000;
	if (DuckDBPyConnection::IsJupyter()) {
		config.max_width = 10000;
	}
	return ToStringInternal(config);
}

static idx_t IndexFromPyInt(const nb::object &object) {
	auto index = nb::cast<idx_t>(object);
	return index;
}

void DuckDBPyRelation::Print(const Optional<nb::int_> &max_width, const Optional<nb::int_> &max_rows,
                             const Optional<nb::int_> &max_col_width, const Optional<nb::str> &null_value,
                             const nb::object &render_mode) {
	BoxRendererConfig config;
	config.limit = 10000;
	if (DuckDBPyConnection::IsJupyter()) {
		config.max_width = 10000;
	}

	if (!nb::none().is(max_width)) {
		config.max_width = IndexFromPyInt(max_width);
	}
	if (!nb::none().is(max_rows)) {
		config.max_rows = IndexFromPyInt(max_rows);
	}
	if (!nb::none().is(max_col_width)) {
		config.max_col_width = IndexFromPyInt(max_col_width);
	}
	if (!nb::none().is(null_value)) {
		config.null_value = nb::cast<std::string>(null_value);
	}
	if (!nb::none().is(render_mode)) {
		if (!nb::try_cast(render_mode, config.render_mode)) {
			throw InvalidInputException("'render_mode' accepts either a string, RenderMode or int value");
		}
	}

	auto str_repr = ToStringInternal(config);
	nb::print(nb::str(str_repr.c_str(), str_repr.size()));
}

static ProfilerPrintFormat GetExplainFormat(ExplainType type) {
	if (DuckDBPyConnection::IsJupyter() && type != ExplainType::EXPLAIN_ANALYZE) {
		return ProfilerPrintFormat::HTML();
	}
	return ProfilerPrintFormat::Default();
}

static void DisplayHTML(const string &html) {
	nb::gil_scoped_acquire gil;
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto html_attr = import_cache.IPython.display.HTML();
	auto html_object = html_attr(nb::str(html.c_str(), html.size()));
	auto display_attr = import_cache.IPython.display.display();
	display_attr(html_object);
}

string DuckDBPyRelation::Explain(ExplainType type, const string &format) {
	// An empty format means "auto": the default format, or HTML when running under Jupyter.
	const bool auto_format = format.empty();
	auto explain_format = auto_format ? GetExplainFormat(type) : ProfilerPrintFormat(format);
	unique_ptr<ColumnDataCollection> collection;
	{
		D_ASSERT(duckdb::PyUtil::GilCheck());
		nb::gil_scoped_release release;
		auto explain = make_shared_ptr<ExplainRelation>(rel, type, explain_format);
		collection = PySubmitRelation(explain, engine::Format::Chunks())->TakeChunks(DuckDBPyConnection::CheckSignals);
	}
	auto &coll = *collection;
	// Only the implicit Jupyter path renders HTML inline; an explicitly requested format always returns a string.
	const bool jupyter_html =
	    auto_format && explain_format == ProfilerPrintFormat::HTML() && DuckDBPyConnection::IsJupyter();
	if (!jupyter_html) {
		string result_;
		for (auto &row : coll.Rows()) {
			// Skip the first column because it just contains 'physical plan'
			for (idx_t col_idx = 1; col_idx < coll.ColumnCount(); col_idx++) {
				if (col_idx > 1) {
					result_ += "\t";
				}
				auto val = row.GetValue(col_idx);
				result_ += val.IsNull() ? "NULL" : StringUtil::Replace(val.ToString(), string("\0", 1), "\\0");
			}
			result_ += "\n";
		}
		return result_;
	}

	for (auto &row : coll.Rows()) {
		DisplayHTML(row.GetValue(1).GetValue<string>());
	}

	const string tree_resize_script = R"(
<script>
function toggleDisplay(button) {
    const parentLi = button.closest('li');
    const nestedUl = parentLi.querySelector('ul');
    if (nestedUl) {
        const currentDisplay = getComputedStyle(nestedUl).getPropertyValue('display');
        if (currentDisplay === 'none') {
            nestedUl.classList.toggle('hidden');
            button.textContent = '-';
        } else {
            nestedUl.classList.toggle('hidden');
            button.textContent = '+';
        }
    }
}

function updateTreeHeight(tfTree) {
	if (!tfTree) {
		return;
	}

	const closestElement = tfTree.closest('.lm-Widget.jp-OutputArea.jp-Cell-outputArea');
	if (!closestElement) {
		return;
	}

	console.log(closestElement);

	const height = getComputedStyle(closestElement).getPropertyValue('height');
	tfTree.style.height = height;
}

function resizeTFTree() {
	const tfTrees = document.querySelectorAll('.tf-tree');
	tfTrees.forEach(tfTree => {
		console.log(tfTree);
		if (tfTree) {
			const jupyterViewPort = tfTree.closest('.lm-Widget.jp-OutputArea.jp-Cell-outputArea');
			console.log(jupyterViewPort);
			if (jupyterViewPort) {
				const resizeObserver = new ResizeObserver(() => {
					updateTreeHeight(tfTree);
				});
				resizeObserver.observe(jupyterViewPort);
			}
		}
	});
}

resizeTFTree();

</script>
	)";
	DisplayHTML(tree_resize_script);
	return "";
}

// TODO: RelationType to a python enum
nb::str DuckDBPyRelation::Type() {
	auto type_str = RelationTypeToString(rel->type);
	return nb::str(type_str.c_str(), type_str.size());
}

nb::list DuckDBPyRelation::Columns() {
	nb::list res;
	for (auto &col : rel->Columns()) {
		res.append(col.Name());
	}
	return res;
}

nb::list DuckDBPyRelation::ColumnTypes() {
	nb::list res;
	for (auto &col : rel->Columns()) {
		res.append(DuckDBPyType(col.Type()));
	}
	return res;
}

bool DuckDBPyRelation::IsRelation(const nb::object &object) {
	return nb::isinstance<DuckDBPyRelation>(object);
}

} // namespace duckdb

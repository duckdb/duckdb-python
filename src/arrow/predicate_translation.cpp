#include "duckdb_python/arrow/predicate_translation.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/sql_identifier.hpp"
#include "duckdb/common/types/date.hpp"
#include "duckdb/common/types/time.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/function/scalar/comparison_functions.hpp"
#include "duckdb/function/scalar/struct_functions.hpp"
#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb_python/engine/connection.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyutil.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {

namespace {

bool SameFieldNames(const ArrowSchema &field, const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return true;
	}
	const ArrowSchema &node = field.dictionary ? *field.dictionary : field;
	const auto child_count = StructType::GetChildCount(type);
	if (!node.format || string(node.format) != "+s" || node.n_children < 0 ||
	    NumericCast<idx_t>(node.n_children) != child_count) {
		return false;
	}
	unordered_set<string> seen;
	for (idx_t i = 0; i < child_count; i++) {
		const auto &child = *node.children[i];
		const string name = child.name ? child.name : "";
		if (name.empty() || !seen.insert(name).second ||
		    name != StructType::GetChildName(type, i).GetIdentifierName() ||
		    !SameFieldNames(child, StructType::GetChildType(type, i))) {
			return false;
		}
	}
	return true;
}

vector<bool> UniqueNames(const vector<PredicateColumn> &columns) {
	unordered_map<string, idx_t> counts;
	for (auto &column : columns) {
		counts[column.name]++;
	}
	vector<bool> result;
	for (auto &column : columns) {
		result.push_back(counts[column.name] == 1);
	}
	return result;
}

const char *ComparisonOperator(ExpressionType type) {
	switch (type) {
	case ExpressionType::COMPARE_EQUAL:
		return "=";
	case ExpressionType::COMPARE_NOTEQUAL:
		return "!=";
	case ExpressionType::COMPARE_LESSTHAN:
		return "<";
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return "<=";
	case ExpressionType::COMPARE_GREATERTHAN:
		return ">";
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return ">=";
	default:
		throw PredicateRefused();
	}
}

nb::str PythonText(const string &text) {
	return nb::str(text.c_str(), text.size());
}

bool PythonYear(int32_t year) {
	return year >= 1 && year <= 9999;
}

timestamp_t Microseconds(const Value &value) {
	auto stored = value.GetValueUnsafe<timestamp_t>();
	if (!Value::IsFinite(stored)) {
		throw PredicateRefused();
	}
	switch (value.type().id()) {
	case LogicalTypeId::TIMESTAMP_SEC:
		return Timestamp::FromEpochSeconds(stored.value);
	case LogicalTypeId::TIMESTAMP_MS:
		return Timestamp::FromEpochMs(stored.value);
	default:
		return stored;
	}
}

using ScalarFunctionPointer = void (*)(DataChunk &, ExpressionState &, Vector &);

//! The bind and execution callbacks of one built in struct field accessor overload.
struct StructAccessorCallbacks {
	bind_scalar_function_t bind;
	ScalarFunctionPointer execute;
};

ScalarFunctionPointer ExecutionPointer(const ScalarFunction &function) {
	auto target = function.GetCallbacks().function.target<ScalarFunctionPointer>();
	return target ? *target : nullptr;
}

//! The field index of a struct field access, or false when the call is anything else. Another function may carry an
//! accessor's name, and a native one may even reuse its binder, so the call is recognized by both the bind and the
//! execution callback of the definition it was bound from, read from the engine's own accessor definitions.
bool StructFieldIndex(const BoundFunctionExpression &function, idx_t &index) {
	static const vector<StructAccessorCallbacks> built_in = [] {
		vector<StructAccessorCallbacks> accessors;
		auto struct_extract = StructExtractFun::GetFunctions();
		for (idx_t i = 0; i < struct_extract.Size(); i++) {
			auto &overload = *struct_extract.GetFunctionByOffset(i);
			accessors.push_back({overload.GetBindCallback(), ExecutionPointer(overload)});
		}
		auto struct_extract_at = StructExtractAtFun::GetFunction();
		accessors.push_back({struct_extract_at.GetBindCallback(), ExecutionPointer(struct_extract_at)});
		return accessors;
	}();
	auto &name = function.Function().GetName();
	auto &definition = function.Function().GetDefinition();
	if ((name != StructExtractFun::Name && name != StructExtractAtFun::Name) || !definition ||
	    function.GetChildren().size() != 2 ||
	    function.GetChildren()[0]->GetReturnType().id() != LogicalTypeId::STRUCT) {
		return false;
	}
	const auto bind = definition->GetBindCallback();
	const auto execute = ExecutionPointer(*definition);
	auto matches = [&](const StructAccessorCallbacks &accessor) {
		return accessor.bind && accessor.execute && accessor.bind == bind && accessor.execute == execute;
	};
	if (std::none_of(built_in.begin(), built_in.end(), matches)) {
		return false;
	}
	auto bind_data = dynamic_cast<const StructExtractBindData *>(function.BindInfo().get());
	if (!bind_data) {
		return false;
	}
	index = bind_data->index;
	return true;
}

class PredicateWalker {
public:
	explicit PredicateWalker(const std::function<PredicateColumn(idx_t)> &resolve)
	    : resolve(resolve), nodes(nb::borrow<nb::module_>(DuckDBPyConnection::ArrowSourcesNodes())) {
	}

	nb::object Predicate(const Expression &expression) {
		switch (expression.GetExpressionClass()) {
		case ExpressionClass::BOUND_CONJUNCTION:
			return Conjunction(expression.Cast<BoundConjunctionExpression>());
		case ExpressionClass::BOUND_OPERATOR:
			return Operator(expression.Cast<BoundOperatorExpression>());
		case ExpressionClass::BOUND_FUNCTION:
			return Function(expression.Cast<BoundFunctionExpression>());
		default:
			throw PredicateRefused();
		}
	}

private:
	struct Column {
		nb::object node;
		LogicalType type;
	};

	nb::object Conjunction(const BoundConjunctionExpression &conjunction) {
		const char *op;
		switch (conjunction.GetExpressionType()) {
		case ExpressionType::CONJUNCTION_AND:
			op = "AND";
			break;
		case ExpressionType::CONJUNCTION_OR:
			op = "OR";
			break;
		default:
			throw PredicateRefused();
		}
		auto &children = conjunction.GetChildren();
		if (children.empty()) {
			throw PredicateRefused();
		}
		nb::object folded = Predicate(*children[0]);
		for (idx_t i = 1; i < children.size(); i++) {
			folded = nodes.attr("Binary")(op, folded, Predicate(*children[i]));
		}
		return folded;
	}

	nb::object Operator(const BoundOperatorExpression &operation) {
		auto &children = operation.GetChildren();
		switch (operation.GetExpressionType()) {
		case ExpressionType::OPERATOR_NOT:
			if (children.size() != 1) {
				throw PredicateRefused();
			}
			return nodes.attr("Unary")("NOT", Predicate(*children[0]));
		case ExpressionType::OPERATOR_IS_NULL:
		case ExpressionType::OPERATOR_IS_NOT_NULL: {
			if (children.size() != 1) {
				throw PredicateRefused();
			}
			const char *op =
			    operation.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ? "IS NULL" : "IS NOT NULL";
			return nodes.attr("Postfix")(op, ColumnOf(*children[0]).node);
		}
		case ExpressionType::COMPARE_IN:
		case ExpressionType::COMPARE_NOT_IN: {
			if (children.size() < 2) {
				throw PredicateRefused();
			}
			auto column = ColumnOf(*children[0]);
			nb::list candidates;
			for (idx_t i = 1; i < children.size(); i++) {
				candidates.append(Constant(*children[i], column.type));
			}
			auto membership = nodes.attr("In")(column.node, candidates);
			if (operation.GetExpressionType() == ExpressionType::COMPARE_NOT_IN) {
				return nodes.attr("Unary")("NOT", membership);
			}
			return membership;
		}
		default:
			throw PredicateRefused();
		}
	}

	nb::object Function(const BoundFunctionExpression &function) {
		auto &children = function.GetChildren();
		if (BoundComparisonExpression::IsComparison(function.GetExpressionType()) && children.size() == 2) {
			return Comparison(function.GetExpressionType(), *children[0], *children[1]);
		}
		if (function.GetExpressionType() == ExpressionType::COMPARE_BETWEEN &&
		    function.Function().GetName() == BetweenFun::Name && BoundBetweenExpression::HasValidBindData(function) &&
		    children.size() == 3) {
			return Between(function);
		}
		throw PredicateRefused();
	}

	nb::object Comparison(ExpressionType type, const Expression &left, const Expression &right) {
		if (left.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT &&
		    right.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
			return Comparison(FlipComparisonExpression(type), right, left);
		}
		const char *op = ComparisonOperator(type);
		auto column = ColumnOf(left);
		return nodes.attr("Binary")(op, column.node, Constant(right, column.type));
	}

	nb::object Between(const BoundFunctionExpression &between) {
		const char *lower_op = ComparisonOperator(BoundBetweenExpression::LowerComparisonType(between));
		const char *upper_op = ComparisonOperator(BoundBetweenExpression::UpperComparisonType(between));
		auto column = ColumnOf(BoundBetweenExpression::Input(between));
		auto lower = Constant(BoundBetweenExpression::LowerBound(between), column.type);
		auto upper = Constant(BoundBetweenExpression::UpperBound(between), column.type);
		nb::object binary = nodes.attr("Binary");
		return binary("AND", binary(lower_op, column.node, lower), binary(upper_op, column.node, upper));
	}

	Column ColumnOf(const Expression &expression) {
		vector<idx_t> path;
		const Expression *current = &expression;
		while (current->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
			auto &function = current->Cast<BoundFunctionExpression>();
			idx_t index;
			if (!StructFieldIndex(function, index)) {
				throw PredicateRefused();
			}
			path.push_back(index);
			current = function.GetChildren()[0].get();
		}
		if (current->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			throw PredicateRefused();
		}
		auto &reference = current->Cast<BoundColumnRefExpression>();
		if (reference.Depth() != 0) {
			throw PredicateRefused();
		}
		auto column = resolve(reference.Binding().column_index.GetIndex());
		if (column.type != reference.GetReturnType()) {
			throw PredicateRefused();
		}
		nb::list parts;
		parts.append(PythonText(column.name));
		auto type = column.type;
		for (auto step = path.rbegin(); step != path.rend(); ++step) {
			if (type.id() != LogicalTypeId::STRUCT || *step >= StructType::GetChildCount(type)) {
				throw PredicateRefused();
			}
			auto &name = StructType::GetChildName(type, *step).GetIdentifierName();
			if (name.empty()) {
				throw PredicateRefused();
			}
			parts.append(PythonText(name));
			type = StructType::GetChildType(type, *step);
		}
		if (type != expression.GetReturnType()) {
			throw PredicateRefused();
		}
		return Column {nodes.attr("Col")(nb::tuple(parts)), type};
	}

	nb::object Constant(const Expression &expression, const LogicalType &column_type) {
		if (expression.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
			throw PredicateRefused();
		}
		auto &value = expression.Cast<BoundConstantExpression>().GetValue();
		// An aliased or extension type, such as JSON, may compare differently from its storage type in the source.
		if (value.IsNull() || value.type() != column_type || column_type.HasAlias() || column_type.HasExtensionInfo()) {
			throw PredicateRefused();
		}
		nb::object python;
		try {
			python = PythonValue(value);
		} catch (nb::python_error &) {
			throw PredicateRefused();
		} catch (Exception &) {
			throw PredicateRefused();
		}
		return nodes.attr("Lit")(python, PythonText(column_type.ToString()));
	}

	//! Only types whose every value a Python value holds exactly, and that a source can compare as the engine does.
	static nb::object PythonValue(const Value &value) {
		switch (value.type().id()) {
		case LogicalTypeId::BOOLEAN:
			return nb::bool_(value.GetValue<bool>());
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT:
			return nb::int_(value.GetValue<int64_t>());
		case LogicalTypeId::UTINYINT:
		case LogicalTypeId::USMALLINT:
		case LogicalTypeId::UINTEGER:
		case LogicalTypeId::UBIGINT:
			return nb::int_(value.GetValue<uint64_t>());
		case LogicalTypeId::HUGEINT:
		case LogicalTypeId::UHUGEINT: {
			auto digits = value.ToString();
			auto number = PyLong_FromString(digits.c_str(), nullptr, 10);
			if (!number) {
				throw nb::python_error();
			}
			return nb::steal<nb::object>(number);
		}
		case LogicalTypeId::FLOAT:
		case LogicalTypeId::DOUBLE: {
			const double number =
			    value.type().id() == LogicalTypeId::FLOAT ? value.GetValue<float>() : value.GetValue<double>();
			if (!std::isfinite(number)) {
				throw PredicateRefused();
			}
			return nb::float_(number);
		}
		case LogicalTypeId::DECIMAL:
			return nb::module_::import_("decimal").attr("Decimal")(PythonText(value.ToString()));
		case LogicalTypeId::VARCHAR:
			return PythonText(StringValue::Get(value));
		case LogicalTypeId::BLOB: {
			auto &blob = StringValue::Get(value);
			return nb::bytes(blob.data(), blob.size());
		}
		case LogicalTypeId::DATE: {
			auto date = value.GetValueUnsafe<date_t>();
			if (!Value::IsFinite(date)) {
				throw PredicateRefused();
			}
			int32_t year, month, day;
			Date::Convert(date, year, month, day);
			if (!PythonYear(year)) {
				throw PredicateRefused();
			}
			return nb::module_::import_("datetime").attr("date")(year, month, day);
		}
		case LogicalTypeId::TIME: {
			int32_t hour, minute, second, micros;
			Time::Convert(value.GetValueUnsafe<dtime_t>(), hour, minute, second, micros);
			if (hour > 23) {
				throw PredicateRefused();
			}
			return nb::module_::import_("datetime").attr("time")(hour, minute, second, micros);
		}
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_SEC:
		case LogicalTypeId::TIMESTAMP_MS:
		case LogicalTypeId::TIMESTAMP_TZ: {
			auto timestamp = Microseconds(value);
			date_t date;
			dtime_t time;
			Timestamp::Convert(timestamp, date, time);
			int32_t year, month, day, hour, minute, second, micros;
			Date::Convert(date, year, month, day);
			if (!PythonYear(year)) {
				throw PredicateRefused();
			}
			if (value.type().id() == LogicalTypeId::TIMESTAMP_TZ) {
				return nb::int_(timestamp.value);
			}
			Time::Convert(time, hour, minute, second, micros);
			return nb::module_::import_("datetime").attr("datetime")(year, month, day, hour, minute, second, micros);
		}
		default:
			throw PredicateRefused();
		}
	}

	const std::function<PredicateColumn(idx_t)> &resolve;
	nb::module_ nodes;
};

struct PredicateProbe : public TableFunctionInfo {
	PredicateProbe(const PredicateColumns &columns, nb::list &entries) : columns(columns), entries(entries) {
	}

	const PredicateColumns &columns;
	nb::list &entries;
	unordered_set<string> offered;
};

void CaptureOfferedPredicates(ClientContext &context, LogicalGet &get, FunctionData *,
                              vector<unique_ptr<Expression>> &filters) {
	engine::CallbackScope callback(context);
	auto &probe = get.function.function_info->Cast<PredicateProbe>();
	auto &column_ids = get.GetColumnIds();
	auto resolve = [&](idx_t binding_index) {
		if (binding_index >= column_ids.size()) {
			throw PredicateRefused();
		}
		return probe.columns.Resolve(column_ids[binding_index].GetPrimaryIndex());
	};
	for (auto &filter : filters) {
		auto text = filter->ToString();
		// Later optimizer passes offer again the filters that stayed above the scan.
		if (!probe.offered.insert(text).second) {
			continue;
		}
		nb::object node = nb::none();
		try {
			node = TranslatePredicate(*filter, resolve);
		} catch (PredicateRefused &) {
		}
		probe.entries.append(nb::make_tuple(PythonText(text), node));
	}
}

struct DropTable {
	Connection &connection;
	string table;

	~DropTable() {
		try {
			connection.Query("DROP TABLE IF EXISTS temp.main." + table);
		} catch (...) {
		}
	}
};

LogicalGet &OnlyGet(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		return op.Cast<LogicalGet>();
	}
	if (op.children.size() != 1) {
		throw InvalidInputException("The predicate must not change the shape of the plan");
	}
	return OnlyGet(*op.children[0]);
}

} // namespace

PredicateColumns::PredicateColumns(const vector<string> &names, const vector<LogicalType> &types) {
	D_ASSERT(names.size() == types.size());
	for (idx_t i = 0; i < names.size(); i++) {
		columns.push_back(PredicateColumn {names[i], types[i]});
	}
	nameable = UniqueNames(columns);
}

PredicateColumns::PredicateColumns(const ArrowSchema &schema, const vector<LogicalType> &types) {
	if (schema.n_children < 0 || NumericCast<idx_t>(schema.n_children) != types.size()) {
		throw InvalidInputException("The Arrow schema has %d columns where the scan declares %d", schema.n_children,
		                            types.size());
	}
	for (idx_t i = 0; i < types.size(); i++) {
		const auto &child = *schema.children[i];
		columns.push_back(PredicateColumn {child.name ? child.name : "", types[i]});
	}
	nameable = UniqueNames(columns);
	for (idx_t i = 0; i < types.size(); i++) {
		nameable[i] = nameable[i] && SameFieldNames(*schema.children[i], types[i]);
	}
}

PredicateColumn PredicateColumns::Resolve(idx_t declared_index) const {
	if (declared_index >= columns.size() || !nameable[declared_index]) {
		throw PredicateRefused();
	}
	return columns[declared_index];
}

nb::object TranslatePredicate(const Expression &predicate,
                              const std::function<PredicateColumn(idx_t column_binding_index)> &resolve) {
	DUCKDB_PY_ASSERT_GIL_HELD();
	return PredicateWalker(resolve).Predicate(predicate);
}

nb::list TranslatePredicateForTesting(DuckDBPyConnection &py_connection, const string &predicate,
                                      const nb::object &columns, const nb::object &schema, const string &select) {
	DuckDBPyConnection::ConnectionLockGuard lock(py_connection);
	auto &connection = py_connection.con.GetConnection();
	auto &context = *connection.context;
	vector<string> names;
	vector<string> type_texts;
	nb::object capsule;
	const ArrowSchema *arrow_schema = nullptr;
	{
		engine::CallbackScope callback(context);
		if (schema.is_none() == columns.is_none()) {
			throw InvalidInputException("Give either the columns or an Arrow schema");
		}
		if (!schema.is_none()) {
			capsule = schema.attr("__arrow_c_schema__")();
			arrow_schema = static_cast<const ArrowSchema *>(PyCapsule_GetPointer(capsule.ptr(), "arrow_schema"));
			if (!arrow_schema) {
				throw nb::python_error();
			}
		} else {
			for (auto item : nb::cast<nb::dict>(columns)) {
				names.push_back(nb::cast<string>(item.first));
				type_texts.push_back(nb::cast<string>(nb::str(item.second)));
			}
		}
	}
	auto uuid = UUID::ToString(UUID::GenerateRandomUUID());
	std::replace(uuid.begin(), uuid.end(), '-', '_');
	const string table = "__duckdb_predicate_probe_" + uuid;
	DropTable drop {connection, SQLIdentifier::ToString(table)};
	auto statements = connection.ExtractStatements("SELECT " + select + " FROM temp.main." +
	                                               SQLIdentifier::ToString(table) + " WHERE " + predicate);
	if (statements.size() != 1) {
		throw InvalidInputException("The predicate must be one expression");
	}
	nb::list entries;
	auto probe_plan = [&]() {
		vector<LogicalType> types;
		if (arrow_schema) {
			ArrowTableSchema arrow_table;
			ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table, *arrow_schema);
			names = arrow_table.GetNames();
			types = arrow_table.GetTypes();
		} else {
			for (auto &type_text : type_texts) {
				types.push_back(TransformStringToLogicalType(type_text, context));
			}
		}
		if (names.empty()) {
			throw InvalidInputException("A predicate needs at least one column");
		}
		auto info = make_uniq<CreateTableInfo>(
		    QualifiedName(Identifier::TempCatalog(), Identifier::DefaultSchema(), Identifier(table)));
		info->temporary = true;
		for (idx_t i = 0; i < names.size(); i++) {
			info->columns.AddColumn(ColumnDefinition(Identifier(names[i]), types[i]));
		}
		Catalog::GetCatalog(context, Identifier::TempCatalog()).CreateTable(context, std::move(info));

		Planner planner(context);
		planner.CreatePlan(std::move(statements[0]));
		if (!planner.plan) {
			throw InvalidInputException("The predicate's parameter types could not be resolved");
		}
		auto &get = OnlyGet(*planner.plan);
		auto resolver = arrow_schema ? PredicateColumns(*arrow_schema, get.returned_types)
		                             : PredicateColumns(names, get.returned_types);
		get.function.filter_pushdown = false;
		get.function.pushdown_complex_filter = CaptureOfferedPredicates;
		get.function.function_info = make_shared_ptr<PredicateProbe>(resolver, entries);
		Optimizer optimizer(*planner.binder, context);
		optimizer.Optimize(std::move(planner.plan));
	};
	context.RunFunctionInTransaction(probe_plan);
	return entries;
}

} // namespace duckdb

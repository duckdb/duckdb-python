#include "duckdb_python/engine/connection.hpp"

#include "duckdb/main/statement_iterator.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/planner/statement_preprocessor.hpp"
#include "duckdb/main/valid_checker.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/tableref/column_data_ref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/transaction/meta_transaction.hpp"

namespace duckdb {
namespace engine {

shared_ptr<ConnectionState> ConnectionState::Get(ClientContext &context) {
	return context.registered_state->GetOrCreate<ConnectionState>("python_connection_state");
}

static constexpr const char *BUSY_ERROR =
    "connection is busy with another call; it refuses instead of waiting, so use a separate connection from cursor()";
static constexpr const char *CALLBACK_ERROR =
    "code running inside a query cannot use that query's connection; use a separate connection from cursor()";

static constexpr const char *UNFINISHED_EXECUTE_ERROR =
    "connection has an unfinished result from execute(); call complete() or abort(), or fetch until a fetch returns "
    "no rows, before other work on this connection, or use a separate connection from cursor()";

static thread_local CallbackScope *innermost_callback = nullptr;

CallbackScope::CallbackScope(const ClientContext &context_p) : context(context_p), outer(innermost_callback) {
	innermost_callback = this;
}

CallbackScope::~CallbackScope() {
	innermost_callback = outer;
}

bool CallbackScope::Inside(const ClientContext &context) {
	for (auto scope = innermost_callback; scope; scope = scope->outer) {
		if (&scope->context == &context) {
			return true;
		}
	}
	return false;
}

ConnectionClaim::ConnectionClaim(ConnectionState &state_p, const ClientContext &context) : state(state_p) {
	if (CallbackScope::Inside(context)) {
		throw ResourceInUseException(CALLBACK_ERROR);
	}
	if (state.busy.exchange(true)) {
		throw ResourceInUseException(BUSY_ERROR);
	}
}

ConnectionClaim::ConnectionClaim(ClientContext &context) : ConnectionClaim(*ConnectionState::Get(context), context) {
}

ConnectionClaim::~ConnectionClaim() {
	state.busy = false;
}

void ConnectionState::ThrowIfBusy(const ClientContext &context) const {
	if (CallbackScope::Inside(context)) {
		throw ResourceInUseException(CALLBACK_ERROR);
	}
	if (busy) {
		throw ResourceInUseException(BUSY_ERROR);
	}
}

void ConnectionState::BeginResolving() {
	lock_guard<mutex> guard(lock);
	resolving = make_uniq<ResolvedTables>();
	resolving_thread = std::this_thread::get_id();
}

ResolvedTables ConnectionState::EndResolving() {
	lock_guard<mutex> guard(lock);
	ResolvedTables tables;
	if (resolving) {
		tables = std::move(*resolving);
		resolving.reset();
	}
	return tables;
}

void ConnectionState::RecordResolved(const string &name, TableRef &table) {
	lock_guard<mutex> guard(lock);
	if (resolving && resolving_thread == std::this_thread::get_id() && resolving->find(name) == resolving->end()) {
		(*resolving)[name] = table.Copy();
	}
}

void ConnectionState::Pin(optional_ptr<const ResolvedTables> tables, bool exclusive) {
	lock_guard<mutex> guard(lock);
	pinned = tables;
	pinned_exclusively = exclusive;
	pinned_thread = std::this_thread::get_id();
}

bool ConnectionState::PinnedExclusively() {
	lock_guard<mutex> guard(lock);
	return pinned && pinned_exclusively && pinned_thread == std::this_thread::get_id();
}

unique_ptr<TableRef> ConnectionState::RecallResolved(const string &name) {
	lock_guard<mutex> guard(lock);
	if (!pinned || pinned_thread != std::this_thread::get_id()) {
		return nullptr;
	}
	auto entry = pinned->find(name);
	if (entry == pinned->end()) {
		return nullptr;
	}
	return entry->second->Copy();
}

static constexpr const char *SOURCE_NAME_PREFIX = "__duckdb_source_";

shared_ptr<SourceRegistry> SourceRegistry::Get(ClientContext &context) {
	return ObjectCache::GetObjectCache(context).GetOrCreate<SourceRegistry>(ObjectType());
}

string SourceRegistry::NextName() {
	static const string process_token = StringUtil::GenerateRandomName(12);
	static atomic<idx_t> count {0};
	return SOURCE_NAME_PREFIX + process_token + "_" + std::to_string(++count);
}

bool SourceRegistry::IsSourceName(const string &name) {
	return StringUtil::StartsWith(name, SOURCE_NAME_PREFIX);
}

void SourceRegistry::Register(const string &name, unique_ptr<TableRef> table, const string &subject) {
	lock_guard<mutex> guard(lock);
	if (sources.find(name) != sources.end()) {
		throw InternalException("Source name \"%s\" registered twice", name);
	}
	sources[name] = Entry {std::move(table), subject};
}

//! The kind of object a registry entry reads, in the words an error shows a user
static string SourceKind(const TableRef &table) {
	switch (table.type) {
	case TableReferenceType::SUBQUERY:
		return "a relation";
	case TableReferenceType::COLUMN_DATA:
		return "a materialized result";
	case TableReferenceType::TABLE_FUNCTION: {
		auto &function = table.Cast<TableFunctionRef>().function;
		if (function && function->GetExpressionType() == ExpressionType::FUNCTION) {
			auto &name = function->Cast<FunctionExpression>().FunctionName().GetIdentifierName();
			if (name == "pandas_scan") {
				return "a pandas DataFrame or NumPy array";
			}
			if (name == "python_arrow_scan" || name == "scan_arrow_ipc") {
				return "an Arrow object";
			}
			if (name == "python_map_function") {
				return "a map() result";
			}
		}
		return "a Python object";
	}
	default:
		return "a Python object";
	}
}

unique_ptr<TableRef> SourceRegistry::Unregister(const string &name) {
	lock_guard<mutex> guard(lock);
	auto entry = sources.find(name);
	if (entry == sources.end()) {
		return nullptr;
	}
	auto table = std::move(entry->second.table);
	if (released.emplace(name, Released {SourceKind(*table), entry->second.subject}).second) {
		released_order.push_back(name);
		if (released_order.size() > RELEASED_MEMORY) {
			released.erase(released_order.front());
			released_order.pop_front();
		}
	}
	sources.erase(entry);
	return table;
}

unique_ptr<TableRef> SourceRegistry::Lookup(const string &name) {
	lock_guard<mutex> guard(lock);
	auto entry = sources.find(name);
	if (entry == sources.end()) {
		return nullptr;
	}
	return entry->second.table->Copy();
}

void SourceRegistry::ThrowIfReleased(const string &name) {
	Released gone;
	{
		lock_guard<mutex> guard(lock);
		auto entry = released.find(name);
		if (entry == released.end()) {
			return;
		}
		gone = entry->second;
	}
	auto what = gone.subject.empty()
	                ? StringUtil::Format("%s that was released", gone.kind)
	                : StringUtil::Format("%s registered as \"%s\", which was released", gone.kind, gone.subject);
	throw InvalidInputException(
	    "This query reads %s. A Python object stays in its database's registry only while the relation or view that "
	    "registered it is alive; keep that alive, or register the object again.",
	    what);
}

namespace {

class ResolvingGuard {
public:
	explicit ResolvingGuard(ConnectionState &state) : state(state) {
		state.BeginResolving();
	}
	~ResolvingGuard() {
		if (!ended) {
			state.EndResolving();
		}
	}
	ResolvedTables End() {
		ended = true;
		return state.EndResolving();
	}

private:
	ConnectionState &state;
	bool ended = false;
};

class PinGuard {
public:
	PinGuard(ConnectionState &state, const ResolvedTables &tables, bool exclusive = false) : state(state) {
		state.Pin(&tables, exclusive);
	}
	~PinGuard() {
		state.Pin(nullptr);
	}

private:
	ConnectionState &state;
};

} // namespace

void ConnectionState::SetDbapiResult(const shared_ptr<Result> &result) {
	lock_guard<mutex> guard(lock);
	dbapi_result = result;
}

void ConnectionState::NameRefusal(const ErrorData &error) {
	if (error.Type() != ExceptionType::RESOURCE_IN_USE) {
		return;
	}
	shared_ptr<Result> result;
	{
		lock_guard<mutex> guard(lock);
		result = dbapi_result.lock();
	}
	// One result at a time holds the connection, so a running DB-API result is the one the engine refused for
	if (result && result->Running()) {
		throw ResourceInUseException(UNFINISHED_EXECUTE_ERROR);
	}
}

void VerifyNoOpenResult(ClientContext &context) {
	try {
		context.VerifyNoOpenResult();
	} catch (std::exception &error) {
		NameRefusal(context, error);
		throw;
	}
}

void NameRefusal(ClientContext &context, const std::exception &error) {
	ConnectionState::Get(context)->NameRefusal(ErrorData(error));
}

static void NameRefusal(ConnectionState &state, QueryResult &handle) {
	if (handle.HasError()) {
		state.NameRefusal(handle.GetErrorObject());
	}
}

Session::Session(shared_ptr<ClientContext> context_p)
    : context(std::move(context_p)), state(ConnectionState::Get(*context)), registry(SourceRegistry::Get(*context)) {
}

Prepared Session::Prepare(unique_ptr<SQLStatement> statement) {
	ConnectionClaim claim(*state, *context);
	Prepared prepared;
	{
		ResolvingGuard guard(*state);
		prepared.statement = context->Prepare(std::move(statement));
		prepared.tables = guard.End();
	}
	if (prepared.statement->HasError()) {
		state->NameRefusal(prepared.statement->GetErrorObject());
		prepared.statement->GetErrorObject().Throw();
	}
	return prepared;
}

ConnectionClaim Session::Claim() {
	return ConnectionClaim(*state, *context);
}

shared_ptr<Result> Session::Submit(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
                                   const Format &format) {
	auto claim = Claim();
	return SubmitClaimed(std::move(statement), values, format);
}

shared_ptr<Result> Session::SubmitClaimed(unique_ptr<SQLStatement> statement,
                                          identifier_map_t<BoundParameterData> &values, const Format &format) {
	auto written_type = statement->type;
	registry->CountStatement(written_type);
	auto handle = context->Submit(std::move(statement), values, QueryParameters(format.EngineFormat()));
	NameRefusal(*state, *handle);
	return make_shared_ptr<Result>(std::move(handle), written_type, format);
}

shared_ptr<Result> Session::Submit(Prepared &prepared, identifier_map_t<BoundParameterData> &values,
                                   const Format &format) {
	ConnectionClaim claim(*state, *context);
	registry->CountStatement(prepared.statement->GetStatementType());
	unique_ptr<QueryResult> handle;
	{
		PinGuard guard(*state, prepared.tables);
		handle = prepared.statement->Submit(values, QueryParameters(format.EngineFormat()));
	}
	NameRefusal(*state, *handle);
	return make_shared_ptr<Result>(std::move(handle), prepared.statement->GetStatementType(), format);
}

shared_ptr<Result> Session::Submit(unique_ptr<ColumnDataCollection> collection, const vector<Identifier> &names,
                                   const Format &format) {
	// The binder rejects duplicate column names; the caller keeps the originals
	auto deduplicated_names = names;
	QueryResult::DeduplicateColumns(deduplicated_names);
	auto table_ref = make_uniq<ColumnDataRef>(std::move(collection), std::move(deduplicated_names));
	// Binding asserts on an unset alias
	table_ref->alias = "materialized";
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	select_node->from_table = std::move(table_ref);
	auto select = make_uniq<SelectStatement>();
	select->node = std::move(select_node);
	identifier_map_t<BoundParameterData> no_values;
	return Submit(std::move(select), no_values, format);
}

StatementSignature Session::Bind(unique_ptr<SQLStatement> statement, ResolvedTables &tables) {
	ConnectionClaim claim(*state, *context);
	ResolvingGuard guard(*state);
	auto signature = context->BindStatement(std::move(statement));
	tables = guard.End();
	return signature;
}

StatementSignature Session::Bind(unique_ptr<SQLStatement> statement, ResolvedTables &tables,
                                 const identifier_map_t<BoundParameterData> &values) {
	ConnectionClaim claim(*state, *context);
	ResolvingGuard guard(*state);
	StatementSignature signature;
	ErrorData bind_error;
	context->RunFunctionInTransaction(
	    [&]() {
		    try {
			    auto expanded = StatementPreprocessor(*context).ExpandStatement(std::move(statement));
			    if (expanded.statements.empty()) {
				    signature.properties.return_type = StatementReturnType::NOTHING;
				    return;
			    }
			    if (expanded.statements.size() > 1) {
				    throw InvalidInputException(ClientContext::EXPANDING_PARAMETERS_ERROR);
			    }
			    // The planner types each parameter from its value, as a rebind for the values run with does
			    Planner planner(*context);
			    planner.parameter_data = values;
			    planner.CreatePlan(std::move(expanded.statements[0]));
			    signature.names = planner.names;
			    signature.types = planner.types;
			    signature.properties = std::move(planner.properties);
		    } catch (const std::exception &ex) {
			    ErrorData error(ex);
			    // Binding changes nothing, so a recoverable error leaves the caller's transaction as it was
			    if (Exception::InvalidatesDatabase(error.Type())) {
				    throw;
			    }
			    bind_error = std::move(error);
		    }
	    },
	    false);
	tables = guard.End();
	if (bind_error.HasError()) {
		bind_error.Throw();
	}
	return signature;
}

void Session::Run(unique_ptr<SQLStatement> statement, const InterruptCheck &check) {
	ConnectionClaim claim(*state, *context);
	RunClaimed(std::move(statement), check);
}

void Session::RunClaimed(unique_ptr<SQLStatement> statement, const InterruptCheck &check) {
	registry->CountStatement(statement->type);
	auto handle = context->Submit(std::move(statement), QueryParameters());
	NameRefusal(*state, *handle);
	Complete(*handle, check);
}

void Session::Run(unique_ptr<SQLStatement> statement, const ResolvedTables &tables, const InterruptCheck &check) {
	ConnectionClaim claim(*state, *context);
	registry->CountStatement(statement->type);
	unique_ptr<QueryResult> handle;
	{
		PinGuard guard(*state, tables, true);
		handle = context->Submit(std::move(statement), QueryParameters());
	}
	NameRefusal(*state, *handle);
	Complete(*handle, check);
}

StatementGroup::StatementGroup(Session session_p, const string &sql, bool aborts_transaction_p)
    : session(std::move(session_p)), query(sql), aborts_transaction(aborts_transaction_p),
      iterator(make_uniq<StatementIterator>(session.Context().IterateStatements(sql))) {
}

StatementGroup::StatementGroup(Session session_p, unique_ptr<SQLStatement> statement)
    : session(std::move(session_p)), query(statement->query), upcoming(std::move(statement)) {
}

optional_ptr<const SQLStatement> StatementGroup::First() {
	D_ASSERT(!kept);
	if (iterator && !upcoming) {
		auto claim = session.Claim();
		try {
			upcoming = iterator->GetStatement();
		} catch (std::exception &ex) {
			Fail(ErrorData(ex));
		}
	}
	return upcoming.get();
}

unique_ptr<SQLStatement> StatementGroup::Next() {
	if (upcoming || !iterator) {
		return std::move(upcoming);
	}
	auto claim = session.Claim();
	try {
		return iterator->GetStatement();
	} catch (std::exception &ex) {
		Fail(ErrorData(ex));
	}
}

bool StatementGroup::HasMore() {
	// Never parses, so the next statement is parsed only after this one ran
	return upcoming || (iterator && iterator->HasMore());
}

unique_ptr<SQLStatement> StatementGroup::RunAllButLast(unique_ptr<SQLStatement> statement,
                                                       const InterruptCheck &check) {
	while (HasMore()) {
		if (!statement->named_param_map.empty()) {
			// Parsed first, so a syntax error in the statement after this one wins over the refusal
			Next();
			auto claim = session.Claim();
			Fail(ErrorData(NotImplementedException(
			    "Prepared parameters are only supported for the last statement, please split your query up into "
			    "separate 'execute' calls if you want to use prepared parameters")));
		}
		{
			// One claim from submission to the end of the run, so no other call finds the statement pending
			auto claim = session.Claim();
			identifier_map_t<BoundParameterData> no_values;
			auto result = session.SubmitClaimed(std::move(statement), no_values, Format::Chunks());
			ReleaseIfReplaced(*result);
			result->RetainClaimed(check);
			Keep(std::move(result));
		}
		statement = Next();
		if (!statement) {
			return nullptr;
		}
	}
	return statement;
}

static bool ReturnsRows(const StatementProperties &properties) {
	return properties.return_type == StatementReturnType::QUERY_RESULT;
}

void StatementGroup::ReleaseIfReplaced(const Result &result) {
	// A statement that expands into several knows whether it returns rows only once it ran
	if (result.MetadataAvailable() && ReturnsRows(result.Properties())) {
		kept.reset();
	}
}

void StatementGroup::Keep(shared_ptr<Result> result) {
	if (kept && ReturnsRows(kept->Properties()) && !ReturnsRows(result->Properties())) {
		return;
	}
	kept = std::move(result);
}

shared_ptr<Result> StatementGroup::Choose(shared_ptr<Result> last) {
	if (last) {
		Keep(std::move(last));
	}
	return std::move(kept);
}

void StatementGroup::RefuseParameters() {
	auto claim = session.Claim();
	Fail(ErrorData(InvalidInputException("Parameters were given, but the last statement of the query takes none")));
}

void StatementGroup::Fail(ErrorData error) {
	auto &context = session.Context();
	// As in the engine's Query: the error gets the location of its statement in the query, or becomes JSON
	context.ProcessError(error, query);
	// Workaround: the engine applies its transaction policy to such an error in ErrorResult, which is private; the
	// catch block of RunFunctionInTransactionInternal applies the same policy, so the error is rethrown through it.
	// Only an explicit transaction is aborted, as in the engine's Query; an autocommit one belongs to an open result.
	if (aborts_transaction && !context.transaction.IsAutoCommit()) {
		context.RunFunctionInTransaction([&]() { error.Throw(); }, false);
	}
	error.Throw();
}

} // namespace engine
} // namespace duckdb

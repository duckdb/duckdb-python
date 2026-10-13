#include "duckdb_python/engine/connection.hpp"

#include "duckdb/main/relation.hpp"
#include "duckdb/main/statement_iterator.hpp"
#include "duckdb/main/valid_checker.hpp"
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

void ConnectionState::Pin(optional_ptr<const ResolvedTables> tables) {
	lock_guard<mutex> guard(lock);
	pinned = tables;
	pinned_thread = std::this_thread::get_id();
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
	PinGuard(ConnectionState &state, const ResolvedTables &tables) : state(state) {
		state.Pin(&tables);
	}
	~PinGuard() {
		state.Pin(nullptr);
	}

private:
	ConnectionState &state;
};

} // namespace

ClaimingContextWrapper::ClaimingContextWrapper(shared_ptr<ClientContextWrapper> inner_p)
    : ClientContextWrapper(inner_p->TryGetContext()), inner(std::move(inner_p)) {
}

void ClaimingContextWrapper::TryBindRelation(Relation &relation, vector<ColumnDefinition> &columns) {
	auto context = GetContext();
	ConnectionClaim claim(*context);
	inner->TryBindRelation(relation, columns);
}

void ClaimingContextWrapper::Install(Relation &relation) {
	if (!relation.context || dynamic_cast<ClaimingContextWrapper *>(relation.context.get())) {
		return;
	}
	relation.context = make_shared_ptr<ClaimingContextWrapper>(relation.context);
}

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
    : context(std::move(context_p)), state(ConnectionState::Get(*context)) {
}

vector<unique_ptr<SQLStatement>> Session::Parse(const string &sql) {
	ConnectionClaim claim(*state, *context);
	// The iterator also runs the preprocessor, without which a PRAGMA cannot be prepared
	auto iterator = context->IterateStatements(sql);
	vector<unique_ptr<SQLStatement>> statements;
	while (iterator.Peek()) {
		if (auto statement = iterator.GetStatement()) {
			statements.push_back(std::move(statement));
		}
	}
	return statements;
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

shared_ptr<Result> Session::Submit(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
                                   const Format &format) {
	ConnectionClaim claim(*state, *context);
	auto written_type = statement->type;
	auto handle = context->Submit(std::move(statement), values, QueryParameters(format.EngineFormat()));
	NameRefusal(*state, *handle);
	return make_shared_ptr<Result>(std::move(handle), written_type, format, nullptr);
}

shared_ptr<Result> Session::Submit(Prepared &prepared, identifier_map_t<BoundParameterData> &values,
                                   const Format &format) {
	ConnectionClaim claim(*state, *context);
	unique_ptr<QueryResult> handle;
	{
		PinGuard guard(*state, prepared.tables);
		handle = prepared.statement->Submit(values, QueryParameters(format.EngineFormat()));
	}
	NameRefusal(*state, *handle);
	return make_shared_ptr<Result>(std::move(handle), prepared.statement->GetStatementType(), format, nullptr);
}

shared_ptr<Result> Session::Submit(const shared_ptr<Relation> &relation, const Format &format) {
	ConnectionClaim claim(*state, *context);
	QueryParameters parameters(format.EngineFormat());
	if (!relation->IsReadOnly()) {
		auto handle = context->Submit(relation, parameters);
		NameRefusal(*state, *handle);
		return make_shared_ptr<Result>(std::move(handle), StatementType::RELATION_STATEMENT, format, relation);
	}
	auto select = make_uniq<SelectStatement>();
	select->node = relation->GetQueryNode();
	select->query = relation->GetQuery();
	auto handle = context->Submit(std::move(select), parameters);
	NameRefusal(*state, *handle);
	return make_shared_ptr<Result>(std::move(handle), StatementType::SELECT_STATEMENT, format, relation);
}

void Session::Run(unique_ptr<SQLStatement> statement, const InterruptCheck &check) {
	ConnectionClaim claim(*state, *context);
	auto handle = context->Submit(std::move(statement), QueryParameters());
	NameRefusal(*state, *handle);
	Complete(*handle, check);
}

} // namespace engine
} // namespace duckdb

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/engine/connection.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/engine/result.hpp"

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/client_context_wrapper.hpp"
#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/main/statement_iterator.hpp"
#include "duckdb/parser/sql_statement.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/storage/object_cache.hpp"

#include <deque>
#include <thread>

namespace duckdb {

namespace engine {

//! The tables a replacement scan resolved, by the name exactly as the statement used it, which is how the
//! client's scan looks it up
using ResolvedTables = unordered_map<string, unique_ptr<TableRef>>;

//! Marks the thread as running Python code for a query of the context. Calls from that code into the same
//! connection are refused: the query waits for the code, so code that waited for the connection would wait on itself.
class CallbackScope {
public:
	explicit CallbackScope(const ClientContext &context);
	~CallbackScope();

	CallbackScope(const CallbackScope &) = delete;
	CallbackScope &operator=(const CallbackScope &) = delete;

	//! Whether this thread is inside a callback of a query of the context, at any depth
	static bool Inside(const ClientContext &context);

private:
	const ClientContext &context;
	CallbackScope *outer;
};

//! Client state kept on the engine's connection. It never owns a result: a result holds the context, and
//! the context holds this state.
class ConnectionState : public ClientContextState {
public:
	static shared_ptr<ConnectionState> Get(ClientContext &context);

public:
	//! Throws the refusal a claim would get, without claiming
	void ThrowIfBusy(const ClientContext &context) const;
	//! The result of the DB-API's execute(). Only the DB-API can end it, so a refusal it causes says how.
	void SetDbapiResult(const shared_ptr<Result> &result);
	//! Throws the refusal naming the DB-API's unfinished result, when the error is the engine refusing work because
	//! that result holds the connection
	void NameRefusal(const ErrorData &error);

	//! Between the two calls, every table the client's replacement scan resolves is recorded
	void BeginResolving();
	ResolvedTables EndResolving();
	void RecordResolved(const string &name, TableRef &table);
	//! While pinned, the replacement scan answers a recorded name with the recorded table; an exclusive pin answers
	//! no other name
	void Pin(optional_ptr<const ResolvedTables> tables, bool exclusive = false);
	unique_ptr<TableRef> RecallResolved(const string &name);
	bool PinnedExclusively();

private:
	friend class ConnectionClaim;

	atomic<bool> busy {false};
	mutex lock;
	weak_ptr<Result> dbapi_result;
	//! Both belong to the thread binding the statement, so another thread's bind on the context does not see them
	unique_ptr<ResolvedTables> resolving;
	std::thread::id resolving_thread;
	optional_ptr<const ResolvedTables> pinned;
	bool pinned_exclusively = false;
	std::thread::id pinned_thread;
};

//! Tables the client registered under generated names, shared by every connection of a database. The client's
//! replacement scan answers these names before any other lookup.
class SourceRegistry : public ObjectCacheEntry {
public:
	static shared_ptr<SourceRegistry> Get(ClientContext &context);
	//! Unique in the process, so a name never reaches another database's sources or a user's table
	static string NextName();
	//! Whether the name has the form NextName gives, which every registry name has
	static bool IsSourceName(const string &name);

	static string ObjectType() {
		return "python_source_registry";
	}
	string GetObjectType() override {
		return ObjectType();
	}
	optional_idx GetEstimatedCacheMemory() const override {
		return optional_idx();
	}

public:
	//! How many statements that can change what a name binds to the client has run on this database, so a schema
	//! read earlier can be re-read when it changed. Lock free: the counter is read before every bind.
	idx_t CatalogVersion() const {
		return catalog_version;
	}
	//! Counts a statement the client submitted, unless it only reads
	void CountStatement(StatementType type) {
		if (type != StatementType::SELECT_STATEMENT) {
			++catalog_version;
		}
	}
	//! Counts a change the client made without a statement of its own, a view it registered for a Python object
	void CountChange() {
		++catalog_version;
	}

	//! subject is the name the object was registered under, a view's or a variable's, or empty for none
	void Register(const string &name, unique_ptr<TableRef> table, const string &subject = string());
	//! The removed table is returned for the caller to destroy outside the lock, since it can own Python objects
	unique_ptr<TableRef> Unregister(const string &name);
	unique_ptr<TableRef> Lookup(const string &name);
	//! Throws for a name this registry held and released, naming what was registered under it; a name it never held
	//! is left to the catalog's own error
	void ThrowIfReleased(const string &name);

private:
	struct Entry {
		unique_ptr<TableRef> table;
		string subject;
	};
	//! What a released name held, so a query that still reads it is told what is gone and not shown the name
	struct Released {
		string kind;
		string subject;
	};
	//! How many released names are remembered, so a long session does not grow this without bound
	static constexpr idx_t RELEASED_MEMORY = 4096;

	atomic<idx_t> catalog_version {0};
	mutex lock;
	unordered_map<string, Entry> sources;
	unordered_map<string, Released> released;
	std::deque<string> released_order;
};

//! Holds the connection for one stretch of engine work. A second claim, and any claim from a callback of the
//! connection's queries, is refused instead of waiting: the work it would wait for may be waiting for it.
class ConnectionClaim {
public:
	ConnectionClaim(ConnectionState &state, const ClientContext &context);
	explicit ConnectionClaim(ClientContext &context);
	~ConnectionClaim();

	ConnectionClaim(const ConnectionClaim &) = delete;
	ConnectionClaim &operator=(const ConnectionClaim &) = delete;

private:
	ConnectionState &state;
};

//! The engine's check that no result holds the connection, with its refusal named as ConnectionState::NameRefusal does
void VerifyNoOpenResult(ClientContext &context);
//! Names the refusal in an error from engine work that runs outside a Session, as ConnectionState::NameRefusal does
void NameRefusal(ClientContext &context, const std::exception &error);

//! A prepared statement with the tables its replacement scans resolved. A rebind at submission, after the
//! catalog moved, finds the same tables instead of looking the names up again where the result is read.
struct Prepared {
	unique_ptr<PreparedStatement> statement;
	ResolvedTables tables;
};

//! The statements, prepared statements and results of one engine connection
class Session {
public:
	explicit Session(shared_ptr<ClientContext> context);

public:
	ClientContext &Context() const {
		return *context;
	}
	Prepared Prepare(unique_ptr<SQLStatement> statement);
	shared_ptr<Result> Submit(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
	                          const Format &format);
	//! Submit for a caller that already holds the connection's claim
	shared_ptr<Result> SubmitClaimed(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
	                                 const Format &format);
	shared_ptr<Result> Submit(Prepared &prepared, identifier_map_t<BoundParameterData> &values, const Format &format);
	//! Scans a collection, for example to read rows a result already holds in another format
	shared_ptr<Result> Submit(unique_ptr<ColumnDataCollection> collection, const vector<Identifier> &names,
	                          const Format &format);
	//! Binds a statement without running it, keeping the tables its replacement scans resolved in tables
	StatementSignature Bind(unique_ptr<SQLStatement> statement, ResolvedTables &tables);
	//! Binds a statement with values for its parameters, as the engine rebinds a prepared statement for the values it
	//! runs with, so the names and types are those of the statement that runs; nothing runs
	StatementSignature Bind(unique_ptr<SQLStatement> statement, ResolvedTables &tables,
	                        const identifier_map_t<BoundParameterData> &values);
	//! Runs a statement to its end and drops its rows
	void Run(unique_ptr<SQLStatement> statement, const InterruptCheck &check);
	//! Run for a caller that already holds the connection's claim
	void RunClaimed(unique_ptr<SQLStatement> statement, const InterruptCheck &check);
	//! Runs a statement to its end and drops its rows, its replacement scans answered from tables and nothing else
	void Run(unique_ptr<SQLStatement> statement, const ResolvedTables &tables, const InterruptCheck &check);
	ConnectionClaim Claim();

private:
	shared_ptr<ClientContext> context;
	shared_ptr<ConnectionState> state;
	//! Counts the statements submitted here, which is where every statement of the client passes
	shared_ptr<SourceRegistry> registry;
};

//! The statements of one SQL string, run in order as the engine's Query runs them: each is parsed only once the ones
//! before it have run, and the engine expands it, a PRAGMA for example, when it is submitted
class StatementGroup {
public:
	//! aborts_transaction false only parses, as the engine's ExtractStatements does: a parse error leaves an explicit
	//! transaction valid
	StatementGroup(Session session, const string &sql, bool aborts_transaction = true);
	//! A group of one statement parsed earlier
	StatementGroup(Session session, unique_ptr<SQLStatement> statement);

	StatementGroup(const StatementGroup &) = delete;
	StatementGroup &operator=(const StatementGroup &) = delete;
	StatementGroup(StatementGroup &&) = default;

public:
	//! The first statement, parsed but not run; null for a group without statements. Only before Next. Inside an
	//! explicit transaction, an error parsing it aborts the transaction, as in the engine.
	optional_ptr<const SQLStatement> First();
	//! The next statement, parsed but not run; null when no statement is left. Errors abort an explicit transaction
	//! as First's do.
	unique_ptr<SQLStatement> Next();
	//! Runs the statement and every one after it but the last, which it returns unrun; null when the text after the
	//! statements it ran holds no statement
	unique_ptr<SQLStatement> RunAllButLast(unique_ptr<SQLStatement> statement, const InterruptCheck &check);
	//! Called once a statement is submitted and before it runs: one known to return rows replaces the kept result,
	//! which is dropped now so that two results are never held at once
	void ReleaseIfReplaced(const Result &result);
	//! The result of the group, given the last statement's: that of the last statement that returns rows, or of the
	//! last statement when none does
	shared_ptr<Result> Choose(shared_ptr<Result> last);
	//! Throws for parameters given to a group whose text after the statements that ran held no statement
	[[noreturn]] void RefuseParameters();

private:
	bool HasMore();
	void Keep(shared_ptr<Result> result);
	//! Throws an error that ends the group before the engine ran the statement it concerns, as the engine's Query
	//! reports it. Requires the claim.
	[[noreturn]] void Fail(ErrorData error);

private:
	Session session;
	//! The text the statements came from, which an error's location refers to
	string query;
	bool aborts_transaction = true;
	//! Null for a group of one statement parsed earlier, which is held in upcoming
	unique_ptr<StatementIterator> iterator;
	//! Parsed by First, not yet returned by Next
	unique_ptr<SQLStatement> upcoming;
	shared_ptr<Result> kept;
};

} // namespace engine
} // namespace duckdb

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
#include "duckdb/parser/sql_statement.hpp"
#include "duckdb/parser/tableref.hpp"

#include <thread>

namespace duckdb {
class Relation;

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
	//! While pinned, the replacement scan answers a recorded name with the recorded table
	void Pin(optional_ptr<const ResolvedTables> tables);
	unique_ptr<TableRef> RecallResolved(const string &name);

private:
	friend class ConnectionClaim;

	atomic<bool> busy {false};
	mutex lock;
	weak_ptr<Result> dbapi_result;
	//! Both belong to the thread binding the statement, so another thread's bind on the context does not see them
	unique_ptr<ResolvedTables> resolving;
	std::thread::id resolving_thread;
	optional_ptr<const ResolvedTables> pinned;
	std::thread::id pinned_thread;
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

//! Claims the connection while a relation binds, so a bind is refused rather than blocked; relations derived
//! from one that has it inherit it
class ClaimingContextWrapper : public ClientContextWrapper {
public:
	explicit ClaimingContextWrapper(shared_ptr<ClientContextWrapper> inner);

	void TryBindRelation(Relation &relation, vector<ColumnDefinition> &columns) override;

	static void Install(Relation &relation);

private:
	shared_ptr<ClientContextWrapper> inner;
};

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
	vector<unique_ptr<SQLStatement>> Parse(const string &sql);
	Prepared Prepare(unique_ptr<SQLStatement> statement);
	shared_ptr<Result> Submit(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
	                          const Format &format);
	shared_ptr<Result> Submit(Prepared &prepared, identifier_map_t<BoundParameterData> &values, const Format &format);
	//! A relation that writes is submitted as itself; one that reads as a statement over its query
	shared_ptr<Result> Submit(const shared_ptr<Relation> &relation, const Format &format);
	//! Runs a statement to its end and drops its rows
	void Run(unique_ptr<SQLStatement> statement, const InterruptCheck &check);

private:
	shared_ptr<ClientContext> context;
	shared_ptr<ConnectionState> state;
};

} // namespace engine
} // namespace duckdb

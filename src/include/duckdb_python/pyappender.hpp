#pragma once

#include "duckdb_python/engine/connection.hpp"
#include "duckdb_python/nb/casters.hpp"

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/sql_statement.hpp"

#include <memory>
#include <new>
#include <optional>

namespace duckdb {

class DuckDB;
struct DuckDBPyConnection;

//! Buffers rows and applies them with one statement per flush. The statement reads the buffered rows as the CTE
//! appender_buffer and the source query over them as the CTE appender_data, ahead of the DML's own CTEs.
class DuckDBPyAppender {
public:
	//! Inserts, upserts or updates rows of a table, resolved once now, over its non-generated columns or the listed
	//! ones, or deletes its rows over just the key columns; an update or delete matches rows on the key
	static std::unique_ptr<DuckDBPyAppender> ForTable(std::shared_ptr<DuckDBPyConnection> connection,
	                                                  const string &table, const std::optional<nb::list> &columns,
	                                                  const string &action, const std::optional<nb::list> &key);
	//! Runs the caller's DML over buffer columns declared as a dict of name to type
	static std::unique_ptr<DuckDBPyAppender> ForQuery(std::shared_ptr<DuckDBPyConnection> connection, const string &dml,
	                                                  const nb::dict &columns, const std::optional<string> &source);
	~DuckDBPyAppender();

	DuckDBPyAppender(const DuckDBPyAppender &) = delete;
	DuckDBPyAppender &operator=(const DuckDBPyAppender &) = delete;

public:
	void AppendRow(const nb::args &values);
	void AppendRows(const nb::handle &rows);
	void Flush();
	void Clear();
	void Enter();
	void Exit(const nb::handle &exc_type);

	static void Initialize(nb::handle &m);

private:
	DuckDBPyAppender(std::shared_ptr<DuckDBPyConnection> connection, ClientContext &context, vector<Identifier> names,
	                 vector<LogicalType> types);

	static std::unique_ptr<DuckDBPyAppender> Create(std::shared_ptr<DuckDBPyConnection> connection,
	                                                shared_ptr<ClientContext> context, vector<Identifier> names,
	                                                vector<LogicalType> types, const string &source, const string &dml);

	//! Refuses a second caller instead of waiting, whether it is another thread or code running inside a flush
	class Claim {
	public:
		explicit Claim(DuckDBPyAppender &appender);
		//! Takes the appender only if no other caller holds it, and refuses nothing
		Claim(DuckDBPyAppender &appender, std::nothrow_t);
		~Claim();

		bool Held() const {
			return held;
		}

		Claim(const Claim &) = delete;
		Claim &operator=(const Claim &) = delete;

	private:
		DuckDBPyAppender &appender;
		bool held;
	};

	void ThrowIfClosed() const;
	void ThrowIfBroken() const;
	//! Each call holds both until it returns: a connection closed mid-call keeps its context and database until then,
	//! and the database's wrapper lets a reconnect to the same file join it instead of waiting for it to shut down
	struct Held {
		shared_ptr<DuckDB> database;
		shared_ptr<ClientContext> context;
	};
	Held Hold() const;
	static Value Convert(ClientContext &context, const nb::handle &object, const LogicalType &type);
	void AppendSequence(ClientContext &context, const nb::handle &row, idx_t count);
	void Stage(const vector<Value> &row);
	void MoveStagedIntoBuffer();
	void DropRows();
	idx_t BufferedRows() const;

private:
	std::shared_ptr<DuckDBPyConnection> connection;
	vector<Identifier> names;
	vector<LogicalType> types;
	//! Its blocks belong to the database, so it holds no reference that would keep a closed connection's database open
	unique_ptr<ColumnDataCollection> buffer;
	DataChunk staged;
	//! Reads the buffer through a reference, so it is declared after it and destroyed first
	unique_ptr<SQLStatement> statement;
	//! The Python objects the statement names, found where the appender was created and reused at every flush
	engine::ResolvedTables tables;
	//! Set when a write into the buffer failed part way, leaving it holding an unknown part of the rows
	bool broken = false;
	atomic<bool> busy {false};
};

} // namespace duckdb

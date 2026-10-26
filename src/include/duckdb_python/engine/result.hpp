//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/engine/result.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/arrow/arrow_format.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/query_result_stream.hpp"

#include <functional>

namespace duckdb {

namespace engine {

//! Runs between engine tasks of a blocking call and throws to abandon it
using InterruptCheck = std::function<void()>;

//! The unit a result is produced in, fixed when the statement is submitted
class Format {
public:
	static Format Chunks();
	//! batch_size is a maximum: the engine also cuts an array at row group and producer boundaries
	static Format Arrow(idx_t batch_size);

public:
	bool IsArrow() const {
		return engine_format != nullptr;
	}
	//! Null for chunks, the engine's default format
	const shared_ptr<ResultFormat> &EngineFormat() const {
		return engine_format;
	}

private:
	explicit Format(shared_ptr<ResultFormat> engine_format);

private:
	shared_ptr<ResultFormat> engine_format;
};

class ConnectionState;
class ConnectionClaim;

//! A submitted statement. Retention is settled by the first consumer: a fetch opens a stream, Retain and the
//! Take calls keep every row. Every call that drives the query claims the connection, takes the client context
//! lock and may wait for running tasks, so callers must not hold a lock those tasks need.
class Result {
public:
	//! written_type is the submitted statement's own type, since the handle of a prepared execution reports EXECUTE
	Result(unique_ptr<QueryResult> submitted, StatementType written_type, Format format);
	~Result();

	Result(const Result &) = delete;
	Result &operator=(const Result &) = delete;

public:
	//! False for a statement that expands into several until it completes
	bool MetadataAvailable() const {
		return metadata_available;
	}
	const vector<Identifier> &Names() const {
		ThrowIfFailed();
		RequireMetadata();
		return names;
	}
	const vector<LogicalType> &Types() const {
		ThrowIfFailed();
		RequireMetadata();
		return types;
	}
	const StatementProperties &Properties() const {
		ThrowIfFailed();
		RequireMetadata();
		return properties;
	}
	const ClientProperties &GetClientProperties() const {
		return client_properties;
	}
	const Format &GetFormat() const {
		return format;
	}
	bool IsRetained() const {
		return state == State::RETAINED;
	}
	//! Whether the query still runs on its connection with rows left to produce. Takes the client context lock.
	bool Running() const;
	//! Only a SELECT that returns rows streams; any other statement runs to its end before its result is returned
	bool CompletesBeforeReturning() const {
		return completes_before_returning;
	}

	//! The context the query ran on, which the result only borrows: a closed connection leaves it gone, and every
	//! call that needs it refuses instead of reading freed memory. A streaming result keeps it through its handle.
	shared_ptr<ClientContext> TryGetContext() const;
	//! The database whose buffers hold the rows, null once it is gone. Weak for the same reason as the context.
	shared_ptr<DatabaseInstance> TryGetDatabase() const;
	//! Keeps every row: drives the query to its end unless it already ended. A failure ends the query.
	void Retain(const InterruptCheck &check);
	//! Retain for a caller that already holds the connection's claim. A failure ends the query.
	void RetainClaimed(const InterruptCheck &check);
	//! Runs the statement until its first unit is ready or it ends, without deciding how its rows are read,
	//! and throws the error it ended with. A failure ends the query.
	void Settle(const InterruptCheck &check);
	//! Settle for a caller that already holds the connection's claim. A failure ends the query.
	void SettleClaimed(const InterruptCheck &check);
	//! Rows of a retained result. Materializes it first.
	idx_t RowCount(const InterruptCheck &check);
	//! The next chunk, not flattened; null at the end. Opens a stream unless the result is retained.
	unique_ptr<DataChunk> FetchChunk(const InterruptCheck &check);
	//! The next Arrow array; null at the end. Opens a stream unless the result is retained.
	unique_ptr<ArrowArrayWrapper> FetchArray(const InterruptCheck &check);
	//! Every remaining row, the result is empty afterwards
	unique_ptr<ColumnDataCollection> TakeChunks(const InterruptCheck &check);
	//! Every remaining array, the result is empty afterwards
	vector<unique_ptr<ArrowArrayWrapper>> TakeArrays(const InterruptCheck &check);
	//! A copy of the result's Arrow schema, owned by the caller. Arrow results only.
	void CopyArrowSchema(ArrowSchema &out) const;
	//! An Arrow schema for the result's types under the given names, built in a transaction on its connection
	void BuildArrowSchema(const vector<string> &names, ArrowSchema &out) const;
	//! Renders the rows of a retained result
	string ToBox(BoxRendererContext &context, const BoxRendererConfig &config);

	//! Ends the query if it is still running. Idempotent, and safe while another thread is fetching.
	void Close();
	//! The rows a completed DML statement changed, or -1
	int64_t ChangedRows();

private:
	enum class State : uint8_t { PENDING, STREAMING, RETAINED, DRAINED, CLOSED };

	void CaptureMetadata();
	void RequireMetadata() const;
	//! Ends the query of a call that failed under the claim, and keeps its error for any later use of the result
	void CloseAfterFailure(std::exception_ptr error) noexcept;
	//! A result that a failure ended reports that failure instead of reading as empty
	void ThrowIfFailed() const;
	//! The connection's claim while the query can still run engine work, otherwise none
	unique_ptr<ConnectionClaim> ClaimWhileRunning();
	void OpenStream();
	template <class FORMAT>
	unique_ptr<typename FORMAT::T> FetchStreamUnit(const InterruptCheck &check);
	void EndStream();

private:
	unique_ptr<QueryResult> handle;
	unique_ptr<ResultStreamBase> stream;
	Format format;
	atomic<State> state;
	bool retained_fetched = false;
	//! Guards handle, stream and failure against Close from another thread
	mutable mutex handle_lock;
	//! The error that ended the query, set when a driving call failed
	std::exception_ptr failure;
	//! The state of the connection the query runs on, whose claim every driving call takes
	shared_ptr<ConnectionState> connection_state;
	//! Weak, so a result never keeps a connection's database alive, and never reads a context that is gone
	weak_ptr<ClientContext> context_owner;
	weak_ptr<DatabaseInstance> database_owner;
	vector<Identifier> names;
	vector<LogicalType> types;
	StatementProperties properties;
	bool metadata_available = false;
	bool completes_before_returning = false;
	ClientProperties client_properties;
	//! An Arrow result never defers its metadata, so the copy happens at submission, while the producing
	//! transaction is live
	ArrowSchemaWrapper arrow_schema;
};

//! Runs a result to its end and keeps every row, as a blocking call
void Complete(QueryResult &result, const InterruptCheck &check);

} // namespace engine
} // namespace duckdb

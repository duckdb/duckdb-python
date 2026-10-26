#include "duckdb_python/engine/result.hpp"
#include "duckdb_python/engine/connection.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/nanoarrow/nanoarrow.hpp"
#include "duckdb/common/enums/query_result_state.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"

namespace duckdb {
namespace engine {

Format::Format(shared_ptr<ResultFormat> engine_format_p) : engine_format(std::move(engine_format_p)) {
}

Format Format::Chunks() {
	return Format(nullptr);
}

Format Format::Arrow(idx_t batch_size) {
	if (batch_size == 0) {
		throw InvalidInputException("The Arrow batch size must be larger than 0");
	}
	return Format(make_shared_ptr<ArrowFormat>(batch_size));
}

//! Runs tasks until the query ends or, when asked, until the engine waits for the consumer
static QueryResultState Drive(QueryResult &result, const InterruptCheck &check, bool stop_when_ready) {
	// A result that arrives finished has no context to run tasks on; Poll reports that without touching it.
	auto state = result.Poll();
	while (!IsTerminal(state) && !(stop_when_ready && state == QueryResultState::READY)) {
		check();
		if (state == QueryResultState::BLOCKED || state == QueryResultState::READY ||
		    state == QueryResultState::NO_TASKS_AVAILABLE) {
			result.WaitForTask();
		}
		state = result.ExecuteTask();
	}
	return state;
}

void Complete(QueryResult &result, const InterruptCheck &check) {
	if (result.HasError()) {
		result.ThrowError();
	}
	result.Materialize();
	Drive(result, check, false);
	// FINISHED only means the executor is done: Complete takes the rows and ends the query with its
	// transaction step. Left open, the next statement would roll it back.
	result.Complete();
	if (result.HasError()) {
		result.ThrowError();
	}
}

Result::Result(unique_ptr<QueryResult> submitted, StatementType written_type, Format format_p)
    : handle(std::move(submitted)), format(std::move(format_p)), state(State::PENDING) {
	if (!handle) {
		throw InternalException("engine::Result created without a query result");
	}
	if (handle->HasError()) {
		handle->ThrowError();
	}
	completes_before_returning = !StreamsByDefault(written_type, *handle);
	client_properties = handle->client_properties;
	if (!client_properties.client_context) {
		throw InternalException("engine::Result created without a client context");
	}
	connection_state = ConnectionState::Get(*client_properties.client_context);
	context_owner = client_properties.client_context->shared_from_this();
	database_owner = client_properties.client_context->db;
	CaptureMetadata();
}

void Result::CaptureMetadata() {
	if (metadata_available || !handle->MetadataAvailable()) {
		return;
	}
	names = handle->GetNames();
	types = handle->GetTypes();
	properties = handle->GetStatementProperties();
	if (format.IsArrow()) {
		auto &schema = handle->FormatState<ArrowFormat>().Schema();
		if (duckdb_nanoarrow::ArrowSchemaDeepCopy(&schema, &arrow_schema.arrow_schema) != NANOARROW_OK) {
			throw OutOfMemoryException("Failed to copy the Arrow schema of the query result");
		}
	}
	metadata_available = true;
}

void Result::RequireMetadata() const {
	if (!metadata_available) {
		throw InternalException("engine::Result metadata read before the statement that produces the result was bound");
	}
}

Result::~Result() {
	try {
		Close();
	} catch (...) { // NOLINT: a destructor must not throw
	}
}

void Result::OpenStream() {
	// Such a statement cannot be streamed; its rows are served from the retained handle instead
	if (CompletesBeforeReturning()) {
		return;
	}
	lock_guard<mutex> guard(handle_lock);
	// A concurrent first consumer can have settled the result while this one waited for the lock
	if (state != State::PENDING || !handle) {
		return;
	}
	try {
		if (format.IsArrow()) {
			stream = make_uniq<QueryResultStream<ArrowFormat>>(std::move(handle));
		} else {
			stream = make_uniq<QueryResultStream<ChunkFormat>>(std::move(handle));
		}
	} catch (...) {
		// A stream that fails to open has consumed and ended the query
		state = State::CLOSED;
		throw;
	}
	state = State::STREAMING;
}

void Result::EndStream() {
	// The stream object stays alive: another thread can still be inside a fetch on it, and a drained
	// stream keeps reporting its terminal state
	state = State::DRAINED;
}

template <class FORMAT>
unique_ptr<typename FORMAT::T> Result::FetchStreamUnit(const InterruptCheck &check) {
	auto &typed = static_cast<QueryResultStream<FORMAT> &>(*stream);
	while (true) {
		unique_ptr<typename FORMAT::T> unit;
		auto query_state = typed.TryFetch(unit);
		if (unit) {
			return unit;
		}
		if (query_state == QueryResultState::FINISHED) {
			EndStream();
			return nullptr;
		}
		if (query_state == QueryResultState::EXECUTION_ERROR) {
			typed.GetErrorObject().Throw();
		}
		check();
		query_state = typed.ExecuteTask();
		if (query_state == QueryResultState::BLOCKED || query_state == QueryResultState::NO_TASKS_AVAILABLE) {
			typed.WaitForTask();
		}
	}
}

shared_ptr<DatabaseInstance> Result::TryGetDatabase() const {
	return database_owner.lock();
}

shared_ptr<ClientContext> Result::TryGetContext() const {
	auto context = context_owner.lock();
	if (!context) {
		throw ConnectionException("Connection has already been closed");
	}
	return context;
}

void Result::Retain(const InterruptCheck &check) {
	auto claim = ClaimWhileRunning();
	RetainClaimed(check);
}

void Result::CloseAfterFailure(std::exception_ptr error) noexcept {
	{
		lock_guard<mutex> guard(handle_lock);
		failure = std::move(error);
	}
	// Called under the claim: ending a query can run Python code, such as a file's close, that reenters the
	// connection, and the claim makes it refuse instead of waiting on the call that holds the connection
	try {
		Close();
	} catch (...) { // NOLINT: the error of the failed call is the one to report
	}
}

void Result::ThrowIfFailed() const {
	lock_guard<mutex> guard(handle_lock);
	if (failure) {
		std::rethrow_exception(failure);
	}
}

unique_ptr<ConnectionClaim> Result::ClaimWhileRunning() {
	// Rows already in the client need no engine work, so a fetch of them claims nothing and works anywhere
	if (state != State::PENDING && state != State::STREAMING) {
		return nullptr;
	}
	return make_uniq<ConnectionClaim>(*connection_state, *TryGetContext());
}

void Result::RetainClaimed(const InterruptCheck &check) {
	if (state != State::PENDING) {
		return;
	}
	try {
		Complete(*handle, check);
		CaptureMetadata();
	} catch (...) {
		CloseAfterFailure(std::current_exception());
		throw;
	}
	state = State::RETAINED;
}

void Result::Settle(const InterruptCheck &check) {
	auto claim = ClaimWhileRunning();
	SettleClaimed(check);
}

void Result::SettleClaimed(const InterruptCheck &check) {
	if (state != State::PENDING) {
		return;
	}
	if (CompletesBeforeReturning()) {
		RetainClaimed(check);
		return;
	}
	try {
		// Undecided, the engine parks the first unit at the result sink; READY means it waits for that decision
		auto query_state = Drive(*handle, check, true);
		if (handle->HasError()) {
			handle->ThrowError();
		}
		if (query_state == QueryResultState::FINISHED) {
			// A statement that sank no row finishes undecided, and Poll reporting FINISHED does not end the query,
			// so it is retained here, which does
			RetainClaimed(check);
		}
	} catch (...) {
		CloseAfterFailure(std::current_exception());
		throw;
	}
}

idx_t Result::RowCount(const InterruptCheck &check) {
	ThrowIfFailed();
	auto claim = ClaimWhileRunning();
	RetainClaimed(check);
	if (state != State::RETAINED) {
		throw InternalException("engine::Result::RowCount on a result that is not retained");
	}
	return handle->RowCount();
}

unique_ptr<DataChunk> Result::FetchChunk(const InterruptCheck &check) {
	D_ASSERT(!format.IsArrow());
	ThrowIfFailed();
	auto claim = ClaimWhileRunning();
	if (state == State::PENDING) {
		OpenStream();
		if (state == State::PENDING) {
			RetainClaimed(check);
		}
	}
	switch (state) {
	case State::STREAMING:
		return FetchStreamUnit<ChunkFormat>(check);
	case State::RETAINED: {
		retained_fetched = true;
		auto chunk = handle->FetchRaw();
		if (handle->HasError()) {
			handle->ThrowError();
		}
		return chunk;
	}
	default:
		return nullptr;
	}
}

unique_ptr<ArrowArrayWrapper> Result::FetchArray(const InterruptCheck &check) {
	D_ASSERT(format.IsArrow());
	ThrowIfFailed();
	auto claim = ClaimWhileRunning();
	if (state == State::PENDING) {
		OpenStream();
		if (state == State::PENDING) {
			RetainClaimed(check);
		}
	}
	switch (state) {
	case State::STREAMING:
		return FetchStreamUnit<ArrowFormat>(check);
	case State::RETAINED:
		retained_fetched = true;
		return handle->Fetch<ArrowFormat>();
	default:
		return nullptr;
	}
}

unique_ptr<ColumnDataCollection> Result::TakeChunks(const InterruptCheck &check) {
	D_ASSERT(!format.IsArrow());
	ThrowIfFailed();
	// Claimed only while the query can still run engine work: rows already in the client need no connection
	auto claim = ClaimWhileRunning();
	if (state == State::PENDING) {
		RetainClaimed(check);
	}
	if (state == State::RETAINED && !retained_fetched) {
		auto collection = handle->TakeCollection<ChunkFormat>();
		state = State::DRAINED;
		return collection;
	}
	auto collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), Types());
	if (state == State::RETAINED) {
		// Fetching copies out of the collection, so only the rows after the fetch cursor remain
		while (auto chunk = handle->FetchRaw()) {
			collection->Append(*chunk);
		}
		state = State::DRAINED;
	} else if (state == State::STREAMING) {
		while (auto chunk = FetchStreamUnit<ChunkFormat>(check)) {
			collection->Append(*chunk);
		}
	}
	return collection;
}

vector<unique_ptr<ArrowArrayWrapper>> Result::TakeArrays(const InterruptCheck &check) {
	D_ASSERT(format.IsArrow());
	ThrowIfFailed();
	// Claimed only while the query can still run engine work, as in TakeChunks
	auto claim = ClaimWhileRunning();
	vector<unique_ptr<ArrowArrayWrapper>> arrays;
	if (state == State::PENDING) {
		RetainClaimed(check);
	}
	if (state == State::RETAINED && retained_fetched) {
		while (auto array = handle->Fetch<ArrowFormat>()) {
			arrays.push_back(std::move(array));
		}
		state = State::DRAINED;
		return arrays;
	}
	if (state == State::RETAINED) {
		auto collection = handle->TakeCollection<ArrowFormat>();
		state = State::DRAINED;
		arrays.reserve(collection->size());
		for (auto &owner : *collection) {
			arrays.push_back(ArrowFormat::ShareArray(owner));
		}
		return arrays;
	}
	if (state == State::STREAMING) {
		while (auto array = FetchStreamUnit<ArrowFormat>(check)) {
			arrays.push_back(std::move(array));
		}
	}
	return arrays;
}

void Result::CopyArrowSchema(ArrowSchema &out) const {
	if (!format.IsArrow()) {
		throw InternalException("engine::Result::CopyArrowSchema on a result that is not in the Arrow format");
	}
	RequireMetadata();
	out.release = nullptr;
	if (duckdb_nanoarrow::ArrowSchemaDeepCopy(&arrow_schema.arrow_schema, &out) != NANOARROW_OK) {
		throw OutOfMemoryException("Failed to copy the Arrow schema of the query result");
	}
}

void Result::BuildArrowSchema(const vector<string> &schema_names, ArrowSchema &out) const {
	auto context = TryGetContext();
	auto properties = client_properties;
	ConnectionClaim claim(*connection_state, *context);
	context->RunFunctionInTransaction(
	    [&]() { ArrowConverter::ToArrowSchema(&out, Types(), schema_names, properties); });
}

string Result::ToBox(BoxRendererContext &context, const BoxRendererConfig &config) {
	ThrowIfFailed();
	if (state != State::RETAINED) {
		throw InternalException("engine::Result::ToBox on a result that is not retained");
	}
	return handle->ToBox(context, config);
}

int64_t Result::ChangedRows() {
	ThrowIfFailed();
	if (state != State::RETAINED || Properties().return_type != StatementReturnType::CHANGED_ROWS) {
		return -1;
	}
	auto &collection = handle->Collection<ChunkFormat>();
	if (collection.Count() == 0) {
		return -1;
	}
	return collection.GetValue(0, 0).GetValue<int64_t>();
}

bool Result::Running() const {
	lock_guard<mutex> guard(handle_lock);
	if (state == State::PENDING) {
		return handle && handle->IsOpen();
	}
	if (state == State::STREAMING) {
		return stream && stream->IsOpen();
	}
	return false;
}

void Result::Close() {
	if (state == State::CLOSED) {
		return;
	}
	{
		lock_guard<mutex> guard(handle_lock);
		if (stream) {
			stream->Close();
		} else if (handle) {
			handle->Close();
		}
	}
	state = State::CLOSED;
}

} // namespace engine
} // namespace duckdb

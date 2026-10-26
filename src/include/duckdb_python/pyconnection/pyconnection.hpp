//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pyconnection/pyconnection.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/import_cache/python_import_cache.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/pyappender.hpp"
#include "duckdb_python/engine/connection.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb_python/path_like.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_reader_options.hpp"
#include "duckdb_python/pyfilesystem.hpp"
#include "duckdb_python/registered_py_object.hpp"
#include "duckdb_python/python_dependency.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb_python/nb/conversions/exception_handling_enum.hpp"
#include "duckdb_python/nb/conversions/python_udf_type_enum.hpp"
#include "duckdb/common/shared_ptr.hpp"

namespace duckdb {
struct BoundParameterData;

enum class PythonEnvironmentType { NORMAL, INTERACTIVE, JUPYTER };

class SourceHandle;
class SourceIdentity;

//! Parameter values as execute() binds them: a list to ?, $1 and so on, a dict to $name
identifier_map_t<BoundParameterData> TransformPreparedParameters(ClientContext &context, const nb::object &params,
                                                                 optional_ptr<PreparedStatement> prep = {});

struct DefaultConnectionHolder {
public:
	DefaultConnectionHolder() {
	}
	~DefaultConnectionHolder() {
		if (connection && (!nb::is_alive() || !PyGILState_Check())) {
			// Can't free Python objects here; static destruction outlives the interpreter, and
			// nb::is_alive() reports true when Py_Finalize is skipped (hence the GIL check).
			new std::shared_ptr<DuckDBPyConnection>(std::move(connection)); // NOLINT: deliberate leak
		}
	}

public:
	DefaultConnectionHolder(const DefaultConnectionHolder &other) = delete;
	DefaultConnectionHolder(DefaultConnectionHolder &&other) = delete;
	DefaultConnectionHolder &operator=(const DefaultConnectionHolder &other) = delete;
	DefaultConnectionHolder &operator=(DefaultConnectionHolder &&other) = delete;

public:
	std::shared_ptr<DuckDBPyConnection> Get();
	void Set(std::shared_ptr<DuckDBPyConnection> conn);
	//! Once the module's cleanup ran and the interpreter is finalizing, the holder neither opens nor keeps a
	//! connection, because nothing would close it
	void Release();

private:
	std::shared_ptr<DuckDBPyConnection> connection;
	bool released = false;
	mutex l;
};

//! The result of the last execute(), read by the connection's fetch methods
class DbapiResult {
public:
	explicit DbapiResult(std::shared_ptr<DuckDBPyResult> result);

public:
	nb::list Description() const;
	int64_t Rowcount() const {
		return changed_rows;
	}
	Optional<nb::tuple> FetchOne();
	nb::list FetchMany(idx_t size);
	nb::list FetchAll();
	nb::dict FetchNumpy();
	PandasDataFrame FetchDF(bool date_as_object);
	PandasDataFrame FetchDFChunk(idx_t vectors_per_chunk, bool date_as_object);
	nb::dict FetchPyTorch();
	nb::dict FetchTF();
	void Complete();
	void Abort();
	//! Makes way for the next statement: one that may write runs to its end, so its effects are not lost; one
	//! that reads is cancelled
	void Supersede();
	//! Every row the result still holds, under names, which leaves it read out as a consuming fetch does; null
	//! when the rows were used up already. The rows are gone even when the take fails, since it ran the
	//! statement to its end or ended it.
	unique_ptr<ColumnDataCollection> TakeRows(vector<Identifier> &names);
	shared_ptr<engine::Result> EngineResult() const {
		return result ? result->EngineResult() : nullptr;
	}

private:
	//! Null once a consuming fetch, complete() or abort() has used the rows up
	std::shared_ptr<DuckDBPyResult> result;
	vector<string> names;
	vector<LogicalType> types;
	int64_t changed_rows;
};

struct ConnectionGuard {
public:
	ConnectionGuard() {
	}
	~ConnectionGuard() {
	}

public:
	DuckDB &GetDatabase() {
		if (!database) {
			ThrowConnectionException();
		}
		return *database;
	}
	const DuckDB &GetDatabase() const {
		if (!database) {
			ThrowConnectionException();
		}
		return *database;
	}
	shared_ptr<DuckDB> ShareDatabase() const {
		if (!database) {
			ThrowConnectionException();
		}
		return database;
	}
	//! Refuses while engine work runs on the connection, so a call from inside that work errors instead of waiting
	Connection &GetConnection() {
		ThrowIfBusy();
		return GetConnectionWhileBusy();
	}
	//! For the calls that must reach a connection whose query is running: interrupting it, reading its progress
	Connection &GetConnectionWhileBusy() {
		if (!connection) {
			ThrowConnectionException();
		}
		return *connection;
	}
	void ThrowIfBusy() const {
		if (state && connection) {
			state->ThrowIfBusy(*connection->context);
		}
	}

	bool ConnectionIsClosed() const {
		return connection == nullptr;
	}
	//! The client state of the open connection
	engine::ConnectionState &GetState() {
		if (!connection || !state) {
			ThrowConnectionException();
		}
		return *state;
	}
	//! The sources of the open connection's database, which every connection of it shares
	engine::SourceRegistry &GetSources() {
		if (!connection || !sources) {
			ThrowConnectionException();
		}
		return *sources;
	}

	const Connection &GetConnection() const {
		ThrowIfBusy();
		if (!connection) {
			ThrowConnectionException();
		}
		return *connection;
	}
	DbapiResult &GetResult() {
		if (!result) {
			ThrowConnectionException();
		}
		return *result;
	}
	const DbapiResult &GetResult() const {
		if (!result) {
			ThrowConnectionException();
		}
		return *result;
	}

public:
	bool HasResult() const {
		return result != nullptr;
	}

public:
	void SetDatabase(shared_ptr<DuckDB> db) {
		database = std::move(db);
	}
	void SetDatabase(ConnectionGuard &con) {
		if (!con.database) {
			ThrowConnectionException();
		}
		database = con.database;
	}
	void SetConnection(unique_ptr<Connection> con) {
		state = con ? engine::ConnectionState::Get(*con->context) : nullptr;
		// Held, so reading the catalog version costs an atomic load and no lookup
		sources = con ? engine::SourceRegistry::Get(*con->context) : nullptr;
		connection = std::move(con);
	}
	//! What the guard owned, taken out in one step with the GIL held, so a call reading the guard under the GIL sees
	//! either all of it or a closed connection; the caller destroys it, which can take long, without the GIL
	struct Released {
		unique_ptr<Connection> connection;
		shared_ptr<engine::ConnectionState> state;
		shared_ptr<engine::SourceRegistry> sources;
		shared_ptr<DuckDB> database;

		//! The connection first, then the database it reads, as the guard's setters always did
		void Destroy() {
			connection.reset();
			state.reset();
			sources.reset();
			database.reset();
		}
	};
	Released Release() {
		Released released;
		released.connection = std::move(connection);
		released.state = std::move(state);
		released.sources = std::move(sources);
		released.database = std::move(database);
		return released;
	}
	void SetResult(std::unique_ptr<DbapiResult> res) {
		result = std::move(res);
		if (state) {
			state->SetDbapiResult(result ? result->EngineResult() : nullptr);
		}
	}

private:
	void ThrowConnectionException() const {
		throw ConnectionException("Connection already closed!");
	}

private:
	shared_ptr<DuckDB> database;
	unique_ptr<Connection> connection;
	shared_ptr<engine::ConnectionState> state;
	shared_ptr<engine::SourceRegistry> sources;
	std::unique_ptr<DbapiResult> result;
};

struct DuckDBPyConnection : public std::enable_shared_from_this<DuckDBPyConnection> {
private:
	class Cursors {
	public:
		Cursors() {
		}

	public:
		void AddCursor(std::shared_ptr<DuckDBPyConnection> conn);
		void ClearCursors();

	private:
		mutex lock;
		vector<std::weak_ptr<DuckDBPyConnection>> cursors;
	};

public:
	// RAII guard for the connection mutex (see py_connection_lock below). Constructing
	// one releases the GIL while waiting for the mutex and reacquires it before
	// returning, so callers always come out of the constructor with the GIL held
	// and the mutex locked. The mutex is released when the guard goes out of scope.
	// Holding the GIL while blocked on this mutex would deadlock against a thread
	// that holds the mutex and is mid-way through a GIL-releasing native call —
	// see duckdb-python#435.
	class ConnectionLockGuard {
	public:
		explicit ConnectionLockGuard(DuckDBPyConnection &conn) : lock_(conn.py_connection_lock, std::defer_lock) {
			DUCKDB_PY_ASSERT_GIL_HELD();
			// The holder of the mutex may be running a query that is waiting for this very call
			conn.con.ThrowIfBusy();
			nb::gil_scoped_release release;
			lock_.lock();
		}

	private:
		std::unique_lock<std::recursive_mutex> lock_;
	};

	ConnectionGuard con;
	Cursors cursors;
	// Recursive so that the outer lock taken at the top of execute/fetch
	// methods (while still holding the GIL) does not deadlock against the
	// inner lock taken by PrepareQuery / ExecuteInternal /
	// PrepareAndSubmitInternal (after releasing the GIL). Serialises every
	// path that touches `con.result` so concurrent calls on a single
	// DuckDBPyConnection cannot dereference an already-freed result — see
	// duckdb-python#435.
	std::recursive_mutex py_connection_lock;
	//! MemoryFileSystem used to temporarily store file-like objects for reading
	std::shared_ptr<ModifiedMemoryFileSystem> internal_object_filesystem;
	case_insensitive_map_t<unique_ptr<ExternalDependency>> registered_functions;
	//! The names register() succeeded with on this connection, bounded oldest out. Only a hint: unregister() of a
	//! name it holds refuses beside an open DB-API result, and of one it does not hold, such as a view renamed in SQL,
	//! asks the catalog when the connection is free and is a no-op beside an open result
	case_insensitive_set_t ever_registered;
	std::deque<string> ever_registered_order;
	static constexpr idx_t EVER_REGISTERED_MEMORY = 4096;
	void RememberRegistered(const string &name);
	//! Registry entries the connection keeps alive for a view by its name, until unregister() or a replacement
	case_insensitive_map_t<vector<std::shared_ptr<SourceHandle>>> owned_sources;
	//! The sources this connection registered, by the identity of the Python object each was registered for
	std::shared_ptr<SourceIdentity> source_identity;
	//! Drops what the two members above own, with the GIL held
	void ReleaseOwnedSources();
	//! The temporary view register() makes over a source, which owns the source's table reference in the catalog
	void RegisterSourceView(const string &name, const SourceHandle &source);

public:
	explicit DuckDBPyConnection() {
	}
	~DuckDBPyConnection();

public:
	static void Initialize(nb::handle &m);
	static void Cleanup();

	std::shared_ptr<DuckDBPyConnection> Enter();

	static void Exit(DuckDBPyConnection &self, const nb::object &exc_type, const nb::object &exc,
	                 const nb::object &traceback);

	static bool DetectAndGetEnvironment();
	static bool IsJupyter();
	static std::string FormattedPythonVersion();
	static std::shared_ptr<DuckDBPyConnection> DefaultConnection();
	static void SetDefaultConnection(std::shared_ptr<DuckDBPyConnection> conn);
	static PythonImportCache *ImportCache();
	static nb::object ConsumedArrowReaders();
	//! Keeps adapt and the nodes module of the duckdb._sources package the package passes in as it imports
	static void SetArrowSources(const nb::object &sources);
	//! duckdb._sources.adapt; call with the GIL held
	static nb::object ArrowSourcesAdapt();
	//! The duckdb._sources.nodes module; call with the GIL held
	static nb::object ArrowSourcesNodes();
	static bool IsInteractive();

	nb::list ExtractStatements(const string &query);

	std::unique_ptr<DuckDBPyType> MapType(const DuckDBPyType &key_type, const DuckDBPyType &value_type);
	std::unique_ptr<DuckDBPyType> StructType(const nb::object &fields);
	std::unique_ptr<DuckDBPyType> ListType(const DuckDBPyType &type);
	std::unique_ptr<DuckDBPyType> ArrayType(const DuckDBPyType &type, idx_t size);
	std::unique_ptr<DuckDBPyType> UnionType(const nb::object &members);
	std::unique_ptr<DuckDBPyType> EnumType(const string &name, const DuckDBPyType &type, const nb::list &values_p);
	std::unique_ptr<DuckDBPyType> DecimalType(int width, int scale);
	std::unique_ptr<DuckDBPyType> StringType(const string &collation = string());
	std::unique_ptr<DuckDBPyType> Type(const string &type_str);

	std::shared_ptr<DuckDBPyConnection>
	RegisterScalarUDF(const string &name, const nb::callable &udf, const nb::object &arguments = nb::none(),
	                  const nb::object &return_type = nb::none(), PythonUDFType type = PythonUDFType::NATIVE,
	                  FunctionNullHandling null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                  PythonExceptionHandling exception_handling = PythonExceptionHandling::FORWARD_ERROR,
	                  bool side_effects = false);

	std::shared_ptr<DuckDBPyConnection> UnregisterUDF(const string &name);

	std::shared_ptr<DuckDBPyConnection> ExecuteMany(const nb::object &query, nb::object params = nb::list());

	engine::Prepared PrepareQuery(unique_ptr<SQLStatement> statement);
	shared_ptr<engine::Result> ExecuteInternal(engine::Prepared &prep, nb::object params,
	                                           engine::StatementGroup &statements);
	//! Binds the parameters and submits the statement. The result is returned undriven.
	shared_ptr<engine::Result> PrepareAndSubmitInternal(unique_ptr<SQLStatement> statement,
	                                                    nb::object params = nb::list());
	engine::Session EngineSession();

	std::shared_ptr<DuckDBPyConnection> Execute(const nb::object &query, nb::object params = nb::list());
	std::shared_ptr<DuckDBPyConnection> ExecuteFromString(const string &query);

	std::shared_ptr<DuckDBPyConnection> Append(const string &name, const PandasDataFrame &value, bool by_name);

	std::unique_ptr<DuckDBPyAppender> CreateAppender(const string &table, const std::optional<nb::list> &columns,
	                                                 const string &action, const std::optional<nb::list> &key);
	std::unique_ptr<DuckDBPyAppender> CreateQueryAppender(const string &dml, const nb::dict &columns,
	                                                      const std::optional<string> &source);

	std::shared_ptr<DuckDBPyConnection> RegisterPythonObject(const string &name, const nb::object &python_object);
	std::shared_ptr<DuckDBPyConnection> RegisterArrowSource(const string &name, const nb::object &source);

	void InstallExtension(const string &extension, bool force_install = false,
	                      const nb::object &repository = nb::none(), const nb::object &repository_url = nb::none(),
	                      const nb::object &version = nb::none());

	void LoadExtension(const string &extension);

	//! Runs every statement of the query but the last, as RunQuery does, and returns (the kept rows result or None,
	//! the last statement's text or None, its parsed type in lower case or None)
	nb::tuple PrepareScript(const nb::object &query, const nb::object &params, bool with_statement = false);

	unordered_set<string> GetTableNames(const string &query, bool qualified);

	std::shared_ptr<DuckDBPyConnection> UnregisterPythonObject(const string &name);

	std::shared_ptr<DuckDBPyConnection> Begin();

	std::shared_ptr<DuckDBPyConnection> Commit();

	std::shared_ptr<DuckDBPyConnection> Rollback();

	std::shared_ptr<DuckDBPyConnection> Checkpoint();

	void Close();

	void Interrupt();

	double QueryProgress();

	ModifiedMemoryFileSystem &GetObjectFileSystem();

	// cursor() is stupid
	std::shared_ptr<DuckDBPyConnection> Cursor();

	Optional<nb::list> GetDescription();

	int64_t GetRowcount();
	//! Runs the result of execute() to its end without reading its rows
	void Complete();
	//! Cancels the rest of the result of execute()
	void Abort();

	// these should be functions on the result but well
	Optional<nb::tuple> FetchOne();

	nb::list FetchMany(idx_t size);

	nb::list FetchAll();

	nb::dict FetchNumpy();
	PandasDataFrame FetchDF(bool date_as_object);
	PandasDataFrame FetchDFChunk(const idx_t vectors_per_chunk = 1, bool date_as_object = false);

	nb::dict FetchPyTorch();

	nb::dict FetchTF();
	//! The rows of the last execute() result as Arrow, read back through a statement over them on this
	//! connection. reader hands out a stream of batches, otherwise a table. None once the rows are used up.
	nb::object ResultAsArrow(idx_t batch_size, bool reader);

	//! Drops the last execute() result the way the next statement requires; cancel skips the drain
	void SupersedeResult(bool cancel);

	static std::shared_ptr<DuckDBPyConnection> Connect(const nb::object &database, bool read_only,
	                                                   const nb::dict &config);

	static vector<Value> TransformPythonParamList(ClientContext &context, const nb::handle &params);
	static identifier_map_t<BoundParameterData> TransformPythonParamDict(ClientContext &context,
	                                                                     const nb::dict &params);

	// Takes nb::object (not AbstractFileSystem) so the binding can accept None: nanobind's .none() does not bypass a
	// nb::object-subclass wrapper's check_(). The body imports fsspec and validates the instance explicitly.
	void RegisterFilesystem(nb::object filesystem);
	void UnregisterFilesystem(const nb::str &name);
	nb::list ListFilesystems();
	bool FileSystemIsRegistered(const string &name);

	// Profiling info
	nb::str GetProfilingInformation(const string &format = "json");
	void EnableProfiling();
	void DisableProfiling();

	static bool IsPandasDataframe(const nb::object &object);
	static NumpyObjectType IsAcceptedNumpyObject(const nb::object &object);

	//! Abandons a blocking engine call when Python has a pending signal. Takes the GIL.
	static void CheckSignals();

private:
	PathLike GetPathLike(const nb::object &object);
	ScalarFunction CreateScalarUDF(const string &name, const nb::callable &udf, const nb::object &parameters,
	                               const nb::object &return_type, bool vectorized, FunctionNullHandling null_handling,
	                               PythonExceptionHandling exception_handling, bool side_effects);
	engine::StatementGroup GetStatements(const nb::object &query);
	//! Runs every statement of the group but the last, after making way for them first when supersede, and returns
	//! the last, unrun; null when none is left. Releases the GIL for the engine work.
	unique_ptr<SQLStatement> RunAllButLast(engine::StatementGroup &statements, bool supersede);

	static void DetectEnvironment();
};

template <typename T>
static bool ModuleIsLoaded() {
	auto dict = nb::cast<nb::dict>(nb::module_::import_("sys").attr("modules"));
	return dict.contains(nb::str(T::Name));
}

} // namespace duckdb

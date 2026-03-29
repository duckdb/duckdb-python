//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pyconnection/pyconnection.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/import_cache/python_import_cache.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/pyrelation.hpp"
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

struct DuckDBPyRelation;

class RegisteredArrow : public RegisteredObject {

public:
	RegisteredArrow(unique_ptr<PythonTableArrowArrayStreamFactory> arrow_factory_p, nb::object obj_p)
	    : RegisteredObject(std::move(obj_p)), arrow_factory(std::move(arrow_factory_p)) {};
	unique_ptr<PythonTableArrowArrayStreamFactory> arrow_factory;
};

struct DefaultConnectionHolder {
public:
	DefaultConnectionHolder() {
	}
	~DefaultConnectionHolder() {
	}

public:
	DefaultConnectionHolder(const DefaultConnectionHolder &other) = delete;
	DefaultConnectionHolder(DefaultConnectionHolder &&other) = delete;
	DefaultConnectionHolder &operator=(const DefaultConnectionHolder &other) = delete;
	DefaultConnectionHolder &operator=(DefaultConnectionHolder &&other) = delete;

public:
	std::shared_ptr<DuckDBPyConnection> Get();
	void Set(std::shared_ptr<DuckDBPyConnection> conn);

private:
	std::shared_ptr<DuckDBPyConnection> connection;
	mutex l;
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
	Connection &GetConnection() {
		if (!connection) {
			ThrowConnectionException();
		}
		return *connection;
	}

	bool ConnectionIsClosed() const {
		return connection == nullptr;
	}

	const Connection &GetConnection() const {
		if (!connection) {
			ThrowConnectionException();
		}
		return *connection;
	}
	DuckDBPyRelation &GetResult() {
		if (!result) {
			ThrowConnectionException();
		}
		return *result;
	}
	const DuckDBPyRelation &GetResult() const {
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
		connection = std::move(con);
	}
	void SetResult(std::unique_ptr<DuckDBPyRelation> res) {
		result = std::move(res);
	}

private:
	void ThrowConnectionException() const {
		throw ConnectionException("Connection already closed!");
	}

private:
	shared_ptr<DuckDB> database;
	unique_ptr<Connection> connection;
	std::unique_ptr<DuckDBPyRelation> result;
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
			D_ASSERT(duckdb::PyUtil::GilCheck());
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
	// PrepareAndExecuteInternal (after releasing the GIL). Serialises every
	// path that touches `con.result` so concurrent calls on a single
	// DuckDBPyConnection cannot dereference an already-freed result — see
	// duckdb-python#435.
	std::recursive_mutex py_connection_lock;
	//! MemoryFileSystem used to temporarily store file-like objects for reading
	std::shared_ptr<ModifiedMemoryFileSystem> internal_object_filesystem;
	case_insensitive_map_t<unique_ptr<ExternalDependency>> registered_functions;
	case_insensitive_set_t registered_objects;

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
	static bool IsInteractive();

	std::unique_ptr<DuckDBPyRelation> ReadCSV(const nb::object &name, nb::kwargs &kwargs);

	nb::list ExtractStatements(const string &query);

	std::unique_ptr<DuckDBPyRelation> ReadJSON(
	    const nb::object &name, const Optional<nb::object> &columns = nb::none(),
	    const Optional<nb::object> &sample_size = nb::none(), const Optional<nb::object> &maximum_depth = nb::none(),
	    const Optional<nb::str> &records = nb::none(), const Optional<nb::str> &format = nb::none(),
	    const Optional<nb::object> &date_format = nb::none(), const Optional<nb::object> &timestamp_format = nb::none(),
	    const Optional<nb::object> &compression = nb::none(),
	    const Optional<nb::object> &maximum_object_size = nb::none(),
	    const Optional<nb::object> &ignore_errors = nb::none(),
	    const Optional<nb::object> &convert_strings_to_integers = nb::none(),
	    const Optional<nb::object> &field_appearance_threshold = nb::none(),
	    const Optional<nb::object> &map_inference_threshold = nb::none(),
	    const Optional<nb::object> &maximum_sample_files = nb::none(),
	    const Optional<nb::object> &filename = nb::none(), const Optional<nb::object> &hive_partitioning = nb::none(),
	    const Optional<nb::object> &union_by_name = nb::none(), const Optional<nb::object> &hive_types = nb::none(),
	    const Optional<nb::object> &hive_types_autocast = nb::none());

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

	void ExecuteImmediately(vector<unique_ptr<SQLStatement>> statements);
	unique_ptr<PreparedStatement> PrepareQuery(unique_ptr<SQLStatement> statement);
	unique_ptr<QueryResult> ExecuteInternal(PreparedStatement &prep, nb::object params = nb::list());
	unique_ptr<QueryResult> PrepareAndExecuteInternal(unique_ptr<SQLStatement> statement,
	                                                  nb::object params = nb::list());

	std::shared_ptr<DuckDBPyConnection> Execute(const nb::object &query, nb::object params = nb::list());
	std::shared_ptr<DuckDBPyConnection> ExecuteFromString(const string &query);

	std::shared_ptr<DuckDBPyConnection> Append(const string &name, const PandasDataFrame &value, bool by_name);

	std::shared_ptr<DuckDBPyConnection> RegisterPythonObject(const string &name, const nb::object &python_object);

	void InstallExtension(const string &extension, bool force_install = false,
	                      const nb::object &repository = nb::none(), const nb::object &repository_url = nb::none(),
	                      const nb::object &version = nb::none());

	void LoadExtension(const string &extension);

	std::unique_ptr<DuckDBPyRelation> RunQuery(const nb::object &query, string alias = "",
	                                           nb::object params = nb::list());

	std::unique_ptr<DuckDBPyRelation> Table(const string &tname);

	std::unique_ptr<DuckDBPyRelation> Values(const nb::args &params);

	std::unique_ptr<DuckDBPyRelation> View(const string &vname);

	std::unique_ptr<DuckDBPyRelation> TableFunction(const string &fname, nb::object params = nb::list());

	std::unique_ptr<DuckDBPyRelation> FromDF(const PandasDataFrame &value);

	std::unique_ptr<DuckDBPyRelation> FromParquet(const nb::object &path_or_buffer, bool binary_as_string,
	                                              bool file_row_number, bool filename, bool hive_partitioning,
	                                              bool union_by_name, const nb::object &compression = nb::none());

	std::unique_ptr<DuckDBPyRelation> FromArrow(nb::object &arrow_object);

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

	int GetRowcount();

	// these should be functions on the result but well
	Optional<nb::tuple> FetchOne();

	nb::list FetchMany(idx_t size);

	nb::list FetchAll();

	nb::dict FetchNumpy();
	PandasDataFrame FetchDF(bool date_as_object);
	PandasDataFrame FetchDFChunk(const idx_t vectors_per_chunk = 1, bool date_as_object = false);

	duckdb::pyarrow::Table FetchArrow(idx_t rows_per_batch);
	PolarsDataFrame FetchPolars(idx_t rows_per_batch, bool lazy);

	nb::dict FetchPyTorch();

	nb::dict FetchTF();

	duckdb::pyarrow::RecordBatchReader FetchRecordBatchReader(const idx_t rows_per_batch);

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
	static PyArrowObjectType GetArrowType(const nb::handle &obj);
	static bool IsAcceptedArrowObject(const nb::object &object);
	static NumpyObjectType IsAcceptedNumpyObject(const nb::object &object);

	static unique_ptr<QueryResult> CompletePendingQuery(PendingQueryResult &pending_query);

private:
	std::unique_ptr<DuckDBPyRelation> CreateRelation(shared_ptr<Relation> rel);
	std::unique_ptr<DuckDBPyRelation> CreateRelation(std::shared_ptr<DuckDBPyResult> result);
	PathLike GetPathLike(const nb::object &object);
	ScalarFunction CreateScalarUDF(const string &name, const nb::callable &udf, const nb::object &parameters,
	                               const nb::object &return_type, bool vectorized, FunctionNullHandling null_handling,
	                               PythonExceptionHandling exception_handling, bool side_effects);
	pair<string, nb::object> ExtractCompiledSqlAndParams(const nb::object &query, nb::object params);
	vector<unique_ptr<SQLStatement>> GetStatements(const nb::object &query);

	static void DetectEnvironment();
};

template <typename T>
static bool ModuleIsLoaded() {
	auto dict = nb::cast<nb::dict>(nb::module_::import_("sys").attr("modules"));
	return dict.contains(nb::str(T::Name));
}

} // namespace duckdb

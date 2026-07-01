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
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb_python/import_cache/python_import_cache.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb_python/path_like.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_reader_options.hpp"
#include "duckdb_python/pyfilesystem.hpp"
#include "duckdb_python/pybind11/registered_py_object.hpp"
#include "duckdb_python/python_dependency.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb_python/pybind11/conversions/exception_handling_enum.hpp"
#include "duckdb_python/pybind11/conversions/python_udf_type_enum.hpp"
#include "duckdb/common/shared_ptr.hpp"

namespace duckdb {
struct BoundParameterData;

enum class PythonEnvironmentType { NORMAL, INTERACTIVE, JUPYTER };

struct DuckDBPyRelation;

class RegisteredArrow : public RegisteredObject {

public:
	RegisteredArrow(unique_ptr<PythonTableArrowArrayStreamFactory> arrow_factory_p, py::object obj_p)
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
			D_ASSERT(py::gil_check());
			py::gil_scoped_release release;
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
	static void Initialize(py::handle &m);
	static void Cleanup();

	std::shared_ptr<DuckDBPyConnection> Enter();

	static void Exit(DuckDBPyConnection &self, const py::object &exc_type, const py::object &exc,
	                 const py::object &traceback);

	static bool DetectAndGetEnvironment();
	static bool IsJupyter();
	static std::string FormattedPythonVersion();
	static std::shared_ptr<DuckDBPyConnection> DefaultConnection();
	static void SetDefaultConnection(std::shared_ptr<DuckDBPyConnection> conn);
	static PythonImportCache *ImportCache();
	static bool IsInteractive();

	std::unique_ptr<DuckDBPyRelation> ReadCSV(const py::object &name, py::kwargs &kwargs);

	py::list ExtractStatements(const string &query);

	std::unique_ptr<DuckDBPyRelation> ReadJSON(
	    const py::object &name, const Optional<py::object> &columns = py::none(),
	    const Optional<py::object> &sample_size = py::none(), const Optional<py::object> &maximum_depth = py::none(),
	    const Optional<py::str> &records = py::none(), const Optional<py::str> &format = py::none(),
	    const Optional<py::object> &date_format = py::none(), const Optional<py::object> &timestamp_format = py::none(),
	    const Optional<py::object> &compression = py::none(),
	    const Optional<py::object> &maximum_object_size = py::none(),
	    const Optional<py::object> &ignore_errors = py::none(),
	    const Optional<py::object> &convert_strings_to_integers = py::none(),
	    const Optional<py::object> &field_appearance_threshold = py::none(),
	    const Optional<py::object> &map_inference_threshold = py::none(),
	    const Optional<py::object> &maximum_sample_files = py::none(),
	    const Optional<py::object> &filename = py::none(), const Optional<py::object> &hive_partitioning = py::none(),
	    const Optional<py::object> &union_by_name = py::none(), const Optional<py::object> &hive_types = py::none(),
	    const Optional<py::object> &hive_types_autocast = py::none());

	std::shared_ptr<DuckDBPyType> MapType(const std::shared_ptr<DuckDBPyType> &key_type,
	                                      const std::shared_ptr<DuckDBPyType> &value_type);
	std::shared_ptr<DuckDBPyType> StructType(const py::object &fields);
	std::shared_ptr<DuckDBPyType> ListType(const std::shared_ptr<DuckDBPyType> &type);
	std::shared_ptr<DuckDBPyType> ArrayType(const std::shared_ptr<DuckDBPyType> &type, idx_t size);
	std::shared_ptr<DuckDBPyType> UnionType(const py::object &members);
	std::shared_ptr<DuckDBPyType> EnumType(const string &name, const std::shared_ptr<DuckDBPyType> &type,
	                                       const py::list &values_p);
	std::shared_ptr<DuckDBPyType> DecimalType(int width, int scale);
	std::shared_ptr<DuckDBPyType> StringType(const string &collation = string());
	std::shared_ptr<DuckDBPyType> Type(const string &type_str);

	std::shared_ptr<DuckDBPyConnection> RegisterScalarUDF(
	    const string &name, const py::function &udf, const py::object &arguments = py::none(),
	    const std::shared_ptr<DuckDBPyType> &return_type = nullptr, PythonUDFType type = PythonUDFType::NATIVE,
	    FunctionNullHandling null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PythonExceptionHandling exception_handling = PythonExceptionHandling::FORWARD_ERROR, bool side_effects = false);

	std::shared_ptr<DuckDBPyConnection> UnregisterUDF(const string &name);

	std::shared_ptr<DuckDBPyConnection> ExecuteMany(const py::object &query, py::object params = py::list());

	void ExecuteImmediately(vector<unique_ptr<SQLStatement>> statements);
	unique_ptr<PreparedStatement> PrepareQuery(unique_ptr<SQLStatement> statement);
	unique_ptr<QueryResult> ExecuteInternal(PreparedStatement &prep, py::object params = py::list());
	unique_ptr<QueryResult> PrepareAndExecuteInternal(unique_ptr<SQLStatement> statement,
	                                                  py::object params = py::list());

	std::shared_ptr<DuckDBPyConnection> Execute(const py::object &query, py::object params = py::list());
	std::shared_ptr<DuckDBPyConnection> ExecuteFromString(const string &query);

	std::shared_ptr<DuckDBPyConnection> Append(const string &name, const PandasDataFrame &value, bool by_name);

	std::shared_ptr<DuckDBPyConnection> RegisterPythonObject(const string &name, const py::object &python_object);

	void InstallExtension(const string &extension, bool force_install = false,
	                      const py::object &repository = py::none(), const py::object &repository_url = py::none(),
	                      const py::object &version = py::none());

	void LoadExtension(const string &extension);

	std::unique_ptr<DuckDBPyRelation> RunQuery(const py::object &query, string alias = "",
	                                           py::object params = py::list());

	std::unique_ptr<DuckDBPyRelation> Table(const string &tname);

	std::unique_ptr<DuckDBPyRelation> Values(const py::args &params);

	std::unique_ptr<DuckDBPyRelation> View(const string &vname);

	std::unique_ptr<DuckDBPyRelation> TableFunction(const string &fname, py::object params = py::list());

	std::unique_ptr<DuckDBPyRelation> FromDF(const PandasDataFrame &value);

	std::unique_ptr<DuckDBPyRelation> FromParquet(const py::object &path_or_buffer, bool binary_as_string,
	                                              bool file_row_number, bool filename, bool hive_partitioning,
	                                              bool union_by_name, const py::object &compression = py::none());

	std::unique_ptr<DuckDBPyRelation> FromArrow(py::object &arrow_object);

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

	Optional<py::list> GetDescription();

	int GetRowcount();

	// these should be functions on the result but well
	Optional<py::tuple> FetchOne();

	py::list FetchMany(idx_t size);

	py::list FetchAll();

	py::dict FetchNumpy();
	PandasDataFrame FetchDF(bool date_as_object);
	PandasDataFrame FetchDFChunk(const idx_t vectors_per_chunk = 1, bool date_as_object = false);

	duckdb::pyarrow::Table FetchArrow(idx_t rows_per_batch);
	PolarsDataFrame FetchPolars(idx_t rows_per_batch, bool lazy);

	py::dict FetchPyTorch();

	py::dict FetchTF();

	duckdb::pyarrow::RecordBatchReader FetchRecordBatchReader(const idx_t rows_per_batch);

	static std::shared_ptr<DuckDBPyConnection> Connect(const py::object &database, bool read_only,
	                                                   const py::dict &config);

	static vector<Value> TransformPythonParamList(ClientContext &context, const py::handle &params);
	static identifier_map_t<BoundParameterData> TransformPythonParamDict(ClientContext &context,
	                                                                     const py::dict &params);

	void RegisterFilesystem(AbstractFileSystem filesystem);
	void UnregisterFilesystem(const py::str &name);
	py::list ListFilesystems();
	bool FileSystemIsRegistered(const string &name);

	// Profiling info
	py::str GetProfilingInformation(const string &format = "json");
	void EnableProfiling();
	void DisableProfiling();

	static bool IsPandasDataframe(const py::object &object);
	static PyArrowObjectType GetArrowType(const py::handle &obj);
	static bool IsAcceptedArrowObject(const py::object &object);
	static NumpyObjectType IsAcceptedNumpyObject(const py::object &object);

	static unique_ptr<QueryResult> CompletePendingQuery(PendingQueryResult &pending_query);

private:
	std::unique_ptr<DuckDBPyRelation> CreateRelation(shared_ptr<Relation> rel);
	std::unique_ptr<DuckDBPyRelation> CreateRelation(std::shared_ptr<DuckDBPyResult> result);
	PathLike GetPathLike(const py::object &object);
	ScalarFunction CreateScalarUDF(const string &name, const py::function &udf, const py::object &parameters,
	                               const std::shared_ptr<DuckDBPyType> &return_type, bool vectorized,
	                               FunctionNullHandling null_handling, PythonExceptionHandling exception_handling,
	                               bool side_effects);
	vector<unique_ptr<SQLStatement>> GetStatements(const py::object &query);

	static void DetectEnvironment();
};

template <typename T>
static bool ModuleIsLoaded() {
	auto dict = pybind11::module_::import("sys").attr("modules");
	return dict.contains(py::str(T::Name));
}

} // namespace duckdb

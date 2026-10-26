#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/engine/connection.hpp"
#include "duckdb_python/registered_py_object.hpp"
#include "duckdb/main/external_dependencies.hpp"

namespace duckdb {

struct DuckDBPyConnection;
class SourceIdentity;

//! Keeps a registered source, or a file on a connection's object store, for as long as it lives
class SourceHandle {
public:
	SourceHandle(string name, weak_ptr<engine::SourceRegistry> registry, shared_ptr<ExternalDependency> keep_alive);
	~SourceHandle();

	SourceHandle(const SourceHandle &) = delete;
	SourceHandle &operator=(const SourceHandle &) = delete;

public:
	const string &Name() const {
		return name;
	}
	//! Records which Python object the source was registered for, so registering that object again finds this handle
	void RememberFor(const std::shared_ptr<SourceIdentity> &identity_p, const nb::handle &object);

private:
	string name;
	//! Weak, so a closed database frees its sources even while handles to them live on
	weak_ptr<engine::SourceRegistry> registry;
	shared_ptr<ExternalDependency> keep_alive;
	//! Weak, so a handle never keeps the connection that registered the source alive
	std::weak_ptr<SourceIdentity> identity;
	//! A map key only, never dereferenced
	PyObject *object_id = nullptr;
};

//! The sources a connection registered, by the identity of the Python object each was registered for, so that object
//! reads as one source however often it is registered. Every call needs the GIL, and only a call under the GIL ever
//! takes its lock.
class SourceIdentity {
public:
	//! The handle of a live source registered for this very object, or null when there is none
	std::shared_ptr<SourceHandle> Lookup(const nb::handle &object);
	void Remember(const nb::handle &object, const std::shared_ptr<SourceHandle> &handle);
	//! Drops what Remember kept, unless the object was registered again since
	void Forget(PyObject *object_id, const string &name);

private:
	struct Entry {
		string name;
		std::weak_ptr<SourceHandle> handle;
		//! A weak reference where the type allows one, so a source never keeps a relation and its connection alive,
		//! and a strong one otherwise, which the source's own scan holds anyway
		unique_ptr<PyObjectHolder> reference;
		bool weak;
	};

	//! Whether the entry's object is alive and still the one the address names
	static bool NamesLiveObject(const Entry &entry, PyObject *object_id);

private:
	mutex lock;
	unordered_map<PyObject *, Entry> entries;
};

//! The private methods a Python relation layer builds on: binding, submission and the source registry
struct DuckDBPySeam {
	static void Initialize(nb::module_ &m);
	static void InitializeConnectionMethods(nb::class_<DuckDBPyConnection> &connection);
	//! Registers obj under a generated name; subject names the object in errors, an empty one makes it nameless
	static std::shared_ptr<SourceHandle> RegisterSource(DuckDBPyConnection &connection, const nb::object &obj,
	                                                    const string &subject);
	//! Registers a table reference built in C++ under a generated name
	static std::shared_ptr<SourceHandle> RegisterTableRef(ClientContext &context, unique_ptr<TableRef> table,
	                                                      const string &subject);
	//! The tag register() puts on its views, which marks a temporary view as one it may replace and drop
	static constexpr const char *REGISTER_TAG = "duckdb_python_register";
	//! A CREATE VIEW statement for the SELECT in sql, with every registry name it reads written in as a copy of the
	//! table reference it names, so the catalog entry owns what the view reads and the transaction decides its life.
	//! Call without the GIL.
	static unique_ptr<SQLStatement> ViewStatement(ClientContext &context, const QualifiedName &name, const string &sql,
	                                              bool temporary, bool replace, bool registered);
	//! Whether the temporary view of this name is one register() made. Call under the claim, without the GIL.
	static bool IsRegisteredView(ClientContext &context, const string &name);
};

} // namespace duckdb

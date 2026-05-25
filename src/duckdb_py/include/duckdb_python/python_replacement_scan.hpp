#pragma once

#include "duckdb/main/client_context_state.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb_python/python_dependency.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

class PythonRegisteredObjectState : public ClientContextState {
public:
	static constexpr const char *Key = "python_registered_objects";

	void Register(const string &name, const py::object &object);
	void Unregister(const string &name);
	py::object Get(const string &name);
	bool Contains(const string &name);

private:
	mutex lock;
	case_insensitive_map_t<shared_ptr<DependencyItem>> registered_objects;
};

struct PythonReplacementScan {
public:
	static unique_ptr<TableRef> Replace(ClientContext &context, ReplacementScanInput &input,
	                                    optional_ptr<ReplacementScanData> data);
	//! Try to perform a replacement, returns NULL on error
	static unique_ptr<TableRef> TryReplacementObject(const py::object &entry, const string &name,
	                                                 ClientContext &context, bool relation = false);
	//! Perform a replacement or throw if it failed
	static unique_ptr<TableRef> ReplacementObject(const py::object &entry, const string &name, ClientContext &context,
	                                              bool relation = false);
};

} // namespace duckdb

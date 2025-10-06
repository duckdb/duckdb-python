//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/module_state.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb_python/module_state.hpp"
#include <stdexcept>
#include <chrono>
#include <thread>

#define DEBUG_MODULE_STATE 0

namespace duckdb {

// Forward declaration from pyconnection.cpp
void InstantiateNewInstance(DuckDB &db);

// Static member initialization - required for all static class members in C++
DuckDBPyModuleState *DuckDBPyModuleState::g_module_state = nullptr;

DuckDBPyModuleState::DuckDBPyModuleState() {
	// Caches are constructed as direct objects - no heap allocation needed

#ifdef Py_GIL_DISABLED
	// Initialize lock object for critical sections
	// TODO: Consider moving to finer-grained locks
	default_con_lock = py::none();
#endif

	// Detects Python environment and version during intialization
	// Moved from DuckDBPyConnection::DetectEnvironment()
	py::module_ sys = py::module_::import("sys");
	py::object version_info = sys.attr("version_info");
	int major = py::cast<int>(version_info.attr("major"));
	int minor = py::cast<int>(version_info.attr("minor"));
	formatted_python_version = std::to_string(major) + "." + std::to_string(minor);

	// If __main__ does not have a __file__ attribute, we are in interactive mode
	auto main_module = py::module_::import("__main__");
	if (!py::hasattr(main_module, "__file__")) {
		environment = PythonEnvironmentType::INTERACTIVE;

		if (ModuleIsLoaded<IpythonCacheItem>()) {
			// Check to see if we are in a Jupyter Notebook
			auto get_ipython = import_cache.IPython.get_ipython();
			if (get_ipython.ptr() != nullptr) {
				auto ipython = get_ipython();
				if (py::hasattr(ipython, "config")) {
					py::dict ipython_config = ipython.attr("config");
					if (ipython_config.contains("IPKernelApp")) {
						environment = PythonEnvironmentType::JUPYTER;
					}
				}
			}
		}
	}
}

DuckDBPyModuleState &DuckDBPyModuleState::GetGlobalModuleState() {
	// TODO: Externalize this static cache when adding multi-interpreter support
	// For now, single interpreter assumption allows simple static caching
	if (!g_module_state) {
		throw InternalException("Module state not initialized - call SetGlobalModuleState() during module init");
	}
	return *g_module_state;
}

void DuckDBPyModuleState::SetGlobalModuleState(DuckDBPyModuleState *state) {
#if DEBUG_MODULE_STATE
	printf("DEBUG: SetGlobalModuleState() called - initializing static cache (built: %s %s)\n", __DATE__, __TIME__);
#endif
	g_module_state = state;
}

DuckDBPyModuleState &GetModuleState() {
#if DEBUG_MODULE_STATE
	printf("DEBUG: GetModuleState() called\n");
#endif
	return DuckDBPyModuleState::GetGlobalModuleState();
}

void SetModuleState(DuckDBPyModuleState *state) {
	DuckDBPyModuleState::SetGlobalModuleState(state);
}

shared_ptr<DuckDBPyConnection> DuckDBPyModuleState::GetDefaultConnection() {
#if defined(Py_GIL_DISABLED)
	// TODO: Consider whether a mutex vs a scoped_critical_section
	py::scoped_critical_section guard(default_con_lock);
#endif
	// Reproduce exact logic from original DefaultConnectionHolder::Get()
	if (!default_connection_ptr || default_connection_ptr->con.ConnectionIsClosed()) {
		py::dict config_dict;
		default_connection_ptr = DuckDBPyConnection::Connect(py::str(":memory:"), false, config_dict);
	}
	return default_connection_ptr;
}

void DuckDBPyModuleState::SetDefaultConnection(shared_ptr<DuckDBPyConnection> connection) {
#if defined(Py_GIL_DISABLED)
	py::scoped_critical_section guard(default_con_lock);
#endif
	default_connection_ptr = std::move(connection);
}

void DuckDBPyModuleState::ClearDefaultConnection() {
#if defined(Py_GIL_DISABLED)
	py::scoped_critical_section guard(default_con_lock);
#endif
	default_connection_ptr = nullptr;
}

PythonImportCache *DuckDBPyModuleState::GetImportCache() {
	return &import_cache;
}

void DuckDBPyModuleState::ClearImportCache() {
	import_cache = PythonImportCache();
}

DBInstanceCache *DuckDBPyModuleState::GetInstanceCache() {
	return &instance_cache;
}

} // namespace duckdb
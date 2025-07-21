cmake_minimum_required(VERSION 3.29)

project(duckdb_py LANGUAGES CXX)

# Always use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the library name
set(DUCKDB_PYTHON_LIB_NAME "_duckdb")

# Detect CCache
include(cmake/compiler_launcher.cmake)
setup_compiler_launcher_if_available()

# ────────────────────────────────────────────
# IDE support
# ────────────────────────────────────────────
# Create compile_commands.json for IntelliSense and clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ────────────────────────────────────────────
# Policy hygiene
# ────────────────────────────────────────────
if(POLICY CMP0148) # Disallow FindPythonLibs
  cmake_policy(SET CMP0148 NEW)
endif()

if(POLICY CMP0003) # No implicit link directories
  cmake_policy(SET CMP0003 NEW)
endif()

# ────────────────────────────────────────────
# Dependencies
# ────────────────────────────────────────────
# PyBind11
find_package(pybind11 REQUIRED CONFIG)

# DuckDB
include(cmake/duckdb_loader.cmake)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  duckdb_configure_for_debug()
else()
  duckdb_configure_for_release()
endif()
duckdb_add_library(duckdb_target)

# Bundle in INTERFACE library
add_library(_duckdb_dependencies INTERFACE)
target_link_libraries(_duckdb_dependencies INTERFACE
        pybind11::pybind11
        duckdb_target
)
# Also add include directory
target_include_directories(_duckdb_dependencies INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/duckdb_py/include>
)

# ────────────────────────────────────────────
# Descend into the real DuckDB‑Python sources
# ────────────────────────────────────────────
add_subdirectory(src/duckdb_py)

pybind11_add_module(_duckdb
        $<TARGET_OBJECTS:python_src>
        $<TARGET_OBJECTS:python_arrow>
        $<TARGET_OBJECTS:python_common>
        $<TARGET_OBJECTS:python_functional>
        $<TARGET_OBJECTS:python_jupyter>
        $<TARGET_OBJECTS:python_native>
        $<TARGET_OBJECTS:python_numpy>
        $<TARGET_OBJECTS:python_pandas>
        $<TARGET_OBJECTS:python_pybind11>
        $<TARGET_OBJECTS:python_connection>
        $<TARGET_OBJECTS:python_expression>
        $<TARGET_OBJECTS:python_relation>
        $<TARGET_OBJECTS:python_type>
)
# add _duckdb_dependencies
target_link_libraries(_duckdb PRIVATE _duckdb_dependencies)

# ────────────────────────────────────────────
# Put the object file in the correct place
# ────────────────────────────────────────────

# If we're not building through scikit-build-core then we have to set a different dest dir
include(GNUInstallDirs)
if(DEFINED SKBUILD_PLATLIB_DIR)
  set(_DUCKDB_PY_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
elseif(DEFINED Python_SITEARCH)
  set(_DUCKDB_PY_INSTALL_DIR "${Python_SITEARCH}")
else()
  message(WARNING "Could not determine Python install dir. Falling back to CMAKE_INSTALL_LIBDIR.")
  set(_DUCKDB_PY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
endif()

install(TARGETS _duckdb LIBRARY DESTINATION "${_DUCKDB_PY_INSTALL_DIR}")

cmake_minimum_required(VERSION 3.29)

project(duckdb_python LANGUAGES CXX)

# Always use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(MSVC)
  # MSVC caps object files at 65,279 sections, we instantiate enough templates
  # to exceed it, so need to set /bigobj
  add_compile_options(/bigobj)
endif()

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
# nanobind (requires Python to be located first; pybind11 used to do this
# internally)
find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)
# Nanobind ships its CMake config inside site-packages/nanobind/cmake, so
# find_package() can't discover it unless we set it. (scikit-build-core does
# this as well)
if(NOT nanobind_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
endif()
find_package(nanobind CONFIG REQUIRED)
# Build nanobind's core support library up front so the object libraries below
# (which include nanobind headers via the umbrella) compile against its include
# dirs + Python headers + flags.
nanobind_build_library(nanobind-static)

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
target_link_libraries(_duckdb_dependencies INTERFACE nanobind-static
                                                     duckdb_target)
# Also add include directory
target_include_directories(
  _duckdb_dependencies
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>)

# We link duckdb_static. Without this define, duckdb.h marks C API symbols
# __declspec(dllimport) on Windows, producing unresolvable __imp_* references at
# link time. No-op on non-Windows.
target_compile_definitions(_duckdb_dependencies INTERFACE DUCKDB_STATIC_BUILD)

# Optional AddressSanitizer instrumentation of the Python binding objects ONLY.
# Every binding object library consumes _duckdb_dependencies (for headers) and
# _duckdb links it, so adding the flag here instruments the bindings and links
# the ASAN runtime, while the engine target (duckdb_target, which does NOT
# consume this) stays uninstrumented and keeps hitting the sccache cache. ASAN's
# allocator is process-global, so heap errors involving the instrumented binding
# code are still caught. OFF by default; enable with -DDUCKDB_PY_ASAN=ON.
option(DUCKDB_PY_ASAN
       "Instrument the Python binding objects with AddressSanitizer" OFF)
if(DUCKDB_PY_ASAN)
  target_compile_options(
    _duckdb_dependencies INTERFACE -fsanitize=address -fno-omit-frame-pointer
                                   -g)
  target_link_options(_duckdb_dependencies INTERFACE -fsanitize=address)
endif()

# ────────────────────────────────────────────
# Descend into the real DuckDB‑Python sources
# ────────────────────────────────────────────
add_subdirectory(src)

nanobind_add_module(
  _duckdb
  NB_STATIC
  $<TARGET_OBJECTS:python_src>
  $<TARGET_OBJECTS:python_arrow>
  $<TARGET_OBJECTS:python_common>
  $<TARGET_OBJECTS:python_functional>
  $<TARGET_OBJECTS:python_jupyter>
  $<TARGET_OBJECTS:python_native>
  $<TARGET_OBJECTS:python_numpy>
  $<TARGET_OBJECTS:python_pandas>
  $<TARGET_OBJECTS:python_connection>
  $<TARGET_OBJECTS:python_expression>
  $<TARGET_OBJECTS:python_relation>
  $<TARGET_OBJECTS:python_type>)
# add _duckdb_dependencies
target_link_libraries(_duckdb PRIVATE _duckdb_dependencies)
duckdb_link_extensions(_duckdb)

# ────────────────────────────────────────────
# Put the object file in the correct place
# ────────────────────────────────────────────

# If we're not building through scikit-build-core then we have to set a
# different dest dir
include(GNUInstallDirs)
if(DEFINED SKBUILD_PLATLIB_DIR)
  set(_DUCKDB_PY_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
elseif(DEFINED Python_SITEARCH)
  set(_DUCKDB_PY_INSTALL_DIR "${Python_SITEARCH}")
else()
  message(
    WARNING
      "Could not determine Python install dir. Falling back to CMAKE_INSTALL_LIBDIR."
  )
  set(_DUCKDB_PY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
endif()

install(TARGETS _duckdb LIBRARY DESTINATION "${_DUCKDB_PY_INSTALL_DIR}")

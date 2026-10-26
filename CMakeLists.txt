cmake_minimum_required(VERSION 3.29)

# C is enabled here, not only by the engine, so the C compiler is known when the
# link time optimization switch checks it.
project(duckdb_python LANGUAGES C CXX)

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
#
# The stable ABI component rides along in this call because FindPython creates
# Python::SABIModule only when asked for it here, and nanobind's abi3 library
# links that target.
find_package(
  Python
  COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} NumPy
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
# nanobind puts Py_LIMITED_API on its abi3 library as a public define. The
# object libraries below compile against that library too, so they see the same
# define as the module; mixing the two libraries would compile the two sides
# against different Python headers. The condition is the one nanobind applies in
# linked mode: CPython 3.12 or later, not free-threaded.
if(TARGET Python::SABIModule
   AND Python_INTERPRETER_ID STREQUAL "Python"
   AND NOT Python_VERSION VERSION_LESS 3.12
   AND NOT NB_ABI MATCHES "[0-9]t")
  set(DUCKDB_PY_STABLE_ABI STABLE_ABI)
  set(DUCKDB_PY_NANOBIND nanobind-static-abi3)
else()
  set(DUCKDB_PY_STABLE_ABI)
  set(DUCKDB_PY_NANOBIND nanobind-static)
endif()
# Build nanobind's core support library up front so the object libraries below
# (which include nanobind headers via the umbrella) compile against its include
# dirs + Python headers + flags.
nanobind_build_library(${DUCKDB_PY_NANOBIND})

# ────────────────────────────────────────────
# Link time optimization
# ────────────────────────────────────────────
# Empty by default. "thin" or "full" compiles the engine, the bindings and the
# nanobind support library to bitcode and resolves the whole module at link
# time. DuckDB reads CMAKE_LTO for its own targets, so it has to be set before
# the engine is configured. DuckDB applies the flag to clang only, outside the
# MSVC branch; asking for it elsewhere is an error rather than a silent native
# build.
set(DUCKDB_PY_LTO
    ""
    CACHE STRING "Link time optimization mode (thin, full) or empty")
if(DUCKDB_PY_LTO)
  if(NOT DUCKDB_PY_LTO MATCHES "^(thin|full)$")
    message(
      FATAL_ERROR
        "DUCKDB_PY_LTO must be thin, full or empty, not '${DUCKDB_PY_LTO}'")
  endif()
  # The engine copies its release C++ flags to C, so both compilers must be
  # clang.
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang$"
     OR NOT CMAKE_C_COMPILER_ID MATCHES "Clang$"
     OR MSVC)
    message(
      FATAL_ERROR
        "DUCKDB_PY_LTO=${DUCKDB_PY_LTO} needs clang with the GNU driver for C and C++, the compilers are ${CMAKE_C_COMPILER_ID} and ${CMAKE_CXX_COMPILER_ID}"
    )
  endif()
  # ld.bfd only reads bitcode through the LLVM gold plugin, which manylinux does
  # not ship.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(DUCKDB_PY_LLD NAMES ld.lld)
    if(NOT DUCKDB_PY_LLD)
      message(FATAL_ERROR "DUCKDB_PY_LTO on Linux needs ld.lld on the PATH")
    endif()
  endif()
endif()
# Forwarded unconditionally so that a build directory configured with LTO once
# returns to a native engine when the switch is dropped; DUCKDB_PY_LTO is the
# only knob.
set(CMAKE_LTO
    "${DUCKDB_PY_LTO}"
    CACHE STRING "Forwarded from DUCKDB_PY_LTO" FORCE)
# The engine adds -flto to its Release flags and derives RelWithDebInfo from
# them, so the module and the bindings follow the same two configurations.
set(DUCKDB_PY_LTO_CONFIGS "$<CONFIG:Release,RelWithDebInfo>")
set(DUCKDB_PY_LTO_FLAG
    "$<$<AND:$<BOOL:${DUCKDB_PY_LTO}>,${DUCKDB_PY_LTO_CONFIGS}>:-flto=${DUCKDB_PY_LTO}>"
)

# DuckDB
include(cmake/duckdb_loader.cmake)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  duckdb_configure_for_debug()
else()
  duckdb_configure_for_release()
endif()
duckdb_add_library(duckdb_target)

# The engine applies its architecture profile (x86-64-v3, armv8-a, apple-m1) to
# its own targets only. LLVM does not inline between functions whose target
# features differ, so the bindings compile with the same flags or the link time
# optimization stops at the engine boundary.
duckdb_resolve_architecture_flags(DUCKDB_PY_ARCHITECTURE_FLAGS)
foreach(flag IN LISTS DUCKDB_PY_ARCHITECTURE_FLAGS)
  target_compile_options(${DUCKDB_PY_NANOBIND}
                         PRIVATE "$<$<CONFIG:Release>:${flag}>")
endforeach()
target_compile_options(${DUCKDB_PY_NANOBIND} PRIVATE "${DUCKDB_PY_LTO_FLAG}")

# Bundle in INTERFACE library
add_library(_duckdb_dependencies INTERFACE)
target_link_libraries(_duckdb_dependencies INTERFACE ${DUCKDB_PY_NANOBIND}
                                                     duckdb_target)
# Also add include directory
target_include_directories(
  _duckdb_dependencies
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>)

# We link duckdb_static. Without this define, duckdb.h marks C API symbols
# __declspec(dllimport) on Windows, producing unresolvable __imp_* references at
# link time. No-op on non-Windows.
target_compile_definitions(_duckdb_dependencies INTERFACE DUCKDB_STATIC_BUILD)

foreach(flag IN LISTS DUCKDB_PY_ARCHITECTURE_FLAGS)
  target_compile_options(_duckdb_dependencies
                         INTERFACE "$<$<CONFIG:Release>:${flag}>")
endforeach()
target_compile_options(_duckdb_dependencies INTERFACE "${DUCKDB_PY_LTO_FLAG}")

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
  ${DUCKDB_PY_STABLE_ABI}
  $<TARGET_OBJECTS:python_src>
  $<TARGET_OBJECTS:python_arrow>
  $<TARGET_OBJECTS:python_engine>
  $<TARGET_OBJECTS:python_common>
  $<TARGET_OBJECTS:python_functional>
  $<TARGET_OBJECTS:python_jupyter>
  $<TARGET_OBJECTS:python_native>
  $<TARGET_OBJECTS:python_numpy>
  $<TARGET_OBJECTS:python_pandas>
  $<TARGET_OBJECTS:python_connection>
  $<TARGET_OBJECTS:python_type>)
# add _duckdb_dependencies
target_link_libraries(_duckdb PRIVATE _duckdb_dependencies)
duckdb_python_link_extensions(_duckdb)
if(DUCKDB_PY_LTO)
  target_link_options(_duckdb PRIVATE "${DUCKDB_PY_LTO_FLAG}")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # lld's default ThinLTO job count ran the whole backend on one thread on
    # aarch64 (a 20 minute link that takes about a minute with every core).
    target_link_options(
      _duckdb PRIVATE "$<${DUCKDB_PY_LTO_CONFIGS}:-fuse-ld=lld;-flto-jobs=all>")
  endif()
endif()

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

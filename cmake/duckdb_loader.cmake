# cmake/duckdb_loader.cmake
#
# Simple DuckDB Build Configuration Module
#
# Sets sensible defaults for DuckDB Python extension builds and provides a clean
# interface for adding DuckDB as a library target.
#
# Usage: include(cmake/duckdb_loader.cmake) # Optionally load extensions
# set(BUILD_EXTENSIONS "json;parquet;icu")
#
# # set sensible defaults for a debug build: duckdb_configure_for_debug()
#
# # ...or, set sensible defaults for a release build:
# duckdb_configure_for_release()
#
# # Link to your target duckdb_add_library(duckdb_target)
# target_link_libraries(my_lib PRIVATE ${duckdb_target})

include_guard(GLOBAL)

# ════════════════════════════════════════════════════════════════════════════════
# Configuration Defaults - Optimized for Python Extension Builds
# ════════════════════════════════════════════════════════════════════════════════

# Helper macro to set default values that can be overridden from command line
macro(_duckdb_set_default var_name default_value)
  if(NOT DEFINED ${var_name})
    set(${var_name} ${default_value})
  endif()
endmacro()

# Source configuration
_duckdb_set_default(DUCKDB_SOURCE_PATH
                    "${CMAKE_CURRENT_SOURCE_DIR}/external/duckdb")

# Extension list - commonly used extensions for Python
_duckdb_set_default(BUILD_EXTENSIONS "core_functions;parquet;icu;json")

# Core build options - disable unnecessary components for Python builds
_duckdb_set_default(BUILD_SHELL OFF)
_duckdb_set_default(BUILD_UNITTESTS OFF)
_duckdb_set_default(BUILD_BENCHMARKS OFF)
_duckdb_set_default(DISABLE_UNITY OFF)

# Extension configuration
_duckdb_set_default(DISABLE_BUILTIN_EXTENSIONS OFF)
_duckdb_set_default(ENABLE_EXTENSION_AUTOINSTALL ON)
_duckdb_set_default(ENABLE_EXTENSION_AUTOLOADING ON)

# Performance options - enable optimizations by default
_duckdb_set_default(NATIVE_ARCH OFF)

# Sanitizers are off for Python by default. Enabling might result in "symbol not
# found" for  '___ubsan_vptr_type_cache'
_duckdb_set_default(ENABLE_SANITIZER OFF)
_duckdb_set_default(ENABLE_UBSAN OFF)

# Debug options - off by default for release builds
_duckdb_set_default(FORCE_ASSERT OFF)
_duckdb_set_default(DEBUG_STACKTRACE OFF)

# Convert to cache variables for CMake GUI/ccmake compatibility
set(DUCKDB_SOURCE_PATH
    "${DUCKDB_SOURCE_PATH}"
    CACHE PATH "Path to DuckDB source directory")
set(BUILD_EXTENSIONS
    "${BUILD_EXTENSIONS}"
    CACHE STRING "Semicolon-separated list of extensions to enable")
set(BUILD_SHELL
    "${BUILD_SHELL}"
    CACHE BOOL "Build the DuckDB shell executable")
set(BUILD_UNITTESTS
    "${BUILD_UNITTESTS}"
    CACHE BOOL "Build DuckDB unit tests")
set(BUILD_BENCHMARKS
    "${BUILD_BENCHMARKS}"
    CACHE BOOL "Build DuckDB benchmarks")
set(DISABLE_UNITY
    "${DISABLE_UNITY}"
    CACHE BOOL "Disable unity builds (slower compilation)")
set(DISABLE_BUILTIN_EXTENSIONS
    "${DISABLE_BUILTIN_EXTENSIONS}"
    CACHE BOOL "Disable all built-in extensions")
set(ENABLE_EXTENSION_AUTOINSTALL
    "${ENABLE_EXTENSION_AUTOINSTALL}"
    CACHE BOOL "Enable extension auto-installing by default.")
set(ENABLE_EXTENSION_AUTOLOADING
    "${ENABLE_EXTENSION_AUTOLOADING}"
    CACHE BOOL "Enable extension auto-loading by default.")
set(NATIVE_ARCH
    "${NATIVE_ARCH}"
    CACHE BOOL "Optimize for native architecture")
set(ENABLE_SANITIZER
    "${ENABLE_SANITIZER}"
    CACHE BOOL "Enable address sanitizer")
set(ENABLE_UBSAN
    "${ENABLE_UBSAN}"
    CACHE BOOL "Enable undefined behavior sanitizer")
set(FORCE_ASSERT
    "${FORCE_ASSERT}"
    CACHE BOOL "Enable assertions in release builds")
set(DEBUG_STACKTRACE
    "${DEBUG_STACKTRACE}"
    CACHE BOOL "Print a stracktrace on asserts and when testing crashes")

# ════════════════════════════════════════════════════════════════════════════════
# Internal Functions
# ════════════════════════════════════════════════════════════════════════════════

function(_duckdb_validate_source_path)
  if(NOT EXISTS "${DUCKDB_SOURCE_PATH}")
    message(
      FATAL_ERROR "DuckDB source path does not exist: ${DUCKDB_SOURCE_PATH}\n"
                  "Please set DUCKDB_SOURCE_PATH to the correct location.")
  endif()

  if(NOT EXISTS "${DUCKDB_SOURCE_PATH}/CMakeLists.txt")
    message(
      FATAL_ERROR
        "DuckDB source path does not contain CMakeLists.txt: ${DUCKDB_SOURCE_PATH}\n"
        "Please ensure this points to the root of DuckDB source tree.")
  endif()
endfunction()

function(_duckdb_create_interface_target target_name)
  add_library(${target_name} INTERFACE)

  # Include directories to deal with leaking 3rd party headers in duckdb headers
  # See https://github.com/duckdblabs/duckdb-internal/issues/5084
  target_include_directories(
    ${target_name}
    INTERFACE
      # Main DuckDB headers
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/src/include>
      # Third-party headers that leak through DuckDB's API
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party>
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party/re2>
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party/fast_float>
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party/utf8proc/include>
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party/libpg_query/include>
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party/fmt/include>)

  # Compile definitions based on configuration
  target_compile_definitions(
    ${target_name} INTERFACE $<$<BOOL:${FORCE_ASSERT}>:DUCKDB_FORCE_ASSERT>
                             $<$<CONFIG:Debug>:DUCKDB_DEBUG_MODE>)

  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(
      ${target_name}
      INTERFACE /wd4244 # suppress Conversion from 'type1' to 'type2', possible
                        # loss of data
                /wd4267 # suppress Conversion from ‘size_t’ to ‘type’, possible
                        # loss of data
                /wd4200 # suppress Nonstandard extension used: zero-sized array
                        # in struct/union
                /wd26451
                /wd26495 # suppress Code Analysis
                /D_CRT_SECURE_NO_WARNINGS # suppress warnings about unsafe
                                          # functions
                /utf-8 # treat source files as UTF-8 encoded
    )
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_compile_options(
      ${target_name}
      INTERFACE -stdlib=libc++ # for libc++ in favor of older libstdc++
                -mmacosx-version-min=10.7 # minimum osx version compatibility
    )
  endif()

  # Link to the DuckDB static library
  target_link_libraries(${target_name} INTERFACE duckdb_static)

  # Enable position independent code for shared library builds
  set_target_properties(${target_name}
                        PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)
endfunction()

function(_duckdb_print_summary)
  message(STATUS "DuckDB Configuration:")
  message(STATUS "  Source: ${DUCKDB_SOURCE_PATH}")
  message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
  message(STATUS "  Native Arch: ${NATIVE_ARCH}")
  message(STATUS "  Unity Build Disabled: ${DISABLE_UNITY}")

  set(debug_opts)
  if(FORCE_ASSERT)
    list(APPEND debug_opts "FORCE_ASSERT")
  endif()
  if(DEBUG_STACKTRACE)
    list(APPEND debug_opts "DEBUG_STACKTRACE")
  endif()

  if(debug_opts)
    message(STATUS "  Debug Options: ${debug_opts}")
  endif()
endfunction()

function(_duckdb_python_write_static_extension_loader output)
  # Writes a C++ source defining duckdb_register_static_extensions(), which
  # registers each extension given in ARGN. This only relies on DuckDB's public
  # static extension contract (duckdb_static_extension.h): every extension
  # archive exports duckdb_extension_<name>_describe, and passing it to
  # duckdb_register_static_extension registers the extension and pulls it out of
  # its archive.
  set(extension_list "")
  set(declarations "")
  set(registrations "")
  foreach(ext IN LISTS ARGN)
    if(NOT ext MATCHES "^[a-z0-9_]+$")
      message(FATAL_ERROR "Invalid DuckDB extension name: '${ext}'")
    endif()
    string(APPEND extension_list " ${ext}")
    string(
      APPEND
      declarations
      "int32_t duckdb_extension_${ext}_describe(duckdb_extension_descriptor *descriptor);\n"
    )
    string(
      APPEND
      registrations
      "\tif (duckdb_register_static_extension(duckdb_extension_${ext}_describe) != 0) {\n\t\tresult = 1;\n\t}\n"
    )
  endforeach()
  if(extension_list STREQUAL "")
    set(extension_list " (none)")
  endif()

  # file(CONFIGURE) leaves the file untouched when the content is unchanged, so
  # an unchanged extension list recompiles nothing
  file(
    CONFIGURE
    OUTPUT
    "${output}"
    CONTENT
    "// Generated by duckdb_python_link_extensions in cmake/duckdb_loader.cmake. Do not edit.
// Registers the statically linked extensions:@extension_list@

#include \"duckdb_static_extension.h\"

extern \"C\" {

@declarations@
int32_t duckdb_register_static_extensions(void) {
\tint32_t result = 0;
@registrations@\treturn result;
}

} // extern \"C\"
"
    @ONLY)
endfunction()

# ════════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════════

function(duckdb_add_library target_name)
  _duckdb_validate_source_path()
  _duckdb_print_summary()

  # Add DuckDB subdirectory - it will use our variables
  add_subdirectory("${DUCKDB_SOURCE_PATH}" duckdb EXCLUDE_FROM_ALL)

  # Create clean interface target
  _duckdb_create_interface_target(${target_name})
endfunction()

function(duckdb_python_link_extensions target_name)
  # Statically link the extensions in LINK_EXTENSIONS into target_name. Which
  # extensions are linked is a link-time decision: BUILD_EXTENSIONS configures
  # the extension targets, LINK_EXTENSIONS picks from them, and changing it only
  # regenerates the loader and relinks. DuckDB is added with EXCLUDE_FROM_ALL,
  # so configured extensions that are not linked are never compiled.
  #
  # An extension archive is only pulled in through its describe function, so we
  # also generate a static extension loader listing them, which defines
  # duckdb_register_static_extensions(). The module calls it at import time. The
  # loader is always generated (empty when no extensions are linked) so that
  # call never needs special-casing.
  if(LINK_EXTENSIONS)
    set(link_extensions ${LINK_EXTENSIONS})
  else()
    set(link_extensions ${BUILD_EXTENSIONS})
  endif()

  set(loader_extensions "")
  if(link_extensions)
    message(STATUS "Linking DuckDB extensions:")
    foreach(ext IN LISTS link_extensions)
      if(NOT TARGET ${ext}_extension)
        message(
          FATAL_ERROR
            "Cannot link extension '${ext}': DuckDB did not configure it, add it to BUILD_EXTENSIONS"
        )
      endif()
      message(STATUS "- ${ext}")
      target_link_libraries(${target_name} PRIVATE ${ext}_extension)
      # Interface-only targets (e.g. DuckDB's jemalloc shim) have no describe
      # function to register
      get_target_property(ext_type ${ext}_extension TYPE)
      if(NOT ext_type STREQUAL "INTERFACE_LIBRARY")
        list(APPEND loader_extensions ${ext})
      endif()
    endforeach()
  else()
    message(STATUS "No DuckDB extensions linked in")
  endif()

  set(loader
      "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_static_extension_loader.cpp")
  _duckdb_python_write_static_extension_loader("${loader}" ${loader_extensions})
  target_sources(${target_name} PRIVATE "${loader}")
endfunction()

# ════════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ════════════════════════════════════════════════════════════════════════════════

function(duckdb_configure_for_debug)
  # Only set if not already defined (allows override from command line)
  if(NOT DEFINED FORCE_ASSERT)
    set(FORCE_ASSERT
        ON
        PARENT_SCOPE)
  endif()
  if(NOT DEFINED DEBUG_STACKTRACE)
    set(DEBUG_STACKTRACE
        ON
        PARENT_SCOPE)
  endif()
  message(STATUS "DuckDB: Configured for debug build")
endfunction()

function(duckdb_configure_for_release)
  message(STATUS "DuckDB: Configured for release build")
endfunction()

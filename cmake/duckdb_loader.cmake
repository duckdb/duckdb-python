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

# Prebuilt archives: a folder with libduckdb_static.a and lib<ext>_extension.a.
# When set, DuckDB is not built: headers come from DUCKDB_SOURCE_PATH and the
# archives from this folder.
_duckdb_set_default(DUCKDB_LIBRARY_DIR "")

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
set(DUCKDB_LIBRARY_DIR
    "${DUCKDB_LIBRARY_DIR}"
    CACHE PATH "Folder with prebuilt DuckDB archives (empty: build DuckDB)")
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

function(_duckdb_python_source_commit out_var)
  # The commit of the DuckDB source tree the headers come from. The build
  # backend passes it as GIT_COMMIT_HASH (also for sdists, which have no git
  # history); a plain CMake run asks git.
  if(GIT_COMMIT_HASH)
    set(${out_var} "${GIT_COMMIT_HASH}" PARENT_SCOPE)
    return()
  endif()
  find_package(Git QUIET)
  if(GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${DUCKDB_SOURCE_PATH}" rev-parse HEAD
      OUTPUT_VARIABLE commit
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE result
      ERROR_QUIET)
    if(result EQUAL 0)
      set(${out_var} "${commit}" PARENT_SCOPE)
      return()
    endif()
  endif()
  message(
    FATAL_ERROR
      "Cannot determine the commit of ${DUCKDB_SOURCE_PATH}, pass GIT_COMMIT_HASH"
  )
endfunction()

function(_duckdb_python_architecture out_var)
  # CMAKE_OSX_ARCHITECTURES is what is built when cross-compiling on macOS
  if(CMAKE_OSX_ARCHITECTURES)
    set(${out_var} "${CMAKE_OSX_ARCHITECTURES}" PARENT_SCOPE)
  else()
    set(${out_var} "${CMAKE_SYSTEM_PROCESSOR}" PARENT_SCOPE)
  endif()
endfunction()

function(_duckdb_python_read_archive_manifest library_dir out_extensions)
  # Reads library_dir/duckdb_archives.json and checks that the archives are
  # usable with this build. DuckDB's C++ API has no ABI guarantee, so the
  # archives must be built from the same commit as the headers, and with the
  # same configuration where it changes how DuckDB's headers compile.
  set(manifest "${library_dir}/duckdb_archives.json")
  if(NOT EXISTS "${manifest}")
    message(FATAL_ERROR "DUCKDB_LIBRARY_DIR has no duckdb_archives.json: ${library_dir}")
  endif()
  file(READ "${manifest}" json)
  foreach(field duckdb_commit build_type force_assert system architecture)
    string(JSON ${field} ERROR_VARIABLE error GET "${json}" ${field})
    if(error)
      message(FATAL_ERROR "Invalid ${manifest}: ${error}")
    endif()
  endforeach()

  set(mismatches "")
  _duckdb_python_source_commit(source_commit)
  if(NOT duckdb_commit STREQUAL source_commit)
    string(APPEND mismatches
           "\n  commit: archives ${duckdb_commit}, headers ${source_commit}")
  endif()
  # Our code is compiled with DUCKDB_DEBUG_MODE in Debug builds only
  set(archives_debug OFF)
  if(build_type STREQUAL "Debug")
    set(archives_debug ON)
  endif()
  set(this_debug OFF)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(this_debug ON)
  endif()
  if(NOT archives_debug STREQUAL this_debug)
    string(APPEND mismatches
           "\n  build type: archives ${build_type}, this build ${CMAKE_BUILD_TYPE}")
  endif()
  set(archives_force_assert OFF)
  if(force_assert)
    set(archives_force_assert ON)
  endif()
  set(this_force_assert OFF)
  if(FORCE_ASSERT)
    set(this_force_assert ON)
  endif()
  if(NOT archives_force_assert STREQUAL this_force_assert)
    string(APPEND mismatches
           "\n  force_assert: archives ${force_assert}, this build ${FORCE_ASSERT}")
  endif()
  if(NOT system STREQUAL CMAKE_SYSTEM_NAME)
    string(APPEND mismatches
           "\n  system: archives ${system}, this build ${CMAKE_SYSTEM_NAME}")
  endif()
  _duckdb_python_architecture(this_architecture)
  if(NOT architecture STREQUAL this_architecture)
    string(APPEND mismatches
           "\n  architecture: archives ${architecture}, this build ${this_architecture}")
  endif()
  if(mismatches)
    message(FATAL_ERROR "The archives in ${library_dir} do not match this build:${mismatches}")
  endif()

  set(extensions "")
  string(JSON count ERROR_VARIABLE error LENGTH "${json}" extensions)
  if(error)
    message(FATAL_ERROR "Invalid ${manifest}: ${error}")
  endif()
  if(count GREATER 0)
    math(EXPR last "${count} - 1")
    foreach(index RANGE ${last})
      string(JSON ext GET "${json}" extensions ${index})
      list(APPEND extensions ${ext})
    endforeach()
  endif()
  set(${out_extensions} "${extensions}" PARENT_SCOPE)
endfunction()

function(_duckdb_python_add_archive_targets library_dir)
  # Creates the targets that DuckDB's own build would provide, as imported
  # static libraries from library_dir: duckdb_static, and an <ext>_extension for
  # each extension in ARGN. libduckdb_static already bundles DuckDB's
  # third-party libraries, so it only needs the system libraries DuckDB links
  # (see DUCKDB_SYSTEM_LIBS in DuckDB's src/CMakeLists.txt)
  set(prefix "${CMAKE_STATIC_LIBRARY_PREFIX}")
  set(suffix "${CMAKE_STATIC_LIBRARY_SUFFIX}")

  find_package(Threads REQUIRED)
  set(system_libs Threads::Threads ${CMAKE_DL_LIBS})
  if(MSVC OR MINGW)
    list(APPEND system_libs ws2_32 rstrtmgr)
  endif()
  if(MSVC)
    list(APPEND system_libs bcrypt)
  endif()

  add_library(duckdb_static STATIC IMPORTED)
  set_target_properties(
    duckdb_static
    PROPERTIES IMPORTED_LOCATION
               "${library_dir}/${prefix}duckdb_static${suffix}"
               INTERFACE_LINK_LIBRARIES "${system_libs}")

  foreach(ext IN LISTS ARGN)
    message(STATUS "- ${ext}_extension")
    add_library(${ext}_extension STATIC IMPORTED)
    # The engine archive must come after the extension archives on the link line
    set_target_properties(
      ${ext}_extension
      PROPERTIES IMPORTED_LOCATION
                 "${library_dir}/${prefix}${ext}_extension${suffix}"
                 INTERFACE_LINK_LIBRARIES duckdb_static)
  endforeach()
endfunction()

function(_duckdb_python_import_archives library_dir)
  # Uses the prebuilt archives in library_dir, after checking its manifest
  _duckdb_python_read_archive_manifest("${library_dir}" extensions)
  set(prefix "${CMAKE_STATIC_LIBRARY_PREFIX}")
  set(suffix "${CMAKE_STATIC_LIBRARY_SUFFIX}")
  if(NOT EXISTS "${library_dir}/${prefix}duckdb_static${suffix}")
    message(
      FATAL_ERROR
        "DUCKDB_LIBRARY_DIR does not contain ${prefix}duckdb_static${suffix}: ${library_dir}"
    )
  endif()
  foreach(ext IN LISTS extensions)
    if(NOT EXISTS "${library_dir}/${prefix}${ext}_extension${suffix}")
      message(
        FATAL_ERROR
          "Extension '${ext}' is in the manifest of ${library_dir} but its archive is missing"
      )
    endif()
  endforeach()
  message(STATUS "Using prebuilt DuckDB archives from ${library_dir}")
  _duckdb_python_add_archive_targets("${library_dir}" ${extensions})
endfunction()

function(_duckdb_python_build_archives library_dir)
  # Builds DuckDB from DUCKDB_SOURCE_PATH as a separate CMake project. It writes
  # its archives straight into library_dir, and a manifest once the build
  # succeeded, so library_dir has the same layout as a prebuilt folder and can
  # be reused as DUCKDB_LIBRARY_DIR by other builds. Only duckdb_static and the
  # extensions to link are built.
  include(ExternalProject)
  if(LINK_EXTENSIONS)
    set(extensions ${LINK_EXTENSIONS})
  else()
    set(extensions ${BUILD_EXTENSIONS})
  endif()
  set(targets duckdb_static)
  foreach(ext IN LISTS extensions)
    if(NOT ext IN_LIST BUILD_EXTENSIONS)
      message(
        FATAL_ERROR
          "Cannot link extension '${ext}': DuckDB did not configure it, add it to BUILD_EXTENSIONS"
      )
    endif()
    list(APPEND targets ${ext}_extension)
  endforeach()

  # Everything DuckDB's build needs from this one. Lists are passed with | as
  # separator, see LIST_SEPARATOR below.
  set(cmake_args "")
  foreach(
    var IN
    ITEMS CMAKE_BUILD_TYPE
          CMAKE_CXX_STANDARD
          CMAKE_MSVC_RUNTIME_LIBRARY
          CMAKE_MAKE_PROGRAM
          CMAKE_TOOLCHAIN_FILE
          CMAKE_C_COMPILER
          CMAKE_CXX_COMPILER
          CMAKE_C_COMPILER_LAUNCHER
          CMAKE_CXX_COMPILER_LAUNCHER
          CMAKE_C_FLAGS
          CMAKE_CXX_FLAGS
          CMAKE_OSX_ARCHITECTURES
          CMAKE_OSX_DEPLOYMENT_TARGET
          CMAKE_OSX_SYSROOT
          BUILD_EXTENSIONS
          BUILD_SHELL
          BUILD_UNITTESTS
          BUILD_BENCHMARKS
          DISABLE_UNITY
          DISABLE_BUILTIN_EXTENSIONS
          ENABLE_EXTENSION_AUTOINSTALL
          ENABLE_EXTENSION_AUTOLOADING
          NATIVE_ARCH
          ENABLE_SANITIZER
          ENABLE_UBSAN
          FORCE_ASSERT
          DEBUG_STACKTRACE
          OVERRIDE_GIT_DESCRIBE
          GIT_COMMIT_HASH)
    if(DEFINED ${var} AND NOT "${${var}}" STREQUAL "")
      string(REPLACE ";" "|" value "${${var}}")
      list(APPEND cmake_args "-D${var}=${value}")
    endif()
  endforeach()

  # Only the archives we link go to library_dir, DuckDB's third-party archives
  # (already bundled in libduckdb_static) stay in its build tree. The
  # properties are set from a file DuckDB includes in its project() call,
  # deferred until DuckDB defined its targets. $<1:...> keeps multi-config
  # generators from appending a per-config directory.
  list(JOIN targets " " target_list)
  set(archive_output "${CMAKE_CURRENT_BINARY_DIR}/duckdb_archive_output.cmake")
  file(
    CONFIGURE
    OUTPUT "${archive_output}"
    CONTENT
      "# Generated by _duckdb_python_build_archives in cmake/duckdb_loader.cmake. Do not edit.
cmake_language(DEFER CALL set_target_properties @target_list@
               PROPERTIES ARCHIVE_OUTPUT_DIRECTORY \"$<1:@library_dir@>\")
"
    @ONLY)
  list(APPEND cmake_args "-DCMAKE_PROJECT_DuckDB_INCLUDE=${archive_output}")

  # The manifest describing the archives, installed next to them after a
  # successful build
  _duckdb_python_source_commit(commit)
  _duckdb_python_architecture(architecture)
  set(force_assert false)
  if(FORCE_ASSERT)
    set(force_assert true)
  endif()
  list(TRANSFORM extensions PREPEND "\"" OUTPUT_VARIABLE quoted)
  list(TRANSFORM quoted APPEND "\"")
  list(JOIN quoted ", " extension_array)
  set(manifest "${CMAKE_CURRENT_BINARY_DIR}/duckdb_archives.json")
  file(
    CONFIGURE
    OUTPUT "${manifest}"
    CONTENT
      "{
  \"duckdb_commit\": \"${commit}\",
  \"build_type\": \"${CMAKE_BUILD_TYPE}\",
  \"force_assert\": ${force_assert},
  \"system\": \"${CMAKE_SYSTEM_NAME}\",
  \"architecture\": \"${architecture}\",
  \"extensions\": [${extension_array}]
}
"
    @ONLY)

  set(prefix "${CMAKE_STATIC_LIBRARY_PREFIX}")
  set(suffix "${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(byproducts "${library_dir}/${prefix}duckdb_static${suffix}")
  foreach(ext IN LISTS extensions)
    list(APPEND byproducts "${library_dir}/${prefix}${ext}_extension${suffix}")
  endforeach()

  message(STATUS "Building DuckDB archives into ${library_dir}")
  ExternalProject_Add(
    duckdb_build
    SOURCE_DIR "${DUCKDB_SOURCE_PATH}"
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/duckdb-prefix"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/duckdb-build"
    LIST_SEPARATOR "|"
    CMAKE_ARGS ${cmake_args}
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG>
                  --target ${targets}
    # Let DuckDB's own build decide what is out of date
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS ${byproducts}
    INSTALL_COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${manifest}"
                    "${library_dir}/duckdb_archives.json"
    USES_TERMINAL_BUILD ON)

  _duckdb_python_add_archive_targets("${library_dir}" ${extensions})
  # Dependencies of imported targets are followed by the targets linking them
  add_dependencies(duckdb_static duckdb_build)
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
  # The headers always come from the source tree
  _duckdb_validate_source_path()

  if(DUCKDB_LIBRARY_DIR)
    _duckdb_python_import_archives("${DUCKDB_LIBRARY_DIR}")
  else()
    _duckdb_print_summary()
    _duckdb_python_build_archives("${CMAKE_CURRENT_BINARY_DIR}/duckdb_archives")
  endif()

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
      if(NOT TARGET ${ext}_extension AND DUCKDB_LIBRARY_DIR)
        message(
          FATAL_ERROR
            "Cannot link extension '${ext}': it is not in the manifest of ${DUCKDB_LIBRARY_DIR}"
        )
      elseif(NOT TARGET ${ext}_extension)
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

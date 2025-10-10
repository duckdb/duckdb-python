# cmake/compiler_launcher.cmake
include_guard(GLOBAL) # only include once
include(CMakeParseArguments)

# ────────────────────────────────────────────
# setup_compiler_launcher_if_available()
#
# Function to look for ccache and sccache to speed up builds, if available
# ────────────────────────────────────────────
function(setup_compiler_launcher_if_available)
  if(NOT DEFINED CMAKE_C_COMPILER_LAUNCHER)
    find_program(COMPILER_LAUNCHER NAMES ccache sccache)
    if(COMPILER_LAUNCHER)
      message(STATUS "Using ${COMPILER_LAUNCHER} as C compiler launcher")
      set(CMAKE_C_COMPILER_LAUNCHER
          "${COMPILER_LAUNCHER}"
          CACHE STRING "" FORCE)
    endif()
  endif()

  if(NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(COMPILER_LAUNCHER NAMES ccache sccache)
    if(COMPILER_LAUNCHER)
      message(STATUS "Using ${COMPILER_LAUNCHER} as C++ compiler launcher")
      set(CMAKE_CXX_COMPILER_LAUNCHER
          "${COMPILER_LAUNCHER}"
          CACHE STRING "" FORCE)
    endif()
  endif()
endfunction()

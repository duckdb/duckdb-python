# this is used for clang-tidy checks
add_subdirectory(pyrelation)
add_subdirectory(pyexpression)
add_subdirectory(pybind11)
add_subdirectory(numpy)
add_subdirectory(native)
add_subdirectory(jupyter)
add_subdirectory(typing)
add_subdirectory(functional)
add_subdirectory(pyconnection)
add_subdirectory(common)
add_subdirectory(pandas)
add_subdirectory(arrow)

add_library(
  python_src OBJECT
  dataframe.cpp
  duckdb_python.cpp
  importer.cpp
  map.cpp
  path_like.cpp
  pyconnection.cpp
  pyexpression.cpp
  pyfilesystem.cpp
  pyrelation.cpp
  pyresult.cpp
  pystatement.cpp
  python_dependency.cpp
  python_import_cache.cpp
  python_replacement_scan.cpp
  python_udf.cpp)

target_link_libraries(python_src PRIVATE _duckdb_dependencies)

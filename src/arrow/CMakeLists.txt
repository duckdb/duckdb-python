# this is used for clang-tidy checks
add_library(
  python_arrow OBJECT
  arrow_export_utils.cpp arrow_importer.cpp predicate_translation.cpp
  python_arrow_scan.cpp single_use.cpp)

target_link_libraries(python_arrow PRIVATE _duckdb_dependencies)

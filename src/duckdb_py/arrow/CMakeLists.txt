# this is used for clang-tidy checks
add_library(
  python_arrow OBJECT arrow_array_stream.cpp arrow_export_utils.cpp
                      polars_filter_pushdown.cpp pyarrow_filter_pushdown.cpp)

target_link_libraries(python_arrow PRIVATE _duckdb_dependencies)

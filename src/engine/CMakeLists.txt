# this is used for clang-tidy checks
add_library(python_engine OBJECT connection.cpp result.cpp)

target_link_libraries(python_engine PRIVATE _duckdb_dependencies)

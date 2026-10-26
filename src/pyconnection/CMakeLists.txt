# this is used for clang-tidy checks
add_library(python_connection OBJECT seam.cpp type_creation.cpp)

target_link_libraries(python_connection PRIVATE _duckdb_dependencies)

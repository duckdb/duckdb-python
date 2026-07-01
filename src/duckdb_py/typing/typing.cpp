#include "duckdb_python/typing.hpp"
#include "duckdb_python/pytype.hpp"

namespace duckdb {

static void DefineBaseTypes(py::handle &m) {
	m.attr("SQLNULL") = std::make_shared<DuckDBPyType>(LogicalType::SQLNULL);
	m.attr("BOOLEAN") = std::make_shared<DuckDBPyType>(LogicalType::BOOLEAN);
	m.attr("TINYINT") = std::make_shared<DuckDBPyType>(LogicalType::TINYINT);
	m.attr("UTINYINT") = std::make_shared<DuckDBPyType>(LogicalType::UTINYINT);
	m.attr("SMALLINT") = std::make_shared<DuckDBPyType>(LogicalType::SMALLINT);
	m.attr("USMALLINT") = std::make_shared<DuckDBPyType>(LogicalType::USMALLINT);
	m.attr("INTEGER") = std::make_shared<DuckDBPyType>(LogicalType::INTEGER);
	m.attr("UINTEGER") = std::make_shared<DuckDBPyType>(LogicalType::UINTEGER);
	m.attr("BIGINT") = std::make_shared<DuckDBPyType>(LogicalType::BIGINT);
	m.attr("UBIGINT") = std::make_shared<DuckDBPyType>(LogicalType::UBIGINT);
	m.attr("HUGEINT") = std::make_shared<DuckDBPyType>(LogicalType::HUGEINT);
	m.attr("UHUGEINT") = std::make_shared<DuckDBPyType>(LogicalType::UHUGEINT);
	m.attr("UUID") = std::make_shared<DuckDBPyType>(LogicalType::UUID);
	m.attr("FLOAT") = std::make_shared<DuckDBPyType>(LogicalType::FLOAT);
	m.attr("DOUBLE") = std::make_shared<DuckDBPyType>(LogicalType::DOUBLE);
	m.attr("DATE") = std::make_shared<DuckDBPyType>(LogicalType::DATE);

	m.attr("TIMESTAMP") = std::make_shared<DuckDBPyType>(LogicalType::TIMESTAMP);
	m.attr("TIMESTAMP_MS") = std::make_shared<DuckDBPyType>(LogicalType::TIMESTAMP_MS);
	m.attr("TIMESTAMP_NS") = std::make_shared<DuckDBPyType>(LogicalType::TIMESTAMP_NS);
	m.attr("TIMESTAMP_S") = std::make_shared<DuckDBPyType>(LogicalType::TIMESTAMP_S);

	m.attr("TIME") = std::make_shared<DuckDBPyType>(LogicalType::TIME);
	m.attr("TIME_NS") = std::make_shared<DuckDBPyType>(LogicalType::TIME_NS);

	m.attr("TIME_TZ") = std::make_shared<DuckDBPyType>(LogicalType::TIME_TZ);
	m.attr("TIMESTAMP_TZ") = std::make_shared<DuckDBPyType>(LogicalType::TIMESTAMP_TZ);

	m.attr("VARCHAR") = std::make_shared<DuckDBPyType>(LogicalType::VARCHAR);

	m.attr("BLOB") = std::make_shared<DuckDBPyType>(LogicalType::BLOB);
	m.attr("BIT") = std::make_shared<DuckDBPyType>(LogicalType::BIT);
	m.attr("INTERVAL") = std::make_shared<DuckDBPyType>(LogicalType::INTERVAL);
	m.attr("VARIANT") = std::make_shared<DuckDBPyType>(LogicalType::VARIANT());
}

void DuckDBPyTyping::Initialize(py::module_ &parent) {
	auto m = parent.def_submodule("_sqltypes", "This module contains classes and methods related to typing");
	DuckDBPyType::Initialize(m);

	DefineBaseTypes(m);
}

} // namespace duckdb

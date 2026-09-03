//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pyappender.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"

#include <optional>

namespace duckdb {

struct DuckDBPyAppender : std::enable_shared_from_this<DuckDBPyAppender> {
public:
	DuckDBPyAppender(std::shared_ptr<DuckDBPyConnection> connection, unique_ptr<BaseAppender> appender);
	~DuckDBPyAppender();

	static void Initialize(nb::handle &m);

	void AppendRow(const nb::args &args);
	void Flush();
	void Close();
	std::shared_ptr<DuckDBPyAppender> Enter();
	void Exit(const nb::object &exc_type, const nb::object &exc, const nb::object &traceback);
	idx_t ColumnCount();

private:
	void CheckOpen() const;
	ClientContext &Context();

	std::shared_ptr<DuckDBPyConnection> connection;
	unique_ptr<BaseAppender> appender;
};

} // namespace duckdb

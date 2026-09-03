#include "duckdb_python/pyappender.hpp"
#include "duckdb_python/python_conversion.hpp"

namespace duckdb {

DuckDBPyAppender::DuckDBPyAppender(std::shared_ptr<DuckDBPyConnection> connection_p,
                                   unique_ptr<BaseAppender> appender_p)
    : connection(std::move(connection_p)), appender(std::move(appender_p)) {
}

DuckDBPyAppender::~DuckDBPyAppender() {
	if (!appender) {
		return;
	}
	try {
		appender->Close();
	} catch (...) { // NOLINT
	}
	appender.reset();
}

void DuckDBPyAppender::Initialize(nb::handle &m) {
	auto appender_module = nb::class_<DuckDBPyAppender>(m, "Appender", nb::is_weak_referenceable());
	appender_module.def("append", &DuckDBPyAppender::AppendRow, "Append a row of values")
	    .def("flush", &DuckDBPyAppender::Flush, "Flush the appender to the table")
	    .def("close", &DuckDBPyAppender::Close, "Flush the appender and close it")
	    .def("__enter__", &DuckDBPyAppender::Enter)
	    .def("__exit__", &DuckDBPyAppender::Exit, nb::arg("exc_type").none(), nb::arg("exc").none(),
	         nb::arg("traceback").none())
	    .def_prop_ro("column_count", &DuckDBPyAppender::ColumnCount, "Number of columns in the appender");
}

void DuckDBPyAppender::CheckOpen() const {
	if (!appender) {
		throw InvalidInputException("This appender has been closed");
	}
}

ClientContext &DuckDBPyAppender::Context() {
	return *connection->con.GetConnection().context;
}

void DuckDBPyAppender::AppendRow(const nb::args &args) {
	DuckDBPyConnection::ConnectionLockGuard conn_lock(*connection);
	CheckOpen();
	auto &types = appender->GetActiveTypes();
	if (args.size() != types.size()) {
		throw InvalidInputException("appender.append expected %d values, got %d", types.size(), args.size());
	}
	vector<Value> values;
	values.reserve(types.size());
	idx_t i = 0;
	for (auto value : args) {
		values.push_back(TransformPythonValue(Context(), value, types[i]));
		i++;
	}
	appender->BeginRow();
	for (auto &value : values) {
		appender->Append(std::move(value));
	}
	appender->EndRow();
}

void DuckDBPyAppender::Flush() {
	DuckDBPyConnection::ConnectionLockGuard conn_lock(*connection);
	CheckOpen();
	nb::gil_scoped_release release;
	appender->Flush();
}

void DuckDBPyAppender::Close() {
	DuckDBPyConnection::ConnectionLockGuard conn_lock(*connection);
	if (!appender) {
		return;
	}
	{
		nb::gil_scoped_release release;
		appender->Close();
	}
	appender.reset();
}

std::shared_ptr<DuckDBPyAppender> DuckDBPyAppender::Enter() {
	DuckDBPyConnection::ConnectionLockGuard conn_lock(*connection);
	CheckOpen();
	return shared_from_this();
}

void DuckDBPyAppender::Exit(const nb::object &, const nb::object &, const nb::object &) {
	Close();
}

idx_t DuckDBPyAppender::ColumnCount() {
	DuckDBPyConnection::ConnectionLockGuard conn_lock(*connection);
	CheckOpen();
	return appender->GetActiveTypes().size();
}

std::shared_ptr<DuckDBPyAppender> DuckDBPyConnection::CreateAppender(const string &table, std::optional<string> schema,
                                                                     std::optional<string> catalog) {
	if (catalog.has_value() && !schema.has_value()) {
		throw InvalidInputException("catalog requires schema");
	}
	DuckDBPyConnection::ConnectionLockGuard conn_lock(*this);
	auto &con = this->con.GetConnection();
	unique_ptr<Appender> appender;
	{
		nb::gil_scoped_release release;
		if (catalog.has_value()) {
			appender = make_uniq<Appender>(con, Identifier(*catalog), Identifier(*schema), Identifier(table));
		} else if (schema.has_value()) {
			appender = make_uniq<Appender>(con, Identifier(*schema), Identifier(table));
		} else {
			appender = make_uniq<Appender>(con, Identifier(table));
		}
	}
	return std::make_shared<DuckDBPyAppender>(shared_from_this(), std::move(appender));
}

} // namespace duckdb

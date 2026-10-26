#include "duckdb_python/pyappender.hpp"

#include "duckdb_python/engine/connection.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_conversion.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb_python/pyutil.hpp"
#include "duckdb_python/staged_rows.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry_retriever.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/sql_identifier.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/cast/cast_function_set.hpp"
#include "duckdb/main/appender.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/parser/query_node/delete_query_node.hpp"
#include "duckdb/parser/query_node/insert_query_node.hpp"
#include "duckdb/parser/query_node/merge_query_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node/update_query_node.hpp"
#include "duckdb/parser/statement/delete_statement.hpp"
#include "duckdb/parser/statement/insert_statement.hpp"
#include "duckdb/parser/statement/merge_into_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/statement/update_statement.hpp"
#include "duckdb/planner/binder.hpp"

#include <algorithm>

namespace duckdb {

static constexpr const char *BUFFER_NAME = "appender_buffer";
static constexpr const char *SOURCE_NAME = "appender_data";
static constexpr const char *DEFAULT_SOURCE = "SELECT * FROM appender_buffer";
// An upsert, update or delete names its target under this alias, so the target's columns stay apart from the source's
// whatever the table is called
static constexpr const char *TARGET_ALIAS = "appender_target";

static constexpr const char *BUSY_ERROR =
    "appender is busy with another call; it refuses instead of waiting, so give each thread its own appender";
static constexpr const char *BROKEN_ERROR =
    "a write into the appender's buffer failed, so it holds an unknown part of its rows; call clear() before using "
    "it again";

DuckDBPyAppender::Claim::Claim(DuckDBPyAppender &appender_p) : Claim(appender_p, std::nothrow) {
	if (!held) {
		throw ResourceInUseException(BUSY_ERROR);
	}
}

DuckDBPyAppender::Claim::Claim(DuckDBPyAppender &appender_p, std::nothrow_t)
    : appender(appender_p), held(!appender.busy.exchange(true)) {
}

DuckDBPyAppender::Claim::~Claim() {
	if (held) {
		appender.busy = false;
	}
}

static CommonTableExpressionMap &DmlCTEMap(SQLStatement &statement) {
	switch (statement.type) {
	case StatementType::INSERT_STATEMENT:
		return statement.Cast<InsertStatement>().node->cte_map;
	case StatementType::UPDATE_STATEMENT:
		return statement.Cast<UpdateStatement>().node->cte_map;
	case StatementType::DELETE_STATEMENT:
		return statement.Cast<DeleteStatement>().node->cte_map;
	case StatementType::MERGE_INTO_STATEMENT:
		return statement.Cast<MergeIntoStatement>().node->cte_map;
	default:
		throw InvalidInputException("an appender's DML is one INSERT, UPDATE, DELETE or MERGE INTO statement, not %s",
		                            StatementTypeToString(statement.type));
	}
}

static unique_ptr<SQLStatement> ParseOne(ClientContext &context, const string &text, const char *what) {
	Parser parser(context);
	parser.ParseQuery(text);
	if (parser.statements.size() != 1) {
		throw InvalidInputException("an appender's %s is exactly one statement, not %d", what,
		                            parser.statements.size());
	}
	auto statement = std::move(parser.statements[0]);
	if (!statement->named_param_map.empty()) {
		throw InvalidInputException("an appender's %s cannot hold a parameter: a flush has no value to bind to it",
		                            what);
	}
	return statement;
}

static unique_ptr<SQLStatement> FlushStatement(ClientContext &context, ColumnDataCollection &buffer,
                                               const vector<Identifier> &names, const string &source,
                                               const string &dml) {
	auto statement = ParseOne(context, dml, "DML");
	auto &cte_map = DmlCTEMap(*statement);
	for (auto name : {BUFFER_NAME, SOURCE_NAME}) {
		if (cte_map.map.contains(Identifier(name))) {
			throw InvalidInputException("the DML defines a CTE named %s, a name the appender reserves", name);
		}
	}
	auto source_statement = ParseOne(context, source, "source");
	if (source_statement->type != StatementType::SELECT_STATEMENT) {
		throw InvalidInputException("an appender's source is a query, not %s",
		                            StatementTypeToString(source_statement->type));
	}

	auto buffer_node = make_uniq<SelectNode>();
	buffer_node->select_list.push_back(make_uniq<StarExpression>());
	buffer_node->from_table = BaseAppender::GetColumnDataTableRef(buffer, Identifier(BUFFER_NAME), names);
	auto buffer_cte = make_uniq<CommonTableExpressionInfo>();
	buffer_cte->query_node = std::move(buffer_node);
	buffer_cte->materialized = CTEMaterialize::CTE_MATERIALIZE_NEVER;
	auto source_cte = make_uniq<CommonTableExpressionInfo>();
	source_cte->query_node = std::move(source_statement->Cast<SelectStatement>().node);

	// A CTE reads only the CTEs defined before it
	CommonTableExpressionMap ordered;
	ordered.map.insert(Identifier(BUFFER_NAME), std::move(buffer_cte));
	ordered.map.insert(Identifier(SOURCE_NAME), std::move(source_cte));
	for (auto &entry : cte_map.map) {
		ordered.map.insert(entry.first, std::move(entry.second));
	}
	cte_map = std::move(ordered);
	return statement;
}

namespace {

struct ResolvedTable {
	QualifiedName name;
	vector<ColumnDefinition> columns;
};

} // namespace

static ResolvedTable ResolveTable(ClientContext &context, const string &text) {
	QualifiedName parsed;
	string parse_error;
	if (!QualifiedName::TryParse(text, parsed, parse_error)) {
		throw ParserException(parse_error);
	}
	ResolvedTable result;
	ErrorData refusal;
	context.RunFunctionInTransaction(
	    [&]() {
		    // Caught inside the transaction function, a refusal leaves the caller's transaction usable, as a failed
		    // bind does
		    try {
			    CatalogEntryRetriever retriever(context);
			    auto &table = Catalog::GetEntry<TableCatalogEntry>(context, Binder::BindTableName(retriever, parsed));
			    auto &table_catalog = table.ParentCatalog();
			    if (table_catalog.GetAttached().IsReadOnly()) {
				    throw InvalidInputException("cannot append to table \"%s\": database \"%s\" is attached read-only",
				                                table.name.GetIdentifierName(),
				                                table_catalog.GetName().GetIdentifierName());
			    }
			    result.name = table.ParentSchema().GetQualifiedName(table.name);
			    for (auto &column : table.GetColumns().Logical()) {
				    result.columns.push_back(column.Copy());
			    }
		    } catch (std::exception &ex) {
			    ErrorData error(ex);
			    if (Exception::InvalidatesDatabase(error.Type())) {
				    throw;
			    }
			    refusal = std::move(error);
		    }
	    },
	    false);
	if (refusal.HasError()) {
		refusal.Throw();
	}
	return result;
}

namespace {

enum class TableAction : uint8_t { INSERT_ROWS, UPSERT_ROWS, UPDATE_ROWS, DELETE_ROWS };

} // namespace

static TableAction ParseAction(const string &action) {
	if (action == "insert") {
		return TableAction::INSERT_ROWS;
	}
	if (action == "upsert") {
		return TableAction::UPSERT_ROWS;
	}
	if (action == "update") {
		return TableAction::UPDATE_ROWS;
	}
	if (action == "delete") {
		return TableAction::DELETE_ROWS;
	}
	throw InvalidInputException("action is \"insert\", \"upsert\", \"update\" or \"delete\", not \"%s\"", action);
}

static bool MatchesOnKey(TableAction action) {
	return action == TableAction::UPDATE_ROWS || action == TableAction::DELETE_ROWS;
}

static optional_ptr<const ColumnDefinition> FindColumn(const ResolvedTable &table, const Identifier &name) {
	auto column = std::find_if(table.columns.begin(), table.columns.end(),
	                           [&](const ColumnDefinition &candidate) { return candidate.Name() == name; });
	return column == table.columns.end() ? nullptr : &*column;
}

static vector<reference<const ColumnDefinition>> ResolveColumns(const ResolvedTable &table,
                                                                const vector<string> &listed, const char *argument) {
	auto table_name = table.name.ToString();
	vector<reference<const ColumnDefinition>> result;
	identifier_set_t seen;
	for (auto &name : listed) {
		Identifier identifier(name);
		auto column = FindColumn(table, identifier);
		if (!column) {
			throw InvalidInputException("%s is not a column of %s", name, table_name);
		}
		if (column->Generated()) {
			throw InvalidInputException("%s is a generated column of %s: its value is computed, so %s cannot name it",
			                            name, table_name, argument);
		}
		if (!seen.insert(identifier).second) {
			throw InvalidInputException("%s names %s twice", argument, name);
		}
		result.push_back(*column);
	}
	return result;
}

static vector<reference<const ColumnDefinition>> BufferColumns(const ResolvedTable &table,
                                                               const std::optional<vector<string>> &requested) {
	if (!requested) {
		vector<reference<const ColumnDefinition>> result;
		for (auto &column : table.columns) {
			if (!column.Generated()) {
				result.push_back(column);
			}
		}
		return result;
	}
	if (requested->empty()) {
		throw InvalidInputException("columns names none of the columns of %s; leave it out to append to all of them",
		                            table.name.ToString());
	}
	return ResolveColumns(table, *requested, "columns");
}

static vector<Identifier> ColumnsToSet(const vector<Identifier> &names, const vector<string> &key,
                                       const vector<Identifier> &key_names) {
	for (idx_t i = 0; i < key_names.size(); i++) {
		if (std::find(names.begin(), names.end(), key_names[i]) == names.end()) {
			throw InvalidInputException(
			    "key column %s is not one of the appender's columns, so no row has a value to match it on", key[i]);
		}
	}
	vector<Identifier> assigned;
	for (auto &name : names) {
		if (std::find(key_names.begin(), key_names.end(), name) == key_names.end()) {
			assigned.push_back(name);
		}
	}
	if (assigned.empty()) {
		throw InvalidInputException("an update sets the appender's columns other than the key, and there are none");
	}
	return assigned;
}

static string TableDML(TableAction action, const QualifiedName &table, const vector<Identifier> &names,
                       const vector<Identifier> &key, const vector<Identifier> &assigned) {
	auto quoted = [](const Identifier &name) {
		return SQLIdentifier::ToString(name);
	};
	auto match = [&](const Identifier &name) {
		return StringUtil::Format("%s.%s = %s.%s", TARGET_ALIAS, quoted(name), SOURCE_NAME, quoted(name));
	};
	switch (action) {
	case TableAction::INSERT_ROWS:
		return StringUtil::Format("INSERT INTO %s (%s) FROM %s", table.ToString(),
		                          StringUtil::Join(names, ", ", quoted), SOURCE_NAME);
	case TableAction::UPSERT_ROWS:
		// The engine rewrites this into a merge that names the target by its bare name beside its own alias excluded,
		// which collides for a table called excluded unless the target has an alias
		return StringUtil::Format("INSERT OR REPLACE INTO %s AS %s (%s) FROM %s", table.ToString(), TARGET_ALIAS,
		                          StringUtil::Join(names, ", ", quoted), SOURCE_NAME);
	case TableAction::UPDATE_ROWS: {
		auto assignment = [&](const Identifier &name) {
			return StringUtil::Format("%s = %s.%s", quoted(name), SOURCE_NAME, quoted(name));
		};
		return StringUtil::Format("UPDATE %s AS %s SET %s FROM %s WHERE %s", table.ToString(), TARGET_ALIAS,
		                          StringUtil::Join(assigned, ", ", assignment), SOURCE_NAME,
		                          StringUtil::Join(key, " AND ", match));
	}
	case TableAction::DELETE_ROWS:
		return StringUtil::Format("DELETE FROM %s AS %s USING %s WHERE %s", table.ToString(), TARGET_ALIAS, SOURCE_NAME,
		                          StringUtil::Join(key, " AND ", match));
	}
	throw InternalException("unknown table action");
}

DuckDBPyAppender::DuckDBPyAppender(std::shared_ptr<DuckDBPyConnection> connection_p, ClientContext &context,
                                   vector<Identifier> names_p, vector<LogicalType> types_p)
    : connection(std::move(connection_p)), names(std::move(names_p)), types(std::move(types_p)),
      buffer(make_uniq<ColumnDataCollection>(context, types, ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR,
                                             ColumnDataCollectionLifetime::THROW_ERROR_AFTER_DATABASE_CLOSES)) {
	staged.Initialize(Allocator::DefaultAllocator(), types);
}

// Checked before any lookup, so the refusal does not depend on whether a type name needed the transaction, and
// again before the bind, because converting a name or a type can run Python code that aborts the transaction
static void RefuseInsideAnAbortedTransaction(ClientContext &context) {
	DUCKDB_PY_ASSERT_GIL_HELD();
	nb::gil_scoped_release release;
	engine::ConnectionClaim claim(context);
	context.RunFunctionInTransaction([]() {}, true);
}

// A factory checks again once its names and types are converted, because converting them can run Python code that
// closes the connection
static void RefuseClosed(const DuckDBPyConnection &connection) {
	if (connection.con.ConnectionIsClosed()) {
		throw ConnectionException("Connection already closed!");
	}
}

std::unique_ptr<DuckDBPyAppender> DuckDBPyAppender::Create(std::shared_ptr<DuckDBPyConnection> connection,
                                                           shared_ptr<ClientContext> context, vector<Identifier> names,
                                                           vector<LogicalType> types, const string &source,
                                                           const string &dml) {
	RefuseInsideAnAbortedTransaction(*context);
	std::unique_ptr<DuckDBPyAppender> appender(
	    new DuckDBPyAppender(std::move(connection), *context, std::move(names), std::move(types)));
	engine::Session session(context);
	DUCKDB_PY_ASSERT_GIL_HELD();
	nb::gil_scoped_release release;
	appender->statement = FlushStatement(*context, *appender->buffer, appender->names, source, dml);
	auto signature = session.Bind(appender->statement->Copy(), appender->tables);
	if (signature.properties.return_type == StatementReturnType::QUERY_RESULT) {
		throw InvalidInputException("an appender's DML cannot return rows: a flush has nowhere to put them");
	}
	return appender;
}

static std::optional<vector<string>> ListedNames(const std::optional<nb::list> &list) {
	if (!list) {
		return std::nullopt;
	}
	// Converting a name can run Python code that changes the list, so the names come from one copy of it
	vector<nb::object> listed;
	for (auto name : *list) {
		listed.push_back(nb::borrow<nb::object>(name));
	}
	vector<string> names;
	for (auto &name : listed) {
		names.push_back(duckdb::PyUtil::CastToString(name));
	}
	return names;
}

std::unique_ptr<DuckDBPyAppender> DuckDBPyAppender::ForTable(std::shared_ptr<DuckDBPyConnection> connection,
                                                             const string &table,
                                                             const std::optional<nb::list> &columns,
                                                             const string &action_name,
                                                             const std::optional<nb::list> &key) {
	Held held {connection->con.ShareDatabase(), connection->con.GetConnection().context};
	RefuseInsideAnAbortedTransaction(*held.context);
	auto requested = ListedNames(columns);
	auto keyed = ListedNames(key);
	RefuseClosed(*connection);
	auto action = ParseAction(action_name);
	if (keyed && !MatchesOnKey(action)) {
		throw InvalidInputException("key applies to the update and delete actions, not to %s", action_name);
	}
	if (MatchesOnKey(action) && (!keyed || keyed->empty())) {
		throw InvalidInputException("the %s action needs key, the columns that pick the rows to %s", action_name,
		                            action_name);
	}
	if (requested && action == TableAction::DELETE_ROWS) {
		throw InvalidInputException("the delete action buffers exactly its key columns, so it takes no columns");
	}
	ResolvedTable resolved;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		engine::ConnectionClaim claim(*held.context);
		resolved = ResolveTable(*held.context, table);
	}
	vector<reference<const ColumnDefinition>> key_columns;
	if (keyed) {
		key_columns = ResolveColumns(resolved, *keyed, "key");
	}
	vector<Identifier> key_names;
	for (auto &column : key_columns) {
		key_names.push_back(column.get().Name());
	}
	vector<Identifier> names;
	vector<LogicalType> types;
	for (auto &column : action == TableAction::DELETE_ROWS ? key_columns : BufferColumns(resolved, requested)) {
		names.push_back(column.get().Name());
		types.push_back(column.get().Type());
	}
	vector<Identifier> assigned;
	if (action == TableAction::UPDATE_ROWS) {
		assigned = ColumnsToSet(names, *keyed, key_names);
	}
	auto dml = TableDML(action, resolved.name, names, key_names, assigned);
	return Create(std::move(connection), held.context, std::move(names), std::move(types), DEFAULT_SOURCE, dml);
}

std::unique_ptr<DuckDBPyAppender> DuckDBPyAppender::ForQuery(std::shared_ptr<DuckDBPyConnection> connection,
                                                             const string &dml, const nb::dict &columns,
                                                             const std::optional<string> &source) {
	Held held {connection->con.ShareDatabase(), connection->con.GetConnection().context};
	RefuseInsideAnAbortedTransaction(*held.context);
	// Converting a name or a type can run Python code that changes the dict, so both passes read one copy of it
	vector<std::pair<nb::object, nb::object>> declared;
	for (auto item : columns) {
		declared.emplace_back(nb::borrow<nb::object>(item.first), nb::borrow<nb::object>(item.second));
	}
	if (declared.empty()) {
		throw InvalidInputException("columns declares no column; an appender buffers at least one");
	}
	vector<Identifier> names;
	identifier_set_t seen;
	// The bind catches a repeat only when something reads the buffer; otherwise it skips the buffer CTE
	for (auto &item : declared) {
		names.emplace_back(duckdb::PyUtil::CastToString(item.first));
		if (names.back().empty()) {
			throw InvalidInputException("columns has an empty column name; every buffer column needs a name");
		}
		if (!seen.insert(names.back()).second) {
			throw BinderException("columns has the duplicate column name \"%s\"; names are compared ignoring case",
			                      names.back().GetIdentifierName());
		}
	}
	vector<LogicalType> types;
	for (auto &item : declared) {
		auto type = DuckDBPyType::Resolve(item.second, connection);
		if (type.id() == LogicalTypeId::SQLNULL) {
			throw InvalidInputException("column %s cannot have the type NULL, which holds no values",
			                            names[types.size()].GetIdentifierName());
		}
		types.push_back(std::move(type));
	}
	RefuseClosed(*connection);
	return Create(std::move(connection), held.context, std::move(names), std::move(types),
	              source ? *source : DEFAULT_SOURCE, dml);
}

DuckDBPyAppender::~DuckDBPyAppender() {
	auto rows = BufferedRows();
	if (rows == 0) {
		return;
	}
	nb::error_scope scope;
	auto message = StringUtil::Format(
	    "an appender with %llu unflushed rows was garbage collected, so they were dropped; call flush() or clear() "
	    "before letting go of it",
	    rows);
	if (PyErr_WarnEx(PyExc_ResourceWarning, message.c_str(), 1) != 0) {
		PyErr_WriteUnraisable(nullptr);
	}
}

void DuckDBPyAppender::ThrowIfClosed() const {
	RefuseClosed(*connection);
}

DuckDBPyAppender::Held DuckDBPyAppender::Hold() const {
	return {connection->con.ShareDatabase(), connection->con.GetConnectionWhileBusy().context};
}

void DuckDBPyAppender::ThrowIfBroken() const {
	if (broken) {
		throw InvalidInputException(BROKEN_ERROR);
	}
}

Value DuckDBPyAppender::Convert(ClientContext &context, const nb::handle &object, const LogicalType &type) {
	return CastToColumn(context, TransformPythonValue(context, object, LogicalType::UNKNOWN, false), type);
}

void DuckDBPyAppender::Stage(const vector<Value> &row) {
	if (StageRow(staged, row)) {
		MoveStagedIntoBuffer();
	}
}

void DuckDBPyAppender::MoveStagedIntoBuffer() {
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		try {
			buffer->Append(staged);
		} catch (...) {
			broken = true;
			throw;
		}
	}
	staged.Reset();
}

void DuckDBPyAppender::AppendSequence(ClientContext &context, const nb::handle &row, idx_t count) {
	if (count != types.size()) {
		throw InvalidInputException("a row of this appender holds %d values, one per column, not %d", types.size(),
		                            count);
	}
	vector<Value> values;
	values.reserve(count);
	for (idx_t i = 0; i < count; i++) {
		nb::object item = nb::steal(PySequence_GetItem(row.ptr(), NumericCast<Py_ssize_t>(i)));
		if (!item) {
			throw nb::python_error();
		}
		values.push_back(Convert(context, item, types[i]));
	}
	Stage(values);
}

void DuckDBPyAppender::AppendRow(const nb::args &values) {
	Claim claim(*this);
	ThrowIfClosed();
	ThrowIfBroken();
	auto held = Hold();
	AppendSequence(*held.context, values, values.size());
}

void DuckDBPyAppender::AppendRows(const nb::handle &rows) {
	Claim claim(*this);
	ThrowIfClosed();
	ThrowIfBroken();
	auto held = Hold();
	for (auto row : rows) {
		ThrowIfClosed();
		// A str, bytes or dict iterates too, so taken as a row it would split into characters, bytes or keys
		if (nb::isinstance<nb::str>(row) || nb::isinstance<nb::bytes>(row) || nb::isinstance<nb::dict>(row) ||
		    !PySequence_Check(row.ptr())) {
			throw InvalidInputException("append_rows takes rows as sequences of values, not a %s",
			                            nb::type_name(row.type()).c_str());
		}
		auto count = PySequence_Size(row.ptr());
		if (count < 0) {
			throw nb::python_error();
		}
		AppendSequence(*held.context, row, NumericCast<idx_t>(count));
	}
}

void DuckDBPyAppender::Flush() {
	Claim claim(*this);
	ThrowIfClosed();
	ThrowIfBroken();
	auto held = Hold();
	if (staged.size() > 0) {
		MoveStagedIntoBuffer();
	}
	if (buffer->Count() == 0) {
		return;
	}
	engine::Session session(held.context);
	bool signalled = false;
	auto check = [&signalled]() {
		try {
			DuckDBPyConnection::CheckSignals();
		} catch (...) {
			signalled = true;
			throw;
		}
	};
	DUCKDB_PY_ASSERT_GIL_HELD();
	nb::gil_scoped_release release;
	try {
		session.Run(statement->Copy(), tables, check);
	} catch (std::exception &ex) {
		ErrorData error(ex);
		// A refused or interrupted flush keeps its rows for a retry, as the stable C++ Appender does
		auto kept =
		    signalled || error.Type() == ExceptionType::RESOURCE_IN_USE || error.Type() == ExceptionType::INTERRUPT;
		if (!kept) {
			DropRows();
		}
		throw;
	}
	DropRows();
}

void DuckDBPyAppender::DropRows() {
	buffer->Reset();
	staged.Reset();
	broken = false;
}

void DuckDBPyAppender::Clear() {
	Claim claim(*this);
	ThrowIfClosed();
	auto held = Hold();
	DropRows();
}

void DuckDBPyAppender::Enter() {
	Claim claim(*this);
	ThrowIfClosed();
}

void DuckDBPyAppender::Exit(const nb::handle &exc_type) {
	if (exc_type.is_none()) {
		Flush();
		return;
	}
	// A refusal raised here would replace the exception that ended the block
	Claim claim(*this, std::nothrow);
	if (!claim.Held()) {
		return;
	}
	if (connection->con.ConnectionIsClosed()) {
		// Resetting the buffer reaches the buffer manager of a database that may be gone; destroying it does not
		statement.reset();
		buffer.reset();
		staged.Reset();
		return;
	}
	auto held = Hold();
	DropRows();
}

idx_t DuckDBPyAppender::BufferedRows() const {
	return (buffer ? buffer->Count() : 0) + staged.size();
}

void DuckDBPyAppender::Initialize(nb::handle &m) {
	auto appender = nb::class_<DuckDBPyAppender>(m, "Appender", nb::is_weak_referenceable());
	appender.def("append_row", &DuckDBPyAppender::AppendRow,
	             "Buffer one row, one value per column, each converted as an executemany parameter would be");
	appender.def("append_rows", &DuckDBPyAppender::AppendRows,
	             "Buffer each row of an iterable of sequences, stopping at the first bad row and keeping the rows "
	             "before it",
	             nb::arg("rows"));
	appender.def("flush", &DuckDBPyAppender::Flush,
	             "Apply the rows buffered since the last flush with one statement on the appender's connection");
	appender.def("clear", &DuckDBPyAppender::Clear, "Drop the buffered rows");
	appender.def("__enter__", [](nb::object self) {
		nb::cast<DuckDBPyAppender &>(self).Enter();
		return self;
	});
	appender.def(
	    "__exit__",
	    [](DuckDBPyAppender &self, const nb::handle &exc_type, const nb::handle &, const nb::handle &) {
		    self.Exit(exc_type);
	    },
	    nb::arg("exc_type").none(), nb::arg("exc").none(), nb::arg("traceback").none());
}

} // namespace duckdb

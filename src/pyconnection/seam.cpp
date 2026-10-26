#include "duckdb_python/pyconnection/seam.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/view_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry_retriever.hpp"
#include "duckdb/catalog/entry_lookup_info.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/box_renderer_context.hpp"
#include "duckdb/common/sql_identifier.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/query_node/cte_node.hpp"
#include "duckdb/parser/query_node/recursive_cte_node.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/expressionlistref.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/pivotref.hpp"
#include "duckdb/parser/tableref/showref.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/result_modifier.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/column_data_ref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb_python/map.hpp"
#include "duckdb_python/path_like.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/pystatement.hpp"
#include "duckdb_python/staged_rows.hpp"
#include "duckdb_python/python_conversion.hpp"
#include "duckdb_python/python_replacement_scan.hpp"
#include "duckdb_python/pytype.hpp"

namespace duckdb {

SourceHandle::SourceHandle(string name_p, weak_ptr<engine::SourceRegistry> registry_p,
                           shared_ptr<ExternalDependency> keep_alive_p)
    : name(std::move(name_p)), registry(std::move(registry_p)), keep_alive(std::move(keep_alive_p)) {
}

void SourceHandle::RememberFor(const std::shared_ptr<SourceIdentity> &identity_p, const nb::handle &object) {
	identity = identity_p;
	object_id = object.ptr();
}

SourceHandle::~SourceHandle() {
	if (object_id) {
		if (auto remembered = identity.lock()) {
			remembered->Forget(object_id, name);
		}
	}
	auto sources = registry.lock();
	if (sources) {
		// Destroyed here rather than under the registry's lock, since it can own Python objects
		auto table = sources->Unregister(name);
	}
}

bool SourceIdentity::NamesLiveObject(const Entry &entry, PyObject *object_id) {
	if (!entry.weak) {
		return true;
	}
	// Reading a weak reference runs no Python code of its own, so it is safe under the lock
	return entry.reference->obj().ptr() == object_id;
}

std::shared_ptr<SourceHandle> SourceIdentity::Lookup(const nb::handle &object) {
	DUCKDB_PY_ASSERT_GIL_HELD();
	// Destroyed after the lock is released, since dropping a strong reference can run Python code
	unique_ptr<PyObjectHolder> dying;
	lock_guard<mutex> guard(lock);
	auto entry = entries.find(object.ptr());
	if (entry == entries.end()) {
		return nullptr;
	}
	auto handle = entry->second.handle.lock();
	if (handle && NamesLiveObject(entry->second, object.ptr())) {
		return handle;
	}
	// The source is gone, or the interpreter gave the address to another object, whose source this is not
	dying = std::move(entry->second.reference);
	entries.erase(entry);
	return nullptr;
}

void SourceIdentity::Remember(const nb::handle &object, const std::shared_ptr<SourceHandle> &handle) {
	DUCKDB_PY_ASSERT_GIL_HELD();
	bool weak = true;
	nb::object reference;
	try {
		reference = nb::weakref(object);
	} catch (nb::python_error &) {
		// A type that takes no weak reference, a dict or a list, which the source's own scan keeps alive anyway
		weak = false;
		reference = nb::borrow<nb::object>(object);
	}
	unique_ptr<PyObjectHolder> dying;
	lock_guard<mutex> guard(lock);
	auto entry = entries.find(object.ptr());
	if (entry != entries.end()) {
		dying = std::move(entry->second.reference);
		entries.erase(entry);
	}
	entries.emplace(object.ptr(),
	                Entry {handle->Name(), handle, make_uniq<PyObjectHolder>(std::move(reference)), weak});
}

void SourceIdentity::Forget(PyObject *object_id, const string &name) {
	unique_ptr<PyObjectHolder> dying;
	lock_guard<mutex> guard(lock);
	auto entry = entries.find(object_id);
	// The address may name an entry of another object, or of a later registration of this one, which both stay
	if (entry == entries.end() || entry->second.name != name) {
		return;
	}
	dying = std::move(entry->second.reference);
	entries.erase(entry);
}

static engine::Session OpenSession(DuckDBPyConnection &self) {
	if (self.con.ConnectionIsClosed()) {
		throw ConnectionException("Connection has already been closed");
	}
	return self.EngineSession();
}

static Connection &OpenConnection(DuckDBPyConnection &self) {
	if (self.con.ConnectionIsClosed()) {
		throw ConnectionException("Connection has already been closed");
	}
	return self.con.GetConnection();
}

static unique_ptr<SQLStatement> ParseOneStatement(const engine::Session &session, const string &sql) {
	// Parsing alone, so an error leaves an explicit transaction usable
	engine::StatementGroup statements(session, sql, false);
	vector<unique_ptr<SQLStatement>> parsed;
	while (auto statement = statements.Next()) {
		parsed.push_back(std::move(statement));
	}
	if (parsed.size() != 1) {
		throw InvalidInputException("Expected exactly one SQL statement, found %llu", parsed.size());
	}
	return std::move(parsed[0]);
}

//! How many statements a text holds, as the parser that runs them splits it
static idx_t StatementCount(DuckDBPyConnection &self, const string &sql) {
	auto session = OpenSession(self);
	DUCKDB_PY_ASSERT_GIL_HELD();
	nb::gil_scoped_release release;
	engine::StatementGroup statements(session, sql, false);
	idx_t count = 0;
	while (statements.Next()) {
		count++;
	}
	return count;
}

static unique_ptr<SelectStatement> ParseSelect(ClientContext &context, const string &sql) {
	Parser parser(context);
	parser.ParseQuery(sql);
	if (parser.statements.size() != 1) {
		throw InvalidInputException("Expected exactly one SELECT statement, found %llu statements",
		                            parser.statements.size());
	}
	if (parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		// A PIVOT whose values come from the data parses into several statements, which no view can hold
		throw InvalidInputException("Expected exactly one SELECT statement, found a %s statement",
		                            StatementTypeToString(parser.statements[0]->type));
	}
	return unique_ptr_cast<SQLStatement, SelectStatement>(std::move(parser.statements[0]));
}

static const char *ReturnTypeName(StatementReturnType type) {
	switch (type) {
	case StatementReturnType::QUERY_RESULT:
		return "query_result";
	case StatementReturnType::CHANGED_ROWS:
		return "changed_rows";
	case StatementReturnType::NOTHING:
		return "nothing";
	}
	throw InternalException("Unknown StatementReturnType");
}

static std::shared_ptr<SourceHandle> RegisterTable(ClientContext &context, unique_ptr<TableRef> table,
                                                   const string &subject = string()) {
	auto registry = engine::SourceRegistry::Get(context);
	auto name = engine::SourceRegistry::NextName();
	// The handle first, so no entry is ever registered without the handle that unregisters it
	auto handle = std::make_shared<SourceHandle>(name, registry, nullptr);
	registry->Register(name, std::move(table), subject);
	return handle;
}

std::shared_ptr<SourceHandle> DuckDBPySeam::RegisterTableRef(ClientContext &context, unique_ptr<TableRef> table,
                                                             const string &subject) {
	return RegisterTable(context, std::move(table), subject);
}

static string StatementTypeName(StatementType type) {
	return StringUtil::Lower(StatementTypeToString(type));
}

//! The name as the relation layer gives it: one string is one identifier, however many dots it holds, and a tuple or
//! list is the parts of a qualified name, outermost first
static QualifiedName NameFromParts(const nb::object &name) {
	if (nb::isinstance<nb::str>(name)) {
		return QualifiedName(Identifier(nb::cast<std::string>(nb::str(name))));
	}
	if (!nb::isinstance<nb::tuple>(name) && !nb::isinstance<nb::list>(name)) {
		throw InvalidInputException("A name is given as a string, or as a tuple of its parts");
	}
	vector<Identifier> path;
	for (auto part : nb::borrow<nb::sequence>(name)) {
		if (!nb::isinstance<nb::str>(part)) {
			throw InvalidInputException("Every part of a name must be a string");
		}
		path.emplace_back(nb::cast<std::string>(nb::str(part)));
	}
	if (path.empty()) {
		throw InvalidInputException("A name must have at least one part");
	}
	return QualifiedName::FromPath(std::move(path));
}

//! The view register() made under this name, or null for any other name. Call in a transaction.
static optional_ptr<CatalogEntry> RegisteredView(ClientContext &context, const string &name) {
	EntryLookupInfo lookup(CatalogType::VIEW_ENTRY,
	                       QualifiedName(Identifier::TempCatalog(), Identifier::DefaultSchema(), Identifier(name)));
	auto entry = Catalog::GetEntry(context, lookup, OnEntryNotFound::RETURN_NULL);
	if (!entry || entry->type != CatalogType::VIEW_ENTRY ||
	    entry->tags.find(DuckDBPySeam::REGISTER_TAG) == entry->tags.end()) {
		return nullptr;
	}
	return entry;
}

//! The kind of the table reference a view register() made reads, which is its source's own, or INVALID for any
//! other name
static TableReferenceType RegisteredSourceKind(ClientContext &context, const string &name) {
	auto kind = TableReferenceType::INVALID;
	context.RunFunctionInTransaction([&]() {
		auto view = RegisteredView(context, name);
		if (!view) {
			return;
		}
		auto &node = *view->Cast<ViewCatalogEntry>().GetQuery().node;
		if (node.type == QueryNodeType::SELECT_NODE && node.Cast<SelectNode>().from_table) {
			kind = node.Cast<SelectNode>().from_table->type;
		}
	});
	return kind;
}

//! Writes every registry name a query reads as a copy of the table reference the name stands for. A Python object's
//! scan, a collection's rows and a relation's text then travel inside the query, which is what a view's catalog entry
//! must hold to own its sources: a name would leave the view reading a registry entry its transaction does not govern.
//! A copy renaming its columns as the reference it replaces did, the way the engine's replacement scans carry the list:
//! a table function or a subquery takes it itself, anything else is read through a subquery that takes it
static unique_ptr<TableRef> WithColumnAliases(unique_ptr<TableRef> copy, const vector<Identifier> &aliases) {
	if (copy->type == TableReferenceType::TABLE_FUNCTION || copy->type == TableReferenceType::SUBQUERY) {
		copy->column_name_alias = aliases;
		return copy;
	}
	auto alias = copy->alias;
	auto select = make_uniq<SelectNode>();
	select->select_list.push_back(make_uniq<StarExpression>());
	select->from_table = std::move(copy);
	auto statement = make_uniq<SelectStatement>();
	statement->node = std::move(select);
	auto subquery = make_uniq<SubqueryRef>(std::move(statement));
	subquery->alias = std::move(alias);
	subquery->column_name_alias = aliases;
	return std::move(subquery);
}

class SourceInliner {
public:
	SourceInliner(ClientContext &context_p, engine::SourceRegistry &registry_p)
	    : context(context_p), registry(registry_p) {
	}

	//! A worklist rather than recursion, so a view as deep as the engine binds is never refused for its depth
	void Node(QueryNode &root) {
		tasks.push_back(Task::OfNode(root));
		while (!tasks.empty()) {
			auto task = std::move(tasks.back());
			tasks.pop_back();
			switch (task.kind) {
			case Task::NODE:
				VisitNode(*task.node);
				break;
			case Task::REFERENCE:
				VisitReference(*task.reference);
				break;
			case Task::EXPRESSION:
				VisitExpression(*task.expression);
				break;
			case Task::DEFINE:
				scope.push_back(task.name);
				break;
			case Task::RESTORE:
				scope.resize(task.scope_size);
				break;
			}
		}
	}

	//! Whether something written in holds a Python object, which no database file can store
	bool HoldsProcessLocalInput() const {
		return process_local;
	}

	bool HoldsRows() const {
		return holds_rows;
	}

private:
	struct Task {
		enum Kind { NODE, REFERENCE, EXPRESSION, DEFINE, RESTORE } kind;
		QueryNode *node = nullptr;
		unique_ptr<TableRef> *reference = nullptr;
		unique_ptr<ParsedExpression> *expression = nullptr;
		Identifier name;
		idx_t scope_size = 0;

		static Task OfNode(QueryNode &node) {
			Task task {NODE};
			task.node = &node;
			return task;
		}
		static Task OfReference(unique_ptr<TableRef> &reference) {
			Task task {REFERENCE};
			task.reference = &reference;
			return task;
		}
		static Task OfExpression(unique_ptr<ParsedExpression> &expression) {
			Task task {EXPRESSION};
			task.expression = &expression;
			return task;
		}
		static Task Defining(const Identifier &name) {
			Task task {DEFINE};
			task.name = name;
			return task;
		}
		static Task Restoring(idx_t size) {
			Task task {RESTORE};
			task.scope_size = size;
			return task;
		}
	};

	//! Runs `steps` next, first to last, ahead of the tasks already waiting
	void Schedule(vector<Task> &steps) {
		for (auto step = steps.rbegin(); step != steps.rend(); ++step) {
			tasks.push_back(std::move(*step));
		}
	}

	void VisitNode(QueryNode &node) {
		vector<Task> steps;
		// A name a common table expression defines is the expression's, never a registry entry of the same name, from
		// where the engine lets it be read: the later expressions and the node, and its own body only when recursive
		for (auto &cte : node.cte_map.map) {
			auto &body = cte.second->query_node;
			auto recursive = body && body->type == QueryNodeType::RECURSIVE_CTE_NODE;
			if (recursive) {
				steps.push_back(Task::Defining(cte.first));
			}
			if (body) {
				steps.push_back(Task::OfNode(*body));
			}
			if (!recursive) {
				steps.push_back(Task::Defining(cte.first));
			}
		}
		ParsedExpressionIterator::EnumerateQueryNodeModifiers(
		    node, [&](unique_ptr<ParsedExpression> &child) { steps.push_back(Task::OfExpression(child)); });
		switch (node.type) {
		case QueryNodeType::SELECT_NODE: {
			auto &select = node.Cast<SelectNode>();
			for (auto &expression : select.select_list) {
				steps.push_back(Task::OfExpression(expression));
			}
			for (auto &expression : select.groups.group_expressions) {
				steps.push_back(Task::OfExpression(expression));
			}
			steps.push_back(Task::OfExpression(select.where_clause));
			steps.push_back(Task::OfExpression(select.having));
			steps.push_back(Task::OfExpression(select.qualify));
			if (select.from_table) {
				steps.push_back(Task::OfReference(select.from_table));
			}
			break;
		}
		case QueryNodeType::SET_OPERATION_NODE:
			for (auto &child : node.Cast<SetOperationNode>().children) {
				steps.push_back(Task::OfNode(*child));
			}
			break;
		case QueryNodeType::RECURSIVE_CTE_NODE: {
			auto &recursive = node.Cast<RecursiveCTENode>();
			steps.push_back(Task::OfNode(*recursive.left));
			steps.push_back(Task::OfNode(*recursive.right));
			break;
		}
		case QueryNodeType::CTE_NODE: {
			auto &cte = node.Cast<CTENode>();
			auto recursive = cte.query->type == QueryNodeType::RECURSIVE_CTE_NODE;
			if (recursive) {
				steps.push_back(Task::Defining(cte.ctename));
			}
			steps.push_back(Task::OfNode(*cte.query));
			if (!recursive) {
				steps.push_back(Task::Defining(cte.ctename));
			}
			steps.push_back(Task::OfNode(*cte.child));
			break;
		}
		default:
			throw NotImplementedException("A view over Python sources cannot be created from this kind of statement");
		}
		steps.push_back(Task::Restoring(scope.size()));
		Schedule(steps);
	}

	void VisitReference(unique_ptr<TableRef> &ref) {
		vector<Task> steps;
		switch (ref->type) {
		case TableReferenceType::BASE_TABLE: {
			auto &base = ref->Cast<BaseTableRef>();
			auto &qualified = base.GetQualifiedName();
			auto &table_name = qualified.Name().GetIdentifierName();
			if (qualified.Path().size() != 1 || Shadowed(qualified.Name())) {
				break;
			}
			if (!engine::SourceRegistry::IsSourceName(table_name)) {
				// A view register() made stays a name here, and is refused by its source's kind as a registry entry is
				auto registered = RegisteredSourceKind(context, table_name);
				if (registered != TableReferenceType::INVALID && registered != TableReferenceType::SUBQUERY) {
					process_local = true;
				}
				break;
			}
			auto copy = registry.Lookup(table_name);
			if (!copy) {
				registry.ThrowIfReleased(table_name);
				break;
			}
			// A relation's text is walked for what it reads in turn; any other source reads rows or an object that
			// live only in this process
			if (copy->type == TableReferenceType::COLUMN_DATA) {
				holds_rows = true;
			} else if (copy->type != TableReferenceType::SUBQUERY) {
				process_local = true;
			}
			// The query refers to the source by its name, so the written in reference keeps answering to it
			if (copy->alias.empty()) {
				copy->alias = base.alias.empty() ? qualified.Name() : base.alias;
			}
			if (base.sample && !copy->sample) {
				copy->sample = std::move(base.sample);
			}
			if (!base.column_name_alias.empty()) {
				copy = WithColumnAliases(std::move(copy), base.column_name_alias);
			}
			ref = std::move(copy);
			// An entry can itself read other entries: a relation's text, or the input of a map
			steps.push_back(Task::OfReference(ref));
			break;
		}
		case TableReferenceType::JOIN: {
			auto &join = ref->Cast<JoinRef>();
			steps.push_back(Task::OfReference(join.left));
			steps.push_back(Task::OfReference(join.right));
			steps.push_back(Task::OfExpression(join.condition));
			break;
		}
		case TableReferenceType::SUBQUERY:
			steps.push_back(Task::OfNode(*ref->Cast<SubqueryRef>().subquery->node));
			break;
		case TableReferenceType::PIVOT: {
			auto &pivot = ref->Cast<PivotRef>();
			steps.push_back(Task::OfReference(pivot.source));
			for (auto &aggregate : pivot.aggregates) {
				steps.push_back(Task::OfExpression(aggregate));
			}
			for (auto &column : pivot.pivots) {
				for (auto &expression : column.pivot_expressions) {
					steps.push_back(Task::OfExpression(expression));
				}
				if (column.subquery) {
					steps.push_back(Task::OfNode(*column.subquery));
				}
			}
			break;
		}
		case TableReferenceType::TABLE_FUNCTION:
			steps.push_back(Task::OfExpression(ref->Cast<TableFunctionRef>().function));
			break;
		case TableReferenceType::EXPRESSION_LIST:
			for (auto &row : ref->Cast<ExpressionListRef>().values) {
				for (auto &value : row) {
					steps.push_back(Task::OfExpression(value));
				}
			}
			break;
		case TableReferenceType::SHOW_REF: {
			auto &show = ref->Cast<ShowRef>();
			if (show.query) {
				steps.push_back(Task::OfNode(*show.query));
			}
			break;
		}
		default:
			break;
		}
		Schedule(steps);
	}

	void VisitExpression(unique_ptr<ParsedExpression> &expression) {
		if (!expression) {
			return;
		}
		vector<Task> steps;
		if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
			auto &subquery = expression->Cast<SubqueryExpression>().SubqueryMutable();
			if (subquery) {
				steps.push_back(Task::OfNode(*subquery->node));
			}
		}
		ParsedExpressionIterator::EnumerateChildren(
		    *expression, [&](unique_ptr<ParsedExpression> &child) { steps.push_back(Task::OfExpression(child)); });
		Schedule(steps);
	}

	bool Shadowed(const Identifier &name) const {
		for (auto &defined : scope) {
			if (defined == name) {
				return true;
			}
		}
		return false;
	}

private:
	ClientContext &context;
	engine::SourceRegistry &registry;
	//! The names of the common table expressions in scope where the walk stands
	vector<Identifier> scope;
	vector<Task> tasks;
	bool process_local = false;
	bool holds_rows = false;
};

//! Whether a view of this name would be written to a file, which cannot hold a Python object: the database its name
//! resolves to as the binder resolves a created name, a lone qualifier naming a database when one has that name
static bool StoredInAFile(ClientContext &context, const QualifiedName &name) {
	bool stored = false;
	context.RunFunctionInTransaction([&]() {
		CatalogEntryRetriever retriever(context);
		auto resolved = Binder::ResolveCatalog(retriever, name, true);
		auto catalog = Catalog::GetCatalogEntry(context, resolved.Path().front());
		stored = catalog && !catalog->InMemory();
	});
	return stored;
}

static string QuotedName(const QualifiedName &name) {
	vector<string> parts;
	for (auto &part : name.Path()) {
		parts.push_back(SQLQuotedIdentifier::ToString(part));
	}
	return StringUtil::Join(parts, ".");
}

unique_ptr<SQLStatement> DuckDBPySeam::ViewStatement(ClientContext &context, const QualifiedName &name,
                                                     const string &sql, bool temporary, bool replace, bool registered) {
	auto query = ParseSelect(context, sql);
	SourceInliner inliner(context, *engine::SourceRegistry::Get(context));
	inliner.Node(*query->node);
	// The engine's own parse of the CREATE VIEW, so a name is read as SQL reads it
	auto head = StringUtil::Format("CREATE %s%sVIEW %s AS SELECT 1", replace ? "OR REPLACE " : "",
	                               temporary ? "TEMPORARY " : "", QuotedName(name));
	Parser parser(context);
	parser.ParseQuery(head);
	auto statement = std::move(parser.statements[0]);
	auto &info = statement->Cast<CreateStatement>().info->Cast<CreateViewInfo>();
	if (!temporary && inliner.HoldsRows()) {
		throw InvalidInputException(
		    "Cannot create the view %s: it reads rows held in this process, which a view cannot store. Use "
		    "rel.create(name) to store them in a table, or con.register(name, rel) for a temporary view for the "
		    "life of the connection.",
		    QuotedName(name));
	}
	if (!temporary && inliner.HoldsProcessLocalInput() && StoredInAFile(context, name)) {
		throw InvalidInputException(
		    "Cannot create the view %s: it reads a Python object, which lives only in this process and cannot be "
		    "stored in a database file. Use register() for a view that reads it in this session.",
		    QuotedName(name));
	}
	info.query = std::move(query);
	if (registered) {
		info.tags[REGISTER_TAG] = "true";
	}
	return statement;
}

bool DuckDBPySeam::IsRegisteredView(ClientContext &context, const string &name) {
	bool registered = false;
	context.RunFunctionInTransaction([&]() { registered = RegisteredView(context, name) != nullptr; });
	return registered;
}

static void CreateView(DuckDBPyConnection &self, const nb::object &name, const string &sql, bool temporary,
                       bool replace) {
	auto qualified = NameFromParts(name);
	auto session = OpenSession(self);
	DUCKDB_PY_ASSERT_GIL_HELD();
	nb::gil_scoped_release release;
	// One claim from reading the registry to creating the view, so no other call changes either in between
	auto claim = session.Claim();
	auto statement = DuckDBPySeam::ViewStatement(session.Context(), qualified, sql, temporary, replace, false);
	session.RunClaimed(std::move(statement), DuckDBPyConnection::CheckSignals);
}

static nb::tuple Bind(DuckDBPyConnection &self, const nb::object &sql, bool capture, bool with_type, bool with_writes,
                      const nb::object &values) {
	auto session = OpenSession(self);
	StatementSignature signature;
	engine::ResolvedTables tables;
	StatementType statement_type;
	identifier_map_t<BoundParameterData> converted;
	if (!values.is_none()) {
		converted = TransformPreparedParameters(session.Context(), values);
	}
	DuckDBPyStatement *parsed = nullptr;
	string text;
	if (!nb::try_cast(sql, parsed)) {
		parsed = nullptr;
		text = nb::cast<std::string>(sql);
	}
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		// A statement parsed earlier is copied rather than parsed again
		auto statement = parsed ? parsed->GetStatement() : ParseOneStatement(session, text);
		statement_type = statement->type;
		if (values.is_none()) {
			signature = session.Bind(std::move(statement), tables);
		} else {
			signature = session.Bind(std::move(statement), tables, converted);
		}
	}
	nb::list result;
	nb::list names;
	for (auto &name : signature.names) {
		auto &text = name.GetIdentifierName();
		names.append(nb::str(text.c_str(), text.size()));
	}
	result.append(names);
	nb::list types;
	for (auto &type : signature.types) {
		types.append(DuckDBPyType(type));
	}
	result.append(types);
	auto ordered = signature.parameters;
	std::sort(ordered.begin(), ordered.end(),
	          [](const StatementParameter &a, const StatementParameter &b) { return a.index < b.index; });
	nb::list parameters;
	for (auto &parameter : ordered) {
		auto &text = parameter.identifier.GetIdentifierName();
		parameters.append(nb::make_tuple(nb::str(text.c_str(), text.size()), DuckDBPyType(parameter.type)));
	}
	result.append(parameters);
	result.append(nb::str(ReturnTypeName(signature.properties.return_type)));
	result.append(nb::bool_(signature.metadata_deferred));
	if (capture) {
		nb::list captured;
		for (auto &entry : tables) {
			auto handle = RegisterTable(session.Context(), std::move(entry.second), entry.first);
			captured.append(nb::make_tuple(nb::str(entry.first.c_str(), entry.first.size()), nb::cast(handle)));
		}
		result.append(captured);
	}
	if (with_type) {
		auto type_name = StatementTypeName(statement_type);
		result.append(nb::str(type_name.c_str(), type_name.size()));
	}
	if (with_writes) {
		// The engine's own account, which a function such as nextval adds to as a writing statement does
		result.append(nb::bool_(!signature.properties.IsReadOnly()));
	}
	return nb::tuple(result);
}

static std::shared_ptr<DuckDBPyResult> Submit(DuckDBPyConnection &self, const string &sql, const nb::object &values,
                                              const string &format_name, idx_t batch_size) {
	bool retain = true;
	engine::Format format = engine::Format::Chunks();
	if (format_name == "arrow" || format_name == "stream") {
		format = engine::Format::Arrow(batch_size);
		retain = format_name == "arrow";
	} else if (format_name != "rows") {
		throw InvalidInputException("format must be 'rows', 'arrow' or 'stream', not '%s'", format_name);
	}
	auto session = OpenSession(self);
	identifier_map_t<BoundParameterData> named_values;
	if (!values.is_none()) {
		named_values = TransformPreparedParameters(session.Context(), values);
	}
	shared_ptr<engine::Result> result;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		auto statement = ParseOneStatement(session, sql);
		// One claim from submission to the last row, or to the first unit of a stream, so no other call takes the
		// connection while the statement is half run
		auto claim = session.Claim();
		result = session.SubmitClaimed(std::move(statement), named_values, format);
		if (retain) {
			result->RetainClaimed(DuckDBPyConnection::CheckSignals);
		} else {
			result->SettleClaimed(DuckDBPyConnection::CheckSignals);
		}
	}
	return std::make_shared<DuckDBPyResult>(std::move(result));
}

//! Whether reading the object uses it up: an Arrow stream, or a reader of batches or of IPC messages
static bool ReadOnce(const nb::handle &obj) {
	if (nb::isinstance<nb::capsule>(obj)) {
		return true;
	}
	if (!ModuleIsLoaded<PyarrowCacheItem>()) {
		return false;
	}
	auto &cache = *DuckDBPyConnection::ImportCache();
	return duckdb::PyUtil::IsInstance(obj, cache.pyarrow.RecordBatchReader()) ||
	       duckdb::PyUtil::IsInstance(obj, cache.pyarrow.ipc.MessageReader());
}

std::shared_ptr<SourceHandle> DuckDBPySeam::RegisterSource(DuckDBPyConnection &self, const nb::object &obj,
                                                           const string &subject) {
	auto &context = *OpenConnection(self).context;
	if (!self.source_identity) {
		self.source_identity = std::make_shared<SourceIdentity>();
	}
	auto identity = self.source_identity;
	// An object read once is one source, so a one shot Arrow stream is never adapted twice; any other object is
	// adapted afresh, so a DataFrame that gained a column since is read as it is now
	auto once = ReadOnce(obj);
	if (once) {
		if (auto remembered = identity->Lookup(obj)) {
			return remembered;
		}
	}
	auto kind = subject.empty() ? ArrowSourceKind::NAMELESS : ArrowSourceKind::REGISTERED;
	unique_ptr<TableRef> table;
	{
		// Classifying and adapting the object runs its own Python, which must not close or use this connection
		engine::ConnectionClaim claim(context);
		table = PythonReplacementScan::ReplacementObject(obj, subject.empty() ? "source" : subject, context, kind);
	}
	auto handle = RegisterTable(context, std::move(table), subject);
	if (once) {
		handle->RememberFor(identity, obj);
		identity->Remember(obj, handle);
	}
	return handle;
}

static std::shared_ptr<SourceHandle> RegisterSourceMethod(DuckDBPyConnection &self, const nb::object &obj,
                                                          const std::optional<string> &name) {
	return DuckDBPySeam::RegisterSource(self, obj, name ? *name : string());
}

static void OwnSources(DuckDBPyConnection &self, const string &name,
                       const vector<std::shared_ptr<SourceHandle>> &handles) {
	OpenConnection(self);
	// The replaced handles are moved out first: dropping one can run a finalizer that writes this map
	auto replaced = std::move(self.owned_sources[name]);
	self.owned_sources[name] = handles;
}

static idx_t CatalogVersion(DuckDBPyConnection &self) {
	// Read without the claim and without a lock, since a relation asks before every bind
	if (self.con.ConnectionIsClosed()) {
		throw ConnectionException("Connection has already been closed");
	}
	return self.con.GetSources().CatalogVersion();
}

static string TableKind(DuckDBPyConnection &self, const nb::object &name) {
	auto qualified = NameFromParts(name);
	// The session owns the context, so a close() on another thread cannot free it under this call
	auto session = OpenSession(self);
	auto &context = session.Context();
	DUCKDB_PY_ASSERT_GIL_HELD();
	nb::gil_scoped_release release;
	auto claim = session.Claim();
	string kind = "none";
	// The lookup a table reference binds through, which answers rather than throws when it finds nothing, so asking
	// costs no statement and leaves an open result alone. Binder::BindTableName settles what a lone qualifier is:
	// the name of an attached database when there is one, a schema otherwise.
	context.RunFunctionInTransaction([&]() {
		CatalogEntryRetriever retriever(context);
		EntryLookupInfo lookup(CatalogType::TABLE_ENTRY, Binder::BindTableName(retriever, qualified));
		auto entry = Catalog::GetEntry(context, lookup, OnEntryNotFound::RETURN_NULL);
		if (!entry) {
			return;
		}
		kind = entry->type == CatalogType::VIEW_ENTRY ? "view" : "table";
	});
	return kind;
}

//! Mirrors Binder::BindSchemaOrCatalog: a lone qualifier is a database's name when one has it, a schema otherwise
static string NameScope(DuckDBPyConnection &self, const string &name) {
	// The session owns the context, as in TableKind
	auto session = OpenSession(self);
	auto &context = session.Context();
	DUCKDB_PY_ASSERT_GIL_HELD();
	nb::gil_scoped_release release;
	auto claim = session.Claim();
	string scope = "none";
	context.RunFunctionInTransaction([&]() {
		if (Catalog::GetCatalogEntry(context, Identifier(name))) {
			scope = "database";
			return;
		}
		if (Catalog::GetSchema(context, Identifier::InvalidCatalog(), Identifier(name), OnEntryNotFound::RETURN_NULL)) {
			scope = "schema";
		}
	});
	return scope;
}

static void RequireFormat(const engine::Result &result, bool arrow, const char *method) {
	if (result.GetFormat().IsArrow() != arrow) {
		throw InvalidInputException("%s needs a result submitted with format %s", method,
		                            arrow ? "'arrow' or 'stream'" : "'rows'");
	}
}

//! The type a VALUES list gives a supplied parameter: an uncollated string stays a string literal, which takes the
//! type of the other rows (BoundParameterData::GetDefaultType, ExpressionBinder::GetExpressionReturnType)
static LogicalType ParameterType(const Value &value) {
	auto &type = value.type();
	if (type.id() == LogicalTypeId::VARCHAR && StringType::GetCollation(type).empty()) {
		return LogicalTypeId::STRING_LITERAL;
	}
	return type;
}

//! Rows of Python values held as a collection: each value converted once, as a parameter is, each column of the type
//! the engine's VALUES list gives it, the max type across its rows, and the columns named col0, col1 as VALUES names
//! them
static std::shared_ptr<SourceHandle> RegisterValues(DuckDBPyConnection &self, const nb::sequence &rows) {
	auto session = OpenSession(self);
	auto &context = session.Context();
	vector<vector<Value>> converted;
	for (auto row : rows) {
		vector<Value> values;
		for (auto item : nb::borrow<nb::sequence>(row)) {
			values.push_back(TransformPythonValue(context, item, LogicalType::UNKNOWN, false));
		}
		if (!converted.empty() && values.size() != converted[0].size()) {
			throw InvalidInputException("Mismatch between length of tuples in input, expected %llu but found %llu",
			                            converted[0].size(), values.size());
		}
		converted.push_back(std::move(values));
	}
	if (converted.empty() || converted[0].empty()) {
		throw InvalidInputException("Please provide a non-empty tuple");
	}
	auto width = converted[0].size();
	unique_ptr<TableRef> table;
	{
		// Typing reads the connection's casting settings, so it runs as every seam call does: without the GIL,
		// holding the connection's claim
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		auto claim = session.Claim();
		vector<LogicalType> types(width, LogicalType::SQLNULL);
		for (auto &row : converted) {
			for (idx_t i = 0; i < width; i++) {
				types[i] = LogicalType::MaxLogicalType(context, types[i], ParameterType(row[i]));
			}
		}
		vector<Identifier> names;
		for (idx_t i = 0; i < width; i++) {
			types[i] = LogicalType::NormalizeType(types[i]);
			names.emplace_back("col" + std::to_string(i));
		}
		auto collection = make_uniq<ColumnDataCollection>(context, types);
		DataChunk staged;
		staged.Initialize(Allocator::DefaultAllocator(), types);
		vector<Value> cast(width);
		for (auto &row : converted) {
			for (idx_t i = 0; i < width; i++) {
				cast[i] = CastToColumn(context, row[i], types[i]);
			}
			if (StageRow(staged, cast)) {
				collection->Append(staged);
				staged.Reset();
			}
		}
		if (staged.size() > 0) {
			collection->Append(staged);
		}
		table = make_uniq<ColumnDataRef>(shared_ptr<ColumnDataCollection>(std::move(collection)), std::move(names));
	}
	return RegisterTable(context, std::move(table));
}

static std::shared_ptr<SourceHandle> RegisterCollection(DuckDBPyConnection &self, DuckDBPyResult &result) {
	// The session owns the context across the take, which is the longest call of the seam
	auto session = OpenSession(self);
	auto &context = session.Context();
	// While CONNECT-ed the engine sends every statement to the remote as text, which cannot name rows held here
	if (context.IsConnected()) {
		throw InvalidInputException("materialize() is not available while CONNECT-ed; DISCONNECT first");
	}
	DuckDBPyResult::Use use(result);
	RequireFormat(*use, false, "_register_collection");
	// The rows live in the buffers of the database that produced them. A database that is gone is not this one,
	// which the connection holds.
	if (use->TryGetDatabase().get() != context.db.get()) {
		throw InvalidInputException("The result belongs to another database");
	}
	auto names = use->Names();
	QueryResult::DeduplicateColumns(names);
	unique_ptr<ColumnDataCollection> collection;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		collection = use->TakeChunks(DuckDBPyConnection::CheckSignals);
	}
	auto table = make_uniq<ColumnDataRef>(shared_ptr<ColumnDataCollection>(std::move(collection)), std::move(names));
	return RegisterTable(context, std::move(table));
}

static std::shared_ptr<SourceHandle> RegisterMap(DuckDBPyConnection &self, const nb::callable &function,
                                                 const nb::object &schema, const string &input_sql) {
	auto session = OpenSession(self);
	auto subquery = make_uniq<SubqueryExpression>();
	{
		// The connection's parser reads its client settings, so it runs as every seam call does: without the GIL,
		// holding the connection's claim
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		auto claim = session.Claim();
		subquery->SubqueryMutable() = ParseSelect(session.Context(), input_sql);
	}
	subquery->GetSubqueryTypeMutable() = SubqueryType::SCALAR;
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(subquery));
	auto table = make_uniq<TableFunctionRef>();
	table->function = make_uniq<FunctionExpression>("python_map_function", std::move(children));
	table->bind_info = make_shared_ptr<MapFunctionInfo>(function, schema);
	return RegisterTable(session.Context(), std::move(table));
}

static std::shared_ptr<SourceHandle> RegisterFilelike(DuckDBPyConnection &self, const nb::object &obj) {
	if (nb::isinstance<nb::list>(obj)) {
		throw InvalidInputException("_register_filelike takes one file-like object, not a list");
	}
	OpenConnection(self);
	auto path_like = PathLike::Create(obj, self);
	return std::make_shared<SourceHandle>(path_like.files[0], weak_ptr<engine::SourceRegistry>(),
	                                      std::move(path_like.dependency));
}

static string ResultToBox(DuckDBPyResult &result, const std::optional<idx_t> &max_width,
                          const std::optional<idx_t> &max_rows, const std::optional<idx_t> &max_col_width,
                          const std::optional<string> &null_value, const nb::object &render_mode) {
	BoxRendererConfig config;
	// The relation layer submits at most this many rows, so a result of exactly this size renders as truncated
	config.limit = 10000;
	if (DuckDBPyConnection::IsJupyter()) {
		config.max_width = 10000;
	}
	if (max_width) {
		config.max_width = *max_width;
	}
	if (max_rows) {
		config.max_rows = *max_rows;
	}
	if (max_col_width) {
		config.max_col_width = *max_col_width;
	}
	if (null_value) {
		config.null_value = *null_value;
	}
	if (!render_mode.is_none() && !nb::try_cast(render_mode, config.render_mode)) {
		throw InvalidInputException("'render_mode' accepts either a string, RenderMode or int value");
	}
	DuckDBPyResult::Use use(result);
	RequireFormat(*use, false, "to_box");
	// Owned for the duration: a result only borrows its context, which a close() would free. The GIL is held
	// throughout, as the relation's own box rendering was before the move, because rendering casts values and a cast
	// can run Python; the claim is taken under the GIL, which deadlocks nothing since a claim refuses instead of
	// waiting. The rows are already in the client, so no engine task runs here.
	auto context = use->TryGetContext();
	engine::ConnectionClaim claim(*context);
	ClientBoxRendererContext box_context(*context);
	return use->ToBox(box_context, config);
}

static string LiteralType(const nb::handle &value) {
	return TransformPythonValue(nullptr, value, LogicalType::UNKNOWN, false).type().ToString();
}

// None where a VARCHAR inside a nested type carries the collation, which the type's text cannot write
static nb::object Collation(const nb::object &type_p) {
	std::unique_ptr<DuckDBPyType> type;
	if (!DuckDBPyType::TryConvert(type_p, type)) {
		throw InvalidInputException("_collation takes a DuckDB type, got '%s'",
		                            nb::cast<std::string>(nb::str(type_p.type())));
	}
	auto &logical = type->Type();
	if (logical.id() == LogicalTypeId::VARCHAR) {
		return nb::str(StringType::GetCollation(logical).c_str());
	}
	auto collated = TypeVisitor::Contains(logical, [](const LogicalType &child) {
		return child.id() == LogicalTypeId::VARCHAR && !StringType::GetCollation(child).empty();
	});
	if (collated) {
		return nb::none();
	}
	return nb::str("");
}

static string ValueLiteral(const nb::handle &value) {
	auto converted = TransformPythonValue(nullptr, value, LogicalType::UNKNOWN, false);
	auto &type = converted.type();
	if (type.id() == LogicalTypeId::SQLNULL) {
		return "NULL";
	}
	auto literal = converted.ToSQLString();
	auto type_name = type.ToString();
	// ToSQLString writes 5 for a TINYINT, 1.5 for a DOUBLE and NULL for any typed NULL, which the parser reads as
	// another type, so the type is written out unless the text already ends in that cast
	if (StringUtil::EndsWith(literal, "::" + type_name)) {
		return literal;
	}
	return "CAST(" + literal + " AS " + type_name + ")";
}

static string TemporalLiteral(const nb::handle &value) {
	auto converted = TransformPythonValue(nullptr, value, LogicalType::UNKNOWN, false);
	switch (converted.type().id()) {
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_NS:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::INTERVAL:
		return converted.ToSQLString();
	default:
		throw InvalidInputException("Expected a date, time, datetime, timedelta or numpy temporal value, got %s",
		                            converted.type().ToString());
	}
}

static nb::object PyArrowSchema(const vector<string> &names_p, const nb::list &types_p, const nb::object &connection) {
	if (names_p.size() != types_p.size()) {
		throw InvalidInputException("_arrow_schema needs one type per name, got %llu names and %llu types",
		                            names_p.size(), types_p.size());
	}
	auto names = names_p;
	QueryResult::DeduplicateColumns(names);
	vector<LogicalType> types;
	for (auto type_p : types_p) {
		std::unique_ptr<DuckDBPyType> type;
		if (!DuckDBPyType::TryConvert(nb::borrow<nb::object>(type_p), type)) {
			throw InvalidInputException("_arrow_schema types must be DuckDB types, got '%s'",
			                            nb::cast<std::string>(nb::str(type_p.type())));
		}
		types.push_back(type->Type());
	}
	auto owner = connection.is_none() ? DuckDBPyConnection::DefaultConnection()
	                                  : nb::cast<std::shared_ptr<DuckDBPyConnection>>(connection);
	// The session owns the context, as in TableKind
	auto session = OpenSession(*owner);
	auto &context = session.Context();
	ArrowSchema schema;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		// As Result::BuildArrowSchema: the converter reads the connection's settings and its catalog's types
		auto claim = session.Claim();
		auto properties = context.GetClientProperties();
		context.RunFunctionInTransaction([&]() { ArrowConverter::ToArrowSchema(&schema, types, names, properties); });
	}
	return pyarrow::ToPyArrowSchema(schema);
}

//! The clause the text carries beyond the expressions, null when it carries none
static const char *UnconsumedClause(SelectNode &node) {
	// Every select node has a from_table; one the text did not write is the placeholder for an empty FROM
	if (node.from_table && node.from_table->type != TableReferenceType::EMPTY_FROM) {
		return "a FROM clause";
	}
	if (node.where_clause) {
		return "a WHERE clause";
	}
	if (!node.groups.group_expressions.empty() || node.aggregate_handling != AggregateHandling::STANDARD_HANDLING) {
		return "a GROUP BY clause";
	}
	if (node.having) {
		return "a HAVING clause";
	}
	if (node.qualify) {
		return "a QUALIFY clause";
	}
	if (node.sample) {
		return "a SAMPLE clause";
	}
	if (!node.modifiers.empty()) {
		return "a modifier";
	}
	return nullptr;
}

//! The parts of the column reference a text is, as the parser reads it, or None when it is anything else: another
//! expression, an alias, more than one item, or text that does not parse
static nb::object ColumnReference(const string &text) {
	auto parser = Parser::GetBuiltinParser();
	try {
		parser.ParseQuery("SELECT " + text);
	} catch (std::exception &) {
		return nb::none();
	}
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		return nb::none();
	}
	auto &select = parser.statements[0]->Cast<SelectStatement>();
	if (select.node->type != QueryNodeType::SELECT_NODE || !select.node->cte_map.map.empty()) {
		return nb::none();
	}
	auto &node = select.node->Cast<SelectNode>();
	if (UnconsumedClause(node) || node.select_list.size() != 1) {
		return nb::none();
	}
	auto &expression = *node.select_list[0];
	if (expression.GetExpressionType() != ExpressionType::COLUMN_REF || expression.HasAlias()) {
		return nb::none();
	}
	nb::list parts;
	for (auto &part : expression.Cast<ColumnRefExpression>().ColumnNames()) {
		auto &name = part.GetIdentifierName();
		parts.append(nb::str(name.c_str(), name.size()));
	}
	return std::move(parts);
}

//! The text must be the whole expression list, so a caller can tell SQL text from a column name by whether this
//! throws. Parsed here rather than through Parser::ParseExpressionList, which drops a FROM clause silently.
static nb::list ParseExpressions(const string &text) {
	auto parser = Parser::GetBuiltinParser();
	parser.ParseQuery("SELECT " + text);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw ParserException("\"%s\" is not an expression list: it holds more than one statement", text);
	}
	auto &select = parser.statements[0]->Cast<SelectStatement>();
	if (select.node->type != QueryNodeType::SELECT_NODE) {
		throw ParserException("\"%s\" is not an expression list: it holds a set operation", text);
	}
	auto &node = select.node->Cast<SelectNode>();
	if (auto clause = UnconsumedClause(node)) {
		throw ParserException("\"%s\" is not an expression list: it holds %s", text, clause);
	}
	nb::list rendered;
	for (auto &expression : node.select_list) {
		auto sql = expression->ToString();
		if (expression->HasAlias()) {
			sql += " AS " + SQLIdentifier::ToString(expression->GetAlias());
		}
		rendered.append(nb::str(sql.c_str(), sql.size()));
	}
	return rendered;
}

void DuckDBPySeam::Initialize(nb::module_ &m) {
	nb::class_<SourceHandle>(m, "SourceHandle", nb::is_weak_referenceable())
	    .def_prop_ro("name", &SourceHandle::Name, "The name statements read the source by")
	    .def("__repr__", [](const SourceHandle &handle) { return "<SourceHandle " + handle.Name() + ">"; });

	nb::class_<DuckDBPyResult>(m, "RelationResult", nb::is_weak_referenceable())
	    .def(
	        "fetchall",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        RequireFormat(*use, false, "fetchall");
		        return result.Fetchall();
	        },
	        "Every remaining row as a list of tuples")
	    .def(
	        "fetchmany",
	        [](DuckDBPyResult &result, idx_t size) {
		        DuckDBPyResult::Use use(result);
		        RequireFormat(*use, false, "fetchmany");
		        return result.Fetchmany(size);
	        },
	        "The next rows of the result as a list of tuples, at most size of them", nb::arg("size") = 1)
	    .def(
	        "fetchnumpy",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        RequireFormat(*use, false, "fetchnumpy");
		        return result.FetchNumpy();
	        },
	        "Every remaining row as a dict of NumPy arrays")
	    .def(
	        "df",
	        [](DuckDBPyResult &result, bool date_as_object) {
		        DuckDBPyResult::Use use(result);
		        RequireFormat(*use, false, "df");
		        return result.FetchDF(date_as_object);
	        },
	        "Every remaining row as a pandas DataFrame", nb::kw_only(), nb::arg("date_as_object") = false)
	    .def(
	        "torch",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        RequireFormat(*use, false, "torch");
		        return result.FetchPyTorch();
	        },
	        "Every remaining row as a dict of PyTorch tensors")
	    .def(
	        "tf",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        RequireFormat(*use, false, "tf");
		        return result.FetchTF();
	        },
	        "Every remaining row as a dict of TensorFlow tensors")
	    .def(
	        "arrow_table",
	        [](DuckDBPyResult &result, bool to_polars) {
		        DuckDBPyResult::Use use(result);
		        RequireFormat(*use, true, "arrow_table");
		        return result.FetchArrowTable(DuckDBPyResult::DEFAULT_ARROW_BATCH_SIZE, to_polars);
	        },
	        "Every remaining row as a pyarrow Table; to_polars makes duplicate column names unique", nb::kw_only(),
	        nb::arg("to_polars") = false)
	    .def(
	        "arrow_reader",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        RequireFormat(*use, true, "arrow_reader");
		        return result.FetchRecordBatchReader();
	        },
	        "A pyarrow RecordBatchReader over the remaining rows")
	    .def(
	        "arrow_capsule",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        RequireFormat(*use, true, "arrow_capsule");
		        return result.FetchArrowCapsule();
	        },
	        "An arrow_array_stream PyCapsule over the remaining rows")
	    .def_prop_ro(
	        "description",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        return DuckDBPyResult::GetDescription(result.GetNames(), result.GetTypes());
	        },
	        "The DB-API description of the result's columns")
	    .def_prop_ro(
	        "return_type",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        // What the statement that ran returned, which a deferred group's bind can only guess
		        return string(ReturnTypeName(use->Properties().return_type));
	        },
	        "What the statement that ran returns: 'query_result', 'changed_rows' or 'nothing'")
	    .def_prop_ro(
	        "rowcount",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        return result.Rowcount();
	        },
	        "The rows a write changed, -1 for any other statement")
	    .def("to_box", &ResultToBox, "The rows rendered as the box text show() prints",
	         nb::arg("max_width").none() = nb::none(), nb::arg("max_rows").none() = nb::none(), nb::kw_only(),
	         nb::arg("max_col_width").none() = nb::none(), nb::arg("null_value").none() = nb::none(),
	         nb::arg("render_mode").none() = nb::none())
	    .def(
	        "drain",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        result.Complete();
	        },
	        "Run the statement to its end without reading its rows, then close")
	    .def(
	        "close",
	        [](DuckDBPyResult &result) {
		        DuckDBPyResult::Use use(result);
		        result.Close();
	        },
	        "End the statement and drop its rows");

	m.def("_literal_type", &LiteralType, "The SQL type a Python value binds as", nb::arg("value").none());
	m.def("_value_literal", &ValueLiteral, "A Python value as the SQL literal it binds as", nb::arg("value").none());
	m.def("_collation", &Collation, "The collation of a VARCHAR type, empty for none", nb::arg("type"));
	m.def("_temporal_literal", &TemporalLiteral, "A temporal Python value as a SQL literal", nb::arg("value").none());
	m.def("_arrow_schema", &PyArrowSchema,
	      "The pyarrow schema a result of these columns has, under the Arrow settings of the connection, or of the "
	      "default connection",
	      nb::arg("names"), nb::arg("types"), nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def("_parse_expressions", &ParseExpressions, "Each expression of a SQL expression list, rendered back",
	      nb::arg("text"));
	m.def("_column_reference", &ColumnReference, "The parts of the column reference a text is, or None",
	      nb::arg("text"));
}

//! Checks a verb's SQL text as the connection parses it, alone and for the shape the verb takes: "expressions" for a
//! select list or one condition, "order" for an ORDER BY list, "groups" for a GROUP BY list. The text is spliced as
//! written afterwards, so this only refuses: a parameter marker, which no verb's text can be given a value for, and
//! anything past the shape. The number of items the text holds.
static idx_t CheckText(DuckDBPyConnection &self, const string &text, const string &shape) {
	string prefix;
	if (shape == "expressions") {
		prefix = "SELECT ";
	} else if (shape == "order") {
		prefix = "SELECT 1 ORDER BY ";
	} else if (shape == "groups") {
		prefix = "SELECT 42 GROUP BY ";
	} else {
		throw InvalidInputException("_check_text takes the shape expressions, order or groups, not '%s'", shape);
	}
	auto session = OpenSession(self);
	vector<unique_ptr<SQLStatement>> parsed;
	{
		DUCKDB_PY_ASSERT_GIL_HELD();
		nb::gil_scoped_release release;
		// Ended by a line break, so a line comment in the text ends where the text does
		engine::StatementGroup statements(session, prefix + text + "\n", false);
		while (auto statement = statements.Next()) {
			parsed.push_back(std::move(statement));
		}
	}
	const char *refusal = shape == "order"    ? "Expected a single ORDER clause"
	                      : shape == "groups" ? "groups takes only a GROUP BY list"
	                                          : "Cannot have any modifiers in the expression list";
	if (parsed.size() != 1 || parsed[0]->type != StatementType::SELECT_STATEMENT) {
		throw ParserException(refusal);
	}
	if (!parsed[0]->named_param_map.empty()) {
		throw BinderException("Unexpected prepared parameter. This type of statement can't be prepared!");
	}
	auto &select = parsed[0]->Cast<SelectStatement>();
	if (select.node->type != QueryNodeType::SELECT_NODE || !select.node->cte_map.map.empty()) {
		throw ParserException(refusal);
	}
	auto &node = select.node->Cast<SelectNode>();
	auto plain = (!node.from_table || node.from_table->type == TableReferenceType::EMPTY_FROM) && !node.where_clause &&
	             !node.having && !node.qualify && !node.sample;
	if (shape == "expressions") {
		if (UnconsumedClause(node)) {
			throw ParserException(refusal);
		}
		return node.select_list.size();
	}
	if (shape == "groups") {
		if (!plain || !node.modifiers.empty()) {
			throw ParserException(refusal);
		}
		return node.groups.group_expressions.size();
	}
	auto grouped =
	    !node.groups.group_expressions.empty() || node.aggregate_handling != AggregateHandling::STANDARD_HANDLING;
	if (!plain || grouped || node.modifiers.size() != 1 ||
	    node.modifiers[0]->type != ResultModifierType::ORDER_MODIFIER) {
		throw ParserException(refusal);
	}
	return node.modifiers[0]->Cast<OrderModifier>().orders.size();
}

void DuckDBPySeam::InitializeConnectionMethods(nb::class_<DuckDBPyConnection> &m) {
	m.def(
	    "_bind", &Bind,
	    "Bind one statement without running it: (names, types, parameters, return_type, deferred), then the "
	    "sources captured from Python variables when capture is set, then the parsed statement type when with_type is, "
	    "then whether the statement changes a database when with_writes is; with values, its parameters are typed "
	    "from them",
	    nb::arg("sql"), nb::kw_only(), nb::arg("capture") = false, nb::arg("with_type") = false,
	    nb::arg("with_writes") = false, nb::arg("values").none() = nb::none());
	m.def("_statement_count", &StatementCount, "How many statements a text holds, as the parser splits it",
	      nb::arg("sql"));
	m.def("_check_text", &CheckText, "Check a verb's SQL text for its shape; the number of items it holds",
	      nb::arg("text"), nb::arg("shape"));
	m.def(
	    "_prepare_script",
	    [](DuckDBPyConnection &self, const nb::object &sql, const nb::object &values, bool with_statement) {
		    OpenConnection(self);
		    return self.PrepareScript(sql, values, with_statement);
	    },
	    "Run every statement but the last: (the kept rows result or None, the last statement's text, its type), then "
	    "the parsed last statement when with_statement is set",
	    nb::arg("sql"), nb::arg("values").none() = nb::none(), nb::kw_only(), nb::arg("with_statement") = false);
	m.def("_submit", &Submit, "Run one statement with parameter values to a result in the given format", nb::arg("sql"),
	      nb::arg("values").none() = nb::none(), nb::arg("format") = "rows",
	      nb::arg("batch_size") = DuckDBPyResult::DEFAULT_ARROW_BATCH_SIZE);
	m.def("_result_as_arrow", &DuckDBPyConnection::ResultAsArrow,
	      "The rows the connection's open result still holds, read back as a pyarrow Table, or as a "
	      "RecordBatchReader when reader is set; None once the rows are used up",
	      nb::arg("batch_size"), nb::kw_only(), nb::arg("reader"));
	m.def("_register_source", &RegisterSourceMethod, "Register a Python object as a source statements can read",
	      nb::arg("obj"), nb::arg("name").none() = nb::none());
	m.def("_register_collection", &RegisterCollection, "Register the rows of a retained result as a source",
	      nb::arg("result"));
	m.def("_register_values", &RegisterValues, "Register rows of Python values, converted once, as a source",
	      nb::arg("rows"));
	m.def("_register_map", &RegisterMap, "Register a map of a Python function over the rows of a query",
	      nb::arg("function"), nb::arg("schema").none(), nb::arg("input_sql"));
	m.def("_register_filelike", &RegisterFilelike, "Put a file-like object on the connection's object store",
	      nb::arg("obj"));
	m.def("_table_kind", &TableKind,
	      "Whether the name is a 'table', a 'view' or 'none' of the two, read from the catalog without running a "
	      "statement; a string is one identifier and a tuple is the parts of a qualified name",
	      nb::arg("name").none());
	m.def("_name_scope", &NameScope,
	      "Whether one identifier names a 'database', a 'schema' of the search path, or 'none' of the two, which says "
	      "whether a dotted name is qualified",
	      nb::arg("name"));
	m.def("_create_view", &CreateView,
	      "Create a view of the SELECT in sql, the Python objects it reads held by the view itself, so the view "
	      "reads them for as long as it exists and its transaction decides when they go",
	      nb::arg("name"), nb::arg("sql"), nb::kw_only(), nb::arg("temporary") = false, nb::arg("replace") = false);
	m.def("_catalog_version", &CatalogVersion,
	      "How many statements that can change what a name binds to have run on this connection's database");
	m.def("_own_sources", &OwnSources,
	      "Keep the handles alive under the name until unregister(name) or the next _own_sources of the name",
	      nb::arg("name"), nb::arg("handles"));
}

} // namespace duckdb

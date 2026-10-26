#pragma once

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb_python/nb/casters.hpp"

#include <exception>
#include <functional>

namespace duckdb {

struct DuckDBPyConnection;

//! A column a predicate refers to: the name the source itself uses for it, and the type the scan declared.
struct PredicateColumn {
	string name;
	LogicalType type;
};

//! A predicate, or a part of one, that cannot be handed to a source exactly; the engine keeps applying all of it.
struct PredicateRefused : std::exception {};

//! The declared columns of a scan as its source names them. A source can only apply a predicate to a column it can
//! name, so a column whose source name is repeated, or whose struct fields the importer renamed (an empty or
//! repeated field name inside a struct), resolves to a refusal. An empty source name is kept: the one column of a
//! nameless array is named by it.
class PredicateColumns {
public:
	PredicateColumns() = default;
	//! Columns whose source names the scan declared unchanged, struct fields included.
	PredicateColumns(const vector<string> &names, const vector<LogicalType> &types);
	//! Columns of an Arrow source: `schema` is the source's top level struct schema and `types` the DuckDB types the
	//! scan declared for its children, in the same order.
	PredicateColumns(const ArrowSchema &schema, const vector<LogicalType> &types);

	//! @throws PredicateRefused For a column the source cannot name exactly.
	PredicateColumn Resolve(idx_t declared_index) const;

private:
	vector<PredicateColumn> columns;
	vector<bool> nameable;
};

//! Turns a predicate the optimizer offers to a scan into duckdb._sources.nodes objects: comparisons of a column with
//! a constant (the column put on the left), BETWEEN as two comparisons under AND, IN and NOT IN over constants,
//! IS NULL, IS NOT NULL, AND, OR and NOT. A column is a column reference, or a chain of struct field accesses on one.
//! A constant is carried only when it has exactly the column's type and a Python value holds it without loss.
//! `resolve` maps a column reference's binding column index to the declared column and may throw PredicateRefused.
//! Call with the GIL held.
//! @throws PredicateRefused For any other shape, and for a constant without an exact Python value of the column's type.
nb::object TranslatePredicate(const Expression &predicate,
                              const std::function<PredicateColumn(idx_t column_binding_index)> &resolve);

//! Plans `SELECT <select> FROM <temporary table> WHERE <predicate>` on the connection through the optimizer,
//! capturing what the optimizer offers a scan with pushdown_complex_filter, and returns a list holding a tuple
//! (engine text of the offered expression, node or None when refused) per offered predicate. The table has either
//! `columns`, a dict of name to type text, or the columns the engine's Arrow scan declares for `schema`, an object
//! with `__arrow_c_schema__`, whose names then also name the columns in predicates.
nb::list TranslatePredicateForTesting(DuckDBPyConnection &connection, const string &predicate,
                                      const nb::object &columns, const nb::object &schema, const string &select);

} // namespace duckdb

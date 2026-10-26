"""The one place where the relation layer and the expression builder ask the engine something.

`bind` asks for the schema of one statement without running it, `submit` runs one statement with its parameter values
and returns a result in the format a terminal needs. The registry functions put a Python object or a retained result
under a name that statements on the same connection can read, for as long as a handle to it lives. The literal and
parser functions say how a Python value binds and reads as SQL, and how SQL expression text parses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import _duckdb

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from duckdb.sqltypes import DuckDBPyType

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "Result",
    "Signature",
    "SourceHandle",
    "arrow_schema",
    "bind",
    "catalog_version",
    "check_text",
    "collation",
    "column_reference",
    "create_view",
    "literal_type",
    "name_scope",
    "parse_expressions",
    "parse_type",
    "prepare_script_parsed",
    "register_collection",
    "register_filelike",
    "register_map",
    "register_source",
    "register_values",
    "result_arrow",
    "statement_count",
    "submit",
    "table_kind",
    "temporal_literal",
    "value_literal",
]

DEFAULT_BATCH_SIZE = 131072

Values = list[Any] | dict[str, Any] | None

Result = _duckdb.RelationResult
SourceHandle = _duckdb.SourceHandle


class Signature(NamedTuple):
    """What the binder says about one statement."""

    names: list[str]
    types: list[DuckDBPyType]
    parameters: list[tuple[str, DuckDBPyType]]
    return_type: str
    deferred: bool
    captured: Sequence[tuple[str, SourceHandle]] = ()
    #: Whether running the statement changes a database, as a write or a sequence's nextval does.
    writes: bool = False


def bind(
    con: _duckdb.DuckDBPyConnection,
    sql: str | _duckdb.Statement,
    capture: bool = False,  # noqa: FBT001
    values: Values = None,
) -> Signature:
    """The output names and types of one statement, its parameters and what it returns, without running it.

    With `capture`, the Python variables the statement reads are registered and listed under the names it used. With
    `values`, its parameters are typed from them, so the names and types are those of the statement that runs.
    """
    if capture:
        names, types, parameters, return_type, deferred, captured, writes = con._bind(
            sql, capture=True, with_writes=True, values=values
        )
        return Signature(names, types, parameters, return_type, deferred, captured, writes)
    names, types, parameters, return_type, deferred, writes = con._bind(sql, with_writes=True, values=values)
    return Signature(names, types, parameters, return_type, deferred, (), writes)


def submit(
    con: _duckdb.DuckDBPyConnection,
    sql: str,
    values: Values,
    format: str = "rows",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Result:
    """Run one statement with its values, to the format a terminal needs."""
    return con._submit(sql, values, format, batch_size)  # type: ignore[arg-type]


def result_arrow(con: _duckdb.DuckDBPyConnection, batch_size: int, *, reader: bool) -> Any:  # noqa: ANN401
    """The rows the connection's open result still holds, as Arrow; None once a fetch has used them up.

    The rows are taken out of the result and read back through a statement over them, so they come out in the
    Arrow encoding this connection's settings produce. With `reader` the statement streams into a
    `RecordBatchReader`, which holds the connection until it is read out, dropped or closed.
    """
    return con._result_as_arrow(batch_size, reader=reader)


def prepare_script_parsed(
    con: _duckdb.DuckDBPyConnection, query: object, values: Values
) -> tuple[Result | None, str | None, str | None, _duckdb.Statement | None]:
    """Run every statement of a script but the last; the rows an earlier one kept, the last one's text and type.

    The parsed last statement comes too, which bind takes without parsing its text again.
    """
    if isinstance(query, _duckdb.Statement):
        return None, query.query, query.type.name, query
    if not isinstance(query, str):
        msg = "Invalid Input Error: Please provide either a DuckDBPyStatement or a string representing the query"
        raise _duckdb.InvalidInputException(msg)
    kept, last_text, last_type, statement = con._prepare_script(query, values, with_statement=True)
    return kept, last_text, last_type.upper() if last_type is not None else None, statement


def arrow_schema(con: _duckdb.DuckDBPyConnection, names: list[str], types: Sequence[DuckDBPyType | str]) -> Any:  # noqa: ANN401
    """The Arrow schema of columns with these names and types, without running a statement."""
    return _duckdb._arrow_schema(names, list(types), connection=con)


def parse_type(con: _duckdb.DuckDBPyConnection, text: str) -> DuckDBPyType:
    """A type name parsed on this connection, so its own types resolve."""
    return con.sqltype(text)


def catalog_version(con: _duckdb.DuckDBPyConnection) -> int:
    """How many statements that may change what a name binds to have run on this connection."""
    return con._catalog_version()


def table_kind(con: _duckdb.DuckDBPyConnection, name: str | tuple[str, ...]) -> str:
    """Whether a name reads a "table", a "view" or "none" of the two, without running a statement.

    A string is one identifier however many dots it holds, and a tuple is the parts of a qualified name, so the two
    readings of a dotted name are asked separately.
    """
    return con._table_kind(name)


def name_scope(con: _duckdb.DuckDBPyConnection, name: str) -> str:
    """Whether one identifier names a "database", a "schema" of the search path, or "none" of the two."""
    return con._name_scope(name)


def register_values(con: _duckdb.DuckDBPyConnection, rows: list[tuple[object, ...]]) -> SourceHandle:
    """Rows of Python values as a source, each value converted once as a parameter is, typed as a VALUES list."""
    return con._register_values(rows)


def register_source(con: _duckdb.DuckDBPyConnection, obj: object, name: str | None = None) -> SourceHandle:
    """Put a Python object under a name statements on this connection can read."""
    return con._register_source(obj, name)


def register_collection(con: _duckdb.DuckDBPyConnection, result: Result) -> SourceHandle:
    """Keep the rows of a retained result under a name statements on this connection can read."""
    return con._register_collection(result)


def register_map(
    con: _duckdb.DuckDBPyConnection,
    function: Callable[..., Any],
    schema: object,
    input_sql: str,
) -> SourceHandle:
    """Register the map of a Python function over the rows of a query as a source."""
    return con._register_map(function, schema, input_sql)  # type: ignore[arg-type]


def register_filelike(con: _duckdb.DuckDBPyConnection, obj: object) -> SourceHandle:
    """Put a file-like object on the connection's object store; the handle's name is its path there."""
    return con._register_filelike(obj)


def create_view(con: _duckdb.DuckDBPyConnection, name: str, sql: str, *, replace: bool) -> None:
    """Create the persistent view `name` over the SELECT in `sql`, the view owning what the registry names it reads."""
    con._create_view(name, sql, replace=replace)


def check_text(con: _duckdb.DuckDBPyConnection, text: str, shape: str) -> int:
    """Refuse a verb's SQL text that is not the shape it takes, or holds a parameter marker; the number of its items.

    The shape is "expressions" for a select list or a condition, "order" for an ORDER BY list, "groups" for a GROUP BY
    list. The connection's own parser reads the text.
    """
    return con._check_text(text, shape)


def statement_count(con: _duckdb.DuckDBPyConnection, sql: str) -> int:
    """How many statements a text holds, as the connection's parser splits it."""
    return con._statement_count(sql)


def literal_type(value: object) -> str:
    """The SQL type `value` binds as when passed as a parameter; raises the conversion error where it cannot bind."""
    return _duckdb._literal_type(value)


def temporal_literal(value: object) -> str:
    """`value` written as a SQL literal that means what it means bound as a parameter."""
    return _duckdb._temporal_literal(value)


def collation(type_: DuckDBPyType) -> str | None:
    """The collation a VARCHAR type carries, empty for none; None where a VARCHAR nested inside the type carries one."""
    return _duckdb._collation(type_)


def value_literal(value: object) -> str:
    """A value written as the SQL literal of its object converted to its type, as a parameter is."""
    return _duckdb._value_literal(value)


def column_reference(text: str) -> list[str] | None:
    """The parts of the column reference `text` is, as the engine's parser reads it, or None when it is not one."""
    return _duckdb._column_reference(text)


def parse_expressions(text: str) -> list[str]:
    """The expressions of a comma separated list, each rendered back by the engine's parser with its alias kept."""
    return _duckdb._parse_expressions(text)

"""Relations: queries built a step at a time as SQL text, bound for their schema and run only by a terminal.

A relation is a value: one step (a verb and its arguments), the relations it reads, the connection, and the column
names and types the binder reported when the step was built. Nothing runs until a terminal such as `fetchall` renders
the whole query, one common table expression per step, and submits it with its parameter values.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import html
import inspect
import operator
import os
import re
import secrets
import struct
import sys
import warnings
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, cast

import _duckdb

from duckdb import _seam
from duckdb._expression import PendingCase, _is_value, constant, fold_name, plain_value, quote, text_tokens
from duckdb._sql_keywords import KEYWORDS

if TYPE_CHECKING:
    import builtins
    from collections.abc import Callable, Iterator, Sequence
    from types import FrameType

    from duckdb.sqltypes import DuckDBPyType

__all__ = ["DuckDBPyRelation", "GroupedRelation"]

_BOX_LIMIT = 10000
_HTML_ROWS = 10
_SAMPLE_METHODS = ("bernoulli", "reservoir", "system")
_NUMERIC_TYPE_IDS = frozenset(
    {
        "tinyint",
        "smallint",
        "integer",
        "bigint",
        "hugeint",
        "utinyint",
        "usmallint",
        "uinteger",
        "ubigint",
        "uhugeint",
        "float",
        "double",
        "decimal",
    }
)
_REMOVED_METHODS = {
    "fetchone": "rel.limit(1).fetchall()",
    "fetchmany": "rel.limit(n).fetchall()",
    "fetch_df_chunk": "rel.to_arrow_reader() to read in batches, or rel.limit(n).df()",
    "execute": "rel.materialize() to run it once and keep its rows, or a method such as fetchall()",
}
_BARE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOWER_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")
_SHOW_PRAGMA = re.compile(r"""^\s*pragma\s+(?:"show"|show)\s*\(\s*'((?:[^']|'')*)'\s*\)\s*;?\s*$""", re.IGNORECASE)
_ERROR_NAMES = {
    "InvalidInputException": "Invalid Input",
    "BinderException": "Binder",
    "ParserException": "Parser",
    "NotImplementedException": "Not implemented",
    "CatalogException": "Catalog",
    "InvalidTypeException": "Invalid type",
    "IOException": "IO",
}


def _error(kind: type[Exception], message: str) -> Exception:
    """An engine exception class with the engine's prefix on its message, as the C++ relation raised them."""
    prefix = _ERROR_NAMES.get(kind.__name__)
    return kind(f"{prefix} Error: {message}" if prefix else message)


def _name_text(name: str) -> str:
    """A name written so SQL folds its case when it is a plain identifier, quoted otherwise."""
    return name if _BARE_NAME.match(name) and fold_name(name) not in KEYWORDS else quote(name)


@dataclasses.dataclass(frozen=True)
class _WithClause:
    recursive: bool
    listed: str
    rest: str
    names: frozenset[str]


def _with_clause(text: str) -> _WithClause | None:
    """The WITH list a query starts with, read from the tokenizer, so a name can be added to that list."""
    tokens = [(o, k) for o, k in text_tokens(text) if k != _duckdb.token_type.comment]
    words = []
    for index, (offset, kind) in enumerate(tokens):
        end = tokens[index + 1][0] if index + 1 < len(tokens) else len(text)
        words.append((offset, kind, text[offset:end].strip()))
    if not words or fold_name(words[0][2]) != "with":
        return None
    position = 1
    recursive = position < len(words) and fold_name(words[position][2]) == "recursive"
    position += int(recursive)
    if position >= len(words):
        return None
    start = words[position][0]
    names: set[str] = set()

    def skip_group(at: int) -> int:
        depth = 0
        while at < len(words):
            word = words[at][2]
            if words[at][1] == _duckdb.token_type.operator:
                depth += word.count("(") - word.count(")")
                at += 1
                if depth <= 0:
                    return at
                continue
            at += 1
        return at

    while position < len(words):
        name = words[position][2]
        names.add(fold_name(name[1:-1].replace('""', '"')) if name.startswith('"') else fold_name(name))
        position += 1
        if position < len(words) and words[position][2].startswith("("):
            position = skip_group(position)
        while position < len(words) and fold_name(words[position][2]) in ("as", "not", "materialized"):
            position += 1
        if position >= len(words) or not words[position][2].startswith("("):
            return None
        position = skip_group(position)
        if position < len(words) and words[position][2] == ",":
            position += 1
            continue
        break
    if position >= len(words):
        return None
    end = words[position][0]
    return _WithClause(recursive, text[start:end].rstrip(), text[end:], frozenset(names))


def _query_statement(step: QueryOver, source: str, ctes: list[tuple[str, str]]) -> str:
    """A `query(name, sql)` statement, with the name in the query's own WITH list where it can be.

    There a data modifying common table expression stays at the top level and duplicate names stay as the query
    gives them.
    """
    own = (_name_text(step.name), f"SELECT * FROM {source}")
    clause = _with_clause(step.text)
    if clause is not None and fold_name(step.name) not in clause.names:
        listed = ",\n".join(f"{name} AS NOT MATERIALIZED (\n{body}\n)" for name, body in [*ctes, own])
        head = "WITH RECURSIVE" if clause.recursive else "WITH"
        return f"{head} {listed},\n{clause.listed}\n{clause.rest}"
    tokens = [(o, k) for o, k in text_tokens(step.text) if k != _duckdb.token_type.comment]
    first = fold_name(step.text[tokens[0][0] :].split(None, 1)[0]) if tokens else ""
    if clause is None and first in ("select", "from", "values"):
        return _assemble([*ctes, own], step.text)
    return _assemble([*ctes, own], _wrap(step.text, "_q"))


def _optional_quote(part: str) -> str:
    """A name part as the engine prints a qualified name: bare when it is a plain lower case identifier."""
    return part if _LOWER_NAME.match(part) and part not in KEYWORDS else quote(part)


def _api() -> Any:  # noqa: ANN401
    """The package, imported when first needed, since it imports this module while it initialises."""
    import duckdb

    return duckdb


def _is_expression(value: object) -> bool:
    return isinstance(value, _api().Expression)


def _expression_text(value: object) -> str:
    """SQL text of an expression, or of a Python value as the constant it converts to."""
    if _is_expression(value):
        return _encodable(str(value))
    if isinstance(value, PendingCase):
        # A verb refuses what it cannot take with the engine's class for a bad argument; the TypeError is for the
        # builder in an expression position.
        raise _error(_duckdb.InvalidInputException, value.unfinished)
    try:
        return _constant_text(value, "ConstantExpression takes a value")
    except (_duckdb.NotImplementedException, TypeError) as error:
        raise _error(_duckdb.InvalidInputException, "Please provide arguments of type Expression!") from error


def _value_text(value: object) -> str:
    """SQL text of one value of the list `values()` takes, where a value no constant holds is refused by its type."""
    if _is_expression(value) or isinstance(value, PendingCase):
        return _expression_text(value)
    return _constant_text(value, "values() takes a value or an expression")


def _constant_text(value: object, refusal: str) -> str:
    if isinstance(value, str):
        _encodable(value)
    return _encodable(str(constant(value, refusal)))


def _encodable(text: str) -> str:
    """Refuse text that cannot be UTF-8, as the conversion of a Python string refuses it."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        escaped = text[error.start].encode("unicode_escape").decode()
        message = (
            f"Python Conversion Failure: cannot encode str as UTF-8, surrogate '{escaped}' at position {error.start}"
        )
        raise _error(_duckdb.InvalidInputException, message) from None
    return text


def _spliced(text: str) -> str:
    """A SQL fragment to splice into a statement, ending its line, so a line comment in it ends there too."""
    return _encodable(text) + "\n"


def _items_of(con: _duckdb.DuckDBPyConnection, args: tuple[object, ...]) -> str:
    """A select list from the arguments of a verb: one string is expression text, several are column names."""
    if len(args) == 1 and type(args[0]) is str:
        return _select_list(con, args[0])
    return ", ".join(_select_item(con, arg) for arg in args)


def _select_list(con: _duckdb.DuckDBPyConnection, text: str) -> str:
    """A select list given as one string, checked by the connection's parser and written as given."""
    _seam.check_text(con, _encodable(text), "expressions")
    return _spliced(text)


def _one_name(value: object) -> str:
    """The single column name an index or a membership test takes; anything else is refused by type."""
    if isinstance(value, bytes):
        return value.decode()
    if not isinstance(value, str):
        message = (
            f"a relation is indexed by one column name, not by {type(value).__name__}; "
            f"use project() for several columns"
        )
        raise TypeError(message)
    return value


def _column_name(value: str | bytes) -> str:
    """A name given where a column is expected: a reference as written when it reads as one, else one whole name.

    So `project("t.x", "i")` names a column of a side, as today, and `project("my col", "x")` names a column whose
    name holds a space; a string that reads as neither is quoted, and the binder refuses it by name.
    """
    text = _encodable(value.decode() if isinstance(value, bytes) else value)
    if any(kind == _duckdb.token_type.comment or text.startswith(";", offset) for offset, kind in text_tokens(text)):
        # The parser reads past a comment or a semicolon, so such text is one name, quoted whole.
        return quote(text)
    parts = _seam.column_reference(text)
    return ".".join(quote(part) for part in parts) if parts is not None else quote(text)


def _select_item(con: _duckdb.DuckDBPyConnection, value: object) -> str:
    """One select list item: an expression with its alias, or one whole column name, or the star."""
    del con
    if isinstance(value, str) and value == "*":
        return "*"
    if isinstance(value, (str, bytes)):
        return _column_name(value)
    if not _is_expression(value):
        return _expression_text(value)
    as_select = getattr(value, "as_select", None)
    if callable(as_select):
        return _encodable(str(as_select()))
    text = _encodable(str(value))
    name = value.get_name()  # type: ignore[attr-defined]
    return text if name == text else f"{text} AS {quote(name)}"


def _order_item(con: _duckdb.DuckDBPyConnection, value: object) -> str:
    """One ORDER BY key: an expression with its direction, or SQL order text, checked and written as given."""
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        _seam.check_text(con, _encodable(value), "order")
        return _spliced(value)
    as_order = getattr(value, "as_order", None)
    if callable(as_order):
        return _encodable(str(as_order()))
    return _expression_text(value)


def _plain_item(con: _duckdb.DuckDBPyConnection, value: object) -> str:
    """An expression's SQL without an alias, for a place that names it itself."""
    del con
    if isinstance(value, (str, bytes)):
        return _column_name(value)
    return _expression_text(value)


def _condition_text(con: _duckdb.DuckDBPyConnection, value: object, what: str | None) -> str:
    """A condition, or the aggregates of a pivot where `what` is None, as SQL; text is checked and written as given."""
    if not isinstance(value, str):
        return _expression_text(value)
    count = _seam.check_text(con, _encodable(value), "expressions")
    if what is not None and count != 1:
        raise _error(_duckdb.ParserException, f"Expected a single expression as {what}")
    return _spliced(value)


def _strip_semicolons(text: str) -> str:
    text = text.strip()
    while text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def _wrap(text: str, alias: str) -> str:
    """Raw query text as a subquery, which admits DESCRIBE, SHOW and SUMMARIZE bodies and a trailing comment."""
    return f"SELECT * FROM (\n{text}\n) AS {quote(alias)}"


def _assemble(ctes: list[tuple[str, str]], body: str) -> str:
    """One statement from named bodies and a last one; a last body with its own WITH becomes a subquery."""
    if not ctes:
        return body
    if fold_name(body.lstrip()[:4]) == "with":
        body = _wrap(body, "_q")
    listed = ",\n".join(f"{name} AS NOT MATERIALIZED (\n{text}\n)" for name, text in ctes)
    return f"WITH {listed}\n{body}"


def _stub(names: list[str], types: list[DuckDBPyType]) -> str | None:
    """An empty SELECT of typed NULLs standing in for these columns, or None where one has a type it cannot write.

    A type's text leaves out its collation, which a column of its own carries as COLLATE; a collation on a type inside
    a list or struct has no such spelling.
    """
    columns: list[str] = []
    for name, type_ in zip(names, types, strict=True):
        collation = _seam.collation(type_)
        if collation is None:
            return None
        collated = f" COLLATE {quote(collation)}" if collation else ""
        columns.append(f"NULL::{_stand_in_type(type_)}{collated} AS {quote(name)}")
    return f"SELECT {', '.join(columns)} WHERE FALSE"


def _stand_in_type(type_: DuckDBPyType) -> str:
    """A type's SQL with every field name quoted, so the names keep their case whatever the connection folds to."""
    kind = type_.id
    children = type_.children if kind in ("struct", "union", "list", "array", "map") else []
    if kind in ("struct", "union"):
        # A union's first child is its tag.
        fields = children[1:] if kind == "union" else children
        written = ", ".join(f"{quote(name)} {_stand_in_type(cast('DuckDBPyType', child))}" for name, child in fields)
        return f"{kind.upper()}({written})"
    if kind == "list":
        return f"{_stand_in_type(cast('DuckDBPyType', children[0][1]))}[]"
    if kind == "array":
        return f"{_stand_in_type(cast('DuckDBPyType', children[0][1]))}[{children[1][1]}]"
    if kind == "map":
        key, value = (cast("DuckDBPyType", child) for _, child in children)
        return f"MAP({_stand_in_type(key)}, {_stand_in_type(value)})"
    return str(type_)


def _split_qualified(text: str) -> list[str]:
    """The parts of a dotted name, where a part in double quotes may hold dots."""
    parts: list[str] = []
    current: list[str] = []
    quoted = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            if quoted and index + 1 < len(text) and text[index + 1] == '"':
                current.append('"')
                index += 2
                continue
            quoted = not quoted
        elif char == "." and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    if quoted:
        raise _error(_duckdb.ParserException, f"Unterminated quoted identifier in qualified name! (input: {text})")
    parts.append("".join(current))
    return parts


def _split_list(text: str) -> list[str]:
    """The items of a comma separated list, split at the commas outside any parenthesis, with no trailing empty one."""
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    depth = 0
    found = text_tokens(text)
    for position, (offset, kind) in enumerate(found):
        if kind != _duckdb.token_type.operator:
            continue
        end = found[position + 1][0] if position + 1 < len(found) else len(text)
        for index in range(offset, end):
            char = text[index]
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                pieces.append(text[start:index])
                start = index + 1
    pieces.append(text[start:])
    if pieces[-1] == "":
        pieces.pop()
    return pieces


def _group_list(con: _duckdb.DuckDBPyConnection, text: str) -> str:
    """A groups argument, which is a GROUP BY list and nothing more, checked and written as given."""
    if not text.strip():
        return ""
    _seam.check_text(con, _encodable(text), "groups")
    return _spliced(text)


def _float_text(value: float) -> str:
    """A quantile as the old shortcuts wrote it: a single precision float with six decimals."""
    return f"{struct.unpack('f', struct.pack('f', value))[0]:.6f}"


def _caller_level() -> int:
    """The stacklevel of the first frame outside this module, so a warning names the line the user wrote."""
    level = 1
    frame: FrameType | None = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        level += 1
        frame = frame.f_back
    return level


def _warn_on_literal_names(value: object, columns: list[str]) -> None:
    """Warn when an expression compares with a string that is also a column name, since strings are values now."""
    if not _is_expression(value) or not columns:
        return
    try:
        from duckdb import _expression
    except ImportError:
        return
    literals = getattr(_expression, "string_literals", None)
    if literals is None:
        return
    folded = {fold_name(name) for name in columns}
    for literal in literals(value):
        if isinstance(literal, str) and fold_name(literal) in folded:
            warnings.warn(
                f"{literal!r} is used as a string value, and it is also a column name; use col({literal!r}) for "
                f"the column, or lit({literal!r}) to keep the value and silence this warning",
                UserWarning,
                stacklevel=_caller_level(),
            )


@dataclasses.dataclass(frozen=True)
class _Marker:
    start: int
    end: int
    name: str
    positional: bool


def _markers(text: str) -> tuple[_Marker, ...]:
    """The parameter markers of a query, from the engine's tokenizer, so string constants and comments are skipped.

    A bare `?` is one past the highest positional number seen so far, as the engine numbers it.
    """
    tokens = text_tokens(text)
    found: list[_Marker] = []
    highest = 0
    for index, (offset, kind) in enumerate(tokens):
        if kind != _duckdb.token_type.operator:
            continue
        if text.startswith("?", offset):
            end = offset + 1
            while end < len(text) and text[end].isdigit():
                end += 1
            # `?N` is positional N, as `$N` is, and a bare `?` is one past the highest number so far.
            number = int(text[offset + 1 : end]) if end > offset + 1 else highest + 1
            highest = max(highest, number)
            found.append(_Marker(offset, end, str(number), positional=True))
            continue
        if not text.startswith("$", offset) or index + 1 >= len(tokens):
            continue
        following, following_kind = tokens[index + 1]
        if following != offset + 1:
            continue
        end = following
        while end < len(text) and (text[end].isalnum() or text[end] == "_"):
            end += 1
        name = text[following:end]
        if not name:
            continue
        if following_kind == _duckdb.token_type.numeric_const and name.isdigit():
            highest = max(highest, int(name))
            found.append(_Marker(offset, end, str(int(name)), positional=True))
        elif following_kind in (_duckdb.token_type.identifier, _duckdb.token_type.keyword):
            found.append(_Marker(offset, end, name, positional=False))
    return tuple(found)


def _parameter_values(markers: tuple[_Marker, ...], params: object) -> list[Any] | dict[str, Any]:
    """The values for a query's markers, checked against them the way the engine checks them at execution."""
    # A marker's name is an identifier, which the engine folds, so `$P` reads the value given as `p`.
    expected = {fold_name(m.name): m.name for m in markers}
    if isinstance(params, Mapping):
        given = {str(k): v for k, v in params.items()}
        if not markers:
            message = f"Parameter/argument count mismatch for prepared statement. Expected 0, got {len(given)}"
            raise _error(_duckdb.BinderException, message)
        folded = {fold_name(name) for name in given}
        missing = sorted(name for key, name in expected.items() if key not in folded)
        if missing:
            message = f"Values were not provided for the following parameters: {', '.join(missing)}"
            raise _error(_duckdb.InvalidInputException, message)
        excess = sorted(name for name in given if fold_name(name) not in expected)
        if excess:
            message = f"Parameter argument/count mismatch, identifiers of the excess parameters: {', '.join(excess)}"
            raise _error(_duckdb.InvalidInputException, message)
        return given
    values: list[Any] = list(params)  # type: ignore[call-overload]
    if not markers:
        message = f"Parameter/argument count mismatch for prepared statement. Expected 0, got {len(values)}"
        raise _error(_duckdb.BinderException, message)
    if any(not m.positional for m in markers):
        message = "Mixing named and positional parameters is not supported yet"
        raise _error(_duckdb.NotImplementedException, message)
    count = max(int(m.name) for m in markers)
    if len(values) < count:
        missing = [str(i) for i in range(len(values) + 1, count + 1)]
        message = f"Values were not provided for the following parameters: {', '.join(missing)}"
        raise _error(_duckdb.InvalidInputException, message)
    if len(values) > count:
        excess = [str(i) for i in range(count + 1, len(values) + 1)]
        message = f"Parameter argument/count mismatch, identifiers of the excess parameters: {', '.join(excess)}"
        raise _error(_duckdb.InvalidInputException, message)
    return values


def _refuse_one_name_twice(params: object) -> None:
    """Refuse a mapping that gives two values to one name, which execution binds as one in its own order."""
    if not isinstance(params, Mapping):
        return
    spellings: dict[str, str] = {}
    for key in params:
        name = str(key)
        first = spellings.setdefault(fold_name(name), name)
        if first != name:
            message = f"The parameters {first!r} and {name!r} are one name to SQL; give a value for one of them"
            raise _error(_duckdb.InvalidInputException, message)


def _value_of(marker: _Marker, values: list[Any] | dict[str, Any]) -> object:
    if isinstance(values, dict):
        if marker.name in values:
            return values[marker.name]
        wanted = fold_name(marker.name)
        return next(value for name, value in values.items() if fold_name(name) == wanted)
    return values[int(marker.name) - 1]


def _bound_literal(value: object) -> str:
    """A value written into SQL text the way execution binds it: a string as a string literal, None as NULL."""
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + _encodable(value).replace("'", "''") + "'"
    return _seam.value_literal(value)


def _literal_text(text: str, markers: tuple[_Marker, ...], values: list[Any] | dict[str, Any]) -> str:
    """The query with each marker replaced by its value written as a literal."""
    pieces: list[str] = []
    last = 0
    for marker in markers:
        pieces.append(text[last : marker.start])
        pieces.append(_bound_literal(_value_of(marker, values)))
        last = marker.end
    pieces.append(text[last:])
    return "".join(pieces)


def _renumbered(text: str, markers: tuple[_Marker, ...], offset: int) -> str:
    pieces: list[str] = []
    last = 0
    for marker in markers:
        pieces.append(text[last : marker.start])
        pieces.append(f"${int(marker.name) + offset}" if marker.positional else text[marker.start : marker.end])
        last = marker.end
    pieces.append(text[last:])
    return "".join(pieces)


def _option_literal(value: object) -> str:
    """A reader or COPY option value as SQL: strings quoted, lists as lists, dicts as structs.

    Written from the value a subclass holds, read through its base type as an expression's literal is.
    """
    value = plain_value(value)
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return "'" + _encodable(value).replace("'", "''") + "'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_option_literal(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_option_literal(str(k))}: {_option_literal(v)}" for k, v in value.items()) + "}"
    return _expression_text(value)


def _row_count(method: str, parameter: str, value: object) -> int:
    """A row count given to a verb, refused by name where it is not an integer, as the old client refused it.

    Any integer counts, a NumPy one included, as the old client's conversion took any object with `__index__`; a bool
    is a flag, not a count.
    """
    if not isinstance(value, bool):
        try:
            return operator.index(value)  # type: ignore[arg-type]
        except TypeError:
            pass
    message = f"{method}(): incompatible function arguments: {parameter} takes an int, not a {_type_name(value)}"
    raise TypeError(message)


@dataclasses.dataclass(frozen=True)
class Step:
    """One verb and its arguments, whose SQL is a function of the record and the FROM items of its inputs."""

    kind: ClassVar[str] = ""

    def render(self, items: tuple[str, ...]) -> str:
        raise NotImplementedError

    def type_name(self) -> str:
        """The relation type string of today's relation for this verb."""
        return self.kind


@dataclasses.dataclass(frozen=True)
class Source(Step):
    """A step whose SQL is a FROM item, so the SELECT reading it is the first step's own."""

    def from_item(self) -> str:
        raise NotImplementedError

    def implicit_alias(self) -> str | None:
        """The name the FROM item already carries, which needs no `AS` of its own."""
        return None


@dataclasses.dataclass(frozen=True)
class Sql(Step):
    """A query as written, which no clause is appended to, so it is read as a common table expression."""

    kind: ClassVar[str] = "QUERY_RELATION"
    text: str
    markers: tuple[_Marker, ...] = ()

    def render(self, items: tuple[str, ...]) -> str:
        return self.text


@dataclasses.dataclass(frozen=True)
class Table(Source):
    """A catalog entry or a file by name."""

    kind: ClassVar[str] = "TABLE_RELATION"
    parts: tuple[str, ...]
    #: Whether the name reads a base table; any other catalog entry or file reads as a query.
    base: bool = True

    def type_name(self) -> str:
        return self.kind if self.base else "QUERY_RELATION"

    def name(self) -> str:
        return ".".join(quote(p) for p in self.parts)

    def from_item(self) -> str:
        return self.name()

    def implicit_alias(self) -> str | None:
        return self.parts[-1]


@dataclasses.dataclass(frozen=True)
class View(Table):
    kind: ClassVar[str] = "VIEW_RELATION"

    def type_name(self) -> str:
        return self.kind


@dataclasses.dataclass(frozen=True)
class Values(Source):
    kind: ClassVar[str] = "VALUE_LIST_RELATION"
    rows: tuple[tuple[str, ...], ...]

    def from_item(self) -> str:
        return "(VALUES " + ", ".join("(" + ", ".join(row) + ")" for row in self.rows) + ")"


@dataclasses.dataclass(frozen=True)
class TableFunction(Source):
    kind: ClassVar[str] = "TABLE_FUNCTION_RELATION"
    name: str
    arguments: tuple[str, ...]

    def from_item(self) -> str:
        return f"{quote(self.name)}({', '.join(self.arguments)})"

    def implicit_alias(self) -> str | None:
        return self.name


@dataclasses.dataclass(frozen=True)
class Reader(Source):
    """A file reader with its options as named arguments."""

    relation_type: str
    function: str
    files: str
    options: tuple[tuple[str, str], ...]
    #: The select list a pinned reader needs, which fills the select list slot of the SELECT reading it.
    projection: str = "*"

    def type_name(self) -> str:
        return self.relation_type

    def from_item(self) -> str:
        arguments = [self.files, *(f"{name} := {value}" for name, value in self.options)]
        return f"{self.function}({', '.join(arguments)})"


@dataclasses.dataclass(frozen=True)
class Registered(Source):
    """A name in the connection's registry: a Python object, a map function, or rows a statement returned."""

    kind: ClassVar[str] = "VIEW_RELATION"
    name: str

    def from_item(self) -> str:
        return quote(self.name)

    def implicit_alias(self) -> str | None:
        return self.name


@dataclasses.dataclass(frozen=True)
class Held(Registered):
    kind: ClassVar[str] = "MATERIALIZED_RELATION"


@dataclasses.dataclass(frozen=True)
class ValueRows(Held):
    kind: ClassVar[str] = "VALUE_LIST_RELATION"


@dataclasses.dataclass(frozen=True)
class Mapped(Registered):
    kind: ClassVar[str] = "TABLE_FUNCTION_RELATION"


@dataclasses.dataclass(frozen=True)
class Project(Step):
    """A select list, written as a list of items or as a star with modifiers."""

    kind: ClassVar[str] = "PROJECTION_RELATION"
    items: str


@dataclasses.dataclass(frozen=True)
class Star(Project):
    """A projection written as a star with modifiers: with_columns, drop, rename, unnest."""


@dataclasses.dataclass(frozen=True)
class Filter(Step):
    kind: ClassVar[str] = "FILTER_RELATION"
    condition: str


@dataclasses.dataclass(frozen=True)
class Order(Step):
    kind: ClassVar[str] = "ORDER_RELATION"
    keys: str


@dataclasses.dataclass(frozen=True)
class Limit(Step):
    kind: ClassVar[str] = "LIMIT_RELATION"
    count: int | None
    offset: int = 0


@dataclasses.dataclass(frozen=True)
class Distinct(Step):
    """`distinct()` fills the DISTINCT slot; `distinct(on=...)` is a SELECT of its own, since it picks rows."""

    kind: ClassVar[str] = "DISTINCT_RELATION"
    on: str | None = None

    def render(self, items: tuple[str, ...]) -> str:
        return f"SELECT DISTINCT ON ({self.on}) * FROM {items[0]}"


@dataclasses.dataclass(frozen=True)
class Aggregate(Step):
    kind: ClassVar[str] = "AGGREGATE_RELATION"
    items: str
    groups: str | None = None


@dataclasses.dataclass(frozen=True)
class Alias(Step):
    """A name for the result, which a consumer reads it under."""

    kind: ClassVar[str] = "SUBQUERY_RELATION"
    alias: str


@dataclasses.dataclass(frozen=True)
class Join(Step):
    """A join of two inputs, each side under its relation's alias, as the engine rendered joins."""

    kind: ClassVar[str] = "JOIN_RELATION"
    keyword: str
    on: str | None = None
    using: tuple[str, ...] = ()
    #: The select list a `suffix=` join needs, which fills the select list slot of its SELECT.
    projection: str = "*"
    #: Whether each side enters the FROM under its own alias, which a condition written as a function needs.
    sides_closed: bool = False

    def source(self, items: tuple[str, ...]) -> str:
        clause = ""
        if self.on is not None:
            clause = f" ON {self.on}"
        elif self.using:
            clause = " USING (" + ", ".join(quote(c) for c in self.using) + ")"
        return f"{items[0]} {self.keyword} JOIN {items[1]}{clause}"


@dataclasses.dataclass(frozen=True)
class Cross(Join):
    kind: ClassVar[str] = "CROSS_PRODUCT_RELATION"


@dataclasses.dataclass(frozen=True)
class SetOp(Step):
    kind: ClassVar[str] = "SET_OPERATION_RELATION"
    keyword: str

    def render(self, items: tuple[str, ...]) -> str:
        return f"SELECT * FROM {items[0]} {self.keyword} SELECT * FROM {items[1]}"


@dataclasses.dataclass(frozen=True)
class Unpivot(Step):
    kind: ClassVar[str] = "PROJECTION_RELATION"
    columns: tuple[str, ...]
    name: str
    value: str

    def render(self, items: tuple[str, ...]) -> str:
        folded = ", ".join(quote(c) for c in self.columns)
        into = f"INTO NAME {quote(self.name)} VALUE {quote(self.value)}"
        return f"SELECT * FROM (UNPIVOT (SELECT * FROM {items[0]}) ON {folded} {into})"


@dataclasses.dataclass(frozen=True)
class Pivot(Step):
    kind: ClassVar[str] = "AGGREGATE_RELATION"
    column: str
    values: tuple[str, ...]
    aggregate: str

    def render(self, items: tuple[str, ...]) -> str:
        listed = ", ".join(self.values)
        return f"SELECT * FROM (PIVOT (SELECT * FROM {items[0]}) ON {self.column} IN ({listed}) USING {self.aggregate})"


@dataclasses.dataclass(frozen=True)
class Sample(Step):
    kind: ClassVar[str] = "SAMPLE_RELATION"
    size: str
    arguments: str

    def render(self, items: tuple[str, ...]) -> str:
        return f"SELECT * FROM {items[0]} USING SAMPLE {self.size} ({self.arguments})"


@dataclasses.dataclass(frozen=True)
class QueryOver(Step):
    """A user query reading the relation under a name: `rel.query(name, sql)`."""

    kind: ClassVar[str] = "QUERY_RELATION"
    name: str
    text: str

    def render(self, items: tuple[str, ...]) -> str:
        own = (_name_text(self.name), f"SELECT * FROM {items[0]}")
        return _assemble([own], _wrap(self.text, "_q"))


#: The clause slots of one SELECT, in the order the engine evaluates them; FROM is filled by the source or join.
#: There is no HAVING slot: a filter after an aggregate reads the aggregate's output, so it opens a new SELECT.
_WHERE, _GROUP, _SELECT, _DISTINCT, _ORDER, _OFFSET, _LIMIT = range(1, 8)


@dataclasses.dataclass(frozen=True)
class _Select:
    """The SELECT a chain of steps is filling, over the FROM items of `source`.

    A step appends when its slot is empty and no later slot is filled, with WHERE accumulating; any other step
    closes this SELECT as a common table expression and opens a new one over it. Only which slots are filled and
    the kind of the step decide, never the text of a step.
    """

    #: The node whose FROM items this SELECT reads: a source, a join, or the closed SELECT it was opened over. None is
    #: the node holding this SELECT itself, a source or a join, which a reference to it would make a reference cycle,
    #: and a relation in a cycle is freed by the cyclic collector at an arbitrary moment instead of when its last
    #: reference goes.
    source: DuckDBPyRelation | None = dataclasses.field(repr=False)
    #: Whether this SELECT is `source`'s own SELECT with more clauses, reading the same FROM, rather than a new one
    #: opened over `source` read closed as a common table expression.
    continues: bool = False
    where: tuple[str, ...] = ()
    group: Aggregate | None = None
    select: str | None = None
    distinct: bool = False
    order: str | None = None
    offset: int | None = None
    limit: int | None = None

    def filled(self) -> set[int]:
        """The slots that hold a clause; an aggregate fills the select list with its own items."""
        slots: set[int] = set()
        if self.where:
            slots.add(_WHERE)
        if self.group is not None:
            slots |= {_GROUP, _SELECT}
        if self.select is not None:
            slots.add(_SELECT)
        if self.distinct:
            slots.add(_DISTINCT)
        if self.order is not None:
            slots.add(_ORDER)
        if self.offset is not None:
            slots.add(_OFFSET)
        if self.limit is not None:
            slots.add(_LIMIT)
        return slots

    def bare(self) -> bool:
        """Whether this SELECT is still the plain read of its FROM items, so its source can be read in place."""
        return not self.filled()

    def admit(self, step: Step) -> _Select | None:
        """This SELECT with the step's clause in it, or None where the step needs a SELECT of its own."""
        filled = self.filled()

        def later(slot: int) -> bool:
            return any(other > slot for other in filled)

        if isinstance(step, Filter):
            # After an aggregate a name in the condition means the aggregate's output, which only a SELECT over it
            # binds; appended as HAVING it would bind the input's group key of that name instead.
            return None if later(_WHERE) else dataclasses.replace(self, where=(*self.where, step.condition))
        if isinstance(step, Aggregate):
            return None if _GROUP in filled or later(_GROUP) else dataclasses.replace(self, group=step)
        if isinstance(step, Project):
            return None if _SELECT in filled or later(_SELECT) else dataclasses.replace(self, select=step.items)
        if isinstance(step, Distinct):
            if step.on is not None or self.distinct or later(_DISTINCT):
                return None
            return dataclasses.replace(self, distinct=True)
        if isinstance(step, Order):
            # After a select list an ORDER BY expression binds the FROM's columns before the list's names, where a
            # SELECT over the list binds its output; only the step's kind may decide, so it closes.
            if self.order is not None or later(_ORDER) or _SELECT in filled:
                return None
            return dataclasses.replace(self, order=step.keys)
        if isinstance(step, Limit):
            first = _OFFSET if step.offset > 0 else _LIMIT
            if any(other >= first for other in filled):
                return None
            offset = step.offset if step.offset > 0 else self.offset
            limit = step.count if step.count is not None else self.limit
            return dataclasses.replace(self, offset=offset, limit=limit)
        return None

    def render(self, source: str) -> str:
        """The SELECT, with its clauses in the order the engine evaluates them."""
        head = "SELECT DISTINCT" if self.distinct else "SELECT"
        # The aggregate list is the select list, and a SELECT with neither reads its FROM items as they are.
        items = self.group.items if self.group is not None else (self.select or "*")
        text = f"{head} {items} FROM {source}"
        if self.where:
            text += " WHERE " + " AND ".join(f"({condition})" for condition in self.where)
        if self.group is not None:
            text += f" GROUP BY {self.group.groups or 'ALL'}"
        if self.order is not None:
            text += f" ORDER BY {self.order}"
        if self.limit is not None and self.limit >= 0:
            text += f" LIMIT {self.limit}"
        if self.offset:
            text += f" OFFSET {self.offset}"
        return text


def _open_for(step: Step, inputs: tuple[DuckDBPyRelation, ...]) -> _Select | None:
    """The SELECT the new node is filling.

    A source or a join opens its own SELECT over itself; a clause step appends to its input's, or opens a new SELECT
    over the input read as a common table expression; any other step is a SELECT of its own.
    """
    if isinstance(step, (Source, Join)):
        projection = step.projection if isinstance(step, (Reader, Join)) and step.projection != "*" else None
        return _Select(None, select=projection)
    if len(inputs) != 1:
        return None
    inp = inputs[0]
    if inp._open is not None:
        appended = inp._open.admit(step)
        if appended is not None and _captures_held_name(step, inp):
            # Every capture is a registration of its own, so a name the SELECT already holds starts the step's own.
            appended = None
        if appended is not None:
            # The input's own SELECT, which it holds over itself, is now held by this node over the input.
            if appended.source is None:
                appended = dataclasses.replace(appended, source=inp, continues=True)
            return appended
    return _Select(inp).admit(step)


def _captures_held_name(step: Step, inp: DuckDBPyRelation) -> bool:
    """Whether the step's text captures a name the input's SELECT holds for another registration.

    The step's clause alone is bound with capture over typed empty stand-ins of the SELECT's FROM items, without the
    captures the SELECT holds, which would answer for those names in the step's own bind.
    """
    held = _held_keys(_rendered_here(inp))
    if not held:
        return False
    alone = _Select(inp).admit(step)
    if alone is None:
        return False
    stubs: list[tuple[str, str]] = []
    source = _probe_from(inp, stubs)
    if source is None:
        return False
    try:
        signature = _seam.bind(inp._con, _assemble(stubs, alone.render(source)), capture=True)
    except _duckdb.Error:
        # The bind of the step itself reports what is wrong with its text.
        return False
    return _clashes(signature.captured, held)


def _held_keys(nodes: Iterable[DuckDBPyRelation]) -> dict[str, str]:
    """The registration each Python variable the nodes' texts captured reads, by its name as SQL folds it."""
    return {fold_name(name): key for name, key in _all_captured(nodes)}


def _clashes(captured: Iterable[tuple[str, _seam.SourceHandle]], held: dict[str, str]) -> bool:
    """Whether a capture registers anew a name, as SQL folds it, that `held` holds."""
    return any(fold_name(name) in held and handle.name != held[fold_name(name)] for name, handle in captured)


def _probe_from(node: DuckDBPyRelation, stubs: list[tuple[str, str]]) -> str | None:
    """The FROM of the node's open SELECT over stand-ins, or None where a column has a type a stand-in cannot write."""
    owner = _owner(node)
    if owner is None:
        return _probe_item(_source_of(node), stubs, inline=False)
    if isinstance(owner._step, Join):
        return _probe_join(owner, stubs)
    return _probe_item(owner, stubs, inline=False)


def _probe_item(node: DuckDBPyRelation, stubs: list[tuple[str, str]], *, inline: bool) -> str | None:
    """A FROM item over typed empty stand-ins under the aliases the real item has, holding no text of the user's."""
    if inline:
        inner = _probe_join(node, stubs)
        return None if inner is None else f"({inner})"
    text = _stub(*node._schema)
    if text is None:
        return None
    name = quote(f"__duckdb_probe{len(stubs)}")
    stubs.append((name, text))
    return f"{name} AS {quote(node.alias)}"


def _probe_join(join: DuckDBPyRelation, stubs: list[tuple[str, str]]) -> str | None:
    """A join over stand-ins with the names in scope the real one has, and a condition that always holds."""
    step = join._step
    assert isinstance(step, Join)
    items = []
    for side, inline in _sides_of(join):
        item = _probe_item(side, stubs, inline=inline)
        if item is None:
            return None
        items.append(item)
    return (step if step.on is None else dataclasses.replace(step, on="TRUE")).source(tuple(items))


def _close_clashing_sides(
    con: _duckdb.DuckDBPyConnection, step: Join, sides: tuple[tuple[DuckDBPyRelation, bool], ...]
) -> tuple[tuple[DuckDBPyRelation, bool], ...]:
    """The sides, with an inlined join entering closed where a name it captured is held for another registration.

    The join's condition is bound with capture over stand-ins of the sides, so it lists the names it captures now; each
    inlined side is then checked against those and against the sides before it.
    """
    held = [_held_keys(_rendered_here(side)) if inline else {} for side, inline in sides]
    if not any(held):
        return sides
    claims: dict[str, str] = {}
    stubs: list[tuple[str, str]] = []
    probed = [_probe_item(side, stubs, inline=inline) for side, inline in sides] if step.on is not None else []
    items = tuple(item for item in probed if item is not None)
    if probed and len(items) == len(probed):
        try:
            signature = _seam.bind(con, _assemble(stubs, "SELECT * FROM " + step.source(items)), capture=True)
        except _duckdb.Error:
            # The bind of the join itself reports what is wrong with its condition.
            return sides
        claims = {fold_name(name): handle.name for name, handle in signature.captured}
    result = []
    for (side, inline), keys in zip(sides, held, strict=True):
        if any(name in claims and claims[name] != key for name, key in keys.items()):
            inline = False
        else:
            for name, key in keys.items():
                claims.setdefault(name, key)
        result.append((side, inline))
    return tuple(result)


def _source_of(node: DuckDBPyRelation) -> DuckDBPyRelation:
    """The node the node's open SELECT reads, the node itself for a source or a join."""
    assert node._open is not None
    return node._open.source if node._open.source is not None else node


def _owner(node: DuckDBPyRelation) -> DuckDBPyRelation | None:
    """The source or join whose own FROM the node's open SELECT reads, or None where it reads one closed."""
    assert node._open is not None
    if node._open.source is None:
        return node
    return node._open.source if node._open.continues else None


def _in_place(node: DuckDBPyRelation) -> bool:
    """Whether the node's FROM item is its own text: a source whose SELECT holds nothing else."""
    return isinstance(node._step, Source) and node._open is not None and node._open.bare()


def _bare_join(node: DuckDBPyRelation) -> bool:
    """Whether a join enters another join's FROM in place, keeping its input aliases visible there."""
    return isinstance(node._step, Join) and node._open is not None and node._open.bare()


def _item(
    node: DuckDBPyRelation,
    names: dict[int, str],
    *,
    in_join: bool,
    alias: str | None = None,
    in_place: bool = True,
) -> str:
    """The FROM item reading one node, under its own alias or the name a consumer gave it."""
    if isinstance(node._step, Alias):
        # Naming the result means using it from outside, so the named SELECT is read rather than inlined.
        named = alias if alias is not None else node._step.alias
        return _item(node._inputs[0], names, in_join=False, alias=named, in_place=in_place)
    spelled = quote(alias if alias is not None else node.alias)
    if in_place and _in_place(node):
        assert isinstance(node._step, Source)
        return f"{node._step.from_item()} AS {spelled}"
    if in_join and _bare_join(node):
        return f"({_join_source(node, names)})"
    return f"{names[id(node)]} AS {spelled}"


def _item_nodes(node: DuckDBPyRelation, *, in_join: bool, in_place: bool = True) -> list[DuckDBPyRelation]:
    """The nodes a FROM item reads by step name: none for text in place, an inlined join's own items, or the node."""
    if isinstance(node._step, Alias):
        return _item_nodes(node._inputs[0], in_join=False, in_place=in_place)
    if in_place and _in_place(node):
        return []
    if in_join and _bare_join(node):
        found: list[DuckDBPyRelation] = []
        for side, inline in _sides_of(node):
            found.extend(_item_nodes(side, in_join=inline))
        return found
    return [node]


def _from_aliases(node: DuckDBPyRelation, *, in_join: bool) -> builtins.list[str]:
    """The aliases a node puts in a join's FROM: its own, or those of the inputs an inlined join brings with it."""
    if in_join and _bare_join(node):
        return [alias for side, inline in _sides_of(node) for alias in _from_aliases(side, in_join=inline)]
    return [node.alias]


def _join_items(
    left: DuckDBPyRelation, right: DuckDBPyRelation, *, sides_closed: bool
) -> builtins.list[tuple[DuckDBPyRelation, bool]]:
    """Each side of a join with whether it is read in place, so no two items of one FROM share an alias.

    A bare join brings its own inputs into the FROM, which only works while the names they come under are free there;
    where one is taken, that join enters as a closed SELECT under its own alias instead. A condition written as
    a function names one side by its alias, which an inlined join does not have, so there both sides stay closed.
    """
    inline = [not sides_closed and _bare_join(side) for side in (left, right)]

    def aliases() -> builtins.list[str]:
        return [
            fold_name(alias)
            for side, inlined in zip((left, right), inline, strict=True)
            for alias in _from_aliases(side, in_join=inlined)
        ]

    for position in (1, 0):
        taken = aliases()
        if len(set(taken)) == len(taken):
            break
        if inline[position]:
            inline[position] = False
    return [(left, inline[0]), (right, inline[1])]


def _sides_of(join: DuckDBPyRelation) -> builtins.list[tuple[DuckDBPyRelation, bool]]:
    """Each side of a join with whether it is read in place, as decided once when the join was made."""
    assert join._sides is not None
    return list(join._sides)


def _join_source(join: DuckDBPyRelation, names: dict[int, str]) -> str:
    step = join._step
    assert isinstance(step, Join)
    items = tuple(_item(side, names, in_join=inline) for side, inline in _sides_of(join))
    return step.source(items)


def _from_text(node: DuckDBPyRelation, names: dict[int, str]) -> str:
    """The FROM of a node's SELECT: the join or source text it continues, or the item a closed source is read as."""
    owner = _owner(node)
    if owner is None:
        return _item(_source_of(node), names, in_join=False)
    if isinstance(owner._step, Join):
        return _join_source(owner, names)
    # The source itself, which needs no name of its own where the FROM item already carries one.
    assert isinstance(owner._step, Source)
    text = owner._step.from_item()
    return text if owner.alias == owner._step.implicit_alias() else f"{text} AS {quote(owner.alias)}"


def _from_nodes(node: DuckDBPyRelation) -> list[DuckDBPyRelation]:
    """The nodes the FROM of a node's SELECT reads by step name."""
    owner = _owner(node)
    if owner is None:
        return _item_nodes(_source_of(node), in_join=False)
    if isinstance(owner._step, Join):
        found: list[DuckDBPyRelation] = []
        for side, inline in _sides_of(owner):
            found.extend(_item_nodes(side, in_join=inline))
        return found
    return []


def _references(node: DuckDBPyRelation) -> list[DuckDBPyRelation]:
    """The nodes a node's SQL reads by a step name; a source and an inlined join are read in place."""
    if node._open is not None:
        return _from_nodes(node)
    found: list[DuckDBPyRelation] = []
    for inp in node._inputs:
        found.extend(_item_nodes(inp, in_join=False, in_place=not isinstance(node._step, SetOp)))
    return found


def _body(node: DuckDBPyRelation, names: dict[int, str], *, top: bool, text: str | None = None) -> str:
    """A node's SELECT, given the step names of the nodes it reads; `text` replaces a query's own text."""
    if node._open is not None:
        return node._open.render(_from_text(node, names))
    step = node._step
    if isinstance(step, Sql):
        written = step.text if text is None else text
        return written if top else _wrap(written, node.alias)
    if isinstance(step, Alias):
        return f"SELECT * FROM {_item(node, names, in_join=False)}"
    in_place = not isinstance(step, SetOp)
    return step.render(tuple(_item(inp, names, in_join=False, in_place=in_place) for inp in node._inputs))


def _order(top: DuckDBPyRelation) -> list[DuckDBPyRelation]:
    """Every node the statement reads by a step name, inputs before the nodes reading them, each once."""
    seen: set[int] = set()
    order: list[DuckDBPyRelation] = []
    stack: list[tuple[DuckDBPyRelation, bool]] = [(top, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        stack.extend((parent, False) for parent in reversed(_references(node)))
    return order


def _tree(top: DuckDBPyRelation) -> list[DuckDBPyRelation]:
    """Every node a relation is built from, inputs first, each once."""
    seen: set[int] = set()
    order: list[DuckDBPyRelation] = []
    stack: list[tuple[DuckDBPyRelation, bool]] = [(top, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        stack.extend((parent, False) for parent in reversed(node._inputs))
    return order


def _captured_ctes(captured: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(_name_text(name), f"SELECT * FROM {quote(key)}") for name, key in captured]


def _all_captured(nodes: Iterable[DuckDBPyRelation]) -> list[tuple[str, str]]:
    """The Python variables a SELECT reads, each once.

    A step or a join that captures a name already read, as SQL folds it, under another registration starts a SELECT of
    its own, so one name here is one registration.
    """
    by_name: dict[str, tuple[str, str]] = {}
    for node in nodes:
        for name, key in node._captured:
            if by_name.setdefault(fold_name(name), (name, key))[1] != key:
                message = f'One SELECT reads two registrations under the name "{name}"'
                raise _error(_duckdb.InternalException, message)
    return list(by_name.values())


def _render(
    top: DuckDBPyRelation, *, literals: bool = False, wrap_text: bool = False
) -> tuple[str, list[Any] | dict[str, Any] | None]:
    """The statement a terminal runs and its parameter values; with `literals` the values are written into the text.

    With `wrap_text` a query's own text is read as a subquery, so the statement can follow CREATE TABLE AS and the like.
    """
    order = _order(top)
    names = {id(node): quote(f"__duckdb_s{i}") for i, node in enumerate(order)}
    holders = [n for n in order if n._params is not None]
    positional = [n for n in holders if not isinstance(n._params, dict)]
    named = [n for n in holders if isinstance(n._params, dict)]
    values: list[Any] | dict[str, Any] | None = None
    offsets: dict[int, int] = {}
    if named:
        merged: dict[str, Any] = {}
        for node in named:
            assert isinstance(node._params, dict)
            merged.update(node._params)
        values = merged
    elif positional:
        flat: list[Any] = []
        for node in positional:
            assert isinstance(node._params, list)
            if len(positional) > 1:
                offsets[id(node)] = len(flat)
            flat.extend(node._params)
        values = flat
    ctes: list[tuple[str, str]] = []
    last = ""
    for node in order:
        text = None
        if isinstance(node._step, Sql) and node._params is not None:
            if literals:
                # Written in, the values would name the columns they make; the names stay those of what runs.
                written = _literal_text(node._step.text, node._step.markers, node._params)
                columns = ", ".join(quote(name) for name in node._schema[0])
                text = f"SELECT * FROM (\n{written}\n) AS {quote(node.alias)}({columns})"
            elif id(node) in offsets:
                text = _renumbered(node._step.text, node._step.markers, offsets[id(node)])
        is_top = node is top
        body = _body(node, names, top=is_top, text=text)
        # A Python variable a node's text read is in scope in that node's own SELECT only, never in the rest of the
        # statement, where the same name can read another object or a catalog table.
        own = _captured_ctes(_all_captured(_rendered_here(node)))
        if isinstance(node._step, QueryOver) and is_top and not wrap_text:
            source = _item(top._inputs[0], names, in_join=False)
            return _query_statement(node._step, source, [*ctes, *own]), None if literals else values
        if own:
            if is_top and isinstance(node._step, Sql):
                # The text as written can be a statement a WITH list cannot lead, DESCRIBE for one.
                body = _wrap(body, node.alias)
            body = _assemble(own, body)
        if is_top:
            last = body
        else:
            ctes.append((names[id(node)], body))
    if isinstance(top._step, Sql) and (ctes or wrap_text) and not top._captured:
        last = _wrap(last, top.alias)
    return _assemble(ctes, last), None if literals else values


def _binds(node: DuckDBPyRelation, statement: str) -> bool:
    try:
        _seam.bind(node._con, statement)
    except _duckdb.Error:
        return False
    return True


def _stub_refused(node: DuckDBPyRelation, stubs: list[tuple[str, str]]) -> bool:
    """Whether the binder refuses one of the stand-ins on its own, which says the step's text is not what failed."""
    return any(not _binds(node, _assemble([stub], f"SELECT * FROM {stub[0]}")) for stub in stubs)


def _bind_with_stand_ins(
    node: DuckDBPyRelation,
    statement: str,
    stubs: list[tuple[str, str]],
    real: Callable[[], str],
    *,
    capture: bool = False,
    values: list[Any] | dict[str, Any] | None = None,
) -> _seam.Signature:
    """Bind a statement over typed stand-ins, or the real statement where a stand-in itself does not bind.

    A type whose text does not parse back cannot go into a stand-in, so the question goes to the real sources instead,
    which is slower and always correct. Only a stand-in the binder refuses qualifies, so a mistake in the step's
    own text still raises.
    """
    try:
        return _seam.bind(node._con, statement, capture=capture, values=values)
    except _duckdb.Error:
        if not _stub_refused(node, stubs):
            raise
    return _seam.bind(node._con, real(), capture=capture)


def _rendered_here(node: DuckDBPyRelation) -> builtins.list[DuckDBPyRelation]:
    """The nodes whose own text a node's SELECT holds: the steps it appended and the joins read in place.

    A variable one of their texts captured is then read under the key it was captured with.
    """
    nodes = [node]
    if node._open is None:
        return nodes
    source = _source_of(node)
    owner = _owner(node)
    current = node
    while current is not source and current._inputs:
        current = current._inputs[0]
        if current is source and owner is None:
            # A source read closed renders its own text in its own common table expression.
            break
        nodes.append(current)
    stack = [owner] if owner is not None and isinstance(owner._step, Join) else []
    while stack:
        join = stack.pop()
        for side, inline in _sides_of(join):
            if inline:
                nodes.append(side)
                stack.append(side)
    return nodes


def _ask(node: DuckDBPyRelation) -> tuple[builtins.list[str], builtins.list[DuckDBPyType]]:
    """The node's columns, asked of the binder over typed empty stand-ins for the steps it reads."""
    names: dict[int, str] = {}
    stubs: list[tuple[str, str]] = []
    writable = True
    for position, inp in enumerate(_references(node)):
        if id(inp) in names:
            continue
        stub = quote(f"__duckdb_in{position}")
        names[id(inp)] = stub
        # The input's own columns, duplicate names and all, which the boundary deduplicates as the real one does.
        text = _stub(*inp._schema)
        writable = writable and text is not None
        stubs.append((stub, text or ""))
    if writable:
        # A query's values are bound as the parameters they are, so its names and types are those of what runs.
        values = node._params if isinstance(node._step, Sql) else None
        signature = _bind_with_stand_ins(
            node, _stand_in_statement(node, names, stubs), stubs, _real(node), capture=True, values=values
        )
    else:
        signature = _seam.bind(node._con, _real(node)(), capture=True)
    if not isinstance(node._step, Held):
        # A held result answers for the statement that produced it, which a re-bind of its rows cannot know.
        node._return_type = signature.return_type
    if signature.captured:
        node._captured = (*node._captured, *((n, h.name) for n, h in signature.captured))
        node._handles = (*node._handles, *(h for _, h in signature.captured))
    return signature.names, signature.types


def _stand_in_statement(node: DuckDBPyRelation, names: dict[int, str], stubs: list[tuple[str, str]]) -> str:
    """The node's own statement over the stand-ins named in `names`, a query's parameters left as markers."""
    body = _body(node, names, top=True)
    ctes = _captured_ctes(_all_captured(_rendered_here(node))) + stubs
    if isinstance(node._step, Sql) and ctes:
        body = _wrap(body, node.alias)
    if isinstance(node._step, QueryOver):
        return _query_statement(node._step, _item(node._inputs[0], names, in_join=False), ctes)
    return _assemble(ctes, body)


def _real(node: DuckDBPyRelation) -> Callable[[], str]:
    return lambda: _render(node, literals=True)[0]


def _refresh(top: DuckDBPyRelation) -> None:
    """Bind again every node whose schema predates a statement that may have changed what its names read."""
    version = _seam.catalog_version(top._con)
    if top._version == version:
        return
    for node in _tree(top):
        if node._version != version:
            node._schema = _ask(node)
            node._version = version


class DuckDBPyRelation:
    """A query built a step at a time; it runs only when a method reads its rows, and runs again on every read."""

    __slots__ = (
        "__weakref__",
        "_alias",
        "_alias_from",
        "_captured",
        "_con",
        "_handles",
        "_inputs",
        "_open",
        "_params",
        "_return_type",
        "_schema",
        "_sides",
        "_step",
        "_table_kind",
        "_version",
    )

    _step: Step
    _inputs: tuple[DuckDBPyRelation, ...]
    _con: _duckdb.DuckDBPyConnection
    _schema: tuple[builtins.list[str], builtins.list[DuckDBPyType]]
    _alias: str | None
    _alias_from: DuckDBPyRelation | None
    _params: builtins.list[Any] | dict[str, Any] | None
    _handles: tuple[object, ...]
    _captured: tuple[tuple[str, str], ...]
    #: The SELECT this node is filling, or None where the node is a SELECT of its own.
    _open: _Select | None
    #: For a join, each side with whether the FROM reads it in place.
    _sides: tuple[tuple[DuckDBPyRelation, bool], ...] | None
    _version: int
    #: What the statement this relation holds returns: "query_result", "changed_rows" or "nothing".
    _return_type: str
    #: Whether the name of a `table()` source reads a base table, asked of the catalog when it is first needed.
    _table_kind: bool | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        msg = "DuckDBPyRelation: No constructor defined!"
        raise TypeError(msg)

    @classmethod
    def _make(
        cls,
        con: _duckdb.DuckDBPyConnection,
        step: Step,
        inputs: tuple[DuckDBPyRelation, ...] = (),
        *,
        alias: str | None = None,
        alias_from: DuckDBPyRelation | None = None,
        params: builtins.list[Any] | dict[str, Any] | None = None,
        handles: tuple[object, ...] = (),
        captured: tuple[tuple[str, str], ...] = (),
        schema: tuple[builtins.list[str], builtins.list[DuckDBPyType]] | None = None,
        return_type: str | None = None,
        sides: tuple[tuple[DuckDBPyRelation, bool], ...] | None = None,
    ) -> DuckDBPyRelation:
        rel = object.__new__(cls)
        rel._step = step
        rel._inputs = inputs
        rel._con = con
        rel._alias = alias
        rel._alias_from = alias_from
        rel._params = params
        rel._handles = handles
        rel._captured = captured
        rel._return_type = return_type or "query_result"
        rel._table_kind = None
        rel._version = _seam.catalog_version(con)
        rel._sides = None
        if isinstance(step, Join):
            # Decided once: asked again on every read, the decisions of the joins below it would double the cost of
            # each level of a chain of joins.
            chosen = (
                sides if sides is not None else tuple(_join_items(inputs[0], inputs[1], sides_closed=step.sides_closed))
            )
            rel._sides = _close_clashing_sides(con, step, chosen)
        rel._open = _open_for(step, inputs)
        rel._schema = schema if schema is not None else _ask(rel)
        if return_type is not None:
            # The statement the relation holds decides, not the bind of the step that reads its rows back.
            rel._return_type = return_type
        return rel

    def _derive(
        self,
        step: Step,
        *others: DuckDBPyRelation,
        sides: tuple[tuple[DuckDBPyRelation, bool], ...] | None = None,
    ) -> DuckDBPyRelation:
        inputs = (self, *others)
        for other in others:
            _ensure_open(other._con)
            if other._con is not self._con:
                message = "Cannot combine LEFT and RIGHT relations of different connections!"
                raise _error(_duckdb.InvalidInputException, message)
        for relation in inputs:
            _refresh(relation)
        if others:
            _refuse_shared_names(inputs)
        alias_from = self if not isinstance(step, (Join, Alias)) else None
        alias = step.alias if isinstance(step, Alias) else None
        if isinstance(step, Join):
            alias = f"{step.kind}_{secrets.token_hex(8)}"
        return DuckDBPyRelation._make(self._con, step, inputs, alias=alias, alias_from=alias_from, sides=sides)

    def _input(self, other: object) -> DuckDBPyRelation:
        """A relation to combine with this one, refused first when it belongs to another connection.

        The connection check comes before any check of names or aliases, so the message a user gets does not depend
        on what the two relations happen to be called.
        """
        if not isinstance(other, DuckDBPyRelation):
            msg = f"Expected a DuckDBPyRelation, not {type(other).__name__}"
            raise TypeError(msg)
        _ensure_open(other._con)
        if other._con is not self._con:
            message = "Cannot combine LEFT and RIGHT relations of different connections!"
            raise _error(_duckdb.InvalidInputException, message)
        return other

    def _fresh(self) -> tuple[builtins.list[str], builtins.list[DuckDBPyType]]:
        _refresh(self)
        return self._schema

    def _statement(self) -> tuple[str, builtins.list[Any] | dict[str, Any] | None]:
        return _render(self)

    def _submit(self, format: str = "rows", batch_size: int = _seam.DEFAULT_BATCH_SIZE) -> _seam.Result:
        sql, values = self._statement()
        return _seam.submit(self._con, sql, values, format, batch_size)

    def _sources(self) -> builtins.list[_seam.SourceHandle]:
        """Every registry handle the relation reads through, so a view over it can keep them alive."""
        found: builtins.list[_seam.SourceHandle] = []
        for node in _tree(self):
            for handle in node._handles:
                if isinstance(handle, DuckDBPyRelation):
                    found.extend(handle._sources())
                else:
                    found.append(handle)  # type: ignore[arg-type]
        return found

    def sql_query(self) -> str:
        """Get the SQL query that is equivalent to the relation; values given as `params=` stay markers."""
        # A user's parameters are bound at execution, so the text names them as the user did and
        # runs with the same values; Python values inside expressions are written in, as they always were.
        return _render(self)[0]

    def _registry_text(self) -> str:
        """The relation's text with its parameter values written in, for a reader that cannot bind them."""
        return _render(self, literals=True)[0]

    @property
    def alias(self) -> str:
        """Get the name of the current alias."""
        # A loop, so a chain of steps longer than Python's recursion limit still has an alias.
        node = self
        while node._alias is None and node._alias_from is not None:
            node = node._alias_from
        if node._alias is None:
            node._alias = _unnamed()
        return node._alias

    @property
    def type(self) -> str:
        """Get the type of the relation."""
        step = self._step
        if type(step) is Table and step.base:
            return "TABLE_RELATION" if self._reads_a_table() else "QUERY_RELATION"
        return step.type_name()

    def _reads_a_table(self) -> bool:
        """Whether the name this source reads is a base table, asked of the catalog once and kept.

        The question is asked when it is first needed rather than at construction, so building a relation never needs
        the connection to be free of an open result.
        """
        step = self._step
        if type(step) is not Table or not step.base:
            return False
        if self._table_kind is None:
            self._table_kind = _seam.table_kind(self._con, step.parts) == "table"
        return self._table_kind

    @property
    def columns(self) -> builtins.list[str]:
        """Return a list containing the names of the columns of the relation."""
        return list(self._fresh()[0])

    @property
    def types(self) -> builtins.list[DuckDBPyType]:
        """Return a list containing the types of the columns of the relation."""
        return list(self._fresh()[1])

    @property
    def dtypes(self) -> builtins.list[DuckDBPyType]:
        """Return a list containing the types of the columns of the relation."""
        return list(self._fresh()[1])

    @property
    def description(self) -> builtins.list[tuple[str, DuckDBPyType, None, None, None, None, None]]:
        """Return the description of the result."""
        return [(n, t, None, None, None, None, None) for n, t in zip(*self._fresh(), strict=True)]

    @property
    def shape(self) -> tuple[int, int]:
        """Tuple of # of rows, # of columns in relation."""
        return len(self), len(self._fresh()[0])

    def __len__(self) -> int:
        """Number of rows in relation."""
        return int(self._aggregate_shortcut("count", "*").fetchall()[0][0])

    def __contains__(self, name: str) -> bool:
        folded = fold_name(_one_name(name))
        return any(fold_name(column) == folded for column in self._fresh()[0])

    def __getattr__(self, name: str) -> DuckDBPyRelation:
        """Get a projection relation created from this relation, on the provided column name."""
        if name.startswith("__") or name in DuckDBPyRelation.__slots__:
            raise AttributeError(name)
        return self._column(name)

    def __getitem__(self, name: str) -> DuckDBPyRelation:
        """Get a projection relation created from this relation, on the provided column name."""
        return self._column(name)

    def _column(self, name: str) -> DuckDBPyRelation:
        name = _one_name(name)
        try:
            names, types = self._fresh()
        except _duckdb.ConnectionException:
            # A closed connection cannot change which names are columns, so the last bound schema answers, and
            # `hasattr` gives False for a name that is not one rather than raising; a name that is a
            # column still reaches the closed connection below when its projection is built.
            names, types = self._schema
        reference: str | None = None
        if len(names) == 1 and types[0].id == "struct" and any(field == name for field, _ in types[0].children):
            reference = f"{quote(names[0])}.{quote(name)}"
        elif any(fold_name(column) == fold_name(name) for column in names):
            reference = quote(name)
        if reference is None:
            replacement = _REMOVED_METHODS.get(name)
            if replacement is not None:
                msg = (
                    f"Relations no longer have {name}(): a relation keeps no result between calls, so every call "
                    f"runs it again. Use {replacement} instead."
                )
                raise AttributeError(msg)
            msg = f"This relation does not contain a column by the name of '{name}'"
            raise AttributeError(msg)
        return self._derive(Project(f"{reference} AS {quote(name)}"))

    def filter(self, filter_expr: object) -> DuckDBPyRelation:
        """Filter the relation object by the filter in filter_expr."""
        _warn_on_literal_names(filter_expr, self.columns)
        return self._derive(Filter(_condition_text(self._con, filter_expr, "filter condition")))

    def project(self, *args: object, groups: str = "") -> DuckDBPyRelation:
        """Project the relation object by the projection in project_expr."""
        return self._projection("project", args, groups)

    def select(self, *args: object, groups: str = "") -> DuckDBPyRelation:
        """Project the relation object by the projection in project_expr."""
        return self._projection("select", args, groups)

    def _projection(self, verb: str, args: tuple[object, ...], groups: str) -> DuckDBPyRelation:
        if not args:
            msg = f"{verb}() needs at least one expression or column name"
            raise TypeError(msg)
        columns = self.columns
        for arg in args:
            _warn_on_literal_names(arg, columns)
        items = _items_of(self._con, args)
        if groups:
            return self._derive(Aggregate(items, _group_list(self._con, groups) or None))
        return self._derive(Project(items))

    def select_types(self, types: object) -> DuckDBPyRelation:
        """Select columns from the relation, by filtering based on type(s)."""
        if not isinstance(types, list):
            raise _error(_duckdb.InvalidInputException, "'columns_by_type' expects a list containing types")
        wanted: builtins.list[DuckDBPyType] = []
        for item in types:
            if isinstance(item, str):
                wanted.append(_seam.parse_type(self._con, item))
            elif isinstance(item, _api().sqltypes.DuckDBPyType):
                wanted.append(item)
            else:
                message = f"Can only project on objects of type DuckDBPyType or str, not '{type(item)}'"
                raise _error(_duckdb.InvalidInputException, message)
        if not wanted:
            raise _error(_duckdb.InvalidInputException, "List of types can not be empty!")
        chosen = [quote(n) for n, t in zip(*self._fresh(), strict=True) if any(t == w for w in wanted)]
        if not chosen:
            raise _error(_duckdb.InvalidInputException, "None of the columns matched the provided type filter!")
        return self._derive(Project(", ".join(chosen)))

    select_dtypes = select_types

    def set_alias(self, alias: str) -> DuckDBPyRelation:
        """Rename the relation object to new alias."""
        return self._derive(Alias(alias))

    def order(self, order_expr: object) -> DuckDBPyRelation:
        """Reorder the relation object by order_expr, which is SQL order text or an expression."""
        _warn_on_literal_names(order_expr, self.columns)
        return self._derive(Order(_order_item(self._con, order_expr)))

    def sort(self, *args: object) -> DuckDBPyRelation:
        """Reorder the relation object by the provided expressions."""
        if not args:
            raise _error(_duckdb.InvalidInputException, "Please provide at least one expression to sort on")
        columns = self.columns
        for arg in args:
            _warn_on_literal_names(arg, columns)
        return self._derive(Order(", ".join(_order_item(self._con, arg) for arg in args)))

    def limit(self, n: int, offset: int = 0) -> DuckDBPyRelation:
        """Only retrieve the first n rows from this relation object, starting at offset."""
        return self._derive(Limit(_row_count("limit", "n", n), _row_count("limit", "offset", offset)))

    def distinct(self, on: object = None) -> DuckDBPyRelation:
        """Retrieve distinct rows from this relation object, or the first row of each distinct value of `on`."""
        if on is None:
            return self._derive(Distinct())
        keys = [on] if isinstance(on, str) or _is_expression(on) else list(on)  # type: ignore[call-overload]
        return self._derive(Distinct(", ".join(_plain_item(self._con, k) for k in keys)))

    def aggregate(self, aggr_expr: object, group_expr: object = "") -> DuckDBPyRelation:
        """Compute the aggregate aggr_expr by the optional groups group_expr on the relation."""
        if isinstance(aggr_expr, str):
            items = _select_list(self._con, aggr_expr)
        elif isinstance(aggr_expr, (list, tuple)):
            columns = self.columns
            for item in aggr_expr:
                _warn_on_literal_names(item, columns)
            items = ", ".join(_select_item(self._con, item) for item in aggr_expr)
        else:
            message = f"Please provide either a string or list of Expression objects, not {type(aggr_expr)}"
            raise _error(_duckdb.InvalidInputException, message)
        groups = None
        if isinstance(group_expr, str):
            groups = _group_list(self._con, group_expr) or None
        elif group_expr is not None:
            groups = _expression_text(group_expr)
        return self._derive(Aggregate(items, groups))

    def union(self, union_rel: DuckDBPyRelation) -> DuckDBPyRelation:
        """Create the set union of this relation object with another relation object in other_rel."""
        return self._derive(SetOp("UNION ALL"), self._input(union_rel))

    def except_(self, other_rel: DuckDBPyRelation) -> DuckDBPyRelation:
        """Create the set except of this relation object with another relation object in other_rel."""
        return self._derive(SetOp("EXCEPT ALL"), self._input(other_rel))

    def intersect(self, other_rel: DuckDBPyRelation) -> DuckDBPyRelation:
        """Create the set intersection of this relation object with another relation object in other_rel."""
        return self._derive(SetOp("INTERSECT ALL"), self._input(other_rel))

    def union_by_name(self, other: DuckDBPyRelation, *, all: bool = True) -> DuckDBPyRelation:
        """Rows from both relations, matching columns by name; keeps duplicates unless `all` is false."""
        keyword = "UNION ALL BY NAME" if all else "UNION BY NAME"
        return self._derive(SetOp(keyword), self._input(other))

    def join(
        self, other_rel: object, condition: object = None, how: str = "inner", suffix: str | None = None
    ) -> DuckDBPyRelation:
        """Join the relation object with another relation object in other_rel.

        The condition is SQL text, a column name or a list of them for a USING join, an Expression, or a function of
        the two sides. Types supported are 'inner', 'left', 'right', 'outer', 'full', 'semi', 'anti', 'cross',
        'positional', 'natural' and 'asof'. With `suffix`, a column of the right side whose name the left side
        already has is renamed with the suffix appended.
        """
        if other_rel is None:
            raise _error(_duckdb.InvalidInputException, "No relation provided for join")
        other = self._input(other_rel)
        kind = fold_name(str(how).strip())
        if kind not in _JOIN_KINDS:
            listed = ", ".join(f"'{k}'" for k in _JOIN_KINDS)
            raise _error(_duckdb.InvalidInputException, f"Unsupported join type {how}, try one of: {listed}")
        left_alias, right_alias = self.alias, other.alias
        # A join option that needs its sides' column names or aliases closes the sides, so each enters the FROM as one
        # item under its own alias: a condition written as a function names each side by its alias, and a suffix
        # renames the right side's names that the left side has, which are the names each side exposes.
        function_condition = callable(condition) and not isinstance(condition, (str, list, tuple))
        renames = suffix is not None and kind not in ("semi", "anti")
        sides_closed = (function_condition and not _is_expression(condition)) or renames
        sides = tuple(_join_items(self, other, sides_closed=sides_closed))
        folded = [fold_name(alias) for side, inline in sides for alias in _from_aliases(side, in_join=inline)]
        if len(set(folded)) != len(folded):
            # Every item of the FROM is in one scope, so a repeated alias would hide the columns bound under it.
            message = (
                "Both relations have the same alias, please change the alias of one or both relations using "
                "'rel = rel.set_alias(<new alias>)'"
            )
            raise _error(_duckdb.InvalidInputException, message)
        keyword, needs_condition = _JOIN_KINDS[kind]
        _warn_on_literal_names(condition, [*self.columns, *other.columns])
        on, using = _join_condition(self._con, condition, left_alias, right_alias)
        if needs_condition and on is None and not using:
            raise _error(_duckdb.InvalidInputException, f"Join type '{kind}' needs a condition")
        if not needs_condition and (on is not None or using):
            raise _error(_duckdb.InvalidInputException, f"Join type '{kind}' takes no condition")
        projection = "*"
        if renames:
            assert suffix is not None
            # Here rather than in _boundary_names: the join knows its condition, user code, has run.
            _refresh(self)
            _refresh(other)
            left_names, right_names = _boundary_names(self), _boundary_names(other)
            # The names the join merges, which a suffix leaves alone: its USING list, or every shared name for a
            # natural join, whose star already gives each once.
            merged = using
            if kind == "natural":
                taken = {fold_name(name) for name in left_names}
                merged = tuple(name for name in right_names if fold_name(name) in taken)
            projection = _suffixed((left_alias, left_names), (right_alias, right_names), merged, suffix)
        step_class = Cross if kind in ("cross", "positional") else Join
        return self._derive(step_class(keyword, on, using, projection, sides_closed), other, sides=sides)

    def cross(self, other_rel: DuckDBPyRelation) -> DuckDBPyRelation:
        """Create cross/cartesian product of two relational objects."""
        return self._derive(Cross("CROSS"), self._input(other_rel))

    def with_columns(self, **columns: object) -> DuckDBPyRelation:
        """Add columns, replacing any of the same name in place."""
        if not columns:
            msg = "with_columns needs at least one column"
            raise TypeError(msg)
        existing = {fold_name(name) for name in self.columns}
        rendered = {n: _plain_item(self._con, v) for n, v in columns.items()}
        replaced = [f"{text} AS {quote(n)}" for n, text in rendered.items() if fold_name(n) in existing]
        added = [f"{text} AS {quote(n)}" for n, text in rendered.items() if fold_name(n) not in existing]
        star = f"* REPLACE ({', '.join(replaced)})" if replaced else "*"
        return self._derive(Star(", ".join([star, *added])))

    def drop(self, *columns: str) -> DuckDBPyRelation:
        """Remove columns."""
        if not columns:
            msg = "drop needs at least one column"
            raise TypeError(msg)
        return self._derive(Star(f"* EXCLUDE ({', '.join(quote(c) for c in columns)})"))

    def rename(self, **columns: str) -> DuckDBPyRelation:
        """Rename columns, given as old=new."""
        if not columns:
            msg = "rename needs at least one column"
            raise TypeError(msg)
        present = {fold_name(name) for name in self.columns}
        for old in columns:
            if fold_name(old) not in present:
                # The engine ignores a missing name in a RENAME list, where it refuses one in EXCLUDE, so `drop` and
                # `rename` would disagree about the same mistake; both refuse it, in the engine's words.
                raise _error(_duckdb.BinderException, f'Column "{old}" in RENAME list not found in FROM clause')
        pairs = ", ".join(f"{quote(old)} AS {quote(new)}" for old, new in columns.items())
        return self._derive(Star(f"* RENAME ({pairs})"))

    def unnest(self, *columns: str) -> DuckDBPyRelation:
        """Expand list columns to one row per element."""
        if not columns:
            msg = "unnest needs at least one column"
            raise TypeError(msg)
        expanded = ", ".join(f"unnest({quote(c)}) AS {quote(c)}" for c in columns)
        return self._derive(Star(f"* REPLACE ({expanded})"))

    def unpivot(self, *columns: str, name: str = "name", value: str = "value") -> DuckDBPyRelation:
        """Turn the named columns into rows; `name` and `value` name the two new columns."""
        if not columns:
            msg = "unpivot needs at least one column"
            raise TypeError(msg)
        return self._derive(Unpivot(tuple(columns), name, value))

    def pivot(self, column: object, values: Iterable[object], aggregate: object) -> DuckDBPyRelation:
        """Turn the given values of `column` into columns holding `aggregate`; the values are explicit."""
        listed = tuple(_expression_text(v) for v in values)
        if not listed:
            msg = "pivot needs at least one value"
            raise TypeError(msg)
        return self._derive(Pivot(_plain_item(self._con, column), listed, _condition_text(self._con, aggregate, None)))

    def sample(
        self, n: int | None = None, *, percent: float | None = None, seed: int | None = None, method: str | None = None
    ) -> DuckDBPyRelation:
        """A random subset: `n` rows, or `percent` of them, with `seed` repeating a draw."""
        if (n is None) == (percent is None):
            msg = "sample takes either n or percent"
            raise TypeError(msg)
        if method is not None and fold_name(method) not in _SAMPLE_METHODS:
            msg = f"unknown sample method {method!r}; one of {', '.join(_SAMPLE_METHODS)}"
            raise ValueError(msg)
        size = f"{_row_count('sample', 'n', n)} ROWS" if n is not None else f"{float(percent or 0)} PERCENT"
        chosen = fold_name(method) if method else ("reservoir" if n is not None else "bernoulli")
        arguments = f"{chosen}, {int(seed)}" if seed is not None else chosen
        return self._derive(Sample(size, arguments))

    def head(self, n: int = 5) -> DuckDBPyRelation:
        """The first `n` rows."""
        return self._derive(Limit(_row_count("head", "n", n), 0))

    def offset(self, n: int) -> DuckDBPyRelation:
        """Skip `n` rows."""
        return self._derive(Limit(None, _row_count("offset", "n", n)))

    def group_by(self, *columns: object) -> GroupedRelation:
        """Begin a grouped aggregation; continue with `.agg(...)`."""
        return GroupedRelation(self, columns)

    def _expression_list(
        self,
        function_name: str,
        aggregated_columns: str,
        function_parameter: str = "",
        projected_columns: str = "",
        window_spec: str = "",
        *,
        ignore_nulls: bool = False,
    ) -> str:
        pieces = _split_list(aggregated_columns)
        if fold_name(function_name) == "count" and not pieces:
            pieces = ["*"]
        # Each piece of user text ends its line, so a line comment in one cannot reach the SQL written after it.
        text = f"{_select_list(self._con, projected_columns)}, " if projected_columns else ""
        nulls = " ignore nulls) " if ignore_nulls else ") "
        after = f"{nulls}{_spliced(window_spec)}" if window_spec else nulls
        parameter = _spliced(function_parameter) if function_parameter else ""
        if not pieces and function_parameter:
            return f"{text}{function_name}({parameter}{after}"
        rendered: builtins.list[str] = []
        for piece in pieces:
            trimmed = piece.strip()
            if not trimmed:
                raise _error(_duckdb.ParserException, f"Invalid column expression: '{piece}'")
            try:
                count = _seam.check_text(self._con, _encodable(trimmed), "expressions")
            except _duckdb.ParserException:
                # Text that does not parse names one column, a name with a space in it for one.
                expression = quote(trimmed)
            else:
                if count != 1:
                    raise _error(_duckdb.ParserException, f"Invalid column expression: {trimmed}")
                expression = _spliced(trimmed)
            argument = expression if not function_parameter else f"{expression},{parameter}"
            rendered.append(f"{function_name}({argument}{after}")
        return text + ",".join(rendered)

    def _aggregate_shortcut(
        self,
        function_name: str,
        columns: str,
        groups: str = "",
        parameter: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        text = self._expression_list(function_name, columns, parameter, projected_columns)
        return self.aggregate(text, groups)

    def _window_shortcut(
        self,
        function_name: str,
        parameters: str,
        columns: str,
        window_spec: str,
        projected_columns: str = "",
        *,
        ignore_nulls: bool = False,
    ) -> DuckDBPyRelation:
        text = self._expression_list(
            function_name, columns, parameters, projected_columns, window_spec, ignore_nulls=ignore_nulls
        )
        return self._derive(Project(_select_list(self._con, text)))

    def _aggregate_or_window(
        self,
        function_name: str,
        columns: str,
        parameter: str = "",
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        if groups and window_spec:
            raise _error(
                _duckdb.InvalidInputException, "Either groups or window must be set (can't be both at the same time)"
            )
        if window_spec:
            return self._window_shortcut(function_name, parameter, columns, window_spec, projected_columns)
        return self._aggregate_shortcut(function_name, columns, groups, parameter, projected_columns)

    def apply(
        self,
        function_name: str,
        function_aggr: str,
        group_expr: str = "",
        function_parameter: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """Compute the function of a single column or a list of columns by the optional groups on the relation."""
        return self._aggregate_shortcut(function_name, function_aggr, group_expr, function_parameter, projected_columns)

    def any_value(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Returns the first non-null value from a given expression."""
        return self._aggregate_or_window("any_value", expression, "", groups, window_spec, projected_columns)

    def arg_max(
        self, arg_column: str, value_column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Finds the row with the maximum value for a value column and returns the value of that row for an argument column."""  # noqa: E501
        return self._aggregate_or_window("arg_max", arg_column, value_column, groups, window_spec, projected_columns)

    def arg_min(
        self, arg_column: str, value_column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Finds the row with the minimum value for a value column and returns the value of that row for an argument column."""  # noqa: E501
        return self._aggregate_or_window("arg_min", arg_column, value_column, groups, window_spec, projected_columns)

    def avg(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the average of a given expression."""
        return self._aggregate_or_window("avg", expression, "", groups, window_spec, projected_columns)

    mean = avg

    def bit_and(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the bitwise AND of all bits present in a given expression."""
        return self._aggregate_or_window("bit_and", expression, "", groups, window_spec, projected_columns)

    def bit_or(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the bitwise OR of all bits present in a given expression."""
        return self._aggregate_or_window("bit_or", expression, "", groups, window_spec, projected_columns)

    def bit_xor(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the bitwise XOR of all bits present in a given expression."""
        return self._aggregate_or_window("bit_xor", expression, "", groups, window_spec, projected_columns)

    def bitstring_agg(
        self,
        expression: str,
        min: int | None = None,
        max: int | None = None,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """Computes a bitstring with bits set for each distinct value in a given expression."""
        if (min is None) != (max is None):
            raise _error(_duckdb.InvalidInputException, "Both min and max values must be set")
        parameter = ""
        if min is not None:
            if not isinstance(min, int) or not isinstance(max, int):
                raise _error(_duckdb.InvalidTypeException, "min and max must be of type int")
            parameter = f"{min},{max}"
        return self._aggregate_or_window("bitstring_agg", expression, parameter, groups, window_spec, projected_columns)

    def bool_and(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the logical AND of all values present in a given expression."""
        return self._aggregate_or_window("bool_and", expression, "", groups, window_spec, projected_columns)

    def bool_or(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the logical OR of all values present in a given expression."""
        return self._aggregate_or_window("bool_or", expression, "", groups, window_spec, projected_columns)

    def count(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the number of elements present in a given expression."""
        return self._aggregate_or_window("count", expression, "", groups, window_spec, projected_columns)

    def value_counts(self, expression: str, groups: str = "") -> DuckDBPyRelation:
        """Computes the number of elements present in a given expression, also projecting the original expression."""
        return self.count(expression, groups, "", expression)

    def favg(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the average of all values present in a given expression using a more accurate floating point summation (Kahan Sum)."""  # noqa: E501
        return self._aggregate_or_window("favg", expression, "", groups, window_spec, projected_columns)

    def first(self, expression: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation:
        """Returns the first value of a given expression."""
        return self._aggregate_shortcut("first", expression, groups, "", projected_columns)

    def fsum(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the sum of all values present in a given expression using a more accurate floating point summation (Kahan Sum)."""  # noqa: E501
        return self._aggregate_or_window("fsum", expression, "", groups, window_spec, projected_columns)

    def geomean(self, expression: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation:
        """Computes the geometric mean over all values present in a given expression."""
        return self._aggregate_shortcut("geomean", expression, groups, "", projected_columns)

    def histogram(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the histogram over all values present in a given expression."""
        return self._aggregate_or_window("histogram", expression, "", groups, window_spec, projected_columns)

    def list(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Returns a list containing all values present in a given expression."""
        return self._aggregate_or_window("list", expression, "", groups, window_spec, projected_columns)

    def last(self, expression: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation:
        """Returns the last value of a given expression."""
        return self._aggregate_shortcut("last", expression, groups, "", projected_columns)

    def max(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Returns the maximum value present in a given expression."""
        return self._aggregate_or_window("max", expression, "", groups, window_spec, projected_columns)

    def min(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Returns the minimum value present in a given expression."""
        return self._aggregate_or_window("min", expression, "", groups, window_spec, projected_columns)

    def product(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Returns the product of all values present in a given expression."""
        return self._aggregate_or_window("product", expression, "", groups, window_spec, projected_columns)

    def string_agg(
        self, expression: str, sep: str = ",", groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Concatenates the values present in a given expression with a separator."""
        separator = "'" + sep.replace("'", "''") + "'"
        return self._aggregate_or_window("string_agg", expression, separator, groups, window_spec, projected_columns)

    def sum(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the sum of all values present in a given expression."""
        return self._aggregate_or_window("sum", expression, "", groups, window_spec, projected_columns)

    def unique(self, unique_aggr: str) -> DuckDBPyRelation:
        """Returns the distinct values in a column."""
        return self._derive(Project(_select_list(self._con, unique_aggr))).distinct()

    def median(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the median over all values present in a given expression."""
        return self._aggregate_or_window("median", expression, "", groups, window_spec, projected_columns)

    def mode(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the mode over all values present in a given expression."""
        return self._aggregate_or_window("mode", expression, "", groups, window_spec, projected_columns)

    def quantile_cont(
        self,
        expression: str,
        q: float | builtins.list[float] = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """Computes the interpolated quantile value for a given expression."""
        return self._aggregate_or_window(
            "quantile_cont", expression, _quantiles(q), groups, window_spec, projected_columns
        )

    def quantile_disc(
        self,
        expression: str,
        q: float | builtins.list[float] = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """Computes the exact quantile value for a given expression."""
        return self._aggregate_or_window(
            "quantile_disc", expression, _quantiles(q), groups, window_spec, projected_columns
        )

    quantile = quantile_disc

    def stddev_pop(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the population standard deviation for a given expression."""
        return self._aggregate_or_window("stddev_pop", expression, "", groups, window_spec, projected_columns)

    def stddev_samp(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the sample standard deviation for a given expression."""
        return self._aggregate_or_window("stddev_samp", expression, "", groups, window_spec, projected_columns)

    stddev = stddev_samp
    std = stddev_samp

    def var_pop(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the population variance for a given expression."""
        return self._aggregate_or_window("var_pop", expression, "", groups, window_spec, projected_columns)

    def var_samp(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """Computes the sample variance for a given expression."""
        return self._aggregate_or_window("var_samp", expression, "", groups, window_spec, projected_columns)

    variance = var_samp
    var = var_samp

    def row_number(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """Computes the row number within the partition."""
        return self._window_shortcut("row_number", "", "*", window_spec, projected_columns)

    def rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """Computes the rank within the partition."""
        return self._window_shortcut("rank", "", "*", window_spec, projected_columns)

    def dense_rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """Computes the dense rank within the partition."""
        return self._window_shortcut("dense_rank", "", "*", window_spec, projected_columns)

    rank_dense = dense_rank

    def percent_rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """Computes the relative rank within the partition."""
        return self._window_shortcut("percent_rank", "", "*", window_spec, projected_columns)

    def cume_dist(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """Computes the cumulative distribution within the partition."""
        return self._window_shortcut("cume_dist", "", "*", window_spec, projected_columns)

    def first_value(self, expression: str, window_spec: str = "", projected_columns: str = "") -> DuckDBPyRelation:
        """Computes the first value within the group or partition."""
        return self._window_shortcut("first_value", "", expression, window_spec, projected_columns)

    def n_tile(self, window_spec: str, num_buckets: int, projected_columns: str = "") -> DuckDBPyRelation:
        """Divides the partition as equally as possible into num_buckets."""
        return self._window_shortcut("ntile", str(int(num_buckets)), "", window_spec, projected_columns)

    def lag(
        self,
        expression: str,
        window_spec: str,
        offset: int = 1,
        default_value: str = "NULL",
        ignore_nulls: bool = False,  # noqa: FBT001
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """Computes the lag within the partition."""
        return self._window_shortcut(
            "lag",
            _offset_parameters(offset, default_value),
            expression,
            window_spec,
            projected_columns,
            ignore_nulls=ignore_nulls,
        )

    def last_value(self, expression: str, window_spec: str = "", projected_columns: str = "") -> DuckDBPyRelation:
        """Computes the last value within the group or partition."""
        return self._window_shortcut("last_value", "", expression, window_spec, projected_columns)

    def lead(
        self,
        expression: str,
        window_spec: str,
        offset: int = 1,
        default_value: str = "NULL",
        ignore_nulls: bool = False,  # noqa: FBT001
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """Computes the lead within the partition."""
        return self._window_shortcut(
            "lead",
            _offset_parameters(offset, default_value),
            expression,
            window_spec,
            projected_columns,
            ignore_nulls=ignore_nulls,
        )

    def nth_value(
        self,
        expression: str,
        window_spec: str,
        offset: int,
        ignore_nulls: bool = False,  # noqa: FBT001
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """Computes the nth value within the partition."""
        return self._window_shortcut(
            "nth_value", str(int(offset)), expression, window_spec, projected_columns, ignore_nulls=ignore_nulls
        )

    def describe(self) -> DuckDBPyRelation:
        """Gives basic statistics (e.g., min, max) and if NULL exists for each column of the relation."""
        statistics = (
            ("count", False),
            ("mean", True),
            ("stddev", True),
            ("min", False),
            ("max", False),
            ("median", True),
        )
        inner: builtins.list[str] = []
        outer = ["UNNEST([" + ", ".join(f"'{name}'" for name, _ in statistics) + "]) AS aggr"]
        for c, (name, column_type) in enumerate(zip(*self._fresh(), strict=True)):
            numeric = column_type.id in _NUMERIC_TYPE_IDS
            aliases = [quote(f"__describe_{c}_{i}") for i in range(len(statistics))]
            for (function, numeric_only), alias in zip(statistics, aliases, strict=True):
                if numeric_only and not numeric:
                    stat = "NULL"
                else:
                    stat = f"{function}({quote(name)})::{'DOUBLE' if numeric else 'VARCHAR'}"
                inner.append(f"{stat} AS {alias}")
            outer.append(f"UNNEST([{', '.join(aliases)}]) AS {quote(name)}")
        stats = self._derive(Aggregate(", ".join(inner)))
        return stats._derive(Project(", ".join(outer)))

    def fetchall(self) -> builtins.list[tuple[Any, ...]]:
        """Execute and fetch all rows as a list of tuples."""
        return self._submit().fetchall()

    def fetchnumpy(self) -> dict[str, Any]:
        """Execute and fetch all rows as a Python dict mapping each column to one numpy arrays."""
        return self._submit().fetchnumpy()

    def df(self, *, date_as_object: bool = False) -> Any:  # noqa: ANN401
        """Execute and fetch all rows as a pandas DataFrame."""
        return self._submit().df(date_as_object=date_as_object)

    fetchdf = df
    to_df = df

    def torch(self) -> dict[str, Any]:
        """Fetch a result as dict of PyTorch Tensors."""
        return self._submit().torch()

    def tf(self) -> dict[str, Any]:
        """Fetch a result as dict of TensorFlow Tensors."""
        return self._submit().tf()

    def to_arrow_table(self, batch_size: int = _seam.DEFAULT_BATCH_SIZE) -> Any:  # noqa: ANN401
        """Execute and fetch all rows as an Arrow Table."""
        return self._submit("arrow", int(batch_size)).arrow_table()

    def to_arrow_reader(self, batch_size: int = _seam.DEFAULT_BATCH_SIZE) -> Any:  # noqa: ANN401
        """Execute and return an Arrow Record Batch Reader that yields all rows."""
        if int(batch_size) == 0:
            msg = "Approximate Batch Size of Record Batch MUST be higher than 0"
            raise RuntimeError(msg)
        return self._submit("stream", int(batch_size)).arrow_reader()

    def arrow(self, batch_size: int = _seam.DEFAULT_BATCH_SIZE) -> Any:  # noqa: ANN401
        """Alias of to_arrow_reader(). We recommend using to_arrow_reader() instead."""
        return self.to_arrow_reader(batch_size)

    def fetch_arrow_table(self, batch_size: int = _seam.DEFAULT_BATCH_SIZE) -> Any:  # noqa: ANN401
        """Deprecated: use to_arrow_table() instead."""
        warnings.warn(
            "fetch_arrow_table() is deprecated, use to_arrow_table() instead.", DeprecationWarning, _caller_level()
        )
        return self.to_arrow_table(batch_size)

    def fetch_record_batch(self, rows_per_batch: int = _seam.DEFAULT_BATCH_SIZE) -> Any:  # noqa: ANN401
        """Deprecated: use to_arrow_reader() instead."""
        warnings.warn(
            "fetch_record_batch() is deprecated, use to_arrow_reader() instead.", DeprecationWarning, _caller_level()
        )
        return self.to_arrow_reader(rows_per_batch)

    def fetch_arrow_reader(self, batch_size: int = _seam.DEFAULT_BATCH_SIZE) -> Any:  # noqa: ANN401
        """Deprecated: use to_arrow_reader() instead."""
        warnings.warn(
            "fetch_arrow_reader() is deprecated, use to_arrow_reader() instead.", DeprecationWarning, _caller_level()
        )
        return self.to_arrow_reader(batch_size)

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> Any:  # noqa: ANN401
        """Execute and return an ArrowArrayStream through the Arrow PyCapsule Interface.

        https://arrow.apache.org/docs/dev/format/CDataInterface/PyCapsuleInterface.html
        """
        return self._submit("stream").arrow_capsule()

    def pl(self, batch_size: int = _seam.DEFAULT_BATCH_SIZE, *, lazy: bool = False) -> Any:  # noqa: ANN401
        """Execute and fetch all rows as a Polars DataFrame."""
        import polars

        if not lazy:
            table = self._submit("arrow", int(batch_size)).arrow_table(to_polars=True)
            return polars.from_arrow(table, rechunk=False)
        from duckdb.polars_io import duckdb_source

        # Polars needs unique names, which are the names the engine gives the relation read as a subquery, so the scan
        # reads it that way too and its rows carry the names the schema promises.
        text = _wrap(_render(self, literals=True)[0], _unnamed())
        unique = DuckDBPyRelation._make(self._con, Sql(text), alias=_unnamed(), handles=(self,))
        schema = polars.DataFrame(_seam.arrow_schema(self._con, unique.columns, unique.types).empty_table()).schema
        return duckdb_source(unique, schema)

    def to_dicts(self) -> builtins.list[dict[str, Any]]:
        """Every row, as a dict keyed by column name."""
        names = self.columns
        return [dict(zip(names, row, strict=True)) for row in self.fetchall()]

    def iter_rows(self, batch_size: int = 1024) -> Iterator[tuple[Any, ...]]:
        """Every row, a batch at a time, over one run of the relation.

        The rows are read once into the engine's memory, so every batch comes from the same run and the connection
        is free between batches; the caller holds one batch at a time rather than the whole result.
        """
        size = _row_count("iter_rows", "batch_size", batch_size)
        if size <= 0:
            msg = "iter_rows takes a batch size above zero"
            raise ValueError(msg)
        # One submission whose rows the engine keeps, read a batch at a time: linear in the rows and one bind,
        # where paging with a limit and an offset rescanned the rows before each batch and bound once per batch. A
        # result of format "rows" runs to its end, so the connection is free between batches.
        result = self._submit()
        while batch := result.fetchmany(size):
            yield from batch

    def close(self) -> None:
        """Does nothing: a relation holds no result."""

    def _shown(self, rows: DuckDBPyRelation) -> DuckDBPyRelation:
        """Rows for a display, drawn under the relation's own names, which `columns` reports.

        A display reads the relation under a row cap, or holds its rows, and either can be a boundary where the
        engine deduplicates names; the rows keep their order, so each takes the relation's name at its position.
        """
        if rows.columns == self.columns:
            return rows
        pairs = zip(rows.columns, self.columns, strict=True)
        return rows._derive(Project(", ".join(f"{quote(held)} AS {quote(own)}" for held, own in pairs)))

    def _box(self, **options: Any) -> str:  # noqa: ANN401
        return self._shown(self.limit(_BOX_LIMIT))._submit().to_box(**options)

    def __str__(self) -> str:
        return self._box()

    def __repr__(self) -> str:
        return self._box()

    def preview(self, limit: int = 10) -> str:
        """The first `limit` rows drawn as a table, with the footer `__repr__` prints for the relation."""
        count = _row_count("preview", "limit", limit)
        # One read under the renderer's row cap, as __repr__ reads, kept as rows so that a source that reads once, an
        # Arrow stream, is read once. From those rows the head alone, so the renderer never splits it into a
        # head and a tail, and the footer the renderer writes for all of them, so the count is the relation's.
        rows = _held(self._con, self.limit(_BOX_LIMIT)._submit(), self.alias)
        head = self._shown(rows).limit(count)._submit().to_box(max_rows=max(count, 1))
        whole = rows._submit().to_box(max_rows=count)
        footer = _box_footer(whole)
        if not footer:
            return head
        # The head's own footer, which an empty box carries too, gives way to the one for all the rows.
        box = head.split("\n")
        closing = max(index for index, line in enumerate(box) if line.startswith("└"))
        return "\n".join([*box[: closing + 1], *footer])

    def _repr_mimebundle_(self, include: object = None, exclude: object = None) -> dict[str, str]:
        """The first rows as text and as an HTML table, for notebooks, from one execution."""
        # The rows __repr__ reads, so the footer of both halves is the relation's own count and not this fetch's.
        result = self._shown(self.limit(_BOX_LIMIT))._submit()
        text = result.to_box(max_rows=_HTML_ROWS)
        rows = result.fetchall()
        names, types = self._fresh()
        heading = "".join(
            f"<th>{html.escape(n)}<br><small>{html.escape(str(t))}</small></th>"
            for n, t in zip(names, types, strict=True)
        )
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape('NULL' if v is None else str(v))}</td>" for v in row) + "</tr>"
            for row in rows[:_HTML_ROWS]
        )
        total = f"{len(rows)} rows" if len(rows) < _BOX_LIMIT else f"{_BOX_LIMIT} rows or more"
        shown = f"<p><small>{min(len(rows), _HTML_ROWS)} of {total}</small></p>" if len(rows) > _HTML_ROWS else ""
        # The project styles no table of its own, so the class is the one notebooks already style for data frames.
        table = f'<table class="dataframe"><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table>{shown}'
        return {"text/plain": text, "text/html": table}

    def show(
        self,
        *,
        max_width: int | None = None,
        max_rows: int | None = None,
        max_col_width: int | None = None,
        null_value: str | None = None,
        render_mode: object = None,
    ) -> None:
        """Display a summary of the data."""
        print(
            self._box(
                max_width=max_width,
                max_rows=max_rows,
                max_col_width=max_col_width,
                null_value=null_value,
                render_mode=render_mode,
            )
        )

    def explain(
        self, type: object = _duckdb.ExplainType.STANDARD, format: str | None = None, *, analyze: bool = False
    ) -> str:
        """The plan the engine runs for this relation, as text; under Jupyter the default is shown as HTML."""
        kind = fold_name(str(getattr(type, "name", type)))
        analyzed = analyze or kind == "analyze"
        keyword = "EXPLAIN ANALYZE" if analyzed else "EXPLAIN"
        notebook = format is None and not analyzed and bool(getattr(_duckdb, "__jupyter__", False))
        chosen = "html" if notebook else format
        if chosen is not None:
            keyword += f" (FORMAT {chosen})"
        sql, values = _render(self, wrap_text=True)
        rows = _seam.submit(self._con, f"{keyword} {sql}", values).fetchall()
        if notebook:
            _display_html([str(row[1]) for row in rows])
            return ""
        lines = []
        for row in rows:
            cells = ["NULL" if v is None else str(v).replace("\0", "\\0") for v in row[1:]]
            lines.append("\t".join(cells) + "\n")
        return "".join(lines)

    def materialize(self) -> DuckDBPyRelation:
        """Execute once and return a relation over the rows, held in client memory."""
        handle = _seam.register_collection(self._con, self._submit())
        return DuckDBPyRelation._make(self._con, Held(handle.name), alias=self.alias, handles=(handle,))

    def query(self, virtual_table_name: str, sql_query: str) -> DuckDBPyRelation | None:
        """Run the given SQL query in sql_query on the view named virtual_table_name that refers to the relation object.

        The name is this query's own: it is a common table expression over the relation, and a catalog entry for the
        length of the call, so a table function or a query that defines the name itself can bind against it. Nothing
        of it is left afterwards. A statement that does not only read, which includes a SELECT whose common table
        expression writes, runs at the call and the relation holds what it returned.
        """
        _ensure_open(self._con)
        text = _strip_semicolons(sql_query)
        if _seam.statement_count(self._con, text) != 1:
            raise _error(_duckdb.InvalidInputException, "'DuckDBPyRelation.query' only accepts a single statement")
        if _markers(text):
            message = "Unexpected prepared parameter. This type of statement can't be prepared!"
            raise _error(_duckdb.BinderException, message)
        shown = _SHOW_PRAGMA.match(text)
        if shown is not None:
            literal = shown.group(1).replace("''", "'")
            parts = _split_qualified(literal)
            if not all(parts):
                raise _error(_duckdb.ParserException, f"Zero-length identifier in qualified name! (input: {literal})")
            text = "DESCRIBE " + ".".join(quote(p) for p in parts)
        _refresh(self)
        with _reading_as(self, virtual_table_name):
            _, last, kind, _ = _seam.prepare_script_parsed(self._con, text, None)
            last = _strip_semicolons(last) if last else text
            if kind != "SELECT" or _seam.bind(self._con, last).writes:
                # A statement that writes runs at the call and the relation holds what it returned.
                return _run_now(self._con, last, None, "", None)
            step = QueryOver(virtual_table_name, last)
        try:
            # Outside the call's catalog entry, so a statement that binds only while the entry is there cannot
            # become a relation that reads after it is gone.
            return self._derive(step)
        except _duckdb.Error:
            with _reading_as(self, virtual_table_name):
                # Only a statement that binds with the entry and not without it runs here, holding its rows as a
                # statement that writes does; any other mistake raises as it came, and binding executes nothing, so
                # an open transaction stays usable.
                _seam.bind(self._con, last)
                return _run_now(self._con, last, None, "", None)

    def _query_view_body(self) -> str:
        """The relation as the body of the temporary view `query()` puts its name on for the length of the call."""
        return _render(self, literals=True, wrap_text=True)[0]

    def map(self, map_function: Callable[..., Any], *, schema: object = None) -> DuckDBPyRelation:
        """Calls the passed function on the relation."""
        body, _ = _render(self, literals=True, wrap_text=True)
        handle = _seam.register_map(self._con, map_function, schema, body)
        # The input's own registry handles, not the input relation: a relation holds its connection, and a mapped
        # relation registered as a source would close a cycle through C++ members that keeps the database open.
        kept: tuple[object, ...] = (handle, *self._sources())
        return DuckDBPyRelation._make(self._con, Mapped(handle.name), alias=_unnamed(), handles=kept)

    def _write(self, before: str, after: str = "") -> _seam.Result:
        """Run a statement built around this relation's query, its parameter values bound."""
        sql, values = _render(self, wrap_text=True)
        result = _seam.submit(self._con, f"{before}{sql}{after}", values)
        result.drain()
        return result

    def create(self, table_name: str) -> None:
        """Creates a new table named table_name with the contents of the relation object.

        The name is table, schema.table or catalog.schema.table, and a part may be double quoted; a name of more than
        three parts needs SQL (CREATE TABLE AS) for now.
        """
        target = _write_target(self._con, table_name, "CREATE TABLE {} AS")
        self._write(f"CREATE TABLE {target} AS ")

    to_table = create

    def insert_into(self, table_name: str) -> None:
        """Inserts the relation object into an existing table named table_name.

        The name is table, schema.table or catalog.schema.table, and a part may be double quoted; a name of more than
        three parts needs SQL (INSERT INTO) for now.
        """
        target = _write_target(self._con, table_name, "INSERT INTO {}")
        self._write(f"INSERT INTO {target} ")

    def insert(self, values: object) -> None:
        """Inserts the given values into the relation."""
        step = self._step
        if not self._reads_a_table():
            raise _error(
                _duckdb.InvalidInputException, "'DuckDBPyRelation.insert' can only be used on a table relation"
            )
        assert isinstance(step, Table)
        row = list(values) if isinstance(values, (list, tuple)) else [values]
        markers = ", ".join("?" for _ in row)
        _seam.submit(self._con, f"INSERT INTO {step.name()} VALUES ({markers})", row).drain()

    def update(self, set: object, *, condition: object = None) -> None:
        """Update the given relation with the provided expressions."""
        where = "" if condition is None else f" WHERE {_condition_text(self._con, condition, 'update condition')}"
        if not isinstance(set, Mapping):
            message = "Please provide 'set' as a dictionary of column name to Expression"
            raise _error(_duckdb.InvalidInputException, message)
        if not set:
            raise _error(_duckdb.InvalidInputException, "Please provide at least one set expression")
        assignments: builtins.list[str] = []
        for name, value in set.items():
            if not isinstance(name, str):
                message = "Please provide the column name as the key of the dictionary"
                raise _error(_duckdb.InvalidInputException, message)
            if isinstance(value, str):
                message = (
                    f"update() takes an expression or a Python value for {name!r}, not the string {value!r}; use "
                    f"col({value!r}) for a column or lit({value!r}) for the text"
                )
                raise TypeError(message)
            try:
                text = _expression_text(value)
            except _duckdb.InvalidInputException as error:
                if isinstance(value, PendingCase):
                    raise
                message = f"Please provide an object of type Expression as the value, not {type(value)}"
                raise _error(_duckdb.InvalidInputException, message) from error
            assignments.append(f"{quote(name)} = {text}")
        step = self._step
        if not self._reads_a_table():
            raise _error(_duckdb.InvalidInputException, "UPDATE can only be used on base tables")
        assert isinstance(step, Table)
        _seam.submit(self._con, f"UPDATE {step.name()} SET {', '.join(assignments)}{where}", None).drain()

    def create_view(self, view_name: str, replace: bool = True) -> DuckDBPyRelation:  # noqa: FBT001
        """Creates a view named view_name that refers to the relation object."""
        body, _ = _render(self, literals=True, wrap_text=True)
        # The view holds the table reference of every registry name it reads, so its catalog entry owns its sources
        # and its transaction decides when they go; a Python object in a database file is refused there.
        _seam.create_view(self._con, view_name, body, replace=replace)
        return self

    to_view = create_view

    def write_csv(
        self,
        file_name: str,
        *,
        sep: str | None = None,
        na_rep: str | None = None,
        header: bool | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        quoting: str | int | None = None,
        encoding: str | None = None,
        compression: str | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: builtins.list[str] | None = None,
        write_partition_columns: bool | None = None,
    ) -> None:
        """Write the relation object to a CSV file in 'file_name'."""
        options: builtins.list[tuple[str, str]] = [("FORMAT", "csv")]
        _string_option(options, "to_csv", "sep", sep, "DELIMITER")
        _string_option(options, "to_csv", "na_rep", na_rep, "NULL")
        _bool_option(options, "to_csv", "header", header, "HEADER")
        _string_option(options, "to_csv", "quotechar", quotechar, "QUOTE")
        _string_option(options, "to_csv", "escapechar", escapechar, "ESCAPE")
        _string_option(options, "to_csv", "date_format", date_format, "DATEFORMAT")
        _string_option(options, "to_csv", "timestamp_format", timestamp_format, "TIMESTAMPFORMAT")
        if quoting is not None:
            if isinstance(quoting, str):
                if fold_name(quoting) not in ("force", "all"):
                    message = "to_csv 'quoting' supported options are ALL or FORCE (both set FORCE_QUOTE=True)"
                    raise _error(_duckdb.InvalidInputException, message)
            elif isinstance(quoting, int) and not isinstance(quoting, bool):
                if quoting != 1:
                    message = "Only csv.QUOTE_ALL is a supported option for 'quoting' currently"
                    raise _error(_duckdb.InvalidInputException, message)
            else:
                message = "to_csv only accepts 'quoting' as a string or a constant from the 'csv' package"
                raise _error(_duckdb.InvalidInputException, message)
            options.append(("FORCE_QUOTE", "*"))
        if encoding is not None:
            if not isinstance(encoding, str):
                raise _error(_duckdb.InvalidInputException, "to_csv only accepts 'encoding' as a string")
            if fold_name(encoding) not in ("utf-8", "utf8"):
                raise _error(_duckdb.InvalidInputException, "The only supported encoding option is 'UTF8")
        _string_option(options, "to_csv", "compression", compression, "COMPRESSION")
        refuse = _overwrite_option(options, "to_csv", overwrite)
        _bool_option(options, "to_csv", "per_thread_output", per_thread_output, "PER_THREAD_OUTPUT")
        _bool_option(options, "to_csv", "use_tmp_file", use_tmp_file, "USE_TMP_FILE")
        _partition_option(options, "to_csv", partition_by)
        _bool_option(options, "to_csv", "write_partition_columns", write_partition_columns, "WRITE_PARTITION_COLUMNS")
        self._copy(file_name, options, refuse=refuse)

    to_csv = write_csv

    def write_parquet(
        self,
        file_name: str,
        *,
        compression: str | None = None,
        field_ids: object = None,
        row_group_size_bytes: int | str | None = None,
        row_group_size: int | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: builtins.list[str] | None = None,
        write_partition_columns: bool | None = None,
        append: bool | None = None,
        filename_pattern: str | None = None,
        file_size_bytes: str | int | None = None,
    ) -> None:
        """Write the relation object to a Parquet file in 'file_name'."""
        options: builtins.list[tuple[str, str]] = [("FORMAT", "parquet")]
        _string_option(options, "to_parquet", "compression", compression, "COMPRESSION")
        if field_ids is not None:
            if isinstance(field_ids, Mapping):
                options.append(("FIELD_IDS", _field_ids(field_ids)))
            elif isinstance(field_ids, str):
                options.append(("FIELD_IDS", _option_literal(field_ids)))
            else:
                message = "to_parquet only accepts 'field_ids' as a dictionary or 'auto'"
                raise _error(_duckdb.InvalidInputException, message)
        if row_group_size_bytes is not None:
            if isinstance(row_group_size_bytes, bool) or not isinstance(row_group_size_bytes, (int, str)):
                message = "to_parquet only accepts 'row_group_size_bytes' as an integer or 'auto' string"
                raise _error(_duckdb.InvalidInputException, message)
            options.append(("ROW_GROUP_SIZE_BYTES", _option_literal(row_group_size_bytes)))
        if row_group_size is not None:
            if isinstance(row_group_size, bool) or not isinstance(row_group_size, int):
                raise _error(_duckdb.InvalidInputException, "to_parquet only accepts 'row_group_size' as an integer")
            options.append(("ROW_GROUP_SIZE", str(row_group_size)))
        _partition_option(options, "to_parquet", partition_by)
        _bool_option(
            options, "to_parquet", "write_partition_columns", write_partition_columns, "WRITE_PARTITION_COLUMNS"
        )
        _bool_option(options, "to_parquet", "append", append, "APPEND")
        refuse = _overwrite_option(options, "to_parquet", overwrite)
        _bool_option(options, "to_parquet", "per_thread_output", per_thread_output, "PER_THREAD_OUTPUT")
        _bool_option(options, "to_parquet", "use_tmp_file", use_tmp_file, "USE_TMP_FILE")
        _string_option(options, "to_parquet", "filename_pattern", filename_pattern, "FILENAME_PATTERN")
        if file_size_bytes is not None:
            if isinstance(file_size_bytes, bool) or not isinstance(file_size_bytes, (int, str)):
                message = "to_parquet only accepts 'file_size_bytes' as an integer or string"
                raise _error(_duckdb.InvalidInputException, message)
            options.append(("FILE_SIZE_BYTES", _option_literal(file_size_bytes)))
        self._copy(file_name, options, refuse=refuse)

    to_parquet = write_parquet

    def _copy(self, file_name: str, options: builtins.list[tuple[str, str]], *, refuse: bool) -> None:
        names = {name for name, _ in options}
        single_file = not names & {"PARTITION_BY", "FILE_SIZE_BYTES", "APPEND"} and (
            ("PER_THREAD_OUTPUT", "true") not in options
        )
        if refuse and single_file:
            exists = _seam.submit(self._con, "SELECT count(*) FROM glob(?)", [file_name]).fetchall()[0][0]
            if exists:
                message = f'Cannot write to "{file_name}" - it exists! Pass overwrite=True to overwrite the file'
                raise _error(_duckdb.IOException, message)
        listed = ", ".join(f"{name} {value}" for name, value in options)
        target = "'" + str(file_name).replace("'", "''") + "'"
        self._write("COPY (", f") TO {target} ({listed})")


_RELATION_DUNDERS = frozenset(
    {
        "__len__",
        "__str__",
        "__repr__",
        "__contains__",
        "__getattr__",
        "__getitem__",
        "__arrow_c_stream__",
        "_repr_mimebundle_",
    }
)


def _box_footer(box: str) -> list[str]:
    """The lines a drawn table prints under its closing line, such as its row count."""
    lines = box.split("\n")
    closing = max(index for index, line in enumerate(lines) if line.startswith("└"))
    return [line for line in lines[closing + 1 :] if line.strip()]


def _trimmed(error: BaseException) -> BaseException:
    """The engine's own exception, with what the engine put on it, but without the layer's frames.

    The answer belongs to the line the user wrote, so the traceback is cut and an exception the layer was handling when
    the engine raised is not shown; a cause the engine or the user set is kept.
    """
    error.__suppress_context__ = True
    return error.with_traceback(None)


def _closed_as_relation(function: Callable[..., Any]) -> Callable[..., Any]:
    """Report a closed connection from a relation method with the text the relation reported it with."""

    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        try:
            return function(*args, **kwargs)
        except _duckdb.ConnectionException as error:
            if "already closed" not in str(error):
                raise _trimmed(error) from error.__cause__
            message = "Connection Error: Connection has already been closed"
            raise _duckdb.ConnectionException(message) from None
        except _duckdb.Error as error:
            raise _trimmed(error) from error.__cause__

    return guarded


for _name, _member in list(vars(DuckDBPyRelation).items()):
    if _name.startswith("_") and _name not in _RELATION_DUNDERS:
        continue
    if isinstance(_member, property) and _member.fget is not None:
        setattr(DuckDBPyRelation, _name, property(_closed_as_relation(_member.fget), doc=_member.__doc__))
    elif callable(_member):
        setattr(DuckDBPyRelation, _name, _closed_as_relation(_member))
del _name, _member


class GroupedRelation:
    """A relation with group keys chosen, waiting for aggregates."""

    def __init__(self, relation: DuckDBPyRelation, keys: tuple[object, ...]) -> None:
        self._relation = relation
        self._keys = keys

    def agg(self, *aggregates: object) -> DuckDBPyRelation:
        """Apply aggregates to each group; the group keys come first.

        The arguments read as `aggregate` reads them: one string is expression text, several are column names.
        """
        relation = self._relation
        con = relation._con
        if not aggregates:
            msg = "agg() needs at least one aggregate"
            raise TypeError(msg)
        columns = relation.columns
        for aggregate in aggregates:
            _warn_on_literal_names(aggregate, columns)
        keys = [_select_item(con, key) for key in self._keys]
        items = ", ".join([*keys, _items_of(con, aggregates)])
        groups = ", ".join(str(i) for i in range(1, len(keys) + 1)) or None
        return relation._derive(Aggregate(items, groups))


def _unnamed() -> str:
    return f"unnamed_relation_{secrets.token_hex(8)}"


@contextlib.contextmanager
def _reading_as(relation: DuckDBPyRelation, name: str) -> Iterator[None]:
    """The relation readable under `name` in the catalog for the length of a call, when the name is free.

    A statement of `query()` that looks the name up in the catalog, a table function or a common table expression of
    its own name, binds against it; the entry goes when the call returns, so the name is visible nowhere else. A name
    the catalog already holds is left alone, and the statement binds against what is there.
    """
    quoted = quote(name)
    # The catalog says whether the name is taken, so a view whose body cannot bind raises with what is missing
    # instead of being taken for a name already in use, which reported the user's own query name.
    created = _seam.table_kind(relation._con, name) == "none"
    if created:
        _seam.submit(relation._con, f"CREATE TEMPORARY VIEW {quoted} AS {relation._query_view_body()}", None).drain()
    try:
        yield
    finally:
        if created:
            _seam.submit(relation._con, f"DROP VIEW IF EXISTS {quoted}", None).drain()


_TREE_SCRIPT = """
<script>
function toggleDisplay(button) {
    const parentLi = button.closest('li');
    const nestedUl = parentLi.querySelector('ul');
    if (nestedUl) {
        const currentDisplay = getComputedStyle(nestedUl).getPropertyValue('display');
        if (currentDisplay === 'none') {
            nestedUl.classList.toggle('hidden');
            button.textContent = '-';
        } else {
            nestedUl.classList.toggle('hidden');
            button.textContent = '+';
        }
    }
}

function updateTreeHeight(tfTree) {
    if (!tfTree) {
        return;
    }
    const closestElement = tfTree.closest('.lm-Widget.jp-OutputArea.jp-Cell-outputArea');
    if (!closestElement) {
        return;
    }
    const height = getComputedStyle(closestElement).getPropertyValue('height');
    tfTree.style.height = height;
}

function resizeTFTree() {
    const tfTrees = document.querySelectorAll('.tf-tree');
    tfTrees.forEach(tfTree => {
        if (tfTree) {
            const jupyterViewPort = tfTree.closest('.lm-Widget.jp-OutputArea.jp-Cell-outputArea');
            if (jupyterViewPort) {
                const resizeObserver = new ResizeObserver(() => {
                    updateTreeHeight(tfTree);
                });
                resizeObserver.observe(jupyterViewPort);
            }
        }
    });
}

resizeTFTree();
</script>
"""


def _display_html(pages: list[str]) -> None:
    import IPython.display

    # IPython ships no annotations, so the two callables are typed here instead of ignoring each call.
    show: Callable[[object], None] = IPython.display.display
    html: Callable[[str], object] = IPython.display.HTML
    for page in pages:
        show(html(page))
    show(html(_TREE_SCRIPT))


def _ensure_open(con: _duckdb.DuckDBPyConnection) -> None:
    """Refuse a closed connection before anything else is checked, as the relation did."""
    # The catalog version is read without a statement and raises the closed connection error, for none of a bind's
    # parse, claim and bind.
    _seam.catalog_version(con)


def _write_target(con: _duckdb.DuckDBPyConnection, name: str, sql_form: str) -> str:
    """The table a write names, as catalog, schema and table parts; a longer path is refused, as today."""
    _ensure_open(con)
    parts = _split_qualified(name) if name else [""]
    if any(not part for part in parts):
        raise _error(_duckdb.ParserException, f"Zero-length identifier in qualified name! (input: {name})")
    if len(parts) > 3:
        rendered = ".".join(_optional_quote(p) for p in parts)
        message = (
            f"The relation API cannot address a nested schema path such as {rendered} yet; use SQL "
            f"({sql_form.format(rendered)}) instead"
        )
        raise _error(_duckdb.NotImplementedException, message)
    return ".".join(quote(p) for p in parts)


def _string_option(options: list[tuple[str, str]], method: str, argument: str, value: object, option: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise _error(_duckdb.InvalidInputException, f"{method} only accepts '{argument}' as a string")
    options.append((option, _option_literal(value)))


def _bool_option(options: list[tuple[str, str]], method: str, argument: str, value: object, option: str) -> None:
    if value is None:
        return
    if not isinstance(value, bool):
        raise _error(_duckdb.InvalidInputException, f"{method} only accepts '{argument}' as a boolean")
    options.append((option, _option_literal(value)))


def _overwrite_option(options: list[tuple[str, str]], method: str, overwrite: object) -> bool:
    """Write the overwrite option; true when an existing single file must be refused, which COPY would replace."""
    if overwrite is None:
        return False
    if not isinstance(overwrite, bool):
        raise _error(_duckdb.InvalidInputException, f"{method} only accepts 'overwrite' as a boolean")
    options.append(("OVERWRITE_OR_IGNORE", _option_literal(overwrite)))
    return not overwrite


def _partition_option(options: list[tuple[str, str]], method: str, partition_by: object) -> None:
    if partition_by is None:
        return
    if not isinstance(partition_by, list) or not all(isinstance(p, str) for p in partition_by):
        raise _error(_duckdb.InvalidInputException, f"{method} only accepts 'partition_by' as a list of strings")
    options.append(("PARTITION_BY", "(" + ", ".join(quote(p) for p in partition_by) + ")"))


def _field_ids(field_ids: Mapping[object, object]) -> str:
    """A field id dictionary as a struct literal of integers and nested dictionaries."""
    entries: list[str] = []
    for key, value in field_ids.items():
        if not isinstance(key, str):
            message = "NestedDictToStruct only accepts a dictionary with string keys"
            raise _error(_duckdb.InvalidInputException, message)
        if isinstance(value, Mapping):
            entries.append(f"{_option_literal(key)}: {_field_ids(value)}")
        elif isinstance(value, int) and not isinstance(value, bool):
            entries.append(f"{_option_literal(key)}: {int(value)}")
        else:
            message = "NestedDictToStruct only accepts a dictionary with integer values or nested dictionaries"
            raise _error(_duckdb.InvalidInputException, message)
    return "{" + ", ".join(entries) + "}"


def _quantiles(q: object) -> str:
    if isinstance(q, float):
        return _float_text(q)
    if isinstance(q, list):
        return "[" + ",".join(f"{float(v):.6f}" for v in q) + "]"
    raise _error(_duckdb.InvalidTypeException, "Unsupported type for quantile")


def _offset_parameters(offset: int, default_value: str) -> str:
    text = str(int(offset)) if int(offset) != 0 else ""
    if default_value:
        text += "," + default_value
    return text


_JOIN_KINDS = {
    "inner": ("INNER", True),
    "left": ("LEFT", True),
    "right": ("RIGHT", True),
    "outer": ("FULL OUTER", True),
    "full": ("FULL OUTER", True),
    "semi": ("SEMI", True),
    "anti": ("ANTI", True),
    "cross": ("CROSS", False),
    "positional": ("POSITIONAL", False),
    "natural": ("NATURAL", False),
    "asof": ("ASOF", True),
}


class _Side:
    """One side of a join, for a condition written as a function: `side["x"]` is that side's column."""

    def __init__(self, alias: str) -> None:
        self._alias = alias

    def __getitem__(self, name: str) -> Any:  # noqa: ANN401
        return _api().ColumnExpression(self._alias, name)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        if name.startswith("__"):
            raise AttributeError(name)
        return self[name]


def _suffixed(left: tuple[str, list[str]], right: tuple[str, list[str]], merged: tuple[str, ...], suffix: str) -> str:
    """A join's select list with the suffix on right side names the left has; a merged name keeps its one copy.

    Each side is one FROM item, by its alias and the names it exposes there, which the engine gives. The merged
    names are a USING list or a natural join's shared names, which the bare star gives once.
    """
    (left_alias, left_names), (right_alias, right_names) = left, right
    taken = {fold_name(name) for name in left_names}
    keys = {fold_name(name) for name in merged}
    shared = [name for name in right_names if fold_name(name) in taken and fold_name(name) not in keys]
    if not shared:
        return "*"
    renaming = {fold_name(name) for name in shared}
    kept = {fold_name(name) for name in right_names} - renaming - keys
    for name in shared:
        # The option exists to keep a join's names apart, so a suffixed name another output column has is refused.
        renamed = name + suffix
        side = "left" if fold_name(renamed) in taken else "right" if fold_name(renamed) in kept else None
        if side is not None:
            message = f"join(suffix={suffix!r}) would name the right side's {name!r} {renamed!r}, which the {side} has"
            raise _error(_duckdb.InvalidInputException, message)
    if merged:
        # Only the bare star merges a USING key, so the right copies are renamed through it by their qualified name.
        renamed = ", ".join(f"{quote(right_alias)}.{quote(n)} AS {quote(n + suffix)}" for n in shared)
        return f"* RENAME ({renamed})"
    renamed = ", ".join(f"{quote(n)} AS {quote(n + suffix)}" for n in shared)
    return f"{quote(left_alias)}.*, {quote(right_alias)}.* RENAME ({renamed})"


def _boundary_names(rel: DuckDBPyRelation) -> list[str]:
    """The names a relation exposes as a FROM item, asked of the binder over a typed stand-in of it."""
    reading = 'SELECT * FROM "__duckdb_side"'

    def real() -> str:
        return _assemble([("__duckdb_side", _render(rel, literals=True)[0])], reading)

    stub = _stub(*rel._schema)
    if stub is None:
        return list(_seam.bind(rel._con, real()).names)
    stubs = [("__duckdb_side", stub)]
    return list(_bind_with_stand_ins(rel, _assemble(stubs, reading), stubs, real).names)


def _using_columns(texts: list[str]) -> tuple[str, ...] | None:
    """The USING list a condition means: the engine reads plain column names, or a list of conditions, as USING."""
    references = [_seam.column_reference(text) for text in texts]
    if len(texts) == 1 and references[0] is None:
        return None
    columns = []
    for parts in references:
        if parts is None:
            raise _error(_duckdb.ParserException, "Expected a single expression as join condition")
        if len(parts) > 1:
            raise _error(_duckdb.ParserException, "Expected unqualified column for column in USING clause")
        columns.append(parts[0])
    return tuple(columns)


def _join_condition(
    con: _duckdb.DuckDBPyConnection, condition: object, left: str, right: str
) -> tuple[str | None, tuple[str, ...]]:
    if condition is None:
        return None, ()
    if isinstance(condition, str):
        count = _seam.check_text(con, _encodable(condition), "expressions")
        using = _using_columns(_split_list(condition))
        if using is not None:
            return None, using
        if count != 1:
            raise _error(_duckdb.ParserException, "Expected a single expression as join condition")
        return _spliced(condition), ()
    if isinstance(condition, (list, tuple)):
        names = []
        for item in condition:
            if not isinstance(item, str):
                message = f"Using clause should be a list of strings, not {type(item)}"
                raise _error(_duckdb.InvalidInputException, message)
            names.append(item)
        if not names:
            message = "Please provide at least one string in the condition to create a USING clause"
            raise _error(_duckdb.InvalidInputException, message)
        return None, tuple(names)
    if callable(condition) and not _is_expression(condition):
        built = condition(_Side(left), _Side(right))
        return _expression_text(built), ()
    text = _expression_text(condition)
    using = _using_columns([text])
    return (None, using) if using is not None else (text, ())


def _refuse_shared_names(inputs: tuple[DuckDBPyRelation, ...]) -> None:
    """Refuse inputs whose parameters one statement cannot bind: a name on two sides, or named beside positional."""
    # Keyed on the node that declares the name, not on which input it came through, so a relation combined with
    # itself carries one name once rather than twice. A parameter name is an identifier, which the engine
    # folds, so `$P` and `$p` are one name.
    seen: dict[str, tuple[int, str]] = {}
    kinds: set[bool] = set()
    for relation in inputs:
        for node in _tree(relation):
            if node._params is None:
                continue
            kinds.add(isinstance(node._params, dict))
            if isinstance(node._params, dict):
                for name in node._params:
                    owner, spelled = seen.setdefault(fold_name(name), (id(node), name))
                    if owner != id(node):
                        if spelled == name:
                            message = f"Both relations have a parameter named ${name}; rename one of them"
                        else:
                            message = (
                                f"The relations have parameters named ${spelled} and ${name}, which SQL reads as one "
                                "name; rename one of them"
                            )
                        raise _error(_duckdb.InvalidInputException, message)
    if len(kinds) > 1:
        message = "A relation with named parameters cannot be combined with one with positional parameters"
        raise _error(_duckdb.InvalidInputException, message)


def _copied_value(value: object) -> object:
    """A container rebuilt, anything else kept: a value the engine converts, such as a duckdb.Value, is not copied."""
    if type(value) is list:
        return [_copied_value(item) for item in value]
    if type(value) is tuple:
        return tuple(_copied_value(item) for item in value)
    if type(value) is dict:
        return {key: _copied_value(item) for key, item in value.items()}
    return value


def _copied(params: object) -> object:
    """The parameter values as of the call, so a list changed later does not change what the relation reads."""
    if isinstance(params, Mapping):
        return {key: _copied_value(value) for key, value in params.items()}
    return [_copied_value(value) for value in params]  # type: ignore[attr-defined]


def _has_values(params: object) -> bool:
    """Whether parameter values were given; an object without a length counts as given, as its length is unknown."""
    if params is None:
        return False
    if not hasattr(type(params), "__len__"):
        return True
    return len(params) > 0  # type: ignore[arg-type]


def _reading_source(
    con: _duckdb.DuckDBPyConnection,
    text: str,
    markers: tuple[_Marker, ...],
    values: list[Any] | dict[str, Any] | None,
    alias: str,
    parsed: _duckdb.Statement | None = None,
) -> DuckDBPyRelation | None:
    """A lazy relation over a query that reads, or None when the binder says it must run now.

    A SELECT the engine says changes a database, through a common table expression that writes or a sequence's
    nextval, runs at the call, because a relation over it would write again on every read.
    """
    signature = _seam.bind(con, parsed if parsed is not None else text, capture=True, values=values)
    if signature.return_type != "query_result" or signature.deferred or signature.writes:
        return None
    return DuckDBPyRelation._make(
        con,
        Sql(text, markers),
        alias=alias or _unnamed(),
        params=values,
        handles=tuple(handle for _, handle in signature.captured),
        captured=tuple((name, handle.name) for name, handle in signature.captured),
        schema=(signature.names, signature.types),
    )


def _held(
    con: _duckdb.DuckDBPyConnection, result: object, alias: str, return_type: str | None = None
) -> DuckDBPyRelation:
    handle = _seam.register_collection(con, result)  # type: ignore[arg-type]
    return DuckDBPyRelation._make(
        con, Held(handle.name), alias=alias or _unnamed(), handles=(handle,), return_type=return_type
    )


def _run_now(
    con: _duckdb.DuckDBPyConnection, text: str, params: object, alias: str, kept: object
) -> DuckDBPyRelation | None:
    """Run a statement at the call and keep what it returned.

    A statement that returns no rows leaves the rows of the latest statement before it that did, as a script does;
    with neither there is no relation.
    """
    given = _has_values(params)
    # The statement runs, and fails as execution fails whatever ran before it. What it returns is the
    # engine's own report from the run, which a layer above reads to tell a command from rows of its own;
    # the shape of what came back cannot, since RETURNING can name a column Count, and a deferred group's bind does
    # not know it until the group has run.
    result = _seam.submit(con, text, params if given else None)  # type: ignore[arg-type]
    return_type = result.return_type
    returns_rows = kept is None or return_type == "query_result"
    earlier = _held(con, kept, alias) if kept is not None else None
    if not result.description:
        if given:
            message = "Parameters were given, but the last statement of the query takes none"
            raise _error(_duckdb.InvalidInputException, message)
        return earlier
    if not returns_rows:
        result.drain()
        return earlier
    return _held(con, result, alias, return_type)


def sql(
    con: _duckdb.DuckDBPyConnection, query: object, alias: str = "", params: object = None
) -> DuckDBPyRelation | None:
    """A relation over the last statement of a query; the statements before it run now.

    The last statement is lazy when it parses as a SELECT and the binder describes it; any other statement runs now
    and the relation holds what it returned.
    """
    given = _has_values(params)
    params = _copied(params) if given else None
    _refuse_one_name_twice(params)
    if isinstance(query, str):
        _encodable(query)
    kept, text, kind, parsed = _seam.prepare_script_parsed(con, query, params)  # type: ignore[arg-type]
    if text is None:
        if given:
            message = "Parameters were given, but the last statement of the query takes none"
            raise _error(_duckdb.InvalidInputException, message)
        return _held(con, kept, alias) if kept is not None else None
    text = _strip_semicolons(text)
    if kind == "SELECT":
        markers = _markers(text)
        values: list[Any] | dict[str, Any] | None = None
        if given:
            values = _parameter_values(markers, params)
        elif markers:
            _parameter_values(markers, [] if markers[0].positional else {})
        relation = _reading_source(con, text, markers, values, alias, parsed)
        if relation is not None:
            return relation
    return _run_now(con, text, params, alias, kept)


def _name_parts(name: str) -> tuple[str, ...]:
    """The parts of a dotted name, or the whole name as one part when it cannot be one (a path such as ./x.csv)."""
    try:
        parts = tuple(_split_qualified(name))
    except _duckdb.ParserException:
        return (name,)
    return (name,) if any(not part for part in parts) else parts


def _catalog_source(
    con: _duckdb.DuckDBPyConnection, parts: tuple[str, ...], step_class: type[Table], *, qualified: bool = True
) -> DuckDBPyRelation:
    step = step_class(parts)
    text = f"SELECT * FROM {step.from_item()}"
    signature = _seam.bind(con, text, capture=True)
    schema = (signature.names, signature.types)
    alias = parts[-1]
    if signature.captured:
        return DuckDBPyRelation._make(
            con,
            Sql(text),
            alias=alias,
            handles=tuple(handle for _, handle in signature.captured),
            captured=tuple((name, handle.name) for name, handle in signature.captured),
            schema=schema,
            return_type=signature.return_type,
        )
    if not qualified:
        step = step_class(parts, base=False)
    return DuckDBPyRelation._make(con, step, alias=alias, schema=schema, return_type=signature.return_type)


def _catalog_relation(con: _duckdb.DuckDBPyConnection, name: str, step_class: type[Table]) -> DuckDBPyRelation:
    """A relation reading a catalog entry or a file by name.

    A dotted name is qualified when its first part names a schema or a database, as SQL reads it, and is one name
    otherwise.
    """
    parts = _name_parts(name)
    if len(parts) > 1 and _seam.name_scope(con, parts[0]) == "none":
        relation = _catalog_source(con, (name,), step_class, qualified=False)
        relation._alias = name
        return relation
    return _catalog_source(con, parts, step_class)


def table(con: _duckdb.DuckDBPyConnection, table_name: str) -> DuckDBPyRelation:
    """A relation reading a table, a view or a file by name."""
    return _catalog_relation(con, table_name, Table)


def view(con: _duckdb.DuckDBPyConnection, view_name: str) -> DuckDBPyRelation:
    """A relation reading a view by name."""
    return _catalog_relation(con, view_name, View)


def _value_row(args: tuple[object, ...]) -> tuple[str, ...]:
    if not args:
        raise _error(_duckdb.InvalidInputException, "Please provide a non-empty tuple")
    return tuple(_expression_text(arg) for arg in args)


def values(con: _duckdb.DuckDBPyConnection, *args: object) -> DuckDBPyRelation:
    """A relation over rows given here: one list of values, expressions, or tuples of expressions."""
    if not args:
        raise _error(_duckdb.InvalidInputException, "Could not create a ValueRelation without any inputs")
    held = _value_rows(args)
    if held is not None:
        # Held as rows, each value converted once now, so the Python objects can change afterwards.
        handle = _seam.register_values(con, held)
        return DuckDBPyRelation._make(con, ValueRows(handle.name), alias=_unnamed(), handles=(handle,))
    if len(args) == 1 and isinstance(args[0], list):
        rows: tuple[tuple[str, ...], ...] = (tuple(_value_text(v) for v in args[0]),)
    elif isinstance(args[0], tuple):
        collected: list[tuple[str, ...]] = []
        for arg in args:
            if not isinstance(arg, tuple):
                raise _error(_duckdb.InvalidInputException, f"Expected objects of type tuple, not {type(arg)}")
            row = _value_row(arg)
            if collected and len(row) != len(collected[0]):
                message = (
                    f"Mismatch between length of tuples in input, expected {len(collected[0])} but found {len(row)}"
                )
                raise _error(_duckdb.InvalidInputException, message)
            collected.append(row)
        rows = tuple(collected)
    else:
        rows = (_value_row(args),)
    return DuckDBPyRelation._make(con, Values(rows), alias=_unnamed())


def _value_rows(args: tuple[object, ...]) -> list[tuple[object, ...]] | None:
    """The rows `values()` was given, when every one of them holds only Python values.

    None where one holds an expression, which only a VALUES list in SQL can hold, or anything that is no value.
    """
    if len(args) == 1 and isinstance(args[0], list):
        rows = [tuple(args[0])]
    elif all(isinstance(arg, tuple) for arg in args):
        rows = [cast("tuple[object, ...]", arg) for arg in args]
    else:
        rows = [args]
    for row in rows:
        if not row or any(_is_expression(value) or isinstance(value, PendingCase) for value in row):
            return None
        if not all(_is_value(value) for value in row) or any(_keyed_by_other_than_text(value) for value in row):
            return None
    # A memoryview is the bytes it shows, as a literal's is, where the parameter path reads its raw buffer.
    return [tuple(plain_value(value) for value in row) for row in rows]


def _keyed_by_other_than_text(value: object) -> bool:
    """Whether a dict with a key that is not a string is in `value`: a MAP, which the parameter path makes a STRUCT."""
    if isinstance(value, dict):
        return any(not isinstance(key, str) or _keyed_by_other_than_text(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_keyed_by_other_than_text(item) for item in value)
    return False


def table_function(con: _duckdb.DuckDBPyConnection, name: str, parameters: object = None) -> DuckDBPyRelation:
    """A relation over a table function called with the given parameters."""
    if parameters is None:
        parameters = []
    if isinstance(parameters, (str, bytes)) or not isinstance(parameters, Iterable):
        raise _error(_duckdb.InvalidInputException, "'params' has to be a list of parameters")
    arguments = tuple(_expression_text(p) for p in parameters)
    return DuckDBPyRelation._make(con, TableFunction(name, arguments), alias=name)


def _files(con: _duckdb.DuckDBPyConnection, source: object) -> tuple[str, tuple[object, ...]]:
    """The files argument of a reader, with file like objects put on the connection's in-memory filesystem."""
    handles: list[object] = []

    def one(item: object) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, bytes) or hasattr(item, "__fspath__"):
            return os.fsdecode(item)
        if not (hasattr(item, "read") and hasattr(item, "seek")):
            msg = "Can not read from a non file-like object"
            raise TypeError(msg)
        handle = _seam.register_filelike(con, item)
        handles.append(handle)
        return handle.name

    if isinstance(source, list):
        if not source:
            raise _error(_duckdb.InvalidInputException, "Please provide a non-empty list of paths or file-like objects")
        paths = [one(item) for item in source]
        return _option_literal(paths), tuple(handles)
    return _option_literal(one(source)), tuple(handles)


#: The readers that read the schema out of the data, so a later read could find a different one.
_SNIFFING_READERS = frozenset({"read_csv", "read_csv_auto", "read_json", "read_json_auto"})
#: The reading options the CSV sniffer reports, as the option each is given to the reader by. The formats are part of
#: it because a date the sniffer read as `%d/%m/%Y` does not parse under the default format once detection is off;
#: the list is the call the engine's own `sniff_csv` prompt writes.
_CSV_DIALECT = (
    ("delim", "Delimiter"),
    ("quote", "Quote"),
    ("escape", "Escape"),
    ("new_line", "NewLineDelimiter"),
    ("comment", "Comment"),
    ("skip", "SkipRows"),
    ("header", "HasHeader"),
    ("dateformat", "DateFormat"),
    ("timestampformat", "TimestampFormat"),
)
#: The options that add columns the files do not hold, and the off value that leaves only the physical ones.
_GENERATED_COLUMN_OPTIONS = (("filename", "false"), ("hive_partitioning", "false"))
#: Options that type the columns, and with the naming ones those the pinned `columns` replaces and the engine refuses
#: beside it.
_TYPE_OPTIONS = frozenset({"dtypes", "column_types", "types"})
_SCHEMA_OPTIONS = _TYPE_OPTIONS | {"names", "column_names"}


def _bound_columns(names: Sequence[str], types: Sequence[object]) -> str:
    """The schema as the `columns` option of a reader: a struct of name to type text."""
    listed = ", ".join(f"{_option_literal(n)}: {_option_literal(str(t))}" for n, t in zip(names, types, strict=True))
    return "{" + listed + "}"


def _csv_dialect(con: _duckdb.DuckDBPyConnection, step: Reader) -> builtins.list[tuple[str, str]]:
    """What the sniffer read of a single CSV file, so the reader can be told to stop detecting it."""
    if not step.files.startswith("'"):
        return []
    asked = ", ".join(f"{name} := {value}" for name, value in step.options)
    text = f"SELECT * FROM sniff_csv({step.files}{', ' + asked if asked else ''})"
    try:
        result = _seam.submit(con, text, None)
    except (_duckdb.ResourceInUseException, _duckdb.NotImplementedException):
        # The dialect is read only where sniff_csv answers: it reads one file, so a pattern matching several is not
        # read, and a connection holding an unfinished DB-API result binds but runs nothing. There the reader keeps
        # detecting the dialect with the columns pinned, and both forms read the same rows.
        return []
    row = result.fetchall()[0]
    reported = {str(column[0]): value for column, value in zip(result.description or [], row, strict=True)}
    # The sniffer reports an absent dialect part as the text "(empty)", which the reader takes as an empty string, and
    # a format it did not need, such as the timestamp format of a file without timestamps, as NULL.
    values = {name: "" if value == "(empty)" else value for name, value in reported.items()}
    # A date column read by the lenient ISO 8601 cast has no format, which sniff_csv reports as `%Y-%m-%d`; pinning
    # that would turn the cast strict and refuse a value such as `2024-01-03 00:00:00` the sniffed read accepts.
    if values.get("DateFormat") == "%Y-%m-%d":
        values["DateFormat"] = None
    return [(option, _option_literal(values[field])) for option, field in _CSV_DIALECT if values.get(field) is not None]


def _physical_columns(rel: DuckDBPyRelation, step: Reader) -> int:
    """How many leading columns of the bound schema the files hold, before the ones the reader generates."""
    generated = {name for name, _ in _GENERATED_COLUMN_OPTIONS} | {"hive_types", "hive_types_autocast"}
    kept = [(name, value) for name, value in step.options if name not in generated]
    physical = dataclasses.replace(step, options=(*kept, *_GENERATED_COLUMN_OPTIONS))
    return len(_seam.bind(rel._con, f"SELECT * FROM {physical.from_item()}").names)


#: The `read_json` options that say how to find the objects, which `read_json_objects` takes too.
_JSON_OBJECT_OPTIONS = frozenset({"format", "compression", "maximum_object_size", "ignore_errors"})


def _refuse_renamed_json_keys(rel: DuckDBPyRelation, step: Reader, names: Sequence[str]) -> None:
    """Refuse a JSON schema that names a key the files do not hold, which the pinned `columns` would read as NULL.

    The reader renames a key that differs from an earlier one only in case, `A` beside `a` becoming `A_1`, and a pinned
    call then asks for a key that is not there. Only a name shaped like such a rename is asked about, against
    the keys the files actually hold, because a real key may be spelled the same way.
    """
    folded = {fold_name(name) for name in names}
    shaped = [name for name in names if (base := re.fullmatch(r"(.+)_\d+", name)) and fold_name(base[1]) in folded]
    if not shaped:
        return
    arguments = [step.files, *(f"{name} := {value}" for name, value in step.options if name in _JSON_OBJECT_OPTIONS)]
    text = f"SELECT DISTINCT unnest(json_keys(json)) FROM read_json_objects({', '.join(arguments)})"
    keys = {str(row[0]) for row in _seam.submit(rel._con, text, None).fetchall()}
    made_up = [name for name in shaped if name not in keys]
    if made_up:
        spellings: dict[str, list[str]] = {}
        for key in sorted(keys):
            spellings.setdefault(fold_name(key), []).append(key)
        clashing = "; ".join(" and ".join(map(repr, group)) for group in spellings.values() if len(group) > 1)
        message = (
            f"{step.function} cannot keep the schema it bound: the keys {clashing} differ only in case, and the reader "
            f"names a column {', '.join(map(repr, made_up))} that no file holds; give the columns with columns= to "
            "read such a file"
        )
        raise _error(_duckdb.InvalidInputException, message)


def _pin_reader_schema(rel: DuckDBPyRelation) -> None:
    """Write the bound schema into the reader call, so every read sees the schema the relation reports.

    The pinned `columns` are the columns the files hold; the ones the reader generates, a file name or a Hive
    partition, stay generated. The pinned call must answer exactly the schema the relation reported, and a call
    that does not is a reader that cannot promise its schema, so it raises at construction rather than reading
    unpinned.
    """
    step = rel._step
    if not isinstance(step, Reader) or step.function not in _SNIFFING_READERS:
        return
    if any(name == "columns" for name, _ in step.options):
        return
    names, types = rel._schema
    if not names:
        return
    width = _physical_columns(rel, step)
    columns = ("columns", _bound_columns(names[:width], types[:width]))
    kept = [(name, value) for name, value in step.options if name not in _SCHEMA_OPTIONS]
    projection = "*"
    if step.function.startswith("read_csv") and ("union_by_name", "true") in step.options:
        # The CSV reader applies `columns` to every file, which a union of files with different columns cannot
        # satisfy. The types are pinned by name instead, which each file applies to the columns it holds, and
        # the select list keeps the bound names, so a column a file gains later is not read. Each file is still
        # sniffed on every read, since its own columns are what the union is made of.
        typed = [(name, value) for name, value in step.options if name not in _TYPE_OPTIONS]
        pinned_options = [*typed, ("types", _bound_columns(names[:width], types[:width]))]
        projection = ", ".join(quote(name) for name in names)
    elif step.function.startswith("read_csv"):
        # Without the dialect the reader would still have to sniff one, which is the cost this pinning removes, so
        # detection is only turned off when the sniffer answered for a single file. What the user set stays theirs.
        given = {name for name, _ in kept}
        dialect = [(name, value) for name, value in _csv_dialect(rel._con, step) if name not in given]
        if dialect:
            kept = [(name, value) for name, value in kept if name != "auto_detect"]
            pinned_options = [*kept, *dialect, ("auto_detect", "false"), columns]
        else:
            pinned_options = [*kept, columns]
    else:
        # Only the columns: a JSON reader still has to detect the file's format, and turning detection off would
        # refuse a file whose records are not one per line.
        _refuse_renamed_json_keys(rel, step, names[:width])
        pinned_options = [*kept, columns]
    pinned = dataclasses.replace(step, options=tuple(pinned_options), projection=projection)
    signature = _seam.bind(rel._con, f"SELECT {projection} FROM {pinned.from_item()}")
    answered = (list(signature.names), [str(t) for t in signature.types])
    if answered != (list(names), [str(t) for t in types]):
        message = (
            f"{step.function} cannot keep the schema it bound: with that schema written into the call it reads "
            f"{_bound_columns(*answered)} instead of {_bound_columns(names, [str(t) for t in types])}"
        )
        raise _error(_duckdb.InvalidInputException, message)
    rel._step = pinned
    rel._open = _open_for(pinned, ())


def _reader(
    con: _duckdb.DuckDBPyConnection,
    function: str,
    relation_type: str,
    files: tuple[str, tuple[object, ...]],
    options: list[tuple[str, str]],
) -> DuckDBPyRelation:
    step = Reader(relation_type, function, files[0], tuple(options))
    rel = DuckDBPyRelation._make(con, step, alias=_unnamed(), handles=files[1])
    _pin_reader_schema(rel)
    return rel


def _type_name(value: object) -> str:
    return type(value).__name__


def _multi_file_options(options: list[tuple[str, str]], **given: object) -> None:
    """The options every multi file reader takes, checked as the old readers checked them."""
    filename = given.get("filename")
    if filename is not None:
        options.append(("filename", _option_literal(filename)))
    hive_types = given.get("hive_types")
    if hive_types is not None:
        options.append(("hive_types", _option_literal(hive_types)))
    for name in ("hive_partitioning", "union_by_name", "hive_types_autocast"):
        value = given.get(name)
        if value is None:
            continue
        if not isinstance(value, bool):
            message = f"read_json only accepts '{name}' as a boolean, not '{_type_name(value)}'"
            raise _error(_duckdb.BinderException, message)
        options.append((name, _option_literal(value)))


def _columns_option(method: str, columns: object) -> str:
    if not isinstance(columns, Mapping):
        raise _error(_duckdb.BinderException, f"{method} only accepts 'columns' as a dict[str, str]")
    fields: dict[str, str] = {}
    for name, kind in columns.items():
        if not isinstance(name, str):
            message = f"The provided column name must be a str, not of type '{_type_name(name)}'"
            raise _error(_duckdb.BinderException, message)
        if not isinstance(kind, str):
            message = f"The provided column type must be a str, not of type '{_type_name(name)}'"
            raise _error(_duckdb.BinderException, message)
        fields[name] = kind
    return _option_literal(fields)


_CSV_ARGUMENTS = (
    "header",
    "strict_mode",
    "compression",
    "comment",
    "sep",
    "delimiter",
    "files_to_sniff",
    "dtype",
    "na_values",
    "skiprows",
    "quotechar",
    "escapechar",
    "encoding",
    "parallel",
    "date_format",
    "timestamp_format",
    "sample_size",
    "auto_detect",
    "all_varchar",
    "normalize_names",
    "null_padding",
    "names",
    "lineterminator",
    "columns",
    "auto_type_candidates",
    "max_line_size",
    "ignore_errors",
    "store_rejects",
    "rejects_table",
    "rejects_scan",
    "rejects_limit",
    "force_not_null",
    "buffer_size",
    "decimal",
    "allow_quoted_nulls",
    "filename",
    "hive_partitioning",
    "union_by_name",
    "hive_types",
    "hive_types_autocast",
    "thousands",
)


def _unknown_csv_argument(name: str) -> NoReturn:
    import difflib

    suggestions = difflib.get_close_matches(name, _CSV_ARGUMENTS, n=3, cutoff=0.0)
    lines = [
        f'The methods read_csv and read_csv_auto do not have the "{name}" argument.',
        "Possible arguments as suggestions: ",
        *(f"* {s}" for s in suggestions),
    ]
    raise _error(_duckdb.InvalidInputException, "\n".join(lines) + "\n")


def _csv_flag(options: list[tuple[str, str]], name: str, value: object, option: str) -> None:
    """A flag the old reader took as a boolean, or as the integer 0."""
    if value is None:
        return
    if isinstance(value, bool):
        flag = value
    elif isinstance(value, int):
        if value != 0:
            raise _error(_duckdb.InvalidInputException, f"read_csv only accepts 0 if '{name}' is given as an integer")
        flag = True
    else:
        message = f"read_csv only accepts '{name}' as an integer, or a boolean"
        raise _error(_duckdb.InvalidInputException, message)
    options.append((option, _option_literal(flag)))


def _checked(
    options: list[tuple[str, str]],
    value: object,
    kinds: tuple[type, ...],
    option: str,
    message: str,
    error: type[Exception] = _duckdb.InvalidInputException,
) -> None:
    if value is None:
        return
    if not isinstance(value, kinds) or (bool not in kinds and isinstance(value, bool) and int in kinds):
        raise _error(error, message.format(type=_type_name(value)))
    options.append((option, _option_literal(value)))


def _type_text(value: object) -> str:
    if isinstance(value, str):
        return value
    kind = value if isinstance(value, _api().sqltypes.DuckDBPyType) else _api().sqltypes.DuckDBPyType(value)
    return str(kind)


def read_csv(con: _duckdb.DuckDBPyConnection, path_or_buffer: object, **kwargs: object) -> DuckDBPyRelation:
    """A relation over CSV files, with today's keywords as named arguments of read_csv."""
    for name in kwargs:
        if name not in _CSV_ARGUMENTS:
            _unknown_csv_argument(name)
    files = _files(con, path_or_buffer)
    get = kwargs.get
    options: list[tuple[str, str]] = []
    _multi_file_options(
        options,
        filename=get("filename"),
        hive_partitioning=get("hive_partitioning"),
        union_by_name=get("union_by_name"),
        hive_types=get("hive_types"),
        hive_types_autocast=get("hive_types_autocast"),
    )
    _csv_flag(options, "header", get("header"), "header")
    _csv_flag(options, "strict_mode", get("strict_mode"), "strict_mode")
    _checked(options, get("compression"), (str,), "compression", "read_csv only accepts 'compression' as a string")
    dtype = get("dtype")
    if dtype is not None:
        if isinstance(dtype, Mapping):
            options.append(("dtypes", _option_literal({str(k): _type_text(v) for k, v in dtype.items()})))
        elif isinstance(dtype, (list, tuple)):
            options.append(("dtypes", _option_literal([_type_text(v) for v in dtype])))
        else:
            message = "read_csv only accepts 'dtype' as a dictionary or a list of strings"
            raise _error(_duckdb.InvalidInputException, message)
    sep, delimiter = get("sep"), get("delimiter")
    if sep is not None and delimiter is not None:
        raise _error(_duckdb.InvalidInputException, "read_csv takes either 'delimiter' or 'sep', not both")
    if sep is not None or delimiter is not None:
        options.append(("delim", _option_literal(str(sep if sep is not None else delimiter))))
    _checked(
        options, get("files_to_sniff"), (int,), "files_to_sniff", "read_csv only accepts 'files_to_sniff' as an integer"
    )
    names = get("names")
    if names is not None:
        if not isinstance(names, (list, tuple)):
            raise _error(_duckdb.InvalidInputException, "read_csv only accepts 'names' as a list of strings")
        if not all(isinstance(n, str) for n in names):
            raise _error(_duckdb.InvalidInputException, "read_csv 'names' list has to consist of only strings")
        options.append(("names", _option_literal(list(names))))
    na_values = get("na_values")
    if na_values is not None:
        if isinstance(na_values, str):
            options.append(("nullstr", _option_literal([na_values])))
        elif isinstance(na_values, (list, tuple)):
            if not all(isinstance(n, str) for n in na_values):
                message = "read_csv 'na_values' list has to consist of only strings"
                raise _error(_duckdb.InvalidInputException, message)
            options.append(("nullstr", _option_literal(list(na_values))))
        else:
            message = "read_csv only accepts 'na_values' as a string or a list of strings"
            raise _error(_duckdb.InvalidInputException, message)
    _checked(options, get("skiprows"), (int,), "skip", "read_csv only accepts 'skiprows' as an integer")
    _checked(options, get("parallel"), (bool,), "parallel", "read_csv only accepts 'parallel' as a boolean")
    _checked(options, get("quotechar"), (str,), "quote", "read_csv only accepts 'quotechar' as a string")
    _checked(options, get("comment"), (str,), "comment", "read_csv only accepts 'comment' as a string")
    _checked(options, get("thousands"), (str,), "thousands", "read_csv only accepts 'thousands' as a string")
    _checked(options, get("escapechar"), (str,), "escape", "read_csv only accepts 'escapechar' as a string")
    encoding = get("encoding")
    if encoding is not None:
        if not isinstance(encoding, str):
            raise _error(_duckdb.InvalidInputException, "read_csv only accepts 'encoding' as a string")
        if fold_name(encoding) not in ("utf8", "utf-8"):
            message = "Copy is only supported for UTF-8 encoded files, ENCODING 'UTF-8'"
            raise _error(_duckdb.BinderException, message)
    _checked(options, get("date_format"), (str,), "dateformat", "read_csv only accepts 'date_format' as a string")
    _csv_flag(options, "auto_detect", get("auto_detect"), "auto_detect")
    _checked(
        options,
        get("timestamp_format"),
        (str,),
        "timestampformat",
        "read_csv only accepts 'timestamp_format' as a string",
    )
    _checked(options, get("sample_size"), (int,), "sample_size", "read_csv only accepts 'sample_size' as an integer")
    for flag in ("all_varchar", "normalize_names", "null_padding"):
        _checked(options, get(flag), (bool,), flag, f"read_csv only accepts '{flag}' as a boolean")
    terminator = get("lineterminator")
    if terminator is not None:
        spelled = getattr(terminator, "name", terminator)
        endings = {"\\n": "\\n", "LINE_FEED": "\\n", "\\r\\n": "\\r\\n", "CARRIAGE_RETURN_LINE_FEED": "\\r\\n"}
        if not isinstance(spelled, str) or spelled not in endings:
            message = (
                "read_csv only accepts 'lineterminator' as a string or CSVLineTerminator, "
                f"not '{_type_name(terminator)}'"
            )
            raise _error(_duckdb.BinderException, message)
        options.append(("new_line", _option_literal(endings[spelled])))
    binder = _duckdb.BinderException
    max_line_size = get("max_line_size")
    if max_line_size is not None:
        if isinstance(max_line_size, bool) or not isinstance(max_line_size, (str, int)):
            message = (
                f"read_csv only accepts 'max_line_size' as a string or an integer, not '{_type_name(max_line_size)}'"
            )
            raise _error(binder, message)
        options.append(("max_line_size", _option_literal(str(max_line_size))))
    candidates = get("auto_type_candidates")
    if candidates is not None:
        if not isinstance(candidates, list):
            message = f"read_csv only accepts 'auto_type_candidates' as a list[str], not '{_type_name(candidates)}'"
            raise _error(binder, message)
        options.append(("auto_type_candidates", _option_literal([_type_text(c) for c in candidates])))
    _checked(
        options,
        get("ignore_errors"),
        (bool,),
        "ignore_errors",
        "read_csv only accepts 'ignore_errors' as a bool, not '{type}'",
        binder,
    )
    _checked(
        options,
        get("store_rejects"),
        (bool,),
        "store_rejects",
        "read_csv only accepts 'store_rejects' as a bool, not '{type}'",
        binder,
    )
    _checked(
        options,
        get("rejects_table"),
        (str,),
        "rejects_table",
        "read_csv only accepts 'rejects_table' as a string, not '{type}'",
        binder,
    )
    _checked(
        options,
        get("rejects_scan"),
        (str,),
        "rejects_scan",
        "read_csv only accepts 'rejects_scan' as a string, not '{type}'",
        binder,
    )
    _checked(
        options,
        get("rejects_limit"),
        (int,),
        "rejects_limit",
        "read_csv only accepts 'rejects_limit' as an int, not '{type}'",
        binder,
    )
    force_not_null = get("force_not_null")
    if force_not_null is not None:
        if not isinstance(force_not_null, list):
            message = f"read_csv only accepts 'force_not_null' as a list[str], not '{_type_name(force_not_null)}'"
            raise _error(binder, message)
        options.append(("force_not_null", _option_literal([str(f) for f in force_not_null])))
    _checked(
        options,
        get("buffer_size"),
        (int,),
        "buffer_size",
        "read_csv only accepts 'buffer_size' as a list[str], not '{type}'",
        binder,
    )
    _checked(
        options,
        get("decimal"),
        (str,),
        "decimal_separator",
        "read_csv only accepts 'decimal' as a string, not '{type}'",
        binder,
    )
    _checked(
        options,
        get("allow_quoted_nulls"),
        (bool,),
        "allow_quoted_nulls",
        "read_csv only accepts 'allow_quoted_nulls' as a bool, not '{type}'",
        binder,
    )
    columns = get("columns")
    if columns is not None:
        options.append(("columns", _columns_option("read_csv", columns)))
    return _reader(con, "read_csv", "SUBQUERY_RELATION", files, options)


def read_json(
    con: _duckdb.DuckDBPyConnection,
    path_or_buffer: object,
    *,
    columns: object = None,
    sample_size: object = None,
    maximum_depth: object = None,
    records: object = None,
    format: object = None,
    date_format: object = None,
    timestamp_format: object = None,
    compression: object = None,
    maximum_object_size: object = None,
    ignore_errors: object = None,
    convert_strings_to_integers: object = None,
    field_appearance_threshold: object = None,
    map_inference_threshold: object = None,
    maximum_sample_files: object = None,
    filename: object = None,
    hive_partitioning: object = None,
    union_by_name: object = None,
    hive_types: object = None,
    hive_types_autocast: object = None,
) -> DuckDBPyRelation:
    """A relation over JSON files, with today's keywords as named arguments of read_json."""
    files = _files(con, path_or_buffer)
    options: list[tuple[str, str]] = []
    _multi_file_options(
        options,
        filename=filename,
        hive_partitioning=hive_partitioning,
        union_by_name=union_by_name,
        hive_types=hive_types,
        hive_types_autocast=hive_types_autocast,
    )
    if columns is not None:
        options.append(("columns", _columns_option("read_json", columns)))
    binder = _duckdb.BinderException
    for name, value in (
        ("records", records),
        ("format", format),
        ("date_format", date_format),
        ("timestamp_format", timestamp_format),
        ("compression", compression),
    ):
        _checked(options, value, (str,), name, f"read_json only accepts '{name}' as a string, not '{{type}}'", binder)
    _checked(
        options,
        sample_size,
        (int,),
        "sample_size",
        "read_json only accepts 'sample_size' as an integer, not '{type}'",
        binder,
    )
    _checked(
        options,
        maximum_depth,
        (int,),
        "maximum_depth",
        "read_json only accepts 'maximum_depth' as an integer, not '{type}'",
        binder,
    )
    _checked(
        options,
        maximum_object_size,
        (int,),
        "maximum_object_size",
        "read_json only accepts 'maximum_object_size' as an unsigned integer, not '{type}'",
        binder,
    )
    _checked(
        options,
        ignore_errors,
        (bool,),
        "ignore_errors",
        "read_json only accepts 'ignore_errors' as a boolean, not '{type}'",
        binder,
    )
    _checked(
        options,
        convert_strings_to_integers,
        (bool,),
        "convert_strings_to_integers",
        "read_json only accepts 'convert_strings_to_integers' as a boolean, not '{type}'",
        binder,
    )
    _checked(
        options,
        field_appearance_threshold,
        (float,),
        "field_appearance_threshold",
        "read_json only accepts 'field_appearance_threshold' as a float, not '{type}'",
        binder,
    )
    _checked(
        options,
        map_inference_threshold,
        (int,),
        "map_inference_threshold",
        "read_json only accepts 'map_inference_threshold' as an integer, not '{type}'",
        binder,
    )
    _checked(
        options,
        maximum_sample_files,
        (int,),
        "maximum_sample_files",
        "read_json only accepts 'maximum_sample_files' as an integer, not '{type}'",
        binder,
    )
    if columns is None:
        options.append(("auto_detect", "true"))
    return _reader(con, "read_json", "TABLE_FUNCTION_RELATION", files, options)


def read_parquet(
    con: _duckdb.DuckDBPyConnection,
    path_or_buffer: object,
    binary_as_string: bool = False,  # noqa: FBT001
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: object = None,
) -> DuckDBPyRelation:
    """A relation over Parquet files through parquet_scan."""
    files, handles = _files(con, path_or_buffer)
    if not files.startswith("["):
        files = f"[{files}]"
    options = [
        ("binary_as_string", _option_literal(bool(binary_as_string))),
        ("file_row_number", _option_literal(bool(file_row_number))),
        ("filename", _option_literal(bool(filename))),
        ("hive_partitioning", _option_literal(bool(hive_partitioning))),
        ("union_by_name", _option_literal(bool(union_by_name))),
    ]
    if compression is not None:
        if not isinstance(compression, str):
            raise _error(_duckdb.InvalidInputException, "from_parquet only accepts 'compression' as a string")
        options.append(("compression", _option_literal(compression)))
    return _reader(con, "parquet_scan", "SUBQUERY_RELATION", (files, handles), options)


def _is_dataframe(obj: object) -> bool:
    return any(c.__module__.startswith("pandas") and c.__name__ == "DataFrame" for c in type(obj).__mro__)


def _arrow_shaped(obj: object) -> bool:
    """Whether an object could be an Arrow source, judged from its class alone, so none of its own code runs."""
    kind = type(obj)
    if kind.__name__ == "PyCapsule":
        return True
    if any(c.__module__.partition(".")[0] in ("pyarrow", "polars") for c in kind.__mro__):
        return True
    for name in ("__arrow_c_stream__", "__arrow_c_array__"):
        found = inspect.getattr_static(kind, name, None)
        if found is None:
            continue
        if isinstance(found, property):
            try:
                getattr(obj, name)
            except AttributeError:
                # An export that is not there after all, which is how the scan judges it too.
                continue
            except Exception:
                # An export that raises anything else is the scan's to report, with the source named.
                return True
        return True
    pairs = (("to_reader", "projected_schema"), ("scanner", "schema"))
    return any(all(inspect.getattr_static(kind, n, None) is not None for n in pair) for pair in pairs)


def from_df(con: _duckdb.DuckDBPyConnection, df: object) -> DuckDBPyRelation:
    """A relation over a pandas DataFrame, read where it lives when the relation runs."""
    _ensure_open(con)
    if not _is_dataframe(df):
        message = f"from_df(): incompatible function arguments. The DataFrame argument is a {_type_name(df)}"
        raise TypeError(message)
    handle = _seam.register_source(con, df)
    return DuckDBPyRelation._make(con, Registered(handle.name), alias=f"df_{secrets.token_hex(8)}", handles=(handle,))


def from_arrow(con: _duckdb.DuckDBPyConnection, arrow_object: object) -> DuckDBPyRelation:
    """A relation over an Arrow shaped object, read where it lives when the relation runs."""
    _ensure_open(con)
    if not _arrow_shaped(arrow_object):
        message = f"Python Object Type {_type_name(arrow_object)} is not an accepted Arrow Object."
        raise _error(_duckdb.InvalidInputException, message)
    handle = _seam.register_source(con, arrow_object)
    return DuckDBPyRelation._make(
        con, Registered(handle.name), alias=f"arrow_object_{secrets.token_hex(8)}", handles=(handle,)
    )


_SQL_DOC = (
    "Run a SQL query. A statement that only reads returns a relation that runs the query each time it is read, "
    "parameter values bound at every run. Any other statement runs once, now, and gives a relation over what it "
    "returned. Statements before the last one run immediately."
)


def _connection(connection: _duckdb.DuckDBPyConnection | None) -> _duckdb.DuckDBPyConnection:
    return connection if connection is not None else _duckdb.default_connection()


def _connection_sql(
    self: _duckdb.DuckDBPyConnection, query: object, *, alias: str = "", params: object = None
) -> DuckDBPyRelation | None:
    return sql(self, query, alias, params)


_connection_sql.__doc__ = _SQL_DOC


def _connection_table(self: _duckdb.DuckDBPyConnection, table_name: str) -> DuckDBPyRelation:
    """Create a relation object for the named table."""
    return table(self, table_name)


def _connection_view(self: _duckdb.DuckDBPyConnection, view_name: str) -> DuckDBPyRelation:
    """Create a relation object for the named view."""
    return view(self, view_name)


def _connection_values(self: _duckdb.DuckDBPyConnection, *args: object) -> DuckDBPyRelation:
    """Create a relation object from the passed values."""
    return values(self, *args)


def _connection_table_function(
    self: _duckdb.DuckDBPyConnection, name: str, parameters: object = None
) -> DuckDBPyRelation:
    """Create a relation object from the named table function with given parameters."""
    return table_function(self, name, parameters)


def _source_argument(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    if args:
        return args[0]
    return kwargs.pop("path_or_buffer", None)


def _connection_read_csv(self: _duckdb.DuckDBPyConnection, *args: object, **kwargs: object) -> DuckDBPyRelation:
    """Create a relation object from the CSV file in 'name'."""
    return read_csv(self, _source_argument(args, kwargs), **kwargs)


def _connection_read_json(
    self: _duckdb.DuckDBPyConnection,
    path_or_buffer: object,
    **kwargs: Any,  # noqa: ANN401
) -> DuckDBPyRelation:
    """Create a relation object from the JSON file in 'name'."""
    return read_json(self, path_or_buffer, **kwargs)


def _connection_read_parquet(
    self: _duckdb.DuckDBPyConnection,
    path_or_buffer: object,
    binary_as_string: bool = False,  # noqa: FBT001
    **kwargs: Any,  # noqa: ANN401
) -> DuckDBPyRelation:
    """Create a relation object from the Parquet path(s) or file-like object(s) in 'path_or_buffer'."""
    return read_parquet(self, path_or_buffer, binary_as_string, **kwargs)


def _connection_from_df(self: _duckdb.DuckDBPyConnection, df: object) -> DuckDBPyRelation:
    """Create a relation object from the DataFrame in df."""
    return from_df(self, df)


def _connection_from_arrow(self: _duckdb.DuckDBPyConnection, arrow_object: object) -> DuckDBPyRelation:
    """Create a relation object from an Arrow object."""
    return from_arrow(self, arrow_object)


_ARROW_TABLE_COPY_WARNING = (
    "Arrow output on a DB-API result makes an extra copy of the rows, use con.sql(query).to_arrow_table() instead."
)
_ARROW_READER_COPY_WARNING = (
    "Arrow output on a DB-API result makes an extra copy of the rows, use con.sql(query).to_arrow_reader() instead."
)


def _connection_to_arrow_table(self: _duckdb.DuckDBPyConnection, batch_size: int = _seam.DEFAULT_BATCH_SIZE) -> Any:  # noqa: ANN401
    """Every row the pending result still holds, as an Arrow Table."""
    warnings.warn(_ARROW_TABLE_COPY_WARNING, UserWarning, _caller_level())
    return _seam.result_arrow(self, int(batch_size), reader=False)


def _connection_to_arrow_reader(self: _duckdb.DuckDBPyConnection, batch_size: int = _seam.DEFAULT_BATCH_SIZE) -> Any:  # noqa: ANN401
    """A reader over every row the pending result still holds."""
    warnings.warn(_ARROW_READER_COPY_WARNING, UserWarning, _caller_level())
    return _seam.result_arrow(self, int(batch_size), reader=True)


#: The Python methods of a connection, by name.
CONNECTION_METHODS: dict[str, Any] = {
    "sql": _connection_sql,
    "query": _connection_sql,
    "from_query": _connection_sql,
    "table": _connection_table,
    "view": _connection_view,
    "values": _connection_values,
    "table_function": _connection_table_function,
    "read_csv": _connection_read_csv,
    "from_csv_auto": _connection_read_csv,
    "read_json": _connection_read_json,
    "read_parquet": _connection_read_parquet,
    "from_parquet": _connection_read_parquet,
    "from_df": _connection_from_df,
    "from_arrow": _connection_from_arrow,
    "to_arrow_table": _connection_to_arrow_table,
    "to_arrow_reader": _connection_to_arrow_reader,
}


def module_sql(
    query: object,
    *,
    alias: str = "",
    params: object = None,
    connection: _duckdb.DuckDBPyConnection | None = None,
) -> DuckDBPyRelation | None:
    return sql(_connection(connection), query, alias, params)


module_sql.__doc__ = _SQL_DOC


def module_table(table_name: str, *, connection: _duckdb.DuckDBPyConnection | None = None) -> DuckDBPyRelation:
    """Create a relation object for the named table."""
    return table(_connection(connection), table_name)


def module_view(view_name: str, *, connection: _duckdb.DuckDBPyConnection | None = None) -> DuckDBPyRelation:
    """Create a relation object for the named view."""
    return view(_connection(connection), view_name)


def module_values(*args: object, connection: _duckdb.DuckDBPyConnection | None = None) -> DuckDBPyRelation:
    """Create a relation object from the passed values."""
    return values(_connection(connection), *args)


def module_table_function(
    name: str, parameters: object = None, *, connection: _duckdb.DuckDBPyConnection | None = None
) -> DuckDBPyRelation:
    """Create a relation object from the named table function with given parameters."""
    return table_function(_connection(connection), name, parameters)


def module_read_csv(*args: object, **kwargs: object) -> DuckDBPyRelation:
    """Create a relation object from the CSV file in 'name'."""
    source = _source_argument(args, kwargs)
    con: object = None
    for key in ("connection", "conn"):
        given = kwargs.pop(key, None)
        if con is None and given is not None:
            con = given
    return read_csv(_connection(con), source, **kwargs)  # type: ignore[arg-type]


def module_read_json(
    path_or_buffer: object,
    *,
    connection: _duckdb.DuckDBPyConnection | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> DuckDBPyRelation:
    """Create a relation object from the JSON file in 'name'."""
    return read_json(_connection(connection), path_or_buffer, **kwargs)


def module_read_parquet(
    path_or_buffer: object,
    binary_as_string: bool = False,  # noqa: FBT001
    *,
    connection: _duckdb.DuckDBPyConnection | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> DuckDBPyRelation:
    """Create a relation object from the Parquet path(s) or file-like object(s) in 'path_or_buffer'."""
    return read_parquet(_connection(connection), path_or_buffer, binary_as_string, **kwargs)


def module_from_df(df: object, *, connection: _duckdb.DuckDBPyConnection | None = None) -> DuckDBPyRelation:
    """Create a relation object from the DataFrame in df."""
    return from_df(_connection(connection), df)


def module_from_arrow(
    arrow_object: object, *, connection: _duckdb.DuckDBPyConnection | None = None
) -> DuckDBPyRelation:
    """Create a relation object from an Arrow object."""
    return from_arrow(_connection(connection), arrow_object)


def _frame_first(call: Callable[..., Any]) -> Callable[..., Any]:
    """Take the DataFrame first, or by keyword with the arguments after it given positionally."""

    @functools.wraps(call)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if "df" in kwargs:
            return call(kwargs.pop("df"), *args, **kwargs)
        return call(*args, **kwargs)

    return wrapper


_FETCH_DF = _duckdb.df


@_frame_first
def module_df(
    df: object = None, *, date_as_object: bool = False, connection: _duckdb.DuckDBPyConnection | None = None
) -> Any:  # noqa: ANN401
    """A relation over a DataFrame, or with no DataFrame the DataFrame of the connection's pending result."""
    if df is None:
        return _FETCH_DF(date_as_object=date_as_object, connection=connection)
    return from_df(_connection(connection), df)


@_frame_first
def module_project(
    df: object, *args: object, groups: str = "", connection: _duckdb.DuckDBPyConnection | None = None
) -> DuckDBPyRelation:
    """Project the relation object by the projection in project_expr."""
    return from_df(_connection(connection), df).project(*args, groups=groups)


@_frame_first
def module_distinct(df: object, *, connection: _duckdb.DuckDBPyConnection | None = None) -> DuckDBPyRelation:
    """Retrieve distinct rows from this relation object."""
    return from_df(_connection(connection), df).distinct()


@_frame_first
def module_write_csv(
    df: object,
    filename: str,
    *,
    connection: _duckdb.DuckDBPyConnection | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Write the relation object to a CSV file in 'file_name'."""
    from_df(_connection(connection), df).write_csv(filename, **kwargs)


@_frame_first
def module_aggregate(
    df: object, aggr_expr: object, group_expr: object = "", *, connection: _duckdb.DuckDBPyConnection | None = None
) -> DuckDBPyRelation:
    """Compute the aggregate aggr_expr by the optional groups group_expr on the relation."""
    return from_df(_connection(connection), df).aggregate(aggr_expr, group_expr)


@_frame_first
def module_alias(df: object, alias: str, *, connection: _duckdb.DuckDBPyConnection | None = None) -> DuckDBPyRelation:
    """Rename the relation object to new alias."""
    return from_df(_connection(connection), df).set_alias(alias)


@_frame_first
def module_filter(
    df: object, filter_expr: object, *, connection: _duckdb.DuckDBPyConnection | None = None
) -> DuckDBPyRelation:
    """Filter the relation object by the filter in filter_expr."""
    return from_df(_connection(connection), df).filter(filter_expr)


@_frame_first
def module_limit(
    df: object, n: int, offset: int = 0, *, connection: _duckdb.DuckDBPyConnection | None = None
) -> DuckDBPyRelation:
    """Only retrieve the first n rows from this relation object, starting at offset."""
    return from_df(_connection(connection), df).limit(n, offset)


@_frame_first
def module_order(
    df: object, order_expr: str, *, connection: _duckdb.DuckDBPyConnection | None = None
) -> DuckDBPyRelation:
    """Reorder the relation object by order_expr."""
    return from_df(_connection(connection), df).order(order_expr)


@_frame_first
def module_query_df(
    df: object, virtual_table_name: str, sql_query: str, *, connection: _duckdb.DuckDBPyConnection | None = None
) -> DuckDBPyRelation | None:
    """Run the given SQL query in sql_query on the view named virtual_table_name that refers to the relation object."""
    return from_df(_connection(connection), df).query(virtual_table_name, sql_query)


def module_arrow(arrow_object: object, *, connection: _duckdb.DuckDBPyConnection | None = None) -> DuckDBPyRelation:
    """Create a relation object from an Arrow object."""
    return from_arrow(_connection(connection), arrow_object)


def module_to_arrow_table(
    batch_size: int = _seam.DEFAULT_BATCH_SIZE, *, connection: _duckdb.DuckDBPyConnection | None = None
) -> Any:  # noqa: ANN401
    """Every row the connection's pending result still holds, as an Arrow Table."""
    return _connection_to_arrow_table(_connection(connection), batch_size)


def module_to_arrow_reader(
    batch_size: int = _seam.DEFAULT_BATCH_SIZE, *, connection: _duckdb.DuckDBPyConnection | None = None
) -> Any:  # noqa: ANN401
    """A reader over every row the connection's pending result still holds."""
    return _connection_to_arrow_reader(_connection(connection), batch_size)

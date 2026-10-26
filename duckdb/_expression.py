"""Build a piece of SQL as a Python value.

An expression such as `col("total") > 1000` describes a computation and holds no connection, schema or types, so the
same one can be used against different tables. Two rules differ from plain Python: a bare string is a value and `col`
names a column, and conditions combine with `&`, `|` and `~`, since Python's `and` and `or` cannot be overloaded.

The constructors of the earlier C++ expression API (`ColumnExpression`, `ConstantExpression` and the rest) build the
same tree, so an expression made either way combines with any other.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import decimal
import inspect
import math
import re
import secrets
import string
import sys
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, NoReturn, TypeAlias, cast

import _duckdb
from _duckdb import Error, InvalidInputException

from duckdb import _seam
from duckdb._expression_aggregates import AggregateMethods
from duckdb._sql_keywords import KEYWORD_CATEGORY, KEYWORDS
from duckdb.sqltypes import DuckDBPyType
from duckdb.value.constant import Value

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from duckdb._expression_namespaces import DtExpr, JsonExpr, ListExpr, StrExpr

ColumnName: TypeAlias = str
ColumnNames: TypeAlias = "str | Iterable[str]"
NewName: TypeAlias = str
QualifiedName: TypeAlias = "str | tuple[str, ...]"
SqlText: TypeAlias = str
PythonValue: TypeAlias = object
ValueOrExpr: TypeAlias = object
ValueSequence: TypeAlias = "Iterable[object]"
Exprs: TypeAlias = "Expression | Sequence[Expression]"
_FieldKey: TypeAlias = "str | int"
_Renames: TypeAlias = "Mapping[str, str] | Iterable[tuple[str, str]]"
_TextValue: TypeAlias = str

__all__ = [
    "Expression",
    "coalesce",
    "col",
    "count_all",
    "fn",
    "lit",
    "sql_expr",
    "star",
    "when",
]


def quote(name: str) -> str:
    """A SQL identifier, always quoted so no name can be read as syntax."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def _invalid_input(message: str) -> InvalidInputException:
    """The engine's class for a mistake in an argument, with the prefix the engine's own messages carry."""
    return InvalidInputException(f"Invalid Input Error: {message}")


def fold_name(name: str) -> str:
    """A name as DuckDB compares names: ASCII letters case folded, every other character as written."""
    return name.translate(_ASCII_LOWER)


def name_parts(name: object, what: str = "name") -> tuple[str, ...]:
    """A string as a one-part path, a tuple of strings as a qualified path. Nothing is ever split."""
    parts = (name,) if isinstance(name, str) else name
    if not isinstance(parts, tuple) or not all(isinstance(part, str) for part in parts):
        message = f"a {what} is a string, or a tuple of strings for a qualified one, not {name!r}"
        raise TypeError(message)
    if not parts or not all(parts):
        message = f"a {what} cannot be empty: {name!r}"
        raise ValueError(message)
    return parts


def _as_name(name: object, what: str) -> str:
    """One name, whole: a string, never split and never empty."""
    if not isinstance(name, str):
        message = f"a {what} is a string, not {name!r}"
        raise TypeError(message)
    if not name:
        message = f"a {what} cannot be empty"
        raise ValueError(message)
    return name


def _items(value: object) -> Iterator[object] | None:
    """The items of what Python can iterate, through `__iter__` or `__getitem__`, or None; text is the caller's."""
    try:
        return iter(cast("Iterable[object]", value))
    except TypeError:
        return None


def _as_names(names: object, what: str) -> tuple[str, ...]:
    """One name or several, where a bare string is one name and not its characters."""
    if isinstance(names, str):
        return (_as_name(names, what),)
    items = None if isinstance(names, bytes) else _items(names)
    if items is None:
        message = f"expected a {what} or a list of them, not {names!r}"
        raise TypeError(message)
    return tuple(_as_name(name, what) for name in items)


def _as_renames(renames: object) -> dict[str, str] | None:
    """Old names to new, as a mapping or as (old, new) pairs; a string is never a pair of names."""
    if renames is None or isinstance(renames, Mapping):
        pairs: Iterator[object] | None = iter(renames.items()) if renames is not None else iter(())
    else:
        pairs = None if isinstance(renames, (str, bytes)) else _items(renames)
    if pairs is None:
        message = f"rename takes a dict from old name to new, or (old, new) pairs, not {renames!r}"
        raise TypeError(message)
    out: dict[str, str] = {}
    for pair in pairs:
        both = None if isinstance(pair, (str, bytes)) else _items(pair)
        named = tuple(both) if both is not None else ()
        if len(named) != 2:
            message = f"a rename is an (old, new) pair of names, not {pair!r}"
            raise TypeError(message)
        out[_as_name(named[0], "column name")] = _as_name(named[1], "column name")
    return out or None


def _as_sql(text: object, what: str) -> str:
    """SQL text, taken as written."""
    if not isinstance(text, str):
        message = f"{what} takes SQL as a string, not {text!r}"
        raise TypeError(message)
    return text


def _as_text(value: object, what: str) -> str:
    """A value that must be text."""
    if not isinstance(value, str):
        message = f"{what} is text, not {value!r}"
        raise TypeError(message)
    return value


def _as_type(type_text: object, what: str) -> str:
    """A SQL type as text: SQL text as written, or what `DuckDBPyType` makes of a type object."""
    if isinstance(type_text, str):
        return type_text
    try:
        return str(type_text if isinstance(type_text, DuckDBPyType) else DuckDBPyType(type_text))
    except (Error, TypeError, ValueError):
        message = f"{what} takes a SQL type as text or a DuckDBPyType, not {type_text!r}"
        raise TypeError(message) from None


def _as_field_key(key: object) -> str | int:
    """A struct field's name, a map key, or a list position."""
    if isinstance(key, bool) or not isinstance(key, (str, int)):
        message = f"an expression is indexed by a field name or a position, not by {type(key).__name__}"
        raise TypeError(message)
    return key


def identifier(parts: tuple[str, ...]) -> str:
    """The SQL for a name's parts, each quoted, joined by dots. The parts were checked where they were made."""
    return ".".join(quote(part) for part in parts)


plain_identifier = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def function_name(name: str | tuple[str, ...]) -> str:
    """A function's name as DuckDB writes it, quoted only when it must be; syntax forms like COALESCE bypass it."""
    return ".".join(
        part if plain_identifier.match(part) and fold_name(part) not in KEYWORDS else quote(part)
        for part in name_parts(name, "function name")
    )


# What an operand may be without an explicit lit(); dates, decimals and UUIDs are here because they convert exactly.
LITERAL_TYPES = (
    bool,
    int,
    float,
    str,
    bytes,
    bytearray,
    list,
    tuple,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    decimal.Decimal,
    uuid.UUID,
    dict,
    Value,
)

# Widening order for numeric literals, so a list of mixed numbers gets a type none overflows; only numbers widen.
_NUMERIC_RANK = {"INTEGER": 1, "BIGINT": 2, "UBIGINT": 3, "HUGEINT": 4, "UHUGEINT": 5, "DOUBLE": 6}

_INT32 = 2**31
_INT64 = 2**63
_INT128 = 2**127


def _integer_type(value: int) -> str:
    """The type the parameter path gives a Python int, refused with its error beyond 128 bits."""
    if -_INT32 <= value < _INT32:
        return "INTEGER"
    if -_INT64 <= value < _INT64:
        return "BIGINT"
    if 0 <= value < 2 * _INT64:
        return "UBIGINT"
    if -_INT128 <= value < _INT128:
        return "HUGEINT"
    if 0 <= value < 2 * _INT128:
        return "UHUGEINT"
    digits = ""
    with contextlib.suppress(ValueError):
        # Past Python's limit on the digits it prints, the message goes without them, as the parameter path's does.
        digits = f": {value}"
    message = f"Python integer too large for 128-bit integer type{digits}"
    raise _invalid_input(message)


def _holds_unsigned(value: object) -> bool:
    """Whether an int only an unsigned BIGINT holds is in `value`, at any depth."""
    if isinstance(value, (list, tuple)):
        return any(_holds_unsigned(item) for item in value)
    if isinstance(value, dict):
        return any(_holds_unsigned(key) or _holds_unsigned(item) for key, item in value.items())
    return isinstance(value, int) and not isinstance(value, bool) and _integer_type(value) == "UBIGINT"


def _widen(types: list[str]) -> str | None:
    """A common element type for a list, or None when there is no safe one."""
    unique = list(dict.fromkeys(types))
    if len(unique) == 1:
        return unique[0]
    if all(t in _NUMERIC_RANK for t in unique):
        if "DOUBLE" in unique:
            return "DOUBLE"
        widest = max(unique, key=lambda t: _NUMERIC_RANK[t])
        # As the parameter path widens: an unsigned BIGINT beside a signed integer needs a HUGEINT.
        return "HUGEINT" if widest == "UBIGINT" else widest
    return None


def _is_nat(value: object) -> bool:
    """Whether `value` is pandas' or numpy's NaT, which stands for a missing value as None does."""
    cls = type(value)
    if cls.__name__ == "NaTType" and cls.__module__.startswith("pandas"):
        return True
    # numpy's NaT is the least count, in any unit and step.
    return _is_numpy_temporal(value) and int(cast("Any", value).view("i8")) == -(2**63)


def _is_numpy_temporal(value: object) -> bool:
    """Whether `value` is a numpy datetime64 or timedelta64 scalar, recognised without importing numpy."""
    cls = type(value)
    return cls.__module__ == "numpy" and cls.__name__ in ("datetime64", "timedelta64")


#: Exact types no numpy scalar can have, so a value of one needs no numpy check; a subclass such as np.float64 does.
_NATIVE = frozenset(
    {
        type(None),
        bool,
        int,
        float,
        str,
        bytes,
        decimal.Decimal,
        uuid.UUID,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
    }
)

_CONTAINERS = (list, tuple, dict)


class _NumpyTypes(NamedTuple):
    scalars: tuple[type, ...]
    duration: type


#: The numpy module the scalar types were read from, and those types; read again only for another module object.
_numpy_seen: list[Any] = [None, None]


def _numpy_types() -> _NumpyTypes | None:
    """The scalar types of numpy, or None when no value can be a numpy scalar."""
    # Looked up rather than imported, so that importing duckdb does not import numpy; without it there are no scalars.
    numpy = sys.modules.get("numpy")
    if numpy is None:
        return None
    if _numpy_seen[0] is numpy:
        return cast("_NumpyTypes", _numpy_seen[1])
    # A numpy still importing on another thread can lack these, and then nothing has made one of its scalars yet, so
    # that answer is not kept.
    found = [getattr(numpy, name, None) for name in ("bool_", "integer", "floating", "timedelta64")]
    if not all(isinstance(kind, type) for kind in found):
        return None
    *scalars, duration = cast("list[type]", found)
    types = _NumpyTypes(tuple(scalars), duration)
    _numpy_seen[:] = [numpy, types]
    return types


def _converted(value: object, numpy: _NumpyTypes | None) -> object:
    """One value that is no container, a numpy bool, integer or float scalar or a memoryview becoming its Python value.

    A memoryview is the bytes it shows, in order, whatever its item format or strides.
    """
    if isinstance(value, memoryview):
        return value.tobytes()
    if numpy is not None and isinstance(value, numpy.scalars):
        # A timedelta64 is a numpy integer too, and its unit would be lost.
        return value if isinstance(value, numpy.duration) else cast("Any", value).item()
    return _base_value(value)


def _base_value(value: object) -> object:
    """An int, float, str, bytes or Decimal of a subclass as the base type's value it holds, anything else as it is.

    Read through the base type's own methods, as the parameter path reads the stored value, so a subclass's own
    `__str__`, `__repr__` or `replace` cannot change what is written into the SQL.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, bytes):
        return bytes.__bytes__(value)
    if isinstance(value, decimal.Decimal) and type(value) is not decimal.Decimal:
        return decimal.Decimal(value)
    return value


def _as_python(value: object) -> object:
    """A numpy bool, int or float scalar or a memoryview as the Python value it stands for, anything else as it is."""
    if type(value) in _NATIVE:
        return value
    return _converted(value, _numpy_types())


def _all_native(container: list[Any] | tuple[Any, ...] | dict[Any, Any]) -> bool:
    """Whether a list, tuple or dict holds only values of the native types, so nothing in it needs converting."""
    if isinstance(container, dict):
        return _NATIVE.issuperset(map(type, container)) and _NATIVE.issuperset(map(type, container.values()))
    return _NATIVE.issuperset(map(type, container))


def _container_values(container: list[Any] | tuple[Any, ...] | dict[Any, Any]) -> Sequence[object]:
    """A container's values in order, a dict's keys and values alternating."""
    if isinstance(container, dict):
        return [part for entry in container.items() for part in entry]
    return container


def _rebuilt(container: object, children: list[object]) -> object:
    """A container of the same kind holding `children`, in the order `_container_values` gives them."""
    if isinstance(container, dict):
        return dict(zip(children[::2], children[1::2], strict=True))
    return children if isinstance(container, list) else tuple(children)


def plain_value(value: object) -> object:
    """`value` with each numpy bool, int or float scalar and each memoryview in it, at any depth, as its Python value.

    A container holding none comes back as the same object, so a caller still sees whatever happens to it later.
    """
    if type(value) in _NATIVE:
        return value
    numpy = _numpy_types()
    if not isinstance(value, _CONTAINERS):
        return _converted(value, numpy)
    if _all_native(value):
        return value
    # A stack, not recursion, so a value nested deeper than Python's recursion limit still binds as it did before.
    # Each entry holds a container, its values, where the walk over them stands, and the values converted so far.
    first = _container_values(value)
    stack: list[tuple[object, Sequence[object], Iterator[object], list[object]]] = [(value, first, iter(first), [])]
    open_ids = {id(value)}
    while True:
        container, values, pending, done = stack[-1]
        for child in pending:
            if type(child) in _NATIVE:
                done.append(child)
            elif not isinstance(child, _CONTAINERS):
                done.append(_converted(child, numpy))
            elif id(child) in open_ids or _all_native(child):
                # A container met again inside itself stays as it is, so a cycle cannot keep the walk going.
                done.append(child)
            else:
                inner = _container_values(child)
                stack.append((child, inner, iter(inner), []))
                open_ids.add(id(child))
                break
        else:
            stack.pop()
            open_ids.discard(id(container))
            unchanged = all(new is old for new, old in zip(done, values, strict=True))
            rebuilt = container if unchanged else _rebuilt(container, done)
            if not stack:
                return rebuilt
            stack[-1][3].append(rebuilt)


def _is_temporal(value: object) -> bool:
    """Whether `value` is a date, time or duration of Python's, pandas' or numpy's, which the engine module converts."""
    return isinstance(value, (datetime.date, datetime.time, datetime.timedelta)) or _is_numpy_temporal(value)


def _conversion_type_of(value: object) -> str | None:
    """The type a parameter binds as, by the engine module's own conversion.

    Refused, the value is written in instead, where the same conversion refuses it again.
    """
    try:
        return _seam.literal_type(value)
    except Error:
        return None


def _holds_temporal(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_holds_temporal(item) for item in value)
    if isinstance(value, dict):
        return any(_holds_temporal(key) or _holds_temporal(item) for key, item in value.items())
    return _is_temporal(value)


def sql_type_of(value: object) -> str | None:
    """The SQL type to bind a Python value as, or None when it is ambiguous."""
    if isinstance(value, decimal.Decimal) and _decimal_shape(value) is not None:
        # Bound alone, a decimal takes its width from its value; only a member of a container needs it written.
        return "DECIMAL"
    return _value_type(value)


def _value_type(value: object) -> str | None:
    """The full SQL type of a Python value, as a member of a container is cast to it, or None when it is ambiguous."""
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return _integer_type(value)
    if isinstance(value, float):
        return "DOUBLE"
    if isinstance(value, str):
        return "VARCHAR"
    if isinstance(value, (bytes, bytearray)):
        return "BLOB"
    if _is_nat(value):
        return None
    if isinstance(value, datetime.datetime):
        # A pandas Timestamp or other subclass carries its own unit, which only the engine's conversion types.
        if type(value) is datetime.datetime:
            return "TIMESTAMP WITH TIME ZONE" if value.utcoffset() is not None else "TIMESTAMP"
        return _conversion_type_of(value)
    if isinstance(value, datetime.date):
        return "DATE"
    if isinstance(value, datetime.time):
        # An aware time's offset may depend on a date, which only the conversion can refuse.
        return "TIME" if value.tzinfo is None else _conversion_type_of(value)
    if isinstance(value, datetime.timedelta):
        return "INTERVAL"
    if _is_numpy_temporal(value):
        return _conversion_type_of(value)
    if isinstance(value, decimal.Decimal):
        shape = _decimal_shape(value)
        # The full type: a bare DECIMAL is DECIMAL(18,3), which would round a value finer than that.
        return f"DECIMAL({shape[0]}, {shape[1]})" if shape is not None else _seam.literal_type(value)
    if isinstance(value, uuid.UUID):
        return "UUID"
    if isinstance(value, (list, tuple)):
        # A NULL takes the type of its neighbours, so it gives none of its own.
        element_types = [_value_type(item) for item in value if item is not None]
        if not element_types or any(t is None for t in element_types):
            return None
        element = _widen([t for t in element_types if t is not None])
        if element:
            return f"{element}[]"
        # Where no numeric widening holds them all, the parameter path's own conversion decides; a date or time among
        # them is refused by the rules for mixing those instead.
        return None if _holds_temporal(value) else _conversion_type_of(value)
    if isinstance(value, dict):
        # Text keys make a STRUCT and any others a MAP, the same rule `render_literal` and the C++ converter use.
        if not value:
            # An empty dict is an empty STRUCT, which has no type to bind as, so it is written into the SQL.
            return None
        if all(isinstance(k, str) for k in value):
            fields = [(k, _value_type(v)) for k, v in value.items()]
            if any(t is None for _, t in fields):
                return None
            return "STRUCT(" + ", ".join(f"{quote(k)} {t}" for k, t in fields) + ")"
        key_types = [_value_type(k) for k in value]
        value_types = [_value_type(v) for v in value.values() if v is not None]
        if not value_types or any(t is None for t in key_types) or any(t is None for t in value_types):
            return None
        key = _widen([t for t in key_types if t is not None])
        val = _widen([t for t in value_types if t is not None])
        return f"MAP({key}, {val})" if key and val else None
    return None


def _decimal_shape(value: decimal.Decimal) -> tuple[int, int] | None:
    """The width and scale of the DECIMAL that holds `value` exactly, or None when no DECIMAL(38) can."""
    if not value.is_finite():
        return None
    # Width and scale come from the value, as the parameter path reads them: a bare DECIMAL is DECIMAL(18,3) and
    # silently rounds anything finer, and a fraction holding fewer digits than its scale has an integer digit too.
    _, digits, exponent = value.as_tuple()
    places = int(exponent)
    if places >= 0:
        width, scale = len(digits) + places, 0
    else:
        scale = -places
        width = scale + 1 if scale > len(digits) else len(digits)
    return (width, scale) if len(digits) <= 38 and width <= 38 else None


def render_literal(value: object) -> str:
    """A value written into the SQL text, only where that is safe.

    A date, time or duration, alone or anywhere in a list or dict, is converted by the engine module as it would be
    bound as a parameter, so it means the same written in, and is refused where a parameter would be.
    """
    value = plain_value(value)
    if isinstance(value, (list, tuple, dict)) and _holds_temporal(value):
        _refuse_mixed_zones(value)
        _seam.literal_type(value)
    return _rendered(value)


def _refuse_mixed_zones(value: list[Any] | tuple[Any, ...] | dict[Any, Any]) -> None:
    """Refuse datetimes with and without a time zone at one place of a value, as the elements of one list.

    Written in, the engine would read the ones without a zone in the session's time zone, where bound it reads them
    as UTC, so the two forms would not mean the same.
    """
    kinds: dict[tuple[str, ...], set[bool]] = {}
    walked: set[int] = set()
    stack: list[tuple[tuple[str, ...], object]] = [((), value)]
    while stack:
        place, item = stack.pop()
        if isinstance(item, datetime.datetime):
            kinds.setdefault(place, set()).add(item.utcoffset() is not None)
        elif isinstance(item, _CONTAINERS) and id(item) not in walked:
            walked.add(id(item))
            if isinstance(item, (list, tuple)):
                stack.extend(((*place, "[]"), child) for child in item)
            elif all(isinstance(key, str) for key in item):
                stack.extend(((*place, "." + fold_name(key)), child) for key, child in item.items())
            else:
                stack.extend(((*place, "{key}"), key) for key in item)
                stack.extend(((*place, "{value}"), child) for child in item.values())
    if any(len(found) > 1 for found in kinds.values()):
        message = "a value holds datetimes with and without a time zone at one place, which no SQL type can hold"
        raise _invalid_input(message)


def _rendered(value: object) -> str:
    if value is None or _is_nat(value):
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        # Bare digits read as the parameter path types the value, except an unsigned BIGINT, which they read as HUGEINT.
        return f"CAST({value} AS UBIGINT)" if _integer_type(value) == "UBIGINT" else str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "'NaN'::DOUBLE"
        if math.isinf(value):
            return "'Infinity'::DOUBLE" if value > 0 else "'-Infinity'::DOUBLE"
        # Written bare, DuckDB would read 1.5 as DECIMAL, where a Python float is a double.
        return f"{value!r}::DOUBLE"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, (bytes, bytearray)):
        # One escape per byte: DuckDB reads exactly two hex digits after each, so one escape would cover only the first.
        escaped = "\\x" + value.hex(" ").replace(" ", "\\x") if value else ""
        return f"'{escaped}'::BLOB"
    if _is_temporal(value):
        return _seam.temporal_literal(value)
    if isinstance(value, decimal.Decimal):
        shape = _decimal_shape(value)
        if shape is None:
            # No DECIMAL holds it, so it is what binding it as a parameter makes of it, a DOUBLE or a FLOAT.
            return _seam.value_literal(value)
        width, scale = shape
        # Written out in full: DuckDB reads '1E-38' as a decimal string only into a type one digit wider.
        return f"CAST('{value:f}' AS DECIMAL({width}, {scale}))"
    if isinstance(value, uuid.UUID):
        return f"UUID '{value}'"
    if isinstance(value, Value):
        return _seam.value_literal(value)
    if isinstance(value, (list, tuple)):
        if _mixed(value):
            # The engine would type the elements one by one and fail; the parameter path converts them to one type.
            return _seam.value_literal(value)
        return _typed_as_bound(value, "[" + ", ".join(_rendered(item) for item in value) + "]")
    if isinstance(value, dict):
        # Reached only when nothing is collecting parameters; text keys make a struct, anything else a map, and an
        # empty dict is the empty map the parameter path makes of it.
        if value and all(isinstance(k, str) for k in value):
            entries = ", ".join(f"{_rendered(k)}: {_rendered(v)}" for k, v in value.items())
            return "{" + entries + "}"
        entries = ", ".join(f"{_rendered(k)}: {_rendered(v)}" for k, v in value.items())
        return _typed_as_bound(value, "MAP {" + entries + "}")
    message = f"cannot render a literal of type {type(value).__name__}: {value!r}"
    raise TypeError(message)


def _typed_as_bound(value: list[Any] | tuple[Any, ...] | dict[Any, Any], text: str) -> str:
    """A list or map literal cast to the type the parameter path gives it, where it holds an unsigned BIGINT.

    Unwritten, the engine would widen that BIGINT with its neighbours to an unsigned type where the parameter path
    widens to HUGEINT.
    """
    if not _holds_unsigned(value):
        return text
    type_text = sql_type_of(value)
    return text if type_text is None else f"CAST({text} AS {type_text})"


def _mixed(value: list[Any] | tuple[Any, ...] | dict[Any, Any]) -> bool:
    """Whether a list holds values of types no numeric widening brings together, which the parameter path converts."""
    if not isinstance(value, (list, tuple)):
        return False
    types = [sql_type_of(item) for item in value if item is not None]
    if None in types or len(set(types)) < 2 or _widen([t for t in types if t is not None]) is not None:
        return False
    return not _holds_temporal(value) and _conversion_type_of(value) is not None


def _children(value: object) -> Iterable[object]:
    """What a walk over an expression descends into: a node's fields, and the items of a list or tuple."""
    if isinstance(value, Expression):
        return vars(value).values()
    if isinstance(value, (list, tuple)):
        return value
    return ()


def string_literals(value: object) -> list[str]:
    """Every text value an expression, or a container of them, took from a bare string operand, in tree order.

    Only strings given where an expression was expected count, as in `col("a") == "x"`: not `lit("x")` or
    `ConstantExpression("x")`, not the text of `sql_expr`, not a string inside a list or a
    `duckdb.Value`, and not a name, a type or a struct field.
    """
    found: list[str] = []
    stack: list[object] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Lit):
            if item.lifted and isinstance(item.value, str):
                found.append(item.value)
            continue
        stack.extend(reversed(list(_children(item))))
    return found


# The tree.


def _coerce(other: object) -> Expression | None:
    """An Expression passes through, a literal wraps, a callable becomes a lambda, and anything else is None."""
    other = _as_python(other)
    if isinstance(other, Expression):
        return other
    if isinstance(other, PendingCase):
        raise TypeError(other.unfinished)
    if other is None or isinstance(other, LITERAL_TYPES) or _is_numpy_temporal(other):
        return Lit(other, lifted=isinstance(other, str))
    if callable(other):
        return _as_lambda(other)
    return None


def _as_lambda(function: Callable[..., object]) -> Expression:
    """A SQL lambda from a Python one, called once when built; a default would pick a different DuckDB overload."""
    label = getattr(function, "__name__", None) or repr(function)
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        message = f"cannot read the signature of {label} to build a SQL lambda from it"
        raise TypeError(message) from None
    if not parameters:
        message = f"a SQL lambda takes at least one parameter and {label} takes none; did you mean to call it?"
        raise TypeError(message)
    plain = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    for parameter in parameters:
        if parameter.kind not in plain or parameter.default is not inspect.Parameter.empty:
            message = (
                f"cannot build a SQL lambda from {label}: every parameter becomes a lambda "
                f"variable, so the signature must be plain positional names without defaults, "
                f"and `{parameter}` ({parameter.kind.description}) is not one"
            )
            raise TypeError(message)
    # Each variable has a name no column or other variable has, even one SQL folds to the same, so the Python scopes
    # hold in SQL as written.
    names = [f"{parameter.name}_{secrets.token_hex(3)}" for parameter in parameters]
    variables = [Variable(name) for name in names]
    try:
        body = function(*variables)
    except TypeError as reason:
        message = f"while building a SQL lambda from {label}: {reason}"
        raise TypeError(message) from reason
    body = _as_python(body)
    if not isinstance(body, Expression) and not _is_value(body):
        message = (
            f"the lambda returned {body!r}, which is not an expression; it runs once, at build "
            f"time, on expressions rather than values, so its body must be built from them"
        )
        raise TypeError(message)
    lifted = _lift(body)
    return Lambda(names, lifted)


def _lift(value: object) -> Expression:
    coerced = _coerce(value)
    if coerced is None:
        message = f"cannot use {value!r} as an expression operand"
        raise TypeError(message)
    return coerced


#: Stands for an argument that was not given, where None is a value in its own right.
_UNSET: Any = object()


def _under_either_name(what: str, name: str, given: object, legacy: str, alias: object) -> object:
    """One argument given under its own keyword name or under the earlier API's name for it."""
    if given is not _UNSET and alias is not _UNSET:
        message = f"{what}() got {name} twice, as {name} and as {legacy}"
        raise TypeError(message)
    if given is not _UNSET:
        return given
    if alias is not _UNSET:
        return alias
    message = f"{what}() takes {name}"
    raise TypeError(message)


def _as_condition(value: object, what: str) -> Expression:
    """A CASE condition: an expression, or any value but a string, which reads as text rather than a column."""
    plain = _as_python(value)
    if value is None:
        # Not lit(None) either: a NULL condition holds for no row, which is the mistake this refusal is for.
        message = f"{what} takes a condition, not None"
        raise TypeError(message)
    if plain is not None and not isinstance(plain, str) and _is_value(plain):
        return Lit(plain)
    return _as_expression(value, what)


def _as_expression(value: object, what: str) -> Expression:
    """An argument only an expression can fill, such as a condition, where a string would silently be a constant."""
    if isinstance(value, Expression):
        return value
    if isinstance(value, PendingCase):
        raise TypeError(value.unfinished)
    if isinstance(value, str):
        message = (
            f"{what} takes an expression, not the string {value!r}; "
            f"name a column with col() or write SQL with sql_expr()"
        )
        raise TypeError(message)
    message = f"{what} takes an expression, not {type(value).__name__}; wrap a value in lit()"
    raise TypeError(message)


def _as_expressions(values: object, what: str) -> list[Expression]:
    """One expression, or a list or tuple of them; a set has no order to give a window, so it is not several."""
    items = list(values) if isinstance(values, (list, tuple)) else [values]
    return [_as_expression(item, what) for item in items]


class FuncNamespaces:
    """The function namespace entries, on their own class because `str` and `list` shadow builtins."""

    def str(self) -> StrExpr:
        """This expression with the string functions in scope, `col("s").str().upper()`; nothing is checked or cast."""
        from duckdb._expression_namespaces import StrExpr

        return StrExpr(cast("Expression", self))

    def dt(self) -> DtExpr:
        """This expression as a date, time or timestamp: `col("placed").dt().year()`."""
        from duckdb._expression_namespaces import DtExpr

        return DtExpr(cast("Expression", self))

    def list(self) -> ListExpr:
        """This expression as a list: `col("tags").list().contains("vip")`."""
        from duckdb._expression_namespaces import ListExpr

        return ListExpr(cast("Expression", self))

    def json(self) -> JsonExpr:
        """This expression as JSON, with the string functions in scope too: `col("p").json().extract("$.id")`."""
        from duckdb._expression_namespaces import JsonExpr

        return JsonExpr(cast("Expression", self))


class _ExpressionClass(type):
    """Makes `Expression(x)` the constructor the earlier API had; the node classes construct as usual."""

    def __call__(cls, *args: object, **kwargs: object) -> Expression:
        if cls is not Expression:
            return cast("Expression", super().__call__(*args, **kwargs))
        if len(args) != 1 or kwargs:
            message = "Expression() takes one argument: a column name, or a value"
            raise TypeError(message)
        (value,) = args
        return ColumnExpression(value) if isinstance(value, str) else ConstantExpression(value)


class Expression(AggregateMethods, FuncNamespaces, metaclass=_ExpressionClass):
    """One node of an expression tree; `Expression("x")` is the column `x` and `Expression(1)` the value 1."""

    # numpy then hands an operator with a numpy scalar on its left back to the expression, which keeps the scalar,
    # where numpy would convert it to a Python value first and lose its unit.
    __array_ufunc__ = None

    def __init__(self, *parts: str) -> None:
        # `parts` types the earlier API's `Expression("x")`, which the metaclass answers; the nodes pass none.
        del parts
        self._alias: str | None = None
        self._direction: str | None = None
        self._nulls: str | None = None

    # Turning into SQL.

    def fragment(self) -> str:
        """The SQL for this node, parenthesised so precedence cannot change what it means."""
        raise NotImplementedError

    def as_select(self) -> str:
        """The SQL for a select list, carrying the alias if there is one."""
        rendered = self.fragment()
        return f"{rendered} AS {quote(self._alias)}" if self._alias is not None else rendered

    def as_order(self) -> str:
        """The SQL for an ORDER BY item, carrying the direction and the null order where they were asked for."""
        parts = (self.fragment(), self._direction, self._nulls)
        return " ".join(part for part in parts if part)

    def _with(self, **changes: object) -> Expression:
        """A copy with different presentation, its lists and dicts copied so a change cannot reach the original."""
        clone = object.__new__(type(self))
        clone.__dict__.update(
            (name, list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else value)
            for name, value in self.__dict__.items()
        )
        clone.__dict__.update(changes)
        return clone

    # Presentation.

    def alias(self, name: NewName) -> Expression:
        """Name the column in the output; an empty name removes the alias, as it did in the earlier API."""
        if not isinstance(name, str):
            message = f"a column name is a string, not {name!r}"  # type: ignore[unreachable]
            raise TypeError(message)
        return self._with(_alias=name or None)

    def asc(self) -> Expression:
        """Sort ascending. Keeps any null order already set."""
        return self._with(_direction="ASC")

    def desc(self) -> Expression:
        """Sort descending. Keeps any null order already set."""
        return self._with(_direction="DESC")

    def nulls_first(self) -> Expression:
        """Where NULLs sort. Leaves the direction as it was, so the connection's default holds until one is given."""
        return self._with(_nulls="NULLS FIRST")

    def nulls_last(self) -> Expression:
        """Where NULLs sort. Leaves the direction as it was, so the connection's default holds until one is given."""
        return self._with(_nulls="NULLS LAST")

    # Operators, all pure Python.

    def _binary(self, op: str, other: object, *, reverse: bool = False) -> Expression | None:
        # None makes the operator return NotImplemented: Python then tries the other operand, and else raises TypeError.
        coerced = _coerce(other)
        if coerced is None:
            return None
        left, right = (coerced, self) if reverse else (self, coerced)
        return Binary(op, left, right)

    def __eq__(self, other: object) -> Expression:  # type: ignore[override]
        # `col("x") == None` is a SQL NULL comparison and never true, which is SQL's answer rather than an invented one.
        return self._equality("=", other)

    def __ne__(self, other: object) -> Expression:  # type: ignore[override]
        return self._equality("!=", other)

    def _equality(self, op: str, other: object) -> Expression:
        # Raised rather than deferred: Python answers an unknown operand by identity, a False no condition can tell.
        return Binary(op, self, _lift(other))

    def __lt__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("<", other)
        return NotImplemented if built is None else built

    def __le__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("<=", other)
        return NotImplemented if built is None else built

    def __gt__(self, other: ValueOrExpr) -> Expression:
        built = self._binary(">", other)
        return NotImplemented if built is None else built

    def __ge__(self, other: ValueOrExpr) -> Expression:
        built = self._binary(">=", other)
        return NotImplemented if built is None else built

    def __add__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("+", other)
        return NotImplemented if built is None else built

    def __radd__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("+", other, reverse=True)
        return NotImplemented if built is None else built

    def __sub__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("-", other)
        return NotImplemented if built is None else built

    def __rsub__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("-", other, reverse=True)
        return NotImplemented if built is None else built

    def __mul__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("*", other)
        return NotImplemented if built is None else built

    def __rmul__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("*", other, reverse=True)
        return NotImplemented if built is None else built

    def __truediv__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("/", other)
        return NotImplemented if built is None else built

    def __rtruediv__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("/", other, reverse=True)
        return NotImplemented if built is None else built

    def __floordiv__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("//", other)
        return NotImplemented if built is None else built

    def __mod__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("%", other)
        return NotImplemented if built is None else built

    def __pow__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("**", other)
        return NotImplemented if built is None else built

    def __and__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("AND", other)
        return NotImplemented if built is None else built

    def __or__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("OR", other)
        return NotImplemented if built is None else built

    def __rfloordiv__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("//", other, reverse=True)
        return NotImplemented if built is None else built

    def __rmod__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("%", other, reverse=True)
        return NotImplemented if built is None else built

    def __rpow__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("**", other, reverse=True)
        return NotImplemented if built is None else built

    def __rand__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("AND", other, reverse=True)
        return NotImplemented if built is None else built

    def __ror__(self, other: ValueOrExpr) -> Expression:
        built = self._binary("OR", other, reverse=True)
        return NotImplemented if built is None else built

    def __invert__(self) -> Expression:
        return Unary("NOT", self)

    def __neg__(self) -> Expression:
        return Unary("-", self)

    def __getitem__(self, key: _FieldKey) -> Expression:
        """A struct field by name, a list element by position, or a map entry by key."""
        return Index(self, _as_field_key(_as_python(key)))

    def __iter__(self) -> Iterator[Any]:
        # Without this, `for x in e` would fall back to e[0], e[1] and never stop.
        message = "an expression is not iterable; index it with a field name or a position"
        raise TypeError(message)

    def __bool__(self) -> bool:
        """Refuse to be a condition: `==` builds an expression, so `if col("x") == 1:` would always be taken."""
        message = "an expression has no truth value; combine with & | ~, and test with .is_null()"
        raise TypeError(message)

    def __hash__(self) -> int:
        # Defined because __eq__ is, and an expression is not a dict key.
        return id(self)

    # Predicates and casts.

    def n_unique(self) -> Expression:
        """How many distinct values there are, ignoring NULL."""
        # count(DISTINCT x) is syntax, not a function name, so it cannot go in the table with the others.
        return Distinct("count", self)

    def concat(self, *others: ValueOrExpr) -> Expression:
        """Join text values."""
        # SQL concatenates with ||, and an expression carries no types, so + cannot tell which was meant.
        return Concat([self, *(_lift(o) for o in others)])

    def is_null(self) -> Expression:
        """Whether this is NULL. Not the same as `== None`, which gives NULL."""
        return Postfix("IS NULL", self)

    def is_not_null(self) -> Expression:
        """Whether this is not NULL."""
        return Postfix("IS NOT NULL", self)

    def isnull(self) -> Expression:
        """`is_null`, under the name of the earlier expression API."""
        return self.is_null()

    def isnotnull(self) -> Expression:
        """`is_not_null`, under the name of the earlier expression API."""
        return self.is_not_null()

    def isin(self, *values: ValueOrExpr) -> Expression:
        """Membership of several values, as `isin(1, 2)`, the form of the earlier expression API.

        One argument is one member, so a list given as one argument is compared whole; its elements are several
        arguments, and asking whether a list-typed column holds a value is `fn("list_contains", ...)`.
        """
        return self._membership("isin", values)

    def isnotin(self, *values: ValueOrExpr) -> Expression:
        """The negation of `isin`, taking the same arguments."""
        return Unary("NOT", self._membership("isnotin", values))

    def _membership(self, method: str, values: tuple[ValueOrExpr, ...]) -> Expression:
        if not values:
            message = f"Incorrect amount of parameters to '{method}', needs at least 1 parameter"
            raise _invalid_input(message)
        if len(values) > 1:
            return In(self, [_lift(v) for v in values])
        only = values[0]
        # One argument is one member, list or not; the elements of a list are given as several arguments.
        return In(self, [_lift(only)])

    def collate(self, collation: str) -> Expression:
        """Compare and sort by a collation, such as `nocase`; `nocase.noaccent` combines two."""
        return Collate(self, _as_name(collation, "collation"))

    def like(self, pattern: ValueOrExpr, *, escape: _TextValue | None = None) -> Expression:
        """Text match where `%` is any run of characters and `_` is any one; negate with `~`."""
        return Like("LIKE", self, _lift(pattern), _as_text(escape, "escape") if escape is not None else None)

    def ilike(self, pattern: ValueOrExpr, *, escape: _TextValue | None = None) -> Expression:
        """`like`, ignoring case."""
        return Like("ILIKE", self, _lift(pattern), _as_text(escape, "escape") if escape is not None else None)

    def between(
        self,
        low: ValueOrExpr = _UNSET,
        high: ValueOrExpr = _UNSET,
        *,
        lower: ValueOrExpr = _UNSET,
        upper: ValueOrExpr = _UNSET,
    ) -> Expression:
        """Inclusive range; `lower` and `upper` are the earlier API's names for the same two bounds."""
        return Between(
            self,
            _lift(_under_either_name("between", "low", low, "lower", lower)),
            _lift(_under_either_name("between", "high", high, "upper", upper)),
        )

    def cast(
        self,
        type_text: SqlText | DuckDBPyType | type = _UNSET,
        *,
        type: SqlText | DuckDBPyType | type = _UNSET,
    ) -> Expression:
        """Cast to a SQL type, a `DuckDBPyType`, or text spliced in as SQL and so never taken from untrusted input.

        `type` is the earlier API's name for the same argument.
        """
        return Cast(self, _as_type(_under_either_name("cast", "type_text", type_text, "type", type), "cast"))

    def try_cast(
        self,
        type_text: SqlText | DuckDBPyType | type = _UNSET,
        *,
        type: SqlText | DuckDBPyType | type = _UNSET,
    ) -> Expression:
        """Cast to a SQL type, giving NULL where the cast would fail; the type is taken as for `cast`."""
        chosen = _under_either_name("try_cast", "type_text", type_text, "type", type)
        return Cast(self, _as_type(chosen, "try_cast"), safe=True)

    def where(self, predicate: Expression) -> Expression:
        """Aggregate only rows where the predicate holds; goes before `.over()`, and a scalar call fails at run time."""
        if not isinstance(self, (Func, Distinct)):
            message = "where() applies to an aggregate call, as col('x').sum() or fn('sum', ...)"
            raise TypeError(message)
        return self._with(_filter=_as_condition(predicate, "where"))

    def over(
        self,
        partition_by: Exprs | None = None,
        order_by: Exprs | None = None,
        *,
        rows: tuple[int | None, int | None] | None = None,
        range: tuple[int | None, int | None] | None = None,
    ) -> Expression:
        """Turn an aggregate into a window function.

        `rows` and `range` bound it as (start, end), counted from the current row, with None meaning unbounded.
        So `rows=(-2, 0)` is the current row and the two before it.
        """
        partitions = _as_expressions(partition_by, "partition_by") if partition_by is not None else []
        orders = _as_expressions(order_by, "order_by") if order_by is not None else []
        if rows is not None and range is not None:
            message = "a window is bounded by rows or by range, not both"
            raise TypeError(message)
        frame = ("ROWS", rows) if rows is not None else ("RANGE", range) if range is not None else None
        return Over(self, partitions, orders, frame)

    def ignore_nulls(self) -> Expression:
        """Skip NULLs, for `first_value`, `last_value`, `lag`, `lead` and `nth_value`."""
        if not isinstance(self, Func):
            message = "ignore_nulls applies to a function call"
            raise TypeError(message)
        return self._with(_ignore_nulls=True)

    def _call(self, function: str, *args: object) -> Expression:
        """A function call with this expression as the first argument."""
        return Func(function, [self._as_argument(), *(_lift(a) for a in args)])

    def _call_at(self, function: str, position: int, *args: object) -> Expression:
        """A function call with this expression at `position`, for functions like `date_trunc` that take it later."""
        lifted = [_lift(a) for a in args]
        return Func(function, [*lifted[:position], self._as_argument(), *lifted[position:]])

    def _as_argument(self) -> Expression:
        # A method's receiver is the call's subject, so its alias must not make it a named argument.
        if self._alias is None and self._direction is None and self._nulls is None:
            return self
        return self._with(_alias=None, _direction=None, _nulls=None)

    def __str__(self) -> str:
        """The SQL text, with every value written in."""
        return self.fragment()

    def __repr__(self) -> str:
        return str(self)

    def get_name(self) -> str:
        """The name a query gives this expression's column: its alias, a column's own name, or else its SQL text."""
        if self._alias is not None:
            return self._alias
        if isinstance(self, Col):
            return self.parts[-1]
        return str(self)

    def show(self) -> None:
        """Print the SQL text."""
        print(self)

    def when(self, condition: ValueOrExpr, value: ValueOrExpr) -> Expression:
        """Another branch of a CASE built by `CaseExpression`, tried after the ones before it."""
        case = self._case("when")
        return case._with(branches=[*case.branches, (_as_condition(condition, "when"), _lift(value))])

    def otherwise(self, value: ValueOrExpr) -> Expression:
        """The value of a CASE built by `CaseExpression` where no branch matches, NULL until given."""
        return self._case("otherwise")._with(default=_lift(value))

    def _case(self, method: str) -> Case:
        if not isinstance(self, Case):
            message = f"{method}() can only be used on an Expression resulting from CaseExpression or when()"
            raise ValueError(message)  # noqa: TRY004
        return self


class Col(Expression):
    """A column reference: one name, or a join side's column as two parts."""

    def __init__(self, parts: tuple[str, ...]) -> None:
        super().__init__()
        self.parts = parts

    def fragment(self) -> str:
        return identifier(self.parts)


class Index(Expression):
    """A struct field, list element or map entry; the key is written into the SQL, since a field name must be."""

    def __init__(self, base: Expression, key: str | int) -> None:
        super().__init__()
        self.base = base
        self.key = key

    def fragment(self) -> str:
        return f"{self.base.fragment()}[{render_literal(self.key)}]"


class FamilyExpr(Expression):
    """The same expression with one group of functions in scope; the SQL, name and sort direction are unchanged."""

    #: Method name to (function, where the expression goes, parameter types), filled in by each generated class.
    SPEC: ClassVar[dict[str, tuple[str, int, list[str]]]] = {}

    def __init__(self, inner: Expression) -> None:
        super().__init__()
        self.inner = inner
        # A name or sort direction set before entering must survive it.
        self._alias = inner._alias
        self._direction = inner._direction
        self._nulls = inner._nulls

    def fragment(self) -> str:
        return self.inner.fragment()

    # These check what the call really is, which only the wrapped expression can answer, then wrap the result again.

    def where(self, predicate: Expression) -> Expression:
        """`Expression.where`, applied to the wrapped aggregate call."""
        return self._rewrapped(self.inner.where(predicate))

    def ignore_nulls(self) -> Expression:
        """`Expression.ignore_nulls`, applied to the wrapped function call."""
        return self._rewrapped(self.inner.ignore_nulls())

    def _rewrapped(self, inner: Expression) -> Expression:
        wrapped = type(self)(inner)
        wrapped._alias = self._alias
        wrapped._direction = self._direction
        wrapped._nulls = self._nulls
        return wrapped


class Star(Expression):
    """`*`, optionally excluding columns, each a name or a qualified name's parts, or renaming them."""

    def __init__(self, exclude: Iterable[str | tuple[str, ...]] = (), rename: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self.exclude = list(exclude)
        self.rename = dict(rename or {})

    def fragment(self) -> str:
        rendered = "*"
        if self.exclude:
            names = (quote(c) if isinstance(c, str) else identifier(c) for c in self.exclude)
            rendered += " EXCLUDE (" + ", ".join(names) + ")"
        if self.rename:
            pairs = ", ".join(f"{quote(k)} AS {quote(v)}" for k, v in self.rename.items())
            rendered += f" RENAME ({pairs})"
        return rendered


class Lit(Expression):
    """A Python value, written into the SQL."""

    def __init__(self, value: object, *, lifted: bool = False) -> None:
        super().__init__()
        #: Made from a bare string operand rather than by lit() or ConstantExpression, so a column name may be meant.
        self.lifted = lifted
        # A snapshot, so a list or dict the caller goes on changing cannot change the query with it.
        try:
            if isinstance(value, Value):
                value = Value(copy.deepcopy(plain_value(value.object)), value.type)
            elif isinstance(value, bytearray):
                value = bytes(value)
            # Plain first: a memoryview anywhere in it cannot be copied, and its bytes can.
            value = plain_value(value)
            self.value = copy.deepcopy(value) if isinstance(value, (list, tuple, dict)) else value
        except (TypeError, copy.Error) as reason:
            message = f"a literal holds a value that is not plain data: {reason}"
            raise TypeError(message) from None
        # Converted once, here, so a value its type cannot hold is refused where it is built.
        self.written = _seam.value_literal(self.value) if isinstance(self.value, Value) else None

    def fragment(self) -> str:
        if self.written is not None:
            return self.written
        return render_literal(self.value)


class Raw(Expression):
    """A piece of SQL supplied by the caller, spliced in unchanged; bare only when the engine's parser wrote it."""

    def __init__(self, sql: str, *, bare: bool = False) -> None:
        super().__init__()
        self.sql = sql
        self.bare = bare

    def fragment(self) -> str:
        return self.sql if self.bare else f"({self.sql})"


class WrittenSql(Raw):
    """An expression's SQL as the caller wrote it, checked by the parser and spliced in parentheses on its own lines.

    Its text shows as written; its name is the one the engine gives the column, the parser's rendering of the text.
    """

    def __init__(self, sql: str, name: str) -> None:
        super().__init__(sql)
        self.name = name

    def get_name(self) -> str:
        """The name a query gives this expression's column: its alias, or the engine's rendering of its text."""
        return self._alias if self._alias is not None else self.name

    def fragment(self) -> str:
        return f"(\n{self.sql}\n)"

    def __str__(self) -> str:
        return self.sql


class Binary(Expression):
    """A binary operator."""

    def __init__(self, op: str, left: Expression, right: Expression) -> None:
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def fragment(self) -> str:
        if self.op == "//":
            return f"({self.left.fragment()} // {self.right.fragment()})"
        if self.op == "**":
            return f"pow({self.left.fragment()}, {self.right.fragment()})"
        return f"({self.left.fragment()} {self.op} {self.right.fragment()})"


class Unary(Expression):
    """A prefix operator."""

    def __init__(self, op: str, operand: Expression) -> None:
        super().__init__()
        self.op = op
        self.operand = operand

    def fragment(self) -> str:
        return f"({self.op} {self.operand.fragment()})"


class Postfix(Expression):
    """A postfix operator, such as IS NULL."""

    def __init__(self, op: str, operand: Expression) -> None:
        super().__init__()
        self.op = op
        self.operand = operand

    def fragment(self) -> str:
        return f"({self.operand.fragment()} {self.op})"


class Cast(Expression):
    """An explicit cast, or a TRY_CAST that gives NULL instead of failing."""

    def __init__(self, operand: Expression, type_text: str, *, safe: bool = False) -> None:
        super().__init__()
        self.operand = operand
        self.type_text = type_text
        self.safe = safe

    def fragment(self) -> str:
        keyword = "TRY_CAST" if self.safe else "CAST"
        return f"{keyword}({self.operand.fragment()} AS {self.type_text})"


class In(Expression):
    """Membership of a value list."""

    def __init__(self, operand: Expression, values: list[Expression]) -> None:
        super().__init__()
        self.operand = operand
        self.values = values

    def fragment(self) -> str:
        rendered = ", ".join(v.fragment() for v in self.values)
        return f"({self.operand.fragment()} IN ({rendered}))"


class Like(Expression):
    """A LIKE or ILIKE match."""

    def __init__(self, keyword: str, operand: Expression, pattern: Expression, escape: str | None) -> None:
        super().__init__()
        self.keyword = keyword
        self.operand = operand
        self.pattern = pattern
        self.escape = escape

    def fragment(self) -> str:
        rendered = f"({self.operand.fragment()} {self.keyword} {self.pattern.fragment()}"
        if self.escape is not None:
            rendered += f" ESCAPE {render_literal(self.escape)}"
        return rendered + ")"


class Variable(Expression):
    """A lambda's parameter, under a name of its own: its Python name with a random suffix."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def fragment(self) -> str:
        return quote(self.name)


class Lambda(Expression):
    """A SQL lambda: `lambda x: body`, as `list_filter` and its siblings take."""

    def __init__(self, names: list[str], body: Expression) -> None:
        super().__init__()
        self.names = names
        self.body = body

    def fragment(self) -> str:
        return f"lambda {', '.join(quote(name) for name in self.names)}: {self.body.fragment()}"


class Collate(Expression):
    """A value compared and sorted by a collation."""

    def __init__(self, operand: Expression, collation: str) -> None:
        super().__init__()
        self.operand = operand
        self.collation = collation

    def fragment(self) -> str:
        return f"({self.operand.fragment()} COLLATE {quote(self.collation)})"


class Default(Expression):
    """`DEFAULT`, a column's default value where an INSERT or UPDATE takes one."""

    def fragment(self) -> str:
        return "DEFAULT"


class Untranslatable(ValueError):
    """Raised by a translator for a node it does not model, or would not answer as the engine does."""


class Between(Expression):
    """An inclusive range test."""

    def __init__(self, operand: Expression, low: Expression, high: Expression) -> None:
        super().__init__()
        self.operand = operand
        self.low = low
        self.high = high

    def fragment(self) -> str:
        return f"({self.operand.fragment()} BETWEEN {self.low.fragment()} AND {self.high.fragment()})"


class Func(Expression):
    """A function call, its named arguments after the positional ones."""

    def __init__(
        self, name: str | tuple[str, ...], args: list[Expression], named: list[tuple[str, Expression]] | None = None
    ) -> None:
        super().__init__()
        self.name = name_parts(name, "function name")
        self.args = args
        self.named = named or []
        self._ignore_nulls = False
        self._filter: Expression | None = None

    def fragment(self) -> str:
        rendered = ", ".join(
            [_argument(a, fold_name(self.name[-1])) for a in self.args]
            + [f"{quote(k)} := {v.fragment()}" for k, v in self.named]
        )
        # IGNORE NULLS goes inside the call, where DuckDB reads it; after the closing parenthesis it is a syntax error.
        tail = " IGNORE NULLS" if self._ignore_nulls else ""
        return f"{function_name(self.name)}({rendered}{tail})" + _filter_clause(self._filter)


class Coalesce(Expression):
    """COALESCE, written as the parser's own form, since it is syntax rather than a function."""

    def __init__(self, args: list[Expression]) -> None:
        super().__init__()
        self.args = args

    def fragment(self) -> str:
        return "COALESCE(" + ", ".join(a.fragment() for a in self.args) + ")"


class Distinct(Expression):
    """An aggregate over the distinct values, such as `count(DISTINCT x)`."""

    def __init__(self, name: str | tuple[str, ...], operand: Expression) -> None:
        super().__init__()
        self.name = name_parts(name, "function name")
        self.operand = operand
        self._filter: Expression | None = None

    def fragment(self) -> str:
        return f"{function_name(self.name)}(DISTINCT {self.operand.fragment()})" + _filter_clause(self._filter)


#: The functions that read an argument's name, so an alias on one of their arguments is written as `name := expr`.
_NAMING_FUNCTIONS = frozenset({"struct_pack", "struct_insert"})


def _argument(argument: Expression, function: str) -> str:
    """A positional argument, or for `struct_pack` and `struct_insert` a named one where it carries an alias."""
    if argument._alias is None or function not in _NAMING_FUNCTIONS:
        return argument.fragment()
    name = argument._alias
    bare = plain_identifier.match(name) and KEYWORD_CATEGORY.get(fold_name(name), "unreserved") == "unreserved"
    return f"{name if bare else quote(name)} := {argument.fragment()}"


def _filter_clause(predicate: Expression | None) -> str:
    """The FILTER clause of an aggregate, if it has one."""
    return f" FILTER (WHERE {predicate.fragment()})" if predicate is not None else ""


class Concat(Expression):
    """Text concatenation, which SQL spells `||`."""

    def __init__(self, parts: list[Expression]) -> None:
        super().__init__()
        self.parts = parts

    def fragment(self) -> str:
        return "(" + " || ".join(p.fragment() for p in self.parts) + ")"


def _bound(offset: int | None, *, start: bool) -> str:
    """One end of a window frame, from an offset relative to the current row."""
    if offset is None:
        return "UNBOUNDED PRECEDING" if start else "UNBOUNDED FOLLOWING"
    if offset == 0:
        return "CURRENT ROW"
    return f"{abs(int(offset))} {'PRECEDING' if offset < 0 else 'FOLLOWING'}"


class Over(Expression):
    """A window function."""

    def __init__(
        self,
        operand: Expression,
        partitions: list[Expression],
        orders: list[Expression],
        frame: tuple[str, tuple[int | None, int | None]] | None = None,
    ) -> None:
        super().__init__()
        self.operand = operand
        self.partitions = partitions
        self.orders = orders
        self.frame = frame

    def fragment(self) -> str:
        parts = []
        if self.partitions:
            parts.append("PARTITION BY " + ", ".join(p.fragment() for p in self.partitions))
        if self.orders:
            parts.append("ORDER BY " + ", ".join(o.as_order() for o in self.orders))
        if self.frame is not None:
            unit, (start, end) = self.frame
            parts.append(f"{unit} BETWEEN {_bound(start, start=True)} AND {_bound(end, start=False)}")
        return f"{self.operand.fragment()} OVER ({' '.join(parts)})"


class Case(Expression):
    """A CASE expression, built by `when(...).then(...)`."""

    def __init__(self, branches: list[tuple[Expression, Expression]], default: Expression | None = None) -> None:
        super().__init__()
        self.branches = branches
        self.default = default

    def fragment(self) -> str:
        parts = ["CASE"]
        parts += [f"WHEN {c.fragment()} THEN {r.fragment()}" for c, r in self.branches]
        if self.default is not None:
            parts.append(f"ELSE {self.default.fragment()}")
        parts.append("END")
        return "(" + " ".join(parts) + ")"


class PendingCase:
    """A CASE that `when()` began, which is an expression only once `.otherwise(value)` or `.end()` completes it."""

    #: Why it is not an expression yet, for every place that wanted one.
    unfinished: ClassVar[str]

    def _refuse(self, *_: object) -> NoReturn:
        raise TypeError(self.unfinished)

    # Without these Python would answer `p == 1` by identity and `bool(p)` as True, a filter that silently matches
    # nothing; every operator an expression has refuses with the way to finish the CASE instead.
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _refuse
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _refuse
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = _refuse
    __pow__ = __rpow__ = __and__ = __rand__ = __or__ = __ror__ = _refuse
    __invert__ = __neg__ = __getitem__ = __iter__ = __bool__ = _refuse
    __hash__ = object.__hash__


class CaseBuilder(PendingCase):
    """The half-built CASE that `when()` returns."""

    unfinished = "a CASE begun with when() needs .then(value), then .otherwise(value) or .end(), to be an expression"

    def __init__(self, branches: list[tuple[Expression, Expression]], condition: Expression) -> None:
        self._branches = branches
        self._condition = condition

    def then(self, result: ValueOrExpr) -> ThenBuilder:
        """The value for the condition just given."""
        return ThenBuilder([*self._branches, (self._condition, _lift(result))])


class ThenBuilder(PendingCase):
    """A CASE with at least one complete branch."""

    unfinished = "a CASE built with when().then() needs .otherwise(value) or .end() to be an expression"

    def __init__(self, branches: list[tuple[Expression, Expression]]) -> None:
        self._branches = branches

    def when(self, condition: ValueOrExpr) -> CaseBuilder:
        """Another condition."""
        return CaseBuilder(self._branches, _as_condition(condition, "when"))

    def otherwise(self, result: ValueOrExpr) -> Expression:
        """The fallback, completing the expression."""
        return Case(self._branches, _lift(result))

    def end(self) -> Expression:
        """Complete the expression with no fallback, so unmatched rows are NULL."""
        return Case(self._branches)


# Constructors.


def col(*parts: ColumnName) -> Expression:
    """A column by name, or by the parts of a qualified name: `col("t", "i")` is `"t"."i"`.

    Every part is quoted whole, so any name works and none is split; a struct field is `col("st")["field"]`.
    """
    for part in parts:
        if not isinstance(part, str):
            message = (  # type: ignore[unreachable]
                f"a column name is a string, not {part!r}; a qualified column is given in parts, as "
                "col('t', 'i'), and a join side's column is l[name] inside the join's on"
            )
            raise TypeError(message)
    return Col(name_parts(parts, "column name"))


def lit(value: PythonValue) -> Expression:
    """A Python value. Rarely needed, since operators accept values directly."""
    return Lit(value)


def star(exclude: ColumnNames = (), rename: _Renames | None = None) -> Expression:
    """`*`, optionally dropping or renaming columns; `exclude="id"` drops the one column `id`."""
    return Star(_as_names(exclude, "column name"), _as_renames(rename))


def fn(name: QualifiedName, /, *args: ValueOrExpr, **named: ValueOrExpr) -> Expression:
    """Any SQL function by name, a tuple for a schema-qualified one, with the arguments treated as anywhere else.

    A keyword argument passes by name: `fn("round", col("x"), precision=1)` is `round("x", "precision" := 1)`.
    """
    return Func(name, [_lift(a) for a in args], [(key, _lift(value)) for key, value in named.items()])


def sql_expr(sql: SqlText) -> Expression:
    """A raw SQL fragment, spliced in unchanged and never escaped, so never build one from untrusted input."""
    return Raw(_as_sql(sql, "sql_expr"))


def when(condition: ValueOrExpr) -> CaseBuilder:
    """Begins a CASE. Continue with `.then(...)`."""
    return CaseBuilder([], _as_condition(condition, "when"))


def coalesce(*values: ValueOrExpr) -> Expression:
    """The first argument that is not NULL."""
    return Coalesce([_lift(v) for v in values])


def count_all() -> Expression:
    """How many rows there are, as `count(*)`; unlike `col("x").count()`, NULLs are not skipped."""
    return Func("count", [Star()])


def _window(name: str) -> Callable[..., Expression]:
    def make(*args: ValueOrExpr) -> Expression:
        return Func(name, [_lift(a) for a in args])

    make.__name__ = name
    make.__doc__ = f"The `{name}` window function."
    return make


row_number = _window("row_number")
rank = _window("rank")
dense_rank = _window("dense_rank")
ntile = _window("ntile")


def _over_subject(name: str, subject: Expression, args: tuple[ValueOrExpr, ...]) -> Expression:
    return Func(name, [_as_expression(subject, name), *(_lift(a) for a in args)])


def lag(subject: Expression, /, *args: ValueOrExpr) -> Expression:
    """The `lag` window function: `subject` some rows back, `lag(col("x"))` or `lag(col("x"), 2, 0)`."""
    return _over_subject("lag", subject, args)


def lead(subject: Expression, /, *args: ValueOrExpr) -> Expression:
    """The `lead` window function: `subject` some rows ahead, `lead(col("x"))` or `lead(col("x"), 2, 0)`."""
    return _over_subject("lead", subject, args)


def first_value(subject: Expression, /, *args: ValueOrExpr) -> Expression:
    """The `first_value` window function: `subject` at the first row of the window frame."""
    return _over_subject("first_value", subject, args)


def last_value(subject: Expression, /, *args: ValueOrExpr) -> Expression:
    """The `last_value` window function: `subject` at the last row of the window frame."""
    return _over_subject("last_value", subject, args)


# The constructors of the earlier expression API, building the same tree; their names keep that API's spelling.

#: The name the expression class had in the branch this module was ported from.
Expr = Expression


def _is_value(value: object) -> bool:
    """Whether `value` is one an expression can hold as a constant."""
    value = _as_python(value)
    return value is None or isinstance(value, LITERAL_TYPES) or _is_numpy_temporal(value)


def ColumnExpression(*names: ColumnName) -> Expression:
    """A column by name, or by the parts of a qualified name: `ColumnExpression("t", "a")` is `"t"."a"`.

    A name is taken whole and never split, so `ColumnExpression("a.b")` is the column called `a.b`; the one name `*`
    is the star, as `StarExpression()` is.
    """
    if names == ("*",):
        return Star()
    return col(*names)


def ConstantExpression(value: PythonValue) -> Expression:
    """A value; a `duckdb.Value` is written as a cast to its type, and a numpy array as the list it holds."""
    return constant(value, "ConstantExpression takes a value")


def constant(value: object, refusal: str) -> Expression:
    """A value as a constant; where no constant holds it, a TypeError that says `refusal` and names its type."""
    kind = type(value)
    if kind.__module__ == "numpy" and kind.__name__ == "ndarray":
        value = cast("Any", value).tolist()
    if isinstance(value, Expression) or not _is_value(value):
        message = f"{refusal}, not {type(value).__name__}"
        raise TypeError(message)
    return Lit(value)


def FunctionExpression(function_name: QualifiedName, *args: ValueOrExpr) -> Expression:
    """A call of the function by its name, a bare string argument being a value as everywhere else."""
    return fn(function_name, *args)


def CaseExpression(condition: ValueOrExpr, value: ValueOrExpr) -> Expression:
    """`CASE WHEN condition THEN value`, extended by `.when(condition, value)` and closed by `.otherwise(value)`."""
    return Case([(_as_condition(condition, "CaseExpression"), _lift(value))], Lit(None))


def CoalesceOperator(*args: ValueOrExpr) -> Expression:
    """The first argument that is not NULL."""
    if not args:
        message = "Please provide at least one argument"
        raise _invalid_input(message)
    return coalesce(*args)


def _excluded(item: object) -> str | tuple[str, ...]:
    """One column `StarExpression` leaves out: a name, whole, or a `ColumnExpression` with up to four parts."""
    if isinstance(item, str):
        return _as_name(item, "column name")
    if not isinstance(item, Col):
        message = "Only ColumnExpressions are accepted Expression types here"
        raise ValueError(message)  # noqa: TRY004
    if len(item.parts) > 4:
        message = "Expected 1 to 4 name parts (catalog.schema.table.column) in an excluded column"
        raise ValueError(message)
    return item.parts[0] if len(item.parts) == 1 else item.parts


def StarExpression(*, exclude: Iterable[str | Expression] | None = None) -> Expression:
    """`*`, leaving out the columns in `exclude`, each a name or a `ColumnExpression` for a qualified one."""
    if exclude is None:
        return Star()
    items: Iterable[object] = (exclude,) if isinstance(exclude, str) else exclude
    return Star([_excluded(item) for item in items])


#: An expression as the engine's parser writes it back with an alias: the alias last, quoted where it must be.
_ALIASED = re.compile(r'(?P<body>.*) AS (?P<alias>"(?:[^"]|"")*"|[^\W\d]\w*)', re.DOTALL)


def SQLExpression(expression: SqlText) -> Expression:
    """One SQL expression, checked by the parser here and written as given, never from untrusted input.

    An alias in the text names the column, as `SQLExpression("sum(x) AS total")` does; the rest of the text is what
    the SQL holds, in parentheses on lines of its own.
    """
    text = _as_sql(expression, "SQLExpression")
    parsed = _seam.parse_expressions(text)
    if len(parsed) != 1:
        message = (
            f"Please provide only a single expression to SQLExpression, found {len(parsed)} expressions in the parsed"
            " string"
        )
        raise _invalid_input(message)
    aliased = _ALIASED.fullmatch(parsed[0])
    if aliased is None or not _is_one_expression(aliased["body"]):
        return WrittenSql(_expression_body(text, aliased=False), parsed[0])
    alias = aliased["alias"]
    if alias.startswith('"'):
        alias = alias[1:-1].replace('""', '"')
    return WrittenSql(_expression_body(text, aliased=True), aliased["body"]).alias(alias)


def _expression_body(text: str, *, aliased: bool) -> str:
    """The text of one expression without a trailing semicolon, and without its alias where the parser read one.

    The alias is the last token, and the AS before it when there is one.
    """
    found = text_tokens(text)
    words = []
    for position, (offset, kind) in enumerate(found):
        end = found[position + 1][0] if position + 1 < len(found) else len(text)
        if kind != _duckdb.token_type.comment:
            words.append((offset, kind, text[offset:end].strip()))
    while words and words[-1][2] == ";":
        words.pop()
    if aliased and words:
        words.pop()
        if words and words[-1][1] == _duckdb.token_type.keyword and fold_name(words[-1][2]) == "as":
            words.pop()
    if not words:
        return text.strip()
    last_offset, _, last_word = words[-1]
    return text[: last_offset + len(last_word)].strip()


def text_tokens(text: str) -> list[tuple[int, _duckdb.token_type]]:
    """The engine's tokens of a text, each with the string index it starts at; the tokenizer counts UTF-8 bytes."""
    found = _duckdb.tokenize(text)
    if text.isascii():
        return found
    index_of = [0] * (len(text.encode()) + 1)
    position = 0
    for index, char in enumerate(text):
        index_of[position] = index
        position += len(char.encode())
    index_of[position] = len(text)
    return [(index_of[offset], kind) for offset, kind in found]


def _is_one_expression(text: str) -> bool:
    """Whether the parser reads `text` as one expression without an alias, so an " AS " before it was the alias."""
    try:
        return len(_seam.parse_expressions(text)) == 1 and _ALIASED.fullmatch(text) is None
    except Error:
        return False


def _lambda_parameter(item: object) -> str:
    if isinstance(item, str):
        return _as_name(item, "lambda parameter")
    if isinstance(item, Col) and len(item.parts) == 1:
        return item.parts[0]
    message = "'lhs' was an Expression, but is not of type ColumnExpression"
    raise ValueError(message)


def LambdaExpression(lhs: str | Expression | tuple[str | Expression, ...], rhs: ValueOrExpr) -> Expression:
    """`lambda lhs: rhs`, `lhs` naming one parameter or a tuple of them, for functions such as `list_transform`."""
    if isinstance(lhs, tuple):
        names = [_lambda_parameter(item) for item in lhs]
    elif isinstance(lhs, (str, Col)):
        names = [_lambda_parameter(lhs)]
    else:
        message = "Please provide 'lhs' as either a tuple containing strings, or a single string"
        raise ValueError(message)  # noqa: TRY004
    if not names:
        message = "a lambda takes at least one parameter"
        raise ValueError(message)
    return Lambda(names, _lift(rhs))


def DefaultExpression() -> Expression:
    """`DEFAULT`, the column's default value, where an INSERT or UPDATE takes one."""
    return Default()


__all__ += [
    "CaseExpression",
    "CoalesceOperator",
    "ColumnExpression",
    "ConstantExpression",
    "DefaultExpression",
    "Expr",
    "FunctionExpression",
    "LambdaExpression",
    "SQLExpression",
    "StarExpression",
    "string_literals",
]


__all__ += [
    "PendingCase",
    "Untranslatable",
    "dense_rank",
    "first_value",
    "fold_name",
    "function_name",
    "identifier",
    "lag",
    "last_value",
    "lead",
    "name_parts",
    "ntile",
    "plain_identifier",
    "quote",
    "rank",
    "row_number",
    "sql_type_of",
]

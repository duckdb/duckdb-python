"""A predicate tree as a polars expression that a LazyFrame filter evaluates exactly as the engine would.

Polars orders NaN above every number, equates -0.0 with 0.0 and combines NULL with three valued logic, as the
engine does. A constant becomes a literal of its own engine type, never of the column's dtype; a zoned literal is
labelled with the column's zone, since polars refuses to compare across zones.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

from duckdb._sources.nodes import Binary, Col, In, Lit, Postfix, Unary, Untranslatable, exact_value

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import polars as pl

    from duckdb._sources.nodes import Node

_COMPARISONS: dict[str, Callable[[Any, Any], Any]] = {
    "=": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

_INTEGERS = {
    "Int8": "TINYINT",
    "Int16": "SMALLINT",
    "Int32": "INTEGER",
    "Int64": "BIGINT",
    "UInt8": "UTINYINT",
    "UInt16": "USMALLINT",
    "UInt32": "UINTEGER",
    "UInt64": "UBIGINT",
}

_MICROS_PER = {"ms": 1_000, "us": 1}

_NAIVE_TIMESTAMPS = {"ms": "TIMESTAMP_MS", "us": "TIMESTAMP"}


def _on_grid(count: int, unit: str) -> int | None:
    """A count of microseconds as a count of `unit`, or None when it lies between two of them."""
    step = _MICROS_PER[unit]
    return count // step if count % step == 0 else None


def to_polars(node: Node, schema: pl.Schema) -> pl.Expr:
    """`node` over `schema`, the schema of the frame, or `Untranslatable` where polars would differ."""
    try:
        return _translate(node, schema)
    except RecursionError as error:
        raise Untranslatable(node) from error


def _translate(node: Node, schema: pl.Schema) -> pl.Expr:
    import polars as pl

    match node:
        case Binary(op="AND", left=left, right=right):
            return _translate(left, schema) & _translate(right, schema)
        case Binary(op="OR", left=left, right=right):
            return _translate(left, schema) | _translate(right, schema)
        case Unary(op="NOT", operand=operand):
            return ~_translate(operand, schema)
        case Postfix(op="IS NULL", operand=Col(parts=parts)):
            return _column(schema, parts)[0].is_null()
        case Postfix(op="IS NOT NULL", operand=Col(parts=parts)):
            return _column(schema, parts)[0].is_not_null()
        case Binary(op=op, left=Col(parts=parts), right=Lit() as constant) if op in _COMPARISONS:
            column, kind = _column(schema, parts)
            if _categories(kind):
                if op not in ("=", "!="):
                    # Polars orders a Categorical or Enum by its categories, where the engine orders the strings.
                    raise Untranslatable(node)
                column, kind = column.cast(pl.String), pl.String
            literal = _literal(constant, kind)
            if constant.type == "TIMESTAMP WITH TIME ZONE":
                # Polars casts a literal down to a coarser column's unit, so the column goes up to microseconds instead.
                column = column.dt.cast_time_unit("us")
            compared: pl.Expr = _COMPARISONS[op](column, literal)
            return compared
        case In(operand=Col(parts=parts), values=values) if values:
            column, kind = _column(schema, parts)
            if _categories(kind):
                column, kind = column.cast(pl.String), pl.String
            member = _membership(column, kind, values)
            # Whether is_in answers NULL for a NULL row depends on the polars version; the engine answers NULL.
            return pl.when(column.is_null()).then(pl.lit(None, dtype=pl.Boolean)).otherwise(member)
    raise Untranslatable(node)


def _categories(kind: object) -> bool:
    """Whether `kind` is a Categorical or Enum, which the engine reads as VARCHAR, so its values compare as strings."""
    import polars as pl

    return isinstance(kind, pl.Categorical | pl.Enum)


def _engine_type(kind: Any) -> str | None:  # noqa: ANN401
    """The engine's name for the type it reads a polars dtype as, for the dtypes translated here."""
    import polars as pl

    if kind == pl.Boolean:
        return "BOOLEAN"
    if kind.is_integer():
        return _INTEGERS.get(str(kind))
    if kind == pl.Float32:
        return "FLOAT"
    if kind == pl.Float64:
        return "DOUBLE"
    if kind == pl.String:
        return "VARCHAR"
    if kind == pl.Binary:
        return "BLOB"
    if kind == pl.Date:
        return "DATE"
    if isinstance(kind, pl.Datetime) and kind.time_zone is None:
        return _NAIVE_TIMESTAMPS.get(kind.time_unit)
    if isinstance(kind, pl.Datetime) and kind.time_unit in _NAIVE_TIMESTAMPS:
        return "TIMESTAMP WITH TIME ZONE"
    if isinstance(kind, pl.Decimal):
        return f"DECIMAL({_precision(kind)},{kind.scale})"
    return None


def _precision(kind: pl.Decimal) -> int:
    """The precision a Decimal dtype exports with, which is the widest when polars leaves it open."""
    return kind.precision if kind.precision is not None else 38


def _column(schema: pl.Schema, parts: tuple[str, ...]) -> tuple[pl.Expr, Any]:
    """The column expression for `parts` and its dtype; a constant is checked against the dtype where it is used."""
    import polars as pl

    if not parts or parts[0] not in schema:
        raise Untranslatable(parts)
    # Polars reads a name like ^x$ or * as a selector, so the column is taken by position.
    column = pl.nth(schema.names().index(parts[0]))
    kind: Any = schema[parts[0]]
    for name in parts[1:]:
        fields = [field for field in kind.fields if field.name == name] if isinstance(kind, pl.Struct) else []
        # A struct field has no positional access polars can plan, so a field named like a selector is refused.
        if len(fields) != 1 or name == "*" or (name.startswith("^") and name.endswith("$")):
            raise Untranslatable(parts)
        column = column.struct.field(name)
        kind = fields[0].dtype
    return column, kind


def _literal(constant: Lit, kind: Any) -> pl.Expr:  # noqa: ANN401
    """A literal of the constant's own engine type, refusing a constant that is not a value of the column's type."""
    import polars as pl

    if constant.type != _engine_type(kind) or not exact_value(constant):
        raise Untranslatable(constant)
    if constant.type == "TIMESTAMP WITH TIME ZONE":
        return pl.lit(constant.value, dtype=pl.Int64).cast(pl.Datetime("us", kind.time_zone))
    return pl.lit(constant.value, dtype=_own_dtype(constant.type, kind))


def _own_dtype(type_text: str, kind: Any) -> Any:  # noqa: ANN401
    import polars as pl

    match type_text:
        case "BOOLEAN":
            return pl.Boolean
        case "FLOAT":
            return pl.Float32
        case "DOUBLE":
            return pl.Float64
        case "VARCHAR":
            return pl.String
        case "BLOB":
            return pl.Binary
        case "DATE":
            return pl.Date
        case "TIMESTAMP_MS":
            return pl.Datetime("ms")
        case "TIMESTAMP":
            return pl.Datetime("us")
    # The integer and decimal types match the column's dtype one to one.
    return kind


def _membership(column: pl.Expr, kind: Any, values: Sequence[Node]) -> pl.Expr:  # noqa: ANN401
    """Membership of `column` in the constants, each converted exactly to the column's own dtype."""
    import polars as pl

    held = []
    for constant in values:
        if not isinstance(constant, Lit):
            raise Untranslatable(constant)
        _literal(constant, kind)
        value = constant.value
        if isinstance(kind, pl.Datetime) and kind.time_zone is not None:
            value = _on_grid(value, kind.time_unit)  # type: ignore[arg-type]
        if value is not None:
            held.append(value)
    if not held:
        return pl.lit(False)
    if isinstance(kind, pl.Datetime) and kind.time_zone is not None:
        candidates = pl.Series(held, dtype=pl.Int64).cast(kind)
    else:
        candidates = pl.Series(held, dtype=kind)
    return column.is_in(candidates.implode())

"""A predicate tree as a pyarrow compute expression that a Dataset scanner evaluates exactly as the engine would.

A constant becomes a scalar of its own engine type, never of the column's Arrow type, so a value between two points
of a coarser column keeps its meaning; a zoned scalar is labelled with the column's zone, since pyarrow compares
across units but not across zones.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

from duckdb._sources.nodes import Binary, Col, In, Lit, Postfix, Unary, Untranslatable, exact_value

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pyarrow as pa
    import pyarrow.compute as pc

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
    "int8": "TINYINT",
    "int16": "SMALLINT",
    "int32": "INTEGER",
    "int64": "BIGINT",
    "uint8": "UTINYINT",
    "uint16": "USMALLINT",
    "uint32": "UINTEGER",
    "uint64": "UBIGINT",
}

_NAIVE_TIMESTAMPS = {"s": "TIMESTAMP_S", "ms": "TIMESTAMP_MS", "us": "TIMESTAMP"}

_UNITS = {name: unit for unit, name in _NAIVE_TIMESTAMPS.items()}

_PER_SECOND = {"s": 1, "ms": 1_000, "us": 1_000_000}

# pyarrow before 26 compares, matches and NULL tests a valid dictionary index pointing at a NULL entry as a value.
DICTIONARY_KERNELS_FIXED_IN = 26


def to_arrow(node: Node, schema: pa.Schema) -> pc.Expression:
    """`node` over `schema`, the schema the source declares, or `Untranslatable` where pyarrow would differ."""
    _refuse_unfilterable(schema)
    try:
        return _translate(node, schema)
    except RecursionError as error:
        raise Untranslatable(node) from error


def _translate(node: Node, schema: pa.Schema) -> pc.Expression:
    import pyarrow as pa
    import pyarrow.compute as pc

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
            return _column(schema, parts)[0].is_valid()
        case Binary(op=op, left=Col(parts=parts), right=Lit() as constant) if op in _COMPARISONS:
            field, kind = _column(schema, parts)
            compared = _COMPARISONS[op](field, _scalar(constant, kind))
            if op in (">", ">=") and pa.types.is_floating(kind):
                # The engine orders NaN above every number; NaN alone is unequal to itself, also in a dictionary,
                # where pyarrow has no is_nan kernel.
                return compared | (field != field)
            return compared
        case In(operand=Col(parts=parts), values=values) if values:
            field, kind = _column(schema, parts)
            member = _membership(field, kind, values)
            # pyarrow's is_in answers false for a NULL row where the engine answers NULL, which NOT turns into true.
            return pc.if_else(field.is_valid(), member, pa.scalar(None, pa.bool_()))
    raise Untranslatable(node)


def nan_may_satisfy(node: Node, schema: pa.Schema) -> bool:
    """Whether a NaN in a floating point column could make `node` true, as > and >= do and anything under NOT may."""
    import pyarrow as pa

    pending: list[tuple[Node, bool]] = [(node, False)]
    while pending:
        current, negated = pending.pop()
        match current:
            case Binary(op="AND" | "OR", left=left, right=right):
                pending.extend(((left, negated), (right, negated)))
            case Unary(op="NOT", operand=operand):
                pending.append((operand, True))
            case Binary(left=Col(parts=parts)) | In(operand=Col(parts=parts)) | Postfix(operand=Col(parts=parts)) if (
                pa.types.is_floating(_resolve(schema, parts)[1])
            ):
                if negated or (isinstance(current, Binary) and current.op in (">", ">=", "!=")):
                    return True
    return False


def _refuse_unfilterable(schema: pa.Schema) -> None:
    """Refuse a view layout or run end encoding at any depth, or a dictionary nested inside another column."""
    import pyarrow as pa

    pending = [(field.type, True) for field in schema]
    while pending:
        kind, column = pending.pop()
        if (
            pa.types.is_string_view(kind)
            or pa.types.is_binary_view(kind)
            or pa.types.is_list_view(kind)
            or pa.types.is_large_list_view(kind)
            or pa.types.is_run_end_encoded(kind)
            or (pa.types.is_dictionary(kind) and not column)
        ):
            raise Untranslatable(kind)
        if pa.types.is_dictionary(kind):
            pending.append((kind.value_type, False))
        elif isinstance(kind, pa.BaseExtensionType):
            pending.append((kind.storage_type, False))
        else:
            pending.extend((kind.field(i).type, False) for i in range(kind.num_fields))


def _engine_type(kind: pa.DataType) -> str | None:
    """The engine's name for the type it reads an Arrow value type as, for the types translated here."""
    import pyarrow as pa

    if pa.types.is_boolean(kind):
        return "BOOLEAN"
    if pa.types.is_integer(kind):
        return _INTEGERS[str(kind)]
    if pa.types.is_float32(kind):
        return "FLOAT"
    if pa.types.is_float64(kind):
        return "DOUBLE"
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return "VARCHAR"
    if pa.types.is_binary(kind) or pa.types.is_large_binary(kind) or pa.types.is_fixed_size_binary(kind):
        return "BLOB"
    # A date64 may hold part of a day, which pyarrow compares and the engine truncates.
    if pa.types.is_date32(kind):
        return "DATE"
    if pa.types.is_time(kind) and kind.unit in _PER_SECOND:
        return "TIME"
    if pa.types.is_timestamp(kind) and kind.tz is None:
        return _NAIVE_TIMESTAMPS.get(kind.unit)
    if pa.types.is_timestamp(kind) and kind.unit in _PER_SECOND:
        return "TIMESTAMP WITH TIME ZONE"
    if pa.types.is_decimal128(kind):
        return f"DECIMAL({kind.precision},{kind.scale})"
    return None


def _column(schema: pa.Schema, parts: tuple[str, ...]) -> tuple[pc.Expression, Any]:
    """The field expression for `parts` and its value type; a constant is checked against the type where it is used."""
    field, kind, encoded = _resolve(schema, parts)
    if encoded and not dictionary_kernels_exact():
        raise Untranslatable(parts)
    return field, kind


def dictionary_kernels_exact() -> bool:
    """Whether the installed pyarrow evaluates comparisons, membership and NULL tests on a dictionary column exactly."""
    import pyarrow

    return int(pyarrow.__version__.split(".", 1)[0]) >= DICTIONARY_KERNELS_FIXED_IN


def _resolve(schema: pa.Schema, parts: tuple[str, ...]) -> tuple[pc.Expression, Any, bool]:
    """The field expression for `parts`, its value type, and whether the column is dictionary encoded."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if not parts:
        raise Untranslatable(parts)
    # A missing or repeated name has index -1, so a name is never resolved to one of two columns.
    index = schema.get_field_index(parts[0])
    if index < 0:
        raise Untranslatable(parts)
    kind = schema.field(index).type
    for name in parts[1:]:
        child = kind.get_field_index(name) if pa.types.is_struct(kind) else -1
        if child < 0:
            raise Untranslatable(parts)
        kind = kind.field(child).type
    encoded = pa.types.is_dictionary(kind)
    if encoded:
        kind = kind.value_type
    # A union has no validity of its own, so whether its row is NULL is not plainly the engine's answer.
    if pa.types.is_union(kind):
        raise Untranslatable(kind)
    return pc.field(*parts), kind, encoded


def _scalar(constant: Lit, kind: pa.DataType) -> pa.Scalar:
    """A scalar of the constant's own engine type, refusing a constant that is not a value of the column's type."""
    import pyarrow as pa

    if constant.type != _engine_type(kind) or not exact_value(constant):
        raise Untranslatable(constant)
    match constant.type:
        case "BOOLEAN":
            own = pa.bool_()
        case "FLOAT":
            own = pa.float32()
        case "DOUBLE":
            own = pa.float64()
        case "VARCHAR":
            own = pa.string()
        case "BLOB":
            own = pa.binary()
        case "DATE":
            own = pa.date32()
        case "TIME":
            own = pa.time64("us")
        case "TIMESTAMP WITH TIME ZONE":
            own = pa.timestamp("us", tz=kind.tz)
        case "TIMESTAMP_S" | "TIMESTAMP_MS" | "TIMESTAMP":
            own = pa.timestamp(_UNITS[constant.type])
        case _:
            # The integer and decimal types match the column's Arrow type one to one.
            own = kind
    return pa.scalar(constant.value, own)


def _membership(field: pc.Expression, kind: pa.DataType, values: Sequence[Node]) -> pc.Expression:
    """Membership of `field` in the constants, each converted exactly to the column's own type."""
    import pyarrow as pa
    import pyarrow.compute as pc

    held = []
    zero = False
    for constant in values:
        if not isinstance(constant, Lit):
            raise Untranslatable(constant)
        value = _held(_scalar(constant, kind), kind)
        if value is None:
            continue
        if pa.types.is_floating(kind) and value == 0.0:
            zero = True
        else:
            held.append(value)
    member = pc.scalar(False) if not held else field.isin(pa.array(held, type=kind))
    if zero:
        # is_in compares the bits of a float, which tells -0.0 from 0.0 where the engine does not.
        member = (field == pa.scalar(0.0, kind)) | member
    return member


def _held(scalar: pa.Scalar, kind: pa.DataType) -> object:
    """The scalar's value as a value of the column's type, or None when no row of the column can hold it."""
    import pyarrow as pa

    if pa.types.is_timestamp(kind) and kind.tz is not None:
        return _on_grid(scalar.value, kind.unit)
    if pa.types.is_time(kind):
        return _on_grid(scalar.value, kind.unit)
    value = scalar.as_py()
    if pa.types.is_fixed_size_binary(kind) and len(value) != kind.byte_width:
        return None
    return value


def _on_grid(count: int, unit: str) -> int | None:
    """A count of microseconds as a count of `unit`, or None when it lies between two of them."""
    step = 1_000_000 // _PER_SECOND[unit]
    return count // step if count % step == 0 else None

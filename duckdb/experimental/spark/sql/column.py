import contextlib
import copy
import datetime
import decimal
import inspect
import itertools
import math
import operator
import re
import string
import struct
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
)
from typing import TYPE_CHECKING, Any, Union, cast

from ..errors import AnalysisException, ParseException, PySparkTypeError, PySparkValueError, SparkRuntimeException
from ..exception import ContributionsAcceptedError
from .types import DataType, TimestampType, _instant_to_local_time, _parse_datatype_string

if TYPE_CHECKING:
    from ._typing import DateTimeLiteral, DecimalLiteral, LiteralType

import duckdb
from duckdb import (
    CaseExpression,
    ColumnExpression,
    ConstantExpression,
    Expression,
    FunctionExpression,
    LambdaExpression,
    SQLExpression,
)
from duckdb.sqltypes import BLOB, DOUBLE, TIMESTAMP_TZ, VARCHAR, DuckDBPyType

__all__ = ["Column"]


_INTEGRAL_TYPE_IDS = frozenset(
    {"tinyint", "smallint", "integer", "bigint", "hugeint", "utinyint", "usmallint", "uinteger", "ubigint", "uhugeint"}
)


# The Spark types PySpark gives NumPy values in a literal, by dtype name.
_NUMPY_LITERAL_TYPES = {
    "bool": "BOOLEAN",
    "int8": "TINYINT",
    "int16": "SMALLINT",
    "int32": "INTEGER",
    "int64": "BIGINT",
    "float32": "FLOAT",
    "float64": "DOUBLE",
}


def _name_key(name: str) -> str:
    """The key a name is compared under: Spark folds case letter by letter, so "straße" and "STRASSE" differ."""
    return name.lower()


class _Ref:
    """A column reference inside a Spark column name, which leaves out a qualifier; see _unqualified."""

    __slots__ = ("parts",)

    def __init__(self, parts: tuple[str, ...]) -> None:
        self.parts = parts

    def render(self, columns: Collection[str] | None, references: Collection[str] = ()) -> str:
        return ".".join(self.parts if columns is None else _unqualified(self.parts, columns, references))


def _unqualified(parts: tuple[str, ...], columns: Collection[str], references: Collection[str]) -> tuple[str, ...]:
    """The parts of a column reference without a leading part that names a relation, as Spark names the column.

    ``columns`` holds the DataFrame's column names and ``references`` the renderings of the qualified references to
    them, both folded to lower case. As in Spark, a leading part is a relation where it is no column, or where it
    and the next part are a qualified reference, which Spark resolves before a struct field of a column.
    """
    if len(parts) > 1 and (
        _name_key(parts[0]) not in columns or _name_key(str(ColumnExpression(*parts[:2]))) in references
    ):
        return parts[1:]
    return parts


Name = tuple[str | _Ref, ...]


def _invalid_attribute_name(name: str) -> AnalysisException:
    return AnalysisException(
        f"[INVALID_ATTRIBUTE_NAME_SYNTAX] Syntax error in the attribute name: {name}. Check that backticks appear in "
        "pairs, a quoted string is a complete name part and use a backtick only inside quoted name parts."
    )


def _name_parts(name: str) -> tuple[str, ...]:
    """Split a Spark attribute name such as ``a.b`` or ```a.b`.c`` into its parts, with Spark's syntax rules.

    A quoted part must be a whole part, a backtick in it is doubled, and no part of an unquoted name is empty.
    """
    parts: list[str] = []
    part: list[str] = []
    quoted = False
    i = 0
    while i < len(name):
        c = name[i]
        if quoted:
            if c != "`":
                part.append(c)
            elif name[i + 1 : i + 2] == "`":
                part.append("`")
                i += 1
            elif name[i + 1 : i + 2] not in ("", "."):
                raise _invalid_attribute_name(name)
            else:
                quoted = False
        elif c == "`":
            if part:
                raise _invalid_attribute_name(name)
            quoted = True
        elif c == ".":
            if i == 0 or name[i - 1] == "." or i == len(name) - 1:
                raise _invalid_attribute_name(name)
            parts.append("".join(part))
            part = []
        else:
            part.append(c)
        i += 1
    if quoted:
        raise _invalid_attribute_name(name)
    parts.append("".join(part))
    return tuple(parts)


def _ref_name(name: str) -> Name | None:
    return None if name == "*" else (_Ref(_name_parts(name)),)


def _bare_column(name: str) -> Expression:
    """A reference to the column ``name`` without a qualifier, whatever characters the name holds."""
    return ColumnExpression(name)


def _struct_extract(expr: Expression, name: str) -> Expression:
    return FunctionExpression("struct_extract", expr, ConstantExpression(name))


def _qualified_star(parts: tuple[str, ...]) -> Expression:
    """The star of the relation or struct column that ``parts``, the parts of a name before ``.*``, name."""
    return SQLExpression(".".join('"' + part.replace('"', '""') + '"' for part in parts) + ".*")


def _column_reference(name: str) -> Expression:
    """The column, qualified column or struct field a Spark attribute name such as ```a.b`.c`` refers to.

    A name such as ``q.*`` binds only where ``select`` expands it.
    """
    if name == "*":
        return ColumnExpression(name)
    parts = _name_parts(name)
    return ColumnExpression(*parts) if len(parts) > 1 else _bare_column(parts[0])


def _shortest_float32(value: float) -> str:
    for precision in range(9):
        text = f"{value:.{precision}e}"
        with contextlib.suppress(OverflowError):
            if struct.unpack("f", struct.pack("f", float(text)))[0] == value:
                return text
    return repr(value)


def _java_float_string(value: float, single_precision: bool) -> str:
    """Formats a float the way Java's Float.toString and Double.toString do."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    text = _shortest_float32(value) if single_precision else repr(value)
    if len(decimal.Decimal(text).normalize().as_tuple().digits) == 1:
        # Java prints at least two significant digits and picks the closest such decimal.
        text = f"{value:.1e}"
    sign, digit_tuple, exponent = decimal.Decimal(text).as_tuple()
    scientific_exponent = len(digit_tuple) + cast("int", exponent) - 1
    digits = "".join(map(str, digit_tuple)).rstrip("0")
    if -3 <= scientific_exponent < 7:
        if scientific_exponent >= 0:
            integer = digits[: scientific_exponent + 1].ljust(scientific_exponent + 1, "0")
            fraction = digits[scientific_exponent + 1 :]
        else:
            integer = "0"
            fraction = "0" * (-scientific_exponent - 1) + digits
        body = f"{integer}.{fraction or '0'}"
    else:
        body = f"{digits[0]}.{digits[1:] or '0'}E{scientific_exponent}"
    return f"-{body}" if sign else body


def _is_numpy(value: object) -> bool:
    return type(value).__module__ == "numpy"


def _python_value(value: Any) -> Any:  # noqa: ANN401
    """The Python value of a NumPy scalar, which is how Spark reads one outside ``lit``; other values unchanged."""
    return value.item() if _is_numpy(value) and getattr(value, "ndim", None) == 0 else value


def _literal_name(value: Any) -> str:  # noqa: ANN401
    value = _python_value(value)
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _java_float_string(value, single_precision=False)
    if isinstance(value, decimal.Decimal):
        return format(value, "f")
    if isinstance(value, datetime.datetime):
        # Spark names a time with a zone after the local time of its instant, as collect() returns it.
        local = value if value.tzinfo is None else _instant_to_local_time(TimestampType().toInternal(value))
        text = local.isoformat(" ")
        return f"TIMESTAMP '{text.rstrip('0') if '.' in text else text}'"
    if isinstance(value, datetime.date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, (bytes, bytearray)):
        return f"X'{value.hex().upper()}'"
    if isinstance(value, (list, tuple)):
        return f"array({', '.join(_literal_name(v) for v in value)})"
    return str(value)


def _sql_name(*pieces: Any) -> Name:  # noqa: ANN401
    """Build a Spark column name from text, Columns, column references and literal values."""
    out: list[str | _Ref] = []
    for piece in pieces:
        if isinstance(piece, Column):
            # Unlike its rendering, an expression's own name leaves out the relation a column reference is bound to.
            out.extend(piece._name if piece._name is not None else (piece.expr.get_name(),))
        elif isinstance(piece, (str, _Ref)):
            out.append(piece)
        else:
            out.append(_literal_name(piece))
    return tuple(out)


def _format_name(template: str, *args: Any) -> Name:  # noqa: ANN401
    """Build a Spark column name from a template such as ``"({0} + {1})"``."""
    pieces: list[Any] = []
    for text, field, _, _ in string.Formatter().parse(template):
        pieces.append(text)
        if field is not None:
            pieces.append(args[int(field)])
    return _sql_name(*pieces)


def _comma_separated(values: Iterable[Any]) -> list[Any]:
    pieces: list[Any] = []
    for i, value in enumerate(values):
        if i:
            pieces.append(", ")
        pieces.append(value)
    return pieces


def _call_name(function: str, *args: Any) -> Name:  # noqa: ANN401
    return _sql_name(function, "(", *_comma_separated(args), ")")


def _render(name: Name, columns: Collection[str] | None = None, references: Collection[str] = ()) -> str:
    return "".join(t if isinstance(t, str) else t.render(columns, references) for t in name)


_sql_type_names = {
    "boolean": "BOOLEAN",
    "tinyint": "TINYINT",
    "smallint": "SMALLINT",
    "integer": "INT",
    "bigint": "BIGINT",
    "float": "FLOAT",
    "double": "DOUBLE",
    "varchar": "STRING",
    "blob": "BINARY",
    "date": "DATE",
    "timestamp": "TIMESTAMP_NTZ",
    "timestamp with time zone": "TIMESTAMP",
    "null": "VOID",
}


def _sql_type_name(dtype: DuckDBPyType) -> str:
    """The Spark SQL name of the atomic type a DuckDB type stands for, as used in names of casts."""
    if dtype.id == "decimal":
        (_, precision), (_, scale) = dtype.children
        return f"DECIMAL({precision},{scale})"
    return _sql_type_names.get(dtype.id, str(dtype).upper())


def _quoted_identifier(name: str) -> str:
    """``name`` as Spark 4 writes an identifier inside a column name: in backticks unless it is a plain word."""
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else "`" + name.replace("`", "``") + "`"


# The public methods of pyspark.sql.Column in PySpark 3.5.9 and 4.2.0, so that a missing one raises instead of
# becoming a struct field access.
_SPARK_COLUMN_METHODS = frozenset(
    {
        "alias",
        "asc",
        "asc_nulls_first",
        "asc_nulls_last",
        "astype",
        "between",
        "bitwiseAND",
        "bitwiseOR",
        "bitwiseXOR",
        "cast",
        "contains",
        "desc",
        "desc_nulls_first",
        "desc_nulls_last",
        "dropFields",
        "endswith",
        "eqNullSafe",
        "getField",
        "getItem",
        "ilike",
        "isNaN",
        "isNotNull",
        "isNull",
        "isin",
        "like",
        "name",
        "otherwise",
        "outer",
        "over",
        "rlike",
        "startswith",
        "substr",
        "transform",
        "try_cast",
        "when",
        "withField",
    }
)


class _MissingColumnMethodError(ContributionsAcceptedError, AttributeError):
    """A method of Spark's Column that the module lacks; an AttributeError so ``hasattr`` keeps working."""


def _constant(value: Any) -> Expression:  # noqa: ANN401
    value = _python_value(value)
    # ConstantExpression converts a float NaN to NULL, because Python value conversion treats NaN as missing data,
    # also inside a list, so lists are built from their elements.
    if isinstance(value, float) and math.isnan(value):
        return ConstantExpression("NaN").cast(DOUBLE)
    if isinstance(value, (list, tuple)):
        return FunctionExpression("list_value", *(_constant(v) for v in value))
    if isinstance(value, dict):
        msg = f"[UNSUPPORTED_FEATURE.LITERAL_TYPE] The feature is not supported: Literal for {value!r} of class dict."
        raise SparkRuntimeException(message=msg)
    if isinstance(value, datetime.datetime) and value.tzinfo is None:
        return ConstantExpression(value).cast(TIMESTAMP_TZ)
    return ConstantExpression(value)


def _numpy_literal(value: Any) -> tuple[Expression, Name, str | None]:  # noqa: ANN401
    """The expression, Spark's name and alias of ``lit`` of a NumPy value, typed by its dtype as Spark does."""
    dtype = value.dtype.name
    sql_type = "VARCHAR" if value.dtype.kind == "U" else _NUMPY_LITERAL_TYPES.get(dtype)
    if value.ndim > 1:
        msg = (
            f"[UNSUPPORTED_FEATURE.LITERAL_TYPE] The feature is not supported: Literal for {value!r} of class ndarray."
        )
        raise SparkRuntimeException(message=msg)
    if sql_type is None:
        raise PySparkTypeError(error_class="UNSUPPORTED_NUMPY_ARRAY_SCALAR", message_parameters={"dtype": dtype})
    if value.ndim == 0:
        duckdb_type = DuckDBPyType(sql_type)
        expr = _constant(value).cast(duckdb_type)
        if sql_type in ("BOOLEAN", "VARCHAR"):
            return expr, (_literal_name(value),), None
        # Spark casts the Python value of a NumPy number to its type and aliases the cast to the number's str().
        return expr, _sql_name("CAST(", value, " AS ", _sql_type_name(duckdb_type), ")"), str(value)
    elements = value.tolist()
    return _constant(elements).cast(DuckDBPyType(f"{sql_type}[]")), (_literal_name(elements),), None


def _get_expr(x: Union["Column", str]) -> Expression:
    return x.expr if isinstance(x, Column) else _constant(x)


def _lambda(body: Callable[..., Expression], *captured: Expression) -> LambdaExpression:
    """A lambda over ``body`` with parameter names that occur in none of ``captured``, the outer expressions it uses.

    So a parameter cannot shadow a column that ``captured`` refers to.
    """
    taken = " ".join(map(str, captured)).casefold()
    free = (name for name in (f"x{i}" for i in itertools.count()) if name not in taken)
    names = tuple(itertools.islice(free, len(inspect.signature(body).parameters)))
    return LambdaExpression(names[0] if len(names) == 1 else names, body(*map(ColumnExpression, names)))


def _evaluated_once(expr: Expression, body: Callable[[Expression], Expression]) -> Expression:
    """``body`` applied to ``expr``, which DuckDB evaluates once per row however often ``body`` uses it."""
    values = FunctionExpression("list_transform", FunctionExpression("list_value", expr), _lambda(body))
    return FunctionExpression("list_extract", values, ConstantExpression(1))


def _truncate(value: Expression, target: DuckDBPyType) -> Expression:
    """Cast to an integral type truncating toward zero like Spark, where DuckDB rounds to the nearest integer.

    A string must be digits with an optional fraction, as Spark without ANSI mode reads it, and otherwise raises as
    Spark in ANSI mode does.
    """
    source_type = FunctionExpression("typeof", value)
    is_string = source_type == ConstantExpression("VARCHAR")
    text = value.cast(VARCHAR)
    is_number = FunctionExpression(
        "regexp_full_match", text, ConstantExpression(r"\s*[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)\s*")
    )
    malformed = FunctionExpression(
        "concat",
        ConstantExpression("[CAST_INVALID_INPUT] The value '"),
        text,
        ConstantExpression(
            f'\' of the type "STRING" cannot be cast to "{_sql_type_name(target)}" because it is malformed.'
        ),
    )
    as_double = value.cast(DOUBLE)
    truncated = FunctionExpression("trunc", as_double)
    # A decimal or a string can have a fraction that its nearest double drops, so its whole part is cut from its text;
    # the leading 0 keeps a bare sign or an empty whole part a valid integer.
    may_drop_fraction = (
        is_string & FunctionExpression("contains", text, ConstantExpression("."))
    ) | FunctionExpression("starts_with", source_type, ConstantExpression("DECIMAL"))
    whole_part = FunctionExpression(
        "regexp_replace", text, ConstantExpression(r"^\s*([+-]?)([0-9]*)(\.[0-9]*)?\s*$"), ConstantExpression(r"\10\2")
    )
    return (
        CaseExpression(is_string & ~is_number, FunctionExpression("error", malformed))
        .when(as_double != truncated, truncated.cast(target))
        .when(may_drop_fraction, whole_part.cast(target))
        .otherwise(value.cast(target))
    )


def _to_binary(value: Expression) -> Expression:
    """Cast to binary like Spark, which takes a string's UTF-8 bytes where DuckDB reads escape sequences."""
    is_string = FunctionExpression("typeof", value) == ConstantExpression("VARCHAR")
    return CaseExpression(is_string, FunctionExpression("encode", value.cast(VARCHAR))).otherwise(value.cast(BLOB))


def _children(dtype: DuckDBPyType) -> list[DuckDBPyType]:
    if dtype.id == "map":
        return [dtype.key, dtype.value]
    if dtype.id in ("list", "array"):
        return [dtype.children[0][1]]
    return [t for _, t in dtype.children] if dtype.id == "struct" else []


def _cast_differs(dtype: DuckDBPyType) -> bool:
    """Whether a DuckDB cast to ``dtype`` can differ from Spark's: an integral, binary or struct type, or one inside."""
    return dtype.id in _INTEGRAL_TYPE_IDS or dtype.id in ("blob", "struct") or any(map(_cast_differs, _children(dtype)))


def _convert(value: Expression, target: DuckDBPyType) -> Expression:
    if not _cast_differs(target):
        return value.cast(target)
    if target.id in _INTEGRAL_TYPE_IDS:
        return _truncate(value, target)
    if target.id == "blob":
        return _to_binary(value)
    if target.id in ("list", "array"):
        (element,) = _children(target)
        return FunctionExpression("list_transform", value, _lambda(lambda e: _convert(e, element))).cast(target)
    if target.id == "map":
        key, item = _children(target)
        entry = _lambda(
            lambda kv: FunctionExpression(
                "row",
                _convert(FunctionExpression("struct_extract_at", kv, ConstantExpression(1)), key),
                _convert(FunctionExpression("struct_extract_at", kv, ConstantExpression(2)), item),
            )
        )
        entries = FunctionExpression("list_transform", FunctionExpression("map_entries", value), entry)
        return FunctionExpression("map_from_entries", entries)
    fields = _children(target)
    # Spark maps struct fields by position, where DuckDB matches them by name.
    converted = [
        _convert(FunctionExpression("struct_extract_at", value, ConstantExpression(i)), t)
        for i, t in enumerate(fields, 1)
    ]
    # The cast to a tuple binds only for a struct of exactly this many fields, and typeof keeps it from running.
    same_size = FunctionExpression("typeof", value.cast(DuckDBPyType(f"TUPLE({', '.join(['VARCHAR'] * len(fields))})")))
    is_null = value.isnull() | same_size.isnull()
    return (
        CaseExpression(is_null, ConstantExpression(None)).otherwise(FunctionExpression("row", *converted)).cast(target)
    )


def _reference(column: "Column") -> _Ref | None:
    """The column reference ``column`` is, or None for any other expression, such as a cast named after one."""
    name = column._name
    is_reference = column._output is None and name is not None and len(name) == 1 and isinstance(name[0], _Ref)
    return name[0] if is_reference else None


def _spark_cast(column: "Column", target: DuckDBPyType) -> Expression:
    """Cast like Spark, which truncates to integral types, takes UTF-8 bytes and maps struct fields by position."""
    if not _cast_differs(target):
        return column.expr.cast(target)
    # A column reference costs nothing to repeat, and evaluating anything else once makes the cast about 3 times slower.
    if _reference(column) is not None:
        return _convert(column.expr, target)
    return _evaluated_once(column.expr, lambda value: _convert(value, target))


def _duckdb_type(data_type: DataType | str, cast: bool) -> DuckDBPyType:
    """The DuckDB type of ``data_type``, read from text as a Spark type, or as a DuckDB type if Spark rejects it.

    ``cast`` reads the target of a cast, where Spark allows char and varchar.
    """
    if not isinstance(data_type, str):
        return data_type.duckdb_type
    try:
        return _parse_datatype_string(data_type, cast=cast).duckdb_type
    except ParseException as spark_error:
        try:
            return DuckDBPyType(data_type)
        except duckdb.Error:
            raise spark_error from None


def _derived(expr: Expression, name: Name | None, *sources: object, output: str | _Ref | None = None) -> "Column":
    """Wrap ``expr`` in a Column that is an aggregate if any of the Columns it was built from is one.

    It is a constant if it was built only from constant Columns and Python values.
    """
    column = Column(expr, name, output)
    column._aggregate_inputs = tuple(x for s in sources if isinstance(s, Column) for x in s._aggregate_inputs)
    if any(isinstance(s, Column) and s._is_aggregate for s in sources):
        column._is_aggregate = True
    elif all(not isinstance(s, Column) or s._is_constant for s in sources):
        column._is_constant = True
    return column


def _func_op(name: str, sql: str, doc: str = "") -> Callable[["Column"], "Column"]:
    def _(self: "Column") -> "Column":
        njc = getattr(self.expr, name)()
        return _derived(njc, _format_name(sql, self), self)

    _.__doc__ = doc
    return _


def _bin_op(
    name: str,
    sql: str,
    doc: str = "binary operator",
) -> Callable[["Column", Union["Column", "LiteralType", "DecimalLiteral", "DateTimeLiteral"]], "Column"]:
    """Create a method for given binary operator.

    ``sql`` is Spark's name for the result, with ``{0}`` for this column and ``{1}`` for the other operand.
    """

    def _(
        self: "Column",
        other: Union["Column", "LiteralType", "DecimalLiteral", "DateTimeLiteral"],
    ) -> "Column":
        jc = _get_expr(other)
        njc = getattr(self.expr, name)(jc)
        return _derived(njc, _format_name(sql, self, other), self, other)

    _.__doc__ = doc
    return _


def _division(
    divide: Callable[[Expression, Expression], Expression], sql: str, reflected: bool
) -> Callable[["Column", Any], "Column"]:
    """Divide or take a remainder like Spark without ANSI mode, which gives NULL for a zero divisor.

    DuckDB gives infinity for ``/`` by zero, and for ``%`` by zero raises on integers and gives NaN on doubles.
    """

    def _(self: "Column", other: Any) -> "Column":  # noqa: ANN401
        left, right = (_get_expr(other), self.expr) if reflected else (self.expr, _get_expr(other))
        result = divide(left, FunctionExpression("nullif", right, ConstantExpression(0)))
        return _derived(result, _format_name(sql, self, other), self, other)

    return _


def _like(name: str, sql: str) -> Callable[["Column", Any], "Column"]:
    """Match a LIKE pattern with Spark's default escape character, a backslash."""

    def _(self: "Column", other: Any) -> "Column":  # noqa: ANN401
        func = FunctionExpression(name, self.expr, _get_expr(other), ConstantExpression("\\"))
        return _derived(func, _format_name(sql, self, other), self, other)

    return _


def _bin_func(
    name: str,
    sql: str,
    doc: str = "binary function",
) -> Callable[["Column", Union["Column", "LiteralType", "DecimalLiteral", "DateTimeLiteral"]], "Column"]:
    """Create a function expression for the given binary function."""

    def _(
        self: "Column",
        other: Union["Column", "LiteralType", "DecimalLiteral", "DateTimeLiteral"],
    ) -> "Column":
        func = FunctionExpression(name, self.expr, _get_expr(other))
        return _derived(func, _format_name(sql, self, other), self, other)

    _.__doc__ = doc
    return _


class Column:
    """A column in a DataFrame.

    :class:`Column` instances can be created by::

        # 1. Select a column out of a DataFrame

        df.colName
        df["colName"]

        # 2. Create from an expression
        df.colName + 1
        1 / df.colName

    .. versionadded:: 1.3.0
    """

    # Set on Columns that contain an aggregate function, so select() knows to aggregate.
    _is_aggregate = False
    # Set on Columns that reference no column, which Spark accepts next to aggregates without a GROUP BY.
    _is_constant = False
    # The arguments of the aggregate functions in the Column, whose types decide the type Spark gives the result.
    _aggregate_inputs: tuple[Expression, ...] = ()
    # Set on aliases and struct field accesses, which Spark names struct fields after; see _struct_field.
    _field: str | None = None

    def __init__(self, expr: Expression, name: Name | None = None, output: str | _Ref | None = None) -> None:
        """Wrap an expression.

        ``name`` is Spark's name for the expression, used for it inside other names and, unless ``output`` is
        set, as its column name; without it, DuckDB names the column. ``output`` is the column name of an
        aliased expression and of a cast of one, or the column reference under a cast.
        """
        self.expr = expr
        self._name = name
        self._output = output

    def __bool__(self) -> bool:  # noqa: D105
        raise PySparkValueError(error_class="CANNOT_CONVERT_COLUMN_INTO_BOOL", message_parameters={})

    def __repr__(self) -> str:  # noqa: D105
        return f"Column<'{_render(self._name) if self._name is not None else self.expr.get_name()}'>"

    def _with_name(self, name: Name | None) -> "Column":
        named = copy.copy(self)
        named._name = name
        named._output = None
        named._field = None
        return named

    def _struct_field(self) -> str | None:
        """The name Spark gives this Column as a field of a struct, or None where Spark numbers the field."""
        reference = _reference(self)
        return reference.parts[-1] if reference is not None else self._field

    def _projection_alias(self, columns: Iterable[str], references: Collection[str] = ()) -> str | None:
        """The alias that gives this expression Spark's column name in a projection over ``columns``.

        ``references`` holds the renderings of the qualified references to the columns, folded to lower case. None
        when DuckDB already gives it Spark's name, or when Spark's name is not known.
        """
        stored = {_name_key(c): c for c in columns}
        if isinstance(self._output, _Ref):
            parts = _unqualified(self._output.parts, stored.keys(), references)
            if len(parts) == 1:
                return parts[0]
        elif self._output is not None:
            return self._output
        if self._name is None:
            return None
        reference = _reference(self)
        if reference is not None:
            # Aliasing a reference that does not resolve would turn DuckDB's "not found" error into a confusing
            # one about a select list alias referencing itself.
            name = reference.parts[-1]
            return name if stored.get(_name_key(name), name) != name else None
        return _render(self._name, stored.keys(), references)

    # arithmetic operators
    def __neg__(self) -> "Column":  # noqa: D105
        return _derived(-self.expr, _format_name("negative({0})", self), self)

    # `and`, `or`, `not` cannot be overloaded in Python,
    # so use bitwise operators as boolean operators
    __and__ = _bin_op("__and__", "({0} AND {1})")
    __or__ = _bin_op("__or__", "({0} OR {1})")
    __invert__ = _func_op("__invert__", "(NOT {0})")
    __rand__ = _bin_op("__rand__", "({0} AND {1})")
    __ror__ = _bin_op("__ror__", "({0} OR {1})")

    __add__ = _bin_op("__add__", "({0} + {1})")

    __sub__ = _bin_op("__sub__", "({0} - {1})")

    __mul__ = _bin_op("__mul__", "({0} * {1})")

    __div__ = _division(operator.truediv, "({0} / {1})", reflected=False)

    __truediv__ = _division(operator.truediv, "({0} / {1})", reflected=False)

    __mod__ = _division(operator.mod, "({0} % {1})", reflected=False)

    __pow__ = _bin_op("__pow__", "POWER({0}, {1})")

    __radd__ = _bin_op("__radd__", "({0} + {1})")

    __rsub__ = _bin_op("__rsub__", "({1} - {0})")

    __rmul__ = _bin_op("__rmul__", "({0} * {1})")

    __rdiv__ = _division(operator.truediv, "({1} / {0})", reflected=True)

    __rtruediv__ = _division(operator.truediv, "({1} / {0})", reflected=True)

    __rmod__ = _division(operator.mod, "({1} % {0})", reflected=True)

    __rpow__ = _bin_op("__rpow__", "POWER({1}, {0})")

    def __getitem__(self, k: Any) -> "Column":  # noqa: ANN401
        """An expression that gets an item at position ``ordinal`` out of a list,
        or gets an item by key out of a dict.

        .. versionadded:: 1.3.0

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Parameters
        ----------
        k
            a literal value, or a slice object without step.

        Returns:
        -------
        :class:`Column`
            Column representing the item got by key out of a dict, or substrings sliced by
            the given slice object.

        Examples:
        --------
        >>> df = spark.createDataFrame([("abcedfg", {"key": "value"})], ["l", "d"])
        >>> df.select(df.l[slice(1, 3)], df.d["key"]).show()
        +------------------+------+
        |substring(l, 1, 3)|d[key]|
        +------------------+------+
        |               abc| value|
        +------------------+------+
        """  # noqa: D205
        if isinstance(k, slice):
            raise ContributionsAcceptedError
            # if k.step is not None:
            #    raise ValueError("Using a slice with a step value is not supported")
            # return self.substr(k.start, k.stop)
        elif isinstance(k, str):
            field = _derived(_struct_extract(self.expr, k), _sql_name(self, ".", k), self)
            field._field = k
            return field
        else:
            # TODO: this is super hacky  # noqa: TD002, TD003
            expr_str = str(self.expr) + "." + str(k)
            return _derived(ColumnExpression(expr_str), _sql_name(self, "[", k, "]"), self)

    def __getattr__(self, item: Any) -> "Column":  # noqa: ANN401
        """An expression that gets an item at position ``ordinal`` out of a list,
        or gets an item by key out of a dict.

        Parameters
        ----------
        item
            a literal value.

        Returns:
        -------
        :class:`Column`
            Column representing the item got by key out of a dict.

        Examples:
        --------
        >>> df = spark.createDataFrame([("abcedfg", {"key": "value"})], ["l", "d"])
        >>> df.select(df.d.key).show()
        +------+
        |d[key]|
        +------+
        | value|
        +------+
        """  # noqa: D205
        if item.startswith("__"):
            msg = "Can not access __ (dunder) method"
            raise AttributeError(msg)
        if item in _SPARK_COLUMN_METHODS:
            msg = f"Column.{item} is not implemented"
            raise _MissingColumnMethodError(msg)
        return self[item]

    def __iter__(self) -> Iterator["Column"]:  # noqa: D105
        # Without it, iteration falls back to __getitem__ with 0, 1, ..., which never ends.
        raise PySparkTypeError(error_class="NOT_ITERABLE", message_parameters={"objectName": "Column"})

    def __contains__(self, item: object) -> bool:  # noqa: D105
        raise PySparkValueError(error_class="CANNOT_APPLY_IN_FOR_COLUMN", message_parameters={})

    def alias(self, alias: str) -> "Column":  # noqa: D102
        name = _sql_name(self, " AS ", _quoted_identifier(alias))
        aliased = _derived(self.expr.alias(alias), name, self, output=alias)
        aliased._field = alias
        return aliased

    def _case_name(self, *pieces: Any) -> Name | None:  # noqa: ANN401
        """Extend the name of a CASE expression, which ends in `` END``, with another branch."""
        if self._name is None:
            return None
        return self._name[:-1] + _sql_name(*pieces, " END")

    def when(self, condition: "Column", value: Union["Column", str]) -> "Column":  # noqa: D102
        if not isinstance(condition, Column):
            msg = "condition should be a Column"
            raise TypeError(msg)
        v = _get_expr(value)
        expr = self.expr.when(condition.expr, v)
        return _derived(expr, self._case_name(" WHEN ", condition, " THEN ", value), self, condition, value)

    def otherwise(self, value: Union["Column", str]) -> "Column":  # noqa: D102
        v = _get_expr(value)
        expr = self.expr.otherwise(v)
        return _derived(expr, self._case_name(" ELSE ", value), self, value)

    def cast(self, dataType: DataType | str) -> "Column":  # noqa: D102
        internal_type = _duckdb_type(dataType, cast=True)
        # Spark keeps the name of an aliased expression or a top level column under a cast, not of a struct field.
        output = self._output if self._output is not None else _reference(self)
        if internal_type.id in ("list", "array", "map", "struct"):
            # Spark's name of a cast to a complex type leaves the cast out, as Hive SQL has no such casts.
            name = _sql_name(self)
        else:
            name = _sql_name("CAST(", self, " AS ", _sql_type_name(internal_type), ")")
        return _derived(_spark_cast(self, internal_type), name, self, output=output)

    def isin(self, *cols: Iterable[Union["Column", str]] | Union["Column", str]) -> "Column":  # noqa: D102
        if len(cols) == 1 and isinstance(cols[0], (list, set)):
            # Only one argument supplied, it's a list
            cols = cast("tuple", cols[0])

        name = _sql_name("(", self, " IN (", *_comma_separated(cols), "))")
        if not cols:
            return _derived(ConstantExpression(False), name, self)
        return _derived(self.expr.isin(*[_get_expr(c) for c in cols]), name, self, *cols)

    # logistic operators
    def __eq__(  # type: ignore[override]
        self,
        other: Union["Column", "LiteralType", "DecimalLiteral", "DateTimeLiteral"],
    ) -> "Column":
        """Binary function."""
        return _derived(self.expr == (_get_expr(other)), _format_name("({0} = {1})", self, other), self, other)

    def __ne__(  # type: ignore[override]
        self,
        other: object,
    ) -> "Column":
        """Binary function."""
        name = _format_name("(NOT ({0} = {1}))", self, other)
        return _derived(self.expr != (_get_expr(other)), name, self, other)

    __lt__ = _bin_op("__lt__", "({0} < {1})")

    __le__ = _bin_op("__le__", "({0} <= {1})")

    __ge__ = _bin_op("__ge__", "({0} >= {1})")

    __gt__ = _bin_op("__gt__", "({0} > {1})")

    # String interrogation methods

    contains = _bin_func("contains", "contains({0}, {1})")
    rlike = _bin_func("regexp_matches", "RLIKE({0}, {1})")
    like = _like("like_escape", "{0} LIKE {1}")
    ilike = _like("ilike_escape", "ilike({0}, {1})")
    startswith = _bin_func("starts_with", "startswith({0}, {1})")
    endswith = _bin_func("suffix", "endswith({0}, {1})")

    # order
    _asc_doc = """
    Returns a sort expression based on the ascending order of the column.
    Examples
    --------
    >>> from pyspark.sql import Row
    >>> df = spark.createDataFrame([('Tom', 80), ('Alice', None)], ["name", "height"])
    >>> df.select(df.name).orderBy(df.name.asc()).collect()
    [Row(name='Alice'), Row(name='Tom')]
    """

    _asc_nulls_first_doc = """
    Returns a sort expression based on ascending order of the column, and null values
    return before non-null values.

    Examples
    --------
    >>> from pyspark.sql import Row
    >>> df = spark.createDataFrame([('Tom', 80), (None, 60), ('Alice', None)], ["name", "height"])
    >>> df.select(df.name).orderBy(df.name.asc_nulls_first()).collect()
    [Row(name=None), Row(name='Alice'), Row(name='Tom')]

    """
    _asc_nulls_last_doc = """
    Returns a sort expression based on ascending order of the column, and null values
    appear after non-null values.

    Examples
    --------
    >>> from pyspark.sql import Row
    >>> df = spark.createDataFrame([('Tom', 80), (None, 60), ('Alice', None)], ["name", "height"])
    >>> df.select(df.name).orderBy(df.name.asc_nulls_last()).collect()
    [Row(name='Alice'), Row(name='Tom'), Row(name=None)]

    """
    _desc_doc = """
    Returns a sort expression based on the descending order of the column.
    Examples
    --------
    >>> from pyspark.sql import Row
    >>> df = spark.createDataFrame([('Tom', 80), ('Alice', None)], ["name", "height"])
    >>> df.select(df.name).orderBy(df.name.desc()).collect()
    [Row(name='Tom'), Row(name='Alice')]
    """
    _desc_nulls_first_doc = """
    Returns a sort expression based on the descending order of the column, and null values
    appear before non-null values.

    Examples
    --------
    >>> from pyspark.sql import Row
    >>> df = spark.createDataFrame([('Tom', 80), (None, 60), ('Alice', None)], ["name", "height"])
    >>> df.select(df.name).orderBy(df.name.desc_nulls_first()).collect()
    [Row(name=None), Row(name='Tom'), Row(name='Alice')]

    """
    _desc_nulls_last_doc = """
    Returns a sort expression based on the descending order of the column, and null values
    appear after non-null values.

    Examples
    --------
    >>> from pyspark.sql import Row
    >>> df = spark.createDataFrame([('Tom', 80), (None, 60), ('Alice', None)], ["name", "height"])
    >>> df.select(df.name).orderBy(df.name.desc_nulls_last()).collect()
    [Row(name='Tom'), Row(name='Alice'), Row(name=None)]
    """

    def asc(self) -> "Column":  # noqa: D102
        return _derived(self.expr.asc(), _sql_name(self, " ASC", " NULLS FIRST"), self)

    def desc(self) -> "Column":  # noqa: D102
        return _derived(self.expr.desc(), _sql_name(self, " DESC", " NULLS LAST"), self)

    asc.__doc__ = _asc_doc
    desc.__doc__ = _desc_doc

    def _null_ordering_name(self, ordering: str) -> Name | None:
        if self._name is None:
            return None
        sorted_by = self._name[:-1] if self._name[-1] in (" NULLS FIRST", " NULLS LAST") else self._name
        return (*sorted_by, ordering)

    def nulls_first(self) -> "Column":  # noqa: D102
        return _derived(self.expr.nulls_first(), self._null_ordering_name(" NULLS FIRST"), self)

    def nulls_last(self) -> "Column":  # noqa: D102
        return _derived(self.expr.nulls_last(), self._null_ordering_name(" NULLS LAST"), self)

    def asc_nulls_first(self) -> "Column":  # noqa: D102
        return self.asc().nulls_first()

    def asc_nulls_last(self) -> "Column":  # noqa: D102
        return self.asc().nulls_last()

    def desc_nulls_first(self) -> "Column":  # noqa: D102
        return self.desc().nulls_first()

    def desc_nulls_last(self) -> "Column":  # noqa: D102
        return self.desc().nulls_last()

    def isNull(self) -> "Column":  # noqa: D102
        return _derived(self.expr.isnull(), _format_name("({0} IS NULL)", self), self)

    def isNotNull(self) -> "Column":  # noqa: D102
        return _derived(self.expr.isnotnull(), _format_name("({0} IS NOT NULL)", self), self)


def _column_name(name: str) -> str:
    """``name`` as the name of a column, which DuckDB cannot leave empty: it reads an empty alias as none."""
    if name == "":
        msg = "a column with an empty name is not supported"
        raise ContributionsAcceptedError(msg)
    return name


def _named_expr(column: Column, columns: Iterable[str], references: Collection[str] = ()) -> Expression:
    """The expression of ``column`` aliased to the name Spark gives it in a projection over ``columns``.

    ``references`` holds the renderings of the qualified references to the columns, folded to lower case.
    """
    if isinstance(column._output, str):
        _column_name(column._output)
    alias = column._projection_alias(columns, references)
    return column.expr if alias is None else column.expr.alias(alias)

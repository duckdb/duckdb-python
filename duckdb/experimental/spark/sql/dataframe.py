import copy
import datetime
import re
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from functools import reduce
from keyword import iskeyword
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    NoReturn,
    TypeVar,
    Union,
    cast,
    overload,
)

import duckdb
from duckdb import (
    CaseExpression,
    ColumnExpression,
    ConstantExpression,
    Expression,
    FunctionExpression,
    LambdaExpression,
    SQLExpression,
    StarExpression,
)
from duckdb.sqltypes import DuckDBPyType

from ..errors import (
    AnalysisException,
    PySparkIndexError,
    PySparkNotImplementedError,
    PySparkTypeError,
    PySparkValueError,
)
from ..exception import ContributionsAcceptedError
from .column import (
    Column,
    _bare_column,
    _column_name,
    _column_reference,
    _convert,
    _java_float_string,
    _name_key,
    _name_parts,
    _named_expr,
    _qualified_star,
    _quoted_identifier,
    _Ref,
    _ref_name,
    _reference,
    _struct_extract,
)
from .readwriter import DataFrameWriter, _spark_sql_type
from .type_utils import duckdb_to_spark_schema
from .types import (
    _INFINITE_MICROSECONDS,
    ArrayType,
    AtomicType,
    BinaryType,
    BooleanType,
    DataType,
    DateType,
    DayTimeIntervalType,
    DecimalType,
    DoubleType,
    FloatType,
    HugeIntegerType,
    LongType,
    MapType,
    NullType,
    NumericType,
    Row,
    StringType,
    StructField,
    StructType,
    TimeNSType,
    TimeNTZType,
    TimestampMillisecondNTZType,
    TimestampNanosecondNTZType,
    TimestampNanosecondType,
    TimestampNTZType,
    TimestampSecondNTZType,
    TimestampType,
    TimeType,
    UnsignedByteType,
    UnsignedHugeIntegerType,
    UnsignedIntegerType,
    UnsignedLongType,
    UnsignedShortType,
    _apply,
    _contains_type,
    _duckdb_to_python,
    _escape_meta_characters,
    _replace_types,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from pandas.core.frame import DataFrame as PandasDataFrame

    from ._typing import ColumnOrName
    from .group import GroupedData
    from .session import SparkSession

from duckdb.experimental.spark.sql import functions as spark_sql_functions

# Spark caps the row count of show() one below its largest array length.
_SHOW_MAX_ROWS = 2_147_483_631
_FULL_WIDTH = re.compile(
    "[\u1100-\u115f\u2e80-\ua4cf\uac00-\ud7a3\uf900-\ufaff\ufe10-\ufe19\ufe30-\ufe6f\uff00-\uff60\uffe0-\uffe6]"
)


def _string_half_width(text: str) -> int:
    # Spark counts UTF-16 code units and one more for each full width character.
    return len(text.encode("utf-16-le")) // 2 + len(_FULL_WIDTH.findall(text))


def _pad(text: str, width: int, align_right: bool) -> str:
    fill = " " * (width - _string_half_width(text))
    return fill + text if align_right else text + fill


def _truncate_cell(text: str, truncate: int) -> str:
    if 0 < truncate < len(text):
        return text[:truncate] if truncate < 4 else text[: truncate - 3] + "..."
    return text


def _fraction_string(microsecond: int) -> str:
    return f".{microsecond:06d}".rstrip("0") if microsecond else ""


def _day_time_interval_string(value: datetime.timedelta) -> str:
    micros = (value.days * 86_400 + value.seconds) * 1_000_000 + value.microseconds
    days, rest = divmod(abs(micros), 86_400_000_000)
    hours, rest = divmod(rest, 3_600_000_000)
    minutes, rest = divmod(rest, 60_000_000)
    seconds, microsecond = divmod(rest, 1_000_000)
    sign = "-" if micros < 0 else ""
    return (
        f"INTERVAL '{sign}{days} {hours:02d}:{minutes:02d}:{seconds:02d}{_fraction_string(microsecond)}' DAY TO SECOND"
    )


def _to_pretty_string(value: Any, data_type: DataType) -> str:  # noqa: ANN401
    """Renders a value as Spark's show() prints it."""
    if value is None:
        return "NULL"
    if isinstance(data_type, BooleanType):
        return "true" if value else "false"
    if isinstance(data_type, (FloatType, DoubleType)):
        return _java_float_string(value, isinstance(data_type, FloatType))
    if isinstance(data_type, DecimalType):
        return f"{value:f}"
    if isinstance(data_type, BinaryType):
        return "[" + " ".join(f"{byte:02X}" for byte in value) + "]"
    if isinstance(data_type, DayTimeIntervalType):
        return _day_time_interval_string(value)
    if isinstance(data_type, ArrayType):
        return "[" + ", ".join(_to_pretty_string(v, data_type.elementType) for v in value) + "]"
    if isinstance(data_type, MapType):
        # DuckDB returns maps with unhashable keys as a dict of a key list and a value list.
        nested_keys = isinstance(data_type.keyType, (ArrayType, MapType, StructType))
        entries = zip(value["key"], value["value"], strict=True) if nested_keys else value.items()
        rendered = (
            f"{_to_pretty_string(k, data_type.keyType)} -> {_to_pretty_string(v, data_type.valueType)}"
            for k, v in entries
        )
        return "{" + ", ".join(rendered) + "}"
    if isinstance(data_type, StructType):
        values = value.values() if isinstance(value, dict) else value
        return (
            "{"
            + ", ".join(_to_pretty_string(v, f.dataType) for v, f in zip(values, data_type.fields, strict=True))
            + "}"
        )
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=None).isoformat(" ", "seconds") + _fraction_string(value.microsecond)
    if isinstance(value, datetime.time):
        return value.replace(tzinfo=None).isoformat("seconds") + _fraction_string(value.microsecond)
    if isinstance(value, datetime.date):
        return value.isoformat()
    return str(value)


def _table_lines(header: list[str], cells: list[list[str]], align_right: bool) -> list[str]:
    widths = [max([3, *(_string_half_width(row[i]) for row in [header, *cells])]) for i in range(len(header))]
    separator = "+" + "+".join("-" * width for width in widths) + "+"

    def line(row: list[str]) -> str:
        return "|" + "|".join(_pad(cell, width, align_right) for cell, width in zip(row, widths, strict=True)) + "|"

    return [separator, line(header), separator, *(line(row) for row in cells), separator]


def _vertical_lines(header: list[str], cells: list[list[str]]) -> list[str]:
    name_width = max([3, *(_string_half_width(name) for name in header)])
    data_width = max([3, *(_string_half_width(cell) for row in cells for cell in row)])
    lines = []
    for i, row in enumerate(cells):
        lines.append(f"-RECORD {i}".ljust(name_width + data_width + 5, "-"))
        lines.extend(
            f" {_pad(name, name_width, False)} | {_pad(cell, data_width, False)} "
            for name, cell in zip(header, row, strict=True)
        )
    return lines


def _raise_no_columns() -> NoReturn:
    raise PySparkNotImplementedError(
        message="This DataFrame has no columns. DuckDB has no relation without columns, so only schema, "
        "columns, dtypes and printSchema are available."
    )


class _NoColumnsRelation:
    """Stands in for a relation without columns, such as the result of a statement that returns none."""

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.types: list[DuckDBPyType] = []

    def __getattr__(self, name: str) -> NoReturn:
        _raise_no_columns()

    def __contains__(self, item: str) -> NoReturn:
        _raise_no_columns()

    def __str__(self) -> str:
        return "DataFrame[]"


_T = TypeVar("_T")

_JOIN_RELATIONS = frozenset({"JOIN_RELATION", "CROSS_PRODUCT_RELATION"})
# DuckDB relations that keep their child's bindings (Relation::InheritsColumnBindings in the engine); a subquery
# keeps them when its child does.
_INHERITS_BINDINGS = frozenset({"FILTER_RELATION", "ORDER_RELATION", "LIMIT_RELATION", "DISTINCT_RELATION"})
# DuckDB relations whose columns are those of their (first) child.
_KEEPS_COLUMNS = _INHERITS_BINDINGS | {"SUBQUERY_RELATION", "SET_OPERATION_RELATION"}

# Spark's join types, after it lowers the case and drops underscores, and the DuckDB join each one is.
_JOIN_TYPES = {
    "inner": "inner",
    "cross": "inner",
    "outer": "outer",
    "full": "outer",
    "fullouter": "outer",
    "left": "left",
    "leftouter": "left",
    "right": "right",
    "rightouter": "right",
    "semi": "semi",
    "leftsemi": "semi",
    "anti": "anti",
    "leftanti": "anti",
}


def _join_type(how: str) -> str:
    join_type = _JOIN_TYPES.get(how.lower().replace("_", ""))
    if join_type is None:
        raise AnalysisException(error_class="UNSUPPORTED_JOIN_TYPE", message_parameters={"join_type": f"'{how}'"})
    return join_type


def _position(index: int) -> Expression:
    """A reference to the column at ``index`` of the relation a projection reads, which tells apart equal names."""
    return SQLExpression(f"#{index + 1}")


def _ambiguous(name: str) -> AnalysisException:
    return AnalysisException(
        error_class="AMBIGUOUS_REFERENCE",
        message_parameters={"name": f"`{name}`", "referenceNames": f"[`{name}`, `{name}`]"},
    )


def _unresolved(name: str, columns: list[str]) -> AnalysisException:
    return AnalysisException(
        error_class="UNRESOLVED_COLUMN.WITH_SUGGESTION",
        message_parameters={"objectName": f"`{name}`", "proposal": ", ".join(f"`{c}`" for c in columns)},
    )


def _distinct_names(names: list[str]) -> list[str]:
    """``names`` with each repeat of an earlier name, ignoring case, renamed to the first ``<name>_<n>`` none has.

    DuckDB's binder renames repeats in a subquery to ``<name>_<n>`` too, but only avoids the names before it, so
    a renamed column can take the name of a later column.
    """
    taken = {_name_key(name) for name in names}
    seen: set[str] = set()
    result = []
    for name in names:
        key = _name_key(name)
        if key in seen:
            suffix = 1
            while _name_key(f"{name}_{suffix}") in taken:
                suffix += 1
            name = f"{name}_{suffix}"
            taken.add(_name_key(name))
        seen.add(key)
        result.append(name)
    return result


class _BoundRef(_Ref):
    """A column reference a DataFrame handed out, such as ``df.a``, with the rendering of its qualified expression."""

    __slots__ = ("reference",)

    def __init__(self, parts: tuple[str, ...], reference: str) -> None:
        super().__init__(parts)
        self.reference = reference


def _join_input_names(names: list[str]) -> list[str]:
    """``names`` with each repeat of an earlier name, ignoring case, renamed to a name of its own.

    No column of another DataFrame has such a name, which matters because a join shows the DuckDB names of all its
    inputs to a bare column reference.
    """
    seen: set[str] = set()
    result = []
    for name in names:
        result.append(f"{name}_{uuid.uuid4().hex}" if _name_key(name) in seen else name)
        seen.add(_name_key(name))
    return result


def _check_references(
    cols: Iterable[Column],
    column_expr: Callable[[str], Expression],
    references: Callable[[], list[set[str]]],
    bindings: frozenset[str],
    check_hidden: Callable[[str], None],
) -> None:
    """Raise where a Column names a column by a name that more than one column in a scope has, or that none has.

    ``column_expr`` resolves a name in the scope and raises for an ambiguous or unresolved one; ``references`` gives
    the qualified references that name each column of the scope, whose relation names are ``bindings``;
    ``check_hidden`` raises for a name DuckDB gave a repeated column, which no Spark name refers to. As in Spark, a
    leading part that names a relation is a qualifier before it is a column.
    """
    counts: Counter[str] | None = None
    for col in cols:
        for ref in (col._name if isinstance(col, Column) else None) or ():
            if not isinstance(ref, _Ref) or ref.parts[-1] == "*":
                continue
            if isinstance(ref, _BoundRef):
                qualified, name = ref.reference, ref.parts[0]
            elif len(ref.parts) > 1 and _name_key(ref.parts[0]) in bindings:
                qualified, name = _name_key(str(ColumnExpression(*ref.parts[:2]))), ref.parts[1]
            else:
                column_expr(ref.parts[0])
                continue
            if counts is None:
                counts = Counter(reference for names in references() for reference in names)
            if counts[qualified] > 1:
                raise _ambiguous(name)
            if counts[qualified] == 0:
                check_hidden(name)


class _JoinInputs(NamedTuple):
    """The inputs of a join; the relation names they expose stay visible in the scope of the joined relation."""

    left: "DataFrame"
    right: "DataFrame"
    using: list[str]
    how: str

    def binding_names(self) -> frozenset[str]:
        return self.left._binding_names() | self.right._binding_names()

    def binding_width(self) -> int:
        """The number of columns the inputs add to the FROM clause; the binder hides a semi or anti join's right."""
        width = self.left._binding_width()
        return width if self.how in ("semi", "anti") else width + self.right._binding_width()

    def column_expr(self, name: str) -> Expression | None:
        """The expression of the joined relation's column ``name``, or None where neither input has the column."""
        if _name_key(name) in (_name_key(key) for key in self.using):
            if self.how == "outer":
                return _bare_column(name)
            return (self.right if self.how == "right" else self.left)._join_input_key_expr(name)
        outputs = (self.left,) if self.how in ("semi", "anti") else (self.left, self.right)
        sides = [side for side in outputs if side._has_column(name)]
        if len(sides) > 1:
            raise _ambiguous(name)
        if not sides:
            self.check_hidden(name)
            return None
        return sides[0]._join_input_column_expr(name)

    def check_hidden(self, name: str) -> None:
        """Raise where ``name`` is a name DuckDB gave a repeated column of an input, which no Spark name refers to."""
        if _name_key(name) in self.left._hidden_names() | self.right._hidden_names():
            raise _unresolved(name, self.column_names())

    def column_exprs(self, offset: int = 0) -> list[Expression]:
        """One expression per column of the joined relation, in its order, which tells apart equal names.

        ``offset`` is the position of the join's first input column among the columns of the FROM clause it is in.
        """
        left = self.left._join_input_column_exprs(offset)
        right = self.right._join_input_column_exprs(offset + self.left._binding_width())
        return self._output_columns(left, right, self.column_expr)

    def relation_columns(self, binding: str, offset: int) -> list[tuple[Expression, str]] | None:
        """The columns and names of the input relation called ``binding``, if there is one; see column_exprs."""
        columns = self.left._join_input_relation_columns(binding, offset)
        if columns is None and self.how not in ("semi", "anti"):
            columns = self.right._join_input_relation_columns(binding, offset + self.left._binding_width())
        return columns

    def column_names(self) -> list[str]:
        """Spark's names of the joined relation's columns; a key of a right join has the right input's name."""
        left, right = self.left.columns, self.right.columns
        if self.how != "right":
            return self._output_columns(left, right, lambda name: name)
        right_names = {_name_key(name): name for name in reversed(right)}
        return self._output_columns(left, right, lambda name: right_names[_name_key(name)])

    def column_references(self) -> list[set[str]]:
        """For each column of the joined relation, the renderings of the qualified references that name it."""
        left, right = self.left._join_input_references(), self.right._join_input_references()
        return self._output_columns(left, right, lambda name: {_name_key(str(self.column_expr(name)))})

    def _output_columns(self, left: list[_T], right: list[_T], join_key: Callable[[str], _T]) -> list[_T]:
        """The items of the joined relation's columns, given one per input column and ``join_key`` for each key."""
        names = self.left.columns
        left_keys = self._key_positions(names)
        items = [join_key(names[i]) if i in left_keys else item for i, item in enumerate(left)]
        if self.how in ("semi", "anti"):
            return items
        right_keys = self._key_positions(self.right.columns)
        return items + [item for i, item in enumerate(right) if i not in right_keys]

    def _key_positions(self, names: list[str]) -> set[int]:
        """The positions of the join keys among ``names``: a key is the first column of its name, as in Spark."""
        first = _first_positions(names)
        return {first[_name_key(key)] for key in self.using if _name_key(key) in first}


def _first_positions(names: list[str]) -> dict[str, int]:
    """The position of the first of ``names`` with each name."""
    first: dict[str, int] = {}
    for i, name in enumerate(names):
        first.setdefault(_name_key(name), i)
    return first


def _aggregate(
    df: "DataFrame", groups: list[Expression], aggregates: list[Column], group_by: str = ""
) -> duckdb.DuckDBPyRelation:
    """Aggregate the relation of ``df`` with Spark's result types."""
    if not group_by and not all(isinstance(c, Column) and (c._is_aggregate or c._is_constant) for c in aggregates):
        raise AnalysisException(error_class="MISSING_GROUP_BY", message_parameters={})
    relation = df.relation
    projections = groups + [_named_expr(c, df.columns, df._qualified_references()) for c in aggregates]
    # Without groups the relational API aggregates only when an aggregate function is present, and Spark always does.
    extra = [] if group_by or any(c._is_aggregate for c in aggregates) else [FunctionExpression("count_star")]
    rel = relation.aggregate(projections + extra, group_by)
    return _narrow_aggregates(rel, relation, len(groups), [c._aggregate_inputs for c in aggregates])


# An aggregate over one of these can be a HUGEINT without being a widened sum of a Spark integer.
_HUGEINT_SOURCES = (
    HugeIntegerType,
    UnsignedHugeIntegerType,
    UnsignedLongType,
    UnsignedIntegerType,
    UnsignedShortType,
    UnsignedByteType,
)


def _narrow_aggregates(
    relation: duckdb.DuckDBPyRelation,
    source: duckdb.DuckDBPyRelation,
    group_count: int,
    inputs: list[tuple[Expression, ...]],
) -> duckdb.DuckDBPyRelation:
    """Give aggregate results Spark's bigint where DuckDB widens integer sums to HUGEINT, which Spark lacks.

    ``inputs`` holds the arguments of the aggregate functions of each result, as expressions over ``source``. A
    result keeps its HUGEINT when one of them has a DuckDB only integer type. Columns of ``relation`` past the
    groups and those results are left out.
    """
    schema = duckdb_to_spark_schema(relation.columns, relation.types)
    fields = schema.fields[: group_count + len(inputs)]
    columns = []
    narrowed = len(fields) < len(schema.fields)
    for i, field in enumerate(fields):
        column = _position(i)
        if (
            i >= group_count
            and _contains_type(field.dataType, (HugeIntegerType,))
            and not _reads_hugeint_source(source, inputs[i - group_count])
        ):
            narrow = _replace_types(field.dataType, {HugeIntegerType: LongType()})
            column = column.cast(narrow.duckdb_type)
            narrowed = True
        columns.append(column.alias(field.name))
    return relation.select(*columns) if narrowed else relation


# The timestamp types with a time zone, which DuckDB returns to Python only with a time zone package.
_INSTANT_TYPES = (TimestampType, TimestampNanosecondType)


def _epoch_microseconds(expr: Expression, data_type: DataType) -> Expression:
    """``expr`` with each timestamp with a time zone in it, at any depth, as its microseconds since the epoch.

    An infinite timestamp becomes the largest or the smallest BIGINT, which no finite one reaches.
    """
    if not _contains_type(data_type, _INSTANT_TYPES):
        return expr
    if isinstance(data_type, _INSTANT_TYPES):
        # A cast to microseconds rounds, where DuckDB's own conversion of nanoseconds truncates.
        if isinstance(data_type, TimestampNanosecondType):
            microseconds = FunctionExpression("epoch_ns", expr) // ConstantExpression(1000)
        else:
            microseconds = FunctionExpression("epoch_us", expr)
        return (
            CaseExpression(expr == ConstantExpression("infinity"), ConstantExpression(_INFINITE_MICROSECONDS))
            .when(expr == ConstantExpression("-infinity"), ConstantExpression(-_INFINITE_MICROSECONDS))
            .otherwise(microseconds)
        )
    if isinstance(data_type, ArrayType):
        element = _epoch_microseconds(ColumnExpression("x"), data_type.elementType)
        return FunctionExpression("list_transform", expr, LambdaExpression("x", element))
    if isinstance(data_type, MapType):
        entry = ColumnExpression("e")
        key = _epoch_microseconds(_struct_extract(entry, "key"), data_type.keyType).alias("key")
        value = _epoch_microseconds(_struct_extract(entry, "value"), data_type.valueType).alias("value")
        entries = FunctionExpression(
            "list_transform",
            FunctionExpression("map_entries", expr),
            LambdaExpression("e", FunctionExpression("struct_pack", key, value)),
        )
        return FunctionExpression("map_from_entries", entries)
    fields = [
        _epoch_microseconds(_struct_extract(expr, f.name), f.dataType).alias(f.name)
        for f in cast("StructType", data_type).fields
    ]
    return CaseExpression(expr.isnull(), ConstantExpression(None)).otherwise(FunctionExpression("struct_pack", *fields))


def _reads_hugeint_source(source: duckdb.DuckDBPyRelation, inputs: tuple[Expression, ...]) -> bool:
    if not inputs:
        return False
    types = duckdb_to_spark_schema([f"c{i}" for i in range(len(inputs))], source.select(*inputs).types)
    return any(_contains_type(f.dataType, _HUGEINT_SOURCES) for f in types.fields)


class DataFrame:  # noqa: D101
    def __init__(  # noqa: D107
        self,
        relation: duckdb.DuckDBPyRelation | None,
        session: "SparkSession",
        names: list[str] | None = None,
        join_inputs: _JoinInputs | None = None,
    ) -> None:
        self.relation = relation if relation is not None else _NoColumnsRelation()
        self.session = session
        self._join_scope: tuple[duckdb.DuckDBPyRelation, _JoinInputs] | None = None
        if join_inputs is not None:
            self._join_scope = (self.relation, join_inputs)
        # For each column, the references that named it in the DataFrame this one filters, sorts or renames.
        self._inherited_references: list[set[str]] | None = None
        names = self.relation.columns if names is None else names
        if relation is not None and join_inputs is None and relation.type not in _JOIN_RELATIONS:
            # The binder renames repeated names of a relation it reads as a subquery, which Spark keeps, and can
            # rename one to the name of another column, so repeats get DuckDB names that no Spark name refers to.
            distinct = _distinct_names(names)
            if list(map(_name_key, relation.columns)) != list(map(_name_key, distinct)):
                self.relation = relation.select(*(_position(i).alias(name) for i, name in enumerate(distinct)))
        self._schema = duckdb_to_spark_schema(names, self.relation.types)
        self._name_counts = Counter(_name_key(f.name) for f in self._schema.fields)

    def show(self, n: int = 20, truncate: bool | int = True, vertical: bool = False) -> None:
        """Prints the first ``n`` rows of the DataFrame to the console.

        Parameters
        ----------
        n : int, optional
            Number of rows to show.
        truncate : bool or int, optional
            If set to ``True``, truncate strings longer than 20 chars by default.
            If set to a number greater than one, truncates long strings to length ``truncate``
            and align cells right.
        vertical : bool, optional
            If set to ``True``, print output rows vertically (one line per column value).

        Examples:
        --------
        >>> df = spark.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
        >>> df.show(2, vertical=True)
        -RECORD 0-----
         age  | 14
         name | Tom
        -RECORD 1-----
         age  | 23
         name | Alice
        only showing top 2 rows
        """
        print(self._show_string(n, truncate, vertical))

    def _show_string(self, n: int = 20, truncate: bool | int = True, vertical: bool = False) -> str:
        if not isinstance(n, int) or isinstance(n, bool):
            raise PySparkTypeError(
                error_class="NOT_INT",
                message_parameters={"arg_name": "n", "arg_type": type(n).__name__},
            )
        if not isinstance(vertical, bool):
            raise PySparkTypeError(
                error_class="NOT_BOOL",
                message_parameters={"arg_name": "vertical", "arg_type": type(vertical).__name__},
            )
        if isinstance(truncate, bool) and truncate:
            int_truncate = 20
        else:
            try:
                int_truncate = int(truncate)
            except ValueError:
                raise PySparkTypeError(
                    error_class="NOT_BOOL",
                    message_parameters={"arg_name": "truncate", "arg_type": type(truncate).__name__},
                ) from None

        num_rows = min(max(n, 0), _SHOW_MAX_ROWS)
        local = self._in_session_time_zone()
        rows = local._fetchall(num_rows + 1)
        fields = local.schema.fields
        header = [_escape_meta_characters(field.name) for field in fields]
        cells = [
            [
                _truncate_cell(_escape_meta_characters(_to_pretty_string(value, field.dataType)), int_truncate)
                for value, field in zip(row, fields, strict=True)
            ]
            for row in rows[:num_rows]
        ]
        lines = _vertical_lines(header, cells) if vertical else _table_lines(header, cells, int_truncate > 0)
        if vertical and not cells:
            lines.append("(0 rows)")
        elif len(rows) > num_rows:
            lines.append(f"only showing top {num_rows} {'row' if num_rows == 1 else 'rows'}")
        else:
            lines.append("")
        return "\n".join(lines)

    def toPandas(self) -> "PandasDataFrame":  # noqa: D102
        # relation.df() reports a missing pandas as InvalidInputException, or names numpy when both are missing.
        import pandas  # noqa: F401

        result = self._in_session_time_zone().relation.df(date_as_object=True)
        for i, field in enumerate(self.schema.fields):
            if isinstance(field.dataType, DateType):
                # DuckDB gives a missing date as NaT, and a column of missing dates the datetime64 type.
                column = result.iloc[:, i]
                result.isetitem(i, column.astype(object).where(column.notna(), None))
        # DuckDB renames repeated column names, which Spark keeps.
        result.columns = self.columns
        return result

    def _in_session_time_zone(self) -> "DataFrame":
        """This DataFrame with each timestamp as its wall clock time in the session time zone, as Spark shows it."""
        local = _replace_types(
            self.schema, {TimestampType: TimestampNTZType(), TimestampNanosecondType: TimestampNanosecondNTZType()}
        )
        return self._with_types(cast("StructType", local))

    def _with_types(self, schema: StructType) -> "DataFrame":
        """This DataFrame with each column cast to the type of the field at its position in ``schema``."""
        if schema == self.schema:
            return self
        columns = [
            expr if new == old else expr.cast(new.dataType.duckdb_type).alias(new.name)
            for expr, old, new in zip(self._column_exprs(), self.schema.fields, schema.fields, strict=True)
        ]
        return self._derive(self.relation.select(*columns))

    def toArrow(self) -> "pa.Table":
        """Returns the contents of this :class:`DataFrame` as PyArrow ``pyarrow.Table``.

        This is only available if PyArrow is installed and available.

        .. versionadded:: 4.0.0

        Notes:
        -----
        This method should only be used if the resulting PyArrow ``pyarrow.Table`` is
        expected to be small, as all the data is loaded into the driver's memory.

        This API is a developer API.

        Examples:
        --------
        >>> df.toArrow()  # doctest: +SKIP
        pyarrow.Table
        age: int64
        name: string
        ----
        age: [[2,5]]
        name: [["Alice","Bob"]]
        """
        return self.relation.to_arrow_table().rename_columns(self.columns)

    def createOrReplaceTempView(self, name: str) -> None:
        """Creates or replaces a local temporary view with this :class:`DataFrame`.

        The lifetime of this temporary table is tied to the :class:`SparkSession`
        that was used to create this :class:`DataFrame`.

        Parameters
        ----------
        name : str
            Name of the view.

        Examples:
        --------
        Create a local temporary view named 'people'.

        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], schema=["age", "name"])
        >>> df.createOrReplaceTempView("people")

        Replace the local temporary view.

        >>> df2 = df.filter(df.age > 3)
        >>> df2.createOrReplaceTempView("people")
        >>> df3 = spark.sql("SELECT * FROM people")
        >>> sorted(df3.collect()) == sorted(df2.collect())
        True
        >>> spark.catalog.dropTempView("people")
        True

        """
        relation = self.relation
        names = _distinct_names(self.columns)
        if list(map(_name_key, relation.columns)) != list(map(_name_key, names)):
            relation = self._relation_named(names)
        relation.create_view(name, True)

    def createGlobalTempView(self, name: str) -> None:  # noqa: D102
        raise NotImplementedError

    def withColumnRenamed(self, columnName: str, newName: str) -> "DataFrame":  # noqa: D102
        if not self._has_column(columnName):
            msg = f"DataFrame does not contain a column named {columnName}"
            raise ValueError(msg)
        cols = []
        for x, col in zip(self.columns, self._column_exprs(), strict=True):
            if _name_key(x) == _name_key(columnName):
                col = col.alias(_column_name(newName))
            cols.append(col)
        rel = self.relation.select(*cols)
        return self._derive(rel)

    def withColumn(self, columnName: str, col: Column) -> "DataFrame":  # noqa: D102
        if not isinstance(col, Column):
            raise PySparkTypeError(
                error_class="NOT_COLUMN",
                message_parameters={"arg_name": "col", "arg_type": type(col).__name__},
            )
        self._check_references(col)
        col = self._through_inherited_reference(col)
        if self._has_column(columnName):
            # We want to replace the existing column with this new expression
            cols = []
            for x, existing in zip(self.columns, self._column_exprs(), strict=True):
                if _name_key(x) == _name_key(columnName):
                    cols.append(col.expr.alias(_column_name(columnName)))
                else:
                    cols.append(existing)
        else:
            cols = [*self._column_exprs(), col.expr.alias(_column_name(columnName))]
        rel = self.relation.select(*cols)
        return self._derive(rel)

    def withColumns(self, *colsMap: dict[str, Column]) -> "DataFrame":
        """Returns a new :class:`DataFrame` by adding multiple columns or replacing the
        existing columns that have the same names.

        The colsMap is a map of column name and column, the column must only refer to attributes
        supplied by this Dataset. It is an error to add columns that refer to some other Dataset.

        .. versionadded:: 3.3.0
           Added support for multiple columns adding

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Parameters
        ----------
        colsMap : dict
            a dict of column name and :class:`Column`. Currently, only a single map is supported.

        Returns:
        -------
        :class:`DataFrame`
            DataFrame with new or replaced columns.

        Examples:
        --------
        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], schema=["age", "name"])
        >>> df.withColumns({"age2": df.age + 2, "age3": df.age + 3}).show()
        +---+-----+----+----+
        |age| name|age2|age3|
        +---+-----+----+----+
        |  2|Alice|   4|   5|
        |  5|  Bob|   7|   8|
        +---+-----+----+----+
        """  # noqa: D205
        # Below code is to help enable kwargs in future.
        assert len(colsMap) == 1
        colsMap = colsMap[0]  # type: ignore[assignment]

        if not isinstance(colsMap, dict):
            raise PySparkTypeError(
                error_class="NOT_DICT",
                message_parameters={
                    "arg_name": "colsMap",
                    "arg_type": type(colsMap).__name__,
                },
            )

        self._check_references(*colsMap.values())
        colsMap = {name: self._through_inherited_reference(col) for name, col in colsMap.items()}
        replacements: dict[str, tuple[str, Column]] = {}
        for name, col in colsMap.items():
            if _name_key(name) in replacements:
                raise AnalysisException(
                    error_class="COLUMN_ALREADY_EXISTS", message_parameters={"columnName": f"`{name.lower()}`"}
                )
            replacements[_name_key(name)] = (name, col)
        cols = []
        for x, existing in zip(self.columns, self._column_exprs(), strict=True):
            if _name_key(x) in replacements:
                col_name, col = replacements[_name_key(x)]
                cols.append(col.expr.alias(col_name))
            else:
                cols.append(existing)

        cols.extend(col.expr.alias(_column_name(name)) for name, col in colsMap.items() if not self._has_column(name))

        rel = self.relation.select(*cols)
        return self._derive(rel)

    def withColumnsRenamed(self, colsMap: dict[str, str]) -> "DataFrame":
        """Returns a new :class:`DataFrame` by renaming multiple columns.
        This is a no-op if the schema doesn't contain the given column names.

        .. versionadded:: 3.4.0
           Added support for multiple columns renaming

        Parameters
        ----------
        colsMap : dict
            a dict of existing column names and corresponding desired column names.
            Currently, only a single map is supported.

        Returns:
        -------
        :class:`DataFrame`
            DataFrame with renamed columns.

        See Also:
        --------
        :meth:`withColumnRenamed`

        Notes:
        -----
        Support Spark Connect

        Examples:
        --------
        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], schema=["age", "name"])
        >>> df = df.withColumns({"age2": df.age + 2, "age3": df.age + 3})
        >>> df.withColumnsRenamed({"age2": "age4", "age3": "age5"}).show()
        +---+-----+----+----+
        |age| name|age4|age5|
        +---+-----+----+----+
        |  2|Alice|   4|   5|
        |  5|  Bob|   7|   8|
        +---+-----+----+----+
        """  # noqa: D205
        if not isinstance(colsMap, dict):
            raise PySparkTypeError(
                error_class="NOT_DICT",
                message_parameters={"arg_name": "colsMap", "arg_type": type(colsMap).__name__},
            )

        unknown_columns = [name for name in colsMap if not self._has_column(name)]
        if unknown_columns:
            msg = f"DataFrame does not contain column(s): {', '.join(unknown_columns)}"
            raise ValueError(msg)

        new_names = {_name_key(old): new for old, new in colsMap.items()}
        cols = []
        for x, col in zip(self.columns, self._column_exprs(), strict=True):
            if _name_key(x) in new_names:
                col = col.alias(_column_name(new_names[_name_key(x)]))
            cols.append(col)

        rel = self.relation.select(*cols)
        return self._derive(rel)

    def transform(self, func: Callable[..., "DataFrame"], *args: Any, **kwargs: Any) -> "DataFrame":  # noqa: ANN401
        """Returns a new :class:`DataFrame`. Concise syntax for chaining custom transformations.

        .. versionadded:: 3.0.0

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Parameters
        ----------
        func : function
            a function that takes and returns a :class:`DataFrame`.
        *args
            Positional arguments to pass to func.

            .. versionadded:: 3.3.0
        **kwargs
            Keyword arguments to pass to func.

            .. versionadded:: 3.3.0

        Returns:
        -------
        :class:`DataFrame`
            Transformed DataFrame.

        Examples:
        --------
        >>> from pyspark.sql.functions import col
        >>> df = spark.createDataFrame([(1, 1.0), (2, 2.0)], ["int", "float"])
        >>> def cast_all_to_int(input_df):
        ...     return input_df.select([col(col_name).cast("int") for col_name in input_df.columns])
        >>> def sort_columns_asc(input_df):
        ...     return input_df.select(*sorted(input_df.columns))
        >>> df.transform(cast_all_to_int).transform(sort_columns_asc).show()
        +-----+---+
        |float|int|
        +-----+---+
        |    1|  1|
        |    2|  2|
        +-----+---+

        >>> def add_n(input_df, n):
        ...     return input_df.select(
        ...         [(col(col_name) + n).alias(col_name) for col_name in input_df.columns]
        ...     )
        >>> df.transform(add_n, 1).transform(add_n, n=10).show()
        +---+-----+
        |int|float|
        +---+-----+
        | 12| 12.0|
        | 13| 13.0|
        +---+-----+
        """
        result = func(self, *args, **kwargs)
        assert isinstance(result, DataFrame), (
            f"Func returned an instance of type [{type(result)}], should have been DataFrame."
        )
        return result

    def sort(self, *cols: str | Column | list[str | Column], **kwargs: Any) -> "DataFrame":  # noqa: ANN401
        """Returns a new :class:`DataFrame` sorted by the specified column(s).

        Parameters
        ----------
        cols : str, list, or :class:`Column`, optional
             list of :class:`Column` or column names to sort by.

        Other Parameters
        ----------------
        ascending : bool or list, optional, default True
            boolean or list of boolean.
            Sort ascending vs. descending. Specify list for multiple sort orders.
            If a list is specified, the length of the list must equal the length of the `cols`.

        Returns:
        -------
        :class:`DataFrame`
            Sorted DataFrame.

        Examples:
        --------
        >>> from pyspark.sql.functions import desc, asc
        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], schema=["age", "name"])

        Sort the DataFrame in ascending order.

        >>> df.sort(asc("age")).show()
        +---+-----+
        |age| name|
        +---+-----+
        |  2|Alice|
        |  5|  Bob|
        +---+-----+

        Sort the DataFrame in descending order.

        >>> df.sort(df.age.desc()).show()
        +---+-----+
        |age| name|
        +---+-----+
        |  5|  Bob|
        |  2|Alice|
        +---+-----+
        >>> df.orderBy(df.age.desc()).show()
        +---+-----+
        |age| name|
        +---+-----+
        |  5|  Bob|
        |  2|Alice|
        +---+-----+
        >>> df.sort("age", ascending=False).show()
        +---+-----+
        |age| name|
        +---+-----+
        |  5|  Bob|
        |  2|Alice|
        +---+-----+

        Specify multiple columns

        >>> df = spark.createDataFrame(
        ...     [(2, "Alice"), (2, "Bob"), (5, "Bob")], schema=["age", "name"]
        ... )
        >>> df.orderBy(desc("age"), "name").show()
        +---+-----+
        |age| name|
        +---+-----+
        |  5|  Bob|
        |  2|Alice|
        |  2|  Bob|
        +---+-----+

        Specify multiple columns for sorting order at `ascending`.

        >>> df.orderBy(["age", "name"], ascending=[False, False]).show()
        +---+-----+
        |age| name|
        +---+-----+
        |  5|  Bob|
        |  2|  Bob|
        |  2|Alice|
        +---+-----+
        """
        if not cols:
            raise PySparkValueError(
                error_class="CANNOT_BE_EMPTY",
                message_parameters={"item": "column"},
            )
        if len(cols) == 1 and isinstance(cols[0], list):
            cols = cols[0]

        columns = []
        for c in cols:
            _c = c
            if isinstance(c, str):
                _c = spark_sql_functions.col(c)
            elif isinstance(c, int) and not isinstance(c, bool):
                # ordinal is 1-based
                if c > 0:
                    _c = self[c - 1]
                # negative ordinal means sort by desc
                elif c < 0:
                    _c = self[-c - 1].desc()
                else:
                    raise PySparkIndexError(
                        error_class="ZERO_INDEX",
                        message_parameters={},
                    )
            columns.append(_c)

        ascending = kwargs.get("ascending", True)

        if isinstance(ascending, (bool, int)):
            if not ascending:
                columns = [c.desc() for c in columns]
        elif isinstance(ascending, list):
            columns = [c if asc else c.desc() for asc, c in zip(ascending, columns, strict=False)]
        else:
            raise PySparkTypeError(
                error_class="NOT_BOOL_OR_LIST",
                message_parameters={"arg_name": "ascending", "arg_type": type(ascending).__name__},
            )

        self._check_references(*columns)
        columns = [spark_sql_functions._to_column_expr(c) for c in columns]
        rel = self.relation.sort(*columns)
        return self._derive(rel)

    orderBy = sort

    def head(self, n: int | None = None) -> Row | None | list[Row]:  # noqa: D102
        if n is None:
            rs = self.head(1)
            return rs[0] if rs else None
        return self.take(n)

    first = head

    def take(self, num: int) -> list[Row]:  # noqa: D102
        return self.limit(num).collect()

    def filter(self, condition: "ColumnOrName") -> "DataFrame":
        """Filters rows using the given condition.

        :func:`where` is an alias for :func:`filter`.

        Parameters
        ----------
        condition : :class:`Column` or str
            a :class:`Column` of :class:`types.BooleanType`
            or a string of SQL expressions.

        Returns:
        -------
        :class:`DataFrame`
            Filtered DataFrame.

        Examples:
        --------
        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], schema=["age", "name"])

        Filter by :class:`Column` instances.

        >>> df.filter(df.age > 3).show()
        +---+----+
        |age|name|
        +---+----+
        |  5| Bob|
        +---+----+
        >>> df.where(df.age == 2).show()
        +---+-----+
        |age| name|
        +---+-----+
        |  2|Alice|
        +---+-----+

        Filter by SQL expression in a string.

        >>> df.filter("age > 3").show()
        +---+----+
        |age|name|
        +---+----+
        |  5| Bob|
        +---+----+
        >>> df.where("age = 2").show()
        +---+-----+
        |age| name|
        +---+-----+
        |  2|Alice|
        +---+-----+
        """
        if isinstance(condition, Column):
            self._check_references(condition)
            cond = condition.expr
        elif isinstance(condition, str):
            cond = condition
        else:
            raise PySparkTypeError(
                error_class="NOT_COLUMN_OR_STR",
                message_parameters={"arg_name": "condition", "arg_type": type(condition).__name__},
            )
        rel = self.relation.filter(cond)
        return self._derive(rel)

    where = filter

    def select(self, *cols) -> "DataFrame":  # noqa: D102
        cols = list(cols)
        if len(cols) == 1:
            cols = cols[0]
        if not isinstance(cols, list):
            cols = [cols]
        cols = [column for x in cols for column in self._selected_columns(x)]
        self._check_references(*cols)
        cols = [self._through_inherited_reference(x) for x in cols]
        if any(x._is_aggregate for x in cols):
            return DataFrame(_aggregate(self, [], cols), self.session)
        columns, references = self.columns, self._qualified_references()
        projections = [_named_expr(x, columns, references) for x in cols]
        # Spark selects no columns at all; DuckDB has no relation without columns, so drop() takes the same path.
        rel = self.relation.select(*projections) if projections else None
        return self._derive(rel)

    def _selected_columns(self, col: "ColumnOrName") -> list[Column]:
        """The Columns ``select`` gives for ``col``; a star stands for every column, under its Spark name."""
        if (isinstance(col, str) and col == "*") or (isinstance(col, Column) and str(col.expr) == "*"):
            return [Column(expr.alias(name)) for expr, name in zip(self._column_exprs(), self.columns, strict=True)]
        if isinstance(col, Column):
            reference = _reference(col)
            parts = () if reference is None else reference.parts
        else:
            parts = _name_parts(col)
        if len(parts) < 2 or parts[-1] != "*":
            return [col] if isinstance(col, Column) else [Column(self._scope_column_expr(col), _ref_name(col))]
        columns = self._relation_columns(_name_key(parts[0])) if len(parts) == 2 else None
        if columns is not None:
            return [Column(expr.alias(name)) for expr, name in columns]
        # A struct column, whose fields DuckDB names as Spark does; a repeated name raises, unless it is a qualifier.
        if _name_key(parts[0]) not in self._check_bindings():
            self._name_column_expr(parts[0])
        return [Column(_qualified_star(parts[:-1]))]

    def _relation_columns(self, binding: str) -> list[tuple[Expression, str]] | None:
        """The columns and names of the relation called ``binding`` in this DataFrame's scope, if there is one."""
        if self._join_inputs is not None:
            return self._join_inputs.relation_columns(binding, 0)
        if binding != _name_key(self.relation.alias):
            return None
        return list(zip(self._column_exprs(), self.columns, strict=True))

    @property
    def columns(self) -> list[str]:
        """Returns all column names as a list.

        Examples:
        --------
        >>> df.columns
        ['age', 'name']
        """
        return [f.name for f in self.schema.fields]

    @property
    def dtypes(self) -> list[tuple[str, str]]:
        """Returns all column names and their data types as a list of tuples.

        Returns:
        -------
        list of tuple
            List of tuples, each tuple containing a column name and its data type as strings.

        Examples:
        --------
        >>> df.dtypes
        [('age', 'bigint'), ('name', 'string')]
        """
        return [(f.name, f.dataType.simpleString()) for f in self.schema.fields]

    def _ipython_key_completions_(self) -> list[str]:
        # Provides tab-completion for column names in PySpark DataFrame
        # when accessed in bracket notation, e.g. df['<TAB>]
        return self.columns

    def __dir__(self) -> list[str]:  # noqa: D105
        out = set(super().__dir__())
        out.update(c for c in self.columns if c.isidentifier() and not iskeyword(c))
        return sorted(out)

    def join(
        self,
        other: "DataFrame",
        on: str | list[str] | Column | list[Column] | None = None,
        how: str | None = None,
    ) -> "DataFrame":
        """Joins with another :class:`DataFrame`, using the given join expression.

        Parameters
        ----------
        other : :class:`DataFrame`
            Right side of the join
        on : str, list or :class:`Column`, optional
            a string for the join column name, a list of column names,
            a join expression (Column), or a list of Columns.
            If `on` is a string or a list of strings indicating the name of the join column(s),
            the column(s) must exist on both sides, and this performs an equi-join.
        how : str, optional
            default ``inner``. Must be one of: ``inner``, ``cross``, ``outer``,
            ``full``, ``fullouter``, ``full_outer``, ``left``, ``leftouter``, ``left_outer``,
            ``right``, ``rightouter``, ``right_outer``, ``semi``, ``leftsemi``, ``left_semi``,
            ``anti``, ``leftanti`` and ``left_anti``.

        Returns:
        -------
        :class:`DataFrame`
            Joined DataFrame.

        Examples:
        --------
        The following performs a full outer join between ``df1`` and ``df2``.

        >>> from pyspark.sql import Row
        >>> from pyspark.sql.functions import desc
        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")]).toDF("age", "name")
        >>> df2 = spark.createDataFrame([Row(height=80, name="Tom"), Row(height=85, name="Bob")])
        >>> df3 = spark.createDataFrame([Row(age=2, name="Alice"), Row(age=5, name="Bob")])
        >>> df4 = spark.createDataFrame(
        ...     [
        ...         Row(age=10, height=80, name="Alice"),
        ...         Row(age=5, height=None, name="Bob"),
        ...         Row(age=None, height=None, name="Tom"),
        ...         Row(age=None, height=None, name=None),
        ...     ]
        ... )

        Inner join on columns (default)

        >>> df.join(df2, "name").select(df.name, df2.height).show()
        +----+------+
        |name|height|
        +----+------+
        | Bob|    85|
        +----+------+
        >>> df.join(df4, ["name", "age"]).select(df.name, df.age).show()
        +----+---+
        |name|age|
        +----+---+
        | Bob|  5|
        +----+---+

        Outer join for both DataFrames on the 'name' column.

        >>> df.join(df2, df.name == df2.name, "outer").select(df.name, df2.height).sort(
        ...     desc("name")
        ... ).show()
        +-----+------+
        | name|height|
        +-----+------+
        |  Bob|    85|
        |Alice|  NULL|
        | NULL|    80|
        +-----+------+
        >>> df.join(df2, "name", "outer").select("name", "height").sort(desc("name")).show()
        +-----+------+
        | name|height|
        +-----+------+
        |  Tom|    80|
        |  Bob|    85|
        |Alice|  NULL|
        +-----+------+

        Outer join for both DataFrams with multiple columns.

        >>> df.join(df3, [df.name == df3.name, df.age == df3.age], "outer").select(
        ...     df.name, df3.age
        ... ).show()
        +-----+---+
        | name|age|
        +-----+---+
        |Alice|  2|
        |  Bob|  5|
        +-----+---+
        """
        if on is not None and not isinstance(on, list):
            on = [on]  # type: ignore[assignment]
        conditions = [x for x in on or [] if isinstance(x, Column)]
        if on is not None and not all(isinstance(x, str) for x in on):
            assert isinstance(on, list)
            # Get (or create) the Expressions from the list of Columns
            on = [spark_sql_functions._to_column_expr(x) for x in on]

            # & all the Expressions together to form one Expression
            assert isinstance(on[0], Expression), "on should be Column or list of Column"
            on = reduce(lambda x, y: x.__and__(y), cast("list[Expression]", on))

        left, right = self._disjoint_join_inputs(other)
        scope = _JoinInputs(left, right, [], "inner")
        _check_references(
            conditions, scope.column_expr, scope.column_references, scope.binding_names(), scope.check_hidden
        )
        using: list[str] = []
        if on is None and how is None:
            how = "inner"
            result = left.relation.cross(right.relation)
        else:
            if how is None:
                how = "inner"
            if on is None:
                on = "true"
            elif isinstance(on, list) and all(isinstance(x, str) for x in on):
                using = on
            else:
                on = str(on)
            assert isinstance(how, str), "how should be a string"

            how = _join_type(how)
            result = left.relation.join(right.relation, on, how)
        return left._joined(result, right, using, how)

    @property
    def _join_inputs(self) -> _JoinInputs | None:
        """The inputs of the join whose scope this DataFrame's relation shares, if any."""
        if self._join_scope is None or self._join_scope[0] is not self.relation:
            return None
        return self._join_scope[1]

    def _derive(self, relation: duckdb.DuckDBPyRelation | None) -> "DataFrame":
        result = self._derive_scope(relation)
        if relation is not None and (relation.type in _INHERITS_BINDINGS or relation.type == "SUBQUERY_RELATION"):
            result._inherited_references = self._column_references()
        return result

    def _derive_scope(self, relation: duckdb.DuckDBPyRelation | None) -> "DataFrame":
        names = self.columns if relation is not None and relation.type in _KEEPS_COLUMNS else None
        if self._join_inputs is None or relation is None or self.relation.type == "CROSS_PRODUCT_RELATION":
            return DataFrame(relation, self.session, names)
        if relation.type not in _INHERITS_BINDINGS and (
            relation.type != "SUBQUERY_RELATION" or self.relation.type not in _INHERITS_BINDINGS
        ):
            return DataFrame(relation, self.session, names)
        if relation.type == "FILTER_RELATION":
            return DataFrame(relation, self.session, names, self._join_inputs)
        # The engine binds a filter, projection or aggregate over this relation inside the join's own query, where
        # it would land below the ORDER BY, LIMIT or DISTINCT, or lose the subquery's alias; a projection prevents it.
        if relation.type == "ORDER_RELATION" and self.relation.type == "FILTER_RELATION":
            # The sort reads the filter as a subquery, which renames repeated column names.
            return DataFrame(relation.select(*self._column_exprs_by_position()), self.session)
        return DataFrame(relation.select(StarExpression()), self.session, self.columns)

    def _column_exprs_by_position(self) -> list[Expression]:
        return [_position(i).alias(name) for i, name in enumerate(self.columns)]

    def _has_column(self, name: str) -> bool:
        return _name_key(name) in self._name_counts

    def _exposes_join_inputs(self) -> bool:
        return self._join_inputs is not None and self.relation.type in _JOIN_RELATIONS

    def _binding_names(self) -> frozenset[str]:
        if self._exposes_join_inputs():
            return self._join_inputs.binding_names()
        return frozenset({_name_key(self.relation.alias)})

    def _disjoint_join_inputs(self, other: "DataFrame") -> tuple["DataFrame", "DataFrame"]:
        left, right = self._as_join_input(), other._as_join_input()
        # DataFrames derived from one another share their relation's alias, so a join can see one name twice;
        # renaming both inputs makes a reference through that name fail instead of silently picking one side.
        if left._binding_names().isdisjoint(right._binding_names()):
            return left, right
        return left._with_fresh_alias(), right._with_fresh_alias()

    def _as_join_input(self) -> "DataFrame":
        """This DataFrame as an input of a join, with its repeated names under DuckDB names of their own.

        A join reads an input that is not itself a join as a subquery, where DuckDB's binder would rename repeated
        names, and shows the DuckDB names of all its inputs to a bare column reference. The inputs of a join that is
        an input were renamed when that join was made.
        """
        if self._exposes_join_inputs() or all(count == 1 for count in self._name_counts.values()):
            return self
        renamed = copy.copy(self)
        renamed.relation = self._relation_named(_join_input_names(self.columns))
        renamed._join_scope = None
        renamed._inherited_references = self._column_references()
        return renamed

    def _relation_named(self, duckdb_names: list[str]) -> duckdb.DuckDBPyRelation:
        """This DataFrame's columns under ``duckdb_names``; over a join it reads the join as one relation."""
        return self.relation.select(
            *(expr.alias(name) for expr, name in zip(self._column_exprs(), duckdb_names, strict=True))
        )

    def _with_fresh_alias(self) -> "DataFrame":
        if any(count > 1 for count in self._name_counts.values()):
            msg = "Joining DataFrames that share an input is not supported when one of them has duplicate column names"
            raise ContributionsAcceptedError(msg)
        return self.alias(_fresh_alias())

    def _joined(self, relation: duckdb.DuckDBPyRelation, right: "DataFrame", using: list[str], how: str) -> "DataFrame":
        inputs = _JoinInputs(self, right, using, how)
        return DataFrame(relation, self.session, inputs.column_names(), inputs)

    def _column_expr(self, name: str) -> Expression:
        if self._join_inputs is not None:
            expr = self._join_inputs.column_expr(name)
            # DuckDB reads a name neither input has as a qualified name or a struct field, as Spark does.
            return _column_reference(name) if expr is None else expr
        self._check_name(name)
        return ColumnExpression(self.relation.alias, name)

    def _scope_column_expr(self, name: str) -> Expression:
        parts = _name_parts(name)
        if len(parts) > 1 or parts[0] == "*":
            return _column_reference(name)
        return self._name_column_expr(parts[0])

    def _name_column_expr(self, name: str) -> Expression:
        """The column called ``name``, a name without quoting or a qualifier, in this DataFrame's scope."""
        if self._join_inputs is not None:
            expr = self._join_inputs.column_expr(name)
            return _bare_column(name) if expr is None else expr
        self._check_name(name)
        return _bare_column(name)

    def _check_name(self, name: str) -> None:
        count = self._name_counts[_name_key(name)]
        if count > 1:
            raise _ambiguous(name)
        if count == 0:
            self._check_hidden(name)

    def _check_hidden(self, name: str) -> None:
        if self._join_inputs is not None:
            self._join_inputs.check_hidden(name)
        elif _name_key(name) in self._hidden_names():
            raise _unresolved(name, self.columns)

    def _hidden_names(self) -> frozenset[str]:
        """The keys of the DuckDB names of this DataFrame's columns that are none of its Spark names."""
        if self._join_inputs is not None:
            return self._join_inputs.left._hidden_names() | self._join_inputs.right._hidden_names()
        return frozenset(map(_name_key, self.relation.columns)) - self._name_counts.keys()

    def _check_bindings(self) -> frozenset[str]:
        return self._join_inputs.binding_names() if self._join_inputs is not None else self._binding_names()

    def _check_references(self, *cols: Column) -> None:
        _check_references(
            cols, self._name_column_expr, self._column_references, self._check_bindings(), self._check_hidden
        )

    def _column_exprs(self) -> list[Expression]:
        if self._join_inputs is not None:
            return self._join_inputs.column_exprs()
        return self._column_exprs_by_position()

    def _binding_width(self) -> int:
        if self._exposes_join_inputs():
            return self._join_inputs.binding_width()
        return len(self.relation.columns)

    def _join_input_column_expr(self, name: str) -> Expression:
        if not self._exposes_join_inputs():
            self._check_name(name)
        return self._join_input_key_expr(name)

    def _join_input_key_expr(self, name: str) -> Expression:
        """The first column named ``name`` of this join input, which a join key on the name refers to."""
        if self._exposes_join_inputs():
            expr = self._join_inputs.column_expr(name)
            return _bare_column(name) if expr is None else expr
        return ColumnExpression(self.relation.alias, name)

    def _join_input_column_exprs(self, offset: int) -> list[Expression]:
        if self._exposes_join_inputs():
            return self._join_inputs.column_exprs(offset)
        # The binder reads a qualified name as the first column of the name, so a repeated name needs its position.
        return [
            _position(offset + i).alias(name)
            if self._name_counts[_name_key(name)] > 1
            else ColumnExpression(self.relation.alias, name)
            for i, name in enumerate(self.columns)
        ]

    def _join_input_relation_columns(self, binding: str, offset: int) -> list[tuple[Expression, str]] | None:
        if self._exposes_join_inputs():
            return self._join_inputs.relation_columns(binding, offset)
        if binding != _name_key(self.relation.alias):
            return None
        return list(zip(self._join_input_column_exprs(offset), self.columns, strict=True))

    def _join_input_references(self) -> list[set[str]]:
        if self._exposes_join_inputs():
            return self._join_inputs.column_references()
        return [{_name_key(str(ColumnExpression(self.relation.alias, name)))} for name in self.columns]

    def _set_operand(self) -> duckdb.DuckDBPyRelation:
        if isinstance(self.relation, _NoColumnsRelation):
            _raise_no_columns()
        return self.relation

    def _checked_set_operand(self, other: "DataFrame", operator: str) -> duckdb.DuckDBPyRelation:
        """``other``'s relation as the second input of a set operation with this DataFrame, of the same width."""
        self._set_operand()
        relation = other._set_operand()
        if len(self.columns) != len(other.columns):
            raise _num_columns_mismatch(operator, len(self.columns), len(other.columns))
        return relation

    def _check_compatible_types(
        self, other: "DataFrame", operator: str, by_name: bool = False, allow_missing: bool = False
    ) -> None:
        for i, (mine, theirs) in enumerate(zip(self.schema.fields, other.schema.fields, strict=True)):
            if not _compatible(mine.dataType, theirs.dataType, by_name, allow_missing):
                raise _incompatible_column_type(operator, i, mine.dataType, theirs.dataType)

    def _positional_operand(self, other: "DataFrame", operator: str) -> duckdb.DuckDBPyRelation:
        """``other``'s relation as the second input of a set operation, which Spark resolves by position.

        DuckDB matches struct fields by name. Spark's union matches them by position and keeps the first input's
        names; its other set operations reject struct fields of other names.
        """
        relation = self._checked_set_operand(other, operator)
        self._check_compatible_types(other, operator)
        columns = other._column_exprs()
        converted = False
        for i, (mine, theirs) in enumerate(zip(self.schema.fields, other.schema.fields, strict=True)):
            if not _renames_struct_fields(mine.dataType, theirs.dataType):
                continue
            if operator != "UNION" or not _same_shape(mine.dataType, theirs.dataType):
                raise _incompatible_column_type(operator, i, mine.dataType, theirs.dataType)
            columns[i] = _convert(columns[i], mine.dataType.duckdb_type).alias(theirs.name)
            converted = True
        if operator != "UNION":
            for mine, theirs in zip(self.schema.fields, other.schema.fields, strict=True):
                _check_comparable(mine.name, mine.dataType, theirs.dataType)
        return relation.select(*columns) if converted else relation

    def crossJoin(self, other: "DataFrame") -> "DataFrame":
        """Returns the cartesian product with another :class:`DataFrame`.

        .. versionadded:: 2.1.0

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Parameters
        ----------
        other : :class:`DataFrame`
            Right side of the cartesian product.

        Returns:
        -------
        :class:`DataFrame`
            Joined DataFrame.

        Examples:
        --------
        >>> from pyspark.sql import Row
        >>> df = spark.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
        >>> df2 = spark.createDataFrame([Row(height=80, name="Tom"), Row(height=85, name="Bob")])
        >>> df.crossJoin(df2.select("height")).select("age", "name", "height").show()
        +---+-----+------+
        |age| name|height|
        +---+-----+------+
        | 14|  Tom|    80|
        | 14|  Tom|    85|
        | 23|Alice|    80|
        | 23|Alice|    85|
        | 16|  Bob|    80|
        | 16|  Bob|    85|
        +---+-----+------+
        """
        left, right = self._disjoint_join_inputs(other)
        return left._joined(left.relation.cross(right.relation), right, [], "inner")

    def alias(self, alias: str) -> "DataFrame":
        """Returns a new :class:`DataFrame` with an alias set.

        Parameters
        ----------
        alias : str
            an alias name to be set for the :class:`DataFrame`.

        Returns:
        -------
        :class:`DataFrame`
            Aliased DataFrame.

        Examples:
        --------
        >>> from pyspark.sql.functions import col, desc
        >>> df = spark.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
        >>> df_as1 = df.alias("df_as1")
        >>> df_as2 = df.alias("df_as2")
        >>> joined_df = df_as1.join(df_as2, col("df_as1.name") == col("df_as2.name"), "inner")
        >>> joined_df.select("df_as1.name", "df_as2.name", "df_as2.age").sort(
        ...     desc("df_as1.name")
        ... ).show()
        +-----+-----+---+
        | name| name|age|
        +-----+-----+---+
        |  Tom|  Tom| 14|
        |  Bob|  Bob| 16|
        |Alice|Alice| 23|
        +-----+-----+---+
        """
        assert isinstance(alias, str), "alias should be a string"
        return self._derive(self.relation.set_alias(alias))

    def drop(self, *cols: "ColumnOrName") -> "DataFrame":  # type: ignore[misc]  # noqa: D102
        names = [_name_key(name) for name in self.columns]
        references = self._column_references()
        dropped: set[int] = set()
        for col in cols:
            if isinstance(col, str):
                dropped.update(i for i, name in enumerate(names) if name == _name_key(col))
            elif isinstance(col, Column):
                dropped.update(self._dropped_positions(col, names, references))
            else:
                raise PySparkTypeError(
                    error_class="NOT_COLUMN_OR_STR",
                    message_parameters={"arg_name": "col", "arg_type": type(col).__name__},
                )
        kept = [expr for i, expr in enumerate(self._column_exprs()) if i not in dropped]
        return self._derive(self.relation.select(*kept) if kept else None)

    def _column_references(self) -> list[set[str]]:
        """For each column, the renderings of the qualified references such as ``df.a`` that name it."""
        if self._join_inputs is not None:
            own = self._join_inputs.column_references()
        else:
            own = [{_name_key(str(ColumnExpression(self.relation.alias, name)))} for name in self.columns]
        if self._inherited_references is None:
            return own
        return [a | b for a, b in zip(own, self._inherited_references, strict=True)]

    def _qualified_references(self) -> frozenset[str]:
        """The renderings of the qualified references that name a column, folded to lower case."""
        return frozenset(reference for references in self._column_references() for reference in references)

    def _through_inherited_reference(self, col: Column) -> Column:
        """``col``, or where it is a qualified reference this DataFrame's own scope no longer sees, its column.

        An alias of such a reference is the column under the alias.
        """
        if self._join_inputs is not None or self._inherited_references is None or not isinstance(col, Column):
            return col
        if _is_qualified_reference(col):
            rendered, name = _name_key(str(col.expr)), None
        else:
            ref = _aliased_reference(col)
            if ref is None:
                return col
            rendered = ref.reference if isinstance(ref, _BoundRef) else _name_key(str(ColumnExpression(*ref.parts)))
            name = col._output
        positions = [i for i, references in enumerate(self._inherited_references) if rendered in references]
        if len(positions) != 1:
            return col
        return Column(_position(positions[0]).alias(name or self.columns[positions[0]]), col._name, col._output)

    @staticmethod
    def _dropped_positions(col: Column, names: list[str], references: list[set[str]]) -> list[int]:
        """The positions a Column given to drop() refers to; anything but a column reference refers to none."""
        if not _is_reference(col):
            return []
        if _is_qualified_reference(col):
            rendered = _name_key(str(col.expr))
            return [i for i, reference in enumerate(references) if rendered in reference]
        parts = col._name[0].parts
        positions = [i for i, name in enumerate(names) if name == _name_key(parts[0])]
        if len(positions) > 1:
            raise _ambiguous(parts[0])
        return positions

    def __repr__(self) -> str:  # noqa: D105
        return "DataFrame[{}]".format(", ".join(f"{name}: {type_name}" for name, type_name in self.dtypes))

    def limit(self, num: int) -> "DataFrame":
        """Limits the result count to the number specified.

        Parameters
        ----------
        num : int
            Number of records to return. Will return this number of records
            or all records if the DataFrame contains less than this number of records.

        Returns:
        -------
        :class:`DataFrame`
            Subset of the records

        Examples:
        --------
        >>> df = spark.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
        >>> df.limit(1).show()
        +---+----+
        |age|name|
        +---+----+
        | 14| Tom|
        +---+----+
        >>> df.limit(0).show()
        +---+----+
        |age|name|
        +---+----+
        +---+----+
        """
        rel = self.relation.limit(num)
        return self._derive(rel)

    def __contains__(self, item: str) -> bool:
        """Check if the :class:`DataFrame` contains a column by the name of `item`."""
        if isinstance(self.relation, _NoColumnsRelation):
            _raise_no_columns()
        return self._has_column(item)

    @property
    def schema(self) -> StructType:
        """Returns the schema of this :class:`DataFrame` as a :class:`duckdb.experimental.spark.sql.types.StructType`.

        Examples:
        --------
        >>> df.schema
        StructType([StructField('age', IntegerType(), True),
                    StructField('name', StringType(), True)])
        """
        return self._schema

    @overload
    def __getitem__(self, item: int | str) -> Column: ...

    @overload
    def __getitem__(self, item: Column | list | tuple) -> "DataFrame": ...

    def __getitem__(self, item: int | str | Column | list | tuple) -> Union[Column, "DataFrame"]:
        """Returns the column as a :class:`Column`.

        Examples:
        --------
        >>> df.select(df["age"]).collect()
        [Row(age=2), Row(age=5)]
        >>> df[["name", "age"]].collect()
        [Row(name='Alice', age=2), Row(name='Bob', age=5)]
        >>> df[df.age > 3].collect()
        [Row(age=5, name='Bob')]
        >>> df[df[0] > 3].collect()
        [Row(age=5, name='Bob')]
        """
        if isinstance(item, str):
            return self._bound_column(item)
        elif isinstance(item, Column):
            return self.filter(item)
        elif isinstance(item, (list, tuple)):
            return self.select(*item)
        elif isinstance(item, int):
            return self[self.columns[item]]
        else:
            msg = f"Unexpected item type: {type(item)}"
            raise TypeError(msg)

    def __getattr__(self, name: str) -> Column:
        """Returns the :class:`Column` denoted by ``name``.

        Examples:
        --------
        >>> df.select(df.age).collect()
        [Row(age=2), Row(age=5)]
        """
        schema = self.__dict__.get("_schema")
        if schema is None or name not in schema.names:
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)
        return self._bound_column(name)

    def _bound_column(self, name: str) -> Column:
        expr = self._column_expr(name)
        return Column(expr, (_BoundRef((name,), _name_key(str(expr))),))

    @overload
    def groupBy(self, *cols: "ColumnOrName") -> "GroupedData": ...

    @overload
    def groupBy(self, __cols: list[Column] | list[str]) -> "GroupedData": ...  # noqa: PYI063

    def groupBy(self, *cols: "ColumnOrName") -> "GroupedData":  # type: ignore[misc]
        """Groups the :class:`DataFrame` using the specified columns,
        so we can run aggregation on them. See :class:`GroupedData`
        for all the available aggregate functions.

        :func:`groupby` is an alias for :func:`groupBy`.

        Parameters
        ----------
        cols : list, str or :class:`Column`
            columns to group by.
            Each element should be a column name (string) or an expression (:class:`Column`)
            or list of them.

        Returns:
        -------
        :class:`GroupedData`
            Grouped data by given columns.

        Examples:
        --------
        >>> df = spark.createDataFrame(
        ...     [(2, "Alice"), (2, "Bob"), (2, "Bob"), (5, "Bob")], schema=["age", "name"]
        ... )

        Empty grouping columns triggers a global aggregation.

        >>> df.groupBy().avg().show()
        +--------+
        |avg(age)|
        +--------+
        |    2.75|
        +--------+

        Group-by 'name', and specify a dictionary to calculate the summation of 'age'.

        >>> df.groupBy("name").agg({"age": "sum"}).sort("name").show()
        +-----+--------+
        | name|sum(age)|
        +-----+--------+
        |Alice|       2|
        |  Bob|       9|
        +-----+--------+

        Group-by 'name', and calculate maximum values.

        >>> df.groupBy(df.name).max().sort("name").show()
        +-----+--------+
        | name|max(age)|
        +-----+--------+
        |Alice|       2|
        |  Bob|       5|
        +-----+--------+

        Group-by 'name' and 'age', and calculate the number of rows in each group.

        >>> df.groupBy(["name", df.age]).count().sort("name", "age").show()
        +-----+---+-----+
        | name|age|count|
        +-----+---+-----+
        |Alice|  2|    1|
        |  Bob|  2|    2|
        |  Bob|  5|    1|
        +-----+---+-----+
        """  # noqa: D205
        from .group import GroupedData, Grouping

        columns = cols[0] if len(cols) == 1 and isinstance(cols[0], list) else cols
        return GroupedData(Grouping(*columns), self)

    groupby = groupBy

    @property
    def write(self) -> DataFrameWriter:  # noqa: D102
        return DataFrameWriter(self)

    def printSchema(self, level: int | None = None) -> None:
        """Prints out the schema in the tree format.

        Parameters
        ----------
        level : int, optional
            How many levels to print for nested schemas. Prints all levels when not set or not positive.

        Examples:
        --------
        >>> df.printSchema()
        root
         |-- age: long (nullable = true)
         |-- name: string (nullable = true)
        """
        if level:
            print(self.schema.treeString(level))
        else:
            print(self.schema.treeString())

    def union(self, other: "DataFrame") -> "DataFrame":
        """Return a new :class:`DataFrame` containing union of rows in this and another
        :class:`DataFrame`.

        Parameters
        ----------
        other : :class:`DataFrame`
            Another :class:`DataFrame` that needs to be unioned

        Returns:
        -------
        :class:`DataFrame`

        See Also:
        --------
        DataFrame.unionAll

        Notes:
        -----
        This is equivalent to `UNION ALL` in SQL. To do a SQL-style set union
        (that does deduplication of elements), use this function followed by :func:`distinct`.

        Also as standard in SQL, this function resolves columns by position (not by name).

        Examples:
        --------
        >>> df1 = spark.createDataFrame([[1, 2, 3]], ["col0", "col1", "col2"])
        >>> df2 = spark.createDataFrame([[4, 5, 6]], ["col1", "col2", "col0"])
        >>> df1.union(df2).show()
        +----+----+----+
        |col0|col1|col2|
        +----+----+----+
        |   1|   2|   3|
        |   4|   5|   6|
        +----+----+----+
        >>> df1.union(df1).show()
        +----+----+----+
        |col0|col1|col2|
        +----+----+----+
        |   1|   2|   3|
        |   1|   2|   3|
        +----+----+----+
        """  # noqa: D205
        return self._derive(self.relation.union(self._positional_operand(other, "UNION")))

    unionAll = union

    def unionByName(self, other: "DataFrame", allowMissingColumns: bool = False) -> "DataFrame":
        """Returns a new :class:`DataFrame` containing union of rows in this and another :class:`DataFrame`.

        This is different from both `UNION ALL` and `UNION DISTINCT` in SQL. To do a SQL-style set
        union (that does deduplication of elements), use this function followed by :func:`distinct`.

        .. versionadded:: 2.3.0

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Parameters
        ----------
        other : :class:`DataFrame`
            Another :class:`DataFrame` that needs to be combined.
        allowMissingColumns : bool, optional, default False
           Specify whether to allow missing columns.

           .. versionadded:: 3.1.0

        Returns:
        -------
        :class:`DataFrame`
            Combined DataFrame.

        Examples:
        --------
        The difference between this function and :func:`union` is that this function
        resolves columns by name (not by position):

        >>> df1 = spark.createDataFrame([[1, 2, 3]], ["col0", "col1", "col2"])
        >>> df2 = spark.createDataFrame([[4, 5, 6]], ["col1", "col2", "col0"])
        >>> df1.unionByName(df2).show()
        +----+----+----+
        |col0|col1|col2|
        +----+----+----+
        |   1|   2|   3|
        |   6|   4|   5|
        +----+----+----+

        When the parameter `allowMissingColumns` is ``True``, the set of column names
        in this and other :class:`DataFrame` can differ; missing columns will be filled with null.
        Further, the missing columns of this :class:`DataFrame` will be added at the end
        in the schema of the union result:

        >>> df1 = spark.createDataFrame([[1, 2, 3]], ["col0", "col1", "col2"])
        >>> df2 = spark.createDataFrame([[4, 5, 6]], ["col1", "col2", "col3"])
        >>> df1.unionByName(df2, allowMissingColumns=True).show()
        +----+----+----+----+
        |col0|col1|col2|col3|
        +----+----+----+----+
        |   1|   2|   3|NULL|
        |NULL|   4|   5|   6|
        +----+----+----+----+
        """
        if allowMissingColumns:
            df1 = self.select(
                *[self[c] for c in self.columns],
                *[spark_sql_functions.lit(None).alias(c) for c in other.columns if not self._has_column(c)],
            )
            df2 = other.select(
                *[other[c] if other._has_column(c) else spark_sql_functions.lit(None).alias(c) for c in df1.columns]
            )
            df1._checked_set_operand(df2, "UNION")
            df2 = df2._with_map_fields_of(df1)
            df1._check_compatible_types(df2, "UNION", by_name=True, allow_missing=True)
            # DuckDB's union fills the struct fields one input lacks, as Spark does with missing columns allowed.
            return df1._derive(df1.relation.union(df2._set_operand()))
        # Stored names are column names, which select would read as Spark attribute names.
        reordered = other.select(*[other[c] for c in self.columns])
        self._checked_set_operand(other, "UNION")
        for mine, theirs in zip(self.schema.fields, reordered.schema.fields, strict=True):
            _check_struct_fields(mine.dataType, theirs.dataType)
        reordered = reordered._with_map_fields_of(self)
        self._check_compatible_types(reordered, "UNION", by_name=True)
        return self._derive(self.relation.union(reordered._set_operand()))

    def _with_map_fields_of(self, first: "DataFrame") -> "DataFrame":
        """This DataFrame as the second input of ``first.unionByName``, each map in it with ``first``'s field names.

        DuckDB's union pairs struct fields by name at any depth, where Spark's unionByName pairs those inside a map by
        position, as its union does.
        """
        columns = self._column_exprs()
        converted = False
        for i, (mine, theirs) in enumerate(zip(first.schema.fields, self.schema.fields, strict=True)):
            target = _maps_named_as(mine.dataType, theirs.dataType)
            if target != theirs.dataType:
                columns[i] = _convert(columns[i], target.duckdb_type).alias(theirs.name)
                converted = True
        return self._derive(self.relation.select(*columns)) if converted else self

    def intersect(self, other: "DataFrame") -> "DataFrame":
        """Return a new :class:`DataFrame` containing rows only in
        both this :class:`DataFrame` and another :class:`DataFrame`.
        Note that any duplicates are removed. To preserve duplicates
        use :func:`intersectAll`.

        .. versionadded:: 1.3.0

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Parameters
        ----------
        other : :class:`DataFrame`
            Another :class:`DataFrame` that needs to be combined.

        Returns:
        -------
        :class:`DataFrame`
            Combined DataFrame.

        Notes:
        -----
        This is equivalent to `INTERSECT` in SQL.

        Examples:
        --------
        >>> df1 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3), ("c", 4)], ["C1", "C2"])
        >>> df2 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3)], ["C1", "C2"])
        >>> df1.intersect(df2).sort(df1.C1.desc()).show()
        +---+---+
        | C1| C2|
        +---+---+
        |  b|  3|
        |  a|  1|
        +---+---+
        """  # noqa: D205
        return self._derive(self.relation.intersect(self._positional_operand(other, "INTERSECT"))).drop_duplicates()

    def intersectAll(self, other: "DataFrame") -> "DataFrame":
        """Return a new :class:`DataFrame` containing rows in both this :class:`DataFrame`
        and another :class:`DataFrame` while preserving duplicates.

        This is equivalent to `INTERSECT ALL` in SQL. As standard in SQL, this function
        resolves columns by position (not by name).

        .. versionadded:: 2.4.0

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Parameters
        ----------
        other : :class:`DataFrame`
            Another :class:`DataFrame` that needs to be combined.

        Returns:
        -------
        :class:`DataFrame`
            Combined DataFrame.

        Examples:
        --------
        >>> df1 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3), ("c", 4)], ["C1", "C2"])
        >>> df2 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3)], ["C1", "C2"])
        >>> df1.intersectAll(df2).sort("C1", "C2").show()
        +---+---+
        | C1| C2|
        +---+---+
        |  a|  1|
        |  a|  1|
        |  b|  3|
        +---+---+
        """  # noqa: D205
        return self._derive(self.relation.intersect(self._positional_operand(other, "INTERSECT ALL")))

    def exceptAll(self, other: "DataFrame") -> "DataFrame":
        """Return a new :class:`DataFrame` containing rows in this :class:`DataFrame` but
        not in another :class:`DataFrame` while preserving duplicates.

        This is equivalent to `EXCEPT ALL` in SQL.
        As standard in SQL, this function resolves columns by position (not by name).

        .. versionadded:: 2.4.0

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Parameters
        ----------
        other : :class:`DataFrame`
            The other :class:`DataFrame` to compare to.

        Returns:
        -------
        :class:`DataFrame`

        Examples:
        --------
        >>> df1 = spark.createDataFrame(
        ...     [("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b", 3), ("c", 4)], ["C1", "C2"]
        ... )
        >>> df2 = spark.createDataFrame([("a", 1), ("b", 3)], ["C1", "C2"])
        >>> df1.exceptAll(df2).show()
        +---+---+
        | C1| C2|
        +---+---+
        |  a|  1|
        |  a|  1|
        |  a|  2|
        |  c|  4|
        +---+---+

        """  # noqa: D205
        return self._derive(self.relation.except_(self._positional_operand(other, "EXCEPT ALL")))

    def dropDuplicates(self, subset: list[str] | None = None) -> "DataFrame":
        """Return a new :class:`DataFrame` with duplicate rows removed,
        optionally only considering certain columns.

        For a static batch :class:`DataFrame`, it just drops duplicate rows. For a streaming
        :class:`DataFrame`, it will keep all data across triggers as intermediate state to drop
        duplicates rows. You can use :func:`withWatermark` to limit how late the duplicate data can
        be and the system will accordingly limit the state. In addition, data older than
        watermark will be dropped to avoid any possibility of duplicates.

        :func:`drop_duplicates` is an alias for :func:`dropDuplicates`.

        Parameters
        ----------
        subset : List of column names, optional
            List of columns to use for duplicate comparison (default All columns).

        Returns:
        -------
        :class:`DataFrame`
            DataFrame without duplicates.

        Examples:
        --------
        >>> from pyspark.sql import Row
        >>> df = spark.createDataFrame(
        ...     [
        ...         Row(name="Alice", age=5, height=80),
        ...         Row(name="Alice", age=5, height=80),
        ...         Row(name="Alice", age=10, height=80),
        ...     ]
        ... )

        Deduplicate the same rows.

        >>> df.dropDuplicates().show()
        +-----+---+------+
        | name|age|height|
        +-----+---+------+
        |Alice|  5|    80|
        |Alice| 10|    80|
        +-----+---+------+

        Deduplicate values on 'name' and 'height' columns.

        >>> df.dropDuplicates(["name", "height"]).show()
        +-----+---+------+
        | name|age|height|
        +-----+---+------+
        |Alice|  5|    80|
        +-----+---+------+
        """  # noqa: D205
        if subset is None:
            return self.distinct()
        for column in subset:
            if not self._has_column(column):
                raise AnalysisException(
                    error_class="UNRESOLVED_COLUMN_AMONG_FIELD_NAMES",
                    message_parameters={"colName": column, "fieldNames": ", ".join(self.columns)},
                )
        rn_col = f"tmp_col_{uuid.uuid1().hex}"
        # Spark keeps one row of each group of equal subset values, so an empty subset keeps one row, and compares
        # every column of a name in the subset, so the keys are positions.
        keys = {_name_key(column) for column in subset}
        for field in self.schema.fields:
            if _name_key(field.name) in keys:
                _check_comparable(field.name, field.dataType)
        positions = ", ".join(f"#{i + 1}" for i, name in enumerate(self.columns) if _name_key(name) in keys)
        partition = f"PARTITION BY {positions}" if positions else ""
        window_spec = f"OVER({partition}) AS {rn_col}"
        # A window over a join binds against its inputs, where the positions are those of the join's own columns.
        relation = self.relation.select(*self._column_exprs()) if self._join_inputs is not None else self.relation
        df = DataFrame(relation.row_number(window_spec, "*"), self.session, [*self.columns, rn_col])
        return df.filter(f"{rn_col} = 1").drop(rn_col)

    drop_duplicates = dropDuplicates

    def distinct(self) -> "DataFrame":
        """Returns a new :class:`DataFrame` containing the distinct rows in this :class:`DataFrame`.

        Returns:
        -------
        :class:`DataFrame`
            DataFrame with distinct records.

        Examples:
        --------
        >>> df = spark.createDataFrame([(14, "Tom"), (23, "Alice"), (23, "Alice")], ["age", "name"])

        Return the number of distinct rows in the :class:`DataFrame`

        >>> df.distinct().count()
        2
        """
        for field in self.schema.fields:
            _check_comparable(field.name, field.dataType)
        distinct_rel = self.relation.distinct()
        return self._derive(distinct_rel)

    def count(self) -> int:
        """Returns the number of rows in this :class:`DataFrame`.

        Returns:
        -------
        int
            Number of rows.

        Examples:
        --------
        >>> df = spark.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])

        Return the number of rows in the :class:`DataFrame`.

        >>> df.count()
        3
        """
        count_rel = self.relation.count("*")
        return int(count_rel.fetchall()[0][0])

    def _cast_types(self, *types: str) -> "DataFrame":
        assert len(types) == len(self.columns)
        columns = [
            expr.cast(DuckDBPyType(target)).alias(name)
            for expr, target, name in zip(self._column_exprs(), types, self.columns, strict=True)
        ]
        return self._derive(self.relation.select(*columns))

    def toDF(self, *cols) -> "DataFrame":  # noqa: D102
        if len(cols) != len(self.columns):
            raise PySparkValueError(message="Provided column names and number of columns in the DataFrame don't match")
        names = [_column_name(new) for new in cols]
        projections = [existing.alias(new) for existing, new in zip(self._column_exprs(), names, strict=True)]
        return self._derive(self.relation.select(*projections))

    def _fetchall(self, limit: int | None = None) -> list[tuple]:
        """The rows, with each timestamp with a time zone as its microseconds since the epoch."""
        relation = self.relation
        if _contains_type(self.schema, _INSTANT_TYPES):
            relation = relation.select(
                *(
                    _epoch_microseconds(expr, field.dataType).alias(field.name)
                    for expr, field in zip(self._column_exprs(), self.schema.fields, strict=True)
                )
            )
        if limit is not None:
            relation = relation.limit(limit)
        return relation.fetchall()

    def collect(self) -> list[Row]:  # noqa: D102
        columns = self.columns
        result = self._fetchall()
        converters = [_duckdb_to_python(field.dataType) for field in self.schema.fields]
        if any(converters):
            result = [[_apply(convert, v) for convert, v in zip(converters, row, strict=True)] for row in result]

        def construct_row(values: list, names: list[str]) -> Row:
            row = tuple.__new__(Row, list(values))
            row.__fields__ = list(names)
            return row

        rows = [construct_row(x, columns) for x in result]
        return rows

    def cache(self) -> "DataFrame":
        """Persists the :class:`DataFrame` with the default storage level (`MEMORY_AND_DISK_DESER`).

        .. versionadded:: 1.3.0

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Notes:
        -----
        The default storage level has changed to `MEMORY_AND_DISK_DESER` to match Scala in 3.0.

        Returns:
        -------
        :class:`DataFrame`
            Cached DataFrame.

        Examples:
        --------
        >>> df = spark.range(1)
        >>> df.cache()
        DataFrame[id: bigint]

        >>> df.explain()
        == Physical Plan ==
        InMemoryTableScan ...
        """
        # A materialized relation drops the join scope and renames duplicate columns, so such frames stay lazy.
        if self._join_inputs is None and all(count == 1 for count in self._name_counts.values()) and self.columns:
            self.relation = self.relation.materialize()
        return self


def _is_reference(col: Column) -> bool:
    return _reference(col) is not None


def _aliased_reference(col: Column) -> _Ref | None:
    """The qualified column reference ``col`` is an alias of, such as ``df.a.alias("b")``, or None."""
    name = col._name
    if name is None or len(name) != 3 or not isinstance(col._output, str) or not isinstance(name[0], _Ref):
        return None
    if name[1:] != (" AS ", _quoted_identifier(col._output)):
        return None
    ref = name[0]
    return ref if isinstance(ref, _BoundRef) or len(ref.parts) > 1 else None


def _is_qualified_reference(col: Column) -> bool:
    """Whether ``col`` is a column reference bound to a relation, such as ``df.a``, rather than a bare name."""
    if not _is_reference(col):
        return False
    parts = col._name[0].parts
    return len(parts) > 1 or _name_key(str(col.expr)) != _name_key(str(_bare_column(parts[0])))


def _ordinal(index: int) -> str:
    """Spark's ordinal word in error messages for the position ``index``, counted from zero."""
    return ("first", "second", "third")[index] if index < 3 else f"{index + 1}th"


def _num_columns_mismatch(operator: str, first: int, second: int) -> AnalysisException:
    return AnalysisException(
        error_class="NUM_COLUMNS_MISMATCH",
        message_parameters={
            "operator": operator,
            "firstNumColumns": str(first),
            "invalidOrdinalNum": "second",
            "invalidNumColumns": str(second),
        },
    )


def _incompatible_column_type(operator: str, index: int, first: DataType, second: DataType) -> AnalysisException:
    return AnalysisException(
        error_class="INCOMPATIBLE_COLUMN_TYPE",
        message_parameters={
            "operator": operator,
            "columnOrdinalNumber": _ordinal(index),
            "tableOrdinalNumber": "second",
            "dataType1": f'"{_spark_sql_type(second)}"',
            "dataType2": f'"{_spark_sql_type(first)}"',
            "hint": "",
        },
    )


def _type_family(data_type: AtomicType) -> type[DataType]:
    """The family of an atomic type; Spark's set operations widen a type to another of its family."""
    if isinstance(data_type, (TimeType, TimeNTZType, TimeNSType)):
        return TimeType
    if isinstance(data_type, NumericType):
        return NumericType
    if isinstance(
        data_type,
        (
            DateType,
            TimestampType,
            TimestampNanosecondType,
            TimestampNTZType,
            TimestampSecondNTZType,
            TimestampMillisecondNTZType,
            TimestampNanosecondNTZType,
        ),
    ):
        return DateType
    return type(data_type)


def _compatible(left: DataType, right: DataType, by_name: bool, allow_missing: bool) -> bool:
    """Whether a set operation accepts ``right`` at the place of ``left``, as both Spark versions do, at any depth.

    Atomic types are compatible within a family, and a string with any atomic type but a boolean, which Spark 3.5.9
    widens to a string, and Spark 4.2.0 a binary to a binary, as DuckDB does. Struct fields pair by position, or by
    name with ``by_name``, where ``allow_missing`` accepts a field that only one of the structs has; inside a map they
    pair by position, and with ``by_name`` they need the same names.
    """
    if left == right or isinstance(left, NullType) or isinstance(right, NullType):
        return True
    if isinstance(left, StructType) and isinstance(right, StructType):
        if by_name:
            fields = {_name_key(f.name): f for f in right.fields}
            pairs = [(f, fields[_name_key(f.name)]) for f in left.fields if _name_key(f.name) in fields]
            if not allow_missing and not len(pairs) == len(left.fields) == len(right.fields):
                return False
        elif len(left.fields) == len(right.fields):
            pairs = list(zip(left.fields, right.fields, strict=True))
        else:
            return False
        return all(_compatible(a.dataType, b.dataType, by_name, allow_missing) for a, b in pairs)
    if isinstance(left, ArrayType) and isinstance(right, ArrayType):
        return _compatible(left.elementType, right.elementType, by_name, allow_missing)
    if isinstance(left, MapType) and isinstance(right, MapType):
        return all(
            _compatible(a, b, by_name=False, allow_missing=False) and not (by_name and _renames_struct_fields(a, b))
            for a, b in ((left.keyType, right.keyType), (left.valueType, right.valueType))
        )
    if not isinstance(left, AtomicType) or not isinstance(right, AtomicType):
        return False
    families = {_type_family(left), _type_family(right)}
    return len(families) == 1 or (StringType in families and BooleanType not in families)


def _check_struct_fields(left: DataType, right: DataType) -> None:
    """Raise where a struct in ``left`` has a field the struct at its place in ``right`` lacks, as unionByName does.

    Spark pairs struct fields by name at any depth outside a map, and by position inside one.
    """
    if isinstance(left, StructType) and isinstance(right, StructType):
        fields = {_name_key(f.name): f for f in right.fields}
        for field in left.fields:
            if _name_key(field.name) not in fields:
                raise AnalysisException(
                    error_class="FIELD_NOT_FOUND",
                    message_parameters={
                        "fieldName": f"`{field.name}`",
                        "fields": ", ".join(f"`{f.name}`" for f in right.fields),
                    },
                )
            _check_struct_fields(field.dataType, fields[_name_key(field.name)].dataType)
    elif isinstance(left, ArrayType) and isinstance(right, ArrayType):
        _check_struct_fields(left.elementType, right.elementType)


def _maps_named_as(left: DataType, right: DataType) -> DataType:
    """``right`` with each map in it given the type of the map at its place in ``left``, where they have one shape.

    Fields pair by name outside a map, at any depth, so only maps that differ in struct field names change.
    """
    if isinstance(left, StructType) and isinstance(right, StructType):
        fields = {_name_key(f.name): f for f in left.fields}
        return StructType(
            [
                StructField(f.name, _maps_named_as(fields[_name_key(f.name)].dataType, f.dataType), f.nullable)
                if _name_key(f.name) in fields
                else f
                for f in right.fields
            ]
        )
    if isinstance(left, ArrayType) and isinstance(right, ArrayType):
        return ArrayType(_maps_named_as(left.elementType, right.elementType), right.containsNull)
    if isinstance(left, MapType) and isinstance(right, MapType) and _same_shape(left, right):
        return left
    return right


def _check_comparable(name: str, *data_types: DataType) -> None:
    """Raise as Spark does where a column that a set operation or a deduplication compares has a map at any depth."""
    for data_type in data_types:
        if _contains_type(data_type, (MapType,)):
            raise AnalysisException(
                error_class="UNSUPPORTED_FEATURE.SET_OPERATION_ON_MAP_TYPE",
                message_parameters={"colName": f"`{name}`", "dataType": f'"{_spark_sql_type(data_type)}"'},
            )


def _renames_struct_fields(left: DataType, right: DataType) -> bool:
    """Whether a struct in ``right`` has other field names than the struct at its place in ``left``."""
    if isinstance(left, StructType) and isinstance(right, StructType):
        return list(map(_name_key, left.names)) != list(map(_name_key, right.names)) or any(
            _renames_struct_fields(a.dataType, b.dataType) for a, b in zip(left.fields, right.fields, strict=True)
        )
    if isinstance(left, ArrayType) and isinstance(right, ArrayType):
        return _renames_struct_fields(left.elementType, right.elementType)
    if isinstance(left, MapType) and isinstance(right, MapType):
        return _renames_struct_fields(left.keyType, right.keyType) or _renames_struct_fields(
            left.valueType, right.valueType
        )
    return False


def _same_shape(left: DataType, right: DataType) -> bool:
    """Whether ``left`` and ``right`` are one type but for the names of struct fields."""
    if isinstance(left, StructType) and isinstance(right, StructType):
        return len(left.fields) == len(right.fields) and all(
            _same_shape(a.dataType, b.dataType) for a, b in zip(left.fields, right.fields, strict=True)
        )
    if isinstance(left, ArrayType) and isinstance(right, ArrayType):
        return _same_shape(left.elementType, right.elementType)
    if isinstance(left, MapType) and isinstance(right, MapType):
        return _same_shape(left.keyType, right.keyType) and _same_shape(left.valueType, right.valueType)
    return left == right


def _fresh_alias() -> str:
    return f"relation_{uuid.uuid4().hex}"


def _source_frame(relation: duckdb.DuckDBPyRelation, session: "SparkSession") -> DataFrame:
    """A DataFrame reading a source under a name of its own, so a reference to it from another DataFrame fails."""
    return DataFrame(relation.set_alias(_fresh_alias()), session)


__all__ = ["DataFrame"]

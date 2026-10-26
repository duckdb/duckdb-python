import builtins
import functools
import inspect
import re
import warnings
from collections.abc import Callable
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional, Union, overload

import duckdb
from duckdb import (
    CaseExpression,
    CoalesceOperator,
    ColumnExpression,
    ConstantExpression,
    Expression,
    FunctionExpression,
    SQLExpression,
)

if TYPE_CHECKING:
    from .dataframe import DataFrame

from duckdb.sqltypes import BIGINT, DOUBLE, INTEGER, VARCHAR, DuckDBPyType

from ..errors import AnalysisException, IllegalArgumentException, PySparkTypeError, PySparkValueError
from ..exception import ContributionsAcceptedError
from . import types as _types
from ._typing import ColumnOrName
from .column import (
    _INTEGRAL_TYPE_IDS,
    Column,
    Name,
    _call_name,
    _column_reference,
    _constant,
    _evaluated_once,
    _format_name,
    _get_expr,
    _is_numpy,
    _lambda,
    _literal_name,
    _name_parts,
    _numpy_literal,
    _Ref,
    _ref_name,
    _spark_cast,
    _sql_name,
    _to_binary,
    _truncate,
)


def _invoke_function_over_columns(name: str, *cols: "ColumnOrName") -> Column:
    """Invokes n-ary JVM function identified by name
    and wraps the result with :class:`~pyspark.sql.Column`.
    """  # noqa: D205
    cols = [_to_column_expr(expr) for expr in cols]
    return _invoke_function(name, *cols)


def _aggregate_column(expr: Expression, *inputs: Expression) -> Column:
    column = Column(expr)
    column._is_aggregate = True
    column._aggregate_inputs = inputs
    return column


def _invoke_aggregate_over_columns(name: str, *cols: "ColumnOrName") -> Column:
    inputs = [_to_column_expr(c) for c in cols]
    return _aggregate_column(_invoke_function(name, *inputs).expr, *inputs)


def col(column: str) -> Column:  # noqa: D103
    return Column(_column_reference(column), _ref_name(column))


def upper(col: "ColumnOrName") -> Column:
    """Converts a string expression to upper case.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        upper case values.

    Examples:
    --------
    >>> df = spark.createDataFrame(["Spark", "PySpark", "Pandas API"], "STRING")
    >>> df.select(upper("value")).show()
    +------------+
    |upper(value)|
    +------------+
    |       SPARK|
    |     PYSPARK|
    |  PANDAS API|
    +------------+
    """
    return _invoke_function_over_columns("upper", col)


def ucase(str: "ColumnOrName") -> Column:
    """Returns `str` with all characters changed to uppercase.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        Input column or strings.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(1).select(sf.ucase(sf.lit("Spark"))).show()
    +------------+
    |ucase(Spark)|
    +------------+
    |       SPARK|
    +------------+
    """
    return upper(str)


def when(condition: "Column", value: Column | str) -> Column:  # noqa: D103
    if not isinstance(condition, Column):
        msg = "condition should be a Column"
        raise TypeError(msg)
    v = _get_expr(value)
    expr = CaseExpression(condition.expr, v)
    return Column(expr, _sql_name("CASE WHEN ", condition, " THEN ", value, " END"))


def struct(*cols: Union["ColumnOrName", list["ColumnOrName"], set["ColumnOrName"]]) -> Column:  # noqa: D103
    if len(cols) == 1 and isinstance(cols[0], (list, set)):
        cols = cols[0]
    columns = [_as_column(c) for c in cols]
    names = [name if (name := c._struct_field()) is not None else f"col{i}" for i, c in enumerate(columns, 1)]
    return Column(_struct_pack(names, [c.expr for c in columns]))


def _struct_pack(names: list[str], values: list[Expression]) -> Expression:
    """A struct of ``values`` with the field ``names``, which Spark allows to be empty.

    DuckDB's unnamed struct has empty field names, but DuckDB has no struct with both empty and other field names.
    """
    if names and not any(names):
        return FunctionExpression("row", *values)
    if not all(names):
        msg = f"a struct with both empty and other field names is not supported: {names}"
        raise ContributionsAcceptedError(msg)
    return FunctionExpression("struct_pack", *(value.alias(name) for name, value in zip(names, values, strict=True)))


def array(*cols: Union["ColumnOrName", list["ColumnOrName"] | tuple["ColumnOrName", ...]]) -> Column:
    r"""Creates a new array column.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    cols : :class:`~pyspark.sql.Column` or str
        column names or :class:`~pyspark.sql.Column`\\s that have
        the same data type.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        a column of array type.

    Examples:
    --------
    >>> df = spark.createDataFrame([("Alice", 2), ("Bob", 5)], ("name", "age"))
    >>> df.select(array("age", "age").alias("arr")).collect()
    [Row(arr=[2, 2]), Row(arr=[5, 5])]
    >>> df.select(array([df.age, df.age]).alias("arr")).collect()
    [Row(arr=[2, 2]), Row(arr=[5, 5])]
    >>> df.select(array("age", "age").alias("col")).printSchema()
    root
     |-- col: array (nullable = false)
     |    |-- element: long (containsNull = true)
    """
    if len(cols) == 1 and isinstance(cols[0], (list, set)):
        cols = cols[0]
    return _invoke_function_over_columns("list_value", *cols)


def lit(col: Any) -> Column:  # noqa: D103, ANN401
    if isinstance(col, Column):
        return col
    expr, name, alias = _numpy_literal(col) if _is_numpy(col) else (_constant(col), (_literal_name(col),), None)
    literal = Column(expr, name)
    literal._is_constant = True
    return literal if alias is None else literal.alias(alias)


def _invoke_function(function: str, *arguments) -> Column:
    return Column(FunctionExpression(function, *arguments))


def _result_type(column: Column, duckdb_type: DuckDBPyType) -> Column:
    """Give a result the type Spark gives it, where DuckDB's function returns a wider or different type."""
    return Column(column.expr.cast(duckdb_type))


def _rounded_to_integral(function: str, col: "ColumnOrName") -> Column:
    """``function``, ceil or floor, as a BIGINT, which is the input itself for an integral input.

    DuckDB has no integral overload of either, so it would round a BIGINT through a DOUBLE.
    """
    expr = _to_column_expr(col)
    names = sorted(name.upper() for name in _INTEGRAL_TYPE_IDS)
    is_integral = FunctionExpression("typeof", expr).isin(*map(ConstantExpression, names))
    return Column(
        CaseExpression(is_integral, expr.cast(BIGINT)).otherwise(FunctionExpression(function, expr).cast(BIGINT))
    )


def _null_if_null(result: Expression, *inputs: Expression) -> Column:
    """``result``, or NULL where any of ``inputs`` is NULL, as Spark gives for a function's null intolerant inputs."""
    any_null = functools.reduce(lambda a, b: a | b, [i.isnull() for i in inputs])
    return Column(CaseExpression(any_null, ConstantExpression(None)).otherwise(result))


def _is_null_literal(col: object) -> bool:
    """Whether ``col`` is ``lit(None)``, which has Spark's NULL type and DuckDB's untyped NULL."""
    return isinstance(col, Column) and str(col.expr) == "NULL"


def _spark_error(message: str) -> Expression:
    """Raise ``message`` when a row reaches this expression, for inputs Spark rejects at run time."""
    return FunctionExpression("error", ConstantExpression(message))


def _distinct_in_order(arr: Expression) -> Expression:
    """Remove repeats keeping first occurrences in order, as Spark does; DuckDB's list_distinct reorders, drops NULL.

    The list is evaluated once, so a nondeterministic element has one value in the list it is looked up in.
    """

    def distinct(values: Expression) -> Expression:
        first = _lambda(lambda x, i: FunctionExpression("list_position", values, x) == i, values)
        return FunctionExpression("list_filter", values, first)

    return _evaluated_once(arr, distinct)


def _in_common_type(
    arr1: Expression, arr2: Expression, body: Callable[[Expression, Expression], Expression]
) -> Expression:
    """``body`` of two lists with their elements converted to one type, the one DuckDB gives a list of both lists."""
    return _evaluated_once(
        FunctionExpression("list_value", arr1, arr2),
        lambda pair: body(
            FunctionExpression("list_extract", pair, ConstantExpression(1)),
            FunctionExpression("list_extract", pair, ConstantExpression(2)),
        ),
    )


def _lit_or_column(x: Union["ColumnOrName", float]) -> "ColumnOrName":
    return lit(x) if isinstance(x, (int, float)) else x


def _to_column_expr(col: ColumnOrName) -> Expression:
    if isinstance(col, Column):
        return col.expr
    elif isinstance(col, str):
        return _column_reference(col)
    else:
        raise PySparkTypeError(
            error_class="NOT_COLUMN_OR_STR",
            message_parameters={"arg_name": "col", "arg_type": type(col).__name__},
        )


def _as_column(col: ColumnOrName) -> Column:
    return col if isinstance(col, Column) else Column(_to_column_expr(col), _ref_name(col))


def regexp_replace(str: "ColumnOrName", pattern: str | Column, replacement: str | Column) -> Column:
    r"""Replace all substrings of the specified string value that match regexp with rep.

    .. versionadded:: 1.5.0

    Examples:
    --------
    >>> df = spark.createDataFrame([("100-200",)], ["str"])
    >>> df.select(regexp_replace("str", r"(\d+)", "--").alias("d")).collect()
    [Row(d='-----')]
    """
    literal = replacement if isinstance(replacement, builtins.str) else _string_literal(replacement)
    if literal is None and not _is_null_literal(replacement):
        msg = "Only a literal replacement, such as lit('$1'), is supported in regexp_replace; a column is not."
        raise ContributionsAcceptedError(msg)
    return _invoke_function(
        "regexp_replace",
        _to_column_expr(str),
        _get_expr(pattern),
        ConstantExpression(None if literal is None else _java_replacement(literal)),
        ConstantExpression("g"),
    )


_JAVA_REPLACEMENT_TOKEN = re.compile(r"(?s)\\(?P<escaped>.?)|\$(?P<group>[0-9]*)|(?P<text>.)")


def _java_replacement(replacement: str) -> str:
    r"""Translate Java's replacement syntax, where $1 is a group and a backslash escapes, to RE2's \1.

    Java reads a group number of two or more digits by the number of groups in the pattern, and RE2 refers to groups
    0 to 9 only, so such a reference raises.
    """
    translated = []
    for token in _JAVA_REPLACEMENT_TOKEN.finditer(replacement):
        escaped, digits = token["escaped"], token["group"]
        if escaped is not None:
            if not escaped:
                raise PySparkValueError(message=f"character to be escaped is missing in replacement {replacement!r}")
            translated.append("\\\\" if escaped == "\\" else escaped)
        elif digits is not None:
            if len(digits) != 1:
                msg = f"only $0 to $9 group references are supported in replacement {replacement!r}"
                raise ContributionsAcceptedError(msg)
            translated.append(f"\\{digits}")
        else:
            translated.append(token["text"])
    return "".join(translated)


def slice(x: "ColumnOrName", start: Union["ColumnOrName", int], length: Union["ColumnOrName", int]) -> Column:
    """Collection function: returns an array containing all the elements in `x` from index `start`
    (array indices start at 1, or from the end if `start` is negative) with the specified `length`.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    x : :class:`~pyspark.sql.Column` or str
        column name or column containing the array to be sliced
    start : :class:`~pyspark.sql.Column` or str or int
        column name, column, or int containing the starting index
    length : :class:`~pyspark.sql.Column` or str or int
        column name, column, or int containing the length of the slice

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        a column of array type. Subset of array.

    Examples:
    --------
    >>> df = spark.createDataFrame([([1, 2, 3],), ([4, 5],)], ["x"])
    >>> df.select(slice(df.x, 2, 2).alias("sliced")).collect()
    [Row(sliced=[2, 3]), Row(sliced=[5])]
    """  # noqa: D205
    start = ConstantExpression(start) if isinstance(start, int) else _to_column_expr(start)
    length = ConstantExpression(length) if isinstance(length, int) else _to_column_expr(length)

    arr = _to_column_expr(x)
    begin = CaseExpression(start < 0, FunctionExpression("len", arr) + start + 1).otherwise(start)
    return _null_if_null(
        CaseExpression(
            start == 0,
            _spark_error(
                "[INVALID_PARAMETER_VALUE.START] The value of parameter(s) `start` in `slice` is invalid: "
                "Expects a positive or a negative value for `start`, but got 0."
            ),
        )
        .when(
            length < 0,
            _spark_error(
                "[INVALID_PARAMETER_VALUE.LENGTH] The value of parameter(s) `length` in `slice` is invalid: "
                "Expects `length` greater than or equal to 0."
            ),
        )
        .when(begin < 1, FunctionExpression("list_slice", arr, 1, 0))
        .otherwise(FunctionExpression("list_slice", arr, begin, begin + length - 1)),
        arr,
        start,
        length,
    )


def asc_nulls_first(col: "ColumnOrName") -> Column:
    """Returns a sort expression based on the ascending order of the given
    column name, and null values return before non-null values.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to sort by in the ascending order.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column specifying the order.

    Examples:
    --------
    >>> df1 = spark.createDataFrame([(1, "Bob"), (0, None), (2, "Alice")], ["age", "name"])
    >>> df1.sort(asc_nulls_first(df1.name)).show()
    +---+-----+
    |age| name|
    +---+-----+
    |  0| NULL|
    |  2|Alice|
    |  1|  Bob|
    +---+-----+

    """  # noqa: D205
    return asc(col).nulls_first()


def asc_nulls_last(col: "ColumnOrName") -> Column:
    """Returns a sort expression based on the ascending order of the given
    column name, and null values appear after non-null values.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to sort by in the ascending order.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column specifying the order.

    Examples:
    --------
    >>> df1 = spark.createDataFrame([(0, None), (1, "Bob"), (2, "Alice")], ["age", "name"])
    >>> df1.sort(asc_nulls_last(df1.name)).show()
    +---+-----+
    |age| name|
    +---+-----+
    |  2|Alice|
    |  1|  Bob|
    |  0| NULL|
    +---+-----+

    """  # noqa: D205
    return asc(col).nulls_last()


def desc_nulls_first(col: "ColumnOrName") -> Column:
    """Returns a sort expression based on the descending order of the given
    column name, and null values appear before non-null values.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to sort by in the descending order.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column specifying the order.

    Examples:
    --------
    >>> df1 = spark.createDataFrame([(0, None), (1, "Bob"), (2, "Alice")], ["age", "name"])
    >>> df1.sort(desc_nulls_first(df1.name)).show()
    +---+-----+
    |age| name|
    +---+-----+
    |  0| NULL|
    |  1|  Bob|
    |  2|Alice|
    +---+-----+

    """  # noqa: D205
    return desc(col).nulls_first()


def desc_nulls_last(col: "ColumnOrName") -> Column:
    """Returns a sort expression based on the descending order of the given
    column name, and null values appear after non-null values.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to sort by in the descending order.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column specifying the order.

    Examples:
    --------
    >>> df1 = spark.createDataFrame([(0, None), (1, "Bob"), (2, "Alice")], ["age", "name"])
    >>> df1.sort(desc_nulls_last(df1.name)).show()
    +---+-----+
    |age| name|
    +---+-----+
    |  1|  Bob|
    |  2|Alice|
    |  0| NULL|
    +---+-----+

    """  # noqa: D205
    return desc(col).nulls_last()


def left(str: "ColumnOrName", len: "ColumnOrName") -> Column:
    """Returns the leftmost `len`(`len` can be string type) characters from the string `str`,
    if `len` is less or equal than 0 the result is an empty string.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        Input column or strings.
    len : :class:`~pyspark.sql.Column` or str
        Input column or strings, the leftmost `len`.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             "Spark SQL",
    ...             3,
    ...         )
    ...     ],
    ...     ["a", "b"],
    ... )
    >>> df.select(left(df.a, df.b).alias("r")).collect()
    [Row(r='Spa')]
    """  # noqa: D205
    # Spark casts the length to an integer, which truncates it, where DuckDB would round it.
    str, len = _to_column_expr(str), _truncate(_to_column_expr(len), INTEGER)
    return Column(
        CaseExpression(str.isnull(), ConstantExpression(None))
        .when(len <= ConstantExpression(0), ConstantExpression(""))
        .otherwise(FunctionExpression("array_slice", str, ConstantExpression(0), len))
    )


def right(str: "ColumnOrName", len: "ColumnOrName") -> Column:
    """Returns the rightmost `len`(`len` can be string type) characters from the string `str`,
    if `len` is less or equal than 0 the result is an empty string.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        Input column or strings.
    len : :class:`~pyspark.sql.Column` or str
        Input column or strings, the rightmost `len`.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             "Spark SQL",
    ...             3,
    ...         )
    ...     ],
    ...     ["a", "b"],
    ... )
    >>> df.select(right(df.a, df.b).alias("r")).collect()
    [Row(r='SQL')]
    """  # noqa: D205
    str, len = _to_column_expr(str), _truncate(_to_column_expr(len), INTEGER)
    return Column(
        CaseExpression(str.isnull(), ConstantExpression(None))
        .when(len <= ConstantExpression(0), ConstantExpression(""))
        .otherwise(FunctionExpression("array_slice", str, -len, ConstantExpression(-1)))
    )


def levenshtein(left: "ColumnOrName", right: "ColumnOrName", threshold: int | None = None) -> Column:
    """Computes the Levenshtein distance of the two given strings.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    left : :class:`~pyspark.sql.Column` or str
        first column value.
    right : :class:`~pyspark.sql.Column` or str
        second column value.
    threshold : int, optional
        if set when the levenshtein distance of the two given strings
        less than or equal to a given threshold then return result distance, or -1

        .. versionchanged: 3.5.0
            Added ``threshold`` argument.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        Levenshtein distance as integer value.

    Examples:
    --------
    >>> df0 = spark.createDataFrame(
    ...     [
    ...         (
    ...             "kitten",
    ...             "sitting",
    ...         )
    ...     ],
    ...     ["l", "r"],
    ... )
    >>> df0.select(levenshtein("l", "r").alias("d")).collect()
    [Row(d=3)]
    >>> df0.select(levenshtein("l", "r", 2).alias("d")).collect()
    [Row(d=-1)]
    """
    distance = _result_type(_invoke_function_over_columns("levenshtein", left, right), INTEGER)
    if threshold is None:
        return distance
    distance = _to_column_expr(distance)
    return Column(CaseExpression(distance > ConstantExpression(threshold), ConstantExpression(-1)).otherwise(distance))


def lpad(col: "ColumnOrName", len: Column | int, pad: Column | str) -> Column:
    """Left-pad the string column to width `len` with `pad`.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.
    len : int
        length of the final string.
    pad : str
        chars to prepend.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        left padded result.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [("abcd",)],
    ...     [
    ...         "s",
    ...     ],
    ... )
    >>> df.select(lpad(df.s, 6, "#").alias("s")).collect()
    [Row(s='##abcd')]
    """
    return _pad("lpad", col, len, pad)


def rpad(col: "ColumnOrName", len: Column | int, pad: Column | str) -> Column:
    """Right-pad the string column to width `len` with `pad`.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.
    len : int
        length of the final string.
    pad : str
        chars to append.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        right padded result.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [("abcd",)],
    ...     [
    ...         "s",
    ...     ],
    ... )
    >>> df.select(rpad(df.s, 6, "#").alias("s")).collect()
    [Row(s='abcd##')]
    """
    return _pad("rpad", col, len, pad)


def _pad(function: str, col: "ColumnOrName", length: Column | int, pad: Column | str) -> Column:
    string, width, padding = _to_column_expr(col), _get_expr(length), _get_expr(pad)
    # Spark truncates to the length with an empty pad, where DuckDB raises when the string is shorter; greatest would
    # turn a NULL width into 0, and left keeps all but the last characters for a negative one.
    non_negative = CaseExpression(width < ConstantExpression(0), ConstantExpression(0)).otherwise(width)
    truncated = FunctionExpression("left", string, non_negative)
    padded = FunctionExpression(function, string, width, padding)
    return Column(CaseExpression(padding == ConstantExpression(""), truncated).otherwise(padded))


def ascii(col: "ColumnOrName") -> Column:
    """Computes the numeric value of the first character of the string column.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        numeric value.

    Examples:
    --------
    >>> df = spark.createDataFrame(["Spark", "PySpark", "Pandas API"], "STRING")
    >>> df.select(ascii("value")).show()
    +------------+
    |ascii(value)|
    +------------+
    |          83|
    |          80|
    |          80|
    +------------+
    """
    return _invoke_function_over_columns("ascii", col)


def asin(col: "ColumnOrName") -> Column:
    """Computes inverse sine of the input column.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        inverse sine of `col`, as if computed by `java.lang.Math.asin()`

    Examples:
    --------
    >>> df = spark.createDataFrame([(0,), (2,)])
    >>> df.select(asin(df.schema.fieldNames()[0])).show()
    +--------+
    |ASIN(_1)|
    +--------+
    |     0.0|
    |     NaN|
    +--------+
    """
    col = _to_column_expr(col)
    # asin domain is [-1, 1]; return NaN for out-of-range values per PySpark semantics
    return Column(
        CaseExpression((col < -1.0) | (col > 1.0), _constant(float("nan"))).otherwise(FunctionExpression("asin", col))
    )


def like(str: "ColumnOrName", pattern: "ColumnOrName", escapeChar: Optional["Column"] = None) -> Column:
    r"""Returns true if str matches `pattern` with `escape`,
    null if any arguments are null, false otherwise.
    The default escape character is the '\'.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        A string.
    pattern : :class:`~pyspark.sql.Column` or str
        A string. See the DuckDB documentation on like_escape for more information.
    escape : :class:`~pyspark.sql.Column`
        An character added since Spark 3.0. The default escape character is the '\'.
        If an escape character precedes a special symbol or another escape character, the
        following character is matched literally. It is invalid to escape any other character.

    Examples:
    --------
    >>> df = spark.createDataFrame([("Spark", "_park")], ["a", "b"])
    >>> df.select(like(df.a, df.b).alias("r")).collect()
    [Row(r=True)]

    >>> df = spark.createDataFrame(
    ...     [("%SystemDrive%/Users/John", "/%SystemDrive/%//Users%")], ["a", "b"]
    ... )
    >>> df.select(like(df.a, df.b, lit("/")).alias("r")).collect()
    [Row(r=True)]
    """  # noqa: D205
    escapeChar = ConstantExpression("\\") if escapeChar is None else _to_column_expr(escapeChar)
    return _invoke_function("like_escape", _to_column_expr(str), _to_column_expr(pattern), escapeChar)


def ilike(str: "ColumnOrName", pattern: "ColumnOrName", escapeChar: Optional["Column"] = None) -> Column:
    r"""Returns true if str matches `pattern` with `escape` case-insensitively,
    null if any arguments are null, false otherwise.
    The default escape character is the '\'.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        A string.
    pattern : :class:`~pyspark.sql.Column` or str
        A string. See the DuckDB documentation on ilike_escape for more information.
    escape : :class:`~pyspark.sql.Column`
        An character added since Spark 3.0. The default escape character is the '\'.
        If an escape character precedes a special symbol or another escape character, the
        following character is matched literally. It is invalid to escape any other character.

    Examples:
    --------
    >>> df = spark.createDataFrame([("Spark", "_park")], ["a", "b"])
    >>> df.select(ilike(df.a, df.b).alias("r")).collect()
    [Row(r=True)]

    >>> df = spark.createDataFrame(
    ...     [("%SystemDrive%/Users/John", "/%SystemDrive/%//Users%")], ["a", "b"]
    ... )
    >>> df.select(ilike(df.a, df.b, lit("/")).alias("r")).collect()
    [Row(r=True)]
    """  # noqa: D205
    escapeChar = ConstantExpression("\\") if escapeChar is None else _to_column_expr(escapeChar)
    return _invoke_function("ilike_escape", _to_column_expr(str), _to_column_expr(pattern), escapeChar)


def array_agg(col: "ColumnOrName") -> Column:
    """Aggregate function: returns a list of objects with duplicates.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        list of objects with duplicates.

    Examples:
    --------
    >>> df = spark.createDataFrame([[1], [1], [2]], ["c"])
    >>> df.agg(array_agg("c").alias("r")).collect()
    [Row(r=[1, 1, 2])]
    """
    x = _to_column_expr(col)
    values = FunctionExpression(
        "list_filter",
        FunctionExpression("list", x),
        _lambda(lambda v: v.isnotnull()),
    )
    return _aggregate_column(CoalesceOperator(values, FunctionExpression("list_value")), x)


def collect_list(col: "ColumnOrName") -> Column:
    """Aggregate function: returns a list of objects with duplicates.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Notes:
    -----
    The function is non-deterministic because the order of collected results depends
    on the order of the rows which may be non-deterministic after a shuffle.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        list of objects with duplicates.

    Examples:
    --------
    >>> df2 = spark.createDataFrame([(2,), (5,), (5,)], ("age",))
    >>> df2.agg(collect_list("age")).collect()
    [Row(collect_list(age)=[2, 5, 5])]
    """
    return array_agg(col)


def array_append(col: "ColumnOrName", value: Column | str) -> Column:
    """Collection function: returns an array of the elements in col1 along
    with the added element in col2 at the last of the array.

    .. versionadded:: 3.4.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column containing array
    value :
        a literal value, or a :class:`~pyspark.sql.Column` expression.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        an array of values from first array along with the element.

    Notes:
    -----
    Supports Spark Connect.

    Examples:
    --------
    >>> from pyspark.sql import Row
    >>> df = spark.createDataFrame([Row(c1=["b", "a", "c"], c2="c")])
    >>> df.select(array_append(df.c1, df.c2)).collect()
    [Row(array_append(c1, c2)=['b', 'a', 'c', 'c'])]
    >>> df.select(array_append(df.c1, "x")).collect()
    [Row(array_append(c1, x)=['b', 'a', 'c', 'x'])]
    """  # noqa: D205
    arr = _to_column_expr(col)
    appended = FunctionExpression("list_append", arr, _get_expr(value))
    return _null_if_null(appended, arr)


def array_insert(arr: "ColumnOrName", pos: Union["ColumnOrName", int], value: Column | str) -> Column:
    """Collection function: adds an item into a given array at a specified array index.
    Array indices start at 1, or start from the end if index is negative.
    Index above array size appends the array, or prepends the array if index is negative,
    with 'null' elements.

    .. versionadded:: 3.4.0

    Parameters
    ----------
    arr : :class:`~pyspark.sql.Column` or str
        name of column containing an array
    pos : :class:`~pyspark.sql.Column` or str or int
        name of Numeric type column indicating position of insertion
        (starting at index 1, negative position is a start from the back of the array)
    value :
        a literal value, or a :class:`~pyspark.sql.Column` expression.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        an array of values, including the new specified value

    Notes:
    -----
    Supports Spark Connect.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [(["a", "b", "c"], 2, "d"), (["c", "b", "a"], -2, "d")], ["data", "pos", "val"]
    ... )
    >>> df.select(array_insert(df.data, df.pos.cast("integer"), df.val).alias("data")).collect()
    [Row(data=['a', 'd', 'b', 'c']), Row(data=['c', 'b', 'd', 'a'])]
    >>> df.select(array_insert(df.data, 5, "hello").alias("data")).collect()
    [Row(data=['a', 'b', 'c', None, 'hello']), Row(data=['c', 'b', 'a', None, 'hello'])]
    """  # noqa: D205
    pos = ConstantExpression(pos) if isinstance(pos, int) else _to_column_expr(pos)
    arr = _to_column_expr(arr)
    value = FunctionExpression("list_value", _get_expr(value))
    length = FunctionExpression("len", arr)
    # Position in the result; a negative position counts from the end, -1 appending.
    index = CaseExpression(pos > 0, pos).otherwise(length + 2 + pos)
    inserted = FunctionExpression(
        "list_concat",
        FunctionExpression("list_concat", FunctionExpression("list_slice", arr, 1, index - 1), value),
        FunctionExpression("list_slice", arr, index, length),
    )
    nulls_after = FunctionExpression("list_concat", FunctionExpression("list_resize", arr, pos - 1), value)
    nulls_before = FunctionExpression(
        "list_concat",
        FunctionExpression(
            "list_concat", value, FunctionExpression("list_resize", FunctionExpression("list_value", None), 1 - index)
        ),
        arr,
    )
    return _null_if_null(
        CaseExpression(
            pos == 0,
            _spark_error(
                "[INVALID_INDEX_OF_ZERO] The index 0 is invalid. "
                "An index shall be either < 0 or > 0 (the first element has index 1)."
            ),
        )
        .when(index > length + 1, nulls_after)
        .when(index < 1, nulls_before)
        .otherwise(inserted),
        arr,
        pos,
    )


def array_contains(col: "ColumnOrName", value: Column | str) -> Column:
    """Collection function: returns null if the array is null, true if the array contains the
    given value, and false otherwise.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column containing array
    value :
        value or column to check for in array

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        a column of Boolean type.

    Examples:
    --------
    >>> df = spark.createDataFrame([(["a", "b", "c"],), ([],)], ["data"])
    >>> df.select(array_contains(df.data, "a")).collect()
    [Row(array_contains(data, a)=True), Row(array_contains(data, a)=False)]
    >>> df.select(array_contains(df.data, lit("a"))).collect()
    [Row(array_contains(data, a)=True), Row(array_contains(data, a)=False)]
    """  # noqa: D205
    if value is None or _is_null_literal(value):
        msg = (
            "[DATATYPE_MISMATCH.NULL_TYPE] Cannot resolve array_contains due to data type mismatch: "
            "Null typed values cannot be used as arguments of `array_contains`."
        )
        raise AnalysisException(msg)
    arr = _to_column_expr(col)
    found = FunctionExpression("list_contains", arr, _get_expr(value))
    # Spark answers NULL, not false, when the value is missing from an array that holds a NULL.
    return Column(CaseExpression(_list_contains_null(arr) & ~found, ConstantExpression(None)).otherwise(found))


def array_distinct(col: "ColumnOrName") -> Column:
    """Collection function: removes duplicate values from the array.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column or expression

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        an array of unique values.

    Examples:
    --------
    >>> df = spark.createDataFrame([([1, 2, 3, 2],), ([4, 5, 5, 4],)], ["data"])
    >>> df.select(array_distinct(df.data)).collect()
    [Row(array_distinct(data)=[1, 2, 3]), Row(array_distinct(data)=[4, 5])]
    """
    return Column(_distinct_in_order(_to_column_expr(col)))


def array_intersect(col1: "ColumnOrName", col2: "ColumnOrName") -> Column:
    """Collection function: returns an array of the elements in the intersection of col1 and col2,
    without duplicates.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col1 : :class:`~pyspark.sql.Column` or str
        name of column containing array
    col2 : :class:`~pyspark.sql.Column` or str
        name of column containing array

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        an array of values in the intersection of two arrays.

    Examples:
    --------
    >>> from pyspark.sql import Row
    >>> df = spark.createDataFrame([Row(c1=["b", "a", "c"], c2=["c", "d", "a", "f"])])
    >>> df.select(array_intersect(df.c1, df.c2)).collect()
    [Row(array_intersect(c1, c2)=['a', 'c'])]
    """  # noqa: D205
    arr1, arr2 = _to_column_expr(col1), _to_column_expr(col2)

    def in_both(left: Expression, right: Expression) -> Expression:
        return FunctionExpression(
            "list_filter",
            _distinct_in_order(left),
            _lambda(lambda x: FunctionExpression("list_position", right, x).isnotnull(), right),
        )

    return _null_if_null(_in_common_type(arr1, arr2, in_both), arr1, arr2)


def array_union(col1: "ColumnOrName", col2: "ColumnOrName") -> Column:
    """Collection function: returns an array of the elements in the union of col1 and col2,
    without duplicates.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col1 : :class:`~pyspark.sql.Column` or str
        name of column containing array
    col2 : :class:`~pyspark.sql.Column` or str
        name of column containing array

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        an array of values in union of two arrays.

    Examples:
    --------
    >>> from pyspark.sql import Row
    >>> df = spark.createDataFrame([Row(c1=["b", "a", "c"], c2=["c", "d", "a", "f"])])
    >>> df.select(array_union(df.c1, df.c2)).collect()
    [Row(array_union(c1, c2)=['b', 'a', 'c', 'd', 'f'])]
    """  # noqa: D205
    return Column(_distinct_in_order(FunctionExpression("||", _to_column_expr(col1), _to_column_expr(col2))))


def array_max(col: "ColumnOrName") -> Column:
    """Collection function: returns the maximum value of the array.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column or expression

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        maximum value of an array.

    Examples:
    --------
    >>> df = spark.createDataFrame([([2, 1, 3],), ([None, 10, -1],)], ["data"])
    >>> df.select(array_max(df.data).alias("max")).collect()
    [Row(max=3), Row(max=10)]
    """
    return _invoke_function_over_columns("list_max", col)


def array_min(col: "ColumnOrName") -> Column:
    """Collection function: returns the minimum value of the array.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column or expression

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        minimum value of array.

    Examples:
    --------
    >>> df = spark.createDataFrame([([2, 1, 3],), ([None, 10, -1],)], ["data"])
    >>> df.select(array_min(df.data).alias("min")).collect()
    [Row(min=1), Row(min=-1)]
    """
    return _invoke_function_over_columns("list_min", col)


def avg(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the average of the values in a group.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> df = spark.range(10)
    >>> df.select(avg(col("id"))).show()
    +-------+
    |avg(id)|
    +-------+
    |    4.5|
    +-------+
    """
    return _invoke_aggregate_over_columns("avg", col)


def sum(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the sum of all values in the expression.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> df = spark.range(10)
    >>> df.select(sum(df["id"])).show()
    +-------+
    |sum(id)|
    +-------+
    |     45|
    +-------+
    """
    return _invoke_aggregate_over_columns("sum", col)


def max(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the maximum value of the expression in a group.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        column for computed results.

    Examples:
    --------
    >>> df = spark.range(10)
    >>> df.select(max(col("id"))).show()
    +-------+
    |max(id)|
    +-------+
    |      9|
    +-------+
    """
    return _invoke_aggregate_over_columns("max", col)


def mean(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the average of the values in a group.
    An alias of :func:`avg`.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> df = spark.range(10)
    >>> df.select(mean(df.id)).show()
    +-------+
    |avg(id)|
    +-------+
    |    4.5|
    +-------+
    """  # noqa: D205
    return _invoke_aggregate_over_columns("mean", col)


def median(col: "ColumnOrName") -> Column:
    """Returns the median of the values in a group.

    .. versionadded:: 3.4.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the median of the values in a group.

    Notes:
    -----
    Supports Spark Connect.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         ("Java", 2012, 20000),
    ...         ("dotNET", 2012, 5000),
    ...         ("Java", 2012, 22000),
    ...         ("dotNET", 2012, 10000),
    ...         ("dotNET", 2013, 48000),
    ...         ("Java", 2013, 30000),
    ...     ],
    ...     schema=("course", "year", "earnings"),
    ... )
    >>> df.groupby("course").agg(median("earnings")).show()
    +------+----------------+
    |course|median(earnings)|
    +------+----------------+
    |  Java|         22000.0|
    |dotNET|         10000.0|
    +------+----------------+
    """
    return _invoke_aggregate_over_columns("median", col)


def mode(col: "ColumnOrName") -> Column:
    """Returns the most frequent value in a group.

    .. versionadded:: 3.4.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the most frequent value in a group.

    Notes:
    -----
    Supports Spark Connect.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         ("Java", 2012, 20000),
    ...         ("dotNET", 2012, 5000),
    ...         ("Java", 2012, 20000),
    ...         ("dotNET", 2012, 5000),
    ...         ("dotNET", 2013, 48000),
    ...         ("Java", 2013, 30000),
    ...     ],
    ...     schema=("course", "year", "earnings"),
    ... )
    >>> df.groupby("course").agg(mode("year")).show()
    +------+----------+
    |course|mode(year)|
    +------+----------+
    |  Java|      2012|
    |dotNET|      2012|
    +------+----------+
    """
    return _invoke_aggregate_over_columns("mode", col)


def min(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the minimum value of the expression in a group.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        column for computed results.

    Examples:
    --------
    >>> df = spark.range(10)
    >>> df.select(min(df.id)).show()
    +-------+
    |min(id)|
    +-------+
    |      0|
    +-------+
    """
    return _invoke_aggregate_over_columns("min", col)


def any_value(col: "ColumnOrName") -> Column:
    """Returns some value of `col` for a group of rows.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.
    ignorenulls : :class:`~pyspark.sql.Column` or bool
        if first value is null then look for first non-null value.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        some value of `col` for a group of rows.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [(None, 1), ("a", 2), ("a", 3), ("b", 8), ("b", 2)], ["c1", "c2"]
    ... )
    >>> df.select(any_value("c1"), any_value("c2")).collect()
    [Row(any_value(c1)=None, any_value(c2)=1)]
    >>> df.select(any_value("c1", True), any_value("c2", True)).collect()
    [Row(any_value(c1)='a', any_value(c2)=1)]
    """
    return _invoke_aggregate_over_columns("first", col)


def count(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the number of items in a group.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        column for computed results.

    Examples:
    --------
    Count by all columns (start), and by a column that does not count ``None``.

    >>> df = spark.createDataFrame([(None,), ("a",), ("b",), ("c",)], schema=["alphabets"])
    >>> df.select(count(expr("*")), count(df.alphabets)).show()
    +--------+----------------+
    |count(1)|count(alphabets)|
    +--------+----------------+
    |       4|               3|
    +--------+----------------+
    """
    if _is_star(col):
        return _aggregate_column(FunctionExpression("count_star"))
    return _invoke_aggregate_over_columns("count", col)


def approx_count_distinct(col: "ColumnOrName", rsd: float | None = None) -> Column:
    """Aggregate function: returns a new :class:`~pyspark.sql.Column` for approximate distinct count
    of column `col`.

    .. versionadded:: 2.1.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
    rsd : float, optional
        maximum relative standard deviation allowed (default = 0.05).
        For rsd < 0.01, it is more efficient to use :func:`count_distinct`

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column of computed results.

    Examples:
    --------
    >>> df = spark.createDataFrame([1, 2, 2, 3], "INT")
    >>> df.agg(approx_count_distinct("value").alias("distinct_values")).show()
    +---------------+
    |distinct_values|
    +---------------+
    |              3|
    +---------------+
    """  # noqa: D205
    if rsd is not None:
        msg = "rsd is not supported by DuckDB"
        raise ValueError(msg)
    return _invoke_aggregate_over_columns("approx_count_distinct", col)


def approxCountDistinct(col: "ColumnOrName", rsd: float | None = None) -> Column:
    """.. versionadded:: 1.3.0.

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    .. deprecated:: 2.1.0
        Use :func:`approx_count_distinct` instead.
    """
    warnings.warn("Deprecated in 2.1, use approx_count_distinct instead.", FutureWarning, stacklevel=3)
    return approx_count_distinct(col, rsd)


@overload
def transform(col: "ColumnOrName", f: Callable[[Column], Column]) -> Column: ...


@overload
def transform(col: "ColumnOrName", f: Callable[[Column, Column], Column]) -> Column: ...


def transform(
    col: "ColumnOrName",
    f: Callable[[Column], Column] | Callable[[Column, Column], Column],
) -> Column:
    """Returns an array of elements after applying a transformation to each element in the input array.

    .. versionadded:: 3.1.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column or expression
    f : function
        a function that is applied to each element of the input array.
        Can take one of the following forms:

        - Unary ``(x: Column) -> Column: ...``
        - Binary ``(x: Column, i: Column) -> Column...``, where the second argument is
            a 0-based index of the element.

        and can use methods of :class:`~pyspark.sql.Column`, functions defined in
        :py:mod:`pyspark.sql.functions` and Scala ``UserDefinedFunctions``.
        Python ``UserDefinedFunctions`` are not supported
        (`SPARK-27052 <https://issues.apache.org/jira/browse/SPARK-27052>`__).

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        a new array of transformed elements.

    Examples:
    --------
    >>> df = spark.createDataFrame([(1, [1, 2, 3, 4])], ("key", "values"))
    >>> df.select(transform("values", lambda x: x * 2).alias("doubled")).show()
    +------------+
    |     doubled|
    +------------+
    |[2, 4, 6, 8]|
    +------------+

    >>> def alternate(x, i):
    ...     return when(i % 2 == 0, x).otherwise(-x)
    >>> df.select(transform("values", alternate).alias("alternated")).show()
    +--------------+
    |    alternated|
    +--------------+
    |[1, -2, 3, -4]|
    +--------------+
    """
    raise NotImplementedError


def concat_ws(sep: str, *cols: "ColumnOrName") -> "Column":
    """Concatenates multiple input string columns together into a single string column,
    using the given separator.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    sep : str
        words separator.
    cols : :class:`~pyspark.sql.Column` or str
        list of columns to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        string of concatenated words.

    Examples:
    --------
    >>> df = spark.createDataFrame([("abcd", "123")], ["s", "d"])
    >>> df.select(concat_ws("-", df.s, df.d).alias("s")).collect()
    [Row(s='abcd-123')]
    """  # noqa: D205
    cols = [_to_column_expr(expr) for expr in cols]
    return _invoke_function("concat_ws", ConstantExpression(sep), *cols)


def lower(col: "ColumnOrName") -> Column:
    """Converts a string expression to lower case.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        lower case values.

    Examples:
    --------
    >>> df = spark.createDataFrame(["Spark", "PySpark", "Pandas API"], "STRING")
    >>> df.select(lower("value")).show()
    +------------+
    |lower(value)|
    +------------+
    |       spark|
    |     pyspark|
    |  pandas api|
    +------------+
    """
    return _invoke_function_over_columns("lower", col)


def lcase(str: "ColumnOrName") -> Column:
    """Returns `str` with all characters changed to lowercase.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        Input column or strings.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(1).select(sf.lcase(sf.lit("Spark"))).show()
    +------------+
    |lcase(Spark)|
    +------------+
    |       spark|
    +------------+
    """
    return lower(str)


def ceil(col: "ColumnOrName") -> Column:
    """Computes the ceiling of the given value.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(ceil(lit(-0.1))).show()
    +----------+
    |CEIL(-0.1)|
    +----------+
    |         0|
    +----------+
    """
    return _rounded_to_integral("ceil", col)


def ceiling(col: "ColumnOrName") -> Column:  # noqa: D103
    return ceil(col)


def floor(col: "ColumnOrName") -> Column:
    """Computes the floor of the given value.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        column to find floor for.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        nearest integer that is less than or equal to given value.

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(floor(lit(2.5))).show()
    +----------+
    |FLOOR(2.5)|
    +----------+
    |         2|
    +----------+
    """
    return _rounded_to_integral("floor", col)


def abs(col: "ColumnOrName") -> Column:
    """Computes the absolute value.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        column for computed results.

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(abs(lit(-1))).show()
    +-------+
    |abs(-1)|
    +-------+
    |      1|
    +-------+
    """
    return _invoke_function_over_columns("abs", col)


def isnan(col: "ColumnOrName") -> Column:
    """An expression that returns true if the column is NaN.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        True if value is NaN and False otherwise.

    Examples:
    --------
    >>> df = spark.createDataFrame([(1.0, float("nan")), (float("nan"), 2.0)], ("a", "b"))
    >>> df.select("a", "b", isnan("a").alias("r1"), isnan(df.b).alias("r2")).show()
    +---+---+-----+-----+
    |  a|  b|   r1|   r2|
    +---+---+-----+-----+
    |1.0|NaN|false| true|
    |NaN|2.0| true|false|
    +---+---+-----+-----+
    """
    return Column(CoalesceOperator(FunctionExpression("isnan", _to_column_expr(col)), ConstantExpression(False)))


def isnull(col: "ColumnOrName") -> Column:
    """An expression that returns true if the column is null.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        True if value is null and False otherwise.

    Examples:
    --------
    >>> df = spark.createDataFrame([(1, None), (None, 2)], ("a", "b"))
    >>> df.select("a", "b", isnull("a").alias("r1"), isnull(df.b).alias("r2")).show()
    +----+----+-----+-----+
    |   a|   b|   r1|   r2|
    +----+----+-----+-----+
    |   1|NULL|false| true|
    |NULL|   2| true|false|
    +----+----+-----+-----+
    """
    return Column(_to_column_expr(col).isnull())


def isnotnull(col: "ColumnOrName") -> Column:
    """Returns true if `col` is not null, or false otherwise.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str

    Examples:
    --------
    >>> df = spark.createDataFrame([(None,), (1,)], ["e"])
    >>> df.select(isnotnull(df.e).alias("r")).collect()
    [Row(r=False), Row(r=True)]
    """
    return Column(_to_column_expr(col).isnotnull())


def equal_null(col1: "ColumnOrName", col2: "ColumnOrName") -> Column:
    """Returns same result as the EQUAL(=) operator for non-null operands,
    but returns true if both are null, false if one of the them is null.
    .. versionadded:: 3.5.0
    Parameters
    ----------
    col1 : :class:`~pyspark.sql.Column` or str
    col2 : :class:`~pyspark.sql.Column` or str

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             None,
    ...             None,
    ...         ),
    ...         (
    ...             1,
    ...             9,
    ...         ),
    ...     ],
    ...     ["a", "b"],
    ... )
    >>> df.select(equal_null(df.a, df.b).alias("r")).collect()
    [Row(r=True), Row(r=False)]
    """  # noqa: D205, D415
    if isinstance(col1, str):
        col1 = col(col1)

    if isinstance(col2, str):
        col2 = col(col2)

    return nvl((col1 == col2) | (col1.isNull() & col2.isNull()), lit(False))


def flatten(col: "ColumnOrName") -> Column:
    """Collection function: creates a single array from an array of arrays.
    If a structure of nested arrays is deeper than two levels,
    only one level of nesting is removed.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column or expression

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        flattened array.

    Examples:
    --------
    >>> df = spark.createDataFrame([([[1, 2, 3], [4, 5], [6]],), ([None, [4, 5]],)], ["data"])
    >>> df.show(truncate=False)
    +------------------------+
    |data                    |
    +------------------------+
    |[[1, 2, 3], [4, 5], [6]]|
    |[NULL, [4, 5]]          |
    +------------------------+
    >>> df.select(flatten(df.data).alias("r")).show()
    +------------------+
    |                 r|
    +------------------+
    |[1, 2, 3, 4, 5, 6]|
    |              NULL|
    +------------------+
    """  # noqa: D205
    col = _to_column_expr(col)
    contains_null = _list_contains_null(col)
    return Column(CaseExpression(contains_null, None).otherwise(FunctionExpression("flatten", col)))


def array_compact(col: "ColumnOrName") -> Column:
    """Collection function: removes null values from the array.

    .. versionadded:: 3.4.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column or expression

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        an array by excluding the null values.

    Notes:
    -----
    Supports Spark Connect.

    Examples:
    --------
    >>> df = spark.createDataFrame([([1, None, 2, 3],), ([4, 5, None, 4],)], ["data"])
    >>> df.select(array_compact(df.data)).collect()
    [Row(array_compact(data)=[1, 2, 3]), Row(array_compact(data)=[4, 5, 4])]
    """
    return _invoke_function("list_filter", _to_column_expr(col), _lambda(lambda x: x.isnotnull()))


def array_remove(col: "ColumnOrName", element: Any) -> Column:  # noqa: ANN401
    """Collection function: Remove all elements that equal to element from the given array.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column containing array
    element :
        element to be removed from the array

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        an array excluding given value.

    Examples:
    --------
    >>> df = spark.createDataFrame([([1, 2, 3, 1, 1],), ([],)], ["data"])
    >>> df.select(array_remove(df.data, 1)).collect()
    [Row(array_remove(data, 1)=[2, 3]), Row(array_remove(data, 1)=[])]
    """
    arr, removed = _to_column_expr(col), _get_expr(element)
    kept = _lambda(lambda x: CoalesceOperator(x != removed, ConstantExpression(True)), removed)
    return _null_if_null(FunctionExpression("list_filter", arr, kept), arr, removed)


def last_day(date: "ColumnOrName") -> Column:
    """Returns the last day of the month which the given date belongs to.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    date : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        last day of the month.

    Examples:
    --------
    >>> df = spark.createDataFrame([("1997-02-10",)], ["d"])
    >>> df.select(last_day(df.d).alias("date")).collect()
    [Row(date=datetime.date(1997, 2, 28))]
    """
    return _invoke_function("last_day", _to_column_expr(date))


def sqrt(col: "ColumnOrName") -> Column:
    """Computes the square root of the specified float value.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        column for computed results.

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(sqrt(lit(4))).show()
    +-------+
    |SQRT(4)|
    +-------+
    |    2.0|
    +-------+
    """
    return _invoke_function_over_columns("sqrt", col)


def cbrt(col: "ColumnOrName") -> Column:
    """Computes the cube-root of the given value.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(cbrt(lit(27))).show()
    +--------+
    |CBRT(27)|
    +--------+
    |     3.0|
    +--------+
    """
    return _invoke_function_over_columns("cbrt", col)


def char(col: "ColumnOrName") -> Column:
    """Returns the ASCII character having the binary equivalent to `col`. If col is larger than 256 the
    result is equivalent to char(col % 256).

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        Input column or strings.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(1).select(sf.char(sf.lit(65))).show()
    +--------+
    |char(65)|
    +--------+
    |       A|
    +--------+
    """  # noqa: D205
    return Column(
        _evaluated_once(
            _spark_cast(_as_column(col), BIGINT),
            lambda n: CaseExpression(n < 0, ConstantExpression("")).otherwise(
                FunctionExpression("chr", (n % 256).cast(INTEGER))
            ),
        )
    )


def corr(col1: "ColumnOrName", col2: "ColumnOrName") -> Column:
    """Returns a new :class:`~pyspark.sql.Column` for the Pearson Correlation Coefficient for
    ``col1`` and ``col2``.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col1 : :class:`~pyspark.sql.Column` or str
        first column to calculate correlation.
    col1 : :class:`~pyspark.sql.Column` or str
        second column to calculate correlation.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        Pearson Correlation Coefficient of these two column values.

    Examples:
    --------
    >>> a = range(20)
    >>> b = [2 * x for x in range(20)]
    >>> df = spark.createDataFrame(zip(a, b), ["a", "b"])
    >>> df.agg(corr("a", "b").alias("c")).collect()
    [Row(c=1.0)]
    """  # noqa: D205
    x, y = _to_column_expr(col1), _to_column_expr(col2)
    zero_variance = (FunctionExpression("regr_sxx", x, y) == 0) | (FunctionExpression("regr_syy", x, y) == 0)
    return _aggregate_column(
        CaseExpression(zero_variance, ConstantExpression(None)).otherwise(FunctionExpression("corr", x, y)), x, y
    )


def cot(col: "ColumnOrName") -> Column:
    """Computes cotangent of the input column.

    .. versionadded:: 3.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        angle in radians.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        cotangent of the angle.

    Examples:
    --------
    >>> import math
    >>> df = spark.range(1)
    >>> df.select(cot(lit(math.radians(45)))).first()
    Row(COT(0.78539...)=1.00000...)
    """
    return _invoke_function_over_columns("cot", col)


def e() -> Column:
    """Returns Euler's number.

    .. versionadded:: 3.5.0

    Examples:
    --------
    >>> spark.range(1).select(e()).show()
    +-----------------+
    |              E()|
    +-----------------+
    |2.718281828459045|
    +-----------------+
    """
    return lit(2.718281828459045)


def negative(col: "ColumnOrName") -> Column:
    """Returns the negative value.
    .. versionadded:: 3.5.0
    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        column to calculate negative value for.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        negative value.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(3).select(sf.negative("id")).show()
    +------------+
    |negative(id)|
    +------------+
    |           0|
    |          -1|
    |          -2|
    +------------+
    """  # noqa: D205, D415
    return Column(-_to_column_expr(col))


def pi() -> Column:
    """Returns Pi.

    .. versionadded:: 3.5.0

    Examples:
    --------
    >>> spark.range(1).select(pi()).show()
    +-----------------+
    |             PI()|
    +-----------------+
    |3.141592653589793|
    +-----------------+
    """
    return _invoke_function("pi")


def positive(col: "ColumnOrName") -> Column:
    """Returns the value.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        input value column.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        value.

    Examples:
    --------
    >>> df = spark.createDataFrame([(-1,), (0,), (1,)], ["v"])
    >>> df.select(positive("v").alias("p")).show()
    +---+
    |  p|
    +---+
    | -1|
    |  0|
    |  1|
    +---+
    """
    return Column(_to_column_expr(col))


def pow(col1: Union["ColumnOrName", float], col2: Union["ColumnOrName", float]) -> Column:
    """Returns the value of the first argument raised to the power of the second argument.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col1 : str, :class:`~pyspark.sql.Column` or float
        the base number.
    col2 : str, :class:`~pyspark.sql.Column` or float
        the exponent number.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the base rased to the power the argument.

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(pow(lit(3), lit(2))).first()
    Row(POWER(3, 2)=9.0)
    """
    return _invoke_function_over_columns("pow", _lit_or_column(col1), _lit_or_column(col2))


def printf(format: "ColumnOrName", *cols: "ColumnOrName") -> Column:
    r"""Formats the arguments in printf-style and returns the result as a string column.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    format : :class:`~pyspark.sql.Column` or str
        string that can contain embedded format tags and used as result column's value
    cols : :class:`~pyspark.sql.Column` or str
        column names or :class:`~pyspark.sql.Column`\\s to be used in formatting

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.createDataFrame(
    ...     [
    ...         (
    ...             "aa%d%s",
    ...             123,
    ...             "cc",
    ...         )
    ...     ],
    ...     ["a", "b", "c"],
    ... ).select(sf.printf("a", "b", "c")).show()
    +---------------+
    |printf(a, b, c)|
    +---------------+
    |        aa123cc|
    +---------------+
    """
    return _invoke_function_over_columns("printf", format, *cols)


def product(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the product of the values in a group.

    .. versionadded:: 3.2.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : str, :class:`Column`
        column containing values to be multiplied together

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> df = spark.range(1, 10).toDF("x").withColumn("mod3", col("x") % 3)
    >>> prods = df.groupBy("mod3").agg(product("x").alias("product"))
    >>> prods.orderBy("mod3").show()
    +----+-------+
    |mod3|product|
    +----+-------+
    |   0|  162.0|
    |   1|   28.0|
    |   2|   80.0|
    +----+-------+
    """
    return _invoke_aggregate_over_columns("product", col)


def rand(seed: int | None = None) -> Column:
    """Generates a random column with independent and identically distributed (i.i.d.) samples
    uniformly distributed in [0.0, 1.0).

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Notes:
    -----
    The function is non-deterministic in general case.

    Parameters
    ----------
    seed : int (default: None)
        seed value for random generator.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        random values.

    Examples:
    --------
    >>> from pyspark.sql import functions as sf
    >>> spark.range(0, 2, 1, 1).withColumn("rand", sf.rand(seed=42) * 3).show()
    +---+------------------+
    | id|              rand|
    +---+------------------+
    |  0|1.8575681106759028|
    |  1|1.5288056527339444|
    +---+------------------+
    """  # noqa: D205
    if seed is not None:
        # Maybe call setseed just before but how do we know when it is executed?
        msg = "Seed is not yet implemented"
        raise ContributionsAcceptedError(msg)
    return _invoke_function("random")


def regexp(str: "ColumnOrName", regexp: "ColumnOrName") -> Column:
    r"""Returns true if `str` matches the Java regex `regexp`, or false otherwise.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        target column to work on.
    regexp : :class:`~pyspark.sql.Column` or str
        regex pattern to apply.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        true if `str` matches a Java regex, or false otherwise.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.createDataFrame([("1a 2b 14m", r"(\d+)")], ["str", "regexp"]).select(
    ...     sf.regexp("str", sf.lit(r"(\d+)"))
    ... ).show()
    +------------------+
    |REGEXP(str, (\d+))|
    +------------------+
    |              true|
    +------------------+

    >>> import pyspark.sql.functions as sf
    >>> spark.createDataFrame([("1a 2b 14m", r"(\d+)")], ["str", "regexp"]).select(
    ...     sf.regexp("str", sf.lit(r"\d{2}b"))
    ... ).show()
    +-------------------+
    |REGEXP(str, \d{2}b)|
    +-------------------+
    |              false|
    +-------------------+

    >>> import pyspark.sql.functions as sf
    >>> spark.createDataFrame([("1a 2b 14m", r"(\d+)")], ["str", "regexp"]).select(
    ...     sf.regexp("str", sf.col("regexp"))
    ... ).show()
    +-------------------+
    |REGEXP(str, regexp)|
    +-------------------+
    |               true|
    +-------------------+
    """
    return _invoke_function_over_columns("regexp_matches", str, regexp)


def regexp_count(str: "ColumnOrName", regexp: "ColumnOrName") -> Column:
    r"""Returns a count of the number of times that the Java regex pattern `regexp` is matched
    in the string `str`.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        target column to work on.
    regexp : :class:`~pyspark.sql.Column` or str
        regex pattern to apply.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the number of times that a Java regex pattern is matched in the string.

    Examples:
    --------
    >>> df = spark.createDataFrame([("1a 2b 14m", r"\d+")], ["str", "regexp"])
    >>> df.select(regexp_count("str", lit(r"\d+")).alias("d")).collect()
    [Row(d=3)]
    >>> df.select(regexp_count("str", lit(r"mmm")).alias("d")).collect()
    [Row(d=0)]
    >>> df.select(regexp_count("str", col("regexp")).alias("d")).collect()
    [Row(d=3)]
    """  # noqa: D205
    return _result_type(
        _invoke_function_over_columns("len", _invoke_function_over_columns("regexp_extract_all", str, regexp)), INTEGER
    )


def regexp_extract(str: "ColumnOrName", pattern: str, idx: int) -> Column:
    r"""Extract a specific group matched by the Java regex `regexp`, from the specified string column.
    If the regex did not match, or the specified group did not match, an empty string is returned.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        target column to work on.
    pattern : str
        regex pattern to apply.
    idx : int
        matched group id.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        matched value specified by `idx` group id.

    Examples:
    --------
    >>> df = spark.createDataFrame([("100-200",)], ["str"])
    >>> df.select(regexp_extract("str", r"(\d+)-(\d+)", 1).alias("d")).collect()
    [Row(d='100')]
    >>> df = spark.createDataFrame([("foo",)], ["str"])
    >>> df.select(regexp_extract("str", r"(\d+)", 1).alias("d")).collect()
    [Row(d='')]
    >>> df = spark.createDataFrame([("aaaac",)], ["str"])
    >>> df.select(regexp_extract("str", "(a+)(b)?(c)", 2).alias("d")).collect()
    [Row(d='')]
    """  # noqa: D205
    return _invoke_function(
        "regexp_extract", _to_column_expr(str), ConstantExpression(pattern), ConstantExpression(idx)
    )


def regexp_extract_all(str: "ColumnOrName", regexp: "ColumnOrName", idx: int | Column | None = None) -> Column:
    r"""Extract all strings in the `str` that match the Java regex `regexp`
    and corresponding to the regex group index.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        target column to work on.
    regexp : :class:`~pyspark.sql.Column` or str
        regex pattern to apply.
    idx : int
        matched group id.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        all strings in the `str` that match a Java regex and corresponding to the regex group index.

    Examples:
    --------
    >>> df = spark.createDataFrame([("100-200, 300-400", r"(\d+)-(\d+)")], ["str", "regexp"])
    >>> df.select(regexp_extract_all("str", lit(r"(\d+)-(\d+)")).alias("d")).collect()
    [Row(d=['100', '300'])]
    >>> df.select(regexp_extract_all("str", lit(r"(\d+)-(\d+)"), 1).alias("d")).collect()
    [Row(d=['100', '300'])]
    >>> df.select(regexp_extract_all("str", lit(r"(\d+)-(\d+)"), 2).alias("d")).collect()
    [Row(d=['200', '400'])]
    >>> df.select(regexp_extract_all("str", col("regexp")).alias("d")).collect()
    [Row(d=['100', '300'])]
    """  # noqa: D205
    if idx is None:
        idx = 1
    matches = FunctionExpression(
        "regexp_extract_all", _to_column_expr(str), _to_column_expr(regexp), ConstantExpression(idx)
    )
    # A group that takes no part in a match is NULL in DuckDB and empty in Spark.
    no_null = _lambda(lambda x: CoalesceOperator(x, ConstantExpression("")))
    return _invoke_function("list_transform", matches, no_null)


def regexp_like(str: "ColumnOrName", regexp: "ColumnOrName") -> Column:
    r"""Returns true if `str` matches the Java regex `regexp`, or false otherwise.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        target column to work on.
    regexp : :class:`~pyspark.sql.Column` or str
        regex pattern to apply.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        true if `str` matches a Java regex, or false otherwise.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.createDataFrame([("1a 2b 14m", r"(\d+)")], ["str", "regexp"]).select(
    ...     sf.regexp_like("str", sf.lit(r"(\d+)"))
    ... ).show()
    +-----------------------+
    |REGEXP_LIKE(str, (\d+))|
    +-----------------------+
    |                   true|
    +-----------------------+

    >>> import pyspark.sql.functions as sf
    >>> spark.createDataFrame([("1a 2b 14m", r"(\d+)")], ["str", "regexp"]).select(
    ...     sf.regexp_like("str", sf.lit(r"\d{2}b"))
    ... ).show()
    +------------------------+
    |REGEXP_LIKE(str, \d{2}b)|
    +------------------------+
    |                   false|
    +------------------------+

    >>> import pyspark.sql.functions as sf
    >>> spark.createDataFrame([("1a 2b 14m", r"(\d+)")], ["str", "regexp"]).select(
    ...     sf.regexp_like("str", sf.col("regexp"))
    ... ).show()
    +------------------------+
    |REGEXP_LIKE(str, regexp)|
    +------------------------+
    |                    true|
    +------------------------+
    """
    return _invoke_function_over_columns("regexp_matches", str, regexp)


def regexp_substr(str: "ColumnOrName", regexp: "ColumnOrName") -> Column:
    r"""Returns the substring that matches the Java regex `regexp` within the string `str`.
    If the regular expression is not found, the result is null.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        target column to work on.
    regexp : :class:`~pyspark.sql.Column` or str
        regex pattern to apply.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the substring that matches a Java regex within the string `str`.

    Examples:
    --------
    >>> df = spark.createDataFrame([("1a 2b 14m", r"\d+")], ["str", "regexp"])
    >>> df.select(regexp_substr("str", lit(r"\d+")).alias("d")).collect()
    [Row(d='1')]
    >>> df.select(regexp_substr("str", lit(r"mmm")).alias("d")).collect()
    [Row(d=None)]
    >>> df.select(regexp_substr("str", col("regexp")).alias("d")).collect()
    [Row(d='1')]
    """  # noqa: D205
    return Column(
        FunctionExpression(
            "nullif",
            FunctionExpression("regexp_extract", _to_column_expr(str), _to_column_expr(regexp)),
            ConstantExpression(""),
        )
    )


def repeat(col: "ColumnOrName", n: int) -> Column:
    """Repeats a string column n times, and returns it as a new string column.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.
    n : int
        number of times to repeat value.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        string with repeated values.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [("ab",)],
    ...     [
    ...         "s",
    ...     ],
    ... )
    >>> df.select(repeat(df.s, 3).alias("s")).collect()
    [Row(s='ababab')]
    """
    return _invoke_function("repeat", _to_column_expr(col), ConstantExpression(n))


def sequence(start: "ColumnOrName", stop: "ColumnOrName", step: Optional["ColumnOrName"] = None) -> Column:
    """Generate a sequence of integers from `start` to `stop`, incrementing by `step`.
    If `step` is not set, incrementing by 1 if `start` is less than or equal to `stop`,
    otherwise -1.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    start : :class:`~pyspark.sql.Column` or str
        starting value (inclusive)
    stop : :class:`~pyspark.sql.Column` or str
        last values (inclusive)
    step : :class:`~pyspark.sql.Column` or str, optional
        value to add to current to get next element (default is 1)

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        an array of sequence values

    Examples:
    --------
    >>> df1 = spark.createDataFrame([(-2, 2)], ("C1", "C2"))
    >>> df1.select(sequence("C1", "C2").alias("r")).collect()
    [Row(r=[-2, -1, 0, 1, 2])]
    >>> df2 = spark.createDataFrame([(4, -4, -2)], ("C1", "C2", "C3"))
    >>> df2.select(sequence("C1", "C2", "C3").alias("r")).collect()
    [Row(r=[4, 2, 0, -2, -4])]
    """  # noqa: D205
    start_expr, stop_expr = _to_column_expr(start), _to_column_expr(stop)
    # Spark's elements have the common type of start and stop, where DuckDB generates bigints or timestamps.
    elements = FunctionExpression("list_value", start_expr, stop_expr)
    if step is None:
        # Spark steps by one, or by a day for dates, which DuckDB adds to a date as an integer. Adding an integer zero
        # to stop keeps the distance of a tinyint or smallint sequence from overflowing, and a date a date.
        distance = (stop_expr + ConstantExpression(0).cast(INTEGER)) - start_expr
        offsets = FunctionExpression("generate_series", ConstantExpression(0), FunctionExpression("abs", distance))
        steps = _lambda(
            lambda k: start_expr + (FunctionExpression("sign", distance) * k).cast(INTEGER), start_expr, stop_expr
        )
        series = FunctionExpression("list_transform", offsets, steps)
        return Column(FunctionExpression("cast_to_type", series, elements))
    step_expr = _to_column_expr(step)
    # Subtracting the step from itself gives a zero of its own type, which also works for interval steps.
    zero = step_expr - step_expr
    interval_step = FunctionExpression("typeof", zero) == ConstantExpression("INTERVAL")
    # Spark widens integral elements to the step's type too, which the element type of start and stop leaves out.
    widening_step = ~interval_step & (
        FunctionExpression("typeof", FunctionExpression("list_value", start_expr + zero, stop_expr + zero))
        != FunctionExpression("typeof", elements)
    )
    first_step = start_expr + step_expr
    # Only a step of whole days keeps the elements of a date sequence dates.
    partial_day_step = interval_step & (
        FunctionExpression("cast_to_type", first_step, FunctionExpression("list_extract", elements, 1)) != first_step
    )
    # DuckDB adds a step of months to each element in turn, where Spark adds multiples of it to start.
    step_interval = SQLExpression(f"TRY_CAST({step_expr} AS INTERVAL)")
    month_step = interval_step & (
        (FunctionExpression("date_part", ConstantExpression("year"), step_interval) != ConstantExpression(0))
        | (FunctionExpression("date_part", ConstantExpression("month"), step_interval) != ConstantExpression(0))
    )
    towards_stop = ((step_expr > zero) & (start_expr < stop_expr)) | ((step_expr < zero) & (start_expr > stop_expr))
    message = FunctionExpression(
        "concat",
        ConstantExpression("Illegal sequence boundaries: "),
        start_expr,
        ConstantExpression(" to "),
        stop_expr,
        ConstantExpression(" by "),
        step_expr,
    )
    series = FunctionExpression("generate_series", start_expr, stop_expr, step_expr)
    return Column(
        CaseExpression(
            widening_step,
            _spark_error("sequence with a step of a wider integral type than its start and stop is not supported"),
        )
        .when((start_expr != stop_expr) & ~towards_stop, FunctionExpression("error", message))
        .when((step_expr == zero) & (start_expr == stop_expr), FunctionExpression("list_value", start_expr))
        .when(
            partial_day_step,
            _spark_error(
                "sequence step must be an interval day to second of day granularity if start and end values are dates"
            ),
        )
        .when(month_step, _spark_error("sequence with a step of months is not supported"))
        .otherwise(FunctionExpression("cast_to_type", series, elements))
    )


def sign(col: "ColumnOrName") -> Column:
    """Computes the signum of the given value.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(1).select(sf.sign(sf.lit(-5)), sf.sign(sf.lit(6))).show()
    +--------+-------+
    |sign(-5)|sign(6)|
    +--------+-------+
    |    -1.0|    1.0|
    +--------+-------+
    """
    x = _to_column_expr(col).cast(DOUBLE)
    # DuckDB's sign of NaN is 0, Spark's is NaN.
    signum = FunctionExpression("sign", x).cast(DOUBLE)
    return Column(CaseExpression(FunctionExpression("isnan", x), x).otherwise(signum))


def signum(col: "ColumnOrName") -> Column:
    """Computes the signum of the given value.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(1).select(sf.signum(sf.lit(-5)), sf.signum(sf.lit(6))).show()
    +----------+---------+
    |SIGNUM(-5)|SIGNUM(6)|
    +----------+---------+
    |      -1.0|      1.0|
    +----------+---------+
    """
    return sign(col)


def sin(col: "ColumnOrName") -> Column:
    """Computes sine of the input column.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        sine of the angle, as if computed by `java.lang.Math.sin()`

    Examples:
    --------
    >>> import math
    >>> df = spark.range(1)
    >>> df.select(sin(lit(math.radians(90)))).first()
    Row(SIN(1.57079...)=1.0)
    """
    return _invoke_function_over_columns("sin", col)


def skewness(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the skewness of the values in a group.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        skewness of given column.

    Examples:
    --------
    >>> df = spark.createDataFrame([[1], [1], [2]], ["c"])
    >>> df.select(skewness(df.c)).first()
    Row(skewness(c)=0.70710...)
    """
    x = _to_column_expr(col)
    # DuckDB's moments raise on a NaN or an infinity, which make Spark's NaN, so they see only finite values.
    finite = FunctionExpression("isfinite", x.cast(DOUBLE))
    finite_x = CaseExpression(finite, x)
    n = FunctionExpression("count", finite_x).cast(DOUBLE)
    # DuckDB's skewness is the sample estimate, undefined below 3 values; Spark's is the population skewness.
    population = FunctionExpression("skewness", finite_x) * (n - 2) / FunctionExpression("sqrt", n * (n - 1))
    two_distinct = (n == 2) & (FunctionExpression("var_pop", finite_x) > 0)
    skew = (
        CaseExpression(FunctionExpression("bool_or", ~finite), _constant(float("nan")))
        .when(two_distinct, ConstantExpression(0.0))
        .otherwise(population)
    )
    return _aggregate_column(skew, x)


def encode(col: "ColumnOrName", charset: str) -> Column:
    """Computes the first argument into a binary from a string using the provided character set
    (one of 'US-ASCII', 'ISO-8859-1', 'UTF-8', 'UTF-16BE', 'UTF-16LE', 'UTF-16').

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.
    charset : str
        charset to use to encode.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> df = spark.createDataFrame([("abcd",)], ["c"])
    >>> df.select(encode("c", "UTF-8")).show()
    +----------------+
    |encode(c, UTF-8)|
    +----------------+
    |   [61 62 63 64]|
    +----------------+
    """  # noqa: D205
    if charset != "UTF-8":
        msg = "Only UTF-8 charset is supported right now"
        raise ContributionsAcceptedError(msg)
    return _invoke_function("encode", _to_column_expr(col))


def find_in_set(str: "ColumnOrName", str_array: "ColumnOrName") -> Column:
    """Returns the index (1-based) of the given string (`str`) in the comma-delimited
    list (`strArray`). Returns 0, if the string was not found or if the given string (`str`)
    contains a comma.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        The given string to be found.
    str_array : :class:`~pyspark.sql.Column` or str
        The comma-delimited list.

    Examples:
    --------
    >>> df = spark.createDataFrame([("ab", "abc,b,ab,c,def")], ["a", "b"])
    >>> df.select(find_in_set(df.a, df.b).alias("r")).collect()
    [Row(r=3)]
    """  # noqa: D205
    str_array = _to_column_expr(str_array)
    str = _to_column_expr(str)
    return Column(
        CaseExpression(str.isnull() | str_array.isnull(), ConstantExpression(None))
        .when(FunctionExpression("contains", str, ConstantExpression(",")), ConstantExpression(0))
        .otherwise(
            CoalesceOperator(
                FunctionExpression(
                    "list_position", FunctionExpression("string_split", str_array, ConstantExpression(",")), str
                ),
                # If the element cannot be found, list_position returns null but we want to return 0
                ConstantExpression(0),
            )
        )
    )


def first(col: "ColumnOrName", ignorenulls: bool = False) -> Column:
    """Aggregate function: returns the first value in a group.

    The function by default returns the first values it sees. It will return the first non-null
    value it sees when ignoreNulls is set to true. If all values are null, then null is returned.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Notes:
    -----
    The function is non-deterministic because its results depends on the order of the
    rows which may be non-deterministic after a shuffle.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        column to fetch first value for.
    ignorenulls : :class:`~pyspark.sql.Column` or str
        if first value is null then look for first non-null value.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        first value of the group.

    Examples:
    --------
    >>> df = spark.createDataFrame([("Alice", 2), ("Bob", 5), ("Alice", None)], ("name", "age"))
    >>> df = df.orderBy(df.age)
    >>> df.groupby("name").agg(first("age")).orderBy("name").show()
    +-----+----------+
    | name|first(age)|
    +-----+----------+
    |Alice|      NULL|
    |  Bob|         5|
    +-----+----------+

    Now, to ignore any nulls we needs to set ``ignorenulls`` to `True`

    >>> df.groupby("name").agg(first("age", ignorenulls=True)).orderBy("name").show()
    +-----+----------+
    | name|first(age)|
    +-----+----------+
    |Alice|         2|
    |  Bob|         5|
    +-----+----------+
    """
    if ignorenulls:
        raise ContributionsAcceptedError()
    return _invoke_aggregate_over_columns("first", col)


def last(col: "ColumnOrName", ignorenulls: bool = False) -> Column:
    """Aggregate function: returns the last value in a group.

    The function by default returns the last values it sees. It will return the last non-null
    value it sees when ignoreNulls is set to true. If all values are null, then null is returned.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Notes:
    -----
    The function is non-deterministic because its results depends on the order of the
    rows which may be non-deterministic after a shuffle.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        column to fetch last value for.
    ignorenulls : :class:`~pyspark.sql.Column` or str
        if last value is null then look for non-null value.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        last value of the group.

    Examples:
    --------
    >>> df = spark.createDataFrame([("Alice", 2), ("Bob", 5), ("Alice", None)], ("name", "age"))
    >>> df = df.orderBy(df.age.desc())
    >>> df.groupby("name").agg(last("age")).orderBy("name").show()
    +-----+---------+
    | name|last(age)|
    +-----+---------+
    |Alice|     NULL|
    |  Bob|        5|
    +-----+---------+

    Now, to ignore any nulls we needs to set ``ignorenulls`` to `True`

    >>> df.groupby("name").agg(last("age", ignorenulls=True)).orderBy("name").show()
    +-----+---------+
    | name|last(age)|
    +-----+---------+
    |Alice|        2|
    |  Bob|        5|
    +-----+---------+
    """
    if ignorenulls:
        raise ContributionsAcceptedError()
    return _invoke_aggregate_over_columns("last", col)


def greatest(*cols: "ColumnOrName") -> Column:
    """Returns the greatest value of the list of column names, skipping null values.
    This function takes at least 2 parameters. It will return null if all parameters are null.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        columns to check for gratest value.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        gratest value.

    Examples:
    --------
    >>> df = spark.createDataFrame([(1, 4, 3)], ["a", "b", "c"])
    >>> df.select(greatest(df.a, df.b, df.c).alias("greatest")).collect()
    [Row(greatest=4)]
    """  # noqa: D205
    if len(cols) < 2:
        msg = "greatest should take at least 2 columns"
        raise ValueError(msg)

    cols = [_to_column_expr(expr) for expr in cols]
    return _invoke_function("greatest", *cols)


def least(*cols: "ColumnOrName") -> Column:
    """Returns the least value of the list of column names, skipping null values.
    This function takes at least 2 parameters. It will return null if all parameters are null.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    cols : :class:`~pyspark.sql.Column` or str
        column names or columns to be compared

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        least value.

    Examples:
    --------
    >>> df = spark.createDataFrame([(1, 4, 3)], ["a", "b", "c"])
    >>> df.select(least(df.a, df.b, df.c).alias("least")).collect()
    [Row(least=1)]
    """  # noqa: D205
    if len(cols) < 2:
        msg = "least should take at least 2 columns"
        raise ValueError(msg)

    cols = [_to_column_expr(expr) for expr in cols]
    return _invoke_function("least", *cols)


def trim(col: "ColumnOrName") -> Column:
    """Trim the spaces from left end for the specified string value.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        left trimmed values.

    Examples:
    --------
    >>> df = spark.createDataFrame(["   Spark", "Spark  ", " Spark"], "STRING")
    >>> df.select(ltrim("value").alias("r")).withColumn("length", length("r")).show()
    +-------+------+
    |      r|length|
    +-------+------+
    |  Spark|     5|
    |Spark  |     7|
    |  Spark|     5|
    +-------+------+
    """
    return _invoke_function_over_columns("trim", col)


def rtrim(col: "ColumnOrName") -> Column:
    """Trim the spaces from right end for the specified string value.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        right trimmed values.

    Examples:
    --------
    >>> df = spark.createDataFrame(["   Spark", "Spark  ", " Spark"], "STRING")
    >>> df.select(rtrim("value").alias("r")).withColumn("length", length("r")).show()
    +--------+------+
    |       r|length|
    +--------+------+
    |   Spark|     8|
    |   Spark|     5|
    |   Spark|     6|
    +--------+------+
    """
    return _invoke_function_over_columns("rtrim", col)


def ltrim(col: "ColumnOrName") -> Column:
    """Trim the spaces from left end for the specified string value.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        left trimmed values.

    Examples:
    --------
    >>> df = spark.createDataFrame(["   Spark", "Spark  ", " Spark"], "STRING")
    >>> df.select(ltrim("value").alias("r")).withColumn("length", length("r")).show()
    +-------+------+
    |      r|length|
    +-------+------+
    |  Spark|     5|
    |Spark  |     7|
    |  Spark|     5|
    +-------+------+
    """
    return _invoke_function_over_columns("ltrim", col)


def btrim(str: "ColumnOrName", trim: Optional["ColumnOrName"] = None) -> Column:
    """Remove the leading and trailing `trim` characters from `str`.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        Input column or strings.
    trim : :class:`~pyspark.sql.Column` or str
        The trim string characters to trim, the default value is a single space

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             "SSparkSQLS",
    ...             "SL",
    ...         )
    ...     ],
    ...     ["a", "b"],
    ... )
    >>> df.select(btrim(df.a, df.b).alias("r")).collect()
    [Row(r='parkSQ')]

    >>> df = spark.createDataFrame([("    SparkSQL   ",)], ["a"])
    >>> df.select(btrim(df.a).alias("r")).collect()
    [Row(r='SparkSQL')]
    """
    if trim is not None:
        return _invoke_function_over_columns("trim", str, trim)
    else:
        return _invoke_function_over_columns("trim", str)


def endswith(str: "ColumnOrName", suffix: "ColumnOrName") -> Column:
    """Returns a boolean. The value is True if str ends with suffix.
    Returns NULL if either input expression is NULL. Otherwise, returns False.
    Both str or suffix must be of STRING or BINARY type.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        A column of string.
    suffix : :class:`~pyspark.sql.Column` or str
        A column of string, the suffix.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             "Spark SQL",
    ...             "Spark",
    ...         )
    ...     ],
    ...     ["a", "b"],
    ... )
    >>> df.select(endswith(df.a, df.b).alias("r")).collect()
    [Row(r=False)]

    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             "414243",
    ...             "4243",
    ...         )
    ...     ],
    ...     ["e", "f"],
    ... )
    >>> df = df.select(to_binary("e").alias("e"), to_binary("f").alias("f"))
    >>> df.printSchema()
    root
     |-- e: binary (nullable = true)
     |-- f: binary (nullable = true)
    >>> df.select(endswith("e", "f"), endswith("f", "e")).show()
    +--------------+--------------+
    |endswith(e, f)|endswith(f, e)|
    +--------------+--------------+
    |          true|         false|
    +--------------+--------------+
    """  # noqa: D205
    return _invoke_function_over_columns("suffix", str, suffix)


def startswith(str: "ColumnOrName", prefix: "ColumnOrName") -> Column:
    """Returns a boolean. The value is True if str starts with prefix.
    Returns NULL if either input expression is NULL. Otherwise, returns False.
    Both str or prefix must be of STRING or BINARY type.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        A column of string.
    prefix : :class:`~pyspark.sql.Column` or str
        A column of string, the prefix.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             "Spark SQL",
    ...             "Spark",
    ...         )
    ...     ],
    ...     ["a", "b"],
    ... )
    >>> df.select(startswith(df.a, df.b).alias("r")).collect()
    [Row(r=True)]

    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             "414243",
    ...             "4142",
    ...         )
    ...     ],
    ...     ["e", "f"],
    ... )
    >>> df = df.select(to_binary("e").alias("e"), to_binary("f").alias("f"))
    >>> df.printSchema()
    root
     |-- e: binary (nullable = true)
     |-- f: binary (nullable = true)
    >>> df.select(startswith("e", "f"), startswith("f", "e")).show()
    +----------------+----------------+
    |startswith(e, f)|startswith(f, e)|
    +----------------+----------------+
    |            true|           false|
    +----------------+----------------+
    """  # noqa: D205
    return _invoke_function_over_columns("starts_with", str, prefix)


def length(col: "ColumnOrName") -> Column:
    """Computes the character length of string data or number of bytes of binary data.
    The length of character data includes the trailing spaces. The length of binary data
    includes binary zeros.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        length of the value.

    Examples:
    --------
    >>> spark.createDataFrame([("ABC ",)], ["a"]).select(length("a").alias("length")).collect()
    [Row(length=4)]
    """  # noqa: D205
    return _result_type(_invoke_function_over_columns("length", col), INTEGER)


def coalesce(*cols: "ColumnOrName") -> Column:
    """Returns the first column that is not null.
    .. versionadded:: 1.4.0
    .. versionchanged:: 3.4.0
        Supports Spark Connect.
    Parameters
    ----------
    cols : :class:`~pyspark.sql.Column` or str
        list of columns to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        value of the first column that is not null.

    Examples:
    --------
    >>> cDf = spark.createDataFrame([(None, None), (1, None), (None, 2)], ("a", "b"))
    >>> cDf.show()
    +----+----+
    |   a|   b|
    +----+----+
    |NULL|NULL|
    |   1|NULL|
    |NULL|   2|
    +----+----+
    >>> cDf.select(coalesce(cDf["a"], cDf["b"])).show()
    +--------------+
    |coalesce(a, b)|
    +--------------+
    |          NULL|
    |             1|
    |             2|
    +--------------+
    >>> cDf.select("*", coalesce(cDf["a"], lit(0.0))).show()
    +----+----+----------------+
    |   a|   b|coalesce(a, 0.0)|
    +----+----+----------------+
    |NULL|NULL|             0.0|
    |   1|NULL|             1.0|
    |NULL|   2|             0.0|
    +----+----+----------------+
    """  # noqa: D205, D415
    cols = [_to_column_expr(expr) for expr in cols]
    return Column(CoalesceOperator(*cols))


def nvl(col1: "ColumnOrName", col2: "ColumnOrName") -> Column:
    """Returns `col2` if `col1` is null, or `col1` otherwise.
    .. versionadded:: 3.5.0
    Parameters
    ----------
    col1 : :class:`~pyspark.sql.Column` or str
    col2 : :class:`~pyspark.sql.Column` or str

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             None,
    ...             8,
    ...         ),
    ...         (
    ...             1,
    ...             9,
    ...         ),
    ...     ],
    ...     ["a", "b"],
    ... )
    >>> df.select(nvl(df.a, df.b).alias("r")).collect()
    [Row(r=8), Row(r=1)]
    """  # noqa: D205, D415
    return coalesce(col1, col2)


def nvl2(col1: "ColumnOrName", col2: "ColumnOrName", col3: "ColumnOrName") -> Column:
    """Returns `col2` if `col1` is not null, or `col3` otherwise.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col1 : :class:`~pyspark.sql.Column` or str
    col2 : :class:`~pyspark.sql.Column` or str
    col3 : :class:`~pyspark.sql.Column` or str

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             None,
    ...             8,
    ...             6,
    ...         ),
    ...         (
    ...             1,
    ...             9,
    ...             9,
    ...         ),
    ...     ],
    ...     ["a", "b", "c"],
    ... )
    >>> df.select(nvl2(df.a, df.b, df.c).alias("r")).collect()
    [Row(r=6), Row(r=9)]
    """
    col1 = _to_column_expr(col1)
    col2 = _to_column_expr(col2)
    col3 = _to_column_expr(col3)
    return Column(CaseExpression(col1.isnull(), col3).otherwise(col2))


def ifnull(col1: "ColumnOrName", col2: "ColumnOrName") -> Column:
    """Returns `col2` if `col1` is null, or `col1` otherwise.
    .. versionadded:: 3.5.0
    Parameters
    ----------
    col1 : :class:`~pyspark.sql.Column` or str
    col2 : :class:`~pyspark.sql.Column` or str

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> df = spark.createDataFrame([(None,), (1,)], ["e"])
    >>> df.select(sf.ifnull(df.e, sf.lit(8))).show()
    +------------+
    |ifnull(e, 8)|
    +------------+
    |           8|
    |           1|
    +------------+
    """  # noqa: D205, D415
    return coalesce(col1, col2)


def nullif(col1: "ColumnOrName", col2: "ColumnOrName") -> Column:
    """Returns null if `col1` equals to `col2`, or `col1` otherwise.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col1 : :class:`~pyspark.sql.Column` or str
    col2 : :class:`~pyspark.sql.Column` or str

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             None,
    ...             None,
    ...         ),
    ...         (
    ...             1,
    ...             9,
    ...         ),
    ...     ],
    ...     ["a", "b"],
    ... )
    >>> df.select(nullif(df.a, df.b).alias("r")).collect()
    [Row(r=None), Row(r=1)]
    """
    return _invoke_function_over_columns("nullif", col1, col2)


def md5(col: "ColumnOrName") -> Column:
    """Calculates the MD5 digest and returns the value as a 32 character hex string.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> spark.createDataFrame([("ABC",)], ["a"]).select(md5("a").alias("hash")).collect()
    [Row(hash='902fbdd2b1df0c4f70b4a5d23525e932')]
    """
    return _invoke_function_over_columns("md5", col)


def sha2(col: "ColumnOrName", numBits: int) -> Column:
    """Returns the hex string result of SHA-2 family of hash functions (SHA-224, SHA-256, SHA-384,
    and SHA-512). The numBits indicates the desired bit length of the result, which must have a
    value of 224, 256, 384, 512, or 0 (which is equivalent to 256).

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.
    numBits : int
        the desired bit length of the result, which must have a
        value of 224, 256, 384, 512, or 0 (which is equivalent to 256).

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column for computed results.

    Examples:
    --------
    >>> df = spark.createDataFrame([["Alice"], ["Bob"]], ["name"])
    >>> df.withColumn("sha2", sha2(df.name, 256)).show(truncate=False)
    +-----+----------------------------------------------------------------+
    |name |sha2                                                            |
    +-----+----------------------------------------------------------------+
    |Alice|3bc51062973c458d5a6f2d8d64a023246354ad7e064b1e4e009ec8a0699a3043|
    |Bob  |cd9fb1e148ccd8442e5aa74904cc73bf6fb54d1d54d333bd596aa9bb4bb4e961|
    +-----+----------------------------------------------------------------+
    """  # noqa: D205
    if numBits not in {224, 256, 384, 512, 0}:
        msg = "numBits should be one of {224, 256, 384, 512, 0}"
        raise ValueError(msg)

    if numBits in {0, 256}:
        return _invoke_function_over_columns("sha256", col)

    msg = "SHA-224, SHA-384, and SHA-512 are not supported yet."
    raise ContributionsAcceptedError(msg)


def curdate() -> Column:
    """Returns the current date at the start of query evaluation as a :class:`DateType` column.
    All calls of current_date within the same query return the same value.

    .. versionadded:: 3.5.0

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        current date.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(1).select(sf.curdate()).show()  # doctest: +SKIP
    +--------------+
    |current_date()|
    +--------------+
    |    2022-08-26|
    +--------------+
    """  # noqa: D205
    return _invoke_function("today")


def current_date() -> Column:
    """Returns the current date at the start of query evaluation as a :class:`DateType` column.
    All calls of current_date within the same query return the same value.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        current date.

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(current_date()).show()  # doctest: +SKIP
    +--------------+
    |current_date()|
    +--------------+
    |    2022-08-26|
    +--------------+
    """  # noqa: D205
    return curdate()


def now() -> Column:
    """Returns the current timestamp at the start of query evaluation.

    .. versionadded:: 3.5.0

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        current timestamp at the start of query evaluation.

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(now()).show(truncate=False)  # doctest: +SKIP
    +-----------------------+
    |now()    |
    +-----------------------+
    |2022-08-26 21:23:22.716|
    +-----------------------+
    """
    return _invoke_function("now")


def desc(col: "ColumnOrName") -> Column:
    """Returns a sort expression based on the descending order of the given column name.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to sort by in the descending order.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column specifying the order.

    Examples:
    --------
    Sort by the column 'id' in the descending order.

    >>> spark.range(5).orderBy(desc("id")).show()
    +---+
    | id|
    +---+
    |  4|
    |  3|
    |  2|
    |  1|
    |  0|
    +---+
    """
    return Column(_to_column_expr(col).desc())


def asc(col: "ColumnOrName") -> Column:
    """Returns a sort expression based on the ascending order of the given column name.

    .. versionadded:: 1.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to sort by in the ascending order.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the column specifying the order.

    Examples:
    --------
    Sort by the column 'id' in the descending order.

    >>> df = spark.range(5)
    >>> df = df.sort(desc("id"))
    >>> df.show()
    +---+
    | id|
    +---+
    |  4|
    |  3|
    |  2|
    |  1|
    |  0|
    +---+

    Sort by the column 'id' in the ascending order.

    >>> df.orderBy(asc("id")).show()
    +---+
    | id|
    +---+
    |  0|
    |  1|
    |  2|
    |  3|
    |  4|
    +---+
    """
    return Column(_to_column_expr(col).asc())


_DATE_TRUNC_UNITS = {
    "year": "year",
    "yyyy": "year",
    "yy": "year",
    "quarter": "quarter",
    "month": "month",
    "mon": "month",
    "mm": "month",
    "week": "week",
    "day": "day",
    "dd": "day",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
    "millisecond": "millisecond",
    "microsecond": "microsecond",
}


def date_trunc(format: str, timestamp: "ColumnOrName") -> Column:
    """Returns timestamp truncated to the unit specified by the format.

    .. versionadded:: 2.3.0

    Parameters
    ----------
    format : str
        'year', 'yyyy', 'yy', 'month', 'mon', 'mm',
        'day', 'dd', 'hour', 'minute', 'second', 'week', 'quarter'
    timestamp : :class:`~pyspark.sql.Column` or str

    Examples:
    --------
    >>> df = spark.createDataFrame([("1997-02-28 05:02:11",)], ["t"])
    >>> df.select(date_trunc("year", df.t).alias("year")).collect()
    [Row(year=datetime.datetime(1997, 1, 1, 0, 0))]
    >>> df.select(date_trunc("mon", df.t).alias("month")).collect()
    [Row(month=datetime.datetime(1997, 2, 1, 0, 0))]
    """
    unit = _DATE_TRUNC_UNITS.get(format.lower())
    if unit is None:
        return lit(None).cast(_types.TimestampType())
    as_timestamp = Column(_to_column_expr(timestamp)).cast(_types.TimestampType())
    return _invoke_function_over_columns("date_trunc", lit(unit), as_timestamp)


_DATE_PART_FIELDS = {
    **dict.fromkeys(["YEAR", "Y", "YEARS", "YR", "YRS"], "year"),
    "YEAROFWEEK": "isoyear",
    **dict.fromkeys(["QUARTER", "QTR"], "quarter"),
    **dict.fromkeys(["MONTH", "MON", "MONS", "MONTHS"], "month"),
    **dict.fromkeys(["WEEK", "W", "WEEKS"], "week"),
    **dict.fromkeys(["DAY", "D", "DAYS"], "day"),
    **dict.fromkeys(["DAYOFWEEK", "DOW"], "dayofweek"),
    **dict.fromkeys(["DAYOFWEEK_ISO", "DOW_ISO"], "isodow"),
    "DOY": "doy",
    **dict.fromkeys(["HOUR", "H", "HOURS", "HR", "HRS"], "hour"),
    **dict.fromkeys(["MINUTE", "M", "MIN", "MINS", "MINUTES"], "minute"),
    **dict.fromkeys(["SECOND", "S", "SEC", "SECONDS", "SECS"], "second"),
}


def date_part(field: "ColumnOrName", source: "ColumnOrName") -> Column:
    """Extracts a part of the date/timestamp or interval source.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    field : :class:`~pyspark.sql.Column` or str
        selects which part of the source should be extracted, and supported string values
        are as same as the fields of the equivalent function `extract`.
    source : :class:`~pyspark.sql.Column` or str
        a date/timestamp or interval column from where `field` should be extracted.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        a part of the date/timestamp or interval source.

    Examples:
    --------
    >>> import datetime
    >>> df = spark.createDataFrame([(datetime.datetime(2015, 4, 8, 13, 8, 15),)], ["ts"])
    >>> df.select(
    ...     date_part(lit("YEAR"), "ts").alias("year"),
    ...     date_part(lit("month"), "ts").alias("month"),
    ...     date_part(lit("WEEK"), "ts").alias("week"),
    ...     date_part(lit("D"), "ts").alias("day"),
    ...     date_part(lit("M"), "ts").alias("minute"),
    ...     date_part(lit("S"), "ts").alias("second"),
    ... ).collect()
    [Row(year=2015, month=4, week=15, day=8, minute=8, second=Decimal('15.000000'))]
    """
    name = _string_literal(field)
    part = None if name is None else _DATE_PART_FIELDS.get(name.upper())
    if part is None:
        msg = (
            f"[INVALID_EXTRACT_FIELD] Cannot extract {field if name is None else name!r}; "
            "the field must be a literal such as lit('YEAR')."
        )
        raise AnalysisException(msg)
    if part == "second":
        microseconds = _invoke_function_over_columns("date_part", lit("microseconds"), source)
        return (microseconds.cast(_types.DecimalType(8, 0)) * lit(Decimal("0.000001"))).cast(_types.DecimalType(8, 6))
    extracted = _invoke_function_over_columns("date_part", lit(part), source)
    return _result_type(extracted + lit(1) if part == "dayofweek" else extracted, INTEGER)


def extract(field: "ColumnOrName", source: "ColumnOrName") -> Column:
    """Extracts a part of the date/timestamp or interval source.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    field : :class:`~pyspark.sql.Column` or str
        selects which part of the source should be extracted.
    source : :class:`~pyspark.sql.Column` or str
        a date/timestamp or interval column from where `field` should be extracted.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        a part of the date/timestamp or interval source.

    Examples:
    --------
    >>> import datetime
    >>> df = spark.createDataFrame([(datetime.datetime(2015, 4, 8, 13, 8, 15),)], ["ts"])
    >>> df.select(
    ...     extract(lit("YEAR"), "ts").alias("year"),
    ...     extract(lit("month"), "ts").alias("month"),
    ...     extract(lit("WEEK"), "ts").alias("week"),
    ...     extract(lit("D"), "ts").alias("day"),
    ...     extract(lit("M"), "ts").alias("minute"),
    ...     extract(lit("S"), "ts").alias("second"),
    ... ).collect()
    [Row(year=2015, month=4, week=15, day=8, minute=8, second=Decimal('15.000000'))]
    """
    return date_part(field, source)


def datepart(field: "ColumnOrName", source: "ColumnOrName") -> Column:
    """Extracts a part of the date/timestamp or interval source.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    field : :class:`~pyspark.sql.Column` or str
        selects which part of the source should be extracted, and supported string values
        are as same as the fields of the equivalent function `extract`.
    source : :class:`~pyspark.sql.Column` or str
        a date/timestamp or interval column from where `field` should be extracted.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        a part of the date/timestamp or interval source.

    Examples:
    --------
    >>> import datetime
    >>> df = spark.createDataFrame([(datetime.datetime(2015, 4, 8, 13, 8, 15),)], ["ts"])
    >>> df.select(
    ...     datepart(lit("YEAR"), "ts").alias("year"),
    ...     datepart(lit("month"), "ts").alias("month"),
    ...     datepart(lit("WEEK"), "ts").alias("week"),
    ...     datepart(lit("D"), "ts").alias("day"),
    ...     datepart(lit("M"), "ts").alias("minute"),
    ...     datepart(lit("S"), "ts").alias("second"),
    ... ).collect()
    [Row(year=2015, month=4, week=15, day=8, minute=8, second=Decimal('15.000000'))]
    """
    return date_part(field, source)


def date_diff(end: "ColumnOrName", start: "ColumnOrName") -> Column:
    """Returns the number of days from `start` to `end`.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    end : :class:`~pyspark.sql.Column` or column name
        to date column to work on.
    start : :class:`~pyspark.sql.Column` or column name
        from date column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        difference in days between two dates.

    See Also:
    --------
    :meth:`pyspark.sql.functions.dateadd`
    :meth:`pyspark.sql.functions.date_add`
    :meth:`pyspark.sql.functions.date_sub`
    :meth:`pyspark.sql.functions.datediff`
    :meth:`pyspark.sql.functions.timestamp_diff`

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> df = spark.createDataFrame([("2015-04-08", "2015-05-10")], ["d1", "d2"])
    >>> df.select(
    ...     "*", sf.date_diff(sf.col("d1").cast("DATE"), sf.col("d2").cast("DATE"))
    ... ).show()
    +----------+----------+-----------------+
    |        d1|        d2|date_diff(d1, d2)|
    +----------+----------+-----------------+
    |2015-04-08|2015-05-10|              -32|
    +----------+----------+-----------------+

    >>> df.select(
    ...     "*", sf.date_diff(sf.col("d2").cast("DATE"), sf.col("d1").cast("DATE"))
    ... ).show()
    +----------+----------+-----------------+
    |        d1|        d2|date_diff(d2, d1)|
    +----------+----------+-----------------+
    |2015-04-08|2015-05-10|               32|
    +----------+----------+-----------------+
    """
    return _result_type(_invoke_function_over_columns("date_diff", lit("day"), start, end), INTEGER)


def year(col: "ColumnOrName") -> Column:
    """Extract the year of a given date/timestamp as integer.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        year part of the date/timestamp as integer.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2015-04-08",)], ["dt"])
    >>> df.select(year("dt").alias("year")).collect()
    [Row(year=2015)]
    """
    return _result_type(_invoke_function_over_columns("year", col), INTEGER)


def quarter(col: "ColumnOrName") -> Column:
    """Extract the quarter of a given date/timestamp as integer.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        quarter of the date/timestamp as integer.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2015-04-08",)], ["dt"])
    >>> df.select(quarter("dt").alias("quarter")).collect()
    [Row(quarter=2)]
    """
    return _result_type(_invoke_function_over_columns("quarter", col), INTEGER)


def month(col: "ColumnOrName") -> Column:
    """Extract the month of a given date/timestamp as integer.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        month part of the date/timestamp as integer.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2015-04-08",)], ["dt"])
    >>> df.select(month("dt").alias("month")).collect()
    [Row(month=4)]
    """
    return _result_type(_invoke_function_over_columns("month", col), INTEGER)


def dayofweek(col: "ColumnOrName") -> Column:
    """Extract the day of the week of a given date/timestamp as integer.
    Ranges from 1 for a Sunday through to 7 for a Saturday.

    .. versionadded:: 2.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        day of the week for given date/timestamp as integer.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2015-04-08",)], ["dt"])
    >>> df.select(dayofweek("dt").alias("day")).collect()
    [Row(day=4)]
    """  # noqa: D205
    return _result_type(_invoke_function_over_columns("dayofweek", col) + lit(1), INTEGER)


def day(col: "ColumnOrName") -> Column:
    """Extract the day of the month of a given date/timestamp as integer.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        day of the month for given date/timestamp as integer.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2015-04-08",)], ["dt"])
    >>> df.select(day("dt").alias("day")).collect()
    [Row(day=8)]
    """
    return _result_type(_invoke_function_over_columns("day", col), INTEGER)


def dayofmonth(col: "ColumnOrName") -> Column:
    """Extract the day of the month of a given date/timestamp as integer.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        day of the month for given date/timestamp as integer.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2015-04-08",)], ["dt"])
    >>> df.select(dayofmonth("dt").alias("day")).collect()
    [Row(day=8)]
    """
    return day(col)


def dayofyear(col: "ColumnOrName") -> Column:
    """Extract the day of the year of a given date/timestamp as integer.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        day of the year for given date/timestamp as integer.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2015-04-08",)], ["dt"])
    >>> df.select(dayofyear("dt").alias("day")).collect()
    [Row(day=98)]
    """
    return _result_type(_invoke_function_over_columns("dayofyear", col), INTEGER)


def hour(col: "ColumnOrName") -> Column:
    """Extract the hours of a given timestamp as integer.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        hour part of the timestamp as integer.

    Examples:
    --------
    >>> import datetime
    >>> df = spark.createDataFrame([(datetime.datetime(2015, 4, 8, 13, 8, 15),)], ["ts"])
    >>> df.select(hour("ts").alias("hour")).collect()
    [Row(hour=13)]
    """
    return _result_type(_invoke_function_over_columns("hour", col), INTEGER)


def minute(col: "ColumnOrName") -> Column:
    """Extract the minutes of a given timestamp as integer.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        minutes part of the timestamp as integer.

    Examples:
    --------
    >>> import datetime
    >>> df = spark.createDataFrame([(datetime.datetime(2015, 4, 8, 13, 8, 15),)], ["ts"])
    >>> df.select(minute("ts").alias("minute")).collect()
    [Row(minute=8)]
    """
    return _result_type(_invoke_function_over_columns("minute", col), INTEGER)


def second(col: "ColumnOrName") -> Column:
    """Extract the seconds of a given date as integer.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        `seconds` part of the timestamp as integer.

    Examples:
    --------
    >>> import datetime
    >>> df = spark.createDataFrame([(datetime.datetime(2015, 4, 8, 13, 8, 15),)], ["ts"])
    >>> df.select(second("ts").alias("second")).collect()
    [Row(second=15)]
    """
    return _result_type(_invoke_function_over_columns("second", col), INTEGER)


def weekofyear(col: "ColumnOrName") -> Column:
    """Extract the week number of a given date as integer.
    A week is considered to start on a Monday and week 1 is the first week with more than 3 days,
    as defined by ISO 8601.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        `week` of the year for given date as integer.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2015-04-08",)], ["dt"])
    >>> df.select(weekofyear(df.dt).alias("week")).collect()
    [Row(week=15)]
    """  # noqa: D205
    return _result_type(_invoke_function_over_columns("weekofyear", col), INTEGER)


def cos(col: "ColumnOrName") -> Column:
    """Computes cosine of the input column.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        angle in radians

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        cosine of the angle, as if computed by `java.lang.Math.cos()`.

    Examples:
    --------
    >>> import math
    >>> df = spark.range(1)
    >>> df.select(cos(lit(math.pi))).first()
    Row(COS(3.14159...)=-1.0)
    """
    return _invoke_function_over_columns("cos", col)


def acos(col: "ColumnOrName") -> Column:
    """Computes inverse cosine of the input column.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        inverse cosine of `col`, as if computed by `java.lang.Math.acos()`

    Examples:
    --------
    >>> df = spark.range(1, 3)
    >>> df.select(acos(df.id)).show()
    +--------+
    |ACOS(id)|
    +--------+
    |     0.0|
    |     NaN|
    +--------+
    """
    col = _to_column_expr(col)
    # acos domain is [-1, 1]; return NaN for out-of-range values per PySpark semantics
    return Column(
        CaseExpression((col < -1.0) | (col > 1.0), _constant(float("nan"))).otherwise(FunctionExpression("acos", col))
    )


def call_function(funcName: str, *cols: "ColumnOrName") -> Column:
    r"""Call a SQL function.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    funcName : str
        function name that follows the SQL identifier syntax (can be quoted, can be qualified)
    cols : :class:`~pyspark.sql.Column` or str
        column names or :class:`~pyspark.sql.Column`\\s to be used in the function

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        result of executed function.

    Examples:
    --------
    >>> from pyspark.sql.functions import call_udf, col
    >>> from pyspark.sql.types import IntegerType, StringType
    >>> df = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["id", "name"])
    >>> _ = spark.udf.register("intX2", lambda i: i * 2, IntegerType())
    >>> df.select(call_function("intX2", "id")).show()
    +---------+
    |intX2(id)|
    +---------+
    |        2|
    |        4|
    |        6|
    +---------+
    >>> _ = spark.udf.register("strX2", lambda s: s * 2, StringType())
    >>> df.select(call_function("strX2", col("name"))).show()
    +-----------+
    |strX2(name)|
    +-----------+
    |         aa|
    |         bb|
    |         cc|
    +-----------+
    >>> df.select(call_function("avg", col("id"))).show()
    +-------+
    |avg(id)|
    +-------+
    |    2.0|
    +-------+
    >>> _ = spark.sql("CREATE FUNCTION custom_avg AS 'test.org.apache.spark.sql.MyDoubleAvg'")
    ... # doctest: +SKIP
    >>> df.select(call_function("custom_avg", col("id"))).show()
    ... # doctest: +SKIP
    +------------------------------------+
    |spark_catalog.default.custom_avg(id)|
    +------------------------------------+
    |                               102.0|
    +------------------------------------+
    >>> df.select(call_function("spark_catalog.default.custom_avg", col("id"))).show()
    ... # doctest: +SKIP
    +------------------------------------+
    |spark_catalog.default.custom_avg(id)|
    +------------------------------------+
    |                               102.0|
    +------------------------------------+
    """
    if funcName.lower() in _aggregate_function_names():
        return _invoke_aggregate_over_columns(funcName, *cols)
    return _invoke_function_over_columns(funcName, *cols)


@functools.cache
def _aggregate_function_names() -> frozenset[str]:
    with duckdb.connect() as connection:
        query = "SELECT DISTINCT function_name FROM duckdb_functions() WHERE function_type = 'aggregate'"
        return frozenset(name for (name,) in connection.sql(query).fetchall())


def covar_pop(col1: "ColumnOrName", col2: "ColumnOrName") -> Column:
    """Returns a new :class:`~pyspark.sql.Column` for the population covariance of ``col1`` and
    ``col2``.

    .. versionadded:: 2.0.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col1 : :class:`~pyspark.sql.Column` or str
        first column to calculate covariance.
    col1 : :class:`~pyspark.sql.Column` or str
        second column to calculate covariance.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        covariance of these two column values.

    Examples:
    --------
    >>> a = [1] * 10
    >>> b = [1] * 10
    >>> df = spark.createDataFrame(zip(a, b), ["a", "b"])
    >>> df.agg(covar_pop("a", "b").alias("c")).collect()
    [Row(c=0.0)]
    """  # noqa: D205
    return _invoke_aggregate_over_columns("covar_pop", col1, col2)


def covar_samp(col1: "ColumnOrName", col2: "ColumnOrName") -> Column:
    """Returns a new :class:`~pyspark.sql.Column` for the sample covariance of ``col1`` and
    ``col2``.

    .. versionadded:: 2.0.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col1 : :class:`~pyspark.sql.Column` or str
        first column to calculate covariance.
    col1 : :class:`~pyspark.sql.Column` or str
        second column to calculate covariance.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        sample covariance of these two column values.

    Examples:
    --------
    >>> a = [1] * 10
    >>> b = [1] * 10
    >>> df = spark.createDataFrame(zip(a, b), ["a", "b"])
    >>> df.agg(covar_samp("a", "b").alias("c")).collect()
    [Row(c=0.0)]
    """  # noqa: D205
    return _invoke_aggregate_over_columns("covar_samp", col1, col2)


def exp(col: "ColumnOrName") -> Column:
    """Computes the exponential of the given value.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        column to calculate exponential for.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        exponential of the given value.

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(exp(lit(0))).show()
    +------+
    |EXP(0)|
    +------+
    |   1.0|
    +------+
    """
    return _invoke_function_over_columns("exp", col)


def factorial(col: "ColumnOrName") -> Column:
    """Computes the factorial of the given value.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        a column to calculate factorial for.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        factorial of given value.

    Examples:
    --------
    >>> df = spark.createDataFrame([(5,)], ["n"])
    >>> df.select(factorial(df.n).alias("f")).collect()
    [Row(f=120)]
    """
    return Column(
        _evaluated_once(
            _spark_cast(_as_column(col), INTEGER),
            lambda n: CaseExpression(
                n.between(ConstantExpression(0), ConstantExpression(20)),
                FunctionExpression("factorial", n).cast(BIGINT),
            ),
        )
    )


def log2(col: "ColumnOrName") -> Column:
    """Returns the base-2 logarithm of the argument.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        a column to calculate logariphm for.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        logariphm of given value.

    Examples:
    --------
    >>> df = spark.createDataFrame([(4,)], ["a"])
    >>> df.select(log2("a").alias("log2")).show()
    +----+
    |log2|
    +----+
    | 2.0|
    +----+
    """
    return _log("log2", col)


def ln(col: "ColumnOrName") -> Column:
    """Returns the natural logarithm of the argument.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        a column to calculate logariphm for.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        natural logarithm of given value.

    Examples:
    --------
    >>> df = spark.createDataFrame([(4,)], ["a"])
    >>> df.select(ln("a")).show()
    +------------------+
    |             ln(a)|
    +------------------+
    |1.3862943611198906|
    +------------------+
    """
    return _log("ln", col)


def _log(name: str, col: "ColumnOrName") -> Column:
    """Spark's logarithms are NULL for values at or below zero, where DuckDB gives -inf or NaN."""
    x = _to_column_expr(col)
    return Column(CaseExpression(x <= 0, ConstantExpression(None)).otherwise(FunctionExpression(name, x)))


def degrees(col: "ColumnOrName") -> Column:
    """Converts an angle measured in radians to an approximately equivalent angle
    measured in degrees.

    .. versionadded:: 2.1.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        angle in radians

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        angle in degrees, as if computed by `java.lang.Math.toDegrees()`

    Examples:
    --------
    >>> import math
    >>> df = spark.range(1)
    >>> df.select(degrees(lit(math.pi))).first()
    Row(DEGREES(3.14159...)=180.0)
    """  # noqa: D205
    return _invoke_function_over_columns("degrees", col)


def radians(col: "ColumnOrName") -> Column:
    """Converts an angle measured in degrees to an approximately equivalent angle
    measured in radians.

    .. versionadded:: 2.1.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        angle in degrees

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        angle in radians, as if computed by `java.lang.Math.toRadians()`

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(radians(lit(180))).first()
    Row(RADIANS(180)=3.14159...)
    """  # noqa: D205
    return _invoke_function_over_columns("radians", col)


def atan(col: "ColumnOrName") -> Column:
    """Compute inverse tangent of the input column.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        inverse tangent of `col`, as if computed by `java.lang.Math.atan()`

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(atan(df.id)).show()
    +--------+
    |ATAN(id)|
    +--------+
    |     0.0|
    +--------+
    """
    return _invoke_function_over_columns("atan", col)


def atan2(col1: Union["ColumnOrName", float], col2: Union["ColumnOrName", float]) -> Column:
    """.. versionadded:: 1.4.0.

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col1 : str, :class:`~pyspark.sql.Column` or float
        coordinate on y-axis
    col2 : str, :class:`~pyspark.sql.Column` or float
        coordinate on x-axis

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the `theta` component of the point
        (`r`, `theta`)
        in polar coordinates that corresponds to the point
        (`x`, `y`) in Cartesian coordinates,
        as if computed by `java.lang.Math.atan2()`

    Examples:
    --------
    >>> df = spark.range(1)
    >>> df.select(atan2(lit(1), lit(2))).first()
    Row(ATAN2(1, 2)=0.46364...)
    """
    return _invoke_function_over_columns("atan2", _lit_or_column(col1), _lit_or_column(col2))


def tan(col: "ColumnOrName") -> Column:
    """Computes tangent of the input column.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        angle in radians

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        tangent of the given value, as if computed by `java.lang.Math.tan()`

    Examples:
    --------
    >>> import math
    >>> df = spark.range(1)
    >>> df.select(tan(lit(math.radians(45)))).first()
    Row(TAN(0.78539...)=0.99999...)
    """
    return _invoke_function_over_columns("tan", col)


def round(col: "ColumnOrName", scale: int = 0) -> Column:
    """Round the given value to `scale` decimal places using HALF_UP rounding mode if `scale` >= 0
    or at integral part when `scale` < 0.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        input column to round.
    scale : int optional default 0
        scale value.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        rounded values.

    Examples:
    --------
    >>> spark.createDataFrame([(2.5,)], ["a"]).select(round("a", 0).alias("r")).collect()
    [Row(r=3.0)]
    """  # noqa: D205
    return _invoke_function_over_columns("round", col, lit(scale))


def bround(col: "ColumnOrName", scale: int = 0) -> Column:
    """Round the given value to `scale` decimal places using HALF_EVEN rounding mode if `scale` >= 0
    or at integral part when `scale` < 0.

    .. versionadded:: 2.0.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        input column to round.
    scale : int optional default 0
        scale value.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        rounded values.

    Examples:
    --------
    >>> spark.createDataFrame([(2.5,)], ["a"]).select(bround("a", 0).alias("r")).collect()
    [Row(r=2.0)]
    """  # noqa: D205
    return _invoke_function_over_columns("round_even", col, lit(scale))


def get(col: "ColumnOrName", index: Union["ColumnOrName", int]) -> Column:
    """Collection function: Returns element of array at given (0-based) index.
    If the index points outside of the array boundaries, then this function
    returns NULL.

    .. versionadded:: 3.4.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column containing array
    index : :class:`~pyspark.sql.Column` or str or int
        index to check for in array

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        value at given position.

    Notes:
    -----
    The position is not 1 based, but 0 based index.
    Supports Spark Connect.

    See Also:
    --------
    :meth:`element_at`

    Examples:
    --------
    >>> df = spark.createDataFrame([(["a", "b", "c"], 1)], ["data", "index"])
    >>> df.select(get(df.data, 1)).show()
    +------------+
    |get(data, 1)|
    +------------+
    |           b|
    +------------+

    >>> df.select(get(df.data, -1)).show()
    +-------------+
    |get(data, -1)|
    +-------------+
    |         NULL|
    +-------------+

    >>> df.select(get(df.data, 3)).show()
    +------------+
    |get(data, 3)|
    +------------+
    |        NULL|
    +------------+

    >>> df.select(get(df.data, "index")).show()
    +----------------+
    |get(data, index)|
    +----------------+
    |               b|
    +----------------+

    >>> df.select(get(df.data, col("index") - 1)).show()
    +----------------------+
    |get(data, (index - 1))|
    +----------------------+
    |                     a|
    +----------------------+
    """  # noqa: D205
    index = ConstantExpression(index) if isinstance(index, int) else _to_column_expr(index)
    # Spark counts from 0 and gives NULL below it, where DuckDB counts from 1 and counts a negative index from the end.
    element = FunctionExpression("list_extract", _to_column_expr(col), index + 1)
    return Column(CaseExpression(index < ConstantExpression(0), ConstantExpression(None)).otherwise(element))


def initcap(col: "ColumnOrName") -> Column:
    """Translate the first letter of each word to upper case in the sentence.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        string with all first letters are uppercase in each word.

    Examples:
    --------
    >>> spark.createDataFrame([("ab cd",)], ["a"]).select(initcap("a").alias("v")).collect()
    [Row(v='Ab Cd')]
    """
    return Column(
        FunctionExpression(
            "array_to_string",
            FunctionExpression(
                "list_transform",
                FunctionExpression("string_split", _to_column_expr(col), ConstantExpression(" ")),
                _lambda(
                    lambda x: FunctionExpression(
                        "concat",
                        FunctionExpression("upper", FunctionExpression("array_extract", x, 1)),
                        FunctionExpression("lower", FunctionExpression("array_slice", x, 2, -1)),
                    )
                ),
            ),
            ConstantExpression(" "),
        )
    )


def octet_length(col: "ColumnOrName") -> Column:
    r"""Calculates the byte length for the specified string column.

    .. versionadded:: 3.3.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        Source column or strings

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        Byte length of the col

    Examples:
    --------
    >>> from pyspark.sql.functions import octet_length
    >>> spark.createDataFrame([('cat',), ( '\U0001f408',)], ['cat']) \\
    ...      .select(octet_length('cat')).collect()
        [Row(octet_length(cat)=3), Row(octet_length(cat)=4)]
    """
    return _result_type(_invoke_function("octet_length", _to_binary(_to_column_expr(col))), INTEGER)


def hex(col: "ColumnOrName") -> Column:
    """Computes hex value of the given column.

    The column can be :class:`~pyspark.sql.types.StringType`, :class:`~pyspark.sql.types.BinaryType`,
    :class:`~pyspark.sql.types.IntegerType` or :class:`~pyspark.sql.types.LongType`.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        hexadecimal representation of given value as string.

    Examples:
    --------
    >>> spark.createDataFrame([("ABC", 3)], ["a", "b"]).select(hex("a"), hex("b")).collect()
    [Row(hex(a)='414243', hex(b)='3')]
    """
    return _invoke_function_over_columns("hex", col)


def unhex(col: "ColumnOrName") -> Column:
    """Inverse of hex.

    Interprets each pair of characters as a hexadecimal number and converts to the byte representation of number column
    and returns it as a binary column.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        string representation of given hexadecimal value.

    Examples:
    --------
    >>> spark.createDataFrame([("414243",)], ["a"]).select(unhex("a")).collect()
    [Row(unhex(a)=bytearray(b'ABC'))]
    """
    return _invoke_function_over_columns("unhex", col)


def base64(col: "ColumnOrName") -> Column:
    """Computes the BASE64 encoding of a binary column and returns it as a string column.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        BASE64 encoding of string value.

    Examples:
    --------
    >>> df = spark.createDataFrame(["Spark", "PySpark", "Pandas API"], "STRING")
    >>> df.select(base64("value")).show()
    +----------------+
    |   base64(value)|
    +----------------+
    |        U3Bhcms=|
    |    UHlTcGFyaw==|
    |UGFuZGFzIEFQSQ==|
    +----------------+
    """
    return _invoke_function("base64", _to_binary(_to_column_expr(col)))


def unbase64(col: "ColumnOrName") -> Column:
    """Decodes a BASE64 encoded string column and returns it as a binary column.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        encoded string value.

    Examples:
    --------
    >>> df = spark.createDataFrame(["U3Bhcms=", "UHlTcGFyaw==", "UGFuZGFzIEFQSQ=="], "STRING")
    >>> df.select(unbase64("value")).show()
    +--------------------+
    |     unbase64(value)|
    +--------------------+
    |    [53 70 61 72 6B]|
    |[50 79 53 70 61 7...|
    |[50 61 6E 64 61 7...|
    +--------------------+
    """
    return _invoke_function_over_columns("from_base64", col)


def add_months(start: "ColumnOrName", months: Union["ColumnOrName", int]) -> Column:
    """Returns the date that is `months` months after `start`. If `months` is a negative value
    then these amount of months will be deducted from the `start`.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    start : :class:`~pyspark.sql.Column` or str
        date column to work on.
    months : :class:`~pyspark.sql.Column` or str or int
        how many months after the given date to calculate.
        Accepts negative value as well to calculate backwards.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        a date after/before given number of months.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2015-04-08", 2)], ["dt", "add"])
    >>> df.select(add_months(df.dt, 1).alias("next_month")).collect()
    [Row(next_month=datetime.date(2015, 5, 8))]
    >>> df.select(add_months(df.dt, df.add.cast("integer")).alias("next_month")).collect()
    [Row(next_month=datetime.date(2015, 6, 8))]
    >>> df.select(add_months("dt", -2).alias("prev_month")).collect()
    [Row(prev_month=datetime.date(2015, 2, 8))]
    """  # noqa: D205
    months = ConstantExpression(months) if isinstance(months, int) else _to_column_expr(months)
    return _invoke_function("date_add", _to_column_expr(start), FunctionExpression("to_months", months)).cast("date")


def array_join(col: "ColumnOrName", delimiter: str, null_replacement: str | None = None) -> Column:
    """Concatenates the elements of `column` using the `delimiter`. Null values are replaced with
    `null_replacement` if set, otherwise they are ignored.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.
    delimiter : str
        delimiter used to concatenate elements
    null_replacement : str, optional
        if set then null values will be replaced by this value

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        a column of string type. Concatenated values.

    Examples:
    --------
    >>> df = spark.createDataFrame([(["a", "b", "c"],), (["a", None],)], ["data"])
    >>> df.select(array_join(df.data, ",").alias("joined")).collect()
    [Row(joined='a,b,c'), Row(joined='a')]
    >>> df.select(array_join(df.data, ",", "NULL").alias("joined")).collect()
    [Row(joined='a,b,c'), Row(joined='a,NULL')]
    """  # noqa: D205
    col = _to_column_expr(col)
    if null_replacement is not None:
        col = FunctionExpression(
            "list_transform",
            col,
            _lambda(
                lambda x: CaseExpression(x.isnull(), ConstantExpression(null_replacement)).otherwise(x.cast(VARCHAR))
            ),
        )
    else:
        # array_to_string skips NULL elements too, but gives NULL where no element remains.
        col = FunctionExpression("list_filter", col, _lambda(lambda x: x.isnotnull()))
    return _invoke_function("array_to_string", col, ConstantExpression(delimiter))


def array_position(col: "ColumnOrName", value: Any) -> Column:  # noqa: ANN401
    """Collection function: Locates the position of the first occurrence of the given value
    in the given array. Returns null if either of the arguments are null.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Notes:
    -----
    The position is not zero based, but 1 based index. Returns 0 if the given
    value could not be found in the array.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to work on.
    value : Any
        value to look for.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        position of the value in the given array if found and 0 otherwise.

    Examples:
    --------
    >>> df = spark.createDataFrame([(["c", "b", "a"],), ([],)], ["data"])
    >>> df.select(array_position(df.data, "a")).collect()
    [Row(array_position(data, a)=3), Row(array_position(data, a)=0)]
    """  # noqa: D205
    arr, element = _to_column_expr(col), _get_expr(value)
    position = CoalesceOperator(FunctionExpression("list_position", arr, element), ConstantExpression(0))
    return _null_if_null(position.cast(BIGINT), arr, element)


def array_prepend(col: "ColumnOrName", value: Any) -> Column:  # noqa: ANN401
    """Collection function: Returns an array containing element as
    well as all elements from array. The new element is positioned
    at the beginning of the array.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column containing array
    value :
        a literal value, or a :class:`~pyspark.sql.Column` expression.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        an array excluding given value.

    Examples:
    --------
    >>> df = spark.createDataFrame([([2, 3, 4],), ([],)], ["data"])
    >>> df.select(array_prepend(df.data, 1)).collect()
    [Row(array_prepend(data, 1)=[1, 2, 3, 4]), Row(array_prepend(data, 1)=[1])]
    """  # noqa: D205
    arr = _to_column_expr(col)
    prepended = FunctionExpression("list_prepend", _get_expr(value), arr)
    return _null_if_null(prepended, arr)


def array_repeat(col: "ColumnOrName", count: Union["ColumnOrName", int]) -> Column:
    """Collection function: creates an array containing a column repeated count times.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        column name or column that contains the element to be repeated
    count : :class:`~pyspark.sql.Column` or str or int
        column name, column, or int containing the number of times to repeat the first argument

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        an array of repeated elements.

    Examples:
    --------
    >>> df = spark.createDataFrame([("ab",)], ["data"])
    >>> df.select(array_repeat(df.data, 3).alias("r")).collect()
    [Row(r=['ab', 'ab', 'ab'])]
    """
    element = _to_column_expr(col)
    count = ConstantExpression(count) if isinstance(count, int) else _to_column_expr(count)
    size = FunctionExpression("greatest", count, ConstantExpression(0))
    repeated = FunctionExpression("list_resize", FunctionExpression("list_value", element), size, element)
    return _null_if_null(repeated, count)


def array_size(col: "ColumnOrName") -> Column:
    """Returns the total number of elements in the array. The function returns null for null input.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        total number of elements in the array.

    Examples:
    --------
    >>> df = spark.createDataFrame([([2, 1, 3],), (None,)], ["data"])
    >>> df.select(array_size(df.data).alias("r")).collect()
    [Row(r=3), Row(r=None)]
    """
    return _result_type(_invoke_function_over_columns("len", col), INTEGER)


def array_sort(col: "ColumnOrName", comparator: Callable[[Column, Column], Column] | None = None) -> Column:
    """Collection function: sorts the input array in ascending order. The elements of the input array
    must be orderable. Null elements will be placed at the end of the returned array.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Can take a `comparator` function.

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column or expression
    comparator : callable, optional
        A binary ``(Column, Column) -> Column: ...``.
        The comparator will take two
        arguments representing two elements of the array. It returns a negative integer, 0, or a
        positive integer as the first element is less than, equal to, or greater than the second
        element. If the comparator function returns null, the function will fail and raise an error.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        sorted array.

    Examples:
    --------
    >>> df = spark.createDataFrame([([2, 1, None, 3],), ([1],), ([],)], ["data"])
    >>> df.select(array_sort(df.data).alias("r")).collect()
    [Row(r=[1, 2, 3, None]), Row(r=[1]), Row(r=[])]
    >>> df = spark.createDataFrame(
    ...     [(["foo", "foobar", None, "bar"],), (["foo"],), ([],)], ["data"]
    ... )
    >>> df.select(
    ...     array_sort(
    ...         "data",
    ...         lambda x, y: when(x.isNull() | y.isNull(), lit(0)).otherwise(
    ...             length(y) - length(x)
    ...         ),
    ...     ).alias("r")
    ... ).collect()
    [Row(r=['foobar', 'foo', None, 'bar']), Row(r=['foo']), Row(r=[])]
    """  # noqa: D205
    if comparator is not None:
        msg = "comparator is not yet supported"
        raise ContributionsAcceptedError(msg)
    else:
        return _invoke_function_over_columns("list_sort", col, lit("ASC"), lit("NULLS LAST"))


def sort_array(col: "ColumnOrName", asc: bool = True) -> Column:
    """Collection function: sorts the input array in ascending or descending order according
    to the natural ordering of the array elements. Null elements will be placed at the beginning
    of the returned array in ascending order or at the end of the returned array in descending
    order.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column or expression
    asc : bool, optional
        whether to sort in ascending or descending order. If `asc` is True (default)
        then ascending and if False then descending.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        sorted array.

    Examples:
    --------
    >>> df = spark.createDataFrame([([2, 1, None, 3],), ([1],), ([],)], ["data"])
    >>> df.select(sort_array(df.data).alias("r")).collect()
    [Row(r=[None, 1, 2, 3]), Row(r=[1]), Row(r=[])]
    >>> df.select(sort_array(df.data, asc=False).alias("r")).collect()
    [Row(r=[3, 2, 1, None]), Row(r=[1]), Row(r=[])]
    """  # noqa: D205
    if asc:
        order = "ASC"
        null_order = "NULLS FIRST"
    else:
        order = "DESC"
        null_order = "NULLS LAST"
    return _invoke_function_over_columns("list_sort", col, lit(order), lit(null_order))


def split(str: "ColumnOrName", pattern: str, limit: int = -1) -> Column:
    """Splits str around matches of the given pattern.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        a string expression to split
    pattern : str
        a string representing a regular expression. The regex string should be
        a Java regular expression.
    limit : int, optional
        an integer which controls the number of times `pattern` is applied.

        * ``limit > 0``: The resulting array's length will not be more than `limit`, and the
                         resulting array's last entry will contain all input beyond the last
                         matched pattern.
        * ``limit <= 0``: `pattern` will be applied as many times as possible, and the resulting
                          array can be of any size.

        .. versionchanged:: 3.0
           `split` now takes an optional `limit` field. If not provided, default limit value is -1.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        array of separated strings.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [("oneAtwoBthreeC",)],
    ...     [
    ...         "s",
    ...     ],
    ... )
    >>> df.select(split(df.s, "[ABC]", 2).alias("s")).collect()
    [Row(s=['one', 'twoBthreeC'])]
    >>> df.select(split(df.s, "[ABC]", -1).alias("s")).collect()
    [Row(s=['one', 'two', 'three', ''])]
    """
    if isinstance(limit, Column) or limit > 0:
        # Unclear how to implement this in DuckDB as we'd need to map back from the split array
        # to the original array which is tricky with regular expressions.
        msg = "limit is not yet supported"
        raise ContributionsAcceptedError(msg)
    return _invoke_function_over_columns("regexp_split_to_array", str, lit(pattern))


def split_part(src: "ColumnOrName", delimiter: "ColumnOrName", partNum: "ColumnOrName") -> Column:
    """Splits `str` by delimiter and return requested part of the split (1-based).
    If any input is null, returns null. if `partNum` is out of range of split parts,
    returns empty string. If `partNum` is 0, throws an error. If `partNum` is negative,
    the parts are counted backward from the end of the string.
    If the `delimiter` is an empty string, the `str` is not split.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    src : :class:`~pyspark.sql.Column` or str
        A column of string to be splited.
    delimiter : :class:`~pyspark.sql.Column` or str
        A column of string, the delimiter used for split.
    partNum : :class:`~pyspark.sql.Column` or str
        A column of string, requested part of the split (1-based).

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             "11.12.13",
    ...             ".",
    ...             3,
    ...         )
    ...     ],
    ...     ["a", "b", "c"],
    ... )
    >>> df.select(split_part(df.a, df.b, df.c).alias("r")).collect()
    [Row(r='13')]
    """  # noqa: D205
    src = _to_column_expr(src)
    delimiter = _to_column_expr(delimiter)
    partNum = _to_column_expr(partNum)
    part = FunctionExpression("split_part", src, delimiter, partNum)

    unsplit = CaseExpression(partNum.isin(ConstantExpression(1), ConstantExpression(-1)), src).otherwise(
        ConstantExpression("")
    )
    return _null_if_null(
        CaseExpression(
            partNum == ConstantExpression(0),
            _spark_error(
                "[INVALID_INDEX_OF_ZERO] The index 0 is invalid. "
                "An index shall be either < 0 or > 0 (the first element has index 1)."
            ),
        )
        .when(delimiter == ConstantExpression(""), unsplit)
        .otherwise(part),
        src,
        delimiter,
        partNum,
    )


def stddev_samp(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the unbiased sample standard deviation of
    the expression in a group.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        standard deviation of given column.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(6).select(sf.stddev_samp("id")).show()
    +------------------+
    |   stddev_samp(id)|
    +------------------+
    |1.8708286933869...|
    +------------------+
    """  # noqa: D205
    return _invoke_aggregate_over_columns("stddev_samp", col)


def stddev(col: "ColumnOrName") -> Column:
    """Aggregate function: alias for stddev_samp.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        standard deviation of given column.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(6).select(sf.stddev("id")).show()
    +------------------+
    |        stddev(id)|
    +------------------+
    |1.8708286933869...|
    +------------------+
    """
    return stddev_samp(col)


def std(col: "ColumnOrName") -> Column:
    """Aggregate function: alias for stddev_samp.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        standard deviation of given column.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(6).select(sf.std("id")).show()
    +------------------+
    |           std(id)|
    +------------------+
    |1.8708286933869...|
    +------------------+
    """
    return stddev_samp(col)


def stddev_pop(col: "ColumnOrName") -> Column:
    """Aggregate function: returns population standard deviation of
    the expression in a group.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        standard deviation of given column.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.range(6).select(sf.stddev_pop("id")).show()
    +-----------------+
    |   stddev_pop(id)|
    +-----------------+
    |1.707825127659...|
    +-----------------+
    """  # noqa: D205
    return _invoke_aggregate_over_columns("stddev_pop", col)


def var_pop(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the population variance of the values in a group.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        variance of given column.

    Examples:
    --------
    >>> df = spark.range(6)
    >>> df.select(var_pop(df.id)).first()
    Row(var_pop(id)=2.91666...)
    """
    return _invoke_aggregate_over_columns("var_pop", col)


def var_samp(col: "ColumnOrName") -> Column:
    """Aggregate function: returns the unbiased sample variance of
    the values in a group.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        variance of given column.

    Examples:
    --------
    >>> df = spark.range(6)
    >>> df.select(var_samp(df.id)).show()
    +------------+
    |var_samp(id)|
    +------------+
    |         3.5|
    +------------+
    """  # noqa: D205
    return _invoke_aggregate_over_columns("var_samp", col)


def variance(col: "ColumnOrName") -> Column:
    """Aggregate function: alias for var_samp.

    .. versionadded:: 1.6.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target column to compute on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        variance of given column.

    Examples:
    --------
    >>> df = spark.range(6)
    >>> df.select(variance(df.id)).show()
    +------------+
    |var_samp(id)|
    +------------+
    |         3.5|
    +------------+
    """
    return var_samp(col)


def weekday(col: "ColumnOrName") -> Column:
    """Returns the day of the week for date/timestamp (0 = Monday, 1 = Tuesday, ..., 6 = Sunday).

    .. versionadded:: 3.5.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        target date/timestamp column to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        the day of the week for date/timestamp (0 = Monday, 1 = Tuesday, ..., 6 = Sunday).

    Examples:
    --------
    >>> df = spark.createDataFrame([("2015-04-08",)], ["dt"])
    >>> df.select(weekday("dt").alias("day")).show()
    +---+
    |day|
    +---+
    |  2|
    +---+
    """
    return _result_type(_invoke_function_over_columns("isodow", col) - 1, INTEGER)


def zeroifnull(col: "ColumnOrName") -> Column:
    """Returns zero if `col` is null, or `col` otherwise.

    .. versionadded:: 4.0.0

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str

    Examples:
    --------
    >>> df = spark.createDataFrame([(None,), (1,)], ["a"])
    >>> df.select(zeroifnull(df.a).alias("result")).show()
    +------+
    |result|
    +------+
    |     0|
    |     1|
    +------+
    """
    return coalesce(col, lit(0))


_YEAR_PATTERN_FIELDS = {1: "variable", 3: "variable", 4: "fixed"}
_NUMERIC_PATTERN_FIELDS = {
    "M": ("month", "%m"),
    "L": ("month", "%m"),
    "d": ("day", "%d"),
    "H": ("hour", "%H"),
    "h": ("hour12", "%I"),
    "m": ("minute", "%M"),
    "s": ("second", "%S"),
}
_UNSUPPORTED_PATTERN_LETTERS = {
    "G": "era",
    "D": "day of year",
    "K": "hour of am-pm (0-11)",
    "k": "clock hour of day (1-24)",
    "V": "time zone ID",
    "z": "time zone name",
    "O": "localized zone offset",
    "X": "zone offset",
    "x": "zone offset",
    "Z": "zone offset",
}


def _datetime_pattern_error(pattern: str, reason: str) -> IllegalArgumentException:
    return IllegalArgumentException(f"Invalid datetime pattern {pattern!r}: {reason}")


def _unsupported_datetime_pattern(pattern: str, reason: str) -> ContributionsAcceptedError:
    return ContributionsAcceptedError(f"Datetime pattern {pattern!r} is not supported for parsing: {reason}")


def _tokenize_datetime_pattern(pattern: str) -> list[tuple[str, int] | str]:
    """Split a Spark datetime pattern into (letter, count) fields and literal strings."""
    tokens: list[tuple[str, int] | str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "'":
            if pattern.startswith("''", i):
                tokens.append("'")
                i += 2
                continue
            text = []
            i += 1
            while True:
                if i == len(pattern):
                    raise _datetime_pattern_error(pattern, "it ends with an incomplete quoted literal")
                if pattern.startswith("''", i):
                    text.append("'")
                    i += 2
                elif pattern[i] == "'":
                    i += 1
                    break
                else:
                    text.append(pattern[i])
                    i += 1
            tokens.append("".join(text))
        elif char.isascii() and char.isalpha():
            end = i
            while end < len(pattern) and pattern[end] == char:
                end += 1
            tokens.append((char, end - i))
            i = end
        elif char in "#{}":
            raise _datetime_pattern_error(pattern, f"{char!r} is reserved")
        elif char in "[]":
            raise _unsupported_datetime_pattern(pattern, "optional sections")
        else:
            tokens.append(char)
            i += 1
    return tokens


def _translate_datetime_pattern_field(pattern: str, letter: str, count: int) -> tuple[str, str, str]:
    """Translate one pattern field to (field, DuckDB strptime code, kind).

    The kind is "fixed" or "variable" for digits of a width Spark fixes or not, "fraction" or "text".
    """
    if letter in _UNSUPPORTED_PATTERN_LETTERS:
        raise _unsupported_datetime_pattern(pattern, f"{letter!r} ({_UNSUPPORTED_PATTERN_LETTERS[letter]})")
    if letter == "y":
        if count > 6:
            raise _datetime_pattern_error(pattern, "too many 'y' letters")
        if count == 2:
            reason = "'yy', which DuckDB reads as the years 1969 to 2068 and Spark as 2000 to 2099"
            raise _unsupported_datetime_pattern(pattern, reason)
        if count not in _YEAR_PATTERN_FIELDS:
            raise _unsupported_datetime_pattern(pattern, f"{'y' * count!r}, as DuckDB reads at most four year digits")
        return "year", "%Y", _YEAR_PATTERN_FIELDS[count]
    if letter in "ML" and count in (3, 4):
        return "month", "%b" if count == 3 else "%B", "text"
    if letter in _NUMERIC_PATTERN_FIELDS:
        if count > 2:
            raise _datetime_pattern_error(pattern, f"too many {letter!r} letters")
        field, code = _NUMERIC_PATTERN_FIELDS[letter]
        return field, code, "variable" if count == 1 else "fixed"
    if letter == "S":
        if count > 9:
            raise _datetime_pattern_error(pattern, "too many 'S' letters")
        if count > 6:
            raise _unsupported_datetime_pattern(pattern, "nanoseconds, which DuckDB rounds where Spark truncates them")
        return "fraction", "%g" if count <= 3 else "%f", "fraction"
    if letter == "a":
        if count > 1:
            raise _datetime_pattern_error(pattern, "too many 'a' letters")
        return "ampm", "%p", "text"
    if letter in "EFQq":
        raise _datetime_pattern_error(pattern, f"{letter!r} can only be used for formatting")
    if letter in "YWceuw":
        raise _datetime_pattern_error(pattern, f"{letter!r} is a week-based field, which Spark does not support")
    raise _datetime_pattern_error(pattern, f"{letter!r} is not a pattern letter; quote literal text, as in 'T'")


def _translate_datetime_pattern(pattern: str) -> str:
    """Translate a Spark datetime pattern for parsing into the DuckDB strptime format of the same fields.

    DuckDB's strptime accepts fewer digits and more spaces than Spark; a pattern for which it would read text that
    Spark accepts to another value raises.
    """
    duckdb_format, fields = [], []
    previous_kind = "text"
    for token in _tokenize_datetime_pattern(pattern):
        if isinstance(token, str):
            if previous_kind in ("variable", "fraction") and token[:1].isdigit():
                raise _unsupported_datetime_pattern(pattern, "a variable width field followed by a digit")
            duckdb_format.append(token.replace("%", "%%"))
            previous_kind = "text"
            continue
        field, code, kind = _translate_datetime_pattern_field(pattern, *token)
        # DuckDB reads digits greedily; Spark fails on a variable width field next to another number.
        follows_variable = previous_kind in ("variable", "fraction") and kind != "text"
        if follows_variable or (previous_kind != "text" and kind == "variable"):
            raise _unsupported_datetime_pattern(pattern, "a variable width field next to another numeric field")
        if field in fields:
            raise _unsupported_datetime_pattern(pattern, f"the field of {token[0]!r} appears more than once")
        fields.append(field)
        duckdb_format.append(code)
        previous_kind = kind
    if "hour" in fields and ("hour12" in fields or "ampm" in fields):
        raise _unsupported_datetime_pattern(pattern, "'H' combined with 'h' or 'a'")
    if "year" not in fields:
        raise _unsupported_datetime_pattern(pattern, "no year, where DuckDB reads the year 1900 and Spark 1970")
    if "hour12" in fields and "ampm" not in fields:
        reason = "'h' without 'a', where DuckDB reads 12 as noon and Spark as midnight"
        raise _unsupported_datetime_pattern(pattern, reason)
    return "".join(duckdb_format)


def _string_literal(col: "ColumnOrName") -> str | None:
    """Return the string behind a literal column such as ``lit("yyyy-MM-dd")``, or None for any other column."""
    rendered = str(col.expr) if isinstance(col, Column) else ""
    if len(rendered) < 2 or rendered[0] != "'" or rendered[-1] != "'" or "'" in rendered[1:-1].replace("''", ""):
        return None
    return rendered[1:-1].replace("''", "'")


def _datetime_pattern_argument(format: Optional["ColumnOrName"]) -> str | None:
    if format is None:
        return None
    pattern = _string_literal(format)
    if pattern is None:
        msg = (
            "Only a literal datetime pattern, such as lit('yyyy-MM-dd'), is supported as format; "
            "a column, or a str, which names a column, is not."
        )
        raise ContributionsAcceptedError(msg)
    return pattern


def _parse_datetime(col: "ColumnOrName", pattern: str, *, raise_on_error: bool) -> Column:
    source = _to_column_expr(col)
    duckdb_format = ConstantExpression(_translate_datetime_pattern(pattern))
    parsed = FunctionExpression("strptime", source, duckdb_format)
    if raise_on_error:
        return Column(parsed)
    # DuckDB's try_strptime reads the words epoch and infinity as 1900-01-01, where strptime reads what they name.
    return Column(CaseExpression(FunctionExpression("try_strptime", source, duckdb_format).isnotnull(), parsed))


def _to_date_or_timestamp(
    col: "ColumnOrName", spark_datatype: _types.DataType, format: str | None = None, *, try_cast: bool = False
) -> Column:
    if format is not None:
        if not isinstance(format, str):
            raise PySparkTypeError(
                error_class="NOT_STR", message_parameters={"arg_name": "format", "arg_type": type(format).__name__}
            )
        return _parse_datetime(col, format, raise_on_error=not try_cast).cast(spark_datatype)
    if try_cast:
        return Column(SQLExpression(f"TRY_CAST({_to_column_expr(col)} AS {spark_datatype.duckdb_type})"))
    return Column(_to_column_expr(col)).cast(spark_datatype)


def to_date(col: "ColumnOrName", format: str | None = None) -> Column:
    """Converts a :class:`~pyspark.sql.Column` into :class:`pyspark.sql.types.DateType`
    using the optionally specified format. Specify formats according to `datetime pattern`_.
    By default, it follows casting rules to :class:`pyspark.sql.types.DateType` if the format
    is omitted. Equivalent to ``col.cast("date")``.

    .. _datetime pattern: https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html

    .. versionadded:: 2.2.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        input column of values to convert.
    format: str, optional
        format to use to convert date values.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        date value as :class:`pyspark.sql.types.DateType` type.

    Examples:
    --------
    >>> df = spark.createDataFrame([("1997-02-28 10:30:00",)], ["t"])
    >>> df.select(to_date(df.t).alias("date")).collect()
    [Row(date=datetime.date(1997, 2, 28))]

    >>> df = spark.createDataFrame([("1997-02-28 10:30:00",)], ["t"])
    >>> df.select(to_date(df.t, "yyyy-MM-dd HH:mm:ss").alias("date")).collect()
    [Row(date=datetime.date(1997, 2, 28))]
    """  # noqa: D205
    return _to_date_or_timestamp(col, _types.DateType(), format)


def to_timestamp(col: "ColumnOrName", format: str | None = None) -> Column:
    """Converts a :class:`~pyspark.sql.Column` into :class:`pyspark.sql.types.TimestampType`
    using the optionally specified format. Specify formats according to `datetime pattern`_.
    By default, it follows casting rules to :class:`pyspark.sql.types.TimestampType` if the format
    is omitted. Equivalent to ``col.cast("timestamp")``.

    .. _datetime pattern: https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html

    .. versionadded:: 2.2.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        column values to convert.
    format: str, optional
        format to use to convert timestamp values.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        timestamp value as :class:`pyspark.sql.types.TimestampType` type.

    Examples:
    --------
    >>> df = spark.createDataFrame([("1997-02-28 10:30:00",)], ["t"])
    >>> df.select(to_timestamp(df.t).alias("dt")).collect()
    [Row(dt=datetime.datetime(1997, 2, 28, 10, 30))]

    >>> df = spark.createDataFrame([("1997-02-28 10:30:00",)], ["t"])
    >>> df.select(to_timestamp(df.t, "yyyy-MM-dd HH:mm:ss").alias("dt")).collect()
    [Row(dt=datetime.datetime(1997, 2, 28, 10, 30))]
    """  # noqa: D205
    return _to_date_or_timestamp(col, _types.TimestampType(), format)


def to_timestamp_ltz(
    timestamp: "ColumnOrName",
    format: Optional["ColumnOrName"] = None,
) -> Column:
    """Parses the `timestamp` with the `format` to a timestamp without time zone.
    Returns null with invalid input.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    timestamp : :class:`~pyspark.sql.Column` or str
        Input column or strings.
    format : :class:`~pyspark.sql.Column` or str, optional
        format to use to convert type `TimestampType` timestamp values.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2016-12-31",)], ["e"])
    >>> df.select(to_timestamp_ltz(df.e, lit("yyyy-MM-dd")).alias("r")).collect()
    ... # doctest: +SKIP
    [Row(r=datetime.datetime(2016, 12, 31, 0, 0))]

    >>> df = spark.createDataFrame([("2016-12-31",)], ["e"])
    >>> df.select(to_timestamp_ltz(df.e).alias("r")).collect()
    ... # doctest: +SKIP
    [Row(r=datetime.datetime(2016, 12, 31, 0, 0))]
    """  # noqa: D205
    pattern = _datetime_pattern_argument(format)
    return _to_date_or_timestamp(timestamp, _types.TimestampType(), pattern)


def to_timestamp_ntz(
    timestamp: "ColumnOrName",
    format: Optional["ColumnOrName"] = None,
) -> Column:
    """Parses the `timestamp` with the `format` to a timestamp without time zone.
    Returns null with invalid input.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    timestamp : :class:`~pyspark.sql.Column` or str
        Input column or strings.
    format : :class:`~pyspark.sql.Column` or str, optional
        format to use to convert type `TimestampNTZType` timestamp values.

    Examples:
    --------
    >>> df = spark.createDataFrame([("2016-04-08",)], ["e"])
    >>> df.select(to_timestamp_ntz(df.e, lit("yyyy-MM-dd")).alias("r")).collect()
    ... # doctest: +SKIP
    [Row(r=datetime.datetime(2016, 4, 8, 0, 0))]

    >>> df = spark.createDataFrame([("2016-04-08",)], ["e"])
    >>> df.select(to_timestamp_ntz(df.e).alias("r")).collect()
    ... # doctest: +SKIP
    [Row(r=datetime.datetime(2016, 4, 8, 0, 0))]
    """  # noqa: D205
    pattern = _datetime_pattern_argument(format)
    return _to_date_or_timestamp(timestamp, _types.TimestampNTZType(), pattern)


def try_to_timestamp(col: "ColumnOrName", format: Optional["ColumnOrName"] = None) -> Column:
    """Parses the `col` with the `format` to a timestamp. The function always
    returns null on an invalid input with/without ANSI SQL mode enabled. The result data type is
    consistent with the value of configuration `spark.sql.timestampType`.
    .. versionadded:: 3.5.0
    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        column values to convert.
    format: str, optional
        format to use to convert timestamp values.

    Examples:
    --------
    >>> df = spark.createDataFrame([("1997-02-28 10:30:00",)], ["t"])
    >>> df.select(try_to_timestamp(df.t).alias("dt")).collect()
    [Row(dt=datetime.datetime(1997, 2, 28, 10, 30))]
    >>> df.select(try_to_timestamp(df.t, lit("yyyy-MM-dd HH:mm:ss")).alias("dt")).collect()
    [Row(dt=datetime.datetime(1997, 2, 28, 10, 30))]
    """  # noqa: D205, D415
    pattern = _datetime_pattern_argument(format)
    return _to_date_or_timestamp(col, _types.TimestampType(), pattern, try_cast=True)


def substr(str: "ColumnOrName", pos: "ColumnOrName", len: Optional["ColumnOrName"] = None) -> Column:
    """Returns the substring of `str` that starts at `pos` and is of length `len`,
    or the slice of byte array that starts at `pos` and is of length `len`.

    .. versionadded:: 3.5.0

    Parameters
    ----------
    src : :class:`~pyspark.sql.Column` or str
        A column of string.
    pos : :class:`~pyspark.sql.Column` or str
        A column of string, the substring of `str` that starts at `pos`.
    len : :class:`~pyspark.sql.Column` or str, optional
        A column of string, the substring of `str` is of length `len`.

    Examples:
    --------
    >>> import pyspark.sql.functions as sf
    >>> spark.createDataFrame(
    ...     [
    ...         (
    ...             "Spark SQL",
    ...             5,
    ...             1,
    ...         )
    ...     ],
    ...     ["a", "b", "c"],
    ... ).select(sf.substr("a", "b", "c")).show()
    +---------------+
    |substr(a, b, c)|
    +---------------+
    |              k|
    +---------------+

    >>> import pyspark.sql.functions as sf
    >>> spark.createDataFrame(
    ...     [
    ...         (
    ...             "Spark SQL",
    ...             5,
    ...             1,
    ...         )
    ...     ],
    ...     ["a", "b", "c"],
    ... ).select(sf.substr("a", "b")).show()
    +------------------------+
    |substr(a, b, 2147483647)|
    +------------------------+
    |                   k SQL|
    +------------------------+
    """  # noqa: D205
    return _spark_substring(_to_column_expr(str), _to_column_expr(pos), None if len is None else _to_column_expr(len))


def _unix_diff(col: "ColumnOrName", part: str) -> Column:
    # Days count calendar dates; the smaller units count from the epoch instant, which is midnight UTC.
    epoch = lit("1970-01-01").cast("date") if part == "days" else lit("1970-01-01 00:00:00+00:00").cast("timestamp")
    return _invoke_function_over_columns("date_diff", lit(part), epoch, col)


def unix_date(col: "ColumnOrName") -> Column:
    """Returns the number of days since 1970-01-01.

    .. versionadded:: 3.5.0

    Examples:
    --------
    >>> spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
    >>> df = spark.createDataFrame([("1970-01-02",)], ["t"])
    >>> df.select(unix_date(to_date(df.t)).alias("n")).collect()
    [Row(n=1)]
    >>> spark.conf.unset("spark.sql.session.timeZone")
    """
    return _result_type(_unix_diff(col, "days"), INTEGER)


def unix_micros(col: "ColumnOrName") -> Column:
    """Returns the number of microseconds since 1970-01-01 00:00:00 UTC.

    .. versionadded:: 3.5.0

    Examples:
    --------
    >>> spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
    >>> df = spark.createDataFrame([("2015-07-22 10:00:00",)], ["t"])
    >>> df.select(unix_micros(to_timestamp(df.t)).alias("n")).collect()
    [Row(n=1437584400000000)]
    >>> spark.conf.unset("spark.sql.session.timeZone")
    """
    return _unix_diff(col, "microseconds")


def unix_millis(col: "ColumnOrName") -> Column:
    """Returns the number of milliseconds since 1970-01-01 00:00:00 UTC.
    Truncates higher levels of precision.

    .. versionadded:: 3.5.0

    Examples:
    --------
    >>> spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
    >>> df = spark.createDataFrame([("2015-07-22 10:00:00",)], ["t"])
    >>> df.select(unix_millis(to_timestamp(df.t)).alias("n")).collect()
    [Row(n=1437584400000)]
    >>> spark.conf.unset("spark.sql.session.timeZone")
    """  # noqa: D205
    return _unix_diff(col, "milliseconds")


def unix_seconds(col: "ColumnOrName") -> Column:
    """Returns the number of seconds since 1970-01-01 00:00:00 UTC.
    Truncates higher levels of precision.

    .. versionadded:: 3.5.0

    Examples:
    --------
    >>> spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
    >>> df = spark.createDataFrame([("2015-07-22 10:00:00",)], ["t"])
    >>> df.select(unix_seconds(to_timestamp(df.t)).alias("n")).collect()
    [Row(n=1437584400)]
    >>> spark.conf.unset("spark.sql.session.timeZone")
    """  # noqa: D205
    return _unix_diff(col, "seconds")


def arrays_overlap(a1: "ColumnOrName", a2: "ColumnOrName") -> Column:
    """Collection function: returns true if the arrays contain any common non-null element; if not,
    returns null if both the arrays are non-empty and any of them contains a null element; returns
    false otherwise.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        a column of Boolean type.

    Examples:
    --------
    >>> df = spark.createDataFrame([(["a", "b"], ["b", "c"]), (["a"], ["b", "c"])], ["x", "y"])
    >>> df.select(arrays_overlap(df.x, df.y).alias("overlap")).collect()
    [Row(overlap=True), Row(overlap=False)]
    """  # noqa: D205
    a1 = _to_column_expr(a1)
    a2 = _to_column_expr(a2)

    a1_has_null = _list_contains_null(a1)
    a2_has_null = _list_contains_null(a2)

    return _null_if_null(
        CaseExpression(FunctionExpression("list_has_any", a1, a2), ConstantExpression(True)).otherwise(
            CaseExpression(
                (FunctionExpression("len", a1) > 0) & (FunctionExpression("len", a2) > 0) & (a1_has_null | a2_has_null),
                ConstantExpression(None),
            ).otherwise(ConstantExpression(False))
        ),
        a1,
        a2,
    )


def _list_contains_null(c: ColumnExpression) -> Expression:
    return FunctionExpression("list_count", c) < FunctionExpression("len", c)


def arrays_zip(*cols: "ColumnOrName") -> Column:
    """Collection function: Returns a merged array of structs in which the N-th struct contains all
    N-th values of input arrays. If one of the arrays is shorter than others then
    resulting struct type value will be a `null` for missing elements.

    .. versionadded:: 2.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    cols : :class:`~pyspark.sql.Column` or str
        columns of arrays to be merged.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        merged array of entries.

    Examples:
    --------
    >>> from pyspark.sql.functions import arrays_zip
    >>> df = spark.createDataFrame(
    ...     [([1, 2, 3], [2, 4, 6], [3, 6])], ["vals1", "vals2", "vals3"]
    ... )
    >>> df = df.select(arrays_zip(df.vals1, df.vals2, df.vals3).alias("zipped"))
    >>> df.show(truncate=False)
    +------------------------------------+
    |zipped                              |
    +------------------------------------+
    |[{1, 2, 3}, {2, 4, 6}, {3, 6, NULL}]|
    +------------------------------------+
    >>> df.printSchema()
    root
     |-- zipped: array (nullable = true)
     |    |-- element: struct (containsNull = false)
     |    |    |-- vals1: long (nullable = true)
     |    |    |-- vals2: long (nullable = true)
     |    |    |-- vals3: long (nullable = true)
    """  # noqa: D205
    if not cols:
        return Column(FunctionExpression("list_value").cast(duckdb.list_type(duckdb.struct_type({}))))
    columns = [_as_column(c) for c in cols]
    arrays = [c.expr for c in columns]
    names = [name if (name := c._struct_field()) is not None else f"{i}" for i, c in enumerate(columns)]

    def zipped_struct(row: Expression) -> Expression:
        fields = [FunctionExpression("struct_extract_at", row, ConstantExpression(i)) for i in range(1, len(names) + 1)]
        return _struct_pack(names, fields)

    zipped = FunctionExpression("list_transform", FunctionExpression("list_zip", *arrays), _lambda(zipped_struct))
    return _null_if_null(zipped, *arrays)


def substring(str: "ColumnOrName", pos: int, len: int) -> Column:
    """Substring starts at `pos` and is of length `len` when str is String type or
    returns the slice of byte array that starts at `pos` in byte and is of length `len`
    when str is Binary type.
    .. versionadded:: 1.5.0
    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Notes:
    -----
    The position is not zero based, but 1 based index.
    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        target column to work on.
    pos : int
        starting position in str.
    len : int
        length of chars.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        substring of given value.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [("abcd",)],
    ...     [
    ...         "s",
    ...     ],
    ... )
    >>> df.select(substring(df.s, 1, 2).alias("s")).collect()
    [Row(s='ab')]
    """  # noqa: D205
    return _spark_substring(_to_column_expr(str), ConstantExpression(pos), ConstantExpression(len))


def _spark_substring(string: Expression, pos: Expression, length: Expression | None) -> Column:
    """Spark reads position 0 as 1, counts a negative position from the end, and gives '' for a negative length."""
    start = (
        CaseExpression(pos > 0, pos - 1)
        .when(pos < 0, FunctionExpression("length", string) + pos)
        .when(pos == 0, ConstantExpression(0))
    )
    # Not greatest(), which skips a NULL where Spark gives NULL.
    begin = CaseExpression(start < 0, ConstantExpression(0)).otherwise(start)
    if length is None:
        return Column(FunctionExpression("substring", string, begin + 1))
    end = start + length
    return Column(
        CaseExpression(string.isnotnull() & (begin >= end), ConstantExpression("")).otherwise(
            FunctionExpression("substring", string, begin + 1, end - begin)
        )
    )


def contains(left: "ColumnOrName", right: "ColumnOrName") -> Column:
    """Returns a boolean. The value is True if right is found inside left.
    Returns NULL if either input expression is NULL. Otherwise, returns False.
    Both left or right must be of STRING or BINARY type.
    .. versionadded:: 3.5.0
    Parameters
    ----------
    left : :class:`~pyspark.sql.Column` or str
        The input column or strings to check, may be NULL.
    right : :class:`~pyspark.sql.Column` or str
        The input column or strings to find, may be NULL.

    Examples:
    --------
    >>> df = spark.createDataFrame([("Spark SQL", "Spark")], ["a", "b"])
    >>> df.select(contains(df.a, df.b).alias("r")).collect()
    [Row(r=True)]
    >>> df = spark.createDataFrame(
    ...     [
    ...         (
    ...             "414243",
    ...             "4243",
    ...         )
    ...     ],
    ...     ["c", "d"],
    ... )
    >>> df = df.select(to_binary("c").alias("c"), to_binary("d").alias("d"))
    >>> df.printSchema()
    root
     |-- c: binary (nullable = true)
     |-- d: binary (nullable = true)
    >>> df.select(contains("c", "d"), contains("d", "c")).show()
    +--------------+--------------+
    |contains(c, d)|contains(d, c)|
    +--------------+--------------+
    |          true|         false|
    +--------------+--------------+
    """  # noqa: D205, D415
    return _invoke_function_over_columns("contains", left, right)


def reverse(col: "ColumnOrName") -> Column:
    """Collection function: returns a reversed string or an array with reverse order of elements.
    .. versionadded:: 1.5.0
    .. versionchanged:: 3.4.0
        Supports Spark Connect.
    Parameters
    ----------
    col : :class:`~pyspark.sql.Column` or str
        name of column or expression

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        array of elements in reverse order.

    Examples:
    --------
    >>> df = spark.createDataFrame([("Spark SQL",)], ["data"])
    >>> df.select(reverse(df.data).alias("s")).collect()
    [Row(s='LQS krapS')]
    >>> df = spark.createDataFrame([([2, 1, 3],), ([1],), ([],)], ["data"])
    >>> df.select(reverse(df.data).alias("r")).collect()
    [Row(r=[3, 1, 2]), Row(r=[1]), Row(r=[])]
    """  # noqa: D205, D415
    return _invoke_function("reverse", _to_column_expr(col))


def concat(*cols: "ColumnOrName") -> Column:
    """Concatenates multiple input columns together into a single column.
    The function works with strings, numeric, binary and compatible array columns.
    .. versionadded:: 1.5.0
    .. versionchanged:: 3.4.0
        Supports Spark Connect.
    Parameters
    ----------
    cols : :class:`~pyspark.sql.Column` or str
        target column or columns to work on.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        concatenated values. Type of the `Column` depends on input columns' type.

    See Also:
    --------
    :meth:`pyspark.sql.functions.array_join` : to concatenate string columns with delimiter

    Examples:
    --------
    >>> df = spark.createDataFrame([("abcd", "123")], ["s", "d"])
    >>> df = df.select(concat(df.s, df.d).alias("s"))
    >>> df.collect()
    [Row(s='abcd123')]
    >>> df
    DataFrame[s: string]
    >>> df = spark.createDataFrame(
    ...     [([1, 2], [3, 4], [5]), ([1, 2], None, [3])], ["a", "b", "c"]
    ... )
    >>> df = df.select(concat(df.a, df.b, df.c).alias("arr"))
    >>> df.collect()
    [Row(arr=[1, 2, 3, 4, 5]), Row(arr=None)]
    >>> df
    DataFrame[arr: array<bigint>]
    """  # noqa: D205, D415
    if not cols:
        return lit("")
    # DuckDB's || keeps binary and arrays and propagates NULL like Spark, where its concat makes text and skips NULL;
    # Spark reads a NULL literal as a string, where DuckDB would make the whole result untyped.
    exprs = [_to_column_expr(c).cast(VARCHAR) if _is_null_literal(c) else _to_column_expr(c) for c in cols]
    if len(exprs) == 1:
        (value,) = exprs
        return Column(FunctionExpression("cast_to_type", value, FunctionExpression("||", value, value)))
    return Column(functools.reduce(lambda left, right: FunctionExpression("||", left, right), exprs))


def instr(str: "ColumnOrName", substr: str) -> Column:
    """Locate the position of the first occurrence of substr column in the given string.
    Returns null if either of the arguments are null.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Notes:
    -----
    The position is not zero based, but 1 based index. Returns 0 if substr
    could not be found in str.

    Parameters
    ----------
    str : :class:`~pyspark.sql.Column` or str
        target column to work on.
    substr : str
        substring to look for.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        location of the first occurrence of the substring as integer.

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [("abcd",)],
    ...     [
    ...         "s",
    ...     ],
    ... )
    >>> df.select(instr(df.s, "b").alias("s")).collect()
    [Row(s=2)]
    """  # noqa: D205
    return _result_type(_invoke_function("instr", _to_column_expr(str), ConstantExpression(substr)), INTEGER)


def expr(str: str) -> Column:
    """Parses the expression string into the column that it represents.

    .. versionadded:: 1.5.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Parameters
    ----------
    str : str
        expression defined in string.

    Returns:
    -------
    :class:`~pyspark.sql.Column`
        column representing the expression.

    Examples:
    --------
    >>> df = spark.createDataFrame([["Alice"], ["Bob"]], ["name"])
    >>> df.select("name", expr("length(name)")).show()
    +-----+------------+
    | name|length(name)|
    +-----+------------+
    |Alice|           5|
    |  Bob|           3|
    +-----+------------+
    """
    return Column(SQLExpression(str))


def broadcast(df: "DataFrame") -> "DataFrame":
    """The broadcast function in Spark is used to optimize joins by broadcasting a smaller
    dataset to all the worker nodes. However, DuckDB operates on a single-node architecture .
    As a result, the function simply returns the input DataFrame without applying any modifications
    or optimizations, since broadcasting is not applicable in the DuckDB context.
    """  # noqa: D205
    return df


def _is_star(value: Any) -> bool:  # noqa: ANN401
    if isinstance(value, Column):
        return str(value.expr) == "*"
    if isinstance(value, str):
        return value == "*"
    return isinstance(value, _Ref) and value.parts[-1] == "*"


# Spark names a call `name(arg, ...)` after the function's own name, so only the functions whose names differ
# are listed: by the name Spark uses, or by a callable that builds the name from the arguments, including the
# defaults Spark adds and leaving out the arguments Spark leaves out.
_SPARK_NAMES: dict[str, str | Callable[..., Name]] = {
    "acos": "ACOS",
    "approxCountDistinct": lambda c, *_: _call_name("approx_count_distinct", c),
    "approx_count_distinct": lambda c, *_: _call_name("approx_count_distinct", c),
    "any_value": lambda c, *_: _call_name("any_value", c),
    "array_agg": "collect_list",
    "asc": lambda c: _sql_name(c, " ASC", " NULLS FIRST"),
    "asc_nulls_first": lambda c: _sql_name(c, " ASC", " NULLS FIRST"),
    "asc_nulls_last": lambda c: _sql_name(c, " ASC", " NULLS LAST"),
    "asin": "ASIN",
    "atan": "ATAN",
    "atan2": "ATAN2",
    "bround": lambda c, scale=0: _call_name("bround", c, scale),
    "call_function": lambda name, *cols: _call_name(name, *cols),
    "cbrt": "CBRT",
    "ceil": "CEIL",
    "cos": "COS",
    "cot": "COT",
    "count": lambda c: _call_name("count", 1 if _is_star(c) else c),
    "curdate": "current_date",
    "datepart": lambda field, source: _format_name("datepart({0} FROM {1})", field, source),
    "degrees": "DEGREES",
    "desc": lambda c: _sql_name(c, " DESC", " NULLS LAST"),
    "desc_nulls_first": lambda c: _sql_name(c, " DESC", " NULLS FIRST"),
    "desc_nulls_last": lambda c: _sql_name(c, " DESC", " NULLS LAST"),
    "e": "E",
    "exp": "EXP",
    "extract": lambda field, source: _format_name("extract({0} FROM {1})", field, source),
    "first": lambda c, *_: _call_name("first", c),
    "floor": "FLOOR",
    "isnotnull": lambda c: _format_name("({0} IS NOT NULL)", c),
    "isnull": lambda c: _format_name("({0} IS NULL)", c),
    "last": lambda c, *_: _call_name("last", c),
    "ilike": lambda s, pattern, *_: _call_name("ilike", s, pattern),
    "like": lambda s, pattern, escape=None: _format_name(
        "{0} LIKE {1}" if escape is None else "{0} LIKE {1} ESCAPE '{2}'", s, pattern, escape
    ),
    "log2": "LOG2",
    "mean": "avg",
    "pi": "PI",
    "positive": lambda c: _format_name("(+ {0})", c),
    "pow": "POWER",
    "radians": "RADIANS",
    "regexp": "REGEXP",
    "regexp_extract_all": lambda s, regexp, idx=1: _call_name("regexp_extract_all", s, regexp, idx),
    "regexp_like": "REGEXP_LIKE",
    "regexp_replace": lambda s, pattern, replacement: _call_name("regexp_replace", s, pattern, replacement, 1),
    "round": lambda c, scale=0: _call_name("round", c, scale),
    "signum": "SIGNUM",
    "sin": "SIN",
    "sort_array": lambda c, asc=True: _call_name("sort_array", c, asc),
    "split": lambda s, pattern, limit=-1: _call_name("split", s, pattern, limit),
    "sqrt": "SQRT",
    "substr": lambda s, pos, length=2147483647: _call_name("substr", s, pos, length),
    "tan": "TAN",
}

# These name their results themselves.
_NAMED_IN_BODY = {"col", "lit", "expr", "when"}


def _spark_named(function: Callable[..., Any]) -> Callable[..., Any]:
    """Give the Columns ``function`` returns the name Spark gives the same call."""
    spark_name = _SPARK_NAMES.get(function.__name__, function.__name__)
    signature = inspect.signature(function)

    @functools.wraps(function)
    def named(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        result = function(*args, **kwargs)
        if not isinstance(result, Column):
            return result
        values: list[Any] = []
        for param_name, value in signature.bind(*args, **kwargs).arguments.items():
            param = signature.parameters[param_name]
            if param.kind is param.VAR_POSITIONAL:
                given = value[0] if len(value) == 1 and isinstance(value[0], (list, tuple, set)) else value
            elif value is None and param.default is None:
                continue
            else:
                given = (value,)
            refs = "ColumnOrName" in str(param.annotation)
            values.extend(_Ref(_name_parts(v)) if refs and isinstance(v, str) else v for v in given)
        name = spark_name(*values) if callable(spark_name) else _call_name(spark_name, *values)
        return result._with_name(name)

    return named


for _function_name, _function in list(globals().items()):
    if (
        inspect.isfunction(_function)
        and _function.__module__ == __name__
        and not _function_name.startswith("_")
        and _function_name not in _NAMED_IN_BODY
    ):
        globals()[_function_name] = _spark_named(_function)


def _carry_markers(function: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        result = function(*args, **kwargs)
        if isinstance(result, Column) and not result._is_aggregate:
            arguments: list[Any] = []
            names_a_column = False
            for param_name, value in signature.bind(*args, **kwargs).arguments.items():
                param = signature.parameters[param_name]
                given = list(value) if param.kind is param.VAR_POSITIONAL else [value]
                given += [x for a in given if isinstance(a, (list, tuple, set)) for x in a]
                arguments += given
                # A str names a column only where the function takes a column.
                if "ColumnOrName" in str(param.annotation) and any(isinstance(a, str) for a in given):
                    names_a_column = True
            if any(isinstance(a, Column) and a._is_aggregate for a in arguments):
                result._is_aggregate = True
                result._aggregate_inputs = tuple(
                    x for a in arguments if isinstance(a, Column) for x in a._aggregate_inputs
                )
            elif not names_a_column and all(a._is_constant for a in arguments if isinstance(a, Column)):
                result._is_constant = True
        return result

    return wrapper


# These read a column name or SQL text from a str, so their result is never a constant.
_READ_COLUMNS_FROM_TEXT = {"col", "expr"}

# A Column built from an aggregate is itself an aggregate, and one built only from constants is a constant;
# DataFrame.select relies on both.
for _name, _function in list(globals().items()):
    if (
        not _name.startswith("_")
        and _name not in _READ_COLUMNS_FROM_TEXT
        and inspect.isfunction(_function)
        and _function.__module__ == __name__
    ):
        globals()[_name] = _carry_markers(_function)

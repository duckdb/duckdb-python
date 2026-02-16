from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

from ..errors import PySparkTypeError
from ..exception import ContributionsAcceptedError
from ._typing import ColumnOrName
from .column import Column


class WindowSpec:
    """A window specification that defines the partitioning, ordering, and frame boundaries.

    Use the static methods in :class:`Window` to create a :class:`WindowSpec`.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.
    """

    def __init__(self) -> None:
        self._partition_by: List[ColumnOrName] = []
        self._order_by: List[ColumnOrName] = []
        self._rows_between: Optional[Tuple[int, int]] = None
        self._range_between: Optional[Tuple[int, int]] = None

    def _copy(self) -> "WindowSpec":
        new_window = WindowSpec()
        new_window._partition_by = self._partition_by.copy()
        new_window._order_by = self._order_by.copy()
        new_window._rows_between = self._rows_between
        new_window._range_between = self._range_between
        return new_window

    def partitionBy(self, *cols: Union[ColumnOrName, Sequence[ColumnOrName]]) -> "WindowSpec":
        """Defines the partitioning columns in a :class:`WindowSpec`.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        cols : str, :class:`Column` or list
            names of columns or expressions
        """
        all_cols: Union[List[ColumnOrName], List[List[ColumnOrName]]] = list(cols)  # type: ignore[assignment]

        if isinstance(all_cols[0], list):
            all_cols = all_cols[0]

        new_window = self._copy()
        new_window._partition_by = all_cols
        return new_window

    def orderBy(self, *cols: Union[ColumnOrName, Sequence[ColumnOrName]]) -> "WindowSpec":
        """Defines the ordering columns in a :class:`WindowSpec`.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        cols : str, :class:`Column` or list
            names of columns or expressions
        """
        all_cols: Union[List[ColumnOrName], List[List[ColumnOrName]]] = list(cols)  # type: ignore[assignment]

        if isinstance(all_cols[0], list):
            all_cols = all_cols[0]

        new_window = self._copy()
        new_window._order_by = all_cols
        return new_window

    def rowsBetween(self, start: int, end: int) -> "WindowSpec":
        """Defines the frame boundaries, from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative positions from the current row.
        For example, "0" means "current row", while "-1" means the row before
        the current row, and "5" means the fifth row after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        start : int
            boundary start, inclusive.
            The frame is unbounded if this is ``Window.unboundedPreceding``, or
            any value less than or equal to max(-sys.maxsize, -9223372036854775808).
        end : int
            boundary end, inclusive.
            The frame is unbounded if this is ``Window.unboundedFollowing``, or
            any value greater than or equal to min(sys.maxsize, 9223372036854775807).
        """
        new_window = self._copy()
        new_window._rows_between = (start, end)
        return new_window

    def rangeBetween(self, start: int, end: int) -> "WindowSpec":
        """Defines the frame boundaries, from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative from the current row. For example,
        "0" means "current row", while "-1" means one off before the current row,
        and "5" means the five off after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        start : int
            boundary start, inclusive.
            The frame is unbounded if this is ``Window.unboundedPreceding``, or
            any value less than or equal to max(-sys.maxsize, -9223372036854775808).
        end : int
            boundary end, inclusive.
            The frame is unbounded if this is ``Window.unboundedFollowing``, or
            any value greater than or equal to min(sys.maxsize, 9223372036854775807).
        """
        new_window = self._copy()
        new_window._range_between = (start, end)
        return new_window

    def _columns_as_str(self, *, cols: List[ColumnOrName], include_order_direction: bool) -> list[str]:
        expressions = []
        for col in cols:
            if isinstance(col, str):
                expressions.append(col)
            elif isinstance(col, Column):
                if include_order_direction:
                    # TODO: Handle ascending/descending order if needed
                    raise ContributionsAcceptedError("Column Expression is not supported in WindowSpec.orderBy yet")

                else:
                    expressions.append(str(col.expr))
            else:
                raise PySparkTypeError(f"Invalid column type: {type(col)}")
        return expressions

    @staticmethod
    def _generate_window_interval_expr(start: int, end: int) -> str:
        if start == Window.currentRow and end == Window.currentRow:
            return "CURRENT ROW AND CURRENT ROW"

        if start == Window.currentRow:
            return f"CURRENT ROW AND {end} FOLLOWING"

        if end == Window.currentRow:
            return f"{start} PRECEDING AND CURRENT ROW"
        if start == Window.unboundedPreceding and end == Window.unboundedFollowing:
            return "UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"

        if start == Window.unboundedPreceding:
            return f"UNBOUNDED PRECEDING AND {end} FOLLOWING"
        if end == Window.unboundedFollowing:
            return f"{start} PRECEDING AND UNBOUNDED FOLLOWING"

        return f"{start} PRECEDING AND {end} FOLLOWING"

    def _window_expr(self) -> str:
        parts = []
        if self._partition_by:
            parts.append(
                "PARTITION BY "
                + ", ".join(self._columns_as_str(cols=self._partition_by, include_order_direction=False))
            )
        if self._order_by:
            parts.append(
                "ORDER BY " + ", ".join(self._columns_as_str(cols=self._order_by, include_order_direction=True))
            )
        if self._rows_between is not None:
            parts.append(f"ROWS BETWEEN {self._generate_window_interval_expr(*self._rows_between)}")
        if self._range_between is not None:
            parts.append(f"RANGE BETWEEN {self._generate_window_interval_expr(*self._range_between)}")
        sql = " ".join(parts)
        return sql


class Window:
    """Utility functions for defining window in DataFrames.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.

    Notes:
    -----
    When ordering is not defined, an unbounded window frame (rowFrame,
    unboundedPreceding, unboundedFollowing) is used by default. When ordering is defined,
    a growing window frame (rangeFrame, unboundedPreceding, currentRow) is used by default.

    Examples:
    --------
    >>> # ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    >>> window = Window.orderBy("date").rowsBetween(
    ...     Window.unboundedPreceding, Window.currentRow
    ... )

    >>> # PARTITION BY country ORDER BY date RANGE BETWEEN 3 PRECEDING AND 3 FOLLOWING
    >>> window = Window.orderBy("date").partitionBy("country").rangeBetween(-3, 3)
    """

    currentRow = 0
    unboundedPreceding: int = -(1 << 63)  # -9223372036854775808 - equivalent to Java's Long.MIN_VALUE
    unboundedFollowing: int = (1 << 63) - 1  # 9223372036854775807 - equivalent to Java's Long.MAX_VALUE

    @classmethod
    def partitionBy(cls, *cols: Union[ColumnOrName, Sequence[ColumnOrName]]) -> WindowSpec:
        """Creates a :class:`WindowSpec` with the partitioning defined.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        cols : str, :class:`Column` or list
            names of columns or expressions

        Returns:
        -------
        :class: `WindowSpec`
            A :class:`WindowSpec` with the partitioning defined.

        Examples:
        --------
        >>> from pyspark.sql import Window
        >>> from pyspark.sql.functions import row_number
        >>> df = spark.createDataFrame(
        ...     [(1, "a"), (1, "a"), (2, "a"), (1, "b"), (2, "b"), (3, "b")], ["id", "category"]
        ... )
        >>> df.show()
        +---+--------+
        | id|category|
        +---+--------+
        |  1|       a|
        |  1|       a|
        |  2|       a|
        |  1|       b|
        |  2|       b|
        |  3|       b|
        +---+--------+

        Show row number order by ``id`` in partition ``category``.

        >>> window = Window.partitionBy("category").orderBy("id")
        >>> df.withColumn("row_number", row_number().over(window)).show()
        +---+--------+----------+
        | id|category|row_number|
        +---+--------+----------+
        |  1|       a|         1|
        |  1|       a|         2|
        |  2|       a|         3|
        |  1|       b|         1|
        |  2|       b|         2|
        |  3|       b|         3|
        +---+--------+----------+
        """
        return WindowSpec().partitionBy(*cols)

    @classmethod
    def orderBy(cls, *cols: Union[ColumnOrName, Sequence[ColumnOrName]]) -> WindowSpec:
        """Creates a :class:`WindowSpec` with the ordering defined.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        cols : str, :class:`Column` or list
            names of columns or expressions

        Returns:
        -------
        :class: `WindowSpec`
            A :class:`WindowSpec` with the ordering defined.

        Examples:
        --------
        >>> from pyspark.sql import Window
        >>> from pyspark.sql.functions import row_number
        >>> df = spark.createDataFrame(
        ...     [(1, "a"), (1, "a"), (2, "a"), (1, "b"), (2, "b"), (3, "b")], ["id", "category"]
        ... )
        >>> df.show()
        +---+--------+
        | id|category|
        +---+--------+
        |  1|       a|
        |  1|       a|
        |  2|       a|
        |  1|       b|
        |  2|       b|
        |  3|       b|
        +---+--------+

        Show row number order by ``category`` in partition ``id``.

        >>> window = Window.partitionBy("id").orderBy("category")
        >>> df.withColumn("row_number", row_number().over(window)).show()
        +---+--------+----------+
        | id|category|row_number|
        +---+--------+----------+
        |  1|       a|         1|
        |  1|       a|         2|
        |  1|       b|         3|
        |  2|       a|         1|
        |  2|       b|         2|
        |  3|       b|         1|
        +---+--------+----------+
        """
        return WindowSpec().orderBy(*cols)

    @classmethod
    def rowsBetween(cls, start: int, end: int) -> WindowSpec:
        """Creates a :class:`WindowSpec` with the frame boundaries defined, from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative positions from the current row.
        For example, "0" means "current row", while "-1" means the row before
        the current row, and "5" means the fifth row after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        A row based boundary is based on the position of the row within the partition.
        An offset indicates the number of rows above or below the current row, the frame for the
        current row starts or ends. For instance, given a row based sliding frame with a lower bound
        offset of -1 and a upper bound offset of +2. The frame for row with index 5 would range from
        index 4 to index 7.

        .. versionadded:: 2.1.0

        Parameters
        ----------
        start : int
            boundary start, inclusive.
            The frame is unbounded if this is ``Window.unboundedPreceding``, or
            any value less than or equal to -9223372036854775808.
        end : int
            boundary end, inclusive.
            The frame is unbounded if this is ``Window.unboundedFollowing``, or
            any value greater than or equal to 9223372036854775807.

        Returns:
        -------
        :class: `WindowSpec`
            A :class:`WindowSpec` with the frame boundaries defined,
            from `start` (inclusive) to `end` (inclusive).

        Examples:
        --------
        >>> from pyspark.sql import Window
        >>> from pyspark.sql import functions as func
        >>> df = spark.createDataFrame(
        ...     [(1, "a"), (1, "a"), (2, "a"), (1, "b"), (2, "b"), (3, "b")], ["id", "category"]
        ... )
        >>> df.show()
        +---+--------+
        | id|category|
        +---+--------+
        |  1|       a|
        |  1|       a|
        |  2|       a|
        |  1|       b|
        |  2|       b|
        |  3|       b|
        +---+--------+

        Calculate sum of ``id`` in the range from currentRow to currentRow + 1
        in partition ``category``

        >>> window = Window.partitionBy("category").orderBy("id").rowsBetween(Window.currentRow, 1)
        >>> df.withColumn("sum", func.sum("id").over(window)).sort("id", "category", "sum").show()
        +---+--------+---+
        | id|category|sum|
        +---+--------+---+
        |  1|       a|  2|
        |  1|       a|  3|
        |  1|       b|  3|
        |  2|       a|  2|
        |  2|       b|  5|
        |  3|       b|  3|
        +---+--------+---+

        """
        return WindowSpec().rowsBetween(start, end)

    @classmethod
    def rangeBetween(cls, start: int, end: int) -> WindowSpec:
        """Creates a :class:`WindowSpec` with the frame boundaries defined, from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative from the current row. For example,
        "0" means "current row", while "-1" means one off before the current row,
        and "5" means the five off after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        A range-based boundary is based on the actual value of the ORDER BY
        expression(s). An offset is used to alter the value of the ORDER BY expression, for
        instance if the current ORDER BY expression has a value of 10 and the lower bound offset
        is -3, the resulting lower bound for the current row will be 10 - 3 = 7. This however puts a
        number of constraints on the ORDER BY expressions: there can be only one expression and this
        expression must have a numerical data type. An exception can be made when the offset is
        unbounded, because no value modification is needed, in this case multiple and non-numeric
        ORDER BY expression are allowed.

        .. versionadded:: 2.1.0

        Parameters
        ----------
        start : int
            boundary start, inclusive.
            The frame is unbounded if this is ``Window.unboundedPreceding``, or
            any value less than or equal to max(-sys.maxsize, -9223372036854775808).
        end : int
            boundary end, inclusive.
            The frame is unbounded if this is ``Window.unboundedFollowing``, or
            any value greater than or equal to min(sys.maxsize, 9223372036854775807).

        Returns:
        -------
        :class: `WindowSpec`
            A :class:`WindowSpec` with the frame boundaries defined,
            from `start` (inclusive) to `end` (inclusive).

        Examples:
        --------
        >>> from pyspark.sql import Window
        >>> from pyspark.sql import functions as func
        >>> df = spark.createDataFrame(
        ...     [(1, "a"), (1, "a"), (2, "a"), (1, "b"), (2, "b"), (3, "b")], ["id", "category"]
        ... )
        >>> df.show()
        +---+--------+
        | id|category|
        +---+--------+
        |  1|       a|
        |  1|       a|
        |  2|       a|
        |  1|       b|
        |  2|       b|
        |  3|       b|
        +---+--------+

        Calculate sum of ``id`` in the range from ``id`` of currentRow to ``id`` of currentRow + 1
        in partition ``category``

        >>> window = Window.partitionBy("category").orderBy("id").rangeBetween(Window.currentRow, 1)
        >>> df.withColumn("sum", func.sum("id").over(window)).sort("id", "category").show()
        +---+--------+---+
        | id|category|sum|
        +---+--------+---+
        |  1|       a|  4|
        |  1|       a|  4|
        |  1|       b|  3|
        |  2|       a|  2|
        |  2|       b|  5|
        |  3|       b|  3|
        +---+--------+---+

        """
        return WindowSpec().rangeBetween(start, end)

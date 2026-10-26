"""The polars sources: a DataFrame read as the pyarrow Table of one `to_arrow()`, a LazyFrame collected per scan.

Polars exports a column of Python objects (`pl.Object`) as the objects' addresses, so a frame holding one is refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from duckdb._sources.arrow import ArrowScanSource, ArrowStreamSource
from duckdb._sources.nodes import Untranslatable
from duckdb._sources.pyarrow import PyArrowTableSource
from duckdb._sources.to_arrow import nan_may_satisfy
from duckdb._sources.to_polars import to_polars

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from duckdb._sources.nodes import Node


def polars_source(obj: object) -> ArrowScanSource | None:
    """The source for a polars object, or None when polars has none for it."""
    import polars

    if isinstance(obj, polars.DataFrame):
        _refuse_object_columns(obj.schema)
        return PyArrowTableSource(obj.to_arrow())
    if isinstance(obj, polars.LazyFrame):
        return LazyFrameSource(obj)
    if isinstance(obj, polars.Series):
        _refuse_object_columns({obj.name: obj.dtype})
        return PolarsStreamSource(obj)
    return None


class PolarsStreamSource(ArrowStreamSource):
    """A polars Series."""

    schema_is_stable = True

    def rows(self) -> int | None:
        """The length of the series."""
        return len(self.obj)


class LazyFrameSource(ArrowScanSource):
    """A polars LazyFrame, filtered, narrowed and collected once per scan, declaring what an empty collect exports."""

    schema_is_stable = True

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        self._empty: Any = None
        self._declared: Any = None

    def _empty_frame(self) -> Any:  # noqa: ANN401
        if self._empty is None:
            empty = self.obj.head(0).collect()
            _refuse_object_columns(empty.schema)
            self._empty = empty
        return self._empty

    def _declared_schema(self) -> Any:  # noqa: ANN401
        if self._declared is None:
            self._declared = self._empty_frame().to_arrow().schema
        return self._declared

    def __arrow_c_schema__(self) -> object:
        """The schema of the frame's first zero rows as Arrow."""
        return self._declared_schema().__arrow_c_schema__()

    def accepts(self, predicate: Node) -> bool:
        """Whether `to_polars` translates `predicate` exactly and statistics pruning cannot drop a row it keeps."""
        try:
            to_polars(predicate, self._empty_frame().schema)
            # A scan over files prunes row groups by minimum and maximum statistics, which leave NaN out.
            return not nan_may_satisfy(predicate, self._declared_schema())
        except Untranslatable:
            return False

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Node]) -> tuple[object, bool]:
        """The collected frame's stream after `filters`, narrowed to `columns` unless none are asked."""
        schema = self._empty_frame().schema
        plan = self.obj
        for predicate in filters:
            plan = plan.filter(to_polars(predicate, schema))
        # A polars frame without columns has no rows, so a request for no columns reads them all.
        if not columns:
            return plan.collect().to_arrow().__arrow_c_stream__(), False
        import polars

        # Polars reads a name like ^x$ or * as a selector, so columns are taken by position.
        plan = plan.select([polars.nth(i) for i in columns])
        return plan.collect().to_arrow().__arrow_c_stream__(), True


def _refuse_object_columns(schema: Mapping[str, object]) -> None:
    import polars

    for name, dtype in schema.items():
        if dtype == polars.Object:
            message = f"the polars column '{name}' holds Python objects, which polars cannot export as Arrow"
            raise TypeError(message)

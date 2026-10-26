"""The pyarrow sources: a Table or RecordBatch, a RecordBatchReader, a Dataset and a Scanner.

An object that only looks like a Dataset or Scanner reaches `adapt` only through the private registration hook,
since the scan's entry points accept the pyarrow classes alone; it is never offered a filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from duckdb._sources.arrow import ArrowArraySource, ArrowScanSource, ArrowStreamSource, arrow_schema
from duckdb._sources.nodes import Untranslatable
from duckdb._sources.to_arrow import nan_may_satisfy, to_arrow

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

    from duckdb._sources.nodes import Node


def pyarrow_source(obj: object, modules: set[str]) -> ArrowScanSource | None:
    """The source for a pyarrow object, given the modules of its classes, or None when pyarrow has none for it."""
    import pyarrow

    if isinstance(obj, pyarrow.RecordBatchReader):
        return PyArrowReaderSource(obj)
    if isinstance(obj, pyarrow.Table | pyarrow.RecordBatch):
        return PyArrowTableSource(obj)
    if "pyarrow._dataset" in modules:
        import pyarrow.dataset

        if isinstance(obj, pyarrow.dataset.Scanner):
            return PyArrowScannerSource(obj)
        if isinstance(obj, pyarrow.dataset.Dataset):
            return PyArrowDatasetSource(obj)
    if isinstance(obj, pyarrow.ChunkedArray):
        return PyArrowStreamSource(obj)
    if isinstance(obj, pyarrow.Array):
        return PyArrowArraySource(obj)
    return None


class PyArrowTableSource(ArrowStreamSource):
    """A pyarrow Table or RecordBatch, narrowed by position with `select`, which copies nothing."""

    schema_is_stable = True

    def rows(self) -> int | None:
        """The number of rows."""
        return len(self.obj)

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Node]) -> tuple[object, bool]:
        """The object's stream, narrowed to `columns` when asked."""
        if columns is None:
            return self.obj.__arrow_c_stream__(), False
        return self.obj.select(list(columns)).__arrow_c_stream__(), True


class PyArrowReaderSource(ArrowStreamSource):
    """A pyarrow RecordBatchReader, every column."""

    one_shot = True
    schema_is_stable = True

    def rows(self) -> int | None:
        """Unknown, since only reading the stream counts it."""
        return None


class PyArrowStreamSource(ArrowStreamSource):
    """A pyarrow ChunkedArray, its schema from its `type`."""

    schema_attribute = "type"

    schema_is_stable = True

    def rows(self) -> int | None:
        """The length of the array."""
        return len(self.obj)


class PyArrowArraySource(ArrowArraySource):
    """A pyarrow Array or StructArray, one batch."""

    schema_is_stable = True

    def rows(self) -> int | None:
        """The length of the array."""
        return len(self.obj)


class DatasetSource(ArrowScanSource):
    """An object with a pyarrow Dataset's `schema` and `scanner(columns=, filter=)`, scanned afresh per stream."""

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        self._declared: Any = None

    def declared_schema(self) -> pa.Schema:
        """The schema the dataset declares, kept for the life of a stable source and read anew for any other."""
        if not self.schema_is_stable:
            return arrow_schema(self.obj.schema)
        if self._declared is None:
            self._declared = arrow_schema(self.obj.schema)
        return self._declared

    def __arrow_c_schema__(self) -> object:
        """The schema the dataset declares."""
        return self.declared_schema().__arrow_c_schema__()

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Node]) -> tuple[object, bool]:
        """A reader over a scanner that applies `filters` and narrows to `columns` by name."""
        schema = self.declared_schema()
        mask = None
        for predicate in filters:
            term = to_arrow(predicate, schema)
            mask = term if mask is None else mask & term
        names = schema.names
        chosen = None if columns is None else [names[i] for i in columns]
        scanner = type(self.obj).scanner(self.obj, columns=chosen, filter=mask)
        return scanner.to_reader().__arrow_c_stream__(), chosen is not None


class PyArrowDatasetSource(DatasetSource):
    """A pyarrow Dataset, which applies every predicate `to_arrow` translates and prunes by it."""

    schema_is_stable = True

    def accepts(self, predicate: Node) -> bool:
        """Whether `to_arrow` translates `predicate` exactly and statistics pruning cannot drop a row it keeps."""
        schema = self.declared_schema()
        try:
            to_arrow(predicate, schema)
        except Untranslatable:
            return False
        # Row groups are pruned by minimum and maximum statistics, which leave NaN out.
        return not nan_may_satisfy(predicate, schema)


class ScannerSource(ArrowScanSource):
    """An object with a pyarrow Scanner's `projected_schema` and `to_reader()`; it keeps its projection and filter."""

    def __arrow_c_schema__(self) -> object:
        """The schema the scanner projects."""
        return arrow_schema(self.obj.projected_schema).__arrow_c_schema__()

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Node]) -> tuple[object, bool]:
        """A fresh reader over the scanner, every column."""
        return self.obj.to_reader().__arrow_c_stream__(), False


class PyArrowScannerSource(ScannerSource):
    """A pyarrow Scanner."""

    schema_is_stable = True

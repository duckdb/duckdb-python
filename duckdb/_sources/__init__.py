"""The Python objects the Arrow scan reads, each behind a source that exports an Arrow C stream for it.

The scan calls `adapt` for an Arrow shaped object and then speaks only to the source: `rows()`, `pull_under_gil`,
`accepts(predicate)`, `stream(columns, filters)` and, when present, `__arrow_c_schema__()`; `one_shot` is
informational, since the scan decides single use from the underlying object.
Every module of the package is imported here, since importing one while a query runs could load the extension a
second time; pyarrow, polars and pandas are imported only when an object's class, or a class it derives from, is
defined in them, and their classes are then recognised by identity.
"""

from __future__ import annotations

import datetime

from duckdb._sources import nodes, to_arrow, to_polars
from duckdb._sources.arrow import (
    ArrowArraySource,
    ArrowCapsuleSource,
    ArrowScanSource,
    ArrowStreamSource,
    Source,
    class_modules,
)
from duckdb._sources.polars import polars_source
from duckdb._sources.pyarrow import DatasetSource, ScannerSource, pyarrow_source

__all__ = ["ArrowScanSource", "Source", "adapt", "nodes", "to_arrow", "to_polars"]

_CAPSULE = type(datetime.datetime_CAPI)


def adapt(obj: object) -> Source:
    """The source for an Arrow shaped object; an existing `Source` is returned unchanged."""
    if isinstance(obj, Source):
        return obj
    if type(obj) is _CAPSULE:
        return ArrowCapsuleSource(obj)
    modules = class_modules(obj)
    libraries = {module.partition(".")[0] for module in modules}
    source = None
    if "pyarrow" in libraries:
        source = pyarrow_source(obj, modules)
    if source is None and "polars" in libraries:
        source = polars_source(obj)
    if source is not None:
        return source
    if hasattr(obj, "__arrow_c_stream__"):
        return ArrowStreamSource(obj)
    if hasattr(obj, "__arrow_c_array__"):
        return ArrowArraySource(obj)
    if hasattr(obj, "to_reader") and hasattr(obj, "projected_schema"):
        return ScannerSource(obj)
    if hasattr(obj, "scanner") and hasattr(obj, "schema"):
        return DatasetSource(obj)
    message = (
        "an Arrow object must export __arrow_c_stream__ or __arrow_c_array__, or be an Arrow stream capsule, a "
        f"pyarrow Dataset or Scanner, or a polars DataFrame or LazyFrame; {type(obj).__name__} is none of these"
    )
    raise TypeError(message)

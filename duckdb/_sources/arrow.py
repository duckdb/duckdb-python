"""The source protocol, and the sources for any object exporting the Arrow PyCapsule interface or a bare capsule."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

    from duckdb._sources.nodes import Node


class Source:
    """An object wrapped for a scan."""

    #: Informational only: the scan decides from the underlying object whether it can be read once.
    one_shot = False
    #: Whether the schema the object declares cannot change while it is registered, so the scan may cache it.
    schema_is_stable = False

    def __init__(self, obj: object) -> None:
        self.obj: Any = obj

    def rows(self) -> int | None:
        """How many rows a scan produces, when the object knows without reading itself, and None otherwise."""
        return None


class ArrowScanSource(Source):
    """A source the Arrow scan reads through the stream `stream()` exports, optionally with `__arrow_c_schema__`."""

    #: An opt in asking the scan to hold the GIL for pulls and release callbacks; no adapter here needs it, since a
    #: producer that reaches Python takes the GIL itself, as pyarrow, ctypes, cffi and Cython callbacks do.
    pull_under_gil = False

    def accepts(self, predicate: Node) -> bool:
        """Whether every stream will hold only rows where `predicate` is true by the engine's semantics."""
        return False

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Node]) -> tuple[object, bool]:
        """A stream capsule or a (schema, array) capsule pair, and whether it holds exactly `columns` in order.

        `columns` are declared column indexes, None for all; a source that cannot narrow returns every declared
        column in declared order and False. `filters` are the accepted predicates, applied before narrowing.
        """
        return self.obj.__arrow_c_stream__(), False


class _ArrowSchema(ctypes.Structure):
    """The C struct an object's `_export_to_c` writes a schema into."""

    _fields_ = (
        ("format", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("metadata", ctypes.c_char_p),
        ("flags", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    )


def class_modules(obj: object) -> set[str]:
    """The modules defining the class of `obj` and the classes it derives from; a class may have no module."""
    return {module for cls in type(obj).__mro__ if isinstance(module := getattr(cls, "__module__", None), str)}


def _exported(exporter: object) -> pa.Schema | pa.Field:
    """What `exporter._export_to_c` writes: a Schema for a struct, otherwise a Field keeping name and metadata."""
    import pyarrow

    exported = _ArrowSchema()
    address = ctypes.addressof(exported)
    exporter._export_to_c(address)  # type: ignore[attr-defined]
    if exported.format == b"+s":
        return pyarrow.Schema._import_from_c(address)
    return pyarrow.Field._import_from_c(address)


def arrow_schema(schema: object) -> pa.Schema:
    """A pyarrow Schema for a schema object exporting through `__arrow_c_schema__` or the older `_export_to_c`."""
    import pyarrow

    if isinstance(schema, pyarrow.Schema):
        return schema
    hook = getattr(schema, "__arrow_c_schema__", None)
    if callable(hook):
        return pyarrow.Schema._import_from_c_capsule(hook())
    exported = _exported(schema)
    if not isinstance(exported, pyarrow.Schema):
        message = f"a table schema must be a struct, not {exported.type}"
        raise TypeError(message)
    return exported


def _schema_capsule(exporter: object) -> object:
    """The schema capsule of a schema or type object exporting through `__arrow_c_schema__` or `_export_to_c`."""
    hook = getattr(exporter, "__arrow_c_schema__", None)
    if callable(hook):
        return hook()
    return _exported(exporter).__arrow_c_schema__()


def _schema_hook(source: ArrowScanSource, attribute: str) -> None:
    """Give `source` a `__arrow_c_schema__`, so bind reads the schema without opening a stream.

    The hook reads the object's own `__arrow_c_schema__`, or else `attribute`, at every call, so an object whose
    schema changes reports the new one.
    """
    obj = source.obj
    if callable(getattr(obj, "__arrow_c_schema__", None)):
        source.__dict__["__arrow_c_schema__"] = lambda: obj.__arrow_c_schema__()
        return
    exporter = getattr(obj, attribute, None)
    if callable(getattr(exporter, "__arrow_c_schema__", None)) or callable(getattr(exporter, "_export_to_c", None)):
        source.__dict__["__arrow_c_schema__"] = lambda: _schema_capsule(getattr(obj, attribute))


class ArrowStreamSource(ArrowScanSource):
    """Anything with `__arrow_c_stream__`: every column, its schema from `__arrow_c_schema__` or `schema`."""

    #: The attribute that holds the object's schema when the object has no `__arrow_c_schema__` of its own.
    schema_attribute = "schema"

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        _schema_hook(self, self.schema_attribute)

    def rows(self) -> int | None:
        """The length of a pandas Series or DataFrame, which is its row count; unknown for anything else."""
        if "pandas" not in {module.partition(".")[0] for module in class_modules(self.obj)}:
            return None
        import pandas

        return len(self.obj) if isinstance(self.obj, pandas.Series | pandas.DataFrame) else None


class ArrowArraySource(ArrowScanSource):
    """Anything with `__arrow_c_array__` and no stream: one batch, read as often as asked."""

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        _schema_hook(self, "type")

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Node]) -> tuple[object, bool]:
        """The schema and array capsules of the array, every column."""
        return self.obj.__arrow_c_array__(), False


class ArrowCapsuleSource(ArrowScanSource):
    """A bare capsule; the scan checks that it is an unreleased `arrow_array_stream`."""

    one_shot = True
    schema_is_stable = True

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Node]) -> tuple[object, bool]:
        """The capsule itself, every column."""
        return self.obj, False

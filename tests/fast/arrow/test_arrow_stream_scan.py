import contextlib
import subprocess
import sys

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")


class ArrowStream:
    """Minimal PyCapsuleInterface wrapper around a PyArrow table.

    This represents any third-party library (not Polars, not PyArrow) that
    implements the Arrow PyCapsule interface. DuckDB's replacement scan
    handles Polars and PyArrow types explicitly before falling through to
    PyCapsuleInterface detection via GetArrowType(), so we need a wrapper
    like this to exercise that code path.
    """

    def __init__(self, tbl) -> None:
        self.tbl = tbl
        self.stream_count = 0

    def __arrow_c_stream__(self, requested_schema=None):  # noqa: ANN204
        self.stream_count += 1
        return self.tbl.__arrow_c_stream__(requested_schema=requested_schema)


class ArrowStreamWithSchema(ArrowStream):
    """PyCapsuleInterface wrapper that also exposes __arrow_c_schema__."""

    def __arrow_c_schema__(self):  # noqa: ANN204
        return self.tbl.schema.__arrow_c_schema__()


class ArrowStreamWithDotSchema(ArrowStream):
    """PyCapsuleInterface wrapper that exposes .schema (pyarrow schema with _export_to_c)."""

    def __init__(self, tbl) -> None:
        super().__init__(tbl)
        self.schema = tbl.schema


class SingleUseArrowStream:
    """PyCapsuleInterface that can only produce one stream, but exposes .schema."""

    def __init__(self, tbl) -> None:
        self.tbl = tbl
        self.schema = tbl.schema
        self.stream_count = 0

    def __arrow_c_stream__(self, requested_schema=None):  # noqa: ANN204
        self.stream_count += 1
        if self.stream_count > 1:
            msg = "Stream already consumed"
            raise RuntimeError(msg)
        return self.tbl.__arrow_c_stream__(requested_schema=requested_schema)


class TestPyCapsuleInterfaceMultiScan:
    """Issue #70: queries requiring multiple scans of an arrow stream.

    PyCapsuleInterface objects support multi-scan because each call to
    __arrow_c_stream__() produces a fresh stream.
    """

    def test_union_all(self, duckdb_cursor):
        """UNION ALL scans the same PyCapsuleInterface twice in one query."""
        obj = ArrowStream(pa.table({"id": [1, 2, 3, 4, 5]}))  # noqa: F841
        result = duckdb_cursor.sql("SELECT id FROM obj UNION ALL SELECT id + 1 FROM obj").fetchall()
        ids = sorted(r[0] for r in result)
        assert ids == sorted([1, 2, 3, 4, 5, 2, 3, 4, 5, 6])

    def test_rescan_across_queries(self, duckdb_cursor):
        """PyCapsuleInterface scanned in two consecutive queries."""
        obj = ArrowStream(pa.table({"id": [1, 2, 3]}))  # noqa: F841
        r1 = duckdb_cursor.sql("SELECT * FROM obj").fetchall()
        r2 = duckdb_cursor.sql("SELECT * FROM obj").fetchall()
        assert r1 == r2 == [(1,), (2,), (3,)]

    def test_register(self, duckdb_cursor):
        """PyCapsuleInterface registered via register() supports multi-scan."""
        obj = ArrowStream(pa.table({"id": [1, 2, 3]}))
        duckdb_cursor.register("my_stream", obj)
        result = duckdb_cursor.sql("SELECT id FROM my_stream UNION ALL SELECT id FROM my_stream").fetchall()
        assert len(result) == 6

    def test_from_arrow(self, duckdb_cursor):
        """PyCapsuleInterface passed to from_arrow() supports multi-scan."""
        obj = ArrowStream(pa.table({"id": [1, 2, 3]}))
        rel = duckdb_cursor.from_arrow(obj)
        r1 = rel.fetchall()
        r2 = rel.fetchall()
        assert r1 == r2 == [(1,), (2,), (3,)]

    def test_self_join(self, duckdb_cursor):
        """Self-join on PyCapsuleInterface requires two scans."""
        obj = ArrowStream(pa.table({"id": [1, 2, 3], "val": [10, 20, 30]}))  # noqa: F841
        result = duckdb_cursor.sql("SELECT a.id, b.val FROM obj a JOIN obj b ON a.id = b.id").fetchall()
        assert sorted(result) == [(1, 10), (2, 20), (3, 30)]


class TestPyCapsuleInterfacePushdown:
    """PyCapsuleInterface objects get projection and filter pushdown via arrow_scan."""

    def test_projection_pushdown(self, duckdb_cursor):
        """Selecting a subset of columns only reads those columns."""
        obj = ArrowStream(pa.table({"a": [1, 2, 3], "b": [10, 20, 30], "c": ["x", "y", "z"]}))  # noqa: F841
        result = duckdb_cursor.sql("SELECT a FROM obj").fetchall()
        assert result == [(1,), (2,), (3,)]

    def test_filter_pushdown(self, duckdb_cursor):
        """Filters are pushed down to the arrow scanner."""
        obj = ArrowStream(pa.table({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}))  # noqa: F841
        result = duckdb_cursor.sql("SELECT a, b FROM obj WHERE a > 3").fetchall()
        assert sorted(result) == [(4, 40), (5, 50)]

    def test_combined_pushdown(self, duckdb_cursor):
        """Projection + filter pushdown combined."""
        obj = ArrowStream(pa.table({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}))  # noqa: F841
        result = duckdb_cursor.sql("SELECT b FROM obj WHERE a <= 2").fetchall()
        assert sorted(result) == [(10,), (20,)]


class TestPyCapsuleInterfaceSchemaOptimization:
    """GetSchema() uses __arrow_c_schema__ when available to avoid allocating a stream."""

    def test_arrow_c_schema_avoids_stream_call(self, duckdb_cursor):
        """When __arrow_c_schema__ is available, GetSchema() does not call __arrow_c_stream__."""
        obj = ArrowStreamWithSchema(pa.table({"a": [1, 2, 3]}))
        duckdb_cursor.sql("SELECT * FROM obj").fetchall()
        # With __arrow_c_schema__: only Produce() calls __arrow_c_stream__ (1 call).
        # Without it: GetSchema() fallback + Produce() = 2 calls.
        assert obj.stream_count == 1

    def test_without_arrow_c_schema_uses_stream_fallback(self, duckdb_cursor):
        """Without __arrow_c_schema__, GetSchema() falls back to __arrow_c_stream__."""
        obj = ArrowStream(pa.table({"a": [1, 2, 3]}))
        duckdb_cursor.sql("SELECT * FROM obj").fetchall()
        # GetSchema() fallback (1) + Produce() (1) = 2 calls minimum
        assert obj.stream_count >= 2

    def test_dot_schema_avoids_stream_call(self, duckdb_cursor):
        """When .schema with _export_to_c is available, GetSchema() uses it instead of __arrow_c_stream__."""
        obj = ArrowStreamWithDotSchema(pa.table({"a": [1, 2, 3]}))
        result = duckdb_cursor.sql("SELECT * FROM obj").fetchall()
        assert result == [(1,), (2,), (3,)]
        # With .schema: only Produce() calls __arrow_c_stream__ (1 call).
        assert obj.stream_count == 1

    def test_schema_via_dotschema_preserves_stream(self, duckdb_cursor):
        """A SingleUseArrowStream can be scanned because GetSchema uses .schema."""
        obj = SingleUseArrowStream(pa.table({"a": [1, 2, 3], "b": [10, 20, 30]}))
        result = duckdb_cursor.sql("SELECT a, b FROM obj").fetchall()
        assert sorted(result) == [(1, 10), (2, 20), (3, 30)]
        # Only 1 call to __arrow_c_stream__ (from Produce), schema came from .schema
        assert obj.stream_count == 1

    def test_schema_fallback_order(self, duckdb_cursor):
        """Schema extraction priority: __arrow_c_schema__ > .schema._export_to_c > __arrow_c_stream__."""
        # Object with __arrow_c_schema__ — should use that, not .schema or stream
        obj_with_capsule_schema = ArrowStreamWithSchema(pa.table({"x": [1]}))
        duckdb_cursor.sql("SELECT * FROM obj_with_capsule_schema").fetchall()
        assert obj_with_capsule_schema.stream_count == 1  # only Produce

        # Object with .schema — should use that, not stream
        obj_with_dot_schema = ArrowStreamWithDotSchema(pa.table({"x": [1]}))
        duckdb_cursor.sql("SELECT * FROM obj_with_dot_schema").fetchall()
        assert obj_with_dot_schema.stream_count == 1  # only Produce

        # Object with neither — falls back to stream
        obj_bare = ArrowStream(pa.table({"x": [1]}))
        duckdb_cursor.sql("SELECT * FROM obj_bare").fetchall()
        assert obj_bare.stream_count >= 2  # GetSchema + Produce


class TestPyArrowTableUnifiedPath:
    """PyArrow Table now enters via __arrow_c_stream__ (PyCapsuleInterface path).

    This verifies that Table gets multi-scan, pushdown, and correct results
    through the unified path instead of the old dedicated Table branch.
    """

    def test_pyarrow_table_scan(self, duckdb_cursor):
        """Basic scan of a PyArrow Table through the unified path."""
        tbl = pa.table({"a": [1, 2, 3], "b": [10, 20, 30]})  # noqa: F841
        result = duckdb_cursor.sql("SELECT * FROM tbl").fetchall()
        assert sorted(result) == [(1, 10), (2, 20), (3, 30)]

    def test_pyarrow_table_projection(self, duckdb_cursor):
        """Projection pushdown on a PyArrow Table."""
        tbl = pa.table({"a": [1, 2, 3], "b": [10, 20, 30], "c": ["x", "y", "z"]})  # noqa: F841
        result = duckdb_cursor.sql("SELECT a FROM tbl").fetchall()
        assert result == [(1,), (2,), (3,)]

    def test_pyarrow_table_filter(self, duckdb_cursor):
        """Filter pushdown on a PyArrow Table."""
        tbl = pa.table({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})  # noqa: F841
        result = duckdb_cursor.sql("SELECT a, b FROM tbl WHERE a > 3").fetchall()
        assert sorted(result) == [(4, 40), (5, 50)]

    def test_pyarrow_table_combined_pushdown(self, duckdb_cursor):
        """Projection + filter pushdown on a PyArrow Table."""
        tbl = pa.table({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})  # noqa: F841
        result = duckdb_cursor.sql("SELECT b FROM tbl WHERE a <= 2").fetchall()
        assert sorted(result) == [(10,), (20,)]

    def test_pyarrow_table_union_all(self, duckdb_cursor):
        """Table scanned twice in one query via UNION ALL."""
        tbl = pa.table({"id": [1, 2, 3]})  # noqa: F841
        result = duckdb_cursor.sql("SELECT id FROM tbl UNION ALL SELECT id FROM tbl").fetchall()
        assert sorted(r[0] for r in result) == [1, 1, 2, 2, 3, 3]

    def test_pyarrow_table_rescan(self, duckdb_cursor):
        """Table can be scanned across multiple queries."""
        tbl = pa.table({"id": [1, 2, 3]})  # noqa: F841
        r1 = duckdb_cursor.sql("SELECT * FROM tbl").fetchall()
        r2 = duckdb_cursor.sql("SELECT * FROM tbl").fetchall()
        assert r1 == r2 == [(1,), (2,), (3,)]


class TestRecordBatchReaderSingleUse:
    """RecordBatchReaders are inherently single-use streams.

    After the first scan consumes the reader, subsequent scans return empty results.
    This is correct behavior — RecordBatchReaders represent forward-only streams
    (e.g., reading from a socket or file).
    """

    def test_second_scan_empty(self, duckdb_cursor):
        """Second scan of a RecordBatchReader returns empty results."""
        reader = pa.RecordBatchReader.from_batches(  # noqa: F841
            pa.schema([("id", pa.int64())]),
            [pa.record_batch([pa.array([1, 2, 3])], names=["id"])],
        )
        r1 = duckdb_cursor.sql("SELECT * FROM reader").fetchall()
        assert r1 == [(1,), (2,), (3,)]
        r2 = duckdb_cursor.sql("SELECT * FROM reader").fetchall()
        assert r2 == []

    def test_register_second_scan_empty(self, duckdb_cursor):
        """Registered RecordBatchReader is also single-use."""
        reader = pa.RecordBatchReader.from_batches(
            pa.schema([("id", pa.int64())]),
            [pa.record_batch([pa.array([1, 2, 3])], names=["id"])],
        )
        duckdb_cursor.register("my_reader", reader)
        r1 = duckdb_cursor.sql("SELECT * FROM my_reader").fetchall()
        assert r1 == [(1,), (2,), (3,)]
        r2 = duckdb_cursor.sql("SELECT * FROM my_reader").fetchall()
        assert r2 == []

    def test_has_pushdown(self, duckdb_cursor):
        """RecordBatchReader gets projection/filter pushdown (not materialized)."""
        reader = pa.RecordBatchReader.from_batches(  # noqa: F841
            pa.schema([("a", pa.int64()), ("b", pa.int64())]),
            [pa.record_batch([pa.array([1, 2, 3]), pa.array([10, 20, 30])], names=["a", "b"])],
        )
        result = duckdb_cursor.sql("SELECT b FROM reader WHERE a > 1").fetchall()
        assert sorted(result) == [(20,), (30,)]


class TestPyCapsuleConsumed:
    """Issue #105: scanning a bare PyCapsule twice.

    Bare PyCapsules are single-use (the capsule IS the stream, not a stream factory).
    The fix ensures a clear InvalidInputException instead of InternalException.
    """

    def test_error_type(self, duckdb_cursor):
        """Consumed PyCapsule raises InvalidInputException, not InternalException."""
        tbl = pa.table({"a": [1]})
        capsule = tbl.__arrow_c_stream__()  # noqa: F841
        duckdb_cursor.sql("SELECT * FROM capsule").fetchall()
        # Error thrown by GetArrowType() in pyconnection.cpp when it detects the released stream.
        with pytest.raises(duckdb.InvalidInputException, match="The ArrowArrayStream was already released"):
            duckdb_cursor.sql("SELECT * FROM capsule")

    def test_pycapsule_interface_not_affected(self, duckdb_cursor):
        """Scanning through the PyCapsuleInterface object (not the capsule) works repeatedly."""
        obj = ArrowStream(pa.table({"a": [1, 2, 3]}))  # noqa: F841

        # First scan
        r1 = duckdb_cursor.sql("SELECT * FROM obj").fetchall()
        assert r1 == [(1,), (2,), (3,)]

        # Second scan — works because __arrow_c_stream__() is called lazily each time
        r2 = duckdb_cursor.sql("SELECT * FROM obj").fetchall()
        assert r2 == [(1,), (2,), (3,)]


class TestSameConnectionRecordBatchReader:
    """Issue #85: DuckDB-originated RecordBatchReader on the same connection.

    When conn.sql(...).to_arrow_reader() returns a RecordBatchReader backed by
    the same connection, scanning it on that connection may deadlock or return
    empty results due to lock contention. Run in subprocess to avoid hanging
    the test suite. The workaround is to use a different connection for the scan.
    """

    def test_same_connection_no_data(self):
        """Same-connection RecordBatchReader scan fails to return data.

        Run in subprocess to prevent hanging the test suite if it deadlocks.
        """
        code = """\
import duckdb
conn = duckdb.connect("")
reader = conn.sql("FROM range(5) T(a)").to_arrow_reader()
result = conn.sql("FROM reader").fetchall()
assert result != [(i,) for i in range(5)], "Expected no data due to lock contention"
"""
        with contextlib.suppress(subprocess.TimeoutExpired):
            subprocess.run(
                [sys.executable, "-c", code],
                timeout=5,
                capture_output=True,
            )

    def test_different_connection_works(self, duckdb_cursor):
        """RecordBatchReader from connection A scanned on connection B works fine."""
        conn_a = duckdb.connect()
        conn_b = duckdb.connect()
        reader = conn_a.sql("FROM range(5) T(a)").to_arrow_reader()  # noqa: F841
        result = conn_b.sql("FROM reader").fetchall()
        assert result == [(i,) for i in range(5)]

    def test_arrow_method_different_connection(self, duckdb_cursor):
        """The .arrow() method (which returns RecordBatchReader) works cross-connection."""
        conn_a = duckdb.connect()
        conn_b = duckdb.connect()
        arrow_reader = conn_a.sql("FROM range(5) T(a)").arrow()  # noqa: F841
        result = conn_b.sql("FROM arrow_reader").fetchall()
        assert result == [(i,) for i in range(5)]


class TestPyCapsuleInterfaceNoPyarrowDataset:
    """Tier B fallback: PyCapsuleInterface objects are scannable without pyarrow.dataset.

    When pyarrow.dataset is not available, PyCapsuleInterface uses arrow_scan_dumb
    (no pushdown). DuckDB handles projection/filter post-scan.
    Run in subprocess to avoid polluting the test process's import state.
    """

    def _run_in_subprocess(self, code):
        result = subprocess.run(
            [sys.executable, "-c", code],
            timeout=30,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            msg = f"Subprocess failed (rc={result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}"
            raise AssertionError(msg)

    def test_pycapsule_interface_no_pyarrow_dataset(self):
        """PyCapsuleInterface objects can be scanned without pyarrow.dataset."""
        self._run_in_subprocess("""\
import pyarrow as pa
import duckdb

class MyStream:
    def __init__(self, tbl):
        self.tbl = tbl
    def __arrow_c_stream__(self, requested_schema=None):
        return self.tbl.__arrow_c_stream__(requested_schema=requested_schema)
    def __arrow_c_schema__(self):
        return self.tbl.schema.__arrow_c_schema__()

obj = MyStream(pa.table({"a": [1, 2, 3], "b": [10, 20, 30]}))
result = duckdb.sql("SELECT * FROM obj").fetchall()
assert sorted(result) == [(1, 10), (2, 20), (3, 30)], f"Unexpected: {result}"
""")

    def test_pycapsule_interface_no_pyarrow_dataset_projection(self):
        """DuckDB applies projection post-scan when pyarrow.dataset unavailable."""
        self._run_in_subprocess("""\
import pyarrow as pa
import duckdb

class MyStream:
    def __init__(self, tbl):
        self.tbl = tbl
    def __arrow_c_stream__(self, requested_schema=None):
        return self.tbl.__arrow_c_stream__(requested_schema=requested_schema)
    def __arrow_c_schema__(self):
        return self.tbl.schema.__arrow_c_schema__()

obj = MyStream(pa.table({"a": [1, 2, 3], "b": [10, 20, 30], "c": ["x", "y", "z"]}))
result = duckdb.sql("SELECT a FROM obj").fetchall()
assert result == [(1,), (2,), (3,)], f"Unexpected: {result}"
""")

    def test_pycapsule_interface_no_pyarrow_dataset_filter(self):
        """DuckDB applies filter post-scan when pyarrow.dataset unavailable."""
        self._run_in_subprocess("""\
import pyarrow as pa
import duckdb

class MyStream:
    def __init__(self, tbl):
        self.tbl = tbl
    def __arrow_c_stream__(self, requested_schema=None):
        return self.tbl.__arrow_c_stream__(requested_schema=requested_schema)
    def __arrow_c_schema__(self):
        return self.tbl.schema.__arrow_c_schema__()

obj = MyStream(pa.table({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}))
result = duckdb.sql("SELECT a, b FROM obj WHERE a > 3").fetchall()
assert sorted(result) == [(4, 40), (5, 50)], f"Unexpected: {result}"
""")

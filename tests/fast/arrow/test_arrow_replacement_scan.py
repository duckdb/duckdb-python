import gc
import subprocess
import sys
import weakref
from pathlib import Path

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
ds = pytest.importorskip("pyarrow.dataset")


class TestArrowReplacementScan:
    def test_arrow_table_replacement_scan(self, duckdb_cursor):
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        userdata_parquet_table = pq.read_table(parquet_filename)
        df = userdata_parquet_table.to_pandas()  # noqa: F841

        con = duckdb.connect()

        for _i in range(5):
            assert con.execute("select count(*) from userdata_parquet_table").fetchone() == (1000,)
            assert con.execute("select count(*) from df").fetchone() == (1000,)

    @pytest.mark.skipif(
        not hasattr(pa.Table, "__arrow_c_stream__"),
        reason="This version of pyarrow does not support the Arrow Capsule Interface",
    )
    def test_arrow_pycapsule_replacement_scan(self, duckdb_cursor):
        tbl = pa.Table.from_pydict({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        capsule = tbl.__arrow_c_stream__()

        rel = duckdb_cursor.sql("select * from capsule")
        assert rel.fetchall() == [(i,) for i in range(1, 10)]

        capsule = tbl.__arrow_c_stream__()
        rel = duckdb_cursor.sql("select * from capsule where a > 3 and a < 5")
        assert rel.fetchall() == [(4,)]

        tbl = pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]})
        capsule = tbl.__arrow_c_stream__()  # noqa: F841

        rel = duckdb_cursor.sql("select b, d from capsule")
        assert rel.fetchall() == [(i, i + 6) for i in range(4, 7)]

        with pytest.raises(duckdb.InvalidInputException, match="The ArrowArrayStream was already released"):
            duckdb_cursor.sql("select b, d from capsule")

        schema_obj = tbl.schema
        schema_capsule = schema_obj.__arrow_c_schema__()  # noqa: F841
        with pytest.raises(
            duckdb.InvalidInputException, match="""Expected a 'arrow_array_stream' PyCapsule, got: arrow_schema"""
        ):
            duckdb_cursor.sql("select b, d from schema_capsule")

    def test_arrow_table_replacement_scan_view(self, duckdb_cursor):
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        userdata_parquet_table = pq.read_table(parquet_filename)

        con = duckdb.connect()

        con.execute("create view x as select * from userdata_parquet_table")
        del userdata_parquet_table
        with pytest.raises(duckdb.CatalogException, match="Table with name userdata_parquet_table does not exist"):
            assert con.execute("select count(*) from x").fetchone()

    def test_arrow_dataset_replacement_scan(self, duckdb_cursor):
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        pq.read_table(parquet_filename)
        userdata_parquet_dataset = ds.dataset(parquet_filename)  # noqa: F841

        con = duckdb.connect()
        assert con.execute("select count(*) from userdata_parquet_dataset").fetchone() == (1000,)


def _make_table():
    return pa.table({"a": [1, 2, 3, 4, 5]})


def _make_dataset():
    return ds.dataset(_make_table())


def _make_scanner():
    return ds.dataset(_make_table()).scanner()


def _make_record_batch_reader():
    table = _make_table()
    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())


def _make_polars_frame():
    pl = pytest.importorskip("polars")
    return pl.DataFrame({"a": [1, 2, 3, 4, 5]})


def _make_polars_lazy_frame():
    pl = pytest.importorskip("polars")
    return pl.DataFrame({"a": [1, 2, 3, 4, 5]}).lazy()


FACTORIES = [
    ("table", _make_table),
    ("dataset", _make_dataset),
    ("scanner", _make_scanner),
    ("record_batch_reader", _make_record_batch_reader),
    ("polars_frame", _make_polars_frame),
    ("polars_lazy_frame", _make_polars_lazy_frame),
]

# An eager polars DataFrame is materialized through .to_arrow() before the scan is built, so the
# scan retains that conversion and not the frame itself.
RETAINED_FACTORIES = [entry for entry in FACTORIES if entry[0] != "polars_frame"]


class TestArrowScanFactoryOwnership:
    """The scan factory owns the Arrow object for as long as the engine holds the scan."""

    @pytest.mark.parametrize(("label", "make"), FACTORIES, ids=[label for label, _ in FACTORIES])
    def test_scanned_object_survives_its_python_variable(self, label, make):
        con = duckdb.connect()
        scanned = make()
        rel = con.sql("select sum(a) as s from scanned")

        del scanned
        gc.collect()

        assert rel.fetchall() == [(15,)]

    @pytest.mark.parametrize(("label", "make"), RETAINED_FACTORIES, ids=[label for label, _ in RETAINED_FACTORIES])
    def test_scanned_object_is_released_with_the_relation(self, label, make):
        con = duckdb.connect()

        # The scan reads the caller's frame locals, and before Python 3.13 that read caches a dict on
        # the frame which keeps the object alive until the frame returns, so the scan gets its own frame.
        def scan(con):
            scanned = make()
            rel = con.sql("select sum(a) as s from scanned")
            return weakref.ref(scanned), rel

        ref, rel = scan(con)
        gc.collect()
        assert ref() is not None

        assert rel.fetchall() == [(15,)]

        del rel
        con.close()
        del con
        gc.collect()
        assert ref() is None

    @pytest.mark.parametrize(("label", "make"), FACTORIES, ids=[label for label, _ in FACTORIES])
    def test_scan_through_execute_outlives_the_bind(self, label, make):
        # Without a relation there is no dependency to keep the scan alive past binding: the bind
        # data alone must hold the factory until the scan runs.
        con = duckdb.connect()
        scanned = make()
        assert con.execute("select sum(a) as s from scanned").fetchall() == [(15,)]
        del scanned

    def test_repeated_scans_of_one_object(self):
        con = duckdb.connect()
        scanned = _make_table()
        rel = con.sql("select sum(a) as s from scanned")

        del scanned
        gc.collect()

        assert rel.fetchall() == [(15,)]
        assert rel.fetchall() == [(15,)]


SOURCE_CLOSING_ITS_CONNECTION = """
import sys

import duckdb
import pyarrow as pa

con = duckdb.connect()
table = pa.table({"a": [1, 2]})


class Source:
    @property
    def schema(self):
        con.close()
        return table.schema

    def __arrow_c_stream__(self):
        return table.__arrow_c_stream__()


try:
    if sys.argv[1] == "from_arrow":
        con.from_arrow(Source()).fetchall()
    elif sys.argv[1] == "register":
        con.register("src", Source())
    else:
        src = Source()
        con.sql("SELECT * FROM src").fetchall()
except duckdb.Error as error:
    print("refused", "cannot use" in str(error) or "busy" in str(error), flush=True)
print("still open", con.sql("SELECT 42").fetchall(), flush=True)
"""


@pytest.mark.parametrize("entry_point", ["from_arrow", "register", "variable"])
def test_a_source_closing_its_connection_while_adapted_is_refused(entry_point):
    completed = subprocess.run(
        [sys.executable, "-c", SOURCE_CLOSING_ITS_CONNECTION, entry_point],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "refused True" in completed.stdout.splitlines(), completed.stdout
    assert "still open [(42,)]" in completed.stdout.splitlines(), completed.stdout

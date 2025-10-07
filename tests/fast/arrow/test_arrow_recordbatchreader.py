from pathlib import Path

import pytest

import duckdb

pyarrow = pytest.importorskip("pyarrow")
pyarrow.parquet = pytest.importorskip("pyarrow.parquet")
pyarrow.dataset = pytest.importorskip("pyarrow.dataset")
np = pytest.importorskip("numpy")


class TestArrowRecordBatchReader:
    def test_parallel_reader(self, duckdb_cursor):
        duckdb_conn = duckdb.connect()
        duckdb_conn.execute("PRAGMA threads=4")

        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")

        userdata_parquet_dataset = pyarrow.dataset.dataset(
            [
                parquet_filename,
                parquet_filename,
                parquet_filename,
            ],
            format="parquet",
        )

        batches = list(userdata_parquet_dataset.to_batches())
        reader = pyarrow.dataset.Scanner.from_batches(batches, schema=userdata_parquet_dataset.schema).to_reader()

        rel = duckdb_conn.from_arrow(reader)

        assert (
            rel.filter("first_name='Jose' and salary > 134708.82").aggregate("count(*)").execute().fetchone()[0] == 12
        )
        # The reader is already consumed so this should be 0
        assert rel.filter("first_name='Jose' and salary > 134708.82").aggregate("count(*)").execute().fetchone()[0] == 0

    def test_parallel_reader_replacement_scans(self, duckdb_cursor):
        duckdb_conn = duckdb.connect()
        duckdb_conn.execute("PRAGMA threads=4")

        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")

        userdata_parquet_dataset = pyarrow.dataset.dataset(
            [
                parquet_filename,
                parquet_filename,
                parquet_filename,
            ],
            format="parquet",
        )

        batches = list(userdata_parquet_dataset.to_batches())
        reader = pyarrow.dataset.Scanner.from_batches(batches, schema=userdata_parquet_dataset.schema).to_reader()  # noqa: F841

        assert (
            duckdb_conn.execute(
                "select count(*) r1 from reader where first_name='Jose' and salary > 134708.82"
            ).fetchone()[0]
            == 12
        )
        assert (
            duckdb_conn.execute(
                "select count(*) r2 from reader where first_name='Jose' and salary > 134708.82"
            ).fetchone()[0]
            == 0
        )

    def test_parallel_reader_register(self, duckdb_cursor):
        duckdb_conn = duckdb.connect()
        duckdb_conn.execute("PRAGMA threads=4")

        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")

        userdata_parquet_dataset = pyarrow.dataset.dataset(
            [
                parquet_filename,
                parquet_filename,
                parquet_filename,
            ],
            format="parquet",
        )

        batches = list(userdata_parquet_dataset.to_batches())
        reader = pyarrow.dataset.Scanner.from_batches(batches, schema=userdata_parquet_dataset.schema).to_reader()

        duckdb_conn.register("bla", reader)

        assert (
            duckdb_conn.execute("select count(*) from bla where first_name='Jose' and salary > 134708.82").fetchone()[0]
            == 12
        )
        assert (
            duckdb_conn.execute("select count(*) from bla where first_name='Jose' and salary > 134708.82").fetchone()[0]
            == 0
        )

    def test_parallel_reader_default_conn(self, duckdb_cursor):
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")

        userdata_parquet_dataset = pyarrow.dataset.dataset(
            [
                parquet_filename,
                parquet_filename,
                parquet_filename,
            ],
            format="parquet",
        )

        batches = list(userdata_parquet_dataset.to_batches())
        reader = pyarrow.dataset.Scanner.from_batches(batches, schema=userdata_parquet_dataset.schema).to_reader()

        rel = duckdb.from_arrow(reader)

        assert (
            rel.filter("first_name='Jose' and salary > 134708.82").aggregate("count(*)").execute().fetchone()[0] == 12
        )
        # The reader is already consumed so this should be 0
        assert rel.filter("first_name='Jose' and salary > 134708.82").aggregate("count(*)").execute().fetchone()[0] == 0

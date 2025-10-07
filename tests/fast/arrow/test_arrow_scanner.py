from pathlib import Path

import duckdb

try:
    import pyarrow
    import pyarrow.compute as pc
    import pyarrow.dataset
    import pyarrow.parquet
    from pyarrow.dataset import Scanner

    can_run = True
except Exception:
    can_run = False


class TestArrowScanner:
    def test_parallel_scanner(self, duckdb_cursor):
        if not can_run:
            return

        duckdb_conn = duckdb.connect()
        duckdb_conn.execute("PRAGMA threads=4")

        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")

        arrow_dataset = pyarrow.dataset.dataset(
            [
                parquet_filename,
                parquet_filename,
                parquet_filename,
            ],
            format="parquet",
        )

        scanner_filter = (pc.field("first_name") == pc.scalar("Jose")) & (pc.field("salary") > pc.scalar(134708.82))

        arrow_scanner = Scanner.from_dataset(arrow_dataset, filter=scanner_filter)

        rel = duckdb_conn.from_arrow(arrow_scanner)

        assert rel.aggregate("count(*)").execute().fetchone()[0] == 12

    def test_parallel_scanner_replacement_scans(self, duckdb_cursor):
        if not can_run:
            return

        duckdb_conn = duckdb.connect()
        duckdb_conn.execute("PRAGMA threads=4")

        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")

        arrow_dataset = pyarrow.dataset.dataset(
            [
                parquet_filename,
                parquet_filename,
                parquet_filename,
            ],
            format="parquet",
        )

        scanner_filter = (pc.field("first_name") == pc.scalar("Jose")) & (pc.field("salary") > pc.scalar(134708.82))

        arrow_scanner = Scanner.from_dataset(arrow_dataset, filter=scanner_filter)  # noqa: F841

        assert duckdb_conn.execute("select count(*) from arrow_scanner").fetchone()[0] == 12

    def test_parallel_scanner_register(self, duckdb_cursor):
        if not can_run:
            return

        duckdb_conn = duckdb.connect()
        duckdb_conn.execute("PRAGMA threads=4")

        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")

        arrow_dataset = pyarrow.dataset.dataset(
            [
                parquet_filename,
                parquet_filename,
                parquet_filename,
            ],
            format="parquet",
        )

        scanner_filter = (pc.field("first_name") == pc.scalar("Jose")) & (pc.field("salary") > pc.scalar(134708.82))

        arrow_scanner = Scanner.from_dataset(arrow_dataset, filter=scanner_filter)

        duckdb_conn.register("bla", arrow_scanner)

        assert duckdb_conn.execute("select count(*) from bla").fetchone()[0] == 12

    def test_parallel_scanner_default_conn(self, duckdb_cursor):
        if not can_run:
            return

        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")

        arrow_dataset = pyarrow.dataset.dataset(
            [
                parquet_filename,
                parquet_filename,
                parquet_filename,
            ],
            format="parquet",
        )

        scanner_filter = (pc.field("first_name") == pc.scalar("Jose")) & (pc.field("salary") > pc.scalar(134708.82))

        arrow_scanner = Scanner.from_dataset(arrow_dataset, filter=scanner_filter)

        rel = duckdb.from_arrow(arrow_scanner)

        assert rel.aggregate("count(*)").execute().fetchone()[0] == 12

from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


class TestArrowView:
    def test_arrow_view(self, duckdb_cursor):
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        userdata_parquet_table = pa.parquet.read_table(parquet_filename)
        userdata_parquet_table.validate(full=True)
        duckdb_cursor.from_arrow(userdata_parquet_table).create_view("arrow_view")
        assert duckdb_cursor.execute("PRAGMA show_tables").fetchone() == ("arrow_view",)
        assert duckdb_cursor.execute("select avg(salary)::INT from arrow_view").fetchone()[0] == 149005

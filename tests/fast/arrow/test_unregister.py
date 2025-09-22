import gc
import tempfile
from pathlib import Path

import pytest

import duckdb

pyarrow = pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")


class TestArrowUnregister:
    def test_arrow_unregister1(self, duckdb_cursor):
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        arrow_table_obj = pyarrow.parquet.read_table(parquet_filename)
        connection = duckdb.connect(":memory:")
        connection.register("arrow_table", arrow_table_obj)

        connection.execute("SELECT * FROM arrow_table;").fetch_arrow_table()
        connection.unregister("arrow_table")
        with pytest.raises(duckdb.CatalogException, match="Table with name arrow_table does not exist"):
            connection.execute("SELECT * FROM arrow_table;").fetch_arrow_table()
        with pytest.raises(duckdb.CatalogException, match="View with name arrow_table does not exist"):
            connection.execute("DROP VIEW arrow_table;")
        connection.execute("DROP VIEW IF EXISTS arrow_table;")

    def test_arrow_unregister2(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name

        connection = duckdb.connect(db)
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        arrow_table_obj = pyarrow.parquet.read_table(parquet_filename)
        connection.register("arrow_table", arrow_table_obj)
        connection.unregister("arrow_table")  # Attempting to unregister.
        connection.close()
        # Reconnecting while Arrow Table still in mem.
        connection = duckdb.connect(db)
        assert len(connection.execute("PRAGMA show_tables;").fetchall()) == 0
        with pytest.raises(duckdb.CatalogException, match="Table with name arrow_table does not exist"):
            connection.execute("SELECT * FROM arrow_table;").fetch_arrow_table()
        connection.close()
        del arrow_table_obj
        gc.collect()
        # Reconnecting after Arrow Table is freed.
        connection = duckdb.connect(db)
        assert len(connection.execute("PRAGMA show_tables;").fetchall()) == 0
        with pytest.raises(duckdb.CatalogException, match="Table with name arrow_table does not exist"):
            connection.execute("SELECT * FROM arrow_table;").fetch_arrow_table()
        connection.close()

import gc
import tempfile
import weakref

import pandas as pd
import pytest

import duckdb


class TestPandasUnregister:
    def test_pandas_unregister1(self, duckdb_cursor):
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        connection = duckdb.connect(":memory:")
        connection.register("dataframe", df)

        df2 = connection.execute("SELECT * FROM dataframe;").fetchdf()  # noqa: F841
        connection.unregister("dataframe")
        with pytest.raises(duckdb.CatalogException, match="Table with name dataframe does not exist"):
            connection.execute("SELECT * FROM dataframe;").fetchdf()
        with pytest.raises(duckdb.CatalogException, match="View with name dataframe does not exist"):
            connection.execute("DROP VIEW dataframe;")
        connection.execute("DROP VIEW IF EXISTS dataframe;")

    def test_pandas_unregister2(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name

        connection = duckdb.connect(db)
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

        connection.register("dataframe", df)
        connection.unregister("dataframe")  # Attempting to unregister.
        connection.close()

        # Reconnecting while DataFrame still in mem.
        connection = duckdb.connect(db)
        assert len(connection.execute("PRAGMA show_tables;").fetchall()) == 0

        with pytest.raises(duckdb.CatalogException, match="Table with name dataframe does not exist"):
            connection.execute("SELECT * FROM dataframe;").fetchdf()

        connection.close()

        del df
        gc.collect()

        # Reconnecting after DataFrame freed.
        connection = duckdb.connect(db)
        assert len(connection.execute("PRAGMA show_tables;").fetchall()) == 0
        with pytest.raises(duckdb.CatalogException, match="Table with name dataframe does not exist"):
            connection.execute("SELECT * FROM dataframe;").fetchdf()
        connection.close()

    def test_pandas_unregister_releases_object_inside_transaction(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t(i BIGINT)")
        duckdb_cursor.begin()

        df = pd.DataFrame({"i": [1, 2, 3]})
        ref = weakref.ref(df)

        duckdb_cursor.register("dataframe", df)
        duckdb_cursor.execute("INSERT INTO t SELECT * FROM dataframe")
        duckdb_cursor.unregister("dataframe")

        del df
        gc.collect()

        assert ref() is None
        duckdb_cursor.rollback()

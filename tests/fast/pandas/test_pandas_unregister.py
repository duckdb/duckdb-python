import gc
import tempfile

import pytest
from conftest import ArrowPandas, NumpyPandas

import duckdb


class TestPandasUnregister:
    @pytest.mark.parametrize("pandas", [NumpyPandas(), ArrowPandas()])
    def test_pandas_unregister1(self, duckdb_cursor, pandas):
        df = pandas.DataFrame([[1, 2, 3], [4, 5, 6]])
        connection = duckdb.connect(":memory:")
        connection.register("dataframe", df)

        df2 = connection.execute("SELECT * FROM dataframe;").fetchdf()  # noqa: F841
        connection.unregister("dataframe")
        with pytest.raises(duckdb.CatalogException, match="Table with name dataframe does not exist"):
            connection.execute("SELECT * FROM dataframe;").fetchdf()
        with pytest.raises(duckdb.CatalogException, match="View with name dataframe does not exist"):
            connection.execute("DROP VIEW dataframe;")
        connection.execute("DROP VIEW IF EXISTS dataframe;")

    @pytest.mark.parametrize("pandas", [NumpyPandas(), ArrowPandas()])
    def test_pandas_unregister2(self, duckdb_cursor, pandas):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name

        connection = duckdb.connect(db)
        df = pandas.DataFrame([[1, 2, 3], [4, 5, 6]])

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

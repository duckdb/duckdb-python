# simple DB API testcase

import pytest

import duckdb
import pandas as pd


class TestImplicitPandasScan:
    def test_pandas_scan_takes_no_sql_arguments(self, duckdb_cursor):
        # The dataframe travels through the bind input, never through a value SQL text can forge
        with pytest.raises(duckdb.BinderException, match="No function matches"):
            duckdb_cursor.execute("select * from pandas_scan(140234567890)")
        with pytest.raises(duckdb.BinderException, match="requires a dataframe bind input"):
            duckdb_cursor.execute("select * from pandas_scan()")

    def test_two_dataframes_in_one_query(self, duckdb_cursor):
        lhs = pd.DataFrame({"k": [1, 2, 3], "a": ["x", "y", "z"]})  # noqa: F841
        rhs = pd.DataFrame({"k": [2, 3, 4], "b": [20, 30, 40]})  # noqa: F841
        rows = duckdb_cursor.execute("select a, b from lhs join rhs using (k) order by k").fetchall()
        assert rows == [("y", 20), ("z", 30)]

    def test_local_pandas_scan(self, duckdb_cursor):
        con = duckdb.connect()
        df = pd.DataFrame([{"COL1": "val1", "CoL2": 1.05}, {"COL1": "val3", "CoL2": 17}])  # noqa: F841
        r1 = con.execute("select * from df").fetchdf()
        assert r1["COL1"][0] == "val1"
        assert r1["COL1"][1] == "val3"
        assert r1["CoL2"][0] == 1.05
        assert r1["CoL2"][1] == 17

    def test_global_pandas_scan(self, duckdb_cursor):
        """Test that DuckDB can scan a module-level DataFrame variable."""
        con = duckdb.connect()
        # Create a global-scope dataframe for this test
        global test_global_df
        test_global_df = pd.DataFrame([{"COL1": "val1", "CoL2": 1.05}, {"COL1": "val4", "CoL2": 17}])
        r1 = con.execute("select * from test_global_df").fetchdf()
        assert r1["COL1"][0] == "val1"
        assert r1["COL1"][1] == "val4"
        assert r1["CoL2"][0] == 1.05
        assert r1["CoL2"][1] == 17

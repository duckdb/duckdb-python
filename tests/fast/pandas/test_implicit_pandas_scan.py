# simple DB API testcase

import pandas as pd

import duckdb


class TestImplicitPandasScan:
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

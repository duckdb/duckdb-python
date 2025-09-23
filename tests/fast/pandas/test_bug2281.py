import io

import pandas as pd


class TestPandasStringNull:
    def test_pandas_string_null(self, duckdb_cursor):
        csv = """what,is_control,is_test
,0,0
foo,1,0"""
        df = pd.read_csv(io.StringIO(csv))
        duckdb_cursor.register("c", df)
        duckdb_cursor.execute("select what, count(*) from c group by what")
        duckdb_cursor.fetchdf()
        assert True  # Should not crash ^^

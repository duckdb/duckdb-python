import pytest

import duckdb

try:
    import pyarrow as pa

    can_run = True
except Exception:
    can_run = False
from conftest import ArrowPandas, NumpyPandas


class Test3654:
    @pytest.mark.parametrize("pandas", [NumpyPandas(), ArrowPandas()])
    def test_3654_pandas(self, duckdb_cursor, pandas):
        df1 = pandas.DataFrame(
            {
                "id": [1, 1, 2],
            }
        )
        con = duckdb.connect()
        con.register("df1", df1)
        rel = con.view("df1")
        print(rel.execute().fetchall())
        assert rel.execute().fetchall() == [(1,), (1,), (2,)]

    @pytest.mark.parametrize("pandas", [NumpyPandas(), ArrowPandas()])
    def test_3654_arrow(self, duckdb_cursor, pandas):
        if not can_run:
            return

        df1 = pandas.DataFrame(
            {
                "id": [1, 1, 2],
            }
        )
        table = pa.Table.from_pandas(df1)
        con = duckdb.connect()
        con.register("df1", table)
        rel = con.view("df1")
        print(rel.execute().fetchall())
        assert rel.execute().fetchall() == [(1,), (1,), (2,)]

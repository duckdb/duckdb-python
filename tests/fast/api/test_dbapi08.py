# test fetchdf with various types
import pandas as pd

import duckdb


class TestType:
    def test_fetchdf(self):
        con = duckdb.connect()
        con.execute("CREATE TABLE items(item VARCHAR)")
        con.execute("INSERT INTO items VALUES ('jeans'), (''), (NULL)")
        res = con.execute("SELECT item FROM items").fetchdf()
        assert isinstance(res, pd.core.frame.DataFrame)

        df = pd.DataFrame({"item": ["jeans", "", None]})

        print(res)
        print(df)
        pd.testing.assert_frame_equal(res, df, check_dtype=False)

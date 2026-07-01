import pandas as pd

import duckdb


class TestCaseAlias:
    def test_case_alias(self, duckdb_cursor):
        con = duckdb.connect(":memory:")

        df = pd.DataFrame([{"COL1": "val1", "CoL2": 1.05}, {"COL1": "val3", "CoL2": 17}])

        r1 = con.from_df(df).query("df", "select * from df").df()
        assert r1["COL1"][0] == "val1"
        assert r1["COL1"][1] == "val3"
        assert r1["CoL2"][0] == 1.05
        assert r1["CoL2"][1] == 17

        # An explicit column reference takes its output name from the casing as written in the query (COL2),
        # unlike `select *` above which preserves the source column's casing (CoL2).
        r2 = con.from_df(df).query("df", "select COL1, COL2 from df").df()
        assert r2["COL1"][0] == "val1"
        assert r2["COL1"][1] == "val3"
        assert r2["COL2"][0] == 1.05
        assert r2["COL2"][1] == 17

        r3 = con.from_df(df).query("df", "select COL1, COL2 from df ORDER BY COL1").df()
        assert r3["COL1"][0] == "val1"
        assert r3["COL1"][1] == "val3"
        assert r3["COL2"][0] == 1.05
        assert r3["COL2"][1] == 17

        r4 = con.from_df(df).query("df", "select COL1, COL2 from df GROUP BY COL1, COL2 ORDER BY COL1").df()
        assert r4["COL1"][0] == "val1"
        assert r4["COL1"][1] == "val3"
        assert r4["COL2"][0] == 1.05
        assert r4["COL2"][1] == 17

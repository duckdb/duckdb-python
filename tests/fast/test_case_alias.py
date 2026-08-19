import duckdb
import pandas as pd


class TestCaseAlias:
    def test_case_alias(self, duckdb_cursor):
        con = duckdb.connect(":memory:")

        df = pd.DataFrame([{"COL1": "val1", "CoL2": 1.05}, {"COL1": "val3", "CoL2": 17}])

        r1 = con.from_df(df).query("df", "select * from df").df()
        assert r1["COL1"][0] == "val1"
        assert r1["COL1"][1] == "val3"
        assert r1["CoL2"][0] == 1.05
        assert r1["CoL2"][1] == 17

        # A column reference does not rename the column: whatever casing it is written in, the output
        # keeps the source column's casing (CoL2). Before duckdb PR #24776 ("Fix case restoration for
        # mismatched case column references") a mismatched-case reference leaked its own casing into
        # the output name, so `select COL2` produced COL2.
        r2 = con.from_df(df).query("df", "select COL1, COL2 from df").df()
        assert r2["COL1"][0] == "val1"
        assert r2["COL1"][1] == "val3"
        assert r2["CoL2"][0] == 1.05
        assert r2["CoL2"][1] == 17

        r3 = con.from_df(df).query("df", "select COL1, COL2 from df ORDER BY COL1").df()
        assert r3["COL1"][0] == "val1"
        assert r3["COL1"][1] == "val3"
        assert r3["CoL2"][0] == 1.05
        assert r3["CoL2"][1] == 17

        r4 = con.from_df(df).query("df", "select COL1, COL2 from df GROUP BY COL1, COL2 ORDER BY COL1").df()
        assert r4["COL1"][0] == "val1"
        assert r4["COL1"][1] == "val3"
        assert r4["CoL2"][0] == 1.05
        assert r4["CoL2"][1] == 17

    def test_case_alias_explicit_alias_renames(self, duckdb_cursor):
        """An explicit alias still sets the output name, which is what distinguishes it from a reference."""
        con = duckdb.connect(":memory:")

        df = pd.DataFrame([{"COL1": "val1", "CoL2": 1.05}])

        renamed = con.from_df(df).query("df", "select COL1, COL2 as COL2 from df").df()
        assert list(renamed.columns) == ["COL1", "COL2"]
        assert renamed["COL2"][0] == 1.05

        # ... and it can rename to any casing, not just the source one
        lowered = con.from_df(df).query("df", "select CoL2 as col2 from df").df()
        assert list(lowered.columns) == ["col2"]

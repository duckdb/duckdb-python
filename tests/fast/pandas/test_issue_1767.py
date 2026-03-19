#!/usr/bin/env python

import pandas as pd

import duckdb


# Join from pandas not matching identical strings #1767
class TestIssue1767:
    def test_unicode_join_pandas(self, duckdb_cursor):
        A = pd.DataFrame({"key": ["a", "п"]})
        B = pd.DataFrame({"key": ["a", "п"]})
        con = duckdb.connect(":memory:")
        arrow = con.register("A", A).register("B", B)
        q = arrow.query("""SELECT key FROM "A" FULL JOIN "B" USING ("key") ORDER BY key""")
        result = q.df()

        d = {"key": ["a", "п"]}
        df = pd.DataFrame(data=d)
        pd.testing.assert_frame_equal(result, df, check_dtype=False)

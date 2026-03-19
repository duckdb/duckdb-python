import pandas as pd

import duckdb


class TestInsert:
    def test_insert(self):
        test_df = pd.DataFrame({"i": [1, 2, 3], "j": ["one", "two", "three"]})
        # connect to an in-memory temporary database
        conn = duckdb.connect()
        # get a cursor
        cursor = conn.cursor()
        conn.execute("CREATE TABLE test (i INTEGER, j STRING)")
        rel = conn.table("test")
        rel.insert([1, "one"])
        rel.insert([2, "two"])
        rel.insert([3, "three"])
        rel_a3 = cursor.table("test").project("CAST(i as BIGINT)i, j").to_df()
        pd.testing.assert_frame_equal(rel_a3, test_df)

    def test_insert_with_schema(self, duckdb_cursor):
        duckdb_cursor.sql("create schema not_main")
        duckdb_cursor.sql("create table not_main.tbl as select * from range(10)")

        res = duckdb_cursor.table("not_main.tbl").fetchall()
        assert len(res) == 10

        # Insert into a schema-qualified table should work; table has a single column from range(10)
        duckdb_cursor.table("not_main.tbl").insert([42])
        res2 = duckdb_cursor.table("not_main.tbl").fetchall()
        assert len(res2) == 11
        assert (42,) in res2

import time


class TestSqlEmptyParams:
    """Empty params should use lazy QueryRelation path (same as params=None)."""

    def test_empty_list_returns_same_result(self, duckdb_cursor):
        """sql(params=[]) returns same data as sql(params=None)."""
        duckdb_cursor.execute("CREATE TABLE t AS SELECT i FROM range(10) t(i)")
        expected = duckdb_cursor.sql("SELECT * FROM t").fetchall()
        result = duckdb_cursor.sql("SELECT * FROM t", params=[]).fetchall()
        assert result == expected

    def test_empty_dict_returns_same_result(self, duckdb_cursor):
        """sql(params={}) returns same data as sql(params=None)."""
        duckdb_cursor.execute("CREATE TABLE t AS SELECT i FROM range(10) t(i)")
        expected = duckdb_cursor.sql("SELECT * FROM t").fetchall()
        result = duckdb_cursor.sql("SELECT * FROM t", params={}).fetchall()
        assert result == expected

    def test_empty_tuple_returns_same_result(self, duckdb_cursor):
        """sql(params=()) returns same data as sql(params=None)."""
        duckdb_cursor.execute("CREATE TABLE t AS SELECT i FROM range(10) t(i)")
        expected = duckdb_cursor.sql("SELECT * FROM t").fetchall()
        result = duckdb_cursor.sql("SELECT * FROM t", params=()).fetchall()
        assert result == expected

    def test_empty_params_is_chainable(self, duckdb_cursor):
        """Empty params produces a real relation that supports chaining."""
        duckdb_cursor.execute("CREATE TABLE t AS SELECT i FROM range(10) t(i)")
        result = duckdb_cursor.sql("SELECT * FROM t", params=[]).filter("i < 3").order("i").fetchall()
        assert result == [(0,), (1,), (2,)]

    def test_empty_params_explain_is_fast(self, duckdb_cursor):
        """Empty params explain should not trigger expensive ToString."""
        duckdb_cursor.execute("CREATE TABLE t AS SELECT i FROM range(100000) t(i)")
        t0 = time.perf_counter()
        duckdb_cursor.sql("SELECT * FROM t", params=[]).explain()
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"explain() took {elapsed:.2f}s, expected < 5s"

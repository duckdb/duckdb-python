import duckdb


class TestAppenderChunk:
    def test_append_chunk(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER, s VARCHAR)")
        appender = duckdb_cursor.appender("t")
        appender.append_chunk([[1, "a"], [2, "b"], [3, "c"]])
        appender.close()
        assert duckdb_cursor.execute("SELECT * FROM t ORDER BY i").fetchall() == [(1, "a"), (2, "b"), (3, "c")]

    def test_append_chunk_wrong_width(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER, j INTEGER)")
        appender = duckdb_cursor.appender("t")
        try:
            appender.append_chunk([[1]])
            raise AssertionError("expected error")
        except duckdb.Error:
            pass
        appender.close()


class TestAppenderNested:
    def test_list(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER, xs INTEGER[])")
        appender = duckdb_cursor.appender("t")
        appender.append(1, [10, 20, 30])
        appender.append(2, [])
        appender.close()
        assert duckdb_cursor.execute("SELECT * FROM t ORDER BY i").fetchall() == [(1, [10, 20, 30]), (2, [])]

    def test_struct(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (x STRUCT(a INTEGER, b VARCHAR))")
        appender = duckdb_cursor.appender("t")
        appender.append({"a": 1, "b": "x"})
        appender.close()
        assert duckdb_cursor.execute("SELECT x FROM t").fetchall() == [({"a": 1, "b": "x"},)]

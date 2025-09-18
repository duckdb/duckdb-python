import duckdb


class TestAmbiguousPrepare:
    def test_bool(self, duckdb_cursor):
        conn = duckdb.connect()
        res = conn.execute("select ?, ?, ?", (True, 42, [1, 2, 3])).fetchall()
        assert res[0][0]
        assert res[0][1] == 42
        assert res[0][2] == [1, 2, 3]

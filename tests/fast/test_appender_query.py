class TestAppenderQuery:
    def test_insert_from_appended_data(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE tbl (i INTEGER PRIMARY KEY, value VARCHAR)")
        duckdb_cursor.execute("INSERT INTO tbl VALUES (1, 'hello')")
        appender = duckdb_cursor.appender_query("INSERT OR REPLACE INTO tbl FROM appended_data", ["INTEGER", "VARCHAR"])
        appender.append(1, "world")
        appender.append(2, "again")
        appender.close()
        assert duckdb_cursor.execute("SELECT * FROM tbl ORDER BY i").fetchall() == [(1, "world"), (2, "again")]

    def test_custom_table_name(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE tbl (i INTEGER)")
        appender = duckdb_cursor.appender_query(
            "INSERT INTO tbl SELECT * FROM incoming", ["INTEGER"], table_name="incoming"
        )
        appender.append(9)
        appender.close()
        assert duckdb_cursor.execute("SELECT i FROM tbl").fetchall() == [(9,)]

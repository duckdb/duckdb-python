# cursor description
from datetime import date, datetime

import pytest

import duckdb


class TestCursorDescription:
    @pytest.mark.parametrize(
        ("query", "column_name", "string_type", "real_type"),
        [
            ("SELECT * FROM integers", "i", "INTEGER", int),
            ("SELECT * FROM timestamps", "t", "TIMESTAMP", datetime),
            ("SELECT DATE '1992-09-20' AS date_col;", "date_col", "DATE", date),
            ("SELECT '\\xAA'::BLOB AS blob_col;", "blob_col", "BLOB", bytes),
            (
                "SELECT {'x': 1, 'y': 2, 'z': 3} AS struct_col",
                "struct_col",
                "STRUCT(x INTEGER, y INTEGER, z INTEGER)",
                dict,
            ),
            ("SELECT [1, 2, 3] AS list_col", "list_col", "INTEGER[]", list),
            ("SELECT 'Frank' AS str_col", "str_col", "VARCHAR", str),
            ("SELECT [1, 2, 3]::JSON AS json_col", "json_col", "JSON", str),
            ("SELECT union_value(tag := 1) AS union_col", "union_col", "UNION(tag INTEGER)", int),
        ],
    )
    def test_description(self, query, column_name, string_type, real_type, duckdb_cursor, timestamps, integers):
        duckdb_cursor.execute(query)
        assert duckdb_cursor.description == [(column_name, string_type, None, None, None, None, None)]
        assert isinstance(duckdb_cursor.fetchone()[0], real_type)

    def test_description_comparisons(self):
        duckdb.execute("select 42 a, 'test' b, true c")
        types = [x[1] for x in duckdb.description()]

        STRING = duckdb.STRING
        NUMBER = duckdb.NUMBER
        DATETIME = duckdb.DATETIME

        assert types[1] == STRING
        assert STRING == types[1]  # noqa: SIM300
        assert types[0] != STRING
        assert types[1] == STRING
        assert STRING == types[1]  # noqa: SIM300

        assert types[1] in [STRING]
        assert types[1] in [STRING, NUMBER]
        assert types[1] not in [NUMBER, DATETIME]

    def test_none_description(self, duckdb_empty_cursor):
        assert duckdb_empty_cursor.description is None


class TestCursorRowcount:
    def test_rowcount(self, duckdb_cursor):
        assert duckdb_cursor.rowcount == -1

    def test_rowcount_no_query_yet(self, duckdb_cursor):
        assert duckdb_cursor.rowcount == -1

    def test_rowcount_insert(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")
        duckdb_cursor.execute("INSERT INTO t VALUES (1), (2), (3)")
        assert duckdb_cursor.rowcount == 3

    def test_rowcount_insert_select(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")
        duckdb_cursor.execute("INSERT INTO t SELECT * FROM range(100)")
        assert duckdb_cursor.rowcount == 100

    def test_rowcount_update(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t AS SELECT * FROM range(10) t(i)")
        duckdb_cursor.execute("UPDATE t SET i = i + 1 WHERE i < 4")
        assert duckdb_cursor.rowcount == 4

    def test_rowcount_delete(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t AS SELECT * FROM range(10) t(i)")
        duckdb_cursor.execute("DELETE FROM t WHERE i < 3")
        assert duckdb_cursor.rowcount == 3

    def test_rowcount_create_table_as(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t AS SELECT * FROM range(42) t(i)")
        assert duckdb_cursor.rowcount == 42

    def test_rowcount_select_is_unknown(self, duckdb_cursor):
        # Matches DB-API 2.0: rowcount is -1 when it can't be determined without consuming the result.
        duckdb_cursor.execute("SELECT * FROM range(10)")
        assert duckdb_cursor.rowcount == -1
        duckdb_cursor.fetchall()
        assert duckdb_cursor.rowcount == -1

    def test_rowcount_reset_by_next_execute(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")
        duckdb_cursor.execute("INSERT INTO t VALUES (1), (2)")
        assert duckdb_cursor.rowcount == 2
        duckdb_cursor.execute("SELECT * FROM t")
        assert duckdb_cursor.rowcount == -1

    def test_rowcount_does_not_disturb_fetch(self, duckdb_cursor):
        # Accessing rowcount must not consume the single-row result that fetchone() also reads.
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")
        duckdb_cursor.execute("INSERT INTO t VALUES (1), (2), (3)")
        assert duckdb_cursor.rowcount == 3
        assert duckdb_cursor.fetchone() == (3,)

    def test_rowcount_ddl_is_unknown(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")
        assert duckdb_cursor.rowcount == -1

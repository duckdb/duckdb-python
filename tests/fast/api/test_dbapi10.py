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

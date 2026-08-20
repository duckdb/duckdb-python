import datetime

import pytest

import duckdb


class TestAppender:
    def test_create_append_flush_close(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE people (id INTEGER, name VARCHAR)")
        appender = duckdb_cursor.appender("people")
        appender.append(1, "Mark")
        appender.append(2, "Hannes")
        appender.flush()
        assert duckdb_cursor.execute("SELECT * FROM people ORDER BY id").fetchall() == [(1, "Mark"), (2, "Hannes")]
        appender.close()

    def test_close_flushes(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")
        appender = duckdb_cursor.appender("t")
        appender.append(42)
        appender.close()
        assert duckdb_cursor.execute("SELECT i FROM t").fetchall() == [(42,)]

    def test_append_after_close_raises(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")
        appender = duckdb_cursor.appender("t")
        appender.close()
        with pytest.raises(duckdb.Error):
            appender.append(1)

    def test_schema_and_catalog(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE SCHEMA s")
        duckdb_cursor.execute("CREATE TABLE s.t (i INTEGER, b BOOLEAN, d DOUBLE)")
        appender = duckdb_cursor.appender("t", schema="s")
        appender.append(7, True, 1.5)
        appender.append(None, False, None)
        appender.close()
        assert duckdb_cursor.execute("SELECT * FROM s.t ORDER BY i NULLS LAST").fetchall() == [
            (7, True, 1.5),
            (None, False, None),
        ]

    def test_scalar_types(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (b BOOLEAN, i INTEGER, l BIGINT, f FLOAT, d DOUBLE, s VARCHAR, dt DATE)")
        appender = duckdb_cursor.appender("t")
        appender.append(True, 1, 2, 1.25, 2.5, "x", datetime.date(2026, 8, 16))
        appender.close()
        row = duckdb_cursor.execute("SELECT * FROM t").fetchone()
        assert row[0] is True
        assert row[1] == 1
        assert row[2] == 2
        assert row[5] == "x"
        assert row[6] == datetime.date(2026, 8, 16)

    def test_wrong_column_count(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER, j INTEGER)")
        appender = duckdb_cursor.appender("t")
        with pytest.raises(duckdb.Error):
            appender.append(1)
        appender.close()

    def test_conversion_error_does_not_drop_prior_rows(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER, j INTEGER)")
        appender = duckdb_cursor.appender("t")
        appender.append(1, 2)
        with pytest.raises(duckdb.Error):
            appender.append(3, "nope")
        appender.append(4, 5)
        appender.close()
        assert duckdb_cursor.execute("SELECT * FROM t ORDER BY i").fetchall() == [(1, 2), (4, 5)]

    def test_missing_table(self, duckdb_cursor):
        with pytest.raises(duckdb.Error):
            duckdb_cursor.appender("no_such_table")

    def test_context_manager(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")
        with duckdb_cursor.appender("t") as appender:
            appender.append(1)
            appender.append(2)
        assert duckdb_cursor.execute("SELECT * FROM t ORDER BY i").fetchall() == [(1,), (2,)]

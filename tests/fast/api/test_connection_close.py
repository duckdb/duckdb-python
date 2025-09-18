# cursor description
import tempfile

import pytest

import duckdb


def check_exception(f):
    had_exception = False
    try:
        f()
    except BaseException:
        had_exception = True
    assert had_exception


class TestConnectionClose:
    def test_connection_close(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name
        con = duckdb.connect(db)
        cursor = con.cursor()
        cursor.execute("create table a (i integer)")
        cursor.execute("insert into a values (42)")
        con.close()
        check_exception(lambda: cursor.execute("select * from a"))

    def test_open_and_exit(self):
        with pytest.raises(TypeError), duckdb.connect():
            # This exception does not get swallowed by DuckDBPyConnection's __exit__
            raise TypeError()

    def test_reopen_connection(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name
        con = duckdb.connect(db)
        cursor = con.cursor()
        cursor.execute("create table a (i integer)")
        cursor.execute("insert into a values (42)")
        con.close()
        con = duckdb.connect(db)
        cursor = con.cursor()
        results = cursor.execute("select * from a").fetchall()
        assert results == [(42,)]

    def test_get_closed_default_conn(self, duckdb_cursor):
        con = duckdb.connect()
        duckdb.set_default_connection(con)
        duckdb.close()

        # 'duckdb.close()' closes this connection, because we explicitly set it as the default
        with pytest.raises(duckdb.ConnectionException, match="Connection Error: Connection already closed"):
            con.sql("select 42").fetchall()

        default_con = duckdb.default_connection()
        default_con.sql("select 42").fetchall()
        default_con.close()

        # This does not error because the closed connection is silently replaced
        duckdb.sql("select 42").fetchall()

        # Show that the 'default_con' is still closed
        with pytest.raises(duckdb.ConnectionException, match="Connection Error: Connection already closed"):
            default_con.sql("select 42").fetchall()

        duckdb.close()

        # This also does not error because we silently receive a new connection
        con2 = duckdb.connect(":default:")
        con2.sql("select 42").fetchall()

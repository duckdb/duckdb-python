import pytest
import duckdb


class TestContextManager:
    def test_context_manager(self):
        with duckdb.connect(database=":memory:", read_only=False) as con:
            assert con.execute("select 1").fetchall() == [(1,)]
        with pytest.raises(duckdb.ConnectionException, match="Connection already closed"):
            con.execute("select 1")

    def test_existing_connection_context_manager(self):
        con = duckdb.connect()
        with con as c:
            assert c is con
            assert c.execute("select 42").fetchall() == [(42,)]
        with pytest.raises(duckdb.ConnectionException, match="Connection already closed"):
            con.execute("select 42")

    def test_cursor_context_manager(self):
        con = duckdb.connect()
        with con.cursor() as cur:
            assert cur.execute("select 'hello'").fetchall() == [("hello",)]
        with pytest.raises(duckdb.ConnectionException, match="Connection already closed"):
            cur.execute("select 'hello'")
        # Parent connection should still be open
        assert con.execute("select 'world'").fetchall() == [("world",)]
        con.close()

    def test_nested_context_managers(self):
        with duckdb.connect() as con:
            assert con.execute("select 1").fetchall() == [(1,)]
            with con.cursor() as cur:
                assert cur.execute("select 2").fetchall() == [(2,)]
            with pytest.raises(duckdb.ConnectionException, match="Connection already closed"):
                cur.execute("select 2")
            # con is still valid
            assert con.execute("select 3").fetchall() == [(3,)]
        with pytest.raises(duckdb.ConnectionException, match="Connection already closed"):
            con.execute("select 3")

    def test_exception_propagation_and_cleanup(self):
        con_ref = None
        with pytest.raises(ValueError, match="test error"):
            with duckdb.connect() as con:
                con_ref = con
                con.execute("select 1")
                raise ValueError("test error")

        assert con_ref is not None
        with pytest.raises(duckdb.ConnectionException, match="Connection already closed"):
            con_ref.execute("select 1")

    def test_pure_python_context_manager_binding(self):
        con = duckdb.connect()
        try:
            assert con.__enter__.__module__ == "duckdb"
            assert con.__exit__.__module__ == "duckdb"
        finally:
            con.close()


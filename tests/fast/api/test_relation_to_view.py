import pytest

import duckdb


class TestRelationToView:
    def test_values_to_view(self, duckdb_cursor):
        rel = duckdb_cursor.values(["test", "this is a long string"])
        res = rel.fetchall()
        assert res == [("test", "this is a long string")]

        with pytest.raises(duckdb.InvalidInputException, match=r"register\("):
            rel.to_view("vw1")

    def test_relation_to_view(self, duckdb_cursor):
        rel = duckdb_cursor.sql("select 'test', 'this is a long string'")

        res = rel.fetchall()
        assert res == [("test", "this is a long string")]

        rel.to_view("vw1")

        view = duckdb_cursor.table("vw1")
        res = view.fetchall()
        assert res == [("test", "this is a long string")]

    def test_registered_relation(self, duckdb_cursor):
        rel = duckdb_cursor.sql("select 'test', 'this is a long string'")

        con = duckdb.connect()
        # Two in-memory connections are two databases, so a relation of one cannot be read on the other.
        with pytest.raises(duckdb.InvalidInputException, match=r"(?s)in-memory database.*another database"):
            con.register("cross_connection", rel)

        # Register on the same connection just creates a view
        duckdb_cursor.register("same_connection", rel)
        view = duckdb_cursor.table("same_connection")
        res = view.fetchall()
        assert res == [("test", "this is a long string")]

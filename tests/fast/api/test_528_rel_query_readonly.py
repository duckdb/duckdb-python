"""Regression test for temp views in relations.

`DuckDBPyRelation.query(view_name, sql)` internally calls CreateView, which
writes to the default catalog. On a read-only attached database the write
fails with InvalidInputException, breaking any user pattern that uses
rel.query() against a read-only database.

`rel.select()` and `conn.sql()` don't create a view and work fine, only
rel.query() trips the bug.
"""

from __future__ import annotations

import duckdb


def test_rel_query_on_readonly_database(tmp_path):
    db_path = tmp_path / "readonly.duckdb"

    # Step 1: create the database with test data using a writable connection
    with duckdb.connect(str(db_path)) as setup_conn:
        setup_conn.execute(
            """
            CREATE TABLE orders AS
            SELECT * FROM (
                VALUES (1, 'A', 100), (2, 'B', 250), (3, 'C', 50)
            ) AS t(order_id, product, quantity)
            """
        )

    # Step 2: reopen read-only and exercise rel.query()
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rel = conn.sql("SELECT * FROM orders")
        result = rel.query(
            "duckdb_settings()",
            "SELECT value FROM duckdb_settings() WHERE name = 'TimeZone'",
        ).fetchall()[0]
        assert result is not None
        assert isinstance(result[0], str)  # value column is a string
    finally:
        conn.close()


def test_rel_select_on_readonly_database_still_works(tmp_path):
    """Sanity: rel.select() (which doesn't create a view) must continue to work."""
    db_path = tmp_path / "readonly.duckdb"
    with duckdb.connect(str(db_path)) as setup_conn:
        setup_conn.execute("CREATE TABLE t AS SELECT 1 AS x")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rel = conn.sql("SELECT * FROM t")
        result = rel.select(
            duckdb.FunctionExpression("current_setting", duckdb.ConstantExpression("TimeZone"))
        ).fetchall()[0]
        assert result is not None
        assert isinstance(result[0], str)
    finally:
        conn.close()


def test_conn_sql_on_readonly_database_still_works(tmp_path):
    """Sanity: conn.sql() (no view created) must continue to work."""
    db_path = tmp_path / "readonly.duckdb"
    with duckdb.connect(str(db_path)) as setup_conn:
        setup_conn.execute("CREATE TABLE t AS SELECT 1 AS x")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        result = conn.sql("SELECT value FROM duckdb_settings() WHERE name = 'TimeZone'").fetchall()[0]
        assert result is not None
        assert isinstance(result[0], str)
    finally:
        conn.close()

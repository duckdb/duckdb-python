from __future__ import annotations

import pytest

import duckdb
from duckdb.template import param, template


def test_connection_sql_accepts_sql_template() -> None:
    conn = duckdb.connect()
    query = template("SELECT i FROM range(5) t(i) WHERE i >= ", 2, " ORDER BY i")
    assert conn.sql(query).fetchall() == [(2,), (3,), (4,)]


def test_connection_query_accepts_sql_template() -> None:
    conn = duckdb.connect()
    query = template("SELECT i FROM range(3) t(i) WHERE i < ", 2, " ORDER BY i")
    assert conn.query(query).fetchall() == [(0,), (1,)]


def test_connection_from_query_accepts_sql_template() -> None:
    conn = duckdb.connect()
    query = template("SELECT i FROM range(4) t(i) WHERE i % ", 2, " = 0 ORDER BY i")
    assert conn.from_query(query).fetchall() == [(0,), (2,)]


def test_connection_execute_accepts_sql_template() -> None:
    conn = duckdb.connect()
    query = template("SELECT ", 42)
    assert conn.execute(query).fetchone() == (42,)


def test_module_level_sql_apis_accept_sql_template() -> None:
    conn = duckdb.connect()
    query = template("SELECT i FROM range(5) t(i) WHERE i BETWEEN ", 1, " AND ", 3, " ORDER BY i")

    assert duckdb.sql(query, connection=conn).fetchall() == [(1,), (2,), (3,)]
    assert duckdb.query(query, connection=conn).fetchall() == [(1,), (2,), (3,)]
    assert duckdb.from_query(query, connection=conn).fetchall() == [(1,), (2,), (3,)]


def test_module_level_execute_accepts_sql_template() -> None:
    conn = duckdb.connect()
    query = template("SELECT ", 5)
    assert duckdb.execute(query, connection=conn).fetchone() == (5,)


def test_connection_sql_accepts_alias_kwarg_with_template() -> None:
    conn = duckdb.connect()
    inner = conn.sql(template("SELECT 42 AS x"), alias="my_alias")
    assert inner.alias == "my_alias"
    outer = conn.sql(template("SELECT x FROM (", inner, ")"))
    assert outer.fetchall() == [(42,)]


def test_connection_sql_template_can_merge_additional_params() -> None:
    conn = duckdb.connect()
    query = template("SELECT ", 10, " + $another")
    assert conn.sql(query, params={"another": 5}).fetchall() == [(15,)]


def test_connection_sql_template_param_name_conflict_with_additional_params_raises() -> None:
    conn = duckdb.connect()
    query = template("SELECT ", param(10, "num", exact=True), " + $num")
    with pytest.raises((duckdb.InvalidInputException, ValueError)):
        conn.sql(query, params={"num": 5}).fetchall()


def test_cant_merge_with_positional_params() -> None:
    conn = duckdb.connect()
    # It doesn't even have a name, but still should error
    query = template("SELECT ", 10, " + ?")
    with pytest.raises(ValueError, match="Cannot merge compiled SQL named parameters with positional parameters"):
        conn.sql(query, params=[5]).fetchall()


def test_sql_apis_accept_compiled_sql() -> None:
    conn = duckdb.connect()
    compiled = template("SELECT i FROM range(5) t(i) WHERE i >= ", 3, " ORDER BY i").compile()

    assert conn.sql(compiled).fetchall() == [(3,), (4,)]
    assert conn.query(compiled).fetchall() == [(3,), (4,)]
    assert conn.from_query(compiled).fetchall() == [(3,), (4,)]
    assert conn.execute(compiled).fetchall() == [(3,), (4,)]


def test_relation_interpolation_works_end_to_end() -> None:
    conn = duckdb.connect()
    rel = conn.sql("SELECT i FROM range(6) t(i)")
    query = template("SELECT i FROM (", rel, ") WHERE i % ", 2, " = 0 ORDER BY i")
    assert conn.sql(query).fetchall() == [(0,), (2,), (4,)]


def test_builtin_duckdbpytype_object_interpolates_in_template() -> None:
    conn = duckdb.connect()
    integer_type = duckdb.sqltype("INTEGER")
    query = template("SELECT 42::", integer_type)
    assert conn.sql(query).fetchall() == [(42,)]


def test_builtin_expression_object_interpolates_in_template() -> None:
    conn = duckdb.connect()
    expr = duckdb.ColumnExpression("i")
    query = template("SELECT ", expr, " FROM range(3) t(i) ORDER BY i")
    assert conn.sql(query).fetchall() == [(0,), (1,), (2,)]


def test_builtin_sqlexpression_object_interpolates_in_template() -> None:
    conn = duckdb.connect()
    expr = duckdb.SQLExpression("i + 1")
    query = template("SELECT ", expr, " FROM range(3) t(i) ORDER BY i")
    assert conn.sql(query).fetchall() == [(1,), (2,), (3,)]

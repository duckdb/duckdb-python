"""An aggregate call takes an ORDER BY inside its parentheses, so `first(x ORDER BY t)` needs no SQL text."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb._expression import col, count_all, fn, lit, row_number

if TYPE_CHECKING:
    from duckdb._expression import Expr


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES "
        "(1, 3, 'a'), (1, 1, 'b'), (1, 2, NULL), (1, NULL, 'd'), (2, 5, 'c'), (2, 5, 'c')"
        ") v(g, t, x)"
    )
    yield connection
    connection.close()


def grouped(con: duckdb.DuckDBPyConnection, expression: Expr, where: str = "") -> list[tuple]:
    return con.execute(f"SELECT g, {expression.fragment()} FROM t {where} GROUP BY g ORDER BY g").fetchall()


def answers(con: duckdb.DuckDBPyConnection, sql: str, where: str = "") -> list[tuple]:
    return con.execute(f"SELECT g, {sql} FROM t {where} GROUP BY g ORDER BY g").fetchall()


class TestRendering:
    def test_renders_inside_the_call(self) -> None:
        assert col("x").first().order_by(col("t")).fragment() == '"first"("x" ORDER BY "t")'

    def test_renders_with_distinct(self) -> None:
        assert col("x").n_unique().order_by(col("x")).fragment() == 'count(DISTINCT "x" ORDER BY "x")'

    def test_renders_before_filter(self) -> None:
        call = col("x").first().order_by(col("t")).where(col("t") > 1)
        assert call.fragment() == '"first"("x" ORDER BY "t") FILTER (WHERE ("t" > 1))'
        reversed_order = col("x").first().where(col("t") > 1).order_by(col("t"))
        assert reversed_order.fragment() == call.fragment()

    def test_renders_before_ignore_nulls(self) -> None:
        call = fn("first_value", col("x")).order_by(col("t")).ignore_nulls()
        assert call.fragment() == 'first_value("x" ORDER BY "t" IGNORE NULLS)'
        assert fn("first_value", col("x")).ignore_nulls().order_by(col("t")).fragment() == call.fragment()

    def test_direction_and_null_order(self) -> None:
        call = col("x").last().order_by(col("t").desc().nulls_last())
        assert call.fragment() == '"last"("x" ORDER BY "t" DESC NULLS LAST)'

    def test_several_keys(self) -> None:
        call = col("x").string_agg(lit(",")).order_by(col("t").desc(), col("x").nulls_first())
        assert call.fragment() == """string_agg("x", ',' ORDER BY "t" DESC, "x" NULLS FIRST)"""

    def test_a_later_order_by_replaces_the_keys(self) -> None:
        call = col("x").first().order_by(col("t")).order_by(col("x").desc())
        assert call.fragment() == '"first"("x" ORDER BY "x" DESC)'
        assert col("x").first(order_by=col("t")).order_by(col("x")).fragment() == '"first"("x" ORDER BY "x")'

    def test_keyword_on_a_method(self) -> None:
        assert col("x").first(order_by=col("t")).fragment() == col("x").first().order_by(col("t")).fragment()
        two = col("x").first(order_by=[col("t"), col("x")])
        assert two.fragment() == col("x").first().order_by(col("t"), col("x")).fragment()
        assert col("x").first(order_by=None).fragment() == '"first"("x")'

    def test_keyword_beside_positional_arguments(self) -> None:
        joined = col("x").string_agg(lit("|"), order_by=col("t").desc())
        assert joined.fragment() == """string_agg("x", '|' ORDER BY "t" DESC)"""
        quantile = col("t").quantile(0.5, order_by=col("t"))
        assert quantile.fragment() == 'quantile_disc("t", 0.5::DOUBLE ORDER BY "t")'

    def test_fn_reaches_it_through_the_method(self) -> None:
        assert fn("first", col("x")).order_by(col("t")).fragment() == '"first"("x" ORDER BY "t")'
        # A keyword argument of fn() is a named SQL argument, whatever its name.
        assert fn("first", col("x"), order_by=col("t")).fragment() == '"first"("x", "order_by" := "t")'

    def test_through_a_family_wrapper(self) -> None:
        wrapped = col("x").first().str().order_by(col("t"))
        assert wrapped.fragment() == '"first"("x" ORDER BY "t")'
        assert wrapped.lower().fragment() == 'lower("first"("x" ORDER BY "t"))'

    def test_a_family_method_takes_no_keyword(self) -> None:
        # `.list().sum()` is the scalar `list_sum`, so it has no rows to order.
        with pytest.raises(TypeError, match="unexpected keyword argument 'order_by'"):
            col("xs").list().sum(order_by=col("t"))  # type: ignore[call-arg]

    def test_the_original_is_unchanged(self) -> None:
        plain = col("x").first()
        ordered = plain.order_by(col("t"))
        assert plain.fragment() == '"first"("x")'
        assert ordered.fragment() == '"first"("x" ORDER BY "t")'


class TestRefusals:
    def test_a_string_key_is_refused(self) -> None:
        with pytest.raises(TypeError, match="order_by takes an expression, not the string 't'"):
            col("x").first().order_by("t")
        with pytest.raises(TypeError, match="order_by takes an expression, not the string 't'"):
            col("x").first(order_by="t")
        with pytest.raises(TypeError, match="order_by takes an expression, not the string 't'"):
            col("x").first(order_by=[col("g"), "t"])

    def test_refused_on_a_non_aggregate(self) -> None:
        with pytest.raises(TypeError, match="order_by\\(\\) applies to an aggregate call"):
            col("x").order_by(col("t"))
        with pytest.raises(TypeError, match="order_by\\(\\) applies to an aggregate call"):
            (col("x") + 1).order_by(col("t"))

    def test_refused_after_over(self) -> None:
        with pytest.raises(TypeError, match="order_by\\(\\) applies to an aggregate call"):
            col("x").first().over(partition_by=col("g")).order_by(col("t"))

    def test_no_keys_is_refused(self) -> None:
        with pytest.raises(TypeError, match="at least one key"):
            col("x").first().order_by()
        with pytest.raises(TypeError, match="at least one key"):
            col("x").first(order_by=[])

    def test_what_only_the_binder_knows_fails_at_run_time(self, con: duckdb.DuckDBPyConnection) -> None:
        """A scalar call, a window-only function, a DISTINCT key outside the arguments and a star all build."""
        scalar = fn("upper", col("x")).order_by(col("t"))
        assert scalar.fragment() == 'upper("x" ORDER BY "t")'
        with pytest.raises(duckdb.InvalidInputException, match="Scalar Function"):
            con.execute(f"SELECT {scalar.fragment()} FROM t")
        with pytest.raises(duckdb.BinderException):
            con.execute(f"SELECT {row_number().order_by(col('t')).fragment()} FROM t")
        with pytest.raises(duckdb.BinderException, match="DISTINCT aggregate"):
            con.execute(f"SELECT {col('x').n_unique().order_by(col('t')).fragment()} FROM t")
        with pytest.raises(duckdb.BinderException, match="STAR"):
            con.execute(f"SELECT {count_all().order_by(col('t')).fragment()} FROM t")


class TestAnswers:
    """Each ordered call answers what the same SQL answers, on groups with NULLs in the key and in the value."""

    def test_first_and_last(self, con: duckdb.DuckDBPyConnection) -> None:
        assert grouped(con, col("x").first().order_by(col("t"))) == [(1, "b"), (2, "c")]
        assert grouped(con, col("x").last().order_by(col("t"))) == [(1, "d"), (2, "c")]
        assert grouped(con, col("x").first().order_by(col("t").nulls_first())) == [(1, "d"), (2, "c")]
        # The engine puts NULLs last in either direction unless told otherwise.
        assert grouped(con, col("x").first().order_by(col("t").desc())) == [(1, "a"), (2, "c")]
        assert grouped(con, col("x").first().order_by(col("t").desc().nulls_first())) == [(1, "d"), (2, "c")]

    def test_a_null_value_is_a_value(self, con: duckdb.DuckDBPyConnection) -> None:
        picked = col("x").first().order_by(col("t")).where(col("t") > 1)
        assert grouped(con, picked) == answers(con, "first(x ORDER BY t) FILTER (WHERE t > 1)")
        assert grouped(con, picked) == [(1, None), (2, "c")]

    def test_a_second_key_breaks_a_tie(self, con: duckdb.DuckDBPyConnection) -> None:
        # In group 1 three rows tie on `x IS NULL`; the second key decides which of them is first.
        lowest = col("t").first().order_by(col("x").is_null(), col("t"))
        highest = col("t").first().order_by(col("x").is_null(), col("t").desc())
        assert grouped(con, lowest) == answers(con, "first(t ORDER BY x IS NULL, t)") == [(1, 1), (2, 5)]
        assert grouped(con, highest) == answers(con, "first(t ORDER BY x IS NULL, t DESC)") == [(1, 3), (2, 5)]

    def test_string_agg_and_array_agg(self, con: duckdb.DuckDBPyConnection) -> None:
        joined = col("x").string_agg(lit(",")).order_by(col("t"))
        assert grouped(con, joined) == [(1, "b,a,d"), (2, "c,c")]
        listed = col("x").array_agg().order_by(col("t").desc())
        assert grouped(con, listed) == [(1, ["a", None, "b", "d"]), (2, ["c", "c"])]

    def test_with_distinct(self, con: duckdb.DuckDBPyConnection) -> None:
        ordered = col("x").n_unique().order_by(col("x"))
        assert grouped(con, ordered) == answers(con, "count(DISTINCT x ORDER BY x)")
        assert grouped(con, ordered) == [(1, 3), (2, 1)]

    def test_as_a_window(self, con: duckdb.DuckDBPyConnection) -> None:
        windowed = col("x").first().order_by(col("t")).over(partition_by=col("g"))
        assert windowed.fragment() == '"first"("x" ORDER BY "t") OVER (PARTITION BY "g")'
        rows = con.execute(f"SELECT {windowed.fragment()} FROM t ORDER BY g, t").fetchall()
        assert [r[0] for r in rows] == ["b", "b", "b", "b", "c", "c"]

    def test_as_a_window_ignoring_nulls(self, con: duckdb.DuckDBPyConnection) -> None:
        keys = col("x").nulls_first()
        with_nulls = fn("first_value", col("x")).order_by(keys).over(partition_by=col("g"))
        without = fn("first_value", col("x")).order_by(keys).ignore_nulls().over(partition_by=col("g"))
        source = " FROM t ORDER BY g, t"
        assert [r[0] for r in con.execute(f"SELECT {with_nulls.fragment()}{source}").fetchall()][:4] == [None] * 4
        assert [r[0] for r in con.execute(f"SELECT {without.fragment()}{source}").fetchall()] == ["a"] * 4 + ["c"] * 2

    def test_over_an_empty_group(self, con: duckdb.DuckDBPyConnection) -> None:
        assert grouped(con, col("x").first().order_by(col("t")), "WHERE false") == []
        empty = con.execute(f"SELECT {col('x').first().order_by(col('t')).fragment()} FROM t WHERE false").fetchall()
        assert empty == [(None,)]

"""Expressions turn into SQL text, and the values in them are written into that text."""

from __future__ import annotations

import concurrent.futures
import datetime
import decimal
import math
import re
import subprocess
import sys
import types
import uuid
from fractions import Fraction

import pytest

import duckdb
from duckdb._expression import (
    Expr,
    coalesce,
    col,
    fn,
    lit,
    plain_value,
    quote,
    render_literal,
    row_number,
    sql_expr,
    sql_type_of,
    star,
    when,
)

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


def _converts(value: object) -> bool:
    try:
        value.tolist()  # type: ignore[attr-defined]
    except OverflowError:
        return False
    return True


class NoOffset(datetime.tzinfo):
    """A time zone that gives no offset, which makes a datetime carrying it naive, as Python defines it."""

    def utcoffset(self, moment: datetime.datetime | None) -> None:
        return None

    def dst(self, moment: datetime.datetime | None) -> None:
        return None

    def tzname(self, moment: datetime.datetime | None) -> str:
        return "none"


class Emptying(datetime.tzinfo):
    """UTC, emptying a list each time it gives an offset: a callback that changes a value while it converts."""

    def __init__(self, container: list[object]) -> None:
        self.container = container

    def utcoffset(self, moment: datetime.datetime | None) -> datetime.timedelta:
        self.container.clear()
        return datetime.timedelta(0)

    def dst(self, moment: datetime.datetime | None) -> None:
        return None

    def tzname(self, moment: datetime.datetime | None) -> str:
        return "emptying"


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    yield connection
    connection.close()


def evaluate(con: duckdb.DuckDBPyConnection, expression: Expr, source: str = "") -> object:
    """Run an expression and return its single value."""
    return con.execute(f"SELECT {expression.fragment()}{source}").fetchall()[0][0]


class TestQuoting:
    def test_identifiers_are_always_quoted(self) -> None:
        assert col("x").fragment() == '"x"'

    def test_embedded_quotes_are_escaped(self) -> None:
        # Without escaping this would end the identifier and inject syntax.
        assert quote('a"b') == '"a""b"'

    def test_a_reserved_word_is_usable_as_a_column(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute('CREATE TABLE t ("select" INTEGER)')
        con.execute("INSERT INTO t VALUES (1)")
        assert evaluate(con, col("select"), " FROM t") == 1


class TestStringOperandsAreLiterals:
    """A break from the previous duckdb package, where `col("state") == "active"` meant a column named active."""

    # Each test reads its own row, so neither depends on the other having run on the module's connection.
    STATES = " FROM (VALUES ('active', 'no')) AS s(state, active)"

    def test_string_compares_as_a_value(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, col("state") == "active", self.STATES) is True

    def test_a_column_operand_is_explicit(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, col("state") == col("active"), self.STATES) is False


class TestNoneAndUnknownOperands:
    def test_equality_with_none_is_a_null_comparison(self, con: duckdb.DuckDBPyConnection) -> None:
        # Comparing with NULL is never true in SQL, and surfacing that beats inventing something friendlier.
        assert evaluate(con, col("v") == None, " FROM (SELECT 1 AS v)") is None  # noqa: E711

    @pytest.mark.parametrize(
        "operand",
        [object(), Fraction(1, 2), np.array([1, 2]), np.longdouble(1)],
        ids=["object", "fraction", "array", "longdouble"],
    )
    def test_equality_with_an_unconvertible_object_raises(self, operand: object) -> None:
        # Python's fallback would answer by identity, and that False combined into a condition reads as SQL's FALSE.
        with pytest.raises(TypeError, match="as an expression operand"):
            col("x") == operand  # noqa: B015
        with pytest.raises(TypeError, match="as an expression operand"):
            col("x") != operand  # noqa: B015
        with pytest.raises(TypeError, match="as an expression operand"):
            operand == col("x")  # noqa: B015
        with pytest.raises(TypeError, match="as an expression operand"):
            operand != col("x")  # noqa: B015

    def test_arithmetic_with_an_unconvertible_object_raises(self) -> None:
        with pytest.raises(TypeError):
            col("x") + object()


class TestValuesAreWrittenIntoTheText:
    """A value in an expression is written into its SQL text as a literal."""

    def test_a_string_is_written_in_as_a_quoted_literal(self) -> None:
        assert (col("name") == "ann").fragment() == "(\"name\" = 'ann')"


class TestSqlTypeOf:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, "BOOLEAN"),
            (1, "INTEGER"),
            (2**40, "BIGINT"),
            (2**70, "HUGEINT"),
            (1.5, "DOUBLE"),
            ("x", "VARCHAR"),
            (b"x", "BLOB"),
            ([1, 2], "INTEGER[]"),
            ([1, 1.5], "DOUBLE[]"),
        ],
    )
    def test_types(self, value: object, expected: str) -> None:
        assert sql_type_of(value) == expected

    def test_ambiguous_values_have_no_type(self) -> None:
        # Without an element type there is nothing honest to claim, so the value is written in instead.
        assert sql_type_of([]) is None
        assert sql_type_of({1: "a", "x": "b"}) is None

    def test_a_mixed_list_takes_the_parameter_path_s_type(self) -> None:
        assert sql_type_of(["a", 1]) == "VARCHAR[]"


class TestOperatorsEvaluate:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            (lit(2) + 3, 5),
            (lit(10) - 4, 6),
            (lit(3) * 4, 12),
            (lit(7) / 2, 3.5),
            (lit(7) // 2, 3),
            (lit(7) % 3, 1),
            (lit(2) ** 10, 1024),
            (-lit(5), -5),
            (~lit(True), False),
            ((lit(True) & lit(False)), False),
            ((lit(True) | lit(False)), True),
            (3 + lit(4), 7),
            (10 - lit(4), 6),
            (7 // lit(2), 3),
            (7 % lit(3), 1),
            (2 ** lit(3), 8),
            (True & lit(False), False),
            (False | lit(True), True),
        ],
    )
    def test_operator(self, con: duckdb.DuckDBPyConnection, expression: Expr, expected: object) -> None:
        assert evaluate(con, expression) == expected

    def test_precedence_is_preserved(self, con: duckdb.DuckDBPyConnection) -> None:
        # Every part parenthesises itself, so Python's precedence survives into the SQL.
        assert evaluate(con, (lit(2) + lit(3)) * lit(4)) == 20
        assert evaluate(con, lit(2) + lit(3) * lit(4)) == 14


class TestPredicatesAndFunctions:
    def test_is_null(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, lit(None).is_null()) is True
        assert evaluate(con, lit(1).is_not_null()) is True

    def test_isin(self, con: duckdb.DuckDBPyConnection) -> None:
        # One argument is one value, so the members are written out; a list argument tests the list itself.
        assert evaluate(con, lit(2).isin(1, 2, 3)) is True
        assert evaluate(con, lit(9).isin(1, 2, 3)) is False
        assert evaluate(con, lit([1, 2, 3]).isin([1, 2, 3])) is True
        assert evaluate(con, lit([1, 2]).isin([1, 2, 3])) is False

    def test_isin_of_a_list_against_a_scalar_is_a_comparison(self, con: duckdb.DuckDBPyConnection) -> None:
        with pytest.raises(duckdb.ConversionException):
            evaluate(con, lit(2).isin([1, 2, 3]))

    def test_isin_of_an_empty_list_is_a_comparison(self, con: duckdb.DuckDBPyConnection) -> None:
        # An empty list is one value, so this compares a number with a list rather than meaning FALSE.
        with pytest.raises(duckdb.ConversionException):
            evaluate(con, lit(1).isin([]))
        assert evaluate(con, lit([]).isin([])) is True

    def test_isin_takes_one_expression_as_one_member(self, con: duckdb.DuckDBPyConnection) -> None:
        # Changed: one expression is one member; list membership is list_contains.
        assert lit(2).isin(col("y")).fragment() == '(2 IN ("y"))'
        assert evaluate(con, lit(2).isin(col("y")), " FROM (SELECT 2 AS y)") is True
        assert evaluate(con, fn("list_contains", col("xs"), 2), " FROM (SELECT [1, 2] AS xs)") is True

    def test_isin_takes_one_string_as_one_member(self, con: duckdb.DuckDBPyConnection) -> None:
        # Changed: one text is one member, never its characters.
        assert col("x").isin("abc").fragment() == "(\"x\" IN ('abc'))"
        assert evaluate(con, col("x").isin("abc"), " FROM (SELECT 'abc' AS x)") is True

    def test_between(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, lit(5).between(1, 10)) is True

    def test_like(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, lit("Alpha").like("Al%")) is True
        assert evaluate(con, lit("Alpha").like("al%")) is False
        assert evaluate(con, lit("ab").like("a_")) is True

    def test_ilike_ignores_case(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, lit("Alpha").ilike("al%")) is True

    def test_like_negates_with_invert(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, ~lit("Alpha").like("b%")) is True

    def test_like_escape(self, con: duckdb.DuckDBPyConnection) -> None:
        # Without an escape the underscore is a wildcard and would match "axb".
        assert evaluate(con, lit("a_b").like("a!_b", escape="!")) is True
        assert evaluate(con, lit("axb").like("a!_b", escape="!")) is False

    def test_cast(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, lit("42").cast("INTEGER")) == 42
        assert evaluate(con, lit("42").try_cast("INTEGER")) == 42
        assert evaluate(con, lit("x").try_cast("INTEGER")) is None

    def test_coalesce(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, coalesce(lit(None), lit(None), lit(7))) == 7

    def test_fn_reaches_any_function(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, fn("upper", lit("ab"))) == "AB"

    def test_fn_passes_named_arguments(self) -> None:
        assert fn("round", col("x"), precision=1).fragment() == 'round("x", "precision" := 1)'

    def test_case(self, con: duckdb.DuckDBPyConnection) -> None:
        expression = when(lit(1) > 2).then("big").otherwise("small")
        assert evaluate(con, expression) == "small"

    def test_case_without_otherwise_is_null(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, when(lit(1) > 2).then("big").end()) is None

    def test_a_case_condition_is_an_expression(self) -> None:
        with pytest.raises(TypeError, match="name a column with col"):
            when("x")

    def test_sql_expr_escape_hatch(self, con: duckdb.DuckDBPyConnection) -> None:
        assert evaluate(con, sql_expr("1 + 1")) == 2


class TestAggregatesAndWindows:
    def test_aggregate_shortcut(self, con: duckdb.DuckDBPyConnection) -> None:
        value = con.execute(f"SELECT {col('i').sum().fragment()} FROM range(4) t(i)").fetchall()[0][0]
        assert value == 6

    def test_window(self, con: duckdb.DuckDBPyConnection) -> None:
        numbered = row_number().over(order_by=col("i"))
        rows = con.execute(f"SELECT {numbered.fragment()} FROM range(3) t(i)").fetchall()
        assert [r[0] for r in rows] == [1, 2, 3]

    def test_window_frame_and_partition(self, con: duckdb.DuckDBPyConnection) -> None:
        running = col("i").sum().over(partition_by=col("g"), order_by=col("i").desc(), rows=(-1, 0))
        assert running.fragment() == (
            'sum("i") OVER (PARTITION BY "g" ORDER BY "i" DESC ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)'
        )
        source = " FROM (SELECT i, i % 2 AS g FROM range(5) t(i)) ORDER BY i"
        rows = con.execute(f"SELECT {running.fragment()}{source}").fetchall()
        assert [r[0] for r in rows] == [2, 4, 6, 3, 4]
        with pytest.raises(TypeError, match="rows or by range"):
            col("i").sum().over(rows=(0, 1), range=(0, 1))

    def test_filtered_aggregate(self, con: duckdb.DuckDBPyConnection) -> None:
        evens = col("i").count().where(col("i") % 2 == 0)
        assert evens.fragment() == 'count("i") FILTER (WHERE (("i" % 2) = 0))'
        assert con.execute(f"SELECT {evens.fragment()} FROM range(5) t(i)").fetchall() == [(3,)]
        with pytest.raises(TypeError, match="applies to an aggregate call"):
            col("i").where(col("i") > 1)

    def test_unknown_attribute_still_raises(self) -> None:
        # A typo must stay an AttributeError, never become a callable that fails later.
        with pytest.raises(AttributeError, match="no_such_aggregate"):
            _ = col("x").no_such_aggregate


class TestPresentation:
    def test_alias(self) -> None:
        assert col("x").alias("y").as_select() == '"x" AS "y"'

    def test_ordering(self) -> None:
        assert col("x").desc().as_order() == '"x" DESC'
        assert col("x").nulls_last().as_order() == '"x" NULLS LAST'
        assert col("x").desc().nulls_first().as_order() == '"x" DESC NULLS FIRST'

    def test_expressions_are_immutable(self) -> None:
        # Aliasing returns a new expression, so one can be reused without carrying an earlier alias.
        base = col("x")
        aliased = base.alias("y")
        assert base.as_select() == '"x"'
        assert aliased.as_select() == '"x" AS "y"'
        assert base is not aliased

    def test_star(self) -> None:
        assert star().fragment() == "*"
        assert star(exclude=["a"]).fragment() == '* EXCLUDE ("a")'
        assert star(rename={"a": "b"}).fragment() == '* RENAME ("a" AS "b")'

    def test_an_expression_has_no_truth_value(self) -> None:
        with pytest.raises(TypeError, match="no truth value"):
            bool(col("x") == 1)


class TestRichLiteralTypes:
    """Every type that can be passed as a query parameter also works inside an expression."""

    def test_a_date_comparison_describes_as_a_boolean(self, con: duckdb.DuckDBPyConnection) -> None:
        # The form with values written in is the one DuckDB is shown when asked for column types.
        expression = col("d") > datetime.date(2026, 8, 1)
        described = con.execute(f"DESCRIBE SELECT {expression.fragment()} AS b FROM (SELECT DATE '2020-01-01' AS d)")
        assert described.fetchall()[0][1] == "BOOLEAN"

    def test_a_date_comparison_evaluates(self, con: duckdb.DuckDBPyConnection) -> None:
        expression = col("d") > datetime.date(2026, 1, 1)
        rows = con.execute(f"SELECT {expression.fragment()} FROM (SELECT DATE '2026-08-01' AS d)").fetchall()
        assert rows[0][0] is True

    def test_decimal_keeps_its_scale_when_written_in(self, con: duckdb.DuckDBPyConnection) -> None:
        assert render_literal(decimal.Decimal("1.50")) == "CAST('1.50' AS DECIMAL(3, 2))"
        assert evaluate(con, lit(decimal.Decimal("0.000001"))) == decimal.Decimal("0.000001")

    @pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_decimal_renders_as_the_parameter_path_types_it(
        self, con: duckdb.DuckDBPyConnection, text: str
    ) -> None:
        # The value is typed as binding it as a parameter types it, so the two paths give one type and one value.
        value = decimal.Decimal(text)
        [(bound_type, bound)] = con.execute("SELECT typeof(?), ?", [value, value]).fetchall()
        assert evaluate(con, fn("typeof", lit(value))) == bound_type
        written = evaluate(con, lit(value))
        assert (math.isnan(written) and math.isnan(bound)) or written == bound
        compared = con.execute(f"SELECT {(col('x') == value).fragment()} FROM (SELECT ?::DOUBLE AS x)", [value])
        assert (
            compared.fetchall() == con.execute("SELECT x = ? FROM (SELECT ?::DOUBLE AS x)", [value, value]).fetchall()
        )

    def test_a_null_inside_a_composite_renders_as_null_text(self) -> None:
        # Written in, the SQL binder types the NULL from its siblings, however it pairs them.
        assert render_literal([None, "x"]) == "[NULL, 'x']"
        assert render_literal({"a": None, "b": 1}) == "{'a': NULL, 'b': 1}"

    def test_a_literal_of_only_nulls_is_written_in_for_the_binder_to_type(self, con: duckdb.DuckDBPyConnection) -> None:
        # sql_type_of gives [None] no type, so instead of binding it bare the list is written into the SQL, where
        # the binder types the NULLs from whatever meets them.
        rendered = (col("x") == lit([None])).fragment()
        assert rendered == '("x" = [NULL])'
        bound = con.execute(f"SELECT {rendered} FROM (SELECT ['a'] AS x)").fetchall()
        assert bound == [(False,)]

    @pytest.mark.parametrize(
        ("value", "written"),
        [
            (1.5, "1.5::DOUBLE"),
            (float("nan"), "'NaN'::DOUBLE"),
            (float("inf"), "'Infinity'::DOUBLE"),
            (b"\x00\xff", "'\\x00\\xff'::BLOB"),
            (uuid.UUID(int=1), "UUID '00000000-0000-0000-0000-000000000001'"),
            ([1, 2], "[1, 2]"),
            ({1: "a"}, "MAP {1: 'a'}"),
            ("it's", "'it''s'"),
        ],
        ids=["double", "nan", "infinity", "blob", "uuid", "list", "map", "quote"],
    )
    def test_written_in(self, con: duckdb.DuckDBPyConnection, value: object, written: str) -> None:
        assert render_literal(value) == written
        assert con.execute(f"SELECT {written}").fetchall()[0][0] is not None


class TestFoundByWritingTheManual:
    """Two shortcuts that produced SQL DuckDB rejects, both caught by running the manual's examples."""

    def test_n_unique_uses_distinct_syntax(self, con: duckdb.DuckDBPyConnection) -> None:
        # It used to call a `count_distinct` function, which DuckDB does not have; DISTINCT is syntax.
        expression = col("c").n_unique()
        assert expression.fragment() == 'count(DISTINCT "c")'
        con.execute("CREATE TABLE u (c VARCHAR)")
        con.execute("INSERT INTO u VALUES ('x'), ('x'), ('y')")
        assert con.execute(f"SELECT {expression.fragment()} FROM u").fetchall()[0][0] == 2

    def test_concat_is_explicit_because_plus_is_not_concatenation(self, con: duckdb.DuckDBPyConnection) -> None:
        # An expression carries no type, so `+` cannot decide between addition and concatenation.
        expression = lit("a").concat("-", lit("b"))
        assert con.execute(f"SELECT {expression.fragment()}").fetchall()[0][0] == "a-b"

    def test_plus_on_text_still_fails_loudly(self, con: duckdb.DuckDBPyConnection) -> None:
        # Not silently reinterpreted as concatenation: DuckDB says so.
        with pytest.raises(duckdb.Error):
            con.execute(f"SELECT {(lit('a') + lit('b')).fragment()}").fetchall()


# How to call each name, so adding a shortcut without adding it here fails the test below.
AGGREGATE_CALLS: dict[str, tuple[object, ...]] = {
    "sum": (),
    "mean": (),
    "avg": (),
    "min": (),
    "max": (),
    "count": (),
    "median": (),
    "std": (),
    "var": (),
    "first": (),
    "last": (),
    "any_value": (),
    "product": (),
    "skewness": (),
    "kurtosis": (),
    "entropy": (),
    "string_agg": (lit(","),),
    "bit_and": (),
    "bit_or": (),
    "bool_and": (),
    "bool_or": (),
    "array_agg": (),
    "mode": (),
    "quantile": (0.5,),
    "quantile_cont": (0.5,),
    "arg_min": (col("v"),),
    "arg_max": (col("v"),),
}
WINDOW_CALLS: dict[str, tuple[object, ...]] = {
    "row_number": (),
    "rank": (),
    "dense_rank": (),
    "ntile": (lit(2),),
    "lag": (col("v"),),
    "lead": (col("v"),),
    "first_value": (col("v"),),
    "last_value": (col("v"),),
}


@pytest.fixture(scope="module")
def data(con: duckdb.DuckDBPyConnection) -> str:
    """A small table for the aggregates: a NULL value, a second value column, and a tie in the ordering column.

    `w` differs from `v` so `arg_min` and `arg_max` cannot answer the same, and `g` ties so `rank` and `dense_rank`
    cannot either.
    """
    con.execute("CREATE TABLE agg (v INTEGER, b BOOLEAN, s VARCHAR, w INTEGER, g INTEGER)")
    con.execute("INSERT INTO agg VALUES (1, true, 'a', 10, 1), (2, false, 'b', 20, 1)")
    con.execute("INSERT INTO agg VALUES (3, true, 'c', 30, 2), (NULL, true, 'd', 40, 2)")
    return "agg"


# The column each aggregate reads and the SQL call it must mean, so a shortcut mapped to another function answers
# differently from the function the caller asked for.
AGGREGATE_TWINS: dict[str, tuple[str, str]] = {
    "sum": ("v", "sum(v)"),
    "mean": ("v", "avg(v)"),
    "avg": ("v", "avg(v)"),
    "min": ("v", "min(v)"),
    "max": ("v", "max(v)"),
    "count": ("v", "count(v)"),
    "median": ("v", "median(v)"),
    "std": ("v", "stddev(v)"),
    "var": ("v", "var_samp(v)"),
    "first": ("v", "first(v)"),
    "last": ("v", "last(v)"),
    "any_value": ("v", "any_value(v)"),
    "product": ("v", "product(v)"),
    "skewness": ("v", "skewness(v)"),
    "kurtosis": ("v", "kurtosis(v)"),
    "entropy": ("v", "entropy(v)"),
    "string_agg": ("s", "string_agg(s, ',')"),
    "bit_and": ("v", "bit_and(v)"),
    "bit_or": ("v", "bit_or(v)"),
    "bool_and": ("b", "bool_and(b)"),
    "bool_or": ("b", "bool_or(b)"),
    "array_agg": ("v", "array_agg(v)"),
    "mode": ("v", "mode(v)"),
    # Quantile means quantile_disc, as SQL and the old relation method do; the continuous form has its own name.
    "quantile": ("v", "quantile(v, 0.5)"),
    "quantile_cont": ("v", "quantile_cont(v, 0.5)"),
    "arg_min": ("w", "arg_min(w, v)"),
    "arg_max": ("w", "arg_max(w, v)"),
}
WINDOW_TWINS: dict[str, str] = {
    "row_number": "row_number()",
    "rank": "rank()",
    "dense_rank": "dense_rank()",
    "ntile": "ntile(2)",
    "lag": "lag(v)",
    "lead": "lead(v)",
    "first_value": "first_value(v)",
    "last_value": "last_value(v)",
}
# The expression carries its own OVER clause; only the SQL twin needs one written out.
ROWS_IN_ORDER = " FROM agg ORDER BY g, v"
TWIN_WINDOW = " OVER (ORDER BY g, v)"


class TestEveryAdvertisedFunctionExecutes:
    """Checking the SQL text cannot catch a wrong function name, so every shortcut is run and its value compared."""

    def test_the_spec_covers_every_advertised_name(self) -> None:
        advertised = set(Expr._CALLABLES) | set(WINDOW_CALLS)
        specified = set(AGGREGATE_CALLS) | set(WINDOW_CALLS)
        assert not advertised - specified, f"advertised but never executed: {sorted(advertised - specified)}"

    def test_every_name_has_a_twin(self) -> None:
        assert set(AGGREGATE_TWINS) == set(AGGREGATE_CALLS)
        assert set(WINDOW_TWINS) == set(WINDOW_CALLS)

    @pytest.mark.parametrize("name", sorted(AGGREGATE_CALLS))
    def test_an_aggregate_answers_what_its_own_sql_function_answers(
        self, con: duckdb.DuckDBPyConnection, data: str, name: str
    ) -> None:
        column, twin = AGGREGATE_TWINS[name]
        expression = getattr(col(column), name)(*AGGREGATE_CALLS[name])
        answered = con.execute(f"SELECT {expression.fragment()} FROM {data}").fetchall()
        expected = con.execute(f"SELECT {twin} FROM {data}").fetchall()
        assert repr(answered) == repr(expected)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("sum", 6),
            ("count", 3),
            ("min", 1),
            ("max", 3),
            ("mean", 2.0),
            ("avg", 2.0),
            ("median", 2.0),
            ("std", 1.0),
            ("var", 1.0),
            ("product", 6.0),
            ("quantile", 2),
            ("quantile_cont", 2.0),
            ("bit_and", 0),
            ("bit_or", 3),
            ("bool_and", False),
            ("bool_or", True),
            ("string_agg", "a,b,c,d"),
            ("array_agg", [1, 2, 3, None]),
            ("arg_min", 10),
            ("arg_max", 30),
            ("skewness", 0.0),
            ("entropy", pytest.approx(1.584962500721156)),
            ("kurtosis", None),
        ],
    )
    def test_an_aggregate_answers_the_value(
        self, con: duckdb.DuckDBPyConnection, data: str, name: str, expected: object
    ) -> None:
        column, _ = AGGREGATE_TWINS[name]
        expression = getattr(col(column), name)(*AGGREGATE_CALLS[name])
        assert evaluate(con, expression, f" FROM {data}") == expected

    @pytest.mark.parametrize(("name", "expected"), [("sum", None), ("count", 0), ("array_agg", None)])
    def test_an_aggregate_over_an_empty_group(
        self, con: duckdb.DuckDBPyConnection, data: str, name: str, expected: object
    ) -> None:
        column, _ = AGGREGATE_TWINS[name]
        expression = getattr(col(column), name)(*AGGREGATE_CALLS[name])
        assert evaluate(con, expression, f" FROM {data} WHERE false") == expected

    @pytest.mark.parametrize("name", sorted(WINDOW_CALLS))
    def test_a_window_answers_what_its_own_sql_function_answers(
        self, con: duckdb.DuckDBPyConnection, data: str, name: str
    ) -> None:
        import duckdb._expression as expr_module

        expression = getattr(expr_module, name)(*WINDOW_CALLS[name]).over(order_by=[col("g"), col("v")])
        answered = con.execute(f"SELECT {expression.fragment()}{ROWS_IN_ORDER}").fetchall()
        expected = con.execute(f"SELECT {WINDOW_TWINS[name]}{TWIN_WINDOW}{ROWS_IN_ORDER}").fetchall()
        assert answered == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("row_number", [1, 2, 3, 4]),
            ("ntile", [1, 1, 2, 2]),
            ("lag", [None, 1, 2, 3]),
            ("lead", [2, 3, None, None]),
            ("first_value", [1, 1, 1, 1]),
            ("last_value", [1, 2, 3, None]),
        ],
    )
    def test_a_window_answers_the_values(
        self, con: duckdb.DuckDBPyConnection, data: str, name: str, expected: list[object]
    ) -> None:
        import duckdb._expression as expr_module

        expression = getattr(expr_module, name)(*WINDOW_CALLS[name]).over(order_by=[col("g"), col("v")])
        assert [row[0] for row in con.execute(f"SELECT {expression.fragment()}{ROWS_IN_ORDER}").fetchall()] == expected

    def test_rank_and_dense_rank_differ_on_a_tie(self, con: duckdb.DuckDBPyConnection, data: str) -> None:
        import duckdb._expression as expr_module

        ranked = expr_module.rank().over(order_by=col("g"))
        dense = expr_module.dense_rank().over(order_by=col("g"))
        assert [row[0] for row in con.execute(f"SELECT {ranked.fragment()}{ROWS_IN_ORDER}").fetchall()] == [1, 1, 3, 3]
        assert [row[0] for row in con.execute(f"SELECT {dense.fragment()}{ROWS_IN_ORDER}").fetchall()] == [1, 1, 2, 2]

    def test_n_unique(self, con: duckdb.DuckDBPyConnection, data: str) -> None:
        # Not in the shortcut table: DISTINCT is syntax, not a function name.
        assert con.execute(f"SELECT {col('v').n_unique().fragment()} FROM {data}").fetchall()[0][0] == 3

    def test_the_window_functions_are_exported(self) -> None:
        assert {name for name in WINDOW_CALLS if hasattr(duckdb, name)} == set(WINDOW_CALLS)


class TestFunctionFamilies:
    """The .str(), .dt(), .list() and .json() families call the function with the expression in its place."""

    def test_each_family_renders_and_runs(self, con: duckdb.DuckDBPyConnection) -> None:
        cases = [
            (col("s").str().upper(), 'upper("s")', "AB"),
            (col("s").str().concat_ws("-", "x"), "concat_ws('-', \"s\", 'x')", "ab-x"),
            (col("d").dt().year(), '"year"("d")', 2020),
            (col("xs").list().contains(2), 'list_contains("xs", 2)', True),
            (col("j").json().extract_string("$.a"), "json_extract_string(\"j\", '$.a')", "z"),
        ]
        source = " FROM (SELECT 'ab' AS s, DATE '2020-02-03' AS d, [1, 2] AS xs, '{\"a\": \"z\"}'::JSON AS j)"
        for expression, text, value in cases:
            assert expression.fragment() == text
            assert evaluate(con, expression, source) == value

    def test_a_family_keeps_alias_and_order(self) -> None:
        assert col("s").alias("n").str().upper().as_select() == 'upper("s")'
        assert col("s").desc().str().as_order() == '"s" DESC'

    def test_a_family_result_is_in_the_family_of_its_return_type(self) -> None:
        assert type(col("s").str().upper()).__name__ == "StrExpr"
        assert type(col("s").str().string_split(",")).__name__ == "ListExpr"

    def test_a_lambda_family_method_takes_a_python_callable(self, con: duckdb.DuckDBPyConnection) -> None:
        doubled = col("xs").list().transform(lambda x: x * 2)
        # The variable is named after its parameter with a suffix drawn per lambda.
        assert re.fullmatch(r'list_transform\("xs", lambda "(x_[0-9a-f]{6})": \("\1" \* 2\)\)', doubled.fragment())
        assert evaluate(con, doubled, " FROM (SELECT [1, 2] AS xs)") == [2, 4]


class TestLambdasFromCallables:
    def test_one_and_two_parameters(self, con: duckdb.DuckDBPyConnection) -> None:
        total = fn("list_reduce", col("xs"), lambda acc, x: acc + x)
        # Each variable is its parameter's name with a suffix of its own, so no two can fold to one name.
        pattern = r'list_reduce\("xs", lambda "acc_([0-9a-f]{6})", "x_([0-9a-f]{6})": \("acc_\1" \+ "x_\2"\)\)'
        found = re.fullmatch(pattern, total.fragment())
        assert found
        assert found.group(1) != found.group(2)
        assert evaluate(con, total, " FROM (SELECT [1, 2, 3] AS xs)") == 6

    def test_a_parameter_shadows_a_column_of_the_same_name(self, con: duckdb.DuckDBPyConnection) -> None:
        kept = fn("list_filter", col("xs"), lambda x: x > 1)
        assert evaluate(con, kept, " FROM (SELECT [1, 2, 3] AS xs, 100 AS x)") == [2, 3]

    @pytest.mark.parametrize(
        "nested",
        [
            pytest.param(
                lambda: col("l").list().transform(lambda y: lit([10]).list().transform(lambda x: x + col("x_1"))),
                id="distinct_names",
            ),
            pytest.param(
                lambda: col("l").list().transform(lambda x: lit([10]).list().transform(lambda x: x + col("x_1"))),
                id="shadowing_names",
            ),
        ],
    )
    def test_a_generated_variable_never_captures_a_column(
        self, con: duckdb.DuckDBPyConnection, nested: types.FunctionType
    ) -> None:
        # The inner variable is renamed when it shadows, and the new name must not be a column the body reads.
        assert evaluate(con, nested(), " FROM (SELECT [1, 2] AS l, 100 AS x_1)") == [[110], [110]]

    @pytest.mark.parametrize(
        ("nested", "expected"),
        [
            # Python's lexical scope: each name reads its own lambda's parameter, an inner name shadowing only itself.
            pytest.param(
                lambda: lit([1]).list().transform(lambda x: lit([10]).list().transform(lambda y: x + y)),
                [[11]],
                id="two_names",
            ),
            pytest.param(
                lambda: lit([1]).list().transform(lambda x: lit([10]).list().transform(lambda x: x + x)),
                [[20]],
                id="the_same_name_shadows",
            ),
            pytest.param(
                lambda: lit([1]).list().transform(lambda x: lit([10]).list().transform(lambda y: x)),
                [[1]],
                id="the_outer_name_reaches_inside",
            ),
            # SQL folds identifier case, so x and X must still be two variables.
            pytest.param(
                lambda: lit([1]).list().transform(lambda x: lit([10]).list().transform(lambda X: x + X)),
                [[11]],
                id="names_that_differ_only_in_case",
            ),
        ],
    )
    def test_nested_lambdas_keep_python_scope(
        self, con: duckdb.DuckDBPyConnection, nested: types.FunctionType, expected: object
    ) -> None:
        assert evaluate(con, nested()) == expected

    @pytest.mark.parametrize(
        ("function", "message"),
        [
            (lambda: col("x"), "takes none"),
            (lambda x=1: x, "plain positional names"),
            (lambda *xs: xs[0], "plain positional names"),
            (lambda x: object(), "not an expression"),
        ],
        ids=["no parameter", "default", "variadic", "not an expression"],
    )
    def test_what_cannot_be_a_sql_lambda_is_refused(self, function: object, message: str) -> None:
        with pytest.raises(TypeError, match=message):
            fn("list_transform", col("xs"), function)


class TestTemporalLiterals:
    """A date or time written into the SQL means what it means bound as a parameter."""

    @pytest.mark.parametrize(
        "value",
        [
            datetime.timedelta(days=100_000, microseconds=1),
            datetime.timedelta(microseconds=-1),
            pd.Timedelta(-1500, "ns"),
            datetime.time(12, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
            np.datetime64("2020-01-02T03:04:05", "s"),
            np.datetime64("2020-01-02T03:04:05.123", "ms"),
            np.datetime64("2020-03", "M"),
            pd.Timestamp("2020-01-02 03:04:05").as_unit("s"),
            pd.Timestamp("2020-01-02 03:04:05.5", tz="Europe/Amsterdam"),
            pd.Timestamp("2020-01-02 03:04:05.123456789"),
        ],
        ids=str,
    )
    def test_written_in_and_bound_agree(self, con: duckdb.DuckDBPyConnection, value: object) -> None:
        written = con.execute(f"SELECT typeof(v), v FROM (SELECT {render_literal(value)} AS v)").fetchall()
        bound = con.execute("SELECT typeof(v), v FROM (SELECT $1 AS v)", [value]).fetchall()
        assert written == bound
        assert sql_type_of(value) == written[0][0]

    @pytest.mark.parametrize(
        ("value", "decided_by"),
        [
            (np.timedelta64(-1500, "ns"), "engine"),
            (np.timedelta64(7_200_000_000_000_000_000, "us"), "engine"),
            (np.array([2**62], "i8").view("datetime64[7D]")[0], "numpy"),
        ],
        ids=["nanoseconds", "microseconds", "weeks"],
    )
    def test_what_a_parameter_cannot_hold_is_refused_written_in(
        self, con: duckdb.DuckDBPyConnection, value: object, decided_by: str
    ) -> None:
        # Numpy before 2.5 converts the weeks value to a plain int, which binds as a number and is no date to write.
        if decided_by == "numpy" and _converts(value):
            assert con.execute("SELECT typeof($1)", [value]).fetchall() == [("BIGINT",)]
            with pytest.raises(duckdb.InvalidInputException, match="Expected a date"):
                render_literal(value)
            return
        with pytest.raises((duckdb.Error, OverflowError)) as written:
            render_literal(value)
        with pytest.raises((duckdb.Error, OverflowError)) as bound:
            con.execute("SELECT $1", [value])
        assert type(written.value) is type(bound.value)

    @pytest.mark.parametrize(
        "value",
        [
            np.datetime64("2020-01-02T03:04:05.123456000", "ns"),
            np.datetime64("2020-01-02T03:04:05.123456789", "ns"),
            np.datetime64(2**40, "M"),
            np.datetime64(2**62, "s"),
        ],
        ids=["nanoseconds", "nanoseconds past microseconds", "months past range", "seconds past range"],
    )
    def test_a_numpy_value_that_binds_as_a_number_is_not_written_in(
        self, con: duckdb.DuckDBPyConnection, value: object
    ) -> None:
        # This engine binds these as BIGINT; the seam's temporal literal refuses to write a number for a date.
        assert con.execute("SELECT typeof($1)", [value]).fetchall() == [("BIGINT",)]
        with pytest.raises(duckdb.InvalidInputException, match="Expected a date, time, datetime"):
            render_literal(value)

    @pytest.mark.parametrize(
        "missing",
        [
            pd.NaT,
            np.datetime64("NaT", "D"),
            np.timedelta64("NaT", "ns"),
            np.datetime64("NaT", ("s", 15)),
            np.array(-(2**63), "datetime64[0M]")[()],
        ],
        ids=["pandas", "days", "nanoseconds", "stepped", "zero step"],
    )
    def test_a_missing_value_is_null(self, missing: object) -> None:
        assert render_literal(missing) == "NULL"

    @pytest.mark.parametrize("missing", [pd.NaT, np.timedelta64("NaT", "ns"), np.datetime64("NaT", "ms")], ids=str)
    def test_a_missing_operand_is_written_in_as_null(self, con: duckdb.DuckDBPyConnection, missing: object) -> None:
        rendered = (col("x") + missing).fragment()
        assert rendered == '("x" + NULL)'
        assert con.execute(f"SELECT {rendered} FROM (SELECT TIMESTAMP '2020-01-01' AS x)").fetchall() == [(None,)]


class TestNestedTemporalLiterals:
    NAIVE = datetime.datetime(2020, 1, 1, 12)
    AWARE = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC)

    @pytest.mark.parametrize(
        "value",
        [
            [NAIVE, AWARE],
            [[NAIVE], [AWARE]],
            [{"a": NAIVE}, {"a": AWARE}],
            {1: NAIVE, 2: AWARE},
            {NAIVE: 1, AWARE: 2},
            [{"b": 1}, {"a": NAIVE}, {"a": AWARE}],
            [{"a": AWARE}, {"b": 1}, {"a": NAIVE}],
            [{"a": NAIVE}, {"A": AWARE}],
            [[], [NAIVE], [AWARE]],
            [{"a": None}, {"a": NAIVE}, {"a": AWARE}],
            {1: [NAIVE], 2: [AWARE]},
            [{NAIVE: 1}, {AWARE: 2}],
            [{"a": [NAIVE]}, {"a": [None, AWARE]}],
        ],
        ids=[
            "one list",
            "lists",
            "structs",
            "map values",
            "map keys",
            "a struct without the field first",
            "a struct without the field between",
            "field names differing in case",
            "an empty list first",
            "a null field first",
            "lists in map values",
            "keys of maps in a list",
            "lists in struct fields",
        ],
    )
    def test_both_kinds_at_one_place_are_refused_written_in(self, value: object) -> None:
        # Written in, the engine reads the zoneless one in the session's time zone, and bound it reads it as UTC.
        assert sql_type_of(value) is None
        with pytest.raises(duckdb.InvalidInputException, match="with and without a time zone"):
            render_literal(value)

    def test_written_in_they_would_mean_another_instant(self) -> None:
        con = duckdb.connect()
        con.execute("SET TimeZone = 'America/New_York'")
        value = [self.NAIVE, self.AWARE]
        bound = con.execute("SELECT $1", [value]).fetchall()[0][0]
        as_written = "['2020-01-01 12:00:00'::TIMESTAMP, '2020-01-01 12:00:00+00'::TIMESTAMP WITH TIME ZONE]"
        assert con.execute(f"SELECT {as_written}").fetchall()[0][0] != bound

    @pytest.mark.parametrize(
        "value",
        [
            [[], [NAIVE]],
            [{"a": None}, {"a": NAIVE}],
            [None, {"a": [None]}, {"a": [AWARE]}],
        ],
        ids=["empty list", "null field", "every depth"],
    )
    def test_a_missing_part_written_in_reads_as_bound(self, con: duckdb.DuckDBPyConnection, value: object) -> None:
        written = con.execute(f"SELECT v::VARCHAR, typeof(v) FROM (SELECT {render_literal(value)} AS v)").fetchall()
        bound = con.execute("SELECT $1::VARCHAR, typeof($1)", [value]).fetchall()
        assert written == bound

    def test_a_struct_keeps_each_field_as_it_is(self, con: duckdb.DuckDBPyConnection) -> None:
        value = {"a": self.NAIVE, "b": self.AWARE}
        assert con.execute(f"SELECT {render_literal(value)}").fetchall() == [(value,)]

    def test_a_datetime_whose_zone_gives_no_offset_is_naive(self) -> None:
        # The one reading of each entry point is pinned by the agreement test; the type alone is decided here.
        assert sql_type_of(datetime.datetime(2020, 1, 1, 12, tzinfo=NoOffset())) == "TIMESTAMP"

    def test_a_tzinfo_that_empties_the_list_being_rendered_cannot_corrupt_the_text(self) -> None:
        # The text honestly shows the list's remaining content; nothing crashes and nothing renders half-read.
        values: list[object] = []
        values.extend(datetime.datetime(2020 + i, 1, 1, tzinfo=Emptying(values)) for i in range(2))
        assert render_literal(values) == "[]"

    def test_a_value_a_deepcopy_cannot_take_is_refused_by_lit(self) -> None:
        values: list[object] = []
        values.append(datetime.datetime(2020, 1, 1, tzinfo=Emptying(values)))
        with pytest.raises(TypeError, match="not plain data"):
            lit(values)

    def test_concurrent_temporal_literals_render_exactly(self) -> None:
        # Each render borrows a pooled literal connection, so racing threads must neither mix values nor deadlock.
        aware = [datetime.datetime(2020, 1, 1 + i, tzinfo=datetime.UTC) for i in range(8)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            out = list(pool.map(render_literal, aware * 50))
        assert out == [f"'2020-01-0{(i % 8) + 1} 00:00:00+00'::TIMESTAMP WITH TIME ZONE" for i in range(400)]


class TestNumpyScalarsAreTheirPythonValues:
    """A numpy bool, integer or float stands for its Python value wherever an expression takes a value."""

    @pytest.mark.parametrize("number", [np.int8(3), np.int64(3), np.uint64(3)], ids=lambda v: type(v).__name__)
    def test_a_numpy_integer_compares_as_its_value(self, con: duckdb.DuckDBPyConnection, number: object) -> None:
        source = " FROM (SELECT 3 AS id)"
        assert (col("id") == number).fragment() == '("id" = 3)'
        assert evaluate(con, col("id") == number, source) is True
        assert evaluate(con, col("id") > number, source) is False
        assert evaluate(con, number == col("id"), source) is True
        assert evaluate(con, number < col("id"), source) is False

    def test_a_numpy_float_is_written_as_its_value(self, con: duckdb.DuckDBPyConnection) -> None:
        assert (col("x") + np.float64(1000.0)).fragment() == '("x" + 1000.0::DOUBLE)'
        assert (col("x") + np.float32(0.5)).fragment() == '("x" + 0.5::DOUBLE)'
        assert evaluate(con, col("x") + np.float64(1000.0), " FROM (SELECT 1.0 AS x)") == 1001.0

    def test_a_numpy_bool_is_a_boolean(self) -> None:
        assert (col("b") == np.bool_(True)).fragment() == '("b" = TRUE)'

    def test_numpy_on_the_left_reaches_the_reflected_operator(self) -> None:
        assert (np.int64(3) + col("x")).fragment() == '(3 + "x")'
        assert (np.int64(3) - col("x")).fragment() == '(3 - "x")'
        assert (np.float64(2.5) * col("x")).fragment() == '(2.5::DOUBLE * "x")'

    def test_isin_between_function_arguments_and_lit_take_numpy(self, con: duckdb.DuckDBPyConnection) -> None:
        source = " FROM (SELECT 3 AS id)"
        assert evaluate(con, col("id").isin(np.int64(3), np.int64(5)), source) is True
        assert evaluate(con, col("id").isin(np.int64(5)), source) is False
        assert evaluate(con, col("id").between(np.int64(2), np.int64(4)), source) is True
        assert evaluate(con, fn("greatest", col("id"), np.int64(4)), source) == 4
        assert lit(np.int16(7)).fragment() == "7"

    def test_numpy_written_into_the_text_is_its_value(self) -> None:
        assert render_literal([np.int64(1), np.float64(0.5)]) == "[1, 0.5::DOUBLE]"
        assert render_literal({np.int64(1): np.bool_(False)}) == "MAP {1: FALSE}"
        day = datetime.date(2020, 1, 1)
        assert render_literal({"n": np.int64(1), "d": day}) == render_literal({"n": 1, "d": day})

    def test_a_numpy_integer_is_a_position(self, con: duckdb.DuckDBPyConnection) -> None:
        assert col("xs")[np.int64(1)].fragment() == col("xs")[1].fragment()
        assert evaluate(con, col("xs")[np.int64(2)], " FROM (SELECT [10, 20] AS xs)") == 20
        with pytest.raises(TypeError, match="not by bool"):
            col("xs")[np.bool_(True)]

    def test_a_lambda_may_return_a_numpy_value(self, con: duckdb.DuckDBPyConnection) -> None:
        ones = fn("list_transform", col("xs"), lambda x: np.int64(1))
        assert evaluate(con, ones, " FROM (SELECT [1, 2] AS xs)") == [1, 1]

    def test_a_numpy_still_importing_elsewhere_leaves_python_values_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Another thread's import puts numpy in sys.modules before its scalar types exist.
        partial = types.ModuleType("numpy")
        partial.bool_ = np.bool_
        monkeypatch.setitem(sys.modules, "numpy", partial)
        assert lit(1).fragment() == "1"
        assert (col("x") + 1.5).fragment() == '("x" + 1.5::DOUBLE)'
        assert render_literal([1, [2, {"a": 3}]]) == "[1, [2, {'a': 3}]]"

    def test_a_container_without_numpy_values_comes_back_as_itself(self) -> None:
        nested = [[1, 2.5, "a"], {"k": (3, None)}, (datetime.date(2020, 1, 1),)]
        assert plain_value(nested) is nested
        mixed = [[1, np.int64(2)], {"k": 3}]
        converted = plain_value(mixed)
        assert isinstance(converted, list)
        assert converted == [[1, 2], {"k": 3}]
        assert type(converted[0][1]) is int
        assert converted[1] is mixed[1]

    def test_a_container_holding_itself_is_walked_once(self) -> None:
        cyclic: list[object] = [np.int64(1)]
        cyclic.append(cyclic)
        converted = plain_value(cyclic)
        assert isinstance(converted, list)
        assert type(converted[0]) is int
        assert converted[1] is cyclic

    def test_a_numpy_value_without_a_sql_counterpart_is_refused(self) -> None:
        with pytest.raises(TypeError):
            col("x") + np.complex128(1)
        with pytest.raises(TypeError, match="cannot render a literal of type complex"):
            lit(np.complex128(1)).fragment()
        with pytest.raises(TypeError, match="cannot render a literal of type longdouble"):
            lit(np.longdouble(1)).fragment()


# A time zone whose offset writes another value into SQL, on this thread and on another, and one whose offset fails.
REENTRY = """
import datetime as dt
import threading
from duckdb._expression import render_literal


class Zone(dt.tzinfo):
    def dst(self, moment):
        return None

    def tzname(self, moment):
        return "zone"


class Nested(Zone):
    def __init__(self, depth):
        self.depth = depth

    def utcoffset(self, moment):
        if self.depth:
            render_literal(dt.datetime(2020, 1, 1, tzinfo=Nested(self.depth - 1)))
        return dt.timedelta(0)


class Elsewhere(Zone):
    def utcoffset(self, moment):
        found = []
        worker = threading.Thread(target=lambda: found.append(render_literal(dt.date(2020, 1, 2))), daemon=True)
        worker.start()
        worker.join(20)
        if found != ["'2020-01-02'::DATE"]:
            raise RuntimeError("the other thread did not finish")
        return dt.timedelta(0)


class Failing(Zone):
    def utcoffset(self, moment):
        raise ValueError("no offset")


print("rendered", render_literal(dt.datetime(2020, 1, 1, tzinfo=Nested(6))))
print("rendered", render_literal(dt.datetime(2020, 1, 1, tzinfo=Elsewhere())))
for _ in range(6):
    try:
        render_literal(dt.datetime(2020, 1, 1, tzinfo=Failing()))
    except ValueError as error:
        print("rendered", error)
print("rendered", render_literal(dt.date(2020, 1, 3)))
"""


class TestLiteralReentry:
    """Writing a value into SQL may run Python code that writes another, on this thread or another, without waiting.

    A conversion that fails leaves nothing held either.
    """

    def test_nested_concurrent_and_failing_conversions_all_finish(self) -> None:
        # A child process, so a conversion that waits forever fails the test instead of hanging the run.
        result = subprocess.run(
            [sys.executable, "-c", REENTRY], capture_output=True, text=True, timeout=120, check=False
        )
        assert result.returncode == 0, result.stderr
        written = "'2020-01-01 00:00:00+00'::TIMESTAMP WITH TIME ZONE"
        lines = [line.removeprefix("rendered ") for line in result.stdout.splitlines() if line.startswith("rendered ")]
        assert lines == [written, written, *["no offset"] * 6, "'2020-01-03'::DATE"]

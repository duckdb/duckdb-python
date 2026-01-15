import pytest

import duckdb


@pytest.fixture(autouse=True)
def setup_and_teardown_of_table(duckdb_cursor):
    duckdb_cursor.execute("create table agg(id int, v int, t int, f float, s varchar);")
    duckdb_cursor.execute(
        """
        insert into agg values
        (1, 1, 2, 0.54, 'h'),
        (1, 1, 1, 0.21, 'e'),
        (1, 2, 3, 0.001, 'l'),
        (2, 10, 4, 0.04, 'l'),
        (2, 11, -1, 10.45, 'o'),
        (3, -1, 0, 13.32, ','),
        (3, 5, -2, 9.87, 'wor'),
        (3, null, 10, 6.56, 'ld');
        """
    )
    yield
    duckdb_cursor.execute("drop table agg")


@pytest.fixture
def table(duckdb_cursor):
    return duckdb_cursor.table("agg")


class TestRAPIAggregations:
    # General aggregate functions

    def test_any_value(self, table):
        result = table.order("id, t").any_value("v").execute().fetchall()
        expected = [(1,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = (
            table.order("id, t").any_value("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        )
        expected = [(1, 1), (2, 11), (3, 5)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_arg_max(self, table):
        result = table.arg_max("t", "v").execute().fetchall()
        expected = [(-1,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.arg_max("t", "v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 3), (2, -1), (3, -2)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_arg_min(self, table):
        result = table.arg_min("t", "v").execute().fetchall()
        expected = [(0,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.arg_min("t", "v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 2), (2, 4), (3, 0)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_avg(self, table):
        result = table.avg("v").execute().fetchall()
        expected = [(4.14,)]
        assert len(result) == len(expected)
        assert round(result[0][0], 2) == expected[0][0]
        result = [
            (r[0], round(r[1], 2))
            for r in table.avg("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        ]
        expected = [(1, 1.33), (2, 10.5), (3, 2)]

    def test_bit_and(self, table):
        result = table.bit_and("v").execute().fetchall()
        expected = [(0,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.bit_and("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 0), (2, 10), (3, 5)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_bit_or(self, table):
        result = table.bit_or("v").execute().fetchall()
        expected = [(-1,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.bit_or("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 3), (2, 11), (3, -1)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_bit_xor(self, table):
        result = table.bit_xor("v").execute().fetchall()
        expected = [(-7,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.bit_xor("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 2), (2, 1), (3, -6)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_bitstring_agg(self, table):
        result = table.bitstring_agg("v").execute().fetchall()
        expected = [("1011001000011",)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.bitstring_agg("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, "0011000000000"), (2, "0000000000011"), (3, "1000001000000")]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        with pytest.raises(duckdb.InvalidInputException):
            table.bitstring_agg("v", min="1")
        with pytest.raises(duckdb.InvalidTypeException):
            table.bitstring_agg("v", min="1", max=11)

    def test_bool_and(self, table):
        result = table.bool_and("v::BOOL").execute().fetchall()
        expected = [(True,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.bool_and("t::BOOL", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, True), (2, True), (3, False)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_bool_or(self, table):
        result = table.bool_or("v::BOOL").execute().fetchall()
        expected = [(True,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.bool_or("v::BOOL", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, True), (2, True), (3, True)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_count(self, table):
        result = table.count("*").execute().fetchall()
        expected = [(8,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.count("*", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 3), (2, 2), (3, 3)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_value_counts(self, table):
        result = table.value_counts("v").execute().fetchall()
        expected = [(None, 0), (-1, 1), (1, 2), (2, 1), (5, 1), (10, 1), (11, 1)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.value_counts("v", groups="v").order("v").execute().fetchall()
        expected = [(-1, 1), (1, 2), (2, 1), (5, 1), (10, 1), (11, 1), (None, 0)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_favg(self, table):
        result = [round(r[0], 2) for r in table.favg("f").execute().fetchall()]
        expected = [5.12]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = [
            (r[0], round(r[1], 2))
            for r in table.favg("f", groups="id", projected_columns="id").order("id").execute().fetchall()
        ]
        expected = [(1, 0.25), (2, 5.24), (3, 9.92)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_first(self, table):
        result = table.first("v").execute().fetchall()
        expected = [(1,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.first("v", "id", "id").order("id").execute().fetchall()
        expected = [(1, 1), (2, 10), (3, -1)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_last(self, table):
        result = table.last("v").execute().fetchall()
        expected = [(None,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.last("v", "id", "id").order("id").execute().fetchall()
        expected = [(1, 2), (2, 11), (3, None)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_fsum(self, table):
        result = [round(r[0], 2) for r in table.fsum("f").execute().fetchall()]
        expected = [40.99]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = [
            (r[0], round(r[1], 2))
            for r in table.fsum("f", groups="id", projected_columns="id").order("id").execute().fetchall()
        ]
        expected = [(1, 0.75), (2, 10.49), (3, 29.75)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_geomean(self, table):
        result = [round(r[0], 2) for r in table.geomean("f").execute().fetchall()]
        expected = [0.67]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = [
            (r[0], round(r[1], 2))
            for r in table.geomean("f", groups="id", projected_columns="id").order("id").execute().fetchall()
        ]
        expected = [(1, 0.05), (2, 0.65), (3, 9.52)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_histogram(self, table):
        result = table.histogram("v").execute().fetchall()
        expected = [({-1: 1, 1: 2, 2: 1, 5: 1, 10: 1, 11: 1},)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.histogram("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, {1: 2, 2: 1}), (2, {10: 1, 11: 1}), (3, {-1: 1, 5: 1})]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_list(self, table):
        result = table.list("v").execute().fetchall()
        expected = [([1, 1, 2, 10, 11, -1, 5, None],)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.list("v", groups="id order by t asc", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, [1, 1, 2]), (2, [10, 11]), (3, [-1, 5, None])]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_max(self, table):
        result = table.max("v").execute().fetchall()
        expected = [(11,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.max("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 2), (2, 11), (3, 5)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_min(self, table):
        result = table.min("v").execute().fetchall()
        expected = [(-1,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.min("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 1), (2, 10), (3, -1)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_product(self, table):
        result = table.product("v").execute().fetchall()
        expected = [(-1100,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.product("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 2), (2, 110), (3, -5)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_string_agg(self, table):
        result = table.string_agg("s", sep="/").execute().fetchall()
        expected = [("h/e/l/l/o/,/wor/ld",)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = (
            table.string_agg("s", sep="/", groups="id order by t asc", projected_columns="id")
            .order("id")
            .execute()
            .fetchall()
        )
        expected = [(1, "h/e/l"), (2, "l/o"), (3, ",/wor/ld")]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_sum(self, table):
        result = table.sum("v").execute().fetchall()
        expected = [(29,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.sum("v", groups="id", projected_columns="id").execute().fetchall()
        expected = [(1, 4), (2, 21), (3, 4)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    # TODO: Approximate aggregate functions  # noqa: TD002, TD003

    # TODO: Statistical aggregate functions  # noqa: TD002, TD003
    def test_median(self, table):
        result = table.median("v").execute().fetchall()
        expected = [(2.0,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.median("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 1.0), (2, 10.5), (3, 2.0)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_mode(self, table):
        result = table.mode("v").execute().fetchall()
        expected = [(1,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.mode("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 1), (2, 10), (3, -1)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_quantile_cont(self, table):
        result = table.quantile_cont("v").execute().fetchall()
        expected = [(2.0,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = [[round(x, 2) for x in r[0]] for r in table.quantile_cont("v", q=[0.1, 0.5]).execute().fetchall()]
        expected = [[0.2, 2.0]]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = table.quantile_cont("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 1.0), (2, 10.5), (3, 2.0)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = [
            (r[0], [round(x, 2) for x in r[1]])
            for r in table.quantile_cont("v", q=[0.2, 0.5], groups="id", projected_columns="id")
            .order("id")
            .execute()
            .fetchall()
        ]
        expected = [(1, [1.0, 1.0]), (2, [10.2, 10.5]), (3, [0.2, 2.0])]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    @pytest.mark.parametrize("f", ["quantile_disc", "quantile"])
    def test_quantile_disc(self, table, f):
        result = getattr(table, f)("v").execute().fetchall()
        expected = [(2,)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = getattr(table, f)("v", q=[0.2, 0.5]).execute().fetchall()
        expected = [([1, 2],)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = getattr(table, f)("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        expected = [(1, 1), (2, 10), (3, -1)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = (
            getattr(table, f)("v", q=[0.2, 0.8], groups="id", projected_columns="id").order("id").execute().fetchall()
        )
        expected = [(1, [1, 2]), (2, [10, 11]), (3, [-1, 5])]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_std_pop(self, table):
        result = [round(r[0], 2) for r in table.stddev_pop("v").execute().fetchall()]
        expected = [4.36]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = [
            (r[0], round(r[1], 2))
            for r in table.stddev_pop("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        ]
        expected = [(1, 0.47), (2, 0.5), (3, 3.0)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    @pytest.mark.parametrize("f", ["stddev_samp", "stddev", "std"])
    def test_std_samp(self, table, f):
        result = [round(r[0], 2) for r in getattr(table, f)("v").execute().fetchall()]
        expected = [4.71]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = [
            (r[0], round(r[1], 2))
            for r in getattr(table, f)("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        ]
        expected = [(1, 0.58), (2, 0.71), (3, 4.24)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_var_pop(self, table):
        result = [round(r[0], 2) for r in table.var_pop("v").execute().fetchall()]
        expected = [18.98]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = [
            (r[0], round(r[1], 2))
            for r in table.var_pop("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        ]
        expected = [(1, 0.22), (2, 0.25), (3, 9.0)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    @pytest.mark.parametrize("f", ["var_samp", "variance", "var"])
    def test_var_samp(self, table, f):
        result = [round(r[0], 2) for r in getattr(table, f)("v").execute().fetchall()]
        expected = [22.14]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))
        result = [
            (r[0], round(r[1], 2))
            for r in getattr(table, f)("v", groups="id", projected_columns="id").order("id").execute().fetchall()
        ]
        expected = [(1, 0.33), (2, 0.5), (3, 18.0)]
        assert len(result) == len(expected)
        assert all(r == e for r, e in zip(result, expected))

    def test_describe(self, table):
        assert table.describe().fetchall() is not None


class TestRAPIAggregationsColumnEscaping:
    """Test that aggregate functions properly escape column names that need quoting."""

    def test_reserved_keyword_column_name(self, duckdb_cursor):
        # Column name "select" is a reserved SQL keyword
        rel = duckdb_cursor.sql('select 1 as "select", 2 as "order"')
        result = rel.sum("select").fetchall()
        assert result == [(1,)]

        result = rel.avg("order").fetchall()
        assert result == [(2.0,)]

    def test_column_name_with_space(self, duckdb_cursor):
        rel = duckdb_cursor.sql('select 10 as "my column"')
        result = rel.sum("my column").fetchall()
        assert result == [(10,)]

    def test_column_name_with_quotes(self, duckdb_cursor):
        # Column name containing a double quote
        rel = duckdb_cursor.sql('select 5 as "col""name"')
        result = rel.sum('col"name').fetchall()
        assert result == [(5,)]

    def test_qualified_column_name(self, duckdb_cursor):
        # Qualified column name like table.column
        rel = duckdb_cursor.sql("select 42 as value")
        # When using qualified names, they should be properly escaped
        result = rel.sum("value").fetchall()
        assert result == [(42,)]


class TestRAPIAggregationsExpressionPassthrough:
    """Test that aggregate functions correctly pass through SQL expressions without escaping."""

    def test_cast_expression(self, duckdb_cursor):
        # Cast expressions should pass through without being quoted
        rel = duckdb_cursor.sql("select 1 as v, 0 as f")
        result = rel.bool_and("v::BOOL").fetchall()
        assert result == [(True,)]

        result = rel.bool_or("f::BOOL").fetchall()
        assert result == [(False,)]

    def test_star_expression(self, duckdb_cursor):
        # Star (*) should pass through for count
        rel = duckdb_cursor.sql("select 1 as a union all select 2")
        result = rel.count("*").fetchall()
        assert result == [(2,)]

    def test_arithmetic_expression(self, duckdb_cursor):
        # Arithmetic expressions should pass through
        rel = duckdb_cursor.sql("select 10 as a, 5 as b")
        result = rel.sum("a + b").fetchall()
        assert result == [(15,)]

    def test_function_expression(self, duckdb_cursor):
        # Function calls should pass through
        rel = duckdb_cursor.sql("select -5 as v")
        result = rel.sum("abs(v)").fetchall()
        assert result == [(5,)]

    def test_case_expression(self, duckdb_cursor):
        # CASE expressions should pass through
        rel = duckdb_cursor.sql("select 1 as v union all select 2 union all select 3")
        result = rel.sum("case when v > 1 then v else 0 end").fetchall()
        assert result == [(5,)]


class TestRAPIAggregationsWithInvalidInput:
    """Test that only expression can be used."""

    def test_injection_with_semicolon_is_neutralized(self, duckdb_cursor):
        # Semicolon injection fails to parse as expression, gets quoted as identifier
        rel = duckdb_cursor.sql("select 1 as v")
        with pytest.raises(duckdb.BinderException, match="not found in FROM clause"):
            rel.sum("v; drop table agg; --").fetchall()

    def test_injection_with_union_is_neutralized(self, duckdb_cursor):
        # UNION fails to parse as single expression, gets quoted
        rel = duckdb_cursor.sql("select 1 as v")
        with pytest.raises(duckdb.BinderException, match="not found in FROM clause"):
            rel.sum("v union select * from agg").fetchall()

    def test_subquery_is_contained(self, duckdb_cursor):
        # Subqueries are valid expressions - they're contained within the aggregate
        # and cannot break out of the expression context
        rel = duckdb_cursor.sql("select 1 as v")
        # This executes sum((select 1)) = sum(1) = 1 - contained, not an injection
        result = rel.sum("(select 1)").fetchall()
        assert result == [(1,)]

    def test_injection_closing_paren_is_neutralized(self, duckdb_cursor):
        # Adding a closing paren fails to parse, gets quoted
        rel = duckdb_cursor.sql("select 1 as v")
        with pytest.raises(duckdb.BinderException, match="not found in FROM clause"):
            rel.sum("v) from agg; drop table agg; --").fetchall()

    def test_comment_is_harmless(self, duckdb_cursor):
        # SQL comments are stripped during parsing, so "v -- comment" parses as just "v"
        rel = duckdb_cursor.sql("select 1 as v")
        result = rel.sum("v -- this is ignored").fetchall()
        assert result == [(1,)]

    def test_empty_expression_rejected(self, duckdb_cursor):
        # Empty or whitespace-only expressions should be rejected
        rel = duckdb_cursor.sql("select 1 as v")
        with pytest.raises(duckdb.ParserException):
            rel.sum("").fetchall()

    def test_whitespace_only_expression_rejected(self, duckdb_cursor):
        # Whitespace-only expressions should be rejected
        rel = duckdb_cursor.sql("select 1 as v")
        with pytest.raises(duckdb.ParserException):
            rel.sum("   ").fetchall()


class TestRAPIStringAggSeparatorEscaping:
    """Test that string_agg separator is properly escaped as a string literal."""

    def test_simple_separator(self, duckdb_cursor):
        rel = duckdb_cursor.sql("select 'a' as s union all select 'b' union all select 'c'")
        result = rel.string_agg("s", ",").fetchall()
        assert result == [("a,b,c",)]

    def test_separator_with_single_quote(self, duckdb_cursor):
        # Separator containing a single quote should be properly escaped
        rel = duckdb_cursor.sql("select 'a' as s union all select 'b'")
        result = rel.string_agg("s", "','").fetchall()
        assert result == [("a','b",)]

    def test_separator_with_special_chars(self, duckdb_cursor):
        rel = duckdb_cursor.sql("select 'x' as s union all select 'y'")
        result = rel.string_agg("s", " | ").fetchall()
        assert result == [("x | y",)]

    def test_separator_injection_attempt(self, duckdb_cursor):
        # Attempt to inject via separator - should be safely quoted as string literal
        rel = duckdb_cursor.sql("select 'a' as s union all select 'b'")
        # This should NOT execute the injection - separator becomes a literal string
        result = rel.string_agg("s", "'); drop table agg; --").fetchall()
        assert result == [("a'); drop table agg; --b",)]

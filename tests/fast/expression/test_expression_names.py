"""A string names one thing and is never split; qualification, sides and fields are Python structure."""

from __future__ import annotations

import pickle

import pytest

import duckdb
from duckdb._expression import coalesce, col, fn, fold_name, function_name, lit, star


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("CREATE SCHEMA s")
    connection.execute("CREATE MACRO s.twice(x) AS x * 2")
    connection.execute("""CREATE TABLE shadow AS SELECT 21 AS "hello.world", {'world': 7} AS hello""")
    connection.execute("""CREATE TABLE dotted AS SELECT 21 AS "hello.world", 1 AS id""")
    yield connection
    connection.close()


def rows(con: duckdb.DuckDBPyConnection, select: object, source: str) -> list[tuple[object, ...]]:
    return con.execute(f"SELECT {select} FROM {source}").fetchall()


class TestAStringIsOneName:
    def test_a_column_is_quoted_whole(self) -> None:
        assert col("hello.world").fragment() == '"hello.world"'
        assert col('say "hi"').fragment() == '"say ""hi"""'
        assert col("order").fragment() == '"order"'

    def test_a_dotted_name_selects_its_column(self, con: duckdb.DuckDBPyConnection) -> None:
        assert rows(con, col("hello.world"), "dotted") == [(21,)]
        assert rows(con, duckdb.ColumnExpression("hello.world"), "dotted") == [(21,)]

    def test_a_struct_named_like_a_prefix_does_not_shadow(self, con: duckdb.DuckDBPyConnection) -> None:
        assert rows(con, col("hello.world"), "shadow") == [(21,)]
        assert rows(con, col("hello")["world"], "shadow") == [(7,)]
        assert rows(con, duckdb.ColumnExpression("hello", "world"), "shadow") == [(7,)]

    def test_col_takes_one_name_or_the_parts_of_a_qualified_one(self, con: duckdb.DuckDBPyConnection) -> None:
        assert col("id").fragment() == '"id"'
        assert col("t", "id").fragment() == '"t"."id"'
        assert col("memory", "main", "dotted", "id").fragment() == '"memory"."main"."dotted"."id"'
        # One string is one name, so a dotted one is a column and never a qualified path.
        assert col("hello.world").fragment() == '"hello.world"'
        assert con.execute(f"SELECT {col('dotted', 'id')} FROM dotted AS dotted").fetchall() == [(1,)]

    def test_col_and_column_expression_build_the_same_column(self) -> None:
        for parts in [("id",), ("t", "id"), ("hello.world",), ("memory", "main", "dotted", "id")]:
            assert col(*parts).fragment() == duckdb.ColumnExpression(*parts).fragment()

    def test_col_takes_names_only(self) -> None:
        # A qualified name is given in parts, so a tuple is not a name.
        with pytest.raises(TypeError, match="column name"):
            col(("l", "x"))
        with pytest.raises(TypeError, match="column name"):
            col(3)
        with pytest.raises(TypeError, match="column name"):
            col("t", 3)
        with pytest.raises(TypeError, match="column name"):
            col("t", None)
        with pytest.raises(ValueError, match="empty"):
            col("")
        with pytest.raises(ValueError, match="empty"):
            col("t", "")
        with pytest.raises(ValueError, match="empty"):
            col()

    def test_column_expression_takes_names_only(self) -> None:
        with pytest.raises(TypeError, match="column name"):
            duckdb.ColumnExpression("t", 3)
        with pytest.raises(ValueError, match="empty"):
            duckdb.ColumnExpression("t", "")

    def test_star_exclude_and_rename_take_any_name(self, con: duckdb.DuckDBPyConnection) -> None:
        assert rows(con, star(exclude=["id"]), "dotted") == [(21,)]
        assert rows(con, star(exclude=["hello.world"]), "dotted") == [(1,)]
        assert rows(con, star(exclude="hello.world"), "dotted") == [(1,)]
        assert rows(con, duckdb.StarExpression(exclude=["hello.world"]), "dotted") == [(1,)]
        renamed = con.execute(f"SELECT {star(rename={'hello.world': 'h'})} FROM dotted")
        assert [d[0] for d in renamed.description] == ["h", "id"]

    def test_star_rename_takes_pairs_but_no_string(self) -> None:
        assert star(rename=[("a", "b")]).fragment() == '* RENAME ("a" AS "b")'
        with pytest.raises(TypeError, match="old name to new"):
            star(rename="ab")


class TestFunctionNames:
    def test_function_names_are_written_as_the_engine_writes_identifiers(self, con: duckdb.DuckDBPyConnection) -> None:
        assert fn("sum", col("x")).fragment() == 'sum("x")'
        assert fn("my func", col("x")).fragment() == '"my func"("x")'
        assert fn("my.func", col("x")).fragment() == '"my.func"("x")'
        assert fn(("util", "f"), col("x")).fragment() == 'util.f("x")'
        assert col("s").str().upper().fragment() == 'upper("s")'
        assert con.execute(f"SELECT {fn(('s', 'twice'), lit(1) + 3)}").fetchall() == [(8,)]

    def test_a_keyword_as_a_function_name_is_quoted(self, con: duckdb.DuckDBPyConnection) -> None:
        assert fn("select", col("x")).fragment() == '"select"("x")'
        assert fn("filter", col("x")).fragment() == '"filter"("x")'
        con.execute('CREATE MACRO "select"(x) AS x + 1')
        assert con.execute(f"SELECT {fn('select', lit(1))}").fetchall() == [(2,)]

    def test_an_operator_is_a_quoted_function_name(self, con: duckdb.DuckDBPyConnection) -> None:
        assert function_name("-") == '"-"'
        assert con.execute(f"SELECT {duckdb.FunctionExpression('-', 5, 2)}").fetchall() == [(3,)]

    def test_coalesce_is_syntax_not_a_function(self, con: duckdb.DuckDBPyConnection) -> None:
        assert coalesce(col("v"), 0).fragment() == 'COALESCE("v", 0)'
        assert rows(con, coalesce(col("v"), 0), "(SELECT NULL::INTEGER AS v)") == [(0,)]

    def test_names_are_validated_when_built(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            fn(("s", ""), 1)
        with pytest.raises(TypeError, match="tuple of strings"):
            fn(("s", 1), 1)


class TestBracketsReachIntoValues:
    def test_struct_list_and_map(self, con: duckdb.DuckDBPyConnection) -> None:
        source = "(SELECT {'a': {'b': 1}, 'x.y': 5} AS st, [10, 20] AS xs, map {'k': 7} AS m)"
        picked = [col("st")["a"]["b"], col("st")["x.y"], col("xs")[1], col("m")["k"]]
        assert rows(con, ", ".join(map(str, picked)), source) == [(1, 5, 10, 7)]
        assert col("st")["a"]["b"].fragment() == "\"st\"['a']['b']"
        assert col("xs")[1].fragment() == '"xs"[1]'

    def test_keys_are_names_or_positions(self) -> None:
        with pytest.raises(TypeError, match="field name or a position"):
            col("x")[1.5]
        with pytest.raises(TypeError, match="field name or a position"):
            col("x")[True]

    def test_an_expression_is_not_a_sequence(self) -> None:
        with pytest.raises(TypeError, match=r"not iterable|not a container or iterable"):
            list(col("x"))
        with pytest.raises(TypeError, match=r"not iterable|not a container or iterable"):
            assert "a" in col("x")

    def test_an_expression_pickles(self) -> None:
        expression = (col("st")["a"] == "x") & duckdb.ColumnExpression("t", "b").isin(1, 2)
        assert str(pickle.loads(pickle.dumps(expression))) == str(expression)


class TestJoinSidesAreTheConditionsArguments:
    def test_a_dotted_side_name_is_a_column_now(self, con: duckdb.DuckDBPyConnection) -> None:
        source = "(SELECT 1 AS id) l JOIN (SELECT 1 AS id) r"
        with pytest.raises(duckdb.Error, match=r"l\.id"):
            con.execute(f"SELECT * FROM {source} ON {col('l.id') == col('r.id')}")

    def test_raw_sql_names_the_sides_itself(self, con: duckdb.DuckDBPyConnection) -> None:
        source = "(SELECT 1 AS id) l JOIN (SELECT 1 AS id) r"
        assert con.execute(f"SELECT * FROM {source} ON {duckdb.sql_expr('l.id = r.id')}").fetchall() == [(1, 1)]


class TestNamesCompareAsTheEngineDoes:
    def test_only_ascii_case_folds(self) -> None:
        assert fold_name("PRICE.USD") == "price.usd"
        assert fold_name("STRASSE") == "strasse"
        assert fold_name("straße") == "straße"
        assert fold_name("ÉCOLE") == "École".replace("É", "É")


class TestFloatsAreDoubles:
    @pytest.mark.parametrize("value", [1.5, 0.1, [1.5], {"a": 1.5}], ids=["one", "tenth", "list", "struct"])
    def test_a_float_is_a_double_written_in(self, con: duckdb.DuckDBPyConnection, value: object) -> None:
        declared = con.execute(f"SELECT typeof({lit(value)})").fetchall()[0][0]
        assert "DOUBLE" in declared, (value, declared)
        got = con.execute(f"SELECT {lit(value)}").fetchall()[0][0]
        leaf = got[0] if isinstance(got, list) else got["a"] if isinstance(got, dict) else got
        assert isinstance(leaf, float), (value, got)

    def test_a_float_operand_is_a_double(self) -> None:
        assert (col("p") * 1.21).fragment() == '("p" * 1.21::DOUBLE)'

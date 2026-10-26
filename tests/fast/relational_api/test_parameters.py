"""con.sql(text, params=...): a lazy relation whose schema is typed from the given values and whose reads bind them."""

import datetime
import decimal
import re

import pytest

import duckdb


@pytest.fixture
def con(tmp_path):
    connection = duckdb.connect()
    register_tick(connection)
    connection.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')) v(i, x)")
    connection.execute("CREATE MACRO plus_one(a) AS a + 1")
    connection.execute(
        "CREATE TABLE dated AS SELECT * FROM (VALUES (1, DATE '2026-01-01'), (2, DATE '2026-06-01')) v(i, d)"
    )
    connection.execute("CREATE MACRO first_n(n) AS TABLE SELECT range AS r FROM range(n)")
    (tmp_path / "in.csv").write_text("i,s\n1,a\n2,b\n")
    yield connection
    TICKS.pop(id(connection), None)
    connection.close()


# A read is observed through a Python function that counts its calls, which the engine does not count as a write.
TICKS = {}


def register_tick(connection):
    calls = []

    def tick() -> int:
        calls.append(1)
        return len(calls)

    connection.create_function("tick", tick, [], "BIGINT", side_effects=True)
    TICKS[id(connection)] = calls


def ran(con):
    return bool(TICKS[id(con)])


def runs(con):
    return len(TICKS[id(con)])


def executed(con, sql, params):
    cursor = con.execute(sql, params)
    description = [(column[0], str(column[1])) for column in cursor.description]
    return description, cursor.fetchall()


def schema(rel):
    return [(name, str(kind)) for name, kind in zip(rel.columns, rel.types, strict=True)]


def rows(result):
    return sorted(result, key=repr)


class TestReadsAreLazy:
    @pytest.mark.parametrize(
        ("sql", "params"),
        [
            pytest.param("SELECT tick() AS v, ? AS p", [1], id="question_mark"),
            pytest.param("SELECT tick() AS v, $1 AS p", [1], id="positional"),
            pytest.param("SELECT tick() AS v, $p AS p", {"p": 1}, id="named"),
        ],
    )
    def test_every_read_binds_and_runs_again(self, con, sql, params):
        rel = con.sql(sql, params=params)
        assert not ran(con)
        assert rel.fetchall() == [(1, 1)]
        assert rel.fetchall() == [(2, 1)]
        assert rel.filter("p = 1").fetchall() == [(3, 1)]

    def test_a_read_sees_rows_added_after_the_call(self, con):
        rel = con.sql("SELECT i FROM t WHERE i > ? ORDER BY i", params=[2])
        con.execute("INSERT INTO t VALUES (5, 'e')")
        assert rel.fetchall() == con.execute("SELECT i FROM t WHERE i > 2 ORDER BY i").fetchall()

    def test_an_error_in_the_values_raises_at_the_read(self, con):
        rel = con.sql("SELECT i, error(?) AS e FROM t", params=["boom"])
        assert rel.columns == ["i", "e"]
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            rel.fetchall()


class TestTheSchemaIsTypedFromTheValues:
    @pytest.mark.parametrize(
        ("sql", "params"),
        [
            pytest.param("SELECT ? AS p, ? + 1 AS q", [1, 2], id="integers"),
            pytest.param("SELECT $1 AS p, $1 + 1 AS q", [1], id="one_value_used_twice"),
            pytest.param("SELECT $a AS p, $b AS q", {"a": 1, "b": "x"}, id="named"),
            pytest.param("SELECT ? AS p", [2**40], id="bigint"),
            pytest.param("SELECT ? AS p", [1.5], id="double"),
            pytest.param("SELECT ? AS p", ["x"], id="varchar"),
            pytest.param("SELECT ? AS p", [True], id="boolean"),
            pytest.param("SELECT ? AS p", [None], id="null"),
            pytest.param("SELECT ? AS p", [[1, 2]], id="list"),
            pytest.param("SELECT ? AS p", [{"a": 1}], id="struct"),
            pytest.param("SELECT ? AS p", [datetime.date(2026, 10, 18)], id="date"),
            pytest.param("SELECT ? AS p", [decimal.Decimal("1.25")], id="decimal"),
            pytest.param("SELECT ?::VARCHAR AS p", [1], id="explicit_cast"),
            pytest.param("SELECT i + ? AS v FROM t", [1], id="arithmetic"),
            pytest.param("SELECT list_value(?, 2) AS l", [1], id="list_value"),
            pytest.param("SELECT plus_one(?) AS r", [1], id="scalar_macro"),
            pytest.param("SELECT * FROM first_n(?)", [3], id="table_macro"),
            pytest.param("SELECT * FROM range(?)", [3], id="table_function"),
            pytest.param("SELECT i FROM t WHERE i = any(?)", [[1, 2]], id="any"),
            pytest.param("SELECT i FROM t WHERE i IN (SELECT ?)", [1], id="in_subquery"),
            pytest.param("SELECT $1 AS a, $1::INTEGER AS b", [1], id="cast_and_bare"),
            pytest.param("SELECT * FROM t WHERE i < ? LIMIT ?", [3, 1], id="limit"),
            pytest.param("SELECT i FROM dated WHERE d > ?", ["2026-03-01"], id="date_column_and_string"),
            pytest.param("SELECT ?[1] AS e", [[5, 6]], id="list_element"),
        ],
    )
    def test_names_and_types_match_executing_the_text(self, con, sql, params):
        rel = con.sql(sql, params=params)
        expected_schema, expected_rows = executed(con, sql, params)
        assert schema(rel) == expected_schema
        assert rows(rel.fetchall()) == rows(expected_rows)

    def test_a_file_named_by_a_value_is_read(self, con, tmp_path):
        path = str(tmp_path / "in.csv")
        rel = con.sql("SELECT * FROM read_csv(?)", params=[path])
        assert (schema(rel), rel.fetchall()) == executed(con, "SELECT * FROM read_csv(?)", [path])

    @pytest.mark.parametrize(
        ("sql", "params"),
        [
            pytest.param("SELECT ? AS p, ? + 1 AS q, tick() AS n", [1, 2], id="integers"),
            pytest.param("SELECT i + $1 AS v, tick() AS n FROM t", [1], id="arithmetic"),
            pytest.param("SELECT r, tick() AS n FROM range(?) AS g(r)", [3], id="table_function"),
            pytest.param("SELECT r, tick() AS n FROM first_n(?)", [3], id="table_macro"),
            pytest.param("SELECT i, tick() AS n FROM t WHERE i = any(?)", [[1, 2]], id="any"),
            pytest.param("SELECT $1 AS a, $1::INTEGER AS b, tick() AS n", [1], id="cast_and_bare"),
            pytest.param("SELECT s, tick() AS n FROM read_csv($path)", None, id="file"),
        ],
    )
    def test_the_schema_is_known_before_any_read(self, con, tmp_path, sql, params):
        if params is None:
            params = {"path": str(tmp_path / "in.csv")}
        rel = con.sql(sql, params=params)
        names, types, description = rel.columns, rel.types, rel.description
        assert not ran(con)
        expected_schema, _ = executed(con, sql, params)
        assert [(name, str(kind)) for name, kind in zip(names, types, strict=True)] == expected_schema
        assert [(column[0], str(column[1])) for column in description] == expected_schema


class TestMarkersInTheText:
    @pytest.mark.parametrize(
        ("sql", "params", "expected"),
        [
            pytest.param("SELECT '$1 ? $a' AS s, ? AS p", [5], [("$1 ? $a", 5)], id="string_constant"),
            pytest.param("SELECT ? AS p /* why? */", [5], [(5,)], id="comment"),
            pytest.param('SELECT ? AS "what?"', [5], [(5,)], id="quoted_name"),
            pytest.param("SELECT $$ $1 $$ AS s, $1 AS p", [5], [(" $1 ", 5)], id="dollar_quoted"),
        ],
    )
    def test_only_real_markers_take_values(self, con, sql, params, expected):
        assert con.sql(sql, params=params).fetchall() == expected

    def test_a_marker_inside_a_cte_of_the_text(self, con):
        sql = "WITH c AS (SELECT i FROM t WHERE i < ?) SELECT i FROM c WHERE i > ? ORDER BY i"
        assert con.sql(sql, params=[4, 1]).fetchall() == con.execute(sql, [4, 1]).fetchall() == [(2,), (3,)]

    def test_named_markers_inside_ctes(self, con):
        sql = "WITH c AS (SELECT i FROM t WHERE i < $hi) SELECT i FROM c WHERE i > $lo ORDER BY i"
        params = {"lo": 1, "hi": 4}
        assert con.sql(sql, params=params).fetchall() == con.execute(sql, params).fetchall() == [(2,), (3,)]


class TestTheNamesOfAParameterisedSource:
    """The names are those of the statement that runs, markers and all, while the types come from the values."""

    def source(self, con):
        return con.sql("SELECT ? + 1, 'a' || ?", params=[1, "b"])

    def test_the_columns_are_the_names_of_the_running_statement(self, con):
        rel = self.source(con)
        assert rel.columns == ["($1 + 1)", "('a' || $2)"] == list(rel.df().columns)
        assert [str(kind) for kind in rel.types] == ["INTEGER", "VARCHAR"]

    def test_a_column_is_reached_by_that_name(self, con):
        assert self.source(con).project('"($1 + 1)"').fetchall() == [(2,)]

    def test_a_view_names_its_columns_so_too(self, con):
        self.source(con).create_view("named")
        cursor = con.execute("SELECT * FROM named")
        assert [column[0] for column in cursor.description] == ["($1 + 1)", "('a' || $2)"]
        assert cursor.fetchall() == [(2, "ab")]


class TestParametersInAPivot:
    def test_a_marker_in_the_on_expression(self, con):
        text = "PIVOT (SELECT 1 AS i) ON (i + ?) IN (2) USING count(*)"
        left = con.sql(text, params=[1])
        assert left.columns == ["2"]
        assert left.cross(con.sql(text, params=[2])).fetchall() == [(1, 0)]

    def test_a_marker_in_the_in_list(self, con):
        text = "PIVOT (SELECT 1 AS k, 5 AS c UNION ALL SELECT 2, 6) ON c IN (?) USING sum(k)"
        pivoted = con.sql(text, params=[5])
        assert pivoted.columns == ["5"]
        assert pivoted.fetchall() == [(1,)]
        assert con.sql("SELECT ?::INT AS z", params=[1]).cross(pivoted).fetchall() == [(1, 1)]


class TestStepsOverAParameterisedRelation:
    def test_a_filter_and_a_limit(self, con):
        rel = con.sql("SELECT i, x FROM t WHERE i > ?", params=[1]).filter("i < 4").order("i").limit(1)
        assert rel.fetchall() == con.execute("SELECT i, x FROM t WHERE i > 1 AND i < 4 ORDER BY i LIMIT 1").fetchall()

    @pytest.mark.parametrize(
        ("left", "left_params", "right", "right_params"),
        [
            pytest.param("SELECT ? AS a, ? AS b", [1, 2], "SELECT ? AS c", [3], id="question_marks"),
            pytest.param("SELECT $1 AS a, $2 AS b", [1, 2], "SELECT $1 AS c", [3], id="positional"),
            pytest.param("SELECT $2 AS a, $1 AS b", [2, 1], "SELECT $1 AS c", [3], id="positional_out_of_order"),
            pytest.param("SELECT $x AS a, $y AS b", {"x": 1, "y": 2}, "SELECT $z AS c", {"z": 3}, id="named"),
        ],
    )
    def test_two_parameterised_relations_combine(self, con, left, left_params, right, right_params):
        joined = con.sql(left, params=left_params).cross(con.sql(right, params=right_params))
        assert joined.fetchall() == [(1, 2, 3)]
        assert joined.columns == ["a", "b", "c"]

    def test_two_parameterised_relations_combine_lazily(self, con):
        left = con.sql("SELECT $1 AS a, tick() AS n", params=[1])
        right = con.sql("SELECT $1 AS c", params=[3])
        joined = left.cross(right)
        assert not ran(con)
        assert joined.fetchall() == [(1, 1, 3)]
        assert joined.fetchall() == [(1, 2, 3)]

    def test_a_union_of_two_parameterised_relations_reads_both_values(self, con):
        left = con.sql("SELECT ? AS a, tick() AS n", params=[1])
        right = con.sql("SELECT ? AS a, tick() AS n", params=[2])
        union = left.union(right)
        assert not ran(con)
        assert sorted(row[0] for row in union.fetchall()) == [1, 2]
        assert runs(con) == 2

    def test_a_join_with_parameters_on_both_sides(self, con):
        left = con.sql("SELECT * FROM t WHERE i > ?", params=[1]).set_alias("a")
        right = con.sql("SELECT * FROM t WHERE i < ?", params=[4]).set_alias("b")
        joined = left.join(right, "a.i = b.i")
        expected = con.execute(
            "SELECT * FROM (FROM t WHERE i > 1) a JOIN (FROM t WHERE i < 4) b ON a.i = b.i"
        ).fetchall()
        assert rows(joined.fetchall()) == rows(expected)

    def test_the_same_name_on_both_sides_is_refused(self, con):
        left = con.sql("SELECT $x AS a", params={"x": 1})
        right = con.sql("SELECT $x AS b", params={"x": 2})
        with pytest.raises(duckdb.InvalidInputException):
            left.cross(right)


class TestARelationWithNamedParametersCombinedWithItself:
    """One relation used twice declares its names once, so it combines with itself where two relations cannot."""

    @pytest.fixture
    def low(self, con):
        return con.sql("SELECT range AS i FROM range(5) WHERE range < $low", params={"low": 3})

    def test_a_union_with_itself(self, con, low):
        expected = con.execute(
            "SELECT range FROM range(5) WHERE range < 3 UNION ALL SELECT range FROM range(5) WHERE range < 3"
        ).fetchall()
        assert sorted(low.union(low).fetchall()) == sorted(expected)

    def test_a_self_join(self, low):
        joined = low.set_alias("a").join(low.set_alias("b"), "a.i = b.i")
        assert sorted(joined.fetchall()) == [(0, 0), (1, 1), (2, 2)]


class TestParameterNamesFoldAsTheEngineFoldsThem:
    """A named parameter is an identifier, so `$P` and `$p` are one name to the engine and to the relation."""

    def test_two_relations_whose_names_differ_only_in_case_are_refused(self, con):
        left = con.sql("SELECT $P AS v", params={"P": 1})
        right = con.sql("SELECT $p AS v", params={"p": 2})
        with pytest.raises(duckdb.InvalidInputException):
            left.union(right).fetchall()

    def test_one_mapping_with_both_spellings_is_refused_at_the_call(self, con):
        # The mapping gives one engine parameter two values, so it is refused rather than bound to one.
        with pytest.raises(duckdb.InvalidInputException) as error:
            con.sql("SELECT $P AS upper_value, $p AS lower_value", params={"P": 1, "p": 2})
        message = str(error.value)
        assert re.search(r"\bP\b", message)
        assert re.search(r"\bp\b", message)

    def test_one_key_answers_both_spellings_as_the_engine_does(self, con):
        expected = con.execute("SELECT $P AS a, $p AS b", {"p": 1}).fetchall()
        assert con.sql("SELECT $P AS a, $p AS b", params={"p": 1}).fetchall() == expected == [(1, 1)]

    def test_one_relation_reads_both_spellings_as_one_parameter(self, con):
        expected = con.execute("SELECT $P AS a, $p AS b", {"P": 1}).fetchall()
        assert con.sql("SELECT $P AS a, $p AS b", params={"P": 1}).fetchall() == expected == [(1, 1)]


class TestTheMarkerGrammarIsTheEngines:
    def test_a_numbered_question_mark_is_that_positional_parameter(self, con):
        text = "SELECT ?1 + ?1 AS s"
        assert con.sql(text, params=[7]).fetchall() == con.execute(text, [7]).fetchall() == [(14,)]

    def test_numbered_question_marks_of_two_relations_each_read_their_own(self, con):
        left = con.sql("SELECT ?1 AS a", params=[1])
        right = con.sql("SELECT ?1 + ?1 AS b", params=[20])
        assert left.cross(right).fetchall() == [(1, 40)]


class TestValuesMustMatchTheMarkers:
    @pytest.mark.parametrize(
        ("sql", "params"),
        [
            pytest.param("SELECT ? AS p, ? AS q", [1], id="missing_positional"),
            pytest.param("SELECT ? AS p", [1, 2], id="unused_positional"),
            pytest.param("SELECT $a AS p, $b AS q", {"a": 1}, id="missing_named"),
            pytest.param("SELECT $a AS p", {"a": 1, "b": 2}, id="unused_named"),
        ],
    )
    def test_the_call_refuses_them(self, con, sql, params):
        with pytest.raises(duckdb.InvalidInputException):
            con.sql(sql, params=params)

    @pytest.mark.parametrize(
        ("sql", "params", "error"),
        [
            pytest.param("SELECT 1 AS p", [1], duckdb.BinderException, id="text_without_markers"),
            pytest.param("SELECT 1 AS p", {"a": 1}, duckdb.BinderException, id="text_without_markers_named"),
            pytest.param("SELECT ? AS a, $b AS b", [1], duckdb.NotImplementedException, id="mixed_markers"),
            pytest.param("SELECT ? AS a, $b AS b", {"b": 1}, duckdb.NotImplementedException, id="mixed_markers_named"),
        ],
    )
    def test_the_call_refuses_markers_that_cannot_take_them(self, con, sql, params, error):
        with pytest.raises(error):
            con.sql(sql, params=params)


class TestValuesAreCopiedAtTheCall:
    def test_changing_the_list_after_the_call(self, con):
        values = [1, [1, 2]]
        rel = con.sql("SELECT ? AS p, ? AS l", params=values)
        values[0] = 9
        values[1].append(3)
        assert rel.fetchall() == [(1, [1, 2])]

    def test_changing_the_dict_after_the_call(self, con):
        values = {"p": 1}
        rel = con.sql("SELECT $p AS p", params=values)
        values["p"] = 9
        assert rel.fetchall() == [(1,)]

    def test_every_read_binds_the_values_of_the_call(self, con):
        values = [1]
        rel = con.sql("SELECT ? AS p, tick() AS n", params=values)
        values[0] = 9
        assert rel.fetchall() == [(1, 1)]
        assert rel.fetchall() == [(1, 2)]


class TestAParameterisedRelationAsASource:
    """The values are written in, so the rows still come from the table as it is at the read."""

    def numbers(self, con):
        con.execute("CREATE TABLE numbers AS SELECT * FROM (VALUES (1), (2), (5)) v(a)")
        return con.sql("SELECT a FROM numbers WHERE a < ? ORDER BY a", params=[4])

    def test_map(self, con):
        rel = self.numbers(con).map(lambda df: df)
        con.execute("INSERT INTO numbers VALUES (3)")
        assert sorted(rel.fetchall()) == [(1,), (2,), (3,)]

    def test_register(self, con):
        con.register("few", self.numbers(con))
        con.execute("INSERT INTO numbers VALUES (3)")
        assert con.sql("SELECT a FROM few ORDER BY a").fetchall() == [(1,), (2,), (3,)]

    def test_read_by_name(self, con):
        few = self.numbers(con)  # noqa: F841
        con.execute("INSERT INTO numbers VALUES (3)")
        assert con.sql("SELECT a FROM few ORDER BY a").fetchall() == [(1,), (2,), (3,)]

    def test_the_values_stay_the_values_of_the_call(self, con):
        few = self.numbers(con)  # noqa: F841
        con.execute("INSERT INTO numbers VALUES (4), (6)")
        assert con.sql("SELECT a FROM few ORDER BY a").fetchall() == [(1,), (2,)]


class TestViewsOverAParameterisedRelation:
    def test_a_view_reads_the_table_with_the_values_of_the_call(self, con):
        values = [2]
        rel = con.sql("SELECT i FROM t WHERE i > ? ORDER BY i", params=values)
        values[0] = 0
        rel.create_view("w")
        con.execute("INSERT INTO t VALUES (5, 'e')")
        assert con.sql("SELECT i FROM w ORDER BY i").fetchall() == [(3,), (4,), (5,)]

    def test_a_view_reads_the_rows_of_the_values(self, con):
        con.sql("SELECT i FROM t WHERE i > ?", params=[2]).create_view("w")
        assert con.sql("SELECT i FROM w ORDER BY i").fetchall() == [(3,), (4,)]

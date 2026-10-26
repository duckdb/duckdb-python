"""The SQL text of a relation: one SELECT until a step closes it, runnable as is on its connection."""

import json
import re

import pytest

import duckdb
from duckdb import ColumnExpression, ConstantExpression

# The engine's error and nothing the library appends: no wording about scope or closed SELECTs anywhere in it.
ENGINE_ONLY = r"(?s)\A(?!.*(?:out of scope|opened a new SELECT|qualify by|closed SELECT))"


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c'), (3, 'c')) v(i, x)")
    connection.execute("CREATE TABLE r AS SELECT * FROM (VALUES (2, 'B'), (3, 'C'), (4, 'D')) v(i, y)")
    connection.execute("CREATE TABLE pairs AS SELECT * FROM (VALUES (1, 10), (1, 20)) v(x, y)")
    yield connection
    connection.close()


def parsed(con, sql):
    tree = json.loads(con.execute("SELECT json_serialize_sql(?)", [sql]).fetchall()[0][0])
    assert not tree["error"], tree
    [statement] = tree["statements"]
    return statement["node"]


def walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def distinct_of(node):
    for modifier in node["modifiers"]:
        if modifier["type"] == "DISTINCT_MODIFIER":
            return modifier
    return None


def ctes(node):
    return {entry["key"]: entry["value"]["query_node"] for entry in node["cte_map"]["map"]}


def table_names(node):
    return [part["table_name"] for part in walk(node) if part.get("type") == "BASE_TABLE"]


def column_refs(node):
    return [part["column_names"] for part in node["select_list"] if part.get("class") == "COLUMN_REF"]


def without_locations(node):
    if isinstance(node, dict):
        return {k: without_locations(v) for k, v in node.items() if not k.startswith("query_location")}
    if isinstance(node, list):
        return [without_locations(v) for v in node]
    return node


def rows(result):
    return sorted(result, key=repr)


def assert_runs_as(con, rel, direct, *, ordered=False):
    """The rendered text and the relation both answer what the SQL answers, row by row when it is ordered."""
    answer = con.execute(direct).fetchall()
    if ordered:
        assert con.sql(rel.sql_query()).fetchall() == answer
        assert rel.fetchall() == answer
    else:
        assert rows(con.sql(rel.sql_query()).fetchall()) == rows(answer)
        assert rows(rel.fetchall()) == rows(answer)


def chain(con):
    return con.sql("SELECT range AS a FROM range(5)").filter("a > 1").project("a + 1 AS b")


def join_project(con):
    return con.table("t").join(con.table("r"), "t.i = r.i").project("t.x, r.y")


def join_filter_order_project(con):
    return con.table("t").join(con.table("r"), "t.i = r.i").filter("t.i > 2").project("t.x, r.y").order("y")


class TestTheTextRuns:
    def test_a_source(self, con):
        rel = con.sql("SELECT 1 AS a, 'x' AS b")
        assert_runs_as(con, rel, "SELECT 1 AS a, 'x' AS b")

    def test_a_chain(self, con):
        assert_runs_as(con, chain(con), "SELECT a + 1 AS b FROM range(5) r(a) WHERE a > 1")

    def test_a_join_with_a_projection(self, con):
        assert_runs_as(con, join_project(con), "SELECT t.x, r.y FROM t JOIN r ON t.i = r.i")

    def test_a_join_with_a_filter_order_and_projection(self, con):
        assert_runs_as(
            con,
            join_filter_order_project(con),
            "SELECT t.x, r.y FROM t JOIN r ON t.i = r.i WHERE t.i > 2 ORDER BY r.y",
            ordered=True,
        )

    def test_a_limit_then_distinct_over_a_join(self, con):
        rel = con.table("t").join(con.table("r"), "t.i = r.i").order("x DESC, y").limit(2).distinct()
        assert_runs_as(
            con,
            rel,
            "SELECT DISTINCT * FROM (SELECT t.i AS ti, t.x, r.i AS ri, r.y FROM t JOIN r ON t.i = r.i "
            "ORDER BY x DESC, y LIMIT 2)",
        )

    def test_a_set_operation_over_joins(self, con):
        joined = con.table("t").join(con.table("r"), "t.i = r.i")
        rel = joined.union(joined).project("x")
        assert_runs_as(con, rel, "SELECT x FROM t JOIN r ON t.i = r.i UNION ALL SELECT x FROM t JOIN r ON t.i = r.i")

    def test_set_alias_over_a_join_names_the_result(self, con):
        rel = con.table("t").join(con.table("r"), "t.i = r.i").set_alias("q").project("q.x")
        assert_runs_as(con, rel, "SELECT q.x FROM (SELECT * FROM t JOIN r ON t.i = r.i) AS q")

    def test_a_step_reads_its_input_under_the_input_alias(self, con):
        rel = con.table("t").filter("i > 1").filter("t.i < 3")
        assert_runs_as(con, rel, "SELECT * FROM t WHERE i > 1 AND t.i < 3")

    def test_literals_are_written_in(self, con):
        rel = con.table("t").filter(ColumnExpression("x") == ConstantExpression("b"))
        node = parsed(con, rel.sql_query())
        assert not [part for part in walk(node) if part.get("class") == "PARAMETER"]
        assert_runs_as(con, rel, "SELECT * FROM t WHERE x = 'b'")


class TestTheShape:
    def test_a_source_renders_its_own_text(self, con):
        text = "SELECT 1 AS a, 'x' AS b"
        assert without_locations(parsed(con, con.sql(text).sql_query())) == without_locations(parsed(con, text))

    def test_a_table_renders_as_a_read_of_its_name(self, con):
        node = parsed(con, con.table("t").sql_query())
        assert ctes(node) == {}
        assert node["from_table"]["type"] == "BASE_TABLE"
        assert node["from_table"]["table_name"] == "t"

    def test_a_chain_in_clause_order_fills_one_select(self, con):
        # The filter and the projection append to one SELECT, so one SELECT holds both and none holds either alone.
        node = parsed(con, chain(con).sql_query())
        filtered = [part for part in walk(node) if part.get("type") == "SELECT_NODE" and part["where_clause"]]
        assert len(filtered) == 1
        [projection] = filtered[0]["select_list"]
        assert projection["alias"] == "b"

    def test_steps_are_named_in_dependency_order(self, con):
        left = con.sql("SELECT range AS a FROM range(5)").filter("a > 0")
        right = con.sql("SELECT range AS b FROM range(3)").project("b * 2 AS b")
        rel = left.set_alias("l").join(right.set_alias("r"), "l.a = r.b").filter("l.a < 4")
        node = parsed(con, rel.sql_query())
        names = list(ctes(node))
        # Both inputs have a filled slot, so each is a closed SELECT the join reads as a CTE.
        assert len(names) >= 2
        assert names == [f"__duckdb_s{n}" for n in range(len(names))]
        for position, name in enumerate(names):
            assert set(table_names(ctes(node)[name])) <= set(names[:position])
        assert_runs_as(
            con,
            rel,
            "SELECT * FROM (SELECT * FROM range(5) t(a) WHERE a > 0) l "
            "JOIN (SELECT b * 2 AS b FROM range(3) t(b)) r ON l.a = r.b WHERE l.a < 4",
        )

    def test_a_step_used_twice_is_rendered_once(self, con):
        shared = con.sql("SELECT range AS a FROM range(5)").filter("a > 1")
        rel = shared.union(shared)
        node = parsed(con, rel.sql_query())
        assert len([part for part in walk(node) if part.get("where_clause")]) == 1
        assert_runs_as(
            con, rel, "SELECT * FROM range(5) t(a) WHERE a > 1 UNION ALL SELECT * FROM range(5) t(a) WHERE a > 1"
        )

    def test_a_step_used_by_both_sides_of_a_self_join_is_rendered_once(self, con):
        shared = con.table("t").filter("i > 1")
        rel = shared.set_alias("a").join(shared.set_alias("b"), "a.i = b.i")
        node = parsed(con, rel.sql_query())
        assert len([part for part in walk(node) if part.get("where_clause")]) == 1
        assert_runs_as(con, rel, "SELECT * FROM (FROM t WHERE i > 1) a JOIN (FROM t WHERE i > 1) b ON a.i = b.i")

    def test_a_volatile_step_used_twice_runs_for_each_use(self, con):
        source = con.sql("SELECT random() AS v FROM range(50)").aggregate("sum(v) AS v")
        rel = source.set_alias("a").cross(source.set_alias("b")).project("a.v = b.v AS same")
        assert rel.fetchall() == [(False,)]
        assert con.sql(rel.sql_query()).fetchall() == [(False,)]

    def test_a_shared_random_step_differs_per_use(self, con):
        shared = con.sql("SELECT random() AS r FROM range(1)").filter("r >= 0")
        rel = shared.union(shared)
        assert len(set(rel.fetchall())) == 2
        assert len(set(con.sql(rel.sql_query()).fetchall())) == 2

    def test_a_projection_over_a_join_renders_into_the_joins_select(self, con):
        node = parsed(con, join_project(con).sql_query())
        assert not [body for body in ctes(node).values() if body["from_table"]["type"] == "JOIN"]
        join = node["from_table"]
        assert join["type"] == "JOIN"
        assert (join["left"]["alias"], join["right"]["alias"]) == ("t", "r")
        assert (join["left"]["table_name"], join["right"]["table_name"]) == ("t", "r")
        assert column_refs(node) == [["t", "x"], ["r", "y"]]

    def test_a_filter_and_projection_over_a_join_render_into_its_select_and_the_order_over_it(self, con):
        # The ORDER BY after the select list closes the join's SELECT and orders the one over it.
        node = parsed(con, join_filter_order_project(con).sql_query())
        assert [modifier["type"] for modifier in node["modifiers"]] == ["ORDER_MODIFIER"]
        assert node["from_table"]["type"] == "BASE_TABLE"
        [join] = [body for body in ctes(node).values() if body["from_table"]["type"] == "JOIN"]
        assert join["where_clause"] is not None
        assert join["modifiers"] == []
        assert column_refs(join) == [["t", "x"], ["r", "y"]]

    def test_an_input_alias_in_the_order_after_the_projection_is_the_engine_s_error(self, con):
        # The ORDER BY closed the SELECT, so r.y is out of scope.
        joined = con.table("t").join(con.table("r"), "t.i = r.i").filter("t.i > 2").project("t.x, r.y")
        with pytest.raises(duckdb.BinderException, match=ENGINE_ONLY + r'.*Referenced table "r" not found'):
            joined.order("r.y")

    def test_a_set_operation_over_a_join_starts_a_new_cte(self, con):
        joined = con.table("t").join(con.table("r"), "t.i = r.i")
        node = parsed(con, joined.union(joined).project("x").sql_query())
        set_operations = [name for name, body in ctes(node).items() if body["type"] == "SET_OPERATION_NODE"]
        assert len(set_operations) == 1
        assert table_names(node["from_table"]) == set_operations

    def test_one_relation_renders_the_same_text_every_time(self, con):
        rel = chain(con)
        assert rel.sql_query() == rel.sql_query()

    def test_the_same_calls_render_the_same_text_up_to_the_unnamed_alias(self, con):
        # An unnamed relation's alias and a join's are unique per relation object, so builds differ there.
        def unaliased(text):
            return re.sub(r"(unnamed_relation|JOIN_RELATION)_[0-9a-f]+", r"\1", text)

        assert unaliased(chain(con).sql_query()) == unaliased(chain(con).sql_query())
        assert unaliased(join_filter_order_project(con).sql_query()) == unaliased(
            join_filter_order_project(con).sql_query()
        )

    def test_user_parameters_stay_markers(self, con):
        rel = con.sql("SELECT * FROM t WHERE i > ?", params=[1]).filter("i < 3")
        node = parsed(con, rel.sql_query())
        assert len([part for part in walk(node) if part.get("class") == "PARAMETER"]) == 1
        assert rows(con.execute(rel.sql_query(), [1]).fetchall()) == rows(rel.fetchall()) == [(2, "b")]


class TestDistinctAndProjectionOrder:
    """The two orders answer differently.

    `distinct()` then `project()` removes duplicates of the input row, the other order removes duplicates of the
    projection, so the text has to say which select list the DISTINCT reads.
    """

    def pair(self, con, source):
        return source.distinct().project("x"), source.project("x").distinct()

    def joined(self, con):
        left = con.sql("SELECT * FROM (VALUES (1, 10), (1, 20)) AS t(x, y)", alias="l")
        right = con.sql("SELECT 0 AS z", alias="r")
        return left.join(right, "true")

    def test_a_join_distinct_then_project_keeps_one_row_per_distinct_input_row(self, con):
        distinct_first, _ = self.pair(con, self.joined(con))
        expected = con.execute(
            "SELECT x FROM (SELECT DISTINCT * FROM (VALUES (1, 10), (1, 20)) AS t(x, y), (SELECT 0 AS z))"
        ).fetchall()
        assert rows(distinct_first.fetchall()) == rows(expected) == [(1,), (1,)]

    def test_a_join_project_then_distinct_keeps_one_row_per_projected_row(self, con):
        _, project_first = self.pair(con, self.joined(con))
        expected = con.execute(
            "SELECT DISTINCT x FROM (VALUES (1, 10), (1, 20)) AS t(x, y), (SELECT 0 AS z)"
        ).fetchall()
        assert rows(project_first.fetchall()) == rows(expected) == [(1,)]

    def test_a_table_distinct_then_project_keeps_one_row_per_distinct_input_row(self, con):
        distinct_first, _ = self.pair(con, con.table("pairs"))
        expected = con.execute("SELECT x FROM (SELECT DISTINCT * FROM pairs)").fetchall()
        assert rows(distinct_first.fetchall()) == rows(expected) == [(1,), (1,)]

    def test_a_table_project_then_distinct_keeps_one_row_per_projected_row(self, con):
        _, project_first = self.pair(con, con.table("pairs"))
        expected = con.execute("SELECT DISTINCT x FROM pairs").fetchall()
        assert rows(project_first.fetchall()) == rows(expected) == [(1,)]

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(lambda self, con: self.joined(con), id="join"),
            pytest.param(lambda self, con: con.table("pairs"), id="table"),
        ],
    )
    def test_the_text_says_which_list_the_distinct_reads(self, con, source):
        distinct_first, project_first = self.pair(con, source(self, con))
        assert distinct_first.sql_query() != project_first.sql_query()
        over_the_projection = distinct_of(parsed(con, project_first.sql_query()))
        assert over_the_projection is not None
        assert over_the_projection["distinct_on_targets"] == []
        assert column_refs(parsed(con, project_first.sql_query())) in ([["x"]], [])
        over_the_input = distinct_of(parsed(con, distinct_first.sql_query()))
        assert over_the_input is None or over_the_input["distinct_on_targets"] != []


def join_blocks(node):
    """Every SELECT in the statement whose FROM is a join, the top one and those inside common table expressions."""
    blocks = [node] if node["from_table"]["type"] == "JOIN" else []
    return blocks + [body for body in ctes(node).values() if body["from_table"]["type"] == "JOIN"]


class TestNoPositionalReference:
    """The names are the engine's for the rendered statement, so no SELECT re-aliases a column by its position."""

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(lambda j: j, id="join"),
            pytest.param(lambda j: j.filter("t.i > 0"), id="filter"),
            pytest.param(lambda j: j.limit(1).limit(1), id="two_limits"),
            pytest.param(lambda j: j.limit(1).filter("true"), id="limit_then_filter"),
            pytest.param(lambda j: j.set_alias("q").limit(1).project("*"), id="alias_limit_star"),
        ],
    )
    @pytest.mark.parametrize(
        "condition",
        [pytest.param("t.i = r.i", id="on"), pytest.param(["i"], id="using")],
    )
    def test_no_select_holds_a_positional_reference(self, con, step, condition):
        rel = step(con.table("t").join(con.table("r"), condition))
        node = parsed(con, rel.sql_query())
        assert join_blocks(node)
        assert [part for part in walk(node) if part.get("class") == "POSITIONAL_REFERENCE"] == []

    def test_a_step_over_the_join_s_cte_reports_the_names_the_cte_exposes(self, con):
        rel = con.table("t").join(con.table("r"), "t.i = r.i").limit(1).limit(1)
        node = parsed(con, rel.sql_query())
        assert ctes(node)
        assert node["from_table"]["type"] == "BASE_TABLE"
        assert rel.columns == ["i", "x", "i_1", "y"]

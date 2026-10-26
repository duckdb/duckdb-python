"""Joins of every kind and condition form, checked against the same join written in SQL on the same connection."""

import pytest

import duckdb
from duckdb import ColumnExpression

KINDS = [
    pytest.param("inner", "INNER", id="inner"),
    pytest.param("left", "LEFT", id="left"),
    pytest.param("right", "RIGHT", id="right"),
    pytest.param("outer", "FULL OUTER", id="outer"),
    pytest.param("semi", "SEMI", id="semi"),
    pytest.param("anti", "ANTI", id="anti"),
]


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE l AS SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (2, 'b'), (3, 'c'), (NULL, 'n')) v(i, x)"
    )
    connection.execute(
        "CREATE TABLE r AS SELECT * FROM (VALUES (2, 'B'), (3, 'C'), (3, 'C2'), (4, 'D'), (NULL, 'N')) v(i, y)"
    )
    connection.execute("CREATE TABLE m AS SELECT * FROM (VALUES ('b', 10), ('c', 20)) v(x, z)")
    connection.execute("CREATE TABLE lk AS SELECT * FROM (VALUES (1, 1, 'a'), (1, 2, 'b'), (2, 2, 'c')) v(g, i, x)")
    connection.execute("CREATE TABLE rk AS SELECT * FROM (VALUES (1, 2, 'B'), (2, 2, 'C'), (2, 3, 'D')) v(g, i, y)")
    connection.execute(
        "CREATE TABLE orders AS SELECT * FROM (VALUES (1, 'order-a'), (2, 'order-b'), (4, 'order-d')) v(id, name)"
    )
    connection.execute(
        "CREATE TABLE customers AS SELECT * FROM (VALUES (1, 'cust-a'), (2, 'cust-b'), (3, 'cust-c')) v(id, name)"
    )
    connection.execute("CREATE TABLE shared_left AS SELECT * FROM (VALUES (1, 'same'), (2, 'only-l')) v(id, name)")
    connection.execute("CREATE TABLE shared_right AS SELECT * FROM (VALUES (1, 'same'), (3, 'only-r')) v(id, name)")
    connection.execute("CREATE TABLE trades AS SELECT * FROM (VALUES (1, 10, 'a'), (2, 20, 'b')) v(id, t, x)")
    connection.execute(
        "CREATE TABLE quotes AS SELECT * FROM (VALUES (1, 5, 'A'), (2, 25, 'B'), (2, 15, 'C')) v(id, t, q)"
    )
    yield connection
    connection.close()


def rows(result):
    return sorted(result, key=repr)


def assert_same(con, rel, direct, *, ordered=False):
    """The relation answers what the SQL answers; an ordered statement is compared row by row in its own order."""
    cursor = con.execute(direct)
    names = [column[0] for column in cursor.description]
    expected = cursor.fetchall()
    assert rel.columns == names
    if ordered:
        assert rel.fetchall() == expected
    else:
        assert rows(rel.fetchall()) == rows(expected)


class TestKinds:
    @pytest.mark.parametrize(("how", "kind"), KINDS)
    def test_on_a_string_condition(self, con, how, kind):
        rel = con.table("l").join(con.table("r"), "l.i = r.i", how=how)
        assert_same(con, rel, f"SELECT * FROM l {kind} JOIN r ON l.i = r.i")

    @pytest.mark.parametrize(("how", "kind"), KINDS)
    def test_on_an_expression(self, con, how, kind):
        condition = ColumnExpression("l", "i") == ColumnExpression("r", "i")
        rel = con.table("l").join(con.table("r"), condition, how=how)
        assert_same(con, rel, f"SELECT * FROM l {kind} JOIN r ON l.i = r.i")

    @pytest.mark.parametrize(("how", "kind"), KINDS)
    def test_on_a_callable_of_the_two_sides(self, con, how, kind):
        rel = con.table("l").join(con.table("r"), lambda left, right: left["i"] == right["i"], how=how)
        assert_same(con, rel, f"SELECT * FROM l {kind} JOIN r ON l.i = r.i")

    def test_cross(self, con):
        assert_same(con, con.table("l").cross(con.table("r")), "SELECT * FROM l CROSS JOIN r")

    @pytest.mark.parametrize(
        ("how", "direct"),
        [
            pytest.param("cross", "SELECT * FROM l CROSS JOIN r", id="cross"),
            pytest.param("positional", "SELECT * FROM l POSITIONAL JOIN r", id="positional"),
            pytest.param("natural", "SELECT * FROM l NATURAL JOIN r", id="natural"),
        ],
    )
    def test_kinds_without_a_condition(self, con, how, direct):
        assert_same(con, con.table("l").join(con.table("r"), how=how), direct)

    def test_asof_on_a_condition(self, con):
        rel = con.table("trades").join(
            con.table("quotes"), "trades.id = quotes.id AND trades.t >= quotes.t", how="asof"
        )
        assert_same(con, rel, "SELECT * FROM trades ASOF JOIN quotes ON trades.id = quotes.id AND trades.t >= quotes.t")

    def test_asof_using_names(self, con):
        rel = con.table("trades").join(con.table("quotes"), ["id", "t"], how="asof")
        assert_same(con, rel, "SELECT * FROM trades ASOF JOIN quotes USING (id, t)")


class TestUsing:
    @pytest.mark.parametrize("names", [pytest.param("i", id="string"), pytest.param(["i"], id="list")])
    @pytest.mark.parametrize(("how", "kind"), KINDS)
    def test_one_name(self, con, how, kind, names):
        rel = con.table("l").join(con.table("r"), names, how=how)
        assert_same(con, rel, f"SELECT * FROM l {kind} JOIN r USING (i)")

    @pytest.mark.parametrize(("how", "kind"), KINDS)
    def test_several_names(self, con, how, kind):
        rel = con.table("lk").join(con.table("rk"), ["g", "i"], how=how)
        assert_same(con, rel, f"SELECT * FROM lk {kind} JOIN rk USING (g, i)")

    @pytest.mark.parametrize(
        ("how", "expected"),
        [
            pytest.param("left", [(1, "a"), (2, "b"), (2, "b"), (3, "c"), (3, "c"), (None, "n")], id="left"),
            pytest.param("right", [(2, "b"), (2, "b"), (3, "c"), (3, "c"), (4, None), (None, None)], id="right"),
            pytest.param(
                "outer",
                [(1, "a"), (2, "b"), (2, "b"), (3, "c"), (3, "c"), (4, None), (None, None), (None, "n")],
                id="outer",
            ),
        ],
    )
    def test_the_key_comes_from_the_side_that_has_the_row(self, con, how, expected):
        rel = con.table("l").join(con.table("r"), "i", how=how).project("i, x")
        assert rows(rel.fetchall()) == rows(expected)


class TestScopeOfTheSides:
    def test_a_self_join_through_set_alias(self, con):
        rel = con.table("l").set_alias("a").join(con.table("l").set_alias("b"), "a.i = b.i")
        assert_same(con, rel, "SELECT * FROM l AS a JOIN l AS b ON a.i = b.i")

    def test_set_alias_names_the_sides(self, con):
        rel = con.table("l").set_alias("a").join(con.table("r").set_alias("b"), "a.i = b.i").project("a.x, b.y")
        assert_same(con, rel, "SELECT a.x, b.y FROM l AS a JOIN r AS b ON a.i = b.i")

    def test_a_step_keeps_the_alias_of_its_input(self, con):
        rel = con.table("l").filter("i > 1").join(con.table("r"), "l.i = r.i")
        assert_same(con, rel, "SELECT * FROM (FROM l WHERE i > 1) AS l JOIN r ON l.i = r.i")

    def test_the_same_alias_on_both_sides_is_refused(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="same alias"):
            con.table("l").join(con.table("l"), "i")

    def test_two_relations_built_from_the_same_text_can_join(self, con):
        left = con.sql("SELECT 1 AS i, 'x' AS v")
        right = con.sql("SELECT 1 AS i, 'x' AS v")
        assert left.join(right, "i").fetchall() == [(1, "x", "x")]

    def test_a_join_of_joins(self, con):
        rel = con.table("l").join(con.table("r"), "l.i = r.i").set_alias("lr").join(con.table("m"), "lr.x = m.x")
        assert_same(con, rel, "SELECT * FROM (SELECT * FROM l JOIN r ON l.i = r.i) AS lr JOIN m ON lr.x = m.x")

    def test_a_join_over_a_join_without_an_alias(self, con):
        # A join whose input is a bare join inlines that join's FROM items, as the shipping client did.
        rel = con.table("l").join(con.table("r"), "l.i = r.i").join(con.table("m"), "x")
        assert_same(con, rel, "SELECT * FROM l JOIN r ON l.i = r.i JOIN m USING (x)")
        assert rel.columns == ["i", "x", "i", "y", "z"]

    def test_a_join_with_one_of_its_inputs(self, con):
        joined = con.table("l").join(con.table("r"), "l.i = r.i").set_alias("j")
        rel = joined.join(con.table("l"), "j.x = l.x")
        assert_same(con, rel, "SELECT * FROM (SELECT * FROM l JOIN r ON l.i = r.i) AS j JOIN l ON j.x = l.x")

    def test_both_keys_are_kept_by_the_join_and_by_a_star_over_it(self, con):
        joined = con.table("l").join(con.table("r"), "l.i = r.i")
        assert joined.columns == ["i", "x", "i", "y"]
        assert joined.project("*").columns == ["i", "x", "i", "y"]

    def test_a_star_over_a_renamed_join_reads_the_names_its_result_exposes(self, con):
        # A set_alias call closes the join's SELECT, so a star over it sees the engine's i_1.
        starred = con.table("l").join(con.table("r"), "l.i = r.i").set_alias("q").project("*")
        assert starred.columns == ["i", "x", "i_1", "y"]
        assert_same(con, starred, "SELECT * FROM (SELECT * FROM l JOIN r ON l.i = r.i) AS q")


class TestQualifiedNamesAfterTheJoin:
    @pytest.mark.parametrize(
        ("step", "direct"),
        [
            pytest.param(lambda j: j.project("l.x, r.y"), "SELECT l.x, r.y FROM {j}", id="project"),
            pytest.param(lambda j: j.filter("r.y = 'C'"), "SELECT * FROM {j} WHERE r.y = 'C'", id="filter"),
            pytest.param(lambda j: j.order("r.y, l.x"), "SELECT * FROM {j} ORDER BY r.y, l.x", id="order"),
            pytest.param(
                lambda j: j.aggregate("l.x, count(r.y) AS n", "l.x"),
                "SELECT l.x, count(r.y) AS n FROM {j} GROUP BY l.x",
                id="aggregate",
            ),
            pytest.param(
                lambda j: j.filter("l.i > 1").project("l.x, r.y"),
                "SELECT l.x, r.y FROM {j} WHERE l.i > 1",
                id="filter_project",
            ),
            pytest.param(
                lambda j: j.filter("l.i > 1").filter("r.y <> 'B'").project("l.x, r.y"),
                "SELECT l.x, r.y FROM {j} WHERE l.i > 1 AND r.y <> 'B'",
                id="filter_filter_project",
            ),
            pytest.param(
                lambda j: j.filter("l.i > 1").aggregate("l.x, count(*) AS n", "l.x"),
                "SELECT l.x, count(*) AS n FROM {j} WHERE l.i > 1 GROUP BY l.x",
                id="filter_aggregate",
            ),
            pytest.param(
                lambda j: j.filter("l.i > 1").order("r.y"),
                "SELECT * FROM {j} WHERE l.i > 1 ORDER BY r.y",
                id="filter_order",
            ),
        ],
    )
    def test_the_sides_stay_visible(self, con, step, direct):
        rel = step(con.table("l").join(con.table("r"), "l.i = r.i", how="left"))
        statement = direct.format(j="l LEFT JOIN r ON l.i = r.i")
        assert_same(con, rel, statement, ordered="ORDER BY" in statement)

    @pytest.mark.parametrize(
        ("step", "direct"),
        [
            pytest.param(
                lambda j: j.order("r.y, l.x").project("x, y"),
                "SELECT x, y FROM (SELECT * FROM {j} ORDER BY r.y, l.x)",
                id="order_project",
            ),
            pytest.param(
                lambda j: j.order("r.y, l.x").limit(2).project("x, y"),
                "SELECT x, y FROM (SELECT * FROM {j} ORDER BY r.y, l.x LIMIT 2)",
                id="limit_project",
            ),
            pytest.param(
                lambda j: j.order("r.y DESC NULLS LAST, l.x DESC").project("x, y"),
                "SELECT x, y FROM (SELECT * FROM {j} ORDER BY r.y DESC NULLS LAST, l.x DESC)",
                id="order_descending_nulls_last",
            ),
            pytest.param(
                lambda j: j.order("r.y NULLS FIRST, l.x").project("x, y"),
                "SELECT x, y FROM (SELECT * FROM {j} ORDER BY r.y NULLS FIRST, l.x)",
                id="order_nulls_first",
            ),
            pytest.param(
                # The projection drops y, so a DISTINCT of the joined row and a DISTINCT of the projection differ.
                lambda j: j.distinct().project("x"),
                "SELECT x FROM (SELECT DISTINCT * FROM {j})",
                id="distinct_project",
            ),
            pytest.param(
                lambda j: j.order("r.y, l.x").filter("i > 1"),
                "SELECT * FROM (SELECT * FROM {j} ORDER BY r.y, l.x) WHERE i > 1",
                id="order_filter",
            ),
            pytest.param(
                lambda j: j.distinct().filter("i > 1"),
                "SELECT * FROM (SELECT DISTINCT * FROM {j}) WHERE i > 1",
                id="distinct_filter",
            ),
            pytest.param(
                lambda j: j.filter("l.i > 1").order("r.y").project("x, y"),
                "SELECT x, y FROM (SELECT * FROM {j} WHERE l.i > 1 ORDER BY r.y)",
                id="filter_order_project",
            ),
        ],
    )
    def test_a_step_after_the_select_closed_reads_the_output_names(self, con, step, direct):
        # Each of these steps closes the join's SELECT, so it binds the names the closed SELECT exposes.
        rel = step(con.table("l").join(con.table("r"), "l.i = r.i", how="left"))
        statement = direct.format(j="l LEFT JOIN r ON l.i = r.i")
        assert_same(con, rel, statement, ordered="ORDER BY" in statement and "DISTINCT" not in statement)

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(lambda j: j.order("r.y").project("l.x, r.y"), id="order_project"),
            pytest.param(lambda j: j.order("r.y").limit(2).project("l.x, r.y"), id="limit_project"),
            pytest.param(lambda j: j.distinct().project("l.x"), id="distinct_project"),
            pytest.param(lambda j: j.order("r.y").filter("l.i > 1"), id="order_filter"),
            pytest.param(lambda j: j.distinct().filter("l.i > 1"), id="distinct_filter"),
        ],
    )
    def test_a_side_after_the_select_closed_is_the_engine_s_error(self, con, step):
        # The input aliases are out of scope once the SELECT closed, and the call itself raises.
        with pytest.raises(duckdb.BinderException, match=r'Referenced table "[lr]" not found'):
            step(con.table("l").join(con.table("r"), "l.i = r.i", how="left"))

    def test_a_step_over_a_projection_reads_the_projected_names(self, con):
        rel = con.table("l").join(con.table("r"), "l.i = r.i").project("l.x AS lx, r.y").filter("lx = 'c'")
        assert_same(con, rel, "SELECT * FROM (SELECT l.x AS lx, r.y FROM l JOIN r ON l.i = r.i) WHERE lx = 'c'")


class TestStepOrderOverAJoin:
    def test_a_filter_after_a_limit_reads_the_limited_rows(self, con):
        joined = con.table("l").join(con.table("r"), "l.i = r.i")
        rel = joined.order("y, x").limit(2).filter("y = 'C'")
        assert rows(rel.fetchall()) == rows(
            con.execute(
                "SELECT * FROM (SELECT * FROM l JOIN r ON l.i = r.i ORDER BY y, x LIMIT 2) WHERE y = 'C'"
            ).fetchall()
        )

    def test_a_projection_after_distinct_keeps_one_row_per_distinct_input_row(self, con):
        joined = con.table("l").join(con.table("r"), "l.i = r.i")
        rel = joined.distinct().project("x")
        assert rows(rel.fetchall()) == rows(
            con.execute(
                "SELECT x FROM (SELECT DISTINCT l.i AS li, l.x, r.i AS ri, r.y FROM l JOIN r ON l.i = r.i)"
            ).fetchall()
        )

    @pytest.mark.parametrize(
        ("step", "flat"),
        [
            pytest.param(
                lambda j: j.filter("l.i > 1").distinct().order("r.y").limit(2),
                "SELECT DISTINCT * FROM l JOIN r ON l.i = r.i WHERE l.i > 1 ORDER BY r.y LIMIT 2",
                id="filter_distinct_order_limit",
            ),
            pytest.param(
                lambda j: j.aggregate("l.x, count(*) AS n", "l.x").filter("n > 1"),
                "SELECT * FROM (SELECT l.x, count(*) AS n FROM l JOIN r ON l.i = r.i GROUP BY l.x) WHERE n > 1",
                id="aggregate_filter",
            ),
            pytest.param(
                # A projection after the ORDER BY closes the SELECT, so it reads the output names.
                lambda j: j.sort("l.i").project("x, y"),
                "SELECT x, y FROM (SELECT * FROM l JOIN r ON l.i = r.i ORDER BY l.i)",
                id="sort_project",
            ),
        ],
    )
    def test_steps_in_clause_order_match_the_flat_sql(self, con, step, flat):
        rel = step(con.table("l").join(con.table("r"), "l.i = r.i"))
        cursor = con.execute(flat)
        names, expected = [column[0] for column in cursor.description], cursor.fetchall()
        assert rel.columns == names
        assert rel.fetchall() == expected

    def test_steps_after_an_aggregate_read_its_rows(self, con):
        joined = con.table("l").join(con.table("r"), "l.i = r.i")
        rel = joined.aggregate("l.x, count(*) AS n", "l.x").order("n DESC, x").limit(1).project("x")
        flat = (
            "SELECT x FROM (SELECT l.x, count(*) AS n FROM l JOIN r ON l.i = r.i GROUP BY l.x) "
            "ORDER BY n DESC, x LIMIT 1"
        )
        assert rel.fetchall() == con.execute(flat).fetchall()

    def test_distinct_after_a_limit_reads_the_limited_rows(self, con):
        joined = con.table("l").join(con.table("r"), "l.i = r.i")
        rel = joined.order("x, y").limit(2).distinct()
        assert rows(rel.fetchall()) == rows(
            con.execute(
                "SELECT DISTINCT * FROM (SELECT l.i AS li, l.x, r.i AS ri, r.y FROM l JOIN r ON l.i = r.i "
                "ORDER BY x, y LIMIT 2)"
            ).fetchall()
        )


class TestConditionsAKindRefuses:
    @pytest.mark.parametrize("how", ["cross", "positional", "natural"])
    @pytest.mark.parametrize(
        "condition",
        [
            pytest.param("i", id="name"),
            pytest.param(["i"], id="names"),
            pytest.param("l.i = r.i", id="sql"),
            pytest.param(ColumnExpression("l", "i") == ColumnExpression("r", "i"), id="expression"),
        ],
    )
    def test_a_condition_is_refused(self, con, how, condition):
        # Every join argument mistake is an InvalidInputException, as the shipping client had it.
        with pytest.raises(duckdb.InvalidInputException, match="takes no condition"):
            con.table("l").join(con.table("r"), condition, how=how)


class TestAliases:
    def test_a_table_is_named_by_the_table(self, con):
        assert con.table("l").alias == "l"

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(lambda rel: rel.filter("i > 1"), id="filter"),
            pytest.param(lambda rel: rel.project("i, x"), id="project"),
            pytest.param(lambda rel: rel.order("i"), id="order"),
            pytest.param(lambda rel: rel.limit(2), id="limit"),
            pytest.param(lambda rel: rel.distinct(), id="distinct"),
            pytest.param(lambda rel: rel.aggregate("i, count(*) AS n", "i"), id="aggregate"),
            pytest.param(lambda rel: rel.union(rel), id="union"),
        ],
    )
    def test_a_step_inherits_the_alias_of_its_input(self, con, step):
        rel = step(con.table("l"))
        assert rel.alias == "l"
        assert len(rel.join(con.table("r"), "l.i = r.i").fetchall()) > 0

    def test_two_sql_relations_of_equal_text_have_their_own_alias(self, con):
        left = con.sql("SELECT 1 AS i")
        right = con.sql("SELECT 1 AS i")
        assert left.alias.startswith("unnamed_relation_")
        assert right.alias.startswith("unnamed_relation_")
        assert left.alias != right.alias

    def test_values_and_readers_are_unnamed(self, con, tmp_path):
        path = tmp_path / "in.csv"
        path.write_text("a\n1\n")
        for rel in (con.values([1, "x"]), con.read_csv(str(path))):
            assert rel.alias.startswith("unnamed_relation_")

    def test_a_data_frame_keeps_its_df_alias(self, con):
        pd = pytest.importorskip("pandas")
        assert con.from_df(pd.DataFrame({"a": [1]})).alias.startswith("df_")


class TestPythonObjectInputs:
    def test_a_data_frame_joined_with_a_table(self, con):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame({"i": [2, 3, 5], "z": ["p", "q", "s"]})
        rel = con.from_df(frame).set_alias("d").join(con.table("l"), "d.i = l.i")
        cursor = con.execute("SELECT * FROM frame AS d JOIN l ON d.i = l.i")
        names, expected = [column[0] for column in cursor.description], rows(cursor.fetchall())
        assert rel.columns == names
        assert rows(rel.fetchall()) == expected

    def test_an_arrow_table_joined_with_a_table_using_a_name(self, con):
        pa = pytest.importorskip("pyarrow")
        arrow = pa.table({"i": [2, 3, 5], "z": ["p", "q", "s"]})
        rel = con.table("l").join(con.from_arrow(arrow), "i", how="left")
        cursor = con.execute("SELECT * FROM l LEFT JOIN arrow USING (i)")
        names, expected = [column[0] for column in cursor.description], rows(cursor.fetchall())
        assert rel.columns == names
        assert rows(rel.fetchall()) == expected


PASS_THROUGH = [
    pytest.param(lambda rel: rel.filter("id = 1"), "WHERE id = 1", id="filter"),
    pytest.param(lambda rel: rel.order("id").limit(1), "ORDER BY id LIMIT 1", id="limit"),
    pytest.param(lambda rel: rel.order("id").offset(1), "ORDER BY id OFFSET 1", id="offset"),
    pytest.param(lambda rel: rel.order("id").head(1), "ORDER BY id LIMIT 1", id="head"),
    pytest.param(lambda rel: rel.distinct(), "", id="distinct"),
    pytest.param(lambda rel: rel.sort("id"), "ORDER BY id", id="sort"),
    pytest.param(lambda rel: rel.order("id DESC"), "ORDER BY id DESC", id="order"),
    pytest.param(lambda rel: rel.set_alias("q"), "", id="set_alias"),
]

JOINS = [
    pytest.param(
        lambda con: con.table("orders").join(con.table("customers"), ["id"]),
        "SELECT * FROM orders JOIN customers USING (id)",
        id="using",
    ),
    pytest.param(
        lambda con: con.table("shared_left").join(con.table("shared_right"), how="natural"),
        "SELECT * FROM shared_left NATURAL JOIN shared_right",
        id="natural",
    ),
    pytest.param(
        lambda con: con.table("orders").join(con.table("customers"), ["id"], how="semi"),
        "SELECT * FROM orders SEMI JOIN customers USING (id)",
        id="semi",
    ),
    pytest.param(
        lambda con: con.table("orders").join(con.table("customers"), ["id"], how="anti"),
        "SELECT * FROM orders ANTI JOIN customers USING (id)",
        id="anti",
    ),
]


class TestAJoinWhoseStarIsNotAConcatenation:
    """A merged key shifts no value into the wrong column, whatever step reads the join.

    A USING or NATURAL join merges the key and a semi or anti join keeps one side, so the star of such a join is not
    the two sides written out one after the other. Every step over it must still answer the row the join holds.
    """

    def unfolded(self, con, direct):
        return con.sql(f"SELECT * FROM ({direct}) AS q")

    @pytest.mark.parametrize(("join", "direct"), JOINS)
    @pytest.mark.parametrize(("step", "clause"), PASS_THROUGH)
    def test_a_pass_through_step_answers_the_rows_of_the_unfolded_form(self, con, join, direct, step, clause):
        # A value in the wrong column changes the row tuple, so it shows in an unordered comparison as well; only an
        # ordered step fixes the sequence, and a DISTINCT or a filter does not.
        folded = step(join(con))
        reference = step(self.unfolded(con, direct))
        if "ORDER BY" in clause:
            assert folded.fetchall() == reference.fetchall()
        else:
            assert rows(folded.fetchall()) == rows(reference.fetchall())

    @pytest.mark.parametrize(("join", "direct"), JOINS)
    @pytest.mark.parametrize(("step", "clause"), PASS_THROUGH)
    def test_a_pass_through_step_answers_the_rows_of_the_sql_that_says_it(self, con, join, direct, step, clause):
        folded = step(join(con))
        statement = f"{direct} {clause}" if clause else direct
        expected = con.execute(statement).fetchall()
        if "ORDER BY" in clause:
            assert folded.fetchall() == expected
        else:
            assert rows(folded.fetchall()) == rows(expected)

    def test_a_using_join_keeps_each_side_s_data_column_in_its_own_place(self, con):
        joined = con.table("orders").join(con.table("customers"), ["id"])
        assert joined.columns == ["id", "name", "name"]
        assert joined.fetchall() == [(1, "order-a", "cust-a"), (2, "order-b", "cust-b")]
        assert joined.filter("id = 1").fetchall() == [(1, "order-a", "cust-a")]
        assert joined.sort("id").fetchall() == [(1, "order-a", "cust-a"), (2, "order-b", "cust-b")]
        assert joined.set_alias("q").fetchall() == [(1, "order-a", "cust-a"), (2, "order-b", "cust-b")]

    def test_a_natural_join_keeps_the_merged_row(self, con):
        joined = con.table("shared_left").join(con.table("shared_right"), how="natural")
        assert joined.columns == ["id", "name"]
        assert joined.filter("id = 1").fetchall() == [(1, "same")]

    @pytest.mark.parametrize(
        ("how", "expected"),
        [
            pytest.param("semi", [(1, "order-a"), (2, "order-b")], id="semi"),
            pytest.param("anti", [(4, "order-d")], id="anti"),
        ],
    )
    def test_a_one_sided_join_keeps_only_the_left_values(self, con, how, expected):
        joined = con.table("orders").join(con.table("customers"), ["id"], how=how)
        assert joined.columns == ["id", "name"]
        assert joined.sort("id").fetchall() == expected
        assert joined.distinct().sort("id").fetchall() == expected

    @pytest.mark.parametrize(("join", "direct"), JOINS)
    def test_a_sample_over_it_draws_whole_rows(self, con, join, direct):
        joined = join(con)
        held = joined.fetchall()
        drawn = joined.sample(1, seed=7).fetchall()
        assert len(drawn) <= 1
        assert all(row in held for row in drawn)


INLINED_JOINS = [
    pytest.param(
        lambda con: con.table("ta").join(con.table("tb"), ["id"]),
        "SELECT * FROM ta JOIN tb USING (id) JOIN tc ON ta.id = tc.id",
        id="using",
    ),
    pytest.param(
        lambda con: con.table("ta").join(con.table("tb"), how="natural"),
        "SELECT * FROM ta NATURAL JOIN tb JOIN tc ON ta.id = tc.id",
        id="natural",
    ),
]
BOUNDARY_STEPS = [
    pytest.param(lambda rel: rel, id="none"),
    pytest.param(lambda rel: rel.limit(1).limit(1), id="two_limits"),
    pytest.param(lambda rel: rel.limit(1).filter("true"), id="limit_then_filter"),
    pytest.param(lambda rel: rel.set_alias("q").limit(1).project("*"), id="alias_limit_star"),
]


class TestAnInlinedJoinWhoseStarIsNotAConcatenation:
    """A USING or natural join inlined into an outer join merges its key too, and every step must see that.

    The outer join's own output does not show the inner merge, so the check walks every join in the FROM.
    """

    @pytest.fixture
    def three(self, con):
        con.execute("CREATE TABLE ta AS SELECT 1 AS id, 10 AS x")
        con.execute("CREATE TABLE tb AS SELECT 1 AS id, 20 AS y")
        con.execute("CREATE TABLE tc AS SELECT 1 AS id, 30 AS z")
        return con

    @pytest.mark.parametrize(("inner", "direct"), INLINED_JOINS)
    @pytest.mark.parametrize("boundary", BOUNDARY_STEPS)
    def test_a_step_over_it_answers_the_rows_of_the_unfolded_form(self, three, inner, direct, boundary):
        con = three
        folded = boundary(inner(con).join(con.table("tc"), "ta.id = tc.id"))
        reference = boundary(con.sql(f"SELECT * FROM ({direct}) AS q"))
        assert rows(folded.fetchall()) == rows(reference.fetchall())

    @pytest.mark.parametrize(("inner", "direct"), INLINED_JOINS)
    def test_the_row_behind_two_limits_is_the_joined_row(self, three, inner, direct):
        con = three
        rel = inner(con).join(con.table("tc"), "ta.id = tc.id")
        assert rel.fetchall() == con.execute(direct).fetchall() == [(1, 10, 20, 1, 30)]
        assert rel.limit(1).limit(1).fetchall() == [(1, 10, 20, 1, 30)]


SET_OPERATIONS = [
    pytest.param("union", "UNION ALL", id="union"),
    pytest.param("intersect", "INTERSECT", id="intersect"),
    pytest.param("except_", "EXCEPT", id="except"),
]


class TestASetOperationOverAJoin:
    @pytest.mark.parametrize(("verb", "keyword"), SET_OPERATIONS)
    def test_it_reports_the_names_the_join_s_cte_exposes_and_its_rows(self, con, verb, keyword):
        # Each side is a closed SELECT read as a CTE, so the names are the left side's as the engine gives them.
        left = con.sql("SELECT 1 AS id, 10 AS x, 11 AS x_1").set_alias("l")
        right = con.sql("SELECT 1 AS id, 20 AS x").set_alias("r")
        joined = left.join(right, ["id"])
        assert joined.columns == ["id", "x", "x_1", "x"]
        side = "SELECT * FROM (SELECT 1 AS id, 10 AS x, 11 AS x_1) l JOIN (SELECT 1 AS id, 20 AS x) r USING (id)"
        rel = getattr(joined, verb)(joined)
        assert_same(con, rel, f"WITH s AS NOT MATERIALIZED ({side}) SELECT * FROM s {keyword} SELECT * FROM s")
        assert rel.columns == ["id", "x", "x_1", "x_2"]

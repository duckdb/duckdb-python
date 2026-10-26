"""Every step appends to its input's open SELECT or closes it, and the names are the ones the engine gives.

A SELECT has clause slots in the order the engine evaluates them: FROM, WHERE, GROUP BY, the select list, DISTINCT,
ORDER BY, OFFSET and LIMIT. A step appends when its slot is empty and no later slot is filled, and a filter after an
aggregate always closes; any other step closes the SELECT as a common table expression and opens a new one over it.
Every case is checked against the same statement written by hand, and the number of SELECTs a chain adds to its start
is read from the parsed text.
"""

import json

import pytest

import duckdb


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES (1, 'c'), (2, 'b'), (3, 'a'), (3, 'a'), (4, 'b')) v(i, x)"
    )
    connection.execute("CREATE TABLE l AS SELECT * FROM t")
    connection.execute("CREATE TABLE r AS SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30), (4, 40)) v(i, y)")
    connection.execute("CREATE TABLE rk AS SELECT * FROM (VALUES (1, 11), (2, 22)) v(k, x)")
    connection.execute("CREATE TABLE r2 AS SELECT * FROM (VALUES (1, 111), (2, 222)) v(i, i_1)")
    connection.execute("CREATE TABLE a AS SELECT * FROM (VALUES (1, 10), (2, 20)) v(id, x)")
    connection.execute("CREATE TABLE b AS SELECT * FROM (VALUES (1, 100), (2, 200)) v(id, y)")
    connection.execute("CREATE TABLE c AS SELECT * FROM (VALUES (1, 1000), (3, 3000)) v(id, z)")
    yield connection
    connection.close()


def walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def parsed(con, rel):
    [(tree,)] = con.execute("SELECT json_serialize_sql(?)", [rel.sql_query()]).fetchall()
    tree = json.loads(tree)
    assert not tree["error"], tree
    [statement] = tree["statements"]
    return statement["node"]


def selects(con, rel):
    return sum(1 for part in walk(parsed(con, rel)) if part.get("type") == "SELECT_NODE")


def rows(result):
    return sorted(result, key=repr)


def assert_same(con, rel, direct, *, ordered=False):
    """The relation reports the names and answers the rows of the SQL, in order when the SQL orders them."""
    cursor = con.execute(direct)
    names = [column[0] for column in cursor.description]
    expected = cursor.fetchall()
    assert rel.columns == names
    if ordered:
        assert rel.fetchall() == expected
    else:
        assert rows(rel.fetchall()) == rows(expected)


def table(con):
    return con.table("t")


def joined(con):
    return con.table("l").join(con.table("r"), "l.i = r.i")


J = "l JOIN r ON l.i = r.i"

TABLE_CASES = [
    pytest.param(lambda t: t.filter("i > 1"), 0, "SELECT * FROM t WHERE i > 1", False, id="filter"),
    pytest.param(
        lambda t: t.filter("i > 1").filter("x <> 'b'"),
        0,
        "SELECT * FROM t WHERE i > 1 AND x <> 'b'",
        False,
        id="filter_filter",
    ),
    pytest.param(
        lambda t: t.filter("i > 1").aggregate("x, count(*) AS n", "x"),
        0,
        "SELECT x, count(*) AS n FROM t WHERE i > 1 GROUP BY x",
        False,
        id="filter_aggregate",
    ),
    pytest.param(
        lambda t: t.aggregate("x, count(*) AS n", "x").filter("n > 1"),
        1,
        "SELECT * FROM (SELECT x, count(*) AS n FROM t GROUP BY x) WHERE n > 1",
        False,
        id="filter_after_aggregate",
    ),
    pytest.param(
        lambda t: t.filter("i > 1").aggregate("x, max(i) AS m", "x").filter("m > 2").order("x").limit(2),
        1,
        "SELECT * FROM (SELECT x, max(i) AS m FROM t WHERE i > 1 GROUP BY x) WHERE m > 2 ORDER BY x LIMIT 2",
        True,
        id="every_slot_with_a_filter_after_the_aggregate",
    ),
    pytest.param(
        lambda t: t.filter("i > 1").project("x"), 0, "SELECT x FROM t WHERE i > 1", False, id="filter_project"
    ),
    pytest.param(lambda t: t.project("x").distinct(), 0, "SELECT DISTINCT x FROM t", False, id="project_distinct"),
    pytest.param(
        lambda t: t.distinct().order("i"), 0, "SELECT DISTINCT * FROM t ORDER BY i", True, id="distinct_order"
    ),
    pytest.param(
        lambda t: t.project("x, i").order("i"),
        1,
        "SELECT * FROM (SELECT x, i FROM t) ORDER BY i",
        True,
        id="order_after_project",
    ),
    pytest.param(
        lambda t: t.filter("i > 1").order("x, i"),
        0,
        "SELECT * FROM t WHERE i > 1 ORDER BY x, i",
        True,
        id="filter_order",
    ),
    pytest.param(lambda t: t.order("i").limit(2), 0, "SELECT * FROM t ORDER BY i LIMIT 2", True, id="order_limit"),
    pytest.param(
        lambda t: t.order("i").offset(1).limit(2),
        0,
        "SELECT * FROM t ORDER BY i LIMIT 2 OFFSET 1",
        True,
        id="offset_then_limit",
    ),
    pytest.param(
        lambda t: t.order("i").limit(2, offset=1),
        0,
        "SELECT * FROM t ORDER BY i LIMIT 2 OFFSET 1",
        True,
        id="limit_with_an_offset",
    ),
    pytest.param(
        lambda t: t.order("i").limit(3).filter("i > 1"),
        1,
        "SELECT * FROM (SELECT * FROM t ORDER BY i LIMIT 3) WHERE i > 1",
        False,
        id="filter_after_limit",
    ),
    pytest.param(
        lambda t: t.order("i").filter("i > 1"),
        1,
        "SELECT * FROM (SELECT * FROM t ORDER BY i) WHERE i > 1",
        False,
        id="filter_after_order",
    ),
    pytest.param(
        lambda t: t.distinct().filter("i > 1"),
        1,
        "SELECT * FROM (SELECT DISTINCT * FROM t) WHERE i > 1",
        False,
        id="filter_after_distinct",
    ),
    pytest.param(
        lambda t: t.project("i AS k, x").filter("k > 1"),
        1,
        "SELECT * FROM (SELECT i AS k, x FROM t) WHERE k > 1",
        False,
        id="filter_after_project",
    ),
    pytest.param(
        lambda t: t.project("i, x").project("x"),
        1,
        "SELECT x FROM (SELECT i, x FROM t)",
        False,
        id="project_after_project",
    ),
    pytest.param(
        lambda t: t.order("i").project("x AS i"),
        1,
        "SELECT x AS i FROM (SELECT * FROM t ORDER BY i)",
        True,
        id="project_after_order_that_renames",
    ),
    pytest.param(
        lambda t: t.distinct().project("x"),
        1,
        "SELECT x FROM (SELECT DISTINCT * FROM t)",
        False,
        id="distinct_then_project",
    ),
    pytest.param(
        lambda t: t.order("i").limit(2).offset(1),
        1,
        "SELECT * FROM (SELECT * FROM t ORDER BY i LIMIT 2) OFFSET 1",
        True,
        id="limit_then_offset",
    ),
    pytest.param(
        lambda t: t.order("i").limit(3).limit(1),
        1,
        "SELECT * FROM (SELECT * FROM t ORDER BY i LIMIT 3) LIMIT 1",
        True,
        id="limit_after_limit",
    ),
    pytest.param(
        lambda t: t.order("i").limit(3).order("x"),
        1,
        "SELECT * FROM (SELECT * FROM t ORDER BY i LIMIT 3) ORDER BY x",
        True,
        id="order_after_limit",
    ),
    pytest.param(
        lambda t: t.order("i").distinct(),
        1,
        "SELECT DISTINCT * FROM (SELECT * FROM t ORDER BY i)",
        False,
        id="distinct_after_order",
    ),
    pytest.param(
        lambda t: t.aggregate("x, count(*) AS n", "x").project("n"),
        1,
        "SELECT n FROM (SELECT x, count(*) AS n FROM t GROUP BY x)",
        False,
        id="project_after_aggregate",
    ),
    pytest.param(
        lambda t: t.project("i, x").aggregate("x, sum(i) AS s", "x"),
        1,
        "SELECT x, sum(i) AS s FROM (SELECT i, x FROM t) GROUP BY x",
        False,
        id="aggregate_after_project",
    ),
    pytest.param(
        lambda t: t.aggregate("x, count(*) AS n", "x").aggregate("max(n) AS m"),
        1,
        "SELECT max(n) AS m FROM (SELECT x, count(*) AS n FROM t GROUP BY x)",
        False,
        id="aggregate_after_aggregate",
    ),
    pytest.param(
        lambda t: t.aggregate("x, count(*) AS n", "x").order("x").filter("n > 1"),
        2,
        "SELECT * FROM (SELECT * FROM (SELECT x, count(*) AS n FROM t GROUP BY x) ORDER BY x) WHERE n > 1",
        False,
        id="filter_after_an_ordered_aggregate",
    ),
    pytest.param(
        lambda t: t.order("i").limit(3).aggregate("count(*) AS n"),
        1,
        "SELECT count(*) AS n FROM (SELECT * FROM t ORDER BY i LIMIT 3)",
        False,
        id="aggregate_after_limit",
    ),
]

JOIN_CASES = [
    pytest.param(lambda j: j.filter("l.i > 1"), 0, f"SELECT * FROM {J} WHERE l.i > 1", False, id="filter"),
    pytest.param(
        lambda j: j.filter("l.i > 1").filter("r.y < 40"),
        0,
        f"SELECT * FROM {J} WHERE l.i > 1 AND r.y < 40",
        False,
        id="filter_filter",
    ),
    pytest.param(
        lambda j: j.filter("l.i > 1").aggregate("l.x, sum(r.y) AS s", "l.x"),
        0,
        f"SELECT l.x, sum(r.y) AS s FROM {J} WHERE l.i > 1 GROUP BY l.x",
        False,
        id="filter_aggregate",
    ),
    pytest.param(
        lambda j: j.aggregate("l.x, count(*) AS n", "l.x").filter("n > 1"),
        1,
        f"SELECT * FROM (SELECT l.x, count(*) AS n FROM {J} GROUP BY l.x) WHERE n > 1",
        False,
        id="filter_after_aggregate",
    ),
    pytest.param(
        lambda j: j.filter("l.i > 1").project("l.x, r.y"),
        0,
        f"SELECT l.x, r.y FROM {J} WHERE l.i > 1",
        False,
        id="filter_project",
    ),
    pytest.param(
        lambda j: j.project("l.x").distinct(), 0, f"SELECT DISTINCT l.x FROM {J}", False, id="project_distinct"
    ),
    pytest.param(
        lambda j: j.project("l.x, r.y").order("y"),
        1,
        f"SELECT * FROM (SELECT l.x, r.y FROM {J}) ORDER BY y",
        True,
        id="order_after_project",
    ),
    pytest.param(
        lambda j: j.distinct().order("r.y"), 0, f"SELECT DISTINCT * FROM {J} ORDER BY r.y", True, id="distinct_order"
    ),
    pytest.param(
        lambda j: j.order("r.y").offset(1).limit(2),
        0,
        f"SELECT * FROM {J} ORDER BY r.y LIMIT 2 OFFSET 1",
        True,
        id="offset_then_limit",
    ),
    pytest.param(
        lambda j: j.filter("l.i > 1").order("r.y DESC").limit(2),
        0,
        f"SELECT * FROM {J} WHERE l.i > 1 ORDER BY r.y DESC LIMIT 2",
        True,
        id="filter_order_limit",
    ),
    pytest.param(
        lambda j: j.order("r.y").limit(3).filter("y > 10"),
        1,
        f"SELECT * FROM (SELECT * FROM {J} ORDER BY r.y LIMIT 3) WHERE y > 10",
        False,
        id="filter_after_limit",
    ),
    pytest.param(
        lambda j: j.order("r.y").filter("y > 10"),
        1,
        f"SELECT * FROM (SELECT * FROM {J} ORDER BY r.y) WHERE y > 10",
        False,
        id="filter_after_order",
    ),
    pytest.param(
        lambda j: j.distinct().filter("y > 10"),
        1,
        f"SELECT * FROM (SELECT DISTINCT * FROM {J}) WHERE y > 10",
        False,
        id="filter_after_distinct",
    ),
    pytest.param(
        lambda j: j.project("l.x AS a, r.y").filter("a = 'b'"),
        1,
        f"SELECT * FROM (SELECT l.x AS a, r.y FROM {J}) WHERE a = 'b'",
        False,
        id="filter_after_project",
    ),
    pytest.param(
        lambda j: j.order("r.y").project("x AS y"),
        1,
        f"SELECT x AS y FROM (SELECT * FROM {J} ORDER BY r.y)",
        True,
        id="project_after_order_that_renames",
    ),
    pytest.param(
        lambda j: j.distinct().project("x"),
        1,
        f"SELECT x FROM (SELECT DISTINCT * FROM {J})",
        False,
        id="distinct_then_project",
    ),
    pytest.param(
        lambda j: j.order("r.y").limit(2).offset(1),
        1,
        f"SELECT * FROM (SELECT * FROM {J} ORDER BY r.y LIMIT 2) OFFSET 1",
        True,
        id="limit_then_offset",
    ),
    pytest.param(
        lambda j: j.order("r.y").limit(3).order("x"),
        1,
        f"SELECT * FROM (SELECT * FROM {J} ORDER BY r.y LIMIT 3) ORDER BY x",
        True,
        id="order_after_limit",
    ),
    pytest.param(
        lambda j: j.aggregate("l.x, count(*) AS n", "l.x").project("n"),
        1,
        f"SELECT n FROM (SELECT l.x, count(*) AS n FROM {J} GROUP BY l.x)",
        False,
        id="project_after_aggregate",
    ),
    pytest.param(
        lambda j: j.project("l.x, r.y").aggregate("x, sum(y) AS s", "x"),
        1,
        f"SELECT x, sum(y) AS s FROM (SELECT l.x, r.y FROM {J}) GROUP BY x",
        False,
        id="aggregate_after_project",
    ),
    pytest.param(
        lambda j: j.aggregate("l.x, count(*) AS n", "l.x").order("x").filter("n > 1"),
        2,
        f"SELECT * FROM (SELECT * FROM (SELECT l.x, count(*) AS n FROM {J} GROUP BY l.x) ORDER BY x) WHERE n > 1",
        False,
        id="filter_after_an_ordered_aggregate",
    ),
    pytest.param(
        lambda j: j.order("r.y").distinct(),
        1,
        f"SELECT DISTINCT * FROM (SELECT * FROM {J} ORDER BY r.y)",
        False,
        id="distinct_after_order",
    ),
]


class TestOneSelectUnderConstruction:
    """A step appends while its slot is empty and no later slot is filled, and otherwise closes the SELECT."""

    @pytest.mark.parametrize(("build", "added", "direct", "ordered"), TABLE_CASES)
    def test_over_a_table_the_text_adds_one_select_per_close(self, con, build, added, direct, ordered):
        assert selects(con, build(table(con))) - selects(con, table(con)) == added

    @pytest.mark.parametrize(("build", "added", "direct", "ordered"), TABLE_CASES)
    def test_over_a_table_the_rows_and_names_are_the_sql_s(self, con, build, added, direct, ordered):
        assert_same(con, build(table(con)), direct, ordered=ordered)

    @pytest.mark.parametrize(("build", "added", "direct", "ordered"), JOIN_CASES)
    def test_over_a_join_the_text_adds_one_select_per_close(self, con, build, added, direct, ordered):
        assert selects(con, build(joined(con))) - selects(con, joined(con)) == added

    @pytest.mark.parametrize(("build", "added", "direct", "ordered"), JOIN_CASES)
    def test_over_a_join_the_rows_and_names_are_the_sql_s(self, con, build, added, direct, ordered):
        assert_same(con, build(joined(con)), direct, ordered=ordered)

    def test_a_filter_after_an_aggregate_reads_the_aggregate_s_output(self, con):
        # As HAVING the name col0 would bind the group key, so the filter closes the SELECT and reads the output.
        con.execute("CREATE TABLE g AS SELECT * FROM (VALUES (1, 10), (2, 20), (2, 30)) v(col0, col1)")
        rel = con.table("g").aggregate("col0 + 1 AS col0, sum(col1) AS s", "col0").filter("col0 = 2")
        direct = "SELECT * FROM (SELECT col0 + 1 AS col0, sum(col1) AS s FROM g GROUP BY col0) WHERE col0 = 2"
        assert_same(con, rel, direct)
        assert rel.fetchall() == [(2, 10)]
        having = "SELECT col0 + 1 AS col0, sum(col1) AS s FROM g GROUP BY col0 HAVING col0 = 2"
        assert con.execute(having).fetchall() == [(3, 50)]

    @pytest.mark.parametrize(
        ("build", "direct", "expected"),
        [
            pytest.param(
                lambda kv: kv.project("v AS k, k AS v").order("k + 0").limit(1),
                "SELECT * FROM (SELECT v AS k, k AS v FROM kv) ORDER BY k + 0 LIMIT 1",
                [(10, 3)],
                id="expression_after_project",
            ),
            pytest.param(
                lambda kv: kv.project("v AS k, k AS v").order("kv.k"),
                "SELECT * FROM (SELECT v AS k, k AS v FROM kv) AS kv ORDER BY kv.k",
                [(10, 3), (20, 2), (30, 1)],
                id="qualified_name_after_project",
            ),
            pytest.param(
                lambda kv: kv.aggregate("k AS g, sum(v) AS k", "k").order("k * -1").limit(1),
                "SELECT * FROM (SELECT k AS g, sum(v) AS k FROM kv GROUP BY k) ORDER BY k * -1 LIMIT 1",
                [(1, 30)],
                id="expression_after_aggregate",
            ),
            pytest.param(
                lambda kv: kv.with_columns(v=duckdb.SQLExpression("k * 100")).order("abs(v)").limit(1),
                "SELECT * FROM (SELECT * REPLACE (k * 100 AS v) FROM kv) ORDER BY abs(v) LIMIT 1",
                [(1, 100)],
                id="expression_after_with_columns",
            ),
        ],
    )
    def test_an_order_after_a_select_list_reads_the_output(self, con, build, direct, expected):
        # Appended, an expression or a qualified name in ORDER BY binds the input column of the same name.
        con.execute("CREATE TABLE kv AS SELECT * FROM (VALUES (1, 30), (2, 20), (3, 10)) v(k, v)")
        rel = build(con.table("kv"))
        assert_same(con, rel, direct, ordered=True)
        assert rel.fetchall() == expected

    @pytest.mark.parametrize(
        ("build", "direct", "expected"),
        [
            pytest.param(
                lambda kv: kv.aggregate("k, sum(v)", "k").filter('"sum(v)" > 25'),
                'SELECT * FROM (SELECT k, sum(v) FROM kv GROUP BY k) WHERE "sum(v)" > 25',
                [(1, 30)],
                id="an_implicit_column_name",
            ),
            pytest.param(
                lambda kv: kv.aggregate("k, sum(v) AS s, rank() OVER (ORDER BY sum(v) DESC) AS r", "k").filter("r = 1"),
                "SELECT * FROM (SELECT k, sum(v) AS s, rank() OVER (ORDER BY sum(v) DESC) AS r FROM kv GROUP BY k) "
                "WHERE r = 1",
                [(1, 30, 1)],
                id="a_window_alias",
            ),
        ],
    )
    def test_a_filter_after_an_aggregate_reads_names_having_could_not(self, con, build, direct, expected):
        # HAVING refuses an implicit name and a window alias, which a filter over the finished aggregate binds.
        con.execute("CREATE TABLE kv AS SELECT * FROM (VALUES (1, 30), (2, 20), (3, 10)) v(k, v)")
        rel = build(con.table("kv"))
        assert_same(con, rel, direct)
        assert rel.fetchall() == expected

    def test_the_order_of_a_closed_select_is_kept_by_a_projection_over_it(self, con):
        # Appending the projection would make ORDER BY i bind the new name i, which holds x.
        rel = table(con).order("i").project("x AS i")
        assert rel.fetchall() == [("c",), ("b",), ("a",), ("a",), ("b",)]

    def test_a_filter_after_a_limit_reads_only_the_limited_rows(self, con):
        assert table(con).order("i").limit(3).filter("i > 1").fetchall() == [(2, "b"), (3, "a")]

    def test_limit_then_offset_and_offset_then_limit_differ(self, con):
        assert table(con).order("i").limit(2).offset(1).fetchall() == [(2, "b")]
        assert table(con).order("i").offset(1).limit(2).fetchall() == [(2, "b"), (3, "a")]


class TestTheStepsThatAlwaysClose:
    """These steps close the SELECT before them and their own, so the next step opens a new SELECT."""

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda t: t.filter("i > 1").distinct(on="i"), id="distinct_on"),
            pytest.param(lambda t: t.filter("i > 1").sample(2, seed=1), id="sample"),
            pytest.param(lambda t: t.filter("i > 1").union(t.filter("i > 1")), id="union"),
        ],
    )
    @pytest.mark.parametrize(
        "after",
        [
            pytest.param(lambda rel: rel.filter("i > 2"), id="filter"),
            pytest.param(lambda rel: rel.order("i"), id="order"),
            pytest.param(lambda rel: rel.limit(1), id="limit"),
        ],
    )
    def test_the_next_step_opens_a_new_select(self, con, build, after):
        closing = build(table(con))
        assert selects(con, after(closing)) - selects(con, closing) == 1

    def test_distinct_on_closes_the_select_before_it(self, con):
        filtered = table(con).filter("i > 1")
        assert selects(con, filtered.distinct(on="i")) - selects(con, filtered) == 1

    def test_an_order_after_distinct_on_does_not_change_the_kept_rows(self, con):
        rel = table(con).filter("i > 1").distinct(on="i").order("i")
        assert_same(con, rel, "SELECT * FROM (SELECT DISTINCT ON (i) * FROM t WHERE i > 1) ORDER BY i", ordered=True)


class TestNamesAreTheEngines:
    """An open SELECT over a join reports its names as they are, and a closed one the names its CTE exposes."""

    @pytest.mark.parametrize(
        ("build", "direct", "inside", "outside"),
        [
            pytest.param(
                lambda con: con.table("l").join(con.table("r"), "l.i = r.i"),
                "SELECT * FROM l JOIN r ON l.i = r.i",
                ["i", "x", "i", "y"],
                ["i", "x", "i_1", "y"],
                id="shared_key",
            ),
            pytest.param(
                lambda con: con.table("l").join(con.table("r"), ["i"]),
                "SELECT * FROM l JOIN r USING (i)",
                ["i", "x", "y"],
                ["i", "x", "y"],
                id="merged_key",
            ),
            pytest.param(
                lambda con: con.table("l").join(con.table("rk"), "l.i = rk.k"),
                "SELECT * FROM l JOIN rk ON l.i = rk.k",
                ["i", "x", "k", "x"],
                ["i", "x", "k", "x_1"],
                id="shared_non_key",
            ),
            pytest.param(
                lambda con: con.table("l").join(con.table("r2"), "l.i = r2.i"),
                "SELECT * FROM l JOIN r2 ON l.i = r2.i",
                ["i", "x", "i", "i_1"],
                ["i", "x", "i_1", "i_1_1"],
                id="a_name_that_already_has_the_suffix",
            ),
        ],
    )
    def test_inside_and_after_the_boundary(self, con, build, direct, inside, outside):
        # A LIMIT appends to the open SELECT and a filter after it closes that SELECT.
        closed = f"SELECT * FROM ({direct} LIMIT 10) WHERE true"
        assert [column[0] for column in con.execute(f"{direct} LIMIT 10").description] == inside
        assert [column[0] for column in con.execute(closed).description] == outside
        assert_same(con, build(con).limit(10), f"{direct} LIMIT 10")
        assert_same(con, build(con).limit(10).filter("true"), closed)

    def test_a_bare_name_both_sides_have_is_ambiguous_in_the_open_select(self, con):
        with pytest.raises(duckdb.BinderException, match="Ambiguous reference"):
            joined(con).filter("i = 1")

    def test_after_the_boundary_the_suffixed_name_is_the_right_side_s_key(self, con):
        rel = joined(con).limit(10).filter("i_1 = 2")
        assert_same(con, rel, f"SELECT * FROM (SELECT * FROM {J} LIMIT 10) WHERE i_1 = 2")
        assert rel.fetchall() == [(2, "b", 2, 20)]

    def test_a_column_already_named_like_the_suffix_has_one_name_for_one_column(self, con):
        # The second i becomes i_1 and the real i_1 becomes i_1_1, and every step reads them by those names.
        limited = con.table("l").join(con.table("r2"), "l.i = r2.i").limit(5)
        direct = "SELECT * FROM (SELECT * FROM l JOIN r2 ON l.i = r2.i LIMIT 5)"
        assert_same(con, limited.filter("i_1_1 = 111"), f"{direct} WHERE i_1_1 = 111")
        assert limited.filter("i_1_1 = 111").fetchall() == [(1, "c", 1, 111)]
        assert limited.filter("i_1 = 111").fetchall() == []
        assert limited.filter("i_1 = 1").project("i_1, i_1_1").fetchall() == [(1, 111)]
        assert limited.order("i_1_1 DESC").project("i_1_1").fetchall() == [(222,), (111,)]

    def test_a_materialized_join_reports_the_names_of_its_rows(self, con):
        # The held rows carry the engine's names, so columns and every step over them read those.
        con.execute("CREATE TABLE rkv AS SELECT * FROM (VALUES (1, 'P'), (2, 'Q')) v(k, x)")
        held = con.table("l").join(con.table("rkv"), "l.i = rkv.k").materialize()
        assert held.columns == held.project("*").columns == ["i", "x", "k", "x_1"]
        assert held.select_types(["VARCHAR"]).fetchall() == [("c", "P"), ("b", "Q")]

    def test_a_materialized_column_named_like_the_suffix_binds_what_columns_reports(self, con):
        # The fourth column is i_1_1 and a filter on that name reads it.
        held = con.table("l").join(con.table("r2"), "l.i = r2.i").materialize()
        assert held.columns == ["i", "x", "i_1", "i_1_1"]
        assert held.filter("i_1_1 = 111").fetchall() == [(1, "c", 1, 111)]
        assert held.filter("i_1 = 2").fetchall() == [(2, "b", 2, 222)]

    def test_no_positional_reference_is_rendered(self, con):
        closed = (
            joined(con).limit(5).filter("y > 0"),
            joined(con).order("r.y").limit(2).limit(1),
            joined(con).set_alias("q").limit(1),
        )
        for rel in closed:
            assert [part for part in walk(parsed(con, rel)) if part.get("class") == "POSITIONAL_REFERENCE"] == []


def engine_only(text):
    """The engine's error and nothing the library appends: no wording about scope or closed SELECTs anywhere in it."""
    return rf"(?s)\A(?!.*(?:out of scope|opened a new SELECT|qualify by|closed SELECT)).*{text}"


def side_error(alias):
    return engine_only(rf'Referenced table "{alias}" not found')


class TestAnInputAliasAfterTheSelectClosedIsLoud:
    """An input alias used after the SELECT closed fails at the call with the engine's own binder error."""

    @pytest.mark.parametrize(
        ("build", "alias"),
        [
            pytest.param(lambda j: j.order("r.y").limit(2).filter("l.x = 'b'"), "l", id="after_limit"),
            pytest.param(lambda j: j.project("l.x, r.y").filter("l.x = 'b'"), "l", id="after_project"),
            pytest.param(lambda j: j.aggregate("l.x, count(*) AS n", "l.x").project("l.x"), "l", id="after_aggregate"),
            pytest.param(lambda j: j.set_alias("q").filter("l.x = 'b'"), "l", id="after_set_alias"),
            pytest.param(lambda j: j.project("l.x, r.y").order("r.y"), "r", id="order_after_project"),
            pytest.param(lambda j: j.distinct(on="l.i"), "l", id="distinct_on"),
            pytest.param(lambda j: j.union(j).filter("l.i = 1"), "l", id="after_union"),
            pytest.param(lambda j: j.limit(5).filter("x <> 'z'").filter("l.x <> 'z'"), "l", id="two_steps_after"),
            pytest.param(lambda j: j.y.filter("r.y > 0"), "r", id="after_a_column_by_attribute"),
        ],
    )
    def test_the_engine_error_at_the_call(self, con, build, alias):
        with pytest.raises(duckdb.BinderException, match=side_error(alias)):
            build(joined(con))

    @pytest.mark.parametrize(
        ("build", "alias"),
        [
            pytest.param(
                lambda con: con.table("a").join(con.table("b"), "a.id = b.id").join(con.table("a"), "b.id = a.id"),
                "b",
                id="taken_alias_in_the_condition",
            ),
            pytest.param(
                lambda con: joined(con).join(con.table("r"), "TRUE").filter("l.x <> 'z'"),
                "l",
                id="taken_alias_in_a_step_after",
            ),
            pytest.param(
                lambda con: (
                    joined(con).join(con.table("l").set_alias("l2"), lambda a, b: a["x"] == b["x"]).filter("r.y > 0")
                ),
                "r",
                id="callable_condition",
            ),
            pytest.param(
                lambda con: (
                    con.sql("SELECT 1 AS k, 10 AS x", alias="p")
                    .join(con.sql("SELECT 20 AS x", alias="q"), "TRUE", suffix="_r")
                    .join(con.sql("SELECT 30 AS z", alias="t"), "TRUE")
                    .filter("q.x > 0")
                ),
                "q",
                id="a_suffix_join_as_a_side",
            ),
            pytest.param(
                lambda con: (
                    con.sql("SELECT 1 AS k, 10 AS x", alias="p")
                    .join(con.sql("SELECT 1 AS k, 20 AS y", alias="q"), "k")
                    .limit(5)
                    .join(con.sql("SELECT 1 AS k, 30 AS z", alias="r"), "p.k = r.k")
                ),
                "p",
                id="a_side_with_a_filled_slot",
            ),
            pytest.param(
                lambda con: (
                    lambda p, q: p.join(q, "TRUE").join(q, "TRUE").join(con.sql("SELECT 30 AS z", alias="t"), "TRUE")
                )(con.sql("SELECT 1 AS k, 10 AS x", alias="p"), con.sql("SELECT 20 AS x", alias="q")).filter("p.k > 0"),
                "p",
                id="an_inlined_join_that_closed_its_inner_join",
            ),
        ],
    )
    def test_an_inner_join_that_entered_closed_hides_its_inputs(self, con, build, alias):
        with pytest.raises(duckdb.BinderException, match=side_error(alias)):
            build(con)

    @pytest.mark.parametrize(
        ("build", "alias"),
        [
            pytest.param(lambda con: table(con).set_alias("n").filter("t.i = 1"), "t", id="a_bare_source_in_a_filter"),
            pytest.param(
                lambda con: table(con).set_alias("n").join(con.table("r"), "t.i = r.i"),
                "t",
                id="a_bare_source_as_a_join_input",
            ),
            pytest.param(
                lambda con: joined(con).set_alias("q").join(con.table("rk"), "q.i = rk.k").filter("l.x <> 'z'"),
                "l",
                id="a_named_join_as_a_join_input",
            ),
        ],
    )
    def test_the_name_set_alias_replaced_is_out_of_scope(self, con, build, alias):
        with pytest.raises(duckdb.BinderException, match=side_error(alias)):
            build(con)

    def test_a_column_a_derived_side_projected_away(self, con):
        # The projection kept its source's alias t, so t is in scope and only the column z is missing.
        source = con.sql("SELECT 1 AS k, 30 AS z", alias="t")
        other = con.sql("SELECT 1 AS k, 10 AS x", alias="l")
        joined_sides = source.project("k").join(other, "t.k = l.k")
        with pytest.raises(duckdb.BinderException, match=engine_only('does not have a column named "z"')):
            joined_sides.filter("t.z > 0")

    def test_a_name_the_boundary_made_on_a_shadowed_alias(self, con):
        p = con.sql("SELECT 1 AS k, 10 AS x", alias="p")
        pair = p.join(con.sql("SELECT 2 AS x, 3 AS x", alias="q"), "TRUE")
        with pytest.raises(duckdb.BinderException, match=engine_only('does not have a column named "x_1"')):
            pair.join(con.sql("SELECT 4 AS z", alias="q"), "q.x_1 = z")

    def test_an_alias_that_shadows_a_closed_input(self, con):
        p = con.sql("SELECT 1 AS k, 10 AS x", alias="p")
        named = p.join(con.sql("SELECT 2 AS x", alias="q"), "TRUE").project("p.k").set_alias("q")
        with pytest.raises(duckdb.BinderException, match=engine_only('does not have a column named "x"')):
            named.filter("q.x > 0")

    @pytest.mark.parametrize(
        "boundary",
        [
            pytest.param(lambda j: j, id="open"),
            pytest.param(lambda j: j.order("r.y").limit(10), id="after_limit"),
            pytest.param(lambda j: j.distinct(), id="after_distinct"),
            pytest.param(lambda j: j.set_alias("q"), id="after_set_alias"),
        ],
    )
    def test_a_filter_by_an_output_name_binds_on_both_sides(self, con, boundary):
        rel = boundary(joined(con)).filter("x = 'b' AND y > 10")
        assert rows(rel.fetchall()) == rows(con.execute(f"SELECT * FROM {J} WHERE x = 'b' AND y > 10").fetchall())
        assert rows(rel.fetchall()) == [(2, "b", 2, 20), (4, "b", 4, 40)]


class TestSetAliasNamesTheResult:
    """A set_alias call names the SELECT for its consumer, so a step after it reads the result under that name."""

    def test_the_alias_qualifies_the_result_of_a_join(self, con):
        rel = joined(con).set_alias("q").filter("q.x = 'b'")
        assert_same(con, rel, f"SELECT * FROM (SELECT * FROM {J}) AS q WHERE q.x = 'b'")
        assert rel.columns == ["i", "x", "i_1", "y"]

    def test_the_alias_qualifies_the_result_of_a_table(self, con):
        assert_same(con, table(con).set_alias("q").filter("q.i > 2"), "SELECT * FROM t AS q WHERE q.i > 2")

    def test_a_step_after_it_opens_one_new_select(self, con):
        assert selects(con, joined(con).set_alias("q").filter("q.x = 'b'")) - selects(con, joined(con)) == 1

    def test_on_a_bare_source_it_names_the_source_inline(self, con):
        # A source with no filled slot is a FROM item under the alias, so the alias and a filter add no SELECT.
        assert selects(con, table(con).set_alias("q")) == selects(con, table(con))
        assert selects(con, table(con).set_alias("q").filter("q.i > 2")) == selects(con, table(con))

    def test_a_star_over_it_reports_the_names_the_result_exposes(self, con):
        rel = joined(con).set_alias("q").project("*")
        assert_same(con, rel, f"SELECT * FROM (SELECT * FROM {J}) AS q")
        assert rel.columns == ["i", "x", "i_1", "y"]


class TestJoinsAsInputs:
    """A bare join inside another join is inlined, and a join with a filled slot is read under its alias."""

    def three(self, con):
        return con.table("a").join(con.table("b"), "a.id = b.id").join(con.table("c"), "a.id = c.id", how="left")

    def test_a_bare_join_keeps_its_input_aliases_in_the_outer_join(self, con):
        direct = "SELECT * FROM a JOIN b ON a.id = b.id LEFT JOIN c ON a.id = c.id"
        assert_same(con, self.three(con), direct)
        assert self.three(con).columns == ["id", "x", "id", "y", "id", "z"]
        assert_same(con, self.three(con).filter("b.y > 100"), f"{direct} WHERE b.y > 100")

    def test_a_bare_join_renders_inside_the_outer_from(self, con):
        node = parsed(con, self.three(con))
        assert node["from_table"]["type"] == "JOIN"
        assert node["from_table"]["left"]["type"] == "JOIN"

    def test_a_join_with_a_filled_slot_is_read_by_its_output_names(self, con):
        inner = con.table("a").join(con.table("b"), "a.id = b.id").filter("a.x > 0")
        rel = inner.join(con.table("c"), "id")
        direct = "SELECT * FROM (SELECT * FROM a JOIN b ON a.id = b.id WHERE a.x > 0) AS q JOIN c USING (id)"
        assert_same(con, rel, direct)
        assert rel.columns == ["id", "x", "id_1", "y", "z"]

    def test_a_join_with_a_filled_slot_hides_its_input_aliases(self, con):
        inner = con.table("a").join(con.table("b"), "a.id = b.id").filter("a.x > 0")
        with pytest.raises(duckdb.BinderException, match='Referenced table "a" not found'):
            inner.join(con.table("c"), "a.id = c.id")

    def test_a_named_join_with_a_filled_slot_joins_under_its_name(self, con):
        inner = con.table("a").join(con.table("b"), "a.id = b.id").filter("a.x > 0").set_alias("q")
        rel = inner.join(con.table("c"), "q.id = c.id")
        direct = "SELECT * FROM (SELECT * FROM a JOIN b ON a.id = b.id WHERE a.x > 0) AS q JOIN c ON q.id = c.id"
        assert_same(con, rel, direct)


class TestInnerJoinsThatEnterClosed:
    """A bare join that cannot inline into another join still builds and gives the SQL's rows."""

    @pytest.fixture
    def four(self, con):
        con.execute("CREATE TABLE p AS SELECT * FROM (VALUES (1, 10), (2, 20)) v(k, x)")
        con.execute("CREATE TABLE q AS SELECT * FROM (VALUES (1, 100), (2, 200)) v(k2, y)")
        con.execute("CREATE TABLE z AS SELECT * FROM (VALUES (1, 'z1'), (2, 'z2')) v(zk, w)")
        con.execute("CREATE TABLE w AS SELECT * FROM (VALUES (1, true), (2, false)) v(wk, f)")
        return con

    def callable_join(self, con):
        return (
            con.table("p")
            .join(con.table("q"), "k = k2")
            .join(con.table("z"), lambda left, right: left["k"] == right["zk"])
        )

    def test_a_join_on_a_callable_as_the_left_input(self, four):
        rel = self.callable_join(four).join(four.table("w"), "zk = wk")
        assert_same(four, rel, "SELECT * FROM p JOIN q ON k = k2 JOIN z ON k = zk JOIN w ON zk = wk")

    def test_a_join_on_a_callable_as_the_right_input(self, four):
        rel = four.table("w").join(self.callable_join(four), "zk = wk")
        assert_same(four, rel, "SELECT * FROM w JOIN (SELECT * FROM p JOIN q ON k = k2 JOIN z ON k = zk) ON zk = wk")

    def test_an_inner_qualifier_in_a_suffix_join_s_condition_is_the_engine_s_error(self, con):
        # A join with suffix= closes both sides, so the inner a is out of scope.
        con.execute("CREATE TABLE e AS SELECT * FROM (VALUES (1, 7), (2, 8)) v(id, x)")
        pair = con.table("a").join(con.table("b"), "a.id = b.id")
        with pytest.raises(duckdb.BinderException, match=side_error("a")):
            pair.join(con.table("e"), "a.id = e.id", suffix="_e")

    def test_a_suffix_over_a_bare_join_reads_it_as_a_subquery(self, con):
        # The left side is one closed item, so its names are the ones its boundary exposes, id_1 included.
        con.execute("CREATE TABLE e AS SELECT * FROM (VALUES (1, 7), (2, 8)) v(id, x)")
        pair = con.table("a").join(con.table("b"), "a.id = b.id")
        rel = pair.join(
            con.table("e"), duckdb.ColumnExpression(pair.alias, "id") == duckdb.ColumnExpression("e", "id"), suffix="_e"
        )
        direct = (
            "SELECT l.*, e.* RENAME (id AS id_e, x AS x_e) "
            "FROM (SELECT * FROM a JOIN b ON a.id = b.id) AS l JOIN e ON l.id = e.id"
        )
        assert_same(con, rel, direct)
        assert rel.columns == ["id", "x", "id_1", "y", "id_e", "x_e"]


class TestASuffixJoinOverAnInnerJoin:
    """Each side of a suffix join is one closed item, so the inner join's own rules decide its names."""

    def test_an_inner_full_join_using_keeps_one_coalesced_key(self, con):
        left = con.sql("SELECT 1 AS k, 10 AS v", alias="a")
        right = con.sql("SELECT 2 AS k, 20 AS w", alias="b")
        rel = left.join(right, "k", how="outer").join(con.sql("SELECT 30 AS v", alias="c"), "TRUE", suffix="_r")
        direct = (
            "SELECT l.*, c.* RENAME (v AS v_r) FROM (SELECT * FROM (SELECT 1 AS k, 10 AS v) a "
            "FULL OUTER JOIN (SELECT 2 AS k, 20 AS w) b USING (k)) l JOIN (SELECT 30 AS v) c ON TRUE"
        )
        assert_same(con, rel, direct)
        assert rel.columns == ["k", "v", "w", "v_r"]
        assert rel.filter("w = 20").project("k").fetchall() == [(2,)]

    def test_the_hidden_side_of_an_inner_semi_join_renames_nothing(self, con):
        inner = con.sql("SELECT 1 AS k", alias="a").join(con.sql("SELECT 1 AS k, 99 AS v", alias="b"), "k", how="semi")
        rel = inner.join(con.sql("SELECT 1 AS k, 7 AS v", alias="c"), "k", suffix="_r")
        direct = (
            "SELECT * FROM (SELECT * FROM (SELECT 1 AS k) a SEMI JOIN (SELECT 1 AS k, 99 AS v) b USING (k)) l "
            "JOIN (SELECT 1 AS k, 7 AS v) c USING (k)"
        )
        assert_same(con, rel, direct)
        assert rel.columns == ["k", "v"]

    def test_a_closed_inner_join_with_duplicate_names_reads_its_exposed_names(self, con):
        source = con.sql("SELECT 1 AS x", alias="a")
        left = source.project("x AS x_1")
        right = source.join(con.sql("SELECT 2 AS x", alias="b"), "TRUE").join(
            con.sql("SELECT 3 AS x", alias="c"), "TRUE"
        )
        automatic = left.join(right, "TRUE", suffix="_r")
        named = left.join(right.set_alias("r"), "TRUE", suffix="_r")
        direct = (
            "SELECT l.*, r.* RENAME (x_1 AS x_1_r) FROM (SELECT x AS x_1 FROM (SELECT 1 AS x) a) l "
            "JOIN (SELECT * FROM (SELECT 1 AS x) a JOIN (SELECT 2 AS x) b ON TRUE "
            "JOIN (SELECT 3 AS x) c ON TRUE) r ON TRUE"
        )
        assert_same(con, automatic, direct)
        assert_same(con, named, direct)
        assert automatic.columns == named.columns == ["x_1", "x", "x_1_r", "x_2"]


class TestWhatASuffixRenames:
    """A suffix renames the shared names the join does not merge, as the sides are when it is built."""

    @pytest.mark.parametrize(
        ("right", "names", "expected"),
        [
            pytest.param("SELECT 1 AS k, 20 AS b", ["k", "a", "b"], [(1, 10, 20)], id="one_shared_name"),
            pytest.param("SELECT 1 AS k, 10 AS a", ["k", "a"], [(1, 10)], id="every_name_shared"),
        ],
    )
    def test_a_natural_join_merges_what_it_shares_so_a_suffix_renames_nothing(self, con, right, names, expected):
        # A natural join merges every name both sides have, as USING merges its keys.
        left = con.sql("SELECT 1 AS k, 10 AS a", alias="l")
        rel = left.join(con.sql(right, alias="r"), how="natural", suffix="_r")
        assert_same(con, rel, f"SELECT * FROM (SELECT 1 AS k, 10 AS a) l NATURAL JOIN ({right}) r")
        assert rel.columns == names
        assert rel.fetchall() == expected

    @pytest.mark.parametrize(
        ("right", "suffix", "name"),
        [
            pytest.param("SELECT 1 AS k, 20 AS x, 30 AS x_r", "_r", "x_r", id="the_right_side_holds_it"),
            pytest.param("SELECT 1 AS k, 20 AS x, 21 AS x", "_1", "x_1", id="the_right_side_exposes_it"),
        ],
    )
    def test_a_suffixed_name_the_right_side_already_has_is_refused(self, con, right, suffix, name):
        # Every output column counts, the right side's own exposed names as much as the left side's.
        left = con.sql("SELECT 1 AS k, 10 AS x", alias="l")
        with pytest.raises(duckdb.InvalidInputException, match=name):
            left.join(con.sql(right, alias="r"), "k", suffix=suffix)

    def test_a_suffixed_name_the_left_side_already_has_is_refused(self, con):
        # The option exists to avoid a duplicate name, so one it would create is refused at the call.
        left = con.sql("SELECT 1 AS k, 10 AS x, 11 AS x_r", alias="l")
        right = con.sql("SELECT 1 AS k, 20 AS x", alias="r")
        with pytest.raises(duckdb.InvalidInputException, match="x_r"):
            left.join(right, "l.k = r.k", suffix="_r")

    def test_a_side_whose_type_does_not_parse_back_still_takes_a_suffix(self, con):
        # The aggregate state type has no name the parser reads back, which the join without suffix= tolerates.
        state = "SELECT sum(i) EXPORT_STATE AS s, 1 AS k FROM range(3) AS t(i)"
        rel = con.sql(state).join(con.sql("SELECT 1 AS k, 2 AS v"), "TRUE", suffix="_r")
        direct = f"SELECT l.*, r.* RENAME (k AS k_r) FROM ({state}) l JOIN (SELECT 1 AS k, 2 AS v) r ON TRUE"
        assert_same(con, rel, direct)
        assert rel.columns == ["s", "k", "k_r", "v"]

    def test_the_suffix_reads_the_sides_after_the_condition_ran(self, con):
        # A condition that alters a side's table runs first, so the new column is suffixed too.
        con.execute("CREATE TABLE sl AS SELECT 1 AS x, 10 AS y")
        con.execute("CREATE TABLE sr AS SELECT 1 AS x")

        def on(left, right):
            con.execute("ALTER TABLE sr ADD COLUMN y INTEGER DEFAULT 20")
            return left["x"] == right["x"]

        rel = con.table("sl").join(con.table("sr"), on, suffix="_r")
        direct = "SELECT sl.*, sr.* RENAME (x AS x_r, y AS y_r) FROM sl JOIN sr ON sl.x = sr.x"
        assert_same(con, rel, direct)
        assert rel.columns == ["x", "y", "x_r", "y_r"]
        assert rel.fetchall() == [(1, 10, 1, 20)]


class TestStepsAfterARenamingSuffixJoin:
    """A suffix join's SELECT holds the renaming select list, so every step that closes it reads the renamed names."""

    RENAMED = "SELECT l.*, r.* RENAME (k AS k_r, x AS x_r) FROM l JOIN r ON l.k = r.k"

    @pytest.fixture
    def pair(self):
        connection = duckdb.connect()
        connection.execute("CREATE TABLE l AS SELECT * FROM (VALUES (1, 10), (2, 20)) v(k, x)")
        connection.execute("CREATE TABLE r AS SELECT * FROM (VALUES (1, 100), (2, 200)) v(k, x)")
        yield connection
        connection.close()

    def suffixed(self, con):
        rel = con.table("l").join(con.table("r"), "l.k = r.k", suffix="_r")
        assert rel.columns == ["k", "x", "k_r", "x_r"]
        return rel

    @pytest.mark.parametrize(
        ("step", "clause", "ordered"),
        [
            pytest.param(lambda rel: rel.filter("x_r > 100"), "SELECT * FROM ({}) WHERE x_r > 100", False, id="filter"),
            pytest.param(lambda rel: rel.order("k"), "SELECT * FROM ({}) ORDER BY k", True, id="order"),
            pytest.param(lambda rel: rel.project("k, x_r"), "SELECT k, x_r FROM ({})", False, id="project"),
            pytest.param(lambda rel: rel.aggregate("sum(x_r)"), "SELECT sum(x_r) FROM ({})", False, id="aggregate"),
            pytest.param(lambda rel: rel.select_types(["INTEGER"]), "SELECT * FROM ({})", False, id="select_types"),
        ],
    )
    def test_a_closing_step_keeps_the_rename(self, pair, step, clause, ordered):
        assert_same(pair, step(self.suffixed(pair)), clause.format(self.RENAMED), ordered=ordered)

    def test_describe_keeps_the_rename(self, pair):
        expected = pair.sql(f"SELECT * FROM ({self.RENAMED})").describe()
        assert self.suffixed(pair).describe().fetchall() == expected.fetchall()

    def test_an_input_alias_after_it_is_the_engine_s_error(self, pair):
        # The suffix fills the join's select list, so the filter opens a new SELECT where r is out of scope.
        with pytest.raises(duckdb.BinderException, match=side_error("r")):
            self.suffixed(pair).filter("r.x > 100")

    def test_a_step_appended_to_a_reader_with_a_select_list_stays_in_its_select(self, con, tmp_path):
        # A union_by_name reader opens its SELECT with the bound names as its list, and a LIMIT appends to it.
        first, second = tmp_path / "first.csv", tmp_path / "second.csv"
        first.write_text("a,b\n1,2\n")
        second.write_text("a,c\n3,4\n")
        files = [str(first), str(second)]
        rel = con.read_csv(files, union_by_name=True).limit(5)
        assert rel.sql_query().upper().count("SELECT") == 1
        assert_same(con, rel, f"SELECT * FROM read_csv({files!r}, union_by_name = true) LIMIT 5")
        assert rows(rel.fetchall()) == rows([(1, 2, None), (3, None, 4)])


class TestUsingAndAsofJoinsWithASuffix:
    """A USING key is merged and keeps no suffix, while every other shared name takes one."""

    @pytest.fixture
    def pair(self):
        connection = duckdb.connect()
        connection.execute("CREATE TABLE l AS SELECT * FROM (VALUES (1, 10), (2, 20)) v(k, x)")
        connection.execute("CREATE TABLE r AS SELECT * FROM (VALUES (1, 100), (3, 300)) v(k, x)")
        connection.execute("CREATE TABLE trades AS SELECT * FROM (VALUES (1, 10, 'a'), (2, 20, 'b')) v(id, t, x)")
        connection.execute(
            "CREATE TABLE quotes AS SELECT * FROM (VALUES (1, 5, 'A'), (2, 25, 'B'), (2, 15, 'C')) v(id, t, x)"
        )
        yield connection
        connection.close()

    @pytest.mark.parametrize(
        ("how", "kind", "expected"),
        [
            pytest.param("inner", "INNER", [(1, 10, 100)], id="inner"),
            pytest.param("left", "LEFT", [(1, 10, 100), (2, 20, None)], id="left"),
            pytest.param("outer", "FULL OUTER", [(1, 10, 100), (2, 20, None), (3, None, 300)], id="full"),
        ],
    )
    def test_a_using_join_renames_the_shared_data_column(self, pair, how, kind, expected):
        rel = pair.table("l").join(pair.table("r"), "k", how=how, suffix="_r")
        assert_same(pair, rel, f"SELECT * FROM l {kind} JOIN (SELECT k, x AS x_r FROM r) r USING (k)")
        assert rel.columns == ["k", "x", "x_r"]
        assert rows(rel.fetchall()) == rows(expected)

    def test_an_asof_join(self, pair):
        condition = "trades.id = quotes.id AND trades.t >= quotes.t"
        rel = pair.table("trades").join(pair.table("quotes"), condition, how="asof")
        assert_same(pair, rel, f"SELECT * FROM trades ASOF JOIN quotes ON {condition}")

    def test_an_asof_join_with_a_suffix(self, pair):
        condition = "trades.id = quotes.id AND trades.t >= quotes.t"
        rel = pair.table("trades").join(pair.table("quotes"), condition, how="asof", suffix="_q")
        direct = (
            "SELECT trades.*, quotes.* RENAME (id AS id_q, t AS t_q, x AS x_q) "
            f"FROM trades ASOF JOIN quotes ON {condition}"
        )
        assert_same(pair, rel, direct)
        assert rel.columns == ["id", "t", "x", "id_q", "t_q", "x_q"]


class TestNullExtendedRows:
    def test_a_filter_after_a_left_join_removes_the_rows_the_join_extended(self, con):
        # A filter is a WHERE over the joined rows, never part of the ON condition that keeps every left row.
        con.execute("CREATE TABLE nl AS SELECT * FROM (VALUES (1, 10), (2, NULL), (NULL, 30)) v(k, x)")
        con.execute("CREATE TABLE nr AS SELECT * FROM (VALUES (1, 100), (NULL, 300)) v(k, y)")
        rel = con.table("nl").join(con.table("nr"), "nl.k = nr.k", how="left").filter("nr.y > 50")
        assert_same(con, rel, "SELECT * FROM nl LEFT JOIN nr ON nl.k = nr.k WHERE nr.y > 50")
        assert rel.fetchall() == [(1, 10, 1, 100)]
        on_placement = "SELECT * FROM nl LEFT JOIN nr ON nl.k = nr.k AND nr.y > 50"
        assert len(con.execute(on_placement).fetchall()) == 3


class TestADeepSetOperation:
    @pytest.mark.timeout(30)
    def test_twenty_two_nested_unions_build_and_report_a_mistake(self, con):
        # The error path reads each shared input once, so its cost grows with the depth and not with its powers.
        u = con.sql("SELECT 1 AS a", alias="u")
        for _ in range(22):
            u = u.union(u)
        assert u.filter("a = 1").columns == ["a"]
        with pytest.raises(duckdb.BinderException):
            u.filter("zz.a = 1")


class TestALongChainOfSteps:
    def test_six_hundred_filters_read_their_alias_and_join(self, con):
        rel = con.sql("SELECT 1 AS a")
        for _ in range(600):
            rel = rel.filter("a = 1")
        assert rel.alias.startswith("unnamed_relation_")
        assert rel.cross(con.sql("SELECT 2 AS b")).columns == ["a", "b"]


class TestALongChainOfJoins:
    @pytest.mark.timeout(30)
    def test_twenty_four_joins_build_and_render(self, con):
        # Each join decides its inlining once, so the cost of a chain grows with its length and not with its powers.
        for i in range(25):
            con.execute(f"CREATE TABLE t{i} AS SELECT 1 AS k")
        rel = con.table("t0")
        for i in range(1, 25):
            rel = rel.join(con.table(f"t{i}"), f"t0.k = t{i}.k")
        rel.sql_query()
        assert len(rel.columns) == 25


class TestTheLastAliasIsTheName:
    """Of two set_alias calls the later names the relation, in its own steps and in a join."""

    @pytest.fixture
    def named(self, con):
        con.execute("CREATE TABLE kv AS SELECT * FROM (VALUES (1, 30), (2, 20)) v(k, v)")
        con.execute("CREATE TABLE qk AS SELECT * FROM (VALUES (2, 'x'), (3, 'y')) v(k, w)")
        return con.table("kv").set_alias("qk").set_alias("r")

    def test_a_filter_reads_it_under_the_later_alias(self, con, named):
        assert named.alias == "r"
        assert named.filter("r.k > 1").fetchall() == [(2, 20)]
        with pytest.raises(duckdb.BinderException, match='Referenced table "qk" not found'):
            named.filter("qk.k > 1")

    def test_a_join_with_a_table_of_the_earlier_alias_has_no_clash(self, con, named):
        rel = named.join(con.table("qk"), "k")
        assert_same(con, rel, "SELECT * FROM kv AS r JOIN qk USING (k)")
        assert rel.fetchall() == [(2, 20, "x")]


SET_OPERATIONS = [
    pytest.param("union", "UNION ALL", id="union"),
    pytest.param("intersect", "INTERSECT ALL", id="intersect"),
    pytest.param("except_", "EXCEPT ALL", id="except"),
]


class TestSetOperations:
    """Each side is a closed SELECT read as a CTE, and the names are the left side's as the engine gives them."""

    @pytest.mark.parametrize(("verb", "keyword"), SET_OPERATIONS)
    def test_over_a_join_the_names_are_those_its_cte_exposes(self, con, verb, keyword):
        rel = getattr(joined(con).filter("l.i < 3"), verb)(joined(con).filter("y > 10"))
        direct = (
            f"WITH s1 AS NOT MATERIALIZED (SELECT * FROM {J} WHERE l.i < 3), "
            f"s2 AS NOT MATERIALIZED (SELECT * FROM {J} WHERE y > 10) "
            f"SELECT * FROM s1 {keyword} SELECT * FROM s2"
        )
        assert_same(con, rel, direct)
        assert rel.columns == ["i", "x", "i_1", "y"]

    def test_a_join_holding_a_name_with_the_suffix_already(self, con):
        left = con.sql("SELECT 1 AS id, 10 AS x, 11 AS x_1").set_alias("l")
        right = con.sql("SELECT 1 AS id, 20 AS x").set_alias("r")
        pair = left.join(right, ["id"])
        side = "SELECT * FROM (SELECT 1 AS id, 10 AS x, 11 AS x_1) l JOIN (SELECT 1 AS id, 20 AS x) r USING (id)"
        assert pair.columns == ["id", "x", "x_1", "x"]
        assert_same(
            con, pair.union(pair), f"WITH s AS NOT MATERIALIZED ({side}) SELECT * FROM s UNION ALL SELECT * FROM s"
        )
        assert pair.union(pair).columns == ["id", "x", "x_1", "x_2"]

    def test_the_left_side_names_the_result(self, con):
        rel = con.table("t").union(con.table("r").project("i, y::VARCHAR AS y"))
        assert_same(con, rel, "SELECT * FROM t UNION ALL SELECT i, y::VARCHAR FROM r")
        assert rel.columns == ["i", "x"]


class TestUnionByNameOverAJoin:
    """Each side is a CTE whose names the engine deduplicates, so union_by_name matches by those names."""

    @pytest.mark.parametrize(
        ("right", "condition", "names"),
        [
            pytest.param("r", "l.i = r.i", ["i", "x", "i_1", "y"], id="shared_key"),
            pytest.param("rk", "l.i = rk.k", ["i", "x", "k", "x_1"], id="shared_non_key"),
        ],
    )
    @pytest.mark.parametrize(("every", "keyword"), [(True, "UNION ALL BY NAME"), (False, "UNION BY NAME")])
    def test_it_matches_the_names_the_sides_expose(self, con, right, condition, names, every, keyword):
        both = con.table("l").join(con.table(right), condition)
        rel = both.union_by_name(both.filter("l.i > 1"), all=every)
        join = f"l JOIN {right} ON {condition}"
        direct = (
            f"WITH s1 AS NOT MATERIALIZED (SELECT * FROM {join}), "
            f"s2 AS NOT MATERIALIZED (SELECT * FROM {join} WHERE l.i > 1) "
            f"SELECT * FROM s1 {keyword} SELECT * FROM s2"
        )
        assert_same(con, rel, direct)
        assert rel.columns == names


class TestOneAliasPerFromItem:
    """A join's two direct inputs need two names, and a bare join that would repeat one enters closed."""

    @pytest.fixture
    def keyed(self, con):
        con.execute("CREATE TABLE p AS SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) v(k, x)")
        con.execute("CREATE TABLE q AS SELECT * FROM (VALUES (1, 100), (2, 200), (4, 400)) v(k2, y)")
        return con

    def keys(self, con):
        return con.table("q").filter("k2 < 2").project("k2 AS key")

    def test_two_direct_inputs_under_one_name_are_refused(self, keyed):
        with pytest.raises(duckdb.InvalidInputException, match="same alias"):
            keyed.table("q").join(keyed.table("q"), "k2")

    @pytest.mark.parametrize(
        ("how", "kind", "expected"),
        [
            pytest.param("inner", "INNER", [(2, 20, 2, 200, 2, 200)], id="inner"),
            pytest.param("semi", "SEMI", [(2, 20, 2, 200)], id="semi"),
            pytest.param("anti", "ANTI", [(1, 10, 1, 100)], id="anti"),
        ],
    )
    def test_a_table_joined_again_after_a_join_holding_it_closes_that_join(self, keyed, how, kind, expected):
        # The inner q is not free beside the new q, so the bare join enters closed and q names the new input.
        pair = keyed.table("p").join(keyed.table("q"), "k = k2")
        rel = pair.join(keyed.table("q"), "k = q.k2 AND q.y > 150", how=how)
        direct = f"SELECT * FROM (SELECT * FROM p JOIN q ON k = k2) AS pq {kind} JOIN q ON k = q.k2 AND q.y > 150"
        assert_same(keyed, rel, direct)
        assert rows(rel.fetchall()) == expected

    @pytest.mark.parametrize(
        ("how", "kind", "expected"),
        [
            pytest.param("semi", "SEMI", [(1, 10, 1, 100)], id="semi"),
            pytest.param("anti", "ANTI", [(2, 20, 2, 200)], id="anti"),
        ],
    )
    def test_a_derived_input_of_the_same_table_keeps_every_left_column(self, keyed, how, kind, expected):
        # Keys keeps the alias q, so the bare join cannot be inlined beside it and enters as a closed SELECT.
        pair = keyed.table("p").join(keyed.table("q"), "k = k2")
        rel = pair.join(self.keys(keyed), "k = key", how=how)
        direct = f"SELECT * FROM p JOIN q ON k = k2 {kind} JOIN (SELECT k2 AS key FROM q WHERE k2 < 2) keys ON k = key"
        assert_same(keyed, rel, direct)
        assert rel.columns == ["k", "x", "k2", "y"]
        assert rows(rel.fetchall()) == expected

    @pytest.mark.parametrize(
        ("how", "kind", "expected"),
        [
            pytest.param("semi", "SEMI", [(1, 10, 1, 100)], id="semi"),
            pytest.param("anti", "ANTI", [(2, 20, 2, 200)], id="anti"),
        ],
    )
    def test_the_same_chain_with_its_own_alias_keeps_every_left_column(self, keyed, how, kind, expected):
        pair = keyed.table("p").join(keyed.table("q"), "k = k2")
        rel = pair.join(self.keys(keyed).set_alias("keys"), "k = key", how=how)
        direct = f"SELECT * FROM p JOIN q ON k = k2 {kind} JOIN (SELECT k2 AS key FROM q WHERE k2 < 2) keys ON k = key"
        assert_same(keyed, rel, direct)
        assert rel.columns == ["k", "x", "k2", "y"]
        assert rows(rel.fetchall()) == expected

    def test_an_inner_input_named_in_the_condition_of_a_join_that_could_not_inline_is_loud(self, keyed):
        # The bare join entered closed, so q is keys there and the inner q.k2 is reached by its name k2.
        pair = keyed.table("p").join(keyed.table("q"), "k = k2")
        with pytest.raises(duckdb.BinderException, match=engine_only('does not have a column named "k2"')):
            pair.join(self.keys(keyed), "q.k2 = key", how="semi")

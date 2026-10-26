"""Call forms of the relational API that user code relies on, checked against direct SQL on the same connection."""

import datetime
import decimal
import gc
import importlib.util
import json
import re
import uuid

import pytest

import duckdb

pd = pytest.importorskip("pandas")

try:
    import pyarrow
except ImportError:
    pyarrow = None

try:
    import polars
except ImportError:
    polars = None


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) v(i, x)")
    connection.execute("CREATE TABLE r AS SELECT * FROM (VALUES (2, 'B'), (3, 'C')) v(i, y)")
    connection.execute("CREATE TABLE c AS SELECT * FROM (VALUES (2, 'Z')) v(i, z)")
    yield connection
    connection.close()


def top_select(con, rel):
    [(tree,)] = con.execute("SELECT json_serialize_sql(?)", [rel.sql_query()]).fetchall()
    [statement] = json.loads(tree)["statements"]
    return statement["node"]


def select_count(con, rel):
    [(tree,)] = con.execute("SELECT json_serialize_sql(?)", [rel.sql_query()]).fetchall()
    pending, found = [json.loads(tree)], 0
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            found += item.get("type") == "SELECT_NODE"
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return found


def rows(result):
    return sorted(result, key=repr)


def read_through_sql(con):
    local_frame = pd.DataFrame({"i": [2, 3]})  # noqa: F841
    return con.sql("SELECT i FROM local_frame")


def read_through_the_module(con):
    local_frame = pd.DataFrame({"i": [2, 3]})  # noqa: F841
    return duckdb.sql("SELECT i FROM local_frame")


def read_through_a_filter(con):
    local_frame = pd.DataFrame({"i": [2, 3]})  # noqa: F841
    return con.table("t").filter("i IN (SELECT i FROM local_frame)").project("i")


def read_through_a_join_condition(con):
    local_frame = pd.DataFrame({"i": [2, 3]})  # noqa: F841
    return con.table("t").join(con.table("r"), "t.i = r.i AND t.i IN (SELECT i FROM local_frame)").project("t.i")


class TestFunctionLocalFrames:
    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(read_through_sql, id="sql"),
            pytest.param(read_through_the_module, id="module_sql"),
            pytest.param(read_through_a_filter, id="filter_text"),
            pytest.param(read_through_a_join_condition, id="join_condition"),
        ],
    )
    def test_a_frame_local_to_the_calling_function_is_read(self, con, build):
        rel = build(con)
        gc.collect()
        assert rows(rel.fetchall()) == [(2,), (3,)]

    def test_create_table_as_and_insert_select_read_a_local_frame(self, con):
        def create(con):
            local_frame = pd.DataFrame({"i": [2, 3]})  # noqa: F841
            con.sql("CREATE TABLE copied AS SELECT * FROM local_frame")

        def insert(con):
            local_frame = pd.DataFrame({"i": [4]})  # noqa: F841
            con.sql("INSERT INTO copied SELECT i FROM local_frame")

        create(con)
        insert(con)
        assert con.execute("SELECT i FROM copied ORDER BY i").fetchall() == [(2,), (3,), (4,)]

    def test_two_frames_bound_in_turn_under_one_name_each_read_their_own(self, con):
        frame = pd.DataFrame({"a": [1]})
        first = con.sql("SELECT a FROM frame")
        frame = pd.DataFrame({"a": [2]})  # noqa: F841
        second = con.sql("SELECT a FROM frame")
        assert first.fetchall() == [(1,)]
        assert second.fetchall() == [(2,)]
        assert con.execute("SELECT a FROM frame").fetchall() == [(2,)]

    def test_two_objects_captured_under_one_name_combine(self, con):
        # Each capture is read inside the source that captured it, so the two never meet under one name.
        data = pd.DataFrame({"a": [1]})
        first = con.sql("SELECT a FROM data")
        data = pd.DataFrame({"a": [2]})  # noqa: F841
        second = con.sql("SELECT a FROM data")
        assert first.fetchall() == [(1,)]
        assert second.fetchall() == [(2,)]
        assert rows(first.union(second).fetchall()) == [(1,), (2,)]

    def test_two_frames_captured_under_one_name_by_two_steps_close_the_select(self, con):
        # A SELECT holds one capture per name, so the step that captures another object under it reads a new SELECT.
        rel = con.sql("SELECT * FROM range(4) t(a)")
        for df in [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [2]})]:  # noqa: B007
            rel = rel.filter("a IN (SELECT a FROM df)")
        assert rel.fetchall() == [(2,)]

    def test_one_frame_captured_by_two_steps_closes_the_select(self, con):
        # Each capture is its own registration, so a second capture of the name reads a new SELECT.
        df = pd.DataFrame({"a": [1, 2, 3]})  # noqa: F841
        once = con.table("t").filter("i IN (SELECT a FROM df)")
        twice = once.filter("i > 1 AND i IN (SELECT a FROM df)")
        in_one_step = con.table("t").filter("i IN (SELECT a FROM df) AND i > 1 AND i IN (SELECT a FROM df)")
        assert select_count(con, twice) > select_count(con, in_one_step)
        assert twice.sql_query().count("__duckdb_source_") == 2
        assert rows(twice.fetchall()) == [(2, "b"), (3, "c")]

    def test_a_frame_changed_in_place_is_read_as_it_is_at_each_step(self, con):
        df = pd.DataFrame({"k": [1, 2]})
        first = con.sql("SELECT * FROM (VALUES (1), (2), (3)) t(a)").filter("a IN (SELECT k FROM df)")
        df["k"] = [2, 3]
        second = first.filter("a IN (SELECT k FROM df)")
        assert rows(second.fetchall()) == [(2,)]

    @pytest.mark.skipif(
        polars is None or importlib.util.find_spec("pyarrow") is None,
        reason="reads a polars frame, which travels through Arrow",
    )
    def test_two_joins_over_frames_local_to_a_helper_cross_each_reading_their_own(self, con):
        def side(value, n):
            df = polars.DataFrame({"k": [value]})  # noqa: F841
            left = con.sql(f"SELECT * FROM (VALUES (1), (2)) t(a{n})").set_alias(f"l{n}")
            right = con.sql(f"SELECT * FROM (VALUES (1), (2)) t(b{n})").set_alias(f"r{n}")
            return left.join(right, f"a{n} = b{n} AND a{n} IN (SELECT k FROM df)")

        first, second = side(1, 1), side(2, 2)
        gc.collect()
        assert first.cross(second).fetchall() == [(1, 1, 2, 2)]

    def test_a_side_named_after_a_second_capture_closed_the_select(self, con):
        con.execute("CREATE TABLE lk AS SELECT * FROM (VALUES (1, 10), (2, 20)) v(k, x)")
        con.execute("CREATE TABLE rk AS SELECT * FROM (VALUES (1, 100), (2, 200)) v(k, x)")
        joined = con.table("lk").join(con.table("rk"), "lk.k = rk.k")
        df = pd.DataFrame({"a": [1, 2]})
        first = joined.filter("lk.k IN (SELECT a FROM df)")
        df = pd.DataFrame({"a": [100]})  # noqa: F841
        with pytest.raises(duckdb.BinderException, match='Referenced table "rk" not found'):
            first.filter("rk.x IN (SELECT a FROM df)")

    def test_frames_bound_in_turn_to_an_uppercase_name_by_two_steps_close_the_select(self, con):
        rel = con.sql("SELECT * FROM range(4) t(a)")
        for DF in [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [2]})]:  # noqa: B007
            rel = rel.filter("a IN (SELECT a FROM DF)")
        assert rel.fetchall() == [(2,)]

    @pytest.mark.skipif(pyarrow is None, reason="reads an Arrow table")
    def test_one_arrow_table_captured_by_two_steps_over_a_join_closes_the_select(self, con):
        frame = pyarrow.table({"a": [1]})  # noqa: F841
        pair = con.sql("SELECT 1 AS i", alias="l").cross(con.sql("SELECT 1 AS j", alias="r"))
        second = pair.filter("l.i IN (SELECT a FROM frame)").filter("j IN (SELECT a FROM frame)")
        in_one_step = pair.filter("l.i IN (SELECT a FROM frame) AND r.j IN (SELECT a FROM frame)")
        assert select_count(con, second) > select_count(con, in_one_step)
        assert second.sql_query().count("__duckdb_source_") == 2
        assert second.fetchall() == [(1, 1)]

    def test_one_relation_captured_by_two_steps_over_a_join_closes_the_select(self, con):
        few = con.sql("SELECT 2 AS a")  # noqa: F841
        joined = con.table("t").join(con.table("r"), "t.i = r.i")
        twice = joined.filter("t.i IN (SELECT a FROM few)").filter("y IN (SELECT 'B' FROM few)")
        in_one_step = joined.filter("t.i IN (SELECT a FROM few) AND y IN (SELECT 'B' FROM few)")
        assert select_count(con, twice) > select_count(con, in_one_step)
        assert twice.fetchall() == [(2, "b", 2, "B")]

    def test_a_column_spelled_like_a_changed_variable_does_not_close_the_select(self, con):
        pair = con.sql("SELECT 1 AS i, 9 AS frame", alias="l").cross(con.sql("SELECT 1 AS j", alias="r"))
        frame = pd.DataFrame({"a": [1]})
        first = pair.filter("l.i IN (SELECT a FROM frame)")
        frame = pd.DataFrame({"a": [2]})  # noqa: F841
        assert first.filter("l.frame = 9").fetchall() == [(1, 9, 1)]

    def join_reading_one_frame(self, con):
        x = con.sql("SELECT * FROM (VALUES (1), (2), (3)) v(a)").set_alias("x")
        y = con.sql("SELECT * FROM (VALUES (1), (2), (3)) v(b)").set_alias("y")
        df = pd.DataFrame({"a": [1, 2]})  # noqa: F841
        return x.join(y, "x.a = y.b AND x.a IN (SELECT a FROM df)")

    def test_two_join_conditions_capturing_different_frames_under_one_name_each_read_their_own(self, con):
        first = self.join_reading_one_frame(con)
        z = con.sql("SELECT * FROM (VALUES (1), (2), (3)) v(c)").set_alias("z")
        df = pd.DataFrame({"a": [2, 3]})  # noqa: F841
        second = first.join(z, "b = c AND c IN (SELECT a FROM df)")
        assert rows(second.fetchall()) == [(2, 2, 2)]

    def test_a_side_named_in_a_join_condition_after_the_inner_join_closed(self, con):
        first = self.join_reading_one_frame(con)
        z = con.sql("SELECT * FROM (VALUES (1), (2), (3)) v(c)").set_alias("z")
        df = pd.DataFrame({"a": [2, 3]})  # noqa: F841
        with pytest.raises(duckdb.BinderException, match='Referenced table "y" not found'):
            first.join(z, "y.b = z.c AND z.c IN (SELECT a FROM df)")

    def test_frames_whose_names_differ_only_in_case_in_one_statement_meet_the_engine_s_ambiguity(self, con):
        # The engine refuses the statement before any capture runs, so its own error is the answer.
        tbl = pd.DataFrame({"a": [1]})  # noqa: F841
        TBL = pd.DataFrame({"a": [2]})  # noqa: F841
        with pytest.raises(duckdb.BinderException, match='Ambiguous reference to table "tbl"'):
            con.sql("SELECT * FROM tbl, TBL")

    @pytest.mark.parametrize(
        ("combine", "expected"),
        [
            pytest.param(lambda left, right: left.cross(right), [(1, 2)], id="cross"),
            pytest.param(lambda left, right: left.union(right), [(1,), (2,)], id="union"),
        ],
    )
    def test_frames_whose_names_differ_only_in_case_from_two_steps_combine(self, con, combine, expected):
        tbl = pd.DataFrame({"a": [1]})  # noqa: F841
        TBL = pd.DataFrame({"a": [2]})  # noqa: F841
        left = con.sql("SELECT a FROM tbl")
        right = con.sql("SELECT a FROM TBL")
        assert rows(combine(left, right).fetchall()) == expected


class TestNamesInVerbs:
    def test_a_qualified_name_of_a_table_source(self, con):
        assert con.table("t").filter("t.i > 1").fetchall() == con.execute("FROM t WHERE t.i > 1").fetchall()

    def test_rowid_of_a_table_source(self, con):
        assert con.table("t").project("rowid").fetchall() == con.execute("SELECT rowid FROM t").fetchall()

    def test_a_three_way_join_names_all_three_sides(self, con):
        rel = (
            con.table("t").join(con.table("r"), "t.i = r.i").join(con.table("c"), "r.i = c.i").project("t.x, r.y, c.z")
        )
        expected = con.execute("SELECT t.x, r.y, c.z FROM t JOIN r ON t.i = r.i JOIN c ON r.i = c.i").fetchall()
        assert rel.fetchall() == expected

    def test_project_takes_a_name_with_a_space_after_a_filter(self, con):
        rel = con.sql('SELECT 1 AS "my col", 2 AS x').filter("x > 1").project("my col", "x")
        assert rel.columns == ["my col", "x"]
        assert rel.fetchall() == [(1, 2)]

    def test_project_takes_a_qualified_name_after_a_filter(self, con):
        rel = con.table("t").filter("i > 1").project("t.x", "i")
        assert rows(rel.fetchall()) == rows(con.execute("SELECT t.x, i FROM t WHERE i > 1").fetchall())

    @pytest.mark.parametrize("verb", ["project", "select"])
    def test_a_projection_without_arguments_is_refused(self, con, verb):
        with pytest.raises(TypeError):
            getattr(con.table("t"), verb)()


class TestAggregatesInAProjection:
    @pytest.mark.parametrize(
        "spelling",
        [
            pytest.param(lambda: "sum(i) AS s", id="sql_text"),
            pytest.param(
                lambda: duckdb.FunctionExpression("sum", duckdb.ColumnExpression("i")).alias("s"), id="constructor"
            ),
            pytest.param(lambda: duckdb.fn("sum", duckdb.col("i")).alias("s"), id="fn"),
            pytest.param(lambda: duckdb.col("i").sum().alias("s"), id="aggregate_method"),
        ],
    )
    def test_a_projection_may_hold_an_aggregate(self, con, spelling):
        rel = con.table("t").project(spelling())
        assert rel.columns == ["s"]
        assert rel.fetchall() == con.execute("SELECT sum(i) AS s FROM t").fetchall()

    def test_a_projection_binds_once_whatever_its_text_holds(self, con, monkeypatch):
        from duckdb import _seam

        source = con.table("t")
        original = _seam.bind
        binds = []

        def counted(connection, sql, *args, **kwargs):
            binds.append(sql)
            return original(connection, sql, *args, **kwargs)

        monkeypatch.setattr(_seam, "bind", counted)
        source.project("i")
        plain = len(binds)
        binds.clear()
        source.project("abs(i)")
        assert (plain, len(binds)) == (1, 1)


class TestReaderOptions:
    def test_a_csv_read_that_sniffs_one_of_two_shapes_raises_at_the_first_read(self, con, tmp_path):
        # The bind sniffs only the files it is told to, so the second shape is met when the rows are read.
        (tmp_path / "a.csv").write_text("i,s\n1,a\n")
        (tmp_path / "b.csv").write_text("i,s,extra\n2,b,9\n")
        rel = con.read_csv(str(tmp_path / "*.csv"), files_to_sniff=1)
        assert rel.columns == ["i", "s"]
        with pytest.raises(duckdb.InvalidInputException, match="CSV Error"):
            rel.fetchall()


class TestSourcesOverStatementsThatDescribe:
    @pytest.mark.parametrize(
        ("sql", "condition", "direct"),
        [
            pytest.param(
                "DESCRIBE t", "column_name = 'x'", "SELECT * FROM (DESCRIBE t) WHERE column_name = 'x'", id="describe"
            ),
            pytest.param("SHOW TABLES", "name = 't'", "SELECT * FROM (SHOW TABLES) WHERE name = 't'", id="show_tables"),
            pytest.param(
                "SUMMARIZE t",
                "column_name = 'i'",
                "SELECT * FROM (SUMMARIZE t) WHERE column_name = 'i'",
                id="summarize",
            ),
        ],
    )
    def test_a_filter_over_it(self, con, sql, condition, direct):
        assert con.sql(sql).filter(condition).fetchall() == con.execute(direct).fetchall()

    @pytest.mark.parametrize("sql", ["DESCRIBE t", "SHOW TABLES", "SUMMARIZE t"])
    def test_repr_and_show(self, con, sql, capsys):
        rel = con.sql(sql)
        assert rel.columns[0] in repr(rel)
        rel.show()
        assert rel.columns[0] in capsys.readouterr().out


class TestQueryNames:
    def test_a_plain_name_folds_case(self, con):
        assert con.sql("SELECT 1 AS a").query("MyView", "SELECT * FROM myview").fetchall() == [(1,)]


class TestTheReturnType:
    """A relation carries the engine's return type of the statement it holds, so a face can tell a command apart."""

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            pytest.param(lambda con: con.sql("SELECT 1 AS a"), "query_result", id="select"),
            pytest.param(lambda con: con.table("t"), "query_result", id="table"),
            pytest.param(lambda con: con.table("t").filter("i > 1"), "query_result", id="step"),
            pytest.param(lambda con: con.sql("CREATE TABLE made (i INTEGER)"), "nothing", id="ddl"),
            pytest.param(lambda con: con.sql("INSERT INTO t VALUES (9, 'z')"), "changed_rows", id="dml"),
        ],
    )
    def test_it_is_the_engine_s_return_type(self, con, build, expected):
        assert build(con)._return_type == expected


class TestSetOperationsKeepMultiplicity:
    """`except_` and `intersect` keep ALL, so multiplicities and NULLs follow SQL's EXCEPT ALL and INTERSECT ALL."""

    LEFT = "SELECT * FROM (VALUES (1), (1), (1), (NULL), (NULL), (2)) v(a)"
    RIGHT = "SELECT * FROM (VALUES (1), (1), (NULL)) v(a)"

    def sides(self, con):
        return con.sql(self.LEFT), con.sql(self.RIGHT)

    def test_except_keeps_the_remaining_copies(self, con):
        left, right = self.sides(con)
        twin = f"{self.LEFT} EXCEPT ALL {self.RIGHT}"
        expected = sorted(con.execute(twin).fetchall(), key=repr)
        assert sorted(left.except_(right).fetchall(), key=repr) == expected
        assert expected == sorted([(1,), (None,), (2,)], key=repr)

    def test_intersect_keeps_the_lesser_count(self, con):
        left, right = self.sides(con)
        twin = f"{self.LEFT} INTERSECT ALL {self.RIGHT}"
        expected = sorted(con.execute(twin).fetchall(), key=repr)
        assert sorted(left.intersect(right).fetchall(), key=repr) == expected
        assert expected == sorted([(1,), (1,), (None,)], key=repr)


class TestTypeStrings:
    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            pytest.param(lambda con: con.sql("SELECT 1 AS a").map(lambda df: df), "TABLE_FUNCTION_RELATION", id="map"),
            pytest.param(
                lambda con: con.sql("SELECT 1 AS a").query("v", "SELECT * FROM v"), "QUERY_RELATION", id="query"
            ),
            pytest.param(lambda con: con.sql("CALL range(2)"), "MATERIALIZED_RELATION", id="ran_at_the_call"),
            pytest.param(
                lambda con: con.table("t").sample(2),
                "SAMPLE_RELATION",
                id="sample",
            ),
        ],
    )
    def test_type(self, con, build, expected):
        assert build(con).type == expected


class TestModuleLevelFunctions:
    @pytest.mark.parametrize(
        ("call", "direct"),
        [
            pytest.param(lambda f: duckdb.filter(f, "a > 1"), "SELECT * FROM f WHERE a > 1", id="filter"),
            pytest.param(lambda f: duckdb.project(f, "a + 1 AS b"), "SELECT a + 1 AS b FROM f", id="project"),
            pytest.param(lambda f: duckdb.aggregate(f, "sum(a) AS s"), "SELECT sum(a) AS s FROM f", id="aggregate"),
            pytest.param(lambda f: duckdb.distinct(f), "SELECT DISTINCT * FROM f", id="distinct"),
            pytest.param(lambda f: duckdb.limit(f, 2), "SELECT * FROM f LIMIT 2", id="limit"),
            pytest.param(lambda f: duckdb.order(f, "a DESC"), "SELECT * FROM f ORDER BY a DESC", id="order"),
            pytest.param(
                lambda f: duckdb.query_df(f, "fr", "SELECT sum(a) FROM fr"), "SELECT sum(a) FROM f", id="query_df"
            ),
        ],
    )
    def test_a_verb_over_a_data_frame(self, call, direct):
        f = pd.DataFrame({"a": [1, 2, 3]})
        expected = duckdb.connect().execute(direct).fetchall()
        assert rows(call(f).fetchall()) == rows(expected)

    def test_df_of_a_data_frame_is_a_relation(self):
        assert duckdb.df(pd.DataFrame({"a": [1, 2]})).fetchall() == [(1,), (2,)]

    def test_alias_over_a_data_frame(self):
        assert duckdb.alias(pd.DataFrame({"a": [1]}), "fr").alias == "fr"

    def test_write_csv_of_a_data_frame(self, tmp_path):
        path = str(tmp_path / "out.csv")
        duckdb.write_csv(pd.DataFrame({"a": [1, 2]}), path)
        assert duckdb.connect().execute(f"SELECT a FROM read_csv('{path}')").fetchall() == [(1,), (2,)]

    def test_arrow_of_an_arrow_table(self):
        pa = pytest.importorskip("pyarrow")
        assert duckdb.arrow(pa.table({"a": [1, 2]})).fetchall() == [(1,), (2,)]


class TestAGroupsText:
    """A groups text is a GROUP BY list and nothing after it."""

    @pytest.fixture
    def grouped(self, con):
        con.execute("CREATE TABLE g AS SELECT * FROM (VALUES (1, 2, 'a'), (1, 3, 'b'), (2, 5, 'c')) v(id, t, v)")
        return con.table("g")

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda g: g.aggregate("id, sum(t) AS s", "id HAVING sum(t) > 3"), id="having"),
            pytest.param(lambda g: g.aggregate("id, sum(t) AS s", "id LIMIT 1"), id="limit"),
            pytest.param(lambda g: g.list("v", groups="id order by t asc"), id="order_by"),
        ],
    )
    def test_a_clause_after_the_list_is_refused(self, grouped, build):
        with pytest.raises(duckdb.ParserException, match="GROUP BY list"):
            build(grouped)

    @pytest.mark.parametrize(
        "groups",
        [
            pytest.param("ROLLUP (id)", id="rollup"),
            pytest.param("GROUPING SETS ((id), ())", id="grouping_sets"),
            pytest.param("ALL", id="all"),
        ],
    )
    def test_a_grouping_form_still_aggregates(self, con, grouped, groups):
        rel = grouped.aggregate("id, sum(t) AS s", groups)
        assert rows(rel.fetchall()) == rows(con.execute(f"SELECT id, sum(t) AS s FROM g GROUP BY {groups}").fetchall())


class TestQueryTextReadsTheConnectionSettings:
    def test_a_query_text_follows_the_regex_semantics_set_on_the_connection(self, con):
        con.execute("SET regex_match_operator_semantics = 'full'")
        assert con.sql("SELECT 1").query("q", "SELECT 'abc' ~ 'b'").fetchall() == [(False,)]


class TestNamesGivenToProject:
    """Among several arguments, a string that is more than one column reference is a column name, quoted whole."""

    @pytest.mark.parametrize("verb", ["project", "select"])
    def test_a_name_holding_a_semicolon_is_that_column(self, con, verb):
        rel = getattr(con.sql('SELECT 1 AS "a;", 2 AS b'), verb)("a;", "b")
        assert rel.fetchall() == [(1, 2)]

    def test_a_name_holding_a_comment_is_a_missing_column_at_the_call(self, con):
        name = "a " + "-" * 2 + " c"
        with pytest.raises(duckdb.BinderException, match=re.escape(f'Referenced column "{name}" not found')):
            con.sql("SELECT 1 AS a, 2 AS b").project(name, "b")


class TestUserTextIsValidatedNotRewritten:
    """A fragment is checked for the shape its verb takes and spliced as it was written."""

    def test_a_list_comprehension_keeps_its_own_variable(self, con):
        rel = con.sql("SELECT 0 AS x, [5, 6] AS l").project("[x for x in l if x > 0] AS lc")
        assert rel.fetchall() == [([5, 6],)]

    def test_a_projection_follows_the_regex_semantics_set_on_the_connection(self, con):
        con.execute("SET regex_match_operator_semantics = 'full'")
        assert con.sql("SELECT 'abc' AS s").project("s ~ 'b' AS m").fetchall() == [(False,)]

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda rel: rel.filter("x = 1) OR (TRUE"), id="filter_closing_its_parenthesis"),
            pytest.param(lambda rel: rel.filter("x = 1) UNION ALL SELECT 42 WHERE (TRUE"), id="filter_with_a_union"),
            pytest.param(lambda rel: rel.order("x DESC LIMIT 1"), id="order_with_a_limit"),
        ],
    )
    def test_a_fragment_of_another_shape_is_refused_at_the_call(self, con, build):
        with pytest.raises(duckdb.ParserException):
            build(con.sql("SELECT * FROM range(5) t(x)"))

    def test_the_connection_s_depth_limit_decides_for_a_verb_s_text(self, con):
        # Lowered, not raised: the engine's recursion past its default depth of 1000 overflows the main thread stack
        # of the Windows wheel jobs.
        con.execute("SET max_expression_depth = 50")
        base = con.sql("SELECT 1 AS a")
        assert base.project("1" + " + 1" * 40 + " AS d").fetchall() == [(41,)]
        with pytest.raises(duckdb.ParserException, match="Max expression depth limit of 50 exceeded"):
            base.project("1" + " + 1" * 60 + " AS d")


class TestAMarkerInAVerbsText:
    """A parameter marker in a verb's text is refused at the call; it never reads another relation's value."""

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda con: con.sql("SELECT $1 AS a", params=[7]).filter("a = $1"), id="filter"),
            pytest.param(lambda con: con.sql("SELECT $1 AS a", params=[7]).project("a, $1 AS b"), id="project"),
            pytest.param(
                lambda con: con.sql("SELECT $1 AS a", params=[7]).query("q", "SELECT a, $1 AS w FROM q"), id="query"
            ),
            pytest.param(lambda con: con.sql("SELECT * FROM range(3) t(x)").filter("x = ?"), id="question_mark"),
        ],
    )
    def test_it_is_refused(self, con, build):
        with pytest.raises((duckdb.BinderException, duckdb.ParserException)):
            build(con)


class TestTheStarAmongSeveralArguments:
    def test_a_star_beside_an_expression(self, con):
        rel = con.sql("SELECT 1 AS i, 'x' AS c").select("*", duckdb.ColumnExpression("i").alias("j"))
        assert rel.columns == ["i", "c", "j"]
        assert rel.fetchall() == [(1, "x", 1)]

    def test_a_star_beside_a_name(self, con):
        assert con.sql("SELECT 1 AS i, 'x' AS c").select("*", "i").columns == ["i", "c", "i"]


class TestReaderOptionValues:
    """An option value is rendered from its base type's payload, as an expression literal is."""

    @pytest.fixture
    def csv(self, tmp_path):
        path = tmp_path / "in.csv"
        path.write_text("a,b\n1,2\n3,4\n")
        return str(path)

    def test_an_int_enum_option(self, con, csv):
        import enum

        class Rows(enum.IntEnum):
            ONE = 1

        rel = con.read_csv(csv, skiprows=Rows.ONE, header=False)
        assert rel.fetchall() == con.execute(f"SELECT * FROM read_csv('{csv}', skip = 1, header = false)").fetchall()

    def test_a_str_subclass_whose_replace_lies(self, con, csv):
        class Sly(str):
            __slots__ = ()

            def replace(self, *args, **kwargs):
                return str(self)

        assert con.read_csv(csv, quotechar=Sly("'")).fetchall() == [(1, 2), (3, 4)]


VALUES_ROWS = [
    pytest.param([(1, "a", 2.5, True, None)], id="one_row_of_mixed_types"),
    pytest.param([(1,), (2.5,)], id="int_then_float"),
    pytest.param([(1,), (2**40,)], id="int_then_big_int"),
    pytest.param([(None,), (1,)], id="null_then_int"),
    pytest.param([([1, 2],)], id="nested_list"),
    pytest.param([({"k": 1},)], id="dict"),
    pytest.param([(decimal.Decimal("1.25"),)], id="decimal"),
    pytest.param([(b"ab", memoryview(b"cd"))], id="bytes_and_memoryview"),
    pytest.param(
        [(datetime.date(2020, 1, 2), datetime.datetime(2020, 1, 2, 3, 4, tzinfo=datetime.timezone.utc))],
        id="date_and_aware_datetime",
    ),
    pytest.param([(1,), ("2",)], id="int_then_a_string_of_an_int"),
    pytest.param(
        [(uuid.UUID(int=1),), ("00000000-0000-0000-0000-000000000002",)],
        id="uuid_then_a_string_of_a_uuid",
    ),
    pytest.param([(datetime.date(2020, 1, 1),), ("2020-01-02",)], id="date_then_a_string_of_a_date"),
    pytest.param([(b"\x00",), ("x",)], id="blob_then_a_string"),
]


def bound_values(con, rows):
    width = len(rows[0])
    marks = ", ".join("(" + ", ".join("?" for _ in range(width)) + ")" for _ in rows)
    cursor = con.execute(f"SELECT * FROM (VALUES {marks})", [value for row in rows for value in row])
    kinds = [str(column[1]) for column in cursor.description]
    return kinds, cursor.fetchall()


def values_of(con, rows):
    return con.values(*rows) if len(rows) > 1 else con.values(list(rows[0]))


class TestValuesRows:
    """values() holds its rows converted as parameters are, with the types the engine's VALUES rule gives them."""

    @pytest.mark.parametrize("rows", VALUES_ROWS)
    def test_the_types_and_rows_are_those_of_the_same_values_bound(self, con, rows):
        rel = values_of(con, rows)
        assert ([str(kind) for kind in rel.types], rel.fetchall()) == bound_values(con, rows)

    def test_an_incompatible_column_is_refused_at_the_call(self, con):
        with pytest.raises(duckdb.NotImplementedException, match="Cannot combine types"):
            con.values(("x",), (1,))

    def test_the_text_reads_a_registry_source(self, con):
        assert re.search(r"__duckdb_source_", con.values([1]).sql_query())

    def test_its_type_is_a_value_list(self, con):
        assert con.values([1]).type == "VALUE_LIST_RELATION"

    @pytest.mark.parametrize(
        ("text", "params"),
        [
            pytest.param("SELECT $v AS b", {"v": 2}, id="named"),
            pytest.param("SELECT ? AS b", [2], id="positional"),
        ],
    )
    def test_it_combines_with_a_parameterised_relation(self, con, text, params):
        assert con.values([1]).cross(con.sql(text, params=params)).fetchall() == [(1, 2)]

    def test_changing_the_python_rows_after_the_call_changes_nothing(self, con):
        row = [[1, 2]]
        rel = con.values(row)
        row[0].append(3)
        assert rel.fetchall() == [([1, 2],)]

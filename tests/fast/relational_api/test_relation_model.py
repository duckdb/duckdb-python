"""A relation is a value: it holds no rows, binds its schema when it is built and runs again at every terminal."""

import gc
import importlib.util
import re
import traceback
import warnings

import pytest

import duckdb
from duckdb import ColumnExpression, ConstantExpression, StarExpression


@pytest.fixture
def con():
    connection = duckdb.connect()
    register_tick(connection)
    connection.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) v(i, x)")
    connection.execute("CREATE TABLE r AS SELECT * FROM (VALUES (2, 'B'), (3, 'C'), (4, 'D')) v(i, y)")
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


ARROW = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="needs pyarrow")
POLARS = pytest.mark.skipif(
    importlib.util.find_spec("polars") is None or importlib.util.find_spec("pyarrow") is None,
    reason="needs polars and pyarrow; a polars frame travels through Arrow",
)


def counter(con):
    return con.sql("SELECT tick() AS v, 'x' AS w")


def described(con, sql):
    cursor = con.execute(sql)
    description = [(name, str(kind)) for name, kind, *_ in cursor.description]
    cursor.fetchall()
    return description


STEPS = [
    pytest.param(lambda rel: rel, id="source"),
    pytest.param(lambda rel: rel.filter("v > 0"), id="filter"),
    pytest.param(lambda rel: rel.project("v, w"), id="project"),
    pytest.param(lambda rel: rel.order("v"), id="order"),
    pytest.param(lambda rel: rel.limit(5), id="limit"),
    pytest.param(lambda rel: rel.distinct(), id="distinct"),
    pytest.param(lambda rel: rel.aggregate("max(v) AS v, w", "w"), id="aggregate"),
    pytest.param(lambda rel: rel.set_alias("a").cross(rel.set_alias("b").project("w AS w2")), id="cross"),
]


class TestNoRowsAreKept:
    def test_a_read_sees_rows_added_after_the_relation_was_built(self, con):
        rel = con.table("t").filter("i > 1").project("i, upper(x) AS u")
        con.execute("INSERT INTO t VALUES (4, 'd')")
        expected = con.execute("SELECT i, upper(x) AS u FROM t WHERE i > 1").fetchall()
        assert sorted(rel.fetchall()) == sorted(expected)

    @pytest.mark.parametrize("step", STEPS)
    def test_every_read_runs_the_source_again(self, con, step):
        rel = step(counter(con))
        rel.fetchall()
        first = runs(con)
        rel.fetchall()
        assert runs(con) == 2 * first

    @pytest.mark.parametrize("step", STEPS)
    def test_building_a_step_runs_nothing(self, con, step):
        step(counter(con))
        assert not ran(con)


class TestErrorsRaiseWhereTheStepIsBuilt:
    @pytest.mark.parametrize(
        ("build", "error"),
        [
            pytest.param(lambda con: con.table("t").filter("nope > 1"), duckdb.BinderException, id="filter_column"),
            pytest.param(lambda con: con.table("t").project("i +"), duckdb.ParserException, id="project_syntax"),
            pytest.param(lambda con: con.table("t").order("nope"), duckdb.BinderException, id="order_column"),
            pytest.param(lambda con: con.table("t").aggregate("sum(nope)"), duckdb.BinderException, id="aggregate"),
            pytest.param(
                lambda con: con.table("t").join(con.table("r"), "t.nope = r.i"), duckdb.BinderException, id="join_on"
            ),
            pytest.param(lambda con: con.table("t").join(con.table("r"), "nope"), duckdb.BinderException, id="using"),
            pytest.param(lambda con: con.sql("SELECT * FROM nope"), duckdb.CatalogException, id="sql_table"),
            pytest.param(
                lambda con: con.table("t").filter("i > 0").project("x + 1"), duckdb.BinderException, id="types"
            ),
        ],
    )
    def test_the_call_that_builds_a_bad_step_raises(self, con, build, error):
        with pytest.raises(error):
            build(con)

    def test_the_error_is_the_one_the_engine_gives_for_the_same_sql(self, con):
        with pytest.raises(duckdb.BinderException) as from_sql:
            con.execute("SELECT * FROM t WHERE nope > 1")
        with pytest.raises(duckdb.BinderException) as from_step:
            con.table("t").filter("nope > 1")
        assert re.sub(r"\n.*", "", str(from_step.value), flags=re.S) == re.sub(
            r"\n.*", "", str(from_sql.value), flags=re.S
        )

    def test_a_failed_build_leaves_the_transaction_usable(self, con):
        con.execute("BEGIN")
        with pytest.raises(duckdb.BinderException):
            con.table("t").filter("nope > 1")
        assert con.table("t").filter("i > 2").fetchall() == [(3, "c")]
        con.execute("COMMIT")


class TestTheSchemaIsReadWithoutRunning:
    @pytest.mark.parametrize(
        "read",
        [
            pytest.param(lambda rel: rel.columns, id="columns"),
            pytest.param(lambda rel: rel.types, id="types"),
            pytest.param(lambda rel: rel.dtypes, id="dtypes"),
            pytest.param(lambda rel: rel.description, id="description"),
        ],
    )
    @pytest.mark.parametrize("step", STEPS)
    def test_the_observer_does_not_run(self, con, step, read):
        read(step(counter(con)))
        assert not ran(con)

    def test_a_derivation_binds_once_over_typed_empty_inputs_and_reads_nothing(self, con, monkeypatch):
        from duckdb import _seam

        source = counter(con)
        binds, submits = [], []
        bind, submit = _seam.bind, _seam.submit

        def counted_bind(connection, sql, *args, **kwargs):
            binds.append(sql)
            return bind(connection, sql, *args, **kwargs)

        def counted_submit(connection, sql, *args, **kwargs):
            submits.append(sql)
            return submit(connection, sql, *args, **kwargs)

        monkeypatch.setattr(_seam, "bind", counted_bind)
        monkeypatch.setattr(_seam, "submit", counted_submit)
        derived = source.filter("v > 0")
        assert len(binds) == 1
        assert "WHERE FALSE" in binds[0].upper()
        assert "NULL::" in binds[0] or "CAST(NULL" in binds[0].upper()
        assert submits == []
        binds.clear()
        for _ in range(2):
            assert derived.columns == ["v", "w"]
            assert [str(kind) for kind in derived.types] == ["BIGINT", "VARCHAR"]
            assert [str(kind) for kind in derived.dtypes] == ["BIGINT", "VARCHAR"]
            assert [column[0] for column in derived.description] == ["v", "w"]
        assert binds == []
        assert submits == []
        assert not ran(con)

    def test_shape_runs_one_count_and_reads_its_width_from_the_schema(self, con, monkeypatch):
        # Shape is a tuple, so its first item, len(), runs a count each time it is read.
        from duckdb import _seam

        derived = counter(con).filter("v > 0")
        binds, submits = [], []
        bind, submit = _seam.bind, _seam.submit

        def counted_bind(connection, sql, *args, **kwargs):
            binds.append(sql)
            return bind(connection, sql, *args, **kwargs)

        def counted_submit(connection, sql, *args, **kwargs):
            submits.append(sql)
            return submit(connection, sql, *args, **kwargs)

        monkeypatch.setattr(_seam, "bind", counted_bind)
        monkeypatch.setattr(_seam, "submit", counted_submit)
        assert derived.shape == (1, 2)
        assert (len(binds), len(submits)) == (1, 1)
        assert runs(con) == 1
        assert derived.shape == (1, 2)
        assert (len(binds), len(submits)) == (2, 2)
        assert runs(con) == 2

    def test_names_and_types_are_those_the_engine_gives(self, con):
        rel = counter(con).project("v + 1 AS a, upper(w) AS b, [v] AS c, {'k': w} AS d").filter("a > 0")
        expected = described(
            con,
            "SELECT * FROM (SELECT v + 1 AS a, upper(w) AS b, [v] AS c, {'k': w} AS d "
            "FROM (SELECT tick() AS v, 'x' AS w)) WHERE a > 0",
        )
        assert [(name, str(kind)) for name, kind in zip(rel.columns, rel.types, strict=True)] == expected
        assert [(name, str(kind)) for name, kind, *_ in rel.description] == expected
        assert rel.shape[1] == len(expected)

    def test_a_table_altered_after_the_build_shows_the_new_column(self, con):
        rel = con.table("t")
        con.execute("ALTER TABLE t ADD COLUMN z INTEGER")
        assert rel.filter("z IS NULL").fetchall() == con.execute("FROM t WHERE z IS NULL").fetchall()
        assert rel.columns == ["i", "x", "z"]


class TestDuplicateNames:
    def test_a_select_may_name_a_column_twice(self, con):
        rel = con.sql("SELECT 1 AS i, 2 AS i")
        assert rel.columns == ["i", "i"]
        assert rel.fetchall() == [(1, 2)]

    def test_a_join_keeping_both_keys_reports_both(self, con):
        rel = con.table("t").join(con.table("r"), "t.i = r.i")
        assert rel.columns == ["i", "x", "i", "y"]
        assert [name for name, _ in described(con, "SELECT * FROM t JOIN r ON t.i = r.i")] == rel.columns

    @pytest.mark.parametrize(
        ("step", "clause"),
        [
            pytest.param(lambda rel: rel.filter("i = 1"), "WHERE i = 1", id="filter"),
            pytest.param(lambda rel: rel.limit(1), "LIMIT 1", id="limit"),
            pytest.param(lambda rel: rel.distinct(), "", id="distinct"),
            pytest.param(lambda rel: rel.set_alias("q"), "", id="set_alias"),
            pytest.param(lambda rel: rel.order("i"), "ORDER BY i", id="order"),
            pytest.param(lambda rel: rel.sort("i"), "ORDER BY i", id="sort"),
        ],
    )
    def test_a_step_over_a_source_with_duplicates_reads_the_names_the_engine_exposes(self, con, step, clause):
        # The source's text is a closed SELECT, so a step over it sees the engine's suffix on the second name.
        rel = step(con.sql("SELECT 1 AS i, 2 AS i"))
        expected = described(con, f"WITH q AS (SELECT 1 AS i, 2 AS i) SELECT * FROM q {clause}")
        assert [name for name, _ in expected] == ["i", "i_1"]
        assert rel.columns == ["i", "i_1"]
        assert rel.fetchall() == [(1, 2)]

    def test_the_source_itself_keeps_both_names(self, con):
        rel = con.sql("SELECT 1 AS i, 2 AS i")
        assert rel.columns == ["i", "i"]
        assert rel.fetchall() == [(1, 2)]

    @pytest.mark.parametrize(
        "star",
        [
            pytest.param(lambda rel: rel.project("*"), id="star_text"),
            pytest.param(lambda rel: rel.project(StarExpression()), id="star_expression"),
            pytest.param(lambda rel: rel.filter("i = 1").project("*"), id="step_then_star"),
        ],
    )
    def test_a_star_over_a_composed_source_renames_the_second(self, con, star):
        rel = star(con.sql("SELECT 1 AS i, 2 AS i"))
        expected = described(con, "WITH q AS (SELECT 1 AS i, 2 AS i) SELECT * FROM q")
        assert [name for name, _ in expected] == ["i", "i_1"]
        assert rel.columns == ["i", "i_1"]
        assert rel.fetchall() == [(1, 2)]

    def test_a_bare_name_binds_the_first(self, con):
        rel = con.sql("SELECT 1 AS i, 2 AS i")
        assert rel.project("i").fetchall() == [(1,)]
        assert rel.filter("i = 1").fetchall() == [(1, 2)]
        assert rel.filter("i = 2").fetchall() == []

    def test_the_suffixed_name_is_reachable(self, con):
        rel = con.sql("SELECT 1 AS i, 2 AS i").project("*")
        assert rel.project("i_1").fetchall() == [(2,)]

    def test_a_star_over_a_renamed_join_reads_the_names_its_result_exposes(self, con):
        # A set_alias call closes the join's SELECT, so a star over it sees the engine's i_1.
        rel = con.table("t").join(con.table("r"), "t.i = r.i").set_alias("q").project("*")
        expected = described(con, "SELECT * FROM (SELECT * FROM t JOIN r ON t.i = r.i) AS q")
        assert [name for name, _ in expected] == ["i", "x", "i_1", "y"]
        assert rel.columns == ["i", "x", "i_1", "y"]
        assert sorted(rel.fetchall()) == sorted(con.execute("SELECT * FROM t JOIN r ON t.i = r.i").fetchall())

    def test_a_bare_name_of_a_key_both_join_sides_have_is_ambiguous(self, con):
        joined = con.table("t").join(con.table("r"), "t.i = r.i")
        with pytest.raises(duckdb.BinderException, match="Ambiguous reference"):
            joined.project("i")


class TestARelationsCleanupLeavesOtherResults:
    """A relation that goes away, whenever the collector frees it, leaves the connection's open result alone."""

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda con: con.values([1]), id="values"),
            pytest.param(lambda con: con.sql("SELECT 1 AS a"), id="sql"),
            pytest.param(lambda con: con.table("t"), id="table"),
            pytest.param(lambda con: con.table("t").join(con.table("r"), "t.i = r.i"), id="join"),
            pytest.param(lambda con: con.table("t").filter("i > 1"), id="step"),
        ],
    )
    @pytest.mark.parametrize("collect", [pytest.param(True, id="collected"), pytest.param(False, id="not_collected")])
    @pytest.mark.parametrize("keep", [pytest.param(True, id="kept"), pytest.param(False, id="dropped")])
    def test_the_pending_result_still_fetches(self, con, build, collect, keep):
        rel = build(con)
        kept = [rel] if keep else []
        del rel
        con.execute("SELECT 42")
        if collect:
            gc.collect()
        assert con.fetchall() == [(42,)]
        assert len(kept) == int(keep)


class TestLazyPolarsNames:
    """Polars needs unique names, and they are the ones the engine gives the text read as a subquery."""

    @POLARS
    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("SELECT 1 AS a, 2 AS a", id="plain_duplicate"),
            pytest.param("SELECT 1 AS a, 2 AS a, 3 AS a_1", id="beside_a_suffixed_name"),
            pytest.param("SELECT 1 AS a_1, 2 AS a, 3 AS a", id="after_a_suffixed_name"),
        ],
    )
    def test_the_names_are_the_engine_s_for_the_wrapped_text(self, con, text):
        cursor = con.execute(f"SELECT * FROM ({text}) AS q")
        names = [column[0] for column in cursor.description]
        expected = cursor.fetchall()
        frame = con.sql(text).pl(lazy=True).collect()
        assert frame.columns == names
        assert frame.rows() == expected

    @POLARS
    def test_a_text_ending_in_a_line_comment_collects(self, con):
        text = "select 1 as a " + "-" * 2 + " note"
        assert con.sql(text).pl(lazy=True).collect().rows() == [(1,)]


class TestEveryDisplayShowsTheRelationsNames:
    """With duplicate names and a filled LIMIT, every display path shows the names columns reports."""

    def test_the_names_agree_everywhere(self, con):
        rel = con.sql("SELECT 1 AS a, 2 AS b").project("a AS x, b AS x").limit(1)
        assert rel.columns == ["x", "x"]
        bundle = rel._repr_mimebundle_()
        shown = {"repr": repr(rel), "preview": str(rel.preview()), **bundle}
        assert set(bundle) == {"text/plain", "text/html"}
        for name, text in shown.items():
            assert "x_1" not in text, name
            assert text.count("x") >= 2, name


class TestEveryTerminalRunsAgain:
    @pytest.mark.parametrize(
        "terminal",
        [
            pytest.param(lambda rel: rel.fetchall()[0][0], id="fetchall"),
            pytest.param(lambda rel: int(rel.df()["v"][0]), id="df"),
            pytest.param(lambda rel: int(rel.fetchnumpy()["v"][0]), id="fetchnumpy"),
            pytest.param(lambda rel: rel.to_arrow_table().column("v")[0].as_py(), id="arrow_table", marks=ARROW),
            pytest.param(
                lambda rel: rel.to_arrow_reader().read_all().column("v")[0].as_py(), id="arrow_reader", marks=ARROW
            ),
            pytest.param(lambda rel: rel.pl()["v"][0], id="polars", marks=POLARS),
            pytest.param(lambda rel: rel.limit(1).fetchall()[0][0], id="limit"),
            pytest.param(lambda rel: int(re.findall(r"(\d+)\s*│", str(rel))[-1]), id="str"),
        ],
    )
    def test_two_calls_read_two_values(self, con, terminal):
        rel = counter(con).filter("v > 0").project("v")
        assert terminal(rel) == 1
        assert terminal(rel) == 2
        assert runs(con) == 2


LINE_COMMENT = "-" * 2


class TestStringOperands:
    @pytest.mark.parametrize("named", ["x", "X"], ids=["same_case", "other_case"])
    def test_a_string_naming_an_input_column_warns_and_stays_a_value(self, con, named):
        with pytest.warns(UserWarning, match=r"col\(.*lit\("):
            rel = con.table("t").filter(ColumnExpression("x") == named)
        assert rel.fetchall() == []

    def test_a_bare_string_that_names_no_column_is_the_value(self, con):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rel = con.table("t").filter(ColumnExpression("x") == "b")
        assert rel.fetchall() == [(2, "b")]

    def test_an_explicit_constant_naming_a_column_does_not_warn(self, con):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rel = con.table("t").filter(ColumnExpression("x") == ConstantExpression("i"))
        assert rel.fetchall() == []


class TestUpdateValues:
    def test_a_python_value_is_a_value(self, con):
        con.table("t").update({"i": 5}, condition=ColumnExpression("i") == ConstantExpression(2))
        assert con.execute("SELECT * FROM t ORDER BY x").fetchall() == [(1, "a"), (5, "b"), (3, "c")]

    def test_an_expression_is_itself(self, con):
        con.table("t").update({"x": ConstantExpression("z")}, condition=ColumnExpression("i") == ConstantExpression(1))
        assert con.execute("SELECT * FROM t ORDER BY i").fetchall() == [(1, "z"), (2, "b"), (3, "c")]

    def test_a_condition_may_be_sql_text(self, con):
        con.table("t").update({"i": 5}, condition="x = 'b'")
        assert con.execute("SELECT * FROM t ORDER BY x").fetchall() == [(1, "a"), (5, "b"), (3, "c")]

    def test_a_bare_string_value_is_refused(self, con):
        with pytest.raises(TypeError, match=r"col\(.*lit\("):
            con.table("t").update({"x": "i"}, condition=ColumnExpression("i") == ConstantExpression(1))


class TestVerbTexts:
    def test_sort_takes_a_direction(self, con):
        assert con.table("t").sort("i DESC").fetchall() == con.execute("SELECT * FROM t ORDER BY i DESC").fetchall()

    @pytest.mark.parametrize("name", ["sc.tt", "memory.sc.tt"])
    def test_a_dotted_table_name_is_split_like_the_catalog(self, con, name):
        con.execute("CREATE SCHEMA sc")
        con.execute("CREATE TABLE sc.tt AS SELECT 1 AS k")
        rel = con.table(name)
        assert rel.alias == "tt"
        assert rel.fetchall() == [(1,)]

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("SELECT 1 AS a;", id="semicolon"),
            pytest.param("SELECT 1 AS a;  ;\n", id="semicolons"),
            pytest.param(f"SELECT 1 AS a {LINE_COMMENT} a trailing note", id="line_comment"),
        ],
    )
    def test_a_step_over_text_with_a_trailing_semicolon_or_comment(self, con, text):
        assert con.sql(text).filter("a = 1").fetchall() == [(1,)]


class TestOrderAndIndexing:
    def test_order_takes_an_expression(self, con):
        rel = con.table("t").order(ColumnExpression("i").desc())
        assert rel.fetchall() == con.execute("SELECT * FROM t ORDER BY i DESC").fetchall()

    @pytest.mark.parametrize("key", [0, ["i", "x"]], ids=["position", "list_of_names"])
    def test_indexing_by_anything_but_one_name_is_refused(self, con, key):
        with pytest.raises(TypeError, match=r"(?i)column name"):
            con.table("t")[key]

    def test_indexing_by_one_name_gives_that_column(self, con):
        assert con.table("t")["x"].fetchall() == con.execute("SELECT x FROM t").fetchall()


class TestTheCatalogBehindARelation:
    def test_a_cursor_altering_the_table_refreshes_the_relation_and_its_derived_step(self, con):
        source = con.table("t")
        derived = source.filter("i > 1")
        assert derived.columns == ["i", "x"]
        cursor = con.cursor()
        cursor.execute("ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 7")
        assert source.columns == ["i", "x", "z"]
        assert derived.columns == ["i", "x", "z"]
        assert derived.fetchall() == con.execute("SELECT * FROM t WHERE i > 1").fetchall()
        assert derived.project("z").fetchall() == [(7,), (7,)]


class TestNewVerbNames:
    def test_offset_is_the_verb(self, con):
        rel = con.sql('SELECT * FROM (VALUES (1, 2), (3, 4)) v("offset", b)').order("b")
        assert rel.offset(1).fetchall() == [(3, 4)]

    def test_a_column_named_like_a_verb_is_reached_by_item(self, con):
        rel = con.sql('SELECT * FROM (VALUES (1, 2), (3, 4)) v("offset", b)')
        assert sorted(rel["offset"].fetchall()) == [(1,), (3,)]


class TestNotebookDisplay:
    def test_a_display_runs_the_relation_once(self, con):
        bundle = counter(con)._repr_mimebundle_()
        assert runs(con) == 1
        assert {"text/plain", "text/html"} <= set(bundle)


class TestTypesThatDoNotParseBack:
    """A column type whose text the parser cannot read back still carries every step over it."""

    def struct_of_two_cases(self, con):
        pa = pytest.importorskip("pyarrow")
        field = pa.StructArray.from_arrays([pa.array([1]), pa.array([2])], names=["a", "A"])
        return con.from_arrow(pa.table({"s": field}))

    def empty_enum(self, con):
        con.execute("CREATE TYPE empty_choice AS ENUM (SELECT 'x' WHERE FALSE)")
        con.execute("CREATE TABLE chosen (c empty_choice)")
        return con.table("chosen")

    def test_a_struct_whose_field_names_differ_only_in_case_carries_a_step(self, con):
        source = self.struct_of_two_cases(con)
        assert [str(kind) for kind in source.types] == ["STRUCT(a BIGINT, A BIGINT)"]
        assert source.filter("true").fetchall() == [({"a": 1, "A": 2},)]
        assert source.limit(1).columns == ["s"]

    def test_an_empty_enum_column_carries_a_step(self, con):
        source = self.empty_enum(con)
        assert [str(kind) for kind in source.types] == ["ENUM()"]
        assert source.filter("true").fetchall() == []
        assert source.limit(1).columns == ["c"]

    def test_a_struct_field_keeps_its_case_in_the_stand_in(self, con):
        con.execute("SET preserve_identifier_case = 'uppercase'")
        projected = con.sql('SELECT NULL::STRUCT("a" INTEGER) AS s').project("unnest(s)")
        direct = con.execute('SELECT unnest(s) FROM (SELECT NULL::STRUCT("a" INTEGER) AS s)')
        assert projected.columns == [column[0] for column in direct.description] == ["a"]
        direct.fetchall()

    def test_two_collations_in_a_join_are_refused_at_construction(self, con):
        nocase = con.sql("SELECT 'a' COLLATE NOCASE AS s")
        noaccent = con.sql("SELECT 'a' COLLATE NOACCENT AS s")
        with pytest.raises(duckdb.BinderException, match="collation"):
            nocase.join(noaccent, "s")

    def test_one_collation_on_both_sides_joins(self, con):
        rel = con.sql("SELECT 'a' COLLATE NOCASE AS s").join(con.sql("SELECT 'A' COLLATE NOCASE AS s"), "s")
        assert rel.fetchall() == [("a",)]

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(lambda self, con: self.struct_of_two_cases(con), id="struct"),
            pytest.param(lambda self, con: self.empty_enum(con), id="empty_enum"),
        ],
    )
    def test_a_missing_column_in_a_step_over_it_still_raises_the_binder_error(self, con, source):
        with pytest.raises(duckdb.BinderException, match="nope"):
            source(self, con).filter("nope > 1")


class TestWhatAMistakeShows:
    """An error and a migration warning point at the user's own call, not at the client's files."""

    def test_a_misspelled_column_raises_a_short_traceback(self, con):
        with pytest.raises(duckdb.BinderException) as error:
            con.table("t").filter("nope > 1")
        assert len(traceback.extract_tb(error.value.__traceback__)) <= 4

    @POLARS
    def test_a_migration_warning_names_the_calling_file(self, con):
        with pytest.warns(DeprecationWarning, match="(?i)deprecated") as warned:
            con.table("t").fetch_arrow_table()
        assert [record.filename for record in warned] == [__file__]


class TestTheEngineExceptionIsKept:
    """The traceback is cut, but the exception is the engine's own object with every attribute it carries."""

    def test_an_http_error_keeps_its_attributes_and_cause(self):
        from duckdb import _relation

        cause = OSError("connection reset")

        def failing():
            error = duckdb.HTTPException("HTTP Error: 404 (Not Found)")
            error.status_code, error.reason, error.body, error.headers = 404, "Not Found", "", {"x": "y"}
            raise error from cause

        with pytest.raises(duckdb.HTTPException) as caught:
            _relation._closed_as_relation(failing)()
        assert (caught.value.status_code, caught.value.reason, caught.value.headers) == (404, "Not Found", {"x": "y"})
        assert caught.value.__cause__ is cause


class TestTwoConnections:
    """A mistake reports its real cause, and probing a missing name answers rather than raises."""

    def test_joining_relations_of_two_connections_names_the_connections_whatever_the_table_names(self):
        first, second = duckdb.connect(), duckdb.connect()
        first.execute("CREATE TABLE t AS SELECT 1 AS i")
        second.execute("CREATE TABLE t AS SELECT 1 AS i")
        with pytest.raises(duckdb.InvalidInputException, match="different connections"):
            first.table("t").join(second.table("t"), "i")

    def test_hasattr_of_a_missing_name_on_a_closed_connection_is_false(self):
        con = duckdb.connect()
        con.execute("CREATE TABLE t AS SELECT 1 AS i")
        rel = con.table("t")
        con.close()
        assert hasattr(rel, "no_such_column") is False


class TestLimitTakesAnInteger:
    @pytest.mark.parametrize("n", ["5", "x", 1.5, None], ids=["digits", "letters", "float", "none"])
    def test_anything_but_an_integer_is_refused(self, con, n):
        with pytest.raises(TypeError, match="limit"):
            con.table("t").limit(n)

    def test_an_integer_is_taken(self, con):
        assert con.table("t").order("i").limit(2).fetchall() == [(1, "a"), (2, "b")]


class TestEveryRowCountTakesAnyInteger:
    """Every row count takes what `operator.index` takes, a NumPy integer included, and refuses a bool."""

    @pytest.fixture
    def ten(self, con):
        return con.sql("SELECT range AS i FROM range(10)").order("i")

    @pytest.mark.parametrize("kind", ["int64", "int32", "uint8"])
    @pytest.mark.parametrize(
        ("call", "expected"),
        [
            pytest.param(lambda rel, n: rel.limit(n(3)).fetchall(), [(0,), (1,), (2,)], id="limit"),
            pytest.param(lambda rel, n: rel.limit(2, offset=n(8)).fetchall(), [(8,), (9,)], id="limit_offset"),
            pytest.param(lambda rel, n: rel.offset(n(8)).fetchall(), [(8,), (9,)], id="offset"),
            pytest.param(lambda rel, n: rel.head(n(2)).fetchall(), [(0,), (1,)], id="head"),
            pytest.param(lambda rel, n: len(rel.sample(n(2)).fetchall()), 2, id="sample"),
            pytest.param(lambda rel, n: len(list(rel.iter_rows(n(3)))), 10, id="iter_rows"),
        ],
    )
    def test_a_numpy_integer_is_a_count(self, ten, kind, call, expected):
        np = pytest.importorskip("numpy")
        assert call(ten, getattr(np, kind)) == expected

    @pytest.mark.parametrize("value", ["3", 2.5, True], ids=["str", "float", "bool"])
    @pytest.mark.parametrize(
        ("verb", "call"),
        [
            pytest.param("limit", lambda rel, v: rel.limit(v), id="limit"),
            pytest.param("offset", lambda rel, v: rel.offset(v), id="offset"),
            pytest.param("head", lambda rel, v: rel.head(v), id="head"),
            pytest.param("sample", lambda rel, v: rel.sample(v), id="sample"),
            pytest.param("iter_rows", lambda rel, v: list(rel.iter_rows(v)), id="iter_rows"),
        ],
    )
    def test_anything_else_is_refused_naming_the_verb(self, ten, verb, call, value):
        with pytest.raises(TypeError, match=rf"\b{verb}\b"):
            call(ten, value)


class TestTheNotebookFooterCountsTheRelation:
    """The display shows the first rows and says how many rows the relation has, as the box does."""

    def counted(self, con):
        con.execute("CREATE TABLE many AS SELECT range AS i FROM range(2000)")
        return con.table("many")

    def test_the_two_footers_agree_on_the_count(self, con):
        rel = self.counted(con)
        bundle = rel._repr_mimebundle_()
        assert "2000 rows" in repr(rel)
        assert "2000 rows" in bundle["text/plain"]
        assert "2000" in bundle["text/html"]
        assert "11 rows" not in bundle["text/plain"]

    def test_preview_counts_the_relation_too(self, con):
        rel = self.counted(con)
        text = rel.preview(limit=10)
        assert "2000 rows" in text
        assert "11 rows" not in text


KELVIN_KEY = "\u212aey"


def engine_reads(con, sql):
    """The column names the engine gives a statement, or None when the engine finds no such column."""
    try:
        cursor = con.execute(sql)
    except duckdb.BinderException:
        return None
    names = [column[0] for column in cursor.description]
    cursor.fetchall()
    return names


class TestIdentifierCaseFoldsAsTheEngineDoes:
    """A name matches a column exactly when the engine's ASCII only case folding says it does.

    The Kelvin sign lowers to an ASCII k in Python but is not a k to the engine, and neither is any other non ASCII
    letter equal to its case partner, so every answer below is checked against the engine running the same name.
    """

    @pytest.fixture
    def keyed(self, con):
        con.execute("CREATE TABLE keyed (key INTEGER)")
        con.execute("INSERT INTO keyed VALUES (1)")
        return con.table("keyed")

    def test_a_name_differing_only_in_ascii_case_is_the_column(self, con, keyed):
        assert engine_reads(con, 'SELECT "KEY" FROM keyed') == ["key"]
        assert "KEY" in keyed
        assert keyed["KEY"].fetchall() == [(1,)]

    def test_the_kelvin_sign_is_not_a_k(self, con, keyed):
        assert engine_reads(con, f'SELECT "{KELVIN_KEY}" FROM keyed') is None
        assert KELVIN_KEY not in keyed

    def test_an_attribute_spelled_with_the_kelvin_sign_is_no_column(self, keyed):
        with pytest.raises(AttributeError):
            getattr(keyed, KELVIN_KEY)

    def test_with_columns_of_a_kelvin_name_adds_a_column(self, con, keyed):
        rel = keyed.with_columns(**{KELVIN_KEY: duckdb.ConstantExpression(2)})
        assert rel.columns == engine_reads(con, f'SELECT *, 2 AS "{KELVIN_KEY}" FROM keyed') == ["key", KELVIN_KEY]
        assert rel.fetchall() == [(1, 2)]

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(lambda rel: rel, id="source"),
            pytest.param(lambda rel: rel.project("*"), id="star"),
            pytest.param(lambda rel: rel.filter("true"), id="filter"),
            pytest.param(lambda rel: rel.set_alias("q").project("*"), id="set_alias_star"),
        ],
    )
    def test_two_names_differing_by_the_kelvin_sign_are_two_columns(self, con, step):
        source = f'SELECT 1 AS "key", 2 AS "{KELVIN_KEY}"'
        expected = engine_reads(con, f"WITH q AS ({source}) SELECT * FROM q")
        assert expected == ["key", KELVIN_KEY]
        assert step(con.sql(source)).columns == expected

    def test_a_join_keeps_a_kelvin_name_beside_its_lookalike(self, con):
        left = con.sql('SELECT 1 AS id, 1 AS "key"')
        right = con.sql(f'SELECT 1 AS id, 2 AS "{KELVIN_KEY}"')
        expected = engine_reads(
            con, f'SELECT * FROM (SELECT 1 AS id, 1 AS "key") a JOIN (SELECT 1 AS id, 2 AS "{KELVIN_KEY}") b USING (id)'
        )
        assert left.join(right, "id").columns == expected == ["id", "key", KELVIN_KEY]

    def test_a_non_ascii_case_pair_is_two_columns(self, con):
        rel = con.sql('SELECT 1 AS "é", 2 AS "É"')
        assert engine_reads(con, 'WITH q AS (SELECT 1 AS "é", 2 AS "É") SELECT * FROM q') == ["é", "É"]
        assert rel.project("*").columns == ["é", "É"]
        single = con.sql('SELECT 1 AS "é"')
        assert engine_reads(con, 'SELECT "É" FROM (SELECT 1 AS "é")') is None
        assert "é" in single
        assert "É" not in single

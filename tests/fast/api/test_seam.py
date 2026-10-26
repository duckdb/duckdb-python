"""The private engine seam the Python relation layer builds on: binding, submission and the source registry."""

import datetime
import decimal
import gc
import io
import re
import sys
import threading
import types
import weakref
from typing import ClassVar

import pytest

import duckdb

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


def type_names(types_list):
    return [str(t) for t in types_list]


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute("CREATE TABLE t (i INTEGER, s VARCHAR)")
    connection.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, NULL)")
    yield connection
    connection.close()


@pytest.fixture
def blocking_udf(con):
    """A UDF that holds the query it runs in until released, to observe the connection while a call runs."""
    started = threading.Event()
    release = threading.Event()

    def block(x):
        started.set()
        release.wait(30)
        return x

    con.create_function("block", block, ["INTEGER"], "INTEGER")
    yield started, release
    release.set()


@pytest.fixture
def relation_class(monkeypatch):
    """Stands in for duckdb._relation.DuckDBPyRelation, which the seam recognises by its module."""
    module = types.ModuleType("duckdb._relation")

    class DuckDBPyRelation:
        def __init__(self, con, sql, handles=(), registry_text=None, inputs=()) -> None:
            self._con = con
            self._sql = sql
            self._registry_text_value = sql if registry_text is None else registry_text
            self._handles = tuple(handles)
            self._inputs = tuple(inputs)

        def sql_query(self):
            return self._sql

        def _registry_text(self):
            return self._registry_text_value

        def _sources(self):
            # As the relation layer flattens them: this node's handles and every input's
            found = list(self._handles)
            for relation in self._inputs:
                found.extend(relation._sources())
            return found

    module.DuckDBPyRelation = DuckDBPyRelation
    monkeypatch.setitem(sys.modules, "duckdb._relation", module)
    return DuckDBPyRelation


class TestBind:
    def test_select(self, con):
        names, types_list, params, return_type, deferred = con._bind("SELECT i, s, i + 1 AS j FROM t")
        assert names == ["i", "s", "j"]
        assert type_names(types_list) == ["INTEGER", "VARCHAR", "INTEGER"]
        assert all(isinstance(t, duckdb.sqltypes.DuckDBPyType) for t in types_list)
        assert params == []
        assert return_type == "query_result"
        assert deferred is False

    @pytest.mark.parametrize(
        ("sql", "names", "return_type"),
        [
            ("CREATE TABLE u (i INTEGER)", ["Count"], "nothing"),
            ("CREATE TABLE u AS SELECT 1 AS x", ["Count"], "changed_rows"),
            ("INSERT INTO t VALUES (4, 'd')", ["Count"], "changed_rows"),
            ("UPDATE t SET i = 2", ["Count"], "changed_rows"),
            ("DELETE FROM t", ["Count"], "changed_rows"),
            ("INSERT INTO t VALUES (4, 'd') RETURNING i", ["i"], "query_result"),
            ("DROP TABLE t", ["Success"], "nothing"),
            ("SET threads = 2", ["Success"], "nothing"),
            ("DESCRIBE t", ["column_name", "column_type", "null", "key", "default", "extra"], "query_result"),
            ("PIVOT t ON s IN ('a') USING sum(i)", ["a"], "query_result"),
        ],
    )
    def test_return_type_per_statement_kind(self, con, sql, names, return_type):
        bound = con._bind(sql)
        assert bound[0] == names
        assert bound[3] == return_type
        assert bound[4] is False

    def test_binding_runs_nothing(self, con):
        con.execute("CREATE SEQUENCE seq")
        con._bind("CREATE TABLE u (i INTEGER)")
        con._bind("INSERT INTO t VALUES (9, 'z')")
        con._bind("SELECT nextval('seq')")
        assert con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'u'").fetchall() == [(0,)]
        assert con.execute("SELECT count(*) FROM t").fetchall() == [(3,)]
        assert con.execute("SELECT nextval('seq')").fetchall() == [(1,)]

    def test_dynamic_pivot_is_deferred(self, con):
        names, types_list, params, return_type, deferred = con._bind("PIVOT t ON s USING sum(i)")
        assert deferred is True
        assert names == ["unknown"]
        assert type_names(types_list) == ["UNKNOWN"]
        assert params == []
        assert return_type == "query_result"

    def test_parameters(self, con):
        _, types_list, params, _, _ = con._bind("SELECT i FROM t WHERE i < $1 AND s = $2")
        assert type_names(types_list) == ["INTEGER"]
        assert [(name, str(t)) for name, t in params] == [("1", "INTEGER"), ("2", "VARCHAR")]
        _, _, params, _, _ = con._bind("SELECT i FROM t WHERE i BETWEEN ? AND ?")
        assert [(name, str(t)) for name, t in params] == [("1", "INTEGER"), ("2", "INTEGER")]
        _, _, params, _, _ = con._bind("SELECT i FROM t WHERE s = $name AND i > $lo")
        assert sorted((name, str(t)) for name, t in params) == [("lo", "INTEGER"), ("name", "VARCHAR")]

    def test_untyped_parameter_collapses_the_schema(self, con):
        names, types_list, params, return_type, deferred = con._bind("SELECT i + $1 AS v FROM t")
        assert names == ["unknown"]
        assert type_names(types_list) == ["UNKNOWN"]
        assert [(name, str(t)) for name, t in params] == [("1", "UNKNOWN")]
        assert (return_type, deferred) == ("query_result", False)

    def test_spliced_casts_type_every_column(self, con):
        sql = "SELECT i, i + $1::INTEGER AS v, ?::VARCHAR AS w FROM t WHERE s = '$1 ?'"
        names, types_list, params, _, _ = con._bind(sql)
        assert names == ["i", "v", "w"]
        assert type_names(types_list) == ["INTEGER", "INTEGER", "VARCHAR"]
        # a bare ? numbers one past the highest positional index before it
        assert [(name, str(t)) for name, t in params] == [("1", "INTEGER"), ("2", "VARCHAR")]

    @pytest.mark.parametrize(("sql", "count"), [("SELECT 1; SELECT 2", 2), ("", 0), ("/* nothing */", 0)])
    def test_exactly_one_statement(self, con, sql, count):
        with pytest.raises(duckdb.InvalidInputException, match=f"exactly one SQL statement, found {count}"):
            con._bind(sql)

    @pytest.mark.parametrize(
        ("sql", "error"),
        [
            ("SELECT * FROM missing", duckdb.CatalogException),
            ("SELECT nope FROM t", duckdb.BinderException),
            ("SELEC 1", duckdb.ParserException),
        ],
    )
    def test_error_leaves_explicit_transaction_usable(self, con, sql, error):
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (10, 'x')")
        with pytest.raises(error):
            con._bind(sql)
        con.execute("INSERT INTO t VALUES (11, 'y')")
        con.execute("COMMIT")
        assert con.execute("SELECT count(*) FROM t WHERE i >= 10").fetchall() == [(2,)]

    def test_capture_registers_frame_variables(self, con):
        frame = pd.DataFrame({"i": [1, 2]})
        result = con._bind("SELECT frame.i FROM frame JOIN t USING (i)", capture=True)
        assert result[:5] == con._bind("SELECT frame.i FROM frame JOIN t USING (i)")
        [(name, handle)] = result[5]
        assert name == "frame"
        assert handle.name.startswith("__duckdb_source_")
        del frame
        gc.collect()
        sql = f'WITH "frame" AS (SELECT * FROM "{handle.name}") SELECT frame.i FROM frame JOIN t USING (i) ORDER BY 1'
        assert con._submit(sql).fetchall() == [(1,), (2,)]

    def test_capture_skips_catalog_tables_and_registered_sources(self, con):
        t = pd.DataFrame({"i": [99]})  # noqa: F841, a catalog table of the same name wins over the variable
        handle = con._register_source(pa.table({"x": [1]}))
        result = con._bind(f'SELECT * FROM t, "{handle.name}"', capture=True)
        assert result[5] == []

    def test_capture_off_registers_nothing(self, con):
        frame = pd.DataFrame({"i": [1]})  # noqa: F841
        assert len(con._bind("SELECT * FROM frame")) == 5

    def test_beside_an_open_dbapi_result(self, con):
        con.execute("SELECT * FROM range(100000)")
        assert con.fetchone() == (0,)
        assert con._bind("SELECT i FROM t")[0] == ["i"]
        assert con.fetchone() == (1,)

    @pytest.mark.timeout(60)
    def test_refused_while_a_call_runs(self, con, blocking_udf):
        started, release = blocking_udf
        out = []
        worker = threading.Thread(target=lambda: out.append(con._submit("SELECT block(1)").fetchall()))
        worker.start()
        assert started.wait(30)
        with pytest.raises(duckdb.ResourceInUseException, match="busy with another call"):
            con._bind("SELECT 1")
        release.set()
        worker.join()
        assert out == [[(1,)]]

    def test_refused_from_inside_its_own_query(self, con):
        seen = []

        def reenter(x):
            try:
                con._bind("SELECT 1")
            except duckdb.ResourceInUseException as error:
                seen.append(str(error))
            return x

        con.create_function("reenter", reenter, ["INTEGER"], "INTEGER")
        assert con._submit("SELECT reenter(1)").fetchall() == [(1,)]
        assert len(seen) == 1
        assert "inside a query cannot use that query's connection" in seen[0]

    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c._bind("SELECT 1"),
            lambda c: c._submit("SELECT 1"),
            lambda c: c._prepare_script("SELECT 1", None),
            lambda c: c._register_source(pa.table({"i": [1]})),
            lambda c: c._own_sources("v", []),
            lambda c: c._catalog_version(),
            lambda c: c._table_kind("t"),
            lambda c: c._name_scope("main"),
        ],
        ids=[
            "bind",
            "submit",
            "prepare_script",
            "register_source",
            "own_sources",
            "catalog_version",
            "table_kind",
            "name_scope",
        ],
    )
    def test_closed_connection(self, con, call):
        con.close()
        with pytest.raises(duckdb.ConnectionException, match="Connection has already been closed"):
            call(con)

    def test_with_type(self, con):
        assert con._bind("FROM t", with_type=True) == (*con._bind("FROM t"), "select")
        assert con._bind("INSERT INTO t VALUES (1, 'x')", with_type=True)[5] == "insert"
        assert con._bind("DESCRIBE t", with_type=True)[5] == "select"
        assert con._bind("PRAGMA version", with_type=True)[5] == "pragma"
        assert con._bind("CREATE TABLE u (i INTEGER)", with_type=True)[5] == "create"
        frame = pd.DataFrame({"i": [1]})  # noqa: F841
        result = con._bind("SELECT * FROM frame", capture=True, with_type=True)
        assert len(result) == 7
        assert result[5][0][0] == "frame"
        assert result[6] == "select"


class TestSubmit:
    def test_rows(self, con):
        result = con._submit("SELECT i FROM t WHERE i < ? ORDER BY i", [3])
        assert type(result).__name__ == "RelationResult"
        assert result.fetchall() == [(1,), (2,)]

    def test_named_values(self, con):
        assert con._submit("SELECT i FROM t WHERE s = $s", {"s": "b"}).fetchall() == [(2,)]

    def test_arrow_batches(self, con):
        table = con._submit("SELECT * FROM range(10)", None, "arrow", 4).arrow_table()
        assert table.num_rows == 10
        assert [batch.num_rows for batch in table.to_batches()] == [4, 4, 2]

    def test_stream_holds_the_connection_until_closed(self, con):
        result = con._submit("SELECT * FROM range(1000000)", None, "stream", 100)
        with pytest.raises(duckdb.ResourceInUseException, match="open result"):
            con._submit("SELECT 1")
        result.close()
        assert con._submit("SELECT 1").fetchall() == [(1,)]

    def test_retained_results_free_the_connection(self, con):
        first = con._submit("SELECT * FROM range(10000)")
        second = con._submit("SELECT * FROM range(5)", None, "arrow")
        assert second.arrow_table().num_rows == 5
        assert len(first.fetchall()) == 10000

    def test_writes_run_at_submit(self, con):
        result = con._submit("INSERT INTO t VALUES (?, ?)", [4, "d"])
        assert result.rowcount == 1
        assert con.execute("SELECT count(*) FROM t").fetchall() == [(4,)]

    @pytest.mark.parametrize("fmt", ["rows", "arrow", "stream"])
    def test_errors_raise_at_submit(self, con, fmt):
        with pytest.raises(duckdb.CatalogException):
            con._submit("SELECT * FROM missing", None, fmt)
        with pytest.raises(duckdb.ConversionException):
            con._submit("SELECT 'x'::INTEGER", None, fmt)

    def test_value_count_mismatch(self, con):
        with pytest.raises(duckdb.InvalidInputException):
            con._submit("SELECT ?, ?", [1])

    def test_values_must_be_list_or_dict(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="list or a dictionary"):
            con._submit("SELECT ?", 5)

    def test_bad_format_and_batch_size(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="'rows', 'arrow' or 'stream'"):
            con._submit("SELECT 1", None, "chunks")
        with pytest.raises(duckdb.InvalidInputException, match="larger than 0"):
            con._submit("SELECT 1", None, "arrow", 0)

    def test_exactly_one_statement(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="found 2"):
            con._submit("INSERT INTO t VALUES (5, 'e'); SELECT 1")
        assert con.execute("SELECT count(*) FROM t").fetchall() == [(3,)]

    def test_refused_beside_an_open_dbapi_result(self, con):
        con.execute("SELECT * FROM range(100000)")
        con.fetchone()
        with pytest.raises(duckdb.ResourceInUseException, match=r"unfinished result from execute\(\)"):
            con._submit("SELECT 1")
        assert con.fetchone() == (1,)
        con.abort()
        assert con._submit("SELECT 1").fetchall() == [(1,)]

    @pytest.mark.timeout(60)
    def test_releases_the_gil_and_refuses_on_the_same_connection(self, con, blocking_udf):
        started, release = blocking_udf
        other = con.cursor()
        out = []
        worker = threading.Thread(target=lambda: out.append(con._submit("SELECT block(7)").fetchall()))
        worker.start()
        # the UDF can run only if the submit released the GIL
        assert started.wait(30)
        with pytest.raises(duckdb.ResourceInUseException, match="busy with another call"):
            con._submit("SELECT 1")
        assert other.execute("SELECT 42").fetchall() == [(42,)]
        release.set()
        worker.join()
        assert out == [[(7,)]]

    def test_bind_error_in_explicit_transaction(self, con):
        con.execute("BEGIN")
        with pytest.raises(duckdb.CatalogException):
            con._submit("SELECT * FROM missing")
        with pytest.raises(duckdb.TransactionException):
            con._submit("SELECT 1")
        con.execute("ROLLBACK")


class TestTableKind:
    @pytest.fixture
    def catalog(self, con, tmp_path):
        con.execute("CREATE VIEW v AS SELECT 1")
        con.execute("CREATE TEMP VIEW tv AS SELECT 1")
        con.execute("CREATE SCHEMA s")
        con.execute("CREATE TABLE s.in_schema (i INTEGER)")
        con.execute("CREATE VIEW s.schema_view AS SELECT 1")
        con.execute('CREATE TABLE "a.b" (i INTEGER)')
        con.execute("CREATE MACRO m(x) AS x + 1")
        con.execute("CREATE TABLE shadow (i INTEGER)")
        con.execute("CREATE TEMP VIEW shadow AS SELECT 2 AS i")
        con.execute(f"ATTACH '{tmp_path / 'other.duckdb'}' AS other")
        con.execute("CREATE TABLE other.main.far (i INTEGER)")
        con.execute("CREATE TABLE other.ot (i INTEGER)")
        return con

    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            ("t", "table"),
            ("v", "view"),
            ("tv", "view"),
            ("missing", "none"),
            ("a.b", "table"),
            (("a.b",), "table"),
            (("a", "b"), "none"),
            (("s", "in_schema"), "table"),
            (("s", "schema_view"), "view"),
            (("memory", "v"), "view"),
            (("s", "missing"), "none"),
            (("main", "t"), "table"),
            (("memory", "t"), "table"),
            (("other", "main", "far"), "table"),
            (("other", "ot"), "table"),
            (("other", "missing"), "none"),
            (("memory", "main", "t"), "table"),
            (("memory", "missing"), "none"),
            ("shadow", "view"),
            (("memory", "shadow"), "table"),
            (("temp", "shadow"), "view"),
            (["s", "in_schema"], "table"),
            ("duckdb_tables", "view"),
            ("m", "none"),
            (("x", "y", "z", "w"), "none"),
        ],
        ids=str,
    )
    def test_names(self, catalog, name, kind):
        assert catalog._table_kind(name) == kind

    def test_a_dotted_name_is_one_identifier(self, catalog):
        # The two readings of "a.b" are told apart by the shape the layer asks with
        assert catalog._table_kind("a.b") == "table"
        assert catalog._table_kind(("a", "b")) == "none"
        catalog.execute("CREATE SCHEMA a")
        catalog.execute("CREATE TABLE a.b (i INTEGER)")
        assert catalog._table_kind(("a", "b")) == "table"
        assert catalog._table_kind("a.b") == "table"

    def test_a_lone_qualifier_is_a_database_before_a_schema(self, catalog):
        # What the binder resolves: "memory.t" is the table of the memory database, as SQL reads it
        assert catalog._table_kind(("memory", "t")) == "table"
        assert catalog._table_kind(("main", "t")) == "table"
        assert catalog._table_kind(("other", "ot")) == "table"
        assert catalog.execute("SELECT count(*) FROM memory.t").fetchall() == [(3,)]
        assert catalog.execute("SELECT count(*) FROM other.ot").fetchall() == [(0,)]

    def test_a_view_under_either_reading(self, catalog):
        assert catalog._table_kind(("s", "schema_view")) == "view"
        assert catalog._table_kind(("memory", "v")) == "view"
        assert catalog.execute("SELECT * FROM s.schema_view").fetchall() == [(1,)]
        assert catalog.execute("SELECT * FROM memory.v").fetchall() == [(1,)]

    def test_a_catalog_name_skips_a_temporary_view(self, catalog):
        assert catalog._table_kind("shadow") == "view"
        assert catalog._table_kind(("memory", "shadow")) == "table"
        assert catalog._table_kind(("temp", "shadow")) == "view"
        assert catalog.execute("SELECT count(*) FROM memory.shadow").fetchall() == [(0,)]
        assert catalog.execute("SELECT count(*) FROM shadow").fetchall() == [(1,)]

    def test_a_table_of_an_attached_database_on_the_search_path(self, catalog):
        assert catalog._table_kind("ot") == "none"
        catalog.execute("SET search_path = 'other.main'")
        assert catalog._table_kind("ot") == "table"
        assert catalog.execute("SELECT count(*) FROM ot").fetchall() == [(0,)]

    def test_an_ambiguous_lone_qualifier_raises_as_sql_does(self, catalog, tmp_path):
        catalog.execute(f"ATTACH '{tmp_path / 's.duckdb'}' AS s")
        with pytest.raises(duckdb.BinderException, match="Ambiguous reference to catalog or schema"):
            catalog._table_kind(("s", "in_schema"))
        with pytest.raises(duckdb.BinderException, match="Ambiguous reference to catalog or schema"):
            catalog.execute("SELECT * FROM s.in_schema")

    def test_the_search_path_applies(self, catalog):
        assert catalog._table_kind("in_schema") == "none"
        catalog.execute("SET search_path = 's'")
        assert catalog._table_kind("in_schema") == "table"

    def test_it_runs_no_statement(self, catalog):
        version = catalog._catalog_version()
        assert catalog._table_kind("t") == "table"
        assert catalog._name_scope("s") == "schema"
        assert catalog._catalog_version() == version

    def test_beside_an_open_dbapi_result(self, catalog):
        catalog.execute("SELECT * FROM range(100000)")
        assert catalog.fetchone() == (0,)
        assert catalog._table_kind("t") == "table"
        assert catalog._table_kind(("s", "in_schema")) == "table"
        assert catalog._name_scope("other") == "database"
        assert catalog.fetchone() == (1,)
        catalog.abort()

    def test_an_uncommitted_table_of_this_connection(self, catalog):
        catalog.execute("BEGIN")
        catalog.execute("CREATE TABLE in_tx (i INTEGER)")
        assert catalog._table_kind("in_tx") == "table"
        catalog.execute("ROLLBACK")
        assert catalog._table_kind("in_tx") == "none"

    @pytest.mark.parametrize(
        ("name", "error", "match"),
        [
            (5, duckdb.InvalidInputException, "given as a string, or as a tuple"),
            (None, duckdb.InvalidInputException, "given as a string, or as a tuple"),
            ((), duckdb.InvalidInputException, "at least one part"),
            (("a", 5), duckdb.InvalidInputException, "must be a string"),
        ],
        ids=["int", "none", "empty", "part_is_not_a_string"],
    )
    def test_a_name_it_cannot_read(self, con, name, error, match):
        with pytest.raises(error, match=match):
            con._table_kind(name)


class TestNameScope:
    @pytest.mark.parametrize(
        ("name", "scope"),
        [
            ("main", "schema"),
            ("pg_catalog", "schema"),
            ("memory", "database"),
            ("temp", "database"),
            ("t", "none"),
            ("missing", "none"),
        ],
    )
    def test_names(self, con, name, scope):
        assert con._name_scope(name) == scope

    def test_an_attached_database_and_a_new_schema(self, con, tmp_path):
        assert con._name_scope("other") == "none"
        con.execute(f"ATTACH '{tmp_path / 'other.duckdb'}' AS other")
        assert con._name_scope("other") == "database"
        assert con._name_scope("s") == "none"
        con.execute("CREATE SCHEMA s")
        assert con._name_scope("s") == "schema"


class TestContextOwnership:
    """Every seam call that releases the GIL keeps the connection's context alive for its own duration."""

    CALLS: ClassVar = [
        pytest.param(lambda c: c._table_kind("t"), id="table_kind"),
        pytest.param(lambda c: c._name_scope("main"), id="name_scope"),
        pytest.param(lambda c: duckdb._duckdb._arrow_schema(["i"], ["INTEGER"], connection=c), id="arrow_schema"),
        pytest.param(lambda c: c._submit("SELECT 1 AS i").to_box(40, 5), id="to_box"),
        pytest.param(lambda c: c._bind("SELECT * FROM t"), id="bind"),
        pytest.param(lambda c: c._submit("SELECT * FROM t").fetchall(), id="submit"),
    ]

    @pytest.mark.parametrize("call", CALLS)
    @pytest.mark.timeout(60)
    def test_while_another_thread_opens_and_closes_connections_of_the_database(self, con, call):
        stop = threading.Event()
        failures = []

        def churn():
            try:
                while not stop.is_set():
                    cursor = con.cursor()
                    cursor.execute("SELECT 42").fetchall()
                    cursor.close()
                    del cursor
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=churn)
        worker.start()
        try:
            answers = [call(con) for _ in range(50)]
        finally:
            stop.set()
            worker.join()
        assert len(answers) == 50
        assert failures == []

    @pytest.mark.parametrize("call", CALLS)
    @pytest.mark.timeout(60)
    def test_a_close_from_another_thread_is_reported_not_crashed(self, call):
        connection = duckdb.connect()
        connection.execute("CREATE TABLE t AS SELECT 1 AS i")
        call(connection)
        closed = threading.Thread(target=connection.close)
        closed.start()
        closed.join()
        with pytest.raises(duckdb.ConnectionException):
            call(connection)


class TestResultUse:
    """One result serialises its own calls and refuses an overlapping one, as a connection does."""

    @pytest.mark.timeout(120)
    def test_a_second_call_on_one_result_is_refused(self, con):
        # df() releases the GIL for the conversion, so another thread can reach the result while the call runs
        result = con._submit("SELECT * FROM range(4000000)")
        started = threading.Event()
        outcomes = []

        def convert():
            started.set()
            outcomes.append(len(result.df()))

        worker = threading.Thread(target=convert)
        worker.start()
        assert started.wait(10)
        refusals = []
        try:
            while worker.is_alive():
                result.fetchall()
        except duckdb.ResourceInUseException as error:
            refusals.append(str(error))
        worker.join()
        assert refusals, "a call on a result that another call is using should refuse"
        assert "busy with another call" in refusals[0]
        assert outcomes == [4000000]

    def test_a_closed_result_refuses_every_call(self, con):
        result = con._submit("SELECT 1 AS i")
        result.close()
        for call in (result.fetchall, lambda: result.df(), result.drain, result.close):
            with pytest.raises(duckdb.InvalidInputException, match="closed"):
                call()
        with pytest.raises(duckdb.InvalidInputException, match="closed"):
            _ = result.description

    def test_rows_in_the_client_are_read_after_the_connection_closed(self):
        connection = duckdb.connect()
        rows = connection._submit("SELECT * FROM range(3)")
        arrow = connection._submit("SELECT * FROM range(3)", None, "arrow")
        connection.close()
        # The rows are in the client, so reading them needs no connection
        assert rows.fetchall() == [(0,), (1,), (2,)]
        # The Arrow schema is built on the connection, so that one reports instead of reading a context that is gone
        with pytest.raises(duckdb.ConnectionException, match="has already been closed"):
            arrow.arrow_table()

    def test_a_stream_of_a_closed_connection_reports_rather_than_reads_freed_memory(self):
        connection = duckdb.connect()
        stream = connection._submit("SELECT * FROM range(1000000)", None, "stream", 1024)
        connection.close()
        # The query result of an open stream owns its context, so this either reads or reports; it never crashes
        try:
            reader = stream.arrow_reader()
            reader.read_all()
        except (duckdb.Error, OSError):
            pass

    def test_to_box_after_the_connection_closed_is_reported(self):
        connection = duckdb.connect()
        result = connection._submit("SELECT 1 AS i")
        connection.close()
        with pytest.raises(duckdb.ConnectionException, match="has already been closed"):
            result.to_box(40, 5)


class _Cycle:
    """A reference cycle, which only the cyclic collector frees."""

    def __init__(self, con) -> None:
        self.con = con
        self.cycle = self


class TestCollectedWrapperEndsNothing:
    """A connection wrapper that only garbage reaches is collected without ending the connection its owners use."""

    @pytest.mark.parametrize("collect", [False, True], ids=["without_collect", "with_collect"])
    def test_the_relation_reproducer(self, collect):
        rel = duckdb.values(1)
        del rel
        duckdb.execute("SELECT 42 AS answer")
        if collect:
            gc.collect()
        assert duckdb.fetch_df()["answer"].tolist() == [42]

    @pytest.mark.parametrize("collect", [False, True], ids=["without_collect", "with_collect"])
    def test_a_cycle_holding_the_default_connection(self, collect):
        gc.collect()
        _Cycle(duckdb.default_connection())
        duckdb.execute("SELECT 42 AS answer")
        if collect:
            gc.collect()
        assert duckdb.fetch_df()["answer"].tolist() == [42]

    def test_a_collection_during_the_fetch(self):
        gc.collect()
        duckdb.execute("SELECT * FROM range(10000) t(i)")
        for _ in range(50):
            _Cycle(duckdb.default_connection())
        thresholds = gc.get_threshold()
        gc.set_threshold(1, 1, 1)
        try:
            frame = duckdb.fetch_df()
        finally:
            gc.set_threshold(*thresholds)
        assert len(frame) == 10000

    def test_a_connection_only_garbage_owns_is_still_torn_down(self, tmp_path):
        path = str(tmp_path / "owned.db")
        _Cycle(duckdb.connect(path))
        gc.collect()
        # The database is gone with its last owner, so it opens again under another configuration
        reopened = duckdb.connect(path, read_only=True)
        assert reopened.execute("SELECT 1").fetchall() == [(1,)]
        reopened.close()

    def test_close_still_ends_the_connection(self):
        connection = duckdb.connect()
        connection.close()
        with pytest.raises(duckdb.ConnectionException):
            connection.execute("SELECT 1")

    def test_a_cursor_outlives_its_connection_collected_through_a_cycle(self):
        # As when the parent goes by reference count, the cursor holds the connection it was made from.
        connection = duckdb.connect()
        cursor = connection.cursor()
        _Cycle(connection)
        del connection
        gc.collect()
        assert cursor.execute("SELECT 1").fetchall() == [(1,)]


class TestCatalogVersion:
    def test_a_statement_that_only_reads_leaves_it(self, con):
        before = con._catalog_version()
        con.execute("SELECT 42").fetchall()
        con._submit("SELECT i FROM t").fetchall()
        con._bind("CREATE TABLE u (i INTEGER)")
        con._prepare_script("SELECT 42", None)
        con._register_source(pd.DataFrame({"i": [1]}))
        assert con._catalog_version() == before
        assert con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'u'").fetchall() == [(0,)]

    @pytest.mark.parametrize(
        "sql",
        [
            "CREATE TABLE u (i INTEGER)",
            "CREATE VIEW v AS SELECT 1",
            "INSERT INTO t VALUES (9, 'z')",
            "UPDATE t SET i = 0",
            "DELETE FROM t",
            "DROP TABLE t",
            "SET threads = 2",
            "BEGIN",
            "PIVOT t ON s USING sum(i)",
        ],
    )
    @pytest.mark.parametrize(
        "run", [lambda c, sql: c.execute(sql), lambda c, sql: c._submit(sql)], ids=["execute", "submit"]
    )
    def test_a_statement_that_is_not_a_select_counts(self, con, sql, run):
        before = con._catalog_version()
        run(con, sql)
        assert con._catalog_version() == before + 1

    def test_the_statements_of_a_script_count(self, con):
        before = con._catalog_version()
        con.execute("CREATE TABLE u (i INTEGER); INSERT INTO u VALUES (1); SELECT * FROM u").fetchall()
        assert con._catalog_version() == before + 2
        before = con._catalog_version()
        con._prepare_script("CREATE TABLE w (i INTEGER); SELECT 1", None)
        assert con._catalog_version() == before + 1

    def test_every_set_of_an_executemany_counts(self, con):
        before = con._catalog_version()
        con.executemany("INSERT INTO t VALUES (?, ?)", [[4, "d"], [5, "e"]])
        assert con._catalog_version() == before + 2

    def test_register_and_unregister_count(self, con):
        before = con._catalog_version()
        con.register("frame_view", pd.DataFrame({"i": [1]}))
        assert con._catalog_version() == before + 1
        con.unregister("frame_view")
        assert con._catalog_version() == before + 2

    def test_ddl_through_execute_is_seen(self, con):
        # The schema a relation layer caches is re-read because the counter moved, however the DDL ran
        before = con._bind("SELECT * FROM t")
        version = con._catalog_version()
        con.execute("ALTER TABLE t ADD COLUMN extra INTEGER")
        assert con._catalog_version() != version
        assert con._bind("SELECT * FROM t")[0] == [*before[0], "extra"]

    def test_every_connection_of_the_database_counts(self, con):
        cursor = con.cursor()
        before = con._catalog_version()
        cursor.execute("CREATE TABLE u (i INTEGER)")
        assert cursor._catalog_version() == before + 1
        assert con._catalog_version() == before + 1
        cursor.register("frame_view", pd.DataFrame({"i": [1]}))
        assert con._catalog_version() == before + 2

    def test_another_database_counts_its_own(self, con):
        before = con._catalog_version()
        other = duckdb.connect()
        other.execute("CREATE TABLE u (i INTEGER)")
        assert other._catalog_version() == 1
        assert con._catalog_version() == before

    def test_ddl_on_a_cursor_is_seen(self, con):
        before = con._bind("SELECT * FROM t")
        version = con._catalog_version()
        con.cursor().execute("ALTER TABLE t ADD COLUMN extra INTEGER")
        assert con._catalog_version() != version
        assert con._bind("SELECT * FROM t")[0] == [*before[0], "extra"]

    def test_a_statement_counts_once_it_is_submitted(self, con):
        # Counted at submission, so a statement that fails later counts as well: re-reading a schema costs a bind
        before = con._catalog_version()
        with pytest.raises(duckdb.ParserException):
            con.execute("SELEC 1")
        assert con._catalog_version() == before
        with pytest.raises(duckdb.CatalogException):
            con._submit("INSERT INTO missing VALUES (1)")
        assert con._catalog_version() == before + 1


class TestRelationResult:
    SQL = "SELECT i, s, DATE '2020-01-01' + i AS d FROM t ORDER BY i"

    def test_description_and_rowcount(self, con):
        result = con._submit(self.SQL)
        assert [(d[0], str(d[1])) for d in result.description] == [("i", "INTEGER"), ("s", "VARCHAR"), ("d", "DATE")]
        assert result.rowcount == -1

    def test_fetchall_consumes(self, con):
        result = con._submit(self.SQL)
        assert result.fetchall() == [
            (1, "a", datetime.date(2020, 1, 2)),
            (2, "b", datetime.date(2020, 1, 3)),
            (3, None, datetime.date(2020, 1, 4)),
        ]
        assert result.fetchall() == []

    def test_fetchmany_reads_the_rows_in_chunks(self, con):
        result = con._submit(self.SQL)
        assert result.fetchmany(2) == [
            (1, "a", datetime.date(2020, 1, 2)),
            (2, "b", datetime.date(2020, 1, 3)),
        ]
        assert result.fetchmany(2) == [(3, None, datetime.date(2020, 1, 4))]
        assert result.fetchmany(2) == []
        assert result.fetchall() == []

    def test_fetchmany_chunks_add_up_to_fetchall(self, con):
        rows = con._submit("SELECT * FROM range(5000) t(i)").fetchall()
        result = con._submit("SELECT * FROM range(5000) t(i)")
        chunked = []
        while batch := result.fetchmany(777):
            chunked.extend(batch)
        assert chunked == rows

    @pytest.mark.parametrize(("size", "expected"), [(1, 1), (3, 3), (10, 3), (0, 0)])
    def test_fetchmany_sizes(self, con, size, expected):
        assert len(con._submit(self.SQL).fetchmany(size)) == expected

    def test_fetchmany_without_a_size(self, con):
        assert con._submit(self.SQL).fetchmany() == [(1, "a", datetime.date(2020, 1, 2))]

    def test_fetchmany_of_an_empty_result(self, con):
        assert con._submit("SELECT i FROM t WHERE false").fetchmany(5) == []

    @pytest.mark.parametrize("fmt", ["arrow", "stream"])
    def test_fetchmany_needs_the_rows_format(self, con, fmt):
        with pytest.raises(duckdb.InvalidInputException, match="fetchmany needs a result submitted with format"):
            con._submit(self.SQL, None, fmt).fetchmany(2)

    def test_df_and_numpy(self, con):
        frame = con._submit(self.SQL).df()
        assert list(frame.columns) == ["i", "s", "d"]
        assert frame["i"].tolist() == [1, 2, 3]
        assert isinstance(con._submit(self.SQL).df(date_as_object=True)["d"][0], datetime.date)
        arrays = con._submit(self.SQL).fetchnumpy()
        assert arrays["i"].tolist() == [1, 2, 3]

    def test_arrow(self, con):
        table = con._submit(self.SQL, None, "arrow").arrow_table()
        assert table.column_names == ["i", "s", "d"]
        assert table.num_rows == 3
        duplicate = "SELECT 1 AS a, 2 AS a"
        assert con._submit(duplicate, None, "arrow").arrow_table().column_names == ["a", "a"]
        assert con._submit(duplicate, None, "arrow").arrow_table(to_polars=True).column_names == ["a", "a_1"]

    def test_reader_and_capsule_on_a_stream(self, con):
        reader = con._submit("SELECT * FROM range(10)", None, "stream", 3).arrow_reader()
        assert isinstance(reader, pa.RecordBatchReader)
        assert reader.read_all().num_rows == 10
        capsule = con._submit("SELECT * FROM range(10)", None, "stream").arrow_capsule()
        assert pa.RecordBatchReader._import_from_c_capsule(capsule).read_all().num_rows == 10

    def test_to_box(self, con):
        box = con._submit(self.SQL).to_box(80, 10)
        assert "│ 2020-01-02 │" in box
        assert "NULL" in box
        assert "-" in con._submit(self.SQL).to_box(80, 10, null_value="-")
        assert "10 rows" in con._submit("SELECT * FROM range(10)").to_box(max_rows=2)
        assert "│ Column │" in con._submit("SELECT * FROM range(3)").to_box(render_mode="columns")
        assert con._submit(self.SQL).to_box() == con._submit(self.SQL).to_box(None, None)

    @pytest.mark.parametrize(
        ("fmt", "method"),
        [
            ("rows", "arrow_table"),
            ("rows", "arrow_reader"),
            ("rows", "arrow_capsule"),
            ("arrow", "fetchall"),
            ("arrow", "df"),
            ("arrow", "fetchnumpy"),
            ("arrow", "to_box"),
            ("stream", "fetchall"),
        ],
    )
    def test_method_of_the_other_format(self, con, fmt, method):
        result = con._submit(self.SQL, None, fmt)
        with pytest.raises(duckdb.InvalidInputException, match=f"{method} needs a result submitted with format"):
            getattr(result, method)()

    def test_writes(self, con):
        created = con._submit("CREATE TABLE u AS SELECT * FROM range(3)")
        assert [d[0] for d in created.description] == ["Count"]
        assert created.rowcount == 3
        assert con._submit("DELETE FROM t WHERE i > 1").rowcount == 2
        returning = con._submit("INSERT INTO t VALUES (8, 'h') RETURNING i")
        assert returning.rowcount == -1
        assert returning.fetchall() == [(8,)]

    def test_return_type_of_the_statement_that_ran(self, con, tmp_path):
        assert con._submit("SELECT 1").return_type == "query_result"
        assert con._submit("INSERT INTO t VALUES (9, 'z')").return_type == "changed_rows"
        assert con._submit("CREATE TABLE u (i INTEGER)").return_type == "nothing"
        assert con._submit("SELECT 1", None, "arrow").return_type == "query_result"

    def test_return_type_of_a_deferred_group_is_known_once_it_ran(self, con, tmp_path):
        source = duckdb.connect()
        source.execute("CREATE TABLE exported AS SELECT 1 AS i")
        source.execute(f"EXPORT DATABASE '{tmp_path / 'exp'}'")
        sql = f"IMPORT DATABASE '{tmp_path / 'exp'}'"
        # The bind can only guess for a group, and guesses a query result
        assert con._bind(sql)[3:] == ("query_result", True)
        result = con._submit(sql)
        assert result.return_type == "nothing"
        assert result.fetchall() == []

    def test_close_and_drain(self, con):
        result = con._submit(self.SQL)
        result.close()
        with pytest.raises(duckdb.InvalidInputException, match="closed"):
            result.fetchall()
        with pytest.raises(duckdb.InvalidInputException, match="closed"):
            _ = result.description
        stream = con._submit("SELECT * FROM range(1000000)", None, "stream", 10)
        stream.drain()
        assert con._submit("SELECT 1").fetchall() == [(1,)]

    def test_weakref(self, con):
        result = con._submit(self.SQL)
        ref = weakref.ref(result)
        del result
        gc.collect()
        assert ref() is None


@pytest.fixture
def package_wrapper(monkeypatch):
    """Functions of a module inside the duckdb package, standing in for the relation layer between user and seam."""
    module = types.ModuleType("duckdb._seam_test_wrapper")
    exec(
        "frame = 'a decoy the scan must not find'\n"
        "def bind(con, sql):\n"
        "    return con._bind(sql, capture=True)\n"
        "def execute(con, sql):\n"
        "    return con.execute(sql).fetchall()\n",
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def read_captured(con, bound):
    [(_, handle)] = bound[5]
    return con._submit(f'SELECT * FROM "{handle.name}"').fetchall()


class TestFrameScan:
    def test_function_local_frame_through_a_package_wrapper(self, con, package_wrapper):
        def user():
            frame = pd.DataFrame({"v": ["local"]})  # noqa: F841
            return package_wrapper.bind(con, "SELECT * FROM frame"), package_wrapper.execute(con, "FROM frame")

        bound, rows = user()
        assert read_captured(con, bound) == [("local",)]
        assert rows == [("local",)]

    def test_two_frames_each_find_their_own(self, con, package_wrapper):
        def inner():
            frame = pd.DataFrame({"v": ["inner"]})  # noqa: F841
            return package_wrapper.bind(con, "SELECT * FROM frame"), package_wrapper.execute(con, "FROM frame")

        def outer():
            frame = pd.DataFrame({"v": ["outer"]})  # noqa: F841
            first = package_wrapper.bind(con, "SELECT * FROM frame"), package_wrapper.execute(con, "FROM frame")
            return first, inner()

        (outer_bound, outer_rows), (inner_bound, inner_rows) = outer()
        assert read_captured(con, outer_bound) == [("outer",)]
        assert read_captured(con, inner_bound) == [("inner",)]
        assert outer_rows == [("outer",)]
        assert inner_rows == [("inner",)]

    def test_package_globals_are_not_searched(self, con, package_wrapper):
        def user():
            return package_wrapper.execute(con, "FROM frame")

        with pytest.raises(duckdb.CatalogException, match="frame"):
            user()

    def test_scan_all_frames_counts_from_the_user_frame(self, con, package_wrapper):
        def inner():
            return package_wrapper.execute(con, "FROM frame")

        def outer():
            frame = pd.DataFrame({"v": ["outer"]})  # noqa: F841
            return inner()

        with pytest.raises(duckdb.CatalogException, match="frame"):
            outer()
        con.execute("SET python_scan_all_frames = true")
        assert outer() == [("outer",)]


class TestPrepareScript:
    def test_runs_all_but_the_last(self, con):
        kept, text, kind = con._prepare_script("SELECT 42; CREATE TABLE u AS SELECT 1 AS x;\n  FROM u", None)
        assert kept.fetchall() == [(42,)]
        assert (text, kind) == ("FROM u", "select")
        assert con.execute("SELECT * FROM u").fetchall() == [(1,)]

    def test_last_statement_is_not_run(self, con):
        kept, text, kind = con._prepare_script("CREATE TABLE u (i INTEGER); INSERT INTO u VALUES (1);", None)
        assert kept is None
        assert (text, kind) == ("INSERT INTO u VALUES (1)", "insert")
        assert con.execute("SELECT count(*) FROM u").fetchall() == [(0,)]

    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            ("SELECT 1", (None, "SELECT 1", "select")),
            ("SELECT 1;  ", (None, "SELECT 1", "select")),
            ("PIVOT t ON s USING sum(i)", (None, "PIVOT t ON s USING sum(i)", "multi")),
            ("", (None, None, None)),
            ("/* only a comment */", (None, None, None)),
        ],
    )
    def test_single_statement(self, con, sql, expected):
        assert con._prepare_script(sql, None) == expected

    def test_parameters_only_on_the_last(self, con):
        assert con._prepare_script("SELECT 1; SELECT ?", [1])[1:] == ("SELECT ?", "select")
        with pytest.raises(duckdb.NotImplementedException, match="only supported for the last statement"):
            con._prepare_script("SELECT ?; SELECT 1", [1])
        with pytest.raises(duckdb.InvalidInputException, match="Parameters were given"):
            con._prepare_script("", [1])

    def test_parse_error_aborts_an_explicit_transaction(self, con):
        con.execute("BEGIN")
        with pytest.raises(duckdb.ParserException):
            con._prepare_script("SELEC 1; SELECT 2", None)
        with pytest.raises(duckdb.TransactionException, match="aborted"):
            con.execute("SELECT 1")
        con.execute("ROLLBACK")

    def test_later_parse_error_after_earlier_statements_ran(self, con):
        with pytest.raises(duckdb.ParserException):
            con._prepare_script("CREATE TABLE u (i INTEGER); SELEC 2", None)
        assert con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'u'").fetchall() == [(1,)]


class TestArrowSchema:
    def test_names_and_types(self, con):
        schema = duckdb._duckdb._arrow_schema(
            ["a", "a", "s"], [duckdb.sqltypes.INTEGER, "BIGINT", duckdb.sqltypes.VARCHAR], connection=con
        )
        assert schema.names == ["a", "a_1", "s"]
        assert [str(t) for t in schema.types] == ["int32", "int64", "string"]
        assert schema.empty_table().num_rows == 0

    def test_matches_a_result(self, con):
        sql = "SELECT i, s, TIMESTAMPTZ '2020-01-01 00:00:00+00' AS ts FROM t"
        bound = con._bind(sql)
        schema = duckdb._duckdb._arrow_schema(bound[0], bound[1], connection=con)
        assert schema == con._submit(sql, None, "arrow").arrow_table().schema

    def test_connection_settings_apply(self, con):
        con.execute("SET TimeZone = 'Europe/Amsterdam'")
        schema = duckdb._duckdb._arrow_schema(["ts"], ["TIMESTAMPTZ"], connection=con)
        assert str(schema.types[0]) == "timestamp[us, tz=Europe/Amsterdam]"

    def test_default_connection(self):
        assert duckdb._duckdb._arrow_schema(["x"], ["DATE"]).names == ["x"]

    def test_mismatch_and_bad_type(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="one type per name"):
            duckdb._duckdb._arrow_schema(["a", "b"], ["INTEGER"], connection=con)
        with pytest.raises(duckdb.InvalidInputException, match="must be DuckDB types"):
            duckdb._duckdb._arrow_schema(["a"], [object()], connection=con)


class TestRegisterSource:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: pa.table({"i": [1, 2]}),
            lambda: pa.table({"i": [1, 2]}).to_reader(),
            lambda: pd.DataFrame({"i": [1, 2]}),
            lambda: {"i": np.array([1, 2])},
            lambda: np.array([1, 2]),
        ],
        ids=["arrow_table", "arrow_reader", "pandas", "dict", "numpy"],
    )
    def test_objects(self, con, make):
        handle = con._register_source(make())
        assert handle.name.startswith("__duckdb_source_")
        rows = con._submit(f'SELECT * FROM "{handle.name}"').fetchall()
        assert [row[0] for row in rows] == [1, 2]

    def test_bind_and_execute_see_the_entry(self, con):
        handle = con._register_source(pa.table({"i": [1, 2, 3], "s": ["a", "b", "c"]}))
        sql = f'SELECT * FROM "{handle.name}"'
        assert con._bind(sql)[:2] == (["i", "s"], con._bind("SELECT 1::BIGINT, 'a'::VARCHAR")[1])
        assert con.execute(sql).fetchall() == [(1, "a"), (2, "b"), (3, "c")]
        assert con.execute(f"SELECT s FROM {handle.name} WHERE i > 1").fetchall() == [("b",), ("c",)]

    def test_generated_names_are_distinct(self, con):
        first = con._register_source(pa.table({"i": [1]}))
        second = con._register_source(pa.table({"i": [2]}))
        assert first.name != second.name

    def test_name_only_names_the_object_in_errors(self, con):
        first = con._register_source(pa.table({"i": [5]}), "My Source")
        second = con._register_source(pa.table({"i": [6]}), "My Source")
        assert first.name != second.name
        assert first.name.startswith("__duckdb_source_")
        with pytest.raises(duckdb.InvalidInputException, match='Python Object "My Source" of type "object"'):
            con._register_source(object(), "My Source")

    def test_names_are_unique_in_the_process(self):
        first = duckdb.connect()._register_source(pa.table({"i": [1]}))
        second = duckdb.connect()._register_source(pa.table({"i": [1]}))
        prefix, token, number = first.name.rsplit("_", 2)
        assert prefix == "__duckdb_source"
        assert len(token) == 12
        assert second.name == f"{prefix}_{token}_{int(number) + 1}"

    def test_unknown_object(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="not suitable for replacement scans"):
            con._register_source(object())

    def test_lifetime_follows_the_handle(self, con):
        frame = pd.DataFrame({"i": [1, 2]})
        frame_ref = weakref.ref(frame)
        handle = con._register_source(frame)
        name = handle.name
        handle_ref = weakref.ref(handle)
        del frame
        gc.collect()
        assert frame_ref() is not None
        assert con._submit(f'SELECT count(*) FROM "{name}"').fetchall() == [(2,)]
        del handle
        gc.collect()
        assert handle_ref() is None
        assert frame_ref() is None
        with pytest.raises(duckdb.InvalidInputException, match="was released"):
            con._submit(f'SELECT * FROM "{name}"')

    def test_entry_survives_until_a_running_result_ends(self, con):
        handle = con._register_source(pa.table({"i": list(range(5000))}))
        stream = con._submit(f'SELECT * FROM "{handle.name}"', None, "stream", 100)
        del handle
        gc.collect()
        assert stream.arrow_reader().read_all().num_rows == 5000

    def test_shared_by_the_connections_of_a_database(self, con, tmp_path):
        handle = con._register_source(pa.table({"i": [1]}))
        cursor = con.cursor()
        assert cursor._submit(f'SELECT * FROM "{handle.name}"').fetchall() == [(1,)]
        cursor.execute(f'CREATE TEMP VIEW cursor_view AS SELECT i + 1 AS j FROM "{handle.name}"')
        assert cursor.execute("SELECT * FROM cursor_view").fetchall() == [(2,)]
        # Another database never held the name, so it is the catalog's own error there
        with pytest.raises(duckdb.CatalogException, match="does not exist"):
            duckdb.connect()._submit(f'SELECT * FROM "{handle.name}"')

    def test_cursor_reads_a_temp_view_over_a_registered_frame(self, con):
        handle = con._register_source(pd.DataFrame({"i": [1, 2]}))
        cursor = con.cursor()
        cursor._create_view("frame_view", f'SELECT * FROM "{handle.name}"', temporary=True)
        del handle
        gc.collect()
        # The cursor's view holds the frame itself, so it reads after the parent's handle is gone
        assert cursor.execute("SELECT * FROM frame_view ORDER BY i").fetchall() == [(1,), (2,)]

    def test_handle_outlives_its_connection(self, con):
        frame = pd.DataFrame({"i": [1]})
        frame_ref = weakref.ref(frame)
        handle = con._register_source(frame)
        del frame
        con.close()
        gc.collect()
        assert frame_ref() is None
        assert handle.name.startswith("__duckdb_source_")
        del handle
        gc.collect()

    def test_ahead_of_the_replacement_settings(self):
        for config in ({"enable_external_access": False}, {"python_enable_replacements": False}):
            connection = duckdb.connect(config=config)
            handle = connection._register_source(pd.DataFrame({"i": [1]}))
            assert connection._submit(f'SELECT * FROM "{handle.name}"').fetchall() == [(1,)]
            connection.close()

    def test_python_relation_keeps_the_sources_of_its_inputs(self, con, relation_class):
        # A step over a frame source: the step's own node holds no handle, its input does
        inner = con._register_source(pd.DataFrame({"a": [1, 2, 3]}))
        inner_ref = weakref.ref(inner)
        source = relation_class(con, f'SELECT * FROM "{inner.name}"', [inner])
        step = relation_class(con, f'SELECT * FROM "{inner.name}" WHERE a > 1', inputs=[source])
        handle = con._register_source(step)
        del inner, source, step
        gc.collect()
        assert inner_ref() is not None
        assert con._submit(f'SELECT * FROM "{handle.name}" ORDER BY a').fetchall() == [(2,), (3,)]

    def test_python_relation_keeps_its_handles(self, con, relation_class):
        inner = con._register_source(pa.table({"i": [1, 2, 3]}))
        inner_ref = weakref.ref(inner)
        relation = relation_class(con, f'SELECT i * 10 AS j FROM "{inner.name}" WHERE i > 1', [inner])
        outer = con._register_source(relation)
        del inner, relation
        gc.collect()
        assert inner_ref() is not None
        assert con._submit(f'SELECT * FROM "{outer.name}" ORDER BY j').fetchall() == [(20,), (30,)]
        assert con._bind(f'SELECT * FROM "{outer.name}"')[0] == ["j"]
        del outer
        gc.collect()
        assert inner_ref() is None

    def test_python_relation_does_not_own_its_connection(self, relation_class):
        connection = duckdb.connect()
        connection_ref = weakref.ref(connection)
        relation = relation_class(connection, "SELECT 1 AS x")
        handle = connection._register_source(relation)
        del relation, connection
        gc.collect()
        assert connection_ref() is None
        assert handle.name.startswith("__duckdb_source_")

    def test_python_relation_read_on_another_database_is_refused(self, relation_class, tmp_path):
        first = duckdb.connect(tmp_path / "a.db")
        first.execute("CREATE TABLE t AS SELECT 1 AS i UNION ALL SELECT 2")
        second = duckdb.connect(tmp_path / "b.db")
        second.execute("CREATE TABLE t AS SELECT 777 AS i")
        rel = relation_class(first, "SELECT * FROM t ORDER BY i")
        assert first.execute("SELECT * FROM rel").fetchall() == [(1,), (2,)]
        with pytest.raises(duckdb.InvalidInputException, match='Python Object "rel"') as raised:
            second.execute("SELECT * FROM rel")
        assert "a.db" in str(raised.value)
        assert "another database" in str(raised.value)
        # Registered rather than read by name, so the error names the object as the source it was given as
        with pytest.raises(duckdb.InvalidInputException, match="cannot be read on a connection to another database"):
            second._register_source(rel)
        # The other database's own table is what its own name reads
        assert second.execute("SELECT * FROM t").fetchall() == [(777,)]
        first.close()
        second.close()

    def test_python_relation_of_a_closed_connection_is_refused(self, con, relation_class):
        other = duckdb.connect()
        rel = relation_class(other, "SELECT 1 AS i")  # noqa: F841, read by name below
        other.close()
        with pytest.raises(duckdb.InvalidInputException, match="closed"):
            con.execute("SELECT * FROM rel")

    def test_python_relation_read_on_a_sibling_connection_of_the_database(self, relation_class, tmp_path):
        first = duckdb.connect(tmp_path / "a.db")
        first.execute("CREATE TABLE t AS SELECT 5 AS i")
        sibling = duckdb.connect(tmp_path / "a.db")
        cursor = first.cursor()
        rel = relation_class(first, "SELECT * FROM t")  # noqa: F841, read by name below
        assert cursor.execute("SELECT * FROM rel").fetchall() == [(5,)]
        assert sibling.execute("SELECT * FROM rel").fetchall() == [(5,)]
        first.close()
        sibling.close()

    def test_python_relation_read_on_another_connection(self, con, relation_class):
        source = con._register_source(pa.table({"i": [4]}))
        relation = relation_class(con, f'SELECT i FROM "{source.name}"', [source])
        handle = con.cursor()._register_source(relation)
        assert con.cursor()._submit(f'SELECT * FROM "{handle.name}"').fetchall() == [(4,)]

    def test_python_relation_variable_found_by_the_frame_scan(self, con, relation_class):
        source = con._register_source(pa.table({"i": [1, 2]}))
        rel = relation_class(con, f'SELECT i * 2 AS k FROM "{source.name}"', [source])  # noqa: F841
        del source
        assert con.execute("SELECT * FROM rel ORDER BY k").fetchall() == [(2,), (4,)]
        [(name, handle)] = con._bind("SELECT * FROM rel", capture=True)[5]
        assert name == "rel"
        assert con._submit(f'SELECT * FROM "{handle.name}"').fetchall() == [(2,), (4,)]

    def test_python_relation_with_markers_registers(self, con, relation_class):
        relation = relation_class(con, "SELECT * FROM t WHERE i < $1")
        handle = con._register_source(relation)
        with pytest.raises(duckdb.InvalidInputException):
            con._submit(f'SELECT * FROM "{handle.name}"')

    def test_python_relation_is_read_through_its_registry_text(self, con, relation_class):
        # sql_query() keeps the markers, so the entry must be the rendering with the values written in
        relation = relation_class(con, "SELECT * FROM t WHERE i < $1", registry_text="SELECT * FROM t WHERE i < 3")
        handle = con._register_source(relation)
        assert con._submit(f'SELECT i FROM "{handle.name}" ORDER BY i').fetchall() == [(1,), (2,)]
        rel = relation  # noqa: F841, read by name below
        assert con.execute("SELECT i FROM rel ORDER BY i").fetchall() == [(1,), (2,)]

    def test_python_relation_that_is_not_a_select(self, con, relation_class):
        relation = relation_class(con, "SELECT 1; SELECT 2")
        with pytest.raises(duckdb.InvalidInputException, match="exactly one SELECT statement"):
            con._register_source(relation)

    def test_lookalike_class_is_not_a_relation(self, con, relation_class):
        class DuckDBPyRelation:
            _con = con
            _handles = ()

            def sql_query(self):
                return "SELECT 1"

            def _registry_text(self):
                return "SELECT 1"

        with pytest.raises(duckdb.InvalidInputException, match="not suitable for replacement scans"):
            con._register_source(DuckDBPyRelation())


class TestSourceIdentity:
    def test_an_object_read_once_is_one_source(self, con):
        reader = pa.table({"i": [1, 2]}).to_reader()
        first = con._register_source(reader)
        second = con._register_source(reader)
        assert second.name == first.name
        assert con._submit(f'SELECT * FROM "{first.name}"').fetchall() == [(1,), (2,)]

    def test_a_frame_is_adapted_afresh_each_time(self, con):
        frame = pd.DataFrame({"i": [1, 2]})
        first = con._register_source(frame)
        frame["j"] = [3, 4]
        second = con._register_source(frame)
        assert second.name != first.name
        assert con._submit(f'SELECT * FROM "{second.name}"').fetchall() == [(1, 3), (2, 4)]

    def test_one_statement_reads_it_twice(self, con):
        frame = pd.DataFrame({"i": [1, 2]})
        left = con._register_source(frame)
        right = con._register_source(frame)
        sql = f'SELECT l.i, r.i FROM "{left.name}" l JOIN "{right.name}" r ON l.i = r.i ORDER BY 1'
        assert con._submit(sql).fetchall() == [(1, 1), (2, 2)]

    def test_equal_objects_are_not_the_same_object(self, con):
        first = con._register_source(pd.DataFrame({"i": [1]}))
        second = con._register_source(pd.DataFrame({"i": [1]}))
        assert first.name != second.name

    def test_the_entry_lives_while_either_handle_does(self, con):
        reader = pa.table({"i": [1]}).to_reader()
        first = con._register_source(reader)
        second = con._register_source(reader)
        name = first.name
        del first
        gc.collect()
        assert con._submit(f'SELECT * FROM "{name}"').fetchall() == [(1,)]
        del second
        gc.collect()
        with pytest.raises(duckdb.InvalidInputException, match="was released"):
            con._submit(f'SELECT * FROM "{name}"')

    def test_a_dropped_key_is_not_resurrected(self, con):
        frame = pd.DataFrame({"i": [1, 2]})
        first = con._register_source(frame)
        name = first.name
        del first
        gc.collect()
        second = con._register_source(frame)
        assert second.name != name
        assert con._submit(f'SELECT * FROM "{second.name}"').fetchall() == [(1,), (2,)]

    @pytest.mark.parametrize(
        "make",
        [
            lambda: pa.table({"i": [1, 2]}).to_reader(),
            lambda: pa.table({"i": [1, 2]}).__arrow_c_stream__(),
        ],
        ids=["arrow_reader", "arrow_stream_capsule"],
    )
    def test_a_kind_read_once_reuses_its_registration(self, con, make):
        obj = make()
        first = con._register_source(obj)
        assert con._register_source(obj).name == first.name
        # A one shot stream proves the object was adapted once, not twice
        assert con._submit(f'SELECT * FROM "{first.name}"').fetchall()[0] == (1,)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: pa.table({"i": [1]}),
            lambda: pd.DataFrame({"i": [1]}),
            lambda: np.array([1]),
            lambda: {"i": np.array([1])},
            lambda: [np.array([1])],
        ],
        ids=["arrow_table", "pandas", "numpy", "dict", "list"],
    )
    def test_a_kind_read_again_gets_a_registration_each_time(self, con, make):
        obj = make()
        first = con._register_source(obj)
        second = con._register_source(obj)
        assert second.name != first.name
        assert con._submit(f'SELECT * FROM "{second.name}"').fetchall()[0] == (1,)

    def test_each_connection_registers_its_own(self, con):
        frame = pd.DataFrame({"i": [1]})
        assert con.cursor()._register_source(frame).name != con._register_source(frame).name

    def test_register_adapts_the_frame_afresh(self, con):
        frame = pd.DataFrame({"i": [1]})
        handle = con._register_source(frame)
        con.register("frame_view", frame)
        view = con.execute("SELECT sql FROM duckdb_views() WHERE view_name = 'frame_view'").fetchall()[0][0]
        assert handle.name not in view
        assert con.execute("SELECT * FROM frame_view").fetchall() == [(1,)]
        del handle
        gc.collect()
        assert con.execute("SELECT * FROM frame_view").fetchall() == [(1,)]

    def test_a_remembered_object_is_not_kept_alive(self, relation_class):
        connection = duckdb.connect()
        relation = relation_class(connection, "SELECT 1 AS x")
        relation_ref = weakref.ref(relation)
        handle = connection._register_source(relation)
        del relation
        gc.collect()
        assert relation_ref() is None
        assert handle.name.startswith("__duckdb_source_")

    def test_a_new_object_at_a_freed_address_gets_its_own_key(self, con, relation_class):
        first = relation_class(con, "SELECT 1 AS x")
        handle = con._register_source(first)
        del first
        gc.collect()
        second = relation_class(con, "SELECT 2 AS x")
        other = con._register_source(second)
        assert other.name != handle.name
        assert con._submit(f'SELECT * FROM "{other.name}"').fetchall() == [(2,)]


class TestRegisterCollection:
    def test_rows_of_a_result(self, con):
        result = con._submit("SELECT i, i AS i FROM t ORDER BY i")
        handle = con._register_collection(result)
        assert result.fetchall() == []
        sql = f'SELECT * FROM "{handle.name}"'
        assert con._bind(sql)[0] == ["i", "i_1"]
        assert con._submit(sql).fetchall() == [(1, 1), (2, 2), (3, 3)]
        con.execute("DELETE FROM t")
        assert con._submit(sql).fetchall() == [(1, 1), (2, 2), (3, 3)]

    def test_lifetime_follows_the_handle(self, con):
        handle = con._register_collection(con._submit("SELECT 1 AS x"))
        name = handle.name
        del handle
        gc.collect()
        with pytest.raises(duckdb.InvalidInputException, match="was released"):
            con._submit(f'SELECT * FROM "{name}"')

    def test_arrow_result_refused(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="needs a result submitted with format 'rows'"):
            con._register_collection(con._submit("SELECT 1", None, "arrow"))

    def test_result_of_a_cursor(self, con):
        handle = con._register_collection(con.cursor()._submit("SELECT 1 AS x"))
        assert con._submit(f'SELECT * FROM "{handle.name}"').fetchall() == [(1,)]

    def test_result_of_another_database(self, con):
        result = duckdb.connect()._submit("SELECT 1")
        with pytest.raises(duckdb.InvalidInputException, match="another database"):
            con._register_collection(result)


class TestRegisterValues:
    @pytest.mark.timeout(60)
    def test_refused_when_a_call_takes_the_connection_while_it_converts(self, con, blocking_udf):
        started, release = blocking_udf
        converting = threading.Event()
        staged, ran = [], []

        class GatedDate(datetime.date):
            @property
            def year(self):
                converting.set()
                assert started.wait(30)
                return 2020

        def stage():
            try:
                staged.append(con._register_values([(GatedDate(2020, 1, 1),)]))
            except duckdb.ResourceInUseException as error:
                staged.append(error)

        staging = threading.Thread(target=stage)
        staging.start()
        assert converting.wait(30)
        running = threading.Thread(target=lambda: ran.append(con._submit("SELECT block(1)").fetchall()))
        running.start()
        staging.join(30)
        release.set()
        running.join(30)
        assert ran == [[(1,)]]
        assert len(staged) == 1
        assert isinstance(staged[0], duckdb.ResourceInUseException)
        assert "busy with another call" in str(staged[0])


class TestRegisterMap:
    def test_over_a_registered_input(self, con):
        source = con._register_source(pd.DataFrame({"a": [1, 2, 3]}))
        mapped = con._register_map(lambda df: df.assign(b=df["a"] * 2), None, f'SELECT * FROM "{source.name}"')
        sql = f'SELECT * FROM "{mapped.name}" ORDER BY a'
        assert con._bind(sql)[0] == ["a", "b"]
        assert con._submit(sql).fetchall() == [(1, 2), (2, 4), (3, 6)]

    def test_explicit_schema(self, con):
        def fn(df):
            return pd.DataFrame({"s": df["s"].fillna("?").str.upper()})

        mapped = con._register_map(fn, {"s": "VARCHAR"}, "SELECT s FROM t ORDER BY i")
        sql = f'SELECT * FROM "{mapped.name}"'
        assert con._bind(sql)[:2] == (["s"], con._bind("SELECT 'a'::VARCHAR")[1])
        assert con._submit(sql).fetchall() == [("A",), ("B",), ("?",)]

    def test_function_lifetime_follows_the_handle(self, con):
        def fn(df):
            return df

        fn_ref = weakref.ref(fn)
        mapped = con._register_map(fn, None, "SELECT i FROM t")
        del fn
        gc.collect()
        assert fn_ref() is not None
        assert len(con._submit(f'SELECT * FROM "{mapped.name}"').fetchall()) == 3
        del mapped
        gc.collect()
        assert fn_ref() is None

    def test_function_error(self, con):
        def fn(df):
            msg = "boom"
            raise ValueError(msg)

        mapped = con._register_map(fn, None, "SELECT i FROM t")
        with pytest.raises(duckdb.InvalidInputException, match="Python error"):
            con._bind(f'SELECT * FROM "{mapped.name}"')

    def test_input_must_be_one_select(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="exactly one SELECT statement"):
            con._register_map(lambda df: df, None, "DELETE FROM t")
        with pytest.raises(duckdb.ParserException):
            con._register_map(lambda df: df, None, "SELEC 1")


class TestRegisterFilelike:
    def test_file_lives_while_the_handle_does(self, con):
        handle = con._register_filelike(io.BytesIO(b"a,b\n1,2\n3,4\n"))
        path = handle.name
        assert path.startswith("DUCKDB_INTERNAL_OBJECTSTORE://")
        assert con._submit(f"SELECT * FROM read_csv('{path}') ORDER BY a").fetchall() == [(1, 2), (3, 4)]
        del handle
        gc.collect()
        with pytest.raises(duckdb.IOException):
            con._submit(f"SELECT * FROM read_csv('{path}')")

    def test_path_passes_through(self, con, tmp_path):
        path = str(tmp_path / "x.csv")
        assert con._register_filelike(path).name == path

    def test_list_refused(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="not a list"):
            con._register_filelike([io.BytesIO(b"a\n1\n")])


class TestOwnSources:
    def test_connection_keeps_handles_until_unregister(self, con):
        handle = con._register_source(pd.DataFrame({"i": [1, 2]}))
        handle_ref = weakref.ref(handle)
        con.execute(f'CREATE TEMP VIEW owned AS SELECT * FROM "{handle.name}"')
        con._own_sources("owned", [handle])
        del handle
        gc.collect()
        assert handle_ref() is not None
        assert con.execute("SELECT count(*) FROM owned").fetchall() == [(2,)]
        con.unregister("owned")
        gc.collect()
        assert handle_ref() is None
        # The view reads sources the unregister released, so it goes with them
        assert con.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = 'owned'").fetchall() == [(0,)]
        with pytest.raises(duckdb.CatalogException, match="owned"):
            con.execute("SELECT * FROM owned")

    def test_a_later_call_replaces_the_handles(self, con):
        first = con._register_source(pa.table({"i": [1]}))
        second = con._register_source(pa.table({"i": [2]}))
        first_ref, second_ref = weakref.ref(first), weakref.ref(second)
        con._own_sources("v", [first, second])
        con._own_sources("v", [second])
        del first, second
        gc.collect()
        assert first_ref() is None
        assert second_ref() is not None
        con._own_sources("v", [])
        gc.collect()
        assert second_ref() is None

    def test_names_fold_case(self, con):
        handle = con._register_source(pa.table({"i": [1]}))
        handle_ref = weakref.ref(handle)
        con._own_sources("View", [handle])
        del handle
        con.unregister("VIEW")
        gc.collect()
        assert handle_ref() is None

    def test_close_releases_them(self):
        connection = duckdb.connect()
        handle = connection._register_source(pa.table({"i": [1]}))
        handle_ref = weakref.ref(handle)
        connection._own_sources("v", [handle])
        del handle
        connection.close()
        gc.collect()
        assert handle_ref() is None

    def test_only_handles(self, con):
        with pytest.raises(TypeError):
            con._own_sources("v", ["not a handle"])

    def test_a_failed_drop_keeps_the_registration(self, con):
        con.register("v", pd.DataFrame({"i": [1]}))
        con.execute("BEGIN")
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM missing")
        with pytest.raises(duckdb.TransactionException):
            con.unregister("v")
        con.execute("ROLLBACK")
        # The name is still registered and its source still there, so the view reads and unregister works
        assert con.execute("SELECT * FROM v").fetchall() == [(1,)]
        con.unregister("v")
        assert con.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = 'v'").fetchall() == [(0,)]

    def test_unregister_of_an_owned_name_without_a_view(self, con):
        handle = con._register_source(pd.DataFrame({"i": [1]}))
        handle_ref = weakref.ref(handle)
        con._own_sources("no_view", [handle])
        del handle
        gc.collect()
        con.unregister("no_view")
        gc.collect()
        assert handle_ref() is None

    def test_a_view_over_a_released_source_names_the_object(self, con):
        handle = con._register_source(pd.DataFrame({"i": [1]}))
        key = handle.name
        con.execute(f'CREATE TEMP VIEW orphan AS SELECT * FROM "{key}"')
        del handle
        gc.collect()
        with pytest.raises(duckdb.InvalidInputException, match="was released") as raised:
            con.execute("SELECT * FROM orphan")
        # The generated key is ours, so no error shows it
        assert key not in str(raised.value)
        with pytest.raises(duckdb.InvalidInputException, match="was released"):
            con._bind("SELECT * FROM orphan")


class TestCreateView:
    """A view of Python sources holds them in its own catalog entry, so the transaction decides when they go."""

    def test_the_view_owns_its_source(self, con):
        frame = pd.DataFrame({"i": [7, 8]})
        frame_ref = weakref.ref(frame)
        handle = con._register_source(frame)
        con._create_view("w", f'SELECT i * 2 AS j FROM "{handle.name}" ORDER BY j', temporary=True)
        del frame, handle
        gc.collect()
        assert frame_ref() is not None
        assert con.execute("SELECT * FROM w").fetchall() == [(14,), (16,)]
        con.execute("DROP VIEW w")
        gc.collect()
        assert frame_ref() is None

    def test_a_rolled_back_drop_brings_the_source_back(self, con):
        handle = con._register_source(pd.DataFrame({"i": [1]}))
        con._create_view("w", f'SELECT * FROM "{handle.name}"', temporary=True)
        del handle
        con.execute("BEGIN")
        con.execute("DROP VIEW w")
        con.execute("ROLLBACK")
        gc.collect()
        assert con.execute("SELECT * FROM w").fetchall() == [(1,)]

    def test_sources_nested_in_a_relation_a_map_and_a_cte(self, con, relation_class):
        frame = con._register_source(pd.DataFrame({"a": [1, 2, 3]}))
        rel = relation_class(con, f'SELECT a FROM "{frame.name}" WHERE a > 1', [frame])
        rel_handle = con._register_source(rel)
        mapped = con._register_map(lambda df: df.assign(b=df["a"] * 10), None, f'SELECT * FROM "{rel_handle.name}"')
        smallest = f'(SELECT min(a) FROM "{frame.name}")'
        sql = f'WITH x AS (SELECT * FROM "{mapped.name}") SELECT a, b FROM x WHERE b > {smallest}'
        con._create_view("w", sql + " ORDER BY a", temporary=True)
        del frame, rel, rel_handle, mapped
        gc.collect()
        assert con.execute("SELECT * FROM w").fetchall() == [(2, 20), (3, 30)]

    def test_a_captured_relation_is_stored_as_its_text(self, relation_class, tmp_path):
        path = tmp_path / "f.db"
        first = duckdb.connect(path)
        first.execute("CREATE TABLE t AS SELECT 1 AS a UNION ALL SELECT 2")
        rel = relation_class(first, "SELECT a FROM t")  # noqa: F841, captured by name below
        [(_, handle)] = first._bind("SELECT * FROM rel", capture=True)[5]
        first._create_view("r", f'WITH "rel" AS (SELECT * FROM "{handle.name}") SELECT * FROM rel ORDER BY a')
        first.close()
        reopened = duckdb.connect(path)
        assert reopened.execute("SELECT * FROM r").fetchall() == [(1,), (2,)]
        reopened.close()

    def test_a_collection_is_refused_in_a_file(self, tmp_path):
        connection = duckdb.connect(tmp_path / "f.db")
        rows = connection._register_collection(connection._submit("SELECT 41 AS x UNION ALL SELECT 42"))
        with pytest.raises(duckdb.InvalidInputException) as caught:
            connection._create_view("c", f'SELECT * FROM "{rows.name}"')
        assert "create(" in str(caught.value)
        assert "register(" in str(caught.value)
        assert connection.execute("SELECT count(*) FROM duckdb_views() WHERE NOT internal").fetchall() == [(0,)]
        connection.close()

    def test_a_python_object_cannot_be_stored_in_a_file(self, tmp_path):
        connection = duckdb.connect(tmp_path / "f.db")
        handle = connection._register_source(pd.DataFrame({"a": [1]}))
        match = r"cannot be stored in a database file\. Use register\(\) for a view that reads it in this session"
        with pytest.raises(duckdb.InvalidInputException, match=match):
            connection._create_view("p", f'SELECT * FROM "{handle.name}"')
        connection._create_view("p", f'SELECT * FROM "{handle.name}"', temporary=True)
        assert connection.execute("SELECT * FROM p").fetchall() == [(1,)]
        connection.close()

    def test_in_memory_a_persistent_view_of_a_python_object_is_created(self, con):
        handle = con._register_source(pd.DataFrame({"a": [1]}))
        con._create_view("p", f'SELECT * FROM "{handle.name}"')
        del handle
        gc.collect()
        assert con.execute("SELECT * FROM memory.main.p").fetchall() == [(1,)]

    def test_replace_and_names(self, con):
        con._create_view(("main", "w"), "SELECT 1 AS i")
        with pytest.raises(duckdb.CatalogException, match="already exists"):
            con._create_view("w", "SELECT 2 AS i")
        con._create_view("w", "SELECT 2 AS i", replace=True)
        assert con.execute("SELECT * FROM main.w").fetchall() == [(2,)]

    def test_counts_as_a_catalog_change(self, con):
        version = con._catalog_version()
        con._create_view("w", "SELECT 1 AS i")
        assert con._catalog_version() == version + 1

    def test_a_released_source_is_reported(self, con):
        handle = con._register_source(pd.DataFrame({"a": [1]}), "frame")
        key = handle.name
        del handle
        gc.collect()
        with pytest.raises(duckdb.InvalidInputException, match='registered as "frame", which was released'):
            con._create_view("w", f'SELECT * FROM "{key}"', temporary=True)

    def test_only_a_select(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="exactly one SELECT statement"):
            con._create_view("w", "DELETE FROM t")

    def test_a_cte_shadows_a_registry_name(self, con):
        handle = con._register_source(pd.DataFrame({"x": [1]}))
        key = f'"{handle.name}"'
        con._create_view("w", f"WITH {key} AS (SELECT 99 AS x) SELECT * FROM {key}", temporary=True)
        assert con.execute("SELECT * FROM w").fetchall() == [(99,)]
        # In scope inside a subquery of its node, and out of scope beside it
        nested = f"WITH {key} AS (SELECT 99 AS x) SELECT * FROM (SELECT * FROM {key})"
        beside = f"SELECT * FROM ({nested}) UNION ALL SELECT * FROM {key} ORDER BY x"
        con._create_view("u", beside, temporary=True)
        del handle
        gc.collect()
        assert con.execute("SELECT * FROM u").fetchall() == [(1,), (99,)]

    def test_a_pivot_source_is_written_in(self, con):
        facts = con._register_source(pd.DataFrame({"year": [2020, 2021], "v": [7, 8]}))
        con._create_view("w", f'SELECT * FROM (PIVOT "{facts.name}" ON year IN (2020) USING sum(v))', temporary=True)
        del facts
        gc.collect()
        assert con.execute("SELECT * FROM w").fetchall() == [(7,)]

    def test_a_pivot_with_values_from_the_data_is_refused_as_sql_refuses_it(self, con):
        # Such a PIVOT parses into several statements, so its value subquery never reaches a view
        facts = con._register_source(pd.DataFrame({"year": [2020], "v": [7]}))
        years = con._register_source(pd.DataFrame({"year": [2020]}))
        sql = f'PIVOT "{facts.name}" ON year IN (SELECT year FROM "{years.name}") USING sum(v)'
        with pytest.raises(duckdb.InvalidInputException, match="found a MULTI statement"):
            con._create_view("w", f"SELECT * FROM ({sql})", temporary=True)
        con.execute("CREATE TABLE f AS SELECT 2020 AS year, 7 AS v")
        with pytest.raises(duckdb.ParserException, match="cannot be used in views"):
            con.execute("CREATE VIEW w AS PIVOT f ON year IN (SELECT year FROM f) USING sum(v)")


class TestRegister:
    def test_view_reads_the_registry(self, con):
        frame = pd.DataFrame({"i": [1, 2]})
        con.register("frame_view", frame)
        assert (
            con.execute("SELECT sql FROM duckdb_views() WHERE view_name = 'frame_view'")
            .fetchall()[0][0]
            .count("__duckdb_source_")
        )
        assert con.sql("SELECT * FROM frame_view").fetchall() == [(1,), (2,)]

    def test_connection_owns_the_object_until_unregister(self, con):
        frame = pd.DataFrame({"i": [1, 2]})
        frame_ref = weakref.ref(frame)
        con.register("frame_view", frame)
        del frame
        gc.collect()
        assert frame_ref() is not None
        assert con.execute("SELECT count(*) FROM frame_view").fetchall() == [(2,)]
        con.unregister("frame_view")
        gc.collect()
        assert frame_ref() is None
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM frame_view")

    def test_a_rolled_back_unregister_keeps_the_object(self, con):
        con.register("v", pd.DataFrame({"i": [1]}))
        con.execute("BEGIN")
        con.unregister("v")
        con.execute("ROLLBACK")
        gc.collect()
        assert con.execute("SELECT * FROM v").fetchall() == [(1,)]
        con.unregister("v")
        assert con.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = 'v'").fetchall() == [(0,)]

    def test_a_rolled_back_replacement_keeps_the_first_object(self, con):
        con.register("v", pd.DataFrame({"i": [1]}))
        con.execute("BEGIN")
        con.register("v", pd.DataFrame({"i": [2]}))
        assert con.execute("SELECT * FROM v").fetchall() == [(2,)]
        con.execute("ROLLBACK")
        gc.collect()
        assert con.execute("SELECT * FROM v").fetchall() == [(1,)]

    def test_register_and_a_view_of_one_name_keep_their_own_objects(self, con):
        con.execute("CREATE TABLE u AS SELECT 10 AS i")
        con.register("v", pd.DataFrame({"i": [1]}))
        con._create_view("v", "SELECT * FROM u")
        gc.collect()
        assert con.execute("SELECT * FROM temp.main.v").fetchall() == [(1,)]
        assert con.execute("SELECT * FROM memory.main.v").fetchall() == [(10,)]
        con.unregister("v")
        assert con.execute("SELECT * FROM v").fetchall() == [(10,)]

    def test_unregister_after_the_user_dropped_the_view(self, con):
        frame = pd.DataFrame({"i": [1]})
        frame_ref = weakref.ref(frame)
        con.register("v", frame)
        del frame
        con.execute("DROP VIEW v")
        con.unregister("v")
        con.unregister("v")
        gc.collect()
        assert frame_ref() is None

    def test_unregister_never_drops_a_persistent_view_of_the_same_name(self, con):
        con.execute("CREATE VIEW v AS SELECT 99 AS i")
        con.register("v", pd.DataFrame({"i": [5]}))
        assert con.execute("SELECT * FROM v").fetchall() == [(5,)]
        con.unregister("v")
        assert con.execute("SELECT * FROM v").fetchall() == [(99,)]
        con.unregister("v")
        assert con.execute("SELECT * FROM v").fetchall() == [(99,)]

    def test_the_names_remembered_are_bounded_and_a_view_is_never_left_behind(self, con):
        frame = pd.DataFrame({"i": [1]})
        con.register("first", frame)
        # One past the 4096 names the connection remembers, so the first one has left the hint
        for number in range(4096):
            con.register(f"other_{number}", frame)
        con.unregister("first")
        assert con.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = 'first'").fetchall() == [(0,)]
        con.unregister("never_registered")

    def test_a_view_renamed_in_sql_is_unregistered_by_its_new_name(self, con):
        frame = pd.DataFrame({"x": [1]})
        frame_ref = weakref.ref(frame)
        con.register("original", frame)
        del frame
        con.execute("ALTER VIEW original RENAME TO renamed")
        con.unregister("original")
        assert con.execute("SELECT * FROM renamed").fetchall() == [(1,)]
        con.unregister("renamed")
        assert con.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = 'renamed'").fetchall() == [(0,)]
        gc.collect()
        assert frame_ref() is None

    def test_a_name_the_hint_lacks_is_a_no_op_beside_an_open_result(self, con):
        con.register("original", pd.DataFrame({"x": [1]}))
        con.execute("ALTER VIEW original RENAME TO renamed")
        con.execute("SELECT * FROM range(100000)")
        assert con.fetchone() == (0,)
        con.unregister("renamed")
        assert con.fetchone() == (1,)
        con.abort()
        assert con.execute("SELECT * FROM renamed").fetchall() == [(1,)]

    def test_the_registered_view_holds_its_object_not_a_key(self, con):
        frame = pd.DataFrame({"i": [1]})
        con.register("v", frame)
        view = con.execute("SELECT sql FROM duckdb_views() WHERE view_name = 'v'").fetchall()[0][0]
        assert "pandas_scan" in view

    def test_reregister_replaces_and_frees_the_old_object(self, con):
        first = pd.DataFrame({"i": [1]})
        first_ref = weakref.ref(first)
        con.register("v", first)
        del first
        con.register("v", pd.DataFrame({"i": [2]}))
        gc.collect()
        assert first_ref() is None
        assert con.execute("SELECT * FROM v").fetchall() == [(2,)]

    def test_reregister_after_a_failed_register_keeps_the_old_object(self, con):
        con.register("v", pd.DataFrame({"i": [1]}))
        with pytest.raises(duckdb.InvalidInputException, match="not suitable for replacement scans"):
            con.register("v", object())
        assert con.execute("SELECT * FROM v").fetchall() == [(1,)]

    def test_name_of_a_view_it_did_not_create(self, con):
        con.execute("CREATE TEMP VIEW taken AS SELECT 1")
        with pytest.raises(duckdb.CatalogException, match="already exists"):
            con.register("taken", pd.DataFrame({"i": [1]}))

    def test_close_frees_registered_objects(self):
        connection = duckdb.connect()
        frame = pd.DataFrame({"i": [1]})
        frame_ref = weakref.ref(frame)
        connection.register("v", frame)
        del frame
        connection.close()
        gc.collect()
        assert frame_ref() is None


def execute_type(value):
    return str(duckdb.connect().execute("SELECT ? AS p", [value]).description[0][1])


class TestLiteralType:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1, "INTEGER"),
            (-(2**31), "INTEGER"),
            (2**31, "BIGINT"),
            (-(2**63), "BIGINT"),
            (2**63, "UBIGINT"),
            (2**64, "HUGEINT"),
            (True, "BOOLEAN"),
            (1.5, "DOUBLE"),
            ("s", "VARCHAR"),
            (b"b", "BLOB"),
            (decimal.Decimal("1.25"), "DECIMAL(3,2)"),
            (datetime.date(2020, 1, 2), "DATE"),
            (datetime.datetime(2020, 1, 2, 3, 4, 5), "TIMESTAMP"),
            (datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc), "TIMESTAMP WITH TIME ZONE"),
            (datetime.time(1, 2, 3), "TIME"),
            (datetime.timedelta(days=1, seconds=3), "INTERVAL"),
            ([1, 2], "INTEGER[]"),
            ([1, "a"], "VARCHAR[]"),
            ({"a": 1, "b": "x"}, "STRUCT(a INTEGER, b VARCHAR)"),
            ({"key": [1], "value": ["a"]}, "MAP(INTEGER, VARCHAR)"),
            (None, '"NULL"'),
            (np.float64(1.5), "DOUBLE"),
            (np.str_("x"), "VARCHAR"),
            (np.datetime64("2020-01-02"), "DATE"),
        ],
    )
    def test_types_as_execute_binds_them(self, value, expected):
        assert duckdb._duckdb._literal_type(value) == expected
        assert execute_type(value) == expected

    @pytest.mark.parametrize("value", [object(), np.int64(3), np.int8(3), np.bool_(True)])
    def test_values_execute_cannot_bind(self, value):
        with pytest.raises(duckdb.NotImplementedException, match="Unable to transform python value"):
            duckdb._duckdb._literal_type(value)
        with pytest.raises(duckdb.NotImplementedException, match="Unable to transform python value"):
            execute_type(value)


class TestTemporalLiteral:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (datetime.date(2020, 1, 2), "'2020-01-02'::DATE"),
            (datetime.datetime(2020, 1, 2, 3, 4, 5, 6), "'2020-01-02 03:04:05.000006'::TIMESTAMP"),
            (
                datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
                "'2020-01-02 01:04:05+00'::TIMESTAMP WITH TIME ZONE",
            ),
            (datetime.time(1, 2, 3), "'01:02:03'::TIME"),
            (datetime.timedelta(days=1, microseconds=5), "'1 day 00:00:00.000005'::INTERVAL"),
            (np.datetime64("2020-01-02"), "'2020-01-02'::DATE"),
        ],
    )
    def test_literal_round_trips(self, value, expected):
        literal = duckdb._duckdb._temporal_literal(value)
        assert literal == expected
        assert duckdb.connect().execute(f"SELECT {literal} = ?", [value]).fetchall() == [(True,)]

    @pytest.mark.parametrize("value", [5, "2020-01-02", 1.5, None])
    def test_not_temporal(self, value):
        with pytest.raises(duckdb.InvalidInputException, match="Expected a date, time, datetime, timedelta"):
            duckdb._duckdb._temporal_literal(value)


class TestValueLiteral:
    @pytest.mark.parametrize(
        ("make", "expected"),
        [
            (lambda: duckdb.Value(5, duckdb.sqltypes.TINYINT), "CAST(5 AS TINYINT)"),
            (lambda: duckdb.Value("2020-01-02", duckdb.sqltypes.DATE), "'2020-01-02'::DATE"),
            (lambda: duckdb.Value(1, duckdb.sqltypes.DOUBLE), "CAST(1.0 AS DOUBLE)"),
            (lambda: {"a": 1, "b": "x"}, "CAST({'a': 1, 'b': 'x'} AS STRUCT(a INTEGER, b VARCHAR))"),
            (lambda: [[1, 2], [3]], "CAST([[1, 2], [3]] AS INTEGER[][])"),
            (lambda: [None, 1], "CAST([NULL::INTEGER, 1] AS INTEGER[])"),
            (lambda: None, "NULL"),
            (lambda: "it's", "CAST('it''s' AS VARCHAR)"),
            (lambda: 1.5, "CAST(1.5 AS DOUBLE)"),
            (lambda: 2**63, "CAST(9223372036854775808 AS UBIGINT)"),
            (lambda: decimal.Decimal("1.25"), "CAST(1.25 AS DECIMAL(3,2))"),
            (lambda: {"key": [1], "value": ["a"]}, "MAP {1: 'a'}::MAP(INTEGER, VARCHAR)"),
            (lambda: datetime.date(2020, 1, 2), "'2020-01-02'::DATE"),
        ],
        ids=[
            "tinyint_value",
            "str_value_as_date",
            "int_value_as_double",
            "dict",
            "nested_list",
            "list_with_null",
            "none",
            "str",
            "float",
            "ubigint",
            "decimal",
            "map",
            "date",
        ],
    )
    def test_literal_binds_as_the_parameter(self, make, expected):
        value = make()
        literal = duckdb._duckdb._value_literal(value)
        assert literal == expected
        connection = duckdb.connect()
        parameter_type = str(connection.execute("SELECT ? AS p", [value]).description[0][1])
        assert str(connection.execute(f"SELECT {literal} AS p").description[0][1]) == parameter_type
        assert connection.execute(f"SELECT {literal} IS NOT DISTINCT FROM ?", [value]).fetchall() == [(True,)]

    def test_temporal_literal_is_a_special_case(self):
        value = datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
        assert duckdb._duckdb._value_literal(value) == duckdb._duckdb._temporal_literal(value)

    def test_value_no_literal_can_hold(self):
        with pytest.raises(duckdb.NotImplementedException, match="Unable to transform python value"):
            duckdb._duckdb._value_literal(object())


class TestParseExpressions:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("a, b + 1 AS c", ["a", "(b + 1) AS c"]),
            ('"my col"', ['"my col"']),
            ("a AS select", ['a AS "select"']),
            ("sum(x) FILTER (WHERE y > 1) AS total", ["sum(x) FILTER (WHERE (y > 1)) AS total"]),
            ("count(*) OVER (PARTITION BY a ORDER BY b)", ["count() OVER (PARTITION BY a ORDER BY b)"]),
        ],
    )
    def test_rendered_back(self, text, expected):
        rendered = duckdb._duckdb._parse_expressions(text)
        assert rendered == expected
        assert duckdb._duckdb._parse_expressions(", ".join(rendered)) == rendered

    @pytest.mark.parametrize("text", ["a,, b", "1 +", ""])
    def test_syntax_error(self, text):
        with pytest.raises(duckdb.ParserException):
            duckdb._duckdb._parse_expressions(text)

    @pytest.mark.parametrize(
        ("text", "held"),
        [
            ("1 FROM t", "a FROM clause"),
            ("a, b FROM t", "a FROM clause"),
            ("1 FROM t WHERE i > 1", "a FROM clause"),
            ("1 LIMIT 2", "a modifier"),
            ("1 ORDER BY 1", "a modifier"),
            ("1 GROUP BY a", "a GROUP BY clause"),
            ("1 GROUP BY ALL", "a GROUP BY clause"),
            ("1 HAVING count(*) > 1", "a HAVING clause"),
            ("1 QUALIFY row_number() OVER () = 1", "a QUALIFY clause"),
            ("1 USING SAMPLE 5", "a SAMPLE clause"),
            ("1 UNION SELECT 2", "a set operation"),
            ("1; SELECT 2", "more than one statement"),
        ],
    )
    def test_text_the_expression_list_does_not_consume(self, text, held):
        with pytest.raises(duckdb.ParserException, match=f"is not an expression list: it holds {re.escape(held)}"):
            duckdb._duckdb._parse_expressions(text)

    @pytest.mark.parametrize(
        "text",
        [
            "(SELECT max(i) FROM t)",
            "exists (SELECT 1 FROM t) AS e",
            "* EXCLUDE (a)",
            "count(*) OVER (PARTITION BY a ORDER BY b)",
            "sum(i) FILTER (WHERE i > 1)",
        ],
    )
    def test_a_clause_inside_an_expression_is_consumed(self, text):
        assert len(duckdb._duckdb._parse_expressions(text)) == 1

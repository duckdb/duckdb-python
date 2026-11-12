import datetime
import importlib.util
import os
import threading
import warnings
from decimal import Decimal
from pathlib import Path

import pytest

import duckdb

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


def _extensions():
    """None when the variable is not set, else each extension's file, or None for one that is not found."""
    root = os.environ.get("DUCKDB_PYTHON_TEST_EXTENSION_PATH")
    if not root:
        return None
    found = {}
    # httpfs first: the quack client refuses to start without it
    for name in ("httpfs", "quack"):
        file = f"{name}.duckdb_extension"
        # The two layouts the require fixture in tests/conftest.py accepts for the same variable
        candidates = [Path(root) / file, *sorted(Path(root).glob(f"*/{file}"))]
        found[name] = next((path for path in candidates if path.is_file()), None)
    return found


EXTENSIONS = _extensions()

REFUSAL = (
    "Invalid Input Error: The relational API, register(), append() and unregister() are not available while "
    "CONNECT-ed, since the engine sends every statement but DISCONNECT to the connected database; use "
    "con.execute(query) with fetchall(), fetchdf() or fetchnumpy() for queries, or DISCONNECT first"
)
APPENDER_REFUSAL = (
    "Invalid Input Error: An appender cannot be created or flushed while CONNECT-ed, since a flush is a statement over "
    "rows held on this connection and the engine sends every statement but DISCONNECT to the connected database; "
    "DISCONNECT first"
)
ARROW_REFUSAL = (
    "Invalid Input Error: Arrow output on a DB-API result is not available while CONNECT-ed; use fetchall(), "
    "fetchdf() or fetchnumpy(), or DISCONNECT first"
)
PARAMETERS_REFUSAL = (
    "Invalid Input Error: Parameterized prepared statements cannot be executed while CONNECT-ed; DISCONNECT first, "
    "or run the SQL as a fresh statement to route through the CONNECT binding"
)
MIXED_TYPES_QUERY = """
SELECT 42 AS n,
       1.23::DECIMAL(10,2) AS amount,
       DATE '2026-10-23' AS day,
       NULL::INTEGER AS empty
"""


def assert_refused(call, message=REFUSAL):
    with pytest.raises(duckdb.InvalidInputException) as refused:
        call()
    assert str(refused.value) == message


def error_of(call):
    """What the call raised, or None, for a test that checks the call's effects before its error."""
    try:
        call()
    except duckdb.Error as error:
        return str(error)
    return None


def _open():
    connection = duckdb.connect(config={"allow_unsigned_extensions": True})
    for path in EXTENSIONS.values():
        connection.execute(f"LOAD '{path}'")
    return connection


@pytest.fixture
def server():
    # CONNECT needs a database that supports it, and only an extension provides one: quack, with the httpfs it
    # requires, built for this engine, since no published build loads into a development build.
    if EXTENSIONS is None:
        pytest.skip("needs httpfs and quack built for this engine in DUCKDB_PYTHON_TEST_EXTENSION_PATH")
    missing = [f"{name}.duckdb_extension" for name, path in EXTENSIONS.items() if path is None]
    if missing:
        pytest.fail(
            f"DUCKDB_PYTHON_TEST_EXTENSION_PATH is set, but neither it nor a folder in it holds {' or '.join(missing)}"
        )
    connection = _open()
    uri, _, token = connection.execute("CALL quack_serve('quack:127.0.0.1:0', token = 'connect-relations')").fetchone()
    connection.execute("CREATE TABLE t AS SELECT 'remote' AS side, 1 AS i")
    yield connection, uri, token
    connection.execute("CALL quack_stop(?)", [uri])
    connection.close()


@pytest.fixture
def local(server):
    """A connection with the server attached as remote and a local table of the same name, not yet CONNECT-ed."""
    _, uri, token = server
    connection = _open()
    connection.execute(f"ATTACH '{uri}' AS remote (TOKEN '{token}')")
    connection.execute("CREATE TABLE t AS SELECT 'local' AS side, 1 AS i")
    yield connection
    connection.close()


@pytest.fixture
def con(local):
    local.execute("CONNECT remote")
    return local


def remote_rows(server):
    return server[0].execute("SELECT side, i FROM t ORDER BY i, side").fetchall()


def remote_tables(server):
    return server[0].execute("SELECT table_name FROM duckdb_tables() ORDER BY table_name").fetchall()


def side(connection):
    return connection.execute("SELECT side FROM t").fetchall()


class TestSql:
    @pytest.mark.parametrize("query", [MIXED_TYPES_QUERY, "SELECT 42"], ids=["mixed_types", "constant"])
    def test_refuses_and_names_what_works(self, con, query):
        assert_refused(lambda: con.sql(query))

    @pytest.mark.parametrize(
        "call",
        [
            lambda con, query: con.query(query),
            lambda con, query: con.from_query(query),
            lambda con, query: duckdb.sql(query, connection=con),
        ],
        ids=["query", "from_query", "module_sql"],
    )
    def test_every_spelling_refuses(self, con, call):
        assert_refused(lambda: call(con, "SELECT 42"))

    @pytest.mark.parametrize(
        "query",
        [
            "INSERT INTO t VALUES ('sql', 2)",
            "CREATE TABLE made_by_sql (k INTEGER)",
            "INSERT INTO t VALUES ('script', 3); SELECT 42",
        ],
        ids=["insert", "create", "script"],
    )
    def test_refuses_before_the_connected_database_runs_anything(self, con, server, query):
        error = error_of(lambda: con.sql(query))
        assert remote_rows(server) == [("remote", 1)]
        assert remote_tables(server) == [("t",)]
        assert error == REFUSAL

    def test_a_statement_parsed_before_connect_refuses(self, local):
        statement = local.extract_statements("SELECT side FROM t")[0]
        local.execute("CONNECT remote")
        assert_refused(lambda: local.sql(statement))

    def test_disconnect_runs_locally(self, con):
        con.sql("DISCONNECT")
        assert side(con) == [("local",)]

    def test_a_script_that_disconnects_first_reads_locally(self, con):
        assert con.sql("DISCONNECT; SELECT side FROM t").fetchall() == [("local",)]

    def test_a_script_refused_at_its_first_statement_does_not_disconnect(self, con):
        error = error_of(lambda: con.sql("SELECT 1; DISCONNECT"))
        assert side(con) == [("remote",)]
        assert error == REFUSAL

    @pytest.mark.parametrize("query", ["CONNECT remote", "SELECT 1; CONNECT remote"], ids=["connect", "script"])
    def test_connect_returns_no_relation(self, local, query):
        assert local.sql(query) is None
        assert side(local) == [("remote",)]

    def test_a_local_write_before_a_connect_survives_the_refusal_after_it(self, local):
        assert_refused(lambda: local.sql("INSERT INTO t VALUES ('kept', 2); CONNECT remote; SELECT 42"))
        assert side(local) == [("remote",)]
        local.execute("DISCONNECT")
        assert local.execute("SELECT side, i FROM t ORDER BY i").fetchall() == [("local", 1), ("kept", 2)]

    def test_a_refusal_leaves_the_local_transaction_usable(self, local):
        local.execute("BEGIN")
        local.execute("CREATE TABLE made_in_transaction (k INTEGER)")
        local.execute("CONNECT remote")
        assert_refused(lambda: local.sql("SELECT 1; SELECT 2"))
        assert_refused(lambda: local.sql("SELECT 42"))
        local.execute("DISCONNECT")
        local.execute("COMMIT")
        assert local.execute("SELECT count(*) FROM made_in_transaction").fetchone() == (0,)

    def test_the_refusal_stays_plain_text_under_errors_as_json(self, local):
        local.execute("SET errors_as_json = true")
        local.execute("CONNECT remote")
        assert_refused(lambda: local.sql("SELECT 42"))
        assert_refused(lambda: local.table("t"))

    def test_a_parameter_that_connects_while_it_is_converted_refuses(self, local):
        class ConnectsWhenRead:
            def __str__(self) -> str:
                local.execute("CONNECT remote")
                return "key"

        assert_refused(lambda: local.sql("SELECT ? AS s", params=[{ConnectsWhenRead(): 7}]))
        assert side(local) == [("remote",)]


def _relation_over(kind, connection):
    if kind == "table":
        return connection.table("t")
    if kind == "sql":
        return connection.sql("SELECT side, i + 1 AS j FROM t")
    if kind == "dataframe":
        return connection.from_df(pd.DataFrame({"a": [1, 2]}))
    if kind == "materialized":
        return connection.sql("SELECT 5 AS m").materialize()
    return connection.values([1, 2])


class TestRelationBuiltBeforeConnect:
    @pytest.mark.parametrize("kind", ["table", "sql", "dataframe", "materialized", "values"])
    def test_reading_refuses(self, local, kind):
        relation = _relation_over(kind, local)
        local.execute("CONNECT remote")
        assert_refused(relation.fetchall)

    @pytest.mark.parametrize(
        "call",
        [
            lambda relation: relation.df(),
            lambda relation: relation.to_arrow_table(),
            lambda relation: relation.filter("i > 0"),
            lambda relation: relation.columns,
            lambda relation: str(relation),
            lambda relation: relation.select_types(["VARCHAR"]),
            lambda relation: relation.map(lambda frame: frame),
            pytest.param(
                lambda relation: relation.pl(lazy=True),
                marks=pytest.mark.skipif(importlib.util.find_spec("polars") is None, reason="needs polars"),
            ),
        ],
        ids=["df", "to_arrow_table", "filter", "columns", "show", "select_types", "map", "lazy_polars"],
    )
    def test_every_method_that_reaches_the_engine_refuses(self, local, call):
        relation = local.table("t")
        local.execute("CONNECT remote")
        assert_refused(lambda: call(relation))

    @pytest.mark.parametrize(
        "call",
        [
            lambda relation: relation.create("made_by_create"),
            lambda relation: relation.to_table("made_by_to_table"),
            lambda relation: relation.insert_into("t"),
            lambda relation: relation.insert(("inserted", 2)),
            lambda relation: relation.update({"side": duckdb.ConstantExpression("updated")}),
        ],
        ids=["create", "to_table", "insert_into", "insert", "update"],
    )
    def test_writing_refuses_and_leaves_the_connected_database_alone(self, local, server, call):
        relation = local.table("t")
        local.execute("CONNECT remote")
        error = error_of(lambda: call(relation))
        assert remote_rows(server) == [("remote", 1)]
        assert remote_tables(server) == [("t",)]
        assert error == REFUSAL

    def test_reads_the_local_table_again_after_disconnect(self, local):
        relation = local.table("t")
        local.execute("CONNECT remote")
        assert_refused(relation.fetchall)
        local.execute("DISCONNECT")
        assert relation.fetchall() == [("local", 1)]

    def test_query_refuses_a_connect_before_it_makes_its_view(self, local, server):
        server[0].execute("CREATE VIEW named AS SELECT 'remote view' AS v")
        relation = local.table("t")
        error = error_of(lambda: relation.query("named", "CONNECT remote"))
        assert server[0].execute("SELECT v FROM named").fetchall() == [("remote view",)]
        assert side(local) == [("local",)]
        assert local.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = 'named'").fetchone() == (0,)
        assert error == (
            "Invalid Input Error: DuckDBPyRelation.query() cannot run a CONNECT, since it could not drop the name it "
            "gives the relation for the call once CONNECT-ed; run the CONNECT through con.sql() or con.execute()"
        )


class TestRelationConstructors:
    @pytest.mark.parametrize(
        "call",
        [
            lambda con, tmp_path: con.table("t"),
            lambda con, tmp_path: con.table("main.t"),
            lambda con, tmp_path: con.values([1, 2]),
            lambda con, tmp_path: con.from_df(pd.DataFrame({"a": [1]})),
            lambda con, tmp_path: con.from_arrow(pa.table({"a": [1]})),
            lambda con, tmp_path: con.table_function("range", [3]),
            lambda con, tmp_path: con.read_csv(str(_csv(tmp_path))),
            lambda con, tmp_path: con.register("registered", pd.DataFrame({"a": [1]})),
            lambda con, tmp_path: con.append("t", pd.DataFrame({"side": ["appended"], "i": [2]})),
        ],
        ids=[
            "table",
            "qualified_table",
            "values",
            "from_df",
            "from_arrow",
            "table_function",
            "read_csv",
            "register",
            "append",
        ],
    )
    def test_refuses(self, con, server, tmp_path, call):
        error = error_of(lambda: call(con, tmp_path))
        assert remote_rows(server) == [("remote", 1)]
        assert error == REFUSAL

    @pytest.mark.parametrize(
        "call",
        [
            lambda con, value: con.values([value]),
            lambda con, value: con._register_values([(value,)]),
        ],
        ids=["values", "register_values"],
    )
    def test_a_value_that_connects_while_it_is_converted_refuses(self, local, call):
        connected = []

        class ConnectingZone(datetime.tzinfo):
            def utcoffset(self, dt: datetime.datetime | None) -> datetime.timedelta:
                if not connected:
                    connected.append(True)
                    local.execute("CONNECT remote")
                return datetime.timedelta(hours=1)

            def dst(self, dt: datetime.datetime | None) -> datetime.timedelta:
                return datetime.timedelta(0)

        assert_refused(lambda: call(local, datetime.datetime(2026, 1, 1, tzinfo=ConnectingZone())))
        assert side(local) == [("remote",)]


def _csv(directory):
    path = directory / "rows.csv"
    path.write_text("a\n1\n")
    return path


class TestUnregister:
    def test_refuses_and_keeps_the_view(self, local):
        local.register("registered", pd.DataFrame({"a": [1, 2]}))
        local.execute("CONNECT remote")
        assert_refused(lambda: local.unregister("registered"))
        local.execute("DISCONNECT")
        assert local.execute("SELECT a FROM registered ORDER BY a").fetchall() == [(1,), (2,)]
        local.unregister("registered")
        with pytest.raises(duckdb.CatalogException):
            local.execute("SELECT a FROM registered")

    def test_refuses_a_name_never_registered(self, con):
        assert_refused(lambda: con.unregister("never_registered"))


class TestAppender:
    @pytest.mark.parametrize(
        "create",
        [
            lambda con: con.appender("t"),
            # Type names, which parse with the connected database's grammar once CONNECT-ed
            lambda con: con.query_appender("INSERT INTO t FROM appender_data", {"side": "VARCHAR", "i": "INTEGER"}),
        ],
        ids=["table", "query"],
    )
    def test_creating_refuses_and_an_appender_created_after_disconnect_appends_locally(self, local, server, create):
        local.execute("CONNECT remote")
        assert_refused(lambda: create(local), APPENDER_REFUSAL)
        local.execute("DISCONNECT")
        appender = create(local)
        appender.append_row("appended", 2)
        appender.flush()
        assert local.execute("SELECT side, i FROM t ORDER BY i").fetchall() == [("local", 1), ("appended", 2)]
        assert remote_rows(server) == [("remote", 1)]

    @pytest.mark.parametrize(
        "create",
        [
            lambda con, name: con.appender("t", columns=[name]),
            lambda con, name: con.query_appender("INSERT INTO t (side) FROM appender_data", {name: str}),
        ],
        ids=["table", "query"],
    )
    def test_a_column_name_that_connects_while_it_is_converted_refuses(self, local, server, create):
        class ConnectingName:
            def __str__(self) -> str:
                local.execute("CONNECT remote")
                return "side"

        assert_refused(lambda: create(local, ConnectingName()), APPENDER_REFUSAL)
        assert side(local) == [("remote",)]
        assert remote_rows(server) == [("remote", 1)]

    def test_a_flush_refuses_and_keeps_its_rows_for_a_flush_after_disconnect(self, local, server):
        appender = local.appender("t")
        appender.append_row("buffered", 2)
        local.execute("CONNECT remote")
        error = error_of(appender.flush)
        assert remote_rows(server) == [("remote", 1)]
        assert error == APPENDER_REFUSAL
        local.execute("DISCONNECT")
        appender.flush()
        assert local.execute("SELECT side, i FROM t ORDER BY i").fetchall() == [("local", 1), ("buffered", 2)]


class TestDbApi:
    def test_fetches_read_the_connected_database(self, con):
        assert con.execute(MIXED_TYPES_QUERY).fetchall() == [(42, Decimal("1.23"), datetime.date(2026, 10, 23), None)]
        assert con.execute("SELECT side FROM t").fetchdf()["side"].tolist() == ["remote"]
        assert con.execute("SELECT i FROM t").fetchnumpy()["i"].tolist() == [1]

    def test_a_statement_parsed_before_connect_runs_on_the_connected_database(self, local, server):
        read = local.extract_statements("SELECT side FROM t")[0]
        write = local.extract_statements("INSERT INTO t VALUES ('parsed', 3)")[0]
        local.execute("CONNECT remote")
        assert local.execute(read).fetchall() == [("remote",)]
        local.execute(write)
        assert remote_rows(server) == [("remote", 1), ("parsed", 3)]

    @pytest.mark.parametrize(
        "call",
        [
            lambda con: con.execute("SELECT ?::INTEGER", [1]),
            lambda con: con.executemany("INSERT INTO t VALUES (?, ?)", [("many", 2)]),
        ],
        ids=["execute", "executemany"],
    )
    def test_the_engine_refuses_parameters(self, con, server, call):
        assert_refused(lambda: call(con), PARAMETERS_REFUSAL)
        assert remote_rows(server) == [("remote", 1)]

    @pytest.mark.parametrize("method", ["to_arrow_table", "to_arrow_reader"])
    def test_arrow_refusal_names_only_what_works(self, con, method):
        con.execute(MIXED_TYPES_QUERY)
        assert_refused(getattr(con, method), ARROW_REFUSAL)

    @pytest.mark.parametrize("method", ["to_arrow_table", "to_arrow_reader"])
    def test_arrow_refusal_warns_nothing_and_keeps_the_rows(self, con, method):
        con.execute("SELECT 42 AS n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert_refused(getattr(con, method), ARROW_REFUSAL)
        assert caught == []
        assert con.fetchall() == [(42,)]


@pytest.mark.parametrize("method", ["to_arrow_table", "to_arrow_reader"])
def test_arrow_output_on_a_closed_connection_warns_then_refuses(method):
    connection = duckdb.connect()
    connection.execute("SELECT 1")
    connection.close()
    with (
        pytest.warns(UserWarning, match="extra copy"),
        pytest.raises(duckdb.ConnectionException, match="Connection already closed"),
    ):
        getattr(connection, method)()


@pytest.mark.parametrize("method", ["to_arrow_table", "to_arrow_reader"])
def test_arrow_output_on_a_busy_connection_warns_then_refuses(method):
    connection = duckdb.connect()
    started, release = threading.Event(), threading.Event()

    def hold() -> int:
        started.set()
        release.wait(30)
        return 1

    connection.create_function("hold", hold, [], "BIGINT", side_effects=True)
    holder = threading.Thread(target=lambda: connection.execute("SELECT hold()").fetchall())
    holder.start()
    try:
        assert started.wait(30)
        with (
            pytest.warns(UserWarning, match="extra copy"),
            pytest.raises(duckdb.ResourceInUseException, match="busy"),
        ):
            getattr(connection, method)()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UserWarning, match="extra copy"):
                getattr(connection, method)()
    finally:
        release.set()
        holder.join()
        connection.close()


def test_a_cursor_of_a_connected_connection_reads_locally(con):
    assert con.cursor().sql("SELECT side FROM t").fetchall() == [("local",)]

import gc
import re
import subprocess
import sys
import textwrap
import threading
import warnings

import pytest

import duckdb
from duckdb.sqltypes import BIGINT, INTEGER, SQLNULL, VARCHAR

WRONG_ARITY = duckdb.InvalidInputException
CONCURRENT_USE = duckdb.ResourceInUseException
CLOSED = duckdb.ConnectionException

UNFINISHED_EXECUTE = "unfinished result from execute"
BUSY = "is busy with another call"
KV = {"k": "INTEGER", "v": "VARCHAR"}
VECTOR = 2048
# The engine's own appenders flush by themselves once they hold this many rows.
ENGINE_FLUSH_COUNT = 100 * VECTOR


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
    yield connection
    connection.close()


@pytest.fixture
def produced(con):
    count = [0]

    def tally(i):
        count[0] += 1
        return i

    con.create_function("tally", tally, [BIGINT], BIGINT, side_effects=True)
    return count


@pytest.fixture
def held(con):
    running, release = threading.Event(), threading.Event()

    def hold(x):
        running.set()
        release.wait(10)
        return x

    con.create_function("hold", hold, [BIGINT], BIGINT)
    return running, release


def table_rows(con, table="t"):
    return con.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()


def open_result(con):
    con.execute("SELECT * FROM range(10)")
    con.fetchone()


def contend(running, release, call, outcome, refusal):
    running.wait(10)
    try:
        call()
        outcome.append("ran")
    except refusal as error:
        outcome.append(error)
    finally:
        release.set()


def append_then_raise(app, exception):
    app.append_row(1, "a")
    raise exception


def append_close_then_raise(app, con, exception):
    app.append_row(1, "a")
    con.close()
    raise exception


def append_until_full(app, row, limit):
    for _ in range(limit):
        app.append_row(*row)


def run_isolated(script, timeout=15):
    # A child process keeps a deadlock from stalling the suite; the marker tells a hang from a slow start.
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as hang:
        started = hang.stdout or ""
        if "started" not in (started.decode() if isinstance(started, bytes) else started):
            msg = "the child never reached the statement under test"
            raise AssertionError(msg) from hang
        raise
    return proc.returncode, proc.stdout + proc.stderr


def attach_read_only(con, path):
    file = (path / "ro.db").as_posix()
    with duckdb.connect(file) as writer:
        writer.execute("CREATE TABLE t (k INTEGER)")
    con.execute(f"ATTACH '{file}' AS ro (READ_ONLY)")


def appender_reading_local_prices(con, build):
    prices = build({"k": [1, 2], "price": [10, 20]})  # noqa: F841
    return con.query_appender(
        "INSERT INTO priced FROM appender_data",
        {"k": "INTEGER"},
        source="SELECT b.k, p.price FROM appender_buffer b JOIN prices p USING (k)",
    )


def held_flush_appender(con):
    return con.query_appender(
        "INSERT INTO t FROM appender_data", KV, source="SELECT hold(k) AS k, v FROM appender_buffer"
    )


def interrupt_the_held_flush(app, con, held):
    running, release = held

    def interrupt_while_held():
        running.wait(10)
        con.interrupt()
        release.set()

    thread = threading.Thread(target=interrupt_while_held)
    thread.start()
    try:
        with pytest.raises(duckdb.InterruptException):
            app.flush()
    finally:
        thread.join(10)


REFUSED_TABLES = pytest.mark.parametrize(
    ("setup", "name", "error"),
    [
        pytest.param(lambda con, path: None, "missing", duckdb.CatalogException, id="missing"),
        pytest.param(
            lambda con, path: con.execute("CREATE VIEW v AS SELECT 1 AS k"), "v", duckdb.CatalogException, id="view"
        ),
        pytest.param(attach_read_only, "ro.t", duckdb.InvalidInputException, id="read_only"),
        pytest.param(
            lambda con, path: con.execute(
                "ATTACH ':memory:' AS s; CREATE SCHEMA s; CREATE TABLE memory.s.t (k INTEGER)"
            ),
            "s.t",
            duckdb.BinderException,
            id="ambiguous",
        ),
    ],
)


class TestDefaultInsert:
    def test_rows_go_into_the_table(self, con):
        app = con.appender("t")
        app.append_row(1, "a")
        app.append_row(2, None)
        app.flush()
        assert table_rows(con) == [(1, "a"), (2, None)]

    @pytest.mark.parametrize(
        ("setup", "name", "target"),
        [
            pytest.param(
                "CREATE SCHEMA s; CREATE TABLE s.t (k INTEGER, v VARCHAR); SET search_path = 's'",
                "t",
                "s.t",
                id="search_path",
            ),
            pytest.param("CREATE SCHEMA s; CREATE TABLE s.t (k INTEGER, v VARCHAR)", "s.t", "s.t", id="schema"),
            pytest.param(
                "ATTACH ':memory:' AS other; CREATE TABLE other.t (k INTEGER, v VARCHAR)",
                "other.t",
                "other.t",
                id="catalog",
            ),
            pytest.param(
                "ATTACH ':memory:' AS other; CREATE SCHEMA other.s; CREATE TABLE other.s.t (k INTEGER, v VARCHAR)",
                "other.s.t",
                "other.s.t",
                id="catalog_and_schema",
            ),
            pytest.param("CREATE TEMP TABLE t (k INTEGER, v VARCHAR)", "t", "temp.t", id="temp_shadows_main"),
            pytest.param("CREATE TABLE Mixed (k INTEGER, v VARCHAR)", "MIXED", "mixed", id="case_insensitive"),
            pytest.param('CREATE TABLE "a.b" (k INTEGER, v VARCHAR)', '"a.b"', '"a.b"', id="quoted"),
        ],
    )
    def test_the_name_resolves_as_sql_resolves_it(self, con, setup, name, target):
        con.execute(setup)
        app = con.appender(name)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con, target) == [(1, "a")]

    def test_the_name_resolves_once_at_creation(self, con):
        con.execute("CREATE SCHEMA s; CREATE TABLE s.t (k INTEGER, v VARCHAR)")
        app = con.appender("t")
        con.execute("SET search_path = 's'")
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con, "main.t") == [(1, "a")]

    def test_a_table_created_in_the_open_transaction_resolves(self, con):
        con.execute("BEGIN")
        con.execute("CREATE TABLE u (k INTEGER)")
        app = con.appender("u")
        app.append_row(1)
        app.flush()
        con.execute("COMMIT")
        assert table_rows(con, "u") == [(1,)]

    @REFUSED_TABLES
    def test_a_name_that_is_not_a_writable_table_is_refused(self, con, tmp_path, setup, name, error):
        setup(con, tmp_path)
        with pytest.raises(error):
            con.appender(name)

    @REFUSED_TABLES
    def test_a_refused_name_leaves_the_open_transaction_usable(self, con, tmp_path, setup, name, error):
        setup(con, tmp_path)
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (1, 'a')")
        with pytest.raises(error):
            con.appender(name)
        con.execute("INSERT INTO t VALUES (2, 'b')")
        con.execute("COMMIT")
        assert table_rows(con) == [(1, "a"), (2, "b")]

    def test_generated_columns_are_not_in_the_buffer(self, con):
        con.execute("CREATE TABLE g (a INTEGER, b INTEGER GENERATED ALWAYS AS (a * 2), c VARCHAR)")
        app = con.appender("g")
        app.append_row(1, "x")
        app.flush()
        assert table_rows(con, "g") == [(1, 2, "x")]

    def test_a_value_for_a_generated_column_is_one_too_many(self, con):
        con.execute("CREATE TABLE g (a INTEGER, b INTEGER GENERATED ALWAYS AS (a * 2), c VARCHAR)")
        app = con.appender("g")
        with pytest.raises(WRONG_ARITY):
            app.append_row(1, 2, "x")

    def test_a_null_for_a_column_with_a_default_stays_null(self, con):
        con.execute("CREATE SEQUENCE seq; CREATE TABLE d (id INTEGER DEFAULT nextval('seq'), v VARCHAR DEFAULT 'none')")
        app = con.appender("d")
        app.append_row(None, None)
        app.flush()
        assert table_rows(con, "d") == [(None, None)]

    def test_a_column_added_after_creation_takes_its_default(self, con):
        con.execute("CREATE TABLE p (k INTEGER)")
        app = con.appender("p")
        con.execute("ALTER TABLE p ADD COLUMN w INTEGER DEFAULT 7")
        app.append_row(1)
        app.flush()
        assert table_rows(con, "p") == [(1, 7)]

    def test_a_column_subset_keeps_its_order_and_the_rest_take_their_defaults(self, con):
        con.execute(
            "CREATE SEQUENCE seq; "
            "CREATE TABLE d (id INTEGER DEFAULT nextval('seq'), a VARCHAR, b VARCHAR, w INTEGER DEFAULT 7, n INTEGER)"
        )
        app = con.appender("d", columns=["b", "a"])
        app.append_rows([("b1", "a1"), ("b2", "a2")])
        app.flush()
        either_order = (
            [(1, "a1", "b1", 7, None), (2, "a2", "b2", 7, None)],
            [(1, "a2", "b2", 7, None), (2, "a1", "b1", 7, None)],
        )
        assert table_rows(con, "d") in either_order

    @pytest.mark.parametrize(
        "columns",
        [[], ["nope"], ["b"], ["a", "a"], ["a", "A"]],
        ids=["empty", "unknown", "generated", "repeated", "repeated_ignoring_case"],
    )
    def test_a_bad_column_subset_is_refused(self, con, columns):
        con.execute("CREATE TABLE g (a INTEGER, b INTEGER GENERATED ALWAYS AS (a * 2), c VARCHAR)")
        with pytest.raises(duckdb.InvalidInputException):
            con.appender("g", columns=columns)

    def test_subset_names_resolve_ignoring_case(self, con):
        app = con.appender("t", columns=["V", "K"])
        app.append_row("a", 1)
        app.flush()
        assert table_rows(con) == [(1, "a")]

    def test_the_columns_must_be_a_list(self, con):
        with pytest.raises(TypeError):
            con.appender("t", columns={"k": "INTEGER"})

    def test_the_columns_list_is_read_once_at_creation(self):
        code, output = run_isolated(
            """
            import duckdb
            con = duckdb.connect()
            con.execute("CREATE TABLE t2 (a INTEGER, b INTEGER)")
            columns = []
            class Column:
                # Converting this name clears the list, so a second read of the list would see no columns
                def __str__(self):
                    columns.clear()
                    return "a"
            columns.extend([Column(), "b"])
            print("started", flush=True)
            app = con.appender("t2", columns=columns)
            app.append_row(1, 2)
            app.flush()
            print("rows:", con.execute("SELECT * FROM t2").fetchall())
            """
        )
        assert code == 0, output
        assert "rows: [(1, 2)]" in output

    def test_a_table_name_that_does_not_parse_is_refused(self, con):
        with pytest.raises(duckdb.ParserException):
            con.appender("a..b")

    def test_the_module_level_function_takes_a_connection(self, con):
        app = duckdb.appender("t", connection=con)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == [(1, "a")]

    def test_the_module_level_function_defaults_to_the_default_connection(self):
        default = duckdb.default_connection()
        default.execute("CREATE TABLE appender_default_target (k INTEGER)")
        try:
            app = duckdb.appender("appender_default_target")
            app.append_row(1)
            app.flush()
            assert table_rows(default, "appender_default_target") == [(1,)]
        finally:
            default.execute("DROP TABLE appender_default_target")


class TestTableActions:
    def test_upsert_replaces_rows_on_the_unique_key(self, con):
        con.execute("INSERT INTO t VALUES (1, 'old'), (2, 'kept')")
        app = con.appender("t", action="upsert")
        app.append_rows([(1, "new"), (3, "added")])
        app.flush()
        assert table_rows(con) == [(1, "new"), (2, "kept"), (3, "added")]

    @pytest.mark.parametrize(
        "ddl",
        [
            pytest.param("CREATE TABLE plain (k INTEGER, v VARCHAR)", id="none"),
            pytest.param("CREATE TABLE plain (k INTEGER PRIMARY KEY, v VARCHAR UNIQUE)", id="several"),
        ],
    )
    def test_upsert_without_exactly_one_unique_constraint_is_refused_at_creation(self, con, ddl):
        con.execute(ddl)
        with pytest.raises(duckdb.BinderException):
            con.appender("plain", action="upsert")

    def test_upsert_of_a_column_subset_writes_only_those_columns(self, con):
        con.execute("CREATE TABLE d (k INTEGER PRIMARY KEY, a VARCHAR, b VARCHAR DEFAULT 'default')")
        con.execute("INSERT INTO d VALUES (1, 'old', 'kept')")
        app = con.appender("d", action="upsert", columns=["k", "a"])
        app.append_rows([(1, "new"), (2, "added")])
        app.flush()
        assert table_rows(con, "d") == [(1, "new", "kept"), (2, "added", "default")]

    def test_an_upsert_that_leaves_out_the_key_is_accepted_and_its_flush_fails(self, con):
        app = con.appender("t", action="upsert", columns=["v"])
        app.append_row("no key")
        with pytest.raises(duckdb.ConstraintException):
            app.flush()

    def test_an_upsert_of_only_the_key_inserts_new_keys_and_leaves_existing_rows_alone(self, con):
        con.execute("INSERT INTO t VALUES (1, 'old')")
        app = con.appender("t", action="upsert", columns=["k"])
        app.append_rows([(1,), (2,)])
        app.flush()
        assert table_rows(con) == [(1, "old"), (2, None)]

    def test_an_upsert_matches_on_a_composite_primary_key(self, con):
        con.execute("CREATE TABLE cp (a INTEGER, b INTEGER, v VARCHAR, PRIMARY KEY (a, b))")
        con.execute("INSERT INTO cp VALUES (1, 1, 'x'), (1, 2, 'y')")
        app = con.appender("cp", action="upsert")
        app.append_rows([(1, 2, "Y"), (2, 1, "new")])
        app.flush()
        assert table_rows(con, "cp") == [(1, 1, "x"), (1, 2, "Y"), (2, 1, "new")]

    def test_update_sets_every_other_buffered_column_on_rows_matching_the_key(self, con):
        con.execute("CREATE TABLE u (k INTEGER PRIMARY KEY, v VARCHAR, w INTEGER)")
        con.execute("INSERT INTO u VALUES (1, 'a', 10), (2, 'b', 20)")
        app = con.appender("u", action="update", key=["k"])
        app.append_rows([(1, "A", 11), (3, "C", 33), (None, "N", 0)])
        app.flush()
        assert table_rows(con, "u") == [(1, "A", 11), (2, "b", 20)]

    def test_update_of_a_column_subset_leaves_the_other_columns_alone(self, con):
        con.execute("CREATE TABLE u (k INTEGER PRIMARY KEY, v VARCHAR, w INTEGER)")
        con.execute("INSERT INTO u VALUES (1, 'a', 10), (2, 'b', 20)")
        app = con.appender("u", action="update", key=["k"], columns=["v", "k"])
        app.append_row("B", 2)
        app.flush()
        assert table_rows(con, "u") == [(1, "a", 10), (2, "B", 20)]

    def test_delete_removes_the_rows_matching_the_key(self, con):
        con.execute("CREATE TABLE pairs (a INTEGER, b INTEGER, note VARCHAR)")
        con.execute("INSERT INTO pairs VALUES (1, 1, 'x'), (1, 2, 'y'), (2, 1, 'z')")
        app = con.appender("pairs", action="delete", key=["a", "b"])
        app.append_rows([(1, 2), (2, 2), (None, 1)])
        app.flush()
        assert table_rows(con, "pairs") == [(1, 1, "x"), (2, 1, "z")]

    def test_a_key_name_resolves_ignoring_case(self, con):
        con.execute("CREATE TABLE u (k INTEGER PRIMARY KEY, v VARCHAR)")
        con.execute("INSERT INTO u VALUES (1, 'a')")
        app = con.appender("u", action="update", key=["K"])
        app.append_row(1, "A")
        app.flush()
        assert table_rows(con, "u") == [(1, "A")]

    def test_delete_buffers_only_the_key(self, con):
        app = con.appender("t", action="delete", key=["k"])
        with pytest.raises(WRONG_ARITY):
            app.append_row(1, "a")

    def test_delete_buffers_its_key_in_the_order_given(self, con):
        con.execute("CREATE TABLE pairs (a INTEGER, b INTEGER, note VARCHAR)")
        con.execute("INSERT INTO pairs VALUES (1, 1, 'x'), (1, 2, 'y'), (2, 1, 'z')")
        app = con.appender("pairs", action="delete", key=["b", "a"])
        app.append_row(2, 1)
        app.flush()
        assert table_rows(con, "pairs") == [(1, 1, "x"), (2, 1, "z")]

    @pytest.mark.parametrize(
        ("options", "reason"),
        [
            pytest.param({"key": ["k"]}, "key applies to the update and delete", id="key_with_insert"),
            pytest.param(
                {"action": "insert", "key": ["k"]},
                "key applies to the update and delete",
                id="key_with_explicit_insert",
            ),
            pytest.param(
                {"action": "upsert", "key": ["k"]}, "key applies to the update and delete", id="key_with_upsert"
            ),
            pytest.param({"action": "update"}, "update action needs key", id="update_without_key"),
            pytest.param({"action": "delete"}, "delete action needs key", id="delete_without_key"),
            pytest.param(
                {"action": "update", "key": ["k"], "columns": ["v"]},
                "not one of the appender's columns",
                id="key_outside_the_buffer",
            ),
            pytest.param({"action": "update", "key": ["nope"]}, "nope is not a column", id="key_not_a_column"),
            pytest.param(
                {"action": "update", "key": ["k"], "columns": ["k"]},
                "columns other than the key",
                id="update_of_nothing_but_the_key",
            ),
            pytest.param(
                {"action": "delete", "key": ["k"], "columns": ["k"]}, "takes no columns", id="delete_with_columns"
            ),
            pytest.param({"action": "merge"}, '"insert", "upsert", "update" or "delete"', id="unknown_action"),
        ],
    )
    def test_a_bad_action_is_refused_at_creation(self, con, options, reason):
        with pytest.raises(duckdb.InvalidInputException, match=re.escape(reason)):
            con.appender("t", **options)

    @pytest.mark.parametrize("action", ["update", "delete"])
    @pytest.mark.parametrize(
        ("key", "reason"),
        [
            pytest.param([], "action needs key", id="empty"),
            pytest.param(["k", "k"], "key names k twice", id="repeated"),
            pytest.param(["k", "K"], "key names K twice", id="repeated_ignoring_case"),
        ],
    )
    def test_an_empty_or_repeated_key_is_refused(self, con, action, key, reason):
        with pytest.raises(duckdb.InvalidInputException, match=reason):
            con.appender("t", action=action, key=key)

    @pytest.mark.parametrize("action", ["update", "delete"])
    def test_a_generated_key_column_is_refused(self, con, action):
        con.execute("CREATE TABLE g (a INTEGER, b INTEGER GENERATED ALWAYS AS (a * 2), c VARCHAR)")
        with pytest.raises(duckdb.InvalidInputException, match="b is a generated column"):
            con.appender("g", action=action, key=["b"])

    def test_the_key_must_be_a_list(self, con):
        with pytest.raises(TypeError):
            con.appender("t", action="delete", key="k")

    def test_update_matches_on_every_key_column(self, con):
        con.execute("CREATE TABLE pairs (a INTEGER, b INTEGER, note VARCHAR)")
        con.execute("INSERT INTO pairs VALUES (1, 1, 'x'), (1, 2, 'y'), (2, 1, 'z')")
        app = con.appender("pairs", action="update", key=["a", "b"])
        app.append_rows([(1, 2, "Y"), (2, 2, "nothing matches")])
        app.flush()
        assert table_rows(con, "pairs") == [(1, 1, "x"), (1, 2, "Y"), (2, 1, "z")]

    @pytest.mark.parametrize("name", ["appender_data", "appender_buffer", "appender_target", "excluded"])
    @pytest.mark.parametrize(
        ("action", "key", "rows", "expected"),
        [
            pytest.param(
                "upsert", None, [(1, "new"), (3, "added")], [(1, "new"), (2, "old"), (3, "added")], id="upsert"
            ),
            pytest.param("update", ["k"], [(1, "new"), (3, "missing")], [(1, "new"), (2, "old")], id="update"),
            pytest.param("delete", ["k"], [(1,)], [(2, "old")], id="delete"),
        ],
    )
    def test_each_action_works_on_a_table_named_like_a_name_the_statement_uses(
        self, con, name, action, key, rows, expected
    ):
        con.execute(f"CREATE TABLE {name} (k INTEGER PRIMARY KEY, v VARCHAR)")
        con.execute(f"INSERT INTO {name} VALUES (1, 'old'), (2, 'old')")
        app = con.appender(name, action=action, key=key)
        app.append_rows(rows)
        app.flush()
        assert table_rows(con, name) == expected

    @pytest.mark.parametrize(
        ("action", "key", "rows", "expected"),
        [
            pytest.param("upsert", None, [(1, "A"), (3, "c")], [(1, "A"), (2, "b"), (3, "c")], id="upsert"),
            pytest.param("update", ["select"], [(1, "A")], [(1, "A"), (2, "b")], id="update"),
            pytest.param("delete", ["select"], [(1,)], [(2, "b")], id="delete"),
        ],
    )
    def test_each_action_quotes_a_qualified_table_and_odd_column_names(self, con, action, key, rows, expected):
        con.execute('CREATE SCHEMA "odd schema"')
        con.execute('CREATE TABLE "odd schema"."my table" ("select" INTEGER PRIMARY KEY, "two words" VARCHAR)')
        con.execute("""INSERT INTO "odd schema"."my table" VALUES (1, 'a'), (2, 'b')""")
        app = con.appender('"odd schema"."my table"', action=action, key=key)
        app.append_rows(rows)
        app.flush()
        assert table_rows(con, '"odd schema"."my table"') == expected

    @pytest.mark.parametrize(
        ("action", "rows", "expected"),
        [
            pytest.param("update", [(1, "z")], [(1, "z"), (1, "z"), (2, "c")], id="update"),
            pytest.param("delete", [(1,)], [(2, "c")], id="delete"),
        ],
    )
    def test_a_key_that_matches_several_rows_acts_on_all_of_them(self, con, action, rows, expected):
        con.execute("CREATE TABLE groups (g INTEGER, v VARCHAR)")
        con.execute("INSERT INTO groups VALUES (1, 'a'), (1, 'b'), (2, 'c')")
        app = con.appender("groups", action=action, key=["g"])
        app.append_rows(rows)
        app.flush()
        assert table_rows(con, "groups") == expected

    @pytest.mark.parametrize(
        ("action", "key"), [pytest.param("upsert", None, id="upsert"), pytest.param("update", ["k"], id="update")]
    )
    def test_a_generated_column_follows_the_columns_an_action_writes(self, con, action, key):
        con.execute("CREATE TABLE gen (k INTEGER PRIMARY KEY, a INTEGER, d INTEGER GENERATED ALWAYS AS (a * 2))")
        con.execute("INSERT INTO gen (k, a) VALUES (1, 10)")
        app = con.appender("gen", action=action, key=key)
        app.append_row(1, 11)
        app.flush()
        assert table_rows(con, "gen") == [(1, 11, 22)]

    def test_the_module_level_function_takes_an_action_and_a_key(self, con):
        con.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
        app = duckdb.appender("t", action="delete", key=["k"], connection=con)
        app.append_row(1)
        app.flush()
        assert table_rows(con) == [(2, "b")]

    def test_an_upsert_refused_by_the_bind_leaves_the_open_transaction_usable(self, con):
        con.execute("CREATE TABLE plain (k INTEGER, v VARCHAR)")
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (1, 'a')")
        with pytest.raises(duckdb.BinderException):
            con.appender("plain", action="upsert")
        con.execute("INSERT INTO t VALUES (2, 'b')")
        con.execute("COMMIT")
        assert table_rows(con) == [(1, "a"), (2, "b")]

    def test_the_key_list_is_read_once_at_creation(self):
        code, output = run_isolated(
            """
            import duckdb
            con = duckdb.connect()
            con.execute("CREATE TABLE u (k INTEGER PRIMARY KEY, v VARCHAR)")
            con.execute("INSERT INTO u VALUES (1, 'a')")
            key = []
            class Name:
                # Converting this name clears the list, so a second read of the list would see no key
                def __str__(self):
                    key.clear()
                    return "k"
            key.append(Name())
            print("started", flush=True)
            app = con.appender("u", action="update", key=key)
            app.append_row(1, "A")
            app.flush()
            print("rows:", con.execute("SELECT * FROM u").fetchall())
            """
        )
        assert code == 0, output
        assert "rows: [(1, 'A')]" in output


class TestCustomColumns:
    def test_columns_declare_the_buffer_in_order(self, con):
        con.execute("CREATE TABLE products (id INTEGER, sku VARCHAR)")
        con.execute("INSERT INTO products VALUES (10, 'widget'), (20, 'gadget')")
        con.execute("CREATE TABLE orders (product_id INTEGER, qty INTEGER)")
        app = con.query_appender(
            "INSERT INTO orders (product_id, qty) FROM appender_data",
            {"sku": "VARCHAR", "qty": "INTEGER"},
            source="SELECT p.id AS product_id, b.qty FROM appender_buffer b JOIN products p USING (sku)",
        )
        app.append_row("widget", 3)
        app.append_row("gadget", None)
        app.flush()
        assert table_rows(con, "orders") == [(10, 3), (20, None)]

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            pytest.param("INTEGER", "INTEGER", id="sql_text"),
            pytest.param("INT[]", "INTEGER[]", id="sql_text_list"),
            pytest.param("STRUCT(a INT, b VARCHAR)", "STRUCT(a INTEGER, b VARCHAR)", id="sql_text_struct"),
            pytest.param(INTEGER, "INTEGER", id="sqltype"),
            pytest.param(int, "BIGINT", id="python_type"),
            pytest.param(list[str], "VARCHAR[]", id="generic_alias"),
            pytest.param({"a": int}, "STRUCT(a BIGINT)", id="python_dict"),
        ],
    )
    def test_a_type_takes_any_form_the_type_conversion_accepts(self, con, declared, expected):
        con.execute("CREATE TABLE types (name VARCHAR)")
        app = con.query_appender(
            "INSERT INTO types FROM appender_data",
            {"x": declared},
            source="SELECT typeof(x) AS name FROM appender_buffer",
        )
        app.append_row(None)
        app.flush()
        assert table_rows(con, "types") == [(expected,)]

    def test_a_type_name_resolves_against_the_connection(self, con):
        con.execute("CREATE TYPE mood AS ENUM ('happy', 'sad')")
        con.execute("CREATE TABLE ranges (r VARCHAR)")
        app = con.query_appender(
            "INSERT INTO ranges FROM appender_data",
            {"m": "mood"},
            source="SELECT enum_range(m)::VARCHAR AS r FROM appender_buffer",
        )
        app.append_row("happy")
        app.flush()
        assert table_rows(con, "ranges") == [("[happy, sad]",)]

    @pytest.mark.parametrize(
        ("declared", "error"),
        [
            pytest.param("NULL", duckdb.InvalidInputException, id="null_text"),
            pytest.param(SQLNULL, duckdb.InvalidInputException, id="null_sqltype"),
            pytest.param("no_such_type", duckdb.CatalogException, id="unknown"),
            pytest.param("DECIMAL(40, 2)", duckdb.BinderException, id="out_of_range"),
        ],
    )
    def test_a_type_that_cannot_hold_values_is_refused_at_creation(self, con, declared, error):
        with pytest.raises(error):
            con.query_appender("INSERT INTO t (k) FROM appender_data", {"x": declared})

    @pytest.mark.parametrize(
        ("dml", "columns"),
        [
            pytest.param("INSERT INTO t FROM appender_data", {"k": "INTEGER", "K": "VARCHAR"}, id="read"),
            pytest.param("INSERT INTO t VALUES (7, 'fixed')", {"x": int, "X": int}, id="never_read"),
        ],
    )
    def test_names_differing_only_in_case_are_refused(self, con, dml, columns):
        with pytest.raises(duckdb.BinderException):
            con.query_appender(dml, columns)

    def test_names_are_identifiers_not_sql(self, con):
        con.execute('CREATE TABLE odd ("my col" INTEGER, "select" VARCHAR)')
        app = con.query_appender(
            "INSERT INTO odd FROM appender_data",
            {"my col": "INTEGER", "select": "VARCHAR"},
            source='SELECT "my col", "select" FROM appender_buffer',
        )
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con, "odd") == [(1, "a")]

    @pytest.mark.parametrize(
        ("columns", "error"),
        [
            pytest.param({}, duckdb.InvalidInputException, id="empty"),
            pytest.param(["k", "v"], TypeError, id="a_list"),
        ],
    )
    def test_the_columns_must_be_a_non_empty_dict(self, con, columns, error):
        with pytest.raises(error):
            con.query_appender("INSERT INTO t FROM appender_data", columns)

    def test_an_empty_column_name_is_refused(self, con):
        with pytest.raises(duckdb.InvalidInputException):
            con.query_appender("INSERT INTO t (k) FROM appender_data", {"": int})

    def test_the_columns_are_read_once_at_creation(self, con):
        columns = {}

        class ChangingType(str):
            # Converting this type removes y from the dict, so a second read of the dict would see one column.
            def __str__(self) -> str:
                columns.pop("y", None)
                return "INTEGER"

        columns.update(x=ChangingType("INTEGER"), y=int)
        con.execute("CREATE TABLE pair (x INTEGER, y BIGINT)")
        app = con.query_appender("INSERT INTO pair FROM appender_data", columns)
        app.append_row(1, 2)
        app.flush()
        assert table_rows(con, "pair") == [(1, 2)]

    def test_the_module_level_query_appender_takes_a_connection(self, con):
        app = duckdb.query_appender("INSERT INTO t FROM appender_data", KV, connection=con)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == [(1, "a")]


class TestStatementKinds:
    @pytest.mark.parametrize(
        ("dml", "expected"),
        [
            pytest.param(
                "INSERT INTO t SELECT k + 10, v FROM appender_data",
                [(1, "old1"), (2, "old2"), (12, "new2"), (13, "new3")],
                id="insert",
            ),
            pytest.param(
                "INSERT OR IGNORE INTO t FROM appender_data", [(1, "old1"), (2, "old2"), (3, "new3")], id="or_ignore"
            ),
            pytest.param(
                "INSERT OR REPLACE INTO t FROM appender_data", [(1, "old1"), (2, "new2"), (3, "new3")], id="or_replace"
            ),
            pytest.param(
                "INSERT INTO t SELECT * FROM appender_data ON CONFLICT (k) DO UPDATE SET v = excluded.v",
                [(1, "old1"), (2, "new2"), (3, "new3")],
                id="on_conflict_do_update",
            ),
            pytest.param(
                "INSERT INTO t SELECT * FROM appender_data ON CONFLICT DO NOTHING",
                [(1, "old1"), (2, "old2"), (3, "new3")],
                id="on_conflict_do_nothing",
            ),
            pytest.param(
                "UPDATE t SET v = d.v FROM appender_data d WHERE t.k = d.k",
                [(1, "old1"), (2, "new2")],
                id="update_from",
            ),
            pytest.param("DELETE FROM t USING appender_data d WHERE t.k = d.k", [(1, "old1")], id="delete_using"),
            pytest.param(
                "MERGE INTO t USING appender_data d ON t.k = d.k "
                "WHEN MATCHED THEN UPDATE SET v = d.v WHEN NOT MATCHED THEN INSERT VALUES (d.k, d.v)",
                [(1, "old1"), (2, "new2"), (3, "new3")],
                id="merge",
            ),
        ],
    )
    def test_each_statement_kind_applies_the_buffered_rows(self, con, dml, expected):
        con.execute("INSERT INTO t VALUES (1, 'old1'), (2, 'old2')")
        app = con.query_appender(dml, KV)
        app.append_rows([(2, "new2"), (3, "new3")])
        app.flush()
        assert table_rows(con) == expected

    @pytest.mark.parametrize(
        ("source", "dml"),
        [
            pytest.param(
                "WITH b AS (SELECT * FROM appender_buffer) SELECT k + 10 AS k, v FROM b",
                "INSERT INTO t FROM appender_data",
                id="in_the_source",
            ),
            pytest.param(
                "SELECT * FROM appender_buffer",
                "WITH d AS (SELECT k + 10 AS k, v FROM appender_data) INSERT INTO t FROM d",
                id="in_the_dml",
            ),
        ],
    )
    def test_a_cte_of_the_users_can_read_the_buffer(self, con, source, dml):
        app = con.query_appender(dml, KV, source=source)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == [(11, "a")]

    @pytest.mark.parametrize(
        "dml",
        [
            pytest.param("SELECT * FROM appender_data", id="select"),
            pytest.param("CREATE TABLE u AS FROM appender_data", id="create_table_as"),
            pytest.param("COPY (FROM appender_data) TO '{out}'", id="copy"),
            pytest.param("EXPLAIN INSERT INTO t FROM appender_data", id="explain"),
            pytest.param("INSERT INTO t FROM appender_data; INSERT INTO t FROM appender_data", id="two_statements"),
            pytest.param("", id="empty"),
        ],
    )
    def test_any_other_statement_is_refused_at_creation(self, con, tmp_path, dml):
        dml = dml.format(out=(tmp_path / "out.csv").as_posix())
        with pytest.raises(duckdb.InvalidInputException):
            con.query_appender(dml, KV)

    @pytest.mark.parametrize(
        "dml",
        [
            pytest.param("INSERT INTO t FROM appender_data RETURNING k", id="insert"),
            pytest.param("INSERT OR REPLACE INTO t FROM appender_data RETURNING *", id="or_replace"),
            pytest.param("UPDATE t SET v = d.v FROM appender_data d WHERE t.k = d.k RETURNING t.k", id="update"),
            pytest.param("DELETE FROM t USING appender_data d WHERE t.k = d.k RETURNING t.k", id="delete"),
            pytest.param(
                "MERGE INTO t USING appender_data d ON t.k = d.k WHEN MATCHED THEN DELETE RETURNING merge_action",
                id="merge",
            ),
        ],
    )
    def test_returning_is_refused_at_creation(self, con, dml):
        with pytest.raises(duckdb.InvalidInputException):
            con.query_appender(dml, KV)

    @pytest.mark.parametrize(
        ("source", "dml"),
        [
            pytest.param(
                "SELECT k, ? AS v FROM appender_buffer", "INSERT INTO t FROM appender_data", id="source_positional"
            ),
            pytest.param(
                "SELECT k, $1 AS v FROM appender_buffer", "INSERT INTO t FROM appender_data", id="source_numbered"
            ),
            pytest.param(
                "SELECT k, $v AS v FROM appender_buffer", "INSERT INTO t FROM appender_data", id="source_named"
            ),
            pytest.param("SELECT * FROM appender_buffer", "INSERT INTO t SELECT k, ? FROM appender_data", id="dml"),
        ],
    )
    def test_a_parameter_is_refused_at_creation(self, con, source, dml):
        with pytest.raises(duckdb.InvalidInputException):
            con.query_appender(dml, KV, source=source)

    @pytest.mark.parametrize(
        "name",
        ["appender_buffer", "appender_data", "Appender_Data"],
        ids=["buffer", "data", "data_ignoring_case"],
    )
    def test_a_cte_of_the_dml_named_like_ours_is_refused_at_creation(self, con, name):
        dml = f"WITH {name} AS (SELECT 1 AS k, 'x' AS v) INSERT INTO t FROM appender_data"
        with pytest.raises(duckdb.InvalidInputException):
            con.query_appender(dml, KV)

    def test_a_dml_that_never_reads_the_buffer_is_allowed(self, con):
        app = con.query_appender("INSERT INTO t VALUES (7, 'fixed')", KV)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == [(7, "fixed")]

    def test_buffer_columns_nothing_reads_are_allowed(self, con):
        app = con.query_appender(
            "INSERT INTO t SELECT k, v FROM appender_data", {"k": "INTEGER", "v": "VARCHAR", "note": "VARCHAR"}
        )
        app.append_row(1, "a", "unread")
        app.flush()
        assert table_rows(con) == [(1, "a")]

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            pytest.param("VALUES (5, 'v')", [(5, "v")], id="values"),
            pytest.param("FROM appender_buffer", [(1, "a")], id="from_first"),
            pytest.param(
                "SELECT k, v FROM appender_buffer UNION ALL SELECT k + 10, v FROM appender_buffer",
                [(1, "a"), (11, "a")],
                id="union_all",
            ),
            pytest.param("WITH b AS (FROM appender_buffer) SELECT k + 20 AS k, v FROM b", [(21, "a")], id="own_with"),
            pytest.param(
                "SELECT b.k + r.range AS k, b.v FROM appender_buffer b, range(2) r",
                [(1, "a"), (2, "a")],
                id="join_with_range",
            ),
        ],
    )
    def test_the_source_is_one_query_returning_rows(self, con, source, expected):
        app = con.query_appender("INSERT INTO t FROM appender_data", KV, source=source)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == expected

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("INSERT INTO t VALUES (9, 'x') RETURNING k, v", id="insert_returning"),
            pytest.param("CALL range(3)", id="call"),
            pytest.param("EXPLAIN SELECT * FROM appender_buffer", id="explain"),
            pytest.param("FROM appender_buffer; FROM appender_buffer", id="two_statements"),
        ],
    )
    def test_a_source_that_is_not_one_query_returning_rows_is_refused(self, con, source):
        with pytest.raises(duckdb.InvalidInputException):
            con.query_appender("INSERT INTO t FROM appender_data", KV, source=source)


BIND_ERRORS = pytest.mark.parametrize(
    ("source", "dml", "error"),
    [
        pytest.param(
            "SELECT * FROM appender_buffer",
            "INSERT INTO missing FROM appender_data",
            duckdb.CatalogException,
            id="table",
        ),
        pytest.param(
            "SELECT nte AS k, v FROM appender_buffer",
            "INSERT INTO t FROM appender_data",
            duckdb.BinderException,
            id="column",
        ),
        pytest.param(
            "SELECT k, v, k AS w FROM appender_buffer",
            "INSERT INTO t FROM appender_data",
            duckdb.BinderException,
            id="arity",
        ),
        pytest.param(
            "SELECT no_such_function(k) AS k, v FROM appender_buffer",
            "INSERT INTO t FROM appender_data",
            duckdb.CatalogException,
            id="function",
        ),
        pytest.param(
            "SELEC k FROM appender_buffer", "INSERT INTO t FROM appender_data", duckdb.ParserException, id="syntax"
        ),
    ],
)


class TestCheckedAtCreation:
    @BIND_ERRORS
    def test_a_statement_that_cannot_bind_is_refused_at_creation(self, con, source, dml, error):
        with pytest.raises(error):
            con.query_appender(dml, KV, source=source)

    @BIND_ERRORS
    def test_a_refusal_at_creation_leaves_the_open_transaction_usable(self, con, source, dml, error):
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (1, 'a')")
        with pytest.raises(error):
            con.query_appender(dml, KV, source=source)
        con.execute("INSERT INTO t VALUES (2, 'b')")
        con.execute("COMMIT")
        assert table_rows(con) == [(1, "a"), (2, "b")]

    def test_creation_runs_nothing(self, con):
        con.execute("INSERT INTO t VALUES (1, 'a')")
        con.query_appender("MERGE INTO t USING appender_data d ON t.k = d.k WHEN NOT MATCHED BY SOURCE THEN DELETE", KV)
        assert table_rows(con) == [(1, "a")]

    @pytest.mark.parametrize(
        ("source", "row", "error"),
        [
            pytest.param(
                "SELECT v::INTEGER AS k, v FROM appender_buffer",
                (1, "not a number"),
                duckdb.ConversionException,
                id="cast",
            ),
            pytest.param("SELECT * FROM appender_buffer", (1, "again"), duckdb.ConstraintException, id="primary_key"),
        ],
    )
    def test_casts_and_constraints_fail_only_at_flush(self, con, source, row, error):
        con.execute("INSERT INTO t VALUES (1, 'a')")
        app = con.query_appender("INSERT INTO t FROM appender_data", KV, source=source)
        app.append_row(*row)
        with pytest.raises(error):
            app.flush()

    def test_a_custom_dml_into_a_read_only_catalog_fails_only_at_flush(self, con, tmp_path):
        attach_read_only(con, tmp_path)
        app = con.query_appender("INSERT INTO ro.t FROM appender_data", {"k": "INTEGER"})
        app.append_row(1)
        with pytest.raises(duckdb.InvalidInputException):
            app.flush()

    @pytest.mark.parametrize(
        "create",
        [
            pytest.param(lambda c: c.appender("t"), id="table"),
            pytest.param(lambda c: c.query_appender("INSERT INTO t FROM appender_data", KV), id="custom"),
        ],
    )
    def test_creation_works_while_a_result_is_open(self, con, create):
        open_result(con)
        create(con)
        assert con.fetchall() == [(i,) for i in range(1, 10)]

    @pytest.mark.parametrize(
        "create",
        [
            pytest.param(lambda c: c.appender("t"), id="table"),
            pytest.param(
                lambda c: c.query_appender("INSERT INTO t FROM appender_data", {"k": INTEGER, "v": VARCHAR}),
                id="query_with_sqltypes",
            ),
            pytest.param(
                lambda c: c.query_appender("INSERT INTO t FROM appender_data", KV), id="query_with_type_names"
            ),
        ],
    )
    def test_creation_inside_an_aborted_transaction_is_refused(self, con, create):
        con.begin()
        with pytest.raises(duckdb.InvalidInputException):
            con.execute("SELECT error('boom')")
        with pytest.raises(duckdb.TransactionException):
            create(con)
        con.rollback()
        create(con)

    @pytest.mark.parametrize(
        "options", [{"action": "merge"}, {"action": "update"}], ids=["unknown_action", "update_without_key"]
    )
    def test_an_aborted_transaction_is_reported_before_a_bad_action(self, con, options):
        con.begin()
        with pytest.raises(duckdb.InvalidInputException):
            con.execute("SELECT error('boom')")
        with pytest.raises(duckdb.TransactionException):
            con.appender("t", **options)
        con.rollback()
        with pytest.raises(duckdb.InvalidInputException):
            con.appender("t", **options)


class TestRows:
    @pytest.mark.parametrize("values", [(2,), (2, "b", "extra"), ()], ids=["too_few", "too_many", "none"])
    def test_wrong_arity_raises_and_buffers_nothing(self, con, values):
        app = con.appender("t")
        app.append_row(1, "a")
        with pytest.raises(WRONG_ARITY):
            app.append_row(*values)
        app.flush()
        assert table_rows(con) == [(1, "a")]

    def test_a_failed_conversion_buffers_nothing_of_its_row(self, con):
        con.execute("CREATE TABLE w (a INTEGER, b SMALLINT)")
        app = con.appender("w")
        app.append_row(1, 1)
        with pytest.raises(duckdb.ConversionException):
            app.append_row(2, 100_000)
        app.flush()
        assert table_rows(con, "w") == [(1, 1)]

    def test_a_failed_row_leaves_no_null_behind(self, con):
        con.execute("CREATE TABLE w (a INTEGER, b SMALLINT)")
        app = con.appender("w")
        app.append_row(1, 1)
        with pytest.raises(duckdb.ConversionException):
            app.append_row(None, 100_000)
        app.append_row(3, 3)
        app.flush()
        assert table_rows(con, "w") == [(1, 1), (3, 3)]

    @pytest.mark.parametrize(
        ("bad", "error"),
        [
            pytest.param((3,), WRONG_ARITY, id="arity"),
            pytest.param(("x", "c"), duckdb.ConversionException, id="value"),
            pytest.param("12", duckdb.InvalidInputException, id="str"),
            pytest.param(b"12", duckdb.InvalidInputException, id="bytes"),
            pytest.param({3: "x", 4: "y"}, duckdb.InvalidInputException, id="dict"),
        ],
    )
    def test_append_rows_stops_at_the_first_bad_row_and_keeps_the_rows_before_it(self, con, bad, error):
        consumed = []

        def rows():
            for row in [(1, "a"), (2, "b"), bad, (4, "d")]:
                consumed.append(row)
                yield row

        app = con.appender("t")
        with pytest.raises(error):
            app.append_rows(rows())
        app.flush()
        assert table_rows(con) == [(1, "a"), (2, "b")]
        assert len(consumed) == 3

    def test_an_exception_from_the_iterable_propagates_unchanged_and_keeps_the_rows_before_it(self, con):
        failure = ValueError("the source of rows broke")

        def rows():
            yield (1, "a")
            raise failure

        app = con.appender("t")
        with pytest.raises(ValueError, match="the source of rows broke") as raised:
            app.append_rows(rows())
        app.flush()
        assert raised.value is failure
        assert table_rows(con) == [(1, "a")]

    def test_append_rows_takes_any_iterable_of_sequences(self, con):
        app = con.appender("t")
        app.append_rows((k, str(k)) for k in range(2))
        app.append_rows([[2, None]])
        app.flush()
        assert table_rows(con) == [(0, "0"), (1, "1"), (2, None)]


class TestAppendingRunsNothing:
    def test_appending_works_while_a_result_is_open(self, con):
        app = con.appender("t")
        open_result(con)
        app.append_row(1, "a")
        app.append_rows([(2, "b")])
        assert con.fetchall() == [(i,) for i in range(1, 10)]

    def test_clear_works_while_a_result_is_open(self, con):
        app = con.appender("t")
        app.append_row(1, "a")
        open_result(con)
        app.clear()
        assert con.fetchall() == [(i,) for i in range(1, 10)]
        app.flush()
        assert table_rows(con) == []

    def test_buffered_rows_are_invisible_until_flushed(self, con):
        other = con.cursor()
        app = con.appender("t")
        app.append_row(1, "a")
        assert table_rows(con) == []
        assert table_rows(other) == []

    def test_appending_runs_no_part_of_the_flush_statement(self, con, produced):
        app = con.query_appender(
            "INSERT INTO t FROM appender_data", KV, source="SELECT tally(k) AS k, v FROM appender_buffer"
        )
        app.append_rows([(1, "a"), (2, "b")])
        assert produced[0] == 0

    def test_appending_works_while_another_thread_runs_a_statement(self, con, held):
        running, release = held
        app = con.appender("t")
        outcome = []
        thread = threading.Thread(
            target=contend, args=(running, release, lambda: app.append_row(1, "a"), outcome, duckdb.Error)
        )
        thread.start()
        con.execute("SELECT hold(i) FROM range(1) t(i)").fetchall()
        thread.join(10)
        assert outcome == ["ran"]

    def test_a_callback_inside_a_query_can_append(self, con):
        app = con.appender("t")

        def collect(x):
            app.append_row(x, "from a callback")
            return x

        con.create_function("collect", collect, [BIGINT], BIGINT, side_effects=True)
        con.execute("SELECT collect(i) FROM range(2) t(i)").fetchall()
        app.flush()
        assert table_rows(con) == [(0, "from a callback"), (1, "from a callback")]


class TestFlush:
    def test_each_flush_applies_only_the_rows_buffered_since_the_last(self, con):
        con.execute("CREATE TABLE log (k INTEGER)")
        app = con.appender("log")
        app.append_row(1)
        app.flush()
        app.append_rows([(2,), (3,)])
        app.flush()
        assert table_rows(con, "log") == [(1,), (2,), (3,)]

    def test_a_flush_applies_every_row_across_chunk_boundaries(self, con):
        con.execute("CREATE TABLE log (k INTEGER)")
        app = con.appender("log")
        rows = 2 * VECTOR + 1
        app.append_rows((i,) for i in range(rows))
        app.flush()
        assert con.execute("SELECT count(*), sum(k) FROM log").fetchall() == [(rows, rows * (rows - 1) // 2)]

    def test_long_strings_survive_across_chunk_boundaries(self, con):
        con.execute("CREATE TABLE words (s VARCHAR)")
        app = con.appender("words")
        strings = [f"longer than the inline length {i:05d}" for i in range(2 * VECTOR + 1)]
        app.append_rows((s,) for s in strings)
        app.flush()
        assert sorted(s for (s,) in con.execute("SELECT s FROM words").fetchall()) == sorted(strings)

    def test_nested_values_survive_across_chunk_boundaries(self, con):
        con.execute("CREATE TABLE nested (l INTEGER[], s STRUCT(a INTEGER, b VARCHAR))")
        app = con.appender("nested")
        rows = [
            (
                None if i % 7 == 0 else [i, None, i + 1][: i % 4],
                None if i % 11 == 0 else {"a": i, "b": f"longer than the inline length {i:05d}"},
            )
            for i in range(2 * VECTOR + 1)
        ]
        app.append_rows(rows)
        app.flush()
        stored = con.execute("SELECT l, s FROM nested").fetchall()
        assert sorted(stored, key=repr) == sorted(rows, key=repr)

    def test_a_plain_insert_keeps_the_append_order(self, con):
        con.execute("SET threads = 4")
        con.execute("CREATE TABLE log (k INTEGER)")
        app = con.appender("log")
        appended = list(range(3 * VECTOR, -1, -1))
        app.append_rows((k,) for k in appended)
        app.flush()
        assert [k for (k,) in con.execute("SELECT k FROM log").fetchall()] == appended

    def test_nulls_that_move_between_chunks_arrive(self, con):
        con.execute("CREATE TABLE log (k INTEGER)")
        app = con.appender("log")
        values = [None if (i % 2 == 0) == (i < VECTOR) else i for i in range(2 * VECTOR + 1)]
        app.append_rows((v,) for v in values)
        app.flush()
        present = [v for v in values if v is not None]
        expected = [(len(values), len(present), sum(present))]
        assert con.execute("SELECT count(*), count(k), sum(k) FROM log").fetchall() == expected

    def test_two_appenders_on_one_connection_flush_only_their_own_rows(self, con):
        con.execute("CREATE TABLE log (k INTEGER)")
        first, second = con.appender("log"), con.appender("log")
        first.append_row(1)
        second.append_row(2)
        first.flush()
        first.append_row(3)
        second.flush()
        first.flush()
        assert table_rows(con, "log") == [(1,), (2,), (3,)]

    def test_the_target_resolves_at_each_flush(self, con):
        app = con.appender("t")
        con.execute("DROP TABLE t; CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR, w INTEGER DEFAULT 5)")
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == [(1, "a", 5)]

    def test_a_frame_named_in_the_source_is_resolved_where_the_appender_is_created(self, con):
        pd = pytest.importorskip("pandas")
        con.execute("CREATE TABLE priced (k INTEGER, price BIGINT)")
        app = appender_reading_local_prices(con, pd.DataFrame)
        app.append_rows([(1,), (2,)])
        app.flush()
        assert table_rows(con, "priced") == [(1, 10), (2, 20)]

    def test_a_frame_of_the_same_name_where_flush_runs_is_not_used(self, con):
        pd = pytest.importorskip("pandas")
        con.execute("CREATE TABLE priced (k INTEGER, price BIGINT)")
        app = appender_reading_local_prices(con, pd.DataFrame)
        prices = pd.DataFrame({"k": [1, 2], "price": [-1, -2]})  # noqa: F841
        app.append_row(1)
        app.flush()
        assert table_rows(con, "priced") == [(1, 10)]

    def test_a_frame_resolved_at_creation_serves_every_flush(self, con):
        pd = pytest.importorskip("pandas")
        con.execute("CREATE TABLE priced (k INTEGER, price BIGINT)")
        app = appender_reading_local_prices(con, pd.DataFrame)
        app.append_row(1)
        app.flush()
        app.append_row(2)
        app.flush()
        assert table_rows(con, "priced") == [(1, 10), (2, 20)]

    def test_an_arrow_table_resolved_at_creation_serves_every_flush(self, con):
        pa = pytest.importorskip("pyarrow")
        con.execute("CREATE TABLE priced (k INTEGER, price BIGINT)")
        app = appender_reading_local_prices(con, pa.table)
        app.append_row(1)
        app.flush()
        app.append_row(2)
        app.flush()
        assert table_rows(con, "priced") == [(1, 10), (2, 20)]

    def test_a_dropped_table_is_not_replaced_by_a_python_object_at_flush(self, con):
        pd = pytest.importorskip("pandas")
        con.execute("CREATE TABLE prices (k INTEGER, price BIGINT)")
        con.execute("INSERT INTO prices VALUES (1, 10)")
        con.execute("CREATE TABLE priced (k INTEGER, price BIGINT)")
        app = con.query_appender(
            "INSERT INTO priced FROM appender_data",
            {"k": "INTEGER"},
            source="SELECT b.k, p.price FROM appender_buffer b JOIN prices p USING (k)",
        )
        con.execute("DROP TABLE prices")
        prices = pd.DataFrame({"k": [1], "price": [-1]})  # noqa: F841
        app.append_row(1)
        with pytest.raises(duckdb.CatalogException):
            app.flush()
        assert table_rows(con, "priced") == []

    def test_an_interrupt_while_no_flush_runs_does_not_affect_the_next(self, con):
        app = con.appender("t")
        app.append_row(1, "a")
        con.interrupt()
        app.flush()
        assert table_rows(con) == [(1, "a")]

    def test_a_source_can_read_the_buffer_twice(self, con):
        con.execute("CREATE TABLE log (k INTEGER, v VARCHAR)")
        app = con.query_appender(
            "INSERT INTO log FROM appender_data",
            KV,
            source="SELECT * FROM appender_buffer UNION ALL SELECT * FROM appender_buffer",
        )
        app.append_rows([(1, "a"), (2, "b")])
        app.flush()
        assert table_rows(con, "log") == [(1, "a"), (1, "a"), (2, "b"), (2, "b")]

    def test_a_flush_outside_a_transaction_commits_by_itself(self, con):
        other = con.cursor()
        app = con.appender("t")
        app.append_row(1, "a")
        app.flush()
        assert table_rows(other) == [(1, "a")]

    def test_a_flush_inside_begin_is_visible_elsewhere_only_after_commit(self, con):
        other = con.cursor()
        app = con.appender("t")
        con.execute("BEGIN")
        app.append_row(1, "a")
        app.flush()
        assert table_rows(other) == []
        con.execute("COMMIT")
        assert table_rows(other) == [(1, "a")]

    def test_a_flush_undone_by_rollback_is_not_applied_again(self, con):
        app = con.appender("t")
        con.execute("BEGIN")
        app.append_row(1, "a")
        app.flush()
        con.execute("ROLLBACK")
        app.flush()
        assert table_rows(con) == []

    def test_a_flush_runs_on_the_appenders_connection(self, con):
        cursor = con.cursor()
        cursor.execute("CREATE TEMP TABLE scratch (k INTEGER)")
        app = cursor.appender("scratch")
        app.append_row(1)
        app.flush()
        assert table_rows(cursor, "scratch") == [(1,)]

    def test_an_empty_flush_runs_nothing(self, con):
        con.execute("INSERT INTO t VALUES (1, 'a')")
        app = con.query_appender(
            "MERGE INTO t USING appender_data d ON t.k = d.k WHEN NOT MATCHED BY SOURCE THEN DELETE", KV
        )
        app.flush()
        assert table_rows(con) == [(1, "a")]

    def test_an_empty_flush_does_not_need_the_connection(self, con):
        app = con.appender("t")
        open_result(con)
        app.flush()
        assert con.fetchall() == [(i,) for i in range(1, 10)]

    @pytest.mark.parametrize(
        "dml",
        [
            pytest.param("INSERT OR REPLACE INTO t FROM appender_data", id="or_replace"),
            pytest.param(
                "INSERT INTO t SELECT * FROM appender_data ON CONFLICT (k) DO UPDATE SET v = excluded.v",
                id="on_conflict_do_update",
            ),
        ],
    )
    def test_a_key_repeated_within_one_upsert_keeps_one_of_its_rows(self, con, dml):
        con.execute("INSERT INTO t VALUES (1, 'old')")
        app = con.query_appender(dml, KV)
        app.append_rows([(1, "first"), (1, "second"), (2, "first"), (2, "second")])
        app.flush()
        one_per_key = [[(1, a), (2, b)] for a in ("first", "second") for b in ("first", "second")]
        assert table_rows(con) in one_per_key

    def test_a_key_repeated_within_one_update_from_applies_one_of_its_rows(self, con):
        con.execute("INSERT INTO t VALUES (1, 'old')")
        app = con.query_appender("UPDATE t SET v = d.v FROM appender_data d WHERE t.k = d.k", KV)
        app.append_rows([(1, "first"), (1, "second")])
        app.flush()
        assert table_rows(con) in ([(1, "first")], [(1, "second")])

    def test_a_key_repeated_within_one_merge_raises(self, con):
        con.execute("INSERT INTO t VALUES (1, 'old')")
        app = con.query_appender(
            "MERGE INTO t USING appender_data d ON t.k = d.k "
            "WHEN MATCHED THEN UPDATE SET v = d.v WHEN NOT MATCHED THEN INSERT VALUES (d.k, d.v)",
            KV,
        )
        app.append_rows([(1, "first"), (1, "second")])
        with pytest.raises(duckdb.InvalidInputException, match="more than once"):
            app.flush()


class TestFlushFailures:
    def test_a_flush_refused_by_an_open_result_keeps_the_rows_and_the_transaction(self, con):
        app = con.appender("t")
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (1, 'a')")
        app.append_row(2, "b")
        open_result(con)
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            app.flush()
        con.fetchall()
        app.flush()
        con.execute("COMMIT")
        assert table_rows(con) == [(1, "a"), (2, "b")]

    def test_a_flush_refused_while_another_thread_runs_a_statement_keeps_the_rows(self, con, held):
        running, release = held
        app = con.appender("t")
        app.append_row(1, "a")
        outcome = []
        thread = threading.Thread(
            target=contend, args=(running, release, app.flush, outcome, duckdb.ResourceInUseException)
        )
        thread.start()
        con.execute("SELECT hold(i) FROM range(1) t(i)").fetchall()
        thread.join(10)
        assert len(outcome) == 1
        assert BUSY in str(outcome[0])
        app.flush()
        assert table_rows(con) == [(1, "a")]

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        ("settle", "expected"),
        [
            pytest.param(lambda app: app.flush(), [(1, "a")], id="retried"),
            pytest.param(lambda app: app.clear() or app.flush(), [], id="cleared"),
        ],
    )
    def test_an_interrupted_flush_keeps_the_rows(self, con, held, settle, expected):
        app = held_flush_appender(con)
        app.append_row(1, "a")
        interrupt_the_held_flush(app, con, held)
        settle(app)
        assert table_rows(con) == expected

    @pytest.mark.timeout(60)
    def test_an_interrupted_flush_inside_begin_keeps_the_rows_for_after_rollback(self, con, held):
        app = held_flush_appender(con)
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (5, 'in the transaction')")
        app.append_row(1, "a")
        interrupt_the_held_flush(app, con, held)
        con.execute("ROLLBACK")
        app.flush()
        assert table_rows(con.cursor()) == [(1, "a")]

    @pytest.mark.timeout(60)
    def test_a_retry_before_rollback_drops_the_rows_of_an_interrupted_flush(self, con, held):
        app = held_flush_appender(con)
        con.execute("BEGIN")
        app.append_row(1, "a")
        interrupt_the_held_flush(app, con, held)
        with pytest.raises(duckdb.TransactionException):
            app.flush()
        con.execute("ROLLBACK")
        app.flush()
        assert table_rows(con) == []

    def test_a_flush_interrupted_by_ctrl_c_keeps_the_rows(self):
        code, output = run_isolated(
            f"""
            import _thread
            import signal
            import threading
            import time
            import duckdb
            from duckdb.sqltypes import BIGINT
            # A background job starts with SIGINT ignored, which would turn the simulated Ctrl-C into nothing
            signal.signal(signal.SIGINT, signal.default_int_handler)
            con = duckdb.connect()
            con.execute("SET threads = 4")
            con.execute("CREATE TABLE t (k INTEGER, v VARCHAR)")
            con.execute("CREATE TABLE big AS SELECT range AS i FROM range(2000000)")
            # A builtin runs no bytecode, so the Ctrl-C reaches the client's signal check while the scan's
            # tasks are still in the UDF, instead of raising inside a UDF call and failing the flush
            calls = []
            con.create_function("record", calls.append, [BIGINT], BIGINT, null_handling="special", side_effects=True)
            app = con.query_appender(
                "INSERT INTO t FROM appender_data",
                {KV!r},
                source="SELECT k, v FROM appender_buffer, (SELECT count(record(i)) AS n FROM big) WHERE n >= 0",
            )
            app.append_row(1, "a")
            print("started", flush=True)

            # Waiting for the UDF to run, not for a fixed time, keeps a fast machine from finishing the scan first
            def interrupt_once_the_scan_runs():
                while len(calls) < 10000:
                    time.sleep(0.001)
                _thread.interrupt_main()

            threading.Thread(target=interrupt_once_the_scan_runs, daemon=True).start()
            try:
                app.flush()
            except (RuntimeError, KeyboardInterrupt) as error:
                print("interrupted:", type(error).__name__, flush=True)
            con.execute("DELETE FROM big")
            app.flush()
            print("kept:", con.execute("SELECT * FROM t").fetchall())
            """,
            # Ending the flush waits for its running scan tasks, so the child lasts about as long as the whole scan
            timeout=60,
        )
        assert code == 0, output
        assert "interrupted:" in output
        assert "kept: [(1, 'a')]" in output

    @pytest.mark.parametrize(
        "row",
        [pytest.param((1, "again"), id="primary_key"), pytest.param((None, "no key"), id="not_null")],
    )
    def test_a_failed_flush_drops_the_rows_and_raises(self, con, row):
        con.execute("INSERT INTO t VALUES (1, 'a')")
        app = con.appender("t")
        app.append_row(2, "b")
        app.append_row(*row)
        with pytest.raises(duckdb.ConstraintException):
            app.flush()
        app.append_row(3, "c")
        app.flush()
        assert table_rows(con) == [(1, "a"), (3, "c")]

    def test_a_failed_flush_inside_begin_aborts_the_transaction(self, con):
        con.execute("INSERT INTO t VALUES (1, 'a')")
        app = con.appender("t")
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (2, 'b')")
        app.append_row(1, "again")
        with pytest.raises(duckdb.ConstraintException):
            app.flush()
        with pytest.raises(duckdb.TransactionException):
            con.execute("SELECT 1")
        con.execute("ROLLBACK")
        assert table_rows(con) == [(1, "a")]

    def test_a_flush_inside_an_aborted_transaction_drops_the_rows(self, con):
        app = con.appender("t")
        con.execute("BEGIN")
        with pytest.raises(duckdb.InvalidInputException):
            con.execute("SELECT error('boom')")
        app.append_row(1, "a")
        with pytest.raises(duckdb.TransactionException):
            app.flush()
        con.execute("ROLLBACK")
        app.flush()
        assert table_rows(con) == []


class TestBuffer:
    @pytest.fixture
    def tight(self):
        connection = duckdb.connect()
        connection.execute("SET memory_limit = '16MB'")
        connection.execute("SET temp_directory = ''")
        connection.execute("CREATE TABLE s (k INTEGER, v VARCHAR)")
        yield connection
        connection.close()

    def test_a_failed_write_into_the_buffer_breaks_the_appender_until_clear(self, tight):
        app = tight.appender("s")
        # Rows reach the limited memory a whole chunk at a time, so one chunk must outgrow the limit.
        with pytest.raises(duckdb.OutOfMemoryException):
            append_until_full(app, (0, "x" * (16 << 10)), 4 * VECTOR)
        with pytest.raises(duckdb.InvalidInputException, match="clear"):
            app.append_row(1, "a")
        with pytest.raises(duckdb.InvalidInputException, match="clear"):
            app.append_rows([(1, "a")])
        with pytest.raises(duckdb.InvalidInputException, match="clear"):
            app.flush()
        app.clear()
        app.append_row(1, "a")
        app.flush()
        assert table_rows(tight, "s") == [(1, "a")]

    def test_a_buffer_larger_than_the_memory_limit_spills_and_arrives_whole(self, tmp_path):
        connection = duckdb.connect((tmp_path / "spill.db").as_posix())
        try:
            # The flush itself needs about 10 MiB beyond what it can evict, and more on macOS x86_64, where 16 MB ran
            # out of memory on CI. 32 MB leaves headroom while staying well below the buffer's 64 MiB.
            connection.execute("SET memory_limit = '32MB'")
            connection.execute(f"SET temp_directory = '{(tmp_path / 'spill').as_posix()}'")
            connection.execute("CREATE TABLE s (k INTEGER, v VARCHAR)")
            app = connection.appender("s")
            # A staged chunk of 8 MiB fits the limit and the 64 MiB buffer does not, so only spilling lets it arrive.
            rows, value = 8 * VECTOR, "x" * (4 << 10)
            app.append_rows((i, value) for i in range(rows))
            app.flush()
            expected = [(rows, rows * (rows - 1) // 2, rows * len(value))]
            assert connection.execute("SELECT count(*), sum(k), sum(length(v)) FROM s").fetchall() == expected
        finally:
            connection.close()


class TestNothingFlushesOnItsOwn:
    @pytest.mark.parametrize(
        "statement",
        [
            pytest.param(lambda c: c.execute("SELECT 1"), id="execute"),
            pytest.param(lambda c: c.execute("BEGIN; COMMIT"), id="transaction"),
            pytest.param(lambda c: c.begin().commit(), id="commit"),
            pytest.param(lambda c: c.execute("CHECKPOINT"), id="checkpoint"),
        ],
    )
    def test_other_statements_do_not_flush(self, con, statement):
        app = con.appender("t")
        app.append_row(1, "a")
        statement(con)
        assert table_rows(con) == []

    def test_many_rows_stay_buffered(self, con):
        con.execute("CREATE TABLE log (k INTEGER)")
        app = con.appender("log")
        app.append_rows((i,) for i in range(ENGINE_FLUSH_COUNT + 1))
        assert con.execute("SELECT count(*) FROM log").fetchall() == [(0,)]

    def test_garbage_collecting_buffered_rows_warns_and_drops_them(self, con):
        app = con.appender("t")
        app.append_row(1, "a")
        with pytest.warns(ResourceWarning):
            del app
        assert table_rows(con) == []

    def test_a_resource_warning_turned_into_an_error_goes_to_the_unraisable_hook(self, con, monkeypatch):
        app = con.appender("t")
        app.append_row(1, "a")
        unraisable = []
        # Keeping the hook's argument would keep its traceback, and with it this frame and the connection, alive.
        monkeypatch.setattr(sys, "unraisablehook", lambda hook: unraisable.append(hook.exc_type))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            del app
        assert unraisable == [ResourceWarning]
        assert table_rows(con) == []

    def test_garbage_collecting_buffered_rows_warns_after_the_connection_is_closed(self, con):
        app = con.appender("t")
        app.append_row(1, "a")
        con.close()
        with pytest.warns(ResourceWarning):
            del app

    @pytest.mark.parametrize(
        "prepare",
        [
            pytest.param(lambda app: app.append_row(1, "a") or app.flush(), id="flushed"),
            pytest.param(lambda app: app.append_row(1, "a") or app.clear(), id="cleared"),
            pytest.param(lambda app: None, id="never_appended"),
        ],
    )
    def test_garbage_collecting_an_appender_without_rows_is_silent(self, con, prepare):
        app = con.appender("t")
        prepare(app)
        gc.collect()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del app
            gc.collect()
        assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []


class TestWithBlock:
    def test_normal_exit_flushes(self, con):
        with con.appender("t") as app:
            app.append_row(1, "a")
        assert table_rows(con) == [(1, "a")]

    def test_a_failed_flush_on_normal_exit_propagates(self, con):
        con.execute("INSERT INTO t VALUES (1, 'a')")
        with pytest.raises(duckdb.ConstraintException), con.appender("t") as app:
            app.append_row(1, "again")

    def test_a_refused_flush_on_normal_exit_propagates_and_keeps_the_rows(self, con):
        app = con.appender("t")
        open_result(con)
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE), app:
            app.append_row(1, "a")
        con.fetchall()
        app.flush()
        assert table_rows(con) == [(1, "a")]

    @pytest.mark.parametrize("raised", [ValueError, KeyboardInterrupt])
    def test_exit_by_exception_clears_and_the_exception_propagates(self, con, raised):
        app = con.appender("t")
        with pytest.raises(raised), app:
            append_then_raise(app, raised)
        app.append_row(2, "b")
        app.flush()
        assert table_rows(con) == [(2, "b")]

    def test_exit_by_exception_after_the_connection_is_closed_propagates_the_exception(self, con):
        app = con.appender("t")
        with pytest.raises(ValueError, match="the block failed"), app:
            append_close_then_raise(app, con, ValueError("the block failed"))

    def test_the_appender_is_usable_across_with_blocks(self, con):
        app = con.appender("t")
        with app:
            app.append_row(1, "a")
        with app:
            app.append_row(2, "b")
        assert table_rows(con) == [(1, "a"), (2, "b")]


class TestOneCallerAtATime:
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda app: app.append_row(9, "z"), id="append_row"),
            pytest.param(lambda app: app.append_rows([(9, "z")]), id="append_rows"),
            pytest.param(lambda app: app.flush(), id="flush"),
            pytest.param(lambda app: app.clear(), id="clear"),
        ],
    )
    def test_another_thread_is_refused_during_a_flush_without_waiting(self, con, held, call):
        running, release = held
        app = con.query_appender(
            "INSERT INTO t FROM appender_data", KV, source="SELECT hold(k) AS k, v FROM appender_buffer"
        )
        app.append_row(1, "a")
        outcome = []
        thread = threading.Thread(target=contend, args=(running, release, lambda: call(app), outcome, CONCURRENT_USE))
        thread.start()
        app.flush()
        thread.join(10)
        assert [type(o) for o in outcome] == [CONCURRENT_USE]
        app.flush()
        assert table_rows(con) == [(1, "a")]

    @pytest.mark.parametrize("call", ["app.append_row(9, 'z')", "app.flush()", "app.clear()"])
    def test_a_callback_during_its_flush_is_refused(self, call):
        code, output = run_isolated(
            f"""
            import duckdb
            from duckdb.sqltypes import BIGINT
            con = duckdb.connect()
            con.execute("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
            def reenter(x):
                try:
                    {call}
                    print("ran", flush=True)
                except duckdb.Error as error:
                    print("refused:", type(error).__name__, flush=True)
                return x
            con.create_function("reenter", reenter, [BIGINT], BIGINT, side_effects=True)
            app = con.query_appender(
                "INSERT INTO t FROM appender_data", {KV!r}, source="SELECT reenter(k) AS k, v FROM appender_buffer"
            )
            app.append_row(1, "a")
            print("started", flush=True)
            app.flush()
            print("after:", con.execute("SELECT * FROM t").fetchall())
            """
        )
        assert code == 0, output
        assert f"refused: {CONCURRENT_USE.__name__}" in output
        assert "after: [(1, 'a')]" in output


class TestClosedConnection:
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda app: app.append_row(2, "b"), id="append_row"),
            pytest.param(lambda app: app.append_rows([(2, "b")]), id="append_rows"),
            pytest.param(lambda app: app.flush(), id="flush"),
            pytest.param(lambda app: app.clear(), id="clear"),
            pytest.param(lambda app: app.__enter__(), id="enter"),
        ],
    )
    def test_every_method_raises_once_the_connection_is_closed(self, con, call):
        app = con.appender("t")
        app.append_row(1, "a")
        con.close()
        with pytest.raises(CLOSED):
            call(app)

    def test_an_empty_flush_raises_once_the_connection_is_closed(self, con):
        app = con.appender("t")
        con.close()
        with pytest.raises(CLOSED):
            app.flush()

    @pytest.mark.parametrize(
        "options",
        [{}, {"action": "merge"}, {"action": "update"}],
        ids=["insert", "unknown_action", "update_without_key"],
    )
    def test_creating_an_appender_on_a_closed_connection_raises(self, con, options):
        con.close()
        with pytest.raises(CLOSED):
            con.appender("t", **options)

    def test_an_appender_on_a_cursor_raises_once_its_parent_is_closed(self, con):
        app = con.cursor().appender("t")
        con.close()
        with pytest.raises(CLOSED):
            app.append_row(1, "a")

    def test_reconnecting_to_the_file_works_while_an_appender_lives(self, tmp_path):
        code, output = run_isolated(
            f"""
            import duckdb
            path = {(tmp_path / "db.duckdb").as_posix()!r}
            con = duckdb.connect(path)
            con.execute("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
            app = con.appender("t")
            app.append_row(1, "a")
            con.close()
            print("started", flush=True)
            again = duckdb.connect(path)
            print("rows:", again.execute("SELECT * FROM t").fetchall())
            """
        )
        assert code == 0, output
        assert "rows: []" in output

    def test_a_transaction_open_at_close_is_released_while_an_appender_lives(self, tmp_path):
        code, output = run_isolated(
            f"""
            import duckdb
            path = {(tmp_path / "db.duckdb").as_posix()!r}
            con = duckdb.connect(path)
            other = duckdb.connect(path)
            con.execute("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
            app = con.appender("t")
            app.append_row(1, "a")
            con.execute("BEGIN")
            con.execute("INSERT INTO t VALUES (2, 'uncommitted')")
            con.close()
            print("started", flush=True)
            other.execute("INSERT INTO t VALUES (2, 'b')")
            print("rows:", other.execute("SELECT * FROM t ORDER BY ALL").fetchall())
            """
        )
        assert code == 0, output
        assert "rows: [(2, 'b')]" in output

    @pytest.mark.parametrize("call", ["app.append_row(1, 2)", "app.clear()"])
    def test_closing_the_connection_inside_a_row_raises_instead_of_crashing(self, call):
        code, output = run_isolated(
            f"""
            import duckdb
            con = duckdb.connect()
            con.execute("CREATE TABLE t (a BIGINT, b BIGINT)")
            app = con.appender("t")
            class Row:
                def __len__(self):
                    return 2
                def __getitem__(self, i):
                    if i == 1:
                        con.close()
                    if i >= 2:
                        raise IndexError
                    return 7
            print("started", flush=True)
            try:
                app.append_rows([Row()])
                print("append_rows: returned")
            except duckdb.Error as error:
                print("append_rows:", type(error).__name__)
            try:
                {call}
                print("next call: returned")
            except duckdb.Error as error:
                print("next call:", type(error).__name__)
            """
        )
        assert code == 0, output
        assert re.search(r"append_rows: (returned|ConnectionException)\n", output), output
        assert "next call: ConnectionException" in output

    def test_reconnecting_from_inside_the_rows_iterator_raises_at_the_next_row(self, tmp_path):
        code, output = run_isolated(
            f"""
            import duckdb
            path = {(tmp_path / "db.duckdb").as_posix()!r}
            con = duckdb.connect(path)
            con.execute("CREATE TABLE t (i INTEGER)")
            app = con.appender("t")
            def rows():
                yield (1,)
                con.close()
                duckdb.connect(path).close()
                yield (2,)
            print("started", flush=True)
            try:
                app.append_rows(rows())
                print("append_rows: returned")
            except duckdb.Error as error:
                print("append_rows:", type(error).__name__)
            print("rows:", duckdb.connect(path).execute("SELECT * FROM t").fetchall())
            """
        )
        assert code == 0, output
        assert "append_rows: ConnectionException" in output
        assert "rows: []" in output

    @pytest.mark.parametrize(
        "create",
        [
            pytest.param('con.appender("t", columns=[Column()])', id="table"),
            pytest.param('con.query_appender("INSERT INTO t FROM appender_data", {Column(): int})', id="query"),
        ],
    )
    def test_reconnecting_while_a_column_name_converts_raises_at_creation(self, tmp_path, create):
        code, output = run_isolated(
            f"""
            import duckdb
            path = {(tmp_path / "db.duckdb").as_posix()!r}
            con = duckdb.connect(path)
            con.execute("CREATE TABLE t (i INTEGER)")
            class Column:
                def __str__(self):
                    con.close()
                    duckdb.connect(path).close()
                    return "i"
            print("started", flush=True)
            try:
                {create}
                print("creation: returned")
            except duckdb.Error as error:
                print("creation:", type(error).__name__)
            print("rows:", duckdb.connect(path).execute("SELECT * FROM t").fetchall())
            """
        )
        assert code == 0, output
        assert "creation: ConnectionException" in output
        assert "rows: []" in output

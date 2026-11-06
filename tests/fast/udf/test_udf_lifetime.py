import ast
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import textwrap

import pytest

import duckdb

PRELUDE = """
import gc
import sys
import threading
import weakref

import duckdb

CALL = "SELECT myf('b')"


def native():
    return lambda x: f"got:{x}"


def other():
    return lambda x: f"other:{x}"


def native_int():
    return lambda x: x + 1


def three():
    return lambda a, b, c: f"three:{a}"


def raising():
    def f(x):
        if x == "boom":
            raise ValueError(x)
        return f"got:{x}"

    return f


def arrow():
    import pyarrow.compute as pc

    return lambda x: pc.binary_join_element_wise("got:", x, "")


def register(con, factory=native, params=("VARCHAR",), ret="VARCHAR", name="myf", **kwargs):
    con.create_function(name, factory(), list(params), ret, **kwargs)


def weakly_registered(con):
    f = native()
    ref = weakref.ref(f)
    con.create_function("myf", f, ["VARCHAR"], "VARCHAR")
    del f
    return ref


def alive(ref):
    gc.collect()
    return ref() is not None


def in_transaction(con, attempt):
    con.execute("CREATE TABLE t (i INTEGER)")
    con.begin()
    con.execute("INSERT INTO t VALUES (1)")
    step(attempt)
    step(con.commit)
    step(lambda: con.sql("SELECT count(*) FROM t").fetchall())


def step(fn):
    try:
        value = fn()
    except Exception as e:
        print("OUTCOME", repr(("raised", type(e).__name__, str(e).splitlines()[0])), flush=True)
    else:
        if isinstance(value, duckdb.DuckDBPyConnection):
            value = None
        print("OUTCOME", repr(("ok", value)), flush=True)
"""


def ok(value):
    return ("ok", value)


def raised(kind, message):
    return ("raised", kind, message)


def not_registered(name):
    return raised(
        "InvalidInputException",
        f"Invalid Input Error: No function by the name of '{name}' was found in the list of registered functions",
    )


def arrow_table_error(message):
    return raised(
        "InvalidInputException", f"Invalid Input Error: Could not convert the result into an Arrow Table: {message}"
    )


def udf_error(message):
    return raised(
        "InvalidInputException", f"Invalid Input Error: Python exception occurred while executing the UDF: {message}"
    )


GOT_B = ok([("got:b",)])
# A Python error raised converting the function's arguments or its value is not the function's own error, so it leaves
# as the engine's error carrying the Python traceback, in every exception handling mode
RAISED_BY_THE_CONVERSION = raised("Error", "Traceback (most recent call last):")
ZONE_UNKNOWN_TO_PYTHON = raised(
    "InvalidInputException",
    "Invalid Input Error: Time zone 'ACT' is not available to Python's zoneinfo. Install the 'tzdata' package, or SET "
    "TimeZone to a name the IANA database has.",
)
MISSING = raised("CatalogException", "Catalog Error: Scalar Function with name myf does not exist!")
CURSOR_CLOSED = raised("ConnectionException", "Connection Error: Connection already closed!")
ALREADY_EXISTS = raised("CatalogException", 'Catalog Error: Scalar Function with name "myf" already exists!')
BUILT_IN_TAKEN = raised("CatalogException", 'Catalog Error: Scalar Function with name "lower" already exists!')

NEEDS_PYARROW = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")


def case(name, script, expected, *marks):
    return pytest.param(script, expected, id=name, marks=marks)


def outcomes_of(stdout):
    # A query that runs long writes the engine's progress bar to stdout, without a newline, ahead of an outcome.
    prefix = "OUTCOME "
    return [ast.literal_eval(line.split(prefix, 1)[1]) for line in stdout.splitlines() if prefix in line]


def run(script, tmp_path, timeout=60):
    # A call through a released callable can crash the interpreter, so such a case runs in a child; -X dev fills
    # freed memory, so such a call never happens to find the old callable intact.
    proc = subprocess.run(
        [
            sys.executable,
            "-X",
            "dev",
            "-c",
            PRELUDE + "DB = sys.argv[1]\n" + textwrap.dedent(script),
            str(tmp_path / "db.duckdb"),
        ],
        capture_output=True,
        # The engine's progress bar writes UTF-8, which the locale's encoding may not decode
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        timeout=timeout,
        check=False,
    )
    return proc.returncode, outcomes_of(proc.stdout), proc.stderr


def check(script, expected, tmp_path):
    returncode, outcomes, stderr = run(script, tmp_path)
    assert (returncode, outcomes) == (0, expected), stderr[-3000:]


def check_in_process(script, expected, tmp_path):
    namespace = {"DB": str(tmp_path / "db.duckdb")}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(PRELUDE + textwrap.dedent(script), namespace)
    finally:
        # The callables refer to this namespace, which holds the connections, so only closing them frees a database
        for value in namespace.values():
            if isinstance(value, duckdb.DuckDBPyConnection):
                value.close()
    assert outcomes_of(stdout.getvalue()) == expected


# A callable made by a factory has the registration as its only reference.
OWNS_ITS_CALLABLE = [
    case(
        "a_cursor_function_is_visible_on_its_parent",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        step(lambda: con.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "a_function_is_visible_on_a_second_connection_to_the_same_file",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        register(con1)
        step(lambda: con2.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "sql_cannot_drop_a_python_function",
        """
        con = duckdb.connect()
        register(con)
        step(lambda: con.execute("DROP FUNCTION myf").fetchall())
        """,
        [raised("CatalogException", 'Catalog Error: Cannot drop internal catalog entry "myf"!')],
    ),
    case(
        "a_relation_of_a_closed_cursor_cannot_run",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        rel = cur.sql(CALL)
        cur.close()
        step(rel.fetchall)
        """,
        [raised("ConnectionException", "Connection Error: Connection has already been closed")],
    ),
    case(
        "closing_the_parent_closes_its_cursors",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(con)
        con.close()
        step(lambda: cur.sql(CALL).fetchall())
        """,
        [raised("ConnectionException", "Connection Error: Connection has already been closed")],
    ),
    case(
        "a_refused_close_of_the_parent_leaves_its_cursors_to_the_next_close",
        """
        con = duckdb.connect()
        cur1 = con.cursor()
        cur2 = con.cursor()


        def closes_the_parent(x):
            con.close()
            return x


        cur1.create_function("closes", closes_the_parent, ["VARCHAR"], "VARCHAR")
        step(lambda: cur1.sql("SELECT closes('b')").fetchall())
        step(con.close)
        step(lambda: cur1.sql("SELECT 1").fetchall())
        step(lambda: cur2.sql("SELECT 1").fetchall())
        """,
        [
            udf_error(
                "ResourceInUseException: Resource In Use Error: code running inside a query cannot use that "
                "query's connection; use a separate connection from cursor()"
            ),
            ok(None),
            raised("ConnectionException", "Connection Error: Connection has already been closed"),
            raised("ConnectionException", "Connection Error: Connection has already been closed"),
        ],
    ),
    case(
        "the_parent_calls_after_the_cursor_is_deleted",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        del cur
        step(lambda: con.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "the_parent_calls_an_arrow_function_after_the_cursor_is_deleted",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur, arrow, type="arrow")
        del cur
        step(lambda: con.sql(CALL).fetchall())
        """,
        [GOT_B],
        NEEDS_PYARROW,
    ),
    case(
        "the_parent_calls_a_function_returning_null_on_exceptions_after_the_cursor_is_deleted",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur, raising, exception_handling="return_null")
        del cur
        step(lambda: con.sql("SELECT myf('b'), myf('boom')").fetchall())
        """,
        [ok([("got:b", None)])],
    ),
    case(
        "the_parent_calls_after_the_cursor_is_closed",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        cur.close()
        step(lambda: con.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "an_arrow_function_runs_after_its_cursor_is_closed",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur, arrow, type="arrow")
        cur.close()
        step(lambda: con.sql(CALL).fetchall())
        """,
        [GOT_B],
        NEEDS_PYARROW,
    ),
    case(
        "a_sibling_cursor_calls_after_the_cursor_is_closed",
        """
        con = duckdb.connect()
        cur = con.cursor()
        sibling = con.cursor()
        register(cur)
        cur.close()
        step(lambda: sibling.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "another_thread_calls_after_the_cursor_is_deleted",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        del cur
        t = threading.Thread(target=lambda: step(lambda: con.cursor().sql(CALL).fetchall()))
        t.start()
        t.join()
        """,
        [GOT_B],
    ),
    case(
        "the_parent_calls_after_the_cursor_is_closed_while_the_user_keeps_the_callable",
        """
        con = duckdb.connect()
        cur = con.cursor()
        f = native()
        cur.create_function("myf", f, ["VARCHAR"], "VARCHAR")
        cur.close()
        step(lambda: con.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "the_function_is_listed_after_the_cursor_is_closed",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        listed = "SELECT count(*) FROM duckdb_functions() WHERE function_name = 'myf'"
        step(lambda: con.sql(listed).fetchall())
        cur.close()
        step(lambda: con.sql(listed).fetchall())
        """,
        [ok([(1,)]), ok([(1,)])],
    ),
    case(
        "a_second_connection_calls_after_the_registering_connection_is_closed",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        register(con1)
        con1.close()
        step(lambda: con2.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "a_second_connection_calls_after_the_registering_connection_is_destroyed",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        register(con1)
        del con1
        step(lambda: con2.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "a_second_connection_calls_after_the_registering_cursor_is_closed",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        cur = con1.cursor()
        register(cur)
        cur.close()
        step(lambda: con2.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "a_second_connection_calls_after_the_registering_connection_is_closed_while_the_user_keeps_the_callable",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        f = native()
        con1.create_function("myf", f, ["VARCHAR"], "VARCHAR")
        con1.close()
        step(lambda: con2.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "a_relation_rebinds_after_the_cursor_is_closed",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        rel = con.sql(CALL)
        cur.close()
        step(rel.fetchall)
        """,
        [GOT_B],
    ),
    case(
        "a_prepared_statement_over_a_table_rebinds_after_the_cursor_is_closed",
        """
        con = duckdb.connect()
        con.execute("CREATE TABLE t AS SELECT 'b' AS s")
        cur = con.cursor()
        register(cur)
        con.execute("PREPARE p AS SELECT myf(s) FROM t")
        cur.close()
        step(lambda: con.execute("EXECUTE p").fetchall())
        """,
        [GOT_B],
    ),
    case(
        "a_prepared_statement_without_a_table_still_runs_after_its_cursor_closes",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        con.execute("PREPARE p AS SELECT myf($1)")
        cur.close()
        step(lambda: con.execute("EXECUTE p('b')").fetchall())
        """,
        [GOT_B],
    ),
    case(
        "an_open_result_finishes_after_the_registering_cursor_closes",
        """
        con = duckdb.connect()
        # One thread computes rows only while fetching, and a small buffer leaves most of them for after the close.
        con.execute("SET threads = 1")
        con.execute("SET streaming_buffer_size = '64KB'")
        cur = con.cursor()
        register(cur)
        res = con.execute("SELECT myf(range::VARCHAR) FROM range(10000)")
        step(res.fetchone)
        cur.close()
        step(lambda: len(res.fetchall()))
        """,
        [ok(("got:0",)), ok(9999)],
    ),
    case(
        "a_running_query_finishes_when_another_thread_closes_the_registering_cursor",
        """
        con = duckdb.connect()
        cur = con.cursor()
        entered = threading.Event()
        closed = threading.Event()


        def waits_for_the_close():
            def f(x):
                if not entered.is_set():
                    entered.set()
                    assert closed.wait(10)
                return f"got:{x}"

            return f


        register(cur, waits_for_the_close)
        closer_errors = []


        def closer():
            try:
                assert entered.wait(10)
                cur.close()
            except Exception as error:
                closer_errors.append(repr(error))
            closed.set()


        t = threading.Thread(target=closer)
        t.start()
        step(lambda: len(con.sql("SELECT myf(range::VARCHAR) FROM range(3)").fetchall()))
        t.join()
        step(lambda: closer_errors)
        """,
        [ok(3), ok([])],
    ),
    case(
        "an_open_result_survives_destroying_its_connection",
        """
        import time

        con = duckdb.connect()
        entered = threading.Event()


        def pauses_in_a_later_chunk():
            def f(x):
                if x == "10000":
                    entered.set()
                    time.sleep(1)
                return f"got:{x}"

            return f


        register(con, pauses_in_a_later_chunk)
        res = con.execute("SELECT myf(range::VARCHAR) FROM range(100000)")
        step(res.fetchone)
        step(lambda: entered.wait(10))
        del con, res
        gc.collect()
        step(lambda: "survived")
        """,
        [ok(("got:0",)), ok(True), ok("survived")],
    ),
    case(
        "an_arrow_reader_over_a_native_function_reads_all_rows_after_its_connection_closes",
        """
        con = duckdb.connect()
        ref = weakly_registered(con)
        reader = con.sql("SELECT myf(range::VARCHAR) AS s FROM range(5000)").to_arrow_reader()
        con.close()
        step(lambda: alive(ref))
        step(lambda: sum(batch.num_rows for batch in reader))
        del reader
        step(lambda: alive(ref))
        """,
        [ok(True), ok(5000), ok(False)],
        NEEDS_PYARROW,
    ),
    case(
        "an_arrow_reader_over_an_arrow_function_reads_all_rows_after_its_connection_closes",
        """
        con = duckdb.connect()
        f = arrow()
        ref = weakref.ref(f)
        con.create_function("myf", f, ["VARCHAR"], "VARCHAR", type="arrow")
        del f
        reader = con.sql("SELECT myf(range::VARCHAR) AS s FROM range(5000)").to_arrow_reader()
        con.close()
        step(lambda: alive(ref))
        step(lambda: sum(batch.num_rows for batch in reader))
        del reader
        step(lambda: alive(ref))
        """,
        [ok(True), ok(5000), ok(False)],
        NEEDS_PYARROW,
    ),
    case(
        "an_open_result_survives_closing_its_connection",
        """
        con = duckdb.connect()
        register(con)
        res = con.execute("SELECT myf(range::VARCHAR) FROM range(100000)")
        step(res.fetchone)
        con.close()
        gc.collect()
        step(lambda: "survived")
        """,
        [ok(("got:0",)), ok("survived")],
    ),
    case(
        "a_function_converts_its_arguments_with_the_settings_of_the_calling_connection",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        con1.execute("SET TimeZone = 'America/New_York'")
        con2.execute("SET TimeZone = 'Asia/Tokyo'")
        con1.create_function("utc_offset", lambda ts: str(ts.utcoffset()), ["TIMESTAMPTZ"], "VARCHAR")
        call = "SELECT utc_offset(TIMESTAMPTZ '2020-01-01 00:00:00+00')"
        step(lambda: con1.sql(call).fetchall())
        step(lambda: con2.sql(call).fetchall())
        con1.close()
        step(lambda: con2.sql(call).fetchall())
        """,
        [ok([("-1 day, 19:00:00",)]), ok([("9:00:00",)]), ok([("9:00:00",)])],
    ),
    case(
        "a_function_converts_its_arguments_with_the_settings_its_connection_has_at_the_call",
        """
        con = duckdb.connect()
        con.execute("SET TimeZone = 'America/New_York'")
        con.create_function("utc_offset", lambda ts: str(ts.utcoffset()), ["TIMESTAMPTZ"], "VARCHAR")
        call = "SELECT utc_offset(TIMESTAMPTZ '2020-01-01 00:00:00+00')"
        step(lambda: con.sql(call).fetchall())
        con.execute("SET TimeZone = 'Asia/Tokyo'")
        step(lambda: con.sql(call).fetchall())
        """,
        [ok([("-1 day, 19:00:00",)]), ok([("9:00:00",)])],
    ),
    case(
        "a_function_in_an_index_expression_takes_a_null_row_without_a_client_context",
        """
        con = duckdb.connect()
        con.create_function("f", lambda x: x, ["INTEGER"], "INTEGER")
        con.execute("CREATE TABLE t (x INTEGER)")
        con.execute("CREATE INDEX idx ON t (f(x))")
        step(lambda: con.execute("INSERT INTO t VALUES (NULL)").fetchall())
        step(lambda: con.execute("INSERT INTO t VALUES (1)").fetchall())
        step(lambda: con.sql("SELECT count(*) FROM t").fetchall())
        """,
        [
            ok([(1,)]),
            raised(
                "TransactionException", 'TransactionContext Error: Failed to commit: Cannot use "f" in this context'
            ),
            ok([(1,)]),
        ],
    ),
    case(
        "a_close_while_a_registration_converts_its_types_closes_and_the_registration_fails_as_closed",
        """
        con = duckdb.connect()
        converting = threading.Event()
        closed = threading.Event()
        outcomes = []


        def slow_type():
            converting.set()
            # The registration converts its types outside the connection's lock, so the close finishes meanwhile
            assert closed.wait(10)
            return int


        def f(x: "slow_type()") -> int:
            return x


        def registers():
            try:
                con.create_function("myf", f, ["BIGINT"], "BIGINT")
                outcomes.append("returned")
            except Exception as e:
                outcomes.append(type(e).__name__)


        registration = threading.Thread(target=registers)
        registration.start()
        step(lambda: converting.wait(10))
        step(con.close)
        closed.set()
        registration.join(20)
        step(lambda: outcomes)
        step(lambda: con.sql("SELECT myf(41)").fetchall())
        """,
        [
            ok(True),
            ok(None),
            ok(["ConnectionException"]),
            raised("ConnectionException", "Connection Error: Connection has already been closed"),
        ],
    ),
    case(
        "the_callable_of_a_cursor_function_after_the_cursor_and_then_the_database_close",
        """
        con = duckdb.connect()
        cur = con.cursor()
        ref = weakly_registered(cur)
        step(lambda: alive(ref))
        cur.close()
        step(lambda: alive(ref))
        con.close()
        step(lambda: alive(ref))
        """,
        [ok(True), ok(True), ok(False)],
    ),
    case(
        "the_callable_after_the_registering_and_then_the_last_connection_to_the_file_close",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        ref = weakly_registered(con1)
        con1.close()
        step(lambda: alive(ref))
        con2.close()
        step(lambda: alive(ref))
        """,
        [ok(True), ok(False)],
    ),
    case(
        "a_removed_callable_is_finalized_once_its_function_is_gone",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        finalized = []


        class Callable:
            def __call__(self, x):
                return f"got:{x}"

            def __del__(self):
                finalized.append(True)


        con1.create_function("myf", Callable(), ["VARCHAR"], "VARCHAR")
        step(lambda: con2.sql(CALL).fetchall())
        con2.remove_function("myf")
        gc.collect()
        step(lambda: list(finalized))
        register(con1, other)
        gc.collect()
        step(lambda: list(finalized))
        con1.close()
        gc.collect()
        step(lambda: list(finalized))
        con2.close()
        gc.collect()
        step(lambda: list(finalized))
        """,
        [GOT_B, ok([True]), ok([True]), ok([True]), ok([True])],
    ),
    case(
        "exit_with_a_function_on_the_default_connection",
        """
        duckdb.create_function("myf", native(), [duckdb.sqltypes.VARCHAR], duckdb.sqltypes.VARCHAR)
        step(lambda: duckdb.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "exit_with_a_function_on_a_cursor_of_the_default_connection",
        """
        cur = duckdb.cursor()
        register(cur)
        step(lambda: duckdb.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "exit_after_the_registering_cursor_of_the_default_connection_closed",
        """
        cur = duckdb.cursor()
        register(cur)
        step(cur.close)
        """,
        [ok(None)],
    ),
    case(
        "exit_with_a_file_database_left_open",
        """
        con = duckdb.connect(DB)
        register(con)
        step(lambda: con.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "exit_with_two_connections_to_a_file_database",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        register(con1)
        step(lambda: con2.sql(CALL).fetchall())
        """,
        [GOT_B],
    ),
    case(
        "exit_with_a_prepared_statement_left",
        """
        con = duckdb.connect(DB)
        register(con)
        con.execute("PREPARE p AS SELECT myf($1)")
        step(lambda: con.execute("EXECUTE p('b')").fetchall())
        """,
        [GOT_B],
    ),
    case(
        "exit_with_an_open_result",
        """
        con = duckdb.connect(DB)
        register(con)
        res = con.execute("SELECT myf(range::VARCHAR) FROM range(100000)")
        step(res.fetchone)
        """,
        [ok(("got:0",))],
    ),
]

REMOVES_FOR_EVERY_CONNECTION = [
    case(
        "the_registering_cursor_removes_the_function_for_every_connection",
        """
        con = duckdb.connect()
        cur = con.cursor()
        sibling = con.cursor()
        register(cur)
        step(lambda: cur.remove_function("myf"))
        step(lambda: con.sql(CALL).fetchall())
        step(lambda: sibling.sql(CALL).fetchall())
        step(lambda: cur.sql(CALL).fetchall())
        """,
        [ok(None), MISSING, MISSING, MISSING],
    ),
    case(
        "the_parent_removes_a_cursor_function",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        step(lambda: con.remove_function("myf"))
        step(lambda: con.sql(CALL).fetchall())
        step(lambda: cur.sql(CALL).fetchall())
        step(lambda: cur.remove_function("myf"))
        """,
        [ok(None), MISSING, MISSING, not_registered("myf")],
    ),
    case(
        "a_cursor_removes_the_function_of_its_parent",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(con)
        step(lambda: cur.remove_function("myf"))
        step(lambda: con.sql(CALL).fetchall())
        step(lambda: cur.sql(CALL).fetchall())
        """,
        [ok(None), MISSING, MISSING],
    ),
    case(
        "a_second_connection_to_the_file_removes_the_function",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        register(con1)
        step(lambda: con2.remove_function("myf"))
        step(lambda: con1.sql(CALL).fetchall())
        step(lambda: con2.sql(CALL).fetchall())
        """,
        [ok(None), MISSING, MISSING],
    ),
    case(
        "the_parent_removes_the_function_of_a_closed_cursor",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        cur.close()
        step(lambda: con.remove_function("myf"))
        step(lambda: con.sql(CALL).fetchall())
        """,
        [ok(None), MISSING],
    ),
    case(
        "a_second_connection_removes_the_function_after_the_registering_connection_is_destroyed",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        register(con1)
        del con1
        step(lambda: con2.remove_function("myf"))
        step(lambda: con2.sql(CALL).fetchall())
        """,
        [ok(None), MISSING],
    ),
    case(
        "remove_function_on_a_closed_connection_is_refused_as_closed",
        """
        con = duckdb.connect()
        con.close()
        step(lambda: con.remove_function("missing"))
        """,
        [raised("ConnectionException", "Connection Error: Connection already closed!")],
    ),
    case(
        "remove_function_refuses_built_in_and_unknown_names",
        """
        con = duckdb.connect()
        step(lambda: con.remove_function("lower"))
        step(lambda: con.remove_function("sum"))
        step(lambda: con.remove_function("not_a_function"))
        step(lambda: con.sql("SELECT lower('A')").fetchall())
        """,
        [not_registered("lower"), not_registered("sum"), not_registered("not_a_function"), ok([("a",)])],
    ),
    case(
        "a_refused_remove_function_leaves_the_transaction_usable",
        """
        con = duckdb.connect()
        in_transaction(con, lambda: con.remove_function("not_a_function"))
        """,
        [not_registered("not_a_function"), ok(None), ok([(1,)])],
    ),
    case(
        "remove_function_loads_no_extension_for_the_name",
        """
        con = duckdb.connect()
        con.execute(f"SET extension_directory = '{DB}.extensions'")
        con.execute("SET autoinstall_known_extensions = false")
        in_transaction(con, lambda: con.remove_function("host"))
        """,
        [not_registered("host"), ok(None), ok([(1,)])],
    ),
    case(
        "a_removed_callable_stays_alive_until_a_transaction_that_used_it_ends",
        """
        con = duckdb.connect()
        other_con = con.cursor()
        ref = weakly_registered(con)
        other_con.begin()
        step(lambda: other_con.sql(CALL).fetchall())
        step(lambda: con.remove_function("myf"))
        step(lambda: alive(ref))
        other_con.commit()
        con.execute("SELECT 1")
        step(lambda: alive(ref))
        """,
        [GOT_B, ok(None), ok(True), ok(False)],
    ),
    case(
        "a_relation_rebinds_after_remove_function",
        """
        con = duckdb.connect()
        register(con)
        rel = con.sql(CALL)
        con.remove_function("myf")
        step(rel.fetchall)
        """,
        [MISSING],
    ),
    case(
        "a_prepared_statement_over_a_table_rebinds_after_remove_function",
        """
        con = duckdb.connect()
        con.execute("CREATE TABLE t AS SELECT 'b' AS s")
        register(con)
        con.execute("PREPARE p AS SELECT myf(s) FROM t")
        con.remove_function("myf")
        step(lambda: con.execute("EXECUTE p").fetchall())
        """,
        [MISSING],
    ),
    case(
        "a_prepared_statement_without_a_table_still_runs_after_remove_function",
        """
        con = duckdb.connect()
        register(con)
        con.execute("PREPARE p AS SELECT myf($1)")
        con.remove_function("myf")
        step(lambda: con.execute("EXECUTE p('b')").fetchall())
        """,
        [GOT_B],
    ),
    case(
        "a_prepared_statement_still_runs_after_another_connection_removes_the_function",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        con.execute("PREPARE p AS SELECT myf($1)")
        step(lambda: con.remove_function("myf"))
        step(lambda: con.execute("EXECUTE p('b')").fetchall())
        step(lambda: con.sql(CALL).fetchall())
        """,
        [ok(None), GOT_B, MISSING],
    ),
    case(
        "remove_function_releases_the_callable",
        """
        con = duckdb.connect()
        ref = weakly_registered(con)
        step(lambda: con.sql(CALL).fetchall())
        con.remove_function("myf")
        step(lambda: alive(ref))
        """,
        [GOT_B, ok(False)],
    ),
    case(
        "removal_by_another_connection_releases_the_callable",
        """
        con = duckdb.connect()
        cur = con.cursor()
        ref = weakly_registered(cur)
        step(lambda: con.remove_function("myf"))
        step(lambda: alive(ref))
        """,
        [ok(None), ok(False)],
    ),
    case(
        "a_callable_replaced_under_its_name_is_released",
        """
        con = duckdb.connect()
        ref = weakly_registered(con)
        con.remove_function("myf")
        register(con, other)
        step(lambda: con.sql(CALL).fetchall())
        step(lambda: alive(ref))
        """,
        [ok([("other:b",)]), ok(False)],
    ),
    case(
        "a_removed_callable_stays_alive_until_its_prepared_statement_is_deallocated",
        """
        con = duckdb.connect()
        ref = weakly_registered(con)
        con.execute("PREPARE p AS SELECT myf($1)")
        con.remove_function("myf")
        step(lambda: alive(ref))
        step(lambda: con.execute("EXECUTE p('b')").fetchall())
        con.execute("DEALLOCATE p")
        # Released at the start of the next statement on the database, outside the one that dropped the last copy
        con.execute("SELECT 1")
        step(lambda: alive(ref))
        """,
        [ok(True), GOT_B, ok(False)],
    ),
    case(
        "a_removed_callable_stays_alive_while_a_reader_still_runs_it",
        """
        con = duckdb.connect()
        cur = con.cursor()
        ref = weakly_registered(con)
        reader = con.sql("SELECT myf(range::VARCHAR) AS s FROM range(5000)").to_arrow_reader(100)
        first = reader.read_next_batch()
        cur.remove_function("myf")
        step(lambda: alive(ref))
        step(lambda: first.num_rows + sum(batch.num_rows for batch in reader))
        del reader, first
        # Released at the next statement on the database, the first point outside engine locks after the reader goes
        con.sql("SELECT 1").fetchall()
        step(lambda: alive(ref))
        """,
        [ok(True), ok(5000), ok(False)],
        NEEDS_PYARROW,
    ),
    case(
        "a_removed_callable_stays_alive_while_an_index_expression_holds_its_function",
        """
        con = duckdb.connect()
        ref = weakly_registered(con)
        con.execute("CREATE TABLE t (x VARCHAR)")
        con.execute("INSERT INTO t VALUES ('1')")
        con.execute("CREATE INDEX i ON t (myf(x))")
        con.remove_function("myf")
        con.execute("SELECT 1")
        step(lambda: alive(ref))
        step(lambda: con.execute("INSERT INTO t VALUES (NULL)").fetchall())
        con.execute("DROP INDEX i")
        con.execute("SELECT 1")
        step(lambda: alive(ref))
        """,
        [ok(True), ok([(1,)]), ok(False)],
    ),
    case(
        "a_removed_callable_stays_alive_while_another_connection_still_runs_it",
        """
        con = duckdb.connect()
        cur = con.cursor()
        ref = weakly_registered(con)
        res = con.execute("SELECT myf(range::VARCHAR) FROM range(5000)")
        step(res.fetchone)
        step(lambda: cur.remove_function("myf"))
        step(lambda: alive(ref))
        step(lambda: len(res.fetchall()))
        con.execute("SELECT 1")
        step(lambda: alive(ref))
        """,
        [ok(("got:0",)), ok(None), ok(True), ok(4999), ok(False)],
    ),
    case(
        "a_rolled_back_remove_function_restores_the_function",
        """
        con = duckdb.connect()
        register(con)
        con.begin()
        con.remove_function("myf")
        con.rollback()
        step(lambda: con.sql(CALL).fetchall())
        step(lambda: con.remove_function("myf"))
        step(lambda: con.sql(CALL).fetchall())
        """,
        [GOT_B, ok(None), MISSING],
    ),
]

# create_function adds an overload to an existing scalar function, from any connection, and refuses a signature it has.
ADDS_AN_OVERLOAD = [
    case(
        "a_built_in_keeps_its_description_and_alias_through_a_python_overload_and_its_removal",
        """
        con = duckdb.connect()
        DESCRIBE = "SELECT description, examples, alias_of FROM duckdb_functions() WHERE function_name = 'split'"
        before = con.sql(DESCRIBE).fetchall()
        step(lambda: before)
        register(con, name="split")
        con.remove_function("split")
        step(lambda: con.sql(DESCRIBE).fetchall() == before)
        """,
        [
            ok([("Splits the `string` along the `separator`.", ["split('hello-world', '-')"], "string_split")]),
            ok(True),
        ],
    ),
    case(
        "the_same_connection_adds_an_overload",
        """
        con = duckdb.connect()
        register(con)
        step(lambda: register(con, native_int, params=("BIGINT",), ret="BIGINT"))
        step(lambda: con.sql("SELECT myf('b'), myf(41)").fetchall())
        """,
        [ok(None), ok([("got:b", 42)])],
    ),
    case(
        "the_same_connection_refuses_a_signature_it_has",
        """
        con = duckdb.connect()
        register(con)
        step(lambda: register(con, other))
        step(lambda: con.sql(CALL).fetchall())
        """,
        [ALREADY_EXISTS, GOT_B],
    ),
    case(
        "the_parent_adds_an_overload_to_a_cursor_function",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        step(lambda: register(con, native_int, params=("BIGINT",), ret="BIGINT"))
        cur.close()
        step(lambda: con.sql("SELECT myf('b'), myf(41)").fetchall())
        """,
        [ok(None), ok([("got:b", 42)])],
    ),
    case(
        "the_parent_refuses_a_signature_a_cursor_function_has",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        step(lambda: register(con, other))
        step(lambda: con.sql(CALL).fetchall())
        """,
        [ALREADY_EXISTS, GOT_B],
    ),
    case(
        "the_parent_refuses_a_signature_of_a_closed_cursor_function",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        cur.close()
        step(lambda: register(con, other))
        step(lambda: con.sql(CALL).fetchall())
        """,
        [ALREADY_EXISTS, GOT_B],
    ),
    case(
        "a_second_connection_to_the_file_adds_an_overload",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        register(con1)
        step(lambda: register(con2, native_int, params=("BIGINT",), ret="BIGINT"))
        step(lambda: con1.sql("SELECT myf('b'), myf(41)").fetchall())
        """,
        [ok(None), ok([("got:b", 42)])],
    ),
    case(
        "a_second_connection_to_the_file_refuses_a_signature_the_function_has",
        """
        con1 = duckdb.connect(DB)
        con2 = duckdb.connect(DB)
        register(con1)
        step(lambda: register(con2, other))
        step(lambda: con1.sql(CALL).fetchall())
        """,
        [ALREADY_EXISTS, GOT_B],
    ),
    case(
        "a_built_in_name_gains_an_overload",
        """
        con = duckdb.connect()
        cur = con.cursor()
        step(lambda: register(con, three, params=("BOOLEAN", "BOOLEAN", "BOOLEAN"), name="lower"))
        step(lambda: cur.sql("SELECT lower('A'), lower(true, true, true)").fetchall())
        """,
        [ok(None), ok([("a", "three:True")])],
    ),
    case(
        "a_built_in_signature_is_refused",
        """
        con = duckdb.connect()
        step(lambda: register(con, other, name="lower"))
        step(lambda: con.sql("SELECT lower('A')").fetchall())
        """,
        [BUILT_IN_TAKEN, ok([("a",)])],
    ),
    case(
        "removing_the_python_overload_of_a_built_in_name_keeps_the_built_in",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(con, three, params=("BOOLEAN", "BOOLEAN", "BOOLEAN"), name="LOWER")
        step(lambda: cur.sql("SELECT lower('A'), lower(true, true, true)").fetchall())
        step(lambda: con.remove_function("Lower"))
        step(lambda: cur.sql("SELECT lower('A')").fetchall())
        step(lambda: cur.sql("SELECT lower(true, true, true)").fetchall())
        step(lambda: con.remove_function("lower"))
        register(con, three, params=("BOOLEAN", "BOOLEAN", "BOOLEAN"), name="LOWER")
        step(lambda: cur.sql("SELECT lower('A'), lower(true, true, true)").fetchall())
        ENTRY = "SELECT DISTINCT function_name FROM duckdb_functions() WHERE lower(function_name) = 'lower'"
        step(lambda: cur.sql(ENTRY).fetchall())
        """,
        [
            ok([("a", "three:True")]),
            ok(None),
            ok([("a",)]),
            raised(
                "BinderException",
                "Binder Error: No function matches the given name and argument types 'lower(BOOLEAN, BOOLEAN, "
                "BOOLEAN)'. You might need to add explicit type casts.",
            ),
            not_registered("lower"),
            ok([("a", "three:True")]),
            ok([("lower",)]),
        ],
    ),
    case(
        "removing_a_function_of_python_overloads_from_several_connections_frees_its_name",
        """
        con = duckdb.connect()
        cur1 = con.cursor()
        cur2 = con.cursor()
        register(cur1)
        register(cur2, native_int, params=("BIGINT",), ret="BIGINT")
        step(lambda: con.remove_function("myf"))
        step(lambda: con.sql(CALL).fetchall())
        step(lambda: register(con, other))
        step(lambda: cur1.sql(CALL).fetchall())
        """,
        [ok(None), MISSING, ok(None), ok([("other:b",)])],
    ),
    case(
        "the_name_is_free_again_after_another_connection_removes_it",
        """
        con = duckdb.connect()
        cur = con.cursor()
        register(cur)
        step(lambda: con.remove_function("myf"))
        step(lambda: register(con, other))
        step(lambda: cur.sql(CALL).fetchall())
        """,
        [ok(None), ok(None), ok([("other:b",)])],
    ),
    case(
        "an_aggregate_name_is_refused",
        """
        con = duckdb.connect()
        step(lambda: register(con, other, name="sum"))
        step(lambda: con.sql("SELECT sum(i) FROM range(3) t(i)").fetchall())
        """,
        [raised("CatalogException", "Catalog Error: sum is not an scalar function"), ok([(3,)])],
    ),
    case(
        "a_new_name_inside_a_transaction_keeps_its_writes",
        """
        con = duckdb.connect()
        in_transaction(con, lambda: register(con))
        step(lambda: con.sql(CALL).fetchall())
        """,
        [ok(None), ok(None), ok([(1,)]), GOT_B],
    ),
    case(
        "an_overload_inside_a_transaction_keeps_its_writes",
        """
        con = duckdb.connect()
        register(con)
        in_transaction(con, lambda: register(con, native_int, params=("BIGINT",), ret="BIGINT"))
        step(lambda: con.sql("SELECT myf('b'), myf(41)").fetchall())
        """,
        [ok(None), ok(None), ok([(1,)]), ok([("got:b", 42)])],
    ),
    case(
        "a_refused_signature_of_the_same_connection_leaves_the_transaction_usable",
        """
        con = duckdb.connect()
        register(con)
        in_transaction(con, lambda: register(con, other))
        """,
        [ALREADY_EXISTS, ok(None), ok([(1,)])],
    ),
    case(
        "a_refused_signature_of_another_connection_leaves_the_transaction_usable",
        """
        con = duckdb.connect()
        register(con.cursor())
        in_transaction(con, lambda: register(con, other))
        """,
        [ALREADY_EXISTS, ok(None), ok([(1,)])],
    ),
    case(
        "a_refused_built_in_signature_leaves_the_transaction_usable",
        """
        con = duckdb.connect()
        in_transaction(con, lambda: register(con, other, name="lower"))
        """,
        [BUILT_IN_TAKEN, ok(None), ok([(1,)])],
    ),
    case(
        "a_refused_aggregate_name_leaves_the_transaction_usable",
        """
        con = duckdb.connect()
        in_transaction(con, lambda: register(con, other, name="sum"))
        """,
        [raised("CatalogException", "Catalog Error: sum is not an scalar function"), ok(None), ok([(1,)])],
    ),
    case(
        "a_refused_macro_name_leaves_the_transaction_usable",
        """
        con = duckdb.connect()
        con.execute("CREATE MACRO mymac(x) AS x")
        in_transaction(con, lambda: register(con, other, name="mymac"))
        """,
        [raised("CatalogException", "Catalog Error: mymac is not an scalar function"), ok(None), ok([(1,)])],
    ),
    case(
        "known_limitation_the_name_of_an_extension_function_that_cannot_load_aborts_the_transaction",
        """
        con = duckdb.connect()
        con.execute(f"SET extension_directory = '{DB}.extensions'")
        con.execute("SET autoinstall_known_extensions = false")
        in_transaction(con, lambda: register(con, other, name="host"))
        """,
        [
            raised("Error", "An error occurred while trying to automatically install the required extension 'inet:"),
            ok(None),
            ok([(0,)]),
        ],
    ),
]

# Removing a function must run no Python code of its callable, such as a finalizer that uses the engine, because
# the engine drops a removed function while it holds its catalog locks; the callable is released outside them, after
# a removal or before the next statement, or when its database closes, where such a finalizer must be free to use the
# engine and every connection.
RELEASES_OUTSIDE_ENGINE_LOCKS = [
    case(
        "a_finalizer_that_uses_the_connection_runs_before_a_statement_and_leaves_it_its_result",
        """
        con = duckdb.connect()
        cur = con.cursor()
        con.execute("CREATE TABLE log (s VARCHAR)")


        class Logger:
            def __call__(self, x):
                return x

            def __del__(self):
                con.execute("INSERT INTO log VALUES ('finalized')")


        cur.create_function("myf", Logger(), ["VARCHAR"], "VARCHAR")
        cur.execute("PREPARE p AS SELECT myf($1)")
        cur.remove_function("myf")
        # The close drops the prepared statement, the function's last copy, where nothing is released
        cur.close()
        step(lambda: con.execute("SELECT 42").fetchall())
        step(lambda: con.sql("SELECT * FROM log").fetchall())
        """,
        [ok([(42,)]), ok([("finalized",)])],
    ),
    case(
        "a_finalizer_that_uses_the_connection_runs_before_executemany_and_leaves_it_its_result",
        """
        con = duckdb.connect()
        cur = con.cursor()
        con.execute("CREATE TABLE log (s VARCHAR)")


        class Logger:
            def __call__(self, x):
                return x

            def __del__(self):
                con.execute("INSERT INTO log VALUES ('finalized')")


        cur.create_function("myf", Logger(), ["VARCHAR"], "VARCHAR")
        cur.execute("PREPARE p AS SELECT myf($1)")
        cur.remove_function("myf")
        cur.close()
        con.executemany("SELECT ?::INTEGER", [[42]])
        step(con.fetchall)
        step(lambda: con.sql("SELECT * FROM log").fetchall())
        """,
        [ok([(42,)]), ok([("finalized",)])],
    ),
    case(
        "a_finalizer_that_closes_the_releasing_connection_ends_its_call_cleanly",
        """
        class Closer:
            def __init__(self, targets):
                self.targets = targets

            def __call__(self, x):
                return f"got:{x}"

            def __del__(self):
                for target in self.targets:
                    target.close()


        # Released inside remove_function
        con = duckdb.connect()
        con.create_function("myf", Closer([con]), ["VARCHAR"], "VARCHAR")
        step(lambda: con.sql(CALL).fetchall())
        step(lambda: con.remove_function("myf"))
        step(lambda: con.sql("SELECT 1").fetchall())
        # Released before execute() runs its statement
        con = duckdb.connect()
        con.create_function("myf", Closer([con]), ["VARCHAR"], "VARCHAR")
        con.execute("PREPARE p AS SELECT myf($1)")
        con.remove_function("myf")
        con.execute("DEALLOCATE p")
        step(lambda: con.execute("SELECT 1"))
        # Released before a relation submits its statement
        con = duckdb.connect()
        con.create_function("myf", Closer([con]), ["VARCHAR"], "VARCHAR")
        con.execute("PREPARE p AS SELECT myf($1)")
        con.remove_function("myf")
        con.execute("DEALLOCATE p")
        rel = con.sql("SELECT 1")
        step(rel.fetchall)
        """,
        [
            ok([("got:b",)]),
            ok(None),
            raised("ConnectionException", "Connection Error: Connection has already been closed"),
            CURSOR_CLOSED,
            raised("ConnectionException", "Connection Error: Connection has already been closed"),
        ],
    ),
    case(
        "a_finalizer_at_the_database_close_can_close_the_parent_that_is_closing",
        """
        con = duckdb.connect()
        cur = con.cursor()
        finalized = []


        class Callable:
            def __call__(self, x):
                return f"got:{x}"

            def __del__(self):
                con.close()
                finalized.append(True)


        cur.create_function("myf", Callable(), ["VARCHAR"], "VARCHAR")
        step(lambda: con.sql(CALL).fetchall())
        step(con.close)
        step(lambda: finalized)
        """,
        [GOT_B, ok(None), ok([True])],
    ),
    case(
        "a_finalizer_at_the_database_close_can_wait_for_another_thread_closing_the_cursor",
        """
        parent = duckdb.connect()
        cur = parent.cursor()
        finalized = []


        class Callable:
            def __call__(self, x):
                return f"got:{x}"

            def __del__(self):
                worker = threading.Thread(target=cur.close)
                worker.start()
                worker.join()
                finalized.append(True)


        parent.create_function("myf", Callable(), ["VARCHAR"], "VARCHAR")
        step(lambda: parent.sql(CALL).fetchall())
        step(parent.close)
        step(lambda: finalized)
        """,
        [GOT_B, ok(None), ok([True])],
    ),
    case(
        "a_finalizer_at_the_database_close_can_wait_for_another_thread_closing_the_parent",
        """
        parent = duckdb.connect()
        cur = parent.cursor()
        finalized = []


        class Callable:
            def __call__(self, x):
                return f"got:{x}"

            def __del__(self):
                worker = threading.Thread(target=parent.close)
                worker.start()
                worker.join()
                finalized.append(True)


        parent.create_function("myf", Callable(), ["VARCHAR"], "VARCHAR")
        step(lambda: parent.sql(CALL).fetchall())
        step(parent.close)
        step(lambda: finalized)
        """,
        [GOT_B, ok(None), ok([True])],
    ),
    case(
        "a_finalizer_at_the_database_close_can_use_the_default_connection_that_replaces_its_own",
        """
        old = duckdb.connect()
        duckdb.set_default_connection(old)
        used = []


        class Callable:
            def __call__(self, x):
                return f"got:{x}"

            def __del__(self):
                used.append(duckdb.sql("SELECT 42").fetchall())


        # Type objects, since a type string is parsed on the default connection, which this call holds
        old.create_function("myf", Callable(), [duckdb.sqltypes.VARCHAR], duckdb.sqltypes.VARCHAR)
        step(lambda: duckdb.sql(CALL).fetchall())
        del old
        step(lambda: duckdb.set_default_connection(duckdb.connect()))
        step(lambda: used)
        """,
        [GOT_B, ok(None), ok([[(42,)]])],
    ),
    case(
        "a_finalizer_at_the_database_close_can_reopen_the_file_and_register_a_function",
        """
        con = duckdb.connect(DB)
        reopened = []


        class Callable:
            def __call__(self, x):
                return f"got:{x}"

            def __del__(self):
                again = duckdb.connect(DB)
                again.create_function("g", native(), ["VARCHAR"], "VARCHAR")
                reopened.append(again.sql("SELECT g('b')").fetchall())
                again.close()


        con.create_function("myf", Callable(), ["VARCHAR"], "VARCHAR")
        step(lambda: con.sql(CALL).fetchall())
        step(con.close)
        step(lambda: reopened)
        """,
        [GOT_B, ok(None), ok([[("got:b",)]])],
    ),
    case(
        "a_removed_callable_is_finalized_after_the_removal_where_it_can_use_a_cursor",
        """
        con = duckdb.connect()
        cur = con.cursor()


        class Callable:
            def __call__(self, x):
                return f"got:{x}"

            def __del__(self):
                step(lambda: cur.create_function("g", native(), ["VARCHAR"], "VARCHAR"))


        con.create_function("myf", Callable(), ["VARCHAR"], "VARCHAR")
        step(lambda: con.remove_function("myf"))
        step(lambda: "removed")
        step(con.close)
        """,
        [ok(None), ok(None), ok("removed"), ok(None)],
    ),
    case(
        "a_removed_callable_finalized_after_the_removal_can_wait_for_a_thread_using_a_cursor",
        """
        con = duckdb.connect()
        cur = con.cursor()
        entered = threading.Event()


        class Callable:
            def __call__(self, x):
                return f"got:{x}"

            def __del__(self):
                entered.set()
                worker.join()


        def registers():
            assert entered.wait(10)
            step(lambda: cur.create_function("g", native(), ["VARCHAR"], "VARCHAR"))


        worker = threading.Thread(target=registers, daemon=True)
        worker.start()
        con.create_function("myf", Callable(), ["VARCHAR"], "VARCHAR")
        step(lambda: con.remove_function("myf"))
        step(entered.is_set)
        step(con.close)
        """,
        [ok(None), ok(None), ok(True), ok(None)],
    ),
]


# The script ends at module level, so the callable's globals hold the connection when the interpreter exits.
EXITS_CLEANLY = [
    case(
        "an_exit_after_the_registering_connection_closes_is_clean",
        """
        con = duckdb.connect(DB)
        con.execute("CREATE TABLE t AS SELECT 'b' AS s")
        register(con)
        step(lambda: con.sql("SELECT myf(s) FROM t").fetchall())
        con.close()
        """,
        [GOT_B],
    ),
]

ERRORS_PRELUDE = """
seen = []


def call_once(con):
    # Two distinct values, so that the planner cannot fold the call from the column statistics and the function runs
    # only in execution
    con.execute("CREATE OR REPLACE TABLE two_rows AS SELECT * FROM (VALUES ('b'), ('c')) t(s)")
    return con.sql("SELECT myf(s) FROM two_rows").fetchall()


def query_the_connection(where):
    try:
        con.sql("SELECT 42").fetchall()
        outcome = "ran"
    except Exception as e:
        outcome = str(e).splitlines()[0]
    seen.append((where, outcome))


class Sentinel:
    def __del__(self):
        query_the_connection("frame released")


class Querying(Exception):
    def __str__(self):
        query_the_connection("error formatted")
        return "querying"


def raise_querying():
    sentinel = Sentinel()
    raise Querying()


class RaisingStr(Exception):
    def __str__(self):
        raise_querying()


class RaisingClass:
    @property
    def __class__(self):
        raise_querying()


class BadChild:
    @property
    def __class__(self):
        raise ValueError("bad child")


class BadArrow:
    def __arrow_array__(self, type=None):
        raise ValueError("arrow conversion failed")


class RenamingName(type):
    @property
    def __name__(cls):
        # A new string each time, so that the rename frees the one the class held
        type.__dict__["__name__"].__set__(cls, "".join(["Renam", "ing"]))
        raise_querying()


Renaming = RenamingName("".join(["Renam", "ing"]), (Exception,), {"__qualname__": "Renaming"})


class RaisingName(type):
    @property
    def __name__(cls):
        raise_querying()


class Unnamed(Exception, metaclass=RaisingName):
    pass


def raising_ndarray():
    import numpy as np

    class RaisingTolist(np.ndarray):
        def tolist(self):
            raise_querying()

    return np.zeros(2).view(RaisingTolist)
"""

INSIDE_A_QUERY = (
    "Resource In Use Error: code running inside a query cannot use that query's connection; use a separate "
    "connection from cursor()"
)
FORMATTED_AND_RELEASED_INSIDE = ok([("error formatted", INSIDE_A_QUERY), ("frame released", INSIDE_A_QUERY)])
RELEASED_INSIDE = ok([("frame released", INSIDE_A_QUERY)])


# The code around the user's function can raise a Python error too, whose message and frames may run Python code.
# Each case compares the distinct places and refusals, so that how often the engine calls a function is not pinned.
CONVERTS_ITS_OWN_PYTHON_ERRORS = [
    case(
        "an_error_of_the_function_is_formatted_and_released_inside_the_call",
        """
        con = duckdb.connect()
        con.create_function("myf", lambda x: raise_querying(), ["VARCHAR"], "VARCHAR")
        step(lambda: call_once(con))
        step(lambda: sorted(set(seen)))
        """,
        [udf_error("Querying: querying"), FORMATTED_AND_RELEASED_INSIDE],
    ),
    case(
        "an_error_whose_type_name_raises_is_reported_by_the_name_it_was_created_with",
        """
        con = duckdb.connect()


        def f(x):
            raise Unnamed("boom")


        con.create_function("myf", f, ["VARCHAR"], "VARCHAR")
        step(lambda: call_once(con))
        step(lambda: sorted(set(seen)))
        """,
        [udf_error("Unnamed: boom"), ok([])],
    ),
    case(
        "an_error_whose_type_name_renames_the_type_is_reported_by_the_name_it_was_created_with",
        """
        con = duckdb.connect()


        def f(x):
            raise Renaming("boom")


        con.create_function("myf", f, ["VARCHAR"], "VARCHAR")
        step(lambda: call_once(con))
        step(lambda: sorted(set(seen)))
        """,
        [udf_error("Renaming: boom"), ok([])],
    ),
    case(
        "an_error_whose_message_cannot_be_encoded_is_reported_by_its_type",
        """
        con = duckdb.connect()


        def f(x):
            raise ValueError("bad \\udcff")


        con.create_function("myf", f, ["VARCHAR"], "VARCHAR")
        step(lambda: call_once(con))
        """,
        [udf_error("ValueError: bad \\udcff")],
    ),
    case(
        "an_error_whose_str_raises_is_reported_by_its_type",
        """
        con = duckdb.connect()


        def f(x):
            raise RaisingStr()


        con.create_function("myf", f, ["VARCHAR"], "VARCHAR")
        step(lambda: call_once(con))
        step(lambda: sorted(set(seen)))
        """,
        [udf_error("RaisingStr"), RELEASED_INSIDE],
    ),
    case(
        "an_arrow_error_whose_str_raises_is_reported_by_its_type",
        """
        con = duckdb.connect()


        def f(x):
            raise RaisingStr()


        con.create_function("myf", f, ["VARCHAR"], "VARCHAR", type="arrow")
        step(lambda: call_once(con))
        step(lambda: sorted(set(seen)))
        """,
        [udf_error("RaisingStr"), RELEASED_INSIDE],
        NEEDS_PYARROW,
    ),
    case(
        "an_argument_whose_conversion_raises_fails_the_query",
        """
        con = duckdb.connect()
        # A zone the engine accepts and Python's zoneinfo does not know, so converting the argument fails
        con.execute("SET TimeZone = 'ACT'")
        con.create_function("myf", lambda ts: "seen", ["TIMESTAMPTZ"], "VARCHAR")
        step(lambda: con.sql("SELECT myf(TIMESTAMPTZ '2020-01-01 00:00:00+00')").fetchall())
        """,
        [ZONE_UNKNOWN_TO_PYTHON],
    ),
    case(
        "an_argument_whose_conversion_raises_fails_the_query_under_return_null",
        """
        con = duckdb.connect()
        # A zone the engine accepts and Python's zoneinfo does not know, so converting the argument fails
        con.execute("SET TimeZone = 'ACT'")
        con.create_function("myf", lambda ts: "seen", ["TIMESTAMPTZ"], "VARCHAR", exception_handling="return_null")
        step(lambda: con.sql("SELECT myf(TIMESTAMPTZ '2020-01-01 00:00:00+00')").fetchall())
        """,
        [ZONE_UNKNOWN_TO_PYTHON],
    ),
    case(
        "a_return_value_whose_type_check_raises_becomes_the_function_error",
        """
        con = duckdb.connect()
        con.create_function("myf", lambda x: RaisingClass(), ["VARCHAR"], "VARCHAR")
        step(lambda: call_once(con))
        step(lambda: sorted(set(seen)))
        """,
        [RAISED_BY_THE_CONVERSION, FORMATTED_AND_RELEASED_INSIDE],
    ),
    case(
        "an_ndarray_whose_tolist_raises_becomes_the_function_error",
        """
        con = duckdb.connect()
        con.create_function("myf", lambda x: raising_ndarray(), ["VARCHAR"], "DOUBLE[]")
        step(lambda: call_once(con))
        step(lambda: sorted(set(seen)))
        """,
        [RAISED_BY_THE_CONVERSION, FORMATTED_AND_RELEASED_INSIDE],
    ),
    case(
        "an_arrow_return_value_whose_type_check_raises_becomes_the_function_error",
        """
        con = duckdb.connect()
        con.create_function("myf", lambda x: RaisingClass(), ["VARCHAR"], "VARCHAR", type="arrow")
        step(lambda: call_once(con))
        step(lambda: sorted(set(seen)))
        """,
        [RAISED_BY_THE_CONVERSION, FORMATTED_AND_RELEASED_INSIDE],
        NEEDS_PYARROW,
    ),
    case(
        "a_return_value_whose_type_check_raises_fails_the_query_under_return_null",
        """
        con = duckdb.connect()
        con.create_function("myf", lambda x: RaisingClass(), ["VARCHAR"], "VARCHAR", exception_handling="return_null")
        step(lambda: call_once(con))
        step(lambda: sorted(set(seen)))
        """,
        [RAISED_BY_THE_CONVERSION, FORMATTED_AND_RELEASED_INSIDE],
    ),
    case(
        "an_arrow_return_value_whose_type_check_raises_fails_the_query_under_return_null",
        """
        con = duckdb.connect()
        con.create_function(
            "myf", lambda x: RaisingClass(), ["VARCHAR"], "VARCHAR", type="arrow", exception_handling="return_null"
        )
        step(lambda: call_once(con))
        step(lambda: sorted(set(seen)))
        """,
        [RAISED_BY_THE_CONVERSION, FORMATTED_AND_RELEASED_INSIDE],
        NEEDS_PYARROW,
    ),
    case(
        "a_nested_return_value_whose_conversion_raises_midway_fails_the_query",
        """
        con = duckdb.connect(config={"threads": 1})
        con.create_function(
            "myf", lambda x: [None, BadChild()] if x == 0 else [42, 43], ["BIGINT"], "BIGINT[]"
        )
        step(lambda: con.sql("SELECT myf(i) FROM range(2) t(i)").fetchall())
        step(lambda: con.sql("SELECT myf(i) FROM range(1, 2) t(i)").fetchall())
        """,
        [RAISED_BY_THE_CONVERSION, ok([([42, 43],)])],
    ),
    case(
        "a_nested_return_value_whose_conversion_raises_midway_fails_the_query_under_return_null",
        """
        con = duckdb.connect(config={"threads": 1})
        nested = lambda x: [None, BadChild()] if x == 0 else [42, 43]
        con.create_function("myf", nested, ["BIGINT"], "BIGINT[]", exception_handling="return_null")
        step(lambda: con.sql("SELECT myf(i) FROM range(2) t(i)").fetchall())
        step(lambda: con.sql("SELECT myf(i) FROM range(1, 2) t(i)").fetchall())
        """,
        [RAISED_BY_THE_CONVERSION, ok([([42, 43],)])],
    ),
    case(
        "an_arrow_return_value_whose_arrow_array_raises_fails_the_query",
        """
        con = duckdb.connect()
        con.create_function("myf", lambda x: BadArrow(), ["BIGINT"], "BIGINT", type="arrow")
        step(lambda: con.sql("SELECT myf(i) FROM range(2) t(i)").fetchall())
        """,
        [arrow_table_error("ValueError: arrow conversion failed")],
        NEEDS_PYARROW,
    ),
    case(
        "an_arrow_return_value_whose_arrow_array_raises_fails_the_query_under_return_null",
        """
        con = duckdb.connect()
        con.create_function(
            "myf", lambda x: BadArrow(), ["BIGINT"], "BIGINT", type="arrow", exception_handling="return_null"
        )
        step(lambda: con.sql("SELECT myf(i) FROM range(2) t(i)").fetchall())
        """,
        [arrow_table_error("ValueError: arrow conversion failed")],
        NEEDS_PYARROW,
    ),
]


@pytest.mark.parametrize(("script", "expected"), OWNS_ITS_CALLABLE)
def test_the_function_owns_its_callable(script, expected, tmp_path):
    check(script, expected, tmp_path)


@pytest.mark.parametrize(("script", "expected"), REMOVES_FOR_EVERY_CONNECTION)
def test_remove_function_removes_for_every_connection(script, expected, tmp_path):
    check_in_process(script, expected, tmp_path)


@pytest.mark.parametrize(("script", "expected"), ADDS_AN_OVERLOAD)
def test_create_function_adds_an_overload(script, expected, tmp_path):
    check_in_process(script, expected, tmp_path)


@pytest.mark.parametrize(("script", "expected"), RELEASES_OUTSIDE_ENGINE_LOCKS)
def test_a_released_callable_runs_outside_engine_locks(script, expected, tmp_path):
    check(script, expected, tmp_path)


@pytest.mark.parametrize(("script", "expected"), EXITS_CLEANLY)
def test_an_exit_after_cleaning_up_leaves_nothing_behind(script, expected, tmp_path):
    returncode, outcomes, stderr = run(script, tmp_path)
    leaked = "nanobind: leaked" in stderr
    write_ahead_logs = sorted(path.name for path in tmp_path.iterdir() if path.suffix == ".wal")
    assert (returncode, outcomes, leaked, write_ahead_logs) == (0, expected, False, []), stderr[-3000:]


@pytest.mark.parametrize(("script", "expected"), CONVERTS_ITS_OWN_PYTHON_ERRORS)
def test_a_python_error_around_the_function_becomes_its_error(script, expected, tmp_path):
    check_in_process(ERRORS_PRELUDE + textwrap.dedent(script), expected, tmp_path)

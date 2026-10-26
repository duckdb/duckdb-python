import subprocess
import sys
import textwrap

import pytest

import duckdb
from duckdb import sqltypes

THREAD_COUNTS_SCRIPT = """
    import os

    def thread_count():
        return len(os.listdir("/proc/self/task"))

    before_import = thread_count()
    import duckdb
    from duckdb import sqltypes

    sqltypes.BIGINT == "int8"
    after_import = thread_count()
    duckdb.execute("SELECT 42").fetchall()
    print(before_import, after_import, thread_count())
"""


def run_bounded(script, *args):
    # A deadlock can hold the GIL, so only faulthandler's own thread can still report where every thread is stuck.
    watchdog = "import faulthandler\nfaulthandler.dump_traceback_later(45, exit=True)\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", watchdog + textwrap.dedent(script), *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as hang:
        stderr = hang.stderr.decode() if isinstance(hang.stderr, bytes) else hang.stderr or ""
        msg = f"the child process hung for 60 seconds and was killed\n{stderr}"
        raise AssertionError(msg) from hang
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # An editable install prints its rebuild progress to stdout before the script runs.
    return proc.stdout.splitlines()[-1]


linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="counts threads through /proc/self/task")


@pytest.fixture(scope="module")
def thread_counts():
    before_import, after_import, after_first_use = map(int, run_bounded(THREAD_COUNTS_SCRIPT).split())
    return before_import, after_import, after_first_use


# Only the OS task list sees the engine's thread pools, and only while a database with threads stays open, so a
# database opened with one thread and no async threads, or opened and closed again, goes unnoticed.
@linux_only
class TestImportOpensNoDatabase:
    def test_import_starts_no_threads(self, thread_counts):
        before_import, after_import, _ = thread_counts
        assert after_import == before_import

    def test_first_use_opens_the_default_database(self, thread_counts):
        _, after_import, after_first_use = thread_counts
        assert after_first_use > after_import


class TestTypeObjectMatching:
    @pytest.mark.parametrize(
        ("type_object", "sql_type", "expected"),
        [
            ("NUMBER", "BIGNUM", True),
            ("NUMBER", "DECIMAL(4,1)", True),
            ("NUMBER", "DECIMAL(10,2)", True),
            ("NUMBER", "DECIMAL(18,3)", True),
            ("NUMBER", "DECIMAL(38,0)", True),
            ("NUMBER", "HUGEINT", True),
            ("NUMBER", "FLOAT", True),
            ("NUMBER", "VARCHAR", False),
            ("STRING", "VARCHAR", True),
            ("STRING", "JSON", True),
            ("STRING", "INTEGER", False),
            ("DATETIME", "TIMESTAMP", True),
            ("DATETIME", "VARCHAR", False),
            ("BINARY", "BLOB", True),
            ("BINARY", "VARCHAR", False),
            ("ROWID", "BIGINT", False),
        ],
    )
    def test_description_type_matches(self, duckdb_cursor, type_object, sql_type, expected):
        duckdb_cursor.execute(f"SELECT NULL::{sql_type} AS c")
        column_type = duckdb_cursor.description[0][1]
        assert str(column_type) == sql_type
        assert (column_type == getattr(duckdb, type_object)) is expected
        assert (getattr(duckdb, type_object) == column_type) is expected

    @pytest.mark.parametrize(
        ("sql_type", "category"),
        [
            *[
                (sql_type, "NUMBER")
                for sql_type in (
                    "TINYINT",
                    "UTINYINT",
                    "SMALLINT",
                    "USMALLINT",
                    "INTEGER",
                    "UINTEGER",
                    "BIGINT",
                    "UBIGINT",
                    "HUGEINT",
                    "UHUGEINT",
                    "BIGNUM",
                    "DECIMAL(4,1)",
                    "FLOAT",
                    "DOUBLE",
                )
            ],
            *[
                (sql_type, "DATETIME")
                for sql_type in (
                    "DATE",
                    "TIME",
                    "TIMETZ",
                    "TIMESTAMP",
                    "TIMESTAMPTZ",
                    "TIMESTAMP_NS",
                    "TIMESTAMP_MS",
                    "TIMESTAMP_S",
                )
            ],
            ("VARCHAR", "STRING"),
            ("JSON", "STRING"),
            ("BLOB", "BINARY"),
            *[
                (sql_type, None)
                for sql_type in (
                    "BOOLEAN",
                    "UUID",
                    "INTERVAL",
                    "TIME_NS",
                    "BIT",
                    "VARIANT",
                    "INTEGER[]",
                    "STRUCT(a INTEGER)",
                    "ENUM('a')",
                )
            ],
        ],
    )
    def test_each_type_belongs_to_exactly_its_category(self, duckdb_cursor, sql_type, category):
        duckdb_cursor.execute(f"SELECT NULL::{sql_type} AS c")
        column_type = duckdb_cursor.description[0][1]
        matches = {name for name in ("STRING", "NUMBER", "DATETIME", "BINARY") if column_type == getattr(duckdb, name)}
        assert matches == ({category} if category else set())
        assert column_type != duckdb.ROWID

    def test_an_aggregate_state_matches_by_the_id_of_its_value(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT count(*) EXPORT_STATE AS c FROM range(3)")
        column_type = duckdb_cursor.description[0][1]
        assert str(column_type) == "AGGREGATE_STATE"
        assert column_type == duckdb.NUMBER

    def test_the_categories_hold_no_type_instances(self):
        from duckdb import _dbapi_type_object

        categories = [duckdb.STRING, duckdb.NUMBER, duckdb.DATETIME, duckdb.BINARY]
        held = [*vars(_dbapi_type_object).values(), *(value for c in categories for value in vars(c).values())]
        assert not any(isinstance(value, sqltypes.DuckDBPyType) for value in held)
        assert all(isinstance(type_id, str) for c in categories for type_id in c.type_ids)

    def test_type_ids_are_copied(self, duckdb_cursor):
        type_ids = {"decimal"}
        category = duckdb.DBAPITypeObject(type_ids)
        type_ids.add("varchar")
        duckdb_cursor.execute("SELECT NULL::DECIMAL(10,2) AS c")
        assert category != sqltypes.VARCHAR
        assert category == duckdb_cursor.description[0][1]

    def test_type_ids_cannot_be_a_bare_string(self):
        with pytest.raises(TypeError):
            duckdb.DBAPITypeObject("decimal")

    def test_number_repr(self):
        assert repr(duckdb.NUMBER) == (
            "<DBAPITypeObject [BIGINT,BIGNUM,DECIMAL,DOUBLE,FLOAT,HUGEINT,INTEGER,SMALLINT,TINYINT,UBIGINT,UHUGEINT,"
            "UINTEGER,USMALLINT,UTINYINT]>"
        )


class TestTypeComparisonIsQuiet:
    def test_type_object_comparison_writes_nothing_to_stderr(self, duckdb_cursor, capfd):
        duckdb_cursor.execute("SELECT 'a'::VARCHAR AS c")
        column_type = duckdb_cursor.description[0][1]
        capfd.readouterr()
        assert column_type == duckdb.STRING
        assert duckdb.STRING == column_type  # noqa: SIM300
        assert capfd.readouterr().err == ""

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (sqltypes.BIGINT, "bigint", True),
            (sqltypes.BIGINT, "INT8", False),
            (sqltypes.BIGINT, "int8", False),
            (sqltypes.BIGINT, int, True),
            (sqltypes.BIGINT, int | None, True),
            (sqltypes.VARCHAR, str, True),
            (sqltypes.BIGINT, str, False),
            (sqltypes.DuckDBPyType(list[int]), list[int], True),
            (sqltypes.BIGINT, 42, False),
            (sqltypes.BIGINT, object(), False),
            (sqltypes.BIGINT, None, False),
        ],
        ids=[
            "BIGINT-bigint_string",
            "BIGINT-INT8_string",
            "BIGINT-int8_string",
            "BIGINT-int",
            "BIGINT-int_or_None",
            "VARCHAR-str",
            "BIGINT-str",
            "BIGINT_list-list_of_int",
            "BIGINT-42",
            "BIGINT-object",
            "BIGINT-None",
        ],
    )
    def test_converted_operand_comparison(self, capfd, left, right, expected):
        capfd.readouterr()
        assert (left == right) is expected
        assert (left != right) is not expected
        assert capfd.readouterr().err == ""

    def test_numpy_type_operand_comparison(self, capfd):
        np = pytest.importorskip("numpy")
        left = sqltypes.BIGINT
        capfd.readouterr()
        assert (left == np.int64) is True
        assert (left != np.int64) is False
        assert capfd.readouterr().err == ""


class TestTypeEquality:
    def test_typed_equality_is_exact_and_text_equality_ignores_case(self, duckdb_cursor):
        lower = sqltypes.DuckDBPyType("ENUM('a')", connection=duckdb_cursor)
        upper = sqltypes.DuckDBPyType("ENUM('A')", connection=duckdb_cursor)
        assert lower != upper
        assert upper != lower
        assert lower == str(upper)
        assert str(upper) == lower
        assert lower not in [upper]


TWO_THREADS_FIRST_USE_SCRIPT = """
    import sys
    import threading
    import duckdb

    if "close_first" in sys.argv:
        duckdb.close()
    barrier = threading.Barrier(2)
    errors = []

    def worker():
        barrier.wait()
        try:
            assert duckdb.cursor().execute("SELECT 1").fetchall() == [(1,)]
        except BaseException as error:
            errors.append(repr(error))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    print("SUCCESS")
"""

EIGHT_THREADS_FIRST_USE_SCRIPT = """
    import os
    import sys
    import threading
    import time
    import duckdb

    count_threads = "count_threads" in sys.argv

    def thread_count():
        return len(os.listdir("/proc/self/task"))

    def settled_thread_count():
        # A joined or closed thread can stay listed for a moment while the OS reaps it.
        deadline = time.monotonic() + 10
        last, since = thread_count(), time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.01)
            now = thread_count()
            if now != last:
                last, since = now, time.monotonic()
            elif time.monotonic() - since >= 0.2:
                break
        return last

    if count_threads:
        before_probe = thread_count()
        probe = duckdb.connect()
        one_database = thread_count() - before_probe
        probe.close()
        del probe
        idle = settled_thread_count()

    count = 8
    barrier = threading.Barrier(count)
    connections = [None] * count
    results = [None] * count
    errors = []

    def worker(i):
        barrier.wait()
        try:
            connections[i] = duckdb.default_connection()
            results[i] = connections[i].cursor().execute("SELECT 1").fetchall()
        except BaseException as error:
            errors.append(repr(error))

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert results == [[(1,)]] * count, results
    default = duckdb.default_connection()
    assert all(connection is default for connection in connections)
    if count_threads:
        remaining = settled_thread_count()
        assert abs(remaining - (idle + one_database)) <= 4, (remaining, idle, one_database)
    print("SUCCESS")
"""

REPLACE_WHILE_FETCHING_SCRIPT = """
    import threading
    import duckdb

    errors = []
    for _ in range(20):
        duckdb.execute("SELECT 1").fetchall()
        replacement = duckdb.connect()
        stop = threading.Event()
        fetching = threading.Event()

        def fetch_default(stop=stop, fetching=fetching):
            try:
                while not stop.is_set():
                    duckdb.default_connection()
                    fetching.set()
            except BaseException as error:
                errors.append(repr(error))
                fetching.set()

        fetcher = threading.Thread(target=fetch_default, daemon=True)
        fetcher.start()
        fetching.wait()
        duckdb.set_default_connection(replacement)
        del replacement
        stop.set()
        fetcher.join()
    assert not errors, errors
    print("SUCCESS")
"""

EXIT_WHILE_A_DAEMON_FETCHES_SCRIPT = """
    import threading
    import duckdb

    duckdb.set_default_connection(duckdb.connect(config={"threads": 1, "async_threads": 0}))
    looping = threading.Event()

    def fetch_forever():
        while True:
            duckdb.default_connection()
            looping.set()

    threading.Thread(target=fetch_forever, daemon=True).start()
    looping.wait()
    print("SUCCESS")
"""

FINALIZER_FETCHES_THE_DEFAULT_SCRIPT = """
    import duckdb
    from duckdb import sqltypes

    fetched = []

    class Identity:
        def __call__(self, x):
            return x

        def __del__(self):
            fetched.append(duckdb.default_connection())

    con = duckdb.connect()
    con.create_function("identity", Identity(), [sqltypes.BIGINT], sqltypes.BIGINT)
    duckdb.set_default_connection(con)
    del con
    replacement = duckdb.connect()
    duckdb.set_default_connection(replacement)
    assert len(fetched) == 1, fetched
    assert fetched[0] is replacement
    print("SUCCESS")
"""

SHUTDOWN_FINALIZER_SCRIPT = """
    import os
    import time
    import duckdb
    from duckdb import sqltypes

    def thread_count(listdir=os.listdir):
        return len(listdir("/proc/self/task"))

    def fetch(default_connection=duckdb.default_connection, refused=duckdb.ConnectionException):
        try:
            default_connection()
        except refused:
            return "refused"
        return "opened"

    class Identity:
        def __call__(self, x):
            return x

        # Module globals may already be gone when this runs at exit, so everything it needs is bound here.
        def __del__(
            self,
            write=os.write,
            fetch=fetch,
            thread_count=thread_count,
            connect=duckdb.connect,
            set_default_connection=duckdb.set_default_connection,
            sleep=time.sleep,
            monotonic=time.monotonic,
        ):
            before = thread_count()
            first = fetch()
            set_default_connection(connect())
            second = fetch()
            # A joined thread can stay listed in /proc for a moment, so the count gets two seconds to settle.
            after = thread_count()
            deadline = monotonic() + 2.0
            while after > before and monotonic() < deadline:
                sleep(0.001)
                after = thread_count()
            write(1, f"{first} {second} {before} {after}\\n".encode())

    duckdb.default_connection().create_function("identity", Identity(), [sqltypes.BIGINT], sqltypes.BIGINT)
    print("STARTED")
"""

RECREATED_AFTER_RELEASE_SCRIPT = """
    import sys
    import duckdb

    before = duckdb.default_connection()
    del duckdb._clean_default_connection
    del sys.modules["_duckdb"]._clean_default_connection
    after = duckdb.default_connection()
    assert after is not before
    assert duckdb.sql("SELECT 42").fetchall() == [(42,)]
    print("SUCCESS")
"""


class TestDefaultConnectionRace:
    @pytest.mark.parametrize("args", [(), ("close_first",)], ids=["first_use", "first_use_after_close"])
    def test_two_threads_open_the_default_connection(self, args):
        assert run_bounded(TWO_THREADS_FIRST_USE_SCRIPT, *args) == "SUCCESS"

    @pytest.mark.parametrize(
        "args",
        [(), pytest.param(("count_threads",), marks=linux_only)],
        ids=["shared_connection", "duplicates_torn_down"],
    )
    def test_eight_threads_share_one_default_connection(self, args):
        assert run_bounded(EIGHT_THREADS_FIRST_USE_SCRIPT, *args) == "SUCCESS"

    def test_replacing_the_default_connection_while_another_thread_fetches_it(self):
        assert run_bounded(REPLACE_WHILE_FETCHING_SCRIPT) == "SUCCESS"

    def test_exit_while_a_daemon_thread_fetches_the_default_connection(self):
        assert run_bounded(EXIT_WHILE_A_DAEMON_FETCHES_SCRIPT) == "SUCCESS"

    def test_replacing_the_default_connection_runs_a_finalizer_that_fetches_it(self):
        assert run_bounded(FINALIZER_FETCHES_THE_DEFAULT_SCRIPT) == "SUCCESS"

    def test_the_default_connection_is_recreated_after_its_cleanup_ran(self):
        assert run_bounded(RECREATED_AFTER_RELEASE_SCRIPT) == "SUCCESS"

    @linux_only
    def test_a_finalizer_at_exit_cannot_reopen_the_default_connection(self):
        first, second, before, after = run_bounded(SHUTDOWN_FINALIZER_SCRIPT).split()
        assert (first, second) == ("refused", "refused")
        assert int(after) <= int(before)

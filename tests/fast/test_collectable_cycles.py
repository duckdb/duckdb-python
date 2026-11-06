"""A cycle through a Python callable that a database keeps is reclaimed once nothing of that database is reachable.

A create_function() callable and a map() function belong to their database, which only C++ references, so the cycle
collector cannot see them on its own. Every open connection of the database holds one object that reports them, so
they become garbage together with the last of those, and never while any of them, a result or stream of the database,
or a DB-API result whose statement engine workers can still run, is still reachable. A database that connect() reaches
by its name, a file or a named in-memory database, never has them reported: a connection to such a database that is
alive at exit leaves the cycle to the end of the process, where nanobind reports its objects as leaked, even once the
connection that registered the function is gone, since the callable belongs to the database.
"""

import ctypes
import gc
import importlib.util
import subprocess
import sys
import textwrap
import threading
import time
import weakref

import pytest

import duckdb

needs_pyarrow = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")

STREAMED_ROWS = 100_000
BATCH_ROWS = 1_000
# One thread runs a statement only while its client fetches, so after execute() its function runs on no engine thread
# and the statement stays unfinished.
ONE_THREAD = {"threads": 1}
RACE_ROUNDS = 100


@pytest.fixture
def without_automatic_collection():
    """Only the explicit collections of the test run, so an object alive before one is alive because of a cycle."""
    enabled = gc.isenabled()
    gc.disable()
    yield
    if enabled:
        gc.enable()


class MapOwner:
    def __init__(self, con, rows=10) -> None:
        self.con = con
        self.rel = con.sql(f"select range as a from range({rows})").map(self.transform)

    def transform(self, df):
        return df


class FunctionOwner:
    def __init__(self, database=":memory:", config=None) -> None:
        self.con = duckdb.connect(database, config=config or {})
        self.con.create_function("plus_one", self.plus_one, ["BIGINT"], "BIGINT")

    def plus_one(self, x):
        return x + 1


def identity(df):
    return df


class OwnedSchema(dict):
    """A map() schema that refers back to its owner, so only the schema closes the cycle."""


def _mapped_relation_held_by_its_function_globals():
    namespace = {"duckdb": duckdb}
    exec(
        "con = duckdb.connect()\nrel = con.sql('select 1 as a').map(lambda df: df)\nassert rel.fetchall() == [(1,)]\n",
        namespace,
    )
    return namespace["rel"]


def _owner_of_a_mapped_relation():
    owner = MapOwner(duckdb.connect())
    assert owner.rel.fetchall() == [(i,) for i in range(10)]
    return owner


def _owner_of_a_filtered_mapped_relation():
    owner = MapOwner(duckdb.connect())
    owner.rel = owner.rel.filter("a > 4")
    assert owner.rel.fetchall() == [(i,) for i in range(5, 10)]
    return owner


def _owner_of_a_map_schema():
    class Owner:
        pass

    owner = Owner()
    owner.con = duckdb.connect()
    owner.schema = OwnedSchema(a="BIGINT")
    owner.schema.owner = owner
    owner.rel = owner.con.sql("select range as a from range(3)").map(identity, schema=owner.schema)
    assert owner.rel.fetchall() == [(0,), (1,), (2,)]
    return owner


def _owner_of_a_view_over_a_mapped_relation():
    owner = MapOwner(duckdb.connect())
    owner.rel.create_view("mapped")
    assert owner.con.sql("select count(*) from mapped").fetchall() == [(10,)]
    return owner


def _connection_held_by_its_function_globals():
    namespace = {"duckdb": duckdb}
    exec(
        "con = duckdb.connect()\n"
        "con.create_function('plus_one', lambda x: x + 1, ['BIGINT'], 'BIGINT')\n"
        "assert con.sql('select plus_one(1)').fetchall() == [(2,)]\n",
        namespace,
    )
    return namespace["con"]


def _owner_of_a_function():
    owner = FunctionOwner()
    assert owner.con.sql("select plus_one(1)").fetchall() == [(2,)]
    return owner


def _owner_of_a_function_and_a_cursor():
    owner = FunctionOwner()
    owner.cursor = owner.con.cursor()
    assert owner.cursor.sql("select plus_one(1)").fetchall() == [(2,)]
    return owner


def _function_that_captures_a_cursor():
    con = duckdb.connect()
    cursor = con.cursor()

    def plus_rows(x):
        return x + cursor.sql("select 1").fetchall()[0][0]

    con.create_function("plus_rows", plus_rows, ["BIGINT"], "BIGINT")
    return plus_rows


def _owner_after_ddl_through_execute():
    owner = FunctionOwner()
    owner.con.execute("create table t as select plus_one(range) as a from range(3)")
    return owner


def _owner_after_a_drained_execute():
    owner = FunctionOwner()
    assert owner.con.execute("select plus_one(range) from range(10)").fetchall() == [(i,) for i in range(1, 11)]
    return owner


class FailingOwner(FunctionOwner):
    def plus_one(self, x):
        if x == STREAMED_ROWS - 1:
            msg = "fails at the last row"
            raise ValueError(msg)
        return x + 1


# How many rows the first fetch runs depends on the result's buffer, so the stream fails at its last row instead
def _owner_of_a_failed_stream():
    owner = FailingOwner(config=ONE_THREAD)
    owner.con.execute(f"select plus_one(range) from range({STREAMED_ROWS})")
    with pytest.raises(duckdb.InvalidInputException, match="fails at the last row"):
        owner.con.fetchall()
    return owner


def _owner_of_an_interrupted_stream():
    owner = FunctionOwner(config=ONE_THREAD)
    owner.con.execute(f"select plus_one(range) from range({STREAMED_ROWS * 10})")
    assert owner.con.fetchone() == (1,)
    owner.con.interrupt()
    with pytest.raises(duckdb.InterruptException):
        owner.con.fetchall()
    return owner


def _owner_after_executemany():
    owner = FunctionOwner()
    owner.con.executemany("select plus_one(?)", [[1], [2]])
    assert owner.con.fetchall() == [(3,)]
    owner.con.execute("create table t (a bigint)")
    owner.con.executemany("insert into t select plus_one(?)", [[3], [4]])
    return owner


def _owner_after_fetchone_through_execute():
    owner = FunctionOwner()
    assert owner.con.execute("select plus_one(range) from range(10)").fetchone() == (1,)
    return owner


def _owner_with_an_unfetched_execute():
    owner = FunctionOwner()
    owner.con.execute("select plus_one(range) from range(10)")
    return owner


def _owner_with_an_unfinished_execute():
    owner = FunctionOwner(config=ONE_THREAD)
    owner.con.execute(f"select plus_one(range) from range({STREAMED_ROWS * 10})")
    return owner


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(_mapped_relation_held_by_its_function_globals, id="map_function_globals_hold_the_relation"),
        pytest.param(_owner_of_a_mapped_relation, id="map_method_of_the_relation_owner"),
        pytest.param(_owner_of_a_filtered_mapped_relation, id="map_method_of_the_owner_of_a_relation_over_it"),
        pytest.param(_owner_of_a_map_schema, id="map_schema_holds_its_owner"),
        pytest.param(_owner_of_a_view_over_a_mapped_relation, id="map_method_behind_a_view"),
        pytest.param(_connection_held_by_its_function_globals, id="udf_globals_hold_the_connection"),
        pytest.param(_owner_of_a_function, id="udf_method_of_the_connection_owner"),
        pytest.param(_owner_of_a_function_and_a_cursor, id="udf_method_of_the_owner_of_the_connection_and_a_cursor"),
        pytest.param(_function_that_captures_a_cursor, id="udf_that_captures_a_cursor_of_its_database"),
        pytest.param(_owner_after_ddl_through_execute, id="udf_owner_after_ddl_through_execute"),
        pytest.param(_owner_after_a_drained_execute, id="udf_owner_after_a_drained_execute"),
        pytest.param(_owner_after_executemany, id="udf_owner_after_executemany"),
        pytest.param(_owner_of_a_failed_stream, id="udf_owner_of_a_failed_stream"),
        pytest.param(_owner_of_an_interrupted_stream, id="udf_owner_of_an_interrupted_stream"),
    ],
)
def test_a_cycle_through_a_callable_its_database_keeps_is_collected(make, without_automatic_collection):
    ref = weakref.ref(make())
    # Alive until the collection, so it is the collector that reclaims it
    assert ref() is not None
    gc.collect()
    # Not `ref() is None`, whose failure report would print the object, which for a relation runs its query.
    collected = ref() is None
    assert collected


@pytest.mark.parametrize("end", ["exhausted", "released"])
@pytest.mark.parametrize(
    "make",
    [
        pytest.param(_owner_after_fetchone_through_execute, id="after_fetchone"),
        pytest.param(_owner_with_an_unfetched_execute, id="unfetched"),
        pytest.param(_owner_with_an_unfinished_execute, id="unfinished"),
    ],
)
def test_a_db_api_result_keeps_its_function_while_its_statement_can_run(make, end, without_automatic_collection):
    ref = weakref.ref(make())
    gc.collect()
    # Engine workers keep running a statement that has rows left between the client's fetches, without any call of it
    assert ref() is not None
    if end == "exhausted":
        ref().con.complete()
    else:
        # A statement that completes at execute takes the result's place and ends the statement before it
        ref().con.execute("create table released (a bigint)")
    gc.collect()
    collected = ref() is None
    assert collected


WORKERS_RUN_THE_FUNCTION = """
import gc
import sys
import time
import weakref

import duckdb

if sys.platform != "win32":
    import resource

    # A core dump of a process with many engine threads takes minutes to write
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def make():
    connections = []

    def plus_one(x):
        assert connections
        return x + 1

    con = duckdb.connect()
    connections.append(con)
    plus_one.con = con
    con.create_function("plus_one", plus_one, ["BIGINT"], "BIGINT")
    assert con.execute("select plus_one(range) from range(20000000)").fetchone() == (1,)
    return plus_one


gc.disable()
outcomes = set()
for _ in range(5):
    ref = weakref.ref(make())
    # The workers fill the result's buffer and park, and no frame of theirs holds the function any more
    time.sleep(0.5)
    gc.collect()
    alive = ref() is not None
    if alive:
        ref().con.execute("create table released (a bigint)")
    gc.collect()
    outcomes.add((alive, ref() is None))
print(sorted(outcomes), flush=True)
"""


def test_engine_workers_running_a_fetched_statement_keep_its_function():
    # A child process, since a function cleared while the workers call it crashes the interpreter
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(WORKERS_RUN_THE_FUNCTION)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    report = f"exit status {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    assert proc.returncode == 0, report
    # Alive while the workers can run the statement, collected once it ended; an editable build may print first
    assert proc.stdout.splitlines()[-1:] == ["[(True, True)]"], report


def test_closing_the_connection_frees_the_owner_of_a_mapped_relation_without_the_collector(
    without_automatic_collection,
):
    owner = MapOwner(duckdb.connect())
    assert owner.rel.fetchall() == [(i,) for i in range(10)]
    owner.con.close()
    ref = weakref.ref(owner)
    del owner
    freed = ref() is None
    assert freed


def test_closing_the_connection_releases_a_map_function_its_relation_still_names():
    con = duckdb.connect()

    def make():
        def transform(df):
            return df

        return transform

    transform = make()
    function_ref = weakref.ref(transform)
    rel = con.sql("select 1 as a").map(transform)
    del transform
    assert rel.fetchall() == [(1,)]
    con.close()
    gc.collect()
    assert function_ref() is None
    assert rel is not None


@needs_pyarrow
def test_a_mapped_relation_stays_callable_while_a_reader_streams_it():
    owner = MapOwner(duckdb.connect(config={"threads": 1}), rows=STREAMED_ROWS)
    reader = owner.rel.to_arrow_reader(BATCH_ROWS)
    first = reader.read_next_batch()
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is not None
    rest = reader.read_all()
    assert first.num_rows + rest.num_rows == STREAMED_ROWS
    assert sum(first.column(0).to_pylist()) + sum(rest.column(0).to_pylist()) == sum(range(STREAMED_ROWS))
    del reader
    gc.collect()
    assert ref() is None


@needs_pyarrow
def test_a_function_stays_callable_while_a_reader_of_its_connection_streams():
    owner = FunctionOwner()
    reader = owner.con.sql(f"select plus_one(range) as a from range({STREAMED_ROWS})").to_arrow_reader(BATCH_ROWS)
    first = reader.read_next_batch()
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is not None
    rest = reader.read_all()
    assert first.num_rows + rest.num_rows == STREAMED_ROWS
    assert sum(first.column(0).to_pylist()) + sum(rest.column(0).to_pylist()) == sum(range(1, STREAMED_ROWS + 1))
    del reader
    gc.collect()
    assert ref() is None


@needs_pyarrow
def test_a_function_stays_callable_while_a_reader_of_its_execute_result_streams():
    owner = FunctionOwner()
    owner.con.execute(f"select plus_one(range) as a from range({STREAMED_ROWS})")
    reader = owner.con.to_arrow_reader(BATCH_ROWS)
    first = reader.read_next_batch()
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is not None
    rest = reader.read_all()
    assert first.num_rows + rest.num_rows == STREAMED_ROWS
    del reader
    gc.collect()
    assert ref() is None


@needs_pyarrow
def test_a_reader_opened_before_the_first_function_keeps_it_while_it_streams():
    con = duckdb.connect()
    cursor = con.cursor()
    reader = cursor.sql(f"select range as a from range({STREAMED_ROWS})").to_arrow_reader(BATCH_ROWS)
    first = reader.read_next_batch()

    class Owner:
        def __init__(self, con) -> None:
            self.con = con
            con.create_function("plus_one", self.plus_one, ["BIGINT"], "BIGINT")

        def plus_one(self, x):
            return x + 1

    ref = weakref.ref(Owner(con))
    del con
    cursor.close()
    gc.collect()
    assert ref() is not None
    rest = reader.read_all()
    assert first.num_rows + rest.num_rows == STREAMED_ROWS
    del reader
    gc.collect()
    assert ref() is None


def test_rows_a_script_kept_keep_its_function_until_they_are_released():
    owner = FunctionOwner()
    kept, _, _ = owner.con._prepare_script("select plus_one(range) as a from range(3); select 1", None)
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is not None
    assert kept.fetchall() == [(1,), (2,), (3,)]
    del kept
    gc.collect()
    assert ref() is None


@needs_pyarrow
def test_a_reader_inside_the_cycle_keeps_it_until_the_reader_is_released():
    owner = FunctionOwner()
    owner.reader = owner.con.sql(f"select plus_one(range) as a from range({STREAMED_ROWS})").to_arrow_reader(BATCH_ROWS)
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    # The reader can still run the function, and nothing outside tells the collector the reader itself is garbage
    assert ref() is not None
    assert ref().reader.read_all().num_rows == STREAMED_ROWS
    del ref().reader
    gc.collect()
    assert ref() is None


def _clear(con):
    """The connection type's clear slot, which the cycle collector calls on garbage, called directly."""
    get_slot = ctypes.pythonapi.PyType_GetSlot
    get_slot.restype = ctypes.c_void_p
    get_slot.argtypes = [ctypes.py_object, ctypes.c_int]
    py_tp_clear = 51
    return ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object)(get_slot(type(con), py_tp_clear))(con)


def test_clearing_a_connection_releases_its_unfinished_result_and_its_functions():
    con = duckdb.connect(config=ONE_THREAD)

    def make():
        return lambda x: x + 1

    plus_one = make()
    function_ref = weakref.ref(plus_one)
    con.create_function("plus_one", plus_one, ["BIGINT"], "BIGINT")
    del plus_one
    con.execute(f"select plus_one(range) from range({STREAMED_ROWS * 10})")
    assert con.fetchone() == (1,)
    assert _clear(con) == 0
    assert function_ref() is None
    with pytest.raises(duckdb.InvalidInputException, match="No open result set"):
        con.fetchone()
    with pytest.raises(duckdb.ConnectionException):
        con.sql("select 1")
    assert _clear(con) == 0


def test_a_finalizer_that_close_runs_can_still_use_the_connection():
    pd = pytest.importorskip("pandas")
    con = duckdb.connect()
    results = []

    def on_free():
        con.create_function("late", lambda x: x + 1, ["BIGINT"], "BIGINT")
        results.append(con.sql("select late(41)").fetchall())
        results.append(con.sql("select 1 as a").map(identity).fetchall())

    frame = pd.DataFrame({"a": [1, 2]})
    weakref.finalize(frame, on_free)
    con._own_sources("v", [con._register_source(frame)])
    del frame
    con.close()
    assert results == [[(42,)], [(1,)]]


def _close_unless_busy(con):
    try:
        con.close()
    except duckdb.ResourceInUseException:
        return False
    return True


# The thread method, since a close or map() blocked in the engine never returns to Python, where a signal would act
@pytest.mark.timeout(180, method="thread")
def test_a_map_on_one_thread_while_another_closes_the_connection_ends_cleanly():
    pytest.importorskip("pandas")
    for _ in range(RACE_ROUNDS):
        con = duckdb.connect()
        rel = con.sql("select 1 as a")
        started = threading.Event()
        unexpected = []

        def map_until_closed(rel=rel, started=started, unexpected=unexpected):
            try:
                rel.map(identity)
                # Only after a first map, so every round races the close against a map that already ran once
                started.set()
                while True:
                    rel.map(identity)
            except duckdb.ConnectionException:
                pass
            except Exception as error:
                unexpected.append(error)
            finally:
                started.set()

        mapper = threading.Thread(target=map_until_closed, daemon=True)
        mapper.start()
        assert started.wait(timeout=60)
        deadline = time.monotonic() + 60
        while not _close_unless_busy(con):
            assert time.monotonic() < deadline
        mapper.join(timeout=60)
        assert not mapper.is_alive()
        assert unexpected == []


def test_a_refused_registration_keeps_nothing():
    con = duckdb.connect()
    con.create_function("plus_one", lambda x: x + 1, ["BIGINT"], "BIGINT")

    def make():
        return lambda x: x + 2

    refused = make()
    ref = weakref.ref(refused)
    with pytest.raises(duckdb.CatalogException, match="already exists"):
        con.create_function("plus_one", refused, ["BIGINT"], "BIGINT")
    del refused
    assert ref() is None
    assert con.sql("select plus_one(41)").fetchall() == [(42,)]


def test_a_resurrected_owner_keeps_a_working_connection():
    resurrected = []

    class Resurrecting(FunctionOwner):
        def __del__(self) -> None:
            resurrected.append(self)

    Resurrecting()
    gc.collect()
    # The collector clears weak references to garbage before it runs finalizers, so only the list shows the owner
    assert len(resurrected) == 1
    assert resurrected[0].con.sql("select plus_one(41)").fetchall() == [(42,)]
    ref = weakref.ref(resurrected.pop())
    gc.collect()
    assert ref() is None


def test_a_function_stays_callable_through_a_cursor_of_its_collected_owner():
    owner = FunctionOwner()
    cursor = owner.con.cursor()
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is not None
    assert cursor.sql("select plus_one(41)").fetchall() == [(42,)]
    cursor.close()
    gc.collect()
    assert ref() is None


def test_a_mapped_relation_stays_readable_while_another_connection_of_its_database_lives():
    con = duckdb.connect()
    owner = MapOwner(con.cursor())
    ref = weakref.ref(owner)
    rel_sql = owner.rel.sql_query()
    del owner
    gc.collect()
    assert ref() is not None
    assert con.sql(f"select count(*) from ({rel_sql})").fetchall() == [(10,)]


def test_a_function_of_the_default_connection_stays_callable():
    class DefaultConnectionOwner:
        def __init__(self) -> None:
            self.con = duckdb.default_connection()
            self.con.create_function("plus_one_default", self.plus_one, ["BIGINT"], "BIGINT")

        def plus_one(self, x):
            return x + 1

    ref = weakref.ref(DefaultConnectionOwner())
    try:
        gc.collect()
        assert ref() is not None
        assert duckdb.sql("select plus_one_default(41)").fetchall() == [(42,)]
    finally:
        duckdb.remove_function("plus_one_default")


def test_a_function_stays_callable_while_an_appender_owns_its_connection():
    owner = FunctionOwner()
    owner.con.sql("create table t (a bigint)")
    owner.appender = owner.con.appender("t")
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    owner = ref()
    assert owner is not None
    owner.appender.append_row(41)
    owner.appender.flush()
    assert owner.con.sql("select plus_one(a) from t").fetchall() == [(42,)]
    # Closing the database releases the callable and with it the cycle, which the collector never reported
    owner.con.close()


def test_a_function_stays_callable_through_a_second_connection_to_its_file(tmp_path):
    database = str(tmp_path / "shared.db")
    owner = FunctionOwner(database)
    second = duckdb.connect(database)
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is not None
    assert second.sql("select plus_one(41)").fetchall() == [(42,)]
    second.close()
    gc.collect()
    # The file can be opened again by its name at any moment, so the cycle is never reported.
    assert ref() is not None
    ref().con.close()
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize(
    "database", [pytest.param("file", id="file"), pytest.param(":memory:reconnected", id="named_memory")]
)
def test_a_function_survives_a_collection_that_reconnects_to_its_database_by_name(tmp_path, database):
    if database == "file":
        database = str(tmp_path / "named.db")
    owner = FunctionOwner(database)
    ref = weakref.ref(owner)
    reconnected = []

    class Reconnects:
        def __del__(self) -> None:
            reconnected.append(duckdb.connect(database))

    garbage = Reconnects()
    garbage.cycle = garbage
    del owner, garbage
    gc.collect()
    assert ref() is not None
    assert len(reconnected) == 1
    assert reconnected[0].sql("select plus_one(41)").fetchall() == [(42,)]
    reconnected[0].close()
    ref().con.close()
    gc.collect()
    assert ref() is None

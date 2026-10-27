"""The Arrow scan's state, release, reentrancy, importer and classification rules, checked through rows.

Each class pins one rule of the scan: state shared across prepare, rebind and execute; every exported stream and
array released once on every path; a source callback that calls back into its own connection refused; only requested
columns converted; one shot identity on the underlying reader; the legacy schema exports the scan still reads.
Nothing imports duckdb._sources at module level.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
import threading
import time
import weakref
from pathlib import Path

import pytest
from _pushdown_helpers import eventually

import duckdb

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")

CONSUMED = "has already been consumed and cannot be scanned again"


@pytest.fixture
def con():
    with duckdb.connect() as connection:
        yield connection


def rows(con, query):
    return con.execute(query).fetchall()


def scan_source(mixin, *args, **kwargs):
    """An instance of `mixin` built on the private ArrowScanSource, which is imported only when a test runs."""
    from duckdb._sources import ArrowScanSource

    return type(mixin.__name__, (mixin, ArrowScanSource), {})(*args, **kwargs)


NUMBERS = pa.table({"n": list(range(10)), "s": [str(i) for i in range(10)]})


def numbers_dataset():
    return ds.dataset(NUMBERS)


def numbers_lazy_frame():
    return pytest.importorskip("polars").from_arrow(NUMBERS).lazy()


REPEATABLE = [
    pytest.param(lambda: NUMBERS, id="table"),
    pytest.param(numbers_dataset, id="dataset"),
    pytest.param(numbers_lazy_frame, id="lazyframe"),
]


class TestStateAcrossPrepareAndExecute:
    """What a scan accepted and projected belongs to its bound query, and every execution reads afresh."""

    @pytest.mark.parametrize("make", REPEATABLE)
    def test_a_parameterless_prepared_statement_executed_twice(self, con, make):
        con.register("t", make())
        con.execute("PREPARE q AS SELECT s FROM t WHERE n > 6 ORDER BY n")
        con.execute("PREPARE either AS SELECT n FROM t WHERE (n = 1 OR s = '8') ORDER BY n")
        for _ in range(2):
            assert rows(con, "EXECUTE q") == [("7",), ("8",), ("9",)]
            assert rows(con, "EXECUTE either") == [(1,), (8,)]

    @pytest.mark.parametrize("make", REPEATABLE)
    def test_a_prepared_statement_with_changing_parameters(self, con, make):
        con.register("t", make())
        con.execute("PREPARE q AS SELECT n FROM t WHERE n > $1 AND s <> $2 ORDER BY n")
        assert rows(con, "EXECUTE q(6, '8')") == [(7,), (9,)]
        assert rows(con, "EXECUTE q(2, '3')") == [(4,), (5,), (6,), (7,), (8,), (9,)]
        assert con.execute("SELECT n FROM t WHERE n > ? ORDER BY n", [7]).fetchall() == [(8,), (9,)]
        assert con.execute("SELECT n FROM t WHERE n > ? ORDER BY n", [8]).fetchall() == [(9,)]

    @pytest.mark.parametrize("make", REPEATABLE)
    def test_a_self_join_with_different_predicates_per_side(self, con, make):
        con.register("t", make())
        query = "SELECT a.n, b.s FROM t a JOIN t b ON a.n = b.n - 1 WHERE a.n < 3 AND b.s > '1' ORDER BY a.n"
        assert rows(con, query) == [(1, "2"), (2, "3")]
        query = (
            "SELECT a.n, b.n FROM t a JOIN t b ON a.n = b.n - 1 "
            "WHERE (a.n = 1 OR a.s = '5') AND (b.n = 6 OR b.s = '2') ORDER BY a.n"
        )
        assert rows(con, query) == [(1, 2), (5, 6)]

    @pytest.mark.parametrize("make", REPEATABLE)
    def test_a_prepared_statement_over_a_registration_rebinds_after_unregister(self, con, make):
        # A registration is a view, and the engine re-binds a prepared statement once a view it read changes, so
        # the statement follows the name rather than keeping the object it was prepared over
        con.register("t", make())
        con.execute("PREPARE q AS SELECT s FROM t WHERE n < 2 ORDER BY n")
        con.unregister("t")
        gc.collect()
        with pytest.raises(duckdb.CatalogException):
            rows(con, "EXECUTE q")
        con.register("t", make())
        for _ in range(2):
            assert rows(con, "EXECUTE q") == [("0",), ("1",)]

    def test_each_execution_applies_what_the_bound_query_accepted(self, con):
        class Applying:
            def __init__(self, table) -> None:
                super().__init__(table)
                self.applied = []

            def __arrow_c_schema__(self) -> object:
                return self.obj.schema.__arrow_c_schema__()

            def accepts(self, predicate) -> bool:
                return True

            def stream(self, columns, filters) -> tuple[object, bool]:
                from duckdb._sources.nodes import fragment
                from duckdb._sources.to_arrow import to_arrow

                self.applied.append(sorted(fragment(predicate) for predicate in filters))
                held = self.obj
                for predicate in filters:
                    held = held.filter(to_arrow(predicate, self.obj.schema))
                return held.__arrow_c_stream__(), False

        source = scan_source(Applying, NUMBERS)
        con._register_arrow_source("t", source)
        con.execute("PREPARE q AS SELECT n FROM t WHERE n > 7 AND (n = 9 OR s = '8') ORDER BY n")
        assert rows(con, "EXECUTE q") == [(8,), (9,)]
        assert rows(con, "EXECUTE q") == [(8,), (9,)]
        both = sorted(['("n" > 7)', '(("n" = 9) OR ("s" = \'8\'))'])
        assert source.applied == [both, both]


def generator_reader(generators, *, batches=3, rows_per_batch=5, column=None):
    """A reader over a fresh generator, with a weak reference to it in `generators`.

    pyarrow frees the generator once nothing holds the reader or a stream exported from it, so a dead reference after
    the caller drops the reader means every stream the scan took was released.
    """
    arrow_schema = pa.schema([("n", pa.int64())]) if column is None else pa.schema([("n", pa.int64()), column])

    def produce():
        for index in range(batches):
            arrays = [pa.array([index] * rows_per_batch, pa.int64())]
            if column is not None:
                arrays.append(pa.array([2**62] * rows_per_batch, column.type))
            yield pa.record_batch(arrays, schema=arrow_schema)

    generator = produce()
    generators.append(weakref.ref(generator))
    return pa.RecordBatchReader.from_batches(arrow_schema, generator)


def all_released(generators, deadline=30.0):
    return bool(generators) and eventually(lambda: all(reference() is None for reference in generators), deadline)


def allocations_back_to(before):
    return eventually(lambda: pa.total_allocated_bytes() == before)


def query_a_variable(con, source, query):
    """Run `query`, which reads the variable `source` through the replacement scan, and return its rows or error class.

    The query runs in this frame so that no frame of the caller names the object: on Python 3.12 and earlier the
    replacement scan's read of f_locals keeps a variable of the querying frame alive until that frame ends.
    """
    try:
        return con.execute(query).fetchall()
    except duckdb.Error as error:
        return type(error)


def start_reading_a_variable(con, source, query, count):
    """Start `query` over the variable `source` and fetch its first `count` rows, leaving the result open."""
    con.execute(query)
    return con.fetchmany(count)


class TestRelease:
    """Every stream and array a scan takes is released once, on every path, for Python and pyarrow producers."""

    def test_a_full_read_releases_the_stream(self, con):
        generators = []
        query = "SELECT count(*), sum(n) FROM source"
        assert query_a_variable(con, generator_reader(generators), query) == [(15, 15)]
        assert all_released(generators)

    def test_a_limit_releases_the_stream_once_the_result_is_replaced(self, con):
        generators = []
        assert query_a_variable(con, generator_reader(generators, batches=1000), "SELECT n FROM source LIMIT 1") == [
            (0,)
        ]
        assert rows(con, "SELECT 42") == [(42,)]
        assert all_released(generators)

    def test_a_stream_still_held_is_not_reported_released(self):
        generators = []
        held = generator_reader(generators).__arrow_c_stream__()
        assert not all_released(generators, deadline=0.05)
        del held
        assert all_released(generators)

    def test_a_conversion_failure_releases_the_stream(self, con):
        generators = []
        reader = generator_reader(generators, batches=1000, column=pa.field("far", pa.timestamp("s")))
        assert query_a_variable(con, reader, "SELECT far FROM source") is duckdb.ConversionException
        del reader
        assert all_released(generators)

    def test_a_retained_result_keeps_the_table_until_it_is_dropped(self, con):
        gc.collect()
        before = pa.total_allocated_bytes()
        retained = pa.table({"s": [f"value number {i}" for i in range(50_000)]})
        first = start_reading_a_variable(con, retained, "SELECT s FROM source", 10)
        del retained
        gc.collect()
        rest = con.fetchall()
        assert [row[0] for row in first + rest] == [f"value number {i}" for i in range(50_000)]
        assert rows(con, "SELECT 1") == [(1,)]
        gc.collect()
        assert allocations_back_to(before), (pa.total_allocated_bytes(), before)

    def test_unregister_releases_a_partly_read_reader(self, con):
        generators = []
        con.register("r", generator_reader(generators, batches=1000))
        assert rows(con, "SELECT n FROM r LIMIT 1") == [(0,)]
        con.unregister("r")
        assert all_released(generators)

    def test_unregister_releases_a_table(self, con):
        gc.collect()
        before = pa.total_allocated_bytes()
        con.register("t", pa.table({"s": [str(i) for i in range(10_000)]}))
        assert rows(con, "SELECT count(*) FROM t WHERE s > '5'") == [(5_554,)]
        con.unregister("t")
        gc.collect()
        assert allocations_back_to(before), (pa.total_allocated_bytes(), before)

    def test_closing_the_connection_releases_every_object(self):
        generators = []
        gc.collect()
        before = pa.total_allocated_bytes()
        connection = duckdb.connect()
        connection.register("r", generator_reader(generators, batches=1000))
        connection.register("unread", generator_reader(generators))
        connection.register("t", pa.table({"s": [str(i) for i in range(10_000)]}))
        assert rows(connection, "SELECT n FROM r LIMIT 1") == [(0,)]
        assert rows(connection, "SELECT count(*) FROM t") == [(10_000,)]
        connection.close()
        assert all_released(generators)
        assert allocations_back_to(before), (pa.total_allocated_bytes(), before)

    @pytest.mark.parametrize(
        ("kind", "query"),
        [
            pytest.param("table", "SELECT n FROM src LIMIT 1", id="table_partly_read"),
            pytest.param("generator_reader", "SELECT 0", id="reader_unread"),
            pytest.param("generator_reader", "RETAINED", id="reader_retained_chunks"),
            pytest.param("table", "RETAINED", id="table_retained_chunks"),
            pytest.param("generator_reader", "SELECT n FROM src LIMIT 1", id="reader_partly_read"),
        ],
    )
    def test_interpreter_shutdown_with_a_registration_is_clean(self, kind, query):
        script = f"""
            import duckdb, pyarrow as pa

            def produce():
                for index in range(1000):
                    yield pa.record_batch([pa.array([index] * 5)], names=["n"])

            con = duckdb.connect()
            if {kind!r} == "table":
                con.register("src", pa.table({{"n": list(range(10))}}))
            else:
                con.register("src", pa.RecordBatchReader.from_batches(pa.schema([("n", pa.int64())]), produce()))
            if {query!r} == "RETAINED":
                con.execute("SELECT n FROM src")
                assert len(con.fetchmany(3)) == 3
            else:
                assert con.execute({query!r}).fetchall() == [(0,)]
            print("BODY_DONE", flush=True)
        """
        # A hang at exit can be a race, so three interpreters run at once and each must exit cleanly.
        for outcome in run_to_exit(textwrap.dedent(script), copies=3):
            assert outcome.body_done, f"the body did not finish within 60 seconds\n{outcome.report}"
            assert outcome.exited, f"the interpreter did not exit within 30 seconds of its body\n{outcome.report}"
            assert outcome.returncode == 0, outcome.report
            assert "Fatal Python error" not in outcome.stderr, outcome.report


class Outcome:
    """What a child interpreter did: finished its body, exited, and what it wrote."""

    def __init__(self) -> None:
        self.lines = []
        self.stderr = ""
        self.body_done = threading.Event()
        self.exited = False
        self.returncode = None

    @property
    def report(self):
        stdout = "\n".join(self.lines)
        return f"exit {self.returncode}\nstdout:\n{stdout}\nstderr:\n{self.stderr[-2000:]}"


def run_to_exit(script, *, copies, body_seconds=60, exit_seconds=30):
    """Run `copies` interpreters on `script` at once, timing their bodies apart from their shutdowns.

    A body ends by printing BODY_DONE; every child is killed if still running and reaped before this returns.
    """
    children = []
    outcomes = []
    readers = []

    def read_stdout(child, outcome):
        for line in child.stdout:
            outcome.lines.append(line.rstrip("\n"))
            if line.strip() == "BODY_DONE":
                outcome.body_done.set()

    try:
        for _ in range(copies):
            child = subprocess.Popen(
                [sys.executable, "-c", script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            children.append(child)
            outcome = Outcome()
            reader = threading.Thread(target=read_stdout, args=(child, outcome), daemon=True)
            reader.start()
            outcomes.append(outcome)
            readers.append(reader)
        for child, outcome in zip(children, outcomes, strict=True):
            if not outcome.body_done.wait(body_seconds):
                continue
            try:
                child.wait(timeout=exit_seconds)
                outcome.exited = True
            except subprocess.TimeoutExpired:
                pass
    finally:
        for child, outcome, reader in zip(children, outcomes, readers, strict=True):
            if child.poll() is None:
                child.kill()
            child.wait()
            reader.join(timeout=10)
            outcome.stderr = child.stderr.read()
            child.stderr.close()
            outcome.returncode = child.returncode
    return outcomes


class TestReentrancy:
    """A source callback that calls into the connection running its scan is refused; another connection works."""

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("callback", ["schema", "stream", "pull"])
    def test_the_same_connection_is_refused_from_a_callback(self, con, callback):
        seen = []

        def reenter():
            try:
                con.execute("SELECT 42").fetchall()
            except duckdb.Error as error:
                seen.append(type(error))
            else:
                seen.append(None)

        class Reentrant:
            def __arrow_c_schema__(self) -> object:
                if callback == "schema":
                    reenter()
                return NUMBERS.schema.__arrow_c_schema__()

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                if callback == "stream":
                    reenter()

                def produce():
                    if callback == "pull":
                        reenter()
                    yield from NUMBERS.to_batches()

                return pa.RecordBatchReader.from_batches(NUMBERS.schema, produce()).__arrow_c_stream__()

        src = Reentrant()  # noqa: F841
        result = con.execute("SELECT count(*) FROM src").fetchall()
        assert result == [(10,)]
        assert seen
        assert all(kind is duckdb.ResourceInUseException for kind in seen), seen

    @pytest.mark.timeout(60)
    def test_closing_the_same_connection_from_a_callback_is_refused(self, con):
        seen = []

        class Closing:
            def __arrow_c_schema__(self) -> object:
                return NUMBERS.schema.__arrow_c_schema__()

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                try:
                    con.close()
                except duckdb.Error as error:
                    seen.append(type(error))
                return NUMBERS.__arrow_c_stream__()

        src = Closing()  # noqa: F841
        assert con.execute("SELECT count(*) FROM src").fetchall() == [(10,)]
        assert seen == [duckdb.ResourceInUseException]
        assert rows(con, "SELECT 1") == [(1,)]

    @pytest.mark.timeout(60)
    def test_a_separate_connection_works_from_a_callback(self, con):
        other = duckdb.connect()
        answers = []

        class Asking:
            def __arrow_c_schema__(self) -> object:
                return NUMBERS.schema.__arrow_c_schema__()

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                answers.append(other.execute("SELECT 42").fetchall())
                return NUMBERS.__arrow_c_stream__()

        src = Asking()  # noqa: F841
        assert con.execute("SELECT count(*) FROM src").fetchall() == [(10,)]
        assert answers == [[(42,)]]
        other.close()

    @pytest.mark.timeout(60)
    def test_the_same_connection_is_refused_from_accepts(self, con):
        seen = []

        class Asking:
            def __arrow_c_schema__(self) -> object:
                return self.obj.schema.__arrow_c_schema__()

            def accepts(self, predicate) -> bool:
                try:
                    con.execute("SELECT 42").fetchall()
                except duckdb.Error as error:
                    seen.append(type(error))
                return False

        con._register_arrow_source("t", scan_source(Asking, NUMBERS))
        assert rows(con, "SELECT count(*) FROM t WHERE n > 7") == [(2,)]
        assert seen == [duckdb.ResourceInUseException]


CORE_TESTS = Path(__file__).with_name("test_python_arrow_scan_core.py")

# A child interpreter feeds the scan a stream built with ctypes (the scan core tests' CountingStream), corrupts one
# field of each exported array, restores it before the array's own release runs, and prints the outcome and the
# count of arrays the scan exported but never released.
MALFORMED = """
import importlib.util, json
spec = importlib.util.spec_from_file_location("scan_core", {core!r})
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
import ctypes, duckdb, pyarrow as pa


def child_count(array):
    array.n_children = 3

    def restore():
        array.n_children = 2

    return restore


def null_child(array):
    kept = ctypes.cast(array.children[0], ctypes.c_void_p).value
    array.children[0] = ctypes.POINTER(core._ArrowArray)()

    def restore():
        array.children[0] = ctypes.cast(kept, ctypes.POINTER(core._ArrowArray))

    return restore


def released_child(array):
    child = array.children[0].contents
    kept = child.release
    child.release = None

    def restore():
        child.release = kept

    return restore


def negative_length(array):
    kept = array.length
    array.length = -1

    def restore():
        array.length = kept

    return restore


def longer_than_children(array):
    kept = array.length
    array.length = kept * 10

    def restore():
        array.length = kept

    return restore


class Undoing(core.CountingStream):
    def __init__(self, schema, batches, mangle):
        super().__init__(schema, batches)
        self.mangle = mangle
        self.undo = {{}}

    def _get_next(self, stream, out):
        status = super()._get_next(stream, out)
        array = core._ArrowArray.from_address(out)
        if array.release:
            self.undo[array.private_data] = self.mangle(array)
        return status

    def _release_one(self, pointer):
        restore = self.undo.pop(core._ArrowArray.from_address(pointer).private_data, None)
        if restore is not None:
            restore()
        super()._release_one(pointer)


class Source(core.PythonStreamSource):
    def stream(self, columns, filters):
        stream = Undoing(self.table.schema, self.table.to_batches(max_chunksize=self.max_chunksize), {mangle})
        self.streams.append(stream)
        return stream.capsule(), False


table = pa.table({{"i": pa.array(range(5000), pa.int64()), "s": pa.array([str(i) for i in range(5000)])}})
source = Source(table)
with duckdb.connect(config={{"threads": 4}}) as con:
    con._register_arrow_source("t", source)
    try:
        con.execute("SELECT * FROM t").fetchall()
        outcome = "rows"
    except duckdb.Error as error:
        outcome = type(error).__name__ + ": " + str(error)
    after = con.execute("SELECT 42").fetchall()
print(json.dumps({{"outcome": outcome, "counts": source.counts(), "after": after}}))
"""


class TestImporter:
    """Only the columns a query reads are converted, malformed input is refused, and order survives odd batches."""

    @pytest.mark.parametrize(
        "export",
        [
            pytest.param(lambda reader: reader, id="reader"),
            pytest.param(lambda reader: reader.__arrow_c_stream__(), id="capsule"),
        ],
    )
    def test_a_narrow_projection_beside_a_column_that_cannot_convert(self, con, export):
        generators = []
        column = pa.field("far", pa.timestamp("s"))
        con.register("r", export(generator_reader(generators, column=column)))
        assert rows(con, "SELECT sum(n) FROM r") == [(15,)]
        con.register("again", export(generator_reader(generators, column=column)))
        with pytest.raises(duckdb.ConversionException):
            rows(con, "SELECT far FROM again")

    def test_a_narrow_projection_beside_a_column_that_cannot_convert_in_a_third_party_stream(self, con):
        reader = generator_reader([], column=pa.field("far", pa.timestamp("s")))

        class Stream:
            def __arrow_c_schema__(self) -> object:
                return reader.schema.__arrow_c_schema__()

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                return reader.__arrow_c_stream__()

        con.register("r", Stream())
        assert rows(con, "SELECT sum(n) FROM r") == [(15,)]

    def test_the_release_count_sees_an_array_still_held(self):
        from scan import test_python_arrow_scan_core as core

        table = pa.table({"i": pa.array(range(3000), pa.int64())})
        stream = core.CountingStream(table.schema, table.to_batches(max_chunksize=1000))
        reader = pa.RecordBatchReader._import_from_c_capsule(stream.capsule())
        held = reader.read_next_batch()
        assert stream.exported - stream.released == 1
        del held
        gc.collect()
        assert stream.exported - stream.released == 0
        reader.read_all()
        del reader
        gc.collect()
        assert (stream.exported - stream.released, stream.stream_released) == (0, 1)

    @pytest.mark.parametrize(
        ("mangle", "message"),
        [
            pytest.param("child_count", "carried 3 columns where 2 were expected", id="child_count"),
            pytest.param("null_child", "carried column 0 missing or released already", id="null_child_pointer"),
            pytest.param("released_child", "carried column 0 missing or released already", id="released_child"),
            pytest.param("negative_length", "which do not describe a range of rows", id="negative_length"),
            pytest.param(
                "longer_than_children",
                "column 0 holds 1000 rows where 10000 were expected",
                id="longer_than_children",
            ),
        ],
    )
    def test_a_malformed_array_is_refused_and_released(self, mangle, message):
        import json

        # A crash in a process with many engine threads would spend minutes writing a core file.
        no_core = (
            "import resource\nresource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n" if sys.platform != "win32" else ""
        )
        script = no_core + MALFORMED.format(core=str(CORE_TESTS), mangle=mangle)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        report = f"exit {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr[-2000:]}"
        assert completed.returncode == 0, report
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        assert result["outcome"].startswith("InvalidInputException"), report
        assert message in result["outcome"], report
        assert result["counts"] == [[0, 1]], report
        assert result["after"] == [[42]], report

    def test_order_survives_empty_batches_and_short_tails_under_four_threads(self, con):
        con.execute("SET threads = 4")
        sizes = [0, 2048, 0, 0, 3000, 1, 0, 2047, 2049, 5, 0, 4096, 3, 0]
        arrow_schema = pa.schema([("n", pa.int64())])

        def produce():
            start = 0
            for size in sizes * 8:
                yield pa.record_batch([pa.array(range(start, start + size), pa.int64())], schema=arrow_schema)
                start += size

        total = sum(sizes) * 8
        con.register("src", pa.RecordBatchReader.from_batches(arrow_schema, produce()))
        assert [row[0] for row in rows(con, "SELECT n FROM src")] == list(range(total))
        con.register("src", pa.RecordBatchReader.from_batches(arrow_schema, produce()))
        con.execute("CREATE TABLE copied AS SELECT * FROM src")
        assert [row[0] for row in rows(con, "SELECT n FROM copied")] == list(range(total))

    @pytest.mark.parametrize("arrays", [1, 2], ids=["single_array", "stream_of_two_arrays"])
    def test_a_null_struct_row_hides_the_values_its_children_hold(self, con, arrays):
        first = pa.StructArray.from_arrays(
            [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])], ["n", "s"], mask=pa.array([False, True, False])
        )
        second = pa.StructArray.from_arrays(
            [pa.array([4, 5, 6]), pa.array(["d", "e", "f"])], ["n", "s"], mask=pa.array([True, False, False])
        )
        expected = [(1, "a"), (None, None), (3, "c")]
        if arrays == 1:
            con.register("masked", first)
        else:
            con.register("masked", pa.chunked_array([first, second]))
            expected += [(None, None), (5, "e"), (6, "f")]
        assert rows(con, "SELECT n, s FROM masked") == expected

    def test_dictionary_contents_change_across_batches(self, con):
        kind = pa.dictionary(pa.int32(), pa.string())
        arrow_schema = pa.schema([("d", kind)])
        batches = [
            pa.record_batch(
                [pa.DictionaryArray.from_arrays(pa.array([0, 1, 1]), pa.array(["a", "b"]))], schema=arrow_schema
            ),
            pa.record_batch(
                [pa.DictionaryArray.from_arrays(pa.array([1, 0]), pa.array(["x", "y"]))], schema=arrow_schema
            ),
            pa.record_batch(
                [pa.DictionaryArray.from_arrays(pa.array([2, None]), pa.array(["p", "q", "r"]))], schema=arrow_schema
            ),
        ]
        expected = [("a",), ("b",), ("b",), ("y",), ("x",), ("r",), (None,)]
        con.register("coded", pa.RecordBatchReader.from_batches(arrow_schema, batches))
        assert rows(con, "SELECT d FROM coded") == expected
        con.register("coded_table", pa.Table.from_batches(batches))
        assert rows(con, "SELECT d FROM coded_table") == expected


class TestClassificationAuthority:
    """One shot behavior follows the underlying reader, never the adapter around it or a class's name."""

    def test_one_reader_registered_on_two_connections_is_read_once(self, con):
        reader = pa.RecordBatchReader.from_batches(NUMBERS.schema, NUMBERS.to_batches())
        other = con.cursor()
        con.register("r", reader)
        other.register("r", reader)
        assert rows(con, "SELECT count(*) FROM r") == [(10,)]
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            rows(other, "SELECT count(*) FROM r")

    def test_two_adapters_around_one_reader_share_its_one_read(self, con):
        from duckdb._sources import adapt

        reader = pa.RecordBatchReader.from_batches(NUMBERS.schema, NUMBERS.to_batches())
        other = con.cursor()
        con._register_arrow_source("r", adapt(reader))
        other._register_arrow_source("r", adapt(reader))
        assert rows(con, "SELECT count(*) FROM r") == [(10,)]
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            rows(other, "SELECT count(*) FROM r")

    def test_a_class_named_like_the_reader_stays_reusable(self, con):
        class RecordBatchReader:
            __module__ = "pyarrow.lib"

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                return NUMBERS.__arrow_c_stream__(requested_schema)

        con.register("forged", RecordBatchReader())
        for _ in range(2):
            assert rows(con, "SELECT count(*) FROM forged") == [(10,)]


class LegacySchema:
    """A schema object that offers only pyarrow's older C export."""

    def __init__(self, arrow_schema) -> None:
        self.arrow_schema = arrow_schema

    def _export_to_c(self, address) -> None:
        self.arrow_schema._export_to_c(address)


class TestRetainedBehaviors:
    def test_the_legacy_schema_export_answers_binding(self, con):
        class Legacy:
            def __init__(self) -> None:
                self.schema = LegacySchema(NUMBERS.schema)
                self.exports = 0

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                self.exports += 1
                return NUMBERS.__arrow_c_stream__()

        source = Legacy()
        con.register("legacy", source)
        assert [row[:2] for row in con.execute("DESCRIBE legacy").fetchall()] == [("n", "BIGINT"), ("s", "VARCHAR")]
        assert source.exports == 0
        for execution in (1, 2, 3):
            assert rows(con, "SELECT count(*) FROM legacy WHERE n > 4") == [(5,)]
            assert source.exports == execution

    def test_a_schema_hook_costs_one_export_per_execution(self, con):
        class Hooked:
            def __init__(self) -> None:
                self.exports = 0
                self.schemas = 0

            def __arrow_c_schema__(self) -> object:
                self.schemas += 1
                return NUMBERS.schema.__arrow_c_schema__()

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                self.exports += 1
                return NUMBERS.__arrow_c_stream__()

        source = Hooked()
        # A generic exporter may change its schema, so every bind asks its schema hook again.
        con.register("hooked", source)
        assert source.schemas >= 1
        reads = []
        for statement in ("DESCRIBE hooked", "EXPLAIN SELECT * FROM hooked"):
            before = source.schemas
            con.execute(statement).fetchall()
            reads.append(source.schemas - before)
        assert source.exports == 0
        for execution in (1, 2, 3):
            before = source.schemas
            assert rows(con, "SELECT sum(n) FROM hooked") == [(45,)]
            assert source.exports == execution
            reads.append(source.schemas - before)
        assert reads == [1, 1, 1, 1, 1]

    def test_a_stable_schema_is_read_once_for_every_bind(self, con):
        from duckdb._sources import adapt

        source = adapt(NUMBERS)
        assert source.schema_is_stable
        hook = source.__arrow_c_schema__
        reads = []

        def counted() -> object:
            reads.append(1)
            return hook()

        source.__dict__["__arrow_c_schema__"] = counted
        con._register_arrow_source("stable", source)
        con.execute("DESCRIBE stable").fetchall()
        con.execute("EXPLAIN SELECT * FROM stable").fetchall()
        for _ in range(3):
            assert rows(con, "SELECT sum(n) FROM stable") == [(45,)]
        assert len(reads) == 1

    @pytest.mark.parametrize("kind", ["dataset", "reader"])
    def test_a_dataset_and_a_reader_read_their_schema_once(self, con, kind):
        import io

        import pyarrow.ipc

        class CountingDataset(ds.Dataset):
            def __init__(self) -> None:
                self.schema_reads = 0

            @property
            def schema(self) -> object:
                self.schema_reads += 1
                return NUMBERS.schema

            def scanner(self, **kwargs) -> object:
                return ds.dataset(NUMBERS).scanner(**kwargs)

        class CountingReader(pyarrow.ipc.RecordBatchStreamReader):
            def __init__(self, source) -> None:
                super().__init__(source)
                self.schema_reads = 0

            @property
            def schema(self) -> object:
                self.schema_reads += 1
                return super().schema

            def __arrow_c_schema__(self) -> object:
                self.schema_reads += 1
                return super().schema.__arrow_c_schema__()

        if kind == "dataset":
            source = CountingDataset()
        else:
            sink = io.BytesIO()
            with pyarrow.ipc.new_stream(sink, NUMBERS.schema) as writer:
                writer.write_table(NUMBERS)
            source = CountingReader(sink.getvalue())
        con.register("counted", source)
        reads = [source.schema_reads]
        con.execute("DESCRIBE counted").fetchall()
        reads.append(source.schema_reads)
        con.execute("EXPLAIN SELECT * FROM counted").fetchall()
        reads.append(source.schema_reads)
        for _ in range(2 if kind == "dataset" else 1):
            assert rows(con, "SELECT count(*), sum(n) FROM counted") == [(10, 45)]
            reads.append(source.schema_reads)
        assert reads == [1] * len(reads)


def built_in_python(batch_count=300, batch_rows=50):
    """A reader whose every batch is built by Python code, so each pull needs the GIL."""
    arrow_schema = pa.schema([("n", pa.int64())])

    def produce():
        for index in range(batch_count):
            yield pa.record_batch([pa.array([index * batch_rows + j for j in range(batch_rows)])], schema=arrow_schema)

    return pa.RecordBatchReader.from_batches(arrow_schema, produce())


def fetch_all(con):
    return con.execute("SELECT n FROM built ORDER BY n").fetchall()


def fetch_one_by_one(con):
    con.execute("SELECT n FROM built ORDER BY n")
    fetched = []
    while (row := con.fetchone()) is not None:
        fetched.append(row)
    return fetched


def fetch_in_parts(con):
    con.execute("SELECT n FROM built ORDER BY n")
    fetched = []
    while part := con.fetchmany(777):
        fetched.extend(part)
    return fetched


def fetch_record_batches(con):
    reader = con.sql("SELECT n FROM built ORDER BY n").to_arrow_reader(1000)
    return [(value,) for value in reader.read_all().column(0).to_pylist()]


def fetch_frame(con):
    return [(value,) for value in con.sql("SELECT n FROM built ORDER BY n").df()["n"].tolist()]


def fetch_arrow(con):
    return [(value,) for value in con.sql("SELECT n FROM built ORDER BY n").arrow().read_all().column(0).to_pylist()]


def fetch_polars(con):
    pytest.importorskip("polars")
    return [(value,) for value in con.sql("SELECT n FROM built ORDER BY n").pl()["n"].to_list()]


def fetch_through_a_relation_chain(con):
    return con.table("built").filter("n >= 0").project("n").order("n").fetchall()


def fetch_with_a_python_function(con):
    from duckdb.sqltypes import BIGINT

    con.create_function("same", lambda value: value, [BIGINT], BIGINT)
    return con.execute("SELECT same(n) FROM built ORDER BY 1").fetchall()


FETCHES = [
    fetch_all,
    fetch_one_by_one,
    fetch_in_parts,
    fetch_record_batches,
    fetch_frame,
    fetch_arrow,
    fetch_polars,
    fetch_through_a_relation_chain,
    fetch_with_a_python_function,
]


class TestGil:
    """A stream whose pulls run Python finishes through every fetch path, on one engine thread or four."""

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, 4])
    @pytest.mark.parametrize("fetch", FETCHES, ids=[fetch.__name__ for fetch in FETCHES])
    def test_a_python_built_stream_completes(self, con, threads, fetch):
        con.execute(f"SET threads = {threads}")
        con.register("built", built_in_python())
        assert fetch(con) == [(n,) for n in range(300 * 50)]

    @pytest.mark.timeout(60)
    def test_an_interrupt_stops_a_scan_that_pulls_python(self, con):
        started = threading.Event()
        arrow_schema = pa.schema([("n", pa.int64())])

        def produce():
            started.set()
            for index in range(1_000_000):
                time.sleep(0.001)
                yield pa.record_batch([pa.array([index])], schema=arrow_schema)

        con.register("endless", pa.RecordBatchReader.from_batches(arrow_schema, produce()))

        def interrupt():
            started.wait(30)
            time.sleep(0.2)
            con.interrupt()

        interrupter = threading.Thread(target=interrupt)
        interrupter.start()
        with pytest.raises(duckdb.InterruptException):
            con.execute("SELECT sum(n) FROM endless").fetchall()
        interrupter.join()
        assert rows(con, "SELECT 42") == [(42,)]


class TestArrowUdfResults:
    """An Arrow scalar function's result is read whole, every batch of it, with its column, row and type checks."""

    def test_a_result_of_several_batches_is_read_whole(self, con):
        from duckdb.sqltypes import BIGINT

        def two_batches(values):
            half = len(values) // 2
            return pa.chunked_array([values.slice(0, half), values.slice(half)])

        con.create_function("two_batches", two_batches, [BIGINT], BIGINT, type="arrow")
        assert rows(con, "SELECT two_batches(i) FROM range(5000) t(i)") == [(i,) for i in range(5000)]

    def test_a_cast_failure_in_a_later_batch_is_reported(self, con):
        from duckdb.sqltypes import BIGINT

        def text_batches(values):
            half = len(values) // 2
            first = pa.array(["1"] * half)
            second = pa.array(["2"] * (len(values) - half - 1) + ["abc"])
            return pa.chunked_array([first, second])

        con.create_function("text_batches", text_batches, [BIGINT], BIGINT, type="arrow")
        with pytest.raises(duckdb.ConversionException, match="Could not convert string 'abc' to INT64"):
            rows(con, "SELECT text_batches(i) FROM range(10) t(i)")
        assert rows(con, "SELECT 42") == [(42,)]

    def test_a_row_count_that_differs_from_the_input_is_refused(self, con):
        from duckdb.sqltypes import BIGINT

        con.create_function("shorter", lambda values: values.slice(1), [BIGINT], BIGINT, type="arrow")
        with pytest.raises(duckdb.InvalidInputException, match="Returned pyarrow table should have 10 tuples, found 9"):
            rows(con, "SELECT shorter(i) FROM range(10) t(i)")

    def test_a_uuid_result_without_the_uuid_extension_type_is_refused(self, con):
        from duckdb.sqltypes import BIGINT, UUID

        def plain_bytes(values):
            return pa.array([b"\x00" * 16] * len(values), pa.binary(16))

        con.create_function("plain_bytes", plain_bytes, [BIGINT], UUID, type="arrow")
        expected = "It seems that you are using the UUID arrow canonical extension, but the same is not yet registered"
        with pytest.raises(duckdb.Error, match=expected):
            rows(con, "SELECT plain_bytes(i) FROM range(2) t(i)")

    def test_a_json_result_without_the_json_extension_type_is_refused(self, con):
        from duckdb.sqltypes import BIGINT

        con.create_function(
            "plain_text",
            lambda values: pa.array(['{"a": 1}'] * len(values)),
            [BIGINT],
            duckdb.sqltype("JSON"),
            type="arrow",
        )
        expected = "It seems that you are using the JSON arrow canonical extension, but the same is not yet registered"
        with pytest.raises(duckdb.Error, match=expected):
            rows(con, "SELECT plain_text(i) FROM range(2) t(i)")

    def test_uuid_and_json_extension_results_arrive(self, con):
        import uuid

        from duckdb.sqltypes import BIGINT, UUID

        identity = uuid.UUID(int=5)

        def uuids(values):
            return pa.ExtensionArray.from_storage(pa.uuid(), pa.array([identity.bytes] * len(values), pa.binary(16)))

        def documents(values):
            return pa.ExtensionArray.from_storage(pa.json_(), pa.array(['{"a": 1}'] * len(values)))

        con.create_function("uuids", uuids, [BIGINT], UUID, type="arrow")
        con.create_function("documents", documents, [BIGINT], duckdb.sqltype("JSON"), type="arrow")
        assert rows(con, "SELECT uuids(i), documents(i)->>'a' FROM range(2) t(i)") == [(identity, "1")] * 2


def fragment_cases():
    """Predicate nodes and the text fragment() renders for each, which plan and filter assertions compare against."""
    import datetime

    from duckdb._sources.nodes import Binary, Col, In, Lit, Postfix, Unary

    def column(name):
        return Col((name,))

    def integers(*values):
        return [Lit(value, "INTEGER") for value in values]

    zoned = Lit(1_714_953_600_000_000, "TIMESTAMP WITH TIME ZONE")
    return [
        (Binary(">", column("n"), Lit(7, "BIGINT")), '("n" > 7)'),
        (Binary("!=", column("s"), Lit("9", "VARCHAR")), "(\"s\" != '9')"),
        (In(column("n"), integers(3, 4)), '("n" IN (3, 4))'),
        (Binary(">", column(""), Lit(1, "BIGINT")), '("" > 1)'),
        (Binary(">=", column("d"), Lit(datetime.date(9999, 12, 31), "DATE")), "(\"d\" >= DATE '9999-12-31')"),
        (
            Binary("AND", Binary(">", column("n"), Lit(1, "INTEGER")), Binary("<", column("n"), Lit(8, "INTEGER"))),
            '(("n" > 1) AND ("n" < 8))',
        ),
        (Unary("NOT", In(column("n"), integers(0, 1, 3))), '(NOT ("n" IN (0, 1, 3)))'),
        (Postfix("IS NOT NULL", column("n")), '("n" IS NOT NULL)'),
        (
            Binary("OR", Binary("=", column("n"), Lit(9, "BIGINT")), Binary("=", column("s"), Lit("8", "VARCHAR"))),
            '(("n" = 9) OR ("s" = \'8\'))',
        ),
        (Binary(">", column("zoned"), zoned), "(\"zoned\" > TIMESTAMPTZ '2024-05-06 00:00:00+00')"),
        (Binary(">", Col(("st", "x")), Lit(6, "INTEGER")), '("st"."x" > 6)'),
    ]


class TestFragments:
    """The texts that offered and applied predicates are compared against, as the renderer writes them."""

    def test_each_node_renders_its_expected_text(self):
        from duckdb._sources.nodes import fragment

        for node, text in fragment_cases():
            assert fragment(node) == text


SUBINTERPRETER_TESTS = Path(__file__).resolve().parents[2] / "test_subinterpreter_gil.py"

# Scanned after the process created a subinterpreter, a Python generator backed reader must still be pulled safely.
GENERATOR_INPUT = """
import pyarrow as pa


def produce():
    for index in range(200):
        yield pa.record_batch([pa.array(range(index * 100, index * 100 + 100))], names=["n"])


print("STARTED")
for threads in (1, 4):
    con = duckdb.connect()
    con.execute(f"SET threads = {threads}")
    con.execute("SET enable_progress_bar = false")
    con.register("src", pa.RecordBatchReader.from_batches(pa.schema([("n", pa.int64())]), produce()))
    print(con.execute("SELECT count(*), sum(n) FROM src").fetchall())
    con.close()
"""


class TestAfterASubinterpreter:
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="the child process hangs in the scan on Windows once a subinterpreter exists; Linux and macOS pass",
    )
    @pytest.mark.timeout(120)
    def test_a_generator_backed_reader_is_scanned_at_one_and_four_threads(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("subinterpreter_gil", SUBINTERPRETER_TESTS)
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        if importlib.util.find_spec(harness.INTERPRETERS) is None:
            pytest.skip(f"{harness.INTERPRETERS} is not available")
        proc = harness._run("", GENERATOR_INPUT)
        report = f"exit status {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        answers = [line.removeprefix(harness.PRECONDITION) for line in proc.stdout.splitlines()]
        answers = [line for line in answers if line in ("0", "1")]
        assert answers, report
        if answers != ["1"]:
            pytest.skip("creating a subinterpreter no longer switches PyGILState_Check off, so nothing here is tested")
        assert proc.returncode == 0, report
        expected = ["[(20000, 199990000)]", "[(20000, 199990000)]", harness.BODY_DONE]
        assert proc.stdout.splitlines()[-3:] == expected, report

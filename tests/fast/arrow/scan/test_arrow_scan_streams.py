"""Python Arrow objects read through the Arrow scan: streams, release, projection, cardinality, types and filters.

The tests read through con.register, the replacement scan and con.execute, and through con._register_arrow_source for
the sources a test defines. Nothing imports duckdb._sources at module level.
"""

from __future__ import annotations

import datetime
import decimal
import gc
import threading
import time
import uuid
import weakref

import _pushdown_helpers
import pytest
from _pushdown_helpers import eventually, scan_of
from conftest import ENGINE_GE_2_1

import duckdb

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

CONSUMED = "has already been consumed and cannot be scanned again"


@pytest.fixture
def con():
    with duckdb.connect() as connection:
        yield connection


@pytest.fixture
def numbers():
    return pa.table({"n": list(range(10)), "s": [str(i) for i in range(10)]})


def rows(con, query, parameters=None):
    return con.execute(query, parameters).fetchall()


def schema(con, name):
    return [(column, kind) for column, kind, *_ in con.execute(f'DESCRIBE "{name}"').fetchall()]


def types(con, name):
    return [kind for _, kind in schema(con, name)]


def columns(con, name):
    return [column for column, _ in schema(con, name)]


def joins_of(nodes):
    """The HASH_JOIN operators of a JSON plan, outermost first."""
    found = []
    for node in nodes:
        if node.get("name") == "HASH_JOIN":
            found.append(node)
        found += joins_of(node.get("children", []))
    return found


def explain(con, query):
    return con.execute(f"EXPLAIN {query}").fetchall()[0][1]


def register_and_read(con, name, obj):
    """Register `obj` and read it, for a refusal that may come from either step."""
    con.register(name, obj)
    return rows(con, f'SELECT * FROM "{name}"')


def reader_over(table):
    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())


def scan_source(mixin, *args, **kwargs):
    """An instance of `mixin` built on the private ArrowScanSource, which is imported only when a test runs."""
    from duckdb._sources import ArrowScanSource

    return type(mixin.__name__, (mixin, ArrowScanSource), {})(*args, **kwargs)


def fragments(predicates):
    from duckdb._sources.nodes import fragment

    return [fragment(predicate) for predicate in predicates]


def greater_than(name, value, type_text):
    from duckdb._sources.nodes import Binary, Col, Lit

    return Binary(">", Col((name,)), Lit(value, type_text))


class Recording:
    """A source over a pyarrow table that remembers which columns each scan asked for."""

    def __init__(self, table, *, narrows=True) -> None:
        super().__init__(table)
        self.narrows = narrows
        self.asked = []

    def __arrow_c_schema__(self) -> object:
        return self.obj.schema.__arrow_c_schema__()

    def stream(self, columns, filters) -> tuple[object, bool]:
        self.asked.append(None if columns is None else list(columns))
        if columns is None or not self.narrows:
            return self.obj.__arrow_c_stream__(), False
        return self.obj.select(list(columns)).__arrow_c_stream__(), True


class Streaming:
    """A source exporting a fresh generator backed stream per call, keeping a weak reference to each generator.

    pyarrow frees a stream's generator when the stream is released, so a dead reference means a released stream.
    """

    def __init__(self, arrow_schema, *, declared=None, fail_at=None, peeked=False) -> None:
        super().__init__(None)
        self.arrow_schema = arrow_schema
        self.declared = declared or arrow_schema
        self.fail_at = fail_at
        self.generators = []
        if not peeked:
            self.__dict__["__arrow_c_schema__"] = self.declared.__arrow_c_schema__

    def stream(self, columns, filters) -> tuple[object, bool]:
        def batches():
            for index in range(3):
                if index == self.fail_at:
                    message = f"boom in batch {index}"
                    raise ValueError(message)
                values = pa.array([index] * 5, self.arrow_schema.field(0).type)
                yield pa.record_batch([values], schema=self.arrow_schema)

        generator = batches()
        self.generators.append(weakref.ref(generator))
        return pa.RecordBatchReader.from_batches(self.arrow_schema, generator).__arrow_c_stream__(), False

    def all_released(self):
        gc.collect()
        return bool(self.generators) and all(ref() is None for ref in self.generators)


class OneArray:
    """A source handing over one freshly allocated pyarrow array per call through `__arrow_c_array__`."""

    def __init__(self, name="value") -> None:
        super().__init__(None)
        self.name = name

    def __arrow_c_schema__(self) -> object:
        return pa.schema([("value", pa.int64())]).__arrow_c_schema__()

    def stream(self, columns, filters) -> tuple[object, bool]:
        values = pc.add(pa.array(range(100_000), pa.int64()), 1)
        return pa.RecordBatch.from_arrays([values], names=[self.name]).__arrow_c_array__(), False


class Filtering:
    """A source over a pyarrow table that applies through pyarrow every predicate the translator models."""

    def __init__(self, table) -> None:
        super().__init__(table)
        self.offered = []
        self.applied = []

    def __arrow_c_schema__(self) -> object:
        return self.obj.schema.__arrow_c_schema__()

    def accepts(self, predicate) -> bool:
        from duckdb._sources.nodes import Untranslatable
        from duckdb._sources.to_arrow import to_arrow

        self.offered.extend(fragments([predicate]))
        try:
            to_arrow(predicate, self.obj.schema)
        except Untranslatable:
            return False
        return True

    def stream(self, columns, filters) -> tuple[object, bool]:
        from duckdb._sources.to_arrow import to_arrow

        self.applied.append(fragments(filters))
        held = self.obj
        for predicate in filters:
            held = held.filter(to_arrow(predicate, self.obj.schema))
        if columns is None:
            return held.__arrow_c_stream__(), False
        return held.select(list(columns)).__arrow_c_stream__(), True


class Noting(Recording):
    """A recording source that keeps every predicate it is offered and refuses them all."""

    def __init__(self, table) -> None:
        super().__init__(table)
        self.offered = []

    def accepts(self, predicate) -> bool:
        self.offered.append(predicate)
        return False


class TestStreams:
    def test_a_reader_is_read_once(self, con, numbers):
        con.register("r", reader_over(numbers))
        assert rows(con, "SELECT count(*) FROM r") == [(10,)]
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            rows(con, "SELECT count(*) FROM r")

    def test_a_capsule_is_read_once(self, con, numbers):
        con.register("s", numbers.__arrow_c_stream__())
        assert rows(con, "SELECT count(*) FROM s") == [(10,)]
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            rows(con, "SELECT count(*) FROM s")

    def test_binding_and_explaining_do_not_consume_a_stream(self, con, numbers):
        con.register("r", reader_over(numbers))
        assert schema(con, "r") == [("n", "BIGINT"), ("s", "VARCHAR")]
        explain(con, "SELECT * FROM r")
        assert rows(con, "SELECT count(*) FROM r") == [(10,)]

    def test_registering_again_makes_it_readable_again(self, con, numbers):
        con.register("r", reader_over(numbers))
        rows(con, "SELECT count(*) FROM r")
        con.register("r", reader_over(numbers))
        assert rows(con, "SELECT count(*) FROM r") == [(10,)]

    def test_two_references_in_sql_fail_at_the_second_read(self, con, numbers):
        con.register("r", reader_over(numbers))
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            rows(con, "SELECT count(*) FROM r a, r b")

    def test_a_cte_in_sql_reads_the_stream_once(self, con, numbers):
        con.register("r", reader_over(numbers))
        assert rows(con, "WITH s AS (SELECT * FROM r) SELECT count(*) FROM s a JOIN s b USING (n)") == [(10,)]

    def test_a_python_error_in_the_stream_fails_the_query(self, con):
        arrow_schema = pa.schema([("a", pa.int64())])

        def batches():
            yield pa.record_batch([pa.array([1, 2])], schema=arrow_schema)
            message = "no more rows for you"
            raise ValueError(message)

        con.register("bad", pa.RecordBatchReader.from_batches(arrow_schema, batches()))
        with pytest.raises(duckdb.InvalidInputException, match=r"registered as 'bad' failed: .*no more rows for you"):
            rows(con, "SELECT sum(a) FROM bad")

    def test_a_stream_error_names_the_variable_or_no_name(self, con):
        arrow_schema = pa.schema([("a", pa.int64())])

        def batches():
            yield pa.record_batch([pa.array([1, 2])], schema=arrow_schema)
            message = "no more rows for you"
            raise ValueError(message)

        bad = pa.RecordBatchReader.from_batches(arrow_schema, batches())  # noqa: F841
        with pytest.raises(
            duckdb.InvalidInputException, match=r"reading the stream of the Python object 'bad' failed: .*no more rows"
        ):
            con.execute("SELECT sum(a) FROM bad").fetchall()
        nameless = con.from_arrow(pa.RecordBatchReader.from_batches(arrow_schema, batches()))
        with pytest.raises(duckdb.InvalidInputException, match=r"reading the Arrow source failed: .*no more rows"):
            nameless.aggregate("sum(a)").fetchall()

    def test_a_python_error_in_the_stream_fails_the_query_under_four_threads(self, con):
        con.execute("SET threads = 4")
        arrow_schema = pa.schema([("a", pa.int64())])

        def batches():
            for i in range(200):
                yield pa.record_batch([pa.array([i])], schema=arrow_schema)
            message = "no more rows for you"
            raise ValueError(message)

        con.register("bad", pa.RecordBatchReader.from_batches(arrow_schema, batches()))
        with pytest.raises(duckdb.InvalidInputException, match=r"registered as 'bad' failed: .*no more rows for you"):
            rows(con, "SELECT sum(a) FROM bad")

    @pytest.mark.parametrize("pull_under_gil", [False, True])
    def test_a_python_error_in_a_pull_on_an_engine_thread_fails_the_query(self, con, pull_under_gil):
        con.execute("SET threads = 4")
        arrow_schema = pa.schema([("a", pa.int64())])
        submitting = threading.get_ident()
        raised_on = []

        def batches():
            for i in range(2000):
                if threading.get_ident() != submitting:
                    raised_on.append(threading.get_ident())
                    message = "no more rows for you"
                    raise ValueError(message)
                yield pa.record_batch([pa.array([i])], schema=arrow_schema)

        class Pulled:
            def __init__(self) -> None:
                super().__init__(None)
                self.pull_under_gil = pull_under_gil

            def __arrow_c_schema__(self) -> object:
                return arrow_schema.__arrow_c_schema__()

            def stream(self, columns, filters) -> tuple[object, bool]:
                return pa.RecordBatchReader.from_batches(arrow_schema, batches()).__arrow_c_stream__(), False

        con._register_arrow_source("t", scan_source(Pulled))
        with pytest.raises(duckdb.InvalidInputException, match=r"registered as 't' failed: .*no more rows for you"):
            rows(con, "SELECT sum(a) FROM t")
        assert raised_on, "every pull ran on the submitting thread, so no engine thread raised"
        assert rows(con, "SELECT 42") == [(42,)]
        con.register("good", pa.table({"a": [1, 2, 3]}))
        assert rows(con, "SELECT sum(a) FROM good") == [(6,)]

    def test_a_stream_read_by_four_threads_is_pulled_one_at_a_time(self, con):
        con.execute("SET threads = 4")
        arrow_schema = pa.schema([("n", pa.int64()), ("s", pa.string())])
        size = 5000
        prebuilt = [
            pa.record_batch(
                [pa.array(range(start, start + size)), pa.array([str(i) for i in range(start, start + size)])],
                schema=arrow_schema,
            )
            for start in range(0, 400 * size, size)
        ]
        pulling = {"now": 0, "overlaps": 0}

        def batches():
            for batch in prebuilt:
                if pulling["now"]:
                    pulling["overlaps"] += 1
                pulling["now"] += 1
                # Held a little, so that a pull another thread started meanwhile would be seen.
                time.sleep(0.0005)
                pulling["now"] -= 1
                yield batch

        con.register("long", pa.RecordBatchReader.from_batches(arrow_schema, batches()))
        total = 400 * size
        assert rows(con, "SELECT count(*), sum(n) FROM long") == [(total, sum(range(total)))]
        assert pulling["overlaps"] == 0

    def test_a_generator_that_builds_each_batch_in_python_is_correct_under_four_threads(self, con):
        con.execute("SET threads = 4")
        arrow_schema = pa.schema([("n", pa.int64())])
        batch_rows = 50
        batch_count = 300

        def batches():
            for i in range(batch_count):
                values = [i * batch_rows + j for j in range(batch_rows)]
                yield pa.record_batch([pa.array(values)], schema=arrow_schema)

        con.register("built", pa.RecordBatchReader.from_batches(arrow_schema, batches()))
        total = batch_count * batch_rows
        assert rows(con, "SELECT count(*), sum(n) FROM built") == [(total, sum(range(total)))]

    def test_a_released_stream_is_refused(self, con, numbers):
        capsule = numbers.__arrow_c_stream__()
        pa.RecordBatchReader._import_from_c_capsule(capsule).read_all()
        with pytest.raises(duckdb.InvalidInputException, match="The ArrowArrayStream was already released"):
            register_and_read(con, "s", capsule)


class TestConcurrency:
    def test_dropping_the_object_during_a_scan_does_not_stop_it(self, con):
        def start_reading(source):
            # The variable lives only in this frame, which ends before the rest is read; on Python 3.12 and earlier
            # the replacement scan's read of f_locals would keep a variable of a frame that is still running alive.
            return con.sql("SELECT i FROM source").to_arrow_reader(1000)

        reader = start_reading(
            pa.Table.from_batches(pa.table({"i": list(range(50_000))}).to_batches(max_chunksize=1_000))
        )
        first = reader.read_next_batch()
        gc.collect()
        rest = reader.read_all()
        assert first.column(0).to_pylist() + rest.column(0).to_pylist() == list(range(50_000))

    @pytest.mark.timeout(60)
    def test_registering_and_querying_on_several_connections_at_once(self, con, numbers):
        workers_count = 4
        rounds = 50
        ready = threading.Barrier(workers_count, timeout=30)
        stop = threading.Event()
        completed = [0] * workers_count
        failures = []

        def churn(index):
            mine = con.cursor()
            try:
                ready.wait()
                for _ in range(rounds):
                    if stop.is_set():
                        return
                    mine.register("shared", numbers)
                    assert rows(mine, "SELECT count(*) FROM shared") == [(10,)]
                    completed[index] += 1
            except BaseException as error:
                failures.append(error)
            finally:
                mine.close()

        # Daemon workers and one shared deadline: a stalled worker fails the test and cannot keep pytest running.
        workers = [threading.Thread(target=churn, args=(index,), daemon=True) for index in range(workers_count)]
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 45
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
        stop.set()
        assert not any(worker.is_alive() for worker in workers), "a worker stalled"
        assert not failures
        assert completed == [rounds] * workers_count

    def test_a_failed_export_does_not_consume_the_object(self, con, numbers):
        class Flaky:
            def __init__(self) -> None:
                self.calls = 0
                self.schema = numbers.schema

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                self.calls += 1
                if self.calls == 1:
                    message = "not yet"
                    raise RuntimeError(message)
                return numbers.__arrow_c_stream__()

        con.register("flaky", Flaky())
        message = "exporting a stream from the object registered as 'flaky' failed: RuntimeError: not yet"
        with pytest.raises(duckdb.InvalidInputException, match=message):
            rows(con, "SELECT count(*) FROM flaky")
        assert rows(con, "SELECT count(*) FROM flaky") == [(10,)]


NUMBERS_SCHEMA = pa.schema([("n", pa.int64())])


class TestArrowRelease:
    """Every schema, array and stream a scan takes from a source is released once, whether the query succeeds or not."""

    def test_a_stream_read_to_the_end_is_released(self, con):
        source = scan_source(Streaming, NUMBERS_SCHEMA)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT count(*), sum(n) FROM t") == [(15, 15)]
        assert source.all_released()

    def test_a_stream_the_query_stops_reading_early_is_released(self, con):
        # pyarrow frees an exhausted stream's generator on its own, so only an unfinished read shows the release.
        source = scan_source(Streaming, NUMBERS_SCHEMA)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT n FROM t LIMIT 1") == [(0,)]
        assert source.all_released()

    def test_a_stream_that_fails_while_read_is_released(self, con):
        source = scan_source(Streaming, NUMBERS_SCHEMA, fail_at=1)
        con._register_arrow_source("t", source)
        with pytest.raises(duckdb.InvalidInputException, match="boom in batch 1"):
            rows(con, "SELECT * FROM t")
        assert eventually(source.all_released)

    def test_a_stream_refused_when_the_scan_opens_is_released(self, con):
        source = scan_source(Streaming, pa.schema([("m", pa.int64())]), declared=NUMBERS_SCHEMA)
        con._register_arrow_source("t", source)
        with pytest.raises(duckdb.InvalidInputException, match="answered with column 'm' at position 0 where 'n'"):
            rows(con, "SELECT * FROM t")
        assert source.all_released()

    def test_a_kept_stream_refused_before_a_row_is_read_is_still_there_for_the_next_query(self, con):
        data = pa.table({"n": [1, 2, 3]})

        class Kept:
            """A read once source over one stored stream, declaring the schema it was built with."""

            one_shot = True

            def __init__(self, capsule, declared) -> None:
                super().__init__(capsule)
                self.declared = declared

            def __arrow_c_schema__(self) -> object:
                return self.declared.__arrow_c_schema__()

            def stream(self, columns, filters) -> tuple[object, bool]:
                return self.obj, False

        # A source keeps its first schema for its whole life, so the second read wraps the same stream afresh.
        capsule = data.__arrow_c_stream__()
        con._register_arrow_source("s", scan_source(Kept, capsule, pa.schema([("wrong", pa.int64())])))
        with pytest.raises(duckdb.InvalidInputException, match="answered with column 'n' at position 0 where 'wrong'"):
            rows(con, "SELECT * FROM s")
        con._register_arrow_source("s", scan_source(Kept, capsule, data.schema))
        assert rows(con, "SELECT * FROM s") == [(1,), (2,), (3,)]
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            rows(con, "SELECT * FROM s")

    def test_a_stream_only_peeked_for_its_schema_is_released(self, con):
        source = scan_source(Streaming, NUMBERS_SCHEMA, peeked=True)
        con._register_arrow_source("t", source)
        assert schema(con, "t") == [("n", "BIGINT")]
        assert source.all_released()
        assert rows(con, "SELECT sum(n) FROM t") == [(15,)]
        assert source.all_released()

    def test_a_schema_the_engine_cannot_import_releases_its_stream(self, con):
        source = scan_source(Streaming, pa.schema([("h", pa.float16())]), peeked=True)
        with pytest.raises(duckdb.NotImplementedException, match="Unsupported Internal Arrow Type"):
            con._register_arrow_source("t", source)
        assert source.all_released()

    def test_an_array_capsule_released_already_is_refused_after_its_schema_was_taken(self, con):
        values = pa.array([1, 2, 3])

        class HalfReleased:
            type = pa.int64()

            def __arrow_c_array__(self, requested_schema=None) -> tuple[object, object]:
                arrow_schema, array = values.__arrow_c_array__()
                other_schema, _ = values.__arrow_c_array__()
                pa.Array._import_from_c_capsule(other_schema, array)
                return arrow_schema, array

        con.register("t", HalfReleased())
        with pytest.raises(duckdb.InvalidInputException, match="'arrow_array' capsule of the object registered as 't'"):
            rows(con, "SELECT * FROM t")
        assert rows(con, "SELECT 42") == [(42,)]

    @pytest.mark.parametrize("refused", [False, True], ids=["read", "refused"])
    def test_a_single_array_is_released(self, con, refused):
        con._register_arrow_source("t", scan_source(OneArray, "other" if refused else "value"))
        gc.collect()
        before = pa.total_allocated_bytes()
        for _ in range(3):
            if refused:
                with pytest.raises(duckdb.InvalidInputException, match="answered with column 'other'"):
                    rows(con, "SELECT * FROM t")
            else:
                assert rows(con, "SELECT count(*), max(value) FROM t") == [(100_000, 100_000)]
        gc.collect()
        assert pa.total_allocated_bytes() == before


class TestLifetimes:
    def test_an_object_serves_a_second_database_after_the_first_closes(self, numbers):
        first = duckdb.connect()
        first.register("kept", numbers)
        assert rows(first, "SELECT count(*) FROM kept") == [(10,)]
        first.close()
        gc.collect()
        with duckdb.connect() as second:
            second.register("kept", numbers)
            assert rows(second, "SELECT sum(n) FROM kept") == [(45,)]

    def test_registrations_do_not_survive_reopening_a_file(self, numbers, tmp_path):
        path = tmp_path / "db.duckdb"
        with duckdb.connect(path) as first:
            first.register("only_while_open", numbers)
            assert rows(first, "SELECT count(*) FROM only_while_open") == [(10,)]
            assert rows(first, "SELECT count(*) FROM duckdb_tables()") == [(0,)]
        with duckdb.connect(path) as reopened:
            assert rows(reopened, "SELECT count(*) FROM duckdb_tables()") == [(0,)]
            with pytest.raises(duckdb.CatalogException):
                rows(reopened, "SELECT * FROM only_while_open")

    def test_a_register_query_unregister_loop_leaks_nothing(self, con):
        loop_table = pa.table({"n": list(range(50_000)), "s": [str(i) for i in range(50_000)]})

        def cycle():
            con.register("loop_table_name", loop_table)
            assert rows(con, "SELECT count(*) FROM loop_table_name") == [(50_000,)]
            con.unregister("loop_table_name")

        for _ in range(5):
            cycle()
        gc.collect()
        start_bytes = pa.total_allocated_bytes()
        start_objects = len(gc.get_objects())
        for _ in range(100):
            cycle()
        gc.collect()
        assert pa.total_allocated_bytes() == start_bytes
        assert len(gc.get_objects()) - start_objects < 100


class TestProjection:
    def test_only_the_columns_a_query_uses_are_asked_for(self, con, numbers):
        source = scan_source(Recording, numbers)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT s FROM t WHERE n = 3") == [("3",)]
        assert rows(con, "SELECT * FROM t LIMIT 1") == [(0, "0")]
        assert rows(con, "SELECT n FROM t WHERE n = 9") == [(9,)]
        assert rows(con, "SELECT s FROM t LIMIT 1") == [("0",)]
        # Every declared column, in order, is asked for as None; a narrower need names the columns.
        assert source.asked == [None, None, [0], [1]]

    def test_the_requested_order_is_the_output_order(self, con, numbers):
        source = scan_source(Recording, numbers)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT s, n FROM t WHERE n = 4") == [("4", 4)]
        assert rows(con, "SELECT s, n FROM t WHERE s = '5'") == [("5", 5)]
        assert all(asked is None or sorted(asked) == [0, 1] for asked in source.asked)

    def test_a_count_needs_no_columns_but_still_scans(self, con, numbers):
        source = scan_source(Recording, numbers)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT count(*) FROM t") == [(10,)]
        assert len(source.asked) == 1

    def test_a_source_that_cannot_narrow_is_picked_from(self, con, numbers):
        source = scan_source(Recording, numbers, narrows=False)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT s FROM t WHERE n = 3") == [("3",)]
        assert rows(con, "SELECT s FROM t LIMIT 1") == [("0",)]
        assert rows(con, "SELECT n FROM t WHERE n = 9") == [(9,)]
        assert source.asked == [None, [1], [0]]

    def test_a_permuted_request_is_picked_from_a_full_stream(self, con, numbers):
        source = scan_source(Recording, numbers, narrows=False)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT s, n FROM t WHERE s = '5'") == [("5", 5)]
        assert rows(con, "SELECT n, s FROM t WHERE n = 6") == [(6, "6")]
        assert all(asked is None or sorted(asked) == [0, 1] for asked in source.asked)

    def test_the_plan_shows_the_projection(self, con, numbers):
        con.register("t", numbers)
        assert "Projections: n" in explain(con, "SELECT n FROM t")
        assert "Projections: s" in explain(con, "SELECT s FROM t WHERE s = '1'")

    def test_a_source_that_raises_while_exporting(self, con, numbers):
        class Broken(Recording):
            def stream(self, columns, filters) -> tuple[object, bool]:
                message = "no stream today"
                raise ValueError(message)

        con._register_arrow_source("t", scan_source(Broken, numbers))
        expected = "exporting a stream from the object registered as 't' failed: ValueError: no stream today"
        with pytest.raises(duckdb.InvalidInputException, match=expected):
            rows(con, "SELECT * FROM t")

    def test_a_source_answering_the_wrong_width_is_refused(self, con, numbers):
        class Lying(Recording):
            def stream(self, columns, filters) -> tuple[object, bool]:
                return self.obj.select([0]).__arrow_c_stream__(), False

        con._register_arrow_source("t", scan_source(Lying, numbers))
        with pytest.raises(duckdb.InvalidInputException, match="answered with 1 columns where it declared 2"):
            rows(con, "SELECT s FROM t")

        class Short(Recording):
            def stream(self, columns, filters) -> tuple[object, bool]:
                return self.obj.__arrow_c_stream__(), True

        con._register_arrow_source("t", scan_source(Short, numbers))
        with pytest.raises(duckdb.InvalidInputException, match="answered with 2 columns where 1 were requested"):
            rows(con, "SELECT s FROM t")

    def test_a_source_answering_in_another_order_is_refused(self, con):
        # A view asks for every declared column in order, so the sources are reached as variables: the replacement
        # scan finds them only in the frame running the query, takes them for an export of their own, and adapt
        # passes a source through unchanged.
        wide = pa.table({"n": [100], "s": ["zero"], "x": [0]})

        class Findable(Recording):
            def __arrow_c_stream__(self, requested_schema=None) -> object:
                return self.obj.__arrow_c_stream__(requested_schema)

        class Sorting(Findable):
            def stream(self, columns, filters) -> tuple[object, bool]:
                if columns is None:
                    return self.obj.__arrow_c_stream__(), False
                return self.obj.select(sorted(columns)).__arrow_c_stream__(), True

        sorting = scan_source(Sorting, wide)  # noqa: F841
        with pytest.raises(duckdb.InvalidInputException, match="answered with column 'n' at position 0 where 'x'"):
            con.execute("SELECT x, s, n FROM sorting WHERE x = 0").fetchall()

        class Swapping(Findable):
            def stream(self, columns, filters) -> tuple[object, bool]:
                return self.obj.select([2, 1, 0]).__arrow_c_stream__(), False

        swapping = scan_source(Swapping, wide)  # noqa: F841
        with pytest.raises(duckdb.InvalidInputException, match="answered with column 'x' at position 0 where 'n'"):
            con.execute("SELECT * FROM swapping").fetchall()

    def test_a_source_answering_another_type_is_refused(self, con, numbers):
        class Drifting(Recording):
            def stream(self, columns, filters) -> tuple[object, bool]:
                drifted = self.obj.set_column(0, "n", self.obj.column(0).cast(pa.string()))
                return drifted.__arrow_c_stream__(), False

        con._register_arrow_source("t", scan_source(Drifting, numbers))
        with pytest.raises(duckdb.InvalidInputException, match="'n'"):
            rows(con, "SELECT * FROM t")

    def test_duplicate_names_project_by_position(self, con):
        arrays = [pa.array([1]), pa.array([2]), pa.array([3]), pa.array([4])]
        source = scan_source(Recording, pa.table(arrays, names=["a_1", "a", "a", "a_2"]))
        con._register_arrow_source("dup", source)
        assert rows(con, "SELECT a_2 FROM dup") == [(3,)]
        assert source.asked == [[2]]

    def test_a_stream_source_yields_every_column(self, con, numbers):
        con.register("r", reader_over(numbers))
        assert rows(con, "SELECT s FROM r WHERE n = 2") == [("2",)]


class TestCardinality:
    """An in-memory object that knows its length tells the planner exactly; a stream leaves the planner guessing."""

    def test_an_exact_count_reaches_the_plan(self, con):
        con.register("t", pa.table({"k": list(range(100_000))}))
        assert "~100,000 rows" in explain(con, "SELECT * FROM t")
        con.register("r", reader_over(pa.table({"k": [1, 2, 3]})))
        assert "~1 row" in explain(con, "SELECT * FROM r")

    def test_the_smaller_side_builds_the_hash_whichever_is_written_first(self, con):
        con.register("big", pa.table({"k": list(range(100_000)), "v": ["x"] * 100_000}))
        con.register("small", pa.table({"k": [1, 2, 3], "w": ["a", "b", "c"]}))
        for query in ("SELECT * FROM big JOIN small USING (k)", "SELECT * FROM small JOIN big USING (k)"):
            [join] = joins_of(_pushdown_helpers.explain_physical_plan(con, query))
            # The second child of a hash join is its build side.
            probe, build = (child["extra_info"]["Estimated Cardinality"] for child in join["children"])
            assert (probe, build) == ("100000", "3"), query
            assert len(rows(con, query)) == 3

    def test_a_source_answering_none_leaves_the_estimate_alone(self, con, numbers):
        class Unknown(Recording):
            def rows(self) -> int | None:
                return None

        con._register_arrow_source("t", scan_source(Unknown, numbers))
        assert "~1 row" in explain(con, "SELECT * FROM t")
        assert rows(con, "SELECT count(*) FROM t") == [(10,)]

    def test_a_source_whose_count_fails(self, con, numbers):
        class Failing(Recording):
            def rows(self) -> int | None:
                message = "no idea"
                raise RuntimeError(message)

        expected = "counting the rows of the object registered as 't' failed"
        with pytest.raises(duckdb.InvalidInputException, match=expected):
            con._register_arrow_source("t", scan_source(Failing, numbers))
        assert rows(con, "SELECT 42") == [(42,)]

    def test_a_length_that_is_not_a_row_count_is_not_trusted(self, con, numbers):
        from duckdb._sources import adapt

        class Mapping:
            def __init__(self) -> None:
                self.columns = {"n": numbers["n"], "s": numbers["s"]}

            def __len__(self) -> int:
                return len(self.columns)

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                return numbers.__arrow_c_stream__()

        assert adapt(Mapping()).rows() is None
        con.register("m", Mapping())
        assert "~1 row" in explain(con, "SELECT * FROM m")
        assert rows(con, "SELECT count(*) FROM m") == [(10,)]


class TestArrays:
    """An object exporting one array is one batch; a plain array or a stream of them is a single column."""

    def test_a_plain_array_is_one_column_named_value(self, con):
        con.register("a", pa.array([1, 2, None]))
        assert schema(con, "a") == [("value", "BIGINT")]
        assert rows(con, "SELECT value FROM a") == [(1,), (2,), (None,)]
        assert rows(con, "SELECT sum(value) FROM a") == [(3,)]

    def test_a_struct_array_is_a_table_of_its_fields(self, con):
        array = pa.StructArray.from_arrays([pa.array([1, 2]), pa.array(["a", "b"])], names=["n", "s"])
        con.register("st", array)
        assert schema(con, "st") == [("n", "BIGINT"), ("s", "VARCHAR")]
        assert rows(con, "SELECT s FROM st WHERE n = 2") == [("b",)]
        assert rows(con, "SELECT n, s FROM st ORDER BY n") == [(1, "a"), (2, "b")]

    def test_a_chunked_array_is_a_stream_of_plain_arrays(self, con):
        con.register("c", pa.chunked_array([list(range(3000)), list(range(3000, 5000))]))
        assert schema(con, "c") == [("value", "BIGINT")]
        assert rows(con, "SELECT count(*), sum(value), max(value) FROM c") == [(5000, 12_497_500, 4999)]

    def test_an_array_with_nulls_and_offsets(self, con):
        con.register("s", pa.array(["x", None, "y", "z"]).slice(1, 2))
        assert rows(con, "SELECT value FROM s") == [(None,), ("y",)]

    def test_a_struct_array_projects_by_position(self, con):
        array = pa.StructArray.from_arrays([pa.array([1]), pa.array(["a"]), pa.array([2.5])], names=["n", "s", "f"])
        con.register("st", array)
        assert rows(con, "SELECT f, n FROM st") == [(2.5, 1)]
        assert "Projections: f, n" in explain(con, "SELECT f, n FROM st")

    def test_a_null_struct_row_reads_as_null_fields_through_an_array_and_a_stream(self, con):
        kind = pa.struct([("n", pa.int64()), ("s", pa.string())])
        array = pa.array([{"n": 1, "s": "a"}, None, {"n": 3, "s": None}], kind)
        expected = [(1, "a"), (None, None), (3, None)]
        con.register("one_batch", array)
        assert rows(con, "SELECT n, s FROM one_batch") == expected
        batch = pa.RecordBatch.from_struct_array(array)
        con.register("streamed", pa.RecordBatchReader.from_batches(batch.schema, [batch]))
        assert rows(con, "SELECT n, s FROM streamed") == expected

    @pytest.mark.parametrize(
        ("names", "expected"),
        [
            pytest.param(["", "value"], ["v0", "value"], id="empty_beside_value"),
            pytest.param(["", "v0"], ["v0", "v0_1"], id="empty_colliding_with_its_own_name"),
            pytest.param(["", "", "value"], ["v0", "v1", "value"], id="two_empty"),
            pytest.param(["v0", ""], ["v0", "v1"], id="empty_after_its_name"),
        ],
    )
    def test_empty_field_names_keep_their_engine_names(self, con, names, expected):
        con.register("blank", pa.table([pa.array([i]) for i in range(len(names))], names=names))
        assert columns(con, "blank") == expected
        assert rows(con, "SELECT * FROM blank") == [tuple(range(len(names)))]


class TestRowsAndTypes:
    def test_many_batches_arrive_in_order(self, con):
        big = pa.table({"i": list(range(100_000))})
        batched = pa.Table.from_batches(big.to_batches(max_chunksize=3_000))
        con.register("big", batched)
        assert rows(con, "SELECT count(*), sum(i), min(i), max(i) FROM big") == [(100_000, 4_999_950_000, 0, 99_999)]
        assert rows(con, "SELECT i FROM big LIMIT 3 OFFSET 2_999") == [(2_999,), (3_000,), (3_001,)]
        assert rows(con, "SELECT i FROM big") == [(i,) for i in range(100_000)]

    def test_a_record_batch_is_a_source(self, con, numbers):
        con.register("b", numbers.to_batches()[0])
        assert rows(con, "SELECT count(*) FROM b") == [(10,)]
        assert rows(con, "SELECT count(*) FROM b") == [(10,)]

    def test_an_empty_table_has_its_schema_and_no_rows(self, con):
        con.register("empty", pa.table({"a": pa.array([], pa.int32())}))
        assert schema(con, "empty") == [("a", "INTEGER")]
        assert rows(con, "SELECT * FROM empty") == []

    def test_a_table_without_columns_is_refused(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="must have at least one column"):
            register_and_read(con, "none", pa.table({}))

    def test_the_arrow_types_arrive_as_duckdb_types(self, con):
        typed = pa.table(
            {
                "i8": pa.array([1, None], pa.int8()),
                "u64": pa.array([2**63, None], pa.uint64()),
                "f": pa.array([1.5, None]),
                "s": pa.array(["a", None]),
                "big_s": pa.array(["b", None], pa.large_string()),
                "bin": pa.array([b"\x00\x01", None], pa.binary()),
                "flag": pa.array([True, None]),
                "d": pa.array([datetime.date(2020, 1, 2), None]),
                "ts": pa.array([datetime.datetime(2020, 1, 2, 3, 4, 5, 6), None]),
                "dec": pa.array([decimal.Decimal("1.23"), None], pa.decimal128(10, 2)),
                "lst": pa.array([[1, 2], None]),
                "st": pa.array([{"x": 1, "y": "z"}, None]),
                "dictionary": pa.array(["p", "q"]).dictionary_encode(),
                "m": pa.array([[("k", 1)], None], pa.map_(pa.string(), pa.int64())),
            }
        )
        con.register("typed", typed)
        assert schema(con, "typed") == [
            ("i8", "TINYINT"),
            ("u64", "UBIGINT"),
            ("f", "DOUBLE"),
            ("s", "VARCHAR"),
            ("big_s", "VARCHAR"),
            ("bin", "BLOB"),
            ("flag", "BOOLEAN"),
            ("d", "DATE"),
            ("ts", "TIMESTAMP"),
            ("dec", "DECIMAL(10,2)"),
            ("lst", "BIGINT[]"),
            ("st", "STRUCT(x BIGINT, y VARCHAR)"),
            ("dictionary", "VARCHAR"),
            ("m", "MAP(VARCHAR, BIGINT)"),
        ]
        first, second = rows(con, "SELECT * FROM typed")
        assert first == (
            1,
            2**63,
            1.5,
            "a",
            "b",
            b"\x00\x01",
            True,
            datetime.date(2020, 1, 2),
            datetime.datetime(2020, 1, 2, 3, 4, 5, 6),
            decimal.Decimal("1.23"),
            [1, 2],
            {"x": 1, "y": "z"},
            "p",
            {"k": 1},
        )
        assert second == (None,) * 12 + ("q", None)

    def test_the_scan_shows_in_the_plan(self, con, numbers):
        con.register("numbers_view", numbers)
        assert "PYTHON_ARROW_SCAN" in explain(con, "SELECT * FROM numbers_view")
        # The replacement scan finds a variable only in the frame running the query, so EXPLAIN runs here.
        reader = reader_over(numbers)  # noqa: F841
        assert "PYTHON_ARROW_SCAN" in con.execute("EXPLAIN SELECT * FROM reader").fetchall()[0][1]
        assert "PYTHON_ARROW_SCAN" in con.from_arrow(numbers).explain()

    def test_the_scan_function_cannot_be_called_from_sql(self, con):
        with pytest.raises(duckdb.BinderException, match="python_arrow_scan"):
            rows(con, "SELECT * FROM python_arrow_scan()")


class StreamOnly:
    """An object that counts how often it is asked for a stream."""

    def __init__(self, table) -> None:
        self.table = table
        self.exports = 0

    def __arrow_c_stream__(self, requested_schema=None) -> object:
        self.exports += 1
        return self.table.__arrow_c_stream__()


class WithSchemaAttribute(StreamOnly):
    @property
    def schema(self) -> object:
        return self.table.schema


class WithSchemaDunder(StreamOnly):
    def __arrow_c_schema__(self) -> object:
        return self.table.schema.__arrow_c_schema__()


class TestSchemaProbing:
    """Binding takes a schema, never a stream, so a source that exports once is not spent by planning."""

    def test_a_schema_attribute_answers_binding(self, con, numbers):
        source = WithSchemaAttribute(numbers)
        con.register("src", source)
        assert schema(con, "src") == [("n", "BIGINT"), ("s", "VARCHAR")]
        explain(con, "SELECT * FROM src")
        assert source.exports == 0
        assert rows(con, "SELECT count(*) FROM src") == [(10,)]
        assert source.exports == 1

    def test_the_schema_dunder_is_preferred(self, con, numbers):
        source = WithSchemaDunder(numbers)
        con.register("src", source)
        assert schema(con, "src") == [("n", "BIGINT"), ("s", "VARCHAR")]
        assert source.exports == 0

    def test_without_either_the_schema_costs_an_export_per_bind(self, con, numbers):
        source = StreamOnly(numbers)
        con.register("src", source)
        registered = source.exports
        assert schema(con, "src") == [("n", "BIGINT"), ("s", "VARCHAR")]
        assert source.exports == registered + 1
        for execution in (1, 2):
            assert rows(con, "SELECT count(*) FROM src") == [(10,)]
            assert source.exports == registered + 1 + 2 * execution

    def test_a_repeatable_object_is_read_twice_in_one_query(self, con, numbers):
        source = WithSchemaAttribute(numbers)
        con.register("src", source)
        assert rows(con, "SELECT count(*) FROM (SELECT * FROM src UNION ALL SELECT * FROM src)") == [(20,)]
        assert source.exports == 2


class TestNamesAndShadowing:
    def test_duplicate_column_names_are_renamed_minimally(self, con):
        arrays = [pa.array([1]), pa.array([2]), pa.array([3]), pa.array([4])]
        con.register("dup", pa.table(arrays, names=["a_1", "a", "a", "a_2"]))
        assert columns(con, "dup") == ["a_1", "a", "a_2", "a_2_1"]
        assert rows(con, "SELECT * FROM dup") == [(1, 2, 3, 4)]

    def test_case_colliding_column_names_are_renamed(self, con):
        con.register("dup", pa.table([pa.array([1]), pa.array([2])], names=["A", "a"]))
        assert columns(con, "dup") == ["A", "a_1"]

    def test_a_schema_capsule_is_not_a_stream(self, con, numbers):
        expected = "Expected a 'arrow_array_stream' PyCapsule, got: arrow_schema"
        with pytest.raises(duckdb.InvalidInputException, match=expected):
            register_and_read(con, "wrong", numbers.schema.__arrow_c_schema__())

    def test_an_object_handing_back_the_wrong_capsule_is_refused(self, con, numbers):
        class WrongKind(WithSchemaAttribute):
            def __arrow_c_stream__(self, requested_schema=None) -> object:
                return self.table.schema.__arrow_c_schema__()

        con.register("wrong", WrongKind(numbers))
        with pytest.raises(duckdb.InvalidInputException, match="did not export an 'arrow_array_stream' capsule but"):
            rows(con, "SELECT * FROM wrong")


class TestExtensionTypes:
    def test_uuid_and_json(self, con):
        identity = uuid.uuid4()
        con.register(
            "ext", pa.table({"u": pa.array([identity.bytes], pa.uuid()), "j": pa.array(['{"a": 1}'], pa.json_())})
        )
        assert schema(con, "ext") == [("u", "UUID"), ("j", "JSON")]
        assert rows(con, "SELECT u, j->>'a' FROM ext") == [(identity, "1")]

    def test_bool8_is_true_for_any_nonzero_byte(self, con):
        stored = pa.array([-1, 0, 1, 2, None], pa.int8())
        con.register("flags", pa.table({"b": pa.ExtensionArray.from_storage(pa.bool8(), stored)}))
        assert schema(con, "flags") == [("b", "BOOLEAN")]
        assert rows(con, "SELECT b FROM flags") == [(True,), (False,), (True,), (True,), (None,)]

    def test_an_unknown_extension_falls_back_to_its_storage(self, con):
        storage = pa.array([7], pa.int32())
        field = pa.field("x", pa.int32(), metadata={"ARROW:extension:name": "mystery.ext"})
        con.register("ext", pa.table([storage], schema=pa.schema([field])))
        assert schema(con, "ext") == [("x", "INTEGER")]
        assert rows(con, "SELECT * FROM ext") == [(7,)]

    def test_opaque_hugeint(self, con):
        big = 2**100
        storage = pa.array([big.to_bytes(16, "little", signed=True)], pa.binary(16))
        opaque = pa.ExtensionArray.from_storage(pa.opaque(pa.binary(16), "hugeint", "DuckDB"), storage)
        con.register("ext", pa.table({"h": opaque}))
        assert schema(con, "ext") == [("h", "HUGEINT")]
        assert rows(con, "SELECT h FROM ext") == [(big,)]

    def test_malformed_metadata_on_a_known_extension_is_an_error(self, con):
        field = pa.field(
            "h",
            pa.binary(16),
            metadata={"ARROW:extension:name": "arrow.opaque", "ARROW:extension:metadata": "not json"},
        )
        malformed = pa.table([pa.array([b"\x00" * 16], pa.binary(16))], schema=pa.schema([field]))
        with pytest.raises(duckdb.Error, match="Failed to parse JSON string"):
            register_and_read(con, "ext", malformed)
        assert rows(con, "SELECT 1") == [(1,)]


class TestTemporalAndBinaryTypes:
    def test_time_at_every_precision(self, con):
        times = pa.table(
            {
                "s": pa.array([datetime.time(1, 2, 3)], pa.time32("s")),
                "ms": pa.array([datetime.time(1, 2, 3, 4000)], pa.time32("ms")),
                "us": pa.array([datetime.time(1, 2, 3, 4)], pa.time64("us")),
                "ns": pa.array([1_500], pa.time64("ns")),
            }
        )
        con.register("times", times)
        assert schema(con, "times") == [("s", "TIME"), ("ms", "TIME"), ("us", "TIME"), ("ns", "TIME_NS")]
        assert rows(con, "SELECT * FROM times") == [
            (datetime.time(1, 2, 3), datetime.time(1, 2, 3, 4000), datetime.time(1, 2, 3, 4), datetime.time(0, 0, 0, 1))
        ]

    def test_timestamps_keep_their_unit_and_zone(self, con):
        utc = datetime.timezone.utc
        stamps = pa.table(
            {
                "s": pa.array([1], pa.timestamp("s")),
                "ms": pa.array([1], pa.timestamp("ms")),
                "ns": pa.array([1_500], pa.timestamp("ns")),
                "tz": pa.array([datetime.datetime(2020, 1, 1, tzinfo=utc)], pa.timestamp("us", "UTC")),
                "ny": pa.array([1], pa.timestamp("ms", "America/New_York")),
            }
        )
        con.register("stamps", stamps)
        assert schema(con, "stamps") == [
            ("s", "TIMESTAMP_S"),
            ("ms", "TIMESTAMP_MS"),
            ("ns", "TIMESTAMP_NS"),
            ("tz", "TIMESTAMP WITH TIME ZONE"),
            ("ny", "TIMESTAMP WITH TIME ZONE"),
        ]
        epoch = datetime.datetime(1970, 1, 1)
        assert rows(con, "SELECT * FROM stamps") == [
            (
                epoch + datetime.timedelta(seconds=1),
                epoch + datetime.timedelta(milliseconds=1),
                epoch + datetime.timedelta(microseconds=1),
                datetime.datetime(2020, 1, 1, tzinfo=utc),
                datetime.datetime(1970, 1, 1, 0, 0, 0, 1000, tzinfo=utc),
            )
        ]

    def test_a_coarse_timestamp_beyond_microseconds_is_unrepresentable_not_wrapped(self, con):
        far = pa.table({"s": pa.array([2**62], pa.timestamp("s")), "ms": pa.array([2**62], pa.timestamp("ms"))})
        con.register("far", far)
        with pytest.raises(duckdb.ConversionException):
            rows(con, "SELECT s FROM far")
        with pytest.raises(duckdb.ConversionException):
            rows(con, "SELECT ms FROM far")

    @pytest.mark.parametrize("unit", ["s", "ms"])
    @pytest.mark.parametrize("target", ["VARCHAR", "TIMESTAMPTZ"])
    @pytest.mark.xfail(
        strict=True,
        raises=duckdb.InternalException,
        reason="the engine's cast of a coarse timestamp beyond microseconds throws an internal error",
    )
    def test_a_coarse_timestamp_beyond_microseconds_try_casts_to_null(self, con, unit, target):
        con.register("far", pa.table({"t": pa.array([2**62], pa.timestamp(unit))}))
        assert rows(con, f"SELECT TRY_CAST(t AS {target}) FROM far") == [(None,)]
        assert rows(con, "SELECT 42") == [(42,)]

    @pytest.mark.xfail(
        not ENGINE_GE_2_1,
        strict=True,
        raises=duckdb.InternalException,
        reason=(
            "the 2.0 engine's text form of a timestamp before the first midnight of its range throws an internal error"
        ),
    )
    def test_a_timestamp_before_the_first_midnight_of_the_range_reads_as_text(self, con):
        con.register("early", pa.table({"t": pa.array([-106_751_991 * 86_400_000_000 - 1], pa.timestamp("us"))}))
        assert rows(con, "SELECT TRY_CAST(t AS VARCHAR) FROM early") == [("290309-12-21 (BC) 23:59:59.999999",)]
        assert rows(con, "SELECT 42") == [(42,)]

    def test_the_first_day_of_nanoseconds_reads_as_text(self, con):
        # Nanoseconds begin during 1677-09-21, whose midnight int64 cannot count.
        counts = [-(2**63), -106_751 * 86_400_000_000_000 - 1]
        con.register("early", pa.table({"t": pa.array(counts, pa.timestamp("ns"))}))
        assert rows(con, "SELECT t::VARCHAR FROM early") == [
            ("1677-09-21 00:12:43.145224192",),
            ("1677-09-21 23:59:59.999999999",),
        ]

    def test_durations_and_intervals(self, con):
        spans = pa.table(
            {
                "s": pa.array([1], pa.duration("s")),
                "ms": pa.array([1], pa.duration("ms")),
                "us": pa.array([1], pa.duration("us")),
                "ns": pa.array([1_500], pa.duration("ns")),
                "mdn": pa.array([pa.MonthDayNano([1, 2, 3_000])], pa.month_day_nano_interval()),
            }
        )
        con.register("spans", spans)
        assert types(con, "spans") == ["INTERVAL"] * 5
        assert rows(con, "SELECT * FROM spans") == [
            (
                datetime.timedelta(seconds=1),
                datetime.timedelta(milliseconds=1),
                datetime.timedelta(microseconds=1),
                datetime.timedelta(microseconds=1),
                datetime.timedelta(days=32, microseconds=3),
            )
        ]

    def test_a_duration_beyond_microseconds_is_a_conversion_error(self, con):
        con.register("spans", pa.table({"s": pa.array([2**62], pa.duration("s"))}))
        with pytest.raises(duckdb.ConversionException, match="Could not convert Interval to Microsecond"):
            rows(con, "SELECT * FROM spans")

    def test_binary_and_date_variants(self, con):
        last = datetime.date(9999, 12, 31)
        variants = pa.table(
            {
                "fixed": pa.array([b"ab"], pa.binary(2)),
                "large": pa.array([b"x"], pa.large_binary()),
                "d64": pa.array([last], pa.date64()),
                "d32": pa.array([last], pa.date32()),
            }
        )
        con.register("bytes_and_days", variants)
        assert types(con, "bytes_and_days") == ["BLOB", "BLOB", "DATE", "DATE"]
        assert rows(con, "SELECT * FROM bytes_and_days") == [(b"ab", b"x", last, last)]


class TestUnsupportedAndUnusualSchemas:
    """Registering or first reading an object reports what the importer cannot take."""

    def test_decimal256_is_refused(self, con):
        wide = pa.table({"d": pa.array([decimal.Decimal("1.5")], pa.decimal256(12, 4))})
        with pytest.raises(duckdb.NotImplementedException, match="Decimal"):
            register_and_read(con, "wide", wide)

    def test_a_dense_union_is_refused(self, con):
        union = pa.UnionArray.from_dense(
            pa.array([0, 1], pa.int8()), pa.array([0, 0], pa.int32()), [pa.array([1]), pa.array(["x"])]
        )
        with pytest.raises(duckdb.NotImplementedException, match="Union"):
            register_and_read(con, "dense", pa.table({"u": union}))

    def test_a_sparse_union_of_struct_arrives(self, con):
        union = pa.UnionArray.from_sparse(
            pa.array([0, 1], pa.int8()), [pa.array([{"a": 1}, {"a": 2}]), pa.array(["x", "y"])]
        )
        con.register("sparse", pa.table({"u": union}))
        assert types(con, "sparse") == ['UNION("0" STRUCT(a BIGINT), "1" VARCHAR)']
        assert rows(con, "SELECT * FROM sparse") == [({"a": 1},), ("y",)]

    def test_null_typed_and_fieldless_struct_columns(self, con):
        con.register("odd", pa.table({"n": pa.array([None, None], pa.null()), "s": pa.array([{}, {}], pa.struct([]))}))
        assert types(con, "odd") == ['"NULL"', "STRUCT"]
        assert rows(con, "SELECT * FROM odd") == [(None, {}), (None, {})]


class TestEncodedLayouts:
    """One smoke test per layout: the importer is the engine's, so only the wiring is checked."""

    def test_run_end_encoding_within_one_batch(self, con):
        con.register("ree", pa.table({"r": pc.run_end_encode(pa.array([1, 1, 2, None, None]))}))
        assert rows(con, "SELECT * FROM ree") == [(1,), (1,), (2,), (None,), (None,)]

    def test_run_end_encoding_across_the_batch_boundary(self, con):
        con.register("ree", pa.table({"r": pc.run_end_encode(pa.array([7] * 3000 + [8] * 3000))}))
        assert rows(con, "SELECT r, count(*) FROM ree GROUP BY r ORDER BY r") == [(7, 3000), (8, 3000)]

    def test_dictionaries_survive_the_batch_boundary(self, con):
        con.register("dict", pa.table({"d": pa.array(["x", "y"] * 3000).dictionary_encode()}))
        assert rows(con, "SELECT d, count(*) FROM dict GROUP BY d ORDER BY d") == [("x", 3000), ("y", 3000)]

    def test_views_and_nested_dictionaries(self, con):
        views = pa.table(
            {
                "l": pa.array([[1, 2], None, []], pa.list_view(pa.int64())),
                "s": pa.array(["a", None, "b" * 20], pa.string_view()),
                "b": pa.array([b"a", None, b"c"], pa.binary_view()),
                "d": pa.array([["p"], ["q"], None]).cast(pa.list_(pa.dictionary(pa.int32(), pa.string()))),
            }
        )
        con.register("views", views)
        assert types(con, "views") == ["BIGINT[]", "VARCHAR", "BLOB", "VARCHAR[]"]
        assert rows(con, "SELECT * FROM views") == [
            ([1, 2], "a", b"a", ["p"]),
            (None, None, None, ["q"]),
            ([], "b" * 20, b"c", None),
        ]

    def test_nested_offsets_past_the_list_boundary_and_a_sliced_table(self, con):
        n = 2**17 + 1
        con.register("deep", pa.table({"s": pa.array([{"l": [i]} for i in range(n)])}).slice(n - 3))
        assert rows(con, "SELECT s.l[1] FROM deep") == [(n - 3,), (n - 2,), (n - 1,)]
        con.register("part", pa.table({"i": list(range(10))}).slice(4, 3))
        assert rows(con, "SELECT i FROM part") == [(4,), (5,), (6,)]


class TestFilters:
    def test_a_source_refuses_by_default_and_the_engine_filters(self, con, numbers):
        con._register_arrow_source("t", scan_source(Recording, numbers))
        assert rows(con, "SELECT s FROM t WHERE n > 7 ORDER BY n") == [("8",), ("9",)]
        scan, filter_above = scan_of(con, "SELECT s FROM t WHERE n > 7")
        assert filter_above
        assert not scan["extra_info"].get("Filters")

    def test_an_accepted_predicate_leaves_the_plan_and_reaches_the_stream(self, con, numbers):
        source = scan_source(Filtering, numbers)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT s FROM t WHERE n > 7 AND s <> '9'") == [("8",)]
        both = sorted(['("n" > 7)', "(\"s\" != '9')"])
        assert sorted(source.offered) == both
        assert [sorted(applied) for applied in source.applied] == [both]
        scan, filter_above = scan_of(con, "SELECT s FROM t WHERE n > 7 AND s <> '9'")
        assert not filter_above
        assert scan["extra_info"].get("Filters")

    def test_a_predicate_names_columns_the_query_does_not_read(self, con, numbers):
        source = scan_source(Filtering, numbers)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT s FROM t WHERE n = 4") == [("4",)]
        assert source.applied == [['("n" = 4)']]
        assert rows(con, "SELECT count(*) FROM t WHERE n > 4") == [(5,)]

    def test_each_reference_carries_its_own_predicates(self, con, numbers):
        source = scan_source(Filtering, numbers)
        con._register_arrow_source("t", source)
        query = "SELECT a.n, b.n FROM t a JOIN t b ON a.n = b.n - 1 WHERE a.n < 2 AND b.n > 1 ORDER BY a.n"
        assert rows(con, query) == [(1, 2)]
        assert sorted(source.applied) == [['("n" < 2)'], ['("n" > 1)']]

    def test_a_prepared_parameter_is_offered_with_each_value(self, con, numbers):
        source = scan_source(Filtering, numbers)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT count(*) FROM t WHERE n > $1", [7]) == [(2,)]
        assert rows(con, "SELECT count(*) FROM t WHERE n > $1", [3]) == [(6,)]
        assert source.applied == [['("n" > 7)'], ['("n" > 3)']]

    def test_the_engine_does_not_filter_again_after_a_promise(self, con, numbers):
        class Lying(Recording):
            def accepts(self, predicate) -> bool:
                return True

        con._register_arrow_source("t", scan_source(Lying, numbers))
        assert rows(con, "SELECT count(*) FROM t WHERE n > 7") == [(10,)]

    def test_a_failing_answer_fails_the_query(self, con, numbers):
        class Broken(Recording):
            def accepts(self, predicate) -> bool:
                message = "no opinion"
                raise RuntimeError(message)

        con._register_arrow_source("t", scan_source(Broken, numbers))
        with pytest.raises(duckdb.InvalidInputException, match="offering a filter to the source registered as 't'"):
            rows(con, "SELECT s FROM t WHERE n > 7")
        assert rows(con, "SELECT count(*) FROM t") == [(10,)]

    def test_an_in_list_is_offered_whole(self, con, numbers):
        source = scan_source(Filtering, numbers)
        con._register_arrow_source("t", source)
        assert rows(con, "SELECT s FROM t WHERE n IN (3, 4) ORDER BY n") == [("3",), ("4",)]
        assert source.applied == [['("n" IN (3, 4))']]

    def test_a_nameless_column_is_offered_by_its_arrow_name(self, con):
        from duckdb._sources.nodes import Binary, Col

        class NotingArray:
            def __init__(self, array) -> None:
                super().__init__(array)
                self.offered = []

            def accepts(self, predicate) -> bool:
                self.offered.append(predicate)
                return False

            def stream(self, columns, filters) -> tuple[object, bool]:
                return self.obj.__arrow_c_array__(), False

        source = scan_source(NotingArray, pa.array([1, 2, 3]))
        con._register_arrow_source("bare", source)
        assert rows(con, "SELECT value FROM bare WHERE value > 1 ORDER BY value") == [(2,), (3,)]
        [offered] = source.offered
        assert isinstance(offered, Binary)
        assert isinstance(offered.left, Col)
        assert offered.left.parts == ("",)
        assert fragments([offered]) == ['("" > 1)']

    def test_a_constant_is_offered_when_a_python_value_holds_it_exactly(self, con):
        moment = datetime.datetime(2024, 5, 6, 7, 8, 9, 123456)
        source = scan_source(
            Noting,
            pa.table(
                {
                    "money": pa.array([decimal.Decimal("1.50"), decimal.Decimal("2.25")], pa.decimal128(10, 2)),
                    "raw": pa.array([b"a", b"b"], pa.binary()),
                    "clock": pa.array([datetime.time(1, 2, 3), datetime.time(4, 5, 6)], pa.time64("us")),
                    "coarse": pa.array([moment, moment], pa.timestamp("ms")),
                    "fine": pa.array([moment, moment], pa.timestamp("ns")),
                    "zoned": pa.array([moment, moment], pa.timestamp("us", tz="UTC")),
                    "tags": pa.array([["a"], ["b"]], pa.list_(pa.string())),
                }
            ),
        )
        con._register_arrow_source("t", source)
        rows(
            con,
            "SELECT count(*) FROM t WHERE money > 1.50 AND raw = '\\x61'::BLOB AND clock < TIME '04:00:00' "
            "AND coarse >= '2024-05-06 07:08:09.123'::TIMESTAMP_MS AND zoned > TIMESTAMPTZ '2024-05-06 00:00:00+00' "
            "AND fine > TIMESTAMP '2024-05-06' AND tags = ['a']",
        )
        carried = {
            predicate.left.parts[0]: (predicate.right.value, predicate.right.type) for predicate in source.offered
        }
        midnight = datetime.datetime(2024, 5, 6, tzinfo=datetime.timezone.utc)
        assert carried == {
            "money": (decimal.Decimal("1.50"), "DECIMAL(10,2)"),
            "raw": (b"a", "BLOB"),
            "clock": (datetime.time(4, 0), "TIME"),
            "coarse": (datetime.datetime(2024, 5, 6, 7, 8, 9, 123000), "TIMESTAMP_MS"),
            "zoned": (int(midnight.timestamp()) * 1_000_000, "TIMESTAMP WITH TIME ZONE"),
        }
        source.offered.clear()
        assert rows(con, "SELECT count(*) FROM t WHERE zoned > TIMESTAMPTZ '2024-05-06 00:00:00+00'") == [(2,)]
        assert fragments(source.offered) == ["(\"zoned\" > TIMESTAMPTZ '2024-05-06 00:00:00+00')"]

    def test_an_infinite_date_is_never_offered(self, con):
        class NotingText(Recording):
            def __init__(self, table) -> None:
                super().__init__(table)
                self.offered = []

            def accepts(self, predicate) -> bool:
                self.offered.extend(fragments([predicate]))
                return False

        edge = pa.table(
            {
                "d": [datetime.date(1, 1, 1), datetime.date(9999, 12, 31)],
                "ts": pa.array([datetime.datetime.min, datetime.datetime.max], pa.timestamp("us")),
            }
        )
        source = scan_source(NotingText, edge)
        con._register_arrow_source("edge", source)
        query = "SELECT count(*) FROM edge WHERE d >= DATE 'infinity' OR ts <= TIMESTAMP '-infinity'"
        assert rows(con, query) == [(0,)]
        query = "SELECT count(*) FROM edge WHERE d < DATE 'infinity' AND ts > TIMESTAMP '-infinity'"
        assert rows(con, query) == [(2,)]
        assert source.offered == []
        assert rows(con, "SELECT count(*) FROM edge WHERE d >= DATE '9999-12-31'") == [(1,)]
        assert source.offered == ["(\"d\" >= DATE '9999-12-31')"]

    def test_a_stream_is_never_asked(self, con, numbers):
        from duckdb._sources import adapt

        source = adapt(reader_over(numbers))
        assert source.one_shot
        assert source.accepts(greater_than("n", 7, "BIGINT")) is False
        con._register_arrow_source("once", source)
        assert rows(con, "SELECT count(*) FROM once WHERE n > 7") == [(2,)]


class TestLayoutsAndLimits:
    def test_a_list_view_with_overlapping_offsets(self, con):
        values = pa.array([1, 2, 3, 4, 5])
        offsets = pa.array([2, 0, 1], pa.int32())
        sizes = pa.array([3, 2, 3], pa.int32())
        con.register("views", pa.table({"l": pa.ListViewArray.from_arrays(offsets, sizes, values)}))
        assert rows(con, "SELECT l FROM views") == [([3, 4, 5],), ([1, 2],), ([2, 3, 4],)]

    def test_narrow_decimals(self, con):
        money = pa.table(
            {
                "d32": pa.array([decimal.Decimal("123.45")], pa.decimal32(5, 2)),
                "d64": pa.array([decimal.Decimal("-1234567890123.456")], pa.decimal64(16, 3)),
            }
        )
        con.register("money", money)
        assert schema(con, "money") == [("d32", "DECIMAL(5,2)"), ("d64", "DECIMAL(16,3)")]
        assert rows(con, "SELECT d32 + 1, d64 FROM money") == [
            (decimal.Decimal("124.45"), decimal.Decimal("-1234567890123.456"))
        ]

    @pytest.mark.parametrize(
        "index", [pa.int8(), pa.uint8(), pa.int16(), pa.uint16(), pa.int32(), pa.uint32(), pa.int64(), pa.uint64()]
    )
    def test_dictionary_index_widths_and_a_null_code(self, con, index):
        codes = pa.array([0, None, 1, 0], index)
        con.register("coded", pa.table({"c": pa.DictionaryArray.from_arrays(codes, pa.array(["x", "y"]))}))
        assert schema(con, "coded") == [("c", "VARCHAR")]
        assert rows(con, "SELECT c FROM coded") == [("x",), (None,), ("y",), ("x",)]

    def test_an_unknown_extension_with_unreadable_metadata_reads_as_its_storage(self, con):
        metadata = {"ARROW:extension:name": "vendor.thing", "ARROW:extension:metadata": "{not json"}
        arrow_schema = pa.schema([pa.field("e", pa.int32(), metadata=metadata)])
        con.register("ext", pa.table({"e": pa.array([1, 2], pa.int32())}, schema=arrow_schema))
        assert schema(con, "ext") == [("e", "INTEGER")]
        assert rows(con, "SELECT sum(e) FROM ext") == [(3,)]

    def test_the_largest_date32_is_the_engines_infinity(self, con):
        """The largest date32 count is the value DuckDB reserves for an infinite date, so it reads as infinity."""
        con.register("edge", pa.table({"d": pa.array([2**31 - 1, 0], pa.date32())}))
        assert rows(con, "SELECT d::VARCHAR, d FROM edge") == [
            ("infinity", datetime.date(9999, 12, 31)),
            ("1970-01-01", datetime.date(1970, 1, 1)),
        ]

    def test_an_interval_beyond_a_timedelta_arrives_exactly(self, con):
        wide = pa.MonthDayNano([2**31 - 1, 0, 0])
        con.register("spans", pa.table({"i": pa.array([wide], pa.month_day_nano_interval())}))
        assert rows(con, "SELECT i::VARCHAR, i = to_months(2147483647) FROM spans") == [
            ("178956970 years 7 months", True)
        ]

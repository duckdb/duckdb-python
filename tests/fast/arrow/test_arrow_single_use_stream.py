"""Scanning single-use Arrow sources more than once.

A pyarrow ``RecordBatchReader`` hands out its batches once: every later ``__arrow_c_stream__()`` call returns a
fresh stream over the same, already advanced reader. A scan of such a reader either returns every row the reader
still holds or raises; it never returns part of them or none. Reusable sources, such as a pyarrow ``Table`` or an
object whose ``__arrow_c_stream__()`` opens a new stream each time, are scanned any number of times.

The engine asks a scan for its stream once per execution of the scan, not when it binds the query, so binding,
``EXPLAIN`` and scans that the optimizer prunes leave a reader untouched.
"""

import gc
import io
import json
import subprocess
import sys
import time
from collections import Counter

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")

CONSUMED = "has already been consumed and cannot be scanned again"

SCHEMA = pa.schema([("id", pa.int64()), ("value", pa.string())])
TABLE = pa.table({"id": [1, 2, 2, None, 5], "value": ["one", None, "two", "four", "five"]}, schema=SCHEMA)
ROWS = Counter([(1, "one"), (2, None), (2, "two"), (None, "four"), (5, "five")])
# Joining on id matches 1 once, the two 2s with each other four times, 5 once, and never NULL.
SELF_JOIN_ROWS = 6

CONCURRENT_SCANS = f"""
import io, json, threading
import duckdb, pyarrow as pa

table = pa.table({{"id": [1, 2, 2, None, 5], "value": ["one", None, "two", "four", "five"]}})
sink = io.BytesIO()
with pa.ipc.new_stream(sink, table.schema) as writer:
    for _ in range(2000):
        writer.write_table(table)
reader = pa.ipc.open_stream(sink.getvalue())
database = duckdb.connect()
barrier = threading.Barrier(2, timeout=30)
results = []

def scan():
    cursor = database.cursor()
    cursor.register("src", reader)
    barrier.wait()
    try:
        results.append(list(cursor.execute("SELECT count(*), count(id), count(value) FROM src").fetchone()))
    except duckdb.Error as error:
        results.append([type(error).__name__, {CONSUMED!r} in str(error)])

threads = [threading.Thread(target=scan) for _ in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
rows = [result for result in results if isinstance(result[0], int)]
errors = [result for result in results if isinstance(result[0], str)]
print(json.dumps({{"rows": rows, "errors": errors}}))
"""


PYARROW_OUT_OF_SYS_MODULES = f"""
import io, sys
from unittest import mock
import duckdb

with mock.patch.dict(sys.modules):
    import pyarrow as pa
    table = pa.table({{"id": [1, 2, 3]}})
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    reader = pa.ipc.open_stream(sink.getvalue())
print("pyarrow listed", "pyarrow" in sys.modules)
con = duckdb.connect()
con.register("src", reader)
print("first", len(con.execute("SELECT * FROM src").fetchall()))
try:
    con.execute("SELECT * FROM src").fetchall()
    print("second returned rows")
except duckdb.InvalidInputException as error:
    print("second raised", {CONSUMED!r} in str(error))
"""

# The three sources that are not pyarrow's reader, classified while pyarrow is out of sys.modules: a class that
# claims the reader's name and module, one whose metaclass makes __mro__ a list, and one with a deep diamond
# hierarchy that a naive walk of the bases visits exponentially often.
NEGATIVES_WITHOUT_PYARROW = """
import sys
from unittest import mock
import duckdb

with mock.patch.dict(sys.modules):
    import pyarrow as pa
    table = pa.table({"id": [1, 2, 3]})
print("pyarrow listed", "pyarrow" in sys.modules)


def stream(self, requested_schema=None):
    return table.__arrow_c_stream__(requested_schema)


class RecordBatchReader:
    __module__ = "pyarrow.lib"
    __qualname__ = "RecordBatchReader"
    __arrow_c_stream__ = stream


class ListMroMeta(type):
    @property
    def __mro__(cls):
        return type.mro(cls)


class ListMroSource(metaclass=ListMroMeta):
    __arrow_c_stream__ = stream


left, right = type("A0", (), {}), type("B0", (), {})
for level in range(25):
    left, right = type(f"A{level + 1}", (left, right), {}), type(f"B{level + 1}", (left, right), {})
Diamond = type("Diamond", (left, right), {"__arrow_c_stream__": stream})

con = duckdb.connect()
for source in (RecordBatchReader(), ListMroSource(), Diamond()):
    rel = con.from_arrow(source)
    print(type(source).__name__, len(rel.fetchall()), len(rel.fetchall()))
"""

CONCURRENT_FIRST_SCANS = f"""
import json, threading
import duckdb, pyarrow as pa

table = pa.table({{"id": [1, 2, 3]}})
readers = [pa.RecordBatchReader.from_batches(table.schema, table.to_batches()) for _ in range(2)]
connections = [duckdb.connect() for _ in range(2)]
barrier = threading.Barrier(2, timeout=30)
results = [None, None]

def scan(index):
    src = readers[index]
    con = connections[index]
    barrier.wait()
    first = len(con.execute("SELECT * FROM src").fetchall())
    try:
        con.execute("SELECT * FROM src").fetchall()
        second = "rows"
    except duckdb.InvalidInputException as error:
        second = {CONSUMED!r} in str(error)
    results[index] = [first, second]

threads = [threading.Thread(target=scan, args=(index,)) for index in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print(json.dumps(results))
"""

MODULE_OUT_OF_SYS_MODULES = f"""
import io, sys
from unittest import mock
import pyarrow as pa

with mock.patch.dict(sys.modules):
    import duckdb
table = pa.table({{"id": [1, 2, 3]}})
sink = io.BytesIO()
with pa.ipc.new_stream(sink, table.schema) as writer:
    writer.write_table(table)
con = duckdb.connect()
con.register("src", pa.ipc.open_stream(sink.getvalue()))
print("first", len(con.execute("SELECT * FROM src").fetchall()))
try:
    con.execute("SELECT * FROM src").fetchall()
    print("second returned rows")
except duckdb.InvalidInputException as error:
    print("second raised", {CONSUMED!r} in str(error))
"""


def _ipc_bytes(batches):
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, SCHEMA) as writer:
        for _ in range(batches):
            writer.write_table(TABLE)
    return sink.getvalue()


def ipc_stream_reader(batches=1):
    return pa.ipc.open_stream(_ipc_bytes(batches))


def from_batches_reader(batches=1):
    return pa.RecordBatchReader.from_batches(SCHEMA, TABLE.to_batches() * batches)


def from_stream_reader(batches=1):
    return pa.RecordBatchReader.from_stream(pa.concat_tables([TABLE] * batches))


PYARROW_READERS = {
    "ipc_stream": ipc_stream_reader,
    "from_batches": from_batches_reader,
    "from_stream": from_stream_reader,
}


@pytest.fixture(params=[*PYARROW_READERS, "duckdb_result"])
def make_reader(request):
    if request.param != "duckdb_result":
        yield PYARROW_READERS[request.param]
        return
    # A reader holds its connection until it is consumed, so each reader gets a connection of its own.
    sources = []

    def duckdb_result_reader(batches=1):
        source = duckdb.connect()
        sources.append(source)
        return source.from_arrow(pa.concat_tables([TABLE] * batches)).to_arrow_reader()

    yield duckdb_result_reader
    for source in sources:
        # The scanner reads ahead on pyarrow's own threads, which can still be pulling a batch from the reader's
        # connection after the query that scanned it raised, so a refused close is retried for a moment
        deadline = time.monotonic() + 5
        while True:
            try:
                source.close()
                break
            except duckdb.ResourceInUseException:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.001)


class Registered:
    def __init__(self, con, source) -> None:
        self.con = con
        con.register("src", source)

    def run(self, sql):
        return self.con.execute(sql).fetchall()


class Replaced:
    def __init__(self, con, source) -> None:
        self.con = con
        self.source = source

    def run(self, sql):
        src = self.source  # noqa: F841
        return self.con.execute(sql).fetchall()


class ReplacedInRelation:
    def __init__(self, con, source) -> None:
        self.con = con
        self.source = source

    def run(self, sql):
        src = self.source  # noqa: F841
        return self.con.sql(sql).fetchall()


ENTRY_POINTS = {"register": Registered, "replacement": Replaced, "relation": ReplacedInRelation}


@pytest.fixture(params=list(ENTRY_POINTS))
def entry_point(request):
    return ENTRY_POINTS[request.param]


def arrow_scans(plan):
    return plan.count("PYTHON_ARROW_SCAN")


class ReusableIPCStream:
    """Reopens its serialized stream on every request, so every scan reads every row."""

    def __init__(self, table) -> None:
        sink = io.BytesIO()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        self.buffer = sink.getvalue()

    def __arrow_c_stream__(self, requested_schema=None):  # noqa: ANN204
        return pa.ipc.open_stream(self.buffer).__arrow_c_stream__(requested_schema)


class ExportFailsOnce(pa.ipc.RecordBatchStreamReader):
    """Its first stream export fails before it hands out any batch."""

    def __init__(self, source) -> None:
        super().__init__(source)
        self.exports = 0

    def __arrow_c_stream__(self, requested_schema=None):  # noqa: ANN204
        self.exports += 1
        if self.exports == 1:
            msg = "the first export fails"
            raise RuntimeError(msg)
        return super().__arrow_c_stream__(requested_schema)


class ListMroMeta(type):
    @property
    def __mro__(cls):  # noqa: ANN204
        return type.mro(cls)


class ListMroSource(metaclass=ListMroMeta):
    """A reusable source whose metaclass makes ``__mro__`` a list."""

    schema = TABLE.schema

    def __arrow_c_stream__(self, requested_schema=None):  # noqa: ANN204
        return TABLE.__arrow_c_stream__(requested_schema)


class UnhashableReader(pa.ipc.RecordBatchStreamReader):
    __hash__ = None


class AlwaysEqualReader(pa.ipc.RecordBatchStreamReader):
    def __eq__(self, other) -> bool:
        return True

    def __hash__(self) -> int:
        return 0


def _run_python(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)


def _polars_frame():
    pl = pytest.importorskip("polars")
    return pl.from_arrow(TABLE)


def _polars_lazy_frame():
    return _polars_frame().lazy()


REUSABLE_SOURCES = {
    "table": lambda: TABLE,
    "record_batch": lambda: TABLE.combine_chunks().to_batches()[0],
    "dataset": lambda: ds.dataset(TABLE),
    "dataset_scanner": lambda: ds.dataset(TABLE).scanner(),
    "reusable_stream_class": lambda: ReusableIPCStream(TABLE),
    "polars_frame": _polars_frame,
    "polars_lazy_frame": _polars_lazy_frame,
}


class TestSingleUseReaderScannedOnce:
    def test_single_scan_returns_every_row(self, duckdb_cursor, make_reader, entry_point):
        scan = entry_point(duckdb_cursor, make_reader())
        assert Counter(scan.run("SELECT * FROM src")) == ROWS

    def test_from_arrow_single_fetch_returns_every_row(self, duckdb_cursor, make_reader):
        assert Counter(duckdb_cursor.from_arrow(make_reader()).fetchall()) == ROWS

    def test_binding_does_not_consume(self, duckdb_cursor, make_reader):
        rel = duckdb_cursor.from_arrow(make_reader())
        assert rel.columns == ["id", "value"]
        assert arrow_scans(rel.explain()) == 1
        assert Counter(rel.fetchall()) == ROWS

    def test_explain_does_not_consume(self, duckdb_cursor, make_reader):
        src = make_reader()  # noqa: F841
        plan = duckdb_cursor.execute("EXPLAIN SELECT * FROM src UNION ALL SELECT * FROM src").fetchall()[0][1]
        assert arrow_scans(plan) == 2
        assert Counter(duckdb_cursor.execute("SELECT * FROM src").fetchall()) == ROWS

    def test_pruned_scan_does_not_consume(self, duckdb_cursor, make_reader):
        src = make_reader()  # noqa: F841
        plan = duckdb_cursor.execute("EXPLAIN SELECT * FROM src WHERE false").fetchall()[0][1]
        assert arrow_scans(plan) == 0
        assert duckdb_cursor.execute("SELECT * FROM src WHERE false").fetchall() == []
        assert Counter(duckdb_cursor.execute("SELECT * FROM src").fetchall()) == ROWS

    def test_prepared_statement_scans_once(self, duckdb_cursor, make_reader):
        src = make_reader()  # noqa: F841
        rows = duckdb_cursor.execute("SELECT * FROM src WHERE id > ?", [1]).fetchall()
        assert Counter(rows) == Counter({row: n for row, n in ROWS.items() if row[0] is not None and row[0] > 1})

    def test_materialized_cte_reads_the_reader_once(self, duckdb_cursor, make_reader):
        sql = "WITH x AS MATERIALIZED (FROM src) SELECT * FROM x UNION ALL SELECT * FROM x"
        src = make_reader()
        plan = duckdb_cursor.execute(f"EXPLAIN {sql}").fetchall()[0][1]
        assert arrow_scans(plan) == 1
        assert plan.count("CTE Scan") == 2
        assert Counter(duckdb_cursor.execute(sql).fetchall()) == ROWS + ROWS

        src = make_reader()  # noqa: F841
        join = "WITH x AS MATERIALIZED (FROM src) SELECT * FROM x a JOIN x b USING (id)"
        assert len(duckdb_cursor.execute(join).fetchall()) == SELF_JOIN_ROWS

    def test_relation_reads_a_replacement_scan_once(self, duckdb_cursor, make_reader):
        # A relation turns each Python object it scans into one common table expression of the query.
        src = make_reader()
        rel = duckdb_cursor.sql("SELECT * FROM src UNION ALL SELECT * FROM src")
        plan = rel.explain()
        assert arrow_scans(plan) == 1
        assert plan.count("CTE Scan") == 2
        assert Counter(rel.fetchall()) == ROWS + ROWS

        src = make_reader()  # noqa: F841
        assert len(duckdb_cursor.sql("SELECT * FROM src a JOIN src b USING (id)").fetchall()) == SELF_JOIN_ROWS

    def test_union_all_of_a_relation_reads_the_reader_once(self, duckdb_cursor, make_reader):
        # A relation turns the object it scans into one common table expression, so both sides read that.
        rel = duckdb_cursor.from_arrow(make_reader())
        union = rel.union(rel)
        plan = union.explain()
        assert arrow_scans(plan) == 1
        assert plan.count("CTE Scan") == 2
        assert Counter(union.fetchall()) == ROWS + ROWS

    def test_materialized_relation_is_reusable(self, duckdb_cursor, make_reader):
        rel = duckdb_cursor.from_arrow(make_reader()).materialize()
        assert Counter(rel.fetchall()) == ROWS
        assert Counter(rel.fetchall()) == ROWS


class TestSingleUseReaderScannedTwice:
    @pytest.mark.parametrize("entry", ["register", "replacement"])
    def test_union_all_raises(self, duckdb_cursor, make_reader, entry):
        scan = ENTRY_POINTS[entry](duckdb_cursor, make_reader())
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            scan.run("SELECT id FROM src UNION ALL SELECT id + 1 FROM src")

    @pytest.mark.parametrize("entry", ["register", "replacement"])
    def test_self_join_raises(self, duckdb_cursor, make_reader, entry):
        scan = ENTRY_POINTS[entry](duckdb_cursor, make_reader())
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            scan.run("SELECT * FROM src a JOIN src b USING (id)")

    def test_scalar_subqueries_raise(self, duckdb_cursor, make_reader):
        scan = Replaced(duckdb_cursor, make_reader())
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            scan.run("SELECT (SELECT count(*) FROM src), (SELECT count(*) FROM src)")

    def test_recursive_cte_raises(self, duckdb_cursor, make_reader):
        # The recursive step scans the reader again on every iteration.
        scan = Replaced(duckdb_cursor, make_reader())
        sql = """
            WITH RECURSIVE r(i) AS (
                SELECT 0 UNION ALL SELECT i + 1 FROM r JOIN src ON src.id = 1 WHERE i < 3
            )
            SELECT count(*) FROM r
        """
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            scan.run(sql)

    def test_rerun_raises(self, duckdb_cursor, make_reader, entry_point):
        scan = entry_point(duckdb_cursor, make_reader())
        assert Counter(scan.run("SELECT * FROM src")) == ROWS
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            scan.run("SELECT * FROM src")
        assert duckdb_cursor.execute("SELECT 42").fetchall() == [(42,)]

    def test_relation_fetched_twice_raises(self, duckdb_cursor, make_reader):
        rel = duckdb_cursor.from_arrow(make_reader())
        assert Counter(rel.fetchall()) == ROWS
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            rel.fetchall()

    # Not with duckdb_result: its source connection can stay busy right after a LIMIT scan, single-use or not.
    @pytest.mark.parametrize("make_reader", list(PYARROW_READERS.values()), ids=list(PYARROW_READERS))
    def test_limit_then_full_scan_raises(self, duckdb_cursor, make_reader, entry_point):
        scan = entry_point(duckdb_cursor, make_reader(batches=400))
        assert len(scan.run("SELECT * FROM src LIMIT 1")) == 1
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            scan.run("SELECT * FROM src")

    def test_explain_analyze_counts_as_a_scan(self, duckdb_cursor, make_reader):
        scan = Replaced(duckdb_cursor, make_reader())
        scan.run("EXPLAIN ANALYZE SELECT * FROM src")
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            scan.run("SELECT * FROM src")

    def test_show_counts_as_a_scan(self, duckdb_cursor, make_reader, capsys):
        rel = duckdb_cursor.from_arrow(make_reader())
        rel.show()
        assert "five" in capsys.readouterr().out
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            rel.fetchall()

    def test_second_connection_raises(self, duckdb_cursor, make_reader):
        src = make_reader()  # noqa: F841
        assert Counter(duckdb_cursor.execute("SELECT * FROM src").fetchall()) == ROWS
        other = duckdb.connect()
        try:
            with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
                other.execute("SELECT * FROM src").fetchall()
        finally:
            other.close()

    def test_concurrent_scans_let_one_scan_read_every_row(self):
        # Two scans reading one reader at the same time can deadlock, so they run in a process of their own.
        try:
            completed = subprocess.run(
                [sys.executable, "-c", CONCURRENT_SCANS], capture_output=True, text=True, timeout=60, check=False
            )
            reports = [line for line in completed.stdout.splitlines() if line.startswith("{")]
            if completed.returncode != 0:
                outcome = f"exit status {completed.returncode}: {completed.stderr[-2000:]}"
            else:
                outcome = json.loads(reports[-1]) if reports else completed.stderr[-2000:]
        except subprocess.TimeoutExpired:
            outcome = "the two scans did not finish within 60 seconds"
        assert outcome == {"rows": [[5 * 2000, 4 * 2000, 4 * 2000]], "errors": [["InvalidInputException", True]]}

    def test_other_entry_points_raise(self, duckdb_cursor, make_reader):
        src = make_reader()
        assert Counter(duckdb_cursor.from_arrow(src).fetchall()) == ROWS
        duckdb_cursor.register("again", src)
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            duckdb_cursor.execute("SELECT * FROM again").fetchall()
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            duckdb_cursor.execute("SELECT * FROM src").fetchall()

    def test_failed_export_does_not_consume(self, duckdb_cursor):
        src = ExportFailsOnce(_ipc_bytes(1))  # noqa: F841
        with pytest.raises(duckdb.Error, match="the first export fails"):
            duckdb_cursor.execute("SELECT * FROM src").fetchall()
        assert Counter(duckdb_cursor.execute("SELECT * FROM src").fetchall()) == ROWS
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            duckdb_cursor.execute("SELECT * FROM src").fetchall()

    def test_empty_reader_scanned_twice_raises(self, duckdb_cursor):
        scan = Replaced(duckdb_cursor, pa.RecordBatchReader.from_batches(SCHEMA, []))
        assert scan.run("SELECT * FROM src") == []
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            scan.run("SELECT * FROM src")

    def test_default_connection_union_all_raises(self):
        sql = "SELECT id FROM arrow_stream UNION ALL SELECT id + 1 FROM arrow_stream"
        with pa.ipc.open_stream(_ipc_bytes(1)) as reader:
            duckdb.register("arrow_stream", reader)
            try:
                with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
                    duckdb.sql(sql).fetchall()
                with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
                    duckdb.sql(sql).fetchall()
            finally:
                duckdb.unregister("arrow_stream")


class TestConsumedReaderRecord:
    def test_reader_at_a_reused_address_is_new(self, duckdb_cursor):
        consumed_addresses = set()
        for _ in range(1000):
            src = from_batches_reader()
            reused = id(src) in consumed_addresses
            assert Counter(duckdb_cursor.execute("SELECT * FROM src").fetchall()) == ROWS
            if reused:
                return
            consumed_addresses.add(id(src))
            del src
            gc.collect()
        pytest.skip("CPython placed no new reader at the address of a consumed one within 1000 readers")

    def test_unhashable_reader_scans_once(self, duckdb_cursor):
        scan = Replaced(duckdb_cursor, UnhashableReader(_ipc_bytes(1)))
        assert Counter(scan.run("SELECT * FROM src")) == ROWS
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            scan.run("SELECT * FROM src")

    def test_equal_readers_are_tracked_apart(self, duckdb_cursor):
        first = Replaced(duckdb_cursor, AlwaysEqualReader(_ipc_bytes(1)))
        second = Replaced(duckdb_cursor, AlwaysEqualReader(_ipc_bytes(1)))
        assert first.source == second.source
        assert Counter(first.run("SELECT * FROM src")) == ROWS
        assert Counter(second.run("SELECT * FROM src")) == ROWS
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            first.run("SELECT * FROM src")

    def test_reader_made_while_pyarrow_was_out_of_sys_modules(self):
        completed = _run_python(PYARROW_OUT_OF_SYS_MODULES)
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert "pyarrow listed False" in completed.stdout
        assert "first 3" in completed.stdout
        assert "second raised True" in completed.stdout

    def test_sources_that_are_not_the_reader_while_pyarrow_is_out_of_sys_modules(self):
        completed = _run_python(NEGATIVES_WITHOUT_PYARROW)
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert "pyarrow listed False" in completed.stdout
        assert "RecordBatchReader 3 3" in completed.stdout
        assert "ListMroSource 3 3" in completed.stdout
        assert "Diamond 3 3" in completed.stdout

    def test_concurrent_first_scans_in_a_fresh_process(self):
        completed = _run_python(CONCURRENT_FIRST_SCANS)
        assert completed.returncode == 0, completed.stderr[-2000:]
        reports = [line for line in completed.stdout.splitlines() if line.startswith("[")]
        assert json.loads(reports[-1]) == [[3, True], [3, True]]

    def test_scan_after_the_module_left_sys_modules(self):
        # Mocking sys.modules around the import leaves the extension loaded but unlisted; importing it again breaks.
        completed = _run_python(MODULE_OUT_OF_SYS_MODULES)
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert "first 3" in completed.stdout
        assert "second raised True" in completed.stdout
        assert "already registered" not in completed.stderr
        assert "nanobind: leaked" not in completed.stderr


class TestSingleUseSourcesThatAlreadyRaise:
    def test_scanner_from_batches_rescan_raises(self, duckdb_cursor):
        src = ds.Scanner.from_batches(ipc_stream_reader())  # noqa: F841
        assert Counter(duckdb_cursor.execute("SELECT * FROM src").fetchall()) == ROWS
        with pytest.raises(duckdb.InvalidInputException, match="OneShotFragment was already scanned"):
            duckdb_cursor.execute("SELECT * FROM src").fetchall()

    def test_scanner_from_batches_union_all_raises(self, duckdb_cursor):
        src = ds.Scanner.from_batches(ipc_stream_reader())  # noqa: F841
        with pytest.raises(duckdb.InvalidInputException, match="OneShotFragment was already scanned"):
            duckdb_cursor.execute("SELECT * FROM src UNION ALL SELECT * FROM src").fetchall()

    def test_bare_capsule_union_all_raises(self, duckdb_cursor):
        src = ipc_stream_reader().__arrow_c_stream__()  # noqa: F841
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            duckdb_cursor.execute("SELECT * FROM src UNION ALL SELECT * FROM src").fetchall()

    def test_registered_bare_capsule_rerun_raises(self, duckdb_cursor):
        duckdb_cursor.register("src", ipc_stream_reader().__arrow_c_stream__())
        assert Counter(duckdb_cursor.execute("SELECT * FROM src").fetchall()) == ROWS
        with pytest.raises(duckdb.InvalidInputException, match=CONSUMED):
            duckdb_cursor.execute("SELECT * FROM src").fetchall()


class TestReusableSourcesScannedManyTimes:
    @pytest.fixture(params=list(REUSABLE_SOURCES))
    def source(self, request):
        return REUSABLE_SOURCES[request.param]()

    def test_union_all(self, duckdb_cursor, source, entry_point):
        scan = entry_point(duckdb_cursor, source)
        assert Counter(scan.run("SELECT * FROM src UNION ALL SELECT * FROM src")) == ROWS + ROWS

    def test_self_join(self, duckdb_cursor, source, entry_point):
        scan = entry_point(duckdb_cursor, source)
        assert len(scan.run("SELECT * FROM src a JOIN src b USING (id)")) == SELF_JOIN_ROWS

    def test_rerun(self, duckdb_cursor, source, entry_point):
        scan = entry_point(duckdb_cursor, source)
        for _ in range(3):
            assert Counter(scan.run("SELECT * FROM src")) == ROWS

    def test_recursive_cte(self, duckdb_cursor, source):
        scan = Replaced(duckdb_cursor, source)
        sql = """
            WITH RECURSIVE r(i) AS (
                SELECT 0 UNION ALL SELECT i + 1 FROM r JOIN src ON src.id = 1 WHERE i < 3
            )
            SELECT count(*) FROM r
        """
        assert scan.run(sql) == [(4,)]

    def test_source_named_like_the_pyarrow_reader(self, duckdb_cursor):
        class RecordBatchReader:
            __module__ = "pyarrow.lib"
            __qualname__ = "RecordBatchReader"
            schema = TABLE.schema

            def __arrow_c_stream__(self, requested_schema=None):  # noqa: ANN204
                return TABLE.__arrow_c_stream__(requested_schema)

        rel = duckdb_cursor.from_arrow(RecordBatchReader())
        assert Counter(rel.fetchall()) == ROWS
        assert Counter(rel.fetchall()) == ROWS

    def test_source_whose_metaclass_overrides_mro(self, duckdb_cursor):
        assert isinstance(ListMroSource.__mro__, list)
        rel = duckdb_cursor.from_arrow(ListMroSource())
        assert Counter(rel.fetchall()) == ROWS
        assert Counter(rel.fetchall()) == ROWS

    @pytest.mark.parametrize("name", [name for name in REUSABLE_SOURCES if name != "polars_lazy_frame"])
    def test_relation_fetched_twice(self, duckdb_cursor, name):
        rel = duckdb_cursor.from_arrow(REUSABLE_SOURCES[name]())
        assert Counter(rel.fetchall()) == ROWS
        assert Counter(rel.fetchall()) == ROWS
        assert Counter(rel.union(rel).fetchall()) == ROWS + ROWS

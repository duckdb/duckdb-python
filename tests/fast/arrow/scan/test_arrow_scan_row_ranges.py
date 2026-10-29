"""Arrow sources whose rows arrive in one large batch, read by python_arrow_scan in row ranges of a row group.

A batch splits into ranges of 122880 rows from its start, and the last range takes a remainder shorter than a range.
The tests count the threads that read a source, check the order of rows across range and batch boundaries, and check
what a split must leave as it is: values, refusals, releases, memory, row groups written and GIL use. Expected rows
come from the numpy, pyarrow or polars values in Python.
"""

from __future__ import annotations

import ctypes
import gc
import json
import re
import sys
import threading
import time

import pytest
from conftest import PANDAS_GE_3

import duckdb

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")
np = pytest.importorskip("numpy")

import scan.test_python_arrow_scan_core as core  # noqa: E402

RANGE_ROWS = 122_880
THREADS = 4
LARGE = 1_000_003


def arange(rows):
    return np.arange(rows, dtype=np.int64)


def install_thread_witness(con):
    seen = set()
    lock = threading.Lock()

    def tid(values):
        ident = threading.get_ident()
        with lock:
            first = ident not in seen
            seen.add(ident)
        # A thread waits on its first vector, so idle threads get the chance to claim the other ranges.
        if first:
            time.sleep(0.05)
        return pa.array(np.full(len(values), ident, dtype=np.int64))

    con.create_function("tid", tid, ["BIGINT"], "BIGINT", type="arrow", side_effects=True)


def first_column(con, name):
    return con.execute(f'DESCRIBE "{name}"').fetchall()[0][0]


def reading_threads(con, name):
    return f'SELECT count(DISTINCT tid("{first_column(con, name)}")) FROM "{name}"'


class StreamOnly:
    def __init__(self, table) -> None:
        self.table = table

    def __arrow_c_stream__(self, requested_schema=None) -> object:
        return self.table.__arrow_c_stream__(requested_schema)


def _pandas():
    return pytest.importorskip("pandas")


def _polars():
    return pytest.importorskip("polars")


ONE_CHUNK_SOURCES = {
    "table": lambda a: pa.table({"a": a}),
    "record_batch": lambda a: pa.record_batch([pa.array(a)], names=["a"]),
    "array": lambda a: pa.array(a),
    "chunked_array": lambda a: pa.chunked_array([a]),
    "struct_array": lambda a: pa.StructArray.from_arrays([pa.array(a)], names=["a"]),
    "reader_with_one_batch": lambda a: pa.RecordBatchReader.from_batches(
        pa.schema([("a", pa.int64())]), [pa.record_batch([pa.array(a)], names=["a"])]
    ),
    "stream_capsule": lambda a: pa.table({"a": a}).__arrow_c_stream__(),
    "third_party_stream": lambda a: StreamOnly(pa.table({"a": a})),
    "scanner_with_one_batch": lambda a: pytest.importorskip("pyarrow.dataset")
    .dataset(pa.table({"a": a}))
    .scanner(batch_size=len(a)),
    "pandas_series": lambda a: _pandas().Series(a, name="a"),
    "pandas_arrow_series": lambda a: _pandas().Series(_pandas().array(a, dtype="int64[pyarrow]"), name="a"),
    "pandas_arrow_frame": lambda a: _pandas().DataFrame({"a": _pandas().array(a, dtype="int64[pyarrow]")}),
    "polars_frame": lambda a: _polars().DataFrame({"a": a}),
    "polars_series": lambda a: _polars().Series("a", a),
    "polars_lazy_frame": lambda a: _polars().DataFrame({"a": a}).lazy(),
}

SERIES_SOURCES = {"pandas_series", "pandas_arrow_series"}
before_pandas_3 = pytest.mark.skipif(not PANDAS_GE_3, reason="before pandas 3 a Series has no Arrow stream")


def _one_chunk_param(name):
    return pytest.param(name, marks=[before_pandas_3] if name in SERIES_SOURCES else [])


class TestThreads:
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("source", [_one_chunk_param(name) for name in ONE_CHUNK_SOURCES])
    def test_a_one_chunk_source_is_read_by_several_threads(self, source):
        obj = ONE_CHUNK_SOURCES[source](arange(LARGE))
        with duckdb.connect(config={"threads": THREADS}) as con:
            install_thread_witness(con)
            con.register("src", obj)
            query = reading_threads(con, "src")
            assert con.execute(query).fetchall()[0][0] > 1

    @pytest.mark.timeout(60)
    def test_a_large_batch_after_a_small_one_is_read_by_several_threads(self):
        schema = pa.schema([("a", pa.int64())])
        batches = [pa.record_batch([pa.array(arange(rows))], schema=schema) for rows in (10, LARGE)]
        with duckdb.connect(config={"threads": THREADS}) as con:
            install_thread_witness(con)
            con.register("src", pa.RecordBatchReader.from_batches(schema, batches))
            query = reading_threads(con, "src")
            # One thread reads the small batch, so only a third thread shows that the large one was split.
            assert con.execute(query).fetchall()[0][0] > 2

    @pytest.mark.timeout(60)
    def test_a_large_batch_of_a_stream_pulled_under_the_gil_is_read_by_several_threads(self):
        source = core.PythonStreamSource(pa.table({"a": arange(LARGE)}), max_chunksize=LARGE)
        with duckdb.connect(config={"threads": THREADS}) as con:
            install_thread_witness(con)
            con._register_arrow_source("src", source)
            query = reading_threads(con, "src")
            assert con.execute(query).fetchall()[0][0] > 1

    @pytest.mark.timeout(60)
    def test_a_numpy_backed_frame_is_read_by_several_threads(self):
        frame = _pandas().DataFrame({"a": arange(LARGE)})
        with duckdb.connect(config={"threads": THREADS}) as con:
            install_thread_witness(con)
            con.register("src", frame)
            query = reading_threads(con, "src")
            assert con.execute(query).fetchall()[0][0] > 1

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("source", ["table", "array"])
    def test_a_batch_shorter_than_two_ranges_is_one_range(self, source):
        obj = ONE_CHUNK_SOURCES[source](arange(2 * RANGE_ROWS - 1))
        with duckdb.connect(config={"threads": THREADS}) as con:
            install_thread_witness(con)
            con.register("src", obj)
            query = reading_threads(con, "src")
            assert con.execute(query).fetchall()[0][0] == 1

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("source", ["table", "array"])
    @pytest.mark.parametrize("rows", [2 * RANGE_ROWS, 3 * RANGE_ROWS - 1], ids=["two_ranges", "two_and_a_remainder"])
    def test_the_last_range_takes_the_remainder(self, source, rows):
        obj = ONE_CHUNK_SOURCES[source](arange(rows))
        with duckdb.connect(config={"threads": THREADS}) as con:
            install_thread_witness(con)
            con.register("src", obj)
            query = reading_threads(con, "src")
            assert con.execute(query).fetchall()[0][0] == 2


BOUNDARY_SIZES = [
    1,
    2047,
    2048,
    RANGE_ROWS - 1,
    RANGE_ROWS,
    RANGE_ROWS + 1,
    2 * RANGE_ROWS - 1,
    2 * RANGE_ROWS,
    300_000,
]


def _batches_of(sizes):
    batches = []
    start = 0
    for size in sizes:
        batches.append(pa.record_batch([pa.array(arange(size) + start)], names=["n"]))
        start += size
    return pa.Table.from_batches(batches)


ORDER_SOURCES = {
    "batches_at_every_boundary_size": lambda: _batches_of(BOUNDARY_SIZES),
    "one_large_batch": lambda: pa.table({"n": arange(LARGE)}),
    "single_array": lambda: pa.StructArray.from_arrays([pa.array(arange(LARGE))], names=["n"]),
    "plain_array": lambda: pa.array(arange(LARGE)),
    "one_chunk_polars_frame": lambda: _polars().DataFrame({"n": arange(LARGE)}),
}


def _total(source):
    return sum(BOUNDARY_SIZES) if source == "batches_at_every_boundary_size" else LARGE


def _as_n(con):
    return f'SELECT "{first_column(con, "src")}" AS n FROM src'


class TestOrder:
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("source", list(ORDER_SOURCES))
    def test_rows_keep_their_order_across_range_and_batch_boundaries(self, source):
        obj = ORDER_SOURCES[source]()
        expected = arange(_total(source))
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.register("src", obj)
            query = _as_n(con)
            assert np.array_equal(con.execute(query).fetchnumpy()["n"], expected)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("source", list(ORDER_SOURCES))
    def test_row_numbers_follow_the_rows_across_range_and_batch_boundaries(self, source):
        obj = ORDER_SOURCES[source]()
        expected = arange(_total(source))
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.register("src", obj)
            query = f"SELECT row_number() OVER () AS r, n FROM ({_as_n(con)})"
            ranked = con.execute(query).fetchnumpy()
            assert np.array_equal(ranked["n"], expected)
            assert np.array_equal(ranked["r"], expected + 1)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("source", list(ORDER_SOURCES))
    def test_a_limit_takes_the_first_rows(self, source):
        obj = ORDER_SOURCES[source]()
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.register("src", obj)
            query = f"{_as_n(con)} LIMIT 5"
            assert con.execute(query).fetchall() == [(i,) for i in range(5)]

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("source", list(ORDER_SOURCES))
    def test_a_table_created_from_the_rows_keeps_their_order(self, source):
        obj = ORDER_SOURCES[source]()
        expected = arange(_total(source))
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.register("src", obj)
            con.execute(f"CREATE TABLE copied AS {_as_n(con)}")
            assert np.array_equal(con.execute("SELECT n FROM copied").fetchnumpy()["n"], expected)

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "offset", [0, 2047, RANGE_ROWS - 2, RANGE_ROWS, 2 * RANGE_ROWS - 1, 5 * RANGE_ROWS + 1, LARGE - 2]
    )
    def test_limit_and_offset_inside_one_large_batch(self, offset):
        table = pa.table({"n": arange(LARGE)})
        expected = [(n,) for n in range(offset, min(offset + 3, LARGE))]
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.register("src", table)
            assert con.execute(f"SELECT n FROM src LIMIT 3 OFFSET {offset}").fetchall() == expected

    @pytest.mark.timeout(60)
    def test_a_large_batch_among_small_and_empty_batches_keeps_its_order(self):
        sizes = [0, 5, 0, 500_000, 0, 1, 2 * RANGE_ROWS, 3, 0]
        table = _batches_of(sizes)
        expected = arange(sum(sizes))
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.register("src", pa.RecordBatchReader.from_batches(table.schema, table.to_batches()))
            assert np.array_equal(con.execute("SELECT n FROM src").fetchnumpy()["n"], expected)


LARGE_BATCH = 300_000
LARGE_BATCHES = 3
DICTIONARY_ROWS = 2_000_000
DICTIONARY_VALUES = 1_000_000


def _dictionary_batches(seed):
    rng = np.random.default_rng(seed)
    for batch in range(LARGE_BATCHES):
        # Each batch has its own dictionary values, so a dictionary reused across batches reads the wrong strings.
        values = np.array([f"b{batch}-{v}" for v in range(50)], dtype=object)
        indices = rng.integers(0, len(values), LARGE_BATCH, dtype=np.int32)
        yield (
            values[indices],
            pa.record_batch([pa.DictionaryArray.from_arrays(pa.array(indices), pa.array(values))], names=["d"]),
        )


def _run_end_batches():
    for batch, run in enumerate((1000, 777, 1)):
        values = batch * 1_000_000 + arange(LARGE_BATCH) // run
        yield values, pa.record_batch([pc.run_end_encode(pa.array(values))], names=["r"])


class TestDictionariesAndRunEnds:
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    def test_dictionaries_that_change_between_large_batches(self, threads):
        expected = [(value,) for decoded, _ in _dictionary_batches(0) for value in decoded]
        schema = pa.schema([("d", pa.dictionary(pa.int32(), pa.string()))])
        reader = pa.RecordBatchReader.from_batches(schema, (batch for _, batch in _dictionary_batches(0)))
        with duckdb.connect(config={"threads": threads}) as con:
            con.register("src", reader)
            assert con.execute("SELECT d FROM src").fetchall() == expected

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    def test_run_ends_that_change_between_large_batches(self, threads):
        expected = np.concatenate([values for values, _ in _run_end_batches()])
        schema = next(_run_end_batches())[1].schema
        reader = pa.RecordBatchReader.from_batches(schema, (batch for _, batch in _run_end_batches()))
        with duckdb.connect(config={"threads": threads}) as con:
            con.register("src", reader)
            assert np.array_equal(con.execute("SELECT r FROM src").fetchnumpy()["r"], expected)

    @pytest.mark.timeout(120)
    def test_a_dictionary_is_decoded_once_per_large_batch(self):
        values = pa.array([f"value-{i:08d}" for i in range(DICTIONARY_VALUES)])
        # In order, since random indices into a large dictionary make the time depend on the machine's caches.
        indices = pa.array(
            (np.arange(DICTIONARY_ROWS, dtype=np.int64) * DICTIONARY_VALUES // DICTIONARY_ROWS).astype(np.int32)
        )
        whole = pa.table({"d": pa.DictionaryArray.from_arrays(indices, values)})
        # Each slice keeps the whole dictionary, which the scan decodes again for every batch because it drops the
        # decoded dictionary when the batch changes.
        split = pa.Table.from_batches(whole.to_batches(max_chunksize=RANGE_ROWS))
        assert whole.column("d").num_chunks == 1
        assert split.column("d").num_chunks == -(-DICTIONARY_ROWS // RANGE_ROWS)
        expected = pc.sum(pc.utf8_length(whole.column("d").cast(pa.string()))).as_py()
        query = "SELECT sum(strlen(d)) FROM {}"

        def timed(con, name):
            start = time.perf_counter()
            con.execute(query.format(name)).fetchall()
            return time.perf_counter() - start

        with duckdb.connect(config={"threads": 1}) as con:
            con._register_arrow_source("whole", core.TableSource(whole))
            con._register_arrow_source("split", core.TableSource(split))
            assert con.execute(query.format("whole")).fetchall() == [(expected,)]
            assert con.execute(query.format("split")).fetchall() == [(expected,)]
            # Alternated, so a slow stretch of the machine falls on both kinds of run alike.
            rounds = [(timed(con, "whole"), timed(con, "split")) for _ in range(3)]
            whole_seconds = min(pair[0] for pair in rounds)
            split_seconds = min(pair[1] for pair in rounds)
        # Decoded once per range, the whole batch takes about as long as the split one; decoded once per batch, an
        # eighth to a quarter as long.
        assert 2 * whole_seconds < split_seconds, (whole_seconds, split_seconds)


SPLIT_ROWS = 250_000


def _keys():
    return arange(SPLIT_ROWS)


def _text(values):
    return pc.cast(pa.array(values), pa.string())


def _null_rows():
    return pa.array(_keys() % 7 == 3)


def _struct_of_struct():
    k = _keys()
    inner = pa.StructArray.from_arrays([pa.array(k), _text(k * 2)], names=["x", "y"])
    return pa.StructArray.from_arrays([inner, pa.array(k)], names=["a", "z"], mask=_null_rows())


def _sparse_union_of_struct():
    k = _keys()
    return pa.UnionArray.from_sparse(
        pa.array(k % 2, pa.int8()), [pa.StructArray.from_arrays([pa.array(k)], names=["x"]), _text(k)], ["st", "s"]
    )


def _struct_of_sparse_union():
    k = _keys()
    union = pa.UnionArray.from_sparse(pa.array(k % 2, pa.int8()), [pa.array(k), _text(k)], ["i", "s"])
    return pa.StructArray.from_arrays([union], names=["u"])


def _list_of_struct():
    offsets = pa.array(np.arange(0, 2 * SPLIT_ROWS + 1, 2, dtype=np.int32))
    return pa.ListArray.from_arrays(
        offsets, pa.StructArray.from_arrays([pa.array(arange(2 * SPLIT_ROWS))], names=["x"])
    )


def _struct_of_list_of_struct():
    return pa.StructArray.from_arrays([_list_of_struct()], names=["l"])


def _map_of_struct():
    offsets = pa.array(np.arange(0, 2 * SPLIT_ROWS + 1, 2, dtype=np.int32))
    keys = pa.array(np.tile(np.array([1, 2], dtype=np.int32), SPLIT_ROWS))
    items = pa.StructArray.from_arrays([pa.array(arange(2 * SPLIT_ROWS))], names=["x"])
    return pa.MapArray.from_arrays(offsets, keys, items)


def _fixed_size_list_of_struct():
    return pa.FixedSizeListArray.from_arrays(
        pa.StructArray.from_arrays([pa.array(arange(2 * SPLIT_ROWS))], names=["x"]), 2
    )


def _struct_of_struct_of_dictionary():
    indices = pa.array((_keys() % 50).astype(np.int32))
    dictionary = pa.DictionaryArray.from_arrays(indices, _text(np.arange(50) * 3))
    return pa.StructArray.from_arrays([pa.StructArray.from_arrays([dictionary], names=["d"])], names=["a"])


def _struct_of_struct_of_run_ends():
    runs = pc.run_end_encode(pa.array(_keys() // 1000))
    return pa.StructArray.from_arrays([pa.StructArray.from_arrays([runs], names=["r"])], names=["a"])


def _list_view():
    k = _keys()
    values = pa.array(arange(2 * SPLIT_ROWS))
    offsets = pa.array(((SPLIT_ROWS - 1 - k) * 2).astype(np.int32))
    sizes = pa.array((k % 3).astype(np.int32))
    return pa.ListViewArray.from_arrays(offsets, sizes, values)


def _as_map(value):
    return None if value is None else dict(value)


def _as_tuple(value):
    return None if value is None else tuple(value)


def _column_layout(make, convert=None):
    def build():
        table = pa.table({"k": _keys(), "c": make()})
        expected = [(row["k"], convert(row["c"]) if convert else row["c"]) for row in table.to_pylist()]
        return table, expected

    return build


def _struct_array_source(offset):
    def build():
        k = _keys()
        array = pa.StructArray.from_arrays([pa.array(k), _text(k)], names=["n", "s"], mask=_null_rows())
        array = array.slice(offset)
        expected = [(None, None) if row is None else (row["n"], row["s"]) for row in array.to_pylist()]
        return array, expected

    return build


def _plain_array_source(offset):
    def build():
        array = pa.array(_keys(), mask=_keys() % 7 == 3).slice(offset)
        return array, [(value,) for value in array.to_pylist()]

    return build


def _sliced_table():
    table = pa.table({"k": _keys()}).slice(3)
    return table, [(k,) for k in range(3, SPLIT_ROWS)]


def _struct_of_struct_table(rows):
    k = arange(rows)
    inner = pa.StructArray.from_arrays([pa.array(k), pa.array(k * 2)], names=["x", "y"])
    return pa.table({"k": k, "s": pa.StructArray.from_arrays([inner, pa.array(k)], names=["a", "z"])})


def _differently_chunked_columns():
    table = _struct_of_struct_table(10)
    return pa.table(
        {
            "k": pa.chunked_array([table["k"].chunks[0].slice(0, 4), table["k"].chunks[0].slice(4)]),
            "s": table["s"],
        }
    )


def _struct_of_struct_source(make_table, wrap=None):
    def build():
        table = make_table()
        expected = [(row["k"], row["s"]) for row in table.to_pylist()]
        return (wrap(table) if wrap else table), expected

    return build


LAYOUTS = {
    "struct_of_struct_with_null_rows": _column_layout(_struct_of_struct),
    "sparse_union_of_struct": _column_layout(_sparse_union_of_struct),
    "struct_of_sparse_union": _column_layout(_struct_of_sparse_union),
    "list_of_struct": _column_layout(_list_of_struct),
    "struct_of_list_of_struct": _column_layout(_struct_of_list_of_struct),
    "map_of_struct": _column_layout(_map_of_struct, _as_map),
    "fixed_size_list_of_struct": _column_layout(_fixed_size_list_of_struct, _as_tuple),
    "struct_of_struct_of_dictionary": _column_layout(_struct_of_struct_of_dictionary),
    "struct_of_struct_of_run_ends": _column_layout(_struct_of_struct_of_run_ends),
    "string_view": _column_layout(lambda: pc.cast(_text(_keys()), pa.string_view())),
    "list_view": _column_layout(_list_view),
    "flat_run_ends_across_a_range_boundary": _column_layout(lambda: pc.run_end_encode(pa.array(_keys() // 1000))),
    "struct_array_with_null_rows": _struct_array_source(0),
    "struct_array_with_null_rows_and_an_offset": _struct_array_source(5),
    "plain_array_with_null_rows": _plain_array_source(0),
    "plain_array_with_null_rows_and_an_offset": _plain_array_source(5),
    "one_chunk_table_slice": _sliced_table,
    "struct_of_struct_table_slice": _struct_of_struct_source(lambda: _struct_of_struct_table(LARGE_BATCH).slice(3)),
    "struct_of_struct_table_slice_in_batches": _struct_of_struct_source(
        lambda: pa.Table.from_batches(_struct_of_struct_table(LARGE_BATCH).slice(3).to_batches(max_chunksize=65_536))
    ),
    "struct_of_struct_dataset": _struct_of_struct_source(
        lambda: _struct_of_struct_table(LARGE_BATCH),
        lambda table: pytest.importorskip("pyarrow.dataset").dataset(table),
    ),
    "struct_of_struct_differently_chunked_columns": _struct_of_struct_source(_differently_chunked_columns),
}


class TestLayouts:
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("threads", [1, THREADS])
    @pytest.mark.parametrize("layout", list(LAYOUTS))
    def test_nested_layouts_read_the_same_whole_and_split(self, layout, threads):
        obj, expected = LAYOUTS[layout]()
        with duckdb.connect(config={"threads": threads}) as con:
            con.register("src", obj)
            assert con.execute("SELECT * FROM src").fetchall() == expected


RELEASE_TABLE_ROWS = 3 * LARGE_BATCH


@pytest.fixture(scope="module")
def release_table():
    k = arange(RELEASE_TABLE_ROWS)
    return pa.table({"i": k, "s": _text(k)})


class ReleaseCountingStream(core.CountingStream):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.released_again = 0

    def _release_one(self, pointer):
        array = core._ArrowArray.from_address(pointer)
        with self.lock:
            original = self.originals.pop(array.private_data, None)
            if original is None:
                # A copy of an already released array still points here with its old key.
                self.released_again += 1
                return
            self.released += 1
        array.private_data = original[1]
        array.release = original[0]
        core._RELEASE(original[0])(pointer)


class ReleaseCountingSource(core.PythonStreamSource):
    """Counts per stream the batches never released, the batches released again, and the stream releases."""

    def stream(self, columns, filters):
        stream = ReleaseCountingStream(
            self.table.schema, self.table.to_batches(max_chunksize=self.max_chunksize), self.corrupt
        )
        self.streams.append(stream)
        return stream.capsule(), False

    def counts(self):
        gc.collect()
        return [
            (stream.exported - stream.released, stream.released_again, stream.stream_released)
            for stream in self.streams
        ]


class ReleaseCountingSourceWithoutGil(ReleaseCountingSource):
    pull_under_gil = False


RELEASE_SOURCES = pytest.mark.parametrize(
    "source_class",
    [ReleaseCountingSource, ReleaseCountingSourceWithoutGil],
    ids=["pulled_under_the_gil", "pulled_without_the_gil"],
)


class TestReleases:
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    @pytest.mark.parametrize("query", ["SELECT count(*), sum(i) FROM t", "SELECT i FROM t LIMIT 3", "SELECT s FROM t"])
    @RELEASE_SOURCES
    def test_a_large_batch_is_released_once(self, release_table, source_class, query, threads):
        source = source_class(release_table, max_chunksize=LARGE_BATCH)
        with duckdb.connect(config={"threads": threads}) as con:
            con._register_arrow_source("t", source)
            con.execute(query).fetchall()
            assert source.counts() == [(0, 0, 1)]

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    @RELEASE_SOURCES
    def test_a_large_batch_is_released_once_when_the_query_fails_inside_it(self, release_table, source_class, threads):
        from _pushdown_helpers import eventually

        def fail_at(values):
            if pc.any(pc.equal(values, LARGE_BATCH // 2)).as_py():
                msg = "the function gave up"
                raise ValueError(msg)
            return values

        source = source_class(release_table, max_chunksize=LARGE_BATCH)
        with duckdb.connect(config={"threads": threads}) as con:
            con.create_function("fail_at", fail_at, ["BIGINT"], "BIGINT", type="arrow")
            con._register_arrow_source("t", source)
            with pytest.raises(duckdb.Error, match="the function gave up"):
                con.execute("SELECT sum(fail_at(i)) FROM t").fetchall()
            assert eventually(lambda: source.counts() == [(0, 0, 1)]), source.counts()

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    @pytest.mark.parametrize(
        "stop",
        [
            pytest.param(lambda con, result: result.close(), id="closed"),
            pytest.param(lambda con, result: con.execute("SELECT 42").fetchall(), id="replaced_by_the_next_query"),
        ],
    )
    @RELEASE_SOURCES
    def test_a_large_batch_is_released_once_when_its_result_stops_half_fetched(
        self, release_table, source_class, stop, threads
    ):
        from _pushdown_helpers import eventually

        source = source_class(release_table, max_chunksize=LARGE_BATCH)
        with duckdb.connect(config={"threads": threads}) as con:
            con._register_arrow_source("t", source)
            result = con.execute("SELECT i, s FROM t")
            result.fetchmany(10)
            stop(con, result)
            assert eventually(lambda: source.counts() == [(0, 0, 1)]), source.counts()

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    def test_a_projected_column_of_large_batches_keeps_its_values(self, release_table, threads):
        source = core.PythonStreamSource(release_table, max_chunksize=LARGE_BATCH)
        expected = [(value,) for value in release_table["s"].to_pylist()]
        with duckdb.connect(config={"threads": threads}) as con:
            con._register_arrow_source("t", source)
            assert con.execute("SELECT s FROM t").fetchall() == expected

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        ("corrupt", "message"),
        [
            pytest.param(
                core._shorten_first_column,
                "Arrow array child length does not match the array length:"
                f" column 0 holds 1 rows where {LARGE_BATCH} were expected",
                id="short_column",
            ),
            pytest.param(
                core._wrong_column_count, "an Arrow batch carried 1 columns where 2 were expected", id="missing_column"
            ),
        ],
    )
    @RELEASE_SOURCES
    def test_a_malformed_large_batch_is_refused_and_released_once(self, release_table, source_class, corrupt, message):
        from _pushdown_helpers import eventually

        source = source_class(release_table, max_chunksize=LARGE_BATCH, corrupt=corrupt)
        with duckdb.connect(config={"threads": THREADS}) as con:
            con._register_arrow_source("t", source)
            with pytest.raises(duckdb.InvalidInputException, match=re.escape(message)):
                con.execute("SELECT * FROM t").fetchall()
            assert eventually(lambda: source.counts() == [(0, 0, 1)]), source.counts()

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    def test_a_failing_stream_after_a_large_batch_fails_the_query(self, threads):
        def batches():
            yield pa.record_batch([pa.array(arange(LARGE_BATCH))], names=["n"])
            msg = "the producer gave up"
            raise OSError(msg)

        reader = pa.RecordBatchReader.from_batches(pa.schema([("n", pa.int64())]), batches())
        with duckdb.connect(config={"threads": threads}) as con:
            con._register_arrow_source("r", core.ReaderSource(reader))
            with pytest.raises(
                duckdb.InvalidInputException,
                match=r"reading the stream registered as 'r' failed: .*the producer gave up",
            ):
                con.execute("SELECT sum(n) FROM r").fetchall()


# Far more batches than the bound, so a scan that keeps every batch it pulled lands well above it.
READAHEAD_BATCHES = 24
READAHEAD_SCHEMA = pa.schema([("id", pa.int64()), ("payload", pa.string())])
PAYLOAD = "x" * 16
# A scan holds one batch per scanning thread plus one; this leaves room for as many again.
PEAK_BATCHES = 2 * THREADS + 2


def _large_readahead_batches():
    for index in range(READAHEAD_BATCHES):
        # pyarrow wraps a numpy array without allocating, so the ids are computed to count in pyarrow's allocator.
        ids = pc.add(pa.array(arange(LARGE_BATCH)), index * LARGE_BATCH)
        yield pa.record_batch([ids, pa.array([PAYLOAD] * LARGE_BATCH)], schema=READAHEAD_SCHEMA)


def _allocated_bytes_of_one_readahead_batch():
    before = pa.total_allocated_bytes()
    batch = next(_large_readahead_batches())
    allocated = pa.total_allocated_bytes() - before
    del batch
    return allocated


def _slow_identity(ids):
    # Batches pile up only behind a consumer that is slower than the producer.
    time.sleep(0.002)
    return ids


class TestMemory:
    @pytest.mark.timeout(120)
    def test_slowly_consumed_large_batches_hold_a_few_batches(self):
        from _pushdown_helpers import peak_arrow_bytes_while

        query = "SELECT count(*), sum(strlen(payload)), sum(slow_identity(id)) FROM source"
        batch_bytes = _allocated_bytes_of_one_readahead_batch()
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.create_function("slow_identity", _slow_identity, ["BIGINT"], "BIGINT", type="arrow")
            reader = pa.RecordBatchReader.from_batches(READAHEAD_SCHEMA, _large_readahead_batches())
            con._register_arrow_source("source", core.ReaderSource(reader))
            rows, peak, samples = peak_arrow_bytes_while(lambda: con.execute(query).fetchall())

        total = READAHEAD_BATCHES * LARGE_BATCH
        assert rows == [(total, total * len(PAYLOAD), total * (total - 1) // 2)]
        # A sampler that saw nothing of the query would pass any bound.
        assert samples > 0
        assert batch_bytes <= peak <= PEAK_BATCHES * batch_bytes, (peak, batch_bytes)

    @pytest.mark.timeout(60)
    def test_a_one_chunk_source_is_read_without_copying_its_buffers(self):
        from _pushdown_helpers import peak_arrow_bytes_while

        table = pa.table({"n": arange(LARGE), "s": _text(arange(LARGE))})
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.register("src", table)

            def read_for_a_while():
                # One query can end within the sampler's first wait, which on Windows can last a whole timer tick.
                deadline = time.monotonic() + 0.2
                while time.monotonic() < deadline:
                    con.execute("SELECT sum(n), sum(strlen(s)) FROM src").fetchall()

            _, peak, samples = peak_arrow_bytes_while(read_for_a_while)
        assert samples > 0
        # The export allocates a few small structures; a copy of one range of either column is over 900 kB.
        assert peak < 64 * 1024, peak


class TestUnchanged:
    @pytest.mark.parametrize("source", ["table", "array", "polars_frame"])
    def test_the_row_estimate_is_unchanged(self, source):
        obj = ONE_CHUNK_SOURCES[source](arange(LARGE))
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.register("src", obj)
            assert "~1,000,003 rows" in con.execute("EXPLAIN SELECT * FROM src").fetchall()[0][1]

    @pytest.mark.timeout(60)
    def test_ctas_from_a_one_chunk_source_writes_full_row_groups(self):
        table = pa.table({"n": arange(LARGE)})
        expected = [RANGE_ROWS] * (LARGE // RANGE_ROWS) + [LARGE % RANGE_ROWS]
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.register("src", table)
            con.execute("CREATE TABLE copied AS SELECT * FROM src")
            sizes = con.execute(
                "SELECT sum(count) FROM pragma_storage_info('copied') WHERE column_path = '[0]' "
                "GROUP BY row_group_id ORDER BY row_group_id"
            ).fetchall()
            assert [size for (size,) in sizes] == expected


class TestVerifyParallelism:
    @pytest.mark.timeout(60)
    def test_verify_parallelism_splits_a_one_chunk_source_into_vectors(self):
        table = pa.table({"a": arange(3 * 2048)})
        with duckdb.connect(config={"threads": THREADS}) as con:
            install_thread_witness(con)
            con.execute("PRAGMA verify_parallelism")
            con.register("src", table)
            query = reading_threads(con, "src")
            # Three one vector ranges give three threads; ranges of two vectors would give one or two.
            assert con.execute(query).fetchall()[0][0] == 3

    @pytest.mark.timeout(60)
    def test_verify_parallelism_keeps_the_order_of_a_one_chunk_source(self):
        table = pa.table({"n": arange(LARGE)})
        with duckdb.connect(config={"threads": THREADS}) as con:
            con.execute("PRAGMA verify_parallelism")
            con.register("src", table)
            assert np.array_equal(con.execute("SELECT n FROM src").fetchnumpy()["n"], arange(LARGE))


HOLD_SECONDS = 0.2
HOLD_ROWS = 1_000_000


def _hold_source(source):
    pd = _pandas()
    text = [f"v{i % 1000}" for i in range(HOLD_ROWS)]
    i = arange(HOLD_ROWS)
    if source == "object_frame":
        return pd.DataFrame({"x": pd.Series(text, dtype=object), "i": i})
    if source == "pandas_arrow_frame":
        return pd.DataFrame({"x": pd.array(text, dtype=pd.ArrowDtype(pa.string())), "i": i})
    table = pa.table({"x": pa.array(text), "i": i})
    if source == "chunked_table":
        return pa.Table.from_batches(table.to_batches(max_chunksize=65_536))
    return table


def _scan_seconds(obj, udf, hold):
    query = (
        "SELECT count(x), sum(CASE WHEN i = 0 THEN hold_gil(i) ELSE i END) FROM src"
        if hold
        else "SELECT count(x), sum(i) FROM src"
    )
    with duckdb.connect(config={"threads": THREADS}) as con:
        con.create_function("hold_gil", udf, ["BIGINT"], "BIGINT")
        con.register("src", obj)
        con.execute("PRAGMA enable_profiling = 'no_output'")
        con.execute(query).fetchall()
        info = json.loads(con.get_profiling_information())
    total = 0.0
    stack = list(info["operator"])
    while stack:
        node = stack.pop()
        if node.get("type") == "TABLE_SCAN":
            total += node["timing"]
        stack.extend(node.get("children", []))
    return total


class TestGil:
    @pytest.mark.skipif(sys.platform == "win32", reason="holds the GIL through ctypes.PyDLL(None)")
    @pytest.mark.skipif(not getattr(sys, "_is_gil_enabled", lambda: True)(), reason="needs an interpreter with a GIL")
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        ("source", "takes_gil"),
        [
            ("one_chunk_table", False),
            ("chunked_table", False),
            ("pandas_arrow_frame", False),
            ("object_frame", True),
        ],
    )
    def test_scanning_threads_keep_reading_while_another_holds_the_gil(self, source, takes_gil):
        libc = ctypes.PyDLL(None)
        libc.usleep.argtypes = [ctypes.c_uint]
        calls = []

        # A call through PyDLL keeps the GIL for the whole sleep.
        def hold_gil(x):
            calls.append(x)
            libc.usleep(int(HOLD_SECONDS * 1_000_000))
            return x

        obj = _hold_source(source)
        rounds = 3 if takes_gil else 1
        holds = max(
            (_scan_seconds(obj, hold_gil, hold=True) - _scan_seconds(obj, hold_gil, hold=False)) / HOLD_SECONDS
            for _ in range(rounds)
        )
        assert calls == [0] * rounds
        if takes_gil:
            assert holds >= 1, holds
        else:
            assert holds < 0.5, holds


STRUCT_VALUES = [{"i": 1, "s": "a"}, {"i": 2, "s": "b"}, {"i": 3, "s": "c"}]


def _list_of_dictionary(empty, valued):
    indices = pa.array(np.arange(2 * valued, dtype=np.int32) % 3)
    child = pa.DictionaryArray.from_arrays(indices, pa.array(["x", "y", "z"]))
    offsets = [0] * (empty + 1) + [2 * (i + 1) for i in range(valued)]
    return pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), child)


class TestEngineConversions:
    @pytest.mark.xfail(
        strict=True,
        raises=duckdb.InternalException,
        reason="the engine fails to read a dictionary of structs that has no null value: Vector::SetSize out of range",
    )
    @pytest.mark.parametrize("threads", [1, THREADS])
    @pytest.mark.parametrize("indices", [[0, None, 1, 2], [0, 1, 2]], ids=["with_a_null_index", "without"])
    def test_a_struct_dictionary_without_a_null_value_reads_its_values(self, indices, threads):
        column = pa.DictionaryArray.from_arrays(pa.array(indices, pa.int8()), pa.array(STRUCT_VALUES))
        expected = [(None if index is None else STRUCT_VALUES[index],) for index in indices]
        with duckdb.connect(config={"threads": threads}) as con:
            con.register("src", pa.table({"d": column}))
            assert con.execute("SELECT d FROM src").fetchall() == expected

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, 8])
    @pytest.mark.parametrize(("empty", "valued"), [(2048, 1), (300_000, 1000)], ids=["one_vector", "several_ranges"])
    def test_a_list_of_dictionaries_after_a_vector_of_empty_lists_reads_its_lists(self, empty, valued, threads):
        column = _list_of_dictionary(empty, valued)
        expected = [(len(column), pc.sum(pc.list_value_length(column)).as_py())]
        first_valued = [(value,) for value in column.slice(empty - 2, 5).to_pylist()]
        last = [(value,) for value in column.slice(len(column) - 3).to_pylist()]
        with duckdb.connect(config={"threads": threads}) as con:
            con.register("src", pa.table({"l": column}))
            assert con.execute("SELECT count(*), sum(len(l)) FROM src").fetchall() == expected
            assert con.execute(f"SELECT l FROM src LIMIT 5 OFFSET {empty - 2}").fetchall() == first_valued
            assert con.execute(f"SELECT l FROM src OFFSET {len(column) - 3}").fetchall() == last

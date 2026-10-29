"""python_arrow_scan read through the private registration hook, with small sources over pyarrow objects.

The sources here follow the protocol of duckdb._sources: stream(columns, filters) answers a capsule and whether it
holds exactly the requested columns, rows() the row count when known, and __arrow_c_schema__ the schema without a read.
"""

from __future__ import annotations

import ctypes
import datetime
import decimal
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import ClassVar

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")
np = pytest.importorskip("numpy")

from duckdb._sources.nodes import fragment  # noqa: E402

CONSUMED = (
    "This RecordBatchReader has already been consumed and cannot be scanned again. A RecordBatchReader can be read "
    "only once; to scan its data more than once, read it into a table first, for example with read_all()."
)
CALLBACK_REFUSAL = "code running inside a query cannot use that query's connection"


class TableSource:
    """A pyarrow Table or RecordBatch, narrowed with select unless told not to."""

    pull_under_gil = False
    schema_is_stable = True

    def __init__(self, obj, narrows=True) -> None:
        self.obj = obj
        self.narrows = narrows
        self.asked = []
        self.schema_calls = 0

    def rows(self):
        return self.obj.num_rows

    def accepts(self, predicate):
        return False

    def __arrow_c_schema__(self) -> object:
        self.schema_calls += 1
        return self.obj.schema.__arrow_c_schema__()

    def stream(self, columns, filters):
        self.asked.append(columns)
        if columns is None or not self.narrows:
            return self.obj.__arrow_c_stream__(), False
        return self.obj.select(columns).__arrow_c_stream__(), True


class ReaderSource:
    """One RecordBatchReader, whose stream is handed out as it is, so it is read once."""

    pull_under_gil = False
    schema_is_stable = True

    def __init__(self, reader) -> None:
        self.obj = reader
        self.asked = []

    def rows(self):
        return None

    def accepts(self, predicate):
        return False

    def __arrow_c_schema__(self) -> object:
        return self.obj.schema.__arrow_c_schema__()

    def stream(self, columns, filters):
        self.asked.append(columns)
        return self.obj.__arrow_c_stream__(), False


class ArraySource:
    """One array through __arrow_c_array__, which is one batch; no schema hook, so bind reads the pair."""

    pull_under_gil = False

    def __init__(self, array) -> None:
        self.obj = array

    def rows(self):
        return len(self.obj)

    def accepts(self, predicate):
        return False

    def stream(self, columns, filters):
        return self.obj.__arrow_c_array__(), False


def reader_over(table, max_chunksize=None):
    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=max_chunksize))


def generator_reader(table, max_chunksize=None):
    def batches():
        yield from table.to_batches(max_chunksize=max_chunksize)

    return pa.RecordBatchReader.from_batches(table.schema, batches())


def struct_array_of(table):
    return pa.StructArray.from_arrays([column.combine_chunks() for column in table.columns], names=table.column_names)


SHAPES = {
    "table": lambda table: TableSource(table),
    "reader": lambda table: ReaderSource(reader_over(table, max_chunksize=1)),
    "generator_reader": lambda table: ReaderSource(generator_reader(table, max_chunksize=1)),
    "struct_array": lambda table: ArraySource(struct_array_of(table)),
}

D = decimal.Decimal
TS = datetime.datetime(2020, 1, 2, 3, 4, 5, 6)
# The types the Arrow type tests of tests/fast/arrow use.
TYPES = {
    "null": lambda: pa.array([None, None, None], pa.null()),
    "bool": lambda: pa.array([True, None, False]),
    "int8": lambda: pa.array([-128, None, 127], pa.int8()),
    "int16": lambda: pa.array([-32768, None, 32767], pa.int16()),
    "int32": lambda: pa.array([-(2**31), None, 2**31 - 1], pa.int32()),
    "int64": lambda: pa.array([-(2**63), None, 2**63 - 1], pa.int64()),
    "uint8": lambda: pa.array([0, None, 255], pa.uint8()),
    "uint16": lambda: pa.array([0, None, 65535], pa.uint16()),
    "uint32": lambda: pa.array([0, None, 2**32 - 1], pa.uint32()),
    "uint64": lambda: pa.array([0, None, 2**64 - 1], pa.uint64()),
    "float16": lambda: pa.array(np.array([1.5, 0, -2.0], dtype=np.float16), mask=np.array([False, True, False])),
    "float32": lambda: pa.array([1.5, None, float("inf")], pa.float32()),
    "float64": lambda: pa.array([1.5, None, float("-inf")], pa.float64()),
    "decimal32": lambda: pa.array([D("1.23"), None, D("-9.99")], pa.decimal32(5, 2)),
    "decimal64": lambda: pa.array([D("1.23"), None, D("-9.99")], pa.decimal64(12, 2)),
    "decimal128": lambda: pa.array([D("1.23"), None, D("-99999999999999999999.99")], pa.decimal128(38, 2)),
    "decimal256": lambda: pa.array([D("1.23"), None, D("-9.99")], pa.decimal256(20, 2)),
    "string": lambda: pa.array(["a", None, "a longer string with ü"]),
    "large_string": lambda: pa.array(["a", None, "b"], pa.large_string()),
    "string_view": lambda: pa.array(["short", None, "a string longer than twelve"], pa.string_view()),
    "binary": lambda: pa.array([b"\x00\x01", None, b""], pa.binary()),
    "large_binary": lambda: pa.array([b"\x00", None, b"x"], pa.large_binary()),
    "binary_view": lambda: pa.array([b"s", None, b"a binary longer than twelve"], pa.binary_view()),
    "fixed_size_binary": lambda: pa.array([b"ab", None, b"cd"], pa.binary(2)),
    "date32": lambda: pa.array([datetime.date(2020, 1, 2), None, datetime.date(1, 1, 1)], pa.date32()),
    "date64": lambda: pa.array([datetime.date(2020, 1, 2), None, datetime.date(1970, 1, 1)], pa.date64()),
    "time32_s": lambda: pa.array([datetime.time(1, 2, 3), None, datetime.time(0)], pa.time32("s")),
    "time32_ms": lambda: pa.array([datetime.time(1, 2, 3, 4000), None, datetime.time(0)], pa.time32("ms")),
    "time64_us": lambda: pa.array(
        [datetime.time(1, 2, 3, 4), None, datetime.time(23, 59, 59, 999999)], pa.time64("us")
    ),
    "time64_ns": lambda: pa.array([3_723_000_000_001, None, 0], pa.time64("ns")),
    "timestamp_s": lambda: pa.array([TS.replace(microsecond=0), None, datetime.datetime(1, 1, 1)], pa.timestamp("s")),
    "timestamp_ms": lambda: pa.array([TS.replace(microsecond=1000), None, TS], pa.timestamp("ms")),
    "timestamp_us": lambda: pa.array([TS, None, datetime.datetime(9999, 12, 31)], pa.timestamp("us")),
    "timestamp_ns": lambda: pa.array([1_600_000_000_123_456_789, None, 0], pa.timestamp("ns")),
    "timestamp_us_utc": lambda: pa.array([TS, None, TS], pa.timestamp("us", tz="UTC")),
    "timestamp_s_new_york": lambda: pa.array([1_600_000_000, None, 0], pa.timestamp("s", tz="America/New_York")),
    "timestamp_ns_new_york": lambda: pa.array(
        [1_600_000_000_123_456_789, None, 0], pa.timestamp("ns", tz="America/New_York")
    ),
    "duration_s": lambda: pa.array([1, None, -5], pa.duration("s")),
    "duration_ms": lambda: pa.array([1, None, -5], pa.duration("ms")),
    "duration_us": lambda: pa.array([1, None, -5], pa.duration("us")),
    "duration_ns": lambda: pa.array([1000, None, -5000], pa.duration("ns")),
    "month_day_nano_interval": lambda: pa.array(
        [pa.MonthDayNano([1, 2, 3000]), None, pa.MonthDayNano([0, 0, 0])], pa.month_day_nano_interval()
    ),
    "list": lambda: pa.array([[1, None], None, []], pa.list_(pa.int64())),
    "large_list": lambda: pa.array([[1], None, [2, 3]], pa.large_list(pa.int64())),
    "list_view": lambda: pa.array([[1], None, [2, 3]], pa.list_view(pa.int64())),
    "fixed_size_list": lambda: pa.array([[1, 2], None, [3, None]], pa.list_(pa.int32(), 2)),
    "map": lambda: pa.array([[("k", 1)], None, []], pa.map_(pa.string(), pa.int64())),
    "struct": lambda: pa.array([{"x": 1, "y": "a"}, None, {"x": None, "y": "b"}]),
    "empty_struct": lambda: pa.array([None, {}, None], pa.struct([])),
    "list_of_struct": lambda: pa.array([[{"a": [1, 2]}], None, [{"a": None}]]),
    "dictionary_of_string": lambda: pa.array(["p", None, "p"]).dictionary_encode(),
    "dictionary_of_int": lambda: pa.array([1, None, 1]).dictionary_encode(),
    "dictionary_of_string_view": lambda: pa.DictionaryArray.from_arrays(
        pa.array([0, None, 1], pa.int8()), pa.array(["a", "a string longer than twelve"], pa.string_view())
    ),
    "run_end_encoded": lambda: pa.RunEndEncodedArray.from_arrays(
        pa.array([2, 3], pa.int32()), pa.array([7, None], pa.int64())
    ),
    "sparse_union": lambda: pa.UnionArray.from_sparse(
        pa.array([0, 1, 0], pa.int8()), [pa.array([1, 2, None]), pa.array(["x", "y", "z"])], ["i", "s"]
    ),
    "dense_union": lambda: pa.UnionArray.from_dense(
        pa.array([0, 1, 0], pa.int8()),
        pa.array([0, 0, 1], pa.int32()),
        [pa.array([1, None]), pa.array(["x"])],
        ["i", "s"],
    ),
    "uuid": lambda: pa.array([uuid.UUID(int=1).bytes, None, uuid.UUID(int=2).bytes], pa.uuid()),
    "bool8": lambda: pa.array([1, None, 0], pa.bool8()),
    "json": lambda: pa.array(['{"a": 1}', None, "[]"], pa.json_()),
}


# Where each type's expected values come from, independently of the scan: DuckDB's parquet reader over the same
# table written by pyarrow, for the types parquet carries; pyarrow's own to_pylist() for the others; or the values.
PARQUET = "parquet"
PYLIST = "pylist"
# The DuckDB type each column declares, and its reference; a refusal names the exception and its text instead.
EXPECTED = {
    "null": ('"NULL"', PARQUET),
    "bool": ("BOOLEAN", PARQUET),
    "int8": ("TINYINT", PARQUET),
    "int16": ("SMALLINT", PARQUET),
    "int32": ("INTEGER", PARQUET),
    "int64": ("BIGINT", PARQUET),
    "uint8": ("UTINYINT", PARQUET),
    "uint16": ("USMALLINT", PARQUET),
    "uint32": ("UINTEGER", PARQUET),
    "uint64": ("UBIGINT", PARQUET),
    "float16": (None, (duckdb.NotImplementedException, "Unsupported Internal Arrow Type e")),
    "float32": ("FLOAT", PARQUET),
    "float64": ("DOUBLE", PARQUET),
    "decimal32": ("DECIMAL(5,2)", PARQUET),
    "decimal64": ("DECIMAL(12,2)", PARQUET),
    "decimal128": ("DECIMAL(38,2)", PARQUET),
    "decimal256": (None, (duckdb.NotImplementedException, "Unsupported Internal Arrow Type for Decimal d:20,2,256")),
    "string": ("VARCHAR", PARQUET),
    "large_string": ("VARCHAR", PARQUET),
    "string_view": ("VARCHAR", PARQUET),
    "binary": ("BLOB", PARQUET),
    "large_binary": ("BLOB", PARQUET),
    "binary_view": ("BLOB", PARQUET),
    "fixed_size_binary": ("BLOB", PARQUET),
    "date32": ("DATE", PARQUET),
    "date64": ("DATE", PARQUET),
    "time32_s": ("TIME", PARQUET),
    "time32_ms": ("TIME", PARQUET),
    "time64_us": ("TIME", PARQUET),
    "time64_ns": ("TIME_NS", PARQUET),
    "timestamp_s": ("TIMESTAMP_S", PARQUET),
    "timestamp_ms": ("TIMESTAMP_MS", PARQUET),
    "timestamp_us": ("TIMESTAMP", PARQUET),
    "timestamp_ns": ("TIMESTAMP_NS", PARQUET),
    "timestamp_us_utc": ("TIMESTAMP WITH TIME ZONE", PARQUET),
    "timestamp_s_new_york": ("TIMESTAMP WITH TIME ZONE", PARQUET),
    "timestamp_ns_new_york": ("TIMESTAMPTZ_NS", PARQUET),
    "duration_s": ("INTERVAL", PYLIST),
    "duration_ms": ("INTERVAL", PYLIST),
    "duration_us": ("INTERVAL", PYLIST),
    "duration_ns": ("INTERVAL", PYLIST),
    # Python's timedelta has no months, so the conversion of an INTERVAL to Python counts one as 30 days.
    "month_day_nano_interval": (
        "INTERVAL",
        [datetime.timedelta(days=32, microseconds=3), None, datetime.timedelta(0)],
    ),
    "list": ("BIGINT[]", PARQUET),
    "large_list": ("BIGINT[]", PARQUET),
    "list_view": ("BIGINT[]", PARQUET),
    # DuckDB returns a fixed size ARRAY to Python as a tuple.
    "fixed_size_list": ("INTEGER[2]", [(1, 2), None, (3, None)]),
    "map": ("MAP(VARCHAR, BIGINT)", PARQUET),
    "struct": ("STRUCT(x BIGINT, y VARCHAR)", PARQUET),
    "empty_struct": ("STRUCT", PYLIST),
    "list_of_struct": ("STRUCT(a BIGINT[])[]", PARQUET),
    "dictionary_of_string": ("VARCHAR", PARQUET),
    "dictionary_of_int": ("BIGINT", PARQUET),
    "dictionary_of_string_view": ("VARCHAR", PYLIST),
    "run_end_encoded": ("BIGINT", PYLIST),
    "sparse_union": ("UNION(i BIGINT, s VARCHAR)", PYLIST),
    "dense_union": (None, (duckdb.NotImplementedException, 'Unsupported Internal Arrow Type: "d" Union')),
    "uuid": ("UUID", PARQUET),
    "bool8": ("BOOLEAN", PYLIST),
    "json": ("JSON", PARQUET),
}


def reference_rows(reference, table, tmp_path):
    if reference == PARQUET:
        pq = pytest.importorskip("pyarrow.parquet")
        path = tmp_path / "reference.parquet"
        pq.write_table(table, path)
        with duckdb.connect() as con:
            return con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
    if reference == PYLIST:
        return [(row["k"], row["c"]) for row in table.to_pylist()]
    return list(enumerate(reference))


class TestRowsAndTypes:
    def test_every_type_has_an_expectation(self):
        assert sorted(EXPECTED) == sorted(TYPES)

    @pytest.mark.parametrize("shape", list(SHAPES))
    @pytest.mark.parametrize("type_name", list(TYPES))
    def test_rows_equal_an_independent_reference(self, shape, type_name, tmp_path):
        table = pa.table({"k": pa.array([0, 1, 2], pa.int64()), "c": TYPES[type_name]()})
        duckdb_type, reference = EXPECTED[type_name]
        with duckdb.connect() as con:
            if duckdb_type is None:
                exception, message = reference

                def scan():
                    con._register_arrow_source("scanned", SHAPES[shape](table))
                    con.execute("SELECT * FROM scanned").fetchall()

                with pytest.raises(exception, match=re.escape(message)):
                    scan()
                return
            con._register_arrow_source("scanned", SHAPES[shape](table))
            assert [row[1] for row in con.execute("DESCRIBE scanned").fetchall()] == ["BIGINT", duckdb_type]
            assert con.execute("SELECT * FROM scanned").fetchall() == reference_rows(reference, table, tmp_path)

    def test_many_batches_arrive_in_order(self):
        big = pa.Table.from_batches(pa.table({"i": list(range(100_000))}).to_batches(max_chunksize=3_000))
        with duckdb.connect() as con:
            con._register_arrow_source("big", TableSource(big))
            assert con.execute("SELECT count(*), sum(i), min(i), max(i) FROM big").fetchall() == [
                (100_000, 4_999_950_000, 0, 99_999)
            ]
            assert con.execute("SELECT i FROM big LIMIT 3 OFFSET 2_999").fetchall() == [(2_999,), (3_000,), (3_001,)]
            assert con.execute("SELECT i FROM big").fetchall() == [(i,) for i in range(100_000)]

    def test_an_array_longer_than_a_vector_is_split(self):
        values = pa.array([f"v{i % 7}" for i in range(10_000)]).dictionary_encode()
        table = pa.table({"i": pa.array(range(10_000)), "d": values})
        with duckdb.connect() as con:
            con._register_arrow_source("t", ArraySource(struct_array_of(table)))
            assert con.execute("SELECT * FROM t").fetchall() == [(i, f"v{i % 7}") for i in range(10_000)]

    def test_a_plain_array_is_one_column_named_value(self):
        with duckdb.connect() as con:
            con._register_arrow_source("a", ArraySource(pa.array([1, 2, None])))
            assert con.execute("DESCRIBE a").fetchall()[0][:2] == ("value", "BIGINT")
            assert con.execute("SELECT value FROM a").fetchall() == [(1,), (2,), (None,)]

    def test_a_sliced_array_reads_its_slice(self):
        with duckdb.connect() as con:
            con._register_arrow_source("s", ArraySource(pa.array(["x", None, "y", "z"]).slice(1, 2)))
            assert con.execute("SELECT value FROM s").fetchall() == [(None,), ("y",)]

    def test_a_sliced_struct_array_applies_its_offset_to_every_column(self):
        array = pa.StructArray.from_arrays([pa.array([1, 2, 3, 4]), pa.array(["a", "b", None, "d"])], names=["n", "s"])
        with duckdb.connect() as con:
            con._register_arrow_source("s", ArraySource(array.slice(1, 2)))
            assert con.execute("SELECT n, s FROM s").fetchall() == [(2, "b"), (3, None)]

    def test_a_sliced_struct_array_reads_nested_columns_at_its_offset(self):
        array = pa.StructArray.from_arrays(
            [
                pa.array([1, 2, 3, 4]),
                pa.StructArray.from_arrays(
                    [pa.array([10, 20, 30, 40])], ["y"], mask=pa.array([False, False, True, False])
                ),
                pa.array([[1], [2, 2], [3, 3, 3], [4]]),
                pa.array(["p", "q", "r", "s"]).dictionary_encode(),
                pa.RunEndEncodedArray.from_arrays(pa.array([1, 3, 4], pa.int32()), pa.array([7, 8, 9], pa.int64())),
            ],
            ["a", "s", "l", "d", "r"],
            mask=pa.array([False, False, True, False]),
        )
        sliced = array.slice(1, 3)
        with duckdb.connect() as con:
            con._register_arrow_source("t", ArraySource(sliced))
            assert con.execute("SELECT * FROM t").fetchall() == [
                (2, {"y": 20}, [2, 2], "q", 8),
                (None, None, None, None, None),
                (4, {"y": 40}, [4], "s", 9),
            ]

    def test_empty_field_names_keep_the_engine_names(self):
        table = pa.table([pa.array([1]), pa.array([2]), pa.array([3])], names=["", "value", ""])
        with duckdb.connect() as con:
            con._register_arrow_source("t", TableSource(table))
            assert [row[0] for row in con.execute("DESCRIBE t").fetchall()] == ["v0", "value", "v2"]
            assert con.execute("SELECT * FROM t").fetchall() == [(1, 2, 3)]

    def test_dictionaries_that_change_from_batch_to_batch(self):
        batches = [
            pa.record_batch([pa.DictionaryArray.from_arrays(pa.array([0, 1, 0]), pa.array(["a", "b"]))], names=["d"]),
            pa.record_batch([pa.DictionaryArray.from_arrays(pa.array([1, 0, 1]), pa.array(["c", "d"]))], names=["d"]),
            pa.record_batch([pa.DictionaryArray.from_arrays(pa.array([0, None]), pa.array(["e"]))], names=["d"]),
        ]
        reader = pa.RecordBatchReader.from_batches(batches[0].schema, batches)
        with duckdb.connect(config={"threads": 1}) as con:
            con._register_arrow_source("r", ReaderSource(reader))
            assert con.execute("SELECT d FROM r").fetchall() == [
                ("a",),
                ("b",),
                ("a",),
                ("d",),
                ("c",),
                ("d",),
                ("e",),
                (None,),
            ]

    def test_a_null_struct_row_is_a_row_of_nulls_from_a_single_array(self):
        array = pa.StructArray.from_arrays(
            [pa.array([1, 2, 3]), pa.array(["a", "b", "c"]), pa.array([[1], [2], [3]])],
            names=["n", "s", "l"],
            mask=pa.array([False, True, False]),
        )
        with duckdb.connect() as con:
            con._register_arrow_source("st", ArraySource(array))
            assert con.execute("SELECT * FROM st").fetchall() == [(1, "a", [1]), (None, None, None), (3, "c", [3])]
            assert con.execute("SELECT count(n), count(*) FROM st").fetchall() == [(2, 3)]

    def test_a_null_struct_row_is_a_row_of_nulls_from_a_stream(self):
        dictionary = pa.array(["p", "q", "p"]).dictionary_encode()
        arrays = [
            pa.StructArray.from_arrays(
                [pa.array([1, 2, 3]), dictionary, pa.array([{"x": 1}, {"x": 2}, {"x": 3}])],
                names=["n", "d", "st"],
                mask=pa.array([True, False, False]),
            ),
            pa.StructArray.from_arrays(
                [pa.array([4, 5]), pa.array(["q", "p"]).dictionary_encode(), pa.array([{"x": 4}, {"x": 5}])],
                names=["n", "d", "st"],
                mask=pa.array([False, True]),
            ),
        ]
        chunked = pa.chunked_array(arrays)

        class ChunkedSource(ArraySource):
            def stream(self, columns, filters):
                return self.obj.__arrow_c_stream__(), False

            def __arrow_c_schema__(self) -> object:
                return self.obj.type.__arrow_c_schema__()

        with duckdb.connect(config={"threads": 1}) as con:
            con._register_arrow_source("st", ChunkedSource(chunked))
            assert con.execute("SELECT * FROM st").fetchall() == [
                (None, None, None),
                (2, "q", {"x": 2}),
                (3, "p", {"x": 3}),
                (4, "q", {"x": 4}),
                (None, None, None),
            ]

    def test_the_scan_shows_in_the_plan(self):
        with duckdb.connect() as con:
            con._register_arrow_source("t", TableSource(pa.table({"n": [1]})))
            plan = con.execute("EXPLAIN (FORMAT JSON) SELECT * FROM t").fetchall()[0][1]
            assert '"name": "PYTHON_ARROW_SCAN"' in plan
            assert "PYTHON_ARROW_SCAN" in con.execute("EXPLAIN SELECT * FROM t").fetchall()[0][1]

    def test_the_function_is_refused_without_a_source(self):
        with duckdb.connect() as con, pytest.raises(duckdb.BinderException, match="python_arrow_scan requires"):
            con.execute("SELECT * FROM python_arrow_scan()")


class TestCardinality:
    def test_a_count_reaches_the_plan(self):
        with duckdb.connect() as con:
            con._register_arrow_source("t", TableSource(pa.table({"k": list(range(100_000))})))
            assert "~100,000 rows" in con.execute("EXPLAIN SELECT * FROM t").fetchall()[0][1]

    def test_a_count_is_only_an_estimate_for_an_object_that_grows(self):
        source = TableSource(pa.table({"k": [1, 2, 3]}))
        with duckdb.connect() as con:
            con._register_arrow_source("t", source)
            con.execute("PREPARE p AS SELECT k FROM t LIMIT 4")
            source.obj = pa.table({"k": [1, 2, 3, 4, 5, 6]})
            assert con.execute("EXECUTE p").fetchall() == [(1,), (2,), (3,), (4,)]

    @pytest.mark.parametrize("answer", [None, -1, True, "12", 2**70])
    def test_an_answer_that_is_not_a_row_count_leaves_the_estimate_alone(self, answer):
        class Unknown(TableSource):
            def rows(self):
                return answer

        with duckdb.connect() as con:
            con._register_arrow_source("t", Unknown(pa.table({"k": [1, 2, 3]})))
            assert "~1 row" in con.execute("EXPLAIN SELECT * FROM t").fetchall()[0][1]
            assert con.execute("SELECT count(*) FROM t").fetchall() == [(3,)]

    def test_a_count_that_fails(self):
        class Failing(TableSource):
            def rows(self):
                msg = "no idea"
                raise RuntimeError(msg)

        with (
            duckdb.connect() as con,
            pytest.raises(
                duckdb.InvalidInputException,
                match="counting the rows of the object registered as 't' failed: RuntimeError: no idea",
            ),
        ):
            con._register_arrow_source("t", Failing(pa.table({"k": [1]})))


# Seconds that overflow microseconds: the scan's own conversion fails on them.
OVERFLOWING_DURATION = pa.array([1, 2**62], pa.duration("s"))
WIDE = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]})


class TestProjection:
    def test_only_the_columns_a_query_uses_are_asked_for(self):
        source = TableSource(WIDE)
        with duckdb.connect() as con:
            con._register_arrow_source("t", source)
            source.asked.clear()
            assert con.execute("SELECT c, a FROM t").fetchall() == [(1.5, 1), (2.5, 2), (3.5, 3)]
            assert con.execute("SELECT * FROM t LIMIT 1").fetchall() == [(1, "x", 1.5)]
            assert con.execute("SELECT b FROM t WHERE a = 2").fetchall() == [("y",)]
        assert sorted(source.asked[0]) == [0, 2]
        assert source.asked[1] is None
        assert sorted(source.asked[2]) == [0, 1]

    def test_a_count_still_scans_one_column(self):
        source = TableSource(WIDE)
        with duckdb.connect() as con:
            con._register_arrow_source("t", source)
            source.asked.clear()
            assert con.execute("SELECT count(*) FROM t").fetchall() == [(3,)]
        assert source.asked == [[0]]

    def test_a_source_that_cannot_narrow_is_picked_from(self):
        source = TableSource(WIDE, narrows=False)
        with duckdb.connect() as con:
            con._register_arrow_source("t", source)
            assert con.execute("SELECT c, a FROM t").fetchall() == [(1.5, 1), (2.5, 2), (3.5, 3)]
            assert con.execute("SELECT b FROM t WHERE c > 2").fetchall() == [("y",), ("z",)]
            assert con.execute("SELECT count(*) FROM t").fetchall() == [(3,)]

    def test_a_narrow_projection_never_converts_the_other_columns(self):
        table = pa.table({"good": [1, 2], "bad": OVERFLOWING_DURATION})
        with duckdb.connect() as con:
            con._register_arrow_source("t", TableSource(table, narrows=False))
            assert con.execute("SELECT good FROM t").fetchall() == [(1,), (2,)]
            with pytest.raises(duckdb.ConversionException, match="Could not convert Interval to Microsecond"):
                con.execute("SELECT count(bad) FROM t").fetchall()

    def test_duplicate_names_project_by_position(self):
        table = pa.table([pa.array([1]), pa.array([2]), pa.array([3]), pa.array([4])], names=["a_1", "a", "a", "a_2"])
        source = TableSource(table)
        with duckdb.connect() as con:
            con._register_arrow_source("dup", source)
            assert [row[0] for row in con.execute("DESCRIBE dup").fetchall()] == ["a_1", "a", "a_2", "a_2_1"]
            source.asked.clear()
            assert con.execute("SELECT a_2 FROM dup").fetchall() == [(3,)]
            assert con.execute("SELECT a_2_1 FROM dup").fetchall() == [(4,)]
        assert source.asked == [[2], [3]]


class Lying(TableSource):
    def stream(self, columns, filters):
        return self.obj.select([0]).__arrow_c_stream__(), False


class Short(TableSource):
    def stream(self, columns, filters):
        return self.obj.__arrow_c_stream__(), True


class Swapping(TableSource):
    def stream(self, columns, filters):
        return self.obj.select([2, 1, 0]).__arrow_c_stream__(), False


class Retyping(TableSource):
    def stream(self, columns, filters):
        table = self.obj.set_column(0, "a", self.obj["a"].cast(pa.int32()))
        return table.__arrow_c_stream__(), False


class Raising(TableSource):
    def stream(self, columns, filters):
        msg = "no stream today"
        raise ValueError(msg)


class NotAPair(TableSource):
    def stream(self, columns, filters):
        return self.obj.__arrow_c_stream__()


class SchemaCapsule(TableSource):
    def stream(self, columns, filters):
        return self.obj.schema.__arrow_c_schema__(), False


class NotACapsule(TableSource):
    def stream(self, columns, filters):
        return 42, False


class TestRefusals:
    @pytest.mark.parametrize(
        ("source", "query", "message"),
        [
            pytest.param(
                Lying,
                "SELECT b FROM t",
                "the source registered as 't' answered with 1 columns where it declared 3",
                id="width_full",
            ),
            pytest.param(
                Short,
                "SELECT b FROM t",
                "the source registered as 't' answered with 3 columns where 1 were requested",
                id="width_projected",
            ),
            pytest.param(
                Swapping,
                "SELECT * FROM t",
                "the source registered as 't' answered with column 'c' at position 0 where 'a' was expected",
                id="order",
            ),
            pytest.param(
                Retyping,
                "SELECT * FROM t",
                "the source registered as 't' answered with column 'a' of type INTEGER where BIGINT was declared",
                id="type",
            ),
            pytest.param(
                Raising,
                "SELECT * FROM t",
                "exporting a stream from the object registered as 't' failed: ValueError: no stream today",
                id="export_raises",
            ),
            pytest.param(
                NotAPair,
                "SELECT * FROM t",
                "the source registered as 't' answered stream() with something other than a (data, projected) pair",
                id="not_a_pair",
            ),
            pytest.param(
                SchemaCapsule,
                "SELECT * FROM t",
                "the object registered as 't' did not export an 'arrow_array_stream' capsule"
                " but a 'arrow_schema' capsule",
                id="wrong_capsule",
            ),
            pytest.param(
                NotACapsule,
                "SELECT * FROM t",
                "the object registered as 't' did not export an 'arrow_array_stream' capsule but something else",
                id="not_a_capsule",
            ),
        ],
    )
    def test_a_stream_that_does_not_match_the_declaration_is_refused(self, source, query, message):
        with duckdb.connect() as con:
            con._register_arrow_source("t", source(WIDE))
            with pytest.raises(duckdb.InvalidInputException) as raised:
                con.execute(query).fetchall()
            assert message in str(raised.value)
            # The refusal leaves the connection usable.
            assert con.execute("SELECT 42").fetchall() == [(42,)]

    def test_a_schema_that_cannot_be_read(self):
        class Broken(TableSource):
            def __arrow_c_schema__(self) -> object:
                msg = "no schema"
                raise KeyError(msg)

        with (
            duckdb.connect() as con,
            pytest.raises(
                duckdb.InvalidInputException,
                match="reading the schema of the object registered as 't' failed: KeyError: 'no schema'",
            ),
        ):
            con._register_arrow_source("t", Broken(WIDE))

    def test_a_released_stream_capsule(self):
        capsule = WIDE.__arrow_c_stream__()

        class SameCapsule(TableSource):
            def stream(self, columns, filters):
                return capsule, False

        with duckdb.connect() as con:
            con._register_arrow_source("t", SameCapsule(WIDE))
            assert con.execute("SELECT count(*) FROM t").fetchall() == [(3,)]
            with pytest.raises(
                duckdb.InvalidInputException, match="has already been consumed and cannot be scanned again"
            ):
                con.execute("SELECT count(*) FROM t").fetchall()

    def test_a_failed_check_leaves_a_kept_stream_for_the_next_scan(self):
        capsule = WIDE.__arrow_c_stream__()

        class KeepsCapsule(TableSource):
            fail = True

            def stream(self, columns, filters):
                return capsule, self.fail

        source = KeepsCapsule(WIDE)
        with duckdb.connect() as con:
            con._register_arrow_source("t", source)
            with pytest.raises(duckdb.InvalidInputException, match="answered with 3 columns where 1 were requested"):
                con.execute("SELECT b FROM t").fetchall()
            source.fail = False
            assert con.execute("SELECT b FROM t").fetchall() == [("x",), ("y",), ("z",)]

    def test_a_stream_that_fails_while_read(self):
        def batches():
            yield pa.record_batch([pa.array([1])], names=["n"])
            msg = "the producer gave up"
            raise OSError(msg)

        reader = pa.RecordBatchReader.from_batches(pa.schema([("n", pa.int64())]), batches())
        with duckdb.connect() as con:
            con._register_arrow_source("r", ReaderSource(reader))
            with pytest.raises(
                duckdb.InvalidInputException,
                match=r"reading the stream registered as 'r' failed: .*the producer gave up",
            ):
                con.execute("SELECT * FROM r").fetchall()


class TestSingleUse:
    def test_a_reader_is_refused_at_the_second_scan(self):
        with duckdb.connect() as con:
            con._register_arrow_source("r", ReaderSource(reader_over(WIDE)))
            assert con.execute("SELECT count(*) FROM r").fetchall() == [(3,)]
            with pytest.raises(duckdb.InvalidInputException) as raised:
                con.execute("SELECT count(*) FROM r").fetchall()
            assert CONSUMED in str(raised.value)

    def test_two_references_in_one_statement(self):
        with duckdb.connect() as con:
            con._register_arrow_source("r", ReaderSource(reader_over(WIDE)))
            with pytest.raises(duckdb.InvalidInputException) as raised:
                con.execute("SELECT * FROM r UNION ALL SELECT * FROM r").fetchall()
            assert CONSUMED in str(raised.value)

    def test_a_parameterless_prepared_statement_is_refused_at_the_second_execution(self):
        with duckdb.connect() as con:
            con._register_arrow_source("r", ReaderSource(reader_over(WIDE)))
            con.execute("PREPARE p AS SELECT a FROM r")
            assert con.execute("EXECUTE p").fetchall() == [(1,), (2,), (3,)]
            with pytest.raises(duckdb.InvalidInputException) as raised:
                con.execute("EXECUTE p").fetchall()
            assert CONSUMED in str(raised.value)

    def test_a_failed_export_gives_the_claim_back(self):
        class FailsOnce(ReaderSource):
            failures = 1

            def stream(self, columns, filters):
                if self.failures:
                    self.failures -= 1
                    msg = "not yet"
                    raise RuntimeError(msg)
                return super().stream(columns, filters)

        with duckdb.connect() as con:
            con._register_arrow_source("r", FailsOnce(reader_over(WIDE)))
            with pytest.raises(duckdb.InvalidInputException, match="failed: RuntimeError: not yet"):
                con.execute("SELECT * FROM r").fetchall()
            assert con.execute("SELECT * FROM r").fetchall() == [(1, "x", 1.5), (2, "y", 2.5), (3, "z", 3.5)]

    @pytest.mark.parametrize(
        ("wrong", "message"),
        [
            pytest.param("width", "answered with 3 columns where 1 were requested", id="width"),
            pytest.param("name", "answered with column 'renamed' at position 0 where 'a' was expected", id="name"),
            pytest.param("type", "answered with column 'a' of type INTEGER where BIGINT was declared", id="type"),
        ],
    )
    def test_a_refusal_before_the_first_pull_gives_the_claim_back(self, wrong, message):
        class WrongOnce(ReaderSource):
            first = True

            def stream(self, columns, filters):
                if not self.first:
                    return super().stream(columns, filters)
                self.first = False
                capsule = self.obj.__arrow_c_stream__()
                if wrong == "width":
                    return capsule, True
                reader = pa.RecordBatchReader._import_from_c_capsule(capsule)
                if wrong == "name":
                    names = ["renamed", "b", "c"]
                    schema = pa.schema(
                        [field.with_name(name) for field, name in zip(reader.schema, names, strict=True)]
                    )
                    wrapped = pa.RecordBatchReader.from_batches(schema, (b.rename_columns(names) for b in reader))
                    return wrapped.__arrow_c_stream__(), False
                schema = reader.schema.set(0, pa.field("a", pa.int32()))
                return reader.cast(schema).__arrow_c_stream__(), False

        with duckdb.connect() as con:
            con._register_arrow_source("r", WrongOnce(reader_over(WIDE)))
            with pytest.raises(duckdb.InvalidInputException, match=message):
                con.execute("SELECT b FROM r" if wrong == "width" else "SELECT * FROM r").fetchall()
            assert con.execute("SELECT * FROM r").fetchall() == [(1, "x", 1.5), (2, "y", 2.5), (3, "z", 3.5)]

    def test_two_sources_around_one_reader_on_two_connections(self):
        reader = reader_over(WIDE)
        with duckdb.connect() as first, duckdb.connect() as second:
            first._register_arrow_source("r", ReaderSource(reader))
            second._register_arrow_source("r", ReaderSource(reader))
            assert first.execute("SELECT count(*) FROM r").fetchall() == [(3,)]
            with pytest.raises(duckdb.InvalidInputException) as raised:
                second.execute("SELECT count(*) FROM r").fetchall()
            assert CONSUMED in str(raised.value)

    def test_a_class_forging_the_reader_name_stays_reusable(self):
        class RecordBatchReader:
            __module__ = "pyarrow.lib"
            __qualname__ = "RecordBatchReader"
            schema = WIDE.schema

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                return WIDE.__arrow_c_stream__(requested_schema)

        with duckdb.connect() as con:
            con._register_arrow_source("r", ReaderSource(RecordBatchReader()))
            for _ in range(3):
                assert con.execute("SELECT count(*) FROM r").fetchall() == [(3,)]


class TestStateAcrossExecutions:
    def test_a_parameterless_prepared_statement_exports_per_execution(self):
        source = TableSource(WIDE)
        with duckdb.connect() as con:
            con._register_arrow_source("t", source)
            con.execute("PREPARE p AS SELECT b FROM t WHERE a = 2")
            source.asked.clear()
            assert con.execute("EXECUTE p").fetchall() == [("y",)]
            assert con.execute("EXECUTE p").fetchall() == [("y",)]
        assert len(source.asked) == 2

    def test_a_prepared_statement_with_changing_parameters(self):
        with duckdb.connect() as con:
            con._register_arrow_source("t", TableSource(WIDE))
            con.execute("PREPARE q AS SELECT b FROM t WHERE a = $1")
            assert con.execute("EXECUTE q(1)").fetchall() == [("x",)]
            assert con.execute("EXECUTE q(3)").fetchall() == [("z",)]
            assert con.execute("SELECT b FROM t WHERE a = ?", [2]).fetchall() == [("y",)]

    def test_a_self_join_with_different_predicates_per_side(self):
        with duckdb.connect() as con:
            con._register_arrow_source("t", TableSource(WIDE))
            con.execute("CREATE TABLE native AS SELECT * FROM t")
            query = "SELECT x.b, y.b FROM {0} x JOIN {0} y ON x.a = y.a + 1 WHERE x.c > 2 AND y.b <> 'z' ORDER BY ALL"
            assert con.execute(query.format("t")).fetchall() == con.execute(query.format("native")).fetchall()
            assert con.execute(query.format("t")).fetchall() == [("y", "x"), ("z", "y")]

    def test_a_prepared_statement_fails_after_unregister(self):
        with duckdb.connect() as con:
            con._register_arrow_source("t", TableSource(WIDE))
            con.execute("PREPARE p AS SELECT a FROM t WHERE c > 2")
            con.unregister("t")
            with pytest.raises(duckdb.CatalogException):
                con.execute("EXECUTE p")

    def test_the_schema_hook_is_asked_once_for_the_life_of_the_source(self):
        source = TableSource(WIDE)
        with duckdb.connect() as con:
            con._register_arrow_source("t", source)
            for _ in range(3):
                assert con.execute("SELECT count(*) FROM t").fetchall() == [(3,)]
            con.execute("PREPARE p AS SELECT a FROM t")
            con.execute("EXECUTE p").fetchall()
        assert source.schema_calls == 1
        assert len(source.asked) == 4

    @pytest.mark.parametrize("stable", ["missing", False])
    def test_a_source_whose_schema_may_change_is_asked_at_every_bind(self, stable):
        class Changing:
            pull_under_gil = False

            def __init__(self, obj) -> None:
                self.obj = obj
                self.schema_calls = 0

            def rows(self):
                return self.obj.num_rows

            def accepts(self, predicate):
                return False

            def __arrow_c_schema__(self) -> object:
                self.schema_calls += 1
                return self.obj.schema.__arrow_c_schema__()

            def stream(self, columns, filters):
                return self.obj.__arrow_c_stream__(), False

        if stable != "missing":
            Changing.schema_is_stable = stable
        source = Changing(WIDE)
        with duckdb.connect() as con:
            con._register_arrow_source("t", source)
            calls = source.schema_calls
            for _ in range(3):
                assert con.execute("SELECT count(*) FROM t").fetchall() == [(3,)]
        assert source.schema_calls == calls + 3

    def test_the_schema_of_a_pyarrow_schema_attribute(self):
        class NoHook:
            pull_under_gil = False

            def __init__(self, reader) -> None:
                self.obj = reader
                self.exports = 0

            def rows(self):
                return None

            def accepts(self, predicate):
                return False

            def stream(self, columns, filters):
                self.exports += 1
                return self.obj.__arrow_c_stream__(), False

        source = NoHook(reader_over(WIDE))
        with duckdb.connect() as con:
            con._register_arrow_source("r", source)
            assert source.exports == 0
            assert con.execute("SELECT b FROM r").fetchall() == [("x",), ("y",), ("z",)]
        assert source.exports == 1

    def test_a_stream_only_source_is_borrowed_for_its_schema(self):
        capsule = WIDE.__arrow_c_stream__()

        class Capsule:
            pull_under_gil = True
            obj = capsule

            def rows(self):
                return None

            def accepts(self, predicate):
                return False

            def stream(self, columns, filters):
                return capsule, False

        with duckdb.connect() as con:
            con._register_arrow_source("c", Capsule())
            assert con.execute("SELECT a FROM c").fetchall() == [(1,), (2,), (3,)]


def evaluate(node, table):
    """The mask of a predicate node over a pyarrow Table, for the shapes the filtering source accepts."""
    kind = type(node).__name__
    if kind == "Col" and len(node.parts) == 1:
        return table[node.parts[0]]
    if kind == "Lit" and isinstance(node.value, (int, float, str)):
        return pa.scalar(node.value)
    if kind == "Binary" and node.op in ("AND", "OR"):
        combine = pc.and_kleene if node.op == "AND" else pc.or_kleene
        return combine(evaluate(node.left, table), evaluate(node.right, table))
    comparisons = {
        "=": pc.equal,
        "!=": pc.not_equal,
        "<": pc.less,
        "<=": pc.less_equal,
        ">": pc.greater,
        ">=": pc.greater_equal,
    }
    if kind == "Binary" and node.op in comparisons:
        return comparisons[node.op](evaluate(node.left, table), evaluate(node.right, table))
    msg = f"not evaluated here: {kind}"
    raise ValueError(msg)


class FilteringSource(TableSource):
    """Accepts every predicate it can evaluate and applies the ones it is given before narrowing."""

    def __init__(self, obj, accepting=True) -> None:
        super().__init__(obj)
        self.accepting = accepting
        self.offered = []
        self.calls = []

    def accepts(self, predicate):
        self.offered.append(fragment(predicate))
        if not self.accepting:
            return False
        try:
            evaluate(predicate, self.obj)
        except ValueError:
            return False
        return True

    def stream(self, columns, filters):
        self.calls.append((columns, [fragment(predicate) for predicate in filters]))
        table = self.obj
        for predicate in filters:
            table = table.filter(evaluate(predicate, table))
        if columns is None:
            return table.__arrow_c_stream__(), False
        return table.select(columns).__arrow_c_stream__(), True


def plan_nodes(con, query):
    def walk(node):
        yield node
        for child in node.get("children", []):
            yield from walk(child)

    plan = json.loads(con.execute(f"EXPLAIN (FORMAT JSON) {query}").fetchall()[0][1])
    return [node for root in plan for node in walk(root)]


def filters_above_scan(con, query):
    return [node for node in plan_nodes(con, query) if node["name"] == "FILTER"]


def scan_filters(con, query):
    return [node["extra_info"].get("Filters") for node in plan_nodes(con, query) if node["name"] == "PYTHON_ARROW_SCAN"]


A_OR_B = """(("a" = 1) OR ("b" = 'z'))"""


class TestFilters:
    @pytest.fixture
    def con(self):
        with duckdb.connect() as con:
            con.execute("CREATE TABLE native AS SELECT * FROM WIDE")
            yield con

    def test_an_accepted_predicate_leaves_the_plan_and_reaches_the_stream(self, con):
        source = FilteringSource(WIDE)
        con._register_arrow_source("t", source)
        query = "SELECT * FROM {} WHERE a = 1 OR b = 'z' ORDER BY a"
        assert filters_above_scan(con, query.format("t")) == []
        assert scan_filters(con, query.format("t")) == [A_OR_B]
        source.calls.clear()
        assert con.execute(query.format("t")).fetchall() == con.execute(query.format("native")).fetchall()
        assert source.calls == [(None, [A_OR_B])]

    def test_a_refused_predicate_stays_above_the_scan(self, con):
        source = FilteringSource(WIDE, accepting=False)
        con._register_arrow_source("t", source)
        query = "SELECT * FROM {} WHERE a = 1 OR b = 'z' ORDER BY a"
        assert len(filters_above_scan(con, query.format("t"))) == 1
        assert scan_filters(con, query.format("t")) == [None]
        source.calls.clear()
        assert con.execute(query.format("t")).fetchall() == con.execute(query.format("native")).fetchall()
        assert source.calls == [(None, [])]

    def test_a_column_only_a_filter_reads_is_not_requested(self, con):
        source = FilteringSource(WIDE)
        con._register_arrow_source("t", source)
        source.calls.clear()
        assert con.execute("SELECT b FROM t WHERE a = 2").fetchall() == [("y",)]
        assert source.calls == [([1], ['("a" = 2)'])]

    def test_a_filter_derived_for_a_cte_accumulates(self, con):
        source = FilteringSource(WIDE)
        con._register_arrow_source("t", source)
        query = "WITH c AS (SELECT * FROM {}) SELECT * FROM c WHERE a > 1 ORDER BY a"
        source.calls.clear()
        assert con.execute(query.format("t")).fetchall() == con.execute(query.format("native")).fetchall()
        assert source.calls == [(None, ['("a" > 1)'])]

    def test_each_side_of_a_self_join_keeps_its_own_predicates(self, con):
        source = FilteringSource(WIDE)
        con._register_arrow_source("t", source)
        query = "SELECT x.b, y.b FROM {0} x JOIN {0} y ON x.a = y.a + 1 WHERE x.c > 2 AND y.b <> 'z' ORDER BY ALL"
        source.calls.clear()
        assert con.execute(query.format("t")).fetchall() == con.execute(query.format("native")).fetchall()
        assert sorted(filters for _, filters in source.calls) == [["(\"b\" != 'z')"], ['("c" > 2.0::DOUBLE)']]

    def test_a_range_arrives_as_one_conjunction(self, con):
        source = FilteringSource(WIDE)
        con._register_arrow_source("t", source)
        source.calls.clear()
        assert con.execute("SELECT a FROM t WHERE a > 1 AND a < 3").fetchall() == [(2,)]
        assert source.calls == [([0], ['(("a" > 1) AND ("a" < 3))'])]

    def test_a_prepared_statement_sees_each_parameter(self, con):
        source = FilteringSource(WIDE)
        con._register_arrow_source("t", source)
        con.execute("PREPARE q AS SELECT b FROM t WHERE a = $1")
        source.calls.clear()
        assert con.execute("EXECUTE q(1)").fetchall() == [("x",)]
        assert con.execute("EXECUTE q(3)").fetchall() == [("z",)]
        assert [filters for _, filters in source.calls] == [['("a" = 1)'], ['("a" = 3)']]

    def test_a_parameterless_prepared_statement_keeps_its_predicates(self, con):
        source = FilteringSource(WIDE)
        con._register_arrow_source("t", source)
        con.execute("PREPARE p AS SELECT b FROM t WHERE a >= 2")
        source.calls.clear()
        for _ in range(2):
            assert con.execute("EXECUTE p").fetchall() == [("y",), ("z",)]
        assert [filters for _, filters in source.calls] == [['("a" >= 2)'], ['("a" >= 2)']]

    def test_an_offer_that_raises(self, con):
        class Raising(FilteringSource):
            def accepts(self, predicate):
                msg = "cannot decide"
                raise LookupError(msg)

        con._register_arrow_source("t", Raising(WIDE))
        with pytest.raises(
            duckdb.InvalidInputException,
            match="offering a filter to the source registered as 't' failed: LookupError: cannot decide",
        ):
            con.execute("SELECT * FROM t WHERE a = 1").fetchall()


THREADS = 4


def batches_with_tails():
    sizes = [0, 2048, 1, 0, 3000, 2047, 0, 5000, 7, 4096, 0, 2049] * 9
    start = 0
    for size in sizes:
        yield pa.record_batch([pa.array(range(start, start + size), pa.int64())], names=["i"])
        start += size


class TestParallelOrdering:
    def test_a_hundred_batch_reader_keeps_its_order_at_four_threads(self):
        table = pa.table({"i": pa.array(range(100 * 1500), pa.int64())})
        with duckdb.connect(config={"threads": THREADS}) as con:
            con._register_arrow_source("r", ReaderSource(reader_over(table, max_chunksize=1500)))
            assert con.execute("SELECT i FROM r").fetchall() == [(i,) for i in range(100 * 1500)]

    def test_empty_batches_and_short_tails_keep_their_order(self):
        expected = [(i,) for batch in batches_with_tails() for i in batch.column(0).to_pylist()]
        schema = pa.schema([("i", pa.int64())])
        with duckdb.connect(config={"threads": THREADS}) as con:
            con._register_arrow_source(
                "r", ReaderSource(pa.RecordBatchReader.from_batches(schema, batches_with_tails()))
            )
            assert con.execute("SELECT i FROM r").fetchall() == expected
            con._register_arrow_source(
                "r", ReaderSource(pa.RecordBatchReader.from_batches(schema, batches_with_tails()))
            )
            con.execute("CREATE TABLE copied AS SELECT i FROM r")
            assert con.execute("SELECT i FROM copied").fetchall() == expected


def fetch_with(api, con, query):
    rel = con.sql(query)
    if api == "fetchall":
        return rel.fetchall()
    if api == "fetchone":
        con.execute(query)
        rows = []
        while (row := con.fetchone()) is not None:
            rows.append(row)
        return rows
    if api == "fetchmany":
        con.execute(query)
        rows = []
        while chunk := con.fetchmany(1000):
            rows.extend(chunk)
        return rows
    if api == "df":
        return [tuple(row) for row in rel.df().itertuples(index=False)]
    if api == "arrow":
        return [tuple(row.values()) for row in rel.arrow().read_all().to_pylist()]
    if api == "fetch_record_batch":
        with pytest.warns(DeprecationWarning, match="fetch_record_batch"):
            reader = rel.fetch_record_batch(500)
        return [tuple(row.values()) for row in reader.read_all().to_pylist()]
    if api == "pl":
        return list(rel.pl().iter_rows())
    raise AssertionError(api)


class TestFetchApis:
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    @pytest.mark.parametrize("api", ["fetchall", "fetchone", "fetchmany", "df", "arrow", "fetch_record_batch", "pl"])
    def test_a_generator_backed_reader_through_every_fetch(self, api, threads):
        if api in ("df",):
            pytest.importorskip("pandas")
        if api == "pl":
            pytest.importorskip("polars")
        table = pa.table({"i": pa.array(range(20_000), pa.int64()), "s": [str(i % 13) for i in range(20_000)]})
        with duckdb.connect(config={"threads": threads}) as con:
            con._register_arrow_source("r", ReaderSource(generator_reader(table, max_chunksize=999)))
            rows = fetch_with(api, con, "SELECT i, s FROM r")
        assert rows == [(i, str(i % 13)) for i in range(20_000)]

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    def test_a_generator_backed_reader_through_execute(self, threads):
        table = pa.table({"i": pa.array(range(20_000), pa.int64())})
        with duckdb.connect(config={"threads": threads}) as con:
            con._register_arrow_source("r", ReaderSource(generator_reader(table, max_chunksize=999)))
            assert con.execute("SELECT i FROM r").fetchone() == (0,)
            assert con.fetchmany(2) == [(1,), (2,)]
            assert len(con.fetchall()) == 20_000 - 3

    @pytest.mark.timeout(60)
    def test_interrupt_during_a_pulling_scan(self):
        pulling = threading.Event()

        def slow():
            for i in range(10_000):
                pulling.set()
                time.sleep(0.01)
                yield pa.record_batch([pa.array([i])], names=["i"])

        reader = pa.RecordBatchReader.from_batches(pa.schema([("i", pa.int64())]), slow())
        con = duckdb.connect()
        con._register_arrow_source("r", ReaderSource(reader))

        def interrupt_once_pulling():
            if pulling.wait(30):
                con.interrupt()

        interrupter = threading.Thread(target=interrupt_once_pulling)
        interrupter.start()
        try:
            with pytest.raises(duckdb.InterruptException):
                con.execute("SELECT sum(i) FROM r").fetchall()
        finally:
            interrupter.join()
        assert pulling.is_set()
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        con.close()


class TestCallbacksIntoTheConnection:
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("call", ["execute", "close"])
    def test_a_stream_callback_cannot_use_its_own_connection(self, call):
        con = duckdb.connect()

        class Reentrant(TableSource):
            def stream(self, columns, filters):
                if call == "execute":
                    con.execute("SELECT 1")
                else:
                    con.close()
                return super().stream(columns, filters)

        con._register_arrow_source("t", Reentrant(WIDE))
        with pytest.raises(duckdb.InvalidInputException) as raised:
            con.execute("SELECT * FROM t").fetchall()
        assert "exporting a stream from the object registered as 't' failed: ResourceInUseException" in str(
            raised.value
        )
        assert CALLBACK_REFUSAL in str(raised.value)
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        con.close()

    @pytest.mark.timeout(60)
    def test_a_row_count_callback_cannot_use_its_own_connection(self):
        con = duckdb.connect()
        source = TableSource(WIDE)
        con._register_arrow_source("t", source)

        def rows():
            con.execute("SELECT 1")
            return 3

        source.rows = rows
        with pytest.raises(duckdb.InvalidInputException, match=CALLBACK_REFUSAL):
            con.execute("SELECT * FROM t").fetchall()
        con.close()

    @pytest.mark.timeout(60)
    def test_a_pull_cannot_use_its_own_connection(self):
        con = duckdb.connect()

        def batches():
            con.execute("SELECT 1")
            yield pa.record_batch([pa.array([1])], names=["n"])

        reader = pa.RecordBatchReader.from_batches(pa.schema([("n", pa.int64())]), batches())
        con._register_arrow_source("r", ReaderSource(reader))
        with pytest.raises(
            duckdb.InvalidInputException, match=f"reading the stream registered as 'r' failed: .*{CALLBACK_REFUSAL}"
        ):
            con.execute("SELECT * FROM r").fetchall()
        con.close()

    @pytest.mark.timeout(60)
    def test_a_callback_can_use_a_separate_connection(self):
        con = duckdb.connect()
        other = con.cursor()
        answers = []

        class UsesCursor(TableSource):
            def stream(self, columns, filters):
                answers.append(other.execute("SELECT 42").fetchall())
                return super().stream(columns, filters)

        con._register_arrow_source("t", UsesCursor(WIDE))
        assert con.execute("SELECT count(*) FROM t").fetchall() == [(3,)]
        assert answers == [[(42,)]]
        other.close()
        con.close()


BATCHES = 40
BATCH_ROWS = 1024
PAYLOAD = "x" * 1024
READAHEAD_SCHEMA = pa.schema([("id", pa.int64()), ("payload", pa.string())])
# A scan holds one batch per scanning thread plus one; this leaves room for as many again.
PEAK_BATCHES = 2 * THREADS + 2


def _readahead_batches():
    for index in range(BATCHES):
        start = index * BATCH_ROWS
        yield pa.record_batch(
            [pa.array(range(start, start + BATCH_ROWS), pa.int64()), pa.array([PAYLOAD] * BATCH_ROWS)],
            schema=READAHEAD_SCHEMA,
        )


def _slow_identity(ids):
    # Batches pile up only behind a consumer that is slower than the producer.
    time.sleep(0.02)
    return ids


class TestMemory:
    @pytest.mark.timeout(60)
    def test_slowly_consumed_stream_holds_a_few_batches(self):
        from _pushdown_helpers import peak_arrow_bytes_while

        query = "SELECT count(*), sum(strlen(payload)), sum(slow_identity(id)) FROM source"
        with duckdb.connect(config={"threads": THREADS, "memory_limit": "64MB"}) as con:
            con.create_function("slow_identity", _slow_identity, ["BIGINT"], "BIGINT", type="arrow")
            reader = pa.RecordBatchReader.from_batches(READAHEAD_SCHEMA, _readahead_batches())
            con._register_arrow_source("source", ReaderSource(reader))
            rows, peak, samples = peak_arrow_bytes_while(lambda: con.execute(query).fetchall())

        total = BATCHES * BATCH_ROWS
        assert rows == [(total, total * len(PAYLOAD), total * (total - 1) // 2)]
        batch_bytes = next(_readahead_batches()).nbytes
        # A sampler that saw nothing of the query would pass any bound.
        assert samples > 0
        assert batch_bytes <= peak <= PEAK_BATCHES * batch_bytes, (peak, batch_bytes)

    @pytest.mark.parametrize("query", ["SELECT count(*), sum(i) FROM {}", "SELECT s FROM {} LIMIT 5"])
    @pytest.mark.parametrize("ending", ["unregister", "close"])
    def test_every_exported_buffer_is_released(self, ending, query):
        from _pushdown_helpers import eventually

        gc.collect()
        baseline = pa.total_allocated_bytes()
        table = pa.table({"i": pa.array(range(50_000), pa.int64()), "s": pa.array([str(i) for i in range(50_000)])})
        con = duckdb.connect(config={"threads": THREADS})
        con._register_arrow_source("t", TableSource(table))
        con._register_arrow_source("r", ReaderSource(generator_reader(table, max_chunksize=1000)))
        for name in ("t", "r"):
            assert con.execute(query.format(name)).fetchall()
        del table
        if ending == "unregister":
            con.unregister("t")
            con.unregister("r")
        else:
            con.close()
        assert eventually(lambda: pa.total_allocated_bytes() == baseline), (pa.total_allocated_bytes(), baseline)
        con.close()


# An Arrow C stream implemented in Python with ctypes, so every callback, the release of each array included, runs
# Python; the counts tell whether the scan released every array it adopted, exactly once.
class _ArrowArray(ctypes.Structure):
    pass


_ArrowArray._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.c_void_p),
    ("children", ctypes.POINTER(ctypes.POINTER(_ArrowArray))),
    ("dictionary", ctypes.c_void_p),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


class _ArrowArrayStream(ctypes.Structure):
    _fields_: ClassVar = [
        ("get_schema", ctypes.c_void_p),
        ("get_next", ctypes.c_void_p),
        ("get_last_error", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


_RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_GET = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
_LAST_ERROR = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
_new_capsule = ctypes.pythonapi.PyCapsule_New
_new_capsule.restype = ctypes.py_object
_new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_STREAM_NAME = b"arrow_array_stream"


class CountingStream:
    def __init__(self, schema, batches, corrupt=None) -> None:
        self.schema = schema
        self.batches = iter(batches)
        self.corrupt = corrupt
        self.lock = threading.Lock()
        self.exported = 0
        self.released = 0
        self.stream_released = 0
        self.originals = {}
        self.next_key = 1
        self.struct = _ArrowArrayStream()
        self.callbacks = [
            _GET(self._get_schema),
            _GET(self._get_next),
            _LAST_ERROR(lambda _: None),
            _RELEASE(self._release_stream),
        ]
        self._release_array = _RELEASE(self._release_one)
        self._destructor = _RELEASE(self._destroy_capsule)
        for field, callback in zip(
            ("get_schema", "get_next", "get_last_error", "release"), self.callbacks, strict=True
        ):
            setattr(self.struct, field, ctypes.cast(callback, ctypes.c_void_p))

    def capsule(self):
        return _new_capsule(ctypes.addressof(self.struct), _STREAM_NAME, ctypes.cast(self._destructor, ctypes.c_void_p))

    def _destroy_capsule(self, capsule):
        pointer = _capsule_pointer(capsule, _STREAM_NAME)
        stream = _ArrowArrayStream.from_address(pointer)
        if stream.release:
            _RELEASE(stream.release)(pointer)

    def _get_schema(self, _, out):
        self.schema._export_to_c(out)
        return 0

    def _get_next(self, _, out):
        batch = next(self.batches, None)
        array = _ArrowArray.from_address(out)
        if batch is None:
            array.release = None
            return 0
        batch._export_to_c(out)
        with self.lock:
            key = self.next_key
            self.next_key += 1
            self.originals[key] = (array.release, array.private_data)
            self.exported += 1
        array.private_data = key
        array.release = ctypes.cast(self._release_array, ctypes.c_void_p)
        if self.corrupt:
            self.corrupt(array)
        return 0

    def _release_one(self, pointer):
        array = _ArrowArray.from_address(pointer)
        with self.lock:
            release, private_data = self.originals.pop(array.private_data)
            self.released += 1
        array.private_data = private_data
        array.release = release
        _RELEASE(release)(pointer)

    def _release_stream(self, pointer):
        _ArrowArrayStream.from_address(pointer).release = None
        self.stream_released += 1


class PythonStreamSource:
    """Each scan gets a fresh stream implemented in Python."""

    pull_under_gil = True

    def __init__(self, table, max_chunksize=1000, corrupt=None) -> None:
        self.obj = self
        self.table = table
        self.max_chunksize = max_chunksize
        self.corrupt = corrupt
        self.streams = []

    def rows(self):
        return None

    def accepts(self, predicate):
        return False

    def __arrow_c_schema__(self) -> object:
        return self.table.schema.__arrow_c_schema__()

    def stream(self, columns, filters):
        stream = CountingStream(
            self.table.schema, self.table.to_batches(max_chunksize=self.max_chunksize), self.corrupt
        )
        self.streams.append(stream)
        return stream.capsule(), False

    def counts(self):
        gc.collect()
        return [(stream.exported - stream.released, stream.stream_released) for stream in self.streams]


def _shorten_first_column(array):
    array.children[0].contents.length = 1


def _wrong_column_count(array):
    array.n_children = 1


def _overflowing_root_offset(array):
    array.offset = 2**63 - 1


def _negative_root_length(array):
    array.length = -1


def _overflowing_column_offset(array):
    array.children[0].contents.offset = 2**63 - 1


def ranged_struct_array():
    # Five row group sized ranges and a shorter remainder, so threads share the one array.
    k = np.arange(5 * 122_880 + 17, dtype=np.int64)
    return pa.StructArray.from_arrays([pa.array(k), pc.cast(pa.array(k % 101), pa.string())], names=["i", "s"])


class CountingArraySource(ArraySource):
    """One struct array whose every export counts its releases.

    A given length overwrites the exported one, so the scan sees rows its columns do not hold.
    """

    def __init__(self, array, length=None) -> None:
        super().__init__(array)
        self.length = length
        self.lock = threading.Lock()
        self.exported = 0
        self.released = 0
        self.released_again = 0
        self.originals = {}
        self._release_array = _RELEASE(self._release_one)

    def stream(self, columns, filters):
        schema, capsule = self.obj.__arrow_c_array__()
        array = _ArrowArray.from_address(_capsule_pointer(id(capsule), b"arrow_array"))
        if self.length is not None:
            array.length = self.length
        with self.lock:
            self.exported += 1
            key = self.exported
            self.originals[key] = (array.release, array.private_data)
        array.private_data = key
        array.release = ctypes.cast(self._release_array, ctypes.c_void_p)
        return (schema, capsule), False

    def _release_one(self, pointer):
        array = _ArrowArray.from_address(pointer)
        with self.lock:
            original = self.originals.pop(array.private_data, None)
            if original is None:
                # A copy of an already released array still points here with its old key.
                self.released_again += 1
                return
            self.released += 1
        array.release, array.private_data = original
        _RELEASE(original[0])(pointer)

    def counts(self):
        gc.collect()
        return (self.exported - self.released, self.released_again)


class _ArrowSchemaHead(ctypes.Structure):
    _fields_: ClassVar = [
        ("format", ctypes.c_void_p),
        ("name", ctypes.c_void_p),
        ("metadata", ctypes.c_void_p),
        ("flags", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("children", ctypes.POINTER(ctypes.c_void_p)),
    ]


def _schema_head(capsule):
    return _ArrowSchemaHead.from_address(_capsule_pointer(id(capsule), b"arrow_schema"))


class _ArrowSchema(ctypes.Structure):
    _fields_: ClassVar = [
        *_ArrowSchemaHead._fields_,
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


class SchemaReleases:
    """Counts the releases of the schemas it wraps: each wrapped release restores the original and calls it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.wrapped = 0
        self.released = 0
        self.originals = {}
        self.next_key = 1
        self._release = _RELEASE(self._release_one)

    def wrap(self, address):
        schema = _ArrowSchema.from_address(address)
        with self.lock:
            key = self.next_key
            self.next_key += 1
            self.originals[key] = (schema.release, schema.private_data)
            self.wrapped += 1
        schema.private_data = key
        schema.release = ctypes.cast(self._release, ctypes.c_void_p)

    def wrap_capsule(self, capsule):
        self.wrap(_capsule_pointer(id(capsule), b"arrow_schema"))
        return capsule

    def _release_one(self, pointer):
        schema = _ArrowSchema.from_address(pointer)
        with self.lock:
            release, private_data = self.originals.pop(schema.private_data)
            self.released += 1
        schema.private_data = private_data
        schema.release = release
        _RELEASE(release)(pointer)


class SchemaCountingStream(CountingStream):
    def __init__(self, schemas, *args) -> None:
        super().__init__(*args)
        self.schemas = schemas

    def _get_schema(self, stream, out):
        code = super()._get_schema(stream, out)
        self.schemas.wrap(out)
        return code


class SchemaCountingSource(PythonStreamSource):
    """A PythonStreamSource whose schemas, from the schema hook and from each stream, are counted as they go."""

    def __init__(self, table, max_chunksize=1000) -> None:
        super().__init__(table, max_chunksize)
        self.schemas = SchemaReleases()

    def __arrow_c_schema__(self) -> object:
        return self.schemas.wrap_capsule(super().__arrow_c_schema__())

    def stream(self, columns, filters):
        stream = SchemaCountingStream(
            self.schemas, self.table.schema, self.table.to_batches(max_chunksize=self.max_chunksize), self.corrupt
        )
        self.streams.append(stream)
        return stream.capsule(), False


class _Pokes:
    """Changes made to an exported batch, undone before the producer's own release reads the batch again."""

    def __init__(self) -> None:
        self.saved = []

    def set(self, ctype, address, value):
        cell = ctype.from_address(address)
        self.saved.append((ctype, address, cell.value))
        cell.value = value

    def undo(self):
        while self.saved:
            ctype, address, value = self.saved.pop()
            ctype.from_address(address).value = value


def _field(struct, name):
    return ctypes.addressof(struct) + getattr(type(struct), name).offset


def _first_field(schema_address):
    return _ArrowSchemaHead.from_address(_ArrowSchemaHead.from_address(schema_address).children[0])


class PokingStream(CountingStream):
    def __init__(self, schema, batches, poke, poke_schema=None) -> None:
        self.pokes = _Pokes()
        self.poke_schema = poke_schema
        super().__init__(schema, batches, poke and (lambda array: poke(array.children[0].contents, self.pokes)))

    def _get_schema(self, stream, out):
        code = super()._get_schema(stream, out)
        if self.poke_schema:
            self.poke_schema(_first_field(out))
        return code

    def _release_one(self, pointer):
        self.pokes.undo()
        super()._release_one(pointer)


class PokingSource(PythonStreamSource):
    """One batch, whose first column `poke(column, pokes)` changes after the export.

    `poke_schema(field)` changes the first field of every schema exported, for a type pyarrow cannot build.
    """

    def __init__(self, table, poke, poke_schema=None) -> None:
        super().__init__(table, max_chunksize=table.num_rows)
        self.poke = poke
        self.poke_schema = poke_schema

    def __arrow_c_schema__(self) -> object:
        capsule = super().__arrow_c_schema__()
        if self.poke_schema:
            self.poke_schema(_first_field(_capsule_pointer(id(capsule), b"arrow_schema")))
        return capsule

    def stream(self, columns, filters):
        stream = PokingStream(
            self.table.schema,
            self.table.to_batches(max_chunksize=self.max_chunksize),
            self.poke,
            self.poke_schema,
        )
        self.streams.append(stream)
        return stream.capsule(), False


def _struct_table():
    return pa.table({"s": pa.StructArray.from_arrays([pa.array(range(2_000))], ["x"])})


def _run_end_array(rows):
    return pa.RunEndEncodedArray.from_arrays(pa.array([rows // 2, rows], pa.int32()), pa.array([7, 8], pa.int64()))


def _run_end_table():
    return pa.table({"r": _run_end_array(2_000)})


def _child(column, index):
    return column.children[index].contents


def _drop_the_validity_bitmap(column, pokes):
    # The export may leave the null count unknown, which the scan accepts without a bitmap.
    pokes.set(ctypes.c_int64, _field(column, "null_count"), 1_000)
    pokes.set(ctypes.c_void_p, column.buffers, None)


# One malformed batch each: the table, the change to its first column after the export, the text the refusal must
# carry, and for a type pyarrow cannot build, the change to the first field of each exported schema.
MALFORMED_NESTED = {
    "short_struct_child": (
        _struct_table,
        lambda column, pokes: pokes.set(ctypes.c_int64, _field(_child(column, 0), "length"), 1),
        "column 0 holds a nested array of 1 rows where 2000 were expected",
    ),
    "nested_offset_overflow": (
        _struct_table,
        lambda column, pokes: pokes.set(ctypes.c_int64, _field(_child(column, 0), "offset"), 2**63 - 1),
        "column 0 with a nested array of offset 9223372036854775807 and length 2000",
    ),
    "missing_child": (
        _struct_table,
        lambda column, pokes: pokes.set(ctypes.c_void_p, ctypes.cast(column.children, ctypes.c_void_p).value, None),
        "column 0 with a nested array missing its child 0",
    ),
    "extra_child": (
        _struct_table,
        lambda column, pokes: pokes.set(ctypes.c_int64, _field(column, "n_children"), 2),
        "column 0 with a struct or union of 2 child arrays where its type has 1",
    ),
    "released_child": (
        _struct_table,
        lambda column, pokes: pokes.set(ctypes.c_void_p, _field(_child(column, 0), "release"), None),
        "column 0 with a nested array whose child 0 is released already",
    ),
    "released_dictionary": (
        lambda: pa.table({"d": pa.array(["a", "b"] * 1_000).dictionary_encode()}),
        lambda column, pokes: pokes.set(ctypes.c_void_p, column.dictionary + _ArrowArray.release.offset, None),
        "column 0 with a dictionary-encoded array whose dictionary is released already",
    ),
    "run_ends_out_of_range": (
        _run_end_table,
        lambda column, pokes: pokes.set(ctypes.c_int64, _field(_child(column, 0), "offset"), -1),
        "column 0 with a nested array of offset -1 and length 2",
    ),
    "run_ends_and_values_unequal": (
        _run_end_table,
        lambda column, pokes: pokes.set(ctypes.c_int64, _field(_child(column, 1), "length"), 1),
        "column 0 with a run end encoded array of 2 run ends and 1 values",
    ),
    "list_offsets_beyond_child": (
        lambda: pa.table({"l": pa.array([[i] for i in range(2_000)])}),
        lambda column, pokes: pokes.set(ctypes.c_int64, _field(_child(column, 0), "length"), 1_999),
        "column 0 with a list whose offsets 0 to 2000 do not fit its child of 1999 rows",
    ),
    "list_view_without_sizes": (
        lambda: pa.table({"l": pa.array([[1], [2, 3]] * 1_000, type=pa.list_view(pa.int64()))}),
        lambda column, pokes: pokes.set(ctypes.c_void_p, column.buffers + 2 * ctypes.sizeof(ctypes.c_void_p), None),
        "column 0 with a list view that has no offsets or sizes",
    ),
    "run_ends_stop_before_the_rows": (
        lambda: pa.table({"l": pa.FixedSizeListArray.from_arrays(_run_end_array(6), 2).slice(1)}),
        lambda column, pokes: [
            pokes.set(ctypes.c_int64, _field(_child(_child(column, 0), index), "length"), 1) for index in (0, 1)
        ],
        "column 0 with a run end encoded array whose runs end at 3 before its 6 rows",
    ),
    "list_view_past_its_child": (
        lambda: pa.table(
            {
                "l": pa.ListViewArray.from_arrays(
                    pa.array([1], pa.int32()), pa.array([1], pa.int32()), pa.array([10, 99])
                )
            }
        ),
        lambda column, pokes: pokes.set(ctypes.c_int64, _field(_child(column, 0), "length"), 1),
        "column 0 with a list view whose list at row 0 reaches past its child of 1 rows",
    ),
    "run_ends_of_tinyint": (
        lambda: pa.table(
            {"r": pa.RunEndEncodedArray.from_arrays(pa.array([4], pa.int16()), pa.array([7], pa.int64()))}
        ),
        None,
        "column 0 with a run end encoded array whose run ends are TINYINT, not SMALLINT, INTEGER or BIGINT",
        # The one int16 run end of 4 reads as the int8 run end 4 on a little endian machine.
        lambda field: ctypes.memmove(_ArrowSchemaHead.from_address(field.children[0]).format, b"c", 1),
    ),
    "null_count_without_bitmap": (
        lambda: pa.table({"i": pa.array([1, None] * 1_000)}),
        _drop_the_validity_bitmap,
        "column 0 with a non-zero null count but no validity bitmap",
    ),
    "struct_child_null_count_without_bitmap": (
        lambda: pa.table({"s": pa.StructArray.from_arrays([pa.array([1, None] * 1_000)], ["x"])}),
        lambda column, pokes: _drop_the_validity_bitmap(_child(column, 0), pokes),
        "column 0 with a non-zero null count but no validity bitmap",
    ),
    "list_child_null_count_without_bitmap": (
        lambda: pa.table({"l": pa.array([[1, None]] * 1_000)}),
        lambda column, pokes: _drop_the_validity_bitmap(_child(column, 0), pokes),
        "column 0 with a non-zero null count but no validity bitmap",
    ),
}


class TestMalformedAndUnusualInput:
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        ("corrupt", "message"),
        [
            pytest.param(
                _overflowing_root_offset,
                "an Arrow batch has offset 9223372036854775807 and length 1000, which do not describe a range of rows",
                id="root_offset_overflow",
            ),
            pytest.param(
                _negative_root_length,
                "an Arrow batch has offset 0 and length -1, which do not describe a range of rows",
                id="negative_root_length",
            ),
            pytest.param(
                _overflowing_column_offset,
                "an Arrow batch carried column 0 with offset 9223372036854775807 and length 1000",
                id="column_offset_overflow",
            ),
        ],
    )
    def test_an_impossible_range_is_refused_and_released(self, corrupt, message):
        from _pushdown_helpers import eventually

        source = PythonStreamSource(TestReleases.TABLE, corrupt=corrupt)
        with duckdb.connect(config={"threads": THREADS}) as con:
            con._register_arrow_source("t", source)
            with pytest.raises(duckdb.InvalidInputException, match=re.escape(message)):
                con.execute("SELECT * FROM t").fetchall()
            assert eventually(lambda: source.counts() == [(0, 1)]), source.counts()

    @staticmethod
    def _x() -> object:
        return pa.array([10, 20, 30, 40, 50])

    @pytest.mark.parametrize(
        "layout",
        [
            pytest.param(
                lambda x: pa.StructArray.from_arrays(
                    [pa.StructArray.from_arrays([pa.StructArray.from_arrays([x], ["x"])], ["b"]).slice(1, 4)], ["a"]
                ),
                id="struct_below_a_sliced_struct",
            ),
            pytest.param(
                lambda x: pa.StructArray.from_arrays(
                    [pa.StructArray.from_arrays([pa.StructArray.from_arrays([x], ["x"])], ["b"]).slice(1, 4)], ["a"]
                ).slice(1, 2),
                id="struct_below_a_sliced_struct_in_a_sliced_batch",
            ),
            pytest.param(
                lambda x: pa.StructArray.from_arrays(
                    [
                        pa.ListArray.from_arrays(
                            pa.array([0, 2, 4], pa.int32()), pa.StructArray.from_arrays([x], ["x"]).slice(1, 4)
                        )
                    ],
                    ["l"],
                ),
                id="sliced_struct_in_a_list",
            ),
            pytest.param(
                lambda x: pa.StructArray.from_arrays(
                    [
                        pa.UnionArray.from_sparse(
                            pa.array([0, 0, 1, 0, 0], pa.int8()),
                            [pa.StructArray.from_arrays([x], ["x"]), pa.array(list("abcde"))],
                            ["st", "s"],
                        ).slice(1, 3)
                    ],
                    ["a"],
                ),
                id="struct_in_a_sliced_union",
            ),
            pytest.param(
                lambda x: pa.StructArray.from_arrays(
                    [
                        pa.ListArray.from_arrays(
                            pa.array([0, 2, 3], pa.int32()),
                            pa.UnionArray.from_sparse(
                                pa.array([0, 1, 0, 1, 0], pa.int8()), [x, pa.array(list("abcde"))], ["i", "s"]
                            ).slice(1, 3),
                        )
                    ],
                    ["l"],
                ),
                id="sliced_union_in_a_list",
            ),
            pytest.param(
                lambda x: pa.StructArray.from_arrays(
                    [pa.StructArray.from_arrays([pa.StructArray.from_arrays([x], ["x"]).slice(1, 4)], ["b"])], ["a"]
                ),
                id="sliced_struct_below_a_struct",
            ),
            pytest.param(
                lambda x: pa.StructArray.from_arrays(
                    [
                        pa.ListArray.from_arrays(
                            pa.array([0, 2, 4, 5], pa.int32()), pa.StructArray.from_arrays([x], ["x"])
                        ).slice(1, 2)
                    ],
                    ["l"],
                ),
                id="struct_in_a_sliced_list",
            ),
            pytest.param(
                lambda x: pa.StructArray.from_arrays(
                    [
                        pa.StructArray.from_arrays(
                            [
                                pa.UnionArray.from_sparse(
                                    pa.array([0, 1, 0, 1, 0], pa.int8()), [x, pa.array(list("abcde"))], ["i", "s"]
                                )
                            ],
                            ["u"],
                        ).slice(1, 3)
                    ],
                    ["a"],
                ),
                id="union_below_a_sliced_struct",
            ),
            pytest.param(
                lambda x: pa.StructArray.from_arrays([pa.StructArray.from_arrays([x], ["x"]).slice(1, 3)], ["a"]),
                id="sliced_struct_column",
            ),
        ],
    )
    def test_an_offset_the_conversion_keeps_reads_the_right_rows(self, layout):
        array = layout(self._x())
        with duckdb.connect() as con:
            con._register_arrow_source("s", ArraySource(array))
            assert con.execute("SELECT * FROM s").fetchall() == [tuple(row.values()) for row in array.to_pylist()]

    def test_a_variant_is_checked_as_the_struct_it_is_stored_as(self):
        with duckdb.connect() as con:
            table = con.sql("SELECT {'v': x::VARIANT} AS s, x::VARIANT AS v, x FROM range(10) t(x)").to_arrow_table()
            con._register_arrow_source("t", TableSource(table.slice(2)))
            assert con.execute("SELECT v::BIGINT, x FROM t").fetchall() == [(x, x) for x in range(2, 10)]
            assert con.execute("SELECT s FROM t").fetchall() == [({"v": x},) for x in range(2, 10)]

    @pytest.mark.parametrize(
        ("make", "expected"),
        [
            pytest.param(
                lambda: pa.ListArray.from_arrays(
                    pa.array([5, 5, 5], pa.int32()), pa.array(list("abcde")).dictionary_encode()
                ),
                [([],), ([],)],
                id="empty_lists_from_offset_five",
            ),
            pytest.param(
                lambda: pa.array([[], [], ["a"]], type=pa.list_(pa.dictionary(pa.int32(), pa.string()))),
                [([],), ([],), (["a"],)],
                id="empty_lists_then_a_value",
            ),
            pytest.param(
                lambda: pa.array([[], None, []], type=pa.map_(pa.string(), pa.dictionary(pa.int32(), pa.string()))),
                [({},), (None,), ({},)],
                id="empty_maps",
            ),
            pytest.param(
                lambda: pa.array([[], None, []], type=pa.list_(pa.dictionary(pa.int32(), pa.string()))),
                [([],), (None,), ([],)],
                id="only_empty_lists",
            ),
            pytest.param(
                lambda: pa.array([[], None, []], type=pa.large_list(pa.dictionary(pa.int32(), pa.string()))),
                [([],), (None,), ([],)],
                id="only_empty_large_lists",
            ),
            pytest.param(
                lambda: pa.ListViewArray.from_arrays(
                    pa.array([5, 5], pa.int32()),
                    pa.array([0, 0], pa.int32()),
                    pa.array(list("abcde")).dictionary_encode(),
                ),
                [([],), ([],)],
                id="only_empty_list_views_from_offset_five",
            ),
            pytest.param(
                lambda: pa.array([[], None, []], type=pa.list_view(pa.dictionary(pa.int32(), pa.string()))),
                [([],), (None,), ([],)],
                id="only_empty_list_views",
            ),
        ],
    )
    def test_empty_lists_over_a_dictionary_the_conversion_reads_are_read(self, make, expected):
        with duckdb.connect() as con:
            con._register_arrow_source("t", TableSource(pa.table({"c": make()})))
            assert con.execute("SELECT c FROM t").fetchall() == expected

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        ("make", "expected"),
        [
            pytest.param(
                lambda: pa.ListArray.from_arrays(pa.array([0, 2, 4], pa.int32()), _run_end_array(4)),
                [([7, 7],), ([8, 8],)],
                id="list",
            ),
            pytest.param(
                lambda: pa.LargeListArray.from_arrays(pa.array([0, 2, 4], pa.int64()), _run_end_array(4)),
                [([7, 7],), ([8, 8],)],
                id="large_list",
            ),
            pytest.param(
                lambda: pa.MapArray.from_arrays(
                    pa.array([0, 2, 4], pa.int32()), pa.array(["a", "b", "c", "d"]), _run_end_array(4)
                ),
                [({"a": 7, "b": 7},), ({"c": 8, "d": 8},)],
                id="map",
            ),
            pytest.param(
                lambda: pa.ListArray.from_arrays(
                    pa.array([0, 2, 4], pa.int32()), pa.StructArray.from_arrays([_run_end_array(4)], ["r"])
                ),
                [([{"r": 7}, {"r": 7}],), ([{"r": 8}, {"r": 8}],)],
                id="struct_in_a_list",
            ),
            pytest.param(
                lambda: pa.FixedSizeListArray.from_arrays(pa.StructArray.from_arrays([_run_end_array(4)], ["r"]), 2),
                [(({"r": 7}, {"r": 7}),), (({"r": 8}, {"r": 8}),)],
                id="struct_in_a_fixed_size_list",
            ),
            pytest.param(
                lambda: pa.FixedSizeListArray.from_arrays(_run_end_array(4), 2),
                [((7, 7),), ((8, 8),)],
                id="fixed_size_list",
            ),
            pytest.param(
                lambda: pa.FixedSizeListArray.from_arrays(_run_end_array(6), 2, mask=pa.array([False, True, False])),
                [((7, 7),), (None,), ((8, 8),)],
                id="fixed_size_list_with_a_null_row",
            ),
            pytest.param(
                lambda: pa.FixedSizeListArray.from_arrays(_run_end_array(6), 2).slice(1),
                [((7, 8),), ((8, 8),)],
                id="sliced_fixed_size_list",
            ),
        ],
    )
    def test_a_run_end_encoded_array_below_a_list_the_conversion_reads_is_read(self, make, expected):
        with duckdb.connect() as con:
            con._register_arrow_source("t", TableSource(pa.table({"c": make()})))
            assert con.execute("SELECT c FROM t").fetchall() == expected

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("case", list(MALFORMED_NESTED))
    def test_a_malformed_nested_array_is_refused_and_released(self, case):
        # In a subprocess, since a batch that got past the checks could take the process down.
        script = f"""
import importlib.util, sys, time
spec = importlib.util.spec_from_file_location("scan_core", {str(Path(__file__))!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import duckdb
make, poke, message, *poke_schema = module.MALFORMED_NESTED[{case!r}]
source = module.PokingSource(make(), poke, *poke_schema)
con = duckdb.connect(config={{"threads": 4}})
con._register_arrow_source("t", source)
try:
    con.execute("SELECT * FROM t").fetchall()
    print("read", flush=True)
except duckdb.InvalidInputException as error:
    print("refused", message in str(error), flush=True)
    print(error, file=sys.stderr)
deadline = time.monotonic() + 5
while source.counts() != [(0, 1)] and time.monotonic() < deadline:
    time.sleep(0.01)
print("unreleased", source.counts(), flush=True)
print(con.execute("SELECT 42").fetchall(), flush=True)
con.close()
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=110, check=False
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert completed.stdout.splitlines()[-3:] == ["refused True", "unreleased [(0, 1)]", "[(42,)]"], (
            completed.stderr[-2000:]
        )

    @pytest.mark.timeout(60)
    def test_a_failed_pull_without_a_message_fails(self):
        class SilentFailure(CountingStream):
            def __init__(self) -> None:
                super().__init__(pa.schema([("x", pa.int64())]), [])
                self.message = ctypes.create_string_buffer(b"")
                self.last_error = _LAST_ERROR(lambda _: ctypes.addressof(self.message))
                self.struct.get_last_error = ctypes.cast(self.last_error, ctypes.c_void_p)

            def _get_next(self, _, out):
                _ArrowArray.from_address(out).release = None
                return 22

        stream = SilentFailure()

        class Source(PythonStreamSource):
            def stream(self, columns, filters):
                return stream.capsule(), False

        with duckdb.connect() as con:
            con._register_arrow_source("s", Source(pa.table({"x": [1]})))
            with pytest.raises(
                duckdb.InvalidInputException,
                match=re.escape("reading the stream registered as 's' failed: no error message (error code 22)"),
            ):
                con.execute("SELECT * FROM s").fetchall()

    def test_a_nameless_array_with_a_null_name_is_named_value(self):
        class NullName(ArraySource):
            def stream(self, columns, filters):
                schema, array = self.obj.__arrow_c_array__()
                _schema_head(schema).name = None
                return (schema, array), False

        with duckdb.connect() as con:
            con._register_arrow_source("s", NullName(pa.array([42, None])))
            assert con.execute("DESCRIBE s").fetchall()[0][:2] == ("value", "BIGINT")
            assert con.execute("SELECT value FROM s").fetchall() == [(42,), (None,)]

    def test_a_column_with_a_null_name_takes_the_engine_name(self):
        def without_first_name(schema):
            _ArrowSchemaHead.from_address(_schema_head(schema).children[0]).name = None
            return schema

        class NullColumnName(ArraySource):
            def __arrow_c_schema__(self) -> object:
                return without_first_name(self.obj.schema.__arrow_c_schema__())

            def stream(self, columns, filters):
                schema, array = self.obj.__arrow_c_array__()
                return (without_first_name(schema), array), False

        batch = pa.record_batch([pa.array([1, 2]), pa.array(["a", "b"])], names=["n", "s"])
        with duckdb.connect() as con:
            con._register_arrow_source("s", NullColumnName(batch))
            assert [row[0] for row in con.execute("DESCRIBE s").fetchall()] == ["v0", "s"]
            assert con.execute("SELECT * FROM s").fetchall() == [(1, "a"), (2, "b")]

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        ("route", "asks", "under_gil"),
        [
            pytest.param("capsule", None, False, id="capsule"),
            pytest.param("pyarrow_reader", None, False, id="pyarrow_reader"),
            pytest.param("capsule", True, True, id="capsule_asking"),
            pytest.param("capsule", 1, False, id="capsule_asking_with_one"),
        ],
    )
    def test_the_gil_around_a_foreign_stream(self, route, asks, under_gil):
        """A stream is pulled without the GIL unless its source's pull_under_gil is exactly True.

        A producer whose callbacks run Python takes the GIL itself, as pyarrow's, ctypes' and Cython's callbacks do;
        holding it across a pull instead deadlocks a producer whose own threads need Python.
        """
        # The stream's get_next is PyGILState_Check, which answers 1, an error code, only when called with the GIL.
        stream = CountingStream(pa.schema([("x", pa.int64())]), [])
        stream.struct.get_next = ctypes.cast(ctypes.pythonapi.PyGILState_Check, ctypes.c_void_p)
        capsule = stream.capsule()
        obj = capsule if route == "capsule" else pa.RecordBatchReader._import_from_c_capsule(capsule)

        class Foreign:
            def __init__(self, obj) -> None:
                self.obj = obj

            def rows(self):
                return None

            def accepts(self, predicate):
                return False

            def __arrow_c_schema__(self) -> object:
                return pa.schema([("x", pa.int64())]).__arrow_c_schema__()

            def stream(self, columns, filters):
                return (self.obj if route == "capsule" else self.obj.__arrow_c_stream__()), False

        if asks is not None:
            Foreign.pull_under_gil = asks
        with duckdb.connect(config={"threads": 1}) as con:
            con._register_arrow_source("s", Foreign(obj))
            if under_gil:
                with pytest.raises(duckdb.InvalidInputException, match="reading the stream registered as 's' failed"):
                    con.execute("SELECT * FROM s").fetchall()
            else:
                # Without the GIL the call answers 0 and leaves the array released, which ends the stream.
                assert con.execute("SELECT * FROM s").fetchall() == []

    @pytest.mark.timeout(120)
    def test_a_scanner_reader_over_a_slow_python_source(self):
        # Pulled under the GIL, the scan would wait for the scanner while the scanner's pool thread waits for the GIL.
        script = """
import time
import duckdb, pyarrow as pa, pyarrow.dataset as ds
schema = pa.schema([("i", pa.int64())])
def batches():
    for index in range(20):
        time.sleep(0.05)
        yield pa.record_batch([pa.array(range(index * 100, index * 100 + 100))], schema=schema)
def run(threads, entry):
    slow_reader = ds.Scanner.from_batches(batches(), schema=schema).to_reader()
    with duckdb.connect(config={"threads": threads}) as con:
        con.execute("SET enable_progress_bar = false")
        if entry == "register":
            con.register("slow_reader", slow_reader)
        elif entry == "capsule":
            con.register("slow_reader", slow_reader.__arrow_c_stream__())
        return con.execute("SELECT count(*), sum(i) FROM slow_reader").fetchall()
for threads in (1, 4):
    for entry in ("register", "variable", "capsule"):
        print(threads, entry, run(threads, entry), flush=True)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert completed.stdout.splitlines()[-6:] == [
            f"{threads} {entry} [(2000, 1999000)]"
            for threads in (1, 4)
            for entry in ("register", "variable", "capsule")
        ]

    @pytest.mark.timeout(120)
    def test_a_csv_reader_over_a_python_file(self):
        # The reader's read ahead thread reads the Python file under the GIL, which a pull holding the GIL would keep.
        script = """
import io
import duckdb, pyarrow.csv as csv
data = ("a,b\\n" + "".join(f"{i},x{i}\\n" for i in range(300_000))).encode()
for threads in (1, 4):
    reader = csv.open_csv(io.BytesIO(data), read_options=csv.ReadOptions(block_size=65_536))
    with duckdb.connect(config={"threads": threads}) as con:
        con.execute("SET enable_progress_bar = false")
        con.register("csv_reader", reader)
        print(threads, con.execute("SELECT count(*), sum(a) FROM csv_reader").fetchall(), flush=True)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert completed.stdout.splitlines()[-2:] == [f"{threads} [(300000, 44999850000)]" for threads in (1, 4)]


class TestReleases:
    TABLE = pa.table({"i": pa.array(range(20_000), pa.int64()), "s": pa.array([str(i) for i in range(20_000)])})

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    @pytest.mark.parametrize("query", ["SELECT count(*), sum(i) FROM t", "SELECT i FROM t LIMIT 3", "SELECT s FROM t"])
    def test_every_adopted_array_is_released_once(self, query, threads):
        source = PythonStreamSource(self.TABLE)
        with duckdb.connect(config={"threads": threads}) as con:
            con._register_arrow_source("t", source)
            con.execute(query).fetchall()
            assert source.counts() == [(0, 1)]
        assert source.streams[0].exported > 0

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        ("corrupt", "message"),
        [
            (
                _shorten_first_column,
                "Arrow array child length does not match the array length:"
                " column 0 holds 1 rows where 1000 were expected",
            ),
            (_wrong_column_count, "an Arrow batch carried 1 columns where 2 were expected"),
        ],
    )
    def test_a_malformed_array_is_refused_and_released(self, corrupt, message):
        from _pushdown_helpers import eventually

        source = PythonStreamSource(self.TABLE, corrupt=corrupt)
        with duckdb.connect(config={"threads": THREADS}) as con:
            con._register_arrow_source("t", source)
            with pytest.raises(duckdb.InvalidInputException, match=message):
                con.execute("SELECT * FROM t").fetchall()
            assert eventually(lambda: source.counts() == [(0, 1)]), source.counts()

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    @pytest.mark.parametrize(
        ("length", "message"),
        [
            (-1, "an Arrow batch has offset 0 and length -1, which do not describe a range of rows"),
            (-300_000, "an Arrow batch has offset 0 and length -300000, which do not describe a range of rows"),
            (
                400_000,
                "Arrow array child length does not match the array length:"
                " column 0 holds 300000 rows where 400000 were expected",
            ),
        ],
        ids=["minus_one", "minus_several_ranges", "longer_than_its_column"],
    )
    def test_a_single_array_with_an_impossible_length_is_refused(self, length, message, threads):
        from _pushdown_helpers import eventually

        array = pa.StructArray.from_arrays([pa.array(range(300_000), pa.int64())], names=["a"])
        source = CountingArraySource(array, length)
        with duckdb.connect(config={"threads": threads}) as con:
            con._register_arrow_source("t", source)
            with pytest.raises(duckdb.InvalidInputException, match=re.escape(message)):
                con.execute("SELECT count(*), sum(a) FROM t").fetchall()
            assert eventually(lambda: source.counts() == (0, 0)), source.counts()
            assert con.execute("SELECT 42").fetchall() == [(42,)]
        assert source.exported > 0

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    @pytest.mark.parametrize("query", ["SELECT count(*), sum(i) FROM t", "SELECT i FROM t LIMIT 3", "SELECT s FROM t"])
    def test_a_single_array_read_in_ranges_is_released_once(self, query, threads):
        source = CountingArraySource(ranged_struct_array())
        with duckdb.connect(config={"threads": threads}) as con:
            con._register_arrow_source("t", source)
            con.execute(query).fetchall()
            assert source.counts() == (0, 0)
        assert source.exported > 0

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("threads", [1, THREADS])
    def test_a_single_array_read_in_ranges_is_released_once_when_its_result_closes_half_fetched(self, threads):
        from _pushdown_helpers import eventually

        source = CountingArraySource(ranged_struct_array())
        with duckdb.connect(config={"threads": threads}) as con:
            con._register_arrow_source("t", source)
            result = con.execute("SELECT i, s FROM t")
            assert len(result.fetchmany(10)) == 10
            result.close()
            assert eventually(lambda: source.counts() == (0, 0)), source.counts()
        assert source.exported > 0

    @pytest.mark.timeout(60)
    def test_a_conversion_failure_releases_every_array(self):
        from _pushdown_helpers import eventually

        table = pa.concat_tables(
            [pa.table({"d": pa.array([1] * 5000, pa.duration("s"))}), pa.table({"d": OVERFLOWING_DURATION})]
        )
        source = PythonStreamSource(table)
        with duckdb.connect(config={"threads": THREADS}) as con:
            con._register_arrow_source("t", source)
            with pytest.raises(duckdb.ConversionException, match="Could not convert Interval to Microsecond"):
                con.execute("SELECT count(d) FROM t").fetchall()
            assert eventually(lambda: source.counts() == [(0, 1)]), source.counts()

    @pytest.mark.timeout(60)
    def test_a_retained_result_holds_its_arrays_until_it_goes(self):
        source = PythonStreamSource(self.TABLE)
        con = duckdb.connect(config={"threads": THREADS})
        con._register_arrow_source("t", source)
        result = con.execute("SELECT i FROM t")
        assert result.fetchone() == (0,)
        assert source.counts()[0][1] == 0
        con.close()
        del result
        assert source.counts() == [(0, 1)]

    @pytest.mark.timeout(60)
    def test_unregister_and_close_release_everything(self):
        source = PythonStreamSource(self.TABLE)
        con = duckdb.connect(config={"threads": THREADS})
        con._register_arrow_source("t", source)
        assert con.execute("SELECT count(*) FROM t").fetchall() == [(20_000,)]
        con.unregister("t")
        con.close()
        assert source.counts() == [(0, 1)]

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("state", ["unread", "retained_result", "registered_reader", "reader_read_with_limit"])
    def test_interpreter_exit(self, state):
        script = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("scan_core", {str(Path(__file__))!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import duckdb, pyarrow as pa
table = pa.table({{"i": list(range(50_000))}})
con = duckdb.connect(config={{"threads": 4}})
if {state!r} in ("registered_reader", "reader_read_with_limit"):
    con._register_arrow_source("t", module.ReaderSource(module.generator_reader(table, max_chunksize=100)))
else:
    con._register_arrow_source("t", module.PythonStreamSource(table, max_chunksize=100))
if {state!r} == "retained_result":
    result = con.execute("SELECT i FROM t")
    print(result.fetchone())
if {state!r} == "reader_read_with_limit":
    print(con.execute("SELECT i FROM t LIMIT 3").fetchall())
keep = [con]
print("exiting")
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=110, check=False
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert "exiting" in completed.stdout
        if state != "retained_result":
            assert "nanobind: leaked" not in completed.stderr

    @pytest.mark.timeout(120)
    def test_a_scan_in_a_late_exit_handler_releases_its_arrays(self):
        script = f"""
import atexit
def handler():
    import duckdb
    source = module.SchemaCountingSource(module.pa.table({{"i": list(range(20_000))}}), max_chunksize=1000)
    with duckdb.connect(config={{"threads": 4}}) as con:
        con._register_arrow_source("t", source)
        print(con.execute("SELECT count(*) FROM t").fetchall(), flush=True)
    print("unreleased", source.counts(), flush=True)
    print("schemas", source.schemas.wrapped > 0, source.schemas.wrapped - source.schemas.released, flush=True)
# Registered before duckdb is imported, so it runs after the scan's own exit hook.
atexit.register(handler)
import importlib.util
spec = importlib.util.spec_from_file_location("scan_core", {str(Path(__file__))!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=110, check=False
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert completed.stdout.splitlines()[-3:] == ["[(20000,)]", "unreleased [(0, 1)]", "schemas True 0"]

    @pytest.mark.timeout(120)
    def test_exit_waits_for_a_pull_longer_than_five_seconds(self):
        script = f"""
import importlib.util, threading, time
spec = importlib.util.spec_from_file_location("scan_core", {str(Path(__file__))!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import duckdb, pyarrow as pa
entered = threading.Event()
armed = threading.Event()
def batches():
    for i in range(1_000_000):
        if armed.is_set():
            entered.set()
            time.sleep(6)
            print("pull returned", flush=True)
            return
        time.sleep(0.005)
        yield pa.record_batch([pa.array(range(1000))], names=["i"])
con = duckdb.connect(config={{"threads": 4}})
reader = pa.RecordBatchReader.from_batches(pa.schema([("i", pa.int64())]), batches())
con._register_arrow_source("r", module.ReaderSource(reader))
con.execute("SELECT * FROM r")
print(con.fetchone(), flush=True)
armed.set()
assert entered.wait(30)
print("exiting", flush=True)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=110, check=False
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        # "pull returned" shows the exit waited for the pull; the main thread may print "exiting" before or after it.
        assert {"exiting", "pull returned"} <= set(completed.stdout.splitlines())

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
    @pytest.mark.timeout(120)
    def test_a_child_forked_during_a_pull_exits(self):
        # The pull runs on a thread the child does not have, so the child's exit hook must not wait for it.
        script = """
import os, signal, sys, threading, time
import duckdb, pyarrow as pa
schema = pa.schema([("n", pa.int64())])
pulling = threading.Event()
def batches():
    yield pa.record_batch([pa.array([0])], schema=schema)
    pulling.set()
    time.sleep(30)
    yield pa.record_batch([pa.array([1])], schema=schema)
con = duckdb.connect()
con.register("slow", pa.RecordBatchReader.from_batches(schema, batches()))
threading.Thread(target=lambda: con.execute("SELECT sum(n) FROM slow").fetchall(), daemon=True).start()
assert pulling.wait(30)
pid = os.fork()
if pid == 0:
    sys.exit(0)
deadline = time.monotonic() + 5
while time.monotonic() < deadline:
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        print("child exited", os.waitstatus_to_exitcode(status), flush=True)
        break
    time.sleep(0.05)
else:
    print("child still running", flush=True)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
os._exit(0)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert completed.stdout.splitlines()[-1:] == ["child exited 0"], completed.stderr[-2000:]


class TestDictionaryHeavyTiming:
    def test_a_dictionary_is_decoded_once_per_array(self):
        distinct = 200_000
        rows = 2_000_000
        values = pa.array([f"value-{i:08d}" for i in range(distinct)])
        indices = pa.array(np.random.default_rng(0).integers(0, distinct, rows, dtype=np.int32))
        table = pa.Table.from_batches(
            pa.table({"d": pa.DictionaryArray.from_arrays(indices, values)}).to_batches(max_chunksize=1_000_000)
        )
        plain = pa.Table.from_batches(
            pa.table({"d": table.column("d").cast(pa.string())}).to_batches(max_chunksize=1_000_000)
        )
        query = "SELECT count(DISTINCT d) FROM {}"

        def timed(con, name):
            start = time.perf_counter()
            con.execute(query.format(name)).fetchall()
            return time.perf_counter() - start

        with duckdb.connect(config={"threads": 1}) as con:
            con._register_arrow_source("dictionary", TableSource(table))
            con._register_arrow_source("plain", TableSource(plain))
            used = pc.count_distinct(indices).as_py()
            assert con.execute(query.format("dictionary")).fetchall() == [(used,)]
            assert con.execute(query.format("plain")).fetchall() == [(used,)]
            # Alternated, so a slow stretch of the machine falls on both kinds of run alike.
            rounds = [(timed(con, "dictionary"), timed(con, "plain")) for _ in range(3)]
            dictionary = min(pair[0] for pair in rounds)
            decoded = min(pair[1] for pair in rounds)
        # Decoding the dictionary once per vector instead of once per array is many times slower than reading the
        # same strings without a dictionary; once per array it costs about one and a half times as much.
        assert dictionary < 10 * decoded, (dictionary, decoded)

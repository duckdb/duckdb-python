"""map() output must stay valid after the Python function returns, and each call must release its DataFrames."""

import ctypes
import gc
import importlib.util
import json
import math
import re
import subprocess
import sys
import threading
import tracemalloc
import weakref

import pytest

import duckdb
import numpy as np
import pandas as pd

ROWS = 5000
# With one thread and a 32KB streaming buffer, a query over this many rows still runs after its first fetch.
RUNNING_SOURCE_ROWS = 10_000_000
PREFIX = "a string longer than the twelve inlined bytes "
LONG = f"'{PREFIX}' || i::varchar"

needs_pyarrow = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")

_released_buffers: list[np.ndarray] = []


def _poisoned(values):
    """int64 array over a buffer that is filled with -1 instead of freed, once its last reference is gone."""
    backing = np.array(values, dtype=np.int64)
    array = np.frombuffer(memoryview(backing), dtype=np.int64)
    weakref.finalize(array.base, lambda: (backing.fill(-1), _released_buffers.append(backing)))
    return array


def _checked(frame, check):
    """Return frame once check holds for its first column, so that a case cannot drift onto another scan path."""
    column = frame.iloc[:, 0]
    assert check(column), f"{column.name} has dtype {column.dtype}"
    return frame


def _modify_in_place(df):
    df["a"] = df["a"] * 3
    return df


def _create_in_place(df):
    df["b"] = df["a"] * 3
    return df


def _create_poisoned_in_place(df):
    df["b"] = pd.Series(_poisoned(df["a"] * 3), copy=False)
    return df


VALUE_CASES = [
    pytest.param("i as a, i::double as d", lambda df: df, "i as a, i::double as d", id="unmodified_columns"),
    pytest.param("i as a", _modify_in_place, "i * 3 as a", id="modified_in_place"),
    pytest.param(
        "i as a",
        lambda df: df.assign(a=df["a"] * 3),
        "i * 3 as a",
        id="modified_in_new_frame",
    ),
    pytest.param("i as a", _create_in_place, "i as a, i * 3 as b", id="created_in_place"),
    pytest.param(
        "i as a",
        lambda df: df.assign(b=df["a"] * 3),
        "i as a, i * 3 as b",
        id="created_in_new_frame",
    ),
    pytest.param(
        "i as a",
        lambda df: pd.DataFrame({"b": df["a"].to_numpy() * 3}),
        "i * 3 as b",
        id="created_in_unrelated_frame",
    ),
    pytest.param("i as a, i * 2 as b", lambda df: df[["b"]], "i * 2 as b", id="column_subset_view"),
    pytest.param(f"{LONG} as s", lambda df: df, f"{LONG} as s", id="strings_unmodified"),
    pytest.param(
        f"{LONG} as s", lambda df: pd.DataFrame({"s": df["s"] + " out"}), f"{LONG} || ' out' as s", id="strings_created"
    ),
    pytest.param(
        f"{LONG} as s",
        lambda df: _checked(
            pd.DataFrame({"s": pd.Series([v + " out" for v in df["s"]], dtype=object)}),
            lambda column: column.dtype == object,
        ),
        f"{LONG} || ' out' as s",
        id="object_strings_created",
    ),
    pytest.param(
        f"{LONG} as s",
        lambda df: _checked(
            pd.DataFrame({"s": (df["s"] + " out").astype(pd.StringDtype("pyarrow"))}),
            lambda column: column.dtype == pd.StringDtype("pyarrow"),
        ),
        f"{LONG} || ' out' as s",
        id="arrow_strings_created",
        marks=needs_pyarrow,
    ),
    pytest.param("('blob ' || i::varchar)::blob as b", lambda df: df, "('blob ' || i::varchar)::blob as b", id="blobs"),
    pytest.param(
        f"case when i % 3 = 0 then null else i::double end as d, case when i % 3 = 0 then null else {LONG} end as s",
        lambda df: df,
        f"case when i % 3 = 0 then null else i::double end as d, case when i % 3 = 0 then null else {LONG} end as s",
        id="nulls_unmodified",
    ),
    pytest.param(
        "i as a",
        lambda df: pd.DataFrame({"d": np.where(df["a"] % 3 == 0, np.nan, df["a"] * 0.5)}),
        "case when i % 3 = 0 then null else i::double * 0.5 end as d",
        id="nulls_created_as_nan",
    ),
    pytest.param(
        "i as a",
        lambda df: pd.DataFrame({"a": pd.array(np.where(df["a"] % 3 == 0, None, df["a"]), dtype="Int64")}),
        "case when i % 3 = 0 then null else i end as a",
        id="nulls_created_as_mask",
    ),
    pytest.param(
        "i::double as d",
        lambda df: df.assign(d=(df["d"] % 100).astype("float16")),
        "(i % 100)::float as d",
        id="float16_converted_by_the_scan",
    ),
    pytest.param("i as a", _create_poisoned_in_place, "i as a, i * 3 as b", id="released_buffer_in_input_frame"),
    pytest.param(
        "i as a",
        lambda df: pd.DataFrame({"b": _poisoned(df["a"] * 3)}, copy=False),
        "i * 3 as b",
        id="released_buffer_in_new_frame",
    ),
]


def _assert_same_rows(actual, expected):
    assert len(actual) == len(expected)
    wrong = [i for i, (a, e) in enumerate(zip(actual, expected, strict=True)) if a != e]
    assert not wrong, f"{len(wrong)} rows differ, first at {wrong[0]}: {actual[wrong[0]]!r} != {expected[wrong[0]]!r}"


@pytest.mark.parametrize(("select", "function", "expected"), VALUE_CASES)
def test_output_is_what_the_function_returned(duckdb_cursor, select, function, expected):
    mapped = duckdb_cursor.sql(f"select {select} from range({ROWS}) t(i)").map(function)
    reference = duckdb_cursor.sql(f"select {expected} from range({ROWS}) t(i)")
    assert mapped.columns == reference.columns
    assert mapped.types == reference.types
    _assert_same_rows(mapped.fetchall(), reference.fetchall())


_utf8_of = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_void_p)(
    ("PyUnicode_AsUTF8AndSize", ctypes.pythonapi)
)


def _has_utf8_cache(text):
    # CPython keeps the UTF-8 form of a non ASCII str only once asked for it, which a fresh copy has not been.
    return sys.getsizeof(text) > sys.getsizeof(text.encode().decode())


def _non_ascii(values, *, utf8_cache):
    """Non ASCII str objects of one, two and four byte kinds, which the scan reads in place only with a UTF-8 cache."""
    texts = [["é ", "€ ", "😀 "][v % 3] + PREFIX + str(v) for v in values]
    if utf8_cache:
        for text in texts:
            _utf8_of(text, None)
    return _checked(
        pd.DataFrame({"s": pd.Series(texts, dtype=object)}),
        lambda column: column.dtype == object and all(_has_utf8_cache(text) == utf8_cache for text in column),
    )


def _strided(df):
    grid = _poisoned(np.column_stack([df["a"] * 3, df["a"]]).ravel()).reshape(-1, 2)
    return _checked(pd.DataFrame({"b": grid[:, 0]}, copy=False), lambda column: column.to_numpy().strides == (16,))


def _objects(values):
    return pd.Series(values, dtype=object)


TYPE_CASES = [
    pytest.param(lambda df: pd.DataFrame({"l": _objects([[v, v + 1] for v in df["a"]])}), id="list"),
    pytest.param(
        lambda df: pd.DataFrame(
            {"st": _objects([None if v % 3 == 0 else {"x": v, "y": PREFIX + str(v)} for v in df["a"]])}
        ),
        id="struct_with_null_parent",
    ),
    pytest.param(
        lambda df: pd.DataFrame({"m": _objects([{"key": [PREFIX + str(v)], "value": [v]} for v in df["a"]])}),
        id="map",
    ),
    pytest.param(
        lambda df: pd.DataFrame({"ls": _objects([[{"s": PREFIX + str(v)}, {"s": f"é {v}"}] for v in df["a"]])}),
        id="list_of_struct_with_strings",
    ),
    pytest.param(lambda df: _non_ascii(df["a"], utf8_cache=False), id="non_ascii_strings_decoded"),
    pytest.param(lambda df: _non_ascii(df["a"], utf8_cache=True), id="non_ascii_strings_read_in_place"),
    pytest.param(_strided, id="strided_numpy_column"),
    pytest.param(
        lambda df: pd.DataFrame({"t": pd.to_datetime(df["a"] * 86400, unit="s").dt.tz_localize("Europe/Amsterdam")}),
        id="timestamp_with_time_zone",
    ),
    pytest.param(
        lambda df: pd.DataFrame(
            {"c": pd.Categorical(np.where(df["a"] % 2 == 0, "even", "odd"), categories=["even", "odd"])}
        ),
        id="categorical",
    ),
    pytest.param(lambda df: pd.DataFrame({"b": df["a"] % 2 == 0}), id="boolean"),
    pytest.param(lambda df: df[df["a"] < 0], id="empty_returned_frame"),
]


@pytest.mark.parametrize("function", TYPE_CASES)
def test_output_of_each_type_is_what_a_scan_of_the_returned_frame_gives(duckdb_cursor, function):
    source = duckdb_cursor.sql(f"select i as a from range({ROWS}) t(i)")
    reference = duckdb_cursor.from_df(function(source.df()))
    # map() refuses a call whose frame scans to other types than the schema, so the fetch also checks the types.
    mapped = source.map(function, schema=dict(zip(reference.columns, reference.types, strict=True)))
    _assert_same_rows(mapped.fetchall(), reference.fetchall())


def test_output_read_on_several_threads_is_what_the_function_returned(duckdb_cursor):
    duckdb_cursor.execute("set threads = 4")
    duckdb_cursor.execute("create table t as select i as a from range(400000) t(i)")
    mapped = duckdb_cursor.table("t").map(lambda df: pd.DataFrame({"b": _poisoned(df["a"] * 3)}, copy=False))
    summary = "count(*), sum(b), min(b), max(b)"
    assert (
        mapped.aggregate(summary).fetchall()
        == duckdb_cursor.sql(f"select {summary} from (select a * 3 as b from t)").fetchall()
    )


def _poisoned_rows(df):
    strings = pd.Series(PREFIX + df["a"].astype(str), dtype="str")
    return pd.DataFrame({"a": df["a"], "b": _poisoned(df["a"] * 3), "s": strings}, copy=False)


def _poisoned_keys(df):
    return pd.DataFrame({"k": df["a"], "c": _poisoned(df["a"] * 5)}, copy=False)


POISONED_ROWS = f"a, a * 3 as b, '{PREFIX}' || a::varchar as s"
POISONED_KEYS = "a as k, a * 5 as c"


def _several_row_groups(con):
    """A table of several row groups, so that four threads scan it in parallel."""
    con.execute("set threads = 4")
    con.execute("create table source as select i as a from range(400000) t(i)")
    return con.table("source")


@pytest.fixture(params=["one_thread", "four_threads"])
def source(request, duckdb_cursor):
    if request.param == "one_thread":
        duckdb_cursor.execute("set threads = 1")
        return duckdb_cursor.sql(f"select i as a from range({ROWS}) t(i)")
    return _several_row_groups(duckdb_cursor)


def _arrow_rows(batches):
    return [row for batch in batches for row in zip(*(column.to_pylist() for column in batch.columns), strict=True)]


def test_limit_and_offset_over_map_releases_its_frames(source):
    inputs, outputs = [], []

    def function(df):
        inputs.append(weakref.ref(df))
        result = _poisoned_rows(df)
        outputs.append(weakref.ref(result))
        return result

    mapped = source.map(function).limit(100, offset=1000)
    _assert_same_rows(mapped.fetchall(), source.project(POISONED_ROWS).limit(100, offset=1000).fetchall())
    gc.collect()
    assert sum(ref() is not None for ref in inputs) == 0
    assert sum(ref() is not None for ref in outputs) == 0


def test_union_all_of_two_maps(source):
    mapped = source.map(_poisoned_rows).union(source.map(_poisoned_rows))
    reference = source.project(POISONED_ROWS).union(source.project(POISONED_ROWS))
    _assert_same_rows(sorted(mapped.fetchall()), sorted(reference.fetchall()))


def test_order_by_over_map(source):
    mapped = source.map(_poisoned_rows).order("b desc")
    _assert_same_rows(mapped.fetchall(), source.project(POISONED_ROWS).order("b desc").fetchall())


def test_join_with_map_on_both_sides(source):
    mapped = source.map(_poisoned_rows).set_alias("l").join(source.map(_poisoned_keys).set_alias("r"), "a = k")
    reference = source.project(POISONED_ROWS).set_alias("l").join(source.project(POISONED_KEYS).set_alias("r"), "a = k")
    _assert_same_rows(sorted(mapped.fetchall()), sorted(reference.fetchall()))


def test_aggregate_over_map(source):
    summary = "count(*), sum(b), min(b), max(b), min(s), max(s), sum(length(s))"
    mapped = source.map(_poisoned_rows).aggregate(summary)
    assert mapped.fetchall() == source.project(POISONED_ROWS).aggregate(summary).fetchall()


@needs_pyarrow
def test_arrow_table_over_map(source):
    table = source.map(_poisoned_rows).to_arrow_table()
    _assert_same_rows(_arrow_rows([table]), source.project(POISONED_ROWS).fetchall())


@needs_pyarrow
def test_partially_read_arrow_reader_releases_its_frames(source):
    inputs, outputs = [], []

    def function(df):
        inputs.append(weakref.ref(df))
        result = _poisoned_rows(df)
        outputs.append(weakref.ref(result))
        return result

    expected = source.project(POISONED_ROWS).limit(2000).fetchall()
    reader = source.map(function).to_arrow_reader(1000)
    batches = [reader.read_next_batch(), reader.read_next_batch()]
    reader.close()
    _assert_same_rows(_arrow_rows(batches), expected)
    del batches
    gc.collect()
    assert sum(ref() is not None for ref in inputs) == 0
    assert sum(ref() is not None for ref in outputs) == 0


@pytest.fixture(params=["one_thread", "four_threads"])
def running_source(request, duckdb_cursor):
    """A source whose map() query still runs after a first fetch, because the streaming buffer holds little."""
    duckdb_cursor.execute("set max_streaming_buffer_size = '32KB'")
    if request.param == "one_thread":
        duckdb_cursor.execute("set threads = 1")
        return duckdb_cursor.sql(f"select i as a from range({RUNNING_SOURCE_ROWS}) t(i)")
    return _several_row_groups(duckdb_cursor)


def test_partially_fetched_streaming_result_releases_its_frames(duckdb_cursor, running_source):
    inputs, outputs, rows_per_call = [], [], []

    def function(df):
        inputs.append(weakref.ref(df))
        rows_per_call.append(len(df))
        result = _poisoned_rows(df)
        outputs.append(weakref.ref(result))
        return result

    [(source_rows,)] = running_source.aggregate("count(*)").fetchall()
    chunks = math.ceil(source_rows / 2048)
    expected = running_source.project(POISONED_ROWS).limit(3005).fetchall()
    running_source.map(function).create_view("mapped")
    duckdb_cursor.execute("select * from mapped")
    rows = duckdb_cursor.fetchmany(5)
    rows += duckdb_cursor.fetchmany(3000)
    calls_with_data = sum(size > 0 for size in rows_per_call)
    duckdb_cursor.execute("select 42")
    gc.collect()
    _assert_same_rows(rows, expected)
    assert 0 < calls_with_data < chunks, "the query was not running when its result was abandoned"
    assert sum(ref() is not None for ref in inputs) == 0
    assert sum(ref() is not None for ref in outputs) == 0


def test_frames_of_completed_calls_are_released_while_streaming_result_is_open(duckdb_cursor):
    """Frames of calls that have returned are released while the query is still running."""
    inputs, outputs, rows_per_call = [], [], []

    def function(df):
        inputs.append(weakref.ref(df))
        rows_per_call.append(len(df))
        result = _poisoned_rows(df)
        outputs.append(weakref.ref(result))
        return result

    duckdb_cursor.execute("set threads = 1")
    duckdb_cursor.execute("set max_streaming_buffer_size = '32KB'")
    duckdb_cursor.sql(f"select i as a from range({RUNNING_SOURCE_ROWS}) t(i)").map(function).create_view("mapped")
    duckdb_cursor.execute("select * from mapped")
    rows = duckdb_cursor.fetchmany(2048)
    assert len(rows) == 2048
    gc.collect()
    calls_with_data = sum(size > 0 for size in rows_per_call)
    assert 0 < calls_with_data < RUNNING_SOURCE_ROWS // 2048, "the query was not running at the check"
    assert sum(ref() is not None for ref in inputs) == 0
    assert sum(ref() is not None for ref in outputs) == 0
    duckdb_cursor.execute("select 1")


LIFETIME_SHAPES = [
    pytest.param(lambda df: df, id="returns_its_input"),
    pytest.param(_create_in_place, id="modifies_its_input"),
    pytest.param(lambda df: df.assign(b=df["a"] * 3), id="returns_a_derived_frame"),
    pytest.param(lambda df: pd.DataFrame({"b": df["a"].to_numpy() * 3}), id="returns_an_unrelated_frame"),
]


@pytest.mark.parametrize("shape", LIFETIME_SHAPES)
def test_frames_of_every_call_are_released_once_fetched(duckdb_cursor, shape):
    inputs, outputs, rows_per_call = [], [], []

    def function(df):
        inputs.append(weakref.ref(df))
        rows_per_call.append(len(df))
        result = shape(df)
        outputs.append(weakref.ref(result))
        return result

    rows = duckdb_cursor.sql(f"select i as a from range({ROWS}) t(i)").map(function).fetchall()
    assert len(rows) == ROWS
    gc.collect()
    assert sum(size > 0 for size in rows_per_call) == math.ceil(ROWS / 2048)
    assert sum(ref() is not None for ref in inputs) == 0
    assert sum(ref() is not None for ref in outputs) == 0


def test_frames_are_released_when_the_function_raises(duckdb_cursor):
    inputs = []

    def function(df):
        inputs.append(weakref.ref(df))
        if len(df) > 0:
            msg = "the function fails on data"
            raise ValueError(msg)
        return df

    rel = duckdb_cursor.sql(f"select i as a from range({ROWS}) t(i)").map(function)
    with pytest.raises(duckdb.InvalidInputException, match="Python error"):
        rel.fetchall()
    # Printing the error stores it in sys.last_exc for post-mortem debugging, which pins the failing call's frame.
    sys.last_exc = sys.last_type = sys.last_value = sys.last_traceback = None
    gc.collect()
    assert sum(ref() is not None for ref in inputs) == 0


def test_frames_are_released_when_the_returned_frame_is_rejected(duckdb_cursor):
    inputs, outputs = [], []

    def function(df):
        inputs.append(weakref.ref(df))
        result = df.assign(a=df["a"].astype("double")) if len(df) > 0 else df
        outputs.append(weakref.ref(result))
        return result

    rel = duckdb_cursor.sql(f"select i as a from range({ROWS}) t(i)").map(function)
    with pytest.raises(duckdb.InvalidInputException, match="UDF column type mismatch"):
        rel.fetchall()
    gc.collect()
    assert sum(ref() is not None for ref in inputs) == 0
    assert sum(ref() is not None for ref in outputs) == 0


def test_memory_does_not_grow_with_the_number_of_calls(duckdb_cursor):
    calls = 30
    # Each call's input frame holds 2048 rows of four int64 columns: 64 KiB that a leak keeps per call.
    leaked_per_call = 2048 * 4 * 8
    rel = duckdb_cursor.sql("select i as a, i * 2 as b, i * 3 as c, i * 4 as d from range(2048) t(i)")

    def function(df):
        return df

    rel.map(function).fetchall()
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for _ in range(calls):
            rel.map(function).fetchall()
        gc.collect()
        grown = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert grown < calls * leaked_per_call // 4


CHILD_TIMEOUT = 30
CALLBACK_REFUSAL = (
    "code running inside a query cannot use that query's connection; use a separate connection from cursor()"
)


def _run_child(script, *args):
    """Run script in a fresh interpreter and return what it printed after "outcome "."""
    # A hang must fail the test instead of stalling the suite.
    try:
        child = subprocess.run(
            [sys.executable, "-c", script, *args], capture_output=True, text=True, timeout=CHILD_TIMEOUT, check=False
        )
    except subprocess.TimeoutExpired as hang:
        output = "".join(out.decode() if isinstance(out, bytes) else out or "" for out in (hang.stdout, hang.stderr))
        pytest.fail(f"the child did not finish within {CHILD_TIMEOUT} seconds:\n{output}")
    assert child.returncode == 0, child.stdout + child.stderr
    outcomes = [line.removeprefix("outcome ") for line in child.stdout.splitlines() if line.startswith("outcome ")]
    assert len(outcomes) == 1, child.stdout + child.stderr
    return json.loads(outcomes[0])


_ABANDON_WHILE_RUNNING = f"""
import faulthandler
import json
import sys
import weakref

import duckdb
import pandas as pd

frame, abandon = sys.argv[1:]
con = duckdb.connect(config={{"threads": 1}})
con.execute("set max_streaming_buffer_size = '32KB'")
abandoning = False
calls = []
during_abandon = []


def reenter():
    if abandoning:
        try:
            con.sql("select 42").fetchall()
        except Exception as error:
            during_abandon.append(f"{{type(error).__name__}}: {{error}}")
        else:
            during_abandon.append("returned")


def function(df):
    result = pd.DataFrame({{"b": df["a"] * 3}})
    if len(df):
        calls.append(len(df))
        weakref.finalize(df if frame == "input" else result, reenter).atexit = False
    return result


con.sql("select i as a from range({RUNNING_SOURCE_ROWS}) t(i)").map(function).create_view("mapped")
if abandon == "arrow_reader_close":
    reader = con.sql("select * from mapped").to_arrow_reader(1000)
    reader.read_next_batch()
else:
    con.execute("select * from mapped")
    con.fetchone()
calls_before_abandon = len(calls)
print("abandoning", flush=True)
faulthandler.dump_traceback_later({CHILD_TIMEOUT // 2}, exit=True)
abandoning = True
if abandon == "abort":
    con.abort()
elif abandon == "close":
    con.close()
else:
    reader.close()
abandoning = False
faulthandler.cancel_dump_traceback_later()
con.close()
print("outcome", json.dumps({{"calls": calls_before_abandon, "during_abandon": during_abandon}}), flush=True)
"""


@pytest.mark.parametrize("frame", ["input", "returned"])
@pytest.mark.parametrize("abandon", ["abort", "close", pytest.param("arrow_reader_close", marks=needs_pyarrow)])
def test_abandoning_a_running_result_runs_no_finalizer_of_a_frame(frame, abandon):
    """Abandoning a running result releases no frame, so a finalizer that uses the connection cannot hang it.

    Every frame is released inside its own call, which leaves nothing for the abandon to release.
    """
    outcome = _run_child(_ABANDON_WHILE_RUNNING, frame, abandon)
    assert 0 < outcome["calls"] < RUNNING_SOURCE_ROWS // 2048, "the query was not running at the abandon"
    assert outcome["during_abandon"] == []


READ_FULLY_ROWS = 10_000

_READ_FULLY = f"""
import json
import sys
import weakref

import duckdb
import pandas as pd

frame = sys.argv[1]
con = duckdb.connect(config={{"threads": 1}})
calls = []
outcomes = []


def reenter():
    try:
        con.sql("select 42").fetchall()
    except Exception as error:
        outcomes.append([type(error).__name__, str(error)])
    else:
        outcomes.append(["returned", ""])


def function(df):
    result = pd.DataFrame({{"b": df["a"] * 3}})
    calls.append(len(df))
    weakref.finalize(df if frame == "input" else result, reenter).atexit = False
    return result


rows = con.sql("select i as a from range({READ_FULLY_ROWS}) t(i)").map(function).fetchall()
finalized = list(outcomes)
con.close()
print("outcome", json.dumps({{"rows": rows, "calls": calls, "finalized": finalized}}), flush=True)
"""


@pytest.mark.parametrize("frame", ["input", "returned"])
def test_a_finalizer_of_a_frame_that_uses_the_connection_is_refused(frame):
    """Each frame is released inside its call, where the connection refuses the finalizer; the query completes."""
    outcome = _run_child(_READ_FULLY, frame)
    assert outcome["rows"] == [[i * 3] for i in range(READ_FULLY_ROWS)]
    assert len(outcome["finalized"]) == len(outcome["calls"])
    for exception, message in outcome["finalized"]:
        assert exception == "ResourceInUseException"
        assert CALLBACK_REFUSAL in message


_BINDING_CODE_RAISES = f"""
import json
import sys
import weakref

import duckdb
import pandas as pd

shape, stage, frame = sys.argv[1:]
con = duckdb.connect(config={{"threads": 1}})
calls = []
outcomes = []


def reenter():
    try:
        con.sql("select 42").fetchall()
    except Exception as error:
        outcomes.append([type(error).__name__, str(error)])
    else:
        outcomes.append(["returned", ""])


class Unprintable:
    def __init__(self, input_frame):
        self.input_frame = input_frame
        self.returned_frame = None

    def __str__(self):
        held = (self.input_frame, self.returned_frame())
        raise ValueError("bad string")


class RaisingDtypes(pd.DataFrame):
    _metadata = ["input_frame"]

    @property
    def dtypes(self):
        raise ValueError("bad dtypes")


def function(df):
    calls.append(len(df))
    if shape == "unprintable_value":
        value = Unprintable(df)
        result = pd.DataFrame({{"s": pd.Series([value] * len(df), dtype=object)}})
        value.returned_frame = weakref.ref(result)
    elif stage == "bind" or len(df):
        result = RaisingDtypes({{"a": df["a"]}})
        result.input_frame = df
    else:
        result = pd.DataFrame({{"a": df["a"]}})
    weakref.finalize(df if frame == "input" else result, reenter).atexit = False
    return result


schema = {{"s": "VARCHAR"}} if shape == "unprintable_value" else None
raised_at = "map"
error_type = message = None
try:
    rel = con.sql("select i as a from range({ROWS}) t(i)").map(function, schema=schema)
    raised_at = "fetchall"
    rel.fetchall()
    raised_at = None
except Exception as error:
    error_type = f"{{type(error).__module__}}.{{type(error).__qualname__}}"
    message = str(error)
finalized = list(outcomes)
con.close()
outcome = {{"raised_at": raised_at, "error_type": error_type, "message": message}}
print("outcome", json.dumps({{**outcome, "calls": calls, "finalized": finalized}}), flush=True)
"""

BINDING_CODE_ERRORS = [
    pytest.param("unprintable_value", "execution", id="unprintable_value_at_execution"),
    pytest.param("raising_dtypes", "execution", id="raising_dtypes_at_execution"),
    pytest.param("raising_dtypes", "bind", id="raising_dtypes_at_bind"),
]


@pytest.mark.parametrize(("shape", "stage"), BINDING_CODE_ERRORS)
def test_a_python_error_from_the_binding_code_keeps_its_type_and_message(shape, stage):
    """A Python error raised by code map() calls around the function surfaces as a duckdb.Error with its traceback."""
    raising_function, text = ("__str__", "bad string") if shape == "unprintable_value" else ("dtypes", "bad dtypes")
    outcome = _run_child(_BINDING_CODE_RAISES, shape, stage, "input")
    assert outcome["raised_at"] == ("map" if stage == "bind" else "fetchall")
    assert outcome["error_type"] == "_duckdb.Error"
    # The layout between the header, the raising frame and the final line is nanobind's: the full API prints bare
    # frame lines, the limited API uses traceback.format_exception, with blank lines, source lines and markers.
    traceback = (
        r"Traceback \(most recent call last\):\n(.*\n)*?"
        rf'  File "<string>", line \d+, in {raising_function}\n(.*\n)*?ValueError: {text}\n?'
    )
    assert re.fullmatch(traceback, outcome["message"]), outcome["message"]


@pytest.mark.parametrize("frame", ["input", "returned"])
@pytest.mark.parametrize(("shape", "stage"), BINDING_CODE_ERRORS)
def test_a_python_error_from_the_binding_code_releases_the_frames_inside_the_call(shape, stage, frame):
    """The error's traceback holds the call's frames, so the error is released inside the call, where it was raised."""
    outcome = _run_child(_BINDING_CODE_RAISES, shape, stage, frame)
    assert outcome["raised_at"] is not None
    assert len(outcome["finalized"]) == len(outcome["calls"])
    for exception, message in outcome["finalized"]:
        assert exception == "ResourceInUseException"
        assert CALLBACK_REFUSAL in message


def test_map_runs_on_multiple_threads(duckdb_cursor):
    """map() over the four thread source runs on more than one thread."""
    source = _several_row_groups(duckdb_cursor)
    thread_ids = []

    def function(df):
        # Binding calls the function with an empty frame on the thread that builds the relation.
        if len(df):
            thread_ids.append(threading.get_ident())
        return pd.DataFrame({"b": df["a"] * 3})

    summary = "count(*), sum(b), min(b), max(b)"
    mapped = source.map(function).aggregate(summary)
    assert mapped.fetchall() == source.project("a * 3 as b").aggregate(summary).fetchall()
    assert len(set(thread_ids)) > 1, f"{len(thread_ids)} calls ran on one thread"

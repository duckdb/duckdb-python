import ctypes
import datetime
import gc
import importlib.metadata
import importlib.util
import itertools
import json
import re
import statistics
import subprocess
import sys
import textwrap
import tracemalloc
import warnings

import pytest
from conftest import PANDAS_GE_3
from packaging.version import Version

import duckdb
import numpy
import pandas as pd


class TestPandasString:
    def test_pandas_string(self, duckdb_cursor):
        strings = numpy.array(["foo", "bar", "baz"])

        # https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html
        df_in = pd.DataFrame(
            {
                "object": pd.Series(strings, dtype="object"),
            }
        )
        # Only available in pandas 1.0.0
        if hasattr(pd, "StringDtype"):
            df_in["string"] = pd.Series(strings, dtype=pd.StringDtype())

        df_out = duckdb.query_df(df_in, "data", "SELECT * FROM data").df()

        assert numpy.all(df_out["object"] == strings)
        if hasattr(pd, "StringDtype"):
            assert numpy.all(df_out["string"] == strings)

    def test_bug_2467(self, duckdb_cursor):
        N = 1_000_000
        # Create DataFrame with string attribute
        df = pd.DataFrame({"city": ["Amsterdam", "New York", "London"] * N})
        # Copy Dataframe to DuckDB
        con = duckdb.connect()
        con.register("df", df)
        con.execute(
            """
            CREATE TABLE t1 AS SELECT * FROM df
        """
        )
        assert con.execute(
            """
            SELECT count(*) from t1
        """
        ).fetchall() == [(3000000,)]


# A pyarrow backed str column averaging at least this many rows per chunk is read from its Arrow chunks, and one
# below it is converted with to_numpy; must equal the threshold the scan uses.
ARROW_ROWS_PER_CHUNK = 50
VECTOR_SIZE = 2048

needs_pyarrow = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow is not installed")
needs_pandas_3 = pytest.mark.skipif(not PANDAS_GE_3, reason="before pandas 3 the str dtype is object")


def _pyarrow_version():
    try:
        return Version(importlib.metadata.version("pyarrow"))
    except importlib.metadata.PackageNotFoundError:
        return None


needs_arrow_chunk_export = pytest.mark.skipif(
    _pyarrow_version() is not None and _pyarrow_version() < Version("14"),
    reason="before pyarrow 14 chunks cannot be exported, so a str column is converted with to_numpy",
)

PYARROW_STORAGE = [
    pytest.param("str", id="default_str", marks=[needs_pyarrow, needs_pandas_3]),
    pytest.param("string[pyarrow]", id="string_pyarrow", marks=needs_pyarrow),
]
PYTHON_STORAGE = [
    pytest.param("string[python]", id="string_python"),
    pytest.param("python_nan", id="python_nan", marks=needs_pandas_3),
]

SAMPLE_VALUES = ["a", None, numpy.nan, "nan", "", "c", "é", "日本語テキスト", "a\x00b", "x" * 40, "🦆"]
# Placed at the edges of the first two vectors and between them.
LAYOUT_VALUES = {
    0: None,
    7: "",
    2046: "nan",
    2047: None,
    2048: "é",
    2049: "a\x00b",
    3000: "日本語テキスト",
    4000: "x" * 40,
    4095: "🦆",
}


@pytest.fixture
def pa():
    return pytest.importorskip("pyarrow")


def _dtype(name):
    if name == "python_nan":
        return pd.StringDtype("python", na_value=numpy.nan)
    return pd.api.types.pandas_dtype(name)


def _missing(value):
    return value is None or value is pd.NA or (isinstance(value, float) and numpy.isnan(value))


def _rows_of(values):
    return [(None if _missing(value) else value,) for value in values]


def _layout_values(count):
    values = [f"v{index}" for index in range(count)]
    for index, value in LAYOUT_VALUES.items():
        if index < count:
            values[index] = value
    values[-1] = None
    return values


def _pieces(values, rows_per_chunk):
    return [values[start : start + rows_per_chunk] for start in range(0, len(values), rows_per_chunk)]


def _split(values, sizes):
    starts = numpy.cumsum([0, *sizes])
    return [values[start:end] for start, end in itertools.pairwise(starts)]


def _chunk_lengths(array):
    return [len(chunk) for chunk in array._pa_array.chunks]


def _chunked(pieces, dtype):
    series = pd.concat([pd.Series(piece, dtype=_dtype(dtype)) for piece in pieces], ignore_index=True)
    assert _chunk_lengths(series.array) == [len(piece) for piece in pieces]
    return series


def _frame(series):
    # Passing the array rather than the Series keeps its chunks: reindexing a Series drops empty chunks.
    frame = pd.DataFrame({"s": series.array, "i": numpy.arange(len(series), dtype=numpy.int64)})
    if isinstance(series.array, pd.arrays.ArrowStringArray):
        assert _chunk_lengths(frame["s"].array) == _chunk_lengths(series.array)
    return frame


def _mismatches(rows, expected):
    if len(rows) != len(expected):
        return f"{len(rows)} rows, expected {len(expected)}"
    wrong = [index for index, (row, want) in enumerate(zip(rows, expected, strict=False)) if row != want]
    if not wrong:
        return ""
    first = wrong[0]
    return f"{len(wrong)} rows differ, first at row {first}: {rows[first]!r} instead of {expected[first]!r}"


def _types(relation):
    return [str(column_type) for column_type in relation.types]


def _spy(monkeypatch, array_class, method, rows, calls, *, record=len, refuse=False):
    original = getattr(array_class, method)

    def spy(self, *args, **kwargs):
        # Column labels are a str Index too, so only arrays of the column's length count.
        if len(self) == rows:
            calls.append(record(self))
            if refuse:
                msg = f"values converted to Python objects through {method}"
                raise RuntimeError(msg)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(array_class, method, spy)


def _to_numpy_calls(monkeypatch, array_class, rows, *, record=len):
    calls = []
    _spy(monkeypatch, array_class, "to_numpy", rows, calls, record=record)
    return calls


def _refuse_conversions(monkeypatch, array_class, rows):
    for method in ("to_numpy", "isna", "__iter__"):
        _spy(monkeypatch, array_class, method, rows, [], refuse=True)


@pytest.mark.parametrize("repeats", [1, ARROW_ROWS_PER_CHUNK], ids=["short", "long"])
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_pyarrow_string_column_reads_values_and_nulls(duckdb_cursor, dtype, repeats):
    values = SAMPLE_VALUES * repeats
    frame = _frame(pd.Series(values, dtype=_dtype(dtype)))  # noqa: F841
    relation = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i")
    types = _types(relation)
    rows = relation.fetchall()
    assert types == ["VARCHAR"]
    assert rows == _rows_of(values)


@pytest.mark.parametrize("dtype", PYTHON_STORAGE)
def test_python_string_column_reads_values_and_nulls(duckdb_cursor, dtype):
    frame = _frame(pd.Series(SAMPLE_VALUES, dtype=_dtype(dtype)))  # noqa: F841
    relation = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i")
    types = _types(relation)
    rows = relation.fetchall()
    assert types == ["VARCHAR"]
    assert rows == _rows_of(SAMPLE_VALUES)


@pytest.mark.parametrize("dtype", ["string", pytest.param("str", marks=needs_pandas_3)])
def test_python_string_column_storage_option(duckdb_cursor, dtype):
    with pd.option_context("mode.string_storage", "python"):
        series = pd.Series(SAMPLE_VALUES, dtype=dtype)
    frame = _frame(series)  # noqa: F841
    rows = duckdb_cursor.sql("SELECT s, s IS NULL FROM frame ORDER BY i").fetchall()
    assert type(series.array) is pd.arrays.StringArray
    assert rows == [(value, value is None) for (value,) in _rows_of(SAMPLE_VALUES)]


@pytest.mark.parametrize("dtype", PYTHON_STORAGE)
def test_python_string_column_strided(duckdb_cursor, dtype):
    values = [None if index % 4 == 0 else f"v{index}" for index in range(9000)]
    frame = _frame(pd.Series(values, dtype=_dtype(dtype))).iloc[::3]
    stored = frame["s"].array._ndarray
    assert stored.strides[0] == 3 * stored.itemsize
    rows = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i").fetchall()
    assert _mismatches(rows, _rows_of(values[::3])) == ""


@needs_pandas_3
def test_python_string_column_without_pyarrow():
    script = textwrap.dedent(
        """
        import json
        import sys

        attempts = []

        class BlockPyarrow:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "pyarrow":
                    attempts.append(name)
                    raise ImportError(name)

        sys.meta_path.insert(0, BlockPyarrow())
        import numpy
        import pandas as pd
        import duckdb

        frame = pd.DataFrame({"s": pd.Series(["a", None, numpy.nan, "c"]), "f": [1.0, numpy.nan, 2.0, 3.0]})
        attempts.clear()
        relation = duckdb.connect().sql("SELECT s, s IS NULL, f FROM frame")
        print(json.dumps({
            "array": type(frame["s"].array).__name__,
            "types": [str(t) for t in relation.types],
            "rows": relation.fetchall(),
            "attempts": attempts,
            "loaded": "pyarrow" in sys.modules,
        }))
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(proc.stdout.splitlines()[-1])
    assert report == {
        "array": "StringArray",
        "types": ["VARCHAR", "BOOLEAN", "DOUBLE"],
        "rows": [["a", False, 1.0], [None, True, None], [None, True, 2.0], ["c", False, 3.0]],
        "attempts": [],
        "loaded": False,
    }


BIND_STORAGE = [
    pytest.param("str", id="default_str", marks=[needs_pyarrow, needs_pandas_3, needs_arrow_chunk_export]),
    pytest.param("string[pyarrow]", id="string_pyarrow", marks=[needs_pyarrow, needs_arrow_chunk_export]),
    pytest.param("string[python]", id="string_python"),
    pytest.param("python_nan", id="python_nan", marks=needs_pandas_3),
]


@pytest.mark.parametrize("dtype", BIND_STORAGE)
@pytest.mark.parametrize("column", ["s", "i"])
def test_string_column_converts_no_values_at_bind(duckdb_cursor, monkeypatch, dtype, column):
    rows = ARROW_ROWS_PER_CHUNK
    values = [None if index % 3 == 1 else f"v{index}" for index in range(rows)]
    series = pd.Series(values, dtype=_dtype(dtype))
    frame = _frame(series)  # noqa: F841
    expected = _rows_of(values) if column == "s" else [(index,) for index in range(rows)]
    _refuse_conversions(monkeypatch, type(series.array), rows)
    result = duckdb_cursor.sql(f"SELECT {column} FROM frame ORDER BY i").fetchall()
    assert result == expected


BIND_HEAP_ROWS = 200_000


@pytest.mark.parametrize("dtype", BIND_STORAGE)
def test_string_column_bind_allocates_nothing_per_row(duckdb_cursor, dtype):
    # Catches a conversion per value that no spy on the array class sees, such as one inside pyarrow.
    values = [None if index % 3 == 1 else f"v{index}" for index in range(BIND_HEAP_ROWS)]
    frame = _frame(pd.Series(values, dtype=_dtype(dtype)))  # noqa: F841
    query = "SELECT i FROM frame LIMIT 0"
    duckdb_cursor.sql(query).fetchall()
    tracemalloc.start()
    try:
        duckdb_cursor.sql(query).fetchall()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < BIND_HEAP_ROWS // 4


@pytest.mark.parametrize("rows", [3, ARROW_ROWS_PER_CHUNK], ids=["short", "long"])
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_pyarrow_string_column_stays_on_pandas_scan(duckdb_cursor, dtype, rows):
    frame = _frame(pd.Series(["a", None, "c"] * rows, dtype=_dtype(dtype)))  # noqa: F841
    plan = duckdb_cursor.sql("EXPLAIN SELECT s FROM frame").fetchall()[0][1]
    assert "PANDAS_SCAN" in plan


STR_LAYOUTS = [
    pytest.param(lambda values: _pieces(values, 1), id="one_row_per_chunk"),
    pytest.param(lambda values: _pieces(values, ARROW_ROWS_PER_CHUNK - 1), id="just_below_the_threshold"),
    pytest.param(lambda values: _pieces(values, ARROW_ROWS_PER_CHUNK), id="at_the_threshold"),
    pytest.param(lambda values: _pieces(values, VECTOR_SIZE // 2), id="two_chunks_per_vector"),
    pytest.param(lambda values: _pieces(values, VECTOR_SIZE), id="chunks_of_one_vector"),
    pytest.param(lambda values: _pieces(values, VECTOR_SIZE + 1), id="chunks_longer_than_a_vector"),
    pytest.param(lambda values: [values], id="one_chunk"),
    pytest.param(lambda values: [values[:-5], *_pieces(values[-5:], 1)], id="long_chunk_with_appended_rows"),
]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("layout", STR_LAYOUTS)
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_str_chunk_layouts_read_same_values(duckdb_cursor, dtype, layout):
    values = _layout_values(2 * VECTOR_SIZE + 4)
    frame = _frame(_chunked(layout(values), dtype))  # noqa: F841
    rows = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i").fetchall()
    assert _mismatches(rows, _rows_of(values)) == ""


SLICED_LAYOUTS = [
    pytest.param(lambda values: _split(values, [3000, 5000]), slice(None), id="chunks_crossing_vectors"),
    pytest.param(lambda values: [values[:6000]], slice(3, None), id="offset_not_byte_aligned"),
    pytest.param(lambda values: [values[:6000]], slice(None, None, 3), id="every_third_row"),
    pytest.param(lambda values: _split(values, [3000, 3000]), slice(2500, None), id="offset_inside_the_first_chunk"),
    pytest.param(lambda values: _split(values, [3000, 3000]), slice(1, None, 2), id="every_other_row_of_two_chunks"),
    pytest.param(lambda values: [[], values[:2100], [], ["tail", None]], slice(None), id="empty_chunks_between_rows"),
    pytest.param(lambda values: [[], values[:2050], []], slice(None), id="empty_chunks_at_both_ends"),
]


@pytest.mark.timeout(60)
@pytest.mark.parametrize(("layout", "rows_slice"), SLICED_LAYOUTS)
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_pyarrow_string_column_chunk_layouts(duckdb_cursor, dtype, layout, rows_slice):
    series = _chunked(layout(_layout_values(8000)), dtype)
    if rows_slice != slice(None):
        series = series.iloc[rows_slice]
    expected = _rows_of(series.tolist())
    frame = _frame(series)  # noqa: F841
    rows = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i").fetchall()
    assert _mismatches(rows, expected) == ""


@needs_pyarrow
@pytest.mark.timeout(60)
@pytest.mark.parametrize("threads", [1, 4])
def test_pyarrow_string_column_across_partitions(duckdb_cursor, threads):
    # Four chunks whose boundaries do not line up with the 102400 row partitions of the pandas scan.
    values = [None if index % 1000 == 0 else f"r{index}" for index in range(4 * 77777)]
    frame = _frame(_chunked(_pieces(values, 77777), "string[pyarrow]"))  # noqa: F841
    duckdb_cursor.execute(f"SET threads = {threads}")
    rows = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i").fetchall()
    assert _mismatches(rows, _rows_of(values)) == ""


@needs_pyarrow
@pytest.mark.timeout(60)
@pytest.mark.parametrize("threads", [1, 4])
def test_pyarrow_string_column_many_small_chunks(duckdb_cursor, threads):
    # More rows than two 102400 row partitions of the pandas scan, so several threads share the chunks.
    count = 2100 * ARROW_ROWS_PER_CHUNK
    middle = ARROW_ROWS_PER_CHUNK // 2
    values = [None if index % ARROW_ROWS_PER_CHUNK == middle else f"v{index}" for index in range(count)]
    frame = _frame(_chunked(_pieces(values, ARROW_ROWS_PER_CHUNK), "string[pyarrow]"))  # noqa: F841
    duckdb_cursor.execute(f"SET threads = {threads}")
    rows = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i").fetchall()
    assert _mismatches(rows, _rows_of(values)) == ""


def _filtered_to_nothing(pa, dtype):
    series = pd.Series(["a", None], dtype=dtype)
    return series[series == "zzz"]


ZERO_CHUNKS = [
    pytest.param(_filtered_to_nothing, id="filtered"),
    pytest.param(lambda pa, dtype: pd.Series([None, None], dtype=dtype).dropna(), id="all_missing_dropped"),
    pytest.param(
        lambda pa, dtype: pd.Series(pd.array(pa.chunked_array([], type=pa.large_string()), dtype=dtype)),
        id="empty_chunked_array",
    ),
]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("with_int_column", [False, True], ids=["alone", "next_to_int"])
@pytest.mark.parametrize("make", ZERO_CHUNKS)
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_pyarrow_string_column_zero_chunks(duckdb_cursor, pa, dtype, make, with_int_column):
    frame = pd.DataFrame({"s": make(pa, _dtype(dtype)).array})
    if with_int_column:
        frame["i"] = numpy.array([], dtype=numpy.int64)
    assert frame["s"].array._pa_array.num_chunks == 0
    relation = duckdb_cursor.sql("SELECT * FROM frame")
    types = _types(relation)
    rows = relation.fetchall()
    assert types == (["VARCHAR", "BIGINT"] if with_int_column else ["VARCHAR"])
    assert rows == []


@pytest.mark.parametrize("rows", [3, ARROW_ROWS_PER_CHUNK], ids=["short", "long"])
@pytest.mark.parametrize("dtype", PYARROW_STORAGE + PYTHON_STORAGE)
def test_all_missing_string_column(duckdb_cursor, dtype, rows):
    frame = _frame(pd.Series([None] * rows, dtype=_dtype(dtype)))  # noqa: F841
    relation = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i")
    types = _types(relation)
    result = relation.fetchall()
    assert types == ["VARCHAR"]
    assert result == [(None,)] * rows


@pytest.mark.timeout(60)
@pytest.mark.parametrize("dtype", PYARROW_STORAGE + PYTHON_STORAGE)
def test_empty_string_column(duckdb_cursor, dtype):
    frame = pd.DataFrame({"s": pd.Series([], dtype=_dtype(dtype)), "i": numpy.array([], dtype=numpy.int64)})  # noqa: F841
    relation = duckdb_cursor.sql("SELECT * FROM frame")
    types = _types(relation)
    rows = relation.fetchall()
    assert types == ["VARCHAR", "BIGINT"]
    assert rows == []


@needs_pyarrow
@needs_pandas_3
@pytest.mark.parametrize("repeats", [1, ARROW_ROWS_PER_CHUNK], ids=["short", "long"])
def test_string_column_leaves_other_columns_alone(duckdb_cursor, repeats):
    stamps = [datetime.datetime(2024, 1, 2, 3, 4, 5, 123456), None, datetime.datetime(2024, 1, 3)]
    frame = pd.DataFrame(
        {
            "s": pd.Series(["a", None, "c"] * repeats, dtype="str"),
            "f": numpy.array([1.5, numpy.nan, 3.0] * repeats, dtype=numpy.float64),
            "o": pd.Series([1, None, 3] * repeats, dtype=object),
            "c": pd.Categorical(["x", "y", None] * repeats),
            "t": pd.Series(stamps * repeats, dtype="datetime64[ns]"),
        }
    )
    frame.index = pd.Index([f"label{index}" for index in range(3 * repeats)], name="label")
    expected = [
        ("a", 1.5, 1, "x", stamps[0]),
        (None, None, None, "y", None),
        ("c", 3.0, 3, None, stamps[2]),
    ] * repeats
    relation = duckdb_cursor.sql("SELECT * FROM frame")
    columns = relation.columns
    types = _types(relation)
    rows = relation.fetchall()
    assert columns == ["s", "f", "o", "c", "t"]
    assert types == ["VARCHAR", "DOUBLE", "INTEGER", "ENUM('x', 'y')", "TIMESTAMP_NS"]
    assert rows == expected


@pytest.mark.parametrize("repeats", [1, ARROW_ROWS_PER_CHUNK], ids=["short", "long"])
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_string_columns_with_duplicate_labels(duckdb_cursor, dtype, repeats):
    frame = pd.DataFrame([["a", "b"]] * repeats, columns=["s", "s"]).astype(_dtype(dtype))  # noqa: F841
    relation = duckdb_cursor.sql("SELECT * FROM frame")
    columns = relation.columns
    rows = relation.fetchall()
    assert columns == ["s", "s_1"]
    assert rows == [("a", "b")] * repeats


def test_frame_with_an_arrow_dtype_column_is_read_whole_as_arrow(duckdb_cursor, pa):
    frame = pd.DataFrame(  # noqa: F841
        {
            "s": pd.Series(["a", None] * ARROW_ROWS_PER_CHUNK, dtype="string[pyarrow]"),
            "n": pd.array([1, None] * ARROW_ROWS_PER_CHUNK, dtype=pd.ArrowDtype(pa.int32())),
        }
    )
    plan = duckdb_cursor.sql("EXPLAIN SELECT * FROM frame").fetchall()[0][1]
    relation = duckdb_cursor.sql("SELECT * FROM frame")
    types = _types(relation)
    rows = relation.fetchall()
    assert "PYTHON_ARROW_SCAN" in plan
    assert types == ["VARCHAR", "INTEGER"]
    assert rows == [("a", 1), (None, None)] * ARROW_ROWS_PER_CHUNK


@needs_pyarrow
@needs_pandas_3
def test_string_column_of_a_result_reads_back(duckdb_cursor):
    query = "SELECT CASE WHEN range % 5 = 0 THEN NULL ELSE 'v' || range::VARCHAR END AS s, range AS i FROM range(5000)"
    expected = duckdb_cursor.sql(f"SELECT s FROM ({query}) ORDER BY i").fetchall()
    frame = duckdb_cursor.sql(query).df()
    rows = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i").fetchall()
    assert type(frame["s"].array) is pd.arrays.ArrowStringArray
    assert rows == expected


@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_registered_frame_keeps_its_string_column(duckdb_cursor, dtype):
    values = ["a", "b"] * ARROW_ROWS_PER_CHUNK
    frame = pd.DataFrame({"s": pd.Series(values, dtype=_dtype(dtype))})
    duckdb_cursor.register("registered", frame)
    before = duckdb_cursor.sql("SELECT s FROM registered").fetchall()
    frame["s"] = pd.Series(["z"] * len(values), dtype=_dtype(dtype))
    after_assignment = duckdb_cursor.sql("SELECT s FROM registered").fetchall()
    frame.loc[0, "s"] = "y"
    after_update = duckdb_cursor.sql("SELECT s FROM registered").fetchall()
    assert before == _rows_of(values)
    assert after_assignment == _rows_of(values)
    assert after_update == _rows_of(values)


def _zero_chunk_series(dtype):
    return pd.Series([None], dtype=_dtype(dtype)).dropna()


SHORT_CHUNKS = [
    pytest.param(lambda dtype: _chunked([["v"]] * 5, dtype), id="one_row_per_chunk"),
    pytest.param(
        lambda dtype: _chunked([["v"] * (ARROW_ROWS_PER_CHUNK - 1)], dtype), id="one_chunk_just_below_the_threshold"
    ),
    pytest.param(
        lambda dtype: _chunked([["v"] * ARROW_ROWS_PER_CHUNK, ["v"], ["v"], ["v"]], dtype),
        id="average_below_the_threshold",
    ),
    pytest.param(
        lambda dtype: _chunked([["v"] * ARROW_ROWS_PER_CHUNK, [], ["v"] * ARROW_ROWS_PER_CHUNK], dtype),
        id="empty_chunk_lowers_the_average",
    ),
]
LONG_CHUNKS = [
    pytest.param(lambda dtype: _chunked([["v"] * ARROW_ROWS_PER_CHUNK], dtype), id="one_chunk_at_the_threshold"),
    pytest.param(lambda dtype: _chunked([["v"] * ARROW_ROWS_PER_CHUNK] * 3, dtype), id="chunks_at_the_threshold"),
    pytest.param(lambda dtype: _chunked([["v"] * (VECTOR_SIZE + 1)] * 2, dtype), id="chunks_longer_than_a_vector"),
    pytest.param(
        lambda dtype: _chunked([["v"] * (4 * ARROW_ROWS_PER_CHUNK), ["v"], ["v"], ["v"]], dtype),
        id="long_chunk_with_appended_rows",
    ),
    pytest.param(_zero_chunk_series, id="zero_chunks"),
]


@pytest.mark.parametrize("layout", SHORT_CHUNKS)
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_str_short_chunks_read_through_to_numpy(duckdb_cursor, monkeypatch, dtype, layout):
    series = layout(dtype)
    frame = _frame(series)  # noqa: F841
    calls = _to_numpy_calls(monkeypatch, pd.arrays.ArrowStringArray, len(series))
    duckdb_cursor.sql("SELECT s FROM frame").fetchall()
    assert calls


@needs_arrow_chunk_export
@pytest.mark.parametrize("layout", LONG_CHUNKS)
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_str_long_chunks_read_as_arrow(duckdb_cursor, monkeypatch, dtype, layout):
    series = layout(dtype)
    frame = _frame(series)  # noqa: F841
    calls = _to_numpy_calls(monkeypatch, pd.arrays.ArrowStringArray, len(series))
    duckdb_cursor.sql("SELECT s FROM frame").fetchall()
    assert calls == []


@needs_arrow_chunk_export
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_str_chunk_length_rule_applies_per_column(duckdb_cursor, monkeypatch, dtype):
    rows = 3 * ARROW_ROWS_PER_CHUNK
    long_values = [f"l{index}" for index in range(rows)]
    short_values = [None if index % 7 == 0 else f"s{index}" for index in range(rows)]
    long_column = _chunked([long_values], dtype).array
    short_column = _chunked(_pieces(short_values, 2), dtype).array
    frame = pd.DataFrame({"long": long_column, "short": short_column, "i": numpy.arange(rows, dtype=numpy.int64)})
    assert _chunk_lengths(frame["short"].array) == [2] * (rows // 2)
    calls = _to_numpy_calls(
        monkeypatch, pd.arrays.ArrowStringArray, rows, record=lambda array: array._pa_array.num_chunks
    )
    result = duckdb_cursor.sql("SELECT long, short FROM frame ORDER BY i").fetchall()
    assert result == list(zip(long_values, short_values, strict=False))
    assert set(calls) == {rows // 2}


@needs_arrow_chunk_export
@pytest.mark.parametrize(
    ("pieces", "converted"),
    [
        pytest.param([ARROW_ROWS_PER_CHUNK, 0], True, id="with_an_empty_chunk"),
        pytest.param([ARROW_ROWS_PER_CHUNK], False, id="without_one"),
    ],
)
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_str_empty_chunk_of_a_map_result_lowers_the_average(duckdb_cursor, monkeypatch, dtype, pieces, converted):
    # On pandas 3 a registered frame is read through a shallow copy, which drops empty chunks at either end.
    values = [f"v{index}" for index in range(ARROW_ROWS_PER_CHUNK)]

    def produce(_):
        return pd.DataFrame({"s": _chunked(_split(values, pieces), dtype).array})

    calls = _to_numpy_calls(monkeypatch, pd.arrays.ArrowStringArray, len(values))
    rows = duckdb_cursor.sql("SELECT 1 AS x").map(produce).fetchall()
    assert rows == _rows_of(values)
    assert bool(calls) == converted


@needs_arrow_chunk_export
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_map_result_str_column_with_zero_chunks_reads_no_rows(duckdb_cursor, pa, dtype):
    def produce(_):
        result = pd.DataFrame({"s": _filtered_to_nothing(pa, _dtype(dtype)).array})
        assert result["s"].array._pa_array.num_chunks == 0
        return result

    relation = duckdb_cursor.sql("SELECT 1 AS x").map(produce)
    types = _types(relation)
    rows = relation.fetchall()
    assert types == ["VARCHAR"]
    assert rows == []


@needs_arrow_chunk_export
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_str_chunks_are_released_once_the_frame_is_gone(pa, dtype):
    # Collected first, so that only the arrays this test makes are counted.
    gc.collect()
    before = pa.total_allocated_bytes()
    values = [f"value {index}" for index in range(3 * ARROW_ROWS_PER_CHUNK)]
    frame = _frame(_chunked(_pieces(values, ARROW_ROWS_PER_CHUNK), dtype))
    with duckdb.connect() as con:
        con.register("frame", frame)
        rows = con.sql("SELECT s FROM frame ORDER BY i").fetchall()
    del frame
    gc.collect()
    assert rows == _rows_of(values)
    assert pa.total_allocated_bytes() == before


@needs_arrow_chunk_export
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_str_chunks_outlive_the_frames_column_while_a_result_streams(pa, dtype):
    gc.collect()
    before = pa.total_allocated_bytes()
    values = [f"value {index}" for index in range(4 * VECTOR_SIZE)]
    frame = _frame(_chunked(_pieces(values, 4 * ARROW_ROWS_PER_CHUNK), dtype))
    gc.collect()
    column_bytes = pa.total_allocated_bytes() - before
    with duckdb.connect(config={"threads": 1}) as con:
        # A small buffer stops the scan from reading ahead, so part of the column is read after it is replaced.
        con.execute("SET streaming_buffer_size = '1KB'")
        con.execute("SET enable_progress_bar = true")
        con.execute("SET enable_progress_bar_print = false")
        con.register("view", frame)
        con.execute("SELECT s FROM view")
        first = con.fetchmany(3)
        assert 0 <= con.query_progress() < 100
        frame["s"] = 0
        gc.collect()
        assert pa.total_allocated_bytes() - before >= column_bytes
        rows = first + con.fetchall()
    del frame
    gc.collect()
    assert rows == _rows_of(values)
    assert pa.total_allocated_bytes() == before


@pytest.mark.parametrize("rows", [3, 3 * ARROW_ROWS_PER_CHUNK], ids=["short", "long"])
@pytest.mark.parametrize("dtype", PYARROW_STORAGE)
def test_pyarrow_string_column_is_never_read_through_data(duckdb_cursor, monkeypatch, dtype, rows):
    # pandas 2 gives the array a deprecated _data attribute, which pandas 3 lacks; adding one reproduces pandas 2.
    reads = []

    def deprecated_data(self):
        reads.append(len(self))
        warnings.warn("ArrowStringArray._data is deprecated", FutureWarning, stacklevel=2)
        return self._pa_array

    values = ["a", None, "c"] * (rows // 3)
    frame = _frame(pd.Series(values, dtype=_dtype(dtype)))  # noqa: F841
    monkeypatch.setattr(pd.arrays.ArrowStringArray, "_data", property(deprecated_data), raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i").fetchall()
    assert result == _rows_of(values)
    assert reads == []
    assert [warning for warning in caught if issubclass(warning.category, FutureWarning)] == []


@pytest.mark.parametrize("rows", [3, 3 * ARROW_ROWS_PER_CHUNK], ids=["short", "long"])
def test_pyarrow_string_array_subclass_is_never_read_through_data(duckdb_cursor, pa, rows):
    # pandas 2 backs string[pyarrow_numpy] and infer_string with a subclass of ArrowStringArray.
    reads = []

    class SubclassWithData(pd.arrays.ArrowStringArray):
        @property
        def _data(self):
            reads.append(len(self))
            warnings.warn("ArrowStringArray._data is deprecated", FutureWarning, stacklevel=2)
            return self._pa_array

    values = ["a", None, "c"] * (rows // 3)
    array = SubclassWithData(pa.chunked_array([pa.array(values, type=pa.large_string())]))
    frame = pd.DataFrame({"s": array, "i": numpy.arange(rows, dtype=numpy.int64)})
    assert type(frame["s"].array) is SubclassWithData
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i").fetchall()
    assert result == _rows_of(values)
    assert reads == []
    assert [warning for warning in caught if issubclass(warning.category, FutureWarning)] == []


@pytest.mark.parametrize("dtype", PYTHON_STORAGE)
def test_python_string_array_subclass_converts_no_values_at_bind(duckdb_cursor, monkeypatch, dtype):
    # pandas 2.3 backs python storage with NaN for missing values by a subclass of StringArray.
    class Subclass(type(pd.array([], dtype=_dtype(dtype)))):
        pass

    rows = 300
    values = [None if index % 3 == 1 else f"v{index}" for index in range(rows)]
    frame = pd.DataFrame(
        {"s": Subclass._from_sequence(values, dtype=_dtype(dtype)), "i": numpy.arange(rows, dtype=numpy.int64)}
    )
    assert type(frame["s"].array) is Subclass
    _refuse_conversions(monkeypatch, Subclass, rows)
    result = duckdb_cursor.sql("SELECT s FROM frame ORDER BY i").fetchall()
    assert result == _rows_of(values)


def _invalid_utf8_frames(pa, pieces):
    compute = pytest.importorskip("pyarrow.compute")
    chunks = []
    for rows in pieces:
        values = [b"\xff\xfe", None, *[b"ok"] * (rows - 2)][:rows]
        chunks.append(compute.cast(pa.array(values, type=pa.large_binary()), pa.large_string(), safe=False))
    column = pa.chunked_array(chunks, type=pa.large_string())
    index = numpy.arange(len(column), dtype=numpy.int64)
    frame = pd.DataFrame({"s": pd.Series(pd.arrays.ArrowStringArray(column)), "i": index})
    arrow_dtype_frame = pd.DataFrame({"s": pd.array(column, dtype=pd.ArrowDtype(pa.large_string())), "i": index})
    return frame, arrow_dtype_frame


def _append_and_count(con, frame):
    con.append("copied", frame)
    return con.sql("SELECT count(*) FROM copied").fetchall()


INVALID_UTF8_USES = [
    pytest.param(None, lambda con, frame: con.sql("SELECT length(s) FROM frame ORDER BY i").fetchall(), id="length"),
    pytest.param(None, lambda con, frame: con.sql("SELECT count(*) FROM frame").fetchall(), id="count"),
    pytest.param(None, lambda con, frame: con.sql("SELECT i FROM frame ORDER BY i").fetchall(), id="other_column"),
    pytest.param(None, lambda con, frame: con.sql("SELECT s FROM frame ORDER BY i").fetchall(), id="fetch_text"),
    pytest.param(
        None,
        lambda con, frame: con.execute("CREATE TABLE copied AS SELECT * FROM frame").fetchall(),
        id="create_table_as",
    ),
    pytest.param("CREATE TABLE copied (s VARCHAR, i BIGINT)", _append_and_count, id="append"),
]


def _outcome(setup, use, frame):
    with duckdb.connect() as con:
        if setup:
            con.execute(setup)
        try:
            return ("result", use(con, frame))
        except duckdb.Error as error:
            return (type(error), str(error))


@needs_arrow_chunk_export
@pytest.mark.parametrize(("setup", "use"), INVALID_UTF8_USES)
@pytest.mark.parametrize(
    "pieces",
    [
        pytest.param([ARROW_ROWS_PER_CHUNK], id="one_chunk_at_the_threshold"),
        pytest.param([2 * ARROW_ROWS_PER_CHUNK, 1], id="long_chunk_with_an_appended_row"),
    ],
)
def test_str_invalid_utf8_by_layout(pa, pieces, setup, use):
    frame, arrow_dtype_frame = _invalid_utf8_frames(pa, pieces)
    expected = _outcome(setup, use, arrow_dtype_frame)
    outcome = _outcome(setup, use, frame)
    assert outcome == expected


@pytest.mark.parametrize(("setup", "use"), INVALID_UTF8_USES)
@pytest.mark.parametrize(
    "pieces",
    [
        pytest.param([1, 1, 1], id="one_row_per_chunk"),
        pytest.param([ARROW_ROWS_PER_CHUNK - 1], id="one_chunk_just_below_the_threshold"),
    ],
)
def test_str_invalid_utf8_below_the_threshold_fails_at_bind(pa, pieces, setup, use):
    frame, _ = _invalid_utf8_frames(pa, pieces)
    error_type, message = _outcome(setup, use, frame)
    assert error_type is duckdb.Error
    assert re.search(r"Wrapping .* failed", message)


GIL_HOLD_SECONDS = 0.2
GIL_ROWS = 1_000_000
# A single measurement can miss the scan, so the object and str columns are measured back to back in each round and
# the medians are compared.
GIL_ROUNDS = 3


def _scan_seconds(frame, held_row):
    libc = ctypes.PyDLL(None)
    libc.usleep.argtypes = [ctypes.c_uint]
    holds = []

    def hold_gil(value):
        holds.append(value)
        libc.usleep(int(GIL_HOLD_SECONDS * 1_000_000))
        return value

    with duckdb.connect(config={"threads": 4}) as con:
        con.create_function("hold_gil", hold_gil, ["BIGINT"], "BIGINT")
        con.register("frame", frame)
        con.execute("PRAGMA enable_profiling = 'no_output'")
        con.execute(f"SELECT count(x), sum(CASE WHEN i = {held_row} THEN hold_gil(i) ELSE i END) FROM frame").fetchall()
        operators = list(json.loads(con.get_profiling_information())["operator"])
    seconds = 0.0
    while operators:
        operator = operators.pop()
        if operator["type"] == "TABLE_SCAN":
            seconds += operator["timing"]
        operators.extend(operator.get("children", []))
    return seconds, len(holds)


def _scan_growth_in_holds(frame):
    held, held_calls = _scan_seconds(frame, 0)
    free, free_calls = _scan_seconds(frame, -1)
    return (held - free) / GIL_HOLD_SECONDS, (held_calls, free_calls)


GIL_COLUMNS = [
    pytest.param(
        lambda pa, values: pd.array(values, dtype="str"),
        id="default_str",
        marks=[needs_pandas_3, needs_arrow_chunk_export],
    ),
    pytest.param(
        lambda pa, values: pd.array(values, dtype="string[pyarrow]"),
        id="string_pyarrow",
        marks=needs_arrow_chunk_export,
    ),
    pytest.param(
        lambda pa, values: _chunked(_pieces(values, 1000), "string[pyarrow]").array,
        id="string_pyarrow_many_chunks",
        marks=needs_arrow_chunk_export,
    ),
]


@pytest.mark.skipif(sys.platform == "win32", reason="ctypes cannot load the C library through a null name on Windows")
@pytest.mark.skipif(not getattr(sys, "_is_gil_enabled", lambda: True)(), reason="the interpreter has no GIL")
@pytest.mark.timeout(120)
@pytest.mark.parametrize("make_column", GIL_COLUMNS)
def test_arrow_str_scan_runs_without_gil(pa, make_column):
    values = [f"v{index % 1000}" for index in range(GIL_ROWS)]
    index = numpy.arange(GIL_ROWS, dtype=numpy.int64)
    calibration = pd.DataFrame({"x": pd.Series(values, dtype=object), "i": index})
    frame = pd.DataFrame({"x": make_column(pa, values), "i": index})
    calibration_growths, growths, calls = [], [], set()
    for _ in range(GIL_ROUNDS):
        calibration_growth, _ = _scan_growth_in_holds(calibration)
        growth, round_calls = _scan_growth_in_holds(frame)
        calibration_growths.append(calibration_growth)
        growths.append(growth)
        calls.add(round_calls)
    # An object column takes the GIL for every vector, so its scan must stall behind the hold; otherwise the hold
    # did not overlap the scan and the measurement says nothing.
    assert statistics.median(calibration_growths) >= 1
    assert calls == {(1, 0)}
    assert statistics.median(growths) < 0.5

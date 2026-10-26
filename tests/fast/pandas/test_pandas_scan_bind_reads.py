"""What binding a pandas frame or a dict of arrays reads, and what each column reads as."""

import collections
import datetime
import decimal
import importlib.util
import json
import re
import subprocess
import sys
import textwrap
import tracemalloc
from collections.abc import Callable
from typing import Any, NamedTuple

import pytest

import duckdb
import numpy as np
import pandas as pd

PATHS = ["register", "variable"]

PANDAS_PARTITION_ROWS = 102400
VECTOR_ROWS = 2048
CHILD_TIMEOUT = 60

_HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None
_NEEDS_PYARROW = pytest.mark.skipif(not _HAS_PYARROW, reason="pyarrow not installed")


def _execute(con, source, path, sql):
    if path == "register":
        con.register("registered_source", source)
        return con.execute(sql.format(source="registered_source"))
    return con.execute(sql.format(source="source"))


def _names_and_rows(con, source, path, sql):
    result = _execute(con, source, path, sql)
    return [column[0] for column in result.description], result.fetchall()


def _run_child(body):
    # A column shorter than the bound row count reads past its buffer, which can kill the process.
    script = textwrap.dedent(
        """
        import json
        import sys

        import numpy as np
        import pandas as pd

        import duckdb

        if sys.platform != "win32":
            import resource

            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        con = duckdb.connect()


        def report(run):
            try:
                outcome = {"rows": run()}
            except duckdb.Error as error:
                outcome = {"error": type(error).__name__, "message": str(error)}
            print("outcome " + json.dumps(outcome, default=repr))
        """
    ) + textwrap.dedent(body)
    try:
        child = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=CHILD_TIMEOUT, check=False
        )
    except subprocess.TimeoutExpired as hang:
        pytest.fail(f"the child did not finish within {CHILD_TIMEOUT} seconds: {hang.stdout!r} {hang.stderr!r}")
    outcomes = [line.removeprefix("outcome ") for line in child.stdout.splitlines() if line.startswith("outcome ")]
    return {
        "returncode": child.returncode,
        "outcome": json.loads(outcomes[0]) if len(outcomes) == 1 else None,
        "stderr": child.stderr[-2000:],
    }


def _refusal(child, words):
    """The error class and which of the given words its message names, or what happened instead."""
    outcome = child["outcome"]
    if child["returncode"] != 0 or outcome is None:
        return {"returncode": child["returncode"], "stderr": child["stderr"]}
    if "error" not in outcome:
        return outcome
    return {"error": outcome["error"], "words": words & set(re.findall(r"\w+", outcome["message"]))}


def _length_refusal(column, rows, bound_rows):
    return {"error": "InvalidInputException", "words": {column, str(rows), str(bound_rows)}}


def _count_after(message, column):
    """The first number the message gives after naming the column."""
    match = re.search(rf"\b{re.escape(column)}\b\D*?(\d+)", message)
    return match.group(1) if match else None


_TZ_TIMES = pd.date_range("2020-01-01 00:00:00.123456", periods=6, freq="h", tz="Europe/Amsterdam")
_NAIVE_TIMES = pd.date_range("2020-01-01", periods=6, freq="D")
# pandas 3 builds these in microseconds and pandas 2 in nanoseconds.
_NAIVE_TIMESTAMP = {"us": "TIMESTAMP", "ns": "TIMESTAMP_NS"}[np.datetime_data(_NAIVE_TIMES.dtype)[0]]
_DECIMALS = [decimal.Decimal(text) if text else None for text in ["1.25", None, "3.50", "4.75", None, "6.25"]]
_TEXT = ["a", None, "c", "d", "e", "f"]


class ColumnCase(NamedTuple):
    build: Callable[[], Any]
    sql_type: str
    values: list


FRAME_COLUMNS = {
    "numpy_int64": ColumnCase(lambda: np.array([0, 1, 2, 3, 4, 5], dtype="int64"), "BIGINT", [0, 1, 2, 3, 4, 5]),
    "numpy_uint8": ColumnCase(lambda: np.array([0, 1, 2, 3, 4, 255], dtype="uint8"), "UTINYINT", [0, 1, 2, 3, 4, 255]),
    "numpy_float32": ColumnCase(
        lambda: np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype="float32"),
        "FLOAT",
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    ),
    "numpy_float64_nan": ColumnCase(
        lambda: np.array([1.5, np.nan, 3.5, 4.5, 5.5, 6.5], dtype="float64"),
        "DOUBLE",
        [1.5, None, 3.5, 4.5, 5.5, 6.5],
    ),
    "numpy_float16": ColumnCase(
        lambda: np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype="float16"),
        "FLOAT",
        [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    ),
    "numpy_bool": ColumnCase(
        lambda: np.array([True, False, True, True, False, False], dtype="bool"),
        "BOOLEAN",
        [True, False, True, True, False, False],
    ),
    "object_text": ColumnCase(lambda: pd.Series(_TEXT, dtype=object), "VARCHAR", _TEXT),
    "object_decimal": ColumnCase(lambda: pd.Series(_DECIMALS, dtype=object), "DECIMAL(3,2)", _DECIMALS),
    "str": ColumnCase(lambda: pd.Series(_TEXT, dtype="str"), "VARCHAR", _TEXT),
    "string_python": ColumnCase(lambda: pd.Series(_TEXT, dtype="string[python]"), "VARCHAR", _TEXT),
    "string_pyarrow": ColumnCase(lambda: pd.Series(_TEXT, dtype="string[pyarrow]"), "VARCHAR", _TEXT),
    "category_text": ColumnCase(
        lambda: pd.Series(["x", "y", None, "x", "y", "x"], dtype="category"),
        "ENUM('x', 'y')",
        ["x", "y", None, "x", "y", "x"],
    ),
    "category_int": ColumnCase(
        lambda: pd.Series(np.array([1, 2, 3, 1, 2, 3], dtype="int64"), dtype="category"),
        "BIGINT",
        [1, 2, 3, 1, 2, 3],
    ),
    "category_int_missing": ColumnCase(
        lambda: pd.Series([1, 2, None, 1, 2, 3], dtype="category"),
        "DOUBLE",
        [1.0, 2.0, None, 1.0, 2.0, 3.0],
    ),
    "category_bool_missing": ColumnCase(
        lambda: pd.Series([True, False, None, True, False, True], dtype="category"),
        "BOOLEAN",
        [True, False, None, True, False, True],
    ),
    "datetime_tz": ColumnCase(
        lambda: pd.Series(_TZ_TIMES),
        "TIMESTAMP WITH TIME ZONE",
        [timestamp.to_pydatetime() for timestamp in _TZ_TIMES],
    ),
    "datetime_naive": ColumnCase(
        lambda: pd.Series(_NAIVE_TIMES),
        _NAIVE_TIMESTAMP,
        [timestamp.to_pydatetime() for timestamp in _NAIVE_TIMES],
    ),
    "timedelta": ColumnCase(
        lambda: pd.Series(pd.to_timedelta(np.array([0, 1, 2, 3, 4, 5], dtype="int64"), unit="s")),
        "INTERVAL",
        [datetime.timedelta(seconds=seconds) for seconds in range(6)],
    ),
    "Int64": ColumnCase(lambda: pd.array([1, None, 3, 4, 5, 6], dtype="Int64"), "BIGINT", [1, None, 3, 4, 5, 6]),
    "Int8": ColumnCase(lambda: pd.array([1, None, 3, -4, 5, 6], dtype="Int8"), "TINYINT", [1, None, 3, -4, 5, 6]),
    "boolean": ColumnCase(
        lambda: pd.array([True, None, False, True, False, True], dtype="boolean"),
        "BOOLEAN",
        [True, None, False, True, False, True],
    ),
    "Float64": ColumnCase(
        lambda: pd.array([1.5, None, 3.5, 4.5, 5.5, 6.5], dtype="Float64"),
        "DOUBLE",
        [1.5, None, 3.5, 4.5, 5.5, 6.5],
    ),
}
_COLUMNS_NEEDING_PYARROW = {"string_pyarrow"}


@pytest.fixture(scope="module")
def wide_frame():
    return pd.DataFrame(
        {
            name: build()
            for name, (build, _, _) in FRAME_COLUMNS.items()
            if _HAS_PYARROW or name not in _COLUMNS_NEEDING_PYARROW
        }
    )


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "name",
    [pytest.param(name, marks=_NEEDS_PYARROW) if name in _COLUMNS_NEEDING_PYARROW else name for name in FRAME_COLUMNS],
)
def test_reading_each_dtype_keeps_type_and_values(duckdb_cursor, wide_frame, path, name):
    _, sql_type, values = FRAME_COLUMNS[name]
    sql = f'SELECT typeof("{name}"), "{name}" FROM {{source}}'
    assert _execute(duckdb_cursor, wide_frame, path, sql).fetchall() == [(sql_type, value) for value in values]


# Five rows under four keys, so a row count taken from the number of keys reads the wrong length.
DICT_COLUMNS = {
    "k": ColumnCase(lambda: np.array([0, 1, 2, 3, 4], dtype="int64"), "BIGINT", [0, 1, 2, 3, 4]),
    "u": ColumnCase(lambda: np.array(["b", "a", "b", "c", "a"]), "ENUM('a', 'b', 'c')", ["b", "a", "b", "c", "a"]),
    "h": ColumnCase(lambda: np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype="float16"), "FLOAT", [1.5, 2.5, 3.5, 4.5, 5.5]),
    "o": ColumnCase(lambda: np.array(["p", None, "q", "r", "s"], dtype=object), "VARCHAR", ["p", None, "q", "r", "s"]),
}


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("name", list(DICT_COLUMNS))
def test_dict_columns_keep_type_and_values(duckdb_cursor, path, name):
    source = {key: case.build() for key, case in DICT_COLUMNS.items()}
    _, sql_type, values = DICT_COLUMNS[name]
    sql = f"SELECT typeof({name}), {name} FROM {{source}}"
    assert _execute(duckdb_cursor, source, path, sql).fetchall() == [(sql_type, value) for value in values]


UNSUPPORTED_SOURCES = [
    pytest.param(
        lambda: pd.DataFrame(
            {"k": np.array([1, 2], dtype="int64"), "p": pd.period_range("2020-01", periods=2, freq="M")}
        ),
        id="frame_period",
    ),
    pytest.param(
        lambda: pd.DataFrame(
            {"k": np.array([1, 2], dtype="int64"), "p": pd.arrays.IntervalArray.from_breaks([0, 1, 2])}
        ),
        id="frame_interval",
    ),
    pytest.param(
        lambda: {"k": np.array([1, 2], dtype="int64"), "p": np.array([1 + 2j, 3j], dtype="complex128")},
        id="dict_complex",
    ),
]


@pytest.mark.parametrize("make_source", UNSUPPORTED_SOURCES)
def test_registering_an_unsupported_dtype_fails(duckdb_cursor, make_source):
    source = make_source()
    with pytest.raises(duckdb.NotImplementedException, match="not recognized"):
        duckdb_cursor.register("registered_source", source)


@pytest.mark.parametrize("make_source", UNSUPPORTED_SOURCES)
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT k FROM source",
        "SELECT count(*) FROM source",
        "DESCRIBE SELECT k FROM source",
    ],
    ids=["read_k", "count_star", "describe"],
)
def test_unsupported_dtype_fails_even_when_unread(duckdb_cursor, make_source, sql):
    source = make_source()  # noqa: F841
    with pytest.raises(duckdb.NotImplementedException, match="not recognized"):
        duckdb_cursor.execute(sql)


def _frame_with_column_answer(answer):
    class AnsweringFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return AnsweringFrame

        def __getitem__(self, key) -> object:
            if isinstance(key, str) and key == "b":
                return pd.Series(answer)
            return super().__getitem__(key)

    return AnsweringFrame({"a": np.arange(4, dtype="int64"), "b": np.arange(4, dtype="int64")})


def _dict_with_array_answer(answer):
    class AnsweringDict(dict):
        def __getitem__(self, key) -> object:
            if key == "b":
                return answer
            return super().__getitem__(key)

    return AnsweringDict({"a": np.arange(4, dtype="int64"), "b": np.arange(4, dtype="int64")})


@pytest.mark.parametrize("path", PATHS)
def test_frame_subclass_type_follows_its_column(duckdb_cursor, path):
    answer = np.array([1.5, 2.5, 3.5, 4.5], dtype="float64")
    source = _frame_with_column_answer(answer)
    expected = [("DOUBLE", value) for value in answer.tolist()]
    assert _execute(duckdb_cursor, source, path, "SELECT typeof(b), b FROM {source}").fetchall() == expected


@pytest.mark.parametrize("path", PATHS)
def test_dict_subclass_type_follows_its_array(duckdb_cursor, path):
    answer = np.array([1.5, 2.5, 3.5, 4.5], dtype="float64")
    source = _dict_with_array_answer(answer)
    expected = [("DOUBLE", value) for value in answer.tolist()]
    assert _execute(duckdb_cursor, source, path, "SELECT typeof(b), b FROM {source}").fetchall() == expected


def _counting_frame(reads):
    class CountingFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return CountingFrame

        def __getitem__(self, key) -> object:
            if isinstance(key, str):
                reads[key] += 1
            return super().__getitem__(key)

    return CountingFrame(
        {
            "k": np.arange(4, dtype="int64"),
            "o": pd.Series(["p", None, "q", "r"], dtype=object),
            "f": np.arange(4, dtype="float64"),
        }
    )


def _counting_dict(reads):
    class CountingDict(dict):
        def __getitem__(self, key) -> object:
            reads[key] += 1
            return super().__getitem__(key)

    return CountingDict(
        {
            "k": np.arange(4, dtype="int64"),
            "o": np.array(["p", None, "q", "r"], dtype=object),
            "f": np.arange(4, dtype="float64"),
        }
    )


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("make_source", [_counting_frame, _counting_dict], ids=["frame", "dict"])
def test_each_column_is_read_once_per_query(duckdb_cursor, path, make_source):
    reads = collections.Counter()
    source = make_source(reads)
    if path == "register":
        duckdb_cursor.register("registered_source", source)
        name = "registered_source"
    else:
        name = "source"
    reads.clear()
    duckdb_cursor.execute(f"SELECT k FROM {name}").fetchall()
    assert dict(reads) == {"k": 1, "o": 1, "f": 1}


_CHANGED_REGISTERED_DICT = """
def run():
    source = SOURCE
    con.register("registered_source", source)
    source["changed"] = REPLACEMENT
    return con.execute(QUERY).fetchall()


report(run)
"""


def _changed_registered_dict(source, replacement, query):
    return (
        _CHANGED_REGISTERED_DICT.replace("SOURCE", source)
        .replace("REPLACEMENT", replacement)
        .replace("QUERY", repr(query))
    )


_KEPT_THEN_CHANGED = '{"kept": np.arange(50, dtype="int64"), "changed": np.arange(50, dtype="int64")}'
_CHANGED_THEN_KEPT = '{"changed": np.arange(50, dtype="int64"), "kept": np.arange(50, dtype="int64")}'


@pytest.mark.parametrize(
    ("replacement", "rows", "query"),
    [
        pytest.param(
            'np.array([7, 8, 9], dtype="int64")',
            3,
            "SELECT count(changed), min(changed), max(changed) FROM registered_source",
            id="shorter_read",
        ),
        pytest.param(
            'np.array([7, 8, 9], dtype="int64")', 3, "SELECT max(kept) FROM registered_source", id="shorter_unread"
        ),
        pytest.param(
            'np.array([7, 8, 9], dtype="int64")', 3, "SELECT count(*) FROM registered_source", id="shorter_count_star"
        ),
        pytest.param(
            'np.arange(60, dtype="int64")',
            60,
            "SELECT count(*), count(changed), max(changed) FROM registered_source",
            id="longer_read",
        ),
    ],
)
def test_registered_dict_with_unequal_lengths_is_refused(replacement, rows, query):
    body = _changed_registered_dict(_KEPT_THEN_CHANGED, replacement, query)
    expected = _length_refusal("changed", rows, 50)
    assert _refusal(_run_child(body), expected["words"]) == expected


def test_registered_dict_with_its_first_array_shrunk_is_refused():
    body = _changed_registered_dict(
        _CHANGED_THEN_KEPT,
        'np.array([7, 8, 9], dtype="int64")',
        "SELECT count(*), max(kept), max(changed) FROM registered_source",
    )
    child = _run_child(body)
    words = {"changed", "kept", "3", "50"}
    assert _refusal(child, words) == {"error": "InvalidInputException", "words": words}
    message = child["outcome"]["message"]
    assert {column: _count_after(message, column) for column in ("kept", "changed")} == {"kept": "50", "changed": "3"}


@pytest.mark.parametrize(
    "replacement",
    ['np.arange(100, dtype="int64").reshape(50, 2)', 'np.arange(50, dtype="int64").reshape(50, 1)'],
    ids=["50x2", "50x1"],
)
def test_registered_dict_with_a_two_dimensional_array_is_refused(replacement):
    body = _changed_registered_dict(_KEPT_THEN_CHANGED, replacement, "SELECT changed FROM registered_source")
    expected = {"error": "InvalidInputException", "words": {"changed"}}
    assert _refusal(_run_child(body), expected["words"]) == expected


_RESIZED_FRAME_COLUMN = """
class ResizedFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return ResizedFrame

    def __getitem__(self, key):
        if isinstance(key, str) and key == "resized":
            return pd.Series(np.arange(LENGTH, dtype="int64"))
        return super().__getitem__(key)


def make_source():
    return ResizedFrame({"kept": np.arange(50, dtype="int64"), "resized": np.arange(50, dtype="int64")})
"""

_RESIZED_DICT_ARRAY = """
class ResizedDict(dict):
    def __getitem__(self, key):
        if key == "resized":
            return np.arange(LENGTH, dtype="int64")
        return super().__getitem__(key)


def make_source():
    return ResizedDict({"kept": np.arange(50, dtype="int64"), "resized": np.arange(50, dtype="int64")})
"""

# Masks are assigned after construction because the pandas and numpy constructors refuse one of another length.
_RESIZED_MASK_FRAME = """
def make_source():
    column = pd.array(np.arange(50, dtype="int64"), dtype="Int64")
    column._mask = np.zeros(LENGTH, dtype=bool)
    return pd.DataFrame({"kept": np.arange(50, dtype="int64"), "resized": column}, copy=False)
"""

_RESIZED_MASK_DICT = """
def make_source():
    column = np.ma.masked_array(np.arange(50, dtype="int64"), mask=np.zeros(50, dtype=bool))
    column._mask = np.zeros(LENGTH, dtype=bool)
    return {"kept": np.arange(50, dtype="int64"), "resized": column}
"""

_RESIZED_STR_FRAME_COLUMN = """
import pyarrow as pa


def resized_column():
    chunks, start = [], 0
    for rows in CHUNKS:
        chunks.append(pa.array([f"v{index}" for index in range(start, start + rows)], type=pa.large_string()))
        start += rows
    return pd.Series(pd.array(pa.chunked_array(chunks, type=pa.large_string()), dtype="str"))


class ResizedFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return ResizedFrame

    def __getitem__(self, key):
        if isinstance(key, str) and key == "resized":
            return resized_column()
        return super().__getitem__(key)


def make_source():
    column = pd.array([f"w{index}" for index in range(BOUND)], dtype="str")
    return ResizedFrame({"kept": np.arange(BOUND, dtype="int64"), "resized": column})
"""

_QUERY_THROUGH_PATH = {
    "register": """
def run():
    con.register("registered_source", make_source())
    return con.execute("SELECT resized FROM registered_source").fetchall()


report(run)
""",
    "variable": """
def run():
    source = make_source()
    return con.execute("SELECT resized FROM source").fetchall()


report(run)
""",
}

RESIZED_LENGTHS = [pytest.param(3, id="shorter"), pytest.param(60, id="longer")]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("length", RESIZED_LENGTHS)
def test_frame_subclass_with_a_column_of_another_length_is_refused(path, length):
    body = _RESIZED_FRAME_COLUMN.replace("LENGTH", str(length)) + _QUERY_THROUGH_PATH[path]
    expected = _length_refusal("resized", length, 50)
    assert _refusal(_run_child(body), expected["words"]) == expected


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("length", RESIZED_LENGTHS)
def test_dict_subclass_with_an_array_of_another_length_is_refused(path, length):
    body = _RESIZED_DICT_ARRAY.replace("LENGTH", str(length)) + _QUERY_THROUGH_PATH[path]
    expected = _length_refusal("resized", length, 50)
    assert _refusal(_run_child(body), expected["words"]) == expected


RESIZED_MASK_LENGTHS = [pytest.param(2, id="shorter"), pytest.param(60, id="longer")]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("length", RESIZED_MASK_LENGTHS)
def test_nullable_column_with_a_mask_of_another_length_is_refused(path, length):
    body = _RESIZED_MASK_FRAME.replace("LENGTH", str(length)) + _QUERY_THROUGH_PATH[path]
    expected = _length_refusal("resized", length, 50)
    assert _refusal(_run_child(body), expected["words"]) == expected


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("length", RESIZED_MASK_LENGTHS)
def test_masked_array_with_a_mask_of_another_length_is_refused(path, length):
    body = _RESIZED_MASK_DICT.replace("LENGTH", str(length)) + _QUERY_THROUGH_PATH[path]
    expected = _length_refusal("resized", length, 50)
    assert _refusal(_run_child(body), expected["words"]) == expected


@_NEEDS_PYARROW
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    ("bound_rows", "chunks"),
    [
        pytest.param(50, [3], id="shorter"),
        pytest.param(50, [60], id="longer"),
        # Chunks of a hundred rows or more are read from Arrow rather than through to_numpy, and need their own check.
        pytest.param(500, [300], id="shorter_long_chunk"),
        pytest.param(500, [600], id="longer_long_chunk"),
        pytest.param(500, [100, 100, 100], id="shorter_three_chunks"),
    ],
)
def test_frame_subclass_with_an_arrow_string_column_of_another_length_is_refused(path, bound_rows, chunks):
    body = (
        _RESIZED_STR_FRAME_COLUMN.replace("BOUND", str(bound_rows)).replace("CHUNKS", repr(chunks))
        + _QUERY_THROUGH_PATH[path]
    )
    expected = _length_refusal("resized", sum(chunks), bound_rows)
    assert _refusal(_run_child(body), expected["words"]) == expected


_CONVERTED_SERIES = {
    "string_categorical": 'pd.Series(pd.Categorical([f"v{index % 3}" for index in range(LENGTH)]))',
    "int_categorical": 'pd.Series(pd.Categorical(np.arange(LENGTH, dtype="int64") % 3))',
    "float16": 'pd.Series(np.arange(LENGTH, dtype="float16"))',
    "tz_datetime": 'pd.Series(pd.date_range("2020-01-01", periods=LENGTH, freq="h", tz="Europe/Amsterdam"))',
}

_CONVERTED_FRAME_COLUMN = """
class ResizedFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return ResizedFrame

    def __getitem__(self, key):
        if isinstance(key, str) and key == "resized":
            return COLUMN
        return super().__getitem__(key)


def make_source():
    return ResizedFrame({"kept": np.arange(50, dtype="int64"), "resized": np.arange(50, dtype="int64")})
"""


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("length", RESIZED_LENGTHS)
@pytest.mark.parametrize("column", list(_CONVERTED_SERIES))
def test_frame_subclass_with_a_converted_column_of_another_length_is_refused(path, length, column):
    body = (
        _CONVERTED_FRAME_COLUMN.replace("COLUMN", _CONVERTED_SERIES[column]).replace("LENGTH", str(length))
        + _QUERY_THROUGH_PATH[path]
    )
    expected = _length_refusal("resized", length, 50)
    assert _refusal(_run_child(body), expected["words"]) == expected


_RESIZED_DATA_FRAME = """
def make_source():
    column = pd.array(np.arange(50, dtype="int64"), dtype="Int64")
    frame = pd.DataFrame({"kept": np.arange(50, dtype="int64"), "resized": column}, copy=False)
    column._data = np.arange(LENGTH, dtype="int64")
    return frame
"""


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("length", RESIZED_LENGTHS)
def test_nullable_column_with_data_of_another_length_is_refused(path, length):
    body = _RESIZED_DATA_FRAME.replace("LENGTH", str(length)) + _QUERY_THROUGH_PATH[path]
    expected = _length_refusal("resized", length, 50)
    assert _refusal(_run_child(body), expected["words"]) == expected


_CONVERTED_ARRAYS = {
    "float16": 'np.arange(LENGTH, dtype="float16")',
    "unicode": 'np.array([f"v{index}" for index in range(LENGTH)])',
}

_CONVERTED_DICT_ARRAY = """
class ResizedDict(dict):
    def __getitem__(self, key):
        if key == "resized":
            return ARRAY
        return super().__getitem__(key)


def make_source():
    arrays = {"kept": np.arange(50, dtype="int64"), "resized": np.arange(50, dtype="int64")}
    return ResizedDict({key: arrays[key] for key in ORDER})
"""


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("length", RESIZED_LENGTHS)
@pytest.mark.parametrize("order", [["kept", "resized"], ["resized", "kept"]], ids=["resized_last", "resized_first"])
@pytest.mark.parametrize("array", list(_CONVERTED_ARRAYS))
def test_dict_subclass_with_a_converted_array_of_another_length_is_refused(path, length, order, array):
    body = (
        _CONVERTED_DICT_ARRAY.replace("ARRAY", _CONVERTED_ARRAYS[array])
        .replace("LENGTH", str(length))
        .replace("ORDER", repr(order))
        + _QUERY_THROUGH_PATH[path]
    )
    expected = _length_refusal("resized", length, 50)
    assert _refusal(_run_child(body), expected["words"]) == expected


_NOT_AN_ARRAY = """
class ListShape:
    shape = [50]
    dtype = np.dtype("int64")


class ArrayLike:
    def __init__(self):
        self._array = np.arange(50, dtype="int64")
        self.shape = self._array.shape
        self.dtype = self._array.dtype
        self.strides = self._array.strides


class NotAnArrayDict(dict):
    def __getitem__(self, key):
        if key == "resized":
            return SLOT
        return super().__getitem__(key)


def make_source():
    arrays = {"kept": np.arange(50, dtype="int64"), "resized": np.arange(50, dtype="int64")}
    return NotAnArrayDict({key: arrays[key] for key in ORDER})
"""


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("order", [["kept", "resized"], ["resized", "kept"]], ids=["resized_last", "resized_first"])
@pytest.mark.parametrize(
    "slot",
    ["ListShape()", "ArrayLike()", 'pd.Series(np.arange(50, dtype="int64"))'],
    ids=["list_shape", "array_like", "series"],
)
def test_dict_subclass_with_a_slot_that_is_not_an_ndarray_is_refused(path, order, slot):
    body = _NOT_AN_ARRAY.replace("SLOT", slot).replace("ORDER", repr(order)) + _QUERY_THROUGH_PATH[path]
    expected = {"error": "InvalidInputException", "words": {"resized", "ndarray"}}
    assert _refusal(_run_child(body), expected["words"]) == expected


@pytest.mark.parametrize("replacement", ['np.array([["a"]] * 50)', 'np.array([["a", "b"]] * 50)'], ids=["50x1", "50x2"])
def test_registered_dict_with_a_two_dimensional_unicode_array_is_refused(replacement):
    body = _changed_registered_dict(_KEPT_THEN_CHANGED, replacement, "SELECT changed FROM registered_source")
    expected = {"error": "InvalidInputException", "words": {"changed"}}
    assert _refusal(_run_child(body), expected["words"]) == expected


def test_length_refusal_names_the_bound_column_name():
    body = """
def run():
    source = {"a": np.arange(50, dtype="int64"), "A": np.arange(50, dtype="int64")}
    con.register("registered_source", source)
    source["A"] = np.array([7, 8, 9], dtype="int64")
    return con.execute("SELECT * FROM registered_source").fetchall()


report(run)
"""
    expected = _length_refusal("A_1", 3, 50)
    assert _refusal(_run_child(body), expected["words"]) == expected


COLUMN_LABELS = [
    pytest.param([1], ["1"], id="int"),
    pytest.param([b"k"], ["b'k'"], id="bytes"),
    pytest.param(["a", "A"], ["a", "A_1"], id="same_ignoring_case"),
    pytest.param([1, "1"], ["1", "1_1"], id="same_after_str"),
]


def _labeled_source(kind, labels):
    arrays = [np.arange(3, dtype="int64") + 10 * position for position in range(len(labels))]
    if kind == "frame":
        source = pd.DataFrame(dict(zip(labels, arrays, strict=False)))
    else:
        source = dict(zip(labels, arrays, strict=False))
    return source, list(zip(*(array.tolist() for array in arrays), strict=False))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(("labels", "names"), COLUMN_LABELS)
def test_frame_column_names_are_str_of_labels_deduplicated(duckdb_cursor, path, labels, names):
    source, rows = _labeled_source("frame", labels)
    assert _names_and_rows(duckdb_cursor, source, path, "SELECT * FROM {source}") == (names, rows)
    assert list(source.columns) == labels


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(("labels", "names"), COLUMN_LABELS)
def test_dict_column_names_are_str_of_keys_deduplicated(duckdb_cursor, path, labels, names):
    source, rows = _labeled_source("dict", labels)
    assert _names_and_rows(duckdb_cursor, source, path, "SELECT * FROM {source}") == (names, rows)
    assert list(source) == labels


def _pandas_scan_projections(con, source, sql):
    explanation = _execute(con, source, "variable", "EXPLAIN " + sql).fetchall()[0][1]
    projections = []
    inside_scan = False
    for line in explanation.splitlines():
        if "Pandas Scan" in line:
            inside_scan = True
        elif inside_scan and line.startswith("╰"):
            inside_scan = False
        elif inside_scan and "Projections:" in line:
            projections.append(line.split("Projections:")[1].strip(" │"))
    return projections


COLUMN_FREE_QUERIES = [
    pytest.param("SELECT count(*) FROM {source}", id="count_star"),
    pytest.param("SELECT EXISTS (SELECT 1 FROM {source})", id="exists"),
]


@pytest.mark.parametrize("sql", COLUMN_FREE_QUERIES)
@pytest.mark.parametrize("first", ["k", "s"])
def test_count_star_converts_no_column(duckdb_cursor, sql, first):
    columns = {"k": np.arange(3, dtype="int64"), "s": pd.Series(["a", "b", "c"], dtype="str")}
    source = pd.DataFrame({first: columns[first], **columns})
    assert _pandas_scan_projections(duckdb_cursor, source, sql) == []


def test_a_read_column_is_requested(duckdb_cursor):
    source = pd.DataFrame({"k": np.arange(3, dtype="int64"), "s": pd.Series(["a", "b", "c"], dtype="str")})
    assert _pandas_scan_projections(duckdb_cursor, source, "SELECT s FROM {source}") == ["s"]


def _unconvertible_first_column(kind):
    if kind == "frame":
        return pd.DataFrame({"s": pd.Series(["ok", "\ud800"], dtype=object), "k": np.array([1, 2], dtype="int64")})
    return {"s": np.array(["ok", "\ud800"], dtype=object), "k": np.array([1, 2], dtype="int64")}


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        pytest.param("SELECT count(*) FROM {source}", [(2,)], id="count_star"),
        pytest.param("SELECT EXISTS (SELECT 1 FROM {source})", [(True,)], id="exists"),
    ],
)
@pytest.mark.parametrize("kind", ["frame", "dict"])
def test_count_star_ignores_unconvertible_column_zero(duckdb_cursor, path, sql, expected, kind):
    source = _unconvertible_first_column(kind)
    assert _execute(duckdb_cursor, source, path, sql).fetchall() == expected


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kind", ["frame", "dict"])
def test_reading_the_unconvertible_column_still_fails(duckdb_cursor, path, kind):
    source = _unconvertible_first_column(kind)
    with pytest.raises(duckdb.InvalidInputException, match="surrogate"):
        _execute(duckdb_cursor, source, path, "SELECT count(s) FROM {source}").fetchall()


_COUNTED_ROWS = 2 * PANDAS_PARTITION_ROWS + VECTOR_ROWS + 5


def _counted_source(kind, rows):
    columns = {"k": np.arange(rows, dtype="int64"), "f": np.arange(rows, dtype="float64")}
    return pd.DataFrame(columns) if kind == "frame" else columns


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    ("sql", "answer"),
    [
        pytest.param("SELECT count(*) FROM {source}", lambda rows: [(rows,)], id="count_star"),
        pytest.param("SELECT EXISTS (SELECT 1 FROM {source})", lambda rows: [(rows > 0,)], id="exists"),
    ],
)
@pytest.mark.parametrize("rows", [_COUNTED_ROWS, 0], ids=["across_partitions", "empty"])
@pytest.mark.parametrize("kind", ["frame", "dict"])
def test_count_star_and_exists_see_every_row(duckdb_cursor, path, sql, answer, rows, kind):
    source = _counted_source(kind, rows)
    expected = answer(rows)
    duckdb_cursor.execute("SET threads=4")
    assert _execute(duckdb_cursor, source, path, sql).fetchall() == expected


_SLICED_ROWS = 3 * (PANDAS_PARTITION_ROWS + VECTOR_ROWS) + 7
_LAYOUT_ROWS = 2 * (PANDAS_PARTITION_ROWS + VECTOR_ROWS) + 7


def _edge_rows(rows):
    return [VECTOR_ROWS - 1, VECTOR_ROWS, PANDAS_PARTITION_ROWS - 1, PANDAS_PARTITION_ROWS, rows - 1]


def _nullable_values(dtype, positions):
    if dtype == "Int64":
        return positions
    if dtype == "boolean":
        return positions % 5 < 2
    return positions * 0.5


def _missing_at(positions, edges):
    missing = positions % 7 == 3
    missing[edges] = True
    return missing


_NULLABLE_ARRAYS = {
    "Int64": pd.arrays.IntegerArray,
    "boolean": pd.arrays.BooleanArray,
    "Float64": pd.arrays.FloatingArray,
}


def _length_and_first_differences(result, expected, limit=5):
    actual = [row[0] for row in result.fetchall()]
    differences = [(position, a, e) for position, (a, e) in enumerate(zip(actual, expected, strict=False)) if a != e]
    return len(actual), differences[:limit]


def _scan_in_order(con, source, path):
    con.execute("SET threads=4")
    return _execute(con, source, path, "SELECT x FROM {source} ORDER BY i")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("rows", [slice(None, None, 2), slice(1, None, 3)], ids=["every_second", "every_third_from_1"])
@pytest.mark.parametrize("dtype", ["Int64", "boolean", "Float64"])
def test_sliced_nullable_column_keeps_its_mask(duckdb_cursor, path, rows, dtype):
    positions = np.arange(_SLICED_ROWS, dtype="int64")
    kept = positions[rows]
    column = pd.array(_nullable_values(dtype, positions), dtype=dtype)
    column[_missing_at(positions, kept[_edge_rows(len(kept))])] = pd.NA
    source = pd.DataFrame({"i": positions, "x": column}).iloc[rows]
    expected = source["x"].to_numpy(dtype=object, na_value=None).tolist()
    assert len(expected) > PANDAS_PARTITION_ROWS
    result = _scan_in_order(duckdb_cursor, source, path)
    assert _length_and_first_differences(result, expected) == (len(expected), [])


def _nullable_frame(dtype, data, mask):
    positions = np.arange(_LAYOUT_ROWS, dtype="int64")
    return pd.DataFrame({"i": positions, "x": _NULLABLE_ARRAYS[dtype](data, mask)}, copy=False)


def _every_second_of(values):
    # The skipped elements hold other values, so a read with the wrong step shows in the result.
    backing = np.empty(2 * len(values), dtype=values.dtype)
    backing[::2] = values
    backing[1::2] = ~values if values.dtype == bool else values + 1
    return backing[::2]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("dtype", ["Int64", "boolean", "Float64"])
def test_nullable_column_with_a_strided_mask_keeps_it(duckdb_cursor, path, dtype):
    positions = np.arange(_LAYOUT_ROWS, dtype="int64")
    data = _nullable_values(dtype, positions)
    source = _nullable_frame(dtype, data, _every_second_of(_missing_at(positions, _edge_rows(_LAYOUT_ROWS))))
    column = source["x"].array
    assert (column._data.strides[0] == column._data.itemsize, column._mask.strides[0]) == (True, 2)
    expected = source["x"].to_numpy(dtype=object, na_value=None).tolist()
    result = _scan_in_order(duckdb_cursor, source, path)
    assert _length_and_first_differences(result, expected) == (len(expected), [])


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("dtype", ["Int64", "boolean", "Float64"])
def test_nullable_column_with_strided_data_keeps_its_mask(duckdb_cursor, path, dtype):
    positions = np.arange(_LAYOUT_ROWS, dtype="int64")
    data = _every_second_of(_nullable_values(dtype, positions))
    source = _nullable_frame(dtype, data, _missing_at(positions, _edge_rows(_LAYOUT_ROWS)))
    column = source["x"].array
    assert (column._data.strides[0] == 2 * column._data.itemsize, column._mask.strides[0]) == (True, 1)
    expected = source["x"].to_numpy(dtype=object, na_value=None).tolist()
    result = _scan_in_order(duckdb_cursor, source, path)
    assert _length_and_first_differences(result, expected) == (len(expected), [])


BIND_HEAP_ROWS = 200_000


BIND_HEAP_SOURCES = [
    pytest.param(lambda positions: pd.array(positions, dtype="Int64"), "frame", id="Int64"),
    pytest.param(lambda positions: pd.array(positions % 2 == 0, dtype="boolean"), "frame", id="boolean"),
    pytest.param(lambda positions: pd.array(positions * 0.5, dtype="Float64"), "frame", id="Float64"),
    pytest.param(lambda positions: np.ma.masked_array(positions), "dict", id="masked_nomask"),
    pytest.param(
        lambda positions: np.ma.masked_array(positions, mask=np.zeros(len(positions), dtype=bool)),
        "dict",
        id="masked_all_false",
    ),
    pytest.param(
        lambda positions: np.ma.masked_array(positions, mask=positions % 3 == 0), "dict", id="masked_some_true"
    ),
]


def _bind_heap_peak(con, source, path):
    if path == "register":
        con.register("registered_source", source)
    query = f"SELECT i FROM {'registered_source' if path == 'register' else 'source'} LIMIT 0"
    con.sql(query).fetchall()
    tracemalloc.start()
    try:
        con.sql(query).fetchall()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(("make_column", "kind"), BIND_HEAP_SOURCES)
def test_masked_column_bind_allocates_nothing_per_row(duckdb_cursor, path, make_column, kind):
    # A copy of a contiguous mask, or a mask built for an array without one, costs a byte per row.
    positions = np.arange(BIND_HEAP_ROWS, dtype="int64")
    columns = {"i": positions, "x": make_column(positions)}
    source = pd.DataFrame(columns, copy=False) if kind == "frame" else columns
    assert _bind_heap_peak(duckdb_cursor, source, path) < BIND_HEAP_ROWS // 4


# Left out: float16 is copied to float32, categoricals copy their codes or values, and str columns have their own tests.
_FRAME_COLUMNS_READ_IN_PLACE = [
    name
    for name in FRAME_COLUMNS
    if name not in {"numpy_float16", "str", "string_python", "string_pyarrow"} and not name.startswith("category")
]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("name", _FRAME_COLUMNS_READ_IN_PLACE)
def test_frame_column_bind_allocates_nothing_per_row(duckdb_cursor, path, name):
    positions = np.arange(BIND_HEAP_ROWS, dtype="int64")
    column = pd.Series(FRAME_COLUMNS[name].build()).iloc[positions % 6].reset_index(drop=True)
    source = pd.DataFrame({"i": positions, "x": column}, copy=False)
    assert _bind_heap_peak(duckdb_cursor, source, path) < BIND_HEAP_ROWS // 4


DICT_HEAP_ARRAYS = [
    pytest.param(lambda positions: positions, id="int64"),
    pytest.param(lambda positions: positions % 2 == 0, id="bool"),
    pytest.param(lambda positions: np.array([f"o{index}" for index in positions], dtype=object), id="object"),
    pytest.param(lambda positions: positions.view("datetime64[ns]"), id="datetime64"),
]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("make_array", DICT_HEAP_ARRAYS)
def test_dict_array_bind_allocates_nothing_per_row(duckdb_cursor, path, make_array):
    positions = np.arange(BIND_HEAP_ROWS, dtype="int64")
    source = {"i": positions, "x": make_array(positions)}
    assert _bind_heap_peak(duckdb_cursor, source, path) < BIND_HEAP_ROWS // 4

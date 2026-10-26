"""The support for scaning over numpy arrays reuses many codes for pandas.
Therefore, we only test the new codes and exec paths.
"""  # noqa: D205

import os
import subprocess
import sys
import textwrap
from datetime import timedelta

import pytest
from conftest import PANDAS_GE_3

import duckdb
import numpy as np


class TestScanNumpy:
    def test_scan_numpy(self, duckdb_cursor):
        z = np.array([1, 2, 3])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1,), (2,), (3,)]

        z = np.array([[1, 2, 3], [4, 5, 6]])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1, 4), (2, 5), (3, 6)]

        z = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1, 4), (2, 5), (3, 6)]

        z = {"z": np.array([1, 2, 3]), "x": np.array([4, 5, 6])}
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1, 4), (2, 5), (3, 6)]

        z = np.array(["zzz", "xxx"])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [("zzz",), ("xxx",)]

        z = [np.array(["zzz", "xxx"]), np.array([1, 2])]
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [("zzz", 1), ("xxx", 2)]

        # test ndarray with dtype = object (python dict)
        z = []
        for i in range(3):
            z.append({str(3 - i): i})
        z = np.array(z)
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [
            ({"3": 0},),
            ({"2": 1},),
            ({"1": 2},),
        ]

        # test timedelta
        delta = timedelta(days=50, seconds=27, microseconds=10, milliseconds=29000, minutes=5, hours=8, weeks=2)
        delta2 = timedelta(days=5, seconds=27, microseconds=10, milliseconds=29000, minutes=5, hours=8, weeks=2)
        z = np.array([delta, delta2])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [
            (timedelta(days=64, seconds=29156, microseconds=10),),
            (timedelta(days=19, seconds=29156, microseconds=10),),
        ]

        # np.empty
        z = np.empty((3,))
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert len(res) == 3

        # empty list
        z = np.array([])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == []

        # np.nan
        z = np.array([np.nan])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(None,)]

        # dict of mixed types
        z = {"z": np.array([1, 2, 3]), "x": np.array(["z", "x", "c"])}
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1, "z"), (2, "x"), (3, "c")]

        # list of mixed types
        z = [np.array([1, 2, 3]), np.array(["z", "x", "c"])]
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1, "z"), (2, "x"), (3, "c")]

        # currently unsupported formats, will throw duckdb.InvalidInputException

        # list of arrays with different length
        z = [np.array([1, 2]), np.array([3])]
        with pytest.raises(duckdb.InvalidInputException):
            duckdb_cursor.sql("select * from z")

        # dict of ndarrays of different length
        z = {"z": np.array([1, 2]), "x": np.array([3])}
        with pytest.raises(duckdb.InvalidInputException):
            duckdb_cursor.sql("select * from z")

        # high dimensional tensors
        z = np.array([[[1, 2]]])
        with pytest.raises(duckdb.InvalidInputException):
            duckdb_cursor.sql("select * from z")

        # list of ndarrys with len(shape) > 1
        z = [np.array([[1, 2], [3, 4]])]
        with pytest.raises(duckdb.InvalidInputException):
            duckdb_cursor.sql("select * from z")

        # dict of ndarrays with len(shape) > 1
        z = {"x": np.array([[1, 2], [3, 4]])}
        with pytest.raises(duckdb.InvalidInputException):
            duckdb_cursor.sql("select * from z")


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(None, id="plain"),
        pytest.param([(0, 0), (0, 0), (0, 0)], id="all_false"),
        pytest.param([(0, 0), (0, 1), (0, 0)], id="partial"),
    ],
)
def test_masked_structured_array_fails_like_the_plain_array(duckdb_cursor, mask):
    source = np.zeros(3, dtype=[("a", "i4"), ("b", "i4")])
    if mask is not None:
        source = np.ma.masked_array(source, mask=mask)
    with pytest.raises(duckdb.NotImplementedException, match="not recognized"):
        duckdb_cursor.sql("SELECT * FROM source")


@pytest.mark.parametrize("route", ["register", "variable"])
@pytest.mark.parametrize("length", [2, 60])
def test_first_column_mask_of_another_length_is_refused(duckdb_cursor, route, length):
    source = np.ma.masked_array(np.arange(50, dtype="int64"), mask=np.zeros(50, dtype=bool))
    source._mask = np.zeros(length, dtype=bool)
    message = f"Column 'column0' has {length} rows where its data has 50"
    if route == "register":
        with pytest.raises(duckdb.InvalidInputException, match=message):
            duckdb_cursor.register("registered", source)
    else:
        with pytest.raises(duckdb.InvalidInputException, match=message):
            duckdb_cursor.sql("SELECT * FROM source")


@pytest.mark.skipif(
    np.lib.NumpyVersion(np.__version__) < "2.0.0", reason="numpy below 2 imports numpy.ma together with numpy"
)
def test_mask_of_a_column_that_imports_numpy_ma_when_read():
    script = textwrap.dedent(
        """
        import sys

        import duckdb
        import numpy as np

        assert "numpy.ma" not in sys.modules


        class ImportsOnRead(dict):
            def __getitem__(self, key):
                if key == "masked":
                    import numpy.ma

                    return numpy.ma.masked_array(np.array([1, 2, 3]), mask=[False, True, False])
                return super().__getitem__(key)


        source = ImportsOnRead(plain=np.array([1, 2, 3]), masked=np.zeros(3, dtype="int64"))
        print(duckdb.connect().execute("SELECT * FROM source").fetchall())
        """
    )
    # An editable install that rebuilds on import otherwise prints its cmake command into the child's stdout.
    quiet = {**os.environ, "SKBUILD_EDITABLE_VERBOSE": "0"}
    child = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=120, check=False, env=quiet
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "[(1, 1), (2, None), (3, 3)]"


@pytest.mark.parametrize("query", ["SELECT k FROM {}", "SELECT * FROM {}", "DESCRIBE SELECT * FROM {}"])
def test_object_field_of_a_packed_structured_array(duckdb_cursor, query):
    packed = np.zeros(10, dtype=[("i", "i4"), ("o", "O")])
    packed["o"] = [7, None] * 5
    source = {"k": np.arange(10, dtype="int64"), "x": packed["o"]}
    plain = {"k": source["k"], "x": np.ascontiguousarray(packed["o"])}  # noqa: F841
    expected = duckdb_cursor.execute(query.format("plain")).fetchall()
    assert duckdb_cursor.execute(query.format("source")).fetchall() == expected


@pytest.mark.skipif(not PANDAS_GE_3, reason="before pandas 3 assigning a column copies its values")
def test_object_field_of_a_packed_structured_array_in_a_frame(duckdb_cursor):
    pd = pytest.importorskip("pandas")
    packed = np.zeros(10, dtype=[("i", "i4"), ("o", "O")])
    packed["o"] = [7, None] * 5
    frame = pd.DataFrame({"k": np.arange(10, dtype="int64")})
    frame["x"] = pd.Series(packed["o"], copy=False)
    assert frame["x"].to_numpy().strides == (12,)
    assert duckdb_cursor.execute("SELECT k FROM frame").fetchall() == [(i,) for i in range(10)]
    assert duckdb_cursor.execute("SELECT x FROM frame").fetchall() == [(7,), (None,)] * 5


@pytest.mark.parametrize("dtype", ["int64", "float64", "datetime64[us]"])
def test_numeric_field_of_a_packed_structured_array(duckdb_cursor, dtype):
    packed = np.zeros(4, dtype=[("a", dtype), ("b", "i4")])
    packed["a"] = np.arange(1, 5).astype(dtype)
    packed["b"] = [10, 20, 30, 40]
    source = {"a": packed["a"], "b": packed["b"]}  # noqa: F841
    plain = {"a": np.ascontiguousarray(packed["a"]), "b": np.ascontiguousarray(packed["b"])}  # noqa: F841
    expected = duckdb_cursor.execute("SELECT * FROM plain").fetchall()
    assert duckdb_cursor.execute("SELECT * FROM source").fetchall() == expected


def test_unaligned_contiguous_column(duckdb_cursor):
    unaligned = np.frombuffer(bytes(1) + np.arange(5, dtype="int64").tobytes(), dtype="int64", offset=1)
    assert unaligned.flags.c_contiguous
    assert not unaligned.flags.aligned
    assert duckdb_cursor.execute("SELECT * FROM unaligned").fetchall() == [(i,) for i in range(5)]


class _ZeroItemsize(np.ndarray):
    @property
    def itemsize(self):
        return 0


def test_array_subclass_overriding_itemsize(duckdb_cursor):
    source = {"a": np.arange(5, dtype="int64").view(_ZeroItemsize)}  # noqa: F841
    assert duckdb_cursor.execute("SELECT * FROM source").fetchall() == [(i,) for i in range(5)]


def _mask_holding_the_byte_two() -> np.ndarray:
    return np.array([0, 2, 0], dtype="uint8").view(np.bool_)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.array([1, 2, 3], dtype=object), id="object"),
        pytest.param(np.array([1, 2, 3], dtype="int64"), id="int64"),
        pytest.param(np.array(["1", "2", "3"]), id="str"),
        pytest.param(np.array([1, 2, 3], dtype="timedelta64[us]"), id="timedelta"),
    ],
)
def test_mask_byte_other_than_one_masks_its_row(duckdb_cursor, values):
    source = {"x": np.ma.masked_array(values, mask=_mask_holding_the_byte_two())}  # noqa: F841
    expected = [(None,) if row == 1 else (value,) for row, value in enumerate(values.tolist())]
    assert duckdb_cursor.execute("SELECT x FROM source").fetchall() == expected


def test_nullable_mask_byte_other_than_one_masks_its_row(duckdb_cursor):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": pd.arrays.IntegerArray(np.array([1, 2, 3]), _mask_holding_the_byte_two())})  # noqa: F841
    assert duckdb_cursor.execute("SELECT x FROM df").fetchall() == [(1,), (None,), (3,)]


# Masked rows on each side of a word, a 64 byte block, a vector and a partition boundary, and a word of masked bytes.
_WORD_EDGE_ROWS = [0, 7, 8, 9, 15, 16, 31, 32, 33, *range(40, 48), 50, 52, 63, 64, 65, 127, 128]
_MASKED_ROW_CANDIDATES = [*_WORD_EDGE_ROWS, 2047, 2048, 2049, 102_399, 102_400, 102_401]
_MASK_BYTES = [1, 2, 0x7F, 0x80, 0xFF]
_LONG_MASK_ROWS = 2 * 102_400 + 2048 + 13


def _mask_slice_with_bytes(rows: int, start: int) -> np.ndarray:
    hidden = [row for row in _MASKED_ROW_CANDIDATES if row < rows - 3] + [rows - 3, rows - 1]
    raw = np.zeros(start + rows, dtype=np.uint8)
    raw[[start + row for row in hidden]] = [_MASK_BYTES[i % len(_MASK_BYTES)] for i in range(len(hidden))]
    return raw.view(np.bool_)[start:]


# Each column holds a value derived from its row number, which the SQL check recomputes; an object column holds a
# value of another type on each masked row, which it reads right only by skipping it.
_EDGE_MASK_COLUMNS = {
    "int64": (lambda ints: ints, None, "v = i"),
    "bool": (lambda ints: ints % 2 == 0, None, "v = (i % 2 = 0)"),
    "float64": (lambda ints: ints + 0.5, None, "v = i + 0.5"),
    "datetime64_us": (lambda ints: ints.view("datetime64[us]"), None, "v = make_timestamp(i)"),
    "timedelta64_us": (lambda ints: ints.view("timedelta64[us]"), None, "v = to_microseconds(i)"),
    "object_str": (lambda ints: np.array([f"s{i}" for i in ints.tolist()], dtype=object), 7, "v = 's' || i"),
    "object_int": (lambda ints: np.array(ints.tolist(), dtype=object), "hidden", "v = i"),
}


@pytest.mark.parametrize("start", [0, 1, 5], ids=["aligned", "odd_address", "address_plus_5"])
@pytest.mark.parametrize("rows", [5, 13, 40, 100, _LONG_MASK_ROWS])
@pytest.mark.parametrize("column_id", list(_EDGE_MASK_COLUMNS))
def test_mask_bytes_at_word_vector_and_partition_edges_match_numpy(duckdb_cursor, column_id, rows, start):
    make, other_type, check = _EDGE_MASK_COLUMNS[column_id]
    ints = np.arange(rows, dtype="int64")
    mask = _mask_slice_with_bytes(rows, start)
    data = make(ints)
    if other_type is not None:
        data[mask] = other_type
    source = np.ma.masked_array(data, mask=mask)
    assert np.ma.getmask(source).ctypes.data == mask.ctypes.data
    duckdb_cursor.execute("SET threads=4")
    duckdb_cursor.register("src", {"i": ints, "v": source})
    query = f"SELECT list(i ORDER BY i) FILTER (WHERE v IS NULL), count(*) FILTER (WHERE {check}) FROM src"
    hidden = np.flatnonzero(mask.view(np.uint8)).tolist()
    assert duckdb_cursor.execute(query).fetchall() == [(hidden, rows - len(hidden))]


@pytest.mark.parametrize("start", [0, 1], ids=["aligned", "odd_address"])
def test_nullable_mask_bytes_at_word_vector_and_partition_edges_match_numpy(duckdb_cursor, start):
    pd = pytest.importorskip("pandas")
    rows = _LONG_MASK_ROWS
    ints = np.arange(rows, dtype="int64")
    mask = _mask_slice_with_bytes(rows, start)
    df = pd.DataFrame({"i": ints, "v": pd.arrays.IntegerArray(ints.copy(), mask)}, copy=False)
    assert df["v"].array._mask.ctypes.data == mask.ctypes.data
    duckdb_cursor.execute("SET threads=4")
    duckdb_cursor.register("src", df)
    query = "SELECT list(i ORDER BY i) FILTER (WHERE v IS NULL), count(*) FILTER (WHERE v = i) FROM src"
    hidden = np.flatnonzero(mask.view(np.uint8)).tolist()
    assert duckdb_cursor.execute(query).fetchall() == [(hidden, rows - len(hidden))]


def _dense_mask_slice(rows: int, start: int) -> np.ndarray:
    raw = np.zeros(start + rows, dtype=np.uint8)
    hidden = raw[start:]
    hidden[64:128] = 0xFF
    hidden[128:192] = 1
    hidden[[128, 191]] = 0
    if rows > 2048 + 64:
        hidden[2048 - 64 : 2048 + 64] = 2
    hidden[rows - (rows % 64 or 64) :] = 0x80
    return raw.view(np.bool_)[start:]


# Whole 64 row blocks masked, a block masked but for its first and last row, and a masked last block.
@pytest.mark.parametrize("start", [0, 1], ids=["aligned", "odd_address"])
@pytest.mark.parametrize("rows", [200, 2048 + 100, _LONG_MASK_ROWS])
@pytest.mark.parametrize("column_id", list(_EDGE_MASK_COLUMNS))
def test_mask_hiding_whole_blocks_matches_numpy(duckdb_cursor, column_id, rows, start):
    make, other_type, check = _EDGE_MASK_COLUMNS[column_id]
    ints = np.arange(rows, dtype="int64")
    mask = _dense_mask_slice(rows, start)
    data = make(ints)
    if other_type is not None:
        data[mask] = other_type
    duckdb_cursor.execute("SET threads=4")
    duckdb_cursor.register("src", {"i": ints, "v": np.ma.masked_array(data, mask=mask)})
    query = f"SELECT list(i ORDER BY i) FILTER (WHERE v IS NULL), count(*) FILTER (WHERE {check}) FROM src"
    hidden = np.flatnonzero(mask.view(np.uint8)).tolist()
    assert duckdb_cursor.execute(query).fetchall() == [(hidden, rows - len(hidden))]


@pytest.mark.parametrize("make_mask", [_mask_slice_with_bytes, _dense_mask_slice], ids=["sparse", "dense"])
def test_masked_struct_object_column_hides_its_fields(duckdb_cursor, make_mask):
    rows = 2048 + 100
    ints = np.arange(rows, dtype="int64")
    mask = make_mask(rows, 1)
    data = np.array([{"k": i} for i in ints.tolist()], dtype=object)
    data[mask] = "hidden"
    duckdb_cursor.register("src", {"i": ints, "v": np.ma.masked_array(data, mask=mask)})
    query = (
        "SELECT list(i ORDER BY i) FILTER (WHERE v IS NULL), list(i ORDER BY i) FILTER (WHERE v.k IS NULL),"
        " count(*) FILTER (WHERE v.k = i) FROM src"
    )
    hidden = np.flatnonzero(mask.view(np.uint8)).tolist()
    assert duckdb_cursor.execute(query).fetchall() == [(hidden, hidden, rows - len(hidden))]


# NaT rows on each side of a 64 row block and of the vector edge, with no mask, a sparse mask and a dense one; each
# masked row that is not NaT hides the largest int64, which overflows in the conversion from seconds.
@pytest.mark.parametrize("unit", ["us", "s"])
@pytest.mark.parametrize(
    "make_mask", [None, _mask_slice_with_bytes, _dense_mask_slice], ids=["no_mask", "sparse", "dense"]
)
def test_timedelta_nat_next_to_masked_rows_matches_numpy(duckdb_cursor, unit, make_mask):
    rows = 2048 + 100
    ints = np.arange(rows, dtype="int64")
    values = ints.copy()
    values[[0, 1, 62, 63, 64, 127, 128, 2047, 2048, rows - 1]] = np.iinfo(np.int64).min
    not_a_time = np.isnat(values.view(f"timedelta64[{unit}]"))
    if make_mask is None:
        mask = np.zeros(rows, dtype=bool)
        source = values.view(f"timedelta64[{unit}]")
    else:
        mask = make_mask(rows, 1)
        values[mask & ~not_a_time] = np.iinfo(np.int64).max
        source = np.ma.masked_array(values.view(f"timedelta64[{unit}]"), mask=mask)
    duckdb_cursor.register("src", {"i": ints, "v": source})
    to_interval = "to_microseconds" if unit == "us" else "to_seconds"
    query = f"SELECT list(i ORDER BY i) FILTER (WHERE v IS NULL), count(*) FILTER (WHERE v = {to_interval}(i)) FROM src"
    hidden = np.flatnonzero(not_a_time | mask).tolist()
    assert duckdb_cursor.execute(query).fetchall() == [(hidden, rows - len(hidden))]

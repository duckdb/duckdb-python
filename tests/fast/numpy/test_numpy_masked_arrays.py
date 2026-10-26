"""A numpy masked array reads its masked elements as NULL, and they never shape the column or get converted."""

import ctypes
import json
import subprocess
import sys
from datetime import datetime, timedelta
from typing import NamedTuple

import pytest

import duckdb
import numpy as np
import pandas as pd

CHILD_TIMEOUT = 60


def _array(dtype, cells):
    if dtype is object:
        # Filled cell by cell so that list, dict and array cells stay single objects.
        array = np.empty(len(cells), dtype=object)
        for position, cell in enumerate(cells):
            array[position] = cell
        return array
    return np.array(cells, dtype=dtype)


def _masked(dtype, cells, mask, **kwargs):
    return np.ma.masked_array(_array(dtype, cells), mask=np.array(mask, dtype=bool), **kwargs)


def _hide(values, mask):
    return [None if hidden else value for value, hidden in zip(values, mask, strict=False)]


def _rows(values):
    return [(value,) for value in values]


def _scan(con, route, source, query="SELECT * FROM src"):
    if route == "register":
        con.register("src", source)
    else:
        src = source  # noqa: F841
    return con.sql(query)


def _types(relation):
    return [str(column_type) for column_type in relation.types]


class DtypeCase(NamedTuple):
    dtype: object
    cells: list
    read_back: list
    sql_type: str


def _case(dtype, cells, sql_type, read_back=None):
    return DtypeCase(dtype, cells, cells if read_back is None else read_back, sql_type)


DTYPE_BY_ID = {
    "bool": _case("bool", [True, False, True], "BOOLEAN"),
    **{
        name: _case(name, [1, 2, 3], sql_type)
        for name, sql_type in (
            ("int8", "TINYINT"),
            ("int16", "SMALLINT"),
            ("int32", "INTEGER"),
            ("int64", "BIGINT"),
            ("uint8", "UTINYINT"),
            ("uint16", "USMALLINT"),
            ("uint32", "UINTEGER"),
            ("uint64", "UBIGINT"),
        )
    },
    **{
        name: _case(name, [1.5, 2.5, 3.5], sql_type)
        for name, sql_type in (("float16", "FLOAT"), ("float32", "FLOAT"), ("float64", "DOUBLE"))
    },
    **{
        f"datetime64_{unit}": _case(f"datetime64[{unit}]", [datetime(2020, 1, day) for day in (1, 2, 3)], sql_type)
        for unit, sql_type in (
            ("ns", "TIMESTAMP_NS"),
            ("us", "TIMESTAMP"),
            ("ms", "TIMESTAMP_MS"),
            ("s", "TIMESTAMP_S"),
        )
    },
    **{
        f"timedelta64_{unit}": _case(
            f"timedelta64[{unit}]", [timedelta(**{name: size}) for size in (1, 2, 3)], "INTERVAL"
        )
        # Nanosecond intervals are read in whole microseconds, so nanosecond cells that differ by less read alike.
        for unit, name in (("ns", "microseconds"), ("us", "microseconds"), ("ms", "milliseconds"), ("s", "seconds"))
    },
    "str": _case("U", ["a", "b", "c"], "ENUM('a', 'b', 'c')"),
    "object_str": _case(object, ["a", "b", "c"], "VARCHAR"),
    "object_int": _case(object, [1, 2, 3], "INTEGER"),
    "object_dict": _case(object, [{"a": 1}, {"a": 2}, {"a": 3}], "STRUCT(a INTEGER)"),
    "object_list": _case(object, [[1], [2], [3]], "INTEGER[]"),
    "object_ndarray": _case(
        object, [np.array([value], dtype="int64") for value in (1, 2, 3)], "BIGINT[]", read_back=[[1], [2], [3]]
    ),
}
DTYPE_IDS = list(DTYPE_BY_ID)
# One dtype per way the scan reads a column.
BRANCH_IDS = [
    "bool",
    "int64",
    "float64",
    "datetime64_us",
    "timedelta64_us",
    "str",
    "object_str",
    "object_int",
    "object_dict",
]


@pytest.mark.parametrize("route", ["register", "variable"])
@pytest.mark.parametrize("dtype_id", DTYPE_IDS)
def test_masked_element_reads_as_null(duckdb_cursor, dtype_id, route):
    dtype, cells, read_back, _ = DTYPE_BY_ID[dtype_id]
    source = _masked(dtype, cells, [0, 1, 0])
    relation = _scan(duckdb_cursor, route, source)
    assert relation.fetchall() == _rows([read_back[0], None, read_back[2]])


@pytest.mark.parametrize("route", ["register", "variable"])
@pytest.mark.parametrize("dtype_id", [dtype_id for dtype_id in DTYPE_IDS if dtype_id != "str"])
def test_masked_array_type_matches_plain(duckdb_cursor, dtype_id, route):
    dtype, cells, _, sql_type = DTYPE_BY_ID[dtype_id]
    source = _masked(dtype, cells, [0, 1, 0])
    assert _types(_scan(duckdb_cursor, route, source)) == [sql_type]


@pytest.mark.parametrize(
    ("cells", "expected_type", "expected"),
    [
        pytest.param(["x", "secret", "x"], "ENUM('x')", ["x", None, "x"], id="hidden_in_middle"),
        pytest.param(["a", "zz", "b"], "ENUM('a', 'b')", ["a", None, "b"], id="hidden_sorts_last"),
        pytest.param(["b", "a", "c"], "ENUM('b', 'c')", ["b", None, "c"], id="hidden_sorts_first"),
    ],
)
def test_masked_strings_enum_holds_unmasked_values_only(duckdb_cursor, cells, expected_type, expected):
    source = _masked("U", cells, [0, 1, 0])
    relation = _scan(duckdb_cursor, "variable", source)
    assert (_types(relation), relation.fetchall()) == ([expected_type], _rows(expected))


@pytest.mark.parametrize(
    ("cells", "expected_type"),
    [
        pytest.param([1, 1.5, 3], "INTEGER", id="scalar"),
        pytest.param([1, "hidden", 3], "INTEGER", id="string"),
        pytest.param([1, np.array([1.5]), 3], "INTEGER", id="ndarray_cell"),
        pytest.param(
            [np.array([1, 2], dtype="int64"), np.array([1.5]), np.array([3], dtype="int64")],
            "BIGINT[]",
            id="ndarray_of_other_dtype",
        ),
    ],
)
def test_object_type_ignores_masked_values(duckdb_cursor, cells, expected_type):
    source = _masked(object, cells, [0, 1, 0])
    assert _types(_scan(duckdb_cursor, "variable", source)) == [expected_type]


@pytest.mark.parametrize(
    ("dtype", "message"),
    [
        pytest.param("complex128", "Data type 'complex128' not recognized", id="complex128"),
        pytest.param("datetime64[D]", r"Data type 'datetime64\[D\]' not recognized", id="datetime64_D"),
        pytest.param("S1", r"Data type '\|S1' not recognized", id="bytes"),
        pytest.param([("a", "i4")], r"Data type '\[\('a', '<i4'\)\]' not recognized", id="structured"),
    ],
)
@pytest.mark.parametrize("mask", [[0, 0, 0], [0, 1, 0]], ids=["unmasked", "masked"])
def test_unsupported_dtype_fails_with_or_without_a_mask(duckdb_cursor, dtype, message, mask):
    source = np.ma.masked_array(np.zeros(3, dtype=dtype), mask=np.array(mask, dtype=bool))
    with pytest.raises(duckdb.NotImplementedException, match=message):
        _scan(duckdb_cursor, "variable", source).fetchall()


@pytest.mark.parametrize("route", ["register", "variable"])
@pytest.mark.parametrize("container", ["dict", "list"])
def test_masked_columns_in_dict_and_list(duckdb_cursor, container, route):
    ints = _masked("int64", [1, 2, 3], [0, 1, 0])
    floats = _masked("float64", [4.0, 5.0, 6.0], [1, 0, 0])
    source = {"a": ints, "b": floats} if container == "dict" else [ints, floats]
    relation = _scan(duckdb_cursor, route, source)
    assert relation.fetchall() == [(1, None), (None, 5.0), (3, 6.0)]


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        pytest.param(
            lambda: np.ma.masked_equal(np.array([1, -1, 3], dtype="int64"), -1), [1, None, 3], id="masked_equal"
        ),
        pytest.param(
            lambda: np.ma.masked_where(np.array([False, True, False]), np.array(["x", "y", "z"])),
            ["x", None, "z"],
            id="masked_where_str",
        ),
        pytest.param(
            lambda: np.ma.masked_invalid(np.array([1.5, np.nan, np.nan, np.inf])),
            [1.5, None, None, None],
            id="masked_invalid",
        ),
        pytest.param(lambda: _masked("int64", [1, 2, 3], [0, 1, 0], hard_mask=True), [1, None, 3], id="hard_mask"),
        pytest.param(
            lambda: type("UserMaskedArray", (np.ma.MaskedArray,), {})(
                np.array([1, 2, 3], dtype="int64"), mask=[False, True, False]
            ),
            [1, None, 3],
            id="subclass",
        ),
    ],
)
def test_mask_helpers_mark_null(duckdb_cursor, make, expected):
    source = make()
    assert _scan(duckdb_cursor, "variable", source).fetchall() == _rows(expected)


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        pytest.param(["a", "b", "c"], ["a", None, "c"], id="object_str"),
        pytest.param([1, 2, 3], ["1", None, "3"], id="object_int"),
    ],
)
def test_masked_object_with_analyzer_off(duckdb_cursor, cells, expected):
    duckdb_cursor.execute("SET pandas_analyze_sample=0")
    source = _masked(object, cells, [0, 1, 0])
    assert _scan(duckdb_cursor, "variable", source).fetchall() == _rows(expected)


def test_masked_struct_cell_is_null_with_null_children(duckdb_cursor):
    source = _masked(object, [{"a": 1}, {"a": 2}, {"a": 3}], [0, 1, 0])
    relation = _scan(duckdb_cursor, "variable", source, "SELECT column0, column0.a FROM src")
    assert relation.fetchall() == [({"a": 1}, 1), (None, None), ({"a": 3}, 3)]


def _view_with_hidden_extreme(dtype):
    return np.ma.masked_array(np.array([0, 2**62, 1], dtype="int64").view(dtype), mask=[False, True, False])


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        pytest.param(
            lambda: _masked(object, [1, "hidden", 3], [0, 1, 0]),
            [1, None, 3],
            id="object_wrong_type",
        ),
        pytest.param(
            lambda: _masked(object, [1, "hidden", 3], [0, 1, 0], hard_mask=True),
            [1, None, 3],
            id="object_wrong_type_hard_mask",
        ),
        pytest.param(lambda: _masked(object, [1, 1.5, 3], [0, 1, 0]), [1, None, 3], id="object_fraction"),
        pytest.param(
            lambda: _masked(object, ["a", "\ud800", "c"], [0, 1, 0]),
            ["a", None, "c"],
            id="object_unencodable_str",
        ),
        pytest.param(
            lambda: _masked(object, [{"a": 1}, {"a": "hidden"}, {"a": 3}], [0, 1, 0]),
            [{"a": 1}, None, {"a": 3}],
            id="object_struct_wrong_leaf",
        ),
        pytest.param(
            lambda: _masked(object, [{"a": 1}, {"b": 2}, {"a": 3}], [0, 1, 0]),
            [{"a": 1}, None, {"a": 3}],
            id="object_struct_other_key",
        ),
        pytest.param(
            lambda: _masked(object, [[1], ["x"], [3]], [0, 1, 0]),
            [[1], None, [3]],
            id="object_list_wrong_element",
        ),
        pytest.param(
            lambda: _view_with_hidden_extreme("datetime64[s]"),
            [datetime(1970, 1, 1), None, datetime(1970, 1, 1, 0, 0, 1)],
            id="datetime64_s_out_of_range",
        ),
        pytest.param(
            lambda: _view_with_hidden_extreme("datetime64[ms]"),
            [datetime(1970, 1, 1), None, datetime(1970, 1, 1, 0, 0, 0, 1000)],
            id="datetime64_ms_out_of_range",
        ),
        pytest.param(
            lambda: _view_with_hidden_extreme("timedelta64[s]"),
            [timedelta(0), None, timedelta(seconds=1)],
            id="timedelta64_s_overflow",
        ),
        pytest.param(
            lambda: _view_with_hidden_extreme("timedelta64[ms]"),
            [timedelta(0), None, timedelta(milliseconds=1)],
            id="timedelta64_ms_overflow",
        ),
    ],
)
def test_masked_value_is_never_converted(duckdb_cursor, make, expected):
    source = make()
    assert _scan(duckdb_cursor, "variable", source).fetchall() == _rows(expected)


_FETCHNUMPY_CHILD = """
import json
import sys

try:
    import resource

    # A crash must end the child at once instead of waiting for a core dump.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass

import duckdb
import numpy as np


def as_python(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.datetime64):
        # Nanosecond datetimes turn into plain integers through item().
        value = value.astype("datetime64[us]")
    value = value.item() if isinstance(value, np.generic) else value
    # Some fetchnumpy columns, ENUM among them, mark NULL with NaN in a plain object array.
    return None if isinstance(value, float) and value != value else value


con = duckdb.connect()
fetched = con.sql(f"SELECT x FROM (VALUES ({sys.argv[1]}), (NULL)) t(x)").fetchnumpy()
column = fetched["x"]
# Only the first value comes from the result, which reads back in its numpy type; the NULL row is fixed by the query.
first = None if sys.argv[1] == "NULL" else as_python(column[0])
expected = [(first, first is None), (None, True)]
if sys.argv[2] == "data":
    fetched = {"x": np.ma.getdata(column)}
rows = con.sql("SELECT x, x IS NULL FROM fetched").fetchall()
print("detail", repr(rows), repr(expected), flush=True)
print("outcome", json.dumps(rows == expected), flush=True)
"""


def _fetchnumpy_child(literal, read):
    try:
        child = subprocess.run(
            [sys.executable, "-I", "-c", _FETCHNUMPY_CHILD, literal, read],
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as hang:
        output = "".join(out.decode() if isinstance(out, bytes) else out or "" for out in (hang.stdout, hang.stderr))
        pytest.fail(f"the child did not finish within {CHILD_TIMEOUT} seconds:\n{output}")
    outcomes = [line.removeprefix("outcome ") for line in child.stdout.splitlines() if line.startswith("outcome ")]
    return child.returncode, [json.loads(outcome) for outcome in outcomes], child.stdout + child.stderr


OBJECT_RESULTS = {
    "VARCHAR": "'a'",
    "TIME": "TIME '01:02:03'",
    "TIME_NS": "'01:02:03.000000001'::TIME_NS",
    "BLOB": "'\\x01'::BLOB",
    "BIT": "'101'::BIT",
    "UUID": "'12345678-1234-5678-1234-567812345678'::UUID",
    "NULL": "NULL",
}
VALUE_RESULTS = {
    "BIGINT": "1::BIGINT",
    "INTEGER": "1::INTEGER",
    "UBIGINT": "1::UBIGINT",
    "DOUBLE": "1.5::DOUBLE",
    "FLOAT": "1.5::FLOAT",
    "DECIMAL": "1.5::DECIMAL(4,1)",
    "HUGEINT": "1::HUGEINT",
    "BOOLEAN": "true",
    "DATE": "DATE '2020-01-01'",
    "TIMESTAMP": "TIMESTAMP '2020-01-01 01:02:03'",
    "TIMESTAMP_NS": "'2020-01-01 01:02:03.000000001'::TIMESTAMP_NS",
    "TIMESTAMPTZ": "TIMESTAMPTZ '2020-01-01 01:02:03+00'",
    "INTERVAL": "INTERVAL 3 SECOND",
}
NESTED_RESULTS = {
    "ENUM": "'x'::ENUM('x', 'y')",
    "LIST": "[1, 2]",
    "ARRAY": "[1, 2]::INTEGER[2]",
    "STRUCT": "{'a': 1}",
    "MAP": "MAP {'k': 1}",
    "UNION": "1::UNION(i INTEGER, s VARCHAR)",
}
ROUND_TRIPS = [
    *(pytest.param(literal, id=name) for name, literal in VALUE_RESULTS.items()),
    *(pytest.param(literal, id=name) for name, literal in OBJECT_RESULTS.items()),
    *(pytest.param(literal, id=name) for name, literal in NESTED_RESULTS.items()),
]
OBJECT_ROUND_TRIPS = [pytest.param(literal, id=name) for name, literal in OBJECT_RESULTS.items()]


@pytest.mark.timeout(CHILD_TIMEOUT * 2)
@pytest.mark.parametrize("literal", ROUND_TRIPS)
def test_fetchnumpy_result_reads_back(literal):
    returncode, outcomes, output = _fetchnumpy_child(literal, "masked")
    assert (returncode, outcomes) == (0, [True]), output


@pytest.mark.timeout(CHILD_TIMEOUT * 2)
@pytest.mark.parametrize("literal", OBJECT_ROUND_TRIPS)
def test_fetchnumpy_object_data_without_its_mask_reads_null(literal):
    returncode, outcomes, output = _fetchnumpy_child(literal, "data")
    assert (returncode, outcomes) == (0, [True]), output


@pytest.mark.parametrize("literal", OBJECT_ROUND_TRIPS)
def test_fetchnumpy_null_object_slot_holds_none(duckdb_cursor, literal):
    column = duckdb_cursor.sql(f"SELECT x FROM (VALUES ({literal}), (NULL)) t(x)").fetchnumpy()["x"]
    data = np.ma.getdata(column)
    null_rows = [0, 1] if literal == "NULL" else [1]
    # The stored pointer is read rather than the element, because numpy hands out None for an empty slot too.
    slots = [ctypes.c_void_p.from_address(data.ctypes.data + row * data.strides[0]).value for row in null_rows]
    assert slots == [id(None)] * len(null_rows)


MISSING_MARKERS = [
    pytest.param("float64", [1.5, np.nan, 2.5, np.inf], [1.5, None, 2.5, float("inf")], id="float64_nan"),
    pytest.param("float32", [1.5, np.nan, 2.5, np.inf], [1.5, None, 2.5, float("inf")], id="float32_nan"),
    pytest.param(
        "datetime64[us]",
        ["2020-01-01", "NaT", "2020-01-03", "2020-01-04"],
        [datetime(2020, 1, 1), None, datetime(2020, 1, 3), datetime(2020, 1, 4)],
        id="datetime64_nat",
    ),
    pytest.param(
        "timedelta64[us]",
        [1, "NaT", 3, 4],
        [timedelta(microseconds=1), None, timedelta(microseconds=3), timedelta(microseconds=4)],
        id="timedelta64_nat",
    ),
    pytest.param(object, ["a", None, "c", "d"], ["a", None, "c", "d"], id="object_none"),
    pytest.param(object, ["a", np.nan, "c", "d"], ["a", None, "c", "d"], id="object_nan"),
    pytest.param(object, [1, pd.NA, 3, 4], [1, None, 3, 4], id="object_pandas_na"),
]


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param([0, 0, 1, 0], id="other_element_masked"),
        pytest.param([0, 0, 0, 0], id="all_false"),
    ],
)
@pytest.mark.parametrize(("dtype", "cells", "read_back"), MISSING_MARKERS)
def test_unmasked_missing_values_stay_null(duckdb_cursor, dtype, cells, read_back, mask):
    source = _masked(dtype, cells, mask)
    assert _scan(duckdb_cursor, "variable", source).fetchall() == _rows(_hide(read_back, mask))


UNSET_OR_FALSE_MASKS = {
    "nomask": {},
    "all_false": {"mask": np.zeros(3, dtype=bool)},
    "mask_false": {"mask": False},
    "hard": {"hard_mask": True},
}


@pytest.mark.parametrize("variant", list(UNSET_OR_FALSE_MASKS))
@pytest.mark.parametrize("dtype_id", BRANCH_IDS)
def test_unset_or_false_mask_reads_like_plain(duckdb_cursor, dtype_id, variant):
    dtype, cells, read_back, sql_type = DTYPE_BY_ID[dtype_id]
    source = np.ma.masked_array(_array(dtype, cells), **UNSET_OR_FALSE_MASKS[variant])
    relation = _scan(duckdb_cursor, "variable", source)
    assert (_types(relation), relation.fetchall()) == ([sql_type], _rows(read_back))


FULLY_MASKED = [
    *(pytest.param(case.dtype, case.cells, case.sql_type, id=dtype_id) for dtype_id, case in DTYPE_BY_ID.items()),
    pytest.param(object, [1, "a", 2.5], None, id="object_mixed"),
]


@pytest.mark.parametrize(("dtype", "cells", "sql_type"), FULLY_MASKED)
def test_fully_masked_column(duckdb_cursor, dtype, cells, sql_type):
    # With no value left, an object column types as an all None one and a string column gets an empty dictionary.
    expected_type = "ENUM()" if dtype == "U" else '"NULL"' if dtype is object else sql_type
    source = np.ma.masked_array(_array(dtype, cells), mask=True)
    relation = _scan(duckdb_cursor, "variable", source)
    assert (_types(relation), relation.fetchall()) == ([expected_type], _rows([None] * len(cells)))


def test_empty_masked_array(duckdb_cursor):
    source = np.ma.masked_array(np.array([], dtype="int64"), mask=np.array([], dtype=bool))
    relation = _scan(duckdb_cursor, "variable", source)
    assert (_types(relation), relation.fetchall()) == (["BIGINT"], [])


def _ten_with_every_third_masked(kind):
    values = np.arange(10, dtype="int64")
    if kind == "int64":
        data = values
    elif kind == "object_int":
        data = _array(object, values.tolist())
    elif kind in ("datetime64[us]", "timedelta64[us]"):
        data = values.view(kind)
    else:
        data = np.array([f"v{value}" for value in values], dtype=kind)
    return np.ma.masked_array(data, mask=values % 3 == 0)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _ten_with_every_third_masked("int64")[::2], id="step_slice"),
        pytest.param(lambda: _ten_with_every_third_masked("int64")[3:], id="offset_slice"),
        pytest.param(lambda: _ten_with_every_third_masked("int64")[1::2], id="offset_step_slice"),
        pytest.param(lambda: _ten_with_every_third_masked(object)[::2], id="object_str_step_slice"),
        pytest.param(lambda: _ten_with_every_third_masked("object_int")[::2], id="object_int_step_slice"),
        pytest.param(lambda: _ten_with_every_third_masked("U2")[::2], id="str_step_slice"),
        pytest.param(lambda: _ten_with_every_third_masked("datetime64[us]")[::2], id="datetime64_us_step_slice"),
        pytest.param(lambda: _ten_with_every_third_masked("timedelta64[us]")[::2], id="timedelta64_us_step_slice"),
        pytest.param(
            lambda: np.ma.masked_array(np.arange(3, dtype="int64"), mask=np.array([True, False, False])[::-1]),
            id="reversed_mask",
        ),
        pytest.param(
            lambda: np.ma.masked_array(np.arange(3, dtype="int64"), mask=np.broadcast_to(np.array(True), 3)),
            id="broadcast_mask",
        ),
    ],
)
def test_mask_of_strided_view(duckdb_cursor, make):
    source = make()
    data = [value.item() if isinstance(value, np.generic) else value for value in np.ma.getdata(source)]
    expected = _rows(_hide(data, np.ma.getmaskarray(source).tolist()))
    assert _scan(duckdb_cursor, "variable", source).fetchall() == expected


# Each column holds a value derived from its row number, and the SQL check recomputes it from that number.
LONG_COLUMNS = {
    "bool": (lambda ints: ints % 2 == 0, "v = (i % 2 = 0)"),
    "int64": (lambda ints: ints, "v = i"),
    "float64": (lambda ints: ints + 0.5, "v = i + 0.5"),
    "datetime64_us": (lambda ints: ints.view("datetime64[us]"), "v = make_timestamp(i)"),
    "timedelta64_us": (lambda ints: ints.view("timedelta64[us]"), "v = to_microseconds(i)"),
    "str": (lambda ints: np.array([f"s{i}" for i in ints.tolist()]), "v::VARCHAR = 's' || i"),
    "object_str": (lambda ints: _array(object, [f"s{i}" for i in ints.tolist()]), "v = 's' || i"),
    "object_int": (lambda ints: _array(object, ints.tolist()), "v = i"),
    "object_dict": (lambda ints: _array(object, [{"a": i} for i in ints.tolist()]), "v.a = i"),
}
# A cell of another type on a masked row, which an object column reads right only by skipping it.
OTHER_TYPE_CELL = {"object_str": 7, "object_int": "hidden", "object_dict": "hidden"}


def _long_masked_column(column_id, rows, hidden_rows, other_type_row):
    make, _ = LONG_COLUMNS[column_id]
    data = make(np.arange(rows, dtype="int64"))
    if column_id in OTHER_TYPE_CELL:
        data[other_type_row] = OTHER_TYPE_CELL[column_id]
    mask = np.zeros(rows, dtype=bool)
    mask[hidden_rows] = True
    return {"i": np.arange(rows, dtype="int64"), "v": np.ma.masked_array(data, mask=mask)}


def _null_rows_and_matches(con, route, source, column_id):
    check = LONG_COLUMNS[column_id][1]
    query = f"SELECT list(i ORDER BY i) FILTER (WHERE v IS NULL), count(*) FILTER (WHERE {check}) FROM src"
    return _scan(con, route, source, query).fetchall()


@pytest.mark.parametrize("column_id", list(LONG_COLUMNS))
def test_mask_across_vector_boundaries(duckdb_cursor, column_id):
    rows = 5000
    # The analyzer samples every fifth row of 5000, so the cell of another type sits on a sampled row.
    other_type_row = 2050
    hidden_rows = [2047, 2048, other_type_row, rows - 1]
    source = _long_masked_column(column_id, rows, hidden_rows, other_type_row)
    expected = [(hidden_rows, rows - len(hidden_rows))]
    assert _null_rows_and_matches(duckdb_cursor, "variable", source, column_id) == expected


@pytest.mark.parametrize("column_id", ["int64", "object_str", "object_int"])
def test_mask_across_partition_boundaries(duckdb_cursor, column_id):
    rows = 300_000
    # The analyzer samples every 300th row of 300000, so the cell of another type sits on a sampled row.
    other_type_row = 204_900
    hidden_rows = [2047, 2048, 102_399, 102_400, 204_800, other_type_row, rows - 1]
    source = _long_masked_column(column_id, rows, hidden_rows, other_type_row)
    duckdb_cursor.execute("SET threads=8")
    expected = [(hidden_rows, rows - len(hidden_rows))]
    assert _null_rows_and_matches(duckdb_cursor, "register", source, column_id) == expected


def _like_plain_with_hidden(con, data, hidden):
    # Which row and column a 2d element lands in is not a mask rule, so the plain read of the data decides it.
    plain = np.ascontiguousarray(data)  # noqa: F841
    rows = con.sql("SELECT * FROM plain").fetchall()
    return [tuple(None if value in hidden else value for value in row) for row in rows]


def _two_dimensional(order):
    if order == "c_order":
        return np.ma.masked_array(np.array([[1, 2, 3], [4, 5, 6]], dtype="int64"), mask=[[0, 1, 0], [1, 0, 0]])
    data = np.asfortranarray(np.arange(6, dtype="int64").reshape(2, 3))
    return np.ma.masked_array(data, mask=np.asfortranarray(np.array([[0, 1, 0], [0, 0, 1]], dtype=bool)))


@pytest.mark.parametrize("route", ["register", "variable"])
@pytest.mark.parametrize("order", ["c_order", "fortran_order"])
def test_two_dimensional_mask_follows_its_element(duckdb_cursor, order, route):
    source = _two_dimensional(order)
    expected = _like_plain_with_hidden(duckdb_cursor, np.ma.getdata(source), set(source.data[source.mask].tolist()))
    assert _scan(duckdb_cursor, route, source).fetchall() == expected


@pytest.mark.parametrize(
    "container",
    ["1d_nomask", "1d_existing_mask", "dict", "list"],
)
def test_mask_change_after_register_shows_in_next_query(duckdb_cursor, container):
    column = np.ma.masked_array(np.array([1, 2, 3], dtype="int64"))
    if container == "1d_existing_mask":
        column = np.ma.masked_array(np.array([1, 2, 3], dtype="int64"), mask=np.zeros(3, dtype=bool))
    other = np.array([4, 5, 6], dtype="int64")
    source = {"dict": {"a": column, "b": other}, "list": [column, other]}.get(container, column)
    duckdb_cursor.register("src", source)
    duckdb_cursor.sql("SELECT * FROM src").fetchall()
    column[1] = np.ma.masked
    expected = [(1, 4), (None, 5), (3, 6)] if container in ("dict", "list") else [(1,), (None,), (3,)]
    assert duckdb_cursor.sql("SELECT * FROM src").fetchall() == expected


def _cells_except(rows, plain_rows, left_open):
    return [
        tuple(value for value, plain in zip(row, plain_row, strict=False) if plain not in left_open)
        for row, plain_row in zip(rows, plain_rows, strict=False)
    ]


@pytest.mark.parametrize(
    ("mask_at_registration", "hidden", "left_open"),
    [
        # A registered 2d array is split into rows once, so a mask added later to an array without one is not promised.
        pytest.param(False, set(), {2}, id="nomask_at_registration"),
        pytest.param(True, {2}, set(), id="mask_at_registration"),
    ],
)
def test_registered_2d_mask_change(duckdb_cursor, mask_at_registration, hidden, left_open):
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype="int64")
    mask = np.zeros((2, 3), dtype=bool) if mask_at_registration else np.ma.nomask
    source = np.ma.masked_array(data, mask=mask)
    duckdb_cursor.register("src", source)
    duckdb_cursor.sql("SELECT * FROM src").fetchall()
    source[0, 1] = np.ma.masked
    source[1, 0] = 40
    plain_rows = _like_plain_with_hidden(duckdb_cursor, np.ma.getdata(source), set())
    hidden_rows = [tuple(None if value in hidden else value for value in row) for row in plain_rows]
    expected = _cells_except(hidden_rows, plain_rows, left_open)
    rows = duckdb_cursor.sql("SELECT * FROM src").fetchall()
    assert _cells_except(rows, plain_rows, left_open) == expected


def test_2d_mask_added_later_shows_through_variable_scan(duckdb_cursor):
    source = np.ma.masked_array(np.array([[1, 2, 3], [4, 5, 6]], dtype="int64"))
    duckdb_cursor.sql("SELECT * FROM source").fetchall()
    source[0, 1] = np.ma.masked
    expected = _like_plain_with_hidden(duckdb_cursor, np.ma.getdata(source), {2})
    assert duckdb_cursor.sql("SELECT * FROM source").fetchall() == expected


@pytest.mark.parametrize(
    ("make", "setup", "expected"),
    [
        pytest.param(
            lambda: _array(object, ["a", np.ma.masked, "c"]),
            None,
            ["a", None, "c"],
            id="numpy_varchar",
        ),
        pytest.param(
            lambda: _array(object, ["a", np.ma.masked, "c"]),
            "SET pandas_analyze_sample=0",
            ["a", None, "c"],
            id="numpy_varchar_analyzer_off",
        ),
        pytest.param(
            lambda: _masked(object, ["a", np.ma.masked, "c"], [0, 0, 0]),
            None,
            ["a", None, "c"],
            id="masked_array_varchar",
        ),
        pytest.param(
            lambda: pd.DataFrame({"x": pd.Series(["a", np.ma.masked, "c"], dtype=object)}),
            None,
            ["a", None, "c"],
            id="pandas_varchar",
        ),
        pytest.param(lambda: _array(object, [1, np.ma.masked, 3]), None, [1, None, 3], id="numpy_integer"),
        pytest.param(lambda: _array(object, [1.5, np.ma.masked, 2.5]), None, [1.5, None, 2.5], id="numpy_double"),
        pytest.param(lambda: _array(object, [["a"], ["b", np.ma.masked]]), None, [["a"], ["b", None]], id="list_leaf"),
        pytest.param(
            lambda: _array(object, [{"k": "a"}, {"k": np.ma.masked}]), None, [{"k": "a"}, {"k": None}], id="struct_leaf"
        ),
    ],
)
def test_masked_constant_as_value_reads_as_null(duckdb_cursor, make, setup, expected):
    if setup:
        duckdb_cursor.execute(setup)
    source = make()
    assert _scan(duckdb_cursor, "variable", source).fetchall() == _rows(expected)

"""Filters on dictionary encoded Arrow columns.

A filter on a dictionary encoded column must compare the dictionary's values, never its integer indices. Every
case compares the rows against a native DuckDB table holding the same data. A pyarrow Table, a Scanner and a
RecordBatchReader refuse filters, so DuckDB applies them above a scan that picks the columns the query reads. A
pyarrow Dataset takes the filters it can evaluate exactly into its own scan, so the cases pyarrow can evaluate also
run through a Dataset over the same table.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import NamedTuple

import pytest
from _pushdown_helpers import assert_fully_pushed, assert_kept_above, assert_query_keeps_filter_above

import duckdb
from duckdb._sources.to_arrow import dictionary_kernels_exact

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")
pytest.importorskip("pyarrow.dataset")

# Dictionary entry 3 is NULL and row 5 has a NULL index, so both kinds of NULL meet every predicate.
INDICES = [0, 1, 2, 3, None, 1, 0]
ROWS = list(range(1, len(INDICES) + 1))

UTC = dt.timezone.utc
NEW_YORK = "America/New_York"


class DictCase(NamedTuple):
    id: str
    value_type: object
    values: tuple  # low, mid, high
    literals: tuple  # SQL literals for low, mid, high
    session_zone: str | None = None
    value_field: str | None = None


def _tstz(year, month, day):
    return dt.datetime(year, month, day, tzinfo=UTC)


def _tstz_literal(value):
    return f"TIMESTAMPTZ '{value:%Y-%m-%d %H:%M:%S}+00'"


TSTZ_VALUES = (_tstz(2020, 1, 1), _tstz(2020, 6, 1), _tstz(2021, 1, 1))
TSTZ_LITERALS = tuple(_tstz_literal(v) for v in TSTZ_VALUES)
BLOB_LITERALS = ("'aa'::BLOB", "'bb'::BLOB", "'cc'::BLOB")
DECIMALS = (Decimal("1.25"), Decimal("2.50"), Decimal("3.75"))
STRUCT_TYPE = pa.struct([("i", pa.int64()), ("s", pa.string())])
STRUCT_VALUES = ({"i": 1, "s": "a"}, {"i": 2, "s": "b"}, {"i": 3, "s": "c"})


def _decimal_literals(width, scale):
    return tuple(f"{v}::DECIMAL({width},{scale})" for v in DECIMALS)


UNTYPED_SCALARS = [
    DictCase("utf8", pa.utf8(), ("a", "b", "c"), ("'a'", "'b'", "'c'")),
    DictCase("large_utf8", pa.large_utf8(), ("a", "b", "c"), ("'a'", "'b'", "'c'")),
    DictCase("int64", pa.int64(), (5, 7, 9), ("5", "7", "9")),
    DictCase(
        "date32",
        pa.date32(),
        (dt.date(2020, 1, 1), dt.date(2020, 6, 1), dt.date(2021, 1, 1)),
        ("DATE '2020-01-01'", "DATE '2020-06-01'", "DATE '2021-01-01'"),
    ),
    DictCase("bool", pa.bool_(), (False, True, True), ("false", "true", "true")),
    DictCase(
        "timestamp_us",
        pa.timestamp("us"),
        (dt.datetime(2020, 1, 1), dt.datetime(2020, 6, 1), dt.datetime(2021, 1, 1)),
        ("TIMESTAMP '2020-01-01'", "TIMESTAMP '2020-06-01'", "TIMESTAMP '2021-01-01'"),
    ),
]

FLOATS = [
    DictCase("float32", pa.float32(), (1.5, 2.5, float("nan")), ("1.5::FLOAT", "2.5::FLOAT", "'nan'::FLOAT")),
    DictCase("float64", pa.float64(), (1.5, 2.5, float("nan")), ("1.5::DOUBLE", "2.5::DOUBLE", "'nan'::DOUBLE")),
]

TYPED_SCALARS = [
    DictCase("binary", pa.binary(), (b"aa", b"bb", b"cc"), BLOB_LITERALS),
    DictCase("large_binary", pa.large_binary(), (b"aa", b"bb", b"cc"), BLOB_LITERALS),
    DictCase(
        "fixed_size_binary",
        pa.binary(4),
        (b"aaaa", b"bbbb", b"cccc"),
        ("'aaaa'::BLOB", "'bbbb'::BLOB", "'cccc'::BLOB"),
    ),
    DictCase("decimal128_30_2", pa.decimal128(30, 2), DECIMALS, _decimal_literals(30, 2)),
    DictCase("decimal128_10_2", pa.decimal128(10, 2), DECIMALS, _decimal_literals(10, 2)),
    DictCase("timestamptz_utc", pa.timestamp("us", tz="UTC"), TSTZ_VALUES, TSTZ_LITERALS, session_zone="UTC"),
    DictCase(
        "timestamptz_new_york",
        pa.timestamp("us", tz=NEW_YORK),
        TSTZ_VALUES,
        TSTZ_LITERALS,
        session_zone=NEW_YORK,
    ),
]

# pyarrow cannot evaluate a comparison on these dictionaries.
NO_PYARROW_KERNEL = [
    DictCase("decimal32", pa.decimal32(5, 2), DECIMALS, _decimal_literals(5, 2)),
    DictCase("decimal64", pa.decimal64(15, 2), DECIMALS, _decimal_literals(15, 2)),
    DictCase("binary_view", pa.binary_view(), (b"aa", b"bb", b"cc"), BLOB_LITERALS),
    DictCase("string_view", pa.string_view(), ("a", "b", "c"), ("'a'", "'b'", "'c'")),
]

# The session zone differs from the column's zone.
ZONE_MISMATCH = [
    DictCase(
        "timestamptz_utc_in_new_york",
        pa.timestamp("us", tz="UTC"),
        TSTZ_VALUES,
        TSTZ_LITERALS,
        session_zone=NEW_YORK,
    ),
    DictCase(
        "timestamptz_new_york_in_utc",
        pa.timestamp("us", tz=NEW_YORK),
        TSTZ_VALUES,
        TSTZ_LITERALS,
        session_zone="UTC",
    ),
]

# pyarrow cannot reference a field inside a dictionary of structs, which DuckDB filters itself.
STRUCT_DICTIONARIES = [
    DictCase("struct_of_int", STRUCT_TYPE, STRUCT_VALUES, ("1", "2", "3"), value_field="i"),
    DictCase("struct_of_string", STRUCT_TYPE, STRUCT_VALUES, ("'a'", "'b'", "'c'"), value_field="s"),
]

LIST_DICTIONARIES = [
    DictCase(
        "list_of_int",
        pa.list_(pa.int64()),
        ([1], [1, 2], [3]),
        ("[1]::BIGINT[]", "[1, 2]::BIGINT[]", "[3]::BIGINT[]"),
    ),
]

PREDICATES = {
    "eq": "{col} = {mid}",
    "ne": "{col} <> {mid}",
    "lt": "{col} < {mid}",
    "le": "{col} <= {mid}",
    "gt": "{col} > {mid}",
    "ge": "{col} >= {mid}",
    "between": "{col} BETWEEN {low} AND {mid}",
    "in": "{col} IN ({low}, {high})",
    "is_null": "{col} IS NULL",
    "eq_nan": "{col} = {high}",
    "ne_nan": "{col} <> {high}",
    "lt_nan": "{col} < {high}",
    "ge_nan": "{col} >= {high}",
}
COMPARISONS = ["eq", "ne", "lt", "le", "gt", "ge", "between"]
# The float cases hold NaN as their high value, so these compare with a NaN constant.
NAN_COMPARISONS = ["eq_nan", "ne_nan", "lt_nan", "ge_nan"]
ALL_OPS = [*COMPARISONS, "in", "is_null"]
SHAPES = ["bare", "in_struct"]
NESTED_SHAPES = ["in_null_struct", "two_structs_deep"]
# The struct is NULL on rows whose dictionary value matches, so a filter ignoring the parent's validity returns them.
NULL_STRUCT_ROWS = [False, True, False, False, False, False, True]

INVALIDATING = (duckdb.InternalException, duckdb.FatalException)


def _dictionary(case):
    dictionary = pa.array([*case.values, None], case.value_type)
    return pa.DictionaryArray.from_arrays(pa.array(INDICES, pa.int8()), dictionary)


def _register(con, case, column, column_name, wrap=None):
    if case.session_zone:
        con.execute(f"SET TimeZone = '{case.session_zone}'")
    table = pa.table({"x": ROWS, column_name: column})
    con.register("arrow_t", table)
    con.execute("CREATE TABLE native_t AS SELECT * FROM arrow_t")
    if wrap is not None:
        con.register("arrow_t", wrap(table))


def _as_dataset(table):
    return pytest.importorskip("pyarrow.dataset").dataset(table)


def _register_shape(con, case, shape, wrap=None):
    dictionary = _dictionary(case)
    if shape == "bare":
        _register(con, case, dictionary, "d", wrap)
        col = "d"
    elif shape == "in_struct":
        _register(con, case, pa.StructArray.from_arrays([dictionary], ["d"]), "s", wrap)
        col = "s.d"
    elif shape == "in_null_struct":
        mask = pa.array(NULL_STRUCT_ROWS)
        _register(con, case, pa.StructArray.from_arrays([dictionary], ["d"], mask=mask), "s", wrap)
        col = "s.d"
    else:
        inner = pa.StructArray.from_arrays([dictionary], ["d"])
        _register(con, case, pa.StructArray.from_arrays([inner], ["t"]), "s", wrap)
        col = "s.t.d"
    return f"{col}.{case.value_field}" if case.value_field else col


def _predicate(case, op, col):
    low, mid, high = case.literals
    return PREDICATES[op].format(col=col, low=low, mid=mid, high=high)


def _rows(con, table, predicate):
    return con.execute(f"SELECT x FROM {table} WHERE {predicate} ORDER BY x").fetchall()


def _arrow_query(predicate):
    return f"SELECT x FROM arrow_t WHERE {predicate}"


def _arrow_rows(con, predicate):
    try:
        return _rows(con, "arrow_t", predicate)
    except INVALIDATING as error:
        # InternalException and FatalException derive from duckdb.Error, which some strict xfail markers expect.
        pytest.fail(f"the Arrow query invalidated the database: {error}")


def _params(cases, ops, marks_for, shapes=SHAPES):
    params = []
    for case in cases:
        for shape in shapes:
            for op in ops:
                marks = marks_for(case, op)
                params.append(pytest.param(case, shape, op, marks=marks, id=f"{case.id}-{shape}-{op}"))
    return params


def _xfail(reason, raises):
    return [pytest.mark.xfail(strict=True, raises=raises, reason=reason)]


def _no_marks(case, op):
    return []


def _case(cases, case_id):
    return next(case for case in cases if case.id == case_id)


NESTING_CASES = [
    _case(TYPED_SCALARS, "binary"),
    _case(UNTYPED_SCALARS, "utf8"),
    _case(TYPED_SCALARS, "decimal128_30_2"),
]

MATCHES_NATIVE = [
    *_params(UNTYPED_SCALARS, ["eq", "between", "in", "is_null"], _no_marks),
    *_params(FLOATS, [*ALL_OPS, *NAN_COMPARISONS], _no_marks),
    *_params(TYPED_SCALARS, ALL_OPS, _no_marks),
    *_params(ZONE_MISMATCH, COMPARISONS, _no_marks),
    *_params(NESTING_CASES, ["eq", "is_null"], _no_marks, shapes=NESTED_SHAPES),
]

# The engine leaves the struct children of a dictionary's NULL sentinel slot uninitialized, so scanning a dictionary
# of structs with a NULL index reads garbage and crashes the process on dirty memory (Windows x86_64 in CI, any
# platform with DUCKDB_JE_MALLOC_CONF=junk:true). A crash cannot be an xfail; skipped until the engine fix is pinned.
STRUCT_SENTINEL = pytest.mark.skip(
    reason="engine: the NULL sentinel of a dictionary of structs has uninitialized children"
)

# pyarrow has no kernel for these comparisons, IN lists or field references, which DuckDB evaluates itself.
UNFILTERABLE_BY_PYARROW = [
    *_params(NO_PYARROW_KERNEL, ALL_OPS, _no_marks),
    *_params(STRUCT_DICTIONARIES, ["eq", "lt", "between", "in", "is_null"], lambda case, op: [STRUCT_SENTINEL]),
    *_params(LIST_DICTIONARIES, ["eq", "in", "is_null"], _no_marks),
]


@pytest.mark.parametrize(("case", "shape", "op"), MATCHES_NATIVE)
def test_filter_matches_native(duckdb_cursor, case, shape, op):
    col = _register_shape(duckdb_cursor, case, shape)
    predicate = _predicate(case, op, col)
    expected = _rows(duckdb_cursor, "native_t", predicate)
    assert _arrow_rows(duckdb_cursor, predicate) == expected
    assert_kept_above(duckdb_cursor, _arrow_query(predicate), len(expected))


@pytest.mark.parametrize(("case", "shape", "op"), MATCHES_NATIVE)
def test_filter_through_a_dataset_matches_native(duckdb_cursor, case, shape, op):
    col = _register_shape(duckdb_cursor, case, shape, _as_dataset)
    predicate = _predicate(case, op, col)
    assert _arrow_rows(duckdb_cursor, predicate) == _rows(duckdb_cursor, "native_t", predicate)
    # A pyarrow whose kernels read a valid index to a NULL dictionary entry as a value gets no dictionary predicate.
    if not dictionary_kernels_exact():
        assert_query_keeps_filter_above(duckdb_cursor, _arrow_query(predicate))


@pytest.mark.parametrize(("case", "shape", "op"), UNFILTERABLE_BY_PYARROW)
def test_filter_pyarrow_cannot_evaluate_matches_native(duckdb_cursor, case, shape, op):
    col = _register_shape(duckdb_cursor, case, shape)
    predicate = _predicate(case, op, col)
    assert _arrow_rows(duckdb_cursor, predicate) == _rows(duckdb_cursor, "native_t", predicate)


RUN_END_ENCODED = [
    pytest.param(case, op, id=f"{case.id}-{op}")
    for case in [
        _case(TYPED_SCALARS, "decimal128_30_2"),
        _case(TYPED_SCALARS, "timestamptz_utc"),
        _case(ZONE_MISMATCH, "timestamptz_utc_in_new_york"),
    ]
    for op in ["eq", "gt"]
]


@pytest.mark.parametrize(("case", "op"), RUN_END_ENCODED)
def test_run_end_encoded_filter_matches_native(duckdb_cursor, case, op):
    plain = pa.array([None if i is None else [*case.values, None][i] for i in INDICES], case.value_type)
    _register(duckdb_cursor, case, pc.run_end_encode(plain), "r")
    predicate = _predicate(case, op, "r")
    assert _arrow_rows(duckdb_cursor, predicate) == _rows(duckdb_cursor, "native_t", predicate)


@pytest.mark.parametrize("path", ["s.r", "s.t.r"])
def test_nested_run_end_encoded_filter_matches_native(duckdb_cursor, path):
    column = pc.run_end_encode(pa.array([Decimal("1.25"), Decimal("2.50"), None], pa.decimal128(30, 2)))
    for name in reversed(path.split(".")[1:]):
        column = pa.StructArray.from_arrays([column], [name])
    duckdb_cursor.register("arrow_t", pa.table({"x": [1, 2, 3], "s": column}))
    duckdb_cursor.execute("CREATE TABLE native_t AS SELECT * FROM arrow_t")
    predicate = f"{path} > 2.00::DECIMAL(30,2)"
    assert _rows(duckdb_cursor, "native_t", predicate) == [(2,)]
    assert _arrow_rows(duckdb_cursor, predicate) == [(2,)]


FLOAT_CHUNK_PREDICATES = {
    "eq_nan": "= 'nan'::{type}",
    "ne_nan": "<> 'nan'::{type}",
    "lt_nan": "< 'nan'::{type}",
    "ge_nan": ">= 'nan'::{type}",
    "gt": "> 1.0::{type}",
}


@pytest.mark.parametrize("op", list(FLOAT_CHUNK_PREDICATES))
@pytest.mark.parametrize(("value_type", "sql_type"), [(pa.float32(), "FLOAT"), (pa.float64(), "DOUBLE")])
def test_float_dictionary_chunks_with_different_dictionaries_match_native(duckdb_cursor, value_type, sql_type, op):
    nan = float("nan")
    first = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 2, None, 0], pa.int8()), pa.array([1.5, nan, None], value_type)
    )
    second = pa.DictionaryArray.from_arrays(
        pa.array([3, 2, 1, 0, None], pa.int8()), pa.array([0.5, None, nan, 2.5], value_type)
    )
    column = pa.chunked_array([first, second])
    duckdb_cursor.register("arrow_t", pa.table({"x": list(range(1, 11)), "d": column}))
    duckdb_cursor.execute("CREATE TABLE native_t AS SELECT * FROM arrow_t")
    predicate = f"d {FLOAT_CHUNK_PREDICATES[op].format(type=sql_type)}"
    expected = _rows(duckdb_cursor, "native_t", predicate)
    assert _arrow_rows(duckdb_cursor, predicate) == expected
    assert_kept_above(duckdb_cursor, _arrow_query(predicate), len(expected))


UNIT_MICROS = {"s": 1_000_000, "ms": 1_000, "us": 1}
EPOCH = dt.datetime(1970, 1, 1, tzinfo=UTC)
# Constants on the column's grid, and half a unit above and below it (the latter before 1970).
GRID_OFFSETS = {"on_grid": 0, "half_above": 1, "half_below": -1}
ZONED_COMPARISONS = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
EPOCH_LITERAL = "TIMESTAMPTZ '1970-01-01 00:00:00+00'"


def _micros_literal(micros):
    return f"TIMESTAMPTZ '{EPOCH + dt.timedelta(microseconds=micros):%Y-%m-%d %H:%M:%S.%f}+00'"


def _register_zoned(con, unit, column_zone, session_zone, shape):
    step = UNIT_MICROS[unit]
    values = tuple(EPOCH + dt.timedelta(microseconds=k * step) for k in (-1, 0, 1))
    case = DictCase(f"{unit}_{column_zone}", pa.timestamp(unit, tz=column_zone), values, (), session_zone)
    if shape == "plain":
        _register(con, case, _dictionary(case).cast(case.value_type), "p")
        return "p"
    return _register_shape(con, case, shape)


@pytest.mark.parametrize("op", [*ZONED_COMPARISONS, "between"])
@pytest.mark.parametrize("shape", [*SHAPES, "plain"])
@pytest.mark.parametrize("offset", list(GRID_OFFSETS))
@pytest.mark.parametrize("session_zone", ["UTC", NEW_YORK])
@pytest.mark.parametrize("column_zone", ["UTC", NEW_YORK])
@pytest.mark.parametrize("unit", ["s", "ms"])
def test_zoned_timestamp_comparison_matches_native(duckdb_cursor, unit, column_zone, session_zone, offset, shape, op):
    step = UNIT_MICROS[unit]
    col = _register_zoned(duckdb_cursor, unit, column_zone, session_zone, shape)
    constant = GRID_OFFSETS[offset] * step // 2
    if op == "between":
        predicate = f"{col} BETWEEN {_micros_literal(constant)} AND {_micros_literal(constant + step)}"
    else:
        predicate = f"{col} {ZONED_COMPARISONS[op]} {_micros_literal(constant)}"
    expected = _rows(duckdb_cursor, "native_t", predicate)
    assert _arrow_rows(duckdb_cursor, predicate) == expected
    assert_kept_above(duckdb_cursor, _arrow_query(predicate), len(expected))


INFINITE_BOUNDS = {
    "minus_infinity_to_infinity": ("'-infinity'::TIMESTAMPTZ", "'infinity'::TIMESTAMPTZ"),
    "minus_infinity_to_epoch": ("'-infinity'::TIMESTAMPTZ", EPOCH_LITERAL),
    "epoch_to_infinity": (EPOCH_LITERAL, "'infinity'::TIMESTAMPTZ"),
}


@pytest.mark.parametrize("bounds", list(INFINITE_BOUNDS))
@pytest.mark.parametrize("shape", [*SHAPES, "plain"])
@pytest.mark.parametrize("session_zone", ["UTC", NEW_YORK])
@pytest.mark.parametrize("column_zone", ["UTC", NEW_YORK])
@pytest.mark.parametrize("unit", list(UNIT_MICROS))
def test_zoned_timestamp_between_infinities_matches_native(
    duckdb_cursor, unit, column_zone, session_zone, shape, bounds
):
    col = _register_zoned(duckdb_cursor, unit, column_zone, session_zone, shape)
    low, high = INFINITE_BOUNDS[bounds]
    predicate = f"{col} BETWEEN {low} AND {high}"
    expected = _rows(duckdb_cursor, "native_t", predicate)
    assert _arrow_rows(duckdb_cursor, predicate) == expected
    assert_kept_above(duckdb_cursor, _arrow_query(predicate), len(expected))


INFINITE_IN_LISTS = {
    "infinities": "'-infinity'::TIMESTAMPTZ, 'infinity'::TIMESTAMPTZ",
    "infinity_and_epoch": f"'infinity'::TIMESTAMPTZ, {EPOCH_LITERAL}",
}
INFINITE_IN_CASES = [
    pytest.param(
        unit,
        column_zone,
        session_zone,
        shape,
        values,
        id=f"{unit}-{column_zone}-{session_zone}-{shape}-{values}",
    )
    for unit in UNIT_MICROS
    for column_zone in ["UTC", NEW_YORK]
    for session_zone in ["UTC", NEW_YORK]
    for shape in [*SHAPES, "plain"]
    for values in INFINITE_IN_LISTS
]


@pytest.mark.parametrize(("unit", "column_zone", "session_zone", "shape", "values"), INFINITE_IN_CASES)
def test_zoned_timestamp_in_with_infinity_matches_native(duckdb_cursor, unit, column_zone, session_zone, shape, values):
    col = _register_zoned(duckdb_cursor, unit, column_zone, session_zone, shape)
    predicate = f"{col} IN ({INFINITE_IN_LISTS[values]})"
    assert _arrow_rows(duckdb_cursor, predicate) == _rows(duckdb_cursor, "native_t", predicate)


@pytest.mark.parametrize("shape", ["plain", "dictionary"])
@pytest.mark.parametrize("column_zone", ["UTC", NEW_YORK])
@pytest.mark.parametrize("session_zone", ["UTC", NEW_YORK])
@pytest.mark.parametrize("unit", ["s", "ms"])
def test_zoned_timestamp_in_with_off_grid_constants_matches_native(
    duckdb_cursor, unit, session_zone, column_zone, shape
):
    col = _register_zoned(duckdb_cursor, unit, column_zone, session_zone, "bare" if shape == "dictionary" else "plain")
    step = UNIT_MICROS[unit]
    predicate = f"{col} IN ({_micros_literal(step // 2)}, {_micros_literal(step)}, {_micros_literal(-step // 2)})"
    assert _arrow_rows(duckdb_cursor, predicate) == _rows(duckdb_cursor, "native_t", predicate)


@pytest.mark.parametrize("op", ["eq", "gt"])
@pytest.mark.parametrize("shape", ["plain", "dictionary"])
def test_zoned_timestamp_two_structs_deep_matches_native(duckdb_cursor, shape, op):
    step = UNIT_MICROS["s"]
    values = tuple(EPOCH + dt.timedelta(microseconds=k * step) for k in (-1, 0, 1))
    case = DictCase("s_utc", pa.timestamp("s", tz="UTC"), values, (), NEW_YORK)
    leaf = _dictionary(case) if shape == "dictionary" else _dictionary(case).cast(case.value_type)
    inner = pa.StructArray.from_arrays([leaf], ["d"])
    _register(duckdb_cursor, case, pa.StructArray.from_arrays([inner], ["t"]), "s")
    predicate = f"s.t.d {ZONED_COMPARISONS[op]} {_micros_literal(step // 2)}"
    expected = _rows(duckdb_cursor, "native_t", predicate)
    assert _arrow_rows(duckdb_cursor, predicate) == expected
    assert_kept_above(duckdb_cursor, _arrow_query(predicate), len(expected))


@pytest.mark.parametrize(
    ("op", "constant", "expected"),
    [
        pytest.param("gt", "'1970-01-01 00:00:00.0005+00'", [(3,)], id="gt"),
        pytest.param("eq", "'1970-01-01 00:00:00+00'", [(2,)], id="eq"),
    ],
)
@pytest.mark.parametrize("shape", ["plain", "dictionary"])
def test_zoned_timestamp_beside_decoy_siblings_matches_native(duckdb_cursor, shape, op, constant, expected):
    # A decoy before the real child at each struct level makes a wrong sibling index read a UTC microsecond column.
    leaf = pa.array([-1, 0, 1], pa.int64()).cast(pa.timestamp("ms", tz="+05:45"))
    if shape == "dictionary":
        leaf = pa.DictionaryArray.from_arrays(pa.array([0, 1, 2], pa.int8()), leaf)
    decoy = pa.array([10**6, 2 * 10**6, 3 * 10**6], pa.int64()).cast(pa.timestamp("us", tz="UTC"))
    inner = pa.StructArray.from_arrays([decoy, leaf], ["decoy", "d"])
    duckdb_cursor.execute(f"SET TimeZone = '{NEW_YORK}'")
    duckdb_cursor.register(
        "arrow_t", pa.table({"x": [1, 2, 3], "s": pa.StructArray.from_arrays([decoy, inner], ["decoy", "t"])})
    )
    duckdb_cursor.execute("CREATE TABLE native_t AS SELECT * FROM arrow_t")
    predicate = f"s.t.d {ZONED_COMPARISONS[op]} TIMESTAMPTZ {constant}"
    assert _rows(duckdb_cursor, "native_t", predicate) == expected
    assert _arrow_rows(duckdb_cursor, predicate) == expected
    assert_kept_above(duckdb_cursor, _arrow_query(predicate), len(expected))


@pytest.mark.parametrize("op", ["eq", "lt", "between"])
@pytest.mark.parametrize("offset", ["on_grid", "half_above"])
@pytest.mark.parametrize("shape", ["bare", "plain"])
@pytest.mark.parametrize("column_zone", ["+05:00", "-03:30"])
@pytest.mark.parametrize("unit", ["s", "ms"])
def test_fixed_offset_zone_comparison_matches_native(duckdb_cursor, unit, column_zone, shape, offset, op):
    step = UNIT_MICROS[unit]
    col = _register_zoned(duckdb_cursor, unit, column_zone, NEW_YORK, shape)
    constant = GRID_OFFSETS[offset] * step // 2
    if op == "between":
        predicate = f"{col} BETWEEN {_micros_literal(constant)} AND {_micros_literal(constant + step)}"
    else:
        predicate = f"{col} {ZONED_COMPARISONS[op]} {_micros_literal(constant)}"
    expected = _rows(duckdb_cursor, "native_t", predicate)
    assert _arrow_rows(duckdb_cursor, predicate) == expected
    assert_kept_above(duckdb_cursor, _arrow_query(predicate), len(expected))


def _zoned_source(unit, column_zone, shape, source):
    step = UNIT_MICROS[unit]
    values = tuple(EPOCH + dt.timedelta(microseconds=k * step) for k in (-1, 0, 1))
    case = DictCase(f"{unit}_{column_zone}", pa.timestamp(unit, tz=column_zone), values, ())
    column = _dictionary(case) if shape == "dictionary" else _dictionary(case).cast(case.value_type)
    table = pa.table({"x": ROWS, "p": column})
    if source == "scanner":
        return table, pytest.importorskip("pyarrow.dataset").dataset(table).scanner()
    return table, table.to_reader()


@pytest.mark.parametrize("op", ["eq", "gt"])
@pytest.mark.parametrize("offset", ["on_grid", "half_above"])
@pytest.mark.parametrize("shape", ["plain", "dictionary"])
@pytest.mark.parametrize("session_zone", ["UTC", NEW_YORK])
@pytest.mark.parametrize("column_zone", ["UTC", NEW_YORK])
@pytest.mark.parametrize("unit", ["s", "ms"])
@pytest.mark.parametrize("source", ["scanner", "record_batch_reader"])
def test_zoned_timestamp_scanner_and_reader_sources_match_native(
    duckdb_cursor, source, unit, column_zone, session_zone, shape, offset, op
):
    step = UNIT_MICROS[unit]
    predicate = f"p {ZONED_COMPARISONS[op]} {_micros_literal(GRID_OFFSETS[offset] * step // 2)}"
    duckdb_cursor.execute(f"SET TimeZone = '{session_zone}'")
    table, arrow_source = _zoned_source(unit, column_zone, shape, source)
    duckdb_cursor.register("native_source", table)
    duckdb_cursor.execute("CREATE TABLE native_t AS SELECT * FROM native_source")
    duckdb_cursor.register("arrow_t", arrow_source)
    expected = _rows(duckdb_cursor, "native_t", predicate)
    assert _arrow_rows(duckdb_cursor, predicate) == expected
    # A record batch reader can be scanned once, so the plan check gets a fresh source.
    duckdb_cursor.register("arrow_t", _zoned_source(unit, column_zone, shape, source)[1])
    assert_kept_above(duckdb_cursor, _arrow_query(predicate), len(expected))


class _SchemaExportOnly:
    """A schema object that supports only the Arrow C export, as some dataset wrappers return."""

    def __init__(self, schema) -> None:
        self._schema = schema

    def _export_to_c(self, *args):
        return self._schema._export_to_c(*args)


@pytest.mark.parametrize("op", ["eq", "gt"])
@pytest.mark.parametrize("offset", ["on_grid", "half_above"])
@pytest.mark.parametrize("shape", ["plain", "dictionary"])
def test_dataset_with_an_export_only_schema_matches_native(duckdb_cursor, shape, offset, op):
    pa_ds = pytest.importorskip("pyarrow.dataset")

    class ExportOnlySchemaDataset(pa_ds.InMemoryDataset):
        @property
        def schema(self):
            return _SchemaExportOnly(super().schema)

    column = pa.array([-1, 0, 1, None], pa.timestamp("ms", tz="UTC"))
    if shape == "dictionary":
        column = pa.DictionaryArray.from_arrays(pa.array([0, 1, 2, None], pa.int8()), column[:3])
    table = pa.table({"x": [1, 2, 3, 4], "p": column})
    duckdb_cursor.execute(f"SET TimeZone = '{NEW_YORK}'")
    duckdb_cursor.register("native_source", table)
    duckdb_cursor.execute("CREATE TABLE native_t AS SELECT * FROM native_source")
    duckdb_cursor.register("arrow_t", ExportOnlySchemaDataset(table))
    predicate = f"p {ZONED_COMPARISONS[op]} {_micros_literal(GRID_OFFSETS[offset] * UNIT_MICROS['ms'] // 2)}"
    expected = _rows(duckdb_cursor, "native_t", predicate)
    assert _arrow_rows(duckdb_cursor, predicate) == expected
    if shape == "dictionary" and not dictionary_kernels_exact():
        assert_kept_above(duckdb_cursor, _arrow_query(predicate), len(expected))
    else:
        assert_fully_pushed(duckdb_cursor, _arrow_query(predicate), len(expected))


# The first stored values whose microsecond form overflows int64; one unit less still fits.
FIRST_OVERFLOWING = {"s": 9_223_372_036_855, "ms": 9_223_372_036_854_776}


def _register_raw_zoned(con, unit, raw_values, shape):
    con.execute("SET TimeZone = 'UTC'")
    column = pa.array([*raw_values, None], pa.int64()).cast(pa.timestamp(unit, tz="UTC"))
    if shape == "dictionary":
        column = pa.DictionaryArray.from_arrays(pa.array(range(len(column)), pa.int8()), column)
    con.register("arrow_t", pa.table({"x": list(range(1, len(column) + 1)), "p": column}))


@pytest.mark.parametrize("op", list(ZONED_COMPARISONS))
@pytest.mark.parametrize("shape", ["plain", "dictionary"])
@pytest.mark.parametrize("unit", list(FIRST_OVERFLOWING))
def test_zoned_timestamp_at_the_microsecond_limit_matches_hand_rows(duckdb_cursor, unit, shape, op):
    _register_raw_zoned(duckdb_cursor, unit, [FIRST_OVERFLOWING[unit] - 1], shape)
    expected = [(1,)] if op in ("ne", "gt", "ge") else []
    assert _arrow_rows(duckdb_cursor, f"p {ZONED_COMPARISONS[op]} {EPOCH_LITERAL}") == expected


@pytest.mark.parametrize("op", list(ZONED_COMPARISONS))
@pytest.mark.parametrize("shape", ["plain", "dictionary"])
@pytest.mark.parametrize("unit", list(FIRST_OVERFLOWING))
def test_zoned_timestamp_beyond_the_microsecond_limit(duckdb_cursor, unit, shape, op):
    first = FIRST_OVERFLOWING[unit]
    _register_raw_zoned(duckdb_cursor, unit, [first - 1, first], shape)
    predicate = f"p {ZONED_COMPARISONS[op]} {EPOCH_LITERAL}"
    # The scan converts every row before DuckDB filters, and the overflowing row has no DuckDB TIMESTAMPTZ.
    with pytest.raises(duckdb.ConversionException, match="Could not convert TimestampTZ to Microsecond"):
        _rows(duckdb_cursor, "arrow_t", predicate)
    assert duckdb_cursor.execute("SELECT 42").fetchall() == [(42,)]


@pytest.mark.parametrize("op", ["gt", "eq", "between"])
@pytest.mark.parametrize("offset", ["on_grid", "half_above"])
@pytest.mark.parametrize("column_zone", ["UTC", NEW_YORK])
@pytest.mark.parametrize("unit", ["s", "ms"])
def test_zoned_timestamp_parquet_dataset_matches_native(duckdb_cursor, tmp_path, unit, column_zone, offset, op):
    pq = pytest.importorskip("pyarrow.parquet")
    pa_ds = pytest.importorskip("pyarrow.dataset")
    step = UNIT_MICROS[unit]
    column = pa.array([*range(20), None], pa.int64()).cast(pa.timestamp(unit, tz=column_zone))
    path = tmp_path / "zoned.parquet"
    pq.write_table(pa.table({"x": list(range(1, 22)), "p": column}), path, row_group_size=4)
    duckdb_cursor.execute(f"SET TimeZone = '{NEW_YORK}'")
    duckdb_cursor.register("arrow_t", pa_ds.dataset(path))
    duckdb_cursor.execute("CREATE TABLE native_t AS SELECT * FROM arrow_t")
    constant = 10 * step + GRID_OFFSETS[offset] * step // 2
    if op == "between":
        predicate = f"p BETWEEN {_micros_literal(constant)} AND {_micros_literal(constant + 3 * step)}"
    else:
        predicate = f"p {ZONED_COMPARISONS[op]} {_micros_literal(constant)}"
    expected = _rows(duckdb_cursor, "native_t", predicate)
    assert _arrow_rows(duckdb_cursor, predicate) == expected
    assert_fully_pushed(duckdb_cursor, _arrow_query(predicate), len(expected))


OFF_GRID_PRUNING = "per unit rounding: an off grid constant prunes only once it is rounded into the column's unit"
PRUNING_CONSTANTS_MS = {"on_grid": 799, "off_grid": 799.5}


def _register_capturing_parquet(con, tmp_path, column_zone):
    pq = pytest.importorskip("pyarrow.parquet")
    pa_ds = pytest.importorskip("pyarrow.dataset")
    captured = []

    class FilterCapturingDataset(pa_ds.FileSystemDataset):
        def scanner(self, **kwargs):
            captured.append(kwargs.get("filter"))
            return super().scanner(**kwargs)

    column = pa.array(range(1000), pa.int64()).cast(pa.timestamp("ms", tz=column_zone))
    path = tmp_path / "pruning.parquet"
    pq.write_table(pa.table({"x": list(range(1000)), "p": column}), path, row_group_size=100)
    source = pa_ds.dataset(path)
    dataset = FilterCapturingDataset(list(source.get_fragments()), source.schema, source.format, source.filesystem)
    con.execute(f"SET TimeZone = '{NEW_YORK}'")
    con.register("arrow_t", dataset)
    return next(iter(source.get_fragments())), captured


def _pruning_count(con, constant_ms):
    predicate = f"p > {_micros_literal(int(constant_ms * 1000))}"
    return con.execute(f"SELECT count(*) FROM arrow_t WHERE {predicate}").fetchone()


@pytest.mark.parametrize("constant", list(PRUNING_CONSTANTS_MS))
@pytest.mark.parametrize("column_zone", ["UTC", NEW_YORK])
def test_zoned_timestamp_parquet_pruning_filter_matches_hand_rows(duckdb_cursor, tmp_path, column_zone, constant):
    _register_capturing_parquet(duckdb_cursor, tmp_path, column_zone)
    assert _pruning_count(duckdb_cursor, PRUNING_CONSTANTS_MS[constant]) == (200,)


@pytest.mark.parametrize(
    "constant",
    [
        pytest.param("on_grid"),
        pytest.param("off_grid", marks=_xfail(OFF_GRID_PRUNING, AssertionError)),
    ],
)
@pytest.mark.parametrize("column_zone", ["UTC", NEW_YORK])
def test_zoned_timestamp_parquet_row_group_pruning(duckdb_cursor, tmp_path, column_zone, constant):
    fragment, captured = _register_capturing_parquet(duckdb_cursor, tmp_path, column_zone)
    _pruning_count(duckdb_cursor, PRUNING_CONSTANTS_MS[constant])
    assert fragment.subset(captured[-1]).num_row_groups == 2

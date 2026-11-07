"""map() fixes one type per output column at bind and reads each chunk's returned column into it."""

import collections
import datetime
import json
import re
from decimal import Decimal
from fractions import Fraction

import pytest
from conftest import PANDAS_GE_3

import duckdb
import numpy as np
import pandas as pd

ENUM = "ENUM('a', 'b')"
SCHEMAS = pytest.mark.parametrize("explicit_schema", [False, True], ids=["inferred", "explicit_schema"])


def _schema_of(relation, explicit_schema):
    return dict(zip(relation.columns, relation.types, strict=True)) if explicit_schema else None


@pytest.mark.parametrize("with_null", [False, True], ids=["no_null", "a_null"])
def test_map_enum_column_reaches_the_function_as_df_gives_it(duckdb_cursor, with_null):
    second = "NULL" if with_null else f"'a'::{ENUM}"
    source = duckdb_cursor.sql(f"select * from (values ('b'::{ENUM}), ({second})) t(e)")
    seen = []

    def record(df):
        seen.append((len(df), df["e"].dtype, [None if pd.isna(v) else v for v in df["e"].tolist()]))
        return df

    mapped = source.map(record)
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()
    categorical = pd.CategoricalDtype(["a", "b"], ordered=True)
    assert source.df()["e"].dtype == categorical
    assert all(dtype == categorical for _, dtype, _ in seen), seen
    assert [values for rows, _, values in seen if rows] == [["b", None if with_null else "a"]]


def test_map_enum_column_reaches_every_call_with_the_dtype_built_at_bind(duckdb_cursor):
    duckdb_cursor.execute("set threads = 1")
    seen = []

    def record(df):
        if len(df):
            seen.append(df["e"].dtype)
        return df

    source = duckdb_cursor.sql(f"select (case when i % 3 = 0 then 'a' else 'b' end)::{ENUM} as e from range(4096) t(i)")
    assert source.map(record).fetchall() == source.fetchall()
    assert len(seen) == 2
    # pandas 3 keeps the dtype object; pandas 2 builds a new dtype around the same categories index
    assert all(dtype.categories is seen[0].categories for dtype in seen)
    if PANDAS_GE_3:
        assert all(dtype is seen[0] for dtype in seen)


@pytest.mark.parametrize(
    ("categories", "ordered", "result_type"),
    [
        pytest.param(["a", "b"], True, ENUM, id="equal_categories"),
        pytest.param(["a", "b"], False, ENUM, id="equal_categories_unordered"),
        pytest.param(["b", "a"], True, "ENUM('b', 'a')", id="reordered_categories"),
    ],
)
def test_map_enum_column_rebuilt_as_a_new_categorical_reads_by_its_categories(
    duckdb_cursor, categories, ordered, result_type
):
    def rebuild(df):
        return df.assign(e=pd.Categorical(df["e"].astype(object), categories=categories, ordered=ordered))

    source = duckdb_cursor.sql(f"select * from (values ('b'::{ENUM}), (NULL), ('a'::{ENUM})) t(e)")
    mapped = source.map(rebuild)
    assert mapped.types == [duckdb.sqltype(result_type)]
    assert mapped.fetchall() == [("b",), (None,), ("a",)]


def test_map_enum_column_compares_with_its_text(duckdb_cursor):
    source = duckdb_cursor.sql(f"select * from (values ('b'::{ENUM}), (NULL), ('a'::{ENUM})) t(e)")
    mapped = source.map(lambda df: df.assign(is_b=df["e"] == "b"))
    assert mapped.types == [source.types[0], duckdb.sqltypes.BOOLEAN]
    assert mapped.fetchall() == [("b", True), (None, False), ("a", False)]


@SCHEMAS
@pytest.mark.parametrize(
    ("sql_type", "value"),
    [
        pytest.param("STRUCT(x BIGINT)", "{'x': i::BIGINT}", id="STRUCT_BIGINT_field"),
        pytest.param("MAP(INTEGER, INTEGER)", "map([i::INTEGER], [i::INTEGER])", id="MAP_INTEGER_keys"),
        pytest.param("MAP(VARCHAR, BIGINT)", "map([i::VARCHAR], [i::BIGINT])", id="MAP_BIGINT_values"),
        pytest.param("INTEGER[2]", "[i, i]::INTEGER[2]", id="ARRAY"),
        pytest.param("MAP(BLOB, INTEGER)", "map([i::VARCHAR::BLOB], [i::INTEGER])", id="MAP_BLOB_keys"),
        pytest.param("MAP(INTERVAL, INTEGER)", "map([to_days(i::INTEGER)], [i::INTEGER])", id="MAP_INTERVAL_keys"),
        pytest.param("MAP(VARCHAR, STRUCT(x INTEGER))", "map(['k'], [NULL::STRUCT(x INTEGER)])", id="MAP_NULL_values"),
        pytest.param("TIME_NS", "'01:00:00'::TIME_NS", id="TIME_NS"),
        pytest.param(
            "STRUCT(a INTEGER[], b MAP(VARCHAR, INTEGER))",
            "{'a': [i::INTEGER], 'b': map(['k'], [i::INTEGER])}",
            id="nested",
        ),
    ],
)
def test_map_identity_returns_an_object_column_in_its_own_type(duckdb_cursor, sql_type, value, explicit_schema):
    source = duckdb_cursor.sql(
        f"select (case when i = 1 then NULL else {value} end)::{sql_type} as a from range(3) t(i)"
    )
    mapped = source.map(lambda df: df, schema=_schema_of(source, explicit_schema))
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()


UNION_AB = "UNION(a INTEGER, b VARCHAR)"
UNION_OVERLAPPING = "UNION(a INTEGER, b BIGINT)"
# The function gets a UNION's member values only, so the tags cannot reach the result through it.
UNION_ROWS = pytest.mark.parametrize(
    ("sql_type", "rows", "returned", "tags_cast"),
    [
        pytest.param(
            UNION_AB,
            f"(union_value(a := 1)::{UNION_AB}), (union_value(b := 'x')::{UNION_AB}), (NULL::{UNION_AB})",
            "INTEGER",
            [("a", 1), ("b", "x"), (None, None)],
            id="both_members",
        ),
        pytest.param(
            UNION_AB,
            f"(union_value(a := 1)::{UNION_AB}), (union_value(b := NULL::VARCHAR)::{UNION_AB})",
            "INTEGER",
            [("a", 1), (None, None)],
            id="null_member",
        ),
        pytest.param(
            UNION_OVERLAPPING,
            f"(union_value(a := 1)::{UNION_OVERLAPPING}), (union_value(b := 7::BIGINT)::{UNION_OVERLAPPING})",
            "INTEGER",
            [("a", 1), ("a", 7)],
            id="overlapping_members",
        ),
    ],
)


@UNION_ROWS
def test_map_identity_of_a_union_column_is_refused(duckdb_cursor, sql_type, rows, returned, tags_cast):
    mapped = duckdb_cursor.sql(f"select * from (values {rows}) t(u)").map(lambda df: df)
    expected = f"Invalid Input Error: UDF column type mismatch, expected [{sql_type}], got [{returned}]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


@UNION_ROWS
def test_map_union_column_in_the_schema_casts_each_member_value(duckdb_cursor, sql_type, rows, returned, tags_cast):
    # The cast picks the member for each value, so a NULL member reads as a NULL UNION and 7 as the INTEGER member.
    source = duckdb_cursor.sql(f"select * from (values {rows}) t(u)")
    mapped = source.map(lambda df: df, schema={"u": duckdb.sqltype(sql_type)})
    assert mapped.types == source.types
    assert mapped.project("union_tag(u), u").fetchall() == tags_cast


@SCHEMAS
def test_map_identity_returns_a_struct_whose_field_is_null_across_a_chunk(duckdb_cursor, explicit_schema):
    duckdb_cursor.execute("set threads = 1")
    # The field holds values in the first 2048 row chunk and only NULLs in the second.
    source = duckdb_cursor.sql("select {'a': case when i < 2048 then i::INTEGER end} as s from range(4096) t(i)")
    mapped = source.map(lambda df: df, schema=_schema_of(source, explicit_schema))
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()


@SCHEMAS
def test_map_identity_returns_an_enum_column_in_its_own_type(duckdb_cursor, explicit_schema):
    source = duckdb_cursor.sql(f"select * from (values ('b'::{ENUM}), (NULL), ('a'::{ENUM})) t(e)")
    mapped = source.map(lambda df: df, schema=_schema_of(source, explicit_schema))
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()


def test_map_column_takes_the_input_type_of_its_name_when_columns_are_added_or_reordered(duckdb_cursor):
    source = duckdb_cursor.sql("select i as n, {'x': i::BIGINT} as s from range(3) t(i)")
    mapped = source.map(lambda df: df[["s", "n"]].assign(k=1))
    assert mapped.types == [source.types[1], duckdb.sqltypes.BIGINT, duckdb.sqltypes.BIGINT]
    assert mapped.fetchall() == [(s, n, 1) for n, s in source.fetchall()]


def _text_cells(values):
    def replace(df):
        return df.assign(a=pd.Series(values[: len(df)], dtype=object, index=df.index)) if len(df) else df

    return replace


MISSING_VALUES = [
    pytest.param(lambda: None, id="None"),
    pytest.param(duckdb.NullValue, id="NullValue"),
    pytest.param(lambda: np.datetime64("NaT", "us"), id="NaT"),
    pytest.param(lambda: pd.NA, id="pd.NA"),
    pytest.param(lambda: float("nan"), id="NaN"),
]


def _missing_cells(make_missing):
    def missing(df):
        return df.assign(a=pd.Series([make_missing() for _ in range(len(df))], dtype=object)) if len(df) else df

    return missing


@pytest.mark.parametrize("sql_type", ["BIGINT", "DOUBLE", "VARCHAR"])
@pytest.mark.parametrize("make_missing", MISSING_VALUES)
def test_map_object_column_of_missing_values_reads_null_in_its_schema_type(duckdb_cursor, make_missing, sql_type):
    source = duckdb_cursor.sql(f"select i::{sql_type} as a from range(3) t(i)")
    mapped = source.map(_missing_cells(make_missing), schema=_schema_of(source, explicit_schema=True))
    assert mapped.types == source.types
    assert mapped.fetchall() == [(None,), (None,), (None,)]


@pytest.mark.parametrize("make_missing", MISSING_VALUES)
def test_map_object_column_of_missing_values_keeps_an_object_input_type(duckdb_cursor, make_missing):
    source = duckdb_cursor.sql("select i::VARCHAR as a from range(3) t(i)")
    mapped = source.map(_missing_cells(make_missing))
    assert mapped.types == source.types
    assert mapped.fetchall() == [(None,), (None,), (None,)]


def _odd_cell_at(row, value):
    """Every cell of column a as its Python value, and value at row."""

    def replace(df):
        if not len(df):
            return df
        cells = df["a"].astype(object).tolist()
        cells[row] = value
        return df.assign(a=pd.Series(cells, dtype=object, index=df.index))

    return replace


# In a 2048 row chunk the analyzer samples the even rows only, so the last row is outside its sample.
ODD_ROWS = pytest.mark.parametrize("row", [0, 2047], ids=["first_row", "row_outside_the_sample"])
BIGINT_A = {"a": duckdb.sqltypes.BIGINT}


@ODD_ROWS
def test_map_object_column_cell_is_cast_into_the_bound_type(duckdb_cursor, row):
    duckdb_cursor.execute("set threads = 1")
    mapped = duckdb_cursor.sql("select i as a from range(2048) t(i)").map(_odd_cell_at(row, "5"), schema=BIGINT_A)
    assert mapped.types == [duckdb.sqltypes.BIGINT]
    assert mapped.fetchall() == [(5 if i == row else i,) for i in range(2048)]


@ODD_ROWS
def test_map_object_column_cell_the_cast_refuses_raises_the_casts_error(duckdb_cursor, row):
    duckdb_cursor.execute("set threads = 1")
    mapped = duckdb_cursor.sql("select i as a from range(2048) t(i)").map(_odd_cell_at(row, "x"), schema=BIGINT_A)
    expected = "Conversion Error: UDF column 'a': Could not convert string 'x' to INT64"
    with pytest.raises(duckdb.ConversionException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


@ODD_ROWS
def test_map_struct_cell_without_a_field_reads_it_as_null(duckdb_cursor, row):
    duckdb_cursor.execute("set threads = 1")
    source = duckdb_cursor.sql("select {'x': i::INTEGER, 'y': i::INTEGER} as a from range(2048) t(i)")
    mapped = source.map(_odd_cell_at(row, {"x": 3}))
    expected = [({"x": 3, "y": None} if i == row else {"x": i, "y": i},) for i in range(2048)]
    assert mapped.types == source.types
    assert mapped.fetchall() == expected


@ODD_ROWS
def test_map_struct_cell_without_a_field_in_a_type_holding_a_map_reads_it_as_null(duckdb_cursor, row):
    duckdb_cursor.execute("set threads = 1")
    source = duckdb_cursor.sql("select {'x': i::INTEGER, 'm': map(['k'], [i::INTEGER])} as a from range(2048) t(i)")
    mapped = source.map(_odd_cell_at(row, {"x": 3}))
    expected = [({"x": 3, "m": None} if i == row else {"x": i, "m": {"k": i}},) for i in range(2048)]
    assert mapped.types == source.types
    assert mapped.fetchall() == expected


def _apply_to_a(convert):
    return lambda df: df.assign(a=df["a"].apply(convert))


CHANGED_IN_KIND = pytest.mark.parametrize(
    ("source", "function", "sql_type", "returned"),
    [
        pytest.param(
            "{'x': i::BIGINT}",
            _apply_to_a(lambda d: {"x": str(d["x"]).zfill(5)}),
            "STRUCT(x BIGINT)",
            "STRUCT(x VARCHAR)",
            id="struct_field_turned_into_text",
        ),
        pytest.param(
            "[i::INTEGER]",
            _apply_to_a(lambda values: [str(v).zfill(5) for v in values]),
            "INTEGER[]",
            "VARCHAR[]",
            id="list_elements_turned_into_text",
        ),
        pytest.param(
            """'{"k": ' || i || '}'""",
            _apply_to_a(json.loads),
            "VARCHAR",
            "STRUCT(k INTEGER)",
            id="text_parsed_as_json",
        ),
        pytest.param(
            "i",
            _apply_to_a(lambda v: Decimal(int(v)) / 3),
            "BIGINT",
            "DECIMAL(28,28)",
            id="integers_divided_as_decimals",
        ),
        pytest.param(
            "map([i::INTEGER], [i::INTEGER])",
            _apply_to_a(lambda m: {str(k).zfill(3): v for k, v in m.items()}),
            "MAP(INTEGER, INTEGER)",
            "MAP(VARCHAR, INTEGER)",
            id="map_keys_turned_into_text",
        ),
        pytest.param(
            "{'x': i::TINYINT}",
            _apply_to_a(lambda d: {"x": d["x"] + 300}),
            "STRUCT(x TINYINT)",
            "STRUCT(x INTEGER)",
            id="struct_field_outside_its_type",
        ),
        pytest.param(
            "[i::INTEGER]",
            _apply_to_a(lambda values: np.asarray(values, dtype=np.float64) / 2),
            "INTEGER[]",
            "DOUBLE[]",
            id="list_given_as_a_float_array",
        ),
        pytest.param(
            "{'x': i::BIGINT}",
            _apply_to_a(lambda d: {"x": d["x"], "y": 1}),
            "STRUCT(x BIGINT)",
            "STRUCT(x INTEGER, y INTEGER)",
            id="struct_given_a_key_it_does_not_name",
        ),
        pytest.param(
            "{'t': TIMESTAMP '2020-01-01'}",
            _apply_to_a(lambda d: {"t": np.datetime64("2020-01-01T00:00:00.000000001", "ns")}),
            "STRUCT(t TIMESTAMP)",
            "STRUCT(t TIMESTAMP_NS)",
            id="nanosecond_datetime64_field",
        ),
        # A float of more than 15 significant digits, or one the cast into a wide scale rounds in double precision
        pytest.param(
            "{'x': i::BIGINT}",
            _apply_to_a(lambda d: {"x": 9007199254740992.0}),
            "STRUCT(x BIGINT)",
            "STRUCT(x DOUBLE)",
            id="float_of_16_digits",
        ),
        pytest.param(
            "{'x': i::DECIMAL(38,30)}",
            _apply_to_a(lambda d: {"x": 0.1}),
            "STRUCT(x DECIMAL(38,30))",
            "STRUCT(x DOUBLE)",
            id="float_the_cast_rounds_at_a_wide_scale",
        ),
        pytest.param(
            "{'x': i::BIGINT}",
            _apply_to_a(lambda d: {"x": d["x"], "X": 2}),
            "STRUCT(x BIGINT)",
            "STRUCT(x INTEGER, X INTEGER)",
            id="struct_given_two_keys_for_one_field",
        ),
    ],
)


@CHANGED_IN_KIND
def test_map_column_whose_cells_the_function_changed_in_kind_is_refused(
    duckdb_cursor, source, function, sql_type, returned
):
    mapped = duckdb_cursor.sql(f"select {source} as a from range(3) t(i)").map(function)
    expected = f"Invalid Input Error: UDF column type mismatch, expected [{sql_type}], got [{returned}]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


NEW_YORK = "SET TimeZone = 'America/New_York'"
UTC = datetime.timezone.utc


@pytest.mark.parametrize(
    ("source", "function", "sql_type", "returned"),
    [
        # df() hands the elements over as naive UTC datetimes
        pytest.param(
            "[TIMESTAMPTZ '2020-01-01 12:00:00+00']",
            lambda df: df,
            "TIMESTAMP WITH TIME ZONE[]",
            "TIMESTAMP[]",
            id="list_of_timestamptz_identity",
        ),
        pytest.param(
            "{'t': TIMESTAMP '2020-01-01 12:00:00'}",
            _apply_to_a(lambda d: {"t": datetime.datetime(2020, 1, 1, 12, tzinfo=UTC)}),
            "STRUCT(t TIMESTAMP)",
            "STRUCT(t TIMESTAMP WITH TIME ZONE)",
            id="aware_datetime_for_timestamp",
        ),
        pytest.param(
            "{'t': TIMESTAMPTZ '2020-01-01 12:00:00+00'}",
            _apply_to_a(lambda d: {"t": datetime.datetime(2020, 1, 1, 7)}),
            "STRUCT(t TIMESTAMP WITH TIME ZONE)",
            "STRUCT(t TIMESTAMP)",
            id="naive_datetime_for_timestamptz",
        ),
        pytest.param(
            "{'t': TIME '01:00:00'}",
            _apply_to_a(lambda d: {"t": datetime.time(1, tzinfo=UTC)}),
            "STRUCT(t TIME)",
            "STRUCT(t TIME WITH TIME ZONE)",
            id="aware_time_for_time",
        ),
        pytest.param(
            "{'t': TIMETZ '01:00:00+00'}",
            _apply_to_a(lambda d: {"t": datetime.time(1)}),
            "STRUCT(t TIME WITH TIME ZONE)",
            "STRUCT(t TIME)",
            id="naive_time_for_timetz",
        ),
    ],
)
def test_map_value_with_a_time_zone_and_one_without_are_of_different_kinds(
    duckdb_cursor, source, function, sql_type, returned
):
    duckdb_cursor.execute(NEW_YORK)
    mapped = duckdb_cursor.sql(f"select {source} as a from range(1) t(i)").map(function)
    expected = f"Invalid Input Error: UDF column type mismatch, expected [{sql_type}], got [{returned}]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


@pytest.mark.parametrize(
    ("source", "function", "sql_type", "returned"),
    [
        pytest.param(
            "{'t': '2020-01-01'::TIMESTAMP_S}",
            _apply_to_a(lambda d: {"t": datetime.datetime(2020, 1, 1, 0, 0, 0, 500000)}),
            "STRUCT(t TIMESTAMP_S)",
            "STRUCT(t TIMESTAMP)",
            id="milliseconds_for_seconds",
        ),
        pytest.param(
            "{'t': '2020-01-01'::TIMESTAMP_MS}",
            _apply_to_a(lambda d: {"t": datetime.datetime(2020, 1, 1, 0, 0, 0, 1)}),
            "STRUCT(t TIMESTAMP_MS)",
            "STRUCT(t TIMESTAMP)",
            id="a_microsecond_for_milliseconds",
        ),
        pytest.param(
            "{'t': TIME '01:00:00'}",
            _apply_to_a(lambda d: {"t": duckdb.Value("01:00:00.000000001", duckdb.sqltype("TIME_NS"))}),
            "STRUCT(t TIME)",
            "STRUCT(t TIME_NS)",
            id="a_nanosecond_for_microseconds",
        ),
    ],
)
def test_map_time_value_finer_than_its_column_keeps_only_values_the_unit_holds(
    duckdb_cursor, source, function, sql_type, returned
):
    mapped = duckdb_cursor.sql(f"select {source} as a from range(1) t(i)").map(function)
    expected = f"Invalid Input Error: UDF column type mismatch, expected [{sql_type}], got [{returned}]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


def test_map_time_value_outside_the_range_of_its_columns_unit_is_refused(duckdb_cursor):
    # TIMESTAMP_NS covers the years 1677 to 2262 only
    mapped = duckdb_cursor.sql("select {'t': '2020-01-01'::TIMESTAMP_NS} as a from range(1) t(i)").map(
        _apply_to_a(lambda d: {"t": datetime.datetime(1500, 1, 1)})
    )
    expected = (
        "Invalid Input Error: UDF column type mismatch, expected [STRUCT(t TIMESTAMP_NS)], got [STRUCT(t TIMESTAMP)]"
    )
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


# df() hands a LIST of HUGEINT or DECIMAL over as float64 elements, and a float of more than 15 significant digits
# may already have rounded the value it came from
@pytest.mark.parametrize(
    ("values", "sql_type"),
    [
        pytest.param("[1152921504606846977::HUGEINT]", "HUGEINT[]", id="hugeint_beyond_2_53"),
        pytest.param("[123456789012345678901::HUGEINT]", "HUGEINT[]", id="hugeint_beyond_bigint"),
        pytest.param("[1152921504606846977::HUGEINT, NULL]", "HUGEINT[]", id="hugeint_with_a_null"),
        pytest.param("[1234567890123456.78::DECIMAL(18,2)]", "DECIMAL(18,2)[]", id="decimal_of_18_digits"),
        pytest.param("[1.000000000000000001::DECIMAL(19,18)]", "DECIMAL(19,18)[]", id="decimal_of_scale_18"),
        pytest.param("[100000000000000.001::DECIMAL(18,3)]", "DECIMAL(18,3)[]", id="decimal_of_15_integer_digits"),
    ],
)
def test_map_identity_of_a_list_whose_floats_may_have_rounded_is_refused(duckdb_cursor, values, sql_type):
    mapped = duckdb_cursor.sql(f"select {values} as a").map(lambda df: df)
    expected = f"Invalid Input Error: UDF column type mismatch, expected [{sql_type}], got [DOUBLE[]]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


@pytest.mark.parametrize(
    "values",
    [
        pytest.param("[5::HUGEINT, NULL, (-999999999999999)::HUGEINT]", id="hugeint_of_15_digits"),
        pytest.param("[1.25::DECIMAL(9,2), NULL, 0.1::DECIMAL(9,2)]", id="decimal"),
        pytest.param("[1.5::DECIMAL(18,3), NULL]", id="decimal_of_scale_3"),
        pytest.param("[(-32768)::SMALLINT, NULL, 32767::SMALLINT]", id="smallint"),
        pytest.param("[(-128)::TINYINT, 127::TINYINT]", id="tinyint"),
        pytest.param("{'x': 32767::SMALLINT}", id="smallint_field"),
    ],
)
def test_map_identity_of_a_list_whose_floats_hold_their_values_round_trips(duckdb_cursor, values):
    source = duckdb_cursor.sql(f"select {values} as a")
    mapped = source.map(lambda df: df)
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()


@pytest.mark.parametrize(
    ("source", "values", "returned"),
    [
        pytest.param("[1::SMALLINT]", [40000], "INTEGER[]", id="smallint_list"),
        pytest.param("[1::TINYINT]", [-129], "INTEGER[]", id="tinyint_list"),
    ],
)
def test_map_narrow_integer_list_with_an_int_beyond_its_range_is_refused(duckdb_cursor, source, values, returned):
    mapped = duckdb_cursor.sql(f"select {source} as a").map(_apply_to_a(lambda v: values))
    sql_type = duckdb_cursor.sql(f"select {source} as a").types[0]
    expected = f"Invalid Input Error: UDF column type mismatch, expected [{sql_type}], got [{returned}]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


@pytest.mark.parametrize("cell", [0, "x"], ids=["int", "str"])
def test_map_scalar_returned_for_a_map_column_is_refused_and_the_database_stays_usable(cell):
    con = duckdb.connect()
    mapped = con.sql("select map(['k'], [1]) as a").map(
        lambda df: df.assign(a=pd.Series([cell] * len(df), dtype=object, index=df.index))
    )
    with pytest.raises(duckdb.InvalidInputException, match=re.escape("UDF column type mismatch")):
        mapped.fetchall()
    assert con.sql("select 42").fetchall() == [(42,)]


def test_map_text_of_a_map_for_a_stated_map_column_reads_as_the_cast_reads_it():
    con = duckdb.connect()
    mapped = con.sql("select map(['k'], [1]) as a").map(
        lambda df: df.assign(a=pd.Series(["{k=2}"] * len(df), dtype=object, index=df.index)),
        schema={"a": duckdb.sqltype("MAP(VARCHAR, INTEGER)")},
    )
    assert mapped.fetchall() == [({"k": 2},)]
    assert con.sql("select 42").fetchall() == [(42,)]


@pytest.mark.parametrize(
    ("value", "nanoseconds"),
    [
        pytest.param("'2020-01-01 00:00:00.000000001'::TIMESTAMP_NS", 1, id="pandas_timestamp"),
        pytest.param("NULL::TIMESTAMP_NS", None, id="null"),
    ],
)
def test_map_timestamp_ns_column_handed_back_as_objects_keeps_its_nanoseconds(duckdb_cursor, value, nanoseconds):
    source = duckdb_cursor.sql(f"select {value} as a")
    mapped = source.map(lambda df: df.assign(a=df["a"].astype(object)))
    assert mapped.types == [duckdb.sqltype("TIMESTAMP_NS")]
    assert mapped.project("a::VARCHAR").fetchall() == source.project("a::VARCHAR").fetchall()


def test_map_timestamp_ns_column_of_numpy_datetime64_cells_keeps_their_nanoseconds(duckdb_cursor):
    cell = np.datetime64("2020-01-01T00:00:00.000000002", "ns")
    mapped = duckdb_cursor.sql("select '2020-01-01'::TIMESTAMP_NS as a").map(
        lambda df: df.assign(a=pd.Series([cell] * len(df), dtype=object, index=df.index))
    )
    assert mapped.project("a::VARCHAR").fetchall() == [("2020-01-01 00:00:00.000000002",)]


def test_map_timestamp_column_given_a_timestamp_with_nanoseconds_is_refused(duckdb_cursor):
    # The cast into microseconds would drop the nanosecond
    stamp = pd.Timestamp("2020-01-01 00:00:00.000000001")
    mapped = duckdb_cursor.sql("select TIMESTAMP '2020-01-01' as a").map(
        lambda df: df.assign(a=pd.Series([stamp] * len(df), dtype=object, index=df.index))
    )
    expected = "Invalid Input Error: UDF column type mismatch, expected [TIMESTAMP], got [TIMESTAMP_NS]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


def _far_from_1970(value):
    return pytest.param(value, id=value[:4])


FAR_YEARS = pytest.mark.parametrize(
    "value", [_far_from_1970("1500-01-01"), _far_from_1970("2500-01-01"), _far_from_1970("9999-12-31")]
)


@FAR_YEARS
@pytest.mark.parametrize("sql_type", ["TIMESTAMP", "TIMESTAMP_MS", "TIMESTAMP_S"])
def test_map_timestamp_column_handed_back_as_timestamps_outside_the_nanosecond_range_round_trips(
    duckdb_cursor, sql_type, value
):
    source = duckdb_cursor.sql(f"select '{value}'::{sql_type} as a")
    mapped = source.map(lambda df: df.assign(a=df["a"].astype(object)))
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()


@FAR_YEARS
def test_map_struct_field_given_a_timestamp_outside_the_nanosecond_range_reads_it(duckdb_cursor, value):
    source = duckdb_cursor.sql(f"select {{'t': '{value}'::TIMESTAMP}} as a")
    mapped = source.map(_returning({"t": pd.Timestamp(value)}))
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()


def test_map_timestamp_column_of_2020_and_9999_handed_back_as_timestamps_round_trips(duckdb_cursor):
    source = duckdb_cursor.sql(
        "select (case when i = 0 then '2020-01-01' else '9999-12-31' end)::TIMESTAMP as a from range(2) t(i)"
    )
    mapped = source.map(lambda df: df.assign(a=df["a"].astype(object)))
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()


@SCHEMAS
def test_map_timestamp_ns_column_given_a_timestamp_beyond_its_range_is_refused(duckdb_cursor, explicit_schema):
    source = duckdb_cursor.sql("select TIMESTAMP_NS '2020-01-01' as a")
    mapped = source.map(_returning(pd.Timestamp("9999-12-31")), schema=_schema_of(source, explicit_schema))
    if explicit_schema:
        expected = "UDF column 'a': Could not convert Timestamp to higher precision"
        with pytest.raises(duckdb.ConversionException, match=re.escape(expected)):
            mapped.fetchall()
    else:
        expected = "Invalid Input Error: UDF column type mismatch, expected [TIMESTAMP_NS], got [TIMESTAMP]"
        with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
            mapped.fetchall()


@pytest.mark.parametrize(
    "value",
    ["infinity", "-infinity", "2262-04-11 23:47:16.854775806"],
    ids=["infinity", "minus_infinity", "largest_finite_nanosecond_but_one"],
)
@SCHEMAS
def test_map_timestamp_ns_identity_handed_back_as_timestamps_keeps_infinity(duckdb_cursor, value, explicit_schema):
    # df() hands an infinity over as pandas' Timestamp.max or Timestamp.min, which read back as that infinity
    source = duckdb_cursor.sql(f"select '{value}'::TIMESTAMP_NS as a")
    mapped = source.map(lambda df: df.assign(a=df["a"].astype(object)), schema=_schema_of(source, explicit_schema))
    assert mapped.types == source.types
    assert mapped.project("a::VARCHAR").fetchall() == [(value,)]


# A scalar inside a container of a type holding a MAP, which meets the engine's cast or the kind check, never a
# conversion target holding no type
NESTED_SCALARS = pytest.mark.parametrize(
    ("sql_type", "cell", "cast_into"),
    [
        pytest.param("MAP(VARCHAR, INTEGER)[]", [0], "MAP(VARCHAR, INTEGER)", id="int_in_a_list_of_maps"),
        pytest.param("MAP(VARCHAR, INTEGER)[]", ["x"], "MAP(VARCHAR, INTEGER)", id="str_in_a_list_of_maps"),
        pytest.param("STRUCT(m MAP(VARCHAR, INTEGER))", {"m": 5}, "MAP(VARCHAR, INTEGER)", id="int_for_a_map_field"),
        pytest.param("MAP(VARCHAR, INTEGER)[]", np.array([0, 1]), "MAP(VARCHAR, INTEGER)", id="int_array_for_a_list"),
        pytest.param("MAP(VARCHAR, INTEGER)[]", (0,), "MAP(VARCHAR, INTEGER)", id="tuple_for_a_list"),
        pytest.param("MAP(VARCHAR, STRUCT(x INTEGER))", {"k": "abc"}, "STRUCT(x INTEGER)", id="str_for_a_struct_value"),
        pytest.param(
            "MAP(VARCHAR, INTEGER)[][]",
            [np.array(5)],
            "MAP(VARCHAR, INTEGER)[]",
            id="zero_dimensional_array_for_a_list",
        ),
    ],
)


def _returning(cell):
    def replace(df):
        return df.assign(a=pd.Series([cell] * len(df), dtype=object, index=df.index))

    return replace


@NESTED_SCALARS
@pytest.mark.parametrize("mode", ["inferred", "explicit_schema", "append"])
def test_map_scalar_inside_a_container_of_a_type_holding_a_map_is_refused_and_the_database_stays_usable(
    sql_type, cell, cast_into, mode
):
    con = duckdb.connect()
    con.execute(f"create table t (a {sql_type})")
    con.execute("insert into t values (NULL)")
    if mode == "append":
        run = lambda: con.append("t", pd.DataFrame({"a": pd.Series([cell], dtype=object)}))  # noqa: E731
    else:
        schema = {"a": duckdb.sqltype(sql_type)} if mode == "explicit_schema" else None
        run = con.table("t").map(_returning(cell), schema=schema).fetchall
    with pytest.raises(duckdb.Error) as error:
        run()
    message = str(error.value)
    assert not isinstance(error.value, duckdb.FatalException), message
    if mode == "inferred":
        assert message.startswith(f"Invalid Input Error: UDF column type mismatch, expected [{sql_type}], got ["), (
            message
        )
    else:
        # The message names the column's own types, never a conversion target
        assert cast_into in message, message
        assert "shape" not in message, message
    assert con.sql("select 42").fetchall() == [(42,)]


def _through(mode, con, sql_type, cell):
    con.execute(f"create or replace table t (a {sql_type})")
    if mode == "append":
        con.append("t", pd.DataFrame({"a": pd.Series([cell], dtype=object)}))
    elif mode == "appender":
        with con.appender("t") as appender:
            appender.append_row(cell)
    elif mode == "parameter":
        con.execute("insert into t values (?)", [cell])
    else:
        con.execute("insert into t values (NULL)")
        schema = {"a": duckdb.sqltype(sql_type)} if mode == "explicit_schema" else None
        return con.table("t").map(_returning(cell), schema=schema).fetchall()
    return con.table("t").fetchall()


@pytest.mark.parametrize(
    ("sql_type", "cell"),
    [
        pytest.param("MAP(VARCHAR, VARCHAR)", {"key": ["a"], "value": "b"}, id="map"),
        pytest.param("STRUCT(m MAP(VARCHAR, VARCHAR))", {"m": {"key": ["a"], "value": "b"}}, id="map_field"),
    ],
)
@pytest.mark.parametrize("mode", ["inferred", "explicit_schema", "append"])
def test_map_of_a_key_list_beside_a_plain_value_for_a_map_type_is_refused_and_the_database_stays_usable(
    sql_type, cell, mode
):
    con = duckdb.connect()
    expected = (
        "A dict of 'key' and 'value' for a MAP takes a list of keys and a list of values, not VARCHAR[] and VARCHAR"
    )
    with pytest.raises(duckdb.InvalidInputException, match=re.escape(expected)):
        _through(mode, con, sql_type, cell)
    assert con.sql("select 42").fetchall() == [(42,)]


@pytest.mark.parametrize("mode", ["appender", "parameter"])
def test_map_of_a_key_list_beside_a_plain_value_without_a_target_converts_as_a_struct(mode):
    # The conversion has no MAP to read for, so the dict is the STRUCT of its fields, which the engine casts into the
    # MAP of the two entries, the list as its text
    con = duckdb.connect()
    assert _through(mode, con, "MAP(VARCHAR, VARCHAR)", {"key": ["a"], "value": "b"}) == [
        ({"key": "[a]", "value": "b"},)
    ]


@pytest.mark.parametrize(
    ("cell", "struct_type"),
    [
        pytest.param({"key": [], "value": "ab"}, 'STRUCT("key" "NULL"[], "value" VARCHAR)', id="empty_list_and_text"),
        pytest.param({"key": [1], "value": 5}, 'STRUCT("key" INTEGER[], "value" INTEGER)', id="list_and_int"),
        pytest.param({"key": "x", "value": None}, 'STRUCT("key" VARCHAR, "value" "NULL")', id="text_and_null"),
    ],
)
@pytest.mark.parametrize("mode", ["parameter", "dataframe_scan"])
def test_dict_of_key_and_value_without_two_lists_is_inferred_as_a_struct(cell, struct_type, mode):
    con = duckdb.connect()
    if mode == "parameter":
        rows = con.execute("select ?, typeof(?)", [cell, cell]).fetchall()
    else:
        frame = pd.DataFrame({"a": pd.Series([cell], dtype=object)})  # noqa: F841
        rows = con.sql("select a, typeof(a) from frame").fetchall()
    assert rows == [(cell, struct_type)]


KEY_AND_VALUE_STRUCTS = pytest.mark.parametrize(
    "value",
    [
        "{'key': 'color', 'value': ['red', 'blue']}::STRUCT(key VARCHAR, value VARCHAR[])",
        "map_entries(MAP {'color': ['red', 'blue']})",
        "{'key': [1], 'value': 5}::STRUCT(key INTEGER[], value INTEGER)",
        "{'key': 1, 'value': [5]}::STRUCT(key INTEGER, value INTEGER[])",
    ],
    ids=["text_and_list", "map_entries", "list_and_int", "int_and_list"],
)


@KEY_AND_VALUE_STRUCTS
@pytest.mark.parametrize("mode", ["inferred", "explicit_schema", "append", "appender", "parameter", "dataframe_scan"])
def test_struct_with_fields_key_and_value_one_a_list_converts_as_its_struct(value, mode):
    con = duckdb.connect()
    source = con.sql(f"select {value} as a")
    expected = source.fetchall()
    frame = source.df()
    if mode == "dataframe_scan":
        assert duckdb.connect().sql("select a from frame").fetchall() == expected
        return
    if mode in ("inferred", "explicit_schema"):
        schema = _schema_of(source, mode == "explicit_schema")
        assert source.map(lambda df: df, schema=schema).fetchall() == expected
        return
    con.execute(f"create table t (a {source.types[0]})")
    if mode == "append":
        con.append("t", frame)
    elif mode == "appender":
        with con.appender("t") as appender:
            appender.append_row(frame["a"][0])
    else:
        con.execute("insert into t values (?)", [frame["a"][0]])
    assert con.table("t").fetchall() == expected


@pytest.mark.parametrize(
    "value",
    [
        "{'key': NULL, 'value': NULL}::STRUCT(key VARCHAR, value VARCHAR)",
        "{'key': [1], 'value': NULL}::STRUCT(key INTEGER[], value INTEGER)",
    ],
    ids=["both_null", "list_and_null"],
)
@pytest.mark.parametrize("mode", ["inferred", "explicit_schema", "append"])
def test_declared_struct_with_fields_key_and_value_reads_a_null_field_as_its_struct(value, mode):
    # A declared STRUCT names its fields, so such a dict is no NULL MAP of key and value lists
    con = duckdb.connect()
    source = con.sql(f"select {value} as a")
    expected = source.fetchall()
    if mode == "append":
        con.execute(f"create table t (a {source.types[0]})")
        con.append("t", source.df())
        assert con.table("t").fetchall() == expected
    else:
        mapped = source.map(lambda df: df, schema=_schema_of(source, mode == "explicit_schema"))
        assert mapped.fetchall() == expected


def test_map_of_an_ordered_dict_for_a_stated_map_keeps_its_iteration_order(duckdb_cursor):
    ordered = collections.OrderedDict([("a", 1), ("b", 2), ("c", 3)])
    ordered.move_to_end("a")
    source = duckdb_cursor.sql("select MAP {'k': 1} as a")
    mapped = source.map(_returning(ordered), schema={"a": duckdb.sqltype("MAP(VARCHAR, INTEGER)")})
    assert mapped.project("map_keys(a)").fetchall() == [(["b", "c", "a"],)]


@pytest.mark.parametrize("mode", ["inferred", "explicit_schema", "append"])
def test_struct_given_an_empty_dict_is_refused(mode):
    # A missing field reads NULL, but a dict of no keys is no STRUCT of the column's fields
    con = duckdb.connect()
    with pytest.raises(duckdb.TypeMismatchException, match="Cannot cast STRUCTs of different size"):
        _through(mode, con, "STRUCT(a INTEGER, b INTEGER)", {})
    assert con.sql("select 42").fetchall() == [(42,)]


@pytest.mark.parametrize("mode", ["inferred", "explicit_schema", "append", "appender", "parameter"])
def test_map_of_key_and_value_arrays_of_no_dimensions_reads_their_two_entries(mode):
    # An array of no dimensions holds a single value, so the dict is a plain one
    con = duckdb.connect()
    cell = {"key": np.array("a"), "value": np.array("b")}
    assert _through(mode, con, "MAP(VARCHAR, VARCHAR)", cell) == [({"key": "a", "value": "b"},)]


@pytest.mark.parametrize(
    "value",
    [
        "MAP {'key': 'ab', 'value': 'cd'}",
        "MAP {'key': 'ab', 'value': 'cde'}",
        "MAP {'key': 1, 'value': 2}",
        "MAP {'key': 'x', 'value': NULL}",
        "MAP {'key': NULL, 'value': 'y'}",
    ],
    ids=["text_of_one_length", "text_of_two_lengths", "ints", "null_value", "null_key_entry"],
)
@SCHEMAS
def test_map_identity_of_a_map_keyed_by_key_and_value_reads_its_entries(duckdb_cursor, value, explicit_schema):
    # df() hands such a MAP over as the dict of its two entries, which no list makes a dict of key and value lists
    source = duckdb_cursor.sql(f"select {value} as a")
    mapped = source.map(lambda df: df, schema=_schema_of(source, explicit_schema))
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()


@pytest.mark.parametrize(
    "value", ["MAP {'key': NULL, 'value': NULL}", "MAP {'key': [1], 'value': NULL}"], ids=["both_null", "list_and_null"]
)
@SCHEMAS
def test_map_identity_of_a_map_keyed_by_key_and_value_of_a_null_beside_a_list_or_a_null_reads_null(
    duckdb_cursor, value, explicit_schema
):
    # df() hands it over as {"key": None, "value": None} or {"key": [1], "value": None}, the forms of a NULL MAP
    source = duckdb_cursor.sql(f"select {value} as a")
    mapped = source.map(lambda df: df, schema=_schema_of(source, explicit_schema))
    assert mapped.types == source.types
    assert mapped.fetchall() == [(None,)]


@pytest.mark.parametrize("mode", ["append", "appender", "parameter"])
def test_map_of_a_null_key_entry_beside_a_plain_value_reads_its_two_entries(mode):
    cell = {"key": None, "value": 5}
    con = duckdb.connect()
    con.execute("create table t (a MAP(VARCHAR, INTEGER))")
    if mode == "append":
        con.append("t", pd.DataFrame({"a": pd.Series([cell], dtype=object)}))
    elif mode == "appender":
        with con.appender("t") as appender:
            appender.append_row(cell)
    else:
        con.execute("insert into t values (?)", [cell])
    assert con.table("t").fetchall() == [({"key": None, "value": 5},)]


@pytest.mark.parametrize(
    "cell",
    [{"key": None, "value": None}, {"key": ["a"], "value": None}, {"key": None, "value": ["a"]}],
    ids=["both_none", "values_none", "keys_none"],
)
@pytest.mark.parametrize("mode", ["inferred", "explicit_schema", "append"])
def test_map_of_key_and_value_with_a_none_reads_a_null_map(cell, mode):
    con = duckdb.connect()
    con.execute("create table t (a MAP(VARCHAR, VARCHAR))")
    if mode == "append":
        con.append("t", pd.DataFrame({"a": pd.Series([cell], dtype=object)}))
        assert con.table("t").fetchall() == [(None,)]
    else:
        con.execute("insert into t values (MAP {'k': 'v'})")
        schema = {"a": duckdb.sqltype("MAP(VARCHAR, VARCHAR)")} if mode == "explicit_schema" else None
        assert con.table("t").map(_returning(cell), schema=schema).fetchall() == [(None,)]


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        pytest.param({"key": ["k"], "value": [5]}, "{k=5}", id="list_of_an_int"),
        pytest.param({"key": ("k",), "value": (5,)}, "{k=5}", id="tuple_of_an_int"),
        pytest.param({"key": np.array(["k"]), "value": np.array([5])}, "{k=5}", id="numpy_arrays"),
        pytest.param({"key": ["a", "b"], "value": [2**60 + 1, 1]}, "{a=1152921504606846977, b=1}", id="wide_ints"),
    ],
)
@SCHEMAS
def test_map_of_key_and_value_lists_reads_each_entry_as_a_dicts_entry_reads(
    duckdb_cursor, cell, expected, explicit_schema
):
    source = duckdb_cursor.sql("select MAP {'k': 1::BIGINT} as a")
    mapped = source.map(_returning(cell), schema=_schema_of(source, explicit_schema))
    assert mapped.types == source.types
    assert mapped.project("a::VARCHAR").fetchall() == [(expected,)]


def test_map_cast_error_for_a_nested_scalar_names_the_columns_type():
    con = duckdb.connect()
    con.execute("create table t (a MAP(VARCHAR, INTEGER)[])")
    expected = "Conversion Error: Unimplemented type for cast (INTEGER -> MAP(VARCHAR, INTEGER))"
    with pytest.raises(duckdb.ConversionException, match=re.escape(expected)):
        con.append("t", pd.DataFrame({"a": pd.Series([[0]], dtype=object)}))
    assert con.sql("select 42").fetchall() == [(42,)]


NANOSECOND = pd.Timestamp("2020-01-01 00:00:00.000000001")


@pytest.mark.parametrize(
    ("value", "cell"),
    [
        pytest.param("{'t': '2020-01-01'::TIMESTAMP_NS}", {"t": NANOSECOND}, id="struct_field"),
        pytest.param("['2020-01-01'::TIMESTAMP_NS]", [NANOSECOND], id="list_element"),
    ],
)
def test_map_nested_timestamp_ns_value_keeps_its_nanoseconds(duckdb_cursor, value, cell):
    source = duckdb_cursor.sql(f"select {value} as a")
    mapped = source.map(_returning(cell))
    assert mapped.types == source.types
    assert "00:00:00.000000001" in mapped.project("a::VARCHAR").fetchall()[0][0]


AWARE_NANOSECOND = pd.Timestamp("2020-01-01 00:00:00.000000001", tz="UTC")


@pytest.mark.parametrize(
    ("value", "cell"),
    [
        pytest.param("'2020-01-01 00:00:00.000000001+00'::TIMESTAMPTZ_NS", AWARE_NANOSECOND, id="top"),
        pytest.param("{'t': '2020-01-01 00:00:00.000000001+00'::TIMESTAMPTZ_NS}", {"t": AWARE_NANOSECOND}, id="field"),
        pytest.param("['2020-01-01 00:00:00.000000001+00'::TIMESTAMPTZ_NS]", [AWARE_NANOSECOND], id="list_element"),
    ],
)
@SCHEMAS
def test_map_zoned_timestamp_ns_value_keeps_its_nanoseconds(duckdb_cursor, value, cell, explicit_schema):
    duckdb_cursor.execute("SET TimeZone = 'UTC'")
    source = duckdb_cursor.sql(f"select {value} as a")
    mapped = source.map(_returning(cell), schema=_schema_of(source, explicit_schema))
    assert mapped.types == source.types
    assert mapped.project("a::VARCHAR").fetchall() == source.project("a::VARCHAR").fetchall()


@pytest.mark.parametrize(
    ("value", "cell", "returned"),
    [
        pytest.param("'2020-01-01 00:00:00+00'::TIMESTAMPTZ", AWARE_NANOSECOND, "TIMESTAMPTZ_NS", id="top"),
        pytest.param(
            "{'t': '2020-01-01 00:00:00+00'::TIMESTAMPTZ}",
            {"t": AWARE_NANOSECOND},
            "STRUCT(t TIMESTAMPTZ_NS)",
            id="field",
        ),
        pytest.param(
            "['2020-01-01 00:00:00+00'::TIMESTAMPTZ]", [AWARE_NANOSECOND], "TIMESTAMPTZ_NS[]", id="list_element"
        ),
    ],
)
def test_map_zoned_timestamp_with_nanoseconds_for_an_implied_timestamptz_is_refused(
    duckdb_cursor, value, cell, returned
):
    duckdb_cursor.execute("SET TimeZone = 'UTC'")
    source = duckdb_cursor.sql(f"select {value} as a")
    expected = f"UDF column type mismatch, expected [{source.types[0]}], got [{returned}]"
    with pytest.raises(duckdb.InvalidInputException, match=re.escape(expected)):
        source.map(_returning(cell)).fetchall()


@pytest.mark.parametrize("place", ["top", "field", "list_element", "map_value"])
@pytest.mark.parametrize("mode", ["explicit_schema", "append"])
def test_map_zoned_timestamp_with_nanoseconds_for_a_stated_timestamptz_reads_as_its_datetime(place, mode):
    stamp = pd.Timestamp("2020-01-01 00:00:00.000005789", tz="UTC")
    text = "2020-01-01 00:00:00.000005+00"
    sql_type, cell, expected = {
        "top": ("TIMESTAMPTZ", stamp, text),
        "field": ("STRUCT(t TIMESTAMPTZ)", {"t": stamp}, f"{{'t': '{text}'}}"),
        "list_element": ("TIMESTAMPTZ[]", [stamp], f"['{text}']"),
        "map_value": ("MAP(VARCHAR, TIMESTAMPTZ)", {"k": stamp}, f"{{k='{text}'}}"),
    }[place]
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"create table t (a {sql_type})")
    if mode == "append":
        con.append("t", pd.DataFrame({"a": pd.Series([cell], dtype=object)}))
        rows = con.sql("select a::VARCHAR from t").fetchall()
    else:
        con.execute("insert into t values (NULL)")
        schema = {"a": duckdb.sqltype(sql_type)}
        rows = con.table("t").map(_returning(cell), schema=schema).project("a::VARCHAR").fetchall()
    assert rows == [(expected,)]


def test_map_nested_timestamp_with_nanoseconds_for_an_implied_timestamp_is_refused(duckdb_cursor):
    mapped = duckdb_cursor.sql("select {'t': TIMESTAMP '2020-01-01'} as a").map(_returning({"t": NANOSECOND}))
    expected = (
        "Invalid Input Error: UDF column type mismatch, expected [STRUCT(t TIMESTAMP)], got [STRUCT(t TIMESTAMP_NS)]"
    )
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


FIVE_MICROSECONDS = np.datetime64("2020-01-01T00:00:00.000005123", "ns")


@pytest.mark.parametrize(
    ("sql_type", "cell", "expected"),
    [
        pytest.param(
            "STRUCT(t TIMESTAMP)",
            {"t": FIVE_MICROSECONDS},
            "{'t': '2020-01-01 00:00:00.000005'}",
            id="datetime64_in_a_struct",
        ),
        pytest.param(
            "STRUCT(t TIMESTAMP)",
            {"t": pd.Timestamp(FIVE_MICROSECONDS)},
            "{'t': '2020-01-01 00:00:00.000005'}",
            id="pandas_timestamp_in_a_struct",
        ),
        pytest.param("TIMESTAMP[]", [FIVE_MICROSECONDS], "['2020-01-01 00:00:00.000005']", id="datetime64_in_a_list"),
        pytest.param(
            "MAP(VARCHAR, TIMESTAMP)",
            {"k": FIVE_MICROSECONDS},
            "{k='2020-01-01 00:00:00.000005'}",
            id="datetime64_as_a_map_value",
        ),
        pytest.param(
            "TIMESTAMP_MS[]",
            [np.datetime64("2020-01-01T00:00:00.005123", "ns")],
            "['2020-01-01 00:00:00.005']",
            id="datetime64_in_a_list_of_milliseconds",
        ),
    ],
)
@pytest.mark.parametrize("mode", ["explicit_schema", "append"])
def test_map_nested_timestamp_with_nanoseconds_for_a_stated_coarser_unit_reads_at_microseconds_as_at_the_top(
    sql_type, cell, expected, mode
):
    con = duckdb.connect()
    con.execute(f"create table t (a {sql_type})")
    if mode == "append":
        con.append("t", pd.DataFrame({"a": pd.Series([cell], dtype=object)}))
        rows = con.sql("select a::VARCHAR from t").fetchall()
    else:
        con.execute("insert into t values (NULL)")
        schema = {"a": duckdb.sqltype(sql_type)}
        rows = con.table("t").map(_returning(cell), schema=schema).project("a::VARCHAR").fetchall()
    assert rows == [(expected,)]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(
            np.datetime64("2020-01-01T00:00:00.000005789", "ns"),
            "2020-01-01 00:00:00.000006",
            id="datetime64_rounds_up",
        ),
        pytest.param(np.datetime64(-1, "ns"), "1970-01-01 00:00:00", id="datetime64_before_the_epoch"),
        pytest.param(
            pd.Timestamp("2020-01-01 00:00:00.000005789"), "2020-01-01 00:00:00.000005", id="timestamp_as_its_datetime"
        ),
        pytest.param(pd.Timestamp(-1, unit="ns"), "1969-12-31 23:59:59.999999", id="timestamp_before_the_epoch"),
    ],
)
@pytest.mark.parametrize("place", ["top", "list_element", "struct_field", "map_value"])
@pytest.mark.parametrize("mode", ["explicit_schema", "append"])
def test_map_timestamp_finer_than_a_stated_coarser_unit_reads_at_every_depth_as_at_the_top(value, text, place, mode):
    # A datetime64 is a TIMESTAMP_NS the engine's cast rounds into microseconds; a Timestamp is the datetime it is
    sql_type, cell, expected = {
        "top": ("TIMESTAMP", value, text),
        "list_element": ("TIMESTAMP[]", [value], f"['{text}']"),
        "struct_field": ("STRUCT(t TIMESTAMP)", {"t": value}, f"{{'t': '{text}'}}"),
        "map_value": ("MAP(VARCHAR, TIMESTAMP)", {"k": value}, f"{{k='{text}'}}"),
    }[place]
    con = duckdb.connect()
    con.execute(f"create table t (a {sql_type})")
    if mode == "append":
        con.append("t", pd.DataFrame({"a": pd.Series([cell], dtype=object)}))
        rows = con.sql("select a::VARCHAR from t").fetchall()
    else:
        con.execute("insert into t values (NULL)")
        schema = {"a": duckdb.sqltype(sql_type)}
        rows = con.table("t").map(_returning(cell), schema=schema).project("a::VARCHAR").fetchall()
    assert rows == [(expected,)]


@pytest.mark.parametrize(
    "sql_type",
    ["STRUCT(a INTEGER, b INTEGER)", "STRUCT(a INTEGER, t TIMESTAMP)", "STRUCT(a INTEGER, m MAP(VARCHAR, INTEGER))"],
    ids=["plain_fields", "timestamp_field", "map_field"],
)
@pytest.mark.parametrize("depth", ["top", "struct_field", "list_element"])
@pytest.mark.parametrize("mode", ["inferred", "explicit_schema", "append"])
def test_map_tuple_reads_as_a_list_at_every_depth_so_a_struct_refuses_it(sql_type, depth, mode):
    column_type, cell = {
        "top": (sql_type, (3, 4)),
        "struct_field": (f"STRUCT(s {sql_type})", {"s": (3, 4)}),
        "list_element": (f"{sql_type}[]", [(3, 4)]),
    }[depth]
    con = duckdb.connect()
    con.execute(f"create table t (a {column_type})")
    con.execute("insert into t values (NULL)")
    if mode == "append":
        run = lambda: con.append("t", pd.DataFrame({"a": pd.Series([cell], dtype=object)}))  # noqa: E731
    else:
        schema = {"a": duckdb.sqltype(column_type)} if mode == "explicit_schema" else None
        run = con.table("t").map(_returning(cell), schema=schema).fetchall
    if mode == "inferred":
        expected = f"UDF column type mismatch, expected [{column_type}], got ["
        with pytest.raises(duckdb.InvalidInputException, match=re.escape(expected)):
            run()
    else:
        with pytest.raises(duckdb.ConversionException, match=re.escape(f"(INTEGER[] -> {sql_type})")):
            run()
    assert con.sql("select 42").fetchall() == [(42,)]


def test_map_decimal_column_handed_back_as_objects_of_18_digits_is_refused(duckdb_cursor):
    # df() hands a DECIMAL column over as float64, which rounds 1234567890123456.78
    source = duckdb_cursor.sql("select 1234567890123456.78::DECIMAL(18,2) as a")
    mapped = source.map(lambda df: df.assign(a=df["a"].astype(object)))
    expected = "Invalid Input Error: UDF column type mismatch, expected [DECIMAL(18,2)], got [DOUBLE]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


@pytest.mark.parametrize(
    ("source", "cell", "expected"),
    [
        pytest.param(
            "{'t': '2020-01-01'::TIMESTAMP_S}",
            datetime.datetime(2020, 1, 1, 0, 0, 3),
            datetime.datetime(2020, 1, 1, 0, 0, 3),
            id="whole_seconds_into_seconds",
        ),
        pytest.param(
            "{'t': '2020-01-01'::TIMESTAMP_NS}",
            datetime.datetime(2020, 1, 1, 0, 0, 0, 7),
            datetime.datetime(2020, 1, 1, 0, 0, 0, 7),
            id="microseconds_into_nanoseconds",
        ),
        pytest.param(
            "{'t': TIME '01:00:00'}",
            duckdb.Value("01:00:00.000001000", duckdb.sqltype("TIME_NS")),
            datetime.time(1, 0, 0, 1),
            id="whole_microseconds_into_microseconds",
        ),
        pytest.param(
            "{'t': '01:00:00'::TIME_NS}",
            datetime.time(2, 0, 0, 5),
            datetime.time(2, 0, 0, 5),
            id="time_into_nanoseconds",
        ),
    ],
)
def test_map_time_value_its_column_unit_holds_reads_unchanged(duckdb_cursor, source, cell, expected):
    relation = duckdb_cursor.sql(f"select {source} as a from range(1) t(i)")
    mapped = relation.map(_apply_to_a(lambda d: {"t": cell}))
    assert mapped.types == relation.types
    assert mapped.fetchall() == [({"t": expected},)]


@pytest.mark.parametrize(
    ("source", "function", "expected"),
    [
        pytest.param("[1, 2]", _apply_to_a(lambda values: [1, True]), [1, 1], id="int_and_bool_list"),
        pytest.param(
            "map(['a'], ['x'])", _apply_to_a(lambda m: {"a": 1, "b": "x"}), {"a": "1", "b": "x"}, id="int_and_str_map"
        ),
    ],
)
def test_map_container_of_mixed_elements_reads_as_the_conversion_types_it(duckdb_cursor, source, function, expected):
    # The conversion types a list or a MAP by the common type of its elements; 2.0.0 refused these
    relation = duckdb_cursor.sql(f"select {source} as a from range(1) t(i)")
    mapped = relation.map(function)
    assert mapped.types == relation.types
    assert mapped.fetchall() == [(expected,)]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("{'t': TIMESTAMPTZ '2020-01-01 12:00:00+00'}", id="struct_field"),
        pytest.param("map(['k'], [TIMESTAMPTZ '2020-01-01 12:00:00+00'])", id="map_value"),
        pytest.param("{'t': TIMETZ '01:00:00+02'}", id="timetz_field"),
    ],
)
def test_map_identity_of_a_zoned_value_keeps_its_instant_in_another_session_time_zone(duckdb_cursor, value):
    duckdb_cursor.execute(NEW_YORK)
    source = duckdb_cursor.sql(f"select {value} as a")
    mapped = source.map(lambda df: df)
    assert mapped.types == source.types
    assert mapped.project("a::VARCHAR").fetchall() == source.project("a::VARCHAR").fetchall()


@SCHEMAS
def test_map_numpy_scalar_inside_a_struct_raises_the_conversions_error(duckdb_cursor, explicit_schema):
    source = duckdb_cursor.sql("select {'x': i::INTEGER} as a from range(1) t(i)")
    mapped = source.map(_apply_to_a(lambda d: {"x": np.int64(5)}), schema=_schema_of(source, explicit_schema))
    expected = (
        "Not implemented Error: UDF column 'a': Unable to transform python value of type "
        "'<class 'numpy.int64'>' to DuckDB LogicalType"
    )
    with pytest.raises(duckdb.NotImplementedException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


def test_map_column_whose_cells_changed_in_kind_casts_into_the_schema_type(duckdb_cursor):
    mapped = duckdb_cursor.sql("select i as a from range(3) t(i)").map(
        _apply_to_a(lambda v: Decimal(int(v)) / 3), schema=BIGINT_A
    )
    assert mapped.fetchall() == [(0,), (0,), (1,)]


@pytest.mark.parametrize(
    ("source", "function", "expected"),
    [
        pytest.param("{'x': i::TINYINT}", _apply_to_a(lambda d: {"x": d["x"] + 100}), {"x": 100}, id="int_that_fits"),
        pytest.param("[i::DOUBLE]", _apply_to_a(lambda values: [int(v) for v in values]), [0.0], id="int_into_double"),
        pytest.param("{'x': i::FLOAT}", _apply_to_a(lambda d: {"x": d["x"] / 2}), {"x": 0.0}, id="float_into_float"),
        pytest.param(
            "{'x': i::DECIMAL(9,2)}", _apply_to_a(lambda d: {"x": Decimal("1.5")}), {"x": Decimal("1.50")}, id="decimal"
        ),
        pytest.param(
            "{'x': i::BIGINT}", _apply_to_a(lambda d: {"X": d["x"]}), {"x": 0}, id="field_named_in_other_case"
        ),
        pytest.param(
            "map([i::INTEGER], [i::INTEGER])",
            _apply_to_a(lambda m: {k + 5: v for k, v in m.items()}),
            {5: 0},
            id="map_keys_of_their_kind",
        ),
        pytest.param(
            "[i::INTEGER]", _apply_to_a(lambda values: np.asarray(values, dtype=np.int64) + 3), [3], id="int64_array"
        ),
        pytest.param(
            "[i::DOUBLE]", _apply_to_a(lambda values: np.asarray(values, dtype=np.float32)), [0.0], id="float32_array"
        ),
        pytest.param("{'x': i::BIGINT}", _apply_to_a(lambda d: {"x": Decimal(5)}), {"x": 5}, id="integral_decimal"),
        pytest.param("{'x': i::BIGINT}", _apply_to_a(lambda d: {"x": 2.0}), {"x": 2}, id="integral_float_for_bigint"),
        pytest.param(
            "{'t': TIMESTAMP '2020-01-01'}",
            _apply_to_a(lambda d: {"t": np.datetime64("2020-01-02T00:00:00", "ns")}),
            {"t": datetime.datetime(2020, 1, 2)},
            id="nanosecond_datetime64_of_whole_microseconds",
        ),
        pytest.param(
            "{'x': i::DECIMAL(18,2)}",
            _apply_to_a(lambda d: {"x": 1234567890123.45}),
            {"x": Decimal("1234567890123.45")},
            id="float_of_15_digits",
        ),
        pytest.param(
            "{'x': i::DECIMAL(9,2)}", _apply_to_a(lambda d: {"x": 2.0}), {"x": Decimal("2.00")}, id="integral_float"
        ),
    ],
)
def test_map_column_whose_cells_keep_their_kind_reads_into_the_input_type(duckdb_cursor, source, function, expected):
    relation = duckdb_cursor.sql(f"select {source} as a from range(1) t(i)")
    mapped = relation.map(function)
    assert mapped.types == relation.types
    assert mapped.fetchall() == [(expected,)]


def test_map_object_column_of_zero_dimensional_arrays_reads_their_values(duckdb_cursor):
    def arrays(df):
        cells = np.empty(len(df), dtype=object)
        for i, v in enumerate(df["a"]):
            cells[i] = np.array(int(v), dtype=object)
        return df.assign(a=pd.Series(cells, dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select i as a from range(3) t(i)").map(arrays, schema={"a": duckdb.sqltypes.BIGINT})
    assert mapped.fetchall() == [(0,), (1,), (2,)]


def test_map_declared_column_read_error_names_the_column(duckdb_cursor):
    source = duckdb_cursor.sql("select i as n, {'x': i::INTEGER} as s from range(3) t(i)")
    mapped = source.map(lambda df: df.assign(n=df["s"]), schema=_schema_of(source, explicit_schema=True))
    expected = "Conversion Error: UDF column 'n': Unimplemented type for cast (STRUCT(x INTEGER) -> BIGINT)"
    with pytest.raises(duckdb.ConversionException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        # A text cell and cells of other kinds in one vector, the NULL among them
        pytest.param(["a", 5, 6, "b", None, 7, "c"], ["a", "5", "6", "b", None, "7", "c"], id="str_int_and_null"),
        pytest.param(["a", 5, np.str_("c"), None], ["a", "5", "c", None], id="with_a_str_subclass"),
    ],
)
def test_map_text_column_mixing_str_with_other_cells_reads_each_as_text_in_the_schema_type(
    duckdb_cursor, cells, expected
):
    source = duckdb_cursor.sql(f"select i::VARCHAR as a from range({len(cells)}) t(i)")
    mapped = source.map(_text_cells(cells), schema={"a": duckdb.sqltypes.VARCHAR})
    assert mapped.fetchall() == [(value,) for value in expected]


@SCHEMAS
def test_map_text_list_mixing_str_with_an_int_reads_as_text_as_2_0_0_reads_it(duckdb_cursor, explicit_schema):
    # A list converts by the common type of its elements, so the int reads as its text inside a text list
    source = duckdb_cursor.sql("select ['x', 'y'] as a from range(1) t(i)")
    mapped = source.map(
        _apply_to_a(lambda values: np.array(["x", 5], dtype=object)), schema=_schema_of(source, explicit_schema)
    )
    assert mapped.types == source.types
    assert mapped.fetchall() == [(["x", "5"],)]


def test_map_text_column_of_str_subclass_cells_reads_them_without_a_schema(duckdb_cursor):
    mapped = duckdb_cursor.sql("select i::VARCHAR as a from range(3) t(i)").map(_text_cells(["a", np.str_("c"), None]))
    assert mapped.fetchall() == [("a",), ("c",), (None,)]


def test_map_text_column_with_a_cell_of_another_kind_is_refused_without_a_schema(duckdb_cursor):
    mapped = duckdb_cursor.sql("select i::VARCHAR as a from range(4) t(i)").map(
        _text_cells(["a", np.str_("c"), 5, None])
    )
    expected = "Invalid Input Error: UDF column type mismatch, expected [VARCHAR], got [INTEGER]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


@SCHEMAS
def test_map_blob_column_mixing_bytes_with_other_cells_reads_each(duckdb_cursor, explicit_schema):
    # Every Python type a BLOB value reads from, in one vector
    cells = [b"a", bytearray(b"b"), None, memoryview(b"c"), b"\x00d"]
    source = duckdb_cursor.sql(f"select i::VARCHAR::BLOB as a from range({len(cells)}) t(i)")
    mapped = source.map(
        lambda df: df.assign(a=pd.Series(cells[: len(df)], dtype=object, index=df.index)),
        schema=_schema_of(source, explicit_schema),
    )
    assert mapped.fetchall() == [(b"a",), (b"b",), (None,), (b"c",), (b"\x00d",)]


@pytest.mark.parametrize(
    ("cell", "name"),
    [pytest.param(Fraction(1, 3), "fractions.Fraction", id="Fraction"), pytest.param({"b"}, "set", id="set")],
)
def test_map_text_column_with_an_object_without_a_duckdb_type_is_refused_without_a_schema(duckdb_cursor, cell, name):
    mapped = duckdb_cursor.sql("select i::VARCHAR as a from range(2) t(i)").map(_text_cells(["a", cell]))
    expected = (
        f"Not implemented Error: UDF column 'a': Unable to transform python value of type '<class '{name}'>' to "
        "DuckDB LogicalType"
    )
    with pytest.raises(duckdb.NotImplementedException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


def test_map_text_column_with_an_object_without_a_duckdb_type_reads_its_text_in_the_schema_type(duckdb_cursor):
    source = duckdb_cursor.sql("select i::VARCHAR as a from range(2) t(i)")
    mapped = source.map(_text_cells(["a", Fraction(1, 3)]), schema={"a": duckdb.sqltypes.VARCHAR})
    assert mapped.fetchall() == [("a",), ("1/3",)]


@SCHEMAS
@pytest.mark.parametrize(
    ("value", "function", "sql_type", "returned"),
    [
        pytest.param("'101'::BIT", lambda df: df, "BIT", "BLOB", id="column"),
        pytest.param("['101'::BIT]", lambda df: df, "BIT[]", "BLOB[]", id="list_element"),
        pytest.param(
            "'101'::BIT",
            lambda df: df.assign(a=pd.Series([memoryview(bytes(v)) for v in df["a"]], dtype=object, index=df.index)),
            "BIT",
            "BLOB",
            id="memoryview",
        ),
    ],
)
def test_map_bit_value_reaching_the_function_as_its_storage_bytes_is_refused(
    duckdb_cursor, explicit_schema, value, function, sql_type, returned
):
    source = duckdb_cursor.sql(f"select {value} as a from range(2) t(i)")
    mapped = source.map(function, schema=_schema_of(source, explicit_schema))
    expected = f"Invalid Input Error: UDF column type mismatch, expected [{sql_type}], got [{returned}]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


@SCHEMAS
def test_map_time_column_of_times_with_and_without_a_zone(duckdb_cursor, explicit_schema):
    source = duckdb_cursor.sql(
        "select (case when i = 1 then NULL else TIME '01:02:03.000004' end) as a from range(3) t(i)"
    )
    schema = _schema_of(source, explicit_schema)
    mapped = source.map(lambda df: df, schema=schema)
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()
    zone = datetime.timezone(datetime.timedelta(hours=2))
    zoned = source.map(
        lambda df: df.assign(a=pd.Series([datetime.time(1, tzinfo=zone)] * len(df), dtype=object, index=df.index)),
        schema=schema,
    )
    if explicit_schema:
        # The cast of a TIMETZ into TIME keeps the wall time and drops the offset
        assert zoned.fetchall() == [(datetime.time(1),)] * 3
    else:
        expected = "Invalid Input Error: UDF column type mismatch, expected [TIME], got [TIME WITH TIME ZONE]"
        with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
            zoned.fetchall()


def test_map_json_column_still_validates_its_text(duckdb_cursor):
    schema = {"a": duckdb.sqltype("JSON")}
    good = duckdb_cursor.sql("select 1 as a").map(_text_cells(['{"k": 1}']), schema=schema)
    assert good.fetchall() == [('{"k": 1}',)]
    bad = duckdb_cursor.sql("select 1 as a").map(_text_cells(["not json"]), schema=schema)
    expected = (
        "Conversion Error: UDF column 'a': Malformed JSON at byte 0 of input: invalid literal.  Input: \"not json\""
    )
    with pytest.raises(duckdb.ConversionException, match=f"^{re.escape(expected)}$"):
        bad.fetchall()


@pytest.mark.parametrize(
    ("projection", "names"),
    [
        pytest.param("col0 as a, col1 as a", ["a", "a_1"], id="one_name"),
        pytest.param('col0 as a, col1 as "A"', ["a", "A_1"], id="names_differing_in_case"),
    ],
)
def test_map_identity_keeps_both_columns_of_one_name(duckdb_cursor, projection, names):
    source = duckdb_cursor.values([1, "x"]).project(projection)
    seen = []

    def record(df):
        seen.append(list(df.columns))
        return df

    mapped = source.map(record)
    assert seen[0] == names == list(source.df().columns)
    assert mapped.types == source.types
    assert mapped.fetchall() == source.fetchall()


def _with_list_column(df):
    return df.assign(s=pd.Series([[v] for v in df["a"]], dtype=object, index=df.index))


def _with_text_column(df):
    # An object column, as pandas 2 gives this expression on every frame and pandas 3 without pyarrow on the empty one.
    return df.assign(s=pd.Series(("prefix " + df["a"].astype(str)).tolist(), dtype=object, index=df.index))


NEW_COLUMNS = pytest.mark.parametrize(
    ("function", "sql_type", "values"),
    [
        pytest.param(_with_list_column, "BIGINT[]", [[0], [1], [2]], id="list"),
        pytest.param(_with_text_column, "VARCHAR", ["prefix 0", "prefix 1", "prefix 2"], id="text"),
    ],
)


@NEW_COLUMNS
def test_map_new_object_column_reads_into_the_schema_type(duckdb_cursor, function, sql_type, values):
    schema = {"a": duckdb.sqltypes.BIGINT, "s": duckdb.sqltype(sql_type)}
    mapped = duckdb_cursor.sql("select i as a from range(3) t(i)").map(function, schema=schema)
    assert mapped.types == [duckdb.sqltypes.BIGINT, duckdb.sqltype(sql_type)]
    assert mapped.fetchall() == list(enumerate(values))


@NEW_COLUMNS
def test_map_new_object_column_the_bind_frame_cannot_type_names_the_column_and_schema(
    duckdb_cursor, function, sql_type, values
):
    mapped = duckdb_cursor.sql("select i as a from range(3) t(i)").map(function)
    with pytest.raises(duckdb.InvalidInputException, match=_no_type("s")):
        mapped.fetchall()


def _no_type(column):
    message = f"UDF column '{column}' has no type: the function returned no values for it at bind time; pass its type"
    return f"^{re.escape(f'Invalid Input Error: {message} with schema=')}$"


def test_map_new_object_column_with_a_cell_the_analyzer_cannot_read_names_the_column_and_schema(duckdb_cursor):
    def array_cell(df):
        cells = np.empty(len(df), dtype=object)
        if len(df):
            cells[-1] = np.array(1, dtype=object)
        return df.assign(s=pd.Series(cells, dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select i as a from range(2048) t(i)").map(array_cell)
    with pytest.raises(duckdb.InvalidInputException, match=_no_type("s")):
        mapped.fetchall()


@pytest.mark.parametrize(
    ("function", "types"),
    [
        pytest.param(
            lambda df: df.assign(a=pd.Series(df["a"].astype(str).str.zfill(5).tolist(), dtype=object, index=df.index)),
            ("BIGINT, VARCHAR", "VARCHAR, VARCHAR"),
            id="typed_input_turned_into_text",
        ),
        # rename keeps the order, so the BIGINT column comes first under the name b
        pytest.param(
            lambda df: df.rename(columns={"a": "b", "b": "a"}),
            ("BIGINT, BIGINT", "BIGINT, VARCHAR"),
            id="names_swapped",
        ),
    ],
)
def test_map_typed_input_column_the_function_moved_to_another_type_is_refused(duckdb_cursor, function, types):
    mapped = duckdb_cursor.sql("select i as a, i::VARCHAR as b from range(3) t(i)").map(function)
    expected = f"Invalid Input Error: UDF column type mismatch, expected [{types[0]}], got [{types[1]}]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


@pytest.mark.parametrize(
    ("sql_type", "offset"),
    [
        pytest.param("INTEGER", 0, id="INTEGER"),
        # Python ints in the INTEGER range for a BIGINT input
        pytest.param("BIGINT", 0, id="BIGINT_holding_INTEGER_values"),
        pytest.param("BIGINT", 2**40, id="BIGINT_holding_BIGINT_values"),
        pytest.param("HUGEINT", 0, id="HUGEINT_holding_INTEGER_values"),
    ],
)
@pytest.mark.parametrize("only_none", [False, True], ids=["python_ints", "only_none"])
def test_map_typed_input_column_the_function_turned_into_objects_keeps_a_confirmed_type(
    duckdb_cursor, sql_type, offset, only_none
):
    def objects(df):
        return df.assign(a=pd.Series([None if only_none else int(v) for v in df["a"]], dtype=object, index=df.index))

    source = duckdb_cursor.sql(f"select (i + {offset})::{sql_type} as a from range(3) t(i)")
    mapped = source.map(objects)
    assert mapped.types == source.types
    assert mapped.fetchall() == [(None if only_none else i + offset,) for i in range(3)]


@ODD_ROWS
def test_map_typed_input_column_turned_into_objects_reads_a_wide_value_at_any_row(duckdb_cursor, row):
    duckdb_cursor.execute("set threads = 1")

    def objects(df):
        return df.assign(a=pd.Series([2**40 if i == row else int(v) for i, v in enumerate(df["a"])], dtype=object))

    mapped = duckdb_cursor.sql("select i as a from range(2048) t(i)").map(objects)
    assert mapped.types == [duckdb.sqltypes.BIGINT]
    assert mapped.fetchall() == [(2**40 if i == row else i,) for i in range(2048)]


@pytest.mark.parametrize("sampled", [0, 2**40], ids=["small_sample", "wide_sample"])
# In a 2048 row chunk the analyzer samples the even rows, so row 6 is inside its sample and row 7 outside it
@pytest.mark.parametrize("row", [6, 7], ids=["sampled_row", "unsampled_row"])
@pytest.mark.parametrize(
    ("cell", "outcome"), [pytest.param(2.0, 2, id="integral_float"), pytest.param(2.5, None, id="fractional_float")]
)
def test_map_typed_input_column_turned_into_objects_reads_a_float_the_same_whatever_the_sample(
    duckdb_cursor, sampled, row, cell, outcome
):
    duckdb_cursor.execute("set threads = 1")

    def objects(df):
        values = [sampled if i == 0 else cell if i == row else int(v) for i, v in enumerate(df["a"])]
        return df.assign(a=pd.Series(values, dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select i as a from range(2048) t(i)").map(objects)
    if outcome is None:
        expected = "Invalid Input Error: UDF column type mismatch, expected [BIGINT], got [DOUBLE]"
        with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
            mapped.fetchall()
    else:
        assert mapped.fetchall() == [(sampled if i == 0 else outcome if i == row else i,) for i in range(2048)]


@pytest.mark.parametrize(
    ("sql_type", "dtype", "values"),
    [
        pytest.param("BIGINT", np.int32, [0, 1, 2], id="int32_chunk_for_bigint"),
        pytest.param("DOUBLE", np.float32, [0.5, 1.5, 2.5], id="float32_chunk_for_double"),
    ],
)
def test_map_typed_chunk_for_a_guess_widens_into_it(duckdb_cursor, sql_type, dtype, values):
    # A Series built from no values is object, so the bind frame's column takes the input's type as a guess
    def typed_chunk(df):
        cells = np.asarray(df["a"], dtype=dtype) if len(df) else []
        return pd.DataFrame({"a": pd.Series(cells, dtype=None if len(df) else object)})

    source = duckdb_cursor.sql(f"select (i + {values[0]})::{sql_type} as a from range(3) t(i)")
    mapped = source.map(typed_chunk)
    assert mapped.types == source.types
    assert mapped.fetchall() == [(v,) for v in values]


def test_map_double_input_column_turned_into_float_values_reads_them_as_double(duckdb_cursor):
    def floats(df):
        cells = [duckdb.Value(float(v), duckdb.sqltypes.FLOAT) for v in df["a"]]
        return df.assign(a=pd.Series(cells, dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select (i + 0.5)::DOUBLE as a from range(3) t(i)").map(floats)
    assert mapped.types == [duckdb.sqltypes.DOUBLE]
    assert mapped.fetchall() == [(0.5,), (1.5,), (2.5,)]


@pytest.mark.parametrize(("offset", "fits"), [(0, True), (40000, False)], ids=["values_it_holds", "values_beyond_it"])
def test_map_typed_input_column_turned_into_objects_of_a_narrow_type_reads_the_values_it_holds(
    duckdb_cursor, offset, fits
):
    def objects(df):
        return df.assign(a=pd.Series([int(v) + offset for v in df["a"]], dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select i::SMALLINT as a from range(3) t(i)").map(objects)
    if fits:
        assert mapped.types == [duckdb.sqltypes.SMALLINT]
        assert mapped.fetchall() == [(0,), (1,), (2,)]
    else:
        expected = "Invalid Input Error: UDF column type mismatch, expected [SMALLINT], got [INTEGER]"
        with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
            mapped.fetchall()


def test_map_new_object_column_named_otherwise_after_bind_reports_the_name_mismatch(duckdb_cursor):
    def late_name(df):
        name = "t" if len(df) else "s"
        return df.assign(**{name: pd.Series(["x"] * len(df), dtype=object, index=df.index)})

    mapped = duckdb_cursor.sql("select i as a from range(3) t(i)").map(late_name)
    expected = "Invalid Input Error: UDF column name mismatch, expected [a, s], got [a, t]"
    with pytest.raises(duckdb.InvalidInputException, match=f"^{re.escape(expected)}$"):
        mapped.fetchall()


def test_map_column_filled_from_a_null_typed_input_names_the_column_and_schema(duckdb_cursor):
    def fill(df):
        return df.assign(c=pd.Series(["x"] * len(df), dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select i as a, NULL as c from range(3) t(i)").map(fill)
    with pytest.raises(duckdb.InvalidInputException, match=_no_type("c")):
        mapped.fetchall()


def test_map_column_a_first_map_added_as_none_and_a_second_fills_names_the_column_and_schema(duckdb_cursor):
    def add_none(df):
        return df.assign(c=pd.Series([None] * len(df), dtype=object, index=df.index))

    def fill(df):
        return df.assign(c=pd.Series([1] * len(df), dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select i as a from range(3) t(i)").map(add_none).map(fill)
    with pytest.raises(duckdb.InvalidInputException, match=_no_type("c")):
        mapped.fetchall()


def test_map_identity_of_a_null_typed_input_column_reads_null(duckdb_cursor):
    mapped = duckdb_cursor.sql("select i as a, NULL as c from range(3) t(i)").map(lambda df: df)
    assert mapped.types == [duckdb.sqltypes.BIGINT, duckdb.sqltypes.SQLNULL]
    assert mapped.fetchall() == [(0, None), (1, None), (2, None)]


def test_map_new_object_column_fails_on_the_first_chunk_with_a_value(duckdb_cursor):
    duckdb_cursor.execute("set threads = 1")
    rows_per_call = []

    def late_values(df):
        rows_per_call.append(len(df))
        values = [None if i < 2048 else str(i) for i in df["a"]]
        return df.assign(s=pd.Series(values, dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select i as a from range(4096) t(i)").map(late_values)
    with pytest.raises(duckdb.InvalidInputException, match=_no_type("s")):
        mapped.fetchall()
    # The first chunk, of only None, was read before the second one failed.
    assert [rows for rows in rows_per_call if rows] == [2048, 2048]


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param(None, id="None"),
        pytest.param(float("nan"), id="NaN"),
        pytest.param(pd.NA, id="pd.NA"),
        pytest.param(pd.NaT, id="pandas_NaT"),
        pytest.param(np.datetime64("NaT", "us"), id="numpy_NaT"),
        # pandas reads neither as missing; the conversion reads both as NULL
        pytest.param(duckdb.NullValue(), id="duckdb_NullValue"),
        pytest.param(np.ma.masked, id="numpy_masked"),
    ],
)
def test_map_new_object_column_of_only_missing_values_stays_null(duckdb_cursor, missing):
    def note(df):
        return df.assign(note=pd.Series([missing] * len(df), dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select i as a from range(3) t(i)").map(note)
    assert mapped.types == [duckdb.sqltypes.BIGINT, duckdb.sqltypes.SQLNULL]
    assert mapped.fetchall() == [(0, None), (1, None), (2, None)]

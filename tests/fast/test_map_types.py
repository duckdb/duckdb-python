"""map() fixes one type per output column at bind and reads each chunk's returned column into it."""

import re

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
        pytest.param("UNION(a INTEGER, b VARCHAR)", "i::INTEGER::UNION(a INTEGER, b VARCHAR)", id="UNION"),
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
        # Every staged value is an int, so one cast reads them; the str rows between them are written directly.
        pytest.param(["a", 5, 6, "b", None, 7, "c"], ["a", "5", "6", "b", None, "7", "c"], id="str_int_and_null"),
        pytest.param(["a", 5, np.str_("c"), None], ["a", "5", "c", None], id="with_a_str_subclass"),
    ],
)
def test_map_text_column_mixing_str_with_other_cells_reads_each_as_text(duckdb_cursor, cells, expected):
    source = duckdb_cursor.sql(f"select i::VARCHAR as a from range({len(cells)}) t(i)")
    mapped = source.map(_text_cells(cells))
    assert mapped.fetchall() == [(value,) for value in expected]


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
    ("function", "column"),
    [
        pytest.param(
            lambda df: df.assign(a=pd.Series(df["a"].astype(str).str.zfill(5).tolist(), dtype=object, index=df.index)),
            "a",
            id="typed_input_turned_into_text",
        ),
        pytest.param(lambda df: df.rename(columns={"a": "b", "b": "a"}), "a", id="names_swapped"),
    ],
)
def test_map_typed_input_column_the_function_moved_to_another_type_is_refused(duckdb_cursor, function, column):
    mapped = duckdb_cursor.sql("select i as a, i::VARCHAR as b from range(3) t(i)").map(function)
    with pytest.raises(duckdb.InvalidInputException, match=_no_type(column)):
        mapped.fetchall()


@pytest.mark.parametrize(
    ("make_cell", "expected"),
    [
        pytest.param(int, [(0,), (1,), (2,)], id="python_ints"),
        pytest.param(lambda v: None, [(None,), (None,), (None,)], id="only_none"),
    ],
)
def test_map_typed_input_column_the_function_turned_into_objects_keeps_a_confirmed_type(
    duckdb_cursor, make_cell, expected
):
    def objects(df):
        return df.assign(a=pd.Series([make_cell(v) for v in df["a"]], dtype=object, index=df.index))

    source = duckdb_cursor.sql("select i::INTEGER as a from range(3) t(i)")
    mapped = source.map(objects)
    assert mapped.types == source.types
    assert mapped.fetchall() == expected


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
    ],
)
def test_map_new_object_column_of_only_missing_values_stays_null(duckdb_cursor, missing):
    def note(df):
        return df.assign(note=pd.Series([missing] * len(df), dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select i as a from range(3) t(i)").map(note)
    assert mapped.types == [duckdb.sqltypes.BIGINT, duckdb.sqltypes.SQLNULL]
    assert mapped.fetchall() == [(0, None), (1, None), (2, None)]


def test_map_new_object_column_of_null_values_is_refused_as_untyped(duckdb_cursor):
    # pandas does not read a duckdb.NullValue() as missing, so the column holds values bind could not type
    def nulls(df):
        return df.assign(s=pd.Series([duckdb.NullValue() for _ in range(len(df))], dtype=object, index=df.index))

    mapped = duckdb_cursor.sql("select i as a from range(3) t(i)").map(nulls)
    with pytest.raises(duckdb.InvalidInputException, match=_no_type("s")):
        mapped.fetchall()

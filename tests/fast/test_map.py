import gc
import itertools
import json
import re
import subprocess
import sys
from datetime import date, timedelta
from typing import NoReturn

import pytest
from conftest import PANDAS_GE_3

import duckdb
import numpy as np
import pandas as pd


# column count differs from bind
def evil1(df):
    if len(df) == 0:
        return df["col0"].to_frame()
    else:
        return df


class TestMap:
    def test_map_does_not_leak_input_dataframe(self, duckdb_cursor):
        """Each chunk's input DataFrame is released once the query is fetched."""
        import gc
        import weakref

        refs: list[weakref.ref] = []

        def capture(df):
            refs.append(weakref.ref(df))
            # Return a fresh, unrelated frame so the OUTPUT never references the input df.
            return pd.DataFrame({"col0": [len(df)]})

        # > STANDARD_VECTOR_SIZE (2048) rows -> several chunks -> several FunctionCall invocations.
        rel = duckdb_cursor.sql("SELECT i AS col0 FROM range(20000) t(i)")
        rel.map(capture, schema={"col0": int}).fetchall()

        gc.collect()
        assert len(refs) >= 2, f"expected multiple chunks, got {len(refs)}"
        alive = sum(1 for r in refs if r() is not None)
        assert alive == 0, f"{alive}/{len(refs)} per-chunk input DataFrames leaked"

    def test_map_function_takes_no_sql_arguments(self, duckdb_cursor):
        # The callable travels through the bind input, never through a value SQL text can forge
        with pytest.raises(duckdb.BinderException, match="requires a subquery as its TABLE parameter"):
            duckdb_cursor.sql("select * from python_map_function(1, 2, 3)")
        with pytest.raises(duckdb.BinderException, match="requires a callable bind input"):
            duckdb_cursor.sql("select * from python_map_function((select 1))")

    def test_mapped_view_outlives_the_relation(self, duckdb_cursor):
        def double(df):
            return df.assign(i=df["i"] * 2)

        rel = duckdb_cursor.sql("select 21 as i").map(double)
        rel.create_view("mapped")
        del rel
        gc.collect()
        assert duckdb_cursor.sql("select i from mapped").fetchall() == [(42,)]

    def test_evil_map(self, duckdb_cursor):
        testrel = duckdb.values([1, 2])
        rel = testrel.map(evil1, schema={"i": str})
        with pytest.raises(duckdb.InvalidInputException, match="Expected 1 columns from UDF, got 2"):
            rel.df()

    def test_map(self, duckdb_cursor):
        testrel = duckdb.values([1, 2])
        conn = duckdb_cursor
        conn.execute("CREATE TABLE t (a integer)")
        empty_rel = conn.table("t")

        testrel.map(lambda df: df["col0"].add(42).to_frame())
        testrel.map(lambda df: df["col0"].astype("string").to_frame())
        testrel.map(lambda df: df)

        # column type differs from bind
        def evil2(df):
            result = df.copy(deep=True)
            if len(result) == 0:
                result["col0"] = result["col0"].astype("double")
            return result

        # column name differs from bind
        def evil3(df):
            if len(df) == 0:
                df = df.rename(columns={"col0": "col42"})
            return df

        # does not return a df
        def evil4(df) -> int:
            return 42

        # straight up throws exception
        def evil5(df) -> NoReturn:
            raise TypeError

        def return_dataframe(df):
            return pd.DataFrame({"A": [1]})

        def return_big_dataframe(df):
            return pd.DataFrame({"A": [1] * 5000})

        def return_none(df) -> None:
            return None

        def return_empty_df(df):
            return pd.DataFrame()

        with pytest.raises(duckdb.InvalidInputException, match="Expected 1 columns from UDF, got 2"):
            print(testrel.map(evil1).df())

        with pytest.raises(duckdb.InvalidInputException, match="UDF column type mismatch"):
            print(testrel.map(evil2).df())

        with pytest.raises(duckdb.InvalidInputException, match="UDF column name mismatch"):
            print(testrel.map(evil3).df())

        with pytest.raises(
            duckdb.InvalidInputException, match=r"Expected the UDF to return an object of type 'pandas\.DataFrame'"
        ):
            print(testrel.map(evil4).df())

        with pytest.raises(duckdb.InvalidInputException):
            print(testrel.map(evil5).df())

        # not a function
        with pytest.raises(TypeError):
            print(testrel.map(42).df())

        # nothing passed to map
        with pytest.raises(TypeError):
            print(testrel.map().df())

        testrel.map(return_dataframe).df().equals(pd.DataFrame({"A": [1]}))

        with pytest.raises(
            duckdb.InvalidInputException, match="UDF returned more than 2048 rows, which is not allowed"
        ):
            testrel.map(return_big_dataframe).df()

        empty_rel.map(return_dataframe).df().equals(pd.DataFrame({"A": []}))

        with pytest.raises(duckdb.InvalidInputException, match="No return value from Python function"):
            testrel.map(return_none).df()

        with pytest.raises(duckdb.InvalidInputException, match="Need a DataFrame with at least one column"):
            testrel.map(return_empty_df).df()

    def test_map_with_object_column(self, duckdb_cursor):
        def return_with_no_modification(df):
            return df

        # BLOB maps to 'object'
        # when a dataframe with 'object' column is returned, we use the content to infer the type
        # when the dataframe is empty, this results in NULL, which is not desirable
        # in this case we assume the returned type should be the same as the input type
        duckdb_cursor.values([b"1234"]).map(return_with_no_modification).fetchall()

    def test_isse_3237(self, duckdb_cursor):
        def process(rel):
            def mapper(x):
                dates = x["date"].to_numpy("datetime64[us]")
                days = x["days_to_add"].to_numpy("int")
                x["result1"] = pd.Series(
                    [pd.to_datetime(y[0]).date() + timedelta(days=y[1].item()) for y in zip(dates, days, strict=False)],
                    dtype="datetime64[us]",
                )
                x["result2"] = pd.Series(
                    [
                        pd.to_datetime(y[0]).date() + timedelta(days=-y[1].item())
                        for y in zip(dates, days, strict=False)
                    ],
                    dtype="datetime64[us]",
                )
                return x

            rel = rel.map(mapper)
            rel = rel.project("*, datediff('day', date, result1) as one")
            rel = rel.project("*, datediff('day', date, result2) as two")
            rel = rel.project("*, IF(ABS(one) > ABS(two), one, two) as three")
            return rel

        df = pd.DataFrame(
            {"date": pd.Series([date(2000, 1, 1), date(2000, 1, 2)], dtype="datetime64[us]"), "days_to_add": [1, 2]}
        )
        rel = duckdb.from_df(df)
        rel = process(rel)
        x = rel.fetchdf()
        assert x["days_to_add"].to_numpy()[0] == 1

    def test_explicit_schema(self):
        def cast_to_string(df):
            df["i"] = df["i"].astype(str)
            return df

        con = duckdb.connect()
        rel = con.sql("select i from range (10) tbl(i)")
        assert rel.types[0] == duckdb.NUMBER
        mapped_rel = rel.map(cast_to_string, schema={"i": str})
        assert mapped_rel.types[0] == duckdb.STRING

    def test_explicit_schema_returntype_mismatch(self):
        def does_nothing(df):
            return df

        con = duckdb.connect()
        rel = con.sql("select i from range(10) tbl(i)")
        # expects the mapper to return a string column
        rel = rel.map(does_nothing, schema={"i": str})
        with pytest.raises(
            duckdb.InvalidInputException, match=re.escape("UDF column type mismatch, expected [VARCHAR], got [BIGINT]")
        ):
            rel.fetchall()

    def test_explicit_schema_name_mismatch(self):
        def renames_column(df):
            return pd.DataFrame({"a": df["i"]})

        con = duckdb.connect()
        rel = con.sql("select i from range(10) tbl(i)")
        rel = rel.map(renames_column, schema={"i": int})
        with pytest.raises(duckdb.InvalidInputException, match=re.escape("UDF column name mismatch")):
            rel.fetchall()

    def test_explicit_schema_error(self):
        def no_op(df):
            return df

        con = duckdb.connect()
        rel = con.sql("select 42")
        with pytest.raises(
            duckdb.InvalidInputException,
            match=re.escape("Invalid Input Error: 'schema' should be given as a Dict[str, DuckDBType]"),
        ):
            rel.map(no_op, schema=[int])

    def test_returns_non_dataframe(self):
        def returns_series(df):
            return df.loc[:, "i"]

        con = duckdb.connect()
        rel = con.sql("select i, i as j from range(10) tbl(i)")
        with pytest.raises(
            duckdb.InvalidInputException,
            match=r"Expected the UDF to return an object of type 'pandas\.DataFrame', found "
            r"'<class 'pandas\.(core\.series\.)?Series'>' instead",
        ):
            rel = rel.map(returns_series)

    def test_explicit_schema_columncount_mismatch(self):
        def returns_subset(df):
            return pd.DataFrame({"i": df.loc[:, "i"]})

        con = duckdb.connect()
        rel = con.sql("select i, i as j from range(10) tbl(i)")
        rel = rel.map(returns_subset, schema={"i": int, "j": int})
        with pytest.raises(
            duckdb.InvalidInputException, match="Invalid Input Error: Expected 2 columns from UDF, got 1"
        ):
            rel.fetchall()

    def test_pyarrow_df(self):
        # PyArrow backed dataframes only exist on pandas >= 2.0.0
        pytest.importorskip("pandas", "2.0.0")

        def basic_function(df):
            # Create a pyarrow backed dataframe
            df = pd.DataFrame({"a": [5, 3, 2, 1, 2]}).convert_dtypes(dtype_backend="pyarrow")
            return df

        con = duckdb.connect()
        with pytest.raises(duckdb.InvalidInputException):
            con.sql("select 42").map(basic_function)


def _assert_connection_answers(connection):
    assert connection.sql("select 42").fetchall() == [(42,)]
    assert connection.cursor().sql("select 42").fetchall() == [(42,)]


def _labelled(*labels):
    """A function returning one int64 column per label, the k-th holding the input column x times k + 1."""

    def build(df):
        x = df["x"].to_numpy(dtype=np.int64)
        return pd.DataFrame({k: x * (k + 1) for k in range(len(labels))}).set_axis(list(labels), axis=1)

    return build


def _rows_of(labels):
    return [tuple(i * (k + 1) for k in range(len(labels))) for i in range(3)]


COLUMN_NAME_CASES = [
    pytest.param(["a", "a"], ["a", "a_1"], id="duplicate"),
    pytest.param(["a", "a_1", "a"], ["a", "a_1", "a_2"], id="duplicate_next_to_its_suffix"),
    pytest.param(["a", "A"], ["a", "A_1"], id="duplicate_up_to_case"),
    pytest.param([1, "1"], ["1", "1_1"], id="int_and_its_text"),
    pytest.param([b"k"], ["b'k'"], id="bytes"),
    pytest.param([1], ["1"], id="int"),
    pytest.param([("a", "b")], ["('a', 'b')"], id="tuple"),
]


@pytest.mark.parametrize(("labels", "names"), COLUMN_NAME_CASES)
def test_map_result_column_names_are_str_of_labels_deduplicated(duckdb_cursor, labels, names):
    source = duckdb_cursor.sql("select range::BIGINT as x from range(3)")
    mapped = source.map(_labelled(*labels))
    assert mapped.columns == names
    assert mapped.fetchall() == _rows_of(labels)
    _assert_connection_answers(duckdb_cursor)


@pytest.mark.parametrize(
    "labels", [pytest.param(case.values[0], id=case.id, marks=case.marks) for case in COLUMN_NAME_CASES]
)
def test_map_result_column_names_equal_those_of_from_df(duckdb_cursor, labels):
    function = _labelled(*labels)
    from_df_names = duckdb_cursor.from_df(function(pd.DataFrame({"x": np.arange(3, dtype=np.int64)}))).columns
    mapped = duckdb_cursor.sql("select range::BIGINT as x from range(3)").map(function)
    assert mapped.columns == from_df_names
    _assert_connection_answers(duckdb_cursor)


@pytest.mark.parametrize(
    ("labels", "schema_names", "names"),
    [
        pytest.param(
            ["a", "a"],
            ["a", "a_1"],
            ["a", "a_1"],
            id="duplicate_labels_under_their_new_names",
        ),
        pytest.param(
            [b"k"],
            [b"k"],
            ["b'k'"],
            id="bytes_key_and_bytes_label",
        ),
        pytest.param(
            ["a", "A"],
            ["a", "A"],
            ["a", "A_1"],
            id="keys_duplicate_up_to_case",
        ),
        pytest.param(
            [1, "1"],
            [1, "1"],
            ["1", "1_1"],
            id="keys_duplicate_after_str",
        ),
    ],
)
def test_map_explicit_schema_names_are_str_of_keys_deduplicated(duckdb_cursor, labels, schema_names, names):
    source = duckdb_cursor.sql("select range::BIGINT as x from range(3)")
    schema = dict.fromkeys(schema_names, duckdb.sqltypes.BIGINT)
    mapped = source.map(_labelled(*labels), schema=schema)
    assert mapped.columns == names
    assert mapped.fetchall() == _rows_of(labels)
    _assert_connection_answers(duckdb_cursor)


def test_map_explicit_schema_str_key_does_not_match_a_bytes_label(duckdb_cursor):
    mapped = duckdb_cursor.sql("select range::BIGINT as x from range(3)").map(
        _labelled(b"k"), schema={"k": duckdb.sqltypes.BIGINT}
    )
    mismatch = re.escape("UDF column name mismatch, expected [k], got [b'k']")
    with pytest.raises(duckdb.InvalidInputException, match=mismatch):
        mapped.fetchall()
    _assert_connection_answers(duckdb_cursor)


def _replace_last_row_of_second_chunk(make_value):
    def replace(df):
        values = df["a"].astype(object).tolist()
        # Within a 2048 row chunk the analyzer samples even rows only, so the last row is outside the sample.
        if len(df) == 2048 and values[0] >= 2048:
            values[-1] = make_value(values[-1])
        return pd.DataFrame({"a": pd.Series(values, dtype=object)})

    return replace


MAP_SCHEMAS = pytest.mark.parametrize(
    "schema", [None, {"a": duckdb.sqltypes.INTEGER}], ids=["inferred", "explicit_schema"]
)


@MAP_SCHEMAS
@pytest.mark.xfail(
    strict=True, reason="a fractional value outside the chunk's sample is converted with loss into INTEGER"
)
def test_map_fractional_value_outside_the_chunk_sample_raises(duckdb_cursor, schema):
    duckdb_cursor.execute("set threads = 1")
    mapped = duckdb_cursor.sql("select i::INTEGER as a from range(4096) t(i)").map(
        _replace_last_row_of_second_chunk(lambda _: 2.5), schema=schema
    )
    with pytest.raises(duckdb.InvalidInputException) as error:
        mapped.fetchall()
    message = str(error.value)
    assert re.search(r"\b2047\b", message)
    assert "4095" not in message
    assert "float" in message
    assert "INTEGER" in message
    _assert_connection_answers(duckdb_cursor)


@MAP_SCHEMAS
def test_map_integral_float_outside_the_chunk_sample_converts(duckdb_cursor, schema):
    duckdb_cursor.execute("set threads = 1")
    mapped = duckdb_cursor.sql("select i::INTEGER as a from range(4096) t(i)").map(
        _replace_last_row_of_second_chunk(float), schema=schema
    )
    assert mapped.fetchall() == [(i,) for i in range(4096)]
    assert mapped.types == [duckdb.sqltypes.INTEGER]


@pytest.mark.parametrize("schema_given", [True, False], ids=["explicit_schema", "inferred"])
@pytest.mark.parametrize(
    ("make_value", "sql_type", "literal"),
    [
        pytest.param(
            lambda: duckdb.Value([1, 2, 3], duckdb.sqltype("INTEGER[3]")),
            duckdb.sqltypes.VARCHAR,
            "[1, 2, 3]::INTEGER[3]::VARCHAR",
            id="fixed_size_array_as_text",
        ),
        pytest.param(
            lambda: duckdb.DoubleValue(2.5), duckdb.sqltypes.DOUBLE, "2.5::DOUBLE", id="double_as_its_own_type"
        ),
    ],
)
def test_map_value_elements_keep_the_database(duckdb_cursor, make_value, sql_type, literal, schema_given):
    expected = duckdb_cursor.sql(f"select {literal} from range(2)").fetchall()

    def two_value_cells(df):
        return pd.DataFrame({"a": pd.Series([make_value(), make_value()], dtype=object)})

    schema = {"a": sql_type} if schema_given else None
    mapped = duckdb_cursor.sql("select 1 as a").map(two_value_cells, schema=schema)
    assert mapped.fetchall() == expected
    assert mapped.types == [sql_type]
    _assert_connection_answers(duckdb_cursor)


@pytest.mark.parametrize("rows", [slice(None, None, 2), slice(1, None, 2)], ids=["even_rows", "odd_rows"])
@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        pytest.param("Int64", [1, None, None, 4, 5, 6], id="Int64"),
        pytest.param("boolean", [True, None, None, True, False, True], id="boolean"),
        pytest.param("Float64", [1.5, None, None, 4.5, 5.5, 6.5], id="Float64"),
    ],
)
def test_map_sliced_nullable_column_keeps_its_mask(duckdb_cursor, dtype, values, rows):
    # Each slice keeps one missing value and drops another, so a mask read at the wrong stride moves a NULL.
    sliced = pd.DataFrame({"a": pd.array(values, dtype=dtype)}).iloc[rows]
    expected = [(None if pd.isna(v) else v,) for v in sliced["a"].tolist()]
    mapped = duckdb_cursor.sql("select 1 as x").map(lambda df: sliced)
    assert mapped.fetchall() == expected


needs_pandas_3 = pytest.mark.skipif(not PANDAS_GE_3, reason="before pandas 3 the str dtype is object")


@pytest.mark.parametrize("chunks", [1, 2], ids=["one_arrow_chunk", "two_arrow_chunks"])
@pytest.mark.parametrize("dtype", [pytest.param("str", marks=needs_pandas_3), "string[pyarrow]"])
def test_map_str_column_is_read_as_arrow(duckdb_cursor, monkeypatch, dtype, chunks):
    pytest.importorskip("pyarrow")
    arrow_string_array = pd.arrays.ArrowStringArray

    returned = []
    converted = []
    to_numpy = arrow_string_array.to_numpy

    def recording_to_numpy(self, *args, **kwargs):
        if any(self is array for array in returned):
            converted.append(len(self))
        return to_numpy(self, *args, **kwargs)

    def exclaim(df):
        text = df["s"].astype(object) + "!"
        bounds = np.linspace(0, len(text), chunks + 1, dtype=np.int64)
        parts = [text.iloc[start:stop].astype(dtype) for start, stop in itertools.pairwise(bounds)]
        out = pd.DataFrame({"s": pd.concat(parts, ignore_index=True)})
        if len(out):
            returned.append(out["s"].array)
        return out

    mapped = duckdb_cursor.sql("select 'v' || i::VARCHAR as s from range(4096) t(i)").map(exclaim)
    monkeypatch.setattr(arrow_string_array, "to_numpy", recording_to_numpy)
    rows = mapped.fetchall()
    monkeypatch.undo()
    shapes = [(type(array), len(array), array.__arrow_array__().num_chunks) for array in returned]
    assert shapes == [(arrow_string_array, 2048, chunks)] * 2
    assert converted == []
    assert mapped.types == [duckdb.sqltypes.VARCHAR]
    assert rows == [(f"v{i}!",) for i in range(4096)]


_RESIZED_COLUMN_CHILD = """
import json
import re
import sys

import numpy as np
import pandas as pd

import duckdb

if sys.platform != "win32":
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def column(size):
    if KIND == "str":
        return pd.array([f"v{i}" for i in range(size)], dtype="str")
    if KIND == "nullable":
        return pd.array(np.arange(size, dtype="int64"), dtype="Int64")
    return np.arange(size, dtype="int64")


class ResizedFrame(pd.DataFrame):
    def __getitem__(self, key):
        if key == "resized" and len(self.index):
            return pd.Series(column(LENGTH))
        return super().__getitem__(key)


def resize(df):
    kept = df["x"].to_numpy(dtype=np.int64)
    if KIND != "nullable":
        return ResizedFrame({"kept": kept, "resized": column(len(kept))})
    resized = column(len(kept))
    if len(kept):
        resized._mask = np.zeros(LENGTH, dtype=bool)
    return pd.DataFrame({"kept": kept, "resized": resized}, copy=False)


con = duckdb.connect()
resized_type = duckdb.sqltypes.VARCHAR if KIND == "str" else duckdb.sqltypes.BIGINT
schema = {"kept": duckdb.sqltypes.BIGINT, "resized": resized_type} if EXPLICIT_SCHEMA else None
mapped = con.sql("select range::BIGINT as x from range(ROWS)").map(resize, schema=schema)
try:
    outcome = {"rows": len(mapped.fetchall())}
except duckdb.Error as error:
    outcome = {"error": type(error).__name__, "words": sorted(set(re.findall(r"\\w+", str(error))))}
outcome["answers"] = [con.sql("select 42").fetchall(), con.cursor().sql("select 42").fetchall()]
print("outcome " + json.dumps(outcome))
"""

RESIZED_CHILD_TIMEOUT = 60


@pytest.mark.timeout(2 * RESIZED_CHILD_TIMEOUT)
@pytest.mark.parametrize("explicit_schema", [False, True], ids=["inferred", "explicit_schema"])
@pytest.mark.parametrize(
    ("kind", "rows", "length"),
    [
        pytest.param("int64", 50, 3, id="int64_shorter"),
        pytest.param("int64", 50, 60, id="int64_longer"),
        pytest.param("nullable", 50, 2, id="nullable_mask_shorter"),
        pytest.param("nullable", 50, 60, id="nullable_mask_longer"),
        # A str column is read as an Arrow chunk only when the chunk is long enough, hence the larger sizes.
        pytest.param("str", 500, 300, id="str_shorter"),
        pytest.param("str", 500, 600, id="str_longer"),
    ],
)
def test_map_result_column_of_another_length_is_refused(kind, rows, length, explicit_schema):
    if kind == "str":
        pytest.importorskip("pyarrow")
    # A column shorter than the frame reads past its buffer, which can kill the process.
    script = (
        _RESIZED_COLUMN_CHILD.replace("KIND", repr(kind))
        .replace("ROWS", str(rows))
        .replace("LENGTH", str(length))
        .replace("EXPLICIT_SCHEMA", str(explicit_schema))
    )
    try:
        child = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=RESIZED_CHILD_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as hang:
        pytest.fail(f"the child did not finish within {RESIZED_CHILD_TIMEOUT} seconds: {hang.stderr!r}")
    lines = child.stdout.splitlines()
    outcomes = [json.loads(line.removeprefix("outcome ")) for line in lines if line.startswith("outcome ")]
    assert child.returncode == 0, child.stderr[-2000:]
    assert len(outcomes) == 1, child.stdout[-2000:]
    outcome = outcomes[0]
    named = {"resized", str(length), str(rows)}
    if "words" in outcome:
        outcome["words"] = named & set(outcome["words"])
    assert outcome == {"error": "InvalidInputException", "words": named, "answers": [[[42]], [[42]]]}

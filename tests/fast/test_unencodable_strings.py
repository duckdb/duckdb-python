"""A Python str that UTF-8 cannot encode is refused where it becomes a DuckDB string.

A str can hold surrogate code points (U+D800 to U+DFFF), which have no UTF-8 encoding. Python keeps two surrogate
escapes written next to each other as two separate code points, so they are refused like a single one.
"""

import re

import pytest

import duckdb
import numpy as np
import pandas as pd
from duckdb.sqltypes import BIGINT, VARCHAR


def execute_positional(con, text):
    return con.execute("SELECT ?", [text]).fetchone()[0]


def executemany(con, text):
    con.execute("CREATE TABLE t (i INTEGER, s VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [[1, "ok"], [2, text]])
    return con.execute("SELECT s FROM t WHERE i = 2").fetchone()[0]


def sql_params(con, text):
    return con.sql("SELECT ?", params=[text]).fetchall()[0][0]


def execute_named(con, text):
    return con.execute("SELECT $s", {"s": text}).fetchone()[0]


def value_parameter(con, text):
    return con.execute("SELECT ?", [duckdb.Value(text, VARCHAR)]).fetchone()[0]


def list_element(con, text):
    return con.execute("SELECT ?", [["a", text]]).fetchone()[0][1]


def dict_field(con, text):
    return con.execute("SELECT ?", [{"f": text}]).fetchone()[0]["f"]


def struct_typed_value(con, text):
    value = duckdb.Value({"f": text}, duckdb.struct_type({"f": VARCHAR}))
    return con.execute("SELECT ?", [value]).fetchone()[0]["f"]


def map_value(con, text):
    return con.execute("SELECT ?", [{"key": ["k"], "value": [text]}]).fetchone()[0]["k"]


def map_key(con, text):
    (key,) = con.execute("SELECT ?", [{"key": [text], "value": [1]}]).fetchone()[0]
    return key


def constant_expression(con, text):
    return con.sql("SELECT 1").select(duckdb.ConstantExpression(text)).fetchall()[0][0]


def udf_result(con, text):
    con.create_function("give", lambda _: text, [BIGINT], VARCHAR)
    return con.sql("SELECT give(range) FROM range(1)").fetchall()[0][0]


def object_column(text):
    return pd.DataFrame({"s": pd.Series(["a", text], dtype=object)})


def pandas_register(con, text):
    con.register("v", object_column(text))
    return con.sql("SELECT s FROM v").fetchall()[1][0]


def pandas_from_df(con, text):
    return con.from_df(object_column(text)).fetchall()[1][0]


def pandas_object_list_column(con, text):
    df = pd.DataFrame({"s": pd.Series([["a"], [text]], dtype=object)})
    return con.from_df(df).fetchall()[1][0][0]


def pandas_object_category(con, text):
    categories = pd.Index(["a", text], dtype=object)
    df = pd.DataFrame({"s": pd.Categorical.from_codes([0, 1], categories=categories)})
    return con.from_df(df).fetchall()[1][0]


def pandas_string_dtype_python(con, text):
    df = pd.DataFrame({"s": pd.Series(["a", text], dtype=pd.StringDtype("python"))})
    return con.from_df(df).fetchall()[1][0]


def numpy_str_array(con, text):
    arrays = {"s": np.array(["a", text])}  # noqa: F841
    return con.sql("SELECT s FROM arrays").fetchall()[1][0]


def numpy_object_array(con, text):
    arrays = {"s": np.array(["a", text], dtype=object)}  # noqa: F841
    return con.sql("SELECT s FROM arrays").fetchall()[1][0]


CONVERSIONS = [
    execute_positional,
    executemany,
    sql_params,
    execute_named,
    value_parameter,
    list_element,
    dict_field,
    struct_typed_value,
    map_value,
    map_key,
    constant_expression,
    udf_result,
    pandas_register,
    pandas_from_df,
    pandas_object_list_column,
    pandas_object_category,
    pandas_string_dtype_python,
    numpy_str_array,
    numpy_object_array,
]


def read_csv_na_values(con, path, text):
    con.read_csv(path, na_values=text)


def read_csv_na_values_list(con, path, text):
    con.read_csv(path, na_values=["NA", text])


def read_csv_delimiter(con, path, text):
    con.read_csv(path, delimiter=text)


def write_csv_na_rep(con, path, text):
    con.sql("SELECT NULL AS a").write_csv(path + ".out", na_rep=text)


CSV_OPTIONS = [read_csv_na_values, read_csv_na_values_list, read_csv_delimiter, write_csv_na_rep]


def refusal(escaped_char, position):
    return re.escape(f"cannot encode str as UTF-8, surrogate '{escaped_char}' at position {position}")


# A value converted as a Python value, which is every reader option, every parameter and every frame column, reports
# the offset inside the string itself. A value rendered into SQL text reports the offset inside that text, so a
# constant expression counts the opening quote of its literal and SQL text given to a verb counts the words before it.
# A CSV quotechar and a file path reach the message only since the conversion stopped escaping as std::bad_cast.
WRITTEN_AS_A_LITERAL = {constant_expression}


def conversion_cases():
    cases = []
    for convert in CONVERSIONS:
        if convert in WRITTEN_AS_A_LITERAL:
            cases.append(pytest.param(convert, 1, id=convert.__name__))
        else:
            cases.append(pytest.param(convert, 0, id=convert.__name__))
    return cases


@pytest.mark.parametrize(("convert", "position"), conversion_cases())
def test_surrogate_is_refused(convert, position):
    con = duckdb.connect()
    with pytest.raises(duckdb.InvalidInputException, match=refusal("\\ud800", position)):
        convert(con, "\ud800")
    assert con.execute("SELECT 42").fetchone() == (42,)


@pytest.mark.parametrize("option", CSV_OPTIONS)
def test_csv_option_with_surrogate_is_refused(option, tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("a,b\n1,NA\n")
    con = duckdb.connect()
    with pytest.raises(duckdb.InvalidInputException, match=refusal("\\ud800", 0)):
        option(con, str(path), "\ud800")
    assert con.execute("SELECT 42").fetchone() == (42,)


def test_csv_quotechar_with_surrogate_is_refused(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("a,b\n1,NA\n")
    con = duckdb.connect()
    with pytest.raises(duckdb.InvalidInputException, match=refusal("\\ud800", 0)):
        con.read_csv(str(path), quotechar="\ud800")
    assert con.execute("SELECT 42").fetchone() == (42,)


def test_csv_path_with_surrogate_is_refused():
    con = duckdb.connect()
    with pytest.raises(duckdb.InvalidInputException, match=refusal("\\ud800", 0)):
        con.read_csv("\ud800")
    assert con.execute("SELECT 42").fetchone() == (42,)


@pytest.mark.parametrize(
    ("text", "escaped_char", "position"),
    [
        pytest.param("ab\udfffc", "\\udfff", 2, id="two_byte_kind"),
        pytest.param("\U0001f600\udc00", "\\udc00", 1, id="four_byte_kind"),
        pytest.param("\ud83d\ude00", "\\ud83d", 0, id="two_escapes"),
    ],
)
@pytest.mark.parametrize("convert", [execute_positional, pandas_from_df])
def test_refusal_names_the_first_surrogate(convert, text, escaped_char, position):
    con = duckdb.connect()
    with pytest.raises(duckdb.InvalidInputException, match=refusal(escaped_char, position)):
        convert(con, text)


@pytest.mark.parametrize("text", ["a\x00b", "\U0001f600"], ids=["embedded_nul", "non_bmp"])
@pytest.mark.parametrize("convert", CONVERSIONS)
def test_encodable_str_round_trips(convert, text):
    assert convert(duckdb.connect(), text) == text


@pytest.mark.parametrize(
    "text", ["\u00eb", "\u4e2d", "\U0001f600"], ids=["one_byte_kind", "two_byte_kind", "four_byte_kind"]
)
def test_object_column_of_each_str_kind_round_trips(text):
    assert pandas_from_df(duckdb.connect(), text) == text


KIND_MARKERS = {
    "one_byte_kind": ("\u00e9", 0xFF),
    "two_byte_kind": ("\u0100", 0xFFFF),
    "four_byte_kind": ("\U0001f600", 0x10FFFF),
}
UTF8_BOUNDARIES = [0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF]


def scan_encoded(text):
    return pandas_from_df(duckdb.connect(), text)


@pytest.mark.parametrize(
    ("kind", "code_point"),
    [
        pytest.param(kind, code_point, id=f"{kind}-U+{code_point:04X}")
        for kind, (_, largest) in KIND_MARKERS.items()
        for code_point in UTF8_BOUNDARIES
        if code_point <= largest
    ],
)
def test_object_column_round_trips_utf8_boundaries(kind, code_point):
    text = KIND_MARKERS[kind][0] + chr(code_point)
    assert scan_encoded(text) == text


@pytest.mark.parametrize("encoded_bytes", [12, 13], ids=["inline", "heap"])
@pytest.mark.parametrize("kind", list(KIND_MARKERS))
def test_object_column_round_trips_inline_and_heap_lengths(kind, encoded_bytes):
    marker = KIND_MARKERS[kind][0]
    text = marker + "a" * (encoded_bytes - len(marker.encode()))
    assert scan_encoded(text) == text


def test_value_construction_does_not_check():
    assert duckdb.Value("\ud800", VARCHAR).object == "\ud800"

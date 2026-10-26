import re

import pytest

import duckdb

AB = "STRUCT(a INTEGER, b INTEGER)"
NESTED = f"STRUCT(s {AB})"
PAIR = "TUPLE(INTEGER, INTEGER)"
# SQL refuses field names equal ignoring case, but struct_type accepts them.
FIELDS_EQUAL_IGNORING_CASE = {"a": "INTEGER", "A": "INTEGER"}


def resolve(con, target):
    return con.type(target) if isinstance(target, str) else con.struct_type(target)


def via_value_parameter(con, target, obj):
    return con.execute("SELECT ?", [duckdb.Value(obj, resolve(con, target))]).fetchone()[0]


def via_constant_expression(con, target, obj):
    expression = duckdb.ConstantExpression(duckdb.Value(obj, resolve(con, target)))
    return con.sql("SELECT 1").select(expression).fetchall()[0][0]


def via_udf_result(con, target, obj):
    con.create_function("produce", lambda: obj, [], resolve(con, target))
    return con.sql("SELECT produce()").fetchall()[0][0]


def via_udf_list_result(con, target, obj):
    con.create_function("produce", lambda: [obj], [], con.list_type(resolve(con, target)))
    (result,) = con.sql("SELECT produce()").fetchall()[0][0]
    return result


PATHS = [
    pytest.param(via_value_parameter, id="value_parameter"),
    pytest.param(via_constant_expression, id="constant_expression"),
    pytest.param(via_udf_result, id="udf_result"),
    pytest.param(via_udf_list_result, id="udf_list_result"),
]

CONVERTED = [
    pytest.param(AB, {"a": 1, "b": 2}, {"a": 1, "b": 2}, id="same_keys"),
    pytest.param(AB, {"b": 2, "a": 1}, {"a": 1, "b": 2}, id="keys_in_another_order"),
    pytest.param(AB, {"A": 1, "B": 2}, {"a": 1, "b": 2}, id="keys_in_another_case"),
    pytest.param(AB, {"a": None, "b": 2}, {"a": None, "b": 2}, id="null_field"),
    pytest.param(NESTED, {"s": {"b": 2, "a": 1}}, {"s": {"a": 1, "b": 2}}, id="nested_keys_in_another_order"),
    pytest.param(NESTED, {"s": None}, {"s": None}, id="nested_null_struct"),
    pytest.param(PAIR, {"k1": 1, "k2": 2}, (1, 2), id="unnamed_target_by_position"),
    pytest.param(PAIR, {"b": 2, "a": 1}, (2, 1), id="unnamed_target_follows_dict_order"),
    pytest.param("STRUCT(\"b'a'\" INTEGER)", {b"a": 42}, {"b'a'": 42}, id="bytes_key_named_by_str"),
    pytest.param(
        "STRUCT(\"b'a'\" INTEGER, a INTEGER)",
        {b"a": 1, "a": 2},
        {"b'a'": 1, "a": 2},
        id="bytes_key_and_its_text_are_distinct_keys",
    ),
    pytest.param('STRUCT("1" INTEGER)', {1: 10}, {"1": 10}, id="integer_key"),
]

# Each case names the dict and the STRUCT type the error reports, which for a nested case is the inner pair.
REFUSED = [
    pytest.param(AB, {"a": 1, "c": 3}, {"a": 1, "c": 3}, AB, id="one_key_missing"),
    pytest.param(AB, {"a": "bad", "c": 2}, {"a": "bad", "c": 2}, AB, id="key_missing_after_a_bad_value"),
    pytest.param(AB, {"k1": 1, "k2": 2}, {"k1": 1, "k2": 2}, AB, id="no_key_matches"),
    pytest.param(AB, {"a": 1, "A": 2}, {"a": 1, "A": 2}, AB, id="keys_equal_ignoring_case"),
    pytest.param(AB, {"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}, AB, id="extra_key"),
    pytest.param(AB, {"a": 1}, {"a": 1}, AB, id="fewer_keys"),
    pytest.param(NESTED, {"s": {"k1": 1, "k2": 2}}, {"k1": 1, "k2": 2}, AB, id="nested_no_key_matches"),
    pytest.param(PAIR, {"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}, PAIR, id="unnamed_target_extra_key"),
    pytest.param(AB, {"key": [1], "value": [2]}, {"key": [1], "value": [2]}, AB, id="map_shaped_dict"),
    pytest.param(AB, {"b": 20, b"a": 10}, {"b": 20, b"a": 10}, AB, id="bytes_key_does_not_name_its_text"),
    pytest.param(
        FIELDS_EQUAL_IGNORING_CASE,
        {"a": 1, "A": 2},
        {"a": 1, "A": 2},
        "STRUCT(a INTEGER, A INTEGER)",
        id="keys_and_fields_equal_ignoring_case",
    ),
    pytest.param(
        FIELDS_EQUAL_IGNORING_CASE,
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
        "STRUCT(a INTEGER, A INTEGER)",
        id="fields_equal_ignoring_case",
    ),
]


def conversion_error(obj, target):
    return re.escape(f"We could not convert the object {obj!r} to the desired target type ({target})")


class TestDictToStruct:
    @pytest.mark.parametrize("path", PATHS)
    @pytest.mark.parametrize(("target", "obj", "expected"), CONVERTED)
    def test_converted(self, path, target, obj, expected):
        con = duckdb.connect()
        assert path(con, target, obj) == expected

    @pytest.mark.parametrize("path", PATHS)
    @pytest.mark.parametrize(("target", "obj", "reported_obj", "reported_target"), REFUSED)
    def test_refused(self, path, target, obj, reported_obj, reported_target):
        con = duckdb.connect()
        with pytest.raises(duckdb.InvalidInputException, match=conversion_error(reported_obj, reported_target)):
            path(con, target, obj)

    def test_untyped_bytes_key_named_by_str(self):
        con = duckdb.connect()
        assert con.execute("SELECT ?", [{b"a": 42}]).fetchall() == [({"b'a'": 42},)]

    @pytest.mark.parametrize(
        ("rows", "expected_type", "expected"),
        [
            pytest.param([{b"a": 1}], "STRUCT(\"b'a'\" INTEGER)", [({"b'a'": 1},)], id="bytes_key"),
            pytest.param(
                [{b"a": 1, "b": 2}],
                "STRUCT(\"b'a'\" INTEGER, b INTEGER)",
                [({"b'a'": 1, "b": 2},)],
                id="bytes_and_str_keys",
            ),
        ],
    )
    def test_pandas_bytes_keys_named_by_str(self, rows, expected_type, expected):
        pd = pytest.importorskip("pandas")
        con = duckdb.connect()
        df = pd.DataFrame({"col": rows})  # noqa: F841
        rel = con.sql("SELECT col FROM df")
        assert str(rel.types[0]) == expected_type
        assert rel.fetchall() == expected

    def test_pandas_keys_equal_ignoring_case_refused(self):
        pd = pytest.importorskip("pandas")
        con = duckdb.connect()
        obj = {"a": 1, "A": 2}
        df = pd.DataFrame({"col": [obj]})  # noqa: F841
        with pytest.raises(duckdb.InvalidInputException, match=conversion_error(obj, "STRUCT(a INTEGER, A INTEGER)")):
            con.sql("SELECT col FROM df").fetchall()

    def test_numpy_object_row_outside_the_sample(self):
        np = pytest.importorskip("numpy")
        con = duckdb.connect()
        con.execute("SET pandas_analyze_sample=1")
        col = np.empty(2, dtype=object)
        col[0] = {"a": 10, "b": 20}
        col[1] = {"b": 2, "a": 1}
        data = {"col": col}  # noqa: F841
        assert con.sql("SELECT col FROM data").fetchall() == [({"a": 10, "b": 20},), ({"a": 1, "b": 2},)]
        col[1] = {"k1": 1, "k2": 2}
        with pytest.raises(duckdb.InvalidInputException, match=conversion_error(col[1], AB)):
            con.sql("SELECT col FROM data").fetchall()

    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            pytest.param({"b": 2, "a": 1}, {"a": 1, "b": 2}, id="keys_in_another_order"),
            pytest.param({"A": 1, "B": 2}, {"a": 1, "b": 2}, id="keys_in_another_case"),
        ],
    )
    def test_pandas_row_outside_the_sample_converted(self, row, expected):
        pd = pytest.importorskip("pandas")
        con = duckdb.connect()
        con.execute("SET pandas_analyze_sample=1")
        df = pd.DataFrame({"col": [{"a": 10, "b": 20}, row]})  # noqa: F841
        assert con.sql("SELECT col FROM df").fetchall() == [({"a": 10, "b": 20},), (expected,)]

    @pytest.mark.parametrize(
        "row",
        [
            pytest.param({"a": 1, "c": 3}, id="one_key_missing"),
            pytest.param({"k1": 1, "k2": 2}, id="no_key_matches"),
            pytest.param({"a": 1, "b": 2, "c": 3}, id="extra_key"),
        ],
    )
    def test_pandas_row_outside_the_sample_refused(self, row):
        pd = pytest.importorskip("pandas")
        con = duckdb.connect()
        con.execute("SET pandas_analyze_sample=1")
        df = pd.DataFrame({"col": [{"a": 10, "b": 20}, row]})  # noqa: F841
        with pytest.raises(duckdb.InvalidInputException, match=conversion_error(row, AB)):
            con.sql("SELECT col FROM df").fetchall()

from decimal import Decimal

import pytest

import duckdb

try:
    import pandas as pd

    can_test_pandas = True
except ImportError:
    can_test_pandas = False

try:
    import pyarrow as pa

    can_test_pyarrow = True
except ImportError:
    can_test_pyarrow = False


WIDTH_38_BOUNDARY = [
    (Decimal("9" * 38), "DECIMAL(38,0)", Decimal("9" * 38)),
    (Decimal("0." + "1" * 38), "DECIMAL(38,38)", Decimal("0." + "1" * 38)),
    (Decimal("1E-37"), "DECIMAL(38,37)", Decimal("1E-37")),
    (Decimal("9" * 39), "DOUBLE", float(Decimal("9" * 39))),
    (Decimal("1E-38"), "DOUBLE", float(Decimal("1E-38"))),
]


class TestDecimalPositiveExponent:
    @pytest.mark.parametrize(
        ("value", "expected_type", "expected_value"),
        [
            (Decimal("1E+2"), "DECIMAL(3,0)", Decimal("100")),
            (Decimal("-1E+2"), "DECIMAL(3,0)", Decimal("-100")),
            (Decimal("0E+2"), "DECIMAL(3,0)", Decimal("0")),
            (Decimal("1E+3"), "DECIMAL(4,0)", Decimal("1000")),
            (Decimal("1E+4"), "DECIMAL(5,0)", Decimal("10000")),
            (Decimal("1E+8"), "DECIMAL(9,0)", Decimal("100000000")),
            (Decimal("1E+9"), "DECIMAL(10,0)", Decimal("1000000000")),
            (Decimal("1E+17"), "DECIMAL(18,0)", Decimal("100000000000000000")),
            (Decimal("1E+18"), "DECIMAL(19,0)", Decimal("1000000000000000000")),
            (Decimal("1E+19"), "DECIMAL(20,0)", Decimal("10000000000000000000")),
            (Decimal("1E+20"), "DECIMAL(21,0)", Decimal("100000000000000000000")),
            (Decimal("1E+21"), "DECIMAL(22,0)", Decimal("1000000000000000000000")),
            (Decimal("1E+36"), "DECIMAL(37,0)", Decimal("1000000000000000000000000000000000000")),
            (Decimal("-1E+37"), "DECIMAL(38,0)", Decimal("-10000000000000000000000000000000000000")),
            (Decimal("1E+37"), "DECIMAL(38,0)", Decimal("10000000000000000000000000000000000000")),
            (Decimal("12E+2"), "DECIMAL(4,0)", Decimal("1200")),
            (Decimal("123E+2"), "DECIMAL(5,0)", Decimal("12300")),
            (Decimal("1.5E+3"), "DECIMAL(4,0)", Decimal("1500")),
            (Decimal("12.34E+5"), "DECIMAL(7,0)", Decimal("1234000")),
            (Decimal("999E+9"), "DECIMAL(12,0)", Decimal("999000000000")),
            (
                Decimal("12345678901234567890123E+15"),
                "DECIMAL(38,0)",
                Decimal("12345678901234567890123000000000000000"),
            ),
            (Decimal("1.23"), "DECIMAL(3,2)", Decimal("1.23")),
            (Decimal("100"), "DECIMAL(3,0)", Decimal("100")),
        ],
    )
    def test_positive_exponent_value_and_type(self, duckdb_cursor, value, expected_type, expected_value):
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == expected_type, f"Type mismatch for {value}"
        assert result_value == expected_value, f"Value mismatch for {value}"

    @pytest.mark.parametrize(
        ("value", "expected_type"),
        [
            (Decimal("1E+38"), "DOUBLE"),
            (Decimal("1E+39"), "DOUBLE"),
            (Decimal("12E+37"), "DOUBLE"),
            (Decimal("123E+36"), "DOUBLE"),
            (Decimal("1E+255"), "DOUBLE"),
        ],
    )
    def test_overflow_to_double(self, duckdb_cursor, value, expected_type):
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == expected_type, f"Type should be DOUBLE for {value}"
        assert result_value == float(value), f"Value should match float for {value}"

    @pytest.mark.parametrize(
        "value",
        [
            Decimal("1E+256"),
            Decimal("1E+257"),
            Decimal("1" * 40 + "E-2"),
        ],
    )
    def test_large_exponent_type_value_agreement(self, duckdb_cursor, value):
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == "DOUBLE", f"Type should be DOUBLE for {value}"
        assert result_value == float(value), f"Value should match float for {value}"

    @pytest.mark.parametrize(
        ("input_str", "should_raise"),
        [
            ("1E+2", False),
            ("123E+2", False),
            ("1.5E+3", False),
            ("12.34E+5", False),
            ("999E+9", False),
            ("100", False),
            ("1.23", False),
            ("1E+18", False),
            ("1E+19", False),
            ("1E+20", True),
            ("1E+21", True),
        ],
    )
    def test_insert_into_decimal_28_8(self, duckdb_cursor, input_str, should_raise):
        duckdb_cursor.execute("CREATE TABLE t (v DECIMAL(28,8))")
        d = Decimal(input_str)
        if should_raise:
            with pytest.raises(duckdb.ConversionException, match="out of range"):
                duckdb_cursor.execute("INSERT INTO t VALUES (?)", [d])
        else:
            duckdb_cursor.execute("INSERT INTO t VALUES (?)", [d])
            stored = duckdb_cursor.execute("SELECT v FROM t").fetchone()[0]
            assert stored == d, f"Stored value mismatch for {input_str}: expected {d}, got {stored}"

    @pytest.mark.parametrize(
        ("params", "expected_values"),
        [
            (
                [[Decimal("1E+2")], [Decimal("1.5")], [None], [Decimal("-5E+4")]],
                [None, Decimal("-50000.00000000"), Decimal("1.50000000"), Decimal("100.00000000")],
            ),
        ],
    )
    def test_executemany_mixed_forms(self, duckdb_cursor, params, expected_values):
        duckdb_cursor.execute("CREATE TABLE t (v DECIMAL(28,8))")
        duckdb_cursor.executemany("INSERT INTO t VALUES (?)", params)
        results = [row[0] for row in duckdb_cursor.execute("SELECT v FROM t ORDER BY v NULLS FIRST").fetchall()]
        assert results == expected_values, "executemany results mismatch"

    @pytest.mark.parametrize(
        ("list_value", "expected_type", "expected_values"),
        [
            (
                [Decimal("1E+2"), Decimal("5E+3")],
                "DECIMAL(4,0)[]",
                [Decimal("100"), Decimal("5000")],
            ),
            (
                [Decimal("1E+2"), Decimal("1.5")],
                "DECIMAL(4,1)[]",
                [Decimal("100.0"), Decimal("1.5")],
            ),
        ],
    )
    def test_list_parameter(self, duckdb_cursor, list_value, expected_type, expected_values):
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [list_value, list_value]).fetchone()
        assert result_type == expected_type, f"Type mismatch for list {list_value}"
        assert result_value == expected_values, f"Values mismatch for list {list_value}"

    @pytest.mark.parametrize(
        ("struct_value", "expected_value"),
        [
            (
                {"amount": Decimal("1E+2")},
                {"amount": Decimal("100")},
            ),
        ],
    )
    def test_struct_parameter(self, duckdb_cursor, struct_value, expected_value):
        result = duckdb_cursor.execute("SELECT ?", [struct_value]).fetchone()[0]
        assert result == expected_value, "Struct value mismatch"

    @pytest.mark.skipif(not can_test_pandas, reason="pandas not available")
    @pytest.mark.parametrize(
        ("values", "expected_type", "expected_values"),
        [
            (
                [Decimal("1E+2"), Decimal("12E+5")],
                "DECIMAL(7,0)",
                [(Decimal("100"),), (Decimal("1200000"),)],
            ),
            (
                [Decimal("1E+2"), Decimal("1.5"), None],
                "DECIMAL(4,1)",
                [(Decimal("100.0"),), (Decimal("1.5"),), (None,)],
            ),
        ],
    )
    def test_pandas_object_column(self, duckdb_cursor, values, expected_type, expected_values):
        _ = pd.DataFrame({"v": pd.Series(values, dtype=object)})
        results = duckdb_cursor.execute("SELECT v, typeof(v) FROM _").fetchall()
        for i, (_val, typ) in enumerate(results):
            assert typ == expected_type, f"Type mismatch at row {i}"
        result_values = [(row[0],) for row in results]
        assert result_values == expected_values, "Values mismatch"

    @pytest.mark.skipif(not can_test_pandas, reason="pandas not available")
    def test_pandas_large_exponent_agreement(self, duckdb_cursor):
        _ = pd.DataFrame({"v": pd.Series([Decimal("1E+256")], dtype=object)})
        result_value, result_type = duckdb_cursor.execute("SELECT v, typeof(v) FROM _").fetchone()
        typeof_result = duckdb_cursor.execute("SELECT typeof(?)", [Decimal("1E+256")]).fetchone()[0]
        assert result_type == typeof_result, f"Type disagreement: scan={result_type} vs typeof={typeof_result}"
        assert result_type == "DOUBLE", "Type should be DOUBLE for 1E+256"
        assert result_value == float(Decimal("1E+256")), "Value should match float for 1E+256"

    def test_udf_return_value(self, duckdb_cursor):
        duckdb_cursor.create_function("test_udf", lambda x: Decimal("1E+2"), return_type="DECIMAL(10,0)")
        result = duckdb_cursor.execute("SELECT test_udf(1)").fetchone()[0]
        assert result == Decimal("100"), f"UDF should return 100, got {result}"

    @pytest.mark.skipif(not can_test_pyarrow, reason="pyarrow not available")
    @pytest.mark.parametrize(
        ("value", "expected_pa_type", "expected_db_type", "expected_value"),
        [
            (
                Decimal("1E+2"),
                "decimal128(3, 0)",
                "DECIMAL(3,0)",
                Decimal("100"),
            ),
            (
                Decimal("12E+5"),
                "decimal128(7, 0)",
                "DECIMAL(7,0)",
                Decimal("1200000"),
            ),
        ],
    )
    def test_pyarrow_compatibility(self, duckdb_cursor, value, expected_pa_type, expected_db_type, expected_value):
        pa_array = pa.array([value])
        assert str(pa_array.type) == expected_pa_type, f"PyArrow type mismatch for {value}"
        assert pa_array[0].as_py() == expected_value, f"PyArrow value mismatch for {value}"
        db_value, db_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert db_type == expected_db_type, f"DuckDB type mismatch for {value}"
        assert db_value == expected_value, f"DuckDB value mismatch for {value}"

    @pytest.mark.parametrize(
        ("value", "expected_type", "expected_value"),
        [
            (Decimal("1.23"), "DECIMAL(3,2)", Decimal("1.23")),
            (Decimal("1E-2"), "DECIMAL(3,2)", Decimal("0.01")),
            (Decimal("123E-5"), "DECIMAL(6,5)", Decimal("0.00123")),
        ],
    )
    def test_negative_exponent_unchanged(self, duckdb_cursor, value, expected_type, expected_value):
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == expected_type, f"Type mismatch for {value}"
        assert result_value == expected_value, f"Value mismatch for {value}"

    def test_special_values_unchanged(self, duckdb_cursor):
        result_nan, type_nan = duckdb_cursor.execute("SELECT ?, typeof(?)", [Decimal("NaN"), Decimal("NaN")]).fetchone()
        assert type_nan == "FLOAT", "NaN type should be FLOAT"
        assert str(result_nan) == "nan", "NaN value mismatch"

        result_inf, type_inf = duckdb_cursor.execute(
            "SELECT ?, typeof(?)", [Decimal("Infinity"), Decimal("Infinity")]
        ).fetchone()
        assert type_inf == "FLOAT", "Infinity type should be FLOAT"
        assert result_inf == float("inf"), "Infinity value mismatch"

    @pytest.mark.parametrize(
        ("value", "expected_type", "expected_value"),
        [
            (Decimal("1E+2147483647"), "DOUBLE", float("inf")),
            (Decimal("1E-2147483647"), "DOUBLE", 0.0),
        ],
    )
    def test_exponent_at_int32_limits(self, duckdb_cursor, value, expected_type, expected_value):
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == expected_type, f"Type mismatch for {value}"
        assert result_value == expected_value, f"Value mismatch for {value}"

    @pytest.mark.parametrize(
        ("value", "expected_type", "expected_value"),
        [
            (Decimal("1E+3000000000"), "DOUBLE", float("inf")),
            (Decimal("1E-3000000000"), "DOUBLE", 0.0),
            (Decimal("1E-2147483648"), "DOUBLE", 0.0),
        ],
    )
    def test_exponent_beyond_int32_range(self, duckdb_cursor, value, expected_type, expected_value):
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == expected_type, f"Type should be {expected_type} for {value}"
        assert result_value == expected_value, f"Value should be {expected_value} for {value}"

    @pytest.mark.parametrize(("value", "expected_type", "expected_value"), WIDTH_38_BOUNDARY)
    def test_38_boundary(self, duckdb_cursor, value, expected_type, expected_value):
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == expected_type, f"Type mismatch for {value}"
        assert result_value == expected_value, f"Value mismatch for {value}"

    @pytest.mark.skipif(not can_test_pandas, reason="pandas not available")
    @pytest.mark.parametrize(("value", "expected_type", "expected_value"), WIDTH_38_BOUNDARY)
    def test_38_boundary_pandas_object_column(self, duckdb_cursor, value, expected_type, expected_value):
        _ = pd.DataFrame({"v": pd.Series([value], dtype=object)})
        result_value, result_type = duckdb_cursor.execute("SELECT v, typeof(v) FROM _").fetchone()
        assert result_type == expected_type, f"Type mismatch for {value}"
        assert result_value == expected_value, f"Value mismatch for {value}"

    @pytest.mark.skipif(not can_test_pandas, reason="pandas not available")
    @pytest.mark.parametrize(
        ("value", "expected_type", "expected_value"),
        [
            (Decimal("1E+37"), "DECIMAL(38,0)", Decimal("1E+37")),
            (Decimal("1E+38"), "DOUBLE", float(Decimal("1E+38"))),
            (Decimal("1E+256"), "DOUBLE", float(Decimal("1E+256"))),
            (Decimal("1E+257"), "DOUBLE", float(Decimal("1E+257"))),
            (Decimal("1E+2147483647"), "DOUBLE", float("inf")),
            (Decimal("1E+3000000000"), "DOUBLE", float("inf")),
            (Decimal("1E-2147483648"), "DOUBLE", 0.0),
        ],
    )
    def test_pandas_object_column_at_limits(self, duckdb_cursor, value, expected_type, expected_value):
        _ = pd.DataFrame({"v": pd.Series([value], dtype=object)})
        result_value, result_type = duckdb_cursor.execute("SELECT v, typeof(v) FROM _").fetchone()
        assert result_type == expected_type, f"Type mismatch for {value}"
        assert result_value == expected_value, f"Value mismatch for {value}"

    @pytest.mark.parametrize(
        ("value", "expected_type", "expected_value"),
        [
            (Decimal("1E-256"), "DOUBLE", float(Decimal("1E-256"))),
            (Decimal("1E-257"), "DOUBLE", float(Decimal("1E-257"))),
        ],
    )
    def test_negative_exponent_256_plus(self, duckdb_cursor, value, expected_type, expected_value):
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == expected_type, f"Type mismatch for {value}"
        assert result_value == expected_value, f"Value mismatch for {value}"

    def test_nan_with_40_digit_payload(self, duckdb_cursor):
        value = Decimal("NaN" + "1" * 40)
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == "FLOAT", f"Type should be FLOAT for {value}"
        assert str(result_value) == "nan", f"Value should be NaN for {value}"

    def test_negative_infinity(self, duckdb_cursor):
        value = Decimal("-Infinity")
        result_value, result_type = duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == "FLOAT", "Type should be FLOAT for -Infinity"
        assert result_value == float("-inf"), "Value should be -inf for -Infinity"

    @pytest.mark.skipif(not can_test_pandas, reason="pandas not available")
    def test_negative_infinity_pandas(self, duckdb_cursor):
        _ = pd.DataFrame({"v": pd.Series([Decimal("-Infinity")], dtype=object)})
        result_value, result_type = duckdb_cursor.execute("SELECT v, typeof(v) FROM _").fetchone()
        assert result_type == "FLOAT", "Type should be FLOAT for -Infinity"
        assert result_value == float("-inf"), "Value should be -inf for -Infinity"

    def test_subclass_custom_str_raises(self, duckdb_cursor):
        class CustomDecimal(Decimal):
            def __str__(self) -> str:
                return "$1E+40"

        value = CustomDecimal("1E+40")
        with pytest.raises(duckdb.InvalidInputException, match="Failed to convert Decimal value to DOUBLE"):
            duckdb_cursor.execute("SELECT ?", [value])

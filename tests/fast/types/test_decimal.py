from decimal import Decimal

import pytest

import duckdb
import numpy


class TestDecimal:
    def test_decimal(self, duckdb_cursor):
        duckdb_cursor.execute(
            "SELECT 1.2::DECIMAL(4,1), 100.3::DECIMAL(9,1), 320938.4298::DECIMAL(18,4), 49082094824.904820482094::DECIMAL(30,12), NULL::DECIMAL"  # noqa: E501
        )
        result = duckdb_cursor.fetchall()
        assert result == [
            (Decimal("1.2"), Decimal("100.3"), Decimal("320938.4298"), Decimal("49082094824.904820482094"), None)
        ]

    def test_decimal_numpy(self, duckdb_cursor):
        duckdb_cursor.execute(
            "SELECT 1.2::DECIMAL(4,1) AS a, 100.3::DECIMAL(9,1) AS b, 320938.4298::DECIMAL(18,4) AS c, 49082094824.904820482094::DECIMAL(30,12) AS d"  # noqa: E501
        )
        result = duckdb_cursor.fetchnumpy()
        assert result == {
            "a": numpy.array([1.2]),
            "b": numpy.array([100.3]),
            "c": numpy.array([320938.4298]),
            "d": numpy.array([49082094824.904820482094]),
        }

    @pytest.mark.parametrize(
        ("text", "expected_type"),
        [
            ("1E+2", "DECIMAL(3,0)"),
            ("123E+2", "DECIMAL(5,0)"),
            ("1.5E+3", "DECIMAL(4,0)"),
            ("12.34E+5", "DECIMAL(7,0)"),
            ("999E+9", "DECIMAL(12,0)"),
            ("100", "DECIMAL(3,0)"),
            ("1.23", "DECIMAL(3,2)"),
            ("1E+18", "DECIMAL(19,0)"),
            ("1E+19", "DECIMAL(20,0)"),
            ("1E+20", "DECIMAL(21,0)"),
            ("1E+21", "DECIMAL(22,0)"),
            ("-1E+2", "DECIMAL(3,0)"),
        ],
    )
    def test_decimal_positive_exponent_roundtrip(self, duckdb_cursor, text, expected_type):
        value = Decimal(text)
        assert duckdb_cursor.execute("SELECT typeof(?)", [value]).fetchone()[0] == expected_type
        assert duckdb_cursor.execute("SELECT ?", [value]).fetchone()[0] == value

    def test_decimal_positive_exponent_insert(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (v DECIMAL(28,8))")
        for text in ["1E+2", "123E+2", "1.5E+3", "12.34E+5", "999E+9", "100", "1.23", "1E+18", "1E+19", "-1E+2"]:
            value = Decimal(text)
            duckdb_cursor.execute("DELETE FROM t")
            duckdb_cursor.execute("INSERT INTO t VALUES (?)", [value])
            assert duckdb_cursor.execute("SELECT v FROM t").fetchone()[0] == value

    def test_decimal_positive_exponent_out_of_range(self, duckdb_cursor):
        # DECIMAL(28,8) has 20 integer digits; 1E+20 used to bind as DECIMAL(21,20) and store garbage.
        duckdb_cursor.execute("CREATE TABLE t (v DECIMAL(28,8))")
        with pytest.raises(duckdb.ConversionException, match="out of range"):
            duckdb_cursor.execute("INSERT INTO t VALUES (?)", [Decimal("1E+20")])

    def test_decimal_normalize_positive_exponent(self, duckdb_cursor):
        value = Decimal("1000.0000").normalize()
        assert value.as_tuple().exponent > 0
        assert duckdb_cursor.execute("SELECT ?", [value]).fetchone()[0] == Decimal("1000")

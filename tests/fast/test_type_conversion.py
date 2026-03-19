"""Regression tests for Python-to-DuckDB type conversion bugs.

Issue #115: Float conversion error with UNION containing float
Issue #171: Dictionary key case sensitivity not respected for parameter bindings
Issue #330: Integers >64-bit lose precision via double conversion
"""

import numpy as np
import pytest

import duckdb
from duckdb.sqltypes import BIGINT, DOUBLE, FLOAT, HUGEINT, UHUGEINT, VARCHAR, DuckDBPyType


class TestIssue115FloatToUnion:
    """HandleDouble should use DefaultCastAs for unknown target types like UNION."""

    def test_udf_float_to_union_type(self):
        conn = duckdb.connect()
        conn.create_function(
            "return_float",
            lambda: 1.5,
            return_type=duckdb.union_type({"u1": VARCHAR, "u2": BIGINT, "u3": DOUBLE}),
        )
        result = conn.sql("SELECT return_float()").fetchone()[0]
        assert result == 1.5

    def test_udf_float_to_ambiguous_union_type(self):
        """UNION with duplicate DOUBLE members (from np.float64 and float) must not raise ambiguity error."""
        conn = duckdb.connect()
        conn.create_function(
            "return_float",
            lambda: 1.5,
            return_type=duckdb.union_type({"u1": VARCHAR, "u2": BIGINT, "u3": DOUBLE, "u4": FLOAT, "u5": DOUBLE}),
        )
        result = conn.sql("SELECT return_float()").fetchone()[0]
        assert result == 1.5

    def test_udf_dict_with_float_in_union_struct(self):
        """Original repro from issue #115 with ambiguous UNION members."""
        conn = duckdb.connect()

        arr = [{"a": 1, "b": 1.2}, {"a": 3, "b": 2.4}]

        def test():
            return arr

        return_type = DuckDBPyType(list[dict[str, str | int | np.float64 | np.float32 | float]])
        conn.create_function("test", test, return_type=return_type)
        result = conn.sql("SELECT test()").fetchone()[0]
        assert len(result) == 2
        assert result[0]["b"] == pytest.approx(1.2)
        assert result[1]["b"] == pytest.approx(2.4)

    def test_udf_int_to_ambiguous_union_type(self):
        """HandleBigint default branch: int into UNION with duplicate BIGINT members."""
        conn = duckdb.connect()
        conn.create_function(
            "return_int",
            lambda: 42,
            return_type=duckdb.union_type({"u1": VARCHAR, "u2": BIGINT, "u3": BIGINT}),
        )
        result = conn.sql("SELECT return_int()").fetchone()[0]
        assert result == 42

    def test_udf_string_to_ambiguous_union_type(self):
        """HandleString default branch: str into UNION with duplicate VARCHAR members."""
        conn = duckdb.connect()
        conn.create_function(
            "return_str",
            lambda: "hello",
            return_type=duckdb.union_type({"u1": VARCHAR, "u2": BIGINT, "u3": VARCHAR}),
        )
        result = conn.sql("SELECT return_str()").fetchone()[0]
        assert result == "hello"


class TestIssue171DictKeyCaseSensitivity:
    """Dict keys differing only by case must preserve their individual values."""

    def test_case_sensitive_dict_keys(self):
        result = duckdb.execute("SELECT ?", [{"Key": "first", "key": "second"}]).fetchone()[0]
        assert result["Key"] == "first"
        assert result["key"] == "second"

    def test_case_sensitive_dict_keys_three_variants(self):
        result = duckdb.execute("SELECT ?", [{"abc": 1, "ABC": 2, "Abc": 3}]).fetchone()[0]
        assert result["abc"] == 1
        assert result["ABC"] == 2
        assert result["Abc"] == 3


class TestIssue330LargeIntegerPrecision:
    """Integers >64-bit must not lose precision via double conversion."""

    # --- Parameter binding path (TryTransformPythonNumeric) ---

    def test_param_hugeint_large(self):
        """Value with >52 significant bits must not lose precision."""
        value = (2**128 - 1) // 15 * 7  # 0x77777777777777777777777777777777
        result = duckdb.execute("SELECT ?::HUGEINT", [value]).fetchone()[0]
        assert result == value

    def test_param_uhugeint_max(self):
        """2**128-1 must not overflow when cast to UHUGEINT."""
        value = 2**128 - 1
        result = duckdb.execute("SELECT ?::UHUGEINT", [value]).fetchone()[0]
        assert result == value

    def test_param_auto_sniff(self):
        """2**64 without explicit cast should sniff as HUGEINT, not lose precision."""
        value = 2**64
        result = duckdb.execute("SELECT ?", [value]).fetchone()[0]
        assert result == value

    def test_param_negative_hugeint_no_regression(self):
        """Negative overflow path (already correct) must not regress."""
        value = -(2**64)
        result = duckdb.execute("SELECT ?::HUGEINT", [value]).fetchone()[0]
        assert result == value

    # --- UDF return path (TransformPythonObjectInternal template) ---

    def test_udf_return_large_hugeint(self):
        value = (2**128 - 1) // 15 * 7
        conn = duckdb.connect()
        conn.create_function("big_hugeint", lambda: value, return_type=HUGEINT)
        result = conn.sql("SELECT big_hugeint()").fetchone()[0]
        assert result == value

    def test_udf_return_large_uhugeint(self):
        value = 2**128 - 1
        conn = duckdb.connect()
        conn.create_function("big_uhugeint", lambda: value, return_type=UHUGEINT)
        result = conn.sql("SELECT big_uhugeint()").fetchone()[0]
        assert result == value

"""A Python int converts to the integer it stores, whatever an int subclass's own methods say.

Above 64 bits the value is split into two 64-bit halves and printed for a range error without calling
any method of the value itself, on the parameter path, a pandas object column and a UDF result alike.
"""

import re
import sys
from decimal import Decimal
from typing import ClassVar, NoReturn

import pytest

import duckdb
from duckdb.sqltypes import BIGINT, HUGEINT

TOO_LARGE = "Python integer too large for 128-bit integer type"


class CustomBaseException(BaseException):
    pass


EXCEPTIONS = [MemoryError, KeyboardInterrupt, SystemExit, CustomBaseException, ValueError, RecursionError, SystemError]


def bad_int(exc, value):
    """An int whose __rshift__ raises: the conversion to a 128-bit integer shifts by 64."""

    class BadInt(int):
        calls: ClassVar[list] = []

        def __rshift__(self, other) -> int:
            type(self).calls.append(other)
            msg = "boom in __rshift__"
            raise exc(msg)

    return BadInt(value)


class NotAnInt:
    def __index__(self) -> int:
        msg = "boom in __index__"
        raise MemoryError(msg)


def rshift_raises():
    msg = "boom in __rshift__"
    raise MemoryError(msg)


RSHIFT = [
    pytest.param(rshift_raises, id="rshift_raises"),
    pytest.param(lambda: 0, id="rshift_returns_zero"),
    pytest.param(NotAnInt, id="rshift_returns_a_non_int"),
]
PLAIN_OR_RSHIFT = [pytest.param(None, id="plain"), *RSHIFT]


def lying_int(rshift, value):
    """An int whose own methods misdescribe its value or refuse to describe it, recording every call."""

    class LyingInt(int):
        def refuse(self, name) -> NoReturn:
            self.calls.append(name)
            msg = f"boom in {name}"
            raise MemoryError(msg)

        def __rshift__(self, other) -> object:
            self.calls.append("__rshift__")
            return rshift()

        def __index__(self) -> int:
            self.refuse("__index__")

        def __int__(self) -> int:
            self.refuse("__int__")

        def __repr__(self) -> str:
            self.refuse("__repr__")

        def __str__(self) -> str:
            self.refuse("__str__")

        def __format__(self, spec) -> str:
            self.refuse("__format__")

        def bit_length(self) -> int:
            self.refuse("bit_length")

    result = LyingInt(value)
    result.calls = []
    return result


def plain_or_lying(rshift, value):
    return value if rshift is None else lying_int(rshift, value)


def calls_of(integer):
    return getattr(integer, "calls", [])


# The negative values with nonzero low bits fail a split that truncates the upper half instead of flooring it.
IN_RANGE = [2**65, -(2**65), -(2**65) + 1, -(2**64) - 1, 2**127, 2**128 - 1, -(2**127)]

AT_THE_BOUNDARIES = [
    pytest.param(-(2**63) - 1, "HUGEINT", id="below_int64"),
    pytest.param(2**127 - 1, "HUGEINT", id="hugeint_max"),
    pytest.param(2**127, "UHUGEINT", id="above_hugeint"),
    pytest.param(2**128 - 1, "UHUGEINT", id="uhugeint_max"),
    pytest.param(-(2**127), "HUGEINT", id="hugeint_min"),
]


class TestIntSubclassConversion:
    @pytest.mark.parametrize("value", IN_RANGE)
    def test_plain_int_converts_exactly(self, value):
        con = duckdb.connect()
        assert con.sql("SELECT ?", params=[value]).fetchall() == [(value,)]

    @pytest.mark.parametrize("value", IN_RANGE)
    @pytest.mark.parametrize("rshift", RSHIFT)
    def test_parameter_converts_the_stored_value(self, rshift, value):
        con = duckdb.connect()
        lying = lying_int(rshift, value)
        assert con.sql("SELECT ?", params=[lying]).fetchall() == [(value,)]
        assert lying.calls == []

    @pytest.mark.parametrize(("value", "expected_type"), AT_THE_BOUNDARIES)
    @pytest.mark.parametrize("rshift", PLAIN_OR_RSHIFT)
    def test_type_at_the_hugeint_boundaries(self, rshift, value, expected_type):
        con = duckdb.connect()
        integer = plain_or_lying(rshift, value)
        assert con.sql("SELECT $1, typeof($1)", params=[integer]).fetchall() == [(value, expected_type)]
        assert calls_of(integer) == []

    @pytest.mark.parametrize("value", [2**128, -(2**127) - 1, 2**130])
    @pytest.mark.parametrize("rshift", PLAIN_OR_RSHIFT)
    def test_beyond_128_bits_still_reports_the_range_error(self, rshift, value):
        con = duckdb.connect()
        integer = plain_or_lying(rshift, value)
        with pytest.raises(duckdb.InvalidInputException, match=re.escape(f"{TOO_LARGE}: {value}") + "$"):
            con.sql("SELECT ?", params=[integer]).fetchall()
        assert calls_of(integer) == []

    def test_value_beyond_the_decimal_digit_limit_still_reports_the_range_error(self):
        # Python refuses to turn an int of more than 4300 decimal digits into a string by default.
        con = duckdb.connect()
        with pytest.raises(duckdb.InvalidInputException, match=f"{TOO_LARGE}$"):
            con.sql("SELECT ?", params=[10**5000]).fetchall()

    def test_value_beyond_the_lifted_digit_limit_is_printed_without_its_own_methods(self):
        # Without the limit, Python prints a large int through code that calls the value's own bit_length.
        con = duckdb.connect()
        limit = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(0)
        try:
            lying = lying_int(rshift_raises, 10**10000)
            with pytest.raises(duckdb.InvalidInputException, match=re.escape(f"{TOO_LARGE}: {10**10000}") + "$"):
                con.sql("SELECT ?", params=[lying]).fetchall()
        finally:
            sys.set_int_max_str_digits(limit)
        assert lying.calls == []

    def test_connection_stays_usable_after_a_conversion_error(self):
        con = duckdb.connect()
        with pytest.raises(duckdb.InvalidInputException, match=TOO_LARGE):
            con.sql("SELECT ?", params=[lying_int(rshift_raises, 2**128)]).fetchall()
        assert con.sql("select 42").fetchall() == [(42,)]

    @pytest.mark.parametrize("exc", [pytest.param(cls, id=cls.__name__) for cls in EXCEPTIONS])
    def test_rshift_is_not_called_below_the_128_bit_path(self, exc):
        con = duckdb.connect()
        value = bad_int(exc, 2**64 - 1)
        assert con.sql("SELECT ?", params=[value]).fetchall() == [(2**64 - 1,)]
        assert type(value).calls == []


class TestUDFResultConversion:
    @pytest.mark.parametrize("value", [2**65, -(2**65)])
    @pytest.mark.parametrize("rshift", RSHIFT)
    def test_native_udf_result_converts_the_stored_value(self, rshift, value):
        con = duckdb.connect()
        lying = lying_int(rshift, value)
        con.create_function("lying", lambda x: lying, [BIGINT], HUGEINT, type="native")
        assert con.sql("select lying(1)").fetchall() == [(value,)]
        assert lying.calls == []

    @pytest.mark.parametrize(
        ("return_type", "expected"),
        [
            pytest.param("UHUGEINT", 2**65, id="UHUGEINT"),
            pytest.param("DOUBLE", float(2**65), id="DOUBLE"),
            pytest.param("DECIMAL(38,0)", Decimal(2**65), id="DECIMAL"),
        ],
    )
    @pytest.mark.parametrize("rshift", RSHIFT)
    def test_native_udf_result_casts_the_stored_value(self, rshift, return_type, expected):
        con = duckdb.connect()
        lying = lying_int(rshift, 2**65)
        con.create_function("lying", lambda x: lying, [BIGINT], return_type, type="native")
        assert con.sql("select lying(1)").fetchall() == [(expected,)]
        assert lying.calls == []


class TestPandasObjectColumn:
    @pytest.fixture
    def pd(self):
        return pytest.importorskip("pandas")

    @pytest.mark.parametrize("rshift", RSHIFT)
    def test_sampled_value_converts_the_stored_value(self, pd, rshift):
        lying = lying_int(rshift, 2**65)
        df = pd.DataFrame({"a": [lying]})  # noqa: F841
        con = duckdb.connect()
        assert con.sql("select * from df").fetchall() == [(2**65,)]
        assert lying.calls == []

    @pytest.mark.parametrize("rshift", RSHIFT)
    def test_value_beyond_the_sample_converts_the_stored_value(self, pd, rshift):
        lying = lying_int(rshift, 2**65)
        df = pd.DataFrame({"a": [2**65, lying]})  # noqa: F841
        con = duckdb.connect()
        # A one row sample types the column from the plain value, so only the scan converts the subclass.
        con.execute("SET pandas_analyze_sample = 1")
        assert con.sql("select a, typeof(a) from df").fetchall() == [(2**65, "HUGEINT"), (2**65, "HUGEINT")]
        assert lying.calls == []

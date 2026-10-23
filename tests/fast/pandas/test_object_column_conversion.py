import datetime
import enum
import json
import os
import re
import subprocess
import sys
import uuid
from decimal import Decimal

import pytest

import duckdb
import numpy as np
import pandas as pd

# At the default pandas_analyze_sample of 1000, 3000 rows sample every third row, so rows 2000 and 2999 are never seen.
ROWS = 3000
UNSAMPLED = 2999

TWO_HOURS = datetime.timezone(datetime.timedelta(hours=2))
NAIVE = datetime.datetime(2024, 5, 6, 7, 8, 9)
AWARE = datetime.datetime(2024, 5, 6, 7, 8, 9, tzinfo=TWO_HOURS)
NAIVE_TIME = datetime.time(4, 5, 6)
AWARE_TIME = datetime.time(4, 5, 6, tzinfo=TWO_HOURS)
JAN_FIRST = datetime.datetime(2024, 1, 1)


def not_yet(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


def quiet_child_env():
    # An editable install that rebuilds on import otherwise prints its cmake command into the child's stdout.
    return {**os.environ, "SKBUILD_EDITABLE_VERBOSE": "0"}


MISMATCH_NOT_RAISED = "a value outside the sample that the bound type cannot hold exactly is still cast into it"
MISMATCH_NOT_NAMED = "the error for a value outside the sample does not name its column, row and Python type yet"


def column(base, value, *, rows=ROWS, at=UNSAMPLED, label="amount", alternate=()):
    bases = [base, *alternate]
    values = [bases[i % len(bases)] for i in range(rows)]
    values[at] = value
    return pd.DataFrame({label: pd.Series(values, dtype=object)})


def fully_sampled(*values):
    return pd.DataFrame({"amount": pd.Series(list(values), dtype=object)})


def python_type_name(value):
    cls = type(value)
    module = cls.__module__
    if not isinstance(module, str) or module in ("builtins", "__main__"):
        return cls.__qualname__
    return f"{module}.{cls.__qualname__}"


def assert_names(message, *pieces):
    for piece in pieces:
        assert re.search(rf"(?<![\w.]){re.escape(str(piece))}(?!\w)", message), f"{piece!r} not named in {message!r}"


def read_row(con, at=UNSAMPLED):
    return con.execute(f"SELECT typeof(amount), amount FROM frame LIMIT 1 OFFSET {at}").fetchall()


def invalid_input_message(con, query="SELECT amount FROM frame"):
    with pytest.raises(duckdb.InvalidInputException) as error:
        con.execute(query).fetchall()
    return str(error.value)


class HalfFloat(float):
    pass


class Shade(enum.IntEnum):
    DARK = 5


class PlainInstance:
    pass


class Holder:
    class Inner:  # noqa: D106
        pass


class NumberModuled:
    __module__ = 5


class RaisingStr:
    def __str__(self) -> str:
        msg = "bad string"
        raise ValueError(msg)


class TuplelessDecimal(Decimal):
    def as_tuple(self):
        msg = "no tuple"
        raise ZeroDivisionError(msg)


INTEGER_BOUNDS = [
    pytest.param(1, "INTEGER", id="INTEGER"),
    pytest.param(2**40, "BIGINT", id="BIGINT"),
    pytest.param(2**63, "UBIGINT", id="UBIGINT"),
    pytest.param(-(2**64), "HUGEINT", id="HUGEINT"),
    pytest.param(2**127, "UHUGEINT", id="UHUGEINT"),
]


@not_yet(MISMATCH_NOT_RAISED)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(2.5, id="float"),
        pytest.param(-2.5, id="negative_float"),
        pytest.param(3.5, id="float_rounding_up"),
        pytest.param(np.float64(2.5), id="numpy_float64"),
        pytest.param(HalfFloat(2.5), id="float_subclass"),
        pytest.param(Decimal("2.5"), id="decimal"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_fractional_value_outside_the_sample_raises(duckdb_cursor, value):
    duckdb_cursor.register("frame", column(1, value))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", UNSAMPLED, python_type_name(value), "INTEGER")


@not_yet(MISMATCH_NOT_RAISED)
@pytest.mark.parametrize(
    ("rows", "at", "sample"),
    [
        pytest.param(20, 17, 1, id="sample_of_one_row"),
        pytest.param(2000, 1999, 1000, id="every_second_row_sampled"),
    ],
)
def test_fractional_value_outside_a_small_sample_raises(duckdb_cursor, rows, at, sample):
    duckdb_cursor.execute(f"SET pandas_analyze_sample={sample}")
    duckdb_cursor.register("frame", column(1, 2.5, rows=rows, at=at))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", at, "float", "INTEGER")


@not_yet(MISMATCH_NOT_RAISED)
def test_fractional_values_deep_in_a_long_column_raise(duckdb_cursor):
    values = [1] * 100000
    values[77777] = 2.5
    values[88888] = 3.7
    duckdb_cursor.register("frame", pd.DataFrame({"amount": pd.Series(values, dtype=object)}))
    message = invalid_input_message(duckdb_cursor, "SELECT amount FROM frame WHERE amount <> 1")
    assert_names(message, "amount", "float", "INTEGER")
    assert re.search(r"(?<!\w)(77777|88888)(?!\w)", message), message


@not_yet(MISMATCH_NOT_RAISED)
def test_money_column_with_integral_sample_raises(duckdb_cursor):
    # Every sampled row is a multiple of 100 cents, so the sample binds a scale of zero.
    cents = [Decimal(i) / 100 for i in range(100000)]
    duckdb_cursor.register("frame", pd.DataFrame({"amount": pd.Series(cents, dtype=object)}))
    message = invalid_input_message(duckdb_cursor, "SELECT amount FROM frame LIMIT 4")
    assert_names(message, "amount", "decimal.Decimal", "DECIMAL(3,0)")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(3.0, id="float"),
        pytest.param(np.float64(3.0), id="numpy_float64"),
        pytest.param(HalfFloat(3.0), id="float_subclass"),
        pytest.param(-0.0, id="negative_zero"),
        pytest.param(Decimal("3"), id="decimal"),
        pytest.param(Decimal("1E+2"), id="decimal_exponent"),
        pytest.param(Decimal("-0"), id="decimal_negative_zero"),
        pytest.param(Shade.DARK, id="int_enum"),
    ],
)
@pytest.mark.parametrize(("base", "bound"), INTEGER_BOUNDS)
def test_integral_value_of_another_number_type_converts(duckdb_cursor, base, bound, value):
    duckdb_cursor.register("frame", column(base, value))
    assert read_row(duckdb_cursor) == [(bound, int(value))]


@pytest.mark.parametrize(
    ("base", "value", "bound"),
    [
        pytest.param(-(2**64), 1e20, "HUGEINT", id="float_into_HUGEINT"),
        pytest.param(-(2**64), Decimal("1E+30"), "HUGEINT", id="decimal_into_HUGEINT"),
        pytest.param(2**127, float(2**100), "UHUGEINT", id="float_into_UHUGEINT"),
        pytest.param(2**63, float(2**63), "UBIGINT", id="float_into_UBIGINT"),
    ],
)
def test_integral_value_past_64_bits_converts(duckdb_cursor, base, value, bound):
    duckdb_cursor.register("frame", column(base, value))
    assert read_row(duckdb_cursor) == [(bound, int(value))]


@not_yet(MISMATCH_NOT_NAMED)
@pytest.mark.parametrize(
    ("base", "value", "bound"),
    [
        pytest.param(1, 2**40, "INTEGER", id="INTEGER"),
        pytest.param(2**40, 2**63, "BIGINT", id="BIGINT"),
        pytest.param(2**63, -3, "UBIGINT", id="UBIGINT_negative"),
        pytest.param(2**63, float(2**64), "UBIGINT", id="UBIGINT_from_a_float"),
        pytest.param(2**40, 2**130, "BIGINT", id="BIGINT_past_128_bits"),
        pytest.param(-(2**64), 2**130, "HUGEINT", id="HUGEINT_past_128_bits"),
    ],
)
def test_integer_outside_the_bound_width_raises(duckdb_cursor, base, value, bound):
    duckdb_cursor.register("frame", column(base, value))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", UNSAMPLED, python_type_name(value), bound)


@pytest.mark.parametrize(
    ("base", "value", "expected"),
    [
        pytest.param(Decimal("1.5"), 3.7, ("DECIMAL(2,1)", Decimal("3.7")), id="float_that_round_trips"),
        pytest.param(Decimal("1.5"), 7, ("DECIMAL(2,1)", Decimal("7.0")), id="int"),
        pytest.param(Decimal("1.25"), Decimal("3.5"), ("DECIMAL(3,2)", Decimal("3.50")), id="narrower_decimal"),
        pytest.param(
            Decimal("1.5"),
            Decimal("1.25"),
            None,
            id="extra_scale",
            marks=not_yet("a decimal with more scale than the bound type is rounded instead of failing"),
        ),
        pytest.param(
            Decimal("1"),
            2.5,
            None,
            id="fractional_float_into_scale_zero",
            marks=not_yet("a float that needs more scale than the bound type is rounded instead of failing"),
        ),
    ],
)
def test_decimal_value_must_fit_the_bound_scale(duckdb_cursor, base, value, expected):
    duckdb_cursor.register("frame", column(base, value))
    if expected is not None:
        assert read_row(duckdb_cursor) == [expected]
    else:
        assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED, python_type_name(value), "DECIMAL")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0.5, ("FLOAT", 0.5), id="exact_in_float"),
        pytest.param(
            0.1,
            None,
            id="needs_double",
            marks=not_yet("a double that a FLOAT cannot hold exactly is narrowed instead of failing"),
        ),
    ],
)
def test_float_column_from_decimal_nan_takes_only_exact_values(duckdb_cursor, value, expected):
    duckdb_cursor.register("frame", column(Decimal("NaN"), value))
    if expected is not None:
        assert read_row(duckdb_cursor) == [expected]
    else:
        assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED, "float", "FLOAT")


@pytest.mark.parametrize(
    ("base", "value", "expected"),
    [
        pytest.param(JAN_FIRST, datetime.date(2024, 5, 6), ("TIMESTAMP", datetime.datetime(2024, 5, 6)), id="date"),
        pytest.param(
            datetime.date(2024, 1, 1),
            datetime.datetime(2024, 5, 6),
            ("DATE", datetime.date(2024, 5, 6)),
            id="midnight_datetime",
        ),
        pytest.param(
            datetime.date(2024, 1, 1),
            pd.Timestamp("2024-05-06"),
            ("DATE", datetime.date(2024, 5, 6)),
            id="midnight_pandas_timestamp",
        ),
        pytest.param(
            datetime.date(2024, 1, 1),
            NAIVE,
            None,
            id="datetime_with_a_time_of_day",
            marks=not_yet("a datetime with a time of day outside a DATE sample loses its time instead of failing"),
        ),
        pytest.param(
            datetime.date(2024, 1, 1),
            pd.Timestamp("2024-05-06 07:08"),
            None,
            id="pandas_timestamp_with_a_time_of_day",
            marks=not_yet("a timestamp with a time of day outside a DATE sample loses its time instead of failing"),
        ),
    ],
)
def test_date_and_timestamp_convert_when_exact(duckdb_cursor, base, value, expected):
    duckdb_cursor.register("frame", column(base, value))
    if expected is not None:
        assert read_row(duckdb_cursor) == [expected]
    else:
        assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED, python_type_name(value), "DATE")


@pytest.mark.parametrize(
    ("base", "value", "expected"),
    [
        pytest.param(0.5, 7, ("DOUBLE", 7.0), id="int_into_double"),
        pytest.param(0.5, 2**53 + 1, ("DOUBLE", 9007199254740992.0), id="int_past_double_precision"),
        pytest.param(0.5, Decimal("0.1"), ("DOUBLE", 0.1), id="decimal_into_double"),
        pytest.param(0.5, float("inf"), ("DOUBLE", float("inf")), id="infinity_into_double"),
        pytest.param(b"a", "x", ("BLOB", b"x"), id="text_into_blob"),
        pytest.param(b"a", r"\x41", ("BLOB", b"A"), id="escaped_text_into_blob"),
        pytest.param(
            uuid.UUID(int=1),
            "12345678-1234-5678-1234-567812345678",
            ("UUID", uuid.UUID("12345678-1234-5678-1234-567812345678")),
            id="text_into_uuid",
        ),
        pytest.param(
            uuid.UUID(int=1), b"0123456789abcdef", ("UUID", uuid.UUID(bytes=b"0123456789abcdef")), id="bytes_into_uuid"
        ),
        pytest.param(uuid.UUID(int=1), 2**127, ("UUID", uuid.UUID(int=2**127)), id="int_into_uuid"),
    ],
)
def test_value_the_analyzer_would_keep_converts_as_when_sampled(duckdb_cursor, base, value, expected):
    duckdb_cursor.register("frame", column(base, value))
    assert read_row(duckdb_cursor) == [expected]


def test_decimal_past_the_capped_width_rounds_as_when_sampled(duckdb_cursor):
    wide = Decimal("123456789012345678901234567890123456.0")
    duckdb_cursor.register("frame", column(wide, Decimal("1.234"), alternate=(Decimal("1.23"),)))
    assert read_row(duckdb_cursor) == [("DECIMAL(38,2)", Decimal("1.234").quantize(Decimal("0.01")))]


def test_decimal_nan_reads_nan_in_a_double_column(duckdb_cursor):
    duckdb_cursor.register("frame", column(0.5, Decimal("NaN")))
    rows = duckdb_cursor.execute(f"SELECT typeof(amount), isnan(amount) FROM frame LIMIT 1 OFFSET {UNSAMPLED}")
    assert rows.fetchall() == [("DOUBLE", True)]


def test_timestamp_nanoseconds_outside_the_sample_truncate_to_microseconds(duckdb_cursor):
    duckdb_cursor.register("frame", column(JAN_FIRST, pd.Timestamp("2024-05-06 07:08:09.123456789")))
    assert read_row(duckdb_cursor) == [("TIMESTAMP", datetime.datetime(2024, 5, 6, 7, 8, 9, 123456))]


def test_decimal_past_38_digits_outside_the_sample_reads_as_double(duckdb_cursor):
    wide = Decimal("1.234567890123456789012345678901234567890")
    duckdb_cursor.register("frame", column(0.5, wide))
    assert read_row(duckdb_cursor) == [("DOUBLE", pytest.approx(float(wide), rel=1e-15))]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param((1, 2.5), [("DOUBLE", 1.0), ("DOUBLE", 2.5)], id="int_and_float"),
        pytest.param((b"a", "x"), [("BLOB", b"a"), ("BLOB", b"x")], id="bytes_and_text"),
        pytest.param(
            (0.5, 2**53 + 1), [("DOUBLE", 0.5), ("DOUBLE", 9007199254740992.0)], id="int_past_double_precision"
        ),
        pytest.param(
            (1, 2**63, 2**127),
            [("DOUBLE", 1.0), ("DOUBLE", float(2**63)), ("DOUBLE", float(2**127))],
            id="ints_past_64_bits_bind_double",
        ),
        pytest.param(
            (Decimal("Infinity"), Decimal("1.1")),
            [("FLOAT", float("inf")), ("FLOAT", float(np.float32(1.1)))],
            id="decimal_infinity_binds_float",
        ),
    ],
)
def test_fully_sampled_mixes_keep_their_values(duckdb_cursor, values, expected):
    duckdb_cursor.register("frame", fully_sampled(*values))
    assert duckdb_cursor.execute("SELECT typeof(amount), amount FROM frame").fetchall() == expected


def test_value_in_a_column_short_enough_to_sample_fully_keeps_its_value(duckdb_cursor):
    duckdb_cursor.register("frame", column(1, 2.5, rows=1511, at=1500))
    assert read_row(duckdb_cursor, at=1500) == [("DOUBLE", 2.5)]


@not_yet("text outside the sample is parsed into the bound type instead of failing")
@pytest.mark.parametrize(
    ("base", "value", "bound"),
    [
        pytest.param(1, "5", "INTEGER", id="integer"),
        pytest.param(1, "2.5", "INTEGER", id="fraction_into_integer"),
        pytest.param(1, np.str_("5"), "INTEGER", id="numpy_str_into_integer"),
        pytest.param(0.5, "5", "DOUBLE", id="double"),
        pytest.param(Decimal("1.5"), "5", "DECIMAL(2,1)", id="decimal"),
        pytest.param(datetime.timedelta(days=1), "5", "INTERVAL", id="interval"),
        pytest.param(datetime.date(2024, 1, 1), "2024-01-02", "DATE", id="date"),
        pytest.param(JAN_FIRST, "2024-01-02", "TIMESTAMP", id="timestamp"),
        pytest.param(datetime.time(1), "01:02:03", "TIME", id="time"),
        pytest.param([1], "[1]", "INTEGER[]", id="list"),
        pytest.param(["a"], "[b, c]", "VARCHAR[]", id="text_list"),
        pytest.param({"x": 1}, "{'x': 2}", "STRUCT(x INTEGER)", id="struct"),
        pytest.param({"key": ["a"], "value": [1]}, "{a=2}", "MAP(VARCHAR, INTEGER)", id="map"),
    ],
)
def test_text_is_never_parsed_outside_blob_and_uuid(duckdb_cursor, base, value, bound):
    duckdb_cursor.register("frame", column(base, value))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", UNSAMPLED, python_type_name(value), bound)


@not_yet("a value outside a BOOLEAN sample is cast to a bool instead of failing")
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(7, id="int"),
        pytest.param(1, id="one"),
        pytest.param(0, id="zero"),
        pytest.param(1.0, id="float"),
        pytest.param(Decimal("1"), id="decimal"),
        pytest.param("true", id="text"),
    ],
)
def test_boolean_column_takes_only_bools(duckdb_cursor, value):
    duckdb_cursor.register("frame", column(True, value))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", UNSAMPLED, python_type_name(value), "BOOLEAN")


@not_yet("a number outside a BOOLEAN leaf sample is cast to a bool instead of failing")
@pytest.mark.parametrize(
    ("base", "value", "alternate"),
    [
        pytest.param({"x": True}, {"x": 7}, (), id="struct_leaf"),
        pytest.param([True], [7], (), id="list_leaf"),
        pytest.param({"a": True}, {"x": 7}, ({"b": False},), id="map_value"),
    ],
)
def test_boolean_leaf_takes_only_bools(duckdb_cursor, base, value, alternate):
    duckdb_cursor.register("frame", column(base, value, alternate=alternate))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", UNSAMPLED)


@not_yet(MISMATCH_NOT_RAISED)
@pytest.mark.parametrize(
    ("base", "value", "bound"),
    [
        pytest.param(2**127, uuid.UUID(int=2), "UHUGEINT", id="uuid_into_uhugeint"),
        pytest.param(b"a", uuid.UUID(int=2), "BLOB", id="uuid_into_blob"),
        pytest.param(datetime.time(1), NAIVE, "TIME", id="datetime_into_time"),
        pytest.param(1, [2], "INTEGER", id="list_into_integer"),
    ],
)
def test_value_of_another_kind_raises(duckdb_cursor, base, value, bound):
    duckdb_cursor.register("frame", column(base, value))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", UNSAMPLED, python_type_name(value), bound)


@pytest.mark.parametrize(
    ("base", "value", "expected"),
    [
        pytest.param(
            JAN_FIRST,
            np.datetime64("2024-05-06T07:08:09", "us"),
            ("TIMESTAMP", NAIVE),
            id="microseconds_into_timestamp",
        ),
        pytest.param(
            JAN_FIRST, np.datetime64("2024-05-06T07:08:09", "s"), ("TIMESTAMP", NAIVE), id="seconds_into_timestamp"
        ),
        pytest.param(
            JAN_FIRST,
            np.datetime64("2024-05-06", "D"),
            ("TIMESTAMP", datetime.datetime(2024, 5, 6)),
            id="days_into_timestamp",
        ),
        pytest.param(
            datetime.date(2024, 1, 1),
            np.datetime64("2024-05-06", "D"),
            ("DATE", datetime.date(2024, 5, 6)),
            id="days_into_date",
        ),
        pytest.param(
            datetime.date(2024, 1, 1),
            np.datetime64("2024-05-06T00:00:00", "us"),
            ("DATE", datetime.date(2024, 5, 6)),
            id="midnight_microseconds_into_date",
        ),
        pytest.param(
            datetime.date(2024, 1, 1),
            np.datetime64("2024-05-06T07:08:09", "us"),
            "DATE",
            id="microseconds_with_a_time_of_day_into_date",
            marks=not_yet("a datetime64 with a time of day outside a DATE sample loses its time instead of failing"),
        ),
        pytest.param(
            JAN_FIRST,
            np.datetime64("2024-05-06T07:08:09", "ns"),
            "TIMESTAMP",
            id="nanoseconds_into_timestamp",
            marks=not_yet(MISMATCH_NOT_NAMED),
        ),
        pytest.param(
            datetime.date(2024, 1, 1),
            np.datetime64("2024-05-06", "ns"),
            "DATE",
            id="nanoseconds_into_date",
            marks=not_yet(MISMATCH_NOT_NAMED),
        ),
    ],
)
def test_datetime64_follows_its_unit(duckdb_cursor, base, value, expected):
    duckdb_cursor.register("frame", column(base, value))
    if isinstance(expected, tuple):
        assert read_row(duckdb_cursor) == [expected]
    else:
        assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED, expected)


@not_yet("a time zone outside the sample is dropped or assumed instead of failing")
@pytest.mark.parametrize(
    ("base", "value", "bound"),
    [
        pytest.param(JAN_FIRST, AWARE, "TIMESTAMP", id="aware_datetime_into_timestamp"),
        pytest.param(JAN_FIRST, pd.Timestamp(AWARE), "TIMESTAMP", id="aware_pandas_timestamp_into_timestamp"),
        pytest.param(datetime.date(2024, 1, 1), AWARE, "DATE", id="aware_datetime_into_date"),
        pytest.param(AWARE, datetime.date(2024, 5, 6), "TIMESTAMP WITH TIME ZONE", id="date_into_zoned_timestamp"),
        pytest.param(AWARE, NAIVE, "TIMESTAMP WITH TIME ZONE", id="naive_datetime_into_zoned_timestamp"),
        pytest.param(AWARE, pd.Timestamp(NAIVE), "TIMESTAMP WITH TIME ZONE", id="naive_pandas_timestamp_into_zoned"),
        pytest.param(
            AWARE, np.datetime64("2024-05-06", "D"), "TIMESTAMP WITH TIME ZONE", id="datetime64_days_into_zoned"
        ),
        pytest.param(
            AWARE,
            np.datetime64("2024-05-06T07:08:09", "us"),
            "TIMESTAMP WITH TIME ZONE",
            id="datetime64_microseconds_into_zoned",
        ),
        pytest.param(AWARE_TIME, NAIVE_TIME, "TIME WITH TIME ZONE", id="naive_time_into_zoned_time"),
        pytest.param(NAIVE_TIME, AWARE_TIME, "TIME", id="aware_time_into_time"),
    ],
)
def test_time_zone_is_never_assumed_or_dropped(duckdb_cursor, base, value, bound):
    duckdb_cursor.register("frame", column(base, value))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", UNSAMPLED, python_type_name(value), bound)


@not_yet("a time zone outside the sample of a nested leaf is dropped instead of failing")
@pytest.mark.parametrize(
    ("base", "value"),
    [
        pytest.param([JAN_FIRST], [AWARE], id="list_leaf"),
        pytest.param({"x": JAN_FIRST}, {"x": AWARE}, id="struct_leaf"),
    ],
)
def test_time_zone_is_never_dropped_in_a_nested_leaf(duckdb_cursor, base, value):
    duckdb_cursor.register("frame", column(base, value))
    assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED)


@not_yet("a fully sampled mix of naive and zoned values binds a zoned or naive type instead of VARCHAR")
@pytest.mark.parametrize(
    "values",
    [
        pytest.param((NAIVE_TIME, AWARE_TIME), id="time_then_aware_time"),
        pytest.param((AWARE_TIME, NAIVE_TIME), id="aware_time_then_time"),
        pytest.param((datetime.date(2024, 5, 6), AWARE), id="date_then_aware_datetime"),
        pytest.param((AWARE, datetime.date(2024, 5, 6)), id="aware_datetime_then_date"),
        pytest.param((NAIVE, AWARE), id="naive_then_aware_datetime"),
        pytest.param(([NAIVE_TIME], [AWARE_TIME]), id="list_leaves"),
        pytest.param(([NAIVE_TIME, AWARE_TIME],), id="one_list"),
        pytest.param(({"x": datetime.date(2024, 5, 6)}, {"x": AWARE}), id="struct_leaves"),
        pytest.param((AWARE, NAIVE), id="aware_then_naive_datetime"),
        pytest.param(([AWARE_TIME, NAIVE_TIME],), id="one_list_aware_first"),
        pytest.param(({"x": NAIVE_TIME}, {"x": AWARE_TIME}), id="struct_time_leaves"),
    ],
)
def test_fully_sampled_time_zone_mixes_bind_varchar(duckdb_cursor, values):
    duckdb_cursor.register("frame", fully_sampled(*values))
    rows = duckdb_cursor.execute("SELECT typeof(amount), amount FROM frame").fetchall()
    assert rows == [("VARCHAR", str(value)) for value in values]


NULL_MARKERS = [
    pytest.param(None, id="None"),
    pytest.param(float("nan"), id="float_nan"),
    pytest.param(np.float64("nan"), id="numpy_float64_nan"),
    pytest.param(pd.NA, id="pandas_NA"),
    pytest.param(pd.NaT, id="pandas_NaT"),
    pytest.param(np.datetime64("NaT", "ns"), id="datetime64_NaT"),
    pytest.param(np.ma.masked, id="masked_constant"),
]


@pytest.mark.parametrize("marker", NULL_MARKERS)
def test_null_markers_stay_null(duckdb_cursor, marker):
    bases = {
        "integer": 1,
        "double": 0.5,
        "decimal": Decimal("1.5"),
        "boolean": True,
        "timestamp": JAN_FIRST,
        "date": datetime.date(2024, 1, 1),
        "interval": datetime.timedelta(days=1),
        "blob": b"a",
        "uuid": uuid.UUID(int=1),
        "list": [1],
        "struct": {"x": 1},
    }
    frame = pd.DataFrame({name: column(base, marker)["amount"] for name, base in bases.items()})
    duckdb_cursor.register("frame", frame)
    row = duckdb_cursor.execute(f"SELECT * FROM frame LIMIT 1 OFFSET {UNSAMPLED}").fetchall()
    assert row == [(None,) * len(bases)]


@pytest.mark.parametrize("marker", NULL_MARKERS)
def test_null_markers_stay_null_in_nested_leaves(duckdb_cursor, marker):
    cells = {
        "integer_list": ([1], [marker], [None]),
        "double_list": ([0.5], [marker], [None]),
        "struct": ({"x": 1}, {"x": marker}, {"x": None}),
        "map": ({"key": ["a"], "value": [1]}, {"key": ["k"], "value": [marker]}, {"k": None}),
    }
    frame = pd.DataFrame({name: column(base, value)["amount"] for name, (base, value, _) in cells.items()})
    duckdb_cursor.register("frame", frame)
    row = duckdb_cursor.execute(f"SELECT * FROM frame LIMIT 1 OFFSET {UNSAMPLED}").fetchall()
    assert row == [tuple(expected for _, _, expected in cells.values())]


@not_yet(MISMATCH_NOT_RAISED)
@pytest.mark.parametrize(
    ("base", "value", "alternate"),
    [
        pytest.param([1], [2.5], (), id="list_leaf"),
        pytest.param({"x": 1}, {"x": 2.5}, (), id="struct_leaf"),
        pytest.param({"a": 1}, {"x": 2.5}, ({"b": 2},), id="map_value"),
        pytest.param({"a": 1, "b": 2}, {"a": 1, "b": 2.5}, (), id="second_struct_field"),
        pytest.param({"a": 1, "b": 2}, {"a": 1, "b": "5"}, (), id="text_in_a_struct_field"),
        pytest.param({"x": 1}, (2.5,), (), id="tuple_into_struct"),
        pytest.param({"x": 1}, (2,), (), id="exact_tuple_into_struct"),
        pytest.param({"key": [1, 2], "value": ["a", "b"]}, {"key": [2.5], "value": ["c"]}, (), id="map_key"),
        pytest.param({"x": [1]}, {"x": [2.5]}, (), id="list_in_a_struct"),
        pytest.param([{"x": 1}], [{"x": 2.5}], (), id="struct_in_a_list"),
        pytest.param([1], (2.5,), (), id="tuple_into_list"),
        pytest.param([1], np.array([2.5], dtype=np.float64), (), id="float_ndarray_into_list"),
    ],
)
def test_nested_values_follow_the_same_rule(duckdb_cursor, base, value, alternate):
    duckdb_cursor.register("frame", column(base, value, alternate=alternate))
    assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([2], id="list"),
        pytest.param(np.array([2], dtype=np.int64), id="int64_ndarray"),
    ],
)
def test_nested_values_that_fit_exactly_convert(duckdb_cursor, value):
    duckdb_cursor.register("frame", column([1], value))
    assert read_row(duckdb_cursor) == [("INTEGER[]", [2])]


STRUCT_LEAF_REFUSES = "a STRUCT or MAP leaf refuses a value that the same LIST leaf converts exactly"


@pytest.mark.parametrize(
    ("base", "value", "alternate", "expected"),
    [
        pytest.param(
            [Decimal("1.5")], [3.7], (), ("DECIMAL(2,1)[]", [Decimal("3.7")]), id="float_in_a_decimal_list_leaf"
        ),
        pytest.param(
            {"x": Decimal("1.5")},
            {"x": 3.7},
            (),
            ("STRUCT(x DECIMAL(2,1))", {"x": Decimal("3.7")}),
            id="float_in_a_decimal_struct_leaf",
            marks=not_yet(STRUCT_LEAF_REFUSES),
        ),
        pytest.param(
            {"a": Decimal("1.5")},
            {"k": 3.7},
            ({"b": Decimal("2.5")},),
            ("MAP(VARCHAR, DECIMAL(2,1))", {"k": Decimal("3.7")}),
            id="float_in_a_decimal_map_value",
            marks=not_yet(STRUCT_LEAF_REFUSES),
        ),
        pytest.param(
            [datetime.date(2024, 1, 1)],
            [datetime.datetime(2024, 5, 7)],
            (),
            ("DATE[]", [datetime.date(2024, 5, 7)]),
            id="midnight_in_a_date_list_leaf",
        ),
        pytest.param(
            {"x": datetime.date(2024, 1, 1)},
            {"x": datetime.datetime(2024, 5, 7)},
            (),
            ("STRUCT(x DATE)", {"x": datetime.date(2024, 5, 7)}),
            id="midnight_in_a_date_struct_leaf",
            marks=not_yet(STRUCT_LEAF_REFUSES),
        ),
    ],
)
def test_struct_and_list_leaves_agree(duckdb_cursor, base, value, alternate, expected):
    duckdb_cursor.register("frame", column(base, value, alternate=alternate))
    assert read_row(duckdb_cursor) == [expected]


@pytest.mark.parametrize(
    ("base", "value"),
    [
        pytest.param(
            {"x": "a"},
            {"x": [1]},
            id="list_in_a_struct_text_leaf",
            marks=not_yet("a list in a text leaf is stringified"),
        ),
        pytest.param(["a"], [b"ab"], id="bytes_in_a_list_text_leaf", marks=not_yet("bytes in a text leaf are decoded")),
        pytest.param({"x": "a"}, {"x": b"ab"}, id="bytes_in_a_struct_text_leaf", marks=not_yet(MISMATCH_NOT_NAMED)),
    ],
)
def test_nested_text_leaves_follow_the_same_rule(duckdb_cursor, base, value):
    duckdb_cursor.register("frame", column(base, value))
    assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED)


def test_int_in_a_text_list_leaf_reads_as_text(duckdb_cursor):
    duckdb_cursor.register("frame", column(["a"], [7]))
    assert read_row(duckdb_cursor) == [("VARCHAR[]", ["7"])]


def test_bool_outside_a_struct_text_leaf_sample_reads_as_text(duckdb_cursor):
    duckdb_cursor.register("frame", column({"x": "a"}, {"x": True}))
    assert read_row(duckdb_cursor) == [("STRUCT(x VARCHAR)", {"x": "true"})]


@not_yet("a bool in a text list leaf fails without naming its column and row")
def test_bool_outside_a_list_text_leaf_sample_names_column_and_row(duckdb_cursor):
    duckdb_cursor.register("frame", column(["a"], [True]))
    with pytest.raises(duckdb.TypeMismatchException) as error:
        duckdb_cursor.execute("SELECT amount FROM frame").fetchall()
    assert_names(str(error.value), "amount", UNSAMPLED)


@not_yet(MISMATCH_NOT_NAMED)
@pytest.mark.parametrize(
    ("base", "value"),
    [
        pytest.param([], [1], id="empty_lists"),
        pytest.param({"x": None}, {"x": 1}, id="struct_of_none"),
        pytest.param({}, {"a": 1}, id="empty_dicts"),
    ],
)
def test_value_outside_a_sample_of_null_children_raises(duckdb_cursor, base, value):
    duckdb_cursor.register("frame", column(base, value))
    assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED)


@not_yet(MISMATCH_NOT_RAISED)
@pytest.mark.parametrize(
    ("label", "query"),
    [
        pytest.param("amount", "SELECT amount FROM frame", id="text_label"),
        pytest.param(17, 'SELECT "17" FROM frame', id="int_label"),
    ],
)
def test_error_names_column_row_and_types(duckdb_cursor, label, query):
    duckdb_cursor.register("frame", column(1, 2.5, label=label))
    message = invalid_input_message(duckdb_cursor, query)
    assert_names(message, label, UNSAMPLED, "float", "INTEGER")


@not_yet(MISMATCH_NOT_RAISED)
def test_error_row_counts_across_batches(duckdb_cursor):
    duckdb_cursor.register("frame", column(1, 2.5, rows=300000, at=250001))
    assert_names(invalid_input_message(duckdb_cursor), "amount", 250001)


@not_yet(MISMATCH_NOT_RAISED)
def test_error_names_one_of_several_rows_across_partitions(duckdb_cursor):
    frame = column(1, 2.5, rows=300000, at=102001)
    frame.loc[204801, "amount"] = 3.5
    duckdb_cursor.register("frame", frame)
    message = invalid_input_message(duckdb_cursor)
    assert re.search(r"(?<!\w)(102001|204801)(?!\w)", message), message


@not_yet(MISMATCH_NOT_NAMED)
@pytest.mark.parametrize(
    ("later", "bound"),
    [
        pytest.param({}, "BIGINT", id="wide_int_binds_bigint"),
        pytest.param({1300: 2**63}, "HUGEINT", id="order_of_wide_ints_binds_hugeint"),
    ],
)
def test_error_for_a_fully_sampled_column_names_column_row_and_types(duckdb_cursor, later, bound):
    frame = column(1, 2**127, rows=1500, at=1234)
    for row, value in later.items():
        frame.loc[row, "amount"] = value
    duckdb_cursor.register("frame", frame)
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", 1234, "int", bound)


@not_yet("an error raised while converting one value does not name its column and row yet")
def test_other_conversion_errors_keep_their_type_with_column_and_row(duckdb_cursor):
    entries = {"key": ["a", "b"], "value": [1, 2]}
    duckdb_cursor.register("frame", column(entries, {"key": ["a", "a"], "value": [1, 2]}, at=2500))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", 2500)
    assert "Map keys must be unique" in message


@not_yet("a value the conversion cannot handle outside the sample does not name its column and row yet")
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.int64(5), id="numpy_int64"),
        pytest.param(np.bool_(True), id="numpy_bool"),
        pytest.param(np.float32(2.5), id="numpy_float32"),
        pytest.param(np.timedelta64(5, "s"), id="numpy_timedelta64"),
        pytest.param(PlainInstance(), id="class_instance"),
        pytest.param(Holder.Inner(), id="nested_class_instance"),
        pytest.param(NumberModuled(), id="class_with_a_number_as_module"),
    ],
)
def test_numpy_scalars_and_class_instances_keep_not_implemented(duckdb_cursor, value):
    duckdb_cursor.register("frame", column(1, value))
    with pytest.raises(duckdb.NotImplementedException) as error:
        duckdb_cursor.execute("SELECT amount FROM frame").fetchall()
    assert_names(str(error.value), "amount", UNSAMPLED, python_type_name(value))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(7, id="int"),
        pytest.param(2.5, id="float"),
        pytest.param(b"xy", id="bytes"),
        pytest.param({"x": 2}, id="dict"),
        pytest.param([1, 2], id="list"),
        pytest.param(True, id="bool"),
        pytest.param(uuid.UUID(int=2), id="uuid"),
    ],
)
def test_text_column_keeps_stringifying(duckdb_cursor, value):
    duckdb_cursor.register("frame", column("a", value))
    assert read_row(duckdb_cursor) == [("VARCHAR", str(value))]


def text_object_array_with(value):
    values = np.array(["a"] * ROWS, dtype=object)
    values[UNSAMPLED] = value
    return {"amount": values}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.datetime64("NaT", "ns"), id="datetime64_NaT"),
        pytest.param(np.timedelta64("NaT", "ns"), id="timedelta64_NaT"),
        pytest.param(np.float32("nan"), id="float32_nan"),
    ],
)
@pytest.mark.parametrize(
    ("make_source", "at"),
    [
        pytest.param(lambda value: column("a", value), UNSAMPLED, id="outside_the_sample"),
        pytest.param(lambda value: fully_sampled("a", value), 1, id="fully_sampled"),
        pytest.param(text_object_array_with, UNSAMPLED, id="numpy_object_array"),
    ],
)
def test_text_column_stringifies_numpy_not_a_time_and_float32_nan(duckdb_cursor, make_source, at, value):
    duckdb_cursor.register("frame", make_source(value))
    assert read_row(duckdb_cursor, at=at) == [("VARCHAR", str(value))]


@not_yet(MISMATCH_NOT_RAISED)
def test_all_null_sample_still_checks_later_values(duckdb_cursor):
    frame = column(None, 5, at=1001)
    frame.loc[2002, "amount"] = 2.5
    duckdb_cursor.register("frame", frame)
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", 2002, "float", "INTEGER")


BOOL_REFUSED = "a bool in a number column is refused instead of reading as 0 or 1"


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(column(1, True), [("INTEGER", 1)], id="integer", marks=not_yet(BOOL_REFUSED)),
        pytest.param(column(2**40, False), [("BIGINT", 0)], id="bigint", marks=not_yet(BOOL_REFUSED)),
        pytest.param(column(2**63, True), [("UBIGINT", 1)], id="ubigint", marks=not_yet(BOOL_REFUSED)),
        pytest.param(column(-(2**64), False), [("HUGEINT", 0)], id="hugeint", marks=not_yet(BOOL_REFUSED)),
        pytest.param(column(2**127, True), [("UHUGEINT", 1)], id="uhugeint", marks=not_yet(BOOL_REFUSED)),
        pytest.param(column(0.5, True), [("DOUBLE", 1.0)], id="double", marks=not_yet(BOOL_REFUSED)),
        pytest.param(column(Decimal("NaN"), True), [("FLOAT", 1.0)], id="float", marks=not_yet(BOOL_REFUSED)),
        pytest.param(
            column(Decimal("1.5"), True), [("DECIMAL(2,1)", Decimal("1.0"))], id="decimal", marks=not_yet(BOOL_REFUSED)
        ),
        pytest.param(column([1], [True]), [("INTEGER[]", [1])], id="list_leaf", marks=not_yet(BOOL_REFUSED)),
        pytest.param(column({"x": 1}, {"x": True}), [("STRUCT(x INTEGER)", {"x": 1})], id="struct_leaf"),
        pytest.param(
            column({"key": ["a"], "value": [1]}, {"key": ["k"], "value": [True]}),
            [("MAP(VARCHAR, INTEGER)", {"k": 1})],
            id="map_value",
        ),
    ],
)
def test_bool_outside_the_sample_of_a_number_column(duckdb_cursor, frame, expected):
    duckdb_cursor.register("frame", frame)
    assert read_row(duckdb_cursor) == expected


@not_yet(BOOL_REFUSED)
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param((1.5, True), [("DOUBLE", 1.5), ("DOUBLE", 1.0)], id="double"),
        pytest.param(
            (Decimal("1.5"), True), [("DECIMAL(2,1)", Decimal("1.5")), ("DECIMAL(2,1)", Decimal("1.0"))], id="decimal"
        ),
        pytest.param((True, 7), [("INTEGER", 1), ("INTEGER", 7)], id="bool_first"),
    ],
)
def test_bool_in_a_number_column(duckdb_cursor, values, expected):
    duckdb_cursor.register("frame", fully_sampled(*values))
    assert duckdb_cursor.execute("SELECT typeof(amount), amount FROM frame").fetchall() == expected


def frame_with_a_refused_value():
    frame = column(1, "x")
    frame["count"] = pd.Series(range(ROWS), dtype="int64")
    return frame


def test_unread_column_does_not_raise(duckdb_cursor):
    duckdb_cursor.register("frame", frame_with_a_refused_value())
    assert duckdb_cursor.execute('SELECT "count" FROM frame WHERE "count" = 2').fetchall() == [(2,)]


def test_read_column_with_the_same_value_raises(duckdb_cursor):
    duckdb_cursor.register("frame", frame_with_a_refused_value())
    with pytest.raises(duckdb.InvalidInputException):
        duckdb_cursor.execute("SELECT amount FROM frame").fetchall()


def array_value():
    return duckdb.Value([1, 2, 3], duckdb.sqltype("INTEGER[3]"))


def union_value():
    return duckdb.Value(1, duckdb.sqltype("UNION(i INTEGER, s VARCHAR)"))


def enum_value():
    return duckdb.Value("a", duckdb.sqltype("ENUM('a', 'b')"))


def value_cells(*cells):
    return pd.DataFrame({"amount": pd.Series(list(cells), dtype=object)})


def all_null_sample_then(*placed):
    values = [None] * ROWS
    for row, value in placed:
        values[row] = value
    return pd.DataFrame({"amount": pd.Series(values, dtype=object)})


def assert_connection_answers(con):
    assert con.execute("SELECT 42").fetchall() == [(42,)]
    assert con.cursor().execute("SELECT 42").fetchall() == [(42,)]


@pytest.mark.parametrize(
    ("make_frame", "query", "expected"),
    [
        pytest.param(
            lambda: value_cells(array_value(), array_value()),
            "SELECT typeof(amount), amount FROM frame",
            [("VARCHAR", str(array_value()))] * 2,
            id="fixed_size_arrays",
        ),
        pytest.param(
            lambda: value_cells(union_value(), union_value()),
            "SELECT typeof(amount), amount FROM frame",
            [("VARCHAR", str(union_value()))] * 2,
            id="unions",
        ),
        pytest.param(
            lambda: value_cells(enum_value(), enum_value()),
            "SELECT typeof(amount), amount FROM frame",
            [("VARCHAR", str(enum_value()))] * 2,
            id="enums",
        ),
        pytest.param(
            lambda: value_cells([array_value(), array_value()]),
            "SELECT DISTINCT typeof(amount) FROM frame",
            [("VARCHAR",)],
            id="list_of_arrays",
        ),
        pytest.param(
            lambda: value_cells({"v": array_value()}, {"v": array_value()}),
            "SELECT DISTINCT typeof(amount) FROM frame",
            [("VARCHAR",)],
            id="dicts_of_arrays",
        ),
        pytest.param(
            lambda: value_cells(duckdb.Value([[1, 2, 3]], duckdb.sqltype("INTEGER[3][]"))),
            "SELECT DISTINCT typeof(amount) FROM frame",
            [("VARCHAR",)],
            id="list_of_array_type",
        ),
        pytest.param(
            lambda: value_cells(duckdb.Value({"v": [1, 2, 3]}, duckdb.sqltype("STRUCT(v INTEGER[3])"))),
            "SELECT DISTINCT typeof(amount) FROM frame",
            [("VARCHAR",)],
            id="struct_of_array_type",
        ),
        pytest.param(
            lambda: value_cells(duckdb.Value({"k": 1}, duckdb.sqltype("MAP(VARCHAR, UNION(i INTEGER))"))),
            "SELECT DISTINCT typeof(amount) FROM frame",
            [("VARCHAR",)],
            id="map_of_union_type",
        ),
        pytest.param(
            lambda: value_cells(duckdb.DoubleValue(2.5), duckdb.DoubleValue(2.5)),
            "SELECT typeof(amount), amount FROM frame",
            [("DOUBLE", 2.5)] * 2,
            id="doubles",
        ),
        pytest.param(
            lambda: all_null_sample_then((1001, array_value()), (2002, [1, 2, 3])),
            "SELECT DISTINCT typeof(amount) FROM frame",
            [("VARCHAR",)],
            id="array_found_after_an_all_null_sample",
        ),
    ],
)
def test_duckdb_value_elements(duckdb_cursor, make_frame, query, expected):
    duckdb_cursor.register("frame", make_frame())
    assert duckdb_cursor.execute(query).fetchall() == expected
    assert_connection_answers(duckdb_cursor)


def test_sampled_array_value_with_plain_lists_after_it_binds_varchar(duckdb_cursor):
    duckdb_cursor.execute("SET pandas_analyze_sample=1")
    duckdb_cursor.register("frame", value_cells(array_value(), *([[1, 2, 3]] * 9)))
    assert duckdb_cursor.execute("SELECT DISTINCT typeof(amount) FROM frame").fetchall() == [("VARCHAR",)]
    assert_connection_answers(duckdb_cursor)


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        pytest.param(
            [duckdb.Value([1, 2], duckdb.sqltype("BIGINT[]")), [3, 4]],
            [("BIGINT[]", [1, 2]), ("BIGINT[]", [3, 4])],
            id="list_value_then_a_plain_list",
        ),
        pytest.param(
            [duckdb.Value({"v": 1}, duckdb.sqltype("STRUCT(v BIGINT)")), {"v": 2}],
            [("STRUCT(v BIGINT)", {"v": 1}), ("STRUCT(v BIGINT)", {"v": 2})],
            id="struct_value_then_a_plain_dict",
        ),
        pytest.param(
            [duckdb.Value({"k": 1}, duckdb.sqltype("MAP(VARCHAR, INTEGER)"))] * 2,
            [("MAP(VARCHAR, INTEGER)", {"k": 1})] * 2,
            id="map_values",
        ),
        pytest.param(
            [duckdb.HugeIntegerValue(2**70), 2**70],
            [("HUGEINT", 2**70)] * 2,
            id="hugeint_value_then_a_plain_int",
        ),
        pytest.param(
            [duckdb.UnsignedHugeIntegerValue(2**127)] * 2,
            [("UHUGEINT", 2**127)] * 2,
            id="uhugeint_values",
        ),
    ],
)
def test_value_elements_keep_their_type(duckdb_cursor, cells, expected):
    duckdb_cursor.register("frame", value_cells(*cells))
    assert duckdb_cursor.execute("SELECT typeof(amount), amount FROM frame").fetchall() == expected


@pytest.mark.parametrize(
    ("value", "rows_equal_to_a"),
    [
        pytest.param(duckdb.Value('{"a": 1}', duckdb.sqltype("JSON")), 0, id="json"),
        # Only the lower case cell equals 'a': a NOCASE column would match the upper case one too.
        pytest.param(duckdb.Value("a", duckdb.sqltype("VARCHAR COLLATE NOCASE")), 1, id="nocase"),
        pytest.param(duckdb.Value(["1"], duckdb.sqltype("JSON[]")), 0, id="list_of_json"),
    ],
)
def test_value_element_of_an_aliased_or_collated_type_binds_plain_varchar(duckdb_cursor, value, rows_equal_to_a):
    duckdb_cursor.register("frame", value_cells(value, "A"))
    assert duckdb_cursor.execute("SELECT typeof(amount), amount FROM frame").fetchall() == [
        ("VARCHAR", str(value)),
        ("VARCHAR", "A"),
    ]
    assert duckdb_cursor.execute("SELECT count(*) FROM frame WHERE amount = 'a'").fetchall() == [(rows_equal_to_a,)]


_TYPE_NAME_ON_THE_DEFAULT_CONNECTION_CHILD = """
import duckdb
import pandas as pd

value = duckdb.Value(5, "INTEGER")
frame = pd.DataFrame({"amount": pd.Series([value, value], dtype=object)})
try:
    print("rows", QUERY.fetchall(), flush=True)
except duckdb.Error as error:
    print("raised", type(error).__name__, str(error).splitlines()[0], flush=True)
print("answers", duckdb.sql("SELECT 42").fetchall(), flush=True)
"""
TYPE_NAME_CHILD_TIMEOUT = 60


@pytest.mark.parametrize(
    "query",
    [
        pytest.param('duckdb.sql("SELECT amount FROM frame")', id="sql"),
        pytest.param('duckdb.default_connection().sql("SELECT amount FROM frame")', id="default_connection_sql"),
        pytest.param("duckdb.from_df(frame)", id="from_df"),
        pytest.param('duckdb.execute("SELECT amount FROM frame")', id="execute"),
    ],
)
def test_value_element_with_a_type_name_on_the_default_connection_raises(query):
    # Resolving the type name needs the connection this query's bind holds, so a wait for it would hang the process.
    script = _TYPE_NAME_ON_THE_DEFAULT_CONNECTION_CHILD.replace("QUERY", query)
    try:
        child = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=TYPE_NAME_CHILD_TIMEOUT,
            check=False,
            env=quiet_child_env(),
        )
    except subprocess.TimeoutExpired as hang:
        pytest.fail(f"the child did not finish within {TYPE_NAME_CHILD_TIMEOUT} seconds: {hang.stderr!r}")
    assert child.returncode == 0, child.stderr[-2000:]
    raised, answers = child.stdout.splitlines()
    assert raised.startswith("raised InvalidInputException "), raised
    assert answers == "answers [(42,)]"


_TYPE_NAME_WHILE_ANOTHER_THREAD_BINDS_CHILD = """
import collections
import json
import sys
import threading
import time

import duckdb
import pandas as pd

rows, at = int(sys.argv[1]), int(sys.argv[2])
cells = [1] * rows
cells[at] = duckdb.Value(5, "INTEGER")
columns = {f"c{i}": pd.Series([str(j) for j in range(rows)], dtype=object) for i in range(40)}
columns["amount"] = pd.Series(cells, dtype=object)
heavy = pd.DataFrame(columns)
light = pd.DataFrame({"x": pd.Series(["a", "b"], dtype=object)})
con = duckdb.connect()
con.execute("SET enable_progress_bar = false")
idle = repr(con.sql("SELECT sum(amount) FROM heavy").fetchall())
stop = time.monotonic() + 4
outcomes = collections.Counter()
default_binds = 0


def read_heavy():
    while time.monotonic() < stop:
        try:
            outcomes[repr(con.sql("SELECT sum(amount) FROM heavy").fetchall())] += 1
        except duckdb.Error as error:
            outcomes[type(error).__name__] += 1


def bind_light_on_the_default_connection():
    global default_binds
    while time.monotonic() < stop:
        duckdb.sql("SELECT * FROM light")
        default_binds += 1


threads = [threading.Thread(target=read_heavy), threading.Thread(target=bind_light_on_the_default_connection)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print(json.dumps({"idle": idle, "outcomes": sorted(outcomes), "default_binds": default_binds}), flush=True)
"""


CROSS_THREAD_CHILD_TIMEOUT = 20


@pytest.mark.parametrize("at", [pytest.param(0, id="sampled"), pytest.param(UNSAMPLED, id="outside_the_sample")])
def test_value_element_with_a_type_name_while_another_thread_binds_on_the_default_connection(at):
    try:
        child = subprocess.run(
            [sys.executable, "-I", "-c", _TYPE_NAME_WHILE_ANOTHER_THREAD_BINDS_CHILD, str(ROWS), str(at)],
            capture_output=True,
            text=True,
            timeout=CROSS_THREAD_CHILD_TIMEOUT,
            check=False,
            env=quiet_child_env(),
        )
    except subprocess.TimeoutExpired as hang:
        pytest.fail(f"the child did not finish within {CROSS_THREAD_CHILD_TIMEOUT} seconds: {hang.stderr!r}")
    assert child.returncode == 0, child.stderr[-2000:]
    report = json.loads(child.stdout)
    total = f"[({ROWS + 4},)]"
    assert report["idle"] == total
    # A read either sums the column or is refused while the other thread holds the default connection.
    assert set(report["outcomes"]) <= {total, "InvalidInputException"}, report
    assert report["default_binds"] > 0


_TYPE_NAME_WHILE_ANOTHER_THREAD_REPLACES_THE_CELLS_CHILD = """
import sys
import threading

import duckdb
import numpy as np

sys.setswitchinterval(1e-4)
rows = 300
cells = np.empty(rows, dtype=object)
cells[0] = 1
for i in range(1, rows):
    cells[i] = duckdb.Value(5, "INTEGER")
data = {"amount": cells}
con = duckdb.connect()
con.execute("SET enable_progress_bar = false")
con.execute("SET pandas_analyze_sample=1")
con.execute("SET threads=1")
stop = threading.Event()


def replace_cells():
    while not stop.is_set():
        for i in range(1, rows):
            cells[i] = duckdb.Value(5, "INTEGER")


replacer = threading.Thread(target=replace_cells)
replacer.start()
try:
    for _ in range(20):
        print(con.execute("SELECT sum(amount) FROM data").fetchall(), flush=True)
finally:
    stop.set()
    replacer.join()
"""


def test_value_elements_with_a_type_name_while_another_thread_replaces_them():
    # Resolving a type name runs Python, so the other thread can replace a cell and drop its Value while it converts.
    child = subprocess.run(
        [sys.executable, "-I", "-c", _TYPE_NAME_WHILE_ANOTHER_THREAD_REPLACES_THE_CELLS_CHILD],
        capture_output=True,
        text=True,
        timeout=TYPE_NAME_CHILD_TIMEOUT,
        check=False,
        env=quiet_child_env(),
    )
    assert child.returncode == 0, child.stderr[-2000:]
    assert child.stdout.splitlines() == ["[(1496,)]"] * 20


def test_value_element_whose_type_is_not_a_duckdb_type_raises(duckdb_cursor):
    value = duckdb.IntegerValue(1)
    value.type = 5
    with pytest.raises(duckdb.InvalidInputException, match="should be of type DuckDBPyType"):
        duckdb_cursor.register("frame", value_cells(value, value))
    assert_connection_answers(duckdb_cursor)


@not_yet(MISMATCH_NOT_RAISED)
def test_double_value_element_outside_an_integer_sample_raises(duckdb_cursor):
    duckdb_cursor.register("frame", column(1, duckdb.DoubleValue(2.5)))
    assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED, "INTEGER")
    assert_connection_answers(duckdb_cursor)


@not_yet(MISMATCH_NOT_RAISED)
def test_array_value_element_outside_a_list_sample_raises(duckdb_cursor):
    duckdb_cursor.register("frame", column([1], array_value()))
    assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED)
    assert_connection_answers(duckdb_cursor)


@pytest.mark.parametrize(
    ("make_value", "expected"),
    [
        pytest.param(lambda: duckdb.IntegerValue(5), 5, id="integer_value"),
        pytest.param(lambda: duckdb.DoubleValue(3.0), 3, id="integral_double_value"),
    ],
)
def test_exact_value_element_outside_an_integer_sample_converts(duckdb_cursor, make_value, expected):
    duckdb_cursor.register("frame", column(1, make_value()))
    assert read_row(duckdb_cursor) == [("INTEGER", expected)]
    assert_connection_answers(duckdb_cursor)


def udf_result(value, type_name):
    def prepare(con):
        con.create_function("produce", lambda: value, [], duckdb.sqltype(type_name), side_effects=True)
        return lambda: con.execute("SELECT produce()").fetchall()

    return prepare


def typed_value_parameter(value, type_name):
    def prepare(con):
        parameter = duckdb.Value(value, duckdb.sqltype(type_name))
        return lambda: con.execute("SELECT $1, typeof($1)", [parameter]).fetchall()

    return prepare


def cast_parameter(con):
    return lambda: con.execute("SELECT ?::INTEGER", [2.5]).fetchall()


def appended_row(con):
    con.execute("CREATE TABLE target (x INTEGER)")
    appender = con.appender("target")

    def append():
        appender.append_row(2.5)
        appender.flush()
        return con.execute("SELECT x FROM target").fetchall()

    return append


def values_relation(con):
    return lambda: con.values([2.5]).fetchall()


@pytest.mark.parametrize(
    ("prepare", "expected"),
    [
        pytest.param(udf_result(2.5, "INTEGER"), [(2,)], id="udf_float_into_integer"),
        pytest.param(udf_result({"x": 2.5}, "STRUCT(x INTEGER)"), [({"x": 2},)], id="udf_float_into_struct_leaf"),
        pytest.param(
            udf_result({"x": 3.7}, "STRUCT(x DECIMAL(2,1))"),
            duckdb.ConversionException,
            id="udf_float_into_decimal_struct_leaf",
        ),
        pytest.param(
            udf_result({"a": 3.7}, "MAP(VARCHAR, DECIMAL(2,1))"),
            duckdb.ConversionException,
            id="udf_float_into_decimal_map_value",
        ),
        pytest.param(
            udf_result({"x": JAN_FIRST}, "STRUCT(x DATE)"), duckdb.ConversionException, id="udf_datetime_into_date_leaf"
        ),
        pytest.param(udf_result({"x": True}, "STRUCT(x INTEGER)"), [({"x": 1},)], id="udf_bool_into_struct_leaf"),
        pytest.param(udf_result({"x": 7}, "STRUCT(x BOOLEAN)"), [({"x": True},)], id="udf_int_into_bool_leaf"),
        pytest.param(udf_result(7, "BOOLEAN"), [(True,)], id="udf_int_into_boolean"),
        pytest.param(
            udf_result((True,), "STRUCT(x INTEGER)"), duckdb.TypeMismatchException, id="udf_bool_tuple_into_struct"
        ),
        pytest.param(udf_result([True], "INTEGER[]"), duckdb.TypeMismatchException, id="udf_bool_into_list_leaf"),
        pytest.param(udf_result(True, "INTEGER"), duckdb.TypeMismatchException, id="udf_bool_into_integer"),
        pytest.param(udf_result(3.7, "DECIMAL(2,1)"), [(Decimal("3.7"),)], id="udf_float_into_decimal"),
        pytest.param(
            typed_value_parameter({"x": 3.7}, "STRUCT(x DECIMAL(2,1))"),
            duckdb.ConversionException,
            id="value_float_into_decimal_struct_leaf",
        ),
        pytest.param(
            typed_value_parameter({"x": JAN_FIRST}, "STRUCT(x DATE)"),
            duckdb.ConversionException,
            id="value_datetime_into_date_leaf",
        ),
        pytest.param(
            typed_value_parameter({"x": 7}, "STRUCT(x BOOLEAN)"),
            [({"x": True}, "STRUCT(x BOOLEAN)")],
            id="value_int_into_bool_leaf",
        ),
        pytest.param(typed_value_parameter(True, "INTEGER"), [(True, "BOOLEAN")], id="value_bool_keeps_boolean"),
        pytest.param(typed_value_parameter(2.5, "INTEGER"), [(2, "INTEGER")], id="value_float_into_integer"),
        pytest.param(cast_parameter, [(2,)], id="parameter_cast_to_integer"),
        pytest.param(appended_row, [(2,)], id="appender_float_into_integer"),
        pytest.param(values_relation, [(2.5,)], id="values_float"),
    ],
)
def test_other_conversions_keep_their_casts(duckdb_cursor, prepare, expected):
    convert = prepare(duckdb_cursor)
    if isinstance(expected, list):
        assert convert() == expected
    else:
        with pytest.raises(expected):
            convert()


def test_bool_refused_by_a_number_type_names_the_expected_type(duckdb_cursor):
    duckdb_cursor.create_function("produce", lambda: True, [], duckdb.sqltype("INTEGER"), side_effects=True)
    with pytest.raises(duckdb.TypeMismatchException) as error:
        duckdb_cursor.execute("SELECT produce()").fetchall()
    message = str(error.value)
    assert "%s" not in message
    assert "Expected a value of type INTEGER, but got a value of type boolean" in message


@not_yet("the error for a long string outside the sample repeats the whole string")
def test_error_preview_of_a_long_string_is_bounded(duckdb_cursor):
    duckdb_cursor.register("frame", column(1, "é" * 100000, at=2000))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", 2000, "str", "INTEGER")
    assert len(message) < 500
    assert "é" * 60 in message
    assert "é" * 61 not in message
    assert_connection_answers(duckdb_cursor)


@not_yet("the error for a large dict outside the sample spells out a STRUCT of every key")
def test_error_for_a_large_dict_previews_only_its_length(duckdb_cursor):
    duckdb_cursor.register("frame", column(1, {i: i for i in range(100000)}))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", UNSAMPLED, "dict", "INTEGER", 100000)
    assert len(message) < 500
    assert_connection_answers(duckdb_cursor)


@not_yet("a str outside the sample that cannot be encoded fails without naming its column and row")
def test_error_for_an_unencodable_string_names_column_row_and_types(duckdb_cursor):
    duckdb_cursor.register("frame", column(1, "\ud800"))
    message = invalid_input_message(duckdb_cursor)
    assert_names(message, "amount", UNSAMPLED, "str", "INTEGER")
    assert_connection_answers(duckdb_cursor)


@not_yet(MISMATCH_NOT_NAMED)
def test_error_names_a_module_qualified_type(duckdb_cursor):
    duckdb_cursor.register("frame", column(2**40, uuid.UUID(int=2)))
    assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED, "uuid.UUID", "BIGINT")


@pytest.mark.parametrize(
    ("frame", "raised"),
    [
        pytest.param(fully_sampled(RaisingStr(), RaisingStr()), "ValueError: bad string", id="sampled_text_column"),
        pytest.param(column("a", RaisingStr()), "ValueError: bad string", id="text_column"),
        pytest.param(column(1, TuplelessDecimal("2")), "ZeroDivisionError: no tuple", id="integer_column"),
    ],
)
def test_python_error_from_a_conversion_passes_through(duckdb_cursor, frame, raised):
    duckdb_cursor.register("frame", frame)
    with pytest.raises(duckdb.Error) as error:
        duckdb_cursor.execute("SELECT amount FROM frame").fetchall()
    assert type(error.value) is duckdb.Error
    assert re.fullmatch(rf"Traceback \(most recent call last\):\n(.*\n)*{re.escape(raised)}\n?", str(error.value))


FALLBACK_ROWS = 4000


def fallback_column(placed, *, base=None, index=None):
    values = [base] * FALLBACK_ROWS
    for row, value in placed:
        values[row] = value
    return pd.DataFrame({"amount": pd.Series(values, dtype=object, index=index)})


def interleaved_cells():
    # A walk that ignores the column's stride meets the text neighbour before the 7.
    cells = np.empty((FALLBACK_ROWS, 2), dtype=object)
    cells[:, 0] = None
    cells[:, 1] = "text"
    cells[3, 0] = 7
    return cells


@pytest.mark.parametrize(
    ("make_source", "query", "expected"),
    [
        pytest.param(
            lambda: fallback_column([(3, 7)], index=[0] * FALLBACK_ROWS),
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("INTEGER", 7)],
            id="duplicate_index_labels",
        ),
        pytest.param(
            lambda: fallback_column([(3, np.ma.masked), (5, 5)]),
            "SELECT typeof(amount), amount FROM frame LIMIT 3 OFFSET 3",
            [("INTEGER", None), ("INTEGER", None), ("INTEGER", 5)],
            id="masked_constant_before_the_value",
        ),
        pytest.param(
            lambda: fallback_column([(3, Decimal("NaN")), (5, 5)]),
            "SELECT typeof(amount), amount::VARCHAR FROM frame WHERE amount IS NOT NULL",
            [("FLOAT", "nan"), ("FLOAT", "5.0")],
            id="decimal_nan_before_the_value",
        ),
        pytest.param(
            lambda: {"amount": np.array([None] * 2001, dtype=object)},
            "SELECT typeof(amount), count(*) FROM frame GROUP BY ALL",
            [('"NULL"', 2001)],
            id="all_none_ndarray",
        ),
        pytest.param(
            lambda: {"amount": np.array([None] * 2999 + [5], dtype=object)},
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("INTEGER", 5)],
            id="ndarray_with_one_late_value",
        ),
        pytest.param(
            lambda: {
                "amount": np.ma.masked_array(
                    np.array(["text"] * 5001 + [7], dtype=object), mask=[True] * 5001 + [False]
                )
            },
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("INTEGER", 7)],
            id="masked_ndarray_with_one_unmasked_value",
        ),
        pytest.param(
            lambda: fallback_column([(3, "x")], index=range(100, FALLBACK_ROWS + 100)),
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("VARCHAR", "x")],
            id="shifted_index",
        ),
        pytest.param(
            lambda: fallback_column([(3, "x")], index=[f"r{i}" for i in range(FALLBACK_ROWS)]),
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("VARCHAR", "x")],
            id="text_index",
        ),
        pytest.param(
            lambda: fallback_column([(3, 7)], index=range(FALLBACK_ROWS, 0, -1)),
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("INTEGER", 7)],
            id="reversed_index",
        ),
        pytest.param(
            lambda: fallback_column([(3, 2.5)], base=float("nan")),
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("DOUBLE", 2.5)],
            id="nan_sample",
        ),
        pytest.param(
            lambda: fallback_column([(3, np.datetime64("NaT", "ns")), (5, 5)]),
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("INTEGER", 5)],
            id="datetime64_nat_before_the_value",
        ),
        pytest.param(
            lambda: fallback_column([(3, pd.NA), (5, 5)]),
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("INTEGER", 5)],
            id="pandas_NA_before_the_value",
        ),
        pytest.param(
            lambda: fallback_column([(3, pd.NaT), (5, 5)]),
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("INTEGER", 5)],
            id="pandas_NaT_before_the_value",
        ),
        pytest.param(
            lambda: pd.DataFrame(interleaved_cells(), columns=["amount", "other"], copy=False),
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("INTEGER", 7)],
            id="strided_series",
        ),
        pytest.param(
            lambda: {"amount": interleaved_cells()[:, 0]},
            "SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL",
            [("INTEGER", 7)],
            id="strided_ndarray",
        ),
        pytest.param(
            lambda: pd.DataFrame({"amount": pd.Series([None] * 2001, dtype=object)}),
            "SELECT typeof(amount), count(*) FROM frame GROUP BY ALL",
            [('"NULL"', 2001)],
            id="all_none_series",
        ),
    ],
)
def test_all_null_sample_falls_back_to_the_first_value(duckdb_cursor, make_source, query, expected):
    duckdb_cursor.register("frame", make_source())
    assert duckdb_cursor.execute(query).fetchall() == expected


def test_all_null_sample_fallback_calls_no_pandas(duckdb_cursor, monkeypatch):
    calls = []

    def recording(name, original):
        def record(self, *args, **kwargs):
            calls.append(name)
            return original(self, *args, **kwargs)

        return record

    for name in ("first_valid_index", "notna", "isna"):
        monkeypatch.setattr(pd.Series, name, recording(name, getattr(pd.Series, name)))
    frame = fallback_column([(3, 7)])
    duckdb_cursor.register("frame", frame)
    rows = duckdb_cursor.execute("SELECT typeof(amount), amount FROM frame WHERE amount IS NOT NULL").fetchall()
    assert rows == [("INTEGER", 7)]
    assert calls == []


def object_array_with(value):
    values = np.array([1] * ROWS, dtype=object)
    values[UNSAMPLED] = value
    return {"amount": values}


def test_numpy_object_array_converts_an_exact_value(duckdb_cursor):
    duckdb_cursor.register("frame", object_array_with(3.0))
    assert read_row(duckdb_cursor) == [("INTEGER", 3)]


@not_yet(MISMATCH_NOT_RAISED)
def test_numpy_object_array_refuses_a_fractional_value(duckdb_cursor):
    duckdb_cursor.register("frame", object_array_with(2.5))
    assert_names(invalid_input_message(duckdb_cursor), "amount", UNSAMPLED, "float", "INTEGER")

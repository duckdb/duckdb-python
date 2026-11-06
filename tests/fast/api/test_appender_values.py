import datetime as dt
import decimal
import os
import subprocess
import sys
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

import duckdb
import numpy as np
import pandas as pd
from duckdb.sqltypes import BIGINT
from duckdb.value.constant import IntegerValue, StringValue, TimestampNanosecondValue, Value

TZ_PLUS_2 = dt.timezone(dt.timedelta(hours=2))
UUID_TEXT = "12345678-1234-5678-1234-567812345678"

# One value per branch of the Python to Value conversion: the object kind it dispatches on, then the
# sub-branch inside its handler (integer width, float special value, Decimal exponent kind, time zone,
# dict layout).
VALUES = [
    ("None", None),
    ("bool True", True),
    ("int 0", 0),
    ("int 1", 1),
    ("int -5", -5),
    ("int 300", 300),
    ("int 70000", 70000),
    ("int 2**31", 2**31),
    ("int 2**63", 2**63),
    ("int -(2**63)-1", -(2**63) - 1),
    ("int 2**64", 2**64),
    ("int 2**127", 2**127),
    ("int 2**128", 2**128),
    ("float 1.5", 1.5),
    ("float 2.5", 2.5),
    ("float 0.1", 0.1),
    ("float 1e300", 1e300),
    ("float nan", float("nan")),
    ("float inf", float("inf")),
    ("np.float64 1.5", np.float64(1.5)),
    ("str abc", "abc"),
    ("str empty", ""),
    ("str past the inline length", "a string longer than the twelve bytes kept inline"),
    ("str with NUL", "a\x00b"),
    ("str lone surrogate", "\ud800"),
    ("str 42", "42"),
    ("str 2.5", "2.5"),
    ("str true", "true"),
    ("str date", "2024-01-02"),
    ("str timestamp", "2024-01-02 03:04:05.123456"),
    ("str timestamp +02", "2024-01-02 03:04:05+02"),
    ("str time", "03:04:05"),
    ("str interval", "1 day"),
    ("str enum member", "a"),
    ("str enum member 299", "m299"),
    ("str enum member 69999", "m69999"),
    ("str list literal", "[1, 2]"),
    ("str struct literal", "{'a': 1, 'b': x}"),
    ("str map literal", "{k1=1}"),
    ("str json object", '{"a": 1}'),
    ("str bits", "0101"),
    ("str uuid", UUID_TEXT),
    ("str wkt point", "POINT (1 2)"),
    ("pd.NaT", pd.NaT),
    ("pd.NA", pd.NA),
    ("Decimal 1.25", decimal.Decimal("1.25")),
    ("Decimal -0.001", decimal.Decimal("-0.001")),
    ("Decimal 1E+3", decimal.Decimal("1E+3")),
    ("Decimal 12345.6", decimal.Decimal("12345.6")),
    ("Decimal 40 digits", decimal.Decimal("1234567890123456789012345678901.123456789")),
    ("Decimal NaN", decimal.Decimal("NaN")),
    ("Decimal Infinity", decimal.Decimal("Infinity")),
    ("uuid.UUID", uuid.UUID(UUID_TEXT)),
    ("datetime naive", dt.datetime(2024, 1, 2, 3, 4, 5, 123456)),
    ("datetime +02", dt.datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=TZ_PLUS_2)),
    ("pd.Timestamp ns", pd.Timestamp("2024-01-02 03:04:05.123456789")),
    ("time naive", dt.time(3, 4, 5, 123456)),
    ("time +02", dt.time(3, 4, 5, 123456, tzinfo=TZ_PLUS_2)),
    ("date", dt.date(2024, 1, 2)),
    ("timedelta", dt.timedelta(days=1, seconds=5, microseconds=7)),
    ("timedelta negative", dt.timedelta(days=-1)),
    ("bytes 0x0102", b"\x01\x02"),
    ("bytes 0101", b"0101"),
    ("bytes abc", b"abc"),
    ("bytearray", bytearray(b"\x01\x02")),
    ("memoryview", memoryview(b"\x01\x02")),
    ("list [1, 2]", [1, 2]),
    ("list []", []),
    ("list [1, None]", [1, None]),
    ("list [None, None]", [None, None]),
    ("list ['a', 'b']", ["a", "b"]),
    ("list [1, 'a']", [1, "a"]),
    ("list [1.5, 2]", [1.5, 2]),
    ("list [1, 2, 3]", [1, 2, 3]),
    ("list nested", [[1, 2], [3]]),
    ("list of dicts", [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]),
    ("list of dicts with a None field", [{"a": None, "b": "x"}]),
    ("list of None beside a dict of None", [None, {"a": None, "b": None}]),
    ("tuple (1, 'x')", (1, "x")),
    ("tuple (1, 2)", (1, 2)),
    ("tuple ()", ()),
    ("dict a b", {"a": 1, "b": "x"}),
    ("dict b a", {"b": "x", "a": 1}),
    ("dict A B", {"A": 1, "B": "x"}),
    ("dict z", {"z": 1}),
    ("dict {}", {}),
    ("dict k1 k2", {"k1": 1, "k2": 2}),
    ("dict key value lists", {"key": ["k1", "k2"], "value": [1, 2]}),
    ("dict int key", {1: 10}),
    ("np.ma.masked", np.ma.masked),
    ("np.array", np.array([1, 2])),
    ("np.int64", np.int64(5)),
    ("np.bool_", np.bool_(True)),
    ("np.datetime64 s", np.datetime64("2024-01-02T03:04:05")),
    ("np.datetime64 ns", np.datetime64("2024-01-02T03:04:05.123456789")),
    ("duckdb Value BIGINT", Value(5, BIGINT)),
    ("duckdb IntegerValue", IntegerValue(5)),
    ("duckdb StringValue", StringValue("abc")),
    ("set", {1}),
    ("object", object()),
]

# Every target type the conversion switches on, and the types it reaches only through its cast fallback.
TYPES = [
    "BOOLEAN",
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL(4,1)",
    "DECIMAL(18,3)",
    "DECIMAL(38,10)",
    "BIGNUM",
    "VARCHAR",
    "JSON",
    "BLOB",
    "BIT",
    "UUID",
    "mood",
    "enum_300",
    "enum_70000",
    "DATE",
    "TIME",
    "TIME_NS",
    "TIMETZ",
    "TIMESTAMP",
    "TIMESTAMP_S",
    "TIMESTAMP_MS",
    "TIMESTAMP_NS",
    "TIMESTAMPTZ",
    "TIMESTAMPTZ_NS",
    "INTERVAL",
    "INTEGER[]",
    "VARCHAR[]",
    "INTEGER[2]",
    "INTEGER[][]",
    "STRUCT(a INTEGER, b VARCHAR)",
    "STRUCT(a INTEGER, b VARCHAR)[]",
    "TUPLE(INTEGER, VARCHAR)",
    "MAP(VARCHAR, INTEGER)",
    "UNION(i INTEGER, s VARCHAR)",
    "UNION(u1 DOUBLE, u2 DOUBLE)",
    "VARIANT",
    "GEOMETRY",
]

# Each executemany costs milliseconds on a debug build, so the reference runs them on parallel cursors of the
# same database; one cursor per thread, because a connection refuses a second caller.
REFERENCE_THREADS = 8

# 300 and 70000 members need a 2 and a 4 byte dictionary index.
ENUMS = {
    "mood": "('a', 'b')",
    "enum_300": "(SELECT 'm' || range FROM range(300))",
    "enum_70000": "(SELECT 'm' || range FROM range(70000))",
}

# Each executemany into the 70000 member ENUM costs tens of milliseconds, and beyond these values it converts as
# the two member ENUM does.
WIDE_ENUM_VALUES = {
    "None",
    "int 1",
    "str abc",
    "str enum member",
    "str enum member 299",
    "str enum member 69999",
    "duckdb StringValue",
}

# A string longer than this is stored outside its vector, in a separate heap.
INLINE_LENGTH = 12
LONG_STRING_VALUES = {
    "VARCHAR": lambda i, text: text,
    "JSON": lambda i, text: f'"{text}"',
    "BLOB": lambda i, text: text.encode(),
    "VARCHAR[]": lambda i, text: [text, text.upper()],
    "STRUCT(a INTEGER, b VARCHAR)": lambda i, text: {"a": i, "b": text},
    "STRUCT(a INTEGER, b VARCHAR)[]": lambda i, text: [{"a": i, "b": text}, None],
    "MAP(VARCHAR, INTEGER)": lambda i, text: {text: i},
}


# Connecting costs a quarter second on a debug build, so the tests share one database.
@pytest.fixture(scope="module")
def database():
    connection = duckdb.connect()
    # GLOBAL, so the reference's cursors convert in the same time zone.
    connection.execute("SET GLOBAL TimeZone = 'America/New_York'")
    yield connection
    connection.close()


@pytest.fixture
def con(database):
    yield database
    database.execute("DROP TABLE IF EXISTS ref")
    database.execute("DROP TABLE IF EXISTS tgt")
    for name in ENUMS:
        database.execute(f"DROP TYPE IF EXISTS {name}")


def failure(call, *args):
    try:
        call(*args)
    except Exception as error:
        return type(error).__name__, str(error).splitlines()[0]
    return None


def cells(column_type):
    if column_type == "enum_70000":
        return [(label, value) for label, value in VALUES if label in WIDE_ENUM_VALUES]
    return VALUES


FINER_THAN_A_MICROSECOND = ("ns", "ps", "fs", "as")


def python_value_of_numpy_scalar(value):
    """The value the appender converts: a numpy scalar, or an array of no dimensions, as its Python value."""
    if isinstance(value, (float, str, bytes)) or not (
        isinstance(value, np.generic) or (type(value) is np.ndarray and value.ndim == 0)
    ):
        return value
    kind = value.dtype.kind
    finer = kind in "Mm" and np.datetime_data(value.dtype)[0] in FINER_THAN_A_MICROSECOND
    if kind == "M" and finer:
        nanoseconds = value.astype("datetime64[ns]")
        return None if np.isnat(nanoseconds) else TimestampNanosecondValue(np.datetime_as_string(nanoseconds))
    item = value.astype("timedelta64[us]").item() if kind == "m" and finer else value.item()
    return value if isinstance(item, np.generic) else item


def reference_failures(con, values):
    cursors = [con.cursor() for _ in range(REFERENCE_THREADS)]

    def share(thread):
        cursor = cursors[thread]
        # The appender reads a numpy scalar as its Python value and converts that as executemany converts it, so the
        # reference takes the value the appender reads; for a datetime64 that differs from executemany's own reading
        # (its integer), a nanosecond one is a TIMESTAMP_NS value since no Python value holds it, and for an int64 or
        # bool executemany refuses the scalar
        return [
            (
                i,
                failure(
                    cursor.executemany, "INSERT INTO ref VALUES (?, ?)", [[i, python_value_of_numpy_scalar(value)]]
                ),
            )
            for i, (_, value) in enumerate(values)
            if i % REFERENCE_THREADS == thread
        ]

    with ThreadPoolExecutor(REFERENCE_THREADS) as pool:
        by_index = dict(pair for part in pool.map(share, range(REFERENCE_THREADS)) for pair in part)
    for cursor in cursors:
        cursor.close()
    return [by_index[i] for i in range(len(values))]


def outcomes(con, table, column_type, values, failures):
    tag = "union_tag(x)" if column_type.startswith("UNION") else "NULL"
    rows = {
        i: (text, member) for i, text, member in con.execute(f"SELECT id, x::VARCHAR, {tag} FROM {table}").fetchall()
    }
    result = {}
    for i, ((label, _), error) in enumerate(zip(values, failures, strict=True)):
        if error is None:
            result[label] = ("stored", *rows.get(i, ("nothing",)))
        else:
            result[label] = ("refused" if i not in rows else "refused but stored", *error)
    return result


@pytest.mark.parametrize("column_type", TYPES)
def test_a_value_converts_as_an_executemany_parameter(con, column_type):
    if column_type in ENUMS:
        con.execute(f"CREATE TYPE {column_type} AS ENUM {ENUMS[column_type]}")
    con.execute(f"CREATE TABLE ref (id INTEGER, x {column_type})")
    con.execute(f"CREATE TABLE tgt (id INTEGER, x {column_type})")
    values = cells(column_type)
    expected = outcomes(con, "ref", column_type, values, reference_failures(con, values))

    app = con.appender("tgt")
    appended = [failure(app.append_row, i, value) for i, (_, value) in enumerate(values)]
    app.flush()

    assert outcomes(con, "tgt", column_type, values, appended) == expected


@pytest.mark.parametrize("column_type", LONG_STRING_VALUES)
def test_strings_past_the_inline_length_survive_several_flushes(con, column_type):
    con.execute(f"CREATE TABLE ref (id INTEGER, x {column_type})")
    con.execute(f"CREATE TABLE tgt (id INTEGER, x {column_type})")
    texts = [f"row {i:02d} holds a string past the inline length" for i in range(20)]
    assert all(len(text) > INLINE_LENGTH for text in texts)
    values = [(f"row {i}", LONG_STRING_VALUES[column_type](i, text)) for i, text in enumerate(texts)]
    expected = outcomes(con, "ref", column_type, values, reference_failures(con, values))
    assert all(outcome[0] == "stored" for outcome in expected.values())

    app = con.appender("tgt")
    appended = []
    for i, (_, value) in enumerate(values):
        appended.append(failure(app.append_row, i, value))
        if i % 5 == 4:
            app.flush()

    assert outcomes(con, "tgt", column_type, values, appended) == expected


NANOSECOND_STAMP = np.datetime64("2024-05-08T01:02:03.123456789", "ns")


class _ItemRaises(np.int64):
    def item(self):
        msg = "no item"
        raise ValueError(msg)


class TestNumpyScalars:
    # The appender reads a numpy scalar as con.append reads one in a declared object column: its Python value, a
    # timedelta64 finer than a microsecond at microseconds, and a datetime64 finer than a microsecond as a
    # TIMESTAMP_NS the engine casts into the column.

    @pytest.mark.parametrize(
        ("column_type", "value", "expected"),
        [
            pytest.param("INTEGER", np.int64(2), 2, id="int64_into_integer"),
            pytest.param("DOUBLE", np.float32(2.5), 2.5, id="float32_into_double"),
            pytest.param("BOOLEAN", np.bool_(False), False, id="bool_into_boolean"),
            pytest.param("DATE", np.datetime64("2024-05-08", "D"), dt.date(2024, 5, 8), id="day_datetime64_into_date"),
            pytest.param(
                "TIMESTAMP_S",
                np.datetime64("2024-05-08T01:02:03", "s"),
                dt.datetime(2024, 5, 8, 1, 2, 3),
                id="second_datetime64_into_timestamp_s",
            ),
            pytest.param("INTERVAL", np.timedelta64(2, "D"), dt.timedelta(days=2), id="timedelta64_day"),
            pytest.param("INTERVAL", np.timedelta64(1500, "us"), dt.timedelta(microseconds=1500), id="timedelta64_us"),
            pytest.param("INTERVAL", np.timedelta64(1501, "ns"), dt.timedelta(microseconds=1), id="timedelta64_ns"),
            pytest.param("INTERVAL", np.timedelta64("NaT", "ns"), None, id="timedelta64_nat_is_null"),
            # A month has no fixed length: item() gives the count, which stays an integer
            pytest.param("BIGINT", np.timedelta64(1, "M"), 1, id="month_timedelta64_reads_as_its_integer"),
            pytest.param("VARCHAR", np.str_("a\x00b"), "a\x00b", id="str_scalar_into_varchar"),
            pytest.param("VARCHAR", np.str_("ab\x00"), "ab\x00", id="str_scalar_keeps_a_trailing_nul"),
            pytest.param("VARCHAR", np.str_("\x00\x00"), "\x00\x00", id="str_scalar_of_nuls_keeps_them"),
            pytest.param("BLOB", np.bytes_(b"\x00\xff"), b"\x00\xff", id="bytes_scalar_into_blob"),
            pytest.param("BLOB", np.bytes_(b"\x01\x00"), b"\x01\x00", id="bytes_scalar_keeps_a_trailing_nul"),
            pytest.param("BLOB", np.bytes_(b"\x00\x00"), b"\x00\x00", id="bytes_scalar_of_nuls_keeps_them"),
            pytest.param(
                "STRUCT(a INTEGER)",
                np.array({"a": 1}, dtype=object),
                {"a": 1},
                id="zero_dimensional_object_array_into_struct",
            ),
            pytest.param("INTEGER", np.array(2), 2, id="zero_dimensional_array_into_integer"),
            pytest.param("DOUBLE", np.float64("nan"), float("nan"), id="float64_nan_stays_a_value"),
            pytest.param("DOUBLE", np.ma.masked, None, id="masked_singleton_is_null"),
            pytest.param("TIMESTAMP", np.datetime64("NaT", "ns"), None, id="datetime64_nat_is_null"),
            pytest.param("TIMESTAMP", np.datetime64("NaT", "D"), None, id="day_datetime64_nat_is_null"),
        ],
    )
    def test_numpy_scalars_append_as_their_python_values(self, con, column_type, value, expected):
        con.execute(f"CREATE OR REPLACE TABLE t (x {column_type})")
        with con.appender("t") as appender:
            appender.append_row(value)
        (stored,) = con.execute("SELECT x FROM t").fetchall()[0]
        if isinstance(expected, float) and expected != expected:
            assert stored != stored
        else:
            assert stored == expected

    @pytest.mark.parametrize(
        ("column_type", "value", "text"),
        [
            pytest.param("VARCHAR", np.datetime64("2024-05-08", "D"), "2024-05-08", id="day_unit_reads_as_a_date"),
            pytest.param("VARCHAR", np.datetime64("2024-05", "M"), "2024-05-01", id="month_unit_reads_as_a_date"),
            pytest.param(
                "VARCHAR",
                np.datetime64("2024-05-08T01:02:03", "s"),
                "2024-05-08 01:02:03",
                id="second_unit_reads_as_a_datetime",
            ),
            pytest.param(
                "VARCHAR", NANOSECOND_STAMP, "2024-05-08 01:02:03.123456789", id="nanosecond_unit_into_varchar"
            ),
            pytest.param(
                "TIMESTAMP_NS",
                NANOSECOND_STAMP,
                "2024-05-08 01:02:03.123456789",
                id="nanosecond_unit_into_timestamp_ns",
            ),
            # The engine's TIMESTAMP_NS cast rounds to the nearest microsecond, as for a datetime64[ns] frame column
            pytest.param(
                "TIMESTAMP", NANOSECOND_STAMP, "2024-05-08 01:02:03.123457", id="nanosecond_unit_rounds_into_timestamp"
            ),
            pytest.param(
                "TIMESTAMP_NS",
                np.datetime64(1500, "ps"),
                "1970-01-01 00:00:00.000000001",
                id="picosecond_unit_at_nanoseconds",
            ),
        ],
    )
    def test_datetime64_stores_the_text_of_its_value(self, con, column_type, value, text):
        con.execute(f"CREATE OR REPLACE TABLE t (x {column_type})")
        with con.appender("t") as appender:
            appender.append_row(value)
        assert con.execute("SELECT x::VARCHAR FROM t").fetchall() == [(text,)]

    @pytest.mark.parametrize(
        ("column_type", "value", "match"),
        [
            pytest.param("TIME", np.datetime64("2024-05-08", "D"), r"DATE -> TIME\)", id="day_datetime64_into_time"),
            pytest.param("INTERVAL", np.timedelta64(1, "M"), r"INT\w* -> INTERVAL", id="month_timedelta64"),
            pytest.param("INTERVAL", np.timedelta64(1, "Y"), r"INT\w* -> INTERVAL", id="year_timedelta64"),
            pytest.param("BIGINT", NANOSECOND_STAMP, r"TIMESTAMP_NS -> BIGINT", id="nanosecond_datetime64_into_bigint"),
        ],
    )
    def test_a_numpy_value_without_a_cast_into_the_column_is_refused(self, con, column_type, value, match):
        con.execute(f"CREATE OR REPLACE TABLE t (x {column_type})")
        with pytest.raises(duckdb.ConversionException, match=match), con.appender("t") as appender:
            appender.append_row(value)

    @pytest.mark.parametrize(
        "column_type",
        [
            "TIMESTAMP",
            "TIMESTAMP_MS",
            "TIMESTAMP_NS",
            "TIMESTAMPTZ",
            "TIMESTAMPTZ_NS",
            "DATE",
            "TIME",
            "TIME_NS",
            "VARCHAR",
            # The engine has no cast from TIMESTAMP_NS into these two, for a frame column as for the appender
            "TIMESTAMP_S",
            "TIMETZ",
        ],
    )
    def test_nanosecond_datetime64_stores_what_a_nanosecond_frame_column_stores(self, con, column_type):
        # The engine's cast from TIMESTAMP_NS rounds to the nearest microsecond, so the last stamp, a nanosecond before
        # 1970, stores 1970 in a microsecond column
        stamps = [NANOSECOND_STAMP, np.datetime64("NaT", "ns"), np.datetime64(-1, "ns")]
        con.execute(f"CREATE OR REPLACE TABLE t (i INTEGER, x {column_type})")
        frame = pd.DataFrame({"i": range(len(stamps)), "x": pd.Series(stamps, dtype="datetime64[ns]")})
        frame_error = failure(con.append, "t", frame)
        expected = con.execute("SELECT i, x::VARCHAR FROM t ORDER BY i").fetchall()
        con.execute(f"CREATE OR REPLACE TABLE t (i INTEGER, x {column_type})")
        with con.appender("t") as appender:
            errors = [failure(appender.append_row, i, stamp) for i, stamp in enumerate(stamps)]
        stored = con.execute("SELECT i, x::VARCHAR FROM t ORDER BY i").fetchall()
        if frame_error is None:
            assert (stored, errors) == (expected, [None, None, None])
        else:
            # The frame refuses the whole column, the appender each stamp that is not NaT, with the same cast error
            assert (stored, errors[1]) == ([(1, None)], None)
            for name, message in (errors[0], errors[2]):
                assert name == frame_error[0] == "ConversionException"
                assert frame_error[1].startswith(message)

    def test_a_refused_numpy_value_keeps_the_rows_around_it(self, con):
        con.execute("CREATE OR REPLACE TABLE t (i INTEGER, x TIMESTAMP)")
        with con.appender("t") as appender:
            appender.append_row(1, np.datetime64("2024-05-08", "D"))
            # Past the range of a Python date item() gives the day count, which no cast takes into a TIMESTAMP
            with pytest.raises(duckdb.ConversionException):
                appender.append_row(2, np.datetime64("300000-01-01", "D"))
            appender.append_row(3, np.datetime64("2024-05-09", "D"))
        assert con.execute("SELECT i, x FROM t ORDER BY i").fetchall() == [
            (1, dt.datetime(2024, 5, 8)),
            (3, dt.datetime(2024, 5, 9)),
        ]

    def test_a_raising_reading_keeps_the_rows_around_it(self, con):
        con.execute("CREATE OR REPLACE TABLE t (i INTEGER, x INTEGER)")
        with con.appender("t") as appender:
            appender.append_row(1, np.int64(1))
            with pytest.raises(ValueError, match="no item"):
                appender.append_row(2, _ItemRaises(2))
            appender.append_row(3, np.int64(3))
        assert con.execute("SELECT i, x FROM t ORDER BY i").fetchall() == [(1, 1), (3, 3)]

    def test_rows_of_a_two_dimensional_array_append(self, con):
        con.execute("CREATE OR REPLACE TABLE t (x INTEGER, y INTEGER)")
        with con.appender("t") as appender:
            appender.append_rows(np.array([[1, 2], [3, 4]]))
        assert con.execute("SELECT x, y FROM t ORDER BY x").fetchall() == [(1, 2), (3, 4)]

    def test_nested_numpy_values_keep_their_existing_conversion(self, con):
        # A numpy value inside a list keeps the conversion it had, in the appender, executemany and con.append alike:
        # a nested int64 is refused, a nested float64 is read as a float
        con.execute("CREATE OR REPLACE TABLE t (x DOUBLE[])")
        with con.appender("t") as appender:
            appender.append_row([np.float64(1.5)])
        assert con.execute("SELECT x FROM t").fetchall() == [([1.5],)]
        with pytest.raises(duckdb.NotImplementedException, match=r"numpy\.int64"), con.appender("t") as appender:
            appender.append_row([np.int64(1)])

    def test_an_ordinary_append_imports_no_numpy(self):
        # With numpy installed but not imported, an ordinary append must not import it (40 ms) or, without numpy,
        # retry the failed import for every value
        script = textwrap.dedent(
            """
            import sys
            import duckdb
            con = duckdb.connect()
            import decimal
            con.execute("CREATE TABLE t (x INTEGER, s VARCHAR)")
            with con.appender("t") as appender:
                appender.append_row(1, "a")
                appender.append_row(decimal.Decimal(2), None)
            print("numpy" in sys.modules, con.execute("SELECT x, s FROM t").fetchall())
            """
        )
        assert run_isolated(script) == "False [(1, 'a'), (2, None)]"

    def test_ordinary_values_append_without_numpy(self):
        # numpy is optional; the scalar reading must not touch its types when it cannot be imported
        script = textwrap.dedent(
            """
            import sys
            sys.modules["numpy"] = None
            import duckdb
            con = duckdb.connect()
            con.execute("CREATE TABLE t (x INTEGER, s VARCHAR)")
            import decimal
            with con.appender("t") as appender:
                appender.append_row(1, "a")
                appender.append_row(decimal.Decimal(2), None)
            print(con.execute("SELECT x, s FROM t").fetchall())
            """
        )
        assert run_isolated(script) == "[(1, 'a'), (2, None)]"

    def test_numpy_long_double_is_refused(self, con):
        con.execute("CREATE OR REPLACE TABLE t (x DOUBLE)")
        with pytest.raises(duckdb.NotImplementedException, match="longdouble"), con.appender("t") as appender:
            appender.append_row(np.longdouble("2.5"))


def run_isolated(script):
    # -I keeps a user site file that imports numpy out of the child; an editable install that rebuilds on import
    # otherwise prints its cmake command into the child's stdout
    env = {**os.environ, "SKBUILD_EDITABLE_VERBOSE": "0"}
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()

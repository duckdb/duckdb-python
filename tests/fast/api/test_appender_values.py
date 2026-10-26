import datetime as dt
import decimal
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

import duckdb
import numpy as np
import pandas as pd
from duckdb.sqltypes import BIGINT
from duckdb.value.constant import IntegerValue, StringValue, Value

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


def reference_failures(con, values):
    cursors = [con.cursor() for _ in range(REFERENCE_THREADS)]

    def share(thread):
        cursor = cursors[thread]
        return [
            (i, failure(cursor.executemany, "INSERT INTO ref VALUES (?, ?)", [[i, value]]))
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

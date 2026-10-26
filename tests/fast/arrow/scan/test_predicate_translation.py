"""What the predicate walker hands a source for each predicate the optimizer offers an Arrow scan.

`_duckdb._translate_predicate` plans the predicate over a temporary table through the real optimizer and records,
for every predicate the optimizer offers the scan, the engine's text for it and the node the walker built, or None
where the walker refused it and the engine keeps filtering above the scan.
"""

import datetime
import decimal
import json
import math
import subprocess
import sys
from typing import ClassVar

import _duckdb
import pytest

import duckdb
from duckdb._sources.nodes import Binary, Col, In, Lit, Postfix, Unary

REFUSED = None

COLUMNS = {
    "n": "INTEGER",
    "x": "INTEGER",
    "s": "VARCHAR",
    "f": "DOUBLE",
    "r": "REAL",
    "b": "BOOLEAN",
    "ti": "TINYINT",
    "u": "UBIGINT",
    "h": "HUGEINT",
    "dec": "DECIMAL(10,2)",
    "wide": "DECIMAL(38,4)",
    "bl": "BLOB",
    "d": "DATE",
    "t": "TIME",
    "tn": "TIME_NS",
    "ttz": "TIMETZ",
    "ts": "TIMESTAMP",
    "tms": "TIMESTAMP_MS",
    "tsec": "TIMESTAMP_S",
    "tns": "TIMESTAMP_NS",
    "tz": "TIMESTAMPTZ",
    "tzns": "TIMESTAMPTZ_NS",
    "iv": "INTERVAL",
    "uu": "UUID",
    "e": "ENUM('p', 'q')",
    "l": "INTEGER[]",
    "bits": "BIT",
    "tiny": "DECIMAL(38,38)",
    "st": "STRUCT(a INTEGER, b STRUCT(c VARCHAR))",
    "j": "JSON",
}


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    # A zoned constant renders in the session time zone, so the pinned texts need one zone on every runner.
    connection.execute("SET TimeZone = 'UTC'")
    yield connection
    connection.close()


def translate(con, predicate, columns=COLUMNS, schema=None, select="*"):
    if schema is not None:
        columns = None
    return _duckdb._translate_predicate(con, predicate, columns, schema, select)


def offered(con, predicate, columns=COLUMNS, schema=None, select="*"):
    """(engine text, fragment or REFUSED) for each predicate the optimizer offers the scan."""
    entries = translate(con, predicate, columns, schema, select)
    return [(text, None if node is None else node.fragment()) for text, node in entries]


def only_node(con, predicate, columns=COLUMNS, schema=None, select="*"):
    [(_, node)] = translate(con, predicate, columns, schema, select)
    return node


def p(predicate, *entries, id):
    return pytest.param(predicate, list(entries), id=id)


SHAPES = [
    # comparisons, the column put on the left
    p("n > 7", ("(n > 7)", '("n" > 7)'), id="greater"),
    p("7 < n", ("(n > 7)", '("n" > 7)'), id="constant-on-the-left"),
    p("n = 4", ("(n = 4)", '("n" = 4)'), id="equal"),
    p("n <= 4", ("(n <= 4)", '("n" <= 4)'), id="at-most"),
    p("n >= 4", ("(n >= 4)", '("n" >= 4)'), id="at-least"),
    p("n < 2", ("(n < 2)", '("n" < 2)'), id="less"),
    p("s <> '9'", ("(s != '9')", "(\"s\" != '9')"), id="not-equal"),
    p("NOT (n = 3)", ("(n != 3)", '("n" != 3)'), id="not-over-a-comparison-is-rewritten"),
    p("n + 1 > 3", ("(n > 2)", '("n" > 2)'), id="arithmetic-moved-to-the-constant"),
    p("n::BIGINT > 3", ("(n > 3)", '("n" > 3)'), id="widening-cast-removed"),
    p("n > 1 + 1", ("(n > 2)", '("n" > 2)'), id="folded-constant"),
    p("b = true", ("(b = true)", '("b" = TRUE)'), id="boolean"),
    p("s LIKE 'abc'", ("(s = 'abc')", "(\"s\" = 'abc')"), id="like-without-wildcards"),
    # BETWEEN, read through the bound inclusivity flags
    p("n BETWEEN 3 AND 6", ("(n BETWEEN 3 AND 6)", '(("n" >= 3) AND ("n" <= 6))'), id="between"),
    p("n > 1 AND n < 8", ("(n BETWEEN 1 AND 8)", '(("n" > 1) AND ("n" < 8))'), id="exclusive-range-folded"),
    p("n >= 1 AND n < 8", ("(n BETWEEN 1 AND 8)", '(("n" >= 1) AND ("n" < 8))'), id="half-open-range-folded"),
    p("n BETWEEN 1 AND 1", ("(n = 1)", '("n" = 1)'), id="between-one-value"),
    p("n NOT BETWEEN 3 AND 6", ("((n < 3) OR (n > 6))", '(("n" < 3) OR ("n" > 6))'), id="not-between"),
    p(
        "NOT (n BETWEEN 1 AND 3)",
        ("((n < 1) OR (n > 3))", '(("n" < 1) OR ("n" > 3))'),
        id="not-over-between",
    ),
    # exact optimizer rewrites of a function over a TIMESTAMP or DATE column arrive as a range
    p(
        "CAST(ts AS DATE) = DATE '2024-01-01'",
        (
            "(ts BETWEEN '2024-01-01 00:00:00'::TIMESTAMP AND '2024-01-02 00:00:00'::TIMESTAMP)",
            "((\"ts\" >= TIMESTAMP '2024-01-01 00:00:00') AND (\"ts\" < TIMESTAMP '2024-01-02 00:00:00'))",
        ),
        id="date-cast-equality-rewritten",
    ),
    p(
        "date_trunc('day', ts) = TIMESTAMP '2024-01-01'",
        (
            "(ts BETWEEN '2024-01-01 00:00:00'::TIMESTAMP AND '2024-01-02 00:00:00'::TIMESTAMP)",
            "((\"ts\" >= TIMESTAMP '2024-01-01 00:00:00') AND (\"ts\" < TIMESTAMP '2024-01-02 00:00:00'))",
        ),
        id="date-trunc-equality-rewritten",
    ),
    p(
        "year(d) = 2024",
        (
            "(d BETWEEN '2024-01-01'::DATE AND '2024-12-31'::DATE)",
            "((\"d\" >= DATE '2024-01-01') AND (\"d\" <= DATE '2024-12-31'))",
        ),
        id="year-equality-rewritten",
    ),
    # IN and NOT IN over constants
    p("n IN (3, 4)", ("(n IN (3, 4))", '("n" IN (3, 4))'), id="in"),
    p("n IN (2)", ("(n IN (2))", '("n" IN (2))'), id="in-one-value"),
    p("n IN (1, 1)", ("(n IN (1, 1))", '("n" IN (1, 1))'), id="in-repeated-value"),
    p("s IN ('a', 'b')", ("(s IN ('a', 'b'))", "(\"s\" IN ('a', 'b'))"), id="in-strings"),
    p("n NOT IN (0, 1)", ("(NOT (n IN (0, 1)))", '(NOT ("n" IN (0, 1)))'), id="not-in"),
    p("n NOT IN (2)", ("(NOT (n IN (2)))", '(NOT ("n" IN (2)))'), id="not-in-one-value"),
    # IS NULL, IS NOT NULL, NOT
    p("n IS NULL", ("(n IS NULL)", '("n" IS NULL)'), id="is-null"),
    p("n IS NOT NULL", ("(n IS NOT NULL)", '("n" IS NOT NULL)'), id="is-not-null"),
    p("NOT (n IS NULL)", ("(NOT (n IS NULL))", '(NOT ("n" IS NULL))'), id="not-over-is-null"),
    p("NOT NOT (n = 1)", ("(n = 1)", '("n" = 1)'), id="double-not-removed"),
    # OR, and AND below the top level; a top level AND reaches the scan as separate predicates
    p("n = 1 OR s = 'x'", ("((n = 1) OR (s = 'x'))", '(("n" = 1) OR ("s" = \'x\'))'), id="or-across-columns"),
    p("n > 5 OR n IS NULL", ("((n > 5) OR (n IS NULL))", '(("n" > 5) OR ("n" IS NULL))'), id="or-with-is-null"),
    p(
        "NOT (n > 1 AND s = 'a')",
        ("((n <= 1) OR (s != 'a'))", '(("n" <= 1) OR ("s" != \'a\'))'),
        id="not-over-and-becomes-or",
    ),
    p("n = 1 AND s = 'a'", ("(n = 1)", '("n" = 1)'), ("(s = 'a')", "(\"s\" = 'a')"), id="top-level-and-split"),
    p("NOT (n > 3 OR s = 'a')", ("(n <= 3)", '("n" <= 3)'), ("(s != 'a')", "(\"s\" != 'a')"), id="not-over-or"),
    p(
        "(n = 1 AND x = 2) OR (n = 3 AND x = 4)",
        (
            "(((n = 1) AND (x = 2)) OR ((n = 3) AND (x = 4)))",
            '((("n" = 1) AND ("x" = 2)) OR (("n" = 3) AND ("x" = 4)))',
        ),
        ("((n = 1) OR (n = 3))", '(("n" = 1) OR ("n" = 3))'),
        ("((x = 2) OR (x = 4))", '(("x" = 2) OR ("x" = 4))'),
        id="and-inside-or-with-derived-predicates",
    ),
    # a predicate naming one column twice
    p("n = 1 OR n = 2", ("((n = 1) OR (n = 2))", '(("n" = 1) OR ("n" = 2))'), id="one-column-twice-in-or"),
    p("n > 1 AND n > 1", ("(n > 1)", '("n" > 1)'), id="one-predicate-twice"),
    p(
        "n IN (1, 2) AND n IN (2, 3)",
        ("(n IN (1, 2))", '("n" IN (1, 2))'),
        ("(n IN (2, 3))", '("n" IN (2, 3))'),
        id="two-in",
    ),
    # struct field paths
    p("st.a = 1", ("(struct_extract(st, 'a') = 1)", '("st"."a" = 1)'), id="struct-field"),
    p(
        "st.b.c = 'x'",
        ("(struct_extract(struct_extract(st, 'b'), 'c') = 'x')", '("st"."b"."c" = \'x\')'),
        id="nested-struct-field",
    ),
    p("struct_extract(st, 'a') > 1", ("(struct_extract(st, 'a') > 1)", '("st"."a" > 1)'), id="struct-extract-call"),
    p(
        "struct_extract_at(st, 1) = 1",
        ("(struct_extract_at(st, 1) = 1)", '("st"."a" = 1)'),
        id="struct-field-by-position",
    ),
    p("st.a IN (1, 2)", ("(struct_extract(st, 'a') IN (1, 2))", '("st"."a" IN (1, 2))'), id="struct-field-in"),
    p(
        "st.a BETWEEN 1 AND 3",
        ("(struct_extract(st, 'a') BETWEEN 1 AND 3)", '(("st"."a" >= 1) AND ("st"."a" <= 3))'),
        id="struct-field-between",
    ),
    p("st.a IS NULL", ("(struct_extract(st, 'a') IS NULL)", '("st"."a" IS NULL)'), id="struct-field-is-null"),
    p("st.b IS NOT NULL", ("(struct_extract(st, 'b') IS NOT NULL)", '("st"."b" IS NOT NULL)'), id="struct-is-not-null"),
    p("st IS NULL", ("(st IS NULL)", '("st" IS NULL)'), id="whole-struct-is-null"),
    p("j IS NULL", ("(j IS NULL)", '("j" IS NULL)'), id="is-null-on-an-aliased-type"),
    # a predicate the optimizer folds away offers nothing
    p("n > 2 AND n < 1", id="unsatisfiable"),
    p("1 = 1", id="always-true"),
    p("n > 1 OR TRUE", id="or-true"),
    p("n = NULL", id="comparison-with-null"),
]

REFUSALS = [
    # a parameter at prepare time: the scan's rebind on execute offers the value as a constant instead
    p("n > $1", ("(n > $1)", REFUSED), id="prepared-parameter"),
    # LIKE, prefix and pattern functions
    p("s LIKE 'ab%'", ("prefix(s, 'ab')", REFUSED), id="like-prefix"),
    p("prefix(s, 'ab')", ("prefix(s, 'ab')", REFUSED), id="prefix"),
    p("regexp_matches(s, '^ab')", ("prefix(s, 'ab')", REFUSED), id="anchored-regex"),
    p("s LIKE '%x%'", ("contains(s, 'x')", REFUSED), id="like-contains"),
    p("s LIKE 'ab%c'", ("(s ~~ 'ab%c')", REFUSED), id="like-pattern"),
    p("s ILIKE 'ab%'", ("(s ~~* 'ab%')", REFUSED), id="ilike"),
    p("s NOT LIKE 'a%'", ("(NOT prefix(s, 'a'))", REFUSED), id="not-like"),
    # casts around the column, and functions
    p("n > 3.5", ("(CAST(n AS DECIMAL(11,1)) > 3.5)", REFUSED), id="cast-to-decimal"),
    p("ti = 300", ("(CAST(ti AS INTEGER) = 300)", REFUSED), id="cast-to-a-wider-integer"),
    p("dec = 1.505", ("(CAST(dec AS DECIMAL(11,3)) = 1.505)", REFUSED), id="cast-to-a-wider-decimal"),
    p(
        "tms > TIMESTAMP '2024-05-06'",
        ("(CAST(tms AS TIMESTAMP) > '2024-05-06 00:00:00'::TIMESTAMP)", REFUSED),
        id="cast-ms",
    ),
    p(
        "CAST(ts AS DATE) > DATE '2024-01-01'",
        ("(CAST(ts AS DATE) > '2024-01-01'::DATE)", REFUSED),
        id="date-cast-not-rewritten",
    ),
    p(
        "CAST(tms AS DATE) = DATE '2024-01-01'",
        ("(CAST(tms AS DATE) = '2024-01-01'::DATE)", REFUSED),
        id="date-cast-of-ms-not-rewritten",
    ),
    p(
        "CAST(tz AS DATE) = DATE '2024-01-01'",
        ("(CAST(tz AS DATE) = '2024-01-01'::DATE)", REFUSED),
        id="date-cast-of-zoned-not-rewritten",
    ),
    p("lower(s) = 'a'", ("(lower(s) = 'a')", REFUSED), id="function-of-the-column"),
    p("n = random()", ("(CAST(n AS DOUBLE) = random())", REFUSED), id="volatile-function"),
    p("st['a'] = 1", ("(array_extract(st, 'a') = 1)", REFUSED), id="struct-by-subscript"),
    # other operators and operands
    p("n IS DISTINCT FROM 3", ("(n IS DISTINCT FROM 3)", REFUSED), id="distinct-from"),
    p("n IS NOT DISTINCT FROM 3", ("(n IS NOT DISTINCT FROM 3)", REFUSED), id="not-distinct-from"),
    p("n = x", ("(n = x)", REFUSED), id="column-against-column"),
    p("n IN (1, x)", ("(n IN (1, x))", REFUSED), id="in-with-a-column"),
    p("n IN (1, 2.5)", ("(CAST(n AS DECIMAL(11,1)) IN (1.0, 2.5))", REFUSED), id="in-casting-the-column"),
    p("b", ("b", REFUSED), id="bare-boolean-column"),
    p("NOT b", ("(NOT b)", REFUSED), id="not-bare-boolean-column"),
    p("rowid = 1", ("(rowid = 1)", REFUSED), id="row-id"),
    # a refused child refuses its whole OR, AND or NOT; a top level AND keeps its other predicates
    p(
        "n > 1 AND (n < 3 OR s LIKE '%x%')",
        ("((n < 3) OR contains(s, 'x'))", REFUSED),
        ("(n > 1)", '("n" > 1)'),
        id="or-with-a-refused-child",
    ),
    p(
        "(n > 1 AND s LIKE '%x%') OR n = 5",
        ("(((n > 1) AND contains(s, 'x')) OR (n = 5))", REFUSED),
        ("((n > 1) OR (n = 5))", '(("n" > 1) OR ("n" = 5))'),
        id="and-with-a-refused-child-under-or",
    ),
    p("n = 3 AND rowid < 5", ("(n = 3)", '("n" = 3)'), ("(rowid < 5)", REFUSED), id="top-level-and-keeps-the-rest"),
    p(
        "dec BETWEEN 1 AND 2.555",
        ("(dec >= 1.00)", "(\"dec\" >= CAST('1.00' AS DECIMAL(10, 2)))"),
        ("(CAST(dec AS DECIMAL(13,3)) <= 2.555)", REFUSED),
        id="between-split-by-a-cast-bound",
    ),
    # constants without an exact Python value of the column's type
    p("n IN (1, NULL)", ("(n IN (1, NULL::INTEGER))", REFUSED), id="null-in-in"),
    p("n NOT IN (1, NULL)", ("(NOT (n IN (1, NULL::INTEGER)))", REFUSED), id="null-in-not-in"),
    p("f > 'nan'::DOUBLE", ("(f > 'nan'::DOUBLE)", REFUSED), id="nan"),
    p("f IN (0.0, 'nan')", ("(f IN (0.0, 'nan'::DOUBLE))", REFUSED), id="nan-in-in"),
    p("f = 'inf'::DOUBLE", ("(f = 1e1000)", REFUSED), id="infinity"),
    p("f < '-inf'::DOUBLE", ("(f < -1e1000)", REFUSED), id="negative-infinity"),
    p("d >= DATE 'infinity'", ("(d >= 'infinity'::DATE)", REFUSED), id="infinite-date"),
    p("d > DATE '-infinity'", ("(d > '-infinity'::DATE)", REFUSED), id="negative-infinite-date"),
    p(
        "d BETWEEN DATE '2024-01-01' AND DATE 'infinity'",
        ("(d BETWEEN '2024-01-01'::DATE AND 'infinity'::DATE)", REFUSED),
        id="infinite-between-bound",
    ),
    p("d > DATE '0000-12-31'", ("(d > '0001-12-31 (BC)'::DATE)", REFUSED), id="date-before-year-one"),
    p("d < DATE '10000-01-01'", ("(d < '10000-01-01'::DATE)", REFUSED), id="date-after-year-9999"),
    p("ts > TIMESTAMP 'infinity'", ("(ts > 'infinity'::TIMESTAMP)", REFUSED), id="infinite-timestamp"),
    p("ts <= TIMESTAMP '-infinity'", ("(ts <= '-infinity'::TIMESTAMP)", REFUSED), id="negative-infinite-timestamp"),
    p("ts < TIMESTAMP '10000-01-01'", ("(ts < '10000-01-01 00:00:00'::TIMESTAMP)", REFUSED), id="timestamp-year-10000"),
    p(
        "tz > TIMESTAMPTZ 'infinity'",
        ("(tz > 'infinity'::TIMESTAMP WITH TIME ZONE)", REFUSED),
        id="infinite-zoned-timestamp",
    ),
    p("t = TIME '24:00:00'", ("(t = '24:00:00'::TIME)", REFUSED), id="time-24"),
    p("tn < '04:00:00'::TIME_NS", ("(tn < '04:00:00'::TIME_NS)", REFUSED), id="nanosecond-time"),
    p("tns > TIMESTAMP '2024-05-06'", ("(tns > '2024-05-06 00:00:00'::TIMESTAMP_NS)", REFUSED), id="nanosecond-ts"),
    p(
        "tzns > '2024-05-06'::TIMESTAMPTZ_NS",
        ("(tzns > '2024-05-06 00:00:00+00'::TIMESTAMPTZ_NS)", REFUSED),
        id="nanosecond-zoned-timestamp",
    ),
    p("ttz = '04:00:00+02'::TIMETZ", ("(timetz_byte_comparable(ttz) = 1087146819584050399)", REFUSED), id="time-tz"),
    p("iv = INTERVAL 1 DAY", ("(normalized_interval(iv) = '1 day'::INTERVAL)", REFUSED), id="interval"),
    p(
        "uu = '00000000-0000-0000-0000-000000000000'::UUID",
        ("(uu = '00000000-0000-0000-0000-000000000000'::UUID)", REFUSED),
        id="uuid",
    ),
    p("e = 'p'", ("(e = 'p')", REFUSED), id="enum"),
    p("l = [1]", ("(l = [1])", REFUSED), id="list"),
    p("j = '{\"a\": 1}'", ("(j = '{\"a\": 1}')", REFUSED), id="json-is-an-aliased-type"),
    p("bits = '101'::BIT", ("(bitstring_byte_comparable(bits) = '\\x03\\x02\\x03'::BLOB)", REFUSED), id="bit"),
    p("st.b = {'c': 'x'}", ("(struct_extract(st, 'b') = {'c': 'x'})", REFUSED), id="struct-constant"),
]


@pytest.mark.parametrize(("predicate", "expected"), SHAPES + REFUSALS)
def test_offered_predicates(con, predicate, expected):
    assert offered(con, predicate) == expected


def literal(node):
    match node:
        case Binary(right=Lit() as constant):
            return constant
        case In(values=[Lit() as constant]):
            return constant
    raise AssertionError(node)


GATE = [
    pytest.param("b = false", False, "BOOLEAN", id="boolean"),
    pytest.param("ti = 3", 3, "TINYINT", id="tinyint"),
    pytest.param("n = -2147483648", -(2**31), "INTEGER", id="integer-minimum"),
    pytest.param("u = 18446744073709551615", 2**64 - 1, "UBIGINT", id="ubigint-maximum"),
    pytest.param("h = 170141183460469231731687303715884105727", 2**127 - 1, "HUGEINT", id="hugeint-maximum"),
    pytest.param("f = 1.5", 1.5, "DOUBLE", id="double"),
    pytest.param("r > 0.1", 0.10000000149011612, "FLOAT", id="real-holds-the-float-value"),
    pytest.param("f > 0.1::REAL", 0.10000000149011612, "DOUBLE", id="real-constant-widened-exactly"),
    pytest.param("dec = 1.50", decimal.Decimal("1.50"), "DECIMAL(10,2)", id="decimal"),
    pytest.param("dec > 1.5", decimal.Decimal("1.50"), "DECIMAL(10,2)", id="decimal-at-the-column-scale"),
    pytest.param("dec = 1", decimal.Decimal("1.00"), "DECIMAL(10,2)", id="integer-against-decimal"),
    pytest.param("wide = 12.5", decimal.Decimal("12.5000"), "DECIMAL(38,4)", id="wide-decimal"),
    pytest.param("s = 'é'", "é", "VARCHAR", id="text"),
    pytest.param("s = ''", "", "VARCHAR", id="empty-text"),
    pytest.param("bl = '\\x61'::BLOB", b"a", "BLOB", id="blob"),
    pytest.param("d = DATE '0001-01-01'", datetime.date(1, 1, 1), "DATE", id="date-year-one"),
    pytest.param("d = DATE '9999-12-31'", datetime.date(9999, 12, 31), "DATE", id="date-year-9999"),
    pytest.param("t < TIME '04:00:00'", datetime.time(4, 0), "TIME", id="time"),
    pytest.param("t > TIME '23:59:59.999999'", datetime.time(23, 59, 59, 999999), "TIME", id="time-last-microsecond"),
    pytest.param(
        "ts = TIMESTAMP '2024-05-06 07:08:09.123456'",
        datetime.datetime(2024, 5, 6, 7, 8, 9, 123456),
        "TIMESTAMP",
        id="timestamp",
    ),
    pytest.param(
        "tms >= '2024-05-06 07:08:09.123'::TIMESTAMP_MS",
        datetime.datetime(2024, 5, 6, 7, 8, 9, 123000),
        "TIMESTAMP_MS",
        id="timestamp-ms",
    ),
    pytest.param(
        "tsec = '2024-05-06 07:08:09'::TIMESTAMP_S",
        datetime.datetime(2024, 5, 6, 7, 8, 9),
        "TIMESTAMP_S",
        id="timestamp-s",
    ),
    pytest.param(
        "tz > TIMESTAMPTZ '2024-05-06 00:00:00+00'",
        1714953600000000,
        "TIMESTAMP WITH TIME ZONE",
        id="zoned-timestamp-as-utc-microseconds",
    ),
    pytest.param(
        "tz = TIMESTAMPTZ '2024-05-06 12:00:00.5+02'",
        1714989600500000,
        "TIMESTAMP WITH TIME ZONE",
        id="zoned-timestamp-off-the-second-grid",
    ),
    pytest.param(
        "tiny = '0.00000000000000000000000000000000000001'",
        decimal.Decimal("1E-38"),
        "DECIMAL(38,38)",
        id="decimal-at-the-widest-scale",
    ),
]


@pytest.mark.parametrize(("predicate", "value", "type_text"), GATE)
def test_a_constant_carries_its_exact_python_value_and_the_column_type(con, predicate, value, type_text):
    constant = literal(only_node(con, predicate))
    assert type(constant.value) is type(value)
    assert constant.value == value
    assert str(constant.value) == str(value)
    assert constant.type == type_text


@pytest.mark.parametrize(("predicate", "value", "type_text"), GATE)
def test_a_constant_renders_as_sql_that_reads_back_as_the_same_value(con, predicate, value, type_text):
    constant = literal(only_node(con, predicate))
    if type_text == "TIMESTAMP WITH TIME ZONE":
        [(read,)] = con.sql(f"SELECT epoch_us(CAST({constant.fragment()} AS TIMESTAMPTZ))").fetchall()
    else:
        [(read,)] = con.sql(f"SELECT CAST({constant.fragment()} AS {type_text})").fetchall()
    assert read == value


@pytest.mark.parametrize(
    ("predicate", "sign"),
    [
        ("f IN (-0.0, 1.0)", 1.0),
        ("f = -0.0", 1.0),
        ("f IN ('-0.0'::DOUBLE, 1.0)", -1.0),
        ("f IN (-0e0, 1e0)", -1.0),
        ("f = '-0.0'::DOUBLE", -1.0),
    ],
    ids=[
        "negated-decimal-literal-in",
        "negated-decimal-literal",
        "cast-text-in",
        "negated-float-literal-in",
        "cast-text",
    ],
)
def test_a_zero_constant_keeps_the_sign_the_engine_bound(con, predicate, sign):
    node = only_node(con, predicate)
    constant = node.values[0] if isinstance(node, In) else literal(node)
    assert constant.value == 0.0
    assert math.copysign(1.0, constant.value) == sign
    assert con.sql("SELECT '-0.0'::DOUBLE IN (0.0, 1.0), 0.0::DOUBLE IN ('-0.0'::DOUBLE, 1.0)").fetchall() == [
        (True, True)
    ]


@pytest.mark.parametrize(
    ("zone", "micros"), [("UTC", 1714953600000000), ("America/New_York", 1714968000000000)], ids=["utc", "new-york"]
)
def test_the_session_zone_decides_a_naive_constant_against_a_zoned_column(con, zone, micros):
    con.execute(f"SET TimeZone = '{zone}'")
    constant = literal(only_node(con, "tz > TIMESTAMP '2024-05-06'"))
    assert constant.value == micros
    assert constant.type == "TIMESTAMP WITH TIME ZONE"


def test_nodes_put_the_column_on_the_left_and_share_it_across_a_range(con):
    node = only_node(con, "8 > n")
    assert isinstance(node, Binary)
    assert (node.op, node.left.parts, node.right.value) == ("<", ("n",), 8)

    node = only_node(con, "n > 1 AND n <= 8")
    match node:
        case Binary(
            op="AND",
            left=Binary(op=">", left=Col(parts=("n",)) as low_column, right=Lit(value=1, type="INTEGER")),
            right=Binary(op="<=", left=Col(parts=("n",)) as high_column, right=Lit(value=8, type="INTEGER")),
        ):
            assert low_column is high_column
        case _:
            raise AssertionError(node.fragment())


def test_in_holds_a_list_of_literals_and_not_in_wraps_it(con):
    node = only_node(con, "n NOT IN (3, 4)")
    assert isinstance(node, Unary)
    assert node.op == "NOT"
    membership = node.operand
    assert isinstance(membership, In)
    assert isinstance(membership.values, list)
    assert [(value.value, value.type) for value in membership.values] == [(3, "INTEGER"), (4, "INTEGER")]


def test_a_struct_path_is_one_column_with_the_field_names(con):
    node = only_node(con, "st.b.c IS NOT NULL")
    assert isinstance(node, Postfix)
    assert node.op == "IS NOT NULL"
    assert node.operand.parts == ("st", "b", "c")
    constant = literal(only_node(con, "st.b.c = 'x'"))
    assert (constant.value, constant.type) == ("x", "VARCHAR")


def test_names_keep_their_case_and_quotes(con):
    columns = {"Mixed Case": "INTEGER", 'quo"te': 'STRUCT("In""ner" INTEGER)'}
    assert offered(con, '"Mixed Case" > 1', columns) == [("(Mixed Case > 1)", '("Mixed Case" > 1)')]
    node = only_node(con, '"quo""te"."In""ner" = 1', columns)
    assert node.left.parts == ('quo"te', 'In"ner')
    assert node.fragment() == '("quo""te"."In""ner" = 1)'


class TestPortedFromNeoFilters:
    def test_a_conjunction_reaches_the_source_as_separate_predicates(self, con):
        assert [fragment for _, fragment in offered(con, "n > 7 AND s <> '9'")] == ['("n" > 7)', "(\"s\" != '9')"]

    def test_a_predicate_names_columns_the_query_does_not_read(self, con):
        columns = {"n": "INTEGER", "s": "VARCHAR", "f": "DOUBLE"}
        assert offered(con, "n = 4", columns, select="s") == [("(n = 4)", '("n" = 4)')]
        assert offered(con, "f > 4.5", columns, select="count(*)") == [("(f > 4.5)", '("f" > 4.5::DOUBLE)')]
        assert offered(con, "f > 4.5 OR n = 2", columns, select="s") == [
            ("((f > 4.5) OR (n = 2))", '(("f" > 4.5::DOUBLE) OR ("n" = 2))')
        ]

    def test_in_from_the_dbapi_query(self, con):
        assert offered(con, "n IN (3, 4)") == [("(n IN (3, 4))", '("n" IN (3, 4))')]

    def test_a_constant_is_offered_when_a_python_value_holds_it_exactly(self, con):
        columns = {
            "money": "DECIMAL(10,2)",
            "raw": "BLOB",
            "clock": "TIME",
            "coarse": "TIMESTAMP_MS",
            "fine": "TIMESTAMP_NS",
            "zoned": "TIMESTAMPTZ",
            "tags": "VARCHAR[]",
        }
        entries = _duckdb._translate_predicate(
            con,
            "money > 1.50 AND raw = '\\x61'::BLOB AND clock < TIME '04:00:00' "
            "AND coarse >= '2024-05-06 07:08:09.123'::TIMESTAMP_MS AND zoned > TIMESTAMPTZ '2024-05-06 00:00:00+00' "
            "AND fine > TIMESTAMP '2024-05-06' AND tags = ['a']",
            columns,
        )
        carried = {node.left.parts[0]: node.right.value for _, node in entries if node is not None}
        assert carried == {
            "money": decimal.Decimal("1.50"),
            "raw": b"a",
            "clock": datetime.time(4, 0),
            "coarse": datetime.datetime(2024, 5, 6, 7, 8, 9, 123000),
            "zoned": 1714953600000000,
        }
        assert sum(node is None for _, node in entries) == 2

    def test_an_infinite_date_is_never_offered(self, con):
        columns = {"d": "DATE", "ts": "TIMESTAMP"}
        assert offered(con, "d >= DATE 'infinity' OR ts <= TIMESTAMP '-infinity'", columns) == [
            ("((d >= 'infinity'::DATE) OR (ts <= '-infinity'::TIMESTAMP))", REFUSED)
        ]
        assert [
            fragment for _, fragment in offered(con, "d < DATE 'infinity' AND ts > TIMESTAMP '-infinity'", columns)
        ] == [REFUSED, REFUSED]
        assert offered(con, "d >= DATE '9999-12-31'", columns) == [
            ("(d >= '9999-12-31'::DATE)", "(\"d\" >= DATE '9999-12-31')")
        ]


class TestSourceNames:
    """Columns declared as the engine's Arrow scan declares them, named in predicates by their Arrow names."""

    def test_the_column_of_a_nameless_field_keeps_its_empty_name(self, con):
        pa = pytest.importorskip("pyarrow")
        schema = pa.schema([pa.field("", pa.int64())])
        node = only_node(con, "v0 > 1", schema=schema)
        assert node.left.parts == ("",)
        assert node.fragment() == '("" > 1)'

    def test_a_repeated_name_is_refused_for_both_columns(self, con):
        pa = pytest.importorskip("pyarrow")
        schema = pa.schema([pa.field("a", pa.int64()), pa.field("a", pa.int64()), pa.field("b", pa.int64())])
        assert offered(con, "a > 1 AND a_1 > 1 AND b > 1", schema=schema) == [
            ("(a > 1)", REFUSED),
            ("(a_1 > 1)", REFUSED),
            ("(b > 1)", '("b" > 1)'),
        ]

    def test_names_differing_only_in_case_are_named_by_the_arrow_name(self, con):
        pa = pytest.importorskip("pyarrow")
        schema = pa.schema([pa.field("A", pa.int64()), pa.field("a", pa.int64())])
        assert offered(con, "A > 1 AND a_1 > 2", schema=schema) == [
            ("(A > 1)", '("A" > 1)'),
            ("(a_1 > 2)", '("a" > 2)'),
        ]

    def test_a_struct_with_an_empty_field_name_is_refused(self, con):
        pa = pytest.importorskip("pyarrow")
        schema = pa.schema([pa.field("s", pa.struct([pa.field("", pa.int64()), pa.field("x", pa.int64())]))])
        assert offered(con, "s.x = 1", schema=schema) == [("(struct_extract(s, 'x') = 1)", REFUSED)]
        assert offered(con, "s IS NULL", schema=schema) == [("(s IS NULL)", REFUSED)]

    def test_a_struct_with_a_repeated_field_name_is_refused(self, con):
        pa = pytest.importorskip("pyarrow")
        schema = pa.schema([pa.field("s", pa.struct([pa.field("p", pa.int64()), pa.field("p", pa.int64())]))])
        assert offered(con, "s IS NULL", schema=schema) == [("(s IS NULL)", REFUSED)]

    def test_a_nested_struct_with_an_empty_field_name_refuses_the_column(self, con):
        pa = pytest.importorskip("pyarrow")
        inner = pa.struct([pa.field("", pa.int64()), pa.field("y", pa.int64())])
        schema = pa.schema([pa.field("s", pa.struct([pa.field("q", inner), pa.field("r", pa.int64())]))])
        assert offered(con, "s.r = 1 AND s.q IS NULL", schema=schema) == [
            ("(struct_extract(s, 'q') IS NULL)", REFUSED),
            ("(struct_extract(s, 'r') = 1)", REFUSED),
        ]

    def test_a_struct_of_only_nameless_fields_is_a_tuple_no_path_enters(self, con):
        pa = pytest.importorskip("pyarrow")
        inner = pa.struct([pa.field("", pa.int64())])
        schema = pa.schema([pa.field("s", pa.struct([pa.field("q", inner), pa.field("r", pa.int64())]))])
        assert offered(con, "s.r = 1 AND s.q IS NULL", schema=schema) == [
            ("(struct_extract(s, 'q') IS NULL)", '("s"."q" IS NULL)'),
            ("(struct_extract(s, 'r') = 1)", '("s"."r" = 1)'),
        ]
        assert offered(con, "struct_extract(s.q, 1) = 1", schema=schema) == [
            ("(struct_extract(struct_extract(s, 'q'), 1) = 1)", REFUSED)
        ]
        assert offered(con, "struct_extract_at(s.q, 1) = 1", schema=schema) == [
            ("(struct_extract_at(struct_extract(s, 'q'), 1) = 1)", REFUSED)
        ]

    def test_a_struct_with_its_arrow_field_names_resolves(self, con):
        pa = pytest.importorskip("pyarrow")
        schema = pa.schema([pa.field("s", pa.struct([pa.field("p", pa.struct([pa.field("q", pa.int64())]))]))])
        node = only_node(con, "s.p.q = 1", schema=schema)
        assert node.left.parts == ("s", "p", "q")

    def test_a_dictionary_encoded_column_resolves_by_its_name(self, con):
        pa = pytest.importorskip("pyarrow")
        schema = pa.schema([pa.field("c", pa.dictionary(pa.int32(), pa.string()))])
        assert offered(con, "c = 'x'", schema=schema) == [("(c = 'x')", "(\"c\" = 'x')")]

    def test_the_declared_types_are_the_arrow_scan_types(self, con):
        pa = pytest.importorskip("pyarrow")
        schema = pa.schema([pa.field("t", pa.timestamp("s", tz="America/New_York")), pa.field("m", pa.timestamp("ms"))])
        assert [
            (node.right.value, node.right.type)
            for _, node in translate(
                con,
                "t > TIMESTAMPTZ '2024-05-06 00:00:00+00' AND m = '2024-05-06 07:08:09.123'::TIMESTAMP_MS",
                schema=schema,
            )
        ] == [
            (1714953600000000, "TIMESTAMP WITH TIME ZONE"),
            (datetime.datetime(2024, 5, 6, 7, 8, 9, 123000), "TIMESTAMP_MS"),
        ]


class TestUserFunctions:
    """A user function that carries the name of a struct field accessor is not a field access.

    A native function that reuses the accessor's binder with another execution callback is refused too, but only C++
    can register one, so that case is checked by a compiled probe outside this suite.
    """

    DATA = "CREATE TABLE data AS SELECT * FROM (VALUES (1, {'a': 1, 'b': 7}), (2, {'a': 7, 'b': 1})) t(id, st)"
    COLUMNS: ClassVar[dict[str, str]] = {"st": "STRUCT(a INTEGER, b INTEGER)"}

    def rows(self, con, predicate):
        return con.sql(f"SELECT id FROM data WHERE {predicate} ORDER BY id").fetchall()

    def test_the_built_in_accessor_by_position_is_a_field_access(self, con):
        con.execute(self.DATA)
        node = only_node(con, "struct_extract_at(st, 1) = 7", self.COLUMNS)
        assert node.fragment() == '("st"."a" = 7)'
        assert self.rows(con, node.fragment()) == self.rows(con, "struct_extract_at(st, 1) = 7") == [(2,)]

    @pytest.mark.parametrize(
        "predicate", ["system.main.struct_extract_at(st, 1) = 7", "main.struct_extract(st, 'a') = 7"]
    )
    def test_the_built_in_accessor_named_with_its_schema_is_a_field_access(self, con, predicate):
        con.execute(self.DATA)
        node = only_node(con, predicate, self.COLUMNS)
        assert node.fragment() == '("st"."a" = 7)'
        assert self.rows(con, node.fragment()) == self.rows(con, predicate) == [(2,)]

    def test_a_user_function_named_like_the_accessor_by_position_is_refused(self, con):
        con.execute(self.DATA)
        con.create_function(
            "struct_extract_at", lambda st, i: st["b"], ["STRUCT(a INTEGER, b INTEGER)", "INTEGER"], "INTEGER"
        )
        assert self.rows(con, "struct_extract_at(st, 1) = 7") == [(1,)]
        assert self.rows(con, '("st"."a" = 7)') == [(2,)]
        assert offered(con, "struct_extract_at(st, 1) = 7", self.COLUMNS) == [
            ("(struct_extract_at(st, 1) = 7)", REFUSED)
        ]

    def test_a_user_function_named_like_the_accessor_by_name_is_refused(self, con):
        con.execute(self.DATA)
        con.create_function(
            "struct_extract",
            lambda st, key, unused: st["b"],
            ["STRUCT(a INTEGER, b INTEGER)", "VARCHAR", "INTEGER"],
            "INTEGER",
        )
        assert self.rows(con, "struct_extract(st, 'a', 0) = 7") == [(1,)]
        assert offered(con, "struct_extract(st, 'a', 0) = 7", self.COLUMNS) == [
            ("(struct_extract(st, 'a', 0) = 7)", REFUSED)
        ]
        assert offered(con, "st.a = 7", self.COLUMNS) == [("(struct_extract(st, 'a') = 7)", '("st"."a" = 7)')]


CALLBACK_REFUSAL = (
    "code running inside a query cannot use that query's connection; use a separate connection from cursor()"
)
CHILD_TIMEOUT = 60

REENTRANT_SCHEMA = """
import json
import _duckdb
import duckdb
import pyarrow as pa

con = duckdb.connect()


class Reentrant:
    def __arrow_c_schema__(self):
        con.sql("SELECT 42").fetchall()
        return pa.schema([pa.field("n", pa.int64())]).__arrow_c_schema__()


try:
    _duckdb._translate_predicate(con, "n > 1", None, Reentrant())
    outcome = "no error"
except Exception as error:
    outcome = str(error)
other = duckdb.connect()


class UsesAnotherConnection:
    def __arrow_c_schema__(self):
        other.sql("SELECT 42").fetchall()
        return pa.schema([pa.field("n", pa.int64())]).__arrow_c_schema__()


entries = _duckdb._translate_predicate(con, "n > 1", None, UsesAnotherConnection())
print("outcome " + json.dumps([outcome, [node.fragment() for _, node in entries]]))
"""


class TestHook:
    def test_the_temporary_table_is_dropped(self, con):
        offered(con, "n > 1")
        with pytest.raises(duckdb.ParserException):
            offered(con, "n >")
        assert con.sql(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name LIKE '__duckdb_predicate_probe%'"
        ).fetchall() == [(0,)]

    def test_a_table_of_the_user_survives(self, con):
        con.execute("CREATE TEMPORARY TABLE __duckdb_predicate_probe AS SELECT 42 AS kept")
        con.execute("CREATE TABLE main.__duckdb_predicate_probe AS SELECT 43 AS kept")
        offered(con, "n > 1")
        assert con.sql("SELECT kept FROM temp.main.__duckdb_predicate_probe").fetchall() == [(42,)]
        assert con.sql("SELECT kept FROM memory.main.__duckdb_predicate_probe").fetchall() == [(43,)]

    def test_a_user_transaction_stays_open(self, con):
        con.execute("BEGIN")
        con.execute("CREATE TABLE kept AS SELECT 1 AS k")
        offered(con, "n > 1")
        con.execute("COMMIT")
        assert con.sql("SELECT k FROM kept").fetchall() == [(1,)]

    def test_more_than_one_statement_is_refused(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="one expression"):
            offered(con, "n > 1; SELECT 1")

    def test_a_subquery_is_refused(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="shape of the plan"):
            offered(con, "n IN (SELECT 1)")

    def test_either_columns_or_a_schema(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="either the columns or an Arrow schema"):
            _duckdb._translate_predicate(con, "n > 1", None, None)
        with pytest.raises(duckdb.InvalidInputException, match="either the columns or an Arrow schema"):
            _duckdb._translate_predicate(con, "n > 1", {"n": "INTEGER"}, object())

    def test_a_type_whose_text_uses_the_connection_is_refused(self, con):
        class Reentrant:
            def __str__(self) -> str:
                con.sql("SELECT 42").fetchall()
                return "INTEGER"

        with pytest.raises(duckdb.Error, match="cannot use that query's connection"):
            _duckdb._translate_predicate(con, "n > 1", {"n": Reentrant()})
        assert offered(con, "n > 1", {"n": "INTEGER"}) == [("(n > 1)", '("n" > 1)')]

    def test_a_schema_that_uses_the_connection_is_refused_without_hanging(self):
        pytest.importorskip("pyarrow")
        try:
            child = subprocess.run(
                [sys.executable, "-c", REENTRANT_SCHEMA],
                capture_output=True,
                text=True,
                timeout=CHILD_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"the hook did not return within {CHILD_TIMEOUT} seconds")
        assert child.returncode == 0, child.stdout + child.stderr
        [outcome] = [line.removeprefix("outcome ") for line in child.stdout.splitlines() if line.startswith("outcome ")]
        refusal, fragments = json.loads(outcome)
        assert CALLBACK_REFUSAL in refusal
        assert fragments == ['("n" > 1)']

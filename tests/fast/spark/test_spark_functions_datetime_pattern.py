import sys

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from datetime import date, datetime, timezone

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import (
    IllegalArgumentException,
    PySparkTypeError,
    SparkRuntimeException,
    SparkUpgradeException,
)
from spark_namespace.sql import functions as F

import duckdb
from duckdb.experimental.spark.exception import ContributionsAcceptedError

if USE_ACTUAL_SPARK:
    import pyspark
    from py4j.protocol import Py4JJavaError
    from pyspark.errors import DateTimeException

    SPARK_3 = pyspark.__version__.startswith("3.")
else:
    SPARK_3 = False

# Spark 4 runs with ANSI mode on, so to_timestamp and to_date raise on text the pattern does not match, as
# DuckDB's strptime does, with a ConversionException for a date that does not exist; Spark 3.5 returns NULL instead.
PARSE_ERROR = (DateTimeException,) if USE_ACTUAL_SPARK else (duckdb.InvalidInputException, duckdb.ConversionException)

PARSED = [
    ("yyyy-MM-dd", "2024-02-29", datetime(2024, 2, 29)),
    ("yyyy-MM-dd HH:mm:ss", "1997-02-28 10:30:00", datetime(1997, 2, 28, 10, 30)),
    ("dd/MM/yyyy", "05/01/2024", datetime(2024, 1, 5)),
    ("d/M/yyyy", "5/1/2024", datetime(2024, 1, 5)),
    pytest.param(
        "y-M-d",
        "24-1-5",
        datetime(24, 1, 5),
        marks=pytest.mark.skipif(
            sys.platform == "win32", reason="outside the range of the Windows C runtime, as for PySpark on Windows"
        ),
    ),
    pytest.param(
        "yyy",
        "024",
        datetime(24, 1, 1),
        marks=pytest.mark.skipif(
            sys.platform == "win32", reason="outside the range of the Windows C runtime, as for PySpark on Windows"
        ),
    ),
    ("yyyyMMddHHmmss", "20240105103000", datetime(2024, 1, 5, 10, 30)),
    ("dd MMM yyyy", "05 JAN 2024", datetime(2024, 1, 5)),
    ("dd LLLL yyyy", "05 january 2024", datetime(2024, 1, 5)),
    ("MMMMyyyy", "May2024", datetime(2024, 5, 1)),
    ("MMM d, yyyy", "Sep 5, 2024", datetime(2024, 9, 5)),
    ("yyyy-MM-dd hh:mm:ss a", "2024-01-05 12:30:00 AM", datetime(2024, 1, 5, 0, 30)),
    ("yyyy-MM-dd hh:mm:ss a", "2024-01-05 01:30:00 pm", datetime(2024, 1, 5, 13, 30)),
    ("yyyy-MM-dd a", "2024-01-05 PM", datetime(2024, 1, 5, 12)),
    ("yyyy-MM-dd HH:mm:ss.SSS", "2024-01-05 10:30:00.1", datetime(2024, 1, 5, 10, 30, 0, 100000)),
    ("yyyy-MM-dd HH:mm:ssSSS", "2024-01-05 10:30:00123", datetime(2024, 1, 5, 10, 30, 0, 123000)),
    ("yyyy-MM-dd HH:mm:ss.SSSSSS", "2024-01-05 10:30:00.123456", datetime(2024, 1, 5, 10, 30, 0, 123456)),
    ("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", "2024-01-05T10:30:00.123Z", datetime(2024, 1, 5, 10, 30, 0, 123000)),
    ("yyyy-MM-dd'''T'''HH", "2024-01-05'T'10", datetime(2024, 1, 5, 10)),
    ("yyyy%MM%dd", "2024%01%05", datetime(2024, 1, 5)),
    ("yyyy年MM月dd日", "2024年01月05日", datetime(2024, 1, 5)),
]

# Text that Spark and DuckDB's strptime both reject. The last element is Spark 3.5's answer: NULL, or an upgrade
# error where Spark 2's parser would have accepted the text.
REJECTED = [
    ("dd MMM yyyy", "05 January 2024", "upgrade"),
    ("yyyy-MM-dd", "2023-02-29", "null"),
    ("yyyy-MM-dd HH:mm:ss.SSS", "2024-01-05 10:30:00.1234", "null"),
    ("yyyy-MM-dd hh:mm:ss a", "2024-01-05 13:30:00 PM", "null"),
    ("yyyy-MM-dd'T'HH:mm:ss", "2024-01-05 10:30:00", "null"),
]

# Text that Spark rejects and DuckDB's strptime reads: it allows fewer digits than a field's width, a year of up to
# four digits for yyyy, up to three or six fraction digits for fewer S letters, any run of spaces, tabs included, for
# a space, spaces around the text, and the words epoch, infinity and -infinity. The third element is Spark 3.5's
# answer as in REJECTED, the last DuckDB's.
ACCEPTED_BY_DUCKDB = [
    ("yyyy-MM-dd HH:mm:ss", "1997-2-28 10:30:00", "upgrade", datetime(1997, 2, 28, 10, 30)),
    pytest.param(
        "yyyy-MM-dd HH:mm:ss",
        "97-02-28 10:30:00",
        "upgrade",
        datetime(97, 2, 28, 10, 30),
        marks=pytest.mark.skipif(
            sys.platform == "win32", reason="outside the range of the Windows C runtime, as for PySpark on Windows"
        ),
    ),
    ("yyyy-MM-dd HH:mm:ss", "1997-02-28  10:30:00", "upgrade", datetime(1997, 2, 28, 10, 30)),
    ("yyyy-MM-dd", " 2024-01-05", "upgrade", datetime(2024, 1, 5)),
    ("yyyy-MM-dd", "2024-01-05 ", "upgrade", datetime(2024, 1, 5)),
    ("yyyy MM", "2024\t01", "null", datetime(2024, 1, 1)),
    ("yyyy-MM-dd", "epoch", "null", datetime(1970, 1, 1)),
    ("yyyy-MM-dd", "infinity", "null", datetime(9999, 12, 31, 23, 59, 59, 999999)),
    ("yyyy-MM-dd", "-infinity", "null", datetime(1, 1, 1)),
    ("yyyy-MM-dd HH:mm:ss.S", "2024-01-05 10:30:00.12", "upgrade", datetime(2024, 1, 5, 10, 30, 0, 120000)),
]

# Text that Spark reads and DuckDB's strptime rejects: leading zeros beyond a field's width, and literal letters in
# another case.
REJECTED_BY_DUCKDB = [
    ("d/M/yyyy", "005/01/2024", datetime(2024, 1, 5)),
    ("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", "2024-01-05t10:30:00.123z", datetime(2024, 1, 5, 10, 30, 0, 123000)),
]


def _first(df, column):
    return df.select(column.alias("r")).collect()[0].r


def _assert_rejected(df, pattern, spark_3):
    if SPARK_3 and spark_3 == "upgrade":
        for column in [F.to_timestamp("s", pattern), F.try_to_timestamp("s", F.lit(pattern))]:
            with pytest.raises((Py4JJavaError, SparkUpgradeException), match="INCONSISTENT_BEHAVIOR_CROSS_VERSION"):
                _first(df, column)
        return
    assert _first(df, F.try_to_timestamp("s", F.lit(pattern))) is None
    for column in [F.to_timestamp("s", pattern), F.to_date("s", pattern)]:
        if SPARK_3:
            assert _first(df, column) is None
        else:
            with pytest.raises(PARSE_ERROR):
                _first(df, column)


class TestSparkDatetimePattern:
    @pytest.mark.parametrize(("pattern", "text", "expected"), PARSED)
    def test_parse(self, spark, pattern, text, expected):
        df = spark.createDataFrame([(text,)], ["s"])
        assert _first(df, F.to_timestamp("s", pattern)) == expected
        assert _first(df, F.try_to_timestamp("s", F.lit(pattern))) == expected
        assert _first(df, F.to_timestamp_ntz("s", F.lit(pattern))) == expected
        assert _first(df, F.to_date("s", pattern)) == expected.date()

    @pytest.mark.parametrize(("pattern", "text", "spark_3"), REJECTED)
    def test_parse_rejects_what_spark_rejects(self, spark, pattern, text, spark_3):
        _assert_rejected(spark.createDataFrame([(text,)], ["s"]), pattern, spark_3)

    @pytest.mark.parametrize(("pattern", "text", "spark_3", "duckdb_value"), ACCEPTED_BY_DUCKDB)
    def test_parse_accepts_more_than_spark(self, spark, pattern, text, spark_3, duckdb_value):
        df = spark.createDataFrame([(text,)], ["s"])
        if USE_ACTUAL_SPARK:
            _assert_rejected(df, pattern, spark_3)
            return
        assert _first(df, F.to_timestamp("s", pattern)) == duckdb_value
        assert _first(df, F.try_to_timestamp("s", F.lit(pattern))) == duckdb_value
        assert _first(df, F.to_date("s", pattern)) == duckdb_value.date()

    @pytest.mark.parametrize(("pattern", "text", "spark_value"), REJECTED_BY_DUCKDB)
    def test_parse_rejects_more_than_spark(self, spark, pattern, text, spark_value):
        df = spark.createDataFrame([(text,)], ["s"])
        if USE_ACTUAL_SPARK:
            assert _first(df, F.to_timestamp("s", pattern)) == spark_value
            assert _first(df, F.try_to_timestamp("s", F.lit(pattern))) == spark_value
            return
        assert _first(df, F.try_to_timestamp("s", F.lit(pattern))) is None
        with pytest.raises(PARSE_ERROR):
            _first(df, F.to_timestamp("s", pattern))

    def test_parse_null(self, spark):
        df = spark.createDataFrame([("2024-01-05",), (None,)], ["s"])
        res = df.select(F.to_timestamp("s", "yyyy-MM-dd").alias("r")).collect()
        assert [r.r for r in res] == [datetime(2024, 1, 5), None]

    @pytest.mark.parametrize(
        ("pattern", "text", "expected"),
        [
            ("yyyy-DDD", "2024-005", datetime(2024, 1, 5)),
            # Spark shifts the instant to the session time zone, which defaults to the local one.
            (
                "yyyy-MM-dd HH:mm:ssXXX",
                "2024-01-05 10:30:00Z",
                datetime(2024, 1, 5, 10, 30, tzinfo=timezone.utc).astimezone().replace(tzinfo=None),
            ),
            ("yyyy-MM-dd[ HH:mm]", "2024-01-05", datetime(2024, 1, 5)),
            ("G yyyy-MM-dd", "AD 2024-01-05", datetime(2024, 1, 5)),
            ("yyyy-MM-dd kk:mm", "2024-01-05 24:30", datetime(2024, 1, 5, 0, 30)),
            ("yyyy-MM-dd HH:mm a", "2024-01-05 13:30 PM", datetime(2024, 1, 5, 13, 30)),
            ("Hmm", "930", datetime(1970, 1, 1, 9, 30)),
            # DuckDB reads the year 1900 where a pattern has none, Spark 1970.
            ("HH:mm:ss", "10:30:00", datetime(1970, 1, 1, 10, 30)),
            ("H", "007", datetime(1970, 1, 1, 7)),
            # DuckDB reads yy as 1969 to 2068, Spark as 2000 to 2099.
            ("yy-MM-dd", "99-01-05", datetime(2099, 1, 5)),
            # DuckDB reads at most four year digits.
            ("yyyyyy", "002024", datetime(2024, 1, 1)),
            # DuckDB reads 12 as noon without an am-pm marker, Spark as midnight.
            ("yyyy-MM-dd hh:mm", "2024-01-05 12:30", datetime(2024, 1, 5, 0, 30)),
            # DuckDB rounds nanoseconds to microseconds, Spark truncates them.
            ("yyyy-MM-dd HH:mm:ss.SSSSSSSSS", "2024-01-05 10:30:00.123456789", datetime(2024, 1, 5, 10, 30, 0, 123456)),
            ("yyyy-MM-dd HH:mm:ss.SSSSSSS", "2024-01-05 10:30:00.9999999", datetime(2024, 1, 5, 10, 30, 0, 999999)),
        ],
    )
    def test_parse_unsupported_pattern(self, spark, pattern, text, expected):
        df = spark.createDataFrame([(text,)], ["s"])
        if USE_ACTUAL_SPARK:
            assert _first(df, F.to_timestamp("s", pattern)) == expected
        else:
            with pytest.raises(ContributionsAcceptedError, match="not supported for parsing"):
                F.to_timestamp("s", pattern)

    @pytest.mark.parametrize(
        "pattern", ["yyyy Q", "EEE yyyy-MM-dd", "YYYY-MM-dd", "yyyy-ww", "yyyy-MM-dd'T", "yyyy#MM"]
    )
    def test_parse_invalid_pattern(self, spark, pattern):
        df = spark.createDataFrame([("2024-01-05",)], ["s"])
        with pytest.raises((IllegalArgumentException, SparkUpgradeException, SparkRuntimeException)):
            _first(df, F.to_timestamp("s", pattern))

    def test_try_to_timestamp_format_column(self, spark):
        df = spark.createDataFrame([("2024-01-05", "yyyy-MM-dd"), ("05/01/2024", "dd/MM/yyyy")], ["s", "f"])
        if USE_ACTUAL_SPARK:
            res = df.select(F.try_to_timestamp("s", F.col("f")).alias("r")).collect()
            assert [r.r for r in res] == [datetime(2024, 1, 5), datetime(2024, 1, 5)]
        else:
            with pytest.raises(ContributionsAcceptedError, match="literal datetime pattern"):
                F.try_to_timestamp("s", F.col("f"))

    def test_to_timestamp_ltz_and_ntz_with_format(self, spark):
        df = spark.createDataFrame([("2016-12-31",), ("bad",)], ["s"])
        good, bad = df.filter("s = '2016-12-31'"), df.filter("s = 'bad'")
        for function in [F.to_timestamp_ltz, F.to_timestamp_ntz]:
            assert _first(good, function("s", F.lit("yyyy-MM-dd"))) == datetime(2016, 12, 31)
            if SPARK_3:
                assert _first(bad, function("s", F.lit("yyyy-MM-dd"))) is None
            else:
                with pytest.raises(PARSE_ERROR):
                    _first(bad, function("s", F.lit("yyyy-MM-dd")))

    def test_to_date_with_format_keeps_date_only(self, spark):
        df = spark.createDataFrame([("1997-02-28 10:30:00",)], ["t"])
        assert _first(df, F.to_date("t", "yyyy-MM-dd HH:mm:ss")) == date(1997, 2, 28)

    def test_to_date_format_must_be_str(self, spark):
        df = spark.createDataFrame([("2024-01-05",)], ["s"])
        for function in [F.to_date, F.to_timestamp]:
            with pytest.raises(PySparkTypeError):
                _first(df, function("s", F.lit("yyyy-MM-dd")))

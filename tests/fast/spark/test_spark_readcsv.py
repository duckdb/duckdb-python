import datetime
import math
import sys
from decimal import Decimal

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException, error_condition
from spark_namespace.sql.types import Row

import duckdb
from duckdb.experimental.spark.exception import ContributionsAcceptedError

if USE_ACTUAL_SPARK:
    import pyspark

    SPARK_3 = pyspark.__version__.startswith("3.")
else:
    SPARK_3 = False


def _read(spark, tmp_path, text, **options):
    path = tmp_path / "data.csv"
    path.write_text(text, newline="", encoding="utf-8")
    return spark.read.csv(path.as_posix(), **options)


def _rows(df):
    return [tuple(row) for row in df.collect()]


def _typed(spark, tmp_path, data_type, texts, **options):
    """Each text read as ``data_type`` from a line of its own, in the order of the texts."""
    text = "".join(f"{i},{t}\n" for i, t in enumerate(texts))
    df = _read(spark, tmp_path, text, schema=f"k int, v {data_type}", **options)
    return [row.v for row in sorted(df.collect())]


def _inferred(spark, tmp_path, *texts, **options):
    text = "v\n" + "".join(f"{t}\n" for t in texts)
    df = _read(spark, tmp_path, text, header=True, inferSchema=True, **options)
    return df.dtypes[0][1], [row.v for row in df.collect()]


class TestSparkReadCSV:
    def test_read_csv(self, spark, tmp_path):
        file_path = tmp_path / "basic.csv"
        file_path.write_text("1,2\n3,4\n5,6\n", newline="", encoding="utf-8")
        df = spark.read.csv(file_path.as_posix())
        res = df.collect()

        assert df.columns == ["_c0", "_c1"]
        # Spark reads every column as a string unless inferSchema is set.
        assert sorted(res) == [
            Row(_c0="1", _c1="2"),
            Row(_c0="3", _c1="4"),
            Row(_c0="5", _c1="6"),
        ]

    def test_read_csv_inferring_types(self, spark, tmp_path):
        file_path = tmp_path / "basic.csv"
        file_path.write_text("1,2\n3,4\n", newline="", encoding="utf-8")
        df = spark.read.csv(file_path.as_posix(), inferSchema=True)
        assert df.columns == ["_c0", "_c1"]
        assert sorted(df.collect()) == [Row(_c0=1, _c1=2), Row(_c0=3, _c1=4)]


# Spark reads the fields of a line by position, NULL past its end, and drops the fields past the last column.
_RAGGED = "a,b\n1,2\n3\n4,5,6\n"


class TestLinesOfAnotherWidth:
    def test_wider_line_after_the_header(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, "a,b\n1,2,3\n", schema="a int, b int", header=True)) == [(1, 2)]

    @pytest.mark.parametrize(
        ("schema", "rows"),
        [
            ("a int, b int", [(1, 2), (3, None), (4, 5)]),
            ("a int, b int, c int", [(1, 2, None), (3, None, None), (4, 5, 6)]),
            ("a int", [(1,), (3,), (4,)]),
        ],
    )
    def test_with_a_schema(self, spark, tmp_path, schema, rows):
        assert _rows(_read(spark, tmp_path, _RAGGED, schema=schema, header=True)) == rows

    def test_with_a_schema_and_no_header(self, spark, tmp_path):
        df = _read(spark, tmp_path, _RAGGED, schema="a string, b string")
        assert _rows(df) == [("a", "b"), ("1", "2"), ("3", None), ("4", "5")]

    def test_without_a_schema(self, spark, tmp_path):
        df = _read(spark, tmp_path, _RAGGED, header=True)
        assert df.columns == ["a", "b"]
        assert _rows(df) == [("1", "2"), ("3", None), ("4", "5")]
        df = _read(spark, tmp_path, _RAGGED)
        assert df.columns == ["_c0", "_c1"]
        assert _rows(df) == [("a", "b"), ("1", "2"), ("3", None), ("4", "5")]

    def test_first_line_narrower_than_a_later_one(self, spark, tmp_path):
        df = _read(spark, tmp_path, "1\n2,3\n")
        assert df.columns == ["_c0"]
        assert _rows(df) == [("1",), ("2",)]

    def test_header_wider_than_the_data(self, spark, tmp_path):
        df = _read(spark, tmp_path, "a,b,c\n1,2\n3,4\n", header=True)
        assert df.columns == ["a", "b", "c"]
        assert _rows(df) == [("1", "2", None), ("3", "4", None)]

    def test_inferring_types(self, spark, tmp_path):
        df = _read(spark, tmp_path, "a,b\n1,x\n3\n", header=True, inferSchema=True)
        assert df.dtypes == [("a", "int" if USE_ACTUAL_SPARK else "bigint"), ("b", "string")]
        assert _rows(df) == [(1, "x"), (3, None)]


class TestDialect:
    def test_separator_is_a_comma_unless_given(self, spark, tmp_path):
        df = _read(spark, tmp_path, "a;b\n1;2\n", header=True)
        assert df.columns == ["a;b"]
        assert _rows(df) == [("1;2",)]
        assert _rows(_read(spark, tmp_path, "a;b\n1;2\n", header=True, sep=";")) == [("1", "2")]

    def test_backslash_escape(self, spark, tmp_path):
        text = 'a,b\n"p\\"q",2\n'
        assert _rows(_read(spark, tmp_path, text, header=True, escape="\\")) == [('p"q', "2")]
        df = _read(spark, tmp_path, text, schema="a string, b int", header=True, escape="\\")
        assert _rows(df) == [('p"q', 2)]
        if USE_ACTUAL_SPARK:
            assert _rows(_read(spark, tmp_path, text, header=True)) == [('p"q', "2")]
        else:
            # DuckDB's escape is the quote, so its sniffer fails on a backslash before a quote, and its lenient
            # reader reads the line as one field.
            with pytest.raises(duckdb.InvalidInputException, match="Error when sniffing file"):
                _read(spark, tmp_path, text, header=True)
            assert _rows(_read(spark, tmp_path, text, schema="a string, b int", header=True)) == [("p\\q,2", None)]

    def test_doubled_quote(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, 'a,b\n"x""y",2\n', header=True, escape='"')) == [('x"y', "2")]
        # Spark's escape is a backslash, so it reads the field as it is written.
        expected = '"x""y"' if USE_ACTUAL_SPARK else 'x"y'
        assert _rows(_read(spark, tmp_path, '"x""y",1\n')) == [(expected, "1")]

    def test_backslashes_are_text(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, 'C:\\temp,1\n"C:\\temp",2\n')) == [("C:\\temp", "1"), ("C:\\temp", "2")]
        assert _rows(_read(spark, tmp_path, 'C:\\temp,a\\"b,x\\\n')) == [("C:\\temp", 'a\\"b', "x\\")]
        rows = _rows(_read(spark, tmp_path, 'C:\\temp,1\n"C:\\temp",2\n', escape="\\"))
        # DuckDB drops a backslash escape that escapes neither the quote nor itself.
        kept = "C:\\temp" if USE_ACTUAL_SPARK else "C:temp"
        assert rows == [(kept, "1"), (kept, "2")]

    def test_doubled_backslash_inside_quotes(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, '"a\\\\b","c\\\\"\n', escape="\\")) == [("a\\b", "c\\")]
        expected = ("a\\b", "c\\") if USE_ACTUAL_SPARK else ("a\\\\b", "c\\\\")
        assert _rows(_read(spark, tmp_path, '"a\\\\b","c\\\\"\n')) == [expected]

    def test_escape_follows_a_given_quote(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, "'a,b',\"c\"\n", quote="'")) == [("a,b", '"c"')]

    def test_mixed_line_ends(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, "a,b\r\n1,2\n3,4\r\n")) == [("a", "b"), ("1", "2"), ("3", "4")]

    def test_unclosed_quote(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, '"ab,2\n', schema="a string, b string")) == [("ab,2", None)]
        if USE_ACTUAL_SPARK:
            assert _rows(_read(spark, tmp_path, '"ab,2\n')) == [("ab,2",)]
        else:
            # Without a schema the width of a line comes from DuckDB's sniffer, which fails here.
            with pytest.raises(duckdb.InvalidInputException, match="Error when sniffing file"):
                _read(spark, tmp_path, '"ab,2\n')
        rows = _rows(_read(spark, tmp_path, 'a,b\n"1,2\n3,4\n5,6\n', header=True, schema="a string, b string"))
        # Spark ends an unclosed quote at the end of the line, DuckDB's lenient reader at the end of the file.
        expected = [("1,2", None), ("3", "4"), ("5", "6")] if USE_ACTUAL_SPARK else [("1,2\n3,4\n5,6", None)]
        assert rows == expected

    def test_wide_line_after_the_sniffer_sample(self, spark, tmp_path):
        text = "a,b\n" + "i,1\n" * 3000 + "9,1,extra\n"
        df = _read(spark, tmp_path, text, header=True)
        assert df.columns == ["a", "b"]
        assert _rows(df)[-1] == ("9", "1")
        assert _rows(_read(spark, tmp_path, text, header=True, schema="a string, b int"))[-1] == ("9", 1)

    def test_header_names(self, spark, tmp_path):
        # Spark names an empty header field after its position and adds the position to names that repeat.
        df = _read(spark, tmp_path, "a,,A,b\n1,2,3,4\n", header=True)
        assert df.columns == ["a0", "_c1", "A2", "b"]
        assert _rows(df) == [("1", "2", "3", "4")]
        assert _read(spark, tmp_path, "a,a,b,A\n1,2,3,4\n", header=True).columns == ["a0", "a1", "b", "A3"]

    def test_header_only(self, spark, tmp_path):
        df = _read(spark, tmp_path, "a,b\n", header=True)
        assert df.columns == ["a", "b"]
        assert df.collect() == []

    def test_header_line_ends(self, spark, tmp_path):
        df = _read(spark, tmp_path, "a,b\r\n1,2\r\n", header=True)
        assert df.columns == ["a", "b"]
        assert _rows(df) == [("1", "2")]

    @pytest.mark.parametrize(
        ("text", "columns", "rows"),
        [
            ("\n\na,b\n1,2\n", ["a", "b"], [("a", "b"), ("1", "2")]),
            ("  \n\n   \na,b\n1,2\n", ["  "], [(None,), ("   ",), ("a",), ("1",)]),
        ],
    )
    def test_header_after_blank_lines(self, spark, tmp_path, text, columns, rows):
        df = _read(spark, tmp_path, text, header=True)
        if USE_ACTUAL_SPARK:
            assert (df.columns, _rows(df)) == (["a", "b"], [("1", "2")])
        else:
            # The names come from the first line that is not empty, and DuckDB skips the first line as the header,
            # empty or not; a line of whitespace is a row.
            assert (df.columns, _rows(df)) == (columns, rows)

    def test_blank_lines(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, "a,b\n\n1,2\n\n")) == [("a", "b"), ("1", "2")]
        # Spark skips a line of whitespace and an empty line; DuckDB reads the first as a row, and the second as a
        # NULL row in a file of one column.
        spaces = _rows(_read(spark, tmp_path, "a,b\n   \n1,2\n"))
        assert spaces == ([("a", "b"), ("1", "2")] if USE_ACTUAL_SPARK else [("a", "b"), ("   ", None), ("1", "2")])
        one_column = _rows(_read(spark, tmp_path, "a\n\n1\n \n2\n"))
        expected = [("a",), ("1",), ("2",)] if USE_ACTUAL_SPARK else [("a",), (None,), ("1",), (" ",), ("2",)]
        assert one_column == expected

    def test_line_of_separators_is_a_row(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, "a,b\n,\n")) == [("a", "b"), (None, None)]

    def test_lines_of_whitespace_and_separators(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, "a,b\n ,\n1,2\n")) == [("a", "b"), (" ", None), ("1", "2")]
        rows = _rows(_read(spark, tmp_path, "a,b\n ,\n1,2\n", nullValue="NA"))
        assert rows == [("a", "b"), (" ", None), ("1", "2")]
        assert _rows(_read(spark, tmp_path, 'x\n"  "\ny\n')) == [("x",), ("  ",), ("y",)]
        df = _read(spark, tmp_path, "  ,,\n1,2,3\n", header=True)
        assert df.columns == ["  ", "_c1", "_c2"]
        assert _rows(df) == [("1", "2", "3")]

    def test_empty_first_field(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, "1,2\n,3\n4,5\n", schema="a string")) == [("1",), (None,), ("4",)]
        assert _rows(_read(spark, tmp_path, "a\n,b\n")) == [("a",), (None,)]
        assert _rows(_read(spark, tmp_path, "h\n,3\n4,5\n", header=True)) == [(None,), ("4",)]

    def test_lines_equal_to_the_header(self, spark, tmp_path):
        rows = _rows(_read(spark, tmp_path, "a,b\na,b\na,b\n1,2\n", header=True))
        assert rows == [("a", "b"), ("a", "b"), ("1", "2")]
        rows = _rows(_read(spark, tmp_path, "a,1\na,2\na,3\nb,4\n", header=True, schema="x string"))
        assert rows == [("a",), ("a",), ("b",)]
        assert _read(spark, tmp_path, "x\n" * 1000).count() == 1000

    def test_files_of_a_glob(self, spark, tmp_path):
        (tmp_path / "a.csv").write_text("h1,h2\n1,x\n2,y\n", newline="", encoding="utf-8")
        (tmp_path / "b.csv").write_text("h1,h2\n3,z\n", newline="", encoding="utf-8")
        path = (tmp_path / "*.csv").as_posix()
        assert sorted(_rows(spark.read.csv(path, header=True))) == [("1", "x"), ("2", "y"), ("3", "z")]
        df = spark.read.csv(path, header=True, inferSchema=True)
        assert df.dtypes == [("h1", "int" if USE_ACTUAL_SPARK else "bigint"), ("h2", "string")]
        assert sorted(_rows(df)) == [(1, "x"), (2, "y"), (3, "z")]

    def test_quoted_line_break(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, 'a,b\n"x\ny",2\n', header=True, multiLine=True)) == [("x\ny", "2")]
        if USE_ACTUAL_SPARK:
            # Without multiLine Spark reads each line as a record of its own.
            assert _rows(_read(spark, tmp_path, '"x\ny"\n')) == [("x",), ('y"',)]
        else:
            with pytest.raises(Exception, match="quoted new lines"):
                _rows(_read(spark, tmp_path, '"x\ny"\n'))


class TestNullValues:
    def test_empty_fields(self, spark, tmp_path):
        # An empty field is NULL, and a quoted one the empty string, which is NULL for the default nullValue only.
        assert _rows(_read(spark, tmp_path, 'a,"",c\n')) == [("a", None, "c")]
        assert _rows(_read(spark, tmp_path, 'a,"",,NA,"NA"\n', nullValue="NA")) == [("a", "", None, None, None)]

    def test_null_value_with_a_header(self, spark, tmp_path):
        df = _read(spark, tmp_path, "a,b\n1,NA\n,2\n", header=True, nullValue="NA")
        assert _rows(df) == [("1", None), (None, "2")]
        df = _read(spark, tmp_path, "a,b\n1,NA\n,2\n", header=True, nullValue="NA", schema="a int, b int")
        assert _rows(df) == [(1, None), (None, 2)]

    def test_header_field_equal_to_the_null_value(self, spark, tmp_path):
        df = _read(spark, tmp_path, "a,NA,b\n1,NA,3\n", header=True, nullValue="NA")
        assert df.columns == ["a", "_c1", "b"]
        assert _rows(df) == [("1", None, "3")]


class TestTypedValues:
    """A field of a schema's type converts with DuckDB's cast, NULL where it does not convert, as in Spark."""

    def test_text_that_does_not_convert_is_null(self, spark, tmp_path):
        # Spark keeps the other fields of the line.
        assert _rows(_read(spark, tmp_path, "x,1\n2,y\n", schema="a int, b int")) == [(None, 1), (2, None)]
        assert _rows(_read(spark, tmp_path, "x,y\n1,x\n", schema="x int, y string")) == [(None, "y"), (1, "x")]

    def test_integers(self, spark, tmp_path):
        texts = ["1", "+1", "-0", "007", " 1", "1 ", "1.0", "1e3", "0x1F", "1_000", "2147483648"]
        spark_values = [1, 1, 0, 7] + [None] * 7
        duckdb_values = [1, 1, 0, 7, 1, 1, 1, 1000, 31, 1000, None]
        assert _typed(spark, tmp_path, "int", texts) == (spark_values if USE_ACTUAL_SPARK else duckdb_values)
        assert _typed(spark, tmp_path, "bigint", ["2147483648", "9223372036854775808"]) == [2147483648, None]
        assert _typed(spark, tmp_path, "tinyint", ["127", "128", "-128"]) == [127, None, -128]
        assert _typed(spark, tmp_path, "smallint", ["32767", "32768"]) == [32767, None]

    def test_doubles(self, spark, tmp_path):
        texts = ["1.5", " 1.5 ", "1e3", ".5", "5.", "1E400", "Infinity", "-Infinity", "+Infinity"]
        inf = math.inf
        assert _typed(spark, tmp_path, "double", texts) == [1.5, 1.5, 1000.0, 0.5, 5.0, inf, inf, -inf, inf]
        assert _typed(spark, tmp_path, "double", ["1.5e", '"1,5"', "x"]) == [None] * 3
        values = _typed(spark, tmp_path, "double", ["1.5d", "1.5f", "0x1p3", "1_0", "+Inf", "inf"])
        # Java's parseDouble reads a type suffix and hexadecimal text; DuckDB reads digit separators and more
        # spellings of infinity.
        assert values == ([1.5, 1.5, 8.0, None, None, None] if USE_ACTUAL_SPARK else [None, None, None, 10.0, inf, inf])
        (nan,) = _typed(spark, tmp_path, "double", ["nan"])
        assert nan is None if USE_ACTUAL_SPARK else math.isnan(nan)

    def test_nan_and_infinity_texts(self, spark, tmp_path):
        nan, inf, minus_inf = _typed(spark, tmp_path, "double", ["NaN", "Inf", "-Inf"])
        assert math.isnan(nan)
        assert (inf, minus_inf) == (math.inf, -math.inf)
        options = {"nanValue": "NA", "positiveInf": "+", "negativeInf": "-"}
        values = _typed(spark, tmp_path, "double", ["NA", "+", "-", "Inf", "NaN"], **options)
        assert math.isnan(values[0])
        assert math.isnan(values[4])
        assert values[1:3] == [math.inf, -math.inf]
        # Spark reads Inf only as its positiveInf text; DuckDB's cast reads it whatever positiveInf is.
        assert values[3] == (None if USE_ACTUAL_SPARK else math.inf)
        values = _typed(spark, tmp_path, "float", ["1.5", "3.4e39", "NaN"])
        assert values[:2] == [1.5, math.inf]
        assert math.isnan(values[2])

    def test_decimals(self, spark, tmp_path):
        texts = ["1.234", "1.235", "-1.235", "1e2", "+1.5", ".5", "123.456", "1234.5"]
        expected = ["1.23", "1.24", "-1.24", "100.00", "1.50", "0.50", "123.46", None]
        assert _typed(spark, tmp_path, "decimal(5,2)", texts) == [None if e is None else Decimal(e) for e in expected]
        assert _typed(spark, tmp_path, "decimal(5,2)", ["1.5d", "abc", '"1,000"']) == [None] * 3
        # Spark leaves out the commas its default locale groups digits with, and does not trim; DuckDB trims.
        spark_values, duckdb_values = [Decimal("15.00"), None], [None, Decimal("1.50")]
        values = _typed(spark, tmp_path, "decimal(5,2)", ['"1,5"', " 1.5"])
        assert values == (spark_values if USE_ACTUAL_SPARK else duckdb_values)

    def test_booleans(self, spark, tmp_path):
        assert _typed(spark, tmp_path, "boolean", ["true", "TRUE", "False", " true"]) == [True, True, False, None]
        values = _typed(spark, tmp_path, "boolean", ["t", "yes", "1"])
        assert values == ([None] * 3 if USE_ACTUAL_SPARK else [True] * 3)

    def test_dates(self, spark, tmp_path):
        texts = ["2024-01-02", "2024-1-2", "2024-01-02 12:00:00", "2024-01-02T12:00", " 2024-01-02 "]
        assert _typed(spark, tmp_path, "date", texts) == [datetime.date(2024, 1, 2)] * 5
        # Spark reads a year alone and a year and month.
        values = _typed(spark, tmp_path, "date", ["2024", "2024-01"])
        assert values == ([datetime.date(2024, 1, 1)] * 2 if USE_ACTUAL_SPARK else [None, None])

    def test_invalid_dates(self, spark, tmp_path):
        if SPARK_3:
            pytest.skip("Spark 3.5 raises on text its legacy date parser reads differently")
        texts = ["02/01/2024", "20240102", "2024-02-30", "2024-01-02Z", "1"]
        assert _typed(spark, tmp_path, "date", texts) == [None] * len(texts)
        # DuckDB reads slashes and years of fewer digits.
        values = _typed(spark, tmp_path, "date", ["2024/01/02", "1-1-1"])
        assert values == ([None, None] if USE_ACTUAL_SPARK else [datetime.date(2024, 1, 2), datetime.date(1, 1, 1)])

    def test_timestamps(self, spark, tmp_path):
        texts = ["2024-01-02 03:04:05", "2024-01-02T03:04:05", "2024-1-2 3:4:5", " 2024-01-02 03:04:05 "]
        texts += ["2024-01-02", "2024-01-02 03:04:05.123456789"]
        expected = [datetime.datetime(2024, 1, 2, 3, 4, 5)] * 4
        expected += [datetime.datetime(2024, 1, 2), datetime.datetime(2024, 1, 2, 3, 4, 5, 123456)]
        assert _typed(spark, tmp_path, "timestamp", texts) == expected
        # Spark reads a year and month, and a date and an hour.
        values = _typed(spark, tmp_path, "timestamp", ["2024-01", "2024-01-02 03"])
        spark_values = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2, 3)]
        assert values == (spark_values if USE_ACTUAL_SPARK else [None, None])

    def test_timestamps_with_a_zone(self, spark, tmp_path):
        instant = datetime.datetime(2024, 1, 2, 11, tzinfo=datetime.timezone.utc).astimezone().replace(tzinfo=None)
        texts = ["2024-01-02T12:00:00+01:00", "2024-01-02 12:00:00+0100", "2024-01-02 12:00:00+01"]
        texts += ["2024-01-02T11:00:00Z", "2024-01-02 12:00:00 Europe/Amsterdam"]
        assert _typed(spark, tmp_path, "timestamp", texts) == [instant] * len(texts)
        # Spark reads an offset after GMT.
        expected = instant if USE_ACTUAL_SPARK else None
        assert _typed(spark, tmp_path, "timestamp", ["2024-01-02 12:00:00GMT+1"]) == [expected]

    def test_invalid_timestamps(self, spark, tmp_path):
        if SPARK_3:
            pytest.skip("Spark 3.5 raises on text its legacy timestamp parser reads differently")
        texts = ["2024-01-02 23:59:60", "2024-01-02T", "1704196800", "x"]
        assert _typed(spark, tmp_path, "timestamp", texts) == [None] * len(texts)
        # DuckDB reads hour 24 as midnight of the next day.
        values = _typed(spark, tmp_path, "timestamp", ["2024-01-02 24:00:00"])
        assert values == ([None] if USE_ACTUAL_SPARK else [datetime.datetime(2024, 1, 3)])

    @pytest.mark.skipif(
        sys.platform == "win32", reason="outside the range of the Windows C runtime, as for PySpark on Windows"
    )
    def test_two_digit_year_timestamp(self, spark, tmp_path):
        if SPARK_3:
            pytest.skip("Spark 3.5 raises on text its legacy timestamp parser reads differently")
        # DuckDB reads a year of two digits as it is.
        values = _typed(spark, tmp_path, "timestamp", ["24-01-02"])
        assert values == ([None] if USE_ACTUAL_SPARK else [datetime.datetime(24, 1, 2)])

    def test_timestamps_without_a_zone(self, spark, tmp_path):
        texts = ["2024-01-02 03:04:05", "2024-01-02T03:04:05"]
        assert _typed(spark, tmp_path, "timestamp_ntz", texts) == [datetime.datetime(2024, 1, 2, 3, 4, 5)] * 2
        # Spark rejects text with a zone and reads a year and month; DuckDB drops the zone.
        values = _typed(spark, tmp_path, "timestamp_ntz", ["2024-01-02T03:04:05Z", "2024-01-02 03:04+01:00", "2024-01"])
        duckdb_values = [datetime.datetime(2024, 1, 2, 3, 4, 5), datetime.datetime(2024, 1, 2, 3, 4), None]
        assert values == ([None, None, datetime.datetime(2024, 1, 1)] if USE_ACTUAL_SPARK else duckdb_values)

    def test_strings(self, spark, tmp_path):
        assert _typed(spark, tmp_path, "string", ["x", " x ", '" y "']) == ["x", " x ", " y "]

    def test_binary(self, spark, tmp_path):
        if SPARK_3:
            with pytest.raises(AnalysisException) as error:
                _typed(spark, tmp_path, "binary", ["abc"])
            assert error_condition(error.value) == "UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE"
            return
        # DuckDB's cast reads a backslash escape of a byte.
        expected = b"\\x41" if USE_ACTUAL_SPARK else b"A"
        assert _typed(spark, tmp_path, "binary", ["abc", "\\x41"]) == [b"abc", expected]

    def test_intervals(self, spark, tmp_path):
        expected = [None, None] if USE_ACTUAL_SPARK else [datetime.timedelta(seconds=1), datetime.timedelta(days=1)]
        assert _typed(spark, tmp_path, "interval day to second", ["1", "1 day"]) == expected

    @pytest.mark.parametrize(
        ("data_type", "name"),
        [
            ("array<int>", "ARRAY<INT>"),
            ("struct<x:int>", "STRUCT<x: INT>"),
            ("map<string,int>", "MAP<STRING, INT>"),
            ("void", "VOID"),
        ],
    )
    def test_types_csv_cannot_hold(self, spark, tmp_path, data_type, name):
        with pytest.raises(AnalysisException, match=f'column `a` of the type "{name}"') as error:
            _read(spark, tmp_path, "1\n", schema=f"a {data_type}").collect()
        assert error_condition(error.value) == "UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE"

    def test_names_differing_in_case_only(self, spark, tmp_path):
        with pytest.raises(AnalysisException, match="The column `a` already exists") as error:
            _read(spark, tmp_path, "1,2\n", schema="a int, A int").collect()
        assert error_condition(error.value) == "COLUMN_ALREADY_EXISTS"


class TestInference:
    """With inferSchema the types are DuckDB's type detection's, read with DuckDB's cast."""

    @pytest.mark.parametrize(
        ("texts", "spark_type", "duckdb_type"),
        [
            (["1", "-0"], "int", "bigint"),
            (["1", "2147483648"], "bigint", "bigint"),
            (["9223372036854775808"], "decimal(19,0)", "hugeint"),
            (["1", "12345678901234567890"], "decimal(20,0)", "hugeint"),
            (["123456789012345678901234567890123456789"], "double", "hugeint"),
            (["1" * 41], "double", "double"),
            (["5.", "1.0E1"], "decimal(2,0)", "double"),
            (["1", "1.5", "1e3", "1E10", " 1", "NaN", "-Inf", "Infinity"], "double", "double"),
            (["1.5d"], "double", "string"),
            (["12345678901234567890", "1.5"], "double", "double"),
            (["2024-01-02"], "date", "date"),
            (["2024-01-02", "2024-01-02 03:04:05"], "timestamp", "string"),
            (["2024-01-02T03:04:05Z"], "timestamp", "timestamp"),
            (["2024-1-2"], "timestamp", "date"),
            (["2024-01", "2024-01-02 03"], "timestamp", "string"),
            (["true", "FALSE"], "boolean", "boolean"),
            (["0x1F"], "string", "bigint"),
            (["yes", "t"], "string", "boolean"),
            (["2024/01/02"], "string", "date"),
            (["03/02/2024"], "string", "date"),
            (["12:00:00"], "timestamp", "time"),
            (["+1", "007"], "int", "string"),
            (["1_000", "nan", "inf", '"1,000"', "1.5x"], "string", "string"),
            (["1", "true"], "string", "string"),
            (["2024-01-02", "1"], "string", "string"),
            (["1", ""], "int", "bigint"),
            (["", ""], "string", "string"),
        ],
    )
    def test_types(self, spark, tmp_path, texts, spark_type, duckdb_type):
        assert _inferred(spark, tmp_path, *texts)[0] == (spark_type if USE_ACTUAL_SPARK else duckdb_type)

    def test_values(self, spark, tmp_path):
        assert _inferred(spark, tmp_path, "1", " 2", "3.5") == ("double", [1.0, 2.0, 3.5])
        times = ("timestamp", [datetime.datetime(2024, 1, 2, 3, 4, 5), datetime.datetime(2024, 1, 3, 4, 5, 6)])
        assert _inferred(spark, tmp_path, "2024-01-02 03:04:05", "2024-01-03T04:05:06") == times
        assert _inferred(spark, tmp_path, "2024-01-02")[1] == [datetime.date(2024, 1, 2)]

    def test_values_of_other_types(self, spark, tmp_path):
        df = _read(spark, tmp_path, "hex,yn,tf,sl\n0x1F,yes,t,2024/01/02\n", header=True, inferSchema=True)
        if USE_ACTUAL_SPARK:
            assert df.dtypes == [("hex", "string"), ("yn", "string"), ("tf", "string"), ("sl", "string")]
            assert _rows(df) == [("0x1F", "yes", "t", "2024/01/02")]
        else:
            assert df.dtypes == [("hex", "bigint"), ("yn", "boolean"), ("tf", "boolean"), ("sl", "date")]
            assert _rows(df) == [(31, True, True, datetime.date(2024, 1, 2))]

    def test_dates_in_a_detected_format(self, spark, tmp_path):
        df = _read(spark, tmp_path, "d\n03/02/2024\n04/02/2024\n", header=True, inferSchema=True)
        if USE_ACTUAL_SPARK:
            assert _rows(df) == [("03/02/2024",), ("04/02/2024",)]
        else:
            assert _rows(df) == [(datetime.date(2024, 2, 3),), (datetime.date(2024, 2, 4),)]

    def test_lines_repeating_the_header(self, spark, tmp_path):
        df = _read(spark, tmp_path, "v\n1\nv\n2\n", header=True, inferSchema=True)
        if USE_ACTUAL_SPARK:
            # Spark leaves lines that repeat the header out of the inference.
            assert df.dtypes == [("v", "int")]
            assert _rows(df) == [(1,), (None,), (2,)]
        else:
            assert df.dtypes == [("v", "string")]
            assert _rows(df) == [("1",), ("v",), ("2",)]

    def test_null_and_nan_values(self, spark, tmp_path):
        integer = "int" if USE_ACTUAL_SPARK else "bigint"
        assert _inferred(spark, tmp_path, "NA", "1", nullValue="NA") == (integer, [None, 1])
        assert _inferred(spark, tmp_path, '""', "1", "NA", nullValue="NA")[1] == [None, 1, None]
        # DuckDB's detection does not know a nanValue.
        expected = "double" if USE_ACTUAL_SPARK else "string"
        assert _inferred(spark, tmp_path, "NA", "1.5", nanValue="NA")[0] == expected

    def test_values_after_the_sample(self, spark, tmp_path):
        """DuckDB detects the types from a sample of the lines and reads the rest as its own reader does."""
        df = _read(spark, tmp_path, "v\n" + "1\n" * 30000 + "x\n", header=True, inferSchema=True)
        if USE_ACTUAL_SPARK:
            assert df.dtypes == [("v", "string")]
            assert _rows(df)[-1] == ("x",)
        else:
            assert df.dtypes == [("v", "bigint")]
            with pytest.raises(duckdb.ConversionException, match='Could not convert string "x"'):
                df.collect()
        df = _read(spark, tmp_path, "v\n" + "1\n" * 30000 + "1.5\n", header=True, inferSchema=True)
        expected = ("double", 1.5) if USE_ACTUAL_SPARK else ("bigint", 2)
        assert (df.dtypes[0][1], _rows(df)[-1][0]) == expected
        text = "v\n" + "2024-01-02\n" * 30000 + "2024-01-03 00:00:00\n"
        df = _read(spark, tmp_path, text, header=True, inferSchema=True)
        last = datetime.datetime(2024, 1, 3)
        expected = ("timestamp", last) if USE_ACTUAL_SPARK else ("date", last.date())
        assert (df.dtypes[0][1], _rows(df)[-1][0]) == expected

    def test_blank_lines(self, spark, tmp_path):
        rows = _rows(_read(spark, tmp_path, "a,b\n1,2\n\n3,4\n", header=True, inferSchema=True))
        assert rows == [(1, 2), (3, 4)]
        # DuckDB reads an empty line of a file of one column as a NULL row, and a line of whitespace as a row.
        rows = _rows(_read(spark, tmp_path, "v\n1\n\n2\n", header=True, inferSchema=True))
        assert rows == ([(1,), (2,)] if USE_ACTUAL_SPARK else [(1,), (None,), (2,)])
        df = _read(spark, tmp_path, "x,y\n1,2\n   \n3,4\n", header=True, inferSchema=True)
        integer = "int" if USE_ACTUAL_SPARK else "bigint"
        assert df.dtypes == [("x", integer if USE_ACTUAL_SPARK else "string"), ("y", integer)]

    def test_header_of_another_width(self, spark, tmp_path):
        df = _read(spark, tmp_path, "v\n1,000\n", header=True, inferSchema=True)
        assert df.columns == ["v"]
        assert _rows(df) == [(1,)]

    def test_header_without_data(self, spark, tmp_path):
        df = _read(spark, tmp_path, "a,b\n", header=True, inferSchema=True)
        assert df.dtypes == [("a", "string"), ("b", "string")]


class TestDatetimePatterns:
    def test_date_format(self, spark, tmp_path):
        df = _read(spark, tmp_path, "2024/02/03\nFeb 3\n", schema="d date", dateFormat="yyyy/MM/dd")
        # Spark reads text that does not match the pattern as NULL.
        assert _rows(df) == [(datetime.date(2024, 2, 3),), (None,)]
        df = _read(spark, tmp_path, "2024/2/3\n", schema="d date", dateFormat="yyyy/M/d")
        assert _rows(df) == [(datetime.date(2024, 2, 3),)]

    def test_timestamp_format(self, spark, tmp_path):
        text = "03.02.2024 04:05:06\n"
        df = _read(spark, tmp_path, text, schema="t timestamp", timestampFormat="dd.MM.yyyy HH:mm:ss")
        assert _rows(df) == [(datetime.datetime(2024, 2, 3, 4, 5, 6),)]

    def test_timestamp_format_is_not_for_timestamps_without_a_zone(self, spark, tmp_path):
        text = "2024-02-03T04:05:06\n"
        df = _read(spark, tmp_path, text, schema="t timestamp_ntz", timestampFormat="dd.MM.yyyy HH:mm:ss")
        assert _rows(df) == [(datetime.datetime(2024, 2, 3, 4, 5, 6),)]

    def test_patterns_apply_only_to_their_types(self, spark, tmp_path):
        df = _read(spark, tmp_path, "2024/02/03,x\n", schema="s string, t string", dateFormat="yyyy/MM/dd")
        assert _rows(df) == [("2024/02/03", "x")]
        assert _read(spark, tmp_path, "2024/02/03\n", dateFormat="yyyy/MM/dd").dtypes == [("_c0", "string")]

    def test_inferring_with_a_pattern(self, spark, tmp_path):
        if USE_ACTUAL_SPARK:
            df = _read(spark, tmp_path, "2024/02/03\n", dateFormat="yyyy/MM/dd", inferSchema=True)
            assert df.dtypes == [("_c0", "date")]
        else:
            with pytest.raises(ContributionsAcceptedError):
                _read(spark, tmp_path, "2024/02/03\n", dateFormat="yyyy/MM/dd", inferSchema=True)


class TestEnforceSchema:
    def test_header_that_does_not_match_the_schema(self, spark, tmp_path):
        assert _rows(_read(spark, tmp_path, "a\nx\n", schema="b string", header=True)) == [("x",)]
        df = _read(spark, tmp_path, "a\nx\n", schema="b string", header=True, enforceSchema="TRUE")
        assert _rows(df) == [("x",)]
        # With enforceSchema off Spark checks the header against the schema and fails the read.
        with pytest.raises(Exception, match="CSV header does not conform to the schema"):
            _read(spark, tmp_path, "a\nx\n", schema="b string", header=True, enforceSchema=False).collect()

    def test_header_of_another_width(self, spark, tmp_path):
        with pytest.raises(Exception, match="Number of column in CSV header is not equal to number of fields"):
            _read(spark, tmp_path, "a,b\n1,2\n", schema="a int", header=True, enforceSchema="false").collect()

    @pytest.mark.parametrize("width", ["a,c", "a,b,c"])
    def test_header_of_each_file_of_a_glob(self, spark, tmp_path, width):
        (tmp_path / "1.csv").write_text("a,b\n1,2\n", newline="", encoding="utf-8")
        (tmp_path / "2.csv").write_text(f"{width}\n3,4\n", newline="", encoding="utf-8")
        path = (tmp_path / "*.csv").as_posix()
        df = spark.read.csv(path, header=True, schema="a int, b int")
        assert sorted(_rows(df)) == [(1, 2), (3, 4)]
        match = "CSV header does not conform" if width == "a,c" else "Number of column in CSV header"
        with pytest.raises(Exception, match=match):
            spark.read.csv(path, header=True, schema="a int, b int", enforceSchema=False).collect()

    def test_header_of_files_of_a_glob_that_match(self, spark, tmp_path):
        (tmp_path / "1.csv").write_text("a,b\n1,2\n", newline="", encoding="utf-8")
        (tmp_path / "2.csv").write_text("A,B\n3,4\n", newline="", encoding="utf-8")
        df = spark.read.csv((tmp_path / "*.csv").as_posix(), header=True, schema="a int, b int", enforceSchema=False)
        assert sorted(_rows(df)) == [(1, 2), (3, 4)]

    def test_header_that_matches_the_schema_ignoring_case(self, spark, tmp_path):
        df = _read(spark, tmp_path, "A,b\n1,2\n", schema="a int, B int", header=True, enforceSchema=False)
        assert df.columns == ["a", "B"]
        assert _rows(df) == [(1, 2)]


class TestLoad:
    def test_csv_by_its_extension(self, spark, tmp_path):
        path = (tmp_path / "data.csv").as_posix()
        (tmp_path / "data.csv").write_text("1,2\n3,4\n", newline="", encoding="utf-8")
        if USE_ACTUAL_SPARK:
            # Spark reads a path without a format as parquet.
            with pytest.raises(Exception):  # noqa: B017, PT011
                spark.read.load(path, schema="a int, b int").collect()
        else:
            assert _rows(spark.read.load(path, schema="a int, b int")) == [(1, 2), (3, 4)]
            assert _rows(spark.read.load(path)) == [("1", "2"), ("3", "4")]

    def test_header_row_is_data(self, spark, tmp_path):
        (tmp_path / "data.csv").write_text("x,y\n1,x\n2,y\n", newline="", encoding="utf-8")
        df = spark.read.load((tmp_path / "data.csv").as_posix(), format="csv", schema="x int, y string")
        assert _rows(df) == [(None, "y"), (1, "x"), (2, "y")]

    def test_tab_separated(self, spark, tmp_path):
        (tmp_path / "data.tsv").write_text("a\tb\n1\t2\n", newline="", encoding="utf-8")
        path = (tmp_path / "data.tsv").as_posix()
        if USE_ACTUAL_SPARK:
            # Spark has no tsv format.
            with pytest.raises(Exception):  # noqa: B017, PT011
                spark.read.load(path, format="tsv")
        else:
            assert _rows(spark.read.load(path, format="tsv")) == [("a", "b"), ("1", "2")]

    def test_options(self, spark, tmp_path):
        (tmp_path / "data.csv").write_text("a,b\n1,2\n", newline="", encoding="utf-8")
        path = (tmp_path / "data.csv").as_posix()
        if USE_ACTUAL_SPARK:
            assert _rows(spark.read.load(path, format="csv", header=True)) == [("1", "2")]
        else:
            with pytest.raises(ContributionsAcceptedError):
                spark.read.load(path, format="csv", header=True)


class TestFlags:
    @pytest.mark.parametrize("option", ["ignoreLeadingWhiteSpace", "ignoreTrailingWhiteSpace"])
    def test_whitespace_is_kept_unless_asked(self, spark, tmp_path, option):
        assert _rows(_read(spark, tmp_path, " a , b \n", **{option: "False"})) == [(" a ", " b ")]
        if not USE_ACTUAL_SPARK:
            with pytest.raises(ContributionsAcceptedError):
                _read(spark, tmp_path, " a \n", **{option: True})

    @pytest.mark.parametrize("value", [True, "TRUE", "true"])
    def test_multiline_spellings(self, spark, tmp_path, value):
        assert _rows(_read(spark, tmp_path, '"x\ny"\n', multiLine=value)) == [("x\ny",)]

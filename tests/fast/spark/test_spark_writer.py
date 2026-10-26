import datetime
import re

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException, IllegalArgumentException, error_condition

from duckdb.experimental.spark.exception import ContributionsAcceptedError

if USE_ACTUAL_SPARK:
    import pyspark

    SPARK_3 = pyspark.__version__.startswith("3.")
else:
    SPARK_3 = False


def _written(path):
    """The text of a CSV output, which Spark writes as a directory of part files."""
    if path.is_dir():
        return "".join(part.read_text(encoding="utf-8") for part in sorted(path.glob("part-*")))
    return path.read_text(encoding="utf-8")


def _one(df):
    return df.coalesce(1) if USE_ACTUAL_SPARK else df


def _offset(value):
    """The offset of the local time zone at ``value`` as Spark's XXX pattern letters write it, and as DuckDB does."""
    total = int(value.astimezone().utcoffset().total_seconds()) // 60
    hours, minutes = divmod(abs(total), 60)
    sign = "+" if total >= 0 else "-"
    spark_offset = "Z" if total == 0 else f"{sign}{hours:02d}:{minutes:02d}"
    return spark_offset if USE_ACTUAL_SPARK else f"{sign}{hours:02d}" + (f":{minutes:02d}" if minutes else "")


@pytest.fixture(params=["csv", "parquet"])
def kind(request):
    return request.param


def _write(df, kind, path, **options):
    getattr(df.write, kind)(path.as_posix(), **options)


def _read_back(spark, kind, path):
    return sorted(tuple(row) for row in getattr(spark.read, kind)(path.as_posix()).collect())


class TestSaveModes:
    @pytest.mark.parametrize("mode", [None, "error", "errorifexists", "default", "ERROR"])
    def test_existing_path_fails_the_write(self, spark, tmp_path, kind, mode):
        path = tmp_path / "out"
        _write(spark.createDataFrame([(1,)], "a int"), kind, path)
        with pytest.raises(AnalysisException, match=r'already exists\. Set mode as "overwrite"') as error:
            _write(spark.createDataFrame([(2,)], "a int"), kind, path, mode=mode)
        assert error_condition(error.value) == "PATH_ALREADY_EXISTS"
        assert _read_back(spark, kind, path) == [("1",) if kind == "csv" else (1,)]

    def test_existing_file_fails_the_write(self, spark, tmp_path, kind):
        path = tmp_path / "out"
        path.write_text("x\n", newline="", encoding="utf-8")
        with pytest.raises(AnalysisException, match=re.escape(f"Path file:{path} already exists")) as error:
            _write(spark.createDataFrame([(1,)], "a int"), kind, path)
        assert error_condition(error.value) == "PATH_ALREADY_EXISTS"

    @pytest.mark.parametrize(("mode", "first"), [("ignore", True), ("Ignore", True), ("overwrite", False)])
    def test_existing_path_kept_or_replaced(self, spark, tmp_path, kind, mode, first):
        path = tmp_path / "out"
        _write(spark.createDataFrame([(1,)], "a int"), kind, path)
        _write(spark.createDataFrame([(2,)], "a int"), kind, path, mode=mode)
        value = 1 if first else 2
        assert _read_back(spark, kind, path) == [(str(value),) if kind == "csv" else (value,)]

    @pytest.mark.parametrize("mode", ["append", "ignore", "overwrite", "error"])
    def test_new_path(self, spark, tmp_path, kind, mode):
        path = tmp_path / "out"
        _write(spark.createDataFrame([(1,)], "a int"), kind, path, mode=mode)
        assert _read_back(spark, kind, path) == [("1",) if kind == "csv" else (1,)]

    def test_append_to_an_existing_path(self, spark, tmp_path, kind):
        path = tmp_path / "out"
        _write(spark.createDataFrame([(1,)], "a int"), kind, path)
        if USE_ACTUAL_SPARK:
            _write(spark.createDataFrame([(2,)], "a int"), kind, path, mode="append")
            assert _read_back(spark, kind, path) == ([("1",), ("2",)] if kind == "csv" else [(1,), (2,)])
        else:
            # DuckDB writes one file, which it cannot add to.
            with pytest.raises(ContributionsAcceptedError):
                _write(spark.createDataFrame([(2,)], "a int"), kind, path, mode="append")

    def test_file_url(self, spark, tmp_path, kind):
        url = (tmp_path / "out").as_uri()
        _write(spark.createDataFrame([(1,)], "a int"), kind, tmp_path / "out")
        with pytest.raises(AnalysisException) as error:
            getattr(spark.createDataFrame([(2,)], "a int").write, kind)(url)
        assert error_condition(error.value) == "PATH_ALREADY_EXISTS"
        getattr(spark.createDataFrame([(2,)], "a int").write, kind)(url, mode="ignore")
        assert _read_back(spark, kind, tmp_path / "out") == [("1",) if kind == "csv" else (1,)]

    def test_invalid_mode(self, spark, tmp_path, kind):
        # Spark 3.5 has no error class for it.
        expected, condition = (IllegalArgumentException, None) if SPARK_3 else (AnalysisException, "INVALID_SAVE_MODE")
        with pytest.raises(expected, match="bogus") as error:
            _write(spark.createDataFrame([(1,)], "a int"), kind, tmp_path / "out", mode="bogus")
        assert error_condition(error.value) == condition


class TestRepeatedColumnNames:
    @pytest.mark.parametrize("names", [["x", "x"], ["x", "X"]])
    def test_repeated_names_fail_the_write(self, spark, tmp_path, kind, names):
        df = spark.createDataFrame([(1, 2)], names)
        path = tmp_path / "out"
        if USE_ACTUAL_SPARK and kind == "csv" and not SPARK_3:
            # Spark 4 writes the repeated names into the header line.
            _write(df, kind, path, header=True)
            assert _written(path).splitlines()[0] == ",".join(names)
            return
        with pytest.raises(AnalysisException) as error:
            _write(df, kind, path)
        assert error_condition(error.value) == "COLUMN_ALREADY_EXISTS"
        assert dict(error.value.getMessageParameters()) == {"columnName": "`x`"}
        assert not path.exists()

    @pytest.mark.parametrize("names", [["x", "x"], ["x", "X"]])
    def test_repeated_names_fail_to_save_a_table(self, spark, names):
        with pytest.raises(AnalysisException) as error:
            spark.createDataFrame([(1, 2)], names).write.saveAsTable("repeated_names_table")
        assert error_condition(error.value) == "COLUMN_ALREADY_EXISTS"


class TestCsvOutput:
    def test_quotes_and_backslashes(self, spark, tmp_path):
        values = ["C:\\temp", "a,b\\c", 'q"\\', "x\\", "", None]
        path = tmp_path / "out"
        _one(spark.createDataFrame([(v,) for v in values], "s string")).write.csv(path.as_posix())
        if USE_ACTUAL_SPARK:
            # Spark escapes a quote and a backslash inside quotes with a backslash.
            assert _written(path) == 'C:\\temp\n"a,b\\\\c"\n"q\\"\\\\"\nx\\\n""\n'
        else:
            # DuckDB doubles a quote inside quotes, and writes an empty line for a row that is NULL in its only column.
            assert _written(path) == 'C:\\temp\n"a,b\\c"\n"q""\\"\nx\\\n""\n\n'
        rows = [row[0] for row in spark.read.csv(path.as_posix()).collect()]
        assert sorted(row for row in rows if row is not None) == sorted(v for v in values if v)
        # The empty string is NULL, and DuckDB reads its empty line as a NULL row of the one column.
        assert rows.count(None) == (1 if USE_ACTUAL_SPARK else 2)

    def test_whitespace(self, spark, tmp_path):
        df = _one(spark.createDataFrame([(" a ",), ("\tb\x01",), ("\xa0c\xa0",)], "s string"))
        path = tmp_path / "kept"
        df.write.csv(path.as_posix(), ignoreLeadingWhiteSpace=False, ignoreTrailingWhiteSpace="FALSE")
        assert _written(path) == " a \n\tb\x01\n\xa0c\xa0\n"
        path = tmp_path / "default"
        df.write.csv(path.as_posix())
        # Spark trims strings when it writes them, DuckDB writes them as they are.
        assert _written(path) == ("a\nb\n\xa0c\xa0\n" if USE_ACTUAL_SPARK else " a \n\tb\x01\n\xa0c\xa0\n")
        if not USE_ACTUAL_SPARK:
            for option in ("ignoreLeadingWhiteSpace", "ignoreTrailingWhiteSpace"):
                with pytest.raises(ContributionsAcceptedError):
                    df.write.csv((tmp_path / option).as_posix(), **{option: "true"})

    def test_escape_quotes_spellings(self, spark, tmp_path):
        df = spark.createDataFrame([('a"b',)], "s string")
        for i, value in enumerate([True, "TRUE", "true"]):
            path = tmp_path / f"out{i}"
            df.write.csv(path.as_posix(), escapeQuotes=value)
            assert _written(path) == ('"a\\"b"\n' if USE_ACTUAL_SPARK else '"a""b"\n')

    def test_datetimes(self, spark, tmp_path):
        moment = datetime.datetime(2024, 2, 3, 4, 5, 6, 789000)
        schema = "t timestamp, n timestamp_ntz, d date"
        df = spark.createDataFrame([(moment, moment, datetime.date(2024, 2, 3))], schema)
        path = tmp_path / "default"
        df.write.csv(path.as_posix())
        if USE_ACTUAL_SPARK:
            expected = f"2024-02-03T04:05:06.789{_offset(moment)},2024-02-03T04:05:06.789,2024-02-03\n"
        else:
            # DuckDB writes its own text of a timestamp, which Spark's default pattern does not.
            expected = f"2024-02-03 04:05:06.789{_offset(moment)},2024-02-03 04:05:06.789,2024-02-03\n"
        assert _written(path) == expected
        path = tmp_path / "patterns"
        df.write.csv(path.as_posix(), timestampFormat="dd.MM.yyyy HH:mm", dateFormat="dd/MM/yyyy")
        # Spark's timestampFormat is for timestamps with a time zone only, DuckDB's for every timestamp.
        without_zone = "2024-02-03T04:05:06.789" if USE_ACTUAL_SPARK else "03.02.2024 04:05"
        assert _written(path) == f"03.02.2024 04:05,{without_zone},03/02/2024\n"

    def test_offset_pattern(self, spark, tmp_path):
        moment = datetime.datetime(2024, 7, 1, 12)
        path = tmp_path / "out"
        df = spark.createDataFrame([(moment,)], "t timestamp")
        if USE_ACTUAL_SPARK:
            df.write.csv(path.as_posix(), timestampFormat="HH:mmXXX")
            assert _written(path) == f"12:00{_offset(moment)}\n"
        else:
            with pytest.raises(ContributionsAcceptedError):
                df.write.csv(path.as_posix(), timestampFormat="HH:mmXXX")

    @pytest.mark.parametrize(
        ("schema", "value", "name"),
        [("a array<int>", [1], "ARRAY<INT>"), ("a struct<x:int>", (1,), "STRUCT<x: INT>"), ("a void", None, "VOID")],
    )
    def test_types_csv_cannot_hold(self, spark, tmp_path, schema, value, name):
        df = spark.createDataFrame([(value,)], schema)
        with pytest.raises(AnalysisException, match=f'column `a` of the type "{name}"') as error:
            df.write.csv((tmp_path / "out").as_posix())
        assert error_condition(error.value) == "UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE"

    def test_save_mode_comes_before_the_types(self, spark, tmp_path):
        df = spark.createDataFrame([([1],)], "a array<int>")
        with pytest.raises(Exception, match="bogus"):
            df.write.csv((tmp_path / "new").as_posix(), mode="bogus")
        spark.createDataFrame([(1,)], "a int").write.csv((tmp_path / "out").as_posix())
        df.write.csv((tmp_path / "out").as_posix(), mode="ignore")
        assert _written(tmp_path / "out") == "1\n"

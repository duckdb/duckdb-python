import datetime
import math
from decimal import Decimal

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from conftest import ENGINE_GE_2_1
from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException, error_condition
from spark_namespace.sql.types import Row

import duckdb
from duckdb.experimental.spark.exception import ContributionsAcceptedError

if USE_ACTUAL_SPARK:
    import pyspark


class TestSparkReadJson:
    def test_read_json(self, duckdb_cursor, spark, tmp_path):
        file_path = tmp_path / "basic.parquet"
        file_path = file_path.as_posix()
        duckdb_cursor.execute(f"COPY (select 42 a, true b, 'this is a long string' c) to '{file_path}' (FORMAT JSON)")
        df = spark.read.json(file_path)
        res = df.collect()
        assert res == [Row(a=42, b=True, c="this is a long string")]


def _read(spark, tmp_path, *lines):
    path = tmp_path / "data.json"
    path.write_text("\n".join(lines) + "\n", newline="", encoding="utf-8")
    return spark.read.json(path.as_posix())


class TestJsonInference:
    def test_strings_stay_strings(self, spark, tmp_path):
        df = _read(
            spark,
            tmp_path,
            '{"d": "2024-01-01", "t": "12:00:00", "ts": "2024-01-01 12:00:00", "z": "2024-01-01T12:00:00Z"}',
            '{"u": "123e4567-e89b-12d3-a456-426614174000", "n": "123"}',
        )
        assert df.dtypes == [(name, "string") for name in ["d", "n", "t", "ts", "u", "z"]]
        assert [tuple(row) for row in df.collect()] == [
            ("2024-01-01", None, "12:00:00", "2024-01-01 12:00:00", None, "2024-01-01T12:00:00Z"),
            (None, "123", None, None, "123e4567-e89b-12d3-a456-426614174000", None),
        ]

    def test_fields_sorted_by_name(self, spark, tmp_path):
        df = _read(spark, tmp_path, '{"z": "x", "a": 1, "C": true}', '{"b": 1.5}')
        assert df.dtypes == [("C", "boolean"), ("a", "bigint"), ("b", "double"), ("z", "string")]
        assert [tuple(row) for row in df.collect()] == [(True, 1, None, "x"), (None, None, 1.5, None)]

    def test_nested_fields_sorted_by_name(self, spark, tmp_path):
        df = _read(spark, tmp_path, '{"s": {"y": 1, "x": "2024-01-01"}, "l": [{"b": 1}, {"a": 2}]}')
        assert df.dtypes == [("l", "array<struct<a:bigint,b:bigint>>"), ("s", "struct<x:string,y:bigint>")]
        assert df.collect() == [Row(l=[Row(a=None, b=1), Row(a=2, b=None)], s=Row(x="2024-01-01", y=1))]

    def test_null_fields_are_strings(self, spark, tmp_path):
        df = _read(spark, tmp_path, '{"n": null, "l": [null], "s": {"m": null}}')
        assert df.dtypes == [("l", "array<string>"), ("n", "string"), ("s", "struct<m:string>")]

    def test_values_of_mixed_types_are_strings(self, spark, tmp_path):
        df = _read(spark, tmp_path, '{"v": 1, "l": [1, "x"]}', '{"v": "x", "l": [true]}')
        assert df.dtypes == [("l", "array<string>"), ("v", "string")]
        assert [tuple(row) for row in df.collect()] == [(["1", "x"], "1"), (["true"], "x")]

    def test_objects_with_many_keys_are_structs(self, spark, tmp_path):
        keys = [f"k{i:03d}" for i in range(250)]
        df = _read(spark, tmp_path, '{"m": {' + ", ".join(f'"{k}": 1' for k in reversed(keys)) + "}}")
        assert df.schema.fields[0].dataType.names == keys

    def test_integers_wider_than_a_long(self, spark, tmp_path):
        df = _read(spark, tmp_path, '{"v": 12345678901234567890}')
        if USE_ACTUAL_SPARK:
            assert df.dtypes == [("v", "decimal(20,0)")]
        else:
            # DuckDB reads them as HUGEINT, as it does in CSV.
            assert df.dtypes == [("v", "hugeint")]
        assert int(df.collect()[0].v) == 12345678901234567890


def _load(spark, tmp_path, schema, *lines):
    path = tmp_path / "data.json"
    path.write_text("\n".join(lines) + "\n", newline="", encoding="utf-8")
    return spark.read.load(path.as_posix(), format="json", schema=schema)


def _field(spark, tmp_path, data_type, *values):
    """Each JSON value read as ``data_type`` from a record of its own, in the order of the values."""
    lines = [f'{{"k": {i}, "v": {value}}}' for i, value in enumerate(values)]
    return [row.v for row in sorted(_load(spark, tmp_path, f"k int, v {data_type}", *lines).collect())]


class TestJsonWithASchema:
    """A field of a schema converts with DuckDB's JSON reader, NULL where a value does not convert, as in Spark."""

    def test_integers(self, spark, tmp_path):
        values = ["1", "-1", "null", '{"x": 1}', "[1]", "2147483648"]
        assert _field(spark, tmp_path, "int", *values) == [1, -1, None, None, None, None]
        # Spark reads an integer only from a JSON integer; DuckDB also from a fraction, a numeric string or a boolean.
        values = _field(spark, tmp_path, "int", "1.7", "1.0", "1e2", '"1"', "true")
        assert values == ([None] * 5 if USE_ACTUAL_SPARK else [2, 1, 100, 1, 1])
        assert _field(spark, tmp_path, "bigint", "2147483648", "9223372036854775808") == [2147483648, None]
        values = ["127", "128", "255", "256", "-128", "-129"]
        # Spark reads a tinyint from -128 to 255, the values above 127 as unsigned bytes.
        expected = [127, -128, -1, None, -128, None] if USE_ACTUAL_SPARK else [127, None, None, None, -128, None]
        assert _field(spark, tmp_path, "tinyint", *values) == expected
        assert _field(spark, tmp_path, "smallint", "32767", "32768", "65535") == [32767, None, None]

    def test_doubles(self, spark, tmp_path):
        assert _field(spark, tmp_path, "double", "1", "1.7", "1e2") == [1.0, 1.7, 100.0]
        assert _field(spark, tmp_path, "double", '"1.5"', "true") == ([None, None] if USE_ACTUAL_SPARK else [1.5, 1.0])
        texts = ["NaN", "+INF", "-INF", "Infinity", "+Infinity", "-Infinity"]
        values = _field(spark, tmp_path, "double", *(f'"{text}"' for text in texts))
        assert math.isnan(values[0])
        assert values[1:] == [math.inf, -math.inf, math.inf, math.inf, -math.inf]
        values = _field(spark, tmp_path, "double", '"INF"', '"Inf"', '"nan"')
        if USE_ACTUAL_SPARK:
            assert values == [None, None, None]
        else:
            # DuckDB's cast reads more spellings of NaN and infinity.
            assert values[:2] == [math.inf, math.inf]
            assert math.isnan(values[2])
        assert _field(spark, tmp_path, "float", "1.5", '"+INF"') == [1.5, math.inf]

    def test_decimals(self, spark, tmp_path):
        values = ["1", "1.7", '"1"', '"1,000"', '" 1"', '"abc"', "9223372036854775808"]
        # Spark leaves out the commas its default locale groups digits with, and DuckDB trims a numeric string.
        thousand, one = (Decimal("1000.00"), None) if USE_ACTUAL_SPARK else (None, Decimal("1.00"))
        expected = [Decimal("1.00"), Decimal("1.70"), Decimal("1.00"), thousand, one, None, None]
        assert _field(spark, tmp_path, "decimal(10,2)", *values) == expected
        # Spark and the 2.1 engine round the decimal text, the 2.0 engine the double it reads.
        values = _field(spark, tmp_path, "decimal(10,2)", "1.005", "true")
        if USE_ACTUAL_SPARK:
            assert values == [Decimal("1.01"), None]
        else:
            assert values == [Decimal("1.01") if ENGINE_GE_2_1 else Decimal("1.00"), Decimal("1.00")]

    def test_booleans(self, spark, tmp_path):
        assert _field(spark, tmp_path, "boolean", "true", "false") == [True, False]
        values = _field(spark, tmp_path, "boolean", '"true"', "1")
        assert values == ([None, None] if USE_ACTUAL_SPARK else [True, True])

    def test_strings(self, spark, tmp_path):
        values = ['"x"', "null", "true", "1", "[1]", '{"x":1}']
        assert _field(spark, tmp_path, "string", *values) == ["x", None, "true", "1", "[1]", '{"x":1}']

    def test_text_of_numbers_and_objects(self, spark, tmp_path):
        values = _field(spark, tmp_path, "string", "1.50", "1E+2", '{"b": 1, "a": [1, 2.50]}')
        if USE_ACTUAL_SPARK and not pyspark.__version__.startswith("3."):
            # Spark 4 keeps the text as written, Spark 3.5 writes the value again as DuckDB does.
            assert values == ["1.50", "1E+2", '{"b": 1, "a": [1, 2.50]}']
        else:
            assert values == ["1.5", "100.0", '{"b":1,"a":[1,2.5]}']

    def test_dates(self, spark, tmp_path):
        values = ['"2024-01-02"', '"2024-1-2"', '"2024-01-02T10:00:00"', "1"]
        day = datetime.date(2024, 1, 2)
        assert _field(spark, tmp_path, "date", *values) == [day, day, day, None]
        # Spark reads digits as days since the epoch, as Spark 1.5 wrote dates; DuckDB reads slashes.
        values = _field(spark, tmp_path, "date", '"2024/01/02"', '"1"')
        assert values == ([None, datetime.date(1970, 1, 2)] if USE_ACTUAL_SPARK else [day, None])

    def test_timestamps(self, spark, tmp_path):
        values = ['"2024-01-02 10:00:00"', '"2024-01-02"', "1.7", '"abc"', "true"]
        expected = [datetime.datetime(2024, 1, 2, 10), datetime.datetime(2024, 1, 2), None, None, None]
        assert _field(spark, tmp_path, "timestamp", *values) == expected
        # Spark reads an integer as seconds since the epoch.
        # A negative epoch second cannot be converted on Windows, so the expectation is built only where it is used.
        expected = (
            [datetime.datetime.fromtimestamp(1), datetime.datetime.fromtimestamp(-1)]
            if USE_ACTUAL_SPARK
            else [None, None]
        )
        assert _field(spark, tmp_path, "timestamp", "1", "-1") == expected
        instant = datetime.datetime(2024, 1, 2, 10, tzinfo=datetime.timezone.utc).astimezone().replace(tzinfo=None)
        assert _field(spark, tmp_path, "timestamp", '"2024-01-02T10:00:00Z"') == [instant]
        rows = _field(spark, tmp_path, "struct<t:timestamp>", '{"t": "2024-01-02 10:00:00"}')
        assert [row.t for row in rows] == [datetime.datetime(2024, 1, 2, 10)]

    def test_timestamps_without_a_zone(self, spark, tmp_path):
        values = _field(spark, tmp_path, "timestamp_ntz", '"2024-01-02 12:00:00"', '"2024-01-02T12:00:00Z"', "1")
        # Spark rejects text with a zone, DuckDB drops the zone.
        noon = datetime.datetime(2024, 1, 2, 12)
        assert values == ([noon, None, None] if USE_ACTUAL_SPARK else [noon, noon, None])

    def test_arrays(self, spark, tmp_path):
        values = ["[1, null, 2]", "[]", "null", '{"a": 1}']
        assert _field(spark, tmp_path, "array<int>", *values) == [[1, None, 2], [], None, None]
        # Spark makes the array NULL when an element does not convert, DuckDB the element.
        values = _field(spark, tmp_path, "array<int>", "[1.5]", '[1, "x"]')
        assert values == ([None, None] if USE_ACTUAL_SPARK else [[2], [1, None]])

    def test_structs(self, spark, tmp_path):
        # A field that does not convert is NULL, and fields are found by their name in its case.
        values = ['{"a": "x", "b": 2}', "{}", '{"A": 1, "b": 2}']
        rows = _field(spark, tmp_path, "struct<a:int,b:int>", *values)
        assert [tuple(row) for row in rows] == [(None, 2), (None, None), (None, 2)]
        rows = _field(spark, tmp_path, "array<struct<a:int,b:int>>", '[{"a": "x", "b": 1}]')
        assert [[tuple(item) for item in row] for row in rows] == [[(None, 1)]]
        # Spark reads a value that is not an object as NULL, DuckDB as a struct of NULL fields.
        rows = _field(spark, tmp_path, "struct<a:int,b:int>", "1", "[1]")
        expected = [None, None] if USE_ACTUAL_SPARK else [(None, None), (None, None)]
        assert [None if row is None else tuple(row) for row in rows] == expected

    def test_maps(self, spark, tmp_path):
        values = ['{"a": 1}', '{"a": null}', "[1]"]
        assert _field(spark, tmp_path, "map<string,int>", *values) == [{"a": 1}, {"a": None}, None]
        # Spark makes the map NULL when a value does not convert, DuckDB the value.
        expected = None if USE_ACTUAL_SPARK else {"a": 1, "b": None}
        assert _field(spark, tmp_path, "map<string,int>", '{"a": 1, "b": "x"}') == [expected]

    def test_names_with_a_slash_or_a_tilde(self, spark, tmp_path):
        df = _load(spark, tmp_path, "s struct<`a/b`:int,`c~d`:int>, `e.f` int", '{"s": {"a/b": 1, "c~d": 2}, "e.f": 3}')
        assert [(tuple(row.s), row[1]) for row in df.collect()] == [((1, 2), 3)]

    def test_names_are_case_sensitive(self, spark, tmp_path):
        df = _load(spark, tmp_path, "a int, s struct<q:int>", '{"A": 1, "s": {"Q": 1}}')
        assert [(row.a, tuple(row.s)) for row in df.collect()] == [(None, (None,))]

    def test_names_differing_in_case_only(self, spark, tmp_path):
        with pytest.raises(AnalysisException, match="The column `a` already exists") as error:
            _load(spark, tmp_path, "a int, A int", '{"a": 1}').collect()
        assert error_condition(error.value) == "COLUMN_ALREADY_EXISTS"

    def test_schema_of_one_field(self, spark, tmp_path):
        rows = _load(spark, tmp_path, "a int", '{"a": 1}', '{"a": 2}', "7").collect()
        assert rows == [Row(a=1), Row(a=2), Row(a=None)]
        lines = [f'{{"k{i}": 1}}' for i in range(30)] + ['{"a": 9}']
        assert _load(spark, tmp_path, "a int", *lines).collect()[-1] == Row(a=9)

    def test_malformed_lines(self, spark, tmp_path):
        rows = _load(spark, tmp_path, "a int", '{"a": 1}', "{bad", '{"a": 3}').collect()
        assert rows == [Row(a=1), Row(a=None), Row(a=3)]

    def test_binary(self, spark, tmp_path):
        # Spark reads base64 text as binary, DuckDB the text's bytes.
        expected = None if USE_ACTUAL_SPARK else b"abc"
        assert _field(spark, tmp_path, "binary", '"abc"') == [expected]

    def test_json_by_its_extension(self, spark, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"b": 1, "a": "x"}\n', newline="", encoding="utf-8")
        if USE_ACTUAL_SPARK:
            # Spark reads a path without a format as parquet.
            with pytest.raises(Exception):  # noqa: B017, PT011
                spark.read.load(path.as_posix()).collect()
        else:
            assert spark.read.load(path.as_posix()).collect() == [Row(a="x", b=1)]
            assert spark.read.load(path.as_posix(), schema="b int").collect() == [Row(b=1)]


class TestJsonInferenceOfKeys:
    def test_keys_differing_in_case_only(self, spark, tmp_path):
        for lines in (['{"a": 1, "A": 2}'], ['{"a": 1}', '{"A": "x"}']):
            expected = AnalysisException if USE_ACTUAL_SPARK else duckdb.NotImplementedException
            with pytest.raises(expected):
                _read(spark, tmp_path, *lines).collect()

    def test_keys_of_one_type_differing_in_case_only(self, spark, tmp_path):
        if USE_ACTUAL_SPARK:
            # Spark keeps the first name, and the field of the other name reads as NULL.
            df = _read(spark, tmp_path, '{"a": 1}', '{"A": 2}')
            assert [tuple(row) for row in df.collect()] == [(1,), (None,)]
        else:
            with pytest.raises(duckdb.NotImplementedException):
                _read(spark, tmp_path, '{"a": 1}', '{"A": 2}')

    def test_empty_file(self, spark, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("", newline="", encoding="utf-8")
        assert spark.read.json(path.as_posix()).columns == []

    def test_lines_that_are_not_objects(self, spark, tmp_path):
        if USE_ACTUAL_SPARK:
            df = _read(spark, tmp_path, '{"a": 1}', "[1,2]")
            assert df.columns == ["_corrupt_record", "a"]
        else:
            with pytest.raises(ContributionsAcceptedError):
                _read(spark, tmp_path, '{"a": 1}', "[1,2]")

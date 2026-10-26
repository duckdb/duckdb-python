import datetime
import decimal
import math
import sys
import time
from typing import NamedTuple

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import ParseException, PySparkTypeError, PySparkValueError, error_condition
from spark_namespace.sql import functions as sf
from spark_namespace.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DayTimeIntervalType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    Row,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)

import duckdb
from duckdb.experimental.spark.exception import ContributionsAcceptedError

if USE_ACTUAL_SPARK:
    import pyspark


def struct_of(*fields):
    return StructType([StructField(name, data_type, True) for name, data_type in fields])


def value_of(data_type):
    return struct_of(("value", data_type))


class TestSchemaString:
    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            ("a int, b string", struct_of(("a", IntegerType()), ("b", StringType()))),
            ("a: int, b: string", struct_of(("a", IntegerType()), ("b", StringType()))),
            ("a INT, b STRING", struct_of(("a", IntegerType()), ("b", StringType()))),
            ("A Int, B String", struct_of(("A", IntegerType()), ("B", StringType()))),
            ("  a   int  ,   b string  ", struct_of(("a", IntegerType()), ("b", StringType()))),
            ("a: byte, b: decimal(  16 , 8   ) ", struct_of(("a", ByteType()), ("b", DecimalType(16, 8)))),
            ("struct<a:int,b:string>", struct_of(("a", IntegerType()), ("b", StringType()))),
            ("STRUCT<a: INT, b: STRING>", struct_of(("a", IntegerType()), ("b", StringType()))),
            ("struct<a int, b string>", struct_of(("a", IntegerType()), ("b", StringType()))),
            ("`a b` int, `c``d` string", struct_of(("a b", IntegerType()), ("c`d", StringType()))),
            ("a int, a string", struct_of(("a", IntegerType()), ("a", StringType()))),
            ("string int", struct_of(("string", IntegerType()))),
            ("interval: int, b: string", struct_of(("interval", IntegerType()), ("b", StringType()))),
            (
                "a bigint, b double, c boolean, d date, e timestamp, f binary",
                struct_of(
                    ("a", LongType()),
                    ("b", DoubleType()),
                    ("c", BooleanType()),
                    ("d", DateType()),
                    ("e", TimestampType()),
                    ("f", BinaryType()),
                ),
            ),
            ("STRING", value_of(StringType())),
            ("int", value_of(IntegerType())),
            ("INT ", value_of(IntegerType())),
            ("integer", value_of(IntegerType())),
            ("bigint", value_of(LongType())),
            ("long", value_of(LongType())),
            ("tinyint", value_of(ByteType())),
            ("byte", value_of(ByteType())),
            ("smallint", value_of(ShortType())),
            ("short", value_of(ShortType())),
            ("float", value_of(FloatType())),
            ("real", value_of(FloatType())),
            ("double", value_of(DoubleType())),
            ("boolean", value_of(BooleanType())),
            ("date", value_of(DateType())),
            ("timestamp", value_of(TimestampType())),
            ("timestamp_ltz", value_of(TimestampType())),
            ("timestamp_ntz", value_of(TimestampNTZType())),
            ("binary", value_of(BinaryType())),
            ("void", value_of(NullType())),
            ("decimal", value_of(DecimalType(10, 0))),
            ("decimal(10,2)", value_of(DecimalType(10, 2))),
            ("DECIMAL( 16 , 8 )", value_of(DecimalType(16, 8))),
            ("decimal(5)", value_of(DecimalType(5, 0))),
            ("dec(5,1)", value_of(DecimalType(5, 1))),
            ("numeric(7,3)", value_of(DecimalType(7, 3))),
            ("interval day to second", value_of(DayTimeIntervalType(0, 3))),
            ("array<int>", value_of(ArrayType(IntegerType()))),
            ("ARRAY<STRING>", value_of(ArrayType(StringType()))),
            ("array< short>", value_of(ArrayType(ShortType()))),
            ("map<string,int>", value_of(MapType(StringType(), IntegerType()))),
            (" map<string , string > ", value_of(MapType(StringType(), StringType()))),
            (
                "a array<struct<x:int,y:string>>",
                struct_of(("a", ArrayType(struct_of(("x", IntegerType()), ("y", StringType()))))),
            ),
            (
                "a map<string,array<int>>, b struct<c:struct<d:decimal(5,2)>>",
                struct_of(
                    ("a", MapType(StringType(), ArrayType(IntegerType()))),
                    ("b", struct_of(("c", struct_of(("d", DecimalType(5, 2)))))),
                ),
            ),
        ],
    )
    def test_schema(self, spark, schema, expected):
        df = spark.createDataFrame([], schema)
        assert df.schema == expected
        assert df.columns == expected.names
        assert df.collect() == []

    @pytest.mark.parametrize(
        "schema",
        [
            "blabla",
            "bool",
            "int a",
            "a: int,",
            "a int b int",
            "a.b int",
            "array<int",
            "array<int,int>",
            "map<int>",
            "map<int, boolean>>",
            "struct<a:int",
            "decimal(40,2)",
            "decimal(5,6)",
        ],
    )
    def test_invalid_schema(self, spark, schema):
        with pytest.raises(ParseException):
            spark.createDataFrame([], schema)

    @pytest.mark.parametrize(
        ("schema", "spark_error", "module_error"),
        [
            ("a int b string", "'b'", "'b'"),
            ("a int,", "end of input", "end of input"),
            ("int,", "','", "','"),
            ("array<int", "'<'", "end of input"),
            ("struct<a:int", "'<'", "end of input"),
            ("map<int>", "'<'", "'>'"),
            ("decimal(10", "'('", "end of input"),
            ("a: int, b", "':'", "end of input"),
            ("interval day to year", "'to'", "'to'"),
            ("interval second to day", "'to'", "'to'"),
            ("interval minute to hour", "'to'", "'to'"),
            ("interval day to day", "'to'", "'to'"),
        ],
    )
    def test_syntax_error_position(self, spark, schema, spark_error, module_error):
        # Spark reports the error of reading the text as a list of fields, even when it then reads it as a type; the
        # module reads the text once, as a type or a list of fields by its first tokens.
        with pytest.raises(ParseException) as raised:
            spark.createDataFrame([], schema)
        assert error_condition(raised.value) == "PARSE_SYNTAX_ERROR"
        error = spark_error if USE_ACTUAL_SPARK else module_error
        assert dict(raised.value.getMessageParameters()) == {"error": error, "hint": ""}
        assert raised.value.getSqlState() == "42601"

    def test_unknown_type_name(self, spark):
        with pytest.raises(ParseException) as raised:
            spark.createDataFrame([], "a int, b strin")
        assert error_condition(raised.value) == "UNSUPPORTED_DATATYPE"
        assert dict(raised.value.getMessageParameters()) == {"typeName": '"STRIN"'}
        assert raised.value.getSqlState() == "0A000"

    @pytest.mark.parametrize("schema", ["a strin b", "a strin, b", "a int, b strin c"])
    def test_unknown_type_name_raises_before_a_later_syntax_error(self, spark, schema):
        with pytest.raises(ParseException) as raised:
            spark.createDataFrame([], schema)
        # Spark checks type names once the text parses; the module raises at the first one it does not know.
        assert error_condition(raised.value) == ("PARSE_SYNTAX_ERROR" if USE_ACTUAL_SPARK else "UNSUPPORTED_DATATYPE")

    def test_char_type_raises_before_a_later_syntax_error(self, spark):
        if USE_ACTUAL_SPARK:
            with pytest.raises(ParseException):
                spark.createDataFrame([], "a varchar(3) b")
        else:
            with pytest.raises(ContributionsAcceptedError, match="varchar"):
                spark.createDataFrame([], "a varchar(3) b")

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("interval day", DayTimeIntervalType(0, 0)),
            ("interval hour", DayTimeIntervalType(1, 1)),
            ("interval hour to minute", DayTimeIntervalType(1, 2)),
            ("interval minute to second", DayTimeIntervalType(2, 3)),
            ("interval day to second", DayTimeIntervalType(0, 3)),
        ],
    )
    def test_interval_type_names(self, name, expected):
        assert StructType().add("a", name).fields[0].dataType == expected

    def test_variant(self, spark):
        if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3."):
            with pytest.raises(ParseException):
                spark.createDataFrame([], "a variant")
        else:
            assert spark.createDataFrame([], "a variant").dtypes == [("a", "variant")]

    @pytest.mark.parametrize("schema", ["s struct<a:int, a:string>", "s struct<a:int, A:string>"])
    def test_struct_fields_named_alike(self, spark, schema):
        if USE_ACTUAL_SPARK:
            row = spark.createDataFrame([((1, "x"),)], schema).collect()[0]
            assert tuple(row.s) == (1, "x")
        else:
            from duckdb.experimental.spark.errors import PySparkNotImplementedError

            # DuckDB matches struct field names case insensitively, so a struct cannot hold both.
            with pytest.raises(PySparkNotImplementedError):
                spark.createDataFrame([((1, "x"),)], schema)

    @pytest.mark.parametrize(
        ("schema", "spark_schema"),
        [
            ("interval day", value_of(DayTimeIntervalType(0, 0))),
            ("a interval hour to minute", struct_of(("a", DayTimeIntervalType(1, 2)))),
            ("", StructType([])),
            ("struct<>", StructType([])),
        ],
    )
    def test_schema_not_supported(self, spark, schema, spark_schema):
        if USE_ACTUAL_SPARK:
            assert spark.createDataFrame([], schema).schema == spark_schema
        else:
            with pytest.raises(ContributionsAcceptedError):
                spark.createDataFrame([], schema)

    @pytest.mark.parametrize(
        ("schema", "nullable", "metadata"),
        [
            ("a int not null", False, {}),
            ("a INT NOT NULL", False, {}),
            ("a: int not null", False, {}),
            ("struct<a:int not null>", False, {}),
            ("a int comment 'hello'", True, {"comment": "hello"}),
            ('a int not null comment "it\'s"', False, {"comment": "it's"}),
        ],
    )
    def test_not_null_and_comment(self, spark, schema, nullable, metadata):
        df = spark.createDataFrame([(1,)], schema)
        assert df.collect() == [Row(a=1)]
        field = df.schema["a"]
        assert field.dataType == IntegerType()
        if USE_ACTUAL_SPARK:
            assert (field.nullable, field.metadata) == (nullable, metadata)
        else:
            # The module reports every field as nullable and keeps no metadata.
            assert (field.nullable, field.metadata) == (True, {})
        if not nullable:
            with pytest.raises(PySparkValueError):
                spark.createDataFrame([(None,)], schema)

    def test_duckdb_only_types(self, spark):
        schema = "a hugeint, b uuid, c ubigint, d utinyint"
        if USE_ACTUAL_SPARK:
            with pytest.raises(ParseException):
                spark.createDataFrame([], schema)
        else:
            from duckdb.experimental.spark.sql.types import (
                HugeIntegerType,
                UnsignedByteType,
                UnsignedLongType,
                UUIDType,
            )

            df = spark.createDataFrame([(2**100, None, 2**63, 255)], schema)
            assert df.schema == struct_of(
                ("a", HugeIntegerType()), ("b", UUIDType()), ("c", UnsignedLongType()), ("d", UnsignedByteType())
            )
            assert df.collect() == [Row(a=2**100, b=None, c=2**63, d=255)]

    @pytest.mark.parametrize("schema", ["a char(5)", "a varchar(5)"])
    def test_char_types_raise(self, spark, schema):
        with pytest.raises(Exception):  # noqa: B017, PT011
            spark.createDataFrame([("x",)], schema).collect()

    @pytest.mark.parametrize(
        ("schema", "data", "expected"),
        [
            ("a int, b string", [(1, "x"), (2, None)], [Row(a=1, b="x"), Row(a=2, b=None)]),
            ("a: long, b: double", [(1, 1.5)], [Row(a=1, b=1.5)]),
            ("a boolean, b float, c tinyint, d smallint", [(True, 1.5, 1, 2)], [Row(a=True, b=1.5, c=1, d=2)]),
            ("struct<x:int,y:string>", [(1, "a")], [Row(x=1, y="a")]),
            ("a decimal(10,2)", [(decimal.Decimal("1.25"),)], [Row(a=decimal.Decimal("1.25"))]),
            (
                "a date, b timestamp, c timestamp_ntz",
                [(datetime.date(2024, 1, 2), datetime.datetime(2024, 1, 2, 3, 4, 5), datetime.datetime(2024, 1, 2))],
                [
                    Row(
                        a=datetime.date(2024, 1, 2),
                        b=datetime.datetime(2024, 1, 2, 3, 4, 5),
                        c=datetime.datetime(2024, 1, 2),
                    )
                ],
            ),
            ("string", ["a", None], [Row(value="a"), Row(value=None)]),
            ("int", [1, 2], [Row(value=1), Row(value=2)]),
            ("array<int>", [[1, 2], None, []], [Row(value=[1, 2]), Row(value=None), Row(value=[])]),
            ("map<string,int>", [{"a": 1}], [Row(value={"a": 1})]),
            (
                "a array<int>, b map<string,bigint>, c struct<x:int,y:string>",
                [([1, None], {"k": 2}, (3, "s"))],
                [Row(a=[1, None], b={"k": 2}, c=Row(x=3, y="s"))],
            ),
            ("a int, b string", [{"b": "x", "a": 1}], [Row(a=1, b="x")]),
            ("a int, b string", [Row(a=1, b="x")], [Row(a=1, b="x")]),
        ],
    )
    def test_schema_with_data(self, spark, schema, data, expected):
        df = spark.createDataFrame(data, schema)
        res = df.collect()
        assert res == expected
        assert [r.asDict(recursive=True) for r in res] == [r.asDict(recursive=True) for r in expected]

    def test_binary_with_schema(self, spark):
        res = spark.createDataFrame([(b"ab",)], "a binary").collect()
        assert bytes(res[0].a) == b"ab"

    def test_bigint_schema_does_not_overflow(self, spark):
        df = spark.createDataFrame([2_000_000_000], "bigint")
        assert df.select((sf.col("value") * 2).alias("v")).collect() == [Row(v=4_000_000_000)]

    @pytest.mark.parametrize(
        ("schema", "data"),
        [
            ("a int", [("x",)]),
            ("a int", [(True,)]),
            ("a double", [(1,)]),
            ("a decimal(5,2)", [(1.5,)]),
            ("a timestamp", [(datetime.date(2024, 1, 2),)]),
            ("a array<int>", [([1, "x"],)]),
            ("a map<string,int>", [([1],)]),
            ("a struct<x:int>", [(5,)]),
            ("int", ["x"]),
        ],
    )
    def test_verify_type(self, spark, schema, data):
        with pytest.raises(PySparkTypeError):
            spark.createDataFrame(data, schema)

    @pytest.mark.parametrize(
        ("schema", "data"),
        [
            ("a tinyint", [(128,)]),
            ("a int", [(2**31,)]),
            ("a map<string,int>", [({None: 1},)]),
            ("a int not null, b int", [{"b": 1}]),
            (StructType([StructField("a", IntegerType(), False)]), [(None,)]),
        ],
    )
    def test_verify_value(self, spark, schema, data):
        with pytest.raises(PySparkValueError):
            spark.createDataFrame(data, schema)

    @pytest.mark.parametrize(
        ("schema", "data", "expected"),
        [
            ("a string", [(1,)], [Row(a="1")]),
            ("a date", [(datetime.datetime(2024, 1, 2, 3),)], [Row(a=datetime.date(2024, 1, 2))]),
            ("a array<int>", [((1, 2),)], [Row(a=[1, 2])]),
            ("int", [None], [Row(value=None)]),
        ],
    )
    def test_verify_accepts(self, spark, schema, data, expected):
        assert spark.createDataFrame(data, schema).collect() == expected

    def test_verify_schema_false(self, spark):
        # Spark skips the check and stores what the JVM makes of the value; the module casts it.
        spark.createDataFrame([(1,)], "a double", verifySchema=False).collect()
        with pytest.raises(PySparkTypeError):
            spark.createDataFrame([(1,)], "a double")

    def test_nan_in_nested_values(self, spark):
        nan = float("nan")
        df = spark.createDataFrame(
            [([1.0, nan], {"k": nan}, (nan,), nan)],
            "a array<double>, b map<string,double>, c struct<x:double>, d double",
        )
        row = df.collect()[0]
        assert row.a[0] == 1.0
        assert math.isnan(row.a[1])
        assert math.isnan(row.b["k"])
        assert math.isnan(row.c.x)
        assert math.isnan(row.d)

    def test_nan_in_inferred_nested_values(self, spark):
        nan = float("nan")
        row = spark.createDataFrame([([nan], {"k": nan}, Row(x=nan))]).collect()[0]
        assert math.isnan(row._1[0])
        assert math.isnan(row._2["k"])
        assert math.isnan(row._3.x)

    def test_row_shorter_than_schema(self, spark):
        with pytest.raises(PySparkValueError):
            spark.createDataFrame([(1,)], "a int, b int").collect()

    def test_struct_type_built_with_add(self, spark):
        schema = StructType().add("a", IntegerType()).add(StructField("b", StringType()))
        df = spark.createDataFrame([(1, "x")], schema)
        assert df.schema == struct_of(("a", IntegerType()), ("b", StringType()))
        assert df.collect() == [Row(a=1, b="x")]

    def test_pandas_with_schema_string(self, spark):
        pd = pytest.importorskip("pandas")
        df = spark.createDataFrame(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), "a int, b string")
        assert df.schema == struct_of(("a", IntegerType()), ("b", StringType()))
        assert df.collect() == [Row(a=1, b="x"), Row(a=2, b="y")]


aware = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))


class Point(NamedTuple):
    x: int
    y: str


class Person:
    def __init__(self) -> None:
        self.name = "Alice"
        self.age = 1


class TestInference:
    @pytest.mark.parametrize(
        ("value", "data_type", "collected"),
        [
            (1, LongType(), 1),
            (2_000_000_000, LongType(), 2_000_000_000),
            (1.5, DoubleType(), 1.5),
            (True, BooleanType(), True),
            ("s", StringType(), "s"),
            (decimal.Decimal("1.5"), DecimalType(38, 18), decimal.Decimal("1.500000000000000000")),
            (datetime.date(2024, 1, 2), DateType(), datetime.date(2024, 1, 2)),
            (datetime.datetime(2024, 1, 2, 3, 4, 5), TimestampType(), datetime.datetime(2024, 1, 2, 3, 4, 5)),
            (aware, TimestampType(), aware.astimezone().replace(tzinfo=None)),
            (datetime.timedelta(days=1, seconds=5), DayTimeIntervalType(0, 3), datetime.timedelta(days=1, seconds=5)),
            ([1, 2], ArrayType(LongType()), [1, 2]),
            ([1, None], ArrayType(LongType()), [1, None]),
            ([[1], [2, 3]], ArrayType(ArrayType(LongType())), [[1], [2, 3]]),
            ([1, "a"], ArrayType(StringType()), ["1", "a"]),
            ({"a": 1}, MapType(StringType(), LongType()), {"a": 1}),
            ({"a": {"b": [1]}}, MapType(StringType(), MapType(StringType(), ArrayType(LongType()))), {"a": {"b": [1]}}),
            (Row(x=1, y="a"), struct_of(("x", LongType()), ("y", StringType())), Row(x=1, y="a")),
            ((1, "a"), struct_of(("_1", LongType()), ("_2", StringType())), Row(_1=1, _2="a")),
            ([Row(x=1), Row(x=2)], ArrayType(struct_of(("x", LongType()))), [Row(x=1), Row(x=2)]),
        ],
    )
    def test_infer_value(self, spark, value, data_type, collected):
        df = spark.createDataFrame([(value,)])
        assert df.schema == struct_of(("_1", data_type))
        res = df.collect()
        assert res == [Row(_1=collected)]
        assert res[0].asDict(recursive=True) == Row(_1=collected).asDict(recursive=True)

    @pytest.mark.parametrize("value", [b"ab", bytearray(b"ab")])
    def test_infer_binary(self, spark, value):
        df = spark.createDataFrame([(value,)])
        assert df.schema == struct_of(("_1", BinaryType()))
        assert bytes(df.collect()[0]._1) == b"ab"

    def test_python_ints_are_bigint(self, spark):
        df = spark.createDataFrame([(2_000_000_000,), (1,)], ["a"])
        assert df.schema == struct_of(("a", LongType()))
        assert df.select((sf.col("a") * 2).alias("b")).collect() == [Row(b=4_000_000_000), Row(b=2)]

    @pytest.mark.parametrize(
        ("rows", "data_type", "collected"),
        [
            ([(None,), (1,)], LongType(), [None, 1]),
            ([(1,), (None,)], LongType(), [1, None]),
            ([([],), ([1],)], ArrayType(LongType()), [[], [1]]),
            ([({},), ({"a": 1},)], MapType(StringType(), LongType()), [{}, {"a": 1}]),
            ([(1,), ("a",)], StringType(), ["1", "a"]),
            ([(True,), ("a",)], StringType(), ["true", "a"]),
            ([(1.33,), ("a",)], StringType(), ["1.33", "a"]),
        ],
    )
    def test_infer_across_rows(self, spark, rows, data_type, collected):
        df = spark.createDataFrame(rows)
        assert df.schema == struct_of(("_1", data_type))
        assert [r._1 for r in df.collect()] == collected

    @pytest.mark.parametrize(
        ("rows", "error", "error_class"),
        [
            ([(None,), (None,)], PySparkValueError, "CANNOT_DETERMINE_TYPE"),
            ([([None],)], PySparkValueError, "CANNOT_DETERMINE_TYPE"),
            ([(1,), (1.5,)], PySparkTypeError, "CANNOT_MERGE_TYPE"),
            ([(1.5,), (1,)], PySparkTypeError, "CANNOT_MERGE_TYPE"),
            ([(1,), (True,)], PySparkTypeError, "CANNOT_MERGE_TYPE"),
            ([(datetime.date(2024, 1, 2),), (datetime.datetime(2024, 1, 2),)], PySparkTypeError, "CANNOT_MERGE_TYPE"),
            ([([1, 1.5],)], PySparkTypeError, "CANNOT_INFER_TYPE_FOR_FIELD"),
            ([1, 2], PySparkTypeError, "CANNOT_INFER_SCHEMA_FOR_TYPE"),
            (["a", "b"], PySparkTypeError, "CANNOT_INFER_SCHEMA_FOR_TYPE"),
            ([], PySparkValueError, "CANNOT_INFER_EMPTY_SCHEMA"),
        ],
    )
    def test_infer_errors(self, spark, rows, error, error_class):
        with pytest.raises(error, match=error_class):
            spark.createDataFrame(rows)

    def test_infer_columns(self, spark):
        df = spark.createDataFrame([(1, "a", 1.5, True)])
        assert df.schema == struct_of(
            ("_1", LongType()), ("_2", StringType()), ("_3", DoubleType()), ("_4", BooleanType())
        )
        assert df.collect() == [Row(_1=1, _2="a", _3=1.5, _4=True)]

    @pytest.mark.parametrize(
        ("rows", "columns", "expected"),
        [
            ([Row(a=1, b="x"), Row(a=2, b="y")], ["a", "b"], [Row(a=1, b="x"), Row(a=2, b="y")]),
            ([Point(1, "a")], ["x", "y"], [Row(x=1, y="a")]),
            ([Person()], ["age", "name"], [Row(age=1, name="Alice")]),
            ([[1, "a"], [2, "b"]], ["_1", "_2"], [Row(_1=1, _2="a"), Row(_1=2, _2="b")]),
        ],
    )
    def test_infer_row_kinds(self, spark, rows, columns, expected):
        df = spark.createDataFrame(rows)
        assert df.columns == columns
        assert df.collect() == expected

    @pytest.mark.parametrize(
        ("names", "rows", "columns", "expected"),
        [
            (["x", "y"], [(1, "a")], ["x", "y"], [Row(x=1, y="a")]),
            (["x"], [(1, "a")], ["x", "_2"], [Row(x=1, _2="a")]),
            (["p", "q"], [{"b": 1, "a": "x"}], ["p", "q"], [Row(p="x", q=1)]),
            (["p", "q"], [Row(a=1, b="x")], ["p", "q"], [Row(p=1, q="x")]),
        ],
    )
    def test_infer_with_names(self, spark, names, rows, columns, expected):
        df = spark.createDataFrame(rows, names)
        assert df.columns == columns
        assert df.collect() == expected


class TestDicts:
    def test_dicts(self, spark):
        # Spark sorts the keys of each dict row, appends keys that later rows add, and gives NULL for a missing key.
        data = [
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "age": 30, "name": "Bob"},
            {"age": 35, "id": 3, "name": "Charlie", "city": "New York"},
        ]
        df = spark.createDataFrame(data)
        assert df.columns == ["age", "id", "name", "city"]
        assert df.schema == struct_of(
            ("age", LongType()), ("id", LongType()), ("name", StringType()), ("city", StringType())
        )
        assert df.collect() == [
            Row(age=25, id=1, name="Alice", city=None),
            Row(age=30, id=2, name="Bob", city=None),
            Row(age=35, id=3, name="Charlie", city="New York"),
        ]

    @pytest.mark.parametrize(
        ("data", "columns", "expected"),
        [
            ([{"b": 1}, {"a": 2, "b": 3}], ["b", "a"], [Row(b=1, a=None), Row(b=3, a=2)]),
            ([{"a": 1, "b": 2}, {"a": 3}], ["a", "b"], [Row(a=1, b=2), Row(a=3, b=None)]),
            ([{"a": None}, {"a": 1}], ["a"], [Row(a=None), Row(a=1)]),
        ],
    )
    def test_dicts_with_different_keys(self, spark, data, columns, expected):
        df = spark.createDataFrame(data)
        assert df.columns == columns
        assert df.collect() == expected

    def test_dicts_with_nested_values(self, spark):
        df = spark.createDataFrame([{"a": [1, 2], "m": {"k": "v"}, "r": Row(x=1)}])
        assert df.schema == struct_of(
            ("a", ArrayType(LongType())),
            ("m", MapType(StringType(), StringType())),
            ("r", struct_of(("x", LongType()))),
        )
        res = df.collect()
        assert res == [Row(a=[1, 2], m={"k": "v"}, r=Row(x=1))]
        assert res[0].r.x == 1


class TestStruct:
    @pytest.mark.parametrize(
        "make_struct",
        [
            lambda: sf.struct("x", "y"),
            lambda: sf.struct(sf.col("x"), sf.col("y")),
            lambda: sf.struct(["x", "y"]),
            lambda: sf.struct([sf.col("x"), sf.col("y")]),
        ],
    )
    def test_struct(self, spark, make_struct):
        df = spark.createDataFrame([(1, "a"), (2, "b")], ["x", "y"])
        res = df.select(make_struct().alias("s")).collect()
        assert res == [Row(s=Row(x=1, y="a")), Row(s=Row(x=2, y="b"))]
        assert res[0].s.y == "a"

    def test_struct_with_list_of_df_columns(self, spark):
        df = spark.createDataFrame([("Alice", 2), ("Bob", 5)], ("name", "age"))
        res = df.select(sf.struct([df.age, df.name]).alias("struct")).collect()
        assert res == [Row(struct=Row(age=2, name="Alice")), Row(struct=Row(age=5, name="Bob"))]
        assert res[0].struct.age == 2

    def test_struct_with_alias(self, spark):
        df = spark.createDataFrame([(1, "a")], ["x", "y"])
        res = df.select(sf.struct(sf.col("x"), sf.col("y").alias("z")).alias("s")).collect()
        assert res[0].s.asDict() == {"x": 1, "z": "a"}

    def test_struct_with_tuple_raises(self, spark):
        df = spark.createDataFrame([(1, "a")], ["x", "y"])
        with pytest.raises(PySparkTypeError):
            df.select(sf.struct(("x", "y")))


class TestReaderSchema:
    def test_load_without_format_quotes_path(self, spark, tmp_path):
        # Spark reads parquet when no format is given.
        sub_dir = tmp_path / "dir with space's"
        sub_dir.mkdir()
        file_path = (sub_dir / "data.parquet").as_posix()
        duckdb.sql("select 1 as a, 2 as b").write_parquet(file_path)

        assert spark.read.load(file_path).collect() == [Row(a=1, b=2)]

    def test_load_with_schema_string(self, spark, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("1,x\n2,y\n", newline="", encoding="utf-8")
        file_path = csv_path.as_posix()

        df = spark.read.load(file_path, format="csv", schema="a int, b string")
        assert df.schema == struct_of(("a", IntegerType()), ("b", StringType()))
        assert df.collect() == [Row(a=1, b="x"), Row(a=2, b="y")]

    def test_csv_with_schema_string(self, spark, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("1,x\n2,y\n", newline="", encoding="utf-8")
        file_path = csv_path.as_posix()

        df = spark.read.csv(file_path, schema="a bigint, b string")
        assert df.schema == struct_of(("a", LongType()), ("b", StringType()))
        assert df.collect() == [Row(a=1, b="x"), Row(a=2, b="y")]


class TestReaderSchemaMatching:
    @pytest.fixture
    def files(self, tmp_path):
        duckdb.sql("select 1 as a, 2 as b").write_parquet((tmp_path / "ab.parquet").as_posix())
        duckdb.sql('select 1 as "Mixed", 2 as "order"').write_parquet((tmp_path / "mixed.parquet").as_posix())
        (tmp_path / "ab.json").write_text('{"a": 1, "b": 2}\n', newline="", encoding="utf-8")
        (tmp_path / "mixed.json").write_text('{"Mixed": 1, "order": 2}\n', newline="", encoding="utf-8")
        (tmp_path / "ab.csv").write_text("1,2\n", newline="", encoding="utf-8")
        return tmp_path

    @pytest.mark.parametrize("fmt", ["parquet", "json"])
    @pytest.mark.parametrize(
        ("schema", "row"),
        [
            ("b int, a int", (2, 1)),
            ("b int", (2,)),
            ("a int, z int", (1, None)),
        ],
    )
    def test_by_name(self, spark, files, fmt, schema, row):
        df = spark.read.load((files / f"ab.{fmt}").as_posix(), format=fmt, schema=schema)
        assert [tuple(r) for r in df.collect()] == [row]
        assert df.columns == [name.split()[0] for name in schema.split(", ")]

    @pytest.mark.parametrize(("fmt", "row"), [("parquet", (1, 2)), ("json", (None, 2))])
    def test_case_of_names(self, spark, files, fmt, row):
        # Spark matches parquet columns ignoring case, JSON fields not.
        df = spark.read.load((files / f"mixed.{fmt}").as_posix(), format=fmt, schema="mixed int, `order` int")
        assert df.columns == ["mixed", "order"]
        assert [tuple(r) for r in df.collect()] == [row]

    def test_without_a_format(self, spark, files):
        df = spark.read.load((files / "ab.parquet").as_posix(), schema="b int, a int")
        assert [tuple(r) for r in df.collect()] == [(2, 1)]

    @pytest.mark.parametrize(
        ("schema", "row"),
        [
            ("b int, a int", (1, 2)),
            ("b int", (1,)),
            ("b int, a int, c int", (1, 2, None)),
            ("`order` int, a int", (1, 2)),
        ],
    )
    @pytest.mark.parametrize("read", ["load", "csv"])
    def test_csv_by_position(self, spark, files, read, schema, row):
        path = (files / "ab.csv").as_posix()
        if read == "load":
            df = spark.read.load(path, format="csv", schema=schema)
        else:
            df = spark.read.csv(path, schema=schema)
        assert [tuple(r) for r in df.collect()] == [row]

    def test_csv_load_reads_the_first_line_as_data(self, spark, tmp_path):
        path = tmp_path / "head.csv"
        path.write_text("id,name\n1,a\n", newline="", encoding="utf-8")
        rows = spark.read.load(path.as_posix(), format="csv").collect()
        assert sorted(tuple(map(str, r)) for r in rows) == [("1", "a"), ("id", "name")]


def _written_text(path):
    if path.is_dir():
        return "".join(part.read_text(encoding="utf-8") for part in sorted(path.glob("*.csv")))
    return path.read_text(encoding="utf-8")


class TestCsvFlags:
    @pytest.fixture
    def head_csv(self, tmp_path):
        path = tmp_path / "head.csv"
        path.write_text("id,name\n1,a\n", newline="", encoding="utf-8")
        return path.as_posix()

    @pytest.mark.parametrize("header", ["true", "TRUE", "True", True])
    def test_header_read(self, spark, head_csv, header):
        df = spark.read.csv(head_csv, header=header)
        assert df.columns == ["id", "name"]
        assert df.count() == 1

    @pytest.mark.parametrize("header", ["false", "False", False])
    def test_no_header_read(self, spark, head_csv, header):
        df = spark.read.csv(head_csv, header=header)
        assert df.columns != ["id", "name"]
        assert df.count() == 2

    @pytest.mark.parametrize("header", ["yes", 1])
    def test_header_that_is_not_a_flag(self, spark, head_csv, header):
        with pytest.raises(Exception, match="header flag can be true or false"):
            spark.read.csv(head_csv, header=header).collect()

    @pytest.mark.parametrize(
        "option", ["inferSchema", "multiLine", "ignoreLeadingWhiteSpace", "ignoreTrailingWhiteSpace"]
    )
    def test_false_flags_read(self, spark, head_csv, option):
        assert spark.read.csv(head_csv, header=True, **{option: "FALSE"}).count() == 1

    @pytest.mark.parametrize(("header", "written"), [("TRUE", True), ("true", True), (True, True), ("false", False)])
    def test_header_write(self, spark, tmp_path, header, written):
        path = tmp_path / "out.csv"
        spark.createDataFrame([(1, "a")], ["id", "name"]).write.csv(path.as_posix(), header=header)
        assert ("id,name" in _written_text(path)) == written

    @pytest.mark.parametrize(("quote_all", "quoted"), [("TRUE", True), (True, True), ("false", False)])
    def test_quote_all_write(self, spark, tmp_path, quote_all, quoted):
        path = tmp_path / "out.csv"
        spark.createDataFrame([(1, "a")], ["id", "name"]).write.csv(path.as_posix(), quoteAll=quote_all)
        assert ('"1","a"' in _written_text(path)) == quoted

    @pytest.mark.parametrize(
        ("column", "options", "written"),
        [
            ("d", {"dateFormat": "yyyy/MM/dd"}, "2024/02/03\n"),
            ("d", {"dateFormat": "dd MMM yy"}, "03 Feb 24\n"),
            ("d", {"dateFormat": "EEEE 'of' MMMM"}, "Saturday of February\n"),
            ("t", {"timestampFormat": "yyyy/MM/dd HH:mm:ss"}, "2024/02/03 04:05:06\n"),
            ("t", {"timestampFormat": "yyyy-MM-dd hh:mm:ss.SSS a"}, "2024-02-03 04:05:06.000 AM\n"),
        ],
    )
    def test_datetime_patterns_write(self, spark, tmp_path, column, options, written):
        path = tmp_path / "out.csv"
        df = spark.createDataFrame(
            [(datetime.date(2024, 2, 3), datetime.datetime(2024, 2, 3, 4, 5, 6))], "d date, t timestamp"
        )
        df.select(column).write.csv(path.as_posix(), **options)
        assert _written_text(path) == written

    def test_datetime_pattern_without_an_exact_format(self, spark, tmp_path):
        path = tmp_path / "out.csv"
        df = spark.createDataFrame([(datetime.date(2024, 2, 3),)], "d date")
        if USE_ACTUAL_SPARK:
            df.write.csv(path.as_posix(), dateFormat="yyyy/M/d")
            assert _written_text(path) == "2024/2/3\n"
        else:
            with pytest.raises(ContributionsAcceptedError):
                df.write.csv(path.as_posix(), dateFormat="yyyy/M/d")

    def test_quotes_write(self, spark, tmp_path):
        df = spark.createDataFrame([('a"b',), ("c,d",), ("e",)], ["s"])
        path = tmp_path / "default.csv"
        df.write.csv(path.as_posix())
        # Spark escapes a quote with a backslash, DuckDB doubles it; each reads its own text back.
        escaped = '"a\\"b"' if USE_ACTUAL_SPARK else '"a""b"'
        assert _written_text(path) == f'{escaped}\n"c,d"\ne\n'
        assert spark.read.csv(path.as_posix()).collect() == df.collect()
        path = tmp_path / "unescaped.csv"
        if USE_ACTUAL_SPARK:
            df.write.csv(path.as_posix(), escapeQuotes=False)
            assert _written_text(path) == 'a"b\n"c,d"\ne\n'
        else:
            # DuckDB quotes every value with a quote character in it.
            with pytest.raises(ContributionsAcceptedError):
                df.write.csv(path.as_posix(), escapeQuotes=False)


class TestTimestampRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(
                datetime.datetime(9999, 12, 31, 12),
                marks=pytest.mark.skipif(
                    sys.platform == "win32",
                    reason="outside the range of the Windows C runtime, as for PySpark on Windows",
                ),
            ),
            datetime.datetime(1970, 1, 1),
            datetime.datetime(2024, 3, 10, 2, 30),
            datetime.datetime(2024, 11, 3, 1, 30),
            datetime.datetime(2024, 7, 1, 12, 0, 0, 123456),
        ],
    )
    def test_naive_timestamp(self, spark, value):
        # Spark reads a naive datetime as local time through mktime, which also settles gaps and overlaps.
        expected = datetime.datetime.fromtimestamp(time.mktime(value.timetuple())).replace(
            microsecond=value.microsecond
        )
        (row,) = spark.createDataFrame([(value,)], "t timestamp").collect()
        assert row.t == expected
        assert row.t.fold == expected.fold

    @pytest.mark.skipif(
        sys.platform == "win32", reason="outside the range of the Windows C runtime, as for PySpark on Windows"
    )
    def test_aware_timestamp_near_the_end_of_the_calendar(self, spark):
        value = datetime.datetime(9999, 12, 31, 12, tzinfo=datetime.timezone.utc)
        (row,) = spark.createDataFrame([(value,)], "t timestamp").collect()
        assert row.t == value.astimezone().replace(tzinfo=None)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="outside the range of the Windows C runtime, as for PySpark on Windows"
    )
    def test_latest_instant(self, spark):
        # The last microsecond of the calendar is a time like any other, not DuckDB's infinity.
        value = datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=datetime.timezone.utc)
        try:
            expected = value.astimezone().replace(tzinfo=None)
        except OverflowError:
            pytest.skip("the local time of the instant is past the end of the calendar")
        (row,) = spark.createDataFrame([(value,)], "t timestamp").collect()
        assert row.t == expected

    def test_ntz_timestamp_near_the_end_of_the_calendar(self, spark):
        value = datetime.datetime(9999, 12, 31, 12)
        assert spark.createDataFrame([(value,)], "t timestamp_ntz").collect() == [Row(t=value)]

    def test_infinite_timestamps(self, spark):
        if USE_ACTUAL_SPARK:
            pytest.skip("Spark has no infinite timestamps")
        # DuckDB returns infinite timestamps as the ends of Python's calendar.
        df = spark.sql("select 'infinity'::timestamptz a, '-infinity'::timestamptz b, 'infinity'::timestamp c")
        assert df.collect() == [Row(a=datetime.datetime.max, b=datetime.datetime.min, c=datetime.datetime.max)]


# File text, Spark's inferred types, and where DuckDB's type detection gives another type, the module's.
_CSV_INFERENCE_CASES = [
    ("ints", "a,b\n1,2147483648\n-3,4\n", ["int", "bigint"], ["bigint", "bigint"]),
    ("doubles", "a,b\n1.5,1e3\n2,-0.25\n", ["double", "double"], None),
    (
        "dates",
        "a,b\n2024-01-02,2024-01-02 03:04:05\n2023-12-31,2023-12-31 23:59:59.5\n",
        ["date", "timestamp"],
        None,
    ),
    ("booleans", "a,b\ntrue,True\nfalse,FALSE\n", ["boolean", "boolean"], None),
    ("empty_fields", "a,b,c\n1,,x\n,2.5,\n", ["int", "double", "string"], ["bigint", "double", "string"]),
    ("mixed", "a,b\n1,x\n2.5,7\n", ["double", "string"], None),
    ("wide_integer", "a\n12345678901234567890\n1\n", ["decimal(20,0)"], ["hugeint"]),
    ("all_empty", "a,b\n,\n,\n", ["string", "string"], None),
]


class TestCsvInference:
    @pytest.mark.parametrize(("name", "text", "spark_types", "module_types"), _CSV_INFERENCE_CASES)
    def test_strings_without_inference(self, spark, tmp_path, name, text, spark_types, module_types):
        path = tmp_path / f"{name}.csv"
        path.write_text(text, newline="", encoding="utf-8")
        df = spark.read.csv(path.as_posix(), header=True)
        lines = [line.split(",") for line in text.splitlines()[1:]]
        assert [t for _, t in df.dtypes] == ["string"] * len(lines[0])
        assert [tuple(r) for r in df.collect()] == [tuple(v or None for v in line) for line in lines]

    @pytest.mark.parametrize(("name", "text", "spark_types", "module_types"), _CSV_INFERENCE_CASES)
    def test_inferred_types(self, spark, tmp_path, name, text, spark_types, module_types):
        path = tmp_path / f"{name}.csv"
        path.write_text(text, newline="", encoding="utf-8")
        df = spark.read.csv(path.as_posix(), header=True, inferSchema=True)
        expected = module_types if module_types is not None and not USE_ACTUAL_SPARK else spark_types
        assert [t for _, t in df.dtypes] == expected

    def test_inferred_values(self, spark, tmp_path):
        path = tmp_path / "values.csv"
        path.write_text(
            "i,d,day,ts,b,s\n1,1.5,2024-01-02,2024-01-02 03:04:05,true,x\n,,,,,\n", newline="", encoding="utf-8"
        )
        rows = spark.read.csv(path.as_posix(), header=True, inferSchema=True).collect()
        assert [tuple(r) for r in rows] == [
            (1, 1.5, datetime.date(2024, 1, 2), datetime.datetime(2024, 1, 2, 3, 4, 5), True, "x"),
            (None,) * 6,
        ]

    def test_inferred_timestamps_are_local_times(self, spark, tmp_path):
        path = tmp_path / "times.csv"
        path.write_text("ts\n2024-01-02 03:04:05\n2024-07-01 12:00:00\n", newline="", encoding="utf-8")
        df = spark.read.csv(path.as_posix(), header=True, inferSchema=True)
        assert df.schema == struct_of(("ts", TimestampType()))
        winter, summer = datetime.datetime(2024, 1, 2, 3, 4, 5), datetime.datetime(2024, 7, 1, 12)
        assert [r.ts for r in df.collect()] == [winter, summer]
        assert df.filter(sf.col("ts") == sf.lit(summer)).count() == 1
        local = spark.createDataFrame([(winter,), (summer,)], "ts timestamp")
        assert df.join(local, "ts").count() == 2


class TestNestedTimestamps:
    def test_collect(self, spark):
        value = datetime.datetime(2024, 3, 10, 2, 30)
        local = datetime.datetime.fromtimestamp(time.mktime(value.timetuple()))
        schema = (
            "a array<timestamp>, aa array<array<timestamp>>, s struct<t: timestamp, n: int>, "
            "m map<string, timestamp>, k map<timestamp, int>, sa array<struct<t: timestamp>>"
        )
        df = spark.createDataFrame(
            [([value, None], [[value]], (value, 1), {"x": value}, {value: 1}, [(value,)]), (None,) * 6], schema
        )
        assert df.collect() == [
            Row(a=[local, None], aa=[[local]], s=Row(t=local, n=1), m={"x": local}, k={local: 1}, sa=[Row(t=local)]),
            Row(a=None, aa=None, s=None, m=None, k=None, sa=None),
        ]
        assert df.first() == df.collect()[0]
        assert df.take(1) == df.collect()[:1]

    def test_show(self, spark, capsys):
        value = datetime.datetime(2024, 1, 2, 3, 4, 5)
        df = spark.createDataFrame(
            [(value, [value], (value,))], "t timestamp, a array<timestamp>, s struct<u: timestamp>"
        )
        df.show(truncate=False)
        assert capsys.readouterr().out == (
            "+-------------------+---------------------+---------------------+\n"
            "|t                  |a                    |s                    |\n"
            "+-------------------+---------------------+---------------------+\n"
            "|2024-01-02 03:04:05|[2024-01-02 03:04:05]|{2024-01-02 03:04:05}|\n"
            "+-------------------+---------------------+---------------------+\n\n"
        )


class TestMapsWithNestedKeys:
    def test_struct_keys(self, spark):
        df = spark.createDataFrame([({Row(a=1): 2, Row(a=None): None},), ({},), (None,)], "m map<struct<a:int>, int>")
        assert df.collect() == [Row(m={Row(a=1): 2, Row(a=None): None}), Row(m={}), Row(m=None)]

    def test_struct_keys_nested(self, spark):
        schema = "m map<struct<a:int>, map<struct<b:int>, struct<c:int>>>, s struct<n: map<struct<a:int>, int>>"
        df = spark.createDataFrame([({Row(a=1): {Row(b=2): Row(c=3)}}, Row(n={Row(a=4): 5}))], schema)
        assert df.collect() == [Row(m={Row(a=1): {Row(b=2): Row(c=3)}}, s=Row(n={Row(a=4): 5}))]

    def test_array_keys_raise(self, spark):
        # Python cannot hash a list, so Spark too fails to build the dict.
        with pytest.raises(TypeError, match="unhashable"):
            spark.createDataFrame([({(1, 2): 3},)], "m map<array<int>, int>").collect()
        if not USE_ACTUAL_SPARK:
            with pytest.raises(TypeError, match="unhashable"):
                spark.sql("select map([[1, 2]], [3]) m").collect()

    def test_struct_keys_from_sql(self, spark):
        if USE_ACTUAL_SPARK:
            pytest.skip("DuckDB SQL")
        assert spark.sql("select map([{'a': 1}], [2]) m").collect() == [Row(m={Row(a=1): 2})]

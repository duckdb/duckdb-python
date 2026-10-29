import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

import datetime
import re
from decimal import Decimal

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import ParseException, PySparkTypeError, PySparkValueError, SparkRuntimeException
from spark_namespace.sql import functions as F
from spark_namespace.sql.functions import array, col, struct
from spark_namespace.sql.types import (
    BooleanType,
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    Row,
    ShortType,
    StringType,
    StructField,
    StructType,
)

if USE_ACTUAL_SPARK:
    import pyspark


def _ansi(spark):
    return USE_ACTUAL_SPARK and spark.conf.get("spark.sql.ansi.enabled") == "true"


class TestSparkColumn:
    def test_struct_column(self, spark):
        df = spark.createDataFrame([Row(a=1, b=2, c=3, d=4)])
        df = df.withColumn("struct", struct(df.a, df.b))
        assert "struct" in df.columns
        assert df.schema["struct"].dataType.fieldNames() == ["a", "b"]

        # The module keeps PySpark 3.5's error table; PySpark 4 renamed this error.
        if USE_ACTUAL_SPARK and not pyspark.__version__.startswith("3."):
            message = "[NOT_EXPECTED_TYPE] Argument `col` should be Column, got str."
        else:
            message = "[NOT_COLUMN] Argument `col` should be a Column, got str."
        with pytest.raises(PySparkTypeError, match=re.escape(message)):
            df = df.withColumn("struct", "yes")

    def test_array_column(self, spark):
        df = spark.createDataFrame([Row(a=1, b=2, c=3, d=4)], ["a", "b", "c", "d"])

        df2 = df.select(
            array(df["a"], df["b"]).alias("array"),
            array(df["a"]).alias("array_single"),
            array("b", "c", "d").alias("array_str"),
        )
        res = df2.collect()

        assert res[0].array == [1, 2]
        assert res[0].array_single == [1]
        assert res[0].array_str == [2, 3, 4]

    def test_isnull(self, spark):
        df = spark.createDataFrame([(1, None), (None, 2)], ("a", "b"))

        res = df.select("a", "b", col("a").isNull().alias("r1"), df.b.isNull().alias("r2")).collect()
        assert res == [
            Row(a=1, b=None, r1=False, r2=True),
            Row(a=None, b=2, r1=True, r2=False),
        ]

    def test_isnotnull(self, spark):
        df = spark.createDataFrame([(1, None), (None, 2)], ("a", "b"))

        res = df.select("a", "b", col("a").isNotNull().alias("r1"), df.b.isNotNull().alias("r2")).collect()
        assert res == [
            Row(a=1, b=None, r1=True, r2=False),
            Row(a=None, b=2, r1=False, r2=True),
        ]

    def test_cast_to_integral_truncates(self, spark):
        schema = StructType(
            [
                StructField("big", LongType()),
                StructField("d", DoubleType()),
                StructField("f", FloatType()),
                StructField("b", BooleanType()),
                StructField("n", LongType()),
            ]
        )
        df = spark.createDataFrame(
            [(9007199254740993, -2.7, 2.7, True, None), (-9007199254740993, 2.5, -0.5, False, None)], schema
        )
        res = df.select(
            col("big").cast("bigint").alias("big"),
            col("big").cast("decimal(20,0)").cast("bigint").alias("big_dec"),
            col("d").cast("int").alias("d_int"),
            col("d").cast(LongType()).alias("d_long"),
            col("d").cast(ShortType()).alias("d_short"),
            col("d").cast(ByteType()).alias("d_byte"),
            col("f").cast("int").alias("f_int"),
            col("d").cast("decimal(5,2)").cast("int").alias("dec_int"),
            col("b").cast("int").alias("b_int"),
            col("n").cast("int").alias("n_int"),
        )
        assert [f.dataType for f in res.schema.fields] == [
            LongType(),
            LongType(),
            IntegerType(),
            LongType(),
            ShortType(),
            ByteType(),
            IntegerType(),
            IntegerType(),
            IntegerType(),
            IntegerType(),
        ]
        assert res.collect() == [
            Row(
                big=9007199254740993,
                big_dec=9007199254740993,
                d_int=-2,
                d_long=-2,
                d_short=-2,
                d_byte=-2,
                f_int=2,
                dec_int=-2,
                b_int=1,
                n_int=None,
            ),
            Row(
                big=-9007199254740993,
                big_dec=-9007199254740993,
                d_int=2,
                d_long=2,
                d_short=2,
                d_byte=2,
                f_int=0,
                dec_int=2,
                b_int=0,
                n_int=None,
            ),
        ]

    def test_cast_inside_expressions(self, spark):
        df = spark.createDataFrame(
            [(2.7, -1.5), (3.2, 0.0)], StructType([StructField("a", DoubleType()), StructField("b", DoubleType())])
        )
        res = df.select(
            (col("a").cast("int") + col("b").cast("int")).alias("s"),
            F.abs(col("b").cast("int")).alias("abs"),
            F.concat(col("a").cast("int").cast("string"), F.lit("x")).alias("c"),
            F.when(col("a").cast("int") == 2, "two").otherwise("other").alias("w"),
        ).collect()
        assert res == [Row(s=1, abs=1, c="2x", w="two"), Row(s=3, abs=0, c="3x", w="other")]
        assert df.filter(col("a").cast("int") == 3).collect() == [Row(a=3.2, b=0.0)]

    def test_cast_strings_to_integral(self, spark):
        schema = StructType([StructField(c, StringType()) for c in "abcd"])
        df = spark.createDataFrame([("7", " 8 ", "-2.7", "1.9")], schema)
        assert df.select(col("a").cast("int").alias("a"), col("b").cast("bigint").alias("b")).collect() == [
            Row(a=7, b=8)
        ]
        if _ansi(spark):
            for fractional in ["c", "d"]:
                with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
                    df.select(col(fractional).cast("int")).collect()
        else:
            res = df.select(col("c").cast("int").alias("c"), col("d").cast("bigint").alias("d"))
            assert res.collect() == [Row(c=-2, d=1)]

    @pytest.mark.parametrize("value", ["abc", "1e3", "-", "", " ", "1_0"])
    def test_cast_invalid_string_to_integral(self, spark, value):
        df = spark.createDataFrame([(value,)], StructType([StructField("s", StringType())]))
        if USE_ACTUAL_SPARK and not _ansi(spark):
            assert df.select(col("s").cast("int").alias("v")).collect() == [Row(v=None)]
        else:
            with pytest.raises(Exception, match=r"(?i)cast|convert"):
                df.select(col("s").cast("int")).collect()

    def test_cast_fractional_strings_to_integral(self, spark):
        values = [".5", "-.5", "3.", "+2.5", " -7.9 ", "-12.99999999999999999999", "9007199254740993.9"]
        df = spark.createDataFrame([(v,) for v in values], StructType([StructField("s", StringType())]))
        if _ansi(spark):
            for value in values:
                with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
                    df.filter(col("s") == value).select(col("s").cast("int")).collect()
        else:
            res = df.select(col("s").cast("bigint").alias("l")).collect()
            assert res == [Row(l=v) for v in [0, 0, 3, 2, -7, -12, 9007199254740993]]
            assert df.filter(col("s") != values[-1]).select(col("s").cast("int").alias("i")).collect() == [
                Row(i=v) for v in [0, 0, 3, 2, -7, -12]
            ]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 1e20])
    def test_cast_out_of_range_double_to_int(self, spark, value):
        df = spark.range(1).select(F.lit(value).alias("v"))
        if USE_ACTUAL_SPARK and not _ansi(spark):
            expected = {"nan": 0, "inf": 2147483647, "-inf": -2147483648}.get(str(value), 2147483647)
            assert df.select(col("v").cast("int").alias("v")).collect() == [Row(v=expected)]
        else:
            with pytest.raises(Exception, match=r"(?i)overflow|out of range"):
                df.select(col("v").cast("int")).collect()

    def test_isin_without_values(self, spark):
        df = spark.createDataFrame([(1,)], StructType([StructField("a", LongType())]))
        assert df.select(col("a").isin().alias("v"), col("a").isin([]).alias("w")).collect() == [Row(v=False, w=False)]

    def test_like_backslash_escape(self, spark):
        df = spark.createDataFrame([("a%b",), ("axb",), ("a\\b",)], StructType([StructField("s", StringType())]))
        res = df.select(
            col("s").like("a\\%b").alias("pct"),
            col("s").like("a\\\\b").alias("slash"),
            col("s").ilike("A\\%B").alias("ipct"),
        ).collect()
        assert res == [
            Row(pct=True, slash=False, ipct=True),
            Row(pct=False, slash=False, ipct=False),
            Row(pct=False, slash=True, ipct=False),
        ]

    def test_struct_values_collect_as_rows(self, spark):
        schema = StructType([StructField("a", LongType()), StructField("b", LongType())])
        df = spark.createDataFrame([(7, -2)], schema)
        res = df.select(struct("a", "b").alias("s"), array(struct(col("a").alias("z"))).alias("l"))
        row = res.collect()[0]
        assert row.s == Row(a=7, b=-2)
        assert row.s.a == 7
        assert row.l == [Row(z=7)]

    @pytest.mark.timeout(60)
    def test_column_is_not_iterable(self, spark):
        with pytest.raises(PySparkTypeError, match=re.escape("[NOT_ITERABLE] Column is not iterable.")):
            iter(col("a"))
        with pytest.raises(PySparkTypeError, match="NOT_ITERABLE"):
            list(col("a"))

    def test_column_has_no_truth_value(self, spark):
        message = "[CANNOT_CONVERT_COLUMN_INTO_BOOL] Cannot convert column into bool: please use '&' for 'and'"
        with pytest.raises(PySparkValueError, match=re.escape(message)):
            bool(col("a"))
        with pytest.raises(PySparkValueError, match="CANNOT_CONVERT_COLUMN_INTO_BOOL"):
            _ = col("a") in [col("b")]
        with pytest.raises(PySparkValueError, match="CANNOT_APPLY_IN_FOR_COLUMN"):
            _ = "x" in col("a")

    def test_remainder(self, spark):
        schema = StructType([StructField(c, LongType()) for c in "az"] + [StructField("d", DoubleType())])
        df = spark.createDataFrame([(-7, 3, -7.5)], schema)
        res = df.select((col("a") % col("z")).alias("i"), (col("d") % 2).alias("d"), (7 % col("z")).alias("r"))
        assert res.columns == ["i", "d", "r"]
        assert res.collect() == [Row(i=-1, d=-1.5, r=1)]
        assert df.select(col("a") % col("z"), 7 % col("z")).columns == ["(a % z)", "(7 % z)"]

    def test_remainder_by_zero(self, spark):
        schema = StructType([StructField(c, LongType()) for c in "az"] + [StructField("d", DoubleType())])
        df = spark.createDataFrame([(7, 0, 7.5)], schema)
        for remainder in [col("a") % col("z"), col("d") % col("z"), col("d") % 0.0, 7 % col("z")]:
            res = df.select(remainder.alias("r"))
            if _ansi(spark):
                with pytest.raises(Exception, match="REMAINDER_BY_ZERO"):
                    res.collect()
            else:
                assert res.collect() == [Row(r=None)]

    @pytest.mark.parametrize("value", [{"k": 1}, {}, {"key": [1], "value": [2]}, [{"k": 1}]])
    def test_lit_of_a_dict_raises(self, spark, value):
        with pytest.raises(SparkRuntimeException, match=re.escape("[UNSUPPORTED_FEATURE.LITERAL_TYPE]")):
            spark.range(1).select(F.lit(value).alias("a")).collect()

    def test_struct_field_names(self, spark):
        inner = StructType([StructField("a", LongType()), StructField("b", StringType())])
        schema = StructType([StructField("n", LongType()), StructField("s", StringType()), StructField("st", inner)])
        df = spark.createDataFrame([(1, "x", (2, "y"))], schema)
        res = df.select(
            struct(
                col("n") + 1,
                "s",
                F.lit(2),
                col("n").alias("m"),
                F.upper("s"),
                col("st.a"),
                col("st")["b"],
                df.n.cast("string"),
            ).alias("o")
        )
        assert res.schema["o"].dataType.fieldNames() == ["col1", "s", "col3", "m", "col5", "a", "b", "col8"]
        assert res.collect() == [Row(o=Row(col1=2, s="x", col3=2, m=1, col5="X", a=2, b="y", col8="1"))]
        assert df.select(struct("s", col("n") * 2)).columns == ["struct(s, (n * 2))"]

    def test_item_access_reads_the_field_of_that_name(self, spark):
        df = spark.createDataFrame([((1, (2,), 3, 4),)], 's struct<`a.b`:int, a:struct<b:int>, `c d`:int, `e"f`:int>')
        res = df.select(col("s")["a.b"], df.s["a.b"], col("s")["a"]["b"], col("s")["c d"], col("s")['e"f'])
        assert res.columns == ["s.a.b", "s.a.b", "s.a.b", "s.c d", 's.e"f']
        assert [tuple(r) for r in res.collect()] == [(1, 1, 2, 3, 4)]
        assert df.select(col("s")["A.B"].alias("v")).collect() == [Row(v=1)]

    def test_item_access_on_a_computed_struct(self, spark):
        computed = struct(F.lit(1).alias("a"), struct(F.lit(2).alias("c")).alias("b"))
        res = spark.range(1).select(computed["a"], computed["b"]["c"])
        assert res.columns == ["struct(1 AS a, struct(2 AS c) AS b).a", "struct(1 AS a, struct(2 AS c) AS b).b.c"]
        assert [tuple(r) for r in res.collect()] == [(1, 2)]


class TestCastTypeStrings:
    @pytest.mark.parametrize(
        ("source", "type_string", "simple", "spark_type", "value"),
        [
            ("t", "timestamp", "timestamp", "TIMESTAMP", datetime.datetime(2024, 1, 2, 3, 4, 5)),
            ("t", "timestamp_ltz", "timestamp", "TIMESTAMP", datetime.datetime(2024, 1, 2, 3, 4, 5)),
            ("t", "timestamp_ntz", "timestamp_ntz", "TIMESTAMP_NTZ", datetime.datetime(2024, 1, 2, 3, 4, 5)),
            ("t", "DATE", "date", "DATE", datetime.date(2024, 1, 2)),
            ("n", "decimal", "decimal(10,0)", "DECIMAL(10,0)", Decimal("13")),
            ("n", "decimal(5,2)", "decimal(5,2)", "DECIMAL(5,2)", Decimal("12.70")),
            ("n", "float", "float", "FLOAT", 12.699999809265137),
            ("i", "byte", "tinyint", "TINYINT", 3),
            ("i", "short", "smallint", "SMALLINT", 3),
            ("i", "long", "bigint", "BIGINT", 3),
            ("i", "string", "string", "STRING", "3"),
        ],
    )
    def test_atomic_types(self, spark, source, type_string, simple, spark_type, value):
        df = spark.createDataFrame([("2024-01-02 03:04:05", "12.7", "3")], "t string, n string, i string")
        res = df.select(F.col(source).cast(type_string), F.trim(source).cast(type_string))
        assert res.columns == [source, f"CAST(trim({source}) AS {spark_type})"]
        assert res.schema.simpleString() == f"struct<{source}:{simple},CAST(trim({source}) AS {spark_type}):{simple}>"
        assert tuple(res.collect()[0]) == (value, value)

    def test_nested_types(self, spark):
        schema = "a array<bigint>, m map<string,bigint>, st struct<p:bigint,q:string>"
        df = spark.createDataFrame([([1, 2], {"k": 1}, (1, "x"))], schema)
        res = df.select(
            F.col("a").cast("array<int>"),
            F.col("m").cast("map<string,int>"),
            F.col("st").cast("struct<p:string,q:string>"),
        )
        assert res.schema.simpleString() == "struct<a:array<int>,m:map<string,int>,st:struct<p:string,q:string>>"
        assert res.collect() == [Row(a=[1, 2], m={"k": 1}, st=Row(p="1", q="x"))]

    def test_duckdb_only_type(self, spark):
        df = spark.createDataFrame([(3,)], "i int")
        if USE_ACTUAL_SPARK:
            with pytest.raises(ParseException):
                df.select(F.col("i").cast("hugeint")).collect()
        else:
            res = df.select(F.col("i").cast("hugeint"))
            assert res.schema.simpleString() == "struct<i:hugeint>"
            assert res.collect() == [Row(i=3)]

    @pytest.mark.parametrize("type_string", ["varchar(10)", "char(3)", "CHARACTER(2)"])
    def test_char_and_varchar_cast_to_string(self, spark, type_string):
        df = spark.createDataFrame([(" ab ",)], "s string")
        res = df.select(F.col("s").cast(type_string), F.trim("s").cast(type_string))
        assert res.columns == ["s", "CAST(trim(s) AS STRING)"]
        assert res.schema.simpleString() == "struct<s:string,CAST(trim(s) AS STRING):string>"
        assert res.collect() == [Row(" ab ", "ab")]

    def test_nested_char_and_varchar_cast_to_string(self, spark):
        df = spark.createDataFrame([(["x"], {"k": "v"})], "a array<string>, m map<string,string>")
        res = df.select(F.col("a").cast("array<varchar(1)>"), F.col("m").cast("map<char(1),varchar(2)>"))
        assert res.schema.simpleString() == "struct<a:array<string>,m:map<string,string>>"
        assert res.collect() == [Row(a=["x"], m={"k": "v"})]

    @pytest.mark.parametrize(
        ("source", "type_string", "value"),
        [
            ("s", "VARCHAR", "{}"),
            ("s", "json", "{}"),
            ("st", "STRUCT(b INT)", Row(b=1)),
            ("a", "INTEGER[]", [1]),
            ("a", "INTEGER[1]", (1,)),
        ],
    )
    def test_duckdb_spellings(self, spark, source, type_string, value):
        df = spark.createDataFrame([("{}", (1.7,), [1.7])], "s string, st struct<a:double>, a array<double>")
        if USE_ACTUAL_SPARK:
            with pytest.raises(ParseException):
                df.select(F.col(source).cast(type_string)).collect()
        else:
            assert df.select(F.col(source).cast(type_string).alias("v")).collect() == [Row(v=value)]

    @pytest.mark.parametrize("type_string", ["inetger", "varchar", "varchar(x)", "array<inetger>", "a int, b string"])
    def test_invalid_type_string_raises_parse_exception(self, spark, type_string):
        df = spark.createDataFrame([("a",)], "s string")
        if type_string == "varchar" and not USE_ACTUAL_SPARK:
            assert df.select(F.col("s").cast(type_string)).collect() == [Row(s="a")]
            return
        with pytest.raises(ParseException):
            df.select(F.col("s").cast(type_string)).collect()

    def test_bare_interval_raises(self, spark):
        df = spark.createDataFrame([("1 day",)], "s string")
        if not USE_ACTUAL_SPARK:
            from duckdb.experimental.spark.exception import ContributionsAcceptedError as expected
        elif pyspark.__version__.startswith("3."):
            expected = PySparkValueError
        else:
            from spark_namespace.errors import PySparkNotImplementedError as expected
        with pytest.raises(expected):
            df.select(F.col("s").cast("interval")).collect()


class TestSparkCastSemantics:
    def test_decimal_to_integral_truncates_exactly(self, spark):
        df = spark.createDataFrame(
            [(Decimal("100000000000.999999"), Decimal("-3.999999999"), Decimal("3.99999999999999999"))],
            "a decimal(18,6), b decimal(18,9), c decimal(38,17)",
        )
        res = df.select(F.col("a").cast("long"), F.col("b").cast("int"), F.col("c").cast("bigint"))
        assert res.collect() == [Row(a=100000000000, b=-3, c=3)]

    def test_decimal_to_small_integral_types(self, spark):
        df = spark.createDataFrame(
            [(Decimal("1.99999999999999999"), Decimal("-0.99999999999999999999"))], "a decimal(18,17), b decimal(38,20)"
        )
        res = df.select(F.col("a").cast("int"), F.col("a").cast("tinyint").alias("t"), F.col("b").cast("short"))
        # Spark in ANSI mode casts a decimal to int, smallint or tinyint through its nearest double.
        if _ansi(spark):
            assert res.collect() == [Row(a=2, t=2, b=-1)]
        else:
            assert res.collect() == [Row(a=1, t=1, b=0)]

    def test_integral_casts_truncate_inside_nested_types(self, spark):
        df = spark.createDataFrame(
            [([1.7, -1.7, None], [[1.7, -2.5]], {"a": 1.7, "b": -2.9}, (-1.7, Decimal("2.99")))],
            "a array<double>, n array<array<double>>, m map<string,double>, s struct<x:double,y:decimal(5,2)>",
        )
        res = df.select(
            F.col("a").cast("array<int>"),
            F.col("n").cast("array<array<bigint>>"),
            F.col("m").cast("map<string,int>"),
            F.col("s").cast("struct<x:int,y:smallint>"),
        )
        assert res.columns == ["a", "n", "m", "s"]
        assert res.schema.simpleString() == (
            "struct<a:array<int>,n:array<array<bigint>>,m:map<string,int>,s:struct<x:int,y:smallint>>"
        )
        assert res.collect() == [Row(a=[1, -1, None], n=[[1, -2]], m={"a": 1, "b": -2}, s=Row(x=-1, y=2))]

    def test_map_key_cast_truncates(self, spark):
        df = spark.createDataFrame([({1.7: "a"},)], "m map<double,string>")
        if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3."):
            with pytest.raises(Exception, match="DATATYPE_MISMATCH"):
                df.select(F.col("m").cast("map<int,string>")).collect()
        else:
            assert df.select(F.col("m").cast("map<int,string>")).collect() == [Row(m={1: "a"})]

    def test_struct_cast_maps_fields_by_position(self, spark):
        df = spark.createDataFrame([((5, "z"),), (None,)], "s struct<p:int,q:string>")
        renamed = df.select(F.col("s").cast("struct<p:int,y:string>"))
        assert renamed.schema.simpleString() == "struct<s:struct<p:int,y:string>>"
        assert renamed.collect() == [Row(s=Row(p=5, y="z")), Row(s=None)]
        swapped = df.select(F.col("s").cast("struct<q:string,p:string>"))
        assert swapped.collect() == [Row(s=Row(q="5", p="z")), Row(s=None)]

    @pytest.mark.parametrize("target", ["struct<p:int>", "struct<p:int,q:string,r:int>"])
    def test_struct_cast_with_another_field_count_raises(self, spark, target):
        df = spark.createDataFrame([((5, "z"),)], "s struct<p:int,q:string>")
        with pytest.raises(Exception, match=r"(?i)cast|size|out of range"):
            df.select(F.col("s").cast(target)).collect()

    def test_nested_struct_casts_map_fields_by_position(self, spark):
        df = spark.createDataFrame(
            [([(5, "z")], {"k": (1, "w")}, ((1.7,), "z"))],
            "a array<struct<p:int,q:string>>, m map<string,struct<p:int,q:string>>, "
            "s struct<p:struct<r:double>,q:string>",
        )
        res = df.select(
            F.col("a").cast("array<struct<u:bigint,v:string>>"),
            F.col("m").cast("map<string,struct<u:bigint,v:string>>"),
            F.col("s").cast("struct<a:struct<b:int>,c:string>"),
            F.col("s").cast("struct<a:struct<b:double>,c:string>").alias("d"),
        )
        assert res.collect() == [
            Row(
                a=[Row(u=5, v="z")],
                m={"k": Row(u=1, v="w")},
                s=Row(a=Row(b=1), c="z"),
                d=Row(a=Row(b=1.7), c="z"),
            )
        ]

    def test_cast_evaluates_its_input_once(self, spark):
        df = spark.createDataFrame([(1,), (2,), (3,)], "v bigint")
        if USE_ACTUAL_SPARK:
            spark.udf.register("counted", lambda v: v * 1.5, DoubleType())
        else:
            calls = []

            def counted(v):
                calls.append(v)
                return v * 1.5

            # A UDF with side effects is evaluated wherever it occurs, so the count shows how often the cast reads it.
            spark.conn.create_function("counted", counted, ["BIGINT"], "DOUBLE", side_effects=True)
        res = df.select(
            F.expr("counted(v)").cast("int").alias("i"),
            F.struct(F.expr("counted(v)").alias("x")).cast("struct<x:int>").alias("s"),
            F.factorial(F.expr("counted(v)") + 1).alias("f"),
            F.char(F.expr("counted(v)") + 64).alias("c"),
        )
        assert sorted(res.collect()) == [
            Row(i=1, s=Row(x=1), f=2, c="A"),
            Row(i=3, s=Row(x=3), f=24, c="C"),
            Row(i=4, s=Row(x=4), f=120, c="D"),
        ]
        if not USE_ACTUAL_SPARK:
            assert len(calls) == 12

    def test_string_to_binary_takes_utf8_bytes(self, spark):
        df = spark.createDataFrame([("é", bytearray(b"\xff\x00\\x")), ("a\\x41", None)], "s string, b binary")
        res = df.select(
            F.col("s").cast("binary"), F.col("b").cast("binary"), F.array("s").cast("array<binary>").alias("a")
        )
        assert res.collect() == [
            Row(s=b"\xc3\xa9", b=b"\xff\x00\\x", a=[b"\xc3\xa9"]),
            Row(s=b"a\\x41", b=None, a=[b"a\\x41"]),
        ]

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="COLLATE NOCASE is DuckDB's collation syntax")
    def test_collated_string_casts_as_a_string(self, spark):
        df = spark.sql(
            "SELECT 'é' COLLATE NOCASE AS s, '9007199254740992.9' COLLATE NOCASE AS t, '1e2' COLLATE NOCASE AS u"
        )
        res = df.select(F.col("s").cast("binary"), F.col("t").cast("bigint"))
        assert res.collect() == [Row(s=b"\xc3\xa9", t=9007199254740992)]
        with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
            df.select(F.col("u").cast("bigint")).collect()

    def test_integral_to_binary(self, spark):
        df = spark.range(1)
        if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3."):
            assert df.select(F.col("id").cast("binary")).collect() == [Row(id=bytearray(8))]
        else:
            with pytest.raises(Exception, match=r"(?i)cast"):
                df.select(F.col("id").cast("binary")).collect()

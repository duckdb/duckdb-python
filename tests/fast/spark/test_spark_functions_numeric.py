import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

import math

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import PySparkTypeError, SparkRuntimeException
from spark_namespace.sql import functions as sf
from spark_namespace.sql.types import (
    ArrayType,
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

import numpy as np

if USE_ACTUAL_SPARK:
    import pyspark

NAN = float("nan")


def _ansi(spark):
    return USE_ACTUAL_SPARK and spark.conf.get("spark.sql.ansi.enabled") == "true"


def _nan_frame(spark):
    """Rows k=1 NaN, k=2 1.0, k=3 NULL; the NaN comes from a literal so it does not depend on createDataFrame."""
    schema = StructType([StructField("k", LongType()), StructField("x", DoubleType())])
    df = spark.createDataFrame([(1, 0.0), (2, 1.0), (3, None)], schema)
    return df.withColumn("x", sf.when(sf.col("k") == 1, sf.lit(NAN)).otherwise(sf.col("x")))


class TestSparkFunctionsNumeric:
    def test_greatest(self, spark):
        data = [
            (1, 2),
            (4, 3),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])
        df = df.withColumn("greatest_value", sf.greatest(sf.col("firstColumn"), sf.col("secondColumn")))
        res = df.select("greatest_value").collect()
        assert res == [
            Row(greatest_value=2),
            Row(greatest_value=4),
        ]

    def test_least(self, spark):
        data = [
            (1, 2),
            (4, 3),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])
        df = df.withColumn("least_value", sf.least(sf.col("firstColumn"), sf.col("secondColumn")))
        res = df.select("least_value").collect()
        assert res == [
            Row(least_value=1),
            Row(least_value=3),
        ]

    def test_ceil(self, spark):
        data = [
            (1.1,),
            (2.9,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("ceil_value", sf.ceil(sf.col("firstColumn")))
        res = df.select("ceil_value").collect()
        assert res == [
            Row(ceil_value=2),
            Row(ceil_value=3),
        ]

    def test_floor(self, spark):
        data = [
            (1.1,),
            (2.9,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("floor_value", sf.floor(sf.col("firstColumn")))
        res = df.select("floor_value").collect()
        assert res == [
            Row(floor_value=1),
            Row(floor_value=2),
        ]

    def test_ceil_and_floor_of_integral_types_are_exact(self, spark):
        df = spark.createDataFrame(
            [(9007199254740993, -9223372036854775808, 7, 5), (None, None, None, None)],
            "b bigint, m bigint, i int, t tinyint",
        )
        res = df.select(sf.ceil("b"), sf.floor("b"), sf.ceil("m"), sf.floor("m"), sf.ceil("i"), sf.floor("t"))
        assert [f.dataType for f in res.schema.fields] == [LongType()] * 6
        assert [tuple(r) for r in res.collect()] == [
            (9007199254740993, 9007199254740993, -9223372036854775808, -9223372036854775808, 7, 5),
            (None,) * 6,
        ]

    def test_abs(self, spark):
        data = [
            (1.1,),
            (-2.9,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("abs_value", sf.abs(sf.col("firstColumn")))
        res = df.select("abs_value").collect()
        assert res == [
            Row(abs_value=1.1),
            Row(abs_value=2.9),
        ]

    def test_sqrt(self, spark):
        data = [
            (4,),
            (9,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("sqrt_value", sf.sqrt(sf.col("firstColumn")))
        res = df.select("sqrt_value").collect()
        assert res == [
            Row(sqrt_value=2.0),
            Row(sqrt_value=3.0),
        ]

    def test_cbrt(self, spark):
        data = [
            (8,),
            (27,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("cbrt_value", sf.cbrt(sf.col("firstColumn")))
        res = df.select("cbrt_value").collect()
        assert pytest.approx(res[0].cbrt_value) == 2.0
        assert pytest.approx(res[1].cbrt_value) == 3.0

    def test_cos(self, spark):
        data = [
            (0.0,),
            (3.14159,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("cos_value", sf.cos(sf.col("firstColumn")))
        res = df.select("cos_value").collect()
        assert len(res) == 2
        assert res[0].cos_value == pytest.approx(1.0)
        assert res[1].cos_value == pytest.approx(-1.0)

    def test_acos(self, spark):
        data = [
            (1,),
            (-1,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("acos_value", sf.acos(sf.col("firstColumn")))
        res = df.select("acos_value").collect()
        assert len(res) == 2
        assert res[0].acos_value == pytest.approx(0.0)
        assert res[1].acos_value == pytest.approx(3.141592653589793)

    def test_exp(self, spark):
        data = [
            (0.693,),
            (0.0,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("exp_value", sf.exp(sf.col("firstColumn")))
        res = df.select("exp_value").collect()
        assert round(res[0].exp_value, 2) == 2
        assert res[1].exp_value == 1

    def test_factorial(self, spark):
        data = [
            (4,),
            (5,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("factorial_value", sf.factorial(sf.col("firstColumn")))
        res = df.select("factorial_value").collect()
        assert res == [
            Row(factorial_value=24),
            Row(factorial_value=120),
        ]

    def test_log2(self, spark):
        data = [
            (4,),
            (8,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("log2_value", sf.log2(sf.col("firstColumn")))
        res = df.select("log2_value").collect()
        assert res == [
            Row(log2_value=2.0),
            Row(log2_value=3.0),
        ]

    def test_ln(self, spark):
        data = [
            (2.718,),
            (1.0,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("ln_value", sf.ln(sf.col("firstColumn")))
        res = df.select("ln_value").collect()
        assert round(res[0].ln_value, 2) == 1
        assert res[1].ln_value == 0

    def test_degrees(self, spark):
        data = [
            (3.14159,),
            (0.0,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("degrees_value", sf.degrees(sf.col("firstColumn")))
        res = df.select("degrees_value").collect()
        assert round(res[0].degrees_value, 2) == 180
        assert res[1].degrees_value == 0

    def test_radians(self, spark):
        data = [
            (180,),
            (0,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("radians_value", sf.radians(sf.col("firstColumn")))
        res = df.select("radians_value").collect()
        assert round(res[0].radians_value, 2) == 3.14
        assert res[1].radians_value == 0

    def test_atan(self, spark):
        data = [
            (1,),
            (0,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("atan_value", sf.atan(sf.col("firstColumn")))
        res = df.select("atan_value").collect()
        assert round(res[0].atan_value, 2) == 0.79
        assert res[1].atan_value == 0

    def test_atan2(self, spark):
        data = [
            (1, 1),
            (0, 0),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])

        # Both columns
        df2 = df.withColumn("atan2_value", sf.atan2(sf.col("firstColumn"), "secondColumn"))
        res = df2.select("atan2_value").collect()
        assert round(res[0].atan2_value, 2) == 0.79
        assert res[1].atan2_value == 0

        # Both literals
        df2 = df.withColumn("atan2_value_lit", sf.atan2(1, 1))
        res = df2.select("atan2_value_lit").collect()
        assert round(res[0].atan2_value_lit, 2) == 0.79
        assert round(res[1].atan2_value_lit, 2) == 0.79

        # One literal, one column
        df2 = df.withColumn("atan2_value_lit_col", sf.atan2(1.0, sf.col("secondColumn")))
        res = df2.select("atan2_value_lit_col").collect()
        assert round(res[0].atan2_value_lit_col, 2) == 0.79
        assert round(res[1].atan2_value_lit_col, 2) == 1.57

    def test_tan(self, spark):
        data = [
            (0,),
            (1,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("tan_value", sf.tan(sf.col("firstColumn")))
        res = df.select("tan_value").collect()
        assert res[0].tan_value == 0
        assert round(res[1].tan_value, 2) == 1.56

    def test_round(self, spark):
        data = [
            (11.15,),
            (2.9,),
            # Test with this that HALF_UP rounding method is used and not HALF_EVEN
            (2.5,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = (
            df.withColumn("round_value", sf.round("firstColumn"))
            .withColumn("round_value_1", sf.round(sf.col("firstColumn"), 1))
            .withColumn("round_value_minus_1", sf.round("firstColumn", -1))
        )
        res = df.select("round_value", "round_value_1", "round_value_minus_1").collect()
        assert res == [
            Row(round_value=11, round_value_1=11.2, round_value_minus_1=10),
            Row(round_value=3, round_value_1=2.9, round_value_minus_1=0),
            Row(round_value=3, round_value_1=2.5, round_value_minus_1=0),
        ]

    def test_bround(self, spark):
        data = [
            (11.15,),
            (2.9,),
            # Test with this that HALF_EVEN rounding method is used and not HALF_UP
            (2.5,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = (
            df.withColumn("round_value", sf.bround(sf.col("firstColumn")))
            .withColumn("round_value_1", sf.bround(sf.col("firstColumn"), 1))
            .withColumn("round_value_minus_1", sf.bround(sf.col("firstColumn"), -1))
        )
        res = df.select("round_value", "round_value_1", "round_value_minus_1").collect()
        assert res == [
            Row(round_value=11, round_value_1=11.2, round_value_minus_1=10),
            Row(round_value=3, round_value_1=2.9, round_value_minus_1=0),
            Row(round_value=2, round_value_1=2.5, round_value_minus_1=0),
        ]

    def test_asin(self, spark):
        df = spark.createDataFrame([(0,), (2,)], ["value"])

        df = df.withColumn("asin_value", sf.asin("value"))
        res = df.select("asin_value").collect()

        assert res[0].asin_value == 0
        assert np.isnan(res[1].asin_value)

    def test_corr(self, spark):
        N = 20
        a = range(N)
        b = [2 * x for x in range(N)]
        # Have to use a groupby to test this as agg is not yet implemented without
        df = spark.createDataFrame(zip(a, b, ["group1"] * N, strict=False), ["a", "b", "g"])

        res = df.groupBy("g").agg(sf.corr("a", "b").alias("c")).collect()
        assert pytest.approx(res[0].c) == 1

    def test_cot(self, spark):
        df = spark.createDataFrame([(math.radians(45),)], ["value"])

        res = df.select(sf.cot(df["value"]).alias("cot")).collect()
        assert pytest.approx(res[0].cot) == 1

    def test_e(self, spark):
        df = spark.createDataFrame([("value",)], ["value"])

        res = df.select(sf.e().alias("e")).collect()
        assert pytest.approx(res[0].e) == math.e

    def test_pi(self, spark):
        df = spark.createDataFrame([("value",)], ["value"])

        res = df.select(sf.pi().alias("pi")).collect()
        assert pytest.approx(res[0].pi) == math.pi

    def test_pow(self, spark):
        df = spark.createDataFrame([(2, 3)], ["a", "b"])

        res = df.select(sf.pow(df["a"], df["b"]).alias("pow")).collect()
        assert res[0].pow == 8

    def test_random(self, spark):
        df = spark.range(0, 2, 1)
        res = df.withColumn("rand", sf.rand()).collect()

        assert isinstance(res[0].rand, float)
        assert res[0].rand >= 0
        assert res[0].rand < 1

        assert isinstance(res[1].rand, float)
        assert res[1].rand >= 0
        assert res[1].rand < 1

    @pytest.mark.parametrize("sign_func", [sf.sign, sf.signum])
    def test_sign(self, spark, sign_func):
        df = spark.range(1).select(sign_func(sf.lit(-5).alias("v1")), sign_func(sf.lit(6).alias("v2")))
        res = df.collect()
        assert res == [Row(v1=-1.0, v2=1.0)]

    def test_sin(self, spark):
        df = spark.range(1)
        res = df.select(sf.sin(sf.lit(math.radians(90))).alias("v")).collect()
        assert res == [Row(v=1.0)]

    def test_negative(self, spark):
        df = spark.createDataFrame([(0,), (2,), (-3,)], ["value"])
        df = df.withColumn("value", sf.negative(sf.col("value")))
        res = df.collect()
        assert res[0].value == 0
        assert res[1].value == -2
        assert res[2].value == 3

    def test_negative_and_positive(self, spark):
        schema = StructType([StructField("i", LongType()), StructField("d", DoubleType())])
        df = spark.createDataFrame([(2, 2.5), (-3, -3.5), (None, None)], schema)
        res = df.select(
            sf.negative("i").alias("ni"),
            sf.negative(sf.col("d")).alias("nd"),
            sf.positive("i").alias("pi"),
            (-df.i).alias("op"),
        ).collect()
        assert res == [
            Row(ni=-2, nd=-2.5, pi=2, op=-2),
            Row(ni=3, nd=3.5, pi=-3, op=3),
            Row(ni=None, nd=None, pi=None, op=None),
        ]

    def test_nan_literal(self, spark):
        res = spark.range(1).select(
            sf.lit(NAN).alias("n"),
            sf.isnan(sf.lit(NAN)).alias("isnan"),
            (sf.lit(NAN) == sf.lit(NAN)).alias("eq"),
            (sf.lit(NAN) > sf.lit(1e308)).alias("gt"),
            sf.when(sf.lit(False), 1.0).otherwise(NAN).alias("otherwise"),
        )
        assert res.schema.fields[0].dataType == DoubleType()
        row = res.collect()[0]
        assert math.isnan(row.n)
        assert math.isnan(row.otherwise)
        assert (row.isnan, row.eq, row.gt) == (True, True, True)

    def test_nan_comparisons(self, spark):
        df = _nan_frame(spark)
        assert df.filter(sf.col("x") == NAN).select("k").collect() == [Row(k=1)]
        assert df.filter(sf.col("x") != NAN).select("k").collect() == [Row(k=2)]
        assert df.filter(sf.col("x").isin(NAN, 5.0)).select("k").collect() == [Row(k=1)]
        res = df.select(
            "k",
            (sf.col("x") > 1e308).alias("gt"),
            (sf.col("x") < NAN).alias("lt"),
            (sf.col("x") >= NAN).alias("ge"),
            sf.isnan("x").alias("isnan"),
        ).orderBy("k")
        assert res.collect() == [
            Row(k=1, gt=True, lt=False, ge=True, isnan=True),
            Row(k=2, gt=False, lt=True, ge=False, isnan=False),
            Row(k=3, gt=None, lt=None, ge=None, isnan=False),
        ]

    def test_nan_is_largest(self, spark):
        df = _nan_frame(spark)
        assert [r.k for r in df.orderBy("x").collect()] == [3, 2, 1]
        assert [r.k for r in df.orderBy(sf.desc("x")).collect()] == [1, 2, 3]
        row = df.select(sf.max("x").alias("max"), sf.min("x").alias("min")).collect()[0]
        assert math.isnan(row.max)
        assert row.min == 1.0
        greatest = df.select("k", sf.greatest("x", sf.lit(5.0)).alias("g"), sf.least("x", sf.lit(5.0)).alias("l"))
        rows = greatest.orderBy("k").collect()
        assert math.isnan(rows[0].g)
        assert rows[0].l == 5.0
        assert (rows[1].g, rows[1].l) == (5.0, 1.0)

    def test_isnan_of_null_is_false(self, spark):
        df = spark.createDataFrame([(None,), (1.0,)], StructType([StructField("x", DoubleType())]))
        assert df.select(sf.isnan("x").alias("v")).collect() == [Row(v=False), Row(v=False)]

    def test_division_by_zero(self, spark):
        df = spark.createDataFrame(
            [(7, 2.0)], StructType([StructField("a", LongType()), StructField("b", DoubleType())])
        )
        assert df.select((df.a / 2).alias("q"), (7 / df.b).alias("r")).collect() == [Row(q=3.5, r=3.5)]
        if _ansi(spark):
            with pytest.raises(Exception, match="DIVIDE_BY_ZERO"):
                df.select(df.a / 0).collect()
        else:
            res = df.select((df.a / 0).alias("q"), (df.b / 0).alias("d"), (1 / (df.a - 7)).alias("r")).collect()
            assert res == [Row(q=None, d=None, r=None)]

    def test_result_types(self, spark):
        schema = StructType([StructField("d", DoubleType()), StructField("i", IntegerType())])
        df = spark.createDataFrame([(2.5, -3)], schema)
        res = df.select(
            sf.ceil("d").alias("c"),
            sf.floor("d").alias("f"),
            sf.ceil("i").alias("ci"),
            sf.sign("i").alias("s"),
            sf.signum("d").alias("sd"),
        )
        assert [f.dataType for f in res.schema.fields] == [
            LongType(),
            LongType(),
            LongType(),
            DoubleType(),
            DoubleType(),
        ]
        assert res.collect() == [Row(c=3, f=2, ci=-3, s=-1.0, sd=1.0)]

    def test_logarithm_of_non_positive_is_null(self, spark):
        df = spark.createDataFrame([(8.0,), (0.0,), (-1.0,), (None,)], StructType([StructField("x", DoubleType())]))
        res = df.select(sf.log2("x").alias("l2"), sf.ln("x").alias("ln")).collect()
        assert res[0].l2 == 3.0
        assert res[0].ln == pytest.approx(math.log(8))
        assert res[1:] == [Row(l2=None, ln=None)] * 3

    def test_pow_with_numbers(self, spark):
        df = spark.createDataFrame([(2, 3)], StructType([StructField("a", LongType()), StructField("b", LongType())]))
        res = df.select(sf.pow("a", 2).alias("p"), sf.pow(2.0, "b").alias("q"), sf.pow(df.a, df.b).alias("r")).collect()
        assert res == [Row(p=4.0, q=8.0, r=8.0)]

    def test_factorial_range(self, spark):
        df = spark.createDataFrame(
            [(5,), (0,), (20,), (21,), (-1,), (None,)], StructType([StructField("n", LongType())])
        )
        res = df.select(sf.factorial("n").alias("f"))
        assert res.schema.fields[0].dataType == LongType()
        assert [r.f for r in res.collect()] == [120, 1, 2432902008176640000, None, None, None]

    def test_numpy_scalars(self, spark):
        scalars = {
            "i8": (np.int8(1), ByteType()),
            "i16": (np.int16(1), ShortType()),
            "i32": (np.int32(1), IntegerType()),
            "i64": (np.int64(1), LongType()),
            "f32": (np.float32(1.5), FloatType()),
            "f64": (np.float64(1.5), DoubleType()),
            "b": (np.bool_(True), BooleanType()),
        }
        res = spark.range(1).select(*[sf.lit(v).alias(name) for name, (v, _) in scalars.items()])
        assert [f.dataType for f in res.schema.fields] == [t for _, t in scalars.values()]
        assert res.collect() == [Row(i8=1, i16=1, i32=1, i64=1, f32=1.5, f64=1.5, b=True)]
        df = spark.createDataFrame([([1, 2, 3],), ([],)], StructType([StructField("data", ArrayType(LongType()))]))
        res = df.select(
            sf.array_contains(df.data, np.int64(1)).alias("b"), (sf.array_size(df.data) == np.int32(3)).alias("c")
        )
        assert res.collect() == [Row(b=True, c=True), Row(b=False, c=False)]

    def test_factorial_truncates_before_the_range_check(self, spark):
        schema = StructType([StructField(c, DoubleType()) for c in "abcde"])
        df = spark.createDataFrame([(5.5, 20.5, -0.5, 21.0, None)], schema)
        res = df.select(*(sf.factorial(c) for c in "abcde"), sf.factorial(sf.lit(5.9)))
        assert res.columns == [*(f"factorial({c})" for c in "abcde"), "factorial(5.9)"]
        assert [f.dataType for f in res.schema.fields] == [LongType()] * 6
        assert res.collect() == [Row(120, 2432902008176640000, 1, None, None, 120)]

    def test_sign_of_nan(self, spark):
        rows = _nan_frame(spark).select(sf.sign("x").alias("s"), sf.signum("x").alias("t")).collect()
        assert math.isnan(rows[0].s)
        assert math.isnan(rows[0].t)
        assert rows[1:] == [Row(s=1.0, t=1.0), Row(s=None, t=None)]

    def test_lit_list_keeps_nan(self, spark):
        res = spark.range(1).select(sf.lit([NAN, 1.0]).alias("a"), sf.lit([[NAN], [2.0, None]]).alias("n"))
        assert res.schema.simpleString() == "struct<a:array<double>,n:array<array<double>>>"
        row = res.collect()[0]
        assert math.isnan(row.a[0])
        assert row.a[1:] == [1.0]
        assert math.isnan(row.n[0][0])
        assert row.n[1] == [2.0, None]

    def test_numpy_scalars_as_operands(self, spark):
        df = spark.createDataFrame([(10,)], StructType([StructField("a", LongType())]))
        res = df.select(df.a * np.float64(1e20), df.a + np.float64(1.5), df.a + np.float32(0.1), df.a + np.int32(3))
        assert res.columns == ["(a * 1.0E20)", "(a + 1.5)", "(a + 0.10000000149011612)", "(a + 3)"]
        assert [f.dataType for f in res.schema.fields] == [DoubleType(), DoubleType(), DoubleType(), LongType()]
        assert res.collect() == [Row(1e21, 11.5, 10.100000001490116, 13)]

    def test_numpy_scalar_literal_names(self, spark):
        res = spark.range(1).select(
            sf.lit(np.float64(0.1)), sf.lit(np.float32(0.1)), sf.lit(np.int8(3)), sf.lit(np.bool_(False))
        )
        assert res.columns == ["0.1", "0.1", "3", "false"]
        assert [f.dataType for f in res.schema.fields] == [DoubleType(), FloatType(), ByteType(), BooleanType()]

    def test_numpy_scalar_literal_names_inside_expressions(self, spark):
        # Spark casts the Python value of a NumPy number to its type and aliases the cast to the number's str().
        literals = [np.int8(5), np.int16(-5), np.int32(5), np.int64(5), np.float32(1.5), np.float64(np.inf)]
        res = spark.range(1).select(*(sf.lit(v).isNull() for v in literals), sf.lit(np.int8(5)).alias("q").isNull())
        assert res.columns == [
            "(CAST(5 AS TINYINT) AS `5` IS NULL)",
            "(CAST(-5 AS SMALLINT) AS `-5` IS NULL)",
            "(CAST(5 AS INT) AS `5` IS NULL)",
            "(CAST(5 AS BIGINT) AS `5` IS NULL)",
            "(CAST(1.5 AS FLOAT) AS `1.5` IS NULL)",
            "(CAST(Infinity AS DOUBLE) AS inf IS NULL)",
            "(CAST(5 AS TINYINT) AS `5` AS q IS NULL)",
        ]
        assert spark.range(1).select(sf.struct(sf.lit(np.int8(5)))).schema.fields[0].dataType.fieldNames() == ["5"]

    @pytest.mark.parametrize(
        ("array", "spark_name", "module_name", "element_type", "value"),
        [
            (
                np.array([1.5, 0.1, 1e20, -0.0]),
                "ARRAY(1.5D, 0.1D, 1.0E20D, -0.0D)",
                "array(1.5, 0.1, 1.0E20, -0.0)",
                DoubleType(),
                [1.5, 0.1, 1e20, -0.0],
            ),
            (np.array([np.inf]), "ARRAY(CAST('Infinity' AS DOUBLE))", "array(Infinity)", DoubleType(), [np.inf]),
            (np.array([1.5], dtype=np.float32), "ARRAY(CAST('1.5' AS FLOAT))", "array(1.5)", FloatType(), [1.5]),
            (np.array([5, -3], dtype=np.int64), "ARRAY(5L, -3L)", "array(5, -3)", LongType(), [5, -3]),
            (np.array([5], dtype=np.int32), "ARRAY(5)", "array(5)", IntegerType(), [5]),
            (np.array([5], dtype=np.int16), "ARRAY(5S)", "array(5)", ShortType(), [5]),
            (np.array([False, True]), "ARRAY(false, true)", "array(false, true)", BooleanType(), [False, True]),
            (np.array(["a'b", "c"]), "ARRAY('a\\'b', 'c')", "array(a'b, c)", StringType(), ["a'b", "c"]),
            (np.array([]), "ARRAY()", "array()", DoubleType(), []),
        ],
    )
    def test_numpy_array_literals(self, spark, array, spark_name, module_name, element_type, value):
        res = spark.range(1).select(sf.lit(array))
        # The module names a NumPy array as the list of its values, where Spark spells each element's type.
        assert res.columns == [spark_name if USE_ACTUAL_SPARK else module_name]
        assert res.schema.fields[0].dataType == ArrayType(element_type)
        assert res.collect() == [Row(value)]

    def test_numpy_int8_array_literal(self, spark):
        res = spark.range(1).select(sf.lit(np.array([5], dtype=np.int8)))
        assert res.columns == ["ARRAY(5S)" if USE_ACTUAL_SPARK else "array(5)"]
        # PySpark 3.5 gives the bytes a short type.
        expected = ShortType() if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3.") else ByteType()
        assert res.schema.fields[0].dataType == ArrayType(expected)

    @pytest.mark.parametrize("value", [np.array([1], dtype=np.uint16), np.array([1, "a"], dtype=object)])
    def test_unsupported_numpy_arrays_raise(self, spark, value):
        with pytest.raises(PySparkTypeError, match="UNSUPPORTED_NUMPY_ARRAY_SCALAR"):
            spark.range(1).select(sf.lit(value)).collect()

    def test_unsupported_numpy_scalar_raises(self, spark):
        # PySpark 3.5 passes the date on to Py4J, which fails on it.
        expected = AttributeError if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3.") else PySparkTypeError
        with pytest.raises(expected):
            spark.range(1).select(sf.lit(np.datetime64("2020-01-01"))).collect()

    def test_two_dimensional_numpy_literal_raises(self, spark):
        with pytest.raises(SparkRuntimeException, match=r"UNSUPPORTED_FEATURE\.LITERAL_TYPE"):
            spark.range(1).select(sf.lit(np.array([[1, 2]]))).collect()

    def test_smallest_double_literal_name(self, spark):
        assert spark.range(1).select(sf.lit(5e-324), sf.lit(2.5e-5)).columns == ["4.9E-324", "2.5E-5"]

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

import math
import numpy as np
from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.sql import functions as sf
from spark_namespace.sql.types import Row


class TestSparkFunctionsNumeric(object):
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
        round(res[0].exp_value, 2) == 2
        res[1].exp_value == 1

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
        round(res[0].ln_value, 2) == 1
        res[1].ln_value == 0

    def test_degrees(self, spark):
        data = [
            (3.14159,),
            (0.0,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("degrees_value", sf.degrees(sf.col("firstColumn")))
        res = df.select("degrees_value").collect()
        round(res[0].degrees_value, 2) == 180
        res[1].degrees_value == 0

    def test_radians(self, spark):
        data = [
            (180,),
            (0,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("radians_value", sf.radians(sf.col("firstColumn")))
        res = df.select("radians_value").collect()
        round(res[0].radians_value, 2) == 3.14
        res[1].radians_value == 0

    def test_atan(self, spark):
        data = [
            (1,),
            (0,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("atan_value", sf.atan(sf.col("firstColumn")))
        res = df.select("atan_value").collect()
        round(res[0].atan_value, 2) == 0.79
        res[1].atan_value == 0

    def test_atan2(self, spark):
        data = [
            (1, 1),
            (0, 0),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])

        # Both columns
        df2 = df.withColumn("atan2_value", sf.atan2(sf.col("firstColumn"), "secondColumn"))
        res = df2.select("atan2_value").collect()
        round(res[0].atan2_value, 2) == 0.79
        res[1].atan2_value == 0

        # Both literals
        df2 = df.withColumn("atan2_value_lit", sf.atan2(1, 1))
        res = df2.select("atan2_value_lit").collect()
        round(res[0].atan2_value_lit, 2) == 0.79
        round(res[1].atan2_value_lit, 2) == 0.79

        # One literal, one column
        df2 = df.withColumn("atan2_value_lit_col", sf.atan2(1.0, sf.col("secondColumn")))
        res = df2.select("atan2_value_lit_col").collect()
        round(res[0].atan2_value_lit_col, 2) == 0.79
        res[1].atan2_value_lit_col == 0

    def test_tan(self, spark):
        data = [
            (0,),
            (1,),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("tan_value", sf.tan(sf.col("firstColumn")))
        res = df.select("tan_value").collect()
        res[0].tan_value == 0
        round(res[1].tan_value, 2) == 1.56

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
        if USE_ACTUAL_SPARK:
            assert np.isnan(res[1].asin_value)
        else:
            # FIXME: DuckDB should return NaN here. Reason is that
            # ConstantExpression(float("nan")) gives NULL and not NaN
            assert res[1].asin_value is None

    def test_corr(self, spark):
        N = 20
        a = range(N)
        b = [2 * x for x in range(N)]
        # Have to use a groupby to test this as agg is not yet implemented without
        df = spark.createDataFrame(zip(a, b, ["group1"] * N), ["a", "b", "g"])

        res = df.groupBy("g").agg(sf.corr("a", "b").alias('c')).collect()
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
        res = df.withColumn('rand', sf.rand()).collect()

        assert isinstance(res[0].rand, float)
        assert res[0].rand >= 0 and res[0].rand < 1

        assert isinstance(res[1].rand, float)
        assert res[1].rand >= 0 and res[1].rand < 1

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
        assert res[2].value == -3
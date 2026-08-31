import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace.sql import functions as F
from spark_namespace.sql.types import Row


class TestSparkAggregateFunctions:
    def test_count_distinct(self, spark):
        df = spark.createDataFrame([("g", 1), ("g", 1), ("g", 2), ("g", None)], ["k", "v"])
        res = df.groupBy("k").agg(F.count_distinct("v").alias("d")).collect()
        assert res == [Row(k="g", d=2)]

    def test_countDistinct_alias(self, spark):
        df = spark.createDataFrame([("g", 1), ("g", 1), ("g", 2)], ["k", "v"])
        res = df.groupBy("k").agg(F.countDistinct("v").alias("d")).collect()
        assert res == [Row(k="g", d=2)]

    def test_count_distinct_multi_col(self, spark):
        df = spark.createDataFrame(
            [
                ("g", 1, "a"),
                ("g", 1, "a"),
                ("g", 1, "b"),
                ("g", None, "c"),
                ("g", 2, None),
                ("g", None, None),
            ],
            ["k", "a", "b"],
        )
        res = df.groupBy("k").agg(F.count_distinct("a", "b").alias("d")).collect()
        assert res == [Row(k="g", d=2)]

    def test_collect_set(self, spark):
        df = spark.createDataFrame([("g", 1), ("g", 1), ("g", 2), ("g", None)], ["k", "v"])
        row = df.groupBy("k").agg(F.collect_set("v").alias("s")).collect()[0]
        assert row.k == "g"
        assert sorted(row.s) == [1, 2]

    def test_count_if(self, spark):
        df = spark.createDataFrame([("g", 1), ("g", 2), ("g", 3)], ["k", "v"])
        res = df.groupBy("k").agg(F.count_if(F.col("v") > 1).alias("c")).collect()
        assert res == [Row(k="g", c=2)]

    def test_max_by(self, spark):
        df = spark.createDataFrame([("g", "a", 1), ("g", "b", 3), ("g", "c", 2)], ["k", "name", "v"])
        res = df.groupBy("k").agg(F.max_by("name", "v").alias("m")).collect()
        assert res == [Row(k="g", m="b")]

    def test_min_by(self, spark):
        df = spark.createDataFrame([("g", "a", 1), ("g", "b", 3), ("g", "c", 2)], ["k", "name", "v"])
        res = df.groupBy("k").agg(F.min_by("name", "v").alias("m")).collect()
        assert res == [Row(k="g", m="a")]

    def test_bool_and(self, spark):
        df = spark.createDataFrame([("g", True), ("g", True), ("g", False)], ["k", "b"])
        res = df.groupBy("k").agg(F.bool_and("b").alias("r")).collect()
        assert res == [Row(k="g", r=False)]

        df2 = spark.createDataFrame([("g", True), ("g", True), ("g", True)], ["k", "b"])
        res2 = df2.groupBy("k").agg(F.bool_and("b").alias("r")).collect()
        assert res2 == [Row(k="g", r=True)]

    def test_every_alias(self, spark):
        df = spark.createDataFrame([("g", True), ("g", False)], ["k", "b"])
        res = df.groupBy("k").agg(F.every("b").alias("r")).collect()
        assert res == [Row(k="g", r=False)]

    def test_bool_or(self, spark):
        df = spark.createDataFrame([("g", True), ("g", False), ("g", False)], ["k", "b"])
        res = df.groupBy("k").agg(F.bool_or("b").alias("r")).collect()
        assert res == [Row(k="g", r=True)]

        df2 = spark.createDataFrame([("g", False), ("g", False)], ["k", "b"])
        res2 = df2.groupBy("k").agg(F.bool_or("b").alias("r")).collect()
        assert res2 == [Row(k="g", r=False)]

    def test_some_alias(self, spark):
        df = spark.createDataFrame([("g", True), ("g", False)], ["k", "b"])
        res = df.groupBy("k").agg(F.some("b").alias("r")).collect()
        assert res == [Row(k="g", r=True)]

    def test_any_alias(self, spark):
        df = spark.createDataFrame([("g", True), ("g", False)], ["k", "b"])
        res = df.groupBy("k").agg(F.any("b").alias("r")).collect()
        assert res == [Row(k="g", r=True)]

    def test_kurtosis(self, spark):
        df = spark.createDataFrame([("g", 1.0), ("g", 2.0), ("g", 3.0), ("g", 4.0)], ["k", "v"])
        row = df.groupBy("k").agg(F.kurtosis("v").alias("kur")).collect()[0]
        assert row.k == "g"
        assert row.kur is not None

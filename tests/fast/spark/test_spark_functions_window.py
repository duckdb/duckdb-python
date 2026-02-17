import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace.sql import functions as F
from spark_namespace.sql.types import Row
from spark_namespace.sql.window import Window

from duckdb.experimental.spark import ContributionsAcceptedError


class TestDataFrameWindowFunction:
    def test_order_by(self, spark):
        simpleData = [
            ("Sales", "NY", 2024, 10000),
            ("Sales", "NY", 2025, 20000),
            ("Sales", "CA", 2024, 23000),
            ("Finance", "CA", 2024, 23000),
            ("Finance", "CA", 2025, 24000),
            ("Finance", "NY", 2025, 19000),
            ("Finance", "NY", 2024, 15000),
            ("Marketing", "CA", 2024, 18000),
            ("Marketing", "NY", 2025, 21000),
        ]
        columns = ["department", "state", "year", "bonus"]
        df = spark.createDataFrame(data=simpleData, schema=columns)
        df = df.withColumn(
            "cumulative_bonus", F.sum("bonus").over(Window.partitionBy("department", "state").orderBy("year"))
        )
        df = df.sort("department", "state", "year")
        res1 = df.collect()
        assert res1 == [
            Row(department="Finance", state="CA", year=2024, bonus=23000, cumulative_bonus=23000),
            Row(department="Finance", state="CA", year=2025, bonus=24000, cumulative_bonus=47000),
            Row(department="Finance", state="NY", year=2024, bonus=15000, cumulative_bonus=15000),
            Row(department="Finance", state="NY", year=2025, bonus=19000, cumulative_bonus=34000),
            Row(department="Marketing", state="CA", year=2024, bonus=18000, cumulative_bonus=18000),
            Row(department="Marketing", state="NY", year=2025, bonus=21000, cumulative_bonus=21000),
            Row(department="Sales", state="CA", year=2024, bonus=23000, cumulative_bonus=23000),
            Row(department="Sales", state="NY", year=2024, bonus=10000, cumulative_bonus=10000),
            Row(department="Sales", state="NY", year=2025, bonus=20000, cumulative_bonus=30000),
        ]

    def test_percent_rank(self, spark):
        df = spark.createDataFrame(data=[(1,), (1,), (2,), (3,), (3,), (4,)], schema=["value"])
        w = Window.orderBy("value")
        df = df.withColumn("pr", F.percent_rank().over(w))
        res = df.sort("value").collect()

        assert res == [
            Row(value=1, pr=0.0),
            Row(value=1, pr=0.0),
            Row(value=2, pr=0.4),
            Row(value=3, pr=0.6),
            Row(value=3, pr=0.6),
            Row(value=4, pr=1.0),
        ]

    def test_cume_dist(self, spark):
        df = spark.createDataFrame(data=[(1,), (2,), (3,), (3,), (4,)], schema=["value"])
        w = Window.orderBy("value")
        df = df.withColumn("cd", F.cume_dist().over(w))
        df = df.sort("value")
        res = df.collect()

        assert res == [
            Row(value=1, cd=0.2),
            Row(value=2, cd=0.4),
            Row(value=3, cd=0.8),
            Row(value=3, cd=0.8),
            Row(value=4, cd=1.0),
        ]

    def test_simple_row_number(self, spark):
        df = spark.createDataFrame(
            data=[(2, "A"), (4, "A"), (3, "A"), (2, "B"), (1, "B"), (3, "B")], schema=["value", "grp"]
        )
        w = Window.partitionBy("grp").orderBy("value")
        df = df.withColumn("rn", F.row_number().over(w))
        res = df.sort("grp", "value").collect()

        assert res == [
            Row(value=2, grp="A", rn=1),
            Row(value=3, grp="A", rn=2),
            Row(value=4, grp="A", rn=3),
            Row(value=1, grp="B", rn=1),
            Row(value=2, grp="B", rn=2),
            Row(value=3, grp="B", rn=3),
        ]

    def test_deduplicate_rows(self, spark):
        df = spark.createDataFrame(
            data=[(2, "A"), (4, "A"), (3, "A"), (2, "B"), (1, "B"), (3, "B")], schema=["value", "grp"]
        )
        w = Window.partitionBy(F.col("grp")).orderBy(F.col("value").desc())

        with pytest.raises(
            ContributionsAcceptedError, match=r"Column Expression is not supported in WindowSpec.orderBy yet"
        ):
            df = df.withColumn("rn", F.row_number().over(w))

    def test_moving_average_last_3_points(self, spark):
        data = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
        df = spark.createDataFrame(data=data, schema=["idx", "value"])
        w = Window.orderBy("idx").rowsBetween(2, Window.currentRow)
        df = df.withColumn("ma3", F.avg("value").over(w))
        res = df.sort("idx").collect()

        assert res == [
            Row(idx=1, value=10, ma3=10.0),
            Row(idx=2, value=20, ma3=15.0),
            Row(idx=3, value=30, ma3=20.0),
            Row(idx=4, value=40, ma3=30.0),
            Row(idx=5, value=50, ma3=40.0),
        ]

    def test_range_between(self, spark):
        # rangeBetween uses the ordering column's values; here we include
        # rows within a value distance of 2 up to the current row.
        data = [(1, 10), (2, 20), (3, 30), (4, 40), (6, 60)]
        df = spark.createDataFrame(data=data, schema=["idx", "value"])
        w = Window.orderBy("idx").rangeBetween(2, Window.currentRow)
        df = df.withColumn("ma_range2", F.avg("value").over(w))
        res = df.sort("idx").collect()

        assert res == [
            Row(idx=1, value=10, ma_range2=10.0),
            Row(idx=2, value=20, ma_range2=15.0),
            Row(idx=3, value=30, ma_range2=20.0),
            Row(idx=4, value=40, ma_range2=30.0),
            Row(idx=6, value=60, ma_range2=50.0),
        ]

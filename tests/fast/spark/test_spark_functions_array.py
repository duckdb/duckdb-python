import datetime
import platform
import re

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")
from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException, PySparkTypeError
from spark_namespace.sql import functions as sf
from spark_namespace.sql.types import ArrayType, IntegerType, LongType, Row, StringType, StructField, StructType

import duckdb

if USE_ACTUAL_SPARK:
    import pyspark


def _arrays(spark, rows, *columns):
    return spark.createDataFrame(rows, StructType([StructField(c, ArrayType(LongType())) for c in columns]))


pytestmark = pytest.mark.skipif(
    platform.system() == "Emscripten",
    reason="This Spark experimental test is not supported on Emscripten",
)


class TestSparkFunctionsArray:
    def test_array_distinct(self, spark):
        data = [
            ([1, 2, 2], 2),
            ([2, 4, 5], 3),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])
        df = df.withColumn("distinct_values", sf.array_distinct(sf.col("firstColumn")))
        res = df.select("distinct_values").collect()
        # Output order can vary across platforms which is why we sort it first
        assert len(res) == 2
        assert sorted(res[0].distinct_values) == [1, 2]
        assert sorted(res[1].distinct_values) == [2, 4, 5]

    def test_array_intersect(self, spark):
        data = [
            (["b", "a", "c"], ["c", "d", "a", "f"]),
        ]
        df = spark.createDataFrame(data, ["c1", "c2"])
        df = df.withColumn("intersect_values", sf.array_intersect(sf.col("c1"), sf.col("c2")))
        res = df.select("intersect_values").collect()
        # Output order can vary across platforms which is why we sort it first
        assert len(res) == 1
        assert sorted(res[0].intersect_values) == ["a", "c"]

    def test_array_intersect_of_two_element_types(self, spark):
        df = spark.createDataFrame(
            [([16777216, 16777217, 1, 1, None], [16777216.0, 1.0, None], [1, 2])],
            "i array<int>, f array<float>, b array<bigint>",
        )
        res = df.select(sf.array_intersect("i", "f").alias("v"), sf.array_intersect("i", "b").alias("w"))
        assert res.schema.simpleString() == "struct<v:array<float>,w:array<bigint>>"
        assert res.collect() == [Row(v=[16777216.0, 1.0, None], w=[1])]

    def test_array_union(self, spark):
        data = [
            (["b", "a", "c"], ["c", "d", "a", "f"]),
        ]
        df = spark.createDataFrame(data, ["c1", "c2"])
        df = df.withColumn("union_values", sf.array_union(sf.col("c1"), sf.col("c2")))
        res = df.select("union_values").collect()
        # Output order can vary across platforms which is why we sort it first
        assert len(res) == 1
        assert sorted(res[0].union_values) == ["a", "b", "c", "d", "f"]

    def test_array_max(self, spark):
        data = [
            ([1, 2, 3], 3),
            ([4, 2, 5], 5),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])
        df = df.withColumn("max_value", sf.array_max(sf.col("firstColumn")))
        res = df.select("max_value").collect()
        assert res == [
            Row(max_value=3),
            Row(max_value=5),
        ]

    def test_array_min(self, spark):
        data = [
            ([2, 1, 3], 3),
            ([2, 4, 5], 5),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])
        df = df.withColumn("min_value", sf.array_min(sf.col("firstColumn")))
        res = df.select("min_value").collect()
        assert res == [
            Row(max_value=1),
            Row(max_value=2),
        ]

    def test_get(self, spark):
        df = spark.createDataFrame([(["a", "b", "c"], 1)], ["data", "index"])

        res = df.select(sf.get(df.data, 1).alias("r")).collect()
        assert res == [Row(r="b")]

        res = df.select(sf.get(df.data, -1).alias("r")).collect()
        assert res == [Row(r=None)]

        res = df.select(sf.get(df.data, 3).alias("r")).collect()
        assert res == [Row(r=None)]

        res = df.select(sf.get(df.data, "index").alias("r")).collect()
        assert res == [Row(r="b")]

        res = df.select(sf.get(df.data, sf.col("index") - 1).alias("r")).collect()
        assert res == [Row(r="a")]

    def test_get_below_zero_is_null(self, spark):
        df = spark.createDataFrame(
            [(["a", "b", "c"], -2), (["a", "b", "c"], -4), (["a", "b", "c"], None)], ["data", "i"]
        )
        res = df.select(sf.get("data", -2).alias("l"), sf.get("data", -3).alias("m"), sf.get("data", "i").alias("c"))
        assert res.collect() == [Row(l=None, m=None, c=None)] * 3

    def test_flatten(self, spark):
        df = spark.createDataFrame([([[1, 2, 3], [4, 5], [6]],), ([None, [4, 5]],)], ["data"])

        res = df.select(sf.flatten(df.data).alias("r")).collect()
        assert res == [Row(r=[1, 2, 3, 4, 5, 6]), Row(r=None)]

    def test_array_compact(self, spark):
        df = spark.createDataFrame([([1, None, 2, 3],), ([4, 5, None, 4],)], ["data"])

        res = df.select(sf.array_compact(df.data).alias("v")).collect()
        assert res == [Row(v=[1, 2, 3]), Row(v=[4, 5, 4])]

    def test_array_remove(self, spark):
        df = spark.createDataFrame([([1, 2, 3, 1, 1],), ([],)], ["data"])

        res = df.select(sf.array_remove(df.data, 1).alias("v")).collect()
        assert res == [Row(v=[2, 3]), Row(v=[])]

    def test_array_agg(self, spark):
        df = spark.createDataFrame([[1, "A"], [1, "A"], [2, "A"]], ["c", "group"])

        res = df.groupBy("group").agg(sf.array_agg("c").alias("r")).collect()
        assert res[0] == Row(group="A", r=[1, 1, 2])

    def test_collect_list(self, spark):
        df = spark.createDataFrame([[1, "A"], [1, "A"], [2, "A"]], ["c", "group"])

        res = df.groupBy("group").agg(sf.collect_list("c").alias("r")).collect()
        assert res[0] == Row(group="A", r=[1, 1, 2])

    def test_array_append(self, spark):
        df = spark.createDataFrame([Row(c1=["b", "a", "c"], c2="c")], ["c1", "c2"])

        res = df.select(sf.array_append(df.c1, df.c2).alias("r")).collect()
        assert res == [Row(r=["b", "a", "c", "c"])]

        res = df.select(sf.array_append(df.c1, "x")).collect()
        assert res == [Row(r=["b", "a", "c", "x"])]

    def test_array_insert(self, spark):
        df = spark.createDataFrame(
            [(["a", "b", "c"], 2, "d"), (["a", "b", "c", "e"], 2, "d"), (["c", "b", "a"], -2, "d")],
            ["data", "pos", "val"],
        )

        res = df.select(sf.array_insert(df.data, df.pos.cast("integer"), df.val).alias("data")).collect()
        assert res == [
            Row(data=["a", "d", "b", "c"]),
            Row(data=["a", "d", "b", "c", "e"]),
            Row(data=["c", "b", "d", "a"]),
        ]

        res = df.select(sf.array_insert(df.data, 5, "hello").alias("data")).collect()
        assert res == [
            Row(data=["a", "b", "c", None, "hello"]),
            Row(data=["a", "b", "c", "e", "hello"]),
            Row(data=["c", "b", "a", None, "hello"]),
        ]

        res = df.select(sf.array_insert(df.data, -5, "hello").alias("data")).collect()
        assert res == [
            Row(data=["hello", None, "a", "b", "c"]),
            Row(data=["hello", "a", "b", "c", "e"]),
            Row(data=["hello", None, "c", "b", "a"]),
        ]

    def test_slice(self, spark):
        df = spark.createDataFrame([([1, 2, 3],), ([4, 5],)], ["x"])
        res = df.select(sf.slice(df.x, 2, 2).alias("sliced")).collect()
        assert res == [Row(sliced=[2, 3]), Row(sliced=[5])]

    def test_sort_array(self, spark):
        df = spark.createDataFrame([([2, 1, None, 3],), ([1],), ([],)], ["data"])

        res = df.select(sf.sort_array(df.data).alias("r")).collect()
        assert res == [Row(r=[None, 1, 2, 3]), Row(r=[1]), Row(r=[])]

        res = df.select(sf.sort_array(df.data, asc=False).alias("r")).collect()
        assert res == [Row(r=[3, 2, 1, None]), Row(r=[1]), Row(r=[])]

    @pytest.mark.parametrize(("null_replacement", "expected_joined_2"), [(None, "a"), ("replaced", "a,replaced")])
    def test_array_join(self, spark, null_replacement, expected_joined_2):
        df = spark.createDataFrame([(["a", "b", "c"],), (["a", None],)], ["data"])

        res = df.select(sf.array_join(df.data, ",", null_replacement=null_replacement).alias("joined")).collect()
        assert res == [Row(joined="a,b,c"), Row(joined=expected_joined_2)]

    def test_array_position(self, spark):
        df = spark.createDataFrame([(["c", "b", "a"],), ([],)], ["data"])

        res = df.select(sf.array_position(df.data, "a").alias("pos")).collect()
        assert res == [Row(pos=3), Row(pos=0)]

    def test_array_preprend(self, spark):
        df = spark.createDataFrame([([2, 3, 4],), ([],)], ["data"])

        res = df.select(sf.array_prepend(df.data, 1).alias("pre")).collect()
        assert res == [Row(pre=[1, 2, 3, 4]), Row(pre=[1])]

    def test_array_repeat(self, spark):
        df = spark.createDataFrame([("ab",)], ["data"])

        res = df.select(sf.array_repeat(df.data, 3).alias("r")).collect()
        assert res == [Row(r=["ab", "ab", "ab"])]

    def test_array_size(self, spark):
        df = spark.createDataFrame([([2, 1, 3],), (None,)], ["data"])

        res = df.select(sf.array_size(df.data).alias("r")).collect()
        assert res == [Row(r=3), Row(r=None)]

    def test_array_sort(self, spark):
        df = spark.createDataFrame([([2, 1, None, 3],), ([1],), ([],)], ["data"])

        res = df.select(sf.array_sort(df.data).alias("r")).collect()
        assert res == [Row(r=[1, 2, 3, None]), Row(r=[1]), Row(r=[])]

    def test_array_sort_name(self, spark):
        (name,) = spark.createDataFrame([([2, 1],)], ["data"]).select(sf.array_sort("data")).columns
        if USE_ACTUAL_SPARK:
            # Spark names the call with the text of its default comparator.
            assert name.startswith("array_sort(data, lambdafunction(")
        else:
            assert name == "array_sort(data)"

    def test_arrays_overlap(self, spark):
        df = spark.createDataFrame(
            [(["a", "b"], ["b", "c"]), (["a"], ["b", "c"]), ([None, "c"], ["a"]), ([None, "c"], [None])], ["x", "y"]
        )

        res = df.select(sf.arrays_overlap(df.x, df.y).alias("overlap")).collect()
        assert res == [Row(overlap=True), Row(overlap=False), Row(overlap=None), Row(overlap=None)]

    def test_arrays_zip(self, spark):
        df = spark.createDataFrame([([1, 2, 3], [2, 4, 6], [3, 6])], ["vals1", "vals2", "vals3"])

        res = df.select(sf.arrays_zip(df.vals1, df.vals2, df.vals3).alias("zipped")).collect()
        # TODO: The structure of the results should be the same  # noqa: TD002, TD003
        if USE_ACTUAL_SPARK:
            assert res == [
                Row(
                    zipped=[
                        Row(vals1=1, vals2=2, vals3=3),
                        Row(vals1=2, vals2=4, vals3=6),
                        Row(vals1=3, vals2=6, vals3=None),
                    ]
                )
            ]
        else:
            assert res == [Row(zipped=[(1, 2, 3), (2, 4, 6), (3, 6, None)])]

    def test_set_operations_keep_order_and_nulls(self, spark):
        df = _arrays(spark, [([1, None, 2, 2, 3], [3, None, 2, 2]), ([2, 5, 2], None)], "a", "b")
        res = df.select(
            sf.array_distinct("a").alias("distinct"),
            sf.array_union("a", "b").alias("union"),
            sf.array_union("b", "a").alias("union_ba"),
            sf.array_intersect("a", "b").alias("intersect"),
        ).collect()
        assert res == [
            Row(distinct=[1, None, 2, 3], union=[1, None, 2, 3], union_ba=[3, None, 2, 1], intersect=[None, 2, 3]),
            Row(distinct=[2, 5], union=None, union_ba=None, intersect=None),
        ]

    def test_array_max_min_skip_nulls(self, spark):
        df = _arrays(spark, [([3, 1, None, 2],), ([None],), (None,)], "a")
        res = df.select(sf.array_max("a").alias("max"), sf.array_min("a").alias("min")).collect()
        assert res == [Row(max=3, min=1), Row(max=None, min=None), Row(max=None, min=None)]

    def test_array_contains_with_nulls(self, spark):
        df = _arrays(spark, [([1, None],), ([1],), ([None],), (None,)], "a")
        res = df.select(sf.array_contains("a", 1).alias("one"), sf.array_contains("a", 2).alias("two")).collect()
        assert res == [
            Row(one=True, two=None),
            Row(one=True, two=False),
            Row(one=None, two=None),
            Row(one=None, two=None),
        ]

    def test_array_position_of_null_array(self, spark):
        df = _arrays(spark, [([1, None, 3],), (None,)], "a")
        res = df.select(sf.array_position("a", 3).alias("p"), sf.array_position("a", 5).alias("q"))
        assert res.schema.fields[0].dataType == LongType()
        assert res.collect() == [Row(p=3, q=0), Row(p=None, q=None)]

    def test_array_remove_keeps_nulls(self, spark):
        df = _arrays(spark, [([1, None, 2, 1],)], "a")
        assert df.select(sf.array_remove("a", 1).alias("r")).collect() == [Row(r=[None, 2])]

    def test_array_insert_positions(self, spark):
        df = _arrays(spark, [([1, 2, 3],), ([],), (None,)], "a")
        positions = {
            "first": 1,
            "middle": 3,
            "append": 4,
            "pad": 6,
            "last": -1,
            "before_last": -2,
            "front": -4,
            "pad_front": -5,
        }
        res = df.select(*[sf.array_insert("a", p, sf.lit(9)).alias(name) for name, p in positions.items()]).collect()
        assert res == [
            Row(
                first=[9, 1, 2, 3],
                middle=[1, 2, 9, 3],
                append=[1, 2, 3, 9],
                pad=[1, 2, 3, None, None, 9],
                last=[1, 2, 3, 9],
                before_last=[1, 2, 9, 3],
                front=[9, 1, 2, 3],
                pad_front=[9, None, 1, 2, 3],
            ),
            Row(
                first=[9],
                middle=[None, None, 9],
                append=[None, None, None, 9],
                pad=[None, None, None, None, None, 9],
                last=[9],
                before_last=[9, None],
                front=[9, None, None, None],
                pad_front=[9, None, None, None, None],
            ),
            Row(
                first=None, middle=None, append=None, pad=None, last=None, before_last=None, front=None, pad_front=None
            ),
        ]
        with pytest.raises(Exception, match="INVALID_INDEX_OF_ZERO"):
            df.select(sf.array_insert("a", 0, sf.lit(9))).collect()

    def test_slice_bounds(self, spark):
        df = _arrays(spark, [([1, 2, 3, 4],), (None,)], "a")
        res = df.select(
            sf.slice("a", 2, 2).alias("mid"),
            sf.slice("a", -2, 2).alias("tail"),
            sf.slice("a", 1, 10).alias("long"),
            sf.slice("a", -5, 2).alias("before"),
            sf.slice("a", 3, 0).alias("empty"),
            sf.slice("a", 5, 1).alias("past"),
        ).collect()
        assert res == [
            Row(mid=[2, 3], tail=[3, 4], long=[1, 2, 3, 4], before=[], empty=[], past=[]),
            Row(mid=None, tail=None, long=None, before=None, empty=None, past=None),
        ]
        with pytest.raises(Exception, match="start"):
            df.select(sf.slice("a", 0, 2)).collect()
        with pytest.raises(Exception, match="length"):
            df.select(sf.slice("a", 1, -1)).collect()

    def test_sequence_counts_down(self, spark):
        res = spark.range(1).select(
            sf.sequence(sf.lit(5), sf.lit(1)).alias("down"),
            sf.sequence(sf.lit(1), sf.lit(3)).alias("up"),
            sf.sequence(sf.lit(2), sf.lit(2)).alias("one"),
            sf.sequence(sf.lit(1), sf.lit(9), sf.lit(3)).alias("step"),
        )
        assert res.collect() == [Row(down=[5, 4, 3, 2, 1], up=[1, 2, 3], one=[2], step=[1, 4, 7])]

    @pytest.mark.parametrize(
        ("start", "stop", "step"),
        [
            ("a", "b", "-1"),
            ("b", "a", "1"),
            ("a", "b", "0"),
            ("d1", "d2", "INTERVAL '-1' DAY"),
        ],
    )
    def test_sequence_step_away_from_stop_raises(self, spark, start, stop, step):
        df = spark.createDataFrame(
            [(1, 5, datetime.date(2024, 1, 1), datetime.date(2024, 1, 5))], ["a", "b", "d1", "d2"]
        )
        if USE_ACTUAL_SPARK:
            from py4j.protocol import Py4JJavaError
            from spark_namespace.errors import IllegalArgumentException

            errors = (IllegalArgumentException, Py4JJavaError)
        else:
            errors = duckdb.InvalidInputException
        with pytest.raises(errors, match="Illegal sequence boundaries"):
            df.select(sf.sequence(start, stop, sf.expr(step))).collect()

    def test_sequence_element_types(self, spark):
        day = datetime.date
        schema = "i int, j int, s smallint, b bigint, d1 date, d2 date"
        df = spark.createDataFrame([(1, 3, 2, 3, day(2024, 1, 1), day(2024, 1, 3))], schema)
        res = df.select(
            sf.sequence("i", "j").alias("ints"),
            sf.sequence(sf.lit(3), sf.lit(1)).alias("literals"),
            sf.sequence("s", sf.lit(4).cast("smallint"), sf.lit(1).cast("smallint")).alias("smallints"),
            sf.sequence("i", "b").alias("widened"),
            sf.sequence("b", sf.lit(1), sf.lit(-1)).alias("narrow_step"),
            sf.sequence("d1", "d2").alias("days"),
            sf.sequence("d2", "d1").alias("days_down"),
            sf.sequence("d1", "d2", sf.lit(datetime.timedelta(days=2))).alias("every_other_day"),
        )
        assert res.schema.simpleString() == (
            "struct<ints:array<int>,literals:array<int>,smallints:array<smallint>,widened:array<bigint>,"
            "narrow_step:array<bigint>,days:array<date>,days_down:array<date>,every_other_day:array<date>>"
        )
        assert res.collect() == [
            Row(
                ints=[1, 2, 3],
                literals=[3, 2, 1],
                smallints=[2, 3, 4],
                widened=[1, 2, 3],
                narrow_step=[3, 2, 1],
                days=[day(2024, 1, 1), day(2024, 1, 2), day(2024, 1, 3)],
                days_down=[day(2024, 1, 3), day(2024, 1, 2), day(2024, 1, 1)],
                every_other_day=[day(2024, 1, 1), day(2024, 1, 3)],
            )
        ]

    def test_sequence_of_small_integers_spans_their_range(self, spark):
        df = spark.createDataFrame([(-100, 100, -30000, 30000)], "a tinyint, b tinyint, c smallint, d smallint")
        res = df.select(sf.sequence("a", "b").alias("tiny"), sf.sequence("c", "d").alias("small"))
        assert res.schema.simpleString() == "struct<tiny:array<tinyint>,small:array<smallint>>"
        row = res.collect()[0]
        assert row.tiny == list(range(-100, 101))
        assert row.small == list(range(-30000, 30001))

    def test_sequence_of_months(self, spark):
        day = datetime.date
        df = spark.createDataFrame([(day(2024, 1, 1), day(2024, 1, 31), day(2024, 4, 30))], "a date, b date, c date")
        months = df.select(sf.sequence("a", "c", sf.expr("INTERVAL 1 MONTH")).alias("m"))
        from_the_end = df.select(sf.sequence("b", "c", sf.expr("INTERVAL 1 MONTH")).alias("m"))
        years = df.select(sf.sequence("a", "c", sf.expr("INTERVAL 1 YEAR")).alias("m"))
        if USE_ACTUAL_SPARK:
            assert months.collect() == [Row(m=[day(2024, 1, 1), day(2024, 2, 1), day(2024, 3, 1), day(2024, 4, 1)])]
            spark_months = [day(2024, 1, 31), day(2024, 2, 29), day(2024, 3, 31), day(2024, 4, 30)]
            assert from_the_end.collect() == [Row(m=spark_months)]
            assert years.collect() == [Row(m=[day(2024, 1, 1)])]
        else:
            # DuckDB adds a step of months to each element in turn, which gives 2024-03-29 after 2024-02-29 where
            # Spark adds multiples of it to start, so the module raises for any step with months or years.
            for frame in [months, from_the_end, years]:
                with pytest.raises(duckdb.InvalidInputException, match="step of months"):
                    frame.collect()

    def test_sequence_of_timestamps(self, spark):
        moment = datetime.datetime
        df = spark.createDataFrame([(moment(2024, 1, 1), moment(2024, 1, 2))], "t1 timestamp, t2 timestamp")
        res = df.select(sf.sequence("t1", "t2", sf.lit(datetime.timedelta(hours=12))).alias("t"))
        assert res.schema.simpleString() == "struct<t:array<timestamp>>"
        assert res.collect() == [Row(t=[moment(2024, 1, 1), moment(2024, 1, 1, 12), moment(2024, 1, 2)])]

    def test_sequence_with_a_null_bound_is_null(self, spark):
        df = spark.createDataFrame([(None, 3, 0), (1, None, 0), (None, None, 0), (2, 2, 0)], "a int, b int, s int")
        res = df.select(sf.sequence("a", "b", "s").alias("zero_step"), sf.sequence("a", "b").alias("no_step"))
        assert res.collect() == [
            Row(zero_step=None, no_step=None),
            Row(zero_step=None, no_step=None),
            Row(zero_step=None, no_step=None),
            Row(zero_step=[2], no_step=[2]),
        ]
        dates = spark.createDataFrame([(None, datetime.date(2024, 1, 3))], "a date, b date")
        assert dates.select(sf.sequence("a", "b").alias("d")).collect() == [Row(d=None)]

    def test_sequence_of_dates_by_part_of_a_day_raises(self, spark):
        df = spark.createDataFrame([(datetime.date(2024, 1, 1), datetime.date(2024, 1, 3))], "d1 date, d2 date")
        if USE_ACTUAL_SPARK:
            from py4j.protocol import Py4JJavaError
            from spark_namespace.errors import IllegalArgumentException

            errors = (IllegalArgumentException, Py4JJavaError)
        else:
            errors = duckdb.InvalidInputException
        with pytest.raises(errors, match="day granularity"):
            df.select(sf.sequence("d1", "d2", sf.lit(datetime.timedelta(hours=12)))).collect()

    def test_sequence_with_a_step_of_a_wider_type(self, spark):
        df = spark.createDataFrame([(1, 3)], "i int, j int")
        res = df.select(sf.sequence("i", "j", sf.lit(1).cast("bigint")).alias("s"))
        if USE_ACTUAL_SPARK:
            assert res.schema.simpleString() == "struct<s:array<bigint>>"
            assert res.collect() == [Row(s=[1, 2, 3])]
        else:
            # Spark gives the elements the step's type, which a cast to the type of start and stop would lose.
            with pytest.raises(duckdb.InvalidInputException, match="wider integral type"):
                res.collect()

    def test_sequence_with_a_zero_step_of_a_wider_type(self, spark):
        df = spark.createDataFrame([(1, 0)], "i int, z bigint")
        for step in ("z", sf.lit(0).cast("bigint")):
            res = df.select(sf.sequence("i", "i", step).alias("s"))
            if USE_ACTUAL_SPARK:
                assert res.schema.simpleString() == "struct<s:array<bigint>>"
                assert res.collect() == [Row(s=[1])]
            else:
                with pytest.raises(duckdb.InvalidInputException, match="wider integral type"):
                    res.collect()

    def test_sequence_without_steps_to_take(self, spark):
        df = spark.createDataFrame(
            [(1, None)], StructType([StructField("a", LongType()), StructField("n", LongType())])
        )
        res = df.select(
            sf.sequence("a", "a", sf.lit(0)).alias("zero"),
            sf.sequence("a", "a", sf.lit(-1)).alias("away"),
            sf.sequence("a", sf.lit(5), "n").alias("null_step"),
        )
        assert res.collect() == [Row(zero=[1], away=[1], null_step=None)]

    def test_array_size_type(self, spark):
        df = _arrays(spark, [([1, 2],), (None,)], "a")
        res = df.select(sf.array_size("a").alias("n"))
        assert res.schema.fields[0].dataType == IntegerType()
        assert res.collect() == [Row(n=2), Row(n=None)]

    def test_array_join_of_null_elements_only(self, spark):
        df = spark.createDataFrame([([],), ([None],), ([None, None],), (None,)], "a array<string>")
        res = df.select(sf.array_join("a", ",").alias("j"), sf.array_join("a", ",", "N").alias("r")).collect()
        assert res == [Row(j="", r=""), Row(j="", r="N"), Row(j="", r="N,N"), Row(j=None, r=None)]

    def test_array_join_null_replacement_on_numbers(self, spark):
        df = _arrays(spark, [([1, None, 2],), (None,)], "a")
        res = df.select(sf.array_join("a", "-").alias("j"), sf.array_join("a", "-", "N").alias("r")).collect()
        assert res == [Row(j="1-2", r="1-N-2"), Row(j=None, r=None)]

    def test_array_repeat_non_positive_count(self, spark):
        schema = StructType([StructField("s", StringType()), StructField("n", IntegerType())])
        df = spark.createDataFrame([("x", 2), ("y", 0), ("z", -1)], schema)
        res = df.select(sf.array_repeat("s", "n").alias("r"), sf.array_repeat("s", -2).alias("c")).collect()
        assert res == [Row(r=["x", "x"], c=[]), Row(r=[], c=[]), Row(r=[], c=[])]

    def test_append_prepend_null_array(self, spark):
        df = _arrays(spark, [(None,), ([1],)], "a")
        res = df.select(sf.array_append("a", sf.lit(9)).alias("app"), sf.array_prepend("a", 0).alias("pre")).collect()
        assert res == [Row(app=None, pre=None), Row(app=[1, 9], pre=[0, 1])]

    def test_null_arguments_give_null(self, spark):
        schema = StructType(
            [
                StructField("a", ArrayType(LongType())),
                StructField("n", ArrayType(LongType())),
                StructField("i", IntegerType()),
            ]
        )
        df = spark.createDataFrame([([1, 2, 3, 2], None, None)], schema)
        res = df.select(
            sf.array_position("a", None).alias("position"),
            sf.array_remove("a", None).alias("remove"),
            sf.array_repeat(sf.lit(5), "i").alias("repeat"),
            sf.array_insert("a", "i", sf.lit(9)).alias("insert"),
            sf.array_insert("n", 1, sf.lit(9)).alias("insert_into_null"),
            sf.slice("n", 0, 2).alias("slice_from_zero"),
            sf.slice("a", "i", sf.lit(-1)).alias("slice_null_start"),
            sf.slice("a", sf.lit(0), "i").alias("slice_null_length"),
            sf.arrays_overlap("a", "n").alias("overlap"),
            sf.arrays_overlap("n", sf.array().cast("array<bigint>")).alias("overlap_empty"),
            sf.arrays_zip("a", "n").alias("zip"),
            sf.array_intersect("n", "a").alias("intersect"),
        )
        assert res.collect() == [Row(*[None] * 12)]

    def test_null_elements_are_kept(self, spark):
        schema = StructType([StructField("a", ArrayType(LongType())), StructField("i", IntegerType())])
        df = spark.createDataFrame([([1, 2], 2)], schema)
        res = df.select(
            sf.array_insert("a", "i", sf.lit(None).cast("bigint")).alias("insert"),
            sf.array_append("a", None).alias("append"),
            sf.array_prepend("a", None).alias("prepend"),
            sf.array_repeat(sf.lit(None).cast("bigint"), "i").alias("repeat"),
        )
        assert res.collect() == [
            Row(insert=[1, None, 2], append=[1, 2, None], prepend=[None, 1, 2], repeat=[None, None]),
        ]

    def test_arrays_zip_field_names(self, spark):
        schema = StructType([StructField("a", ArrayType(LongType())), StructField("b", ArrayType(StringType()))])
        df = spark.createDataFrame([([1, 2], ["x"])], schema)
        res = df.select(
            sf.arrays_zip("a", "b").alias("z"), sf.arrays_zip(sf.col("a"), sf.array_compact("b")).alias("w")
        )
        assert [f.dataType.elementType.fieldNames() for f in res.schema.fields] == [["a", "b"], ["a", "1"]]
        assert res.collect() == [Row(z=[Row(a=1, b="x"), Row(a=2, b=None)], w=[Row(1, "x"), Row(2, None)])]

    def test_array_distinct_evaluates_its_input_once(self, spark):
        rows = spark.range(20).select(sf.array_distinct(sf.array(sf.rand(), sf.lit(2.0))).alias("a")).collect()
        assert {len(r.a) for r in rows} == {2}

    def test_arrays_zip_of_no_arrays(self, spark):
        res = spark.range(1).select(sf.arrays_zip())
        assert res.schema.simpleString() == "struct<arrays_zip():array<struct<>>>"
        assert res.collect() == [Row([])]

    def test_array_contains_null_literal_raises(self, spark):
        df = _arrays(spark, [([1, 2],)], "a")
        for value in [None, sf.lit(None)]:
            with pytest.raises(AnalysisException, match=re.escape("[DATATYPE_MISMATCH.NULL_TYPE]")):
                df.select(sf.array_contains("a", value)).collect()
        assert df.select(sf.array_contains("a", sf.lit(None).cast("bigint")).alias("c")).collect() == [Row(c=None)]

    @pytest.mark.parametrize(("first", "second"), [("x", "i"), ("x0", "x1"), ("x1", "x0"), ("X0", "row")])
    def test_columns_named_like_lambda_parameters(self, spark, first, second):
        df = _arrays(spark, [([1, 2, 2, None], [2, 3, None])], first, second)
        res = df.select(
            sf.array_distinct(first).alias("distinct"),
            sf.array_intersect(first, second).alias("intersect"),
            sf.array_union(first, second).alias("union"),
            sf.array_compact(first).alias("compact"),
            sf.array_join(first, ",", "-").alias("join"),
            sf.arrays_zip(first, second).alias("zip"),
        )
        assert res.collect() == [
            Row(
                distinct=[1, 2, None],
                intersect=[2, None],
                union=[1, 2, None, 3],
                compact=[1, 2, 2],
                join="1,2,2,-",
                zip=[Row(1, 2), Row(2, 3), Row(2, None), Row(None, None)],
            )
        ]
        removed = spark.createDataFrame([([1, 2, 3], 2)], f"{first} array<bigint>, {second} bigint")
        if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3."):
            # PySpark 3.5 takes only a literal element.
            with pytest.raises(PySparkTypeError, match="NOT_ITERABLE"):
                removed.select(sf.array_remove(first, sf.col(second)))
        else:
            assert removed.select(sf.array_remove(first, sf.col(second)).alias("r")).collect() == [Row(r=[1, 3])]
        grouped = spark.createDataFrame([(1,), (None,), (2,)], f"{first} bigint")
        assert sorted(grouped.groupBy().agg(sf.collect_list(first).alias("l")).collect()[0].l) == [1, 2]

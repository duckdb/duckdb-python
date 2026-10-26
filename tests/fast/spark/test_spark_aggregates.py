import datetime
import math
from decimal import Decimal

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException, error_condition
from spark_namespace.sql import functions as F
from spark_namespace.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    Row,
    ShortType,
    StringType,
    StructField,
    StructType,
)


def _ansi(spark):
    return USE_ACTUAL_SPARK and spark.conf.get("spark.sql.ansi.enabled") == "true"


@pytest.fixture
def numbers(spark):
    schema = StructType(
        [
            StructField("g", StringType()),
            StructField("i", IntegerType()),
            StructField("l", LongType()),
            StructField("s", ShortType()),
            StructField("d", DoubleType()),
        ]
    )
    rows = [("a", 1, 10, 1, 1.0), ("a", 2, None, 2, 2.0), ("b", 4, 40, 4, 4.0), ("b", None, 50, None, None)]
    return spark.createDataFrame(rows, schema)


class TestSparkAggregates:
    def test_select_global_aggregates(self, numbers):
        res = numbers.select(
            F.sum("i").alias("sum"),
            F.count("i").alias("count"),
            F.avg("d").alias("avg"),
            (F.max("l") * 2 + F.min("i")).alias("expr"),
            F.round(F.avg("l"), 1).alias("rounded"),
            F.when(F.count("l") > 2, "many").otherwise("few").alias("when"),
        ).collect()
        assert res == [Row(sum=7, count=3, avg=pytest.approx(7 / 3), expr=101, rounded=33.3, when="many")]

    def test_select_aggregate_of_empty_input(self, numbers):
        res = (
            numbers.filter(F.col("i") > 100)
            .select(
                F.sum("i").alias("sum"),
                F.count("i").alias("count"),
                F.max("d").alias("max"),
                F.coalesce(F.sum("l"), F.lit(0)).alias("zero"),
                F.collect_list("i").alias("list"),
            )
            .collect()
        )
        assert res == [Row(sum=None, count=0, max=None, zero=0, list=[])]

    def test_global_groupby_agg(self, numbers):
        assert numbers.groupBy().agg(F.sum("l").alias("sum")).collect() == [Row(sum=100)]

    def test_constants_next_to_aggregates(self, numbers):
        res = numbers.select(F.sum("i"), F.lit(1), (F.lit(2) * 3).alias("six"), F.abs(F.lit(-4)), F.lit("x"))
        assert res.columns == ["sum(i)", "1", "six", "abs(-4)", "x"]
        assert [tuple(r) for r in res.collect()] == [(7, 1, 6, 4, "x")]
        res = numbers.groupBy().agg(F.max("i"), F.lit(None).alias("n"))
        assert [tuple(r) for r in res.collect()] == [(4, None)]

    def test_string_arguments_that_are_not_columns_keep_a_constant(self, numbers):
        res = numbers.select(
            F.sum("i"), F.lpad(F.lit("x"), 3, "-"), F.concat_ws("-", F.lit("a"), F.lit("b")), F.instr(F.lit("ab"), "b")
        )
        assert res.columns == ["sum(i)", "lpad(x, 3, -)", "concat_ws(-, a, b)", "instr(ab, b)"]
        assert [tuple(r) for r in res.collect()] == [(7, "--x", "a-b", 2)]

    def test_call_function_of_an_aggregate(self, numbers):
        res = numbers.groupBy().agg(F.call_function("sum", "l"), F.call_function("max", F.col("i")))
        assert res.columns == ["sum(l)", "max(i)"]
        assert res.schema.simpleString() == "struct<sum(l):bigint,max(i):int>"
        assert res.collect() == [Row(100, 4)]
        assert numbers.select(F.call_function("count", "i")).collect() == [Row(3)]
        grouped = numbers.groupBy("g").agg(F.call_function("avg", "d").alias("avg")).orderBy("g").collect()
        assert grouped == [Row(g="a", avg=1.5), Row(g="b", avg=4.0)]

    @pytest.mark.parametrize(
        "make",
        [
            lambda df: df.select(F.sum("i"), "i"),
            lambda df: df.select(F.sum("i"), F.col("i")),
            lambda df: df.groupBy().agg(F.col("i")),
            lambda df: df.groupBy().agg(F.sum("i"), F.col("i")),
            lambda df: df.select(F.sum("i"), F.col("i") + F.lit(1)),
            lambda df: df.select(F.sum("i"), F.abs("i")),
            lambda df: df.select(F.sum("i"), F.lpad("g", 3, "-")),
            lambda df: df.select(F.sum("i"), F.expr("g")),
            lambda df: df.select(F.sum("i"), F.struct({"g"})),
            lambda df: df.select(F.sum("i"), F.array({"g"})),
        ],
    )
    def test_aggregate_mixed_with_plain_column_raises(self, numbers, make):
        with pytest.raises(AnalysisException, match="MISSING_GROUP_BY"):
            make(numbers).collect()

    def test_sum_of_integers_is_bigint(self, numbers):
        df = numbers.groupBy("g").agg(F.sum("i"), F.sum("l"), F.sum("s"), F.sum("d"), F.sum("i") * 2).orderBy("g")
        assert [f.dataType for f in df.schema.fields] == [
            StringType(),
            LongType(),
            LongType(),
            LongType(),
            DoubleType(),
            LongType(),
        ]
        assert df.collect() == [Row("a", 3, 10, 3, 3.0, 6), Row("b", 4, 90, 4, 4.0, 8)]
        global_sum = numbers.select(F.sum("i"))
        assert global_sum.schema.fields[0].dataType == LongType()

    def test_grouped_data_sum_is_bigint(self, numbers):
        df = numbers.groupBy("g").sum("i", "d").orderBy("g")
        assert [f.dataType for f in df.schema.fields] == [StringType(), LongType(), DoubleType()]
        assert df.collect() == [Row("a", 3, 3.0), Row("b", 4, 4.0)]

    def test_sum_overflow(self, spark):
        df = spark.createDataFrame([(9223372036854775807,), (1,)], StructType([StructField("v", LongType())]))
        if USE_ACTUAL_SPARK and not _ansi(spark):
            assert df.select(F.sum("v")).collect() == [Row(-9223372036854775808)]
        else:
            with pytest.raises(Exception, match=r"(?i)overflow|out of range"):
                df.select(F.sum("v")).collect()

    def test_sum_of_int_past_int_range(self, spark):
        df = spark.createDataFrame([(2147483647,), (1,)], StructType([StructField("v", IntegerType())]))
        assert df.select(F.sum("v").alias("s")).collect() == [Row(s=2147483648)]

    def test_aggregate_result_types(self, numbers):
        df = numbers.select(
            F.avg("i"), F.mean("l"), F.count("i"), F.count("*"), F.min("i"), F.max("l"), F.stddev("i"), F.var_pop("i")
        )
        types = [f.dataType for f in df.schema.fields]
        assert types == [
            DoubleType(),
            DoubleType(),
            LongType(),
            LongType(),
            IntegerType(),
            LongType(),
            DoubleType(),
            DoubleType(),
        ]

    def test_count_star_and_literal(self, numbers):
        res = (
            numbers.groupBy("g")
            .agg(F.count("*").alias("star"), F.count(F.lit(1)).alias("one"), F.count("i").alias("i"))
            .orderBy("g")
            .collect()
        )
        assert res == [Row(g="a", star=2, one=2, i=2), Row(g="b", star=2, one=2, i=1)]
        assert numbers.select(F.count("*").alias("n")).collect() == [Row(n=4)]
        assert numbers.select(F.count(F.lit(None)).alias("n")).collect() == [Row(n=0)]

    def test_count_star_names(self, numbers):
        res = numbers.select(F.count("*"), F.count(F.col("*")), F.count(F.lit(1)))
        assert res.columns == ["count(1)", "count(1)", "count(1)"]
        assert [tuple(r) for r in res.collect()] == [(4, 4, 4)]
        assert numbers.groupBy("g").agg(F.count("*")).columns == ["g", "count(1)"]

    def test_unaliased_struct_names(self, numbers):
        res = numbers.select(F.struct("g", "i"), F.struct(["g", numbers.i]), F.struct(F.col("i").alias("z")))
        assert res.columns == ["struct(g, i)", "struct(g, i)", "struct(i AS z)"]

    def test_skewness_population(self, spark):
        schema = StructType([StructField("g", StringType()), StructField("x", DoubleType())])
        rows = [
            ("a", 1.0),
            ("a", 1.0),
            ("a", 2.0),
            ("a", 10.0),
            ("b", 1.0),
            ("b", 3.0),
            ("c", 3.0),
            ("c", 3.0),
            ("d", 5.0),
        ]
        res = spark.createDataFrame(rows, schema).groupBy("g").agg(F.skewness("x").alias("v")).orderBy("g").collect()
        assert res[0].v == pytest.approx(1.1153956690)
        assert res[1].v == pytest.approx(0.0)
        assert res[2].v is None
        assert res[3].v is None

    @pytest.mark.parametrize(
        "values",
        [[float("nan"), 1.0, 2.0], [float("nan"), 1.0], [float("nan")], [float("inf"), 1.0, 2.0], [-float("inf"), 1.0]],
    )
    def test_skewness_of_nan_or_infinity_is_nan(self, spark, values):
        df = spark.createDataFrame([(v,) for v in values], StructType([StructField("x", DoubleType())]))
        assert math.isnan(df.select(F.skewness("x").alias("v")).collect()[0].v)

    def test_corr_without_variance(self, spark):
        schema = StructType([StructField("x", DoubleType()), StructField("y", DoubleType())])
        df = spark.createDataFrame([(1.0, 1.0), (1.0, 2.0)], schema)
        assert spark.createDataFrame([(1.0, 1.0)], schema).select(F.corr("x", "y").alias("c")).collect() == [
            Row(c=None)
        ]
        if _ansi(spark):
            with pytest.raises(Exception, match="DIVIDE_BY_ZERO"):
                df.select(F.corr("x", "y")).collect()
        else:
            assert df.select(F.corr("x", "y").alias("c")).collect() == [Row(c=None)]
        varied = spark.createDataFrame([(1.0, 2.0), (2.0, 4.5), (4.0, 7.0)], schema)
        assert varied.select(F.corr("x", "y").alias("c")).collect()[0].c == pytest.approx(0.9819805061)

    def test_collect_list_skips_nulls(self, spark):
        schema = StructType([StructField("g", StringType()), StructField("v", LongType())])
        df = spark.createDataFrame([("a", 1), ("a", None), ("a", 2), ("b", None)], schema)
        res = (
            df.groupBy("g")
            .agg(F.sort_array(F.collect_list("v")).alias("c"), F.sort_array(F.array_agg("v")).alias("a"))
            .orderBy("g")
            .collect()
        )
        assert res == [Row(g="a", c=[1, 2], a=[1, 2]), Row(g="b", c=[], a=[])]

    def test_any_value_keeps_first_null(self, spark):
        schema = StructType([StructField("g", LongType()), StructField("v", LongType())])
        df = spark.createDataFrame([(1, None), (1, 2)], schema)
        if USE_ACTUAL_SPARK:
            df = df.coalesce(1)
        assert df.groupBy("g").agg(F.any_value("v").alias("v")).collect() == [Row(g=1, v=None)]

    def test_missing_group_by_sql_state(self, numbers):
        with pytest.raises(AnalysisException) as error:
            numbers.select(F.sum("i"), "i").collect()
        assert error.value.getSqlState() == "42803"


@pytest.fixture
def typed(spark):
    schema = "k string, i int, l bigint, d decimal(10,2)"
    return spark.createDataFrame([("x", 1, 2, Decimal("1.50")), ("x", 3, 4, Decimal("2.25"))], schema)


@pytest.fixture
def duckdb_only(spark):
    if USE_ACTUAL_SPARK:
        pytest.skip("HUGEINT and UBIGINT are DuckDB only types")
    return spark.sql(
        f"select 'x' k, 1::int i, {2**127 - 1}::hugeint h, {2**64 - 1}::ubigint u union all "
        "select 'x', 2, -1, 0::ubigint"
    )


class TestAggregateResultTypes:
    def test_sum_and_max_of_spark_types(self, typed):
        df = typed.select(F.sum("i"), F.sum("l"), F.max("i"), F.max("l"), F.sum("d"), F.max("d"))
        assert df.dtypes[:4] == [("sum(i)", "bigint"), ("sum(l)", "bigint"), ("max(i)", "int"), ("max(l)", "bigint")]
        assert df.dtypes[5] == ("max(d)", "decimal(10,2)")
        assert df.collect() == [Row(4, 6, 3, 4, Decimal("3.75"), Decimal("2.25"))]

    def test_aggregate_nested_in_an_expression(self, typed):
        df = typed.select(F.array(F.sum("i")).alias("a"), F.struct(F.sum("l").alias("s")).alias("t"))
        assert df.dtypes == [("a", "array<bigint>"), ("t", "struct<s:bigint>")]
        assert df.collect() == [Row(a=[4], t=Row(s=6))]

    def test_aggregates_over_a_join(self, spark, typed):
        other = spark.createDataFrame([("x", 10)], "k string, j int")
        joined = typed.join(other, "k")
        df = joined.groupBy("k").agg(F.sum(typed.i), F.array(F.sum(other.j)).alias("a"), F.max(typed.l))
        assert df.dtypes == [("k", "string"), ("sum(i)", "bigint"), ("a", "array<bigint>"), ("max(l)", "bigint")]
        assert df.collect() == [Row(k="x", **{"sum(i)": 4}, a=[20], **{"max(l)": 4})]

    def test_duckdb_only_inputs_keep_their_types(self, duckdb_only):
        df = duckdb_only.select(F.max("h"), F.sum("h"), F.sum("u"), F.max("u"), F.sum("i"))
        assert df.dtypes == [
            ("max(h)", "hugeint"),
            ("sum(h)", "hugeint"),
            ("sum(u)", "hugeint"),
            ("max(u)", "ubigint"),
            ("sum(i)", "bigint"),
        ]
        assert df.collect() == [Row(2**127 - 1, 2**127 - 2, 2**64 - 1, 2**64 - 1, 3)]

    def test_duckdb_only_inputs_of_grouped_data(self, duckdb_only):
        df = duckdb_only.groupBy("k").sum("i", "h", "u")
        assert df.dtypes == [("k", "string"), ("sum(i)", "bigint"), ("sum(h)", "hugeint"), ("sum(u)", "hugeint")]
        assert df.collect() == [Row("x", 3, 2**127 - 2, 2**64 - 1)]
        assert duckdb_only.groupBy("k").max("h").collect() == [Row("x", 2**127 - 1)]

    def test_duckdb_only_inputs_over_a_join(self, spark, duckdb_only):
        other = spark.createDataFrame([("x", 10)], "k string, j int")
        df = duckdb_only.join(other, "k").groupBy("k").agg(F.max(duckdb_only.h), F.sum(other.j))
        assert df.dtypes == [("k", "string"), ("max(h)", "hugeint"), ("sum(j)", "bigint")]
        assert df.collect() == [Row("x", 2**127 - 1, 20)]

    def test_expression_mixing_a_spark_sum_with_a_duckdb_only_input(self, duckdb_only):
        # Every HUGEINT of a result with a DuckDB only input stays, including that of the integer sum beside it.
        df = duckdb_only.select(F.struct(F.sum("i").alias("s"), F.max("h").alias("m")).alias("t"))
        assert df.dtypes == [("t", "struct<s:hugeint,m:hugeint>")]
        assert df.collect() == [Row(t=Row(s=3, m=2**127 - 1))]


class TestGroupedDataNumericColumns:
    @pytest.mark.parametrize("method", ["max", "min", "sum", "avg", "mean"])
    @pytest.mark.parametrize("column", ["s", "dt", "b", "`s`"])
    def test_non_numeric_column_raises(self, spark, method, column):
        df = spark.createDataFrame([(1, "x", datetime.date(2024, 1, 1), True)], "k int, s string, dt date, b boolean")
        with pytest.raises(AnalysisException) as error:
            getattr(df.groupBy("k"), method)(column).collect()
        assert error_condition(error.value) == "_LEGACY_ERROR_TEMP_1323"
        assert dict(error.value.getMessageParameters()) == {"colName": column}

    @pytest.mark.parametrize("schema", ["k int, `a``b` int", "k int, `c.d` int"])
    def test_numeric_columns_are_read_as_attribute_names(self, spark, schema):
        # Spark reads the names of the numeric columns as attribute names, so these raise there too.
        with pytest.raises(Exception):  # noqa: B017, PT011
            spark.createDataFrame([(1, 2)], schema).groupBy("k").max().collect()

    def test_numeric_columns_are_aggregated(self, spark):
        df = spark.createDataFrame([(1, "x", 2.0)], "k int, s string, d double")
        assert df.groupBy("k").max("d", "k").collect() == [Row(k=1, **{"max(d)": 2.0, "max(k)": 1})]
        assert df.groupBy("k").max().collect() == [Row(k=1, **{"max(k)": 1, "max(d)": 2.0})]


class TestRepeatedResultNames:
    @pytest.fixture
    def pairs(self, spark):
        return spark.createDataFrame([(1, 10, 3), (1, 20, 4)], ["k", "a", "b"])

    def test_aggregate_named_like_the_key(self, pairs):
        res = pairs.groupBy("k").agg(F.sum("a").alias("k"))
        assert res.columns == ["k", "k"]
        assert res.dtypes == [("k", "bigint"), ("k", "bigint")]
        assert [tuple(r) for r in res.collect()] == [(1, 30)]

    def test_aggregates_with_one_name(self, pairs):
        res = pairs.select(F.sum("a").alias("x"), F.sum("b").alias("x"))
        assert [tuple(r) for r in res.collect()] == [(30, 7)]

    def test_names_that_differ_in_case(self, pairs):
        res = pairs.groupBy("k").agg(F.sum("a").alias("K"), F.max("b").alias("A"))
        assert res.columns == ["k", "K", "A"]
        assert [tuple(r) for r in res.collect()] == [(1, 30, 4)]


class TestGlobalAggregateOfConstants:
    @pytest.fixture
    def two_rows(self, spark):
        return spark.createDataFrame([(1, 2), (3, 4)], ["a", "b"])

    def test_one_row(self, two_rows):
        res = two_rows.groupBy().agg(F.lit(1), F.lit("s").alias("s"))
        assert res.columns == ["1", "s"]
        assert [tuple(r) for r in res.collect()] == [(1, "s")]

    def test_one_row_of_empty_input(self, two_rows):
        empty = two_rows.filter(F.col("a") > 100)
        assert [tuple(r) for r in empty.groupBy().agg(F.lit(1)).collect()] == [(1,)]
        assert [tuple(r) for r in empty.groupBy().agg(F.lit(1), F.sum("a")).collect()] == [(1, None)]

    def test_constants_with_keys(self, two_rows):
        assert sorted(tuple(r) for r in two_rows.groupBy("a").agg(F.lit(1)).collect()) == [(1, 1), (3, 1)]

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

import datetime
from decimal import Decimal

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException
from spark_namespace.sql import functions as F
from spark_namespace.sql import types as T

if USE_ACTUAL_SPARK:
    import pyspark

    SPARK_3 = pyspark.__version__.startswith("3.")
else:
    SPARK_3 = False


@pytest.fixture
def df(spark):
    schema = T.StructType(
        [
            T.StructField("a", T.LongType()),
            T.StructField("b", T.DoubleType()),
            T.StructField("s", T.StringType()),
            T.StructField("flag", T.BooleanType()),
            T.StructField("d", T.DateType()),
            T.StructField("arr", T.ArrayType(T.LongType())),
            T.StructField("st", T.StructType([T.StructField("x", T.LongType()), T.StructField("y", T.StringType())])),
            T.StructField("CamelCase", T.LongType()),
        ]
    )
    return spark.createDataFrame([(1, 1.5, "x", True, datetime.date(2020, 1, 2), [1, 2], (1, "q"), 3)], schema)


@pytest.fixture
def df2(spark):
    schema = T.StructType([T.StructField("a", T.LongType()), T.StructField("c", T.StringType())])
    return spark.createDataFrame([(1, "c1")], schema)


def names(df, *cols):
    return df.select(*cols).columns


class TestColumnReferenceNames:
    def test_plain_and_bound_references(self, df):
        assert names(df, "a", F.col("b"), df.s, df["flag"]) == ["a", "b", "s", "flag"]

    def test_requested_case_is_kept(self, df):
        assert names(df, F.col("A"), "B", F.col("camelcase"), "CAMELCASE") == ["A", "B", "camelcase", "CAMELCASE"]
        assert df.select(F.col("A")).collect()[0].asDict() == {"A": 1}

    def test_bound_references_keep_requested_case_and_cast_name(self, df):
        assert names(df, df["S"], df.a.cast("string"), df.a.cast("int").cast("string")) == ["S", "a", "a"]

    def test_missing_column_is_reported_as_missing(self, df):
        with pytest.raises(Exception, match=r"not found|cannot be resolved"):
            df.select("missing").collect()
        with pytest.raises(Exception, match=r"not found|cannot be resolved"):
            df.groupBy("missing").count().collect()

    def test_star(self, df):
        assert names(df, "*") == df.columns
        assert names(df, F.col("*")) == df.columns

    def test_struct_fields(self, df):
        assert names(df, F.col("st.x"), "st.y", df.st.x, df.st["y"]) == ["x", "y", "st.x", "st.y"]
        assert names(df, F.upper(df.st.y), F.upper(F.col("st.y")), F.upper("st.y")) == ["upper(st.y)"] * 3

    def test_names_with_spaces(self, spark):
        df = spark.createDataFrame([(1,)], ["my col"])
        assert names(df, F.col("my col"), F.col("my col") + 1) == ["my col", "(my col + 1)"]

    def test_backtick_quoted_names(self, spark):
        df = spark.createDataFrame([("p", 2, (3,))], "a string, `a.b` int, st struct<x:int>")
        res = df.select(F.col("`a`"), "`a`", F.col("`a.b`"), "`a.b`", F.col("`st`.`x`"), F.upper("`a`"))
        assert res.columns == ["a", "a", "a.b", "a.b", "x", "upper(a)"]
        assert res.collect() == [("p", "p", 2, 2, 3, "P")]

    @pytest.mark.parametrize("name", ["a`b", "`a`b", "a``b", "`a", "a..b", "a.", "`a` "])
    def test_invalid_attribute_names_raise(self, spark, name):
        # Each of these would otherwise read one of the columns a, ab or a`b.
        df = spark.createDataFrame([(1, 2, 3)], ["a", "ab", "a`b"])
        for select in [lambda: df.select(F.col(name)), lambda: df.select(name)]:
            with pytest.raises(AnalysisException, match=r"Syntax error in (the )?attribute name"):
                select().collect()

    def test_struct_path_through_a_repeated_name_is_ambiguous(self, spark):
        df = spark.createDataFrame([((1,), (2,))], "s struct<x:int>, t struct<x:int>").toDF("s", "s")
        for name in ["s.x", "`s`.x", F.col("s.x")]:
            with pytest.raises(AnalysisException, match="AMBIGUOUS_REFERENCE"):
                df.select(name).collect()

    def test_star_of_a_struct_or_an_alias_through_col(self, spark):
        df = spark.createDataFrame([((1, "a"), 2)], "s struct<x:int,y:string>, n int")
        assert df.select(F.col("s.*")).collect() == [(1, "a")]
        assert df.select(F.col("`s`.*"), "n").columns == ["x", "y", "n"]
        assert df.alias("q").select(F.col("q.*")).columns == ["s", "n"]
        # Spark allows such a star only in select.
        for make in [lambda: df.withColumn("z", F.col("s.*")), lambda: df.alias("q").select(F.col("q.*").alias("w"))]:
            with pytest.raises(Exception):  # noqa: B017, PT011
                make().collect()
        repeated = spark.createDataFrame([((1,), (2,))], "s struct<x:int>, t struct<x:int>").toDF("s", "s")
        for star in ["s.*", F.col("s.*")]:
            with pytest.raises(AnalysisException, match="AMBIGUOUS_REFERENCE"):
                repeated.select(star).collect()

    def test_backtick_quoted_names_after_a_join(self, spark):
        left = spark.createDataFrame([(1, (5,))], "`k.k` int, st struct<x:int>")
        right = spark.createDataFrame([(1, "y"), (2, "z")], "`k.k` int, w string")
        outer = left.join(right, "k.k", "outer").select("`k.k`")
        assert outer.columns == ["k.k"]
        assert sorted(outer.collect()) == [(1,), (2,)]
        inner = left.join(right, "k.k").select("`k.k`", "`st`.`x`", "w")
        assert inner.columns == ["k.k", "x", "w"]
        assert inner.collect() == [(1, 5, "y")]
        # A quoted name with a dot is one column name, not a struct field.
        with pytest.raises(Exception, match=r"st\.x"):
            left.join(right, "k.k").select("`st.x`").collect()
        joined = left.join(right, "k.k")
        assert joined.select("*").columns == ["k.k", "st", "w"]
        assert joined.select(joined["st.x"]).collect() == [(5,)]

    def test_nested_struct_paths(self, spark):
        # The field b next to a makes a path that drops a middle part read another value.
        schema = "s struct<a: struct<b: struct<c: int>>, b: struct<c: int>>"
        df = spark.createDataFrame([({"a": {"b": {"c": 1}}, "b": {"c": 2}},)], schema)
        res = df.select(
            F.col("s.a.b.c"),
            "s.a.b.c",
            df.s.a.b.c,
            df["s"]["a"]["b"]["c"],
            F.col("s")["a"]["b"]["c"],
            F.col("s.a.b"),
            "s.a.b",
            df.s.a.b,
            F.col("s.b.c"),
        )
        assert res.columns == ["c", "c", "s.a.b.c", "s.a.b.c", "s.a.b.c", "b", "b", "s.a.b", "c"]
        assert res.collect() == [(1, 1, 1, 1, 1, (1,), (1,), (1,), 2)]

    def test_selecting_a_generated_name(self, df):
        assert names(df.select(F.abs("a")), F.col("abs(a)") + 1) == ["(abs(a) + 1)"]
        assert names(df.select(df.a + 1), "(a + 1)") == ["(a + 1)"]
        assert names(df.select(F.col("A")), "A") == ["A"]


class TestFunctionCallNames:
    def test_one_argument(self, df):
        assert names(df, F.abs(df.a), F.abs("a"), F.abs(F.col("A")), F.upper(df.s), F.lower("s")) == [
            "abs(a)",
            "abs(a)",
            "abs(A)",
            "upper(s)",
            "lower(s)",
        ]

    def test_spark_function_name_differs_from_python_name(self, df):
        assert names(df, F.sqrt("a"), F.exp("a"), F.ceil("b"), F.floor("b"), F.signum("b"), F.ucase("s")) == [
            "SQRT(a)",
            "EXP(a)",
            "CEIL(b)",
            "FLOOR(b)",
            "SIGNUM(b)",
            "ucase(s)",
        ]
        assert names(df, F.pow("a", "b"), F.atan2("b", "b"), F.e(), F.pi()) == [
            "POWER(a, b)",
            "ATAN2(b, b)",
            "E()",
            "PI()",
        ]
        assert names(df, F.curdate(), F.isnull("a"), F.isnotnull("a"), F.positive("a"), F.negative("a")) == [
            "current_date()",
            "(a IS NULL)",
            "(a IS NOT NULL)",
            "(+ a)",
            "negative(a)",
        ]

    def test_function_implemented_by_another_expression(self, df):
        assert names(df, F.array_max("arr"), F.to_date("s"), F.equal_null("a", "a"), F.nvl("a", "a")) == [
            "array_max(arr)",
            "to_date(s)",
            "equal_null(a, a)",
            "nvl(a, a)",
        ]

    def test_several_arguments(self, df):
        assert names(
            df,
            F.round("b", 1),
            F.greatest("a", "b"),
            F.least(df.a, df.b),
            F.coalesce("a", "b"),
            F.concat_ws("-", "s", "s"),
            F.substring("s", 1, 2),
            F.lpad("s", 3, "-"),
            F.date_trunc("year", "d"),
            F.array_contains("arr", 1),
            F.struct(df.a, F.col("s").alias("z")),
        ) == [
            "round(b, 1)",
            "greatest(a, b)",
            "least(a, b)",
            "coalesce(a, b)",
            "concat_ws(-, s, s)",
            "substring(s, 1, 2)",
            "lpad(s, 3, -)",
            "date_trunc(year, d)",
            "array_contains(arr, 1)",
            "struct(a, s AS z)",
        ]

    def test_arguments_spark_adds(self, df):
        assert names(
            df,
            F.round("b"),
            F.bround("b"),
            F.regexp_replace("s", "x", "y"),
            F.sort_array("arr"),
            F.split("s", ","),
            F.substr("s", F.lit(1)),
        ) == [
            "round(b, 0)",
            "bround(b, 0)",
            "regexp_replace(s, x, y, 1)",
            "sort_array(arr, true)",
            "split(s, ,, -1)",
            "substr(s, 1, 2147483647)",
        ]

    def test_special_forms(self, df):
        assert names(df, F.like("s", F.lit("x%"), F.lit("!")), F.ilike("s", F.lit("x%"), F.lit("!"))) == [
            "s LIKE x%" if SPARK_3 else "s LIKE x% ESCAPE '!'",
            "ilike(s, x%)",
        ]
        assert names(df, df.a.asc_nulls_last(), df.a.desc(), df.a.desc_nulls_first()) == [
            "a ASC NULLS LAST",
            "a DESC NULLS LAST",
            "a DESC NULLS FIRST",
        ]
        assert names(
            df,
            F.like("s", F.lit("x%")),
            F.extract(F.lit("YEAR"), "d"),
            F.datepart(F.lit("YEAR"), "d"),
            F.asc("a"),
            F.desc_nulls_first("a"),
            F.call_function("abs", "a"),
        ) == [
            "s LIKE x%",
            "extract(YEAR FROM d)",
            "datepart(YEAR FROM d)",
            "a ASC NULLS FIRST",
            "a DESC NULLS FIRST",
            "abs(a)",
        ]

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="Spark's Column has no nulls_first or nulls_last")
    def test_null_ordering_without_direction(self, df):
        assert names(df, F.col("a").nulls_first(), F.col("a").asc().nulls_last(), F.col("a").desc().nulls_first()) == [
            "a NULLS FIRST",
            "a ASC NULLS LAST",
            "a DESC NULLS FIRST",
        ]

    def test_nested(self, df):
        assert names(
            df,
            F.abs(F.sqrt("a")),
            F.upper(F.lower("s")),
            F.round(F.col("b") * 2, 1),
            F.concat(F.upper("s"), F.lit("!")),
            F.abs(df.a.alias("x")),
        ) == ["abs(SQRT(a))", "upper(lower(s))", "round((b * 2), 1)", "concat(upper(s), !)", "abs(a AS x)"]


class TestOperatorNames:
    def test_arithmetic(self, df):
        assert names(df, df.a + 1, df.a - df.b, df.a * 2, df.a / 2, df.a % 2, df.a**2) == [
            "(a + 1)",
            "(a - b)",
            "(a * 2)",
            "(a / 2)",
            "(a % 2)",
            "POWER(a, 2)",
        ]

    def test_reflected_arithmetic(self, df):
        assert names(df, 1 + df.a, 2 - df.a, 2 * df.a, 2 / df.a, 2**df.a) == [
            "(a + 1)",
            "(2 - a)",
            "(a * 2)",
            "(2 / a)",
            "POWER(2, a)",
        ]

    def test_precedence_is_explicit(self, df):
        assert names(df, df.a + df.b * 2, (df.a + df.b) * 2) == ["(a + (b * 2))", "((a + b) * 2)"]

    def test_comparison(self, df):
        assert names(df, df.a == 1, df.a != 1, df.a < 1, df.a <= 1, df.a > 1, df.a >= 1, df.a > 1) == [
            "(a = 1)",
            "(NOT (a = 1))",
            "(a < 1)",
            "(a <= 1)",
            "(a > 1)",
            "(a >= 1)",
            "(a > 1)",
        ]

    def test_boolean(self, df):
        assert names(df, (df.a > 1) & (df.b < 2), (df.a > 1) | df.flag, ~df.flag, ~(df.a > 1)) == [
            "((a > 1) AND (b < 2))",
            "((a > 1) OR flag)",
            "(NOT flag)",
            "(NOT (a > 1))",
        ]

    def test_unary_minus(self, df):
        expected = ["(- a)", "(- b)"] if SPARK_3 else ["negative(a)", "negative(b)"]
        assert names(df, -df.a, -F.col("b")) == expected


class TestLiteralNames:
    def test_scalars(self, df):
        assert names(df, F.lit(1), F.lit(-1), F.lit(2**40), F.lit("x"), F.lit("it's"), F.lit(True), F.lit(None)) == [
            "1",
            "-1",
            "1099511627776",
            "x",
            "it's",
            "true",
            "NULL",
        ]

    def test_doubles_are_named_like_java(self, df):
        assert names(df, F.lit(1.5), F.lit(100.0), F.lit(-2.5), F.lit(0.001), F.lit(1e-5), F.lit(12345678.0)) == [
            "1.5",
            "100.0",
            "-2.5",
            "0.001",
            "1.0E-5",
            "1.2345678E7",
        ]
        assert names(df, F.lit(1e20), F.lit(float("inf")), F.lit(float("-inf"))) == [
            "1.0E20",
            "Infinity",
            "-Infinity",
        ]

    def test_dates_decimals_and_bytes(self, df):
        assert names(
            df,
            F.lit(datetime.date(2020, 1, 2)),
            F.lit(datetime.datetime(2020, 1, 2, 3, 4, 5)),
            F.lit(datetime.datetime(2020, 1, 2, 3, 4, 5, 120000)),
            F.lit(Decimal("1.50")),
            F.lit(Decimal("1E+2")),
            F.lit(bytearray(b"ab")),
        ) == [
            "DATE '2020-01-02'",
            "TIMESTAMP '2020-01-02 03:04:05'",
            "TIMESTAMP '2020-01-02 03:04:05.12'",
            "1.50",
            "100",
            "X'6162'",
        ]

    def test_timestamps_with_a_zone_are_named_by_their_local_time(self, df):
        value = datetime.datetime(2024, 1, 2, 3, 4, 5, 500000, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
        local = value.astimezone().replace(tzinfo=None)
        text = f"TIMESTAMP '{local.isoformat(' ').rstrip('0')}'"
        res = df.select(F.lit(value), F.lit(value) > datetime.datetime(2000, 1, 1))
        assert res.columns == [text, f"({text} > TIMESTAMP '2000-01-01 00:00:00')"]
        assert res.collect()[0][0] == local

    def test_operands(self, df):
        assert names(
            df,
            df.a + F.lit(1.5),
            df.s == "x",
            df.s == "it's",
            df.a + None,
            df.b > 1e20,
            df.d > datetime.date(2020, 1, 1),
            F.upper(F.lit("x")),
        ) == [
            "(a + 1.5)",
            "(s = x)",
            "(s = it's)",
            "(a + NULL)",
            "(b > 1.0E20)",
            "(d > DATE '2020-01-01')",
            "upper(x)",
        ]


class TestCastNames:
    def test_cast_of_a_column_keeps_its_name(self, df):
        assert names(df, F.col("a").cast("string"), F.col("a").cast(T.IntegerType()), F.col("A").cast("double")) == [
            "a",
            "a",
            "A",
        ]
        assert names(df, F.col("a").cast("int").cast("string")) == ["a"]

    def test_cast_of_an_alias_keeps_the_alias(self, df):
        assert names(df, F.abs("a").alias("x").cast("string")) == ["x"]

    def test_cast_of_an_expression(self, df):
        assert names(
            df,
            F.lit(1).cast("string"),
            (df.a + 1).cast("string"),
            F.abs("a").cast("string"),
            F.lit(1).cast("long"),
            F.lit(1).cast("integer"),
            F.lit(1).cast(T.ShortType()),
            F.lit(1).cast(T.DecimalType(5, 2)),
            F.lit("x").cast("binary"),
            F.lit("2020-01-01").cast("date"),
            F.col("st.x").cast("string"),
            df.st.x.cast("string"),
        ) == [
            "CAST(1 AS STRING)",
            "CAST((a + 1) AS STRING)",
            "CAST(abs(a) AS STRING)",
            "CAST(1 AS BIGINT)",
            "CAST(1 AS INT)",
            "CAST(1 AS SMALLINT)",
            "CAST(1 AS DECIMAL(5,2))",
            "CAST(x AS BINARY)",
            "CAST(2020-01-01 AS DATE)",
            "CAST(st.x AS STRING)",
            "CAST(st.x AS STRING)",
        ]

    def test_cast_to_a_complex_type_is_named_after_its_input(self, df):
        assert names(
            df,
            F.array(F.lit(1), F.lit(2)).cast("array<int>"),
            F.array(F.lit(1), F.lit(2)).cast("array<bigint>"),
            F.array("a", "a").cast(T.ArrayType(T.StringType())),
            F.struct(F.lit(1).alias("p")).cast("struct<p:bigint>"),
            F.lit(None).cast("array<int>"),
            F.array(F.lit(1)).cast("array<bigint>").cast("array<int>"),
            F.array(F.lit(1)).cast("string"),
        ) == [
            "array(1, 2)",
            "array(1, 2)",
            "array(a, a)",
            "struct(1 AS p)",
            "NULL",
            "array(1)",
            "CAST(array(1) AS STRING)",
        ]
        res = df.select(F.struct(F.col("arr").cast("array<int>"), F.array("a").cast("array<int>")))
        assert res.columns == ["struct(arr, array(a))"]
        assert res.schema.fields[0].dataType.fieldNames() == ["col1", "col2"]

    def test_cast_of_a_struct_path_to_a_complex_type(self, spark):
        df = spark.createDataFrame([(([1, 2], (3,)),)], "st struct<arr: array<int>, inner: struct<x:int>>")
        res = df.select(F.col("st.arr").cast("array<bigint>"), F.col("st.inner").cast("struct<x:bigint>"))
        assert res.columns == ["st.arr", "st.inner"]
        assert res.collect() == [([1, 2], (3,))]


class TestAliasNames:
    def test_alias(self, df):
        assert names(
            df,
            df.a.alias("x"),
            F.abs("a").alias("y"),
            F.abs("a").alias("p").alias("q"),
            (df.a + 1).alias("A Plus One"),
            F.col("a").alias("A"),
        ) == ["x", "y", "q", "A Plus One", "A"]

    def test_alias_inside_a_name_is_quoted_unless_a_plain_word(self, df):
        aliases = ["a_1", "a b", "1", "a`b"]
        res = df.select(F.struct(*(F.lit(i).alias(alias) for i, alias in enumerate(aliases, 1))))
        assert res.columns == ["struct(1 AS a_1, 2 AS `a b`, 3 AS `1`, 4 AS `a``b`)"]
        assert res.schema.fields[0].dataType.fieldNames() == aliases

    def test_empty_string_literal(self, df):
        assert df.select(F.lit("")).collect() == [("",)]

    def test_empty_top_level_alias(self, df):
        makes = [
            lambda: df.select(F.lit(1).alias("")),
            lambda: df.withColumn("", F.lit(1)),
            lambda: df.withColumns({"": F.lit(1)}),
            lambda: df.select("a").withColumnRenamed("a", ""),
            lambda: df.select("a").withColumnsRenamed({"a": ""}),
            lambda: df.select("a").toDF(""),
        ]
        if USE_ACTUAL_SPARK:
            assert all("" in make().columns for make in makes)
            return
        from duckdb.experimental.spark.exception import ContributionsAcceptedError

        # DuckDB has no column with an empty name.
        for make in makes:
            with pytest.raises(ContributionsAcceptedError, match="empty name"):
                make()

    def test_empty_alias_names_a_struct_field(self, df):
        res = df.select(
            F.struct(F.lit(1).alias("")).alias("one"),
            F.struct(F.lit(1).alias(""), F.lit("x").alias("")).alias("two"),
            F.arrays_zip(F.array(F.lit(1)).alias("")).alias("zip"),
        )
        assert res.schema.simpleString() == "struct<one:struct<:int>,two:struct<:int,:string>,zip:array<struct<:int>>>"
        assert res.collect() == [((1,), (1, "x"), [(1,)])]
        assert names(df, F.struct(F.lit(1).alias(""))) == ["struct(1 AS ``)"]

    def test_struct_with_an_empty_and_another_field_name(self, df):
        if USE_ACTUAL_SPARK:
            res = df.select(F.struct(F.lit(1).alias(""), F.lit(2).alias("b")).alias("s"))
            assert res.schema.simpleString() == "struct<s:struct<:int,b:int>>"
        else:
            from duckdb.experimental.spark.exception import ContributionsAcceptedError

            with pytest.raises(ContributionsAcceptedError, match="both empty and other field names"):
                F.struct(F.lit(1).alias(""), F.lit(2).alias("b"))


class TestConditionalAndPredicateNames:
    def test_when_otherwise(self, df):
        assert names(
            df,
            F.when(df.a > 1, 1),
            F.when(df.a > 1, 1).otherwise(0),
            F.when(df.a > 1, "x").when(df.a > 0, "y"),
            F.when(df.a > 1, "x").when(df.a > 0, "y").otherwise("z"),
            F.when(df.flag, df.a).otherwise(df.b),
            F.when(df.a > 1, F.lit(None)),
        ) == [
            "CASE WHEN (a > 1) THEN 1 END",
            "CASE WHEN (a > 1) THEN 1 ELSE 0 END",
            "CASE WHEN (a > 1) THEN x WHEN (a > 0) THEN y END",
            "CASE WHEN (a > 1) THEN x WHEN (a > 0) THEN y ELSE z END",
            "CASE WHEN flag THEN a ELSE b END",
            "CASE WHEN (a > 1) THEN NULL END",
        ]

    def test_null_tests(self, df):
        assert names(df, df.a.isNull(), df.a.isNotNull(), F.col("A").isNull()) == [
            "(a IS NULL)",
            "(a IS NOT NULL)",
            "(A IS NULL)",
        ]

    def test_isin(self, df):
        assert names(df, df.a.isin(1, 2), df.a.isin([1, 2]), df.s.isin("x", "y"), df.a.isin(df.a, 2)) == [
            "(a IN (1, 2))",
            "(a IN (1, 2))",
            "(s IN (x, y))",
            "(a IN (a, 2))",
        ]

    def test_string_predicates(self, df):
        assert names(
            df,
            df.s.like("x%"),
            df.s.ilike("X%"),
            df.s.rlike("^x"),
            df.s.contains("x"),
            df.s.startswith("x"),
            df.s.endswith("x"),
        ) == ["s LIKE x%", "ilike(s, X%)", "RLIKE(s, ^x)", "contains(s, x)", "startswith(s, x)", "endswith(s, x)"]


class TestAggregateNames:
    def test_agg(self, df):
        assert df.groupBy("s").agg(
            F.sum("a"),
            F.sum(df.a),
            F.count("a"),
            F.count(F.lit(1)),
            F.avg("a"),
            F.mean("a"),
            F.first("a"),
            F.last("a"),
            F.collect_list("a"),
            F.array_agg("a"),
            F.stddev("a"),
            F.sum(df.a + 1),
            F.sum("a") + 1,
            F.sum("a").alias("total"),
        ).columns == [
            "s",
            "sum(a)",
            "sum(a)",
            "count(a)",
            "count(1)",
            "avg(a)",
            "avg(a)",
            "first(a)",
            "last(a)",
            "collect_list(a)",
            "collect_list(a)",
            "stddev(a)",
            "sum((a + 1))",
            "(sum(a) + 1)",
            "total",
        ]

    def test_aggregate_spark_names_differently_per_version(self, df):
        assert df.groupBy("s").agg(F.variance("a")).columns == ["s", "var_samp(a)" if SPARK_3 else "variance(a)"]

    def test_grouping_keys(self, df):
        assert df.groupBy("S").agg(F.sum("a")).columns == ["S", "sum(a)"]
        assert df.groupBy(df.s).agg(F.sum("a")).columns == ["s", "sum(a)"]
        assert df.groupBy(F.upper("s")).agg(F.sum("a")).columns == ["upper(s)", "sum(a)"]

    def test_grouped_data_shortcuts(self, df):
        assert df.groupBy("s").count().columns == ["s", "count"]
        assert df.groupBy("s").sum("a").columns == ["s", "sum(a)"]
        assert df.groupBy("s").avg("a").columns == ["s", "avg(a)"]
        assert df.groupBy("s").mean("a").columns == ["s", "avg(a)"]
        assert df.groupBy("s").min("a", "b").columns == ["s", "min(a)", "min(b)"]
        assert df.groupBy("S").count().columns == ["S", "count"]
        assert df.groupBy(F.col("S")).max("a").columns == ["S", "max(a)"]
        assert df.groupBy().sum("a").columns == ["sum(a)"]


class TestNamesAfterOtherOperations:
    def test_after_joins(self, df, df2):
        assert df.join(df2, "a").select(F.abs(df2.a), df2.c, F.upper(df2.c)).columns == ["abs(a)", "c", "upper(c)"]
        assert df.join(df2, df.a == df2.a).select(df.a, df2.c).columns == ["a", "c"]
        assert df.join(df2, "a").select("c", F.upper("c")).columns == ["c", "upper(c)"]
        assert df.crossJoin(df2).select(df2.c).columns == ["c"]

    def test_qualifiers_of_aliased_dataframes_are_left_out(self, df, df2):
        joined = df.alias("l").join(df2.alias("r"), "a")
        assert joined.select(F.col("l.a"), F.col("r.c"), F.upper(F.col("r.c")), F.upper("r.c")).columns == [
            "a",
            "c",
            "upper(c)",
            "upper(c)",
        ]
        assert joined.select(F.col("l.st.x")).columns == ["x"]
        assert joined.select(F.col("l.a").cast("string"), F.col("l.st.x").cast("string")).columns == [
            "a",
            "CAST(st.x AS STRING)",
        ]

    def test_qualifier_that_is_also_a_column_name_is_left_out(self, spark):
        struct_only = spark.createDataFrame([((-1,),)], "s struct<x:int>").alias("s")
        assert struct_only.select(F.abs("s.s.x"), F.abs("s.x"), F.col("s.s.x").cast("string")).columns == [
            "abs(s.x)",
            "abs(s.x)",
            "CAST(s.x AS STRING)",
        ]
        both = spark.createDataFrame([(-1, (2,))], "x int, s struct<x:int>").alias("s")
        res = both.select(F.abs("s.x"), F.col("s.x") + 0, F.col("s.x").cast("string"))
        assert res.columns == ["abs(x)", "(x + 0)", "x"]
        assert res.collect() == [(1, -1, "-1")]
        # Spark 3.5.9 names a grouping key as it was written, with its qualifier.
        assert both.groupBy(F.abs("s.x")).count().columns == ["abs(s.x)" if SPARK_3 else "abs(x)", "count"]

    def test_after_filter_sort_and_limit(self, df):
        assert df.filter(df.a > 0).select(F.abs(df.a)).columns == ["abs(a)"]
        assert df.orderBy(df.a).select(F.abs(df.a)).columns == ["abs(a)"]
        assert df.limit(1).select(df.a + 1).columns == ["(a + 1)"]

    def test_with_column_uses_the_given_name(self, df):
        assert df.withColumn("n", F.abs("a")).columns[-1] == "n"
        assert df.select(F.abs("a")).withColumn("abs(a)", F.lit(1)).columns == ["abs(a)"]

    def test_row_fields_and_sql_expressions(self, df):
        row = df.select(F.abs(df.a), df.b + 1, F.expr("a + 2"), F.expr("upper(s)")).collect()[0]
        assert list(row.asDict()) == ["abs(a)", "(b + 1)", "(a + 2)", "upper(s)"]


def test_no_internal_relation_name_in_any_column_name(df, df2):
    cols = [
        df.a,
        df["b"],
        -df.a,
        ~df.flag,
        df.a + df.b,
        df.a.isin(df.a),
        df.a.cast("string"),
        df.a.alias("x"),
        df.st.x,
        df.s.like("x%"),
        df.s.contains(df.s),
        F.when(df.flag, df.a).otherwise(df.b),
        F.abs(df.a),
        F.concat_ws("-", df.s, df.s),
        F.coalesce(df.a, df.b),
        F.struct(df.a, df.s),
        F.array(df.a, df.a),
        F.array_max(df.arr),
        F.asc(df.a),
        F.date_diff(df.d, df.d),
        F.to_date(df.s),
        F.nvl2(df.a, df.a, df.a),
        F.equal_null(df.a, df.a),
        F.array_insert(df.arr, 1, df.a),
    ]
    joined = df.join(df2, df.a == df2.a).select(F.abs(df.a), F.upper(df2.c), df2.c)
    for name in df.select(*cols).columns + joined.columns:
        assert "unnamed_relation" not in name


class TestColumnRepr:
    def test_repr_shows_the_name(self, df):
        cols = [
            F.col("a"),
            df.a,
            F.col("A"),
            F.col("st.x"),
            F.abs("a"),
            F.round("b"),
            F.mean("a"),
            F.upper(F.lower("s")),
            F.lit(1.5),
            F.lit(1e20),
            F.lit(True),
            F.lit(None),
            df.a.alias("x"),
            df.a.cast("string"),
            df.a.asc_nulls_last(),
            F.struct("a", "s"),
            F.negative("a"),
        ]
        assert [repr(c) for c in cols] == [
            "Column<'a'>",
            "Column<'a'>",
            "Column<'A'>",
            "Column<'st.x'>",
            "Column<'abs(a)'>",
            "Column<'round(b, 0)'>",
            "Column<'avg(a)'>",
            "Column<'upper(lower(s))'>",
            "Column<'1.5'>",
            "Column<'1.0E20'>",
            "Column<'true'>",
            "Column<'NULL'>",
            "Column<'a AS x'>",
            "Column<'CAST(a AS STRING)'>",
            "Column<'a ASC NULLS LAST'>",
            "Column<'struct(a, s)'>",
            "Column<'negative(a)'>",
        ]
        assert str(F.abs("a")) == "Column<'abs(a)'>"


class TestSparkColumnMethods:
    @pytest.mark.parametrize("method", ["between", "getItem", "getField", "substr", "eqNullSafe", "bitwiseAND"])
    def test_missing_methods_raise(self, df, method):
        if USE_ACTUAL_SPARK:
            assert callable(getattr(df.a, method))
        else:
            from duckdb.experimental.spark.exception import ContributionsAcceptedError

            with pytest.raises(ContributionsAcceptedError, match=f"Column.{method} is not implemented"):
                getattr(df.a, method)
            with pytest.raises(AttributeError, match=f"Column.{method} is not implemented"):
                getattr(df.a, method)
            assert not hasattr(df.a, method)
            assert getattr(df.a, method, None) is None

    def test_other_names_are_struct_fields(self, df):
        assert df.select(df.st.x, F.col("st").y).collect()[0] == (1, "q")

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="checks the module's list against the installed PySpark")
    def test_method_list_covers_pyspark(self):
        pyspark_sql = pytest.importorskip("pyspark.sql")
        from duckdb.experimental.spark.sql.column import _SPARK_COLUMN_METHODS

        spark_methods = {n for n in dir(pyspark_sql.Column) if not n.startswith("_")}
        assert spark_methods <= _SPARK_COLUMN_METHODS


@pytest.mark.skipif(USE_ACTUAL_SPARK, reason="checks every function the module implements")
def test_every_function_gives_a_spark_name(spark):
    import contextlib
    import inspect

    from duckdb.experimental.spark.sql import functions
    from duckdb.experimental.spark.sql.column import Column

    generic = {int: 1, float: 0.5, bool: True, str: "a"}
    calls = {
        "call_function": ("abs", "a"),
        "date_part": (F.lit("YEAR"), "d"),
        "datepart": (F.lit("YEAR"), "d"),
        "date_trunc": ("month", "d"),
        "encode": ("s", "UTF-8"),
        "extract": (F.lit("YEAR"), "d"),
        "regexp_replace": ("s", "a", "b"),
        "sha2": ("s", 256),
    }
    unnamed = []
    for name, function in list(vars(functions).items()):
        if name.startswith("_") or not inspect.isfunction(function) or function.__module__ != functions.__name__:
            continue
        # transform raises; expr takes SQL text, whose names are DuckDB's; broadcast returns a DataFrame.
        if name in {"transform", "expr", "broadcast"}:
            continue
        args = list(calls.get(name, ()))
        if name not in calls:
            for parameter in inspect.signature(function).parameters.values():
                if parameter.kind is parameter.VAR_POSITIONAL:
                    args += [F.col("a"), F.col("b")]
                elif parameter.default is not parameter.empty:
                    continue
                elif "Callable" in str(parameter.annotation):
                    args.append(lambda x: x)
                elif "Column" in str(parameter.annotation):
                    args.append(F.col("a"))
                else:
                    args.append(generic.get(parameter.annotation, 1))
        with pytest.warns(FutureWarning) if name == "approxCountDistinct" else contextlib.nullcontext():
            result = function(*args)
        assert isinstance(result, Column), name
        if result._name is None:
            unnamed.append(name)
    assert unnamed == []

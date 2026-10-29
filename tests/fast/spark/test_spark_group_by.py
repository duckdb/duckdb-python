import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace.sql.functions import (
    any_value,
    approx_count_distinct,
    avg,
    col,
    covar_pop,
    covar_samp,
    first,
    last,
    max,
    median,
    mode,
    product,
    skewness,
    std,
    stddev,
    stddev_pop,
    stddev_samp,
    sum,
    var_pop,
    var_samp,
    variance,
)
from spark_namespace.sql.types import (
    Row,
)


class TestDataFrameGroupBy:
    def test_group_by(self, spark):
        simpleData = [
            ("James", "Sales", "NY", 90000, 34, 10000),
            ("Michael", "Sales", "NY", 86000, 56, 20000),
            ("Robert", "Sales", "CA", 81000, 30, 23000),
            ("Maria", "Finance", "CA", 90000, 24, 23000),
            ("Raman", "Finance", "CA", 99000, 40, 24000),
            ("Scott", "Finance", "NY", 83000, 36, 19000),
            ("Jen", "Finance", "NY", 79000, 53, 15000),
            ("Jeff", "Marketing", "CA", 80000, 25, 18000),
            ("Kumar", "Marketing", "NY", 91000, 50, 21000),
        ]

        schema = ["employee_name", "department", "state", "salary", "age", "bonus"]
        df = spark.createDataFrame(data=simpleData, schema=schema)

        df2 = df.groupBy("department").sum("salary").sort("department")
        res = df2.collect()
        expected = (
            "[Row(department='Finance', sum(salary)=351000), Row(department='Marketing', sum(salary)=171000), "
            "Row(department='Sales', sum(salary)=257000)]"
        )
        assert str(res) == expected

        df2 = df.groupBy("department").count().sort("department")
        res = df2.collect()
        assert (
            str(res) == "[Row(department='Finance', count=4), Row(department='Marketing', count=2), "
            "Row(department='Sales', count=3)]"
        )

        df2 = df.groupBy("department").min("salary").sort("department")
        res = df2.collect()
        assert (
            str(res)
            == "[Row(department='Finance', min(salary)=79000), Row(department='Marketing', min(salary)=80000), "
            "Row(department='Sales', min(salary)=81000)]"
        )

        df2 = df.groupBy("department").max("salary").sort("department")
        res = df2.collect()
        assert (
            str(res)
            == "[Row(department='Finance', max(salary)=99000), Row(department='Marketing', max(salary)=91000), "
            "Row(department='Sales', max(salary)=90000)]"
        )

        df2 = df.groupBy("department").avg("salary").sort("department")
        res = df2.collect()
        assert (
            str(res)
            == "[Row(department='Finance', avg(salary)=87750.0), Row(department='Marketing', avg(salary)=85500.0), "
            "Row(department='Sales', avg(salary)=85666.66666666667)]"
        )

        df2 = df.groupBy("department").mean("salary").sort("department")
        res = df2.collect()
        assert (
            str(res)
            == "[Row(department='Finance', avg(salary)=87750.0), Row(department='Marketing', avg(salary)=85500.0), "
            "Row(department='Sales', avg(salary)=85666.66666666667)]"
        )

        df2 = df.groupBy("department", "state").sum("salary", "bonus").sort("department", "state")
        res = df2.collect()
        assert (
            str(res) == "[Row(department='Finance', state='CA', sum(salary)=189000, sum(bonus)=47000), "
            "Row(department='Finance', state='NY', sum(salary)=162000, sum(bonus)=34000), "
            "Row(department='Marketing', state='CA', sum(salary)=80000, sum(bonus)=18000), "
            "Row(department='Marketing', state='NY', sum(salary)=91000, sum(bonus)=21000), "
            "Row(department='Sales', state='CA', sum(salary)=81000, sum(bonus)=23000), "
            "Row(department='Sales', state='NY', sum(salary)=176000, sum(bonus)=30000)]"
        )

        df2 = (
            df.groupBy("department")
            .agg(
                sum("salary").alias("sum_salary"),
                avg("salary").alias("avg_salary"),
                sum("bonus").alias("sum_bonus"),
                max("bonus").alias("max_bonus"),
                any_value("state").alias("any_state"),
                approx_count_distinct("state").alias("distinct_state"),
            )
            .sort("department")
        )
        res = df2.collect()
        assert (
            str(res)
            == "[Row(department='Finance', sum_salary=351000, avg_salary=87750.0, sum_bonus=81000, max_bonus=24000, "
            "any_state='CA', distinct_state=2), Row(department='Marketing', sum_salary=171000, avg_salary=85500.0, "
            "sum_bonus=39000, max_bonus=21000, any_state='CA', distinct_state=2), Row(department='Sales', "
            "sum_salary=257000, avg_salary=85666.66666666667, sum_bonus=53000, max_bonus=23000, any_state='NY', "
            "distinct_state=2)]"
        )

        df2 = (
            df.groupBy("department")
            .agg(
                sum("salary").alias("sum_salary"),
                avg("salary").alias("avg_salary"),
                sum("bonus").alias("sum_bonus"),
                max("bonus").alias("max_bonus"),
                any_value("state").alias("any_state"),
            )
            .where(col("sum_bonus") >= 50000)
            .sort("department")
        )
        res = df2.collect()
        print(str(res))
        assert (
            str(res)
            == "[Row(department='Finance', sum_salary=351000, avg_salary=87750.0, sum_bonus=81000, max_bonus=24000, "
            "any_state='CA'), Row(department='Sales', sum_salary=257000, avg_salary=85666.66666666667, "
            "sum_bonus=53000, max_bonus=23000, any_state='NY')]"
        )

        df = spark.createDataFrame(
            [
                (1, 1, "a"),
                (2, 2, "a"),
                (2, 3, "a"),
                (5, 4, "a"),
            ],
            schema=["age", "other_variable", "group"],
        )
        df2 = df.groupBy("group").agg(
            covar_pop("age", "other_variable").alias("covar_pop"),
            covar_samp("age", "other_variable").alias("covar_samp"),
        )
        res = df2.collect()
        assert str(res) == "[Row(group='a', covar_pop=1.5, covar_samp=2.0)]"

    def test_group_by_empty(self, spark):
        df = spark.createDataFrame(
            [(2, 1.0, "1"), (2, 2.0, "2"), (2, 3.0, "3"), (5, 4.0, "4")], schema=["age", "extra", "name"]
        )

        res = df.groupBy().avg().collect()
        assert str(res) == "[Row(avg(age)=2.75, avg(extra)=2.5)]"

        res = df.groupBy(["name", "age"]).count().sort("name").collect()
        assert (
            str(res) == "[Row(name='1', age=2, count=1), Row(name='2', age=2, count=1), Row(name='3', age=2, count=1), "
            "Row(name='4', age=5, count=1)]"
        )

        res = df.groupBy("name").count().columns
        assert res == ["name", "count"]

    def test_group_by_first_and_last(self, spark):
        df = spark.createDataFrame([("Alice", 2), ("Bob", 5), ("Alice", None)], ("name", "age"))

        res = (
            df.groupBy("name")
            .agg(first("age").alias("first_age"), last("age").alias("last_age"))
            .orderBy("name")
            .collect()
        )

        # A grouped first and last read the rows in no set order, in Spark as in DuckDB. With three rows both still
        # come from different rows of Alice's, so the pair is her two values in either order.
        alice, bob = res
        assert alice.name == "Alice"
        assert {alice.first_age, alice.last_age} == {None, 2}
        assert bob == Row(name="Bob", first_age=5, last_age=5)

    def test_standard_deviations(self, spark):
        df = spark.createDataFrame(
            [
                (1, "a"),
                (2, "a"),
                (3, "a"),
                (4, "a"),
                (5, "a"),
                (6, "a"),
            ],
            schema=["value", "group"],
        )

        res = (
            df.groupBy("group")
            .agg(
                stddev_samp("value").alias("stddev_samp"),
                stddev("value").alias("stddev"),
                std("value").alias("std"),
                stddev_pop("value").alias("stddev_pop"),
            )
            .collect()
        )
        r = res[0]

        samp = 1.8708286933869
        assert pytest.approx(r.stddev_samp) == samp
        assert pytest.approx(r.stddev) == samp
        assert pytest.approx(r.std) == samp
        assert pytest.approx(r.stddev_pop) == 1.707825127659

    def test_variances(self, spark):
        df = spark.createDataFrame(
            [
                (1, "a"),
                (2, "a"),
                (3, "a"),
                (4, "a"),
                (5, "a"),
                (6, "a"),
            ],
            schema=["value", "group"],
        )

        res = (
            df.groupBy("group")
            .agg(
                var_samp("value").alias("var_samp"),
                var_pop("value").alias("var_pop"),
                variance("value").alias("variance"),
            )
            .collect()
        )
        r = res[0]

        samp = 3.5
        assert pytest.approx(r.var_samp) == samp
        assert pytest.approx(r.variance) == samp
        assert pytest.approx(r.var_pop) == 2.9166666666666

    def test_group_by_mean(self, spark):
        df = spark.createDataFrame(
            [
                ("Java", 2012, 20000),
                ("dotNET", 2012, 5000),
                ("Java", 2012, 22000),
                ("dotNET", 2012, 10000),
                ("dotNET", 2013, 48000),
                ("Java", 2013, 30000),
            ],
            schema=("course", "year", "earnings"),
        )

        res = df.groupBy("course").agg(median("earnings").alias("m")).collect()

        assert sorted(res, key=lambda x: x.course) == [Row(course="Java", m=22000), Row(course="dotNET", m=10000)]

    def test_group_by_mode(self, spark):
        df = spark.createDataFrame(
            [
                ("Java", 2012, 20000),
                ("dotNET", 2012, 5000),
                ("Java", 2012, 20000),
                ("dotNET", 2012, 5000),
                ("dotNET", 2013, 48000),
                ("Java", 2013, 30000),
            ],
            schema=("course", "year", "earnings"),
        )

        res = df.groupby("course").agg(mode("year").alias("mode")).collect()

        assert sorted(res, key=lambda x: x.course) == [Row(course="Java", mode=2012), Row(course="dotNET", mode=2012)]

    def test_group_by_product(self, spark):
        df = spark.range(1, 10).toDF("x").withColumn("mod3", col("x") % 3)
        res = df.groupBy("mod3").agg(product("x").alias("product")).orderBy("mod3").collect()
        assert res == [Row(mod3=0, product=162), Row(mod3=1, product=28), Row(mod3=2, product=80)]

    def test_group_by_skewness(self, spark):
        df = spark.createDataFrame([[1, "A"], [1, "A"], [2, "A"]], ["c", "group"])
        res = df.groupBy("group").agg(skewness(df.c).alias("v")).collect()
        assert pytest.approx(res[0].v) == 0.7071067811865475


@pytest.fixture
def mixed(spark):
    rows = [(1, 2, 1.5, "x", True), (1, 3, None, "y", False), (2, 5, 2.5, "x", None)]
    return spark.createDataFrame(rows, "a int, b long, d double, s string, f boolean")


def _sorted_tuples(df):
    return sorted((tuple(r) for r in df.collect()), key=repr)


class TestGroupedDataWithoutColumns:
    @pytest.mark.parametrize(
        ("method", "name", "x", "y"),
        [
            ("sum", "sum", (3, 7, 4.0), (1, 3, None)),
            ("avg", "avg", (1.5, 3.5, 2.0), (1.0, 3.0, None)),
            ("mean", "avg", (1.5, 3.5, 2.0), (1.0, 3.0, None)),
            ("min", "min", (1, 2, 1.5), (1, 3, None)),
            ("max", "max", (2, 5, 2.5), (1, 3, None)),
        ],
    )
    def test_every_numeric_column(self, mixed, method, name, x, y):
        res = getattr(mixed.groupBy("s"), method)()
        assert res.columns == ["s", f"{name}(a)", f"{name}(b)", f"{name}(d)"]
        assert _sorted_tuples(res) == [("x", *x), ("y", *y)]

    def test_keys_are_aggregated_too(self, mixed):
        res = mixed.groupBy("a").sum()
        assert res.columns == ["a", "sum(a)", "sum(b)", "sum(d)"]
        assert _sorted_tuples(res) == [(1, 2, 5, 1.5), (2, 2, 5, 2.5)]
        assert mixed.groupBy("A").max().columns == ["A", "max(a)", "max(b)", "max(d)"]

    def test_without_keys(self, mixed):
        assert mixed.groupBy().sum().collect() == [Row(**{"sum(a)": 4, "sum(b)": 10, "sum(d)": 4.0})]
        assert _sorted_tuples(mixed.groupBy().max()) == [(2, 5, 2.5)]


class TestGroupByExpressions:
    def test_expression_keys(self, spark, mixed):
        res = mixed.groupBy(mixed.a + 1).count()
        assert res.columns == ["(a + 1)", "count"]
        assert _sorted_tuples(res) == [(2, 2), (3, 1)]
        res = mixed.groupBy((col("a") * 10).alias("k")).agg(sum("b"))
        assert res.columns == ["k", "sum(b)"]
        assert _sorted_tuples(res) == [(10, 5), (20, 5)]
        assert _sorted_tuples(mixed.groupBy(mixed.a + 1).sum()) == [(2, 2, 5, 1.5), (3, 2, 5, 2.5)]

    @pytest.mark.parametrize("derive", ["withColumn", "filter", "select"])
    def test_expression_keys_on_derived_frames(self, mixed, derive):
        derived = {
            "withColumn": lambda: mixed.withColumn("c", col("b") * 2),
            "filter": lambda: mixed.filter(mixed.b > 0),
            "select": lambda: mixed.select("a", "b"),
        }[derive]()
        res = derived.groupBy(derived.a + 1).agg(sum("b"))
        assert res.columns == ["(a + 1)", "sum(b)"]
        assert _sorted_tuples(res) == [(2, 5), (3, 5)]

    def test_expression_keys_on_a_join(self, spark, mixed):
        other = spark.createDataFrame([(1, "p"), (2, "q")], "a int, w string")
        joined = mixed.join(other, "a")
        res = joined.groupBy(joined.w, joined.s).count()
        assert res.columns == ["w", "s", "count"]
        assert _sorted_tuples(res) == [("p", "x", 1), ("p", "y", 1), ("q", "x", 1)]
        assert _sorted_tuples(joined.groupBy(other.w).count()) == [("p", 2), ("q", 1)]


class TestGroupedDataColumnNames:
    @pytest.fixture
    def odd_names(self, spark):
        return spark.createDataFrame([("x", 5, 1, 2), ("x", 7, 1, 3)], ["k", "2024", "null", "my value"])

    @pytest.mark.parametrize(
        ("method", "values"),
        [("sum", (12, 2, 5)), ("avg", (6.0, 1.0, 2.5)), ("max", (7, 1, 3)), ("min", (5, 1, 2))],
    )
    def test_every_numeric_column(self, odd_names, method, values):
        res = getattr(odd_names.groupBy("k"), method)()
        assert res.columns == ["k", f"{method}(2024)", f"{method}(null)", f"{method}(my value)"]
        assert [tuple(r) for r in res.collect()] == [("x", *values)]

    def test_named_columns(self, odd_names):
        assert [tuple(r) for r in odd_names.groupBy("k").sum("2024").collect()] == [("x", 12)]
        res = odd_names.groupBy("k").mean("my value")
        assert res.columns == ["k", "avg(my value)"]
        assert [tuple(r) for r in res.collect()] == [("x", 2.5)]
        res = odd_names.groupBy("k").sum("MY VALUE")
        assert res.columns == ["k", "sum(MY VALUE)"]
        assert [tuple(r) for r in res.collect()] == [("x", 5)]

    def test_global(self, odd_names):
        res = odd_names.groupBy().sum()
        assert res.columns == ["sum(2024)", "sum(null)", "sum(my value)"]
        assert [tuple(r) for r in res.collect()] == [(12, 2, 5)]
        assert odd_names.groupBy().count().collect() == [Row(count=2)]
        assert odd_names.groupBy("k").count().columns == ["k", "count"]

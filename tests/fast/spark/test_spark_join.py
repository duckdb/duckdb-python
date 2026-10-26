import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException, IllegalArgumentException, error_condition
from spark_namespace.sql import functions as sf
from spark_namespace.sql.functions import col, lit
from spark_namespace.sql.types import (
    Row,
)

import duckdb

if USE_ACTUAL_SPARK:
    import pyspark

# Spark raises AnalysisException for every ambiguous reference; DuckDB's binder raises its own error for some.
AMBIGUOUS_REFERENCE_ERRORS = (AnalysisException, duckdb.BinderException)


@pytest.fixture
def dataframe_a(spark):
    emp = [
        (1, "Smith", -1, "2018", "10", "M", 3000),
        (2, "Rose", 1, "2010", "20", "M", 4000),
        (3, "Williams", 1, "2010", "10", "M", 1000),
        (4, "Jones", 2, "2005", "10", "F", 2000),
        (5, "Brown", 2, "2010", "40", "", -1),
        (6, "Brown", 2, "2010", "50", "", -1),
    ]
    empColumns = ["emp_id", "name", "superior_emp_id", "year_joined", "emp_dept_id", "gender", "salary"]
    dataframe = spark.createDataFrame(data=emp, schema=empColumns)
    return dataframe


@pytest.fixture
def dataframe_b(spark):
    dept = [("Finance", 10), ("Marketing", 20), ("Sales", 30), ("IT", 40)]
    deptColumns = ["dept_name", "dept_id"]
    dataframe = spark.createDataFrame(data=dept, schema=deptColumns)
    return dataframe


class TestDataFrameJoin:
    def test_inner_join(self, dataframe_a, dataframe_b):
        df = dataframe_a.join(dataframe_b, dataframe_a.emp_dept_id == dataframe_b.dept_id, "inner")
        df = df.sort(*df.columns)
        res = df.collect()
        expected = [
            Row(
                emp_id=1,
                name="Smith",
                superior_emp_id=-1,
                year_joined="2018",
                emp_dept_id="10",
                gender="M",
                salary=3000,
                dept_name="Finance",
                dept_id=10,
            ),
            Row(
                emp_id=2,
                name="Rose",
                superior_emp_id=1,
                year_joined="2010",
                emp_dept_id="20",
                gender="M",
                salary=4000,
                dept_name="Marketing",
                dept_id=20,
            ),
            Row(
                emp_id=3,
                name="Williams",
                superior_emp_id=1,
                year_joined="2010",
                emp_dept_id="10",
                gender="M",
                salary=1000,
                dept_name="Finance",
                dept_id=10,
            ),
            Row(
                emp_id=4,
                name="Jones",
                superior_emp_id=2,
                year_joined="2005",
                emp_dept_id="10",
                gender="F",
                salary=2000,
                dept_name="Finance",
                dept_id=10,
            ),
            Row(
                emp_id=5,
                name="Brown",
                superior_emp_id=2,
                year_joined="2010",
                emp_dept_id="40",
                gender="",
                salary=-1,
                dept_name="IT",
                dept_id=40,
            ),
        ]
        assert sorted(res) == sorted(expected)

    @pytest.mark.parametrize("how", ["outer", "fullouter", "full", "full_outer"])
    def test_outer_join(self, dataframe_a, dataframe_b, how):
        df = dataframe_a.join(dataframe_b, dataframe_a.emp_dept_id == dataframe_b.dept_id, how)
        df = df.sort(*df.columns)
        res1 = df.collect()
        assert sorted(res1, key=lambda x: x.emp_id or 0) == sorted(
            [
                Row(
                    emp_id=1,
                    name="Smith",
                    superior_emp_id=-1,
                    year_joined="2018",
                    emp_dept_id="10",
                    gender="M",
                    salary=3000,
                    dept_name="Finance",
                    dept_id=10,
                ),
                Row(
                    emp_id=2,
                    name="Rose",
                    superior_emp_id=1,
                    year_joined="2010",
                    emp_dept_id="20",
                    gender="M",
                    salary=4000,
                    dept_name="Marketing",
                    dept_id=20,
                ),
                Row(
                    emp_id=3,
                    name="Williams",
                    superior_emp_id=1,
                    year_joined="2010",
                    emp_dept_id="10",
                    gender="M",
                    salary=1000,
                    dept_name="Finance",
                    dept_id=10,
                ),
                Row(
                    emp_id=4,
                    name="Jones",
                    superior_emp_id=2,
                    year_joined="2005",
                    emp_dept_id="10",
                    gender="F",
                    salary=2000,
                    dept_name="Finance",
                    dept_id=10,
                ),
                Row(
                    emp_id=5,
                    name="Brown",
                    superior_emp_id=2,
                    year_joined="2010",
                    emp_dept_id="40",
                    gender="",
                    salary=-1,
                    dept_name="IT",
                    dept_id=40,
                ),
                Row(
                    emp_id=6,
                    name="Brown",
                    superior_emp_id=2,
                    year_joined="2010",
                    emp_dept_id="50",
                    gender="",
                    salary=-1,
                    dept_name=None,
                    dept_id=None,
                ),
                Row(
                    emp_id=None,
                    name=None,
                    superior_emp_id=None,
                    year_joined=None,
                    emp_dept_id=None,
                    gender=None,
                    salary=None,
                    dept_name="Sales",
                    dept_id=30,
                ),
            ],
            key=lambda x: x.emp_id or 0,
        )

    @pytest.mark.parametrize("how", ["right", "rightouter", "right_outer"])
    def test_right_join(self, dataframe_a, dataframe_b, how):
        df = dataframe_a.join(dataframe_b, dataframe_a.emp_dept_id == dataframe_b.dept_id, how)
        df = df.sort(*df.columns)
        res = df.collect()
        assert sorted(res, key=lambda x: x.emp_id or 0) == sorted(
            [
                Row(
                    emp_id=1,
                    name="Smith",
                    superior_emp_id=-1,
                    year_joined="2018",
                    emp_dept_id="10",
                    gender="M",
                    salary=3000,
                    dept_name="Finance",
                    dept_id=10,
                ),
                Row(
                    emp_id=2,
                    name="Rose",
                    superior_emp_id=1,
                    year_joined="2010",
                    emp_dept_id="20",
                    gender="M",
                    salary=4000,
                    dept_name="Marketing",
                    dept_id=20,
                ),
                Row(
                    emp_id=3,
                    name="Williams",
                    superior_emp_id=1,
                    year_joined="2010",
                    emp_dept_id="10",
                    gender="M",
                    salary=1000,
                    dept_name="Finance",
                    dept_id=10,
                ),
                Row(
                    emp_id=4,
                    name="Jones",
                    superior_emp_id=2,
                    year_joined="2005",
                    emp_dept_id="10",
                    gender="F",
                    salary=2000,
                    dept_name="Finance",
                    dept_id=10,
                ),
                Row(
                    emp_id=5,
                    name="Brown",
                    superior_emp_id=2,
                    year_joined="2010",
                    emp_dept_id="40",
                    gender="",
                    salary=-1,
                    dept_name="IT",
                    dept_id=40,
                ),
                Row(
                    emp_id=None,
                    name=None,
                    superior_emp_id=None,
                    year_joined=None,
                    emp_dept_id=None,
                    gender=None,
                    salary=None,
                    dept_name="Sales",
                    dept_id=30,
                ),
            ],
            key=lambda x: x.emp_id or 0,
        )

    @pytest.mark.parametrize("how", ["semi", "leftsemi", "left_semi"])
    def test_semi_join(self, dataframe_a, dataframe_b, how):
        df = dataframe_a.join(dataframe_b, dataframe_a.emp_dept_id == dataframe_b.dept_id, how)
        df = df.sort(*df.columns)
        res = df.collect()
        assert sorted(res) == sorted(
            [
                Row(
                    emp_id=1,
                    name="Smith",
                    superior_emp_id=-1,
                    year_joined="2018",
                    emp_dept_id="10",
                    gender="M",
                    salary=3000,
                ),
                Row(
                    emp_id=2,
                    name="Rose",
                    superior_emp_id=1,
                    year_joined="2010",
                    emp_dept_id="20",
                    gender="M",
                    salary=4000,
                ),
                Row(
                    emp_id=3,
                    name="Williams",
                    superior_emp_id=1,
                    year_joined="2010",
                    emp_dept_id="10",
                    gender="M",
                    salary=1000,
                ),
                Row(
                    emp_id=4,
                    name="Jones",
                    superior_emp_id=2,
                    year_joined="2005",
                    emp_dept_id="10",
                    gender="F",
                    salary=2000,
                ),
                Row(
                    emp_id=5,
                    name="Brown",
                    superior_emp_id=2,
                    year_joined="2010",
                    emp_dept_id="40",
                    gender="",
                    salary=-1,
                ),
            ]
        )

    @pytest.mark.parametrize("how", ["anti", "leftanti", "left_anti"])
    def test_anti_join(self, dataframe_a, dataframe_b, how):
        df = dataframe_a.join(dataframe_b, dataframe_a.emp_dept_id == dataframe_b.dept_id, how)
        df = df.sort(*df.columns)
        res = df.collect()
        assert res == [
            Row(emp_id=6, name="Brown", superior_emp_id=2, year_joined="2010", emp_dept_id="50", gender="", salary=-1)
        ]

    def test_self_join(self, dataframe_a):
        empDF = dataframe_a

        df = (
            empDF.alias("emp1")
            .join(empDF.alias("emp2"), col("emp1.superior_emp_id") == col("emp2.emp_id"), "inner")
            .select(
                col("emp1.emp_id"),
                col("emp1.name"),
                col("emp2.emp_id").alias("superior_emp_id"),
                col("emp2.name").alias("superior_emp_name"),
            )
        )
        df = df.orderBy(*df.columns)
        res = df.collect()
        assert sorted(res, key=lambda x: x.emp_id) == sorted(
            [
                Row(emp_id=2, name="Rose", superior_emp_id=1, superior_emp_name="Smith"),
                Row(emp_id=3, name="Williams", superior_emp_id=1, superior_emp_name="Smith"),
                Row(emp_id=4, name="Jones", superior_emp_id=2, superior_emp_name="Rose"),
                Row(emp_id=5, name="Brown", superior_emp_id=2, superior_emp_name="Rose"),
                Row(emp_id=6, name="Brown", superior_emp_id=2, superior_emp_name="Rose"),
            ],
            key=lambda x: x.emp_id,
        )

    def test_cross_join(self, spark):
        data1 = [(1, "Carol"), (2, "Alice"), (3, "Dave")]
        data2 = [(4, "A"), (5, "B")]
        df1 = spark.createDataFrame(data1, ["age", "name"])
        df2 = spark.createDataFrame(data2, ["id", "rank"])

        df = df1.crossJoin(df2)

        res = df.orderBy("rank", "age").collect()

        assert sorted(res) == sorted(
            [
                Row(age=1, name="Carol", id=4, rank="A"),
                Row(age=2, name="Alice", id=4, rank="A"),
                Row(age=3, name="Dave", id=4, rank="A"),
                Row(age=1, name="Carol", id=5, rank="B"),
                Row(age=2, name="Alice", id=5, rank="B"),
                Row(age=3, name="Dave", id=5, rank="B"),
            ]
        )

    def test_join_with_using_clause(self, spark, dataframe_a):
        dataframe_a = dataframe_a.select("name", "year_joined")

        df = dataframe_a.alias("df1")
        df2 = dataframe_a.alias("df2")
        res = df.join(df2, ["name", "year_joined"]).sort("name", "year_joined")
        res = res.collect()
        assert res == [
            Row(name="Brown", year_joined="2010"),
            Row(name="Brown", year_joined="2010"),
            Row(name="Brown", year_joined="2010"),
            Row(name="Brown", year_joined="2010"),
            Row(name="Jones", year_joined="2005"),
            Row(name="Rose", year_joined="2010"),
            Row(name="Smith", year_joined="2018"),
            Row(name="Williams", year_joined="2010"),
        ]

    def test_join_with_common_column(self, spark, dataframe_a):
        dataframe_a = dataframe_a.select("name", "year_joined")

        df = dataframe_a.alias("df1")
        df2 = dataframe_a.alias("df2")
        res = df.join(df2, df.name == df2.name).sort("df1.name")
        res = res.collect()
        assert (
            str(res) == "[Row(name='Brown', year_joined='2010', name='Brown', year_joined='2010'), Row(name='Brown', "
            "year_joined='2010', name='Brown', year_joined='2010'), Row(name='Brown', year_joined='2010', "
            "name='Brown', year_joined='2010'), Row(name='Brown', year_joined='2010', name='Brown', "
            "year_joined='2010'), Row(name='Jones', year_joined='2005', name='Jones', year_joined='2005'), "
            "Row(name='Rose', year_joined='2010', name='Rose', year_joined='2010'), Row(name='Smith', "
            "year_joined='2018', name='Smith', year_joined='2018'), Row(name='Williams', year_joined='2010', "
            "name='Williams', year_joined='2010')]"
        )

    def test_join_on_joined_data_error(self, spark):
        df = spark.createDataFrame([(2, "Alice"), (5, "Bob")]).toDF("age", "name")
        df2 = spark.createDataFrame([Row(height=80, name="Tom"), Row(height=85, name="Bob")], ["height", "name"])

        first_join = df.join(df2, "name")
        second_join = df.join(df2, "name")
        third_join = first_join.join(second_join, "name")

        third_join.show()

    def test_join_on_column_name_stored_in_variable(self, spark):
        df = spark.createDataFrame([(2, "Alice"), (5, "Bob")]).toDF("age", "name")
        df2 = spark.createDataFrame([Row(height=80, name="Tom"), Row(height=85, name="Bob")], ["height", "name"])

        column = "name"
        col1 = df[column]
        col2 = df2[column]
        res = df.join(df2, col1 == col2)
        res = res.collect()
        assert str(res) == "[Row(age=5, name='Bob', height=85, name='Bob')]"


@pytest.fixture
def people(spark):
    return spark.createDataFrame([(1, "Ann", 2), (2, "Bob", 3), (3, "Cid", 1)], ["id", "name", "manager_id"])


@pytest.fixture
def teams(spark):
    return spark.createDataFrame([(1, "red"), (2, "blue"), (4, "gold")], ["id", "team"])


class TestJoinColumnResolution:
    def test_self_join_on_column_name(self, people):
        df = people.join(people, "id")
        assert df.columns == ["id", "name", "manager_id", "name", "manager_id"]
        assert sorted(map(tuple, df.collect())) == [
            (1, "Ann", 2, "Ann", 2),
            (2, "Bob", 3, "Bob", 3),
            (3, "Cid", 1, "Cid", 1),
        ]

    @pytest.mark.parametrize("how", ["inner", "left", "full"])
    def test_join_filtered_views_of_one_dataframe(self, people, how):
        low = people.filter(col("id") < 3)
        high = people.filter(col("id") > 1)
        df = low.join(high, "id", how).select("id")
        expected = {"inner": [2], "left": [1, 2], "full": [1, 2, 3]}[how]
        assert sorted(row.id for row in df.collect()) == expected

    def test_cross_join_with_itself(self, people):
        assert people.crossJoin(people).count() == 9
        assert people.join(people).count() == 9

    def test_join_without_condition_is_a_cross_join(self, people, teams):
        df = people.join(teams)
        assert df.columns == ["id", "name", "manager_id", "id", "team"]
        assert df.count() == 9

    def test_self_join_with_aliases(self, people):
        df = (
            people.alias("a")
            .join(people.alias("b"), col("a.manager_id") == col("b.id"))
            .select(col("a.name").alias("employee"), col("b.name").alias("manager"))
        )
        assert sorted(df.collect()) == [
            Row(employee="Ann", manager="Bob"),
            Row(employee="Bob", manager="Cid"),
            Row(employee="Cid", manager="Ann"),
        ]

    def test_join_two_joins_of_the_same_inputs(self, people, teams):
        first = people.join(teams, "id")
        second = people.join(teams, "id")
        df = first.join(second, "id")
        assert df.columns == ["id", "name", "manager_id", "team", "name", "manager_id", "team"]
        assert sorted(map(tuple, df.collect())) == [
            (1, "Ann", 2, "red", "Ann", 2, "red"),
            (2, "Bob", 3, "blue", "Bob", 3, "blue"),
        ]

    def test_join_a_join_with_one_of_its_inputs(self, people, teams):
        df = people.join(teams, "id").join(people, "id")
        assert df.columns == ["id", "name", "manager_id", "team", "name", "manager_id"]
        assert sorted(map(tuple, df.collect())) == [
            (1, "Ann", 2, "red", "Ann", 2),
            (2, "Bob", 3, "blue", "Bob", 3),
        ]

    def test_select_the_join_column_of_each_side(self, people, teams):
        df = people.join(teams, people.id == teams.id, "left").select(people.id, teams.id, "team")
        assert df.columns == ["id", "id", "team"]
        assert sorted(map(tuple, df.collect())) == [
            (1, 1, "red"),
            (2, 2, "blue"),
            (3, None, None),
        ]

    def test_column_on_both_sides_is_ambiguous(self, people, teams):
        joined = people.join(teams, people.id == teams.id)
        with pytest.raises(AMBIGUOUS_REFERENCE_ERRORS):
            joined.select(col("id")).collect()
        for reference in [lambda: joined.id, lambda: joined["id"], lambda: joined.select("id")]:
            with pytest.raises(AnalysisException) as error:
                reference().collect()
            assert error_condition(error.value) == "AMBIGUOUS_REFERENCE"
            # Spark 3.5 returns a Java map.
            assert dict(error.value.getMessageParameters()) == {"name": "`id`", "referenceNames": "[`id`, `id`]"}
            assert error.value.getSqlState() == "42704"

    def test_column_of_a_joined_dataframe(self, people, teams):
        joined = people.join(teams, people.id == teams.id)
        assert sorted(joined.select(joined.name, joined["team"]).collect()) == [
            Row(name="Ann", team="red"),
            Row(name="Bob", team="blue"),
        ]
        assert joined.filter(joined.team == "red").select("name").collect() == [Row(name="Ann")]

    @pytest.mark.parametrize(
        ("how", "expected_ids"),
        [("inner", [1, 2]), ("left", [1, 2, 3]), ("right", [1, 2, 4]), ("full", [1, 2, 3, 4])],
    )
    def test_join_key_of_a_join_on_column_names(self, spark, people, teams, how, expected_ids):
        joined = people.join(teams, "id", how)
        assert sorted(row.id for row in joined.select(joined.id).collect()) == expected_ids
        flags = spark.createDataFrame([(1, True), (4, False)], ["id", "flag"])
        df = joined.join(flags, joined.id == flags.id).select(joined.id, flags.flag)
        assert sorted(df.collect()) == [Row(id=i, flag=i == 1) for i in [1, 4] if i in expected_ids]

    def test_with_column_on_a_column_of_a_joined_dataframe(self, spark):
        visits = spark.createDataFrame([(1, "Marc", 10), (2, "Bob", 20)], ["site_id", "person", "minutes"])
        sites = spark.createDataFrame([(1, "A"), (2, "B"), (3, "C")], ["site_id", "name"])
        visits = visits.join(sites, on="site_id", how="left")
        visits = visits.withColumn("minutes", visits.minutes * 2)
        assert sorted(visits.collect()) == [
            Row(site_id=1, person="Marc", minutes=20, name="A"),
            Row(site_id=2, person="Bob", minutes=40, name="B"),
        ]

    def test_chained_joins(self, spark, people, teams):
        flags = spark.createDataFrame([(1, True), (2, False)], ["id", "flag"])
        df = (
            people.join(teams, people.id == teams.id)
            .join(flags, people.id == flags.id)
            .select(people.id, teams.team, flags.flag)
        )
        assert sorted(df.collect()) == [Row(id=1, team="red", flag=True), Row(id=2, team="blue", flag=False)]
        df = people.join(teams, "id").join(flags, "id")
        assert df.columns == ["id", "name", "manager_id", "team", "flag"]
        assert sorted(df.collect()) == [
            Row(id=1, name="Ann", manager_id=2, team="red", flag=True),
            Row(id=2, name="Bob", manager_id=3, team="blue", flag=False),
        ]

    def test_join_after_with_column(self, people, teams):
        doubled = people.withColumn("twice", col("id") * 2)
        df = doubled.join(teams, doubled.id == teams.id).select(doubled.id, doubled.twice, teams.team)
        assert sorted(df.collect()) == [Row(id=1, twice=2, team="red"), Row(id=2, twice=4, team="blue")]

    def test_with_column_after_join_with_a_shared_column_name(self, people, teams):
        joined = people.join(teams, people.id == teams.id)
        df = joined.withColumn("one", lit(1))
        assert df.columns == ["id", "name", "manager_id", "id", "team", "one"]
        assert sorted(map(tuple, df.collect())) == [
            (1, "Ann", 2, 1, "red", 1),
            (2, "Bob", 3, 2, "blue", 1),
        ]
        df = joined.withColumn("team", lit("x")).withColumns({"name": lit("y")})
        assert sorted(map(tuple, df.collect())) == [
            (1, "y", 2, 1, "x"),
            (2, "y", 3, 2, "x"),
        ]
        df = joined.withColumnRenamed("team", "colour")
        assert df.columns == ["id", "name", "manager_id", "id", "colour"]
        df = joined.withColumn("doubled", people.manager_id * 2).select("doubled", "team")
        assert sorted(df.collect()) == [Row(doubled=4, team="red"), Row(doubled=6, team="blue")]

    def test_drop_the_join_column_of_one_side(self, people, teams):
        joined = people.join(teams, people.id == teams.id, "left")
        assert sorted(joined.drop(teams.id).collect()) == [
            Row(id=1, name="Ann", manager_id=2, team="red"),
            Row(id=2, name="Bob", manager_id=3, team="blue"),
            Row(id=3, name="Cid", manager_id=1, team=None),
        ]
        df = joined.drop(people.id)
        assert df.columns == ["name", "manager_id", "id", "team"]
        assert sorted(df.collect(), key=lambda row: row.name) == [
            Row(name="Ann", manager_id=2, id=1, team="red"),
            Row(name="Bob", manager_id=3, id=2, team="blue"),
            Row(name="Cid", manager_id=1, id=None, team=None),
        ]

    def test_reference_through_a_dataframe_on_both_sides_raises(self, people):
        with pytest.raises(AMBIGUOUS_REFERENCE_ERRORS):
            people.join(people, "id").select(people.name).collect()
        with pytest.raises(AMBIGUOUS_REFERENCE_ERRORS):
            people.join(people, people.id == people.manager_id).collect()

    def test_self_join_on_the_same_column_reference(self, people):
        low = people.filter(col("id") < 3)
        high = people.filter(col("id") > 1)
        if USE_ACTUAL_SPARK:
            # Spark rewrites an equality between the same column of both sides to left equals right.
            assert people.join(people, people.id == people.id).count() == 3
            assert low.join(high, low.id == high.id).select(low.name).collect() == [Row(name="Bob")]
        else:
            with pytest.raises(duckdb.BinderException):
                people.join(people, people.id == people.id).collect()
            with pytest.raises(duckdb.BinderException):
                low.join(high, low.id == high.id).collect()

    @pytest.mark.parametrize(
        "derive",
        [
            lambda df: df.filter(col("team") != "none"),
            lambda df: df.where(col("team") != "none"),
            lambda df: df.orderBy("team"),
            lambda df: df.sort("team"),
            lambda df: df.limit(5),
            lambda df: df.distinct(),
            lambda df: df.dropDuplicates(),
            lambda df: df.filter(col("team") != "none").limit(5),
            lambda df: df.filter(col("team") != "none").alias("x"),
        ],
    )
    def test_columns_of_a_frame_derived_from_a_join(self, people, teams, derive):
        derived = derive(people.join(teams, "id"))
        assert sorted(derived.select(derived["name"], derived.team).collect()) == [
            Row(name="Ann", team="red"),
            Row(name="Bob", team="blue"),
        ]
        assert derived.filter(derived.team == "red").select(derived.id).collect() == [Row(id=1)]
        assert sorted(derived.select(people.name, teams.team).collect()) == [
            Row(name="Ann", team="red"),
            Row(name="Bob", team="blue"),
        ]

    def test_columns_of_a_cached_join(self, people, teams):
        cached = people.join(teams, "id").cache()
        assert sorted(cached.select(cached.name, cached["team"]).collect()) == [
            Row(name="Ann", team="red"),
            Row(name="Bob", team="blue"),
        ]

    def test_frame_derived_from_a_join_as_a_join_input(self, spark, people, teams):
        derived = people.join(teams, "id").filter(col("team") != "none")
        flags = spark.createDataFrame([(1, True), (2, False)], ["flag_id", "flag"])
        expected = [Row(name="Ann", flag=True), Row(name="Bob", flag=False)]
        df = derived.join(flags, col("id") == col("flag_id")).select("name", "flag")
        assert sorted(df.collect()) == expected
        df = derived.join(flags.withColumnRenamed("flag_id", "id"), "id").select("name", "flag")
        assert sorted(df.collect()) == expected
        if USE_ACTUAL_SPARK:
            df = derived.join(flags, derived.id == flags.flag_id).select(derived.name, flags.flag)
            assert sorted(df.collect()) == expected
        else:
            # The derived frame's columns resolve in its own scope, which a later join wraps in a subquery.
            with pytest.raises(duckdb.BinderException):
                derived.join(flags, derived.id == flags.flag_id).collect()


@pytest.fixture
def left(spark):
    return spark.createDataFrame([(1, 2), (2, 4)], ["a", "b"])


@pytest.fixture
def right(spark):
    return spark.createDataFrame([(1, 3), (3, 9)], ["a", "c"])


class TestDropAfterJoin:
    @pytest.mark.parametrize(
        ("how", "drop_left", "drop_right", "drop_name"),
        [
            ("inner", [(2, 3)], [(1, 2, 3)], [(2, 3)]),
            ("left", [(2, 3), (4, None)], [(1, 2, 3), (2, 4, None)], [(2, 3), (4, None)]),
            ("right", [(1, 2, 3), (3, None, 9)], [(2, 3), (None, 9)], [(2, 3), (None, 9)]),
            (
                "full",
                [(1, 2, 3), (2, 4, None), (3, None, 9)],
                [(1, 2, 3), (2, 4, None), (3, None, 9)],
                [(2, 3), (4, None), (None, 9)],
            ),
        ],
    )
    def test_drop_the_key_of_a_join_on_a_name(self, left, right, how, drop_left, drop_right, drop_name):
        # The key of a join on a name is the left input's, the right input's for a right join and neither's for a
        # full join, so dropping the other input's column changes nothing.
        joined = left.join(right, "a", how)

        def rows(df):
            return sorted(map(tuple, df.collect()), key=str)

        assert rows(joined.drop(left.a)) == drop_left
        assert rows(joined.drop(right.a)) == drop_right
        assert rows(joined.drop("a")) == drop_name
        assert rows(joined.drop(col("A"))) == drop_name
        assert joined.drop("A").columns == ["b", "c"]

    def test_drop_after_a_join_on_an_expression(self, left, right):
        joined = left.join(right, left.a == right.a)
        assert joined.drop(left.a).columns == ["b", "a", "c"]
        assert joined.drop(left.a).collect() == [Row(b=2, a=1, c=3)]
        assert joined.drop(right.a).columns == ["a", "b", "c"]
        assert joined.drop(left.a, right.a).columns == ["b", "c"]
        assert joined.drop("A").columns == ["b", "c"]
        assert joined.drop("b", left.a).columns == ["a", "c"]
        with pytest.raises(AnalysisException) as error:
            joined.drop(col("a"))
        assert error_condition(error.value) == "AMBIGUOUS_REFERENCE"

    def test_drop_through_a_frame_derived_from_a_join(self, left, right):
        assert left.join(right, "a").filter(left.a > 0).drop(left.a).columns == ["b", "c"]

    def test_drop_through_aliases(self, left):
        joined = left.alias("x").join(left.alias("y"), col("x.a") == col("y.a"))
        assert joined.drop(col("x.a")).columns == ["b", "a", "b"]
        assert sorted(map(tuple, joined.drop(col("y.b")).collect())) == [(1, 2, 1), (2, 4, 2)]


class TestCacheOfAJoin:
    def test_cache_keeps_join_input_references(self, people, teams):
        cached = people.join(teams, "id").cache()
        assert sorted(cached.select(people.name, teams.team).collect()) == [
            Row(name="Ann", team="red"),
            Row(name="Bob", team="blue"),
        ]

    def test_cache_keeps_duplicate_names(self, people, teams):
        cached = people.join(teams, people.id == teams.id).cache()
        assert cached.columns == ["id", "name", "manager_id", "id", "team"]
        with pytest.raises(AMBIGUOUS_REFERENCE_ERRORS):
            cached.select("id").collect()
        assert sorted(cached.select(teams.id).collect()) == [Row(id=1), Row(id=2)]


class TestSourceFramesAreDistinct:
    def test_reference_to_another_range(self, spark):
        with pytest.raises((AnalysisException, duckdb.BinderException)):
            spark.range(3).select(spark.range(5).id).collect()

    def test_reference_to_another_created_frame(self, spark):
        first = spark.createDataFrame([(1,)], ["x"])
        second = spark.createDataFrame([(1,)], ["x"])
        with pytest.raises((AnalysisException, duckdb.BinderException)):
            first.select(second.x).collect()

    def test_join_of_two_ranges(self, spark):
        small, large = spark.range(2), spark.range(3)
        assert sorted(small.join(large, small.id == large.id).select(small.id).collect()) == [Row(id=0), Row(id=1)]
        assert sorted(small.join(large, "id").collect()) == [Row(id=0), Row(id=1)]

    def test_join_of_two_reads_of_a_file(self, spark, tmp_path):
        path = str(tmp_path / "data.parquet")
        spark.createDataFrame([(1, "a"), (2, "b")], ["id", "v"]).write.parquet(path)
        first, second = spark.read.parquet(path), spark.read.parquet(path)
        df = first.join(second, first.id == second.id).select(first.id, second.v)
        assert sorted(df.collect()) == [Row(id=1, v="a"), Row(id=2, v="b")]


class TestDeliberateDifferences:
    def test_join_on_a_name_keeps_the_column_order(self, spark):
        a = spark.createDataFrame([(1, 10)], ["a", "k"])
        b = spark.createDataFrame([(2, 10)], ["b", "k"])
        joined = a.join(b, "k")
        if USE_ACTUAL_SPARK:
            assert joined.columns == ["k", "a", "b"]
        else:
            # The module keeps DuckDB's order: the left columns, then the right columns that are not keys.
            assert joined.columns == ["a", "k", "b"]
        assert joined.select("k", "a", "b").collect() == [Row(k=10, a=1, b=2)]

    def test_aliased_self_join_answers(self, people):
        def managers():
            return people.alias("a").join(people, col("a.manager_id") == people.id).select(people.name).collect()

        if USE_ACTUAL_SPARK:
            with pytest.raises(AnalysisException):
                managers()
        else:
            # The module can tell the sides apart, so it answers with the rows of the side without the alias.
            assert sorted(managers()) == [
                Row(name="Ann"),
                Row(name="Bob"),
                Row(name="Cid"),
            ]


@pytest.fixture
def xs(spark):
    return spark.createDataFrame([(1, "x1"), (2, "x2")], ["id", "x"])


@pytest.fixture
def ys(spark):
    return spark.createDataFrame([(1, "y1"), (3, "y3")], ["id", "y"])


_JOIN_TYPE_KEYS = [
    ("inner", [1]),
    ("cross", [1]),
    ("outer", [1, 2, 3]),
    ("full", [1, 2, 3]),
    ("fullouter", [1, 2, 3]),
    ("full_outer", [1, 2, 3]),
    ("left", [1, 2]),
    ("leftouter", [1, 2]),
    ("left_outer", [1, 2]),
    ("right", [1, 3]),
    ("rightouter", [1, 3]),
    ("right_outer", [1, 3]),
    ("semi", [1]),
    ("leftsemi", [1]),
    ("left_semi", [1]),
    ("anti", [2]),
    ("leftanti", [2]),
    ("left_anti", [2]),
]


class TestJoinTypes:
    @pytest.mark.parametrize(
        ("how", "keys"),
        [(how, keys) for name, keys in _JOIN_TYPE_KEYS for how in (name, name.upper())] + [("Left_Outer", [1, 2])],
    )
    def test_join_type_spellings(self, xs, ys, how, keys):
        assert sorted(row.id for row in xs.join(ys, "id", how).select("id").collect()) == keys

    @pytest.mark.parametrize(("how", "rows"), [("CROSS", [(1, "x1", 1, "y1")]), ("FULL", [(1, "x1", 1, "y1")])])
    def test_join_type_with_a_condition(self, xs, ys, how, rows):
        joined = xs.join(ys, xs.id == ys.id, how).filter(xs.id.isNotNull() & ys.id.isNotNull())
        assert sorted(map(tuple, joined.collect())) == rows

    def test_unknown_join_type_raises(self, xs, ys):
        # Spark 4.2.0 raises an AnalysisException, Spark 3.5.9 an IllegalArgumentException.
        with pytest.raises((AnalysisException, IllegalArgumentException), match="Unsupported join type"):
            xs.join(ys, "id", "sideways").collect()


@pytest.fixture
def staff(spark):
    return spark.createDataFrame([(1, "Ann"), (1, "Abe"), (2, "Bob")], ["id", "name"])


@pytest.fixture
def depts(spark):
    return spark.createDataFrame([(1, "red"), (2, "blue")], ["id", "team"])


class TestFramesDerivedFromAJoin:
    @pytest.mark.parametrize(
        "derive",
        [
            lambda j: j.filter(col("team") != "none").sort("team"),
            lambda j: j.filter(col("team") != "none").filter(col("name") != "none"),
            lambda j: j.sort("team").filter(col("team") != "none"),
            lambda j: j.sort("team").sort("name"),
            lambda j: j.limit(5).sort("name").filter(col("team") != "none"),
            lambda j: j.filter(col("team") != "none").sort("name").limit(5).distinct(),
            lambda j: j.distinct().filter(col("team") != "none").sort("name"),
            lambda j: j.filter(col("team") != "none").alias("q").sort("name"),
        ],
    )
    def test_names_in_chains_of_steps(self, staff, depts, derive):
        derived = derive(staff.join(depts, "id"))
        expected = [Row(name="Abe"), Row(name="Ann"), Row(name="Bob")]
        assert sorted(derived.select("name").collect()) == expected
        assert sorted(derived.select(col("name")).collect()) == expected
        assert sorted(derived.select(derived.name).collect()) == expected
        assert sorted(derived.select(staff.name).collect()) == expected
        assert sorted(derived.withColumn("n", lit(1)).select("name", "n").collect()) == [
            Row(name="Abe", n=1),
            Row(name="Ann", n=1),
            Row(name="Bob", n=1),
        ]
        assert derived.filter(col("team") == "blue").select("name").collect() == [Row(name="Bob")]

    @pytest.mark.parametrize(
        "derive",
        [
            lambda j: j.filter(col("team") != "none"),
            lambda j: j.filter(col("team") != "none").sort("team"),
        ],
    )
    def test_names_in_chains_over_a_cross_join(self, staff, depts, derive):
        derived = derive(staff.crossJoin(depts.withColumnRenamed("id", "dept_id")))
        assert derived.count() == 6
        assert sorted(derived.filter(col("team") == "red").select("name").collect()) == [
            Row(name="Abe"),
            Row(name="Ann"),
            Row(name="Bob"),
        ]
        assert sorted(derived.filter(col("dept_id") == 2).select(col("name"), staff.id).collect()) == [
            Row(name="Abe", id=1),
            Row(name="Ann", id=1),
            Row(name="Bob", id=2),
        ]

    def test_filter_after_a_limit(self, staff, depts):
        first = staff.join(depts, "id").orderBy("name").limit(1)
        assert first.filter(col("name") == "Bob").collect() == []
        assert first.collect() == [Row(id=1, name="Abe", team="red")]

    def test_aggregates_after_a_limit_or_a_sort(self, staff, depts):
        joined = staff.join(depts, "id")
        assert joined.limit(1).count() == 1
        assert joined.orderBy("name").limit(2).groupBy().count().collect() == [Row(count=2)]
        assert joined.orderBy("name").count() == 3
        assert sorted(joined.orderBy("name").groupBy("team").count().collect()) == [
            Row(team="blue", count=1),
            Row(team="red", count=2),
        ]

    def test_projection_after_a_distinct(self, staff, depts):
        distinct = staff.join(depts, "id").distinct()
        assert sorted(distinct.select("id").collect()) == [Row(id=1), Row(id=1), Row(id=2)]
        assert sorted(distinct.select(staff.id).collect()) == [Row(id=1), Row(id=1), Row(id=2)]
        assert distinct.select("team").distinct().count() == 2
        assert distinct.count() == 3

    def test_projection_after_a_sort_keeps_the_order(self, staff, depts):
        sorted_join = staff.join(depts, "id").orderBy(col("name").desc())
        assert sorted_join.select("name", col("id").alias("name_2")).collect() == [
            Row(name="Bob", name_2=2),
            Row(name="Ann", name_2=1),
            Row(name="Abe", name_2=1),
        ]
        assert sorted_join.select(col("team").alias("name")).collect() == [
            Row(name="blue"),
            Row(name="red"),
            Row(name="red"),
        ]

    def test_repeated_names_after_a_sort_of_a_filtered_join(self, staff, depts):
        joined = staff.join(depts, staff.id == depts.id)
        derived = joined.filter(col("team") != "none").sort("name")
        assert derived.columns == ["id", "name", "id", "team"]
        assert derived.drop(depts.id).collect() == [
            Row(id=1, name="Abe", team="red"),
            Row(id=1, name="Ann", team="red"),
            Row(id=2, name="Bob", team="blue"),
        ]
        with pytest.raises(AnalysisException) as error:
            derived.select("id")
        assert error_condition(error.value) == "AMBIGUOUS_REFERENCE"

    def test_to_df_after_a_join_with_repeated_names(self, staff, depts):
        renamed = staff.join(depts, staff.id == depts.id).toDF("a", "b", "c", "d")
        assert renamed.columns == ["a", "b", "c", "d"]
        assert sorted(renamed.collect()) == [
            Row(a=1, b="Abe", c=1, d="red"),
            Row(a=1, b="Ann", c=1, d="red"),
            Row(a=2, b="Bob", c=2, d="blue"),
        ]


class TestSemiAndAntiJoinColumns:
    @pytest.fixture
    def named(self, spark):
        return spark.createDataFrame([(1, "c")], ["id", "name"])

    @pytest.mark.parametrize(("how", "names"), [("left_anti", ["Bob"]), ("semi", ["Abe", "Ann"])])
    def test_name_both_inputs_have(self, staff, named, how, names):
        joined = staff.join(named, "id", how)
        assert sorted(row.name for row in joined.select("name").collect()) == names
        assert sorted(row.name for row in joined.select(joined.name).collect()) == names
        assert sorted(row.name for row in joined.withColumn("n", lit(1)).collect()) == names

    def test_name_both_inputs_have_on_a_condition(self, staff, named):
        joined = staff.join(named, staff.id == named.id, "semi")
        assert sorted(joined.select("name").collect()) == [Row(name="Abe"), Row(name="Ann")]


class TestDropThroughAnAlias:
    def test_drop_a_reference_to_the_aliased_frame(self, people):
        assert people.alias("q").drop(people.name).columns == ["id", "manager_id"]
        assert people.alias("q").alias("r").drop(people.name).columns == ["id", "manager_id"]
        assert people.alias("q").filter(col("id") > 1).drop(people.name).collect() == [
            Row(id=2, manager_id=3),
            Row(id=3, manager_id=1),
        ]
        assert people.alias("q").drop(col("q.name")).columns == ["id", "manager_id"]

    def test_drop_a_reference_to_another_frame(self, people, teams):
        assert people.drop(teams.id).columns == ["id", "name", "manager_id"]

    def test_select_a_reference_to_the_aliased_frame(self, people):
        assert people.alias("q").filter(col("id") == 1).select(people.name).collect() == [Row(name="Ann")]


def _rows(df):
    return sorted(map(tuple, df.collect()), key=str)


@pytest.fixture
def unmatched(spark):
    # A full join of inputs without a common key, so the two id columns differ in every row.
    left = spark.createDataFrame([(1, "a")], ["id", "x"])
    right = spark.createDataFrame([(2, "b")], ["id", "y"])
    return left, right, left.join(right, left.id == right.id, "full")


# Frames over the join with the columns id, x, id and y.
_OVER_THE_JOIN = {
    "join": lambda left, right, joined: joined,
    "alias": lambda left, right, joined: joined.alias("q"),
    "filter": lambda left, right, joined: joined.filter(col("x").isNull() | (col("x") == "a")),
    "sort": lambda left, right, joined: joined.orderBy("x"),
    "limit": lambda left, right, joined: joined.limit(10),
    "distinct": lambda left, right, joined: joined.distinct(),
    "projection": lambda left, right, joined: joined.select(left.id, "x", right.id, "y"),
}


class TestRepeatedColumnNames:
    @pytest.mark.parametrize("derive", _OVER_THE_JOIN.values(), ids=_OVER_THE_JOIN.keys())
    def test_frame_with_repeated_names_as_a_join_input(self, spark, unmatched, derive):
        other = spark.createDataFrame([(9,)], ["c"])
        joined = derive(*unmatched).crossJoin(other)
        assert joined.columns == ["id", "x", "id", "y", "c"]
        rows = [(1, "a", None, None, 9), (None, None, 2, "b", 9)]
        assert _rows(joined) == rows
        assert _rows(joined.withColumn("n", lit(1))) == [(*row, 1) for row in rows]
        assert _rows(joined.toDF("a", "b", "c", "d", "e")) == rows
        assert _rows(joined.drop("c")) == [row[:4] for row in rows]
        assert _rows(joined.drop(other.c)) == [row[:4] for row in rows]
        assert _rows(joined.withColumnRenamed("x", "z")) == rows

    @pytest.mark.parametrize("derive", _OVER_THE_JOIN.values(), ids=_OVER_THE_JOIN.keys())
    def test_repeated_name_in_a_frame_with_repeated_names_as_a_join_input(self, spark, unmatched, derive):
        other = spark.createDataFrame([(9,)], ["c"])
        joined = derive(*unmatched).crossJoin(other)
        for reference in [lambda: joined.select("id"), lambda: joined.select(col("id")), lambda: joined["id"]]:
            with pytest.raises(AnalysisException) as error:
                reference()
            assert error_condition(error.value) == "AMBIGUOUS_REFERENCE"
        with pytest.raises(AnalysisException) as error:
            derive(*unmatched).join(other, col("id") == col("c"))
        assert error_condition(error.value) == "AMBIGUOUS_REFERENCE"
        assert _rows(joined.select(col("x"), "c")) == [("a", 9), (None, 9)]

    @pytest.mark.parametrize(
        "use",
        [
            lambda df: df.select(col("id")),
            lambda df: df.select(col("id") + 1),
            lambda df: df.select(sf.sum("id")),
            lambda df: df.filter(col("id") > 0),
            lambda df: df.withColumn("n", col("id") * 2),
            lambda df: df.withColumns({"n": col("id")}),
            lambda df: df.groupBy("id").count(),
            lambda df: df.groupBy(col("id")).count(),
            lambda df: df.groupBy("x").agg(sf.max(col("id"))),
        ],
    )
    @pytest.mark.parametrize("derive", _OVER_THE_JOIN.values(), ids=_OVER_THE_JOIN.keys())
    def test_reference_to_a_repeated_name_is_ambiguous(self, unmatched, derive, use):
        with pytest.raises(AnalysisException) as error:
            use(derive(*unmatched)).collect()
        assert error_condition(error.value) == "AMBIGUOUS_REFERENCE"

    @pytest.mark.parametrize("derive", _OVER_THE_JOIN.values(), ids=_OVER_THE_JOIN.keys())
    def test_sort_by_a_repeated_name_raises(self, unmatched, derive):
        # Spark reports these as unresolved, the module as ambiguous.
        with pytest.raises(AnalysisException):
            derive(*unmatched).sort(col("id")).collect()
        with pytest.raises(AnalysisException):
            derive(*unmatched).orderBy("id").collect()

    def test_reference_through_the_alias_of_a_frame_with_repeated_names(self, unmatched):
        with pytest.raises(AnalysisException) as error:
            unmatched[2].alias("q").select(col("q.id")).collect()
        assert error_condition(error.value) == "AMBIGUOUS_REFERENCE"
        assert _rows(unmatched[2].alias("q").select(col("q.x"))) == [("a",), (None,)]

    def test_repeated_names_from_to_df(self, spark):
        df = spark.createDataFrame([(1, 2)], ["a", "b"])
        renamed = df.toDF("x", "x")
        for use in [lambda: renamed.select(col("x")), lambda: renamed.filter(col("x") > 0), lambda: renamed.x]:
            with pytest.raises(AnalysisException) as error:
                use()
            assert error_condition(error.value) == "AMBIGUOUS_REFERENCE"
        # Spark has no column of df left after toDF; the module finds two of df.a's name.
        with pytest.raises(AnalysisException):
            df.toDF("a", "a").select(df.a).collect()

    def test_references_to_the_join_inputs_still_resolve(self, unmatched):
        left, right, joined = unmatched
        assert _rows(joined.select(left.id + 1, right.id)) == [(2, None), (None, 2)]
        assert _rows(joined.filter(right.id == 2).select("y")) == [("b",)]
        assert _rows(joined.withColumn("n", left.id * 10).select("n")) == [(10,), (None,)]
        assert _rows(joined.groupBy(left.id).count()) == [(1, 1), (None, 1)]
        assert _rows(joined.filter(col("x") == "a").select(left.id, right.id)) == [(1, None)]

    def test_integer_index_keeps_the_frame(self, unmatched):
        left, right, joined = unmatched
        with pytest.raises((AnalysisException, duckdb.BinderException)):
            left.select(right[0]).collect()
        assert left.select(left[1]).collect() == [Row(x="a")]
        assert _rows(joined.select(left[0], right[1])) == [(1, None), (None, "b")]
        with pytest.raises(AnalysisException) as error:
            joined[0]
        assert error_condition(error.value) == "AMBIGUOUS_REFERENCE"

    def test_join_on_a_name_that_repeats_in_an_input(self, spark, unmatched):
        # The key is the first column of the name, as Spark resolves it.
        other = spark.createDataFrame([(1, "c")], ["id", "z"])
        joined = unmatched[2].limit(10).join(other, "id")
        assert joined.columns == ["id", "x", "id", "y", "z"]
        assert _rows(joined) == [(1, "a", None, None, "c")]
        assert _rows(joined.withColumn("n", lit(0))) == [(1, "a", None, None, "c", 0)]


@pytest.fixture
def abc(spark):
    a = spark.createDataFrame([(1, "a1"), (2, "a2")], ["id", "x"])
    b = spark.createDataFrame([(1, "b1"), (3, "b3")], ["id", "y"])
    c = spark.createDataFrame([(1, "c1"), (2, "c2")], ["k", "z"])
    return a, b, c


class TestColumnsOfChainedJoins:
    def test_join_on_a_name_then_on_an_expression(self, abc):
        a, b, c = abc
        joined = a.join(b, "id", "full").join(c, col("id") == c.k, "left").withColumn("n", lit(0))
        assert joined.columns == ["id", "x", "y", "k", "z", "n"]
        assert _rows(joined) == [
            (1, "a1", "b1", 1, "c1", 0),
            (2, "a2", None, 2, "c2", 0),
            (3, None, "b3", None, None, 0),
        ]

    def test_right_join_on_a_name(self, abc):
        a, b, c = abc
        joined = a.join(b, "id", "right").join(c, b.id == c.k, "left").withColumn("n", lit(0))
        assert _rows(joined) == [(1, "a1", "b1", 1, "c1", 0), (3, None, "b3", None, None, 0)]

    @pytest.mark.parametrize(("how", "rows"), [("semi", [(1, "a1", 1, "c1")]), ("anti", [(2, "a2", 2, "c2")])])
    def test_semi_and_anti_joins_as_inputs(self, abc, how, rows):
        a, b, c = abc
        joined = a.join(b, "id", how).join(c, a.id == c.k)
        assert _rows(joined.withColumn("n", lit(0))) == [(*row, 0) for row in rows]
        assert _rows(joined.toDF("p", "q", "r", "s")) == rows

    def test_join_as_the_right_input(self, abc):
        a, b, c = abc
        joined = c.crossJoin(a.join(b, a.id == b.id, "full")).filter(col("k") == 1).withColumn("n", lit(0))
        assert joined.columns == ["k", "z", "id", "x", "id", "y", "n"]
        assert _rows(joined) == [
            (1, "c1", 1, "a1", 1, "b1", 0),
            (1, "c1", 2, "a2", None, None, 0),
            (1, "c1", None, None, 3, "b3", 0),
        ]
        assert _rows(joined.drop(b.id).select("x", "y")) == [("a1", "b1"), ("a2", None), (None, "b3")]

    @pytest.mark.parametrize(("how", "columns"), [("inner", ["id", "x", "y"]), ("right", ["ID", "x", "y"])])
    def test_join_on_a_name_in_another_case(self, spark, how, columns):
        a = spark.createDataFrame([(1, "a1")], ["id", "x"])
        b = spark.createDataFrame([(1, "b1")], ["ID", "y"])
        joined = a.join(b, "id", how)
        assert joined.columns == columns
        assert joined.select(a.x, b.y, a.x == "a1").collect() == [Row(x="a1", y="b1", **{"(x = a1)": True})]


class TestNamesThroughDerivedFrames:
    def test_star_keeps_repeated_names(self, spark, unmatched):
        left, _, joined = unmatched
        assert joined.orderBy(left.id).select("*").columns == ["id", "x", "id", "y"]
        assert joined.limit(5).select(col("*")).columns == ["id", "x", "id", "y"]
        assert _rows(joined.distinct().select("*")) == [(1, "a", None, None), (None, None, 2, "b")]

    def test_drop_after_a_sort_of_a_filtered_cross_join(self, spark):
        a = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "v"])
        b = spark.createDataFrame([(2, "x"), (3, "y")], ["id", "w"])
        df = a.crossJoin(b).filter(a.id > 1).sort("w").drop("id")
        assert df.columns == ["v", "w"]
        assert df.collect() == [Row(v="b", w="x"), Row(v="b", w="y")]

    def test_drop_duplicates_keeps_repeated_names(self, unmatched):
        derived = unmatched[2].limit(10)
        assert derived.dropDuplicates([]).columns == ["id", "x", "id", "y"]
        assert derived.dropDuplicates().columns == ["id", "x", "id", "y"]

    def test_input_columns_inside_expressions_over_a_join_with_repeated_names(self, spark, unmatched):
        other = spark.createDataFrame([(9, "c")], ["c", "z"])
        joined = unmatched[2].limit(10).crossJoin(other)
        assert _rows(joined.filter(other.c == 9).select("x")) == [("a",), (None,)]
        assert _rows(joined.withColumn("n", other.c + 1).select("n")) == [(10,), (10,)]
        assert joined.groupBy(other.z).count().collect() == [Row(z="c", count=2)]
        assert _rows(joined.orderBy(other.c).select("y")) == [("b",), (None,)]

    def test_plain_input_references_through_a_limit_of_a_join(self, spark):
        a = spark.createDataFrame([(1, 10, "a")], ["x", "u", "v"])
        b = spark.createDataFrame([(1, "p")], ["x", "w"])
        limited = a.join(b, a.x == b.x).orderBy(a.x).limit(1)
        assert limited.withColumn("q", a.v).select("q").collect() == [Row(q="a")]
        assert limited.withColumns({"q": a.u}).select("q").collect() == [Row(q=10)]
        assert limited.select(a.u.alias("z"), b.w).collect() == [Row(z=10, w="p")]

    def test_qualifier_before_a_column_of_that_name(self, spark):
        # The frames have a column k too; Spark resolves k.id through the alias k first.
        c = spark.createDataFrame([(10, 20)], ["k", "id"])
        joined = c.alias("k").join(c.alias("m"), col("k.id") == col("m.id"))
        assert joined.select(col("k.id")).collect() == [Row(id=20)]
        assert joined.select("k.id", "m.k").collect() == [Row(id=20, k=10)]

    def test_names_that_differ_only_in_case_mapping(self, spark):
        # Spark compares names ignoring case letter by letter, so these two names differ.
        df = spark.createDataFrame([(1, 2)], ["straße", "STRASSE"])
        assert df.select("straße").collect() == [Row(straße=1)]
        assert df.select(df["STRASSE"], col("straße")).collect() == [Row(STRASSE=2, straße=1)]


class TestQualifiedStar:
    def test_star_of_a_struct_column(self, spark):
        df = spark.createDataFrame([(1, "a")], ["id", "v"]).select(sf.struct("id", "v").alias("x"))
        assert df.select("x.*").collect() == [Row(id=1, v="a")]

    def test_star_of_a_frame_alias(self, spark):
        a = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "v"])
        b = spark.createDataFrame([(2, "x")], ["id", "w"])
        joined = a.alias("a").join(b.alias("b"), col("a.id") == col("b.id"))
        assert joined.select("a.*").collect() == [Row(id=2, v="b")]
        assert joined.select("b.*", "v").collect() == [Row(id=2, w="x", v="b")]
        assert a.alias("q").select("q.*").columns == ["id", "v"]

    def test_star_of_a_frame_alias_with_repeated_names(self, spark, unmatched):
        other = spark.createDataFrame([(9,)], ["c"])
        df = unmatched[2].alias("q").crossJoin(other).select("q.*")
        assert df.columns == ["id", "x", "id", "y"]
        assert _rows(df.withColumn("n", lit(1))) == [(1, "a", None, None, 1), (None, None, 2, "b", 1)]

    def test_to_arrow_keeps_repeated_names(self, spark, unmatched):
        if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3."):
            pytest.skip("toArrow is new in Spark 4.0")
        pytest.importorskip("pyarrow")
        other = spark.createDataFrame([(9,)], ["c"])
        assert unmatched[2].limit(10).crossJoin(other).toArrow().column_names == ["id", "x", "id", "y", "c"]

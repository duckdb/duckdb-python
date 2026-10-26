import pytest
from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException, error_condition
from spark_namespace.sql.types import (
    Row,
)

_ = pytest.importorskip("duckdb.experimental.spark")

if USE_ACTUAL_SPARK:
    import pyspark

    SPARK_3 = pyspark.__version__.startswith("3.")
else:
    SPARK_3 = False


class TestDataFrameDropDuplicates:
    @pytest.mark.parametrize("method", ["dropDuplicates", "drop_duplicates"])
    def test_spark_drop_duplicates(self, method, spark):
        # Prepare Data
        data = [
            ("James", "Sales", 3000),
            ("Michael", "Sales", 4600),
            ("Robert", "Sales", 4100),
            ("Maria", "Finance", 3000),
            ("James", "Sales", 3000),
            ("Scott", "Finance", 3300),
            ("Jen", "Finance", 3900),
            ("Jeff", "Marketing", 3000),
            ("Kumar", "Marketing", 2000),
            ("Saif", "Sales", 4100),
        ]

        # Create DataFrame
        columns = ["employee_name", "department", "salary"]
        df = spark.createDataFrame(data=data, schema=columns)

        distinctDF = df.distinct().sort("employee_name")
        assert distinctDF.count() == 9
        res = distinctDF.collect()
        # James | Sales had a duplicate, has been removed
        expected = [
            Row(employee_name="James", department="Sales", salary=3000),
            Row(employee_name="Jeff", department="Marketing", salary=3000),
            Row(employee_name="Jen", department="Finance", salary=3900),
            Row(employee_name="Kumar", department="Marketing", salary=2000),
            Row(employee_name="Maria", department="Finance", salary=3000),
            Row(employee_name="Michael", department="Sales", salary=4600),
            Row(employee_name="Robert", department="Sales", salary=4100),
            Row(employee_name="Saif", department="Sales", salary=4100),
            Row(employee_name="Scott", department="Finance", salary=3300),
        ]
        assert res == expected

        df2 = getattr(df, method)().sort("employee_name")
        assert df2.count() == 9
        res2 = df2.collect()
        assert res2 == res

        expected_subset = [
            Row(department="Finance", salary=3000),
            Row(department="Finance", salary=3300),
            Row(department="Finance", salary=3900),
            Row(department="Marketing", salary=2000),
            Row(department="Marketing", salary=3000),
            Row(epartment="Sales", salary=3000),
            Row(department="Sales", salary=4100),
            Row(department="Sales", salary=4600),
        ]

        dropDisDF = getattr(df, method)(["department", "salary"]).sort("department", "salary")
        assert dropDisDF.columns == ["employee_name", "department", "salary"]
        assert dropDisDF.count() == len(expected_subset)
        res = dropDisDF.select("department", "salary").collect()
        assert res == expected_subset

    def test_spark_drop_duplicates_with_keywords_cols(self, spark):
        data = [
            ("abc", 1, ""),
            ("abc", 1, ""),
            ("def", 2, ""),
            ("def", 2, ""),
            ("def", 2, ""),
            ("def", 3, ""),
            ("def", 3, ""),
            ("def", 3, ""),
            ("def", 3, ""),
        ]

        columns = ["table", "min", "null"]
        df = spark.createDataFrame(data=data, schema=columns)

        dropDisDF = df.dropDuplicates(["table", "min"]).sort("table", "min")
        assert dropDisDF.count() == 3

    @pytest.mark.parametrize("subset", [[], ()])
    def test_drop_duplicates_with_an_empty_subset(self, spark, subset):
        # Spark compares rows on no columns at all, so they are all duplicates of the first.
        df = spark.createDataFrame([(1,), (2,), (2,)], ["x"])
        assert df.dropDuplicates(subset).count() == 1
        assert df.dropDuplicates().count() == 2
        assert df.dropDuplicates(None).count() == 2
        assert df.filter(df.x > 5).dropDuplicates(subset).count() == 0

    def test_drop_duplicates_on_a_repeated_name(self, spark):
        # Spark compares every column of the name, so rows that differ in the second "id" both stay.
        df = spark.createDataFrame([(1, 1), (1, 2), (1, 2), (2, 2)], ["id", "x"]).toDF("id", "id")
        assert sorted(tuple(row) for row in df.dropDuplicates(["id"]).collect()) == [(1, 1), (1, 2), (2, 2)]
        assert sorted(tuple(row) for row in df.dropDuplicates(["ID"]).collect()) == [(1, 1), (1, 2), (2, 2)]

    def test_drop_duplicates_after_a_join(self, spark):
        # The window must compare the join's own columns, not the inputs' columns a positional key would reach.
        a = spark.createDataFrame([(1,)], ["id"])
        b = spark.createDataFrame([(1, 10), (1, 20)], ["id", "v"])
        joined = a.join(b, "id")
        assert [row.v for row in joined.dropDuplicates(["v"]).sort("v").collect()] == [10, 20]
        assert joined.filter(joined.v > 0).dropDuplicates(["v"]).count() == 2
        assert joined.dropDuplicates(["id"]).count() == 1

    def test_drop_duplicates_on_an_unknown_name(self, spark):
        df = spark.createDataFrame([(1, 1)], ["id", "x"])
        with pytest.raises(AnalysisException) as error:
            df.dropDuplicates(["nope"]).collect()
        # Spark 3.5 raises the same error under a legacy class.
        expected = "_LEGACY_ERROR_TEMP_1201" if SPARK_3 else "UNRESOLVED_COLUMN_AMONG_FIELD_NAMES"
        assert error_condition(error.value) == expected

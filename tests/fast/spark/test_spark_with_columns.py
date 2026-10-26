import pytest

_ = pytest.importorskip("duckdb.experimental.spark")


from spark_namespace.errors import AnalysisException, error_condition
from spark_namespace.sql.functions import col, lit
from spark_namespace.sql.types import Row


class TestWithColumns:
    def test_with_columns(self, spark):
        data = [
            ("James", "", "Smith", "1991-04-01", "M", 3000),
            ("Michael", "Rose", "", "2000-05-19", "M", 4000),
            ("Robert", "", "Williams", "1978-09-05", "M", 4000),
            ("Maria", "Anne", "Jones", "1967-12-01", "F", 4000),
            ("Jen", "Mary", "Brown", "1980-02-17", "F", -1),
        ]

        columns = ["firstname", "middlename", "lastname", "dob", "gender", "salary"]
        df = spark.createDataFrame(data=data, schema=columns)
        assert df.schema["salary"].dataType.typeName() == "long"

        # The type of 'salary' has been cast to Bigint
        new_df = df.withColumns({"salary": col("salary").cast("BIGINT")})
        assert new_df.schema["salary"].dataType.typeName() == "long"

        # Replace the 'salary' column with '(salary * 100)' and add a new column
        # from an existing column
        df2 = df.withColumns({"salary": col("salary") * 100, "CopiedColumn": col("salary") * -1})
        res = df2.collect()
        assert res[0].salary == 300_000
        assert res[0].CopiedColumn == -3000

        df2 = df.withColumns({"Country": lit("USA")})
        res = df2.collect()
        assert res[0].Country == "USA"

        df2 = df.withColumns({"Country": lit("USA")}).withColumns({"anotherColumn": lit("anotherValue")})
        res = df2.collect()
        assert res[0].Country == "USA"
        assert res[1].anotherColumn == "anotherValue"

        df2 = df.drop("salary")
        assert "salary" not in df2.columns

    def test_replace_several_existing_columns(self, spark):
        df = spark.createDataFrame([(1, 2)], ["a", "b"])
        res = df.withColumns({"a": lit(0), "b": lit(5)})
        assert res.columns == ["a", "b"]
        assert res.collect() == [(0, 5)]
        res = df.withColumns({"b": lit(5), "c": lit(7), "a": lit(0)})
        assert res.columns == ["a", "b", "c"]
        assert res.collect() == [(0, 5, 7)]

    def test_replace_column_named_in_another_case(self, spark):
        df = spark.createDataFrame([(1, 2)], ["A", "b"])
        res = df.withColumns({"a": lit(0)})
        assert res.columns == ["a", "b"]
        assert res.collect() == [(0, 2)]

    @pytest.mark.parametrize("names", [("x", "X"), ("B", "b")])
    def test_names_differing_only_in_case_raise(self, spark, names):
        df = spark.createDataFrame([("a", 1)], "a string, b int")
        with pytest.raises(AnalysisException) as error:
            df.withColumns({names[0]: lit(1), names[1]: lit(2)})
        assert error_condition(error.value) == "COLUMN_ALREADY_EXISTS"
        assert dict(error.value.getMessageParameters()) == {"columnName": f"`{names[1].lower()}`"}
        assert error.value.getSqlState() == "42711"

    def test_name_in_another_case_replaces_the_column(self, spark):
        df = spark.createDataFrame([("a", 1)], "a string, b int")
        assert df.withColumns({"A": lit(1)}).collect() == [Row(A=1, b=1)]

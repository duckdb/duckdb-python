import datetime
import platform
from decimal import Decimal

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException, error_condition
from spark_namespace.sql.functions import col, lit
from spark_namespace.sql.types import Row

if USE_ACTUAL_SPARK:
    import pyspark


@pytest.fixture
def df(spark):
    simpleData = [
        ("James", "Sales", "NY", 90000, 34, 10000),
        ("Michael", "Sales", "NY", 86000, 56, 20000),
        ("Robert", "Sales", "CA", 81000, 30, 23000),
        ("Maria", "Finance", "CA", 90000, 24, 23000),
    ]

    columns = ["employee_name", "department", "state", "salary", "age", "bonus"]
    dataframe = spark.createDataFrame(data=simpleData, schema=columns)
    return dataframe


@pytest.fixture
def df2(spark):
    simpleData2 = [
        ("James", "Sales", "NY", 90000, 34, 10000),
        ("Maria", "Finance", "CA", 90000, 24, 23000),
        ("Jen", "Finance", "NY", 79000, 53, 15000),
        ("Jeff", "Marketing", "CA", 80000, 25, 18000),
        ("Kumar", "Marketing", "NY", 91000, 50, 21000),
    ]
    columns2 = ["employee_name", "department", "state", "salary", "age", "bonus"]
    dataframe = spark.createDataFrame(data=simpleData2, schema=columns2)
    return dataframe


class TestDataFrameUnion:
    def test_merge_with_union(self, df, df2):
        unionDF = df.union(df2)
        res = unionDF.collect()
        assert res == [
            Row(employee_name="James", department="Sales", state="NY", salary=90000, age=34, bonus=10000),
            Row(employee_name="Michael", department="Sales", state="NY", salary=86000, age=56, bonus=20000),
            Row(employee_name="Robert", department="Sales", state="CA", salary=81000, age=30, bonus=23000),
            Row(employee_name="Maria", department="Finance", state="CA", salary=90000, age=24, bonus=23000),
            Row(employee_name="James", department="Sales", state="NY", salary=90000, age=34, bonus=10000),
            Row(employee_name="Maria", department="Finance", state="CA", salary=90000, age=24, bonus=23000),
            Row(employee_name="Jen", department="Finance", state="NY", salary=79000, age=53, bonus=15000),
            Row(employee_name="Jeff", department="Marketing", state="CA", salary=80000, age=25, bonus=18000),
            Row(employee_name="Kumar", department="Marketing", state="NY", salary=91000, age=50, bonus=21000),
        ]
        unionDF = df.unionAll(df2)
        res2 = unionDF.collect()
        assert res == res2

    @pytest.mark.xfail(condition=platform.system() == "Emscripten", reason="Broken on Pyodide")
    def test_merge_without_duplicates(self, df, df2):
        # 'sort' has been added to make the result deterministic
        disDF = df.union(df2).distinct().sort(col("employee_name"))
        res = disDF.collect()
        assert res == [
            Row(employee_name="James", department="Sales", state="NY", salary=90000, age=34, bonus=10000),
            Row(employee_name="Jeff", department="Marketing", state="CA", salary=80000, age=25, bonus=18000),
            Row(employee_name="Jen", department="Finance", state="NY", salary=79000, age=53, bonus=15000),
            Row(employee_name="Kumar", department="Marketing", state="NY", salary=91000, age=50, bonus=21000),
            Row(employee_name="Maria", department="Finance", state="CA", salary=90000, age=24, bonus=23000),
            Row(employee_name="Michael", department="Sales", state="NY", salary=86000, age=56, bonus=20000),
            Row(employee_name="Robert", department="Sales", state="CA", salary=81000, age=30, bonus=23000),
        ]

    def test_union_of_an_integer_and_a_string_column(self, spark):
        numbers = spark.createDataFrame([(1,)], ["v"])
        strings = spark.createDataFrame([("x",)], ["v"])
        if USE_ACTUAL_SPARK and not pyspark.__version__.startswith("3."):
            # Spark 4's ANSI mode casts the string to the integer type.
            with pytest.raises(Exception) as error:  # noqa: PT011
                numbers.union(strings).collect()
            assert error_condition(error.value) == "CAST_INVALID_INPUT"
        else:
            # Spark 3.5 and DuckDB widen both to a string.
            union = numbers.union(strings)
            assert union.dtypes == [("v", "string")]
            assert sorted(union.collect()) == [Row(v="1"), Row(v="x")]


@pytest.fixture
def ab(spark):
    return spark.createDataFrame([((1, 2),)], "s struct<a:int,b:int>")


@pytest.fixture
def ba(spark):
    return spark.createDataFrame([((3, 4),)], "s struct<b:int,a:int>")


class TestUnionByPosition:
    def test_struct_fields_by_position(self, ab, ba):
        union = ab.union(ba)
        assert union.schema.simpleString() == "struct<s:struct<a:int,b:int>>"
        assert union.collect() == [Row(s=Row(a=1, b=2)), Row(s=Row(a=3, b=4))]
        union = ba.union(ab)
        assert union.schema.simpleString() == "struct<s:struct<b:int,a:int>>"
        assert union.collect() == [Row(s=Row(b=3, a=4)), Row(s=Row(b=1, a=2))]

    def test_struct_fields_of_other_names(self, spark, ab):
        cd = spark.createDataFrame([((3, 4),)], "s struct<c:int,d:int>")
        assert ab.union(cd).collect() == [Row(s=Row(a=1, b=2)), Row(s=Row(a=3, b=4))]

    @pytest.mark.parametrize(
        ("first", "second", "values", "expected"),
        [
            ("array<struct<a:int,b:int>>", "array<struct<b:int,a:int>>", [(3, 4)], [Row(a=3, b=4)]),
            ("map<string,struct<a:int,b:int>>", "map<string,struct<b:int,a:int>>", {"k": (3, 4)}, {"k": Row(a=3, b=4)}),
            ("struct<t:struct<a:int,b:int>>", "struct<t:struct<b:int,a:int>>", ((3, 4),), Row(t=Row(a=3, b=4))),
        ],
    )
    def test_nested_struct_fields_by_position(self, spark, first, second, values, expected):
        left = spark.createDataFrame([], f"s {first}")
        right = spark.createDataFrame([(values,)], f"s {second}")
        assert left.union(right).collect() == [Row(s=expected)]

    def test_union_by_name_matches_struct_fields_by_name(self, ab, ba):
        assert ab.unionByName(ba).collect() == [Row(s=Row(a=1, b=2)), Row(s=Row(a=4, b=3))]

    @pytest.mark.parametrize(
        "second",
        ["struct<b:string,a:int>", "struct<a:int>", "struct<b:bigint,a:int>", "array<struct<b:string,a:int>>"],
    )
    def test_struct_fields_that_do_not_match_by_position(self, spark, second):
        first = "array<struct<a:int,b:string>>" if second.startswith("array") else "struct<a:int,b:string>"
        left = spark.createDataFrame([], f"s {first}")
        right = spark.createDataFrame([], f"s {second}")
        with pytest.raises(AnalysisException) as error:
            left.union(right).collect()
        assert error_condition(error.value) == "INCOMPATIBLE_COLUMN_TYPE"

    def test_struct_fields_of_the_same_names_widen(self, spark):
        narrow = spark.createDataFrame([((1, "x"),)], "s struct<a:int,b:string>")
        wide = spark.createDataFrame([((2, "y"),)], "s struct<a:bigint,b:string>")
        union = narrow.union(wide)
        assert union.schema.simpleString() == "struct<s:struct<a:bigint,b:string>>"
        assert union.collect() == [Row(s=Row(a=1, b="x")), Row(s=Row(a=2, b="y"))]

    def test_number_of_columns(self, spark):
        one = spark.createDataFrame([(1,)], ["a"])
        two = spark.createDataFrame([(1, 2)], ["a", "b"])
        for left, right in [(one, two), (two, one)]:
            with pytest.raises(AnalysisException) as error:
                left.union(right).collect()
            assert error_condition(error.value) == "NUM_COLUMNS_MISMATCH"

    @pytest.mark.parametrize("operation", ["union", "intersect", "exceptAll"])
    def test_struct_field_names_in_another_case(self, spark, operation):
        lower = spark.createDataFrame([((1,),)], "s struct<a:int>")
        upper = spark.createDataFrame([((1,),)], "s struct<A:int>")
        assert getattr(lower, operation)(upper).count() == {"union": 2, "intersect": 1, "exceptAll": 0}[operation]

    @pytest.mark.parametrize("operation", ["intersect", "intersectAll", "exceptAll"])
    def test_other_set_operations_reject_other_struct_field_names(self, spark, ab, ba, operation):
        with pytest.raises(AnalysisException) as error:
            getattr(ab, operation)(ba).collect()
        assert error_condition(error.value) == "INCOMPATIBLE_COLUMN_TYPE"
        same = spark.createDataFrame([((1, 2),)], "s struct<a:int,b:int>")
        assert getattr(ab, operation)(same).count() == (0 if operation == "exceptAll" else 1)


_VALUES = {
    "boolean": True,
    "int": 1,
    "tinyint": 1,
    "bigint": 1,
    "double": 1.5,
    "decimal(10,2)": Decimal("1.50"),
    "string": "1",
    "binary": bytearray(b"1"),
    "date": datetime.date(2024, 1, 2),
    "timestamp": datetime.datetime(2024, 1, 2, 3, 4, 5),
    "timestamp_ntz": datetime.datetime(2024, 1, 2, 3, 4, 5),
    "interval day to second": datetime.timedelta(days=1),
    "array<int>": [1],
    "array<boolean>": [True],
    "array<string>": ["1"],
    "map<string,int>": {"a": 1},
    "map<string,boolean>": {"a": True},
    "struct<a:int>": (1,),
    "struct<a:boolean>": (True,),
}


def _column(spark, type_string):
    return spark.createDataFrame([(_VALUES[type_string],)], f"v {type_string}")


class TestSetOperationTypes:
    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("boolean", "int"),
            ("boolean", "double"),
            ("boolean", "decimal(10,2)"),
            ("boolean", "binary"),
            ("boolean", "date"),
            ("boolean", "timestamp"),
            ("date", "int"),
            ("timestamp", "bigint"),
            ("binary", "int"),
            ("interval day to second", "int"),
            ("array<int>", "int"),
            ("array<int>", "array<boolean>"),
            ("map<string,int>", "map<string,boolean>"),
            ("struct<a:int>", "struct<a:boolean>"),
            ("string", "array<string>"),
            ("string", "struct<a:int>"),
            ("string", "map<string,int>"),
        ],
    )
    @pytest.mark.parametrize("operation", ["union", "unionByName", "intersect", "intersectAll", "exceptAll"])
    def test_types_of_other_families_raise(self, spark, first, second, operation):
        for left, right in [(first, second), (second, first)]:
            with pytest.raises(AnalysisException) as error:
                getattr(_column(spark, left), operation)(_column(spark, right)).collect()
            assert error_condition(error.value) == "INCOMPATIBLE_COLUMN_TYPE"

    def test_type_of_another_family_in_a_later_column(self, spark):
        left = spark.createDataFrame([(1, True)], "a int, v boolean")
        right = spark.createDataFrame([(1, 2)], "a int, v int")
        with pytest.raises(AnalysisException, match="second column") as error:
            left.union(right).collect()
        assert error_condition(error.value) == "INCOMPATIBLE_COLUMN_TYPE"

    def test_boolean_and_string(self, spark):
        for left, right in [("boolean", "string"), ("string", "boolean")]:
            if USE_ACTUAL_SPARK and not pyspark.__version__.startswith("3."):
                # Spark 4's ANSI mode casts the string to a boolean.
                assert _column(spark, left).union(_column(spark, right)).dtypes == [("v", "boolean")]
            else:
                with pytest.raises(AnalysisException) as error:
                    _column(spark, left).union(_column(spark, right)).collect()
                assert error_condition(error.value) == "INCOMPATIBLE_COLUMN_TYPE"

    @pytest.mark.parametrize(
        ("first", "second", "widened"),
        [
            ("int", "double", "double"),
            ("int", "decimal(10,2)", "decimal(12,2)"),
            ("tinyint", "bigint", "bigint"),
            ("date", "timestamp", "timestamp"),
            ("date", "timestamp_ntz", "timestamp_ntz"),
            ("timestamp", "timestamp_ntz", "timestamp"),
        ],
    )
    def test_types_of_one_family_widen(self, spark, first, second, widened):
        for left, right in [(first, second), (second, first)]:
            union = _column(spark, left).union(_column(spark, right))
            assert union.dtypes == [("v", widened)]
            assert len(union.collect()) == 2

    @pytest.mark.parametrize(
        ("type_string", "value"),
        [
            ("map<string,int>", {"k": 1}),
            ("array<map<string,int>>", [{"k": 1}]),
            ("struct<m:map<string,int>>", ({"k": 1},)),
        ],
    )
    def test_maps(self, spark, type_string, value):
        df = spark.createDataFrame([(1, value)], f"i int, v {type_string}")
        nulls = spark.range(1).select(lit(None).alias("i"), lit(None).alias("v"))
        assert df.union(df).count() == 2
        assert df.unionByName(df).count() == 2
        assert df.dropDuplicates(["i"]).count() == 1
        compare = [
            *(getattr(df, operation) for operation in ("intersect", "intersectAll", "exceptAll")),
            nulls.intersect,
            nulls.exceptAll,
            lambda other: other.distinct(),
            lambda other: other.dropDuplicates(),
            lambda other: other.dropDuplicates(["v"]),
            lambda other: other.union(other).distinct(),
        ]
        for operation in compare:
            with pytest.raises(AnalysisException) as error:
                operation(df).collect()
            assert error_condition(error.value) == "UNSUPPORTED_FEATURE.SET_OPERATION_ON_MAP_TYPE"
            assert dict(error.value.getMessageParameters())["colName"] == "`v`"

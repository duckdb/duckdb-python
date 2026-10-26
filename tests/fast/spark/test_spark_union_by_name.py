import pytest

_ = pytest.importorskip("duckdb.experimental.spark")


from spark_namespace.errors import AnalysisException, error_condition
from spark_namespace.sql.types import (
    Row,
)

import duckdb


@pytest.fixture
def df1(spark):
    data = [("James", 34), ("Michael", 56), ("Robert", 30), ("Maria", 24)]
    dataframe = spark.createDataFrame(data=data, schema=["name", "id"])
    return dataframe


@pytest.fixture
def df2(spark):
    data2 = [(34, "James"), (45, "Maria"), (45, "Jen"), (34, "Jeff")]
    dataframe = spark.createDataFrame(data=data2, schema=["id", "name"])
    return dataframe


class TestDataFrameUnion:
    def test_union_by_name(self, df1, df2):
        rel = df1.unionByName(df2)
        res = rel.collect()
        expected = [
            Row(name="James", id=34),
            Row(name="Michael", id=56),
            Row(name="Robert", id=30),
            Row(name="Maria", id=24),
            Row(name="James", id=34),
            Row(name="Maria", id=45),
            Row(name="Jen", id=45),
            Row(name="Jeff", id=34),
        ]
        assert res == expected

    def test_union_by_name_allow_missing_cols(self, df1, df2):
        rel = df1.unionByName(df2.drop("id"), allowMissingColumns=True)
        res = rel.collect()
        expected = [
            Row(name="James", id=34),
            Row(name="Michael", id=56),
            Row(name="Robert", id=30),
            Row(name="Maria", id=24),
            Row(name="James", id=None),
            Row(name="Maria", id=None),
            Row(name="Jen", id=None),
            Row(name="Jeff", id=None),
        ]
        assert res == expected

    def test_union_by_name_allow_missing_cols_rev(self, df1, df2):
        rel = df2.drop("id").unionByName(df1, allowMissingColumns=True)
        res = rel.collect()
        expected = [
            Row(name="James", id=None),
            Row(name="Maria", id=None),
            Row(name="Jen", id=None),
            Row(name="Jeff", id=None),
            Row(name="James", id=34),
            Row(name="Michael", id=56),
            Row(name="Robert", id=30),
            Row(name="Maria", id=24),
        ]
        assert res == expected

    def test_union_by_name_allow_missing_cols_ignores_case(self, spark):
        left = spark.createDataFrame([(1, 2)], ["a", "B"])
        rel = left.unionByName(spark.createDataFrame([(3, 4)], ["b", "c"]), allowMissingColumns=True)
        assert rel.columns == ["a", "B", "c"]
        assert rel.collect() == [Row(a=1, B=2, c=None), Row(a=None, B=3, c=4)]
        rel = left.unionByName(spark.createDataFrame([(5, 6)], ["A", "z"]), allowMissingColumns=True)
        assert rel.columns == ["a", "B", "z"]
        assert rel.collect() == [Row(a=1, B=2, z=None), Row(a=5, B=None, z=6)]

    @pytest.mark.parametrize(("left", "right"), [(["a"], ["a", "b"]), (["a"], ["b", "a"])])
    def test_union_by_name_with_extra_columns_in_the_other(self, spark, left, right):
        first = spark.createDataFrame([tuple(range(len(left)))], left)
        second = spark.createDataFrame([tuple(range(len(right)))], right)
        with pytest.raises(AnalysisException) as error:
            first.unionByName(second).collect()
        assert error_condition(error.value) == "NUM_COLUMNS_MISMATCH"

    def test_union_by_name_with_extra_columns_in_the_first(self, spark):
        first = spark.createDataFrame([(1, 2)], ["a", "b"])
        second = spark.createDataFrame([(3,)], ["a"])
        # Spark raises AnalysisException; DuckDB's binder does not find the column.
        with pytest.raises((AnalysisException, duckdb.BinderException)):
            first.unionByName(second).collect()

    def test_union_by_name_of_names_with_a_backtick_or_a_dot(self, spark):
        df = spark.createDataFrame([(1, 2, 3)], "k int, `a``b` int, `c.d` int")
        other = spark.createDataFrame([(3, 4, 5)], "`C.D` int, `A``B` int, K int")
        rel = df.unionByName(other)
        assert rel.columns == ["k", "a`b", "c.d"]
        assert [tuple(r) for r in rel.collect()] == [(1, 2, 3), (5, 4, 3)]
        rel = df.unionByName(df.select("k"), allowMissingColumns=True)
        assert [tuple(r) for r in rel.collect()] == [(1, 2, 3), (1, None, None)]


class TestUnionByNameOfStructFields:
    @pytest.mark.parametrize(
        ("first", "second", "values", "field", "fields"),
        [
            ("struct<a:int>", "struct<b:int>", ((1,), (2,)), "a", "b"),
            ("struct<a:int,b:int>", "struct<a:int>", ((1, 2), (3,)), "b", "a"),
            ("struct<t:struct<a:int>>", "struct<t:struct<b:int>>", (((1,),), ((2,),)), "a", "b"),
            ("array<struct<a:int>>", "array<struct<b:int>>", ([(1,)], [(2,)]), "a", "b"),
        ],
    )
    def test_missing_struct_field_raises(self, spark, first, second, values, field, fields):
        left = spark.createDataFrame([(values[0],)], f"s {first}")
        right = spark.createDataFrame([(values[1],)], f"s {second}")
        with pytest.raises(AnalysisException) as error:
            left.unionByName(right).collect()
        assert error_condition(error.value) == "FIELD_NOT_FOUND"
        assert dict(error.value.getMessageParameters()) == {"fieldName": f"`{field}`", "fields": f"`{fields}`"}

    def test_extra_struct_field_in_the_other_raises(self, spark):
        left = spark.createDataFrame([((1,),)], "s struct<a:int>")
        right = spark.createDataFrame([((1, 2),)], "s struct<a:int,b:int>")
        with pytest.raises(AnalysisException) as error:
            left.unionByName(right).collect()
        assert error_condition(error.value) == "INCOMPATIBLE_COLUMN_TYPE"

    def test_missing_struct_fields_are_filled_where_missing_columns_are_allowed(self, spark):
        a = spark.createDataFrame([((1,),)], "s struct<a:int>")
        b = spark.createDataFrame([((2,),)], "s struct<b:int>")
        ab = spark.createDataFrame([((3, 4),)], "s struct<a:int,b:int>")
        union = a.unionByName(b, allowMissingColumns=True)
        assert union.schema.simpleString() == "struct<s:struct<a:int,b:int>>"
        assert union.collect() == [Row(s=Row(a=1, b=None)), Row(s=Row(a=None, b=2))]
        assert ab.unionByName(a, allowMissingColumns=True).collect() == [Row(s=Row(a=3, b=4)), Row(s=Row(a=1, b=None))]

    @pytest.mark.parametrize("allow_missing", [False, True])
    @pytest.mark.parametrize(
        ("first", "second", "values", "expected"),
        [
            (
                "map<string,struct<a:int,b:int>>",
                "map<string,struct<b:int,a:int>>",
                ({"k": (1, 2)}, {"k": (3, 4)}),
                [{"k": Row(a=1, b=2)}, {"k": Row(a=3, b=4)}],
            ),
            (
                "map<string,struct<a:int>>",
                "map<string,struct<b:int>>",
                ({"k": (1,)}, {"k": (2,)}),
                [{"k": Row(a=1)}, {"k": Row(a=2)}],
            ),
            (
                "map<struct<a:int,b:int>,int>",
                "map<struct<b:int,a:int>,int>",
                ({(1, 2): 1}, {(3, 4): 2}),
                [{Row(a=1, b=2): 1}, {Row(a=3, b=4): 2}],
            ),
            (
                "array<map<string,struct<a:int,b:int>>>",
                "array<map<string,struct<b:int,a:int>>>",
                ([{"k": (1, 2)}], [{"k": (3, 4)}]),
                [[{"k": Row(a=1, b=2)}], [{"k": Row(a=3, b=4)}]],
            ),
            (
                "map<string,array<struct<a:int,b:int>>>",
                "map<string,array<struct<b:int,a:int>>>",
                ({"k": [(1, 2)]}, {"k": [(3, 4)]}),
                [{"k": [Row(a=1, b=2)]}, {"k": [Row(a=3, b=4)]}],
            ),
        ],
    )
    def test_struct_fields_inside_a_map_pair_by_position(self, spark, first, second, values, expected, allow_missing):
        left = spark.createDataFrame([(values[0],)], f"v {first}")
        right = spark.createDataFrame([(values[1],)], f"v {second}")
        union = left.unionByName(right, allowMissingColumns=allow_missing)
        assert union.schema.simpleString() == f"struct<v:{first}>"
        assert union.collect() == [Row(v=value) for value in expected]

    @pytest.mark.parametrize("allow_missing", [False, True])
    @pytest.mark.parametrize(
        ("first", "second", "values"),
        [
            ("map<string,struct<a:int>>", "map<string,struct<b:bigint>>", ({"k": (1,)}, {"k": (2,)})),
            ("map<string,struct<a:int,b:int>>", "map<string,struct<a:int>>", ({"k": (1, 2)}, {"k": (3,)})),
        ],
    )
    def test_struct_fields_inside_a_map_of_another_shape(self, spark, first, second, values, allow_missing):
        left = spark.createDataFrame([(values[0],)], f"v {first}")
        right = spark.createDataFrame([(values[1],)], f"v {second}")
        with pytest.raises(AnalysisException) as error:
            left.unionByName(right, allowMissingColumns=allow_missing).collect()
        assert error_condition(error.value) == "INCOMPATIBLE_COLUMN_TYPE"

    def test_struct_fields_inside_an_array_pair_by_name(self, spark):
        left = spark.createDataFrame([([(1, 2)],)], "v array<struct<a:int,b:int>>")
        right = spark.createDataFrame([([(3, 4)],)], "v array<struct<b:int,a:int>>")
        assert left.unionByName(right).collect() == [Row(v=[Row(a=1, b=2)]), Row(v=[Row(a=4, b=3)])]

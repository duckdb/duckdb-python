import datetime
from decimal import Decimal

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import ParseException, PySparkTypeError
from spark_namespace.sql import functions as F
from spark_namespace.sql.types import IntegerType, Row, StringType

import duckdb
import numpy as np

if USE_ACTUAL_SPARK:
    import pyspark

    SPARK_3 = pyspark.__version__.startswith("3.")
else:
    SPARK_3 = False


class TestSparkUDF:
    def test_udf_register(self, spark):
        def to_upper_fn(s: str) -> str:
            return s.upper()

        spark.udf.register("to_upper_fn", to_upper_fn)
        assert spark.sql("select to_upper_fn('quack') as vl").collect()[0].vl == "QUACK"

    def test_udf_register_with_return_type(self, spark):
        spark.udf.register("int_x2", lambda i: i * 2, IntegerType())
        spark.udf.register("str_x2", lambda s: s * 2, StringType())
        row = spark.sql("select int_x2(21) as i, str_x2('ab') as s").collect()[0]
        assert (row.i, row.s) == (42, "abab")

    def test_udf_register_with_return_type_string(self, spark):
        spark.udf.register("int_plus", lambda i: i + 1, "int")
        spark.udf.register("arr_of", lambda i: [i, i], "array<bigint>")
        res = spark.sql("select int_plus(41) as i, arr_of(2) as a")
        assert res.schema.simpleString() == "struct<i:int,a:array<bigint>>"
        assert res.collect()[0].asDict() == {"i": 42, "a": [2, 2]}

    @pytest.mark.parametrize(
        ("type_string", "value", "spark_type", "expected"),
        [
            ("VARCHAR", "x", "string", "x"),
            ("json", '{"a":1}', "string", '{"a":1}'),
            ("STRUCT(a INT)", {"a": 1}, "struct<a:int>", Row(a=1)),
        ],
    )
    def test_udf_return_type_in_duckdb_syntax(self, spark, type_string, value, spark_type, expected):
        if USE_ACTUAL_SPARK:
            with pytest.raises(ParseException):
                spark.udf.register("duckdb_typed", lambda _: value, type_string)
            return
        # DuckDB's type syntax is read where Spark's grammar rejects the text, as in Column.cast.
        registered = spark.udf.register("duckdb_typed", lambda _: value, type_string)
        res = spark.range(1).select(registered("id").alias("v"))
        assert registered.returnType.simpleString() == spark_type
        assert res.schema.simpleString() == f"struct<v:{spark_type}>"
        assert res.collect() == [Row(v=expected)]

    @pytest.mark.parametrize("type_string", ["VARCHAR(10)[]", "CHAR(3)[]", "INTERVAL[]"])
    def test_udf_return_type_starting_with_a_type_the_module_lacks(self, spark, type_string):
        if USE_ACTUAL_SPARK:
            with pytest.raises(ParseException):
                spark.udf.register("lacking_typed", lambda _: None, type_string)
            return
        from duckdb.experimental.spark.exception import ContributionsAcceptedError

        # Spark's grammar reads the first type before the text turns out to be DuckDB's, and the module lacks it.
        with pytest.raises(ContributionsAcceptedError, match="not supported"):
            spark.udf.register("lacking_typed", lambda _: None, type_string)

    def test_udf_in_sql_ignores_annotations(self, spark):
        def describe(x: int, *, unused: int = 0) -> str:
            return repr(x)

        spark.udf.register("describe_annotated", describe, "string")
        # Spark passes the argument as it is and takes only the positional parameters.
        row = spark.sql("select describe_annotated(1.5) as d, describe_annotated('3') as s").collect()[0]
        assert (row.d, row.s) == ("Decimal('1.5')", "'3'")

    def test_udf_without_a_signature_in_sql(self, spark):
        spark.udf.register("largest", max, "int")
        assert spark.sql("select largest(1, 3, 2) as v").collect() == [Row(v=3)]

    def test_udf_replaces_a_temporary_macro_in_sql(self, spark):
        if not USE_ACTUAL_SPARK:
            spark.sql("CREATE TEMP MACRO udf_over_macro(a) AS a")
        spark.udf.register("udf_over_macro", lambda v: v + 1, "int")
        assert spark.sql("select udf_over_macro(1) as v").collect() == [Row(v=2)]

    def test_udf_register_defaults_to_string_and_returns_the_function(self, spark):
        def plus_one(v):
            return "none" if v is None else v + 1

        registered = spark.udf.register("plus_one", plus_one)
        assert registered.returnType == StringType()
        res = spark.sql("select plus_one(1) as a, plus_one(NULL) as b")
        assert res.schema.simpleString() == "struct<a:string,b:string>"
        assert res.collect()[0].asDict() == {"a": "2", "b": "none"}

    def test_udf_register_result_is_callable_on_columns(self, spark):
        registered = spark.udf.register("times_two", lambda v: None if v is None else v * 2, "bigint")
        res = spark.range(2).select(registered("id"), registered(F.col("id") + 1).alias("next"))
        assert res.columns == ["times_two(id)", "next"]
        assert res.collect() == [Row(0, 2), Row(2, 4)]

    def test_udf_register_result_carries_aggregates(self, spark):
        registered = spark.udf.register("plus_ten", lambda v: v + 10, "bigint")
        assert spark.range(3).select(registered(F.max("id")).alias("m")).collect() == [Row(m=12)]

    def test_udf_register_replaces_a_function_of_the_same_name(self, spark):
        spark.udf.register("twice", lambda v: v * 2, "bigint")
        spark.udf.register("twice", lambda v: v * 3, "bigint")
        assert spark.sql("select twice(2) as v").collect() == [Row(v=6)]

    def test_earlier_callable_keeps_its_function(self, spark):
        first = spark.udf.register("same_name", lambda v: 1, "int")
        second = spark.udf.register("same_name", lambda v: 2, "int")
        res = spark.range(1).select(first("id").alias("a"), second("id").alias("b")).collect()
        assert res == [Row(a=1, b=2)]
        assert spark.sql("select same_name(1) as v").collect() == [Row(v=2)]

    def test_udf_struct_arguments_are_rows(self, spark):
        schema = "s struct<x:int,y:string>, m map<string,int>, l array<struct<z:int>>, ms map<string,struct<w:int>>"
        df = spark.createDataFrame([((2, "a"), {"k": 1}, [(3,)], {"m": (4,)}), (None, None, None, None)], schema)

        def describe(s, m, l, ms):  # noqa: E741
            if s is None:
                return "none"
            kinds = " ".join(type(v).__name__ for v in (s, m, l[0], ms["m"]))
            return f"{kinds} {s['x']} {s['y']} {l[0]['z']} {ms['m']['w']}"

        registered = spark.udf.register("describe_values", describe, "string")
        # Spark gives a struct as a Row, also inside an array or a map, and a map as a dict.
        expected = [Row(v="Row dict Row Row 2 a 3 4"), Row(v="none")]
        assert df.select(registered("s", "m", "l", "ms").alias("v")).collect() == expected
        if not USE_ACTUAL_SPARK:
            # DuckDB calls the function SQL text names with its own values, a struct as a dict.
            expected = [Row(v="dict dict dict dict 2 a 3 4"), Row(v="none")]
        assert df.select(F.call_function("describe_values", "s", "m", "l", "ms").alias("v")).collect() == expected
        df.createOrReplaceTempView("udf_struct_arguments")
        assert spark.sql("select describe_values(s, m, l, ms) as v from udf_struct_arguments").collect() == expected

    def test_udf_timestamp_argument_is_a_local_time(self, spark):
        df = spark.createDataFrame([(datetime.datetime(2024, 1, 1, 12),)], "t timestamp")
        registered = spark.udf.register("show_time", lambda t: f"{t!r}", "string")
        assert df.select(registered("t").alias("v")).collect() == [Row(v="datetime.datetime(2024, 1, 1, 12, 0)")]

    def test_udf_variadic_arguments_in_sql(self, spark):
        registered = spark.udf.register("count_values", lambda *values: len(values), "int")
        assert spark.sql("select count_values(1, 2) as v").collect() == [Row(v=2)]
        assert spark.range(1).select(registered("id", "id").alias("v")).collect() == [Row(v=2)]

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="spark.sql takes DuckDB SQL, with its transactions")
    def test_register_inside_a_transaction(self, spark):
        spark.sql("BEGIN TRANSACTION")
        spark.sql("CREATE TABLE udf_in_transaction AS SELECT 1 AS a")
        spark.udf.register("plus_one_in_transaction", lambda v: v + 1, "int")
        assert spark.sql("SELECT plus_one_in_transaction(a) AS v FROM udf_in_transaction").collect() == [Row(v=2)]
        spark.sql("COMMIT")

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="spark.sql takes DuckDB SQL, with its schemas and macros")
    def test_sql_name_beside_a_macro_of_another_schema(self, spark):
        spark.sql("CREATE SCHEMA udf_other_schema")
        spark.sql("CREATE MACRO udf_other_schema.udf_beside(a) AS a")
        spark.udf.register("udf_beside", lambda v: v + 1, "int")
        assert spark.sql("SELECT udf_beside(1) AS v").collect() == [Row(v=2)]

    def test_udf_optional_arguments_in_sql(self, spark):
        def plus(a, b=10):
            return a + b

        spark.udf.register("plus_optional", plus, "int")
        if USE_ACTUAL_SPARK:
            assert spark.sql("select plus_optional(1) as a, plus_optional(1, 2) as b").collect() == [Row(a=11, b=3)]
            return
        # DuckDB takes every parameter of a Python function, defaults included.
        assert spark.sql("select plus_optional(1, 2) as b").collect() == [Row(b=3)]
        with pytest.raises(duckdb.BinderException):
            spark.sql("select plus_optional(1) as a")

    def test_udf_named_like_a_builtin(self, spark):
        df = spark.createDataFrame([("a b",)], "t string")
        reverse = spark.udf.register("reverse", lambda v: "u", "string")
        concat = spark.udf.register("concat", lambda a, b: "c", "string")
        res = df.select(reverse("t"), F.reverse("t"), concat("t", "t"), F.concat("t", "t"))
        assert res.columns == ["reverse(t)", "reverse(t)", "concat(t, t)", "concat(t, t)"]
        assert res.collect() == [("u", "b a", "c", "a ba b")]
        # Spark 3.5 runs the function SQL text names by a builtin's name, Spark 4 and DuckDB the builtin.
        assert spark.sql("select reverse('a b') as v").collect() == [Row(v="u" if SPARK_3 else "b a")]

    def test_registering_a_registered_udf(self, spark):
        df = spark.createDataFrame([("a",)], "t string")
        registered = spark.udf.register("exclaim", lambda v: v + "!", "string")
        again = spark.udf.register("exclaim_again", registered)
        assert again.returnType == StringType()
        assert df.select(again("t").alias("v")).collect() == [Row(v="a!")]
        assert spark.sql("select exclaim_again('x') as v").collect() == [Row(v="x!")]
        with pytest.raises(PySparkTypeError, match="CANNOT_SPECIFY_RETURN_TYPE_FOR_UDF"):
            spark.udf.register("exclaim_typed", registered, "int")

    @pytest.mark.parametrize(
        ("value", "type_string", "spark_value", "duckdb_value"),
        [
            (1, "double", None, 1.0),
            (1, "float", None, 1.0),
            (1.5, "int", None, 2),
            ("1", "int", None, 1),
            (True, "int", None, duckdb.TypeMismatchException),
            (1, "boolean", None, True),
            (Decimal("1.5"), "double", None, 1.5),
            (1, "decimal(10,2)", None, Decimal("1.00")),
            (1, "string", "1", "1"),
            (1.5, "string", "1.5", "1.5"),
            (True, "string", "true", duckdb.TypeMismatchException),
            ([1, "a"], "string", "[1, a]", "[1, a]"),
            ({"a": 1}, "string", "{a=1}", "{'a': 1}"),
            (2**40, "int", 0, duckdb.InvalidInputException),
            (300, "tinyint", 44, duckdb.InvalidInputException),
            ([1.5, 2], "array<int>", [None, 2], [2, 2]),
            ({"a": 1.5}, "map<string,int>", {"a": None}, {"a": 2}),
            ((1.5, "x"), "struct<a:int,b:string>", Row(a=None, b="x"), Row(a=2, b="x")),
            ({"b": "x", "a": 1}, "struct<a:int,b:string>", Row(a=1, b="x"), Row(a=1, b="x")),
            (datetime.datetime(2024, 1, 1, 5), "date", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)),
        ],
    )
    def test_udf_result_of_another_python_type(self, spark, value, type_string, spark_value, duckdb_value):
        # Spark gives NULL for a result of another Python type, cuts an integer to a narrower type and gives Java's
        # text for a string; DuckDB converts the value as it converts any Python value, or raises.
        registered = spark.udf.register("typed_result", lambda _: value, type_string)
        res = spark.range(1).select(registered("id").alias("v"))
        if not USE_ACTUAL_SPARK and isinstance(duckdb_value, type) and issubclass(duckdb_value, duckdb.Error):
            with pytest.raises(duckdb_value):
                res.collect()
            return
        assert res.collect() == [Row(v=spark_value if USE_ACTUAL_SPARK else duckdb_value)]

    @pytest.mark.parametrize(
        ("value", "spark_text", "duckdb_text"),
        [(b"ab", "[B@", "ab"), (datetime.date(2024, 1, 1), "java.util.GregorianCalendar[", "2024-01-01")],
    )
    def test_udf_string_result_of_bytes_or_a_date(self, spark, value, spark_text, duckdb_text):
        registered = spark.udf.register("text_result", lambda _: value, "string")
        (row,) = spark.range(1).select(registered("id").alias("v")).collect()
        if USE_ACTUAL_SPARK:
            # Spark gives the text of Java's toString, which names an object.
            assert row.v.startswith(spark_text)
        else:
            assert row.v == duckdb_text

    def test_udf_timestamp_result_is_a_local_time(self, spark):
        zone = datetime.timezone(datetime.timedelta(hours=2))
        noon = datetime.datetime(2024, 1, 1, 12)
        local = spark.udf.register("local_noon", lambda _: noon, "timestamp")
        aware = spark.udf.register("aware_noon", lambda _: noon.replace(tzinfo=zone), "timestamp")
        res = spark.range(1).select(local("id").alias("l"), aware("id").alias("a")).collect()
        assert res == [Row(l=noon, a=noon.replace(tzinfo=zone).astimezone().replace(tzinfo=None))]
        nested = spark.udf.register("nested_noon", lambda _: [noon], "array<timestamp>")
        if USE_ACTUAL_SPARK:
            assert spark.range(1).select(nested("id").alias("t")).collect() == [Row(t=[noon])]
        else:
            # DuckDB converts a datetime to a timestamp with a time zone only if it has one.
            with pytest.raises(duckdb.ConversionException):
                spark.range(1).select(nested("id")).collect()

    def test_udf_arguments_are_those_of_the_call(self, spark):
        def plus(a: str, b=0):
            return a + b

        def count(*arguments):
            return len(arguments)

        df = spark.createDataFrame([(1, 2)], "x int, y int")
        added = spark.udf.register("plus", plus, "int")
        counted = spark.udf.register("count_arguments", count, "int")
        seven = spark.udf.register("seven", lambda: 7, "int")
        res = df.select(
            added("x").alias("one"),
            added("x", "y").alias("both"),
            counted("x", "y").alias("two"),
            seven().alias("none"),
        )
        # Spark ignores the annotation, which DuckDB would cast the argument to.
        assert res.collect() == [Row(one=1, both=3, two=2, none=7)]

    def test_udf_timestamp_ntz_and_interval_results(self, spark):
        zone = datetime.timezone(datetime.timedelta(hours=2))
        noon = datetime.datetime(2024, 1, 1, 12, tzinfo=zone)
        wall_clock = spark.udf.register("wall_clock", lambda _: noon, "timestamp_ntz")
        span = spark.udf.register("span", lambda _: datetime.timedelta(days=1, seconds=5), "interval day to second")
        res = spark.range(1).select(wall_clock("id").alias("t"), span("id").alias("d"))
        assert res.collect() == [Row(t=datetime.datetime(2024, 1, 1, 12), d=datetime.timedelta(days=1, seconds=5))]
        registered = spark.udf.register("unreadable", lambda _: 5, "interval day to second")
        with pytest.raises(Exception):  # noqa: B017, PT011
            spark.range(1).select(registered("id")).collect()
        registered = spark.udf.register("date_as_timestamp_ntz", lambda _: datetime.date(2024, 1, 1), "timestamp_ntz")
        if USE_ACTUAL_SPARK:
            with pytest.raises(Exception):  # noqa: B017, PT011
                spark.range(1).select(registered("id")).collect()
        else:
            # DuckDB converts a date to a timestamp at midnight.
            res = spark.range(1).select(registered("id").alias("t")).collect()
            assert res == [Row(t=datetime.datetime(2024, 1, 1))]

    def test_udf_struct_result(self, spark):
        class Point:
            def __init__(self) -> None:
                self.b = "x"
                self.a = 1

        for value in [Row(1, "x"), (1, "x"), {"b": "x", "a": 1}]:
            registered = spark.udf.register("struct_result", lambda _, value=value: value, "a int, b string")
            assert spark.range(1).select(registered("id").alias("v")).collect() == [Row(v=Row(a=1, b="x"))]
        # Spark reads a Row made from keywords by position, as a tuple.
        registered = spark.udf.register("struct_result", lambda _: Row(b=1, a=2), "a int, b int")
        assert spark.range(1).select(registered("id").alias("v")).collect() == [Row(v=Row(a=1, b=2))]
        registered = spark.udf.register("struct_result", lambda _: Point(), "a int, b string")
        if USE_ACTUAL_SPARK:
            # Spark reads an object's attributes by field name.
            assert spark.range(1).select(registered("id").alias("v")).collect() == [Row(v=Row(a=1, b="x"))]
        else:
            with pytest.raises(duckdb.NotImplementedException):
                spark.range(1).select(registered("id")).collect()
        registered = spark.udf.register("struct_result", lambda _: "x", "a string")
        error = "UNEXPECTED_TUPLE_WITH_STRUCT" if USE_ACTUAL_SPARK else "can't be cast to the destination type STRUCT"
        with pytest.raises(Exception, match=error):
            spark.range(1).select(registered("id")).collect()

    def test_udf_result_spark_cannot_read_raises(self, spark):
        registered = spark.udf.register("date_as_timestamp", lambda _: datetime.date(2024, 1, 1), "timestamp")
        with pytest.raises(Exception, match="date"):
            spark.range(1).select(registered("id")).collect()
        registered = spark.udf.register("numpy_result", lambda _: np.int64(3), "bigint")
        with pytest.raises(Exception):  # noqa: B017, PT011
            spark.range(1).select(registered("id")).collect()
        registered = spark.udf.register("numpy_result", lambda _: np.float64(1.5), "double")
        if USE_ACTUAL_SPARK:
            with pytest.raises(Exception):  # noqa: B017, PT011
                spark.range(1).select(registered("id")).collect()
        else:
            # A NumPy float is a Python float, which DuckDB converts.
            assert spark.range(1).select(registered("id").alias("v")).collect() == [Row(v=1.5)]

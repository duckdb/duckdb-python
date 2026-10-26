import datetime
import math
from decimal import Decimal

import pytest

import duckdb

_ = pytest.importorskip("duckdb.experimental.spark")
pd = pytest.importorskip("pandas")

from pandas.testing import assert_frame_equal
from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import PySparkTypeError, PySparkValueError
from spark_namespace.sql.types import (
    ArrayType,
    IntegerType,
    Row,
    StringType,
    StructField,
    StructType,
)

if USE_ACTUAL_SPARK:
    import pyspark


@pytest.fixture
def pandasDF(spark):
    data = [["Scott", 50], ["Jeff", 45], ["Thomas", 54], ["Ann", 34]]
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=["Name", "Age"])
    return df


class TestPandasDataFrame:
    def test_pd_conversion_basic(self, spark, pandasDF):
        sparkDF = spark.createDataFrame(pandasDF)
        res = sparkDF.collect()
        sparkDF.show()
        expected = [
            Row(Name="Scott", Age=50),
            Row(Name="Jeff", Age=45),
            Row(Name="Thomas", Age=54),
            Row(Name="Ann", Age=34),
        ]
        assert res == expected

    def test_pd_conversion_schema(self, spark, pandasDF):
        mySchema = StructType([StructField("First Name", StringType(), True), StructField("Age", IntegerType(), True)])
        sparkDF = spark.createDataFrame(pandasDF, schema=mySchema)
        sparkDF.show()
        res = sparkDF.collect()
        expected = (
            "[Row(First Name='Scott', Age=50), Row(First Name='Jeff', Age=45), "
            "Row(First Name='Thomas', Age=54), Row(First Name='Ann', Age=34)]"
        )
        assert str(res) == expected

    def test_spark_to_pandas_dataframe(self, spark, pandasDF):
        sparkDF = spark.createDataFrame(pandasDF)
        res = sparkDF.toPandas()
        assert_frame_equal(res, pandasDF)

    def test_categorical_column(self, spark):
        pdf = pd.DataFrame({"c": pd.Categorical(["a", "b", None, "a"])})
        df = spark.createDataFrame(pdf)
        assert df.dtypes == [("c", "string")]
        # Spark 3.5 stores the text of the missing value, as it does for other pandas missing values.
        missing = "NaN" if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3.") else None
        assert [row.c for row in df.collect()] == ["a", "b", missing, "a"]


def _local(value):
    return value.astimezone().replace(tzinfo=None)


class TestPandasTimestamps:
    def test_naive_datetimes_are_timestamps(self, spark):
        values = [datetime.datetime(2024, 1, 1, 12), datetime.datetime(2024, 7, 1, 12), None]
        df = spark.createDataFrame(pd.DataFrame({"t": values, "i": [1, 2, 3]}))
        assert df.dtypes == [("t", "timestamp"), ("i", "bigint")]
        assert df.collect() == [Row(t=values[0], i=1), Row(t=values[1], i=2), Row(t=None, i=3)]
        assert df.toPandas()["t"].tolist()[:2] == values[:2]

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_naive_datetime_units(self, spark, unit):
        series = pd.Series([datetime.datetime(2024, 1, 1, 12)]).astype(f"datetime64[{unit}]")
        df = spark.createDataFrame(pd.DataFrame({"t": series}))
        assert df.dtypes == [("t", "timestamp")]
        assert df.collect() == [Row(t=datetime.datetime(2024, 1, 1, 12))]

    @pytest.mark.parametrize("zone", ["UTC", "Asia/Tokyo"])
    def test_aware_datetimes(self, spark, zone):
        series = pd.Series([datetime.datetime(2024, 1, 1, 12)]).dt.tz_localize(zone)
        df = spark.createDataFrame(pd.DataFrame({"t": series}))
        expected = _local(series[0].to_pydatetime())
        assert df.dtypes == [("t", "timestamp")]
        assert df.collect() == [Row(t=expected)]
        assert df.toPandas()["t"].tolist() == [expected]

    def test_to_pandas_gives_naive_local_times(self, spark):
        value = datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.timezone.utc)
        df = spark.createDataFrame([(value, datetime.datetime(2024, 1, 1, 12))], "t timestamp, n timestamp_ntz")
        res = df.toPandas()
        assert res["t"].dt.tz is None
        assert res["t"].tolist() == [_local(value)]
        assert res["n"].tolist() == [datetime.datetime(2024, 1, 1, 12)]

    def test_to_pandas_after_a_join(self, spark):
        left = spark.createDataFrame([(1, datetime.datetime(2024, 1, 1, 12))], "id int, t timestamp")
        right = spark.createDataFrame([(1, datetime.datetime(2024, 2, 1, 12))], "id int, t timestamp")
        res = left.join(right, left.id == right.id).toPandas()
        assert res.values.tolist() == [[1, datetime.datetime(2024, 1, 1, 12), 1, datetime.datetime(2024, 2, 1, 12)]]

    def test_to_pandas_of_dates(self, spark):
        df = spark.createDataFrame([(datetime.date(2024, 1, 2), 1), (None, 2)], "d date, i int")
        res = df.toPandas()
        assert str(res["d"].dtype) == "object"
        assert res["d"].tolist() == [datetime.date(2024, 1, 2), None]
        missing = spark.createDataFrame([(None,)], "d date").toPandas()
        assert str(missing["d"].dtype) == "object"
        assert missing["d"].tolist() == [None]
        res = spark.createDataFrame(
            [(datetime.date(1, 1, 1), {"k": datetime.date(9999, 12, 31)})], "d date, m map<string,date>"
        )
        assert res.toPandas().values.tolist() == [[datetime.date(1, 1, 1), {"k": datetime.date(9999, 12, 31)}]]

    def test_to_pandas_keeps_repeated_labels(self, spark):
        df = spark.createDataFrame([(1, 2)], ["a", "b"])
        assert df.toDF("x", "x").toPandas().columns.tolist() == ["x", "x"]
        assert df.toDF("x", "X").toPandas().columns.tolist() == ["x", "X"]
        left = spark.createDataFrame([(1, "l")], ["id", "v"])
        right = spark.createDataFrame([(1, "r")], ["id", "v"])
        res = left.join(right, left.id == right.id).toPandas()
        assert res.columns.tolist() == ["id", "v", "id", "v"]
        assert res.values.tolist() == [[1, "l", 1, "r"]]


class TestPandasNullability:
    @pytest.mark.parametrize(
        ("data", "schema"),
        [
            ({"v": [None]}, "v string not null"),
            ({"a": ["x"], "v": [None]}, "a string not null, v string not null"),
            ({"v": pd.Series([1, None], dtype="object")}, "v bigint not null"),
            ({"v": [pd.NaT, pd.Timestamp("2024-01-01")]}, "v timestamp not null"),
        ],
    )
    def test_none_in_a_field_that_is_not_nullable(self, spark, data, schema):
        with pytest.raises(PySparkValueError):
            spark.createDataFrame(pd.DataFrame(data), schema)

    def test_values_in_fields_that_are_not_nullable(self, spark):
        df = spark.createDataFrame(pd.DataFrame({"a": ["x", "y"], "v": [None, "z"]}), "a string not null, v string")
        assert df.collect() == [Row(a="x", v=None), Row(a="y", v="z")]
        floats = pd.DataFrame({"v": [1.0, None]})
        if USE_ACTUAL_SPARK:
            # Spark keeps a NaN, which is how pandas stores a missing float.
            (_, row) = spark.createDataFrame(floats, "v double not null").collect()
            assert math.isnan(row.v)
        else:
            # DuckDB reads a NaN in a pandas float column as NULL, which the field does not allow.
            for schema in ("v double not null", "v float not null"):
                with pytest.raises(PySparkValueError):
                    spark.createDataFrame(floats, schema)

    @pytest.mark.parametrize(
        "values", [pd.Series([float("nan"), "a"], dtype=object), pd.Series([pd.NA, "a"], dtype=object), [1.5, None]]
    )
    def test_missing_values_in_a_string_field_that_is_not_nullable(self, spark, values):
        df = pd.DataFrame({"v": values})
        if USE_ACTUAL_SPARK:
            # Spark keeps the text of the missing value.
            assert None not in [row.v for row in spark.createDataFrame(df, "v string not null").collect()]
        else:
            # DuckDB would read the missing value as NULL.
            with pytest.raises(PySparkValueError):
                spark.createDataFrame(df, "v string not null")

    def test_none_in_a_nested_field_that_is_not_nullable(self, spark):
        with pytest.raises(PySparkValueError):
            spark.createDataFrame(pd.DataFrame({"s": [{"a": None}]}), "s struct<a:int not null>")
        elements_not_null = StructType([StructField("l", ArrayType(IntegerType(), False))])
        with pytest.raises(PySparkValueError):
            spark.createDataFrame(pd.DataFrame({"l": [[1, None]]}), elements_not_null)
        df = spark.createDataFrame(pd.DataFrame({"s": [{"a": 1}, None]}), "s struct<a:int not null>")
        assert df.collect() == [Row(s=Row(a=1)), Row(s=None)]


class TestPandasValueTypes:
    @pytest.mark.parametrize(
        ("value", "schema"),
        [
            (["001"], "v array<int>"),
            (["001"], "v array<int> not null"),
            ({"a": "001"}, "v struct<a:int>"),
            ({"k": "001"}, "v map<string,int>"),
            ([1, "x"], "v array<bigint>"),
        ],
    )
    def test_nested_value_of_another_type(self, spark, value, schema):
        with pytest.raises(PySparkTypeError):
            spark.createDataFrame(pd.DataFrame({"v": [value]}), schema)

    def test_nested_values_that_fit(self, spark):
        stamp = pd.Timestamp("2024-01-02 03:04:05")
        df = pd.DataFrame({"v": [[1, None], None], "s": [{"a": "x"}, {"a": None}], "t": [[stamp], [None]]})
        res = spark.createDataFrame(df, "v array<int>, s struct<a:string>, t array<timestamp>")
        assert res.collect() == [
            Row(v=[1, None], s=Row(a="x"), t=[stamp.to_pydatetime()]),
            Row(v=None, s=Row(a=None), t=[None]),
        ]

    def test_integers_for_a_double_field(self, spark):
        df = pd.DataFrame({"v": [[1, 2]], "d": pd.Series([1], dtype=object)})
        if USE_ACTUAL_SPARK and not pyspark.__version__.startswith("3."):
            # Spark 4 converts the integers; the module follows Spark 3.5's verifier.
            assert spark.createDataFrame(df, "v array<double>, d double").collect() == [Row(v=[1.0, 2.0], d=1.0)]
        else:
            for schema in ("v array<double>, d string", "v string, d double"):
                with pytest.raises(PySparkTypeError):
                    spark.createDataFrame(df, schema)

    @pytest.mark.parametrize(
        ("values", "schema"),
        [
            ([1.5], "a int"),
            ([1.0, None], "a bigint not null"),
            ([1.5], "a decimal(10,2)"),
        ],
    )
    def test_float_that_does_not_fit(self, spark, values, schema):
        with pytest.raises(PySparkTypeError):
            spark.createDataFrame(pd.DataFrame({"a": values}), schema)

    def test_whole_floats_for_an_integral_field(self, spark):
        data = pd.DataFrame({"a": [2.0, None]})
        if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3."):
            # Spark 3.5 rejects every float for an integral field; the module follows Spark 4.
            with pytest.raises(PySparkTypeError):
                spark.createDataFrame(data, "a bigint")
        else:
            assert spark.createDataFrame(data, "a bigint").collect() == [Row(a=2), Row(a=None)]

    def test_values_that_fit(self, spark):
        assert spark.createDataFrame(pd.DataFrame({"a": [1, 2]}), "a bigint").collect() == [Row(a=1), Row(a=2)]
        assert spark.createDataFrame(pd.DataFrame({"a": [1.5]}), "a double").collect() == [Row(a=1.5)]


ROWS = 3000
NAIVE = datetime.datetime(2024, 5, 7, 7, 8, 9)
AWARE = NAIVE.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=2)))


def _outside_the_sample(sampled, odd):
    # The analyzer reads every third row of 3000, so the last row is never sampled.
    values = [sampled] * ROWS
    values[-1] = odd
    return pd.Series(values, dtype=object)


class _Point:
    def __init__(self, x) -> None:
        self.x = x


_typed_from_the_sample = pytest.mark.xfail(
    not USE_ACTUAL_SPARK, strict=True, reason="the object column is typed from its sample before the schema applies"
)
_not_turned_into_dicts = pytest.mark.xfail(
    not USE_ACTUAL_SPARK,
    strict=True,
    reason="lists, tuples and objects in a struct field are not turned into dicts in field order before the scan",
)
_not_turned_into_text = pytest.mark.xfail(
    not USE_ACTUAL_SPARK, strict=True, reason="values in a string field are not turned into text before the scan"
)


class TestPandasObjectColumnsWithASchema:
    @pytest.mark.parametrize(
        ("column", "schema", "expected_first", "expected_last"),
        [
            pytest.param(
                _outside_the_sample(datetime.date(2024, 5, 6), NAIVE),
                "a date",
                datetime.date(2024, 5, 6),
                datetime.date(2024, 5, 7),
                id="datetime_into_date",
            ),
            pytest.param(
                _outside_the_sample(datetime.date(2024, 5, 6), pd.Timestamp(NAIVE)),
                "a date",
                datetime.date(2024, 5, 6),
                datetime.date(2024, 5, 7),
                id="pandas_timestamp_into_date",
            ),
            pytest.param(
                _outside_the_sample(NAIVE, datetime.date(2024, 5, 6)),
                "a date",
                datetime.date(2024, 5, 7),
                datetime.date(2024, 5, 6),
                id="date_among_datetimes_into_date",
            ),
            pytest.param(
                _outside_the_sample(Decimal("1.5"), Decimal("1.25")),
                "a decimal(10,2)",
                Decimal("1.50"),
                Decimal("1.25"),
                id="wider_decimal_into_decimal",
                marks=_typed_from_the_sample,
            ),
            pytest.param(
                _outside_the_sample(1, "x"),
                "a string",
                "1",
                "x",
                id="str_among_ints_into_string",
                marks=_typed_from_the_sample,
            ),
            pytest.param(
                _outside_the_sample(1, 2**40),
                "a bigint",
                1,
                2**40,
                id="large_int_into_bigint",
                marks=_typed_from_the_sample,
            ),
            pytest.param(
                _outside_the_sample(b"ab", bytearray(b"cd")),
                "a binary",
                b"ab",
                b"cd",
                id="bytearray_into_binary",
            ),
            pytest.param(
                _outside_the_sample("a", b"ab"), "a string", "a", "b'ab'", id="bytes_into_string_reads_as_its_text"
            ),
            pytest.param(
                pd.Series([b"ab", b"cd"], dtype=object),
                "a string",
                "b'ab'",
                "b'cd'",
                id="fully_sampled_bytes_into_string_read_as_their_text",
                marks=_not_turned_into_text,
            ),
            pytest.param(_outside_the_sample([1, 2], (3, 4)), "a array<int>", [1, 2], [3, 4], id="tuple_into_array"),
        ],
    )
    def test_object_value_outside_the_sample_with_a_schema(self, spark, column, schema, expected_first, expected_last):
        df = spark.createDataFrame(pd.DataFrame({"a": column}), schema)
        rows = df.collect()
        assert (rows[0].a, rows[-1].a) == (expected_first, expected_last)

    @pytest.mark.parametrize(
        ("column", "schema", "expected_first", "expected_last"),
        [
            pytest.param(
                _outside_the_sample({"x": 1}, (2,)), "a struct<x:int>", Row(x=1), Row(x=2), id="tuple_among_dicts"
            ),
            pytest.param(
                _outside_the_sample({"x": 1}, Row(x=2)), "a struct<x:int>", Row(x=1), Row(x=2), id="row_among_dicts"
            ),
            pytest.param(
                _outside_the_sample({"x": 1}, [2]),
                "a struct<x:int>",
                Row(x=1),
                Row(x=2),
                id="list_among_dicts",
                marks=_not_turned_into_dicts,
            ),
            pytest.param(
                _outside_the_sample({"x": 1}, _Point(2)),
                "a struct<x:int>",
                Row(x=1),
                Row(x=2),
                id="object_among_dicts",
                marks=_not_turned_into_dicts,
            ),
            pytest.param(
                pd.Series([(1,), (2,)], dtype=object),
                "a struct<x:int>",
                Row(x=1),
                Row(x=2),
                id="tuples_only",
                marks=_not_turned_into_dicts,
            ),
            pytest.param(
                _outside_the_sample([{"x": 1}], [(2,)]),
                "a array<struct<x:int>>",
                [Row(x=1)],
                [Row(x=2)],
                id="tuple_element_of_an_array",
            ),
            pytest.param(
                _outside_the_sample({"k": {"x": 1}}, {"k": (2,)}),
                "a map<string,struct<x:int>>",
                {"k": Row(x=1)},
                {"k": Row(x=2)},
                id="tuple_value_of_a_map",
            ),
            pytest.param(
                _outside_the_sample({"s": {"x": 1}}, {"s": (2,)}),
                "a struct<s:struct<x:int>>",
                Row(s=Row(x=1)),
                Row(s=Row(x=2)),
                id="tuple_field_of_a_struct",
            ),
            pytest.param(
                pd.Series([{"s": (1,)}, {"s": (2,)}], dtype=object),
                "a struct<s:struct<x:int>>",
                Row(s=Row(x=1)),
                Row(s=Row(x=2)),
                id="tuple_fields_only",
                marks=_not_turned_into_dicts,
            ),
            pytest.param(
                _outside_the_sample({"y": 2, "x": 1}, (3, 4)),
                "a struct<x:int,y:int>",
                Row(x=1, y=2),
                Row(x=3, y=4),
                id="tuple_among_dicts_of_another_key_order",
                marks=_not_turned_into_dicts,
            ),
        ],
    )
    def test_struct_field_takes_each_value_the_schema_check_accepts(
        self, spark, column, schema, expected_first, expected_last
    ):
        df = spark.createDataFrame(pd.DataFrame({"a": column}), schema)
        rows = df.collect()
        assert (rows[0].a, rows[-1].a) == (expected_first, expected_last)

    def test_struct_values_of_the_frame_stay_as_given(self, spark):
        frame = pd.DataFrame({"a": _outside_the_sample({"y": 2, "x": 1}, (3, 4))})
        # repr keeps the key order, which dict equality ignores.
        given = repr(list(frame["a"]))
        spark.createDataFrame(frame, "a struct<x:int,y:int>").collect()
        assert repr(list(frame["a"])) == given

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="sets the session time zone through the DuckDB connection")
    @pytest.mark.xfail(strict=True, reason="an aware datetime outside a naive sample keeps its wall time")
    def test_aware_datetime_outside_the_sample_of_a_timestamp_field(self, spark):
        spark.conn.execute("SET TimeZone = 'UTC'")
        df = spark.createDataFrame(pd.DataFrame({"t": _outside_the_sample(NAIVE, AWARE)}), "t timestamp")
        rows = df.collect()
        assert (rows[0].t, rows[-1].t) == (_local(NAIVE.replace(tzinfo=datetime.timezone.utc)), _local(AWARE))

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="sets the session time zone through the DuckDB connection")
    @pytest.mark.parametrize(
        "column",
        [
            pytest.param(_outside_the_sample(NAIVE, AWARE), id="outside_a_naive_sample"),
            pytest.param(
                pd.Series([AWARE, AWARE], dtype=object),
                id="only_aware_values",
                marks=pytest.mark.xfail(
                    strict=True, reason="an aware datetime in a timestamp_ntz field reads in the session time zone"
                ),
            ),
        ],
    )
    def test_aware_datetime_in_a_timestamp_ntz_field_keeps_its_wall_time(self, spark, column):
        spark.conn.execute("SET TimeZone = 'UTC'")
        df = spark.createDataFrame(pd.DataFrame({"t": column}), "t timestamp_ntz")
        rows = df.collect()
        assert (rows[0].t, rows[-1].t) == (NAIVE, NAIVE)


@pytest.mark.skipif(USE_ACTUAL_SPARK, reason="Spark refuses to merge the inferred types with PySparkTypeError")
class TestPandasObjectColumnsWithoutASchema:
    @pytest.mark.parametrize(
        ("column", "match"),
        [
            pytest.param(_outside_the_sample(1, 2.5), r"(?s)(?=.*\bfloat\b)(?=.*\b2999\b)", id="float_among_ints"),
            pytest.param(
                _outside_the_sample(datetime.date(2024, 5, 6), NAIVE),
                r"(?s)(?=.*\bdatetime\b)(?=.*\b2999\b)",
                id="datetime_among_dates",
            ),
        ],
    )
    @pytest.mark.xfail(strict=True, reason="a value outside the sample is converted into the sampled type")
    def test_object_value_outside_the_sample_that_does_not_fit_raises(self, spark, column, match):
        df = spark.createDataFrame(pd.DataFrame({"a": column}))
        with pytest.raises(duckdb.InvalidInputException, match=match):
            df.collect()

    def test_midnight_datetime_outside_the_sample_of_dates_reads_as_its_date(self, spark):
        column = _outside_the_sample(datetime.date(2024, 5, 6), datetime.datetime(2024, 5, 7))
        df = spark.createDataFrame(pd.DataFrame({"a": column}))
        rows = df.collect()
        assert (rows[0].a, rows[-1].a) == (datetime.date(2024, 5, 6), datetime.date(2024, 5, 7))

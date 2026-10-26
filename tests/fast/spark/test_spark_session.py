import uuid

import pytest
from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException, IllegalArgumentException, PySparkNotImplementedError
from spark_namespace.sql.types import Row, StructType

_ = pytest.importorskip("duckdb.experimental.spark")
from spark_namespace.sql import SparkSession


class TestSparkSession:
    def test_spark_session_default(self):
        SparkSession.builder.getOrCreate()

    def test_spark_session(self):
        SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()

    def test_new_session(self, spark: SparkSession):
        spark.newSession()

    @pytest.mark.skip(reason="not tested yet")
    def test_retrieve_same_session(self):
        spark1 = SparkSession.builder.master("test").appName("test2").getOrCreate()
        spark2 = SparkSession.builder.getOrCreate()
        # Same connection should be returned
        assert spark1 == spark2

    def test_config(self):
        # Usage of config()
        SparkSession.builder.master("local[1]").appName("SparkByExamples.com").config(
            "spark.some.config.option", "config-value"
        ).getOrCreate()

    @pytest.mark.skip(reason="enableHiveSupport is not implemented yet")
    def test_hive_support(self):
        # Enabling Hive to use in Spark
        SparkSession.builder.master("local[1]").appName("SparkByExamples.com").config(
            "spark.sql.warehouse.dir", "<path>/spark-warehouse"
        ).enableHiveSupport().getOrCreate()

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="Different version numbers")
    def test_version(self, spark):
        version = spark.version
        assert version == "1.0.0"

    def test_get_active_session(self, spark):
        spark.getActiveSession()

    def test_read(self, spark):
        reader = spark.read  # noqa: F841

    def test_write(self, spark):
        df = spark.sql("select 42")
        writer = df.write  # noqa: F841

    def test_read_stream(self, spark):
        reader = spark.readStream  # noqa: F841

    def test_spark_context(self, spark):
        context = spark.sparkContext  # noqa: F841

    def test_sql(self, spark):
        spark.sql("select 42")

    def test_stop_context(self, spark):
        context = spark.sparkContext  # noqa: F841
        spark.stop()

    @pytest.mark.skipif(
        USE_ACTUAL_SPARK, reason="Can't create table with the local PySpark setup in the CI/CD pipeline"
    )
    def test_table(self, spark):
        spark.sql("create table tbl(a varchar(10))")
        spark.table("tbl")

    def test_range(self, spark):
        res_1 = spark.range(3).collect()
        res_2 = spark.range(3, 10, 2).collect()
        res_3 = spark.range(3, 6).collect()

        assert res_1 == [Row(id=0), Row(id=1), Row(id=2)]
        assert res_2 == [Row(id=3), Row(id=5), Row(id=7), Row(id=9)]
        assert res_3 == [Row(id=3), Row(id=4), Row(id=5)]

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            ((5,), [0, 1, 2, 3, 4]),
            ((0,), []),
            ((-3,), []),
            ((1, 5), [1, 2, 3, 4]),
            ((5, 1), []),
            ((3, 3), []),
            ((1, 10, 3), [1, 4, 7]),
            ((10, 1, -3), [10, 7, 4]),
            ((1, 5, 1, 2), [1, 2, 3, 4]),
        ],
    )
    def test_range_forms(self, spark, args, expected):
        df = spark.range(*args)
        assert df.columns == ["id"]
        assert df.schema.simpleString() == "struct<id:bigint>"
        assert [r.id for r in df.collect()] == expected

    def test_range_keywords(self, spark):
        assert spark.range(start=2, end=4, step=1, numPartitions=3).collect() == [Row(id=2), Row(id=3)]

    def test_range_step_zero(self, spark):
        with pytest.raises(IllegalArgumentException, match="step \\(0\\) cannot be 0"):
            spark.range(1, 5, 0).collect()

    def test_udf(self, spark):
        udf_registration = spark.udf  # noqa: F841


class TestStatementWithoutResult:
    # Spark 3.5 creates a Hive table unless the statement names a data source.
    TABLE_SOURCE = " USING parquet" if USE_ACTUAL_SPARK else ""

    @pytest.fixture
    def table_name(self, spark):
        name = f"no_result_{uuid.uuid4().hex}"
        yield name
        spark.sql(f"DROP TABLE IF EXISTS {name}")

    @pytest.fixture
    def statements(self, spark, table_name):
        created = spark.sql(f"CREATE TABLE {table_name} (a INT)" + self.TABLE_SOURCE)
        inserted = spark.sql(f"INSERT INTO {table_name} VALUES (1), (2)")
        return [created, inserted]

    def test_statement_runs(self, spark, table_name, statements):
        assert sorted(spark.table(table_name).collect()) == [Row(a=1), Row(a=2)]

    def test_schema(self, statements):
        for df in statements:
            assert df.schema == StructType([])
            assert df.columns == []
            assert df.dtypes == []

    @pytest.mark.skipif(not USE_ACTUAL_SPARK, reason="DuckDB has no relation without columns")
    def test_rows(self, spark, statements):
        other = spark.createDataFrame([(1,)], ["a"])
        for df in statements:
            assert df.count() == 0
            assert df.collect() == []
            assert df.union(df).collect() == []
            assert other.crossJoin(df).collect() == []

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="Spark represents a DataFrame without columns")
    @pytest.mark.parametrize(
        "use",
        [
            lambda df, other: df.count(),
            lambda df, other: df.collect(),
            lambda df, other: df.show(),
            lambda df, other: df.toPandas(),
            lambda df, other: df.union(df),
            lambda df, other: df.filter("true"),
            lambda df, other: df.join(other),
            lambda df, other: other.crossJoin(df),
            lambda df, other: "a" in df,
        ],
    )
    def test_only_the_schema_is_available(self, spark, statements, use):
        other = spark.createDataFrame([(1,)], ["a"])
        for df in statements:
            with pytest.raises(PySparkNotImplementedError):
                use(df, other)

    @pytest.mark.parametrize("operation", ["union", "unionByName", "intersect", "intersectAll", "exceptAll"])
    def test_set_operation_with_a_frame_without_columns(self, spark, statements, operation):
        # Spark raises because the column counts differ; DuckDB has no relation without columns.
        expected = AnalysisException if USE_ACTUAL_SPARK else PySparkNotImplementedError
        other = spark.createDataFrame([(1,)], ["a"])
        for df in statements:
            with pytest.raises(expected):
                getattr(other, operation)(df).collect()
            with pytest.raises(expected):
                getattr(df, operation)(other).collect()


class TestFrameWithoutColumns:
    # A DataFrame without columns, reached by selecting none as well as by a statement that returns none.
    @pytest.fixture
    def table_name(self, spark):
        name = f"no_columns_{uuid.uuid4().hex}"
        yield name
        spark.sql(f"DROP TABLE IF EXISTS {name}")

    @pytest.fixture(params=["varargs", "list", "statement"])
    def empty(self, spark, table_name, request):
        if request.param == "statement":
            return spark.sql(f"CREATE TABLE {table_name} (a INT)" + TestStatementWithoutResult.TABLE_SOURCE)
        df = spark.createDataFrame([(1, 2)], ["a", "b"])
        return df.select() if request.param == "varargs" else df.select([])

    @pytest.mark.parametrize("select", [lambda df: df.select(), lambda df: df.select([])], ids=["varargs", "list"])
    def test_select_of_no_columns(self, spark, select):
        empty = select(spark.createDataFrame([(1, 2)], ["a", "b"]))
        assert empty.columns == []
        assert empty.dtypes == []
        assert empty.schema == StructType([])

    @pytest.mark.parametrize("first", [True, False])
    def test_union_by_name_refuses_it(self, spark, empty, first):
        expected = AnalysisException if USE_ACTUAL_SPARK else PySparkNotImplementedError
        other = spark.createDataFrame([(1,)], ["a"])
        with pytest.raises(expected):
            (empty.unionByName(other) if first else other.unionByName(empty)).collect()

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="Spark fills the missing columns of a DataFrame without columns")
    @pytest.mark.parametrize("first", [True, False])
    def test_union_by_name_allowing_missing_columns_refuses_it(self, spark, empty, first):
        other = spark.createDataFrame([(1,)], ["a"])
        left, right = (empty, other) if first else (other, empty)
        with pytest.raises(PySparkNotImplementedError):
            left.unionByName(right, allowMissingColumns=True).collect()

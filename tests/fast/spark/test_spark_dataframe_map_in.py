import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace.sql import functions as F
from spark_namespace.sql.types import Row


class TestDataFrameMapInMethods:
    data = ((56, "Carol"), (20, "Alice"), (3, "Dave"), (3, "Anna"), (1, "Ben"))

    def test_map_in_pandas(self, spark):
        def filter_func(iterator):
            for pdf in iterator:
                yield pdf[pdf.age == 3]

        df = spark.createDataFrame(self.data, ["age", "name"])
        df = df.mapInPandas(filter_func, schema=df.schema)
        df = df.sort(["age", "name"])

        expected = [
            Row(age=3, name="Anna"),
            Row(age=3, name="Dave"),
        ]

        assert df.collect() == expected

    def test_map_in_pandas_empty_result(self, spark):
        def filter_func(iterator):
            for pdf in iterator:
                yield pdf[pdf.age > 100]

        df = spark.createDataFrame(self.data, ["age", "name"])
        df = df.mapInPandas(filter_func, schema=df.schema)

        expected = []

        assert df.collect() == expected
        assert df.schema == spark.createDataFrame([], schema=df.schema).schema

    def test_map_in_pandas_large_dataset_ensure_no_data_loss(self, spark):
        def identity_func(iterator):
            for pdf in iterator:
                pdf = pdf[pdf.id >= 0]  # Apply a filter to ensure the DataFrame is evaluated
                yield pdf

        n = 10_000_000

        pandas_df = pd.DataFrame(
            {
                "id": np.arange(n, dtype=np.int64),
                "value_float": np.random.rand(n).astype(np.float32),
                "value_int": np.random.randint(0, 1000, size=n, dtype=np.int32),
                "category": np.random.randint(0, 10, size=n, dtype=np.int8),
            }
        )

        df = spark.createDataFrame(pandas_df)
        df = df.mapInPandas(identity_func, schema=df.schema)
        # Apply filters to evaluate all dataframe
        df = df.filter(F.col("id") <= n).filter(F.col("id") >= 0).filter(F.col("category") >= 0)

        generated_pandas_df = df.toPandas()
        total_records = df.count()

        assert total_records == n
        assert pandas_df["id"].equals(generated_pandas_df["id"])

    def test_map_in_arrow(self, spark):
        def filter_func(iterator):
            for batch in iterator:
                df = batch.to_pandas()
                df = df[df.age == 3]
                yield pa.RecordBatch.from_pandas(df)

        df = spark.createDataFrame(self.data, ["age", "name"])
        df = df.mapInArrow(filter_func, schema=df.schema)
        df = df.sort(["age", "name"])

        expected = [
            Row(age=3, name="Anna"),
            Row(age=3, name="Dave"),
        ]

        assert df.collect() == expected

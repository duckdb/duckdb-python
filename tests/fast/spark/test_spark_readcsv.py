import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.sql.types import Row


class TestSparkReadCSV:
    def test_read_csv(self, spark, tmp_path):
        file_path = tmp_path / "basic.csv"
        file_path.write_text("1,2\n3,4\n5,6\n")
        df = spark.read.csv(file_path.as_posix())
        res = df.collect()

        expected_res = sorted([Row(column0=1, column1=2), Row(column0=3, column1=4), Row(column0=5, column1=6)])
        if USE_ACTUAL_SPARK:
            # Convert all values to strings as this is how Spark reads them by default
            expected_res = [Row(column0=str(row.column0), column1=str(row.column1)) for row in expected_res]
        assert sorted(res) == expected_res

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")


from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.sql.types import Row


@pytest.mark.skipif(
    USE_ACTUAL_SPARK,
    reason="load() without a format reads whatever the extension says here, while Spark defaults to parquet",
)
class TestSparkReadLoad:
    def test_read_load_no_format(self, spark, tmp_path):
        # 'load' without a 'format' falls back to building a bare SQL query out of the
        # path, which must be quoted as a string literal or it cannot survive a path
        # that contains anything the parser treats as syntax: a separator, a space, a colon.
        sub_dir = tmp_path / "dir with space"
        sub_dir.mkdir()
        file_path = sub_dir / "basic.csv"
        file_path.write_text("a,b\n1,2\n3,4\n")

        df = spark.read.load(file_path.as_posix())
        res = df.collect()

        assert sorted(res) == sorted([Row(a=1, b=2), Row(a=3, b=4)])

    def test_read_load_no_format_quote_in_path(self, spark, tmp_path):
        sub_dir = tmp_path / "dir's"
        sub_dir.mkdir()
        file_path = sub_dir / "basic.csv"
        file_path.write_text("a,b\n1,2\n")

        df = spark.read.load(file_path.as_posix())
        res = df.collect()

        assert res == [Row(a=1, b=2)]

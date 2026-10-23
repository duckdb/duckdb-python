import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import pytest

OPTIONAL_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "polars",
    "fsspec",
    "typing_extensions",
    "pyspark",
    "IPython",
    "adbc_driver_manager",
)

# The child starts with -S and runs site.main() after installing the finder, so no .pth file imports a package first.
BLOCK_IMPORTS = """
import sys
from importlib.abc import MetaPathFinder

BLOCKED = frozenset(sys.argv[1].split(","))


class BlockImports(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, BlockImports())

import site

site.main()
"""


def run_blocked(cwd, script, blocked=OPTIONAL_PACKAGES, pytz_tripwire=False):
    # An empty working directory keeps modules next to the caller's off the path that -c puts first.
    if pytz_tripwire:
        # pandas and pyarrow import pytz inside a try block, so the stand-in only serves scripts without them.
        (pathlib.Path(cwd) / "pytz.py").write_text('raise AssertionError("pytz imported")\n')
    proc = subprocess.run(
        [sys.executable, "-S", "-c", BLOCK_IMPORTS + textwrap.dedent(script), ",".join(blocked)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=cwd,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("package", OPTIONAL_PACKAGES)
def test_blocked_package_does_not_import(tmp_path, package):
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is not installed, so blocking it proves nothing")
    run_blocked(
        tmp_path,
        f"""
        try:
            import {package}
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("{package} imported")
        """,
    )


def test_duckdb_runs_a_query(tmp_path):
    run_blocked(
        tmp_path,
        """
        import duckdb

        assert duckdb.sql("SELECT 42 AS answer").fetchall() == [(42,)]
        """,
    )


def test_spark_basic_use(tmp_path):
    run_blocked(
        tmp_path,
        """
        from duckdb.experimental.spark.sql import SparkSession
        from duckdb.experimental.spark.sql import functions as F
        from duckdb.experimental.spark.sql.types import Row

        spark = SparkSession.builder.getOrCreate()
        df = spark.createDataFrame([(1, "a", 10), (2, "b", 20), (3, "a", 30)], ["id", "k", "v"])

        assert df.select("id", "k").filter(F.col("id") > 1).orderBy("id").collect() == [
            Row(id=2, k="b"),
            Row(id=3, k="a"),
        ]
        assert df.groupBy("k").agg(F.sum("v").alias("total")).orderBy("k").collect() == [
            Row(k="a", total=40),
            Row(k="b", total=20),
        ]
        names = spark.createDataFrame([("a", "first")], ["k", "name"])
        assert df.join(names, "k").orderBy("id").select("id", "name").collect() == [
            Row(id=1, name="first"),
            Row(id=3, name="first"),
        ]
        assert spark.sql("SELECT 1 AS one").collect() == [Row(one=1)]
        """,
    )


@pytest.mark.parametrize(
    ("method", "package", "blocked"),
    [
        ("toPandas", "pandas", OPTIONAL_PACKAGES),
        ("toPandas", "pandas", ("pandas",)),
        ("toArrow", "pyarrow", OPTIONAL_PACKAGES),
        ("toArrow", "pyarrow", ("pyarrow",)),
    ],
)
def test_spark_conversion_names_the_missing_package(tmp_path, method, package, blocked):
    run_blocked(
        tmp_path,
        f"""
        from duckdb.experimental.spark.sql import SparkSession

        df = SparkSession.builder.getOrCreate().createDataFrame([(1, "a")], ["id", "k"])
        try:
            df.{method}()
        except ImportError as error:
            assert error.name == {package!r}, repr(error)
            assert {package!r} in str(error), repr(error)
        else:
            raise AssertionError("{method}() did not raise ImportError")
        """,
        blocked,
    )


def test_spark_timestamps(tmp_path):
    run_blocked(
        tmp_path,
        """
        import datetime

        from duckdb.experimental.spark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        value = datetime.datetime(2024, 1, 2, 3, 4, 5)
        schema = "id long, ts timestamp, l array<timestamp>, s struct<t: timestamp>"
        df = spark.createDataFrame([(1, value, [value], (value,))], schema)
        row = df.collect()[0]
        assert (row.ts, row.l, row.s.t) == (value, [value], value), row
        assert df.first().ts == value
        df.show()
        (row,) = spark.sql("SELECT TIMESTAMPTZ '2024-01-02 03:04:05+00'::TIMESTAMPTZ_NS AS t").collect()
        assert row.t == datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc), row
        """,
    )


def test_spark_timestamps_to_pandas(tmp_path):
    pytest.importorskip("pandas", minversion="3.0.0")
    run_blocked(
        tmp_path,
        """
        import datetime

        from duckdb.experimental.spark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        value = datetime.datetime(2024, 1, 2, 3, 4, 5)
        pdf = spark.createDataFrame([(1, value)], ["id", "ts"]).toPandas()
        assert pdf["ts"].tolist() == [value], pdf
        """,
        tuple(package for package in OPTIONAL_PACKAGES if package not in ("numpy", "pandas")),
    )


@pytest.mark.parametrize("blocked", [OPTIONAL_PACKAGES, ("numpy",)])
def test_spark_udf_needs_numpy(tmp_path, blocked):
    """The binding's Python UDF path needs NumPy to register a function."""
    run_blocked(
        tmp_path,
        """
        import duckdb
        from duckdb.experimental.spark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        try:
            spark.udf.register("plus_one", lambda n: n + 1, "int")
        except duckdb.InvalidInputException as error:
            assert "'numpy' is required" in str(error), repr(error)
        else:
            raise AssertionError("registering a UDF without numpy did not raise")
        """,
        blocked,
    )


def test_spark_udf_with_a_timestamp_needs_no_package(tmp_path):
    """A timestamp reaches a Python UDF through the standard library's zoneinfo, so pytz is not needed."""
    if importlib.util.find_spec("numpy") is None:
        pytest.skip("a UDF needs numpy")
    run_blocked(
        tmp_path,
        """
        import datetime
        import sys

        import duckdb
        from duckdb.experimental.spark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        plus_one = spark.udf.register("plus_one", lambda n: n + 1, "int")
        assert spark.createDataFrame([(1,)], "n int").select(plus_one("n")).collect()[0][0] == 2
        year = spark.udf.register("year_of", lambda t: t.year, "int")
        df = spark.createDataFrame([(datetime.datetime(2024, 1, 2),)], "t timestamp")
        assert df.select(year("t")).collect()[0][0] == 2024
        assert "pytz" not in sys.modules
        """,
        blocked=tuple(package for package in OPTIONAL_PACKAGES if package != "numpy"),
        pytz_tripwire=True,
    )


def test_timestamptz_fetch_needs_no_package(tmp_path):
    run_blocked(
        tmp_path,
        """
        import datetime
        import sys
        import zoneinfo

        import duckdb

        con = duckdb.connect()
        con.execute("SET TimeZone = 'Europe/Amsterdam'")
        value = con.execute("SELECT TIMESTAMPTZ '2024-07-01 00:00:00+00'").fetchone()[0]
        assert value == datetime.datetime(2024, 7, 1, 2, tzinfo=zoneinfo.ZoneInfo("Europe/Amsterdam"))
        assert value.tzinfo is zoneinfo.ZoneInfo("Europe/Amsterdam")
        assert "pytz" not in sys.modules
        """,
        pytz_tripwire=True,
    )

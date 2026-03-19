import os
import warnings
from importlib import import_module
from pathlib import Path

import pytest

import duckdb

try:
    # need to ignore warnings that might be thrown deep inside pandas's import tree (from dateutil in this case)
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    pandas = import_module("pandas")
    warnings.resetwarnings()

    pyarrow_dtype = getattr(pandas, "ArrowDtype", None)
except ImportError:
    pandas = None
    pyarrow_dtype = None


# Version-aware helpers for Pandas 2.x vs 3.0 compatibility
def _get_pandas_ge_3():
    if pandas is None:
        return False
    from packaging.version import Version

    return Version(pandas.__version__) >= Version("3.0.0")


PANDAS_GE_3 = _get_pandas_ge_3()


def is_string_dtype(dtype):
    """Check if a dtype is a string dtype (works across Pandas 2.x and 3.0).

    Uses pd.api.types.is_string_dtype() which handles:
    - Pandas 2.x: object dtype for strings
    - Pandas 3.0+: str (StringDtype) for strings
    """
    return pandas.api.types.is_string_dtype(dtype)


def import_pandas():
    if pandas:
        return pandas
    else:
        pytest.skip("Couldn't import pandas")


# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
# https://stackoverflow.com/a/47700320
def pytest_addoption(parser):
    parser.addoption("--skiplist", action="append", nargs="+", type=str, help="skip listed tests")


def pytest_collection_modifyitems(config, items):
    tests_to_skip = config.getoption("--skiplist")
    if not tests_to_skip:
        # --skiplist not given in cli, therefore move on
        return

    # Combine all the lists into one
    skipped_tests = []
    for item in tests_to_skip:
        skipped_tests.extend(item)

    skip_listed = pytest.mark.skip(reason="included in --skiplist")
    for item in items:
        if item.name in skipped_tests:
            # test is named specifically
            item.add_marker(skip_listed)
        elif item.parent is not None and item.parent.name in skipped_tests:
            # the class is named specifically
            item.add_marker(skip_listed)


@pytest.fixture
def duckdb_empty_cursor(request):
    connection = duckdb.connect("")
    cursor = connection.cursor()
    return cursor


def getTimeSeriesData(nper=None, freq: "Frequency" = "B"):  # noqa: F821
    import string
    from datetime import datetime

    import numpy as np
    from pandas import DatetimeIndex, Series, bdate_range
    from pandas._typing import Frequency

    _N = 30
    _K = 4

    def getCols(k) -> str:
        return string.ascii_uppercase[:k]

    def makeDateIndex(k: int = 10, freq: Frequency = "B", name=None, **kwargs) -> DatetimeIndex:
        dt = datetime(2000, 1, 1)
        dr = bdate_range(dt, periods=k, freq=freq, name=name)
        return DatetimeIndex(dr, name=name, **kwargs)

    def makeTimeSeries(nper=None, freq: Frequency = "B", name=None) -> Series:
        if nper is None:
            nper = _N
        return Series(np.random.randn(nper), index=makeDateIndex(nper, freq=freq), name=name)

    return {c: makeTimeSeries(nper, freq) for c in getCols(_K)}


def pandas_2_or_higher():
    from packaging.version import Version

    return Version(import_pandas().__version__) >= Version("2.0.0")


def pandas_supports_arrow_backend():
    try:
        from pandas.compat import pa_version_under11p0

        if pa_version_under11p0:
            return False
    except ImportError:
        return False
    return pandas_2_or_higher()


@pytest.fixture
def require():
    def _require(extension_name, db_name="") -> duckdb.DuckDBPyConnection | None:
        # Paths to search for extensions

        build = Path(__file__).parent.parent / "build"
        extension = "extension/*/*.duckdb_extension"

        extension_search_patterns = [
            build / "release" / extension,
            build / "debug" / extension,
        ]

        # DUCKDB_PYTHON_TEST_EXTENSION_PATH can be used to add a path for the extension test to search for extensions
        if "DUCKDB_PYTHON_TEST_EXTENSION_PATH" in os.environ:
            env_extension_path = os.getenv("DUCKDB_PYTHON_TEST_EXTENSION_PATH")
            env_extension_path = env_extension_path.rstrip("/")
            extension_search_patterns.append(env_extension_path + "/*/*.duckdb_extension")
            extension_search_patterns.append(env_extension_path + "/*.duckdb_extension")

        extension_paths_found = []
        for pattern in extension_search_patterns:
            extension_paths_found.extend(list(Path(pattern).resolve().glob("*")))

        for path in extension_paths_found:
            print(path)
            if path.endswith(extension_name + ".duckdb_extension"):
                conn = duckdb.connect(db_name, config={"allow_unsigned_extensions": "true"})
                conn.execute(f"LOAD '{path}'")
                return conn
        pytest.skip(f"could not load {extension_name}")

    return _require


# By making the scope 'function' we ensure that a new connection gets created for every function that uses the fixture
@pytest.fixture
def spark():
    if not hasattr(spark, "session"):
        # Cache the import
        from spark_namespace.sql import SparkSession as session

        spark.session = session

    return spark.session.builder.appName("pyspark").getOrCreate()


@pytest.fixture
def duckdb_cursor():
    connection = duckdb.connect("")
    yield connection
    connection.close()


@pytest.fixture
def integers(duckdb_cursor):
    cursor = duckdb_cursor
    cursor.execute("CREATE TABLE integers (i integer)")
    cursor.execute(
        """
        INSERT INTO integers VALUES
            (0),
            (1),
            (2),
            (3),
            (4),
            (5),
            (6),
            (7),
            (8),
            (9),
            (NULL)
    """
    )
    yield
    cursor.execute("drop table integers")


@pytest.fixture
def timestamps(duckdb_cursor):
    cursor = duckdb_cursor
    cursor.execute("CREATE TABLE timestamps (t timestamp)")
    cursor.execute("INSERT INTO timestamps VALUES ('1992-10-03 18:34:45'), ('2010-01-01 00:00:01'), (NULL)")
    yield
    cursor.execute("drop table timestamps")

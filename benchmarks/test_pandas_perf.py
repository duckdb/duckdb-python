"""pandas read/write, numpy-backed vs arrow-backed frames. See benchmarks/README.md.

Column backing selects the path: numpy-backed -> NumpyArray scan; arrow-backed (ArrowDtype) -> zero-copy arrow.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest
from _scale import scaled

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

N = scaled(500_000)
WRITE_Q_NUM = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N}) t(i)"
WRITE_Q_STR = f"SELECT ('str_value_' || i) AS s FROM range({N}) t(i)"
_STRINGS = [f"str_value_{i}" for i in range(N)]
# mixed ASCII + non-ASCII + null, as the README traps require for string reads that pass through object values
_MIXED = ["ascii_value_", "café_", "naïve_", "日本語_", None]
_MIXED_STRINGS = [None if _MIXED[i % 5] is None else f"{_MIXED[i % 5]}{i}" for i in range(N)]

N_CHUNKED = scaled(100_000)
_NON_NULL = [p for p in _MIXED if p is not None]
_MIXED_STRINGS_NO_NULLS = [f"{_NON_NULL[i % len(_NON_NULL)]}{i}" for i in range(N_CHUNKED)]
CHUNKED_QUERIES = {
    "bind": "SELECT s FROM t LIMIT 0",
    "sum_i": "SELECT sum(i) FROM t",
    "sum_length": "SELECT sum(length(s)) FROM t",
}

# READ (sum over a registered frame) is engine-aggregate dominated -> informational. Only the NUMPY-backed df()
# WRITE is binding-dominated -> gate; the arrow-backed WRITE goes through pyarrow's to_pandas -> informational.


@pytest.fixture(scope="module")
def df_numpy_numeric() -> pd.DataFrame:
    return pd.DataFrame({"a": np.arange(N, dtype="int64"), "b": np.arange(N, dtype="float64") * 1.5})


@pytest.fixture(scope="module")
def df_numpy_string() -> pd.DataFrame:
    # explicit object dtype -> the reworked numpy-backed object-string / analyzer path
    return pd.DataFrame({"s": pd.Series(_STRINGS, dtype=object)})


@pytest.fixture(scope="module")
def df_arrow_numeric() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": pd.array(np.arange(N), dtype=pd.ArrowDtype(pa.int64())),
            "b": pd.array(np.arange(N) * 1.5, dtype=pd.ArrowDtype(pa.float64())),
        }
    )


@pytest.fixture(scope="module")
def df_arrow_string() -> pd.DataFrame:
    return pd.DataFrame({"s": pd.array(_STRINGS, dtype=pd.ArrowDtype(pa.string()))})


@pytest.fixture(scope="module", params=["str", "string[python]"], ids=["str", "python"])
def df_string_dtype(request: pytest.FixtureRequest) -> pd.DataFrame:
    # "str" is the pandas 3 default (Arrow backed); "string[python]" holds Python str objects
    return pd.DataFrame({"s": pd.Series(_MIXED_STRINGS, dtype=request.param)})


@pytest.fixture(
    scope="module",
    params=[(rows, nulls) for rows in (1, 10, 100, 1000) for nulls in (False, True)],
    ids=lambda p: f"{p[0]}-{'nulls' if p[1] else 'no_nulls'}",
)
def df_str_chunked(request: pytest.FixtureRequest) -> pd.DataFrame:
    # row appends and concat leave a str column with one Arrow chunk per piece; the rows per chunk set the bind cost
    rows, nulls = request.param
    # nulls send pandas' to_numpy through a per chunk masked copy, so its cost per chunk differs with and without them
    values = _MIXED_STRINGS[:N_CHUNKED] if nulls else _MIXED_STRINGS_NO_NULLS
    pieces = [values[k : k + rows] for k in range(0, N_CHUNKED, rows)]
    s = pd.Series(pa.chunked_array(pieces, type=pa.string()), dtype="str")
    return pd.DataFrame({"i": np.arange(N_CHUNKED, dtype="int64"), "s": s})


@pytest.fixture(scope="module", params=["naive", "aware"])
def df_object_datetime(request: pytest.FixtureRequest) -> pd.DataFrame:
    # explicit object dtype keeps datetime objects, so the scan converts each one instead of a datetime64 column
    zone = datetime.timezone(datetime.timedelta(hours=2)) if request.param == "aware" else None
    start = datetime.datetime(2020, 1, 1, 12, 30, 15, 250_000, tzinfo=zone)
    return pd.DataFrame({"t": pd.Series([start + datetime.timedelta(seconds=i) for i in range(N)], dtype=object)})


# READ: pandas -> duckdb. sum/length force a full scan.


@pytest.mark.informational
def test_read_pandas_numpy_numeric(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_numpy_numeric: pd.DataFrame
) -> None:
    con.register("t", df_numpy_numeric)
    con.execute("SELECT sum(a), sum(b) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_numpy_string(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_numpy_string: pd.DataFrame
) -> None:
    con.register("t", df_numpy_string)
    con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_string_dtype(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_string_dtype: pd.DataFrame
) -> None:
    con.register("t", df_string_dtype)
    con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


@pytest.mark.informational
@pytest.mark.parametrize("query", list(CHUNKED_QUERIES.values()), ids=list(CHUNKED_QUERIES))
def test_read_pandas_str_chunked(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_str_chunked: pd.DataFrame, query: str
) -> None:
    con.register("t", df_str_chunked)
    con.execute(query).fetchall()  # warm
    benchmark(lambda: con.execute(query).fetchall())


@pytest.mark.informational
def test_read_pandas_object_datetime(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_object_datetime: pd.DataFrame
) -> None:
    con.register("t", df_object_datetime)
    con.execute("SELECT count(t), max(t) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(t), max(t) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_arrow_numeric(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_arrow_numeric: pd.DataFrame
) -> None:
    con.register("t", df_arrow_numeric)
    con.execute("SELECT sum(a), sum(b) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_arrow_string(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_arrow_string: pd.DataFrame
) -> None:
    con.register("t", df_arrow_string)
    con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


# WRITE: duckdb -> pandas. df() is the reworked numpy-backed path; the arrow-backed frame goes via
# duckdb-arrow + pyarrow.to_pandas(ArrowDtype). Both eagerly materialize the whole frame.


@pytest.mark.gate
def test_write_pandas_numpy_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_NUM).df())


@pytest.mark.gate
def test_write_pandas_numpy_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_STR).df())


@pytest.mark.gate
def test_write_pandas_numpy_numeric_with_nulls(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    # REAL nulls -> the masked_array build + masked-to-pd.NA rewrite the cutover reworked (see README traps)
    q = (
        "SELECT CASE WHEN i % 10 = 0 THEN NULL ELSE i::BIGINT END AS a, "
        f"CASE WHEN i % 10 = 0 THEN NULL ELSE (i * 1.5)::DOUBLE END AS b FROM range({N}) t(i)"
    )
    benchmark(lambda: con.sql(q).df())


@pytest.mark.gate
def test_write_pandas_numpy_timestamp(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    q = f"SELECT TIMESTAMP '2020-01-01' + (i * INTERVAL 1 SECOND) AS t FROM range({N}) t(i)"
    benchmark(lambda: con.sql(q).df())


@pytest.mark.informational  # to_pandas() half is pyarrow library code
def test_write_pandas_arrow_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_NUM).to_arrow_table().to_pandas(types_mapper=pd.ArrowDtype))


@pytest.mark.informational  # to_pandas() half is pyarrow library code
def test_write_pandas_arrow_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_STR).to_arrow_table().to_pandas(types_mapper=pd.ArrowDtype))

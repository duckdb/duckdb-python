"""Columnar produce: df(), fetchnumpy(), fetch_df_chunk(), per type, null vs no-null. See benchmarks/README.md.

Covers the with-NULLS masked_array branch, datetime, and wide-internal types (hugeint/uuid/decimal128).
"""

from __future__ import annotations

import gc
import sys
import tracemalloc
from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import duckdb
import numpy as np  # noqa: F401  (pinned identically A/B so the env matches the other modules)

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

N = scaled(500_000)
TYPE_N = scaled(200_000)  # wide-internal types (hugeint/uuid/decimal128) are heavier per cell

Q_NUM = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N}) t(i)"
Q_NUM_NULLS = (
    "SELECT CASE WHEN i % 10 = 0 THEN NULL ELSE i::BIGINT END AS a, "
    f"CASE WHEN i % 10 = 0 THEN NULL ELSE (i * 1.5)::DOUBLE END AS b FROM range({N}) t(i)"
)
Q_STR = f"SELECT ('str_value_' || i) AS s FROM range({N}) t(i)"
Q_TS = f"SELECT TIMESTAMP '2020-01-01' + (i * INTERVAL 1 SECOND) AS t FROM range({N}) t(i)"
Q_HUGEINT = f"SELECT (i::HUGEINT * 1000000000000) AS h FROM range({TYPE_N}) t(i)"
Q_UUID = f"SELECT gen_random_uuid() AS u FROM range({TYPE_N}) t(i)"
Q_DEC128 = f"SELECT ((i * 1.5)::DECIMAL(28, 6)) AS d FROM range({TYPE_N}) t(i)"


# gate: df()/fetchnumpy() fully materialize numpy-backed columns (ArrayWrapper fill, binding-dominated).
def _bench_df(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, query: str) -> None:
    con.sql(query).df()  # warm
    benchmark(lambda: con.sql(query).df())


def _bench_numpy(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, query: str) -> None:
    con.sql(query).fetchnumpy()  # warm
    benchmark(lambda: con.sql(query).fetchnumpy())


# df(): the production numpy-backed columnar path. no-null vs REAL-null vs string vs timestamp vs wide types.


@pytest.mark.informational
def test_df_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_df(benchmark, con, Q_NUM)


@pytest.mark.gate
def test_df_numeric_with_nulls(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_df(benchmark, con, Q_NUM_NULLS)  # REAL nulls -> masked_array branch (see README traps)


@pytest.mark.gate
def test_df_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_df(benchmark, con, Q_STR)


@pytest.mark.gate
def test_df_timestamp(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_df(benchmark, con, Q_TS)


@pytest.mark.gate
def test_df_hugeint(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_df(benchmark, con, Q_HUGEINT)


@pytest.mark.gate
def test_df_uuid(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_df(benchmark, con, Q_UUID)


@pytest.mark.gate
def test_df_decimal128(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_df(benchmark, con, Q_DEC128)


# fetchnumpy(): same FetchNumpyInternal, without the DataFrame wrap.


@pytest.mark.informational
def test_fetchnumpy_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_numpy(benchmark, con, Q_NUM)


@pytest.mark.gate
def test_fetchnumpy_numeric_with_nulls(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_numpy(benchmark, con, Q_NUM_NULLS)


@pytest.mark.informational  # per-chunk streaming drain (GIL-per-chunk), not gated
def test_fetch_df_chunk_loop(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    def run() -> int:
        con.execute(Q_NUM)
        rows = 0
        while True:
            chunk = con.fetch_df_chunk()
            if len(chunk) == 0:
                break
            rows += len(chunk)
        return rows

    con.execute(Q_NUM).fetch_df_chunk()  # warm
    benchmark(run)


@pytest.mark.informational  # torch is local-only (importorskip); torch lib work dilutes it
def test_torch_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    pytest.importorskip("torch")
    q = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({TYPE_N}) t(i)"
    con.sql(q).torch()  # warm
    benchmark(lambda: con.sql(q).torch())


# Memory guard (secondary signal, not a codspeed benchmark; codspeed walltime tracks neither memory nor allocs).
# tracemalloc captures the PEAK Python-tracked allocation of ONE df()-with-nulls call. reset_peak() runs AFTER
# the warm so the warm does not set a high-water mark that swallows the measured call. tracemalloc reports bytes
# on every platform (portable to Linux CI). CAVEAT: it only sees Python-level allocs, not the C numpy buffers, so
# it catches a gross Python-object blowup (masked-to-pd.NA gone wrong) but is not a total-RSS gate; that is
# codspeed memory mode's job (deferred, see PLAN.md).


def test_mem_df_with_nulls() -> None:
    con = duckdb.connect(config={"threads": 1})
    try:
        tracemalloc.start()
        warm = con.sql(Q_NUM_NULLS).df()  # populate one-time import / type caches
        del warm
        gc.collect()
        tracemalloc.reset_peak()  # discount the warm's transient peak BEFORE the measured call
        out = con.sql(Q_NUM_NULLS).df()
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del out
    finally:
        con.close()
    print(f"\n[mem] df()-with-nulls tracemalloc peak = {peak / 1e6:.1f} MB", file=sys.stderr)
    # a 500k x 2-col masked df is a few MB of Python-tracked allocs; a gross blowup is tens+ MB. 100 MB ceiling
    # catches that without flaking.
    assert peak < 100_000_000

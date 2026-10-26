"""Concurrency / GIL pressure across thread counts. Walltime-only, never gated. See benchmarks/README.md.

The ONE dimension the single-threaded rest of the suite cannot see: Python objects threading through PARALLEL
core execution. Primary signal is LOCAL WALLTIME:
  * scan benches  -> parallel speedup; a per-batch Produce GIL regression shows as reduced speedup.
  * native UDF    -> ~flat scaling = the GIL tax on per-row Python calls.
  * arrow UDF     -> observed NEGATIVE scaling (per-chunk convert + GIL contention).

Under CI Callgrind threads are serialized, so wall-clock contention is invisible there; the deterministic count
still captures per-batch Produce GIL calls + UDF dispatch. Never gated either way.

GOTCHA: the arrow scan splits each batch into ranges of 122880 rows, so a batch shorter than two ranges is one serial
scan unit. The arrow scan bench MUST use a CPU-heavy aggregate (a cheap sum is bandwidth-bound and won't parallelize).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb.sqltypes import BIGINT

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")
import numpy as np  # noqa: E402  (after importorskip, matching the suite convention)
import pandas as pd  # noqa: E402

pytestmark = pytest.mark.informational

N_SCAN = 1_000_000
BATCH = 20_000  # -> 50 record batches, each one serial scan unit (see GOTCHA)
N_UDF_NATIVE = 200_000  # native UDF = one Python call per row; keep modest (Callgrind instruments every call)
N_UDF_ARROW = 1_000_000  # arrow UDF = one call per chunk (vectorized)
THREADS = [1, 4, 8]

# CPU-heavy aggregate so the parallel scan engages worker threads. The binding signal is the per-batch Produce
# GIL handoff.
HEAVY = "sin(a) * cos(b) + sqrt(abs(a)) + ln(abs(a) + 1)"


@pytest.fixture(scope="module")
def arrow_multibatch() -> pa.Table:
    a = pa.array(np.arange(N_SCAN), type=pa.int64())
    b = pa.array(np.arange(N_SCAN, dtype="float64") * 1.5, type=pa.float64())
    return pa.Table.from_batches(pa.table({"a": a, "b": b}).to_batches(max_chunksize=BATCH))


@pytest.fixture(scope="module")
def arrow_one_chunk() -> pa.Table:
    return pa.table(
        {
            "a": pa.array(np.arange(N_SCAN), type=pa.int64()),
            "b": pa.array(np.arange(N_SCAN, dtype="float64") * 1.5, type=pa.float64()),
        }
    )


@pytest.fixture(scope="module")
def pandas_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": np.arange(N_SCAN), "b": np.arange(N_SCAN, dtype="float64") * 1.5})


# Parallel SCAN: arrow batches / pandas chunks pulled through the binding by engine worker threads; the scan
# Produce acquires/releases the GIL per batch across threads.


@pytest.mark.parametrize("threads", THREADS)
def test_scan_arrow_parallel(benchmark: BenchmarkFixture, arrow_multibatch: pa.Table, threads: int) -> None:
    con = duckdb.connect(config={"threads": threads})
    try:
        con.register("t", arrow_multibatch)
        q = f"SELECT sum({HEAVY}) FROM t"
        con.execute(q).fetchall()  # warm
        benchmark(lambda: con.execute(q).fetchall())
    finally:
        con.close()


@pytest.mark.parametrize("threads", THREADS)
def test_scan_arrow_one_chunk_parallel(benchmark: BenchmarkFixture, arrow_one_chunk: pa.Table, threads: int) -> None:
    # the speedup over threads shows whether the scan splits a single batch into ranges (see GOTCHA)
    con = duckdb.connect(config={"threads": threads})
    try:
        con.register("t", arrow_one_chunk)
        q = f"SELECT sum({HEAVY}) FROM t"
        con.execute(q).fetchall()  # warm
        benchmark(lambda: con.execute(q).fetchall())
    finally:
        con.close()


@pytest.mark.parametrize("threads", THREADS)
def test_scan_pandas_parallel(benchmark: BenchmarkFixture, pandas_frame: pd.DataFrame, threads: int) -> None:
    con = duckdb.connect(config={"threads": threads})
    try:
        con.register("t", pandas_frame)
        q = f"SELECT sum({HEAVY}) FROM t"
        con.execute(q).fetchall()  # warm
        benchmark(lambda: con.execute(q).fetchall())
    finally:
        con.close()


# Parallel UDF: the engine scans a MATERIALIZED table (range() does not parallelize) and invokes a Python UDF
# from multiple worker threads. Native = per-row call under the GIL (GIL tax); arrow = per-chunk convert.


@pytest.mark.parametrize("threads", THREADS)
def test_udf_native_parallel(benchmark: BenchmarkFixture, threads: int) -> None:
    con = duckdb.connect(config={"threads": threads})
    try:
        con.execute(f"CREATE TABLE t AS SELECT i AS a FROM range({N_UDF_NATIVE}) s(i)")  # materialized -> parallel scan
        con.create_function("pyf", lambda x: (x * 2 + 1) % 97, [BIGINT], BIGINT)
        con.execute("SELECT sum(pyf(a)) FROM t").fetchall()  # warm
        benchmark(lambda: con.execute("SELECT sum(pyf(a)) FROM t").fetchall())
    finally:
        con.close()


@pytest.mark.parametrize("threads", THREADS)
def test_udf_arrow_parallel(benchmark: BenchmarkFixture, threads: int) -> None:
    con = duckdb.connect(config={"threads": threads})
    try:
        con.execute(f"CREATE TABLE t AS SELECT i AS a FROM range({N_UDF_ARROW}) s(i)")  # materialized -> parallel scan
        con.create_function("af", lambda x: pc.add(x, 1), [BIGINT], BIGINT, type="arrow")
        con.execute("SELECT sum(af(a)) FROM t").fetchall()  # warm
        benchmark(lambda: con.execute("SELECT sum(af(a)) FROM t").fetchall())
    finally:
        con.close()

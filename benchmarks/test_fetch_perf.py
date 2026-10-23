"""OUT-row fetch: fetchall, fetchone/fetchmany loops, wide/expensive scalar types. See benchmarks/README.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

# gate: OUT-row fetch materializes every row to Python (binding-dominated); the range() scan is cheap.
pytestmark = pytest.mark.gate

# scaled() shrinks N under BENCH_SCALE in the CI sweep; full N locally. The range(2048) *_gate probes are the
# compile+fetch fixed-cost baseline and are deliberately NOT scaled.
N_ROW = scaled(200_000)  # numeric fetch (BIGINT/INTEGER/DOUBLE/2col/null/decimal128)
N_STR = scaled(100_000)  # varchar/blob/mixed-wide/timestamptz + fetchone/fetchmany loops
N_NEST = scaled(50_000)  # heterogeneous scalar/list/struct row


def _bench_fetchall(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, query: str) -> None:
    con.execute(query).fetchall()  # warm the engine before measuring
    benchmark(lambda: con.execute(query).fetchall())


def test_fetchall_int(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(benchmark, con, f"SELECT i::BIGINT AS a FROM range({N_ROW}) t(i)")


def test_fetchall_smallint(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(benchmark, con, f"SELECT (i % 100)::INTEGER AS a FROM range({N_ROW}) t(i)")


def test_fetchall_double(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(benchmark, con, f"SELECT (i * 1.5)::DOUBLE AS a FROM range({N_ROW}) t(i)")


def test_fetchall_2int(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(benchmark, con, f"SELECT i::BIGINT AS a, (i + 1)::BIGINT AS b FROM range({N_ROW}) t(i)")


def test_fetchall_str(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(benchmark, con, f"SELECT ('str_value_' || i) AS s FROM range({N_STR}) t(i)")


def test_fetchall_mixed(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    query = (
        "SELECT i::BIGINT AS bi, ('str_' || i) AS s, [i, i + 1, i + 2] AS lst, "
        f"{{'a': i, 'b': i + 1}} AS st FROM range({N_NEST}) t(i)"
    )
    _bench_fetchall(benchmark, con, query)


def test_fetchone_iter(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    query = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N_STR}) t(i)"

    def run() -> None:
        rel = con.execute(query)
        while rel.fetchone() is not None:
            pass

    benchmark(run)


# small-N *_gate variants: at range(2048) the measured region is ~60% SQL compile + engine, ~40% fetch, so these
# catch a fixed-cost regression (not a pure per-row one). Plus expensive scalar types (timestamptz zoneinfo per row,
# blob, null-heavy), a heterogeneous per-cell-dispatch row, and the batched fetchmany loop.


def test_fetchall_int_gate(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a FROM range(2048) t(i)")


def test_fetchall_2int_gate(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a, (i + 1)::BIGINT AS b FROM range(2048) t(i)")


def test_fetchall_null_heavy(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(benchmark, con, f"SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE i::BIGINT END FROM range({N_ROW}) t(i)")


def test_fetchall_timestamptz(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(
        benchmark, con, f"SELECT (TIMESTAMPTZ '2020-01-01' + (i * INTERVAL 1 SECOND)) FROM range({N_STR}) t(i)"
    )


def test_fetchall_timetz(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    # TIMETZ is fetchall only: df() does not support it, so it cannot join TYPE_EXPR in the roundtrip module
    _bench_fetchall(benchmark, con, f"SELECT TIMETZ '00:00:00+02' + to_seconds(i % 86400) FROM range({N_STR}) t(i)")


def test_fetchall_decimal128(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(benchmark, con, f"SELECT ((i * 1.5)::DECIMAL(28, 6)) FROM range({N_ROW}) t(i)")


def test_fetchall_blob(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    _bench_fetchall(benchmark, con, f"SELECT ('blob_value_' || i)::BLOB FROM range({N_STR}) t(i)")


def test_fetchall_mixed_wide(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    # heterogeneous row: per-cell type dispatch in the Fetchone loop (distinct branch/cache profile from the
    # homogeneous single-type columns above)
    query = (
        "SELECT (i::HUGEINT * 1000000000000) AS h, gen_random_uuid() AS u, "
        f"((i * 1.5)::DECIMAL(28, 6)) AS d, ('string_' || i) AS s FROM range({N_STR}) t(i)"
    )
    _bench_fetchall(benchmark, con, query)


def test_fetchmany_batched(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    query = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N_STR}) t(i)"

    def run() -> None:
        rel = con.execute(query)
        while True:
            rows = rel.fetchmany(10_000)
            if not rows:
                break

    benchmark(run)

"""Shared helpers for filter pushdown tests.

Used by both ``test_filter_pushdown.py`` (pyarrow) and
``test_polars_filter_pushdown.py``. The leading underscore prevents pytest
from collecting this module as a test file.

The factories and parametrization data make the same set of comparison
correctness assertions reusable across every Arrow-shaped input the
replacement scan recognises (pyarrow Table, pyarrow Dataset, pyarrow-backed
pandas, polars LazyFrame, polars DataFrame).
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from conftest import PANDAS_GE_3

pa_ds = pytest.importorskip("pyarrow.dataset")


# ===========================================================================
# Conversion factories — pyarrow side
# ===========================================================================


def to_arrow_table(rel):
    return rel.to_arrow_table()


def to_arrow_via_pandas(rel):
    if PANDAS_GE_3:
        return rel.df()
    return rel.df().convert_dtypes(dtype_backend="pyarrow")


def to_arrow_dataset(rel):
    return pa_ds.dataset(rel.to_arrow_table())


# Standard parametrization: every test that doesn't care about the conversion
# path runs against both the table and pandas factories.
ARROW_FACTORIES = [
    pytest.param(to_arrow_table, id="table"),
    pytest.param(to_arrow_via_pandas, id="pandas"),
]

ARROW_FACTORIES_WITH_DATASET = [
    *ARROW_FACTORIES,
    pytest.param(to_arrow_dataset, id="dataset"),
]


# ===========================================================================
# Typed data fixtures
#
# For every type we test the same fixed 4-row layout:
#     row 0: (low,  low,  low)
#     row 1: (mid,  mid,  mid)
#     row 2: (high, mid,  high)
#     row 3: (NULL, NULL, NULL)
#
# That layout makes the expected row counts for every comparison the same
# across all types, which is what lets us parametrize the comparison tests.
# ===========================================================================


class TypedCase(NamedTuple):
    id: str  # pytest id
    sql_type: str
    low: str  # SQL literal for the smallest value
    mid: str  # SQL literal for the middle value (duplicated in col b row 2)
    high: str  # SQL literal for the largest value


COMPARABLE_TYPES: list[TypedCase] = [
    # numeric
    TypedCase("tinyint", "TINYINT", "1", "10", "100"),
    TypedCase("smallint", "SMALLINT", "1", "10", "100"),
    TypedCase("integer", "INTEGER", "1", "10", "100"),
    TypedCase("bigint", "BIGINT", "1", "10", "100"),
    TypedCase("utinyint", "UTINYINT", "1", "10", "100"),
    TypedCase("usmallint", "USMALLINT", "1", "10", "100"),
    TypedCase("uinteger", "UINTEGER", "1", "10", "100"),
    TypedCase("ubigint", "UBIGINT", "1", "10", "100"),
    TypedCase("hugeint", "HUGEINT", "1", "10", "100"),
    TypedCase("float", "FLOAT", "1.0", "10.0", "100.0"),
    TypedCase("double", "DOUBLE", "1.0", "10.0", "100.0"),
    TypedCase("decimal_4_1", "DECIMAL(4,1)", "1.0", "10.0", "100.0"),
    TypedCase("decimal_9_1", "DECIMAL(9,1)", "1.0", "10.0", "100.0"),
    TypedCase("decimal_18_4", "DECIMAL(18,4)", "1.0", "10.0", "100.0"),
    TypedCase("decimal_30_12", "DECIMAL(30,12)", "1.0", "10.0", "100.0"),
    # string / blob
    TypedCase("varchar", "VARCHAR", "'1'", "'10'", "'100'"),
    TypedCase("blob", "BLOB", r"'\x01'", r"'\x02'", r"'\x03'"),
    # temporal
    TypedCase("date", "DATE", "'2000-01-01'", "'2000-10-01'", "'2010-01-01'"),
    TypedCase("time", "TIME", "'00:01:00'", "'00:10:00'", "'01:00:00'"),
    TypedCase("timestamp", "TIMESTAMP", "'2008-01-01 00:00:01'", "'2010-01-01 10:00:01'", "'2020-03-01 10:00:01'"),
    TypedCase("timestamptz", "TIMESTAMPTZ", "'2008-01-01 00:00:01'", "'2010-01-01 10:00:01'", "'2020-03-01 10:00:01'"),
]


def make_typed_table(con, factory, case: TypedCase) -> object:
    """Create the standard table for `case`, convert via `factory`, and register it as ``arrow_table``."""
    name = f"_t_{case.id}"
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(f"CREATE TABLE {name} (a {case.sql_type}, b {case.sql_type}, c {case.sql_type})")
    con.execute(
        f"""INSERT INTO {name} VALUES
            ({case.low},  {case.low},  {case.low}),
            ({case.mid},  {case.mid},  {case.mid}),
            ({case.high}, {case.mid},  {case.high}),
            (NULL, NULL, NULL)"""
    )
    arrow_table = factory(con.table(name))
    con.register("arrow_table", arrow_table)
    return arrow_table


def count(con, predicate: str) -> int:
    return con.execute(f"SELECT count(*) FROM arrow_table WHERE {predicate}").fetchone()[0]


# ===========================================================================
# Predicate templates parametrized in `test_comparisons`
# ===========================================================================


# (predicate template, expected row count). Templates reference {low}, {mid}, {high}.
COMPARISON_CASES = [
    pytest.param("a = {low}", 1, id="eq"),
    pytest.param("a != {low}", 2, id="ne"),
    pytest.param("a > {low}", 2, id="gt"),
    pytest.param("a >= {mid}", 2, id="ge"),
    pytest.param("a < {mid}", 1, id="lt"),
    pytest.param("a <= {mid}", 2, id="le"),
    pytest.param("a IS NULL", 1, id="is_null"),
    pytest.param("a IS NOT NULL", 3, id="is_not_null"),
    pytest.param("a = {mid} AND b = {low}", 0, id="and_empty"),
    pytest.param("a = {high} AND b = {mid} AND c = {high}", 1, id="and_match"),
    pytest.param("a = {high} OR b = {low}", 2, id="or"),
]


# ===========================================================================
# Plan-inspection helpers
# ===========================================================================


def arrow_scan_block(plan: str) -> str | None:
    """Return the ARROW_SCAN box (top border to bottom border) from an EXPLAIN plan.

    Works uniformly for pyarrow Tables, pyarrow Datasets, pl.LazyFrame, and
    pl.DataFrame inputs — they all bind through ``arrow_scan`` in DuckDB and
    render as ``ARROW_SCAN`` in the plan.
    """
    lines = plan.splitlines()
    scan_idx = next((i for i, line in enumerate(lines) if "ARROW_SCAN" in line), None)
    if scan_idx is None:
        return None
    top = scan_idx
    while top > 0 and "┌" not in lines[top]:
        top -= 1
    bot = scan_idx
    while bot < len(lines) and "└" not in lines[bot]:
        bot += 1
    return "\n".join(lines[top : bot + 1])


def was_pushed(con, query: str) -> bool:
    """True if EXPLAIN of `query` shows a ``Filters:`` line in the ARROW_SCAN block."""
    plan = con.execute(f"EXPLAIN {query}").fetchone()[1]
    block = arrow_scan_block(plan)
    return block is not None and "Filters:" in block

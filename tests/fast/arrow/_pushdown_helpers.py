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


def to_polars_lazyframe(rel):
    return rel.pl().lazy()


def to_polars_dataframe(rel):
    return rel.pl()


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
    """Return the Arrow scan box (top border to bottom border) from an EXPLAIN plan.

    Works uniformly for every Arrow input: they all bind through ``python_arrow_scan`` in DuckDB and render as a
    box titled ``Python Arrow Scan`` with ``Function: PYTHON_ARROW_SCAN``.

    Finds the box by its left and right corners nearest to the title or function line,
    then returns only that column range for each line. This excludes neighboring
    boxes in plans with side-by-side operators.
    """
    lines = plan.splitlines()
    scan_idx = None
    for i, line in enumerate(lines):
        if "Arrow Scan" in line or "ARROW_SCAN" in line:
            scan_idx = i
            break
    if scan_idx is None:
        return None

    line = lines[scan_idx]
    scan_pos = line.find("Arrow Scan") if "Arrow Scan" in line else line.find("ARROW_SCAN")

    left_col = None
    for i in range(scan_pos - 1, -1, -1):
        if i < len(line) and line[i] in ("┌", "╭"):
            left_col = i
            break
    if left_col is None:
        return None

    right_col = None
    for i in range(scan_pos + 1, len(line)):
        if line[i] in ("┐", "╮"):
            right_col = i
            break
    if right_col is None:
        return None

    bot = scan_idx
    while bot < len(lines) and left_col < len(lines[bot]) and lines[bot][left_col] not in ("└", "╰"):
        bot += 1

    result = [lines[i][left_col : right_col + 1] for i in range(scan_idx, bot + 1) if i < len(lines)]
    return "\n".join(result)


ARROW_SCAN_FUNCTIONS = ("PYTHON_ARROW_SCAN",)


def _arrow_scans_below_filters(nodes, nearest_filter=None):
    """Return (scan, nearest FILTER ancestor or None) for every Arrow scan node in a JSON plan tree."""
    result = []
    for node in nodes:
        node_name = node.get("name") or node.get("type")
        filter_above = node if node_name == "FILTER" else nearest_filter

        function = node.get("extra_info", {}).get("Function", "")
        if function in ARROW_SCAN_FUNCTIONS or node_name in ARROW_SCAN_FUNCTIONS:
            result.append((node, nearest_filter))

        result.extend(_arrow_scans_below_filters(node.get("children", []), filter_above))
    return result


def _find_arrow_scan_nodes(nodes, parent_is_filter=False):
    """Recursively find all Arrow scan nodes in JSON plan tree.

    Every Arrow input is read by PYTHON_ARROW_SCAN. Returns list of (node, has_filter_ancestor) tuples.
    """
    return [
        (scan, parent_is_filter or filter_above is not None) for scan, filter_above in _arrow_scans_below_filters(nodes)
    ]


def was_pushed(con, query: str) -> bool:
    """True if EXPLAIN of `query` shows filters pushed to any Arrow scan.

    A filter is considered pushed if it appears in any Arrow scan's Filters field.
    Filters can be a string or a list; empty strings or empty lists mean no filter.
    """
    import json

    result = con.execute(f"EXPLAIN (FORMAT JSON) {query}").fetchone()
    plan = json.loads(result[1])

    scans = _find_arrow_scan_nodes(plan)
    for node, _ in scans:
        filters = node.get("extra_info", {}).get("Filters")
        if filters:
            if isinstance(filters, str):
                if filters.strip():
                    return True
            elif isinstance(filters, list) and len(filters) > 0:
                return True
    return False


def assert_fully_pushed(con, query: str, expected_count: int) -> None:
    """Assert that EXPLAIN ANALYZE of `query` shows its filter inside the only Arrow scan, with nothing above it.

    The scan must emit exactly `expected_count` rows, which shows the pushed filter ran inside the scan. Only a
    source that accepts filters, a pyarrow Dataset or a polars LazyFrame, takes them into its scan.
    """
    import json

    plan = json.loads(con.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}").fetchone()[1])
    scans = _find_arrow_scan_nodes(plan["operator"])
    assert len(scans) == 1
    scan, has_filter_above = scans[0]
    assert scan["extra_info"].get("Filters"), "filter was not pushed into the Arrow scan"
    assert not has_filter_above, "a FILTER operator stayed above the Arrow scan"
    assert scan["intermediate_rows"] == expected_count, "the Arrow scan emitted rows the pushed filter rejects"


def physical_plan(explain_rows):
    """Return the physical plan from the rows of EXPLAIN (FORMAT JSON), whatever explain_output is set to."""
    import json

    plans = dict(explain_rows)
    assert "physical_plan" in plans, f"EXPLAIN returned no physical plan, only {sorted(plans)}"
    return json.loads(plans["physical_plan"])


def explain_physical_plan(con, query: str):
    """Return the physical plan of `query` from EXPLAIN (FORMAT JSON), which does not run the query."""
    return physical_plan(con.execute(f"EXPLAIN (FORMAT JSON) {query}").fetchall())


def _assert_scan_keeps_filter_above(scans) -> dict:
    assert len(scans) == 1, f"expected one Arrow scan, found {len(scans)}"
    scan, filter_above = scans[0]
    assert not scan["extra_info"].get("Filters"), "a filter was pushed into the Arrow scan"
    assert scan["extra_info"].get("Projections"), "the Arrow scan shows no projection"
    assert filter_above is not None, "no FILTER operator above the Arrow scan"
    return filter_above


def assert_plan_keeps_filter_above(plan) -> None:
    """Assert that a physical plan reads its only Arrow scan with a projection and filters above it.

    A source that refuses filters, such as a pyarrow Table, a RecordBatchReader, a Scanner or a stream, gets the
    columns a query reads picked inside the scan, while its filter stays in a FILTER operator above the scan.
    """
    _assert_scan_keeps_filter_above(_arrow_scans_below_filters(plan))


def assert_kept_above(con, query: str, expected_count: int) -> None:
    """Assert that EXPLAIN ANALYZE of `query` keeps its filter above the only Arrow scan.

    The scan shows a projection and no filter, and the nearest FILTER above it emits exactly `expected_count`
    rows. A single use stream is consumed by the analysis, so callers hand it a fresh one.
    """
    import json

    plan = json.loads(con.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}").fetchone()[1])
    filter_above = _assert_scan_keeps_filter_above(_arrow_scans_below_filters(plan["operator"]))
    assert filter_above["intermediate_rows"] == expected_count, "the FILTER above the Arrow scan emitted other rows"


# ===========================================================================
# Plan, release and memory utilities shared by the Arrow scan tests
# ===========================================================================


def scan_of(con, query: str):
    """The only Arrow scan of `query`'s physical plan, and whether a FILTER sits above it."""
    [(scan, filter_above)] = _find_arrow_scan_nodes(explain_physical_plan(con, query))
    return scan, filter_above


def assert_query_keeps_filter_above(con, query: str) -> None:
    """The only Arrow scan of `query` takes no filter, and a FILTER operator above it applies the filter."""
    assert_plan_keeps_filter_above(explain_physical_plan(con, query))


def eventually(check, deadline: float = 30.0) -> bool:
    """Whether `check()` comes true within `deadline` seconds, asked again a millisecond apart.

    A query's error can reach Python while one of its scan threads still winds down holding what it took, waiting
    for the GIL to release it, so a release is not checked at a single instant.
    """
    import gc
    import time

    end = time.monotonic() + deadline
    while True:
        gc.collect()
        if check():
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.001)


def peak_arrow_bytes_while(run):
    """Run `run` while sampling Arrow's allocator; return its result, the peak growth and the samples it spanned."""
    import threading

    import pyarrow as pa

    baseline = pa.total_allocated_bytes()
    peak = baseline
    samples = 0
    sampling = threading.Event()
    done = threading.Event()

    def sample():
        nonlocal peak, samples
        while True:
            peak = max(peak, pa.total_allocated_bytes())
            samples += 1
            sampling.set()
            if done.wait(0.001):
                return

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        assert sampling.wait(10), "the allocator sampler never started"
        first = samples
        result = run()
        during = samples - first
    finally:
        done.set()
        sampler.join(timeout=10)
    return result, peak - baseline, during

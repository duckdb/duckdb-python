"""NaN comparison filters with NULL handling.

DuckDB treats NaN as the greatest float value and a comparison with NULL
returns NULL (which drops the row). Every Arrow scan must match this
behavior exactly. A NaN constant is never offered to a source, so DuckDB
applies a filter holding one above the scan; a pyarrow Table, and the pandas
and polars frames converted to one, refuse every filter.
"""

from __future__ import annotations

import json
import math
import re
import struct

import pytest
from _pushdown_helpers import (
    ARROW_SCAN_FUNCTIONS,
    _find_arrow_scan_nodes,
    arrow_scan_block,
    assert_query_keeps_filter_above,
    scan_of,
    to_arrow_dataset,
    to_arrow_table,
    to_polars_dataframe,
    to_polars_lazyframe,
    was_pushed,
)

import duckdb

try:
    import pyarrow as pa

    import pandas as pd
except ImportError:
    pass


VALUES_WITH_NULL = [1.0, -1.0, 0.0, float("inf"), float("-inf"), float("nan"), None]


def build_minus_nan():
    """Build a NaN with sign bit set (IEEE 754 -NaN)."""
    return struct.unpack("<d", struct.pack("<Q", 0xFFF8000000000000))[0]


def to_arrow_via_pandas_arrowdtype(rel):
    """Pandas with ArrowDtype keeps NaN and NULL distinct and reaches the Arrow scan."""
    return rel.to_arrow_table().to_pandas(types_mapper=pd.ArrowDtype)


def to_arrow_table_dict_encoded(rel):
    """Dictionary-encoded float column."""
    arrow_table = rel.to_arrow_table()
    arrays = []
    fields = []
    for i, field in enumerate(arrow_table.schema):
        arr = arrow_table.column(i)
        if pa.types.is_floating(field.type):
            arr = arr.dictionary_encode()
            assert pa.types.is_dictionary(arr.type), f"Expected dictionary type, got {arr.type}"
            fields.append(pa.field(field.name, arr.type))
        else:
            fields.append(field)
        arrays.append(arr)
    return pa.table(arrays, schema=pa.schema(fields))


def _has_pandas():
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False
    else:
        return True


def _has_polars():
    try:
        import polars  # noqa: F401
    except ImportError:
        return False
    else:
        return True


ALL_FACTORIES = [
    pytest.param(to_arrow_table, id="table"),
    pytest.param(
        to_arrow_via_pandas_arrowdtype,
        id="pandas",
        marks=pytest.mark.skipif(not _has_pandas(), reason="pandas not available"),
    ),
    pytest.param(to_arrow_dataset, id="dataset"),
    pytest.param(
        to_polars_lazyframe, id="lazyframe", marks=pytest.mark.skipif(not _has_polars(), reason="polars not available")
    ),
    pytest.param(
        to_polars_dataframe, id="dataframe", marks=pytest.mark.skipif(not _has_polars(), reason="polars not available")
    ),
    pytest.param(to_arrow_table_dict_encoded, id="dict"),
]


# The factories whose source takes the filters it can evaluate exactly into its own scan.
ACCEPTING_FACTORIES = (to_arrow_dataset, to_polars_lazyframe)


def _make_factory_params_with_xfail(xfail_on_dict):
    """Generate factory params with optional xfail mark on dict factory.

    Args:
        xfail_on_dict: If True, dict factory gets xfail mark.

    Returns:
        List of pytest.param for factory parametrization.
    """
    result = []
    for factory_param in ALL_FACTORIES:
        factory = factory_param.values[0]
        if factory.__name__ == "to_arrow_table_dict_encoded" and xfail_on_dict:
            marks = [
                *factory_param.marks,
                pytest.mark.xfail(
                    strict=True, raises=duckdb.Error, reason="follow up: pyarrow is_nan has no dictionary kernel"
                ),
            ]
            result.append(pytest.param(factory, marks=marks, id=factory_param.id))
        else:
            result.append(factory_param)
    return result


def _make_combined_params_coltype_op(coltype_op_tuples):
    """Generate combined (coltype, op, factory) params with xfail on dict for specific ops.

    Args:
        coltype_op_tuples: List of (coltype, op, needs_xfail_on_dict) tuples.

    Returns:
        List of pytest.param for combined parametrization.
    """
    result = []
    for coltype, op, needs_xfail in coltype_op_tuples:
        for factory_param in ALL_FACTORIES:
            factory = factory_param.values[0]
            param_id = f"{coltype}-{op}-{factory_param.id}"
            if factory.__name__ == "to_arrow_table_dict_encoded" and needs_xfail:
                marks = [
                    *factory_param.marks,
                    pytest.mark.xfail(
                        strict=True, raises=duckdb.Error, reason="follow up: pyarrow is_nan has no dictionary kernel"
                    ),
                ]
                result.append(pytest.param(coltype, op, factory, marks=marks, id=param_id))
            else:
                result.append(pytest.param(coltype, op, factory, marks=factory_param.marks, id=param_id))
    return result


def _make_combined_params_coltype_op_const(coltype_op_const_tuples):
    """Generate combined (coltype, op, const, factory) params with xfail on dict for specific ops.

    Args:
        coltype_op_const_tuples: List of (coltype, op, const, needs_xfail_on_dict) tuples.

    Returns:
        List of pytest.param for combined parametrization.
    """
    result = []
    for coltype, op, const, needs_xfail in coltype_op_const_tuples:
        for factory_param in ALL_FACTORIES:
            factory = factory_param.values[0]
            const_short = const.split("::")[0].replace("'", "")
            param_id = f"{coltype}-{op}-{const_short}-{factory_param.id}"
            if factory.__name__ == "to_arrow_table_dict_encoded" and needs_xfail:
                marks = [
                    *factory_param.marks,
                    pytest.mark.xfail(
                        strict=True, raises=duckdb.Error, reason="follow up: pyarrow is_nan has no dictionary kernel"
                    ),
                ]
                result.append(pytest.param(coltype, op, const, factory, marks=marks, id=param_id))
            else:
                result.append(pytest.param(coltype, op, const, factory, marks=factory_param.marks, id=param_id))
    return result


def normalize_result(rows):
    """NaN != NaN, so convert to string for comparison."""
    return sorted(
        (str(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else repr(v),) for (v,) in rows
    )


def _filters_text(filters):
    """The Filters of a scan as one text without identifier quotes, so `"b" < 2` reads as `b < 2`."""
    return (str(filters) if not isinstance(filters, str) else filters).replace('"', "")


def assert_filter_pushed_to_scan(con, query: str, expected_filter: str):
    """Verify filter is in an Arrow scan's Filters and no FILTER operator is above it."""
    result = con.execute(f"EXPLAIN (FORMAT JSON) {query}").fetchone()
    plan = json.loads(result[1])

    scans = _find_arrow_scan_nodes(plan)
    assert len(scans) > 0, "No Arrow scan in plan"

    found = False
    for node, has_filter_ancestor in scans:
        filters = node.get("extra_info", {}).get("Filters")
        if filters and expected_filter in _filters_text(filters):
            assert not has_filter_ancestor, "FILTER operator above the Arrow scan in plan"
            found = True
            break

    assert found, f"Expected filter '{expected_filter}' not found in any Arrow scan"


def _find_arrow_scan_in_analyzed_plan(nodes):
    """Find Arrow scan node in analyzed plan's operator tree."""
    for node in nodes:
        extra_info = node.get("extra_info", {})
        if extra_info.get("Function") in ARROW_SCAN_FUNCTIONS:
            return node
        children = node.get("children", [])
        if children:
            result = _find_arrow_scan_in_analyzed_plan(children)
            if result:
                return result
    return None


def assert_in_filter_pushed(con, query: str, predicate: str):
    """Verify IN filter was pushed by checking Arrow Scan emits only matching rows."""
    native_matching = con.execute(f"SELECT COUNT(*) FROM native_t WHERE {predicate}").fetchone()[0]

    result = con.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}").fetchone()
    plan = json.loads(result[1])

    operator_list = plan.get("operator", [])
    node = _find_arrow_scan_in_analyzed_plan(operator_list)
    assert node is not None, "No Arrow scan in EXPLAIN ANALYZE plan"

    emitted_rows = node.get("intermediate_rows", 0)

    assert emitted_rows == native_matching, (
        f"IN filter was pushed: scan emitted {emitted_rows} rows, native has {native_matching}"
    )


def assert_in_filter_not_pushed(con, query: str):
    """Verify IN filter was NOT pushed by checking Arrow Scan emits all rows."""
    total_rows = con.execute("SELECT COUNT(*) FROM native_t").fetchone()[0]

    result = con.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}").fetchone()
    plan = json.loads(result[1])

    operator_list = plan.get("operator", [])
    node = _find_arrow_scan_in_analyzed_plan(operator_list)
    assert node is not None, "No Arrow scan in EXPLAIN ANALYZE plan"

    emitted_rows = node.get("intermediate_rows", 0)

    assert emitted_rows == total_rows, (
        f"IN filter was incorrectly pushed: scan emitted {emitted_rows} rows, native has {total_rows}"
    )


def assert_matches_native(con, predicate: str):
    native_result = normalize_result(con.execute(f"SELECT b FROM native_t WHERE {predicate}").fetchall())
    arrow_result = normalize_result(con.execute(f"SELECT b FROM arrow_t WHERE {predicate}").fetchall())
    assert arrow_result == native_result, f"Arrow scan result differs from native for predicate: {predicate}"


class TestNaNWithNullRows:
    """NaN comparison filter pushdown with NULL rows in the data."""

    @pytest.fixture(autouse=True)
    def _setup(self, duckdb_cursor):
        self.con = duckdb_cursor

    def _make_tables(self, coltype: str, values: list, factory):
        """Create native and arrow tables with the given values."""
        self.con.execute("DROP TABLE IF EXISTS native_t")
        self.con.execute(f"CREATE TABLE native_t (b {coltype})")
        self.con.executemany("INSERT INTO native_t VALUES (?)", [[v] for v in values])
        self.con.register("arrow_t", factory(self.con.table("native_t")))

    def _make_large_tables(self, coltype: str, total_rows: int, special_values: dict, factory):
        """Create large native and arrow tables efficiently with one SQL statement.

        Args:
            coltype: Column type (DOUBLE or FLOAT)
            total_rows: Total number of rows
            special_values: Dict mapping row index to value (e.g., {100: float('nan'), 200: 1.0})
            factory: Factory function to convert DuckDB table to arrow format
        """
        self.con.execute("DROP TABLE IF EXISTS native_t")
        # Build CASE expression for special values
        case_expr = "CASE "
        for idx, val in special_values.items():
            if val is None:
                case_expr += f"WHEN i = {idx} THEN NULL "
            elif isinstance(val, float):
                if math.isnan(val):
                    # Check sign bit for -nan
                    if struct.pack("<d", val)[7] & 0x80:
                        # Negative NaN: use -('nan'::DOUBLE)
                        case_expr += f"WHEN i = {idx} THEN -('nan'::{coltype}) "
                    else:
                        case_expr += f"WHEN i = {idx} THEN 'nan'::{coltype} "
                elif val == 0.0:
                    # Check sign bit for -0.0
                    if struct.pack("<d", val)[7] & 0x80:
                        case_expr += f"WHEN i = {idx} THEN -(0.0::{coltype}) "
                    else:
                        case_expr += f"WHEN i = {idx} THEN 0.0::{coltype} "
                else:
                    case_expr += f"WHEN i = {idx} THEN {val}::{coltype} "
        case_expr += f"ELSE i::{coltype} END"

        self.con.execute(f"CREATE TABLE native_t AS SELECT {case_expr} AS b FROM range({total_rows}) t(i)")
        self.con.register("arrow_t", factory(self.con.table("native_t")))

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_le_nan_matching_type(self, coltype, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        predicate = f"b <= 'nan'::{coltype}"
        assert_matches_native(self.con, predicate)
        assert_query_keeps_filter_above(self.con, f"SELECT b FROM arrow_t WHERE {predicate}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_le_nan_is_not_distinct_from_true(self, coltype, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        predicate = f"b <= 'nan'::{coltype} IS NOT DISTINCT FROM true"
        assert_matches_native(self.con, predicate)
        assert_query_keeps_filter_above(self.con, f"SELECT b FROM arrow_t WHERE {predicate}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_le_nan_double_col_float_const(self, factory):
        self._make_tables("DOUBLE", [1.0, float("nan"), None], factory)
        predicate = "b <= 'nan'::FLOAT"
        assert_matches_native(self.con, predicate)
        assert_query_keeps_filter_above(self.con, f"SELECT b FROM arrow_t WHERE {predicate}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_le_nan_float_col_double_const(self, factory):
        self._make_tables("FLOAT", [1.0, float("nan"), None], factory)
        predicate = "b <= 'nan'::DOUBLE"
        assert_matches_native(self.con, predicate)

    @pytest.mark.parametrize(
        ("coltype", "op", "factory"),
        _make_combined_params_coltype_op(
            [
                ("DOUBLE", "<", False),
                ("DOUBLE", "=", False),
                ("DOUBLE", "<>", False),
                ("DOUBLE", ">=", False),
                ("DOUBLE", ">", False),
                ("FLOAT", "<", False),
                ("FLOAT", "=", False),
                ("FLOAT", "<>", False),
                ("FLOAT", ">=", False),
                ("FLOAT", ">", False),
            ]
        ),
    )
    def test_other_nan_comparisons(self, coltype, op, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        assert_matches_native(self.con, f"b {op} 'nan'::{coltype}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_in_list_with_nan(self, coltype, factory):
        """IN list with NaN: no source is offered it, so the scan emits every row."""
        # Create large table with 2000 rows, only 3 matching.
        self._make_large_tables(coltype, 2000, {100: float("nan"), 200: 1.0}, factory)
        predicate = f"b IN ('nan'::{coltype}, 1.0)"
        assert_matches_native(self.con, predicate)
        assert_in_filter_not_pushed(self.con, f"SELECT b FROM arrow_t WHERE {predicate}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_in_list_with_zero(self, coltype, factory):
        """IN list with zero: a Dataset and a LazyFrame take it into their scan, a Table refuses it."""
        # Create large table with 2000 rows, only 4 matching.
        self._make_large_tables(coltype, 2000, {100: 0.0, 200: 1.0}, factory)
        predicate = f"b IN (0.0::{coltype}, 1.0)"
        assert_matches_native(self.con, predicate)
        if factory in ACCEPTING_FACTORIES:
            assert_in_filter_pushed(self.con, f"SELECT b FROM arrow_t WHERE {predicate}", predicate)
        else:
            assert_in_filter_not_pushed(self.con, f"SELECT b FROM arrow_t WHERE {predicate}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_in_list_without_nan_or_zero(self, coltype, factory):
        """IN list without NaN or zero: a Dataset and a LazyFrame take it into their scan, a Table refuses it."""
        # Create large table with 2000 rows, only 6 matching.
        self._make_large_tables(coltype, 2000, {100: 1.0, 200: 2.0, 300: 3.0}, factory)
        predicate = f"b IN (1.0::{coltype}, 2.0, 3.0)"
        assert_matches_native(self.con, predicate)
        if factory in ACCEPTING_FACTORIES:
            assert_in_filter_pushed(self.con, f"SELECT b FROM arrow_t WHERE {predicate}", predicate)
        else:
            assert_in_filter_not_pushed(self.con, f"SELECT b FROM arrow_t WHERE {predicate}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_not_in_list_with_nan(self, coltype, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        assert_matches_native(self.con, f"b NOT IN ('nan'::{coltype}, 1.0)")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_le_nan_or_eq(self, coltype, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        assert_matches_native(self.con, f"b <= 'nan'::{coltype} OR b = 1.0")

    @pytest.mark.parametrize("factory", _make_factory_params_with_xfail(xfail_on_dict=False))
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_le_nan_and_gt(self, coltype, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        assert_matches_native(self.con, f"b <= 'nan'::{coltype} AND b > 0")

    @pytest.mark.parametrize("factory", _make_factory_params_with_xfail(xfail_on_dict=False))
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_between_zero_and_nan(self, coltype, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        assert_matches_native(self.con, f"b BETWEEN 0 AND 'nan'::{coltype}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_not_le_nan(self, coltype, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        assert_matches_native(self.con, f"NOT (b <= 'nan'::{coltype})")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_le_nan_or_is_null(self, coltype, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        assert_matches_native(self.con, f"b <= 'nan'::{coltype} OR b IS NULL")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_gt_nan_or_is_null(self, coltype, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        assert_matches_native(self.con, f"b > 'nan'::{coltype} OR b IS NULL")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_in_list_with_nan_over_minus_nan(self, coltype, factory):
        """IN list with NaN constant over -nan data: no source is offered it, so the scan emits every row."""
        minus_nan = build_minus_nan()
        # Few of the 2000 rows match, so whether the IN list reached the scan shows in its emitted row count.
        self._make_large_tables(coltype, 2000, {100: minus_nan, 200: 1.0}, factory)
        predicate = f"b IN ('nan'::{coltype}, 1.0)"
        assert_matches_native(self.con, predicate)
        assert_in_filter_not_pushed(self.con, f"SELECT b FROM arrow_t WHERE {predicate}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_in_list_with_zero_over_minus_zero(self, coltype, factory):
        """IN list with 0.0 constant over -0.0 data: a Dataset and a LazyFrame take it, a Table refuses it."""
        # Few of the 2000 rows match, so whether the IN list reached the scan shows in its emitted row count.
        self._make_large_tables(coltype, 2000, {100: -0.0, 200: 1.0}, factory)
        predicate = f"b IN (0.0::{coltype}, 1.0)"
        assert_matches_native(self.con, predicate)
        if factory in ACCEPTING_FACTORIES:
            assert_in_filter_pushed(self.con, f"SELECT b FROM arrow_t WHERE {predicate}", predicate)
        else:
            assert_in_filter_not_pushed(self.con, f"SELECT b FROM arrow_t WHERE {predicate}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("coltype", ["DOUBLE", "FLOAT"])
    def test_in_list_only_nan_and_zero(self, coltype, factory):
        """IN list with only NaN and zero: no source is offered it, so the scan emits every row."""
        # Few of the 2000 rows match, so whether the IN list reached the scan shows in its emitted row count.
        self._make_large_tables(coltype, 2000, {100: float("nan"), 200: 0.0}, factory)
        predicate = f"b IN ('nan'::{coltype}, 0.0)"
        assert_matches_native(self.con, predicate)
        assert_in_filter_not_pushed(self.con, f"SELECT b FROM arrow_t WHERE {predicate}")

    @pytest.mark.parametrize(
        ("coltype", "op", "const", "factory"),
        _make_combined_params_coltype_op_const(
            [
                ("DOUBLE", "<", "0::DOUBLE", False),
                ("DOUBLE", "<=", "0::DOUBLE", False),
                ("DOUBLE", "=", "0::DOUBLE", False),
                ("DOUBLE", "<>", "0::DOUBLE", False),
                ("DOUBLE", ">=", "0::DOUBLE", False),
                ("DOUBLE", ">", "0::DOUBLE", False),
                ("DOUBLE", "<", "'inf'::DOUBLE", False),
                ("DOUBLE", "<=", "'inf'::DOUBLE", False),
                ("DOUBLE", "=", "'inf'::DOUBLE", False),
                ("DOUBLE", "<>", "'inf'::DOUBLE", False),
                ("DOUBLE", ">=", "'inf'::DOUBLE", False),
                ("DOUBLE", ">", "'inf'::DOUBLE", False),
                ("DOUBLE", "<", "'-inf'::DOUBLE", False),
                ("DOUBLE", "<=", "'-inf'::DOUBLE", False),
                ("DOUBLE", "=", "'-inf'::DOUBLE", False),
                ("DOUBLE", "<>", "'-inf'::DOUBLE", False),
                ("DOUBLE", ">=", "'-inf'::DOUBLE", False),
                ("DOUBLE", ">", "'-inf'::DOUBLE", False),
                ("FLOAT", "<", "0::FLOAT", False),
                ("FLOAT", "<=", "0::FLOAT", False),
                ("FLOAT", "=", "0::FLOAT", False),
                ("FLOAT", "<>", "0::FLOAT", False),
                ("FLOAT", ">=", "0::FLOAT", False),
                ("FLOAT", ">", "0::FLOAT", False),
                ("FLOAT", "<", "'inf'::FLOAT", False),
                ("FLOAT", "<=", "'inf'::FLOAT", False),
                ("FLOAT", "=", "'inf'::FLOAT", False),
                ("FLOAT", "<>", "'inf'::FLOAT", False),
                ("FLOAT", ">=", "'inf'::FLOAT", False),
                ("FLOAT", ">", "'inf'::FLOAT", False),
                ("FLOAT", "<", "'-inf'::FLOAT", False),
                ("FLOAT", "<=", "'-inf'::FLOAT", False),
                ("FLOAT", "=", "'-inf'::FLOAT", False),
                ("FLOAT", "<>", "'-inf'::FLOAT", False),
                ("FLOAT", ">=", "'-inf'::FLOAT", False),
                ("FLOAT", ">", "'-inf'::FLOAT", False),
            ]
        ),
    )
    def test_finite_constant_with_null(self, coltype, op, const, factory):
        self._make_tables(coltype, VALUES_WITH_NULL, factory)
        assert_matches_native(self.con, f"b {op} {const}")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_filter_not_pushed_negative_check(self, factory):
        """Negative test: assert_filter_pushed_to_scan correctly rejects a filter that is not fully pushed.

        A Dataset or LazyFrame takes b <= 1.0 and leaves abs(b) = 1 above its scan; a Table takes neither.
        """
        self._make_tables("DOUBLE", [1.0, float("nan"), None], factory)
        predicate = "b <= 1.0 AND abs(b) = 1"
        refusal = "FILTER operator above the Arrow scan" if factory in ACCEPTING_FACTORIES else "not found"
        with pytest.raises(AssertionError, match=refusal):
            assert_filter_pushed_to_scan(self.con, f"SELECT b FROM arrow_t WHERE {predicate}", "b <= 1.0")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_helper_handles_wide_projection(self, factory):
        """arrow_scan_block finds Filters line even with wide projection beside the scan."""
        self.con.execute("DROP TABLE IF EXISTS wide_t")
        self.con.execute("""
            CREATE TABLE wide_t (
                b DOUBLE,
                long_projection_column_1 INT,
                long_projection_column_2 INT,
                long_projection_column_3 INT,
                long_projection_column_4 INT,
                long_projection_column_5 INT,
                long_projection_column_6 INT,
                long_projection_column_7 INT,
                long_projection_column_8 INT
            )
        """)
        self.con.execute(
            "INSERT INTO wide_t VALUES (1.0, 1, 2, 3, 4, 5, 6, 7, 8), (2.5, 9, 10, 11, 12, 13, 14, 15, 16)"
        )
        self.con.register("wide_arrow_t", factory(self.con.table("wide_t")))
        # CROSS JOIN with projection creates a wide plan where Filters may fall outside a naive box extraction.
        query = "SELECT a.* FROM wide_arrow_t a CROSS JOIN range(1) t WHERE b < 2"
        plan = self.con.execute(f"EXPLAIN {query}").fetchone()[1]
        block = arrow_scan_block(plan)
        assert block is not None, "arrow_scan_block failed to extract the Arrow scan box"
        flat = re.sub(r'[│"()\s]+', " ", block)
        if factory in ACCEPTING_FACTORIES:
            assert "Filters: b < 2" in flat, f"Filters line not in block: {block}"
        else:
            assert "Filters:" not in block, f"a Table took a filter into its scan: {block}"
            assert "Projections:" in block, f"Projections line not in block: {block}"

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_helper_excludes_neighbor_box_filters(self, factory):
        """JSON parser excludes filters from neighboring native scan."""
        self.con.execute("SET disabled_optimizers='join_order,build_side_probe_side'")
        try:
            self._make_tables("DOUBLE", [1.0, 2.0], factory)
            self.con.execute("CREATE OR REPLACE TABLE n AS SELECT i FROM range(10) t(i)")
            query = "SELECT * FROM arrow_t a CROSS JOIN n WHERE i < 2"

            result = self.con.execute(f"EXPLAIN (FORMAT JSON) {query}").fetchone()
            plan = json.loads(result[1])
            scans = _find_arrow_scan_nodes(plan)
            assert len(scans) > 0, "No Arrow scan found"

            for node, _ in scans:
                filters = node.get("extra_info", {}).get("Filters")
                if filters:
                    assert not re.search(r"\bi\b", _filters_text(filters)), (
                        f"Native table filter incorrectly on Arrow scan: {filters}"
                    )
        finally:
            self.con.execute("RESET disabled_optimizers")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_scan_beside_a_filtered_native_scan(self, factory):
        """JSON parser correctly finds Arrow scan when neighboring a filtered native scan."""
        self.con.execute("SET disabled_optimizers='join_order,build_side_probe_side'")
        try:
            self._make_tables("DOUBLE", [1.0, float("nan"), None], factory)
            self.con.execute("CREATE OR REPLACE TABLE n AS SELECT i FROM range(10) t(i)")
            query = "SELECT n.i, a.b FROM n CROSS JOIN arrow_t a WHERE n.i < 2 AND b <= 'nan'::DOUBLE"

            # The NaN constant is never offered, so the Arrow scan has no filter and a FILTER above it applies it.
            assert_query_keeps_filter_above(self.con, query)
            assert_matches_native(self.con, "b <= 'nan'::DOUBLE")
        finally:
            self.con.execute("RESET disabled_optimizers")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_scan_beside_a_range_function(self, factory):
        """CROSS JOIN with range function: the Arrow scan is found and the NaN filter stays above it."""
        self._make_tables("DOUBLE", [1.0, float("nan"), None], factory)
        query = "SELECT a.* FROM arrow_t a CROSS JOIN range(1) WHERE b <= 'nan'::DOUBLE"

        # The NaN constant is never offered, so a FILTER above the scan applies it.
        assert_query_keeps_filter_above(self.con, query)

        # Verify results match native
        assert_matches_native(self.con, "b <= 'nan'::DOUBLE")


def test_helpers_read_several_pushed_filters(duckdb_cursor):
    # A Dataset takes the filters it can evaluate into its scan, where a Table refuses them.
    duckdb_cursor.register(
        "several", to_arrow_dataset(duckdb_cursor.sql("SELECT range AS x, range AS y FROM range(1, 4)"))
    )
    query = "SELECT * FROM several WHERE x > 1 AND y < 3"
    assert was_pushed(duckdb_cursor, query)
    assert_filter_pushed_to_scan(duckdb_cursor, query, "x > 1")
    assert_filter_pushed_to_scan(duckdb_cursor, query, "y < 3")


def test_helpers_find_the_filtered_scan_of_two_arrow_scans(duckdb_cursor):
    duckdb_cursor.execute("SET disabled_optimizers = 'join_order,build_side_probe_side'")
    duckdb_cursor.register("first_arrow", pa.table({"x": [1, 2, 3]}))
    duckdb_cursor.register("second_arrow", to_arrow_dataset(duckdb_cursor.sql("SELECT * FROM range(1, 4) t(y)")))
    query = "SELECT * FROM first_arrow CROSS JOIN second_arrow WHERE y < 2"
    assert was_pushed(duckdb_cursor, query)
    assert_filter_pushed_to_scan(duckdb_cursor, query, "y < 2")


# NaN satisfies these, as the greatest value, and a Dataset's row group statistics leave NaN out, so pruning by them
# would drop NaN rows: they stay above the scan.
NAN_MAY_SATISFY = ["f > 1.5", "f >= 2.0", "f <> 2.0", "NOT (f = 1.0)", "f = 1.0 OR f > 2.0", "f NOT IN (1.0)"]
NAN_CANNOT_SATISFY = ["f = 2.0", "f < 2.0", "f <= 2.0", "f IN (1.0, 3.0)", "f IS NULL", "f IS NOT NULL"]


@pytest.mark.parametrize("sql_type", ["DOUBLE", "FLOAT"])
class TestDatasetWithNaNInRowGroups:
    """A parquet Dataset whose row groups of two hold NaN, numbers and NULL gives the engine's rows for every filter."""

    @pytest.fixture(autouse=True)
    def _dataset(self, duckdb_cursor, tmp_path, sql_type):
        pq = pytest.importorskip("pyarrow.parquet")
        ds = pytest.importorskip("pyarrow.dataset")
        float_type = pa.float64() if sql_type == "DOUBLE" else pa.float32()
        table = pa.table({"x": [1, 2, 3, 4, 5], "f": pa.array([float("nan"), 1.0, 2.0, 3.0, None], float_type)})
        pq.write_table(table, tmp_path / "floats.parquet", row_group_size=2)
        duckdb_cursor.register("native_source", table)
        duckdb_cursor.execute("CREATE TABLE native_t AS SELECT * FROM native_source")
        duckdb_cursor.unregister("native_source")
        duckdb_cursor.register("arrow_t", ds.dataset(tmp_path / "floats.parquet"))
        self.con = duckdb_cursor

    def _assert_rows_match_native(self, predicate):
        expected = self.con.execute(f"SELECT x FROM native_t WHERE {predicate} ORDER BY x").fetchall()
        assert self.con.execute(f"SELECT x FROM arrow_t WHERE {predicate} ORDER BY x").fetchall() == expected

    @pytest.mark.parametrize("predicate", NAN_MAY_SATISFY)
    def test_a_filter_nan_may_satisfy_stays_above_the_scan(self, predicate):
        self._assert_rows_match_native(predicate)
        assert_query_keeps_filter_above(self.con, f"SELECT x FROM arrow_t WHERE {predicate}")

    @pytest.mark.parametrize("predicate", NAN_CANNOT_SATISFY)
    def test_a_filter_nan_cannot_satisfy_is_taken_by_the_scan(self, predicate):
        self._assert_rows_match_native(predicate)
        scan, filter_above = scan_of(self.con, f"SELECT x FROM arrow_t WHERE {predicate}")
        assert scan["extra_info"].get("Filters"), predicate
        assert not filter_above, predicate

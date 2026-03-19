# ruff: noqa: F841
import math

import pytest

import duckdb

pl = pytest.importorskip("polars")
pytest.importorskip("pyarrow")


class TestPolarsLazyFrameFilterPushdown:
    """Tests for filter pushdown on LazyFrames.

    All tests use pl.LazyFrame (the target of this change). DuckDB pushes filters and projections into the Polars lazy
    plan before collection, so only surviving rows are ever materialized.
    """

    ##### CONSTANT_COMPARISON: all six comparison operators

    def test_comparison_equal(self):
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        assert duckdb.sql("SELECT * FROM lf WHERE a = 3").fetchall() == [(3,)]

    def test_comparison_not_equal(self):
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        assert duckdb.sql("SELECT * FROM lf WHERE a != 3").fetchall() == [(1,), (2,), (4,), (5,)]

    def test_comparison_less_than(self):
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        assert duckdb.sql("SELECT * FROM lf WHERE a < 3").fetchall() == [(1,), (2,)]

    def test_comparison_less_than_or_equal(self):
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        assert duckdb.sql("SELECT * FROM lf WHERE a <= 3").fetchall() == [(1,), (2,), (3,)]

    def test_comparison_greater_than(self):
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        assert duckdb.sql("SELECT * FROM lf WHERE a > 3").fetchall() == [(4,), (5,)]

    def test_comparison_greater_than_or_equal(self):
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        assert duckdb.sql("SELECT * FROM lf WHERE a >= 3").fetchall() == [(3,), (4,), (5,)]

    def test_string_comparison(self):
        lf = pl.LazyFrame({"name": ["alice", "bob", "charlie"], "val": [1, 2, 3]})
        assert duckdb.sql("SELECT * FROM lf WHERE name = 'bob'").fetchall() == [("bob", 2)]

    ##### NaN comparisons (CONSTANT_COMPARISON with is_nan path)

    def test_nan_equal(self):
        """NaN = NaN is true in DuckDB; pushes is_nan()."""
        lf = pl.LazyFrame({"a": [1.0, float("nan"), 3.0]})
        result = duckdb.sql("SELECT * FROM lf WHERE a = 'NaN'::DOUBLE").fetchall()
        assert len(result) == 1
        assert math.isnan(result[0][0])

    def test_nan_greater_than_or_equal(self):
        """NaN >= NaN is true; pushes is_nan()."""
        lf = pl.LazyFrame({"a": [1.0, float("nan"), 3.0]})
        result = duckdb.sql("SELECT * FROM lf WHERE a >= 'NaN'::DOUBLE").fetchall()
        assert len(result) == 1
        assert math.isnan(result[0][0])

    def test_nan_less_than(self):
        """X < NaN is true for non-NaN values; pushes is_nan().__invert__()."""
        lf = pl.LazyFrame({"a": [1.0, float("nan"), 3.0]})
        result = duckdb.sql("SELECT * FROM lf WHERE a < 'NaN'::DOUBLE").fetchall()
        assert sorted(result) == [(1.0,), (3.0,)]

    def test_nan_not_equal(self):
        """X != NaN is true for non-NaN values; pushes is_nan().__invert__()."""
        lf = pl.LazyFrame({"a": [1.0, float("nan"), 3.0]})
        result = duckdb.sql("SELECT * FROM lf WHERE a != 'NaN'::DOUBLE").fetchall()
        assert sorted(result) == [(1.0,), (3.0,)]

    def test_nan_greater_than(self):
        """X > NaN is always false; pushes lit(false)."""
        lf = pl.LazyFrame({"a": [1.0, float("nan"), 3.0]})
        result = duckdb.sql("SELECT * FROM lf WHERE a > 'NaN'::DOUBLE").fetchall()
        assert result == []

    def test_nan_less_than_or_equal(self):
        """X <= NaN is always true; pushes lit(true)."""
        lf = pl.LazyFrame({"a": [1.0, float("nan"), 3.0]})
        result = duckdb.sql("SELECT * FROM lf WHERE a <= 'NaN'::DOUBLE").fetchall()
        assert len(result) == 3

    ##### IS_NULL / IS_NOT_NULL (triggered via DISTINCT FROM NULL inside OR)

    def test_is_null_filter(self):
        """IS NOT DISTINCT FROM NULL inside an OR pushes IS_NULL as a child of CONJUNCTION_OR."""
        lf = pl.LazyFrame({"a": [1, None, 3, None, 5]})
        result = duckdb.sql("SELECT * FROM lf WHERE a = 1 OR a IS NOT DISTINCT FROM NULL").fetchall()
        values = [row[0] for row in result]
        assert values.count(None) == 2
        assert 1 in values
        assert len(values) == 3

    def test_is_not_null_filter(self):
        """IS DISTINCT FROM NULL inside an OR pushes IS_NOT_NULL as a child of CONJUNCTION_OR."""
        lf = pl.LazyFrame({"a": [1, None, 3, None, 5]})
        result = duckdb.sql("SELECT * FROM lf WHERE a = 1 OR a IS DISTINCT FROM NULL").fetchall()
        assert sorted(result) == [(1,), (3,), (5,)]

    # ── CONJUNCTION_AND ──

    def test_conjunction_and_range(self):
        """BETWEEN on a single column pushes a CONJUNCTION_AND with GTE + LTE children."""
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        result = duckdb.sql("SELECT * FROM lf WHERE a BETWEEN 2 AND 4").fetchall()
        assert result == [(2,), (3,), (4,)]

    def test_conjunction_and_multi_column(self):
        """Filters on two different columns combine via AND in TransformFilter."""
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "x", "y", "x"]})
        result = duckdb.sql("SELECT * FROM lf WHERE a > 2 AND b = 'x'").fetchall()
        assert result == [(3, "x"), (5, "x")]

    ##### CONJUNCTION_OR

    def test_conjunction_or(self):
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        result = duckdb.sql("SELECT * FROM lf WHERE a = 1 OR a = 5").fetchall()
        assert sorted(result) == [(1,), (5,)]

    ##### IN_FILTER

    def test_in_filter(self):
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        result = duckdb.sql("SELECT * FROM lf WHERE a IN (2, 4)").fetchall()
        assert sorted(result) == [(2,), (4,)]

    ##### STRUCT_EXTRACT

    def test_struct_extract(self):
        lf = pl.LazyFrame({"s": [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, {"x": 3, "y": "c"}]})
        result = duckdb.sql("SELECT * FROM lf WHERE s.x > 1").fetchall()
        assert len(result) == 2
        assert all(row[0]["x"] > 1 for row in result)

    ##### OPTIONAL_FILTER

    def test_optional_filter(self):
        """OR filters are wrapped in OPTIONAL_FILTER by DuckDB's optimizer."""
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        result = duckdb.sql("SELECT * FROM lf WHERE a = 1 OR a = 3").fetchall()
        assert sorted(result) == [(1,), (3,)]

    ##### Produce path, no filters

    def test_unfiltered_scan(self):
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = duckdb.sql("SELECT * FROM lf").fetchall()
        assert result == [(1, 4), (2, 5), (3, 6)]

    ##### Produce path, column projection

    def test_column_projection(self):
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = duckdb.sql("SELECT a, c FROM lf").fetchall()
        assert result == [(1, 7), (2, 8), (3, 9)]

    ##### Produce path, cached DataFrame reuse

    def test_cached_dataframe_reuse(self):
        """Repeated unfiltered scans on a registered LazyFrame reuse the cached DataFrame."""
        con = duckdb.connect()
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        con.register("my_lf", lf)
        r1 = con.sql("SELECT * FROM my_lf").fetchall()
        r2 = con.sql("SELECT * FROM my_lf").fetchall()
        assert r1 == r2 == [(1,), (2,), (3,)]

    ##### Produce path, filter + collect (no cache)

    def test_filtered_scan_not_cached(self):
        """Filtered scans collect a new DataFrame each time (not cached)."""
        con = duckdb.connect()
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        con.register("my_lf", lf)
        r1 = con.sql("SELECT * FROM my_lf WHERE a > 3").fetchall()
        r2 = con.sql("SELECT * FROM my_lf WHERE a < 3").fetchall()
        assert sorted(r1) == [(4,), (5,)]
        assert sorted(r2) == [(1,), (2,)]

    ##### Empty result

    def test_empty_result(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = duckdb.sql("SELECT * FROM lf WHERE a > 100").fetchall()
        assert result == []

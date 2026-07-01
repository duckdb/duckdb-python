# ruff: noqa: F841
"""Filter pushdown tests for the arrow scan integration.

What's tested here:

* **Comparison correctness** across every supported type:
  ``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``IS NULL``, ``IS NOT NULL``,
  ``AND``, ``OR``.
* **Optimizer pushdown decisions** — which predicate shapes get pushed into
  the ``ARROW_SCAN`` operator and which the optimizer keeps above. Verified
  by inspecting the EXPLAIN plan, not just row counts.
* **Special filter shapes** — ``IN``, ``LIKE``, ``CAST(ts AS DATE) = …``,
  ``IS DISTINCT FROM NULL`` inside ``OR``, NaN ordering, struct extraction
  (one- and multi-level), optional filter, dynamic top-N filter, join
  filter pushdown.
* **Unsupported-type fallback** — ``UHUGEINT``, ``string_view``,
  ``binary_view`` columns must not crash; the filter is applied above the
  scan instead.
* **Regressions** — issues that were fixed previously and need to stay
  fixed.
* **Canaries** — markers for behaviour we expect to change upstream
  (pyarrow gaining view-filter support, DuckDB starting to push IS_NULL or
  struct IN, etc.).

Two conversion paths are exercised everywhere it makes sense — `.to_arrow_table()`
and `pandas`-via-pyarrow `df()`. Some tests also run through
`pyarrow.dataset` to cover the dataset-scanner code path.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest
from _pushdown_helpers import (
    ARROW_FACTORIES,
    ARROW_FACTORIES_WITH_DATASET,
    COMPARABLE_TYPES,
    COMPARISON_CASES,
    to_arrow_table,
)
from _pushdown_helpers import (
    arrow_scan_block as _arrow_scan_block,
)
from _pushdown_helpers import (
    count as _count,
)
from _pushdown_helpers import (
    make_typed_table as _make_typed_table,
)
from _pushdown_helpers import (
    was_pushed as _was_pushed,
)
from packaging.version import Version

import duckdb

pa = pytest.importorskip("pyarrow")
pa_ds = pytest.importorskip("pyarrow.dataset")
pa_lib = pytest.importorskip("pyarrow.lib")
pa_parquet = pytest.importorskip("pyarrow.parquet")
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


# ===========================================================================
# 1. Comparison correctness across types
# ===========================================================================


@pytest.mark.parametrize("factory", ARROW_FACTORIES)
@pytest.mark.parametrize("case", COMPARABLE_TYPES, ids=lambda c: c.id)
@pytest.mark.parametrize(("predicate_tpl", "expected"), COMPARISON_CASES)
def test_comparisons(duckdb_cursor, factory, case, predicate_tpl, expected):
    """Each (type, factory, predicate) tuple produces the expected row count."""
    _make_typed_table(duckdb_cursor, factory, case)
    predicate = predicate_tpl.format(low=case.low, mid=case.mid, high=case.high)
    assert _count(duckdb_cursor, predicate) == expected


# BOOL has no ordering, so it gets its own tiny suite.
@pytest.mark.parametrize("factory", ARROW_FACTORIES)
def test_bool_comparisons(duckdb_cursor, factory):
    """Equality / IS NULL / AND / OR on BOOL columns."""
    duckdb_cursor.execute("CREATE TABLE _b (a BOOL, b BOOL)")
    duckdb_cursor.execute("INSERT INTO _b VALUES (TRUE, TRUE), (TRUE, FALSE), (FALSE, TRUE), (NULL, NULL)")
    arrow_table = factory(duckdb_cursor.table("_b"))
    duckdb_cursor.register("arrow_table", arrow_table)

    assert _count(duckdb_cursor, "a = TRUE") == 2
    assert _count(duckdb_cursor, "a IS NULL") == 1
    assert _count(duckdb_cursor, "a IS NOT NULL") == 3
    assert _count(duckdb_cursor, "a = TRUE AND b = TRUE") == 1
    assert _count(duckdb_cursor, "a = TRUE OR b = TRUE") == 3


# Integer boundary values are worth a separate test because the GetScalar path
# has to coerce each (DuckDB Value) -> (pyarrow scalar) at the limit.
@pytest.mark.parametrize("factory", ARROW_FACTORIES)
@pytest.mark.parametrize(
    ("data_type", "max_value"),
    [
        ("TINYINT", 127),
        ("SMALLINT", 32767),
        ("INTEGER", 2147483647),
        ("BIGINT", 9223372036854775807),
        ("UTINYINT", 255),
        ("USMALLINT", 65535),
        ("UINTEGER", 4294967295),
        ("UBIGINT", 18446744073709551615),
    ],
)
def test_integer_max_value(duckdb_cursor, factory, data_type, max_value):
    """Pushdown round-trips through every integer's maximum representable value."""
    duckdb_cursor.execute(f"CREATE TABLE _t AS SELECT {max_value}::{data_type} AS i")
    arrow_table = factory(duckdb_cursor.table("_t"))
    duckdb_cursor.register("arrow_table", arrow_table)
    expected = [(max_value,)]
    assert duckdb_cursor.sql("SELECT * FROM arrow_table WHERE i > 0").fetchall() == expected
    assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE i > ?", (0,)).fetchall() == expected
    assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE i = ?", (max_value,)).fetchall() == expected


# ===========================================================================
# 2. OR pushdown decisions
#
# The optimizer pushes only a subset of OR shapes. These tests verify which
# shapes survive by inspecting the EXPLAIN plan.
# ===========================================================================


class TestOrPushdownDecisions:
    """Same-column ORs push; multi-column ORs and OR-with-LIKE/NULL don't."""

    @pytest.fixture(autouse=True)
    def _arrow_table(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _o (a INTEGER, b INTEGER, c INTEGER)")
        duckdb_cursor.execute("INSERT INTO _o VALUES (1,1,1),(10,10,10),(100,10,100),(NULL,NULL,NULL)")
        duckdb_cursor.register("arrow_table", to_arrow_table(duckdb_cursor.table("_o")))

    def test_single_column_or_pushes(self, duckdb_cursor):
        assert _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR a = 10")

    def test_single_column_or_with_and_does_not_push(self, duckdb_cursor):
        # The optimizer does not currently push ``a = 1 OR (a > 3 AND a < 5)``
        # — the AND inside the OR keeps it as a filter node above the scan.
        # The original test had a vacuous regex (``...|$``) that always matched;
        # this is the real behavior on current DuckDB main.
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR (a > 3 AND a < 5)")

    def test_multiple_or_terms_push(self, duckdb_cursor):
        assert _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR a > 3 OR a < 5")

    def test_or_with_not_equal_pushes(self, duckdb_cursor):
        assert _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a != 1 OR a > 3 OR a < 2")

    def test_multi_column_or_does_not_push(self, duckdb_cursor):
        # Optimizer refuses to push a root OR that references multiple columns.
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR b = 2 AND (a > 3 OR b < 5)")


class TestStringOrSpecifics:
    """VARCHAR has stricter OR-pushdown rules than numeric types."""

    @pytest.fixture(autouse=True)
    def _arrow_table(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _v (a VARCHAR, b VARCHAR, c VARCHAR)")
        duckdb_cursor.execute(
            "INSERT INTO _v VALUES ('1','1','1'),('10','10','10'),('100','10','100'),(NULL,NULL,NULL)"
        )
        duckdb_cursor.register("arrow_table", to_arrow_table(duckdb_cursor.table("_v")))

    def test_string_range_or_pushes(self, duckdb_cursor):
        assert _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a >= '1' OR a <= '10'")

    def test_or_with_is_null_does_not_push(self, duckdb_cursor):
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a IS NULL OR a = '1'")

    def test_or_with_is_not_null_does_not_push(self, duckdb_cursor):
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a IS NOT NULL OR a = '1'")

    def test_or_with_like_does_not_push(self, duckdb_cursor):
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = '1' OR a LIKE '10%'")


# ===========================================================================
# 3. IN-list pushdown
# ===========================================================================


class TestInPushdown:
    """IN (...) pushdown test.

    IN (...) reaches the walker as a ``COMPARE_IN`` ``BOUND_OPERATOR``, wrapped in an
    ``__internal_tablefilter_optional`` function that the walker must unwrap before it
    sees the operator.
    """

    def test_basic(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _t AS SELECT range a FROM range(1000)")
        duckdb_cursor.register("arrow_table", to_arrow_table(duckdb_cursor.table("_t")))
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE a = ANY([1, 999])").fetchall() == [(1,), (999,)]

    @pytest.mark.timeout(10)
    def test_large_in_list_does_not_hang(self):
        """Regression: https://github.com/duckdb/duckdb-python/issues/52."""
        duckdb.register("arrow_table", pa.table({"a": pa.array(range(5000))}))
        in_list = ", ".join(str(i) for i in range(0, 5000, 2))
        result = duckdb.sql(f"SELECT count(*) FROM arrow_table WHERE a IN ({in_list})").fetchone()
        assert result == (2500,)

    def test_in_with_no_nulls_in_list(self):
        duckdb.register("arrow_table", pa.table({"a": pa.array([1, 2, None, 4, None, 6])}))
        result = duckdb.sql("SELECT a FROM arrow_table WHERE a IN (1, 4) ORDER BY a").fetchall()
        assert result == [(1,), (4,)]

    def test_in_with_null_in_list(self):
        """SQL semantics: NULL in the IN list still doesn't match NULL rows."""
        duckdb.register("arrow_table", pa.table({"a": pa.array([1, 2, None, 4, None, 6])}))
        result = duckdb.sql("SELECT a FROM arrow_table WHERE a IN (1, 4, NULL) ORDER BY a").fetchall()
        assert result == [(1,), (4,)]

    def test_in_varchar(self):
        duckdb.register("arrow_table", pa.table({"s": pa.array(["alice", "bob", "charlie", "dave", None])}))
        result = duckdb.sql("SELECT s FROM arrow_table WHERE s IN ('bob', 'dave') ORDER BY s").fetchall()
        assert result == [("bob",), ("dave",)]

    def test_in_float(self):
        duckdb.register("arrow_table", pa.table({"f": pa.array([1.0, 2.5, 3.75, 4.0, None], type=pa.float64())}))
        result = duckdb.sql("SELECT f FROM arrow_table WHERE f IN (2.5, 4.0) ORDER BY f").fetchall()
        assert result == [(2.5,), (4.0,)]


# ===========================================================================
# 4. NaN pushdown
#
# DuckDB intentionally violates IEEE-754: NaN is the greatest value.
# The pyarrow_filter_pushdown special-cases this so the pyarrow side gets
# is_nan() / its inverse / constant(true|false) depending on the operator.
# ===========================================================================


class TestNaNPushdown:
    """Six comparison operators against a NaN constant on a DOUBLE column."""

    @pytest.fixture(autouse=True)
    def _nan_arrow_table(self, duckdb_cursor):
        duckdb_cursor.execute(
            "CREATE TABLE _n AS SELECT a::DOUBLE a FROM VALUES "
            "('inf'), ('nan'), ('0.34234'), ('34234234.00005'), ('-nan') t(a)"
        )
        arrow_table = to_arrow_table(duckdb_cursor.table("_n"))
        duckdb_cursor.register("arrow_table", arrow_table)

    @pytest.mark.parametrize(
        "op",
        ["=", "!=", "<", "<=", ">", ">="],
    )
    def test_nan_comparison_matches_duckdb(self, duckdb_cursor, op):
        """Each NaN comparison through the arrow scan agrees with DuckDB's own answer."""
        q_arrow = f"SELECT count(*) FROM arrow_table WHERE a {op} 'NaN'::FLOAT"
        q_duck = f"SELECT count(*) FROM _n WHERE a {op} 'NaN'::FLOAT"
        assert duckdb_cursor.execute(q_arrow).fetchone() == duckdb_cursor.execute(q_duck).fetchone()


# ===========================================================================
# 5. Struct extract pushdown
# ===========================================================================


class TestOneLevelStruct:
    """``struct_extract`` chains build the path inside ``ResolveColumn``.

    The EXPLAIN plan renders the predicate using the function form
    ``(struct_extract(s, 'a') < 2)`` rather than the dot form ``s.a < 2``.
    """

    @pytest.fixture(autouse=True)
    def _one_level_struct(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _s (s STRUCT(a INTEGER, b BOOL))")
        duckdb_cursor.execute(
            "INSERT INTO _s VALUES "
            "({'a': 1, 'b': true}), ({'a': 2, 'b': false}), (NULL), "
            "({'a': 3, 'b': true}), ({'a': NULL, 'b': NULL})"
        )
        arrow_table = to_arrow_table(duckdb_cursor.table("_s"))
        duckdb_cursor.register("arrow_table", arrow_table)

    def test_one_level_comparison_is_pushed(self, duckdb_cursor):
        plan = duckdb_cursor.execute("EXPLAIN SELECT * FROM arrow_table WHERE s.a < 2").fetchone()[1]
        assert re.search(r"struct_extract\(s,\s*'a'\)\s*<", plan)

    def test_one_level_comparison_correct(self, duckdb_cursor):
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s.a < 2").fetchone()[0] == {"a": 1, "b": True}

    def test_one_level_and_across_fields_is_pushed(self, duckdb_cursor):
        # Strip box-drawing/padding so we can pattern-match across line wraps.
        plan = duckdb_cursor.execute("EXPLAIN SELECT * FROM arrow_table WHERE s.a < 3 AND s.b = true").fetchone()[1]
        block = _arrow_scan_block(plan)
        assert block is not None
        flat = re.sub(r"[│|\s]+", " ", block)
        assert re.search(r"struct_extract\(s, 'a'\).*struct_extract\(s, 'b'\)", flat)

    def test_one_level_and_correct(self, duckdb_cursor):
        assert duckdb_cursor.execute("SELECT count(*) FROM arrow_table WHERE s.a < 3 AND s.b = true").fetchone() == (1,)
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s.a < 3 AND s.b = true").fetchone()[0] == {
            "a": 1,
            "b": True,
        }


class TestNestedStruct:
    """Two-level ``struct_extract`` chains."""

    @pytest.fixture(autouse=True)
    def _nested_struct(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _n (s STRUCT(a STRUCT(b INTEGER, c BOOL), d STRUCT(e INTEGER, f VARCHAR)))")
        duckdb_cursor.execute(
            "INSERT INTO _n VALUES "
            "({'a': {'b': 1, 'c': false}, 'd': {'e': 2, 'f': 'foo'}}), "
            "(NULL), "
            "({'a': {'b': 3, 'c': true}, 'd': {'e': 4, 'f': 'bar'}}), "
            "({'a': {'b': NULL, 'c': true}, 'd': {'e': 5, 'f': 'qux'}}), "
            "({'a': NULL, 'd': NULL})"
        )
        arrow_table = to_arrow_table(duckdb_cursor.table("_n"))
        duckdb_cursor.register("arrow_table", arrow_table)

    def test_nested_two_level_is_pushed(self, duckdb_cursor):
        plan = duckdb_cursor.execute("EXPLAIN SELECT * FROM arrow_table WHERE s.a.b < 2").fetchone()[1]
        # Outer struct_extract(_, 'b') around inner struct_extract(s, 'a').
        assert re.search(
            r"struct_extract.*\(struct_extract\(s,\s*'a'\),.*'b'\)\s*<\s*2",
            plan,
            flags=re.DOTALL,
        )

    def test_nested_two_level_correct(self, duckdb_cursor):
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s.a.b < 2").fetchone()[0] == {
            "a": {"b": 1, "c": False},
            "d": {"e": 2, "f": "foo"},
        }

    def test_nested_and_across_branches(self, duckdb_cursor):
        assert duckdb_cursor.execute(
            "SELECT count(*) FROM arrow_table WHERE s.a.c = true AND s.d.e = 5"
        ).fetchone() == (1,)

    def test_nested_varchar_comparison(self, duckdb_cursor):
        plan = duckdb_cursor.execute("EXPLAIN SELECT * FROM arrow_table WHERE s.d.f = 'bar'").fetchone()[1]
        assert re.search(
            r"struct_extract.*\(struct_extract\(s,\s*'d'\),.*'f'\)\s*=\s*'bar'",
            plan,
            flags=re.DOTALL,
        )
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s.d.f = 'bar'").fetchone()[0] == {
            "a": {"b": 3, "c": True},
            "d": {"e": 4, "f": "bar"},
        }


# ===========================================================================
# 6. LIKE pushdown
# ===========================================================================


class TestLikePushdown:
    """Test LIKE filter pushdown.

    LIKE with a fixed prefix decomposes into ``>= prefix AND < prefix+1``; a
    constant LIKE (no wildcards) decomposes into ``=``. Both produce regular
    comparison ExpressionFilters that the walker handles.
    """

    @pytest.fixture(autouse=True)
    def _s_arrow_table(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _l AS SELECT 'str_' || lpad(i::VARCHAR, 4, '0') AS s FROM range(100) t(i)")
        arrow_table = to_arrow_table(duckdb_cursor.table("_l"))
        duckdb_cursor.register("arrow_table", arrow_table)

    def test_like_with_prefix_is_pushed(self, duckdb_cursor):
        assert _was_pushed(duckdb_cursor, "SELECT s FROM arrow_table WHERE s LIKE 'str_001%'")

    def test_like_constant_is_pushed(self, duckdb_cursor):
        assert _was_pushed(duckdb_cursor, "SELECT s FROM arrow_table WHERE s LIKE 'str_0042'")

    def test_like_with_prefix_correct(self, duckdb_cursor):
        rows = duckdb_cursor.execute("SELECT s FROM arrow_table WHERE s LIKE 'str_001%' ORDER BY s").fetchall()
        assert rows == [(f"str_001{d}",) for d in "0123456789"]


# ===========================================================================
# 7. CAST temporal pushdown
# ===========================================================================


class TestTemporalCastPushdown:
    """``CAST(timestamp_col AS DATE) = …`` pushes an optional relaxed range filter.

    See `TryPushdownTemporalCastFilter`.
    """

    def test_cast_timestamp_to_date_is_pushed(self, duckdb_cursor):
        duckdb_cursor.execute(
            "CREATE TABLE _ct AS "
            "SELECT TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i) SECOND AS ts FROM range(86400) t(i)"
        )
        arrow_table = to_arrow_table(duckdb_cursor.table("_ct"))
        duckdb_cursor.register("arrow_table", arrow_table)
        assert _was_pushed(
            duckdb_cursor,
            "SELECT * FROM arrow_table WHERE CAST(ts AS DATE) = DATE '2024-01-01'",
        )


# ===========================================================================
# 8. IS DISTINCT FROM NULL inside OR
#
# This is the one realistic SQL path that produces an
# ExpressionFilter(OPERATOR_IS_NULL / IS_NOT_NULL) for the walker — see
# filter_combiner.cpp:615-625.
# ===========================================================================


class TestDistinctFromNullOrPushdown:
    """``IS DISTINCT FROM NULL OR ...`` produces an IS_NOT_NULL ExpressionFilter."""

    @pytest.fixture(autouse=True)
    def _with_nulls(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _d AS SELECT * FROM (VALUES (1), (NULL), (5), (10)) t(a)")
        arrow_table = to_arrow_table(duckdb_cursor.table("_d"))
        duckdb_cursor.register("arrow_table", arrow_table)

    def test_distinct_from_null_or_eq_is_pushed(self, duckdb_cursor):
        assert _was_pushed(
            duckdb_cursor,
            "SELECT a FROM arrow_table WHERE a IS DISTINCT FROM NULL OR a = 5",
        )

    def test_distinct_from_null_or_eq_correct(self, duckdb_cursor):
        rows = duckdb_cursor.execute(
            "SELECT a FROM arrow_table WHERE a IS DISTINCT FROM NULL OR a = 5 ORDER BY a"
        ).fetchall()
        assert rows == [(1,), (5,), (10,)]

    def test_not_distinct_from_null_or_eq_correct(self, duckdb_cursor):
        rows = duckdb_cursor.execute(
            "SELECT a FROM arrow_table WHERE a IS NOT DISTINCT FROM NULL OR a = 5 ORDER BY a NULLS FIRST"
        ).fetchall()
        # NULL row + a=5 row
        assert rows == [(None,), (5,)]


# ===========================================================================
# 9. Special-shape filters: optional, dynamic top-N, join
# ===========================================================================


class TestOptionalFilter:
    """An OptionalFilter is allowed to silently fail.

    The engine reapplies it above the scan. The result must remain correct.
    """

    def test_no_crash_correct_result(self):
        cardinality_table = pa.Table.from_pydict(
            {
                "column_name": [
                    "id",
                    "product_code",
                    "price",
                    "quantity",
                    "category",
                    "is_available",
                    "rating",
                    "discount",
                    "color",
                ],
                "cardinality": [100, 100, 100, 45, 5, 3, 6, 39, 5],
            }
        )
        result = duckdb.query(
            "SELECT * FROM cardinality_table WHERE cardinality > 1 ORDER BY cardinality ASC"
        ).fetchall()
        assert result == [
            ("is_available", 3),
            ("category", 5),
            ("color", 5),
            ("rating", 6),
            ("discount", 39),
            ("quantity", 45),
            ("id", 100),
            ("product_code", 100),
            ("price", 100),
        ]


class TestDynamicFilter:
    """The top-N optimization installs a dynamic filter.

    The walker returns ``py::none()`` for those (DuckDB applies them above the scan).
    """

    def test_topn_dynamic_filter(self, duckdb_cursor):
        t = pa.Table.from_pydict({"a": [3, 24, 234, 234, 234, 234, 234, 234, 234, 45, 2, 5, 2, 45]})
        duckdb_cursor.register("t", t)
        rows = duckdb_cursor.sql("SELECT a FROM t ORDER BY a LIMIT 11").fetchall()
        assert len(rows) == 11


class TestJoinFilterPushdown:
    """Join pushdown between two arrow tables must produce the right count.

    The join's runtime filters take a separate code path that doesn't
    reach the static-filter walker — see TestCanaries below.
    """

    def test_two_arrow_tables(self):
        con = duckdb.connect()
        con.execute("CREATE TABLE probe AS SELECT range a FROM range(10000)")
        con.execute("CREATE TABLE build AS SELECT (random()*9999)::INT b FROM range(20)")
        con.register("probe_arrow", to_arrow_table(con.table("probe")))
        con.register("build_arrow", to_arrow_table(con.table("build")))
        assert con.execute("SELECT count(*) FROM probe_arrow, build_arrow WHERE a = b").fetchall() == [(20,)]


# ===========================================================================
# 10. Unsupported-type fallback (filter applied above the scan)
# ===========================================================================


class TestUnsupportedTypes:
    """``UHUGEINT``, ``string_view``, ``binary_view`` filters must not crash.

    The filter is applied above the scan instead of being pushed down.
    """

    def test_uhugeint_single_filter(self):
        con = duckdb.connect()
        con.execute(
            "CREATE TABLE t AS SELECT i::INTEGER a, i::VARCHAR b, i::UHUGEINT c, i::INTEGER d FROM range(5) tbl(i)"
        )
        arrow_tbl = to_arrow_table(con.table("t"))
        con.register("arrow_tbl", arrow_tbl)
        assert con.execute("FROM arrow_tbl WHERE c = 3").fetchall() == [(3, "3", 3, 3)]

    def test_dynamic_filter_nulls_first_pyarrow(self, duckdb_cursor):
        # Regression for #460(a): TOP_N with ASC NULLS FIRST pushes an
        # OPTIONAL(x IS NULL OR DYNAMIC_FILTER(x)) into the arrow scan. The
        # pyarrow translation must NOT collapse the OR by dropping the
        # untranslatable DYNAMIC_FILTER branch — doing so produces a
        # stricter (`field("x").is_null()`) predicate that drops every row.
        t = pa.Table.from_pydict({"x": pa.array([3, 1, 2], type=pa.int32())})
        duckdb_cursor.register("src", t)
        res = duckdb_cursor.sql("SELECT * FROM src ORDER BY x ASC NULLS FIRST LIMIT 1").fetchall()
        assert res == [(1,)]

    def test_dynamic_filter_nulls_first_polars_dataframe(self, duckdb_cursor):
        # pl.DataFrame is materialized to a pyarrow.Table before scanning,
        # so it exercises PyarrowFilterPushdown the same way pa.Table does.
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"x": [3, 1, 2]})
        duckdb_cursor.register("src", df)
        res = duckdb_cursor.sql("SELECT * FROM src ORDER BY x ASC NULLS FIRST LIMIT 1").fetchall()
        assert res == [(1,)]

    def test_uhugeint_mixed_with_supported(self):
        con = duckdb.connect()
        con.execute(
            "CREATE TABLE t AS SELECT i::INTEGER a, i::VARCHAR b, i::UHUGEINT c, i::INTEGER d FROM range(5) tbl(i)"
        )
        arrow_tbl = to_arrow_table(con.table("t"))
        con.register("arrow_tbl", arrow_tbl)
        assert con.execute("FROM arrow_tbl WHERE c < 4 AND a > 2").fetchall() == [(3, "3", 3, 3)]
        assert con.execute("FROM arrow_tbl WHERE a > 2 AND c < 4 AND b = '3'").fetchall() == [(3, "3", 3, 3)]
        assert con.execute("FROM arrow_tbl WHERE a > 2 AND c < 4 AND b = '0'").fetchall() == []

    def test_uhugeint_with_projection(self):
        con = duckdb.connect()
        con.execute(
            "CREATE TABLE t AS SELECT i::INTEGER a, i::VARCHAR b, i::UHUGEINT c, i::INTEGER d FROM range(5) tbl(i)"
        )
        arrow_tbl = to_arrow_table(con.table("t"))
        con.register("arrow_tbl", arrow_tbl)
        assert con.execute("SELECT c, b FROM arrow_tbl WHERE c < 4 AND b = '3' AND a > 2").fetchall() == [(3, "3")]
        # Projection list doesn't include the unpushable column
        assert con.execute("SELECT a, b FROM arrow_tbl WHERE a > 2 AND c < 4 AND b = '3'").fetchall() == [(3, "3")]

    def test_multiple_unpushable_filters(self):
        con = duckdb.connect()
        con.execute(
            "CREATE TABLE t AS SELECT i::INTEGER a, i::VARCHAR b, i::UHUGEINT c, "
            "i::INTEGER d, i::UHUGEINT e, i::SMALLINT f, i::UHUGEINT g "
            "FROM range(50) tbl(i)"
        )
        arrow_tbl = to_arrow_table(con.table("t"))
        con.register("arrow_tbl", arrow_tbl)
        assert con.execute(
            "SELECT a, b FROM arrow_tbl WHERE a > 2 AND c < 40 AND b = '28' AND g > 15 AND e < 30"
        ).fetchall() == [(28, "28")]

    def test_binary_view_filter_does_not_crash(self):
        """Binary view filters cannot be pushed (pyarrow limitation).

        Results must still be correct.
        """
        table = pa.table({"col": pa.array([b"abc", b"efg"], type=pa.binary_view())})
        dset = pa_ds.dataset(table)
        res = duckdb.sql("SELECT * FROM dset WHERE col = 'abc'::BINARY").fetchall()
        assert res == [(b"abc",)]

    def test_string_view_filter_does_not_crash(self):
        """String view filters cannot be pushed (pyarrow limitation).

        Results must still be correct.
        """
        table = pa.table({"col": pa.array(["abc", "efg"], type=pa.string_view())})
        dset = pa_ds.dataset(table)
        res = duckdb.sql("SELECT * FROM dset WHERE col = 'abc'").fetchall()
        assert res == [("abc",)]


# ===========================================================================
# 11. Projection / scanner-path interactions
# ===========================================================================


@pytest.mark.parametrize("factory", ARROW_FACTORIES_WITH_DATASET)
def test_filter_without_projection(duckdb_cursor, factory):
    """Filter applied when no projection is specified.

    Covers all three conversion paths including the dataset scanner.
    """
    duckdb_cursor.execute("CREATE TABLE _np (a INTEGER, b INTEGER, c INTEGER)")
    duckdb_cursor.execute("INSERT INTO _np VALUES (1,1,1),(10,10,10),(100,10,100),(NULL,NULL,NULL)")
    arrow_table = factory(duckdb_cursor.table("_np"))
    duckdb_cursor.register("arrow_table", arrow_table)
    assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE a = 1").fetchall() == [(1, 1, 1)]


# ===========================================================================
# 12. Decimal pushdown via polars (the only path that exercises decimal
# scalar coercion in the walker)
# ===========================================================================


def test_decimal_filter_pushdown_via_polars(duckdb_cursor):
    """Polars decimal frames stress GetScalar's decimal branch."""
    pl = pytest.importorskip("polars")
    np.random.seed(10)
    df = pl.DataFrame({"x": pl.Series(np.random.uniform(-10, 10, 1000)).cast(pl.Decimal(precision=18, scale=4))})
    rows = duckdb_cursor.sql(
        """
        SELECT x, x > 0.05 AS is_x_good, x::FLOAT > 0.05 AS is_float_x_good
        FROM df
        WHERE is_x_good
        ORDER BY x ASC
        """
    ).fetchall()
    assert len(rows) == 495


# ===========================================================================
# 13. Regressions
# ===========================================================================


class TestRegressions:
    """Issues that were fixed and need to stay fixed."""

    @pytest.mark.skipif(
        Version(pa.__version__) < Version("15.0.0"),
        reason="pyarrow 14.0.2 'to_pandas' causes a DeprecationWarning",
    )
    def test_9371_arrow_dataset_with_tz_parameter(self, duckdb_cursor, tmp_path):
        """Parameterized timestamp filter against a pandas-indexed parquet dataset.

        https://github.com/duckdb/duckdb/issues/9371
        """
        duckdb_cursor.execute("SET TimeZone='UTC'")
        file_path = tmp_path / "test.parquet"
        timestamp = dt.datetime(2023, 8, 29, 1, tzinfo=dt.timezone.utc)
        my_arrow_table = pa.Table.from_pydict({"ts": [timestamp] * 3, "value": [1, 2, 3]})
        df = my_arrow_table.to_pandas().set_index("ts")
        df.to_parquet(str(file_path))

        my_arrow_dataset = pa_ds.dataset(str(file_path))
        res = duckdb_cursor.execute(
            "SELECT * FROM my_arrow_dataset WHERE ts = ?", parameters=[timestamp]
        ).to_arrow_table()
        assert duckdb_cursor.sql("SELECT * FROM res").fetchall() == [(1, timestamp), (2, timestamp), (3, timestamp)]

    def test_2145_parquet_glob_through_arrow(self, duckdb_cursor, tmp_path):
        """Filter pushdown into a parquet glob via the arrow scan.

        https://github.com/duckdb/duckdb/issues/2145
        """
        date1 = pd.date_range("2018-01-01", "2018-12-31", freq="B")
        df1 = pd.DataFrame(np.random.randn(date1.shape[0], 5), columns=list("ABCDE"))
        df1["date"] = date1
        date2 = pd.date_range("2019-01-01", "2019-12-31", freq="B")
        df2 = pd.DataFrame(np.random.randn(date2.shape[0], 5), columns=list("ABCDE"))
        df2["date"] = date2

        data1 = tmp_path / "data1.parquet"
        data2 = tmp_path / "data2.parquet"
        duckdb_cursor.execute(f"COPY (SELECT * FROM df1) TO '{data1.as_posix()}'")
        duckdb_cursor.execute(f"COPY (SELECT * FROM df2) TO '{data2.as_posix()}'")

        glob_pattern = (tmp_path / "data*.parquet").as_posix()
        table = duckdb_cursor.read_parquet(glob_pattern).to_arrow_table()
        output_df = duckdb.arrow(table).filter("date > '2019-01-01'").df()
        expected_df = duckdb.from_parquet(glob_pattern).filter("date > '2019-01-01'").df()
        pd.testing.assert_frame_equal(expected_df, output_df)


# ===========================================================================
# 14. Canaries
#
# Each canary documents a current limitation or an expected future change.
# When upstream behaviour shifts, a canary either xpasses (failing the suite
# and forcing us to update) or fails outright.
# ===========================================================================


class TestCanaries:
    """Markers for behaviours we expect to change upstream eventually."""

    # ----- pyarrow capabilities ----------------------------------------

    @pytest.mark.xfail(
        raises=pa_lib.ArrowNotImplementedError,
        reason="pyarrow does not yet implement string_view filter compare kernels",
        strict=True,
    )
    def test_pyarrow_gains_string_view_filter_support(self):
        """When pyarrow adds string_view comparison kernels this will xpass.

        At that point we should remove the post-scan fallback in TestUnsupportedTypes.
        """
        filter_expr = pa_ds.field("col") == pa_ds.scalar("val1")
        table = pa.table({"col": pa.array(["val1", "val2"], type=pa.string_view())})
        pa_ds.dataset(table).scanner(columns=["col"], filter=filter_expr)

    @pytest.mark.xfail(
        raises=pa_lib.ArrowNotImplementedError,
        reason="pyarrow does not yet implement binary_view filter compare kernels",
        strict=True,
    )
    def test_pyarrow_gains_binary_view_filter_support(self):
        """When pyarrow adds binary_view comparison kernels this will xpass."""
        filter_expr = pa_ds.field("col") == pa_ds.scalar(pa.scalar(b"bin1", pa.binary_view()))
        table = pa.table({"col": pa.array([b"bin1", b"bin2"], type=pa.binary_view())})
        pa_ds.dataset(table).scanner(columns=["col"], filter=filter_expr)

    # ----- DuckDB optimizer decisions we expect to change --------------

    @pytest.mark.xfail(
        reason="DuckDB does not currently push IS NULL into the arrow scan",
        strict=True,
    )
    def test_is_null_pushes_into_arrow_scan(self, duckdb_cursor):
        """If the optimizer starts pushing standalone IS NULL into arrow scans, this canary xpasses.

        The walker already has the OPERATOR_IS_NULL arm.
        """
        duckdb_cursor.execute("CREATE TABLE _t AS SELECT * FROM (VALUES (1), (NULL), (3)) v(a)")
        arrow_table = to_arrow_table(duckdb_cursor.table("_t"))
        duckdb_cursor.register("arrow_table", arrow_table)
        assert _was_pushed(duckdb_cursor, "SELECT a FROM arrow_table WHERE a IS NULL")

    @pytest.mark.xfail(
        reason="DuckDB does not currently push IS NULL on struct fields into the arrow scan",
        strict=True,
    )
    def test_struct_is_null_pushes(self, duckdb_cursor):
        """If the optimizer starts pushing struct-field IS NULL, this canary xpasses."""
        duckdb_cursor.execute("CREATE TABLE _s (s STRUCT(a INTEGER))")
        duckdb_cursor.execute("INSERT INTO _s VALUES ({'a': 1}), ({'a': NULL}), (NULL)")
        arrow_table = to_arrow_table(duckdb_cursor.table("_s"))
        duckdb_cursor.register("arrow_table", arrow_table)
        assert _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE s.a IS NULL")

    @pytest.mark.xfail(
        reason="DuckDB does not currently push struct.a IN (...) into the arrow scan; "
        "TryPushdownInFilter requires a bare BoundColumnRef (filter_combiner.cpp:505-508)",
        strict=True,
    )
    def test_struct_in_pushes(self, duckdb_cursor):
        """When DuckDB extends TryPushdownInFilter to allow struct_extract column sides, this canary xpasses.

        ResolveColumn already handles the path.
        """
        duckdb_cursor.execute("CREATE TABLE _s (s STRUCT(a INTEGER))")
        duckdb_cursor.execute("INSERT INTO _s VALUES ({'a': 1}), ({'a': 2}), ({'a': 42}), (NULL)")
        arrow_table = to_arrow_table(duckdb_cursor.table("_s"))
        duckdb_cursor.register("arrow_table", arrow_table)
        assert _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE s.a IN (1, 42, 99)")

    # ----- Join filter pushdown ----------------------------------------
    #
    # BLOOM_FILTER, PERFECT_HASH_JOIN_FILTER, and PREFIX_RANGE_FILTER are
    # generated by hash joins. Today they take a separate runtime path
    # (`info.dynamic_filters->PushFilter(...)` in physical_hash_join.cpp) that
    # does NOT reach PyArrowFilterPushdown::TransformFilter. These canaries
    # document the current behaviour: the join runs to the correct answer but
    # the walker is not invoked, so the new filter types never surface to the
    # bindings. The day arrow_array_stream.cpp starts receiving them via the
    # static filter set, we'll fail here.

    @pytest.mark.xfail(
        reason="BLOOM_FILTER from joins reaches arrow scans via runtime dynamic_filters, "
        "not PyArrowFilterPushdown::TransformFilter",
        strict=True,
    )
    def test_bloom_filter_reaches_walker(self, duckdb_cursor):
        """When join bloom filters start flowing through the static filter set, this canary xpasses.

        Used by arrow_array_stream.cpp. We'll need to add a BLOOM_FILTER case in
        TransformFilterRecursive.
        """
        duckdb_cursor.execute("CREATE TABLE _probe AS SELECT range AS k FROM range(100_000)")
        duckdb_cursor.execute("CREATE TABLE _build AS SELECT (i*2)::BIGINT AS k FROM range(50_000) t(i)")
        probe_arrow = to_arrow_table(duckdb_cursor.table("_probe"))
        duckdb_cursor.register("probe_arrow", probe_arrow)
        assert _was_pushed(duckdb_cursor, "SELECT count(*) FROM probe_arrow JOIN _build USING(k)")

    @pytest.mark.xfail(
        reason="PERFECT_HASH_JOIN_FILTER from joins reaches arrow scans via runtime dynamic_filters, "
        "not PyArrowFilterPushdown::TransformFilter",
        strict=True,
    )
    def test_perfect_hash_join_filter_reaches_walker(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _probe AS SELECT range AS k FROM range(10_000)")
        duckdb_cursor.execute("CREATE TABLE _build AS SELECT i AS k FROM range(100) t(i)")
        probe_arrow = to_arrow_table(duckdb_cursor.table("_probe"))
        duckdb_cursor.register("probe_arrow", probe_arrow)
        assert _was_pushed(duckdb_cursor, "SELECT count(*) FROM probe_arrow JOIN _build USING(k)")

    @pytest.mark.xfail(
        reason="PREFIX_RANGE_FILTER from joins reaches arrow scans via runtime dynamic_filters, "
        "not PyArrowFilterPushdown::TransformFilter",
        strict=True,
    )
    def test_prefix_range_filter_reaches_walker(self, duckdb_cursor):
        duckdb_cursor.execute(
            "CREATE TABLE _probe AS SELECT 'str_' || lpad(i::VARCHAR, 4, '0') AS k FROM range(10_000) t(i)"
        )
        duckdb_cursor.execute(
            "CREATE TABLE _build AS SELECT 'str_' || lpad((i*2)::VARCHAR, 4, '0') AS k FROM range(500) t(i)"
        )
        probe_arrow = to_arrow_table(duckdb_cursor.table("_probe"))
        duckdb_cursor.register("probe_arrow", probe_arrow)
        assert _was_pushed(duckdb_cursor, "SELECT count(*) FROM probe_arrow JOIN _build USING(k)")

    # ----- Optimizer canonicalization expected to stay -----------------

    def test_not_in_does_not_push(self, duckdb_cursor):
        """``NOT IN`` is rewritten to AND-of-!= by the optimizer rather than being pushed as a single operator.

        If this ever changes the assertion flips to ``_was_pushed`` and the walker's
        BOUND_OPERATOR arm needs an ``OPERATOR_NOT`` case.
        """
        duckdb_cursor.execute("CREATE TABLE _t AS SELECT range AS a FROM range(1000)")
        arrow_table = to_arrow_table(duckdb_cursor.table("_t"))
        duckdb_cursor.register("arrow_table", arrow_table)
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a NOT IN (1, 5, 100)")

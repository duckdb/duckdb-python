# ruff: noqa: F841
"""Filter pushdown tests for the arrow scan integration.

What's tested here:

* **Comparison correctness** across every supported type:
  ``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``IS NULL``, ``IS NOT NULL``,
  ``AND``, ``OR``.
* **Filters kept above the scan**: a pyarrow Table refuses every filter, so
  each predicate shape stays in a FILTER operator above the Arrow scan, which
  picks only the columns the query reads. Verified by inspecting the EXPLAIN
  plan, not just row counts.
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
  (pyarrow gaining view-filter support).

Two conversion paths are exercised everywhere it makes sense — `.to_arrow_table()`
and `pandas`-via-pyarrow `df()`. Some tests also run through
`pyarrow.dataset` to cover the dataset-scanner code path.
"""

from __future__ import annotations

import datetime as dt

import pytest
from _pushdown_helpers import (
    ARROW_FACTORIES,
    ARROW_FACTORIES_WITH_DATASET,
    COMPARABLE_TYPES,
    COMPARISON_CASES,
    assert_plan_keeps_filter_above,
    explain_physical_plan,
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
from packaging.version import Version

import duckdb

pa = pytest.importorskip("pyarrow")
pa_ds = pytest.importorskip("pyarrow.dataset")
pa_lib = pytest.importorskip("pyarrow.lib")
pa_parquet = pytest.importorskip("pyarrow.parquet")
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _assert_kept_above(con, query):
    """The Table refuses the filter, so the scan takes none and a FILTER above it applies the filter."""
    assert_plan_keeps_filter_above(explain_physical_plan(con, query))


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


# Integer boundary values are worth a separate test because each constant has to
# reach the pyarrow scalar unchanged at the limit.
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
# 2. OR shapes
#
# A Table refuses every OR shape, so each stays above the scan. These tests
# verify that by inspecting the EXPLAIN plan.
# ===========================================================================


class TestOrPushdownDecisions:
    """Every OR shape over a Table stays in a FILTER above the scan."""

    @pytest.fixture(autouse=True)
    def _arrow_table(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _o (a INTEGER, b INTEGER, c INTEGER)")
        duckdb_cursor.execute("INSERT INTO _o VALUES (1,1,1),(10,10,10),(100,10,100),(NULL,NULL,NULL)")
        duckdb_cursor.register("arrow_table", to_arrow_table(duckdb_cursor.table("_o")))

    def test_single_column_or_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR a = 10")

    def test_single_column_or_with_and_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR (a > 3 AND a < 5)")

    def test_multiple_or_terms_stay_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR a > 3 OR a < 5")

    def test_or_with_not_equal_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a != 1 OR a > 3 OR a < 2")

    def test_multi_column_or_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR b = 2 AND (a > 3 OR b < 5)")


class TestStringOrSpecifics:
    """Every VARCHAR OR shape over a Table stays in a FILTER above the scan."""

    @pytest.fixture(autouse=True)
    def _arrow_table(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _v (a VARCHAR, b VARCHAR, c VARCHAR)")
        duckdb_cursor.execute(
            "INSERT INTO _v VALUES ('1','1','1'),('10','10','10'),('100','10','100'),(NULL,NULL,NULL)"
        )
        duckdb_cursor.register("arrow_table", to_arrow_table(duckdb_cursor.table("_v")))

    def test_string_range_or_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a >= '1' OR a <= '10'")

    def test_or_with_is_null_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a IS NULL OR a = '1'")

    def test_or_with_is_not_null_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a IS NOT NULL OR a = '1'")

    def test_or_with_like_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = '1' OR a LIKE '10%'")


# ===========================================================================
# 3. IN-list pushdown
# ===========================================================================


class TestInPushdown:
    """IN (...) pushdown test.

    IN (...) reaches the scan as a ``COMPARE_IN`` ``BOUND_OPERATOR`` over constants.
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
        result = duckdb.sql(f"SELECT count(*) FROM arrow_table WHERE a IN ({in_list})").fetchall()[0]
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
# A NaN constant is never offered to a source, so the engine applies these
# comparisons above the scan with its own NaN ordering.
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

    @pytest.mark.parametrize(
        "op",
        ["=", "!=", "<", "<=", ">", ">="],
    )
    def test_finite_constant_includes_nan_rows(self, duckdb_cursor, op):
        """Regression (#9): a finite constant against a column that CONTAINS NaN.

        DuckDB orders NaN as the greatest value, so `nan > finite` / `nan >= finite` are TRUE; IEEE/pyarrow
        make them FALSE. Before the fix the arrow scan silently dropped the NaN rows for `>` / `>=` (the scan
        never re-applies pushed filters). Every operator must agree with DuckDB's own answer.
        """
        rows_arrow = duckdb_cursor.execute(f"SELECT a FROM arrow_table WHERE a {op} 4.0").fetchall()
        rows_duck = duckdb_cursor.execute(f"SELECT a FROM _n WHERE a {op} 4.0").fetchall()

        # NaN-safe row-set comparison: NaN != NaN, so bucket NaNs by count and sort the finite rows.
        def summarize(rows):
            vals = [r[0] for r in rows]
            nan_count = sum(1 for v in vals if v != v)
            finite = sorted(v for v in vals if v == v)
            return nan_count, finite

        assert summarize(rows_arrow) == summarize(rows_duck)


# ===========================================================================
# 5. Struct extract pushdown
# ===========================================================================


class TestOneLevelStruct:
    """A predicate on a struct field over a Table stays in a FILTER above the scan."""

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

    def test_one_level_comparison_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE s.a < 2")

    def test_one_level_comparison_correct(self, duckdb_cursor):
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s.a < 2").fetchone()[0] == {"a": 1, "b": True}

    def test_one_level_and_across_fields_stays_above(self, duckdb_cursor):
        query = "SELECT * FROM arrow_table WHERE s.a < 3 AND s.b = true"
        _assert_kept_above(duckdb_cursor, query)
        block = _arrow_scan_block(duckdb_cursor.execute(f"EXPLAIN {query}").fetchone()[1])
        assert block is not None
        assert "Filters:" not in block

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

    def test_nested_two_level_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE s.a.b < 2")

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
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE s.d.f = 'bar'")
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s.d.f = 'bar'").fetchone()[0] == {
            "a": {"b": 3, "c": True},
            "d": {"e": 4, "f": "bar"},
        }


# ===========================================================================
# 6. LIKE pushdown
# ===========================================================================


class TestLikePushdown:
    """A LIKE over a Table, with a prefix or a constant pattern, stays in a FILTER above the scan."""

    @pytest.fixture(autouse=True)
    def _s_arrow_table(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _l AS SELECT 'str_' || lpad(i::VARCHAR, 4, '0') AS s FROM range(100) t(i)")
        arrow_table = to_arrow_table(duckdb_cursor.table("_l"))
        duckdb_cursor.register("arrow_table", arrow_table)

    def test_like_with_prefix_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT s FROM arrow_table WHERE s LIKE 'str_001%'")

    def test_like_constant_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT s FROM arrow_table WHERE s LIKE 'str_0042'")

    def test_like_with_prefix_correct(self, duckdb_cursor):
        rows = duckdb_cursor.execute("SELECT s FROM arrow_table WHERE s LIKE 'str_001%' ORDER BY s").fetchall()
        assert rows == [(f"str_001{d}",) for d in "0123456789"]


# ===========================================================================
# 7. CAST temporal pushdown
# ===========================================================================


class TestTemporalCastPushdown:
    """``CAST(timestamp_col AS DATE) = …`` over a Table stays in a FILTER above the scan."""

    def test_cast_timestamp_to_date_stays_above(self, duckdb_cursor):
        duckdb_cursor.execute(
            "CREATE TABLE _ct AS "
            "SELECT TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i) SECOND AS ts FROM range(86400) t(i)"
        )
        arrow_table = to_arrow_table(duckdb_cursor.table("_ct"))
        duckdb_cursor.register("arrow_table", arrow_table)
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE CAST(ts AS DATE) = DATE '2024-01-01'")


# ===========================================================================
# 8. IS DISTINCT FROM NULL inside OR
# ===========================================================================


class TestDistinctFromNullOrPushdown:
    """``IS DISTINCT FROM NULL OR ...`` over a Table stays in a FILTER above the scan."""

    @pytest.fixture(autouse=True)
    def _with_nulls(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _d AS SELECT * FROM (VALUES (1), (NULL), (5), (10)) t(a)")
        arrow_table = to_arrow_table(duckdb_cursor.table("_d"))
        duckdb_cursor.register("arrow_table", arrow_table)

    def test_distinct_from_null_or_eq_stays_above(self, duckdb_cursor):
        _assert_kept_above(duckdb_cursor, "SELECT a FROM arrow_table WHERE a IS DISTINCT FROM NULL OR a = 5")

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
    """The top-N optimization installs a dynamic filter, which DuckDB applies itself; the rows match native."""

    def test_topn_dynamic_filter(self, duckdb_cursor):
        t = pa.Table.from_pydict({"a": [3, 24, 234, 234, 234, 234, 234, 234, 234, 45, 2, 5, 2, 45]})
        duckdb_cursor.register("t", t)
        rows = duckdb_cursor.sql("SELECT a FROM t ORDER BY a LIMIT 11").fetchall()
        assert len(rows) == 11

    @pytest.mark.parametrize(
        "order",
        ["a", "a DESC", "a ASC NULLS FIRST", "a DESC NULLS LAST", "b, a DESC"],
    )
    def test_topn_matches_native(self, duckdb_cursor, order):
        # 20011 is a prime above the row count, so every non NULL a is distinct and the order has no ties.
        duckdb_cursor.execute(
            "CREATE TABLE _top AS SELECT CASE WHEN i % 97 = 0 THEN NULL ELSE (i * 7919) % 20011 END AS a, "
            "(i % 3)::VARCHAR AS b FROM range(20000) t(i)"
        )
        duckdb_cursor.register("top_arrow", to_arrow_table(duckdb_cursor.table("_top")))
        query = "SELECT a FROM {} ORDER BY " + order + " LIMIT 25"
        assert duckdb_cursor.execute(query.format("top_arrow")).fetchall() == (
            duckdb_cursor.execute(query.format("_top")).fetchall()
        )


class TestJoinFilterPushdown:
    """Joins over Arrow tables must produce the same rows as over native tables.

    A hash join's runtime filters are applied by DuckDB, never offered to the Arrow source.
    """

    def test_two_arrow_tables(self):
        con = duckdb.connect()
        con.execute("CREATE TABLE probe AS SELECT range a FROM range(10000)")
        con.execute("CREATE TABLE build AS SELECT (random()*9999)::INT b FROM range(20)")
        con.register("probe_arrow", to_arrow_table(con.table("probe")))
        con.register("build_arrow", to_arrow_table(con.table("build")))
        assert con.execute("SELECT count(*) FROM probe_arrow, build_arrow WHERE a = b").fetchall() == [(20,)]

    @pytest.mark.parametrize(
        ("probe_sql", "build_sql"),
        [
            pytest.param(
                "SELECT range AS k FROM range(100_000)",
                "SELECT (i * 2)::BIGINT AS k FROM range(50_000) t(i)",
                id="bloom_filter",
            ),
            pytest.param(
                "SELECT range AS k FROM range(10_000)",
                "SELECT i AS k FROM range(100) t(i)",
                id="perfect_hash_join_filter",
            ),
            pytest.param(
                "SELECT 'str_' || lpad(i::VARCHAR, 4, '0') AS k FROM range(10_000) t(i)",
                "SELECT 'str_' || lpad((i * 2)::VARCHAR, 4, '0') AS k FROM range(500) t(i)",
                id="prefix_range_filter",
            ),
        ],
    )
    @pytest.mark.parametrize("build_side", ["native", "arrow"])
    def test_join_runtime_filters_match_native(self, duckdb_cursor, probe_sql, build_sql, build_side):
        duckdb_cursor.execute(f"CREATE TABLE _probe AS {probe_sql}")
        duckdb_cursor.execute(f"CREATE TABLE _build AS {build_sql}")
        duckdb_cursor.register("probe_arrow", to_arrow_table(duckdb_cursor.table("_probe")))
        duckdb_cursor.register("build_arrow", to_arrow_table(duckdb_cursor.table("_build")))
        build = "build_arrow" if build_side == "arrow" else "_build"
        query = "SELECT count(*), min(k), max(k) FROM {} JOIN {} USING (k)"
        assert duckdb_cursor.execute(query.format("probe_arrow", build)).fetchall() == (
            duckdb_cursor.execute(query.format("_probe", "_build")).fetchall()
        )


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
        # With ASC NULLS FIRST a Top N query filters on
        # x IS NULL OR DYNAMIC_FILTER(x). Collapsing the OR by dropping the
        # DYNAMIC_FILTER branch would leave a stricter x IS NULL predicate
        # that drops every row.
        t = pa.Table.from_pydict({"x": pa.array([3, 1, 2], type=pa.int32())})
        duckdb_cursor.register("src", t)
        res = duckdb_cursor.sql("SELECT * FROM src ORDER BY x ASC NULLS FIRST LIMIT 1").fetchall()
        assert res == [(1,)]

    def test_dynamic_filter_nulls_first_polars_dataframe(self, duckdb_cursor):
        # pl.DataFrame is materialized to a pyarrow.Table before scanning,
        # so it is read the same way pa.Table is.
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
    """Polars decimal frames stress decimal constants in filters."""
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
        res = duckdb_cursor.sql("SELECT * FROM my_arrow_dataset WHERE ts = ?", params=[timestamp]).to_arrow_table()
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


# pyarrow view type filter pushdown canaries.
# No released pyarrow fully executes view type filters (compare plus array_filter/take); checked
# against 25.0.0. These stay xfail until one does. They run .to_table() so scanner construction
# alone does not flip them: pyarrow accepts a binary_view predicate at build time but still fails
# inside array_filter at execution. When one XPASSes: bump its version below, enable view type
# filter pushdown, and drop the matching TestUnsupportedTypes fallback. See the view type filter
# pushdown tracking issue.
_PA_VERSION = Version(pa.__version__)
_PYARROW_STRING_VIEW_FILTER_VERSION = Version("99")
_PYARROW_BINARY_VIEW_FILTER_VERSION = Version("99")


class TestCanaries:
    """Markers for behaviours we expect to change upstream eventually."""

    # ----- pyarrow capabilities ----------------------------------------

    @pytest.mark.xfail(
        _PA_VERSION < _PYARROW_STRING_VIEW_FILTER_VERSION,
        raises=pa_lib.ArrowNotImplementedError,
        reason="pyarrow does not execute string_view filters fully (equal kernel)",
        strict=True,
    )
    def test_pyarrow_gains_string_view_filter_support(self):
        """XPASSes when pyarrow executes a string_view filter fully.

        Runs .to_table() so it reflects execution, not just scanner construction. When it
        XPASSes: bump the version, enable view type pushdown, drop the TestUnsupportedTypes
        fallback.
        """
        filter_expr = pa_ds.field("col") == pa_ds.scalar("val1")
        table = pa.table({"col": pa.array(["val1", "val2"], type=pa.string_view())})
        result = pa_ds.dataset(table).scanner(columns=["col"], filter=filter_expr).to_table()
        assert result.num_rows == 1

    @pytest.mark.xfail(
        _PA_VERSION < _PYARROW_BINARY_VIEW_FILTER_VERSION,
        raises=pa_lib.ArrowNotImplementedError,
        reason="pyarrow does not execute binary_view filters fully (array_filter kernel)",
        strict=True,
    )
    def test_pyarrow_gains_binary_view_filter_support(self):
        """XPASSes when pyarrow executes a binary_view filter fully.

        Runs .to_table() so it reflects execution, not just scanner construction. When it
        XPASSes: bump the version, enable view type pushdown, drop the TestUnsupportedTypes
        fallback.
        """
        filter_expr = pa_ds.field("col") == pa_ds.scalar(pa.scalar(b"bin1", pa.binary_view()))
        table = pa.table({"col": pa.array([b"bin1", b"bin2"], type=pa.binary_view())})
        result = pa_ds.dataset(table).scanner(columns=["col"], filter=filter_expr).to_table()
        assert result.num_rows == 1


class TestTableKeepsFiltersAbove:
    """Predicate shapes a Dataset or LazyFrame may take, which a Table leaves above its scan."""

    def test_is_null_stays_above(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _t AS SELECT * FROM (VALUES (1), (NULL), (3)) v(a)")
        duckdb_cursor.register("arrow_table", to_arrow_table(duckdb_cursor.table("_t")))
        _assert_kept_above(duckdb_cursor, "SELECT a FROM arrow_table WHERE a IS NULL")
        assert duckdb_cursor.execute("SELECT a FROM arrow_table WHERE a IS NULL").fetchall() == [(None,)]

    def test_struct_is_null_stays_above(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _s (s STRUCT(a INTEGER))")
        duckdb_cursor.execute("INSERT INTO _s VALUES ({'a': 1}), ({'a': NULL}), (NULL)")
        duckdb_cursor.register("arrow_table", to_arrow_table(duckdb_cursor.table("_s")))
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE s.a IS NULL")
        assert duckdb_cursor.execute("SELECT count(*) FROM arrow_table WHERE s.a IS NULL").fetchall() == [(2,)]

    def test_struct_in_stays_above(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _s (s STRUCT(a INTEGER))")
        duckdb_cursor.execute("INSERT INTO _s VALUES ({'a': 1}), ({'a': 2}), ({'a': 42}), (NULL)")
        duckdb_cursor.register("arrow_table", to_arrow_table(duckdb_cursor.table("_s")))
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE s.a IN (1, 42, 99)")
        assert duckdb_cursor.execute("SELECT count(*) FROM arrow_table WHERE s.a IN (1, 42, 99)").fetchall() == [(2,)]

    def test_not_in_stays_above(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE _t AS SELECT range AS a FROM range(1000)")
        duckdb_cursor.register("arrow_table", to_arrow_table(duckdb_cursor.table("_t")))
        _assert_kept_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a NOT IN (1, 5, 100)")
        assert duckdb_cursor.execute("SELECT count(*) FROM arrow_table WHERE a NOT IN (1, 5, 100)").fetchall() == [
            (997,)
        ]

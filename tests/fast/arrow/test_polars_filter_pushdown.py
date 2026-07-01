# ruff: noqa: F841
"""Filter pushdown tests for polars-backed scans.

What's tested here:

* **Comparison correctness** across every supported type, run against two
  polars factories (`pl.LazyFrame` and `pl.DataFrame`) — see below.
* **Optimizer pushdown decisions** — same EXPLAIN-based checks as the pyarrow
  test file; verified that polars LazyFrame and DataFrame both bind through
  ``arrow_scan`` and render as ``ARROW_SCAN`` in the plan.
* **Special filter shapes** — same coverage as the pyarrow file: IN, LIKE,
  CAST temporal, ``IS DISTINCT FROM NULL`` inside OR, NaN ordering, struct
  extraction (single and multi-level), OptionalFilter, top-N dynamic, join.
* **Produce-path** tests specific to the LazyFrame branch in
  ``arrow_array_stream.cpp`` (cached materialised table, repeated filtered
  scans, empty result, etc.).
* **Regressions** and **canaries** mirroring the pyarrow file.

Two factories cover the two distinct C++ scan paths that polars inputs
exercise:

* ``to_polars_lazyframe(rel) → pl.LazyFrame`` — routes through
  ``PyArrowObjectType::PolarsLazyFrame`` and is handled by
  ``polars_filter_pushdown.cpp``. This is the path with the current gaps.
* ``to_polars_dataframe(rel) → pl.DataFrame`` — converted to a pyarrow Table
  via ``.to_arrow()`` at registration and then handled by
  ``pyarrow_filter_pushdown.cpp``. Acts as a "reference implementation":
  whatever passes here defines the expected polars-side behaviour.

The pre-Phase-3 expectation is that ``dataframe`` factory cases mostly pass
while ``lazyframe`` factory cases fail wherever the polars walker has a gap
(EXPRESSION_FILTER, deep struct, decimal IN, OR-bail). Those failures drive
the C++ implementation phases.
"""

from __future__ import annotations

import math
import re

import pytest
from _pushdown_helpers import (
    COMPARABLE_TYPES,
    COMPARISON_CASES,
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

import duckdb

pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")


# ===========================================================================
# Conversion factories — polars side
# ===========================================================================


def to_polars_lazyframe(rel):
    return rel.pl().lazy()


def to_polars_dataframe(rel):
    return rel.pl()


POLARS_FACTORIES = [
    pytest.param(to_polars_lazyframe, id="lazyframe"),
    pytest.param(to_polars_dataframe, id="dataframe"),
]


# ===========================================================================
# 1. Comparison correctness across types
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
@pytest.mark.parametrize("case", COMPARABLE_TYPES, ids=lambda c: c.id)
@pytest.mark.parametrize(("predicate_tpl", "expected"), COMPARISON_CASES)
def test_comparisons(duckdb_cursor, factory, case, predicate_tpl, expected):
    """Each (type, factory, predicate) tuple produces the expected row count."""
    _make_typed_table(duckdb_cursor, factory, case)
    predicate = predicate_tpl.format(low=case.low, mid=case.mid, high=case.high)
    assert _count(duckdb_cursor, predicate) == expected


# BOOL has no ordering, so it gets its own tiny suite.
@pytest.mark.parametrize("factory", POLARS_FACTORIES)
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


# Integer boundary values are worth a separate test because GetScalar / FromValue
# has to coerce each (DuckDB Value) -> (target backend scalar) at the limit.
@pytest.mark.parametrize("factory", POLARS_FACTORIES)
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
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestOrPushdownDecisions:
    """Same-column ORs push; multi-column ORs and OR-with-AND-child don't."""

    @pytest.fixture(autouse=True)
    def _arrow_table(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _o (a INTEGER, b INTEGER, c INTEGER)")
        duckdb_cursor.execute("INSERT INTO _o VALUES (1,1,1),(10,10,10),(100,10,100),(NULL,NULL,NULL)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_o")))

    def test_single_column_or_pushes(self, duckdb_cursor, factory):
        assert _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR a = 10")

    def test_single_column_or_with_and_does_not_push(self, duckdb_cursor, factory):
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR (a > 3 AND a < 5)")

    def test_multiple_or_terms_push(self, duckdb_cursor, factory):
        assert _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR a > 3 OR a < 5")

    def test_or_with_not_equal_pushes(self, duckdb_cursor, factory):
        assert _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a != 1 OR a > 3 OR a < 2")

    def test_multi_column_or_does_not_push(self, duckdb_cursor, factory):
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR b = 2 AND (a > 3 OR b < 5)")


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestStringOrSpecifics:
    """VARCHAR has stricter OR-pushdown rules than numeric types."""

    @pytest.fixture(autouse=True)
    def _arrow_table(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _v (a VARCHAR, b VARCHAR, c VARCHAR)")
        duckdb_cursor.execute(
            "INSERT INTO _v VALUES ('1','1','1'),('10','10','10'),('100','10','100'),(NULL,NULL,NULL)"
        )
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_v")))

    def test_string_range_or_pushes(self, duckdb_cursor, factory):
        assert _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a >= '1' OR a <= '10'")

    def test_or_with_is_null_does_not_push(self, duckdb_cursor, factory):
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a IS NULL OR a = '1'")

    def test_or_with_is_not_null_does_not_push(self, duckdb_cursor, factory):
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a IS NOT NULL OR a = '1'")

    def test_or_with_like_does_not_push(self, duckdb_cursor, factory):
        assert not _was_pushed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a LIKE '1%' OR a = '10'")


# ===========================================================================
# 3. IN pushdown
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestInPushdown:
    """Small IN lowers to OR of equalities (CONJUNCTION_OR); large IN forces a real IN_FILTER."""

    def test_basic(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _i AS SELECT i AS a FROM range(10) t(i)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_i")))
        assert sorted(duckdb_cursor.execute("SELECT * FROM arrow_table WHERE a IN (2, 5, 7)").fetchall()) == [
            (2,),
            (5,),
            (7,),
        ]

    def test_large_in_list_does_not_hang(self, duckdb_cursor, factory):
        """A 200-element IN list forces the IN_FILTER path (not the OR-of-eq rewrite)."""
        duckdb_cursor.execute("CREATE TABLE _i AS SELECT i AS a FROM range(500) t(i)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_i")))
        in_list = ", ".join(str(i) for i in range(200))
        rows = duckdb_cursor.execute(f"SELECT count(*) FROM arrow_table WHERE a IN ({in_list})").fetchone()
        assert rows == (200,)

    def test_in_varchar(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _i AS SELECT 'str_' || i::VARCHAR AS s FROM range(10) t(i)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_i")))
        rows = sorted(duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s IN ('str_2', 'str_5')").fetchall())
        assert rows == [("str_2",), ("str_5",)]

    def test_in_float(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _i AS SELECT i::DOUBLE AS a FROM range(10) t(i)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_i")))
        rows = sorted(duckdb_cursor.execute("SELECT * FROM arrow_table WHERE a IN (2.0, 5.0)").fetchall())
        assert rows == [(2.0,), (5.0,)]

    def test_in_with_no_nulls_in_list(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _i AS SELECT i AS a FROM range(10) t(i)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_i")))
        assert sorted(duckdb_cursor.execute("SELECT * FROM arrow_table WHERE a IN (1, 2, 3)").fetchall()) == [
            (1,),
            (2,),
            (3,),
        ]

    def test_in_with_null_in_list(self, duckdb_cursor, factory):
        """``a IN (NULL, …)`` returns no rows for the NULL entry; non-null matches survive."""
        duckdb_cursor.execute("CREATE TABLE _i AS SELECT * FROM (VALUES (1), (NULL), (2), (3)) t(a)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_i")))
        assert sorted(duckdb_cursor.execute("SELECT * FROM arrow_table WHERE a IN (1, NULL, 3)").fetchall()) == [
            (1,),
            (3,),
        ]

    def test_in_decimal_large_list(self, duckdb_cursor, factory):
        """Large IN list on a decimal column must force the IN_FILTER path and pass.

        For LazyFrame, the current C++ walker constructs a plain Python list of
        ``Decimal(...)`` values for ``pl.col(d).is_in(...)``. Polars infers a
        higher-precision dtype for the literal list and refuses to compare against
        the column's actual ``Decimal(precision, scale)``. The post-Phase-3
        PolarsBackend builds a typed Series matching the column to close this.
        """
        duckdb_cursor.execute("CREATE TABLE _i AS SELECT i::DECIMAL(18,4) AS d FROM range(500) t(i)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_i")))
        in_list = ", ".join(f"{i}.0000" for i in range(200))
        rows = duckdb_cursor.execute(f"SELECT count(*) FROM arrow_table WHERE d IN ({in_list})").fetchone()
        assert rows == (200,)


# ===========================================================================
# 4. NaN pushdown
# ===========================================================================
#
# DuckDB intentionally violates IEEE-754: NaN is the greatest value.
# Each backend has to translate that to its target operators (is_nan / lit).
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestNaNPushdown:
    """Six comparison operators against a NaN constant on a DOUBLE column."""

    @pytest.fixture(autouse=True)
    def _nan_arrow_table(self, duckdb_cursor, factory):
        duckdb_cursor.execute(
            "CREATE TABLE _n AS SELECT a::DOUBLE a FROM VALUES "
            "('inf'), ('nan'), ('0.34234'), ('34234234.00005'), ('-nan') t(a)"
        )
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_n")))

    @pytest.mark.parametrize(
        "op",
        ["=", "!=", "<", "<=", ">", ">="],
    )
    def test_nan_comparison_matches_duckdb(self, duckdb_cursor, factory, op):
        """Each NaN comparison through the arrow scan agrees with DuckDB's own answer."""
        q_arrow = f"SELECT count(*) FROM arrow_table WHERE a {op} 'NaN'::FLOAT"
        q_duck = f"SELECT count(*) FROM _n WHERE a {op} 'NaN'::FLOAT"
        assert duckdb_cursor.execute(q_arrow).fetchone() == duckdb_cursor.execute(q_duck).fetchone()


# ===========================================================================
# 5. Struct extract pushdown
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestOneLevelStruct:
    """``struct_extract`` chains build the path inside ``ResolveColumn``."""

    @pytest.fixture(autouse=True)
    def _one_level_struct(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _s (s STRUCT(a INTEGER, b BOOL))")
        duckdb_cursor.execute(
            "INSERT INTO _s VALUES "
            "({'a': 1, 'b': true}), ({'a': 2, 'b': false}), (NULL), "
            "({'a': 3, 'b': true}), ({'a': NULL, 'b': NULL})"
        )
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_s")))

    def test_one_level_comparison_is_pushed(self, duckdb_cursor, factory):
        plan = duckdb_cursor.execute("EXPLAIN SELECT * FROM arrow_table WHERE s.a < 2").fetchone()[1]
        assert re.search(r"struct_extract\(s,\s*'a'\)\s*<", plan)

    def test_one_level_comparison_correct(self, duckdb_cursor, factory):
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s.a < 2").fetchone()[0] == {"a": 1, "b": True}

    def test_one_level_and_correct(self, duckdb_cursor, factory):
        assert duckdb_cursor.execute("SELECT count(*) FROM arrow_table WHERE s.a < 3 AND s.b = true").fetchone() == (1,)


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestNestedStruct:
    """Multi-level ``struct_extract`` chains.

    Polars supports arbitrary depth via chained ``struct.field``; pyarrow uses
    tuple-path field references. Both are driven by the same ``ResolveColumn``
    recursion in the shared walker.
    """

    @pytest.fixture(autouse=True)
    def _nested_struct(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _n (s STRUCT(a STRUCT(b INTEGER, c BOOL), d STRUCT(e INTEGER, f VARCHAR)))")
        duckdb_cursor.execute(
            "INSERT INTO _n VALUES "
            "({'a': {'b': 1, 'c': false}, 'd': {'e': 2, 'f': 'foo'}}), "
            "(NULL), "
            "({'a': {'b': 3, 'c': true}, 'd': {'e': 4, 'f': 'bar'}}), "
            "({'a': {'b': NULL, 'c': true}, 'd': {'e': 5, 'f': 'qux'}}), "
            "({'a': NULL, 'd': NULL})"
        )
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_n")))

    def test_nested_two_level_correct(self, duckdb_cursor, factory):
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s.a.b < 2").fetchone()[0] == {
            "a": {"b": 1, "c": False},
            "d": {"e": 2, "f": "foo"},
        }

    def test_nested_and_across_branches(self, duckdb_cursor, factory):
        assert duckdb_cursor.execute(
            "SELECT count(*) FROM arrow_table WHERE s.a.c = true AND s.d.e = 5"
        ).fetchone() == (1,)

    def test_nested_varchar_comparison(self, duckdb_cursor, factory):
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s.d.f = 'bar'").fetchone()[0] == {
            "a": {"b": 3, "c": True},
            "d": {"e": 4, "f": "bar"},
        }


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestThreeLevelStruct:
    """Three-level ``struct_extract`` — verifies depth-N support, not just one or two."""

    @pytest.fixture(autouse=True)
    def _three_level(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _3 (s STRUCT(a STRUCT(b STRUCT(c INTEGER, d VARCHAR), e INTEGER), f BOOL))")
        duckdb_cursor.execute(
            "INSERT INTO _3 VALUES "
            "({'a': {'b': {'c': 1, 'd': 'one'}, 'e': 10}, 'f': true}), "
            "({'a': {'b': {'c': 2, 'd': 'two'}, 'e': 20}, 'f': false}), "
            "({'a': {'b': {'c': 3, 'd': 'three'}, 'e': 30}, 'f': true})"
        )
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_3")))

    def test_three_level_comparison_correct(self, duckdb_cursor, factory):
        rows = duckdb_cursor.execute("SELECT count(*) FROM arrow_table WHERE s.a.b.c > 1").fetchone()
        assert rows == (2,)

    def test_three_level_varchar_correct(self, duckdb_cursor, factory):
        rows = duckdb_cursor.execute("SELECT count(*) FROM arrow_table WHERE s.a.b.d = 'two'").fetchone()
        assert rows == (1,)


# ===========================================================================
# 6. LIKE pushdown
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestLikePushdown:
    """LIKE pushdown decomposes into comparison filters.

    A LIKE with a fixed prefix decomposes into ``>= prefix AND < prefix+1``; a
    constant LIKE (no wildcards) decomposes into ``=``. Both produce regular
    comparison ExpressionFilters that the walker handles.
    """

    @pytest.fixture(autouse=True)
    def _s_arrow_table(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _l AS SELECT 'str_' || lpad(i::VARCHAR, 4, '0') AS s FROM range(100) t(i)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_l")))

    def test_like_with_prefix_is_pushed(self, duckdb_cursor, factory):
        assert _was_pushed(duckdb_cursor, "SELECT s FROM arrow_table WHERE s LIKE 'str_001%'")

    def test_like_constant_is_pushed(self, duckdb_cursor, factory):
        assert _was_pushed(duckdb_cursor, "SELECT s FROM arrow_table WHERE s LIKE 'str_0042'")

    def test_like_with_prefix_correct(self, duckdb_cursor, factory):
        rows = duckdb_cursor.execute("SELECT s FROM arrow_table WHERE s LIKE 'str_001%' ORDER BY s").fetchall()
        assert rows == [(f"str_001{d}",) for d in "0123456789"]


# ===========================================================================
# 7. CAST temporal pushdown
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestTemporalCastPushdown:
    """``CAST(timestamp_col AS DATE) = …`` pushes an optional relaxed range filter."""

    def test_cast_timestamp_to_date_is_pushed(self, duckdb_cursor, factory):
        duckdb_cursor.execute(
            "CREATE TABLE _ct AS "
            "SELECT TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i) SECOND AS ts FROM range(86400) t(i)"
        )
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_ct")))
        assert _was_pushed(
            duckdb_cursor,
            "SELECT * FROM arrow_table WHERE CAST(ts AS DATE) = DATE '2024-01-01'",
        )


# ===========================================================================
# 8. IS DISTINCT FROM NULL inside OR
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestDistinctFromNullOrPushdown:
    """``IS DISTINCT FROM NULL OR ...`` produces an IS_NOT_NULL ExpressionFilter."""

    @pytest.fixture(autouse=True)
    def _with_nulls(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _d AS SELECT * FROM (VALUES (1), (NULL), (5), (10)) t(a)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_d")))

    def test_distinct_from_null_or_eq_is_pushed(self, duckdb_cursor, factory):
        assert _was_pushed(
            duckdb_cursor,
            "SELECT a FROM arrow_table WHERE a IS DISTINCT FROM NULL OR a = 5",
        )

    def test_distinct_from_null_or_eq_correct(self, duckdb_cursor, factory):
        rows = duckdb_cursor.execute(
            "SELECT a FROM arrow_table WHERE a IS DISTINCT FROM NULL OR a = 5 ORDER BY a"
        ).fetchall()
        assert rows == [(1,), (5,), (10,)]

    def test_not_distinct_from_null_or_eq_correct(self, duckdb_cursor, factory):
        rows = duckdb_cursor.execute(
            "SELECT a FROM arrow_table WHERE a IS NOT DISTINCT FROM NULL OR a = 5 ORDER BY a NULLS FIRST"
        ).fetchall()
        assert rows == [(None,), (5,)]


# ===========================================================================
# 9. Special-shape filters: optional, dynamic top-N, join
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestOptionalFilter:
    """An OptionalFilter is allowed to silently fail.

    The engine reapplies it above the scan. The result must remain correct.
    """

    def test_no_crash_correct_result(self, factory):
        con = duckdb.connect()
        con.execute(
            "CREATE TABLE _t AS SELECT * FROM (VALUES "
            "('id', 100), ('product_code', 100), ('price', 100), ('quantity', 45), ('category', 5), "
            "('is_available', 3), ('rating', 6), ('discount', 39), ('color', 5)) t(column_name, cardinality)"
        )
        cardinality_table = factory(con.table("_t"))
        con.register("cardinality_table", cardinality_table)
        result = con.execute(
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

    def test_top_n_nulls_first_includes_min(self, factory):
        """ORDER BY x ASC NULLS FIRST LIMIT 1 pushes OPTIONAL(IS_NULL OR DYNAMIC_FILTER) into the scan.

        The OR branch must not be partially translated: dropping the
        untranslatable DYNAMIC_FILTER child would leave just IS_NULL and
        silently discard every non-null row. See pyarrow_filter_pushdown
        sibling regression test in test_filter_pushdown.py.
        """
        lf = pl.LazyFrame({"x": [3, 1, 2]})
        result = duckdb.sql("SELECT * FROM lf ORDER BY x ASC NULLS FIRST LIMIT 1").fetchall()
        assert result == [(1,)]


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestDynamicFilter:
    """The top-N optimization installs a dynamic filter.

    The walker returns ``py::none()`` for those (DuckDB applies them above
    the scan).
    """

    def test_topn_dynamic_filter(self, duckdb_cursor, factory):
        duckdb_cursor.execute(
            "CREATE TABLE _t AS SELECT * FROM (VALUES "
            "(3), (24), (234), (234), (234), (234), (234), (234), (234), (45), (2), (5), (2), (45)) t(a)"
        )
        duckdb_cursor.register("t", factory(duckdb_cursor.table("_t")))
        rows = duckdb_cursor.sql("SELECT a FROM t ORDER BY a LIMIT 11").fetchall()
        assert len(rows) == 11


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestJoinFilterPushdown:
    """Join pushdown between two polars-backed scans must produce the right count.

    (The join's runtime filters take a separate code path that doesn't reach
    the static-filter walker — see TestCanaries below.)
    """

    def test_two_polars_tables(self, factory):
        con = duckdb.connect()
        con.execute("CREATE TABLE probe AS SELECT range a FROM range(10000)")
        con.execute("CREATE TABLE build AS SELECT (random()*9999)::INT b FROM range(20)")
        con.register("probe_arrow", factory(con.table("probe")))
        con.register("build_arrow", factory(con.table("build")))
        assert con.execute("SELECT count(*) FROM probe_arrow, build_arrow WHERE a = b").fetchall() == [(20,)]


# ===========================================================================
# 10. Unsupported-type fallback (filter applied above the scan)
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestUnsupportedTypes:
    """``UHUGEINT`` filters must not crash.

    The filter is applied above the scan instead of being pushed down.
    """

    def test_uhugeint_single_filter(self, factory):
        con = duckdb.connect()
        con.execute(
            "CREATE TABLE t AS SELECT i::INTEGER a, i::VARCHAR b, i::UHUGEINT c, i::INTEGER d FROM range(5) tbl(i)"
        )
        con.register("arrow_tbl", factory(con.table("t")))
        assert con.execute("FROM arrow_tbl WHERE c = 3").fetchall() == [(3, "3", 3, 3)]

    def test_uhugeint_mixed_with_supported(self, factory):
        con = duckdb.connect()
        con.execute(
            "CREATE TABLE t AS SELECT i::INTEGER a, i::VARCHAR b, i::UHUGEINT c, i::INTEGER d FROM range(5) tbl(i)"
        )
        con.register("arrow_tbl", factory(con.table("t")))
        assert con.execute("FROM arrow_tbl WHERE c < 4 AND a > 2").fetchall() == [(3, "3", 3, 3)]
        assert con.execute("FROM arrow_tbl WHERE a > 2 AND c < 4 AND b = '3'").fetchall() == [(3, "3", 3, 3)]
        assert con.execute("FROM arrow_tbl WHERE a > 2 AND c < 4 AND b = '0'").fetchall() == []


# ===========================================================================
# 11. Produce-path interactions (LazyFrame branch in arrow_array_stream.cpp)
#
# These exercise the cache reuse and repeated-scan behaviour of the
# LazyFrame produce path. Only meaningful for the LazyFrame factory.
# ===========================================================================


class TestLazyFrameProducePath:
    """Behaviour specific to the LazyFrame branch in ``arrow_array_stream.cpp``."""

    def test_unfiltered_scan(self):
        con = duckdb.connect()
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        con.register("lf", lf)
        result = con.sql("SELECT * FROM lf").fetchall()
        assert result == [(1, 4), (2, 5), (3, 6)]

    def test_column_projection(self):
        con = duckdb.connect()
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        con.register("lf", lf)
        result = con.sql("SELECT a, c FROM lf").fetchall()
        assert result == [(1, 7), (2, 8), (3, 9)]

    def test_cached_dataframe_reuse(self):
        """Repeated unfiltered scans on a registered LazyFrame reuse the cached materialised table."""
        con = duckdb.connect()
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        con.register("my_lf", lf)
        r1 = con.sql("SELECT * FROM my_lf").fetchall()
        r2 = con.sql("SELECT * FROM my_lf").fetchall()
        assert r1 == r2 == [(1,), (2,), (3,)]

    def test_filtered_scan_not_cached(self):
        """Filtered scans collect a new DataFrame each time (not cached)."""
        con = duckdb.connect()
        lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        con.register("my_lf", lf)
        r1 = con.sql("SELECT * FROM my_lf WHERE a > 3").fetchall()
        r2 = con.sql("SELECT * FROM my_lf WHERE a < 3").fetchall()
        assert sorted(r1) == [(4,), (5,)]
        assert sorted(r2) == [(1,), (2,)]

    def test_empty_result(self):
        con = duckdb.connect()
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        con.register("lf", lf)
        result = con.sql("SELECT * FROM lf WHERE a > 100").fetchall()
        assert result == []


# ===========================================================================
# 12. Regressions
# ===========================================================================


class TestRegressions:
    """Bug-fix regressions specific to polars."""

    def test_nan_comparison_uses_is_nan(self):
        """NaN equality must produce ``is_nan()`` on the polars side, not literal-NaN equality."""
        lf = pl.LazyFrame({"a": [1.0, float("nan"), 3.0]})
        result = duckdb.sql("SELECT * FROM lf WHERE a = 'NaN'::DOUBLE").fetchall()
        assert len(result) == 1
        assert math.isnan(result[0][0])


# ===========================================================================
# 13. Canaries — behaviour we expect to change upstream
# ===========================================================================


class TestCanaries:
    """If any of these starts passing, the upstream behaviour changed."""

    @pytest.mark.xfail(reason="DuckDB does not push IS_NULL as a root TableFilter into arrow scans")
    def test_is_null_pushes_into_arrow_scan(self):
        con = duckdb.connect()
        lf = pl.LazyFrame({"a": [1, None, 3]})
        con.register("arrow_table", lf)
        assert _was_pushed(con, "SELECT a FROM arrow_table WHERE a IS NULL")

    @pytest.mark.xfail(reason="DuckDB does not push struct IS_NULL into arrow scans")
    def test_struct_is_null_pushes(self):
        con = duckdb.connect()
        lf = pl.LazyFrame({"s": [{"x": 1}, None, {"x": 3}]})
        con.register("arrow_table", lf)
        assert _was_pushed(con, "SELECT s FROM arrow_table WHERE s.x IS NULL")

    @pytest.mark.xfail(reason="filter_combiner does not currently rewrite struct IN to IN_FILTER")
    def test_struct_in_pushes(self):
        con = duckdb.connect()
        lf = pl.LazyFrame({"s": [{"x": 1}, {"x": 2}, {"x": 3}]})
        con.register("arrow_table", lf)
        assert _was_pushed(con, "SELECT s FROM arrow_table WHERE s.x IN (1, 2)")

    @pytest.mark.xfail(reason="Bloom filters never reach PolarsFilterPushdown::TransformFilter")
    def test_bloom_filter_reaches_walker(self):
        # If this ever flips, BLOOM_FILTER reached the walker and should be handled.
        con = duckdb.connect()
        con.execute("CREATE TABLE build AS SELECT (i*2)::BIGINT AS k FROM range(50000) t(i)")
        con.execute("CREATE TABLE probe_src AS SELECT i::BIGINT AS k FROM range(50000) t(i)")
        con.register("probe", to_polars_lazyframe(con.table("probe_src")))
        plan = con.execute("EXPLAIN SELECT count(*) FROM probe JOIN build USING (k)").fetchone()[1]
        block = _arrow_scan_block(plan)
        assert block is not None
        # If a bloom filter reaches the walker, it would show up as a Filters: line.
        assert "Filters:" in block

# ruff: noqa: F841
"""Filter pushdown tests for polars-backed scans.

What's tested here:

* **Comparison correctness** across every supported type, run against two
  polars factories (`pl.LazyFrame` and `pl.DataFrame`) — see below.
* **Filter placement**: EXPLAIN-based checks that a polars LazyFrame takes every
  filter it can evaluate exactly into its own scan, while a polars DataFrame,
  converted to a pyarrow Table, refuses filters and keeps them above the scan.
* **Special filter shapes** — same coverage as the pyarrow file: IN, LIKE,
  CAST temporal, ``IS DISTINCT FROM NULL`` inside OR, NaN ordering, struct
  extraction (single and multi-level), OptionalFilter, top-N dynamic, join.
* **Produce-path** tests specific to a LazyFrame (repeated filtered scans,
  empty result, etc.).
* **Regressions** mirroring the pyarrow file.

Two factories cover the two ways polars inputs are scanned:

* ``to_polars_lazyframe(rel) → pl.LazyFrame``: the scan offers it each
  filter, and it applies the ones it accepts through polars before collecting.
* ``to_polars_dataframe(rel) → pl.DataFrame``: converted to a pyarrow Table
  via ``.to_arrow()`` at registration, which refuses every filter.
"""

from __future__ import annotations

import datetime
import math

import pytest
from _pushdown_helpers import (
    COMPARABLE_TYPES,
    COMPARISON_CASES,
    _find_arrow_scan_nodes,
    assert_query_keeps_filter_above,
    explain_physical_plan,
    to_polars_dataframe,
    to_polars_lazyframe,
)
from _pushdown_helpers import (
    count as _count,
)
from _pushdown_helpers import (
    make_typed_table as _make_typed_table,
)

import duckdb

pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")


POLARS_FACTORIES = [
    pytest.param(to_polars_lazyframe, id="lazyframe"),
    pytest.param(to_polars_dataframe, id="dataframe"),
]


def _assert_pushed(con, query):
    """The only Arrow scan of `query` takes its filter, with no FILTER operator above it."""
    [(scan, has_filter_above)] = _find_arrow_scan_nodes(explain_physical_plan(con, query))
    assert scan["extra_info"].get("Filters"), "the filter was not pushed into the Arrow scan"
    assert not has_filter_above, "a FILTER operator stayed above the Arrow scan"


def _assert_rows_match_native(con, native, where):
    """The rows the Arrow scan returns under `where` are the native table's, and there are some."""
    rows = con.execute(f"SELECT * FROM arrow_table WHERE {where} ORDER BY ALL").fetchall()
    assert rows == con.execute(f"SELECT * FROM {native} WHERE {where} ORDER BY ALL").fetchall()
    assert rows, where


def _assert_placed(con, query, factory):
    """A LazyFrame takes an exactly translatable filter into its scan; a DataFrame, read as a Table, refuses it."""
    if factory is to_polars_lazyframe:
        _assert_pushed(con, query)
    else:
        assert_query_keeps_filter_above(con, query)


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


# Integer boundary values are worth a separate test because each constant has to
# reach the polars literal unchanged at the limit.
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
# 2. OR shapes
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestOrPushdownDecisions:
    """A LazyFrame takes every OR of exact comparisons; a DataFrame keeps them above the scan."""

    @pytest.fixture(autouse=True)
    def _arrow_table(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _o (a INTEGER, b INTEGER, c INTEGER)")
        # Rows 4 to 8 let each branch of the compound ORs below select rows the other does not, NULLs among them.
        duckdb_cursor.execute(
            "INSERT INTO _o VALUES (1,1,1),(10,10,10),(100,10,100),(NULL,NULL,NULL),"
            "(4,NULL,4),(4,7,5),(7,2,2),(NULL,2,5),(2,2,9),(1,2,NULL)"
        )
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_o")))

    def _assert_rows_match_native(self, con, where, branches):
        rows = con.execute(f"SELECT a, b, c FROM arrow_table WHERE {where} ORDER BY ALL").fetchall()
        assert rows == con.execute(f"SELECT a, b, c FROM _o WHERE {where} ORDER BY ALL").fetchall()
        selected = [
            set(con.execute(f"SELECT a, b, c FROM _o WHERE ({branch}) IS TRUE").fetchall()) for branch in branches
        ]
        for index, mine in enumerate(selected):
            others = set().union(*(other for position, other in enumerate(selected) if position != index))
            assert mine - others, f"no row is selected by {branches[index]!r} alone"

    def test_single_column_or(self, duckdb_cursor, factory):
        _assert_placed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR a = 10", factory)
        _assert_rows_match_native(duckdb_cursor, "_o", "a = 1 OR a = 10")

    def test_single_column_or_with_and(self, duckdb_cursor, factory):
        _assert_placed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR (a > 3 AND a < 5)", factory)
        self._assert_rows_match_native(duckdb_cursor, "a = 1 OR (a > 3 AND a < 5)", ["a = 1", "a > 3 AND a < 5"])

    def test_multiple_or_terms(self, duckdb_cursor, factory):
        _assert_placed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR a > 3 OR a < 5", factory)
        _assert_rows_match_native(duckdb_cursor, "_o", "a = 1 OR a > 3 OR a < 5")

    def test_or_with_not_equal(self, duckdb_cursor, factory):
        _assert_placed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a != 1 OR a > 3 OR a < 2", factory)
        _assert_rows_match_native(duckdb_cursor, "_o", "a != 1 OR a > 3 OR a < 2")

    def test_multi_column_or(self, duckdb_cursor, factory):
        _assert_placed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a = 1 OR b = 2 AND (a > 3 OR b < 5)", factory)
        self._assert_rows_match_native(
            duckdb_cursor, "a = 1 OR b = 2 AND (a > 3 OR b < 5)", ["a = 1", "b = 2 AND (a > 3 OR b < 5)"]
        )


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestStringOrSpecifics:
    """A LazyFrame takes VARCHAR ORs of exact comparisons and NULL checks, never one holding a LIKE."""

    @pytest.fixture(autouse=True)
    def _arrow_table(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _v (a VARCHAR, b VARCHAR, c VARCHAR)")
        duckdb_cursor.execute(
            "INSERT INTO _v VALUES ('1','1','1'),('10','10','10'),('100','10','100'),(NULL,NULL,NULL)"
        )
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_v")))

    def test_string_range_or(self, duckdb_cursor, factory):
        _assert_placed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a >= '1' OR a <= '10'", factory)
        _assert_rows_match_native(duckdb_cursor, "_v", "a >= '1' OR a <= '10'")

    def test_or_with_is_null(self, duckdb_cursor, factory):
        _assert_placed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a IS NULL OR a = '1'", factory)
        _assert_rows_match_native(duckdb_cursor, "_v", "a IS NULL OR a = '1'")

    def test_or_with_is_not_null(self, duckdb_cursor, factory):
        _assert_placed(duckdb_cursor, "SELECT * FROM arrow_table WHERE a IS NOT NULL OR a = '1'", factory)
        _assert_rows_match_native(duckdb_cursor, "_v", "a IS NOT NULL OR a = '1'")

    def test_or_with_like_stays_above(self, duckdb_cursor, factory):
        assert_query_keeps_filter_above(duckdb_cursor, "SELECT * FROM arrow_table WHERE a LIKE '1%' OR a = '10'")


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

        For a plain Python list of ``Decimal(...)`` values in ``pl.col(d).is_in(...)``,
        polars infers a higher-precision dtype for the literal list and refuses to
        compare against the column's actual ``Decimal(precision, scale)``, so the
        list has to be built with the column's own dtype.
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
    """A LazyFrame takes a comparison on a struct field; a DataFrame keeps it above the scan."""

    @pytest.fixture(autouse=True)
    def _one_level_struct(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _s (s STRUCT(a INTEGER, b BOOL))")
        duckdb_cursor.execute(
            "INSERT INTO _s VALUES "
            "({'a': 1, 'b': true}), ({'a': 2, 'b': false}), (NULL), "
            "({'a': 3, 'b': true}), ({'a': NULL, 'b': NULL})"
        )
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_s")))

    def test_one_level_comparison(self, duckdb_cursor, factory):
        _assert_placed(duckdb_cursor, "SELECT * FROM arrow_table WHERE s.a < 2", factory)

    def test_one_level_comparison_correct(self, duckdb_cursor, factory):
        assert duckdb_cursor.execute("SELECT * FROM arrow_table WHERE s.a < 2").fetchone()[0] == {"a": 1, "b": True}

    def test_one_level_and_correct(self, duckdb_cursor, factory):
        assert duckdb_cursor.execute("SELECT count(*) FROM arrow_table WHERE s.a < 3 AND s.b = true").fetchone() == (1,)


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestNestedStruct:
    """Multi-level ``struct_extract`` chains.

    Polars supports arbitrary depth via chained ``struct.field``; pyarrow uses
    tuple-path field references. Both receive the same path: the column name, then
    the struct field names.
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
    constant LIKE (no wildcards) decomposes into ``=``. No source is offered
    either, so both stay in a FILTER above the scan.
    """

    @pytest.fixture(autouse=True)
    def _s_arrow_table(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _l AS SELECT 'str_' || lpad(i::VARCHAR, 4, '0') AS s FROM range(100) t(i)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_l")))

    def test_like_with_prefix_stays_above(self, duckdb_cursor, factory):
        assert_query_keeps_filter_above(duckdb_cursor, "SELECT s FROM arrow_table WHERE s LIKE 'str_001%'")

    def test_like_constant_stays_above(self, duckdb_cursor, factory):
        assert_query_keeps_filter_above(duckdb_cursor, "SELECT s FROM arrow_table WHERE s LIKE 'str_0042'")

    def test_like_with_prefix_correct(self, duckdb_cursor, factory):
        rows = duckdb_cursor.execute("SELECT s FROM arrow_table WHERE s LIKE 'str_001%' ORDER BY s").fetchall()
        assert rows == [(f"str_001{d}",) for d in "0123456789"]


# ===========================================================================
# 7. CAST temporal pushdown
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestTemporalCastPushdown:
    """``CAST(timestamp_col AS DATE) = …`` reaches the scan as an exact BETWEEN, which a LazyFrame takes."""

    def test_cast_timestamp_to_date(self, duckdb_cursor, factory):
        # The rows of the neighbouring days, the first instant of the next day among them, fail a bound that is
        # inclusive at the top or shifted.
        duckdb_cursor.execute(
            "CREATE TABLE _ct AS "
            "SELECT TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i) SECOND AS ts FROM range(86400) t(i) "
            "UNION ALL SELECT * FROM (VALUES (TIMESTAMP '2023-12-31 23:59:59'), (TIMESTAMP '2024-01-02 00:00:00'), "
            "(TIMESTAMP '2024-01-02 12:00:00')) t(ts)"
        )
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_ct")))
        query = "SELECT * FROM arrow_table WHERE CAST(ts AS DATE) = DATE '2024-01-01'"
        _assert_placed(duckdb_cursor, query, factory)
        assert duckdb_cursor.execute(query.replace("*", "count(*)")).fetchall() == [(86400,)]
        _assert_rows_match_native(duckdb_cursor, "_ct", "CAST(ts AS DATE) = DATE '2024-01-02'")
        assert duckdb_cursor.execute(
            "SELECT ts FROM arrow_table WHERE CAST(ts AS DATE) = DATE '2024-01-02' ORDER BY ts"
        ).fetchall() == [(datetime.datetime(2024, 1, 2),), (datetime.datetime(2024, 1, 2, 12),)]


# ===========================================================================
# 8. IS DISTINCT FROM NULL inside OR
# ===========================================================================


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestDistinctFromNullOrPushdown:
    """``IS DISTINCT FROM NULL OR ...`` reaches the scan as ``a IS NOT NULL OR a = 5``, which a LazyFrame takes."""

    @pytest.fixture(autouse=True)
    def _with_nulls(self, duckdb_cursor, factory):
        duckdb_cursor.execute("CREATE TABLE _d AS SELECT * FROM (VALUES (1), (NULL), (5), (10)) t(a)")
        duckdb_cursor.register("arrow_table", factory(duckdb_cursor.table("_d")))

    def test_distinct_from_null_or_eq(self, duckdb_cursor, factory):
        _assert_placed(duckdb_cursor, "SELECT a FROM arrow_table WHERE a IS DISTINCT FROM NULL OR a = 5", factory)

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
            "SELECT * FROM cardinality_table WHERE cardinality > 1 ORDER BY cardinality ASC, column_name ASC"
        ).fetchall()
        assert result == [
            ("is_available", 3),
            ("category", 5),
            ("color", 5),
            ("rating", 6),
            ("discount", 39),
            ("quantity", 45),
            ("id", 100),
            ("price", 100),
            ("product_code", 100),
        ]

    def test_top_n_nulls_first_includes_min(self, duckdb_cursor, factory):
        """ORDER BY x ASC NULLS FIRST LIMIT 1 filters on IS_NULL OR DYNAMIC_FILTER.

        The OR must not be partially applied: dropping the DYNAMIC_FILTER
        child would leave just IS_NULL and silently discard every non-null
        row. See the sibling regression test for pyarrow in test_filter_pushdown.py.
        """
        duckdb_cursor.register("src", factory(duckdb_cursor.sql("SELECT * FROM (VALUES (3), (1), (2)) t(x)")))
        result = duckdb_cursor.sql("SELECT * FROM src ORDER BY x ASC NULLS FIRST LIMIT 1").fetchall()
        assert result == [(1,)]


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestDynamicFilter:
    """The top-N optimization installs a dynamic filter, which DuckDB applies itself."""

    def test_topn_dynamic_filter(self, duckdb_cursor, factory):
        duckdb_cursor.execute(
            "CREATE TABLE _t AS SELECT * FROM (VALUES "
            "(3), (24), (234), (234), (234), (234), (234), (234), (234), (45), (2), (5), (2), (45)) t(a)"
        )
        duckdb_cursor.register("t", factory(duckdb_cursor.table("_t")))
        rows = duckdb_cursor.sql("SELECT a FROM t ORDER BY a LIMIT 11").fetchall()
        assert len(rows) == 11
        assert rows == duckdb_cursor.sql("SELECT a FROM _t ORDER BY a LIMIT 11").fetchall()


@pytest.mark.parametrize("factory", POLARS_FACTORIES)
class TestJoinFilterPushdown:
    """Joins between polars-backed scans must produce the right count.

    A hash join's runtime filters are applied by DuckDB, never offered to the polars source.
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
# 11. Repeated scans of a LazyFrame
#
# These exercise scanning one LazyFrame several times, with and without
# filters. Only meaningful for the LazyFrame factory.
# ===========================================================================


class TestLazyFrameProducePath:
    """Behaviour specific to scanning a LazyFrame."""

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
        """Repeated unfiltered scans on a registered LazyFrame return the same rows."""
        con = duckdb.connect()
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        con.register("my_lf", lf)
        r1 = con.sql("SELECT * FROM my_lf").fetchall()
        r2 = con.sql("SELECT * FROM my_lf").fetchall()
        assert r1 == r2 == [(1,), (2,), (3,)]

    def test_filtered_scan_not_cached(self):
        """Each filtered scan collects the LazyFrame with its own filter."""
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

    @pytest.mark.parametrize("op", ["=", "!=", "<", "<=", ">", ">="])
    def test_finite_constant_includes_nan_rows(self, duckdb_cursor, op):
        """A finite constant against a column holding NaN gives the engine's rows through polars too.

        DuckDB orders NaN as the greatest value, and so does polars, so a comparison the LazyFrame takes
        keeps the NaN rows the engine keeps.
        """
        duckdb_cursor.execute(
            "CREATE TABLE _pn AS SELECT a::DOUBLE a FROM VALUES "
            "('inf'), ('nan'), ('0.34234'), ('34234234.00005'), ('-nan') t(a)"
        )
        lf = to_polars_lazyframe(duckdb_cursor.table("_pn"))
        duckdb_cursor.register("arrow_table", lf)
        rows_polars = duckdb_cursor.execute(f"SELECT a FROM arrow_table WHERE a {op} 4.0").fetchall()
        rows_duck = duckdb_cursor.execute(f"SELECT a FROM _pn WHERE a {op} 4.0").fetchall()

        # NaN-safe row-set comparison: NaN != NaN, so bucket NaNs by count and sort the finite rows.
        def summarize(rows):
            vals = [r[0] for r in rows]
            return sum(1 for v in vals if v != v), sorted(v for v in vals if v == v)

        assert summarize(rows_polars) == summarize(rows_duck)


# ===========================================================================
# 13. NULL checks and struct IN on a LazyFrame, and a join over one
# ===========================================================================


class TestLazyFrameTakesNullChecksAndStructIn:
    """A LazyFrame takes IS NULL and an IN list, on a column or a struct field, into its scan."""

    @pytest.mark.parametrize(
        ("frame", "query"),
        [
            pytest.param({"a": [1, None, 3]}, "SELECT a FROM {} WHERE a IS NULL", id="is_null"),
            pytest.param({"s": [{"x": 1}, None, {"x": 3}]}, "SELECT s FROM {} WHERE s.x IS NULL", id="struct_is_null"),
            pytest.param({"s": [{"x": 1}, {"x": 2}, {"x": 3}]}, "SELECT s FROM {} WHERE s.x IN (1, 2)", id="struct_in"),
        ],
    )
    def test_is_pushed_and_matches_native(self, frame, query):
        con = duckdb.connect()
        con.register("arrow_table", pl.LazyFrame(frame))
        con.execute("CREATE TABLE native_t AS SELECT * FROM arrow_table")
        _assert_pushed(con, query.format("arrow_table"))
        assert sorted(con.execute(query.format("arrow_table")).fetchall(), key=repr) == sorted(
            con.execute(query.format("native_t")).fetchall(), key=repr
        )


class TestLazyFrameJoin:
    """DuckDB applies a hash join's runtime filters itself, so a join over a LazyFrame matches a native join."""

    def test_join_with_runtime_filters_matches_native(self):
        con = duckdb.connect()
        con.execute("CREATE TABLE build AS SELECT (i*2)::BIGINT AS k FROM range(50000) t(i)")
        con.execute("CREATE TABLE probe_src AS SELECT i::BIGINT AS k FROM range(50000) t(i)")
        con.register("probe", to_polars_lazyframe(con.table("probe_src")))
        query = "SELECT count(*), min(k), max(k) FROM {} JOIN build USING (k)"
        assert con.execute(query.format("probe")).fetchall() == con.execute(query.format("probe_src")).fetchall()

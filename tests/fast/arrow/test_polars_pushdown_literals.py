import datetime
import decimal
import functools
import itertools
import json
import math
import operator
import subprocess
import sys
import textwrap
from zoneinfo import ZoneInfo

import pytest
from packaging.version import parse as parse_version

import duckdb

pl = pytest.importorskip("polars")
pytest.importorskip("pyarrow")
pl_testing = pytest.importorskip("polars.testing")

from duckdb import polars_io  # noqa: E402
from duckdb.polars_io import _pl_tree_to_sql, _predicate_to_expression  # noqa: E402

EPOCH = datetime.datetime(1970, 1, 1)
EPOCH_UTC = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
ORDERINGS = (operator.eq, operator.lt, operator.le, operator.gt, operator.ge)
POLARS_UNITS = ("ms", "us", "ns")
UNIT_RESOLUTION = {"ms": 0, "us": 1, "ns": 2}
COLUMN_UNITS = {"TIMESTAMP_S": "ms", "TIMESTAMP_MS": "ms", "TIMESTAMP": "us", "TIMESTAMP_NS": "ns"}
SERIALIZED_UNITS = {"Milliseconds": "ms", "Microseconds": "us", "Nanoseconds": "ns"}
POLARS_2 = parse_version(pl.__version__) >= parse_version("2")
INTEGER_RANGES = {
    "TINYINT": (-(2**7), 2**7 - 1),
    "SMALLINT": (-(2**15), 2**15 - 1),
    "INTEGER": (-(2**31), 2**31 - 1),
    "BIGINT": (-(2**63), 2**63 - 1),
    "UTINYINT": (0, 2**8 - 1),
    "USMALLINT": (0, 2**16 - 1),
    "UINTEGER": (0, 2**32 - 1),
    "UBIGINT": (0, 2**64 - 1),
}
ZONE_MISMATCHES = [("America/New_York", "UTC"), ("UTC", "Asia/Tokyo"), ("America/New_York", "Asia/Tokyo")]
ZONE_MISMATCH_ROWS = """
    INSERT INTO t VALUES
        ('2024-01-01 10:00:00+00'), ('2024-01-01 12:00:00+00'), ('2024-01-01 14:00:00+00'), (NULL)
"""


class Pushdowns(list):
    """One entry per conjunct the source translated or refused, and the rows of each batch it handed polars."""

    def __init__(self) -> None:
        super().__init__()
        self.rows = []


@pytest.fixture
def pushdowns(monkeypatch):
    # The rows alone cannot tell a pushed filter from one polars applied after the source declined it.
    seen = Pushdowns()
    to_sql_expression = polars_io._tree_to_sql_expression
    from_arrow = pl.from_arrow

    def recording(tree, *schema):
        expression = to_sql_expression(tree, *schema)
        try:
            polars_io._predicate_to_sql(tree, *schema)
        except Exception as error:
            seen.append((tree, expression, type(error)))
        else:
            seen.append((tree, expression, None))
        return expression

    def counting(batch, *args, **kwargs):
        seen.rows.append(batch.num_rows)
        return from_arrow(batch, *args, **kwargs)

    monkeypatch.setattr(polars_io, "_tree_to_sql_expression", recording)
    monkeypatch.setattr(pl, "from_arrow", counting)
    return seen


def assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, *, pushed, polars_predicate=None):
    """`pushed` holds for every conjunct of a top level AND, or is "partly" when some reached the engine."""
    pushdowns.clear()
    pushdowns.rows.clear()
    actual = lazy_df.filter(predicate).collect()
    arrived = sum(pushdowns.rows)
    everything = lazy_df.collect()
    expected = everything.filter(predicate if polars_predicate is None else polars_predicate)
    pl_testing.assert_frame_equal(actual, expected)
    assert pushdowns, predicate
    states = [expression is not None for _, expression, _ in pushdowns]
    # A conjunct that translated proves nothing until the engine filters by it, so count the rows reaching polars
    if pushed == "partly":
        assert any(states), (predicate, pushdowns)
        assert not all(states), (predicate, pushdowns)
        translated = [
            polars_io._expression_from_tree(tree) for tree, expression, _ in pushdowns if expression is not None
        ]
        # The engine kept exactly the rows its conjuncts keep in polars, no more and no fewer
        assert arrived == everything.filter(functools.reduce(operator.and_, translated)).height, (predicate, arrived)
        assert arrived < everything.height, (predicate, arrived)
    else:
        assert all(state == pushed for state in states), (predicate, pushdowns)
        assert arrived == (actual.height if pushed else everything.height), (predicate, arrived)
    for tree, expression, error in pushdowns:
        if expression is None:
            assert error is NotImplementedError, (predicate, tree, error)
    return [tree for tree, _, _ in pushdowns]


def refused_columns(pushdowns):
    return sorted(name for tree, expression, _ in pushdowns if expression is None for name in column_names(tree))


def column_names(node):
    if isinstance(node, dict):
        if isinstance(node.get("Column"), str):
            yield node["Column"]
        for child in node.values():
            yield from column_names(child)
    elif isinstance(node, list):
        for child in node:
            yield from column_names(child)


def datetime_units(node):
    if isinstance(node, dict):
        scalar = node.get("Scalar")
        if isinstance(scalar, dict) and "Datetime" in scalar:
            yield SERIALIZED_UNITS[scalar["Datetime"][1]]
        for child in node.values():
            yield from datetime_units(child)
    elif isinstance(node, list):
        for child in node:
            yield from datetime_units(child)


def widened_to_range(compare, literal_unit, column_unit):
    # polars turns == against a finer column into is_between over the literal's span, which is not pushed down.
    return compare is operator.eq and UNIT_RESOLUTION[literal_unit] < UNIT_RESOLUTION[column_unit]


class TestDatetimeLiteralPushdown:
    # polars recasts the literal to the column's unit before the source sees it, so the literal unit varies only the
    # values and the shape of the predicate the source receives.
    @pytest.mark.parametrize("unit", POLARS_UNITS)
    @pytest.mark.parametrize("data_type", ["TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP", "TIMESTAMP_NS"])
    def test_naive_literal_matches_polars(self, data_type, unit, duckdb_cursor, pushdowns):
        # A naive comparison must not depend on the session zone.
        duckdb_cursor.execute("SET TimeZone = 'America/New_York'")
        duckdb_cursor.execute(f"CREATE TABLE t (a {data_type})")
        duckdb_cursor.execute(
            """
            INSERT INTO t VALUES
                ('1960-01-01 00:00:00.001'), ('2010-01-01 10:00:01'), ('2010-01-01 10:00:01.0005'),
                ('2010-01-01 10:00:01.001'), ('2020-03-01 10:00:01'), (NULL)
            """
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        column_unit = COLUMN_UNITS[data_type]
        for value in (
            datetime.datetime(1960, 1, 1, 0, 0, 0, 1000),
            datetime.datetime(2010, 1, 1, 10, 0, 1),
            datetime.datetime(2010, 1, 1, 10, 0, 1, 500),
            datetime.datetime(2010, 1, 1, 10, 0, 1, 1000),
        ):
            literal = pl.lit(value, dtype=pl.Datetime(unit))
            for compare in ORDERINGS:
                pushed = column_unit != "ns" and not widened_to_range(compare, unit, column_unit)
                [tree] = assert_pushdown_matches_polars(
                    lazy_df, pushdowns, compare(pl.col("a"), literal), pushed=pushed
                )
                assert set(datetime_units(tree)) == {column_unit}

    @pytest.mark.parametrize("unit", POLARS_UNITS)
    @pytest.mark.parametrize("zone", ["UTC", "America/New_York"])
    def test_zoned_literal_in_session_zone_matches_polars(self, zone, unit, duckdb_cursor, pushdowns):
        duckdb_cursor.execute(f"SET TimeZone = '{zone}'")
        duckdb_cursor.execute("CREATE TABLE t (a TIMESTAMPTZ)")
        duckdb_cursor.execute(
            """
            INSERT INTO t VALUES
                ('2024-01-01 10:00:00+00'), ('2024-01-01 12:00:00+00'), ('2024-01-01 12:00:00.0005+00'),
                ('2024-01-01 14:00:00+00'), (NULL)
            """
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for instant in (
            datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.timezone.utc),
            datetime.datetime(2024, 1, 1, 12, 0, 0, 500, tzinfo=datetime.timezone.utc),
        ):
            literal = pl.lit(instant.astimezone(ZoneInfo(zone)), dtype=pl.Datetime(unit, zone))
            for compare in ORDERINGS:
                pushed = not widened_to_range(compare, unit, "us")
                [tree] = assert_pushdown_matches_polars(
                    lazy_df, pushdowns, compare(pl.col("a"), literal), pushed=pushed
                )
                assert set(datetime_units(tree)) == {"us"}

    # polars itself rejects comparing a column and a literal of different zones, but it hands such a predicate to the
    # source unchecked and without recasting the literal, so the pushed filter is held to the instant the literal names.
    @pytest.mark.parametrize("unit", ["ms", "us"])
    @pytest.mark.parametrize(("session_zone", "literal_zone"), ZONE_MISMATCHES)
    def test_zoned_literal_outside_session_zone_compares_instants(
        self, session_zone, literal_zone, unit, duckdb_cursor, pushdowns
    ):
        duckdb_cursor.execute(f"SET TimeZone = '{session_zone}'")
        duckdb_cursor.execute("CREATE TABLE t (a TIMESTAMPTZ)")
        duckdb_cursor.execute(ZONE_MISMATCH_ROWS)
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        instant = datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.timezone.utc)
        literal = pl.lit(instant.astimezone(ZoneInfo(literal_zone)), dtype=pl.Datetime(unit, literal_zone))
        same_instant = pl.lit(instant.astimezone(ZoneInfo(session_zone)), dtype=pl.Datetime(unit, session_zone))
        for compare in ORDERINGS:
            [tree] = assert_pushdown_matches_polars(
                lazy_df,
                pushdowns,
                compare(pl.col("a"), literal),
                pushed=True,
                polars_predicate=compare(pl.col("a"), same_instant),
            )
            assert set(datetime_units(tree)) == {unit}

    # A nanosecond literal is not pushed down, so polars applies the filter and rejects the zone mismatch itself.
    @pytest.mark.parametrize(("session_zone", "literal_zone"), ZONE_MISMATCHES)
    def test_nanosecond_zoned_literal_outside_session_zone_raises_like_polars(
        self, session_zone, literal_zone, duckdb_cursor, pushdowns
    ):
        duckdb_cursor.execute(f"SET TimeZone = '{session_zone}'")
        duckdb_cursor.execute("CREATE TABLE t (a TIMESTAMPTZ)")
        duckdb_cursor.execute(ZONE_MISMATCH_ROWS)
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        instant = datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.timezone.utc)
        literal = pl.lit(instant.astimezone(ZoneInfo(literal_zone)), dtype=pl.Datetime("ns", literal_zone))
        for compare in ORDERINGS:
            pushdowns.clear()
            with pytest.raises(pl.exceptions.SchemaError):
                lazy_df.filter(compare(pl.col("a"), literal)).collect()
            assert [(expression is None, error) for _, expression, error in pushdowns] == [(True, NotImplementedError)]

    @pytest.mark.parametrize("unit", ["ms", "us"])
    @pytest.mark.parametrize("zone", [None, "America/New_York"])
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(datetime.datetime(2024, 6, 1, 12, 30, 1, 250000), id="2024"),
            pytest.param(datetime.datetime(1960, 1, 1, 0, 0, 0, 1000), id="pre_epoch"),
            pytest.param(datetime.datetime(9000, 1, 1, 0, 0, 0, 123000), id="year_9000"),
        ],
    )
    def test_literal_renders_its_value_and_type(self, value, zone, unit):
        if zone is None:
            literal = pl.lit(value, dtype=pl.Datetime(unit))
            expected_type = "TIMESTAMP"
            expected_us = (value - EPOCH) // datetime.timedelta(microseconds=1)
        else:
            aware = value.replace(tzinfo=ZoneInfo(zone))
            literal = pl.lit(aware, dtype=pl.Datetime(unit, zone))
            expected_type = "TIMESTAMP WITH TIME ZONE"
            expected_us = (aware - EPOCH_UTC) // datetime.timedelta(microseconds=1)
        sql = _pl_tree_to_sql(json.loads(literal.meta.serialize(format="json")))
        con = duckdb.connect()
        con.execute("SET TimeZone = 'Asia/Kathmandu'")
        assert con.execute(f"SELECT typeof({sql}), epoch_us({sql})").fetchone() == (expected_type, expected_us)

    def test_far_future_microsecond_literal_is_exact(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (a TIMESTAMP)")
        duckdb_cursor.execute("INSERT INTO t VALUES ('9000-01-01 00:00:00.123457'), ('9000-01-01 00:00:00.123458')")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        predicate = pl.col("a") == datetime.datetime(9000, 1, 1, 0, 0, 0, 123457)
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=True)


class TestNonFiniteFloatLiteralPushdown:
    @pytest.mark.parametrize(
        "compare", [pytest.param(op, id=op.__name__) for op in (operator.lt, operator.gt, operator.eq, operator.ne)]
    )
    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    @pytest.mark.parametrize("data_type", ["FLOAT", "DOUBLE"])
    def test_non_finite_literal_matches_polars(self, data_type, value, compare, duckdb_cursor, pushdowns):
        duckdb_cursor.execute(f"CREATE TABLE t (a {data_type})")
        duckdb_cursor.execute("INSERT INTO t VALUES (1), ('inf'), ('-inf'), ('nan'), (NULL), (1)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("a"), value), pushed=False)

    def test_non_finite_literal_inside_conjunction_matches_polars(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (a DOUBLE)")
        duckdb_cursor.execute("INSERT INTO t VALUES (1), ('inf'), ('-inf'), ('nan'), (NULL)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        predicate = (pl.col("a") > 0) & (pl.col("a") < float("inf"))
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed="partly")

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    @pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
    def test_non_finite_literal_is_not_pushed_down(self, dtype, value):
        assert _predicate_to_expression(pl.col("a") < pl.lit(value, dtype=dtype)) is None

    def test_untyped_non_finite_literal_is_not_pushed_down_without_assertions(self):
        untyped = json.loads((pl.col("a") < float("inf")).meta.serialize(format="json"))
        assert untyped["BinaryExpr"]["right"] == {"Literal": {"Dyn": {"Float": None}}}
        script = textwrap.dedent(
            """
            import sys

            import polars as pl

            from duckdb.polars_io import _predicate_to_expression

            if sys.flags.optimize == 0:
                sys.exit("assertions are not stripped")
            for value in (float("inf"), float("-inf"), float("nan")):
                expression = _predicate_to_expression(pl.col("a") < value)
                if expression is not None:
                    sys.exit(f"pushed down: {expression}")
            """
        )
        result = subprocess.run(
            [sys.executable, "-I", "-O", "-c", script], capture_output=True, text=True, timeout=60, check=False
        )
        assert result.returncode == 0, result.stderr


class TestTimeLiteralPushdown:
    def test_literal_finer_than_a_microsecond_is_not_pushed(self, duckdb_cursor, pushdowns):
        # A TIME holds microseconds, and truncated to them the literal reads as 12:00:00: as the whole predicate ==
        # keeps that row, which polars drops, and beside a refused conjunct < drops it, which polars keeps
        duckdb_cursor.execute("CREATE TABLE t (t TIME, s VARCHAR)")
        duckdb_cursor.execute(
            "INSERT INTO t VALUES ('12:00:00', 'x'), ('12:00:00.000001', 'x'), ('11:59:59.999999', 'x'), (NULL, 'x')"
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        nanosecond = pl.lit(12 * 3600 * 10**9 + 500, dtype=pl.Int64).cast(pl.Time)
        for compare in ORDERINGS:
            assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("t"), nanosecond), pushed=False)
            predicate = compare(pl.col("t"), nanosecond) & pl.col("s").str.contains("x")
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)
        microsecond = pl.lit(datetime.time(12, 0, 0, 1))
        for compare in ORDERINGS:
            assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("t"), microsecond), pushed=True)


class TestColumnCastPushdown:
    # A cast of the column, whether the user wrote it or polars inserted it to reconcile types, can truncate, turn an
    # overflow into NULL or change the type the comparison runs in, and the bare column carries none of that, so the
    # predicate stays with polars; only a cast whose target type holds every value of the column's type reads the same
    # as the bare column and is pushed.
    ROWS = """
        INSERT INTO t VALUES
            ('2024-01-01 12:00:00+00'), ('2024-01-01 12:00:00.0005+00'), ('2024-01-01 12:00:00.001+00'),
            ('2024-01-01 12:00:00.000999+00'), (NULL)
    """

    @pytest.mark.parametrize("session_zone", ["UTC", "America/New_York"])
    @pytest.mark.parametrize("zone", [None, "UTC", "Asia/Tokyo"])
    def test_column_cast_to_coarser_unit_matches_polars(self, zone, session_zone, duckdb_cursor, pushdowns):
        duckdb_cursor.execute(f"SET TimeZone = '{session_zone}'")
        duckdb_cursor.execute(f"CREATE TABLE t (a {'TIMESTAMP' if zone is None else 'TIMESTAMPTZ'})")
        duckdb_cursor.execute(self.ROWS)
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        noon = datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.timezone.utc)
        value = noon.replace(tzinfo=None) if zone is None else noon.astimezone(ZoneInfo(zone))
        literal = pl.lit(value, dtype=pl.Datetime("ms", zone))
        for compare in ORDERINGS:
            predicate = compare(pl.col("a").cast(pl.Datetime("ms", zone)), literal)
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)

    @pytest.mark.parametrize("unit", POLARS_UNITS)
    def test_column_cast_to_other_units_matches_polars(self, unit, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("SET TimeZone = 'UTC'")
        duckdb_cursor.execute("CREATE TABLE t (a TIMESTAMPTZ)")
        duckdb_cursor.execute(self.ROWS)
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        literal = pl.lit(
            datetime.datetime(2024, 1, 1, 12, 0, 0, 500, tzinfo=datetime.timezone.utc), dtype=pl.Datetime(unit, "UTC")
        )
        for compare in ORDERINGS:
            predicate = compare(pl.col("a").cast(pl.Datetime(unit, "UTC")), literal)
            # polars 2 drops a cast to the column's own unit and zone before the source sees the predicate
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=unit == "us" and POLARS_2)

    def test_numeric_column_cast_matches_polars(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")
        duckdb_cursor.execute("INSERT INTO t VALUES (1), (127), (200), (NULL)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        # A non strict cast of 200 into Int8 is NULL for polars, where the engine's cast would raise
        for compare in ORDERINGS:
            predicate = compare(pl.col("i").cast(pl.Int8, strict=False), pl.lit(127, dtype=pl.Int8))
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)

    def test_two_column_comparisons(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER, j BIGINT, d20 DECIMAL(20,0), d38 DECIMAL(38,0))")
        duckdb_cursor.execute("INSERT INTO t VALUES (1, 2, 1, 2), (3, 1, 3, 1), (2, 2, 2, 2), (NULL, 1, NULL, 1)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for compare in ORDERINGS:
            # polars widens the narrower column to Int64, which holds every INTEGER, so the bare column reads the same
            assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("i"), pl.col("j")), pushed=True)
            assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("d20"), pl.col("d38")), pushed=True)

    def test_cast_of_an_expression_matches_polars(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")
        duckdb_cursor.execute("INSERT INTO t VALUES (1), (127), (200), (NULL)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        predicate = (pl.col("i") % 256).cast(pl.Int8, strict=False).is_null()
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)
        assert lazy_df.filter(predicate).collect()["i"].to_list() == [200, None]

    def test_fallback_filters_every_batch(self, duckdb_cursor, pushdowns):
        # One match in the first batch and one past it with empty batches between, so a read that stopped after the
        # first batch or at an empty one would lose the last match
        duckdb_cursor.execute(
            "CREATE TABLE t AS SELECT range AS id, CASE WHEN range IN (100, 250000) THEN 100 ELSE 200 END AS i "
            "FROM range(300000)"
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        # Pushed bare, the predicate kept every row; the non strict cast turns 200 into NULL
        predicate = pl.col("i").cast(pl.Int8, strict=False).is_not_null()
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)
        pushdowns.rows.clear()
        assert lazy_df.filter(predicate).collect()["id"].to_list() == [100, 250000]
        assert len(pushdowns.rows) > 1, pushdowns.rows
        assert pushdowns.rows[0] <= 250000, pushdowns.rows

    def test_float_column_against_a_double_literal_matches_polars(self, duckdb_cursor, pushdowns):
        # polars widens the FLOAT column with a cast; pushed bare, the engine compared 0.1 as a FLOAT and dropped a row
        duckdb_cursor.execute("CREATE TABLE t (f FLOAT)")
        duckdb_cursor.execute("INSERT INTO t VALUES (0.1), (1), (127), (200), (NULL)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for compare in ORDERINGS:
            predicate = compare(pl.col("f"), pl.lit(0.1, dtype=pl.Float64))
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)

    def test_date_column_against_a_zoned_literal_matches_polars(self, duckdb_cursor, pushdowns):
        # polars casts the DATE column to the literal's zoned datetime; pushed bare, the engine read the date in the
        # session zone and kept a day polars drops
        duckdb_cursor.execute("SET TimeZone = 'America/New_York'")
        duckdb_cursor.execute("CREATE TABLE t (d DATE)")
        duckdb_cursor.execute("INSERT INTO t VALUES ('2023-12-31'), ('2024-01-01'), ('2024-01-02'), (NULL)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        literal = pl.lit(datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc), dtype=pl.Datetime("us", "UTC"))
        for compare in ORDERINGS:
            assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("d"), literal), pushed=False)

    def test_decimal_column_against_a_literal_is_still_pushed(self, duckdb_cursor, pushdowns):
        # polars 2 folds the literal into the column's decimal type; polars 1 widens the column to Decimal(38, 0)
        duckdb_cursor.execute("CREATE TABLE t (a DECIMAL(20,0))")
        duckdb_cursor.execute("INSERT INTO t VALUES (1), (10), (100), (99999999999999999999), (NULL)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for compare in ORDERINGS:
            assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("a"), pl.lit(10)), pushed=True)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("TINYINT", "SMALLINT"),
            ("SMALLINT", "INTEGER"),
            ("INTEGER", "BIGINT"),
            ("UTINYINT", "TINYINT"),
            ("UINTEGER", "INTEGER"),
            ("UTINYINT", "UINTEGER"),
            pytest.param(
                "UBIGINT",
                "BIGINT",
                marks=pytest.mark.skipif(not POLARS_2, reason="polars 1 compares UInt64 with Int64 as Float64"),
            ),
        ],
    )
    def test_exact_widening_is_pushed(self, left, right, duckdb_cursor, pushdowns):
        duckdb_cursor.execute(f"CREATE TABLE t (a {left}, b {right})")
        (left_low, left_high), (right_low, right_high) = INTEGER_RANGES[left], INTEGER_RANGES[right]
        rows = [
            (left_low, right_low),
            (left_high, right_high),
            (left_low, right_high),
            (left_high, right_low),
            (1, 1),
            (None, 1),
            (1, None),
        ]
        duckdb_cursor.executemany("INSERT INTO t VALUES (?, ?)", rows)
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for compare in ORDERINGS:
            [tree] = assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("a"), pl.col("b")), pushed=True)
            assert "Cast" in json.dumps(tree)

    @pytest.mark.parametrize(
        ("left", "right", "rows"),
        [
            ("DECIMAL(10,2)", "DECIMAL(12,2)", [("-99999999.99", "-9999999999.99"), ("99999999.99", "99999999.99")]),
            ("DECIMAL(10,2)", "DECIMAL(12,4)", [("-99999999.99", "-99999999.9999"), ("99999999.99", "99999999.9900")]),
            ("DECIMAL(20,0)", "DECIMAL(38,0)", [("-" + "9" * 20, "-" + "9" * 38), ("9" * 20, "9" * 20)]),
        ],
    )
    def test_decimal_widening_is_pushed(self, left, right, rows, duckdb_cursor, pushdowns):
        # polars 2 compares the two decimals as they are, polars 1 widens the narrower one to the wider precision or
        # scale, which keeps every digit
        duckdb_cursor.execute(f"CREATE TABLE t (a {left}, b {right})")
        duckdb_cursor.executemany("INSERT INTO t VALUES (?, ?)", [*rows, ("1", "1"), (None, "1"), ("1", None)])
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for compare in ORDERINGS:
            assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("a"), pl.col("b")), pushed=True)

    def test_written_widening_is_pushed(self, duckdb_cursor, pushdowns):
        # polars 2 marks a cast that cannot fail Overflowing, polars 1 keeps the user's Strict
        duckdb_cursor.execute("CREATE TABLE t (a INTEGER)")
        duckdb_cursor.execute("INSERT INTO t VALUES (-2147483648), (5), (6), (2147483647), (NULL)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for compare in ORDERINGS:
            predicate = compare(pl.col("a").cast(pl.Int64), pl.lit(5, dtype=pl.Int64))
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=True)

    @pytest.mark.parametrize(
        ("columns", "rows", "predicate"),
        [
            pytest.param(
                "a BIGINT", [(2**53,), (2**53 + 1,), (None,)], pl.col("a") == 2.0**53, id="BIGINT to Float64 past 2**53"
            ),
            pytest.param(
                "a UINTEGER",
                [(1,), (4000000000,), (None,)],
                pl.col("a").cast(pl.Int32, strict=False).is_null(),
                id="UInt32 to Int32",
            ),
            pytest.param(
                "a SMALLINT",
                [(1,), (-1,), (None,)],
                pl.col("a").cast(pl.UInt16, strict=False).is_null(),
                id="Int16 to UInt16",
            ),
            pytest.param(
                "a DECIMAL(10,2)",
                [("1.25",), ("1234567.89",), (None,)],
                pl.col("a").cast(pl.Decimal(10, 4), strict=False).is_null(),
                id="Decimal(10,2) to Decimal(10,4)",
            ),
            pytest.param(
                "a DECIMAL(10,4)",
                [("1.2345",), ("1.2300",), (None,)],
                pl.col("a").cast(pl.Decimal(10, 2), strict=False) == decimal.Decimal("1.23"),
                id="Decimal(10,4) to Decimal(10,2)",
            ),
            pytest.param(
                "a DATE",
                [("2024-01-01",), ("infinity",), ("300000-01-01",), (None,)],
                pl.col("a") >= datetime.datetime(2020, 1, 1),
                id="DATE past the microsecond range against a naive datetime",
            ),
            pytest.param(
                "a DATE",
                [("2019-01-01",), ("-infinity",), (None,)],
                pl.col("a") < datetime.datetime(2020, 1, 1),
                id="DATE before the microsecond range against a naive datetime",
            ),
            pytest.param(
                "a DATE, b TIMESTAMP",
                [("infinity", "2020-01-01"), ("2020-01-02", "2020-01-01"), (None, "2020-01-01")],
                pl.col("a") >= pl.col("b"),
                id="DATE against a TIMESTAMP column",
            ),
            pytest.param(
                "a DATE, b TIMESTAMP_NS",
                [("2300-01-01", "2020-01-01"), ("2020-01-02", "2020-01-01"), (None, "2020-01-01")],
                pl.col("a") >= pl.col("b"),
                id="DATE against a TIMESTAMP_NS column",
            ),
            pytest.param(
                "a BIGINT, b DECIMAL(38,20)",
                [(2**63 - 1, 1), (1, 1), (None, 1)],
                pl.col("a") == pl.col("b"),
                id="BIGINT against a DECIMAL(38,20) column",
            ),
        ],
    )
    def test_lossy_cast_is_not_pushed(self, columns, rows, predicate, duckdb_cursor, pushdowns):
        # Each predicate carries a cast next to the widenings that changes a value or the comparison; pushed bare, the
        # engine kept rows polars drops, dropped rows polars keeps, or raised
        duckdb_cursor.execute(f"CREATE TABLE t ({columns})")
        duckdb_cursor.executemany(f"INSERT INTO t VALUES ({', '.join('?' * len(rows[0]))})", rows)
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        [tree] = assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)
        assert "Cast" in json.dumps(tree)

    @pytest.mark.parametrize(
        ("source", "target", "pushed"),
        [
            pytest.param(pl.Int32, '"Int64"', True, id="Int32 to Int64"),
            pytest.param(pl.UInt8, '"Int16"', True, id="UInt8 to Int16"),
            pytest.param(pl.UInt64, '"Int128"', True, id="UInt64 to Int128"),
            pytest.param(pl.UInt8, '"UInt32"', True, id="UInt8 to UInt32"),
            pytest.param(pl.Int64, '"UInt64"', False, id="Int64 to UInt64"),
            pytest.param(pl.UInt32, '"Int32"', False, id="UInt32 to Int32"),
            pytest.param(pl.Int64, '"Int32"', False, id="Int64 to Int32"),
            pytest.param(pl.Decimal(20, 0), '{"Decimal": [38, 0]}', True, id="Decimal(20,0) to Decimal(38,0)"),
            pytest.param(pl.Decimal(10, 2), '{"Decimal": [12, 4]}', True, id="Decimal(10,2) to Decimal(12,4)"),
            pytest.param(pl.Decimal(10, 2), '{"Decimal": [10, 4]}', False, id="Decimal(10,2) to Decimal(10,4)"),
            pytest.param(pl.Decimal(10, 4), '{"Decimal": [10, 2]}', False, id="Decimal(10,4) to Decimal(10,2)"),
            pytest.param(pl.Int64, '{"Decimal": [38, 0]}', False, id="Int64 to Decimal(38,0)"),
            pytest.param(pl.Date, '{"Datetime": ["Microseconds", null]}', False, id="Date to Datetime(Microseconds)"),
            pytest.param(pl.Date, '{"Datetime": ["Milliseconds", null]}', False, id="Date to Datetime(Milliseconds)"),
            pytest.param(pl.Float32, '"Float64"', False, id="Float32 to Float64"),
            pytest.param(pl.Int32, '"Float64"', True, id="Int32 to Float64"),
            pytest.param(pl.UInt32, '"Float64"', True, id="UInt32 to Float64"),
            pytest.param(pl.Int64, '"Float64"', False, id="Int64 to Float64"),
            pytest.param(pl.Decimal(15, 2), '"Float64"', True, id="Decimal(15,2) to Float64"),
            pytest.param(pl.Decimal(16, 2), '"Float64"', False, id="Decimal(16,2) to Float64"),
            pytest.param(pl.Int16, '{"Unknown": "Float"}', True, id="Int16 to the dynamic float"),
            pytest.param(pl.Int32, '"Float32"', False, id="Int32 to Float32"),
        ],
    )
    def test_cast_rule_on_hand_built_trees(self, source, target, pushed):
        schema = pl.Schema({"a": source})
        for options in ("Strict", "NonStrict", "Overflowing"):
            tree = json.loads(
                f'{{"Cast": {{"expr": {{"Column": "a"}}, "dtype": {{"Literal": {target}}}, "options": "{options}"}}}}'
            )
            if pushed:
                assert _pl_tree_to_sql(tree, schema) == '"a"'
            else:
                with pytest.raises(NotImplementedError):
                    _pl_tree_to_sql(tree, schema)
            # Without the source schema a widening cannot be told from a cast that truncates
            with pytest.raises(NotImplementedError):
                _pl_tree_to_sql(tree)
            with pytest.raises(NotImplementedError):
                _pl_tree_to_sql(tree, pl.Schema({"b": source}))

    @pytest.mark.parametrize(
        ("data_type", "value", "unit"),
        [
            ("DECIMAL(15,2)", "9999999999999.98", "0.01"),
            ("DECIMAL(15,0)", "-999999999999998", "1"),
            ("DECIMAL(15,15)", "0.123456789012345", "0.000000000000001"),
            ("DECIMAL(9,4)", "-12345.6789", "0.0001"),
            ("TINYINT", "-127", "1"),
            ("SMALLINT", "32766", "1"),
            ("INTEGER", "-2147483647", "1"),
            ("UTINYINT", "254", "1"),
            ("USMALLINT", "65534", "1"),
            ("UINTEGER", "4294967294", "1"),
        ],
    )
    def test_float_against_a_round_tripping_column_is_pushed(self, data_type, value, unit, duckdb_cursor, pushdowns):
        # Such a value converts to a double no other value of its type shares, so the bare column compares with a
        # double's shortest text, which the engine reads as an exact decimal, or with a DOUBLE column as polars' Float64
        # cast of it does
        value, unit = decimal.Decimal(value), decimal.Decimal(unit)
        double = float(value)
        doubles = [double, math.nextafter(double, math.inf), math.nextafter(double, -math.inf)]
        duckdb_cursor.execute(f"CREATE TABLE t (a {data_type}, x DOUBLE)")
        duckdb_cursor.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [
                (None if a is None else str(a), x)
                for a, x in itertools.product(
                    [value - unit, value, value + unit, None], [*doubles, math.nan, math.inf, -math.inf, None]
                )
            ],
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for compare in ORDERINGS:
            for literal in doubles:
                assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("a"), literal), pushed=True)
            assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("a"), pl.col("x")), pushed=True)

    def test_float_against_a_sixteen_digit_decimal_is_not_pushed(self, duckdb_cursor, pushdowns):
        # 99999999999999.99 and .98 share a double; pushed bare, the engine compares .99 with the double's shortest
        # text, .98, and drops the row polars keeps
        duckdb_cursor.execute("CREATE TABLE t (a DECIMAL(16,2))")
        duckdb_cursor.execute("INSERT INTO t VALUES (99999999999999.99), (99999999999999.98), (1), (NULL)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for compare in ORDERINGS:
            predicate = compare(pl.col("a"), float(decimal.Decimal("99999999999999.99")))
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)


class TestConjunctPushdown:
    # The source pushes the conjuncts of a top level AND that translate and, when one stays behind, applies the whole
    # predicate in polars to every batch; a row the predicate keeps has every conjunct TRUE, which the engine keeps.
    @pytest.mark.parametrize(("disc_type", "pushed"), [("DECIMAL(15,2)", True), ("DECIMAL(16,2)", "partly")])
    def test_tpch_q6_shape_pushes_the_translatable_conjuncts(self, disc_type, pushed, duckdb_cursor, pushdowns):
        duckdb_cursor.execute(f"CREATE TABLE t (id INTEGER, ship DATE, disc {disc_type}, qty DECIMAL(15,2))")
        # Rows the pushed date and quantity bounds keep but a discount bound or a NULL discount removes, so only the
        # polars pass over the batch can drop them
        duckdb_cursor.execute(
            """
            INSERT INTO t VALUES
                (1, '1994-03-01', 0.06, 10), (2, '1994-03-01', 0.04, 10), (3, '1994-03-01', 0.08, 10),
                (4, '1994-03-01', NULL, 10), (5, '1994-01-01', 0.05, 23), (6, '1994-12-31', 0.07, 23.99),
                (7, '1994-03-01', 0.06, 24), (8, '1993-12-31', 0.06, 10), (9, '1995-01-01', 0.06, 10),
                (10, NULL, 0.06, 10), (11, '1994-03-01', 0.06, NULL), (12, '1994-03-01', 0.06, 10)
            """
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        predicate = (
            (pl.col("ship") >= datetime.date(1994, 1, 1))
            & (pl.col("ship") < datetime.date(1995, 1, 1))
            & (pl.col("disc") >= 0.05)
            & (pl.col("disc") <= 0.07)
            & (pl.col("qty") < 24)
        )
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=pushed)
        # polars compares a DECIMAL with a Python float through a Float64 cast of the column, pushed up to 15 digits
        assert refused_columns(pushdowns) == ([] if pushed is True else ["disc", "disc"])
        pushdowns.rows.clear()
        assert lazy_df.filter(predicate).collect()["id"].to_list() == [1, 5, 6, 12]
        # Partly pushed, the seven rows inside the date and quantity bounds reach polars
        assert sum(pushdowns.rows) == (4 if pushed is True else 7)

    def test_refused_conjunct_inside_or_keeps_the_rest_pushed(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (id INTEGER, a INTEGER, b INTEGER, f FLOAT)")
        duckdb_cursor.execute(
            """
            INSERT INTO t VALUES
                (1, 1, 1, 0.5), (2, 1, 2, 0.1), (3, 1, 2, 0.5), (4, 0, 1, 0.1), (5, 1, 3, 0.1), (6, 1, NULL, 0.1),
                (7, NULL, 1, 0.1), (8, 1, 2, NULL)
            """
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        # Nested on the right, so the AND chain is split on both sides
        predicate = (pl.col("a") > 0) & (
            ((pl.col("f") == pl.lit(0.1, dtype=pl.Float64)) | (pl.col("b") == 1)) & (pl.col("b") < 3)
        )
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed="partly")
        assert refused_columns(pushdowns) == ["b", "f"]
        assert len(pushdowns) == 3

    def test_partly_pushed_predicate_under_head_and_top_k(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t AS SELECT range AS id, (range % 300)::INTEGER AS i FROM range(1000)")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        predicate = (pl.col("id") >= 100) & (pl.col("i").cast(pl.Int8, strict=False).is_null())
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed="partly")
        eager = lazy_df.collect().filter(predicate)
        pl_testing.assert_frame_equal(lazy_df.filter(predicate).head(3).collect(), eager.head(3))
        pl_testing.assert_frame_equal(
            lazy_df.filter(predicate).sort("id", descending=True).head(3).collect(),
            eager.sort("id", descending=True).head(3),
        )
        pl_testing.assert_frame_equal(lazy_df.filter(predicate).select("i").collect(), eager.select("i"))


class TestModulusPushdown:
    # polars' integer modulus takes the divisor's sign, the engine's % the dividend's
    @pytest.mark.parametrize("divisor", [3, -3, -1, 2**40])
    @pytest.mark.parametrize("data_type", list(INTEGER_RANGES))
    def test_modulus_takes_the_divisors_sign(self, data_type, divisor, duckdb_cursor, pushdowns):
        low, high = INTEGER_RANGES[data_type]
        values = sorted({low, high, *(value for value in (-7, -1, 0, 1, 7) if low <= value <= high)})
        duckdb_cursor.execute(f"CREATE TABLE t (a {data_type})")
        duckdb_cursor.executemany("INSERT INTO t VALUES (?)", [(value,) for value in [*values, None]])
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        floored = {value % divisor for value in values}
        truncated = {abs(value) % abs(divisor) * (-1 if value < 0 else 1) for value in values}
        # polars computes an unsigned 64 bit column against a negative divisor in Int64, through a narrowing cast
        pushed = data_type != "UBIGINT" or divisor > 0
        for remainder in sorted(floored | truncated):
            assert_pushdown_matches_polars(lazy_df, pushdowns, (pl.col("a") % divisor) == remainder, pushed=pushed)

    def test_modulus_with_a_negative_dividend(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (a INTEGER, s VARCHAR)")
        duckdb_cursor.execute(
            "INSERT INTO t VALUES (-1, 'x'), (2, 'x'), (-4, 'y'), (-7, 'x'), (5, 'x'), (3, 'x'), (NULL, 'x')"
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        # The engine's % gives -1 for -1 % 3 and for -1 % 2, which would drop the row
        assert_pushdown_matches_polars(lazy_df, pushdowns, (pl.col("a") % 3) == 2, pushed=True)
        predicate = ((pl.col("a") % 3) == 2) & pl.col("s").str.contains("x")
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed="partly")
        assert lazy_df.filter(predicate).collect()["a"].to_list() == [-1, 2, -7, 5]
        predicate = (pl.col("a").cast(pl.Int64) % 2) == 1
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=True)
        assert lazy_df.filter(predicate).collect()["a"].to_list() == [-1, -7, 5, 3]

    @pytest.mark.parametrize(
        ("columns", "rows", "predicate"),
        [
            pytest.param("a INTEGER", [(5,), (-5,), (None,)], (pl.col("a") % 0).is_null(), id="zero divisor"),
            pytest.param(
                "a INTEGER, b INTEGER",
                [(5, 0), (-5, 3), (7, -3), (None, 1)],
                (pl.col("a") % pl.col("b")) == 1,
                id="column divisor",
            ),
            pytest.param("a DOUBLE", [(-1.5,), (1.5,), (None,)], (pl.col("a") % 1.0) == 0.5, id="DOUBLE"),
            pytest.param(
                "a DECIMAL(10,2)",
                [("-1.50",), ("1.50",), (None,)],
                (pl.col("a") % 1) == decimal.Decimal("0.5"),
                id="DECIMAL",
                marks=pytest.mark.skipif(not POLARS_2, reason="polars 1 has no decimal modulus"),
            ),
            pytest.param(
                "a UBIGINT",
                [(2**64 - 1,), (5,), (None,)],
                (pl.col("a") % -3).is_null(),
                id="UBIGINT by a negative divisor",
            ),
        ],
    )
    def test_modulus_that_differs_is_not_pushed(self, columns, rows, predicate, duckdb_cursor, pushdowns):
        # Pushed as the engine's %, each raises on a zero divisor or keeps rows polars drops
        duckdb_cursor.execute(f"CREATE TABLE t ({columns})")
        duckdb_cursor.executemany(f"INSERT INTO t VALUES ({', '.join('?' * len(rows[0]))})", rows)
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        [tree] = assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)
        assert "Modulus" in json.dumps(tree)

    @pytest.mark.parametrize(
        ("dtype", "pushed"),
        [(pl.Int32, True), (pl.UInt8, True), (pl.Float64, False), (pl.Decimal(10, 2), False), (pl.Boolean, False)],
    )
    def test_modulus_dividend_type_on_hand_built_trees(self, dtype, pushed):
        # polars pairs a float or decimal dividend with a divisor of its own type, so only a built tree reaches this
        tree = {
            "BinaryExpr": {"left": {"Column": "a"}, "op": "Modulus", "right": {"Literal": {"Scalar": {"Int32": 3}}}}
        }
        if pushed:
            assert _pl_tree_to_sql(tree, {"a": dtype}).startswith('(CASE WHEN ("a" % 3) < 0')
        else:
            with pytest.raises(NotImplementedError):
                _pl_tree_to_sql(tree, {"a": dtype})

    def test_modulus_without_the_schema_is_not_pushed(self):
        # Without the source schema an integer column cannot be told from a float or decimal one
        assert _predicate_to_expression((pl.col("a") % 3) == 2) is None


class TestBooleanOperatorPushdown:
    # polars applies & and | bitwise to integers and types them Boolean only between Booleans, where they are AND and
    # OR, so they reach the engine only at the level of the predicate
    def test_bitwise_and_or_is_not_pushed(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (a INTEGER, b INTEGER, s VARCHAR)")
        duckdb_cursor.execute("INSERT INTO t VALUES (6, 3, 'x'), (4, 3, 'x'), (5, 2, 'y'), (8, 1, 'x'), (NULL, 1, 'x')")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        # Pushed as AND and OR, the engine reads the integers as Booleans
        for predicate in (
            (pl.col("a") & pl.col("b")) == 0,
            ((pl.col("a") & pl.col("b")) == 0) & pl.col("s").str.contains("x"),
            (pl.col("a") | pl.col("b")) == 7,
        ):
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)
        predicate = ((pl.col("a") & pl.col("b")) == 0) & (pl.col("a") > 4)
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed="partly")
        assert lazy_df.filter(predicate).collect().rows() == [(5, 2, "y"), (8, 1, "x")]
        assert _predicate_to_expression((pl.col("a") & pl.col("b")) == 0) is None

    def test_logical_and_or_is_pushed(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (p BOOLEAN, q BOOLEAN, i INTEGER, s VARCHAR)")
        duckdb_cursor.executemany(
            "INSERT INTO t VALUES (?, ?, ?, ?)",
            [
                (p, q, i, "x" if i % 2 else "y")
                for i, (p, q) in enumerate(itertools.product([True, False, None], repeat=2))
            ],
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        # Both sides use three valued logic: TRUE OR NULL keeps a row, FALSE OR NULL drops it
        assert_pushdown_matches_polars(lazy_df, pushdowns, pl.col("p") | pl.col("q"), pushed=True)
        for predicate in (
            (pl.col("p") | pl.col("q")) & pl.col("s").str.contains("x"),
            (pl.col("p") | (pl.col("q") & (pl.col("i") > 2))) & pl.col("s").str.contains("x"),
        ):
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed="partly")


class TestDecimalLiteralPushdown:
    @pytest.mark.parametrize(
        ("unscaled", "scale"),
        [
            (12345678901234567890123456789, 29),
            (11111111111111111111111111111111111111, 38),
            (10**37 + 1, 0),
            (-1, 30),
            (-5, 2),
            (150, 2),
            (0, 3),
        ],
    )
    def test_decimal_literal_renders_its_exact_value(self, unscaled, scale):
        # Decimal arithmetic under the default context rounds past 28 digits and writes small and large values in
        # exponent form, which the engine reads as a DOUBLE
        sql = _pl_tree_to_sql({"Scalar": {"Decimal": [unscaled, 38, scale]}})
        value, data_type = duckdb.execute(f"SELECT {sql}, typeof({sql})").fetchone()
        assert value == decimal.Decimal(f"{unscaled}E-{scale}")
        assert data_type.startswith("DECIMAL" if scale else ("INTEGER", "BIGINT", "HUGEINT")), data_type

    @pytest.mark.parametrize(
        ("data_type", "value", "neighbour"),
        [
            ("DECIMAL(30,29)", "0.12345678901234567890123456789", "0.12345678901234567890123456788"),
            # Untyped, the engine reads this literal, 39 digits with the zero before the point, as a DOUBLE
            ("DECIMAL(38,38)", "0." + "1" * 38, "0." + "1" * 37 + "2"),
            ("DECIMAL(38,0)", str(10**37 + 1), str(10**37)),
            ("DECIMAL(38,30)", "-0." + "0" * 29 + "1", "-0." + "0" * 29 + "2"),
            ("DECIMAL(15,2)", "1.5", "1.51"),
        ],
    )
    def test_decimal_literal_is_exact(self, data_type, value, neighbour, duckdb_cursor, pushdowns):
        duckdb_cursor.execute(f"CREATE TABLE t (a {data_type}, s VARCHAR)")
        duckdb_cursor.executemany(
            "INSERT INTO t VALUES (?, ?)", [(value, "x"), (neighbour, "x"), (value, "y"), (None, "x")]
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        literal = decimal.Decimal(value)
        for compare in ORDERINGS:
            assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("a"), literal), pushed=True)
        predicate = (pl.col("a") == literal) & pl.col("s").str.starts_with("x")
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed="partly")
        assert lazy_df.filter(predicate).collect()["s"].to_list() == ["x"]

    @pytest.mark.parametrize(
        ("columns", "rows", "predicate"),
        [
            pytest.param(
                "a DECIMAL(38,0)", [(str(10**38 - 1),), ("1",)], pl.col("a") < decimal.Decimal("0.5"), id="38,0 < 0.5"
            ),
            pytest.param(
                "a DECIMAL(38,38)", [("0." + "9" * 38,), ("0.5",)], pl.col("a") < decimal.Decimal("1"), id="38,38 < 1"
            ),
            pytest.param(
                "a DECIMAL(37,0)", [(str(10**37 - 1),), ("1",)], pl.col("a") < decimal.Decimal("0.05"), id="37,0 < 0.05"
            ),
            pytest.param(
                "a DECIMAL(30,10)",
                [("9" * 20 + "." + "9" * 10,), ("0",)],
                pl.col("a") < decimal.Decimal("1E-19"),
                id="30,10 < 1E-19",
            ),
            pytest.param("a HUGEINT", [(2**127 - 1,), (-5,)], pl.col("a") < decimal.Decimal("0.5"), id="HUGEINT < 0.5"),
            pytest.param(
                "a DECIMAL(10,2), b DECIMAL(38,0)",
                [("1.50", str(10**37)), ("2.00", "1")],
                pl.col("a") < pl.col("b"),
                id="10,2 < 38,0 column",
            ),
            pytest.param(
                "a DECIMAL(20,0), b DECIMAL(38,20)",
                [(str(10**19), "1"), ("0", "1")],
                pl.col("a").cast(pl.Decimal(38, 1)) > pl.col("b"),
                id="20,0 raised to scale 1 > 38,20 column",
            ),
        ],
    )
    def test_decimal_comparison_past_38_digits_is_not_pushed(self, columns, rows, predicate, duckdb_cursor, pushdowns):
        # Refused until the engine compares such decimals: it clamps their common type to 38 digits and raises for a
        # value that does not fit, where polars compares
        duckdb_cursor.execute(f"CREATE TABLE t ({columns}, s VARCHAR)")
        duckdb_cursor.executemany(
            f"INSERT INTO t VALUES ({', '.join('?' * (len(rows[0]) + 1))})",
            [(*row, "x") for row in [*rows, (None,) * len(rows[0])]],
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate & pl.col("s").str.contains("x"), pushed=False)

    def test_decimal_comparison_at_38_digits_is_pushed(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (a DECIMAL(37,0), s VARCHAR)")
        duckdb_cursor.execute(f"INSERT INTO t VALUES ({10**37 - 1}, 'x'), (0, 'x'), (NULL, 'x')")
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for compare in ORDERINGS:
            assert_pushdown_matches_polars(
                lazy_df, pushdowns, compare(pl.col("a"), decimal.Decimal("0.5")), pushed=True
            )


class TestColumnTypePushdown:
    # UUID and JSON reach polars as a String, BIT as a Binary, TIME_NS and TIME WITH TIME ZONE as a Time, and HUGEINT
    # and UHUGEINT, which hold 39 digits, as a Decimal(38, 0); the engine compares each by its own type and raises or
    # keeps other rows than polars
    @pytest.mark.parametrize(
        ("data_type", "values", "predicate"),
        [
            pytest.param(
                "UUID",
                ["a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11", "10eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"],
                pl.col("a") > "B0000000-0000-0000-0000-000000000000",
                id="UUID in text order",
            ),
            pytest.param(
                "UUID",
                ["a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11", "10eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"],
                pl.col("a") == "abc",
                id="UUID against text that is no UUID",
            ),
            pytest.param("JSON", ['{"a": 1}', "[1]"], pl.col("a") == "abc", id="JSON"),
            pytest.param("BIT", ["0101", "1"], pl.col("a") == b"\x01", id="BIT"),
            pytest.param("TIMETZ", ["12:00:00+05", "03:00:00+00"], pl.col("a") == datetime.time(12), id="TIMETZ"),
            pytest.param("TIME_NS", ["12:00:00.000001", "11:00:00"], pl.col("a") > datetime.time(12), id="TIME_NS"),
            pytest.param("UHUGEINT", [2**128 - 1, 5], pl.col("a") < 0, id="UHUGEINT"),
            pytest.param("HUGEINT", [2**127 - 1, 5], pl.col("a") == 5, id="HUGEINT"),
        ],
    )
    def test_comparison_on_a_column_polars_reads_as_another_type_is_not_pushed(
        self, data_type, values, predicate, duckdb_cursor, pushdowns
    ):
        duckdb_cursor.execute("SET TimeZone = 'UTC'")
        duckdb_cursor.execute(f"CREATE TABLE t (a {data_type}, s VARCHAR)")
        duckdb_cursor.executemany("INSERT INTO t VALUES (?, 'x')", [(value,) for value in [*values, None]])
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate & pl.col("s").str.contains("x"), pushed=False)
        # A NULL test reads the same whatever the type
        assert_pushdown_matches_polars(lazy_df, pushdowns, pl.col("a").is_null(), pushed=True)

    def test_hugeint_against_a_decimal_column_is_not_pushed(self, duckdb_cursor, pushdowns):
        # Compared with a DECIMAL(20,0), the engine casts 2**127 - 1, a 39 digit HUGEINT, to DECIMAL(38,0) and raises
        duckdb_cursor.execute("CREATE TABLE t (h HUGEINT, d DECIMAL(20,0), s VARCHAR)")
        duckdb_cursor.executemany(
            "INSERT INTO t VALUES (?, ?, 'x')", [(2**127 - 1, "1"), (5, "6"), (None, "1"), (1, None)]
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        predicate = pl.col("h") > pl.col("d")
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate & pl.col("s").str.contains("x"), pushed=False)


class TestCollationPushdown:
    # polars compares text by its bytes, the engine by the column's or the session's collation unless told otherwise
    @pytest.mark.parametrize(
        ("column_type", "default_collation"),
        [
            ("VARCHAR COLLATE NOCASE", None),
            ("VARCHAR COLLATE NOACCENT", None),
            ("VARCHAR", "nocase"),
            ("mood", "nocase"),
        ],
    )
    def test_text_compares_by_its_bytes(self, column_type, default_collation, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TYPE mood AS ENUM ('x', 'X', 'e', 'é', 'Y')")
        if default_collation is not None:
            duckdb_cursor.execute(f"SET default_collation = '{default_collation}'")
        duckdb_cursor.execute(f"CREATE TABLE t (a {column_type}, s VARCHAR)")
        duckdb_cursor.executemany(
            "INSERT INTO t VALUES (?, ?)", [("x", "x"), ("X", "x"), ("e", "x"), ("é", "x"), (None, "x"), ("x", "y")]
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        for predicate in (pl.col("a") == "x", pl.col("a") == "e", pl.col("a") < "Y", pl.col("a") > "e"):
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=True)
            partly = predicate & pl.col("s").str.contains("x")
            assert_pushdown_matches_polars(lazy_df, pushdowns, partly, pushed="partly")

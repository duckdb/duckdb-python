import datetime
import json
import operator
import subprocess
import sys
import textwrap
from zoneinfo import ZoneInfo

import pytest

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
ZONE_MISMATCHES = [("America/New_York", "UTC"), ("UTC", "Asia/Tokyo"), ("America/New_York", "Asia/Tokyo")]
ZONE_MISMATCH_ROWS = """
    INSERT INTO t VALUES
        ('2024-01-01 10:00:00+00'), ('2024-01-01 12:00:00+00'), ('2024-01-01 14:00:00+00'), (NULL)
"""


@pytest.fixture
def pushdowns(monkeypatch):
    # The rows alone cannot tell a pushed filter from one polars applied after the source declined it.
    seen = []
    to_sql_expression = polars_io._tree_to_sql_expression

    def recording(tree):
        expression = to_sql_expression(tree)
        try:
            polars_io._pl_tree_to_sql(tree)
        except Exception as error:
            seen.append((tree, expression, type(error)))
        else:
            seen.append((tree, expression, None))
        return expression

    monkeypatch.setattr(polars_io, "_tree_to_sql_expression", recording)
    return seen


def assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, *, pushed, polars_predicate=None):
    pushdowns.clear()
    actual = lazy_df.filter(predicate).collect()
    assert len(pushdowns) == 1, pushdowns
    tree, expression, error = pushdowns[0]
    if pushed:
        assert expression is not None, (predicate, tree, error)
    else:
        assert expression is None, (predicate, tree, expression)
        assert error is NotImplementedError, (predicate, tree, error)
    expected = lazy_df.collect().filter(predicate if polars_predicate is None else polars_predicate)
    pl_testing.assert_frame_equal(actual, expected)
    return tree


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
                tree = assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("a"), literal), pushed=pushed)
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
                tree = assert_pushdown_matches_polars(lazy_df, pushdowns, compare(pl.col("a"), literal), pushed=pushed)
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
            tree = assert_pushdown_matches_polars(
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
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)

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

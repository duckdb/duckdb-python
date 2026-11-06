import datetime
import decimal
import json
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
    "UINTEGER": (0, 2**32 - 1),
    "UBIGINT": (0, 2**64 - 1),
}
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

    def recording(tree, *schema):
        expression = to_sql_expression(tree, *schema)
        try:
            polars_io._pl_tree_to_sql(tree, *schema)
        except Exception as error:
            seen.append((tree, expression, type(error)))
        else:
            seen.append((tree, expression, None))
        return expression

    monkeypatch.setattr(polars_io, "_tree_to_sql_expression", recording)
    return seen


def assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, *, pushed, polars_predicate=None):
    """`pushed` holds for every conjunct of a top level AND, or is "partly" when some reached the engine."""
    pushdowns.clear()
    actual = lazy_df.filter(predicate).collect()
    expected = lazy_df.collect().filter(predicate if polars_predicate is None else polars_predicate)
    pl_testing.assert_frame_equal(actual, expected)
    assert pushdowns, predicate
    states = [expression is not None for _, expression, _ in pushdowns]
    if pushed == "partly":
        assert any(states), (predicate, pushdowns)
        assert not all(states), (predicate, pushdowns)
    else:
        assert all(state == pushed for state in states), (predicate, pushdowns)
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
            # polars drops a cast to the column's own unit and zone before the source sees the predicate
            assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=unit == "us")

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

    def test_fallback_filters_every_batch(self, duckdb_cursor, pushdowns, monkeypatch):
        # One match in the first batch and one past it with empty batches between, so a read that stopped after the
        # first batch or at an empty one would lose the last match
        duckdb_cursor.execute(
            "CREATE TABLE t AS SELECT range AS id, CASE WHEN range IN (100, 250000) THEN 100 ELSE 200 END AS i "
            "FROM range(300000)"
        )
        lazy_df = duckdb_cursor.table("t").pl(lazy=True)
        batches = []
        from_arrow = pl.from_arrow

        def counting(batch, *args, **kwargs):
            batches.append(batch.num_rows)
            return from_arrow(batch, *args, **kwargs)

        monkeypatch.setattr(pl, "from_arrow", counting)
        # Pushed bare, the predicate kept every row; the non strict cast turns 200 into NULL
        predicate = pl.col("i").cast(pl.Int8, strict=False).is_not_null()
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed=False)
        batches.clear()
        assert lazy_df.filter(predicate).collect()["id"].to_list() == [100, 250000]
        assert len(batches) > 1, batches
        assert batches[0] <= 250000, batches

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
            pytest.param(pl.Int32, '"Float64"', False, id="Int32 to Float64"),
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


class TestConjunctPushdown:
    # The source pushes the conjuncts of a top level AND that translate and, when one stays behind, applies the whole
    # predicate in polars to every batch; a row the predicate keeps has every conjunct TRUE, which the engine keeps.
    def test_tpch_q6_shape_pushes_the_translatable_conjuncts(self, duckdb_cursor, pushdowns):
        duckdb_cursor.execute("CREATE TABLE t (id INTEGER, ship DATE, disc DECIMAL(15,2), qty DECIMAL(15,2))")
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
        assert_pushdown_matches_polars(lazy_df, pushdowns, predicate, pushed="partly")
        # polars compares a DECIMAL with a Python float through a Float64 cast of the column, which is refused
        assert refused_columns(pushdowns) == ["disc", "disc"]
        assert lazy_df.filter(predicate).collect()["id"].to_list() == [1, 5, 6, 12]

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

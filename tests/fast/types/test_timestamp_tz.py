import datetime
import os
import subprocess
import sys
import textwrap
import zoneinfo

import pytest

import duckdb

UTC = datetime.timezone.utc


def fetch(con, literal):
    return con.execute(f"SELECT TIMESTAMPTZ '{literal}'").fetchone()[0]


@pytest.fixture
def con():
    with duckdb.connect() as con:
        yield con


@pytest.mark.parametrize("zone", ["UTC", "Europe/Amsterdam", "America/Los_Angeles", "Asia/Kolkata"])
def test_fetch_in_the_session_time_zone(con, zone):
    con.execute(f"SET TimeZone = '{zone}'")
    value = fetch(con, "2024-07-01 12:00:00+00")
    expected = datetime.datetime(2024, 7, 1, 12, tzinfo=UTC).astimezone(zoneinfo.ZoneInfo(zone))
    assert value == expected
    assert value.utcoffset() == expected.utcoffset()
    assert value.tzinfo is zoneinfo.ZoneInfo(zone)
    assert value.tzinfo.key == zone


@pytest.mark.parametrize("literal", ["2024-10-27 00:30:00+00", "2024-10-27 01:30:00+00"])
def test_repeated_hour_keeps_the_instant(con, literal):
    con.execute("SET TimeZone = 'Europe/Amsterdam'")
    value = fetch(con, literal)
    instant = datetime.datetime.fromisoformat(literal.replace(" ", "T") + ":00")
    assert value.replace(tzinfo=None) == datetime.datetime(2024, 10, 27, 2, 30)
    assert value.fold == (0 if literal.startswith("2024-10-27 00") else 1)
    assert value.utcoffset() == datetime.timedelta(hours=2 if value.fold == 0 else 1)
    assert value.timestamp() == instant.timestamp()
    assert value.astimezone(UTC) == instant


def test_values_of_one_session_share_the_zone_and_compare_by_wall_clock(con):
    con.execute("SET TimeZone = 'Europe/Amsterdam'")
    first, second = con.execute(
        "SELECT TIMESTAMPTZ '2024-10-27 00:30:00+00', TIMESTAMPTZ '2024-10-27 01:30:00+00'"
    ).fetchone()
    assert first.tzinfo is second.tzinfo
    assert first == second
    assert first.fold == 0
    assert second.fold == 1
    assert first.astimezone(UTC) != second.astimezone(UTC)
    assert second.timestamp() - first.timestamp() == 3600
    keyed = con.execute(
        "SELECT MAP {TIMESTAMPTZ '2024-10-27 00:30:00+00': 1, TIMESTAMPTZ '2024-10-27 01:30:00+00': 2}"
    ).fetchone()[0]
    assert keyed == {second: 2}


def test_timestamptz_ns_takes_the_same_path(con):
    con.execute("SET TimeZone = 'Asia/Tokyo'")
    value = con.execute("SELECT '2024-01-01 00:00:00.123456789+00'::TIMESTAMPTZ_NS").fetchone()[0]
    assert value.tzinfo is zoneinfo.ZoneInfo("Asia/Tokyo")
    assert value.replace(tzinfo=None) == datetime.datetime(2024, 1, 1, 9, 0, 0, 123456)
    assert value == datetime.datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=UTC)


def test_dst_offset_after_2037(con):
    con.execute("SET TimeZone = 'America/Chicago'")
    assert fetch(con, "2037-08-01 08:00:00+00").isoformat() == "2037-08-01T03:00:00-05:00"
    assert fetch(con, "2038-08-01 08:00:00+00").isoformat() == "2038-08-01T03:00:00-05:00"
    assert fetch(con, "2050-01-15 08:00:00+00").isoformat() == "2050-01-15T02:00:00-06:00"


def test_nested_values_carry_the_zone(con):
    con.execute("SET TimeZone = 'Europe/Amsterdam'")
    literal = "TIMESTAMPTZ '2024-07-01 00:00:00+00'"
    row = con.execute(f"SELECT [{literal}], {{'t': {literal}}}, MAP {{'k': {literal}}}").fetchone()
    expected = datetime.datetime(2024, 7, 1, 2, tzinfo=zoneinfo.ZoneInfo("Europe/Amsterdam"))
    assert row == ([expected], {"t": expected}, {"k": expected})
    assert row[0][0].tzinfo is expected.tzinfo


def test_udf_argument_is_aware(con):
    pytest.importorskip("numpy")
    con.execute("SET TimeZone = 'Asia/Tokyo'")
    con.create_function(
        "describe_tz", lambda t: f"{t.isoformat()} {type(t.tzinfo).__name__} {t.tzinfo.key}", ["TIMESTAMPTZ"], "VARCHAR"
    )
    assert con.execute("SELECT describe_tz(TIMESTAMPTZ '2024-01-01 00:00:00+00')").fetchone()[0] == (
        "2024-01-01T09:00:00+09:00 ZoneInfo Asia/Tokyo"
    )


def test_icu_only_zone_name_raises(con):
    con.execute("SET TimeZone = 'ACT'")
    with pytest.raises(duckdb.InvalidInputException, match=r"'ACT'.*tzdata"):
        fetch(con, "2024-01-01 00:00:00+00")


def test_without_a_tz_database_raises(tmp_path):
    script = textwrap.dedent(
        """
        import sys

        sys.modules["tzdata"] = None

        import duckdb
        con = duckdb.connect()
        con.execute("SET TimeZone = 'Europe/Amsterdam'")
        try:
            con.execute("SELECT TIMESTAMPTZ '2024-01-01 00:00:00+00'").fetchone()
        except duckdb.InvalidInputException as error:
            assert "'Europe/Amsterdam'" in str(error) and "tzdata" in str(error), repr(error)
        else:
            raise AssertionError("a fetch without a tz database did not raise")
        """
    )
    env = {**os.environ, "PYTHONTZPATH": ""}
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize(
    ("zone", "literal"),
    [
        ("Asia/Tokyo", "9999-12-31 23:00:00+00"),
        ("America/Los_Angeles", "0001-01-01 00:00:00+00"),
    ],
)
def test_local_wall_time_outside_python_range_falls_back_to_str(con, zone, literal):
    con.execute(f"SET TimeZone = '{zone}'")
    assert fetch(con, literal) == literal


def test_utc_wall_time_outside_python_range_is_already_str(con):
    con.execute("SET TimeZone = 'America/Los_Angeles'")
    assert fetch(con, "9999-12-31 23:59:59.999999-08") == "10000-01-01 07:59:59.999999+00"


def test_local_mean_time_offset_keeps_seconds(con):
    con.execute("SET TimeZone = 'Asia/Tokyo'")
    value = fetch(con, "0001-01-01 00:00:00+00")
    assert value == datetime.datetime(1, 1, 1, tzinfo=UTC)
    assert value.utcoffset() == datetime.timedelta(hours=9, minutes=18, seconds=59)


def test_infinity_and_null_are_unchanged(con):
    con.execute("SET TimeZone = 'Europe/Amsterdam'")
    row = con.execute("SELECT TIMESTAMPTZ '-infinity', TIMESTAMPTZ 'infinity', NULL::TIMESTAMPTZ").fetchone()
    assert row == (datetime.datetime.min, datetime.datetime.max, None)
    assert row[0].tzinfo is None
    assert row[1].tzinfo is None

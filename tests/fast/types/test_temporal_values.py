"""TIMESTAMP, DATE and TIME values come back as the Python objects built from the same fields.

The binding builds each object from its fields packed into bytes, so the cases cover every byte boundary of that
packing (year above 255, microseconds above 255 and above 65535) and Python's year range.
"""

import datetime

import pytest

import duckdb
import numpy as np

YEARS = [1, 255, 256, 1970, 2024, 9999]
MICROSECONDS = [0, 1, 255, 256, 65535, 65536, 123456, 999999]
TIMES = [(0, 0, 0), (1, 2, 3), (12, 30, 15), (23, 59, 59)]
MONTH_DAYS = [(1, 1), (2, 29), (6, 15), (12, 31)]


def _timestamps():
    for year in YEARS:
        for month, day in MONTH_DAYS:
            if (month, day) == (2, 29) and year not in (2024,):
                continue
            for hour, minute, second in TIMES:
                for micros in MICROSECONDS:
                    yield datetime.datetime(year, month, day, hour, minute, second, micros)


class TestTemporalValues:
    def test_timestamps(self, duckdb_cursor):
        expected = list(_timestamps())
        literals = ", ".join(f"(TIMESTAMP '{value.isoformat(sep=' ')}')" for value in expected)
        got = [row[0] for row in duckdb_cursor.sql(f"SELECT * FROM (VALUES {literals})").fetchall()]
        assert got == expected
        assert all(type(value) is datetime.datetime for value in got)
        assert [value.isoformat() for value in got] == [value.isoformat() for value in expected]

    def test_dates(self, duckdb_cursor):
        expected = [
            datetime.date(year, month, day) for year in YEARS for month, day in MONTH_DAYS if (month, day) != (2, 29)
        ]
        literals = ", ".join(f"(DATE '{value.isoformat()}')" for value in expected)
        got = [row[0] for row in duckdb_cursor.sql(f"SELECT * FROM (VALUES {literals})").fetchall()]
        assert got == expected
        assert all(type(value) is datetime.date for value in got)

    def test_times(self, duckdb_cursor):
        expected = [datetime.time(h, m, s, micros) for h, m, s in TIMES for micros in MICROSECONDS]
        literals = ", ".join(f"(TIME '{value.isoformat()}')" for value in expected)
        got = [row[0] for row in duckdb_cursor.sql(f"SELECT * FROM (VALUES {literals})").fetchall()]
        assert got == expected
        assert all(type(value) is datetime.time for value in got)

    @pytest.mark.parametrize(
        ("literal", "text"),
        [
            ("TIMESTAMP '10000-01-01 00:00:00'", "10000-01-01 00:00:00"),
            ("TIMESTAMP '0000-12-31 00:00:00'", "0001-12-31 (BC) 00:00:00"),
            ("DATE '10000-01-01'", "10000-01-01"),
            ("DATE '0000-12-31'", "0001-12-31 (BC)"),
            ("TIME '24:00:00'", "24:00:00"),
        ],
    )
    def test_values_outside_pythons_range_come_back_as_text(self, duckdb_cursor, literal, text):
        assert duckdb_cursor.sql(f"SELECT {literal}").fetchall() == [(text,)]

    @pytest.mark.parametrize("hours", [128, 256])
    def test_times_the_engine_cannot_format_raise(self, duckdb_cursor, hours):
        # Arrow hands TIME values over unchecked. A release build cannot print these hours with two digits and a debug
        # build asserts on them, but neither may come back as midnight (hour 128 packs into the byte Python reads as
        # the fold flag, 256 wraps to 0)
        pa = pytest.importorskip("pyarrow")
        table = pa.table({"t": pa.array([hours * 3_600_000_000], type=pa.time64("us"))})
        with pytest.raises(duckdb.Error):
            duckdb_cursor.from_arrow(table).fetchall()

    def test_infinities_are_the_extremes(self, duckdb_cursor):
        row = duckdb_cursor.sql(
            "SELECT TIMESTAMP 'infinity', TIMESTAMP '-infinity', DATE 'infinity', DATE '-infinity'"
        ).fetchall()[0]
        assert row == (datetime.datetime.max, datetime.datetime.min, datetime.date.max, datetime.date.min)

    def test_nanosecond_timestamps_truncate_to_microseconds(self, duckdb_cursor):
        got = duckdb_cursor.sql("SELECT TIMESTAMP_NS '2020-06-15 12:30:15.123456789'").fetchall()[0][0]
        assert got == datetime.datetime(2020, 6, 15, 12, 30, 15, 123456)

    def test_aware_values_keep_their_offsets(self, duckdb_cursor):
        duckdb_cursor.execute("SET TimeZone = 'UTC'")
        tstz, ttz = duckdb_cursor.sql("SELECT TIMESTAMPTZ '2020-06-15 12:30:15+02', TIMETZ '12:30:15.5+02'").fetchall()[
            0
        ]
        # aware values compare by instant, so the offset and the local fields are pinned apart
        assert tstz.utcoffset() == datetime.timedelta(0)
        assert tstz.replace(tzinfo=None) == datetime.datetime(2020, 6, 15, 10, 30, 15)
        assert ttz.utcoffset() == datetime.timedelta(hours=2)
        assert ttz.replace(tzinfo=None) == datetime.time(12, 30, 15, 500000)


class TestTemporalSubclasses:
    """Fields are read as Python reads them, so a subclass's properties apply, and they may run arbitrary code."""

    def test_an_overridden_field_is_what_gets_converted(self, duckdb_cursor):
        class LaterYear(datetime.date):
            @property
            def year(self):
                return 2030

        assert duckdb_cursor.values([LaterYear(2020, 1, 2)]).fetchall() == [(datetime.date(2030, 1, 2),)]

    @pytest.mark.parametrize(
        ("kind", "field", "value", "error", "message"),
        [
            (datetime.time, "hour", 256, duckdb.InvalidInputException, "Time out of range"),
            (datetime.datetime, "minute", 60, duckdb.InvalidInputException, "Time out of range"),
            (datetime.date, "month", 13, duckdb.ConversionException, "Date out of range"),
        ],
    )
    def test_an_invalid_field_is_refused(self, duckdb_cursor, kind, field, value, error, message):
        bad = type("Bad", (kind,), {field: property(lambda self: value)})
        sample = bad(2020, 1, 2) if kind is not datetime.time else bad(1, 2, 3)
        with pytest.raises(error, match=message):
            duckdb_cursor.values([sample]).fetchall()
        assert duckdb_cursor.sql("SELECT 42").fetchall() == [(42,)]

    def test_a_raising_field_propagates(self, duckdb_cursor):
        class Boom(datetime.time):
            @property
            def minute(self):
                message = "boom"
                raise ValueError(message)

        with pytest.raises(ValueError, match="boom"):
            duckdb_cursor.values([Boom(1, 2, 3)]).fetchall()
        assert duckdb_cursor.sql("SELECT 42").fetchall() == [(42,)]

    @pytest.mark.parametrize(
        ("field", "sample"),
        [
            ("hour", datetime.datetime(2020, 1, 2, 3, 4, 5)),
            ("year", datetime.date(2020, 1, 2)),
            ("hour", datetime.time(3, 4, 5)),
            ("days", datetime.timedelta(days=7, seconds=5)),
        ],
    )
    def test_a_field_read_that_drops_the_only_reference(self, duckdb_cursor, field, sample):
        # the frame's slot is the object's only owner; the read runs Python code that empties it mid-conversion
        kind = type(sample)
        values = np.empty(1, dtype=object)
        inner = getattr(kind, field)

        def dropping(self):
            values[0] = None
            return inner.__get__(self, type(self))

        vanishing = type("Vanishing", (kind,), {field: property(dropping)})
        values[0] = vanishing(*sample.__reduce__()[1]) if kind is not datetime.timedelta else vanishing(7, 5)
        duckdb_cursor.register("src", {"v": values})
        assert duckdb_cursor.execute("SELECT v FROM src").fetchall() == [(sample,)]

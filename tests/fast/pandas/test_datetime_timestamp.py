import datetime

import pandas as pd
import pytest
from packaging.version import Version


class TestDateTimeTimeStamp:
    def test_timestamp_high(self, duckdb_cursor):
        duckdb_time = duckdb_cursor.sql("SELECT '2260-01-01 23:59:00'::TIMESTAMP AS '0'").df()
        df_in = pd.DataFrame(  # noqa: F841
            {
                0: pd.Series(
                    data=[datetime.datetime(year=2260, month=1, day=1, hour=23, minute=59)],
                    dtype="datetime64[us]",
                )
            }
        )
        df_out = duckdb_cursor.sql("select * from df_in").df()
        pd.testing.assert_frame_equal(df_out, duckdb_time)

    def test_timestamp_low(self, duckdb_cursor):
        duckdb_time = duckdb_cursor.sql(
            """
            SELECT '1680-01-01 23:59:00.234243'::TIMESTAMP AS '0'
        """
        ).df()
        df_in = pd.DataFrame(
            {
                "0": pd.Series(
                    data=[
                        pd.Timestamp(
                            datetime.datetime(year=1680, month=1, day=1, hour=23, minute=59, microsecond=234243),
                            unit="us",
                        )
                    ],
                    dtype="datetime64[us]",
                )
            }
        )
        print("original:", duckdb_time["0"].dtype)
        print("df_in:", df_in["0"].dtype)
        df_out = duckdb_cursor.sql("select * from df_in").df()
        print("df_out:", df_out["0"].dtype)
        pd.testing.assert_frame_equal(df_out, duckdb_time)

    @pytest.mark.skipif(
        Version(pd.__version__) < Version("2.0.2"), reason="pandas < 2.0.2 does not properly convert timezones"
    )
    def test_timestamp_timezone_regular(self, duckdb_cursor):
        duckdb_time = duckdb_cursor.sql(
            """
            SELECT timestamp '2022-01-01 12:00:00' AT TIME ZONE 'Pacific/Easter' as "0"
        """
        ).df()

        offset = datetime.timedelta(hours=-2)
        timezone = datetime.timezone(offset)
        df_in = pd.DataFrame(  # noqa: F841
            {
                0: pd.Series(
                    data=[datetime.datetime(year=2022, month=1, day=1, hour=15, tzinfo=timezone)], dtype="object"
                )
            }
        )
        df_out = duckdb_cursor.sql("select * from df_in").df()
        print(df_out)
        print(duckdb_time)
        pd.testing.assert_frame_equal(df_out, duckdb_time)

    @pytest.mark.skipif(
        Version(pd.__version__) < Version("2.0.2"), reason="pandas < 2.0.2 does not properly convert timezones"
    )
    def test_timestamp_timezone_negative_extreme(self, duckdb_cursor):
        duckdb_time = duckdb_cursor.sql(
            """
            SELECT timestamp '2022-01-01 12:00:00' AT TIME ZONE 'Chile/EasterIsland' as "0"
        """
        ).df()

        offset = datetime.timedelta(hours=-19)
        timezone = datetime.timezone(offset)

        df_in = pd.DataFrame(  # noqa: F841
            {
                0: pd.Series(
                    data=[datetime.datetime(year=2021, month=12, day=31, hour=22, tzinfo=timezone)], dtype="object"
                )
            }
        )
        df_out = duckdb_cursor.sql("select * from df_in").df()
        pd.testing.assert_frame_equal(df_out, duckdb_time)

    @pytest.mark.skipif(
        Version(pd.__version__) < Version("2.0.2"), reason="pandas < 2.0.2 does not properly convert timezones"
    )
    def test_timestamp_timezone_positive_extreme(self, duckdb_cursor):
        duckdb_time = duckdb_cursor.sql(
            """
            SELECT timestamp '2021-12-31 23:00:00' AT TIME ZONE 'Etc/GMT-14' as "0"
        """
        ).df()

        offset = datetime.timedelta(hours=14)
        timezone = datetime.timezone(offset)

        df_in = pd.DataFrame(  # noqa: F841
            {
                0: pd.Series(
                    data=[datetime.datetime(year=2021, month=12, day=31, hour=23, tzinfo=timezone)], dtype="object"
                )
            }
        )
        df_out = duckdb_cursor.sql("""select * from df_in""").df()
        pd.testing.assert_frame_equal(df_out, duckdb_time)

    @pytest.mark.skipif(
        Version(pd.__version__) < Version("2.0.2"), reason="pandas < 2.0.2 does not properly convert timezones"
    )
    @pytest.mark.parametrize("unit", ["ms", "ns", "s"])
    def test_timestamp_timezone_coverage(self, unit, duckdb_cursor):
        ts_df = pd.DataFrame(  # noqa: F841
            {"ts": pd.Series(data=[pd.Timestamp(datetime.datetime(1990, 12, 21))], dtype=f"datetime64[{unit}]")}
        )
        usecond_df = pd.DataFrame(  # noqa: F841
            {"ts": pd.Series(data=[pd.Timestamp(datetime.datetime(1990, 12, 21))], dtype="datetime64[us]")}
        )

        query = """
            select
                cast(ts as timestamptz) as tstz
            from {}
        """

        duckdb_cursor.sql("set TimeZone = 'UTC'")
        utc_usecond = duckdb_cursor.sql(query.format("usecond_df")).df()
        utc_other = duckdb_cursor.sql(query.format("ts_df")).df()

        duckdb_cursor.sql("set TimeZone = 'America/Los_Angeles'")
        us_usecond = duckdb_cursor.sql(query.format("usecond_df")).df()
        us_other = duckdb_cursor.sql(query.format("ts_df")).df()

        pd.testing.assert_frame_equal(utc_usecond, utc_other)
        pd.testing.assert_frame_equal(us_usecond, us_other)


class TestConvertedColumnsKeepTheirPlace:
    @pytest.mark.parametrize(
        ("column", "kwargs"),
        [
            pytest.param("TIMESTAMPTZ '2020-01-01 00:00:00+00'", {}, id="timestamptz"),
            pytest.param("DATE '2020-01-01'", {"date_as_object": True}, id="date_as_object"),
        ],
    )
    def test_df_replaces_the_column_without_a_pandas_warning(self, duckdb_cursor, column, kwargs):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = duckdb_cursor.sql(f"SELECT 1 AS a, {column} AS converted, 2 AS b").df(**kwargs)
        assert list(df.columns) == ["a", "converted", "b"]
        assert df["a"].tolist() == [1]
        assert df["b"].tolist() == [2]

    @pytest.fixture
    def zoned_table(self, duckdb_cursor):
        duckdb_cursor.execute("SET TimeZone = 'Asia/Kathmandu'")
        duckdb_cursor.execute(
            """
            CREATE TABLE t AS
            SELECT * FROM (VALUES
                (1, TIMESTAMPTZ '2024-01-01 10:00:00+00', DATE '1992-07-30', TIMESTAMPTZ '2000-06-15 00:00:00+00', 'x'),
                (2, NULL, NULL, NULL, 'y')
            ) v(i, tz1, d, tz2, s)
            """
        )
        return duckdb_cursor

    @staticmethod
    def check_frame(df, date_as_object) -> None:
        assert list(df.columns) == ["i", "tz1", "d", "tz2", "s"]
        assert str(df["tz1"].dt.tz) == "Asia/Kathmandu"
        assert str(df["tz2"].dt.tz) == "Asia/Kathmandu"
        assert df["tz1"][0] == pd.Timestamp("2024-01-01 15:45:00", tz="Asia/Kathmandu")
        assert df["tz2"][0] == pd.Timestamp("2000-06-15 05:45:00", tz="Asia/Kathmandu")
        assert pd.isnull(df["tz1"][1])
        assert pd.isnull(df["tz2"][1])
        if date_as_object:
            assert type(df["d"][0]) is datetime.date
            assert df["d"][0] == datetime.date(1992, 7, 30)
        else:
            assert pd.api.types.is_datetime64_dtype(df["d"])
            assert df["d"][0] == pd.Timestamp("1992-07-30")
        assert pd.isnull(df["d"][1])
        assert list(df["i"]) == [1, 2]
        assert list(df["s"]) == ["x", "y"]

    @pytest.mark.parametrize("date_as_object", [False, True])
    def test_two_converted_columns_with_nulls_keep_their_place(self, zoned_table, date_as_object):
        import warnings

        fetches = [
            lambda: zoned_table.execute("SELECT * FROM t").df(date_as_object=date_as_object),
            lambda: zoned_table.execute("SELECT * FROM t").fetch_df_chunk(date_as_object=date_as_object),
            lambda: zoned_table.table("t").df(date_as_object=date_as_object),
        ]
        for fetch in fetches:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                df = fetch()
            self.check_frame(df, date_as_object)

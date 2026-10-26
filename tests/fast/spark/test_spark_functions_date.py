import warnings

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")
from datetime import date, datetime, timezone
from decimal import Decimal

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import AnalysisException
from spark_namespace.sql import functions as F
from spark_namespace.sql.functions import col
from spark_namespace.sql.types import (
    DateType,
    IntegerType,
    Row,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)


class TestsSparkFunctionsDate:
    def test_date_trunc(self, spark):
        df = spark.createDataFrame(
            [(datetime(2019, 1, 23, 14, 34, 9, 87539),)],
            ["dt_ref"],
        )

        expected = [
            {
                "year": datetime(2019, 1, 1, 0, 0, 0, 0),
                "yyyy": datetime(2019, 1, 1, 0, 0, 0, 0),
                "yy": datetime(2019, 1, 1, 0, 0, 0, 0),
                "quarter": datetime(2019, 1, 1, 0, 0, 0, 0),
                "month": datetime(2019, 1, 1, 0, 0, 0, 0),
                "mon": datetime(2019, 1, 1, 0, 0, 0, 0),
                "mm": datetime(2019, 1, 1, 0, 0, 0, 0),
                "week": datetime(2019, 1, 21, 0, 0, 0, 0),
                "day": datetime(2019, 1, 23, 0, 0, 0, 0),
                "dd": datetime(2019, 1, 23, 0, 0, 0, 0),
                "hour": datetime(2019, 1, 23, 14, 0, 0, 0),
                "minute": datetime(2019, 1, 23, 14, 34, 0, 0),
                "second": datetime(2019, 1, 23, 14, 34, 9, 0),
            }
        ]

        cols = list(expected[0].keys())
        gen_record = df.select(*[F.date_trunc(fmt, "dt_ref").alias(fmt) for fmt in cols]).collect()[0]

        expected_record = spark.createDataFrame(
            # Need to convert to a list for Spark which otherwise throws a TypeError.
            # It would work without it for DuckDB.
            [list(r.values()) for r in expected],
            cols,
        ).collect()[0]

        assert gen_record.year.timetuple() == expected_record.year.timetuple()
        assert gen_record.yyyy.timetuple() == expected_record.yyyy.timetuple()
        assert gen_record.yy.timetuple() == expected_record.yy.timetuple()
        assert gen_record.quarter.timetuple() == expected_record.quarter.timetuple()
        assert gen_record.month.timetuple() == expected_record.month.timetuple()
        assert gen_record.mon.timetuple() == expected_record.mon.timetuple()
        assert gen_record.mm.timetuple() == expected_record.mm.timetuple()
        assert gen_record.week.timetuple() == expected_record.week.timetuple()
        assert gen_record.day.timetuple() == expected_record.day.timetuple()
        assert gen_record.dd.timetuple() == expected_record.dd.timetuple()
        assert gen_record.hour.timetuple() == expected_record.hour.timetuple()
        assert gen_record.minute.timetuple() == expected_record.minute.timetuple()
        assert gen_record.second.timetuple() == expected_record.second.timetuple()

    def test_date_part(self, spark):
        df = spark.createDataFrame([(datetime(2015, 4, 8, 13, 8, 15),)], ["ts"])
        result = df.select(
            F.date_part(F.lit("YEAR"), "ts").alias("year"),
            F.date_part(F.lit("month"), "ts").alias("month"),
            F.date_part(F.lit("WEEK"), "ts").alias("week"),
            F.date_part(F.lit("D"), "ts").alias("day"),
            F.date_part(F.lit("M"), "ts").alias("minute"),
            F.date_part(F.lit("S"), "ts").alias("second"),
        ).collect()

        expected = [Row(year=2015, month=4, week=15, day=8, minute=8, second=15)]

        assert result == expected

    def test_dayofweek(self, spark):
        spark_sunday_index = 1
        spark_saturday_index = 7

        df = spark.createDataFrame([(date(2024, 5, 12),), (date(2024, 5, 18),)], ["dt"])
        result = df.select(F.dayofweek("dt").alias("week_num")).collect()

        assert result[0].week_num == spark_sunday_index
        assert result[1].week_num == spark_saturday_index

    def test_dayofmonth(self, spark):
        df = spark.createDataFrame([(date(2024, 5, 12),), (date(2024, 5, 18),)], ["dt"])
        result = df.select(F.dayofmonth("dt").alias("day_num")).collect()

        assert result[0].day_num == 12
        assert result[1].day_num == 18

    def test_dayofyear(self, spark):
        df = spark.createDataFrame([(date(2024, 5, 12),), (date(2024, 5, 18),)], ["dt"])
        result = df.select(F.dayofyear("dt").alias("day_num")).collect()

        assert result[0].day_num == 133
        assert result[1].day_num == 139

    def test_month(self, spark):
        df = spark.createDataFrame([(date(2024, 5, 12),), (date(2024, 5, 18),)], ["dt"])
        result = df.select(F.month("dt").alias("month_num")).collect()

        assert result[0].month_num == 5
        assert result[1].month_num == 5

    def test_year(self, spark):
        df = spark.createDataFrame([(date(2024, 5, 12),), (date(2024, 5, 18),)], ["dt"])
        result = df.select(F.year("dt").alias("year_num")).collect()

        assert result[0].year_num == 2024
        assert result[1].year_num == 2024

    def test_weekofyear(self, spark):
        df = spark.createDataFrame([(date(2024, 5, 12),), (date(2024, 5, 18),)], ["dt"])
        result = df.select(F.weekofyear("dt").alias("week_num")).collect()

        assert result[0].week_num == 19
        assert result[1].week_num == 20

    def test_quarter(self, spark):
        df = spark.createDataFrame([(date(2024, 5, 12),), (date(2024, 5, 18),)], ["dt"])
        result = df.select(F.quarter("dt").alias("quarter_num")).collect()

        assert result[0].quarter_num == 2
        assert result[1].quarter_num == 2

    def test_hour(self, spark):
        df = spark.createDataFrame([(datetime(2024, 5, 12, 13, 30, 45),)], ["dt"])
        result = df.select(F.hour("dt").alias("hour_num")).collect()

        assert result[0].hour_num == 13

    def test_minute(self, spark):
        df = spark.createDataFrame([(datetime(2024, 5, 12, 13, 30, 45),)], ["dt"])
        result = df.select(F.minute("dt").alias("minute_num")).collect()

        assert result[0].minute_num == 30

    def test_second(self, spark):
        df = spark.createDataFrame([(datetime(2024, 5, 12, 13, 30, 45),)], ["dt"])
        result = df.select(F.second("dt").alias("second_num")).collect()

        assert result[0].second_num == 45

    def test_unix_date(self, spark):
        df = spark.createDataFrame([("1970-01-02",)], ["t"])
        res = df.select(F.unix_date(df.t.cast("date")).alias("n")).collect()
        assert res == [Row(n=1)]

    def test_unix_micros(self, spark):
        df = spark.createDataFrame([("2015-07-22 10:00:00+00:00",)], ["t"])
        res = df.select(F.unix_micros(df.t.cast("timestamp")).alias("n")).collect()
        assert res == [Row(n=1437559200000000)]

    def test_unix_millis(self, spark):
        df = spark.createDataFrame([("2015-07-22 10:00:00+00:00",)], ["t"])
        res = df.select(F.unix_millis(df.t.cast("timestamp")).alias("n")).collect()
        assert res == [Row(n=1437559200000)]

    def test_unix_seconds(self, spark):
        df = spark.createDataFrame([("2015-07-22 10:00:00+00:00",)], ["t"])
        res = df.select(F.unix_seconds(df.t.cast("timestamp")).alias("n")).collect()
        assert res == [Row(n=1437559200)]

    def test_weekday(self, spark):
        df = spark.createDataFrame([("2015-04-08",)], ["dt"])
        res = df.select(F.weekday(df.dt.cast("date")).alias("day")).collect()
        assert res == [Row(day=2)]

    def test_to_date(self, spark):
        df = spark.createDataFrame([("1997-02-28 10:30:00",)], ["t"])
        res = df.select(F.to_date(df.t).alias("date")).collect()
        assert res == [Row(date=date(1997, 2, 28))]

    def test_to_timestamp(self, spark):
        df = spark.createDataFrame([("1997-02-28 10:30:00",)], ["t"])
        res = df.select(F.to_timestamp(df.t).alias("dt")).collect()
        assert res == [Row(dt=datetime(1997, 2, 28, 10, 30))]

    def test_to_timestamp_ltz(self, spark):
        df = spark.createDataFrame([("2016-12-31",)], ["e"])
        res = df.select(F.to_timestamp_ltz(df.e).alias("r")).collect()

        assert res == [Row(r=datetime(2016, 12, 31, 0, 0))]

    def test_to_timestamp_ntz(self, spark):
        df = spark.createDataFrame([("2016-04-08",)], ["e"])

        # The DeprecationWarning is issued within PySpark. As we treat
        # warnings as errors in the tests, we need to filter it out
        # for this test to pass.
        if USE_ACTUAL_SPARK:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                res = df.select(F.to_timestamp_ntz(df.e).alias("r")).collect()
        else:
            res = df.select(F.to_timestamp_ntz(df.e).alias("r")).collect()
        assert res == [Row(r=datetime(2016, 4, 8, 0, 0))]

    def test_last_day(self, spark):
        df = spark.createDataFrame([("1997-02-10",)], ["d"])

        res = df.select(F.last_day(df.d.cast("date")).alias("date")).collect()
        assert res == [Row(date=date(1997, 2, 28))]

    def test_add_months(self, spark):
        df = spark.createDataFrame([(datetime(2024, 5, 12, 13, 30, 45), 2)], ["dt", "months"])

        result = df.select(
            F.add_months("dt", 1).alias("with_literal"),
            F.add_months("dt", "months").alias("with_str"),
            F.add_months(col("dt"), df["months"]).alias("with_col"),
        ).collect()

        assert result[0].with_literal == date(2024, 6, 12)
        assert result[0].with_str == date(2024, 7, 12)
        assert result[0].with_col == date(2024, 7, 12)

    def test_date_diff(self, spark):
        df = spark.createDataFrame([("2015-04-08", "2015-05-10")], ["d1", "d2"])

        result_data = df.select(F.date_diff(col("d2").cast("DATE"), col("d1").cast("DATE")).alias("diff")).collect()
        assert result_data[0]["diff"] == 32

        result_data = df.select(F.date_diff(col("d1").cast("DATE"), col("d2").cast("DATE")).alias("diff")).collect()
        assert result_data[0]["diff"] == -32

    def test_date_diff_timestamps_count_calendar_days(self, spark):
        df = spark.createDataFrame([(datetime(2015, 4, 8, 23, 0), datetime(2015, 4, 9, 1, 0))], ["t1", "t2"])
        res = df.select(F.date_diff("t2", "t1").alias("later"), F.date_diff("t1", "t2").alias("earlier")).collect()
        assert res == [Row(later=1, earlier=-1)]

    def test_try_to_timestamp(self, spark):
        df = spark.createDataFrame([("1997-02-28 10:30:00",), ("2024-01-01",), ("invalid",)], ["t"])
        res = df.select(F.try_to_timestamp(df.t).alias("dt")).collect()
        assert res[0].dt == datetime(1997, 2, 28, 10, 30)
        assert res[1].dt == datetime(2024, 1, 1, 0, 0)
        assert res[2].dt is None

    def test_try_to_timestamp_with_format(self, spark):
        df = spark.createDataFrame([("1997-02-28 10:30:00",), ("2024-01-01",), ("invalid",)], ["t"])
        res = df.select(F.try_to_timestamp(df.t, format=F.lit("yyyy-MM-dd HH:mm:ss")).alias("dt")).collect()
        assert res[0].dt == datetime(1997, 2, 28, 10, 30)
        assert res[1].dt is None
        assert res[2].dt is None

    def test_try_to_timestamp_follows_cast_rules(self, spark):
        texts = [
            "2024-01-01T10:30:00",
            "2024-01-01 10:30:00.123456",
            "2024-01-01 10:30:00Z",
            "2024-1-5",
            "invalid",
            None,
        ]
        df = spark.createDataFrame([(t,) for t in texts], ["t"])
        res = df.select(F.try_to_timestamp("t").alias("dt")).collect()
        # Text with an offset is an instant, shown in the session time zone, which defaults to the local one.
        utc_local = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
        assert [r.dt for r in res] == [
            datetime(2024, 1, 1, 10, 30),
            datetime(2024, 1, 1, 10, 30, 0, 123456),
            utc_local,
            datetime(2024, 1, 5),
            None,
            None,
        ]

    def test_to_timestamp_applies_offset(self, spark):
        df = spark.createDataFrame([("2024-01-01 10:30:00+01:00",)], ["t"])
        columns = [
            F.to_timestamp("t").alias("ltz"),
            F.to_timestamp_ltz("t").alias("ltz2"),
            F.try_to_timestamp("t").alias("try_ltz"),
        ]
        # PySpark's to_timestamp_ntz issues a DeprecationWarning, which the test settings turn into an error.
        with warnings.catch_warnings():
            if USE_ACTUAL_SPARK:
                warnings.filterwarnings("ignore", category=DeprecationWarning)
            res = df.select(*columns, F.to_timestamp_ntz("t").alias("ntz")).collect()
        # Spark shifts the instant to the session time zone, which defaults to the local one.
        local = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
        assert res == [Row(ltz=local, ltz2=local, try_ltz=local, ntz=datetime(2024, 1, 1, 10, 30))]

    def test_date_trunc_unknown_unit_is_null(self, spark):
        df = spark.createDataFrame([(datetime(2024, 1, 31, 13, 8, 15),)], ["t"])
        res = df.select(
            F.date_trunc("hh", "t").alias("hh"),
            F.date_trunc("decade", "t").alias("decade"),
            F.date_trunc("bogus", "t").alias("bogus"),
            F.date_trunc("HOUR", "t").alias("hour"),
        ).collect()
        assert res == [Row(hh=None, decade=None, bogus=None, hour=datetime(2024, 1, 31, 13))]

    def test_date_trunc_sub_second(self, spark):
        df = spark.createDataFrame([(datetime(2024, 1, 31, 13, 8, 15, 123456),)], ["t"])
        res = df.select(
            F.date_trunc("millisecond", "t").alias("ms"), F.date_trunc("microsecond", "t").alias("us")
        ).collect()
        assert res == [Row(ms=datetime(2024, 1, 31, 13, 8, 15, 123000), us=datetime(2024, 1, 31, 13, 8, 15, 123456))]

    def test_date_part_fields(self, spark):
        df = spark.createDataFrame([(datetime(2021, 1, 3, 13, 8, 15, 123456),)], ["ts"])
        res = df.select(
            F.date_part(F.lit("DAYOFWEEK"), "ts").alias("dow"),
            F.date_part(F.lit("dow_iso"), "ts").alias("dow_iso"),
            F.date_part(F.lit("YEAROFWEEK"), "ts").alias("year_of_week"),
            F.date_part(F.lit("WEEK"), "ts").alias("week"),
            F.date_part(F.lit("QTR"), "ts").alias("quarter"),
            F.date_part(F.lit("DOY"), "ts").alias("doy"),
            F.date_part(F.lit("SECONDS"), "ts").alias("second"),
            F.extract(F.lit("S"), "ts").alias("s"),
            F.datepart(F.lit("dow"), "ts").alias("dow2"),
        ).collect()
        assert res == [
            Row(
                dow=1,
                dow_iso=7,
                year_of_week=2020,
                week=53,
                quarter=1,
                doy=3,
                second=Decimal("15.123456"),
                s=Decimal("15.123456"),
                dow2=1,
            )
        ]
        assert isinstance(res[0].second, Decimal)

    def test_date_part_unknown_field(self, spark):
        df = spark.createDataFrame([(datetime(2021, 1, 3, 13, 8, 15),)], ["ts"])
        with pytest.raises(AnalysisException, match="INVALID_EXTRACT_FIELD"):
            df.select(F.date_part(F.lit("bogus"), "ts")).collect()

    def test_add_months_month_end(self, spark):
        dates = [date(2024, 1, 31), date(2024, 2, 29), date(2023, 1, 31), date(2024, 4, 30)]
        df = spark.createDataFrame([(d,) for d in dates], ["d"])
        res = df.select(
            F.add_months("d", 1).alias("next"),
            F.add_months("d", -12).alias("year_ago"),
            F.add_months("d", 0).alias("same"),
        ).collect()
        assert res == [
            Row(next=date(2024, 2, 29), year_ago=date(2023, 1, 31), same=date(2024, 1, 31)),
            Row(next=date(2024, 3, 29), year_ago=date(2023, 2, 28), same=date(2024, 2, 29)),
            Row(next=date(2023, 2, 28), year_ago=date(2022, 1, 31), same=date(2023, 1, 31)),
            Row(next=date(2024, 5, 30), year_ago=date(2023, 4, 30), same=date(2024, 4, 30)),
        ]

    def test_week_numbering_at_year_boundaries(self, spark):
        dates = [date(2021, 1, 3), date(2020, 12, 31), date(2024, 12, 30), date(2023, 1, 1)]
        df = spark.createDataFrame([(d,) for d in dates], ["d"])
        res = df.select(
            F.weekofyear("d").alias("week"),
            F.dayofweek("d").alias("dow"),
            F.weekday("d").alias("weekday"),
            F.last_day("d").alias("last"),
        ).collect()
        assert res == [
            Row(week=53, dow=1, weekday=6, last=date(2021, 1, 31)),
            Row(week=53, dow=5, weekday=3, last=date(2020, 12, 31)),
            Row(week=1, dow=2, weekday=0, last=date(2024, 12, 31)),
            Row(week=52, dow=1, weekday=6, last=date(2023, 1, 31)),
        ]

    def test_timestamp_results_have_timestamp_type(self, spark):
        schema = StructType([StructField("s", StringType()), StructField("t", TimestampType())])
        df = spark.createDataFrame([("2024-01-02 03:04:05", datetime(2024, 5, 6, 7, 8, 9))], schema)
        res = df.select(
            F.to_timestamp("s"),
            F.to_timestamp("s", "yyyy-MM-dd HH:mm:ss"),
            F.try_to_timestamp("s"),
            F.try_to_timestamp("s", F.lit("yyyy-MM-dd HH:mm:ss")),
            F.to_timestamp_ltz("s"),
            F.to_timestamp_ntz("s"),
            F.lit(datetime(2024, 1, 2, 3, 4, 5)),
            F.date_trunc("month", "t"),
            F.date_trunc("year", F.to_date("s")),
            F.date_trunc("bogus", "t"),
        )
        assert [f.dataType for f in res.schema.fields] == [TimestampType()] * 5 + [TimestampNTZType()] + [
            TimestampType()
        ] * 4
        parsed = datetime(2024, 1, 2, 3, 4, 5)
        assert res.collect() == [
            Row(*[parsed] * 7, datetime(2024, 5, 1), datetime(2024, 1, 1), None),
        ]

    def test_date_parts_are_int(self, spark):
        schema = StructType([StructField("t", TimestampType()), StructField("d", DateType())])
        df = spark.createDataFrame([(datetime(2024, 5, 6, 7, 8, 9), date(2024, 5, 6))], schema)
        res = df.select(
            F.year("d"),
            F.quarter("d"),
            F.month("d"),
            F.day("d"),
            F.dayofmonth("d"),
            F.dayofweek("d"),
            F.dayofyear("d"),
            F.weekofyear("d"),
            F.weekday("d"),
            F.hour("t"),
            F.minute("t"),
            F.second("t"),
            F.date_part(F.lit("YEAR"), "d"),
            F.extract(F.lit("DOW"), "d"),
            F.datepart(F.lit("HOUR"), "t"),
            F.date_diff("d", F.lit(date(2024, 5, 1))),
            F.unix_date("d"),
        )
        assert [f.dataType for f in res.schema.fields] == [IntegerType()] * 17
        assert res.collect() == [Row(2024, 2, 5, 6, 6, 2, 127, 19, 0, 7, 8, 9, 2024, 2, 7, 5, 19849)]

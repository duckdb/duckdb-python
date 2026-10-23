import datetime

import duckdb
import pandas as pd


def run_checks(df):
    assert type(df["d"][0]) is datetime.date
    assert df["d"][0] == datetime.date(1992, 7, 30)
    assert pd.isnull(df["d"][1])


def test_date_as_datetime():
    con = duckdb.connect()
    con.execute("create table t (d date)")
    con.execute("insert into t values ('1992-07-30'), (NULL)")

    # Connection Methods
    run_checks(con.execute("Select * from t").df(date_as_object=True))
    run_checks(con.execute("Select * from t").fetchdf(date_as_object=True))
    run_checks(con.execute("Select * from t").fetch_df_chunk(date_as_object=True))
    run_checks(con.execute("Select * from t").fetch_df(date_as_object=True))

    # Relation Methods
    rel = con.table("t")
    run_checks(rel.df(date_as_object=True))
    run_checks(rel.to_df(date_as_object=True))

    # Result Methods
    run_checks(rel.query("t_1", "select * from t_1").df(date_as_object=True))


def test_column_replacement_emits_no_warnings(recwarn):
    # Replacing converted columns must not use deprecated pandas keywords (e.g. `inplace`)
    con = duckdb.connect()
    tz_df = con.sql("select timestamptz '2020-01-01 00:00:00+00' as tz").df()
    assert str(tz_df["tz"].dtype).startswith("datetime64[")
    date_df = con.sql("select date '2020-01-01' as d").df(date_as_object=True)
    assert date_df["d"][0] == datetime.date(2020, 1, 1)
    assert not recwarn.list

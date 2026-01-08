import datetime

import pytest

from duckdb import sqltypes


def test_time_ns_select(duckdb_cursor):
    duckdb_cursor.execute("SELECT TIME_NS '1992-09-20 11:30:00.123456'")
    result = duckdb_cursor.fetchone()[0]
    assert result
    assert isinstance(result, datetime.time)


def test_time_ns_insert(duckdb_cursor):
    """This tests that datetime.time values are inserted as TIME_NS by default."""
    duckdb_cursor.execute("SELECT TIME_NS '1992-09-20 11:30:00.123456'")
    result1 = duckdb_cursor.fetchone()[0]
    duckdb_cursor.execute("CREATE OR REPLACE TEMP TABLE time_ns_test (time_ns_col TIME_NS)")
    duckdb_cursor.execute("INSERT INTO time_ns_test VALUES (?)", [result1])
    duckdb_cursor.execute("SELECT time_ns_col FROM time_ns_test")
    result2 = duckdb_cursor.fetchone()[0]
    assert isinstance(result2, datetime.time)
    assert result1 == result2


def test_time_insert(duckdb_cursor):
    """This tests that datetime.time values are casted to TIME when needed."""
    duckdb_cursor.execute("SELECT TIME_NS '1992-09-20 11:30:00.123456'")
    result1 = duckdb_cursor.fetchone()[0]
    duckdb_cursor.execute("CREATE OR REPLACE TEMP TABLE time_test (time_col TIME)")
    duckdb_cursor.execute("INSERT INTO time_test VALUES (?)", [result1])
    duckdb_cursor.execute("SELECT time_col FROM time_test")
    result2 = duckdb_cursor.fetchone()[0]
    assert isinstance(result2, datetime.time)
    assert result1 == result2


def test_time_ns_arrow_roundtrip(duckdb_cursor):
    pa = pytest.importorskip("pyarrow")

    # Get a time_ns in an arrow table
    arrow_table = duckdb_cursor.execute("SELECT TIME_NS '12:34:56.123456789' AS time_ns_col").fetch_arrow_table()

    value = arrow_table.column("time_ns_col")[0]
    assert isinstance(value, pa.lib.Time64Scalar)

    # Roundtrip back into duckdb and assert the column's type is TIME_NS
    duckdb_cursor.execute("CREATE OR REPLACE TEMP TABLE time_ns_test AS SELECT * FROM arrow_table")
    col_type = duckdb_cursor.execute("SELECT time_ns_col FROM time_ns_test").description[0][1]
    assert col_type == sqltypes.TIME_NS


def test_time_ns_pandas_roundtrip(duckdb_cursor):
    """Test that we can roundtrip using Pandas."""
    pytest.importorskip("pandas")
    df = duckdb_cursor.execute("SELECT TIME_NS '12:34:56.123456789' AS time_ns_col").df()
    assert df["time_ns_col"].dtype == "object"
    duckdb_cursor.execute("CREATE OR REPLACE TEMP TABLE time_ns_test AS SELECT * FROM df")
    col_type = duckdb_cursor.execute("SELECT time_ns_col FROM time_ns_test").description[0][1]
    assert col_type == sqltypes.TIME_NS


def test_time_pandas_roundtrip(duckdb_cursor):
    """For Pandas, creating a table using CREATE .... AS SELECT FROM df, will create TIME_NS cols by default."""
    pytest.importorskip("pandas")
    df = duckdb_cursor.execute("SELECT TIME '12:34:56.123456789' AS time_col").df()
    assert df["time_col"].dtype == "object"
    duckdb_cursor.execute("CREATE OR REPLACE TEMP TABLE time_test AS SELECT * FROM df")
    col_type = duckdb_cursor.execute("SELECT time_col FROM time_test").description[0][1]
    assert col_type == sqltypes.TIME_NS

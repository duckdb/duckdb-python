import datetime

import pytest

import duckdb


class TestPythonResult:
    def test_result_closed(self, duckdb_cursor):
        connection = duckdb.connect("")
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE integers (i integer)")
        cursor.execute("INSERT INTO integers VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(NULL)")
        rel = connection.table("integers")
        res = rel.aggregate("sum(i)").execute()
        res.close()
        with pytest.raises(duckdb.InvalidInputException, match="result closed"):
            res.fetchone()
        with pytest.raises(duckdb.InvalidInputException, match="result closed"):
            res.fetchall()
        with pytest.raises(duckdb.InvalidInputException, match="result closed"):
            res.fetchnumpy()
        with pytest.raises(duckdb.InvalidInputException, match="There is no query result"):
            res.fetch_arrow_table()
        with pytest.raises(duckdb.InvalidInputException, match="There is no query result"):
            res.fetch_arrow_reader(1)

    def test_result_describe_types(self, duckdb_cursor):
        connection = duckdb.connect("")
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE test (i bool, j TIME, k VARCHAR)")
        cursor.execute("INSERT INTO test VALUES (TRUE, '01:01:01', 'bla' )")
        rel = connection.table("test")
        res = rel.execute()
        assert res.description == [
            ("i", "BOOLEAN", None, None, None, None, None),
            ("j", "TIME", None, None, None, None, None),
            ("k", "VARCHAR", None, None, None, None, None),
        ]

    def test_result_timestamps(self, duckdb_cursor):
        connection = duckdb.connect("")
        cursor = connection.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS timestamps (sec TIMESTAMP_S, milli TIMESTAMP_MS,micro TIMESTAMP_US, nano TIMESTAMP_NS );"  # noqa: E501
        )
        cursor.execute(
            "INSERT INTO timestamps VALUES ('2008-01-01 00:00:11','2008-01-01 00:00:01.794','2008-01-01 00:00:01.98926','2008-01-01 00:00:01.899268321' )"  # noqa: E501
        )

        rel = connection.table("timestamps")
        assert rel.execute().fetchall() == [
            (
                datetime.datetime(2008, 1, 1, 0, 0, 11),
                datetime.datetime(2008, 1, 1, 0, 0, 1, 794000),
                datetime.datetime(2008, 1, 1, 0, 0, 1, 989260),
                datetime.datetime(2008, 1, 1, 0, 0, 1, 899268),
            )
        ]

    def test_result_interval(self):
        connection = duckdb.connect()
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS intervals (ivals INTERVAL)")
        cursor.execute("INSERT INTO intervals VALUES ('1 day'), ('2 second'), ('1 microsecond')")

        rel = connection.table("intervals")
        res = rel.execute()
        assert res.description == [("ivals", "INTERVAL", None, None, None, None, None)]
        assert res.fetchall() == [
            (datetime.timedelta(days=1.0),),
            (datetime.timedelta(seconds=2.0),),
            (datetime.timedelta(microseconds=1.0),),
        ]

    def test_description_uuid(self):
        connection = duckdb.connect()
        connection.execute("select uuid();")
        connection.description  # noqa: B018

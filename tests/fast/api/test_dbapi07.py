# timestamp ms precision

from datetime import datetime

import numpy


class TestNumpyTimestampMilliseconds:
    def test_numpy_timestamp(self, duckdb_cursor):
        res = duckdb_cursor.execute("SELECT TIMESTAMP '2019-11-26 21:11:42.501' as test_time").fetchnumpy()
        assert res["test_time"] == numpy.datetime64("2019-11-26 21:11:42.501")


class TestTimestampMilliseconds:
    def test_numpy_timestamp(self, duckdb_cursor):
        res = duckdb_cursor.execute("SELECT TIMESTAMP '2019-11-26 21:11:42.501' as test_time").fetchone()[0]
        assert res == datetime.strptime("2019-11-26 21:11:42.501", "%Y-%m-%d %H:%M:%S.%f")

import datetime
from datetime import time, timezone

import pytest

pandas = pytest.importorskip("pandas")


class TestTimeTz:
    def test_time_tz(self, duckdb_cursor):
        df = pandas.DataFrame({"col1": [time(1, 2, 3, tzinfo=timezone.utc)]})  # noqa: F841

        sql = "SELECT * FROM df"

        duckdb_cursor.execute(sql)

        res = duckdb_cursor.fetchall()
        assert res == [(datetime.time(1, 2, 3, tzinfo=datetime.timezone.utc),)]

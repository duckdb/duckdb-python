from pathlib import Path

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")


def test_10795():
    arrow_filename = Path(__file__).parent / "data" / "arrow_table"
    with pa.memory_map(str(arrow_filename), "r") as source:
        reader = pa.ipc.RecordBatchFileReader(source)
        taxi_fhvhv_arrow = reader.read_all()
        con = duckdb.connect(database=":memory:")
        con.execute("SET TimeZone='UTC';")
        con.register("taxi_fhvhv", taxi_fhvhv_arrow)
        res = con.execute("""
            SELECT PULocationID, pickup_datetime
            FROM taxi_fhvhv
            WHERE pickup_datetime >= '2023-01-01T00:00:00-05:00' AND PULocationID = 244
        """).fetchall()

        assert len(res) == 3685

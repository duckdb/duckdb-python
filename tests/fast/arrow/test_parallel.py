from pathlib import Path

import duckdb

try:
    import numpy as np
    import pyarrow
    import pyarrow.parquet

    can_run = True
except Exception:
    can_run = False


class TestArrowParallel:
    def test_parallel_run(self, duckdb_cursor):
        if not can_run:
            return
        duckdb_conn = duckdb.connect()
        duckdb_conn.execute("PRAGMA threads=4")
        duckdb_conn.execute("PRAGMA verify_parallelism")
        data = pyarrow.array(np.random.randint(800, size=1000000), type=pyarrow.int32())
        tbl = pyarrow.Table.from_batches(pyarrow.Table.from_arrays([data], ["a"]).to_batches(10000))
        rel = duckdb_conn.from_arrow(tbl)
        # Also test multiple reads
        assert rel.aggregate("(count(a))::INT").execute().fetchone()[0] == 1000000
        assert rel.aggregate("(count(a))::INT").execute().fetchone()[0] == 1000000

    def test_parallel_types_and_different_batches(self, duckdb_cursor):
        if not can_run:
            return
        duckdb_conn = duckdb.connect()
        duckdb_conn.execute("PRAGMA threads=4")
        duckdb_conn.execute("PRAGMA verify_parallelism")

        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        userdata_parquet_table = pyarrow.parquet.read_table(parquet_filename)
        for i in [7, 51, 99, 100, 101, 500, 1000, 2000]:
            data = pyarrow.array(np.arange(3, 7), type=pyarrow.int32())
            tbl = pyarrow.Table.from_arrays([data], ["a"])
            duckdb_conn.from_arrow(tbl)
            userdata_parquet_table2 = pyarrow.Table.from_batches(userdata_parquet_table.to_batches(i))
            rel = duckdb_conn.from_arrow(userdata_parquet_table2)
            result = rel.filter("first_name='Jose' and salary > 134708.82").aggregate("count(*)")
            assert result.execute().fetchone()[0] == 4

    def test_parallel_fewer_batches_than_threads(self, duckdb_cursor):
        if not can_run:
            return
        duckdb_conn = duckdb.connect()
        duckdb_conn.execute("PRAGMA threads=4")
        duckdb_conn.execute("PRAGMA verify_parallelism")

        data = pyarrow.array(np.random.randint(800, size=1000), type=pyarrow.int32())
        tbl = pyarrow.Table.from_batches(pyarrow.Table.from_arrays([data], ["a"]).to_batches(2))
        rel = duckdb_conn.from_arrow(tbl)
        # Also test multiple reads
        assert rel.aggregate("(count(a))::INT").execute().fetchone()[0] == 1000

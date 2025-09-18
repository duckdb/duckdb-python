import os

import duckdb

try:
    import pyarrow
    import pyarrow.parquet

    can_run = True
except Exception:
    can_run = False


class TestArrowReads:
    def test_multiple_queries_same_relation(self, duckdb_cursor):
        if not can_run:
            return
        parquet_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "userdata1.parquet")  # noqa: PTH118
        userdata_parquet_table = pyarrow.parquet.read_table(parquet_filename)
        userdata_parquet_table.validate(full=True)
        rel = duckdb.from_arrow(userdata_parquet_table)
        assert rel.aggregate("(avg(salary))::INT").execute().fetchone()[0] == 149005
        assert rel.aggregate("(avg(salary))::INT").execute().fetchone()[0] == 149005

from pathlib import Path

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
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        userdata_parquet_table = pyarrow.parquet.read_table(parquet_filename)
        userdata_parquet_table.validate(full=True)
        rel = duckdb.from_arrow(userdata_parquet_table)
        assert rel.aggregate("(avg(salary))::INT").execute().fetchone()[0] == 149005
        assert rel.aggregate("(avg(salary))::INT").execute().fetchone()[0] == 149005

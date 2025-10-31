import duckdb


class TestColumnOrder:
    def test_column_order(self, duckdb_cursor):
        to_execute = """
		CREATE OR REPLACE TABLE t1 AS (
			SELECT NULL AS col1,
			NULL::TIMESTAMPTZ AS timepoint,
			NULL::DATE AS date,
		);
		SELECT timepoint, date, col1 FROM t1;
		"""
        df = duckdb.execute(to_execute).fetchdf()
        cols = list(df.columns)
        assert cols == ["timepoint", "date", "col1"]

import tempfile

import pytest

pa = pytest.importorskip("pyarrow", minversion="11")
pq = pytest.importorskip("pyarrow.parquet", minversion="11")


class Test7652:
    def test_7652(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            temp_file_name = tmp.name
        # Generate a list of values that aren't uniform in changes.
        generated_list = [1, 0, 2]

        # Convert list of values to a PyArrow table with a single column.
        fake_table = pa.Table.from_arrays([pa.array(generated_list, pa.int64())], names=["n0"])

        # Write that column with DELTA_BINARY_PACKED encoding
        with pq.ParquetWriter(
            temp_file_name, fake_table.schema, column_encoding={"n0": "DELTA_BINARY_PACKED"}, use_dictionary=False
        ) as writer:
            writer.write_table(fake_table)

        # Check to make sure that PyArrow can read the file and retrieve the expected values.
        # Assert the values read from PyArrow are the same
        read_table = pq.read_table(temp_file_name, use_threads=False)

        read_list = read_table["n0"].to_pylist()
        assert min(read_list) == min(generated_list)
        assert max(read_list) == max(generated_list)
        assert read_list == generated_list

        # Attempt to perform the same thing with duckdb.
        print("Retrieving from duckdb")
        duckdb_result = [v[0] for v in duckdb_cursor.sql(f"select * from '{temp_file_name}'").fetchall()]

        print("DuckDB result:", duckdb_result)
        assert min(duckdb_result) == min(generated_list)
        assert max(duckdb_result) == max(generated_list)
        assert duckdb_result == generated_list

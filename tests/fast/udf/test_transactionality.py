import pytest

import duckdb


class TestUDFTransactionality:
    def test_type_coverage(self, duckdb_cursor):
        duckdb_cursor.execute("select * from range(4096)")
        assert duckdb_cursor.fetchone() == (0,)

        def my_func(x: str) -> int:
            return int(x)

        with pytest.raises(duckdb.ResourceInUseException, match="unfinished result from execute"):
            duckdb_cursor.create_function("test", my_func)

        assert duckdb_cursor.fetchone() == (1,)

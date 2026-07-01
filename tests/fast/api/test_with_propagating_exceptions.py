import pytest

import duckdb


class TestWithPropagatingExceptions:
    def test_with(self):
        # Should propagate exception raised in the 'with duckdb.connect() ..'
        with (
            pytest.raises(duckdb.CatalogException, match="Table with name invalid does not exist"),
            duckdb.connect() as con,
        ):
            con.execute("invalid")

        # Does not raise an exception
        with duckdb.connect() as con:
            con.execute("select 1")

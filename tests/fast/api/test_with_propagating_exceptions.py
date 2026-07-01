import pytest

import duckdb


class TestWithPropagatingExceptions:
    def test_with(self):
        # Should propagate exception raised in the 'with duckdb.connect() ..'
        with pytest.raises(duckdb.ParserException, match=r"syntax error at or near *"), duckdb.connect() as con:
            con.execute("invalid")

        # Does not raise an exception
        with duckdb.connect() as con:
            con.execute("select 1")

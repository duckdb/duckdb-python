import pytest

import duckdb


class TestExplain:
    def test_explain_basic(self, duckdb_cursor):
        res = duckdb_cursor.sql("select 42").explain()
        assert isinstance(res, str)

    def test_explain_standard(self, duckdb_cursor):
        res = duckdb_cursor.sql("select 42").explain("standard")
        assert isinstance(res, str)

        res = duckdb_cursor.sql("select 42").explain("STANDARD")
        assert isinstance(res, str)

        res = duckdb_cursor.sql("select 42").explain(duckdb.ExplainType.STANDARD)
        assert isinstance(res, str)

        res = duckdb_cursor.sql("select 42").explain(0)
        assert isinstance(res, str)

    def test_explain_analyze(self, duckdb_cursor):
        res = duckdb_cursor.sql("select 42").explain("analyze")
        assert isinstance(res, str)

        res = duckdb_cursor.sql("select 42").explain("ANALYZE")
        assert isinstance(res, str)

        res = duckdb_cursor.sql("select 42").explain(duckdb.ExplainType.ANALYZE)
        assert isinstance(res, str)

        res = duckdb_cursor.sql("select 42").explain(1)
        assert isinstance(res, str)

    def test_explain_df(self, duckdb_cursor):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [42]})  # noqa: F841
        res = duckdb_cursor.sql("select * from df").explain("ANALYZE")
        assert isinstance(res, str)

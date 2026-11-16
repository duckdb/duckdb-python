import pytest


class TestMaterializedRelationSlow:
    @pytest.mark.parametrize("num_rows", [1000000, 10000000])
    def test_materialized_relation(self, duckdb_cursor, num_rows):
        # Anything that is not a SELECT statement becomes a materialized relation, so we use `CALL`
        query = f"call repeat_row(42, 'test', 'this is a long string', true, num_rows={num_rows})"
        rel = duckdb_cursor.sql(query)
        assert rel.limit(1).fetchall() == [(42, "test", "this is a long string", True)]
        assert len(rel.fetchall()) == num_rows
        assert len(rel.fetchall()) == num_rows

        projection = rel.select("column0")
        assert projection.fetchall() == [(42,) for _ in range(num_rows)]

        filtered = rel.filter("column1 != 'test'")
        assert filtered.fetchall() == []

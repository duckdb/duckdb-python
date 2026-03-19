import platform

import pytest


class TestMaterializedRelationSlow:
    @pytest.mark.parametrize(
        "num_rows",
        [
            1000000,
            pytest.param(
                10000000,
                marks=pytest.mark.skipif(
                    condition=platform.system() == "Emscripten",
                    reason="Emscripten/Pyodide builds run out of memory at this scale, and error might not "
                    "thrown reliably",
                ),
            ),
        ],
    )
    def test_materialized_relation(self, duckdb_cursor, num_rows):
        # Anything that is not a SELECT statement becomes a materialized relation, so we use `CALL`
        query = f"call repeat_row(42, 'test', 'this is a long string', true, num_rows={num_rows})"
        rel = duckdb_cursor.sql(query)
        res = rel.fetchone()
        assert res is not None

        res = rel.fetchmany(num_rows)
        assert len(res) == num_rows - 1

        res = rel.fetchmany(5)
        assert len(res) == 0
        res = rel.fetchmany(5)
        assert len(res) == 0
        res = rel.fetchone()
        assert res is None

        rel.execute()
        res = rel.fetchone()
        assert res is not None

        res = rel.fetchall()
        assert len(res) == num_rows - 1
        res = rel.fetchall()
        assert len(res) == num_rows

        rel = duckdb_cursor.sql(query)
        projection = rel.select("column0")
        assert projection.fetchall() == [(42,) for _ in range(num_rows)]

        filtered = rel.filter("column1 != 'test'")
        assert filtered.fetchall() == []

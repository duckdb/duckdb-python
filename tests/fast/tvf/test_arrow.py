from typing import Iterator

import pytest

import duckdb
from duckdb.functional import PythonTVFType


def simple_generator(count: int = 10) -> Iterator[tuple[str, int]]:
    for i in range(count):
        yield (f"name_{i}", i)


def simple_arrow_table(count: int):
    import pyarrow as pa

    data = {
        "id": list(range(count)),
        "value": [i * 2 for i in range(count)],
        "name": [f"row_{i}" for i in range(count)],
    }
    return pa.table(data)


def test_arrow_small(tmp_path):
    pa = pytest.importorskip("pyarrow")

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            "simple_arrow",
            simple_arrow_table,
            schema=[("x", "BIGINT"), ("y", "VARCHAR")],  # Wrong schema!
            type=PythonTVFType.ARROW_TABLE,
        )

        with pytest.raises(Exception) as exc_info:
            result = conn.execute("SELECT * FROM simple_arrow(5)").fetchall()

        assert (
            "Vector::Reference" in str(exc_info.value)
            or "schema" in str(exc_info.value).lower()
        )


def test_arrow_large_1(tmp_path):
    pa = pytest.importorskip("pyarrow")

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        n = 2048 * 1000

        conn.create_table_function(
            "large_arrow",
            simple_arrow_table,
            schema=[("id", "BIGINT"), ("value", "BIGINT"), ("name", "VARCHAR")],
            type="arrow_table",
        )

        result = conn.execute(
            "SELECT COUNT(*) FROM large_arrow(?)", parameters=(n,)
        ).fetchone()
        assert result[0] == n

        df = conn.sql(f"SELECT * FROM large_arrow({n}) LIMIT 10").df()
        assert len(df) == 10
        assert df["id"].tolist() == list(range(10))

        arrow_result = conn.execute(
            "SELECT * FROM large_arrow(?)", parameters=(n,)
        ).fetch_arrow_table()
        assert len(arrow_result) == n

        result = conn.sql(
            "SELECT SUM(value) FROM large_arrow(?)", params=(n,)
        ).fetchone()
        expected_sum = sum(i * 2 for i in range(n))
        assert result[0] == expected_sum


def test_large_arrow_execute(tmp_path):
    pytest.importorskip("pyarrow")

    count = 2048 * 1000
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        conn.create_table_function(
            name="gen_function",
            callable=simple_generator,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.execute(
            "SELECT * FROM gen_function(?)",
            parameters=(count,),
        ).fetch_arrow_table()

        assert len(result) == count


def test_large_arrow_sql(tmp_path):
    pytest.importorskip("pyarrow")

    count = 2048 * 1000
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        conn.create_table_function(
            name="gen_function",
            callable=simple_generator,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.sql(
            "SELECT * FROM gen_function(?)",
            params=(count,),
        ).fetch_arrow_table()

        assert len(result) == count


def test_arrowbatched_execute(tmp_path):
    pytest.importorskip("pyarrow")

    count = 2048 * 1000
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        conn.create_table_function(
            name="gen_function",
            callable=simple_generator,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.execute(
            "SELECT * FROM gen_function(?)",
            parameters=(count,),
        ).fetch_record_batch()

        result = conn.execute(
            f"SELECT * FROM gen_function({count})",
        ).fetch_record_batch()

        c = 0
        for batch in result:
            c += batch.num_rows
        assert c == count


def test_arrowbatched_sql_relation(tmp_path):
    pytest.importorskip("pyarrow")

    count = 2048 * 1000
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        conn.create_table_function(
            name="gen_function",
            callable=simple_generator,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.sql(
            f"SELECT * FROM gen_function({count})",
        ).fetch_arrow_reader()

        c = 0
        for batch in result:
            c += batch.num_rows
        assert c == count


def test_arrowbatched_sql_materialized(tmp_path):
    pytest.importorskip("pyarrow")

    count = 2048 * 1000
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        conn.create_table_function(
            name="gen_function",
            callable=simple_generator,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        # passing parameters makes it non-lazy /materialized
        result = conn.sql(
            "SELECT * FROM gen_function(?)",
            params=(count,),
        ).fetch_arrow_reader()

        c = 0
        for batch in result:
            c += batch.num_rows
        assert c == count

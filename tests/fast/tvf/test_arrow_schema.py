"""Test Arrow TVF schema validation"""

import pytest

import duckdb
from duckdb.functional import PythonTVFType


def simple_arrow_table(count: int = 10):
    import pyarrow as pa

    data = {
        "id": list(range(count)),
        "value": [i * 2 for i in range(count)],
        "name": [f"row_{i}" for i in range(count)],
    }
    return pa.table(data)


def test_arrow_correct_schema(tmp_path):
    pa = pytest.importorskip("pyarrow")

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            "arrow_func",
            simple_arrow_table,
            schema=[("id", "BIGINT"), ("value", "BIGINT"), ("name", "VARCHAR")],
            type=PythonTVFType.ARROW_TABLE,
        )

        result = conn.execute("SELECT * FROM arrow_func(5)").fetchall()
        assert len(result) == 5
        assert result[0] == (0, 0, "row_0")


def test_arrow_more_columns(tmp_path):
    pa = pytest.importorskip("pyarrow")

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        # table has 3 cols, but declare only 2
        conn.create_table_function(
            "arrow_func",
            simple_arrow_table,
            schema=[("x", "BIGINT"), ("y", "BIGINT")],  # Missing third column
            type=PythonTVFType.ARROW_TABLE,
        )

        with pytest.raises(duckdb.InvalidInputException) as exc_info:
            conn.execute("SELECT * FROM arrow_func(5)").fetchall()

        error_msg = str(exc_info.value).lower()
        assert (
            "schema mismatch" in error_msg
            or "3 columns" in error_msg
            or "2 were declared" in error_msg
        )


def test_arrow_fewer_columns(tmp_path):
    pa = pytest.importorskip("pyarrow")

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        # table has 3 columns, but declare 4
        conn.create_table_function(
            "arrow_func",
            simple_arrow_table,
            schema=[
                ("id", "BIGINT"),
                ("value", "BIGINT"),
                ("name", "VARCHAR"),
                ("extra", "INT"),  # Extra column that doesn't exist
            ],
            type=PythonTVFType.ARROW_TABLE,
        )

        with pytest.raises(duckdb.InvalidInputException) as exc_info:
            conn.execute("SELECT * FROM arrow_func(5)").fetchall()

        error_msg = str(exc_info.value).lower()
        assert (
            "schema mismatch" in error_msg
            or "3 columns" in error_msg
            or "4 were declared" in error_msg
        )


def test_arrow_type_mismatch(tmp_path):
    pa = pytest.importorskip("pyarrow")

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            "arrow_func",
            simple_arrow_table,
            schema=[
                ("id", "VARCHAR"),  # Wrong type - should be BIGINT
                ("value", "BIGINT"),
                ("name", "VARCHAR"),
            ],
            type=PythonTVFType.ARROW_TABLE,
        )

        with pytest.raises(duckdb.InvalidInputException) as exc_info:
            conn.execute("SELECT * FROM arrow_func(5)").fetchall()

        error_msg = str(exc_info.value).lower()
        assert "type" in error_msg or "mismatch" in error_msg


def test_arrow_name_mismatch_allowed(tmp_path):
    pa = pytest.importorskip("pyarrow")

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            "arrow_func",
            simple_arrow_table,
            schema=[
                ("a", "BIGINT"),  # Arrow has 'id'
                ("b", "BIGINT"),  # Arrow has 'value'
                ("c", "VARCHAR"),  # Arrow has 'name'
            ],
            type=PythonTVFType.ARROW_TABLE,
        )

        result = conn.execute("SELECT * FROM arrow_func(3)").fetchall()
        assert len(result) == 3

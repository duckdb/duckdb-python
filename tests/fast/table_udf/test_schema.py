"""Test schema validation for table-valued functions."""

from typing import Iterator

import pytest

import duckdb
import duckdb.sqltypes as sqltypes


def test_valid_schema_basic_types(tmp_path):
    def gen_function():
        return [("test", 42)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql("SELECT * FROM gen_function()").fetchall()
        assert len(result) == 1
        assert result[0] == ("test", 42)


def test_valid_schema_numeric_types(tmp_path):
    def gen_function():
        return [(1, 2, 3, 4, 5, 6.5, 7.25)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {
            "tiny": sqltypes.TINYINT,
            "small": sqltypes.SMALLINT,
            "int": sqltypes.INTEGER,
            "big": sqltypes.BIGINT,
            "huge": sqltypes.HUGEINT,
            "float": sqltypes.FLOAT,
            "double": sqltypes.DOUBLE,
        }

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql("SELECT * FROM gen_function()").fetchall()
        assert len(result) == 1


def test_valid_schema_temporal_types(tmp_path):
    from datetime import date, datetime, time

    def gen_function():
        return [(date(2024, 1, 1), time(12, 30, 45), datetime(2024, 1, 1, 12, 30, 45))]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {
            "d": sqltypes.DATE,
            "t": sqltypes.TIME,
            "ts": sqltypes.TIMESTAMP,
        }

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql("SELECT * FROM gen_function()").fetchall()
        assert len(result) == 1


def test_valid_schema_boolean_and_blob(tmp_path):
    def gen_function():
        return [(True, b"binary_data")]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {
            "flag": sqltypes.BOOLEAN,
            "data": sqltypes.BLOB,
        }

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql("SELECT * FROM gen_function()").fetchall()
        assert len(result) == 1
        assert result[0][0] is True
        assert result[0][1] == b"binary_data"


def test_valid_schema_single_column(tmp_path):
    def gen_function():
        return [(42,), (43,), (44,)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"value": sqltypes.INTEGER}

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql("SELECT * FROM gen_function()").fetchall()
        assert len(result) == 3


def test_valid_schema_many_columns(tmp_path):
    def gen_function():
        return [tuple(range(20))]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {f"col{i}": sqltypes.INTEGER for i in range(20)}

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql("SELECT * FROM gen_function()").fetchall()
        assert len(result) == 1
        assert len(result[0]) == 20


def test_invalid_schema_none(tmp_path):
    def gen_function():
        return [("test", 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        with pytest.raises(duckdb.InvalidInputException, match="Table functions require a schema"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema=None,
                type="tuples",
            )


def test_invalid_schema_empty_dict(tmp_path):
    def gen_function():
        return [("test", 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        with pytest.raises(duckdb.InvalidInputException, match="schema cannot be empty"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema={},
                type="tuples",
            )


def test_invalid_schema_list_format(tmp_path):
    def gen_function():
        return [("test", 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        with pytest.raises(duckdb.InvalidInputException, match="schema must be a dict"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema=schema,
                type="tuples",
            )


def test_invalid_schema_tuple_format(tmp_path):
    def gen_function():
        return [("test", 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [("name", "VARCHAR"), ("id", "INT")]

        with pytest.raises(duckdb.InvalidInputException, match="schema must be a dict"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema=schema,
                type="tuples",
            )


def test_invalid_schema_string_value(tmp_path):
    """Test that string type values are rejected."""

    def gen_function():
        return [("test", 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        # String types should be rejected
        schema = {"name": "VARCHAR", "id": "INT"}

        with pytest.raises(duckdb.InvalidInputException, match="must be a duckdb.sqltype"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema=schema,
                type="tuples",
            )


def test_invalid_schema_integer_value(tmp_path):
    def gen_function():
        return [("test", 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": 123}

        with pytest.raises(duckdb.InvalidInputException, match="must be a duckdb.sqltype"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema=schema,
                type="tuples",
            )


def test_invalid_schema_none_value(tmp_path):
    def gen_function():
        return [("test", 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": None}

        with pytest.raises(duckdb.InvalidInputException, match="must be a duckdb.sqltype"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema=schema,
                type="tuples",
            )


def test_invalid_schema_mixed_types(tmp_path):
    """Test that schema with mix of DuckDBPyType and strings is rejected."""

    def gen_function():
        return [("test", 1, 2.5)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        # Mix of DuckDBPyType and string - should reject strings
        schema = {"name": sqltypes.VARCHAR, "id": "INT", "value": sqltypes.DOUBLE}

        with pytest.raises(duckdb.InvalidInputException, match="must be a duckdb.sqltype"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema=schema,
                type="tuples",
            )


def test_invalid_schema_python_type(tmp_path):
    def gen_function():
        return [("test", 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": str, "id": int}

        with pytest.raises(duckdb.InvalidInputException, match="must be a duckdb.sqltype"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema=schema,
                type="tuples",
            )


def test_invalid_schema_column_name_not_string(tmp_path):
    def gen_function():
        return [(1, 2)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {1: sqltypes.INTEGER, 2: sqltypes.INTEGER}

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql("SELECT * FROM gen_function()").fetchall()
        assert len(result) == 1


def test_schema_column_name_special_characters(tmp_path):
    def gen_function():
        return [("test", 42, 3.14)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {
            "my-column": sqltypes.VARCHAR,
            "another_column": sqltypes.INTEGER,
            "column.with.dots": sqltypes.DOUBLE,
        }

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql('SELECT "my-column", another_column FROM gen_function()').fetchall()
        assert len(result) == 1


def test_schema_preserved_order(tmp_path):
    def gen_function():
        return [(1, 2, 3, 4, 5)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {
            "first": sqltypes.INTEGER,
            "second": sqltypes.INTEGER,
            "third": sqltypes.INTEGER,
            "fourth": sqltypes.INTEGER,
            "fifth": sqltypes.INTEGER,
        }

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql("DESCRIBE select * from gen_function()").fetchall()
        column_names = [row[0] for row in result]
        assert column_names == ["first", "second", "third", "fourth", "fifth"]


def test_schema(tmp_path):
    def gen_function(count: int = 10) -> Iterator[tuple[str, int]]:
        for i in range(count):
            yield (f"name_{i}", i)

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql("SELECT * FROM gen_function(5)").fetchall()
        assert len(result) == 5
        assert result[0][0] == "name_0"
        assert result[-1][-1] == 4


def test_schema_2(tmp_path):
    """Test various types."""

    def gen_function(count: int = 10) -> Iterator[tuple[str, int, float]]:
        for i in range(count):
            yield (f"name_{i}", i, i * 1.5)

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER, "value": sqltypes.DOUBLE}

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        result = conn.sql("SELECT * FROM gen_function(3)").fetchall()
        assert len(result) == 3
        assert result[0] == ("name_0", 0, 0.0)
        assert result[2] == ("name_2", 2, 3.0)


def test_schema_invalid_type(tmp_path):
    def gen_function():
        return [("test", 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": 123}  # int is not valid

        with pytest.raises(duckdb.InvalidInputException, match="must be a duckdb.sqltype"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema=schema,
                type="tuples",
            )


def test_schema_not_dict(tmp_path):
    def gen_function():
        return [("test", 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        # Schema must be dict, not list (this is the old list format)
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        with pytest.raises(duckdb.InvalidInputException, match="schema must be a dict"):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                schema=schema,
                type="tuples",
            )

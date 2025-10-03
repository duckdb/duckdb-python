from typing import Iterator

import pytest

import duckdb
from duckdb.functional import PythonTVFType


def simple_generator(count: int = 10) -> Iterator[tuple[str, int]]:
    for i in range(count):
        yield (f"name_{i}", i)


def simple_pylist(count: int = 10) -> list[tuple[str, int]]:
    return [(f"name_{i}", i) for i in range(count)]


def simple_pylistlist(count: int = 10) -> list[list[str, int]]:
    return [[f"name_{i}", i] for i in range(count)]


@pytest.mark.parametrize(
    "gen_function", [simple_generator, simple_pylist, simple_pylistlist]
)
def test_simple(tmp_path, gen_function):
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            parameters=None,
            schema=schema,
            type=PythonTVFType.TUPLES,
        )

        result = conn.sql("SELECT * FROM gen_function(5)").fetchall()

        assert len(result) == 5
        assert result[0][0] == "name_0"
        assert result[-1][-1] == 4

        result = conn.sql("SELECT * FROM gen_function()").fetchall()

        assert len(result) == 10
        assert result[-1][0] == "name_9"
        assert result[-1][1] == 9


@pytest.mark.parametrize("gen_function", [simple_generator])
def test_simple_large_fetchall(tmp_path, gen_function):
    count = 2048 * 1000
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.sql(
            "SELECT * FROM gen_function(?)",
            params=(count,),
        ).fetchall()

        assert len(result) == count
        assert result[0][0] == "name_0"
        assert result[-1][-1] == count - 1


@pytest.mark.parametrize("gen_function", [simple_generator])
def test_simple_large_df(tmp_path, gen_function):
    count = 2048 * 1000
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.sql(
            "SELECT * FROM gen_function(?)",
            params=(count,),
        ).df()

        assert len(result) == count


def test_no_schema(tmp_path):
    def gen_function(n):
        return n

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        with pytest.raises(duckdb.InvalidInputException):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                type="tuples",
            )


def test_returns_scalar(tmp_path):
    def gen_function(n):
        return n

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        with pytest.raises(duckdb.InvalidInputException):
            conn.create_table_function(
                name="gen_function",
                callable=gen_function,
                parameters=["n"],
                schema=["value"],
                type="tuples",
            )


def test_returns_list_scalar(tmp_path):
    def gen_function_2(n):
        return [n]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        with pytest.raises(duckdb.InvalidInputException):
            conn.create_table_function(
                name="gen_function_2",
                callable=gen_function_2,
                schema=["value"],
                type="tuples",
            )


def test_returns_wrong_schema(tmp_path):
    def gen_function(n):
        return list[range(n)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )
        with pytest.raises(duckdb.InvalidInputException):
            conn.sql("SELECT * FROM gen_function(5)").fetchall()


def test_kwargs(tmp_path):
    def simple_pylist(count, foo=10):
        return [(f"name_{i}_{foo}", i) for i in range(count)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            name="simple_pylist",
            callable=simple_pylist,
            parameters=["count"],
            schema=[["name", "VARCHAR"], ["id", "INT"]],
            type="tuples",
        )
        result = conn.sql("SELECT * FROM simple_pylist(3)").fetchall()
        assert result[-1][0] == "name_2_10"

        result = conn.sql("SELECT * FROM simple_pylist(count:=3)").fetchall()
        assert result[-1][0] == "name_2_10"

        with pytest.raises(duckdb.BinderException):
            result = conn.sql(
                "SELECT * FROM simple_pylist(count:=3, foo:=2)"
            ).fetchall()


def test_large_2(tmp_path):
    """aggregtes and filtering"""
    with duckdb.connect(tmp_path / "test.db") as conn:
        count = 500000

        def large_generator():
            return [(f"item_{i}", i) for i in range(count)]

        schema = [("name", "VARCHAR"), ("id", "INT")]

        conn.create_table_function(
            name="large_tvf",
            callable=large_generator,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.execute("SELECT COUNT(*) FROM large_tvf()").fetchone()
        assert result[0] == count

        result = conn.sql("SELECT MAX(id) FROM large_tvf()").fetchone()
        assert result[0] == count - 1

        result = conn.execute(
            "SELECT COUNT(*) FROM large_tvf() WHERE id < 100"
        ).fetchone()
        assert result[0] == 100


def test__parameters(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:

        def parametrized_function(count=10, prefix="item"):
            return [(f"{prefix}_{i}", i) for i in range(count)]

        schema = [("name", "VARCHAR"), ("id", "INT")]

        conn.create_table_function(
            name="param_tvf",
            callable=parametrized_function,
            parameters=["count", "prefix"],
            schema=schema,
            type="tuples",
        )

        result1 = conn.execute("SELECT COUNT(*) FROM param_tvf(5, 'test')").fetchone()
        assert result1[0] == 5

        result2 = conn.execute(
            "SELECT COUNT(*) FROM param_tvf(20, prefix:='data')"
        ).fetchone()
        assert result2[0] == 20

        # Test parameter order
        result3 = conn.execute(
            "SELECT name FROM param_tvf(3, 'xyz') ORDER BY id LIMIT 1"
        ).fetchone()
        assert result3[0] == "xyz_0"


def test_error(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:

        def error_function():
            raise ValueError("Intentional")

        schema = [("name", "VARCHAR"), ("id", "INT")]

        conn.create_table_function(
            name="error_tvf",
            callable=error_function,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        with pytest.raises(Exception):
            conn.execute("SELECT * FROM error_tvf()").fetchall()


def test_callable_refcount(tmp_path):
    import sys

    def gen_function(n):
        return [(f"name_{i}", i) for i in range(n)]

    initial_refcount = sys.getrefcount(gen_function)

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            schema=schema,
            type="tuples",
        )

        after_register_refcount = sys.getrefcount(gen_function)
        assert after_register_refcount > initial_refcount, (
            f"Expected refcount to increase after registration, "
            f"but got {after_register_refcount} (initial: {initial_refcount})"
        )

        for _ in range(3):
            result = conn.sql("SELECT * FROM gen_function(5)").fetchall()
            assert len(result) == 5

        after_execution_refcount = sys.getrefcount(gen_function)
        assert after_execution_refcount == after_register_refcount, (
            f"Expected refcount to remain stable after execution, "
            f"but got {after_execution_refcount} (after register: {after_register_refcount})"
        )

    final_refcount = sys.getrefcount(gen_function)
    assert final_refcount == initial_refcount, (
        f"Expected refcount to return to initial after unregistration, "
        f"but got {final_refcount} (initial: {initial_refcount})"
    )

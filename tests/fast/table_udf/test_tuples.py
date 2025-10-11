from collections.abc import Iterator

import pytest

import duckdb
import duckdb.sqltypes as sqltypes
from duckdb.functional import PythonTableUDFType


def simple_generator(count: int = 10) -> Iterator[tuple[str, int]]:
    for i in range(count):
        yield (f"name_{i}", i)


def simple_pylist(count: int = 10) -> list[tuple[str, int]]:
    return [(f"name_{i}", i) for i in range(count)]


def simple_pylistlist(count: int = 10) -> list[list[str, int]]:
    return [[f"name_{i}", i] for i in range(count)]


@pytest.mark.parametrize("gen_function", [simple_generator, simple_pylist, simple_pylistlist])
def test_simple(tmp_path, gen_function):
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            parameters=None,
            schema=schema,
            type=PythonTableUDFType.TUPLES,
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
def test_simple_large_fetchall_default_type(tmp_path, gen_function):
    count = 2048 * 1000
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        # don't pass type="tuples" to verify default is tuples
        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            parameters=None,
            schema=schema,
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
        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

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

    with duckdb.connect(tmp_path / "test.duckdb") as conn, pytest.raises((duckdb.InvalidInputException, TypeError)):
        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            type="tuples",
        )


def test_returns_scalar(tmp_path):
    def gen_function(n):
        return n

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            name="gen_function",
            callable=gen_function,
            parameters=["n"],
            schema={"value": sqltypes.INTEGER},
            type="tuples",
        )
        # Error happens at execution time, not registration
        with pytest.raises(duckdb.InvalidInputException):
            conn.sql("SELECT * FROM gen_function(5)").fetchall()


def test_returns_list_scalar(tmp_path):
    def gen_function_2(n):
        return [n]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            name="gen_function_2",
            callable=gen_function_2,
            schema={"value": sqltypes.INTEGER},
            type="tuples",
        )
        # Error happens at execution time, not registration
        with pytest.raises(duckdb.InvalidInputException):
            conn.sql("SELECT * FROM gen_function_2(5)").fetchall()


def test_returns_wrong_schema(tmp_path):
    def gen_function(n):
        return list[range(n)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

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
            schema={"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER},
            type="tuples",
        )
        result = conn.sql("SELECT * FROM simple_pylist(3)").fetchall()
        assert result[-1][0] == "name_2_10"

        result = conn.sql("SELECT * FROM simple_pylist(count:=3)").fetchall()
        assert result[-1][0] == "name_2_10"

        with pytest.raises(duckdb.BinderException):
            result = conn.sql("SELECT * FROM simple_pylist(count:=3, foo:=2)").fetchall()


def test_large_2(tmp_path):
    """Aggregation and filtering."""
    with duckdb.connect(tmp_path / "test.db") as conn:
        count = 500000

        def large_generator():
            return [(f"item_{i}", i) for i in range(count)]

        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function(
            name="large_table_udf",
            callable=large_generator,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.execute("SELECT COUNT(*) FROM large_table_udf()").fetchone()
        assert result[0] == count

        result = conn.sql("SELECT MAX(id) FROM large_table_udf()").fetchone()
        assert result[0] == count - 1

        result = conn.execute("SELECT COUNT(*) FROM large_table_udf() WHERE id < 100").fetchone()
        assert result[0] == 100


def test__parameters(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:

        def parametrized_function(count=10, prefix="item"):
            return [(f"{prefix}_{i}", i) for i in range(count)]

        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function(
            name="param_table_udf",
            callable=parametrized_function,
            parameters=["count", "prefix"],
            schema=schema,
            type="tuples",
        )

        result1 = conn.execute("SELECT COUNT(*) FROM param_table_udf(5, 'test')").fetchone()
        assert result1[0] == 5

        result2 = conn.execute("SELECT COUNT(*) FROM param_table_udf(20, prefix:='data')").fetchone()
        assert result2[0] == 20

        # Test parameter order
        result3 = conn.execute("SELECT name FROM param_table_udf(3, 'xyz') ORDER BY id LIMIT 1").fetchone()
        assert result3[0] == "xyz_0"


def test_error(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:

        def error_function():
            error_message = "Intentional Error"
            raise ValueError(error_message)

        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function(
            name="error_table_udf",
            callable=error_function,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        with pytest.raises(duckdb.Error):
            conn.execute("SELECT * FROM error_table_udf()").fetchall()


def test_callable_refcount(tmp_path):
    import sys

    def gen_function(n):
        return [(f"name_{i}", i) for i in range(n)]

    initial_refcount = sys.getrefcount(gen_function)

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

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


def test_callable_lifetime_in_view(tmp_path):
    # registers a table UDF within a function scope
    # and make sure it's still accessible from another scope (not GC'd)
    with duckdb.connect(tmp_path / "test.duckdb") as conn:

        def create_and_register():
            def gen_data(count=5):
                return [(f"item_{i}", i * 10) for i in range(count)]

            conn.create_table_function(
                name="temp_function",
                callable=gen_data,
                schema={"name": sqltypes.VARCHAR, "value": sqltypes.INTEGER},
                type="tuples",
            )

        create_and_register()

        conn.execute("CREATE VIEW my_view AS SELECT * FROM temp_function(3)")

        # Unregister only allows the function to be reused - it'll still be accessible in the connection
        conn.unregister_table_function("temp_function")

        result = conn.execute("SELECT * FROM my_view").fetchall()
        assert len(result) == 3
        assert result[0] == ("item_0", 0)
        assert result[1] == ("item_1", 10)
        assert result[2] == ("item_2", 20)

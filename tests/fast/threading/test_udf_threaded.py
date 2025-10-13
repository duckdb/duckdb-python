"""Test User Defined Function (UDF)."""

import concurrent.futures

import pytest

import duckdb


def test_concurrent_udf_registration():
    """Test UDF registration."""
    with duckdb.connect(":memory:") as conn:

        def my_add(x: int, y: int) -> int:
            return x + y

        udf_name = "my_add_1"
        conn.create_function(udf_name, my_add)

        result = conn.execute(f"SELECT {udf_name}(1, 2)").fetchone()
        assert result[0] == 3


def test_mixed_udf_operations():
    conn = duckdb.connect(":memory:")
    try:
        # Register and use UDF
        def thread_func(x: int) -> int:
            return x * 2

        udf_name = "thread_func_1"
        conn.create_function(udf_name, thread_func)
        result1 = conn.execute(f"SELECT {udf_name}(5)").fetchone()
        assert result1[0] == 10

        # Simple query
        result2 = conn.execute("SELECT 42").fetchone()
        assert result2[0] == 42

        # Create table and use built-in functions
        conn.execute("CREATE TABLE test_table (x INTEGER)")
        conn.execute("INSERT INTO test_table VALUES (1), (2), (3)")
        result3 = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()
        assert result3[0] == 3
    finally:
        conn.close()


@pytest.mark.parallel_threads(1)
def test_scalar_udf_concurrent():
    num_threads = 5
    conn = duckdb.connect(":memory:")

    # Create test data
    conn.execute("CREATE TABLE numbers (x INTEGER)")
    conn.execute("INSERT INTO numbers SELECT * FROM range(100)")

    # Create a simple scalar UDF instead of vectorized (simpler for testing)
    def simple_square(x: int) -> int:
        """Square a single value."""
        return x * x

    conn.create_function("simple_square", simple_square)

    def execute_scalar_udf(thread_id):
        start = thread_id * 10
        end = start + 10
        query = f"SELECT simple_square(x) FROM numbers WHERE x BETWEEN {start} AND {end}"
        with conn.cursor() as c:
            assert c.execute(query).fetchone()[0] == (start**2)

        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(execute_scalar_udf, i) for i in range(num_threads)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    conn.close()

    assert all(results)

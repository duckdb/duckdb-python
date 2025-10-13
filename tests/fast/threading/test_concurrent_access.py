"""Concurrent access tests for DuckDB Python bindings with free threading support.

These tests verify that the DuckDB Python module can handle concurrent access
from multiple threads safely, testing module state isolation, memory management,
and connection handling under various stress conditions.
"""

import concurrent.futures
import gc
import random
import time

import pytest

import duckdb


def test_concurrent_connections():
    with duckdb.connect() as conn:
        result = conn.execute("SELECT random() as id, random()*2 as doubled").fetchone()
        assert result is not None


@pytest.mark.parallel_threads(1)
def test_shared_connection_stress(num_threads_testing):
    """Test concurrent operations on shared connection using cursors."""
    iterations = 10

    with duckdb.connect(":memory:") as connection:
        connection.execute("CREATE TABLE stress_test (id INTEGER, thread_id INTEGER, value TEXT)")

        def worker_thread(thread_id: int) -> None:
            cursor = connection.cursor()
            for i in range(iterations):
                cursor.execute(
                    "INSERT INTO stress_test VALUES (?, ?, ?)",
                    [i, thread_id, f"thread_{thread_id}_value_{i}"],
                )
                cursor.execute("SELECT COUNT(*) FROM stress_test WHERE thread_id = ?", [thread_id]).fetchone()
                time.sleep(random.uniform(0.0001, 0.001))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_testing) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads_testing)]
            # Wait for all to complete, will raise if any fail
            for future in concurrent.futures.as_completed(futures):
                future.result()

        total_rows = connection.execute("SELECT COUNT(*) FROM stress_test").fetchone()[0]
        expected_rows = num_threads_testing * iterations
        assert total_rows == expected_rows


@pytest.mark.parallel_threads(1)
def test_module_state_isolation():
    """Test that module state is properly accessible."""
    with duckdb.connect(":memory:"):
        assert hasattr(duckdb, "__version__")

        with duckdb.connect() as default_conn:
            result = default_conn.execute("SELECT 'default' as type").fetchone()
            assert result[0] == "default"

        int_type = duckdb.type("INTEGER")
        string_type = duckdb.type("VARCHAR")
        assert int_type is not None
        assert string_type is not None


def test_rapid_connect_disconnect():
    connections_count = 10
    """Test rapid connection creation and destruction."""
    for i in range(connections_count):
        conn = duckdb.connect(":memory:")
        try:
            result = conn.execute("SELECT 1").fetchone()[0]
            assert result == 1
        finally:
            conn.close()

        # Sometimes force GC to increase pressure
        if i % 3 == 0:
            gc.collect()


def test_exception_handling():
    """Test exception handling doesn't affect module state."""
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE TABLE test (x INTEGER)")
        conn.execute("INSERT INTO test VALUES (1), (2), (3)")

        for i in range(10):
            if i % 3 == 0:
                with pytest.raises(duckdb.CatalogException):
                    conn.execute("SELECT * FROM nonexistent_table")
            else:
                result = conn.execute("SELECT COUNT(*) FROM test").fetchone()[0]
                assert result == 3
    finally:
        conn.close()

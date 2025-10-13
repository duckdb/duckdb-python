"""Test connection lifecycle races.

Focused on DuckDBPyConnection constructor and Close
"""

import concurrent.futures
import gc

import pytest

import duckdb


def test_concurrent_connection_creation_destruction():
    conn = duckdb.connect()
    try:
        result = conn.execute("SELECT 1").fetchone()
        assert result[0] == 1
    finally:
        conn.close()


def test_connection_destructor_race():
    conn = duckdb.connect()
    result = conn.execute("SELECT COUNT(*) FROM range(1)").fetchone()
    assert result[0] == 1

    del conn
    gc.collect()


@pytest.mark.parallel_threads(1)
def test_concurrent_close_operations(num_threads_testing):
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE shared_table (id INTEGER, data VARCHAR)")
        conn.execute("INSERT INTO shared_table VALUES (1, 'test')")

        def attempt_close_connection(cursor, thread_id):
            _result = cursor.execute("SELECT COUNT(*) FROM shared_table").fetchone()

            cursor.close()

            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_testing) as executor:
            futures = [executor.submit(attempt_close_connection, conn.cursor(), i) for i in range(num_threads_testing)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert all(results)


@pytest.mark.parallel_threads(1)
def test_cursor_operations_race(num_threads_testing):
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE TABLE cursor_test (id INTEGER, name VARCHAR)")
        conn.execute("INSERT INTO cursor_test SELECT i, 'name_' || i FROM range(100) t(i)")

        def cursor_operations(thread_id):
            """Perform cursor operations concurrently."""
            # Get a cursor
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM cursor_test WHERE id % {num_threads_testing} = {thread_id}")
            cursor.fetchall()

            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_testing) as executor:
            futures = [executor.submit(cursor_operations, i) for i in range(num_threads_testing)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert all(results)
    finally:
        conn.close()


def test_rapid_connection_cycling():
    """Test rapid connection creation and destruction cycles."""
    num_cycles = 5
    for cycle in range(num_cycles):
        conn = duckdb.connect(":memory:")
        try:
            result = conn.execute(f"SELECT 1 + {cycle}").fetchone()
            assert result[0] == 1 + cycle
        finally:
            conn.close()

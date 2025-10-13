"""Test concurrent query execution races.

This tests race conditions in query execution paths where GIL is released
during query processing, as identified in pyconnection.cpp.
"""

import concurrent.futures
import threading
from threading import get_ident

import pytest

import duckdb


class QueryRaceTester:
    """Increases contention by aligning tests w a barrier."""

    def setup_barrier(self, num_threads):
        self.barrier = threading.Barrier(num_threads)

    def synchronized_execute(self, db, query, description="query"):
        with db.cursor() as conn:
            self.barrier.wait()
            conn.execute(query).fetchall()
            return True


@pytest.mark.parallel_threads(1)
def test_concurrent_prepare_execute():
    num_threads = 5
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE TABLE test_data (id INTEGER, value VARCHAR)")
        conn.execute("INSERT INTO test_data SELECT i, 'value_' || i FROM range(1000) t(i)")

        tester = QueryRaceTester()
        tester.setup_barrier(num_threads)

        def prepare_and_execute(thread_id, conn):
            queries = [
                f"SELECT COUNT(*) FROM test_data WHERE id > {thread_id * 10}",
                f"SELECT value FROM test_data WHERE id = {thread_id + 1}",
                f"SELECT id, value FROM test_data WHERE id BETWEEN {thread_id} AND {thread_id + 10}",
                f"INSERT INTO test_data VALUES ({1000 + thread_id}, 'thread_{thread_id}')",
                f"UPDATE test_data SET value = 'updated_{thread_id}' WHERE id = {thread_id + 500}",
            ]

            query = queries[thread_id % len(queries)]
            return tester.synchronized_execute(conn, query, f"Prepared query {thread_id}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(prepare_and_execute, i, conn) for i in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(results) == num_threads
        assert all(results)
    finally:
        conn.close()


@pytest.mark.parallel_threads(1)
def test_concurrent_pending_query_execution():
    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE large_data AS SELECT i, i*2 as double_val, 'row_' || i as str_val FROM range(10000) t(i)"
        )

        num_threads = 8
        tester = QueryRaceTester()
        tester.setup_barrier(num_threads)

        def execute_long_query(thread_id):
            queries = [
                "SELECT COUNT(*), AVG(double_val) FROM large_data",
                "SELECT str_val, double_val FROM large_data WHERE i % 100 = 0 ORDER BY double_val",
                f"SELECT * FROM large_data WHERE i BETWEEN {thread_id * 1000} AND {(thread_id + 1) * 1000}",
                "SELECT i, double_val, str_val FROM large_data WHERE double_val > 5000 ORDER BY i DESC",
                f"SELECT COUNT(*) as cnt FROM large_data WHERE str_val LIKE '%{thread_id}%'",
            ]

            query = queries[thread_id % len(queries)]
            return tester.synchronized_execute(conn, query, f"Long query {thread_id}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(execute_long_query, i) for i in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert all(results)
        assert len(results) == num_threads
    finally:
        conn.close()


def test_execute_many_race():
    """Test executemany operations."""
    iterations = 10
    thread_id = get_ident()

    conn = duckdb.connect()
    try:
        batch_data = [(thread_id * 100 + i, f"name_{thread_id}_{i}") for i in range(iterations)]
        conn.execute("CREATE TABLE batch_data (id BIGINT, name VARCHAR)")
        conn.executemany("INSERT INTO batch_data VALUES (?, ?)", batch_data)
        result = conn.execute(f"SELECT COUNT(*) FROM batch_data WHERE name LIKE 'name_{thread_id}_%'").fetchone()
        assert result[0] == iterations
    finally:
        conn.close()


@pytest.mark.parallel_threads(1)
def test_query_interruption_race():
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE TABLE interrupt_test AS SELECT i FROM range(100000) t(i)")

        num_threads = 6

        def run_interruptible_query(thread_id):
            with conn.cursor() as conn2:
                if thread_id % 2 == 0:
                    # Fast query
                    conn2.execute("SELECT COUNT(*) FROM interrupt_test").fetchall()
                    return True
                else:
                    # Potentially slower query
                    conn2.execute("SELECT i, i*i FROM interrupt_test WHERE i % 1000 = 0 ORDER BY i").fetchall()
                    return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_interruptible_query, i) for i in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures, timeout=30)]

        assert all(results)
    finally:
        conn.close()


def test_mixed_query_operations():
    """Test mixed query operations."""
    thread_id = get_ident()

    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE mixed_ops (id BIGINT PRIMARY KEY, data VARCHAR, num_val DOUBLE)")
        conn.execute("INSERT INTO mixed_ops SELECT i, 'initial_' || i, i * 1.5 FROM range(1000) t(i)")

        queries = [
            f"SELECT COUNT(*) FROM mixed_ops WHERE id > {thread_id * 50}",
            f"INSERT INTO mixed_ops VALUES ({10000 + thread_id}, 'thread_{thread_id}', {thread_id * 2.5})",
            f"UPDATE mixed_ops SET data = 'updated_{thread_id}' WHERE id = {thread_id + 100}",
            "SELECT AVG(num_val), MAX(id) FROM mixed_ops WHERE data LIKE 'initial_%'",
            """
                SELECT m1.id, m1.data, m2.num_val
                FROM mixed_ops m1
                JOIN mixed_ops m2 ON m1.id = m2.id - 1
                LIMIT 10
            """,
        ]

        for query in queries:
            result = conn.execute(query)
            if "SELECT" in query.upper():
                rows = result.fetchall()
                assert len(rows) >= 0

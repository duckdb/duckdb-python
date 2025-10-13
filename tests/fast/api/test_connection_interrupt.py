import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import duckdb


class TestConnectionInterrupt:
    @pytest.mark.xfail(
        condition=platform.system() == "Emscripten",
        reason="threads not allowed on Emscripten",
    )
    @pytest.mark.timeout(10)
    def test_connection_interrupt(self):
        conn = duckdb.connect()
        barrier = threading.Barrier(2)

        def execute_query():
            barrier.wait()
            return conn.execute("select * from range(1000000) t1, range(1000000) t2").fetchall()

        def interrupt_query():
            barrier.wait()
            time.sleep(2)
            conn.interrupt()

        with ThreadPoolExecutor() as executor:
            query_future = executor.submit(execute_query)
            interrupt_future = executor.submit(interrupt_query)

            interrupt_future.result()

            with pytest.raises((duckdb.InterruptException, duckdb.InvalidInputException)):
                query_future.result()

    def test_interrupt_closed_connection(self):
        conn = duckdb.connect()
        conn.close()
        with pytest.raises(duckdb.ConnectionException):
            conn.interrupt()

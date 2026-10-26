import _thread as thread
import platform
import threading
import time

import pytest

import duckdb


def drain(res):
    while res.fetchone() is not None:
        pass


def send_keyboard_interrupt():
    # Wait a little, so we're sure the 'execute' has started
    time.sleep(0.1)
    # Send an interrupt to the main thread
    thread.interrupt_main()


class TestQueryInterruption:
    @pytest.mark.xfail(
        condition=platform.system() == "Emscripten",
        reason="Emscripten builds cannot use threads",
    )
    def test_query_interruption(self):
        con = duckdb.connect()
        thread = threading.Thread(target=send_keyboard_interrupt)
        # Start the thread
        thread.start()
        try:
            con.execute("select count(*) from range(100000000000)").fetchall()
        except RuntimeError:
            # If this is not reached, we could not cancel the query before it completed
            # indicating that the query interruption functionality is broken
            assert True
        except KeyboardInterrupt:
            pytest.fail("Interrupted by user")
        thread.join()

    @pytest.mark.xfail(
        condition=platform.system() == "Emscripten",
        reason="Emscripten builds cannot use threads",
    )
    @pytest.mark.timeout(120)
    def test_streaming_fetch_interruption(self):
        con = duckdb.connect()
        con.execute("SET max_streaming_buffer_size='1MB'")
        con.execute("select i from range(8000000000) t(i)")
        assert con.fetchone() == (0,)

        interrupter = threading.Thread(target=send_keyboard_interrupt)
        interrupter.start()
        # The interrupt lands either in the fetch loop's own signal check or between two fetches in
        # the interpreter, so either exception proves the drain stopped.
        with pytest.raises((RuntimeError, KeyboardInterrupt)):
            drain(con)
        interrupter.join()

    @pytest.mark.xfail(
        condition=platform.system() == "Emscripten",
        reason="Emscripten builds cannot use threads",
    )
    @pytest.mark.timeout(120)
    def test_materializing_fetch_interruption(self):
        con = duckdb.connect()
        rel = con.sql("select count(*) from range(8000000000)")
        interrupter = threading.Thread(target=send_keyboard_interrupt)
        interrupter.start()
        with pytest.raises(RuntimeError, match="Query interrupted"):
            rel.df()
        interrupter.join()

import _thread as thread
import platform
import threading
import time

import pytest

import duckdb


class TestQueryInterruption:
    @pytest.mark.xfail(
        condition=platform.system() == "Emscripten",
        reason="Emscripten builds cannot use threads",
    )
    @pytest.mark.timeout(10)
    def test_query_interruption(self):
        con = duckdb.connect()
        barrier = threading.Barrier(2)

        def send_keyboard_interrupt():
            barrier.wait()
            time.sleep(2)
            thread.interrupt_main()

        interrupt_thread = threading.Thread(target=send_keyboard_interrupt)
        interrupt_thread.start()

        barrier.wait()

        with pytest.raises((KeyboardInterrupt, RuntimeError)):
            con.execute("select * from range(1000000) t1,range(1000000) t2").fetchall()

        interrupt_thread.join()

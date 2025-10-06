import platform
import threading
import time
import _thread as thread

import duckdb
import pytest


class TestQueryInterruption(object):
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

        with pytest.raises((KeyboardInterrupt, RuntimeError)):
            barrier.wait()
            con.execute('select * from range(1000000) t1,range(1000000) t2').fetchall()

        interrupt_thread.join()

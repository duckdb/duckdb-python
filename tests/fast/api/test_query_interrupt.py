import duckdb
import time
import pytest
import platform
import threading
import _thread as thread


def send_keyboard_interrupt():
    time.sleep(0.1)
    thread.interrupt_main()


class TestQueryInterruption(object):
    
    @pytest.mark.xfail(
        condition=platform.system() == "Emscripten",
        reason="Emscripten builds cannot use threads",
    )
    @pytest.mark.timeout(15)
    def test_keyboard_interruption(self):
        con = duckdb.connect()
        thread = threading.Thread(target=send_keyboard_interrupt)
        # Start the thread
        thread.start()
        try:
            with pytest.raises((KeyboardInterrupt, RuntimeError)):
                res = con.execute('select * from range(100000) t1,range(100000) t2').fetchall()
        finally:
            # Ensure the thread completes regardless of what happens
            thread.join()

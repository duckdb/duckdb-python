import platform
import threading
import time
import sys
import duckdb
import pytest


class TestConnectionInterrupt(object):
    @pytest.mark.xfail(sys.platform == "win32" and sys.version_info[:2] == (3, 14) and __import__('sysconfig').get_config_var("Py_GIL_DISABLED") == 1, reason="known issue on Windows 3.14t (free-threaded)", strict=False)
    @pytest.mark.xfail(
        condition=platform.system() == "Emscripten",
        reason="threads not allowed on Emscripten",
    )
    def test_connection_interrupt(self):
        conn = duckdb.connect()

        def interrupt():
            # Wait for query to start running before interrupting
            time.sleep(0.1)
            conn.interrupt()

        thread = threading.Thread(target=interrupt)
        thread.start()
        with pytest.raises(duckdb.InterruptException):
            conn.execute("select count(*) from range(100000000000)").fetchall()
        thread.join()

    def test_interrupt_closed_connection(self):
        conn = duckdb.connect()
        conn.close()
        with pytest.raises(duckdb.ConnectionException):
            conn.interrupt()

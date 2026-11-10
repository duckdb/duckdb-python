import contextlib
import platform
import subprocess
import sys
import threading
import time

import pytest

import duckdb


class TestQueryProgress:
    @pytest.mark.xfail(
        condition=platform.system() == "Emscripten",
        reason="threads not allowed on Emscripten",
    )
    def test_query_progress(self, reraise):
        conn = duckdb.connect()
        conn.sql("set enable_progress_bar_print=false")
        conn.sql("set progress_bar_time=0")
        conn.sql("create table t as (select range as n from range(10000000))")

        def thread_target() -> None:
            # run a very slow query which hopefully isn't too memory intensive.
            with reraise, contextlib.suppress(duckdb.InterruptException):
                conn.execute("select max(sha1(n::varchar)) from t").fetchall()

        thread = threading.Thread(target=thread_target)
        thread.start()

        # monitor the query running in the thread, wait for progress > 0
        # the 'for/else' is just so the test times out after 5 seconds if the
        # query never progresses.  This will also fail if the query is too
        # quick as it will be back at -1 as soon as the query is finished.

        for _ in range(500):
            assert thread.is_alive(), "query finished too quick"
            if (qp1 := conn.query_progress()) > 0:
                break
            time.sleep(0.01)
        else:
            pytest.fail("query start timeout")

        # keep monitoring and wait for the progress to increase
        for _ in range(500):
            assert thread.is_alive(), "query finished too quick"
            if (qp2 := conn.query_progress()) > qp1:
                break
            time.sleep(0.01)
        else:
            pytest.fail("query progress timeout")

        # check that progress numbers are sensible
        assert 100 >= qp2 > qp1 > 0

        # kill the query to reduce CPU usage.
        conn.interrupt()
        thread.join()

    def test_query_progress_closed_connection(self):
        conn = duckdb.connect()
        conn.close()
        with pytest.raises(duckdb.ConnectionException):
            conn.query_progress()


_JUPYTER_KERNEL = """
import sys
import types

shown = []
ipython = types.ModuleType("IPython")


class Shell:
    config = {"IPKernelApp": {}}


ipython.get_ipython = Shell
display = types.ModuleType("IPython.display")
# Like IPython's, this display keeps the widgets it shows, so the engine never drops the last reference.
display.display = shown.append
ipython.display = display
widgets = types.ModuleType("ipywidgets")


class FloatProgress:
    def __init__(self, **kwargs):
        self.layout = types.SimpleNamespace(width=None)
        self.value = None


widgets.FloatProgress = FloatProgress
sys.modules.update({"IPython": ipython, "IPython.display": display, "ipywidgets": widgets})
import duckdb

con = duckdb.connect()
con.execute("SET progress_bar_time = 0")
con.execute("SELECT sum(a.range * b.range) FROM range(1000000) a, range(4) b").fetchall()
print("shown", [type(widget).__name__ for widget in shown])
"""


def test_a_jupyter_kernel_shows_the_progress_bar_as_a_widget():
    # The client detects a kernel at import, from a __main__ without __file__ and the shell's config.
    child = subprocess.run(
        [sys.executable, "-c", _JUPYTER_KERNEL], capture_output=True, text=True, timeout=60, check=False
    )
    assert child.returncode == 0, child.stderr
    assert "shown ['FloatProgress']" in child.stdout.splitlines(), child.stdout

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
# Stands in for the ipywidgets registry, which keeps every widget until it is closed, so none is freed by the engine.
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
print("enabled", con.execute("SELECT current_setting('enable_progress_bar')").fetchone()[0])
con.execute("SET progress_bar_time = 0")
con.execute("SELECT sum(a.range * b.range) FROM range(1000000) a, range(4) b").fetchall()
print("shown", [type(widget).__name__ for widget in shown])
"""


def _reported(stdout: str) -> list[str]:
    # The editable build prints its own lines in a child process
    return [
        line
        for line in stdout.splitlines()
        if line.startswith(("enabled", "cursor", "after set", "after reset", "refused"))
    ]


_READ_AND_SET = """
import duckdb

con = duckdb.connect()
print("enabled", con.execute("SELECT current_setting('enable_progress_bar')").fetchone()[0])
print("cursor", con.cursor().execute("SELECT current_setting('enable_progress_bar')").fetchone()[0])
con.execute("SET enable_progress_bar = false")
print("after set", con.execute("SELECT current_setting('enable_progress_bar')").fetchone()[0])
con.execute("RESET enable_progress_bar")
print("after reset", con.execute("SELECT current_setting('enable_progress_bar')").fetchone()[0])
"""


def test_an_interactive_session_enables_the_progress_bar(tmp_path):
    # The client treats a __main__ without __file__ as an interactive session, so a child run from -c is one and a
    # child run from a script is not. The value is the session's own: a cursor, which gets no default, reads the
    # engine's default, and a RESET returns to it.
    interactive = subprocess.run(
        [sys.executable, "-c", _READ_AND_SET], capture_output=True, text=True, timeout=60, check=False
    )
    assert interactive.returncode == 0, interactive.stderr
    assert _reported(interactive.stdout) == ["enabled True", "cursor False", "after set False", "after reset False"]

    script = tmp_path / "script.py"
    script.write_text(_READ_AND_SET)
    plain = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60, check=False)
    assert plain.returncode == 0, plain.stderr
    assert _reported(plain.stdout) == ["enabled False", "cursor False", "after set False", "after reset False"]


_KERNEL_WITHOUT_IPYWIDGETS = """
import sys
import types

ipython = types.ModuleType("IPython")


class Shell:
    config = {"IPKernelApp": {}}


ipython.get_ipython = Shell
sys.modules.update({"IPython": ipython, "ipywidgets": None})
import duckdb

con = duckdb.connect()
print("enabled", con.execute("SELECT current_setting('enable_progress_bar')").fetchone()[0])
try:
    con.execute("SET enable_progress_bar = true")
except duckdb.InvalidInputException as error:
    print("refused", str(error))
"""


def test_a_jupyter_kernel_without_ipywidgets_refuses_the_progress_bar():
    child = subprocess.run(
        [sys.executable, "-c", _KERNEL_WITHOUT_IPYWIDGETS], capture_output=True, text=True, timeout=60, check=False
    )
    assert child.returncode == 0, child.stderr
    lines = _reported(child.stdout)
    assert len(lines) == 2
    assert lines[0] == "enabled False"
    assert lines[1].startswith("refused")
    assert "required package 'ipywidgets' is missing" in lines[1]


def test_a_jupyter_kernel_shows_the_progress_bar_as_a_widget():
    # The client detects a kernel at import, from a __main__ without __file__ and the shell's config.
    child = subprocess.run(
        [sys.executable, "-c", _JUPYTER_KERNEL], capture_output=True, text=True, timeout=60, check=False
    )
    assert child.returncode == 0, child.stderr
    # Read before the SET of progress_bar_time, which switches the bar on by itself
    assert "enabled True" in child.stdout.splitlines(), child.stdout
    assert "shown ['FloatProgress']" in child.stdout.splitlines(), child.stdout

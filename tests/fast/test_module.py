import subprocess
import sys

import duckdb


class TestModule:
    def test_paramstyle(self):
        assert duckdb.paramstyle == "qmark"

    def test_threadsafety(self):
        assert duckdb.threadsafety == 1

    def test_apilevel(self):
        assert duckdb.apilevel == "2.0"


class TestModuleShutdown:
    """Module state is static, so its members can be destroyed *after* the interpreter is gone.

    Deleting '_clean_default_connection' forces that: nothing then releases the import cache
    or default connection while the interpreter is alive, so both are torn down from static
    destruction, where neither may touch the GIL.
    """

    def test_module_state_freed_after_finalize(self):
        code = """\
import _duckdb
import duckdb

del _duckdb._clean_default_connection
del duckdb._clean_default_connection

assert duckdb.sql("select 42").fetchall() == [(42,)]
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr}"

    def test_import_cache_released_on_a_normal_exit(self):
        # Passing a duckdb.Value caches the 'duckdb' module itself, so an
        # uncleared cache keeps it alive past nanobind's leak check.
        code = """\
import duckdb

value = duckdb.Value('{"duck": 42}', duckdb.type("JSON"))
assert duckdb.execute("select typeof($1)", [value]).fetchone() == ("JSON",)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr}"
        assert "leaked" not in result.stderr, result.stderr

    def test_no_crash_when_process_exits_without_finalize(self):
        # nanobind stays 'alive' if Py_Finalize never runs as its cleanup is a Py_AtExit hook.
        code = """\
import ctypes
import sys

import duckdb

assert duckdb.sql("select 42").fetchall() == [(42,)]

# ctypes drops the GIL and never returns, so static destruction runs without it.
if sys.platform == "win32":
    ctypes.windll.kernel32.ExitProcess(0)
else:
    ctypes.CDLL(None).exit(0)

raise AssertionError("unreachable")
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr}"

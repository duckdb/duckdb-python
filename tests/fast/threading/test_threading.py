#!/usr/bin/env python3
"""Tests designed to expose specific threading bugs in the DuckDB implementation."""

import sys
from threading import get_ident

import duckdb


def test_gil_enabled():
    # Safeguard to ensure GIL is disabled if this is a free-threading build to ensure test validity
    # this would fail if tests were run with PYTHON_GIL=1, as one example
    if "free-threading" in sys.version:
        import sysconfig

        print(f"Free-threading Python detected: {sys.version}")
        print(f"Py_GIL_DISABLED = {sysconfig.get_config_var('Py_GIL_DISABLED')}")

        assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1, (
            f"Py_GIL_DISABLED must be 1 in free-threading build, got: {sysconfig.get_config_var('Py_GIL_DISABLED')}"
        )


def test_instance_cache_race(tmp_path):
    """Test opening connections to different files."""
    tid = get_ident()
    with duckdb.connect(tmp_path / f"{tid}_testing.db") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS test (x INTEGER, y INTEGER)")
        conn.execute("INSERT INTO test VALUES (123, 456)")
        result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
        assert result[0] >= 1

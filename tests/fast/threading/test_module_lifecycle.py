"""Test module lifecycle.

Reloading and unload are not expected nor required behaviors -
these tests are to document current behavior so that changes
are visible.
"""

import importlib
import sys
from threading import get_ident

import pytest


@pytest.mark.parallel_threads(1)
def test_module_reload_safety():
    """Test module reloading scenarios to detect use-after-free issues."""
    import duckdb

    with duckdb.connect(":memory:") as conn1:
        conn1.execute("CREATE TABLE test (id INTEGER)")
        conn1.execute("INSERT INTO test VALUES (1)")
        result1 = conn1.execute("SELECT * FROM test").fetchone()[0]
        assert result1 == 1

        initial_module_id = id(sys.modules["duckdb"])

        # Test importlib.reload() -
        # does NOT create new module in Python
        importlib.reload(duckdb)

        # Verify module instance is the same (expected Python behavior)
        reload_module_id = id(sys.modules["duckdb"])
        assert initial_module_id == reload_module_id, "importlib.reload() should reuse same module instance"

        # Test if old connection still works after importlib.reload()
        result2 = conn1.execute("SELECT * FROM test").fetchone()[0]
        assert result2 == 1

        # Test new connection after importlib.reload()
        with duckdb.connect(":memory:") as conn2:
            conn2.execute("CREATE TABLE test2 (id INTEGER)")
            conn2.execute("INSERT INTO test2 VALUES (2)")
            result3 = conn2.execute("SELECT * FROM test2").fetchone()[0]
            assert result3 == 2


@pytest.mark.parallel_threads(1)
def test_dynamic_module_loading():
    import duckdb

    with duckdb.connect(":memory:") as conn:
        conn.execute("SELECT 1").fetchone()

    module_id_1 = id(sys.modules["duckdb"])

    # "Unload" module (not really, just to try it)
    if "duckdb" in sys.modules:
        del sys.modules["duckdb"]

    # Remove from local namespace
    if "duckdb" in locals():
        del duckdb

    # Verify module is unloaded
    assert "duckdb" not in sys.modules, "Module not properly unloaded"

    # import (load) module
    import duckdb

    module_id_2 = id(sys.modules["duckdb"])

    # Verify we have a new module instance
    assert module_id_1 != module_id_2, "Module not actually reloaded"

    # Test functionality after reload
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (42)")
        result = conn.execute("SELECT * FROM test").fetchone()[0]
        assert result == 42


def test_import_cache_consistency():
    """Test that import cache remains consistent across module operations."""
    import pandas as pd

    import duckdb

    conn = duckdb.connect(":memory:")

    df = pd.DataFrame({"a": [1, 2, 3]})

    conn.register("test_df", df)
    result = conn.execute("SELECT COUNT(*) FROM test_df").fetchone()[0]
    assert result == 3

    conn.close()


def test_module_state_memory_safety():
    """Test memory safety of module state access patterns."""
    import duckdb

    connections = []
    for i in range(10):
        conn = duckdb.connect(":memory:")
        conn.execute(f"CREATE TABLE test_{i} (id INTEGER)")
        conn.execute(f"INSERT INTO test_{i} VALUES ({i})")
        connections.append(conn)

    import gc

    gc.collect()

    for i, conn in enumerate(connections):
        result = conn.execute(f"SELECT * FROM test_{i}").fetchone()[0]
        assert result == i

    for conn in connections:
        conn.close()


def test_static_cache_stress():
    """Test rapid module state access."""
    import duckdb

    iterations = 5
    for _ in range(iterations):
        conn = duckdb.connect(":memory:")
        result = conn.execute("SELECT 1").fetchone()
        assert result[0] == 1
        conn.close()


def test_concurrent_module_access():
    import duckdb

    thread_id = get_ident()
    with duckdb.connect(":memory:") as conn:
        conn.execute(f"CREATE TABLE test_{thread_id} (id BIGINT)")
        conn.execute(f"INSERT INTO test_{thread_id} VALUES ({thread_id})")
        result = conn.execute(f"SELECT * FROM test_{thread_id}").fetchone()[0]
        assert result == thread_id

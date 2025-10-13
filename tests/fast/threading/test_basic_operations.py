import gc
import random
import time
import uuid
import weakref
from threading import get_ident

import duckdb


def test_basic():
    with duckdb.connect(":memory:") as conn:
        result = conn.execute("SELECT 1").fetchone()
        assert result[0] == 1
        int_type = duckdb.type("INTEGER")
        assert int_type is not None, "type creation failed"


def test_connection_instance_cache(tmp_path):
    thread_id = get_ident()
    for i in range(10):
        with duckdb.connect(tmp_path / f"{thread_id}_{uuid.uuid4()}.db") as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS thread_{thread_id}_data_{i} (x BIGINT)")
            conn.execute(f"INSERT INTO thread_{thread_id}_data_{i} VALUES (100), (100)")

            time.sleep(random.uniform(0.0001, 0.001))

            result = conn.execute(f"SELECT COUNT(*) FROM thread_{thread_id}_data_{i}").fetchone()[0]
            assert result == 2, f"Iteration {i}: expected 2 rows, got {result}"


def test_cleanup():
    weak_refs = []

    for i in range(5):
        conn = duckdb.connect(":memory:")
        weak_refs.append(weakref.ref(conn))
        try:
            conn.execute("CREATE TABLE test (x INTEGER)")
            conn.execute("INSERT INTO test VALUES (1), (2), (3)")
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] == 3
        finally:
            conn.close()
            conn = None

        if i % 3 == 0:
            with duckdb.connect(":memory:") as new_conn:
                result = new_conn.execute("SELECT 1").fetchone()
                assert result[0] == 1

        if i % 10 == 0:
            gc.collect()
            time.sleep(random.uniform(0.0001, 0.0005))

    gc.collect()
    time.sleep(0.1)
    gc.collect()

    alive_refs = [ref for ref in weak_refs if ref() is not None]
    assert len(alive_refs) <= 10, f"{len(alive_refs)} connections still alive (expected <= 10)"


def test_default_connection():
    with duckdb.connect() as conn1:
        r1 = conn1.execute("SELECT 1").fetchone()[0]
        assert r1 == 1, f"expected 1, got {r1}"

    with duckdb.connect(":memory:") as conn2:
        r2 = conn2.execute("SELECT 2").fetchone()[0]
        assert r2 == 2, f"expected 2, got {r2}"


def test_type_system():
    for i in range(20):
        types = [
            duckdb.type("INTEGER"),
            duckdb.type("VARCHAR"),
            duckdb.type("DOUBLE"),
            duckdb.type("BOOLEAN"),
            duckdb.list_type(duckdb.type("INTEGER")),
            duckdb.struct_type({"a": duckdb.type("INTEGER"), "b": duckdb.type("VARCHAR")}),
        ]

        for t in types:
            assert t is not None, "type creation failed"

        if i % 5 == 0:
            with duckdb.connect(":memory:") as conn:
                conn.execute("CREATE TABLE test (a INTEGER, b VARCHAR, c DOUBLE, d BOOLEAN)")
                result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
                assert result[0] == 0


def test_import_cache():
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE test AS SELECT range as x FROM range(10)")
        result = conn.fetchdf()
        assert len(result) > 0, "fetchdf failed"

        result = conn.execute("SELECT range as x FROM range(5)").fetchnumpy()
        assert len(result["x"]) == 5, "fetchnumpy failed"

        conn.execute("DROP TABLE test")

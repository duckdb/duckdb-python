from threading import get_ident

import duckdb


def test_concurrent_connection_creation():
    thread_id = get_ident()
    for i in range(5):
        with duckdb.connect(":memory:") as conn:
            conn.execute(f"CREATE TABLE test_{i} (x BIGINT)")
            conn.execute(f"INSERT INTO test_{i} VALUES ({thread_id})")
            result = conn.execute(f"SELECT * FROM test_{i}").fetchall()
            assert result == [(thread_id,)], f"Table {i} failed"


def test_concurrent_instance_cache_access(tmp_path):
    thread_id = get_ident()
    for i in range(10):
        db_path = str(tmp_path / f"test_{thread_id}_{i}.db")
        with duckdb.connect(db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS test (x BIGINT, thread_id BIGINT)")
            conn.execute(f"INSERT INTO test VALUES ({i}, {thread_id})")
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] >= 1


def test_environment_detection():
    version = duckdb.__formatted_python_version__
    interactive = duckdb.__interactive__

    assert isinstance(version, str), "version should be string"
    assert isinstance(interactive, bool), "interactive should be boolean"

    with duckdb.connect(":memory:") as conn:
        result = conn.execute("SELECT 1").fetchone()
        assert result[0] == 1

"""Test fetching operations."""

from threading import get_ident

import duckdb


def test_fetching():
    """Test different fetching methods."""
    iterations = 10
    thread_id = get_ident()

    conn = duckdb.connect()
    try:
        batch_data = [(thread_id * 100 + i, f"name_{thread_id}_{i}") for i in range(iterations)]
        conn.execute("CREATE TABLE batch_data (id BIGINT, name VARCHAR)")
        conn.executemany("INSERT INTO batch_data VALUES (?, ?)", batch_data)

        # Test different fetch methods
        result1 = conn.execute(f"SELECT COUNT(*) FROM batch_data WHERE name LIKE 'name_{thread_id}_%'").fetchone()
        assert result1[0] == iterations

        result2 = conn.execute(f"SELECT COUNT(*) FROM batch_data WHERE name LIKE 'name_{thread_id}_%'").fetchall()
        assert result2[0][0] == iterations

        result3 = conn.execute(f"SELECT COUNT(*) FROM batch_data WHERE name LIKE 'name_{thread_id}_%'").fetchdf()
        assert len(result3) == 1

        result4 = conn.execute(
            f"SELECT COUNT(*) FROM batch_data WHERE name LIKE 'name_{thread_id}_%'"
        ).fetch_arrow_table()
        assert result4.num_rows == 1
    finally:
        conn.close()

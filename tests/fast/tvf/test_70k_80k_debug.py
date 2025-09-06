#!/usr/bin/env python3
"""
Debug the 70k vs 80k hanging issue
"""

import duckdb
import time
import threading
import sys


def generate_names(count: int = 500):
    """Generate a list of sample names with progress tracking."""
    print(f"[DEBUG] Python function called with count={count}")
    sys.stdout.flush()

    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    result = []

    for i in range(count):
        result.append((names[i % len(names)], i))

        # Print progress every 10k items
        if i > 0 and i % 10000 == 0:
            print(f"[DEBUG] Generated {i} items so far...")
            sys.stdout.flush()

    print(f"[DEBUG] Python function completed, returning {len(result)} items")
    sys.stdout.flush()
    return result


def test_specific_size(size=1000, timeout=30):
    """Test a specific size with timeout and monitoring."""
    print(f"\n🔬 Testing {size} rows...")

    def timeout_handler():
        """Kill the test after timeout."""
        time.sleep(timeout)
        print(f"\n❌ {size} rows test timed out after {timeout}s!")
        import os

        os._exit(1)

    # Start timeout thread
    timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
    timeout_thread.start()

    try:
        start_time = time.time()

        with duckdb.connect() as conn:
            schema = [["name", "VARCHAR"], ["id", "INT"]]

            print(f"[DEBUG] Registering TVF for {size} rows...")
            tvf = conn.create_table_function(
                name=f"test_{size}",
                callable=generate_names,
                parameters=None,
                schema=schema,
                return_type="records",
            )
            print(f"[DEBUG] TVF registered successfully for {size} rows")

            print(f"[DEBUG] Starting query execution for {size} rows...")
            result = conn.sql(f"SELECT COUNT(*) FROM test_{size}({size})").fetchone()

            elapsed = time.time() - start_time
            print(f"✅ {size} rows: {result[0]} in {elapsed:.3f}s")

            assert result[0] == size
            return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {size} rows failed after {elapsed:.3f}s: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔍 Debugging 70k vs 80k hanging issue...")

    # Test a range around the problem area
    test_sizes = [65000, 70000, 75000, 76000, 77000, 78000, 79000, 80000, 81000, 85000]

    results = {}
    for size in test_sizes:
        success = test_specific_size(size, timeout=20)
        results[size] = success

        if not success:
            print(f"🚨 Found hanging at {size} rows!")
            break

    print(f"\n📊 Results:")
    for size, success in results.items():
        status = "✅ PASS" if success else "❌ HANG"
        print(f"  {size:,} rows: {status}")

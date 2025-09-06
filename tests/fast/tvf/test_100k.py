#!/usr/bin/env python3
"""
Test the 100k records case that was hanging
"""

import duckdb
import time

def generate_names(count: int = 500):
    """Generate a list of sample names."""
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    return [(names[i % len(names)], i) for i in range(count)]

# Test with smaller sizes first
print("Testing 10k rows...")
with duckdb.connect() as conn:
    schema = [["name", "VARCHAR"], ["id", "INT"]]
    tvf = conn.create_table_function(
        name="generate_names",
        callable=generate_names,
        parameters=None,
        schema=schema,
        return_type="records",
    )

    start = time.time()
    result = conn.execute("SELECT COUNT(*) FROM generate_names(10000)").fetchall()
    elapsed = time.time() - start
    print(f"10k rows: {result[0][0]} in {elapsed:.3f}s")

print("\nTesting 50k rows...")
with duckdb.connect() as conn:
    schema = [["name", "VARCHAR"], ["id", "INT"]]
    tvf = conn.create_table_function(
        name="generate_names",
        callable=generate_names,
        parameters=None,
        schema=schema,
        return_type="records",
    )

    start = time.time()
    result = conn.execute("SELECT COUNT(*) FROM generate_names(50000)").fetchall()
    elapsed = time.time() - start
    print(f"50k rows: {result[0][0]} in {elapsed:.3f}s")

print("\nTesting 100k rows...")
try:
    with duckdb.connect() as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]
        tvf = conn.create_table_function(
            name="generate_names",
            callable=generate_names,
            parameters=None,
            schema=schema,
            return_type="records",
        )

        start = time.time()
        result = conn.execute("SELECT COUNT(*) FROM generate_names(100000)").fetchall()
        elapsed = time.time() - start
        print(f"100k rows: {result[0][0]} in {elapsed:.3f}s")
        print("✓ 100k test passed!")

except Exception as e:
    print(f"❌ 100k test failed: {e}")
    import traceback
    traceback.print_exc()
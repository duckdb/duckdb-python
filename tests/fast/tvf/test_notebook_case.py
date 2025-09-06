#!/usr/bin/env python3
"""
Test the exact notebook case that's hanging
"""

import duckdb


def generate_names(count: int = 500):
    """Generate a list of sample names."""
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    return [(names[i % len(names)], i) for i in range(count)]


print("Testing the exact notebook case...")

try:
    with duckdb.connect() as conn:
        # Register the TVF with schema definition
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        tvf = conn.create_table_function(
            name="generate_names",
            callable=generate_names,
            parameters=None,  # Will infer from function signature
            schema=schema,
            return_type="records",  # Return list of tuples
        )

        print("TVF registered successfully")

        # Test different ways of calling the function
        print("Testing with conn.execute()...")
        result1 = conn.execute("SELECT COUNT(*) FROM generate_names(100000)").fetchall()
        print(f"conn.execute() result: {result1[0][0]} rows")

        print("Testing with conn.sql()...")
        result2 = conn.sql("SELECT COUNT(*) FROM generate_names(100000)").fetchall()
        print(f"conn.sql() result: {result2[0][0]} rows")

        print("Testing full query with conn.sql()...")
        result3 = conn.sql("SELECT * FROM generate_names(100000)").df()
        print(f"conn.sql().df() result: {len(result3)} rows")

        assert len(result3) == 100000
        print("✓ All tests passed!")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback

    traceback.print_exc()

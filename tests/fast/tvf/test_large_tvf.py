#!/usr/bin/env python3
"""
Test large TVF with debug output to find hanging point.
"""

import duckdb
import time

def simple_list_tvf(count: int = 100000):
    """Return a simple list."""
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    return [(names[i % len(names)], i) for i in range(count)]

print("🔬 Testing large TVF with 100,000 rows...")
print("This should show where exactly it hangs with debug output")

try:
    with duckdb.connect() as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]
        
        print("Registering TVF...")
        tvf = conn.create_table_function(
            name="test_large",
            callable=simple_list_tvf,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        print("Running query...")
        start_time = time.time()
        result = conn.sql("SELECT COUNT(*) FROM test_large(100000)").fetchone()
        elapsed = time.time() - start_time
        
        if result[0] == 100000:
            print(f"✅ Success: {result[0]} rows in {elapsed:.3f}s")
        else:
            print(f"❌ Wrong count: {result[0]} != 100000")
            
except Exception as e:
    elapsed = time.time() - start_time
    print(f"❌ Error after {elapsed:.3f}s: {e}")
    import traceback
    traceback.print_exc()

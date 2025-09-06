#!/usr/bin/env python3
"""
Test that simulates VS Code notebook cell execution behavior
"""

import duckdb
import time

def generate_names(count: int = 500):
    """Generate a list of sample names.""" 
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    return [(names[i % len(names)], i) for i in range(count)]

print("🔬 Testing VS Code notebook behavior with aggressive GIL releases...")

start_time = time.time()

try:
    with duckdb.connect() as conn:
        # Register the TVF with schema definition
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        tvf = conn.create_table_function(
            name="generate_names",
            callable=generate_names,
            parameters=None,  
            schema=schema,
            return_type="records",
        )

        print("✅ TVF registered successfully")
        
        # Test the problematic large query
        print("⚡ Executing large query (100k rows)...")
        result = conn.sql("SELECT * FROM generate_names(100000)").df()
        
        elapsed = time.time() - start_time
        print(f"🎉 Success! Retrieved {len(result)} rows in {elapsed:.2f} seconds")
        print(f"📊 First few rows:\n{result.head()}")
        
        assert len(result) == 100000
        print("✅ All assertions passed!")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"❌ Test failed after {elapsed:.2f} seconds: {e}")
    import traceback
    traceback.print_exc()

total_elapsed = time.time() - start_time
print(f"⏱️  Total execution time: {total_elapsed:.2f} seconds")
#!/usr/bin/env python3
"""
Test just 80k with debug output
"""

import duckdb
import time

def generate_names(count: int = 500):
    """Generate a list of sample names with progress tracking."""
    print(f"[DEBUG Python] Function called with count={count}")
    
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    result = []
    
    for i in range(count):
        result.append((names[i % len(names)], i))
        
        # Print progress every 10k items
        if i > 0 and i % 10000 == 0:
            print(f"[DEBUG Python] Generated {i} items so far...")
    
    print(f"[DEBUG Python] Function completed, returning {len(result)} items")
    return result

print("🔍 Testing 80k with debug output...")

with duckdb.connect() as conn:
    schema = [["name", "VARCHAR"], ["id", "INT"]]
    
    print("[DEBUG] Registering TVF...")
    tvf = conn.create_table_function(
        name="test_80k",
        callable=generate_names,
        parameters=None,
        schema=schema,
        return_type="records",
    )
    print("[DEBUG] TVF registered successfully")
    
    print("[DEBUG] Starting query execution...")
    start_time = time.time()
    result = conn.sql("SELECT COUNT(*) FROM test_80k(80000)").fetchone()
    elapsed = time.time() - start_time
    
    print(f"✅ Result: {result[0]} in {elapsed:.3f}s")
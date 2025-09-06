#!/usr/bin/env python3
"""
Test that exactly replicates the notebook scenario where hanging occurs.
This test mimics the exact code pattern from notebooks/tvf_examples.ipynb
"""

import duckdb
import time

# Exact same function as in your notebook
def generate_names(count: int = 100000):
    """Generate a list of sample names."""
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    return [(names[i % len(names)], i) for i in range(count)]

print("🔬 Testing EXACT notebook scenario...")
print("Replicating the exact pattern from notebooks/tvf_examples.ipynb")

try:
    with duckdb.connect() as conn:
        # Exact same schema definition from notebook
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        print("Registering TVF...")
        start_register = time.time()
        
        tvf = conn.create_table_function(
            name="generate_names",
            callable=generate_names,
            parameters=None,  # Will infer from function signature
            schema=schema,
            return_type="records",  # Return list of tuples
        )
        
        register_time = time.time() - start_register
        print(f"TVF registered in {register_time:.3f}s")
        
        # Use the exact same query pattern as notebook  
        print("Running query...")
        start_query = time.time()
        
        n = 100000  # Same size that hangs in notebook
        result = conn.sql(f"SELECT * FROM generate_names({n})").df()
        
        query_time = time.time() - start_query
        
        print(f"✅ Query completed in {query_time:.3f}s")
        print(f"Result shape: {result.shape}")
        
        assert len(result) == n, f"Wrong count: {len(result)} != {n}"
        print(f"✅ Assertion passed: {len(result)} rows returned")
        
except Exception as e:
    elapsed = time.time() - start_register
    print(f"❌ Error after {elapsed:.3f}s: {e}")
    import traceback
    traceback.print_exc()

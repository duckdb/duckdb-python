#!/usr/bin/env python3
"""
Test script to run in ipython to reproduce the hanging
"""

# This will be executed in ipython kernel
test_code = """
import duckdb

def generate_names(count: int = 500):
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    return [(names[i % len(names)], i) for i in range(count)]

print("Testing 80k rows in ipython...")

with duckdb.connect() as conn:
    schema = [["name", "VARCHAR"], ["id", "INT"]]
    
    tvf = conn.create_table_function(
        name="generate_names",
        callable=generate_names,
        parameters=None,
        schema=schema,
        return_type="records",
    )
    
    print("TVF registered, executing query...")
    result = conn.sql("SELECT * FROM generate_names(80000)").df()
    print(f"Success! Got {len(result)} rows")
"""

# Write the test to a temp file and execute it with ipython
with open('/tmp/ipython_test.py', 'w') as f:
    f.write(test_code)

print("Test code written to /tmp/ipython_test.py")
print("Now running in ipython...")
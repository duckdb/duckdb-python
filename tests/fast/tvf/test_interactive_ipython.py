#!/usr/bin/env python3
"""
Try to simulate VS Code notebook environment more closely
"""
import subprocess
import sys
import time

# Create the exact code that would run in a VS Code notebook cell
notebook_cell_code = '''
import duckdb
import time

def generate_names(count: int = 500):
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    return [(names[i % len(names)], i) for i in range(count)]

print("Starting 80k test in notebook-like environment...")
start_time = time.time()

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
    
elapsed = time.time() - start_time
print(f"Result: {len(result)} rows in {elapsed:.2f}s")
'''

# Write to temp file
with open('/tmp/notebook_test.py', 'w') as f:
    f.write(notebook_cell_code)

print("Testing with ipython -i (interactive mode)...")

# Run with timeout to catch hangs
try:
    result = subprocess.run([
        'bash', '-c', 
        'cd /home/ec2-user/git/duckdb-python && source .venv/bin/activate && timeout 30s ipython -i /tmp/notebook_test.py'
    ], capture_output=True, text=True, timeout=35)
    
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")  
        print(result.stderr)
    print(f"Return code: {result.returncode}")
        
except subprocess.TimeoutExpired:
    print("❌ Process timed out - likely hanging!")
except Exception as e:
    print(f"❌ Error: {e}")
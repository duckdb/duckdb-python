#!/usr/bin/env python3
"""Debug the scan issue with minimal output"""

import duckdb
import sys
import os

# Set up IPython to trigger progress bar
try:
    import IPython
    import ipywidgets
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    shell = TerminalInteractiveShell.instance()
    IPython.get_ipython = lambda: shell
    print("IPython environment set up")
except:
    print("No IPython environment")

def simple_generator(count: int = 10000):  # Smaller count for debugging
    """Generate data."""
    names = ["Alice", "Bob", "Charlie"]
    for i in range(count):
        yield (names[i % len(names)], i)

print("Creating connection...")
with duckdb.connect() as conn:
    schema = [["name", "VARCHAR"], ["id", "INT"]]
    
    print("Registering TVF...")
    tvf = conn.create_table_function(
        name="debug_gen", 
        callable=simple_generator,
        schema=schema,
        return_type="records"
    )
    
    print("Starting query...")
    # Use smaller count to see debug pattern faster
    result = conn.sql("SELECT COUNT(*) FROM debug_gen(10000)").fetchone()
    print(f"Result: {result}")
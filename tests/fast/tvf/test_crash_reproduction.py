#!/usr/bin/env python3
"""
Reproduce the exact crash scenario from the notebook.
This creates an IPython environment to trigger the progress bar, then runs the 1M rows generator.
"""

import sys
import os
import time

# Mock IPython environment to trigger progress bar
def setup_ipython_environment():
    """Set up IPython environment to trigger DuckDB's Jupyter progress bar."""
    try:
        # Try to import and initialize IPython environment
        import IPython
        import ipywidgets
        
        # Create a minimal IPython shell to make DuckDB think we're in Jupyter
        from IPython.terminal.interactiveshell import TerminalInteractiveShell
        
        # Initialize IPython shell
        shell = TerminalInteractiveShell.instance()
        IPython.get_ipython = lambda: shell
        
        print("✅ IPython environment initialized - DuckDB should enable progress bar")
        return True
        
    except ImportError as e:
        print(f"❌ Cannot create IPython environment: {e}")
        print("Run: pip install ipython ipywidgets")
        return False
    except Exception as e:
        print(f"⚠️  IPython setup issue: {e}")
        # Continue anyway, DuckDB might still detect the modules
        return True

import duckdb

def generate_names(count: int = 500):
    """Generate a list of sample names."""
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]

    for i in range(count):
        yield (names[i % len(names)], i)
    #return [(names[i % len(names)], i) for i in range(count)]

print("🔥 Attempting to reproduce the crash with 1M rows generator...")
print("This should hang/crash after ~35 seconds based on user report")
print("=" * 70)

# Set up IPython environment to trigger progress bar
if not setup_ipython_environment():
    print("⚠️  Warning: Running without IPython environment - may not reproduce crash")
    print("The crash requires progress bar to be enabled")

start_time = time.time()

try:
    with duckdb.connect() as conn:
        print(f"✅ Created connection at {time.time() - start_time:.1f}s")
        
        # Check if progress bar is enabled (this is what causes the crash)
        try:
            progress_enabled = conn.execute('PRAGMA enable_progress_bar').fetchone()[0]
            print(f"📊 Progress bar enabled: {progress_enabled}")
            if not progress_enabled:
                print("⚠️  Progress bar NOT enabled - crash may not reproduce!")
                print("   This usually means IPython/Jupyter environment not detected")
        except:
            print("⚠️  Could not check progress bar status")
        
        # Register the TVF with schema definition
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        print(f"🔗 Registering TVF at {time.time() - start_time:.1f}s")
        tvf = conn.create_table_function(
            name="generate_names",
            callable=generate_names,
            parameters=None,  # Will infer from function signature
            schema=schema,
            return_type="records",  # Return list of tuples
        )
        
        print(f"⚡ Starting 1M row query at {time.time() - start_time:.1f}s")

        # Use the TVF in a query
        n = 1000000
        result = conn.sql(f"set threads = 1;SELECT * FROM generate_names({n})").df()
        
        elapsed = time.time() - start_time
        print(f"✅ Query completed successfully at {elapsed:.1f}s")
        print(f"📊 Result length: {len(result)}")
        
        assert len(result) == n, f"Expected {n} rows, got {len(result)}"
        print("✅ Assertion passed - correct number of rows")
        
except Exception as e:
    elapsed = time.time() - start_time
    print(f"💥 CRASH/ERROR at {elapsed:.1f}s: {e}")
    import traceback
    traceback.print_exc()
    
print(f"\n🏁 Test completed after {time.time() - start_time:.1f}s")
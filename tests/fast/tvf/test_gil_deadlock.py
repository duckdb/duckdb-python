#!/usr/bin/env python3
"""
Test to reproduce GIL deadlock between TVF processing and Jupyter progress bar.

This test simulates IPython environment conditions to trigger the progress bar
and potentially reproduce the GIL contention issue.
"""

import os
import sys
import threading
import time

# Mock IPython environment to trigger Jupyter progress bar
def mock_ipython_environment():
    """Set up environment to make DuckDB think we're in IPython/Jupyter"""
    # This is what DuckDB checks for to enable Jupyter mode
    try:
        # Try to import the modules that DuckDB checks for
        import IPython
        import ipywidgets
        print("✅ IPython environment detected - this should trigger progress bar")
        return True
    except ImportError as e:
        print(f"❌ Missing IPython dependencies: {e}")
        print("Installing ipywidgets might be needed: pip install ipywidgets")
        return False

def simulate_long_running_tvf():
    """Create a large TVF that will trigger progress bar updates"""
    import duckdb

    def generate_large_list(count: int = 100000):
        """Generate a large list to trigger batch processing and progress bar"""
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
        result = []
        for i in range(count):
            # Add some computation to make it slower and more likely to trigger progress bar
            result.append((names[i % len(names)], i))
            # Occasionally yield to allow other threads to run
            if i % 10000 == 0:
                print(f"📊 Generated {i}/{count} rows so far...")
        return result

    print("🔧 Creating DuckDB connection (this should enable progress bar if in IPython)")
    
    try:
        with duckdb.connect() as conn:
            # Check if progress bar is enabled
            print(f"📈 Progress bar enabled: {conn.execute('PRAGMA enable_progress_bar').fetchone()[0]}")
            
            schema = [["name", "VARCHAR"], ["id", "INT"]]
            
            print("🔗 Registering TVF...")
            tvf = conn.create_table_function(
                name="large_tvf",
                callable=generate_large_list,
                parameters=None,
                schema=schema,
                return_type="records",
            )
            
            print("⚡ Running large TVF query (this should trigger progress bar updates)...")
            print("🕐 If GIL deadlock occurs, this will hang...")
            
            start_time = time.time()
            
            # Use COUNT(*) to minimize data transfer but still process all data
            result = conn.sql("SELECT COUNT(*) FROM large_tvf(100000)").fetchone()
            
            elapsed = time.time() - start_time
            
            print(f"✅ Success! Query completed in {elapsed:.3f}s")
            print(f"📊 Result: {result[0]} rows processed")
            
            if result[0] == 100000:
                print("✅ Correct row count returned")
            else:
                print(f"❌ Unexpected row count: {result[0]} != 100000")
                
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ Error after {elapsed:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def main():
    print("🧪 Testing GIL deadlock between TVF processing and Jupyter progress bar")
    print("=" * 70)
    
    # Check if we can simulate IPython environment
    if not mock_ipython_environment():
        print("⚠️  Cannot test Jupyter progress bar without IPython dependencies")
        print("   Run: pip install ipywidgets jupyter")
        return
    
    print("🚀 Starting TVF test that should trigger progress bar...")
    success = simulate_long_running_tvf()
    
    if success:
        print("🎉 Test completed successfully - no GIL deadlock detected!")
    else:
        print("💥 Test failed - possible GIL deadlock or other issue")

if __name__ == "__main__":
    main()

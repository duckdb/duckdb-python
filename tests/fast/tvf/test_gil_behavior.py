#!/usr/bin/env python3
"""
Test GIL behavior similar to Jupyter notebook environment
"""

import duckdb
import time
import threading
import sys

def generate_names(count: int = 500):
    """Generate a list of sample names."""
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    print(f"Python function called with count={count}")
    sys.stdout.flush()
    
    # Simulate some work that might interact with GIL
    result = []
    for i in range(count):
        result.append((names[i % len(names)], i))
        # Add a small yield every 10k items to test GIL interactions
        if i > 0 and i % 10000 == 0:
            print(f"Generated {i} items so far...")
            sys.stdout.flush()
            # Small sleep to potentially trigger GIL issues
            time.sleep(0.001)
    
    print(f"Python function completed, returning {len(result)} items")
    sys.stdout.flush()
    return result

def test_with_gil_monitoring():
    """Test with GIL monitoring similar to notebook"""
    print("Starting GIL behavior test...")
    
    def gil_monitor():
        """Monitor thread to see if we're blocked on GIL"""
        for i in range(60):  # Monitor for up to 60 seconds
            time.sleep(1)
            print(f"Monitor: {i+1}s elapsed - still running...")
            sys.stdout.flush()
    
    # Start monitor thread
    monitor = threading.Thread(target=gil_monitor, daemon=True)
    monitor.start()
    
    try:
        with duckdb.connect() as conn:
            print("Creating connection...")
            
            # Register the TVF with schema definition
            schema = [["name", "VARCHAR"], ["id", "INT"]]
            print("Registering TVF...")
            
            tvf = conn.create_table_function(
                name="generate_names",
                callable=generate_names,
                parameters=None,
                schema=schema,
                return_type="records",
            )
            print("TVF registered successfully")
            
            # Test the problematic query
            print("Starting query execution...")
            start_time = time.time()
            
            result = conn.sql("SELECT * FROM generate_names(100000)").df()
            
            elapsed = time.time() - start_time
            print(f"Query completed in {elapsed:.2f} seconds")
            print(f"Result shape: {result.shape}")
            
            assert len(result) == 100000
            print("✓ Test passed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_smaller_batches():
    """Test with smaller datasets to see if there's a threshold issue"""
    print("\nTesting different dataset sizes...")
    
    for size in [1000, 10000, 25000, 50000, 75000, 100000]:
        print(f"\nTesting {size} rows...")
        try:
            with duckdb.connect() as conn:
                schema = [["name", "VARCHAR"], ["id", "INT"]]
                conn.create_table_function(
                    name=f"test_{size}",
                    callable=generate_names,
                    parameters=None,
                    schema=schema,
                    return_type="records",
                )
                
                start_time = time.time()
                result = conn.sql(f"SELECT COUNT(*) FROM test_{size}({size})").fetchone()
                elapsed = time.time() - start_time
                
                print(f"  {size} rows: {result[0]} in {elapsed:.3f}s")
                assert result[0] == size
                
        except Exception as e:
            print(f"  ❌ {size} rows failed: {e}")
            return False
    
    print("✓ All size tests passed!")
    return True

if __name__ == "__main__":
    print("Testing GIL behavior in notebook-like conditions...")
    
    success1 = test_with_gil_monitoring()
    success2 = test_smaller_batches()
    
    if success1 and success2:
        print("\n🎉 All GIL behavior tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
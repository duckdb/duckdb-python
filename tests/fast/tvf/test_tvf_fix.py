#!/usr/bin/env python3
"""
Test script to verify that the fixed CreateTableFunctionFromCallable works.
"""

import duckdb

def generate_names(count: int = 5):
    """Generate a list of sample names."""
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    return [(names[i % len(names)],) for i in range(count)]

def test_basic_tvf():
    print("Testing basic TVF functionality...")
    
    try:
        # Create connection
        conn = duckdb.connect()
        
        # Register the TVF
        schema = [("name", "VARCHAR")]
        
        tvf = conn.create_table_function(
            name="generate_names",
            callable=generate_names,
            parameters=None,
            schema=schema,
            return_type="strings"
        )
        
        print("✓ TVF registered successfully")
        
        # Test the bind phase by calling the function
        try:
            result = conn.execute("SELECT * FROM generate_names(3)").fetchall()
            print(f"✓ Query executed, got {len(result)} rows")
            print("✓ Result:", result)
            
        except Exception as e:
            print(f"✗ Query execution failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ TVF registration failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_basic_tvf()
    if success:
        print("\n🎉 Basic test passed - the bind function works!")
    else:
        print("\n💥 Test failed - need more fixes")
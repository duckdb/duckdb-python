#!/usr/bin/env python3
"""
Minimal test to reproduce the infinite loop issue with debug logging.
This test simplifies everything to focus on the root cause.
"""

import os
import sys
import duckdb

def simple_generator(count: int = 5):
    """Simple generator that yields a few rows"""
    names = ["Alice", "Bob", "Charlie"]
    for i in range(count):
        print(f"PYTHON: Generating row {i}: {names[i % len(names)]}, {i}")
        yield (names[i % len(names)], i)

def main():
    print("🧪 Testing minimal TVF with enhanced debug logging")
    print("=" * 60)
    
    try:
        with duckdb.connect() as conn:
            print("✅ Created DuckDB connection")
            
            schema = [["name", "VARCHAR"], ["id", "INT"]]
            
            print("🔗 Registering simple_generator TVF...")
            tvf = conn.create_table_function(
                name="simple_generator",
                callable=simple_generator,
                parameters=None,
                schema=schema,
                return_type="records",
            )
            print("✅ TVF registered successfully")
            
            print("⚡ Running simple query...")
            result = conn.execute("SELECT * FROM simple_generator(3)").fetchall()
            
            print("✅ Query completed successfully!")
            print(f"📊 Result: {result}")
            
            if len(result) == 3:
                print("✅ Correct number of rows returned")
            else:
                print(f"❌ Expected 3 rows, got {len(result)}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("💥 Test failed!")
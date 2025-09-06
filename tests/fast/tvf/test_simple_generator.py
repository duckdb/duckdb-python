#!/usr/bin/env python3
"""
Simple generator test to avoid segfault
"""

import duckdb

def simple_generator(count: int = 10):
    """Generate simple tuples."""
    for i in range(count):
        yield ("name", i)

print("🔬 Testing simple generator...")

try:
    with duckdb.connect() as conn:
        schema = [["name", "VARCHAR"], ["id", "INT"]]

        tvf = conn.create_table_function(
            name="simple_generator",
            callable=simple_generator,
            parameters=None,
            schema=schema,
            return_type="records",
        )

        result = conn.sql("SELECT * FROM simple_generator(5)").fetchall()
        print(f"✅ Result: {result}")
        
        assert len(result) == 5
        print("✅ Simple generator test passed!")

except Exception as e:
    print(f"❌ Simple generator test failed: {e}")
    import traceback
    traceback.print_exc()
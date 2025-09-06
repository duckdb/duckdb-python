#!/usr/bin/env python3
"""
Test multiple sizes to verify the fix works across the range
"""

import duckdb


def generate_names(count: int = 500):
    """Generate a list of sample names."""
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    return [(names[i % len(names)], i) for i in range(count)]


def test_size(size=1000):
    """Test a specific size like the notebook does."""
    try:
        with duckdb.connect() as conn:
            schema = [["name", "VARCHAR"], ["id", "INT"]]

            tvf = conn.create_table_function(
                name="generate_names",
                callable=generate_names,
                parameters=None,
                schema=schema,
                return_type="records",
            )

            # Use exact notebook pattern - SELECT * and .df()
            result = conn.sql(f"SELECT * FROM generate_names({size})").df()

            if len(result) == size:
                print(f"✅ {size:,} rows: SUCCESS")
                return True
            else:
                print(f"❌ {size:,} rows: WRONG COUNT ({len(result)} != {size})")
                return False

    except Exception as e:
        print(f"❌ {size:,} rows: EXCEPTION ({e})")
        return False


if __name__ == "__main__":
    print("🔍 Testing range of sizes with exact notebook pattern...")

    # Test the problematic range plus some extra
    test_sizes = [60000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]

    all_passed = True
    for size in test_sizes:
        success = test_size(size)
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests PASSED! The fix is working!")
    else:
        print("\n❌ Some tests failed.")

#!/usr/bin/env python3
"""
Test generator-based TVF with large datasets
"""

import duckdb
import time


def generate_names_generator(count: int = 500):
    """Generate a list of sample names using generator."""
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]

    for i in range(count):
        yield (names[i % len(names)], i)


def generate_names_list(count: int = 500):
    """Generate a list of sample names using list."""
    names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    return [(names[i % len(names)], i) for i in range(count)]


def test_generator_vs_list(size: int = 100):
    """Test generator vs list performance"""
    print(f"\n🔬 Testing {size:,} rows:")

    # Test generator approach
    start_time = time.time()
    try:
        with duckdb.connect() as conn:
            schema = [["name", "VARCHAR"], ["id", "INT"]]

            tvf = conn.create_table_function(
                name="test_generator",
                callable=generate_names_generator,
                parameters=None,
                schema=schema,
                return_type="records",
            )

            result = conn.sql(f"SELECT COUNT(*) FROM test_generator({size})").fetchone()

            elapsed_gen = time.time() - start_time
            print(f"  🔁 Generator: {result[0]:,} rows in {elapsed_gen:.3f}s")
            assert result[0] == size

    except Exception as e:
        print(f"  ❌ Generator failed: {e}")
        return False, None

    # Test list approach
    start_time = time.time()
    try:
        with duckdb.connect() as conn:
            schema = [["name", "VARCHAR"], ["id", "INT"]]

            tvf = conn.create_table_function(
                name="test_list",
                callable=generate_names_list,
                parameters=None,
                schema=schema,
                return_type="records",
            )

            result = conn.sql(f"SELECT COUNT(*) FROM test_list({size})").fetchone()

            elapsed_list = time.time() - start_time
            print(f"  📋 List: {result[0]:,} rows in {elapsed_list:.3f}s")
            assert result[0] == size

    except Exception as e:
        print(f"  ❌ List failed: {e}")
        return None, False

    return elapsed_gen, elapsed_list


if __name__ == "__main__":
    print("🚀 Testing generator-based TVF implementation...")

    # Test different sizes
    test_sizes = [1000, 10000, 50000, 80000, 100000]

    for size in test_sizes:
        gen_time, list_time = test_generator_vs_list(size)

        if gen_time and list_time:
            speedup = list_time / gen_time
            print(
                f"  📊 Generator is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
            )

    print("\n🎉 Generator tests completed!")

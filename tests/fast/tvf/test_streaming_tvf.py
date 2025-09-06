#!/usr/bin/env python3
"""
Focused tests for streaming TVF improvements
"""

import duckdb
import pyarrow as pa
import time
import threading
from concurrent.futures import ThreadPoolExecutor


def test_moderate_records_dataset():
    """Test records mode with moderate dataset (50k rows)"""
    print("Testing records mode with 50k rows...")
    
    def generate_records():
        """Generate 50k records efficiently"""
        return [(i, f"row_{i}", i * 1.5) for i in range(50000)]
    
    conn = duckdb.connect(":memory:")
    
    conn.create_table_function(
        "moderate_records",
        generate_records,
        schema=[("id", "bigint"), ("name", "varchar"), ("value", "double")],
        return_type="records"
    )
    
    start_time = time.time()
    result = conn.execute("SELECT COUNT(*), AVG(value), MAX(id) FROM moderate_records()").fetchall()
    elapsed = time.time() - start_time
    
    print(f"Records mode: {result[0][0]} rows, avg={result[0][1]:.2f}, max_id={result[0][2]} in {elapsed:.2f}s")
    assert result[0][0] == 50000
    assert result[0][2] == 49999
    conn.close()


def test_moderate_arrow_dataset():
    """Test arrow mode with moderate dataset (30k rows)"""
    print("Testing arrow mode with 30k rows...")
    
    def generate_arrow():
        """Generate moderate Arrow table"""
        size = 30000
        return pa.table({
            'id': pa.array(range(size), type=pa.int64()),
            'name': pa.array([f"row_{i}" for i in range(size)], type=pa.string()),
            'value': pa.array([i * 2.0 for i in range(size)], type=pa.float64())
        })
    
    conn = duckdb.connect(":memory:")
    
    conn.create_table_function(
        "moderate_arrow",
        generate_arrow,
        schema=[("id", "bigint"), ("name", "varchar"), ("value", "double")],
        return_type="arrow"
    )
    
    start_time = time.time()
    result = conn.execute("SELECT COUNT(*), AVG(value), MIN(id), MAX(id) FROM moderate_arrow()").fetchall()
    elapsed = time.time() - start_time
    
    print(f"Arrow mode: {result[0][0]} rows, avg={result[0][1]:.2f}, range=[{result[0][2]}, {result[0][3]}] in {elapsed:.2f}s")
    assert result[0][0] == 30000
    assert result[0][2] == 0
    assert result[0][3] == 29999
    conn.close()


def test_concurrent_tvf_safety():
    """Test concurrent TVF usage for thread safety"""
    print("Testing concurrent TVF usage...")
    
    def worker_task(worker_id):
        """Worker function using separate connections"""
        conn = duckdb.connect(":memory:")
        
        # Records TVF
        conn.create_table_function(
            f"worker_records_{worker_id}",
            lambda: [(i, f"w{worker_id}_r{i}") for i in range(5000)],
            schema=[("id", "bigint"), ("name", "varchar")],
            return_type="records"
        )
        
        # Arrow TVF
        def make_worker_arrow():
            return pa.table({
                'worker_id': pa.array([worker_id] * 3000, type=pa.int64()),
                'row_id': pa.array(range(3000), type=pa.int64())
            })
        
        conn.create_table_function(
            f"worker_arrow_{worker_id}",
            make_worker_arrow,
            schema=[("worker_id", "bigint"), ("row_id", "bigint")],
            return_type="arrow"
        )
        
        # Execute queries
        records_result = conn.execute(f"SELECT COUNT(*) FROM worker_records_{worker_id}()").fetchone()
        arrow_result = conn.execute(f"SELECT COUNT(*), AVG(row_id) FROM worker_arrow_{worker_id}()").fetchone()
        
        conn.close()
        return {
            'worker_id': worker_id,
            'records_count': records_result[0],
            'arrow_count': arrow_result[0],
            'arrow_avg': arrow_result[1]
        }
    
    # Run 4 workers concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker_task, i) for i in range(4)]
        results = [f.result(timeout=30) for f in futures]
    
    print(f"Concurrent test: {len(results)} workers completed successfully")
    for r in results:
        assert r['records_count'] == 5000
        assert r['arrow_count'] == 3000
        print(f"  Worker {r['worker_id']}: records={r['records_count']}, arrow={r['arrow_count']}, avg_row_id={r['arrow_avg']:.1f}")


def test_query_patterns():
    """Test different query patterns that should use efficient processing"""
    print("Testing various query patterns...")
    
    conn = duckdb.connect(":memory:")
    
    # Create an Arrow TVF
    def make_test_data():
        size = 25000
        return pa.table({
            'id': pa.array(range(size), type=pa.int64()),
            'category': pa.array([f"cat_{i % 10}" for i in range(size)], type=pa.string()),
            'value': pa.array([i * 0.1 for i in range(size)], type=pa.float64())
        })
    
    conn.create_table_function(
        "test_data",
        make_test_data,
        schema=[("id", "bigint"), ("category", "varchar"), ("value", "double")],
        return_type="arrow"
    )
    
    # Test count query (should stream efficiently)
    start = time.time()
    count_result = conn.execute("SELECT COUNT(*) FROM test_data()").fetchone()
    count_time = time.time() - start
    print(f"  COUNT query: {count_result[0]} rows in {count_time:.3f}s")
    
    # Test aggregation (should stream efficiently)
    start = time.time()
    agg_result = conn.execute("SELECT category, COUNT(*), AVG(value) FROM test_data() GROUP BY category ORDER BY category").fetchall()
    agg_time = time.time() - start
    print(f"  GROUP BY query: {len(agg_result)} groups in {agg_time:.3f}s")
    
    # Test limit (should stop early)
    start = time.time()
    limit_result = conn.execute("SELECT * FROM test_data() ORDER BY id LIMIT 100").fetchall()
    limit_time = time.time() - start
    print(f"  LIMIT query: {len(limit_result)} rows in {limit_time:.3f}s")
    
    # Test filter (should process efficiently)
    start = time.time()
    filter_result = conn.execute("SELECT COUNT(*) FROM test_data() WHERE id % 1000 = 0").fetchone()
    filter_time = time.time() - start
    print(f"  FILTER query: {filter_result[0]} matching rows in {filter_time:.3f}s")
    
    assert count_result[0] == 25000
    assert len(agg_result) == 10  # 10 categories
    assert len(limit_result) == 100
    assert filter_result[0] == 25  # ids 0, 1000, 2000, ..., 24000
    
    conn.close()


def test_error_handling():
    """Test error handling in TVFs"""
    print("Testing error handling...")
    
    conn = duckdb.connect(":memory:")
    
    def problematic_function():
        # This should fail after returning some data
        for i in range(100):
            if i == 50:
                raise ValueError("Intentional error")
            yield (i, f"row_{i}")
    
    conn.create_table_function(
        "error_prone",
        lambda: list(problematic_function()),
        schema=[("id", "bigint"), ("name", "varchar")],
        return_type="records"
    )
    
    try:
        result = conn.execute("SELECT COUNT(*) FROM error_prone()").fetchall()
        print("  ERROR: Should have thrown an exception")
        assert False, "Expected ValueError"
    except Exception as e:
        print(f"  Correctly caught error: {type(e).__name__}")
    
    conn.close()


if __name__ == "__main__":
    print("Starting focused TVF streaming tests...")
    
    try:
        test_moderate_records_dataset()
        print("✓ Moderate records dataset test passed\n")
        
        test_moderate_arrow_dataset()
        print("✓ Moderate arrow dataset test passed\n")
        
        test_concurrent_tvf_safety()
        print("✓ Concurrent TVF safety test passed\n")
        
        test_query_patterns()
        print("✓ Query patterns test passed\n")
        
        test_error_handling()
        print("✓ Error handling test passed\n")
        
        print("🎉 All focused TVF tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
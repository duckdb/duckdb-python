import duckdb
import threading
import time
import pytest


def test_same_callable_different_names():
    """Test registering the same Python callable under different TVF names"""
    with duckdb.connect() as conn:
        
        def shared_function():
            return [("value", i) for i in range(100)]
        
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        # Register the same callable under two different names
        conn.create_table_function(
            name="tvf_name_1",
            callable=shared_function,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        conn.create_table_function(
            name="tvf_name_2", 
            callable=shared_function,  # Same callable!
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        # Test each TVF separately first
        result1 = conn.execute("SELECT COUNT(*) FROM tvf_name_1()").fetchone()
        assert result1[0] == 100
        
        result2 = conn.execute("SELECT COUNT(*) FROM tvf_name_2()").fetchone()
        assert result2[0] == 100
        
        # Now test them together - cartesian product: 100 * 100 = 10,000
        result3 = conn.execute("SELECT COUNT(*) FROM tvf_name_1(), tvf_name_2()").fetchone()
        assert result3[0] == 10000
        
        # Test calling them in sequence
        result4 = conn.execute("SELECT COUNT(*) FROM tvf_name_1()").fetchone()
        result5 = conn.execute("SELECT COUNT(*) FROM tvf_name_2()").fetchone()
        assert result4[0] == 100
        assert result5[0] == 100


def test_stateful_callable_shared():
    """Test if different TVFs maintain separate state but share callable state"""
    with duckdb.connect() as conn:
        
        counter = 0
        def stateful_function():
            nonlocal counter
            counter += 1
            # Return different data based on counter
            return [(f"call_{counter}", i) for i in range(5)]
        
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        # Register same stateful callable under different names
        conn.create_table_function(
            name="stateful_1",
            callable=stateful_function,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        conn.create_table_function(
            name="stateful_2",
            callable=stateful_function,  # Same stateful callable
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        # Each call should increment the counter
        result1 = conn.execute("SELECT name FROM stateful_1() LIMIT 1").fetchone()
        assert result1[0] == "call_1"
        
        result2 = conn.execute("SELECT name FROM stateful_2() LIMIT 1").fetchone()
        assert result2[0] == "call_2"
        
        # Call first one again - should be third call
        result3 = conn.execute("SELECT name FROM stateful_1() LIMIT 1").fetchone()
        assert result3[0] == "call_3"


def test_concurrent_tvf_execution():
    """Test concurrent execution of multiple TVFs to check for race conditions"""
    with duckdb.connect() as conn:
        
        # Create different functions to avoid shared state
        def slow_function_1():
            # Simulate some processing time
            time.sleep(0.1)
            return [(f"slow1_{i}", i) for i in range(50)]
            
        def slow_function_2():
            time.sleep(0.1)
            return [(f"slow2_{i}", i) for i in range(30)]
        
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        conn.create_table_function(
            name="slow_tvf_1",
            callable=slow_function_1,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        conn.create_table_function(
            name="slow_tvf_2",
            callable=slow_function_2,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        results = []
        errors = []
        
        def run_query(query, result_list):
            try:
                # Each thread should create its own connection to avoid contention
                with duckdb.connect() as thread_conn:
                    # Re-register TVFs in the new connection
                    thread_conn.create_table_function(
                        name="slow_tvf_1",
                        callable=slow_function_1,
                        parameters=None,
                        schema=schema,
                        return_type="records",
                    )
                    thread_conn.create_table_function(
                        name="slow_tvf_2",
                        callable=slow_function_2,
                        parameters=None,
                        schema=schema,
                        return_type="records",
                    )
                    
                    result = thread_conn.execute(query).fetchone()
                    result_list.append(result[0])
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent queries
        threads = []
        for i in range(3):
            t1 = threading.Thread(target=run_query, args=("SELECT COUNT(*) FROM slow_tvf_1()", results))
            t2 = threading.Thread(target=run_query, args=("SELECT COUNT(*) FROM slow_tvf_2()", results))
            threads.extend([t1, t2])
            t1.start()
            t2.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 6, f"Expected 6 results, got {len(results)}"
        
        # Count occurrences of each result
        count_50 = results.count(50)  # slow_tvf_1 results
        count_30 = results.count(30)  # slow_tvf_2 results
        
        assert count_50 == 3, f"Expected 3 results of 50, got {count_50}"
        assert count_30 == 3, f"Expected 3 results of 30, got {count_30}"


def test_large_dataset_performance():
    """Test performance with large datasets to ensure no contention issues"""
    with duckdb.connect() as conn:
        
        def large_generator():
            # Generate a reasonably large dataset (but not huge for test speed)
            return [(f"item_{i}", i) for i in range(5000)]
        
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        conn.create_table_function(
            name="large_tvf",
            callable=large_generator,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        # Test the TVF works correctly
        result = conn.execute("SELECT COUNT(*) FROM large_tvf()").fetchone()
        assert result[0] == 5000
        
        # Test aggregation works
        result = conn.execute("SELECT MAX(id) FROM large_tvf()").fetchone()
        assert result[0] == 4999
        
        # Test filtering works
        result = conn.execute("SELECT COUNT(*) FROM large_tvf() WHERE id < 100").fetchone()
        assert result[0] == 100


def test_multiple_connections_same_tvf():
    """Test that TVFs work correctly across multiple connections"""
    counter = 0
    def shared_counter_function():
        nonlocal counter
        counter += 1
        return [(f"connection_call_{counter}", counter) for i in range(10)]
    
    schema = [("name", "VARCHAR"), ("id", "INT")]
    
    # Create multiple connections and register the same function
    results = []
    
    for i in range(3):
        with duckdb.connect() as conn:
            conn.create_table_function(
                name="shared_counter_tvf",
                callable=shared_counter_function,
                parameters=None,
                schema=schema,
                return_type="records",
            )
            
            result = conn.execute("SELECT name FROM shared_counter_tvf() LIMIT 1").fetchone()
            results.append(result[0])
    
    # Each connection should have incremented the shared counter
    assert results == ["connection_call_1", "connection_call_2", "connection_call_3"]


def test_error_handling_in_tvf():
    """Test error handling when TVF functions raise exceptions"""
    with duckdb.connect() as conn:
        
        def error_function():
            raise ValueError("Intentional test error")
        
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        conn.create_table_function(
            name="error_tvf",
            callable=error_function,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        # This should raise an exception
        with pytest.raises(Exception):
            conn.execute("SELECT * FROM error_tvf()").fetchall()


def test_tvf_with_parameters():
    """Test TVFs that accept parameters for potential contention issues"""
    with duckdb.connect() as conn:
        
        def parametrized_function(count=10, prefix="item"):
            return [(f"{prefix}_{i}", i) for i in range(count)]
        
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        conn.create_table_function(
            name="param_tvf",
            callable=parametrized_function,
            parameters=["count", "prefix"],
            schema=schema,
            return_type="records",
        )
        
        # Test with different parameters
        result1 = conn.execute("SELECT COUNT(*) FROM param_tvf(5, 'test')").fetchone()
        assert result1[0] == 5
        
        result2 = conn.execute("SELECT COUNT(*) FROM param_tvf(20, 'data')").fetchone()
        assert result2[0] == 20
        
        # Test parameter order
        result3 = conn.execute("SELECT name FROM param_tvf(3, 'xyz') ORDER BY id LIMIT 1").fetchone()
        assert result3[0] == "xyz_0"
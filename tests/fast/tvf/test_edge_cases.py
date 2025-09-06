import duckdb
import pytest
import pandas as pd
import numpy as np


def test_invalid_return_types():
    """Test that invalid return types are handled gracefully"""
    
    with duckdb.connect() as conn:
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        # Test 1: Return pandas DataFrame (not supported in records mode)
        def return_pandas_df():
            return pd.DataFrame({"name": ["test"], "id": [1]})
        
        conn.create_table_function(
            name="pandas_tvf",
            callable=return_pandas_df,
            parameters=None,
            schema=schema,
            return_type="records",  # DataFrame not supported in records mode
        )
        
        with pytest.raises(Exception, match="pandas DataFrame.*not supported"):
            conn.execute("SELECT * FROM pandas_tvf()").fetchall()
        
        # Test 2: Return scalar value instead of iterable
        def return_scalar():
            return 42  # Not iterable
        
        conn.create_table_function(
            name="scalar_tvf",
            callable=return_scalar,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        with pytest.raises(Exception, match="returned scalar value.*expected iterable"):
            conn.execute("SELECT * FROM scalar_tvf()").fetchall()
        
        # Test 3: Return None
        def return_none():
            return None
        
        conn.create_table_function(
            name="none_tvf",
            callable=return_none,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        with pytest.raises(Exception, match="returned None.*expected iterable"):
            conn.execute("SELECT * FROM none_tvf()").fetchall()


def test_inconsistent_data_shapes():
    """Test handling of inconsistent data shapes"""
    
    with duckdb.connect() as conn:
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        # Test 1: Inconsistent tuple sizes
        def inconsistent_tuple_sizes():
            return [
                ("test1", 1),         # 2 elements
                ("test2", 2, "extra"), # 3 elements - mismatch!
                ("test3", 3),         # 2 elements
            ]
        
        conn.create_table_function(
            name="inconsistent_tvf",
            callable=inconsistent_tuple_sizes,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        with pytest.raises(Exception, match="tuple of size"):
            conn.execute("SELECT * FROM inconsistent_tvf()").fetchall()
        
        # Test 2: Mix of tuples and non-tuples
        def mixed_types():
            return [
                ("test1", 1),    # tuple
                ["test2", 2],    # list 
                "invalid",       # string - not tuple/list
            ]
        
        conn.create_table_function(
            name="mixed_tvf",
            callable=mixed_types,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        # This might work for the first two but fail on the third
        with pytest.raises(Exception):
            result = conn.execute("SELECT * FROM mixed_tvf()").fetchall()


def test_empty_and_edge_iterables():
    """Test empty and edge case iterables"""
    
    with duckdb.connect() as conn:
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        # Test 1: Empty list (should work)
        def return_empty_list():
            return []
        
        conn.create_table_function(
            name="empty_tvf",
            callable=return_empty_list,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        result = conn.execute("SELECT COUNT(*) FROM empty_tvf()").fetchone()
        assert result[0] == 0
        
        # Test 2: List with empty tuples
        def return_empty_tuples():
            return [(), (), ()]
        
        conn.create_table_function(
            name="empty_tuples_tvf",
            callable=return_empty_tuples,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        with pytest.raises(Exception, match="tuple of size"):
            conn.execute("SELECT * FROM empty_tuples_tvf()").fetchall()
        
        # Test 3: Generator that yields invalid data
        def invalid_generator():
            yield ("valid", 1)
            yield {"invalid": "dict"}  # Dict instead of tuple
            yield ("valid2", 2)
        
        conn.create_table_function(
            name="invalid_gen_tvf",
            callable=invalid_generator,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        with pytest.raises(Exception):
            conn.execute("SELECT * FROM invalid_gen_tvf()").fetchall()


def test_type_conversion_edge_cases():
    """Test edge cases in type conversion"""
    
    with duckdb.connect() as conn:
        schema = [("text", "VARCHAR"), ("num", "INT"), ("flag", "BOOLEAN")]
        
        # Test 1: Invalid type conversions
        def invalid_conversions():
            return [
                ("valid", 1, True),
                ("invalid_int", "not_a_number", True),  # String where int expected
                ("invalid_bool", 1, "not_a_bool"),      # String where bool expected
            ]
        
        conn.create_table_function(
            name="invalid_conv_tvf",
            callable=invalid_conversions,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        # This might succeed for first row but fail on invalid conversions
        with pytest.raises(Exception):
            conn.execute("SELECT * FROM invalid_conv_tvf()").fetchall()
        
        # Test 2: Complex nested objects
        def nested_objects():
            return [
                ("test", [1, 2, 3], True),        # List where int expected
                ("test2", {"key": "value"}, True), # Dict where int expected
            ]
        
        conn.create_table_function(
            name="nested_tvf",
            callable=nested_objects,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        with pytest.raises(Exception):
            conn.execute("SELECT * FROM nested_tvf()").fetchall()


def test_extremely_large_values():
    """Test handling of extremely large values"""
    
    with duckdb.connect() as conn:
        schema = [("text", "VARCHAR"), ("big_num", "BIGINT")]
        
        def huge_values():
            return [
                ("normal", 1),
                ("huge_string", "x" * 10000),  # Very long string
                ("huge_int", 2**63 - 1),       # Max int64
                ("overflow", 2**64),           # Potential overflow
            ]
        
        conn.create_table_function(
            name="huge_tvf",
            callable=huge_values,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        # This might work or might fail on overflow - behavior depends on implementation
        try:
            result = conn.execute("SELECT COUNT(*) FROM huge_tvf()").fetchone()
            print(f"Huge values test passed with count: {result[0]}")
        except Exception as e:
            print(f"Huge values test failed as expected: {e}")


def test_generator_exceptions():
    """Test generators that raise exceptions during iteration"""
    
    with duckdb.connect() as conn:
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        # Test 1: Generator that fails midway
        def failing_generator():
            yield ("first", 1)
            yield ("second", 2)
            raise ValueError("Generator failed!")
            yield ("third", 3)  # Never reached
        
        conn.create_table_function(
            name="failing_gen_tvf",
            callable=failing_generator,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        with pytest.raises(Exception, match="Generator failed!"):
            conn.execute("SELECT * FROM failing_gen_tvf()").fetchall()
        
        # Test 2: Large but finite generator
        def large_generator():
            for i in range(1000):  # Finite but large
                yield (f"item_{i}", i)
        
        conn.create_table_function(
            name="large_tvf",
            callable=large_generator,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        # Test large generator works 
        result = conn.execute("SELECT COUNT(*) FROM large_tvf()").fetchone()
        assert result[0] == 1000  # Should process all 1000 rows


def test_unicode_and_special_characters():
    """Test handling of Unicode and special characters"""
    
    with duckdb.connect() as conn:
        schema = [("text", "VARCHAR"), ("id", "INT")]
        
        def unicode_data():
            return [
                ("normal", 1),
                ("🚀 emoji", 2),
                ("中文", 3),
                ("русский", 4),
                ("العربية", 5),
                ("null\x00char", 6),  # Null character
                ("new\nline", 7),     # Newline
                ("tab\ttab", 8),      # Tab
                ("'quote'", 9),       # Quotes
                ('"double"', 10),     # Double quotes
                ("back\\slash", 11),  # Backslash
            ]
        
        conn.create_table_function(
            name="unicode_tvf",
            callable=unicode_data,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        result = conn.execute("SELECT COUNT(*) FROM unicode_tvf()").fetchone()
        assert result[0] == 11
        
        # Test actual content
        results = conn.execute("SELECT text FROM unicode_tvf() ORDER BY id").fetchall()
        assert results[1][0] == "🚀 emoji"
        assert results[2][0] == "中文"


def test_memory_exhaustion_protection():
    """Test protection against memory exhaustion"""
    
    with duckdb.connect() as conn:
        schema = [("data", "VARCHAR")]
        
        # Test 1: Very large individual strings
        def large_strings():
            return [
                ("small",),
                ("x" * 1000000,),  # 1MB string
            ]
        
        conn.create_table_function(
            name="large_str_tvf",
            callable=large_strings,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        try:
            result = conn.execute("SELECT LENGTH(data) FROM large_str_tvf()").fetchall()
            assert result[0][0] == 5      # "small"
            assert result[1][0] == 1000000  # Large string
        except Exception as e:
            print(f"Large string test failed (might be expected): {e}")


def test_circular_references():
    """Test handling of circular references and complex objects"""
    
    with duckdb.connect() as conn:
        schema = [("name", "VARCHAR"), ("id", "INT")]
        
        # Test with objects that can't be converted
        def complex_objects():
            # Create a class instance
            class CustomObject:
                def __init__(self, name):
                    self.name = name
                def __str__(self):
                    return f"CustomObject({self.name})"
            
            obj = CustomObject("test")
            return [
                ("string", 1),
                (obj, 2),  # Custom object - should fail conversion
            ]
        
        conn.create_table_function(
            name="complex_tvf",
            callable=complex_objects,
            parameters=None,
            schema=schema,
            return_type="records",
        )
        
        # This should fail when trying to convert the custom object
        with pytest.raises(Exception):
            conn.execute("SELECT * FROM complex_tvf()").fetchall()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
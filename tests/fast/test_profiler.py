import pytest
import duckdb

class TestProfiler:
    def test_profiler_matches_expected_format(self):
        con = duckdb.connect()
        con.enable_profiling()
        con.execute("from range(100_000);")
        
        # Test JSON format
        profiling_info_json = con.get_profiling_information(format="json")
        assert isinstance(profiling_info_json, str)
        print(profiling_info_json)
        
        # Test Text format
        profiling_info_text = con.get_profiling_information(format="query_tree_optimizer")
        assert isinstance(profiling_info_text, str)
        print(profiling_info_text)  # Print to visually inspect the text format

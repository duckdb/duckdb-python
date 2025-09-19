import platform
import pytest


class TestRelationSlow(object):
    @pytest.mark.skipif(
        condition=platform.system() == "Emscripten",
        reason="Emscripten/Pyodide builds run out of memory at this scale, and error might not thrown reliably",
    )
    def test_materialized_relation_large(self, duckdb_cursor):
        """Test materialized relation with 10M rows - moved from fast tests due to 1+ minute runtime"""
        # Import the implementation function from the fast test
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fast'))
        from test_relation import TestRelation

        # Create instance and call the test with large parameter
        test_instance = TestRelation()
        test_instance.test_materialized_relation(duckdb_cursor, 10000000)
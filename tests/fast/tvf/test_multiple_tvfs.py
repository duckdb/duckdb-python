# Test if the connection is being stored properly in the registry
import duckdb


def test_multiple_tvfs():
    with duckdb.connect() as conn:

        def simple_test():
            return [("test", 1) for i in range(1000)]

        def simple_test2():
            return [("test2", 2) for i in range(100)]

        schema = [("name", "VARCHAR"), ("id", "INT")]

        try:
            # Try registering and immediately querying
            conn.create_table_function(
                name="simple_test",
                callable=simple_test,
                parameters=None,
                schema=schema,
                return_type="records",
            )
            conn.create_table_function(
                name="simple_test2",
                callable=simple_test2,
                parameters=None,
                schema=schema,
                return_type="records",
            )

            print("TVF registered successfully")

            # Try to use it immediately in the same context
            result = conn.execute(
                "SELECT COUNT(*) FROM simple_test()"
            ).fetchone()
            print(f"Result 1: {result}")
            
            result2 = conn.execute(
                "SELECT COUNT(*) FROM simple_test2()"
            ).fetchone()
            print(f"Result 2: {result2}")
            
            # Test both TVFs in same query - should return 1000 * 100 = 100,000 rows
            result3 = conn.execute(
                "SELECT COUNT(*) FROM simple_test(), simple_test2()"
            ).fetchone()
            print(f"Cartesian product result: {result3}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

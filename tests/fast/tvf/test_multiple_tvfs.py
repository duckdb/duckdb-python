# Test if the connection is being stored properly in the registry
import duckdb


def test_multiple_tvfs():
    with duckdb.connect() as conn:

        def simple_test():
            return [("test", 1) for i in range(10000)]

        def simple_test2():
            return [("test", 1) for i in range(1000)]

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
                callable=simple_test,
                parameters=None,
                schema=schema,
                return_type="records",
            )

            print("TVF registered successfully")

            # Try to use it immediately in the same context
            result = conn.execute(
                "SELECT * FROM simple_test(), simple_test2()"
            ).fetchall()
            print(f"Result: {result}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
    traceback.print_exc()

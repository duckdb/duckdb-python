"""Tests for the register_function decorator."""

import duckdb
from duckdb import func
from duckdb.sqltypes import BIGINT, DOUBLE


class TestRegisterFunctionDecorator:
    """Tests for @con.register_function decorator."""

    def test_decorator_without_arguments(self) -> None:
        """Test using the decorator without parentheses."""
        con = duckdb.connect()

        @con.register_function
        def add_one(x: int) -> int:
            return x + 1

        result = con.execute("SELECT add_one(5)").fetchone()
        assert result is not None
        assert result[0] == 6

        # Verify the function is still usable in Python
        assert add_one(10) == 11

    def test_decorator_with_custom_name(self) -> None:
        """Test using the decorator with a custom SQL function name."""
        con = duckdb.connect()

        @con.register_function(name="my_double")
        def double_value(x: int) -> int:
            return x * 2

        result = con.execute("SELECT my_double(7)").fetchone()
        assert result is not None
        assert result[0] == 14

    def test_decorator_with_explicit_types(self) -> None:
        """Test using the decorator with explicit parameter and return types."""
        con = duckdb.connect()

        @con.register_function(
            parameters=[BIGINT, BIGINT],
            return_type=BIGINT,
        )
        def add_two_numbers(a: int, b: int) -> int:
            return a + b

        result = con.execute("SELECT add_two_numbers(3, 4)").fetchone()
        assert result is not None
        assert result[0] == 7

    def test_decorator_with_all_options(self) -> None:
        """Test using the decorator with all options specified."""
        con = duckdb.connect()

        @con.register_function(
            name="custom_multiply",
            parameters=[DOUBLE, DOUBLE],
            return_type=DOUBLE,
            type=func.PythonUDFType.NATIVE,
            null_handling=func.FunctionNullHandling.DEFAULT,
            exception_handling=duckdb.PythonExceptionHandling.DEFAULT,
            side_effects=False,
        )
        def multiply(x: float, y: float) -> float:
            return x * y

        result = con.execute("SELECT custom_multiply(2.5, 4.0)").fetchone()
        assert result is not None
        assert result[0] == 10.0

    def test_decorator_returns_original_function(self) -> None:
        """Test that the decorator returns the original function unchanged."""
        con = duckdb.connect()

        def original_func(x: int) -> int:
            return x * 3

        decorated_func = con.register_function(original_func)

        # Should be the same function object
        assert decorated_func is original_func
        assert decorated_func(5) == 15

    def test_decorator_with_string_function(self) -> None:
        """Test decorator with a string manipulation function."""
        con = duckdb.connect()

        @con.register_function
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = con.execute("SELECT greet('World')").fetchone()
        assert result is not None
        assert result[0] == "Hello, World!"

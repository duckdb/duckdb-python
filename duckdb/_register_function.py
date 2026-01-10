from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, overload

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection, PythonExceptionHandling, func, sqltypes


@overload
def _register_function_impl(
    self: DuckDBPyConnection,
    function: Callable[..., Any],
    /,
) -> Callable[..., Any]: ...


@overload
def _register_function_impl(
    self: DuckDBPyConnection,
    function: None = None,
    /,
    *,
    name: str | None = None,
    parameters: list[sqltypes.DuckDBPyType] | None = None,
    return_type: sqltypes.DuckDBPyType | None = None,
    type: func.PythonUDFType | None = None,
    null_handling: func.FunctionNullHandling | None = None,
    exception_handling: PythonExceptionHandling | None = None,
    side_effects: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def _register_function_impl(
    self: DuckDBPyConnection,
    function: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    parameters: list[sqltypes.DuckDBPyType] | None = None,
    return_type: sqltypes.DuckDBPyType | None = None,
    type: func.PythonUDFType | None = None,
    null_handling: func.FunctionNullHandling | None = None,
    exception_handling: PythonExceptionHandling | None = None,
    side_effects: bool = False,
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a Python function as a DuckDB scalar UDF using decorator syntax.

    Can be used as a decorator with or without arguments:

        @con.register_function
        def my_func(x: int) -> int:
            return x + 1

        @con.register_function(name="custom_name", return_type=duckdb.INTEGER)
        def my_func(x):
            return x + 1

    Args:
        self: A DuckDBPyConnection with which to register the function against.
        function: The function to register (when used without parentheses).
        name: SQL function name. Defaults to the Python function's name.
        parameters: List of parameter types. Inferred from annotations if None.
        return_type: Return type. Inferred from annotations if None.
        type: UDF type (NATIVE or ARROW).
        null_handling: How to handle NULL values.
        exception_handling: How to handle Python exceptions.
        side_effects: Whether the function has side effects.

    Returns:
        The original function (unmodified), allowing it to be used normally in Python.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        func_name = name if name is not None else fn.__name__
        kwargs: dict[str, Any] = {}
        if type is not None:
            kwargs["type"] = type
        if null_handling is not None:
            kwargs["null_handling"] = null_handling
        if exception_handling is not None:
            kwargs["exception_handling"] = exception_handling
        if side_effects:
            kwargs["side_effects"] = side_effects
        self.create_function(func_name, fn, parameters, return_type, **kwargs)
        return fn

    if function is not None:
        # Used as @con.register_function (without parentheses)
        return decorator(function)
    # Used as @con.register_function(...) (with parentheses)
    return decorator

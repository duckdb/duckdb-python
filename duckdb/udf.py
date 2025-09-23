# ruff: noqa: D100
from typing import Callable


def vectorized(func: Callable) -> Callable:
    """Decorate a function with annotated function parameters.

    This allows DuckDB to infer that the function should be provided with pyarrow arrays and should expect
    pyarrow array(s) as output.
    """
    import types
    from inspect import signature

    new_func = types.FunctionType(func.__code__, func.__globals__, func.__name__, func.__defaults__, func.__closure__)
    # Construct the annotations:
    import pyarrow as pa

    new_annotations = {}
    sig = signature(func)
    for param in sig.parameters:
        new_annotations[param] = pa.lib.ChunkedArray

    new_func.__annotations__ = new_annotations
    return new_func

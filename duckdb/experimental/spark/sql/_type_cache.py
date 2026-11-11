"""The DuckDBPyType of each spark type text that needs a parse, parsed once: the singletons' names and unnamed structs.

The types module cannot keep them itself: its dict outlives finalization through the functions that hold it, and
nanobind would report every type in it as leaked. This module defines no function, so its dict is cleared with the
module at finalization.
"""

from duckdb.sqltypes import DuckDBPyType

TYPES: dict[str, DuckDBPyType] = {}

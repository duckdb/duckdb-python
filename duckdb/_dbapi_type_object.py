"""DuckDB DB API 2.0 Type Objects Module.

This module provides DB API 2.0 compliant type objects for DuckDB, allowing applications
to check column types returned by queries against standard database API categories.

Example:
    >>> import duckdb
    >>>
    >>> conn = duckdb.connect()
    >>> cursor = conn.cursor()
    >>> cursor.execute("SELECT 'hello' as text_col, 42 as num_col, CURRENT_DATE as date_col")
    >>>
    >>> # Check column types using DB API type objects
    >>> for i, desc in enumerate(cursor.description):
    >>>     col_name, col_type = desc[0], desc[1]
    >>>     if col_type == duckdb.STRING:
    >>>         print(f"{col_name} is a string type")
    >>>     elif col_type == duckdb.NUMBER:
    >>>         print(f"{col_name} is a numeric type")
    >>>     elif col_type == duckdb.DATETIME:
    >>>         print(f"{col_name} is a date/time type")

See Also:
    - PEP 249: https://peps.python.org/pep-0249/
    - DuckDB Type System: https://duckdb.org/docs/sql/data_types/overview
"""

import collections.abc

from duckdb.sqltypes import DuckDBPyType


class DBAPITypeObject:
    """DB API 2.0 type object for categorizing database column types.

    This class implements the type objects defined in PEP 249 (DB API 2.0).
    It allows checking whether a specific DuckDB type belongs to a broader
    category like STRING, NUMBER, DATETIME, etc.

    A category is a set of type ids, as DuckDBPyType.id gives them. A DuckDBPyType
    belongs to the category when its id is in the set, whatever its parameters
    (such as the width and scale of a DECIMAL) or its alias (such as JSON, a VARCHAR).

    Args:
        type_ids: Lowercase type ids of the types that belong to this category.

    Example:
        >>> string_types = DBAPITypeObject({"varchar"})
        >>> result = duckdb.sqltypes.VARCHAR == string_types  # True
        >>> result = duckdb.sqltypes.INTEGER == string_types  # False

    Note:
        This follows the DB API 2.0 specification where type objects are compared
        using equality operators rather than isinstance() checks.
    """

    def __init__(self, type_ids: collections.abc.Iterable[str]) -> None:
        """Initialize a DB API type object.

        Args:
            type_ids: Lowercase type ids, as DuckDBPyType.id gives them, of the types that belong to this category.
        """
        if isinstance(type_ids, str):
            msg = "type_ids must be a collection of type ids, not a single str"
            raise TypeError(msg)
        self.type_ids = frozenset(type_ids)

    def __eq__(self, other: object) -> bool:
        """Check if a DuckDB type belongs to this type category.

        This method implements the DB API 2.0 type checking mechanism.
        It returns True if the other object is a DuckDBPyType whose id
        is in this type category.

        Args:
            other: The object to compare, typically a DuckDBPyType instance.

        Returns:
            True if other is a DuckDBPyType in this category, False otherwise.

        Example:
            >>> NUMBER == duckdb.sqltypes.INTEGER  # True
            >>> NUMBER == duckdb.sqltypes.VARCHAR  # False
        """
        return isinstance(other, DuckDBPyType) and other.id in self.type_ids

    def __repr__(self) -> str:
        """Return a string representation of this type object.

        Returns:
            A string listing the type ids of this category, uppercase and sorted.

        Example:
            >>> repr(STRING)
            '<DBAPITypeObject [VARCHAR]>'
        """
        return f"<DBAPITypeObject [{','.join(sorted(type_id.upper() for type_id in self.type_ids))}]>"


# Define the standard DB API 2.0 type objects for DuckDB

STRING = DBAPITypeObject({"varchar"})
"""
STRING type object for text-based database columns.

This type object represents all string/text types in DuckDB. Currently includes:
- VARCHAR: Variable-length character strings, and the types that alias it, such as JSON

Use this to check if a column contains textual data that should be handled
as Python strings.

DB API 2.0 Reference:
    https://peps.python.org/pep-0249/#string

Example:
    >>> cursor.description[0][1] == STRING  # Check if first column is text
"""

NUMBER = DBAPITypeObject(
    {
        "tinyint",
        "utinyint",
        "smallint",
        "usmallint",
        "integer",
        "uinteger",
        "bigint",
        "ubigint",
        "hugeint",
        "uhugeint",
        "bignum",
        "decimal",
        "float",
        "double",
    }
)
"""
NUMBER type object for numeric database columns.

This type object represents all numeric types in DuckDB, including:

Integer Types:
- TINYINT, UTINYINT: 8-bit signed/unsigned integers
- SMALLINT, USMALLINT: 16-bit signed/unsigned integers
- INTEGER, UINTEGER: 32-bit signed/unsigned integers
- BIGINT, UBIGINT: 64-bit signed/unsigned integers
- HUGEINT, UHUGEINT: 128-bit signed/unsigned integers

Decimal Types:
- BIGNUM: Arbitrary precision integers
- DECIMAL: Fixed-point decimal numbers, of every width and scale

Floating Point Types:
- FLOAT: 32-bit floating point
- DOUBLE: 64-bit floating point

Use this to check if a column contains numeric data that should be handled
as Python int, float, or Decimal objects.

DB API 2.0 Reference:
    https://peps.python.org/pep-0249/#number

Example:
    >>> cursor.description[1][1] == NUMBER  # Check if second column is numeric
"""

DATETIME = DBAPITypeObject(
    {
        "date",
        "time",
        "time with time zone",
        "timestamp",
        "timestamp with time zone",
        "timestamp_ns",
        "timestamp_ms",
        "timestamp_s",
    }
)
"""
DATETIME type object for date and time database columns.

This type object represents all date/time types in DuckDB, including:

Date Types:
- DATE: Calendar dates (year, month, day)

Time Types:
- TIME: Time of day without timezone
- TIME_TZ: Time of day with timezone

Timestamp Types:
- TIMESTAMP: Date and time without timezone (microsecond precision)
- TIMESTAMP_TZ: Date and time with timezone
- TIMESTAMP_NS: Nanosecond precision timestamps
- TIMESTAMP_MS: Millisecond precision timestamps
- TIMESTAMP_S: Second precision timestamps

Use this to check if a column contains temporal data that should be handled
as Python datetime, date, or time objects.

DB API 2.0 Reference:
    https://peps.python.org/pep-0249/#datetime

Example:
    >>> cursor.description[2][1] == DATETIME  # Check if third column is date/time
"""

BINARY = DBAPITypeObject({"blob"})
"""
BINARY type object for binary data database columns.

This type object represents binary data types in DuckDB:
- BLOB: Binary Large Objects for storing arbitrary binary data

Use this to check if a column contains binary data that should be handled
as Python bytes objects.

DB API 2.0 Reference:
    https://peps.python.org/pep-0249/#binary

Example:
    >>> cursor.description[3][1] == BINARY  # Check if fourth column is binary
"""

ROWID = None
"""
ROWID type object for row identifier columns.

DB API 2.0 Reference:
    https://peps.python.org/pep-0249/#rowid

Note:
    This will always be None for DuckDB connections. Applications should not
    rely on ROWID functionality when using DuckDB.
"""

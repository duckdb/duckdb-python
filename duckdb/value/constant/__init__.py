# ruff: noqa: D101, D104, D105, D107, ANN401
from typing import TYPE_CHECKING, Any

from duckdb import sqltypes
from duckdb.sqltypes import DuckDBPyType

# A type constant bound in this module would sit in a dict that the Value classes' functions keep alive past
# finalization, so the names resolve through sqltypes at access time; only type checkers see them as imports.
_TYPE_CONSTANTS = (
    "BIGINT",
    "BIT",
    "BLOB",
    "BOOLEAN",
    "DATE",
    "DOUBLE",
    "FLOAT",
    "HUGEINT",
    "INTEGER",
    "INTERVAL",
    "SMALLINT",
    "SQLNULL",
    "TIME",
    "TIME_TZ",
    "TIMESTAMP",
    "TIMESTAMP_MS",
    "TIMESTAMP_NS",
    "TIMESTAMP_S",
    "TIMESTAMP_TZ",
    "TINYINT",
    "UBIGINT",
    "UHUGEINT",
    "UINTEGER",
    "USMALLINT",
    "UTINYINT",
    "UUID",
    "VARCHAR",
)

if TYPE_CHECKING:
    from duckdb.sqltypes import (  # noqa: F401
        BIGINT,
        BIT,
        BLOB,
        BOOLEAN,
        DATE,
        DOUBLE,
        FLOAT,
        HUGEINT,
        INTEGER,
        INTERVAL,
        SMALLINT,
        SQLNULL,
        TIME,
        TIME_TZ,
        TIMESTAMP,
        TIMESTAMP_MS,
        TIMESTAMP_NS,
        TIMESTAMP_S,
        TIMESTAMP_TZ,
        TINYINT,
        UBIGINT,
        UHUGEINT,
        UINTEGER,
        USMALLINT,
        UTINYINT,
        UUID,
        VARCHAR,
    )


class Value:
    def __init__(self, object: Any, type: DuckDBPyType) -> None:
        self.object = object
        self.type = type

    def __repr__(self) -> str:
        return str(self.object)


# Miscellaneous


class NullValue(Value):
    def __init__(self) -> None:
        super().__init__(None, sqltypes.SQLNULL)


class BooleanValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.BOOLEAN)


# Unsigned numerics


class UnsignedBinaryValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.UTINYINT)


class UnsignedShortValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.USMALLINT)


class UnsignedIntegerValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.UINTEGER)


class UnsignedLongValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.UBIGINT)


# Signed numerics


class BinaryValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.TINYINT)


class ShortValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.SMALLINT)


class IntegerValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.INTEGER)


class LongValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.BIGINT)


class HugeIntegerValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.HUGEINT)


class UnsignedHugeIntegerValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.UHUGEINT)


# Fractional


class FloatValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.FLOAT)


class DoubleValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.DOUBLE)


class DecimalValue(Value):
    def __init__(self, object: Any, width: int, scale: int) -> None:
        import duckdb

        decimal_type = duckdb.decimal_type(width, scale)
        super().__init__(object, decimal_type)


# String


class StringValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.VARCHAR)


class UUIDValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.UUID)


class BitValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.BIT)


class BlobValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.BLOB)


# Temporal


class DateValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.DATE)


class IntervalValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.INTERVAL)


class TimestampValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.TIMESTAMP)


class TimestampSecondValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.TIMESTAMP_S)


class TimestampMillisecondValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.TIMESTAMP_MS)


class TimestampNanosecondValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.TIMESTAMP_NS)


class TimestampTimeZoneValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.TIMESTAMP_TZ)


class TimeValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.TIME)


class TimeTimeZoneValue(Value):
    def __init__(self, object: Any) -> None:
        super().__init__(object, sqltypes.TIME_TZ)


class ListValue(Value):
    def __init__(self, object: Any, child_type: DuckDBPyType) -> None:
        import duckdb

        list_type = duckdb.list_type(child_type)
        super().__init__(object, list_type)


class StructValue(Value):
    def __init__(self, object: Any, children: dict[str, DuckDBPyType]) -> None:
        import duckdb

        struct_type = duckdb.struct_type(children)
        super().__init__(object, struct_type)


class MapValue(Value):
    def __init__(self, object: Any, key_type: DuckDBPyType, value_type: DuckDBPyType) -> None:
        import duckdb

        map_type = duckdb.map_type(key_type, value_type)
        super().__init__(object, map_type)


class UnionType(Value):
    def __init__(self, object: Any, members: dict[str, DuckDBPyType]) -> None:
        import duckdb

        union_type = duckdb.union_type(members)
        super().__init__(object, union_type)


if not TYPE_CHECKING:

    def __getattr__(name: str) -> DuckDBPyType:
        if name in _TYPE_CONSTANTS:
            return getattr(sqltypes, name)
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    def __dir__() -> list[str]:
        return sorted([*globals(), *_TYPE_CONSTANTS])


# TODO: add EnumValue once `duckdb.enum_type` is added  # noqa: TD002, TD003

__all__ = [
    "BinaryValue",
    "BitValue",
    "BlobValue",
    "BooleanValue",
    "DateValue",
    "DecimalValue",
    "DoubleValue",
    "FloatValue",
    "HugeIntegerValue",
    "IntegerValue",
    "IntervalValue",
    "LongValue",
    "NullValue",
    "ShortValue",
    "StringValue",
    "TimeTimeZoneValue",
    "TimeValue",
    "TimestampMillisecondValue",
    "TimestampNanosecondValue",
    "TimestampSecondValue",
    "TimestampTimeZoneValue",
    "TimestampValue",
    "UUIDValue",
    "UnsignedBinaryValue",
    "UnsignedHugeIntegerValue",
    "UnsignedIntegerValue",
    "UnsignedLongValue",
    "UnsignedShortValue",
    "Value",
]

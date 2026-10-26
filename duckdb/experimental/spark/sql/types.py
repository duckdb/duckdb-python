# This code is based on code from Apache Spark under the license found in the LICENSE
# file located in the 'spark' folder.

import calendar
import datetime
import decimal
import json
import math
import re
import time
import uuid
from builtins import tuple
from collections.abc import Callable, Iterator, Mapping
from functools import reduce
from types import MappingProxyType
from typing import Any, ClassVar, NoReturn, Self, TypeVar, Union, cast, overload

import duckdb
from duckdb.sqltypes import DuckDBPyType

from ..errors import ParseException, PySparkNotImplementedError, PySparkTypeError, PySparkValueError
from ..exception import ContributionsAcceptedError

T = TypeVar("T")
U = TypeVar("U")

_JVM_INT_MAX = 2**31 - 1

__all__ = [
    "ArrayType",
    "BinaryType",
    "BitstringType",
    "BooleanType",
    "ByteType",
    "DataType",
    "DateType",
    "DayTimeIntervalType",
    "DecimalType",
    "DoubleType",
    "FloatType",
    "HugeIntegerType",
    "IntegerType",
    "LongType",
    "MapType",
    "NullType",
    "Row",
    "ShortType",
    "StringType",
    "StructField",
    "StructType",
    "TimeNSType",
    "TimeNTZType",
    "TimeType",
    "TimestampMillisecondNTZType",
    "TimestampNTZType",
    "TimestampNanosecondNTZType",
    "TimestampNanosecondType",
    "TimestampSecondNTZType",
    "TimestampType",
    "UUIDType",
    "UnsignedByteType",
    "UnsignedHugeIntegerType",
    "UnsignedIntegerType",
    "UnsignedLongType",
    "UnsignedShortType",
    "VariantType",
]


class DataType:
    """Base class for data types."""

    def __init__(self, duckdb_type: DuckDBPyType) -> None:  # noqa: D107
        self.duckdb_type = duckdb_type

    def __repr__(self) -> str:  # noqa: D105
        return self.__class__.__name__ + "()"

    def __hash__(self) -> int:  # noqa: D105
        return hash(str(self))

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other: object) -> bool:  # noqa: D105
        return not self.__eq__(other)

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return cls.__name__[:-4].lower()

    def simpleString(self) -> str:  # noqa: D102
        return self.typeName()

    def jsonValue(self) -> str | dict[str, Any]:  # noqa: D102
        return self.typeName()

    def json(self) -> str:  # noqa: D102
        return json.dumps(self.jsonValue(), separators=(",", ":"), sort_keys=True)

    def needConversion(self) -> bool:
        """Does this type needs conversion between Python object and internal SQL object.

        This is used to avoid the unnecessary conversion for ArrayType/MapType/StructType.
        """
        return False

    def toInternal(self, obj: Any) -> Any:  # noqa: ANN401
        """Converts a Python object into an internal SQL object."""
        return obj

    def fromInternal(self, obj: Any) -> Any:  # noqa: ANN401
        """Converts an internal SQL object into a native Python object."""
        return obj


# This singleton pattern does not work with pickle, you will get
# another object after pickle and unpickle
class DataTypeSingleton(type):
    """Metaclass for DataType."""

    _instances: ClassVar[dict[type["DataTypeSingleton"], "DataTypeSingleton"]] = {}

    def __call__(cls: type[T]) -> T:  # type: ignore[override]
        if cls not in cls._instances:  # type: ignore[attr-defined]
            cls._instances[cls] = super().__call__()  # type: ignore[misc, attr-defined]
        return cls._instances[cls]  # type: ignore[attr-defined]


class NullType(DataType, metaclass=DataTypeSingleton):
    """Null type.

    The data type representing None, used for the types that cannot be inferred.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("NULL"))

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "void"


class AtomicType(DataType):
    """An internal type used to represent everything that is not
    null, UDTs, arrays, structs, and maps.
    """  # noqa: D205


class NumericType(AtomicType):
    """Numeric data types."""


class IntegralType(NumericType, metaclass=DataTypeSingleton):
    """Integral data types."""


class FractionalType(NumericType):
    """Fractional data types."""


class StringType(AtomicType, metaclass=DataTypeSingleton):
    """String data type."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("VARCHAR"))


class BitstringType(AtomicType, metaclass=DataTypeSingleton):
    """Bitstring data type."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("BIT"))


class UUIDType(AtomicType, metaclass=DataTypeSingleton):
    """UUID data type."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("UUID"))


class BinaryType(AtomicType, metaclass=DataTypeSingleton):
    """Binary (byte array) data type."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("BLOB"))


class BooleanType(AtomicType, metaclass=DataTypeSingleton):
    """Boolean data type."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("BOOLEAN"))


class VariantType(AtomicType, metaclass=DataTypeSingleton):
    """Variant (semi-structured) data type."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("VARIANT"))


class DateType(AtomicType, metaclass=DataTypeSingleton):
    """Date (datetime.date) data type."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("DATE"))

    EPOCH_ORDINAL = datetime.datetime(1970, 1, 1).toordinal()

    def needConversion(self) -> bool:  # noqa: D102
        return True

    def toInternal(self, d: datetime.date) -> int:  # noqa: D102
        if d is not None:
            return d.toordinal() - self.EPOCH_ORDINAL

    def fromInternal(self, v: int) -> datetime.date:  # noqa: D102
        if v is not None:
            return datetime.date.fromordinal(v + self.EPOCH_ORDINAL)


class TimestampType(AtomicType, metaclass=DataTypeSingleton):
    """Timestamp (datetime.datetime) data type."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("TIMESTAMPTZ"))

    def needConversion(self) -> bool:  # noqa: D102
        return True

    def toInternal(self, dt: datetime.datetime) -> int:  # noqa: D102
        if dt is not None:
            seconds = calendar.timegm(dt.utctimetuple()) if dt.tzinfo else time.mktime(dt.timetuple())
            return int(seconds) * 1000000 + dt.microsecond

    def fromInternal(self, ts: int) -> datetime.datetime:  # noqa: D102
        if ts is not None:
            # using int to avoid precision loss in float
            return datetime.datetime.fromtimestamp(ts // 1000000).replace(microsecond=ts % 1000000)


class TimestampNanosecondType(AtomicType, metaclass=DataTypeSingleton):
    """Timestamp (datetime.datetime) data type with timezone information with nanosecond precision."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("TIMESTAMPTZ_NS"))

    def needConversion(self) -> bool:  # noqa: D102
        return True

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "timestamptz_ns"

    def toInternal(self, dt: datetime.datetime) -> int:  # noqa: D102
        raise ContributionsAcceptedError

    def fromInternal(self, ts: int) -> datetime.datetime:  # noqa: D102
        raise ContributionsAcceptedError


class TimestampNTZType(AtomicType, metaclass=DataTypeSingleton):
    """Timestamp (datetime.datetime) data type without timezone information with microsecond precision."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("TIMESTAMP"))

    def needConversion(self) -> bool:  # noqa: D102
        return True

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "timestamp_ntz"

    def toInternal(self, dt: datetime.datetime) -> int:  # noqa: D102
        if dt is not None:
            seconds = calendar.timegm(dt.timetuple())
            return int(seconds) * 1000000 + dt.microsecond

    def fromInternal(self, ts: int) -> datetime.datetime:  # noqa: D102
        if ts is not None:
            # using int to avoid precision loss in float
            return datetime.datetime.utcfromtimestamp(ts // 1000000).replace(microsecond=ts % 1000000)


class TimestampSecondNTZType(AtomicType, metaclass=DataTypeSingleton):
    """Timestamp (datetime.datetime) data type without timezone information with second precision."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("TIMESTAMP_S"))

    def needConversion(self) -> bool:  # noqa: D102
        return True

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "timestamp_s"

    def toInternal(self, dt: datetime.datetime) -> int:  # noqa: D102
        raise ContributionsAcceptedError

    def fromInternal(self, ts: int) -> datetime.datetime:  # noqa: D102
        raise ContributionsAcceptedError


class TimestampMillisecondNTZType(AtomicType, metaclass=DataTypeSingleton):
    """Timestamp (datetime.datetime) data type without timezone information with millisecond precision."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("TIMESTAMP_MS"))

    def needConversion(self) -> bool:  # noqa: D102
        return True

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "timestamp_ms"

    def toInternal(self, dt: datetime.datetime) -> int:  # noqa: D102
        raise ContributionsAcceptedError

    def fromInternal(self, ts: int) -> datetime.datetime:  # noqa: D102
        raise ContributionsAcceptedError


class TimestampNanosecondNTZType(AtomicType, metaclass=DataTypeSingleton):
    """Timestamp (datetime.datetime) data type without timezone information with nanosecond precision."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("TIMESTAMP_NS"))

    def needConversion(self) -> bool:  # noqa: D102
        return True

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "timestamp_ns"

    def toInternal(self, dt: datetime.datetime) -> int:  # noqa: D102
        raise ContributionsAcceptedError

    def fromInternal(self, ts: int) -> datetime.datetime:  # noqa: D102
        raise ContributionsAcceptedError


class DecimalType(FractionalType):
    """Decimal (decimal.Decimal) data type.

    The DecimalType must have fixed precision (the maximum total number of digits)
    and scale (the number of digits on the right of dot). For example, (5, 2) can
    support the value from [-999.99 to 999.99].

    The precision can be up to 38, the scale must be less or equal to precision.

    When creating a DecimalType, the default precision and scale is (10, 0). When inferring
    schema from decimal.Decimal objects, it will be DecimalType(38, 18).

    Parameters
    ----------
    precision : int, optional
        the maximum (i.e. total) number of digits (default: 10)
    scale : int, optional
        the number of digits on right side of dot. (default: 0)
    """

    def __init__(self, precision: int = 10, scale: int = 0) -> None:  # noqa: D107
        super().__init__(duckdb.decimal_type(precision, scale))
        self.precision = precision
        self.scale = scale
        self.hasPrecisionInfo = True  # this is a public API

    def simpleString(self) -> str:  # noqa: D102
        return f"decimal({int(self.precision):d},{int(self.scale):d})"

    jsonValue = simpleString

    def __repr__(self) -> str:  # noqa: D105
        return f"DecimalType({int(self.precision):d},{int(self.scale):d})"


class DoubleType(FractionalType, metaclass=DataTypeSingleton):
    """Double data type, representing double precision floats."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("DOUBLE"))


class FloatType(FractionalType, metaclass=DataTypeSingleton):
    """Float data type, representing single precision floats."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("FLOAT"))


class ByteType(IntegralType):
    """Byte data type, i.e. a signed integer in a single byte."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("TINYINT"))

    def simpleString(self) -> str:  # noqa: D102
        return "tinyint"


class UnsignedByteType(IntegralType):
    """Unsigned byte data type, i.e. a unsigned integer in a single byte."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("UTINYINT"))

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "utinyint"


class ShortType(IntegralType):
    """Short data type, i.e. a signed 16-bit integer."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("SMALLINT"))

    def simpleString(self) -> str:  # noqa: D102
        return "smallint"


class UnsignedShortType(IntegralType):
    """Unsigned short data type, i.e. a unsigned 16-bit integer."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("USMALLINT"))

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "usmallint"


class IntegerType(IntegralType):
    """Int data type, i.e. a signed 32-bit integer."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("INTEGER"))

    def simpleString(self) -> str:  # noqa: D102
        return "int"


class UnsignedIntegerType(IntegralType):
    """Unsigned int data type, i.e. a unsigned 32-bit integer."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("UINTEGER"))

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "uinteger"


class LongType(IntegralType):
    """Long data type, i.e. a signed 64-bit integer.

    If the values are beyond the range of [-9223372036854775808, 9223372036854775807],
    please use :class:`DecimalType`.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("BIGINT"))

    def simpleString(self) -> str:  # noqa: D102
        return "bigint"


class UnsignedLongType(IntegralType):
    """Unsigned long data type, i.e. a unsigned 64-bit integer.

    If the values are beyond the range of [0, 18446744073709551615],
    please use :class:`HugeIntegerType`.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("UBIGINT"))

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "ubigint"


class HugeIntegerType(IntegralType):
    """Huge integer data type, i.e. a signed 128-bit integer.

    If the values are beyond the range of [-170141183460469231731687303715884105728,
    170141183460469231731687303715884105727], please use :class:`DecimalType`.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("HUGEINT"))

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "hugeint"


class UnsignedHugeIntegerType(IntegralType):
    """Unsigned huge integer data type, i.e. a unsigned 128-bit integer.

    If the values are beyond the range of [0, 340282366920938463463374607431768211455],
    please use :class:`DecimalType`.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("UHUGEINT"))

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "uhugeint"


class TimeType(IntegralType):
    """Time (datetime.time) data type."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("TIMETZ"))

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "timetz"


class TimeNTZType(IntegralType):
    """Time (datetime.time) data type without timezone information."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("TIME"))

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "time"


class TimeNSType(IntegralType):
    """Time NS (datetime.time) data type without timezone information."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(DuckDBPyType("TIME_NS"))

    @classmethod
    def typeName(cls) -> str:  # noqa: D102
        return "time_ns"


class DayTimeIntervalType(AtomicType):
    """DayTimeIntervalType (datetime.timedelta)."""

    DAY = 0
    HOUR = 1
    MINUTE = 2
    SECOND = 3

    _fields: Mapping[str, int] = MappingProxyType(
        {
            DAY: "day",
            HOUR: "hour",
            MINUTE: "minute",
            SECOND: "second",
        }
    )

    _inverted_fields: Mapping[int, str] = MappingProxyType(dict(zip(_fields.values(), _fields.keys(), strict=False)))

    def __init__(self, startField: int | None = None, endField: int | None = None) -> None:  # noqa: D107
        if startField is None and endField is None:
            # Default matched to scala side.
            startField = DayTimeIntervalType.DAY
            endField = DayTimeIntervalType.SECOND
        elif startField is not None and endField is None:
            endField = startField

        fields = DayTimeIntervalType._fields
        if startField not in fields or endField not in fields:
            msg = f"interval {startField} to {endField} is invalid"
            raise RuntimeError(msg)
        self.startField = cast("int", startField)
        self.endField = cast("int", endField)

    @property
    def duckdb_type(self) -> DuckDBPyType:  # noqa: D102
        # A DuckDB interval has no fields, so a DataFrame could only report it as interval day to second.
        if (self.startField, self.endField) != (DayTimeIntervalType.DAY, DayTimeIntervalType.SECOND):
            msg = f"The Spark type {self.simpleString()} is not supported"
            raise ContributionsAcceptedError(msg)
        return DuckDBPyType("INTERVAL")

    def _str_repr(self) -> str:
        fields = DayTimeIntervalType._fields
        start_field_name = fields[self.startField]
        end_field_name = fields[self.endField]
        if start_field_name == end_field_name:
            return f"interval {start_field_name}"
        else:
            return f"interval {start_field_name} to {end_field_name}"

    simpleString = _str_repr
    jsonValue = _str_repr

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}({int(self.startField):d}, {int(self.endField):d})"

    def needConversion(self) -> bool:  # noqa: D102
        return True

    def toInternal(self, dt: datetime.timedelta) -> int | None:  # noqa: D102
        if dt is not None:
            return (math.floor(dt.total_seconds()) * 1000000) + dt.microseconds

    def fromInternal(self, micros: int) -> datetime.timedelta | None:  # noqa: D102
        if micros is not None:
            return datetime.timedelta(microseconds=micros)


class ArrayType(DataType):
    """Array data type.

    Parameters
    ----------
    elementType : :class:`DataType`
        :class:`DataType` of each element in the array.
    containsNull : bool, optional
        whether the array can contain null (None) values.

    Examples:
    --------
    >>> ArrayType(StringType()) == ArrayType(StringType(), True)
    True
    >>> ArrayType(StringType(), False) == ArrayType(StringType())
    False
    """

    def __init__(self, elementType: DataType, containsNull: bool = True) -> None:  # noqa: D107
        super().__init__(duckdb.list_type(elementType.duckdb_type))
        assert isinstance(elementType, DataType), f"elementType {elementType} should be an instance of {DataType}"
        self.elementType = elementType
        self.containsNull = containsNull

    def simpleString(self) -> str:  # noqa: D102
        return f"array<{self.elementType.simpleString()}>"

    def __repr__(self) -> str:  # noqa: D105
        return f"ArrayType({self.elementType}, {self.containsNull!s})"

    def jsonValue(self) -> dict[str, Any]:  # noqa: D102
        return {
            "type": self.typeName(),
            "elementType": self.elementType.jsonValue(),
            "containsNull": self.containsNull,
        }

    def needConversion(self) -> bool:  # noqa: D102
        return self.elementType.needConversion()

    def toInternal(self, obj: list[T | None]) -> list[T | None]:  # noqa: D102
        if not self.needConversion():
            return obj
        return obj and [self.elementType.toInternal(v) for v in obj]

    def fromInternal(self, obj: list[T | None]) -> list[T | None]:  # noqa: D102
        if not self.needConversion():
            return obj
        return obj and [self.elementType.fromInternal(v) for v in obj]


class MapType(DataType):
    """Map data type.

    Parameters
    ----------
    keyType : :class:`DataType`
        :class:`DataType` of the keys in the map.
    valueType : :class:`DataType`
        :class:`DataType` of the values in the map.
    valueContainsNull : bool, optional
        indicates whether values can contain null (None) values.

    Notes:
    -----
    Keys in a map data type are not allowed to be null (None).

    Examples:
    --------
    >>> (MapType(StringType(), IntegerType()) == MapType(StringType(), IntegerType(), True))
    True
    >>> (MapType(StringType(), IntegerType(), False) == MapType(StringType(), FloatType()))
    False
    """

    def __init__(self, keyType: DataType, valueType: DataType, valueContainsNull: bool = True) -> None:  # noqa: D107
        super().__init__(duckdb.map_type(keyType.duckdb_type, valueType.duckdb_type))
        assert isinstance(keyType, DataType), f"keyType {keyType} should be an instance of {DataType}"
        assert isinstance(valueType, DataType), f"valueType {valueType} should be an instance of {DataType}"
        self.keyType = keyType
        self.valueType = valueType
        self.valueContainsNull = valueContainsNull

    def simpleString(self) -> str:  # noqa: D102
        return f"map<{self.keyType.simpleString()},{self.valueType.simpleString()}>"

    def __repr__(self) -> str:  # noqa: D105
        return f"MapType({self.keyType}, {self.valueType}, {self.valueContainsNull!s})"

    def jsonValue(self) -> dict[str, Any]:  # noqa: D102
        return {
            "type": self.typeName(),
            "keyType": self.keyType.jsonValue(),
            "valueType": self.valueType.jsonValue(),
            "valueContainsNull": self.valueContainsNull,
        }

    def needConversion(self) -> bool:  # noqa: D102
        return self.keyType.needConversion() or self.valueType.needConversion()

    def toInternal(self, obj: dict[T, U | None]) -> dict[T, U | None]:  # noqa: D102
        if not self.needConversion():
            return obj
        return obj and {self.keyType.toInternal(k): self.valueType.toInternal(v) for k, v in obj.items()}

    def fromInternal(self, obj: dict[T, U | None]) -> dict[T, U | None]:  # noqa: D102
        if not self.needConversion():
            return obj
        return obj and {self.keyType.fromInternal(k): self.valueType.fromInternal(v) for k, v in obj.items()}


class StructField(DataType):
    """A field in :class:`StructType`.

    Parameters
    ----------
    name : str
        name of the field.
    dataType : :class:`DataType`
        :class:`DataType` of the field.
    nullable : bool, optional
        whether the field can be null (None) or not.
    metadata : dict, optional
        a dict from string to simple type that can be toInternald to JSON automatically

    Examples:
    --------
    >>> (StructField("f1", StringType(), True) == StructField("f1", StringType(), True))
    True
    >>> (StructField("f1", StringType(), True) == StructField("f2", StringType(), True))
    False
    """

    def __init__(  # noqa: D107
        self,
        name: str,
        dataType: DataType,
        nullable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        assert isinstance(dataType, DataType), f"dataType {dataType} should be an instance of {DataType}"
        assert isinstance(name, str), f"field name {name} should be a string"
        self.name = name
        self.dataType = dataType
        self.nullable = nullable
        self.metadata = metadata or {}

    @property
    def duckdb_type(self) -> DuckDBPyType:  # noqa: D102
        return self.dataType.duckdb_type

    def simpleString(self) -> str:  # noqa: D102
        return f"{self.name}:{self.dataType.simpleString()}"

    def __repr__(self) -> str:  # noqa: D105
        return f"StructField('{self.name}', {self.dataType}, {self.nullable!s})"

    def jsonValue(self) -> dict[str, Any]:  # noqa: D102
        return {
            "name": self.name,
            "type": self.dataType.jsonValue(),
            "nullable": self.nullable,
            "metadata": self.metadata,
        }

    def needConversion(self) -> bool:  # noqa: D102
        return self.dataType.needConversion()

    def toInternal(self, obj: T) -> T:  # noqa: D102
        return self.dataType.toInternal(obj)

    def fromInternal(self, obj: T) -> T:  # noqa: D102
        return self.dataType.fromInternal(obj)

    def typeName(self) -> str:  # type: ignore[override]  # noqa: D102
        raise PySparkTypeError(error_class="INVALID_TYPENAME_CALL", message_parameters={})


class StructType(DataType):
    r"""Struct type, consisting of a list of :class:`StructField`.

    This is the data type representing a :class:`Row`.

    Iterating a :class:`StructType` will iterate over its :class:`StructField`\\s.
    A contained :class:`StructField` can be accessed by its name or position.

    Examples:
    --------
    >>> struct1 = StructType([StructField("f1", StringType(), True)])
    >>> struct1["f1"]
    StructField('f1', StringType(), True)
    >>> struct1[0]
    StructField('f1', StringType(), True)

    >>> struct1 = StructType([StructField("f1", StringType(), True)])
    >>> struct2 = StructType([StructField("f1", StringType(), True)])
    >>> struct1 == struct2
    True
    >>> struct1 = StructType([StructField("f1", StringType(), True)])
    >>> struct2 = StructType(
    ...     [StructField("f1", StringType(), True), StructField("f2", IntegerType(), False)]
    ... )
    >>> struct1 == struct2
    False
    """

    def __init__(self, fields: list[StructField] | None = None) -> None:  # noqa: D107
        if not fields:
            self.fields = []
            self.names = []
        else:
            self.fields = fields
            self.names = [f.name for f in fields]
            assert all(isinstance(f, StructField) for f in fields), "fields should be a list of StructField"
        # Precalculated list of fields that need conversion with fromInternal/toInternal functions
        self._needConversion = [f.needConversion() for f in self]
        self._needSerializeAnyField = any(self._needConversion)

    @property
    def duckdb_type(self) -> DuckDBPyType:  # noqa: D102
        if self.names and not any(self.names):
            # The struct DuckDB builds without field names, such as the elements of list_zip.
            return DuckDBPyType(f"TUPLE({', '.join(str(f.duckdb_type) for f in self.fields)})")
        folded = [name.casefold() for name in self.names]
        if len(set(folded)) < len(folded):
            # DuckDB matches struct field names case insensitively, so it cannot hold two fields with one name.
            msg = f"DuckDB has no struct type with two fields named alike: {self.simpleString()}"
            raise PySparkNotImplementedError(message=msg)
        return duckdb.struct_type(dict(zip(self.names, [x.duckdb_type for x in self.fields], strict=True)))

    @overload
    def add(
        self,
        field: str,
        data_type: str | DataType,
        nullable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> "StructType": ...

    @overload
    def add(self, field: StructField) -> "StructType": ...

    def add(
        self,
        field: str | StructField,
        data_type: str | DataType | None = None,
        nullable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> "StructType":
        r"""Construct a :class:`StructType` by adding new elements to it, to define the schema.
        The method accepts either:

            a) A single parameter which is a :class:`StructField` object.
            b) Between 2 and 4 parameters as (name, data_type, nullable (optional),
               metadata(optional). The data_type parameter may be either a String or a
               :class:`DataType` object.

        Parameters
        ----------
        field : str or :class:`StructField`
            Either the name of the field or a :class:`StructField` object
        data_type : :class:`DataType`, optional
            If present, the DataType of the :class:`StructField` to create
        nullable : bool, optional
            Whether the field to add should be nullable (default True)
        metadata : dict, optional
            Any additional metadata (default None)

        Returns:
        -------
        :class:`StructType`

        Examples:
        --------
        >>> struct1 = StructType().add("f1", StringType(), True).add("f2", StringType(), True, None)
        >>> struct2 = StructType([StructField("f1", StringType(), True), \\
        ...     StructField("f2", StringType(), True, None)])
        >>> struct1 == struct2
        True
        >>> struct1 = StructType().add(StructField("f1", StringType(), True))
        >>> struct2 = StructType([StructField("f1", StringType(), True)])
        >>> struct1 == struct2
        True
        >>> struct1 = StructType().add("f1", "string", True)
        >>> struct2 = StructType([StructField("f1", StringType(), True)])
        >>> struct1 == struct2
        True
        """  # noqa: D205, D415
        if isinstance(field, StructField):
            self.fields.append(field)
            self.names.append(field.name)
        else:
            if isinstance(field, str) and data_type is None:
                msg = "Must specify DataType if passing name of struct_field to create."
                raise ValueError(msg)
            elif isinstance(data_type, str):
                data_type_f = _parse_type_name(data_type)
            else:
                data_type_f = data_type
            self.fields.append(StructField(field, data_type_f, nullable, metadata))
            self.names.append(field)
        # Precalculated list of fields that need conversion with fromInternal/toInternal functions
        self._needConversion = [f.needConversion() for f in self]
        self._needSerializeAnyField = any(self._needConversion)
        return self

    def __iter__(self) -> Iterator[StructField]:
        """Iterate the fields."""
        return iter(self.fields)

    def __len__(self) -> int:
        """Return the number of fields."""
        return len(self.fields)

    def __getitem__(self, key: str | int) -> StructField:
        """Access fields by name or slice."""
        if isinstance(key, str):
            for field in self:
                if field.name == key:
                    return field
            msg = f"No StructField named {key}"
            raise KeyError(msg)
        elif isinstance(key, int):
            try:
                return self.fields[key]
            except IndexError:
                msg = "StructType index out of range"
                raise IndexError(msg)  # noqa: B904
        elif isinstance(key, slice):
            return StructType(self.fields[key])
        else:
            msg = "StructType keys should be strings, integers or slices"
            raise TypeError(msg)

    def simpleString(self) -> str:  # noqa: D102
        return "struct<{}>".format(",".join(f.simpleString() for f in self))

    def __repr__(self) -> str:  # noqa: D105
        return "StructType([{}])".format(", ".join(str(field) for field in self))

    def jsonValue(self) -> dict[str, Any]:  # noqa: D102
        return {"type": self.typeName(), "fields": [f.jsonValue() for f in self]}

    def __contains__(self, item: str) -> bool:  # noqa: D105
        return item in self.names

    def extract_types_and_names(self) -> tuple[list[str], list[str]]:  # noqa: D102
        names = []
        types = []
        for f in self.fields:
            types.append(str(f.dataType.duckdb_type))
            names.append(f.name)
        return (types, names)

    def fieldNames(self) -> list[str]:
        """Returns all field names in a list.

        Examples:
        --------
        >>> struct = StructType([StructField("f1", StringType(), True)])
        >>> struct.fieldNames()
        ['f1']
        """
        return list(self.names)

    def treeString(self, maxDepth: int = _JVM_INT_MAX) -> str:
        """Returns a string representation of the schema in tree format.

        Parameters
        ----------
        maxDepth : int, optional
            Maximum depth to print. Prints all levels when not positive.

        Returns:
        -------
        str
            Tree-formatted schema string

        Examples:
        --------
        >>> schema = StructType([StructField("age", IntegerType(), True)])
        >>> print(schema.treeString())
        root
         |-- age: integer (nullable = true)
        <BLANKLINE>
        """
        lines = ["root", *_tree_lines(self, " |", maxDepth if maxDepth > 0 else _JVM_INT_MAX)]
        return "\n".join(lines) + "\n"

    def needConversion(self) -> bool:  # noqa: D102
        # We need convert Row()/namedtuple into tuple()
        return True

    def toInternal(self, obj: tuple) -> tuple:  # noqa: D102
        if obj is None:
            return

        if self._needSerializeAnyField:
            # Only calling toInternal function for fields that need conversion
            if isinstance(obj, dict):
                return tuple(
                    f.toInternal(obj.get(n)) if c else obj.get(n)
                    for n, f, c in zip(self.names, self.fields, self._needConversion, strict=False)
                )
            elif isinstance(obj, (tuple, list)):
                return tuple(
                    f.toInternal(v) if c else v for f, v, c in zip(self.fields, obj, self._needConversion, strict=False)
                )
            elif hasattr(obj, "__dict__"):
                d = obj.__dict__
                return tuple(
                    f.toInternal(d.get(n)) if c else d.get(n)
                    for n, f, c in zip(self.names, self.fields, self._needConversion, strict=False)
                )
            else:
                msg = f"Unexpected tuple {obj!r} with StructType"
                raise ValueError(msg)
        else:
            if isinstance(obj, dict):
                return tuple(obj.get(n) for n in self.names)
            elif isinstance(obj, (list, tuple)):
                return tuple(obj)
            elif hasattr(obj, "__dict__"):
                d = obj.__dict__
                return tuple(d.get(n) for n in self.names)
            else:
                msg = f"Unexpected tuple {obj!r} with StructType"
                raise ValueError(msg)

    def fromInternal(self, obj: tuple) -> "Row":  # noqa: D102
        if obj is None:
            return
        if isinstance(obj, Row):
            # it's already converted by pickler
            return obj

        values: tuple | list
        if self._needSerializeAnyField:
            # Only calling fromInternal function for fields that need conversion
            values = [
                f.fromInternal(v) if c else v for f, v, c in zip(self.fields, obj, self._needConversion, strict=False)
            ]
        else:
            values = obj
        return _create_row(self.names, values)


def _tree_type_name(data_type: DataType) -> str:
    if isinstance(data_type, (DecimalType, DayTimeIntervalType)):
        return data_type.simpleString()
    return data_type.typeName()


def _tree_lines(data_type: DataType, prefix: str, max_depth: int) -> list[str]:
    """Returns the lines treeString prints for the children of a nested type."""
    if max_depth <= 0:
        return []
    if isinstance(data_type, StructType):
        children = [
            (_escape_meta_characters(f.name), f.dataType, f" (nullable = {str(f.nullable).lower()})")
            for f in data_type.fields
        ]
    elif isinstance(data_type, ArrayType):
        children = [("element", data_type.elementType, f" (containsNull = {str(data_type.containsNull).lower()})")]
    elif isinstance(data_type, MapType):
        children = [
            ("key", data_type.keyType, ""),
            ("value", data_type.valueType, f" (valueContainsNull = {str(data_type.valueContainsNull).lower()})"),
        ]
    else:
        return []
    lines = []
    for name, child_type, flag in children:
        lines.append(f"{prefix}-- {name}: {_tree_type_name(child_type)}{flag}")
        lines.extend(_tree_lines(child_type, f"{prefix}    |", max_depth - 1))
    return lines


_META_CHARACTERS = str.maketrans(
    {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\f": "\\f", "\b": "\\b", "\v": "\\v", "\a": "\\a"}
)


def _escape_meta_characters(text: str) -> str:
    return text.translate(_META_CHARACTERS)


class UnionType(DataType):
    def __init__(self) -> None:
        raise ContributionsAcceptedError


class UserDefinedType(DataType):
    """User-defined type (UDT).

    .. note:: WARN: Spark Internal Use Only
    """

    def __init__(self) -> None:
        raise ContributionsAcceptedError

    @classmethod
    def typeName(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def sqlType(cls) -> DataType:
        """Underlying SQL storage type for this UDT."""
        msg = "UDT must implement sqlType()."
        raise NotImplementedError(msg)

    @classmethod
    def module(cls) -> str:
        """The Python module of the UDT."""
        msg = "UDT must implement module()."
        raise NotImplementedError(msg)

    @classmethod
    def scalaUDT(cls) -> str:
        """The class name of the paired Scala UDT (could be '', if there
        is no corresponding one).
        """  # noqa: D205
        return ""

    def needConversion(self) -> bool:
        return True

    @classmethod
    def _cachedSqlType(cls) -> DataType:
        """Cache the sqlType() into class, because it's heavily used in `toInternal`."""
        if not hasattr(cls, "_cached_sql_type"):
            cls._cached_sql_type = cls.sqlType()  # type: ignore[attr-defined]
        return cls._cached_sql_type  # type: ignore[attr-defined]

    def toInternal(self, obj: Any) -> Any:  # noqa: ANN401
        if obj is not None:
            return self._cachedSqlType().toInternal(self.serialize(obj))

    def fromInternal(self, obj: Any) -> Any:  # noqa: ANN401
        v = self._cachedSqlType().fromInternal(obj)
        if v is not None:
            return self.deserialize(v)

    def serialize(self, obj: Any) -> NoReturn:  # noqa: ANN401
        """Converts a user-type object into a SQL datum."""
        msg = "UDT must implement toInternal()."
        raise NotImplementedError(msg)

    def deserialize(self, datum: Any) -> NoReturn:  # noqa: ANN401
        """Converts a SQL datum into a user-type object."""
        msg = "UDT must implement fromInternal()."
        raise NotImplementedError(msg)

    def simpleString(self) -> str:
        return "udt"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)


_atomic_types: list[type[DataType]] = [
    StringType,
    BinaryType,
    BooleanType,
    DecimalType,
    FloatType,
    DoubleType,
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    DateType,
    TimestampType,
    TimestampNTZType,
    NullType,
]
_all_atomic_types: dict[str, type[DataType]] = {t.typeName(): t for t in _atomic_types}

_complex_types: list[type[ArrayType | MapType | StructType]] = [
    ArrayType,
    MapType,
    StructType,
]
_all_complex_types: dict[str, type[ArrayType | MapType | StructType]] = {v.typeName(): v for v in _complex_types}


_FIXED_DECIMAL = re.compile(r"decimal\(\s*(\d+)\s*,\s*(-?\d+)\s*\)")
_INTERVAL_DAYTIME = re.compile(r"interval (day|hour|minute|second)( to (day|hour|minute|second))?")


_DDL_TOKEN = re.compile(r"""\s*(`(?:[^`]|``)*`|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\w+|\S)""")

_DUCKDB_ONLY_ATOMIC_TYPES: list[type[DataType]] = [
    UnsignedByteType,
    UnsignedShortType,
    UnsignedIntegerType,
    UnsignedLongType,
    HugeIntegerType,
    UnsignedHugeIntegerType,
    UUIDType,
    BitstringType,
    TimeType,
    TimeNSType,
    TimestampSecondNTZType,
    TimestampMillisecondNTZType,
    TimestampNanosecondNTZType,
    TimestampNanosecondType,
]

# DuckDB only types are known by the name the module prints for them; Spark's names take precedence.
_DDL_ATOMIC_TYPES: dict[str, type[DataType]] = {t().simpleString(): t for t in _DUCKDB_ONLY_ATOMIC_TYPES} | {
    "boolean": BooleanType,
    "tinyint": ByteType,
    "byte": ByteType,
    "smallint": ShortType,
    "short": ShortType,
    "int": IntegerType,
    "integer": IntegerType,
    "bigint": LongType,
    "long": LongType,
    "float": FloatType,
    "real": FloatType,
    "double": DoubleType,
    "string": StringType,
    "binary": BinaryType,
    "date": DateType,
    "timestamp": TimestampType,
    "timestamp_ltz": TimestampType,
    "timestamp_ntz": TimestampNTZType,
    "time": TimeNTZType,
    "variant": VariantType,
    "void": NullType,
}

# The types by the name typeName() gives them; Spark's names take precedence.
_TYPE_NAMES: dict[str, type[DataType]] = {
    t.typeName(): t for t in [*_DUCKDB_ONLY_ATOMIC_TYPES, TimeNTZType]
} | _all_atomic_types


def _parse_type_name(name: str) -> DataType:
    """Reads a type from its JSON name, which is what Spark expects of a str given to ``StructType.add``."""
    if name in _TYPE_NAMES:
        return _TYPE_NAMES[name]()
    if match := _FIXED_DECIMAL.match(name):
        return DecimalType(int(match.group(1)), int(match.group(2)))
    if _INTERVAL_DAYTIME.match(name):
        return _parse_datatype_string(name)
    raise PySparkValueError(error_class="CANNOT_PARSE_DATATYPE", message_parameters={"msg": name})


_DDL_DECIMAL_TYPES = frozenset(["decimal", "dec", "numeric"])
_DDL_UNSUPPORTED_TYPES = frozenset(["char", "character", "varchar"])
_DDL_DAY_TIME_FIELDS = DayTimeIntervalType._inverted_fields
_DDL_YEAR_MONTH_FIELDS = {"year": 0, "month": 1}


def _is_ddl_name(token: str | None) -> bool:
    if token is None:
        return False
    return re.fullmatch(r"\w+", token) is not None or (len(token) > 1 and token[0] == token[-1] == "`")


class _DDLParser:
    """Parser for Spark's DDL syntax for types and schemas, such as ``a int, b array<string>``."""

    def __init__(self, text: str, cast: bool = False) -> None:
        self.cast = cast
        self.tokens: list[str] = []
        pos = 0
        while (match := _DDL_TOKEN.match(text, pos)) is not None:
            self.tokens.append(match.group(1))
            pos = match.end()
        self.pos = 0

    def parse(self) -> DataType:
        """Parses a list of fields, with or without a colon after each name, or else a single type, as Spark does.

        Text is a list of fields if it is empty or a colon or a name follows its first name, except an interval type,
        the one type of two names. The type of a cast is a single type.
        """
        second = self.tokens[1].lower() if len(self.tokens) > 1 else None
        is_interval = self._peek() == "interval" and second in {*_DDL_DAY_TIME_FIELDS, *_DDL_YEAR_MONTH_FIELDS}
        is_fields = not self.tokens or ((second == ":" or _is_ddl_name(second)) and not is_interval)
        result = self._fields() if is_fields and not self.cast else self._type()
        if self.pos < len(self.tokens):
            self._fail()
        return result

    def _peek(self) -> str | None:
        return self.tokens[self.pos].lower() if self.pos < len(self.tokens) else None

    def _next(self) -> str:
        if self.pos == len(self.tokens):
            self._fail()
        self.pos += 1
        return self.tokens[self.pos - 1]

    def _accept(self, token: str) -> bool:
        if self._peek() == token:
            self.pos += 1
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._accept(token):
            self._fail()

    def _fail(self) -> NoReturn:
        error = f"'{self.tokens[self.pos]}'" if self.pos < len(self.tokens) else "end of input"
        raise ParseException(error_class="PARSE_SYNTAX_ERROR", message_parameters={"error": error, "hint": ""})

    def _fields(self, closing: str | None = None) -> StructType:
        fields = []
        if self._peek() != closing:
            fields.append(self._field())
            while self._accept(","):
                fields.append(self._field())
        return StructType(fields)

    def _field(self) -> StructField:
        name = self._identifier()
        self._accept(":")
        data_type = self._type()
        nullable = True
        if self._accept("not"):
            self._expect("null")
            nullable = False
        metadata = {"comment": self._string()} if self._accept("comment") else None
        return StructField(name, data_type, nullable, metadata)

    def _string(self) -> str:
        token = self._next()
        if len(token) < 2 or token[0] not in "'\"" or token[-1] != token[0]:
            self.pos -= 1
            self._fail()
        return re.sub(r"\\(.)", r"\1", token[1:-1])

    def _identifier(self) -> str:
        token = self._next()
        if not _is_ddl_name(token):
            self.pos -= 1
            self._fail()
        return token[1:-1].replace("``", "`") if token[0] == "`" else token

    def _integer(self) -> int:
        token = self._next()
        if not token.isdigit():
            self.pos -= 1
            self._fail()
        return int(token)

    def _type(self) -> DataType:
        name = self._next().lower()
        if name == "array":
            self._expect("<")
            element_type = self._type()
            self._expect(">")
            return ArrayType(element_type)
        if name == "map":
            self._expect("<")
            key_type = self._type()
            self._expect(",")
            value_type = self._type()
            self._expect(">")
            return MapType(key_type, value_type)
        if name == "struct":
            self._expect("<")
            struct = self._fields(">")
            self._expect(">")
            return struct
        if name in _DDL_DECIMAL_TYPES:
            return self._decimal()
        if name == "interval":
            return self._interval()
        if name in _DDL_ATOMIC_TYPES:
            return _DDL_ATOMIC_TYPES[name]()
        if name in _DDL_UNSUPPORTED_TYPES:
            self._expect("(")
            self._integer()
            self._expect(")")
            if not self.cast:
                msg = f"The Spark type {name} is not supported"
                raise ContributionsAcceptedError(msg)
            # Spark reads a char or varchar of any length as a string where it casts.
            return StringType()
        if not re.fullmatch(r"\w+", name):
            self.pos -= 1
            self._fail()
        raise ParseException(error_class="UNSUPPORTED_DATATYPE", message_parameters={"typeName": f'"{name.upper()}"'})

    def _decimal(self) -> DecimalType:
        if not self._accept("("):
            return DecimalType()
        start = self.pos
        precision = self._integer()
        scale = self._integer() if self._accept(",") else 0
        self._expect(")")
        if not 0 < precision <= 38 or scale > precision:
            self.pos = start
            self._fail()
        return DecimalType(precision, scale)

    def _interval(self) -> DataType:
        if self._peek() not in _DDL_DAY_TIME_FIELDS and self._peek() not in _DDL_YEAR_MONTH_FIELDS:
            # Spark 4 reads a bare interval as a calendar interval, a type the module lacks.
            msg = "The Spark type interval is not supported"
            raise ContributionsAcceptedError(msg)
        start = self._next().lower()
        fields = _DDL_DAY_TIME_FIELDS if start in _DDL_DAY_TIME_FIELDS else _DDL_YEAR_MONTH_FIELDS
        end = start
        if self._accept("to"):
            end = self._next().lower()
            if end not in fields or fields[end] <= fields[start]:
                self.pos -= 2
                self._fail()
        if fields is _DDL_YEAR_MONTH_FIELDS:
            msg = f"The Spark type interval {start} to {end} is not supported"
            raise ContributionsAcceptedError(msg)
        return DayTimeIntervalType(fields[start], fields[end])


def _parse_datatype_string(s: str, cast: bool = False) -> DataType:
    """Parses a type or schema in Spark's DDL syntax.

    A list of fields becomes a :class:`StructType`, with or without a colon between name and type
    (``a int, b string`` or ``a: int, b: string``); anything else must be a single type, such as
    ``int``, ``array<string>`` or ``struct<a:int>``. Type names are case insensitive. ``cast`` reads
    the type of a cast, where Spark allows char and varchar.
    """
    return _DDLParser(s, cast).parse()


_type_mappings: dict[type, type[DataType]] = {
    type(None): NullType,
    bool: BooleanType,
    int: LongType,
    float: DoubleType,
    str: StringType,
    bytearray: BinaryType,
    bytes: BinaryType,
    decimal.Decimal: DecimalType,
    datetime.date: DateType,
    datetime.datetime: TimestampType,
    datetime.time: TimeNTZType,
    datetime.timedelta: DayTimeIntervalType,
}


def _infer_type(obj: Any) -> DataType:  # noqa: ANN401
    if obj is None:
        return NullType()

    data_type = _type_mappings.get(type(obj))
    if data_type is DecimalType:
        # The precision and scale of each value may differ from row to row.
        return DecimalType(38, 18)
    if data_type is not None:
        return data_type()

    if isinstance(obj, dict):
        key_type: DataType = NullType()
        value_type: DataType = NullType()
        for key, value in obj.items():
            if key is not None:
                key_type = _merge_type(key_type, _infer_type(key))
            if value is not None:
                value_type = _merge_type(value_type, _infer_type(value))
        return MapType(key_type, value_type)
    if isinstance(obj, list):
        if obj:
            return ArrayType(reduce(_merge_type, (_infer_type(v) for v in obj)))
        return ArrayType(NullType())
    try:
        return _infer_schema(obj)
    except TypeError:
        raise PySparkTypeError(
            error_class="UNSUPPORTED_DATA_TYPE",
            message_parameters={"data_type": type(obj).__name__},
        ) from None


def _infer_schema(row: Any, names: list[str] | None = None) -> StructType:  # noqa: ANN401
    if isinstance(row, dict):
        items = sorted(row.items())
    elif isinstance(row, (tuple, list)):
        if hasattr(row, "__fields__"):
            items = zip(row.__fields__, tuple(row), strict=False)
        elif hasattr(row, "_fields"):
            items = zip(row._fields, tuple(row), strict=False)
        else:
            names = list(names or [])
            names.extend(f"_{i}" for i in range(len(names) + 1, len(row) + 1))
            items = zip(names, row, strict=False)
    elif hasattr(row, "__dict__"):
        items = sorted(row.__dict__.items())
    else:
        raise PySparkTypeError(
            error_class="CANNOT_INFER_SCHEMA_FOR_TYPE",
            message_parameters={"data_type": type(row).__name__},
        )

    fields = []
    for name, value in items:
        try:
            fields.append(StructField(name, _infer_type(value), True))
        except TypeError:  # noqa: PERF203
            raise PySparkTypeError(
                error_class="CANNOT_INFER_TYPE_FOR_FIELD",
                message_parameters={"field_name": name},
            ) from None
    return StructType(fields)


def _has_nulltype(dt: DataType) -> bool:
    if isinstance(dt, StructType):
        return any(_has_nulltype(f.dataType) for f in dt.fields)
    if isinstance(dt, ArrayType):
        return _has_nulltype(dt.elementType)
    if isinstance(dt, MapType):
        return _has_nulltype(dt.keyType) or _has_nulltype(dt.valueType)
    return isinstance(dt, NullType)


def _merge_type(a: DataType, b: DataType) -> DataType:
    if isinstance(a, NullType):
        return b
    if isinstance(b, NullType):
        return a
    if isinstance(a, TimestampType) and isinstance(b, TimestampNTZType):
        return a
    if isinstance(a, TimestampNTZType) and isinstance(b, TimestampType):
        return b
    if isinstance(a, AtomicType) and isinstance(b, StringType):
        return b
    if isinstance(a, StringType) and isinstance(b, AtomicType):
        return a
    if type(a) is not type(b):
        raise PySparkTypeError(
            error_class="CANNOT_MERGE_TYPE",
            message_parameters={"data_type1": type(a).__name__, "data_type2": type(b).__name__},
        )

    if isinstance(a, StructType):
        b_types = {f.name: f.dataType for f in cast("StructType", b).fields}
        fields = [StructField(f.name, _merge_type(f.dataType, b_types.get(f.name, NullType()))) for f in a.fields]
        names = {f.name for f in fields}
        fields.extend(StructField(name, data_type) for name, data_type in b_types.items() if name not in names)
        return StructType(fields)
    if isinstance(a, ArrayType):
        return ArrayType(_merge_type(a.elementType, cast("ArrayType", b).elementType))
    if isinstance(a, MapType):
        b = cast("MapType", b)
        return MapType(_merge_type(a.keyType, b.keyType), _merge_type(a.valueType, b.valueType))
    return a


_acceptable_types: dict[type[DataType], tuple[type, ...]] = {
    NullType: (type(None),),
    BooleanType: (bool,),
    ByteType: (int,),
    ShortType: (int,),
    IntegerType: (int,),
    LongType: (int,),
    UnsignedByteType: (int,),
    UnsignedShortType: (int,),
    UnsignedIntegerType: (int,),
    UnsignedLongType: (int,),
    HugeIntegerType: (int,),
    UnsignedHugeIntegerType: (int,),
    FloatType: (float,),
    DoubleType: (float,),
    DecimalType: (decimal.Decimal,),
    StringType: (str,),
    BitstringType: (str,),
    UUIDType: (uuid.UUID,),
    BinaryType: (bytearray, bytes),
    DateType: (datetime.date, datetime.datetime),
    TimeType: (datetime.time,),
    TimeNTZType: (datetime.time,),
    TimeNSType: (datetime.time,),
    TimestampType: (datetime.datetime,),
    TimestampNanosecondType: (datetime.datetime,),
    TimestampNTZType: (datetime.datetime,),
    TimestampSecondNTZType: (datetime.datetime,),
    TimestampMillisecondNTZType: (datetime.datetime,),
    TimestampNanosecondNTZType: (datetime.datetime,),
    DayTimeIntervalType: (datetime.timedelta,),
    ArrayType: (list, tuple),
    MapType: (dict,),
    StructType: (tuple, list, dict),
}

_integral_bounds: dict[type[DataType], tuple[int, int]] = {
    ByteType: (-128, 127),
    ShortType: (-32768, 32767),
    IntegerType: (-2147483648, 2147483647),
    LongType: (-9223372036854775808, 9223372036854775807),
}


def _make_type_verifier(data_type: DataType, nullable: bool = True, name: str | None = None) -> Callable[[Any], None]:
    """Returns a function that raises when a Python value does not fit ``data_type``.

    Ported from ``pyspark.sql.types._make_type_verifier`` in PySpark 3.5.9, with its error classes:
    a wrong Python type raises ``PySparkTypeError``, a None where the type is not nullable and an
    integer out of range raise ``PySparkValueError``.
    """

    def new_name(n: str) -> str:
        return f"field {n}" if name is None else f"field {n} in {name}"

    def verify_acceptable_types(obj: Any) -> None:  # noqa: ANN401
        # Subclasses are rejected, as in Spark, so a bool is not accepted as an int.
        if type(obj) not in _acceptable_types[type(data_type)]:
            raise PySparkTypeError(
                error_class="CANNOT_ACCEPT_OBJECT_IN_TYPE",
                message_parameters={
                    "data_type": str(data_type),
                    "obj_name": str(obj),
                    "obj_type": type(obj).__name__,
                },
            )

    verify_value: Callable[[Any], None]
    if isinstance(data_type, (StringType, VariantType)):

        def verify_value(obj: Any) -> None:  # noqa: ANN401
            pass

    elif type(data_type) in _integral_bounds:
        lower, upper = _integral_bounds[type(data_type)]

        def verify_value(obj: Any) -> None:  # noqa: ANN401
            verify_acceptable_types(obj)
            if obj < lower or obj > upper:
                raise PySparkValueError(
                    error_class="VALUE_OUT_OF_BOUND",
                    message_parameters={
                        "arg_name": "obj",
                        "lower_bound": str(lower),
                        "upper_bound": str(upper),
                        "actual": str(obj),
                    },
                )

    elif isinstance(data_type, ArrayType):
        verify_element = _make_type_verifier(data_type.elementType, data_type.containsNull, f"element in array {name}")

        def verify_value(obj: Any) -> None:  # noqa: ANN401
            verify_acceptable_types(obj)
            for element in obj:
                verify_element(element)

    elif isinstance(data_type, MapType):
        verify_key = _make_type_verifier(data_type.keyType, False, f"key of map {name}")
        verify_map_value = _make_type_verifier(data_type.valueType, data_type.valueContainsNull, f"value of map {name}")

        def verify_value(obj: Any) -> None:  # noqa: ANN401
            verify_acceptable_types(obj)
            for key, value in obj.items():
                verify_key(key)
                verify_map_value(value)

    elif isinstance(data_type, StructType):
        verifiers = [(f.name, _make_type_verifier(f.dataType, f.nullable, new_name(f.name))) for f in data_type.fields]

        def verify_value(obj: Any) -> None:  # noqa: ANN401
            if isinstance(obj, dict):
                for field_name, verify in verifiers:
                    verify(obj.get(field_name))
            elif isinstance(obj, (tuple, list)):
                if len(obj) != len(verifiers):
                    raise PySparkValueError(
                        error_class="LENGTH_SHOULD_BE_THE_SAME",
                        message_parameters={
                            "arg1": "obj",
                            "arg2": "fields",
                            "arg1_length": str(len(obj)),
                            "arg2_length": str(len(verifiers)),
                        },
                    )
                for value, (_, verify) in zip(obj, verifiers, strict=True):
                    verify(value)
            elif hasattr(obj, "__dict__"):
                for field_name, verify in verifiers:
                    verify(obj.__dict__.get(field_name))
            else:
                raise PySparkTypeError(
                    error_class="CANNOT_ACCEPT_OBJECT_IN_TYPE",
                    message_parameters={
                        "data_type": "StructType",
                        "obj_name": str(obj),
                        "obj_type": type(obj).__name__,
                    },
                )

    else:
        verify_value = verify_acceptable_types

    def verify(obj: Any) -> None:  # noqa: ANN401
        if obj is not None:
            verify_value(obj)
        elif not nullable:
            raise PySparkValueError(error_class="CANNOT_BE_NONE", message_parameters={"arg_name": "obj"})

    return verify


def _to_string(obj: Any) -> str | None:  # noqa: ANN401
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj).lower() if isinstance(obj, bool) else str(obj)


def _apply(convert: Callable[[Any], Any] | None, obj: Any) -> Any:  # noqa: ANN401
    return obj if convert is None else convert(obj)


def _contains_type(data_type: DataType, classes: tuple[type[DataType], ...]) -> bool:
    """Whether ``data_type`` is, or has at any depth, a type of one of ``classes``."""
    if isinstance(data_type, classes):
        return True
    if isinstance(data_type, ArrayType):
        return _contains_type(data_type.elementType, classes)
    if isinstance(data_type, MapType):
        return _contains_type(data_type.keyType, classes) or _contains_type(data_type.valueType, classes)
    if isinstance(data_type, StructType):
        return any(_contains_type(f.dataType, classes) for f in data_type.fields)
    return False


def _replace_types(data_type: DataType, replacements: Mapping[type[DataType], DataType]) -> DataType:
    """``data_type`` with each type, at any depth, whose class is in ``replacements`` replaced."""
    if isinstance(data_type, ArrayType):
        return ArrayType(_replace_types(data_type.elementType, replacements), data_type.containsNull)
    if isinstance(data_type, MapType):
        return MapType(
            _replace_types(data_type.keyType, replacements),
            _replace_types(data_type.valueType, replacements),
            data_type.valueContainsNull,
        )
    if isinstance(data_type, StructType):
        return StructType(
            [StructField(f.name, _replace_types(f.dataType, replacements), f.nullable, f.metadata) for f in data_type]
        )
    return replacements.get(type(data_type), data_type)


def _python_to_duckdb(data_type: DataType) -> Callable[[Any], Any]:
    """Returns a function that turns a Python value into the value DuckDB stores for ``data_type``.

    Rows, tuples, dicts and objects become tuples in field order, values in a string field become
    strings, and a naive datetime is a local time for a ``TimestampType`` and a wall clock time for
    a ``TimestampNTZType``, as in Spark.
    """
    if isinstance(data_type, StructType):
        names = data_type.names
        converters = [_python_to_duckdb(f.dataType) for f in data_type.fields]

        def convert_struct(obj: Any) -> tuple | None:  # noqa: ANN401
            if obj is None:
                return None
            if isinstance(obj, (tuple, list)):
                values = obj
            elif isinstance(obj, dict):
                values = [obj.get(name) for name in names]
            elif hasattr(obj, "__dict__"):
                values = [obj.__dict__.get(name) for name in names]
            else:
                raise PySparkTypeError(
                    error_class="UNSUPPORTED_DATA_TYPE",
                    message_parameters={"data_type": type(obj).__name__},
                )
            if len(values) != len(converters):
                raise PySparkValueError(
                    error_class="LENGTH_SHOULD_BE_THE_SAME",
                    message_parameters={
                        "arg1": "obj",
                        "arg2": "fields",
                        "arg1_length": str(len(values)),
                        "arg2_length": str(len(converters)),
                    },
                )
            return tuple(convert(value) for convert, value in zip(converters, values, strict=True))

        return convert_struct
    if isinstance(data_type, ArrayType):
        convert_element = _python_to_duckdb(data_type.elementType)
        return lambda obj: None if obj is None else [convert_element(v) for v in obj]
    if isinstance(data_type, MapType):
        convert_key = _python_to_duckdb(data_type.keyType)
        convert_value = _python_to_duckdb(data_type.valueType)
        return lambda obj: None if obj is None else {convert_key(k): convert_value(v) for k, v in obj.items()}
    if isinstance(data_type, StringType):
        return _to_string
    if isinstance(data_type, NullType):
        return lambda obj: None
    if isinstance(data_type, TimestampType):
        return _local_time_to_instant
    if isinstance(data_type, TimestampNTZType):
        return lambda obj: obj.replace(tzinfo=None) if isinstance(obj, datetime.datetime) else obj
    return lambda obj: obj


def _duckdb_to_python(data_type: DataType, aware: bool = False) -> Callable[[Any], Any] | None:
    """Returns a function that turns a value DuckDB returns into the value Spark returns.

    Structs become :class:`Row` objects and a ``TimestampType`` value becomes a naive local time.
    Returns None where the two already agree. ``aware`` reads instants that come as aware datetimes, as DuckDB gives
    them to a Python function, rather than as microseconds since the epoch.
    """
    if isinstance(data_type, StructType):
        names = data_type.names
        converters = [_duckdb_to_python(f.dataType, aware) for f in data_type.fields]

        def convert_struct(obj: dict | tuple | None) -> Row | None:
            if obj is None:
                return None
            values = obj.values() if isinstance(obj, dict) else obj
            return _create_row(names, [_apply(convert, v) for convert, v in zip(converters, values, strict=True)])

        return convert_struct
    if isinstance(data_type, ArrayType):
        convert_element = _duckdb_to_python(data_type.elementType, aware)
        if convert_element is None:
            return None
        return lambda obj: None if obj is None else [convert_element(v) for v in obj]
    if isinstance(data_type, MapType):
        convert_key = _duckdb_to_python(data_type.keyType, aware)
        convert_value = _duckdb_to_python(data_type.valueType, aware)
        if isinstance(data_type.keyType, (ArrayType, MapType, StructType)):
            # DuckDB returns a map whose keys Python cannot hash as a dict of a key list and a value list.
            return lambda obj: (
                None
                if obj is None
                else {
                    _apply(convert_key, k): _apply(convert_value, v)
                    for k, v in zip(obj["key"], obj["value"], strict=True)
                }
            )
        if convert_key is None and convert_value is None:
            return None
        return lambda obj: (
            None if obj is None else {_apply(convert_key, k): _apply(convert_value, v) for k, v in obj.items()}
        )
    if isinstance(data_type, TimestampType):
        return _aware_to_local_time if aware else _instant_to_local_time
    if isinstance(data_type, TimestampNanosecondType) and not aware:
        return _instant_to_utc_time
    return None


def _aware_to_local_time(obj: datetime.datetime | None) -> datetime.datetime | None:
    return None if obj is None else _instant_to_local_time(TimestampType().toInternal(obj))


def _local_time_to_instant(obj: Any) -> Any:  # noqa: ANN401
    """Reads a naive datetime as a local time the way Spark's ``TimestampType.toInternal`` does."""
    if not isinstance(obj, datetime.datetime) or obj.tzinfo is not None:
        return obj
    # A fixed offset keeps the wall clock, so a time at either end of the calendar never shifts out of range.
    fields = obj.timetuple()
    offset = datetime.timedelta(seconds=calendar.timegm(fields) - time.mktime(fields))
    return obj.replace(tzinfo=datetime.timezone(offset))


# The microseconds that stand for an infinite timestamp.
_INFINITE_MICROSECONDS = 2**63 - 1


def _infinity(microseconds: int) -> datetime.datetime | None:
    """The end of Python's calendar that DuckDB returns for an infinite timestamp, or None for a finite one."""
    if microseconds == _INFINITE_MICROSECONDS:
        return datetime.datetime.max
    if microseconds == -_INFINITE_MICROSECONDS:
        return datetime.datetime.min
    return None


def _instant_to_local_time(microseconds: int | None) -> datetime.datetime | None:
    """Turns microseconds since the epoch into a naive local time the way Spark's ``TimestampType.fromInternal`` does.

    An instant whose local time is outside Python's range raises, as in Spark.
    """
    if microseconds is None:
        return None
    return _infinity(microseconds) or TimestampType().fromInternal(microseconds)


def _instant_to_utc_time(microseconds: int | None) -> datetime.datetime | None:
    """Turns microseconds since the epoch into the aware datetime in UTC of that instant."""
    if microseconds is None:
        return None
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    return _infinity(microseconds) or epoch + datetime.timedelta(microseconds=microseconds)


def _create_row(fields: Union["Row", list[str]], values: tuple[Any, ...] | list[Any]) -> "Row":
    row = Row(*values)
    row.__fields__ = fields
    return row


class Row(tuple):
    """A row in :class:`DataFrame`.
    The fields in it can be accessed:

    * like attributes (``row.key``)
    * like dictionary values (``row[key]``)

    ``key in row`` will search through row keys.

    Row can be used to create a row object by using named arguments.
    It is not allowed to omit a named argument to represent that the value is
    None or missing. This should be explicitly set to None in this case.

    .. versionchanged:: 3.0.0
        Rows created from named arguments no longer have
        field names sorted alphabetically and will be ordered in the position as
        entered.

    Examples:
    --------
    >>> row = Row(name="Alice", age=11)
    >>> row
    Row(name='Alice', age=11)
    >>> row["name"], row["age"]
    ('Alice', 11)
    >>> row.name, row.age
    ('Alice', 11)
    >>> "name" in row
    True
    >>> "wrong_key" in row
    False

    Row also can be used to create another Row like class, then it
    could be used to create Row objects, such as

    >>> Person = Row("name", "age")
    >>> Person
    <Row('name', 'age')>
    >>> "name" in Person
    True
    >>> "wrong_key" in Person
    False
    >>> Person("Alice", 11)
    Row(name='Alice', age=11)

    This form can also be used to create rows as tuple values, i.e. with unnamed
    fields.

    >>> row1 = Row("Alice", 11)
    >>> row2 = Row(name="Alice", age=11)
    >>> row1 == row2
    True
    """  # noqa: D205, D415

    @overload
    def __new__(cls, *args: str) -> "Row": ...

    @overload
    def __new__(cls, **kwargs: Any) -> "Row": ...  # noqa: ANN401

    def __new__(cls, *args: str | None, **kwargs: Any | None) -> Self:  # noqa: D102
        if args and kwargs:
            msg = "Can not use both args and kwargs to create Row"
            raise ValueError(msg)
        if kwargs:
            # create row objects
            row = tuple.__new__(cls, list(kwargs.values()))
            row.__fields__ = list(kwargs.keys())
            return row
        # create row class or objects
        return tuple.__new__(cls, args)

    def asDict(self, recursive: bool = False) -> dict[str, Any]:
        """Return as a dict.

        Parameters
        ----------
        recursive : bool, optional
            turns the nested Rows to dict (default: False).

        Notes:
        -----
        If a row contains duplicate field names, e.g., the rows of a join
        between two :class:`DataFrame` that both have the fields of same names,
        one of the duplicate fields will be selected by ``asDict``. ``__getitem__``
        will also return one of the duplicate fields, however returned value might
        be different to ``asDict``.

        Examples:
        --------
        >>> Row(name="Alice", age=11).asDict() == {"name": "Alice", "age": 11}
        True
        >>> row = Row(key=1, value=Row(name="a", age=2))
        >>> row.asDict() == {"key": 1, "value": Row(name="a", age=2)}
        True
        >>> row.asDict(True) == {"key": 1, "value": {"name": "a", "age": 2}}
        True
        """
        if not hasattr(self, "__fields__"):
            msg = "Cannot convert a Row class into dict"
            raise TypeError(msg)

        if recursive:

            def conv(obj: Row | list | dict | object) -> list | dict | object:
                if isinstance(obj, Row):
                    return obj.asDict(True)
                elif isinstance(obj, list):
                    return [conv(o) for o in obj]
                elif isinstance(obj, dict):
                    return {k: conv(v) for k, v in obj.items()}
                else:
                    return obj

            return dict(zip(self.__fields__, (conv(o) for o in self), strict=False))
        else:
            return dict(zip(self.__fields__, self, strict=False))

    def __contains__(self, item: Any) -> bool:  # noqa: D105, ANN401
        if hasattr(self, "__fields__"):
            return item in self.__fields__
        else:
            return super().__contains__(item)

    # let object acts like class
    def __call__(self, *args: Any) -> "Row":  # noqa: ANN401
        """Create new Row object."""
        if len(args) > len(self):
            msg = f"Can not create Row with fields {self}, expected {len(self):d} values but got {args}"
            raise ValueError(msg)
        return _create_row(self, args)

    def __getitem__(self, item: Any) -> Any:  # noqa: D105, ANN401
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        try:
            # it will be slow when it has many fields,
            # but this will not be used in normal cases
            idx = self.__fields__.index(item)
            return super().__getitem__(idx)
        except IndexError:
            raise KeyError(item)  # noqa: B904
        except ValueError:
            raise ValueError(item)  # noqa: B904

    def __getattr__(self, item: str) -> Any:  # noqa: D105, ANN401
        if item.startswith("__"):
            raise AttributeError(item)
        try:
            # it will be slow when it has many fields,
            # but this will not be used in normal cases
            idx = self.__fields__.index(item)
            return self[idx]
        except IndexError:
            raise AttributeError(item)  # noqa: B904
        except ValueError:
            raise AttributeError(item)  # noqa: B904

    def __setattr__(self, key: Any, value: Any) -> None:  # noqa: D105, ANN401
        if key != "__fields__":
            msg = "Row is read-only"
            raise RuntimeError(msg)
        self.__dict__[key] = value

    def __reduce__(
        self,
    ) -> str | tuple[Any, ...]:
        """Returns a tuple so Python knows how to pickle Row."""
        if hasattr(self, "__fields__"):
            return (_create_row, (self.__fields__, tuple(self)))
        else:
            return tuple.__reduce__(self)

    def __repr__(self) -> str:
        """Printable representation of Row used in Python REPL."""
        if hasattr(self, "__fields__"):
            return "Row({})".format(", ".join(f"{k}={v!r}" for k, v in zip(self.__fields__, tuple(self), strict=False)))
        else:
            return "<Row({})>".format(", ".join(f"{field!r}" for field in self))

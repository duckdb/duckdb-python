import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace.errors import PySparkTypeError, PySparkValueError, error_condition
from spark_namespace.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DayTimeIntervalType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)


class TestDataTypeNames:
    # Expected values are copied from PySpark 4.2.0 and 3.5.9, which agree on all of them.
    @pytest.mark.parametrize(
        ("data_type", "type_name", "simple_string", "json_value", "json", "representation"),
        [
            pytest.param(
                ByteType(),
                "byte",
                "tinyint",
                "byte",
                '"byte"',
                "ByteType()",
                id="byte",
            ),
            pytest.param(
                ShortType(),
                "short",
                "smallint",
                "short",
                '"short"',
                "ShortType()",
                id="short",
            ),
            pytest.param(
                IntegerType(),
                "integer",
                "int",
                "integer",
                '"integer"',
                "IntegerType()",
                id="int",
            ),
            pytest.param(
                LongType(),
                "long",
                "bigint",
                "long",
                '"long"',
                "LongType()",
                id="long",
            ),
            pytest.param(
                FloatType(),
                "float",
                "float",
                "float",
                '"float"',
                "FloatType()",
                id="float",
            ),
            pytest.param(
                DoubleType(),
                "double",
                "double",
                "double",
                '"double"',
                "DoubleType()",
                id="double",
            ),
            pytest.param(
                DecimalType(10, 2),
                "decimal",
                "decimal(10,2)",
                "decimal(10,2)",
                '"decimal(10,2)"',
                "DecimalType(10,2)",
                id="decimal",
            ),
            pytest.param(
                DecimalType(),
                "decimal",
                "decimal(10,0)",
                "decimal(10,0)",
                '"decimal(10,0)"',
                "DecimalType(10,0)",
                id="decimal_default",
            ),
            pytest.param(
                StringType(),
                "string",
                "string",
                "string",
                '"string"',
                "StringType()",
                id="string",
            ),
            pytest.param(
                BooleanType(),
                "boolean",
                "boolean",
                "boolean",
                '"boolean"',
                "BooleanType()",
                id="bool",
            ),
            pytest.param(
                DateType(),
                "date",
                "date",
                "date",
                '"date"',
                "DateType()",
                id="date",
            ),
            pytest.param(
                TimestampType(),
                "timestamp",
                "timestamp",
                "timestamp",
                '"timestamp"',
                "TimestampType()",
                id="ts",
            ),
            pytest.param(
                TimestampNTZType(),
                "timestamp_ntz",
                "timestamp_ntz",
                "timestamp_ntz",
                '"timestamp_ntz"',
                "TimestampNTZType()",
                id="ntz",
            ),
            pytest.param(
                BinaryType(),
                "binary",
                "binary",
                "binary",
                '"binary"',
                "BinaryType()",
                id="binary",
            ),
            pytest.param(
                NullType(),
                "void",
                "void",
                "void",
                '"void"',
                "NullType()",
                id="null",
            ),
            pytest.param(
                DayTimeIntervalType(),
                "daytimeinterval",
                "interval day to second",
                "interval day to second",
                '"interval day to second"',
                "DayTimeIntervalType(0, 3)",
                id="interval",
            ),
            pytest.param(
                DayTimeIntervalType(0, 0),
                "daytimeinterval",
                "interval day",
                "interval day",
                '"interval day"',
                "DayTimeIntervalType(0, 0)",
                id="interval_day",
            ),
            pytest.param(
                DayTimeIntervalType(1, 2),
                "daytimeinterval",
                "interval hour to minute",
                "interval hour to minute",
                '"interval hour to minute"',
                "DayTimeIntervalType(1, 2)",
                id="interval_h_m",
            ),
            pytest.param(
                ArrayType(IntegerType()),
                "array",
                "array<int>",
                {"type": "array", "elementType": "integer", "containsNull": True},
                '{"containsNull":true,"elementType":"integer","type":"array"}',
                "ArrayType(IntegerType(), True)",
                id="array",
            ),
            pytest.param(
                ArrayType(IntegerType(), False),
                "array",
                "array<int>",
                {"type": "array", "elementType": "integer", "containsNull": False},
                '{"containsNull":false,"elementType":"integer","type":"array"}',
                "ArrayType(IntegerType(), False)",
                id="array_nn",
            ),
            pytest.param(
                MapType(StringType(), LongType()),
                "map",
                "map<string,bigint>",
                {"type": "map", "keyType": "string", "valueType": "long", "valueContainsNull": True},
                '{"keyType":"string","type":"map","valueContainsNull":true,"valueType":"long"}',
                "MapType(StringType(), LongType(), True)",
                id="map",
            ),
            pytest.param(
                StructType([StructField("a", IntegerType()), StructField("b", StringType(), False)]),
                "struct",
                "struct<a:int,b:string>",
                {
                    "type": "struct",
                    "fields": [
                        {"name": "a", "type": "integer", "nullable": True, "metadata": {}},
                        {"name": "b", "type": "string", "nullable": False, "metadata": {}},
                    ],
                },
                '{"fields":[{"metadata":{},"name":"a","nullable":true,"type":"integer"},{"metadata":{},"name":"b","nullable":false,"type":"string"}],"type":"struct"}',
                "StructType([StructField('a', IntegerType(), True), StructField('b', StringType(), False)])",
                id="struct",
            ),
        ],
    )
    def test_data_type_names(self, data_type, type_name, simple_string, json_value, json, representation):
        assert data_type.typeName() == type_name
        assert data_type.simpleString() == simple_string
        assert data_type.jsonValue() == json_value
        assert data_type.json() == json
        assert repr(data_type) == representation

    def test_struct_field_names(self):
        field = StructField("x", LongType(), False)
        assert field.simpleString() == "x:bigint"
        assert field.jsonValue() == {"name": "x", "type": "long", "nullable": False, "metadata": {}}
        assert field.json() == '{"metadata":{},"name":"x","nullable":false,"type":"long"}'
        assert repr(field) == "StructField('x', LongType(), False)"
        with pytest.raises(PySparkTypeError, match="INVALID_TYPENAME_CALL") as error:
            field.typeName()
        assert isinstance(error.value, TypeError)
        assert error_condition(error.value) == "INVALID_TYPENAME_CALL"

    @pytest.mark.parametrize(
        ("type_name", "expected"),
        [
            ("string", StringType()),
            ("integer", IntegerType()),
            ("long", LongType()),
            ("byte", ByteType()),
            ("timestamp", TimestampType()),
            ("timestamp_ntz", TimestampNTZType()),
            ("decimal", DecimalType()),
            ("decimal(5,2)", DecimalType(5, 2)),
            ("interval day to second", DayTimeIntervalType()),
        ],
    )
    def test_struct_type_add_reads_type_names(self, type_name, expected):
        schema = StructType().add("f", type_name, False)
        assert schema == StructType([StructField("f", expected, False)])

    @pytest.mark.parametrize("type_name", ["int", "bigint", "array<int>", "struct<a:int>", "bogus"])
    def test_struct_type_add_rejects_other_strings(self, type_name):
        with pytest.raises(PySparkValueError, match="CANNOT_PARSE_DATATYPE"):
            StructType().add("f", type_name)

    def test_nested_json_value(self):
        schema = StructType(
            [
                StructField("a", ArrayType(StructType([StructField("x", DecimalType(5, 2), False)]), False)),
                StructField("m", MapType(StringType(), ArrayType(LongType()), False)),
            ]
        )
        assert schema.jsonValue() == {
            "type": "struct",
            "fields": [
                {
                    "name": "a",
                    "type": {
                        "type": "array",
                        "elementType": {
                            "type": "struct",
                            "fields": [{"name": "x", "type": "decimal(5,2)", "nullable": False, "metadata": {}}],
                        },
                        "containsNull": False,
                    },
                    "nullable": True,
                    "metadata": {},
                },
                {
                    "name": "m",
                    "type": {
                        "type": "map",
                        "keyType": "string",
                        "valueType": {"type": "array", "elementType": "long", "containsNull": True},
                        "valueContainsNull": False,
                    },
                    "nullable": True,
                    "metadata": {},
                },
            ],
        }
        assert schema.simpleString() == "struct<a:array<struct<x:decimal(5,2)>>,m:map<string,array<bigint>>>"

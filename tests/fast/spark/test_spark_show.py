import datetime
from datetime import timedelta as td
from decimal import Decimal

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import PySparkTypeError
from spark_namespace.sql.functions import col
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
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
)

if USE_ACTUAL_SPARK:
    import pyspark

    # PySpark 3.5 ends the "only showing top" and "(0 rows)" footers with an extra newline.
    FOOTER_END = "\n\n" if pyspark.__version__ < "4" else "\n"
else:
    FOOTER_END = "\n"


def F(name, data_type):
    return StructField(name, data_type, True)


def S(*fields):
    return StructType(list(fields))


def printed(lines):
    if lines[-1].startswith("only showing top") or lines[-1] == "(0 rows)":
        return "\n".join(lines) + FOOTER_END
    return "\n".join(lines) + "\n"


# Expected output is copied from PySpark 4.2.0 running the same calls.
SHOW_CASES = [
    pytest.param(
        [(1, 2, 3, 4), (-100, -32768, -2147483648, -9223372036854775808), (None, None, None, None)],
        S(F("b", ByteType()), F("s", ShortType()), F("i", IntegerType()), F("l", LongType())),
        lambda df: df.show(),
        (
            "+----+------+-----------+--------------------+",
            "|   b|     s|          i|                   l|",
            "+----+------+-----------+--------------------+",
            "|   1|     2|          3|                   4|",
            "|-100|-32768|-2147483648|-9223372036854775808|",
            "|NULL|  NULL|       NULL|                NULL|",
            "+----+------+-----------+--------------------+",
            "",
        ),
        id="ints",
    ),
    pytest.param(
        [
            (Decimal("1.50"), Decimal("1.5"), Decimal("12345")),
            (Decimal("-0.05"), Decimal("0"), Decimal("0")),
            (Decimal("0"), Decimal("-123.000000000000000001"), None),
        ],
        S(F("d1", DecimalType(10, 2)), F("d2", DecimalType(38, 18)), F("d3", DecimalType(5, 0))),
        lambda df: df.show(),
        (
            "+-----+--------------------+-----+",
            "|   d1|                  d2|   d3|",
            "+-----+--------------------+-----+",
            "| 1.50|1.500000000000000000|12345|",
            "|-0.05|0.000000000000000000|    0|",
            "| 0.00|-123.000000000000...| NULL|",
            "+-----+--------------------+-----+",
            "",
        ),
        id="decimals",
    ),
    pytest.param(
        [(Decimal("0E-10"),), (Decimal("0.0000000001"),)],
        S(F("d", DecimalType(10, 10))),
        lambda df: df.show(),
        (
            "+------------+",
            "|           d|",
            "+------------+",
            "|0.0000000000|",
            "|0.0000000001|",
            "+------------+",
            "",
        ),
        id="decimal_small_scale",
    ),
    pytest.param(
        [("",), ("a",), ("abcdefghijklmnopqrstuvwxyz",), ("a\nb\tc",), ("日本語",), ("é",), (None,)],
        S(F("s", StringType())),
        lambda df: df.show(),
        (
            "+--------------------+",
            "|                   s|",
            "+--------------------+",
            "|                    |",
            "|                   a|",
            "|abcdefghijklmnopq...|",
            "|             a\\nb\\tc|",
            "|              日本語|",
            "|                   é|",
            "|                NULL|",
            "+--------------------+",
            "",
        ),
        id="strings",
    ),
    pytest.param(
        [("日本語日本語",), ("ab",)],
        S(F("s", StringType())),
        lambda df: df.show(),
        (
            "+------------+",
            "|           s|",
            "+------------+",
            "|日本語日本語|",
            "|          ab|",
            "+------------+",
            "",
        ),
        id="cjk_width",
    ),
    pytest.param(
        [(True,), (False,), (None,)],
        S(F("b", BooleanType())),
        lambda df: df.show(),
        (
            "+-----+",
            "|    b|",
            "+-----+",
            "| true|",
            "|false|",
            "| NULL|",
            "+-----+",
            "",
        ),
        id="booleans",
    ),
    pytest.param(
        [(datetime.date(2024, 1, 15),), (datetime.date(1, 1, 1),), (datetime.date(9999, 12, 31),), (None,)],
        S(F("d", DateType())),
        lambda df: df.show(),
        (
            "+----------+",
            "|         d|",
            "+----------+",
            "|2024-01-15|",
            "|0001-01-01|",
            "|9999-12-31|",
            "|      NULL|",
            "+----------+",
            "",
        ),
        id="dates",
    ),
    pytest.param(
        [(datetime.datetime(2024, 1, 15, 10, 30),), (datetime.datetime(2024, 1, 15, 10, 30, 0, 123),), (None,)],
        S(F("t", TimestampNTZType())),
        lambda df: df.show(),
        (
            "+--------------------+",
            "|                   t|",
            "+--------------------+",
            "| 2024-01-15 10:30:00|",
            "|2024-01-15 10:30:...|",
            "|                NULL|",
            "+--------------------+",
            "",
        ),
        id="timestamps_ntz",
    ),
    pytest.param(
        [(bytearray(b"hello"),), (bytearray(b""),), (bytearray(b"\x00\xff\x10"),), (None,)],
        S(F("b", BinaryType())),
        lambda df: df.show(),
        (
            "+----------------+",
            "|               b|",
            "+----------------+",
            "|[68 65 6C 6C 6F]|",
            "|              []|",
            "|      [00 FF 10]|",
            "|            NULL|",
            "+----------------+",
            "",
        ),
        id="binary",
    ),
    pytest.param(
        [([1, 2, 3], ["a", "b"], [[1, 2], [3]]), ([], [], []), ([1, None], [None], [None, [None]]), (None, None, None)],
        S(
            F("a", ArrayType(IntegerType())),
            F("s", ArrayType(StringType())),
            F("n", ArrayType(ArrayType(IntegerType()))),
        ),
        lambda df: df.show(),
        (
            "+---------+------+--------------+",
            "|        a|     s|             n|",
            "+---------+------+--------------+",
            "|[1, 2, 3]|[a, b]| [[1, 2], [3]]|",
            "|       []|    []|            []|",
            "|[1, NULL]|[NULL]|[NULL, [NULL]]|",
            "|     NULL|  NULL|          NULL|",
            "+---------+------+--------------+",
            "",
        ),
        id="arrays",
    ),
    pytest.param(
        [([1.0, 2.5, 1e10], [Decimal("1.50")], [datetime.date(2024, 1, 2)], [True, False], [bytearray(b"hi")])],
        S(
            F("d", ArrayType(DoubleType())),
            F("m", ArrayType(DecimalType(10, 2))),
            F("t", ArrayType(DateType())),
            F("b", ArrayType(BooleanType())),
            F("y", ArrayType(BinaryType())),
        ),
        lambda df: df.show(),
        (
            "+------------------+------+------------+-------------+---------+",
            "|                 d|     m|           t|            b|        y|",
            "+------------------+------+------------+-------------+---------+",
            "|[1.0, 2.5, 1.0E10]|[1.50]|[2024-01-02]|[true, false]|[[68 69]]|",
            "+------------------+------+------------+-------------+---------+",
            "",
        ),
        id="arrays_of_other",
    ),
    pytest.param(
        [({"a": 1, "b": 2},), ({},), ({"a": None},), (None,)],
        S(F("m", MapType(StringType(), IntegerType()))),
        lambda df: df.show(),
        (
            "+----------------+",
            "|               m|",
            "+----------------+",
            "|{a -> 1, b -> 2}|",
            "|              {}|",
            "|     {a -> NULL}|",
            "|            NULL|",
            "+----------------+",
            "",
        ),
        id="maps",
    ),
    pytest.param(
        [({1: ["x", "y"]}, {"k": {"a": 1}})],
        S(
            F("m", MapType(IntegerType(), ArrayType(StringType()))),
            F("n", MapType(StringType(), MapType(StringType(), IntegerType()))),
        ),
        lambda df: df.show(),
        (
            "+-------------+---------------+",
            "|            m|              n|",
            "+-------------+---------------+",
            "|{1 -> [x, y]}|{k -> {a -> 1}}|",
            "+-------------+---------------+",
            "",
        ),
        id="map_nested",
    ),
    pytest.param(
        [
            ((1, "x"), (1, (2, None)), ([1, 2], {"a": 1})),
            ((None, None), (None, None), (None, None)),
            (None, None, None),
        ],
        S(
            F("s", S(F("a", IntegerType()), F("b", StringType()))),
            F("n", S(F("a", IntegerType()), F("b", S(F("c", IntegerType()), F("d", StringType()))))),
            F("c", S(F("a", ArrayType(IntegerType())), F("m", MapType(StringType(), IntegerType())))),
        ),
        lambda df: df.show(),
        (
            "+------------+--------------+------------------+",
            "|           s|             n|                 c|",
            "+------------+--------------+------------------+",
            "|      {1, x}|{1, {2, NULL}}|{[1, 2], {a -> 1}}|",
            "|{NULL, NULL}|  {NULL, NULL}|      {NULL, NULL}|",
            "|        NULL|          NULL|              NULL|",
            "+------------+--------------+------------------+",
            "",
        ),
        id="structs",
    ),
    pytest.param(
        [([(1, "a"), (2, None)],), ([None],)],
        S(F("a", ArrayType(S(F("x", IntegerType()), F("y", StringType()))))),
        lambda df: df.show(),
        (
            "+-------------------+",
            "|                  a|",
            "+-------------------+",
            "|[{1, a}, {2, NULL}]|",
            "|             [NULL]|",
            "+-------------------+",
            "",
        ),
        id="array_of_struct",
    ),
    pytest.param(
        [((bytearray(b"hi"), 1.0),)],
        S(F("s", S(F("b", BinaryType()), F("d", DoubleType())))),
        lambda df: df.show(),
        (
            "+--------------+",
            "|             s|",
            "+--------------+",
            "|{[68 69], 1.0}|",
            "+--------------+",
            "",
        ),
        id="struct_binary",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(),
        (
            "+---------------------+---+",
            "|long_column_name_here|  n|",
            "+---------------------+---+",
            "| abcdefghijklmnopq...|  1|",
            "|                short| 22|",
            "| exactly twenty chars|333|",
            "+---------------------+---+",
            "",
        ),
        id="truncate_default",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(truncate=True),
        (
            "+---------------------+---+",
            "|long_column_name_here|  n|",
            "+---------------------+---+",
            "| abcdefghijklmnopq...|  1|",
            "|                short| 22|",
            "| exactly twenty chars|333|",
            "+---------------------+---+",
            "",
        ),
        id="truncate_true",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(truncate=False),
        (
            "+--------------------------+---+",
            "|long_column_name_here     |n  |",
            "+--------------------------+---+",
            "|abcdefghijklmnopqrstuvwxyz|1  |",
            "|short                     |22 |",
            "|exactly twenty chars      |333|",
            "+--------------------------+---+",
            "",
        ),
        id="truncate_false",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(truncate=0),
        (
            "+--------------------------+---+",
            "|long_column_name_here     |n  |",
            "+--------------------------+---+",
            "|abcdefghijklmnopqrstuvwxyz|1  |",
            "|short                     |22 |",
            "|exactly twenty chars      |333|",
            "+--------------------------+---+",
            "",
        ),
        id="truncate_0",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(truncate=5),
        (
            "+---------------------+---+",
            "|long_column_name_here|  n|",
            "+---------------------+---+",
            "|                ab...|  1|",
            "|                short| 22|",
            "|                ex...|333|",
            "+---------------------+---+",
            "",
        ),
        id="truncate_5",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(truncate=4),
        (
            "+---------------------+---+",
            "|long_column_name_here|  n|",
            "+---------------------+---+",
            "|                 a...|  1|",
            "|                 s...| 22|",
            "|                 e...|333|",
            "+---------------------+---+",
            "",
        ),
        id="truncate_4",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(truncate=3),
        (
            "+---------------------+---+",
            "|long_column_name_here|  n|",
            "+---------------------+---+",
            "|                  abc|  1|",
            "|                  sho| 22|",
            "|                  exa|333|",
            "+---------------------+---+",
            "",
        ),
        id="truncate_3",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(truncate=1),
        (
            "+---------------------+---+",
            "|long_column_name_here|  n|",
            "+---------------------+---+",
            "|                    a|  1|",
            "|                    s|  2|",
            "|                    e|  3|",
            "+---------------------+---+",
            "",
        ),
        id="truncate_1",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(truncate=-1),
        (
            "+--------------------------+---+",
            "|long_column_name_here     |n  |",
            "+--------------------------+---+",
            "|abcdefghijklmnopqrstuvwxyz|1  |",
            "|short                     |22 |",
            "|exactly twenty chars      |333|",
            "+--------------------------+---+",
            "",
        ),
        id="truncate_neg",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(truncate=5.7),
        (
            "+---------------------+---+",
            "|long_column_name_here|  n|",
            "+---------------------+---+",
            "|                ab...|  1|",
            "|                short| 22|",
            "|                ex...|333|",
            "+---------------------+---+",
            "",
        ),
        id="truncate_float",
    ),
    pytest.param(
        [("abcdefghijklmnopqrstuvwxyz", 1), ("short", 22), ("exactly twenty chars", 333)],
        S(F("long_column_name_here", StringType()), F("n", IntegerType())),
        lambda df: df.show(truncate="5"),
        (
            "+---------------------+---+",
            "|long_column_name_here|  n|",
            "+---------------------+---+",
            "|                ab...|  1|",
            "|                short| 22|",
            "|                ex...|333|",
            "+---------------------+---+",
            "",
        ),
        id="truncate_str_digit",
    ),
    pytest.param(
        [(["abcdefghij", "klmnopqrst"],)],
        S(F("a", ArrayType(StringType()))),
        lambda df: df.show(),
        (
            "+--------------------+",
            "|                   a|",
            "+--------------------+",
            "|[abcdefghij, klmn...|",
            "+--------------------+",
            "",
        ),
        id="truncate_array",
    ),
    pytest.param(
        [(i, f"r{i}") for i in range(25)],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(),
        (
            "+---+---+",
            "| id|  v|",
            "+---+---+",
            "|  0| r0|",
            "|  1| r1|",
            "|  2| r2|",
            "|  3| r3|",
            "|  4| r4|",
            "|  5| r5|",
            "|  6| r6|",
            "|  7| r7|",
            "|  8| r8|",
            "|  9| r9|",
            "| 10|r10|",
            "| 11|r11|",
            "| 12|r12|",
            "| 13|r13|",
            "| 14|r14|",
            "| 15|r15|",
            "| 16|r16|",
            "| 17|r17|",
            "| 18|r18|",
            "| 19|r19|",
            "+---+---+",
            "only showing top 20 rows",
        ),
        id="n_default_25_rows",
    ),
    pytest.param(
        [(i, f"r{i}") for i in range(5)],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(n=2),
        (
            "+---+---+",
            "| id|  v|",
            "+---+---+",
            "|  0| r0|",
            "|  1| r1|",
            "+---+---+",
            "only showing top 2 rows",
        ),
        id="n_2",
    ),
    pytest.param(
        [(i, f"r{i}") for i in range(5)],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(n=1),
        (
            "+---+---+",
            "| id|  v|",
            "+---+---+",
            "|  0| r0|",
            "+---+---+",
            "only showing top 1 row",
        ),
        id="n_1",
    ),
    pytest.param(
        [(i, f"r{i}") for i in range(5)],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(n=0),
        (
            "+---+---+",
            "| id|  v|",
            "+---+---+",
            "+---+---+",
            "only showing top 0 rows",
        ),
        id="n_0",
    ),
    pytest.param(
        [(i, f"r{i}") for i in range(5)],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(n=-1),
        (
            "+---+---+",
            "| id|  v|",
            "+---+---+",
            "+---+---+",
            "only showing top 0 rows",
        ),
        id="n_neg",
    ),
    pytest.param(
        [(i, f"r{i}") for i in range(5)],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(n=5),
        (
            "+---+---+",
            "| id|  v|",
            "+---+---+",
            "|  0| r0|",
            "|  1| r1|",
            "|  2| r2|",
            "|  3| r3|",
            "|  4| r4|",
            "+---+---+",
            "",
        ),
        id="n_exact",
    ),
    pytest.param(
        [(i, f"r{i}") for i in range(5)],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(n=10),
        (
            "+---+---+",
            "| id|  v|",
            "+---+---+",
            "|  0| r0|",
            "|  1| r1|",
            "|  2| r2|",
            "|  3| r3|",
            "|  4| r4|",
            "+---+---+",
            "",
        ),
        id="n_more",
    ),
    pytest.param(
        [(i, f"r{i}") for i in range(5)],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(3, False),
        (
            "+---+---+",
            "|id |v  |",
            "+---+---+",
            "|0  |r0 |",
            "|1  |r1 |",
            "|2  |r2 |",
            "+---+---+",
            "only showing top 3 rows",
        ),
        id="n_positional",
    ),
    pytest.param(
        [],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(),
        (
            "+---+---+",
            "| id|  v|",
            "+---+---+",
            "+---+---+",
            "",
        ),
        id="empty",
    ),
    pytest.param(
        [(0, "r0"), (1, "r1"), (2, "r2"), (None, "abcdefghijklmnopqrstuvwxyz")],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(vertical=True),
        (
            "-RECORD 0-------------------",
            " id  | 0                    ",
            " v   | r0                   ",
            "-RECORD 1-------------------",
            " id  | 1                    ",
            " v   | r1                   ",
            "-RECORD 2-------------------",
            " id  | 2                    ",
            " v   | r2                   ",
            "-RECORD 3-------------------",
            " id  | NULL                 ",
            " v   | abcdefghijklmnopq... ",
            "",
        ),
        id="vertical",
    ),
    pytest.param(
        [(i, f"r{i}") for i in range(5)],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(n=2, vertical=True),
        (
            "-RECORD 0--",
            " id  | 0   ",
            " v   | r0  ",
            "-RECORD 1--",
            " id  | 1   ",
            " v   | r1  ",
            "only showing top 2 rows",
        ),
        id="vertical_n",
    ),
    pytest.param(
        [(i, f"r{i}") for i in range(5)],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(n=1, vertical=True),
        (
            "-RECORD 0--",
            " id  | 0   ",
            " v   | r0  ",
            "only showing top 1 row",
        ),
        id="vertical_n1",
    ),
    pytest.param(
        [],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(vertical=True),
        ("(0 rows)",),
        id="vertical_empty",
    ),
    pytest.param(
        [(1, "abcdefghijklmnopqrstuvwxyz")],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(vertical=True, truncate=False),
        (
            "-RECORD 0-------------------------",
            " id  | 1                          ",
            " v   | abcdefghijklmnopqrstuvwxyz ",
            "",
        ),
        id="vertical_truncate_false",
    ),
    pytest.param(
        [(1, "abcdefghijklmnopqrstuvwxyz")],
        S(F("id", LongType()), F("v", StringType())),
        lambda df: df.show(vertical=True, truncate=3),
        (
            "-RECORD 0--",
            " id  | 1   ",
            " v   | abc ",
            "",
        ),
        id="vertical_truncate_3",
    ),
    pytest.param(
        [(1, "x")],
        S(F("a_rather_long_field_name", LongType()), F("v", StringType())),
        lambda df: df.show(vertical=True),
        (
            "-RECORD 0-----------------------",
            " a_rather_long_field_name | 1   ",
            " v                        | x   ",
            "",
        ),
        id="vertical_long_name",
    ),
    pytest.param(
        [(1,)],
        S(F("a\nb", LongType())),
        lambda df: df.show(),
        (
            "+----+",
            "|a\\nb|",
            "+----+",
            "|   1|",
            "+----+",
            "",
        ),
        id="header_escape",
    ),
    pytest.param(
        [
            (td(days=1, hours=2, minutes=3, seconds=4),),
            (td(seconds=4.5),),
            (td(0),),
            (-td(days=1, seconds=1),),
            (td(days=-1, hours=23),),
            (td(microseconds=1),),
            (td(days=12345),),
            (None,),
        ],
        S(F("i", DayTimeIntervalType())),
        lambda df: df.show(truncate=False),
        (
            "+------------------------------------------+",
            "|i                                         |",
            "+------------------------------------------+",
            "|INTERVAL '1 02:03:04' DAY TO SECOND       |",
            "|INTERVAL '0 00:00:04.5' DAY TO SECOND     |",
            "|INTERVAL '0 00:00:00' DAY TO SECOND       |",
            "|INTERVAL '-1 00:00:01' DAY TO SECOND      |",
            "|INTERVAL '-0 01:00:00' DAY TO SECOND      |",
            "|INTERVAL '0 00:00:00.000001' DAY TO SECOND|",
            "|INTERVAL '12345 00:00:00' DAY TO SECOND   |",
            "|NULL                                      |",
            "+------------------------------------------+",
            "",
        ),
        id="intervals_untruncated",
    ),
    pytest.param(
        [
            (td(days=1, hours=2, minutes=3, seconds=4),),
            (td(seconds=4.5),),
            (td(0),),
            (-td(days=1, seconds=1),),
            (td(days=-1, hours=23),),
            (td(microseconds=1),),
            (td(days=12345),),
            (None,),
        ],
        S(F("i", DayTimeIntervalType())),
        lambda df: df.show(),
        (
            "+--------------------+",
            "|                   i|",
            "+--------------------+",
            "|INTERVAL '1 02:03...|",
            "|INTERVAL '0 00:00...|",
            "|INTERVAL '0 00:00...|",
            "|INTERVAL '-1 00:0...|",
            "|INTERVAL '-0 01:0...|",
            "|INTERVAL '0 00:00...|",
            "|INTERVAL '12345 0...|",
            "|                NULL|",
            "+--------------------+",
            "",
        ),
        id="intervals",
    ),
    pytest.param(
        [("a\rb\fc\bd\x0be\x07f\\g",), ("q\"uote's",)],
        S(F("s", StringType())),
        lambda df: df.show(truncate=False),
        (
            "+------------------+",
            "|s                 |",
            "+------------------+",
            "|a\\rb\\fc\\bd\\ve\\af\\g|",
            "|q\"uote's          |",
            "+------------------+",
            "",
        ),
        id="escapes_untruncated",
    ),
    pytest.param(
        [("a\rb\fc\bd\x0be\x07f\\g",), ("q\"uote's",)],
        S(F("s", StringType())),
        lambda df: df.show(),
        (
            "+------------------+",
            "|                 s|",
            "+------------------+",
            "|a\\rb\\fc\\bd\\ve\\af\\g|",
            "|          q\"uote's|",
            "+------------------+",
            "",
        ),
        id="escapes",
    ),
    pytest.param(
        [("\ud55c\uae00",), ("\uff21\uff22",), ("\uff71\uff72",), ("\U0001f600x",), ("\u3000",), ("\u00e9\u0301",)],
        S(F("s", StringType())),
        lambda df: df.show(truncate=False),
        (
            "+----+",
            "|s   |",
            "+----+",
            "|한글|",
            "|\uff21\uff22|",
            "|\uff71\uff72  |",
            "|😀x |",
            "|\u3000  |",
            "|é́  |",
            "+----+",
            "",
        ),
        id="widths_untruncated",
    ),
    pytest.param(
        [("\ud55c\uae00",), ("\uff21\uff22",), ("\uff71\uff72",), ("\U0001f600x",), ("\u3000",), ("\u00e9\u0301",)],
        S(F("s", StringType())),
        lambda df: df.show(),
        (
            "+----+",
            "|   s|",
            "+----+",
            "|한글|",
            "|\uff21\uff22|",
            "|  \uff71\uff72|",
            "| 😀x|",
            "|  \u3000|",
            "|  é́|",
            "+----+",
            "",
        ),
        id="widths",
    ),
    pytest.param(
        [(1,)],
        S(F("日本", LongType())),
        lambda df: df.show(truncate=False),
        (
            "+----+",
            "|日本|",
            "+----+",
            "|1   |",
            "+----+",
            "",
        ),
        id="cjk_header_untruncated",
    ),
    pytest.param(
        [(1,)],
        S(F("日本", LongType())),
        lambda df: df.show(),
        (
            "+----+",
            "|日本|",
            "+----+",
            "|   1|",
            "+----+",
            "",
        ),
        id="cjk_header",
    ),
    pytest.param(
        [({"a": (1, None)},)],
        S(F("m", MapType(StringType(), S(F("x", IntegerType()), F("y", StringType()))))),
        lambda df: df.show(truncate=False),
        (
            "+----------------+",
            "|m               |",
            "+----------------+",
            "|{a -> {1, NULL}}|",
            "+----------------+",
            "",
        ),
        id="map_null_key_struct_untruncated",
    ),
    pytest.param(
        [({"a": (1, None)},)],
        S(F("m", MapType(StringType(), S(F("x", IntegerType()), F("y", StringType()))))),
        lambda df: df.show(),
        (
            "+----------------+",
            "|               m|",
            "+----------------+",
            "|{a -> {1, NULL}}|",
            "+----------------+",
            "",
        ),
        id="map_null_key_struct",
    ),
    pytest.param(
        [(["a,b", "", None, "x\ny"], {"k k": "v, v"})],
        S(F("a", ArrayType(StringType())), F("m", MapType(StringType(), StringType()))),
        lambda df: df.show(truncate=False),
        (
            "+-------------------+-------------+",
            "|a                  |m            |",
            "+-------------------+-------------+",
            "|[a,b, , NULL, x\\ny]|{k k -> v, v}|",
            "+-------------------+-------------+",
            "",
        ),
        id="nested_strings_untruncated",
    ),
    pytest.param(
        [(["a,b", "", None, "x\ny"], {"k k": "v, v"})],
        S(F("a", ArrayType(StringType())), F("m", MapType(StringType(), StringType()))),
        lambda df: df.show(),
        (
            "+-------------------+-------------+",
            "|                  a|            m|",
            "+-------------------+-------------+",
            "|[a,b, , NULL, x\\ny]|{k k -> v, v}|",
            "+-------------------+-------------+",
            "",
        ),
        id="nested_strings",
    ),
    pytest.param(
        [
            (x,)
            for x in [
                1e-3,
                9.999e-4,
                1e7 - 0.5,
                1e7,
                5e-324,
                1.7976931348623157e308,
                2.0**53,
                0.30000000000000004,
                1 / 3,
                100.5,
                1e21,
                1e-7,
                123e-20,
            ]
        ],
        S(F("d", DoubleType())),
        lambda df: df.show(truncate=False),
        (
            "+----------------------+",
            "|d                     |",
            "+----------------------+",
            "|0.001                 |",
            "|9.999E-4              |",
            "|9999999.5             |",
            "|1.0E7                 |",
            "|4.9E-324              |",
            "|1.7976931348623157E308|",
            "|9.007199254740992E15  |",
            "|0.30000000000000004   |",
            "|0.3333333333333333    |",
            "|100.5                 |",
            "|1.0E21                |",
            "|1.0E-7                |",
            "|1.23E-18              |",
            "+----------------------+",
            "",
        ),
        id="float_edges_untruncated",
    ),
    pytest.param(
        [
            (x,)
            for x in [
                1e-3,
                9.999e-4,
                1e7 - 0.5,
                1e7,
                5e-324,
                1.7976931348623157e308,
                2.0**53,
                0.30000000000000004,
                1 / 3,
                100.5,
                1e21,
                1e-7,
                123e-20,
            ]
        ],
        S(F("d", DoubleType())),
        lambda df: df.show(),
        (
            "+--------------------+",
            "|                   d|",
            "+--------------------+",
            "|               0.001|",
            "|            9.999E-4|",
            "|           9999999.5|",
            "|               1.0E7|",
            "|            4.9E-324|",
            "|1.797693134862315...|",
            "|9.007199254740992E15|",
            "| 0.30000000000000004|",
            "|  0.3333333333333333|",
            "|               100.5|",
            "|              1.0E21|",
            "|              1.0E-7|",
            "|            1.23E-18|",
            "+--------------------+",
            "",
        ),
        id="float_edges",
    ),
    pytest.param(
        [(x,) for x in [0.3, 1 / 3, 16777216.0, 1e-45, 3.4028235e38, 1.17549435e-38, 9999999.0, 0.001, 2.5]],
        S(F("f", FloatType())),
        lambda df: df.show(truncate=False),
        (
            "+-------------+",
            "|f            |",
            "+-------------+",
            "|0.3          |",
            "|0.33333334   |",
            "|1.6777216E7  |",
            "|1.4E-45      |",
            "|3.4028235E38 |",
            "|1.1754944E-38|",
            "|9999999.0    |",
            "|0.001        |",
            "|2.5          |",
            "+-------------+",
            "",
        ),
        id="float32_edges_untruncated",
    ),
    pytest.param(
        [(x,) for x in [0.3, 1 / 3, 16777216.0, 1e-45, 3.4028235e38, 1.17549435e-38, 9999999.0, 0.001, 2.5]],
        S(F("f", FloatType())),
        lambda df: df.show(),
        (
            "+-------------+",
            "|            f|",
            "+-------------+",
            "|          0.3|",
            "|   0.33333334|",
            "|  1.6777216E7|",
            "|      1.4E-45|",
            "| 3.4028235E38|",
            "|1.1754944E-38|",
            "|    9999999.0|",
            "|        0.001|",
            "|          2.5|",
            "+-------------+",
            "",
        ),
        id="float32_edges",
    ),
    pytest.param(
        [
            (x,)
            for x in [
                1.0,
                0.1,
                100.0,
                1e7,
                9999999.0,
                12345678.9,
                1e10,
                1e20,
                0.001,
                0.0001,
                1.5e-5,
                -0.0,
                123456789.123,
                1.23456789012e-10,
                float("inf"),
                float("-inf"),
                None,
            ]
        ],
        S(F("d", DoubleType())),
        lambda df: df.show(truncate=False),
        (
            "+-----------------+",
            "|d                |",
            "+-----------------+",
            "|1.0              |",
            "|0.1              |",
            "|100.0            |",
            "|1.0E7            |",
            "|9999999.0        |",
            "|1.23456789E7     |",
            "|1.0E10           |",
            "|1.0E20           |",
            "|0.001            |",
            "|1.0E-4           |",
            "|1.5E-5           |",
            "|-0.0             |",
            "|1.23456789123E8  |",
            "|1.23456789012E-10|",
            "|Infinity         |",
            "|-Infinity        |",
            "|NULL             |",
            "+-----------------+",
            "",
        ),
        id="doubles_untruncated",
    ),
    pytest.param(
        [
            (x,)
            for x in [
                1.0,
                0.1,
                100.0,
                1e7,
                9999999.0,
                12345678.9,
                1e10,
                1e20,
                0.001,
                0.0001,
                1.5e-5,
                -0.0,
                123456789.123,
                1.23456789012e-10,
                float("inf"),
                float("-inf"),
                None,
            ]
        ],
        S(F("d", DoubleType())),
        lambda df: df.show(),
        (
            "+-----------------+",
            "|                d|",
            "+-----------------+",
            "|              1.0|",
            "|              0.1|",
            "|            100.0|",
            "|            1.0E7|",
            "|        9999999.0|",
            "|     1.23456789E7|",
            "|           1.0E10|",
            "|           1.0E20|",
            "|            0.001|",
            "|           1.0E-4|",
            "|           1.5E-5|",
            "|             -0.0|",
            "|  1.23456789123E8|",
            "|1.23456789012E-10|",
            "|         Infinity|",
            "|        -Infinity|",
            "|             NULL|",
            "+-----------------+",
            "",
        ),
        id="doubles",
    ),
    pytest.param(
        [(x,) for x in [1.0, 1.1, 0.1, 3.4e38, 16777217.0, 1e-3, 1e-4, 1e7, -2.5, None]],
        S(F("f", FloatType())),
        lambda df: df.show(truncate=False),
        (
            "+-----------+",
            "|f          |",
            "+-----------+",
            "|1.0        |",
            "|1.1        |",
            "|0.1        |",
            "|3.4E38     |",
            "|1.6777216E7|",
            "|0.001      |",
            "|1.0E-4     |",
            "|1.0E7      |",
            "|-2.5       |",
            "|NULL       |",
            "+-----------+",
            "",
        ),
        id="floats_untruncated",
    ),
    pytest.param(
        [(x,) for x in [1.0, 1.1, 0.1, 3.4e38, 16777217.0, 1e-3, 1e-4, 1e7, -2.5, None]],
        S(F("f", FloatType())),
        lambda df: df.show(),
        (
            "+-----------+",
            "|          f|",
            "+-----------+",
            "|        1.0|",
            "|        1.1|",
            "|        0.1|",
            "|     3.4E38|",
            "|1.6777216E7|",
            "|      0.001|",
            "|     1.0E-4|",
            "|      1.0E7|",
            "|       -2.5|",
            "|       NULL|",
            "+-----------+",
            "",
        ),
        id="floats",
    ),
    pytest.param(
        [(Decimal("12345678901234567890.123456789012345678"),)],
        S(F("d", DecimalType(38, 18))),
        lambda df: df.show(truncate=False),
        (
            "+---------------------------------------+",
            "|d                                      |",
            "+---------------------------------------+",
            "|12345678901234567890.123456789012345678|",
            "+---------------------------------------+",
            "",
        ),
        id="big_decimal_untruncated",
    ),
    pytest.param(
        [(Decimal("12345678901234567890.123456789012345678"),)],
        S(F("d", DecimalType(38, 18))),
        lambda df: df.show(),
        (
            "+--------------------+",
            "|                   d|",
            "+--------------------+",
            "|12345678901234567...|",
            "+--------------------+",
            "",
        ),
        id="big_decimal",
    ),
]


class TestDataFrameShow:
    @pytest.mark.parametrize(("data", "schema", "show", "expected"), SHOW_CASES)
    def test_show(self, spark, capsys, data, schema, show, expected):
        df = spark.createDataFrame(data, schema)
        show(df)
        assert capsys.readouterr().out == printed(expected)

    def test_show_nan(self, spark, capsys):
        df = spark.createDataFrame([("NaN",), ("1.5",)], ["s"])
        df.select(col("s").cast("double").alias("d"), col("s").cast("float").alias("f")).show()
        assert capsys.readouterr().out == printed(
            ("+---+---+", "|  d|  f|", "+---+---+", "|NaN|NaN|", "|1.5|1.5|", "+---+---+", "")
        )

    def test_show_no_arguments_match_defaults(self, spark, capsys):
        df = spark.createDataFrame([(i, "x" * i) for i in range(30)], S(F("i", LongType()), F("s", StringType())))
        df.show()
        default = capsys.readouterr().out
        df.show(20, True, False)
        assert capsys.readouterr().out == default

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"n": "2"}, id="n_str"),
            pytest.param({"n": True}, id="n_bool"),
            pytest.param({"n": 2.0}, id="n_float"),
            pytest.param({"vertical": "yes"}, id="vertical_str"),
            pytest.param({"vertical": 1}, id="vertical_int"),
            pytest.param({"truncate": "x"}, id="truncate_str"),
        ],
    )
    def test_show_argument_types(self, spark, capsys, kwargs):
        df = spark.createDataFrame([(1, "a")], S(F("id", LongType()), F("v", StringType())))
        with pytest.raises(PySparkTypeError, match=f"`{next(iter(kwargs))}`"):
            df.show(**kwargs)
        assert capsys.readouterr().out == ""

    def test_show_truncate_none(self, spark):
        df = spark.createDataFrame([(1, "a")], S(F("id", LongType()), F("v", StringType())))
        with pytest.raises(TypeError, match="NoneType"):
            df.show(truncate=None)

    def test_repr(self, spark):
        df = spark.createDataFrame(
            [],
            S(
                F("id", LongType()),
                F("v", StringType()),
                F("a", ArrayType(IntegerType())),
                F("m", MapType(StringType(), DecimalType(10, 2))),
            ),
        )
        assert repr(df) == "DataFrame[id: bigint, v: string, a: array<int>, m: map<string,decimal(10,2)>]"


@pytest.fixture
def kathmandu(spark):
    if USE_ACTUAL_SPARK:
        previous = spark.conf.get("spark.sql.session.timeZone")
        spark.conf.set("spark.sql.session.timeZone", "Asia/Kathmandu")
        yield
        spark.conf.set("spark.sql.session.timeZone", previous)
    else:
        # The module's session time zone is DuckDB's TimeZone setting.
        (previous,) = spark.conn.execute("SELECT current_setting('TimeZone')").fetchone()
        spark.conn.execute("SET TimeZone = 'Asia/Kathmandu'")
        yield
        spark.conn.execute(f"SET TimeZone = '{previous}'")


class TestShowInTheSessionTimeZone:
    def test_timestamps(self, spark, capsys, kathmandu):
        instant = datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.timezone.utc)
        df = spark.createDataFrame(
            [(instant, [instant], (instant, 1), {"k": instant})],
            "t timestamp, a array<timestamp>, s struct<x: timestamp, n: int>, m map<string, timestamp>",
        )
        df.show(truncate=False)
        # Kathmandu is 5 hours and 45 minutes ahead of UTC.
        separator = "+-------------------+---------------------+------------------------+--------------------------+"
        assert capsys.readouterr().out == printed(
            (
                separator,
                "|t                  |a                    |s                       |m                         |",
                separator,
                "|2024-01-01 17:45:00|[2024-01-01 17:45:00]|{2024-01-01 17:45:00, 1}|{k -> 2024-01-01 17:45:00}|",
                separator,
                "",
            )
        )
        # collect gives the local time of the process, as Spark does.
        assert df.collect()[0].t == instant.astimezone().replace(tzinfo=None)

    def test_nanosecond_timestamps(self, spark, capsys, kathmandu):
        if USE_ACTUAL_SPARK:
            pytest.skip("Spark has no TIMESTAMPTZ_NS")
        df = spark.sql("SELECT TIMESTAMPTZ '2024-01-01 12:00:00.123456789+00'::TIMESTAMPTZ_NS AS t")
        df.show(truncate=False)
        assert capsys.readouterr().out == printed(
            (
                "+--------------------------+",
                "|t                         |",
                "+--------------------------+",
                "|2024-01-01 17:45:00.123456|",
                "+--------------------------+",
                "",
            )
        )
        # DuckDB truncates nanoseconds to microseconds.
        utc = datetime.datetime(2024, 1, 1, 12, 0, 0, 123456, tzinfo=datetime.timezone.utc)
        assert df.collect()[0].t == utc

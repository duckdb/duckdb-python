import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from decimal import Decimal

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import PySparkTypeError
from spark_namespace.sql import functions as F
from spark_namespace.sql.types import (
    ArrayType,
    BinaryType,
    DecimalType,
    DoubleType,
    IntegerType,
    LongType,
    Row,
    StringType,
    StructField,
    StructType,
)

from duckdb.experimental.spark.exception import ContributionsAcceptedError

if USE_ACTUAL_SPARK:
    import pyspark

    SPARK_3 = pyspark.__version__.startswith("3.")
else:
    SPARK_3 = False


def _strings(spark, *columns):
    return StructType([StructField(c, StringType()) for c in columns])


class TestSparkFunctionsString:
    def test_length(self, spark):
        data = [
            ("firstRowFirstColumn",),
            ("2ndRowFirstColumn",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("length", F.length(F.col("firstColumn")))
        res = df.select("length").collect()
        assert res == [
            Row(length=19),
            Row(length=17),
        ]

    def test_upper(self, spark):
        data = [
            ("firstRowFirstColumn",),
            ("2ndRowFirstColumn",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("upper", F.upper(F.col("firstColumn")))
        res = df.select("upper").collect()
        assert res == [
            Row(upper="FIRSTROWFIRSTCOLUMN"),
            Row(upper="2NDROWFIRSTCOLUMN"),
        ]

    def test_ucase(self, spark):
        data = [
            ("firstRowFirstColumn",),
            ("2ndRowFirstColumn",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("upper", F.ucase(F.col("firstColumn")))
        res = df.select("upper").collect()
        assert res == [
            Row(upper="FIRSTROWFIRSTCOLUMN"),
            Row(upper="2NDROWFIRSTCOLUMN"),
        ]

    def test_lower(self, spark):
        data = [
            ("firstRowFirstColumn",),
            ("2ndRowFirstColumn",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("lower", F.lower(F.col("firstColumn")))
        res = df.select("lower").collect()
        assert res == [
            Row(lower="firstrowfirstcolumn"),
            Row(lower="2ndrowfirstcolumn"),
        ]

    def test_lcase(self, spark):
        data = [
            ("firstRowFirstColumn",),
            ("2ndRowFirstColumn",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("lower", F.lcase(F.col("firstColumn")))
        res = df.select("lower").collect()
        assert res == [
            Row(lower="firstrowfirstcolumn"),
            Row(lower="2ndrowfirstcolumn"),
        ]

    def test_trim(self, spark):
        data = [
            (" firstRowFirstColumn ",),
            (" 2ndRowFirstColumn ",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("trimmed", F.trim(F.col("firstColumn")))
        res = df.select("trimmed").collect()
        assert res == [
            Row(trimmed="firstRowFirstColumn"),
            Row(trimmed="2ndRowFirstColumn"),
        ]

    def test_ltrim(self, spark):
        data = [
            (" firstRowFirstColumn ",),
            (" 2ndRowFirstColumn ",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("ltrimmed", F.ltrim(F.col("firstColumn")))
        res = df.select("ltrimmed").collect()
        assert res == [
            Row(ltrimmed="firstRowFirstColumn "),
            Row(ltrimmed="2ndRowFirstColumn "),
        ]

    def test_rtrim(self, spark):
        data = [
            (" firstRowFirstColumn ",),
            (" 2ndRowFirstColumn ",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("rtrimmed", F.rtrim(F.col("firstColumn")))
        res = df.select("rtrimmed").collect()
        assert res == [
            Row(rtrimmed=" firstRowFirstColumn"),
            Row(rtrimmed=" 2ndRowFirstColumn"),
        ]

    def test_endswith(self, spark):
        data = [
            ("firstRowFirstColumn", "Column"),
            ("2ndRowFirstColumn", "column"),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])
        df = df.withColumn("endswith", F.endswith(F.col("firstColumn"), F.col("secondColumn")))
        res = df.select("endswith").collect()
        assert res == [
            Row(endswith=True),
            Row(endswith=False),
        ]

    def test_startswith(self, spark):
        data = [
            ("firstRowFirstColumn", "irst"),
            ("2ndRowFirstColumn", "2nd"),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])
        df = df.withColumn("startswith", F.startswith(F.col("firstColumn"), F.col("secondColumn")))
        res = df.select("startswith").collect()
        assert res == [
            Row(startswith=False),
            Row(startswith=True),
        ]

    def test_ascii(self, spark):
        df = spark.createDataFrame([("Spark",), ("PySpark",), ("Pandas API",)], ["value"])

        res = df.select(F.ascii("value").alias("a")).collect()
        assert res == [Row(a=83), Row(a=80), Row(a=80)]

    def test_btrim(self, spark):
        df = spark.createDataFrame(
            [
                (
                    "SSparkSQLS",
                    "SL",
                )
            ],
            ["a", "b"],
        )

        res = df.select(F.btrim(df.a, df.b).alias("r")).collect()
        assert res == [Row(r="parkSQ")]

        df = spark.createDataFrame([("    SparkSQL   ",)], ["a"])
        res = df.select(F.btrim(df.a).alias("r")).collect()
        assert res == [Row(r="SparkSQL")]

    def test_char(self, spark):
        df = spark.createDataFrame(
            [(65,), (65 + 256,), (66 + 256,)],
            [
                "a",
            ],
        )

        res = df.select(F.char(df.a).alias("ch")).collect()
        assert res == [Row(ch="A"), Row(ch="A"), Row(ch="B")]

    def test_encode(self, spark):
        df = spark.createDataFrame([("abcd",)], ["c"])

        res = df.select(F.encode("c", "UTF-8").alias("encoded")).collect()
        # TODO: Should return the same type  # noqa: TD002, TD003
        if USE_ACTUAL_SPARK:
            assert res == [Row(encoded=bytearray(b"abcd"))]
        else:
            assert res == [Row(encoded=b"abcd")]

    def test_split(self, spark):
        df = spark.createDataFrame(
            [("oneAtwoBthreeC",)],
            [
                "s",
            ],
        )

        res = df.select(F.split(df.s, "[ABC]").alias("s")).collect()
        assert res == [Row(s=["one", "two", "three", ""])]

    def test_split_part(self, spark):
        df = spark.createDataFrame(
            [
                (
                    "11.12.13",
                    ".",
                    3,
                )
            ],
            ["a", "b", "c"],
        )

        res = df.select(F.split_part(df.a, df.b, df.c).alias("r")).collect()
        assert res == [Row(r="13")]

        # If any input is null, should return null
        df = spark.createDataFrame(
            [
                (
                    "11.12.13",
                    ".",
                    None,
                ),
                (
                    "11.12.13",
                    ".",
                    1,
                ),
            ],
            ["a", "b", "c"],
        )
        res = df.select(F.split_part(df.a, df.b, df.c).alias("r")).collect()
        assert res == [Row(r=None), Row(r="11")]

        # If partNum is out of range, should return an empty string
        df = spark.createDataFrame(
            [
                (
                    "11.12.13",
                    ".",
                    4,
                )
            ],
            ["a", "b", "c"],
        )
        res = df.select(F.split_part(df.a, df.b, df.c).alias("r")).collect()
        assert res == [Row(r="")]

        # If partNum is negative, parts are counted backwards
        df = spark.createDataFrame(
            [
                (
                    "11.12.13",
                    ".",
                    -1,
                )
            ],
            ["a", "b", "c"],
        )
        res = df.select(F.split_part(df.a, df.b, df.c).alias("r")).collect()
        assert res == [Row(r="13")]

        # If the delimiter is an empty string, the return should be empty
        df = spark.createDataFrame(
            [
                (
                    "11.12.13",
                    "",
                    2,
                )
            ],
            ["a", "b", "c"],
        )
        res = df.select(F.split_part(df.a, df.b, df.c).alias("r")).collect()
        assert res == [Row(r="")]

    def test_substr(self, spark):
        df = spark.createDataFrame(
            [
                (
                    "Spark SQL",
                    5,
                    1,
                )
            ],
            ["a", "b", "c"],
        )
        res = df.select(F.substr("a", "b", "c").alias("s")).collect()
        assert res == [Row(s="k")]

        df = spark.createDataFrame(
            [
                (
                    "Spark SQL",
                    5,
                    1,
                )
            ],
            ["a", "b", "c"],
        )
        res = df.select(F.substr("a", "b").alias("s")).collect()
        assert res == [Row(s="k SQL")]

    def test_find_in_set(self, spark):
        string_array = "abc,b,ab,c,def"
        df = spark.createDataFrame([("ab", string_array), ("b,c", string_array), ("z", string_array)], ["a", "b"])

        res = df.select(F.find_in_set(df.a, df.b).alias("r")).collect()

        assert res == [Row(r=3), Row(r=0), Row(r=0)]

    def test_initcap(self, spark):
        df = spark.createDataFrame([("ab cd",)], ["a"])

        res = df.select(F.initcap("a").alias("v")).collect()
        assert res == [Row(v="Ab Cd")]

    def test_left(self, spark):
        df = spark.createDataFrame(
            [
                (
                    "Spark SQL",
                    3,
                ),
                (
                    "Spark SQL",
                    0,
                ),
                (
                    "Spark SQL",
                    -3,
                ),
            ],
            ["a", "b"],
        )

        res = df.select(F.left(df.a, df.b).alias("r")).collect()
        assert res == [Row(r="Spa"), Row(r=""), Row(r="")]

    def test_right(self, spark):
        df = spark.createDataFrame(
            [
                (
                    "Spark SQL",
                    3,
                ),
                (
                    "Spark SQL",
                    0,
                ),
                (
                    "Spark SQL",
                    -3,
                ),
            ],
            ["a", "b"],
        )

        res = df.select(F.right(df.a, df.b).alias("r")).collect()
        assert res == [Row(r="SQL"), Row(r=""), Row(r="")]

    def test_left_and_right_of_null(self, spark):
        df = spark.createDataFrame([(None, -1), (None, 0), (None, 2), ("abc", None)], "s string, n int")
        res = df.select(F.left("s", "n").alias("l"), F.right("s", "n").alias("r")).collect()
        assert res == [Row(l=None, r=None)] * 4

    def test_left_and_right_at_the_integer_bounds(self, spark):
        rows = [(None, -2147483648), ("abc", -2147483648), (None, 2147483647), ("abc", 2147483647)]
        df = spark.createDataFrame(rows, "s string, n int")
        res = df.select(F.left("s", "n").alias("l"), F.right("s", "n").alias("r")).collect()
        assert res == [Row(l=None, r=None), Row(l="", r=""), Row(l=None, r=None), Row(l="abc", r="abc")]

    def test_left_and_right_of_a_fractional_length(self, spark):
        # Spark casts the length to an integer, which truncates toward zero.
        df = spark.createDataFrame(
            [("abcde", 1.5, Decimal("2.9")), ("abcde", -0.5, Decimal("0.4"))], "s string, n double, d decimal(3,1)"
        )
        res = df.select(
            F.left("s", "n").alias("ln"),
            F.right("s", "n").alias("rn"),
            F.left("s", "d").alias("ld"),
            F.right("s", "d").alias("rd"),
        ).collect()
        assert res == [Row(ln="a", rn="e", ld="ab", rd="de"), Row(ln="", rn="", ld="", rd="")]

    def test_levenshtein(self, spark):
        df = spark.createDataFrame([("kitten", "sitting"), ("saturdays", "sunday")], ["a", "b"])

        res = df.select(F.levenshtein(df.a, df.b).alias("r"), F.levenshtein(df.a, df.b, 3).alias("r_th")).collect()
        assert res == [Row(r=3, r_th=3), Row(r=4, r_th=-1)]

    def test_lpad(self, spark):
        df = spark.createDataFrame(
            [("abcd",)],
            [
                "s",
            ],
        )

        res = df.select(F.lpad(df.s, 6, "#").alias("s")).collect()
        assert res == [Row(s="##abcd")]

    def test_rpad(self, spark):
        df = spark.createDataFrame(
            [("abcd",)],
            [
                "s",
            ],
        )

        res = df.select(F.rpad(df.s, 6, "#").alias("s")).collect()
        assert res == [Row(s="abcd##")]

    def test_printf(self, spark):
        df = spark.createDataFrame(
            [
                (
                    "aa%d%s",
                    123,
                    "cc",
                )
            ],
            ["a", "b", "c"],
        )
        res = df.select(F.printf("a", "b", "c").alias("r")).collect()
        assert res == [Row(r="aa123cc")]

    @pytest.mark.parametrize("regexp_func", [F.regexp, F.regexp_like])
    def test_regexp_and_regexp_like(self, spark, regexp_func):
        df = spark.createDataFrame([("1a 2b 14m", r"(\d+)")], ["str", "regexp"])
        res = df.select(regexp_func("str", F.lit(r"(\d+)")).alias("m")).collect()
        assert res[0].m is True

        df = spark.createDataFrame([("1a 2b 14m", r"(\d+)")], ["str", "regexp"])
        res = df.select(regexp_func("str", F.lit(r"\d{2}b")).alias("m")).collect()
        assert res[0].m is False

        df = spark.createDataFrame([("1a 2b 14m", r"(\d+)")], ["str", "regexp"])
        res = df.select(regexp_func("str", F.col("regexp")).alias("m")).collect()
        assert res[0].m is True

    def test_regexp_count(self, spark):
        df = spark.createDataFrame([("1a 2b 14m", r"\d+")], ["str", "regexp"])
        res = df.select(F.regexp_count("str", F.lit(r"\d+")).alias("d")).collect()
        assert res == [Row(d=3)]
        res = df.select(F.regexp_count("str", F.lit(r"mmm")).alias("d")).collect()
        assert res == [Row(d=0)]
        res = df.select(F.regexp_count("str", F.col("regexp")).alias("d")).collect()
        assert res == [Row(d=3)]

    def test_regexp_extract(self, spark):
        df = spark.createDataFrame([("100-200",)], ["str"])
        res = df.select(F.regexp_extract("str", r"(\d+)-(\d+)", 1).alias("d")).collect()
        assert res == [Row(d="100")]

        df = spark.createDataFrame([("foo",)], ["str"])
        res = df.select(F.regexp_extract("str", r"(\d+)", 1).alias("d")).collect()
        assert res == [Row(d="")]

        df = spark.createDataFrame([("aaaac",)], ["str"])
        res = df.select(F.regexp_extract("str", "(a+)(b)?(c)", 2).alias("d")).collect()
        assert res == [Row(d="")]

    def test_regexp_extract_all(self, spark):
        df = spark.createDataFrame([("100-200, 300-400", r"(\d+)-(\d+)")], ["str", "regexp"])
        res = df.select(F.regexp_extract_all("str", F.lit(r"(\d+)-(\d+)")).alias("d")).collect()
        assert res == [Row(d=["100", "300"])]

        res = df.select(F.regexp_extract_all("str", F.lit(r"(\d+)-(\d+)"), 1).alias("d")).collect()
        assert res == [Row(d=["100", "300"])]

        res = df.select(F.regexp_extract_all("str", F.lit(r"(\d+)-(\d+)"), 2).alias("d")).collect()
        assert res == [Row(d=["200", "400"])]

        res = df.select(F.regexp_extract_all("str", F.col("regexp")).alias("d")).collect()
        assert res == [Row(d=["100", "300"])]

    def test_regexp_extract_all_of_a_group_outside_the_match(self, spark):
        df = spark.createDataFrame([("a b",), ("zz",), (None,)], "s string")
        res = df.select(
            F.regexp_extract_all("s", F.lit("(a)|(b)"), 1).alias("first"),
            F.regexp_extract_all("s", F.lit("(a)|(b)"), 2).alias("second"),
        )
        assert res.collect() == [Row(first=["a", ""], second=["", "b"]), Row(first=[], second=[]), Row(None, None)]

    def test_regexp_substr(self, spark):
        df = spark.createDataFrame([("1a 2b 14m", r"\d+")], ["str", "regexp"])

        res = df.select(F.regexp_substr("str", F.lit(r"\d+")).alias("d")).collect()
        assert res == [Row(d="1")]

        res = df.select(F.regexp_substr("str", F.lit(r"mmm")).alias("d")).collect()
        assert res == [Row(d=None)]

        res = df.select(F.regexp_substr("str", F.col("regexp")).alias("d")).collect()
        assert res == [Row(d="1")]

    def test_repeat(self, spark):
        df = spark.createDataFrame(
            [("ab",)],
            [
                "s",
            ],
        )
        res = df.select(F.repeat(df.s, 3).alias("s")).collect()
        assert res == [Row(s="ababab")]

    def test_reverse(self, spark):
        data = [
            ("firstRowFirstColumn",),
            ("2ndRowFirstColumn",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("reversed", F.reverse(F.col("firstColumn")))
        res = df.select("reversed").collect()
        assert res == [
            Row(reversed=data[0][0][::-1]),
            Row(reversed=data[1][0][::-1]),
        ]

    def test_concat(self, spark):
        data = [
            ("firstRow", "FirstColumn"),
            ("2ndRow", "FirstColumn"),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])
        df = df.withColumn("concatenated", F.concat(F.col("firstColumn"), F.col("secondColumn")))
        res = df.select("concatenated").collect()
        assert res == [
            Row(concatenated="firstRowFirstColumn"),
            Row(concatenated="2ndRowFirstColumn"),
        ]

    def test_substring(self, spark):
        data = [("abcd",)]
        df = spark.createDataFrame(data, ["s"])
        res = df.select(F.substring(df.s, 1, 2).alias("s")).collect()
        assert res == [Row(s="ab")]

    def test_contains(self, spark):
        data = [
            ("firstRowFirstColumn",),
            ("2ndRowFirstColumn",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("contains", F.contains(F.col("firstColumn"), F.lit("Row")))
        res = df.select("contains").collect()
        assert res == [
            Row(contains=True),
            Row(contains=True),
        ]

    def test_instr(self, spark):
        data = [
            ("firstRowFirstColumn",),
            ("2ndRowFirstColumn",),
        ]
        df = spark.createDataFrame(data, ["firstColumn"])
        df = df.withColumn("instr", F.instr(F.col("firstColumn"), "Row"))
        res = df.select("instr").collect()
        assert res == [
            Row(instr=6),
            Row(instr=4),
        ]

    def test_concat_with_null_is_null(self, spark):
        df = spark.createDataFrame([("a", None), ("a", "b")], _strings(spark, "a", "b"))
        assert df.select(F.concat("a", "b").alias("c")).collect() == [Row(c=None), Row(c="ab")]
        arrays = StructType([StructField("x", ArrayType(LongType())), StructField("y", ArrayType(LongType()))])
        adf = spark.createDataFrame([([1, 2], [3]), ([1], None)], arrays)
        assert adf.select(F.concat("x", "y").alias("c")).collect() == [Row(c=[1, 2, 3]), Row(c=None)]
        assert spark.range(1).select(F.concat().alias("c")).collect() == [Row(c="")]

    def test_concat_keeps_binary_and_arrays(self, spark):
        schema = "a binary, b binary, s string, l array<int>, m array<int>, i int"
        df = spark.createDataFrame([(b"\x00A", b"B", "x", [1, 2], [3], 7), (None, b"C", None, None, [4], None)], schema)
        res = df.select(
            F.concat("a", "b", "b").alias("binary"),
            F.concat("l", "m").alias("arrays"),
            F.concat("b", "s").alias("mixed"),
            F.concat("i", "s").alias("number"),
        )
        assert res.schema.simpleString() == "struct<binary:binary,arrays:array<int>,mixed:string,number:string>"
        assert res.collect() == [
            Row(binary=b"\x00ABB", arrays=[1, 2, 3], mixed="Bx", number="7x"),
            Row(binary=None, arrays=None, mixed=None, number=None),
        ]

    def test_concat_of_one_column_has_the_type_of_a_concatenation(self, spark):
        schema = "b binary, l array<int>, i int, s string"
        df = spark.createDataFrame([(b"\x00A", [1], 7, "x"), (None, None, None, None)], schema)
        res = df.select(*(F.concat(c).alias(c) for c in ["b", "l", "i", "s"]))
        assert res.schema.simpleString() == "struct<b:binary,l:array<int>,i:string,s:string>"
        assert res.collect() == [Row(b=b"\x00A", l=[1], i="7", s="x"), Row(b=None, l=None, i=None, s=None)]

    def test_concat_reads_a_null_literal_as_a_string(self, spark):
        df = spark.createDataFrame([(b"B", "x")], "b binary, s string")
        res = df.select(
            F.concat("s", F.lit(None)).alias("s"),
            F.concat("b", F.lit(None)).alias("b"),
            F.concat(F.lit(None)).alias("n"),
        )
        assert res.schema.simpleString() == "struct<s:string,b:string,n:string>"
        assert res.collect() == [Row(s=None, b=None, n=None)]

    def test_concat_ws_skips_nulls_and_flattens_arrays(self, spark):
        schema = StructType(
            [
                StructField("arr", ArrayType(StringType())),
                StructField("s", StringType()),
                StructField("n", StringType()),
            ]
        )
        df = spark.createDataFrame([(["a", None, "b"], "z", None), ([], "z", None), (None, None, None)], schema)
        res = df.select(F.concat_ws(",", "arr").alias("a"), F.concat_ws("-", "arr", "s", "n").alias("m")).collect()
        assert res == [Row(a="a,b", m="a-b-z"), Row(a="", m="z"), Row(a="", m="")]

    def test_substring_positions(self, spark):
        df = spark.createDataFrame([("hello",), (None,)], _strings(spark, "s"))
        res = df.select(
            F.substring("s", 0, 2).alias("zero"),
            F.substring("s", 1, 2).alias("one"),
            F.substring("s", -3, 2).alias("neg"),
            F.substring("s", -7, 4).alias("before"),
            F.substring("s", -10, 3).alias("far"),
            F.substring("s", 2, -1).alias("neglen"),
            F.substring("s", 3, 100).alias("long"),
        ).collect()
        assert res == [
            Row(zero="he", one="he", neg="ll", before="he", far="", neglen="", long="llo"),
            Row(zero=None, one=None, neg=None, before=None, far=None, neglen=None, long=None),
        ]

    def test_substr_with_columns(self, spark):
        schema = StructType(
            [StructField("s", StringType()), StructField("p", LongType()), StructField("n", LongType())]
        )
        df = spark.createDataFrame([("hello", 0, 2), ("hello", -3, 2), ("hello", 2, -1)], schema)
        res = df.select(F.substr("s", "p", "n").alias("a"), F.substr(df.s, df.p).alias("b")).collect()
        assert res == [Row(a="he", b="hello"), Row(a="ll", b="llo"), Row(a="", b="ello")]

    def test_initcap_lowers_the_rest(self, spark):
        df = spark.createDataFrame([("hELLO wORLD",), ("ÉCOLE",)], _strings(spark, "s"))
        assert df.select(F.initcap("s").alias("s")).collect() == [Row(s="Hello World"), Row(s="École")]

    def test_pad_with_empty_padding(self, spark):
        df = spark.createDataFrame([("spark",), (None,)], _strings(spark, "s"))
        res = df.select(
            F.lpad("s", 8, "").alias("a"),
            F.rpad("s", 3, "").alias("b"),
            F.lpad("s", 7, "ab").alias("c"),
            F.lpad("s", 0, "").alias("d"),
        ).collect()
        assert res == [Row(a="spark", b="spa", c="abspark", d=""), Row(a=None, b=None, c=None, d=None)]

    def test_levenshtein_threshold_null(self, spark):
        df = spark.createDataFrame([("kitten", "sitting"), (None, "a")], _strings(spark, "a", "b"))
        res = df.select(
            F.levenshtein("a", "b").alias("d"),
            F.levenshtein("a", "b", 2).alias("t2"),
            F.levenshtein("a", "b", 3).alias("t3"),
        )
        assert res.collect() == [Row(d=3, t2=-1, t3=3), Row(d=None, t2=None, t3=None)]

    def test_split_part_edges(self, spark):
        df = spark.createDataFrame([("a,b,c",)], _strings(spark, "s"))
        parts = {"first": 1, "second": 2, "last": -1, "second_last": -2}
        res = df.select(*[F.split_part("s", F.lit(""), F.lit(i)).alias(name) for name, i in parts.items()]).collect()
        assert res == [Row(first="a,b,c", second="", last="a,b,c", second_last="")]
        with pytest.raises(Exception, match="INVALID_INDEX_OF_ZERO"):
            df.select(F.split_part("s", F.lit(","), F.lit(0))).collect()

    def test_char_wraps_and_handles_negatives(self, spark):
        df = spark.createDataFrame(
            [(65,), (256,), (512 + 66,), (-1,), (None,)], StructType([StructField("i", LongType())])
        )
        assert df.select(F.char("i").alias("c")).collect() == [
            Row(c="A"),
            Row(c="\x00"),
            Row(c="B"),
            Row(c=""),
            Row(c=None),
        ]

    def test_sha2_zero_bits_is_sha256(self, spark):
        df = spark.createDataFrame([("ab",)], _strings(spark, "s"))
        digest = "fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603"
        assert df.select(F.sha2("s", 0).alias("a"), F.sha2("s", 256).alias("b")).collect() == [Row(a=digest, b=digest)]

    def test_integer_result_types(self, spark):
        df = spark.createDataFrame([("hello", "help")], _strings(spark, "a", "b"))
        res = df.select(F.length("a"), F.instr("a", "l"), F.levenshtein("a", "b"), F.regexp_count("a", F.lit("l")))
        assert [f.dataType for f in res.schema.fields] == [IntegerType()] * 4
        assert res.collect() == [Row(5, 3, 2, 2)]

    def test_regexp_replace_java_replacement(self, spark):
        df = spark.createDataFrame([("100-200",)], _strings(spark, "s"))
        res = df.select(
            F.regexp_replace("s", r"(\d+)-(\d+)", "$2-$1").alias("swap"),
            F.regexp_replace("s", "-", r"\$").alias("dollar"),
            F.regexp_replace("s", "-", r"\\").alias("backslash"),
            F.regexp_replace("s", r"(\d+)", "<$0>").alias("whole"),
        ).collect()
        assert res == [Row(swap="200-100", dollar="100$200", backslash="100\\200", whole="<100>-<200>")]

    def test_regexp_replace_of_an_empty_match_after_a_match(self, spark):
        df = spark.createDataFrame([("aa",), ("ba",), ("abc",)], _strings(spark, "s"))
        res = df.select(F.regexp_replace("s", "a*", "x").alias("a"), F.regexp_replace("s", "b*", "+").alias("b"))
        if USE_ACTUAL_SPARK:
            assert res.collect() == [Row(a="xx", b="+a+a+"), Row(a="xbxx", b="++a+"), Row(a="xxbxcx", b="+a++c+")]
        else:
            # RE2 skips an empty match that starts where the previous match ended, where Java's matcher takes it.
            assert res.collect() == [Row(a="x", b="+a+a+"), Row(a="xbx", b="+a+"), Row(a="xbxcx", b="+a+c+")]

    def test_regexp_replace_reads_group_numbers_like_java(self, spark):
        df = spark.createDataFrame([("abcdefghij",)], _strings(spark, "s"))
        assert df.select(F.regexp_replace("s", "(a)", "$1$1").alias("r")).collect() == [Row(r="aabcdefghij")]
        cases = [
            ("(a)(b)(c)(d)(e)(f)(g)(h)(i)", "$10", "a0j"),
            ("(a)(b)", "$12", "a2cdefghij"),
            ("(a)(b)", "$01", "acdefghij"),
        ]
        for pattern, replacement, spark_value in cases:
            if USE_ACTUAL_SPARK:
                # Java reads another digit while the group number stays within the pattern's groups.
                res = df.select(F.regexp_replace("s", pattern, replacement).alias("r"))
                assert res.collect() == [Row(r=spark_value)]
            else:
                with pytest.raises(ContributionsAcceptedError, match=r"only \$0 to \$9"):
                    F.regexp_replace("s", pattern, replacement)

    def test_regexp_replace_group_ten(self, spark):
        df = spark.createDataFrame([("abcdefghij", "(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)")], _strings(spark, "s", "p"))
        ten = "(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)"
        if USE_ACTUAL_SPARK:
            res = df.select(
                F.regexp_replace("s", ten, "$10").alias("a"), F.regexp_replace("s", F.col("p"), "$10").alias("b")
            )
            assert res.collect() == [Row(a="j", b="j")]
            return
        # DuckDB refers to groups 0 to 9 only.
        for pattern in [ten, F.col("p")]:
            with pytest.raises(ContributionsAcceptedError, match=r"only \$0 to \$9"):
                F.regexp_replace("s", pattern, "$10")

    def test_regexp_replace_column_reads_group_numbers_like_java(self, spark):
        rows = [("abcdefghij", "$10"), ("abcdefghij", "<$1>")]
        df = spark.createDataFrame(rows, _strings(spark, "s", "r"))
        if not USE_ACTUAL_SPARK:
            with pytest.raises(ContributionsAcceptedError, match="literal replacement"):
                F.regexp_replace("s", "(a)", F.col("r"))
            return
        nine = df.select(F.regexp_replace("s", "(a)(b)(c)(d)(e)(f)(g)(h)(i)", F.col("r")).alias("r"))
        assert nine.collect() == [Row(r="a0j"), Row(r="<a>j")]
        ten = df.select(F.regexp_replace("s", "(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)", F.col("r")).alias("r"))
        assert ten.collect() == [Row(r="j"), Row(r="<a>")]

    def test_find_in_set_null(self, spark):
        df = spark.createDataFrame([(None, "a,b"), ("a", None), ("b", "a,b"), ("c", "a,b")], _strings(spark, "s", "l"))
        assert df.select(F.find_in_set("s", "l").alias("i")).collect() == [Row(i=None), Row(i=None), Row(i=2), Row(i=0)]

    def test_octet_length_type(self, spark):
        df = spark.createDataFrame([("abc",)], _strings(spark, "s"))
        res = df.select(F.octet_length("s").alias("n"))
        assert res.schema.fields[0].dataType == IntegerType()
        assert res.collect() == [Row(n=3)]

    def test_regexp_replace_with_columns(self, spark):
        rows = [("100-200", "$2-$1", "-"), ("1-2", "\\$x", "(\\d)"), ("3-4", None, "-")]
        df = spark.createDataFrame(rows, _strings(spark, "s", "r", "p"))
        res = df.select(
            F.regexp_replace("s", F.col("p"), "+").alias("pattern"),
            F.regexp_replace("s", "-", F.lit("<$0>")).alias("literal"),
        )
        assert res.collect() == [
            Row(pattern="100+200", literal="100<->200"),
            Row(pattern="+-+", literal="1<->2"),
            Row(pattern="3+4", literal="3<->4"),
        ]
        if not USE_ACTUAL_SPARK:
            with pytest.raises(ContributionsAcceptedError, match="literal replacement"):
                F.regexp_replace("s", r"(\d+)-(\d+)", F.col("r"))
            return
        per_row = df.select(F.regexp_replace("s", r"(\d+)-(\d+)", F.col("r")).alias("r"))
        assert per_row.collect() == [Row(r="200-100"), Row(r="$x"), Row(r=None)]
        assert df.select(F.regexp_replace("s", "-", F.col("r"))).columns == ["regexp_replace(s, -, r, 1)"]

    @pytest.mark.skipif(not USE_ACTUAL_SPARK, reason="every replacement column raises in the module")
    @pytest.mark.parametrize("replacement", ["x\\", "$x"])
    def test_regexp_replace_invalid_replacement_column(self, spark, replacement):
        df = spark.createDataFrame([("a", replacement)], _strings(spark, "s", "r"))
        with pytest.raises(Exception):  # noqa: B017, PT011
            df.select(F.regexp_replace("s", "a", F.col("r"))).collect()

    def test_regexp_replace_with_a_null_replacement(self, spark):
        df = spark.createDataFrame([("a",)], _strings(spark, "s"))
        assert df.select(F.regexp_replace("s", "a", F.lit(None)).alias("r")).collect() == [Row(r=None)]

    @pytest.mark.parametrize(("replacement", "error"), [("x\\", "character to be escaped"), ("$x", r"only \$0 to \$9")])
    def test_regexp_replace_invalid_replacement(self, spark, replacement, error):
        df = spark.createDataFrame([("a",)], _strings(spark, "s"))
        with pytest.raises(Exception, match=None if USE_ACTUAL_SPARK else error):
            df.select(F.regexp_replace("s", "a", replacement)).collect()

    def test_pad_with_an_empty_pad(self, spark):
        df = spark.createDataFrame([("abc",)], _strings(spark, "s"))
        res = df.select(F.lpad("s", 5, ""), F.rpad("s", 2, ""), F.lpad("s", -1, ""))
        assert res.columns == ["lpad(s, 5, )", "rpad(s, 2, )", "lpad(s, -1, )"]
        assert res.collect() == [Row("abc", "ab", "")]

    def test_pad_with_columns(self, spark):
        schema = StructType(
            [StructField("s", StringType()), StructField("n", IntegerType()), StructField("p", StringType())]
        )
        df = spark.createDataFrame([("abc", 5, "#"), ("abc", 2, ""), ("abc", None, None)], schema)
        if SPARK_3:
            with pytest.raises(PySparkTypeError, match="NOT_ITERABLE"):
                F.lpad("s", F.col("n"), F.col("p"))
            return
        res = df.select(F.lpad("s", F.col("n"), F.col("p")), F.rpad("s", F.col("n"), "#"))
        assert res.columns == ["lpad(s, n, p)", "rpad(s, n, #)"]
        assert res.collect() == [Row("##abc", "abc##"), Row("ab", "ab"), Row(None, None)]

    def test_pad_with_a_null_width_and_an_empty_pad(self, spark):
        schema = StructType(
            [StructField("s", StringType()), StructField("n", IntegerType()), StructField("p", StringType())]
        )
        df = spark.createDataFrame([("abc", None, ""), ("abc", -1, ""), ("abc", 2, "")], schema)
        if SPARK_3:
            with pytest.raises(PySparkTypeError, match="NOT_ITERABLE"):
                F.lpad("s", F.col("n"), F.col("p"))
            return
        res = df.select(F.lpad("s", F.col("n"), F.col("p")).alias("l"), F.rpad("s", F.col("n"), F.col("p")).alias("r"))
        assert res.collect() == [Row(l=None, r=None), Row(l="", r=""), Row(l="ab", r="ab")]

    def test_split_with_a_column_limit(self, spark):
        df = spark.createDataFrame([("a,b,c",)], _strings(spark, "s"))
        if not USE_ACTUAL_SPARK:
            with pytest.raises(ContributionsAcceptedError, match="limit is not yet supported"):
                F.split("s", ",", F.lit(2))
        elif SPARK_3:
            with pytest.raises(PySparkTypeError, match="NOT_ITERABLE"):
                F.split("s", ",", F.lit(2))
        else:
            assert df.select(F.split("s", ",", F.lit(2)).alias("p")).collect() == [Row(p=["a", "b,c"])]

    def test_substr_with_null_position_or_length(self, spark):
        schema = StructType([StructField("s", StringType()), StructField("p", LongType())])
        df = spark.createDataFrame([("abcdef", None)], schema)
        res = df.select(
            F.substr("s", "p", F.lit(2)).alias("pos"),
            F.substr("s", "p").alias("pos_only"),
            F.substr("s", F.lit(2), "p").alias("len"),
        )
        assert res.collect() == [Row(pos=None, pos_only=None, len=None)]

    def test_char_truncates_fractional_codes(self, spark):
        schema = StructType([StructField("d", DoubleType()), StructField("m", DecimalType(5, 1))])
        df = spark.createDataFrame([(65.7, Decimal("66.9")), (-0.5, Decimal("-0.9")), (321.9, None)], schema)
        assert df.select(F.char("d").alias("d"), F.char("m").alias("m")).collect() == [
            Row(d="A", m="B"),
            Row(d="\x00", m="\x00"),
            Row(d="A", m=None),
        ]

    def test_binary_functions_take_utf8_bytes(self, spark):
        df = spark.createDataFrame([("é",), ("aé€😀",), ("a\\x41",)], _strings(spark, "s"))
        res = df.select(F.octet_length("s").alias("n"), F.base64("s").alias("b"))
        assert res.collect() == [Row(n=2, b="w6k="), Row(n=10, b="YcOp4oKs8J+YgA=="), Row(n=5, b="YVx4NDE=")]
        binary = spark.createDataFrame([(bytearray(b"\xff\x00\\x"),)], StructType([StructField("b", BinaryType())]))
        assert binary.select(F.octet_length("b").alias("n"), F.base64("b").alias("b")).collect() == [
            Row(n=4, b="/wBceA==")
        ]

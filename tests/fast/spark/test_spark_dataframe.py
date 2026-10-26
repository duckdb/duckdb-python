import pytest

_ = pytest.importorskip("duckdb.experimental.spark")


from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import (
    AnalysisException,
    ParseException,
    PySparkTypeError,
    PySparkValueError,
    error_condition,
)
from spark_namespace.sql.column import Column
from spark_namespace.sql.functions import col, lit, struct, when
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
    Row,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)

requires_tree_string = pytest.mark.skipif(
    not hasattr(StructType, "treeString"), reason="StructType.treeString is only introduced in PySpark 4.0.0"
)

PERSON_SCHEMA = StructType(
    [
        StructField("id", IntegerType(), True),
        StructField(
            "person",
            StructType([StructField("name", StringType(), True), StructField("age", IntegerType(), True)]),
            True,
        ),
        StructField("hobbies", ArrayType(StringType()), True),
    ]
)
PERSON_TREE = (
    "root\n"
    " |-- id: integer (nullable = true)\n"
    " |-- person: struct (nullable = true)\n"
    " |    |-- name: string (nullable = true)\n"
    " |    |-- age: integer (nullable = true)\n"
    " |-- hobbies: array (nullable = true)\n"
    " |    |-- element: string (containsNull = true)\n"
)
FLAT_TYPES = [
    ("b", ByteType()),
    ("s", ShortType()),
    ("i", IntegerType()),
    ("l", LongType()),
    ("f", FloatType()),
    ("d", DoubleType()),
    ("m", DecimalType(10, 2)),
    ("str", StringType()),
    ("bool", BooleanType()),
    ("date", DateType()),
    ("ts", TimestampType()),
    ("ntz", TimestampNTZType()),
    ("bin", BinaryType()),
    ("iv", DayTimeIntervalType()),
    ("null", NullType()),
]


def assert_column_objects_equal(col1: Column, col2: Column):
    assert type(col1) is type(col2)
    if not USE_ACTUAL_SPARK:
        # An expression has no truth value, so the comparison is on the SQL text.
        assert str(col1.expr) == str(col2.expr)


class TestDataFrame:
    def test_row_is_exported_from_sql_module(self):
        if USE_ACTUAL_SPARK:
            from pyspark.sql import Row as SqlRow
        else:
            from duckdb.experimental.spark.sql import Row as SqlRow
        assert SqlRow is Row

    def test_dataframe_from_list_of_tuples(self, spark):
        # Valid
        address = [(1, "14851 Jeffrey Rd", "DE"), (2, "43421 Margarita St", "NY"), (3, "13111 Siemon Ave", "CA")]
        df = spark.createDataFrame(address, ["id", "address", "state"])
        res = df.collect()
        assert res == [
            Row(id=1, address="14851 Jeffrey Rd", state="DE"),
            Row(id=2, address="43421 Margarita St", state="NY"),
            Row(id=3, address="13111 Siemon Ave", state="CA"),
        ]

        # Tuples of different sizes
        address = [(1, "14851 Jeffrey Rd", "DE"), (2, "43421 Margarita St", "NY"), (3, "13111 Siemon Ave")]

        if USE_ACTUAL_SPARK:
            from py4j.protocol import Py4JJavaError
            from spark_namespace.errors import IllegalArgumentException

            # Spark only checks the row length when the rows are collected.
            ragged = spark.createDataFrame(address, ["id", "address", "state"])
            with pytest.raises(
                (IllegalArgumentException, Py4JJavaError), match="3 fields are required while 2 values are provided"
            ):
                ragged.collect()
        else:
            with pytest.raises(PySparkValueError, match="LENGTH_SHOULD_BE_THE_SAME"):
                spark.createDataFrame(address, ["id", "address", "state"])

        # Dataframe instead of list; Spark 4 renamed the error class from SHOULD_NOT_DATAFRAME to INVALID_TYPE.
        with pytest.raises(PySparkTypeError, match="should not be a DataFrame"):
            spark.createDataFrame(df, ["id", "address", "state"])

        # Not a list
        with pytest.raises(TypeError, match="not iterable"):
            spark.createDataFrame(5, ["id", "address", "test"])

        # Empty list
        with pytest.raises(PySparkValueError, match="CANNOT_INFER_EMPTY_SCHEMA"):
            spark.createDataFrame([], ["id", "address", "test"])

        # Duplicate column names: inference merges the fields by name, so both become strings
        address = [(1, "14851 Jeffrey Rd", "DE"), (2, "43421 Margarita St", "NY"), (3, "13111 Siemon Ave", "DE")]
        df = spark.createDataFrame(address, ["id", "address", "id"])
        assert df.columns == ["id", "address", "id"]
        assert df.collect() == [
            ("1", "14851 Jeffrey Rd", "DE"),
            ("2", "43421 Margarita St", "NY"),
            ("3", "13111 Siemon Ave", "DE"),
        ]

        # Not enough column names: the remaining columns get default names
        df = spark.createDataFrame(address, ["id", "address"])
        assert df.columns == ["id", "address", "_3"]

        # Empty column names list
        df = spark.createDataFrame(address, [])
        assert df.columns == ["_1", "_2", "_3"]
        assert df.collect() == [
            Row(_1=1, _2="14851 Jeffrey Rd", _3="DE"),
            Row(_1=2, _2="43421 Margarita St", _3="NY"),
            Row(_1=3, _2="13111 Siemon Ave", _3="DE"),
        ]

        # Too many column names
        with pytest.raises(IndexError):
            spark.createDataFrame(address, ["id", "address", "one", "two", "three"])

        # Column names is not a list
        with pytest.raises(PySparkTypeError):
            spark.createDataFrame(address, {"a": 5, "b": 6, "c": 42})

        with pytest.raises(PySparkTypeError):
            spark.createDataFrame(address, 5)

        # A string is a schema, and "a" is not a type
        with pytest.raises(ParseException):
            spark.createDataFrame(address, "a")

    def test_dataframe(self, spark):
        # Create DataFrame
        df = spark.createDataFrame([("Scala", 25000), ("Spark", 35000), ("PHP", 21000)])
        assert df.columns == ["_1", "_2"]
        res = df.collect()
        assert res == [Row(_1="Scala", _2=25000), Row(_1="Spark", _2=35000), Row(_1="PHP", _2=21000)]

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="We can't create tables with our Spark test setup")
    def test_writing_to_table(self, spark):
        # Create Hive table & query it.
        spark.sql(
            """
            create table sample_table("_1" bool, "_2" integer)
        """
        )
        spark.sql("insert into sample_table VALUES (True, 42)")
        spark.table("sample_table").write.saveAsTable("sample_hive_table")
        df3 = spark.sql("SELECT _1,_2 FROM sample_hive_table")
        res = df3.collect()
        assert res == [Row(_1=True, _2=42)]
        schema = df3.schema
        assert schema == StructType([StructField("_1", BooleanType(), True), StructField("_2", IntegerType(), True)])

    def test_dataframe_collect(self, spark):
        df = spark.createDataFrame([(42,), (21,)]).toDF("a")
        res = df.collect()
        assert str(res) == "[Row(a=42), Row(a=21)]"

    def test_dataframe_from_rows(self, spark):
        columns = ["language", "users_count"]
        data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000")]

        rowData = (Row(*x) for x in data)
        df = spark.createDataFrame(rowData, columns)
        res = df.collect()
        assert res == [
            Row(language="Java", users_count="20000"),
            Row(language="Python", users_count="100000"),
            Row(language="Scala", users_count="3000"),
        ]

    def test_empty_df(self, spark):
        schema = StructType(
            [
                StructField("firstname", StringType(), True),
                StructField("middlename", StringType(), True),
                StructField("lastname", StringType(), True),
            ]
        )
        df = spark.createDataFrame([], schema=schema)
        res = df.collect()
        # TODO: assert that the types and column names are correct  # noqa: TD002, TD003
        assert res == []

    def test_df_from_pandas(self, spark):
        import pandas as pd

        df = spark.createDataFrame(pd.DataFrame({"a": [42, 21], "b": [True, False]}))
        res = df.collect()
        assert res == [Row(a=42, b=True), Row(a=21, b=False)]

    def test_df_from_struct_type(self, spark):
        schema = StructType([StructField("a", LongType()), StructField("b", BooleanType())])
        df = spark.createDataFrame([(42, True), (21, False)], schema)
        res = df.collect()
        assert res == [Row(a=42, b=True), Row(a=21, b=False)]

    def test_df_from_name_list(self, spark):
        df = spark.createDataFrame([(42, True), (21, False)], ["a", "b"])
        res = df.collect()
        assert res == [Row(a=42, b=True), Row(a=21, b=False)]

    def test_df_creation_coverage(self, spark):
        from spark_namespace.sql.types import IntegerType, StringType, StructField, StructType

        data2 = [
            ("James", "", "Smith", "36636", "M", 3000),
            ("Michael", "Rose", "", "40288", "M", 4000),
            ("Robert", "", "Williams", "42114", "M", 4000),
            ("Maria", "Anne", "Jones", "39192", "F", 4000),
            ("Jen", "Mary", "Brown", "", "F", -1),
        ]

        schema = StructType(
            [
                StructField("firstname", StringType(), True),
                StructField("middlename", StringType(), True),
                StructField("lastname", StringType(), True),
                StructField("id", StringType(), True),
                StructField("gender", StringType(), True),
                StructField("salary", IntegerType(), True),
            ]
        )

        df = spark.createDataFrame(data=data2, schema=schema)
        res = df.collect()
        assert res == [
            Row(firstname="James", middlename="", lastname="Smith", id="36636", gender="M", salary=3000),
            Row(firstname="Michael", middlename="Rose", lastname="", id="40288", gender="M", salary=4000),
            Row(firstname="Robert", middlename="", lastname="Williams", id="42114", gender="M", salary=4000),
            Row(firstname="Maria", middlename="Anne", lastname="Jones", id="39192", gender="F", salary=4000),
            Row(firstname="Jen", middlename="Mary", lastname="Brown", id="", gender="F", salary=-1),
        ]

    def test_df_nested_struct(self, spark):
        structureData = [
            (("James", "", "Smith"), "36636", "M", 3100),
            (("Michael", "Rose", ""), "40288", "M", 4300),
            (("Robert", "", "Williams"), "42114", "M", 1400),
            (("Maria", "Anne", "Jones"), "39192", "F", 5500),
            (("Jen", "Mary", "Brown"), "", "F", -1),
        ]
        structureSchema = StructType(
            [
                StructField(
                    "name",
                    StructType(
                        [
                            StructField("firstname", StringType(), True),
                            StructField("middlename", StringType(), True),
                            StructField("lastname", StringType(), True),
                        ]
                    ),
                ),
                StructField("id", StringType(), True),
                StructField("gender", StringType(), True),
                StructField("salary", IntegerType(), True),
            ]
        )

        df2 = spark.createDataFrame(data=structureData, schema=structureSchema)
        res = df2.collect()
        name = Row("firstname", "middlename", "lastname")
        assert res == [
            Row(name=name("James", "", "Smith"), id="36636", gender="M", salary=3100),
            Row(name=name("Michael", "Rose", ""), id="40288", gender="M", salary=4300),
            Row(name=name("Robert", "", "Williams"), id="42114", gender="M", salary=1400),
            Row(name=name("Maria", "Anne", "Jones"), id="39192", gender="F", salary=5500),
            Row(name=name("Jen", "Mary", "Brown"), id="", gender="F", salary=-1),
        ]
        assert res[0].name.firstname == "James"
        schema = df2.schema
        assert schema == StructType(
            [
                StructField(
                    "name",
                    StructType(
                        [
                            StructField("firstname", StringType(), True),
                            StructField("middlename", StringType(), True),
                            StructField("lastname", StringType(), True),
                        ]
                    ),
                    True,
                ),
                StructField("id", StringType(), True),
                StructField("gender", StringType(), True),
                StructField("salary", IntegerType(), True),
            ]
        )

    def test_df_columns(self, spark):
        from spark_namespace.sql.functions import col

        structureData = [
            (("James", "", "Smith"), "36636", "M", 3100),
            (("Michael", "Rose", ""), "40288", "M", 4300),
            (("Robert", "", "Williams"), "42114", "M", 1400),
            (("Maria", "Anne", "Jones"), "39192", "F", 5500),
            (("Jen", "Mary", "Brown"), "", "F", -1),
        ]
        structureSchema = StructType(
            [
                StructField(
                    "name",
                    StructType(
                        [
                            StructField("firstname", StringType(), True),
                            StructField("middlename", StringType(), True),
                            StructField("lastname", StringType(), True),
                        ]
                    ),
                ),
                StructField("id", StringType(), True),
                StructField("gender", StringType(), True),
                StructField("salary", IntegerType(), True),
            ]
        )

        df2 = spark.createDataFrame(data=structureData, schema=structureSchema)
        updatedDF = df2.withColumn(
            "OtherInfo",
            struct(
                col("id").alias("identifier"),
                col("gender").alias("gender"),
                col("salary").alias("salary"),
                when(col("salary").cast(IntegerType()) < 2000, "Low")
                .when(col("salary").cast(IntegerType()) < 4000, "Medium")
                .otherwise("High")
                .alias("Salary_Grade"),
            ),
        ).drop("id", "gender", "salary")

        assert "OtherInfo" in updatedDF.columns

    def test_array_and_map_type(self, spark):
        """Array & Map."""
        StructType(
            [
                StructField(
                    "name",
                    StructType(
                        [
                            StructField("firstname", StringType(), True),
                            StructField("middlename", StringType(), True),
                            StructField("lastname", StringType(), True),
                        ]
                    ),
                ),
                StructField("hobbies", ArrayType(StringType()), True),
                StructField("properties", MapType(StringType(), StringType()), True),
            ]
        )

    def test_getitem_filter(self, spark):
        data = [(56, "Carol"), (20, "Alice"), (3, "Dave"), (3, "Anna"), (1, "Ben")]
        df = spark.createDataFrame(data, ["age", "name"])
        expected = [
            Row(age=56, name="Carol"),
            Row(age=20, name="Alice"),
        ]

        df = df[col("age") > 18]
        assert df.collect() == expected

    def test_getitem_column(self, spark):
        data = [(56, "Carol"), (20, "Alice")]
        df = spark.createDataFrame(data, ["age", "name"])

        assert_column_objects_equal(df["age"], df.age)
        if not USE_ACTUAL_SPARK:
            assert str(df["age"].expr).endswith('."age"')
            assert str(col("age").expr) == '"age"'

    def test_getitem_dataframe(self, spark):
        data = [(56, "Ben", "Street1"), (20, "Tom", "Street2")]
        df = spark.createDataFrame(data, ["age", "name", "address"])

        res = df[["name", "address"]].collect()
        expected = df.select("name", "address").collect()

        assert res == expected

    def test_getattr_dataframe(self, spark):
        data = [(56, "Ben", "Street1"), (20, "Tom", "Street2")]
        df = spark.createDataFrame(data, ["age", "name", "address"])

        assert_column_objects_equal(df.age, df["age"])
        if not USE_ACTUAL_SPARK:
            assert str(df.age.expr).endswith('."age"')

    def test_head_first(self, spark):
        data = [(56, "Carol"), (20, "Alice"), (3, "Dave"), (3, "Anna"), (1, "Ben")]
        df = spark.createDataFrame(data, ["age", "name"])
        expected = Row(age=1, name="Ben")

        head = df.orderBy(df.age).head()
        first = df.orderBy(df.age).first()
        assert head == first == expected

    def test_head_take_n(self, spark):
        data = [(56, "Carol"), (20, "Alice"), (3, "Dave"), (3, "Anna"), (1, "Ben")]
        df = spark.createDataFrame(data, ["age", "name"])
        expected = [
            Row(age=1, name="Ben"),
            Row(age=3, name="Anna"),
        ]
        df = df.orderBy(df.age, df.name)
        rows = df.head(2)
        take = df.take(2)
        assert rows == take == expected

    def test_drop(self, spark):
        data = [(1, 2, 3, 4)]
        df = spark.createDataFrame(data, ["one", "two", "three", "four"])
        expected = ["one", "four"]
        assert df.drop("two", "three").columns == expected
        assert df.drop("two", col("three")).columns == expected
        assert df.drop("two", col("three"), col("missing")).columns == expected

    def test_drop_ignores_case(self, spark):
        df = spark.createDataFrame([(1, 2)], ["a", "b"])
        assert df.drop("A").columns == ["b"]
        assert df.drop(col("A")).columns == ["b"]
        assert df.drop(df["A"]).columns == ["b"]
        assert df.drop(df.a).collect() == [Row(b=2)]

    def test_drop_ignores_what_is_not_a_column_reference(self, spark):
        df = spark.createDataFrame([(1, 2)], ["a", "b"])
        other = spark.createDataFrame([(1, 3)], ["a", "c"])
        assert df.drop(df.a + 1).columns == ["a", "b"]
        assert df.drop(df.a.alias("x")).columns == ["a", "b"]
        assert df.drop(other.c).columns == ["a", "b"]
        assert df.drop(other.a).columns == ["a", "b"]

    def test_drop_every_column(self, spark):
        df = spark.createDataFrame([(1, 2)], ["a", "b"])
        assert df.drop("a", "b").columns == []
        assert df.drop("a", "b").schema == StructType([])

    def test_cache(self, spark):
        data = [(1, 2, 3, 4)]
        df = spark.createDataFrame(data, ["one", "two", "three", "four"])
        one = df.one
        cached = df.cache()
        assert cached is df
        assert cached.collect() == [Row(one=1, two=2, three=3, four=4)]
        assert df.select(one).collect() == [Row(one=1)]
        assert df.filter(df.two == 2).select(df.three).collect() == [Row(three=3)]

    def test_dtypes(self, spark):
        data = [("Alice", 25, 5000.0), ("Bob", 30, 6000.0)]
        df = spark.createDataFrame(data, ["name", "age", "salary"])
        dtypes = df.dtypes

        assert isinstance(dtypes, list)
        assert len(dtypes) == 3
        for col_name, col_type in dtypes:
            assert isinstance(col_name, str)
            assert isinstance(col_type, str)

        col_names = [name for name, _ in dtypes]
        assert col_names == ["name", "age", "salary"]
        for _, col_type in dtypes:
            assert len(col_type) > 0

    def test_dtypes_complex_types(self, spark):
        from spark_namespace.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

        schema = StructType(
            [
                StructField("name", StringType(), True),
                StructField("scores", ArrayType(IntegerType()), True),
                StructField(
                    "address",
                    StructType([StructField("city", StringType(), True), StructField("zip", StringType(), True)]),
                    True,
                ),
            ]
        )
        data = [
            ("Alice", [90, 85, 88], {"city": "NYC", "zip": "10001"}),
            ("Bob", [75, 80, 82], {"city": "LA", "zip": "90001"}),
        ]
        df = spark.createDataFrame(data, schema)
        dtypes = df.dtypes

        assert len(dtypes) == 3
        col_names = [name for name, _ in dtypes]
        assert col_names == ["name", "scores", "address"]

    def test_printSchema(self, spark, capsys):
        data = [("Alice", 25, 5000), ("Bob", 30, 6000)]
        schema = StructType(
            [StructField("name", StringType()), StructField("age", LongType()), StructField("salary", LongType())]
        )
        df = spark.createDataFrame(data, schema)
        df.printSchema()
        assert capsys.readouterr().out == (
            "root\n"
            " |-- name: string (nullable = true)\n"
            " |-- age: long (nullable = true)\n"
            " |-- salary: long (nullable = true)\n"
            "\n"
        )

    def test_printSchema_nested(self, spark, capsys):
        df = spark.createDataFrame(
            [
                (1, {"name": "Alice", "age": 25}, ["reading", "coding"]),
                (2, {"name": "Bob", "age": 30}, ["gaming", "music"]),
            ],
            PERSON_SCHEMA,
        )
        df.printSchema()
        assert capsys.readouterr().out == PERSON_TREE + "\n"
        df.printSchema(1)
        assert capsys.readouterr().out == (
            "root\n"
            " |-- id: integer (nullable = true)\n"
            " |-- person: struct (nullable = true)\n"
            " |-- hobbies: array (nullable = true)\n"
            "\n"
        )
        df.printSchema(level=2)
        assert capsys.readouterr().out == PERSON_TREE + "\n"

    @pytest.mark.parametrize("level", [0, -1])
    def test_printSchema_negative_level(self, spark, capsys, level):
        df = spark.createDataFrame([(1, {"name": "Alice", "age": 25}, ["reading"])], PERSON_SCHEMA)
        df.printSchema(level=level)
        assert capsys.readouterr().out == PERSON_TREE + "\n"

    def test_printSchema_type_names(self, spark, capsys):
        schema = StructType([StructField(name, data_type) for name, data_type in FLAT_TYPES])
        df = spark.createDataFrame([], schema)
        df.printSchema()
        assert capsys.readouterr().out == (
            "root\n"
            " |-- b: byte (nullable = true)\n"
            " |-- s: short (nullable = true)\n"
            " |-- i: integer (nullable = true)\n"
            " |-- l: long (nullable = true)\n"
            " |-- f: float (nullable = true)\n"
            " |-- d: double (nullable = true)\n"
            " |-- m: decimal(10,2) (nullable = true)\n"
            " |-- str: string (nullable = true)\n"
            " |-- bool: boolean (nullable = true)\n"
            " |-- date: date (nullable = true)\n"
            " |-- ts: timestamp (nullable = true)\n"
            " |-- ntz: timestamp_ntz (nullable = true)\n"
            " |-- bin: binary (nullable = true)\n"
            " |-- iv: interval day to second (nullable = true)\n"
            " |-- null: void (nullable = true)\n"
            "\n"
        )
        assert df.dtypes == [
            ("b", "tinyint"),
            ("s", "smallint"),
            ("i", "int"),
            ("l", "bigint"),
            ("f", "float"),
            ("d", "double"),
            ("m", "decimal(10,2)"),
            ("str", "string"),
            ("bool", "boolean"),
            ("date", "date"),
            ("ts", "timestamp"),
            ("ntz", "timestamp_ntz"),
            ("bin", "binary"),
            ("iv", "interval day to second"),
            ("null", "void"),
        ]

    @requires_tree_string
    def test_treeString_basic(self, spark):
        data = [("Alice", 25, 5000)]
        df = spark.createDataFrame(data, ["name", "age", "salary"])
        tree = df.schema.treeString()

        assert tree.startswith("root\n")
        assert " |-- name:" in tree
        assert " |-- age:" in tree
        assert " |-- salary:" in tree
        assert "(nullable = true)" in tree
        assert tree.count(" |-- ") == 3

    @requires_tree_string
    def test_treeString_nested_struct(self, spark):
        from spark_namespace.sql.types import IntegerType, StringType, StructField, StructType

        schema = StructType(
            [
                StructField("id", IntegerType(), True),
                StructField(
                    "person",
                    StructType([StructField("name", StringType(), True), StructField("age", IntegerType(), True)]),
                    True,
                ),
            ]
        )
        data = [(1, {"name": "Alice", "age": 25})]
        df = spark.createDataFrame(data, schema)
        tree = df.schema.treeString()

        assert "root\n" in tree
        assert " |-- id:" in tree
        assert " |-- person: struct (nullable = true)" in tree
        assert "name:" in tree
        assert "age:" in tree

    @requires_tree_string
    def test_treeString_with_level(self, spark):
        from spark_namespace.sql.types import IntegerType, StringType, StructField, StructType

        schema = StructType(
            [
                StructField("id", IntegerType(), True),
                StructField(
                    "person",
                    StructType(
                        [
                            StructField("name", StringType(), True),
                            StructField("details", StructType([StructField("address", StringType(), True)]), True),
                        ]
                    ),
                    True,
                ),
            ]
        )

        data = [(1, {"name": "Alice", "details": {"address": "123 Main St"}})]
        df = spark.createDataFrame(data, schema)

        assert df.schema.treeString(maxDepth=1) == (
            "root\n |-- id: integer (nullable = true)\n |-- person: struct (nullable = true)\n"
        )
        assert df.schema.treeString(2) == (
            "root\n"
            " |-- id: integer (nullable = true)\n"
            " |-- person: struct (nullable = true)\n"
            " |    |-- name: string (nullable = true)\n"
            " |    |-- details: struct (nullable = true)\n"
        )
        full = df.schema.treeString()
        assert full.endswith(" |    |    |-- address: string (nullable = true)\n")
        assert df.schema.treeString(3) == full
        assert df.schema.treeString(0) == full
        assert df.schema.treeString(-1) == full

    @requires_tree_string
    def test_treeString_array_type(self, spark):
        from spark_namespace.sql.types import ArrayType, StringType, StructField, StructType

        schema = StructType(
            [StructField("name", StringType(), True), StructField("hobbies", ArrayType(StringType()), True)]
        )

        data = [("Alice", ["reading", "coding"])]
        df = spark.createDataFrame(data, schema)
        assert df.schema.treeString() == (
            "root\n"
            " |-- name: string (nullable = true)\n"
            " |-- hobbies: array (nullable = true)\n"
            " |    |-- element: string (containsNull = true)\n"
        )

    @requires_tree_string
    @pytest.mark.parametrize(
        ("max_depth", "expected"),
        [
            pytest.param(
                None,
                (
                    "root\n"
                    " |-- id: integer (nullable = true)\n"
                    " |-- person: struct (nullable = true)\n"
                    " |    |-- name: string (nullable = true)\n"
                    " |    |-- details: struct (nullable = true)\n"
                    " |    |    |-- address: string (nullable = false)\n"
                    " |-- arr: array (nullable = true)\n"
                    " |    |-- element: string (containsNull = false)\n"
                    " |-- arr_struct: array (nullable = true)\n"
                    " |    |-- element: struct (containsNull = true)\n"
                    " |    |    |-- x: integer (nullable = true)\n"
                    " |-- arr_arr: array (nullable = true)\n"
                    " |    |-- element: array (containsNull = true)\n"
                    " |    |    |-- element: long (containsNull = true)\n"
                    " |-- map: map (nullable = true)\n"
                    " |    |-- key: string\n"
                    " |    |-- value: integer (valueContainsNull = false)\n"
                    " |-- map_nested: map (nullable = true)\n"
                    " |    |-- key: string\n"
                    " |    |-- value: struct (valueContainsNull = true)\n"
                    " |    |    |-- y: decimal(5,2) (nullable = true)\n"
                    " |-- a\\nb: string (nullable = true)\n"
                ),
                id="all",
            ),
            pytest.param(
                2,
                (
                    "root\n"
                    " |-- id: integer (nullable = true)\n"
                    " |-- person: struct (nullable = true)\n"
                    " |    |-- name: string (nullable = true)\n"
                    " |    |-- details: struct (nullable = true)\n"
                    " |-- arr: array (nullable = true)\n"
                    " |    |-- element: string (containsNull = false)\n"
                    " |-- arr_struct: array (nullable = true)\n"
                    " |    |-- element: struct (containsNull = true)\n"
                    " |-- arr_arr: array (nullable = true)\n"
                    " |    |-- element: array (containsNull = true)\n"
                    " |-- map: map (nullable = true)\n"
                    " |    |-- key: string\n"
                    " |    |-- value: integer (valueContainsNull = false)\n"
                    " |-- map_nested: map (nullable = true)\n"
                    " |    |-- key: string\n"
                    " |    |-- value: struct (valueContainsNull = true)\n"
                    " |-- a\\nb: string (nullable = true)\n"
                ),
                id="2",
            ),
        ],
    )
    def test_treeString_nesting(self, max_depth, expected):
        schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField(
                    "person",
                    StructType(
                        [
                            StructField("name", StringType()),
                            StructField("details", StructType([StructField("address", StringType(), False)])),
                        ]
                    ),
                ),
                StructField("arr", ArrayType(StringType(), False)),
                StructField("arr_struct", ArrayType(StructType([StructField("x", IntegerType())]))),
                StructField("arr_arr", ArrayType(ArrayType(LongType()))),
                StructField("map", MapType(StringType(), IntegerType(), False)),
                StructField("map_nested", MapType(StringType(), StructType([StructField("y", DecimalType(5, 2))]))),
                StructField("a\nb", StringType()),
            ]
        )
        assert (schema.treeString() if max_depth is None else schema.treeString(max_depth)) == expected


class TestRepeatedColumnNames:
    @pytest.fixture
    def repeated(self, spark):
        return spark.createDataFrame([(1, "Ann")], ["id", "name"]).toDF("x", "x")

    def test_reference_is_ambiguous(self, repeated):
        assert repeated.columns == ["x", "x"]
        for make in (lambda: repeated.select("x"), lambda: repeated.x, lambda: repeated["x"]):
            with pytest.raises(AnalysisException) as error:
                make().collect()
            assert error_condition(error.value) == "AMBIGUOUS_REFERENCE"

    def test_columns_keep_their_values(self, repeated):
        assert [tuple(r) for r in repeated.withColumn("z", lit(1)).collect()] == [(1, "Ann", 1)]
        assert [tuple(r) for r in repeated.withColumnRenamed("x", "y").collect()] == [(1, "Ann")]
        assert [tuple(r) for r in repeated.toDF("a", "b").collect()] == [(1, "Ann")]

    @pytest.fixture
    def beside(self, spark):
        """A repeated name beside a name equal to the name DuckDB's binder gives the repeat's second column."""
        return spark.createDataFrame([(10, 20, 30)], ["x", "x", "x_1"])

    @pytest.mark.parametrize(
        "derive",
        [
            lambda df: df,
            lambda df: df.limit(1),
            lambda df: df.select("*"),
            lambda df: df.distinct(),
            lambda df: df.sort("x_1"),
            lambda df: df.filter(col("x_1") > 0),
        ],
    )
    def test_unique_name_beside_a_repeat(self, beside, derive):
        df = derive(beside)
        assert df.select("x_1").collect() == [Row(x_1=30)]
        assert [tuple(r) for r in df.select(df.x_1, df["x_1"], col("x_1") + 1).collect()] == [(30, 30, 31)]
        assert [tuple(r) for r in df.filter(col("x_1") == 30).collect()] == [(10, 20, 30)]
        assert df.groupBy().sum("x_1").collect() == [Row(**{"sum(x_1)": 30})]
        assert df.groupBy("x_1").count().collect() == [Row(x_1=30, count=1)]
        assert df.withColumn("y", col("x_1") * 2).select("y").collect() == [Row(y=60)]
        assert df.sort(col("x_1").desc()).select("x_1").collect() == [Row(x_1=30)]
        assert [tuple(r) for r in df.drop("x_1").collect()] == [(10, 20)]

    def test_unique_name_beside_a_repeat_from_sql_and_renames(self, spark):
        assert spark.sql("SELECT 10 AS x, 20 AS x, 30 AS x_1").select("x_1").collect() == [Row(x_1=30)]
        source = spark.createDataFrame([(10, 20, 30)], ["a", "b", "c"])
        assert source.toDF("x", "x", "x_1").select("x_1").collect() == [Row(x_1=30)]
        renamed = source.toDF("x", "y", "x_1").withColumnRenamed("y", "x")
        assert renamed.select("x_1").collect() == [Row(x_1=30)]
        assert source.select("a", "b").toDF("x", "x_1").select("x", "x", "x_1").select("x_1").collect() == [Row(x_1=20)]
        assert source.toDF("x", "X", "x_1").select("x_1").collect() == [Row(x_1=30)]

    def test_name_of_a_repeat_in_duckdb_is_unresolved(self, repeated):
        for make in (
            lambda: repeated.select("x_1"),
            lambda: repeated.select(col("x_1")),
            lambda: repeated.filter(col("x_1") == "Ann"),
            lambda: repeated["x_1"],
        ):
            with pytest.raises(AnalysisException) as error:
                make().collect()
            assert error_condition(error.value) == "UNRESOLVED_COLUMN.WITH_SUGGESTION"

    def test_join_on_a_unique_name_beside_a_repeat(self, spark, beside):
        other = spark.createDataFrame([(30, "hit")], ["k", "v"])
        assert beside.join(other, beside["x_1"] == other["k"]).select("v").collect() == [Row(v="hit")]
        assert beside.join(other, col("x_1") == col("k")).select("v").collect() == [Row(v="hit")]
        keyed = spark.createDataFrame([(30,)], ["x_1"])
        assert [tuple(r) for r in beside.join(keyed, "x_1").select("x_1").collect()] == [(30,)]

    @pytest.mark.parametrize(
        "derive",
        [
            lambda j, a: j.limit(1),
            lambda j, a: j.sort("x"),
            lambda j, a: j.distinct(),
            lambda j, a: j.select("*"),
            lambda j, a: j.filter(a.x > 0),
            lambda j, a: j.limit(1).filter(col("x") > 0),
        ],
    )
    def test_join_output_name_beside_a_repeat(self, spark, derive):
        a = spark.createDataFrame([(1, 10)], ["id", "x"])
        b = spark.createDataFrame([(1, 30)], ["id", "id_1"])
        df = derive(a.join(b, a.id == b.id), a)
        assert df.select("id_1").collect() == [Row(id_1=30)]
        assert df.filter(col("id_1") == 30).select("x").collect() == [Row(x=10)]
        assert df.groupBy().sum("id_1").collect() == [Row(**{"sum(id_1)": 30})]
        crossed = a.crossJoin(b).filter(a.id == b.id)
        assert crossed.select("id_1").collect() == [Row(id_1=30)]

    @pytest.fixture
    def filtered_join(self, spark):
        a = spark.createDataFrame([(1, 10)], ["id", "x"]).alias("a")
        b = spark.createDataFrame([(1, 30)], ["id", "id_1"]).alias("b")
        return a.join(b, a.id == b.id).filter(a.x > 0)

    def test_filtered_join_as_a_join_input(self, spark, filtered_join):
        other = spark.createDataFrame([(100,)], ["z"])
        for joined in (filtered_join.crossJoin(other), other.crossJoin(filtered_join)):
            assert joined.select("id_1", col("id_1") + 1).collect() == [Row(id_1=30, **{"(id_1 + 1)": 31})]
            assert joined.groupBy().sum("id_1").collect() == [Row(**{"sum(id_1)": 30})]
        assert [tuple(r) for r in filtered_join.crossJoin(other).select("*").collect()] == [(1, 10, 1, 30, 100)]
        keyed = spark.createDataFrame([(30, "hit")], ["k", "v"])
        assert filtered_join.join(keyed, col("id_1") == col("k")).select("v").collect() == [Row(v="hit")]

    def test_filtered_join_on_names_as_a_join_input(self, spark):
        a = spark.createDataFrame([(1, 10, 7)], ["id", "x", "k"])
        b = spark.createDataFrame([(1, 20)], ["id", "x"])
        joined = a.join(b, "id").filter(col("k") > 0).crossJoin(spark.createDataFrame([(100,)], ["z"]))
        assert [tuple(r) for r in joined.collect()] == [(1, 10, 7, 20, 100)]

    def test_view_of_a_join_with_repeated_names(self, spark, filtered_join):
        a = spark.createDataFrame([(1, 10)], ["id", "x"]).alias("a")
        b = spark.createDataFrame([(1, 30)], ["id", "id_1"]).alias("b")
        for name, df in (("repeated_join_view", a.join(b, a.id == b.id)), ("repeated_filter_view", filtered_join)):
            df.createOrReplaceTempView(name)
            assert spark.table(name).select("id_1").collect() == [Row(id_1=30)]
        repeated = spark.createDataFrame([(10, 20)], ["x", "x"]).alias("r")
        repeated.crossJoin(spark.createDataFrame([(30,)], ["x_1"])).createOrReplaceTempView("repeated_input_view")
        assert spark.table("repeated_input_view").select("x_1").collect() == [Row(x_1=30)]
        # Spark's view keeps the repeated names; DuckDB names a view's columns apart.
        expected = ["x", "x", "x_1"] if USE_ACTUAL_SPARK else ["x", "x_2", "x_1"]
        assert spark.table("repeated_input_view").columns == expected

    def test_repeat_beside_a_name_of_another_join_input(self, spark):
        a = spark.createDataFrame([(10, 20)], ["x", "x"]).alias("a")
        b = spark.createDataFrame([(30,)], ["x_1"]).alias("b")
        y = spark.createDataFrame([(7,)], ["y"]).alias("y")
        for joined in (a.crossJoin(b), b.crossJoin(a), a.crossJoin(y).crossJoin(b)):
            assert joined.select("x_1").collect() == [Row(x_1=30)]
            assert joined.select(col("x_1")).collect() == [Row(x_1=30)]
            assert joined.filter(col("x_1") == 30).select("x_1").collect() == [Row(x_1=30)]
            assert joined.groupBy(col("x_1")).count().collect() == [Row(x_1=30, count=1)]
            assert joined.sort(col("x_1")).select("x_1").collect() == [Row(x_1=30)]
        assert [tuple(r) for r in a.crossJoin(y).crossJoin(b).select(y.y, col("y.y")).collect()] == [(7, 7)]
        assert a.join(b, col("x_1") == 30).select("x_1").collect() == [Row(x_1=30)]

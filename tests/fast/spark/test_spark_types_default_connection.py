"""The engine type behind a spark type: where it is built, that reads never parse, pickling and the cache.

A name is parsed on duckdb's default connection when the types module is imported, which builds every singleton;
decimals, arrays and maps are composed without a connection and held by their instances; a read never parses, with one
exception: a struct without field names, the shape DuckDB builds for a tuple, parses its text on the first read of that
text, because the text needs the engine's parser and no constructor builds it. A session has a connection of its own,
so a type built while the default connection was idle and first read while it is busy, a type built while it is busy,
and the session's casts, frames and queries must not depend on the default connection's state.
"""

import copy
import datetime
import decimal
import os
import pickle
import subprocess
import sys
import textwrap
import threading

import pytest

import duckdb

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.sql import functions as F
from spark_namespace.sql.types import (
    ArrayType,
    ByteType,
    DayTimeIntervalType,
    DecimalType,
    DoubleType,
    FloatType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructField,
    StructType,
    UnsignedIntegerType,
    UnsignedShortType,
)

pytestmark = pytest.mark.skipif(USE_ACTUAL_SPARK, reason="pyspark has no duckdb default connection")

WAIT = 60

# Runs inside a query on the default connection, so the connection is held once the function is reached, and the
# connection then refuses another call.
HOLD_THE_DEFAULT_CONNECTION = """
    import threading

    import duckdb

    gate, held = threading.Event(), threading.Event()

    def wait_for_gate(x: int) -> int:
        held.set()
        gate.wait(WAIT)
        return x

    duckdb.create_function("wait_for_gate", wait_for_gate, [int], int)
    worker = threading.Thread(target=lambda: duckdb.sql("select wait_for_gate(1)").fetchall())
    worker.start()
    assert held.wait(WAIT)
    try:
        duckdb.sql("select 1").fetchall()
    except duckdb.ResourceInUseException:
        pass
    else:
        raise SystemExit("the default connection did not refuse")
    """.replace("WAIT", str(WAIT))


def _run(*parts: str) -> subprocess.CompletedProcess[str]:
    script = "".join(textwrap.dedent(part) for part in parts)
    # An editable install that rebuilds on import otherwise prints its build output to stdout.
    env = {**os.environ, "SKBUILD_EDITABLE_VERBOSE": "0"}
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False, env=env
    )


def _built_while_idle():
    # Composed and, but for the struct built through add, never read; nothing built here shares a type text with
    # what the state half builds. The struct's one read here is what makes its reads inside the state parse nothing.
    unnamed = StructType([StructField("", UnsignedShortType())]).add("", LongType())
    str(unnamed.duckdb_type)
    return {
        "decimal": DecimalType(17, 4),
        "array_of_shorts": ArrayType(ShortType()),
        "map_of_doubles": MapType(StringType(), DoubleType()),
        "unnamed_struct_read_once": unnamed,
    }


def _reads_casts_and_queries_inside_the_state(spark, built):
    assert str(UnsignedShortType().duckdb_type) == "USMALLINT"
    assert str(built["decimal"].duckdb_type) == "DECIMAL(17,4)"
    assert str(built["array_of_shorts"].duckdb_type) == "SMALLINT[]"
    assert str(built["map_of_doubles"].duckdb_type) == "MAP(VARCHAR, DOUBLE)"
    assert str(built["unnamed_struct_read_once"].duckdb_type) == "TUPLE(USMALLINT, BIGINT)"
    for kind in (*built.values(), UnsignedShortType()):
        assert kind.duckdb_type is kind.duckdb_type
    # Built, not read: the text of a struct without field names is parsed on its first read, never when it is built.
    added = StructType().add("", ByteType()).add("", FloatType())
    assert added.simpleString() == "struct<:tinyint,:float>"
    assert str(DecimalType(14, 4).duckdb_type) == "DECIMAL(14,4)"
    assert str(ArrayType(DayTimeIntervalType()).duckdb_type) == "INTERVAL[]"
    assert str(MapType(StringType(), DayTimeIntervalType()).duckdb_type) == "MAP(VARCHAR, INTERVAL)"
    assert str(F.col("a").cast(UnsignedIntegerType())) == "Column<'CAST(a AS UINTEGER)'>"
    second = datetime.timedelta(seconds=1)
    schema = StructType([StructField("a", UnsignedIntegerType()), StructField("b", ArrayType(DayTimeIntervalType()))])
    frame = spark.createDataFrame([(2, [second])], schema)
    assert frame.collect()[0].b == [second]
    assert frame.select(F.col("a").cast(DecimalType(14, 4)).alias("d")).collect()[0].d == 2
    rows = spark.sql("select row(1, 2) r, 3 x")
    assert rows.schema.names == ["r", "x"]
    assert rows.select("x").collect()[0].x == 3
    assert spark.sql("select row(1, 2) r").collect()[0].r == (1, 2)


def test_reads_casts_and_queries_while_the_default_connection_runs_a_query_on_another_thread(spark):
    built = _built_while_idle()
    gate, held = threading.Event(), threading.Event()

    def wait_for_gate(x: int) -> int:
        # Runs inside the query, so the default connection is held once this is reached.
        held.set()
        gate.wait(WAIT)
        return x

    duckdb.create_function("wait_for_gate", wait_for_gate, [int], int)
    worker = threading.Thread(target=lambda: duckdb.sql("select wait_for_gate(1)").fetchall())
    try:
        worker.start()
        assert held.wait(WAIT), "the query on the other thread did not reach the function"
        with pytest.raises(duckdb.ResourceInUseException):
            duckdb.sql("select 1").fetchall()
        _reads_casts_and_queries_inside_the_state(spark, built)
    finally:
        gate.set()
        worker.join(WAIT)
        duckdb.remove_function("wait_for_gate")
    assert not worker.is_alive()


def test_reads_casts_and_queries_after_a_statement_failed_in_the_default_connections_transaction(spark):
    built = _built_while_idle()
    duckdb.execute("BEGIN")
    try:
        with pytest.raises(duckdb.Error):
            duckdb.execute("SELECT error('boom')")
        with pytest.raises(duckdb.TransactionException):
            duckdb.sql("select 1").fetchall()
        _reads_casts_and_queries_inside_the_state(spark, built)
    finally:
        duckdb.execute("ROLLBACK")


def test_types_first_built_while_the_default_connection_is_held_in_a_fresh_interpreter():
    # Nothing ran before the import, so the cache holds the singletons' names the import parsed and no other text.
    proc = _run(
        "import duckdb.experimental.spark.sql.types as T\n",
        HOLD_THE_DEFAULT_CONNECTION,
        """
        try:
            array, decimal = T.ArrayType(T.DayTimeIntervalType()), T.DecimalType(9, 3)
            mapping = T.MapType(T.StringType(), T.DayTimeIntervalType())
            built = (T.LongType(), T.TimestampNanosecondNTZType(), array, decimal, mapping)
            print(*(kind.duckdb_type for kind in built))
        finally:
            gate.set()
            worker.join(WAIT)
        assert not worker.is_alive()
        """.replace("WAIT", str(WAIT)),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "BIGINT TIMESTAMP_NS INTERVAL[] DECIMAL(9,3) MAP(VARCHAR, INTERVAL)"


def test_pickled_types_answer_in_a_fresh_interpreter(spark, tmp_path):
    types = {
        "decimal": DecimalType(17, 4),
        "array": ArrayType(LongType(), containsNull=False),
        "map": MapType(StringType(), DoubleType(), valueContainsNull=False),
        "schema": StructType([StructField("a", DecimalType(17, 4)), StructField("b", ArrayType(LongType()))]),
    }
    rows = [(decimal.Decimal("1.5"), [2])]
    expected = [
        f"{types[name]!r} {types[name].duckdb_type} {F.col('x').cast(types[name])}"
        for name in ("decimal", "array", "map")
    ]
    expected.append(str(spark.createDataFrame(rows, types["schema"]).collect()))
    path = tmp_path / "types.pkl"
    path.write_bytes(pickle.dumps(types))
    proc = _run(
        f"""
        import pickle
        from decimal import Decimal

        from duckdb.experimental.spark.sql import SparkSession
        from duckdb.experimental.spark.sql import functions as F

        types = pickle.load(open({str(path)!r}, "rb"))
        spark = SparkSession.builder.getOrCreate()
        spark.conn.execute("SET threads = 1")
        for name in ("decimal", "array", "map"):
            print(repr(types[name]), types[name].duckdb_type, F.col("x").cast(types[name]))
        print(spark.createDataFrame({rows!r}, types["schema"]).collect())
        """
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.splitlines() == expected


class Money(DecimalType):
    def __init__(self, currency: str = "EUR") -> None:
        super().__init__(18, 2)
        self.currency = currency


class Tags(ArrayType):
    def __init__(self) -> None:
        super().__init__(StringType(), containsNull=False)


class Lookup(MapType):
    def __init__(self, note: str) -> None:
        super().__init__(StringType(), DecimalType(9, 3))
        self.note = note


@pytest.mark.parametrize(
    "original",
    [
        pytest.param(Money("USD"), id="decimal_subclass_with_its_own_constructor"),
        pytest.param(Tags(), id="array_subclass_without_arguments"),
        pytest.param(Lookup("kept"), id="map_subclass_with_an_attribute"),
    ],
)
@pytest.mark.parametrize("duplicate", [copy.copy, copy.deepcopy, lambda kind: pickle.loads(pickle.dumps(kind))])
def test_a_subclass_keeps_its_class_attributes_and_engine_type_through_a_copy(original, duplicate):
    # pickle and copy restore the instance state and compose the engine type again, so a subclass's own constructor
    # never runs and nothing it set is lost.
    original.note = "set after construction"
    twin = duplicate(original)
    assert type(twin) is type(original)
    assert twin == original
    assert twin.__dict__ == original.__dict__
    assert str(twin.duckdb_type) == str(original.duckdb_type)


def test_building_many_distinct_decimals_arrays_and_maps_leaves_the_cache_as_it_was():
    from duckdb.experimental.spark.sql import _type_cache

    before = len(_type_cache.TYPES)
    for i in range(1000):
        ArrayType(StructType([StructField(f"f{i}", LongType())]))
        MapType(StringType(), ArrayType(StructType([StructField(f"g{i}", LongType())])))
        DecimalType(10 + i % 28, i % 10)
    assert len(_type_cache.TYPES) == before

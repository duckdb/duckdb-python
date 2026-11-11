"""The engine type behind a spark type: where it is built, that reads never parse, pickling and the cache.

A name is parsed on duckdb's default connection when the types module is imported, which builds every singleton; the
unnamed struct's text when such a struct is built; decimals, arrays and maps are composed without a connection and held
by their instances; a read never parses. A session has a connection of its own, so a type built while the default
connection was idle and first read while it is busy, a type built while it is busy, and the session's casts and frames
must not depend on the default connection's state.
"""

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
    DayTimeIntervalType,
    DecimalType,
    DoubleType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
    UnsignedIntegerType,
    UnsignedShortType,
)

pytestmark = pytest.mark.skipif(USE_ACTUAL_SPARK, reason="pyspark has no duckdb default connection")

WAIT = 60

# Runs inside a query on the default connection, so the connection is held once the function is reached.
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
    """.replace("WAIT", str(WAIT))

RELEASE_THE_DEFAULT_CONNECTION = """
    gate.set()
    worker.join(WAIT)
    assert not worker.is_alive()
    """.replace("WAIT", str(WAIT))


def _run(*parts: str) -> subprocess.CompletedProcess[str]:
    script = "".join(textwrap.dedent(part) for part in parts)
    # An editable install that rebuilds on import otherwise prints its build output to stdout.
    env = {**os.environ, "SKBUILD_EDITABLE_VERBOSE": "0"}
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False, env=env
    )


def _built_while_idle():
    # An array or map reads its element when it is built; the outer type's first read happens inside the state.
    return {
        "decimal": DecimalType(17, 4),
        "array_of_intervals": ArrayType(DayTimeIntervalType()),
        "map_of_intervals": MapType(StringType(), DayTimeIntervalType()),
        "unnamed_struct": StructType([StructField("", UnsignedShortType())]),
        "array_of_unnamed_structs": ArrayType(StructType([StructField("", UnsignedShortType())])),
    }


def _first_reads_and_casts(spark, built):
    assert str(UnsignedShortType().duckdb_type) == "USMALLINT"
    assert str(LongType().duckdb_type) == "BIGINT"
    assert str(built["decimal"].duckdb_type) == "DECIMAL(17,4)"
    assert str(built["array_of_intervals"].duckdb_type) == "INTERVAL[]"
    assert str(built["map_of_intervals"].duckdb_type) == "MAP(VARCHAR, INTERVAL)"
    assert str(built["unnamed_struct"].duckdb_type) == "TUPLE(USMALLINT)"
    assert str(built["array_of_unnamed_structs"].duckdb_type) == "TUPLE(USMALLINT)[]"
    for name in ("array_of_intervals", "map_of_intervals", "unnamed_struct"):
        assert built[name].duckdb_type is built[name].duckdb_type
    assert str(DecimalType(14, 4).duckdb_type) == "DECIMAL(14,4)"
    assert str(ArrayType(DayTimeIntervalType()).duckdb_type) == "INTERVAL[]"
    assert str(F.col("a").cast(UnsignedIntegerType())) == "Column<'CAST(a AS UINTEGER)'>"
    second = datetime.timedelta(seconds=1)
    schema = StructType([StructField("a", UnsignedIntegerType()), StructField("b", ArrayType(DayTimeIntervalType()))])
    frame = spark.createDataFrame([(2, [second])], schema)
    assert frame.collect()[0].b == [second]
    assert frame.select(F.col("a").cast(DecimalType(14, 4)).alias("d")).collect()[0].d == 2


def test_first_reads_and_casts_while_the_default_connection_runs_a_query_on_another_thread(spark):
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
        _first_reads_and_casts(spark, built)
    finally:
        gate.set()
        worker.join(WAIT)
        duckdb.remove_function("wait_for_gate")
    assert not worker.is_alive()


def test_first_reads_and_casts_after_a_statement_failed_in_the_default_connections_transaction(spark):
    built = _built_while_idle()
    duckdb.execute("BEGIN")
    try:
        with pytest.raises(duckdb.Error):
            duckdb.execute("SELECT error('boom')")
        with pytest.raises(duckdb.TransactionException):
            duckdb.sql("select 1").fetchall()
        _first_reads_and_casts(spark, built)
    finally:
        duckdb.execute("ROLLBACK")


def test_a_singleton_first_built_while_the_default_connection_is_held_was_parsed_at_import():
    # A fresh interpreter, so that the only construction before the busy state is the import's.
    proc = _run(
        "import duckdb.experimental.spark.sql.types as T\n",
        HOLD_THE_DEFAULT_CONNECTION,
        """
        try:
            long, timestamp, array = T.LongType(), T.TimestampNanosecondNTZType(), T.ArrayType(T.ShortType())
            print(long.duckdb_type, timestamp.duckdb_type, array.duckdb_type)
        finally:
            gate.set()
            worker.join(WAIT)
        assert not worker.is_alive()
        """.replace("WAIT", str(WAIT)),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "BIGINT TIMESTAMP_NS SMALLINT[]"


def test_pickled_types_answer_in_a_fresh_interpreter(spark, tmp_path):
    types = {
        "decimal": DecimalType(17, 4),
        "array": ArrayType(LongType()),
        "map": MapType(StringType(), DoubleType()),
        "schema": StructType([StructField("a", DecimalType(17, 4)), StructField("b", ArrayType(LongType()))]),
    }
    rows = [(decimal.Decimal("1.5"), [2])]
    expected = [f"{types[name].duckdb_type} {F.col('x').cast(types[name])}" for name in ("decimal", "array", "map")]
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
            print(types[name].duckdb_type, F.col("x").cast(types[name]))
        print(spark.createDataFrame({rows!r}, types["schema"]).collect())
        """
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.splitlines() == expected


def test_building_many_distinct_decimals_arrays_and_maps_leaves_the_cache_as_it_was():
    from duckdb.experimental.spark.sql import _type_cache

    before = len(_type_cache.TYPES)
    for i in range(1000):
        ArrayType(StructType([StructField(f"f{i}", LongType())]))
        MapType(StringType(), ArrayType(StructType([StructField(f"g{i}", LongType())])))
        DecimalType(10 + i % 28, i % 10)
    assert len(_type_cache.TYPES) == before

"""Spark types keep answering while duckdb's default connection is busy or in a failed transaction.

The DuckDBPyType behind a spark type is parsed on the default connection when the types module is imported, which
builds every singleton, and never when a type is read; decimals and containers are composed without a connection. A
session has a connection of its own, so a type built while the default connection was idle and first read while it is
busy, a type built while it is busy, and the session's casts and frames must not depend on the default connection's
state.
"""

import datetime
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


def _built_while_idle():
    # Constructed, never read: the first read happens inside the state the test puts the default connection in.
    return {
        "decimal": DecimalType(17, 4),
        "array_of_intervals": ArrayType(DayTimeIntervalType()),
        "map_of_intervals": MapType(StringType(), DayTimeIntervalType()),
        "array_of_unnamed_structs": ArrayType(StructType([StructField("", UnsignedShortType())])),
    }


def _first_reads_and_casts(spark, built):
    assert str(UnsignedShortType().duckdb_type) == "USMALLINT"
    assert str(LongType().duckdb_type) == "BIGINT"
    assert str(built["decimal"].duckdb_type) == "DECIMAL(17,4)"
    assert str(built["array_of_intervals"].duckdb_type) == "INTERVAL[]"
    assert str(built["map_of_intervals"].duckdb_type) == "MAP(VARCHAR, INTERVAL)"
    assert str(built["array_of_unnamed_structs"].duckdb_type) == "TUPLE(USMALLINT)[]"
    assert built["array_of_intervals"].duckdb_type is built["array_of_intervals"].duckdb_type
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

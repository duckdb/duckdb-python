"""A spark type read once keeps answering while duckdb's default connection is busy or in a failed transaction.

The DuckDBPyType behind a spark type is parsed on the default connection the first time its name is read; a session
has a connection of its own, so later reads and casts must not depend on the default connection's state.
"""

import threading

import pytest

import duckdb

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.sql import functions as F
from spark_namespace.sql.types import ArrayType, DecimalType, IntegerType, StructField, StructType

pytestmark = pytest.mark.skipif(USE_ACTUAL_SPARK, reason="pyspark has no duckdb default connection")


def _cast_and_build(spark):
    schema = StructType([StructField("a", IntegerType()), StructField("b", ArrayType(IntegerType()))])
    assert str(IntegerType().duckdb_type) == "INTEGER"
    assert str(DecimalType(10, 2).duckdb_type) == "DECIMAL(10,2)"
    assert str(F.col("a").cast(IntegerType())) == "Column<'CAST(a AS INT)'>"
    assert spark.createDataFrame([(2, [3])], schema).collect()[0].b == [3]


def test_reads_and_casts_while_the_default_connection_runs_a_query_on_another_thread(spark):
    _cast_and_build(spark)
    gate, started = threading.Event(), threading.Event()
    duckdb.create_function("wait_for_gate", lambda x: (gate.wait(), x)[1], [int], int)
    try:

        def occupy():
            started.set()
            duckdb.sql("select wait_for_gate(1)").fetchall()

        worker = threading.Thread(target=occupy)
        worker.start()
        started.wait()
        with pytest.raises(duckdb.ResourceInUseException):
            duckdb.sql("select 1").fetchall()
        _cast_and_build(spark)
    finally:
        gate.set()
        worker.join()
        duckdb.remove_function("wait_for_gate")


def test_reads_and_casts_after_a_statement_failed_in_the_default_connections_transaction(spark):
    _cast_and_build(spark)
    duckdb.execute("BEGIN")
    try:
        with pytest.raises(duckdb.Error):
            duckdb.execute("SELECT error('boom')")
        with pytest.raises(duckdb.TransactionException):
            duckdb.sql("select 1").fetchall()
        _cast_and_build(spark)
    finally:
        duckdb.execute("ROLLBACK")

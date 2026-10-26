from . import USE_ACTUAL_SPARK

if USE_ACTUAL_SPARK:
    from pyspark.errors import *
else:
    from duckdb.experimental.spark.errors import *


def error_condition(error: Exception) -> str | None:
    """The error condition of a Spark exception; Spark 4 deprecates getErrorClass, and Spark 3.5 has only it."""
    return error.getCondition() if hasattr(error, "getCondition") else error.getErrorClass()

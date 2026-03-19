from .. import USE_ACTUAL_SPARK

if USE_ACTUAL_SPARK:
    from pyspark.sql.window import *
else:
    from duckdb.experimental.spark.sql.window import *

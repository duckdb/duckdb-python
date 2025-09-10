from .. import USE_ACTUAL_SPARK

if USE_ACTUAL_SPARK:
    from pyspark.sql.functions import * # noqa F403
else:
    from duckdb.experimental.spark.sql.functions import * # noqa F403

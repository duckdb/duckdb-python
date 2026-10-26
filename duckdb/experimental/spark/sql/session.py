import datetime
import math
import uuid
from collections.abc import Iterable
from functools import reduce
from typing import TYPE_CHECKING, Any, NoReturn, Union, cast

import duckdb
from duckdb import ColumnExpression

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame as PandasDataFrame

    from .catalog import Catalog


from ..conf import SparkConf
from ..context import SparkContext
from ..errors import IllegalArgumentException, PySparkTypeError, PySparkValueError
from ..exception import ContributionsAcceptedError
from .column import _is_numpy
from .conf import RuntimeConfig
from .dataframe import DataFrame, _source_frame
from .readwriter import DataFrameReader
from .streaming import DataStreamReader
from .types import (
    ArrayType,
    DataType,
    DecimalType,
    FractionalType,
    IntegralType,
    MapType,
    StructField,
    StructType,
    TimestampMillisecondNTZType,
    TimestampNanosecondNTZType,
    TimestampNanosecondType,
    TimestampNTZType,
    TimestampSecondNTZType,
    TimestampType,
    _has_nulltype,
    _infer_schema,
    _make_type_verifier,
    _merge_type,
    _parse_datatype_string,
    _python_to_duckdb,
    _replace_types,
)
from .udf import UDFRegistration

# In spark:
# SparkSession holds a SparkContext
# SparkContext gets created from SparkConf
# At this level the check is made to determine whether the instance already exists and just needs
# to be retrieved or it needs to be created.

# For us this is done inside of `duckdb.connect`, based on the passed in path + configuration
# SparkContext can be compared to our Connection class, and SparkConf to our ClientContext class


def _sql_value(value: Any, data_type: DataType, parameters: list[Any]) -> str:  # noqa: ANN401
    """Returns SQL for ``value`` and appends the parameters it refers to."""
    if value is None:
        return "NULL"
    nested = isinstance(data_type, (ArrayType, MapType, StructType))
    # duckdb.Value gives DuckDB the type of a nested value, which it would read wrongly from a dict.
    parameters.append(duckdb.Value(value, data_type.duckdb_type) if nested else value)
    return "?"


# Spark reads every pandas datetime column as a timestamp, a naive one as a wall clock time in the session time zone.
_PANDAS_TIMESTAMPS = {
    TimestampNTZType: TimestampType(),
    TimestampMillisecondNTZType: TimestampType(),
    TimestampNanosecondNTZType: TimestampType(),
    TimestampSecondNTZType: TimestampType(),
    TimestampNanosecondType: TimestampType(),
}


def _is_missing(value: Any) -> bool:  # noqa: ANN401
    """Whether ``value`` is None or a pandas missing value, which DuckDB reads as NULL."""
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or type(value).__name__ in ("NAType", "NaTType")


def _python_value(value: Any) -> Any:  # noqa: ANN401
    """A value of a pandas column as the Python value DuckDB reads, at any depth, which Spark's verifier can check.

    A pandas missing value is None, and NumPy values and pandas timestamps and durations are Python values.
    """
    if _is_numpy(value):
        value = value.tolist()
    if _is_missing(value):
        return None
    if isinstance(value, datetime.datetime) and hasattr(value, "to_pydatetime"):
        return value.to_pydatetime(warn=False)
    if isinstance(value, datetime.timedelta) and hasattr(value, "to_pytimedelta"):
        return value.to_pytimedelta()
    if isinstance(value, dict):
        return {_python_value(k): _python_value(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(map(_python_value, value))
    return value


def _verify_pandas_values(data: "PandasDataFrame", schema: StructType) -> None:
    """Raise where Spark rejects a value of ``data`` for ``schema``, which it reads through the rows of ``data``.

    A missing value is None in those rows, except in a float column, whose missing values are NaN. Spark keeps a NaN
    or another pandas missing value as a value, but DuckDB reads it as NULL, so it counts as None. Spark 4.2.0 takes
    a float column for an integral field only where every value is whole or, in a nullable field, missing; it never
    takes one for a decimal field.
    """
    for (_, series), field in zip(data.items(), schema.fields, strict=False):
        verify = _make_type_verifier(field.dataType, field.nullable, name=f"field {field.name}")
        if series.dtype.kind == "f":
            missing = series.isna()
            if isinstance(field.dataType, IntegralType):
                rejected = series[(~missing & (series % 1 != 0)) | (missing & (not field.nullable))]
            elif isinstance(field.dataType, DecimalType):
                rejected = series
            elif isinstance(field.dataType, FractionalType):
                if missing.any():
                    verify(None)
                continue
            else:
                if missing.any():
                    verify(None)
                rejected = series[~missing]
            if len(rejected):
                verify(float(rejected.iloc[0]))
        elif series.dtype == object:
            for value in series:
                verify(_python_value(value))
        elif series.isna().any():
            verify(None)


# The engine's return types of a statement that returns no result set of its own: a command, or a changed row count.
_NO_RESULT_TYPES = frozenset({"nothing", "changed_rows"})


def _returned_no_result(relation: duckdb.DuckDBPyRelation) -> bool:
    """Whether ``relation`` is what a statement that returns no result set left, rather than rows of its own."""
    # The return type the engine gave the statement, not the shape of what came back: RETURNING can name a column
    # Count, and a real result set of one BIGINT column called Count is rows like any other.
    return relation._return_type in _NO_RESULT_TYPES


class SparkSession:  # noqa: D101
    def __init__(self, context: SparkContext) -> None:  # noqa: D107
        self.conn = context.connection
        self._context = context
        self._conf = RuntimeConfig(self.conn)

    def _create_dataframe(self, data: list[Any], schema: StructType, names: list[str] | None = None) -> DataFrame:
        if not schema.fields:
            msg = "DuckDB can not create a DataFrame without columns"
            raise ContributionsAcceptedError(msg)
        column_names = list(schema.names)
        for i, name in enumerate(names or []):
            column_names[i] = name
        convert = _python_to_duckdb(schema)
        rows = [convert(row) for row in data] or [tuple(None for _ in schema.fields)]
        parameters: list[Any] = []
        values = ", ".join(
            "("
            + ", ".join(_sql_value(v, f.dataType, parameters) for v, f in zip(row, schema.fields, strict=True))
            + ")"
            for row in rows
        )
        rel = self.conn.sql(f"select * from (values {values})", params=parameters)
        rel = rel.project(
            *[
                ColumnExpression(column).cast(field.dataType.duckdb_type).alias(name)
                for column, field, name in zip(rel.columns, schema.fields, column_names, strict=True)
            ]
        )
        if not data:
            rel = rel.limit(0)
        return DataFrame(rel, self)

    def _inferSchemaFromList(self, data: list[Any], names: list[str] | None = None) -> StructType:
        if not data:
            raise PySparkValueError(error_class="CANNOT_INFER_EMPTY_SCHEMA", message_parameters={})
        schema = reduce(_merge_type, (_infer_schema(row, names) for row in data))
        if _has_nulltype(schema):
            raise PySparkValueError(error_class="CANNOT_DETERMINE_TYPE", message_parameters={})
        return schema

    def _createDataFrameFromPandas(
        self, data: "PandasDataFrame", types: list[str] | None, names: list[str] | None
    ) -> DataFrame:
        unique_name = f"pyspark_pandas_df_{uuid.uuid1()}"
        self.conn.register(unique_name, data)
        df = DataFrame(self.conn.sql(f'select * from "{unique_name}"'), self)

        # Cast to types
        if types:
            df = df._cast_types(*types)
        else:
            df = df._with_types(cast("StructType", _replace_types(df.schema, _PANDAS_TIMESTAMPS)))
        # Alias to names
        if names:
            df = df.toDF(*names)
        return df

    def createDataFrame(  # noqa: D102
        self,
        data: Union["PandasDataFrame", Iterable[Any]],
        schema: DataType | str | list[str] | None = None,
        samplingRatio: float | None = None,
        verifySchema: bool = True,
    ) -> DataFrame:
        if samplingRatio:
            raise NotImplementedError

        if isinstance(data, DataFrame):
            raise PySparkTypeError(
                error_class="SHOULD_NOT_DATAFRAME",
                message_parameters={"arg_name": "data"},
            )

        if isinstance(schema, str):
            schema = _parse_datatype_string(schema)
        elif isinstance(schema, (list, tuple)):
            schema = list(schema)
        elif schema is not None and not isinstance(schema, DataType):
            raise PySparkTypeError(
                error_class="NOT_LIST_OR_NONE_OR_STRUCT",
                message_parameters={"arg_name": "schema", "arg_type": type(schema).__name__},
            )

        try:
            import pandas

            has_pandas = True
        except ImportError:
            has_pandas = False
        if has_pandas and isinstance(data, pandas.DataFrame):
            if isinstance(schema, StructType):
                if verifySchema:
                    _verify_pandas_values(data, schema)
                types, names = schema.extract_types_and_names()
                return self._createDataFrameFromPandas(data, types, names)
            if isinstance(schema, DataType):
                msg = "A pandas DataFrame with a schema that is not a StructType is not supported"
                raise ContributionsAcceptedError(msg)
            return self._createDataFrameFromPandas(data, None, schema)

        if not isinstance(data, list):
            data = list(data)
        if isinstance(schema, DataType) and verifySchema:
            verify = _make_type_verifier(schema, name=None if isinstance(schema, StructType) else "field value")
            for obj in data:
                verify(obj)
        if isinstance(schema, StructType):
            return self._create_dataframe(data, schema)
        if isinstance(schema, DataType):
            return self._create_dataframe([(obj,) for obj in data], StructType([StructField("value", schema)]))
        return self._create_dataframe(data, self._inferSchemaFromList(data, names=schema), names=schema)

    def newSession(self) -> "SparkSession":  # noqa: D102
        return SparkSession(self._context)

    def range(  # noqa: D102
        self,
        start: int,
        end: int | None = None,
        step: int = 1,
        numPartitions: int | None = None,
    ) -> "DataFrame":
        if step == 0:
            msg = "requirement failed: step (0) cannot be 0"
            raise IllegalArgumentException(msg)
        if end is None:
            end = start
            start = 0
        # numPartitions only splits Spark's rows across partitions; it does not change them.
        relation = self.conn.table_function("range", parameters=[start, end, step])
        return _source_frame(relation.select(ColumnExpression("range").alias("id")), self)

    def sql(self, sqlQuery: str, **kwargs: Any) -> DataFrame:  # noqa: D102, ANN401
        if kwargs:
            raise NotImplementedError
        relation = self.conn.sql(sqlQuery)
        # Spark gives a statement without a result a DataFrame without columns, not the engine's count or flag.
        if relation is not None and _returned_no_result(relation):
            relation = None
        return DataFrame(relation, self)

    def stop(self) -> None:  # noqa: D102
        self._context.stop()

    def table(self, tableName: str) -> DataFrame:  # noqa: D102
        relation = self.conn.table(tableName)
        return DataFrame(relation, self)

    def getActiveSession(self) -> "SparkSession":  # noqa: D102
        return self

    @property
    def catalog(self) -> "Catalog":  # noqa: D102
        if not hasattr(self, "_catalog"):
            from duckdb.experimental.spark.sql.catalog import Catalog

            self._catalog = Catalog(self)
        return self._catalog

    @property
    def conf(self) -> RuntimeConfig:  # noqa: D102
        return self._conf

    @property
    def read(self) -> DataFrameReader:  # noqa: D102
        return DataFrameReader(self)

    @property
    def readStream(self) -> DataStreamReader:  # noqa: D102
        return DataStreamReader(self)

    @property
    def sparkContext(self) -> SparkContext:  # noqa: D102
        return self._context

    @property
    def streams(self) -> NoReturn:  # noqa: D102
        raise ContributionsAcceptedError

    @property
    def udf(self) -> UDFRegistration:  # noqa: D102
        return UDFRegistration(self)

    @property
    def version(self) -> str:  # noqa: D102
        return "1.0.0"

    class Builder:  # noqa: D106
        def __init__(self) -> None:  # noqa: D107
            pass

        def master(self, name: str) -> "SparkSession.Builder":  # noqa: D102
            # no-op
            return self

        def appName(self, name: str) -> "SparkSession.Builder":  # noqa: D102
            # no-op
            return self

        def remote(self, url: str) -> "SparkSession.Builder":  # noqa: D102
            # no-op
            return self

        def getOrCreate(self) -> "SparkSession":  # noqa: D102
            context = SparkContext("__ignored__")
            return SparkSession(context)

        def config(  # noqa: D102
            self,
            key: str | None = None,
            value: Any | None = None,  # noqa: ANN401
            conf: SparkConf | None = None,
        ) -> "SparkSession.Builder":
            return self

        def enableHiveSupport(self) -> "SparkSession.Builder":  # noqa: D102
            # no-op
            return self

    builder = Builder()


__all__ = ["SparkSession"]

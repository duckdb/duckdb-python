# https://sparkbyexamples.com/pyspark/pyspark-udf-user-defined-function/
import contextlib
import datetime
import inspect
import uuid
from typing import TYPE_CHECKING, Any, Optional, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

import duckdb
from duckdb import Expression, FunctionExpression
from duckdb.sqltypes import DuckDBPyType

from ..errors import PySparkTypeError
from ..exception import ContributionsAcceptedError
from ._typing import ColumnOrName
from .column import Column, _call_name, _derived, _duckdb_type
from .functions import _as_column
from .type_utils import convert_type
from .types import DataType, StringType, TimestampType, _duckdb_to_python, _local_time_to_instant

if TYPE_CHECKING:
    from .session import SparkSession

DataTypeOrString = DataType | str
UserDefinedFunctionLike = TypeVar("UserDefinedFunctionLike")


def _timestamp(value: Any) -> Any:  # noqa: ANN401
    # DuckDB cannot convert a naive datetime to a TIMESTAMP WITH TIME ZONE, and reads a date as midnight in UTC.
    if value is None or isinstance(value, datetime.datetime):
        return _local_time_to_instant(value)
    msg = f"a UDF result of type {type(value).__name__} for a timestamp return type"
    raise TypeError(msg)


def _result_reader(data_type: DataType | None) -> "Callable[[Any], Any]":
    """How a UDF's result goes to DuckDB, which converts it to the return type: a naive datetime as a local time."""
    return _timestamp if isinstance(data_type, TimestampType) else lambda v: v


def _spark_type(dtype: DuckDBPyType) -> DataType | None:
    """The Spark type a DuckDB type stands for, or None for a DuckDB type that none stands for, such as an enum."""
    try:
        return convert_type(dtype)
    except (KeyError, ContributionsAcceptedError):
        return None


def _argument_reader(type_text: str) -> "Callable[[Any], Any] | None":
    """How Spark gives a UDF an argument of the DuckDB type ``type_text``: a struct as a Row, also inside others."""
    try:
        data_type = _spark_type(DuckDBPyType(type_text))
    except duckdb.Error:
        return None
    return None if data_type is None else _duckdb_to_python(data_type, aware=True)


def _spark_udf(f: "Callable[..., Any]", data_type: DataType | None) -> "Callable[[Any, Any], Any]":
    """``f`` called with Spark's values of its arguments.

    DuckDB gives a Python function a struct and a map alike as a dict, so the arguments come with their DuckDB types.
    """
    read = _result_reader(data_type)
    readers: dict[str, Callable[[Any], Any] | None] = {}

    def call(types: Any, values: Any) -> Any:  # noqa: ANN401
        arguments = []
        for type_text, value in zip(types, values, strict=True):
            if type_text not in readers:
                readers[type_text] = _argument_reader(type_text)
            reader = readers[type_text]
            arguments.append(value if reader is None or value is None else reader(value))
        return read(f(*arguments))

    return call


def _typed_arguments(arguments: list[Expression]) -> list[Expression]:
    """The arguments of a call to a function ``_spark_udf`` made: the arguments' types, and the arguments."""
    types = FunctionExpression("list_value", *(FunctionExpression("typeof", a) for a in arguments))
    return [types.cast(DuckDBPyType("VARCHAR[]")), FunctionExpression("row", *arguments)]


_POSITIONAL = frozenset(
    [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]
)


def _positional_signature(f: "Callable[..., Any]") -> inspect.Signature:
    """The positional parameters of ``f``, without annotations."""
    parameters = inspect.signature(f).parameters.values()
    return inspect.Signature([p.replace(annotation=p.empty) for p in parameters if p.kind in _POSITIONAL])


def _sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _register_sql_name(
    connection: duckdb.DuckDBPyConnection,
    name: str,
    f: "Callable[..., Any]",
    data_type: DataType | None,
    return_type: DuckDBPyType,
) -> None:
    """Make SQL text call ``f`` by ``name``, as DuckDB calls a Python function: with DuckDB's values, a struct a dict.

    SQL text keeps calling a function of DuckDB or of an extension of that name; a Python function or a temporary
    macro of that name is replaced, as Spark replaces a temporary function.
    """
    connection.execute(f"DROP MACRO IF EXISTS temp.main.{_sql_identifier(name)}")
    with contextlib.suppress(duckdb.InvalidInputException):
        connection.remove_function(name)
    # A failing statement would abort a transaction the session has open, so the catalog tells whether the name is
    # taken, not a call that fails to bind.
    taken = connection.sql(
        "SELECT 1 FROM duckdb_functions() WHERE lower(function_name) = lower(?) "
        "AND function_type IN ('scalar', 'aggregate', 'macro', 'window') AND schema_name IN ('main', 'pg_catalog')",
        params=[name],
    ).fetchall()
    if taken:
        return
    read = _result_reader(data_type)

    def call(*values: Any) -> Any:  # noqa: ANN401
        return read(f(*values))

    # DuckDB reads its parameters from the signature, casting an argument to an annotation, which Spark ignores;
    # without a signature it takes any number of arguments.
    with contextlib.suppress(TypeError, ValueError):
        call.__signature__ = _positional_signature(f)  # type: ignore[attr-defined]
    connection.create_function(name, call, return_type=return_type, null_handling="special")


class UDFRegistration:  # noqa: D101
    def __init__(self, sparkSession: "SparkSession") -> None:  # noqa: D107
        self.sparkSession = sparkSession

    def register(  # noqa: D102
        self,
        name: str,
        f: "Callable[..., Any] | UserDefinedFunctionLike",
        returnType: Optional["DataTypeOrString"] = None,
    ) -> "Callable[..., Column]":
        registered = getattr(f, "_spark_udf_function", None)
        if registered is not None:
            if returnType is not None:
                raise PySparkTypeError(
                    error_class="CANNOT_SPECIFY_RETURN_TYPE_FOR_UDF",
                    message_parameters={"arg_name": "f", "return_type": str(returnType)},
                )
            f, returnType = registered, f.returnType
        if returnType is None:
            returnType = StringType()
        return_type = returnType if isinstance(returnType, DuckDBPyType) else _duckdb_type(returnType, cast=False)
        data_type = returnType if isinstance(returnType, DataType) else _spark_type(return_type)
        connection = self.sparkSession.conn
        # A function for each registration keeps an earlier callable on its own function, as in Spark, and the
        # callable away from a builtin of the same name.
        internal_name = f"spark_udf_{uuid.uuid4().hex}"
        # Spark calls a Python function on NULL arguments too.
        connection.create_function(
            internal_name, _spark_udf(f, data_type), return_type=return_type, null_handling="special"
        )
        _register_sql_name(connection, name, f, data_type, return_type)

        def call(*cols: ColumnOrName) -> Column:
            columns = [_as_column(c) for c in cols]
            expr = FunctionExpression(internal_name, *_typed_arguments([c.expr for c in columns]))
            return _derived(expr, _call_name(name, *columns), *columns)

        call.returnType = returnType if data_type is None else data_type
        call._spark_udf_function = f
        return call

    def registerJavaFunction(  # noqa: D102
        self,
        name: str,
        javaClassName: str,
        returnType: Optional["DataTypeOrString"] = None,
    ) -> None:
        raise NotImplementedError

    def registerJavaUDAF(self, name: str, javaClassName: str) -> None:  # noqa: D102
        raise NotImplementedError


__all__ = ["UDFRegistration"]

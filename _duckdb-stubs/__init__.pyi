import os
import pathlib
import typing
from typing_extensions import Self

from duckdb._relation import DuckDBPyRelation
from ._enums import (
    CSVLineTerminator,
    StatementType,
    ExpectedResultType,
    ExplainType,
    PythonExceptionHandling,
    RenderMode,
    token_type,
)

if typing.TYPE_CHECKING:
    import fsspec
    import numpy as np
    import polars
    import pandas
    import pyarrow.lib
    from builtins import list as lst
    from collections.abc import Callable, Iterable, Sequence, Mapping
    from ._typing import (
        PythonLiteral,
        IntoValues,
        IntoPyType,
        IntoFields,
        StrIntoPyType,
        JsonCompression,
        JsonFormat,
        JsonRecordOptions,
        CsvEncoding,
        CsvCompression,
        HiveTypes,
        ColumnsTypes,
        ProfilerFormat,
        ParquetCompression,
        ArrowUDF,
    )
    from ._enums import ExplainTypeLiteral, RenderModeLiteral
    from duckdb import sqltypes, func

__all__: lst[str] = [
    "Appender",
    "BinderException",
    "CSVLineTerminator",
    "CatalogException",
    "ConnectionException",
    "ConstraintException",
    "ConversionException",
    "DataError",
    "DatabaseError",
    "DependencyException",
    "DuckDBPyConnection",
    "Error",
    "ExpectedResultType",
    "ExplainType",
    "FatalException",
    "HTTPException",
    "IOException",
    "IntegrityError",
    "InternalError",
    "InternalException",
    "InterruptException",
    "InvalidInputException",
    "InvalidTypeException",
    "NotImplementedException",
    "NotSupportedError",
    "OperationalError",
    "OutOfMemoryException",
    "OutOfRangeException",
    "ParserException",
    "PermissionException",
    "ProgrammingError",
    "PythonExceptionHandling",
    "RenderMode",
    "ResourceInUseException",
    "SequenceException",
    "SerializationException",
    "Statement",
    "StatementType",
    "SyntaxException",
    "TransactionException",
    "TypeMismatchException",
    "Warning",
    "abort",
    "apilevel",
    "append",
    "appender",
    "array_type",
    "begin",
    "checkpoint",
    "close",
    "commit",
    "complete",
    "connect",
    "create_function",
    "cursor",
    "decimal_type",
    "default_connection",
    "description",
    "df",
    "disable_profiling",
    "dtype",
    "duplicate",
    "enable_profiling",
    "enum_type",
    "execute",
    "executemany",
    "extract_statements",
    "fetch_df",
    "fetch_df_chunk",
    "fetchall",
    "fetchdf",
    "fetchmany",
    "fetchnumpy",
    "fetchone",
    "filesystem_is_registered",
    "get_profiling_information",
    "get_table_names",
    "install_extension",
    "interrupt",
    "list_filesystems",
    "list_type",
    "load_extension",
    "map_type",
    "paramstyle",
    "query_appender",
    "query_progress",
    "register",
    "register_filesystem",
    "remove_function",
    "rollback",
    "row_type",
    "rowcount",
    "set_default_connection",
    "sqltype",
    "string_type",
    "struct_type",
    "tf",
    "threadsafety",
    "token_type",
    "tokenize",
    "torch",
    "type",
    "union_type",
    "unregister",
    "unregister_filesystem",
]

class Appender:
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def append_row(self, *values: typing.Any) -> None: ...
    def append_rows(self, rows: Iterable[Sequence[typing.Any]]) -> None: ...
    def clear(self) -> None: ...
    def flush(self) -> None: ...

class BinderException(ProgrammingError): ...
class CatalogException(ProgrammingError): ...
class ConnectionException(OperationalError): ...
class ConstraintException(IntegrityError): ...
class ConversionException(DataError): ...
class DataError(DatabaseError): ...
class DatabaseError(Error): ...
class DependencyException(DatabaseError): ...

class DuckDBPyConnection:
    def __del__(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def append(self, table_name: str, df: pandas.DataFrame, *, by_name: bool = False) -> DuckDBPyConnection: ...
    def appender(
        self,
        table: str,
        *,
        columns: lst[str] | None = None,
        action: typing.Literal["insert", "upsert", "update", "delete"] = "insert",
        key: lst[str] | None = None,
    ) -> Appender: ...
    def array_type(self, type: IntoPyType, size: typing.SupportsInt) -> sqltypes.DuckDBPyType: ...
    def begin(self) -> DuckDBPyConnection: ...
    def checkpoint(self) -> DuckDBPyConnection: ...
    def close(self) -> None: ...
    def commit(self) -> DuckDBPyConnection: ...
    @typing.overload
    def create_function(
        self,
        name: str,
        function: Callable[..., PythonLiteral],
        parameters: lst[IntoPyType] | None = None,
        return_type: IntoPyType | None = None,
        *,
        type: func.PythonUDFType = func.PythonUDFType.NATIVE,
        null_handling: func.FunctionNullHandling = ...,
        exception_handling: PythonExceptionHandling = ...,
        side_effects: bool = False,
    ) -> DuckDBPyConnection: ...
    @typing.overload
    def create_function(
        self,
        name: str,
        function: ArrowUDF,
        parameters: lst[IntoPyType] | None = None,
        return_type: IntoPyType | None = None,
        *,
        type: func.PythonUDFType = func.PythonUDFType.ARROW,
        null_handling: func.FunctionNullHandling = ...,
        exception_handling: PythonExceptionHandling = ...,
        side_effects: bool = False,
    ) -> DuckDBPyConnection: ...
    def cursor(self) -> DuckDBPyConnection: ...
    def decimal_type(self, width: typing.SupportsInt, scale: typing.SupportsInt) -> sqltypes.DuckDBPyType: ...
    def df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def dtype(self, type_str: StrIntoPyType) -> sqltypes.DuckDBPyType: ...
    def duplicate(self) -> DuckDBPyConnection: ...
    def enum_type(self, name: str, type: sqltypes.DuckDBPyType, values: lst[typing.Any]) -> sqltypes.DuckDBPyType: ...
    def execute(self, query: Statement | str, parameters: object = None) -> DuckDBPyConnection: ...
    def executemany(self, query: Statement | str, parameters: object = None) -> DuckDBPyConnection: ...
    def extract_statements(self, query: str) -> lst[Statement]: ...
    def fetch_df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def fetch_df_chunk(
        self, vectors_per_chunk: typing.SupportsInt = 1, *, date_as_object: bool = False
    ) -> pandas.DataFrame: ...
    def fetchall(self) -> lst[tuple[typing.Any, ...]]: ...
    def fetchdf(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def fetchmany(self, size: typing.SupportsInt = 1) -> lst[tuple[typing.Any, ...]]: ...
    def fetchnumpy(self) -> dict[str, np.typing.NDArray[typing.Any] | pandas.Categorical]: ...
    def fetchone(self) -> tuple[typing.Any, ...] | None: ...
    def filesystem_is_registered(self, name: str) -> bool: ...
    def from_arrow(self, arrow_object: object) -> DuckDBPyRelation: ...
    def from_csv_auto(
        self,
        path_or_buffer: str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str],
        header: bool | int | None = None,
        compression: CsvCompression | None = None,
        sep: str | None = None,
        delimiter: str | None = None,
        files_to_sniff: int | None = None,
        comment: str | None = None,
        thousands: str | None = None,
        dtype: IntoFields | None = None,
        na_values: str | lst[str] | None = None,
        skiprows: int | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        encoding: CsvEncoding | None = None,
        parallel: bool | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        sample_size: int | None = None,
        auto_detect: bool | int | None = None,
        all_varchar: bool | None = None,
        normalize_names: bool | None = None,
        null_padding: bool | None = None,
        names: lst[str] | None = None,
        lineterminator: CSVLineTerminator | None = None,
        columns: ColumnsTypes | None = None,
        auto_type_candidates: lst[StrIntoPyType] | None = None,
        max_line_size: int | None = None,
        ignore_errors: bool | None = None,
        store_rejects: bool | None = None,
        rejects_table: str | None = None,
        rejects_scan: str | None = None,
        rejects_limit: int | None = None,
        force_not_null: lst[str] | None = None,
        buffer_size: int | None = None,
        decimal: str | None = None,
        allow_quoted_nulls: bool | None = None,
        filename: bool | str | None = None,
        hive_partitioning: bool | None = None,
        union_by_name: bool | None = None,
        hive_types: HiveTypes | None = None,
        hive_types_autocast: bool | None = None,
        strict_mode: bool | None = None,
    ) -> DuckDBPyRelation: ...
    def from_df(self, df: pandas.DataFrame) -> DuckDBPyRelation: ...
    def from_parquet(
        self,
        path_or_buffer: str
        | bytes
        | os.PathLike[str]
        | os.PathLike[bytes]
        | typing.IO[bytes]
        | typing.IO[str]
        | Sequence[str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str]],
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: ParquetCompression | None = None,
    ) -> DuckDBPyRelation: ...
    def from_query(self, query: str, *, alias: str = "", params: object = None) -> DuckDBPyRelation: ...
    def get_table_names(self, query: str, *, qualified: bool = False) -> set[str]: ...
    def install_extension(
        self,
        extension: str,
        *,
        force_install: bool = False,
        repository: str | None = None,
        repository_url: str | None = None,
        version: str | None = None,
    ) -> None: ...
    def get_profiling_information(self, format: ProfilerFormat = "json") -> str: ...
    def enable_profiling(self) -> None: ...
    def disable_profiling(self) -> None: ...
    def interrupt(self) -> None: ...
    def complete(self) -> None: ...
    def abort(self) -> None: ...
    def list_filesystems(self) -> lst[str]: ...
    def list_type(self, type: IntoPyType) -> sqltypes.DuckDBPyType: ...
    def load_extension(self, extension: str) -> None: ...
    def map_type(self, key: IntoPyType, value: IntoPyType) -> sqltypes.DuckDBPyType: ...
    def query(self, query: str, *, alias: str = "", params: object = None) -> DuckDBPyRelation: ...
    def query_appender(self, dml: str, columns: Mapping[str, IntoPyType], *, source: str | None = None) -> Appender: ...
    def query_progress(self) -> float: ...
    def read_csv(
        self,
        path_or_buffer: str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str],
        header: bool | int | None = None,
        compression: CsvCompression | None = None,
        sep: str | None = None,
        delimiter: str | None = None,
        files_to_sniff: int | None = None,
        comment: str | None = None,
        thousands: str | None = None,
        dtype: IntoFields | None = None,
        na_values: str | lst[str] | None = None,
        skiprows: int | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        encoding: CsvEncoding | None = None,
        parallel: bool | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        sample_size: int | None = None,
        auto_detect: bool | int | None = None,
        all_varchar: bool | None = None,
        normalize_names: bool | None = None,
        null_padding: bool | None = None,
        names: lst[str] | None = None,
        lineterminator: CSVLineTerminator | None = None,
        columns: ColumnsTypes | None = None,
        auto_type_candidates: lst[StrIntoPyType] | None = None,
        max_line_size: int | None = None,
        ignore_errors: bool | None = None,
        store_rejects: bool | None = None,
        rejects_table: str | None = None,
        rejects_scan: str | None = None,
        rejects_limit: int | None = None,
        force_not_null: lst[str] | None = None,
        buffer_size: int | None = None,
        decimal: str | None = None,
        allow_quoted_nulls: bool | None = None,
        filename: bool | str | None = None,
        hive_partitioning: bool | None = None,
        union_by_name: bool | None = None,
        hive_types: HiveTypes | None = None,
        hive_types_autocast: bool | None = None,
        strict_mode: bool | None = None,
    ) -> DuckDBPyRelation: ...
    def read_json(
        self,
        path_or_buffer: str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str],
        *,
        columns: ColumnsTypes | None = None,
        sample_size: int | None = None,
        maximum_depth: int | None = None,
        records: JsonRecordOptions | None = None,
        format: JsonFormat | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        compression: JsonCompression | None = None,
        maximum_object_size: int | None = None,
        ignore_errors: bool | None = None,
        convert_strings_to_integers: bool | None = None,
        field_appearance_threshold: float | None = None,
        map_inference_threshold: int | None = None,
        maximum_sample_files: int | None = None,
        filename: bool | str | None = None,
        hive_partitioning: bool | None = None,
        union_by_name: bool | None = None,
        hive_types: HiveTypes | None = None,
        hive_types_autocast: bool | None = None,
    ) -> DuckDBPyRelation: ...
    def read_parquet(
        self,
        path_or_buffer: str
        | bytes
        | os.PathLike[str]
        | os.PathLike[bytes]
        | typing.IO[bytes]
        | typing.IO[str]
        | Sequence[str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str]],
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: ParquetCompression | None = None,
    ) -> DuckDBPyRelation: ...
    def register(self, view_name: str, python_object: object) -> DuckDBPyConnection: ...
    def _register_arrow_source(self, view_name: str, source: object) -> DuckDBPyConnection: ...
    @typing.overload
    def _bind(
        self,
        sql: str | Statement,
        *,
        capture: bool = False,
        with_type: bool = False,
        with_writes: typing.Literal[True],
        values: lst[object] | dict[str, object] | None = None,
    ) -> tuple[typing.Any, ...]: ...
    @typing.overload
    def _bind(
        self, sql: str, *, capture: typing.Literal[False] = False, with_type: typing.Literal[False] = False
    ) -> tuple[lst[str], lst[sqltypes.DuckDBPyType], lst[tuple[str, sqltypes.DuckDBPyType]], str, bool]: ...
    @typing.overload
    def _bind(
        self, sql: str, *, capture: typing.Literal[True], with_type: typing.Literal[False] = False
    ) -> tuple[
        lst[str],
        lst[sqltypes.DuckDBPyType],
        lst[tuple[str, sqltypes.DuckDBPyType]],
        str,
        bool,
        lst[tuple[str, SourceHandle]],
    ]: ...
    @typing.overload
    def _bind(
        self, sql: str, *, capture: typing.Literal[False] = False, with_type: typing.Literal[True]
    ) -> tuple[lst[str], lst[sqltypes.DuckDBPyType], lst[tuple[str, sqltypes.DuckDBPyType]], str, bool, str]: ...
    @typing.overload
    def _bind(
        self, sql: str, *, capture: typing.Literal[True], with_type: typing.Literal[True]
    ) -> tuple[
        lst[str],
        lst[sqltypes.DuckDBPyType],
        lst[tuple[str, sqltypes.DuckDBPyType]],
        str,
        bool,
        lst[tuple[str, SourceHandle]],
        str,
    ]: ...
    def _statement_count(self, sql: str) -> int: ...
    def _register_values(self, rows: lst[tuple[object, ...]]) -> SourceHandle: ...
    def _check_text(self, text: str, shape: str) -> int: ...
    @typing.overload
    def _prepare_script(
        self,
        sql: str,
        values: lst[object] | dict[str, object] | None = None,
        *,
        with_statement: typing.Literal[False] = False,
    ) -> tuple[RelationResult | None, str | None, str | None]: ...
    @typing.overload
    def _prepare_script(
        self, sql: str, values: lst[object] | dict[str, object] | None = None, *, with_statement: typing.Literal[True]
    ) -> tuple[RelationResult | None, str | None, str | None, Statement | None]: ...
    def _submit(
        self,
        sql: str,
        values: lst[object] | dict[str, object] | None = None,
        format: typing.Literal["rows", "arrow", "stream"] = "rows",
        batch_size: int = 131072,
    ) -> RelationResult: ...
    def _result_as_arrow(
        self, batch_size: typing.SupportsInt, *, reader: bool
    ) -> pyarrow.lib.Table | pyarrow.lib.RecordBatchReader | None: ...
    def _register_source(self, obj: object, name: str | None = None) -> SourceHandle: ...
    def _register_collection(self, result: RelationResult) -> SourceHandle: ...
    def _register_map(
        self,
        function: Callable[[pandas.DataFrame], pandas.DataFrame],
        schema: dict[str, sqltypes.DuckDBPyType] | None,
        input_sql: str,
    ) -> SourceHandle: ...
    def _register_filelike(self, obj: object) -> SourceHandle: ...
    def _own_sources(self, name: str, handles: lst[SourceHandle]) -> None: ...
    def _catalog_version(self) -> int: ...
    def _create_view(
        self, name: str | tuple[str, ...] | lst[str], sql: str, *, temporary: bool = False, replace: bool = False
    ) -> None: ...
    def _table_kind(self, name: str | tuple[str, ...] | lst[str]) -> str: ...
    def _name_scope(self, name: str) -> str: ...
    def register_filesystem(self, filesystem: fsspec.AbstractFileSystem) -> None: ...
    def remove_function(self, name: str) -> DuckDBPyConnection: ...
    def rollback(self) -> DuckDBPyConnection: ...
    def row_type(self, fields: IntoFields) -> sqltypes.DuckDBPyType: ...
    def sql(self, query: Statement | str, *, alias: str = "", params: object = None) -> DuckDBPyRelation: ...
    def sqltype(self, type_str: str) -> sqltypes.DuckDBPyType: ...
    def string_type(self, collation: str = "") -> sqltypes.DuckDBPyType: ...
    def struct_type(self, fields: IntoFields) -> sqltypes.DuckDBPyType: ...
    def table(self, table_name: str) -> DuckDBPyRelation: ...
    def table_function(self, name: str, parameters: object = None) -> DuckDBPyRelation: ...
    def tf(self) -> dict[str, typing.Any]: ...
    def to_arrow_reader(self, batch_size: typing.SupportsInt = 131072) -> pyarrow.lib.RecordBatchReader | None: ...
    def to_arrow_table(self, batch_size: typing.SupportsInt = 131072) -> pyarrow.lib.Table | None: ...
    def torch(self) -> dict[str, typing.Any]: ...
    def type(self, type_str: str) -> sqltypes.DuckDBPyType: ...
    def union_type(self, members: IntoFields) -> sqltypes.DuckDBPyType: ...
    def unregister(self, view_name: str) -> DuckDBPyConnection: ...
    def unregister_filesystem(self, name: str) -> None: ...
    def values(self, *args: IntoValues) -> DuckDBPyRelation: ...
    def view(self, view_name: str) -> DuckDBPyRelation: ...
    @property
    def description(self) -> lst[tuple[str, sqltypes.DuckDBPyType, None, None, None, None, None]]: ...
    @property
    def rowcount(self) -> int: ...

class Error(Exception): ...
class FatalException(DatabaseError): ...

class HTTPException(IOException):
    status_code: int
    body: str
    reason: str
    headers: dict[str, str]

class IOException(OperationalError): ...
class IntegrityError(DatabaseError): ...
class InternalError(DatabaseError): ...
class InternalException(InternalError): ...
class InterruptException(DatabaseError): ...
class InvalidInputException(ProgrammingError): ...
class InvalidTypeException(ProgrammingError): ...
class NotImplementedException(NotSupportedError): ...
class NotSupportedError(DatabaseError): ...
class OperationalError(DatabaseError): ...
class OutOfMemoryException(OperationalError): ...
class OutOfRangeException(DataError): ...
class ParserException(ProgrammingError): ...
class PermissionException(DatabaseError): ...
class ProgrammingError(DatabaseError): ...
class ResourceInUseException(OperationalError): ...
class SequenceException(DatabaseError): ...
class SerializationException(OperationalError): ...

class RelationResult:
    def fetchall(self) -> lst[tuple[typing.Any, ...]]: ...
    def fetchmany(self, size: int = 1) -> lst[tuple[typing.Any, ...]]: ...
    def fetchnumpy(self) -> dict[str, np.typing.NDArray[typing.Any] | pandas.Categorical]: ...
    def df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def torch(self) -> dict[str, typing.Any]: ...
    def tf(self) -> dict[str, typing.Any]: ...
    def arrow_table(self, *, to_polars: bool = False) -> pyarrow.lib.Table: ...
    def arrow_reader(self) -> pyarrow.lib.RecordBatchReader: ...
    def arrow_capsule(self) -> typing.Any: ...
    @property
    def description(self) -> lst[tuple[str, sqltypes.DuckDBPyType, None, None, None, None, None]]: ...
    @property
    def rowcount(self) -> int: ...
    @property
    def return_type(self) -> typing.Literal["query_result", "changed_rows", "nothing"]: ...
    def to_box(
        self,
        max_width: int | None = None,
        max_rows: int | None = None,
        *,
        max_col_width: int | None = None,
        null_value: str | None = None,
        render_mode: RenderMode | RenderModeLiteral | None = None,
    ) -> str: ...
    def drain(self) -> None: ...
    def close(self) -> None: ...

class SourceHandle:
    @property
    def name(self) -> str: ...

class Statement:
    @property
    def expected_result_type(self) -> lst[StatementType]: ...
    @property
    def named_parameters(self) -> set[str]: ...
    @property
    def query(self) -> str: ...
    @property
    def type(self) -> StatementType: ...

class SyntaxException(ProgrammingError): ...
class TransactionException(OperationalError): ...
class TypeMismatchException(DataError): ...
class Warning(Exception): ...

def append(
    table_name: str, df: pandas.DataFrame, *, by_name: bool = False, connection: DuckDBPyConnection | None = None
) -> DuckDBPyConnection: ...
def appender(
    table: str,
    *,
    columns: lst[str] | None = None,
    action: typing.Literal["insert", "upsert", "update", "delete"] = "insert",
    key: lst[str] | None = None,
    connection: DuckDBPyConnection | None = None,
) -> Appender: ...
def array_type(
    type: IntoPyType, size: typing.SupportsInt, *, connection: DuckDBPyConnection | None = None
) -> sqltypes.DuckDBPyType: ...
def begin(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def checkpoint(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def close(*, connection: DuckDBPyConnection | None = None) -> None: ...
def commit(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def connect(
    database: str | pathlib.Path = ":memory:",
    read_only: bool = False,
    config: dict[str, str | bool | int | float | lst[str]] | None = None,
) -> DuckDBPyConnection: ...
@typing.overload
def create_function(
    name: str,
    function: Callable[..., PythonLiteral],
    parameters: lst[IntoPyType] | None = None,
    return_type: IntoPyType | None = None,
    *,
    type: func.PythonUDFType = func.PythonUDFType.NATIVE,
    null_handling: func.FunctionNullHandling = ...,
    exception_handling: PythonExceptionHandling = ...,
    side_effects: bool = False,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
@typing.overload
def create_function(
    name: str,
    function: ArrowUDF,
    parameters: lst[IntoPyType] | None = None,
    return_type: IntoPyType | None = None,
    *,
    type: func.PythonUDFType = func.PythonUDFType.ARROW,
    null_handling: func.FunctionNullHandling = ...,
    exception_handling: PythonExceptionHandling = ...,
    side_effects: bool = False,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
def cursor(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def decimal_type(
    width: typing.SupportsInt, scale: typing.SupportsInt, *, connection: DuckDBPyConnection | None = None
) -> sqltypes.DuckDBPyType: ...
def default_connection() -> DuckDBPyConnection: ...
def description(
    *, connection: DuckDBPyConnection | None = None
) -> lst[tuple[str, sqltypes.DuckDBPyType, None, None, None, None, None]] | None: ...
def df(*, date_as_object: bool = False, connection: DuckDBPyConnection | None = None) -> pandas.DataFrame: ...
def dtype(type_str: StrIntoPyType, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def duplicate(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def enum_type(
    name: str,
    type: sqltypes.DuckDBPyType,
    values: lst[typing.Any],
    *,
    connection: DuckDBPyConnection | None = None,
) -> sqltypes.DuckDBPyType: ...
def execute(
    query: Statement | str,
    parameters: object = None,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
def executemany(
    query: Statement | str,
    parameters: object = None,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
def extract_statements(query: str, *, connection: DuckDBPyConnection | None = None) -> lst[Statement]: ...
def fetch_df(*, date_as_object: bool = False, connection: DuckDBPyConnection | None = None) -> pandas.DataFrame: ...
def fetch_df_chunk(
    vectors_per_chunk: typing.SupportsInt = 1,
    *,
    date_as_object: bool = False,
    connection: DuckDBPyConnection | None = None,
) -> pandas.DataFrame: ...
def fetchall(*, connection: DuckDBPyConnection | None = None) -> lst[tuple[typing.Any, ...]]: ...
def fetchdf(*, date_as_object: bool = False, connection: DuckDBPyConnection | None = None) -> pandas.DataFrame: ...
def fetchmany(
    size: typing.SupportsInt = 1, *, connection: DuckDBPyConnection | None = None
) -> lst[tuple[typing.Any, ...]]: ...
def fetchnumpy(
    *, connection: DuckDBPyConnection | None = None
) -> dict[str, np.typing.NDArray[typing.Any] | pandas.Categorical]: ...
def fetchone(*, connection: DuckDBPyConnection | None = None) -> tuple[typing.Any, ...] | None: ...
def filesystem_is_registered(name: str, *, connection: DuckDBPyConnection | None = None) -> bool: ...
def get_table_names(
    query: str, *, qualified: bool = False, connection: DuckDBPyConnection | None = None
) -> set[str]: ...
def install_extension(
    extension: str,
    *,
    force_install: bool = False,
    repository: str | None = None,
    repository_url: str | None = None,
    version: str | None = None,
    connection: DuckDBPyConnection | None = None,
) -> None: ...
def interrupt(*, connection: DuckDBPyConnection | None = None) -> None: ...
def complete(*, connection: DuckDBPyConnection | None = None) -> None: ...
def abort(*, connection: DuckDBPyConnection | None = None) -> None: ...
def get_profiling_information(
    *, connection: DuckDBPyConnection | None = None, format: ProfilerFormat = "json"
) -> str: ...
def enable_profiling(*, connection: DuckDBPyConnection | None = None) -> None: ...
def disable_profiling(*, connection: DuckDBPyConnection | None = None) -> None: ...
def list_filesystems(*, connection: DuckDBPyConnection | None = None) -> lst[str]: ...
def list_type(type: IntoPyType, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def load_extension(extension: str, *, connection: DuckDBPyConnection | None = None) -> None: ...
def map_type(
    key: IntoPyType, value: IntoPyType, *, connection: DuckDBPyConnection | None = None
) -> sqltypes.DuckDBPyType: ...
def query_appender(
    dml: str,
    columns: Mapping[str, IntoPyType],
    *,
    source: str | None = None,
    connection: DuckDBPyConnection | None = None,
) -> Appender: ...
def query_progress(*, connection: DuckDBPyConnection | None = None) -> float: ...
def register(
    view_name: str,
    python_object: object,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
def register_filesystem(
    filesystem: fsspec.AbstractFileSystem, *, connection: DuckDBPyConnection | None = None
) -> None: ...
def remove_function(name: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def rollback(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def row_type(fields: IntoFields, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def rowcount(*, connection: DuckDBPyConnection | None = None) -> int: ...
def set_default_connection(connection: DuckDBPyConnection) -> None: ...
def sqltype(type_str: str, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def string_type(collation: str = "", *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def struct_type(fields: IntoFields, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def tf(*, connection: DuckDBPyConnection | None = None) -> dict[str, typing.Any]: ...
def _set_arrow_sources(sources: object) -> None: ...
def _literal_type(value: object) -> str: ...
def _value_literal(value: object) -> str: ...
def _collation(type: sqltypes.DuckDBPyType) -> str | None: ...
def _arrow_schema(
    names: lst[str], types: lst[sqltypes.DuckDBPyType | str], *, connection: DuckDBPyConnection | None = None
) -> pyarrow.lib.Schema: ...
def _temporal_literal(value: object) -> str: ...
def _parse_expressions(text: str) -> lst[str]: ...
def _column_reference(text: str) -> lst[str] | None: ...
def _translate_predicate(
    connection: DuckDBPyConnection,
    predicate: str,
    columns: object = None,
    schema: object = None,
    select: str = "*",
) -> lst[tuple[str, object | None]]: ...
def tokenize(query: str) -> lst[tuple[int, token_type]]: ...
def torch(*, connection: DuckDBPyConnection | None = None) -> dict[str, typing.Any]: ...
def type(type_str: str, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def union_type(members: IntoFields, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def unregister(view_name: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def unregister_filesystem(name: str, *, connection: DuckDBPyConnection | None = None) -> None: ...

__formatted_python_version__: str
__git_revision__: str
__interactive__: bool
__jupyter__: bool
__standard_vector_size__: int
__version__: str
_clean_default_connection: typing.Any  # value = <capsule object>
apilevel: str
paramstyle: str
threadsafety: int

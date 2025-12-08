import os
import pathlib
import typing as pytyping
from typing_extensions import Self

if pytyping.TYPE_CHECKING:
    import fsspec
    import numpy as np
    import polars
    import pandas
    import pyarrow.lib
    import torch as pytorch
    import tensorflow
    from collections.abc import Callable, Sequence, Mapping
    from duckdb import sqltypes, func

    # the field_ids argument to to_parquet and write_parquet has a recursive structure
    ParquetFieldIdsType = Mapping[str, pytyping.Union[int, "ParquetFieldIdsType"]]

__all__: list[str] = [
    "BinderException",
    "CSVLineTerminator",
    "CaseExpression",
    "CatalogException",
    "CoalesceOperator",
    "ColumnExpression",
    "ConnectionException",
    "ConstantExpression",
    "ConstraintException",
    "ConversionException",
    "DataError",
    "DatabaseError",
    "DefaultExpression",
    "DependencyException",
    "DuckDBPyConnection",
    "DuckDBPyRelation",
    "Error",
    "ExpectedResultType",
    "ExplainType",
    "Expression",
    "FatalException",
    "FunctionExpression",
    "HTTPException",
    "IOException",
    "IntegrityError",
    "InternalError",
    "InternalException",
    "InterruptException",
    "InvalidInputException",
    "InvalidTypeException",
    "LambdaExpression",
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
    "SQLExpression",
    "SequenceException",
    "SerializationException",
    "StarExpression",
    "Statement",
    "StatementType",
    "SyntaxException",
    "TransactionException",
    "TypeMismatchException",
    "Warning",
    "aggregate",
    "alias",
    "apilevel",
    "append",
    "array_type",
    "arrow",
    "begin",
    "checkpoint",
    "close",
    "commit",
    "connect",
    "create_function",
    "cursor",
    "decimal_type",
    "default_connection",
    "description",
    "df",
    "distinct",
    "dtype",
    "duplicate",
    "enum_type",
    "execute",
    "executemany",
    "extract_statements",
    "fetch_arrow_table",
    "fetch_df",
    "fetch_df_chunk",
    "fetch_record_batch",
    "fetchall",
    "fetchdf",
    "fetchmany",
    "fetchnumpy",
    "fetchone",
    "filesystem_is_registered",
    "filter",
    "from_arrow",
    "from_csv_auto",
    "from_df",
    "from_parquet",
    "from_query",
    "get_table_names",
    "install_extension",
    "interrupt",
    "limit",
    "list_filesystems",
    "list_type",
    "load_extension",
    "map_type",
    "order",
    "paramstyle",
    "pl",
    "project",
    "query",
    "query_df",
    "query_progress",
    "read_csv",
    "read_json",
    "read_parquet",
    "register",
    "register_filesystem",
    "remove_function",
    "rollback",
    "row_type",
    "rowcount",
    "set_default_connection",
    "sql",
    "sqltype",
    "string_type",
    "struct_type",
    "table",
    "table_function",
    "tf",
    "threadsafety",
    "token_type",
    "tokenize",
    "torch",
    "type",
    "union_type",
    "unregister",
    "unregister_filesystem",
    "values",
    "view",
    "write_csv",
]

class BinderException(ProgrammingError): ...

class CSVLineTerminator:
    CARRIAGE_RETURN_LINE_FEED: pytyping.ClassVar[
        CSVLineTerminator
    ]  # value = <CSVLineTerminator.CARRIAGE_RETURN_LINE_FEED: 1>
    LINE_FEED: pytyping.ClassVar[CSVLineTerminator]  # value = <CSVLineTerminator.LINE_FEED: 0>
    __members__: pytyping.ClassVar[
        dict[str, CSVLineTerminator]
    ]  # value = {'LINE_FEED': <CSVLineTerminator.LINE_FEED: 0>, 'CARRIAGE_RETURN_LINE_FEED': <CSVLineTerminator.CARRIAGE_RETURN_LINE_FEED: 1>}  # noqa: E501
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: pytyping.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: pytyping.SupportsInt) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

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
    def array_type(self, type: sqltypes.DuckDBPyType, size: pytyping.SupportsInt) -> sqltypes.DuckDBPyType: ...
    def arrow(self, rows_per_batch: pytyping.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader: ...
    def begin(self) -> DuckDBPyConnection: ...
    def checkpoint(self) -> DuckDBPyConnection: ...
    def close(self) -> None: ...
    def commit(self) -> DuckDBPyConnection: ...
    def create_function(
        self,
        name: str,
        function: Callable[..., pytyping.Any],
        parameters: list[sqltypes.DuckDBPyType] | None = None,
        return_type: sqltypes.DuckDBPyType | None = None,
        *,
        type: func.PythonUDFType = ...,
        null_handling: func.FunctionNullHandling = ...,
        exception_handling: PythonExceptionHandling = ...,
        side_effects: bool = False,
    ) -> DuckDBPyConnection: ...
    def cursor(self) -> DuckDBPyConnection: ...
    def decimal_type(self, width: pytyping.SupportsInt, scale: pytyping.SupportsInt) -> sqltypes.DuckDBPyType: ...
    def df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def dtype(self, type_str: str) -> sqltypes.DuckDBPyType: ...
    def duplicate(self) -> DuckDBPyConnection: ...
    def enum_type(
        self, name: str, type: sqltypes.DuckDBPyType, values: list[pytyping.Any]
    ) -> sqltypes.DuckDBPyType: ...
    def execute(self, query: Statement | str, parameters: object = None) -> DuckDBPyConnection: ...
    def executemany(self, query: Statement | str, parameters: object = None) -> DuckDBPyConnection: ...
    def extract_statements(self, query: str) -> list[Statement]: ...
    def fetch_arrow_table(self, rows_per_batch: pytyping.SupportsInt = 1000000) -> pyarrow.lib.Table: ...
    def fetch_df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def fetch_df_chunk(
        self, vectors_per_chunk: pytyping.SupportsInt = 1, *, date_as_object: bool = False
    ) -> pandas.DataFrame: ...
    def fetch_record_batch(self, rows_per_batch: pytyping.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader: ...
    def fetchall(self) -> list[tuple[pytyping.Any, ...]]: ...
    def fetchdf(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def fetchmany(self, size: pytyping.SupportsInt = 1) -> list[tuple[pytyping.Any, ...]]: ...
    def fetchnumpy(self) -> dict[str, np.typing.NDArray[pytyping.Any] | pandas.Categorical]: ...
    def fetchone(self) -> tuple[pytyping.Any, ...] | None: ...
    def filesystem_is_registered(self, name: str) -> bool: ...
    def from_arrow(self, arrow_object: object) -> DuckDBPyRelation: ...
    def from_csv_auto(
        self,
        path_or_buffer: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        header: bool | int | None = None,
        compression: str | None = None,
        sep: str | None = None,
        delimiter: str | None = None,
        files_to_sniff: int | None = None,
        comment: str | None = None,
        thousands: str | None = None,
        dtype: dict[str, str] | list[str] | None = None,
        na_values: str | list[str] | None = None,
        skiprows: int | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        encoding: str | None = None,
        parallel: bool | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        sample_size: int | None = None,
        auto_detect: bool | int | None = None,
        all_varchar: bool | None = None,
        normalize_names: bool | None = None,
        null_padding: bool | None = None,
        names: list[str] | None = None,
        lineterminator: str | None = None,
        columns: dict[str, str] | None = None,
        auto_type_candidates: list[str] | None = None,
        max_line_size: int | None = None,
        ignore_errors: bool | None = None,
        store_rejects: bool | None = None,
        rejects_table: str | None = None,
        rejects_scan: str | None = None,
        rejects_limit: int | None = None,
        force_not_null: list[str] | None = None,
        buffer_size: int | None = None,
        decimal: str | None = None,
        allow_quoted_nulls: bool | None = None,
        filename: bool | str | None = None,
        hive_partitioning: bool | None = None,
        union_by_name: bool | None = None,
        hive_types: dict[str, str] | None = None,
        hive_types_autocast: bool | None = None,
        strict_mode: bool | None = None,
    ) -> DuckDBPyRelation: ...
    def from_df(self, df: pandas.DataFrame) -> DuckDBPyRelation: ...
    @pytyping.overload
    def from_parquet(
        self,
        file_glob: str,
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: str | None = None,
    ) -> DuckDBPyRelation: ...
    @pytyping.overload
    def from_parquet(
        self,
        file_globs: Sequence[str],
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: str | None = None,
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
    def interrupt(self) -> None: ...
    def list_filesystems(self) -> list[str]: ...
    def list_type(self, type: sqltypes.DuckDBPyType) -> sqltypes.DuckDBPyType: ...
    def load_extension(self, extension: str) -> None: ...
    def map_type(self, key: sqltypes.DuckDBPyType, value: sqltypes.DuckDBPyType) -> sqltypes.DuckDBPyType: ...
    @pytyping.overload
    def pl(
        self, rows_per_batch: pytyping.SupportsInt = 1000000, *, lazy: pytyping.Literal[False] = ...
    ) -> polars.DataFrame: ...
    @pytyping.overload
    def pl(
        self, rows_per_batch: pytyping.SupportsInt = 1000000, *, lazy: pytyping.Literal[True]
    ) -> polars.LazyFrame: ...
    @pytyping.overload
    def pl(
        self, rows_per_batch: pytyping.SupportsInt = 1000000, *, lazy: bool = False
    ) -> pytyping.Union[polars.DataFrame, polars.LazyFrame]: ...
    def query(self, query: str, *, alias: str = "", params: object = None) -> DuckDBPyRelation: ...
    def query_progress(self) -> float: ...
    def read_csv(
        self,
        path_or_buffer: str | bytes | os.PathLike[str],
        header: bool | int | None = None,
        compression: str | None = None,
        sep: str | None = None,
        delimiter: str | None = None,
        files_to_sniff: int | None = None,
        comment: str | None = None,
        thousands: str | None = None,
        dtype: dict[str, str] | list[str] | None = None,
        na_values: str | list[str] | None = None,
        skiprows: int | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        encoding: str | None = None,
        parallel: bool | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        sample_size: int | None = None,
        auto_detect: bool | int | None = None,
        all_varchar: bool | None = None,
        normalize_names: bool | None = None,
        null_padding: bool | None = None,
        names: list[str] | None = None,
        lineterminator: str | None = None,
        columns: dict[str, str] | None = None,
        auto_type_candidates: list[str] | None = None,
        max_line_size: int | None = None,
        ignore_errors: bool | None = None,
        store_rejects: bool | None = None,
        rejects_table: str | None = None,
        rejects_scan: str | None = None,
        rejects_limit: int | None = None,
        force_not_null: list[str] | None = None,
        buffer_size: int | None = None,
        decimal: str | None = None,
        allow_quoted_nulls: bool | None = None,
        filename: bool | str | None = None,
        hive_partitioning: bool | None = None,
        union_by_name: bool | None = None,
        hive_types: dict[str, str] | None = None,
        hive_types_autocast: bool | None = None,
        strict_mode: bool | None = None,
    ) -> DuckDBPyRelation: ...
    def read_json(
        self,
        path_or_buffer: str | bytes | os.PathLike[str],
        *,
        columns: dict[str, str] | None = None,
        sample_size: int | None = None,
        maximum_depth: int | None = None,
        records: str | None = None,
        format: str | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        compression: str | None = None,
        maximum_object_size: int | None = None,
        ignore_errors: bool | None = None,
        convert_strings_to_integers: bool | None = None,
        field_appearance_threshold: float | None = None,
        map_inference_threshold: int | None = None,
        maximum_sample_files: int | None = None,
        filename: bool | str | None = None,
        hive_partitioning: bool | None = None,
        union_by_name: bool | None = None,
        hive_types: dict[str, str] | None = None,
        hive_types_autocast: bool | None = None,
    ) -> DuckDBPyRelation: ...
    @pytyping.overload
    def read_parquet(
        self,
        file_glob: str,
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: str | None = None,
    ) -> DuckDBPyRelation: ...
    @pytyping.overload
    def read_parquet(
        self,
        file_globs: Sequence[str],
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: pytyping.Any = None,
    ) -> DuckDBPyRelation: ...
    def register(self, view_name: str, python_object: object) -> DuckDBPyConnection: ...
    def register_filesystem(self, filesystem: fsspec.AbstractFileSystem) -> None: ...
    def remove_function(self, name: str) -> DuckDBPyConnection: ...
    def rollback(self) -> DuckDBPyConnection: ...
    def row_type(
        self, fields: dict[str, sqltypes.DuckDBPyType] | list[sqltypes.DuckDBPyType]
    ) -> sqltypes.DuckDBPyType: ...
    def sql(self, query: Statement | str, *, alias: str = "", params: object = None) -> DuckDBPyRelation: ...
    def sqltype(self, type_str: str) -> sqltypes.DuckDBPyType: ...
    def string_type(self, collation: str = "") -> sqltypes.DuckDBPyType: ...
    def struct_type(
        self, fields: dict[str, sqltypes.DuckDBPyType] | list[sqltypes.DuckDBPyType]
    ) -> sqltypes.DuckDBPyType: ...
    def table(self, table_name: str) -> DuckDBPyRelation: ...
    def table_function(self, name: str, parameters: object = None) -> DuckDBPyRelation: ...
    def tf(self) -> dict[str, tensorflow.Tensor]: ...
    def torch(self) -> dict[str, pytorch.Tensor]: ...
    def type(self, type_str: str) -> sqltypes.DuckDBPyType: ...
    def union_type(
        self, members: list[sqltypes.DuckDBPyType] | dict[str, sqltypes.DuckDBPyType]
    ) -> sqltypes.DuckDBPyType: ...
    def unregister(self, view_name: str) -> DuckDBPyConnection: ...
    def unregister_filesystem(self, name: str) -> None: ...
    def values(self, *args: list[pytyping.Any] | tuple[Expression, ...] | Expression) -> DuckDBPyRelation: ...
    def view(self, view_name: str) -> DuckDBPyRelation: ...
    @property
    def description(self) -> list[tuple[str, sqltypes.DuckDBPyType, None, None, None, None, None]]: ...
    @property
    def rowcount(self) -> int: ...

class DuckDBPyRelation:
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> pytyping.Any: ...
    def __contains__(self, name: str) -> bool: ...
    def __getattr__(self, name: str) -> DuckDBPyRelation: ...
    def __getitem__(self, name: str) -> DuckDBPyRelation: ...
    def __len__(self) -> int: ...
    def aggregate(self, aggr_expr: Expression | str, group_expr: Expression | str = "") -> DuckDBPyRelation: ...
    def any_value(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def apply(
        self,
        function_name: str,
        function_aggr: str,
        group_expr: str = "",
        function_parameter: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def arg_max(
        self, arg_column: str, value_column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def arg_min(
        self, arg_column: str, value_column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def arrow(self, batch_size: pytyping.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader: ...
    def avg(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def bit_and(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def bit_or(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def bit_xor(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def bitstring_agg(
        self,
        column: str,
        min: int | None = None,
        max: int | None = None,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def bool_and(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def bool_or(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def close(self) -> None: ...
    def count(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def create(self, table_name: str) -> None: ...
    def create_view(self, view_name: str, replace: bool = True) -> DuckDBPyRelation: ...
    def cross(self, other_rel: DuckDBPyRelation) -> DuckDBPyRelation: ...
    def cume_dist(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    def dense_rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    def describe(self) -> DuckDBPyRelation: ...
    def df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def distinct(self) -> DuckDBPyRelation: ...
    def except_(self, other_rel: DuckDBPyRelation) -> DuckDBPyRelation: ...
    def execute(self) -> DuckDBPyRelation: ...
    def explain(self, type: ExplainType = ExplainType.STANDARD) -> str: ...
    def favg(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def fetch_arrow_reader(self, batch_size: pytyping.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader: ...
    def fetch_arrow_table(self, batch_size: pytyping.SupportsInt = 1000000) -> pyarrow.lib.Table: ...
    def fetch_df_chunk(
        self, vectors_per_chunk: pytyping.SupportsInt = 1, *, date_as_object: bool = False
    ) -> pandas.DataFrame: ...
    def fetch_record_batch(self, rows_per_batch: pytyping.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader: ...
    def fetchall(self) -> list[tuple[pytyping.Any, ...]]: ...
    def fetchdf(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def fetchmany(self, size: pytyping.SupportsInt = 1) -> list[tuple[pytyping.Any, ...]]: ...
    def fetchnumpy(self) -> dict[str, np.typing.NDArray[pytyping.Any] | pandas.Categorical]: ...
    def fetchone(self) -> tuple[pytyping.Any, ...] | None: ...
    def filter(self, filter_expr: Expression | str) -> DuckDBPyRelation: ...
    def first(self, column: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation: ...
    def first_value(self, column: str, window_spec: str = "", projected_columns: str = "") -> DuckDBPyRelation: ...
    def fsum(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def geomean(self, column: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation: ...
    def histogram(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def insert(self, values: pytyping.List[object]) -> None: ...
    def insert_into(self, table_name: str) -> None: ...
    def intersect(self, other_rel: DuckDBPyRelation) -> DuckDBPyRelation: ...
    def join(
        self, other_rel: DuckDBPyRelation, condition: Expression | str, how: str = "inner"
    ) -> DuckDBPyRelation: ...
    def lag(
        self,
        column: str,
        window_spec: str,
        offset: pytyping.SupportsInt = 1,
        default_value: str = "NULL",
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def last(self, column: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation: ...
    def last_value(self, column: str, window_spec: str = "", projected_columns: str = "") -> DuckDBPyRelation: ...
    def lead(
        self,
        column: str,
        window_spec: str,
        offset: pytyping.SupportsInt = 1,
        default_value: str = "NULL",
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def limit(self, n: pytyping.SupportsInt, offset: pytyping.SupportsInt = 0) -> DuckDBPyRelation: ...
    def list(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def map(
        self, map_function: Callable[..., pytyping.Any], *, schema: dict[str, sqltypes.DuckDBPyType] | None = None
    ) -> DuckDBPyRelation: ...
    def max(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def mean(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def median(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def min(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def mode(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def n_tile(
        self, window_spec: str, num_buckets: pytyping.SupportsInt, projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def nth_value(
        self,
        column: str,
        window_spec: str,
        offset: pytyping.SupportsInt,
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def order(self, order_expr: str) -> DuckDBPyRelation: ...
    def percent_rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    @pytyping.overload
    def pl(
        self, batch_size: pytyping.SupportsInt = 1000000, *, lazy: pytyping.Literal[False] = ...
    ) -> polars.DataFrame: ...
    @pytyping.overload
    def pl(self, batch_size: pytyping.SupportsInt = 1000000, *, lazy: pytyping.Literal[True]) -> polars.LazyFrame: ...
    @pytyping.overload
    def pl(
        self, batch_size: pytyping.SupportsInt = 1000000, *, lazy: bool = False
    ) -> pytyping.Union[polars.DataFrame, polars.LazyFrame]: ...
    def product(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def project(self, *args: str | Expression, groups: str = "") -> DuckDBPyRelation: ...
    def quantile(
        self,
        column: str,
        q: float | pytyping.List[float] = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def quantile_cont(
        self,
        column: str,
        q: float | pytyping.List[float] = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def quantile_disc(
        self,
        column: str,
        q: float | pytyping.List[float] = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def query(self, virtual_table_name: str, sql_query: str) -> DuckDBPyRelation: ...
    def rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    def rank_dense(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    def record_batch(self, batch_size: pytyping.SupportsInt = 1000000) -> pyarrow.RecordBatchReader: ...
    def row_number(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    def select(self, *args: str | Expression, groups: str = "") -> DuckDBPyRelation: ...
    def select_dtypes(self, types: pytyping.List[sqltypes.DuckDBPyType | str]) -> DuckDBPyRelation: ...
    def select_types(self, types: pytyping.List[sqltypes.DuckDBPyType | str]) -> DuckDBPyRelation: ...
    def set_alias(self, alias: str) -> DuckDBPyRelation: ...
    def show(
        self,
        *,
        max_width: pytyping.SupportsInt | None = None,
        max_rows: pytyping.SupportsInt | None = None,
        max_col_width: pytyping.SupportsInt | None = None,
        null_value: str | None = None,
        render_mode: RenderMode | None = None,
    ) -> None: ...
    def sort(self, *args: Expression) -> DuckDBPyRelation: ...
    def sql_query(self) -> str: ...
    def std(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def stddev(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def stddev_pop(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def stddev_samp(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def string_agg(
        self, column: str, sep: str = ",", groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def sum(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def tf(self) -> dict[str, tensorflow.Tensor]: ...
    def to_arrow_table(self, batch_size: pytyping.SupportsInt = 1000000) -> pyarrow.lib.Table: ...
    def to_csv(
        self,
        file_name: str,
        *,
        sep: str | None = None,
        na_rep: str | None = None,
        header: bool | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        quoting: str | int | None = None,
        encoding: str | None = None,
        compression: str | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: pytyping.List[str] | None = None,
        write_partition_columns: bool | None = None,
    ) -> None: ...
    def to_df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def to_parquet(
        self,
        file_name: str,
        *,
        compression: str | None = None,
        field_ids: ParquetFieldIdsType | pytyping.Literal["auto"] | None = None,
        row_group_size_bytes: int | str | None = None,
        row_group_size: int | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: pytyping.List[str] | None = None,
        write_partition_columns: bool | None = None,
        append: bool | None = None,
        filename_pattern: str | None = None,
        file_size_bytes: str | int | None = None,
    ) -> None: ...
    def to_table(self, table_name: str) -> None: ...
    def to_view(self, view_name: str, replace: bool = True) -> DuckDBPyRelation: ...
    def torch(self) -> dict[str, pytorch.Tensor]: ...
    def union(self, union_rel: DuckDBPyRelation) -> DuckDBPyRelation: ...
    def unique(self, unique_aggr: str) -> DuckDBPyRelation: ...
    def update(self, set: Expression | str, *, condition: Expression | str | None = None) -> None: ...
    def value_counts(self, column: str, groups: str = "") -> DuckDBPyRelation: ...
    def var(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def var_pop(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def var_samp(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def variance(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def write_csv(
        self,
        file_name: str,
        sep: str | None = None,
        na_rep: str | None = None,
        header: bool | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        quoting: str | int | None = None,
        encoding: str | None = None,
        compression: str | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: pytyping.List[str] | None = None,
        write_partition_columns: bool | None = None,
    ) -> None: ...
    def write_parquet(
        self,
        file_name: str,
        compression: str | None = None,
        field_ids: ParquetFieldIdsType | pytyping.Literal["auto"] | None = None,
        row_group_size_bytes: str | int | None = None,
        row_group_size: int | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: pytyping.List[str] | None = None,
        write_partition_columns: bool | None = None,
        append: bool | None = None,
        filename_pattern: str | None = None,
        file_size_bytes: str | int | None = None,
    ) -> None: ...
    @property
    def alias(self) -> str: ...
    @property
    def columns(self) -> pytyping.List[str]: ...
    @property
    def description(self) -> pytyping.List[tuple[str, sqltypes.DuckDBPyType, None, None, None, None, None]]: ...
    @property
    def dtypes(self) -> pytyping.List[str]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def type(self) -> str: ...
    @property
    def types(self) -> pytyping.List[sqltypes.DuckDBPyType]: ...

class Error(Exception): ...

class ExpectedResultType:
    CHANGED_ROWS: pytyping.ClassVar[ExpectedResultType]  # value = <ExpectedResultType.CHANGED_ROWS: 1>
    NOTHING: pytyping.ClassVar[ExpectedResultType]  # value = <ExpectedResultType.NOTHING: 2>
    QUERY_RESULT: pytyping.ClassVar[ExpectedResultType]  # value = <ExpectedResultType.QUERY_RESULT: 0>
    __members__: pytyping.ClassVar[
        dict[str, ExpectedResultType]
    ]  # value = {'QUERY_RESULT': <ExpectedResultType.QUERY_RESULT: 0>, 'CHANGED_ROWS': <ExpectedResultType.CHANGED_ROWS: 1>, 'NOTHING': <ExpectedResultType.NOTHING: 2>}  # noqa: E501
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: pytyping.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: pytyping.SupportsInt) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ExplainType:
    ANALYZE: pytyping.ClassVar[ExplainType]  # value = <ExplainType.ANALYZE: 1>
    STANDARD: pytyping.ClassVar[ExplainType]  # value = <ExplainType.STANDARD: 0>
    __members__: pytyping.ClassVar[
        dict[str, ExplainType]
    ]  # value = {'STANDARD': <ExplainType.STANDARD: 0>, 'ANALYZE': <ExplainType.ANALYZE: 1>}
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: pytyping.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: pytyping.SupportsInt) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Expression:
    def __add__(self, other: Expression) -> Expression: ...
    def __and__(self, other: Expression) -> Expression: ...
    def __div__(self, other: Expression) -> Expression: ...
    def __eq__(self, other: Expression) -> Expression: ...  # type: ignore[override]
    def __floordiv__(self, other: Expression) -> Expression: ...
    def __ge__(self, other: Expression) -> Expression: ...
    def __gt__(self, other: Expression) -> Expression: ...
    @pytyping.overload
    def __init__(self, arg0: str) -> None: ...
    @pytyping.overload
    def __init__(self, arg0: pytyping.Any) -> None: ...
    def __invert__(self) -> Expression: ...
    def __le__(self, other: Expression) -> Expression: ...
    def __lt__(self, other: Expression) -> Expression: ...
    def __mod__(self, other: Expression) -> Expression: ...
    def __mul__(self, other: Expression) -> Expression: ...
    def __ne__(self, other: Expression) -> Expression: ...  # type: ignore[override]
    def __neg__(self) -> Expression: ...
    def __or__(self, other: Expression) -> Expression: ...
    def __pow__(self, other: Expression) -> Expression: ...
    def __radd__(self, other: Expression) -> Expression: ...
    def __rand__(self, other: Expression) -> Expression: ...
    def __rdiv__(self, other: Expression) -> Expression: ...
    def __rfloordiv__(self, other: Expression) -> Expression: ...
    def __rmod__(self, other: Expression) -> Expression: ...
    def __rmul__(self, other: Expression) -> Expression: ...
    def __ror__(self, other: Expression) -> Expression: ...
    def __rpow__(self, other: Expression) -> Expression: ...
    def __rsub__(self, other: Expression) -> Expression: ...
    def __rtruediv__(self, other: Expression) -> Expression: ...
    def __sub__(self, other: Expression) -> Expression: ...
    def __truediv__(self, other: Expression) -> Expression: ...
    def alias(self, name: str) -> Expression: ...
    def asc(self) -> Expression: ...
    def between(self, lower: Expression, upper: Expression) -> Expression: ...
    def cast(self, type: sqltypes.DuckDBPyType) -> Expression: ...
    def collate(self, collation: str) -> Expression: ...
    def desc(self) -> Expression: ...
    def get_name(self) -> str: ...
    def isin(self, *args: Expression) -> Expression: ...
    def isnotin(self, *args: Expression) -> Expression: ...
    def isnotnull(self) -> Expression: ...
    def isnull(self) -> Expression: ...
    def nulls_first(self) -> Expression: ...
    def nulls_last(self) -> Expression: ...
    def otherwise(self, value: Expression) -> Expression: ...
    def show(self) -> None: ...
    def when(self, condition: Expression, value: Expression) -> Expression: ...

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

class PythonExceptionHandling:
    DEFAULT: pytyping.ClassVar[PythonExceptionHandling]  # value = <PythonExceptionHandling.DEFAULT: 0>
    RETURN_NULL: pytyping.ClassVar[PythonExceptionHandling]  # value = <PythonExceptionHandling.RETURN_NULL: 1>
    __members__: pytyping.ClassVar[
        dict[str, PythonExceptionHandling]
    ]  # value = {'DEFAULT': <PythonExceptionHandling.DEFAULT: 0>, 'RETURN_NULL': <PythonExceptionHandling.RETURN_NULL: 1>}  # noqa: E501
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: pytyping.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: pytyping.SupportsInt) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class RenderMode:
    COLUMNS: pytyping.ClassVar[RenderMode]  # value = <RenderMode.COLUMNS: 1>
    ROWS: pytyping.ClassVar[RenderMode]  # value = <RenderMode.ROWS: 0>
    __members__: pytyping.ClassVar[
        dict[str, RenderMode]
    ]  # value = {'ROWS': <RenderMode.ROWS: 0>, 'COLUMNS': <RenderMode.COLUMNS: 1>}
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: pytyping.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: pytyping.SupportsInt) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SequenceException(DatabaseError): ...
class SerializationException(OperationalError): ...

class Statement:
    @property
    def expected_result_type(self) -> list[StatementType]: ...
    @property
    def named_parameters(self) -> set[str]: ...
    @property
    def query(self) -> str: ...
    @property
    def type(self) -> StatementType: ...

class StatementType:
    ALTER_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.ALTER_STATEMENT: 8>
    ANALYZE_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.ANALYZE_STATEMENT: 11>
    ATTACH_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.ATTACH_STATEMENT: 25>
    CALL_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.CALL_STATEMENT: 19>
    COPY_DATABASE_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.COPY_DATABASE_STATEMENT: 28>
    COPY_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.COPY_STATEMENT: 10>
    CREATE_FUNC_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.CREATE_FUNC_STATEMENT: 13>
    CREATE_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.CREATE_STATEMENT: 4>
    DELETE_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.DELETE_STATEMENT: 5>
    DETACH_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.DETACH_STATEMENT: 26>
    DROP_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.DROP_STATEMENT: 15>
    EXECUTE_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.EXECUTE_STATEMENT: 7>
    EXPLAIN_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.EXPLAIN_STATEMENT: 14>
    EXPORT_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.EXPORT_STATEMENT: 16>
    EXTENSION_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.EXTENSION_STATEMENT: 23>
    INSERT_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.INSERT_STATEMENT: 2>
    INVALID_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.INVALID_STATEMENT: 0>
    LOAD_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.LOAD_STATEMENT: 21>
    LOGICAL_PLAN_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.LOGICAL_PLAN_STATEMENT: 24>
    MERGE_INTO_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.MERGE_INTO_STATEMENT: 30>
    MULTI_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.MULTI_STATEMENT: 27>
    PRAGMA_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.PRAGMA_STATEMENT: 17>
    PREPARE_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.PREPARE_STATEMENT: 6>
    RELATION_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.RELATION_STATEMENT: 22>
    SELECT_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.SELECT_STATEMENT: 1>
    SET_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.SET_STATEMENT: 20>
    TRANSACTION_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.TRANSACTION_STATEMENT: 9>
    UPDATE_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.UPDATE_STATEMENT: 3>
    VACUUM_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.VACUUM_STATEMENT: 18>
    VARIABLE_SET_STATEMENT: pytyping.ClassVar[StatementType]  # value = <StatementType.VARIABLE_SET_STATEMENT: 12>
    __members__: pytyping.ClassVar[
        dict[str, StatementType]
    ]  # value = {'INVALID_STATEMENT': <StatementType.INVALID_STATEMENT: 0>, 'SELECT_STATEMENT': <StatementType.SELECT_STATEMENT: 1>, 'INSERT_STATEMENT': <StatementType.INSERT_STATEMENT: 2>, 'UPDATE_STATEMENT': <StatementType.UPDATE_STATEMENT: 3>, 'CREATE_STATEMENT': <StatementType.CREATE_STATEMENT: 4>, 'DELETE_STATEMENT': <StatementType.DELETE_STATEMENT: 5>, 'PREPARE_STATEMENT': <StatementType.PREPARE_STATEMENT: 6>, 'EXECUTE_STATEMENT': <StatementType.EXECUTE_STATEMENT: 7>, 'ALTER_STATEMENT': <StatementType.ALTER_STATEMENT: 8>, 'TRANSACTION_STATEMENT': <StatementType.TRANSACTION_STATEMENT: 9>, 'COPY_STATEMENT': <StatementType.COPY_STATEMENT: 10>, 'ANALYZE_STATEMENT': <StatementType.ANALYZE_STATEMENT: 11>, 'VARIABLE_SET_STATEMENT': <StatementType.VARIABLE_SET_STATEMENT: 12>, 'CREATE_FUNC_STATEMENT': <StatementType.CREATE_FUNC_STATEMENT: 13>, 'EXPLAIN_STATEMENT': <StatementType.EXPLAIN_STATEMENT: 14>, 'DROP_STATEMENT': <StatementType.DROP_STATEMENT: 15>, 'EXPORT_STATEMENT': <StatementType.EXPORT_STATEMENT: 16>, 'PRAGMA_STATEMENT': <StatementType.PRAGMA_STATEMENT: 17>, 'VACUUM_STATEMENT': <StatementType.VACUUM_STATEMENT: 18>, 'CALL_STATEMENT': <StatementType.CALL_STATEMENT: 19>, 'SET_STATEMENT': <StatementType.SET_STATEMENT: 20>, 'LOAD_STATEMENT': <StatementType.LOAD_STATEMENT: 21>, 'RELATION_STATEMENT': <StatementType.RELATION_STATEMENT: 22>, 'EXTENSION_STATEMENT': <StatementType.EXTENSION_STATEMENT: 23>, 'LOGICAL_PLAN_STATEMENT': <StatementType.LOGICAL_PLAN_STATEMENT: 24>, 'ATTACH_STATEMENT': <StatementType.ATTACH_STATEMENT: 25>, 'DETACH_STATEMENT': <StatementType.DETACH_STATEMENT: 26>, 'MULTI_STATEMENT': <StatementType.MULTI_STATEMENT: 27>, 'COPY_DATABASE_STATEMENT': <StatementType.COPY_DATABASE_STATEMENT: 28>, 'MERGE_INTO_STATEMENT': <StatementType.MERGE_INTO_STATEMENT: 30>}  # noqa: E501
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: pytyping.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: pytyping.SupportsInt) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SyntaxException(ProgrammingError): ...
class TransactionException(OperationalError): ...
class TypeMismatchException(DataError): ...
class Warning(Exception): ...

class token_type:
    __members__: pytyping.ClassVar[
        dict[str, token_type]
    ]  # value = {'identifier': <token_type.identifier: 0>, 'numeric_const': <token_type.numeric_const: 1>, 'string_const': <token_type.string_const: 2>, 'operator': <token_type.operator: 3>, 'keyword': <token_type.keyword: 4>, 'comment': <token_type.comment: 5>}  # noqa: E501
    comment: pytyping.ClassVar[token_type]  # value = <token_type.comment: 5>
    identifier: pytyping.ClassVar[token_type]  # value = <token_type.identifier: 0>
    keyword: pytyping.ClassVar[token_type]  # value = <token_type.keyword: 4>
    numeric_const: pytyping.ClassVar[token_type]  # value = <token_type.numeric_const: 1>
    operator: pytyping.ClassVar[token_type]  # value = <token_type.operator: 3>
    string_const: pytyping.ClassVar[token_type]  # value = <token_type.string_const: 2>
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: pytyping.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: pytyping.SupportsInt) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

def CaseExpression(condition: Expression, value: Expression) -> Expression: ...
def CoalesceOperator(*args: Expression) -> Expression: ...
def ColumnExpression(*args: str) -> Expression: ...
def ConstantExpression(value: Expression | str) -> Expression: ...
def DefaultExpression() -> Expression: ...
def FunctionExpression(function_name: str, *args: Expression) -> Expression: ...
def LambdaExpression(lhs: Expression | str | tuple[str], rhs: Expression) -> Expression: ...
def SQLExpression(expression: str) -> Expression: ...
@pytyping.overload
def StarExpression(*, exclude: Expression | str | tuple[str]) -> Expression: ...
@pytyping.overload
def StarExpression() -> Expression: ...
def aggregate(
    df: pandas.DataFrame,
    aggr_expr: Expression | list[Expression] | str | list[str],
    group_expr: str = "",
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def alias(df: pandas.DataFrame, alias: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def append(
    table_name: str, df: pandas.DataFrame, *, by_name: bool = False, connection: DuckDBPyConnection | None = None
) -> DuckDBPyConnection: ...
def array_type(
    type: sqltypes.DuckDBPyType, size: pytyping.SupportsInt, *, connection: DuckDBPyConnection | None = None
) -> sqltypes.DuckDBPyType: ...
@pytyping.overload
def arrow(
    rows_per_batch: pytyping.SupportsInt = 1000000, *, connection: DuckDBPyConnection | None = None
) -> pyarrow.lib.RecordBatchReader: ...
@pytyping.overload
def arrow(arrow_object: pytyping.Any, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def begin(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def checkpoint(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def close(*, connection: DuckDBPyConnection | None = None) -> None: ...
def commit(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def connect(
    database: str | pathlib.Path = ":memory:",
    read_only: bool = False,
    config: dict[str, str | bool | int | float | list[str]] | None = None,
) -> DuckDBPyConnection: ...
def create_function(
    name: str,
    function: Callable[..., pytyping.Any],
    parameters: list[sqltypes.DuckDBPyType] | None = None,
    return_type: sqltypes.DuckDBPyType | None = None,
    *,
    type: func.PythonUDFType = ...,
    null_handling: func.FunctionNullHandling = ...,
    exception_handling: PythonExceptionHandling = ...,
    side_effects: bool = False,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
def cursor(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def decimal_type(
    width: pytyping.SupportsInt, scale: pytyping.SupportsInt, *, connection: DuckDBPyConnection | None = None
) -> sqltypes.DuckDBPyType: ...
def default_connection() -> DuckDBPyConnection: ...
def description(
    *, connection: DuckDBPyConnection | None = None
) -> list[tuple[str, sqltypes.DuckDBPyType, None, None, None, None, None]] | None: ...
@pytyping.overload
def df(*, date_as_object: bool = False, connection: DuckDBPyConnection | None = None) -> pandas.DataFrame: ...
@pytyping.overload
def df(df: pandas.DataFrame, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def distinct(df: pandas.DataFrame, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def dtype(type_str: str, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def duplicate(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def enum_type(
    name: str,
    type: sqltypes.DuckDBPyType,
    values: list[pytyping.Any],
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
def extract_statements(query: str, *, connection: DuckDBPyConnection | None = None) -> list[Statement]: ...
def fetch_arrow_table(
    rows_per_batch: pytyping.SupportsInt = 1000000, *, connection: DuckDBPyConnection | None = None
) -> pyarrow.lib.Table: ...
def fetch_df(*, date_as_object: bool = False, connection: DuckDBPyConnection | None = None) -> pandas.DataFrame: ...
def fetch_df_chunk(
    vectors_per_chunk: pytyping.SupportsInt = 1,
    *,
    date_as_object: bool = False,
    connection: DuckDBPyConnection | None = None,
) -> pandas.DataFrame: ...
def fetch_record_batch(
    rows_per_batch: pytyping.SupportsInt = 1000000, *, connection: DuckDBPyConnection | None = None
) -> pyarrow.lib.RecordBatchReader: ...
def fetchall(*, connection: DuckDBPyConnection | None = None) -> list[tuple[pytyping.Any, ...]]: ...
def fetchdf(*, date_as_object: bool = False, connection: DuckDBPyConnection | None = None) -> pandas.DataFrame: ...
def fetchmany(
    size: pytyping.SupportsInt = 1, *, connection: DuckDBPyConnection | None = None
) -> list[tuple[pytyping.Any, ...]]: ...
def fetchnumpy(
    *, connection: DuckDBPyConnection | None = None
) -> dict[str, np.typing.NDArray[pytyping.Any] | pandas.Categorical]: ...
def fetchone(*, connection: DuckDBPyConnection | None = None) -> tuple[pytyping.Any, ...] | None: ...
def filesystem_is_registered(name: str, *, connection: DuckDBPyConnection | None = None) -> bool: ...
def filter(
    df: pandas.DataFrame,
    filter_expr: Expression | str,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def from_arrow(
    arrow_object: object,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def from_csv_auto(
    path_or_buffer: str | bytes | os.PathLike[str],
    header: bool | int | None = None,
    compression: str | None = None,
    sep: str | None = None,
    delimiter: str | None = None,
    files_to_sniff: int | None = None,
    comment: str | None = None,
    thousands: str | None = None,
    dtype: dict[str, str] | list[str] | None = None,
    na_values: str | list[str] | None = None,
    skiprows: int | None = None,
    quotechar: str | None = None,
    escapechar: str | None = None,
    encoding: str | None = None,
    parallel: bool | None = None,
    date_format: str | None = None,
    timestamp_format: str | None = None,
    sample_size: int | None = None,
    auto_detect: bool | int | None = None,
    all_varchar: bool | None = None,
    normalize_names: bool | None = None,
    null_padding: bool | None = None,
    names: list[str] | None = None,
    lineterminator: str | None = None,
    columns: dict[str, str] | None = None,
    auto_type_candidates: list[str] | None = None,
    max_line_size: int | None = None,
    ignore_errors: bool | None = None,
    store_rejects: bool | None = None,
    rejects_table: str | None = None,
    rejects_scan: str | None = None,
    rejects_limit: int | None = None,
    force_not_null: list[str] | None = None,
    buffer_size: int | None = None,
    decimal: str | None = None,
    allow_quoted_nulls: bool | None = None,
    filename: bool | str | None = None,
    hive_partitioning: bool | None = None,
    union_by_name: bool | None = None,
    hive_types: dict[str, str] | None = None,
    hive_types_autocast: bool | None = None,
    strict_mode: bool | None = None,
) -> DuckDBPyRelation: ...
def from_df(df: pandas.DataFrame, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
@pytyping.overload
def from_parquet(
    file_glob: str,
    binary_as_string: bool = False,
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: str | None = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
@pytyping.overload
def from_parquet(
    file_globs: Sequence[str],
    binary_as_string: bool = False,
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: pytyping.Any = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def from_query(
    query: Statement | str,
    *,
    alias: str = "",
    params: object = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
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
def limit(
    df: pandas.DataFrame,
    n: pytyping.SupportsInt,
    offset: pytyping.SupportsInt = 0,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def list_filesystems(*, connection: DuckDBPyConnection | None = None) -> list[str]: ...
def list_type(
    type: sqltypes.DuckDBPyType, *, connection: DuckDBPyConnection | None = None
) -> sqltypes.DuckDBPyType: ...
def load_extension(extension: str, *, connection: DuckDBPyConnection | None = None) -> None: ...
def map_type(
    key: sqltypes.DuckDBPyType,
    value: sqltypes.DuckDBPyType,
    *,
    connection: DuckDBPyConnection | None = None,
) -> sqltypes.DuckDBPyType: ...
def order(
    df: pandas.DataFrame, order_expr: str, *, connection: DuckDBPyConnection | None = None
) -> DuckDBPyRelation: ...
@pytyping.overload
def pl(
    rows_per_batch: pytyping.SupportsInt = 1000000,
    *,
    lazy: pytyping.Literal[False] = ...,
    connection: DuckDBPyConnection | None = None,
) -> polars.DataFrame: ...
@pytyping.overload
def pl(
    rows_per_batch: pytyping.SupportsInt = 1000000,
    *,
    lazy: pytyping.Literal[True],
    connection: DuckDBPyConnection | None = None,
) -> polars.LazyFrame: ...
@pytyping.overload
def pl(
    rows_per_batch: pytyping.SupportsInt = 1000000,
    *,
    lazy: bool = False,
    connection: DuckDBPyConnection | None = None,
) -> pytyping.Union[polars.DataFrame, polars.LazyFrame]: ...
def project(
    df: pandas.DataFrame, *args: str | Expression, groups: str = "", connection: DuckDBPyConnection | None = None
) -> DuckDBPyRelation: ...
def query(
    query: Statement | str,
    *,
    alias: str = "",
    params: object = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def query_df(
    df: pandas.DataFrame,
    virtual_table_name: str,
    sql_query: str,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def query_progress(*, connection: DuckDBPyConnection | None = None) -> float: ...
def read_csv(
    path_or_buffer: str | bytes | os.PathLike[str],
    header: bool | int | None = None,
    compression: str | None = None,
    sep: str | None = None,
    delimiter: str | None = None,
    files_to_sniff: int | None = None,
    comment: str | None = None,
    thousands: str | None = None,
    dtype: dict[str, str] | list[str] | None = None,
    na_values: str | list[str] | None = None,
    skiprows: int | None = None,
    quotechar: str | None = None,
    escapechar: str | None = None,
    encoding: str | None = None,
    parallel: bool | None = None,
    date_format: str | None = None,
    timestamp_format: str | None = None,
    sample_size: int | None = None,
    auto_detect: bool | int | None = None,
    all_varchar: bool | None = None,
    normalize_names: bool | None = None,
    null_padding: bool | None = None,
    names: list[str] | None = None,
    lineterminator: str | None = None,
    columns: dict[str, str] | None = None,
    auto_type_candidates: list[str] | None = None,
    max_line_size: int | None = None,
    ignore_errors: bool | None = None,
    store_rejects: bool | None = None,
    rejects_table: str | None = None,
    rejects_scan: str | None = None,
    rejects_limit: int | None = None,
    force_not_null: list[str] | None = None,
    buffer_size: int | None = None,
    decimal: str | None = None,
    allow_quoted_nulls: bool | None = None,
    filename: bool | str | None = None,
    hive_partitioning: bool | None = None,
    union_by_name: bool | None = None,
    hive_types: dict[str, str] | None = None,
    hive_types_autocast: bool | None = None,
    strict_mode: bool | None = None,
) -> DuckDBPyRelation: ...
def read_json(
    path_or_buffer: str | bytes | os.PathLike[str],
    *,
    columns: dict[str, str] | None = None,
    sample_size: int | None = None,
    maximum_depth: int | None = None,
    records: str | None = None,
    format: str | None = None,
    date_format: str | None = None,
    timestamp_format: str | None = None,
    compression: str | None = None,
    maximum_object_size: int | None = None,
    ignore_errors: bool | None = None,
    convert_strings_to_integers: bool | None = None,
    field_appearance_threshold: float | None = None,
    map_inference_threshold: int | None = None,
    maximum_sample_files: int | None = None,
    filename: bool | str | None = None,
    hive_partitioning: bool | None = None,
    union_by_name: bool | None = None,
    hive_types: dict[str, str] | None = None,
    hive_types_autocast: bool | None = None,
) -> DuckDBPyRelation: ...
@pytyping.overload
def read_parquet(
    file_glob: str,
    binary_as_string: bool = False,
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: str | None = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
@pytyping.overload
def read_parquet(
    file_globs: Sequence[str],
    binary_as_string: bool = False,
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: pytyping.Any = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
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
def row_type(
    fields: dict[str, sqltypes.DuckDBPyType] | list[sqltypes.DuckDBPyType],
    *,
    connection: DuckDBPyConnection | None = None,
) -> sqltypes.DuckDBPyType: ...
def rowcount(*, connection: DuckDBPyConnection | None = None) -> int: ...
def set_default_connection(connection: DuckDBPyConnection) -> None: ...
def sql(
    query: Statement | str,
    *,
    alias: str = "",
    params: object = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def sqltype(type_str: str, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def string_type(collation: str = "", *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def struct_type(
    fields: dict[str, sqltypes.DuckDBPyType] | list[sqltypes.DuckDBPyType],
    *,
    connection: DuckDBPyConnection | None = None,
) -> sqltypes.DuckDBPyType: ...
def table(table_name: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def table_function(
    name: str,
    parameters: object = None,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def tf(*, connection: DuckDBPyConnection | None = None) -> dict[str, tensorflow.Tensor]: ...
def tokenize(query: str) -> list[tuple[int, token_type]]: ...
def torch(*, connection: DuckDBPyConnection | None = None) -> dict[str, pytorch.Tensor]: ...
def type(type_str: str, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def union_type(
    members: dict[str, sqltypes.DuckDBPyType] | list[sqltypes.DuckDBPyType],
    *,
    connection: DuckDBPyConnection | None = None,
) -> sqltypes.DuckDBPyType: ...
def unregister(view_name: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def unregister_filesystem(name: str, *, connection: DuckDBPyConnection | None = None) -> None: ...
def values(
    *args: list[pytyping.Any] | tuple[Expression, ...] | Expression, connection: DuckDBPyConnection | None = None
) -> DuckDBPyRelation: ...
def view(view_name: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def write_csv(
    df: pandas.DataFrame,
    filename: str,
    *,
    sep: str | None = None,
    na_rep: str | None = None,
    header: bool | None = None,
    quotechar: str | None = None,
    escapechar: str | None = None,
    date_format: str | None = None,
    timestamp_format: str | None = None,
    quoting: str | int | None = None,
    encoding: str | None = None,
    compression: str | None = None,
    overwrite: bool | None = None,
    per_thread_output: bool | None = None,
    use_tmp_file: bool | None = None,
    partition_by: list[str] | None = None,
    write_partition_columns: bool | None = None,
) -> None: ...

__formatted_python_version__: str
__git_revision__: str
__interactive__: bool
__jupyter__: bool
__standard_vector_size__: int
__version__: str
_clean_default_connection: pytyping.Any  # value = <capsule object>
apilevel: str
paramstyle: str
threadsafety: int

import pathlib
import urllib.parse
import urllib.request
from collections import Counter
from typing import TYPE_CHECKING, Any, cast

from duckdb import CaseExpression, ColumnExpression, ConstantExpression, FunctionExpression, SQLExpression
from duckdb.sqltypes import DuckDBPyType

from ..errors import AnalysisException, IllegalArgumentException, PySparkNotImplementedError, PySparkTypeError
from ..exception import ContributionsAcceptedError
from .column import Column, _quoted_identifier
from .functions import _parse_datetime, _tokenize_datetime_pattern, _translate_datetime_pattern_field
from .type_utils import convert_type
from .types import (
    ArrayType,
    DataType,
    DateType,
    DoubleType,
    FloatType,
    MapType,
    NullType,
    StringType,
    StructField,
    StructType,
    TimeNTZType,
    TimestampNTZType,
    TimestampType,
    UUIDType,
    _parse_datatype_string,
    _replace_types,
)

PrimitiveType = bool | float | int | str
OptionalPrimitiveType = PrimitiveType | None

if TYPE_CHECKING:
    import duckdb
    from duckdb import Expression
    from duckdb.experimental.spark.sql.dataframe import DataFrame
    from duckdb.experimental.spark.sql.session import SparkSession

# The DuckDB strftime code that writes a Spark datetime pattern field, by letter and count, where one is exact.
_STRFTIME_CODES = {
    ("y", 2): "%y",
    ("y", 4): "%Y",
    **{(letter, 2): "%m" for letter in "ML"},
    **{(letter, 3): "%b" for letter in "ML"},
    **{(letter, 4): "%B" for letter in "ML"},
    ("d", 2): "%d",
    ("H", 2): "%H",
    ("h", 2): "%I",
    ("m", 2): "%M",
    ("s", 2): "%S",
    ("S", 3): "%g",
    ("S", 6): "%f",
    ("S", 9): "%n",
    ("a", 1): "%p",
    **{("E", count): "%a" for count in (1, 2, 3)},
    ("E", 4): "%A",
}
# Pattern letters Spark accepts only for formatting, which the translation for parsing rejects.
_FORMATTING_LETTERS = frozenset("EFQq")
# DuckDB's JSON reader detects these types in strings, which Spark's JSON inference leaves strings; a field that is
# only ever NULL is a string to Spark too.
_JSON_STRING_TYPES = {
    cls: StringType() for cls in (DateType, TimeNTZType, TimestampNTZType, TimestampType, UUIDType, NullType)
}
_SAVE_MODES = {
    "append": "append",
    "overwrite": "overwrite",
    "ignore": "ignore",
    "error": "error",
    "errorifexists": "error",
    "default": "error",
}
_FORMATS_BY_EXTENSION = {".csv": "csv", ".tsv": "tsv", ".json": "json", ".jsonl": "json", ".ndjson": "json"}
# DuckDB's default type candidates without BIGNUM, which the module has no type for; wider integers read as doubles.
_CSV_TYPE_CANDIDATES = ["BOOLEAN", "BIGINT", "HUGEINT", "DOUBLE", "TIME", "DATE", "TIMESTAMP", "TIMESTAMPTZ", "VARCHAR"]


def _parse_schema(schema: StructType | str) -> StructType:
    if isinstance(schema, str):
        schema = _parse_datatype_string(schema)
    if not isinstance(schema, StructType):
        raise PySparkTypeError(
            error_class="NOT_STR_OR_STRUCT",
            message_parameters={"arg_name": "schema", "arg_type": type(schema).__name__},
        )
    _check_unique_names(schema.names)
    return schema


def _check_unique_names(names: list[str]) -> None:
    """Raise where two of ``names`` are one name ignoring case, as Spark does for a schema or a written file."""
    seen: set[str] = set()
    for name in names:
        if name.lower() in seen:
            raise AnalysisException(
                error_class="COLUMN_ALREADY_EXISTS", message_parameters={"columnName": f"`{name.lower()}`"}
            )
        seen.add(name.lower())


def _flag(name: str, value: bool | str | None, default: bool = False) -> bool:
    """A boolean option, which Spark takes as a bool or as the text true or false in any case."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise IllegalArgumentException(message=f"{name} flag can be true or false.")


def _positions_by_name(columns: list[str], schema: StructType, ignore_case: bool) -> list[int | None]:
    """For each field of ``schema``, the position of the column of the same name, or None where there is none."""
    positions: dict[str, int] = {}
    for i, column in enumerate(columns):
        positions.setdefault(column.casefold() if ignore_case else column, i)
    return [positions.get(f.name.casefold() if ignore_case else f.name) for f in schema.fields]


def _with_schema(df: "DataFrame", schema: StructType, positions: list[int | None]) -> "DataFrame":
    """``df`` read as ``schema``: each field from the column at its position, or NULL where it has none."""
    from duckdb.experimental.spark.sql.dataframe import _position

    columns = [
        (ConstantExpression(None) if i is None else _position(i)).cast(f.dataType.duckdb_type).alias(f.name)
        for i, f in zip(positions, schema.fields, strict=True)
    ]
    return df._derive(df.relation.select(*columns))


def _format_of(path: str) -> str:
    """The format ``load`` reads a path in without a format: by its extension, otherwise parquet as Spark does."""
    name = path.lower().removesuffix(".gz").removesuffix(".zst")
    return _FORMATS_BY_EXTENSION.get(pathlib.PurePath(name).suffix, "parquet")


def _local_path(path: str) -> pathlib.Path | None:
    """The local file ``path`` names, as DuckDB expands it, or None for a path of another file system."""
    if path.startswith("file:"):
        return pathlib.Path(urllib.request.url2pathname(urllib.parse.urlparse(path).path))
    if "://" in path:
        return None
    return pathlib.Path(path).expanduser()


def _path_exists(session: "SparkSession", path: str) -> bool:
    local = _local_path(path)
    if local is not None:
        return local.exists()
    # A directory of an object store is the objects under its prefix.
    query = "SELECT 1 FROM glob($1) UNION ALL SELECT 1 FROM glob($2) LIMIT 1"
    return bool(session.conn.execute(query, [path, path.rstrip("/") + "/*"]).fetchall())


def _duckdb_path(path: str) -> str:
    """``path`` as DuckDB opens it: a file URL as the local path it names, since DuckDB reads it as a relative path."""
    local = _local_path(path)
    return path if local is None or not path.startswith("file:") else str(local)


def _qualified_path(path: str) -> str:
    """``path`` as Spark prints it in a message, a local file as a file URI."""
    local = _local_path(path)
    return path if local is None else local.absolute().as_uri()


def _should_write(session: "SparkSession", path: str, mode: str | None) -> bool:
    """Whether a writer in save ``mode`` writes ``path``, raising where Spark does when the path exists."""
    if mode is not None and (not isinstance(mode, str) or mode.lower() not in _SAVE_MODES):
        raise AnalysisException(error_class="INVALID_SAVE_MODE", message_parameters={"mode": f'"{mode}"'})
    save_mode = "error" if mode is None else _SAVE_MODES[mode.lower()]
    if not _path_exists(session, path):
        return True
    if save_mode == "error":
        local = _local_path(path)
        location = path if local is None else f"file:{local.absolute()}"
        raise AnalysisException(error_class="PATH_ALREADY_EXISTS", message_parameters={"outputPath": location})
    if save_mode == "append":
        msg = "Appending to an existing path is not supported: DuckDB writes a single file"
        raise ContributionsAcceptedError(msg)
    return save_mode == "overwrite"


def _spark_sql_type(data_type: DataType) -> str:
    """The SQL text of a type in Spark's error messages."""
    if isinstance(data_type, ArrayType):
        return f"ARRAY<{_spark_sql_type(data_type.elementType)}>"
    if isinstance(data_type, MapType):
        return f"MAP<{_spark_sql_type(data_type.keyType)}, {_spark_sql_type(data_type.valueType)}>"
    if isinstance(data_type, StructType):
        fields = (f"{_quoted_identifier(f.name)}: {_spark_sql_type(f.dataType)}" for f in data_type.fields)
        return f"STRUCT<{', '.join(fields)}>"
    return data_type.simpleString().upper()


def _unsupported_csv_type(field: StructField) -> AnalysisException:
    return AnalysisException(
        error_class="UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE",
        message_parameters={
            "format": "CSV",
            "columnName": f"`{field.name}`",
            "columnType": f'"{_spark_sql_type(field.dataType)}"',
        },
    )


def _strftime_format(pattern: str | None) -> str | None:
    """The DuckDB strftime format that writes values as the Spark datetime ``pattern`` formats them."""
    if pattern is None:
        return None
    codes = []
    for token in _tokenize_datetime_pattern(pattern):
        if isinstance(token, str):
            codes.append(token.replace("%", "%%"))
            continue
        code = _STRFTIME_CODES.get(token)
        if code is None:
            if token[0] not in _FORMATTING_LETTERS:
                _translate_datetime_pattern_field(pattern, *token)
            msg = f"Datetime pattern {pattern!r} is not supported for formatting: {token[0] * token[1]!r}"
            raise ContributionsAcceptedError(msg)
        codes.append(code)
    return "".join(codes)


def _csv_dialect(sep: str | None, quote: str | None, escape: str | None) -> dict[str, Any]:
    """The CSV dialect given, with Spark's separator and quote and DuckDB's escape where none is, never sniffed.

    DuckDB's escape is the quote, so a backslash is text and a quote inside quotes is doubled.
    """
    quote = '"' if quote is None else quote
    return {"sep": "," if sep is None else sep, "quotechar": quote, "escapechar": quote if escape is None else escape}


def _try_cast(value: "Expression", target: DuckDBPyType) -> "Expression":
    return SQLExpression(f"TRY_CAST({value} AS {target})")


def _csv_value(
    text: "Expression", field: StructField, patterns: dict[type[DataType], str], specials: tuple[str, str, str]
) -> "Expression":
    """The CSV field ``text`` read as the type of ``field``, NULL where it does not convert, as in Spark.

    A date or timestamp with a Spark datetime pattern in ``patterns`` reads with it, any other type with DuckDB's
    cast. ``specials`` are the texts of NaN, infinity and minus infinity.
    """
    data_type = field.dataType
    kind, target = type(data_type), data_type.duckdb_type
    if kind in (ArrayType, MapType, StructType, NullType):
        raise _unsupported_csv_type(field)
    if kind is StringType:
        return text
    if kind in patterns:
        return _parse_datetime(Column(text), patterns[kind], raise_on_error=False).expr.cast(target)
    if kind not in (FloatType, DoubleType):
        return _try_cast(text, target)
    nan, infinity, minus_infinity = specials
    return (
        CaseExpression(text == ConstantExpression(nan), ConstantExpression("NaN").cast(target))
        .when(text == ConstantExpression(infinity), ConstantExpression("Infinity").cast(target))
        .when(text == ConstantExpression(minus_infinity), ConstantExpression("-Infinity").cast(target))
        .otherwise(_try_cast(text, target))
    )


def _csv_header_names(fields: list[str | None], null_value: str | None) -> list[str]:
    """Spark's column names for the fields of a CSV header line, made unique by their position."""
    counts = Counter(field.lower() for field in fields if field is not None)
    return [
        f"_c{i}" if not field or field == null_value else f"{field}{i}" if counts[field.lower()] > 1 else field
        for i, field in enumerate(fields)
    ]


def _first_csv_line(session: "SparkSession", path: str, options: dict[str, Any]) -> list[str | None] | None:
    """The fields of the first line of the CSV files ``path`` names that is not empty, or None."""
    width = len(session.conn.read_csv(path, auto_detect=True, header=False, all_varchar=True, **options).columns)
    columns = {f"column{i}": "VARCHAR" for i in range(width)}
    # A null string no line contains keeps an empty field apart from a field past the end of the line.
    line_options = {**options, "header": False, "na_values": ["\0"]}
    rows = session.conn.read_csv(path, auto_detect=False, columns=columns, **line_options).limit(1).fetchall()
    if not rows:
        return None
    fields = list(rows[0])
    while fields and fields[-1] is None:
        fields.pop()
    return fields


def _check_csv_header(fields: list[str | None], schema: StructType, path: str) -> None:
    """Raise as Spark does with enforceSchema off where the header does not name the schema's fields in order."""
    header = ["" if field is None else field for field in fields]
    names = schema.names
    source = f"CSV file: {_qualified_path(path)}"
    if len(header) != len(names):
        msg = (
            "Number of column in CSV header is not equal to number of fields in the schema:\n"
            f" Header length: {len(header)}, schema size: {len(names)}\n{source}"
        )
        raise IllegalArgumentException(message=msg)
    for column, name in zip(header, names, strict=True):
        if column.lower() != name.lower():
            msg = (
                "CSV header does not conform to the schema.\n"
                f" Header: {', '.join(header)}\n Schema: {', '.join(names)}\n"
                f"Expected: {name} but found: {column}\n{source}"
            )
            raise IllegalArgumentException(message=msg)


def _spark_json_type(data_type: DataType) -> DataType:
    """The type Spark's JSON inference gives a value DuckDB's JSON reader infers ``data_type`` for."""
    if isinstance(data_type, StructType):
        fields = sorted(data_type.fields, key=lambda f: f.name)
        return StructType([StructField(f.name, _spark_json_type(f.dataType)) for f in fields])
    if isinstance(data_type, ArrayType):
        return ArrayType(_spark_json_type(data_type.elementType))
    return _JSON_STRING_TYPES.get(type(data_type), data_type)


def _read_json(session: "SparkSession", path: str) -> "duckdb.DuckDBPyRelation | None":
    """Read JSON lines with the schema Spark infers from every record: no map type, fields sorted by name.

    None for a file without records. Each record is inferred as one value, so DuckDB raises for keys that differ
    only in case, where it would rename one of them.
    """
    inferred = session.conn.read_json(path, sample_size=-1, map_inference_threshold=-1, records="false")
    (record_type,) = inferred.types
    if record_type.id != "struct":
        if not inferred.limit(1).fetchall():
            return None
        msg = "Reading JSON lines that are not all objects is not supported"
        raise ContributionsAcceptedError(msg)
    schema = _spark_json_type(convert_type(record_type))
    columns = {f.name: str(f.dataType.duckdb_type) for f in cast("StructType", schema).fields}
    return session.conn.read_json(path, columns=columns)


def _read_json_with_schema(session: "SparkSession", path: str, schema: StructType) -> "DataFrame":
    """Read JSON lines as ``schema`` with DuckDB's JSON conversion, NULL where a value does not convert."""
    from duckdb.experimental.spark.sql.dataframe import _position, _source_frame

    # DuckDB's JSON reader reads a timestamp without an offset as UTC, where its cast reads it in the session zone.
    read_types = [_replace_types(f.dataType, {TimestampType: StringType()}) for f in schema.fields]
    columns = {f.name: str(t.duckdb_type) for f, t in zip(schema.fields, read_types, strict=True)}
    rel = session.conn.read_json(path, columns=columns, records="true", ignore_errors=True)
    df = _source_frame(rel, session)
    values = [_try_cast(_position(i), f.dataType.duckdb_type).alias(f.name) for i, f in enumerate(schema.fields)]
    return df._derive(df.relation.select(*values))


def _read_inferred_csv(
    session: "SparkSession", path: str, names: list[str], header: bool, null_value: str, options: dict[str, Any]
) -> "DataFrame":
    """The lines of a CSV file read by DuckDB with the types it detects, named ``names``.

    DuckDB detects the types from a sample of the lines, as its own reader does: a later value its reader cannot read
    as the detected type raises, and a fraction in a column detected as integers is rounded.
    """
    from duckdb.experimental.spark.sql.dataframe import _source_frame

    rel = session.conn.read_csv(
        path,
        # DuckDB's sniffer fails on a header line of another width than the other lines, so it skips it as a line.
        header=False,
        skiprows=1 if header else 0,
        na_values=["", null_value],
        auto_type_candidates=_CSV_TYPE_CANDIDATES,
        **options,
    )
    fields = [ColumnExpression(name) for name in rel.columns]
    values = []
    for i, name in enumerate(names):
        data_type = convert_type(rel.types[i]) if i < len(fields) else StringType()
        # Spark infers a timestamp as its TimestampType, the text read as a local time.
        data_type = TimestampType() if isinstance(data_type, TimestampNTZType) else data_type
        value = fields[i] if i < len(fields) else ConstantExpression(None)
        values.append(value.cast(data_type.duckdb_type).alias(name))
    df = _source_frame(rel, session)
    return df._derive(df.relation.select(*values))


class DataFrameWriter:  # noqa: D101
    def __init__(self, dataframe: "DataFrame") -> None:  # noqa: D107
        self.dataframe = dataframe

    def saveAsTable(self, table_name: str) -> None:  # noqa: D102
        _check_unique_names(self.dataframe.columns)
        self.dataframe.relation.create(table_name)

    def parquet(  # noqa: D102
        self,
        path: str,
        mode: str | None = None,
        partitionBy: str | list[str] | None = None,
        compression: str | None = None,
    ) -> None:
        relation = self.dataframe.relation
        if partitionBy:
            raise NotImplementedError
        _check_unique_names(self.dataframe.columns)
        if _should_write(self.dataframe.session, path, mode):
            relation.write_parquet(_duckdb_path(path), compression=compression)

    def csv(  # noqa: D102
        self,
        path: str,
        mode: str | None = None,
        compression: str | None = None,
        sep: str | None = None,
        quote: str | None = None,
        escape: str | None = None,
        header: bool | str | None = None,
        nullValue: str | None = None,
        escapeQuotes: bool | str | None = None,
        quoteAll: bool | str | None = None,
        dateFormat: str | None = None,
        timestampFormat: str | None = None,
        ignoreLeadingWhiteSpace: bool | str | None = None,
        ignoreTrailingWhiteSpace: bool | str | None = None,
        charToEscapeQuoteEscaping: str | None = None,
        encoding: str | None = None,
        emptyValue: str | None = None,
        lineSep: str | None = None,
    ) -> None:
        if not _flag("escapeQuotes", escapeQuotes, default=True):
            msg = "escapeQuotes=False is not supported: DuckDB quotes every value that contains the quote character"
            raise ContributionsAcceptedError(msg)
        if _flag("ignoreLeadingWhiteSpace", ignoreLeadingWhiteSpace):
            msg = "ignoreLeadingWhiteSpace=True is not supported: DuckDB writes strings as they are"
            raise ContributionsAcceptedError(msg)
        if _flag("ignoreTrailingWhiteSpace", ignoreTrailingWhiteSpace):
            msg = "ignoreTrailingWhiteSpace=True is not supported: DuckDB writes strings as they are"
            raise ContributionsAcceptedError(msg)
        if charToEscapeQuoteEscaping:
            raise NotImplementedError
        if emptyValue:
            raise NotImplementedError
        if lineSep:
            raise NotImplementedError
        _check_unique_names(self.dataframe.columns)
        if not _should_write(self.dataframe.session, path, mode):
            return
        for field in self.dataframe.schema.fields:
            if isinstance(field.dataType, (ArrayType, MapType, StructType, NullType)):
                raise _unsupported_csv_type(field)
        self.dataframe.relation.write_csv(
            _duckdb_path(path),
            **_csv_dialect(sep, quote, escape),
            na_rep=nullValue,
            compression=compression,
            header=_flag("header", header),
            encoding=encoding,
            quoting="ALL" if _flag("quoteAll", quoteAll) else None,
            date_format=_strftime_format(dateFormat),
            timestamp_format=_strftime_format(timestampFormat),
        )


class DataFrameReader:  # noqa: D101
    def __init__(self, session: "SparkSession") -> None:  # noqa: D107
        self.session = session

    def load(  # noqa: D102
        self,
        path: str | list[str] | None = None,
        format: str | None = None,
        schema: StructType | str | None = None,
        **options: OptionalPrimitiveType,
    ) -> "DataFrame":
        from duckdb.experimental.spark.sql.dataframe import DataFrame, _source_frame

        if not isinstance(path, str):
            raise TypeError
        if options:
            raise ContributionsAcceptedError

        format = format.lower() if format else _format_of(path)
        if format in ("csv", "tsv"):
            return self.csv(path, schema=schema, sep="\t" if format == "tsv" else None)
        if format == "json":
            if schema:
                return _read_json_with_schema(self.session, path, _parse_schema(schema))
            rel = _read_json(self.session, path)
            return DataFrame(None, self.session) if rel is None else _source_frame(rel, self.session)
        if format != "parquet":
            raise ContributionsAcceptedError
        df = _source_frame(self.session.conn.read_parquet(path), self.session)
        if not schema:
            return df
        # Spark finds the fields of a parquet schema by name, ignoring case.
        schema = _parse_schema(schema)
        return _with_schema(df, schema, _positions_by_name(df.relation.columns, schema, ignore_case=True))

    def csv(  # noqa: D102
        self,
        path: str | list[str],
        schema: StructType | str | None = None,
        sep: str | None = None,
        encoding: str | None = None,
        quote: str | None = None,
        escape: str | None = None,
        comment: str | None = None,
        header: bool | str | None = None,
        inferSchema: bool | str | None = None,
        ignoreLeadingWhiteSpace: bool | str | None = None,
        ignoreTrailingWhiteSpace: bool | str | None = None,
        nullValue: str | None = None,
        nanValue: str | None = None,
        positiveInf: str | None = None,
        negativeInf: str | None = None,
        dateFormat: str | None = None,
        timestampFormat: str | None = None,
        maxColumns: int | str | None = None,
        maxCharsPerColumn: int | str | None = None,
        maxMalformedLogPerPartition: int | str | None = None,
        mode: str | None = None,
        columnNameOfCorruptRecord: str | None = None,
        multiLine: bool | str | None = None,
        charToEscapeQuoteEscaping: str | None = None,
        samplingRatio: float | str | None = None,
        enforceSchema: bool | str | None = None,
        emptyValue: str | None = None,
        locale: str | None = None,
        lineSep: str | None = None,
        pathGlobFilter: bool | str | None = None,
        recursiveFileLookup: bool | str | None = None,
        modifiedBefore: bool | str | None = None,
        modifiedAfter: bool | str | None = None,
        unescapedQuoteHandling: str | None = None,
    ) -> "DataFrame":
        if not isinstance(path, str):
            raise NotImplementedError
        if comment:
            raise ContributionsAcceptedError
        if _flag("ignoreLeadingWhiteSpace", ignoreLeadingWhiteSpace):
            raise ContributionsAcceptedError
        if _flag("ignoreTrailingWhiteSpace", ignoreTrailingWhiteSpace):
            raise ContributionsAcceptedError
        if maxColumns:
            raise ContributionsAcceptedError
        if maxCharsPerColumn:
            raise ContributionsAcceptedError
        if maxMalformedLogPerPartition:
            raise ContributionsAcceptedError
        if mode:
            raise ContributionsAcceptedError
        if columnNameOfCorruptRecord:
            raise ContributionsAcceptedError
        if charToEscapeQuoteEscaping:
            raise ContributionsAcceptedError
        if samplingRatio:
            raise ContributionsAcceptedError
        if emptyValue:
            raise ContributionsAcceptedError
        if locale:
            raise ContributionsAcceptedError
        if pathGlobFilter:
            raise ContributionsAcceptedError
        if _flag("recursiveFileLookup", recursiveFileLookup):
            raise ContributionsAcceptedError
        if modifiedBefore:
            raise ContributionsAcceptedError
        if modifiedAfter:
            raise ContributionsAcceptedError
        if unescapedQuoteHandling:
            raise ContributionsAcceptedError
        if lineSep:
            # We have support for custom newline, just needs to be ported to 'read_csv'
            raise NotImplementedError
        from ..sql.dataframe import DataFrame, _source_frame

        has_header = _flag("header", header)
        enforce = _flag("enforceSchema", enforceSchema, default=True)
        infer = _flag("inferSchema", inferSchema)
        parsed = _parse_schema(schema) if schema else None
        if parsed is None and infer and (dateFormat or timestampFormat):
            msg = "Inferring dates and timestamps with dateFormat or timestampFormat is not supported"
            raise ContributionsAcceptedError(msg)
        null_value = "" if nullValue is None else nullValue
        options: dict[str, Any] = {
            **_csv_dialect(sep, quote, escape),
            "encoding": encoding,
            # Spark reads the fields of a line by position, NULL past its end, and drops those past the last column.
            "null_padding": True,
            "strict_mode": False,
        }
        if _flag("multiLine", multiLine):
            # DuckDB reads quoted line breaks with null padding only in its serial reader.
            options["parallel"] = False
        fields: list[str | None] = []
        if parsed is None:
            first = _first_csv_line(self.session, path, options)
            if first is None:
                return DataFrame(None, self.session)
            fields = first
        elif has_header and not enforce:
            # Spark checks the header line of each file.
            for (file,) in self.session.conn.execute("SELECT file FROM glob($1) ORDER BY file", [path]).fetchall():
                first = _first_csv_line(self.session, file, options)
                if first is not None:
                    _check_csv_header(first, parsed, file)
        if parsed is not None:
            names = parsed.names
        elif has_header:
            names = _csv_header_names(fields, nullValue)
        else:
            names = [f"_c{i}" for i in range(len(fields))]
        if parsed is None and infer:
            return _read_inferred_csv(self.session, path, names, has_header, null_value, options)
        columns = [ColumnExpression(f"column{i}") for i in range(len(names))]
        rel = self.session.conn.read_csv(
            path,
            auto_detect=False,
            columns={column.get_name(): "VARCHAR" for column in columns},
            header=has_header,
            # An unquoted empty field is NULL and a quoted one the empty string, which nullValue then makes NULL.
            na_values=[""],
            allow_quoted_nulls=False,
            **options,
        )
        df = _source_frame(rel, self.session)
        texts = df.relation.select(
            *(FunctionExpression("nullif", c, ConstantExpression(null_value)).alias(c.get_name()) for c in columns)
        )
        if parsed is not None:
            schema_fields = parsed.fields
            patterns = {kind: p for kind, p in ((DateType, dateFormat), (TimestampType, timestampFormat)) if p}
        else:
            schema_fields, patterns = [StructField(name, StringType()) for name in names], {}
        specials = (
            "NaN" if nanValue is None else nanValue,
            "Inf" if positiveInf is None else positiveInf,
            "-Inf" if negativeInf is None else negativeInf,
        )
        values = [
            _csv_value(column, field, patterns, specials).alias(field.name)
            for column, field in zip(columns, schema_fields, strict=True)
        ]
        return df._derive(texts.select(*values))

    def parquet(self, *paths: str, **options: "OptionalPrimitiveType") -> "DataFrame":  # noqa: D102
        input = list(paths)
        if len(input) != 1:
            msg = "Only single paths are supported for now"
            raise NotImplementedError(msg)
        option_amount = len(options.keys())
        if option_amount != 0:
            msg = "Options are not supported"
            raise ContributionsAcceptedError(msg)
        path = input[0]
        rel = self.session.conn.read_parquet(path)
        from ..sql.dataframe import _source_frame

        return _source_frame(rel, self.session)

    def json(
        self,
        path: str | list[str],
        schema: StructType | str | None = None,
        primitivesAsString: bool | str | None = None,
        prefersDecimal: bool | str | None = None,
        allowComments: bool | str | None = None,
        allowUnquotedFieldNames: bool | str | None = None,
        allowSingleQuotes: bool | str | None = None,
        allowNumericLeadingZero: bool | str | None = None,
        allowBackslashEscapingAnyCharacter: bool | str | None = None,
        mode: str | None = None,
        columnNameOfCorruptRecord: str | None = None,
        dateFormat: str | None = None,
        timestampFormat: str | None = None,
        multiLine: bool | str | None = None,
        allowUnquotedControlChars: bool | str | None = None,
        lineSep: str | None = None,
        samplingRatio: float | str | None = None,
        dropFieldIfAllNull: bool | str | None = None,
        encoding: str | None = None,
        locale: str | None = None,
        pathGlobFilter: bool | str | None = None,
        recursiveFileLookup: bool | str | None = None,
        modifiedBefore: bool | str | None = None,
        modifiedAfter: bool | str | None = None,
        allowNonNumericNumbers: bool | str | None = None,
    ) -> "DataFrame":
        """Loads JSON files and returns the results as a :class:`DataFrame`.

        `JSON Lines <http://jsonlines.org/>`_ (newline-delimited JSON) is supported by default.
        For JSON (one record per file), set the ``multiLine`` parameter to ``true``.

        If the ``schema`` parameter is not specified, this function goes
        through the input once to determine the input schema.

        .. versionadded:: 1.4.0

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Parameters
        ----------
        path : str, list or :class:`RDD`
            string represents path to the JSON dataset, or a list of paths,
            or RDD of Strings storing JSON objects.
        schema : :class:`pyspark.sql.types.StructType` or str, optional
            an optional :class:`pyspark.sql.types.StructType` for the input schema or
            a DDL-formatted string (For example ``col0 INT, col1 DOUBLE``).

        Other Parameters
        ----------------
        Extra options
            For the extra options, refer to
            `Data Source Option <https://spark.apache.org/docs/latest/sql-data-sources-json.html#data-source-option>`_
            for the version you use.

            .. # noqa

        Examples:
        --------
        Write a DataFrame into a JSON file and read it back.

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     # Write a DataFrame into a JSON file
        ...     spark.createDataFrame([{"age": 100, "name": "Hyukjin Kwon"}]).write.mode(
        ...         "overwrite"
        ...     ).format("json").save(d)
        ...
        ...     # Read the JSON file as a DataFrame.
        ...     spark.read.json(d).show()
        +---+------------+
        |age|        name|
        +---+------------+
        |100|Hyukjin Kwon|
        +---+------------+
        """
        if schema is not None:
            msg = "The 'schema' option is not supported"
            raise ContributionsAcceptedError(msg)
        if primitivesAsString is not None:
            msg = "The 'primitivesAsString' option is not supported"
            raise ContributionsAcceptedError(msg)
        if prefersDecimal is not None:
            msg = "The 'prefersDecimal' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowComments is not None:
            msg = "The 'allowComments' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowUnquotedFieldNames is not None:
            msg = "The 'allowUnquotedFieldNames' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowSingleQuotes is not None:
            msg = "The 'allowSingleQuotes' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowNumericLeadingZero is not None:
            msg = "The 'allowNumericLeadingZero' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowBackslashEscapingAnyCharacter is not None:
            msg = "The 'allowBackslashEscapingAnyCharacter' option is not supported"
            raise ContributionsAcceptedError(msg)
        if mode is not None:
            msg = "The 'mode' option is not supported"
            raise ContributionsAcceptedError(msg)
        if columnNameOfCorruptRecord is not None:
            msg = "The 'columnNameOfCorruptRecord' option is not supported"
            raise ContributionsAcceptedError(msg)
        if dateFormat is not None:
            msg = "The 'dateFormat' option is not supported"
            raise ContributionsAcceptedError(msg)
        if timestampFormat is not None:
            msg = "The 'timestampFormat' option is not supported"
            raise ContributionsAcceptedError(msg)
        if multiLine is not None:
            msg = "The 'multiLine' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowUnquotedControlChars is not None:
            msg = "The 'allowUnquotedControlChars' option is not supported"
            raise ContributionsAcceptedError(msg)
        if lineSep is not None:
            msg = "The 'lineSep' option is not supported"
            raise ContributionsAcceptedError(msg)
        if samplingRatio is not None:
            msg = "The 'samplingRatio' option is not supported"
            raise ContributionsAcceptedError(msg)
        if dropFieldIfAllNull is not None:
            msg = "The 'dropFieldIfAllNull' option is not supported"
            raise ContributionsAcceptedError(msg)
        if encoding is not None:
            msg = "The 'encoding' option is not supported"
            raise ContributionsAcceptedError(msg)
        if locale is not None:
            msg = "The 'locale' option is not supported"
            raise ContributionsAcceptedError(msg)
        if pathGlobFilter is not None:
            msg = "The 'pathGlobFilter' option is not supported"
            raise ContributionsAcceptedError(msg)
        if recursiveFileLookup is not None:
            msg = "The 'recursiveFileLookup' option is not supported"
            raise ContributionsAcceptedError(msg)
        if modifiedBefore is not None:
            msg = "The 'modifiedBefore' option is not supported"
            raise ContributionsAcceptedError(msg)
        if modifiedAfter is not None:
            msg = "The 'modifiedAfter' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowNonNumericNumbers is not None:
            msg = "The 'allowNonNumericNumbers' option is not supported"
            raise ContributionsAcceptedError(msg)

        if isinstance(path, str):
            path = [path]
        if isinstance(path, list):
            if len(path) == 1:
                rel = _read_json(self.session, path[0])
                from .dataframe import DataFrame, _source_frame

                return DataFrame(None, self.session) if rel is None else _source_frame(rel, self.session)
            raise PySparkNotImplementedError(message="Only a single path is supported for now")
        else:
            raise PySparkTypeError(
                error_class="NOT_STR_OR_LIST_OF_RDD",
                message_parameters={
                    "arg_name": "path",
                    "arg_type": type(path).__name__,
                },
            )


__all__ = ["DataFrameReader", "DataFrameWriter"]

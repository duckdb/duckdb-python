from typing import NamedTuple

from .session import SparkSession


class Database(NamedTuple):  # noqa: D101
    name: str
    catalog: str
    description: str | None
    locationUri: str


class Table(NamedTuple):  # noqa: D101
    name: str
    database: str | None
    catalog: str
    description: str | None
    tableType: str
    isTemporary: bool


class Column(NamedTuple):  # noqa: D101
    name: str
    description: str | None
    dataType: str
    nullable: bool
    isPartition: bool
    isBucket: bool
    isCluster: bool


class Function(NamedTuple):  # noqa: D101
    name: str
    catalog: str | None
    namespace: list[str] | None
    description: str | None
    className: str
    isTemporary: bool


class Catalog:
    """Implements the spark catalog API.

    Implementation notes:
    Spark has the concept of a catalog and inside each catalog there are schemas
    which contain tables. But spark calls the schemas as databases through
    the catalog API.
    For Duckdb, there are databases, which in turn contain schemas. DuckDBs
    databases therefore overlap with the concept of the spark catalog.
    So to summarize
    ------------------------------
    | Spark           | DuckDB   |
    ------------------------------
    ! Catalog         | Database |
    | Database/Schema | Schema   |
    ------------------------------
    The consequence is that this catalog API refers in several locations to a
    database name, which is the DuckDB schema.
    """

    def __init__(self, session: SparkSession) -> None:  # noqa: D107
        self._session = session

    def listDatabases(self, pattern: str | None = None) -> list[Database]:
        """Returns a list of database object for all available databases."""
        if pattern:
            pattern = pattern.replace("*", "%")
            where_sql = " WHERE schema_name LIKE ?"
            params = (pattern,)
        else:
            where_sql = ""
            params = ()

        sql_text = "select schema_name, database_name from duckdb_schemas()" + where_sql
        res = self._session.conn.sql(sql_text, params=params).fetchall()

        def transform_to_database(x: tuple[str, ...]) -> Database:
            return Database(name=x[0], catalog=x[1], description=None, locationUri="")

        databases = [transform_to_database(x) for x in res]
        return databases

    def listTables(self, dbName: str | None = None, pattern: str | None = None) -> list[Table]:
        """Returns a list of tables/views in the specified database.

        If dbName nor pattern are provided, the current active database is used.
        """
        dbName = dbName or self.currentDatabase()
        current_catalog = self._currentCatalog()
        where_sql1 = where_sql2 = ""
        params = (current_catalog, dbName)

        if pattern:
            where_sql1 = " and table_name LIKE ?"
            where_sql2 = " and view_name LIKE ?"
            params += (pattern.replace("*", "%"),)

        sql_text = (
            "select database_name, schema_name, table_name, comment, temporary, 'TABLE'"
            f"from duckdb_tables() where database_name = ? and schema_name = ?{where_sql1}"
            " union all"
            " select database_name, schema_name, view_name, comment, temporary, 'VIEW'"
            f" from duckdb_views() where database_name = ? and schema_name = ?{where_sql2}"
        )

        res = self._session.conn.sql(sql_text, params=(*params, *params)).fetchall()

        def transform_to_table(x: list[str]) -> Table:
            return Table(
                name=x[2], database=x[1], catalog=x[0], description=x[3], tableType=x[5], isTemporary=bool(x[4])
            )

        tables = [transform_to_table(x) for x in res]
        return tables

    def listColumns(self, tableName: str, dbName: str | None = None) -> list[Column]:
        """Returns a list of columns for the given table/view in the specified database."""
        query = (
            "select column_name, data_type, is_nullable"
            " from duckdb_columns()"
            " where table_name = ? and schema_name = ? and database_name = ?"
        )
        dbName = dbName or self.currentDatabase()
        params = (tableName, dbName, self._currentCatalog())
        res = self._session.conn.sql(query, params=params).fetchall()

        if len(res) == 0:
            from duckdb.experimental.spark.errors import AnalysisException

            msg = f"[TABLE_OR_VIEW_NOT_FOUND] The table or view `{tableName}` cannot be found"
            raise AnalysisException(msg)

        def transform_to_column(x: list[str | bool]) -> Column:
            return Column(
                name=x[0],
                description=None,
                dataType=x[1],
                nullable=x[2],
                isPartition=False,
                isBucket=False,
                isCluster=False,
            )

        columns = [transform_to_column(x) for x in res]
        return columns

    def listFunctions(self, dbName: str | None = None, pattern: str | None = None) -> list[Function]:
        """Returns a list of functions registered in the specified database."""
        dbName = dbName or self.currentDatabase()
        where_sql = ""
        params = (dbName,)

        if pattern:
            pattern = pattern.replace("*", "%")
            where_sql = " AND function_name LIKE ?"
            params = (pattern,)

        sql_text = (
            "SELECT DISTINCT database_name, schema_name, function_name, description, function_type"
            " FROM duckdb_functions()"
            " WHERE schema_name = ? " + where_sql
        )

        res = self._session.conn.sql(sql_text, params=params).fetchall()

        columns = [
            Function(
                name=x[2],
                catalog=x[0],
                namespace=[x[1]],
                description=x[3],
                className=x[4],
                isTemporary=x[0] == "temp",
            )
            for x in res
        ]
        return columns

    def currentDatabase(self) -> str:
        """Retrieves the name of the active database/schema."""
        res = self._session.conn.sql("SELECT current_schema()").fetchone()
        return res[0]

    def setCurrentDatabase(self, dbName: str) -> None:
        """Sets the active database/schema. Equivalent to executing 'USE dbName'."""
        self._session.conn.sql(f"USE {_sql_quote(dbName)}")

    def _currentCatalog(self) -> str:
        res = self._session.conn.sql("SELECT current_database()").fetchone()
        return res[0]


def _sql_quote(value: str) -> str:
    return f'"{value.replace('"', '""')}"'


__all__ = ["Catalog", "Column", "Database", "Function", "Table"]

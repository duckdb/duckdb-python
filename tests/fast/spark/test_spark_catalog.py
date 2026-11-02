import pytest

_ = pytest.importorskip("duckdb.experimental.spark")

from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.sql.catalog import Column, Database, Table


class TestSparkCatalog:
    def test_list_databases_all(self, spark):
        dbs = spark.catalog.listDatabases()
        if USE_ACTUAL_SPARK:
            assert all(isinstance(db, Database) for db in dbs)
        else:
            assert dbs == [
                Database(name="main", catalog="memory", description=None, locationUri=""),
                Database(name="information_schema", catalog="system", description=None, locationUri=""),
                Database(name="main", catalog="system", description=None, locationUri=""),
                Database(name="pg_catalog", catalog="system", description=None, locationUri=""),
                Database(name="main", catalog="temp", description=None, locationUri=""),
            ]

    def test_create_use_schema(self, spark):
        assert spark.catalog.currentDatabase() == "main"

        spark.sql("CREATE SCHEMA my_schema1")
        spark.catalog.setCurrentDatabase("my_schema1")
        assert spark.catalog.currentDatabase() == "my_schema1"

        dbs = spark.catalog.listDatabases("*schema1")
        assert len(dbs) == 1
        assert spark.catalog.currentDatabase() == "my_schema1"

        if USE_ACTUAL_SPARK:
            return

        # Verifying the table goes to the right schema.
        spark.sql("create table tbl1(a varchar)")
        spark.sql("create table main.tbl2(a varchar)")
        expected = [
            Table(
                name="tbl1",
                catalog="memory",
                database="my_schema1",
                description=None,
                tableType="TABLE",
                isTemporary=False,
            )
        ]
        tbls = spark.catalog.listTables()
        assert tbls == expected

        spark.sql("DROP TABLE my_schema1.tbl1")
        spark.sql("DROP SCHEMA my_schema1")
        assert len(spark.catalog.listDatabases("my_schema1")) == 0
        assert spark.catalog.currentDatabase() == "main"

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="Checking duckdb specific databases")
    def test_list_databases_pattern(self, spark):
        expected = [
            Database(name="pg_catalog", catalog="system", description=None, locationUri=""),
        ]
        dbs = spark.catalog.listDatabases("pg*")
        assert dbs == expected
        dbs = spark.catalog.listDatabases("pg_catalog")
        assert dbs == expected
        dbs = spark.catalog.listDatabases("notfound")
        assert dbs == []

    def test_list_tables_empty(self, spark):
        tbls = spark.catalog.listTables()
        assert tbls == []

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="Checking duckdb specific tables")
    def test_list_tables_create(self, spark):
        spark.sql("create table tbl1(a varchar)")
        spark.sql("create table tbl2(b varchar); COMMENT ON TABLE tbl2 IS 'hello world'")
        expected = [
            Table(
                name="tbl1", catalog="memory", database="main", description=None, tableType="TABLE", isTemporary=False
            ),
            Table(
                name="tbl2",
                catalog="memory",
                database="main",
                description="hello world",
                tableType="TABLE",
                isTemporary=False,
            ),
        ]
        tbls = spark.catalog.listTables()
        assert tbls == expected

        tbls = spark.catalog.listTables(pattern="*l2")
        assert tbls == expected[1:]

        tbls = spark.catalog.listTables(pattern="tbl2")
        assert tbls == expected[1:]

        tbls = spark.catalog.listTables(dbName="notfound")
        assert tbls == []

        spark.sql("create view vw as select * from tbl1")
        expected += [
            Table(name="vw", catalog="memory", database="main", description=None, tableType="VIEW", isTemporary=False),
        ]
        tbls = spark.catalog.listTables()
        assert tbls == expected

    @pytest.mark.skipif(USE_ACTUAL_SPARK, reason="We can't create tables with our Spark test setup")
    def test_list_columns(self, spark):
        spark.sql("create table tbl(a varchar, b bool)")

        columns = spark.catalog.listColumns("tbl")
        kwds = dict(description=None, nullable=True, isPartition=False, isBucket=False, isCluster=False)  # noqa: C408
        assert columns == [Column(name="a", dataType="VARCHAR", **kwds), Column(name="b", dataType="BOOLEAN", **kwds)]

        spark.sql("create view vw as select * from tbl")
        view_columns = spark.catalog.listColumns("vw")
        assert view_columns == columns

        from spark_namespace.errors import AnalysisException

        with pytest.raises(AnalysisException):
            assert spark.catalog.listColumns("tbl", "notfound")

    def test_list_columns_not_found(self, spark):
        from spark_namespace.errors import AnalysisException

        with pytest.raises(AnalysisException):
            spark.catalog.listColumns("none_existant")

    def test_list_functions(self, spark):
        fns = spark.catalog.listFunctions()
        assert len(fns)
        assert any(f.name == "current_database" for f in fns)

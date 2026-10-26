"""rel.insert_into and rel.create write to the table SQL resolves for the same name, and refuse a name as SQL does.

A name of more than three parts is refused, because the relation API addresses only a catalog, a schema and a name.
"""

import pytest

import duckdb
import pandas as pd

INSERT_SQL = "INSERT INTO {}"
CREATE_SQL = "CREATE TABLE {} AS"


def nested_path_refusal(rendered, sql_form):
    return (
        "Not implemented Error: The relation API cannot address a nested schema path such as "
        f"{rendered} yet; use SQL ({sql_form.format(rendered)}) instead"
    )


FLAT_SETUP = [
    "CREATE SCHEMA s",
    "CREATE TABLE main.t(i INTEGER)",
    "CREATE TABLE s.t(i INTEGER)",
    'CREATE TABLE main."s.t"(i INTEGER)',
    'CREATE SCHEMA "my.schema"',
    'CREATE TABLE "my.schema"."my.t"(i INTEGER)',
    "ATTACH ':memory:' AS db2",
    "CREATE TABLE db2.main.t(i INTEGER)",
    "CREATE SCHEMA db2.s",
    "CREATE TABLE db2.s.t(i INTEGER)",
    # A schema and an attached database both called amb make "amb.x" ambiguous.
    "ATTACH ':memory:' AS amb",
    "CREATE SCHEMA memory.amb",
    "CREATE TABLE memory.amb.t(i INTEGER)",
    "CREATE TABLE amb.main.t(i INTEGER)",
]
FLAT_TABLES = [
    "memory.main.t",
    "memory.s.t",
    'memory.main."s.t"',
    'memory."my.schema"."my.t"',
    "db2.main.t",
    "db2.s.t",
    "memory.amb.t",
    "amb.main.t",
]

NESTED_SETUP = [
    *FLAT_SETUP,
    "CREATE SCHEMA extra",
    "CREATE SCHEMA extra.s",
    "CREATE TABLE extra.s.t(i INTEGER)",
    "CREATE SCHEMA extra.s.deep",
    "CREATE TABLE extra.s.deep.t(i INTEGER)",
    "CREATE SCHEMA db2.extra",
    "CREATE SCHEMA db2.extra.s",
    "CREATE TABLE db2.extra.s.t(i INTEGER)",
    'CREATE SCHEMA "my.schema".s',
    'CREATE TABLE "my.schema".s."my t"(i INTEGER)',
]
NESTED_TABLES = [
    *FLAT_TABLES,
    "memory.extra.s.t",
    "memory.extra.s.deep.t",
    "db2.extra.s.t",
    'memory."my.schema".s."my t"',
]


def connect(setup):
    con = duckdb.connect()
    for statement in setup:
        con.execute(statement)
    return con


def source(con):
    return con.sql("SELECT 42 AS i")


def contents(con, tables):
    return {table: con.sql(f"FROM {table}").fetchall() for table in tables}


def table_count(con):
    [(count,)] = con.sql("SELECT count(*) FROM duckdb_tables()").fetchall()
    return count


def assert_inserts_into(setup, tables, name, target):
    expected = {table: [(42,)] if table == target else [] for table in tables}
    via_sql = connect(setup)
    via_sql.execute(f"INSERT INTO {name} SELECT 42 AS i")
    assert contents(via_sql, tables) == expected
    via_relation = connect(setup)
    source(via_relation).insert_into(name)
    assert contents(via_relation, tables) == expected


def assert_creates(setup, name, target, method="create"):
    via_sql = connect(setup)
    before = table_count(via_sql)
    via_sql.execute(f"CREATE TABLE {name} AS SELECT 42 AS i")
    assert via_sql.sql(f"FROM {target}").fetchall() == [(42,)]
    assert table_count(via_sql) == before + 1
    via_relation = connect(setup)
    getattr(source(via_relation), method)(name)
    assert via_relation.sql(f"FROM {target}").fetchall() == [(42,)]
    assert table_count(via_relation) == before + 1


def assert_refused_as_sql_refuses(setup, tables, sql, write, error):
    via_sql = connect(setup)
    with pytest.raises(error) as sql_error:
        via_sql.execute(sql)
    via_relation = connect(setup)
    before = table_count(via_relation)
    with pytest.raises(error) as relation_error:
        write(source(via_relation))
    assert str(relation_error.value) == str(sql_error.value)
    assert table_count(via_relation) == before
    assert all(rows == [] for rows in contents(via_relation, tables).values())


class TestFlatNames:
    @pytest.mark.parametrize(
        ("name", "target"),
        [
            ("t", "memory.main.t"),
            ("s.t", "memory.s.t"),
            ("memory.s.t", "memory.s.t"),
            ("db2.t", "db2.main.t"),
            ("db2.s.t", "db2.s.t"),
            ('"my.schema"."my.t"', 'memory."my.schema"."my.t"'),
            ('"s.t"', 'memory.main."s.t"'),
            ("memory.amb.t", "memory.amb.t"),
            ("amb.main.t", "amb.main.t"),
        ],
    )
    def test_insert_into(self, name, target):
        assert_inserts_into(FLAT_SETUP, FLAT_TABLES, name, target)

    @pytest.mark.parametrize(
        ("name", "error"),
        [
            ("nosuch.t", duckdb.CatalogException),
            ("nosuch.s.t", duckdb.CatalogException),
            ("amb.t", duckdb.BinderException),
            ("s.missing", duckdb.CatalogException),
        ],
    )
    def test_insert_into_refused(self, name, error):
        assert_refused_as_sql_refuses(
            FLAT_SETUP, FLAT_TABLES, f"INSERT INTO {name} SELECT 42 AS i", lambda rel: rel.insert_into(name), error
        )

    @pytest.mark.parametrize(
        ("name", "target"),
        [
            ("n", "memory.main.n"),
            ("s.n", "memory.s.n"),
            ("memory.s.n", "memory.s.n"),
            ("db2.n", "db2.main.n"),
            ("db2.main.n", "db2.main.n"),
            ("db2.s.n", "db2.s.n"),
            ('"my.schema"."my.n"', 'memory."my.schema"."my.n"'),
            ('"s.n"', 'memory.main."s.n"'),
            ("memory.amb.n", "memory.amb.n"),
            ("amb.main.n", "amb.main.n"),
        ],
    )
    def test_create(self, name, target):
        assert_creates(FLAT_SETUP, name, target)

    def test_to_table(self):
        assert_creates(FLAT_SETUP, "db2.s.n", "db2.s.n", method="to_table")

    @pytest.mark.parametrize(
        ("name", "error"),
        [
            ("nosuch.n", duckdb.CatalogException),
            ("nosuch.s.n", duckdb.CatalogException),
            ("amb.n", duckdb.BinderException),
            ("temp.n", duckdb.ParserException),
            ("temp.main.n", duckdb.ParserException),
            ("s.t", duckdb.CatalogException),
        ],
    )
    def test_create_refused(self, name, error):
        assert_refused_as_sql_refuses(
            FLAT_SETUP, FLAT_TABLES, f"CREATE TABLE {name} AS SELECT 42 AS i", lambda rel: rel.create(name), error
        )


class TestNestedNames:
    def test_insert_into_three_parts(self):
        assert_inserts_into(NESTED_SETUP, NESTED_TABLES, "extra.s.t", "memory.extra.s.t")

    def test_insert_into_three_parts_refused(self):
        assert_refused_as_sql_refuses(
            NESTED_SETUP,
            NESTED_TABLES,
            "INSERT INTO extra.nosuch.t SELECT 42 AS i",
            lambda rel: rel.insert_into("extra.nosuch.t"),
            duckdb.CatalogException,
        )

    def test_create_three_parts(self):
        assert_creates(NESTED_SETUP, "extra.s.n", "memory.extra.s.n")

    def test_create_three_parts_refused(self):
        assert_refused_as_sql_refuses(
            NESTED_SETUP,
            NESTED_TABLES,
            "CREATE TABLE extra.nosuch.n AS SELECT 42 AS i",
            lambda rel: rel.create("extra.nosuch.n"),
            duckdb.CatalogException,
        )

    @pytest.mark.parametrize(
        ("name", "rendered"),
        [
            ("memory.extra.s.t", "memory.extra.s.t"),
            ("memory.extra.s.deep.t", "memory.extra.s.deep.t"),
            ("db2.extra.s.t", "db2.extra.s.t"),
            ("memory.extra.nosuch.t", "memory.extra.nosuch.t"),
            ('"memory"."extra"."s"."t"', "memory.extra.s.t"),
        ],
    )
    def test_insert_into_more_than_three_parts_refused(self, name, rendered):
        con = connect(NESTED_SETUP)
        with pytest.raises(duckdb.NotImplementedException) as error:
            source(con).insert_into(name)
        assert str(error.value) == nested_path_refusal(rendered, INSERT_SQL)
        assert all(rows == [] for rows in contents(con, NESTED_TABLES).values())

    @pytest.mark.parametrize("method", ["create", "to_table"])
    @pytest.mark.parametrize(
        "name", ["memory.extra.s.n", "memory.extra.s.deep.n", "db2.extra.s.n", "memory.extra.nosuch.n"]
    )
    def test_create_more_than_three_parts_refused(self, name, method):
        con = connect(NESTED_SETUP)
        before = table_count(con)
        with pytest.raises(duckdb.NotImplementedException) as error:
            getattr(source(con), method)(name)
        assert str(error.value) == nested_path_refusal(name, CREATE_SQL)
        assert table_count(con) == before

    def test_insert_into_refusal_quotes_the_suggested_sql(self):
        name = 'memory."my.schema".s."my t"'
        con = connect(NESTED_SETUP)
        with pytest.raises(duckdb.NotImplementedException) as error:
            source(con).insert_into('"memory"."my.schema"."s"."my t"')
        assert str(error.value) == nested_path_refusal(name, INSERT_SQL)
        con.execute(f"{INSERT_SQL.format(name)} SELECT 42 AS i")
        assert con.sql(f"FROM {name}").fetchall() == [(42,)]

    def test_create_refusal_quotes_the_suggested_sql(self):
        name = 'memory."my.schema".s."my n"'
        con = connect(NESTED_SETUP)
        with pytest.raises(duckdb.NotImplementedException) as error:
            source(con).create('"memory"."my.schema"."s"."my n"')
        assert str(error.value) == nested_path_refusal(name, CREATE_SQL)
        con.execute(f"{CREATE_SQL.format(name)} SELECT 42 AS i")
        assert con.sql(f"FROM {name}").fetchall() == [(42,)]

    @pytest.mark.parametrize(
        ("method", "name"),
        [("insert_into", "memory.extra.s.t"), ("create", "memory.extra.s.n"), ("to_table", "memory.extra.s.n")],
    )
    def test_refusal_keeps_the_transaction_usable(self, method, name):
        con = connect(NESTED_SETUP)
        con.execute("BEGIN")
        con.execute("CREATE TABLE sentinel AS SELECT 1 AS i")
        with pytest.raises(duckdb.NotImplementedException):
            getattr(source(con), method)(name)
        con.execute("INSERT INTO sentinel VALUES (2)")
        assert con.sql("FROM sentinel ORDER BY i").fetchall() == [(1,), (2,)]
        con.execute("ROLLBACK")
        assert con.sql("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'sentinel'").fetchall() == [(0,)]
        assert all(rows == [] for rows in contents(con, NESTED_TABLES).values())


class TestEmptyNamePart:
    @pytest.mark.parametrize("name", ["", "s..t", ".t"])
    def test_insert_into_refused(self, name):
        con = connect(FLAT_SETUP)
        with pytest.raises(duckdb.ParserException) as error:
            source(con).insert_into(name)
        assert str(error.value) == f"Parser Error: Zero-length identifier in qualified name! (input: {name})"
        assert all(rows == [] for rows in contents(con, FLAT_TABLES).values())

    @pytest.mark.parametrize("name", ["", "s..n", ".n"])
    def test_create_refused(self, name):
        con = connect(FLAT_SETUP)
        before = table_count(con)
        with pytest.raises(duckdb.ParserException) as error:
            source(con).create(name)
        assert str(error.value) == f"Parser Error: Zero-length identifier in qualified name! (input: {name})"
        assert table_count(con) == before


def frame_source(con):
    frame = pd.DataFrame({"i": [42]})  # noqa: F841
    return con.sql("SELECT * FROM frame")


class TestSourceOutlivesItsScope:
    def test_insert_into(self):
        con = connect(FLAT_SETUP)
        frame_source(con).insert_into("s.t")
        assert con.sql("FROM s.t").fetchall() == [(42,)]

    def test_create(self):
        con = connect(FLAT_SETUP)
        frame_source(con).create("s.n")
        assert con.sql("FROM s.n").fetchall() == [(42,)]

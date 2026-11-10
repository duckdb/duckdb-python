"""Relation writes and views land where the engine's reader of a qualified name puts the same name.

The name is a dotted string or a tuple of its parts, of any depth: a nested schema path or an attached database. The
reader is the engine's own for such a string, so a few spellings read otherwise than in SQL text (TestReaderRules).
"""

import enum
import gc
import re

import pytest

import duckdb
import pandas as pd

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


def assert_inserts_into(setup, tables, name, target, rendered=None):
    expected = {table: [(42,)] if table == target else [] for table in tables}
    via_sql = connect(setup)
    via_sql.execute(f"INSERT INTO {rendered or name} SELECT 42 AS i")
    assert contents(via_sql, tables) == expected
    via_relation = connect(setup)
    source(via_relation).insert_into(name)
    assert contents(via_relation, tables) == expected


def assert_creates(setup, name, target, method="create", rendered=None):
    via_sql = connect(setup)
    before = table_count(via_sql)
    via_sql.execute(f"CREATE TABLE {rendered or name} AS SELECT 42 AS i")
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


def quoted(parts):
    return ".".join('"' + part.replace('"', '""') + '"' for part in parts)


VIEW_PATHS = """
WITH RECURSIVE chain(oid, path) AS (
    SELECT oid, '"' || replace(schema_name, '"', '""') || '"' FROM duckdb_schemas() WHERE parent_schema_oid IS NULL
    UNION ALL
    SELECT c.oid, p.path || '.' || '"' || replace(c.schema_name, '"', '""') || '"'
    FROM duckdb_schemas() c JOIN chain p ON c.parent_schema_oid = p.oid
)
SELECT '"' || replace(v.database_name, '"', '""') || '"' || '.' || chain.path || '.'
    || '"' || replace(v.view_name, '"', '""') || '"'
FROM duckdb_views() v JOIN chain ON v.schema_oid = chain.oid
WHERE NOT v.internal
ORDER BY ALL
"""


def view_paths(con):
    """Every view as database.schema path.name, each part quoted, so a wrong schema or a dotted name shows."""
    return [path for (path,) in con.sql(VIEW_PATHS).fetchall()]


def assert_creates_view(setup, name, target, method="create_view", rendered=None):
    via_sql = connect(setup)
    before = view_paths(via_sql)
    via_sql.execute(f"CREATE OR REPLACE VIEW {rendered or name} AS SELECT 42 AS i")
    assert via_sql.sql(f"FROM {target}").fetchall() == [(42,)]
    assert len(view_paths(via_sql)) == len(before) + 1
    via_relation = connect(setup)
    assert getattr(source(via_relation), method)(name) is not None
    assert via_relation.sql(f"FROM {target}").fetchall() == [(42,)]
    assert view_paths(via_relation) == view_paths(via_sql)


def assert_view_refused_as_sql_refuses(setup, sql, write, error):
    via_sql = connect(setup)
    with pytest.raises(error) as sql_error:
        via_sql.execute(sql)
    via_relation = connect(setup)
    before = view_paths(via_relation)
    with pytest.raises(error) as relation_error:
        write(source(via_relation))
    assert str(relation_error.value) == str(sql_error.value)
    assert view_paths(via_relation) == before


VIEW_METHODS = ["create_view", "to_view"]
WRITE_METHODS = ["create", "to_table", "insert_into", "create_view", "to_view"]


class TestViewNames:
    @pytest.mark.parametrize("method", VIEW_METHODS)
    @pytest.mark.parametrize(
        ("name", "target"),
        [
            ("v", "memory.main.v"),
            ("s.v", "memory.s.v"),
            ("memory.s.v", "memory.s.v"),
            ("db2.v", "db2.main.v"),
            ("db2.main.v", "db2.main.v"),
            ("db2.s.v", "db2.s.v"),
            ('"my.schema"."my.v"', 'memory."my.schema"."my.v"'),
            ('"s.v"', 'memory.main."s.v"'),
            ("memory.amb.v", "memory.amb.v"),
            ("amb.main.v", "amb.main.v"),
        ],
    )
    def test_create_view(self, method, name, target):
        assert_creates_view(FLAT_SETUP, name, target, method)

    def test_create_view_doubled_quote(self):
        assert_creates_view(FLAT_SETUP, 's."a""b"', 'memory.s."a""b"')
        con = connect(FLAT_SETUP)
        source(con).create_view('s."a""b"')
        assert con.sql("SELECT view_name FROM duckdb_views() WHERE schema_name = 's'").fetchall() == [('a"b',)]

    @pytest.mark.parametrize(
        ("name", "error"),
        [
            ("nosuch.v", duckdb.CatalogException),
            ("nosuch.s.v", duckdb.CatalogException),
            ("amb.v", duckdb.BinderException),
            ("temp.v", duckdb.ParserException),
            ("temp.main.v", duckdb.ParserException),
            ("s.t", duckdb.CatalogException),
        ],
    )
    def test_create_view_refused(self, name, error):
        assert_view_refused_as_sql_refuses(
            FLAT_SETUP, f"CREATE OR REPLACE VIEW {name} AS SELECT 42 AS i", lambda rel: rel.create_view(name), error
        )

    def test_create_view_without_replace_refused(self):
        assert_view_refused_as_sql_refuses(
            FLAT_SETUP,
            "CREATE VIEW s.t AS SELECT 42 AS i",
            lambda rel: rel.create_view("s.t", replace=False),
            duckdb.CatalogException,
        )


class TestNestedViewNames:
    def test_create_view_three_parts(self):
        assert_creates_view(NESTED_SETUP, "extra.s.v", "memory.extra.s.v")

    def test_create_view_three_parts_refused(self):
        assert_view_refused_as_sql_refuses(
            NESTED_SETUP,
            "CREATE OR REPLACE VIEW extra.nosuch.v AS SELECT 42 AS i",
            lambda rel: rel.create_view("extra.nosuch.v"),
            duckdb.CatalogException,
        )

    @pytest.mark.parametrize("method", VIEW_METHODS)
    @pytest.mark.parametrize(
        "name", ["memory.extra.s.v", "memory.extra.s.deep.v", "db2.extra.s.v", '"memory"."my.schema"."s"."my v"']
    )
    def test_create_view_more_than_three_parts(self, method, name):
        assert_creates_view(NESTED_SETUP, name, name, method)

    def test_create_view_more_than_three_parts_refused(self):
        assert_view_refused_as_sql_refuses(
            NESTED_SETUP,
            "CREATE OR REPLACE VIEW memory.extra.nosuch.v AS SELECT 42 AS i",
            lambda rel: rel.create_view("memory.extra.nosuch.v"),
            duckdb.CatalogException,
        )

    @pytest.mark.parametrize("name", ["memory.extra.s.deep.v", ("db2", "extra", "s", "v")])
    def test_replaces_an_existing_nested_view(self, name):
        con = connect(NESTED_SETUP)
        target = name if isinstance(name, str) else ".".join(name)
        con.execute(f"CREATE VIEW {target} AS SELECT 1 AS i")
        before = view_paths(con)
        source(con).create_view(name)
        assert con.sql(f"FROM {target}").fetchall() == [(42,)]
        assert view_paths(con) == before
        with pytest.raises(duckdb.CatalogException, match="already exists"):
            source(con).create_view(name, replace=False)
        assert con.sql(f"FROM {target}").fetchall() == [(42,)]


class TestMoreThanThreeParts:
    @pytest.mark.parametrize(
        ("name", "target"),
        [
            ("memory.extra.s.t", "memory.extra.s.t"),
            ("memory.extra.s.deep.t", "memory.extra.s.deep.t"),
            ("db2.extra.s.t", "db2.extra.s.t"),
            ('"memory"."my.schema"."s"."my t"', 'memory."my.schema".s."my t"'),
        ],
    )
    def test_insert_into(self, name, target):
        assert_inserts_into(NESTED_SETUP, NESTED_TABLES, name, target)

    @pytest.mark.parametrize("method", ["create", "to_table"])
    @pytest.mark.parametrize(
        "name", ["memory.extra.s.n", "memory.extra.s.deep.n", "db2.extra.s.n", '"memory"."my.schema"."s"."my n"']
    )
    def test_create(self, name, method):
        assert_creates(NESTED_SETUP, name, name, method)

    def test_insert_into_refused(self):
        assert_refused_as_sql_refuses(
            NESTED_SETUP,
            NESTED_TABLES,
            "INSERT INTO memory.extra.nosuch.t SELECT 42 AS i",
            lambda rel: rel.insert_into("memory.extra.nosuch.t"),
            duckdb.CatalogException,
        )

    def test_create_refused(self):
        assert_refused_as_sql_refuses(
            NESTED_SETUP,
            NESTED_TABLES,
            "CREATE TABLE memory.extra.nosuch.n AS SELECT 42 AS i",
            lambda rel: rel.create("memory.extra.nosuch.n"),
            duckdb.CatalogException,
        )


class TestTupleNames:
    @pytest.mark.parametrize("method", VIEW_METHODS)
    @pytest.mark.parametrize(
        ("parts", "target"),
        [
            (("v",), "memory.main.v"),
            (("s", "v"), "memory.s.v"),
            (("db2", "v"), "db2.main.v"),
            (("db2", "main", "v"), "db2.main.v"),
            (("extra", "s", "v"), "memory.extra.s.v"),
            (("memory", "extra", "s", "deep", "v"), "memory.extra.s.deep.v"),
            (("db2", "extra", "s", "v"), "db2.extra.s.v"),
        ],
    )
    def test_create_view(self, method, parts, target):
        assert_creates_view(NESTED_SETUP, parts, target, method, rendered=quoted(parts))

    @pytest.mark.parametrize("method", ["create", "to_table"])
    @pytest.mark.parametrize(
        ("parts", "target"),
        [(("n",), "memory.main.n"), (("s", "n"), "memory.s.n"), (("db2", "extra", "s", "n"), "db2.extra.s.n")],
    )
    def test_create(self, method, parts, target):
        assert_creates(NESTED_SETUP, parts, target, method, rendered=quoted(parts))

    @pytest.mark.parametrize(
        ("parts", "target"),
        [
            (("t",), "memory.main.t"),
            (("s", "t"), "memory.s.t"),
            (("memory", "extra", "s", "deep", "t"), "memory.extra.s.deep.t"),
        ],
    )
    def test_insert_into(self, parts, target):
        assert_inserts_into(NESTED_SETUP, NESTED_TABLES, parts, target, rendered=quoted(parts))

    def test_dotted_part_stays_one_part(self):
        assert_creates_view(
            FLAT_SETUP, ("my.schema", "my.v"), 'memory."my.schema"."my.v"', rendered='"my.schema"."my.v"'
        )
        assert_creates_view(FLAT_SETUP, ("s.v",), 'memory.main."s.v"', rendered='"s.v"')
        assert_creates(FLAT_SETUP, ("s.n",), 'memory.main."s.n"', rendered='"s.n"')
        assert_inserts_into(FLAT_SETUP, FLAT_TABLES, ("s.t",), 'memory.main."s.t"', rendered='"s.t"')

    @pytest.mark.parametrize("method", WRITE_METHODS)
    def test_empty_tuple_refused(self, method):
        con = connect(FLAT_SETUP)
        before = table_count(con)
        with pytest.raises(duckdb.InvalidInputException, match="A name must have at least one part"):
            getattr(source(con), method)(())
        assert table_count(con) == before
        assert view_paths(con) == []

    @pytest.mark.parametrize("method", WRITE_METHODS)
    def test_non_string_part_refused(self, method):
        con = connect(FLAT_SETUP)
        before = table_count(con)
        with pytest.raises(duckdb.InvalidInputException, match="Every part of a name must be a string"):
            getattr(source(con), method)(("s", 1))
        assert table_count(con) == before
        assert view_paths(con) == []

    @pytest.mark.parametrize("method", WRITE_METHODS)
    @pytest.mark.parametrize("name", [None, 5])
    def test_other_type_refused(self, method, name):
        con = connect(FLAT_SETUP)
        before = table_count(con)
        with pytest.raises(duckdb.InvalidInputException) as error:
            getattr(source(con), method)(name)
        assert str(error.value) == "Invalid Input Error: A name is given as a string, or as a tuple of its parts"
        assert table_count(con) == before
        assert view_paths(con) == []

    @pytest.mark.parametrize("method", WRITE_METHODS)
    @pytest.mark.parametrize(("name", "rendered"), [(("s", ""), '"s".""'), (("",), '""'), (("", "t"), '""."t"')])
    def test_empty_part_refused(self, method, name, rendered):
        con = connect(FLAT_SETUP)
        before = table_count(con)
        with pytest.raises(duckdb.ParserException) as error:
            getattr(source(con), method)(name)
        assert str(error.value) == f"Parser Error: Zero-length identifier in qualified name! (input: {rendered})"
        assert table_count(con) == before
        assert view_paths(con) == []

    @pytest.mark.parametrize("method", WRITE_METHODS)
    def test_str_subclass_part_is_read_by_its_value(self, method):
        class Names(str, enum.Enum):
            SCHEMA = "s"
            TABLE = "t"
            NEW = "sub"

        con = connect(FLAT_SETUP)
        target = Names.TABLE if method == "insert_into" else Names.NEW
        getattr(source(con), method)((Names.SCHEMA, target))
        assert con.sql(f"FROM s.{target.value}").fetchall() == [(42,)]
        if method != "insert_into":
            assert stored_as(con, method, "sub") == [("s", "sub")]
        # The seam's own reading of the parts, which create_view reaches with plain strings
        assert duckdb._seam.table_kind(con, (Names.SCHEMA, Names.TABLE)) == "table"
        assert duckdb._seam.table_kind(con, Names.TABLE) == "table"
        read = con.table((Names.SCHEMA, Names.TABLE))
        assert read.fetchall() == [(42,)] if method == "insert_into" else read.fetchall() == []
        assert type(read.alias) is str

    def test_annotation(self):
        relation = duckdb.DuckDBPyRelation
        for method in ("create_view", "to_view"):
            assert getattr(relation, method).__annotations__["view_name"] == "str | tuple[str, ...]"
        for method in ("create", "to_table", "insert_into"):
            assert getattr(relation, method).__annotations__["table_name"] == "str | tuple[str, ...]"


class TestEmptyViewNamePart:
    @pytest.mark.parametrize("method", VIEW_METHODS)
    @pytest.mark.parametrize("name", ["", "s..v", ".v"])
    def test_create_view_refused(self, method, name):
        con = connect(FLAT_SETUP)
        with pytest.raises(duckdb.ParserException) as error:
            getattr(source(con), method)(name)
        assert str(error.value) == f"Parser Error: Zero-length identifier in qualified name! (input: {name})"
        assert view_paths(con) == []

    @pytest.mark.parametrize("method", WRITE_METHODS)
    @pytest.mark.parametrize("name", ['""', 's.""'])
    def test_empty_quoted_part_refused(self, method, name):
        con = connect(FLAT_SETUP)
        before = table_count(con)
        with pytest.raises(duckdb.ParserException) as error:
            getattr(source(con), method)(name)
        message = f"Parser Error: Zero-length delimited identifier in qualified name! (input: {name})"
        assert str(error.value) == message
        assert table_count(con) == before
        assert view_paths(con) == []

    @pytest.mark.parametrize("method", WRITE_METHODS)
    @pytest.mark.parametrize("name", ['"', 's."v'])
    def test_unterminated_quote_refused(self, method, name):
        con = connect(FLAT_SETUP)
        before = table_count(con)
        with pytest.raises(duckdb.ParserException) as error:
            getattr(source(con), method)(name)
        assert str(error.value) == f"Parser Error: Unterminated quote in qualified name! (input: {name})"
        assert table_count(con) == before
        assert view_paths(con) == []

    @pytest.mark.parametrize("method", WRITE_METHODS)
    @pytest.mark.parametrize("name", ["s..v", 'a"b"c', ("s", ""), ("",)])
    def test_refusal_keeps_the_transaction_usable(self, method, name):
        con = connect(FLAT_SETUP)
        con.execute("BEGIN")
        con.execute("CREATE TABLE sentinel AS SELECT 1 AS i")
        with pytest.raises(duckdb.ParserException):
            getattr(source(con), method)(name)
        con.execute("INSERT INTO sentinel VALUES (2)")
        assert con.sql("FROM sentinel ORDER BY i").fetchall() == [(1,), (2,)]
        con.execute("ROLLBACK")
        assert all(rows == [] for rows in contents(con, FLAT_TABLES).values())
        assert view_paths(con) == []


class TestViewOwnsItsSources:
    @pytest.mark.parametrize("name", ["s.v", ("db2", "s", "v")])
    def test_frame(self, name):
        con = connect(FLAT_SETUP)
        frame_source(con).create_view(name)
        gc.collect()
        target = name if isinstance(name, str) else ".".join(name)
        assert con.sql(f"FROM {target}").fetchall() == [(42,)]

    @pytest.mark.parametrize("name", ["s.v", ("db2", "s", "v")])
    def test_parameters(self, name):
        con = connect(FLAT_SETUP)
        con.sql("SELECT $1::INTEGER AS i", params=[42]).create_view(name)
        target = name if isinstance(name, str) else ".".join(name)
        assert con.sql(f"FROM {target}").fetchall() == [(42,)]

    def test_held_rows_refused(self):
        con = connect(FLAT_SETUP)
        with pytest.raises(duckdb.InvalidInputException, match=r'Cannot create the view "s"\."v": it reads rows held'):
            con.values([42]).create_view("s.v")
        assert view_paths(con) == []

    @pytest.mark.parametrize(("name", "quoted_path"), [("f.v", '"f"."v"'), (("f", "main", "v2"), '"f"."main"."v2"')])
    def test_frame_into_an_attached_file_database_refused(self, tmp_path, name, quoted_path):
        # The database the name resolves to decides: a file cannot hold a Python object, whatever the default is.
        con = duckdb.connect()
        con.execute(f"ATTACH '{tmp_path / 'f.db'}' AS f")
        before = view_paths(con)
        message = rf"Cannot create the view {re.escape(quoted_path)}: it reads a Python object"
        with pytest.raises(duckdb.InvalidInputException, match=message):
            frame_source(con).create_view(name)
        assert view_paths(con) == before
        frame_source(con).create_view("v3")
        assert con.sql("FROM v3").fetchall() == [(42,)]

    @pytest.mark.parametrize("name", ["m.v", ("m", "main", "v2")])
    def test_frame_into_an_attached_memory_database_from_a_file(self, tmp_path, name):
        con = duckdb.connect(str(tmp_path / "d.db"))
        con.execute("ATTACH ':memory:' AS m")
        frame_source(con).create_view(name)
        gc.collect()
        target = name if isinstance(name, str) else ".".join(name)
        assert con.sql(f"FROM {target}").fetchall() == [(42,)]
        with pytest.raises(
            duckdb.InvalidInputException, match=r'Cannot create the view "v3": it reads a Python object'
        ):
            frame_source(con).create_view("v3")


class TestUseInEffect:
    @pytest.mark.parametrize("method", VIEW_METHODS)
    @pytest.mark.parametrize(
        ("use", "name", "target"),
        [("db2.s", "v", "db2.s.v"), ("db2", "s.v", "db2.s.v"), ("db2", "memory.s.v", "memory.s.v")],
    )
    def test_create_view(self, method, use, name, target):
        via_sql = connect(FLAT_SETUP)
        via_sql.execute(f"USE {use}")
        via_sql.execute(f"CREATE VIEW {name} AS SELECT 42 AS i")
        via_relation = connect(FLAT_SETUP)
        via_relation.execute(f"USE {use}")
        getattr(source(via_relation), method)(name)
        for con in (via_sql, via_relation):
            con.execute("USE memory.main")
        assert via_relation.sql(f"FROM {target}").fetchall() == [(42,)]
        assert view_paths(via_relation) == view_paths(via_sql) == [quoted(target.split("."))]

    def test_create_and_insert_into(self):
        con = connect(FLAT_SETUP)
        con.execute("USE db2")
        source(con).create("s.n")
        source(con).insert_into("s.t")
        con.execute("USE memory.main")
        assert con.sql("FROM db2.s.n").fetchall() == [(42,)]
        assert contents(con, FLAT_TABLES) == {table: [(42,)] if table == "db2.s.t" else [] for table in FLAT_TABLES}


class TestNameSpellingRefused:
    """A quote inside a bare part, or anything but a dot after a closing quote, is refused as SQL refuses it."""

    @pytest.mark.parametrize("method", WRITE_METHODS)
    @pytest.mark.parametrize(
        ("name", "message"),
        [
            ('a"b"c', "Unexpected quote in the middle of a qualified name component!"),
            ('"a"b', "Unexpected character after a quoted identifier in a qualified name!"),
            ('s."a"b', "Unexpected character after a quoted identifier in a qualified name!"),
        ],
    )
    def test_refused(self, method, name, message):
        con = connect(FLAT_SETUP)
        before = table_count(con)
        with pytest.raises(duckdb.ParserException) as error:
            getattr(source(con), method)(name)
        assert str(error.value) == f"Parser Error: {message} (input: {name})"
        assert table_count(con) == before
        assert view_paths(con) == []


SPELLING_SETUP = [
    *FLAT_SETUP,
    'CREATE TABLE "select"(i INTEGER)',
    "CREATE SCHEMA sp",
    'CREATE TABLE sp." t"(i INTEGER)',
    'CREATE TABLE sp."T"(i INTEGER)',
]
SPELLING_TABLES = [*FLAT_TABLES, 'memory.main."select"', 'memory.sp." t"', 'memory.sp."T"']


def stored_as(con, method, stored):
    """The (schema, name) rows of the catalog listing the method writes to whose name has exactly this spelling."""
    listing, column = ("duckdb_views()", "view_name") if method in VIEW_METHODS else ("duckdb_tables()", "table_name")
    return con.sql(
        f"SELECT schema_name, {column} FROM {listing} WHERE {column} = ? ORDER BY ALL", params=[stored]
    ).fetchall()


class TestReaderRules:
    """What a read alone cannot show: a trailing dot is dropped, whitespace and case are kept, a keyword is a name."""

    @pytest.mark.parametrize("method", ["create", "to_table", "create_view", "to_view"])
    @pytest.mark.parametrize(
        ("name", "schema", "stored"),
        [
            ("s.new.", "s", "new"),
            ("new.", "main", "new"),
            ("s. new", "s", " new"),
            ("S.New", "s", "New"),
            ("select", "main", "select"),
            (("s", " new"), "s", " new"),
            (("S", "New"), "s", "New"),
            (("select",), "main", "select"),
        ],
    )
    def test_created_spelling(self, method, name, schema, stored):
        con = connect(FLAT_SETUP)
        getattr(source(con), method)(name)
        assert stored_as(con, method, stored) == [(schema, stored)]
        if stored != stored.strip().lower():
            assert stored_as(con, method, stored.strip().lower()) == []

    @pytest.mark.parametrize(
        ("name", "target"),
        [
            ("s.t.", "memory.s.t"),
            ("t.", "memory.main.t"),
            ("sp. t", 'memory.sp." t"'),
            ("SP.T", 'memory.sp."T"'),
            ("select", 'memory.main."select"'),
            (("sp", " t"), 'memory.sp." t"'),
            (("SP", "T"), 'memory.sp."T"'),
            (("select",), 'memory.main."select"'),
        ],
    )
    def test_insert_into(self, name, target):
        con = connect(SPELLING_SETUP)
        source(con).insert_into(name)
        expected = {table: [(42,)] if table == target else [] for table in SPELLING_TABLES}
        assert contents(con, SPELLING_TABLES) == expected

    def test_appender_reads_a_trailing_dot_the_same_way(self):
        con = connect(SPELLING_SETUP)
        appender = con.appender("s.t.")
        appender.append_row(7)
        appender.flush()
        source(con).insert_into("s.t.")
        assert con.sql("FROM s.t ORDER BY i").fetchall() == [(7,), (42,)]

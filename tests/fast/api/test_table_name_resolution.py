import pytest

import duckdb
import pandas as pd

WRITES = "CREATE TABLE memory.main.writes (k INTEGER)"


def register_frames(con):
    con.register("r", pd.DataFrame({"k": ["registered r"]}))
    con.register("s.r", pd.DataFrame({"k": ["registered s.r"]}))


def register_dotted_frame(con):
    con.register("s.r", pd.DataFrame({"k": ["registered s.r"]}))


def register_dotted_frame_named_like_a_view(con):
    con.register("s.v", pd.DataFrame({"k": ["registered s.v"]}))


BASE = [
    WRITES,
    "CREATE TABLE t AS SELECT 't' AS k",
    "CREATE VIEW v AS SELECT 'v' AS k",
    "CREATE SCHEMA s",
    "CREATE TABLE s.t AS SELECT 's.t' AS k",
    "CREATE VIEW s.v AS SELECT 's.v' AS k",
    "CREATE TABLE \"a.b\" AS SELECT 'table a.b' AS k",
    "ATTACH ':memory:' AS other",
    "CREATE TABLE other.ot AS SELECT 'other.ot' AS k",
    "CREATE VIEW other.ov AS SELECT 'other.ov' AS k",
    "CREATE TEMP VIEW tv AS SELECT 'temp.tv' AS k",
    "CREATE TABLE shadow AS SELECT 'main table shadow' AS k",
    "CREATE TEMP VIEW shadow AS SELECT 'temp view shadow' AS k",
    register_frames,
]
SEARCH_PATH_ON_SCHEMA = [*BASE, "SET search_path = 's'"]
DOTTED_FRAME_WITHOUT_SCHEMA = [WRITES, register_dotted_frame]
SCHEMA_VIEW_BESIDE_DOTTED_FRAME = [
    WRITES,
    "CREATE SCHEMA s",
    "CREATE VIEW s.r AS SELECT 'schema view s.r' AS k",
    register_dotted_frame,
]
DATABASE_VIEW_BESIDE_DOTTED_FRAME = [
    WRITES,
    "ATTACH ':memory:' AS s",
    "CREATE VIEW s.main.v AS SELECT 'database view s.v' AS k",
    register_dotted_frame_named_like_a_view,
]
NESTED = [
    WRITES,
    "CREATE SCHEMA p",
    "CREATE SCHEMA p.child",
    "CREATE VIEW p.child.v AS SELECT 'p.child.v' AS k",
    "CREATE TABLE p.child.t AS SELECT 'p.child.t' AS k",
]
NESTED_BESIDE_TOP_LEVEL_SCHEMA = [
    *NESTED,
    "CREATE SCHEMA child",
    "CREATE VIEW child.v AS SELECT 'top level child.v' AS k",
    "CREATE TABLE child.t AS SELECT 'top level child.t' AS k",
]
AMBIGUOUS = [
    WRITES,
    "CREATE SCHEMA s",
    "CREATE VIEW s.v AS SELECT 'schema s.v' AS k",
    "CREATE TABLE s.t AS SELECT 'schema s.t' AS k",
    "ATTACH ':memory:' AS s",
    "CREATE VIEW s.main.v AS SELECT 'catalog s.main.v' AS k",
    "CREATE TABLE s.main.t AS SELECT 'catalog s.main.t' AS k",
]
SEARCH_PATH_ON_ATTACHED_SCHEMA = [
    WRITES,
    "ATTACH ':memory:' AS other",
    "CREATE SCHEMA other.s",
    "CREATE TABLE other.main.t AS SELECT 1 AS k",
    "CREATE TABLE other.s.t AS SELECT 2 AS k",
    "CREATE TABLE other.main.v AS SELECT 1 AS k",
    "CREATE VIEW other.s.v AS SELECT 2 AS k",
    "CREATE VIEW other.main.w AS SELECT 1 AS k",
    "CREATE VIEW other.s.w AS SELECT 2 AS k",
    "SET search_path = 'other.s'",
]

frame_var = pd.DataFrame({"k": ["frame_var"]})

TABLE = "TABLE_RELATION"
QUERY = "QUERY_RELATION"


def case(case_id, setup, name, *expected, sql=None, outside=None, inside=None):
    return case_id, (setup, name, *expected), sql or name, outside, inside


READS = [
    case("table", BASE, "t", "t", TABLE),
    case("view", BASE, "v", "v", QUERY),
    case("table_on_the_search_path", SEARCH_PATH_ON_SCHEMA, "t", "s.t", TABLE),
    case("main_table", BASE, "main.t", "t", TABLE),
    case("main_view", BASE, "main.v", "v", QUERY),
    case("schema_table", BASE, "s.t", "s.t", TABLE),
    case("schema_view", BASE, "s.v", "s.v", QUERY),
    case("schema_view_other_case", BASE, "S.V", "s.v", QUERY),
    case("catalog_schema_table", BASE, "memory.main.t", "t", TABLE),
    case("catalog_schema_view", BASE, "memory.main.v", "v", QUERY),
    case("catalog_table", BASE, "memory.t", "t", TABLE),
    case("catalog_view", BASE, "memory.v", "v", QUERY),
    case("attached_table", BASE, "other.ot", "other.ot", TABLE),
    case("attached_view", BASE, "other.ov", "other.ov", QUERY),
    case("attached_schema_view", BASE, "other.main.ov", "other.ov", QUERY),
    case("quoted_schema_table", BASE, '"s"."t"', "s.t", TABLE),
    case("quoted_schema_view", BASE, '"s"."v"', "s.v", QUERY),
    case("quoted_schema_unquoted_view", BASE, '"s".v', "s.v", QUERY),
    case("quoted_dotted_table", BASE, '"a.b"', "table a.b", TABLE),
    case("dotted_table_name", BASE, "a.b", "table a.b", QUERY, sql='"a.b"'),
    case("temp_view", BASE, "tv", "temp.tv", QUERY),
    case("temp_catalog_view", BASE, "temp.tv", "temp.tv", QUERY),
    case("temp_catalog_schema_view", BASE, "temp.main.tv", "temp.tv", QUERY),
    case("temp_view_shadows_table", BASE, "shadow", "temp view shadow", QUERY),
    case("main_name_finds_temp_view_first", BASE, "main.shadow", "temp view shadow", QUERY),
    case("catalog_name_skips_temp_view", BASE, "memory.shadow", "main table shadow", TABLE),
    case("temp_catalog_shadowing_view", BASE, "temp.shadow", "temp view shadow", QUERY),
    case("dataframe_variable", BASE, "frame_var", "frame_var", QUERY),
    case("registered_frame", BASE, "r", "registered r", QUERY),
    case("temp_registered_frame", BASE, "temp.r", "registered r", QUERY),
    case("quoted_dotted_registered_frame", BASE, '"s.r"', "registered s.r", QUERY),
    case("dotted_frame_without_schema", DOTTED_FRAME_WITHOUT_SCHEMA, "s.r", "registered s.r", QUERY, sql='"s.r"'),
    case(
        "schema_view_wins_over_dotted_frame",
        SCHEMA_VIEW_BESIDE_DOTTED_FRAME,
        "s.r",
        "schema view s.r",
        QUERY,
    ),
    case(
        "database_view_wins_over_dotted_frame",
        DATABASE_VIEW_BESIDE_DOTTED_FRAME,
        "s.v",
        "database view s.v",
        QUERY,
    ),
    case(
        "database_table_on_the_search_path",
        SEARCH_PATH_ON_ATTACHED_SCHEMA,
        "other.t",
        2,
        TABLE,
    ),
    case(
        "database_view_on_the_search_path_beside_a_main_table",
        SEARCH_PATH_ON_ATTACHED_SCHEMA,
        "other.v",
        2,
        QUERY,
    ),
    case(
        "database_view_on_the_search_path_beside_a_main_view",
        SEARCH_PATH_ON_ATTACHED_SCHEMA,
        "other.w",
        2,
        QUERY,
    ),
    case("nested_view", NESTED, "memory.p.child.v", "p.child.v", QUERY),
    case("nested_table", NESTED, "memory.p.child.t", "p.child.t", TABLE),
    case("nested_view_without_catalog", NESTED, "p.child.v", "p.child.v", QUERY),
    case("nested_table_without_catalog", NESTED, "p.child.t", "p.child.t", TABLE),
    case("nested_view_beside_top_level_schema", NESTED_BESIDE_TOP_LEVEL_SCHEMA, "memory.p.child.v", "p.child.v", QUERY),
    case(
        "nested_table_beside_top_level_schema",
        NESTED_BESIDE_TOP_LEVEL_SCHEMA,
        "memory.p.child.t",
        "p.child.t",
        TABLE,
    ),
]

FAILURES = [
    case("missing", BASE, "missing", duckdb.CatalogException),
    case("missing_in_schema", BASE, "s.missing", duckdb.CatalogException),
    case("neither_schema_nor_database", BASE, "nosuch.v", duckdb.CatalogException, sql='"nosuch.v"'),
    case("ambiguous_view", AMBIGUOUS, "s.v", duckdb.BinderException),
    case("ambiguous_table", AMBIGUOUS, "s.t", duckdb.BinderException),
    # A schema wins over a frame registered with the whole dotted name, as in SQL.
    case("dotted_frame_beside_schema", BASE, "s.r", duckdb.CatalogException),
]

FILES = [
    pytest.param("people.csv", "people.csv", id="plain"),
    pytest.param('"people.csv"', "people.csv", id="quoted"),
    pytest.param("./people.csv", "people.csv", id="current_directory"),
    pytest.param("../files/people.csv", "people.csv", id="parent_directory"),
    pytest.param("my.data.csv", "my.data.csv", id="two_dots"),
    pytest.param("a.b.c.csv", "a.b.c.csv", id="three_dots"),
    pytest.param("dir.d/people.csv", "dir.d/people.csv", id="dotted_directory"),
]
IN_TRANSACTION = pytest.mark.parametrize("in_transaction", [False, True], ids=["autocommit", "transaction"])


def params(cases, inside_a_transaction, with_sql=False):
    result = []
    for case_id, values, sql, outside, inside in cases:
        mark = inside if inside_a_transaction else outside
        args = (*values, sql) if with_sql else values
        result.append(pytest.param(*args, id=case_id, marks=[mark] if mark else []))
    return result


def connect(setup):
    con = duckdb.connect()
    for step in setup:
        if callable(step):
            step(con)
        else:
            con.execute(step)
    return con


def begin(con):
    con.execute("BEGIN")
    con.execute("INSERT INTO memory.main.writes VALUES (1)")


def writes_after_second_insert(con, in_transaction):
    try:
        con.execute("INSERT INTO memory.main.writes VALUES (2)")
        if in_transaction:
            con.execute("COMMIT")
    except duckdb.TransactionException:
        con.execute("ROLLBACK")
    return con.execute("SELECT k FROM memory.main.writes ORDER BY k").fetchall()


def failure(setup, read, in_transaction):
    with connect(setup) as con:
        if in_transaction:
            begin(con)
        with pytest.raises(duckdb.Error) as raised:
            read(con).fetchall()
        return raised.type, str(raised.value).splitlines()[0], writes_after_second_insert(con, in_transaction)


@pytest.mark.parametrize(("setup", "name", "row", "relation_type"), params(READS, inside_a_transaction=False))
def test_reads_what_sql_reads(setup, name, row, relation_type):
    with connect(setup) as con:
        rel = con.table(name)
        assert rel.type == relation_type
        assert rel.fetchall() == [(row,)]


@pytest.mark.parametrize(("setup", "name", "row", "relation_type"), params(READS, inside_a_transaction=True))
def test_reads_what_sql_reads_and_keeps_the_transaction(setup, name, row, relation_type):
    with connect(setup) as con:
        begin(con)
        assert con.table(name).fetchall() == [(row,)]
        assert writes_after_second_insert(con, in_transaction=True) == [(1,), (2,)]


@pytest.mark.parametrize(
    ("setup", "sql", "row"), [pytest.param(values[0], sql, values[2], id=case_id) for case_id, values, sql, *_ in READS]
)
def test_sql_reads_the_expected_row(setup, sql, row):
    with connect(setup) as con:
        assert con.sql(f"FROM {sql}").fetchall() == [(row,)]


@pytest.mark.parametrize(("setup", "name", "error", "sql"), params(FAILURES, inside_a_transaction=False, with_sql=True))
def test_fails_as_sql_fails(setup, name, error, sql):
    table_failure = failure(setup, lambda con: con.table(name), in_transaction=False)
    assert table_failure[0] is error
    assert table_failure == failure(setup, lambda con: con.sql(f"FROM {sql}"), in_transaction=False)


@pytest.mark.parametrize(("setup", "name", "error", "sql"), params(FAILURES, inside_a_transaction=True, with_sql=True))
def test_fails_as_sql_fails_inside_a_transaction(setup, name, error, sql):
    table_failure = failure(setup, lambda con: con.table(name), in_transaction=True)
    assert table_failure[0] is error
    assert table_failure == failure(setup, lambda con: con.sql(f"FROM {sql}"), in_transaction=True)


@pytest.mark.parametrize(
    ("setup", "name", "column", "row"),
    [
        pytest.param(BASE, "s.v", "v.k", "s.v", id="schema_view"),
        pytest.param(BASE, '"s"."v"', "v.k", "s.v", id="quoted_schema_view"),
        pytest.param(BASE, "memory.main.v", "v.k", "v", id="catalog_schema_view"),
        pytest.param(BASE, "s.t", "t.k", "s.t", id="schema_table"),
        pytest.param(NESTED, "memory.p.child.v", "v.k", "p.child.v", id="nested_view"),
    ],
)
def test_alias_is_the_last_part(setup, name, column, row):
    with connect(setup) as con:
        rel = con.table(name)
        assert rel.alias == column.split(".")[0]
        assert rel.project(column).fetchall() == [(row,)]


@pytest.fixture
def files(tmp_path, monkeypatch):
    for relative in ["people.csv", "my.data.csv", "a.b.c.csv", "dir.d/people.csv"]:
        path = tmp_path / "files" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"k\nfile {relative}\n")
    monkeypatch.chdir(tmp_path / "files")
    return tmp_path / "files"


@IN_TRANSACTION
@pytest.mark.parametrize(("name", "relative"), FILES)
@pytest.mark.usefixtures("files")
def test_reads_a_file(in_transaction, name, relative):
    csv = pd.DataFrame({"k": ["frame named csv"]})  # noqa: F841
    with connect([WRITES]) as con:
        if in_transaction:
            begin(con)
        assert con.table(name).fetchall() == [(f"file {relative}",)]
        assert writes_after_second_insert(con, in_transaction) == ([(1,), (2,)] if in_transaction else [(2,)])


@IN_TRANSACTION
def test_reads_a_file_by_absolute_path(files, in_transaction):
    csv = pd.DataFrame({"k": ["frame named csv"]})  # noqa: F841
    with connect([WRITES]) as con:
        if in_transaction:
            begin(con)
        assert con.table(str(files / "dir.d" / "people.csv")).fetchall() == [("file dir.d/people.csv",)]
        assert writes_after_second_insert(con, in_transaction) == ([(1,), (2,)] if in_transaction else [(2,)])


@pytest.mark.usefixtures("files")
def test_name_with_an_empty_part_reads_the_file_not_the_table_its_parts_name():
    with connect([WRITES, 'CREATE SCHEMA "/people"', "CREATE TABLE \"/people\".csv AS SELECT 'decoy' AS k"]) as con:
        assert con.table("./people.csv").fetchall() == [("file people.csv",)]


def test_module_function_reads_a_qualified_view():
    with connect(BASE) as con:
        assert duckdb.table("s.v", connection=con).fetchall() == [("s.v",)]

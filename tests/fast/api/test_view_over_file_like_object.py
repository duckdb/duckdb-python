"""A view keeps what its relation reads, such as the store entry of a file like object, for as long as it exists."""

import gc
import io
import weakref

import pytest

import duckdb

pytest.importorskip("fsspec")

CSV = "a,b\n1,x\n2,y\n"
JSON = '[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]'
ROWS = [(1, "x"), (2, "y")]


@pytest.fixture(scope="module")
def parquet_data(tmp_path_factory):
    path = tmp_path_factory.mktemp("parquet") / "t.parquet"
    with duckdb.connect() as con:
        con.execute(f"copy (select * from (values (1, 'x'), (2, 'y')) t(a, b)) to '{path}'")
    return path.read_bytes()


def object_store_files():
    from duckdb.filesystem import ModifiedMemoryFileSystem

    return set(ModifiedMemoryFileSystem.store)


def read_view(con):
    return con.sql("select a, b from v order by a").fetchall()


def read_csv_text(con, parquet_data):
    return con.read_csv(io.StringIO(CSV))


def read_csv_bytes(con, parquet_data):
    return con.read_csv(io.BytesIO(CSV.encode()))


def read_json_text(con, parquet_data):
    # Schema detection waits on worker threads that need the GIL this thread holds while binding.
    con.execute("set threads = 1")
    return con.read_json(io.StringIO(JSON))


def read_parquet_bytes(con, parquet_data):
    return con.read_parquet(io.BytesIO(parquet_data))


def read_parquet_list(con, parquet_data):
    return con.read_parquet([io.BytesIO(parquet_data)])


READERS = [
    pytest.param(read_csv_text, id="read_csv StringIO"),
    pytest.param(read_csv_bytes, id="read_csv BytesIO"),
    pytest.param(read_json_text, id="read_json StringIO"),
    pytest.param(read_parquet_bytes, id="read_parquet BytesIO"),
    pytest.param(read_parquet_list, id="read_parquet list"),
]


class TestViewOutlivesItsRelation:
    @pytest.mark.parametrize("read", READERS)
    def test_to_view(self, duckdb_cursor, parquet_data, read):
        read(duckdb_cursor, parquet_data).to_view("v")
        gc.collect()
        assert read_view(duckdb_cursor) == ROWS

    @pytest.mark.parametrize("replace", [pytest.param(True, id="replace"), pytest.param(False, id="no replace")])
    def test_create_view(self, duckdb_cursor, replace):
        duckdb_cursor.read_csv(io.StringIO(CSV)).create_view("v", replace=replace)
        gc.collect()
        assert read_view(duckdb_cursor) == ROWS

    def test_rolled_back_drop(self, duckdb_cursor):
        duckdb_cursor.read_csv(io.StringIO(CSV)).to_view("v")
        gc.collect()
        duckdb_cursor.execute("begin")
        duckdb_cursor.execute("drop view v")
        duckdb_cursor.execute("rollback")
        gc.collect()
        assert read_view(duckdb_cursor) == ROWS

    def test_query_temporary_view(self, duckdb_cursor):
        duckdb_cursor.read_csv(io.StringIO(CSV)).query("v", "select count(*) from v")
        gc.collect()
        assert read_view(duckdb_cursor) == ROWS

    def test_query_temporary_view_of_a_statement(self, duckdb_cursor):
        duckdb_cursor.read_csv(io.StringIO(CSV)).query("v", "create table t as select * from v")
        gc.collect()
        assert duckdb_cursor.sql("select a, b from t order by a").fetchall() == ROWS
        assert read_view(duckdb_cursor) == ROWS

    def test_registered_relation(self, duckdb_cursor):
        duckdb_cursor.register("v", duckdb_cursor.read_csv(io.StringIO(CSV)))
        gc.collect()
        assert read_view(duckdb_cursor) == ROWS


class TestViewKeepsItsShape:
    def test_empty_view_name(self, duckdb_cursor):
        duckdb_cursor.read_csv(io.StringIO(CSV)).to_view("")
        assert duckdb_cursor.sql("select view_name from duckdb_views() where not internal").fetchall() == [("",)]

    def test_duplicate_column_names(self, duckdb_cursor):
        duckdb_cursor.sql("select 1 as a, 2 as a").to_view("v")
        assert [row[1] for row in duckdb_cursor.sql("pragma table_info('v')").fetchall()] == ["a", "a"]
        rel = duckdb_cursor.sql("select * from v")
        assert rel.columns == ["a", "a_1"]
        assert rel.fetchall() == [(1, 2)]


def drop_view(con):
    con.execute("drop view v")


def replace_view(con):
    con.sql("select 3 as a, 'z' as b").to_view("v")


def query_again(con):
    con.sql("select 3 as a, 'z' as b").query("v", "select count(*) from v")


def close(con):
    con.close()


RELEASES = [
    pytest.param(drop_view, id="dropped"),
    pytest.param(replace_view, id="replaced"),
    pytest.param(close, id="connection closed"),
]


def assert_store_entry_lives_with_the_view(make, release):
    before = object_store_files()
    con = duckdb.connect()
    make(con)
    gc.collect()
    assert read_view(con) == ROWS
    assert len(object_store_files() - before) == 1
    release(con)
    gc.collect()
    assert object_store_files() - before == set()
    con.close()


class TestStoreEntryLivesWithTheView:
    @pytest.mark.parametrize("release", RELEASES)
    @pytest.mark.parametrize("read", READERS)
    def test_view(self, parquet_data, read, release):
        assert_store_entry_lives_with_the_view(lambda con: read(con, parquet_data).to_view("v"), release)

    @pytest.mark.parametrize("release", RELEASES)
    def test_view_created_without_replace(self, release):
        assert_store_entry_lives_with_the_view(
            lambda con: con.read_csv(io.StringIO(CSV)).create_view("v", replace=False), release
        )

    @pytest.mark.parametrize(
        "release",
        [
            pytest.param(drop_view, id="dropped"),
            pytest.param(query_again, id="next query"),
            pytest.param(close, id="connection closed"),
        ],
    )
    def test_temporary_view(self, release):
        assert_store_entry_lives_with_the_view(
            lambda con: con.read_csv(io.StringIO(CSV)).query("v", "select count(*) from v"), release
        )

    def test_view_over_a_join(self):
        before = object_store_files()
        con = duckdb.connect()
        left = con.read_csv(io.StringIO(CSV))
        right = con.read_csv(io.StringIO("a,c\n1,p\n2,q\n"))
        left.join(right, "a").to_view("v")
        del left, right
        gc.collect()
        assert con.sql("select a, b, c from v order by a").fetchall() == [(1, "x", "p"), (2, "y", "q")]
        assert len(object_store_files() - before) == 2
        con.execute("drop view v")
        gc.collect()
        assert object_store_files() - before == set()
        con.close()

    def test_view_over_a_union(self):
        before = object_store_files()
        con = duckdb.connect()
        con.read_csv(io.StringIO(CSV)).union(con.read_csv(io.StringIO("a,b\n3,z\n"))).to_view("v")
        gc.collect()
        assert read_view(con) == [*ROWS, (3, "z")]
        assert len(object_store_files() - before) == 2
        con.execute("drop view v")
        gc.collect()
        assert object_store_files() - before == set()
        con.close()

    def test_view_that_reads_its_source_only_through_the_query_name(self):
        before = object_store_files()
        con = duckdb.connect()
        con.read_csv(io.StringIO(CSV)).query("x", "select (select count(*) from x) as n").to_view("v")
        con.execute("drop view x")
        gc.collect()
        assert con.sql("select n from v").fetchall() == [(2,)]
        assert len(object_store_files() - before) == 1
        con.execute("drop view v")
        gc.collect()
        assert object_store_files() - before == set()
        con.close()

    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            pytest.param("with x as (select 0) select (select sum(a) from ({source})) as n", 3, id="select list"),
            pytest.param("with x as (select 0) select 3 as n where (select sum(a) from ({source})) = 3", 3, id="where"),
            pytest.param(
                "with x as (select 0) select case when true then (select sum(a) from ({source})) end as n",
                3,
                id="case",
            ),
            pytest.param(
                "with x as (select 0) select (select sum(a) from ({source})) = any(select 3) as n",
                True,
                id="any operand",
            ),
        ],
    )
    def test_view_that_reads_its_source_only_through_a_subquery_expression(self, sql, expected):
        before = object_store_files()
        con = duckdb.connect()
        source = con.read_csv(io.StringIO(CSV))
        query = source.query("x", sql.format(source=source.sql_query()))
        del source
        con.execute("drop view x")
        query.to_view("v")
        del query
        gc.collect()
        assert con.sql("select n from v").fetchall() == [(expected,)]
        assert len(object_store_files() - before) == 1
        con.execute("drop view v")
        gc.collect()
        assert object_store_files() - before == set()
        con.close()

    def test_rolled_back_replacement(self):
        before = object_store_files()
        con = duckdb.connect()
        con.read_csv(io.StringIO(CSV)).to_view("v")
        con.execute("begin")
        con.read_csv(io.StringIO("a,b\n3,z\n")).create_view("v", replace=True)
        gc.collect()
        assert len(object_store_files() - before) == 2
        con.execute("rollback")
        gc.collect()
        assert read_view(con) == ROWS
        assert len(object_store_files() - before) == 1
        con.close()

    @pytest.mark.xfail(
        strict=True,
        raises=SystemError,
        reason="pre existing error path: a failing call on an unnamed file like relation raises SystemError "
        "(error return without exception set) and leaves its store entry, as to_table and insert_into do",
    )
    def test_failed_create_view(self, duckdb_cursor):
        duckdb_cursor.execute("create view v as select 1 as a")
        before = object_store_files()
        with pytest.raises(duckdb.CatalogException, match="already exists"):
            duckdb_cursor.read_csv(io.StringIO(CSV)).create_view("v", replace=False)
        gc.collect()
        assert object_store_files() - before == set()


class TestDataFrameLivesWithTheView:
    @pytest.mark.parametrize(
        "release", [pytest.param(drop_view, id="dropped"), pytest.param(close, id="connection closed")]
    )
    def test_released(self, release):
        pd = pytest.importorskip("pandas")
        con = duckdb.connect()
        df = pd.DataFrame({"a": [1, 2]})
        frame = weakref.ref(df)
        con.from_df(df).to_view("v")
        del df
        gc.collect()
        assert frame() is not None
        assert con.sql("select a from v order by a").fetchall() == [(1,), (2,)]
        release(con)
        gc.collect()
        assert frame() is None
        con.close()

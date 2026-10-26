import pytest

import duckdb
from duckdb.sqltypes import BIGINT, BOOLEAN, HUGEINT, INTEGER

ROWS = 300_000
TALLY_QUERY = f"SELECT tally(i) AS i FROM range({ROWS}) t(i)"


@pytest.fixture
def produced(duckdb_cursor):
    """Counts the rows the tally query produced, which shows whether and how far it ran."""
    count = [0]

    def tally(i):
        count[0] += 1
        return i

    duckdb_cursor.create_function("tally", tally, [BIGINT], BIGINT, side_effects=True)
    return count


@pytest.fixture
def returning_table(duckdb_cursor):
    duckdb_cursor.execute("CREATE TABLE r (i INTEGER)")
    return duckdb_cursor


def count_rows(con, table="r"):
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


class TestExecutedQuery:
    def test_description_is_known_after_execute(self, duckdb_cursor, produced):
        duckdb_cursor.execute(TALLY_QUERY)
        assert [column[0] for column in duckdb_cursor.description] == ["i"]
        assert duckdb_cursor.description[0][1] == BIGINT
        assert duckdb_cursor.rowcount == -1

    @pytest.mark.parametrize(
        "consume",
        [
            pytest.param(lambda con: len(con.fetchall()), id="fetchall"),
            pytest.param(lambda con: len(con.fetchmany(ROWS + 1)), id="fetchmany"),
            pytest.param(lambda con: len(con.df()), id="df"),
            pytest.param(lambda con: len(con.fetchnumpy()["i"]), id="fetchnumpy"),
        ],
    )
    def test_the_first_consumer_runs_the_query_once(self, duckdb_cursor, produced, consume):
        pytest.importorskip("pandas")
        duckdb_cursor.execute(TALLY_QUERY)
        assert consume(duckdb_cursor) == ROWS
        assert produced[0] == ROWS

    def test_a_whole_fetch_after_a_row_fetch_returns_the_remainder(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT i FROM range(10) t(i)")
        assert duckdb_cursor.fetchone() == (0,)
        assert duckdb_cursor.fetchall() == [(i,) for i in range(1, 10)]

    def test_a_runtime_error_surfaces_in_execute(self, duckdb_cursor):
        with pytest.raises(duckdb.ConversionException):
            duckdb_cursor.execute("SELECT 'a'::INTEGER")

    def test_a_bind_error_surfaces_in_execute(self, duckdb_cursor):
        with pytest.raises(duckdb.CatalogException):
            duckdb_cursor.execute("SELECT * FROM missing_table")

    def test_a_parameter_without_a_known_type_is_described_after_binding(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT ? AS p", [42])
        assert duckdb_cursor.description[0][1] == INTEGER
        assert duckdb_cursor.fetchall() == [(42,)]


class TestStatementGroup:
    """A statement the engine expands into several learns its columns only once the one producing the rows is bound."""

    @pytest.fixture
    def pivot_source(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE p AS SELECT * FROM (VALUES ('a', 1), ('b', 2), ('a', 3)) t(k, v)")
        return duckdb_cursor

    def test_description_is_known_after_execute(self, pivot_source):
        pivot_source.execute("PIVOT p ON k USING sum(v)")
        assert [(name, kind) for name, kind, *_ in pivot_source.description] == [("a", HUGEINT), ("b", HUGEINT)]
        assert pivot_source.fetchall() == [(4, 2)]

    def test_a_group_without_rows_fetches_nothing(self, pivot_source):
        pivot_source.execute("ALTER TABLE p ADD COLUMN r DOUBLE DEFAULT random()")
        assert [(name, kind) for name, kind, *_ in pivot_source.description] == [("Success", BOOLEAN)]
        assert pivot_source.rowcount == -1
        assert pivot_source.fetchall() == []
        assert pivot_source.execute("SELECT count(*) FILTER (r IS NOT NULL) FROM p").fetchone() == (3,)

    def test_a_failure_in_the_result_statement_undoes_the_group(self, pivot_source):
        with pytest.raises(duckdb.BinderException, match="missing"):
            pivot_source.execute("PIVOT p ON k USING sum(v) ORDER BY missing")
        enums = "SELECT count(*) FROM duckdb_types() WHERE type_name LIKE '__pivot_enum%'"
        assert pivot_source.execute(enums).fetchone() == (0,)

    def test_a_group_as_a_relation(self, pivot_source):
        rel = pivot_source.sql("PIVOT p ON k USING sum(v)")
        assert rel.columns == ["a", "b"]
        assert rel.types == [HUGEINT, HUGEINT]
        assert rel.fetchall() == [(4, 2)]


class TestAnotherStatement:
    @pytest.mark.parametrize("read_first", [False, True], ids=["settled", "streaming"])
    @pytest.mark.parametrize(
        "statement",
        [
            pytest.param(lambda con: con.sql("SELECT 1").fetchall(), id="relation"),
            pytest.param(lambda con: con.sql("CREATE TABLE side (i INTEGER)"), id="ddl"),
            pytest.param(lambda con: con.register("view_of_df", con.sql("SELECT 1")), id="register"),
        ],
    )
    def test_it_is_refused_and_leaves_an_unfinished_result(self, duckdb_cursor, read_first, statement):
        duckdb_cursor.execute("SET max_streaming_buffer_size='100KB'")
        duckdb_cursor.execute(f"SELECT i FROM range({ROWS}) t(i)")
        if read_first:
            assert duckdb_cursor.fetchone() == (0,)
        with pytest.raises(duckdb.ResourceInUseException, match="unfinished result from execute"):
            statement(duckdb_cursor)
        assert len(duckdb_cursor.fetchall()) == ROWS - read_first

    def test_it_leaves_a_completed_result(self, returning_table):
        con = returning_table
        con.execute("CALL pragma_table_info('r')")
        con.sql("SELECT 1").fetchall()
        assert [row[1] for row in con.fetchall()] == ["i"]

    def test_a_cursor_is_its_own_connection(self, duckdb_cursor):
        cursor = duckdb_cursor.cursor()
        cursor.execute("SELECT i FROM range(10) t(i)")
        duckdb_cursor.execute("SELECT 1").fetchall()
        assert len(cursor.fetchall()) == 10


class TestCompleteAndAbort:
    def test_complete_runs_a_settled_query_without_reading_it(self, duckdb_cursor, produced):
        duckdb_cursor.execute(TALLY_QUERY)
        duckdb_cursor.complete()
        assert produced[0] == ROWS
        assert duckdb_cursor.fetchall() == []

    def test_complete_finishes_a_partly_read_stream(self, duckdb_cursor, produced):
        duckdb_cursor.execute(TALLY_QUERY)
        assert duckdb_cursor.fetchone() == (0,)
        duckdb_cursor.complete()
        assert produced[0] == ROWS
        assert duckdb_cursor.fetchall() == []

    def test_complete_applies_an_unread_returning_write(self, returning_table):
        con = returning_table
        con.execute("INSERT INTO r VALUES (1), (2) RETURNING i")
        con.complete()
        assert count_rows(con) == 2

    def test_abort_leaves_a_completed_write(self, returning_table):
        con = returning_table
        con.execute("INSERT INTO r VALUES (1)")
        con.abort()
        assert count_rows(con) == 1

    def test_on_the_default_connection(self):
        duckdb.execute("SELECT 42")
        duckdb.complete()
        duckdb.execute("SELECT 42")
        duckdb.abort()
        assert duckdb.execute("SELECT 42").fetchall() == [(42,)]


class TestWrites:
    def test_dml_sets_the_rowcount(self, returning_table):
        assert returning_table.execute("INSERT INTO r SELECT * FROM range(5)").rowcount == 5

    def test_a_read_returning_write_is_applied(self, returning_table):
        con = returning_table
        assert con.execute("INSERT INTO r VALUES (1), (2) RETURNING i").fetchall() == [(1,), (2,)]
        assert count_rows(con) == 2

    def test_commit_keeps_an_unread_returning_write(self, returning_table):
        con = returning_table
        con.execute("BEGIN")
        con.execute("INSERT INTO r VALUES (1) RETURNING i")
        con.commit()
        assert count_rows(con) == 1

    def test_an_unread_nextval_advances(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE SEQUENCE s")
        duckdb_cursor.execute("SELECT nextval('s')")
        assert duckdb_cursor.execute("SELECT nextval('s')").fetchall() == [(2,)]

    def test_explain_analyze_of_a_write_runs_in_execute(self, returning_table):
        con = returning_table
        con.execute("EXPLAIN ANALYZE INSERT INTO r VALUES (1)")
        assert count_rows(con) == 1

    def test_a_sql_execute_runs_in_execute(self, returning_table):
        con = returning_table
        con.execute("PREPARE p AS INSERT INTO r VALUES (1) RETURNING i")
        con.execute("EXECUTE p")
        assert count_rows(con) == 1

    def test_copy_returning_files_runs_in_execute(self, duckdb_cursor, tmp_path):
        target = tmp_path / "out.csv"
        duckdb_cursor.execute(f"COPY (SELECT 1 AS x) TO '{target}' (RETURN_FILES)")
        duckdb_cursor.execute("SELECT 1")
        assert target.exists()

    def test_checkpoint_runs_in_execute(self, tmp_path):
        path = tmp_path / "db.duckdb"
        wal = tmp_path / "db.duckdb.wal"
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE t AS SELECT range i FROM range(1000)")
        assert wal.exists()
        assert wal.stat().st_size > 0
        con.execute("CHECKPOINT")
        con.execute("SELECT 1")
        assert not wal.exists() or wal.stat().st_size == 0
        con.close()

    def test_executemany_runs_every_set(self, returning_table):
        con = returning_table
        con.executemany("INSERT INTO r VALUES (?)", [[1], [2], [3]])
        assert count_rows(con) == 3


class TestPythonObjects:
    def test_a_statement_reads_the_object_it_was_bound_with(self, duckdb_cursor):
        pa = pytest.importorskip("pyarrow")
        con = duckdb_cursor

        def over_a_local_table():
            tbl = pa.table({"x": [1, 2, 3]})  # noqa: F841 - read by the replacement scan
            return con.execute("SELECT sum(x) FROM tbl")

        result = over_a_local_table()
        tbl = pa.table({"x": [100]})  # noqa: F841
        assert result.fetchall() == [(6,)]

    def test_names_differing_only_in_case_are_different_objects(self, duckdb_cursor):
        pa = pytest.importorskip("pyarrow")
        tbl = pa.table({"x": [1]})  # noqa: F841 - read by the replacement scan
        TBL = pa.table({"x": [10]})  # noqa: F841 - read by the replacement scan
        duckdb_cursor.execute("SELECT (SELECT sum(x) FROM tbl) + (SELECT sum(x) FROM TBL)")
        assert duckdb_cursor.fetchall() == [(11,)]


class TestBindFailureTransactionPolicy:
    def failing_execute(self, con):
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM missing_table")

    def test_all_errors_policy_invalidates(self, duckdb_cursor):
        con = duckdb_cursor
        con.execute("CREATE TABLE t(i INT)")
        con.begin()
        con.execute("SET current_transaction_invalidation_policy='ALL_ERRORS_INVALIDATE_TRANSACTION'")
        con.execute("INSERT INTO t VALUES (7)")
        self.failing_execute(con)
        con.commit()
        assert con.execute("SELECT * FROM t").fetchall() == []

    def test_default_policy_invalidates(self, duckdb_cursor):
        con = duckdb_cursor
        con.execute("CREATE TABLE t(i INT)")
        con.begin()
        con.execute("INSERT INTO t VALUES (7)")
        self.failing_execute(con)
        con.commit()
        assert con.execute("SELECT * FROM t").fetchall() == []

    def test_lenient_policy_preserves(self, duckdb_cursor):
        con = duckdb_cursor
        con.execute("CREATE TABLE t(i INT)")
        con.begin()
        con.execute("SET current_transaction_invalidation_policy='SYNTACTIC_ERRORS_DO_NOT_INVALIDATE'")
        con.execute("INSERT INTO t VALUES (7)")
        self.failing_execute(con)
        con.commit()
        assert con.execute("SELECT * FROM t").fetchall() == [(7,)]

    def test_relational_binding_is_not_execute(self, duckdb_cursor):
        """Relation construction keeps its own, inspection-grade semantics."""
        con = duckdb_cursor
        con.execute("CREATE TABLE t(i INT)")
        con.begin()
        con.execute("SET current_transaction_invalidation_policy='SYNTACTIC_ERRORS_DO_NOT_INVALIDATE'")
        con.execute("INSERT INTO t VALUES (7)")
        with pytest.raises(duckdb.CatalogException):
            con.sql("SELECT * FROM missing_table")
        con.commit()
        assert con.execute("SELECT * FROM t").fetchall() == [(7,)]


class TestResultOwnership:
    def collected(self, make):
        import gc

        refs = make()
        gc.collect()
        return [ref() is None for ref in refs]

    def test_an_unreachable_query_is_collected(self):
        import weakref

        def make():
            con = duckdb.connect()
            rel = con.sql("SELECT 1 AS i")
            refs = weakref.ref(con), weakref.ref(rel)
            con.execute("SELECT * FROM rel")
            return refs

        assert self.collected(make) == [True, True]

    def test_a_superseded_query_is_collected(self):
        import weakref

        def make():
            con = duckdb.connect()
            rel = con.sql("SELECT 1 AS i")
            refs = weakref.ref(con), weakref.ref(rel)
            con.execute("SELECT * FROM rel")
            con.execute("SELECT 42").fetchall()
            return refs

        assert self.collected(make) == [True, True]

    def test_a_failed_execute_releases_ownership(self):
        import weakref

        def make():
            con = duckdb.connect()
            rel = con.sql("SELECT CAST('bad' || i::VARCHAR AS INTEGER) AS i FROM range(3) t(i)")
            refs = weakref.ref(con), weakref.ref(rel)
            with pytest.raises(duckdb.ConversionException):
                con.execute("SELECT * FROM rel")
            return refs

        assert self.collected(make) == [True, True]

    def test_the_source_outlives_its_defining_scope(self, duckdb_cursor):
        pa = pytest.importorskip("pyarrow")
        con = duckdb_cursor

        def start():
            local = pa.table({"x": [1, 2, 3]})  # noqa: F841 - read by the replacement scan
            rel = con.sql("SELECT sum(x) AS s FROM local")  # noqa: F841 - read by the replacement scan
            con.execute("SELECT s + 1 FROM rel")

        start()
        assert con.fetchall() == [(7,)]

    def test_releasing_the_query_leaves_the_source_relation_usable(self, duckdb_cursor):
        con = duckdb_cursor
        rel = con.sql("SELECT 1 AS i")
        con.execute("SELECT * FROM rel")
        con.abort()
        assert rel.fetchall() == [(1,)]


class TestRowcountSurvivesConsumption:
    @pytest.mark.parametrize(
        "consume",
        [
            pytest.param(lambda con: con.fetchall(), id="fetchall"),
            pytest.param(lambda con: con.fetchone(), id="fetchone"),
            pytest.param(lambda con: con.df(), id="df"),
            pytest.param(lambda con: con.fetchnumpy(), id="fetchnumpy"),
        ],
    )
    def test_consuming_the_result_keeps_the_count(self, duckdb_cursor, consume):
        pytest.importorskip("pandas")
        con = duckdb_cursor
        con.execute("CREATE TABLE t(i INT)")
        con.execute("INSERT INTO t VALUES (1), (2), (3)")
        assert con.rowcount == 3
        consume(con)
        assert con.rowcount == 3

    def test_zero_affected_rows(self, duckdb_cursor):
        con = duckdb_cursor
        con.execute("CREATE TABLE t(i INT)")
        con.execute("INSERT INTO t SELECT * FROM range(0)")
        con.fetchall()
        assert con.rowcount == 0

    def test_executemany_total_survives(self, duckdb_cursor):
        con = duckdb_cursor
        con.execute("CREATE TABLE t(i INT)")
        con.executemany("INSERT INTO t VALUES (?)", [[1], [2], [3]])
        con.fetchall()
        assert con.rowcount == 3

    def test_the_next_statement_resets_the_count(self, duckdb_cursor):
        con = duckdb_cursor
        con.execute("CREATE TABLE t(i INT)")
        con.execute("INSERT INTO t VALUES (1)")
        assert con.rowcount == 1
        con.execute("SELECT 1")
        assert con.rowcount == -1


class TestIntegrations:
    def test_the_dbapi_row_loop(self, duckdb_cursor):
        """The calls polars read_database makes on a native connection."""
        cursor = duckdb_cursor.cursor()
        cursor.execute("SELECT i FROM range(10) t(i)")
        assert [column[0] for column in cursor.description] == ["i"]
        batches = []
        while rows := cursor.fetchmany(4):
            batches.append(len(rows))
        assert batches == [4, 4, 2]
        cursor.close()


class TestRelationArrow:
    def test_polars_runs_the_query_once(self, duckdb_cursor, produced):
        pytest.importorskip("polars")
        pytest.importorskip("pyarrow")
        assert len(duckdb_cursor.sql(TALLY_QUERY).pl()) == ROWS
        assert produced[0] == ROWS

    def test_an_arrow_reader_streams(self, duckdb_cursor, produced):
        pytest.importorskip("pyarrow")
        duckdb_cursor.execute("SET max_streaming_buffer_size='100KB'")
        reader = duckdb_cursor.sql(TALLY_QUERY).to_arrow_reader(1024)
        assert len(reader.read_next_batch()) > 0
        del reader
        assert produced[0] < ROWS

    def test_an_arrow_reader_keeps_its_source(self, duckdb_cursor):
        pa = pytest.importorskip("pyarrow")
        con = duckdb_cursor

        def start():
            local = pa.table({"x": list(range(100))})  # noqa: F841 - read by the replacement scan
            rel = con.sql("SELECT x FROM local")  # noqa: F841 - read by the replacement scan
            return con.sql("SELECT x FROM rel").to_arrow_reader()

        reader = start()
        assert reader.read_all().num_rows == 100

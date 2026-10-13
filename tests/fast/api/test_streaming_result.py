import gc

import pytest

import duckdb
from duckdb.sqltypes import BIGINT


class TestStreamingResult:
    def test_fetch_one(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT * FROM range(100000)")
        result = []
        while len(result) < 5000:
            tpl = duckdb_cursor.fetchone()
            result.append(tpl[0])
        assert result == list(range(5000))

    def test_fetch_many(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT * FROM range(100000)")
        result = []
        while len(result) < 5000:
            tpl = duckdb_cursor.fetchmany(10)
            result += [x[0] for x in tpl]
        assert result == list(range(5000))

    def test_record_batch_reader(self, duckdb_cursor):
        pytest.importorskip("pyarrow")
        pytest.importorskip("pyarrow.dataset")
        # record batch reader
        res = duckdb_cursor.sql("SELECT * FROM range(100000) t(i)")
        reader = res.to_arrow_reader(batch_size=16_384)
        result = []
        for batch in reader:
            result += batch.to_pydict()["i"]
        assert result == list(range(100000))

        # record batch reader with error
        res = duckdb_cursor.sql(
            "SELECT CASE WHEN i < 10000 THEN i ELSE concat('hello', i::VARCHAR)::INT END FROM range(100000) t(i)"
        )
        reader = res.to_arrow_reader(batch_size=16_384)
        with pytest.raises(OSError, match="Could not convert string 'hello10000' to INT32"):
            for _ in reader:
                pass

    def test_9801(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE test(id INTEGER , name VARCHAR NOT NULL);")

        words = ["aaaaaaaaaaaaaaaaaaaaaaa", "bbbb", "ccccccccc", "ííííííííí"]
        lines = [(i, words[i % 4]) for i in range(1000)]
        duckdb_cursor.executemany("INSERT INTO TEST (id, name) VALUES (?, ?)", lines)

        duckdb_cursor.execute(
            """
            SELECT id, name FROM test ORDER BY id ASC
        """
        )
        result = duckdb_cursor.fetchmany(size=5)
        counter = 0
        while result != []:
            for x in result:
                assert x == (counter, words[counter % 4])
                counter += 1
            result = duckdb_cursor.fetchmany(size=5)


ROW_COUNT = 1_000_000
SMALL_BUFFER = "1MB"


def drain(res):
    while res.fetchone() is not None:
        pass


class TestStreamingSemantics:
    """Which entry points stream, and what a stream promises once it is open."""

    @pytest.fixture
    def produced(self, duckdb_cursor):
        counter = [0]

        def tally(i):
            counter[0] += 1
            return i

        duckdb_cursor.execute(f"SET max_streaming_buffer_size='{SMALL_BUFFER}'")
        duckdb_cursor.create_function("tally", tally, [BIGINT], BIGINT)
        return counter

    TALLY_QUERY = f"SELECT tally(i) AS i FROM range({ROW_COUNT}) t(i)"

    def test_connection_result_streams(self, duckdb_cursor, produced):
        duckdb_cursor.execute(self.TALLY_QUERY)
        assert duckdb_cursor.fetchone() == (0,)
        duckdb_cursor.execute("SELECT 1")
        assert produced[0] < ROW_COUNT // 2

    def test_arrow_reader_streams(self, duckdb_cursor, produced):
        pytest.importorskip("pyarrow")
        reader = duckdb_cursor.sql(self.TALLY_QUERY).to_arrow_reader(1024)
        assert len(reader.read_next_batch()) == 1024
        del reader
        assert produced[0] < ROW_COUNT // 2

    def test_arrow_capsule_streams(self, duckdb_cursor, produced):
        pa = pytest.importorskip("pyarrow")
        # The capsule has no batch size parameter and its batches hold a million rows
        query = f"SELECT tally(i) AS i FROM range({2 * ROW_COUNT}) t(i)"
        capsule = duckdb_cursor.sql(query).__arrow_c_stream__()
        reader = pa.RecordBatchReader._import_from_c_capsule(capsule)
        assert len(reader.read_next_batch()) > 0
        del reader
        assert produced[0] < ROW_COUNT + ROW_COUNT // 2

    def test_a_partial_arrow_reader_holds_the_relation_it_came_from(self, duckdb_cursor):
        pytest.importorskip("pyarrow")
        duckdb_cursor.execute("SET max_streaming_buffer_size='100KB'")
        res = duckdb_cursor.sql(f"SELECT i FROM range({ROW_COUNT}) t(i)")
        reader = res.to_arrow_reader(1024)
        assert 0 < len(reader.read_next_batch()) <= 1024
        with pytest.raises(duckdb.ResourceInUseException, match="open result"):
            res.fetchall()
        reader.close()
        assert len(res.fetchall()) == ROW_COUNT

    def test_arrow_reader_over_a_materialized_relation(self, duckdb_cursor):
        pytest.importorskip("pyarrow")
        res = duckdb_cursor.sql("SELECT i FROM range(10) t(i)").materialize()
        assert res.to_arrow_reader(4).read_all().column("i").to_pylist() == list(range(10))

    def test_a_refused_statement_leaves_the_open_arrow_reader_readable(self, duckdb_cursor):
        pytest.importorskip("pyarrow")
        duckdb_cursor.execute(f"SET max_streaming_buffer_size='{SMALL_BUFFER}'")
        reader = duckdb_cursor.sql(f"SELECT i FROM range({ROW_COUNT}) t(i)").to_arrow_reader(1024)
        assert len(reader.read_next_batch()) == 1024

        with pytest.raises(duckdb.ResourceInUseException, match="open result"):
            duckdb_cursor.execute("SELECT 42")

        assert reader.read_all().num_rows == ROW_COUNT - 1024

    def test_relation_whole_fetch_runs_to_the_end(self, duckdb_cursor, produced):
        assert len(duckdb_cursor.sql(self.TALLY_QUERY).fetchall()) == ROW_COUNT
        assert produced[0] == ROW_COUNT

    def test_whole_fetch_after_partial_fetch_returns_the_remainder(self, duckdb_cursor):
        query = "SELECT i FROM range(10) t(i)"

        duckdb_cursor.execute(query)
        assert duckdb_cursor.fetchone() == (0,)
        assert duckdb_cursor.fetchall() == [(i,) for i in range(1, 10)]

        duckdb_cursor.execute(query)
        assert duckdb_cursor.fetchmany(3) == [(0,), (1,), (2,)]
        assert duckdb_cursor.df()["i"].tolist() == list(range(3, 10))

        duckdb_cursor.execute(query)
        assert duckdb_cursor.fetchone() == (0,)
        assert len(duckdb_cursor.fetch_df_chunk(0)) == 0
        assert duckdb_cursor.fetch_df_chunk()["i"].tolist() == list(range(1, 10))

        duckdb_cursor.execute(query)
        assert duckdb_cursor.fetchone() == (0,)
        assert duckdb_cursor.df()["i"].tolist() == list(range(1, 10))

    def test_error_surfaces_on_the_fetch_that_reaches_it(self, duckdb_cursor):
        duckdb_cursor.execute(f"SET max_streaming_buffer_size='{SMALL_BUFFER}'")
        duckdb_cursor.execute(
            f"""
            SELECT CASE WHEN i < {ROW_COUNT} THEN i ELSE concat('hello', i::VARCHAR)::INT END AS i
            FROM range({ROW_COUNT} + 1) t(i)
            """
        )
        assert duckdb_cursor.fetchone() == (0,)
        with pytest.raises(duckdb.ConversionException):
            drain(duckdb_cursor)

    def test_a_returning_write_applies_when_its_rows_are_read(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (i INTEGER)")

        res = duckdb_cursor.execute("INSERT INTO t VALUES (1), (2), (3) RETURNING i")
        assert res.fetchone() == (1,)
        assert res.fetchall() == [(2,), (3,)]
        assert duckdb_cursor.execute("SELECT count(*) FROM t").fetchone() == (3,)

    @pytest.mark.timeout(60)
    def test_dropping_a_running_stream_does_not_deadlock(self, duckdb_cursor):
        # Ending a query waits for its running tasks. A task inside a Python UDF cannot finish while
        # the dropping thread holds the GIL, so dropping the result must release it first.
        def identity(i):
            return i

        pytest.importorskip("pyarrow")
        duckdb_cursor.execute(f"SET max_streaming_buffer_size='{SMALL_BUFFER}'")
        duckdb_cursor.create_function("identity", identity, [BIGINT], BIGINT)
        reader = duckdb_cursor.sql(f"SELECT identity(i) AS i FROM range({2 * ROW_COUNT}) t(i)").to_arrow_reader(1024)
        assert len(reader.read_next_batch()) == 1024

        del reader
        gc.collect()

        assert duckdb_cursor.execute("SELECT 42").fetchall() == [(42,)]

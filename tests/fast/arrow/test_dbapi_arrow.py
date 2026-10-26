"""Arrow output of a DB-API result, read back through the relational API.

``con.execute(query)`` leaves the connection holding an open result. ``to_arrow_table`` and
``to_arrow_reader`` take the rows that result still holds into one collection and re-feed it through
the engine in the Arrow format on the same connection, carrying the result's original column names.
Each call warns once that it makes an extra copy of the rows and points at the relational form.

Both methods existed on a DB-API result before October 2026, which this file calls the old path, with
the tests that pinned it called the old suite; they were taken out when the engine began fixing a
result's format at submission, which this file calls the removal, and come back here over the re-feed.

Where the re-feed and ``con.sql(query)`` are meant to agree, the test compares the two on the same
query rather than a written out expectation, so the expectation cannot drift from the relation.
"""

import gc
import inspect
import re
import threading
import warnings
from contextlib import contextmanager

import pytest

import duckdb
from duckdb.sqltypes import BIGINT

pa = pytest.importorskip("pyarrow")

DEFAULT_BATCH_SIZE = 131072
VECTOR = 2048
# A row group is 122880 rows, so a parallel scan of fewer rows runs on one thread and proves nothing.
ROW_GROUPS = 400_000
# Enough rows that execute() settles only the first chunk and the take has work left to do.
TALLY_ROWS = 100_000
TALLY_QUERY = f"SELECT tally(i) AS i FROM range({TALLY_ROWS}) t(i)"

TABLE = "to_arrow_table"
READER = "to_arrow_reader"
# The same two names on the connection and as module functions over the default connection.
FORMS = [TABLE, READER]
# Deprecated before the removal, so they stay gone; duckdb.arrow builds a relation from an Arrow object.
GONE = ["fetch_arrow_table", "fetch_record_batch", "pl"]

# A plain warning, not a deprecation: one text per method, naming no version.
TABLE_TEXT = (
    "Arrow output on a DB-API result makes an extra copy of the rows, use con.sql(query).to_arrow_table() instead."
)
READER_TEXT = (
    "Arrow output on a DB-API result makes an extra copy of the rows, use con.sql(query).to_arrow_reader() instead."
)

RICH_SQL = """
    SELECT
        i AS int_col,
        i::DOUBLE AS double_col,
        'row_' || i::VARCHAR AS str_col,
        i % 2 = 0 AS bool_col,
        CASE WHEN i % 3 = 0 THEN NULL ELSE i END AS nullable_col,
        {'x': i, 'y': i::VARCHAR} AS struct_col,
        [i, i + 1, i + 2] AS list_col,
        MAP {i::VARCHAR: i * 10} AS map_col,
    FROM range(1000) t(i)
"""
GEOM_SQL = "SELECT id, ('POINT(' || id || ' ' || id || ')')::GEOMETRY AS g FROM range(8) t(id)"
DUP_SQL = "SELECT i AS a, i + 100 AS a, i + 200 AS a FROM range(50) t(i)"
SUFFIX_SQL = "SELECT i AS a, i + 100 AS a, i + 200 AS a_1 FROM range(50) t(i)"
CASE_SQL = "SELECT i AS A, i + 100 AS a FROM range(50) t(i)"

NO_RESULT = "No open result set"
OPEN_RESULT = "open result"
CLOSED = "closed"
# The two methods keep the two refusals they had before the removal.
TABLE_ZERO_BATCH = "The Arrow batch size must be larger than 0"
READER_ZERO_BATCH = "Approximate Batch Size of Record Batch MUST be higher than 0"


def expected_text(form):
    return READER_TEXT if form == READER else TABLE_TEXT


def zero_batch_refusal(form):
    if form == READER:
        return RuntimeError, READER_ZERO_BATCH
    return duckdb.InvalidInputException, TABLE_ZERO_BATCH


def warns(form):
    return pytest.warns(UserWarning, match=re.escape(expected_text(form)))


def call(owner, form, *args, **kwargs):
    """The method on a connection or on the module, with its one warning."""
    method = getattr(owner, form)
    with warns(form):
        return method(*args, **kwargs)


def as_table(owner, form, *args, **kwargs):
    out = call(owner, form, *args, **kwargs)
    return out.read_all() if form == READER else out


@contextmanager
def unwarned():
    """For a call whose refusal, not whose warning, is the subject."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def rows_of(table):
    return list(zip(*(column.to_pylist() for column in table.columns), strict=True))


@pytest.fixture
def produced(duckdb_cursor):
    """Counts the rows the tally query produced, which shows whether and how far the statement ran."""
    count = [0]

    def tally(i):
        count[0] += 1
        return i

    duckdb_cursor.create_function("tally", tally, [BIGINT], BIGINT, side_effects=True)
    return count


@pytest.fixture
def big(duckdb_cursor):
    duckdb_cursor.execute(f"CREATE TABLE big AS SELECT i FROM range({ROW_GROUPS}) t(i)")
    return duckdb_cursor


# These libraries reach Arrow on a DB-API result only through names deprecated before the removal,
# which stay gone, so the call they make raises AttributeError and leaves the result to be fetched.
ECOSYSTEM_CALLS_THAT_STAY_GONE = [
    pytest.param(lambda cursor: cursor.fetch_arrow_table(), id="polars-2.0.0-read_database"),
    pytest.param(lambda cursor: cursor.fetch_record_batch(1000), id="polars-2.0.0-iter_batches"),
    pytest.param(lambda cursor: cursor.fetch_record_batch(rows_per_batch=1000), id="ibis-12.0.0-to_pyarrow_batches"),
    pytest.param(lambda cursor: cursor.fetch_arrow_table(), id="sqlframe-4.5.0-fetch_arrow_table"),
    pytest.param(lambda cursor: cursor.fetch_record_batch(1000), id="sqlframe-4.5.0-fetch_record_batch"),
]


class TestSurface:
    """The two names, their parameters, the names that stay gone, and the call shapes the ecosystem makes."""

    @pytest.mark.parametrize("form", FORMS)
    def test_the_connection_form_takes_one_batch_size(self, form, duckdb_cursor):
        signature = inspect.signature(getattr(duckdb_cursor, form))
        assert [(p.name, p.default, p.kind) for p in signature.parameters.values()] == [
            ("batch_size", DEFAULT_BATCH_SIZE, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

    @pytest.mark.parametrize("form", FORMS)
    def test_the_module_form_adds_a_keyword_only_connection(self, form):
        parameters = list(inspect.signature(getattr(duckdb, form)).parameters.values())
        assert [(p.name, p.default, p.kind) for p in parameters] == [
            ("batch_size", DEFAULT_BATCH_SIZE, inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ("connection", None, inspect.Parameter.KEYWORD_ONLY),
        ]

    @pytest.mark.parametrize("form", FORMS)
    def test_a_module_form_reads_the_default_connection(self, form):
        duckdb.execute("SELECT 42 AS a")
        assert as_table(duckdb, form).column("a").to_pylist() == [42]

    def test_a_module_form_takes_the_batch_size_positionally(self):
        duckdb.execute("SELECT i AS a FROM range(3000) t(i)")
        table = as_table(duckdb, READER, 1024)
        assert [len(batch) for batch in table.to_batches()] == [1024, 1024, 952]
        assert table.column("a").to_pylist() == list(range(3000))

    def test_a_module_form_takes_the_connection_by_keyword(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT 42 AS a")
        assert as_table(duckdb, TABLE, connection=duckdb_cursor).column("a").to_pylist() == [42]

    @pytest.mark.parametrize("name", GONE)
    def test_a_name_deprecated_before_the_removal_stays_gone(self, name, duckdb_cursor):
        assert not hasattr(duckdb_cursor, name)
        assert not hasattr(duckdb, name)

    def test_arrow_stays_a_relation_builder_and_not_a_reader(self, duckdb_cursor):
        assert not hasattr(duckdb_cursor, "arrow")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            relation = duckdb.arrow(pa.table({"a": [1]}))
        assert isinstance(relation, duckdb.DuckDBPyRelation)

    @pytest.mark.parametrize("ecosystem_call", ECOSYSTEM_CALLS_THAT_STAY_GONE)
    def test_an_ecosystem_call_through_a_deprecated_name_still_fails(self, ecosystem_call, duckdb_cursor):
        cursor = duckdb_cursor.cursor()
        cursor.execute("SELECT i FROM range(10) t(i)")
        with pytest.raises(AttributeError):
            ecosystem_call(cursor)
        assert cursor.fetchall() == [(i,) for i in range(10)]

    def test_the_dlt_shape(self, duckdb_cursor):
        """Dlt 1.31.0 calls to_arrow_reader with the chunk size positionally, for every duckdb from 1.5.0 up."""
        duckdb_cursor.execute("SELECT i FROM range(5000) t(i)")
        reader = call(duckdb_cursor, READER, 2048)
        assert sum(len(batch) for batch in reader) == 5000

    def test_a_cursor_reads_its_own_result(self, duckdb_cursor):
        """A cursor is its own connection, so its result and its parent's are read independently."""
        cursor = duckdb_cursor.cursor()
        cursor.execute("SELECT 42 AS a")
        duckdb_cursor.execute("SELECT 7 AS a")
        assert as_table(cursor, TABLE).column("a").to_pylist() == [42]
        assert as_table(duckdb_cursor, TABLE).column("a").to_pylist() == [7]


class TestRows:
    """Exactly the rows the result still held, in their original order."""

    @pytest.mark.parametrize("form", FORMS)
    def test_the_rows_match_the_relation_path(self, form, duckdb_cursor):
        expected = duckdb_cursor.sql(RICH_SQL).to_arrow_table()
        duckdb_cursor.execute(RICH_SQL)
        assert as_table(duckdb_cursor, form).equals(expected)

    @pytest.mark.parametrize("form", FORMS)
    def test_a_whole_arrow_fetch_after_a_row_fetch_returns_the_remainder(self, form, duckdb_cursor):
        # The old path refused this with "already fetched"; fetchall and df return the remainder.
        # The whole result fits one chunk, so every remaining row is the buffered suffix of the chunk
        # the fetch already pulled, and none of it may be lost.
        duckdb_cursor.execute("SELECT i FROM range(10) t(i)")
        assert duckdb_cursor.fetchone() == (0,)
        assert as_table(duckdb_cursor, form).column("i").to_pylist() == list(range(1, 10))

    @pytest.mark.parametrize("form", FORMS)
    def test_the_remainder_crosses_a_chunk_boundary(self, form, duckdb_cursor):
        # The buffered suffix of the first chunk comes before the chunks the engine still holds.
        duckdb_cursor.execute(f"SELECT i FROM range({VECTOR + 1}) t(i)")
        assert duckdb_cursor.fetchone() == (0,)
        assert as_table(duckdb_cursor, form).column("i").to_pylist() == list(range(1, VECTOR + 1))

    @pytest.mark.parametrize("form", FORMS)
    def test_the_remainder_of_a_result_completed_at_execute(self, form, duckdb_cursor):
        # A RETURNING write completes at execute(), so the fetch reads from rows the engine already
        # holds rather than from a stream, and the take copies what is left after the fetch cursor.
        duckdb_cursor.execute("CREATE TABLE r (i INTEGER)")
        duckdb_cursor.execute("INSERT INTO r SELECT range FROM range(5000) RETURNING i")
        assert duckdb_cursor.fetchone() == (0,)
        assert as_table(duckdb_cursor, form).column("i").to_pylist() == list(range(1, 5000))

    @pytest.mark.parametrize("form", FORMS)
    def test_a_result_crossing_a_vector_boundary(self, form, duckdb_cursor):
        duckdb_cursor.execute(f"SELECT i FROM range({VECTOR + 1}) t(i)")
        assert as_table(duckdb_cursor, form).column("i").to_pylist() == list(range(VECTOR + 1))

    @pytest.mark.parametrize("form", FORMS)
    def test_a_source_of_several_row_groups(self, form, big):
        big.execute("SELECT i FROM big")
        assert as_table(big, form).column("i").to_pylist() == list(range(ROW_GROUPS))

    @pytest.mark.parametrize("form", FORMS)
    def test_the_statement_runs_once_in_total(self, form, duckdb_cursor, produced):
        duckdb_cursor.execute(TALLY_QUERY)
        assert as_table(duckdb_cursor, form).num_rows == TALLY_ROWS
        assert produced[0] == TALLY_ROWS

    def test_a_sequence_advances_once_per_row(self, duckdb_cursor):
        """The rows come from the result, so the statement's side effects do not happen twice."""
        duckdb_cursor.execute("CREATE SEQUENCE s")
        duckdb_cursor.execute("SELECT nextval('s') AS v FROM range(3)")
        assert as_table(duckdb_cursor, TABLE).column("v").to_pylist() == [1, 2, 3]
        assert duckdb_cursor.execute("SELECT nextval('s')").fetchone() == (4,)


class TestConsumption:
    """The call consumes the rows and leaves the result in the slot."""

    @pytest.mark.parametrize("form", FORMS)
    def test_a_second_call_answers_none(self, form, duckdb_cursor):
        # The old path answered the table once and then None, as df() still does.
        duckdb_cursor.execute("SELECT 42 AS a")
        assert as_table(duckdb_cursor, form).num_rows == 1
        assert call(duckdb_cursor, form) is None

    @pytest.mark.parametrize(
        ("terminal", "consumed"),
        [
            pytest.param(lambda con: con.fetchall(), [], id="fetchall"),
            pytest.param(lambda con: con.fetchmany(5), [], id="fetchmany"),
            pytest.param(lambda con: con.fetchone(), None, id="fetchone"),
            pytest.param(lambda con: con.df(), None, id="df"),
            pytest.param(lambda con: con.fetchdf(), None, id="fetchdf"),
            pytest.param(lambda con: con.fetch_df_chunk(), None, id="fetch_df_chunk"),
            pytest.param(lambda con: con.fetchnumpy(), None, id="fetchnumpy"),
        ],
    )
    @pytest.mark.parametrize("form", FORMS)
    def test_a_row_terminal_answers_as_it_does_for_a_drained_result(self, terminal, consumed, form, duckdb_cursor):
        # The consumed answer is two answers: the row lists answer [] and the rest answer None.
        pytest.importorskip("pandas")
        query = "SELECT i FROM range(10) t(i)"
        drained = duckdb.connect()
        drained.execute(query).fetchall()
        assert terminal(drained) == consumed
        duckdb_cursor.execute(query)
        as_table(duckdb_cursor, form)
        assert terminal(duckdb_cursor) == consumed

    @pytest.mark.parametrize("form", FORMS)
    def test_rowcount_and_description_keep_their_values(self, form, duckdb_cursor):
        duckdb_cursor.execute("SELECT 42 AS a")
        description = duckdb_cursor.description
        rowcount = duckdb_cursor.rowcount
        as_table(duckdb_cursor, form)
        assert (duckdb_cursor.description, duckdb_cursor.rowcount) == (description, rowcount)

    @pytest.mark.parametrize("form", FORMS)
    def test_the_rowcount_of_a_write_survives_the_call(self, form, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t(i INT)")
        duckdb_cursor.execute("INSERT INTO t VALUES (1), (2), (3)")
        assert duckdb_cursor.rowcount == 3
        as_table(duckdb_cursor, form)
        assert duckdb_cursor.rowcount == 3


class TestTheConnectionIsFree:
    """A table leaves the connection free; a reader holds it until it is read out."""

    @pytest.mark.parametrize(
        ("follow", "rows"),
        [
            pytest.param(lambda con: con.to_arrow_table().num_rows, 0, id=TABLE),
            pytest.param(lambda con: con.to_arrow_reader().read_all().num_rows, 0, id=READER),
            pytest.param(lambda con: len(con.sql("SELECT 1").fetchall()), 1, id="relational-statement"),
        ],
    )
    def test_an_empty_result_leaves_the_connection_free(self, follow, rows, duckdb_cursor):
        # An empty result could finish before the client's first poll at execute() without the query
        # ending in the engine, which then refused the next statement as if the result were still
        # open, intermittently and only with more than one thread; a result the client reads as
        # complete has also ended in the engine, so two hundred rounds never see a refusal.
        duckdb_cursor.execute("SET threads = 2")
        with unwarned():
            for _ in range(200):
                duckdb_cursor.execute("SELECT 1 AS i WHERE false")
                assert follow(duckdb_cursor) == rows

    def test_the_next_statement_after_a_table_succeeds(self, duckdb_cursor):
        duckdb_cursor.execute(f"SELECT i FROM range({TALLY_ROWS}) t(i)")
        assert as_table(duckdb_cursor, TABLE).num_rows == TALLY_ROWS
        assert duckdb_cursor.execute("SELECT 42").fetchall() == [(42,)]

    def test_an_open_reader_holds_the_connection(self, big):
        big.execute("SET max_streaming_buffer_size='100KB'")
        big.execute("SELECT i FROM big")
        reader = call(big, READER, 1024)
        assert 0 < len(reader.read_next_batch()) <= 1024
        # The rows were taken out of the DB-API result, so the refusal names the open Arrow stream
        # rather than the unfinished result of execute().
        with pytest.raises(duckdb.ResourceInUseException, match=OPEN_RESULT):
            big.execute("SELECT 42")
        assert reader.read_all().num_rows == ROW_GROUPS - 1024
        assert big.execute("SELECT 42").fetchall() == [(42,)]

    def test_closing_a_reader_frees_the_connection(self, big):
        big.execute("SET max_streaming_buffer_size='100KB'")
        big.execute("SELECT i FROM big")
        reader = call(big, READER, 1024)
        assert 0 < len(reader.read_next_batch()) <= 1024
        reader.close()
        assert big.execute("SELECT 42").fetchall() == [(42,)]

    def test_dropping_a_reader_frees_the_connection(self, big):
        big.execute("SET max_streaming_buffer_size='100KB'")
        big.execute("SELECT i FROM big")
        reader = call(big, READER, 1024)
        assert 0 < len(reader.read_next_batch()) <= 1024
        del reader
        gc.collect()
        assert big.execute("SELECT 42").fetchall() == [(42,)]

    def test_an_open_reader_still_lets_a_plan_ask_for_its_columns(self, big):
        big.execute("SET max_streaming_buffer_size='100KB'")
        big.execute("SELECT i FROM big")
        reader = call(big, READER, 1024)
        assert 0 < len(reader.read_next_batch()) <= 1024
        assert big.sql("SELECT 1 AS x").columns == ["x"]
        assert reader.read_all().num_rows == ROW_GROUPS - 1024

    def test_a_reader_survives_del_conn(self):
        con = duckdb.connect()
        con.execute("SELECT i, i::VARCHAR AS s FROM range(2000) t(i)")
        reader = call(con, READER, 512)
        del con
        gc.collect()
        table = reader.read_all()
        assert table.num_rows == 2000
        assert table.column("i").to_pylist() == list(range(2000))

    def test_a_reader_keeps_reading_after_the_connection_is_closed(self):
        # A reader owns the stream it is reading, so close() beside it is accepted and the rest
        # reads out, because those rows are in client memory. A call started after close() is
        # refused instead, since it needs a live connection to run a statement on.
        con = duckdb.connect()
        con.execute(f"SELECT i FROM range({ROW_GROUPS}) t(i)")
        reader = call(con, READER, 1024)
        first = len(reader.read_next_batch())
        con.close()
        assert reader.read_all().num_rows == ROW_GROUPS - first

    def test_a_reader_keeps_a_replacement_scanned_source(self, duckdb_cursor):
        con = duckdb_cursor

        def start():
            local = pa.table({"x": list(range(100))})  # noqa: F841 - read by the replacement scan
            con.execute("SELECT x FROM local")
            return call(con, READER)

        reader = start()
        gc.collect()
        assert reader.read_all().num_rows == 100

    def test_the_whole_query_runs_before_the_first_batch(self, duckdb_cursor, produced):
        # Taking the rows materializes the result, so the reader cannot stream the query and a
        # dropped reader cannot stop it early. The old path promised the opposite.
        duckdb_cursor.execute("SET max_streaming_buffer_size='100KB'")
        duckdb_cursor.execute(TALLY_QUERY)
        reader = call(duckdb_cursor, READER, 1024)
        assert produced[0] == TALLY_ROWS
        assert 0 < len(reader.read_next_batch()) <= 1024
        del reader
        gc.collect()
        assert produced[0] == TALLY_ROWS


def empty_keeps_its_columns(table):
    assert table.num_rows == 0
    assert table.schema.names == ["a", "b"]


def dictionary_encoded(table):
    assert pa.types.is_dictionary(table.schema.field("m").type)


def zoned_new_york(table):
    assert table.schema.field("ts").type.tz == "America/New_York"


def large_string(table):
    assert table.schema.field("s").type == pa.large_string()


def null_typed(table):
    assert table.schema.field("v").type == pa.null()
    assert table.column("v").to_pylist() == [None]


ENCODING_CASES = [
    pytest.param((), RICH_SQL, None, id="rich-types"),
    pytest.param(
        (),
        "SELECT i AS a, i::VARCHAR AS b FROM range(10) t(i) WHERE i < 0",
        empty_keeps_its_columns,
        id="empty-keeps-its-schema",
    ),
    pytest.param(
        ("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')",),
        "SELECT 'ok'::mood AS m FROM range(100)",
        dictionary_encoded,
        id="enum-is-a-dictionary",
    ),
    pytest.param(
        ("SET TimeZone = 'America/New_York'",),
        "SELECT TIMESTAMPTZ '2021-06-01 12:00:00' AS ts",
        zoned_new_york,
        id="session-time-zone",
    ),
    pytest.param(
        ("SET arrow_large_buffer_size = true",),
        "SELECT 'hello' AS s FROM range(10)",
        large_string,
        id="arrow-large-buffer-size",
    ),
    pytest.param((), "SELECT NULL AS v", null_typed, id="null-typed-column"),
    pytest.param((), "SELECT 1::HUGEINT AS h, 2::UHUGEINT AS u, MAP {'k': 1} AS m", None, id="wide-and-map"),
]


class TestEncoding:
    """The Arrow encoding is the connection's own, read at the Arrow call.

    Both paths run the same conversion on the same connection, so the comparison with
    ``con.sql(query).to_arrow_table()`` is full equality, column names included.
    """

    @pytest.mark.parametrize(("setup", "query", "check"), ENCODING_CASES)
    @pytest.mark.parametrize("form", FORMS)
    def test_it_matches_the_relation_path(self, setup, query, check, form, duckdb_cursor):
        for line in setup:
            duckdb_cursor.execute(line)
        expected = duckdb_cursor.sql(query).to_arrow_table()
        duckdb_cursor.execute(query)
        actual = as_table(duckdb_cursor, form)
        assert actual.equals(expected)
        if check is not None:
            check(actual)

    def test_a_bound_parameter_flows_through(self, duckdb_cursor):
        expected = duckdb_cursor.sql("SELECT ? AS p", params=[42]).to_arrow_table()
        duckdb_cursor.execute("SELECT ? AS p", [42])
        assert as_table(duckdb_cursor, TABLE).equals(expected)

    def test_lossless_conversion_switches_the_export(self, duckdb_cursor):
        pytest.importorskip("pyarrow", "18.0.0")
        query = "SELECT '10203040-5060-7080-8090-a0b0c0d0e0f0'::UUID AS u"
        duckdb_cursor.execute("SET arrow_lossless_conversion = true")
        expected = duckdb_cursor.sql(query).to_arrow_table()
        duckdb_cursor.execute(query)
        actual = as_table(duckdb_cursor, TABLE)
        assert actual.equals(expected)
        assert actual.schema.field("u").type.extension_name == "arrow.uuid"

    @pytest.mark.parametrize("form", FORMS)
    def test_the_time_zone_is_read_at_the_arrow_call(self, form, duckdb_cursor):
        # The conversion runs at the Arrow call, so a setting changed in between is the one that
        # applies. A RETURNING write completes inside execute(), which leaves the connection free for
        # con.sql to change the setting beside the open result; another execute() would replace it.
        duckdb_cursor.execute("SET TimeZone = 'America/New_York'")
        duckdb_cursor.execute("CREATE TABLE t (ts TIMESTAMPTZ)")
        duckdb_cursor.execute("INSERT INTO t VALUES (TIMESTAMPTZ '2021-06-01 12:00:00') RETURNING ts")
        duckdb_cursor.sql("SET TimeZone = 'Asia/Kathmandu'")
        assert as_table(duckdb_cursor, form).schema.field("ts").type.tz == "Asia/Kathmandu"

    @staticmethod
    def assert_geoarrow(table) -> None:
        field = table.schema.field("g")
        assert field.metadata is not None
        assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"

    def test_geometry_reads_as_geoarrow_wkb(self, duckdb_cursor):
        expected = duckdb_cursor.sql(GEOM_SQL).to_arrow_table()
        duckdb_cursor.execute(GEOM_SQL)
        actual = as_table(duckdb_cursor, TABLE)
        self.assert_geoarrow(actual)
        assert actual.column("g").to_pylist() == expected.column("g").to_pylist()

    def test_geometry_reads_as_geoarrow_wkb_after_the_connection_is_gone(self):
        expected = duckdb.connect().sql(GEOM_SQL).to_arrow_table()
        con = duckdb.connect()
        con.execute(GEOM_SQL)
        reader = call(con, READER)
        del con
        gc.collect()
        actual = reader.read_all()
        self.assert_geoarrow(actual)
        assert actual.column("g").to_pylist() == expected.column("g").to_pylist()


def read_sizes_and_rows(reader):
    sizes, rows = [], []
    for batch in reader:
        sizes.append(len(batch))
        rows.extend(batch.column("i").to_pylist())
    return sizes, rows


class TestBatchSize:
    """The batch size is a hard maximum, and a bad one costs no rows.

    Measured: held rows honour the maximum exactly as a base table scan does, at 1, 4 and 12
    threads, so the sizes the old suite pinned come back. A maximum above the row count
    gives fewer and larger batches, one per producer, which is "at most" behaving correctly.
    """

    def test_the_default_reads_the_whole_result_in_order(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT i FROM range(3000) t(i)")
        reader = call(duckdb_cursor, READER)
        sizes, rows = read_sizes_and_rows(reader)
        assert all(0 < size <= DEFAULT_BATCH_SIZE for size in sizes)
        assert rows == list(range(3000))
        with pytest.raises(StopIteration):
            reader.read_next_batch()

    @pytest.mark.parametrize("batch_size", [1, 33, 1024, 2048, DEFAULT_BATCH_SIZE, 999999])
    def test_every_batch_size_answers_the_whole_result_in_order(self, batch_size, duckdb_cursor):
        duckdb_cursor.execute("SELECT i FROM range(3000) t(i)")
        sizes, rows = read_sizes_and_rows(call(duckdb_cursor, READER, batch_size))
        assert all(size > 0 for size in sizes)
        assert rows == list(range(3000))

    def test_a_large_batch_size_is_a_maximum(self, big):
        big.execute("SELECT i FROM big")
        sizes, rows = read_sizes_and_rows(call(big, READER, 50000))
        assert all(0 < size <= 50000 for size in sizes)
        assert rows == list(range(ROW_GROUPS))

    def test_a_batch_size_caps_the_tables_batches(self, big):
        big.execute("SELECT i FROM big")
        table = as_table(big, TABLE, 50000)
        assert all(0 < len(batch) <= 50000 for batch in table.to_batches())
        assert table.column("i").to_pylist() == list(range(ROW_GROUPS))

    def test_a_small_batch_size_is_a_maximum_too(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT i FROM range(3000) t(i)")
        sizes, rows = read_sizes_and_rows(call(duckdb_cursor, READER, 100))
        assert all(0 < size <= 100 for size in sizes)
        assert rows == list(range(3000))

    @pytest.mark.parametrize("form", FORMS)
    def test_the_exact_batch_sizes_the_old_suite_pinned(self, form, duckdb_cursor):
        # 3000 rows at a maximum of 1024 give 1024, 1024, 952 off held rows as off a base table, at
        # every thread count, which is what the old suite pinned.
        duckdb_cursor.execute("SELECT i FROM range(3000) t(i)")
        table = as_table(duckdb_cursor, form, 1024)
        assert [len(batch) for batch in table.to_batches()] == [1024, 1024, 952]
        assert table.column("i").to_pylist() == list(range(3000))

    @pytest.mark.parametrize("form", FORMS)
    def test_a_zero_batch_size_is_refused_before_the_rows_are_taken(self, form, duckdb_cursor):
        # Warn, then check the batch size, then take. A typo must not destroy the result. The two
        # methods keep the two different refusals they had before the removal, so the check cannot
        # be one shared message.
        refusal, text = zero_batch_refusal(form)
        duckdb_cursor.execute("SELECT i FROM range(10) t(i)")
        method = getattr(duckdb_cursor, form)
        with warns(form), pytest.raises(refusal, match=text):
            method(0)
        assert duckdb_cursor.fetchall() == [(i,) for i in range(10)]

    @pytest.mark.parametrize("form", FORMS)
    def test_a_negative_batch_size_is_refused_before_the_rows_are_taken(self, form, duckdb_cursor):
        # As on the old path, an unsigned parameter refuses a negative size with a TypeError.
        duckdb_cursor.execute("SELECT i FROM range(10) t(i)")
        method = getattr(duckdb_cursor, form)
        with unwarned(), pytest.raises(TypeError, match="incompatible function arguments"):
            method(-1)
        assert duckdb_cursor.fetchall() == [(i,) for i in range(10)]


class TestNames:
    """The Arrow carries the column names of the statement, duplicates and all.

    The re-feed deduplicates the names only for the binder, which rejects duplicates, and reports
    the originals. The two odd shapes below are here because the deduplicated names exist inside the
    call: a name that already carries the suffix the deduplication hands out, and two names that
    differ only in case, are the shapes where an internal rename would leak into the output.
    """

    @pytest.mark.parametrize("form", FORMS)
    def test_duplicate_names_are_the_statements_own(self, form, duckdb_cursor):
        duckdb_cursor.execute(DUP_SQL)
        table = as_table(duckdb_cursor, form)
        assert table.column_names == ["a", "a", "a"]
        assert rows_of(table)[:2] == [(0, 100, 200), (1, 101, 201)]

    @pytest.mark.parametrize("form", FORMS)
    def test_a_name_that_already_carries_the_suffix(self, form, duckdb_cursor):
        # A leak of the internal renaming would read as ["a", "a_1", "a_1_1"] here, renaming the
        # user's own a_1 column. The distinct values show which column sits under which name.
        duckdb_cursor.execute(SUFFIX_SQL)
        table = as_table(duckdb_cursor, form)
        assert table.column_names == ["a", "a", "a_1"]
        assert rows_of(table)[:2] == [(0, 100, 200), (1, 101, 201)]

    @pytest.mark.parametrize("form", FORMS)
    def test_names_differing_only_in_case(self, form, duckdb_cursor):
        # The binder's identifier map folds case, so a leak would read as ["A", "a_1"].
        duckdb_cursor.execute(CASE_SQL)
        table = as_table(duckdb_cursor, form)
        assert table.column_names == ["A", "a"]
        assert rows_of(table)[:2] == [(0, 100), (1, 101)]

    def test_a_readers_schema_carries_the_same_names(self, duckdb_cursor):
        # The reader advertises its schema before any batch is read, so the names are pinned there
        # too: a reader whose schema and whose batches disagree is worse than either answer.
        duckdb_cursor.execute(DUP_SQL)
        reader = call(duckdb_cursor, READER)
        assert reader.schema.names == ["a", "a", "a"]

    def test_a_join_of_two_frames_with_a_shared_name(self, duckdb_cursor):
        # The old path passed this and the relation path fails it today, so it is the case
        # that says the names come from the result rather than from a bind.
        pd = pytest.importorskip("pandas")
        df_a = pd.DataFrame({"join_key": [1, 2], "col_a": [10, 20]})  # noqa: F841
        df_b = pd.DataFrame({"join_key": [1, 2], "col_a": [30, 40]})  # noqa: F841
        duckdb_cursor.execute(
            "SELECT * FROM df_a one JOIN df_b two ON one.join_key = two.join_key ORDER BY one.join_key"
        )
        table = as_table(duckdb_cursor, TABLE)
        assert table.schema.names == ["join_key", "col_a", "join_key", "col_a"]
        assert rows_of(table) == [(1, 10, 1, 30), (2, 20, 2, 40)]


class TestWarning:
    """One plain UserWarning per call, at the caller's line, pointing at the relational form.

    The class is checked exactly, not with isinstance: a DeprecationWarning, or any other subclass,
    would say something about the future of these calls that nothing has decided.
    """

    @pytest.mark.parametrize("form", FORMS)
    def test_a_call_warns_once_at_the_callers_line(self, form, duckdb_cursor):
        duckdb_cursor.execute("SELECT 42 AS a")
        method = getattr(duckdb_cursor, form)
        line = inspect.currentframe().f_lineno + 2
        with warns(form) as record:
            method()
        assert [type(warning.message) for warning in record] == [UserWarning]
        assert (record[0].filename, record[0].lineno) == (__file__, line)

    @pytest.mark.parametrize("form", FORMS)
    def test_the_text_names_the_relational_form_and_no_version_and_is_not_a_deprecation(self, form, duckdb_cursor):
        duckdb_cursor.execute("SELECT 42 AS a")
        method = getattr(duckdb_cursor, form)
        with pytest.warns(UserWarning, match="extra copy") as record:
            method()
        assert [(type(warning.message), str(warning.message)) for warning in record] == [
            (UserWarning, expected_text(form))
        ]

    @pytest.mark.parametrize("form", FORMS)
    def test_a_module_form_warns_too(self, form):
        method = getattr(duckdb, form)
        duckdb.execute("SELECT 42 AS a")
        with pytest.warns(UserWarning, match="extra copy") as record:
            method()
        assert [(type(warning.message), str(warning.message)) for warning in record] == [
            (UserWarning, expected_text(form))
        ]

    def test_every_call_warns_again(self, duckdb_cursor):
        """A consumed result still warns, so a caller in a loop is not warned once and then silently served."""
        duckdb_cursor.execute("SELECT 42 AS a")
        method = duckdb_cursor.to_arrow_table
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            method()
            method()
        assert [(type(warning.message), str(warning.message)) for warning in record] == [
            (UserWarning, TABLE_TEXT),
            (UserWarning, TABLE_TEXT),
        ]


STATEMENT_KINDS = [
    pytest.param(("CREATE TABLE t(i INT)",), "CREATE TABLE u(i INT)", id="ddl"),
    pytest.param(("CREATE TABLE t(i INT)",), "INSERT INTO t VALUES (1), (2)", id="dml"),
    pytest.param(("CREATE TABLE t(i INT)",), "DROP TABLE t", id="drop"),
    pytest.param((), "SET threads = 2", id="set"),
    pytest.param(("CREATE TABLE t(i INT)",), "CALL pragma_table_info('t')", id="call"),
    pytest.param(("CREATE TABLE t(i INT)",), "PRAGMA show_tables", id="pragma-with-rows"),
    pytest.param((), "EXPLAIN SELECT 1", id="explain"),
    pytest.param(("CREATE TABLE t(i INT)",), "CREATE TABLE u AS SELECT i FROM range(3) t(i)", id="create-as"),
    pytest.param((), "PIVOT (VALUES ('a', 1), ('b', 2)) t(k, v) ON k USING sum(v)", id="expanding"),
    pytest.param((), "SELECT 1 AS a; CREATE TABLE z(i INT)", id="script-ending-without-rows"),
]


class TestStatementKinds:
    """Every statement kind answers whatever rows its result holds."""

    @pytest.mark.parametrize(("setup", "statement"), STATEMENT_KINDS)
    @pytest.mark.parametrize("form", FORMS)
    def test_a_statement_kind_answers_what_its_rows_answer(self, setup, statement, form):
        # No special case per kind: the expectation is what the same statement's result gives
        # through the DB-API's own whole fetches, under the names description reports. Measured over
        # 21 kinds, including Count BIGINT with no rows for a DDL statement and Success BOOLEAN with
        # none for SET, with the side effects applied once.
        pytest.importorskip("pandas")
        reference, actual = duckdb.connect(), duckdb.connect()
        for con in (reference, actual):
            for line in setup:
                con.execute(line)
        expected = reference.execute(statement).fetchall()
        names = [column[0] for column in reference.description]
        actual.execute(statement)
        table = as_table(actual, form)
        assert table.schema.names == names
        assert rows_of(table) == expected
        # The DB-API's DataFrame answer for the same statement agrees on the columns and the count.
        frame_con = duckdb.connect()
        for line in setup:
            frame_con.execute(line)
        frame = frame_con.execute(statement).df()
        assert (list(frame.columns), len(frame)) == (names, table.num_rows)

    @pytest.mark.parametrize("form", FORMS)
    def test_a_returning_write_answers_its_rows_and_stays_applied(self, form, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE r (i INTEGER)")
        duckdb_cursor.execute("INSERT INTO r VALUES (1), (2) RETURNING i")
        assert as_table(duckdb_cursor, form).column("i").to_pylist() == [1, 2]
        assert duckdb_cursor.execute("SELECT count(*) FROM r").fetchone() == (2,)

    def test_executemany_answers_the_last_set(self, duckdb_cursor):
        # The DB-API keeps the last parameter set's result, so Arrow answers what fetchall does.
        reference = duckdb.connect()
        reference.executemany("SELECT ? AS v", [[1], [2]])
        expected = reference.fetchall()
        duckdb_cursor.executemany("SELECT ? AS v", [[1], [2]])
        assert rows_of(as_table(duckdb_cursor, TABLE)) == expected

    @pytest.mark.parametrize("form", FORMS)
    def test_executemany_of_a_write_keeps_the_accumulated_rowcount(self, form, duckdb_cursor):
        # The Arrow answers the last set's rows, as fetchall does, while rowcount keeps the total
        # over every set.
        reference = duckdb.connect()
        for con in (reference, duckdb_cursor):
            con.execute("CREATE TABLE t (i INTEGER)")
            con.executemany("INSERT INTO t VALUES (?)", [[1], [2], [3]])
        expected = reference.fetchall()
        assert duckdb_cursor.rowcount == 3
        assert rows_of(as_table(duckdb_cursor, form)) == expected
        assert duckdb_cursor.rowcount == 3


class TestOrder:
    """The order the rows come out in, over a source of several row groups.

    The re-feed scans the held rows in parallel, so the source has to span several row groups or the
    scan runs on one thread and the test proves nothing.
    """

    @pytest.mark.parametrize("form", FORMS)
    def test_the_rows_keep_their_order_under_the_default_setting(self, form, big):
        big.execute("SET threads = 4")
        big.execute("SELECT i FROM big ORDER BY i")
        assert as_table(big, form).column("i").to_pylist() == list(range(ROW_GROUPS))

    @pytest.mark.parametrize("form", FORMS)
    def test_only_the_values_survive_a_relaxed_insertion_order(self, form, big):
        # Measured: the re-feed permutes blocks at the 2048 row chunk when insertion order is
        # not preserved, differently between rounds and between the table and the reader, so no
        # order is promised here and only the multiset of values is.
        big.execute("SET preserve_insertion_order = false")
        big.execute("SET threads = 4")
        big.execute("SELECT i FROM big ORDER BY i")
        assert sorted(as_table(big, form).column("i").to_pylist()) == list(range(ROW_GROUPS))


class TestFailures:
    """A refusal that can be asked before the take costs no rows; a failure after it puts none back."""

    LATE_ERROR = (
        f"SELECT CASE WHEN i = {TALLY_ROWS - 1} THEN error('late') ELSE i END AS i FROM range({TALLY_ROWS}) t(i)"
    )

    @pytest.mark.parametrize("form", FORMS)
    def test_a_late_engine_error_raises_from_the_arrow_call(self, form, duckdb_cursor):
        # The take runs the statement to its end, so an error past the first chunk surfaces here
        # rather than at a later read, and it stays a typed duckdb exception.
        duckdb_cursor.execute("SET threads = 1")
        duckdb_cursor.execute(self.LATE_ERROR)
        method = getattr(duckdb_cursor, form)
        with unwarned(), pytest.raises(duckdb.InvalidInputException, match="late"):
            method()
        assert duckdb_cursor.execute("SELECT 42").fetchall() == [(42,)]

    def test_a_failed_call_leaves_the_result_consumed(self, duckdb_cursor):
        duckdb_cursor.execute("SET threads = 1")
        duckdb_cursor.execute(self.LATE_ERROR)
        description = duckdb_cursor.description
        with unwarned(), pytest.raises(duckdb.InvalidInputException, match="late"):
            duckdb_cursor.to_arrow_table()
        assert duckdb_cursor.fetchall() == []
        assert duckdb_cursor.description == description

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("form", FORMS)
    def test_an_interrupt_during_the_take_leaves_the_result_consumed(self, form, duckdb_cursor):
        # No machine reads a hundred billion rows in a second, so the interrupt always lands inside
        # the take; the timer decides only when it arrives. The rows are moved out of the result
        # before the take runs, so an interrupt reads like any other failure after the take.
        duckdb_cursor.execute("SET threads = 2")
        duckdb_cursor.execute("SELECT i FROM range(100000000000) t(i)")
        description = duckdb_cursor.description
        method = getattr(duckdb_cursor, form)
        timer = threading.Timer(1.0, duckdb_cursor.interrupt)
        timer.daemon = True
        timer.start()
        try:
            with unwarned(), pytest.raises(duckdb.InterruptException):
                method()
        finally:
            timer.cancel()
            timer.join()
        assert duckdb_cursor.fetchall() == []
        assert duckdb_cursor.rowcount == -1
        assert duckdb_cursor.description == description
        assert duckdb_cursor.execute("SELECT 42").fetchall() == [(42,)]

    @staticmethod
    def with_an_aborted_transaction() -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.execute("CREATE TABLE t (i INTEGER)")
        con.execute("BEGIN")
        # A completed result leaves the connection free, so the failing statement can run beside it
        # and invalidate the transaction the result still sits in.
        con.execute("CALL pragma_table_info('t')")
        with pytest.raises(duckdb.ParserException):
            con.sql("INSERT INTO t VALUES (1); SELEC 2")
        return con

    @staticmethod
    def beside_an_open_relation_reader(con) -> "pa.RecordBatchReader":
        # A RETURNING write completes at execute(), which leaves the connection free for a relation
        # reader to take it; the reader holds it after one batch until it is read out or closed.
        con.execute("SET max_streaming_buffer_size='100KB'")
        con.execute("CREATE TABLE r (i INTEGER)")
        con.execute("INSERT INTO r SELECT range FROM range(5) RETURNING i")
        reader = con.sql(f"SELECT i FROM range({ROW_GROUPS}) t(i)").to_arrow_reader(1024)
        assert 0 < len(reader.read_next_batch()) <= 1024
        return reader

    @pytest.mark.parametrize("form", FORMS)
    def test_an_open_relation_reader_is_refused_before_the_take(self, form, duckdb_cursor):
        # The re-feed would be refused at its submit while the reader holds the connection, so the
        # refusal is asked for before the take, and the rows stay fetchable as fetchall, df and
        # fetchnumpy return them in the same state.
        reader = self.beside_an_open_relation_reader(duckdb_cursor)
        method = getattr(duckdb_cursor, form)
        try:
            with unwarned(), pytest.raises(duckdb.ResourceInUseException, match=OPEN_RESULT):
                method()
            assert duckdb_cursor.fetchall() == [(i,) for i in range(5)]
        finally:
            reader.close()

    @pytest.mark.parametrize("form", FORMS)
    def test_the_call_answers_every_row_once_the_reader_is_closed(self, form, duckdb_cursor):
        reader = self.beside_an_open_relation_reader(duckdb_cursor)
        method = getattr(duckdb_cursor, form)
        try:
            with unwarned(), pytest.raises(duckdb.ResourceInUseException, match=OPEN_RESULT):
                method()
        finally:
            reader.close()
        assert as_table(duckdb_cursor, form).column("i").to_pylist() == list(range(5))

    @pytest.mark.parametrize("form", FORMS)
    def test_an_aborted_transaction_is_refused_before_the_take(self, form):
        # A deliberate difference from the old path. The rows are in the client, but the re-feed
        # has to run a statement, and an aborted transaction refuses that only at the submit, by
        # which point the rows have been taken. So the transaction is checked first and the rows
        # stay fetchable, for the same reason a bad batch size is refused before the take.
        con = self.with_an_aborted_transaction()
        method = getattr(con, form)
        with unwarned(), pytest.raises(duckdb.TransactionException, match="aborted"):
            method()
        assert len(con.fetchall()) == 1


class TestRefusals:
    """What the methods do without a result to read."""

    @pytest.mark.parametrize("form", FORMS)
    def test_no_open_result_is_refused(self, form, duckdb_cursor):
        method = getattr(duckdb_cursor, form)
        with unwarned(), pytest.raises(duckdb.InvalidInputException, match=NO_RESULT):
            method()

    @pytest.mark.parametrize("form", FORMS)
    def test_a_module_form_without_a_result_is_refused(self, form, duckdb_cursor):
        method = getattr(duckdb, form)
        with unwarned(), pytest.raises(duckdb.InvalidInputException, match=NO_RESULT):
            method(connection=duckdb_cursor)

    @pytest.mark.parametrize("form", FORMS)
    def test_a_closed_connection_raises_the_connection_error(self, form):
        # The connection is checked before the result, so this is the connection error and not
        # "No open result set", which is what a row fetch answers because close() emptied the slot
        # and the fetch never reaches a connection check. A reader built before the close reads on.
        # The match is loose because the client and the engine word the refusal differently.
        con = duckdb.connect()
        con.execute("SELECT 42 AS a")
        con.close()
        method = getattr(con, form)
        with pytest.raises(duckdb.InvalidInputException, match=NO_RESULT):
            con.fetchall()
        with unwarned(), pytest.raises(duckdb.ConnectionException, match=CLOSED):
            method()

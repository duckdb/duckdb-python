import gc
import importlib.util
import re
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import duckdb
from duckdb.sqltypes import BIGINT

ROWS = 1_000_000
VECTOR = 2048
# An undecided result parks its first chunk at the sink, so settling runs about one vector.
SETTLED = 4 * VECTOR
OPEN_RESULT = "open result"
# The refusal while a result of execute() holds the connection, which only the DB-API can end
UNFINISHED_EXECUTE = "unfinished result from execute"
BUSY = "is busy with another call"
INSIDE_A_QUERY = "inside a query cannot use"
# Which of the two refuses a call from another thread depends on whether it lands during engine work or between it
REFUSED = f"{OPEN_RESULT}|{BUSY}"

needs_polars = pytest.mark.skipif(importlib.util.find_spec("polars") is None, reason="polars not installed")
needs_pandas = pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="pandas not installed")
needs_pyarrow = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")


@pytest.fixture
def con():
    connection = duckdb.connect()
    # One thread bounds how far a statement runs before it is read.
    connection.execute("SET threads = 1")
    yield connection
    connection.close()


@pytest.fixture
def produced(con):
    count = [0]

    def tally(i):
        count[0] += 1
        return i

    con.create_function("tally", tally, [BIGINT], BIGINT, side_effects=True)
    return count


def run_isolated(script, timeout=15):
    # A child process keeps a deadlock from stalling the suite; the marker tells a hang from a slow start.
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as hang:
        started = hang.stdout or ""
        if "started" not in (started.decode() if isinstance(started, bytes) else started):
            msg = "the child never reached the statement under test"
            raise AssertionError(msg) from hang
        raise
    return proc.returncode, proc.stdout + proc.stderr


class TestExecuteSettles:
    def test_an_error_in_the_first_chunk_raises_from_execute(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute("SELECT error('boom')")

    def test_a_parameterized_error_raises_from_execute(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute("SELECT error(?)", ["boom"])

    def test_the_last_statement_of_a_script_settles_too(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute("SELECT 1; SELECT error('boom')")

    def test_a_failing_statement_inside_a_script_stops_it(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute("CREATE TABLE a (i INT); SELECT error('boom'); CREATE TABLE b (i INT)")
        tables = con.execute("SELECT table_name FROM duckdb_tables() ORDER BY 1").fetchall()
        assert tables == [("a",)]

    @pytest.mark.parametrize(("parameter_sets", "ran"), [([[1], [2]], 0), ([[2], [1]], 1)])
    def test_executemany_stops_at_the_failing_parameter_set(self, con, produced, parameter_sets, ran):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.executemany("SELECT tally(CASE WHEN ? = 1 THEN error('boom')::BIGINT ELSE 0 END)", parameter_sets)
        assert produced[0] == ran

    def test_the_first_chunk_runs_and_the_rest_waits(self, con, produced):
        con.execute(f"SELECT tally(i) FROM range({ROWS}) t(i)")
        assert 0 < produced[0] <= SETTLED

    def test_settling_loses_no_rows(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        assert [column[0] for column in con.description] == ["range"]
        rows = con.fetchall()
        assert len(rows) == ROWS
        assert rows[:3] == [(0,), (1,), (2,)]
        assert rows[-1] == (ROWS - 1,)

    def test_an_empty_result_settles(self, con):
        con.execute("SELECT * FROM range(0)")
        assert [column[0] for column in con.description] == ["range"]
        assert con.rowcount == -1
        assert con.fetchall() == []

    def test_an_error_after_the_first_chunk_waits_for_the_fetch(self, con):
        con.execute("BEGIN")
        con.execute(f"SELECT CASE WHEN i = {ROWS - 1} THEN error('late') ELSE i END FROM range({ROWS}) t(i)")
        with pytest.raises(duckdb.InvalidInputException, match="late"):
            con.fetchall()
        with pytest.raises(duckdb.TransactionException):
            con.execute("SELECT 1")
        con.execute("ROLLBACK")
        assert con.execute("SELECT 42").fetchall() == [(42,)]

    @pytest.mark.timeout(60)
    def test_an_interrupt_while_settling_raises_from_execute(self, con):
        timer = threading.Timer(0.5, con.interrupt)
        timer.start()
        try:
            with pytest.raises(duckdb.InterruptException):
                # An aggregate emits its only row at the end and needs no memory while it runs.
                con.execute("SELECT sum(a.range * b.range) FROM range(200000) a, range(200000) b")
        finally:
            timer.cancel()

    RETURNING = pytest.mark.parametrize(
        ("statement", "returned", "remaining"),
        [
            pytest.param(
                "INSERT INTO t SELECT range FROM range(5000, 10000) RETURNING i", range(5000, 10000), 10000, id="insert"
            ),
            pytest.param("DELETE FROM t WHERE i < 5000 RETURNING i", range(5000), 0, id="delete"),
        ],
    )

    @RETURNING
    def test_returning_completes_in_execute(self, con, statement, returned, remaining):
        con.execute("CREATE TABLE t AS SELECT range AS i FROM range(5000)")
        con.execute(statement)
        assert con.cursor().execute("SELECT count(*) FROM t").fetchall() == [(remaining,)]
        assert sorted(con.fetchall()) == [(i,) for i in returned]

    @RETURNING
    def test_unread_returning_survives_the_next_statement(self, con, statement, returned, remaining):
        con.execute("CREATE TABLE t AS SELECT range AS i FROM range(5000)")
        con.execute(statement)
        con.execute("SELECT 1")
        assert con.execute("SELECT count(*) FROM t").fetchall() == [(remaining,)]

    @pytest.mark.parametrize(
        "statement",
        [
            pytest.param("CREATE TABLE u AS SELECT {late} AS x FROM range({rows}) t(i)", id="ctas"),
            pytest.param("INSERT INTO t SELECT {late} FROM range({rows}) t(i)", id="insert"),
            pytest.param("UPDATE t SET i = {late}", id="update"),
        ],
    )
    def test_a_write_that_fails_late_raises_from_execute_and_leaves_no_trace(self, con, statement):
        con.execute(f"CREATE TABLE t AS SELECT range AS i FROM range({ROWS})")
        late = f"CASE WHEN i = {ROWS - 1} THEN error('boom')::BIGINT ELSE i + 1 END"
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute(statement.format(late=late, rows=ROWS))
        assert con.execute("SELECT count(*), sum(i) FROM t").fetchall() == [(ROWS, ROWS * (ROWS - 1) // 2)]
        assert con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'u'").fetchall() == [(0,)]

    @pytest.mark.skip(
        reason="engine deadlock in the asynchronous CSV writer's abort path; "
        "see agent-notes duckdb-python runbooks/engine-issues-to-report-upstream.md"
    )
    @pytest.mark.timeout(60, method="thread")
    def test_a_copy_that_fails_late_raises_from_execute(self, con, tmp_path):
        late = f"CASE WHEN i = {ROWS - 1} THEN error('boom')::BIGINT ELSE i END"
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute(f"COPY (SELECT {late} FROM range({ROWS}) t(i)) TO '{(tmp_path / 'out.csv').as_posix()}'")

    def test_a_failed_select_aborts_the_transaction(self, con):
        con.execute("BEGIN")
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        with pytest.raises(duckdb.InvalidInputException):
            con.execute("SELECT error('boom')")
        with pytest.raises(duckdb.TransactionException):
            con.execute("SELECT 1")
        con.execute("ROLLBACK")
        assert con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 't'").fetchall() == [(0,)]


class TestSupersede:
    def test_a_new_statement_replaces_the_open_result(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        con.execute("SELECT 42")
        assert con.fetchall() == [(42,)]

    def test_a_superseded_read_only_select_leaves_the_transaction_valid(self, con):
        con.execute("BEGIN")
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        con.execute("INSERT INTO t VALUES (2)")
        con.execute("COMMIT")
        assert con.execute("SELECT a FROM t ORDER BY a").fetchall() == [(1,), (2,)]

    def test_a_superseded_statement_that_may_write_is_drained(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        con.execute(f"SELECT nextval('s') FROM range({ROWS})")
        con.fetchone()
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        con.execute("COMMIT")
        assert con.execute("SELECT currval('s')").fetchall() == [(ROWS,)]

    def test_an_error_in_the_drained_remainder_raises_from_the_next_execute(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        late = f"CASE WHEN i = {ROWS - 1} THEN error('late')::BIGINT ELSE 0 END"
        con.execute(f"SELECT nextval('s') + {late} FROM range({ROWS}) t(i)")
        con.fetchone()
        with pytest.raises(duckdb.InvalidInputException, match="late"):
            con.execute("SELECT 42")
        with pytest.raises(duckdb.TransactionException):
            con.execute("SELECT 1")
        con.execute("ROLLBACK")

    def test_a_superseded_read_only_select_keeps_its_side_effects_and_runs_no_further(self, con, produced):
        con.execute(f"SELECT tally(i) FROM range({ROWS}) t(i)")
        settled = produced[0]
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        time.sleep(0.5)
        assert 0 < settled <= SETTLED
        assert produced[0] == settled

    @pytest.mark.parametrize(
        "rollback",
        [
            pytest.param(lambda c: c.rollback(), id="method"),
            pytest.param(lambda c: c.execute("ROLLBACK"), id="statement"),
            pytest.param(lambda c: c.execute("ABORT"), id="abort"),
            pytest.param(lambda c: c.execute("ROLLBACK TRANSACTION"), id="rollback_transaction"),
            pytest.param(lambda c: c.execute("ABORT WORK"), id="abort_work"),
            pytest.param(lambda c: c.execute(c.extract_statements("ROLLBACK")[0]), id="prepared_statement"),
        ],
    )
    def test_a_rollback_cancels_an_open_result_instead_of_draining_it(self, con, rollback):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        late = f"CASE WHEN i = {ROWS - 1} THEN error('late')::BIGINT ELSE 0 END"
        con.execute(f"SELECT nextval('s') + {late} FROM range({ROWS}) t(i)")
        con.fetchone()
        rollback(con)
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        (advanced,) = con.execute("SELECT currval('s')").fetchone()
        assert advanced < ROWS

    @pytest.mark.parametrize("statement", ["PRAGMA show_tables", "SHOW TABLES"])
    def test_parsing_a_preprocessed_statement_leaves_the_open_result_alone(self, con, statement):
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        assert con.execute(statement).fetchall() == [("t",)]

    def test_a_commit_drains_an_open_result_that_may_write(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        con.execute(f"SELECT nextval('s') FROM range({ROWS})")
        con.fetchone()
        con.commit()
        assert con.execute("SELECT currval('s')").fetchall() == [(ROWS,)]

    def test_executemany_drains_an_open_result_that_may_write(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute(f"SELECT nextval('s') FROM range({ROWS})")
        con.fetchone()
        con.executemany("SELECT ?", [[1], [2]])
        assert con.execute("SELECT currval('s')").fetchall() == [(ROWS,)]

    def test_a_refused_relation_leaves_the_open_result_to_the_next_statement(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute(f"SELECT nextval('s') FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            con.sql("SELECT 1").fetchall()
        assert con.execute("SELECT 2").fetchall() == [(2,)]
        assert con.execute("SELECT currval('s')").fetchall() == [(ROWS,)]

    @pytest.mark.parametrize(
        "commit",
        [pytest.param(lambda c: c.execute("COMMIT"), id="statement"), pytest.param(lambda c: c.commit(), id="method")],
    )
    def test_commit_while_a_read_only_result_is_open(self, con, commit):
        con.execute("BEGIN")
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        commit(con)
        assert con.execute("SELECT a FROM t").fetchall() == [(1,)]


class TestOpenResultIsExclusive:
    @pytest.fixture
    def open_result(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        return con

    @pytest.mark.parametrize(
        ("prepare", "run"),
        [
            pytest.param(lambda c: c.sql("SELECT 1"), lambda r: r.fetchall(), id="relation"),
            pytest.param(lambda c: c.table("t"), lambda r: r.fetchall(), id="table"),
            pytest.param(
                lambda c: c, lambda c: c.create_function("f", lambda x: x, [BIGINT], BIGINT), id="create_function"
            ),
        ],
    )
    def test_executing_anything_but_the_dbapi_errors(self, con, prepare, run):
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        # Building a relation only binds, so it succeeds; running it is what conflicts with the open result.
        target = prepare(con)
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            run(target)
        assert con.fetchone() == (1,)

    def test_constructing_relations_leaves_the_open_result_alone(self, con):
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        relations = [con.sql("SELECT 1 AS b"), con.table("t"), con.values([1, 2])]
        assert [r.columns for r in relations] == [["b"], ["a"], ["col0", "col1"]]
        assert con.fetchone() == (1,)

    @needs_pandas
    def test_register_errors(self, con):
        import pandas as pd

        frame = pd.DataFrame({"a": [1]})
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            con.register("v", frame)
        assert con.fetchone() == (1,)

    @pytest.mark.parametrize("release", ["complete", "abort", "fetchall"])
    def test_releasing_the_result_frees_the_connection(self, open_result, release):
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            open_result.sql("SELECT 1").fetchall()
        getattr(open_result, release)()
        assert open_result.sql("SELECT 42").fetchall() == [(42,)]

    def test_complete_runs_the_rest_of_the_result(self, con, produced):
        con.execute(f"SELECT tally(i) FROM range({ROWS}) t(i)")
        con.complete()
        assert produced[0] == ROWS

    def test_complete_keeps_a_transaction_valid_after_a_write(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        con.execute(f"SELECT nextval('s') FROM range({ROWS})")
        con.complete()
        con.execute("COMMIT")
        assert con.execute("SELECT currval('s')").fetchall() == [(ROWS,)]

    def test_abort_of_a_statement_that_may_write_invalidates_the_transaction(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        con.execute(f"SELECT nextval('s') FROM range({ROWS})")
        con.fetchone()
        con.abort()
        with pytest.raises(duckdb.TransactionException):
            con.execute("SELECT 1")
        con.execute("ROLLBACK")

    def test_abort_stops_the_result_where_it_settled(self, con, produced):
        con.execute(f"SELECT tally(i) FROM range({ROWS}) t(i)")
        settled = produced[0]
        con.abort()
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        time.sleep(0.5)
        assert 0 < settled <= SETTLED
        assert produced[0] == settled

    def test_a_cursor_is_an_independent_connection(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        assert con.fetchone() == (0,)
        cursor = con.cursor()
        assert cursor.execute("SELECT 42").fetchall() == [(42,)]
        assert con.fetchone() == (1,)
        cursor.execute(f"SELECT * FROM range({ROWS})")
        assert cursor.fetchone() == (0,)
        con.fetchall()
        assert con.sql("SELECT 43").fetchall() == [(43,)]
        assert cursor.fetchone() == (1,)

    @pytest.mark.parametrize(
        "use",
        [
            pytest.param(lambda c: c.execute("SELECT 1"), id="execute"),
            pytest.param(lambda c: c.sql("SELECT 1").fetchall(), id="relation"),
        ],
    )
    @needs_pyarrow
    def test_a_relation_stream_blocks_its_connection(self, con, use):
        reader = con.sql(f"SELECT * FROM range({ROWS})").to_arrow_reader(1000)
        try:
            reader.read_next_batch()
            with pytest.raises(duckdb.ResourceInUseException, match=OPEN_RESULT):
                use(con)
        finally:
            reader.close()
        assert con.execute("SELECT 42").fetchall() == [(42,)]

    @needs_polars
    @needs_pyarrow
    def test_a_lazy_polars_frame_scanned_on_its_own_connection_errors(self):
        code, output = run_isolated(
            """
            import duckdb
            con = duckdb.connect()
            con.execute("CREATE TABLE t AS SELECT range AS x FROM range(10)")
            lf = con.table("t").pl(lazy=True)
            print("started", flush=True)
            try:
                con.execute("SELECT count(*) FROM lf").fetchall()
            except duckdb.Error as error:
                print("refused:", error)
            print("usable after:", con.execute("SELECT 42").fetchall())
            """
        )
        assert code == 0, output
        assert "refused:" in output
        assert re.search(REFUSED, output)
        assert "usable after: [(42,)]" in output

    def test_a_udf_that_reenters_its_connection_errors(self):
        code, output = run_isolated(
            """
            import duckdb
            from duckdb.sqltypes import BIGINT
            con = duckdb.connect()
            con.create_function("inner", lambda x: con.execute("SELECT 1").fetchone()[0], [BIGINT], BIGINT)
            print("started", flush=True)
            try:
                con.execute("SELECT inner(1)").fetchall()
            except Exception as error:
                print(type(error).__name__, error)
            """
        )
        assert code == 0, output
        assert "InvalidInputException" in output
        assert INSIDE_A_QUERY in output

    @needs_polars
    @needs_pyarrow
    def test_a_lazy_polars_frame_cannot_open_while_a_result_is_open(self, con):
        con.execute("CREATE TABLE b AS SELECT range::INT AS k FROM range(3)")
        frame = con.table("b").pl(lazy=True)
        reader = con.sql(f"SELECT * FROM range({ROWS})").to_arrow_reader(1000)
        reader.read_next_batch()
        try:
            # polars may re-raise the source's error under its own exception types.
            with pytest.raises(Exception, match=OPEN_RESULT):
                frame.collect()
        finally:
            reader.close()
        assert frame.collect().height == 3

    @needs_polars
    @needs_pyarrow
    def test_joining_two_lazy_polars_frames_of_one_connection_settles(self, con):
        con.execute("CREATE TABLE a AS SELECT range::INT AS k FROM range(5)")
        con.execute("CREATE TABLE b AS SELECT range::INT AS k FROM range(3)")
        joined = con.table("a").pl(lazy=True).join(con.table("b").pl(lazy=True), on="k")
        # Whether polars opens both sources at once depends on its scheduling, so a refusal (either kind, since the
        # second call may land during engine work) and the joined rows are both correct; the connection must work after.
        refused = None
        try:
            rows = sorted(joined.collect()["k"].to_list())
        except Exception as error:
            refused, rows = str(error), None
        if refused is not None:
            assert re.search(REFUSED, refused)
        else:
            assert rows == [0, 1, 2]
        assert con.execute("SELECT 42").fetchall() == [(42,)]

    @pytest.mark.parametrize(
        "callback",
        [
            pytest.param('len(rel.filter("a > 0").columns)', id="builds_a_relation"),
            pytest.param('len(con.table("t").columns)', id="looks_up_a_table"),
            pytest.param('con.sql("SELECT 1").fetchall()[0][0]', id="runs_a_relation"),
            pytest.param('len(rel.select_types(["INTEGER"]).columns)', id="selects_by_type"),
            pytest.param("len(rel.map(lambda df: df).columns)", id="maps_a_relation"),
            pytest.param("rel.insert([5]) or 0", id="inserts"),
            pytest.param('rel.update({"a": duckdb.ConstantExpression(3)}) or 0', id="updates"),
            pytest.param('len(rel.explain(type="analyze"))', id="explains"),
            pytest.param('con.execute("SELECT 1").fetchall()[0][0]', id="executes"),
            pytest.param("con.close() or 0", id="closes"),
        ],
    )
    def test_a_udf_that_uses_its_connection_while_the_query_runs_errors(self, callback):
        code, output = run_isolated(
            f"""
            import duckdb
            from duckdb.sqltypes import BIGINT
            con = duckdb.connect()
            con.execute("CREATE TABLE t AS SELECT 1 AS a")
            rel = con.table("t")
            con.create_function("inner", lambda x: {callback}, [BIGINT], BIGINT)
            print("started", flush=True)
            try:
                con.execute("SELECT inner(i) FROM range(3) t(i)").fetchall()
            except duckdb.Error as error:
                print("refused:", error)
            print("usable after:", con.execute("SELECT 42").fetchall())
            """
        )
        assert code == 0, output
        assert "refused:" in output
        assert INSIDE_A_QUERY in output
        assert "usable after: [(42,)]" in output

    def test_a_udf_on_a_worker_between_two_fetches_is_refused(self):
        # A worker can still be inside a UDF after a fetch returns, while no call drives the query at all
        code, output = run_isolated(
            """
            import threading
            import duckdb
            from duckdb.sqltypes import BIGINT
            con = duckdb.connect()
            # Workers are what runs the scan between fetches, so a one-CPU host must still get some
            con.execute("SET threads = 4")
            # An unordered result lets any chunk answer the fetch, so the worker held below cannot stall it
            con.execute("SET preserve_insertion_order = false")
            main = threading.get_ident()
            claim, fetched, called = threading.Lock(), threading.Event(), threading.Event()
            def inner(x):
                # Producers park once the result buffer fills, possibly before the fetch returns, so one worker is
                # held here until it does
                if threading.get_ident() != main and claim.acquire(blocking=False):
                    assert fetched.wait(30), "the fetch never returned"
                    try:
                        con.execute("SELECT 1")
                        print("ran", flush=True)
                    except duckdb.Error as error:
                        print("refused:", error, flush=True)
                    finally:
                        called.set()
                return x
            con.create_function("inner", inner, [BIGINT], BIGINT)
            # A table scan splits into a task per row group; one range() source may only run when a fetch drives it
            con.execute("CREATE TABLE src AS SELECT range AS i FROM range(5000000)")
            print("started", flush=True)
            con.execute("SELECT inner(i) FROM src")
            con.fetchone()
            fetched.set()
            assert called.wait(10)
            con.abort()
            print("usable after:", con.execute("SELECT 42").fetchall())
            """,
            # Ending the query waits for its running scan tasks, each in the UDF, which takes long under sanitizers
            timeout=60,
        )
        assert code == 0, output
        assert "refused:" in output
        assert INSIDE_A_QUERY in output
        assert "usable after: [(42,)]" in output

    def test_another_thread_is_refused_while_a_statement_runs(self, con):
        running, release = threading.Event(), threading.Event()

        def hold(x):
            running.set()
            release.wait(10)
            return x

        con.create_function("hold", hold, [BIGINT], BIGINT)
        outcome = []

        def contend():
            running.wait(10)
            try:
                con.execute("SELECT 1")
                outcome.append("ran")
            except duckdb.ResourceInUseException as error:
                outcome.append(str(error))
            finally:
                release.set()

        thread = threading.Thread(target=contend)
        thread.start()
        assert con.execute("SELECT hold(i) FROM range(1) t(i)").fetchall() == [(0,)]
        thread.join(10)
        assert len(outcome) == 1
        assert BUSY in outcome[0]

    @pytest.mark.timeout(60)
    def test_progress_and_interrupt_reach_a_running_statement(self, con):
        seen = []

        def poll():
            try:
                seen.append(con.query_progress())
            finally:
                con.interrupt()

        timer = threading.Timer(0.5, poll)
        timer.start()
        try:
            with pytest.raises(duckdb.InterruptException):
                con.execute("SELECT sum(a.range * b.range) FROM range(200000) a, range(200000) b")
        finally:
            timer.cancel()
        assert len(seen) == 1
        assert isinstance(seen[0], float)

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda c: c.remove_function("f"), id="remove_function"),
            pytest.param(lambda c: c.unregister("v"), id="unregister"),
            pytest.param(lambda c: c.load_extension("json"), id="load_extension"),
        ],
    )
    def test_catalog_changes_that_run_no_statement_error(self, con, call):
        con.create_function("f", lambda x: x, [BIGINT], BIGINT)
        con.register("v", con.sql("SELECT 1"))
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            call(con)
        assert con.fetchone() == (1,)

    def test_a_refused_statement_leaves_the_transaction_valid(self, con):
        con.execute("CREATE TABLE t (i INTEGER)")
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (1)")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.OperationalError, match=UNFINISHED_EXECUTE):
            con.sql("INSERT INTO t VALUES (2)")
        assert len(con.fetchall()) == ROWS - 1
        con.execute("COMMIT")
        assert con.execute("SELECT i FROM t").fetchall() == [(1,)]

    @needs_pyarrow
    def test_dropping_a_relation_stream_frees_the_connection(self, con):
        reader = con.sql(f"SELECT * FROM range({ROWS})").to_arrow_reader(1000)
        reader.read_next_batch()
        del reader
        gc.collect()
        assert con.execute("SELECT 42").fetchall() == [(42,)]

    def test_materializing_a_relation_while_a_result_is_open_errors(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            con.sql("SELECT 1").materialize()
        assert con.fetchone() == (1,)

    @needs_pandas
    def test_append_while_a_result_is_open_errors(self, con):
        import pandas as pd

        con.execute("CREATE TABLE t (a BIGINT)")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            con.append("t", pd.DataFrame({"a": [1]}))
        assert con.fetchone() == (1,)

    def test_the_refusal_says_how_to_end_the_result(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.ResourceInUseException) as refusal:
            con.sql("SELECT 1").fetchall()
        assert re.search(r"complete\(\) or abort\(\).*cursor\(\)", str(refusal.value))

    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(lambda c: c.table("t").insert([2]), id="insert"),
            pytest.param(lambda c: c.sql("SELECT 1 AS a").create_view("v"), id="create_view"),
            pytest.param(lambda c: c.sql("SELECT 1 AS a").query("x", "SELECT a FROM x"), id="query"),
        ],
    )
    def test_relation_work_outside_a_statement_names_the_result(self, con, run):
        con.execute("CREATE TABLE t (a BIGINT)")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            run(con)
        assert con.fetchone() == (1,)

    @needs_pyarrow
    def test_a_reader_that_holds_the_connection_is_not_blamed_on_execute(self, con):
        con.execute("SELECT 1")
        assert con.fetchone() == (1,)
        assert con.fetchone() is None
        reader = con.sql(f"SELECT * FROM range({ROWS})").to_arrow_reader(1024)
        reader.read_next_batch()
        with pytest.raises(duckdb.ResourceInUseException, match=OPEN_RESULT):
            con.sql("SELECT 1").fetchall()
        reader.close()
        assert con.sql("SELECT 42").fetchall() == [(42,)]

    def test_a_single_row_result_holds_the_connection_until_a_fetch_finds_its_end(self, con):
        con.execute("SELECT count(*) FROM range(10)")
        assert con.fetchone() == (10,)
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            con.sql("SELECT 1").fetchall()
        assert con.fetchone() is None
        assert con.sql("SELECT 42").fetchall() == [(42,)]

    def test_a_cursor_result_holds_only_the_cursor(self, con):
        cursor = con.cursor()
        cursor.execute(f"SELECT * FROM range({ROWS})")
        cursor.fetchone()
        assert con.sql("SELECT 42").fetchall() == [(42,)]
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            cursor.sql("SELECT 1").fetchall()
        assert cursor.fetchone() == (1,)

    def test_an_interrupted_materialize_does_not_hang(self):
        code, output = run_isolated(
            """
            import _thread
            import signal
            import threading
            import time
            import duckdb
            from duckdb.sqltypes import BIGINT
            # A background job starts with SIGINT ignored, which would turn the simulated Ctrl-C into nothing
            signal.signal(signal.SIGINT, signal.default_int_handler)
            con = duckdb.connect()
            con.execute("SET threads = 4")
            con.execute("CREATE TABLE t AS SELECT range AS i FROM range(2000000)")
            # A builtin runs no bytecode, so the Ctrl-C reaches the client's signal check while the scan's
            # tasks are still in the UDF, instead of raising inside a UDF call and failing the query
            calls = []
            con.create_function("record", calls.append, [BIGINT], BIGINT, null_handling="special")
            rel = con.sql("SELECT record(i) FROM t")
            print("started", flush=True)

            # Waiting for the UDF to run, not for a fixed time, keeps a fast machine from finishing the scan first
            def interrupt_once_the_scan_runs():
                while len(calls) < 10000:
                    time.sleep(0.001)
                _thread.interrupt_main()

            threading.Thread(target=interrupt_once_the_scan_runs, daemon=True).start()
            try:
                rel.materialize()
            except (RuntimeError, KeyboardInterrupt) as error:
                print("interrupted:", type(error).__name__, flush=True)
            print("usable after:", con.execute("SELECT 42").fetchall())
            """,
            # Ending the query waits for its running scan tasks, so the child lasts about as long as the whole query
            timeout=60,
        )
        assert code == 0, output
        assert "interrupted:" in output
        assert "usable after: [(42,)]" in output


def fetch_rows_into(con, rows, limit):
    while len(rows) < limit:
        rows.append(con.fetchone())


class TestRowFetches:
    def test_an_interrupt_between_fetches_is_reported_once_the_fetched_chunk_runs_out(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        assert [con.fetchone() for _ in range(5)] == [(i,) for i in range(5)]
        con.interrupt()
        rows = []
        with pytest.raises(duckdb.InterruptException):
            fetch_rows_into(con, rows, 2 * VECTOR)
        # A row fetch asks the engine only for the next chunk, so the rest of the first one comes back first
        assert rows == [(i,) for i in range(5, VECTOR)]
        assert con.execute("SELECT 42").fetchall() == [(42,)]

    @needs_pandas
    def test_a_bulk_fetch_after_an_interrupt_takes_the_fetched_rows_then_raises(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        assert con.fetchone() == (0,)
        con.interrupt()
        assert con.fetch_df_chunk(1)["range"].tolist() == list(range(1, VECTOR))
        with pytest.raises(duckdb.InterruptException):
            con.fetch_df_chunk(1)
        assert con.execute("SELECT 42").fetchall() == [(42,)]

    @pytest.mark.timeout(60)
    def test_a_failure_on_a_worker_is_reported_once_the_fetched_chunk_runs_out(self, con):
        # Workers keep running the scan after a fetch returns, until the buffer of rows produced ahead is full
        con.execute("SET threads = 4")
        con.execute("SET max_streaming_buffer_size = '1GB'")
        failing, failed = threading.Event(), threading.Event()

        def checked(i):
            if failing.is_set():
                failed.set()
                msg = "failed between fetches"
                raise ValueError(msg)
            return i

        con.create_function("checked", checked, [BIGINT], BIGINT)
        # A table scan splits into a task per row group; one range() source may only run when a fetch drives it
        con.execute("CREATE TABLE src AS SELECT range AS i FROM range(5000000)")
        con.execute("SELECT checked(i) FROM src")
        assert con.fetchone() == (0,)
        failing.set()
        assert failed.wait(10)
        rows = []
        with pytest.raises(duckdb.InvalidInputException, match="failed between fetches"):
            fetch_rows_into(con, rows, ROWS)
        assert len(rows) >= VECTOR - 1
        assert rows == [(i,) for i in range(1, len(rows) + 1)]
        assert con.execute("SELECT 42").fetchall() == [(42,)]

import importlib.util
import json
import re
import subprocess
import sys
import textwrap

import pytest

import duckdb

ROWS = 1_000_000
UNFINISHED_EXECUTE = "unfinished result from execute"
ABORTED = "Current transaction is aborted"
NO_STATEMENT_TAKES_PARAMETERS = "Parameters were given, but the last statement of the query takes none"

needs_fsspec = pytest.mark.skipif(importlib.util.find_spec("fsspec") is None, reason="fsspec not installed")

RUNNERS = {
    "execute": lambda con, script: con.execute(script),
    "sql": lambda con, script: con.sql(script),
    "executemany": lambda con, script: con.executemany(script, [[]]),
}


def runner(name):
    return pytest.param(RUNNERS[name], id=name)


WITH_PARAMETERS = [
    pytest.param(lambda con, script, values: con.execute(script, values), id="execute"),
    pytest.param(lambda con, script, values: con.sql(script, params=values), id="sql"),
    pytest.param(lambda con, script, values: con.executemany(script, [values]), id="executemany"),
]


@pytest.fixture
def con():
    connection = duckdb.connect()
    # One thread bounds how far a statement runs before it is read.
    connection.execute("SET threads = 1")
    yield connection
    connection.close()


def tables(con):
    return [name for (name,) in con.execute("SELECT table_name FROM duckdb_tables() ORDER BY 1").fetchall()]


class BrokenLength:
    def __len__(self) -> int:
        msg = "no length today"
        raise ValueError(msg)


class BrokenLengthSet(list):
    def __len__(self) -> int:
        msg = "no length today"
        raise ValueError(msg)


class RefusedLength:
    def __len__(self) -> int:
        msg = "length refused"
        raise TypeError(msg)


def run_isolated(script, timeout):
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


INTERRUPTED_SCAN = """
            import _thread
            import io
            import signal
            import threading
            import time
            import duckdb
            import fsspec
            from duckdb.sqltypes import BIGINT
            # A background job starts with SIGINT ignored, which would turn the simulated Ctrl-C into nothing
            signal.signal(signal.SIGINT, signal.default_int_handler)
            con = duckdb.connect(config={"threads": 1})
            calls = []
            data = b"1\\n" * 2000000

            class File(io.BytesIO):
                def close(self):
                    if not self.closed:
                        try:
                            con.execute("SELECT 99")
                            print("close ran", flush=True)
                        except duckdb.Error as error:
                            print("close refused:", type(error).__name__, flush=True)
                    super().close()

            class FileSystem(fsspec.AbstractFileSystem):
                protocol = "closingfs"

                def _open(self, path, mode="rb", **kwargs):
                    handle = File(data)
                    handle.size = len(data)
                    return handle

                def info(self, path, **kwargs):
                    return {"name": path, "size": len(data), "type": "file"}

                def glob(self, path, **kwargs):
                    return [path]

            con.register_filesystem(FileSystem())
            con.create_function("record", calls.append, [BIGINT], BIGINT, null_handling="special")

            def interrupt_once_the_scan_runs():
                while len(calls) < 2000:
                    time.sleep(0.001)
                _thread.interrupt_main()

            threading.Thread(target=interrupt_once_the_scan_runs, daemon=True).start()
            print("started", flush=True)
            try:
                con.execute(@QUERY@)
            except (RuntimeError, KeyboardInterrupt) as error:
                print("interrupted:", type(error).__name__, flush=True)
            print("usable after:", con.execute("SELECT 42").fetchall())
"""
SCAN = "SELECT record(i) FROM read_csv('closingfs://data.csv', columns={'i': 'BIGINT'}, header=false)"


def interrupt_a_scan_whose_file_reenters_the_connection_on_close(query):
    code, output = run_isolated(
        INTERRUPTED_SCAN.replace("@QUERY@", repr(query)),
        # Ending the query waits for its running scan, which runs the UDF on every row it reads
        timeout=60,
    )
    assert code == 0, output
    assert "interrupted:" in output
    # The file closes while the interrupted query ends, which still holds the connection
    assert "close refused: ResourceInUseException" in output
    assert "usable after: [(42,)]" in output


class TestEachStatementSeesTheOnesBefore:
    @pytest.mark.parametrize(
        "importing",
        [
            pytest.param("PRAGMA import_database('{}')", id="pragma"),
            pytest.param("IMPORT DATABASE '{}'", id="import"),
        ],
    )
    @pytest.mark.parametrize(
        "run",
        [
            runner("execute"),
            runner("sql"),
            runner("executemany"),
        ],
    )
    def test_an_import_reads_the_export_written_before_it(self, con, tmp_path, run, importing):
        # The import reads the export's files while it is expanded, so it fails unless the export already ran.
        export = (tmp_path / "export").as_posix()
        con.execute("CREATE TABLE integers AS SELECT 42 AS i")
        run(con, f"EXPORT DATABASE '{export}'; DROP TABLE integers; {importing.format(export)}")
        assert con.execute("SELECT i FROM integers").fetchall() == [(42,)]

    @pytest.mark.parametrize("run", [runner("execute"), runner("sql")])
    def test_a_pragma_reads_a_table_created_before_it(self, con, run):
        result = run(con, "CREATE TABLE t (a INTEGER, b VARCHAR); PRAGMA table_info('t')")
        assert [row[1] for row in result.fetchall()] == ["a", "b"]

    def test_an_earlier_statement_finds_the_callers_python_objects(self, con):
        source = con.sql("SELECT range AS a FROM range(3)")  # noqa: F841
        assert con.execute("CREATE TABLE r AS SELECT * FROM source; SELECT sum(a) FROM r").fetchall() == [(3,)]


class TestAnErrorStopsTheScriptAtItsStatement:
    @pytest.mark.parametrize(
        ("failing", "error"),
        [
            pytest.param("SELEC 1", duckdb.ParserException, id="syntax"),
            pytest.param("PRAGMA import_database('{missing}')", duckdb.IOException, id="expansion"),
            pytest.param("SELECT * FROM missing", duckdb.CatalogException, id="binder"),
        ],
    )
    @pytest.mark.parametrize("run", [runner("execute"), runner("sql")])
    def test_the_statements_before_the_error_ran_and_the_ones_after_did_not(self, con, tmp_path, run, failing, error):
        missing = (tmp_path / "missing").as_posix()
        with pytest.raises(error):
            run(con, f"CREATE TABLE before (i INT); {failing.format(missing=missing)}; CREATE TABLE after (i INT)")
        assert tables(con) == ["before"]

    @needs_fsspec
    def test_an_interrupted_earlier_statement_ends_its_query_before_the_connection_is_free(self):
        interrupt_a_scan_whose_file_reenters_the_connection_on_close(SCAN + "; SELECT 42")

    @needs_fsspec
    def test_an_interrupted_last_statement_ends_its_query_before_the_connection_is_free(self):
        # An aggregate emits its only row at the end, so the interrupt lands while execute() still drives the scan
        interrupt_a_scan_whose_file_reenters_the_connection_on_close(SCAN.replace("record(i)", "count(record(i))"))

    @pytest.mark.parametrize("run", [runner("execute"), runner("sql"), runner("executemany")])
    def test_a_parse_error_shows_its_place_in_the_query_as_the_engine_does(self, con, run):
        # get_table_names parses through the engine's own path, which adds the place of the error
        with pytest.raises(duckdb.ParserException) as from_the_engine:
            con.get_table_names("SELEC 2")
        with pytest.raises(duckdb.ParserException) as alone:
            run(con, "SELEC 2")
        assert str(alone.value) == str(from_the_engine.value)
        with pytest.raises(duckdb.ParserException, match=re.escape("LINE 1: SELECT 1; SELEC 2")):
            run(con, "SELECT 1; SELEC 2")

    @pytest.mark.parametrize("run", [runner("execute"), runner("sql"), runner("executemany")])
    def test_a_parse_error_follows_errors_as_json(self, con, run):
        con.execute("SET errors_as_json = true")
        with pytest.raises(duckdb.ParserException) as error:
            run(con, "SELECT 1; SELEC 2")
        details = json.loads(str(error.value).split("Parser Error: ", 1)[1])
        assert details["error_subtype"] == "SYNTAX_ERROR"
        assert details["position"] == "10"

    def test_an_unterminated_string_anywhere_stops_the_whole_script(self, con):
        # The engine tokenizes the whole string before it parses the first statement.
        with pytest.raises(duckdb.ParserException, match="unterminated"):
            con.execute("CREATE TABLE before (i INT); SELECT 'unterminated")
        assert tables(con) == []


class TestTheLastStatement:
    def test_parameters_bind_to_the_last_statement(self, con):
        con.execute("CREATE TABLE p (i INT); INSERT INTO p VALUES (?)", [7])
        assert con.execute("SELECT i FROM p").fetchall() == [(7,)]

    def test_a_syntax_error_after_a_placeholder_wins_over_the_refusal(self, con):
        with pytest.raises(duckdb.ParserException, match="syntax error"):
            con.execute("SELECT ?; SELEC 2", [1])

    def test_a_placeholder_before_the_last_statement_errors_after_the_statements_in_front_of_it(self, con):
        with pytest.raises(duckdb.NotImplementedException, match="only supported for the last statement"):
            con.execute("CREATE TABLE p (i INT); INSERT INTO p VALUES (?); SELECT 1", [7])
        assert con.execute("SELECT count(*) FROM p").fetchall() == [(0,)]

    @pytest.mark.parametrize("run", WITH_PARAMETERS)
    def test_parameters_without_a_statement_to_take_them_are_refused(self, con, tmp_path, run):
        empty = (tmp_path / "empty").as_posix()
        con.execute(f"EXPORT DATABASE '{empty}'")
        with pytest.raises(duckdb.InvalidInputException, match=NO_STATEMENT_TAKES_PARAMETERS):
            run(con, f"SELECT 42; IMPORT DATABASE '{empty}'", [1])

    @pytest.mark.parametrize("run", WITH_PARAMETERS)
    def test_parameters_for_a_query_without_statements_are_refused(self, con, run):
        with pytest.raises(duckdb.InvalidInputException, match=NO_STATEMENT_TAKES_PARAMETERS):
            run(con, "", [1])

    @pytest.mark.parametrize("run", WITH_PARAMETERS)
    def test_parameters_for_a_statement_that_expands_into_several_are_refused_by_the_engine(self, con, run):
        con.execute("CREATE TABLE p AS SELECT * FROM (VALUES ('a', 1), ('b', 2), ('a', 3)) t(k, v)")
        with pytest.raises(
            duckdb.InvalidInputException, match="not supported for a statement that expands into several"
        ):
            run(con, "PIVOT p ON k USING sum(v + ?)", [1])

    def test_executemany_parameters_for_an_expanding_statement_without_placeholders_are_refused_by_the_engine(
        self, con
    ):
        con.execute("CREATE TABLE p AS SELECT * FROM (VALUES ('a', 1), ('b', 2), ('a', 3)) t(k, v)")
        with pytest.raises(
            duckdb.InvalidInputException, match="not supported for a statement that expands into several"
        ):
            con.executemany("PIVOT p ON k USING sum(v)", [[1]])

    @pytest.mark.parametrize("run", WITH_PARAMETERS)
    def test_a_query_without_statements_or_parameters_returns_nothing(self, con, run):
        assert run(con, "", []) is None

    @pytest.mark.parametrize("run", [runner("execute"), runner("executemany")])
    def test_a_statement_that_expands_to_nothing_returns_an_empty_result(self, con, tmp_path, run):
        # The engine expands a statement when it is submitted, so one that expands to no statements still ran
        empty = (tmp_path / "empty").as_posix()
        con.execute(f"EXPORT DATABASE '{empty}'")
        assert run(con, f"IMPORT DATABASE '{empty}'") is con
        assert con.description == []
        assert con.rowcount == -1
        assert con.fetchall() == []

    def test_sql_of_a_statement_that_expands_to_nothing_returns_no_relation(self, con, tmp_path):
        empty = (tmp_path / "empty").as_posix()
        con.execute(f"EXPORT DATABASE '{empty}'")
        assert con.sql(f"IMPORT DATABASE '{empty}'") is None

    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(lambda con: con.execute("", BrokenLength()), id="execute"),
            pytest.param(lambda con: con.sql("SELECT 1", params=BrokenLength()), id="sql"),
            pytest.param(lambda con: con.executemany("", [BrokenLength()]), id="executemany"),
        ],
    )
    def test_an_error_reading_the_parameters_length_propagates(self, con, run):
        with pytest.raises(ValueError, match="no length today"):
            run(con)

    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(lambda con: con.execute("CREATE TABLE t (i INT); SELECT 1", BrokenLength()), id="execute"),
            pytest.param(lambda con: con.sql("CREATE TABLE t (i INT); SELECT 1", params=BrokenLength()), id="sql"),
            pytest.param(
                lambda con: con.executemany("CREATE TABLE t (i INT); SELECT ?", [BrokenLength()]), id="executemany"
            ),
        ],
    )
    def test_the_parameters_are_read_before_anything_runs(self, con, run):
        with pytest.raises(ValueError, match="no length today"):
            run(con)
        assert tables(con) == []

    def test_executemany_reads_every_parameter_set_before_anything_runs(self, con):
        with pytest.raises(ValueError, match="no length today"):
            con.executemany("CREATE TABLE x (i INT); INSERT INTO x VALUES (?)", [[1], BrokenLengthSet([2])])
        assert tables(con) == []

    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(lambda con: con.execute("", RefusedLength()), id="execute"),
            pytest.param(lambda con: con.sql("SELECT 1", params=RefusedLength()), id="sql"),
            pytest.param(lambda con: con.executemany("", [RefusedLength()]), id="executemany"),
        ],
    )
    def test_a_type_error_reading_the_parameters_length_propagates(self, con, run):
        with pytest.raises(TypeError, match="length refused"):
            run(con)

    @pytest.mark.parametrize(
        ("script", "parameter_sets", "error"),
        [
            pytest.param("CREATE TABLE t (i INT); INSERT INTO t VALUES (?)", 42, "requires a list", id="not_a_list"),
            pytest.param(
                "CREATE TABLE t (i INT); INSERT INTO t VALUES (?)", [], "requires a non-empty list", id="empty_list"
            ),
        ],
    )
    def test_executemany_refuses_malformed_parameter_sets_before_running_anything(
        self, con, script, parameter_sets, error
    ):
        with pytest.raises(duckdb.InvalidInputException, match=error):
            con.executemany(script, parameter_sets)
        assert tables(con) == []

    @pytest.mark.parametrize("parameter_sets", [pytest.param([], id="empty_list"), pytest.param(None, id="absent")])
    def test_executemany_without_statements_returns_nothing_for_no_parameter_sets(self, con, parameter_sets):
        assert con.executemany("", parameter_sets) is None

    def test_executemany_checks_the_parameter_list_of_a_statement_that_expands_to_nothing(self, con, tmp_path):
        # The first statement is known as parsed, before it expands, so the list is checked as for any statement
        empty = (tmp_path / "empty").as_posix()
        con.execute(f"EXPORT DATABASE '{empty}'")
        with pytest.raises(duckdb.InvalidInputException, match="requires a non-empty list"):
            con.executemany(f"IMPORT DATABASE '{empty}'", [])

    def test_executemany_runs_a_statement_before_one_that_expands_to_nothing_once(self, con, tmp_path):
        empty = (tmp_path / "empty").as_posix()
        con.execute(f"EXPORT DATABASE '{empty}'")
        con.execute("CREATE TABLE t (i INT)")
        con.executemany(f"INSERT INTO t VALUES (5); IMPORT DATABASE '{empty}'", [[], [], []])
        # The parameterized statement is the import, which changes no rows
        assert con.rowcount == -1
        assert con.execute("SELECT count(*) FROM t").fetchall() == [(1,)]

    @pytest.mark.parametrize("run", WITH_PARAMETERS)
    def test_no_parameters_without_a_statement_to_take_them_leave_the_result(self, con, tmp_path, run):
        empty = (tmp_path / "empty").as_posix()
        con.execute(f"EXPORT DATABASE '{empty}'")
        assert run(con, f"SELECT 42; IMPORT DATABASE '{empty}'", []).fetchall() == [(42,)]

    @pytest.mark.parametrize("tail", ["", ";;", " /* a comment */", " /* a comment */ ;"])
    @pytest.mark.parametrize("run", [runner("execute"), runner("sql")])
    def test_text_without_a_statement_after_the_last_leaves_its_result(self, con, run, tail):
        assert run(con, "SELECT 1; SELECT 42;" + tail).fetchall() == [(42,)]

    @pytest.mark.parametrize("tail", ["", ";;", " /* a comment */", " /* a comment */ ;"])
    def test_text_without_a_statement_after_the_last_keeps_it_the_last(self, con, tail):
        con.execute("CREATE TABLE t (i INT)")
        relation = con.sql("SELECT 1; SELECT count(*) FROM t;" + tail)
        con.execute("INSERT INTO t VALUES (1)")
        # A relation runs when it is read, so it sees the row inserted after sql() returned
        assert relation.fetchall() == [(1,)]
        assert con.execute("SELECT 1; SELECT ? + 1;" + tail, [41]).fetchall() == [(42,)]

    @pytest.mark.parametrize("run", [runner("execute"), runner("sql")])
    def test_a_trailing_statement_that_expands_to_nothing_leaves_the_result_before_it(self, con, tmp_path, run):
        empty = (tmp_path / "empty").as_posix()
        con.execute(f"EXPORT DATABASE '{empty}'")
        assert run(con, f"SELECT 42; IMPORT DATABASE '{empty}'").fetchall() == [(42,)]

    def test_executemany_runs_the_statements_before_the_last_once(self, con):
        con.executemany("CREATE TABLE em (i INT); INSERT INTO em VALUES (?)", [[1], [2], [3]])
        assert con.rowcount == 3
        assert con.execute("SELECT i FROM em ORDER BY i").fetchall() == [(1,), (2,), (3,)]

    def test_sql_runs_the_statements_before_the_last_and_keeps_the_last_lazy(self, con):
        relation = con.sql("CREATE TABLE h (i INT); INSERT INTO h VALUES (1), (2); SELECT sum(i) FROM h")
        con.execute("INSERT INTO h VALUES (3)")
        assert relation.fetchall() == [(6,)]


class TestTheGroupResult:
    @pytest.mark.parametrize(
        ("script", "rows"),
        [
            pytest.param("SELECT 42; CREATE TABLE t (i INT)", [(42,)], id="create_after"),
            pytest.param("SELECT 1; INSERT INTO src VALUES (1), (2)", [(1,)], id="insert_after"),
            pytest.param("SELECT 42; SET threads = 2", [(42,)], id="set_after"),
            pytest.param("SELECT 42; COPY src TO '{out}'", [(42,)], id="copy_after"),
            pytest.param("INSERT INTO src VALUES (1); SELECT 7; CREATE TABLE t (i INT)", [(7,)], id="between"),
            pytest.param("CREATE TABLE t (i INT); SELECT 42", [(42,)], id="last"),
            pytest.param("SELECT 1; SELECT 2", [(2,)], id="both"),
        ],
    )
    @pytest.mark.parametrize("run", [runner("execute"), runner("sql")])
    def test_the_last_statement_that_returns_rows_gives_the_result(self, con, tmp_path, run, script, rows):
        con.execute("CREATE TABLE src (i INT)")
        assert run(con, script.format(out=(tmp_path / "out.csv").as_posix())).fetchall() == rows

    @pytest.mark.parametrize("before", [pytest.param("", id="alone"), pytest.param("SELECT 1; ", id="after_rows")])
    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(lambda con, script: con.execute(script), id="execute"),
            pytest.param(lambda con, script: con.executemany(script, [[]]), id="executemany"),
        ],
    )
    def test_a_statement_that_expands_into_several_gives_its_rows_as_the_result(self, con, run, before):
        # Its result is known only once it ran, so the rows before it cannot be released when it is submitted
        con.execute("CREATE TABLE p AS SELECT * FROM (VALUES ('a', 1), ('b', 2), ('a', 3)) t(k, v)")
        run(con, before + "PIVOT p ON k USING sum(v)")
        assert [column[0] for column in con.description] == ["a", "b"]
        assert con.fetchall() == [(4, 2)]

    @pytest.mark.parametrize("before", [pytest.param("", id="alone"), pytest.param("SELECT 1; ", id="after_rows")])
    def test_sql_of_a_statement_that_expands_into_several_gives_its_rows(self, con, before):
        con.execute("CREATE TABLE p AS SELECT * FROM (VALUES ('a', 1), ('b', 2), ('a', 3)) t(k, v)")
        relation = con.sql(before + "PIVOT p ON k USING sum(v)")
        assert relation.columns == ["a", "b"]
        assert relation.fetchall() == [(4, 2)]

    def test_sql_of_a_statement_that_expands_into_several_runs_when_sql_returns(self, con):
        # The engine hands sql() the statement as parsed, and only a SELECT becomes a lazy relation
        con.execute("CREATE TABLE p AS SELECT * FROM (VALUES ('a', 1), ('b', 2), ('a', 3)) t(k, v)")
        relation = con.sql("PIVOT p ON k USING sum(v)")
        con.execute("INSERT INTO p VALUES ('b', 10)")
        assert relation.fetchall() == [(4, 2)]

    def test_sql_of_a_pragma_runs_when_sql_returns(self, con):
        con.execute("CREATE TABLE t (a INTEGER)")
        relation = con.sql("PRAGMA table_info('t')")
        con.execute("ALTER TABLE t ADD COLUMN b INTEGER")
        assert [row[1] for row in relation.fetchall()] == ["a"]

    def test_rows_before_an_insert_come_without_its_rowcount(self, con):
        con.execute("CREATE TABLE src (i INT)")
        con.execute("SELECT 1; INSERT INTO src VALUES (1), (2)")
        assert con.rowcount == -1
        assert con.fetchall() == [(1,)]
        assert con.execute("SELECT count(*) FROM src").fetchall() == [(2,)]

    @pytest.mark.parametrize(
        ("script", "rowcount", "rows"),
        [
            pytest.param("INSERT INTO src VALUES (1), (2); CREATE TABLE t (i INT)", -1, [], id="create_last"),
            pytest.param("BEGIN; INSERT INTO src VALUES (1), (2); COMMIT", -1, [], id="commit_last"),
            pytest.param("CREATE TABLE t (i INT); INSERT INTO src VALUES (1), (2)", 2, [(2,)], id="insert_last"),
        ],
    )
    def test_without_rows_the_last_statement_gives_the_result(self, con, script, rowcount, rows):
        con.execute("CREATE TABLE src (i INT)")
        con.execute(script)
        assert con.rowcount == rowcount
        assert con.fetchall() == rows

    def test_sql_without_rows_returns_a_relation_over_the_last_result(self, con):
        con.execute("CREATE TABLE src (i INT)")
        rel = con.sql("INSERT INTO src VALUES (1), (2); CREATE TABLE t (i INT)")
        assert tables(con) == ["src", "t"]
        assert (rel.columns, rel.fetchall()) == (["Count"], [])
        assert con.execute("SELECT count(*) FROM src").fetchall() == [(2,)]

    def test_an_import_returns_nothing_of_the_statements_it_expands_to(self, con, tmp_path):
        export = (tmp_path / "export").as_posix()
        con.execute("CREATE TABLE first_table AS SELECT range AS i FROM range(5)")
        con.execute("CREATE TABLE second_table AS SELECT range AS i FROM range(3)")
        con.execute(f"EXPORT DATABASE '{export}'")
        importing = duckdb.connect()
        importing.execute(f"IMPORT DATABASE '{export}'")
        assert importing.rowcount == -1
        assert importing.fetchall() == []
        assert importing.execute("SELECT count(*) FROM second_table").fetchall() == [(3,)]

    def test_a_trailing_statement_that_expands_to_nothing_leaves_a_result_without_rows(self, con, tmp_path):
        empty = (tmp_path / "empty").as_posix()
        con.execute(f"EXPORT DATABASE '{empty}'")
        con.execute(f"CREATE TABLE t (i INT); IMPORT DATABASE '{empty}'")
        assert con.description == []
        assert con.fetchall() == []

    def test_executemany_counts_the_rows_its_parameterized_statement_changed(self, con):
        con.execute("CREATE TABLE t (i INT)")
        con.executemany("SELECT 42; INSERT INTO t VALUES (?)", [[1], [2], [3]])
        assert con.rowcount == 3
        assert con.fetchall() == [(42,)]

    def test_sql_materializes_the_rows_of_a_statement_before_one_that_expands_to_nothing(self, con, tmp_path):
        empty = (tmp_path / "empty").as_posix()
        con.execute(f"EXPORT DATABASE '{empty}'")
        con.execute("CREATE TABLE h AS SELECT 1 AS i")
        relation = con.sql(f"SELECT sum(i) FROM h; IMPORT DATABASE '{empty}'")
        con.execute("INSERT INTO h VALUES (2)")
        assert relation.fetchall() == [(1,)]

    def test_executemany_returns_earlier_rows_when_its_last_statement_returns_none(self, con):
        con.executemany("SELECT 42; CREATE TABLE IF NOT EXISTS t (i INT)", [[], []])
        assert con.rowcount == -1
        assert con.fetchall() == [(42,)]


class TestAnErrorInAnExplicitTransaction:
    @pytest.mark.parametrize(
        ("failing", "error"),
        [
            pytest.param("SELEC 1", duckdb.ParserException, id="syntax"),
            pytest.param("SELECT 'unterminated", duckdb.ParserException, id="tokenizer"),
            pytest.param("PRAGMA import_database('{missing}')", duckdb.IOException, id="expansion"),
        ],
    )
    @pytest.mark.parametrize(
        "before", [pytest.param("", id="alone"), pytest.param("CREATE TABLE t (i INT); ", id="after_a_statement")]
    )
    @pytest.mark.parametrize("run", [runner("execute"), runner("sql")])
    def test_a_statement_that_fails_to_parse_aborts_the_transaction(self, con, tmp_path, run, before, failing, error):
        con.execute("BEGIN")
        with pytest.raises(error):
            run(con, before + failing.format(missing=(tmp_path / "missing").as_posix()))
        with pytest.raises(duckdb.TransactionException, match=ABORTED):
            con.execute("SELECT 42")
        con.execute("ROLLBACK")
        assert tables(con) == []

    def test_a_placeholder_before_the_last_statement_aborts_the_transaction(self, con):
        con.execute("CREATE TABLE p (i INT)")
        con.execute("BEGIN")
        with pytest.raises(duckdb.NotImplementedException, match="only supported for the last statement"):
            con.execute("INSERT INTO p VALUES (1); INSERT INTO p VALUES (?); SELECT 1", [7])
        with pytest.raises(duckdb.TransactionException, match=ABORTED):
            con.execute("SELECT 42")
        con.execute("ROLLBACK")
        assert con.execute("SELECT count(*) FROM p").fetchall() == [(0,)]

    def test_outside_a_transaction_a_parse_error_leaves_the_connection_usable(self, con):
        with pytest.raises(duckdb.ParserException):
            con.execute("SELEC 1")
        assert con.execute("SELECT 42").fetchall() == [(42,)]

    def test_parameters_without_a_statement_to_take_them_abort_the_transaction(self, con, tmp_path):
        empty = (tmp_path / "empty").as_posix()
        con.execute(f"EXPORT DATABASE '{empty}'")
        con.execute("BEGIN")
        with pytest.raises(duckdb.InvalidInputException, match=NO_STATEMENT_TAKES_PARAMETERS):
            con.execute(f"SELECT 42; IMPORT DATABASE '{empty}'", [1])
        with pytest.raises(duckdb.TransactionException, match=ABORTED):
            con.execute("SELECT 42")
        con.execute("ROLLBACK")

    def test_extract_statements_leaves_the_transaction_valid(self, con):
        con.execute("BEGIN")
        with pytest.raises(duckdb.ParserException):
            con.extract_statements("SELEC 1")
        assert con.execute("SELECT 42").fetchall() == [(42,)]


class TestAnOpenResult:
    @pytest.fixture
    def open_result(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        late = f"CASE WHEN i = {ROWS - 1} THEN error('late')::BIGINT ELSE 0 END"
        con.execute(f"SELECT nextval('s') + {late} FROM range({ROWS}) t(i)")
        con.fetchone()
        return con

    def test_a_script_that_starts_with_rollback_cancels_the_open_result(self, open_result):
        assert open_result.execute("ROLLBACK; SELECT 42").fetchall() == [(42,)]
        (advanced,) = open_result.execute("SELECT currval('s')").fetchone()
        assert advanced < ROWS

    def test_a_rollback_after_the_first_statement_drains_the_open_result_first(self, open_result):
        with pytest.raises(duckdb.InvalidInputException, match="late"):
            open_result.execute("SELECT 1; ROLLBACK")

    @pytest.mark.parametrize("script", ["", ";", "/* only a comment */"])
    def test_a_script_without_statements_ends_the_open_result_and_returns_nothing(self, con, script):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        assert con.execute(script) is None
        with pytest.raises(duckdb.InvalidInputException, match="No open result set"):
            con.fetchone()

    def test_a_parse_error_in_the_first_statement_leaves_the_open_result(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.ParserException):
            con.execute("SELEC 1; CREATE TABLE t (i INT)")
        assert con.fetchone() == (1,)
        assert tables(con) == []

    def test_a_parse_error_in_a_later_statement_comes_after_the_open_result_made_way(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.ParserException):
            con.execute("CREATE TABLE t (i INT); SELEC 1")
        with pytest.raises(duckdb.InvalidInputException, match="No open result set"):
            con.fetchone()
        assert tables(con) == ["t"]

    def test_an_expansion_error_in_the_first_statement_comes_after_the_open_result_made_way(self, con, tmp_path):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.IOException):
            con.execute(f"PRAGMA import_database('{(tmp_path / 'missing').as_posix()}')")
        with pytest.raises(duckdb.InvalidInputException, match="No open result set"):
            con.fetchone()

    def test_an_import_runs_in_its_own_transaction_once_the_open_result_made_way(self, con, tmp_path):
        export = (tmp_path / "export").as_posix()
        source = duckdb.connect()
        source.execute("CREATE TABLE a AS SELECT 1 AS i")
        source.execute("CREATE TABLE b AS SELECT 2 AS i")
        source.execute(f"EXPORT DATABASE '{export}'")
        con.execute("CREATE TABLE b (i INT)")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.CatalogException, match="already exists"):
            con.execute(f"IMPORT DATABASE '{export}'")
        assert tables(con) == ["b"]
        policy = con.execute("SELECT current_setting('current_transaction_invalidation_policy')").fetchall()
        assert policy == [("STANDARD_POLICY",)]

    def test_a_statement_the_parser_splits_rolls_back_whole_once_the_open_result_made_way(self, con):
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        # The parser splits this into adding the column and adding its constraint, which the existing row violates
        with pytest.raises(duckdb.ConstraintException):
            con.execute("ALTER TABLE t ADD COLUMN b INT NOT NULL")
        assert [row[0] for row in con.execute("DESCRIBE t").fetchall()] == ["a"]

    def test_sql_refuses_the_statements_before_the_last_while_execute_holds_a_result(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.ResourceInUseException, match=UNFINISHED_EXECUTE):
            con.sql("CREATE TABLE w (i INT); SELECT 1")
        assert con.fetchone() == (1,)
        assert tables(con) == []


class TestExtractStatements:
    def test_extract_statements_returns_statements_as_parsed_and_runs_nothing(self, con):
        statements = con.extract_statements("CREATE TABLE x (i INT); PRAGMA table_info('x')")
        assert [s.type for s in statements] == [duckdb.StatementType.CREATE, duckdb.StatementType.PRAGMA]
        assert tables(con) == []

    def test_extract_statements_reads_no_export_a_pragma_names(self, con, tmp_path):
        missing = (tmp_path / "missing").as_posix()
        script = f"EXPORT DATABASE '{missing}'; DROP TABLE integers; PRAGMA import_database('{missing}')"
        statements = con.extract_statements(script)
        assert [s.type for s in statements] == [
            duckdb.StatementType.EXPORT,
            duckdb.StatementType.DROP,
            duckdb.StatementType.PRAGMA,
        ]

"""What con.sql() returns for each kind of statement, whether it ran at the call, and whether a read runs it again."""

import pytest

import duckdb


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute("CREATE SEQUENCE s")
    connection.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a', 10), (2, 'b', 20), (3, 'a', 30)) v(i, k, v)")
    connection.execute("CREATE TABLE log (v BIGINT)")
    connection.execute("CREATE MACRO counted() AS TABLE SELECT nextval('s') AS v")
    connection.execute("PREPARE counting AS SELECT nextval('s') AS v")
    connection.execute("SET VARIABLE x = 1")
    yield connection
    connection.close()


def sequence_used(con):
    try:
        con.execute("SELECT currval('s')").fetchall()
    except duckdb.SequenceException:
        return False
    return True


def currval(con):
    return con.execute("SELECT currval('s')").fetchall()[0][0]


def executed(con, sql):
    cursor = con.execute(sql)
    names = [column[0] for column in cursor.description]
    return names, cursor.fetchall()


def rows(result):
    return sorted(result, key=repr)


class TestReadsAreLazy:
    @pytest.fixture
    def ticks(self, con):
        calls = []

        def tick() -> int:
            calls.append(1)
            return len(calls)

        con.create_function("tick", tick, [], "BIGINT", side_effects=True)
        con.execute("CREATE MACRO ticked() AS TABLE SELECT tick() AS v")
        return calls

    @pytest.mark.parametrize(
        "sql",
        [
            pytest.param("SELECT tick() AS v", id="select"),
            pytest.param("FROM range(1) SELECT tick() AS v", id="from_first"),
            pytest.param("VALUES (tick())", id="values"),
            pytest.param("WITH c AS (SELECT tick() AS v) SELECT * FROM c", id="cte"),
            pytest.param("SELECT * FROM (SELECT tick() AS v) USING SAMPLE 100%", id="sample"),
            pytest.param("FROM ticked()", id="table_macro"),
        ],
    )
    def test_the_call_runs_nothing_and_every_read_runs_it(self, con, ticks, sql):
        # The observer is a Python function, which the engine does not count as a write.
        rel = con.sql(sql)
        assert len(ticks) == 0
        rel.fetchall()
        first = len(ticks)
        assert first > 0
        rel.fetchall()
        assert len(ticks) == 2 * first

    @pytest.mark.parametrize(
        "sql",
        [
            pytest.param("SELECT nextval('s') AS v", id="select"),
            pytest.param("WITH c AS (SELECT nextval('s') AS v) SELECT * FROM c", id="cte"),
        ],
    )
    def test_a_select_the_engine_says_writes_runs_once_at_the_call(self, con, sql):
        # Advancing a sequence modifies the database, so the engine reports the statement as a write.
        rel = con.sql(sql)
        assert sequence_used(con)
        after_the_call = currval(con)
        assert rel.fetchall() == rel.fetchall()
        assert currval(con) == after_the_call

    @pytest.mark.parametrize(
        ("sql", "change"),
        [
            pytest.param("DESCRIBE t", "ALTER TABLE t ADD COLUMN z INTEGER", id="describe"),
            pytest.param("SHOW TABLES", "CREATE TABLE u (a INTEGER)", id="show_tables"),
            pytest.param("SUMMARIZE t", "INSERT INTO t VALUES (4, 'c', 40)", id="summarize"),
            pytest.param("PIVOT t ON k IN ('a', 'b') USING sum(v)", "INSERT INTO t VALUES (4, 'b', 40)", id="pivot_in"),
            pytest.param("UNPIVOT t ON v INTO NAME n VALUE val", "INSERT INTO t VALUES (4, 'c', 40)", id="unpivot"),
            pytest.param("SELECT getvariable('x') AS x", "SET VARIABLE x = 2", id="getvariable"),
            pytest.param("SELECT table_name FROM duckdb_tables()", "CREATE TABLE u (a INTEGER)", id="table_function"),
        ],
    )
    def test_a_read_sees_the_catalog_and_rows_of_the_read(self, con, sql, change):
        rel = con.sql(sql)
        con.execute(change)
        names, expected = executed(con, sql)
        assert rel.columns == names
        assert rows(rel.fetchall()) == rows(expected)

    def test_a_file_read_sees_the_file_of_the_read(self, con, tmp_path):
        path = tmp_path / "in.csv"
        path.write_text("i,s\n1,a\n")
        rel = con.sql(f"SELECT * FROM read_csv('{path}')")
        path.write_text("i,s\n1,a\n2,b\n")
        assert rel.columns == ["i", "s"]
        assert rel.fetchall() == [(1, "a"), (2, "b")]

    def test_statements_before_a_last_read_run_at_the_call_and_the_read_stays_lazy(self, con):
        rel = con.sql("INSERT INTO log VALUES (1); SELECT count(*) AS n FROM log")
        assert con.execute("SELECT count(*) FROM log").fetchall() == [(1,)]
        con.execute("INSERT INTO log VALUES (2)")
        assert rel.fetchall() == [(2,)]
        assert con.execute("SELECT count(*) FROM log").fetchall() == [(2,)]


class TestOtherStatementsWithRowsRunAtTheCall:
    """Only a parsed SELECT is lazy; EXECUTE, EXPLAIN, CALL and PRAGMA run at the call and keep what they returned."""

    @pytest.mark.parametrize(
        "sql",
        [
            pytest.param("EXECUTE counting", id="execute"),
            pytest.param("EXPLAIN ANALYZE SELECT nextval('s')", id="explain_analyze"),
        ],
    )
    def test_the_call_runs_it_and_a_read_does_not(self, con, sql):
        rel = con.sql(sql)
        assert sequence_used(con)
        after_the_call = currval(con)
        first = rel.fetchall()
        assert rel.fetchall() == first
        assert currval(con) == after_the_call

    @pytest.mark.parametrize(
        ("sql", "change"),
        [
            pytest.param("CALL duckdb_tables()", "CREATE TABLE u (a INTEGER)", id="call"),
            pytest.param("PRAGMA show_tables", "CREATE TABLE u (a INTEGER)", id="pragma_show_tables"),
            pytest.param("PRAGMA table_info('t')", "ALTER TABLE t ADD COLUMN z INTEGER", id="pragma_table_info"),
        ],
    )
    def test_a_read_keeps_the_rows_of_the_call(self, con, sql, change):
        names, expected = executed(con, sql)
        rel = con.sql(sql)
        con.execute(change)
        assert rel.columns == names
        assert rows(rel.fetchall()) == rows(expected)

    def test_explain_returns_its_rows(self, con):
        rel = con.sql("EXPLAIN SELECT 1")
        names, expected = executed(con, "EXPLAIN SELECT 1")
        assert rel.columns == names == ["explain_key", "explain_value"]
        assert len(rel.fetchall()) == len(expected) > 0

    def test_checkpoint_runs_at_the_call(self, con):
        rel = con.sql("CHECKPOINT")
        assert (rel.columns, rel.fetchall()) == executed(con, "CHECKPOINT")


class TestAWriteKeywordThatIsNoWrite:
    """Only a common table expression body can write inside a SELECT, so a keyword elsewhere leaves it lazy."""

    @pytest.fixture
    def worded(self, con):
        con.execute('CREATE TABLE worded ("insert" INTEGER, "delete" INTEGER)')
        con.execute("INSERT INTO worded VALUES (1, 2)")
        return con

    @pytest.mark.parametrize(
        ("sql", "after_insert"),
        [
            pytest.param("SELECT coalesce(insert, 0) AS v FROM worded", [(1,), (3,)], id="function_argument"),
            pytest.param("SELECT (insert + 1) AS v FROM worded", [(2,), (4,)], id="inside_parentheses"),
            pytest.param('SELECT "delete", insert FROM worded', [(2, 1), (4, 3)], id="column_after_a_comma"),
        ],
    )
    def test_the_select_stays_lazy(self, worded, sql, after_insert):
        rel = worded.sql(sql)
        assert rel.type == "QUERY_RELATION"
        worded.execute("INSERT INTO worded VALUES (3, 4)")
        assert sorted(rel.fetchall()) == after_insert

    def test_a_writing_common_table_expression_still_runs_once(self, worded):
        rel = worded.sql('WITH w AS (INSERT INTO worded VALUES (5, 6) RETURNING "insert") SELECT * FROM w')
        assert worded.execute("SELECT count(*) FROM worded").fetchall() == [(2,)]
        assert rel.fetchall() == [(5,)]
        assert rel.fetchall() == [(5,)]
        assert worded.execute("SELECT count(*) FROM worded").fetchall() == [(2,)]


class TestADataModifyingCommonTableExpression:
    """A SELECT that writes through a common table expression runs at the call, so a read never writes again."""

    def test_it_runs_at_the_call_and_a_read_does_not_write_again(self, con):
        rel = con.sql("WITH w AS (INSERT INTO log VALUES (1), (2) RETURNING v) SELECT v FROM w ORDER BY v")
        assert con.execute("SELECT count(*) FROM log").fetchall() == [(2,)]
        assert rel.fetchall() == [(1,), (2,)]
        assert rel.fetchall() == [(1,), (2,)]
        assert con.execute("SELECT count(*) FROM log").fetchall() == [(2,)]

    def test_a_step_over_it_reads_the_rows_it_returned(self, con):
        rel = con.sql("WITH w AS (INSERT INTO log VALUES (1), (2) RETURNING v) SELECT v FROM w")
        assert rel.filter("v > 1").fetchall() == [(2,)]
        assert con.execute("SELECT count(*) FROM log").fetchall() == [(2,)]


class TestGroupsRunAtTheCall:
    @pytest.mark.parametrize(
        "sql",
        [
            pytest.param("PIVOT t ON k USING sum(v)", id="pivot"),
            pytest.param("SELECT * FROM (PIVOT t ON k USING sum(v)) ORDER BY ALL", id="pivot_in_subquery"),
        ],
    )
    def test_a_dynamic_pivot_keeps_the_rows_of_the_call(self, con, sql):
        names, expected = executed(con, sql)
        rel = con.sql(sql)
        con.execute("INSERT INTO t VALUES (4, 'c', 40)")
        assert rel.columns == names
        assert rows(rel.fetchall()) == rows(expected)
        assert rows(rel.fetchall()) == rows(expected)

    def test_a_dynamic_pivot_runs_once(self, con):
        rel = con.sql("PIVOT (SELECT k, nextval('s') AS v FROM t) ON k USING sum(v)")
        after_the_call = currval(con)
        rel.fetchall()
        rel.fetchall()
        assert currval(con) == after_the_call

    @pytest.mark.parametrize(
        ("sql", "twin", "check"),
        [
            pytest.param(
                "CREATE TABLE p AS PIVOT t ON k USING sum(v)",
                "CREATE TABLE p2 AS PIVOT t ON k USING sum(v)",
                "SELECT count(*) FROM p",
                id="create_table_as_pivot",
            ),
            pytest.param(
                "ALTER TABLE t ADD COLUMN r DOUBLE DEFAULT random()",
                "ALTER TABLE log ADD COLUMN r DOUBLE DEFAULT random()",
                "SELECT count(r) FROM t",
                id="alter_with_a_volatile_default",
            ),
        ],
    )
    def test_a_statement_group_without_rows_runs_at_the_call(self, con, sql, twin, check):
        rel = con.sql(sql)
        assert rel is not None
        con.execute(check).fetchall()
        assert (rel.columns, rel.fetchall()) == executed(con, twin)

    def test_import_database_runs_at_the_call(self, con, tmp_path):
        con.execute(f"EXPORT DATABASE '{tmp_path}'")
        target = duckdb.connect()
        twin = duckdb.connect()
        rel = target.sql(f"IMPORT DATABASE '{tmp_path}'")
        assert rel is not None
        assert target.execute("SELECT count(*) FROM t").fetchall() == [(3,)]
        assert (rel.columns, rel.fetchall()) == executed(twin, f"IMPORT DATABASE '{tmp_path}'")


class TestStatementsWithoutRowsRunAtTheCall:
    @pytest.mark.parametrize(
        ("sql", "twin", "check"),
        [
            pytest.param(
                "CREATE TABLE a (i INTEGER)",
                "CREATE TABLE b (i INTEGER)",
                "FROM a",
                id="create_table",
            ),
            pytest.param(
                "CREATE TABLE a AS FROM t",
                "CREATE TABLE b AS FROM t",
                "FROM a",
                id="create_table_as",
            ),
            pytest.param("CREATE VIEW a AS FROM t", "CREATE VIEW b AS FROM t", "FROM a", id="create_view"),
            pytest.param(
                "CREATE SEQUENCE a",
                "CREATE SEQUENCE b",
                "SELECT nextval('a')",
                id="create_sequence",
            ),
            pytest.param(
                "CREATE MACRO a(x) AS x + 1",
                "CREATE MACRO b(x) AS x + 1",
                "SELECT a(1)",
                id="create_macro",
            ),
            pytest.param(
                "INSERT INTO log VALUES (1), (2)",
                "INSERT INTO log VALUES (3), (4)",
                "FROM log",
                id="insert",
            ),
            pytest.param(
                "UPDATE t SET v = v + 1 WHERE k = 'a'",
                "UPDATE t SET v = v WHERE k = 'a'",
                "FROM t",
                id="update",
            ),
            pytest.param("DROP TABLE log", "DROP TABLE IF EXISTS nope", "FROM t", id="drop"),
            pytest.param(
                "SET threads = 2",
                "SET threads = 2",
                "SELECT current_setting('threads')",
                id="set",
            ),
            pytest.param(
                "COMMENT ON TABLE t IS 'c'",
                "COMMENT ON TABLE log IS 'c'",
                "FROM t",
                id="comment",
            ),
            pytest.param("CHECKPOINT", "CHECKPOINT", "FROM t", id="checkpoint"),
        ],
    )
    def test_the_relation_holds_what_the_statement_returned(self, con, sql, twin, check):
        rel = con.sql(sql)
        assert rel is not None
        con.execute(check).fetchall()
        assert (rel.columns, rel.fetchall()) == executed(con, twin)

    def test_a_delete_reports_the_rows_it_removed_and_removes_them_once(self, con):
        rel = con.sql("DELETE FROM t WHERE k = 'b'")
        assert rel is not None
        assert rel.columns == ["Count"]
        assert rel.fetchall() == [(1,)]
        assert rel.fetchall() == [(1,)]
        assert sorted(con.execute("SELECT i FROM t").fetchall()) == [(1,), (3,)]

    def test_a_write_runs_once(self, con):
        rel = con.sql("INSERT INTO log VALUES (nextval('s'))")
        assert rel is not None
        assert rel.fetchall() == [(1,)]
        assert rel.fetchall() == [(1,)]
        assert len(rel) == 1
        assert con.execute("FROM log").fetchall() == [(1,)]

    def test_a_parameterised_write_runs_once(self, con):
        rel = con.sql("INSERT INTO log VALUES (?)", params=[5])
        assert rel is not None
        assert rel.columns == ["Count"]
        assert rel.fetchall() == [(1,)]
        assert con.execute("FROM log").fetchall() == [(5,)]

    def test_steps_over_a_write_read_its_count(self, con):
        rel = con.sql("INSERT INTO log VALUES (1), (2), (3)")
        assert rel is not None
        assert rel.project("Count * 2 AS doubled").fetchall() == [(6,)]
        assert con.execute("SELECT count(*) FROM log").fetchall() == [(3,)]

    def test_returning_gives_the_returned_rows_once(self, con):
        rel = con.sql("INSERT INTO log VALUES (nextval('s')), (nextval('s')) RETURNING v")
        assert rel.columns == ["v"]
        assert rows(rel.fetchall()) == [(1,), (2,)]
        assert rows(rel.fetchall()) == [(1,), (2,)]
        assert currval(con) == 2
        assert con.execute("SELECT count(*) FROM log").fetchall() == [(2,)]

    def test_copy_with_return_files_gives_its_files_once(self, con, tmp_path):
        path = tmp_path / "out.parquet"
        rel = con.sql(f"COPY (SELECT 1 AS a) TO '{path}' (RETURN_FILES)")
        assert path.exists()
        path.unlink()
        assert rel.columns == ["Count", "Files"]
        assert rel.fetchall() == [(1, [str(path)])]
        assert not path.exists()

    def test_copy_without_return_files_gives_its_count(self, con, tmp_path):
        path = tmp_path / "out.csv"
        rel = con.sql(f"COPY t TO '{path}'")
        assert rel is not None
        assert path.exists()
        path.unlink()
        assert (rel.columns, rel.fetchall()) == executed(con, f"COPY t TO '{tmp_path / 'twin.csv'}'")
        assert not path.exists()


class TestScripts:
    def test_a_parse_error_inside_a_transaction_aborts_it(self, con):
        con.execute("BEGIN")
        with pytest.raises(duckdb.ParserException):
            con.sql("INSERT INTO log VALUES (1); SELEC 2")
        with pytest.raises(duckdb.TransactionException, match="aborted"):
            con.execute("SELECT count(*) FROM log").fetchall()
        con.execute("ROLLBACK")
        assert con.execute("SELECT count(*) FROM log").fetchall() == [(0,)]

    def test_parameters_before_the_last_statement_are_refused(self, con):
        with pytest.raises(duckdb.NotImplementedException, match="only supported for the last statement"):
            con.sql("SELECT ? AS p; SELECT 1 AS q", params=[1])

    def test_a_setting_an_earlier_statement_changes_reaches_the_last_statement_s_schema(self, con):
        rel = con.sql("SET integer_division = true; SELECT 7 / 2 AS d")
        assert (rel.columns, [str(kind) for kind in rel.types], rel.fetchall()) == (["d"], ["INTEGER"], [(3,)])

    def test_the_earlier_rows_are_kept_when_the_last_statement_returns_none(self, con):
        rel = con.sql("SELECT 7 AS a; CREATE TABLE later (i INTEGER)")
        assert (rel.columns, rel.fetchall()) == (["a"], [(7,)])
        assert con.execute("SELECT count(*) FROM later").fetchall() == [(0,)]

    def test_a_select_with_a_marker_and_no_values_raises_at_the_call(self, con):
        with pytest.raises(duckdb.Error):
            con.sql("SELECT ? AS a")


def transaction_state(con):
    """Whether the open transaction still runs a statement, or the class of the refusal if it does not."""
    try:
        con.execute("SELECT 1").fetchall()
    except duckdb.Error as error:
        return type(error)
    return "usable"


class TestAFailedCallInsideATransaction:
    """A failure leaves the transaction alone exactly when nothing was executed.

    Building a relation runs nothing, so its failure cannot abort the transaction; a statement that runs at the call
    executes, so its failure aborts the transaction as execution always did.
    """

    @pytest.mark.parametrize(
        ("sql", "error"),
        [
            pytest.param("SELECT * FROM nope", duckdb.CatalogException, id="lazy_select_missing_table"),
            pytest.param("SELECT nope FROM t", duckdb.BinderException, id="lazy_select_bad_column"),
        ],
    )
    def test_a_lazy_select_that_fails_leaves_the_transaction_usable(self, con, sql, error):
        con.execute("BEGIN TRANSACTION")
        with pytest.raises(error):
            con.sql(sql)
        assert transaction_state(con) == "usable"
        con.execute("COMMIT")

    @pytest.mark.parametrize(
        ("sql", "error"),
        [
            pytest.param("PRAGMA import_database('{missing}')", duckdb.IOException, id="pragma_import_database"),
            pytest.param("CALL nope()", duckdb.CatalogException, id="call_of_a_missing_function"),
            pytest.param("INSERT INTO nope VALUES (1)", duckdb.CatalogException, id="insert_into_a_missing_table"),
        ],
    )
    def test_a_statement_that_runs_at_the_call_and_fails_aborts_it(self, con, tmp_path, sql, error):
        con.execute("BEGIN TRANSACTION")
        with pytest.raises(error):
            con.sql(sql.format(missing=tmp_path / "missing"))
        assert transaction_state(con) is duckdb.TransactionException
        con.execute("ROLLBACK")
        assert transaction_state(con) == "usable"


class TestADeferredGroupReportsWhatItReturned:
    """A statement group's return type is the executed statement's, not the placeholder bind's."""

    @pytest.fixture
    def exported(self, con, tmp_path):
        con.execute(f"EXPORT DATABASE '{tmp_path / 'exported'}'")
        return tmp_path / "exported"

    def test_import_database_reports_no_result(self, exported):
        target = duckdb.connect()
        rel = target.sql(f"IMPORT DATABASE '{exported}'")
        assert rel is None or rel._return_type == "nothing"
        assert target.execute("SELECT count(*) FROM t").fetchall() == [(3,)]

    def test_an_alter_with_a_volatile_default_reports_no_result(self, con):
        deferred = con.sql("ALTER TABLE t ADD COLUMN r DOUBLE DEFAULT random()")
        constant = con.sql("ALTER TABLE t ADD COLUMN c DOUBLE DEFAULT 1")
        assert deferred._return_type == constant._return_type == "nothing"

    def test_spark_gives_an_empty_frame_for_both(self, exported):
        spark_sql = pytest.importorskip("duckdb.experimental.spark.sql")
        spark = spark_sql.SparkSession.builder.getOrCreate()
        assert spark.sql(f"IMPORT DATABASE '{exported}'").columns == []
        assert spark.sql("ALTER TABLE t ADD COLUMN r DOUBLE DEFAULT random()").columns == []

    def test_spark_gives_checkpoint_the_frame_the_engine_reports(self):
        # The engine reports CHECKPOINT as a result with a Success column, and Spark follows the engine.
        spark_sql = pytest.importorskip("duckdb.experimental.spark.sql")
        spark = spark_sql.SparkSession.builder.getOrCreate()
        frame = spark.sql("CHECKPOINT")
        assert frame.columns == ["Success"]
        assert frame.collect() == []


class TestAScriptsLastStatement:
    """A last statement that runs at the call fails as execution fails, whatever the statements before it."""

    @pytest.mark.parametrize(
        "script",
        [
            pytest.param("SELECT 1; INSERT INTO missing VALUES (1)", id="after_rows"),
            pytest.param("CREATE TABLE made (i INTEGER); INSERT INTO missing VALUES (1)", id="after_a_create"),
            pytest.param("INSERT INTO missing VALUES (1)", id="alone"),
        ],
    )
    def test_a_failing_last_statement_aborts_the_open_transaction(self, con, script):
        con.execute("BEGIN TRANSACTION")
        with pytest.raises(duckdb.CatalogException):
            con.sql(script)
        assert transaction_state(con) is duckdb.TransactionException
        con.execute("ROLLBACK")


class TestASequenceReadOnAFreshSequence:
    def test_select_nextval_runs_at_the_call(self, con):
        rel = con.sql("SELECT nextval('s') AS v")
        assert currval(con) == 1
        assert rel.fetchall() == [(1,)]
        assert rel.fetchall() == [(1,)]
        assert currval(con) == 1


class TestAWritingCteWithAQuotedConflictTarget:
    def test_it_constructs_and_writes_once(self, con):
        con.execute('CREATE TABLE tw ("two words" INTEGER PRIMARY KEY)')
        text = 'WITH w AS (INSERT INTO tw VALUES (?) ON CONFLICT ("two words") DO NOTHING RETURNING *) SELECT * FROM w'
        rel = con.sql(text, params=[2])
        assert con.execute("SELECT count(*) FROM tw").fetchall() == [(1,)]
        assert rel.fetchall() == [(2,)]
        assert rel.fetchall() == [(2,)]
        assert con.execute("SELECT count(*) FROM tw").fetchall() == [(1,)]


class TestTextBeyondAscii:
    """Text beyond ASCII before a token changes nothing the engine reads after it."""

    def test_a_marker_after_it_is_counted(self, con):
        assert con.sql("SELECT 'é' AS s, ? AS p", params=[7]).fetchall() == [("é", 7)]

    def test_a_writing_cte_after_it_runs_once(self, con):
        con.execute("CREATE TABLE written (i INTEGER)")
        rel = con.sql("WITH q AS (SELECT 'é' AS s), w AS (INSERT INTO written VALUES (1) RETURNING i) SELECT * FROM w")
        assert con.execute("SELECT count(*) FROM written").fetchall() == [(1,)]
        rel.fetchall()
        rel.fetchall()
        assert con.execute("SELECT count(*) FROM written").fetchall() == [(1,)]

    def test_a_filter_ending_in_a_line_comment_after_it(self, con):
        rel = con.sql("SELECT 'é' AS s").filter("s = 'é' " + "-" * 2 + " note")
        assert rel.fetchall() == [("é",)]

    def test_a_join_condition_naming_such_a_column_alone_is_using(self, con):
        left = con.sql("SELECT 1 AS é, 2 AS x", alias="l")
        right = con.sql("SELECT 1 AS é, 3 AS y", alias="r")
        rel = left.join(right, "é")
        assert rel.columns == ["é", "x", "y"]
        assert rel.fetchall() == [(1, 2, 3)]

    def test_a_query_text_with_a_semicolon_inside_a_literal_is_one_statement(self, con):
        rel = con.sql("SELECT 1 AS a").query("q", "SELECT 'é;' AS s FROM q")
        assert rel.fetchall() == [("é;",)]


class TestAKeywordAsAColumnName:
    def test_a_column_named_select_renders_quoted_and_binds(self, con):
        rel = con.sql('SELECT 1 AS "select"').project(duckdb.ColumnExpression("select"))
        assert '"select"' in rel.sql_query()
        assert rel.fetchall() == [(1,)]

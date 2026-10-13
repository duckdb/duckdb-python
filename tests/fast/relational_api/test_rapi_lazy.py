import re

import pytest

import duckdb


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute("CREATE SEQUENCE s")
    yield connection
    connection.close()


def counter(con):
    return con.sql("SELECT nextval('s') AS v")


def currval(con):
    return con.execute("SELECT currval('s')").fetchall()[0][0]


class TestRelationsKeepNoResult:
    @pytest.mark.parametrize(
        "terminal",
        [
            pytest.param(lambda rel: rel.fetchall()[0][0], id="fetchall"),
            pytest.param(lambda rel: int(rel.df()["v"][0]), id="df"),
            pytest.param(lambda rel: int(rel.fetchnumpy()["v"][0]), id="fetchnumpy"),
            pytest.param(lambda rel: int(re.findall(r"(\d+)\s*\u2502", str(rel))[-1]), id="str"),
        ],
    )
    def test_every_call_runs_the_relation_again(self, con, terminal):
        rel = counter(con)
        assert terminal(rel) == 1
        assert terminal(rel) == 2
        assert currval(con) == 2

    def test_a_relation_is_not_run_when_it_is_built(self, con):
        counter(con).filter("v > 0")
        con.execute("SELECT nextval('s')")
        assert currval(con) == 1

    @pytest.mark.parametrize(
        "terminal",
        [
            pytest.param(lambda rel: rel.fetchall(), id="fetchall"),
            pytest.param(lambda rel: rel.df(), id="df"),
            pytest.param(lambda rel: str(rel), id="str"),
            pytest.param(lambda rel: len(rel), id="len"),
            pytest.param(lambda rel: rel.limit(1).fetchall(), id="limit"),
        ],
    )
    def test_a_finished_call_leaves_the_connection_free(self, con, terminal):
        rel = con.sql("SELECT range AS i FROM range(100000)")
        terminal(rel)
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        assert con.sql("SELECT 43").fetchall() == [(43,)]

    def test_close_leaves_the_relation_usable(self, con):
        rel = counter(con)
        rel.close()
        assert rel.fetchall() == [(1,)]

    @pytest.mark.parametrize(
        ("method", "replacement"),
        [
            ("fetchone", "limit(1).fetchall()"),
            ("fetchmany", "limit(n).fetchall()"),
            ("fetch_df_chunk", "to_arrow_reader()"),
            ("execute", "materialize()"),
        ],
    )
    def test_a_removed_method_names_its_replacement(self, con, method, replacement):
        rel = con.sql("SELECT 1 AS a")
        pattern = rf"no longer have {method}\(\).*{re.escape(replacement)}"
        with pytest.raises(AttributeError, match=pattern):
            getattr(rel, method)

    def test_a_column_named_like_a_removed_method_is_still_reachable(self, con):
        rel = con.sql("SELECT 1 AS execute, 2 AS fetchone")
        assert rel["execute"].fetchall() == [(1,)]
        assert rel.fetchone.fetchall() == [(2,)]


class TestMaterialize:
    def test_it_runs_once_and_every_read_sees_the_same_rows(self, con):
        materialized = counter(con).materialize()
        assert currval(con) == 1
        assert materialized.fetchall() == [(1,)]
        assert materialized.fetchall() == [(1,)]
        assert currval(con) == 1

    def test_it_composes_with_other_relations(self, con):
        materialized = con.sql("SELECT range AS i FROM range(5)").materialize()
        other = con.sql("SELECT range AS j FROM range(3)")
        joined = materialized.filter("i > 0").join(other, "i = j").order("i")
        assert joined.fetchall() == [(1, 1), (2, 2)]

    def test_sql_on_its_connection_can_scan_it_by_name(self, con):
        materialized = con.sql("SELECT range AS i FROM range(3)").materialize()  # noqa: F841
        assert con.sql("SELECT sum(i) FROM materialized").fetchall() == [(3,)]

    def test_it_keeps_the_column_names_and_types(self, con):
        rel = con.sql("SELECT 1::TINYINT AS small, 'x' AS text")
        materialized = rel.materialize()
        assert materialized.columns == rel.columns
        assert materialized.types == rel.types


class TestSqlThatRunsOnce:
    """A SELECT with params, and any other statement, runs at the call; its relation reads the rows back."""

    @pytest.mark.parametrize(
        ("query", "params"),
        [
            pytest.param("SELECT nextval('s') AS v, ? AS p", [1], id="positional"),
            pytest.param("SELECT nextval('s') AS v, $p AS p", {"p": 1}, id="named"),
        ],
    )
    def test_a_select_with_params_runs_at_the_call_and_never_again(self, con, query, params):
        rel = con.sql(query, params=params)
        assert currval(con) == 1
        assert rel.fetchall() == [(1, 1)]
        assert rel.fetchall() == [(1, 1)]
        assert rel.filter("p = 1").fetchall() == [(1, 1)]
        assert currval(con) == 1

    def test_it_reads_the_rows_of_the_call_not_the_table_now(self, con):
        con.execute("CREATE TABLE t AS SELECT range AS i FROM range(5)")
        rel = con.sql("SELECT i FROM t WHERE i < ? ORDER BY i", params=[3])
        con.execute("DROP TABLE t")
        assert rel.fetchall() == [(0,), (1,), (2,)]

    def test_an_error_raises_at_the_call(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.sql("SELECT error(?)", params=["boom"])

    def test_a_statement_without_rows_runs_and_returns_none(self, con):
        con.execute("CREATE TABLE log (v BIGINT)")
        assert con.sql("INSERT INTO log VALUES (?)", params=[5]) is None
        assert con.execute("SELECT * FROM log").fetchall() == [(5,)]

    def test_a_statement_other_than_select_runs_once_without_params_too(self, con):
        con.execute("CREATE TABLE log (v BIGINT)")
        rel = con.sql("INSERT INTO log VALUES (nextval('s')) RETURNING v")
        assert rel.fetchall() == [(1,)]
        assert rel.fetchall() == [(1,)]
        assert con.execute("SELECT * FROM log").fetchall() == [(1,)]

    def test_statements_before_the_last_one_run_at_the_call(self, con):
        con.execute("CREATE TABLE log (v BIGINT)")
        rel = con.sql("INSERT INTO log VALUES (1); SELECT ? AS x", params=[2])
        assert con.execute("SELECT * FROM log").fetchall() == [(1,)]
        assert rel.fetchall() == [(2,)]

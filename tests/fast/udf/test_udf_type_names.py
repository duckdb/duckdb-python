import re
import threading
import types
from collections.abc import Callable
from typing import Any, NamedTuple

import pytest

import duckdb
from duckdb.sqltypes import INTEGER


@pytest.fixture
def default_con():
    previous = duckdb.connect(":default:")
    con = duckdb.connect()
    duckdb.set_default_connection(con)
    yield con
    duckdb.set_default_connection(previous)
    con.close()


def module(target, *args):
    return duckdb.create_function(*args)


def default_method(target, *args):
    return duckdb.default_connection().create_function(*args)


def module_with_connection(target, *args):
    return duckdb.create_function(*args, connection=target)


def method(target, *args):
    return target.create_function(*args)


ENTRY_POINTS = {
    "module": module,
    "default_method": default_method,
    "module_with_connection": module_with_connection,
    "method": method,
}
ON_THE_DEFAULT_CONNECTION = {"module", "default_method"}


def target_connection(entry, default_con):
    return default_con if entry in ON_THE_DEFAULT_CONNECTION else duckdb.connect()


def signature(con, name):
    return con.sql(
        f"SELECT parameter_types, return_type FROM duckdb_functions() WHERE function_name = '{name}'"
    ).fetchall()


def plus_one(x):
    return x + 1


def identity(x):
    return x


def annotated_int(x: int) -> int:
    return x + 1


def annotated_list_of_names(x: list["INTEGER"]) -> int:
    return len(x)


class Form(NamedTuple):
    function: Callable
    parameters: Any
    return_type: Any
    argument: str
    rows: list
    signature: tuple


FORMS = {
    "names": Form(plus_one, ["INTEGER"], "INTEGER", "1", [(2,)], (["INTEGER"], "INTEGER")),
    "nested_names": Form(
        identity,
        ["STRUCT(a INTEGER)"],
        "STRUCT(a INTEGER)",
        "{'a': 1}",
        [({"a": 1},)],
        (["STRUCT(a INTEGER)"], "STRUCT(a INTEGER)"),
    ),
    "name_parameters": Form(plus_one, ["INTEGER"], INTEGER, "1", [(2,)], (["INTEGER"], "INTEGER")),
    "name_return_type": Form(plus_one, [INTEGER], "INTEGER", "1", [(2,)], (["INTEGER"], "INTEGER")),
    "name_return_type_only": Form(plus_one, None, "INTEGER", "1", [(2,)], (["ANY"], "INTEGER")),
    "names_in_a_dict": Form(
        identity,
        [{"a": "INTEGER"}],
        {"a": "INTEGER"},
        "{'a': 1}",
        [({"a": 1},)],
        (["STRUCT(a INTEGER)"], "STRUCT(a INTEGER)"),
    ),
    "names_in_a_list_alias": Form(
        identity, [list["INTEGER"]], list["INTEGER"], "[1]", [([1],)], (["INTEGER[]"], "INTEGER[]")
    ),
    "names_in_an_annotation": Form(annotated_list_of_names, None, None, "[1, 2]", [(2,)], (["INTEGER[]"], "BIGINT")),
    "sqltypes": Form(plus_one, [INTEGER], INTEGER, "1", [(2,)], (["INTEGER"], "INTEGER")),
    "python_types": Form(plus_one, [int], int, "1", [(2,)], (["BIGINT"], "BIGINT")),
    "annotations": Form(annotated_int, None, None, "1", [(2,)], (["BIGINT"], "BIGINT")),
}


class NotAType:
    pass


def annotated_with_call(name, call):
    def template(x):
        return x

    function = types.FunctionType(template.__code__, {name: call})
    function.__annotations__ = {"x": f"{name}()", "return": "int"}
    return function


def in_each_position(value):
    return [pytest.param([value], INTEGER, id="parameter"), pytest.param([INTEGER], value, id="return_type")]


class TestUDFTypeNames:
    @pytest.mark.parametrize("form", FORMS)
    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_types_resolve_on_the_registering_connection(self, default_con, entry, form):
        case = FORMS[form]
        target = target_connection(entry, default_con)
        ENTRY_POINTS[entry](target, "f", case.function, case.parameters, case.return_type)
        assert signature(target, "f") == [case.signature]
        assert target.sql(f"SELECT f({case.argument})").fetchall() == case.rows

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_an_enum_created_on_the_connection_resolves_by_name(self, default_con, entry):
        target = target_connection(entry, default_con)
        target.execute("CREATE TYPE mood AS ENUM ('sad', 'happy')")
        ENTRY_POINTS[entry](target, "echo", identity, ["mood"], "mood")
        assert target.sql("SELECT echo('happy') AS r, typeof(r)").fetchall() == [("happy", "ENUM('sad', 'happy')")]

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_an_enum_inside_an_annotation_resolves_by_name(self, default_con, entry):
        def count_moods(x: list["mood"]) -> int:  # noqa: F821
            return len(x)

        target = target_connection(entry, default_con)
        target.execute("CREATE TYPE mood AS ENUM ('sad', 'happy')")
        ENTRY_POINTS[entry](target, "count_moods", count_moods, None, None)
        assert signature(target, "count_moods") == [(["ENUM('sad', 'happy')[]"], "BIGINT")]

    @pytest.mark.parametrize("entry", ["module_with_connection", "method"])
    def test_a_name_on_both_connections_means_the_registering_ones(self, default_con, entry):
        default_con.execute("CREATE TYPE mood AS ENUM ('x', 'y')")
        target = duckdb.connect()
        target.execute("CREATE TYPE mood AS ENUM ('sad', 'happy')")
        ENTRY_POINTS[entry](target, "echo", identity, ["mood"], "mood")
        assert signature(target, "echo") == [(["ENUM('sad', 'happy')"], "ENUM('sad', 'happy')")]
        assert target.sql("SELECT echo('happy')").fetchall() == [("happy",)]

    @pytest.mark.parametrize(("parameters", "return_type"), in_each_position("no_such_type"))
    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_an_unknown_type_name_reports_the_catalog_error(self, default_con, entry, parameters, return_type):
        target = target_connection(entry, default_con)
        with pytest.raises(duckdb.CatalogException, match="Type with name no_such_type does not exist"):
            ENTRY_POINTS[entry](target, "f", identity, parameters, return_type)

    @pytest.mark.parametrize(("parameters", "return_type"), in_each_position("INTEGER("))
    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_a_malformed_type_name_reports_the_parse_error(self, default_con, entry, parameters, return_type):
        target = target_connection(entry, default_con)
        with pytest.raises(duckdb.InvalidInputException, match=r'Value "INTEGER\(" can not be converted'):
            ENTRY_POINTS[entry](target, "f", identity, parameters, return_type)

    @pytest.mark.parametrize(
        ("value", "error", "message"),
        [
            pytest.param(
                42,
                duckdb.NotImplementedException,
                "Could not convert from object of type '<class 'int'>' to DuckDBPyType",
                id="an_int",
            ),
            pytest.param(NotAType, TypeError, "Could not convert from unknown 'type' to DuckDBPyType", id="a_class"),
        ],
    )
    @pytest.mark.parametrize("position", ["parameter", "return_type"])
    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_an_object_that_is_not_a_type_reports_the_conversion_error(
        self, default_con, entry, position, value, error, message
    ):
        target = target_connection(entry, default_con)
        parameters, return_type = ([value], INTEGER) if position == "parameter" else ([INTEGER], value)
        with pytest.raises(error, match=re.escape(message)):
            ENTRY_POINTS[entry](target, "f", identity, parameters, return_type)

    @pytest.mark.parametrize("entry", ["module_with_connection", "method"])
    def test_annotation_code_that_closes_the_connection_registers_nothing(self, entry):
        database = f":memory:annotation_closes_{entry}"
        target = duckdb.connect(database)
        observer = duckdb.connect(database)
        ENTRY_POINTS[entry](target, "registered_before", plus_one, [INTEGER], INTEGER)

        def close_then_int():
            target.close()
            return int

        with pytest.raises(duckdb.ConnectionException, match="Connection already closed"):
            ENTRY_POINTS[entry](target, "closes", annotated_with_call("close_then_int", close_then_int), None, None)
        assert observer.sql(
            "SELECT function_name FROM duckdb_functions() WHERE function_name IN ('registered_before', 'closes')"
        ).fetchall() == [("registered_before",)]
        observer.close()

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_a_duplicate_name_with_an_unknown_type_name_reports_the_type_error(self, default_con, entry):
        target = target_connection(entry, default_con)
        ENTRY_POINTS[entry](target, "f", identity, ["INTEGER"], "INTEGER")
        with pytest.raises(duckdb.CatalogException, match="Type with name no_such_type does not exist"):
            ENTRY_POINTS[entry](target, "f", identity, ["no_such_type"], "INTEGER")
        assert target.sql("SELECT f(1)").fetchall() == [(1,)]

    @pytest.mark.parametrize(
        ("name", "error"),
        [
            pytest.param("INTEGER(", duckdb.InvalidInputException, id="malformed"),
            pytest.param("no_such_type", duckdb.CatalogException, id="unknown"),
        ],
    )
    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_a_bad_type_name_leaves_the_open_transaction_usable(self, default_con, entry, name, error):
        target = target_connection(entry, default_con)
        target.execute("CREATE TABLE t (i INTEGER)")
        target.execute("BEGIN")
        target.execute("INSERT INTO t VALUES (42)")
        with pytest.raises(error):
            ENTRY_POINTS[entry](target, "f", identity, [name], INTEGER)
        assert target.sql("SELECT count(*) FROM t").fetchall() == [(1,)]
        target.execute("COMMIT")
        assert target.sql("SELECT i FROM t").fetchall() == [(42,)]

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_an_unknown_name_in_an_annotation_leaves_the_open_transaction_usable(self, default_con, entry):
        def count_items(x: list["no_such_type"]) -> int:  # noqa: F821
            return len(x)

        target = target_connection(entry, default_con)
        target.execute("CREATE TABLE t (i INTEGER)")
        target.execute("BEGIN")
        target.execute("INSERT INTO t VALUES (42)")
        ENTRY_POINTS[entry](target, "count_items", count_items, None, None)
        assert signature(target, "count_items") == [(["ANY"], "BIGINT")]
        assert target.sql("SELECT count(*) FROM t").fetchall() == [(1,)]
        target.execute("COMMIT")
        assert target.sql("SELECT i FROM t").fetchall() == [(42,)]

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_an_unfinished_result_refuses_before_any_type_lookup(self, default_con, entry):
        target = target_connection(entry, default_con)
        target.execute("CREATE TABLE t (i INTEGER)")
        target.execute("BEGIN")
        target.execute("INSERT INTO t VALUES (42)")
        target.execute("SELECT * FROM range(10000)")
        assert target.fetchone() == (0,)
        with pytest.raises(duckdb.ResourceInUseException, match="unfinished result from execute"):
            ENTRY_POINTS[entry](target, "f", identity, ["no_such_type"], INTEGER)
        assert len(target.fetchall()) == 9999
        target.execute("COMMIT")
        assert target.sql("SELECT i FROM t").fetchall() == [(42,)]

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_an_unfinished_result_refuses_before_annotation_code_runs(self, default_con, entry):
        target = target_connection(entry, default_con)
        ran = []

        def query_then_int():
            ran.append(target.execute("SELECT 1").fetchall())
            return int

        target.execute("SELECT * FROM range(10000)")
        assert target.fetchone() == (0,)
        with pytest.raises(duckdb.ResourceInUseException, match="unfinished result from execute"):
            ENTRY_POINTS[entry](target, "f", annotated_with_call("query_then_int", query_then_int), None, None)
        assert ran == []
        assert len(target.fetchall()) == 9999

    def test_a_connection_held_by_another_thread_during_conversion_refuses(self):
        target = duckdb.connect()
        entered = threading.Event()
        release = threading.Event()
        held = []
        released = []
        freed = []
        outcome = []

        def block(x):
            entered.set()
            released.append(release.wait(10))
            return x

        def query():
            try:
                outcome.extend(target.sql("SELECT block(1)").fetchall())
            except Exception as error:
                outcome.append(error)

        target.create_function("block", block, [INTEGER], INTEGER, side_effects=True)
        worker = threading.Thread(target=query)

        # Converting a class annotation instantiates it, so the annotations around y hold and free the connection
        class HoldTheConnection:
            dtype = "int32"

            def __init__(self) -> None:
                worker.start()
                held.append(entered.wait(10))

        class FreeTheConnection:
            dtype = "int32"

            def __init__(self) -> None:
                freed.append(True)
                release.set()
                worker.join(10)

        def f(x: HoldTheConnection, y: list["INTEGER"], z: FreeTheConnection) -> int:
            return len(y)

        try:
            with pytest.raises(duckdb.ResourceInUseException, match="connection is busy"):
                target.create_function("f", f)
        finally:
            release.set()
            if worker.ident:
                worker.join(10)
        assert held == [True]
        assert freed == []
        assert released == [True]
        assert not worker.is_alive()
        assert outcome == [(1,)]

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_a_name_in_an_annotation_inside_an_aborted_transaction_refuses(self, default_con, entry):
        target = target_connection(entry, default_con)
        target.execute("BEGIN")
        with pytest.raises(duckdb.BinderException):
            target.execute("SELECT missing")

        # Converting a class annotation instantiates it, so a later annotation can end the aborted transaction
        class RollBack:
            dtype = "int32"

            def __init__(self) -> None:
                target.rollback()

        def f(x: list["INTEGER"], y: RollBack) -> int:
            return len(x)

        with pytest.raises(duckdb.TransactionException, match="transaction is aborted"):
            ENTRY_POINTS[entry](target, "f", f, None, None)
        target.rollback()
        assert signature(target, "f") == []

    def test_registering_from_inside_a_query_on_the_default_connection(self, default_con):
        other = duckdb.connect()

        def register_on_other(x):
            other.create_function("plus_one", plus_one, ["INTEGER"], "INTEGER")
            return x

        default_con.create_function("register_on_other", register_on_other, [INTEGER], INTEGER)
        assert default_con.sql("SELECT register_on_other(1)").fetchall() == [(1,)]
        assert other.sql("SELECT plus_one(1)").fetchall() == [(2,)]

    def test_registering_while_another_thread_holds_the_default_connection(self, default_con):
        entered = threading.Event()
        release = threading.Event()
        released = []
        rows = []

        def block(x):
            entered.set()
            released.append(release.wait(10))
            return x

        default_con.create_function("block", block, [INTEGER], INTEGER)
        query = threading.Thread(target=lambda: rows.extend(default_con.sql("SELECT block(1)").fetchall()))
        query.start()
        try:
            assert entered.wait(10)
            other = duckdb.connect()
            other.create_function("plus_one", plus_one, ["INTEGER"], "INTEGER")
            assert other.sql("SELECT plus_one(1)").fetchall() == [(2,)]
        finally:
            release.set()
            query.join(10)
        assert released == [True]
        assert rows == [(1,)]


TYPE_NAME_PARSE_SITES = {
    "connection_type": lambda con, name: con.type(name),
    "select_types": lambda con, name: con.table("t").select_types([name]),
}


class TestTypeNameParseSites:
    @pytest.mark.parametrize(
        ("name", "error", "message"),
        [
            pytest.param(
                "INTEGER(", duckdb.InvalidInputException, r'Value "INTEGER\(" can not be converted', id="malformed"
            ),
            pytest.param(
                "no_such_type", duckdb.CatalogException, "Type with name no_such_type does not exist", id="unknown"
            ),
        ],
    )
    @pytest.mark.parametrize("site", TYPE_NAME_PARSE_SITES)
    def test_a_bad_type_name_leaves_the_open_transaction_usable(self, site, name, error, message):
        con = duckdb.connect()
        con.execute("CREATE TABLE t (i INTEGER)")
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (42)")
        with pytest.raises(error, match=message):
            TYPE_NAME_PARSE_SITES[site](con, name)
        assert con.sql("SELECT count(*) FROM t").fetchall() == [(1,)]
        con.execute("COMMIT")
        assert con.sql("SELECT i FROM t").fetchall() == [(42,)]

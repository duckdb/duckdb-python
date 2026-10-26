"""A Python exception raised inside a UDF must not be swallowed or replaced.

An exception that says the interpreter cannot continue reaches the caller, and an ordinary error from
user code is only turned into NULL where the user asked for that with exception_handling="return_null".
"""

import re
import threading
from typing import NoReturn

import pytest

import duckdb
from duckdb.sqltypes import BIGINT


class CustomBaseException(BaseException):
    pass


class UnprintableMemoryError(MemoryError):
    def __str__(self) -> str:
        msg = "boom in __str__"
        raise RuntimeError(msg)


class RaisingNameMeta(type):
    @property
    def __name__(cls) -> str:
        msg = "boom in __name__"
        raise RuntimeError(msg)


class NamelessMemoryError(MemoryError, metaclass=RaisingNameMeta):
    pass


# An exception that is not an Exception, or a MemoryError, says the interpreter cannot continue.
MUST_REACH_CALLER = [MemoryError, KeyboardInterrupt, SystemExit, CustomBaseException]
# An ordinary error from user code.
ORDINARY = [ValueError, RecursionError, SystemError]

FORWARDED = "Python exception occurred while executing the UDF: "
ARROW_TABLE = "Could not convert the result into an Arrow Table: "


def params(classes):
    return [pytest.param(cls, id=cls.__name__) for cls in classes]


def raising_udf(exc):
    def boom(x: int) -> int:
        msg = "boom in udf"
        raise exc(msg)

    return boom


def recovering_udf(udf_type):
    """Raises MemoryError on 0 and ValueError on 2, and multiplies every other value by ten."""

    def check(values):
        msg = "boom in udf"
        if 0 in values:
            raise MemoryError(msg)
        if 2 in values:
            raise ValueError(msg)

    def native(x):
        check([x])
        return x * 10

    def arrow(column):
        values = column.to_pylist()
        check(values)
        return [v * 10 for v in values]

    return native if udf_type == "native" else arrow


class TestUDFErrorPropagation:
    @pytest.fixture(autouse=True)
    def _pyarrow(self):
        pytest.importorskip("pyarrow", "18.0.0")

    @pytest.mark.parametrize("udf_type", ["native", "arrow"])
    @pytest.mark.parametrize("exc", params(MUST_REACH_CALLER))
    def test_return_null_lets_a_fatal_exception_reach_the_caller(self, exc, udf_type):
        con = duckdb.connect()
        con.create_function("boom", raising_udf(exc), exception_handling="return_null", type=udf_type)
        with pytest.raises(duckdb.InvalidInputException, match=f"{FORWARDED}{exc.__name__}: boom in udf"):
            con.sql("select boom(i) from range(3) t(i)").fetchall()

    @pytest.mark.parametrize("udf_type", ["native", "arrow"])
    @pytest.mark.parametrize("exc", params(ORDINARY))
    def test_return_null_turns_an_ordinary_exception_into_null(self, exc, udf_type):
        con = duckdb.connect()
        con.create_function("boom", raising_udf(exc), exception_handling="return_null", type=udf_type)
        assert con.sql("select boom(i) from range(3) t(i)").fetchall() == [(None,), (None,), (None,)]

    @pytest.mark.parametrize("udf_type", ["native", "arrow"])
    @pytest.mark.parametrize("exc", params(MUST_REACH_CALLER + ORDINARY))
    def test_forward_error_reports_every_class(self, exc, udf_type):
        con = duckdb.connect()
        con.create_function("boom", raising_udf(exc), type=udf_type)
        with pytest.raises(duckdb.InvalidInputException, match=f"{FORWARDED}{exc.__name__}: boom in udf"):
            con.sql("select boom(i) from range(3) t(i)").fetchall()

    @pytest.mark.parametrize("exception_handling", ["default", "return_null"])
    @pytest.mark.parametrize("udf_type", ["native", "arrow"])
    def test_unprintable_fatal_exception_is_still_reported(self, udf_type, exception_handling):
        con = duckdb.connect()
        con.create_function(
            "boom", raising_udf(UnprintableMemoryError), exception_handling=exception_handling, type=udf_type
        )
        with pytest.raises(duckdb.InvalidInputException, match=f"{FORWARDED}UnprintableMemoryError$"):
            con.sql("select boom(1)").fetchall()

    @pytest.mark.parametrize("exception_handling", ["default", "return_null"])
    @pytest.mark.parametrize("udf_type", ["native", "arrow"])
    def test_class_whose_name_lookup_raises_is_still_reported(self, udf_type, exception_handling):
        con = duckdb.connect()
        con.create_function(
            "boom", raising_udf(NamelessMemoryError), exception_handling=exception_handling, type=udf_type
        )
        with pytest.raises(duckdb.InvalidInputException, match=f"{FORWARDED}NamelessMemoryError: boom in udf"):
            con.sql("select boom(1)").fetchall()

    @pytest.mark.parametrize("exception_handling", ["default", "return_null"])
    @pytest.mark.parametrize("udf_type", ["native", "arrow"])
    def test_message_without_a_utf8_encoding_is_escaped(self, udf_type, exception_handling):
        def boom(x: int) -> int:
            msg = "bad \udcff message"
            raise MemoryError(msg)

        con = duckdb.connect()
        con.create_function("boom", boom, exception_handling=exception_handling, type=udf_type)
        with pytest.raises(
            duckdb.InvalidInputException, match=FORWARDED + re.escape("MemoryError: bad \\udcff message")
        ):
            con.sql("select boom(1)").fetchall()

    @pytest.mark.parametrize(
        ("udf_type", "expected_with_a_null_row"),
        [
            pytest.param("native", [(10,), (None,), (None,), (30,)], id="native"),
            # An arrow UDF is called once for the whole chunk, so its ValueError turns every row into NULL.
            pytest.param("arrow", [(None,), (None,), (None,), (None,)], id="arrow"),
        ],
    )
    def test_connection_stays_usable_after_a_udf_error(self, udf_type, expected_with_a_null_row):
        con = duckdb.connect()
        con.create_function(
            "f", recovering_udf(udf_type), [BIGINT], BIGINT, exception_handling="return_null", type=udf_type
        )
        with pytest.raises(duckdb.InvalidInputException, match=f"{FORWARDED}MemoryError: boom in udf"):
            con.sql("select f(0)").fetchall()
        rows = "(values (1, 1), (2, 2), (3, NULL), (4, 3)) t(k, x)"
        assert con.sql(f"select f(x) from {rows} order by k").fetchall() == expected_with_a_null_row
        rows = "(values (1, 1), (2, NULL), (3, 3)) t(k, x)"
        assert con.sql(f"select f(x) from {rows} order by k").fetchall() == [(10,), (None,), (30,)]

    @pytest.mark.parametrize("udf_type", ["native", "arrow"])
    def test_return_null_forwards_a_fatal_exception_from_a_worker_thread(self, udf_type):
        submitting = threading.get_ident()
        raised_on = []
        raised = threading.Event()
        lock = threading.Lock()

        def raise_once_off_the_submitting_thread():
            if threading.get_ident() == submitting:
                # Waiting releases the GIL and holds this task, so a worker thread takes another one and raises.
                raised.wait(timeout=10)
                return
            with lock:
                if raised_on:
                    return
                raised_on.append(threading.get_ident())
            raised.set()
            msg = "boom in udf"
            raise MemoryError(msg)

        def native(x):
            raise_once_off_the_submitting_thread()
            return x

        def arrow(column):
            raise_once_off_the_submitting_thread()
            return column

        udf = native if udf_type == "native" else arrow
        con = duckdb.connect(config={"threads": 4})
        # Five row groups give the scan several tasks, where range() gives one.
        con.execute("create table t as select range i from range(500000)")
        con.create_function("boom", udf, [BIGINT], BIGINT, exception_handling="return_null", type=udf_type)
        forwarded = ""
        try:
            con.sql("select sum(boom(i)) from t").fetchall()
        except duckdb.InvalidInputException as error:
            forwarded = str(error)
        assert raised_on, "every call of the UDF ran on the submitting thread, so no engine worker thread raised"
        assert raised_on[0] != submitting
        assert f"{FORWARDED}MemoryError: boom in udf" in forwarded
        assert con.sql("select 42").fetchall() == [(42,)]

    def test_arrow_table_conversion_error_names_the_pyarrow_reason(self):
        con = duckdb.connect()
        con.create_function("mixed", lambda x: [1, "a"], [BIGINT], BIGINT, type="arrow")
        with pytest.raises(
            duckdb.InvalidInputException, match=f"{ARROW_TABLE}ArrowInvalid: Could not convert 'a' with type str"
        ):
            con.sql("select mixed(i) from range(2) t(i)").fetchall()

    def test_unprintable_arrow_table_conversion_error_is_still_reported(self):
        class Unconvertible:
            def __arrow_array__(self, type=None) -> NoReturn:
                raise UnprintableMemoryError

        con = duckdb.connect()
        con.create_function("unconvertible", lambda x: Unconvertible(), [BIGINT], BIGINT, type="arrow")
        with pytest.raises(duckdb.InvalidInputException, match=f"{ARROW_TABLE}UnprintableMemoryError$"):
            con.sql("select unconvertible(1)").fetchall()

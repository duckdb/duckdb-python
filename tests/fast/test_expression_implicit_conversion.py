"""Tests that every Python value an expression operand may be is accepted where an expression is expected.

Key semantics:
- ``str`` is a value, a string constant; a column is named with ``ColumnExpression``.
- ``bytes`` is a BLOB constant.
- ``None`` is SQL NULL, and the other plain values become constants of their SQL type.
- An object no expression can hold raises TypeError.
- The relation methods still read a bare string as a column name (``rel.select("i")``, ``rel.sort("x")``).
"""

import datetime
import decimal
import platform
import uuid

import pytest

import duckdb
from duckdb import (
    CaseExpression,
    CoalesceOperator,
    ColumnExpression,
    FunctionExpression,
)

pytestmark = pytest.mark.skipif(
    platform.system() == "Emscripten",
    reason="Extensions are not supported on Emscripten",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rel():
    """A one-row relation with columns of various types."""
    con = duckdb.connect()
    r = con.sql(
        """
        SELECT
            42 AS i,
            3.14 AS f,
            'hello' AS s,
            TRUE AS b,
            DATE '2024-01-15' AS d,
            TIMESTAMP '2024-01-15 10:30:00' AS ts,
            TIME '10:30:00' AS t,
            INTERVAL 5 DAY AS iv,
            1.23::DECIMAL(18,2) AS dec,
            'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'::UUID AS u
        """
    )
    yield r
    con.close()


# ---------------------------------------------------------------------------
# Constant types: these become ConstantExpression via TransformPythonValue.
# Each entry maps (value, compatible_column) so the == comparison is valid.
# ---------------------------------------------------------------------------

CONSTANT_VALUES = {
    "int": (42, "i"),
    "float": (3.14, "f"),
    "bool": (True, "b"),
    "None": (None, "i"),  # NULL compares with anything
    "date": (datetime.date(2024, 1, 15), "d"),
    "datetime": (datetime.datetime(2024, 1, 15, 10, 30), "ts"),
    "time": (datetime.time(10, 30), "t"),
    "timedelta": (datetime.timedelta(days=5), "iv"),
    "Decimal": (decimal.Decimal("1.23"), "dec"),
    "UUID": (uuid.UUID("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"), "u"),
}


# ---------------------------------------------------------------------------
# 1. Binary operator with constant types: col == <value>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "column"),
    list(CONSTANT_VALUES.values()),
    ids=list(CONSTANT_VALUES.keys()),
)
def test_binary_operator_constant_rhs(rel, value, column):
    """Expression == <constant> should work for every constant type."""
    expr = ColumnExpression(column) == value
    # `==` must build a SQL Expression, never fall back to a Python bool: a bool RHS would still let
    # select() yield one row, masking a None/operator regression -- so assert the type explicitly.
    assert isinstance(expr, duckdb.Expression)
    result = rel.select(expr).fetchall()
    assert len(result) == 1


# ---------------------------------------------------------------------------
# 1b. None operand: None is a meaningful value (SQL NULL), not "argument absent".
#     nanobind gates None for bound-type params before implicit conversion, so the
#     operators/between take py::object + route None through ToExpression -> NULL constant.
#     These guard the P0 (`== None` -> Python bool) and P1 (operators/between raise on None).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda c: c == None,  # noqa: E711
        lambda c: c != None,  # noqa: E711
        lambda c: c + None,
        lambda c: c - None,
        lambda c: c * None,
        lambda c: c < None,
        lambda c: c > None,
        lambda c: c & None,
        lambda c: c | None,
        lambda c: c.between(None, 5),
        lambda c: c.between(1, None),
        lambda c: None + c,  # reflected (__radd__)
        lambda c: None & c,  # reflected (__rand__)
    ],
    ids=[
        "eq",
        "ne",
        "add",
        "sub",
        "mul",
        "lt",
        "gt",
        "and",
        "or",
        "between_lower",
        "between_upper",
        "reflected_add",
        "reflected_and",
    ],
)
def test_none_operand_builds_sql_null_expression(build):
    """A None operand becomes a SQL NULL constant on every operator/between, yielding a real Expression."""
    expr = build(ColumnExpression("a"))
    assert isinstance(expr, duckdb.Expression)
    assert "NULL" in str(expr)


def test_none_filter_keeps_no_rows():
    """`col != None` builds `(col != NULL)`: SQL NULL semantics keep no rows (a Python-bool True kept all)."""
    rel = duckdb.connect().sql("SELECT * FROM (VALUES (1), (NULL), (3)) t(a)")
    assert rel.filter(ColumnExpression("a") != None).fetchall() == []  # noqa: E711


def test_unconvertible_operand_raises_type_error():
    """An unconvertible operand raises TypeError, never a thrown duckdb error.

    Changed: an operand no expression can hold raises TypeError; `==` returned False.
    """
    a = ColumnExpression("a")
    with pytest.raises(TypeError, match="as an expression operand"):
        a == object()  # noqa: B015
    with pytest.raises(TypeError, match="as an expression operand"):
        a != object()  # noqa: B015
    with pytest.raises(TypeError):
        a + object()


# ---------------------------------------------------------------------------
# 2. Binary operator with str: str is a string constant
# ---------------------------------------------------------------------------


def test_binary_operator_str_rhs(rel):
    """Str on the RHS is a string constant.

    Changed: a bare string operand is a value; it was a column reference.
    """
    expr = ColumnExpression("s") == "hello"
    assert str(expr) == "(\"s\" = 'hello')"
    result = rel.select(expr).fetchall()
    assert result == [(True,)]


def test_binary_operator_bytes_rhs(rel):
    """Bytes on the RHS is a BLOB constant.

    Changed: bytes, like bytearray, is a BLOB value; it was a column reference.
    """
    expr = ColumnExpression("s").cast("BLOB") == b"hello"
    assert isinstance(expr, duckdb.Expression)
    assert str(expr) == "(CAST(\"s\" AS BLOB) = '\\x68\\x65\\x6c\\x6c\\x6f'::BLOB)"
    assert rel.select(expr).fetchall() == [(True,)]


def test_project_with_bytes_column_name(rel):
    """rel.select(b'col') references the column (bytes decoded), not a silent BLOB constant (regression guard)."""
    assert rel.select(b"i").fetchall() == [(42,)]


# ---------------------------------------------------------------------------
# 3. Reflected operators: <value> + col
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [1, 1.0, decimal.Decimal("1")],
    ids=["int", "float", "Decimal"],
)
def test_reflected_operator_lhs(rel, value):
    """<scalar> + Expression should work via __radd__."""
    expr = value + ColumnExpression("i")
    result = rel.select(expr).fetchall()
    assert len(result) == 1


# ---------------------------------------------------------------------------
# 4. Expression.isin() / isnotin() with mixed scalar types
# ---------------------------------------------------------------------------


def test_isin_with_scalars(rel):
    expr = ColumnExpression("i").isin(42, 99, None)
    result = rel.select(expr).fetchall()
    assert result == [(True,)]


def test_isnotin_with_scalars(rel):
    expr = ColumnExpression("i").isnotin(1, 2, 3)
    result = rel.select(expr).fetchall()
    assert result == [(True,)]


# ---------------------------------------------------------------------------
# 5. Expression.between() with scalar bounds
# ---------------------------------------------------------------------------


def test_between_with_int_scalars(rel):
    expr = ColumnExpression("i").between(0, 100)
    result = rel.select(expr).fetchall()
    assert result == [(True,)]


def test_between_with_date_scalars(rel):
    expr = ColumnExpression("d").between(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    result = rel.select(expr).fetchall()
    assert result == [(True,)]


# ---------------------------------------------------------------------------
# 6. CaseExpression / when / otherwise with scalar values
# ---------------------------------------------------------------------------


def test_case_expression_with_scalars(rel):
    case = CaseExpression(ColumnExpression("i") == 42, 1)
    case = case.otherwise(0)
    result = rel.select(case).fetchall()
    assert result == [(1,)]


def test_when_otherwise_with_scalars(rel):
    case = CaseExpression(ColumnExpression("i") == 0, 0)
    case = case.when(ColumnExpression("i") == 42, 42)
    case = case.otherwise(None)
    result = rel.select(case).fetchall()
    assert result == [(42,)]


# ---------------------------------------------------------------------------
# 7. CoalesceOperator with scalars
# ---------------------------------------------------------------------------


def test_coalesce_with_scalars(rel):
    expr = CoalesceOperator(None, None, 42)
    result = rel.select(expr).fetchall()
    assert result == [(42,)]


# ---------------------------------------------------------------------------
# 8. FunctionExpression with scalar args
# ---------------------------------------------------------------------------


def test_function_expression_with_scalars(rel):
    expr = FunctionExpression("greatest", ColumnExpression("i"), 99)
    result = rel.select(expr).fetchall()
    assert result == [(99,)]


# ---------------------------------------------------------------------------
# 9. Relation.sort() with str (column reference)
# ---------------------------------------------------------------------------


def test_sort_with_string():
    con = duckdb.connect()
    rel = con.sql("SELECT * FROM (VALUES (2, 'b'), (1, 'a'), (3, 'c')) t(x, y)")
    result = rel.sort("x").fetchall()
    assert result == [(1, "a"), (2, "b"), (3, "c")]


# ---------------------------------------------------------------------------
# 10. Relation.select() with constant scalars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        42,
        3.14,
        True,
        None,
        datetime.date(2024, 1, 15),
        datetime.datetime(2024, 1, 15, 10, 30),
        datetime.time(10, 30),
        datetime.timedelta(days=5),
        decimal.Decimal("1.23"),
        uuid.UUID("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"),
    ],
    ids=[
        "int",
        "float",
        "bool",
        "None",
        "date",
        "datetime",
        "time",
        "timedelta",
        "Decimal",
        "UUID",
    ],
)
def test_select_with_constant(rel, value):
    """rel.select(<constant>) should produce a one-row result."""
    result = rel.select(value).fetchall()
    assert len(result) == 1


def test_select_with_string(rel):
    """rel.select(<str>) selects a column by name."""
    result = rel.select("i").fetchall()
    assert result == [(42,)]


# ---------------------------------------------------------------------------
# 11. Relation.project() with scalar
# ---------------------------------------------------------------------------


def test_project_with_scalar(rel):
    result = rel.project(42).fetchall()
    assert result == [(42,)]


# ---------------------------------------------------------------------------
# 12. Relation.aggregate() with scalar in list
# ---------------------------------------------------------------------------


def test_aggregate_with_scalar():
    con = duckdb.connect()
    rel = con.sql("SELECT * FROM (VALUES (1), (2), (3)) t(a)")
    # A bare int as an aggregate expression is accepted (non-aggregate, one per row)
    result = rel.aggregate([5]).fetchall()
    assert len(result) == 3
    assert all(row == (5,) for row in result)


# ---------------------------------------------------------------------------
# 13. Value-semantic invariants
#
# DuckDBPyExpression is a value-semantic bound type: returned by std::unique_ptr,
# with no shared_ptr holder, no enable_shared_from_this, and no custom type_caster.
# Every combinator deep-copies its operands into a fresh tree, so two wrappers never
# alias the same expression. These lock in the two contracts that design relies on:
#   1. expressions are never cached/aliased by identity (each builder returns fresh)
#   2. an unconvertible argument raises a clear InvalidInputException, not a leaked
#      C++ exception (the helper that replaced the caster must catch + re-raise)
# ---------------------------------------------------------------------------


def test_expressions_are_not_identity_cached():
    """Every builder call yields a fresh object; expressions are value-like, never aliased."""
    a = ColumnExpression("a")
    assert a.alias("x") is not a.alias("x")
    assert (a == 5) is not (a == 5)
    assert a.isin(1, 2) is not a.isin(1, 2)
    # A non-modifier passthrough still yields a distinct wrapper.
    assert a.cast("INTEGER") is not a.cast("INTEGER")


@pytest.mark.parametrize(
    "build",
    [
        lambda bad: ColumnExpression("i").isin(bad),  # py::args path
        lambda bad: CoalesceOperator(bad),  # py::args path
        lambda bad: FunctionExpression("greatest", bad),  # py::args path
    ],
    ids=["isin", "coalesce", "function_expression"],
)
def test_unconvertible_arg_raises_clean_error(build):
    """A value with no expression conversion raises TypeError, not a raw C++ error.

    Changed: an operand no expression can hold raises TypeError; it was InvalidInput.
    """

    class NotConvertible:
        pass

    with pytest.raises(TypeError, match="as an expression operand"):
        build(NotConvertible())

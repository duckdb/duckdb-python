"""Tests that all types in _ExpressionLike are accepted by the implicit conversion path.

pybind11 registers implicit conversions so that any C++ method taking
``const DuckDBPyExpression &`` silently accepts Python scalars.  The stubs
declare these as ``_ExpressionLike``.  This file verifies every type in that
union actually works at runtime.

Key semantics:
- ``str`` always becomes a **ColumnExpression** (column reference), never a string constant.
- ``bytes`` is decoded as UTF-8 by pybind11 and also becomes a ColumnExpression.
- All other types become **ConstantExpression** via ``TransformPythonValue``.
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
    result = rel.select(expr).fetchall()
    assert len(result) == 1


# ---------------------------------------------------------------------------
# 2. Binary operator with str: str becomes a ColumnExpression (column ref)
# ---------------------------------------------------------------------------


def test_binary_operator_str_rhs(rel):
    """Str on the RHS becomes a ColumnExpression (column reference)."""
    # ColumnExpression("i") == "i"  →  column i == column i  →  True
    expr = ColumnExpression("i") == "i"
    result = rel.select(expr).fetchall()
    assert result == [(True,)]


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
#    Note: str values become column refs, so we use int/None scalars here.
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

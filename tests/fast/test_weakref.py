"""Bound types must be weak-referenceable.

pybind11 set ``tp_weaklistoffset`` on every bound type by default, so
``weakref.ref``/``proxy``/``finalize`` and ``WeakValueDictionary`` worked out of the box.
nanobind opts out by default and requires ``py::is_weak_referenceable()`` at registration; without
it those calls raise ``TypeError: cannot create weak reference``. This guards that regression for
every publicly handed-out bound type (Connection, Relation, Expression, Type, Statement).
"""

import weakref

import pytest

import duckdb


@pytest.fixture
def bound_objects():
    con = duckdb.connect()
    objs = {
        "Connection": con,
        "Relation": con.sql("SELECT 42 AS a"),
        "Expression": duckdb.ColumnExpression("a"),
        "Type": duckdb.type("INTEGER"),
        "Statement": con.extract_statements("SELECT 42")[0],
    }
    yield objs
    con.close()


@pytest.mark.parametrize(
    "name",
    ["Connection", "Relation", "Expression", "Type", "Statement"],
)
def test_bound_type_is_weak_referenceable(bound_objects, name):
    obj = bound_objects[name]

    ref = weakref.ref(obj)
    assert ref() is obj

    weakref.proxy(obj)  # must not raise

    finalized = []
    weakref.finalize(obj, finalized.append, name)

    wvd = weakref.WeakValueDictionary()
    wvd["k"] = obj
    assert wvd["k"] is obj

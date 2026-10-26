"""The generated expression modules match what their generators make of this build's catalog."""

import importlib
from pathlib import Path

import pytest

import duckdb

ROOT = Path(__file__).resolve().parents[3]

# The generators live in the source checkout, which a wheel's test job does not have beside the tests.
pytestmark = pytest.mark.skipif(not (ROOT / "scripts").is_dir(), reason="needs the source checkout's scripts")


def generator(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module(name)


def committed(module):
    return (ROOT / "duckdb" / f"{module}.py").read_text()


def test_the_keyword_module_is_current(monkeypatch):
    gen_keywords = generator(monkeypatch, "gen_keywords")
    assert committed("_sql_keywords") == gen_keywords.render(duckdb.connect()), "run scripts/gen_keywords.py"


def test_the_aggregate_module_is_current(monkeypatch):
    gen_aggregates = generator(monkeypatch, "gen_aggregates")
    assert gen_aggregates.verify(duckdb.connect()) == []
    assert committed("_expression_aggregates") == gen_aggregates.render(), "run scripts/gen_aggregates.py"


def test_the_function_family_module_is_current(monkeypatch):
    gen_func_namespaces = generator(monkeypatch, "gen_func_namespaces")
    resolved, report = gen_func_namespaces.build(duckdb.connect())
    assert report["problems"] == []
    assert committed("_expression_namespaces") == gen_func_namespaces.render(resolved), (
        "run scripts/gen_func_namespaces.py"
    )

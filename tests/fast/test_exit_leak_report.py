"""What nanobind prints at interpreter exit about module objects that Python never freed.

A package that keeps a module's globals alive past finalization leaves a fixed set of duckdb objects behind, which
nanobind reports as leaked. Builds without assertions, every wheel among them, stay silent; Debug builds and builds
with FORCE_ASSERT keep the report so that a real leak stays visible there.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

import _duckdb
import pytest

REPORT = "nanobind: leaked"


def _needs(module: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(importlib.util.find_spec(module) is None, reason=f"{module} is not installed")


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True, timeout=60, check=False
    )


# With numpy imported, typing's globals outlive finalization, and so does the overload registry that holds these
# signatures.
OVERLOAD_SIGNATURE_WITH_NUMPY = """
    from typing import overload

    import duckdb
    import numpy

    @overload
    def connect(database: str) -> duckdb.DuckDBPyConnection: ...
    @overload
    def connect(database: None) -> None: ...
    def connect(database):
        return None if database is None else duckdb.connect(database)
    """

POLARS_FRAME_FROM_RELATION = """
    import duckdb
    import polars as pl

    pl.DataFrame(duckdb.connect().sql("select 1 as a"))
    """

CONNECTION_TYPE_KEPT = """
    import ctypes

    import duckdb

    ctypes.pythonapi.Py_IncRef(ctypes.py_object(duckdb.DuckDBPyConnection))
    """

CONNECTION_KEPT = """
    import ctypes

    import duckdb

    ctypes.pythonapi.Py_IncRef(ctypes.py_object(duckdb.connect()))
    """

HELD_BY_CONSTRUCTION = [
    pytest.param(CONNECTION_TYPE_KEPT, id="connection_type_kept"),
    pytest.param(CONNECTION_KEPT, id="connection_kept"),
]


@pytest.mark.skipif(_duckdb._assertions, reason="a build with assertions keeps nanobind's leak report")
@pytest.mark.parametrize(
    "script",
    [
        pytest.param(OVERLOAD_SIGNATURE_WITH_NUMPY, id="overload_signature_with_numpy", marks=_needs("numpy")),
        pytest.param(POLARS_FRAME_FROM_RELATION, id="polars_frame_from_relation", marks=_needs("polars")),
        *HELD_BY_CONSTRUCTION,
    ],
)
def test_objects_held_past_finalization_are_not_reported(script):
    proc = _run(script)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert REPORT not in proc.stderr, proc.stderr[-2000:]


@pytest.mark.skipif(not _duckdb._assertions, reason="a build without assertions silences nanobind's leak report")
@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")), reason="nanobind has no leak report on a free threaded build"
)
@pytest.mark.parametrize("script", HELD_BY_CONSTRUCTION)
def test_objects_held_past_finalization_are_reported_with_assertions(script):
    proc = _run(script)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert REPORT in proc.stderr, proc.stderr[-2000:]


def _build_tree_cache() -> dict[str, str] | None:
    # An editable install loads the module from below its CMake build directory; an installed wheel has none.
    for directory in Path(_duckdb.__file__).resolve().parents[:3]:
        cache = directory / "CMakeCache.txt"
        if cache.is_file():
            entries = {}
            for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
                key, sep, value = line.partition("=")
                if sep and not line.startswith(("#", "//")):
                    entries[key.split(":", 1)[0]] = value
            return entries
    return None


def _cmake_bool(value: str) -> bool:
    # The rule of the $<BOOL:...> generator expression that turns FORCE_ASSERT into DUCKDB_FORCE_ASSERT.
    false = {"", "0", "false", "off", "n", "no", "ignore", "notfound"}
    return value.casefold() not in false and not value.endswith("-NOTFOUND")


def test_the_assertions_flag_follows_the_build_configuration():
    cache = _build_tree_cache()
    if cache is None:
        pytest.skip("the module was not loaded from below a CMake build tree")
    if cache.get("CMAKE_CONFIGURATION_TYPES"):
        pytest.skip("a multi configuration build tree does not record which configuration the module was built in")
    debug = cache.get("CMAKE_BUILD_TYPE", "").casefold() == "debug"
    assert _duckdb._assertions is (debug or _cmake_bool(cache.get("FORCE_ASSERT", "")))

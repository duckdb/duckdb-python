"""What nanobind prints at interpreter exit about module objects that Python never freed.

A package that keeps a module's globals alive past finalization leaves a fixed set of duckdb objects behind, which
nanobind reports as leaked. Builds without assertions, every wheel among them, stay silent; Debug builds and builds
with FORCE_ASSERT keep the report so that a real leak stays visible there. The private switch
``_duckdb._set_leak_report`` turns the report on or off in any build, and the module's own nanobind domain keeps the
flag apart from other nanobind extensions in the process.
"""

from __future__ import annotations

import ctypes
import importlib.util
import platform
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

import _duckdb
import pytest

REPORT = "nanobind: leaked"

# A core dump of a process with hundreds of engine threads takes minutes to write.
NO_CORE_DUMP = """
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    """

REPORT_ON = """
    import _duckdb

    _duckdb._set_leak_report(True)
    """

REPORT_OFF = """
    import _duckdb

    _duckdb._set_leak_report(False)
    """


def _needs(module: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(importlib.util.find_spec(module) is None, reason=f"{module} is not installed")


def _join(*parts: str) -> str:
    return "".join(textwrap.dedent(part) for part in parts)


def _run(*parts: str) -> subprocess.CompletedProcess[str]:
    if sys.platform != "win32":
        parts = (NO_CORE_DUMP, *parts)
    return subprocess.run(
        [sys.executable, "-c", _join(*parts)], capture_output=True, text=True, timeout=60, check=False
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

    pl.DataFrame(duckdb.connect(config={"threads": 1}).sql("select 1 as a"))
    """

CONNECTION_TYPE_KEPT = """
    import ctypes

    import duckdb

    ctypes.pythonapi.Py_IncRef(ctypes.py_object(duckdb.DuckDBPyConnection))
    """

CONNECTION_KEPT = """
    import ctypes

    import duckdb

    ctypes.pythonapi.Py_IncRef(ctypes.py_object(duckdb.connect(config={"threads": 1})))
    """

HELD_BY_CONSTRUCTION = [
    pytest.param(CONNECTION_TYPE_KEPT, id="connection_type_kept"),
    pytest.param(CONNECTION_KEPT, id="connection_kept"),
]

no_report_on_a_free_threaded_build = pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")), reason="nanobind has no leak report on a free threaded build"
)


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
@no_report_on_a_free_threaded_build
@pytest.mark.parametrize("script", HELD_BY_CONSTRUCTION)
def test_objects_held_past_finalization_are_reported_with_assertions(script):
    proc = _run(script)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert REPORT in proc.stderr, proc.stderr[-2000:]


@no_report_on_a_free_threaded_build
@pytest.mark.parametrize("script", HELD_BY_CONSTRUCTION)
def test_the_switch_turns_the_report_on_in_every_build(script):
    proc = _run(REPORT_ON, script)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert REPORT in proc.stderr, proc.stderr[-2000:]


@pytest.mark.parametrize("script", HELD_BY_CONSTRUCTION)
def test_the_switch_turns_the_report_off_in_every_build(script):
    proc = _run(REPORT_OFF, script)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert REPORT not in proc.stderr, proc.stderr[-2000:]


def _nanobind_internals_keys() -> list[str]:
    # nanobind keeps one state record per domain in the interpreter's state dict, under a key that ends in the domain.
    get_state, get_dict = ctypes.pythonapi.PyInterpreterState_Get, ctypes.pythonapi.PyInterpreterState_GetDict
    saved = (get_state.restype, get_dict.restype, get_dict.argtypes)
    try:
        get_state.restype = get_dict.restype = ctypes.c_void_p
        get_dict.argtypes = [ctypes.c_void_p]
        state = ctypes.cast(get_dict(get_state()), ctypes.py_object).value
    finally:
        get_state.restype, get_dict.restype, get_dict.argtypes = saved
    return [key for key in state if isinstance(key, str) and key.startswith("__nb_internals_")]


@pytest.mark.skipif(platform.python_implementation() != "CPython", reason="reads CPython's interpreter state dict")
def test_the_module_has_its_own_nanobind_domain():
    keys = _nanobind_internals_keys()
    assert any(key.endswith("_duckdb__") for key in keys), keys


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

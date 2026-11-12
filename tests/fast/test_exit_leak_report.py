"""What nanobind prints at interpreter exit about module objects that Python never freed.

The report names duckdb objects that are still alive after finalization. CPython does not clear a module dict that a
function still reachable at exit keeps alive, and such a function is easily kept: typing's overload registry survives
once numpy is imported, a polars cache holds a relation type. A ``DuckDBPyType`` instance bound in such a dict, or in a
class level cache a function of the module reaches, would be reported, so the instances live only in modules that
define no function (the two ``sqltypes`` modules and the spark type cache), whose dicts CPython clears at finalization,
and every other module reads them at use time; a program that keeps a connection itself is reported for that one
instance. Builds without assertions, every wheel among them, stay silent; Debug builds and builds with
FORCE_ASSERT keep the report. The private switch ``_duckdb._set_leak_report`` turns the report on or off in any build,
and the module's own nanobind domain keeps the flag apart from other nanobind extensions in the process.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import pkgutil
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
        [sys.executable, "-c", _join(*parts)], capture_output=True, text=True, timeout=120, check=False
    )


# With numpy or torch imported, typing's globals outlive finalization, and so does the overload registry that holds
# these signatures.
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

OVERLOAD_SIGNATURE_WITH_TORCH = OVERLOAD_SIGNATURE_WITH_NUMPY.replace("import numpy", "import torch")

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

SQLTYPES_MODULES = ("duckdb.sqltypes", "_duckdb._sqltypes")

# The modules that may hold DuckDBPyType instances: CPython clears their dicts at finalization, and no function of
# theirs keeps the dict alive past that. A cache dict in the spark types module instead of in _type_cache would be
# reported by the child that uses spark types and then keeps every module dict.
TYPE_STORES = (*SQLTYPES_MODULES, "duckdb.experimental.spark.sql._type_cache")

# Every module of the package imported, then every module dict but the type stores' kept alive while the module
# objects leave sys.modules and die: what happens to a module whose function something holds past finalization.
EVERY_MODULE_DICT_KEPT = """
    import ctypes
    import importlib
    import pkgutil
    import sys

    import duckdb

    for info in pkgutil.walk_packages(duckdb.__path__, "duckdb."):
        try:
            importlib.import_module(info.name)
        except ModuleNotFoundError as error:
            if error.name is None or error.name.partition(".")[0] in ("duckdb", "_duckdb"):
                raise
    kept = [name for name, module in sys.modules.items() if module is not None and name not in TYPE_STORES
            and (name == "duckdb" or name.startswith("duckdb."))]
    for name in kept:
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(vars(sys.modules[name])))
    for name in sorted(kept, reverse=True):
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            sys.modules[parent].__dict__.pop(child, None)
        del sys.modules[name]
    del duckdb
    """.replace("TYPE_STORES", repr(TYPE_STORES))

# A type constant planted in the value module, which the child above must then report.
PLANTED_IN_THE_VALUE_MODULE = """
    import duckdb.value.constant
    from duckdb import sqltypes

    duckdb.value.constant.planted = sqltypes.INTEGER
    """

# A program that casts to spark types and builds a nested frame, then keeps a spark function alive.
SPARK_TYPES_USED_AND_FUNCTION_KEPT = """
    import ctypes

    import duckdb
    from duckdb.experimental.spark.sql import SparkSession
    from duckdb.experimental.spark.sql import functions as F
    from duckdb.experimental.spark.sql.types import ArrayType, DecimalType, IntegerType, StringType
    from duckdb.experimental.spark.sql.types import StructField, StructType

    F.col("a").cast(IntegerType())
    F.col("b").cast(StringType())
    F.col("c").cast(DecimalType(10, 2))
    spark = SparkSession.builder.getOrCreate()
    spark.conn.execute("SET threads = 1")
    schema = StructType([StructField("a", IntegerType()), StructField("b", ArrayType(IntegerType()))])
    assert spark.createDataFrame([(1, [2, 3])], schema).collect()[0].b == [2, 3]
    ctypes.pythonapi.Py_IncRef(ctypes.py_object(F.length))
    """

# These keep a type, a function or a module of duckdb alive, and nothing duckdb instantiates.
NOTHING_OF_DUCKDBS_OWN_KEPT = [
    pytest.param(EVERY_MODULE_DICT_KEPT, id="every_module_dict_kept"),
    pytest.param(SPARK_TYPES_USED_AND_FUNCTION_KEPT, id="spark_types_used_and_function_kept"),
    pytest.param(
        _join(SPARK_TYPES_USED_AND_FUNCTION_KEPT, EVERY_MODULE_DICT_KEPT),
        id="spark_types_used_then_every_module_dict_kept",
    ),
    pytest.param(OVERLOAD_SIGNATURE_WITH_NUMPY, id="overload_signature_with_numpy", marks=_needs("numpy")),
    pytest.param(OVERLOAD_SIGNATURE_WITH_TORCH, id="overload_signature_with_torch", marks=_needs("torch")),
    pytest.param(POLARS_FRAME_FROM_RELATION, id="polars_frame_from_relation", marks=_needs("polars")),
    pytest.param(CONNECTION_TYPE_KEPT, id="connection_type_kept"),
]

no_report_on_a_free_threaded_build = pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")), reason="nanobind has no leak report on a free threaded build"
)


@pytest.mark.parametrize("script", NOTHING_OF_DUCKDBS_OWN_KEPT)
def test_held_types_and_functions_leave_no_instance_behind(script):
    proc = _run(REPORT_ON, script)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert REPORT not in proc.stderr, proc.stderr[-2000:]


@no_report_on_a_free_threaded_build
def test_the_module_dict_child_reports_a_planted_type_constant():
    proc = _run(REPORT_ON, PLANTED_IN_THE_VALUE_MODULE, EVERY_MODULE_DICT_KEPT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "nanobind: leaked 1 instances!" in proc.stderr, proc.stderr[-2000:]
    instances = [line for line in proc.stderr.splitlines() if "leaked instance" in line]
    assert all('"_duckdb._sqltypes.DuckDBPyType"' in line for line in instances), proc.stderr[-2000:]


def _assert_one_connection_reported(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "nanobind: leaked 1 instances!" in proc.stderr, proc.stderr[-2000:]
    # nanobind lists the instances only outside the limited API, so an abi3 module prints the count alone.
    instances = [line for line in proc.stderr.splitlines() if "leaked instance" in line]
    assert all('"_duckdb.DuckDBPyConnection"' in line for line in instances), proc.stderr[-2000:]


@no_report_on_a_free_threaded_build
def test_a_kept_connection_is_reported_as_the_one_instance_it_is():
    _assert_one_connection_reported(_run(REPORT_ON, CONNECTION_KEPT))


@pytest.mark.skipif(_duckdb._assertions, reason="a build with assertions keeps nanobind's leak report")
@pytest.mark.parametrize("script", [*NOTHING_OF_DUCKDBS_OWN_KEPT, pytest.param(CONNECTION_KEPT, id="connection_kept")])
def test_nothing_is_reported_without_assertions(script):
    proc = _run(script)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert REPORT not in proc.stderr, proc.stderr[-2000:]


@pytest.mark.skipif(not _duckdb._assertions, reason="a build without assertions silences nanobind's leak report")
@no_report_on_a_free_threaded_build
def test_a_kept_connection_is_reported_with_assertions():
    _assert_one_connection_reported(_run(CONNECTION_KEPT))


def test_the_switch_turns_the_report_off_in_every_build():
    proc = _run(REPORT_OFF, CONNECTION_KEPT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert REPORT not in proc.stderr, proc.stderr[-2000:]


def _import_or_say_why_not(info: pkgutil.ModuleInfo) -> str | None:
    try:
        importlib.import_module(info.name)
    except ModuleNotFoundError as error:
        if error.name is None or error.name.partition(".")[0] in ("duckdb", "_duckdb"):
            raise
        return f"{info.name} ({error.name} is not installed)"
    return None


def test_no_module_binds_a_type_constant_by_name_outside_sqltypes():
    # A static guard for the module level bindings, readable in a failure; the every_module_dict_kept child above
    # catches a type reachable from any module dict.
    import duckdb

    skipped = [
        reason for reason in map(_import_or_say_why_not, pkgutil.walk_packages(duckdb.__path__, "duckdb.")) if reason
    ]
    print("not imported:", ", ".join(skipped) or "none")
    holders = {}
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "duckdb" or name.startswith(("duckdb.", "_duckdb"))):
            continue
        bound = sorted(key for key, value in vars(module).items() if isinstance(value, _duckdb._sqltypes.DuckDBPyType))
        if bound and name not in TYPE_STORES:
            holders[name] = bound
    assert holders == {}


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

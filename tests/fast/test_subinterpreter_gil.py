"""duckdb keeps working in the main interpreter after the process has created a subinterpreter.

Creating any subinterpreter, even one that never imports duckdb, makes PyGILState_Check report the GIL as held on
every thread for the rest of the process. That switch cannot be undone, so each case runs in a fresh process.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

BODY_DONE = "BODY_DONE"
STARTED = "STARTED"
PRECONDITION = "PyGILState_Check with the GIL released: "
INTERPRETERS = "_interpreters" if sys.version_info >= (3, 13) else "_xxsubinterpreters"


def _needs(module: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(importlib.util.find_spec(module) is None, reason=f"{module} is not installed")


# A core dump of a process with hundreds of engine threads takes minutes to write.
NO_CORE_DUMP = """
import resource

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
"""

# A CDLL call drops the GIL, so the answer shows whether the subinterpreter switched the check off.
SUBINTERPRETER = f"""
import ctypes
import importlib

interpreters = importlib.import_module({INTERPRETERS!r})
interpreters.destroy(interpreters.create())
answer = ctypes.CDLL(None, handle=getattr(sys, "dllhandle", None), winmode=0).PyGILState_Check()
print({PRECONDITION!r} + str(answer))
if answer != 1:
    sys.exit(0)
"""

# Another Python thread holds the GIL while duckdb asks whether this one does; before 3.12 that tells the check
# apart from one asking only whether some thread state is current. It starts after the imports, which it would
# slow down by a GIL handover on every file access.
SPINNER = """
import threading

stop = threading.Event()


def spin():
    while not stop.is_set():
        pass


spinner = threading.Thread(target=spin, daemon=True)
spinner.start()
"""

SPINNER_DONE = """
stop.set()
spinner.join()
"""

PANDAS_IMPORTS = """
import pandas as pd
"""

PANDAS_SCAN = f"""
df = pd.DataFrame({{"a": [1, 2, 3]}})
print({STARTED!r})
con = duckdb.connect()
con.execute("SET enable_progress_bar = false")
print(con.sql("SELECT sum(a) FROM df").fetchall())
"""

ARROW_IMPORTS = """
import pyarrow

from duckdb.sqltypes import BIGINT
"""

# The UDF makes ending the query wait for a task that needs the GIL.
ARROW_READER = f"""
con = duckdb.connect()
con.execute("SET enable_progress_bar = false")
con.execute("SET max_streaming_buffer_size='1MB'")
con.create_function("identity", lambda i: i, [BIGINT], BIGINT)
reader = con.sql("SELECT identity(i) AS i FROM range(2000000) t(i)").to_arrow_reader(1024)
print({STARTED!r})
print(sum(len(reader.read_next_batch()) for _ in range(50)))
del reader
print(con.execute("SELECT 42").fetchall())
"""


# Inputs go in with the imports, before the spinner, which slows building them several fold. At 102400 rows per
# partition, 300000 rows let several of the four threads scan at once.
STR_FRAME = """
df = pd.DataFrame({"s": [str(i) for i in range(300000)]})
"""

PYTHON_STORAGE_FRAME = """
df = pd.DataFrame({"s": pd.array([str(i) for i in range(300000)], dtype=pd.StringDtype("python"))})
"""

NUMPY_IMPORTS = """
import numpy as np
"""

# The object type analysis imports pandas at bind, an import the spinner slows to many seconds.
OBJECT_IMPORTS = NUMPY_IMPORTS + PANDAS_IMPORTS

# The object mask spares the rows the type sample reads, so the type does not hang on how a masked sample is typed.
MASKED_DICTS = """
ints = {"a": np.ma.masked_array(np.arange(300000, dtype=np.int64), mask=np.arange(300000) % 10 == 0)}
objects = {
    "a": np.ma.masked_array(np.array([str(i) for i in range(300000)], dtype=object), mask=np.arange(300000) % 10 == 1)
}
"""

# Python ints bind INTEGER from the type sample, which never reads the last row, so the float there is first met on an
# engine thread; the mask spares the sampled rows as in the dicts above.
OBJECT_INTS = """
object_ints = pd.DataFrame({"a": pd.Series(list(range(300000)), dtype=object)})
mismatched = pd.DataFrame({"amount": pd.Series([*range(299999), 2.5], dtype=object)})
masked_object_ints = {
    "a": np.ma.masked_array(np.array(list(range(300000)), dtype=object), mask=np.arange(300000) % 10 == 1)
}
"""

MISMATCH_QUERY = """
import re

try:
    print(con.sql("SELECT sum(amount) FROM mismatched").fetchall())
except duckdb.Error as error:
    pieces = ("amount", "299999", "float", "INTEGER")
    named = all(re.search(rf"(?<![\\w.]){piece}(?!\\w)", str(error)) for piece in pieces)
    print(type(error).__name__, named)
"""

ONE_CHUNK_TABLE = """
t = pyarrow.table(
    {"i": pyarrow.array(range(1000000), pyarrow.int64()), "s": pyarrow.array([str(i) for i in range(1000000)])}
)
"""

# A pyarrow Table never asks for the GIL around its pulls and batch releases; this source does.
HELD_SOURCE = """
from duckdb._sources import ArrowScanSource


class HeldSource(ArrowScanSource):
    pull_under_gil = True
"""

# map calls the function, binds the frame it returns and scans it on engine threads while holding the GIL.
MAP_FUNCTIONS = """
def as_str(df):
    return pd.DataFrame({"s": df["i"].astype(str)})


def every_other_of_doubled(df):
    doubled = pd.array(np.repeat(df["i"].to_numpy(dtype=np.int64), 2), dtype="Int64")
    doubled[np.repeat(df["i"].to_numpy() % 10 == 0, 2)] = pd.NA
    return pd.DataFrame({"a": doubled}).iloc[::2]
"""

MAPPED_RANGE = 'con.sql("SELECT i FROM range(300000) t(i)").map'

STR_QUERY = 'print(con.sql("SELECT count(s), sum(CAST(s AS BIGINT)) FROM df").fetchall())'
STR_RESULT = f"[(300000, {sum(range(300000))})]"
TABLE_CHUNKS = 'print(t.column("i").num_chunks, t.column("s").num_chunks)'
TABLE_QUERY = "SELECT sum(i), count(s), sum(CAST(s AS BIGINT))"
TABLE_RESULT = ["1 1", f"[({sum(range(1000000))}, 1000000, {sum(range(1000000))})]"]
MASKED_MAX_QUERY = "SELECT (SELECT max(a) FROM ints), (SELECT max(CAST(a AS BIGINT)) FROM objects)"


def _at_four_threads(*statements: str) -> str:
    setup = f"""
print({STARTED!r})
con = duckdb.connect()
con.execute("SET enable_progress_bar = false")
con.execute("SET threads = 4")
"""
    return setup + "".join(f"{statement}\n" for statement in statements)


def _text(stream: str | bytes | None) -> str:
    return stream.decode() if isinstance(stream, bytes) else stream or ""


def _run(imports: str, body: str) -> subprocess.CompletedProcess[str]:
    script = "".join(
        textwrap.dedent(part)
        for part in (
            NO_CORE_DUMP if sys.platform != "win32" else "",
            "import sys\n",
            SUBINTERPRETER,
            "import duckdb\n",
            imports,
            SPINNER,
            body,
            SPINNER_DONE,
            f"print({BODY_DONE!r})\n",
        )
    )
    try:
        return subprocess.run(
            [sys.executable, "-u", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as hang:
        started = _text(hang.stdout)
        where = "after" if STARTED in started else "before"
        msg = f"the child hung {where} reaching the statement under test"
        msg += f"\nstdout:\n{started[-2000:]}\nstderr:\n{_text(hang.stderr)[-2000:]}"
        raise AssertionError(msg) from hang


@_needs(INTERPRETERS)
@pytest.mark.parametrize(
    ("imports", "body", "expected"),
    [
        pytest.param(PANDAS_IMPORTS, PANDAS_SCAN, ["[(6,)]"], id="pandas_scan", marks=_needs("pandas")),
        pytest.param(ARROW_IMPORTS, ARROW_READER, ["51200", "[(42,)]"], id="arrow_reader", marks=_needs("pyarrow")),
        pytest.param(
            PANDAS_IMPORTS + STR_FRAME,
            _at_four_threads('print(type(df["s"].array).__name__)', STR_QUERY),
            ["ArrowStringArray", STR_RESULT],
            id="str_column",
            marks=[_needs("pandas"), _needs("pyarrow")],
        ),
        pytest.param(
            PANDAS_IMPORTS + PYTHON_STORAGE_FRAME,
            _at_four_threads('print(type(df["s"].array).__name__)', STR_QUERY),
            ["StringArray", STR_RESULT],
            id="python_storage",
            marks=_needs("pandas"),
        ),
        pytest.param(
            NUMPY_IMPORTS + MASKED_DICTS,
            _at_four_threads('print(con.sql("SELECT count(a), sum(a) FROM ints").fetchall())'),
            [f"[({sum(1 for i in range(300000) if i % 10)}, {sum(i for i in range(300000) if i % 10)})]"],
            id="numpy_masked_dict",
            marks=_needs("numpy"),
        ),
        pytest.param(
            OBJECT_IMPORTS + MASKED_DICTS,
            _at_four_threads('print(con.sql("SELECT count(a), sum(CAST(a AS BIGINT)) FROM objects").fetchall())'),
            [f"[({sum(1 for i in range(300000) if i % 10 != 1)}, {sum(i for i in range(300000) if i % 10 != 1)})]"],
            id="numpy_masked_object_dict",
            marks=[_needs("numpy"), _needs("pandas")],
        ),
        # The largest values are unmasked, so a crash or hang hidden by the expected failures above still shows here.
        pytest.param(
            OBJECT_IMPORTS + MASKED_DICTS,
            _at_four_threads(f'print(con.sql("{MASKED_MAX_QUERY}").fetchall())'),
            [f"[({max(i for i in range(300000) if i % 10)}, {max(i for i in range(300000) if i % 10 != 1)})]"],
            id="numpy_masked_dicts_max",
            marks=[_needs("numpy"), _needs("pandas")],
        ),
        pytest.param(
            OBJECT_IMPORTS + OBJECT_INTS,
            _at_four_threads('print(con.sql("SELECT count(a), sum(a) FROM object_ints").fetchall())'),
            [f"[(300000, {sum(range(300000))})]"],
            id="object_typed_column",
            marks=[_needs("numpy"), _needs("pandas")],
        ),
        pytest.param(
            OBJECT_IMPORTS + OBJECT_INTS,
            _at_four_threads('print(con.sql("SELECT count(a), sum(a) FROM masked_object_ints").fetchall())'),
            [f"[({sum(1 for i in range(300000) if i % 10 != 1)}, {sum(i for i in range(300000) if i % 10 != 1)})]"],
            id="object_typed_masked_dict",
            marks=[_needs("numpy"), _needs("pandas")],
        ),
        pytest.param(
            OBJECT_IMPORTS + OBJECT_INTS,
            _at_four_threads(MISMATCH_QUERY),
            ["InvalidInputException True"],
            id="object_typed_mismatch",
            marks=[
                _needs("numpy"),
                _needs("pandas"),
                pytest.mark.xfail(
                    strict=True, reason="a float after the type sample of an integer column is truncated silently"
                ),
            ],
        ),
        pytest.param(
            OBJECT_IMPORTS + MAP_FUNCTIONS,
            _at_four_threads(
                'print(type(as_str(pd.DataFrame({"i": [0, 1]}))["s"].array).__name__)',
                f'print({MAPPED_RANGE}(as_str).aggregate("count(s), sum(CAST(s AS BIGINT))").fetchall())',
            ),
            ["ArrowStringArray", STR_RESULT],
            id="map_str_output",
            marks=[_needs("numpy"), _needs("pandas"), _needs("pyarrow")],
        ),
        pytest.param(
            OBJECT_IMPORTS + MAP_FUNCTIONS,
            _at_four_threads(
                'print(every_other_of_doubled(pd.DataFrame({"i": [0, 1]}))["a"].array._mask.flags.c_contiguous)',
                f'print({MAPPED_RANGE}(every_other_of_doubled).aggregate("count(a), sum(a)").fetchall())',
            ),
            ["False", f"[({sum(1 for i in range(300000) if i % 10)}, {sum(i for i in range(300000) if i % 10)})]"],
            id="map_sliced_nullable_output",
            marks=[_needs("numpy"), _needs("pandas")],
        ),
        pytest.param(
            ARROW_IMPORTS + ONE_CHUNK_TABLE,
            _at_four_threads(TABLE_CHUNKS, f'print(con.sql("{TABLE_QUERY} FROM t").fetchall())'),
            TABLE_RESULT,
            id="arrow_one_chunk",
            marks=_needs("pyarrow"),
        ),
        pytest.param(
            ARROW_IMPORTS + HELD_SOURCE + ONE_CHUNK_TABLE,
            _at_four_threads(
                TABLE_CHUNKS,
                'con._register_arrow_source("held", HeldSource(t))',
                f'print(con.sql("{TABLE_QUERY} FROM held").fetchall())',
            ),
            TABLE_RESULT,
            id="arrow_one_chunk_pulled_under_gil",
            marks=_needs("pyarrow"),
        ),
    ],
)
def test_works_after_a_subinterpreter(imports: str, body: str, expected: list[str]) -> None:
    proc = _run(imports, body)
    report = f"exit status {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    answers = [line.removeprefix(PRECONDITION) for line in proc.stdout.splitlines() if line.startswith(PRECONDITION)]
    assert answers, report
    if answers != ["1"]:
        pytest.skip("creating a subinterpreter no longer switches PyGILState_Check off, so nothing here is tested")
    assert proc.returncode == 0, report
    assert proc.stdout.splitlines()[-len(expected) - 1 :] == [*expected, BODY_DONE], report

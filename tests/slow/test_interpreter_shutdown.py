"""The interpreter exits cleanly whatever happens to duckdb's process wide state during shutdown.

duckdb keeps a default connection and a cache of imported Python objects in process wide C++ state.
Each case runs a script in a fresh interpreter and checks how that interpreter exits: a crash at exit
shows up only as the exit status, so nothing here can run in the test process itself.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

BODY_DONE = "BODY_DONE"


def _needs(module: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(importlib.util.find_spec(module) is None, reason=f"{module} is not installed")


# A core dump of a process with hundreds of engine threads takes minutes to write.
NO_CORE_DUMP = """
import resource

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
"""


def _join(*parts: str) -> str:
    return "".join(textwrap.dedent(part) for part in parts)


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    if sys.platform != "win32":
        script = _join(NO_CORE_DUMP, script)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _assert_clean_exit(
    proc: subprocess.CompletedProcess[str], expected_lines: tuple[str, ...], *, leak_free: bool
) -> None:
    report = f"exit status {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    assert proc.returncode == 0, report
    lines = proc.stdout.splitlines()
    assert BODY_DONE in lines, report
    for line in expected_lines:
        assert line in lines, report
    assert "Fatal Python error" not in proc.stderr, report
    if leak_free:
        assert "nanobind: leaked" not in proc.stderr, report


UNLISTED_IMPORT = """
import sys
from unittest import mock

with mock.patch.dict(sys.modules):
    import duckdb
"""

FROZEN_HOLDER = """
import gc

def holder():
    pass

holder.module = duckdb
"""

DROP_CLEANUP_CAPSULE = """
import sys
import duckdb

before = duckdb.default_connection()
del duckdb._clean_default_connection
del sys.modules["_duckdb"]._clean_default_connection
assert duckdb.default_connection() is not before
"""

LATE_ATEXIT_QUERY = """
import atexit

def late():
    print("atexit", holder.module.sql("select 42").fetchall(), flush=True)

atexit.register(late)
"""

# The module is gone from sys.modules and its dict is never freed, so nothing releases the module
# state before the interpreter finalizes.
MODULE_OUTLIVES_SHUTDOWN = [
    pytest.param(
        _join(UNLISTED_IMPORT, FROZEN_HOLDER, 'gc.freeze()\nprint("BODY_DONE", flush=True)\n'),
        (),
        id="frozen_reference",
    ),
    pytest.param(
        _join(
            UNLISTED_IMPORT,
            """
            import ctypes

            ctypes.pythonapi.Py_IncRef(ctypes.py_object(duckdb))
            assert duckdb.sql("select 42").fetchall() == [(42,)]
            print("BODY_DONE", flush=True)
            """,
        ),
        (),
        id="leaked_reference_after_query",
    ),
    pytest.param(
        _join(
            UNLISTED_IMPORT,
            FROZEN_HOLDER,
            """
            import pyarrow as pa

            tbl = pa.table({"a": [1, 2, 3]})
            assert duckdb.sql("select sum(a) from tbl").fetchall() == [(6,)]
            gc.freeze()
            print("BODY_DONE", flush=True)
            """,
        ),
        (),
        id="frozen_reference_after_arrow_scan",
        marks=_needs("pyarrow"),
    ),
    pytest.param(
        _join(LATE_ATEXIT_QUERY, UNLISTED_IMPORT, FROZEN_HOLDER, 'gc.freeze()\nprint("BODY_DONE", flush=True)\n'),
        ("atexit [(42,)]",),
        id="frozen_reference_used_by_earlier_atexit_handler",
    ),
    pytest.param(
        _join(UNLISTED_IMPORT, FROZEN_HOLDER, 'print("BODY_DONE", flush=True)\n'),
        (),
        id="collectable_reference",
    ),
]


@pytest.mark.parametrize(("script", "expected_lines"), MODULE_OUTLIVES_SHUTDOWN)
def test_module_outside_sys_modules(script: str, expected_lines: tuple[str, ...]) -> None:
    _assert_clean_exit(_run(script), expected_lines, leak_free=False)


def test_module_outside_sys_modules_keeps_committed_data(tmp_path) -> None:
    database = str(tmp_path / "exit.db")
    script = _join(
        UNLISTED_IMPORT,
        FROZEN_HOLDER,
        """
        duckdb.set_default_connection(duckdb.connect(sys.argv[1]))
        duckdb.sql("create table t as select 42 as a")
        gc.freeze()
        print("BODY_DONE", flush=True)
        """,
    )
    _assert_clean_exit(_run(script, database), (), leak_free=False)
    reopen = """
        import sys
        import duckdb

        print(duckdb.connect(sys.argv[1]).sql("from t").fetchall())
        print("BODY_DONE", flush=True)
    """
    _assert_clean_exit(_run(reopen, database), ("[(42,)]",), leak_free=True)


# The module state is released while the process runs, then recreated by later use.
STATE_RECREATED_AFTER_RELEASE = [
    pytest.param(
        """
        assert duckdb.sql("select 42").fetchall() == [(42,)]
        print("BODY_DONE", flush=True)
        """,
        id="query",
    ),
    pytest.param(
        """
        import pyarrow as pa

        tbl = pa.table({"a": [1, 2, 3]})
        assert duckdb.sql("select sum(a) from tbl").fetchall() == [(6,)]
        print("BODY_DONE", flush=True)
        """,
        id="arrow_scan",
        marks=_needs("pyarrow"),
    ),
    pytest.param(
        """
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        assert duckdb.sql("select sum(a) from df").fetchall() == [(6,)]
        print("BODY_DONE", flush=True)
        """,
        id="pandas_scan",
        marks=_needs("pandas"),
    ),
    pytest.param(
        """
        import io

        rel = duckdb.read_csv(io.BytesIO(b"a\\n1\\n2\\n"))
        assert rel.fetchall() == [(1,), (2,)]
        print("BODY_DONE", flush=True)
        """,
        id="file_like_relation_alive_at_exit",
        marks=_needs("fsspec"),
    ),
    pytest.param(
        """
        import pandas as pd

        duckdb.register("v", pd.DataFrame({"a": [1, 2, 3]}))
        rel = duckdb.sql("select sum(a) from v")
        assert rel.fetchall() == [(6,)]
        print("BODY_DONE", flush=True)
        """,
        id="registered_dataframe_alive_at_exit",
        marks=_needs("pandas"),
    ),
]


@pytest.mark.parametrize("script", STATE_RECREATED_AFTER_RELEASE)
def test_state_recreated_after_release(script: str) -> None:
    _assert_clean_exit(_run(_join(DROP_CLEANUP_CAPSULE, script)), (), leak_free=True)


# Shutdown paths that must keep exiting cleanly.
ORDINARY_SHUTDOWN = [
    pytest.param(
        """
        import duckdb

        assert duckdb.sql("select 42").fetchall() == [(42,)]
        print("BODY_DONE", flush=True)
        """,
        (),
        True,
        id="plain_query",
    ),
    pytest.param(
        """
        import duckdb
        import pandas as pd
        import pyarrow as pa

        tbl = pa.table({"a": [1, 2, 3]})
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert duckdb.sql("select sum(a) from tbl").fetchall() == [(6,)]
        assert duckdb.sql("select sum(a) from df").fetchall() == [(6,)]
        print("BODY_DONE", flush=True)
        """,
        (),
        True,
        id="arrow_and_pandas_scans",
        marks=[_needs("pyarrow"), _needs("pandas")],
    ),
    pytest.param(
        """
        import atexit

        def late():
            import duckdb

            print("atexit", duckdb.sql("select 42").fetchall(), flush=True)

        atexit.register(late)
        import duckdb

        duckdb.sql("select 1").fetchall()
        print("BODY_DONE", flush=True)
        """,
        ("atexit [(42,)]",),
        True,
        id="atexit_handler_registered_before_import",
    ),
    pytest.param(
        """
        import atexit
        import pyarrow as pa

        tbl = pa.table({"a": [1, 2, 3]})

        def late():
            import duckdb

            print("atexit", duckdb.sql("select sum(a) from tbl").fetchall(), flush=True)

        atexit.register(late)
        import duckdb

        duckdb.sql("select 1").fetchall()
        print("BODY_DONE", flush=True)
        """,
        ("atexit [(6,)]",),
        True,
        id="atexit_handler_registered_before_import_arrow_scan",
        marks=_needs("pyarrow"),
    ),
    pytest.param(
        """
        import atexit
        import duckdb
        import pyarrow as pa

        tbl = pa.table({"a": [1, 2, 3]})

        def late():
            print("atexit", duckdb.sql("select sum(a) from tbl").fetchall(), flush=True)

        atexit.register(late)
        print("BODY_DONE", flush=True)
        """,
        ("atexit [(6,)]",),
        True,
        id="atexit_handler_registered_after_import_arrow_scan",
        marks=_needs("pyarrow"),
    ),
    pytest.param(
        """
        import atexit
        import duckdb
        import pyarrow as pa

        tbl = pa.table({"a": [1, 2, 3]})

        def late():
            print("atexit", duckdb.sql("select sum(a) from tbl").fetchall(), flush=True)

        atexit.register(late)
        print("body", duckdb.sql("select sum(a) from tbl").fetchall(), flush=True)
        print("BODY_DONE", flush=True)
        """,
        ("body [(6,)]", "atexit [(6,)]"),
        True,
        id="atexit_handler_scans_after_a_body_arrow_scan",
        marks=_needs("pyarrow"),
    ),
    pytest.param(
        """
        import atexit
        import duckdb
        import pyarrow as pa

        # Built here, since building arrays during exit imports pandas, which can no longer start threads.
        batches = [pa.record_batch([pa.array(range(i * 100, i * 100 + 100))], names=["i"]) for i in range(500)]
        con = duckdb.connect(config={"threads": 4})
        kept = []

        def late():
            reader = pa.RecordBatchReader.from_batches(batches[0].schema, (batch for batch in batches))
            con.register("r", reader)
            result = con.execute("SELECT i FROM r")
            print("atexit", result.fetchone(), flush=True)
            kept.append((result, reader))

        atexit.register(late)
        print("BODY_DONE", flush=True)
        """,
        ("atexit (0,)",),
        False,
        id="atexit_handler_leaves_an_arrow_query_open",
        marks=_needs("pyarrow"),
    ),
    pytest.param(
        """
        import duckdb

        class Finalized:
            def __del__(self):
                print("finalizer", duckdb.sql("select 42").fetchall(), flush=True)

        kept = Finalized()
        print("BODY_DONE", flush=True)
        """,
        ("finalizer [(42,)]",),
        True,
        id="finalizer_in_main_module",
    ),
    pytest.param(
        """
        import builtins
        import duckdb

        class Finalized:
            def __del__(self):
                print("finalizer", duckdb.sql("select 42").fetchall(), flush=True)

        builtins.kept = Finalized()
        print("BODY_DONE", flush=True)
        """,
        ("finalizer [(42,)]",),
        True,
        id="finalizer_in_builtins",
    ),
    pytest.param(
        """
        import duckdb

        rel = duckdb.sql("select 42")
        con = duckdb.connect()
        con_rel = con.sql("select 42")
        print("BODY_DONE", flush=True)
        """,
        (),
        True,
        id="relations_alive_at_exit",
    ),
    pytest.param(
        """
        import duckdb
        import pandas as pd
        import pyarrow as pa

        tbl = pa.table({"a": [1, 2, 3]})
        con = duckdb.connect()
        arrow_rel = con.from_arrow(tbl)
        default_rel = duckdb.sql("select * from tbl")
        con.register("v", pd.DataFrame({"a": [1, 2, 3]}))
        pandas_rel = con.sql("select * from v")
        print("BODY_DONE", flush=True)
        """,
        (),
        True,
        id="relations_over_python_objects_alive_at_exit",
        marks=[_needs("pyarrow"), _needs("pandas")],
    ),
    pytest.param(
        """
        import io
        import duckdb

        rel = duckdb.read_csv(io.BytesIO(b"a\\n1\\n2\\n"))
        assert rel.fetchall() == [(1,), (2,)]
        con = duckdb.connect()
        con_rel = con.read_csv(io.BytesIO(b"a\\n1\\n2\\n"))
        assert con_rel.fetchall() == [(1,), (2,)]
        print("BODY_DONE", flush=True)
        """,
        (),
        True,
        id="file_like_relations_alive_at_exit",
        marks=_needs("fsspec"),
    ),
    pytest.param(
        """
        import threading
        import time
        import duckdb

        con = duckdb.connect()

        def work():
            cursor = con.cursor()
            while True:
                cursor.execute("select sum(range) from range(200000)").fetchall()

        for _ in range(4):
            threading.Thread(target=work, daemon=True).start()
        time.sleep(0.5)
        print("BODY_DONE", flush=True)
        """,
        (),
        False,
        id="daemon_threads_querying_at_exit",
        marks=pytest.mark.skipif(
            sys.version_info < (3, 13),
            reason="before 3.13 a daemon thread woken after finalization unwinds through duckdb and aborts (1.5.6 too)",
        ),
    ),
    pytest.param(
        """
        import atexit
        import threading
        import duckdb
        import pyarrow as pa

        tbl = pa.table({"a": [1, 2, 3]})
        finished = []

        def work():
            con = duckdb.connect()
            for _ in range(50):
                assert con.execute("select sum(a) from tbl").fetchall() == [(6,)]
            finished.append(True)

        atexit.register(lambda: print("finished workers", len(finished), flush=True))
        for _ in range(4):
            threading.Thread(target=work).start()
        print("BODY_DONE", flush=True)
        """,
        ("finished workers 4",),
        True,
        id="threads_finishing_during_shutdown",
        marks=_needs("pyarrow"),
    ),
    pytest.param(
        """
        import duckdb
        import numpy as np
        import pandas as pd

        column = pd.concat(
            [
                pd.Series([f"v{i}" for i in range(3000)], dtype="string[pyarrow]"),
                pd.Series([None, "x"] * 2000, dtype="string[pyarrow]"),
            ],
            ignore_index=True,
        )
        assert column.array._pa_array.num_chunks == 2
        frame = pd.DataFrame({"s": column, "i": np.arange(len(column), dtype=np.int64)})
        con = duckdb.connect()
        con.register("f", frame)
        rel = con.sql("SELECT s, i FROM f")
        result = con.execute("SELECT s FROM f")
        assert result.fetchmany(10) == [(f"v{i}",) for i in range(10)]
        default_rel = duckdb.sql("SELECT s FROM frame")
        print("BODY_DONE", flush=True)
        """,
        (),
        True,
        id="two_chunk_str_frame_with_open_results_at_exit",
        marks=[_needs("pyarrow"), _needs("pandas")],
    ),
    pytest.param(
        """
        import signal
        import duckdb
        import numpy as np
        import pandas as pd

        column = pd.concat(
            [
                pd.Series([f"v{i}" for i in range(300000)], dtype="string[pyarrow]"),
                pd.Series([None, "x"] * 150000, dtype="string[pyarrow]"),
            ],
            ignore_index=True,
        )
        assert column.array._pa_array.num_chunks == 2
        frame = pd.DataFrame({"s": column, "i": np.arange(len(column), dtype=np.int64)})
        con = duckdb.connect()
        con.register("f", frame)
        rel = con.sql("SELECT s, i FROM f")
        result = con.execute("SELECT s FROM f")
        assert result.fetchmany(10) == [(f"v{i}",) for i in range(10)]
        # The alarm ends a process that hangs at exit well before the subprocess timeout.
        if hasattr(signal, "alarm"):
            signal.alarm(20)
        print("BODY_DONE", flush=True)
        """,
        (),
        True,
        id="two_chunk_str_frame_with_a_running_scan_at_exit",
        marks=[_needs("pyarrow"), _needs("pandas")],
    ),
    pytest.param(
        """
        import signal
        import duckdb
        import numpy as np
        import pandas as pd
        import pyarrow as pa

        chunks = [pa.array([f"v{i}" for i in range(start, start + 50)]) for start in range(0, 600000, 50)]
        column = pd.Series(pd.arrays.ArrowStringArray(pa.chunked_array(chunks)))
        assert column.dtype == "string[pyarrow]"
        assert column.array._pa_array.num_chunks == 12000
        frame = pd.DataFrame({"s": column, "i": np.arange(len(column), dtype=np.int64)})
        con = duckdb.connect()
        con.register("f", frame)
        rel = con.sql("SELECT s, i FROM f")
        result = con.execute("SELECT s FROM f")
        assert result.fetchmany(10) == [(f"v{i}",) for i in range(10)]
        # The alarm ends a process that hangs at exit well before the subprocess timeout.
        if hasattr(signal, "alarm"):
            signal.alarm(20)
        print("BODY_DONE", flush=True)
        """,
        (),
        True,
        id="short_chunk_str_frame_with_a_running_scan_at_exit",
        marks=[_needs("pandas"), _needs("pyarrow")],
    ),
    pytest.param(
        """
        import duckdb
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc

        numbers = pa.array(np.arange(1_000_003, dtype=np.int64))
        tbl = pa.table({"a": numbers, "s": pc.cast(numbers, pa.string())})
        assert tbl.column("a").num_chunks == 1
        assert tbl.column("s").num_chunks == 1
        con = duckdb.connect(config={"threads": 4})
        con.register("t", tbl)
        rel = con.sql("SELECT a, s FROM t")
        result = con.execute("SELECT a, s FROM t")
        assert result.fetchmany(10) == [(i, str(i)) for i in range(10)]
        print("BODY_DONE", flush=True)
        """,
        (),
        True,
        id="one_chunk_arrow_table_with_open_results_at_exit",
        marks=_needs("pyarrow"),
    ),
    pytest.param(
        """
        import threading
        import time
        import duckdb
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc

        numbers = pa.array(np.arange(1_000_003, dtype=np.int64))
        tbl = pa.table({"a": numbers, "s": pc.cast(numbers, pa.string())})
        assert tbl.column("a").num_chunks == 1
        con = duckdb.connect(config={"threads": 4})

        def work():
            cursor = con.cursor()
            while True:
                cursor.execute("select count(s), min(a), max(s) from tbl").fetchall()

        for _ in range(4):
            threading.Thread(target=work, daemon=True).start()
        time.sleep(0.5)
        print("BODY_DONE", flush=True)
        """,
        (),
        False,
        id="daemon_threads_scanning_one_chunk_arrow_table_at_exit",
        marks=[
            _needs("pyarrow"),
            pytest.mark.skipif(
                sys.version_info < (3, 13),
                reason="before 3.13 a daemon thread woken after finalization unwinds through duckdb and aborts",
            ),
        ],
    ),
    *[
        pytest.param(
            """
            import threading
            import time
            import duckdb
            import numpy as np
            import pandas as pd

            frame = pd.DataFrame({"a": COLUMN})
            con = duckdb.connect()

            def work():
                cursor = con.cursor()
                while True:
                    cursor.execute("select count(a), min(a) from frame").fetchall()

            for _ in range(4):
                threading.Thread(target=work, daemon=True).start()
            time.sleep(0.5)
            print("BODY_DONE", flush=True)
            """.replace("COLUMN", column),
            (),
            False,
            id=f"daemon_threads_scanning_{kind}_frame_at_exit",
            marks=[
                *needs,
                pytest.mark.skipif(
                    sys.version_info < (3, 13),
                    reason="before 3.13 a daemon thread woken after finalization unwinds through duckdb and aborts",
                ),
            ],
        )
        for kind, column, needs in (
            ("numeric", "np.arange(200000, dtype=np.int64)", [_needs("pandas")]),
            ("object", 'pd.Series([f"v{i % 100}" for i in range(200000)], dtype=object)', [_needs("pandas")]),
            (
                "pyarrow_str",
                'pd.Series([f"v{i % 100}" for i in range(200000)], dtype="string[pyarrow]")',
                [_needs("pandas"), _needs("pyarrow")],
            ),
        )
    ],
]


@pytest.mark.parametrize(("script", "expected_lines", "leak_free"), ORDINARY_SHUTDOWN)
def test_ordinary_shutdown(script: str, expected_lines: tuple[str, ...], *, leak_free: bool) -> None:
    _assert_clean_exit(_run(script), expected_lines, leak_free=leak_free)


def test_ordinary_shutdown_checkpoints_default_connection(tmp_path) -> None:
    database = tmp_path / "exit.db"
    script = """
        import sys
        import duckdb

        duckdb.set_default_connection(duckdb.connect(sys.argv[1]))
        duckdb.sql("create table t as select range as a from range(3)")
        duckdb.sql("insert into t values (3)")
        print("BODY_DONE", flush=True)
    """
    _assert_clean_exit(_run(script, str(database)), (), leak_free=True)
    assert database.exists()
    assert not database.with_name(database.name + ".wal").exists()
    reopen = """
        import sys
        import duckdb

        print(duckdb.connect(sys.argv[1]).sql("from t order by a").fetchall())
        print("BODY_DONE", flush=True)
    """
    _assert_clean_exit(_run(reopen, str(database)), ("[(0,), (1,), (2,), (3,)]",), leak_free=True)


@pytest.mark.skipif(sys.platform == "win32", reason="ctypes cannot load the C library through a null name on Windows")
@pytest.mark.parametrize("library", [pytest.param("PyDLL", id="gil_held"), pytest.param("CDLL", id="gil_released")])
def test_c_exit_keeps_committed_data(tmp_path, library: str) -> None:
    database = str(tmp_path / "exit.db")
    script = f"""
        import ctypes
        import sys
        import duckdb

        duckdb.set_default_connection(duckdb.connect(sys.argv[1]))
        duckdb.sql("create table t as select range as a from range(3)")
        print("BODY_DONE", flush=True)
        ctypes.{library}(None).exit(0)
    """
    _assert_clean_exit(_run(script, database), (), leak_free=True)
    reopen = """
        import sys
        import duckdb

        print(duckdb.connect(sys.argv[1]).sql("from t order by a").fetchall())
        print("BODY_DONE", flush=True)
    """
    _assert_clean_exit(_run(reopen, database), ("[(0,), (1,), (2,)]",), leak_free=True)

"""Regression tests for duckdb-python#435 and #456.

These reproducers crash the interpreter (SIGSEGV / heap corruption / abort)
when the underlying bug is triggered, so each test runs the workload in a
fresh Python subprocess and asserts on the exit code. A non-zero exit (in
particular -11 / 139 / 134 / -6) is treated as the bug reproducing.

The workloads also report a `success_count` to stdout, and the assertions
verify it crossed a sensible floor — without that, a subprocess that
silently errored on every iteration (e.g. import failure, API change)
would still satisfy a "no crash" check and we would lose the signal we
care about.

These tests spawn subprocesses, so they live under tests/slow rather than
tests/fast.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
import textwrap

import pytest

_SUCCESS_RE = re.compile(rb"^success_count=(\d+)\r?$", re.MULTILINE)


def _run(script: str, timeout: float) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        timeout=timeout,
    )


def _assert_no_crash_and_min_successes(
    result: subprocess.CompletedProcess[bytes],
    min_successes: int,
) -> None:
    if result.returncode != 0:
        stdout = result.stdout.decode(errors="replace")[-2000:]
        stderr = result.stderr.decode(errors="replace")[-2000:]
        msg = (
            f"subprocess exited with returncode={result.returncode} "
            f"(negative = killed by signal on POSIX; 139=SIGSEGV, 134=SIGABRT)\n"
            f"--- stdout (tail) ---\n{stdout}\n"
            f"--- stderr (tail) ---\n{stderr}"
        )
        raise AssertionError(msg)
    match = _SUCCESS_RE.search(result.stdout)
    if match is None:
        stdout = result.stdout.decode(errors="replace")[-2000:]
        msg = (
            "subprocess did not report success_count — the workload may have "
            "errored on every iteration without crashing, which would mask "
            "the bug we are guarding against.\n"
            f"--- stdout (tail) ---\n{stdout}"
        )
        raise AssertionError(msg)
    success_count = int(match.group(1))
    assert success_count >= min_successes, (
        f"subprocess only had success_count={success_count} (need >= "
        f"{min_successes}). The workload is not exercising the code path the "
        f"test is supposed to guard."
    )


# ---------------------------------------------------------------------------
# #435 — concurrent execute/fetchall on the same connection
# ---------------------------------------------------------------------------
#
# Both variants reliably segfault pre-fix (returncode -11 on macOS, 139 on
# Linux). After the fix, every iteration should succeed.
#

_REPRO_435_SINGLE_CONN = textwrap.dedent(
    """
    import concurrent.futures
    import duckdb

    # One shared connection, many threads. ThreadPoolExecutor(8) means up to
    # 8 worker threads concurrently hit the same DuckDBPyConnection's result
    # slot via execute() + fetchall().
    conn = duckdb.connect()
    successes = 0

    def run(_):
        conn.execute('SELECT 1').fetchall()

    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for f in [ex.submit(run, i) for i in range(200)]:
            try:
                f.result()
                successes += 1
            except Exception:
                # A clean Python-level exception is acceptable — the bug we
                # test for is an interpreter crash. Per-iteration failures
                # still count against `success_count` so the test asserts
                # that the workload actually ran.
                pass

    print(f"success_count={successes}")
    """
).lstrip()


_REPRO_435_MULTI_CONN = textwrap.dedent(
    """
    # Exact reproducer from issue #435.
    import concurrent.futures
    import duckdb

    conns = [duckdb.connect() for _ in range(8)]
    successes = 0

    def run(i):
        conns[i % 8].execute('SELECT 1').fetchall()

    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for f in [ex.submit(run, i) for i in range(200)]:
            try:
                f.result()
                successes += 1
            except Exception:
                pass

    print(f"success_count={successes}")
    """
).lstrip()


def test_435_concurrent_execute_fetchall_single_connection():
    """Eight threads sharing one connection must not crash the interpreter.

    Pre-fix: SIGSEGV. Post-fix: all 200 iterations should succeed because
    the connection lock serialises them.
    """
    result = _run(_REPRO_435_SINGLE_CONN, timeout=30.0)
    _assert_no_crash_and_min_successes(result, min_successes=200)


def test_435_concurrent_execute_fetchall_multi_connection():
    """Reporter's 8-connection / 200-task pattern must not crash.

    Pre-fix: SIGSEGV. The race is per-connection on the result slot;
    multiple workers landing on the same ``conns[i % 8]`` trigger it.
    """
    result = _run(_REPRO_435_MULTI_CONN, timeout=30.0)
    _assert_no_crash_and_min_successes(result, min_successes=200)


# ---------------------------------------------------------------------------
# #456 — DuckDBPyResult destructor must hold the GIL
# ---------------------------------------------------------------------------
#
# Reporter sees hard crashes on Windows + Python 3.12 under heavy executemany
# workloads with Python primitives flowing into the result graph. The fix
# removes the unconditional `py::gil_scoped_release` from
# `DuckDBPyResult::~DuckDBPyResult` (and tightens the same scope in
# `DuckDBPyConnection::Close` so the `case_insensitive_map_t<unique_ptr<ExternalDependency>>`
# is destroyed with the GIL held).
#
# These tests are skipped off-Windows. We verified empirically by
# temp-reverting the fix and rebuilding that the workloads below do NOT
# crash on macOS/Linux even with the destructor's GIL release restored —
# `Py_DECREF` without the GIL is undefined behaviour but POSIX CPython
# rarely faults on it deterministically (different thread-state storage /
# allocator behaviour from Windows). Running these tests off-Windows
# would always pass regardless of whether the fix is in place, so they
# would be a false signal rather than a regression guard.
#
# On Windows + Python 3.12 (the reporter's environment) the same
# workloads are expected to crash pre-fix and pass post-fix, which is
# what a regression test should do.
#
# Correctness of the fix on POSIX is established by inspection rather
# than by these tests: the destructor's `D_ASSERT(py::gil_check())` already
# requires the caller to hold the GIL on entry, and the previous
# `py::gil_scoped_release` was the only thing dropping it during
# destruction of pybind-managed members.
#

_skip_456_off_windows = pytest.mark.skipif(
    platform.system() != "Windows",
    reason=(
        "Issue #456 crashes only on Windows CPython (Py_DECREF without the GIL "
        "is UB but POSIX CPython does not consistently fault on it). Verified "
        "empirically that this workload does not crash on macOS even with the "
        "destructor's GIL release restored — running it off-Windows would "
        "always pass and provide no regression signal. Correctness on POSIX "
        "is established by code inspection."
    ),
)


_REPRO_456_UDF_DESTRUCTOR = textwrap.dedent(
    """
    # Stress the result destructor on multiple threads concurrently, with
    # Python str references in the result graph (via a UDF). Each worker
    # has its OWN connection, so the connection lock from #435 does not
    # serialise them — we want them genuinely concurrent so that one
    # thread's destructor (releasing the GIL pre-fix) overlaps with
    # another thread's Python work / allocation.
    import concurrent.futures
    import duckdb
    from duckdb.sqltypes import INTEGER, VARCHAR

    def make_label(i):
        return f"label-{i}-padded-out-a-bit-to-defeat-small-string-optimisation"

    def worker(_):
        conn = duckdb.connect()
        conn.create_function('make_label', make_label, [INTEGER], VARCHAR)
        rows = 4 * 1024
        local_successes = 0
        for _ in range(50):
            n = len(conn.execute(f'SELECT make_label(i::INTEGER) FROM range({rows}) t(i)').fetchall())
            if n == rows:
                local_successes += 1
        return local_successes

    successes = 0
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for f in [ex.submit(worker, i) for i in range(16)]:
            try:
                successes += f.result()
            except Exception:
                pass

    print(f"success_count={successes}")
    """
).lstrip()


_REPRO_456_CLOSE_REGISTERED = textwrap.dedent(
    """
    # Stress the Close() path that destroys `registered_functions`. Each
    # connection has a UDF registered, so closing it tears down the
    # `case_insensitive_map_t<unique_ptr<ExternalDependency>>` which
    # transitively owns the Python callable. This must happen with the GIL
    # held — see duckdb-python#456.
    import duckdb
    from duckdb.sqltypes import INTEGER

    def identity(x):
        return x

    successes = 0
    for _ in range(500):
        conn = duckdb.connect()
        conn.create_function('identity', identity, [INTEGER], INTEGER)
        # Make sure the UDF actually got bound to the connection.
        ok = conn.execute('SELECT identity(7)').fetchone()[0] == 7
        conn.close()  # triggers registered_functions.clear()
        if ok:
            successes += 1

    print(f"success_count={successes}")
    """
).lstrip()


@_skip_456_off_windows
def test_456_destructor_with_python_refs_in_result_graph():
    """DuckDBPyResult destructor must not crash with Python refs in chunks.

    Repeated destruction where the result chunks hold Python ``str``
    references (via a UDF), run on 8 threads concurrently with
    independent connections. Pre-fix Windows + Python 3.12: crash.
    Post-fix Windows: clean exit, all 16*50=800 iterations succeed.
    """
    result = _run(_REPRO_456_UDF_DESTRUCTOR, timeout=120.0)
    _assert_no_crash_and_min_successes(result, min_successes=800)


# ---------------------------------------------------------------------------
# #638 — segfault at interpreter exit when duckdb left sys.modules
# ---------------------------------------------------------------------------
#
# Removing the module from sys.modules means its dict is never cleared, so
# the _clean_default_connection capsule never fires, Cleanup() never runs
# while the interpreter is alive, and the process-global module state
# (holding live Python references) is torn down after finalization.
# Pre-fix on Linux: SIGSEGV (returncode 139). Post-fix: clean exit 0.
#

_REPRO_638_MODULE_REMOVED_FROM_SYS_MODULES = textwrap.dedent(
    """
    import gc
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules):
        import duckdb


    def holder():
        pass


    holder.module = duckdb
    gc.freeze()
    print("exiting", duckdb.__version__, flush=True)
    """
).lstrip()


@_skip_456_off_windows
def test_638_no_segfault_when_module_removed_from_sys_modules():
    """Interpreter must exit 0 when duckdb outlives shutdown outside sys.modules.

    Runs the issue's repro in a subprocess: the module is imported under a
    patched sys.modules (so it is absent afterwards) but kept alive by a
    stray reference, then the interpreter exits. Pre-fix: returncode 139
    (SIGSEGV). Post-fix: 0.
    """
    result = _run(_REPRO_638_MODULE_REMOVED_FROM_SYS_MODULES, timeout=60.0)
    assert result.returncode == 0, (
        f"interpreter crashed at exit with returncode={result.returncode} "
        f"(139/-11 = SIGSEGV, see duckdb-python#638)\n"
        f"--- stdout (tail) ---\n{result.stdout.decode(errors='replace')[-2000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr.decode(errors='replace')[-2000:]}"
    )


@_skip_456_off_windows
def test_456_close_clears_registered_functions_with_gil():
    """``close()`` must hold the GIL while clearing ``registered_functions``.

    Destroying those ``ExternalDependency`` entries transitively
    decrements pybind-owned Python references; doing so without the GIL
    is undefined behaviour.
    """
    result = _run(_REPRO_456_CLOSE_REGISTERED, timeout=60.0)
    _assert_no_crash_and_min_successes(result, min_successes=500)

"""A process that ends with an open result over a pandas object column must exit, not hang.

Each case runs a script in a fresh interpreter and checks that it exits within the deadline, with the script's own
exit status and no fatal error.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
import time

import pytest

pytest.importorskip("pandas")

ROWS = 1_000_000
DEADLINE = 60
BODY_DONE = "BODY_DONE"

FRAME = f"""
import duckdb
import pandas as pd

df = pd.DataFrame({{"s": pd.Series([f"v{{i}}" for i in range({ROWS})], dtype=object), "i": range({ROWS})}})
"""

DECIMAL_FRAME = f"""
import decimal
import duckdb
import pandas as pd

df = pd.DataFrame({{"d": pd.Series([decimal.Decimal(i) for i in range({ROWS})], dtype=object), "i": range({ROWS})}})
"""


def _run(body: str, *, head: str = FRAME, timeout: float = DEADLINE) -> subprocess.CompletedProcess[str]:
    script = textwrap.dedent(head) + textwrap.dedent(body) + f"\nprint({BODY_DONE!r}, flush=True)\n"
    try:
        return subprocess.run(
            [sys.executable, "-u", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={"SKBUILD_EDITABLE_VERBOSE": "0", **os.environ},
        )
    except subprocess.TimeoutExpired as hang:
        stdout = hang.stdout.decode() if isinstance(hang.stdout, bytes) else hang.stdout or ""
        stderr = hang.stderr.decode() if isinstance(hang.stderr, bytes) else hang.stderr or ""
        pytest.fail(f"the process hung at exit\nstdout:\n{stdout[-2000:]}\nstderr:\n{stderr[-2000:]}")


def _assert_exit(proc: subprocess.CompletedProcess[str], *, status: int = 0, body_done: bool = True) -> None:
    report = f"exit status {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    assert proc.returncode == status, report
    assert (BODY_DONE in proc.stdout.splitlines()) == body_done, report
    assert "Fatal Python error" not in proc.stderr, report


OPEN_RESULT_CASES = [
    pytest.param(
        """
        con = duckdb.connect()
        print(con.execute("SELECT * FROM df").fetchone())
        """,
        id="dbapi_fetchone",
    ),
    pytest.param(
        """
        con = duckdb.connect()
        print(len(con.execute("SELECT * FROM df").fetchmany(10)))
        """,
        id="dbapi_fetchmany",
    ),
    pytest.param(
        """
        con = duckdb.connect()
        cur = con.cursor()
        print(cur.execute("SELECT * FROM df").fetchone())
        """,
        id="cursor",
    ),
    pytest.param(
        """
        print(duckdb.execute("SELECT * FROM df").fetchone())
        """,
        id="default_connection",
    ),
    pytest.param(
        """
        con = duckdb.connect()
        con.execute("SELECT * FROM df")
        print(len(con.fetch_df_chunk()))
        """,
        id="df_chunk",
    ),
    pytest.param(
        """
        con = duckdb.connect()
        capsule = con.sql("SELECT * FROM df").__arrow_c_stream__()
        print(type(capsule).__name__)
        """,
        id="relation_arrow_stream",
    ),
    pytest.param(
        """
        con = duckdb.connect()
        reader = con.sql("SELECT * FROM df").to_arrow_reader(1000)
        print(reader.read_next_batch().num_rows)
        """,
        id="arrow_reader",
        marks=pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="needs pyarrow"),
    ),
    pytest.param(
        """
        con = duckdb.connect(config={"threads": 4})
        print(con.execute("SELECT * FROM df WHERE i % 2 = 0").fetchone())
        """,
        id="filter_four_threads",
    ),
    pytest.param(
        """
        import sys
        con = duckdb.connect()
        print(con.execute("SELECT * FROM df").fetchone())
        print("BODY_DONE", flush=True)
        sys.exit(0)
        """,
        id="sys_exit",
    ),
]


class TestOpenResultAtExit:
    @pytest.mark.parametrize("body", OPEN_RESULT_CASES)
    def test_exits_with_an_open_result_over_an_object_column(self, body: str) -> None:
        _assert_exit(_run(body))

    def test_exits_with_an_open_result_over_decimal_cells(self) -> None:
        body = """
        con = duckdb.connect()
        print(con.execute("SELECT * FROM df").fetchone())
        """
        _assert_exit(_run(body, head=DECIMAL_FRAME))

    def test_an_exception_after_the_fetch_exits_with_status_one(self) -> None:
        body = """
        con = duckdb.connect()
        print(con.execute("SELECT * FROM df").fetchone())
        raise RuntimeError("boom")
        """
        proc = _run(body)
        _assert_exit(proc, status=1, body_done=False)
        assert "RuntimeError: boom" in proc.stderr

    @pytest.mark.skipif(
        sys.version_info < (3, 13),
        reason="before 3.13 a daemon thread woken after finalization unwinds through duckdb and aborts",
    )
    def test_a_result_open_on_a_busy_thread_does_not_block_the_exit(self) -> None:
        body = """
        import threading
        con = duckdb.connect()
        started = threading.Event()

        def work():
            cursor = con.cursor()
            cursor.execute("SELECT * FROM df")
            started.set()
            try:
                while True:
                    cursor.fetchmany(1)
            except duckdb.Error:
                pass

        threading.Thread(target=work, daemon=True).start()
        started.wait()
        """
        _assert_exit(_run(body))

    def test_a_sort_left_open_by_the_main_thread_is_interrupted_before_its_close(self) -> None:
        # Its workers check the interrupt and not the cancel, so without the interrupt the close waits for the sort
        body = """
        import time

        query = "SELECT * FROM range(3) t(i) UNION ALL (SELECT i FROM range(100_000_000) t(i) ORDER BY hash(i))"
        for _ in range(5):
            con = duckdb.connect(config={"threads": 4, "memory_limit": "2GB"})
            con.execute("SET enable_progress_bar = false")
            started = time.time()
            con.execute(query)
            print("FIRST", con.fetchone(), flush=True)
            # A slow first row means the fetching thread ran the sort itself and left nothing to interrupt
            if time.time() - started < 1:
                print("WORKERS_SORT", flush=True)
                break
            con.close()
        # Lets the engine workers take up the sort before the main thread returns
        time.sleep(1)
        print("MAIN_RETURNS", time.time(), flush=True)
        """
        proc = _run(body, head="import duckdb\n", timeout=120)
        exited = time.time()
        _assert_exit(proc)
        assert "WORKERS_SORT" in proc.stdout.splitlines(), proc.stdout
        returned = next(float(line.split()[1]) for line in proc.stdout.splitlines() if line.startswith("MAIN_RETURNS"))
        assert exited - returned < 2.5, f"exit took {exited - returned:.1f} s after the main thread"


NEEDS_FIFO = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")


class TestQueryRunningOnAnotherThread:
    # The hook's own bound is one second; the rest covers finalization and process teardown on a slow runner.
    EXIT_BOUND = 10

    def _exit_wait(self, fifo: os.PathLike[str], body: str) -> tuple[subprocess.CompletedProcess[str], float]:
        os.mkfifo(fifo)
        # Open and silent for as long as the child lives, so its read of the FIFO never returns
        writer = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        try:
            proc = _run(body, head="import duckdb\n", timeout=3 * self.EXIT_BOUND)
            exited = time.time()
        finally:
            os.close(writer)
        _assert_exit(proc)
        returned = next(float(line.split()[1]) for line in proc.stdout.splitlines() if line.startswith("MAIN_RETURNS"))
        return proc, exited - returned

    def _assert_bounded(self, wait: float) -> None:
        # Sooner than the hook's one second means nothing was stuck in the read when the process exited
        assert 1 <= wait < self.EXIT_BOUND, f"exit took {wait:.1f} s after the main thread"

    @NEEDS_FIFO
    @pytest.mark.parametrize(
        "threads",
        [
            # The daemon thread runs the blocked scan itself and holds the engine's context lock while it reads
            pytest.param(1, id="read_on_the_fetching_thread"),
            # An engine worker or the fetching thread runs the scan, whichever takes its one task first
            pytest.param(4, id="four_threads"),
        ],
    )
    def test_a_read_no_interrupt_reaches_does_not_hold_up_the_exit(self, tmp_path, threads: int) -> None:
        fifo = tmp_path / "silent.fifo"
        body = f"""
        import threading
        import time

        con = duckdb.connect(config={{"threads": {threads}}})
        con.execute("SET enable_progress_bar = false")
        query = "SELECT * FROM read_text('{fifo}')"

        def read():
            try:
                con.sql(query).fetchall()
            finally:
                print("READ_ENDED", flush=True)

        threading.Thread(target=read, daemon=True).start()
        time.sleep(1)
        print("MAIN_RETURNS", time.time(), flush=True)
        """
        proc, wait = self._exit_wait(fifo, body)
        assert "READ_ENDED" not in proc.stdout, proc.stdout
        self._assert_bounded(wait)

    def test_an_interruptible_query_on_another_thread_is_interrupted(self) -> None:
        # Registered before the import, so it runs after duckdb's exit hook and reports what the daemon thread saw
        handler = """
        import atexit
        import threading

        state = {"done": threading.Event()}

        def after():
            state["done"].wait(30)
            print("DAEMON", state.get("outcome"), flush=True)
            print("LATER", state["con"].execute("SELECT 42").fetchall(), flush=True)

        atexit.register(after)
        """
        body = """
        import time

        con = duckdb.connect(config={"threads": 1})
        con.execute("SET enable_progress_bar = false")
        state["con"] = con
        query = "SELECT count(*) FROM range(10_000_000_000_000) t(i) WHERE i % 7 = 3"
        started = threading.Event()

        def work():
            started.set()
            try:
                state["outcome"] = con.sql(query).fetchall()
            except duckdb.Error as error:
                state["outcome"] = type(error).__name__
            state["done"].set()

        threading.Thread(target=work, daemon=True).start()
        started.wait()
        time.sleep(0.5)
        """
        proc = _run(body, head=textwrap.dedent(handler) + "import duckdb\n")
        _assert_exit(proc)
        assert "DAEMON InterruptException" in proc.stdout.splitlines(), proc.stdout
        assert "LATER [(42,)]" in proc.stdout.splitlines(), proc.stdout

    @NEEDS_FIFO
    def test_reads_stuck_on_several_connections_share_the_hooks_one_second(self, tmp_path) -> None:
        fifo = tmp_path / "silent.fifo"
        body = f"""
        import threading
        import time

        def read():
            con = duckdb.connect(config={{"threads": 1}})
            con.execute("SET enable_progress_bar = false")
            con.sql("SELECT * FROM read_text('{fifo}')").fetchall()

        # More connections than the class's bound has seconds, so a bound per result cannot pass
        for _ in range(12):
            threading.Thread(target=read, daemon=True).start()
        time.sleep(1)
        print("MAIN_RETURNS", time.time(), flush=True)
        """
        _, wait = self._exit_wait(fifo, body)
        self._assert_bounded(wait)


class TestAfterTheHook:
    def test_a_later_exit_handler_still_runs_a_statement(self) -> None:
        # Registered before the import, so it runs after duckdb's exit hook
        handler = """
        import atexit

        state = {}

        def after():
            print("AFTER", state["con"].execute("SELECT count(*) FROM df").fetchone(), flush=True)

        atexit.register(after)
        """
        head = textwrap.dedent(handler) + FRAME
        body = """
        con = duckdb.connect()
        state["con"] = con
        print(con.execute("SELECT * FROM df").fetchone())
        """
        proc = _run(body, head=head)
        _assert_exit(proc)
        assert f"AFTER ({ROWS},)" in proc.stdout, proc.stdout

    def test_a_later_exit_handler_converts_object_cells_on_worker_threads(self) -> None:
        # Registered before the import, so it runs after duckdb's exit hook; four threads read the frame's ten parts.
        handler = """
        import atexit

        state = {}

        def after():
            print("AFTER", state["con"].execute("SELECT count(s), min(s) FROM df").fetchone(), flush=True)

        atexit.register(after)
        """
        body = """
        con = duckdb.connect(config={"threads": 4})
        state["con"] = con
        """
        proc = _run(body, head=textwrap.dedent(handler) + FRAME)
        _assert_exit(proc)
        assert f"AFTER ({ROWS}, 'v0')" in proc.stdout, proc.stdout

    def test_a_query_a_daemon_thread_starts_after_the_hook_is_refused(self) -> None:
        # The exit handler runs after duckdb's exit hook and waits while the daemon thread queries.
        handler = """
        import atexit
        import threading

        state = {"go": threading.Event(), "done": threading.Event()}

        def after():
            state["go"].set()
            state["done"].wait(30)
            print("DAEMON", state.get("outcome"), flush=True)

        atexit.register(after)
        """
        body = """
        con = duckdb.connect(config={"threads": 4})

        def query_after_the_hook():
            state["go"].wait()
            try:
                state["outcome"] = con.cursor().execute("SELECT count(s), min(s) FROM df").fetchone()
            except duckdb.Error as error:
                state["outcome"] = f"{type(error).__name__}: {error}"
            state["done"].set()

        threading.Thread(target=query_after_the_hook, daemon=True).start()
        """
        proc = _run(body, head=textwrap.dedent(handler) + FRAME)
        _assert_exit(proc)
        outcome = next((line for line in proc.stdout.splitlines() if line.startswith("DAEMON ")), "")
        assert outcome.startswith("DAEMON InvalidInputException: "), proc.stdout
        assert outcome.endswith("the Python interpreter is shutting down"), proc.stdout

    @pytest.mark.parametrize(
        "left_open",
        [
            pytest.param("", id="nothing_running"),
            pytest.param(
                """
                con.execute("SELECT * FROM range(100_000_000)")
                print("FIRST", con.fetchone(), flush=True)
                """,
                id="streaming_result",
            ),
        ],
    )
    def test_a_later_exit_handler_binds_inside_an_open_transaction(self, left_open: str) -> None:
        # Registered before the import, so it runs after duckdb's exit hook; inside a transaction the engine clears an
        # interrupt only when a statement starts, and binding the macro checks it before that
        handler = """
        import atexit

        state = {}

        def after():
            print("LATER", state["con"].sql("SELECT answer()").fetchall(), flush=True)

        atexit.register(after)
        """
        body = """
        con = duckdb.connect(config={"threads": 1})
        state["con"] = con
        con.execute("CREATE MACRO answer() AS 42")
        con.execute("BEGIN")
        """
        proc = _run(
            textwrap.dedent(body) + textwrap.dedent(left_open), head=textwrap.dedent(handler) + "import duckdb\n"
        )
        _assert_exit(proc)
        assert "LATER [(42,)]" in proc.stdout.splitlines(), proc.stdout


class TestUnchangedExits:
    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(
                """
                con = duckdb.connect()
                print(con.execute("SELECT * FROM df").fetchone())
                con.close()
                """,
                id="closed_before_exit",
            ),
            pytest.param(
                """
                con = duckdb.connect()
                print(len(con.execute("SELECT * FROM df").fetchall()))
                """,
                id="fetched_to_the_end",
            ),
            pytest.param(
                """
                def read():
                    con = duckdb.connect()
                    return con.execute("SELECT * FROM df").fetchone()

                print(read())
                """,
                id="connection_in_function",
            ),
        ],
    )
    def test_exits_as_before(self, body: str) -> None:
        _assert_exit(_run(body))

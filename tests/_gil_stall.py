"""Measures how much a scan stalls when a Python UDF holds the GIL at its first row.

A scan that takes the GIL for every vector waits out the hold on every thread but the one that holds it; a scan that
reads without the GIL does not. The profiler's TABLE_SCAN timing is wall time summed over the threads that ran the
scan, so it grows by the time those threads waited.
"""

import ctypes
import json

import duckdb

HOLD_SECONDS = 0.2
ROUNDS = 5


def scan_seconds(source, held_row, threads):
    """Summed TABLE_SCAN seconds of one scan of the source, and how many times the UDF ran.

    The UDF holds the GIL at the row whose ``i`` equals ``held_row``; a row the source does not have, such as -1,
    gives a free run of the same query.
    """
    libc = ctypes.PyDLL(None)
    libc.usleep.argtypes = [ctypes.c_uint]
    calls = [0]

    # A call through PyDLL keeps the GIL for the whole sleep.
    def hold_gil(value):
        calls[0] += 1
        libc.usleep(int(HOLD_SECONDS * 1_000_000))
        return value

    with duckdb.connect(config={"threads": threads}) as con:
        con.create_function("hold_gil", hold_gil, ["BIGINT"], "BIGINT")
        con.register("src", source)
        con.execute("PRAGMA enable_profiling = 'no_output'")
        con.execute(f"SELECT count(x), sum(CASE WHEN i = {held_row} THEN hold_gil(i) ELSE i END) FROM src").fetchall()
        operators = list(json.loads(con.get_profiling_information())["operator"])
    seconds = 0.0
    while operators:
        operator = operators.pop()
        if operator["type"] == "TABLE_SCAN":
            seconds += operator["timing"]
        operators.extend(operator.get("children", []))
    return seconds, calls[0]


class ScanTimings:
    """Held and free scan times of one source, one of each per round, and the UDF calls each pair of runs made."""

    def __init__(self, threads) -> None:
        self.threads = threads
        self.held = []
        self.free = []
        self.calls = set()

    def add_round(self, source):
        held, held_calls = scan_seconds(source, 0, self.threads)
        free, free_calls = scan_seconds(source, -1, self.threads)
        self.held.append(held)
        self.free.append(free)
        self.calls.add((held_calls, free_calls))

    def growth_in_holds(self):
        """The fastest held run minus the fastest free run, for bounding a stall from above.

        Load adds time to any run, so each kind counts at its fastest.
        """
        return (min(self.held) - min(self.free)) / HOLD_SECONDS

    def best_round_in_holds(self):
        """The largest held minus free difference of one round, for bounding a stall from below.

        One round that shows the stall is enough; load can take stall away from a held run when fewer threads wait
        out the hold, so the fastest held run would understate it.
        """
        return max(held - free for held, free in zip(self.held, self.free, strict=True)) / HOLD_SECONDS

"""Tests that Arrow streams remain valid after their originating connection is destroyed.

The Arrow PyCapsule paths produce lazy streams — schema and data are consumed
later.  If the stream wrapper holds only a non-owning pointer to the
ClientContext and the connection is GC'd in between, the pointer dangles and we
crash (mutex-lock-on-destroyed-object).

Each test creates a capsule from a short-lived connection, destroys that
connection, then consumes the capsule from a *different* connection.
"""

import gc

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")

EXPECTED = [(i, i + 1, -i) for i in range(100)]
SQL = "SELECT i, i + 1 AS j, -i AS k FROM range(100) t(i)"


class TestArrowConnectionLifetime:
    """Capsules must stay valid after the originating connection is destroyed."""

    def test_capsule_fast_path_survives_connection_gc(self):
        """__arrow_c_stream__ fast path (ArrowQueryResult): connection destroyed before capsule is consumed."""
        conn = duckdb.connect()
        capsule = conn.sql(SQL).__arrow_c_stream__()  # noqa: F841
        del conn
        gc.collect()
        result = duckdb.connect().sql("SELECT * FROM capsule").fetchall()
        assert result == EXPECTED

    def test_materialized_capsule_survives_connection_gc(self):
        """A capsule over a materialized relation: connection destroyed before the capsule is consumed."""
        conn = duckdb.connect()
        rel = conn.sql(SQL).materialize()
        capsule = rel.__arrow_c_stream__()  # noqa: F841
        del rel, conn
        gc.collect()
        result = duckdb.connect().sql("SELECT * FROM capsule").fetchall()
        assert result == EXPECTED

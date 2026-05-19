"""Tests for subcursor() — a cursor that shares the same connection and transaction."""

import pytest
import duckdb


class TestSubcursor:
    def test_subcursor_basic_interleaved(self):
        """Single scan + update pattern with subcursor."""
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t AS SELECT i AS id, false AS processed FROM range(100000) tbl(i)")
        con.execute("SET enable_suspended_queries = true")
        con.execute("BEGIN TRANSACTION")

        scanner = con.subcursor()
        updater = con.subcursor()

        result = scanner.execute("SELECT id FROM t WHERE NOT processed ORDER BY id")
        batch = result.fetchmany(1000)
        assert len(batch) == 1000
        assert batch[0][0] == 0

        # Update some rows via the subcursor
        ids = ",".join(str(r[0]) for r in batch[:10])
        updater.execute(f"UPDATE t SET processed = true WHERE id IN ({ids})")

        # Resume fetching from the scanner — should still work
        batch2 = result.fetchmany(1000)
        assert len(batch2) > 0

        con.execute("COMMIT")

        # Verify updates persisted
        res = con.execute("SELECT COUNT(*) FROM t WHERE processed").fetchone()
        assert res[0] == 10

    def test_subcursor_multi_table(self):
        """Multi-table scan pattern with subcursors."""
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t1 AS SELECT i AS id FROM range(100000) tbl(i)")
        con.execute("CREATE TABLE t2 AS SELECT i AS id FROM range(100000) tbl(i)")
        con.execute("SET enable_suspended_queries = true")
        con.execute("BEGIN TRANSACTION")

        scan_a = con.subcursor()
        scan_b = con.subcursor()
        updater = con.subcursor()

        result_a = scan_a.execute("SELECT id FROM t1 ORDER BY id")
        result_b = scan_b.execute("SELECT id FROM t2 ORDER BY id")

        # Interleave fetches from both tables
        batch_a = result_a.fetchmany(100)
        assert len(batch_a) == 100

        batch_b = result_b.fetchmany(100)
        assert len(batch_b) == 100

        # Update via subcursor
        updater.execute("UPDATE t1 SET id = id + 1000000 WHERE id < 5")

        # Resume both scans
        batch_a2 = result_a.fetchmany(100)
        assert len(batch_a2) > 0

        batch_b2 = result_b.fetchmany(100)
        assert len(batch_b2) > 0

        con.execute("COMMIT")

    def test_subcursor_transaction_guard(self):
        """Subcursor cannot manage transactions."""
        con = duckdb.connect(":memory:")
        sub = con.subcursor()

        with pytest.raises(duckdb.InvalidInputException, match="subcursor"):
            sub.begin()

        with pytest.raises(duckdb.InvalidInputException, match="subcursor"):
            sub.commit()

        with pytest.raises(duckdb.InvalidInputException, match="subcursor"):
            sub.rollback()

    def test_subcursor_shares_transaction(self):
        """Subcursor sees data from the same transaction."""
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t (id INTEGER)")
        con.execute("BEGIN TRANSACTION")
        con.execute("INSERT INTO t VALUES (1), (2), (3)")

        sub = con.subcursor()
        # Subcursor should see uncommitted data from the parent's transaction
        res = sub.execute("SELECT COUNT(*) FROM t").fetchone()
        assert res[0] == 3

        con.execute("COMMIT")

    def test_subcursor_close(self):
        """Subcursor close doesn't destroy parent connection."""
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t AS SELECT 1 AS id")

        sub = con.subcursor()
        sub.execute("SELECT * FROM t")
        sub.close()

        # Parent connection should still work
        res = con.execute("SELECT * FROM t").fetchone()
        assert res[0] == 1

    def test_subcursor_full_consumption(self):
        """Verify stream can be fully consumed after suspend/resume."""
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t AS SELECT i AS id FROM range(100000) tbl(i)")
        con.execute("SET enable_suspended_queries = true")
        con.execute("BEGIN TRANSACTION")

        scanner = con.subcursor()
        updater = con.subcursor()

        result = scanner.execute("SELECT id FROM t ORDER BY id")
        total = 0

        while True:
            batch = result.fetchmany(2048)
            if not batch:
                break
            total += len(batch)
            # Interleave an update every few batches
            if total % 8192 == 0:
                updater.execute(f"UPDATE t SET id = id WHERE id = {total}")

        assert total == 100000
        con.execute("COMMIT")

    def test_subcursor_requires_explicit_transaction(self):
        """Subcursor interleaving requires an explicit transaction (BEGIN)."""
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t AS SELECT i AS id FROM range(100000) tbl(i)")

        # With explicit transaction and setting enabled, interleaving works
        con.execute("SET enable_suspended_queries = true")
        con.execute("BEGIN TRANSACTION")
        scanner = con.subcursor()
        updater = con.subcursor()

        result = scanner.execute("SELECT id FROM t ORDER BY id")
        batch = result.fetchmany(1000)
        assert len(batch) == 1000

        updater.execute("UPDATE t SET id = id WHERE id < 5")

        # Stream survives the update in explicit transaction mode
        batch2 = result.fetchmany(1000)
        assert len(batch2) > 0

        con.execute("COMMIT")

    def test_subcursor_close_cancels_suspended_stream(self):
        """Closing a subcursor cancels its suspended stream and frees resources."""
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t AS SELECT i AS id FROM range(100000) tbl(i)")
        con.execute("SET enable_suspended_queries = true")
        con.execute("BEGIN TRANSACTION")

        scanner = con.subcursor()
        updater = con.subcursor()

        result = scanner.execute("SELECT id FROM t ORDER BY id")
        batch = result.fetchmany(1000)
        assert len(batch) == 1000

        # Suspend the stream by executing on the updater
        updater.execute("UPDATE t SET id = id WHERE id < 5")

        # Close the scanner — should cancel the suspended stream
        scanner.close()

        # The connection should still work normally
        updater.execute("SELECT COUNT(*) FROM t")
        count = updater.fetchone()[0]
        assert count == 100000

        con.execute("COMMIT")

    def test_subcursor_close_one_of_multiple(self):
        """Closing one subcursor doesn't affect other suspended streams."""
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t1 AS SELECT i AS id FROM range(100000) tbl(i)")
        con.execute("CREATE TABLE t2 AS SELECT i AS id FROM range(100000) tbl(i)")
        con.execute("SET enable_suspended_queries = true")
        con.execute("BEGIN TRANSACTION")

        scan1 = con.subcursor()
        scan2 = con.subcursor()

        result1 = scan1.execute("SELECT id FROM t1 ORDER BY id")
        result2 = scan2.execute("SELECT id FROM t2 ORDER BY id")

        # Fetch from both
        batch1 = result1.fetchmany(100)
        assert len(batch1) == 100
        batch2 = result2.fetchmany(100)
        assert len(batch2) == 100

        # Close scan1 — scan2 should still work
        scan1.close()

        # scan2 should still be able to fetch
        total2 = len(batch2)
        while True:
            batch = result2.fetchmany(2048)
            if not batch:
                break
            total2 += len(batch)
        assert total2 == 100000

        # scan1 should be unusable
        with pytest.raises(duckdb.ConnectionException):
            scan1.execute("SELECT 1")

        con.execute("COMMIT")

    def test_subcursor_abandoned_without_close(self):
        """Subcursor that goes out of scope should clean up its suspended stream."""
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t AS SELECT i AS id FROM range(100000) tbl(i)")
        con.execute("SET enable_suspended_queries = true")
        con.execute("BEGIN TRANSACTION")

        def start_and_abandon():
            scanner = con.subcursor()
            result = scanner.execute("SELECT id FROM t ORDER BY id")
            result.fetchmany(100)
            # Suspend by running something else
            con.execute("SELECT 1")
            # scanner goes out of scope here — destructor should clean up

        start_and_abandon()
        import gc
        gc.collect()

        # Connection should still work — no leaked suspended state blocking it
        result = con.execute("SELECT COUNT(*) FROM t").fetchone()
        assert result[0] == 100000

        con.execute("COMMIT")

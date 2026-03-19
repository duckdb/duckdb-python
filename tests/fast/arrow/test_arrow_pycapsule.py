import pytest

import duckdb

pa = pytest.importorskip("pyarrow")
pl = pytest.importorskip("polars")


def polars_supports_capsule():
    from packaging.version import Version

    return Version(pl.__version__) >= Version("1.4.1")


class TestArrowPyCapsuleExport:
    """Tests for the PyCapsule export path (rel.__arrow_c_stream__).

    Validates that the fast path (PhysicalArrowCollector + ArrowQueryResultStreamWrapper)
    produces correct data, matching to_arrow_table() across types and edge cases.
    """

    def test_capsule_matches_to_arrow_table(self):
        """Fast path produces identical data to to_arrow_table for various types."""
        conn = duckdb.connect()
        sql = """
            SELECT
                i AS int_col,
                i::DOUBLE AS double_col,
                'row_' || i::VARCHAR AS str_col,
                i % 2 = 0 AS bool_col,
                CASE WHEN i % 3 = 0 THEN NULL ELSE i END AS nullable_col
            FROM range(1000) t(i)
        """
        expected = conn.sql(sql).to_arrow_table()
        actual = pa.table(conn.sql(sql))
        assert actual.equals(expected)

    def test_capsule_matches_to_arrow_table_nested_types(self):
        """Fast path handles nested types (struct, list, map)."""
        conn = duckdb.connect()
        sql = """
            SELECT
                {'x': i, 'y': i::VARCHAR} AS struct_col,
                [i, i+1, i+2] AS list_col,
                MAP {i::VARCHAR: i*10} AS map_col,
            FROM range(100) t(i)
        """
        expected = conn.sql(sql).to_arrow_table()
        actual = pa.table(conn.sql(sql))
        assert actual.equals(expected)

    def test_capsule_multi_batch(self):
        """Data exceeding the 1M batch size produces multiple batches, all yielded correctly."""
        conn = duckdb.connect()
        sql = "SELECT i, i::DOUBLE AS d FROM range(1500000) t(i)"
        expected = conn.sql(sql).to_arrow_table()
        actual = pa.table(conn.sql(sql))
        assert actual.num_rows == 1500000
        assert actual.equals(expected)

    def test_capsule_empty_result(self):
        """Empty result set produces a valid empty table with correct schema."""
        conn = duckdb.connect()
        sql = "SELECT i AS a, i::VARCHAR AS b FROM range(10) t(i) WHERE i < 0"
        expected = conn.sql(sql).to_arrow_table()
        actual = pa.table(conn.sql(sql))
        assert actual.num_rows == 0
        assert actual.schema.equals(expected.schema)

    def test_capsule_slow_path_after_execute(self):
        """Pre-executed relation takes the slow path (MaterializedQueryResult) and still works."""
        conn = duckdb.connect()
        sql = "SELECT i, i::DOUBLE AS d FROM range(500) t(i)"
        expected = conn.sql(sql).to_arrow_table()

        rel = conn.sql(sql)
        rel.execute()  # forces MaterializedCollector, not PhysicalArrowCollector
        actual = pa.table(rel)
        assert actual.equals(expected)


@pytest.mark.skipif(
    not polars_supports_capsule(), reason="Polars version does not support the Arrow PyCapsule interface"
)
class TestArrowPyCapsule:
    def test_polars_pycapsule_scan(self, duckdb_cursor):
        class MyObject:
            def __init__(self, obj) -> None:
                self.obj = obj
                self.count = 0

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                self.count += 1
                return self.obj.__arrow_c_stream__(requested_schema=requested_schema)

        df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        obj = MyObject(df)

        # Call the __arrow_c_stream__ from within DuckDB
        # MyObject has no __arrow_c_schema__, so GetSchema() falls back to __arrow_c_stream__ (1 call),
        # then Produce() calls __arrow_c_stream__ again (1 call) = 2 calls minimum per scan.
        res = duckdb_cursor.sql("select * from obj")
        assert res.fetchall() == [(1, 5), (2, 6), (3, 7), (4, 8)]
        count_after_first = obj.count
        assert count_after_first >= 2

        # Call the __arrow_c_stream__ method and pass in the capsule instead
        capsule = obj.__arrow_c_stream__()
        res = duckdb_cursor.sql("select * from capsule")
        assert res.fetchall() == [(1, 5), (2, 6), (3, 7), (4, 8)]
        assert obj.count == count_after_first + 1

        # Ensure __arrow_c_stream__ accepts a requested_schema argument as noop
        capsule = obj.__arrow_c_stream__(requested_schema="foo")  # noqa: F841
        res = duckdb_cursor.sql("select * from capsule")
        assert res.fetchall() == [(1, 5), (2, 6), (3, 7), (4, 8)]
        assert obj.count == count_after_first + 2

    def test_capsule_roundtrip(self, duckdb_cursor):
        def create_capsule():
            conn = duckdb.connect()
            rel = conn.sql("select i, i+1, -i from range(100) t(i)")

            capsule = rel.__arrow_c_stream__()
            return capsule

        capsule = create_capsule()  # noqa: F841
        rel2 = duckdb_cursor.sql("select * from capsule")
        assert rel2.fetchall() == [(i, i + 1, -i) for i in range(100)]

    def test_automatic_reexecution(self, duckdb_cursor):
        other_con = duckdb_cursor.cursor()
        rel = duckdb_cursor.sql("select i, i+1, -i from range(100) t(i)")

        capsule_one = rel.__arrow_c_stream__()  # noqa: F841
        res1 = other_con.sql("select * from capsule_one").fetchall()
        capsule_two = rel.__arrow_c_stream__()  # noqa: F841
        res2 = other_con.sql("select * from capsule_two").fetchall()
        assert len(res1) == 100
        assert res1 == res2

    def test_pycapsule_rescan_error_type(self, duckdb_cursor):
        """Issue #105: re-executing a relation backed by a consumed PyCapsule."""
        pa = pytest.importorskip("pyarrow")
        tbl = pa.table({"a": [1]})
        capsule = tbl.__arrow_c_stream__()  # noqa: F841
        rel = duckdb_cursor.sql("SELECT * FROM capsule")
        rel.fetchall()  # consumes the capsule
        with pytest.raises(duckdb.InvalidInputException):
            rel.fetchall()  # re-execution should be InvalidInputException, not InternalException

    def test_consumer_interface_roundtrip(self, duckdb_cursor):
        def create_table():
            class MyTable:
                def __init__(self, rel, conn) -> None:
                    self.rel = rel
                    self.conn = conn

                def __arrow_c_stream__(self, requested_schema=None) -> object:
                    return self.rel.__arrow_c_stream__(requested_schema=requested_schema)

            conn = duckdb.connect()
            rel = conn.sql("select i, i+1, -i from range(100) t(i)")
            return MyTable(rel, conn)

        tbl = create_table()  # noqa: F841
        rel2 = duckdb_cursor.sql("select * from tbl")
        assert rel2.fetchall() == [(i, i + 1, -i) for i in range(100)]

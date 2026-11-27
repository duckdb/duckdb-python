from decimal import Decimal

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")

WIDE = Decimal("123456789012345678901234567890.1234567890")


class TestArrowDecimal256:
    def test_a_decimal256_of_38_digits_or_fewer_reads_as_a_fixed_size_decimal(self, duckdb_cursor):
        with duckdb.connect() as conn:
            pa_decimal256 = pa.Table.from_pylist(  # noqa: F841
                [{"data": Decimal("100.00")} for _ in range(4)],
                pa.schema([("data", pa.decimal256(12, 4))]),
            )
            rel = conn.sql("select * from pa_decimal256;")
            assert [str(t) for t in rel.types] == ["DECIMAL(12,4)"]
            assert rel.to_arrow_table().to_pylist() == [{"data": Decimal("100.0000")}] * 4

    def test_a_wide_decimal256_round_trips(self, duckdb_cursor):
        with duckdb.connect() as conn:
            conn.execute("SET arrow_output_version = '1.1'")
            wide = pa.table({"data": pa.array([WIDE, None, WIDE.copy_negate()], pa.decimal256(50, 10))})
            rel = conn.sql("select * from wide")
            assert [str(t) for t in rel.types] == ["DECIMAL(50,10)"]
            assert rel.fetchall() == [(WIDE,), (None,), (WIDE.copy_negate(),)]
            assert rel.to_arrow_table().equals(wide)

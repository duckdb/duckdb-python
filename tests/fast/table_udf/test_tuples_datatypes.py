import pytest

import duckdb
import duckdb.sqltypes as sqltypes


def test_bigint_params(tmp_path):
    def bigint_func(big_value):
        return [(big_value, big_value + 1, big_value * 2)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            name="bigint_func",
            callable=bigint_func,
            schema={"orig": sqltypes.BIGINT, "plus_one": sqltypes.BIGINT, "doubled": sqltypes.BIGINT},
            type="tuples",
        )

        large_val = 4611686018427387900  # Half of max int64
        result = conn.sql("SELECT * FROM bigint_func(?)", params=(large_val,)).fetchall()
        assert result[0][0] == large_val
        assert result[0][1] == large_val + 1
        assert result[0][2] == large_val * 2


def test_hugeint_params(tmp_path):
    def hugeint_func(huge_value):
        return [(huge_value, huge_value + 1)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            name="hugeint_func",
            callable=hugeint_func,
            schema={"orig": sqltypes.HUGEINT, "plus_one": sqltypes.HUGEINT},
            type="tuples",
        )

        huge_val = 9223372036854775808
        result = conn.sql(f"SELECT * FROM hugeint_func(?)", params=(huge_val,)).fetchall()
        assert result[0][0] == huge_val
        assert result[0][1] == huge_val + 1


def test_decimal_params(tmp_path):
    from decimal import Decimal

    def decimal_func(dec_value):
        result = dec_value * 2 if isinstance(dec_value, float) else Decimal(str(dec_value)) * 2
        return [(dec_value, result)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            name="decimal_func",
            callable=decimal_func,
            schema={"orig": duckdb.decimal_type(10, 2), "doubled": duckdb.decimal_type(10, 2)},
            type="tuples",
        )

        result = conn.sql("SELECT * FROM decimal_func(?::decimal)", params=(123.45,)).fetchall()
        assert float(result[0][0]) == 123.45
        assert float(result[0][1]) == 246.90


def test_uuid_params(tmp_path):
    def uuid_func(uuid_value):
        return [(str(uuid_value),)]

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            name="uuid_func",
            callable=uuid_func,
            schema={"orig": sqltypes.UUID},
            type="tuples",
        )

        test_uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = conn.sql(f"SELECT * FROM uuid_func(?::uuid)", params=(test_uuid,)).fetchall()
        assert str(result[0][0]) == test_uuid

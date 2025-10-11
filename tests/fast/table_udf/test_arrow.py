from typing import Iterator

import pytest

import duckdb
import duckdb.sqltypes as sqltypes
from duckdb.functional import PythonTableUDFType


def tuple_generator(count: int = 10) -> Iterator[tuple[str, int]]:
    for i in range(count):
        yield (f"name_{i}", i)


def simple_arrow_table(count: int):
    pa = pytest.importorskip("pyarrow")

    data = {
        "id": list(range(count)),
        "value": [i * 2 for i in range(count)],
        "name": [f"row_{i}" for i in range(count)],
    }
    return pa.table(data)


def arrow_all_types(count: int):
    pa = pytest.importorskip("pyarrow")
    from decimal import Decimal
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    data = {
        "col_tinyint": pa.array(range(count), type=pa.int8()),
        "col_smallint": pa.array(range(count), type=pa.int16()),
        "col_int": pa.array(range(count), type=pa.int32()),
        "col_bigint": pa.array(range(count), type=pa.int64()),
        "col_utinyint": pa.array(range(count), type=pa.uint8()),
        "col_usmallint": pa.array(range(count), type=pa.uint16()),
        "col_uint": pa.array(range(count), type=pa.uint32()),
        "col_ubigint": pa.array(range(count), type=pa.uint64()),
        "col_float": pa.array((i * 1.5 for i in range(count)), type=pa.float32()),
        "col_double": pa.array((i * 2.5 for i in range(count)), type=pa.float64()),
        "col_varchar": pa.array((f"row_{i}" for i in range(count)), type=pa.string()),
        "col_bool": pa.array((i % 2 == 0 for i in range(count)), type=pa.bool_()),
        "col_timestamp": pa.array(
            (now + timedelta(seconds=i) for i in range(count)), type=pa.timestamp("us", tz="UTC")
        ),
        "col_date": pa.array((now.date() + timedelta(days=i) for i in range(count)), type=pa.date32()),
        "col_time": pa.array(((now + timedelta(microseconds=i)).time() for i in range(count)), type=pa.time64("ns")),
        "col_decimal": pa.array((Decimal(i) / 10 for i in range(count)), type=pa.decimal128(10, 2)),
        "col_blob": pa.array((f"bin_{i}".encode() for i in range(count)), type=pa.binary()),
        "col_list": pa.array(([i, i + 1] for i in range(count)), type=pa.list_(pa.int32())),
        "col_struct": pa.array(
            ({"x": i, "y": float(i)} for i in range(count)), type=pa.struct([("x", pa.int32()), ("y", pa.float32())])
        ),
    }
    return pa.table(data)


ALL_TYPES_SCHEMA = {
    "col_tinyint": sqltypes.TINYINT,
    "col_smallint": sqltypes.SMALLINT,
    "col_int": sqltypes.INTEGER,
    "col_bigint": sqltypes.BIGINT,
    "col_utinyint": sqltypes.UTINYINT,
    "col_usmallint": sqltypes.USMALLINT,
    "col_uint": sqltypes.UINTEGER,
    "col_ubigint": sqltypes.UBIGINT,
    "col_float": sqltypes.FLOAT,
    "col_double": sqltypes.DOUBLE,
    "col_varchar": sqltypes.VARCHAR,
    "col_bool": sqltypes.BOOLEAN,
    "col_timestamp": sqltypes.TIMESTAMP_TZ,
    "col_date": sqltypes.DATE,
    "col_time": sqltypes.TIME,
    "col_decimal": duckdb.decimal_type(10, 2),
    "col_blob": sqltypes.BLOB,
    "col_list": duckdb.list_type(sqltypes.INTEGER),
    "col_struct": duckdb.struct_type({"x": sqltypes.INTEGER, "y": sqltypes.FLOAT}),
}


def test_arrow_small(tmp_path):
    """Defines and creates a Table UDF with only positional parameters, verifies that it works
    and verifies it fails from another connection scope.
    """
    pa = pytest.importorskip("pyarrow")

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            "simple_arrow",
            simple_arrow_table,
            schema={"x": sqltypes.BIGINT, "y": sqltypes.BIGINT, "name": sqltypes.VARCHAR},
            type=PythonTableUDFType.ARROW_TABLE,
        )

        result = conn.execute("SELECT * FROM simple_arrow(5)").fetchall()

        assert len(result) == 5

    # Should fail because it's not defined in this conn
    with duckdb.connect(tmp_path / "test2.duckdb") as conn, pytest.raises(duckdb.CatalogException):
        result = conn.execute("SELECT * FROM simple_arrow(5)").fetchall()

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        # Should fail because schema is missing a col
        conn.create_table_function(
            "simple_arrow",
            simple_arrow_table,
            schema={"x": sqltypes.BIGINT, "y": sqltypes.BIGINT},
            type=PythonTableUDFType.ARROW_TABLE,
        )
        with pytest.raises(duckdb.InvalidInputException) as exc_info:
            result = conn.execute("SELECT * FROM simple_arrow(5)").fetchall()
        assert "Vector::Reference" in str(exc_info.value) or "schema" in str(exc_info.value).lower()


def test_arrow_large_1(tmp_path):
    """tests: more rows, aggregation, limits, named parameters, parameters."""
    pa = pytest.importorskip("pyarrow")

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        n = 2048 * 1000

        conn.create_table_function(
            "large_arrow",
            simple_arrow_table,
            schema={"id": sqltypes.BIGINT, "value": sqltypes.BIGINT, "name": sqltypes.VARCHAR},
            type="arrow_table",
            parameters=["count"],
        )

        result = conn.execute("SELECT COUNT(*) FROM large_arrow(count:=?)", parameters=(n,)).fetchone()
        assert result[0] == n

        df = conn.sql(f"SELECT * FROM large_arrow({n}) LIMIT 10").df()
        assert len(df) == 10
        assert df["id"].tolist() == list(range(10))

        arrow_result = conn.execute("SELECT * FROM large_arrow(?)", parameters=(n,)).fetch_arrow_table()
        assert len(arrow_result) == n

        result = conn.sql("SELECT SUM(value) FROM large_arrow(count:=$count)", params={"count": n}).fetchone()
        expected_sum = sum(i * 2 for i in range(n))
        assert result[0] == expected_sum


def test_arrowbatched_execute(tmp_path):
    pytest.importorskip("pyarrow")

    count = 2048 * 1000
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function(
            name="gen_function",
            callable=tuple_generator,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.execute(
            "SELECT * FROM gen_function(?)",
            parameters=(count,),
        ).fetch_record_batch()

        result = conn.execute(
            f"SELECT * FROM gen_function({count})",
        ).fetch_record_batch()

        c = 0
        for batch in result:
            c += batch.num_rows
        assert c == count


def test_arrowbatched_sql_relation(tmp_path):
    pytest.importorskip("pyarrow")

    count = 2048 * 1000
    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function(
            name="gen_function",
            callable=tuple_generator,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.sql(
            f"SELECT * FROM gen_function({count})",
        ).fetch_arrow_reader()

        c = 0
        for batch in result:
            c += batch.num_rows
        assert c == count


def test_arrow_types(tmp_path):
    """Return many types from an arrow table UDF, and verify the results are correct"""
    pa = pytest.importorskip("pyarrow")

    with duckdb.connect(tmp_path / "test.duckdb") as conn:
        conn.create_table_function(
            "all_types_arrow",
            arrow_all_types,
            schema=ALL_TYPES_SCHEMA,
            type=PythonTableUDFType.ARROW_TABLE,
        )

        result = conn.execute("SELECT * FROM all_types_arrow(3)").fetchall()
        assert len(result) == 3

        first_row = result[0]
        assert first_row[0] == 0  # col_tinyint
        assert first_row[1] == 0  # col_smallint
        assert first_row[2] == 0  # col_int
        assert first_row[3] == 0  # col_bigint
        assert first_row[10] == "row_0"  # col_varchar
        assert first_row[11] is True  # col_bool

        result = conn.execute("SELECT SUM(col_int), AVG(col_float) FROM all_types_arrow(100)").fetchone()
        expected_sum = sum(range(100))
        assert result[0] == expected_sum

        result = conn.execute("SELECT COUNT(*) FROM all_types_arrow(50) WHERE col_bool = true").fetchone()
        assert result[0] == 25  # Half should be true (even numbers)

        result = conn.execute("SELECT col_varchar, col_int FROM all_types_arrow(5)").fetchall()
        assert len(result) == 5
        assert result[2] == ("row_2", 2)

        result = conn.execute("SELECT col_list FROM all_types_arrow(2)").fetchall()
        assert result[0][0] == [0, 1]
        assert result[1][0] == [1, 2]

        result = conn.sql("SELECT col_struct FROM all_types_arrow(2)").fetchall()
        assert result[0][0] == {"x": 0, "y": 0.0}
        assert result[1][0] == {"x": 1, "y": 1.0}

        schema_result = conn.sql("DESCRIBE SELECT * FROM all_types_arrow(1)").fetchall()
        column_names = [row[0] for row in schema_result]
        assert list(ALL_TYPES_SCHEMA.keys()) == column_names

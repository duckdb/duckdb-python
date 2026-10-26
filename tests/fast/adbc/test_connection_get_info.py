import pytest

import duckdb

pa = pytest.importorskip("pyarrow")
pytest.importorskip("adbc_driver_manager")
adbc_driver_duckdb_dbapi = pytest.importorskip("adbc_driver_duckdb.dbapi")
from adbc_driver_manager import AdbcInfoCode  # noqa: E402

# ADBC_INFO_DRIVER_ADBC_VERSION and ADBC_VERSION_1_1_0 from adbc.h; the driver manager enum stops at 102.
INFO_DRIVER_ADBC_VERSION = 103
DRIVER_ADBC_VERSION_1_1_0 = 1001000


class TestADBCConnectionGetInfo:
    def test_connection_basic(self):
        con = adbc_driver_duckdb_dbapi.connect()
        with con.cursor() as cursor:
            cursor.execute("select 42")
            res = cursor.fetchall()
            assert res == [(42,)]

    def test_connection_get_info_all(self):
        con = adbc_driver_duckdb_dbapi.connect()
        adbc_con = con.adbc_connection
        res = adbc_con.get_info()
        reader = pa.RecordBatchReader._import_from_c(res.address)
        table = reader.read_all()
        values = table["info_value"]

        version = "v" + duckdb.__duckdb_version__  # don't hardcode this, as it will change every version
        expected_strings = pa.array(
            ["duckdb", version, "ADBC DuckDB Driver", version, "(unknown)"],
            type=pa.string(),
        )

        assert values.num_chunks == 1
        chunk = values.chunk(0)
        assert chunk.type.mode == "dense"
        assert chunk.field(0) == expected_strings
        assert chunk.field(2) == pa.array([DRIVER_ADBC_VERSION_1_1_0], type=pa.int64())
        assert table.to_pylist() == [
            {"info_name": AdbcInfoCode.VENDOR_NAME, "info_value": "duckdb"},
            {"info_name": AdbcInfoCode.VENDOR_VERSION, "info_value": version},
            {"info_name": AdbcInfoCode.DRIVER_NAME, "info_value": "ADBC DuckDB Driver"},
            {"info_name": AdbcInfoCode.DRIVER_VERSION, "info_value": version},
            {"info_name": AdbcInfoCode.DRIVER_ARROW_VERSION, "info_value": "(unknown)"},
            {"info_name": INFO_DRIVER_ADBC_VERSION, "info_value": DRIVER_ADBC_VERSION_1_1_0},
        ]

    def test_empty_result(self):
        con = adbc_driver_duckdb_dbapi.connect()
        adbc_con = con.adbc_connection
        res = adbc_con.get_info([1337])
        reader = pa.RecordBatchReader._import_from_c(res.address)
        table = reader.read_all()
        values = table["info_value"]

        # Because all the codes we asked for were unrecognized, the result set is empty
        assert table.num_rows == 0
        assert values.length() == 0

    def test_unrecognized_codes(self):
        con = adbc_driver_duckdb_dbapi.connect()
        adbc_con = con.adbc_connection
        res = adbc_con.get_info([0, 1000, 4, 2000])
        reader = pa.RecordBatchReader._import_from_c(res.address)
        table = reader.read_all()
        values = table["info_value"]

        expected_result = pa.array(["duckdb"], type=pa.string())

        assert values.num_chunks == 1
        chunk = values.chunk(0)
        string_values = chunk.field(0)
        assert string_values == expected_result

"""Regression tests for duckdb-python issue #517.

Under ``arrow_output_version >= 1.4`` several string/blob-backed types are
appended to Arrow using the newer 4-buffer StringView/BinaryView layout, while
DuckDB's exported Arrow *schema* still declared the old non-view 3-buffer
format. A consumer importing through the Arrow C Data interface (pyarrow) then
rejects the array with::

    pyarrow.lib.ArrowInvalid: Expected 3 buffers for imported type <X>,
    ArrowArray struct has 4

For some variants the mismatch is silent (wrong / empty values) rather than an
exception, and the UUID + string-view variant could even crash the importer, so
every case here verifies DATA correctness -- by round-tripping the exported
table back through DuckDB and/or comparing the concrete values -- not merely
that no exception is raised.

Fixed in duckdb core (arrow_converter.cpp / arrow_type_extension.cpp /
appender/enum_data.hpp); these tests pin the behaviour from the Python client.
"""

import subprocess
import sys
import textwrap

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")

requires_string_view = pytest.mark.skipif(
    not hasattr(pa, "string_view"),
    reason="This version of PyArrow does not support StringViews",
)

# The four-row enum fixture reused across the ENUM cases.
_ENUM_QUERY = "SELECT x::ENUM('happy', 'sad', 'angry') AS v FROM (VALUES ('happy'), ('sad'), ('angry'), ('happy')) t(x)"
_ENUM_VALUES = ["happy", "sad", "angry", "happy"]


def _export_and_roundtrip(con, query):
    """Export ``query`` to a pyarrow table and re-import it through DuckDB.

    ``pa.table(con.sql(...))`` is exactly the operation that raised
    ``ArrowInvalid`` before the fix. ``con.from_arrow`` additionally exercises
    the import side (re-reading DuckDB's own ``>= 1.4`` view output).

    Returns ``(expected_rows, arrow_table, roundtrip_rows)``.
    """
    expected = con.sql(query).fetchall()
    table = pa.table(con.sql(query))
    roundtrip = con.from_arrow(table).fetchall()
    return expected, table, roundtrip


class TestArrowOutputVersionBuffers:
    def test_varint_default(self):
        # Defaults already worked before the fix; this guards the baseline path.
        con = duckdb.connect()
        expected, table, roundtrip = _export_and_roundtrip(con, "SELECT (2**100)::VARINT AS v")
        assert table.num_rows == 1
        assert expected == [(str(2**100),)]
        assert roundtrip == expected

    def test_varint_v1_4(self):
        # VARINT/BIGNUM at arrow_output_version >= 1.4 raised ArrowInvalid
        # ("Expected 3 buffers ... extension<arrow.opaque[... bignum]>, ... has 4").
        con = duckdb.connect()
        con.execute("SET arrow_output_version='1.4'")
        # Build exact, distinct big values from string literals (DuckDB's ``**``
        # is floating point, so ``2**100 + i`` would round to the same value).
        values = [
            "1267650600228229401496703205376",  # 2**100
            "-1267650600228229401496703205376",
            "0",
            "99999999999999999999999999999999",
        ]
        query = "SELECT x::VARINT AS v FROM (VALUES ('" + "'), ('".join(values) + "')) t(x)"
        expected, table, roundtrip = _export_and_roundtrip(con, query)
        assert table.num_rows == 4
        assert expected == [(v,) for v in values]
        assert roundtrip == expected

    def test_geometry_v1_5(self):
        # Built-in GEOMETRY (WKB/binary storage) at arrow_output_version >= 1.5
        # raised ArrowInvalid ("Expected 3 buffers for imported type binary ...
        # has 4").
        con = duckdb.connect()
        con.execute("SET arrow_output_version='1.5'")
        query = "SELECT ('POINT(' || i || ' ' || (i + 1) || ')')::GEOMETRY AS g FROM range(5) t(i)"
        try:
            expected = con.sql(query).fetchall()
        except duckdb.Error as exc:  # pragma: no cover - depends on the build
            pytest.skip(f"built-in GEOMETRY type unavailable in this build: {exc}")
        table = pa.table(con.sql(query))  # raised ArrowInvalid before the fix
        roundtrip = con.from_arrow(table).fetchall()
        assert len(expected) == 5
        assert roundtrip == expected

    @requires_string_view
    def test_enum_string_view_v1_5(self):
        # ENUM with produce_arrow_string_view + arrow_output_version >= 1.5 raised
        # ArrowInvalid ("Expected 3 buffers for imported type string ... has 4").
        # A naive schema-only fix imported without error but yielded empty / wrong
        # dictionary values, so assert the concrete strings, not just "no raise".
        con = duckdb.connect()
        con.execute("SET produce_arrow_string_view=true")
        con.execute("SET arrow_output_version='1.5'")
        table = pa.table(con.sql(_ENUM_QUERY))  # raised ArrowInvalid before the fix
        assert table.column("v").combine_chunks().to_pylist() == _ENUM_VALUES
        assert con.from_arrow(table).fetchall() == [(v,) for v in _ENUM_VALUES]

    def test_bit_lossless_v1_4(self):
        # BIT with arrow_lossless_conversion + arrow_output_version >= 1.4 raised
        # ArrowInvalid ("Expected 3 buffers ... extension<arrow.opaque[... bit]>,
        # ... has 4").
        con = duckdb.connect()
        con.execute("SET arrow_lossless_conversion=true")
        con.execute("SET arrow_output_version='1.4'")
        query = "SELECT x::BIT AS v FROM (VALUES ('1010'), ('111'), ('0'), ('110011')) t(x)"
        expected, table, roundtrip = _export_and_roundtrip(con, query)
        assert table.num_rows == 4
        assert expected == [("1010",), ("111",), ("0",), ("110011",)]
        assert roundtrip == expected

    def test_enum_large_buffer(self):
        # ENUM with arrow_large_buffer_size: the dictionary value child must stay
        # a regular (int32-offset) string and the values must be correct.
        con = duckdb.connect()
        con.execute("SET arrow_large_buffer_size=true")
        table = pa.table(con.sql(_ENUM_QUERY))
        col_type = table.schema.field("v").type
        assert pa.types.is_dictionary(col_type)
        # Never a large_string: the dictionary is always written with int32 offsets.
        assert col_type.value_type == pa.string()
        assert table.column("v").combine_chunks().to_pylist() == _ENUM_VALUES
        assert con.from_arrow(table).fetchall() == [(v,) for v in _ENUM_VALUES]

    @requires_string_view
    def test_uuid_string_view_v1_4_subprocess(self):
        # UUID with produce_arrow_string_view + arrow_output_version >= 1.4: the
        # schema declared the "vu" string-view layout over normal-layout data.
        # Without the fix this produced wrong data (empty strings) and could crash
        # the importer on other platforms, so run it isolated and assert on the
        # child process result.
        code = textwrap.dedent(
            """
            import duckdb
            import pyarrow as pa

            con = duckdb.connect()
            con.execute("SET produce_arrow_string_view=true")
            con.execute("SET arrow_output_version='1.4'")
            uid = '550e8400-e29b-41d4-a716-446655440000'
            table = pa.table(con.sql("SELECT '" + uid + "'::UUID AS v"))
            vals = table.column("v").combine_chunks().to_pylist()
            assert vals == [uid], vals
            # Non-lossless UUID exports as a regular string, so it re-imports as
            # a VARCHAR (string), not a uuid.UUID.
            assert con.from_arrow(table).fetchall() == [(uid,)]
            print("UUID_OK")
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            f"UUID + string_view export failed (returncode={proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        assert "UUID_OK" in proc.stdout


class TestIssue517Reproducer:
    """The exact reproducer from issue #517, asserting it no longer raises."""

    def test_original_reproducer(self):
        # VARINT/BIGNUM: defaults are fine
        con = duckdb.connect()
        pa.table(con.sql("SELECT (2**100)::VARINT AS v"))

        # VARINT/BIGNUM: arrow_output_version >= 1.4 -> FAILED before fix
        con = duckdb.connect()
        con.execute("SET arrow_output_version='1.4'")
        pa.table(con.sql("SELECT (2**100)::VARINT AS v"))

        # GEOMETRY: arrow_output_version = 1.5 -> FAILED before fix
        con = duckdb.connect()
        geom_query = "SELECT ST_Point(1, 2) AS g"
        try:
            con.execute("INSTALL spatial")
            con.execute("LOAD spatial")
        except duckdb.Error:
            # spatial unavailable (e.g. offline); fall back to the built-in
            # GEOMETRY type, which hits the same export path.
            geom_query = "SELECT 'POINT(1 2)'::GEOMETRY AS g"
        con.execute("SET arrow_output_version='1.5'")
        pa.table(con.sql(geom_query))

        # ENUM: needs BOTH produce_arrow_string_view=true AND aov >= 1.5 -> FAILED before fix
        con = duckdb.connect()
        con.execute("SET produce_arrow_string_view=true")
        con.execute("SET arrow_output_version='1.5'")
        pa.table(con.sql("SELECT 'happy'::ENUM('happy', 'sad') AS v"))

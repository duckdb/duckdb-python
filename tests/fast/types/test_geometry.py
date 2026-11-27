import pytest

import numpy as np
import pandas as pd

# GEOGRAPHY is the spherical sibling of GEOMETRY and reaches Python the same way: as its WKB bytes.
GEO_TYPES = ["GEOMETRY", "GEOGRAPHY"]

# Little endian WKB of POINT(1 2): byte order, type 1, then x and y as doubles.
POINT_WKB = bytes.fromhex("0101000000000000000000f03f0000000000000040")


@pytest.mark.parametrize("geo_type", GEO_TYPES)
class TestGeometry:
    def test_fetchall(self, duckdb_cursor, geo_type):
        duckdb_cursor.execute(f"SELECT 'POINT(1 2)'::{geo_type} AS geom")
        assert duckdb_cursor.fetchall() == [(POINT_WKB,)]

    def test_fetchnumpy(self, duckdb_cursor, geo_type):
        duckdb_cursor.execute(f"SELECT 'POINT(1 2)'::{geo_type} AS geom")
        results = duckdb_cursor.fetchnumpy()
        assert results["geom"].dtype == np.dtype(object)
        assert bytes(results["geom"][0]) == POINT_WKB

    def test_df(self, duckdb_cursor, geo_type):
        duckdb_cursor.execute(f"SELECT 'POINT(1 2)'::{geo_type} AS geom")
        df = duckdb_cursor.df()
        assert df["geom"].dtype == np.dtype(object)
        assert bytes(df["geom"].iloc[0]) == POINT_WKB

    def test_null(self, duckdb_cursor, geo_type):
        duckdb_cursor.execute(f"SELECT NULL::{geo_type} AS geom")
        results = duckdb_cursor.fetchall()
        assert results[0][0] is None

    def test_multiple_rows(self, duckdb_cursor, geo_type):
        duckdb_cursor.execute(f"SELECT 'POINT(1 2)'::{geo_type} AS geom UNION ALL SELECT NULL::{geo_type}")
        df = duckdb_cursor.df()
        assert df.shape[0] == 2
        assert bytes(df["geom"].iloc[0]) == POINT_WKB
        assert pd.isna(df["geom"].iloc[1])

    def test_a_bytes_parameter_binds(self, duckdb_cursor, geo_type):
        assert duckdb_cursor.execute(f"SELECT ?::{geo_type}", [POINT_WKB]).fetchall() == [(POINT_WKB,)]

    def test_types_and_description_name_the_type(self, duckdb_cursor, geo_type):
        rel = duckdb_cursor.sql(f"SELECT 'POINT(1 2)'::{geo_type} AS geom")
        assert [str(t) for t in rel.types] == [geo_type]
        description = duckdb_cursor.execute(f"SELECT 'POINT(1 2)'::{geo_type} AS geom").description
        assert [(name, str(code)) for name, code, *_ in description] == [("geom", geo_type)]

    def test_nested_values(self, duckdb_cursor, geo_type):
        point = f"'POINT(1 2)'::{geo_type}"
        row = duckdb_cursor.execute(
            f"SELECT MAP {{{point}: 1}}, MAP {{1: {point}}}, [{point}, NULL], {{'g': {point}}}"
        ).fetchone()
        assert row == ({POINT_WKB: 1}, {1: POINT_WKB}, [POINT_WKB, None], {"g": POINT_WKB})

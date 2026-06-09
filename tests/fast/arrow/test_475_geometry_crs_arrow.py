"""Regression test for duckdb-python#475 / duckdb-spatial#788.

Converting a GEOMETRY column that carries a CRS to Arrow used to raise
``InternalException: TransactionContext::ActiveTransaction called without
active transaction``. The GeoArrow schema callback does a catalog lookup to
resolve the CRS, which needs an active transaction -- but the Arrow schema was
rebuilt at consume time, after the producing (auto-commit) transaction had
already closed.

The fix builds and caches the schema on ArrowQueryResult while the producing
transaction is still active, and the consumers below reuse it. Each test
exercises one of those consumers. The geometry-with-CRS value is built with a
pure-core cast, so no spatial extension is required.
"""

from __future__ import annotations

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")

# An authority-code CRS forces the catalog lookup that used to require an open
# transaction. No spatial extension needed -- the cast and geoarrow.wkb mapping
# are both in core.
GEOM_SQL = "SELECT 'POINT(0 1)'::GEOMETRY('OGC:CRS84') AS g"


def _assert_geoarrow_with_crs(field: pa.Field) -> None:
    metadata = field.metadata or {}
    assert metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb"
    assert b"crs" in metadata.get(b"ARROW:extension:metadata", b"")


def test_475_to_arrow_table_geometry_with_crs():
    con = duckdb.connect()
    table = con.sql(GEOM_SQL).to_arrow_table()
    assert table.num_rows == 1
    _assert_geoarrow_with_crs(table.schema.field("g"))


def test_475_arrow_capsule_geometry_with_crs():
    # pa.table(rel) consumes via __arrow_c_stream__ (the capsule / ADBC path).
    con = duckdb.connect()
    table = pa.table(con.sql(GEOM_SQL))
    assert table.num_rows == 1
    _assert_geoarrow_with_crs(table.schema.field("g"))


def test_475_record_batch_reader_geometry_with_crs():
    con = duckdb.connect()
    table = con.sql(GEOM_SQL).to_arrow_reader().read_all()
    assert table.num_rows == 1
    _assert_geoarrow_with_crs(table.schema.field("g"))


def test_475_polars_geometry_with_crs():
    pl = pytest.importorskip("polars")
    con = duckdb.connect()
    # polars.DataFrame(rel) pulls the relation's Arrow C stream directly.
    df = pl.DataFrame(con.sql(GEOM_SQL))
    assert df.height == 1

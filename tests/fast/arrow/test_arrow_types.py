import pytest

import duckdb

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")


class TestArrowTypes:
    def test_null_type(self, duckdb_cursor):
        schema = pa.schema([("data", pa.null())])
        inputs = [pa.array([None, None, None], type=pa.null())]
        arrow_table = pa.Table.from_arrays(inputs, schema=schema)
        duckdb_cursor.register("testarrow", arrow_table)
        rel = duckdb.from_arrow(arrow_table).to_arrow_table()
        # NULL type now round-trips faithfully (previously it was coerced to int32)
        assert rel["data"] == arrow_table["data"]

    def test_empty_struct(self, duckdb_cursor):
        # Empty structs are now supported by DuckDB core. This previously raised
        # "Attempted to convert a STRUCT with no fields to DuckDB which is not supported";
        # the core check was removed and empty structs now round-trip faithfully.
        empty_struct_type = pa.struct([])
        arrow_table = pa.Table.from_arrays(  # noqa: F841
            [pa.array([None, None], type=empty_struct_type)],
            schema=pa.schema([("data", empty_struct_type)]),
        )
        result = duckdb_cursor.sql("select * from arrow_table").to_arrow_table()
        assert result["data"].type == empty_struct_type
        assert result["data"].to_pylist() == [None, None]

    def test_invalid_union(self, duckdb_cursor):
        # Create a sparse union array from dense arrays
        types = pa.array([0, 1, 1], type=pa.int8())
        sparse_union_array = pa.UnionArray.from_sparse(types, [], type_codes=[])

        arrow_table = pa.Table.from_arrays([sparse_union_array], schema=pa.schema([("data", sparse_union_array.type)]))
        with pytest.raises(
            duckdb.InvalidInputException,
            match="Attempted to convert a UNION with no fields to DuckDB which is not supported",
        ):
            duckdb_cursor.register("invalid_union", arrow_table)

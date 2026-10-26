import pytest

import duckdb

pa = pytest.importorskip("pyarrow")

pytestmark = pytest.mark.skipif(
    not hasattr(pa, "string_view"), reason="This version of PyArrow does not support StringViews"
)


# Compares with manually constructed arrow tables
def RoundTripStringView(query, array):
    con = duckdb.connect()
    con.execute("SET produce_arrow_string_view=True")
    arrow_tbl = con.sql(query).to_arrow_table()
    # Assert that we spit the same as the defined array
    arrow_tbl[0].validate(full=True)
    assert arrow_tbl[0].combine_chunks().tolist() == array.tolist()

    # Generate an arrow table
    # Create a field for the array with a specific data type
    field = pa.field("str_val", pa.string_view())

    # Create a schema for the table using the field
    schema = pa.schema([field])

    # Create a table using the schema and the array
    gt_table = pa.Table.from_arrays([array], schema=schema)  # noqa: F841
    arrow_table = con.sql("select * from gt_table").to_arrow_table()  # noqa: F841
    assert arrow_tbl[0].combine_chunks().tolist() == array.tolist()


def RoundTripDuckDBInternal(query):
    con = duckdb.connect()
    con.execute("SET produce_arrow_string_view=True")
    arrow_tbl = con.sql(query).to_arrow_table()
    arrow_tbl.validate(full=True)
    res = con.execute(query).fetchall()
    from_arrow_res = con.execute("FROM arrow_tbl order by str").fetchall()
    print(from_arrow_res)
    for i in range(len(res) - 1):
        assert res[i] == from_arrow_res[i]


class TestArrowStringView:
    # Test Small Inlined String View
    def test_inlined_string_view(self):
        RoundTripStringView(
            "SELECT (i*10^i)::varchar str FROM range(5) tbl(i) ",
            pa.array(["0.0", "10.0", "200.0", "3000.0", "40000.0"], type=pa.string_view()),
        )

    # Test Small Inlined String View With Nulls
    def test_inlined_string_view_null(self):
        RoundTripStringView(
            "SELECT (i*10^i)::varchar str FROM range(5) tbl(i) UNION Select NULL Order By str",
            pa.array(["0.0", "10.0", "200.0", "3000.0", "40000.0", None], type=pa.string_view()),
        )

    # Test Small Not-Inlined Strings
    def test_not_inlined_string_view(self):
        RoundTripStringView(
            "SELECT 'Imaverybigstringmuchbiggerthanfourbytes' str FROM range(5) tbl(i)",
            pa.array(
                [
                    "Imaverybigstringmuchbiggerthanfourbytes",
                    "Imaverybigstringmuchbiggerthanfourbytes",
                    "Imaverybigstringmuchbiggerthanfourbytes",
                    "Imaverybigstringmuchbiggerthanfourbytes",
                    "Imaverybigstringmuchbiggerthanfourbytes",
                ],
                type=pa.string_view(),
            ),
        )

    # Test Small Not-Inlined Strings with Null
    def test_not_inlined_string_view_with_null(self):
        RoundTripStringView(
            "SELECT 'Imaverybigstringmuchbiggerthanfourbytes'||i::varchar str FROM range(5) tbl(i) UNION SELECT NULL order by str",  # noqa: E501
            pa.array(
                [
                    "Imaverybigstringmuchbiggerthanfourbytes0",
                    "Imaverybigstringmuchbiggerthanfourbytes1",
                    "Imaverybigstringmuchbiggerthanfourbytes2",
                    "Imaverybigstringmuchbiggerthanfourbytes3",
                    "Imaverybigstringmuchbiggerthanfourbytes4",
                    None,
                ],
                type=pa.string_view(),
            ),
        )

    # Test Mix of Inlined and Not-Inlined Strings with Null
    def test_not_inlined_string_view_2(self):
        RoundTripStringView(
            "SELECT '8bytestr'||(i*10^i)::varchar str FROM range(5) tbl(i) UNION SELECT NULL order by str",
            pa.array(
                ["8bytestr0.0", "8bytestr10.0", "8bytestr200.0", "8bytestr3000.0", "8bytestr40000.0", None],
                type=pa.string_view(),
            ),
        )

    # Test Over-Vector Size
    def test_large_string_view_inlined(self):
        RoundTripDuckDBInternal("""select * from (SELECT i::varchar str FROM range(10000) tbl(i))  order by str""")

    def test_large_string_view_inlined_with_null(self):
        RoundTripDuckDBInternal(
            """select * from (SELECT i::varchar str FROM range(10000) tbl(i) UNION select null)  order by str"""
        )

    def test_large_string_view_not_inlined(self):
        RoundTripDuckDBInternal(
            """select * from (SELECT 'Imaverybigstringmuchbiggerthanfourbytes'||i::varchar str FROM range(10000) tbl(i) UNION select null)  order by str"""  # noqa: E501
        )

    def test_large_string_view_not_inlined_with_null(self):
        RoundTripDuckDBInternal(
            """select * from (SELECT 'Imaverybigstringmuchbiggerthanfourbytes'||i::varchar str FROM range(10000) tbl(i) UNION select null)  order by str"""  # noqa: E501
        )

    def test_large_string_view_mixed_with_null(self):
        RoundTripDuckDBInternal(
            """select * from (SELECT i::varchar str FROM range(10000) tbl(i) UNION SELECT 'Imaverybigstringmuchbiggerthanfourbytes'||i::varchar str FROM range(10000) tbl(i) UNION select null)  order by str"""  # noqa: E501
        )

    def test_multiple_data_buffers(self):
        arr = pa.array(["Imaverybigstringmuchbiggerthanfourbytes"], type=pa.string_view())
        arr = pa.concat_arrays([arr, arr, arr, arr, arr, arr, arr, arr, arr, arr])
        RoundTripStringView(
            "SELECT 'Imaverybigstringmuchbiggerthanfourbytes' str FROM range(10) tbl(i)",
            arr,
        )
        arr = pa.concat_arrays([arr, arr])
        RoundTripStringView(
            "SELECT 'Imaverybigstringmuchbiggerthanfourbytes' str FROM range(20) tbl(i)",
            arr,
        )

    def test_large_string_polars(self):
        pass
        # pl = pytest.importorskip('polars')
        # con = duckdb.connect()
        # con.execute("SET produce_arrow_string_view=True")
        # query = '''select * from (SELECT i::varchar str FROM range(10000) tbl(i) UNION SELECT 'Imaverybigstringmuchbiggerthanfourbytes'||i::varchar str FROM range(10000) tbl(i) UNION select null)  order by str'''  # noqa: E501
        # polars_df = con.execute(query).pl()
        # result = con.execute(query).fetchall()
        # con.register('polars_df', polars_df)

        # result_roundtrip_pl = con.execute("select * from polars_df order by all").fetchall()

        # assert result == result_roundtrip_pl


def _view_typed(con, sql, view_type):
    table = con.sql(sql).to_arrow_table()
    kind = pa.string_view() if view_type == "string_view" else pa.binary_view()
    return table.set_column(0, "k", table["k"].cast(kind))


@pytest.mark.parametrize("build_side", ["native", "arrow"])
@pytest.mark.parametrize("source", ["table", "reader"])
@pytest.mark.parametrize("view_type", ["string_view", "binary_view"])
def test_join_on_a_view_typed_key_matches_native(source, view_type, build_side):
    """A hash join's runtime filters on a view typed key are applied by DuckDB; the rows match a native join."""
    con = duckdb.connect()
    key_type = "VARCHAR" if view_type == "string_view" else "BLOB"
    con.execute(
        "CREATE TABLE probe_native AS SELECT CASE WHEN i % 997 = 0 THEN NULL "
        f"ELSE ('key' || lpad(i::VARCHAR, 6, '0'))::{key_type} END AS k, i AS v FROM range(50000) t(i)"
    )
    con.execute(
        f"CREATE TABLE build_native AS SELECT ('key' || lpad((i * 401)::VARCHAR, 6, '0'))::{key_type} AS k, "
        "i AS w FROM range(200) t(i)"
    )
    probe = _view_typed(con, "FROM probe_native", view_type)
    build = _view_typed(con, "FROM build_native", view_type)
    query = "SELECT count(*), min(v), max(v), sum(w) FROM {} JOIN {} USING (k)"
    expected = con.execute(query.format("probe_native", "build_native")).fetchall()
    assert expected[0][0] > 0
    con.register("probe_arrow", probe if source == "table" else probe.to_reader())
    con.register("build_arrow", build if source == "table" else build.to_reader())
    build_name = "build_arrow" if build_side == "arrow" else "build_native"
    assert con.execute(query.format("probe_arrow", build_name)).fetchall() == expected

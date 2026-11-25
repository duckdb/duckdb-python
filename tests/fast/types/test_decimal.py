from decimal import Decimal

import pytest

import duckdb
import numpy


class TestDecimal:
    def test_decimal(self, duckdb_cursor):
        duckdb_cursor.execute(
            "SELECT 1.2::DECIMAL(4,1), 100.3::DECIMAL(9,1), 320938.4298::DECIMAL(18,4), 49082094824.904820482094::DECIMAL(30,12), NULL::DECIMAL"  # noqa: E501
        )
        result = duckdb_cursor.fetchall()
        assert result == [
            (Decimal("1.2"), Decimal("100.3"), Decimal("320938.4298"), Decimal("49082094824.904820482094"), None)
        ]

    def test_decimal_numpy(self, duckdb_cursor):
        duckdb_cursor.execute(
            "SELECT 1.2::DECIMAL(4,1) AS a, 100.3::DECIMAL(9,1) AS b, 320938.4298::DECIMAL(18,4) AS c, 49082094824.904820482094::DECIMAL(30,12) AS d"  # noqa: E501
        )
        result = duckdb_cursor.fetchnumpy()
        assert result == {
            "a": numpy.array([1.2]),
            "b": numpy.array([100.3]),
            "c": numpy.array([320938.4298]),
            "d": numpy.array([49082094824.904820482094]),
        }


# A DECIMAL wider than 38 digits holds its unscaled value as a BIGNUM; a Python Decimal of any width becomes the exact
# DECIMAL it needs, up to the session's max_decimal_width.
WIDE = Decimal("123456789012345678901234567890.1234567890")
NEGATIVE_WIDE = WIDE.copy_negate()
TOO_WIDE = (
    r"DECIMAL type width 40 exceeds the maximum DECIMAL width of 38 \(SET max_decimal_width to allow wider decimals\)"
)
POLARS_REFUSAL = r"Converting DECIMAL\(48,2\) to polars is not supported"
# The sum of a DECIMAL(18,2) is DECIMAL(38,2), and twice it is DECIMAL(48,2).
SUM_TWICE = "SELECT sum(p) * 2 AS x FROM prices"


@pytest.fixture
def prices(duckdb_cursor):
    duckdb_cursor.execute("CREATE TABLE prices AS SELECT (i / 100)::DECIMAL(18,2) AS p FROM range(10) t(i)")
    return duckdb_cursor


@pytest.fixture
def wide_table(duckdb_cursor):
    duckdb_cursor.execute("CREATE TABLE t (d DECIMAL(50,10))")
    return duckdb_cursor


def rows_of(cursor):
    return cursor.execute("SELECT d FROM t").fetchall()


class TestWideDecimalResults:
    def test_the_row_fetch_is_exact(self, prices):
        assert prices.execute(f"SELECT x, typeof(x) FROM ({SUM_TWICE})").fetchall() == [
            (Decimal("0.90"), "DECIMAL(48,2)")
        ]
        assert prices.execute(f"SELECT '{WIDE}'::DECIMAL(50,10), NULL::DECIMAL(50,10)").fetchall() == [(WIDE, None)]

    @pytest.mark.parametrize("fetch", ["df", "fetchnumpy", "fetch_df_chunk"])
    def test_numpy_and_pandas_export_float64(self, prices, fetch):
        prices.execute(f"{SUM_TWICE} UNION ALL SELECT NULL ORDER BY x NULLS LAST")
        column = getattr(prices, fetch)()["x"]
        assert numpy.asarray(column).dtype == numpy.float64
        values = numpy.ma.filled(numpy.ma.masked_invalid(numpy.ma.asarray(column, dtype=numpy.float64)), numpy.nan)
        assert values[0] == 0.9
        assert numpy.isnan(values[1])

    def test_numpy_export_follows_the_engine_cast_to_double(self, duckdb_cursor):
        duckdb_cursor.execute(f"SELECT '{NEGATIVE_WIDE}'::DECIMAL(50,10) AS d")
        exported = duckdb_cursor.fetchnumpy()["d"][0]
        assert exported == duckdb_cursor.execute(f"SELECT '{NEGATIVE_WIDE}'::DECIMAL(50,10)::DOUBLE").fetchone()[0]
        duckdb_cursor.execute("SELECT ('1' || repeat('0', 400))::DECIMAL(401,0) AS d")
        with pytest.raises(duckdb.ConversionException, match="to type DOUBLE failed: value is out of range"):
            duckdb_cursor.fetchnumpy()

    @pytest.mark.parametrize("width", [38, 39])
    def test_widths_38_and_39_fetch_as_rows_and_numpy(self, duckdb_cursor, width):
        digits = Decimal("1" + "2" * (width - 11) + ".1234567890")
        query = f"SELECT '{digits}'::DECIMAL({width},10) AS d"
        assert duckdb_cursor.execute(query).fetchall() == [(digits,)]
        # The fixed-size tiers divide by the power of ten in double, so the last bit may differ from one rounding.
        assert duckdb_cursor.execute(query).fetchnumpy()["d"][0] == pytest.approx(float(digits), rel=1e-15)

    def test_the_arrow_export_keeps_the_engine_default(self, prices):
        # At the default arrow_output_version the engine has no decimal256, and its error names the setting.
        message = (
            r"Cannot convert DECIMAL\(39,2\) to Arrow format version 1\.0: decimals wider than 38 digits require "
            r"decimal256, SET arrow_output_version to 1\.1 or later"
        )
        with pytest.raises(duckdb.NotImplementedException, match=message):
            prices.sql("SELECT sum(p) + sum(p) AS x FROM prices").to_arrow_table()

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("version", ["1.0", "1.1"])
    @pytest.mark.parametrize(
        "query",
        [SUM_TWICE, f"SELECT [x] AS l FROM ({SUM_TWICE})", f"SELECT row(x, 1) AS r FROM ({SUM_TWICE})"],
    )
    def test_polars_raises_above_38_digits(self, prices, query, version, lazy):
        pytest.importorskip("polars")
        prices.execute(f"SET arrow_output_version = '{version}'")
        with pytest.raises(duckdb.NotImplementedException, match=POLARS_REFUSAL):
            prices.sql(query).pl(lazy=lazy)


class TestWidePythonDecimals:
    def test_an_untyped_parameter_is_the_exact_decimal(self, duckdb_cursor):
        assert duckdb_cursor.execute("SELECT ?, typeof(?)", [WIDE, WIDE]).fetchall() == [(WIDE, "DECIMAL(40,10)")]
        assert duckdb_cursor.execute("SELECT $a, $b", {"a": NEGATIVE_WIDE, "b": [WIDE, None]}).fetchall() == [
            (NEGATIVE_WIDE, [WIDE, None])
        ]

    @pytest.mark.parametrize(("width", "digits"), [(38, 28), (39, 29)])
    def test_widths_38_and_39_at_the_boundary(self, duckdb_cursor, width, digits):
        value = Decimal("9" * digits + ".1234567890")
        assert duckdb_cursor.execute("SELECT ?, typeof(?)", [value, value]).fetchall() == [
            (value, f"DECIMAL({width},10)")
        ]

    def test_a_typed_parameter_is_exact(self, duckdb_cursor):
        assert duckdb_cursor.execute("SELECT ?::DECIMAL(50,10)", [WIDE]).fetchall() == [(WIDE,)]

    def test_executemany_inserts_exactly(self, wide_table):
        wide_table.executemany("INSERT INTO t VALUES (?)", [[WIDE], [None], [NEGATIVE_WIDE]])
        assert rows_of(wide_table) == [(WIDE,), (None,), (NEGATIVE_WIDE,)]

    def test_append_of_a_dataframe_is_exact(self, wide_table):
        pd = pytest.importorskip("pandas")
        wide_table.append("t", pd.DataFrame({"d": [WIDE, None, NEGATIVE_WIDE]}))
        assert rows_of(wide_table) == [(WIDE,), (None,), (NEGATIVE_WIDE,)]

    def test_the_appender_is_exact(self, wide_table):
        appender = wide_table.appender("t")
        appender.append_row(WIDE)
        appender.append_row(None)
        appender.flush()
        query_appender = wide_table.query_appender("INSERT INTO t FROM appender_data", {"d": "DECIMAL(50,10)"})
        query_appender.append_row(NEGATIVE_WIDE)
        query_appender.flush()
        assert rows_of(wide_table) == [(WIDE,), (None,), (NEGATIVE_WIDE,)]

    def test_a_udf_takes_and_returns_it_exactly(self, duckdb_cursor):
        wide_type = duckdb.decimal_type(50, 10)
        duckdb_cursor.create_function("negated", lambda x: x.copy_negate(), [wide_type], wide_type)
        assert duckdb_cursor.execute(f"SELECT negated('{WIDE}'::DECIMAL(50,10))").fetchall() == [(NEGATIVE_WIDE,)]

    def test_a_pandas_object_column_is_the_exact_decimal(self, duckdb_cursor):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame({"d": [WIDE, None, Decimal("1.5")]})  # noqa: F841
        rel = duckdb_cursor.sql("SELECT d FROM frame")
        assert [str(t) for t in rel.types] == ["DECIMAL(40,10)"]
        assert rel.fetchall() == [(WIDE,), (None,), (Decimal("1.5"),)]

    def test_a_written_out_literal_is_the_exact_decimal(self, duckdb_cursor):
        rel = duckdb_cursor.sql("SELECT 1 AS one").select(duckdb.ConstantExpression(WIDE).alias("c"))
        assert [str(t) for t in rel.types] == ["DECIMAL(40,10)"]
        assert rel.fetchall() == [(WIDE,)]
        listed = duckdb_cursor.sql("SELECT 1 AS one").select(duckdb.ConstantExpression([WIDE, None]).alias("c"))
        assert [str(t) for t in listed.types] == ["DECIMAL(40,10)[]"]
        assert listed.fetchall() == [([WIDE, None],)]

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(lambda con: con.execute("SELECT ?", [WIDE]).fetchall(), id="parameter"),
            pytest.param(
                lambda con: con.sql("SELECT 1").select(duckdb.ConstantExpression(WIDE)).fetchall(), id="literal"
            ),
            pytest.param(lambda con: con.appender("t").append_row(WIDE), id="appender"),
        ],
    )
    def test_a_decimal_wider_than_the_setting_raises_the_engine_error(self, wide_table, path):
        wide_table.execute("SET max_decimal_width = 38")
        with pytest.raises(duckdb.BinderException, match=TOO_WIDE):
            path(wide_table)

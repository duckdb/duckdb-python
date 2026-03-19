import platform

import numpy as np
import pandas as pd
import pytest
from conftest import is_string_dtype

import duckdb


def assert_nullness(items, null_indices):
    for i in range(len(items)):
        if i in null_indices:
            assert pd.isna(items[i])
        else:
            assert not pd.isna(items[i])


@pytest.mark.skipif(platform.system() == "Emscripten", reason="Pandas interaction is broken in Pyodide 3.11")
class TestPandasNA:
    @pytest.mark.parametrize("rows", [100, duckdb.__standard_vector_size__, 5000, 1000000])
    def test_pandas_string_null(self, duckdb_cursor, rows):
        df = pd.DataFrame(index=np.arange(rows))
        df["string_column"] = pd.Series(dtype="string")
        e_df_rel = duckdb_cursor.from_df(df)
        assert e_df_rel.types == ["VARCHAR"]
        roundtrip = e_df_rel.df()
        assert is_string_dtype(roundtrip["string_column"].dtype)
        expected = pd.DataFrame({"string_column": [None for _ in range(rows)]})
        pd.testing.assert_frame_equal(expected, roundtrip, check_dtype=False)

    def test_pandas_na(self, duckdb_cursor):
        # DataFrame containing a single pd.NA
        df = pd.DataFrame(pd.Series([pd.NA]))

        res = duckdb_cursor.execute("select * from df").fetchall()
        assert res[0][0] is None

        # DataFrame containing multiple values, with a pd.NA mixed in
        null_index = 3
        df = pd.DataFrame(pd.Series([3, 1, 2, pd.NA, 8, 6]))  # noqa: F841
        res = duckdb_cursor.execute("select * from df").fetchall()
        items = [x[0] for x in list(res)]
        assert_nullness(items, [null_index])

        # Test if pd.NA behaves the same as np.nan once converted
        nan_df = pd.DataFrame(
            {
                "a": [
                    1.123,
                    5.23234,
                    np.nan,
                    7234.0000124,
                    0.000000124,
                    0000000000000.0000001,
                    np.nan,
                    -2342349234.00934580345,
                ]
            }
        )
        na_df = pd.DataFrame(
            {
                "a": [
                    1.123,
                    5.23234,
                    pd.NA,
                    7234.0000124,
                    0.000000124,
                    0000000000000.0000001,
                    pd.NA,
                    -2342349234.00934580345,
                ]
            }
        )
        assert str(nan_df["a"].dtype) == "float64"
        # pd.NA values turn the column into 'object' in Pandas 2.x
        # In Pandas 3.0+, it may be different but we just check it's not float64
        assert str(na_df["a"].dtype) != "float64"

        nan_result = duckdb_cursor.execute("select * from nan_df").df()
        na_result = duckdb_cursor.execute("select * from na_df").df()
        pd.testing.assert_frame_equal(nan_result, na_result)

        # Mixed with stringified pd.NA values
        na_string_df = pd.DataFrame({"a": [str(pd.NA), str(pd.NA), pd.NA, str(pd.NA), pd.NA, pd.NA, pd.NA, str(pd.NA)]})  # noqa: F841
        null_indices = [2, 4, 5, 6]
        res = duckdb_cursor.execute("select * from na_string_df").fetchall()
        items = [x[0] for x in list(res)]
        assert_nullness(items, null_indices)

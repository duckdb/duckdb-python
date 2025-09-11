import pytest
from conftest import ArrowPandas, NumpyPandas

import duckdb


@pytest.mark.parametrize("pandas", [NumpyPandas(), ArrowPandas()])
def test_import_cache_explicit_dtype(pandas):
    df = pandas.DataFrame(  # noqa: F841
        {
            "id": [1, 2, 3],
            "value": pandas.Series(["123.123", pandas.NaT, pandas.NA], dtype=pandas.StringDtype(storage="python")),
        }
    )
    con = duckdb.connect()
    result_df = con.query("select id, value from df").df()

    assert result_df["value"][1] is None
    assert result_df["value"][2] is None


@pytest.mark.parametrize("pandas", [NumpyPandas(), ArrowPandas()])
def test_import_cache_implicit_dtype(pandas):
    df = pandas.DataFrame({"id": [1, 2, 3], "value": pandas.Series(["123.123", pandas.NaT, pandas.NA])})  # noqa: F841
    con = duckdb.connect()
    result_df = con.query("select id, value from df").df()

    assert result_df["value"][1] is None
    assert result_df["value"][2] is None

import warnings

import pytest
from packaging.version import Version

import duckdb
import pandas as pd


@pytest.mark.skipif(
    Version(pd.__version__) < Version("3.0"), reason="Pandas < 3.0 doesn't have the new string type yet"
)
def test_new_str_type_pandas_3_0():
    df = pd.DataFrame({"s": ["DuckDB"]})  # noqa: F841
    duckdb.sql("select * from df")


def _infer_string_frame(values):
    with pd.option_context("future.infer_string", True):
        return pd.DataFrame({"s": values})


@pytest.mark.skipif(Version(pd.__version__) >= Version("3.0"), reason="Pandas >= 3.0 has the new string type")
@pytest.mark.parametrize("rows", [2, 1000])
@pytest.mark.parametrize(
    "make_frame",
    [
        pytest.param(_infer_string_frame, id="infer_string"),
        pytest.param(
            lambda values: pd.DataFrame({"s": pd.Series(values, dtype="string[pyarrow]")}), id="string_pyarrow"
        ),
    ],
)
def test_new_str_type_pandas_lt_3_0(duckdb_cursor, make_frame, rows):
    values = ["DuckDB", None] * (rows // 2)
    df = make_frame(values)  # noqa: F841
    # Under the warnings as errors filter the deprecation warning on _data makes it look absent, which hides the
    # strides error that reading it causes.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = duckdb_cursor.sql("select * from df").fetchall()
    assert result == [("DuckDB",), (None,)] * (rows // 2)
    assert [warning for warning in caught if issubclass(warning.category, FutureWarning)] == []

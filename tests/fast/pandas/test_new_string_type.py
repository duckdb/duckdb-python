import pandas as pd
import pytest
from packaging.version import Version

import duckdb


@pytest.mark.skipif(
    Version(pd.__version__) < Version("3.0"), reason="Pandas < 3.0 doesn't have the new string type yet"
)
def test_new_str_type_pandas_3_0():
    df = pd.DataFrame({"s": ["DuckDB"]})  # noqa: F841
    duckdb.sql("select * from df")


@pytest.mark.skipif(Version(pd.__version__) >= Version("3.0"), reason="Pandas >= 3.0 has the new string type")
def test_new_str_type_pandas_lt_3_0():
    pd.options.future.infer_string = True
    df = pd.DataFrame({"s": ["DuckDB"]})  # noqa: F841
    duckdb.sql("select * from df")

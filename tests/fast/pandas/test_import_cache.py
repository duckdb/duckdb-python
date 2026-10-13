import importlib.util
import subprocess
import sys
import textwrap

import pytest

import duckdb
import pandas as pd


@pytest.mark.parametrize(
    "string_dtype",
    [
        "python",
        pytest.param(
            "pyarrow", marks=pytest.mark.skipif(not importlib.util.find_spec("pyarrow"), reason="pyarrow not installed")
        ),
    ],
)
def test_import_cache_explicit_dtype(string_dtype):
    df = pd.DataFrame(  # noqa: F841
        {
            "id": [1, 2, 3],
            "value": pd.Series(["123.123", pd.NaT, pd.NA], dtype=pd.StringDtype(storage=string_dtype)),
        }
    )
    con = duckdb.connect()
    result_df = con.query("select id, value from df").df()

    assert pd.isna(result_df["value"][1])
    assert pd.isna(result_df["value"][2])


def test_import_cache_implicit_dtype():
    df = pd.DataFrame({"id": [1, 2, 3], "value": pd.Series(["123.123", pd.NaT, pd.NA])})  # noqa: F841
    con = duckdb.connect()
    result_df = con.query("select id, value from df").df()

    assert pd.isna(result_df["value"][1])
    assert pd.isna(result_df["value"][2])


def run_without_pandas(script, *args):
    # A fresh interpreter, because this one imported pandas at the top of the file
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args], capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


@pytest.mark.parametrize("pandas_state", ["not_imported", "blocked"])
def test_converting_values_does_not_import_pandas(pandas_state):
    output = run_without_pandas(
        """
        import builtins
        import sys
        import duckdb
        from duckdb.sqltypes import BIGINT

        if sys.argv[1] == "blocked":
            sys.modules["pandas"] = None
        attempts = []
        real_import = builtins.__import__

        def recording_import(name, *args, **kwargs):
            if name == "pandas":
                attempts.append(name)
            return real_import(name, *args, **kwargs)

        builtins.__import__ = recording_import
        con = duckdb.connect()
        con.execute("SET threads = 4")
        con.create_function("identity", abs, [BIGINT], BIGINT)
        assert con.execute("SELECT sum(identity(i)) FROM range(10000) t(i)").fetchall() == [(49995000,)]
        assert con.execute("SELECT ?::BIGINT + 1, ?", [41, "a"]).fetchall() == [(42, "a")]
        assert con.values([1, None]).fetchall() == [(1, None)]
        print("attempts", len(attempts), "imported", sys.modules.get("pandas") is not None)
        """,
        pandas_state,
    )
    assert "attempts 0 imported False" in output


def test_nat_and_na_convert_to_null_once_pandas_is_imported():
    output = run_without_pandas(
        """
        import duckdb
        from duckdb.sqltypes import BIGINT, TIMESTAMP

        con = duckdb.connect()
        con.create_function("identity", abs, [BIGINT], BIGINT)
        con.execute("SELECT identity(1)").fetchall()

        import pandas as pd

        con.create_function("nat", lambda x: pd.NaT, [BIGINT], TIMESTAMP)
        con.create_function("na", lambda x: pd.NA, [BIGINT], BIGINT)
        for _ in range(2):
            assert con.execute("SELECT nat(1), na(1)").fetchall() == [(None, None)]
            assert con.execute("SELECT ?::TIMESTAMP, ?::BIGINT", [pd.NaT, pd.NA]).fetchall() == [(None, None)]
        print("converted")
        """
    )
    assert "converted" in output


@pytest.mark.parametrize("source", ["numpy_dict", "pandas_frame"])
def test_nat_and_na_in_a_string_object_column_scan_as_null(source):
    output = run_without_pandas(
        """
        import sys
        import duckdb
        import numpy as np
        import pandas as pd

        values = np.array(["x"] * 100000, dtype=object)
        values[1], values[2], values[-1] = pd.NA, pd.NaT, pd.NA
        frame = {"a": values} if sys.argv[1] == "numpy_dict" else pd.DataFrame({"a": values})
        con = duckdb.connect()
        query = "SELECT count(*) FILTER (a IS NULL), count(*) FILTER (a IN ('<NA>', 'NaT')) FROM frame"
        rows = con.execute(query).fetchall()
        print("nulls, strings:", rows)
        """,
        source,
    )
    assert "nulls, strings: [(3, 0)]" in output

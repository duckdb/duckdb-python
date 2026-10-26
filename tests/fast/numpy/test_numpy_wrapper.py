"""Correctness contract for the internal NumpyArray façade.

The C++ `NumpyArray` wrapper is the single place that owns the numpy-array
representation (allocate / raw-buffer pointer / resize). It is exercised on two
paths: building a result into numpy (`fetchnumpy`) and scanning a numpy-backed
DataFrame back into DuckDB. These tests pin the properties we rely on:

  * the resize-across-capacity path stays correct -- the result buffer is grown
    by doubling once a result exceeds the initial vector size, and the wrapper
    must refresh its cached data pointer afterwards. A stale pointer here would
    silently corrupt rows past the first resize boundary (not crash), so we
    assert exact element equality across sizes that force several doublings.
  * the `object` dtype works -- strings / nulls / nested values map to numpy
    `object` arrays, which the DLPack-based `nb::ndarray` cannot represent and we
    therefore route around.
  * empty and single-row results don't misbehave at the boundaries.

The wrapper is C++-internal, so it is verified through its observable behaviour
rather than directly. These checks are backend-agnostic (pybind11 or nanobind).
"""

import pytest

import duckdb
import numpy as np
import pandas as pd


@pytest.fixture
def con():
    return duckdb.connect()


class TestNumpyArrayResize:
    """The result -> numpy path, across sizes that force 0..several Resize() calls."""

    # 0/1 = edges; 2048 = the standard vector size; 2049/5000/20001 force resizes.
    @pytest.mark.parametrize("n", [0, 1, 2048, 2049, 5000, 20001])
    def test_int_column_exact(self, con, n):
        got = con.execute(f"SELECT i FROM range({n}) t(i)").fetchnumpy()["i"]
        assert len(got) == n
        np.testing.assert_array_equal(got, np.arange(n, dtype=got.dtype))

    def test_float_column_exact_after_resize(self, con):
        n = 10000
        got = con.execute(f"SELECT i::DOUBLE * 0.5 AS v FROM range({n}) t(i)").fetchnumpy()["v"]
        np.testing.assert_array_equal(got, np.arange(n, dtype="float64") * 0.5)


class TestNumpyArrayObjectDtype:
    """`object`-dtype arrays (strings/nulls/nested) -- unrepresentable in nb::ndarray."""

    def test_strings_roundtrip_with_resize(self, con):
        n = 5000  # > vector size: the object-dtype buffer is resized too
        got = con.execute(f"SELECT ('s' || i::VARCHAR) AS s FROM range({n}) t(i)").fetchnumpy()["s"]
        assert got.dtype == object
        assert list(got) == [f"s{i}" for i in range(n)]

    def test_strings_with_nulls(self, con):
        n = 5000
        got = con.execute(
            f"SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE i::VARCHAR END AS s FROM range({n}) t(i)"
        ).fetchnumpy()["s"]
        # NULLs in an object column come back as a numpy masked array (this also exercises the
        # separate mask buffer, which is allocated/resized through the same NumpyArray façade).
        mask = np.ma.getmaskarray(got)
        assert mask.tolist() == [i % 2 == 0 for i in range(n)]
        for i in range(1, n, 2):  # non-null (odd) positions hold the expected strings
            assert got[i] == str(i)

    def test_nested_list_is_object(self, con):
        got = con.execute("SELECT [i, i + 1] AS l FROM range(3000) t(i)").fetchnumpy()["l"]
        assert got.dtype == object
        assert list(got[0]) == [0, 1]
        assert list(got[-1]) == [2999, 3000]


class TestNumpyArrayRoundtrip:
    """Scan (read via NumpyArray.Data) + materialize (write via Resize/MutableData)."""

    def test_large_mixed_dataframe_roundtrip(self, con):
        n = 7000  # forces resizes on the result side; large enough to span chunks
        df = pd.DataFrame(
            {
                "i": np.arange(n, dtype="int64"),
                "f": np.arange(n, dtype="float64") / 3.0,
                "s": [f"x{i}" for i in range(n)],  # object dtype
            }
        )
        con.register("t", df)
        out = con.execute("SELECT * FROM t ORDER BY i").df()
        pd.testing.assert_frame_equal(out.reset_index(drop=True), df)


class TestNumpyArrayNullSlots:
    """The data slot under a NULL holds zero rather than whatever the reused buffer held before."""

    @pytest.mark.parametrize("sql_type", ["DECIMAL(4,1)", "DECIMAL(18,3)", "DECIMAL(38,3)", "HUGEINT", "BIGINT"])
    @pytest.mark.parametrize("n", [10, 3000])
    def test_numeric_null_slot_holds_zero(self, con, sql_type, n):
        # Fetching and dropping a result of the same size first can leave its values in memory the next result reuses.
        con.execute(f"SELECT 7::{sql_type} AS x FROM range({n})").fetchnumpy()
        got = con.execute(f"SELECT CASE WHEN i % 2 = 0 THEN 7::{sql_type} END AS x FROM range({n}) t(i)").fetchnumpy()
        data = np.ma.getdata(got["x"])
        assert np.ma.getmaskarray(got["x"]).tolist() == [i % 2 == 1 for i in range(n)]
        assert data[1::2].tolist() == [0] * (n // 2)
        assert data[::2].tolist() == [7] * (n // 2)

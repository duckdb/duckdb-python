import gc
import platform
import sys

import pytest

import duckdb


@pytest.mark.parametrize(("rows", "iters"), [(1000, 20)])
def test_python_scalar_udf_return_value_refcount_does_not_leak(rows, iters):
    if platform.python_implementation() != "CPython":
        pytest.skip("refcount-based test requires CPython")

    payload = b"processed_data_" + b"x" * 8192  # large-ish bytes to mimic the reported issue

    def udf_bytes(_):
        return payload  # Always return the exact same object so we can track its refcount.

    # Baseline refcount (note: getrefcount adds a temporary ref)
    baseline = sys.getrefcount(payload)

    con = duckdb.connect()
    con.create_function("udf_bytes", udf_bytes, ["BIGINT"], "VARCHAR")

    for _ in range(iters):
        con.execute(f"SELECT udf_bytes(range) FROM range({rows})")
        res = con.fetchall()
        # Drop the result ASAP so we don't keep any refs alive in Python
        del res
        gc.collect()

    # Re-check refcount. In the buggy version this grows by rows*iters (huge).
    after = sys.getrefcount(payload)

    # Allow a tiny tolerance for transient references/caches.
    # In the presence of the leak, this will be thousands+ higher.
    assert after <= baseline + 10, (baseline, after)

    con.close()

"""String coercion at identifier / parameter-key / separator sites.

nanobind's nb::cast<std::string> is stricter than pybind11's: it rejects bytes and non-str scalars and surfaces a
raw ``RuntimeError: std::bad_cast`` instead of pybind11's lenient conversion. The ``cast_to_string`` helper restores
the lenient behavior (str as-is, bytes UTF-8 decoded, anything else stringified via str()). These guard the
std::bad_cast regression and confirm the realistic cases still match pybind11.
"""

import duckdb


def test_execute_int_param_key():
    """An int parameter-dict key stringifies (so {1: v} fills positional $1), matching pybind11."""
    con = duckdb.connect()
    assert con.execute("SELECT $1 AS a", {1: 5}).fetchall() == [(5,)]


def test_execute_str_param_key():
    con = duckdb.connect()
    assert con.execute("SELECT $name AS a", {"name": 7}).fetchall() == [(7,)]


def test_struct_type_int_field_key():
    """An int struct field-name key stringifies to "1" (matching pybind11), not a raw std::bad_cast."""
    assert str(duckdb.struct_type({1: "INTEGER"})) == 'STRUCT("1" INTEGER)'


def test_struct_type_str_field_key():
    assert str(duckdb.struct_type({"a": "INTEGER"})) == "STRUCT(a INTEGER)"


def test_bytes_param_key_decodes():
    """A bytes param-dict key is UTF-8 decoded (b'1' -> '1'); bytes consistently decode at coercion sites."""
    con = duckdb.connect()
    assert con.execute("SELECT $1 AS a", {b"1": 5}).fetchall() == [(5,)]


def test_bytes_struct_field_key_decodes():
    """A bytes struct field-name key is UTF-8 decoded (b'a' -> 'a'); bytes consistently decode at coercion sites.

    These coercion sites previously surfaced a raw 'std::bad_cast' for non-str input; each test here asserting a
    concrete result also guards that regression (a std::bad_cast would raise and fail the assertion).
    """
    assert str(duckdb.struct_type({b"a": "INTEGER"})) == "STRUCT(a INTEGER)"

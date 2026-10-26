"""Regression tests for bugs found in the pybind11 -> nanobind cutover (PR #522).

Each class targets one finding from the adversarial review and is written to FAIL on the
pre-fix binary and PASS after the fix. Findings that live in existing subsystem suites
(arrow NaN pushdown #9, .map leak #10, filesystem hardening #11/#12/#13) have their
regression tests next to those suites instead.
"""

from __future__ import annotations

import inspect

import pytest

import duckdb
import numpy as np


def _write_csv(path):
    path.write_text("a,b\n1,2\n3,4\n")
    return str(path)


# ===========================================================================
# #1 read_csv / from_csv_auto lost the `path_or_buffer` keyword argument
# ===========================================================================


class TestReadCsvPathOrBufferKeyword:
    def test_module_positional(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv")
        assert duckdb.read_csv(p).fetchall() == [(1, 2), (3, 4)]

    def test_module_path_or_buffer_keyword(self, tmp_path):
        # The regression: `path_or_buffer=` raised TypeError on the branch (stubs still advertise it).
        p = _write_csv(tmp_path / "f.csv")
        assert duckdb.read_csv(path_or_buffer=p).fetchall() == [(1, 2), (3, 4)]

    def test_module_from_csv_auto_path_or_buffer_keyword(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv")
        assert duckdb.from_csv_auto(path_or_buffer=p).fetchall() == [(1, 2), (3, 4)]

    def test_connection_positional(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv")
        con = duckdb.connect()
        assert con.read_csv(p).fetchall() == [(1, 2), (3, 4)]

    def test_connection_path_or_buffer_keyword(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv")
        con = duckdb.connect()
        assert con.read_csv(path_or_buffer=p).fetchall() == [(1, 2), (3, 4)]

    def test_module_connection_keyword_resolves(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv")
        con = duckdb.connect()
        assert duckdb.read_csv(p, connection=con).fetchall() == [(1, 2), (3, 4)]

    def test_module_conn_keyword_resolves(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv")
        con = duckdb.connect()
        assert duckdb.read_csv(p, conn=con).fetchall() == [(1, 2), (3, 4)]

    def test_module_path_or_buffer_and_connection_keywords(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv")
        con = duckdb.connect()
        assert duckdb.read_csv(path_or_buffer=p, connection=con).fetchall() == [(1, 2), (3, 4)]

    def test_real_csv_option_still_honored(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv")
        assert duckdb.read_csv(p, header=True).fetchall() == [(1, 2), (3, 4)]

    def test_unknown_keyword_still_raises(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv")
        with pytest.raises(duckdb.InvalidInputException, match="not_a_real_option"):
            duckdb.read_csv(p, not_a_real_option=1).fetchall()


# ===========================================================================
# #4 module-level duckdb.project made `df` positional-only
# ===========================================================================


class TestProjectDfKeyword:
    def _df(self):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_positional_still_works(self):
        assert duckdb.project(self._df(), "x").fetchall() == [(1,), (2,), (3,)]

    def test_positional_with_connection_keyword(self):
        con = duckdb.connect()
        assert duckdb.project(self._df(), "x", connection=con).fetchall() == [(1,), (2,), (3,)]

    def test_df_keyword_matches_positional_semantics(self):
        # The regression: `df=` raised TypeError (df was positional-only). It must be accepted and behave
        # identically to the positional-df form. With no projection both now refuse.
        with pytest.raises(TypeError):
            duckdb.project(df=self._df(), groups="x")
        with pytest.raises(TypeError):
            duckdb.project(self._df(), groups="x")

    @pytest.mark.parametrize(
        ("projection", "expected"),
        [
            pytest.param("x", [(1,), (2,), (3,)], id="column"),
            pytest.param("x + 1 AS b", [(2,), (3,), (4,)], id="expression"),
        ],
    )
    def test_the_frame_by_keyword_with_the_projection_positional(self, projection, expected):
        assert duckdb.project(projection, df=self._df()).fetchall() == expected

    def test_df_keyword_leaves_only_the_projection_missing(self):
        # A TypeError may come from the missing projection, never from the keyword itself.
        with pytest.raises(TypeError):
            duckdb.project(df=self._df())
        assert duckdb.project(self._df(), "x").fetchall() == [(1,), (2,), (3,)]


# ===========================================================================
# #5 pandas/bind.cpp rejected non-string column labels
# ===========================================================================


class TestPandasNonStringColumnLabels:
    """A DataFrame bound with int/tuple/MultiIndex/datetime labels must not throw."""

    @pytest.fixture(autouse=True)
    def _pd(self):
        self.pd = pytest.importorskip("pandas")

    def test_integer_labels(self):
        df = self.pd.DataFrame(np.arange(6).reshape(2, 3))
        assert duckdb.from_df(df).fetchall() == [(0, 1, 2), (3, 4, 5)]

    def test_transpose_labels(self):
        df = self.pd.DataFrame({"a": [1], "b": [2]}).T
        assert duckdb.from_df(df).fetchall() == [(1,), (2,)]

    def test_tuple_labels(self):
        df = self.pd.DataFrame([[1, 2]], columns=[("x", "y"), ("z", "w")])
        assert duckdb.from_df(df).fetchall() == [(1, 2)]

    def test_multiindex_labels(self):
        df = self.pd.DataFrame([[1, 2]], columns=self.pd.MultiIndex.from_tuples([("a", "b"), ("c", "d")]))
        assert duckdb.from_df(df).fetchall() == [(1, 2)]

    def test_datetime_labels(self):
        df = self.pd.DataFrame([[1, 2]], columns=self.pd.to_datetime(["2020-01-01", "2020-01-02"]))
        assert duckdb.from_df(df).fetchall() == [(1, 2)]


# ===========================================================================
# #3 enum default arguments must render as the registered enum member (not int)
# ===========================================================================


class TestEnumDefaultRendersAsMember:
    """Enum default args must render as the registered member, not a bare int.

    Defaults are materialized through the enum caster's from_cpp at bind time, so it must produce
    `Enum.MEMBER` (not `0`) for help()/__signature__/stubs to be correct.
    """

    def test_create_function_signature_shows_enum_members(self):
        doc = duckdb.create_function.__doc__ or ""
        assert "type: PythonUDFType = PythonUDFType.NATIVE" in doc, doc
        assert "null_handling: FunctionNullHandling = FunctionNullHandling.DEFAULT" in doc, doc
        assert "exception_handling: PythonExceptionHandling = PythonExceptionHandling." in doc, doc
        # The pre-fix regression rendered these as `= 0`.
        assert "type: PythonUDFType = 0" not in doc

    def test_explain_signature_shows_enum_member(self):
        rel = duckdb.sql("select 1 i")
        signature = inspect.signature(type(rel).explain)
        assert signature.parameters["type"].default is duckdb.ExplainType.STANDARD

    def test_nb_signature_default_object_is_enum_member(self):
        # The embedded default objects must be the actual enum members.
        sig = duckdb.create_function.__nb_signature__
        defaults = sig[0][2]
        member_names = {type(d).__name__ for d in defaults if d is not None}
        assert "PythonUDFType" in member_names, defaults


# ===========================================================================
# #14 enum caster still accepts str / int / enum members (convert-path preserved)
#
# The convert-flag gating only changes overload resolution's no-convert first pass, which
# has no live trigger (every enum-typed parameter is a single, non-overloaded def, so the
# convert flag is always set). This test confirms the str/int/enum acceptance the caster is
# supposed to provide still works after the gating change.
# ===========================================================================


class TestEnumCasterAcceptsStrIntEnum:
    def test_explain_accepts_string(self):
        rel = duckdb.sql("select 1 i")
        assert isinstance(rel.explain(type="standard"), str)

    def test_explain_accepts_enum_member(self):
        rel = duckdb.sql("select 1 i")
        assert isinstance(rel.explain(type=duckdb.ExplainType.STANDARD), str)

    def test_create_function_accepts_string_and_enum(self):
        from duckdb.func import PythonUDFType

        con = duckdb.connect()
        con.create_function("f_str", lambda x: x, [int], int, type="native")
        con.create_function("f_enum", lambda x: x, [int], int, type=PythonUDFType.NATIVE)
        assert con.sql("select f_str(21) + f_enum(21)").fetchall()[0] == (42,)


# ===========================================================================
# #2 / #7 numpy object-array allocation (PyArray_NewFromDescr): object columns with NULLs
# must be byte-identical, and the object-dtype descr cache must survive heavy reuse.
#
# #2 is an over-decref on the numpy *allocation-failure* path (proven against numpy source).
# Reliable fault injection from Python is not feasible: a numpy MemoryError needs either true
# OOM or an absurd element count DuckDB will not reach through a query. We therefore rely on
# the numpy-source proof + this success-path byte-identical check + heavy cache reuse (also
# run under ASan by the reviewer). #6 (NumpyArray move-only) is enforced by a compile-time
# static_assert in numpy_array.hpp.
# ===========================================================================


class TestNumpyObjectColumns:
    def test_varchar_with_nulls_fetchnumpy(self):
        na = duckdb.sql("SELECT CASE WHEN i%3=0 THEN NULL ELSE 's'||i END AS v FROM range(9) t(i)").fetchnumpy()
        got = [None if isinstance(x, np.ma.core.MaskedConstant) else x for x in list(na["v"])]
        assert got == [None, "s1", "s2", None, "s4", "s5", None, "s7", "s8"]

    def test_varchar_with_nulls_df(self):
        pd = pytest.importorskip("pandas")
        df = duckdb.sql("SELECT CASE WHEN i%3=0 THEN NULL ELSE 'x'||i END AS v FROM range(6) t(i)").df()
        vals = df["v"].tolist()
        # nulls at i%3==0 -> indices 0 and 3; the rest are 'x<i>'
        assert vals[1] == "x1"
        assert vals[2] == "x2"
        assert vals[4] == "x4"
        assert vals[5] == "x5"
        assert pd.isna(vals[0])
        assert pd.isna(vals[3])

    def test_blob_with_nulls_fetchnumpy(self):
        b = duckdb.sql("SELECT CASE WHEN i%2=0 THEN NULL ELSE ('b'||i)::BLOB END AS v FROM range(6) t(i)").fetchnumpy()
        got = [None if isinstance(x, np.ma.core.MaskedConstant) else bytes(x) for x in list(b["v"])]
        assert got == [None, b"b1", None, b"b3", None, b"b5"]

    def test_list_of_varchar_object_arrays(self):
        lv = duckdb.sql("SELECT [v, v] AS l FROM (SELECT 's'||i AS v FROM range(5) t(i))").fetchnumpy()
        assert [list(x) for x in lv["l"]] == [[f"s{i}", f"s{i}"] for i in range(5)]

    def test_object_descr_cache_heavy_reuse(self):
        # Exercise the process-lifetime object-dtype descr cache many times across several object
        # dtypes; a mismanaged cache ref (the #2 class of bug) tends to surface as a crash here.
        for _ in range(200):
            r = duckdb.sql("SELECT i::VARCHAR v, ('b'||i)::BLOB b, [i::VARCHAR] l FROM range(64) t(i)").fetchnumpy()
            assert len(r["v"]) == 64

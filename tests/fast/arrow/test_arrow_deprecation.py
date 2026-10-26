import warnings

import pytest

import duckdb

pytest.importorskip("pyarrow")


class TestArrowDeprecation:
    @pytest.fixture(autouse=True)
    def setup(self, duckdb_cursor):
        self.con = duckdb_cursor
        self.con.execute("CREATE TABLE t AS SELECT 1 AS a")

    def test_relation_fetch_arrow_table_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(
            DeprecationWarning, match="fetch_arrow_table\\(\\) is deprecated, use to_arrow_table\\(\\) instead"
        ):
            rel.fetch_arrow_table()

    def test_relation_fetch_record_batch_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(
            DeprecationWarning, match="fetch_record_batch\\(\\) is deprecated, use to_arrow_reader\\(\\) instead"
        ):
            rel.fetch_record_batch()

    def test_relation_fetch_arrow_reader_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(
            DeprecationWarning, match="fetch_arrow_reader\\(\\) is deprecated, use to_arrow_reader\\(\\) instead"
        ):
            rel.fetch_arrow_reader()

    def test_relation_to_arrow_table_works(self):
        rel = self.con.table("t")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = rel.to_arrow_table()
        assert result.num_rows == 1

    def test_relation_to_arrow_reader_works(self):
        rel = self.con.table("t")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = rel.to_arrow_reader()
        assert reader.read_all().num_rows == 1

    def test_relation_arrow_no_warning(self):
        """relation.arrow() should NOT emit a deprecation warning (soft deprecated)."""
        rel = self.con.table("t")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = rel.arrow()
        assert reader.read_all().num_rows == 1

    def test_from_arrow_not_deprecated(self):
        """duckdb.arrow(arrow_object) should NOT emit a deprecation warning."""
        import pyarrow as pa

        table = pa.table({"a": [1, 2, 3]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rel = duckdb.arrow(table)
        assert rel.fetchall() == [(1,), (2,), (3,)]

    def test_the_dbapi_has_to_arrow_table_and_to_arrow_reader_and_no_deprecated_arrow_names(self):
        pa = pytest.importorskip("pyarrow")
        present = ["to_arrow_table", "to_arrow_reader"]
        gone = ["fetch_arrow_table", "fetch_record_batch", "pl"]
        with duckdb.connect() as con:
            for name in [*gone, "arrow"]:
                assert not hasattr(con, name)
            for name in present:
                assert callable(getattr(con, name))
        for name in gone:
            assert not hasattr(duckdb, name)
        for name in present:
            assert callable(getattr(duckdb, name))
        # duckdb.arrow() builds a relation from an Arrow object and has no reader form.
        with pytest.raises(TypeError):
            duckdb.arrow()
        assert isinstance(duckdb.arrow(pa.table({"a": [1]})), duckdb.DuckDBPyRelation)

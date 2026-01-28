import warnings

import pytest

import duckdb


class TestArrowDeprecation:
    @pytest.fixture(autouse=True)
    def setup(self, duckdb_cursor):
        self.con = duckdb_cursor
        self.con.execute("CREATE TABLE t AS SELECT 1 AS a")

    def test_relation_fetch_arrow_table_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(
            DeprecationWarning, match="fetch_arrow_table\\(\\) is deprecated, use arrow_table\\(\\) instead"
        ):
            rel.fetch_arrow_table()

    def test_relation_to_arrow_table_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(
            DeprecationWarning, match="to_arrow_table\\(\\) is deprecated, use arrow_table\\(\\) instead"
        ):
            rel.to_arrow_table()

    def test_relation_arrow_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(DeprecationWarning, match="arrow\\(\\) is deprecated, use arrow_reader\\(\\) instead"):
            rel.arrow()

    def test_relation_fetch_record_batch_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(
            DeprecationWarning, match="fetch_record_batch\\(\\) is deprecated, use arrow_reader\\(\\) instead"
        ):
            rel.fetch_record_batch()

    def test_relation_fetch_arrow_reader_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(
            DeprecationWarning, match="fetch_arrow_reader\\(\\) is deprecated, use arrow_reader\\(\\) instead"
        ):
            rel.fetch_arrow_reader()

    def test_connection_fetch_arrow_table_deprecated(self):
        self.con.execute("SELECT 1")
        with pytest.warns(
            DeprecationWarning, match="fetch_arrow_table\\(\\) is deprecated, use arrow_table\\(\\) instead"
        ):
            self.con.fetch_arrow_table()

    def test_connection_fetch_record_batch_deprecated(self):
        self.con.execute("SELECT 1")
        with pytest.warns(
            DeprecationWarning, match="fetch_record_batch\\(\\) is deprecated, use arrow_reader\\(\\) instead"
        ):
            self.con.fetch_record_batch()

    def test_connection_arrow_deprecated(self):
        self.con.execute("SELECT 1")
        with pytest.warns(DeprecationWarning, match="arrow\\(\\) is deprecated, use arrow_reader\\(\\) instead"):
            self.con.arrow()

    def test_module_fetch_arrow_table_deprecated(self):
        duckdb.execute("SELECT 1")
        with pytest.warns(
            DeprecationWarning, match="fetch_arrow_table\\(\\) is deprecated, use arrow_table\\(\\) instead"
        ):
            duckdb.fetch_arrow_table()

    def test_module_fetch_record_batch_deprecated(self):
        duckdb.execute("SELECT 1")
        with pytest.warns(
            DeprecationWarning, match="fetch_record_batch\\(\\) is deprecated, use arrow_reader\\(\\) instead"
        ):
            duckdb.fetch_record_batch()

    def test_relation_arrow_table_works(self):
        rel = self.con.table("t")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = rel.arrow_table()
        assert result.num_rows == 1

    def test_relation_arrow_reader_works(self):
        rel = self.con.table("t")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = rel.arrow_reader()
        assert reader.read_all().num_rows == 1

    def test_connection_arrow_table_works(self):
        self.con.execute("SELECT 1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = self.con.arrow_table()
        assert result.num_rows == 1

    def test_connection_arrow_reader_works(self):
        self.con.execute("SELECT 1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = self.con.arrow_reader()
        assert reader.read_all().num_rows == 1

    def test_module_arrow_table_works(self):
        duckdb.execute("SELECT 1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = duckdb.arrow_table()
        assert result.num_rows == 1

    def test_module_arrow_reader_works(self):
        duckdb.execute("SELECT 1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = duckdb.arrow_reader()
        assert reader.read_all().num_rows == 1

    def test_from_arrow_not_deprecated(self):
        """duckdb.arrow(arrow_object) should NOT emit a deprecation warning."""
        import pyarrow as pa

        table = pa.table({"a": [1, 2, 3]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rel = duckdb.arrow(table)
        assert rel.fetchall() == [(1,), (2,), (3,)]

from io import BytesIO
from pathlib import Path
from typing import NoReturn

import pytest

import duckdb


@pytest.fixture
def parquet_file(tmp_path):
    path = tmp_path / "integers.parquet"
    duckdb.sql("SELECT i FROM range(10) t(i)").write_parquet(str(path))
    return path


@pytest.fixture
def parquet_files(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    file1 = directory / "file1.parquet"
    file2 = directory / "file2.parquet"
    duckdb.sql("SELECT 1 AS i").write_parquet(str(file1))
    duckdb.sql("SELECT 2 AS i").write_parquet(str(file2))
    return file1, file2


@pytest.fixture
def parquet_bytes(parquet_file):
    return Path(parquet_file).read_bytes()


class TestReadParquet:
    # Regression / backwards-compat

    def test_read_string(self, duckdb_cursor, parquet_file):
        res = duckdb_cursor.read_parquet(str(parquet_file)).fetchall()
        assert res == [(i,) for i in range(10)]

    def test_read_list_of_strings(self, duckdb_cursor, parquet_files):
        file1, file2 = parquet_files
        res = duckdb_cursor.read_parquet([str(file1), str(file2)]).order("i").fetchall()
        assert res == [(1,), (2,)]

    def test_read_glob_string(self, duckdb_cursor, parquet_files):
        glob = str(parquet_files[0].parent / "*.parquet")
        res = duckdb_cursor.read_parquet(glob).order("i").fetchall()
        assert res == [(1,), (2,)]

    def test_path_not_mangled(self, duckdb_cursor):
        # The path string should be forwarded verbatim to the engine
        with pytest.raises(duckdb.IOException, match="no_such_directory"):
            duckdb_cursor.read_parquet("no_such_directory/*.parquet").fetchall()

    def test_options_still_thread_through(self, duckdb_cursor, parquet_file):
        rel = duckdb_cursor.read_parquet(
            parquet_file,
            binary_as_string=True,
            file_row_number=True,
            filename=True,
            hive_partitioning=False,
            union_by_name=False,
        )
        assert set(rel.columns) == {"i", "file_row_number", "filename"}

    def test_compression_option(self, duckdb_cursor, parquet_file):
        res = duckdb_cursor.read_parquet(str(parquet_file), compression="snappy").fetchall()
        assert res == [(i,) for i in range(10)]
        with pytest.raises(duckdb.InvalidInputException, match="only accepts 'compression' as a string"):
            duckdb_cursor.read_parquet(str(parquet_file), compression=42)

    # New capability: pathlib.Path

    def test_read_pathlib_path(self, duckdb_cursor, parquet_file):
        res = duckdb_cursor.read_parquet(parquet_file).fetchall()
        assert res == [(i,) for i in range(10)]

    def test_read_pathlib_path_glob(self, duckdb_cursor, parquet_files):
        # Globs survive Path stringification and resolve downstream
        glob = parquet_files[0].parent / "*.parquet"
        res = duckdb_cursor.read_parquet(glob).order("i").fetchall()
        assert res == [(1,), (2,)]

    def test_read_pathlike(self, duckdb_cursor, parquet_file):
        class MyPath:
            def __init__(self, path) -> None:
                self._path = path

            def __fspath__(self) -> str:
                return str(self._path)

        res = duckdb_cursor.read_parquet(MyPath(parquet_file)).fetchall()
        assert res == [(i,) for i in range(10)]

    def test_read_bytes_path(self, duckdb_cursor, parquet_file):
        res = duckdb_cursor.read_parquet(str(parquet_file).encode()).fetchall()
        assert res == [(i,) for i in range(10)]

    def test_read_mixed_list(self, duckdb_cursor, parquet_files):
        file1, file2 = parquet_files
        res = duckdb_cursor.read_parquet([Path(file1), str(file2)]).order("i").fetchall()
        assert res == [(1,), (2,)]

    def test_read_list_of_paths(self, duckdb_cursor, parquet_files):
        file1, file2 = parquet_files
        res = duckdb_cursor.read_parquet([Path(file1), Path(file2)]).order("i").fetchall()
        assert res == [(1,), (2,)]

    # New capability: file-like objects

    def test_read_filelike(self, duckdb_cursor, parquet_bytes):
        pytest.importorskip("fsspec")
        res = duckdb_cursor.read_parquet(BytesIO(parquet_bytes)).fetchall()
        assert res == [(i,) for i in range(10)]

    def test_read_filelike_list(self, duckdb_cursor, parquet_bytes):
        pytest.importorskip("fsspec")
        res = duckdb_cursor.read_parquet([BytesIO(parquet_bytes), BytesIO(parquet_bytes)]).fetchall()
        assert res == [(i,) for i in range(10)] * 2

    def test_read_filelike_filename_column(self, duckdb_cursor, parquet_bytes):
        # The filename column exposes the generated internal name for file-like objects
        pytest.importorskip("fsspec")
        rel = duckdb_cursor.read_parquet(BytesIO(parquet_bytes), filename=True)
        res = rel.fetchall()
        assert all(row[1].startswith("DUCKDB_INTERNAL_OBJECTSTORE://") for row in res)

    def test_read_filelike_rel_out_of_scope(self, duckdb_cursor, parquet_bytes):
        pytest.importorskip("fsspec")

        def keep_in_scope():
            # The relation keeps the registered file-like object alive
            return duckdb_cursor.read_parquet(BytesIO(parquet_bytes))

        def close_scope():
            return duckdb_cursor.read_parquet(BytesIO(parquet_bytes)).fetchall()

        relation = keep_in_scope()
        res = relation.fetchall()

        res2 = close_scope()
        assert res == res2

    # All four entry points

    def test_module_read_parquet(self, parquet_file):
        assert duckdb.read_parquet(parquet_file).fetchall() == [(i,) for i in range(10)]

    def test_module_from_parquet(self, parquet_file):
        assert duckdb.from_parquet(parquet_file).fetchall() == [(i,) for i in range(10)]

    def test_connection_from_parquet(self, duckdb_cursor, parquet_file):
        assert duckdb_cursor.from_parquet(parquet_file).fetchall() == [(i,) for i in range(10)]

    def test_module_level_with_connection_kwarg(self, duckdb_cursor, parquet_file):
        res = duckdb.read_parquet(parquet_file, connection=duckdb_cursor).fetchall()
        assert res == [(i,) for i in range(10)]

    # Error handling

    def test_nonexistent_file(self, duckdb_cursor, tmp_path):
        missing = tmp_path / "missing.parquet"
        with pytest.raises(duckdb.IOException, match=r"missing\.parquet"):
            duckdb_cursor.read_parquet(str(missing)).fetchall()

    def test_empty_list(self, duckdb_cursor):
        with pytest.raises(duckdb.InvalidInputException, match="non-empty list or tuple of paths or file-like objects"):
            duckdb_cursor.read_parquet([])

    def test_filelike_exception(self, duckdb_cursor):
        pytest.importorskip("fsspec")

        class ReadError:
            def read(self, amount=-1) -> NoReturn:
                raise ValueError(amount)

            def seek(self, loc) -> int:
                return 0

        class SeekError:
            def read(self, amount=-1) -> bytes:
                return b"test"

            def seek(self, loc) -> NoReturn:
                raise ValueError(loc)

        # The MemoryFileSystem copies the content with 'read', so this fails instantly
        with pytest.raises(ValueError, match="-1"):
            duckdb_cursor.read_parquet(ReadError())

        # 'seek' is never called on the object itself; the copied content is just not valid parquet
        with pytest.raises(duckdb.InvalidInputException, match="too small to be a Parquet file"):
            duckdb_cursor.read_parquet(SeekError())

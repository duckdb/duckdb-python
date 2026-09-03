import datetime
import io

import pytest

import duckdb

fsspec = pytest.importorskip("fsspec")


def _register_blob_filesystem(duckdb_cursor, protocol, modified_fn):
    """Register a tiny fsspec filesystem that serves one in-memory blob."""

    class BlobFileSystem(fsspec.AbstractFileSystem):
        def ls(self, path, detail=True, **kwargs):
            vals = [k for k in self._data if k.startswith(path)]
            if detail:
                return [
                    {"name": name, "size": len(self._data[name]), "type": "file", "created": 0, "islink": False}
                    for name in vals
                ]
            return vals

        def modified(self, path):
            return modified_fn(path)

        def _open(self, path, **kwargs):
            return io.BytesIO(self._data[path])

        def info(self, path, **kwargs):
            return {"name": path, "size": len(self._data[path]), "type": "file"}

        def __init__(self) -> None:
            super().__init__()
            self._data = {"blob": b"hello"}

    BlobFileSystem.protocol = protocol
    fsspec.register_implementation(protocol, BlobFileSystem, clobber=True)
    duckdb_cursor.register_filesystem(fsspec.filesystem(protocol))
    return f"{protocol}://blob"


class TestReadParquet:
    def test_fsspec_deadlock(self, duckdb_cursor, tmp_path):
        # Create test parquet data
        file_path = tmp_path / "data.parquet"
        duckdb_cursor.sql(f"COPY (FROM range(50_000)) TO '{file_path!s}' (FORMAT parquet)")
        parquet_data = file_path.read_bytes()

        class TestFileSystem(fsspec.AbstractFileSystem):
            protocol = "deadlock"

            @property
            def fsid(self) -> str:
                return "deadlock"

            def ls(self, path, detail=True, **kwargs):
                vals = [k for k in self._data if k.startswith(path)]
                if detail:
                    return [
                        {
                            "name": path,
                            "size": len(self._data[path]),
                            "type": "file",
                            "created": 0,
                            "islink": False,
                        }
                        for path in vals
                    ]
                else:
                    return vals

            def modified(self, path):
                # this is needed since PR #16463 because the Parquet reader now always fetches the modified timestamp
                return datetime.datetime.now()

            def _open(self, path, **kwargs):
                return io.BytesIO(self._data[path])

            def __init__(self) -> None:
                super().__init__()
                self._data = {"a": parquet_data, "b": parquet_data}

        fsspec.register_implementation("deadlock", TestFileSystem, clobber=True)
        fs = fsspec.filesystem("deadlock")
        duckdb_cursor.register_filesystem(fs)

        result = duckdb_cursor.read_parquet(["deadlock://a", "deadlock://b"], union_by_name=True)
        assert len(result.fetchall()) == 100_000

    def test_fsspec_seek_read_atomicity(self, duckdb_cursor, tmp_path):
        """Regression test: concurrent positional reads must be atomic (seek+read under one GIL hold).

        Without the fix, separate seek and read GIL acquisitions allow another thread to
        seek the same handle between them, corrupting data. We stress this by reading 4 files
        with distinct data in parallel (union_by_name) and verifying no cross-contamination.
        """
        files = {}
        for i, name in enumerate(["a", "b", "c", "d"]):
            file_path = tmp_path / f"{name}.parquet"
            duckdb_cursor.sql(f"COPY (SELECT {i} AS file_id FROM range(10000)) TO '{file_path!s}' (FORMAT parquet)")
            files[name] = file_path.read_bytes()

        class AtomicityTestFS(fsspec.AbstractFileSystem):
            protocol = "atomtest"

            @property
            def fsid(self):
                return "atomtest"

            def ls(self, path, detail=True, **kwargs):
                vals = [k for k in self._data if k.startswith(path)]
                if detail:
                    return [
                        {"name": p, "size": len(self._data[p]), "type": "file", "created": 0, "islink": False}
                        for p in vals
                    ]
                return vals

            def modified(self, path):
                return datetime.datetime.now()

            def _open(self, path, **kwargs):
                return io.BytesIO(self._data[path])

            def __init__(self) -> None:
                super().__init__()
                self._data = files

        fsspec.register_implementation("atomtest", AtomicityTestFS, clobber=True)
        duckdb_cursor.register_filesystem(fsspec.filesystem("atomtest"))

        globs = ["atomtest://a", "atomtest://b", "atomtest://c", "atomtest://d"]
        for _ in range(10):
            result = duckdb_cursor.sql(
                f"SELECT file_id, count(*) AS cnt FROM read_parquet({globs}, union_by_name=true) "
                "GROUP BY ALL ORDER BY file_id"
            ).fetchall()
            assert result == [(0, 10000), (1, 10000), (2, 10000), (3, 10000)]


class TestLastModified:
    def test_unsupported_modified_is_null(self, duckdb_cursor):
        def raise_not_implemented(_path):
            msg = "no mtime"
            raise NotImplementedError(msg)

        path = _register_blob_filesystem(duckdb_cursor, "nomtime", raise_not_implemented)
        result = duckdb_cursor.sql(f"SELECT last_modified FROM read_blob('{path}')").fetchall()
        assert result == [(None,)]

    def test_mtime_keyerror_is_null(self, duckdb_cursor):
        def raise_mtime_key_error(_path):
            key = "mtime"
            raise KeyError(key)

        path = _register_blob_filesystem(duckdb_cursor, "gcsmt", raise_mtime_key_error)
        result = duckdb_cursor.sql(f"SELECT last_modified FROM read_blob('{path}')").fetchall()
        assert result == [(None,)]

    def test_other_modified_errors_still_fail(self, duckdb_cursor):
        def raise_os_error(_path):
            msg = "simulated I/O failure"
            raise OSError(msg)

        path = _register_blob_filesystem(duckdb_cursor, "badmtime", raise_os_error)
        with pytest.raises(duckdb.Error, match="simulated I/O failure"):
            duckdb_cursor.sql(f"SELECT last_modified FROM read_blob('{path}')").fetchall()

    def test_modified_timestamp_is_returned(self, duckdb_cursor):
        def known_mtime(_path):
            return datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc)

        path = _register_blob_filesystem(duckdb_cursor, "okmtime", known_mtime)
        result = duckdb_cursor.sql(f"SELECT last_modified FROM read_blob('{path}')").fetchall()
        assert result[0][0] is not None

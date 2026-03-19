import datetime
import io

import pytest

fsspec = pytest.importorskip("fsspec")


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

        result = duckdb_cursor.read_parquet(file_globs=["deadlock://a", "deadlock://b"], union_by_name=True)
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

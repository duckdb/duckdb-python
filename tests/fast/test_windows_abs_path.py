import shutil
import sys
from pathlib import Path

import pytest

import duckdb


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="Tests only run on Windows")
class TestWindowsAbsPath:
    def test_windows_path_accent(self, monkeypatch):
        test_dir = Path.cwd() / "tést"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir()

        dbname = "test.db"
        dbpath = test_dir / dbname
        con = duckdb.connect(str(dbpath))
        con.execute("CREATE OR REPLACE TABLE int AS SELECT * FROM range(10) t(i)")
        res = con.execute("SELECT COUNT(*) FROM int").fetchall()
        assert res[0][0] == 10
        del res
        del con

        monkeypatch.chdir("tést")
        rel_dbpath = Path("..") / dbpath
        con = duckdb.connect(str(rel_dbpath))
        res = con.execute("SELECT COUNT(*) FROM int").fetchall()
        assert res[0][0] == 10
        del res
        del con

        con = duckdb.connect(dbname)
        res = con.execute("SELECT COUNT(*) FROM int").fetchall()
        assert res[0][0] == 10
        del res
        del con

    def test_windows_abs_path(self):
        # setup paths to test with
        dbpath = Path.cwd() / "test.db"
        abspath = str(dbpath.resolve())
        assert abspath[1] == ":"
        no_drive_path = abspath[2:]
        fwd_slash_path = no_drive_path.replace("\\", "/")

        for testpath in (abspath, no_drive_path, fwd_slash_path):
            con = duckdb.connect(testpath)
            con.execute("CREATE OR REPLACE TABLE int AS SELECT * FROM range(10) t(i)")
            res = con.execute("SELECT COUNT(*) FROM int").fetchall()
            assert res[0][0] == 10
            del res
            del con

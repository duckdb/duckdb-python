import logging
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from shutil import copyfileobj

import pytest

import duckdb
from duckdb import DuckDBPyConnection, InvalidInputException

fsspec = pytest.importorskip("fsspec", "2022.11.0")
# fsspec no longer pulls in its implementations submodules on package import
pytest.importorskip("fsspec.implementations.memory")

FILENAME = "integers.csv"

logging.basicConfig(level=logging.DEBUG)


def intercept(monkeypatch: pytest.MonkeyPatch, obj: object, name: str) -> list[str]:
    error_occurred = []
    orig = getattr(obj, name)

    def ceptor(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except Exception as e:
            error_occurred.append(e)
            raise

    monkeypatch.setattr(obj, name, ceptor)
    return error_occurred


@pytest.fixture
def duckdb_cursor():
    with duckdb.connect() as conn:
        yield conn


@pytest.fixture
def memory():
    fs = fsspec.filesystem("memory", skip_instance_cache=True)

    # ensure each instance is independent (to work around a weird quirk in fsspec)
    fs.store = {}
    fs.pseudo_dirs = [""]

    # copy csv into memory filesystem
    add_file(fs)
    return fs


def add_file(fs, filename=FILENAME):
    with (Path(__file__).parent / "data" / filename).open("rb") as source, fs.open(filename, "wb") as dest:
        copyfileobj(source, dest)


class TestPythonFilesystem:
    def test_unregister_non_existent_filesystem(self, duckdb_cursor: DuckDBPyConnection):
        duckdb_cursor.unregister_filesystem("fake")

    def test_memory_filesystem(self, duckdb_cursor: DuckDBPyConnection, memory: fsspec.AbstractFileSystem):
        duckdb_cursor.register_filesystem(memory)

        assert memory.protocol == "memory"

        duckdb_cursor.execute(f"select * from 'memory://{FILENAME}'")

        assert duckdb_cursor.fetchall() == [(1, 10, 0), (2, 50, 30)]

        duckdb_cursor.unregister_filesystem("memory")

    def test_reject_abstract_filesystem(self, duckdb_cursor: DuckDBPyConnection):
        with pytest.raises(InvalidInputException):
            duckdb_cursor.register_filesystem(fsspec.AbstractFileSystem())

    def test_unregister_builtin(self, require: Callable[[str], DuckDBPyConnection]):
        duckdb_cursor = require("httpfs")
        assert duckdb_cursor.filesystem_is_registered("S3FileSystem")
        duckdb_cursor.unregister_filesystem("S3FileSystem")
        assert not duckdb_cursor.filesystem_is_registered("S3FileSystem")

    def test_multiple_protocol_filesystems(self, duckdb_cursor: DuckDBPyConnection):
        class ExtendedMemoryFileSystem(fsspec.implementations.memory.MemoryFileSystem):
            protocol = ("file", "local")
            # defer to the original implementation that doesn't hardcode the protocol
            _strip_protocol = classmethod(fsspec.AbstractFileSystem._strip_protocol.__func__)

        memory = ExtendedMemoryFileSystem(skip_instance_cache=True)
        add_file(memory)
        duckdb_cursor.register_filesystem(memory)
        for protocol in memory.protocol:
            duckdb_cursor.execute(f"select * from '{protocol}://{FILENAME}'")

            assert duckdb_cursor.fetchall() == [(1, 10, 0), (2, 50, 30)]

    def test_write(self, duckdb_cursor: DuckDBPyConnection, memory: fsspec.AbstractFileSystem):
        duckdb_cursor.register_filesystem(memory)

        duckdb_cursor.execute("copy (select 1) to 'memory://01.csv' (FORMAT CSV, HEADER 0)")

        assert memory.open("01.csv").read() == b"1\n"

    def test_null_bytes(self, duckdb_cursor: DuckDBPyConnection, memory: fsspec.AbstractFileSystem):
        with memory.open("test.csv", "wb") as fh:
            fh.write(b"hello\n\0world\0")
        duckdb_cursor.register_filesystem(memory)

        duckdb_cursor.execute("select * from read_csv(\"memory://test.csv\", header = 0, quote = '\"', escape = '\"')")

        assert duckdb_cursor.fetchall() == [("hello",), ("\0world\0",)]

    def test_read_parquet(self, duckdb_cursor: DuckDBPyConnection, memory: fsspec.AbstractFileSystem):
        filename = "binary_string.parquet"
        add_file(memory, filename)

        duckdb_cursor.register_filesystem(memory)

        duckdb_cursor.execute(f"select * from read_parquet('memory://{filename}')")

        assert duckdb_cursor.fetchall() == [(b"foo",), (b"bar",), (b"baz",)]

    def test_write_parquet(self, duckdb_cursor: DuckDBPyConnection, memory: fsspec.AbstractFileSystem):
        duckdb_cursor.register_filesystem(memory)
        filename = "output.parquet"

        duckdb_cursor.execute(f"""COPY (SELECT 1) TO 'memory://{filename}' (FORMAT PARQUET);""")

        assert memory.open(filename).read().startswith(b"PAR1")

    def test_when_fsspec_not_installed(self, duckdb_cursor: DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "fsspec", None)

        with pytest.raises(ModuleNotFoundError):
            duckdb_cursor.register_filesystem(None)

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="ArrowFSWrapper requires python 3.8 or higher")
    def test_arrow_fs_wrapper(self, tmp_path: Path, duckdb_cursor: DuckDBPyConnection):
        fs = pytest.importorskip("pyarrow.fs")
        from fsspec.implementations.arrow import ArrowFSWrapper

        local = fs.LocalFileSystem()
        local_fsspec = ArrowFSWrapper(local, skip_instance_cache=True)
        # posix calls here required as ArrowFSWrapper only supports url-like paths (not Windows paths)
        filename = str(PurePosixPath(tmp_path.as_posix()) / "test.csv")
        with local_fsspec.open(filename, mode="w") as f:
            f.write("a,b,c\n")
            f.write("1,2,3\n")
            f.write("4,5,6\n")

        duckdb_cursor.register_filesystem(local_fsspec)
        duckdb_cursor.execute(f"select * from read_csv_auto('local://{filename}', header=true)")

        assert duckdb_cursor.fetchall() == [(1, 2, 3), (4, 5, 6)]

    def test_database_attach(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        db_path = tmp_path / "hello.db"

        # setup a database to attach later
        with duckdb.connect(str(db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE t (id int);
                INSERT INTO t VALUES (0)
            """
            )

        assert db_path.exists()

        with duckdb.connect() as conn:
            fs = fsspec.filesystem("file", skip_instance_cache=True)
            write_errors = intercept(monkeypatch, fsspec.implementations.local.LocalFileOpener, "write")
            conn.register_filesystem(fs)
            db_path_posix = str(PurePosixPath(tmp_path.as_posix()) / "hello.db")
            conn.execute(f"ATTACH 'file:///{db_path_posix}'")

            conn.execute("INSERT INTO hello.t VALUES (1)")

            conn.execute("FROM hello.t")
            assert conn.fetchall() == [(0,), (1,)]

        # duckdb sometimes seems to swallow write errors, so we use this to ensure that
        # isn't happening
        assert not write_errors

    def test_copy_partition(self, duckdb_cursor: DuckDBPyConnection, memory: fsspec.AbstractFileSystem):
        duckdb_cursor.register_filesystem(memory)

        duckdb_cursor.execute("copy (select 1 as a, 2 as b) to 'memory://root' (partition_by (a), HEADER 0)")

        assert memory.open("/root/a=1/data_0.csv").read() == b"2\n"

    def test_copy_partition_with_columns_written(
        self, duckdb_cursor: DuckDBPyConnection, memory: fsspec.AbstractFileSystem
    ):
        duckdb_cursor.register_filesystem(memory)

        duckdb_cursor.execute(
            "copy (select 1 as a) to 'memory://root' (partition_by (a), HEADER 0, WRITE_PARTITION_COLUMNS)"
        )

        assert memory.open("/root/a=1/data_0.csv").read() == b"1\n"

    def test_read_hive_partition(self, duckdb_cursor: DuckDBPyConnection, memory: fsspec.AbstractFileSystem):
        duckdb_cursor.register_filesystem(memory)
        duckdb_cursor.execute(
            "copy (select 2 as a, 3 as b, 4 as c) to 'memory://partition' (partition_by (a), HEADER 0)"
        )

        path = "memory:///partition/*/*.csv"

        query = "SELECT * FROM read_csv_auto('" + path + "'"

        # hive partitioning
        duckdb_cursor.execute(query + ", HIVE_PARTITIONING=1" + ");")
        assert duckdb_cursor.fetchall() == [(3, 4, 2)]

        # hive partitioning: auto detection
        duckdb_cursor.execute(query + ");")
        assert duckdb_cursor.fetchall() == [(3, 4, 2)]

        # hive partitioning: cast to int
        duckdb_cursor.execute(query + ", HIVE_PARTITIONING=1" + ", HIVE_TYPES_AUTOCAST=1" + ");")
        assert duckdb_cursor.fetchall() == [(3, 4, 2)]

        # hive partitioning: no cast to int
        duckdb_cursor.execute(query + ", HIVE_PARTITIONING=1" + ", HIVE_TYPES_AUTOCAST=0" + ");")
        assert duckdb_cursor.fetchall() == [(3, 4, "2")]

    def test_read_hive_partition_with_columns_written(
        self, duckdb_cursor: DuckDBPyConnection, memory: fsspec.AbstractFileSystem
    ):
        duckdb_cursor.register_filesystem(memory)
        duckdb_cursor.execute(
            "copy (select 2 as a) to 'memory://partition' (partition_by (a), HEADER 0, WRITE_PARTITION_COLUMNS)"
        )

        path = "memory:///partition/*/*.csv"

        query = "SELECT * FROM read_csv_auto('" + path + "'"

        # hive partitioning
        duckdb_cursor.execute(query + ", HIVE_PARTITIONING=1" + ");")
        assert duckdb_cursor.fetchall() == [(2, 2)]

        # hive partitioning: auto detection
        duckdb_cursor.execute(query + ");")
        assert duckdb_cursor.fetchall() == [(2, 2)]

        # hive partitioning: cast to int
        duckdb_cursor.execute(query + ", HIVE_PARTITIONING=1" + ", HIVE_TYPES_AUTOCAST=1" + ");")
        assert duckdb_cursor.fetchall() == [(2, 2)]

        # hive partitioning: no cast to int
        duckdb_cursor.execute(query + ", HIVE_PARTITIONING=1" + ", HIVE_TYPES_AUTOCAST=0" + ");")
        assert duckdb_cursor.fetchall() == [(2, "2")]

    def test_parallel_union_by_name(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        pytest.importorskip("fsspec")

        table1 = pa.Table.from_pylist(
            [
                {"time": 1719568210134107692, "col1": 1},
            ]
        )
        table1_path = tmp_path / "table1.parquet"
        pa.parquet.write_table(table1, table1_path)

        table2 = pa.Table.from_pylist(
            [
                {"time": 1719568210134107692, "col1": 1},
            ]
        )
        table2_path = tmp_path / "table2.parquet"
        pq.write_table(table2, table2_path)

        c = duckdb.connect()
        c.register_filesystem(fsspec.implementations.local.LocalFileSystem())

        q = f"SELECT * FROM read_parquet('file://{tmp_path}/table*.parquet', union_by_name = TRUE) ORDER BY time DESC LIMIT 1"  # noqa: E501

        res = c.sql(q).fetchall()
        assert res == [(1719568210134107692, 1)]


class TestNanobindFilesystemHardening:
    """Regressions for the pre-existing filesystem safety gaps the nanobind cutover surfaced."""

    def test_read_returning_more_bytes_does_not_overflow(self, monkeypatch, memory):
        """A read(n) that returns MORE than n bytes must not overflow the read buffer (#11).

        PythonFilesystem::Read memcpy'd data.size() bytes (Python-controlled) into a buffer sized for
        nr_bytes, so a greedy read overflowed it (heap overflow, caught by ASan). The copy must be
        clamped to nr_bytes; the extra bytes are dropped and the content still parses correctly.
        """
        from fsspec.implementations.memory import MemoryFile

        # A large file so DuckDB issues full-buffer reads that the greedy read can overflow.
        big = "\n".join(f"{i};{i * 10};{i % 7}" for i in range(200000)).encode() + b"\n"
        with memory.open("big.csv", "wb") as f:
            f.write(big)

        orig_read = MemoryFile.read

        def greedy_read(self, length=-1):
            data = orig_read(self, length)
            # Only append when the read filled the request, so the returned size exceeds nr_bytes.
            if length is not None and length >= 0 and len(data) == length:
                return data + b"\x00" * 64
            return data

        monkeypatch.setattr(MemoryFile, "read", greedy_read)

        con = duckdb.connect()
        con.register_filesystem(memory)
        # Must not overflow (ASan) and must count correctly despite the injected trailing bytes.
        query = "SELECT count(*), sum(column0) FROM read_csv('memory://big.csv', sep=';', header=false)"
        res = con.sql(query).fetchone()
        assert res == (200000, sum(range(200000)))

    def test_filesystem_object_destructor_swallows_delete_error(self, monkeypatch):
        """A raising fsspec delete in ~FileSystemObject must not abort the process (#12).

        The destructor called obj.delete(file) with no try/catch, so a KeyError (missing entry) escaped
        the implicitly-noexcept destructor and aborted the process. Reading a file-like object registers
        such a cleanup dependency; its destruction must survive a raising delete.
        """
        import gc
        import io

        from duckdb.filesystem import ModifiedMemoryFileSystem

        def raising_delete(self, *args, **kwargs):
            msg = "simulated missing entry"
            raise KeyError(msg)

        monkeypatch.setattr(ModifiedMemoryFileSystem, "delete", raising_delete, raising=False)

        con = duckdb.connect()
        rel = con.read_csv(io.BytesIO(b"a,b\n1,2\n3,4\n"))
        assert rel.fetchall() == [(1, 2), (3, 4)]
        del rel
        del con
        gc.collect()  # runs ~FileSystemObject -> delete() raises -> must not std::terminate
        # Reaching this line means the process survived the throwing destructor.
        assert True

    def test_modified_memory_filesystem_importable(self):
        """#13 note: ModifiedMemoryFileSystem::check_ must not throw from noexcept contexts.

        check_ was missing the try/catch its sibling AbstractFileSystem::check_ has; nanobind can invoke
        it from noexcept caster/isinstance contexts where a throw would std::terminate. The throwing path
        (a failed duckdb.filesystem import or IsInstance == -1) cannot be induced from Python without
        breaking the module itself, so this only asserts the module stays importable; the fix is verified
        by compile + sibling parity and re-checked under ASan by the reviewer.
        """
        from duckdb.filesystem import ModifiedMemoryFileSystem

        assert ModifiedMemoryFileSystem is not None

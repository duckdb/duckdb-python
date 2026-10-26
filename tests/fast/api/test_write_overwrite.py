import re
from pathlib import Path

import pytest

import duckdb

FIRST = "SELECT 1 AS v, p FROM (VALUES ('a'), ('c')) t(p)"
SECOND = "SELECT 2 AS v, p FROM (VALUES ('a'), ('b')) t(p)"
SECOND_ROWS = [(2, "a"), (2, "b")]

REFUSED_SINGLE_FILE = 'Cannot write to "{path}" - it exists! Pass overwrite=True to overwrite the file'
ENGINE_REFUSED_FILE = (
    'Cannot write to "{path}" - it exists and is a file, not a directory! Enable OVERWRITE option to overwrite the file'
)
ENGINE_REFUSED_DIRECTORY = 'Directory "{path}" is not empty! Enable OVERWRITE option to overwrite files'
APPEND_WITH_OVERWRITE = "Can only set one of OVERWRITE_OR_IGNORE, OVERWRITE or APPEND"


def rel_write_csv(con, path, **kwargs):
    con.sql(SECOND).write_csv(path, **kwargs)


def rel_to_csv(con, path, **kwargs):
    con.sql(SECOND).to_csv(path, **kwargs)


def rel_write_parquet(con, path, **kwargs):
    con.sql(SECOND).write_parquet(path, **kwargs)


def rel_to_parquet(con, path, **kwargs):
    con.sql(SECOND).to_parquet(path, **kwargs)


def module_write_csv(con, path, **kwargs):
    duckdb.write_csv(con.sql(SECOND).df(), path, connection=con, **kwargs)


WRITERS = {
    "rel.write_csv": (rel_write_csv, "csv"),
    "rel.to_csv": (rel_to_csv, "csv"),
    "rel.write_parquet": (rel_write_parquet, "parquet"),
    "rel.to_parquet": (rel_to_parquet, "parquet"),
    "duckdb.write_csv": (module_write_csv, "csv"),
}
PARQUET_WRITERS = ["rel.write_parquet", "rel.to_parquet"]

SINGLE_FILE = {
    "single": {},
    "single_no_tmp_file": {"use_tmp_file": False},
    "single_per_thread_output_false": {"per_thread_output": False},
}
DIRECTORY = {
    "partition_by": {"partition_by": ["p"]},
    "per_thread_output": {"per_thread_output": True},
}
ROTATING = {"file_size_bytes": {"file_size_bytes": 1}}


def cases(modes, *extra, writers=WRITERS):
    return [
        pytest.param(*WRITERS[writer], options, *extra, id=f"{writer}-{mode}")
        for writer in writers
        for mode, options in modes.items()
    ]


def directory_cases(*extra):
    return cases(DIRECTORY, *extra) + cases(ROTATING, *extra, writers=PARQUET_WRITERS)


def files_under(root):
    return {str(f.relative_to(root)): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}


def read_rows(con, path, fmt):
    source = f"{path}/**/*.{fmt}" if Path(path).is_dir() else path
    return con.sql(f"SELECT v, p FROM read_{fmt}('{source}', hive_partitioning = true) ORDER BY ALL").fetchall()


def write_first_file(con, path, fmt):
    con.execute(f"COPY ({FIRST}) TO '{path}' (FORMAT {fmt})")


def write_first_partitioned(con, path, fmt):
    con.execute(f"COPY ({FIRST}) TO '{path}' (FORMAT {fmt}, PARTITION_BY (p))")


def exactly(message, path):
    return re.escape(message.format(path=path)) + "$"


def kwargs_with(options, overwrite):
    kwargs = dict(options)
    if overwrite is not None:
        kwargs["overwrite"] = overwrite
    return kwargs


class TestWriteOverwrite:
    @pytest.mark.parametrize(
        ("writer", "fmt", "options", "message"),
        cases(SINGLE_FILE, REFUSED_SINGLE_FILE) + directory_cases(ENGINE_REFUSED_FILE),
    )
    def test_false_refuses_an_existing_file(self, duckdb_cursor, tmp_path, writer, fmt, options, message):
        target = tmp_path / f"out.{fmt}"
        write_first_file(duckdb_cursor, target, fmt)
        before = files_under(tmp_path)

        with pytest.raises(duckdb.IOException, match=exactly(message, target)):
            writer(duckdb_cursor, str(target), overwrite=False, **options)

        assert files_under(tmp_path) == before

    @pytest.mark.parametrize(
        "overwrite", [pytest.param(None, id="overwrite_none"), pytest.param(False, id="overwrite_false")]
    )
    @pytest.mark.parametrize(("writer", "fmt", "options"), directory_cases())
    def test_none_and_false_refuse_a_nonempty_directory(self, duckdb_cursor, tmp_path, writer, fmt, options, overwrite):
        target = tmp_path / "out"
        write_first_partitioned(duckdb_cursor, target, fmt)
        before = files_under(tmp_path)

        with pytest.raises(duckdb.IOException, match=exactly(ENGINE_REFUSED_DIRECTORY, target)):
            writer(duckdb_cursor, str(target), **kwargs_with(options, overwrite))

        assert files_under(tmp_path) == before

    @pytest.mark.parametrize(("writer", "fmt"), [WRITERS[writer] for writer in PARQUET_WRITERS], ids=PARQUET_WRITERS)
    def test_false_with_append_keeps_the_engine_refusal(self, duckdb_cursor, tmp_path, writer, fmt):
        target = tmp_path / f"out.{fmt}"
        write_first_file(duckdb_cursor, target, fmt)
        before = files_under(tmp_path)

        with pytest.raises(duckdb.BinderException, match=re.escape(APPEND_WITH_OVERWRITE) + "$"):
            writer(duckdb_cursor, str(target), overwrite=False, append=True)

        assert files_under(tmp_path) == before

    @pytest.mark.parametrize(
        "overwrite",
        [
            pytest.param(None, id="overwrite_none"),
            pytest.param(False, id="overwrite_false"),
            pytest.param(True, id="overwrite_true"),
        ],
    )
    @pytest.mark.parametrize(("writer", "fmt", "options"), cases(SINGLE_FILE) + directory_cases())
    def test_a_missing_target_is_written(self, duckdb_cursor, tmp_path, writer, fmt, options, overwrite):
        target = tmp_path / f"out.{fmt}"

        writer(duckdb_cursor, str(target), **kwargs_with(options, overwrite))

        assert read_rows(duckdb_cursor, target, fmt) == SECOND_ROWS

    @pytest.mark.parametrize(
        "overwrite", [pytest.param(None, id="overwrite_none"), pytest.param(True, id="overwrite_true")]
    )
    @pytest.mark.parametrize(("writer", "fmt", "options"), cases(SINGLE_FILE))
    def test_none_and_true_replace_an_existing_file(self, duckdb_cursor, tmp_path, writer, fmt, options, overwrite):
        target = tmp_path / f"out.{fmt}"
        write_first_file(duckdb_cursor, target, fmt)

        writer(duckdb_cursor, str(target), **kwargs_with(options, overwrite))

        assert read_rows(duckdb_cursor, target, fmt) == SECOND_ROWS
        assert sorted(files_under(tmp_path)) == [f"out.{fmt}"]

    @pytest.mark.parametrize(("writer", "fmt", "options"), directory_cases())
    def test_false_writes_into_an_existing_empty_directory(self, duckdb_cursor, tmp_path, writer, fmt, options):
        target = tmp_path / "out"
        target.mkdir()

        writer(duckdb_cursor, str(target), overwrite=False, **options)

        assert read_rows(duckdb_cursor, target, fmt) == SECOND_ROWS


@pytest.fixture
def memory_fs(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec")
    pytest.importorskip("fsspec.implementations.memory")
    fs = fsspec.filesystem("memory", skip_instance_cache=True)
    fs.store = {}
    fs.pseudo_dirs = [""]
    duckdb_cursor.register_filesystem(fs)
    return fs


MEMORY_WRITERS = [
    pytest.param(rel_write_csv, "csv", id="rel.write_csv"),
    pytest.param(rel_write_parquet, "parquet", id="rel.write_parquet"),
]


class TestWriteOverwriteRegisteredFilesystem:
    @pytest.mark.parametrize(("writer", "fmt"), MEMORY_WRITERS)
    def test_false_refuses_an_existing_file(self, duckdb_cursor, memory_fs, writer, fmt):
        target = f"memory://out.{fmt}"
        write_first_file(duckdb_cursor, target, fmt)
        before = memory_fs.cat(f"out.{fmt}")

        with pytest.raises(duckdb.IOException, match=exactly(REFUSED_SINGLE_FILE, target)):
            writer(duckdb_cursor, target, overwrite=False)

        assert memory_fs.find("/") == [f"/out.{fmt}"]
        assert memory_fs.cat(f"out.{fmt}") == before

    @pytest.mark.parametrize(
        "overwrite",
        [
            pytest.param(None, id="overwrite_none"),
            pytest.param(False, id="overwrite_false"),
            pytest.param(True, id="overwrite_true"),
        ],
    )
    @pytest.mark.parametrize(("writer", "fmt"), MEMORY_WRITERS)
    def test_a_missing_target_is_written(self, duckdb_cursor, memory_fs, writer, fmt, overwrite):
        target = f"memory://out.{fmt}"

        writer(duckdb_cursor, target, **kwargs_with({}, overwrite))

        assert read_rows(duckdb_cursor, target, fmt) == SECOND_ROWS

    @pytest.mark.parametrize(
        "overwrite", [pytest.param(None, id="overwrite_none"), pytest.param(True, id="overwrite_true")]
    )
    @pytest.mark.parametrize(("writer", "fmt"), MEMORY_WRITERS)
    def test_none_and_true_replace_an_existing_file(self, duckdb_cursor, memory_fs, writer, fmt, overwrite):
        target = f"memory://out.{fmt}"
        write_first_file(duckdb_cursor, target, fmt)

        writer(duckdb_cursor, target, **kwargs_with({}, overwrite))

        assert read_rows(duckdb_cursor, target, fmt) == SECOND_ROWS

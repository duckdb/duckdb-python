"""The extension directory that duckdb-core-ext wheels install into joins the engine's extension search path.

Each test builds a stand-in for site-packages in a temporary directory: the duckdb distribution's metadata, first on
sys.path and taken as the place the imported package lives, and next to it the .duckdb_extensions directory the
extension wheels create. Nothing here installs or downloads an extension.
"""

import errno
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import _duckdb
import pytest

import duckdb
from duckdb import _extension_directories

DEFAULT_DIRECTORY = "~/.duckdb/extensions"
SETTING = "SELECT current_setting('extension_directories')"
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file names and descriptors")


def directories(connection):
    return connection.execute(SETTING).fetchone()[0]


def make_site(site):
    dist_info = site / "duckdb-0.0.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: duckdb\nVersion: 0.0.0\n")
    (site / ".duckdb_extensions").mkdir()


def place_package_in(site, monkeypatch):
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(_extension_directories, "_PACKAGE_PARENT", str(site))


def assert_engine_lists_a_file_in(connection, directory):
    version, source_id = connection.execute("SELECT library_version, source_id FROM pragma_version()").fetchone()
    platform = connection.execute("PRAGMA platform").fetchone()[0]
    installed = Path(directory, source_id if "-dev" in version else version, platform)
    installed.mkdir(parents=True)
    (installed / "fakeext.duckdb_extension").write_bytes(b"\0" * 1024)
    rows = connection.execute(
        "SELECT installed, install_path FROM duckdb_extensions() WHERE extension_name = 'fakeext'"
    ).fetchall()
    assert [(flag, Path(path)) for flag, path in rows] == [(True, installed / "fakeext.duckdb_extension")]


def fail_creates_in(directory, monkeypatch, error):
    """Make every file create in the directory raise the error, and record each attempt."""
    attempts = []
    real_open = os.open

    def open_failing(path, flags, *args, **kwargs):
        if Path(path).parent == Path(directory):
            attempts.append(path)
            raise error
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_failing)
    return attempts


def deny_creates_in(directory, monkeypatch):
    return fail_creates_in(directory, monkeypatch, PermissionError(errno.EACCES, "Access is denied"))


def watch_probe_descriptors(directory, monkeypatch, on_close):
    """Call on_close with the probe's path and descriptor whenever a file created in the directory is closed."""
    probes = {}
    real_open, real_close = os.open, os.close

    def open_watching(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path).parent == Path(directory):
            probes[descriptor] = Path(path)
        return descriptor

    def close_watching(descriptor):
        if descriptor in probes:
            on_close(probes.pop(descriptor), descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "open", open_watching)
    monkeypatch.setattr(os, "close", close_watching)


@pytest.fixture(autouse=True)
def located_afresh():
    _extension_directories._pip_directory.cache_clear()
    yield
    _extension_directories._pip_directory.cache_clear()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """A home without extensions, so the default directory in the list holds nothing of the machine's."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture
def site(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    make_site(site)
    place_package_in(site, monkeypatch)
    return site


@pytest.fixture
def pip_directory(site):
    return str(site / ".duckdb_extensions")


def test_writable_directory_comes_first(pip_directory):
    assert directories(duckdb.connect()) == [pip_directory, DEFAULT_DIRECTORY]


def test_read_only_directory_comes_last(pip_directory, monkeypatch):
    deny_creates_in(pip_directory, monkeypatch)
    assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, pip_directory]


@posix_only
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes into a read-only directory")
def test_directory_without_write_permission_comes_last(pip_directory):
    Path(pip_directory).chmod(0o555)
    try:
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, pip_directory]
    finally:
        Path(pip_directory).chmod(0o755)


def test_writability_is_probed_with_one_create(pip_directory, monkeypatch):
    attempts = deny_creates_in(pip_directory, monkeypatch)
    duckdb.connect()
    assert len(attempts) == 1


def test_probe_leaves_no_file_behind(pip_directory):
    duckdb.connect()
    assert list(Path(pip_directory).iterdir()) == []


@posix_only
def test_probe_is_removed_before_it_is_closed(pip_directory, monkeypatch):
    present_at_close = []
    watch_probe_descriptors(pip_directory, monkeypatch, lambda path, _: present_at_close.append(path.exists()))
    duckdb.connect()
    assert present_at_close == [False]


def fail_close(path, descriptor):
    os.close(descriptor)
    raise OSError(errno.EIO, "Input/output error", str(path))


def fail_unlink(directory, monkeypatch):
    real_unlink = os.unlink

    def unlink_failing(path, *args, **kwargs):
        if Path(path).parent == Path(directory):
            raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink_failing)


@pytest.mark.parametrize(
    "failing", ["close", pytest.param("unlink", marks=posix_only)], ids=["close-fails", "unlink-fails"]
)
def test_failure_after_the_create_reads_as_read_only(pip_directory, monkeypatch, failing):
    if failing == "close":
        watch_probe_descriptors(pip_directory, monkeypatch, fail_close)
    else:
        fail_unlink(pip_directory, monkeypatch)
    assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, pip_directory]


@pytest.mark.parametrize("opener", [duckdb.connect, lambda: duckdb.sql(SETTING)], ids=["connect", "default"])
def test_error_while_choosing_the_directories_reaches_the_caller(pip_directory, monkeypatch, opener):
    duckdb.default_connection().close()
    failure = RuntimeError("the probe failed")
    fail_creates_in(pip_directory, monkeypatch, failure)
    with pytest.raises(RuntimeError) as raised:
        opener()
    assert raised.value is failure


@pytest.mark.parametrize("result", ["/a", ["/a", 1]], ids=["str", "list-with-an-int"])
def test_hook_result_other_than_a_list_of_str_is_a_type_error(result):
    _duckdb._set_extension_search_path(lambda configured, deprecated_directory: result)
    try:
        with pytest.raises(TypeError, match=r"extension search path hook .* must return a list of str or None"):
            duckdb.connect()
    finally:
        _duckdb._set_extension_search_path(_extension_directories.search_path)


EXIT_SCRIPT = """
    import atexit, os, sys
    import duckdb
    from duckdb import _extension_directories, sqltypes

    SETTING = "SELECT current_setting('extension_directories')"
    sys.path.insert(0, sys.argv[1])
    _extension_directories._PACKAGE_PARENT = sys.argv[1]

    # Module globals may already be gone when the finalizer runs, so everything the report needs is bound here.
    def report(label, write=os.write, connect=duckdb.connect, setting=SETTING):
        write(1, f"{label} {connect().execute(setting).fetchone()[0]}\\n".encode())

    class Identity:
        def __call__(self, x):
            return x

        # Runs once the default connection is released while the interpreter finalizes.
        def __del__(self, report=report):
            report("finalizer")

    atexit.register(report, "atexit")
    duckdb.default_connection().create_function("identity", Identity(), [sqltypes.BIGINT], sqltypes.BIGINT)
"""


def test_connect_at_exit(pip_directory, site):
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(EXIT_SCRIPT), str(site)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    reports = [line for line in proc.stdout.splitlines() if line.startswith(("atexit ", "finalizer "))]
    assert reports == [f"atexit {[pip_directory, DEFAULT_DIRECTORY]}", "finalizer []"]


def test_absent_directory_changes_nothing(site):
    (site / ".duckdb_extensions").rmdir()
    assert directories(duckdb.connect()) == []


def test_directory_created_after_a_connect_is_found_by_the_next(site):
    (site / ".duckdb_extensions").rmdir()
    assert directories(duckdb.connect()) == []
    (site / ".duckdb_extensions").mkdir()
    assert directories(duckdb.connect()) == [str(site / ".duckdb_extensions"), DEFAULT_DIRECTORY]


def test_file_in_place_of_the_directory_changes_nothing(site):
    (site / ".duckdb_extensions").rmdir()
    (site / ".duckdb_extensions").write_text("")
    assert directories(duckdb.connect()) == []


def test_without_duckdb_metadata_on_sys_path_nothing_changes(monkeypatch):
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if not any(Path(entry).glob("duckdb-*.dist-info"))])
    assert directories(duckdb.connect()) == []


INSTALLED_PACKAGE_PARENT = Path(duckdb.__file__).absolute().parent.parent


class TestLocation:
    @pytest.mark.skipif(
        not any(INSTALLED_PACKAGE_PARENT.glob("duckdb-*.dist-info")),
        reason="an editable install keeps its dist-info away from its code",
    )
    def test_installed_package_is_located_beside_its_own_distribution(self):
        site = _extension_directories._site_directory()
        assert site is not None
        assert os.path.normcase(os.path.normpath(site)) == os.path.normcase(os.path.normpath(INSTALLED_PACKAGE_PARENT))

    def test_distribution_beside_the_package_wins_over_an_earlier_one(self, tmp_path, monkeypatch):
        own, earlier = tmp_path / "own", tmp_path / "earlier"
        make_site(own)
        make_site(earlier)
        place_package_in(own, monkeypatch)
        monkeypatch.syspath_prepend(str(earlier))
        assert directories(duckdb.connect()) == [str(own / ".duckdb_extensions"), DEFAULT_DIRECTORY]

    def test_without_a_distribution_beside_the_package_the_first_counts(self, tmp_path, monkeypatch):
        first, second = tmp_path / "first", tmp_path / "second"
        make_site(first)
        make_site(second)
        monkeypatch.syspath_prepend(str(second))
        monkeypatch.syspath_prepend(str(first))
        monkeypatch.setattr(_extension_directories, "_PACKAGE_PARENT", str(tmp_path / "checkout"))
        assert directories(duckdb.connect()) == [str(first / ".duckdb_extensions"), DEFAULT_DIRECTORY]

    def test_location_is_found_once_per_process(self, tmp_path, monkeypatch):
        first, later = tmp_path / "first", tmp_path / "later"
        make_site(first)
        make_site(later)
        monkeypatch.syspath_prepend(str(first))
        monkeypatch.setattr(_extension_directories, "_PACKAGE_PARENT", str(tmp_path / "checkout"))
        duckdb.connect()
        monkeypatch.syspath_prepend(str(later))
        assert directories(duckdb.connect()) == [str(first / ".duckdb_extensions"), DEFAULT_DIRECTORY]

    def test_relative_sys_path_entry_gives_an_absolute_directory(self, tmp_path, monkeypatch):
        make_site(tmp_path / "site-packages")
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend("site-packages")
        monkeypatch.setattr(_extension_directories, "_PACKAGE_PARENT", str(tmp_path / "site-packages"))
        expected = str(Path.cwd() / "site-packages" / ".duckdb_extensions")
        assert directories(duckdb.connect()) == [expected, DEFAULT_DIRECTORY]


class TestPaths:
    @pytest.mark.parametrize(
        "name",
        ["site packages, it's [café]", pytest.param('double "quoted"', marks=posix_only)],
    )
    def test_path_reaches_the_engine_as_given(self, name, tmp_path, monkeypatch):
        site = tmp_path / name
        make_site(site)
        place_package_in(site, monkeypatch)
        connection = duckdb.connect()
        assert directories(connection) == [str(site / ".duckdb_extensions"), DEFAULT_DIRECTORY]
        assert_engine_lists_a_file_in(connection, site / ".duckdb_extensions")

    @posix_only
    def test_path_with_a_backslash_is_left_out(self, tmp_path, monkeypatch):
        site = tmp_path / "back\\slash"
        make_site(site)
        place_package_in(site, monkeypatch)
        assert directories(duckdb.connect()) == []

    @pytest.mark.skipif(sys.platform != "linux", reason="only Linux file systems take a name that is not UTF-8")
    def test_path_that_is_not_utf8_is_left_out(self, tmp_path, monkeypatch):
        site = tmp_path / os.fsdecode(b"site-\xff")
        make_site(site)
        place_package_in(site, monkeypatch)
        assert directories(duckdb.connect()) == []


class TestPreviousList:
    def test_user_list_follows_a_writable_directory(self, pip_directory):
        connection = duckdb.connect(config={"extension_directories": ["/a", "/b"]})
        assert directories(connection) == [pip_directory, "/a", "/b"]

    def test_user_list_precedes_a_read_only_directory(self, pip_directory, monkeypatch):
        deny_creates_in(pip_directory, monkeypatch)
        connection = duckdb.connect(config={"extension_directories": ["/a", "/b"]})
        assert directories(connection) == ["/a", "/b", pip_directory]

    @pytest.mark.parametrize(
        "config",
        [{"extension_directories": "[/a, /b]"}, {"EXTENSION_DIRECTORIES": ["/a", "/b"]}],
        ids=["list-literal-text", "upper-case-key"],
    )
    def test_user_list_is_read_as_the_engine_parsed_it(self, pip_directory, config):
        assert directories(duckdb.connect(config=config)) == [pip_directory, "/a", "/b"]

    def test_explicit_empty_list_gets_the_default(self, pip_directory):
        connection = duckdb.connect(config={"extension_directories": []})
        assert directories(connection) == [pip_directory, DEFAULT_DIRECTORY]

    def test_deprecated_directory_setting_gets_no_default(self, pip_directory):
        connection = duckdb.connect(config={"extension_directory": "/singular"})
        assert connection.execute("SELECT current_setting('extension_directory')").fetchone()[0] == "/singular"
        assert directories(connection) == [pip_directory]

    def test_deprecated_directory_naming_the_directory_is_kept(self, pip_directory):
        connection = duckdb.connect(config={"extension_directory": pip_directory})
        assert connection.execute("SELECT current_setting('extension_directory')").fetchone()[0] == pip_directory
        assert directories(connection) == [pip_directory]

    @pytest.mark.parametrize("suffix", ["", os.sep], ids=["as-given", "trailing-separator"])
    def test_user_list_naming_the_directory_holds_it_once(self, pip_directory, suffix):
        connection = duckdb.connect(config={"extension_directories": [DEFAULT_DIRECTORY, pip_directory + suffix]})
        assert directories(connection) == [pip_directory, DEFAULT_DIRECTORY]


class TestEveryConnection:
    @pytest.mark.parametrize("read_only", [False, True])
    def test_database_file(self, pip_directory, tmp_path, read_only):
        path = str(tmp_path / "file.duckdb")
        duckdb.connect(path).close()
        connection = duckdb.connect(path, read_only=read_only)
        assert directories(connection) == [pip_directory, DEFAULT_DIRECTORY]

    def test_second_connect_to_an_open_file_keeps_its_list(self, pip_directory, tmp_path, monkeypatch):
        path = str(tmp_path / "file.duckdb")
        first = duckdb.connect(path)
        attempts = deny_creates_in(pip_directory, monkeypatch)
        second = duckdb.connect(path)
        assert len(attempts) == 1
        assert directories(second) == directories(first) == [pip_directory, DEFAULT_DIRECTORY]

    def test_cursor(self, pip_directory):
        cursor = duckdb.connect().cursor()
        assert directories(cursor) == [pip_directory, DEFAULT_DIRECTORY]

    def test_default_connection(self, pip_directory):
        duckdb.default_connection().close()
        try:
            assert duckdb.sql(SETTING).fetchall() == [([pip_directory, DEFAULT_DIRECTORY],)]
        finally:
            duckdb.default_connection().close()

    def test_engine_lists_an_extension_file_in_the_directory(self, pip_directory):
        assert_engine_lists_a_file_in(duckdb.connect(), pip_directory)

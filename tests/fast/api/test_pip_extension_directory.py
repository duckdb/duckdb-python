"""The extension directory that duckdb-core-ext wheels install into joins the engine's extension search path.

Each test builds a stand-in for site-packages in a temporary directory: the duckdb distribution's metadata, first on
sys.path and taken as the place the imported package lives, and next to it the .duckdb_extensions directory the
extension wheels create. The interpreter's own site directories are kept out, and a test that needs more of them
declares them through the site module. Nothing here downloads an extension; the autoload, LOAD, INSTALL and UPDATE
EXTENSIONS tests use stand-in files and an empty local repository.
"""

import os
import re
import site as site_module
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
not_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses directory permissions"
)


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


def make_extension_site(path):
    """A site directory holding .duckdb_extensions and no duckdb metadata, as one that got only extension wheels."""
    (path / ".duckdb_extensions").mkdir(parents=True)
    return str(path / ".duckdb_extensions")


def on_sys_path(monkeypatch, *paths):
    for path in reversed(paths):
        monkeypatch.syspath_prepend(str(path))


def last_on_sys_path(monkeypatch, path):
    monkeypatch.setattr(sys, "path", [*sys.path, str(path)])


def site_packages_are(monkeypatch, *paths):
    monkeypatch.setattr(site_module, "getsitepackages", lambda prefixes=None: [str(path) for path in paths])


def user_site_is(monkeypatch, path, enabled=True):
    monkeypatch.setattr(site_module, "getusersitepackages", lambda: str(path))
    monkeypatch.setattr(site_module, "ENABLE_USER_SITE", enabled)


def installed_path(connection, directory, name):
    """Where the engine keeps an installed extension under a directory: <directory>/<version>/<platform>/<name>."""
    version, source_id = connection.execute("SELECT library_version, source_id FROM pragma_version()").fetchone()
    platform = connection.execute("PRAGMA platform").fetchone()[0]
    return Path(directory, source_id if "-dev" in version else version, platform, f"{name}.duckdb_extension")


def place_stand_in(connection, directory, name, size=1024):
    """Write a file where an extension wheel puts its binary, and return its path.

    A file shorter than the engine's metadata footer makes LOAD name the file it opened in its error.
    """
    stand_in = installed_path(connection, directory, name)
    stand_in.parent.mkdir(parents=True, exist_ok=True)
    stand_in.write_bytes(b"\0" * size)
    return stand_in


def count_lookups(monkeypatch):
    """Record each time the hook looks the pip directories up, which it does for every database it is asked about."""
    lookups = []
    real = _extension_directories._pip_directories

    def counting():
        lookups.append(1)
        return real()

    counting.cache_clear = real.cache_clear
    monkeypatch.setattr(_extension_directories, "_pip_directories", counting)
    return lookups


def skip_unless_loadable(connection, name):
    linked = connection.execute(
        "SELECT install_mode = 'STATICALLY_LINKED' FROM duckdb_extensions() WHERE extension_name = ?", [name]
    ).fetchone()
    if linked and linked[0]:
        pytest.skip(f"{name} is linked into this build")


def assert_engine_lists_a_file_in(connection, directory):
    stand_in = place_stand_in(connection, directory, "fakeext")
    rows = connection.execute(
        "SELECT installed, install_path FROM duckdb_extensions() WHERE extension_name = 'fakeext'"
    ).fetchall()
    assert [(flag, Path(path)) for flag, path in rows] == [(True, stand_in)]


@pytest.fixture(autouse=True)
def located_afresh():
    _extension_directories._pip_directories.cache_clear()
    yield
    _extension_directories._pip_directories.cache_clear()


@pytest.fixture(autouse=True)
def no_site_directories(monkeypatch):
    """The interpreter's own site directories stay out, so a .duckdb_extensions of the machine's cannot reach a test."""
    site_packages_are(monkeypatch)
    monkeypatch.setattr(site_module, "ENABLE_USER_SITE", False)


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


def test_directory_comes_after_the_default(pip_directory):
    assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, pip_directory]


@pytest.mark.parametrize("opener", [duckdb.connect, lambda: duckdb.sql(SETTING)], ids=["connect", "default"])
def test_error_while_choosing_the_directories_reaches_the_caller(pip_directory, monkeypatch, opener):
    duckdb.default_connection().close()
    _extension_directories._pip_directories.cache_clear()
    failure = RuntimeError("the lookup failed")

    def failing(prefixes=None):
        raise failure

    monkeypatch.setattr(site_module, "getsitepackages", failing)
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
    import site
    site.getsitepackages = lambda prefixes=None: []
    site.ENABLE_USER_SITE = False

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
    assert reports == [f"atexit {[DEFAULT_DIRECTORY, pip_directory]}", "finalizer []"]


def test_absent_directory_changes_nothing(site):
    (site / ".duckdb_extensions").rmdir()
    assert directories(duckdb.connect()) == []


def test_directory_created_after_a_connect_is_found_by_the_next(site):
    (site / ".duckdb_extensions").rmdir()
    assert directories(duckdb.connect()) == []
    (site / ".duckdb_extensions").mkdir()
    assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, str(site / ".duckdb_extensions")]


def test_file_in_place_of_the_directory_changes_nothing(site):
    (site / ".duckdb_extensions").rmdir()
    (site / ".duckdb_extensions").write_text("")
    assert directories(duckdb.connect()) == []


@posix_only
@not_root
def test_symlink_into_an_unsearchable_directory_changes_nothing(site, tmp_path):
    (site / ".duckdb_extensions").rmdir()
    locked = tmp_path / "locked"
    (locked / "inner").mkdir(parents=True)
    (site / ".duckdb_extensions").symlink_to(locked / "inner")
    locked.chmod(0)
    try:
        assert directories(duckdb.connect()) == []
    finally:
        locked.chmod(0o755)


def test_without_duckdb_metadata_on_sys_path_nothing_changes(site, monkeypatch):
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
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, str(own / ".duckdb_extensions")]

    def test_without_a_distribution_beside_the_package_the_first_counts(self, tmp_path, monkeypatch):
        first, second = tmp_path / "first", tmp_path / "second"
        make_site(first)
        make_site(second)
        monkeypatch.syspath_prepend(str(second))
        monkeypatch.syspath_prepend(str(first))
        monkeypatch.setattr(_extension_directories, "_PACKAGE_PARENT", str(tmp_path / "checkout"))
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, str(first / ".duckdb_extensions")]

    def test_location_is_found_once_per_process(self, tmp_path, monkeypatch):
        first, later = tmp_path / "first", tmp_path / "later"
        make_site(first)
        make_site(later)
        monkeypatch.syspath_prepend(str(first))
        monkeypatch.setattr(_extension_directories, "_PACKAGE_PARENT", str(tmp_path / "checkout"))
        duckdb.connect()
        monkeypatch.syspath_prepend(str(later))
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, str(first / ".duckdb_extensions")]

    def test_relative_sys_path_entry_gives_an_absolute_directory(self, tmp_path, monkeypatch):
        make_site(tmp_path / "site-packages")
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend("site-packages")
        monkeypatch.setattr(_extension_directories, "_PACKAGE_PARENT", str(tmp_path / "site-packages"))
        expected = str(Path.cwd() / "site-packages" / ".duckdb_extensions")
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, expected]


class TestSiteDirectories:
    """Extension wheels installed into another site directory of the interpreter than the one holding duckdb."""

    def test_user_site_holding_the_directory_while_duckdb_sits_elsewhere(self, site, tmp_path, monkeypatch):
        (site / ".duckdb_extensions").rmdir()
        user = tmp_path / "user-site"
        pip_directory = make_extension_site(user)
        user_site_is(monkeypatch, user)
        on_sys_path(monkeypatch, user)
        connection = duckdb.connect()
        assert directories(connection) == [DEFAULT_DIRECTORY, pip_directory]
        assert_engine_lists_a_file_in(connection, pip_directory)

    def test_disabled_user_site_is_not_searched(self, site, tmp_path, monkeypatch):
        (site / ".duckdb_extensions").rmdir()
        user = tmp_path / "user-site"
        make_extension_site(user)
        user_site_is(monkeypatch, user, enabled=False)
        on_sys_path(monkeypatch, user)
        assert directories(duckdb.connect()) == []

    def test_site_directory_off_sys_path_is_not_searched(self, site, tmp_path, monkeypatch):
        (site / ".duckdb_extensions").rmdir()
        other = tmp_path / "other-site"
        make_extension_site(other)
        site_packages_are(monkeypatch, other)
        assert directories(duckdb.connect()) == []

    def test_directories_follow_sys_path_order(self, site, tmp_path, monkeypatch):
        (site / ".duckdb_extensions").rmdir()
        first, second = tmp_path / "first", tmp_path / "second"
        a, b = make_extension_site(first), make_extension_site(second)
        site_packages_are(monkeypatch, second, first)
        on_sys_path(monkeypatch, first, second)
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, a, b]
        connection = duckdb.connect(config={"extension_directories": ["/a"]})
        assert directories(connection) == ["/a", a, b]

    def test_directory_named_several_times_is_one(self, site, pip_directory, monkeypatch):
        site_packages_are(monkeypatch, site, str(site) + os.sep)
        user_site_is(monkeypatch, site)
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, pip_directory]

    @posix_only
    def test_directory_reached_through_a_symlink_is_one(self, site, tmp_path, monkeypatch):
        alias = tmp_path / "lib64-site-packages"
        alias.symlink_to(site, target_is_directory=True)
        site_packages_are(monkeypatch, alias, site)
        on_sys_path(monkeypatch, alias)
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, str(alias / ".duckdb_extensions")]

    def test_site_directory_spelled_through_a_parent_step_is_found(self, site, tmp_path, monkeypatch):
        (site / ".duckdb_extensions").rmdir()
        other = tmp_path / "other-site"
        pip_directory = make_extension_site(other)
        site_packages_are(monkeypatch, tmp_path / "elsewhere" / ".." / "other-site")
        on_sys_path(monkeypatch, other)
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, pip_directory]

    def test_distribution_directory_keeps_its_sys_path_position(self, site, pip_directory, tmp_path, monkeypatch):
        later = tmp_path / "later-site"
        other = make_extension_site(later)
        site_packages_are(monkeypatch, later)
        last_on_sys_path(monkeypatch, later)
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, pip_directory, other]

    def test_distribution_reported_from_outside_sys_path_counts_last(self, site, pip_directory, tmp_path, monkeypatch):
        import importlib.metadata

        elsewhere = tmp_path / "elsewhere"
        make_site(elsewhere)

        class Finder:
            @staticmethod
            def find_spec(name, path=None, target=None) -> None:
                return None

            @staticmethod
            def find_distributions(context=None) -> list[importlib.metadata.Distribution]:
                return [importlib.metadata.PathDistribution(elsewhere / "duckdb-0.0.0.dist-info")]

        monkeypatch.setattr(sys, "meta_path", [Finder, *sys.meta_path])
        monkeypatch.setattr(_extension_directories, "_PACKAGE_PARENT", str(elsewhere))
        site_packages_are(monkeypatch, site)
        assert directories(duckdb.connect()) == [
            DEFAULT_DIRECTORY,
            pip_directory,
            str(elsewhere / ".duckdb_extensions"),
        ]

    @pytest.mark.parametrize("how", ["missing", "raising"])
    @pytest.mark.parametrize("function", ["getsitepackages", "getusersitepackages"])
    def test_site_module_that_cannot_name_its_directories_adds_none(self, pip_directory, monkeypatch, function, how):
        monkeypatch.setattr(site_module, "ENABLE_USER_SITE", True)
        failure = OSError("site lookup failed")
        if how == "missing":
            monkeypatch.delattr(site_module, function)
        else:

            def failing(*args):
                raise failure

            monkeypatch.setattr(site_module, function, failing)
        assert directories(duckdb.connect()) == [DEFAULT_DIRECTORY, pip_directory]

    def test_user_list_naming_the_directories_is_kept_as_given(self, site, pip_directory, tmp_path, monkeypatch):
        user = tmp_path / "user-site"
        other = make_extension_site(user)
        user_site_is(monkeypatch, user)
        on_sys_path(monkeypatch, user)
        connection = duckdb.connect(config={"extension_directories": [other, DEFAULT_DIRECTORY, pip_directory]})
        assert directories(connection) == [other, DEFAULT_DIRECTORY, pip_directory]


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
        assert directories(connection) == [DEFAULT_DIRECTORY, str(site / ".duckdb_extensions")]
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
    def test_user_list_precedes_the_directory(self, pip_directory):
        connection = duckdb.connect(config={"extension_directories": ["/a", "/b"]})
        assert directories(connection) == ["/a", "/b", pip_directory]

    @pytest.mark.parametrize(
        "config",
        [{"extension_directories": "[/a, /b]"}, {"EXTENSION_DIRECTORIES": ["/a", "/b"]}],
        ids=["list-literal-text", "upper-case-key"],
    )
    def test_user_list_is_read_as_the_engine_parsed_it(self, pip_directory, config):
        assert directories(duckdb.connect(config=config)) == ["/a", "/b", pip_directory]

    def test_explicit_empty_list_gets_the_default(self, pip_directory):
        connection = duckdb.connect(config={"extension_directories": []})
        assert directories(connection) == [DEFAULT_DIRECTORY, pip_directory]

    def test_deprecated_directory_setting_gets_no_default(self, pip_directory):
        connection = duckdb.connect(config={"extension_directory": "/singular"})
        assert connection.execute("SELECT current_setting('extension_directory')").fetchone()[0] == "/singular"
        assert directories(connection) == [pip_directory]

    def test_deprecated_directory_naming_the_directory_is_kept(self, pip_directory):
        connection = duckdb.connect(config={"extension_directory": pip_directory})
        assert connection.execute("SELECT current_setting('extension_directory')").fetchone()[0] == pip_directory
        assert directories(connection) == [pip_directory]

    @pytest.mark.parametrize("suffix", ["", os.sep], ids=["as-given", "trailing-separator"])
    def test_user_list_naming_the_directory_is_kept_as_given(self, pip_directory, suffix):
        connection = duckdb.connect(config={"extension_directories": [DEFAULT_DIRECTORY, pip_directory + suffix]})
        assert directories(connection) == [DEFAULT_DIRECTORY, pip_directory + suffix]

    def test_user_list_putting_the_directory_first_keeps_it_first(self, pip_directory):
        connection = duckdb.connect(config={"extension_directories": [pip_directory, "/a"]})
        assert directories(connection) == [pip_directory, "/a"]


class TestEveryConnection:
    @pytest.mark.parametrize("read_only", [False, True])
    def test_database_file(self, pip_directory, tmp_path, read_only):
        path = str(tmp_path / "file.duckdb")
        duckdb.connect(path).close()
        connection = duckdb.connect(path, read_only=read_only)
        assert directories(connection) == [DEFAULT_DIRECTORY, pip_directory]

    def test_second_connect_to_an_open_file_keeps_its_list(self, pip_directory, tmp_path, monkeypatch):
        path = str(tmp_path / "file.duckdb")
        first = duckdb.connect(path)
        lookups = count_lookups(monkeypatch)
        second = duckdb.connect(path)
        assert lookups == []
        assert directories(second) == directories(first) == [DEFAULT_DIRECTORY, pip_directory]

    def test_connect_after_the_file_is_closed_looks_up_again(self, pip_directory, tmp_path, monkeypatch):
        path = str(tmp_path / "file.duckdb")
        duckdb.connect(path).close()
        lookups = count_lookups(monkeypatch)
        assert directories(duckdb.connect(path)) == [DEFAULT_DIRECTORY, pip_directory]
        assert len(lookups) == 1

    def test_cursor(self, pip_directory):
        cursor = duckdb.connect().cursor()
        assert directories(cursor) == [DEFAULT_DIRECTORY, pip_directory]

    def test_default_connection(self, pip_directory):
        duckdb.default_connection().close()
        try:
            assert duckdb.sql(SETTING).fetchall() == [([DEFAULT_DIRECTORY, pip_directory],)]
        finally:
            duckdb.default_connection().close()

    def test_engine_lists_an_extension_file_in_the_directory(self, pip_directory):
        assert_engine_lists_a_file_in(duckdb.connect(), pip_directory)


class TestEngineUse:
    """What the engine does with the pip directory last: the first directory stays the install target and wins a load.

    Every case is offline: autoinstall_extension_repository points at an empty local directory, and the extension
    files are stand-ins. tpch is the extension, so these skip when it is linked into the build.
    """

    @pytest.fixture
    def connection(self, tmp_path):
        repository = tmp_path / "empty-repository"
        repository.mkdir()
        connection = duckdb.connect(config={"autoinstall_extension_repository": str(repository)})
        skip_unless_loadable(connection, "tpch")
        return connection

    def test_autoload_of_a_wheel_extension_attempts_an_install_but_load_finds_it(self, pip_directory, connection):
        place_stand_in(connection, pip_directory, "tpch")
        with pytest.raises(duckdb.Error, match="Failed to install local extension"):
            connection.execute("SELECT count(*) FROM tpch_queries()")
        with pytest.raises(duckdb.Error, match="is not a DuckDB extension"):
            connection.execute("LOAD tpch")

    def test_copy_in_the_default_directory_wins_a_load(self, pip_directory, home, connection):
        place_stand_in(connection, pip_directory, "tpch", size=16)
        in_home = place_stand_in(connection, home / ".duckdb" / "extensions", "tpch", size=16)
        with pytest.raises(duckdb.Error, match=re.escape(str(in_home))):
            connection.execute("LOAD tpch")

    @pytest.mark.parametrize("where", ["default", "pip"])
    def test_update_extensions_reads_the_default_directory(self, pip_directory, home, connection, where):
        directory = home / ".duckdb" / "extensions" if where == "default" else pip_directory
        place_stand_in(connection, directory, "tpch")
        updated = [row[0] for row in connection.execute("UPDATE EXTENSIONS").fetchall()]
        assert updated == (["tpch"] if where == "default" else [])

    def test_install_writes_into_the_default_directory(self, pip_directory, home, tmp_path):
        connection = duckdb.connect(
            config={"allow_unsigned_extensions": True, "allow_extensions_metadata_mismatch": True}
        )
        skip_unless_loadable(connection, "tpch")
        stand_in = tmp_path / "tpch.duckdb_extension"
        stand_in.write_bytes(b"\0" * 1024)
        connection.execute(f"INSTALL '{stand_in}'")
        assert installed_path(connection, home / ".duckdb" / "extensions", "tpch").is_file()
        assert list(Path(pip_directory).iterdir()) == []

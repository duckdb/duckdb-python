"""The extension directories that duckdb-core-ext wheels install into.

Every database ``duckdb.connect`` opens searches them. The wheels put their binaries under ``.duckdb_extensions`` in
the site directory an installer picks for them, a directory that is both an extension directory and a local repository.
"""

import functools
import os
import sys
import uuid
from pathlib import Path
from types import ModuleType

_DEFAULT_DIRECTORY = "~/.duckdb/extensions"
_PACKAGE_PARENT = str(Path(__file__).absolute().parent.parent)


def search_path(configured: list[str], deprecated_directory: str) -> list[str] | None:
    """The extension_directories of a database about to open, or None to keep the configured ones.

    INSTALL and autoinstall write into and check only the first directory, so a pip installed extension is used
    without a download only when its directory comes first; a read-only one goes last and stays searchable for LOAD.
    Setting the list replaces the engine default, which the list then names, unless the deprecated
    extension_directory is set: the engine searches that one before the list.

    Args:
        configured: The extension_directories the engine parsed from the connection's config.
        deprecated_directory: The extension_directory from the same config, empty when not set.

    Returns:
        The directories to search, each pip directory once among them, or None when there is no pip directory.
    """
    # Path.is_dir raises on a lookup error, such as a symlink into an unsearchable directory, and would fail connect.
    directories = [directory for directory in _pip_directories() if os.path.isdir(directory)]  # noqa: PTH112
    if not directories:
        return None
    if configured or deprecated_directory:
        taken = {_normalized(directory) for directory in directories}
        others = [entry for entry in configured if _normalized(entry) not in taken]
    else:
        others = [_DEFAULT_DIRECTORY]
    writable: list[str] = []
    read_only: list[str] = []
    for directory in directories:
        (writable if _is_writable(directory) else read_only).append(directory)
    return [*writable, *others, *read_only]


@functools.cache
def _pip_directories() -> tuple[str, ...]:
    return tuple(path for path in map(_extension_directory_of, _site_directories()) if path is not None)


def _extension_directory_of(site: str) -> str | None:
    path = str(Path(site, ".duckdb_extensions"))
    try:
        path.encode()
    except UnicodeEncodeError:
        # The engine takes paths as UTF-8, which a Linux file name made of undecodable bytes cannot be given as.
        return None
    if os.sep == "/" and "\\" in path:
        # The engine reads a backslash as a separator on every platform, so it would search another directory.
        return None
    return path


def _site_directories() -> list[str]:
    """The site directories the interpreter uses, in sys.path order, each once.

    An installer puts an extension wheel into the site directory of the scheme it picks for that wheel, which need not
    be the one holding duckdb: pip falls back to the user site when site-packages is read only, and a virtual
    environment with system site packages sees both. The directory holding the duckdb dist-info is added for a
    target directory or another sys.path entry that holds duckdb and the wheels together.
    """
    import site

    entries: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(sys.path):
        entry = str(Path(entry).absolute())
        entries.setdefault(_normalized(entry), (index, entry))
    placed: list[tuple[int, str]] = []
    for candidate in _site_candidates(site):
        entry_found = entries.get(_normalized(str(Path(candidate).absolute())))
        if entry_found is not None:
            placed.append(entry_found)
    distribution_site = _site_directory()
    if distribution_site is not None:
        # A meta path finder may report a dist-info from outside sys.path; it then counts last instead of failing.
        placed.append(entries.get(_normalized(distribution_site), (len(sys.path), distribution_site)))
    directories: list[str] = []
    seen: set[object] = set()
    for _, directory in sorted(placed):
        identity = _identity(directory)
        if identity not in seen:
            seen.add(identity)
            directories.append(directory)
    return directories


def _site_candidates(site: ModuleType) -> list[str]:
    candidates: list[str] = []
    # A stand-in site module of a frozen or embedded interpreter may lack these functions or fail in them.
    try:
        candidates.extend(site.getsitepackages())
        if site.ENABLE_USER_SITE:
            candidates.append(site.getusersitepackages())
    except (AttributeError, OSError):
        pass
    return candidates


def _identity(directory: str) -> object:
    # A lib64 and lib symlink pair puts one site directory on sys.path under two spellings.
    try:
        status = Path(directory).stat()
    except OSError:
        return _normalized(directory)
    return (status.st_dev, status.st_ino)


def _site_directory() -> str | None:
    # The first duckdb dist-info on sys.path need not belong to the imported package, so the one beside the package
    # wins; an editable install keeps its code in the checkout, away from its dist-info, and takes the first.
    import importlib.metadata

    first = None
    for distribution in importlib.metadata.distributions(name="duckdb"):
        site = str(Path(str(distribution.locate_file(""))).absolute())
        if _normalized(site) == _normalized(_PACKAGE_PARENT):
            return site
        if first is None:
            first = site
    return first


def _normalized(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _is_writable(directory: str) -> bool:
    # os.access grants write on every Windows directory, and tempfile retries an access-denied create up to
    # os.TMP_MAX times there before Python 3.13, so the probe is one explicit create.
    probe = Path(directory, f".write-probe-{os.getpid()}-{uuid.uuid4().hex}")
    temporary = getattr(os, "O_TEMPORARY", 0)
    try:
        descriptor = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY | temporary, 0o600)
        try:
            if not temporary:
                # Removed while still open, so a process killed before the close leaves no file behind.
                probe.unlink()
        finally:
            os.close(descriptor)
    except OSError:
        return False
    return True

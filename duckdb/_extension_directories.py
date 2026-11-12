"""The extension directories that duckdb-core-ext wheels install into.

Every database ``duckdb.connect`` opens searches them. The wheels put their binaries under ``.duckdb_extensions`` in
the site directory an installer picks for them, a directory that is both an extension directory and a local repository.
"""

import functools
import os
import sys
from pathlib import Path
from types import ModuleType

_DEFAULT_DIRECTORY = "~/.duckdb/extensions"
_PACKAGE_PARENT = str(Path(__file__).absolute().parent.parent)


def search_path(configured: list[str], deprecated_directory: str) -> list[str] | None:
    """The extension_directories of a database about to open, or None to keep the configured ones.

    INSTALL, FORCE INSTALL, UPDATE EXTENSIONS and autoinstall use the first directory only, so the pip directories go
    last: whatever is installed in the engine's own directory keeps precedence and stays the install target, an explicit
    LOAD finds a wheel's extension, and autoload of one that is not in the first directory attempts an install until
    the engine looks for a copy in every configured directory. Setting the list replaces the engine default, which the
    list then names, unless the deprecated extension_directory is set: the engine searches that one before the list.

    Args:
        configured: The extension_directories the engine parsed from the connection's config.
        deprecated_directory: The extension_directory from the same config, empty when not set.

    Returns:
        The configured directories as given, or the engine default when neither the list nor the deprecated directory
        is set, then each pip directory they do not name; None when no pip directory exists.
    """
    # Path.is_dir raises on a lookup error, such as a symlink into an unsearchable directory, and would fail connect.
    directories = [directory for directory in _pip_directories() if os.path.isdir(directory)]  # noqa: PTH112
    if not directories:
        return None
    others = list(configured) if configured or deprecated_directory else [_DEFAULT_DIRECTORY]
    named = {_normalized(entry) for entry in others}
    return [*others, *(directory for directory in directories if _normalized(directory) not in named)]


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
        absolute = _absolute(entry)
        if absolute is not None:
            entries.setdefault(_normalized(absolute), (index, absolute))
    placed: list[tuple[int, str]] = []
    for candidate in _site_candidates(site):
        absolute = _absolute(candidate)
        entry_found = entries.get(_normalized(absolute)) if absolute is not None else None
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


def _absolute(path: str) -> str | None:
    try:
        return str(Path(path).absolute())
    except OSError:
        # An empty entry stands for the working directory, which may be gone; the import system skips it then too.
        return None


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

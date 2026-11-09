"""The extension directory that duckdb-core-ext wheels install into, in the search path of every database opened here.

The wheels put their binaries under ``.duckdb_extensions`` beside the duckdb distribution, a directory that is both
an extension directory and a local repository.
"""

import functools
import os
import uuid
from pathlib import Path

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
        The directories to search, the pip directory once among them, or None when there is no pip directory.
    """
    directory = _pip_directory()
    if directory is None or not Path(directory).is_dir():
        return None
    if configured or deprecated_directory:
        others = [entry for entry in configured if _normalized(entry) != _normalized(directory)]
    else:
        others = [_DEFAULT_DIRECTORY]
    return [directory, *others] if _is_writable(directory) else [*others, directory]


@functools.cache
def _pip_directory() -> str | None:
    site = _site_directory()
    if site is None:
        return None
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

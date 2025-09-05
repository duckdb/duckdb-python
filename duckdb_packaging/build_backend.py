"""DuckDB PEP 517 and PEP 660 build backend.

This module wraps the scikit-build-core build backend because:
1. We need to be able to determine the version of the DuckDB submodule while building
   a source distribution, so that we can pass it in when building a wheel. The backend
   tries to figure out which duckdb version will be included in the sdist and saves
   the output
2. We want to use a custom version scheme with setuptools-scm, and PEP 621 provides no
   way to specify local code as a build-backend plugin. However, PEP 517 allows us to
   put our own build backend on the python path with the `build.backend-path` key. The
   side effect is that our version scheme is also on the path during the build.

Also see https://peps.python.org/pep-0517/#in-tree-build-backends.
"""
import sys
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Union
from scikit_build_core.build import (
    build_wheel as skbuild_build_wheel,
    build_editable,
    build_sdist as skbuild_build_sdist,
    get_requires_for_build_wheel,
    get_requires_for_build_sdist,
    get_requires_for_build_editable,
    prepare_metadata_for_build_wheel,
    prepare_metadata_for_build_editable,
)

from duckdb_packaging._versioning import create_git_tag, pep440_to_git_tag, get_git_describe, strip_post_from_version
from duckdb_packaging.setuptools_scm_version import forced_version_from_env, MAIN_BRANCH_VERSIONING


_DUCKDB_VERSION_FILENAME = "duckdb_version.txt"
_LOGGING_FORMAT = "[duckdb_pytooling.build_backend] {}"
_SKBUILD_CMAKE_OVERRIDE_GIT_DESCRIBE = "cmake.define.OVERRIDE_GIT_DESCRIBE"
# The below will check whether we should set a specific version in our build, and if so, set the version
_FORCED_PEP440_VERSION = forced_version_from_env()


def _log(msg: str, is_error: bool=False) -> None:
    """Log a message with build backend prefix.

    Args:
        msg: The message to log.
        is_error: If True, log to stderr; otherwise log to stdout.
    """
    print(_LOGGING_FORMAT.format(msg), flush=True, file=sys.stderr if is_error else sys.stdout)


def _in_git_repository() -> bool:
    """Check if the current directory is inside a git repository.

    Returns:
        True if .git directory exists, False otherwise.
    """
    return Path(".git").exists()


def _in_sdist() -> bool:
    """Check if the current directory is inside a git repository.

    Returns:
        True if the duckdb version file exists and PKG-INFO exists, False otherwise.
    """
    return _version_file_path().exists() and Path("PKG-INFO").exists()


def _duckdb_submodule_path() -> Path:
    """Verify that the duckdb submodule is checked out and usable and return its path."""
    if not _in_git_repository():
        raise RuntimeError("Not in a git repository, no duckdb submodule present")
    # search the duckdb submodule
    gitmodules_path = Path(".gitmodules")
    modules = dict()
    with gitmodules_path.open("r") as f:
        cur_module_path = None
        cur_module_reponame = None
        for line in f:
            if line.strip().startswith("[submodule"):
                if cur_module_reponame is not None and cur_module_path is not None:
                    modules[cur_module_reponame] = cur_module_path
                    cur_module_reponame = None
                    cur_module_path = None
            elif line.strip().startswith("path"):
                cur_module_path = line.split('=')[-1].strip()
            elif line.strip().startswith("url"):
                basename = os.path.basename(line.split('=')[-1].strip())
                cur_module_reponame = basename[:-4] if basename.endswith(".git") else basename
        if cur_module_reponame is not None and cur_module_path is not None:
            modules[cur_module_reponame] = cur_module_path

    if "duckdb" not in modules:
        raise RuntimeError("DuckDB submodule missing")

    duckdb_path = modules["duckdb"]
    # now check that the submodule is usable
    proc = subprocess.Popen(["git", "submodule", "status", duckdb_path], stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith("-"):
            raise RuntimeError(f"Duckdb submodule not initialized: {line}")
        if line.startswith("U"):
            raise RuntimeError(f"Duckdb submodule has merge conflicts: {line}")
        if line.startswith("+"):
            _log(f"WARNING: Duckdb submodule not clean: {line}")
    # all good
    return Path(duckdb_path)


def _version_file_path() -> Path:
    package_dir = Path(__file__).parent
    return package_dir / _DUCKDB_VERSION_FILENAME


def _write_duckdb_long_version(long_version: str)-> None:
    """Write the given version string to a file in the same directory as this module."""
    _version_file_path().write_text(long_version, encoding="utf-8")


def _read_duckdb_long_version() -> str:
    """Read the given version string from a file in the same directory as this module."""
    return _version_file_path().read_text(encoding="utf-8").strip()


def _skbuild_config_add(
        key: str, value: Union[List, str], config_settings: Dict[str, Union[List[str],str]], fail_if_exists: bool=False
):
    """Add or modify a configuration setting for scikit-build-core.

    This function handles adding values to scikit-build-core configuration settings,
    supporting both string and list types with appropriate merging behavior.

    Args:
        key: The configuration key to set (will be prefixed with 'skbuild.' if needed).
        value: The value to add (string or list).
        config_settings: The configuration dictionary to modify.
        fail_if_exists: If True, raise an error if the key already exists.

    Raises:
        RuntimeError: If fail_if_exists is True and key exists, or on type mismatches.
        AssertionError: If config_settings is None.

    Behavior Rules:
        - String value + list setting: value is appended to the list
        - String value + string setting: existing value is overridden
        - List value + list setting: existing list is extended
        - List value + string setting: raises RuntimeError

    Note:
        scikit-build-core's preference logic for config sources still applies,
        considering env vars, config_settings and pyproject in that order,
        without merging between those sources.
    """
    assert config_settings is not None, "config_settings must not be None"
    store_key = key if key in config_settings else "skbuild." + key
    key_exists = store_key in config_settings
    key_exists_as_str = key_exists and isinstance(config_settings[store_key], str)
    key_exists_as_list = key_exists and isinstance(config_settings[store_key], list)
    val_is_str = isinstance(value, str)
    val_is_list = isinstance(value, list)
    if not key_exists:
        config_settings[store_key] = value
    elif fail_if_exists:
        raise RuntimeError(f"{key} already present in config and may not be overridden")
    elif key_exists_as_list and val_is_list:
        config_settings[store_key].extend(value)
    elif key_exists_as_list and val_is_str:
        config_settings[store_key].append(value)
    elif key_exists_as_str and val_is_str:
        _log(f"WARNING: overriding existing value in {store_key}")
        config_settings[store_key] = value
    else:
        raise RuntimeError(
            f"Type mismatch: cannot set {store_key} ({type(config_settings[store_key])}) to `{value}` ({type(value)})"
        )


def build_sdist(sdist_directory: str, config_settings: Optional[Dict[str, Union[List[str],str]]] = None) -> str:
    """Build a source distribution using the DuckDB submodule.

    This function extracts the DuckDB version from either the git submodule and saves it
    to a version file before building the sdist with scikit-build-core. If _FORCED_PEP440_VERSION
    was set then we first create a tag on the submodule.

    Args:
        sdist_directory: Directory where the sdist will be created.
        config_settings: Optional build configuration settings.

    Returns:
        The filename of the created sdist.

    Raises:
        RuntimeError: If not in a git repository or DuckDB submodule issues.
    """
    if not _in_git_repository():
        raise RuntimeError("Not in a git repository, can't create an sdist")
    submodule_path = _duckdb_submodule_path()
    if _FORCED_PEP440_VERSION is not None:
        duckdb_version = pep440_to_git_tag(strip_post_from_version(_FORCED_PEP440_VERSION))
    else:
        duckdb_version = get_git_describe(repo_path=submodule_path, since_minor=MAIN_BRANCH_VERSIONING)
    _write_duckdb_long_version(duckdb_version)
    return skbuild_build_sdist(sdist_directory, config_settings=config_settings)


def build_wheel(
        wheel_directory: str,
        config_settings: Optional[Dict[str, Union[List[str],str]]] = None,
        metadata_directory: Optional[str] = None,
) -> str:
    """Build a wheel from either git submodule or extracted sdist sources.

    This function builds a wheel using scikit-build-core, handling two scenarios:
    1. In a git repository: builds directly from the DuckDB submodule
    2. In an sdist: reads the saved DuckDB version and passes it to CMake

    Args:
        wheel_directory: Directory where the wheel will be created.
        config_settings: Optional build configuration settings.
        metadata_directory: Optional directory for metadata preparation.

    Returns:
        The filename of the created wheel.

    Raises:
        RuntimeError: If not in a git repository or sdist environment.
    """
    # First figure out the duckdb version we should use
    duckdb_version = None
    if not _in_git_repository():
        if not _in_sdist():
            raise RuntimeError("Not in a git repository nor in an sdist, can't build a wheel")
        _log("Building duckdb wheel from sdist. Reading duckdb version from file.")
        config_settings = config_settings or {}
        duckdb_version = _read_duckdb_long_version()
    elif _FORCED_PEP440_VERSION is not None:
        duckdb_version = pep440_to_git_tag(strip_post_from_version(_FORCED_PEP440_VERSION))

    # We add the found version to the OVERRIDE_GIT_DESCRIBE cmake var
    if duckdb_version is not None:
        _skbuild_config_add(_SKBUILD_CMAKE_OVERRIDE_GIT_DESCRIBE, duckdb_version, config_settings, fail_if_exists=True)
        _log(f"{_SKBUILD_CMAKE_OVERRIDE_GIT_DESCRIBE} set to {duckdb_version}")
    else:
        _log("No explicit DuckDB submodule version provided. Letting CMake figure it out.")


    return skbuild_build_wheel(wheel_directory, config_settings=config_settings, metadata_directory=metadata_directory)


__all__ = [
    "build_wheel",
    "build_sdist",
    "build_editable",
    "get_requires_for_build_wheel",
    "get_requires_for_build_sdist",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_wheel",
    "prepare_metadata_for_build_editable",
]

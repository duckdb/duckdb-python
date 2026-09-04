"""DuckDB PEP 517 and PEP 660 build backend.

This module wraps the scikit-build-core build backend because the identity of a
build, the package version and commit and the DuckDB version and commit, has to
be settled once and reach three places: the package metadata, the generated
duckdb/_build_info.py module, and DuckDB's CMake. An sdist has no git history,
so it carries that identity in duckdb_packaging/build_info.json for the wheel
build to read. PEP 517 allows an in-tree backend through `build-system.backend-path`,
which also makes duckdb_packaging importable as the metadata provider.

Also see https://peps.python.org/pep-0517/#in-tree-build-backends.
"""

import sys
from collections.abc import Callable

from scikit_build_core.build import (
    build_editable as skbuild_build_editable,
)
from scikit_build_core.build import (
    build_sdist as skbuild_build_sdist,
)
from scikit_build_core.build import (
    build_wheel as skbuild_build_wheel,
)
from scikit_build_core.build import (
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)

from duckdb_packaging.build_info import (
    BuildInfo,
    is_git_checkout,
    resolve_build_info,
    write_build_info_json,
    write_build_info_module,
)

_LOGGING_FORMAT = "[duckdb_packaging.build_backend] {}"
_SKBUILD_CMAKE_OVERRIDE_GIT_DESCRIBE = "cmake.define.OVERRIDE_GIT_DESCRIBE"
_SKBUILD_CMAKE_GIT_COMMIT_HASH = "cmake.define.GIT_COMMIT_HASH"

ConfigSettings = dict[str, list[str] | str]
Builder = Callable[..., str]


def _log(msg: str) -> None:
    """Log a message with build backend prefix.

    Args:
        msg: The message to log.
    """
    print(_LOGGING_FORMAT.format(msg), flush=True, file=sys.stderr)


def _skbuild_config_add(key: str, value: list | str, config_settings: ConfigSettings) -> None:
    """Add or modify a configuration setting for scikit-build-core.

    This function handles adding values to scikit-build-core configuration settings,
    supporting both string and list types with appropriate merging behavior.

    Args:
        key: The configuration key to set (will be prefixed with 'skbuild.' if needed).
        value: The value to add (string or list).
        config_settings: The configuration dictionary to modify.

    Raises:
        RuntimeError: If this would overwrite an existing value, or on type mismatches.
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
    elif key_exists_as_list and val_is_list:
        config_settings[store_key].extend(value)
    elif key_exists_as_list and val_is_str:
        config_settings[store_key].append(value)
    elif key_exists_as_str and val_is_str:
        msg = f"{key} already present in config and may not be overridden"
        raise RuntimeError(msg)
    else:
        msg = f"Type mismatch: cannot set {store_key} ({type(config_settings[store_key])}) to `{value}` ({type(value)})"
        raise RuntimeError(msg)


def _describe(info: BuildInfo) -> str:
    return (
        f"package {info.package_version} at {info.package_commit}, DuckDB {info.duckdb_version} at {info.duckdb_commit}"
    )


def _add_duckdb_defines(info: BuildInfo, config_settings: ConfigSettings) -> None:
    # DuckDB's CMake takes the version string literally and does not read the commit
    # out of it, so an sdist build without git needs the commit passed alongside.
    _skbuild_config_add(_SKBUILD_CMAKE_OVERRIDE_GIT_DESCRIBE, info.duckdb_version, config_settings)
    _skbuild_config_add(_SKBUILD_CMAKE_GIT_COMMIT_HASH, info.duckdb_commit, config_settings)


def _build(
    builder: Builder, directory: str, config_settings: ConfigSettings | None, metadata_directory: str | None
) -> str:
    config_settings = config_settings or {}
    info = resolve_build_info()
    _log(f"Building {_describe(info)}")
    write_build_info_module(info)
    _add_duckdb_defines(info, config_settings)
    return builder(directory, config_settings=config_settings, metadata_directory=metadata_directory)


def build_sdist(sdist_directory: str, config_settings: ConfigSettings | None = None) -> str:
    """Build a source distribution that carries the identity it was built from.

    Args:
        sdist_directory: Directory where the sdist will be created.
        config_settings: Optional build configuration settings.

    Returns:
        The filename of the created sdist.

    Raises:
        RuntimeError: If not in a git repository or the DuckDB submodule is unusable.
    """
    if not is_git_checkout():
        msg = "Not in a git repository, can't create an sdist"
        raise RuntimeError(msg)
    info = resolve_build_info()
    _log(f"Packaging {_describe(info)}")
    write_build_info_json(info)
    write_build_info_module(info)
    return skbuild_build_sdist(sdist_directory, config_settings=config_settings)


def build_wheel(
    wheel_directory: str,
    config_settings: ConfigSettings | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build a wheel from either a git checkout or an unpacked sdist.

    Args:
        wheel_directory: Directory where the wheel will be created.
        config_settings: Optional build configuration settings.
        metadata_directory: Optional directory for metadata preparation.

    Returns:
        The filename of the created wheel.

    Raises:
        RuntimeError: If not in a git repository nor in an sdist.
    """
    return _build(skbuild_build_wheel, wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: ConfigSettings | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build an editable wheel from a git checkout.

    Args:
        wheel_directory: Directory where the wheel will be created.
        config_settings: Optional build configuration settings.
        metadata_directory: Optional directory for metadata preparation.

    Returns:
        The filename of the created wheel.

    Raises:
        RuntimeError: If not in a git repository.
    """
    if not is_git_checkout():
        msg = "Not in a git repository, can't build an editable install"
        raise RuntimeError(msg)
    return _build(skbuild_build_editable, wheel_directory, config_settings, metadata_directory)


__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]

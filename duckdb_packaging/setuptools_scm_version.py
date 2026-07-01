"""setuptools_scm integration for DuckDB Python versioning.

This module provides the setuptools_scm version scheme and handles environment variable overrides
to match the exact behavior of the original DuckDB Python package.
"""

import os
import re
from typing import Protocol

# Import from our own versioning module to avoid duplication
from ._versioning import format_version, parse_version

# MAIN_BRANCH_VERSIONING should be 'True' on main branch only
MAIN_BRANCH_VERSIONING = False

SCM_PRETEND_ENV_VAR = "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DUCKDB"
SCM_GLOBAL_PRETEND_ENV_VAR = "SETUPTOOLS_SCM_PRETEND_VERSION"
OVERRIDE_GIT_DESCRIBE_ENV_VAR = "OVERRIDE_GIT_DESCRIBE"


class _VersionObject(Protocol):
    tag: object
    distance: int
    dirty: bool


def _main_branch_versioning() -> bool:
    from_env = os.getenv("MAIN_BRANCH_VERSIONING")
    return from_env == "1" if from_env is not None else MAIN_BRANCH_VERSIONING


def version_scheme(version: _VersionObject) -> str:
    """setuptools_scm version scheme that matches DuckDB's original behavior.

    Args:
        version: setuptools_scm version object

    Returns:
        PEP440 compliant version string
    """
    print(f"[version_scheme] version object: {version}")
    print(f"[version_scheme] version.tag: {version.tag}")
    print(f"[version_scheme] version.distance: {version.distance}")
    print(f"[version_scheme] version.dirty: {version.dirty}")

    # Handle case where tag is None
    if version.tag is None:
        msg = "Need a valid version. Did you set a fallback_version in pyproject.toml?"
        raise ValueError(msg)

    distance = int(version.distance or 0)
    try:
        if distance == 0 and not version.dirty:
            return _tag_to_version(str(version.tag))
        return _bump_dev_version(str(version.tag), distance)
    except Exception as e:
        msg = f"Failed to bump version: {e}"
        raise RuntimeError(msg) from e


def _tag_to_version(tag: str) -> str:
    """Bump the version when we're on a tag."""
    major, minor, patch, post, rc = parse_version(tag)
    return format_version(major, minor, patch, post=post, rc=rc)


def _bump_dev_version(base_version: str, distance: int) -> str:
    """Bump the given version."""
    if distance == 0:
        msg = "Dev distance is 0, cannot bump version."
        raise ValueError(msg)
    major, minor, patch, post, rc = parse_version(base_version)

    if post != 0:
        # We're developing on top of a post-release
        return f"{format_version(major, minor, patch, post=post + 1)}.dev{distance}"
    elif rc != 0:
        # We're developing on top of an rc
        return f"{format_version(major, minor, patch, rc=rc + 1)}.dev{distance}"
    elif _main_branch_versioning():
        return f"{format_version(major, minor + 1, 0)}.dev{distance}"
    return f"{format_version(major, minor, patch + 1)}.dev{distance}"


def forced_version_from_env() -> str:
    """Handle getting versions from environment variables.

    Only supports a single way of manually overriding the version through
    OVERRIDE_GIT_DESCRIBE. If SETUPTOOLS_SCM_PRETEND_VERSION* is set, it gets unset.
    """
    override_value = os.getenv(OVERRIDE_GIT_DESCRIBE_ENV_VAR)
    pep440_version = None

    if override_value:
        print(f"[versioning] Found {OVERRIDE_GIT_DESCRIBE_ENV_VAR}={override_value}")
        pep440_version = _git_describe_override_to_pep_440(override_value)
        os.environ[SCM_PRETEND_ENV_VAR] = pep440_version
        print(f"[versioning] Injected {SCM_PRETEND_ENV_VAR}={pep440_version}")
    elif SCM_PRETEND_ENV_VAR in os.environ:
        _remove_unsupported_env_var(SCM_PRETEND_ENV_VAR)

    # Always check and remove unsupported SETUPTOOLS_SCM_PRETEND_VERSION
    if SCM_GLOBAL_PRETEND_ENV_VAR in os.environ:
        _remove_unsupported_env_var(SCM_GLOBAL_PRETEND_ENV_VAR)

    return pep440_version


def _git_describe_override_to_pep_440(override_value: str) -> str:
    """Process the OVERRIDE_GIT_DESCRIBE value."""
    describe_pattern = re.compile(
        r"""
        ^v(?P<tag>\d+\.\d+\.\d+(?:-post\d+|-rc\d+)?) # vX.Y.Z or vX.Y.Z-postN or vX.Y.Z-rcN
        (?:-(?P<distance>\d+))?                      # optional -N
        (?:-g(?P<hash>[0-9a-fA-F]+))?                # optional -g<sha>
        $""",
        re.VERBOSE,
    )

    match = describe_pattern.match(override_value)
    if not match:
        msg = f"Invalid git describe override: {override_value}"
        raise ValueError(msg)

    version, distance, commit_hash = match.groups()

    # Convert version format to PEP440 format (v1.3.1-post1 -> 1.3.1.post1)
    if "-post" in version:
        version = version.replace("-post", ".post")
    elif "-rc" in version:
        version = version.replace("-rc", "rc")

    # Bump version and format according to PEP440
    distance = int(distance or 0)
    pep440_version = _tag_to_version(str(version)) if distance == 0 else _bump_dev_version(str(version), distance)
    if commit_hash:
        pep440_version += f"+g{commit_hash.lower()}"

    return pep440_version


def _remove_unsupported_env_var(env_var: str) -> None:
    """Remove an unsupported environment variable with a warning."""
    print(f"[versioning] WARNING: We do not support {env_var}! Removing.")
    del os.environ[env_var]

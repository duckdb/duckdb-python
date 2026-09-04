"""DuckDB Python versioning utilities.

Version parsing and formatting, the git tag spelling of a version, the release
version declared in pyproject.toml, and the git primitives a build identity is
derived from.
"""

import datetime
import os
import pathlib
import re
import subprocess
import sys

import tomllib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

# Accepts the PEP440 alternative pre-release spellings and separators, see
# https://packaging.python.org/en/latest/specifications/version-specifiers/#pre-release-spelling
VERSION_RE = re.compile(
    r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)"
    r"(?:[._-]?(?P<pre_kind>alpha|beta|preview|pre|rc|a|b|c)[._-]?(?P<pre_num>[0-9]+)?"
    r"|[._-]?post(?P<post>[0-9]+))?$",
    re.IGNORECASE,
)

PRE_RELEASE_KINDS = ("a", "b", "rc")

# The post component of a version in git tag form. The only part we strip
# before handing a forced version to DuckDB, see duckdb_describe_from_override.
POST_COMPONENT_RE = re.compile(r"-post[0-9]+", re.IGNORECASE)

# PEP440 alternative pre-release spellings and their canonical form
_PRE_KIND_ALIASES = {
    "a": "a",
    "alpha": "a",
    "b": "b",
    "beta": "b",
    "c": "rc",
    "pre": "rc",
    "preview": "rc",
    "rc": "rc",
}


def parse_version(version: str) -> tuple[int, int, int, int, tuple[str, int] | None]:
    """Parse a version string into its components.

    Alternative PEP440 pre-release spellings ("1.3.1-alpha1", "1.3.1.c3") are
    accepted and normalized to the canonical kinds "a", "b" and "rc". An
    omitted pre-release numeral means 0 ("1.3.1a" == "1.3.1a0").

    Args:
        version: Version string (e.g., "1.3.1", "1.3.1a1", "1.3.1b2", "1.3.1rc3" or "1.3.1.post2")

    Returns:
        Tuple of (major, minor, patch, post, pre). pre is a (kind, number)
        tuple with kind one of "a", "b", "rc", or None for non-pre-releases.

    Raises:
        ValueError: If version format is invalid
    """
    match = VERSION_RE.match(version)
    if not match:
        msg = f"Invalid version format: {version} (expected X.Y.Z, X.Y.Z(a|b|rc)N or X.Y.Z.postN)"
        raise ValueError(msg)

    major, minor, patch, pre_kind, pre_num, post = match.groups()
    pre = (_PRE_KIND_ALIASES[pre_kind.lower()], int(pre_num or 0)) if pre_kind else None
    return int(major), int(minor), int(patch), int(post or 0), pre


def format_version(major: int, minor: int, patch: int, post: int = 0, pre: tuple[str, int] | None = None) -> str:
    """Format version components into a version string.

    Args:
        major: Major version number
        minor: Minor version number
        patch: Patch version number
        post: Post-release number
        pre: Pre-release as a (kind, number) tuple, kind one of "a", "b", "rc"

    Returns:
        Formatted version string
    """
    version = f"{major}.{minor}.{patch}"
    if post != 0 and pre is not None:
        msg = "post and pre are mutually exclusive"
        raise ValueError(msg)
    if post != 0:
        version += f".post{post}"
    if pre is not None:
        kind, num = pre
        if kind not in PRE_RELEASE_KINDS:
            msg = f"Invalid pre-release kind: {kind} (expected one of {PRE_RELEASE_KINDS})"
            raise ValueError(msg)
        version += f"{kind}{num}"
    return version


def git_tag_to_pep440(git_tag: str) -> str:
    """Convert git tag format to canonical PEP440 format.

    Alternative pre-release spellings are normalized ("v1.3.1-alpha1" -> "1.3.1a1").

    Args:
        git_tag: Git tag (e.g., "v1.3.1", "v1.3.1-post1", "v1.3.1-a1")

    Returns:
        Canonical PEP440 version string (e.g., "1.3.1", "1.3.1.post1", "1.3.1a1")

    Raises:
        ValueError: If the tag does not denote a valid version
    """
    # Remove 'v' prefix if present, the suffixes parse as-is (PEP440 allows a dash separator)
    version = git_tag[1:] if git_tag.startswith("v") else git_tag
    major, minor, patch, post, pre = parse_version(version)
    return format_version(major, minor, patch, post=post, pre=pre)


def pep440_to_git_tag(version: str) -> str:
    """Convert PEP440 version to canonical git tag format.

    Alternative pre-release spellings are normalized ("1.3.1-alpha1" -> "v1.3.1-a1").

    Args:
        version: PEP440 version string (e.g., "1.3.1.post1", "1.3.1rc2" or "1.3.1a1")

    Returns:
        Git tag format (e.g., "v1.3.1-post1", "v1.3.1-rc2" or "v1.3.1-a1")

    Raises:
        ValueError: If the version is invalid
    """
    major, minor, patch, post, pre = parse_version(version)
    tag = f"v{major}.{minor}.{patch}"
    if post != 0:
        tag += f"-post{post}"
    if pre is not None:
        tag += f"-{pre[0]}{pre[1]}"
    return tag


def release_version(pyproject_path: pathlib.Path = PYPROJECT_PATH) -> str:
    """The version this branch releases next, declared in pyproject.toml.

    Read from ``[tool.duckdb_packaging] release_version``. A release build must
    force exactly this version, a dev build appends its dev segment to it.

    Args:
        pyproject_path: The pyproject.toml to read

    Returns:
        The declared version in canonical PEP440 form

    Raises:
        ValueError: If the setting is missing or is not a valid version
        TypeError: If the setting is not a string
    """
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    try:
        value = data["tool"]["duckdb_packaging"]["release_version"]
    except KeyError:
        msg = f"[tool.duckdb_packaging].release_version is missing from {pyproject_path}"
        raise ValueError(msg) from None
    if not isinstance(value, str):
        msg = f"[tool.duckdb_packaging].release_version must be a string, got {value!r}"
        raise TypeError(msg)
    major, minor, patch, post, pre = parse_version(value)
    return format_version(major, minor, patch, post=post, pre=pre)


def next_release_version(version: str) -> str:
    """The version a branch moves to after releasing the given one.

    A final or post release moves to the next patch. A pre-release moves to its
    next number, since whether the next step is another phase or the final is a
    human decision.

    Args:
        version: The version just released, e.g. "2.0.0", "2.0.1.post1", "2.0.0a1"

    Returns:
        The next version, e.g. "2.0.1", "2.0.2", "2.0.0a2"
    """
    major, minor, patch, _post, pre = parse_version(version)
    if pre is not None:
        kind, num = pre
        return format_version(major, minor, patch, pre=(kind, num + 1))
    return format_version(major, minor, patch + 1)


def _git(repo_path: pathlib.Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=repo_path)
    except FileNotFoundError as e:
        msg = "git executable can't be found"
        raise RuntimeError(msg) from e
    return result.stdout.strip()


def commit_hash(repo_path: pathlib.Path) -> str:
    """The full hash of HEAD in the given repository.

    Raises:
        subprocess.CalledProcessError: If the path is not a git repository
        RuntimeError: If the git executable can't be found
    """
    return _git(repo_path, "rev-parse", "HEAD")


def commit_time(repo_path: pathlib.Path) -> int:
    """The committer time of HEAD in the given repository, as a unix timestamp.

    The committer time, not the author time: rebases and merges keep the author
    time, so only the committer time reflects when a commit landed on a branch.

    Raises:
        subprocess.CalledProcessError: If the path is not a git repository
        RuntimeError: If the git executable can't be found
    """
    return int(_git(repo_path, "log", "-1", "--format=%ct", "HEAD"))


def dev_number(*commit_times: int) -> int:
    """The dev segment of a build made from commits with the given committer times.

    The latest of the times, formatted as yymmddhhmm in UTC. A build of the same
    commits always yields the same number, and any later commit in either
    repository yields a larger one.

    Args:
        commit_times: Committer times as unix timestamps, at least one

    Returns:
        The dev number, e.g. 2609031653 for 2026-09-03 16:53 UTC
    """
    latest = datetime.datetime.fromtimestamp(max(commit_times), tz=datetime.UTC)
    return int(latest.strftime("%y%m%d%H%M"))


# DuckDB's version script honours these itself. They never reach it: a forced
# package version must not leak into the engine version derived from the submodule.
_DUCKDB_VERSION_SCRIPT_ENV_VARS = ("OVERRIDE_GIT_DESCRIBE", "DUCKDB_VERSION", "DUCKDB_COMMIT")


def duckdb_dev_version(submodule_path: pathlib.Path) -> str:
    """The version DuckDB gives itself at the submodule's HEAD, e.g. "v2.0.0-dev14120".

    Computed by DuckDB's own ``scripts/ci/version.py``, so that DuckDB stays
    the single authority on its version string.

    Raises:
        subprocess.CalledProcessError: If the script fails
    """
    env = {name: value for name, value in os.environ.items() if name not in _DUCKDB_VERSION_SCRIPT_ENV_VARS}
    result = subprocess.run(
        [sys.executable, "scripts/ci/version.py"],
        capture_output=True,
        text=True,
        check=True,
        cwd=submodule_path,
        env=env,
    )
    return result.stdout.strip()


def duckdb_describe_from_override(override: str) -> str:
    """Map a forced package version to the version string DuckDB gets built with.

    The value passes through untouched apart from a post component. Post
    releases repackage the same engine, so DuckDB never sees the post suffix.

    Everything else reaches DuckDB byte for byte. Its CMake is the only
    authority on which version strings are valid, and it fails loud on the ones
    it does not support. Normalizing here would corrupt them: PEP440 spells an
    alpha "a1", DuckDB only accepts "alpha1" and rejects "a1".

    Args:
        override: Forced version in git tag form (e.g. "v1.3.1", "v1.3.1-post1",
            "v1.3.1-alpha1", "v1.3.1-rc2-5-g1234567")

    Returns:
        The version string for DuckDB's build (e.g. "v1.3.1", "v1.3.1-alpha1")
    """
    return POST_COMPONENT_RE.sub("", override, count=1)

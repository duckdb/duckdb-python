"""DuckDB Python versioning utilities. This will only work on Python >= 3.3 and on non-mobile platforms.

This module provides utilities for version management including:
- Version bumping (major, minor, patch, post)
- Git tag creation and management
- Version parsing and validation
"""

import pathlib
import re
import subprocess

# Accepts the PEP440 alternative pre-release spellings and separators, see
# https://packaging.python.org/en/latest/specifications/version-specifiers/#pre-release-spelling
VERSION_RE = re.compile(
    r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)"
    r"(?:[._-]?(?P<pre_kind>alpha|beta|preview|pre|rc|a|b|c)[._-]?(?P<pre_num>[0-9]+)?"
    r"|[._-]?post(?P<post>[0-9]+))?$",
    re.IGNORECASE,
)

PRE_RELEASE_KINDS = ("a", "b", "rc")

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


def get_current_version() -> str | None:
    """Get the current version from git tags.

    Returns:
        Current version string or None if no tags exist
    """
    try:
        # Get the latest tag
        result = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True, check=True)
        tag = result.stdout.strip()
        return git_tag_to_pep440(tag)
    except subprocess.CalledProcessError:
        return None


def create_git_tag(version: str, message: str | None = None, repo_path: pathlib.Path | None = None) -> None:
    """Create a git tag for the given version.

    Args:
        version: Version string (PEP440 format)
        message: Optional tag message
        repo_path: Optional path to git repository (defaults to current directory)

    Raises:
        subprocess.CalledProcessError: If git command fails
    """
    tag_name = pep440_to_git_tag(version)

    cmd = ["git", "tag"]
    if message:
        cmd.extend(["-a", tag_name, "-m", message])
    else:
        cmd.append(tag_name)

    # If a repository path is provided, use it as the working directory
    cwd = repo_path if repo_path is not None else None
    subprocess.run(cmd, check=True, cwd=cwd)


def duckdb_tag_from_pep440(version: str) -> str:
    """Map a forced package version to a DuckDB version tag.

    Post releases repackage the stable engine, so the post suffix is dropped.
    Pre-release suffixes pass through: DuckDB's build validates what it
    supports and fails on versions it does not (yet) accept.

    Args:
        version: PEP440 version string of the Python package

    Returns:
        DuckDB git tag (e.g. "v1.3.1", "v1.3.1-rc2" or "v1.3.1-a1")
    """
    major, minor, patch, _post, pre = parse_version(version)
    return pep440_to_git_tag(format_version(major, minor, patch, pre=pre))


def get_git_describe(
    repo_path: pathlib.Path | None = None,
    since_major: bool = False,  # noqa: FBT001
    since_minor: bool = False,  # noqa: FBT001
) -> str:
    """Get git describe output for version determination.

    Returns:
        Git describe output

    Raises:
        subprocess.CalledProcessError: If git describe fails (e.g. no tags exist)
        RuntimeError: If the git executable can't be found
    """
    cwd = repo_path if repo_path is not None else None
    pattern = "v*.*.*"
    if since_major:
        pattern = "v*.0.0"
    elif since_minor:
        pattern = "v*.*.0"
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--long", "--match", pattern],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )
        result.check_returncode()
        return result.stdout.strip()
    except FileNotFoundError as e:
        msg = "git executable can't be found"
        raise RuntimeError(msg) from e

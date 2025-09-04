"""DuckDB Python versioning utilities. This will only work on Python >= 3.3 and on non-mobile platforms.

This module provides utilities for version management including:
- Version bumping (major, minor, patch, post)
- Git tag creation and management
- Version parsing and validation
"""
import pathlib
import subprocess
from typing import Optional
import re


VERSION_RE = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)(?:rc(?P<rc>[0-9]+)|\.post(?P<post>[0-9]+))?$")


def parse_version(version: str) -> tuple[int, int, int, int, int]:
    """Parse a version string into its components.

    Args:
        version: Version string (e.g., "1.3.1", "1.3.2.rc3" or "1.3.1.post2")

    Returns:
        Tuple of (major, minor, patch, post, rc)

    Raises:
        ValueError: If version format is invalid
    """
    match = VERSION_RE.match(version)
    if not match:
        raise ValueError(f"Invalid version format: {version} (expected X.Y.Z, X.Y.Z.rcM or X.Y.Z.postN)")

    major, minor, patch, rc, post = match.groups()
    return int(major), int(minor), int(patch), int(post or 0), int(rc or 0)


def format_version(major: int, minor: int, patch: int, post: int = 0, rc: int = 0) -> str:
    """Format version components into a version string.

    Args:
        major: Major version number
        minor: Minor version number
        patch: Patch version number
        post: Post-release number
        rc: RC number

    Returns:
        Formatted version string
    """
    version = f"{major}.{minor}.{patch}"
    if post != 0 and rc != 0:
        raise ValueError("post and rc are mutually exclusive")
    if post != 0:
        version += f".post{post}"
    if rc != 0:
        version += f"rc{rc}"
    return version


def git_tag_to_pep440(git_tag: str) -> str:
    """Convert git tag format to PEP440 format.

    Args:
        git_tag: Git tag (e.g., "v1.3.1", "v1.3.1-post1")

    Returns:
        PEP440 version string (e.g., "1.3.1", "1.3.1.post1")
    """
    # Remove 'v' prefix if present
    version = git_tag[1:] if git_tag.startswith('v') else git_tag

    if "-post" in version:
        assert 'rc' not in version
        version = version.replace("-post", ".post")
    elif '-rc' in version:
        version = version.replace("-rc", "rc")

    return version


def pep440_to_git_tag(version: str) -> str:
    """Convert PEP440 version to git tag format.

    Args:
        version: PEP440 version string (e.g., "1.3.1.post1" or "1.3.1rc2")

    Returns:
        Git tag format (e.g., "v1.3.1-post1")
    """
    if '.post' in version:
        assert 'rc' not in version
        version = version.replace(".post", "-post")
    elif 'rc' in version:
        version = version.replace("rc", "-rc")

    return f"v{version}"


def get_current_version() -> Optional[str]:
    """Get the current version from git tags.

    Returns:
        Current version string or None if no tags exist
    """
    try:
        # Get the latest tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True
        )
        tag = result.stdout.strip()
        return git_tag_to_pep440(tag)
    except subprocess.CalledProcessError:
        return None


def create_git_tag(version: str, message: Optional[str] = None, repo_path: Optional[pathlib.Path] = None) -> None:
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


def strip_post_from_version(version: str) -> str:
    """
    Removing post-release suffixes from the given version.

    DuckDB doesn't allow post-release versions, so .post* suffixes are stripped.
    """
    return re.sub(r"[\.-]post[0-9]+", "", version)


def get_git_describe(repo_path: Optional[pathlib.Path] = None, since_major=False, since_minor=False) -> Optional[str]:
    """Get git describe output for version determination.

    Returns:
        Git describe output or None if no tags exist
    """
    cwd = repo_path if repo_path is not None else None
    pattern="v*.*.*"
    if since_major:
        pattern="v*.0.0"
    elif since_minor:
        pattern="v*.*.0"
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--long", "--match", pattern],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd
        )
        result.check_returncode()
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("git executable can't be found")

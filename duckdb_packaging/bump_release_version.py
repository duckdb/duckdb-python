"""Move release_version in pyproject.toml past the version just released.

Run by the release workflow once a stable release is published, so the next dev
build sorts after it. See next_release_version for the rule.
"""

import argparse
import re
import sys
from pathlib import Path

from ._versioning import PYPROJECT_PATH, next_release_version, release_version

_RELEASE_VERSION_LINE_RE = re.compile(r'^(release_version\s*=\s*)"[^"]*"', re.MULTILINE)


def bump(pyproject_path: Path = PYPROJECT_PATH) -> tuple[str, str]:
    """Rewrite release_version in place.

    Returns:
        The version before and after

    Raises:
        ValueError: If the setting is missing, or the rewrite did not land on it
    """
    current = release_version(pyproject_path)
    new = next_release_version(current)
    text = pyproject_path.read_text(encoding="utf-8")
    updated, hits = _RELEASE_VERSION_LINE_RE.subn(rf'\g<1>"{new}"', text, count=1)
    if hits != 1:
        msg = f"No release_version line to rewrite in {pyproject_path}"
        raise ValueError(msg)
    pyproject_path.write_text(updated, encoding="utf-8")
    if release_version(pyproject_path) != new:
        msg = (
            f"Rewrote the wrong line in {pyproject_path}, [tool.duckdb_packaging].release_version still reads {current}"
        )
        raise ValueError(msg)
    return current, new


def main(argv: list[str] | None = None) -> int:
    """Bump and print "<before> -> <after>"."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT_PATH, help="the pyproject.toml to rewrite")
    args = parser.parse_args(argv)
    current, new = bump(args.pyproject)
    print(f"{current} -> {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

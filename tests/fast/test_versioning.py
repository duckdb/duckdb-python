"""Tests for duckdb_packaging versioning functionality."""

import contextlib
import datetime
import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

duckdb_packaging = pytest.importorskip("duckdb_packaging")

from duckdb_packaging import build_info as bi  # noqa: E402
from duckdb_packaging._versioning import (  # noqa: E402
    commit_hash,
    commit_time,
    dev_number,
    duckdb_describe_from_override,
    duckdb_dev_version,
    format_version,
    git_tag_to_pep440,
    next_release_version,
    parse_version,
    pep440_to_git_tag,
    release_version,
)
from duckdb_packaging.build_info import (  # noqa: E402
    BuildInfo,
    dynamic_metadata,
    forced_duckdb_version_from_env,
    forced_version_from_env,
    get_requires_for_dynamic_metadata,
    read_build_info,
    resolve_build_info,
    write_build_info_json,
    write_build_info_module,
)
from duckdb_packaging.bump_release_version import bump  # noqa: E402
from duckdb_packaging.bump_release_version import main as bump_main  # noqa: E402

# 2026-09-03 16:53:07 UTC, the example in the pyproject.toml comment
T_EXAMPLE = int(datetime.datetime(2026, 9, 3, 16, 53, 7, tzinfo=datetime.UTC).timestamp())
DAY = 86400

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, env=GIT_ENV, capture_output=True, text=True).stdout


def commit_file(repo: Path, name: str) -> None:
    (repo / name).write_text(name)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", name)


class TestVersionParsing(unittest.TestCase):
    """Test version parsing and formatting functions."""

    def test_parse_version_basic(self):
        """Test parsing basic semantic versions."""
        assert parse_version("1.2.3") == (1, 2, 3, 0, None)
        assert parse_version("0.0.1") == (0, 0, 1, 0, None)
        assert parse_version("10.20.30") == (10, 20, 30, 0, None)

    def test_parse_version_post_release(self):
        """Test parsing post-release versions."""
        assert parse_version("1.2.3.post1") == (1, 2, 3, 1, None)
        assert parse_version("1.2.3.post10") == (1, 2, 3, 10, None)

    def test_parse_version_rc_release(self):
        """Test parsing rc versions."""
        assert parse_version("1.2.3rc1") == (1, 2, 3, 0, ("rc", 1))
        assert parse_version("1.2.3rc10") == (1, 2, 3, 0, ("rc", 10))

    def test_parse_version_alpha_beta_release(self):
        """Test parsing alpha and beta versions."""
        assert parse_version("1.2.3a1") == (1, 2, 3, 0, ("a", 1))
        assert parse_version("1.2.3a10") == (1, 2, 3, 0, ("a", 10))
        assert parse_version("1.2.3b1") == (1, 2, 3, 0, ("b", 1))
        assert parse_version("1.2.3b10") == (1, 2, 3, 0, ("b", 10))

    def test_parse_version_alternative_pre_release_spellings(self):
        """Test the PEP440 alternative pre-release spellings and separators."""
        # alternative spellings normalize to a, b and rc
        assert parse_version("1.2.3alpha1") == (1, 2, 3, 0, ("a", 1))
        assert parse_version("1.2.3beta2") == (1, 2, 3, 0, ("b", 2))
        assert parse_version("1.2.3c3") == (1, 2, 3, 0, ("rc", 3))
        assert parse_version("1.2.3pre4") == (1, 2, 3, 0, ("rc", 4))
        assert parse_version("1.2.3preview5") == (1, 2, 3, 0, ("rc", 5))
        # separators before the signifier and before the numeral are allowed
        assert parse_version("1.2.3-a1") == (1, 2, 3, 0, ("a", 1))
        assert parse_version("1.2.3.b2") == (1, 2, 3, 0, ("b", 2))
        assert parse_version("1.2.3_rc3") == (1, 2, 3, 0, ("rc", 3))
        assert parse_version("1.2.3-alpha.1") == (1, 2, 3, 0, ("a", 1))
        # case insensitive
        assert parse_version("1.2.3RC1") == (1, 2, 3, 0, ("rc", 1))
        assert parse_version("1.2.3Alpha2") == (1, 2, 3, 0, ("a", 2))
        # an omitted numeral means 0
        assert parse_version("1.2.3a") == (1, 2, 3, 0, ("a", 0))
        assert parse_version("1.2.3-alpha") == (1, 2, 3, 0, ("a", 0))

    def test_parse_version_invalid(self):
        """Test parsing invalid version formats."""
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2")
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2.3.4")
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("v1.2.3")
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2.3x1")
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2.3.dev4")
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2.3rc5.post2")

    def test_format_version_basic(self):
        """Test formatting basic semantic versions."""
        assert format_version(1, 2, 3) == "1.2.3"
        assert format_version(0, 0, 1) == "0.0.1"
        assert format_version(10, 20, 30) == "10.20.30"

    def test_format_version_post_release(self):
        """Test formatting post-release versions."""
        assert format_version(1, 2, 3, post=1) == "1.2.3.post1"
        assert format_version(1, 2, 3, post=10) == "1.2.3.post10"

    def test_format_version_pre_release(self):
        """Test formatting pre-release versions."""
        assert format_version(1, 2, 3, pre=("rc", 1)) == "1.2.3rc1"
        assert format_version(1, 2, 3, pre=("rc", 10)) == "1.2.3rc10"
        assert format_version(1, 2, 3, pre=("a", 1)) == "1.2.3a1"
        assert format_version(1, 2, 3, pre=("b", 2)) == "1.2.3b2"

    def test_format_version_post_pre_exclusive(self):
        """Test that post and pre-release are mutually exclusive."""
        with pytest.raises(ValueError, match="post and pre are mutually exclusive"):
            format_version(1, 2, 3, post=1, pre=("rc", 1))

    def test_format_version_invalid_pre_kind(self):
        """Test that invalid pre-release kinds are rejected."""
        with pytest.raises(ValueError, match="Invalid pre-release kind"):
            format_version(1, 2, 3, pre=("alpha", 1))


class TestGitTagConversion(unittest.TestCase):
    """Test git tag to PEP440 conversion and vice versa."""

    def test_git_tag_to_pep440_basic(self):
        """Test basic git tag to PEP440 conversion."""
        assert git_tag_to_pep440("v1.2.3") == "1.2.3"
        assert git_tag_to_pep440("1.2.3") == "1.2.3"

    def test_git_tag_to_pep440_post_release(self):
        """Test post-release git tag to PEP440 conversion."""
        assert git_tag_to_pep440("v1.2.3-post1") == "1.2.3.post1"
        assert git_tag_to_pep440("1.2.3-post10") == "1.2.3.post10"

    def test_git_tag_to_pep440_pre_release(self):
        """Test pre-release git tag to PEP440 conversion."""
        assert git_tag_to_pep440("v1.2.3-a1") == "1.2.3a1"
        assert git_tag_to_pep440("v1.2.3-b2") == "1.2.3b2"
        assert git_tag_to_pep440("v1.2.3-rc10") == "1.2.3rc10"
        # alternative spellings normalize
        assert git_tag_to_pep440("v1.2.3-alpha1") == "1.2.3a1"
        assert git_tag_to_pep440("v1.2.3-beta2") == "1.2.3b2"
        assert git_tag_to_pep440("v1.2.3-pre3") == "1.2.3rc3"

    def test_pep440_to_git_tag_basic(self):
        """Test basic PEP440 to git tag conversion."""
        assert pep440_to_git_tag("1.2.3") == "v1.2.3"

    def test_pep440_to_git_tag_post_release(self):
        """Test post-release PEP440 to git tag conversion."""
        assert pep440_to_git_tag("1.2.3.post1") == "v1.2.3-post1"
        assert pep440_to_git_tag("1.2.3.post10") == "v1.2.3-post10"

    def test_pep440_to_git_tag_pre_release(self):
        """Test pre-release PEP440 to git tag conversion."""
        assert pep440_to_git_tag("1.2.3a1") == "v1.2.3-a1"
        assert pep440_to_git_tag("1.2.3b2") == "v1.2.3-b2"
        assert pep440_to_git_tag("1.2.3rc10") == "v1.2.3-rc10"
        # alternative spellings normalize
        assert pep440_to_git_tag("1.2.3alpha1") == "v1.2.3-a1"
        assert pep440_to_git_tag("1.2.3-beta2") == "v1.2.3-b2"
        assert pep440_to_git_tag("1.2.3preview3") == "v1.2.3-rc3"

    def test_roundtrip_conversion(self):
        """Test that conversions are reversible."""
        versions = ["1.2.3", "1.2.3.post1", "10.20.30.post5", "1.2.3a1", "1.2.3b2", "1.2.3rc3"]
        for version in versions:
            git_tag = pep440_to_git_tag(version)
            converted_back = git_tag_to_pep440(git_tag)
            assert converted_back == version


class TestDuckDBDescribeFromOverride(unittest.TestCase):
    """Test the mapping of a forced package version to what DuckDB gets built with."""

    def test_stable_version(self):
        assert duckdb_describe_from_override("v1.2.3") == "v1.2.3"

    def test_post_version_strips_post(self):
        """Post releases repackage the same engine, so DuckDB never sees the post."""
        assert duckdb_describe_from_override("v1.2.3-post1") == "v1.2.3"
        assert duckdb_describe_from_override("v1.2.3-post10") == "v1.2.3"

    def test_post_version_with_distance_keeps_the_distance(self):
        assert duckdb_describe_from_override("v1.2.3-post1-3-g1234567") == "v1.2.3-3-g1234567"

    def test_pre_release_spelling_is_preserved(self):
        """The whole point: DuckDB accepts -alphaN and rejects the PEP440 -aN."""
        assert duckdb_describe_from_override("v2.0.0-alpha38426") == "v2.0.0-alpha38426"
        assert duckdb_describe_from_override("v1.2.3-rc2") == "v1.2.3-rc2"

    def test_unsupported_spellings_pass_through_untouched(self):
        """We never normalize. DuckDB's CMake is the only authority, and it fails loud."""
        assert duckdb_describe_from_override("v1.2.3-a1") == "v1.2.3-a1"
        assert duckdb_describe_from_override("v1.2.3-b2") == "v1.2.3-b2"
        assert duckdb_describe_from_override("v1.2.3-beta2") == "v1.2.3-beta2"

    def test_full_describe_passes_through(self):
        assert duckdb_describe_from_override("v1.2.3-5-g1234567") == "v1.2.3-5-g1234567"
        assert duckdb_describe_from_override("v2.0.0-alpha1-42-gabc123") == "v2.0.0-alpha1-42-gabc123"


class TestReleaseVersion(unittest.TestCase):
    """release_version reads [tool.duckdb_packaging] from pyproject.toml."""

    def _pyproject(self, body: str) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "pyproject.toml"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def test_reads_the_declared_version(self):
        path = self._pyproject("""
            [tool.duckdb_packaging]
            release_version = "2.0.0"
        """)
        assert release_version(path) == "2.0.0"

    def test_normalizes_alternative_spellings(self):
        path = self._pyproject("""
            [tool.duckdb_packaging]
            release_version = "2.0.0-alpha1"
        """)
        assert release_version(path) == "2.0.0a1"

    def test_post_release(self):
        path = self._pyproject("""
            [tool.duckdb_packaging]
            release_version = "2.0.1.post1"
        """)
        assert release_version(path) == "2.0.1.post1"

    def test_missing_table(self):
        path = self._pyproject("""
            [project]
            name = "duckdb"
        """)
        with pytest.raises(ValueError, match="release_version is missing"):
            release_version(path)

    def test_missing_key(self):
        path = self._pyproject("""
            [tool.duckdb_packaging]
            other = "2.0.0"
        """)
        with pytest.raises(ValueError, match="release_version is missing"):
            release_version(path)

    def test_not_a_string(self):
        path = self._pyproject("""
            [tool.duckdb_packaging]
            release_version = 2
        """)
        with pytest.raises(TypeError, match="must be a string"):
            release_version(path)

    def test_invalid_version(self):
        path = self._pyproject("""
            [tool.duckdb_packaging]
            release_version = "2.0"
        """)
        with pytest.raises(ValueError, match="Invalid version format"):
            release_version(path)

    def test_this_repository_declares_a_valid_version(self):
        parse_version(release_version())


class TestDevNumber(unittest.TestCase):
    """The dev segment is the latest commit time, yymmddhhmm in UTC."""

    def test_formats_in_utc(self):
        assert dev_number(T_EXAMPLE) == 2609031653

    def test_latest_time_wins_regardless_of_order(self):
        assert dev_number(T_EXAMPLE, T_EXAMPLE - DAY) == 2609031653
        assert dev_number(T_EXAMPLE - DAY, T_EXAMPLE) == 2609031653

    def test_seconds_are_dropped(self):
        assert dev_number(T_EXAMPLE + 50) == 2609031653
        assert dev_number(T_EXAMPLE + 53) == 2609031654

    def test_a_later_commit_in_either_repository_yields_a_larger_number(self):
        base = dev_number(T_EXAMPLE, T_EXAMPLE - DAY)
        assert dev_number(T_EXAMPLE + 60, T_EXAMPLE - DAY) > base
        assert dev_number(T_EXAMPLE, T_EXAMPLE + 60) > base

    def test_same_commits_yield_the_same_number(self):
        assert dev_number(T_EXAMPLE, T_EXAMPLE - DAY) == dev_number(T_EXAMPLE, T_EXAMPLE - DAY)

    @unittest.skipUnless(hasattr(time, "tzset"), "needs tzset")
    def test_independent_of_the_local_timezone(self):
        with patch.dict(os.environ, {"TZ": "America/New_York"}):
            time.tzset()
            try:
                assert dev_number(T_EXAMPLE) == 2609031653
            finally:
                del os.environ["TZ"]
                time.tzset()


@unittest.skipUnless(shutil.which("git"), "needs git")
class TestGitPrimitives(unittest.TestCase):
    """commit_hash and commit_time against a real repository."""

    AUTHOR_DATE = "2020-01-01T00:00:00Z"
    COMMITTER_DATE = "2026-09-03T16:53:07Z"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        env = {**GIT_ENV, "GIT_AUTHOR_DATE": self.AUTHOR_DATE, "GIT_COMMITTER_DATE": self.COMMITTER_DATE}
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True, env=env)
        (self.repo / "f").write_text("x")
        subprocess.run(["git", "add", "f"], cwd=self.repo, check=True, env=env)
        subprocess.run(["git", "commit", "-q", "-m", "m"], cwd=self.repo, check=True, env=env)

    def test_commit_time_is_the_committer_time_not_the_author_time(self):
        assert commit_time(self.repo) == T_EXAMPLE
        author = int(datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC).timestamp())
        assert commit_time(self.repo) != author

    def test_dev_number_of_a_real_commit(self):
        assert dev_number(commit_time(self.repo)) == 2609031653

    def test_commit_hash_is_the_full_hash(self):
        expected = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        assert commit_hash(self.repo) == expected
        assert len(expected) == 40

    def test_outside_a_repository(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with (
            patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": tmp.name}),
            pytest.raises(subprocess.CalledProcessError),
        ):
            commit_time(Path(tmp.name))


class TestForcedVersionFromEnv(unittest.TestCase):
    """OVERRIDE_GIT_DESCRIBE forces an exact package version."""

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.0"})
    def test_stable(self):
        assert forced_version_from_env() == "2.0.0"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.0-alpha1"})
    def test_alpha_alternative_spelling(self):
        assert forced_version_from_env() == "2.0.0a1"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.0-rc2"})
    def test_rc(self):
        assert forced_version_from_env() == "2.0.0rc2"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.1-post1"})
    def test_post(self):
        assert forced_version_from_env() == "2.0.1.post1"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": ""})
    def test_empty_means_unset(self):
        assert forced_version_from_env() is None

    def test_unset(self):
        with patch.dict("os.environ"):
            os.environ.pop("OVERRIDE_GIT_DESCRIBE", None)
            assert forced_version_from_env() is None

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.0-5-g1234567"})
    def test_describe_distance_is_rejected(self):
        """A release is an exact version, there is no such thing as a forced dev distance."""
        with pytest.raises(ValueError, match="Invalid OVERRIDE_GIT_DESCRIBE"):
            forced_version_from_env()

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "2.0.0"})
    def test_missing_v_prefix_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid OVERRIDE_GIT_DESCRIBE"):
            forced_version_from_env()

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "invalid-format"})
    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid OVERRIDE_GIT_DESCRIBE"):
            forced_version_from_env()


class TestForcedDuckDBVersionFromEnv(unittest.TestCase):
    """Test which version DuckDB gets built with, per the two env overrides.

    The two overrides are independent. A nightly that vendors a specific DuckDB
    version sets only OVERRIDE_DUCKDB_GIT_DESCRIBE and keeps deriving its own
    package version. A stable release sets OVERRIDE_GIT_DESCRIBE and DuckDB
    inherits it.
    """

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "", "OVERRIDE_DUCKDB_GIT_DESCRIBE": ""})
    def test_neither_set_derives_from_the_submodule(self):
        assert forced_duckdb_version_from_env() is None

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "", "OVERRIDE_DUCKDB_GIT_DESCRIBE": "v2.0.0-alpha38426"})
    def test_duckdb_override_passes_through_verbatim(self):
        """The nightly case. DuckDB accepts -alphaN, so it must survive untouched."""
        assert forced_duckdb_version_from_env() == "v2.0.0-alpha38426"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.5.4", "OVERRIDE_DUCKDB_GIT_DESCRIBE": ""})
    def test_package_override_carries_over(self):
        """The stable release case, one version identifier for both."""
        assert forced_duckdb_version_from_env() == "v1.5.4"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.5.4-post1", "OVERRIDE_DUCKDB_GIT_DESCRIBE": ""})
    def test_package_override_drops_post(self):
        """A post release repackages the same engine."""
        assert forced_duckdb_version_from_env() == "v1.5.4"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.0-alpha1", "OVERRIDE_DUCKDB_GIT_DESCRIBE": ""})
    def test_package_override_keeps_the_alpha_spelling(self):
        """Regression: normalizing this to v2.0.0-a1 makes DuckDB's CMake fail."""
        assert forced_duckdb_version_from_env() == "v2.0.0-alpha1"

    @patch.dict(
        "os.environ",
        {"OVERRIDE_GIT_DESCRIBE": "v1.5.4-post1", "OVERRIDE_DUCKDB_GIT_DESCRIBE": "v2.0.0-alpha38426"},
    )
    def test_duckdb_override_wins(self):
        assert forced_duckdb_version_from_env() == "v2.0.0-alpha38426"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.5.4", "OVERRIDE_DUCKDB_GIT_DESCRIBE": "v2.0.0-alpha38426"})
    def test_package_version_is_untouched_by_the_duckdb_override(self):
        """The two channels do not interfere: forcing DuckDB leaves the package alone."""
        assert forced_version_from_env() == "1.5.4"
        assert forced_duckdb_version_from_env() == "v2.0.0-alpha38426"


PACKAGE_COMMIT = "c" * 40
DUCKDB_COMMIT = "d" * 40
T_PACKAGE = T_EXAMPLE - DAY
T_DUCKDB = T_EXAMPLE


class TestResolveBuildInfo(unittest.TestCase):
    """The build identity in its three situations: git checkout, forced release, sdist."""

    def setUp(self):
        resolve_build_info.cache_clear()
        self.addCleanup(resolve_build_info.cache_clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "duckdb").mkdir()
        self.submodule = self.root / "external" / "duckdb"
        self.json_path = self.root / "build_info.json"
        self.commit_hash = self._patch("commit_hash", side_effect=self._commit_hash)
        self.commit_time = self._patch("commit_time", side_effect=self._commit_time)
        self.duckdb_dev_version = self._patch("duckdb_dev_version", return_value="v2.0.0-dev14120")
        self._patch("duckdb_submodule_path", return_value=self.submodule)
        self.release_version = self._patch("release_version", return_value="2.0.0")
        for name, value in {
            "PROJECT_ROOT": self.root,
            "BUILD_INFO_JSON": self.json_path,
            "BUILD_INFO_MODULE": self.root / "duckdb" / "_build_info.py",
        }.items():
            p = patch.object(bi, name, value)
            p.start()
            self.addCleanup(p.stop)
        env = patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "", "OVERRIDE_DUCKDB_GIT_DESCRIBE": ""})
        env.start()
        self.addCleanup(env.stop)

    def _patch(self, name, **kwargs):
        p = patch.object(bi, name, **kwargs)
        mock = p.start()
        self.addCleanup(p.stop)
        return mock

    def _commit_hash(self, path):
        return PACKAGE_COMMIT if path == self.root else DUCKDB_COMMIT

    def _commit_time(self, path):
        return T_PACKAGE if path == self.root else T_DUCKDB

    def _git_checkout(self):
        (self.root / ".git").mkdir()

    def _sdist(self, stored: BuildInfo):
        write_build_info_json(stored, self.json_path)
        (self.root / "PKG-INFO").write_text(f"Version: {stored.package_version}\n")

    def test_git_checkout_is_a_dev_build(self):
        self._git_checkout()
        assert resolve_build_info() == BuildInfo(
            package_version="2.0.0.dev2609031653",
            package_commit=PACKAGE_COMMIT,
            duckdb_version="v2.0.0-dev14120",
            duckdb_commit=DUCKDB_COMMIT,
        )

    def test_dev_number_follows_the_later_commit_of_the_two(self):
        self._git_checkout()
        assert resolve_build_info().package_version == f"2.0.0.dev{dev_number(T_DUCKDB)}"
        self.commit_time.side_effect = lambda path: T_DUCKDB + DAY if path == self.root else T_DUCKDB
        resolve_build_info.cache_clear()
        assert resolve_build_info().package_version == f"2.0.0.dev{dev_number(T_DUCKDB + DAY)}"

    def test_the_answer_is_cached_within_a_process(self):
        self._git_checkout()
        assert resolve_build_info() is resolve_build_info()
        assert self.commit_hash.call_count == 2

    def test_dev_build_of_a_pre_release_line(self):
        self._git_checkout()
        self.release_version.return_value = "2.0.0a1"
        assert resolve_build_info().package_version == "2.0.0a1.dev2609031653"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.0"})
    def test_forced_release_matching_pyproject(self):
        self._git_checkout()
        assert resolve_build_info() == BuildInfo(
            package_version="2.0.0",
            package_commit=PACKAGE_COMMIT,
            duckdb_version="v2.0.0",
            duckdb_commit=DUCKDB_COMMIT,
        )
        self.duckdb_dev_version.assert_not_called()

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.1"})
    def test_forced_release_that_was_not_bumped_in_pyproject_fails(self):
        self._git_checkout()
        with pytest.raises(ValueError, match=r"release_version = 2\.0\.0"):
            resolve_build_info()

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.0-alpha1"})
    def test_forced_pre_release_keeps_the_duckdb_spelling(self):
        self._git_checkout()
        self.release_version.return_value = "2.0.0a1"
        info = resolve_build_info()
        assert info.package_version == "2.0.0a1"
        assert info.duckdb_version == "v2.0.0-alpha1"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.1-post1"})
    def test_forced_post_release_repackages_the_same_engine(self):
        self._git_checkout()
        self.release_version.return_value = "2.0.1.post1"
        info = resolve_build_info()
        assert info.package_version == "2.0.1.post1"
        assert info.duckdb_version == "v2.0.1"

    @patch.dict("os.environ", {"OVERRIDE_DUCKDB_GIT_DESCRIBE": "v2.0.0-alpha40145"})
    def test_forced_duckdb_version_leaves_the_package_a_dev_build(self):
        self._git_checkout()
        info = resolve_build_info()
        assert info.package_version == "2.0.0.dev2609031653"
        assert info.duckdb_version == "v2.0.0-alpha40145"
        assert info.duckdb_commit == DUCKDB_COMMIT
        self.duckdb_dev_version.assert_not_called()

    def test_sdist_uses_the_stored_identity_without_git(self):
        stored = BuildInfo("2.0.0.dev2609011200", "a" * 40, "v2.0.0-dev14000", "b" * 40)
        self._sdist(stored)
        assert resolve_build_info() == stored
        self.commit_hash.assert_not_called()
        self.commit_time.assert_not_called()
        self.duckdb_dev_version.assert_not_called()

    @patch.dict("os.environ", {"OVERRIDE_DUCKDB_GIT_DESCRIBE": "v2.0.0-alpha40145"})
    def test_sdist_accepts_the_duckdb_override_it_was_built_with(self):
        stored = BuildInfo("2.0.0.dev2609011200", "a" * 40, "v2.0.0-alpha40145", "b" * 40)
        self._sdist(stored)
        assert resolve_build_info() == stored

    @patch.dict("os.environ", {"OVERRIDE_DUCKDB_GIT_DESCRIBE": "v2.0.0-alpha40145"})
    def test_sdist_rejects_a_different_duckdb_override(self):
        """An sdist compiles one fixed engine source; a label for another engine would lie about it."""
        stored = BuildInfo("2.0.0.dev2609011200", "a" * 40, "v2.0.0-dev14000", "b" * 40)
        self._sdist(stored)
        with pytest.raises(ValueError, match=r"built with DuckDB v2\.0\.0-dev14000 at b"):
            resolve_build_info()

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.0"})
    def test_sdist_accepts_the_package_override_it_was_built_with(self):
        stored = BuildInfo("2.0.0", "a" * 40, "v2.0.0", "b" * 40)
        self._sdist(stored)
        assert resolve_build_info() == stored

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v2.0.0"})
    def test_sdist_rejects_a_different_package_override(self):
        stored = BuildInfo("2.0.0.dev2609011200", "a" * 40, "v2.0.0-dev14000", "b" * 40)
        self._sdist(stored)
        with pytest.raises(ValueError, match=r"built as 2\.0\.0\.dev2609011200"):
            resolve_build_info()

    def test_a_stored_identity_needs_pkg_info_to_count_as_an_sdist(self):
        """A stale build_info.json in a git checkout must not shadow git."""
        stored = BuildInfo("2.0.0.dev2609011200", "a" * 40, "v2.0.0-dev14000", "b" * 40)
        write_build_info_json(stored, self.json_path)
        self._git_checkout()
        assert resolve_build_info().package_commit == PACKAGE_COMMIT

    def test_neither_checkout_nor_sdist(self):
        with pytest.raises(RuntimeError, match="Not in a git repository nor in an sdist"):
            resolve_build_info()


class TestBuildInfoFiles(unittest.TestCase):
    INFO = BuildInfo("2.0.0.dev2609031653", PACKAGE_COMMIT, "v2.0.0-dev14120", DUCKDB_COMMIT)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_json_round_trip(self):
        path = self.dir / "build_info.json"
        write_build_info_json(self.INFO, path)
        assert read_build_info(path) == self.INFO

    def test_generated_module_exposes_the_package_commit(self):
        path = self.dir / "_build_info.py"
        write_build_info_module(self.INFO, path)
        namespace: dict = {}
        exec(path.read_text(encoding="utf-8"), namespace)
        assert namespace["PACKAGE_COMMIT"] == PACKAGE_COMMIT


class TestScikitBuildProvider(unittest.TestCase):
    INFO = BuildInfo("2.0.0.dev2609031653", PACKAGE_COMMIT, "v2.0.0-dev14120", DUCKDB_COMMIT)

    def test_version_field(self):
        with patch.object(bi, "resolve_build_info", return_value=self.INFO):
            assert dynamic_metadata("version") == "2.0.0.dev2609031653"
            assert dynamic_metadata("version", {}) == "2.0.0.dev2609031653"

    def test_other_fields_are_rejected(self):
        with pytest.raises(ValueError, match="Only the 'version' field"):
            dynamic_metadata("description")

    def test_inline_settings_are_rejected(self):
        with pytest.raises(ValueError, match="No inline configuration"):
            dynamic_metadata("version", {"x": 1})

    def test_no_extra_requirements(self):
        assert get_requires_for_dynamic_metadata() == []


@unittest.skipUnless(bi.is_git_checkout() and shutil.which("git"), "needs this repository as a git checkout")
class TestThisCheckout(unittest.TestCase):
    """The real thing: this checkout resolves to a dev build of the declared version."""

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "", "OVERRIDE_DUCKDB_GIT_DESCRIBE": ""})
    def test_resolves(self):
        resolve_build_info.cache_clear()
        self.addCleanup(resolve_build_info.cache_clear)
        info = resolve_build_info()
        expected = dev_number(commit_time(bi.PROJECT_ROOT), commit_time(bi.duckdb_submodule_path()))
        assert info.package_version == f"{release_version()}.dev{expected}"
        assert len(info.package_commit) == 40
        assert len(info.duckdb_commit) == 40
        assert info.duckdb_version.startswith("v")


@unittest.skipUnless(shutil.which("git"), "needs git")
class TestDuckDBSubmodulePath(unittest.TestCase):
    """duckdb_submodule_path against a real checkout with a real submodule."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        # the parser keys on the url's basename, which has to be duckdb
        core = base / "duckdb"
        git(base, "init", "-q", "duckdb")
        commit_file(core, "one")
        self.client = base / "client"
        git(base, "init", "-q", "client")
        commit_file(self.client, "init")
        git(self.client, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(core), "external/duckdb")
        git(self.client, "commit", "-q", "-m", "add submodule")
        p = patch.object(bi, "PROJECT_ROOT", self.client)
        p.start()
        self.addCleanup(p.stop)

    def test_clean_submodule(self):
        assert bi.duckdb_submodule_path() == self.client / "external" / "duckdb"

    def test_missing_gitmodules(self):
        (self.client / ".gitmodules").unlink()
        with pytest.raises(RuntimeError, match="submodule missing"):
            bi.duckdb_submodule_path()

    def test_gitmodules_without_duckdb(self):
        (self.client / ".gitmodules").write_text('[submodule "other"]\n\tpath = external/other\n\turl = ../other.git\n')
        with pytest.raises(RuntimeError, match="submodule missing"):
            bi.duckdb_submodule_path()

    def test_uninitialized_submodule(self):
        git(self.client, "submodule", "deinit", "-f", "-q", "external/duckdb")
        with pytest.raises(RuntimeError, match="not initialized"):
            bi.duckdb_submodule_path()

    def test_submodule_at_another_commit_warns_and_passes(self):
        commit_file(self.client / "external" / "duckdb", "two")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            assert bi.duckdb_submodule_path() == self.client / "external" / "duckdb"
        assert "not clean" in stderr.getvalue()

    def test_not_a_checkout(self):
        with (
            patch.object(bi, "PROJECT_ROOT", self.client / "external"),
            pytest.raises(RuntimeError, match="Not in a git"),
        ):
            bi.duckdb_submodule_path()


class TestDuckDBDevVersion(unittest.TestCase):
    """duckdb_dev_version runs DuckDB's own script, with the version overrides kept away from it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.submodule = Path(tmp.name)
        script = self.submodule / "scripts" / "ci" / "version.py"
        script.parent.mkdir(parents=True)
        script.write_text(
            "import os\n"
            "print('v9.9.9-dev1', os.environ.get('OVERRIDE_GIT_DESCRIBE', 'unset'), "
            "os.environ.get('DUCKDB_VERSION', 'unset'), os.environ.get('DUCKDB_COMMIT', 'unset'), "
            "os.environ.get('KEPT', 'unset'))\n"
        )

    @patch.dict(
        "os.environ",
        {"OVERRIDE_GIT_DESCRIBE": "v2.0.0", "DUCKDB_VERSION": "v3.0.0", "DUCKDB_COMMIT": "abc", "KEPT": "yes"},
    )
    def test_runs_the_script_without_the_version_overrides(self):
        assert duckdb_dev_version(self.submodule) == "v9.9.9-dev1 unset unset unset yes"

    def test_a_failing_script_fails_loud(self):
        (self.submodule / "scripts" / "ci" / "version.py").write_text("raise SystemExit(3)\n")
        with pytest.raises(subprocess.CalledProcessError):
            duckdb_dev_version(self.submodule)


class TestPackageCommitAtRuntime(unittest.TestCase):
    """duckdb.build_info() degrades to unknown without a usable generated module."""

    def setUp(self):
        self.version_module = pytest.importorskip("duckdb._version")

    def test_missing_module(self):
        with patch.dict(sys.modules, {"duckdb._build_info": None}):
            assert self.version_module._package_commit() == "unknown"

    def test_module_without_the_symbol(self):
        with patch.dict(sys.modules, {"duckdb._build_info": types.ModuleType("duckdb._build_info")}):
            assert self.version_module._package_commit() == "unknown"

    def test_generated_module(self):
        generated = types.ModuleType("duckdb._build_info")
        generated.PACKAGE_COMMIT = PACKAGE_COMMIT
        with patch.dict(sys.modules, {"duckdb._build_info": generated}):
            assert self.version_module._package_commit() == PACKAGE_COMMIT


class TestNextReleaseVersion(unittest.TestCase):
    """What a branch moves to after a release."""

    def test_final_moves_to_the_next_patch(self):
        assert next_release_version("2.0.0") == "2.0.1"
        assert next_release_version("2.0.9") == "2.0.10"

    def test_post_release_moves_to_the_next_patch(self):
        assert next_release_version("2.0.1.post1") == "2.0.2"

    def test_pre_release_moves_to_its_next_number(self):
        assert next_release_version("2.0.0a1") == "2.0.0a2"
        assert next_release_version("2.0.0b2") == "2.0.0b3"
        assert next_release_version("2.0.0rc1") == "2.0.0rc2"

    def test_alternative_spelling_is_normalized(self):
        assert next_release_version("2.0.0-alpha1") == "2.0.0a2"

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid version format"):
            next_release_version("2.0")


class TestBumpReleaseVersion(unittest.TestCase):
    """The rewrite lands on the one line and nothing else."""

    PYPROJECT = textwrap.dedent("""
        [project]
        name = "duckdb"
        dynamic = ["version"]

        [tool.duckdb_packaging]
        # a comment that mentions release_version = "9.9.9" and stays
        release_version = "2.0.0"

        [tool.other]
        release_version = "1.0.0"
    """)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "pyproject.toml"
        self.path.write_text(self.PYPROJECT, encoding="utf-8")

    def test_bumps_the_declared_version_only(self):
        assert bump(self.path) == ("2.0.0", "2.0.1")
        assert release_version(self.path) == "2.0.1"
        text = self.path.read_text(encoding="utf-8")
        assert 'mentions release_version = "9.9.9"' in text
        assert text.count('release_version = "2.0.1"') == 1
        assert text.replace('release_version = "2.0.1"', 'release_version = "2.0.0"') == self.PYPROJECT

    def test_bumps_a_pre_release(self):
        self.path.write_text(self.PYPROJECT.replace('"2.0.0"', '"2.0.0a1"'), encoding="utf-8")
        assert bump(self.path) == ("2.0.0a1", "2.0.0a2")

    def test_missing_setting(self):
        self.path.write_text('[project]\nname = "duckdb"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="release_version is missing"):
            bump(self.path)

    def test_rewrite_that_misses_the_declared_line_fails(self):
        """A release_version line in another table must not be mistaken for ours."""
        self.path.write_text(
            '[tool.other]\nrelease_version = "1.0.0"\n\n[tool.duckdb_packaging]\nrelease_version = "2.0.0"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Rewrote the wrong line"):
            bump(self.path)
        assert release_version(self.path) == "2.0.0"

    def test_main_prints_the_move(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            assert bump_main(["--pyproject", str(self.path)]) == 0
        assert stdout.getvalue().strip() == "2.0.0 -> 2.0.1"

"""Tests for duckdb_pytooling versioning functionality."""

import os
import subprocess
import unittest
from unittest.mock import MagicMock, patch

import pytest

duckdb_packaging = pytest.importorskip("duckdb_packaging")

from duckdb_packaging._versioning import (  # noqa: E402
    format_version,
    get_current_version,
    get_git_describe,
    git_tag_to_pep440,
    parse_version,
    pep440_to_git_tag,
)
from duckdb_packaging.setuptools_scm_version import (  # noqa: E402
    _bump_dev_version,
    _tag_to_version,
    forced_version_from_env,
    version_scheme,
)


class TestVersionParsing(unittest.TestCase):
    """Test version parsing and formatting functions."""

    def test_parse_version_basic(self):
        """Test parsing basic semantic versions."""
        assert parse_version("1.2.3") == (1, 2, 3, 0, 0)
        assert parse_version("0.0.1") == (0, 0, 1, 0, 0)
        assert parse_version("10.20.30") == (10, 20, 30, 0, 0)

    def test_parse_version_post_release(self):
        """Test parsing post-release versions."""
        assert parse_version("1.2.3.post1") == (1, 2, 3, 1, 0)
        assert parse_version("1.2.3.post10") == (1, 2, 3, 10, 0)

    def test_parse_version_rc_release(self):
        """Test parsing post-release versions."""
        assert parse_version("1.2.3rc1") == (1, 2, 3, 0, 1)
        assert parse_version("1.2.3rc10") == (1, 2, 3, 0, 10)

    def test_parse_version_invalid(self):
        """Test parsing invalid version formats."""
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2")
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2.3.4")
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("v1.2.3")
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2.3-alpha")
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

    def test_format_version_rc_release(self):
        """Test formatting post-release versions."""
        assert format_version(1, 2, 3, rc=1) == "1.2.3rc1"
        assert format_version(1, 2, 3, rc=10) == "1.2.3rc10"


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

    def test_pep440_to_git_tag_basic(self):
        """Test basic PEP440 to git tag conversion."""
        assert pep440_to_git_tag("1.2.3") == "v1.2.3"

    def test_pep440_to_git_tag_post_release(self):
        """Test post-release PEP440 to git tag conversion."""
        assert pep440_to_git_tag("1.2.3.post1") == "v1.2.3-post1"
        assert pep440_to_git_tag("1.2.3.post10") == "v1.2.3-post10"

    def test_roundtrip_conversion(self):
        """Test that conversions are reversible."""
        versions = ["1.2.3", "1.2.3.post1", "10.20.30.post5"]
        for version in versions:
            git_tag = pep440_to_git_tag(version)
            converted_back = git_tag_to_pep440(git_tag)
            assert converted_back == version


class TestSetupToolsScmIntegration(unittest.TestCase):
    """Test setuptools_scm integration functions."""

    def test_bump_version_exact_tag(self):
        """Test bump_version with exact tag (distance=0, dirty=False)."""
        assert _tag_to_version("1.2.3") == "1.2.3"
        assert _tag_to_version("1.2.3.post1") == "1.2.3.post1"

    @patch.dict("os.environ", {"MAIN_BRANCH_VERSIONING": "1"})
    def test_bump_version_with_distance(self):
        """Test bump_version with distance from tag."""
        assert _bump_dev_version("1.2.3", 5) == "1.3.0.dev5"

        # Post-release development
        assert _bump_dev_version("1.2.3.post1", 3) == "1.2.3.post2.dev3"

    @patch.dict("os.environ", {"MAIN_BRANCH_VERSIONING": "1", "DUCKDB_DISTANCE": "123"})
    def test_bump_version_with_duckdb_distance(self):
        """Test bump_version with DUCKDB_DISTANCE env var set."""
        assert _bump_dev_version("1.2.3", 5) == "1.3.0.dev000005000123"

        # Post-release development
        assert _bump_dev_version("1.2.3.post1", 3) == "1.2.3.post2.dev000003000123"

    @patch.dict("os.environ", {"MAIN_BRANCH_VERSIONING": "0"})
    def test_bump_version_release_branch(self):
        """Test bump_version on bugfix branch."""
        assert _bump_dev_version("1.2.3", 5) == "1.2.4.dev5"

    @patch.dict("os.environ", {"MAIN_BRANCH_VERSIONING": "1"})
    def test_bump_version_dirty(self):
        """Test bump_version with dirty working directory."""
        with pytest.raises(ValueError, match="Dev distance is 0, cannot bump version"):
            _bump_dev_version("1.2.3", 0)

    @patch.dict("os.environ", {"MAIN_BRANCH_VERSIONING": "1"})
    def test_version_scheme_function(self):
        """Test the version_scheme function that setuptools_scm calls."""
        # Mock setuptools_scm version object
        mock_version = MagicMock()
        mock_version.tag = "1.2.3"
        mock_version.distance = 5
        mock_version.dirty = False

        result = version_scheme(mock_version)
        assert result == "1.3.0.dev5"

    def test_bump_version_invalid_format(self):
        """Test bump_version with invalid version format."""
        with pytest.raises(ValueError, match="Invalid version format"):
            _tag_to_version("invalid")
        with pytest.raises(ValueError, match="Invalid version format"):
            _bump_dev_version("invalid", 1)


class TestGitOperations(unittest.TestCase):
    """Test git-related operations (mocked)."""

    @patch("subprocess.run")
    def test_get_current_version_success(self, mock_run):
        """Test successful current version retrieval."""
        mock_run.return_value.stdout = "v1.2.3\n"
        mock_run.return_value.check = True

        result = get_current_version()
        assert result == "1.2.3"
        mock_run.assert_called_once_with(
            ["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True, check=True
        )

    @patch("subprocess.run")
    def test_get_current_version_with_post_release(self, mock_run):
        """Test current version retrieval with post-release tag."""
        mock_run.return_value.stdout = "v1.2.3-post1\n"
        mock_run.return_value.check = True

        result = get_current_version()
        assert result == "1.2.3.post1"

    @patch("subprocess.run")
    def test_get_current_version_no_tags(self, mock_run):
        """Test current version retrieval when no tags exist."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")

        result = get_current_version()
        assert result is None

    @patch("subprocess.run")
    def test_get_git_describe_success(self, mock_run):
        """Test successful git describe."""
        mock_run.return_value.stdout = "v1.2.3-5-g1234567\n"
        mock_run.return_value.check = True

        result = get_git_describe()
        assert result == "v1.2.3-5-g1234567"

    @patch("subprocess.run")
    def test_get_git_describe_no_tags(self, mock_run):
        """Test git describe when no tags exist."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")

        with pytest.raises(subprocess.CalledProcessError, match="exit status 1"):
            get_git_describe()


class TestEnvironmentVariableHandling(unittest.TestCase):
    """Test environment variable handling in setuptools_scm integration."""

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.2.3-5-g1234567"})
    def test_override_git_describe_basic(self):
        """Test OVERRIDE_GIT_DESCRIBE with basic format."""
        forced_version_from_env()
        # Check that the environment variable was processed
        assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DUCKDB" in os.environ

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.2.3-post1-3-g1234567"})
    def test_override_git_describe_post_release(self):
        """Test OVERRIDE_GIT_DESCRIBE with post-release format."""
        forced_version_from_env()
        # Check that post-release was converted correctly
        assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DUCKDB" in os.environ

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "invalid-format"})
    def test_override_git_describe_invalid(self):
        """Test OVERRIDE_GIT_DESCRIBE with invalid format."""
        with pytest.raises(ValueError, match="Invalid git describe override"):
            forced_version_from_env()

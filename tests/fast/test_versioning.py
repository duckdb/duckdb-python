"""Tests for duckdb_pytooling versioning functionality."""

import os
import subprocess
import unittest
from unittest.mock import MagicMock, patch

import pytest

duckdb_packaging = pytest.importorskip("duckdb_packaging")

from duckdb_packaging._versioning import (  # noqa: E402
    duckdb_tag_from_pep440,
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


class TestDuckDBTagFromPep440(unittest.TestCase):
    """Test the mapping of forced package versions to DuckDB version tags."""

    def test_stable_version(self):
        assert duckdb_tag_from_pep440("1.2.3") == "v1.2.3"

    def test_post_version_strips_post(self):
        """Post releases repackage the stable engine."""
        assert duckdb_tag_from_pep440("1.2.3.post1") == "v1.2.3"

    def test_pre_release_passes_through(self):
        """Pre-releases pass through normalized, DuckDB's build validates them."""
        assert duckdb_tag_from_pep440("1.2.3rc2") == "v1.2.3-rc2"
        assert duckdb_tag_from_pep440("1.2.3a1") == "v1.2.3-a1"
        assert duckdb_tag_from_pep440("1.2.3b2") == "v1.2.3-b2"
        # alternative spellings normalize
        assert duckdb_tag_from_pep440("1.2.3alpha1") == "v1.2.3-a1"
        assert duckdb_tag_from_pep440("1.2.3pre1") == "v1.2.3-rc1"


class TestSetupToolsScmIntegration(unittest.TestCase):
    """Test setuptools_scm integration functions."""

    def test_bump_version_exact_tag(self):
        """Test bump_version with exact tag (distance=0, dirty=False)."""
        assert _tag_to_version("1.2.3") == "1.2.3"
        assert _tag_to_version("1.2.3.post1") == "1.2.3.post1"
        assert _tag_to_version("1.2.3a1") == "1.2.3a1"
        assert _tag_to_version("1.2.3b2") == "1.2.3b2"
        assert _tag_to_version("1.2.3rc3") == "1.2.3rc3"
        # alternative spellings normalize
        assert _tag_to_version("1.2.3-alpha1") == "1.2.3a1"

    @patch.dict("os.environ", {"MAIN_BRANCH_VERSIONING": "1"})
    def test_bump_version_with_distance(self):
        """Test bump_version with distance from tag."""
        assert _bump_dev_version("1.2.3", 5) == "1.3.0.dev5"

        # Post-release development
        assert _bump_dev_version("1.2.3.post1", 3) == "1.2.3.post2.dev3"

    @patch.dict("os.environ", {"MAIN_BRANCH_VERSIONING": "1"})
    def test_bump_version_pre_release(self):
        """Test bump_version on top of a pre-release tag bumps within the same phase."""
        assert _bump_dev_version("1.2.3a1", 4) == "1.2.3a2.dev4"
        assert _bump_dev_version("1.2.3b1", 4) == "1.2.3b2.dev4"
        assert _bump_dev_version("1.2.3rc1", 4) == "1.2.3rc2.dev4"

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

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.2.3-5-g1234567", "MAIN_BRANCH_VERSIONING": "0"})
    def test_override_git_describe_basic(self):
        """Test OVERRIDE_GIT_DESCRIBE with basic format."""
        assert forced_version_from_env() == "1.2.4.dev5+g1234567"
        assert os.environ["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DUCKDB"] == "1.2.4.dev5+g1234567"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.2.3-post1-3-g1234567", "MAIN_BRANCH_VERSIONING": "0"})
    def test_override_git_describe_post_release(self):
        """Test OVERRIDE_GIT_DESCRIBE with post-release format."""
        assert forced_version_from_env() == "1.2.3.post2.dev3+g1234567"
        assert os.environ["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DUCKDB"] == "1.2.3.post2.dev3+g1234567"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.2.3-rc1"})
    def test_override_git_describe_rc_release(self):
        """Test OVERRIDE_GIT_DESCRIBE with an rc tag."""
        assert forced_version_from_env() == "1.2.3rc1"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.2.3-a1"})
    def test_override_git_describe_alpha_release(self):
        """Test OVERRIDE_GIT_DESCRIBE with an alpha tag."""
        assert forced_version_from_env() == "1.2.3a1"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.2.3-alpha1"})
    def test_override_git_describe_alpha_alternative_spelling(self):
        """Test OVERRIDE_GIT_DESCRIBE with an alternatively spelled alpha tag."""
        assert forced_version_from_env() == "1.2.3a1"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "v1.2.3-b1-5-g1234567", "MAIN_BRANCH_VERSIONING": "0"})
    def test_override_git_describe_beta_with_distance(self):
        """Test OVERRIDE_GIT_DESCRIBE with a beta tag and distance."""
        assert forced_version_from_env() == "1.2.3b2.dev5+g1234567"

    @patch.dict("os.environ", {"OVERRIDE_GIT_DESCRIBE": "invalid-format"})
    def test_override_git_describe_invalid(self):
        """Test OVERRIDE_GIT_DESCRIBE with invalid format."""
        with pytest.raises(ValueError, match="Invalid git describe override"):
            forced_version_from_env()

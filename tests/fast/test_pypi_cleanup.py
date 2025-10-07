#!/usr/bin/env python3
"""Unit tests for pypi_cleanup.py.

Run with: python -m pytest test_pypi_cleanup.py -v
"""

import logging
import os
from unittest.mock import Mock, patch

import pytest
import requests
from urllib3 import Retry

duckdb_packaging = pytest.importorskip("duckdb_packaging")

from duckdb_packaging.pypi_cleanup import (  # noqa: E402
    AuthenticationError,
    CleanMode,
    CsrfParser,
    PyPICleanup,
    PyPICleanupError,
    ValidationError,
    create_argument_parser,
    load_credentials,
    main,
    session_with_retries,
    setup_logging,
    validate_arguments,
    validate_username,
)


class TestValidation:
    """Test input validation functions."""

    def test_validate_username_valid(self):
        """Test valid usernames."""
        assert validate_username("user123") == "user123"
        assert validate_username("  user.name  ") == "user.name"
        assert validate_username("test-user_name") == "test-user_name"
        assert validate_username("a") == "a"

    def test_validate_username_invalid(self):
        """Test invalid usernames."""
        from argparse import ArgumentTypeError

        with pytest.raises(ArgumentTypeError, match="cannot be empty"):
            validate_username("")

        with pytest.raises(ArgumentTypeError, match="cannot be empty"):
            validate_username("   ")

        with pytest.raises(ArgumentTypeError, match="too long"):
            validate_username("a" * 101)

        with pytest.raises(ArgumentTypeError, match="Invalid username format"):
            validate_username("-invalid")

        with pytest.raises(ArgumentTypeError, match="Invalid username format"):
            validate_username("invalid-")

    def test_validate_arguments_dry_run(self):
        """Test argument validation for dry run mode."""
        args = Mock(dry_run=True, username=None, max_nightlies=2)
        validate_arguments(args)  # Should not raise

    def test_validate_arguments_live_mode_no_username(self):
        """Test argument validation for live mode without username."""
        args = Mock(dry_run=False, username=None, max_nightlies=2)
        with pytest.raises(ValidationError, match="username is required"):
            validate_arguments(args)

    def test_validate_arguments_negative_nightlies(self):
        """Test argument validation with negative max nightlies."""
        args = Mock(dry_run=True, username="test", max_nightlies=-1)
        with pytest.raises(ValidationError, match="must be non-negative"):
            validate_arguments(args)


class TestCredentials:
    """Test credential loading."""

    @patch.dict(os.environ, {"PYPI_CLEANUP_PASSWORD": "test_pass", "PYPI_CLEANUP_OTP": "test_otp"})
    def test_load_credentials_live_mode_success(self):
        """Test successful credential loading in live mode."""
        password, otp = load_credentials()
        assert password == "test_pass"
        assert otp == "test_otp"

    @patch.dict(os.environ, {}, clear=True)
    def test_load_credentials_missing_password(self):
        """Test credential loading with missing password."""
        with pytest.raises(ValidationError, match="PYPI_CLEANUP_PASSWORD"):
            load_credentials()

    @patch.dict(os.environ, {"PYPI_CLEANUP_PASSWORD": "test_pass"})
    def test_load_credentials_missing_otp(self):
        """Test credential loading with missing OTP."""
        with pytest.raises(ValidationError, match="PYPI_CLEANUP_OTP"):
            load_credentials()


class TestUtilities:
    """Test utility functions."""

    def test_create_session_with_retries(self):
        """Test session creation with retry configuration."""
        with session_with_retries() as session:
            assert isinstance(session, requests.Session)
            # Verify retry adapter is mounted
            adapter = session.get_adapter("https://example.com")
            assert hasattr(adapter, "max_retries")
            retries = adapter.max_retries
            assert isinstance(retries, Retry)

    @patch("duckdb_packaging.pypi_cleanup.logging.basicConfig")
    def test_setup_logging_normal(self, mock_basicConfig):
        """Test logging setup in normal mode."""
        setup_logging()
        mock_basicConfig.assert_called_once()
        call_args = mock_basicConfig.call_args[1]
        assert call_args["level"] == 20  # INFO level

    @patch("duckdb_packaging.pypi_cleanup.logging.basicConfig")
    def test_setup_logging_verbose(self, mock_basicConfig):
        """Test logging setup in verbose mode."""
        setup_logging(level=logging.DEBUG)
        mock_basicConfig.assert_called_once()
        call_args = mock_basicConfig.call_args[1]
        assert call_args["level"] == 10  # DEBUG level


class TestCsrfParser:
    """Test CSRF token parser."""

    def test_csrf_parser_simple_form(self):
        """Test parsing CSRF token from simple form."""
        html = """
        <form action="/test">
            <input name="csrf_token" value="abc123">
            <input name="username" value="">
        </form>
        """
        parser = CsrfParser("/test")
        parser.feed(html)
        assert parser.csrf == "abc123"

    def test_csrf_parser_multiple_forms(self):
        """Test parsing CSRF token when multiple forms exist."""
        html = """
        <form action="/other">
            <input name="csrf_token" value="wrong">
        </form>
        <form action="/test">
            <input name="csrf_token" value="correct">
        </form>
        """
        parser = CsrfParser("/test")
        parser.feed(html)
        assert parser.csrf == "correct"

    def test_csrf_parser_no_token(self):
        """Test parser when no CSRF token is found."""
        html = '<form action="/test"><input name="username" value=""></form>'
        parser = CsrfParser("/test")
        parser.feed(html)
        assert parser.csrf is None


class TestPyPICleanup:
    """Test the main PyPICleanup class."""

    @pytest.fixture
    def cleanup_dryrun_max_2(self) -> PyPICleanup:
        return PyPICleanup("https://test.pypi.org/", CleanMode.LIST_ONLY, 2)

    @pytest.fixture
    def cleanup_dryrun_max_0(self) -> PyPICleanup:
        return PyPICleanup("https://test.pypi.org/", CleanMode.LIST_ONLY, 0)

    @pytest.fixture
    def cleanup_max_2(self) -> PyPICleanup:
        return PyPICleanup(
            "https://test.pypi.org/", CleanMode.DELETE, 2, username="<USERNAME>", password="<PASSWORD>", otp="<OTP>"
        )

    def test_determine_versions_to_delete_max_2(self, cleanup_dryrun_max_2):
        start_state = {
            "0.1.0",
            "1.0.0.dev1",
            "1.0.0.dev2",
            "1.0.0.rc1",
            "1.0.0",
            "1.0.1.dev3",
            "1.0.1.dev5",
            "1.0.1.dev8",
            "1.0.1.dev13",
            "1.0.1.dev21",
            "1.0.1",
            "1.1.0.dev34",
            "1.1.0.dev54",
            "1.1.0.dev88",
            "1.1.0",
            "1.1.0.post1",
            "1.1.1.dev142",
            "1.1.1.dev230",
            "1.1.1.dev372",
            "2.0.0.dev602",
            "2.0.0.rc1",
            "2.0.0.rc2",
            "2.0.0.rc3",
            "2.0.0.rc4",
            "2.0.0",
            "2.0.1.dev974",
            "2.0.1.rc1",
            "2.0.1.rc2",
            "2.0.1.rc3",
        }
        expected_deletions = {
            "1.0.0.dev1",
            "1.0.0.dev2",
            "1.0.0.rc1",
            "1.0.1.dev3",
            "1.0.1.dev5",
            "1.0.1.dev8",
            "1.0.1.dev13",
            "1.0.1.dev21",
            "1.1.0.dev34",
            "1.1.0.dev54",
            "1.1.0.dev88",
            "1.1.1.dev142",
            "2.0.0.dev602",
            "2.0.0.rc1",
            "2.0.0.rc2",
            "2.0.0.rc3",
            "2.0.0.rc4",
            "2.0.1.dev974",
        }
        versions_to_delete = cleanup_dryrun_max_2._determine_versions_to_delete(start_state)
        assert versions_to_delete == expected_deletions

    def test_determine_versions_to_delete_max_0(self, cleanup_dryrun_max_0):
        start_state = {
            "0.1.0",
            "1.0.0.dev1",
            "1.0.0.dev2",
            "1.0.0.rc1",
            "1.0.0",
            "1.0.1.dev3",
            "1.0.1.dev5",
            "1.0.1.dev8",
            "1.0.1.dev13",
            "1.0.1.dev21",
            "1.0.1",
            "1.1.0.dev34",
            "1.1.0.dev54",
            "1.1.0.dev88",
            "1.1.0",
            "1.1.0.post1",
            "1.1.1.dev142",
            "1.1.1.dev230",
            "1.1.1.dev372",
            "2.0.0.dev602",
            "2.0.0.rc1",
            "2.0.0.rc2",
            "2.0.0.rc3",
            "2.0.0.rc4",
            "2.0.0",
            "2.0.1.dev974",
            "2.0.1.rc1",
            "2.0.1.rc2",
            "2.0.1.rc3",
        }
        expected_deletions = {
            "1.0.0.dev1",
            "1.0.0.dev2",
            "1.0.0.rc1",
            "1.0.1.dev3",
            "1.0.1.dev5",
            "1.0.1.dev8",
            "1.0.1.dev13",
            "1.0.1.dev21",
            "1.1.0.dev34",
            "1.1.0.dev54",
            "1.1.0.dev88",
            "1.1.1.dev142",
            "1.1.1.dev230",
            "1.1.1.dev372",
            "2.0.0.dev602",
            "2.0.0.rc1",
            "2.0.0.rc2",
            "2.0.0.rc3",
            "2.0.0.rc4",
            "2.0.1.dev974",
        }
        versions_to_delete = cleanup_dryrun_max_0._determine_versions_to_delete(start_state)
        assert versions_to_delete == expected_deletions

    def test_determine_versions_to_delete_only_devs_max_2(self, cleanup_dryrun_max_2):
        start_state = {
            "1.0.0.dev1",
            "1.0.0.dev2",
            "1.0.1.dev3",
            "1.0.1.dev5",
            "1.0.1.dev8",
            "1.0.1.dev13",
            "1.0.1.dev21",
            "1.1.0.dev34",
            "1.1.0.dev54",
            "1.1.0.dev88",
            "1.1.1.dev142",
            "1.1.1.dev230",
            "1.1.1.dev372",
            "2.0.0.dev602",
            "2.0.1.dev974",
        }
        expected_deletions = {
            "1.0.1.dev3",
            "1.0.1.dev5",
            "1.0.1.dev8",
            "1.1.0.dev34",
            "1.1.1.dev142",
        }
        versions_to_delete = cleanup_dryrun_max_2._determine_versions_to_delete(start_state)
        assert versions_to_delete == expected_deletions

    def test_determine_versions_to_delete_only_devs_max_0_fails(self, cleanup_dryrun_max_0):
        start_state = {
            "1.0.0.dev1",
            "1.0.0.dev2",
            "1.0.1.dev3",
            "1.0.1.dev5",
            "1.0.1.dev8",
            "1.0.1.dev13",
            "1.0.1.dev21",
            "1.1.0.dev34",
            "1.1.0.dev54",
            "1.1.0.dev88",
            "1.1.1.dev142",
            "1.1.1.dev230",
            "1.1.1.dev372",
            "2.0.0.dev602",
            "2.0.1.dev974",
        }
        with pytest.raises(PyPICleanupError, match="Safety check failed"):
            cleanup_dryrun_max_0._determine_versions_to_delete(start_state)

    @patch("duckdb_packaging.pypi_cleanup.PyPICleanup._delete_versions")
    @patch("duckdb_packaging.pypi_cleanup.PyPICleanup._fetch_released_versions")
    @patch("duckdb_packaging.pypi_cleanup.PyPICleanup._determine_versions_to_delete")
    def test_execute_cleanup_dry_run(self, mock_determine, mock_fetch, mock_delete, cleanup_dryrun_max_2):
        mock_fetch.return_value = {"1.0.0.dev1"}
        mock_determine.return_value = {"1.0.0.dev1"}

        with session_with_retries() as session:
            result = cleanup_dryrun_max_2._execute_cleanup(session)

        assert result == 0
        mock_fetch.assert_called_once()
        mock_determine.assert_called_once()
        mock_delete.assert_not_called()

    @patch("duckdb_packaging.pypi_cleanup.PyPICleanup._fetch_released_versions")
    def test_execute_cleanup_no_releases(self, mock_fetch, cleanup_dryrun_max_2):
        mock_fetch.return_value = {}
        with session_with_retries() as session:
            result = cleanup_dryrun_max_2._execute_cleanup(session)
        assert result == 0

    @patch("requests.Session.get")
    def test_fetch_released_versions_success(self, mock_get, cleanup_dryrun_max_2):
        """Test successful package release fetching."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "releases": {
                "1.0.0": [{"upload_time": "2023-01-01T10:00:00"}],
                "1.0.0.dev1": [{"upload_time": "2022-12-01T10:00:00"}],
            }
        }
        mock_get.return_value = mock_response

        with session_with_retries() as session:
            releases = cleanup_dryrun_max_2._fetch_released_versions(session)

        assert releases == {"1.0.0", "1.0.0.dev1"}

    @patch("requests.Session.get")
    def test_fetch_released_versions_not_found(self, mock_get, cleanup_dryrun_max_2):
        """Test package release fetching when package not found."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.return_value = mock_response

        with (
            pytest.raises(PyPICleanupError, match="Failed to fetch package information"),
            session_with_retries() as session,
        ):
            cleanup_dryrun_max_2._fetch_released_versions(session)

    @patch("duckdb_packaging.pypi_cleanup.PyPICleanup._get_csrf_token")
    @patch("requests.Session.post")
    def test_authenticate_success(self, mock_post, mock_csrf, cleanup_max_2):
        """Test successful authentication."""
        mock_csrf.return_value = "csrf123"
        mock_response = Mock()
        mock_response.url = "https://test.pypi.org/manage/"
        mock_post.return_value = mock_response

        with session_with_retries() as session:
            cleanup_max_2._authenticate(session)  # Should not raise
            mock_csrf.assert_called_once_with(session, "/account/login/")

        mock_post.assert_called_once()
        assert mock_post.call_args.args[0].endswith("/account/login/")

    @patch("duckdb_packaging.pypi_cleanup.PyPICleanup._get_csrf_token")
    @patch("requests.Session.post")
    @patch("duckdb_packaging.pypi_cleanup.PyPICleanup._handle_two_factor_auth")
    def test_authenticate_with_2fa(self, mock_2fa, mock_post, mock_csrf, cleanup_max_2):
        mock_csrf.return_value = "csrf123"
        mock_response = Mock()
        mock_response.url = "https://test.pypi.org/account/two-factor/totp"
        mock_post.return_value = mock_response

        with session_with_retries() as session:
            cleanup_max_2._authenticate(session)
            mock_2fa.assert_called_once_with(session, mock_response)

    def test_authenticate_missing_credentials(self, cleanup_dryrun_max_2):
        with pytest.raises(AuthenticationError, match="Username and password are required"):
            cleanup_dryrun_max_2._authenticate(None)

    @patch("duckdb_packaging.pypi_cleanup.PyPICleanup._delete_single_version")
    def test_delete_versions_success(self, mock_delete, cleanup_max_2):
        """Test successful version deletion."""
        versions = {"1.0.0.dev1", "1.0.0.dev2"}
        mock_delete.side_effect = [None, None]  # Successful deletions

        with session_with_retries() as session:
            cleanup_max_2._delete_versions(session, versions)

        assert mock_delete.call_count == 2

    @patch("duckdb_packaging.pypi_cleanup.PyPICleanup._delete_single_version")
    def test_delete_versions_partial_failure(self, mock_delete, cleanup_max_2):
        """Test version deletion with partial failures."""
        versions = {"1.0.0.dev1", "1.0.0.dev2"}
        mock_delete.side_effect = [None, Exception("Delete failed")]

        with pytest.raises(PyPICleanupError, match="Failed to delete 1/2 versions"):
            cleanup_max_2._delete_versions(None, versions)

    def test_delete_single_version_safety_check(self, cleanup_max_2):
        """Test single version deletion safety check."""
        with pytest.raises(PyPICleanupError, match="Refusing to delete non-\\[dev\\|rc\\] version"):
            cleanup_max_2._delete_single_version(None, "1.0.0")  # Non-dev version


class TestArgumentParser:
    """Test command line argument parsing."""

    def test_argument_parser_creation(self):
        """Test argument parser creation."""
        parser = create_argument_parser()
        assert parser.prog is not None

    def test_parse_args_prod_dry_run(self):
        """Test parsing arguments for production dry run."""
        parser = create_argument_parser()
        args = parser.parse_args(["--prod", "--dry-run"])

        assert args.prod is True
        assert args.test is False
        assert args.dry_run is True
        assert args.max_nightlies == 2
        assert args.verbose is False

    def test_parse_args_test_with_username(self):
        """Test parsing arguments for test with username."""
        parser = create_argument_parser()
        args = parser.parse_args(["--test", "-u", "testuser", "--verbose"])

        assert args.test is True
        assert args.prod is False
        assert args.username == "testuser"
        assert args.verbose is True

    def test_parse_args_missing_host(self):
        """Test parsing arguments with missing host selection."""
        parser = create_argument_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--dry-run"])  # Missing --prod or --test


class TestMainFunction:
    """Test the main function."""

    @patch("duckdb_packaging.pypi_cleanup.setup_logging")
    @patch("duckdb_packaging.pypi_cleanup.PyPICleanup")
    @patch.dict(os.environ, {"PYPI_CLEANUP_PASSWORD": "test", "PYPI_CLEANUP_OTP": "test"})
    def test_main_success(self, mock_cleanup_class, mock_setup_logging):
        """Test successful main function execution."""
        mock_cleanup = Mock()
        mock_cleanup.run.return_value = 0
        mock_cleanup_class.return_value = mock_cleanup

        with patch("sys.argv", ["pypi_cleanup.py", "--test", "-u", "testuser"]):
            result = main()

        assert result == 0
        mock_setup_logging.assert_called_once()
        mock_cleanup.run.assert_called_once()

    @patch("duckdb_packaging.pypi_cleanup.setup_logging")
    def test_main_validation_error(self, mock_setup_logging):
        """Test main function with validation error."""
        with patch("sys.argv", ["pypi_cleanup.py", "--test"]):  # Missing username for live mode
            result = main()

        assert result == 2  # Validation error exit code

    @patch("duckdb_packaging.pypi_cleanup.setup_logging")
    @patch("duckdb_packaging.pypi_cleanup.validate_arguments")
    def test_main_keyboard_interrupt(self, mock_validate, mock_setup_logging):
        """Test main function with keyboard interrupt."""
        mock_validate.side_effect = KeyboardInterrupt()

        with patch("sys.argv", ["pypi_cleanup.py", "--test", "--dry-run"]):
            result = main()

        assert result == 130  # Keyboard interrupt exit code

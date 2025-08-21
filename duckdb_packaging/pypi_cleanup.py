"""
!!HERE BE DRAGONS!! Use this script with care!

PyPI package cleanup tool. This script will:
* Never remove a stable version (including a post release version)
* Remove all release candidates for versions that have stable releases
* Remove all dev releases for versions that have stable releases
* Keep the configured amount of dev releases per version, and remove older dev releases
"""

import argparse
import contextlib
import heapq
import logging
import os
import re
import sys
import time
from collections import defaultdict
from html.parser import HTMLParser
from typing import Optional, Set, Generator
from urllib.parse import urlparse

import pyotp
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3 import Retry

_PYPI_URL_PROD = 'https://pypi.org/'
_PYPI_URL_TEST = 'https://test.pypi.org/'
_DEFAULT_MAX_NIGHTLIES = 2
_LOGIN_RETRY_ATTEMPTS = 3
_LOGIN_RETRY_DELAY = 5


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="""
PyPI cleanup script for removing development versions.

!!HERE BE DRAGONS!! Use this script with care!

This script will:
* Never remove a stable version (including a post release version)
* Remove all release candidates for versions that have stable releases
* Remove all dev releases for versions that have stable releases
* Keep the configured amount of dev releases per version, and remove older dev releases
        """,
        epilog="Environment variables required (unless --dry-run): PYPI_CLEANUP_PASSWORD, PYPI_CLEANUP_OTP",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted but don't actually do it"
    )

    host_group = parser.add_mutually_exclusive_group(required=True)
    host_group.add_argument(
        "--prod",
        action="store_true",
        help="Use production PyPI (pypi.org)"
    )
    host_group.add_argument(
        "--test",
        action="store_true",
        help="Use test PyPI (test.pypi.org)"
    )

    parser.add_argument(
        "-m", "--max-nightlies",
        type=int,
        default=_DEFAULT_MAX_NIGHTLIES,
        help=f"Max number of nightlies of unreleased versions (default={_DEFAULT_MAX_NIGHTLIES})"
    )

    parser.add_argument(
        "-u", "--username",
        type=validate_username,
        help="PyPI username (required unless --dry-run)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )

    return parser

class PyPICleanupError(Exception):
    """Base exception for PyPI cleanup operations."""
    pass


class AuthenticationError(PyPICleanupError):
    """Raised when authentication fails."""
    pass


class ValidationError(PyPICleanupError):
    """Raised when input validation fails."""
    pass


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with appropriate level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_username(value: str) -> str:
    """Validate and sanitize username input."""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("Username cannot be empty")
    
    username = value.strip()
    if len(username) > 100:  # Reasonable limit
        raise argparse.ArgumentTypeError("Username too long (max 100 characters)")
    
    # Basic validation - PyPI usernames are alphanumeric with limited special chars
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$', username):
        raise argparse.ArgumentTypeError("Invalid username format")
    
    return username

@contextlib.contextmanager
def session_with_retries() -> Generator[Session, None, None]:
    """Create a requests session with retry strategy for ephemeral errors."""
    with requests.Session() as session:
        retry_strategy = Retry(
            allowed_methods=["GET", "POST"],
            total=None,  # disable to make the below take effect
            redirect=10,  # Don't follow more than 10 redirects in a row
            connect=3,  # try 3 times before giving up on connection errors
            read=3,  # try 3 times before giving up on read errors
            status=3,  # try 3 times before giving up on status errors (see forcelist below)
            status_forcelist=[429] + [status for status in range(500, 512)],
            other=0,  # whatever else may cause an error should break
            backoff_factor=0.1,  # [0.0s, 0.2s, 0.4s]
            raise_on_redirect=True,  # raise exception when redirect error retries are exhausted
            raise_on_status=True,  # raise exception when status error retries are exhausted
            respect_retry_after_header=True,  # respect Retry-After headers
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        yield session

def load_credentials(dry_run: bool) -> tuple[Optional[str], Optional[str]]:
    """Load credentials from environment variables."""
    if dry_run:
        return None, None
    
    password = os.getenv('PYPI_CLEANUP_PASSWORD')
    otp = os.getenv('PYPI_CLEANUP_OTP')
    
    if not password:
        raise ValidationError("PYPI_CLEANUP_PASSWORD environment variable is required when not in dry-run mode")
    if not otp:
        raise ValidationError("PYPI_CLEANUP_OTP environment variable is required when not in dry-run mode")
    
    return password, otp


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate parsed arguments."""
    if not args.dry_run and not args.username:
        raise ValidationError("--username is required when not in dry-run mode")
    
    if args.max_nightlies < 0:
        raise ValidationError("--max-nightlies must be non-negative")

class CsrfParser(HTMLParser):
    """HTML parser to extract CSRF tokens from PyPI forms.
    
    Based on pypi-cleanup package (https://github.com/arcivanov/pypi-cleanup/tree/master)
    """
    def __init__(self, target, contains_input=None):
        super().__init__()
        self._target = target
        self._contains_input = contains_input
        self.csrf = None  # Result value from all forms on page
        self._csrf = None  # Temp value from current form
        self._in_form = False  # Currently parsing a form with an action we're interested in
        self._input_contained = False  # Input field requested is contained in the current form

    def handle_starttag(self, tag, attrs):
        if tag == "form":
            attrs = dict(attrs)
            action = attrs.get("action")  # Might be None.
            if action and (action == self._target or action.startswith(self._target)):
                self._in_form = True
            return

        if self._in_form and tag == "input":
            attrs = dict(attrs)
            if attrs.get("name") == "csrf_token":
                self._csrf = attrs["value"]

            if self._contains_input and attrs.get("name") == self._contains_input:
                self._input_contained = True

            return

    def handle_endtag(self, tag):
        if tag == "form":
            self._in_form = False
            # If we're in a right form that contains the requested input and csrf is not set
            if (not self._contains_input or self._input_contained) and not self.csrf:
                self.csrf = self._csrf
            return


class PyPICleanup:
    """Main class for performing PyPI package cleanup operations."""

    def __init__(self, index_url: str, do_delete: bool, max_dev_releases: int=_DEFAULT_MAX_NIGHTLIES,
                 username: Optional[str]=None, password: Optional[str]=None, otp: Optional[str]=None):
        parsed_url = urlparse(index_url)
        self._index_url = parsed_url.geturl().rstrip('/')
        self._index_host = parsed_url.hostname
        self._do_delete = do_delete
        self._max_dev_releases = max_dev_releases
        self._username = username
        self._password = password
        self._otp = otp
        self._package = 'duckdb'
        self._dev_version_pattern = re.compile(r"^(?P<version>\d+\.\d+\.\d+)\.dev(?P<dev_id>\d+)$")
        self._rc_version_pattern = re.compile(r"^(?P<version>\d+\.\d+\.\d+)\.rc\d+$")
        self._stable_version_pattern = re.compile(r"^\d+\.\d+\.\d+(\.post\d+)?$")

    def run(self) -> int:
        """Execute the cleanup process.
        
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        if self._do_delete:
            logging.warning(f"NOT A DRILL: WILL DELETE PACKAGES")
        else:
            logging.info("Running in DRY RUN mode, nothing will be deleted")

        logging.info(f"Max development releases to keep per unreleased version: {self._max_dev_releases}")

        try:
            with session_with_retries() as http_session:
                return self._execute_cleanup(http_session)
        except PyPICleanupError as e:
            logging.error(f"Cleanup failed: {e}")
            return 1
        except Exception as e:
            logging.error(f"Unexpected error: {e}", exc_info=True)
            return 1

    def _execute_cleanup(self, http_session: Session) -> int:
        """Execute the main cleanup logic."""

        # Get released versions
        versions = self._fetch_released_versions(http_session)
        if not versions:
            logging.info(f"No releases found for {self._package}")
            return 0
        
        # Determine versions to delete
        versions_to_delete = self._determine_versions_to_delete(versions)
        if not versions_to_delete:
            logging.info("No versions to delete (no stale rc's or dev releases)")
            return 0
        
        logging.warning(f"Found {len(versions_to_delete)} versions to clean up:")
        for version in sorted(versions_to_delete):
            logging.warning(version)
        
        if not self._do_delete:
            logging.info("Dry run complete - no packages were deleted")
            return 0

        # Perform authentication and deletion
        self._authenticate(http_session)
        self._delete_versions(http_session, versions_to_delete)
        
        logging.info(f"Successfully cleaned up {len(versions_to_delete)} development versions")
        return 0
    
    def _fetch_released_versions(self, http_session: Session) -> Set[str]:
        """Fetch package release information from PyPI API."""
        logging.debug(f"Fetching package information for '{self._package}'")
        
        try:
            req = http_session.get(f"{self._index_url}/pypi/{self._package}/json")
            req.raise_for_status()
            data = req.json()
            versions = {v for v, files in data["releases"].items() if len(files) > 0}
            logging.debug(f"Found {len(versions)} releases with files")
            return versions
        except RequestException as e:
            raise PyPICleanupError(f"Failed to fetch package information for '{self._package}': {e}") from e

    def _is_stable_release_version(self, version: str) -> bool:
        """Determine whether a version string denotes a stable release."""
        return self._stable_version_pattern.match(version) is not None

    def _is_rc_version(self, version: str) -> bool:
        """Determine whether a version string denotes a stable release."""
        return self._rc_version_pattern.match(version) is not None

    def _is_dev_version(self, version: str) -> bool:
        """Determine whether a version string denotes a dev release."""
        return self._dev_version_pattern.match(version) is not None

    def _parse_rc_version(self, version: str) -> str:
        """Parse a rc version string to determine the base version."""
        match = self._rc_version_pattern.match(version)
        if not match:
            raise PyPICleanupError(f"Invalid rc version '{version}'")
        return match.group("version") if match else None

    def _parse_dev_version(self, version: str) -> tuple[str, int]:
        """Parse a dev version string to determine the base version and dev version id."""
        match = self._dev_version_pattern.match(version)
        if not match:
            raise PyPICleanupError(f"Invalid dev version '{version}'")
        return match.group("version"), int(match.group("dev_id"))

    def _determine_versions_to_delete(self, versions: Set[str]) -> Set[str]:
        """Determine which package versions should be deleted."""
        logging.debug("Analyzing versions to determine cleanup candidates")

        # Get all stable, rc and dev versions
        stable_versions = {v for v in versions if self._is_stable_release_version(v)}
        rc_versions = {v for v in versions if self._is_rc_version(v)}
        rc_base_versions = {self._parse_rc_version(v) for v in versions if self._is_rc_version(v)}
        dev_versions = {v for v in versions if self._is_dev_version(v)}

        # Set of all rc releases of versions that have a stable release
        rcs_of_stable = {v for v in rc_versions if self._parse_rc_version(v) in stable_versions}
        # Set of all dev releases of versions that have a stable or rc release
        devs_of_stable = {v for v in dev_versions if self._parse_dev_version(v)[0] in stable_versions}
        devs_of_rc = {v for v in dev_versions if self._parse_dev_version(v)[0] in rc_base_versions}
        # Set of orphan dev versions
        orphan_devs = dev_versions.difference(devs_of_stable).difference(devs_of_rc)

        # Construct list of orphan dev
        orphan_devs_per_version = defaultdict(list)
        # 1. put all dev keep candidates on a max heap indexed by negative dev id (i.e. dev10 -> -10)
        for version in orphan_devs:
            base_version, dev_id = self._parse_dev_version(version)
            heapq.heappush(orphan_devs_per_version[base_version], (-dev_id, version))
        # 2. remove the amount of latest dev releases we want to keep
        for version_list in orphan_devs_per_version.values():
            for _ in range(min(self._max_dev_releases, len(version_list))):
                heapq.heappop(version_list)
        # 3. Result: set of outdated dev versions
        devs_outdated = {v for version_list in orphan_devs_per_version.values() for _, v in version_list}

        # Construct final deletion set
        versions_to_delete = set()
        if rcs_of_stable:
            versions_to_delete.update(rcs_of_stable)
            logging.info(f"Found {len(rcs_of_stable)} release candidates that have stable releases")
        if devs_of_stable:
            versions_to_delete.update(devs_of_stable)
            logging.info(f"Found {len(devs_of_stable)} dev releases that have stable releases")
        if devs_of_rc:
            versions_to_delete.update(devs_of_rc)
            logging.info(f"Found {len(devs_of_rc)} dev releases that have release candidates")
        if devs_outdated:
            versions_to_delete.update(devs_outdated)
            logging.info(f"Found {len(devs_outdated)} dev releases that are outdated")

        # Final safety checks
        if versions_to_delete == versions:
            raise PyPICleanupError(
                f"Safety check failed: cleanup would delete ALL versions of '{self._package}'. "
                "This would make the package permanently inaccessible. Aborting."
            )
        if len(versions_to_delete.intersection(stable_versions)) > 0:
            raise PyPICleanupError(
                f"Safety check failed: cleanup would delete one or more stable versions of '{self._package}'. "
                f"A regexp might be broken? (would delete {versions_to_delete.intersection(stable_versions)})"
            )
        unknown_versions = versions.difference(stable_versions).difference(rc_versions).difference(dev_versions)
        if unknown_versions:
            logging.warning(f"Found version string(s) in an unsupported format: {unknown_versions}")

        return versions_to_delete
    
    def _authenticate(self, http_session: Session) -> None:
        """Authenticate with PyPI."""
        if not self._username or not self._password:
            raise AuthenticationError("Username and password are required for authentication")
        
        logging.info(f"Authenticating user '{self._username}' with PyPI")

        try:
            # Attempt login
            login_response = self._perform_login(http_session)

            # Handle two-factor authentication if required
            if login_response.url.startswith(f"{self._index_url}/account/two-factor/"):
                logging.debug("Two-factor authentication required")
                self._handle_two_factor_auth(http_session, login_response)
            
            logging.info("Authentication successful")

        except RequestException as e:
            raise AuthenticationError(f"Network error during authentication: {e}") from e
    
    def _get_csrf_token(self, http_session: Session, form_action: str) -> str:
        """Extract CSRF token from a form page."""
        resp = http_session.get(f"{self._index_url}{form_action}")
        resp.raise_for_status()
        parser = CsrfParser(form_action)
        parser.feed(resp.text)
        if not parser.csrf:
            raise AuthenticationError(f"No CSRF token found in {form_action}")
        return parser.csrf
    
    def _perform_login(self, http_session: Session) -> requests.Response:
        """Perform the initial login with username/password."""

        # Get login form and CSRF token
        csrf_token = self._get_csrf_token(http_session, "/account/login/")

        login_data = {
            "csrf_token": csrf_token,
            "username": self._username,
            "password": self._password
        }

        response = http_session.post(
            f"{self._index_url}/account/login/",
            data=login_data,
            headers={"referer": f"{self._index_url}/account/login/"}
        )
        response.raise_for_status()

        # Check if login failed (redirected back to login page)
        if response.url == f"{self._index_url}/account/login/":
            raise AuthenticationError(f"Login failed for user '{self._username}' - check credentials")

        return response
    
    def _handle_two_factor_auth(self, http_session: Session, response: requests.Response) -> None:
        """Handle two-factor authentication."""
        if not self._otp:
            raise AuthenticationError("Two-factor authentication required but no OTP secret provided")
        
        two_factor_url = response.url
        form_action = two_factor_url[len(self._index_url):]
        csrf_token = self._get_csrf_token(http_session, form_action)
        
        # Try authentication with retries
        for attempt in range(_LOGIN_RETRY_ATTEMPTS):
            try:
                auth_code = pyotp.TOTP(self._otp).now()
                logging.debug(f"Attempting 2FA with code (attempt {attempt + 1}/{_LOGIN_RETRY_ATTEMPTS})")

                auth_response = http_session.post(
                    two_factor_url,
                    data={"csrf_token": csrf_token, "method": "totp", "totp_value": auth_code},
                    headers={"referer": two_factor_url}
                )
                auth_response.raise_for_status()

                # Check if 2FA succeeded (redirected away from 2FA page)
                if auth_response.url != two_factor_url:
                    logging.debug("Two-factor authentication successful")
                    return

                if attempt < _LOGIN_RETRY_ATTEMPTS - 1:
                    logging.debug(f"2FA code rejected, retrying in {_LOGIN_RETRY_DELAY} seconds...")
                    time.sleep(_LOGIN_RETRY_DELAY)
                
            except RequestException as e:
                if attempt == _LOGIN_RETRY_ATTEMPTS - 1:
                    raise AuthenticationError(f"Network error during 2FA: {e}") from e
                logging.debug(f"Network error during 2FA attempt {attempt + 1}, retrying...")
                time.sleep(_LOGIN_RETRY_DELAY)
        
        raise AuthenticationError("Two-factor authentication failed after all attempts")
    
    def _delete_versions(self, http_session: Session, versions_to_delete: Set[str]) -> None:
        """Delete the specified package versions."""
        logging.info(f"Starting deletion of {len(versions_to_delete)} development versions")
        
        failed_deletions = list()
        for version in sorted(versions_to_delete):
            try:
                self._delete_single_version(http_session, version)
                logging.info(f"Successfully deleted {self._package} version {version}")
            except Exception as e:
                # Continue with other versions rather than failing completely
                logging.error(f"Failed to delete version {version}: {e}")
                failed_deletions.append(version)
        
        if failed_deletions:
            raise PyPICleanupError(
                f"Failed to delete {len(failed_deletions)}/{len(versions_to_delete)} versions: {failed_deletions}"
            )
    
    def _delete_single_version(self, http_session: Session, version: str) -> None:
        """Delete a single package version."""
        # Safety check
        if not self._is_dev_version(version) or self._is_rc_version(version):
            raise PyPICleanupError(f"Refusing to delete non-[dev|rc] version: {version}")
        
        logging.debug(f"Deleting {self._package} version {version}")
        
        # Get deletion form and CSRF token
        form_action = f"/manage/project/{self._package}/release/{version}/"
        form_url = f"{self._index_url}{form_action}"
        
        csrf_token = self._get_csrf_token(http_session, form_action)

        # Submit deletion request
        delete_response = http_session.post(
            form_url,
            data={
                "csrf_token": csrf_token,
                "confirm_delete_version": version,
            },
            headers={"referer": form_url}
        )
        delete_response.raise_for_status()


def main() -> int:
    """Main entry point for the script."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Validate arguments
        validate_arguments(args)
        
        # Load credentials
        password, otp = load_credentials(args.dry_run)
        
        # Determine PyPI URL
        pypi_url = _PYPI_URL_PROD if args.prod else _PYPI_URL_TEST
        
        # Create and run cleanup
        cleanup = PyPICleanup(pypi_url, not args.dry_run, args.max_nightlies, username=args.username,
            password=password, otp=otp)
        
        return cleanup.run()
        
    except ValidationError as e:
        logging.error(f"Configuration error: {e}")
        return 2
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())

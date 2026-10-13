import os
import platform
import subprocess
import sys

import pytest

import duckdb

pytestmark = pytest.mark.skipif(
    platform.system() == "Emscripten",
    reason="Extensions are not supported on Emscripten",
)


def test_extension_loading(require):
    if not os.getenv("DUCKDB_PYTHON_TEST_EXTENSION_REQUIRED", False):
        return
    extensions_list = ["json", "excel", "httpfs", "tpch", "tpcds", "icu", "fts"]
    for extension in extensions_list:
        connection = require(extension)
        assert connection is not None


# http.server's HTTPServer resolves its own name with getfqdn() before it listens,
# which can stall on macOS CI runners.
_SERVE_DIRECTORY = """
import functools, http.server, socketserver, sys
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=sys.argv[1])
with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as server:
    print(server.server_address[1], flush=True)
    server.serve_forever()
"""


@pytest.fixture
def empty_extension_repository(tmp_path):
    # install_extension holds the GIL for the whole request, so a server in this
    # interpreter would deadlock with it. Serving an empty directory answers
    # every extension request with a 404.
    server = subprocess.Popen(
        [sys.executable, "-c", _SERVE_DIRECTORY, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        port = server.stdout.readline().strip()
        if not port:
            pytest.fail("the extension repository server did not come up")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.terminate()
        server.stdout.close()
        server.wait(timeout=30)


@pytest.mark.timeout(120)
def test_install_non_existent_extension(empty_extension_repository):
    conn = duckdb.connect()
    conn.execute(f"set custom_extension_repository = '{empty_extension_repository}'")

    with pytest.raises(duckdb.HTTPException) as exc:
        conn.install_extension("non-existent")

    value = exc.value
    assert value.status_code == 404
    assert value.reason
    assert value.body
    assert "Server" in value.headers


def test_install_from_local_repository(tmp_path):
    conn = duckdb.connect()
    conn.execute(f"set custom_extension_repository = '{tmp_path.as_posix()}'")

    with pytest.raises(duckdb.IOException, match='no access to the file at PATH "'):
        conn.install_extension("non-existent")


def test_install_rejects_a_file_that_is_not_an_extension(tmp_path):
    not_an_extension = tmp_path / "not-an-extension.duckdb_extension"
    not_an_extension.write_bytes(b"\0" * 2048)

    conn = duckdb.connect()
    with pytest.raises(duckdb.IOException, match="not a DuckDB extension"):
        conn.install_extension(not_an_extension.as_posix())


def test_load_non_existent_extension(tmp_path):
    conn = duckdb.connect()

    # Reaching the missing file at all is the point: without the
    # loadable_extensions library LOAD refuses before it looks for one.
    with pytest.raises(duckdb.IOException, match="not found"):
        conn.load_extension((tmp_path / "non-existent.duckdb_extension").as_posix())


def test_install_misuse_errors(duckdb_cursor):
    with pytest.raises(
        duckdb.InvalidInputException,
        match="Both 'repository' and 'repository_url' are set which is not allowed, please pick one or the other",
    ):
        duckdb_cursor.install_extension("name", repository="hello", repository_url="hello.com")

    with pytest.raises(
        duckdb.InvalidInputException, match="The provided 'repository' or 'repository_url' can not be empty!"
    ):
        duckdb_cursor.install_extension("name", repository_url="")

    with pytest.raises(
        duckdb.InvalidInputException, match="The provided 'repository' or 'repository_url' can not be empty!"
    ):
        duckdb_cursor.install_extension("name", repository="")

    with pytest.raises(duckdb.InvalidInputException, match="The provided 'version' can not be empty!"):
        duckdb_cursor.install_extension("name", version="")

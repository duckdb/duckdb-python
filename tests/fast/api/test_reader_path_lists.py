"""A connection's readers take a list of paths, and their annotations say so."""

import json
import re

import pytest

import duckdb

READERS = ("read_json", "read_csv", "from_csv_auto")
#: Every path shape the readers take at runtime: one path, a list literal, a list[str] variable, a list of
#: pathlib.Path and a list of a file like object.
ACCEPTED_ARGUMENTS = ('"a"', '["a", "b"]', "paths", '[pathlib.Path("a")]', '[io.BytesIO(b"")]')


def annotated_calls():
    """The snippet mypy checks, and the line of each reader's call with a number."""
    lines = ["import io", "import pathlib", "", "import duckdb", "", "con = duckdb.connect()"]
    lines.append('paths: list[str] = ["a", "b"]')
    refused = {}
    for reader in READERS:
        lines.extend(f"con.{reader}({argument})" for argument in ACCEPTED_ARGUMENTS)
        lines.append(f"con.{reader}(42)")
        refused[reader] = len(lines)
    return "\n".join(lines) + "\n", refused


def test_read_json_reads_every_file_of_a_list(tmp_path):
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({"x": name}) + "\n")
        paths.append(path)
    con = duckdb.connect()
    assert con.read_json([str(p) for p in paths]).order("x").fetchall() == [("a",), ("b",)]
    assert con.read_json(paths, format="newline_delimited").order("x").fetchall() == [("a",), ("b",)]


def test_mypy_accepts_every_path_shape_and_refuses_a_number(tmp_path):
    api = pytest.importorskip("mypy.api")
    text, refused = annotated_calls()
    snippet = tmp_path / "calls.py"
    snippet.write_text(text)
    dash = "-" * 2
    flags = [f"{dash}{flag}" for flag in ("strict", "no-incremental", "follow-imports=silent", "no-pretty")]
    report, errors, _ = api.run([str(snippet), *flags, f"{dash}cache-dir", str(tmp_path / "cache")])
    found = {int(line): message for line, message in re.findall(r"calls\.py:(\d+): error: (.*)", report)}
    assert sorted(found) == sorted(refused.values()), report + errors
    for reader, line in refused.items():
        expected = f'Argument 1 to "{reader}" of "DuckDBPyConnection" has incompatible type "int"'
        assert expected in found[line], report

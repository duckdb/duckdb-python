"""Check the code generation flags of a build from its compile database.

Usage: check_compile_commands.py <compile_commands.json> [--lto MODE] [--compiler NAME]

--lto MODE      every compile command must carry -flto=MODE.
--compiler NAME every compile command must invoke a compiler whose file name
                contains NAME, such as clang.

Exits 1 and lists the offending sources when a check fails.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path, PurePath


def arguments_of(entry: dict) -> list[str]:
    if "arguments" in entry:
        return list(entry["arguments"])
    return shlex.split(entry["command"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database", type=Path)
    parser.add_argument("--lto", metavar="MODE")
    parser.add_argument("--compiler", metavar="NAME")
    args = parser.parse_args()

    if not args.database.is_file():
        print(f"no compile database at {args.database}", file=sys.stderr)
        return 1
    entries = json.loads(args.database.read_text())
    print(f"{len(entries)} compile commands")
    if not entries:
        print("the compile database is empty", file=sys.stderr)
        return 1

    failures: list[str] = []
    if args.lto:
        flag = f"-flto={args.lto}"
        missing = [e["file"] for e in entries if flag not in arguments_of(e)]
        if missing:
            failures.append(f"{len(missing)} compile commands lack {flag}:")
            failures.extend(f"  {f}" for f in missing[:10])
    if args.compiler:
        wrong = [e["file"] for e in entries if args.compiler not in PurePath(arguments_of(e)[0]).name]
        if wrong:
            failures.append(f"{len(wrong)} compile commands do not invoke {args.compiler}:")
            failures.extend(f"  {f}" for f in wrong[:10])

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    checks = [c for c in (args.lto and f"-flto={args.lto}", args.compiler and f"compiler {args.compiler}") if c]
    print("ok: " + ", ".join(checks) if checks else "ok: nothing to check")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import subprocess
import sys
import textwrap

import pytest

import duckdb


@pytest.mark.parametrize("main_setup", ["del sys.modules['__main__']", "sys.modules['__main__'] = None"])
def test_import_without_main_module(main_setup):
    script = textwrap.dedent(f"""\
        import sys
        {main_setup}
        import duckdb

        with duckdb.connect(config={{"threads": 1}}) as connection:
            print(connection.execute("SELECT 42, current_setting('enable_progress_bar')").fetchone())
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "(42, False)"


class TestModule:
    def test_paramstyle(self):
        assert duckdb.paramstyle == "qmark"

    def test_threadsafety(self):
        assert duckdb.threadsafety == 1

    def test_apilevel(self):
        assert duckdb.apilevel == "2.0"

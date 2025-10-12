import contextlib
import shutil
from pathlib import Path

import duckdb


class TestMultiStatement:
    def test_multi_statement(self, duckdb_cursor):
        con = duckdb.connect(":memory:")

        # test empty statement
        con.execute("")

        # run multiple statements in one call to execute
        con.execute(
            """
        CREATE TABLE integers(i integer);
        insert into integers select * from range(10);
        select * from integers;
        """
        )
        results = [x[0] for x in con.fetchall()]
        assert results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # test export/import
        export_location = Path.cwd() / "duckdb_pytest_dir_export"
        with contextlib.suppress(Exception):
            shutil.rmtree(export_location)
        con.execute("CREATE TABLE integers2(i INTEGER)")
        con.execute("INSERT INTO integers2 VALUES (1), (5), (7), (1928)")
        con.execute(f"EXPORT DATABASE '{export_location}'")
        # reset connection
        con = duckdb.connect(":memory:")
        con.execute(f"IMPORT DATABASE '{export_location}'")
        integers = [x[0] for x in con.execute("SELECT * FROM integers").fetchall()]
        integers2 = [x[0] for x in con.execute("SELECT * FROM integers2").fetchall()]
        assert integers == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert integers2 == [1, 5, 7, 1928]
        shutil.rmtree(export_location)

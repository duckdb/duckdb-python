import pytest

import duckdb
import pandas as pd


class TestPandasEnum:
    def test_3480(self, duckdb_cursor):
        duckdb_cursor.execute(
            """
        create type cat as enum ('marie', 'duchess', 'toulouse');
        create table tab (
            cat cat,
            amt int
        );
        """
        )
        df = duckdb_cursor.query("SELECT * FROM tab LIMIT 0;").to_df()
        assert df["cat"].cat.categories.equals(pd.Index(["marie", "duchess", "toulouse"]))
        duckdb_cursor.execute("DROP TABLE tab")
        duckdb_cursor.execute("DROP TYPE cat")

    def test_3479(self, duckdb_cursor):
        duckdb_cursor.execute(
            """
        create type cat as enum ('marie', 'duchess', 'toulouse');
        create table tab (
            cat cat,
            amt int
        );
        """
        )

        df = pd.DataFrame(
            {
                "cat2": pd.Series(["duchess", "toulouse", "marie", None, "berlioz", "o_malley"], dtype="category"),
                "amt": [1, 2, 3, 4, 5, 6],
            }
        )
        duckdb_cursor.register("df", df)
        with pytest.raises(
            duckdb.ConversionException,
            match="with value berlioz can't be cast to the destination type ENUM",
        ):
            duckdb_cursor.execute("INSERT INTO tab SELECT * FROM df;")

        assert duckdb_cursor.execute("select * from tab").fetchall() == []
        duckdb_cursor.execute("DROP TABLE tab")
        duckdb_cursor.execute("DROP TYPE cat")

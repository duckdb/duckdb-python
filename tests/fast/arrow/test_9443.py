from datetime import time
from pathlib import PurePosixPath

import pytest

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")


class Test9443:
    def test_9443(self, tmp_path, duckdb_cursor):
        arrow_table = pa.Table.from_pylist(
            [
                {"col1": time(1, 2, 3)},
            ]
        )  # col1: time64[us]

        print(arrow_table)

        temp_file = str(PurePosixPath(tmp_path.as_posix()) / "test9443.parquet")
        pq.write_table(arrow_table, temp_file)

        sql = f'SELECT * FROM "{temp_file}"'

        duckdb_cursor.execute(sql)
        duckdb_cursor.to_arrow_reader()

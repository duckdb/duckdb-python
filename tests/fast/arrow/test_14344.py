import hashlib

import pytest

pa = pytest.importorskip("pyarrow")


def test_14344(duckdb_cursor):
    my_table = pa.Table.from_pydict({"foo": pa.array([hashlib.sha256(b"foo").digest()], type=pa.binary())})  # noqa: F841
    my_table2 = pa.Table.from_pydict(  # noqa: F841
        {"foo": pa.array([hashlib.sha256(b"foo").digest()], type=pa.binary()), "a": ["123"]}
    )

    res = duckdb_cursor.sql(
        """
		SELECT
			my_table2.* EXCLUDE (foo)
		FROM
			my_table
		LEFT JOIN
			my_table2
		USING (foo)
	"""
    ).fetchall()
    assert res == [("123",)]

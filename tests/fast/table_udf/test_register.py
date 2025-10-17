import pytest

import duckdb
import duckdb.sqltypes as sqltypes


def test_registry_collision(tmp_path):
    """Two table_udfs on different connections with same name""" ""
    conn1 = duckdb.connect(tmp_path / "db1.db")
    conn2 = duckdb.connect(tmp_path / "db2.db")

    def func_for_conn1():
        return [("conn1_data", 1)]

    def func_for_conn2():
        return [("conn2_data", 2)]

    schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

    conn1.create_table_function(
        name="same_name",
        callable=func_for_conn1,
        parameters=None,
        schema=schema,
        type="tuples",
    )

    conn2.create_table_function(
        name="same_name",
        callable=func_for_conn2,
        parameters=None,
        schema=schema,
        type="tuples",
    )

    result1 = conn1.execute("SELECT * FROM same_name()").fetchall()
    assert result1[0][0] == "conn1_data"
    assert result1[0][1] == 1

    result2 = conn2.execute("SELECT * FROM same_name()").fetchall()
    assert result2[0][0] == "conn2_data"
    assert result2[0][1] == 2

    result1 = conn1.sql("SELECT * FROM same_name()").fetchall()
    assert result1[0][0] == "conn1_data"
    assert result1[0][1] == 1

    conn1.close()
    conn2.close()


def test_replace_without_unregister(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:

        def func_v1():
            return [("version_1", 1)]

        def func_v2():
            return [("version_2", 2)]

        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function("test_func", func_v1, schema=schema, type="tuples")

        result = conn.execute("SELECT * FROM test_func()").fetchall()
        assert result[0][0] == "version_1"
        assert result[0][1] == 1

        with pytest.raises(duckdb.NotImplementedException) as exc_info:
            conn.create_table_function("test_func", func_v2, schema=schema, type="tuples")
        assert "already registered" in str(exc_info.value)


def test_replace_after_unregister(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:

        def func_v1():
            return [("version_1", 1)]

        def func_v2():
            return [("version_2", 2)]

        def func_v3():
            return [("version_3", 3)]

        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function("test_func", func_v1, schema=schema, type="tuples")

        result = conn.execute("SELECT * FROM test_func()").fetchall()
        assert result[0][0] == "version_1"

        conn.unregister_table_function("test_func")
        conn.create_table_function("test_func", func_v2, schema=schema, type="tuples")

        result = conn.execute("SELECT * FROM test_func()").fetchall()
        assert result[0][0] == "version_2"

        conn.unregister_table_function("test_func")

        result = conn.execute("SELECT * FROM test_func()").fetchall()
        assert result[0][0] == "version_2"

        conn.create_table_function("test_func", func_v3, schema=schema, type="tuples")

        result = conn.execute("SELECT * FROM test_func()").fetchall()
        assert result[0][0] == "version_3"
        assert result[0][1] == 3


def test_multiple_replacements(tmp_path):
    """Replacing Table UDFs multiple times."""
    with duckdb.connect(tmp_path / "test.db") as conn:
        schema = {"value": sqltypes.INTEGER}

        for i in range(1, 6):

            def make_func(val=i):
                def func():
                    return [(val,)]

                return func

            if i > 1:
                conn.unregister_table_function("counter")

            conn.create_table_function("counter", make_func(), schema=schema, type="tuples")

            result = conn.execute("SELECT * FROM counter()").fetchone()
            assert result[0] == i


def test_replacement_with_different_schemas(tmp_path):
    """Changing schema with replacements."""
    with duckdb.connect(tmp_path / "test.db") as conn:

        def func_v1():
            return [("test", 1)]

        def func_v2():
            return [("modified", 2, 3.14)]

        schema_v1 = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}
        conn.create_table_function("evolving_func", func_v1, schema=schema_v1, type="tuples")

        result = conn.execute("SELECT * FROM evolving_func()").fetchall()
        assert len(result[0]) == 2
        assert result[0][0] == "test"

        schema_v2 = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER, "value": sqltypes.DOUBLE}
        conn.unregister_table_function("evolving_func")  # Must unregister first
        conn.create_table_function("evolving_func", func_v2, schema=schema_v2, type="tuples")

        result = conn.execute("SELECT * FROM evolving_func()").fetchall()
        assert len(result[0]) == 3
        assert result[0][0] == "modified"
        assert result[0][2] == 3.14


def test_replacement_2(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:

        def func_v1():
            return [("v1",)]

        def func_v2():
            return [("v2",)]

        schema = {"version": sqltypes.VARCHAR}

        conn.create_table_function("tracked_func", func_v1, schema=schema, type="tuples")

        conn.unregister_table_function("tracked_func")  # Must unregister first
        conn.create_table_function("tracked_func", func_v2, schema=schema, type="tuples")

        conn.unregister_table_function("tracked_func")

        with pytest.raises(duckdb.InvalidInputException) as exc_info:
            conn.unregister_table_function("tracked_func")
        assert "No table function by the name of 'tracked_func'" in str(exc_info.value)

        result = conn.execute("SELECT * FROM tracked_func()").fetchone()
        assert result[0] == "v2"


def test_sql_drop_table_function(tmp_path):
    """Documents current behavior - that dropping functions has no effect on Table UDFs."""
    with duckdb.connect(tmp_path / "test.db") as conn:

        def test_func():
            return [("test_value", 1)]

        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}
        conn.create_table_function("test_func", test_func, schema=schema, type="tuples")

        result = conn.execute("SELECT * FROM test_func()").fetchall()
        assert result[0][0] == "test_value"
        assert result[0][1] == 1

        with pytest.raises(duckdb.CatalogException):
            conn.execute("DROP FUNCTION test_func")

        result = conn.execute("SELECT * FROM test_func()").fetchall()
        assert result[0][0] == "test_value"
        assert result[0][1] == 1


def test_unregister_table_function(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:

        def simple_function():
            return [("test_value", 1)]

        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function(
            name="test_func",
            callable=simple_function,
            parameters=None,
            schema=schema,
            type="tuples",
        )

        result = conn.execute("SELECT * FROM test_func()").fetchall()
        assert len(result) == 1
        assert result[0][0] == "test_value"
        assert result[0][1] == 1

        conn.unregister_table_function("test_func")

        result = conn.execute("SELECT * FROM test_func()").fetchall()
        assert len(result) == 1
        assert result[0][0] == "test_value"
        assert result[0][1] == 1

        with pytest.raises(duckdb.InvalidInputException) as exc_info:
            conn.unregister_table_function("test_func")

        assert "No table function by the name of 'test_func'" in str(exc_info.value)


def test_unregister_doesntexist(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:
        with pytest.raises(duckdb.InvalidInputException) as exc_info:
            conn.unregister_table_function("nonexistent_func")

        assert "No table function by the name of 'nonexistent_func'" in str(exc_info.value)


def test_reregister(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:

        def func_v1():
            return [("version_1", 1)]

        def func_v2():
            return [("version_2", 2)]

        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        conn.create_table_function(
            name="versioned_func",
            callable=func_v1,
            schema=schema,
            type="tuples",
        )

        result = conn.execute("SELECT * FROM versioned_func()").fetchall()
        assert result[0][0] == "version_1"

        conn.unregister_table_function("versioned_func")

        conn.create_table_function(
            name="versioned_func",
            callable=func_v2,
            schema=schema,
            type="tuples",
        )

        result = conn.execute("SELECT * FROM versioned_func()").fetchall()
        assert result[0][0] == "version_2"


def test_unregister_multi(tmp_path):
    with duckdb.connect(tmp_path / "test.db") as conn:
        cursor1 = conn.cursor()
        cursor2 = conn.cursor()

        def test_func():
            return [("test_data", 1)]

        schema = {"name": sqltypes.VARCHAR, "id": sqltypes.INTEGER}

        cursor1.create_table_function(
            name="shared_func",
            callable=test_func,
            schema=schema,
            type="tuples",
        )

        result1 = cursor1.execute("SELECT * FROM shared_func()").fetchall()
        assert result1[0][0] == "test_data"

        result2 = cursor2.execute("SELECT * FROM shared_func()").fetchall()
        assert result2[0][0] == "test_data"

        cursor1.unregister_table_function("shared_func")

        result1 = cursor1.execute("SELECT * FROM shared_func()").fetchall()
        assert result1[0][0] == "test_data"

        result2 = cursor2.execute("SELECT * FROM shared_func()").fetchall()
        assert result2[0][0] == "test_data"

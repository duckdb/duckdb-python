import datetime
import enum
import math
import pathlib
import zoneinfo
from decimal import Decimal

import pytest

import duckdb
import numpy as np
import pandas as pd


class TestAppendDF:
    def test_df_to_table_append(self, duckdb_cursor):
        conn = duckdb.connect()
        conn.execute("Create table integers (i integer)")
        df_in = pd.DataFrame(
            {
                "numbers": [1, 2, 3, 4, 5],
            }
        )
        conn.append("integers", df_in)
        assert conn.execute("select count(*) from integers").fetchone()[0] == 5

    def test_append_by_name(self):
        con = duckdb.connect()
        con.execute("create table tbl (a integer, b bool, c varchar)")
        df_in = pd.DataFrame({"c": ["duck", "db"], "b": [False, True], "a": [4, 2]})
        # By default we append by position, causing the following exception:
        with pytest.raises(
            duckdb.ConversionException, match="Conversion Error: Could not convert string 'duck' to INT32"
        ):
            con.append("tbl", df_in)

        # When we use 'by_name' we instead append by name
        con.append("tbl", df_in, by_name=True)
        res = con.table("tbl").fetchall()
        assert res == [(4, False, "duck"), (2, True, "db")]

    def test_append_by_name_quoted(self):
        con = duckdb.connect()
        con.execute(
            """
            create table tbl ("needs to be quoted" integer, other varchar)
        """
        )
        df_in = pd.DataFrame({"needs to be quoted": [1, 2, 3]})
        con.append("tbl", df_in, by_name=True)
        res = con.table("tbl").fetchall()
        assert res == [(1, None), (2, None), (3, None)]

    def test_append_by_name_no_exact_match(self):
        con = duckdb.connect()
        con.execute("create table tbl (a integer, b bool)")
        df_in = pd.DataFrame({"c": ["a", "b"], "b": [True, False], "a": [42, 1337]})
        # Too many columns raises an error, because the columns cant be found in the targeted table
        with pytest.raises(duckdb.BinderException, match='Table "tbl" does not have a column with name "c"'):
            con.append("tbl", df_in, by_name=True)

        df_in = pd.DataFrame({"b": [False, False, False]})

        # Not matching all columns is not a problem, as they will be filled with NULL instead
        con.append("tbl", df_in, by_name=True)
        res = con.table("tbl").fetchall()
        # 'a' got filled by NULL automatically because it wasn't inserted into
        assert res == [(None, False), (None, False), (None, False)]

        # Empty the table
        con.execute("create or replace table tbl (a integer, b bool)")

        df_in = pd.DataFrame({"a": [1, 2, 3]})
        con.append("tbl", df_in, by_name=True)
        res = con.table("tbl").fetchall()
        # Also works for missing columns *after* the supplied ones
        assert res == [(1, None), (2, None), (3, None)]


ROWS = 3000
P2 = datetime.timezone(datetime.timedelta(hours=2))
NAIVE = datetime.datetime(2024, 5, 7, 7, 8, 9)
AWARE = NAIVE.replace(tzinfo=P2)
PATH = pathlib.PurePosixPath("/x")
LABEL = "a label longer than a string the engine keeps inline"
ENUM_VALUE = duckdb.Value(LABEL, f"ENUM('{LABEL}')")


class _Color(enum.Enum):
    RED = 1


def _outside_the_sample(sampled, odd):
    # The analyzer reads every third row of 3000, so the last row is never sampled.
    values = [sampled] * ROWS
    values[-1] = odd
    return pd.Series(values, dtype=object)


def _column(con, table="t"):
    return [value for (value,) in con.execute(f"SELECT a FROM {table} ORDER BY rowid").fetchall()]


def _mismatches(actual, expected):
    return [i for i, (a, e) in enumerate(zip(actual, expected, strict=True)) if a != e]


def _append_rows(con, table, values):
    with con.appender(table) as appender:
        for value in values:
            appender.append_row(value)


class TestAppendObjectColumnsToDeclaredTypes:
    @pytest.mark.parametrize(
        ("target", "column", "expected_first", "expected_last"),
        [
            pytest.param(
                "DOUBLE",
                _outside_the_sample(1, 2.5),
                1.0,
                2.5,
                id="int_sample_float_into_double",
            ),
            pytest.param(
                "DECIMAL(10,2)",
                _outside_the_sample(Decimal("1.5"), Decimal("1.25")),
                Decimal("1.50"),
                Decimal("1.25"),
                id="narrow_decimal_sample_into_wider_decimal",
            ),
            pytest.param(
                "TIMESTAMP",
                _outside_the_sample(datetime.date(2024, 5, 6), NAIVE),
                datetime.datetime(2024, 5, 6),
                NAIVE,
                id="date_sample_datetime_into_timestamp",
            ),
            pytest.param(
                "VARCHAR",
                _outside_the_sample(1, 2.5),
                "1",
                "2.5",
                id="int_sample_float_into_varchar",
            ),
            pytest.param(
                "VARCHAR",
                _outside_the_sample(1, "x"),
                "1",
                "x",
                id="int_sample_str_into_varchar",
            ),
            pytest.param(
                "VARCHAR",
                pd.Series([1, 2.5], dtype=object),
                "1",
                "2.5",
                id="fully_sampled_int_and_float_into_varchar",
            ),
            pytest.param(
                "VARCHAR",
                _outside_the_sample("a", b"ab"),
                "a",
                "ab",
                id="str_sample_bytes_into_varchar",
            ),
            pytest.param(
                "VARCHAR", _outside_the_sample("a", PATH), "a", str(PATH), id="path_into_varchar_reads_as_its_text"
            ),
            pytest.param(
                "VARCHAR",
                pd.Series([PATH, PATH], dtype=object),
                str(PATH),
                str(PATH),
                id="fully_sampled_path_into_varchar_reads_as_its_text",
            ),
            pytest.param(
                "VARCHAR",
                _outside_the_sample("a", _Color.RED),
                "a",
                str(_Color.RED),
                id="enum_member_into_varchar_reads_as_its_text",
            ),
            pytest.param(
                "VARCHAR",
                pd.Series([_Color.RED, _Color.RED], dtype=object),
                str(_Color.RED),
                str(_Color.RED),
                id="fully_sampled_enum_member_into_varchar_reads_as_its_text",
            ),
            pytest.param(
                "BIGINT",
                _outside_the_sample(1, 2**40),
                1,
                2**40,
                id="small_int_sample_large_int_into_bigint",
            ),
            pytest.param(
                "UBIGINT",
                _outside_the_sample(1, 2**63),
                1,
                2**63,
                id="int_past_the_signed_range_into_ubigint",
            ),
            pytest.param(
                "HUGEINT",
                _outside_the_sample(1, 2**70),
                1,
                2**70,
                id="int_past_64_bits_into_hugeint",
            ),
            pytest.param(
                "DOUBLE",
                _outside_the_sample(1, 2**70),
                1.0,
                float(2**70),
                id="int_past_64_bits_into_double",
            ),
            pytest.param(
                "UNION(i INTEGER, s VARCHAR)",
                _outside_the_sample(1, "x"),
                1,
                "x",
                id="int_sample_str_into_union",
            ),
            pytest.param(
                "MAP(VARCHAR, INTEGER)",
                _outside_the_sample({"a": 1}, {"b": 2}),
                {"a": 1},
                {"b": 2},
                id="dict_with_other_key_into_map",
            ),
            pytest.param(
                "JSON",
                _outside_the_sample(1, {"a": 1}),
                "1",
                '{"a":1}',
                id="int_sample_dict_into_json",
            ),
            pytest.param(
                "INTEGER",
                _outside_the_sample(1, True),
                1,
                1,
                id="int_sample_bool_into_integer",
            ),
            pytest.param(
                "DOUBLE",
                _outside_the_sample(1, True),
                1.0,
                1.0,
                id="int_sample_bool_into_double",
            ),
            pytest.param("BOOLEAN", _outside_the_sample(True, 1), True, True, id="bool_sample_int_into_boolean"),
            pytest.param(
                "INTEGER[3]",
                _outside_the_sample([1, 2, 3], [4, 5, 6]),
                (1, 2, 3),
                (4, 5, 6),
                id="lists_into_array",
            ),
            pytest.param("DOUBLE", _outside_the_sample(1.5, float("nan")), 1.5, None, id="nan_into_double_is_null"),
            pytest.param("DOUBLE", _outside_the_sample(1.5, None), 1.5, None, id="none_into_double_is_null"),
            pytest.param("DOUBLE", _outside_the_sample(1.5, np.ma.masked), 1.5, None, id="masked_into_double_is_null"),
            pytest.param(
                "DOUBLE",
                _outside_the_sample(1.5, np.ma.array(42.0, mask=True)),
                1.5,
                None,
                id="masked_zero_dimensional_array_into_double_is_null",
            ),
            pytest.param("INTEGER", _outside_the_sample(1, np.int64(2)), 1, 2, id="numpy_int_into_integer"),
            pytest.param("DOUBLE", _outside_the_sample(1.5, np.float32(2.5)), 1.5, 2.5, id="numpy_float_into_double"),
            pytest.param(
                "BOOLEAN", _outside_the_sample(True, np.bool_(False)), True, False, id="numpy_bool_into_boolean"
            ),
            pytest.param(
                "TIMESTAMP",
                _outside_the_sample(NAIVE, np.datetime64("2024-05-08T01:02:03.123456789", "ns")),
                NAIVE,
                datetime.datetime(2024, 5, 8, 1, 2, 3, 123456),
                id="numpy_datetime_into_timestamp",
            ),
            pytest.param(
                "TIMESTAMP",
                _outside_the_sample(NAIVE, np.array("2024-05-08T01:02:03.123456789", dtype="datetime64[ns]")),
                NAIVE,
                datetime.datetime(2024, 5, 8, 1, 2, 3, 123456),
                id="numpy_zero_dimensional_datetime_into_timestamp",
            ),
            pytest.param(
                "STRUCT(x VARCHAR)",
                _outside_the_sample({"x": "abc"}, None),
                {"x": "abc"},
                None,
                id="none_into_struct_of_text_is_null",
            ),
            pytest.param(
                "STRUCT(x UNION(i INTEGER, s VARCHAR))",
                _outside_the_sample({"x": 1}, None),
                {"x": 1},
                None,
                id="none_into_struct_of_union_is_null",
            ),
            pytest.param(
                "VARCHAR",
                pd.Series([ENUM_VALUE] * ROWS, dtype=object),
                LABEL,
                LABEL,
                id="enum_values_into_varchar_keep_their_text",
            ),
            pytest.param("DOUBLE", _outside_the_sample(1.5, pd.NA), 1.5, None, id="pandas_na_into_double_is_null"),
            pytest.param("VARCHAR", _outside_the_sample("a", float("nan")), "a", None, id="nan_into_varchar_is_null"),
            pytest.param("INTEGER", _outside_the_sample(1, float("nan")), 1, None, id="nan_into_integer_is_null"),
            pytest.param(
                "INTEGER",
                pd.Series([1, float("nan")], dtype=object),
                1,
                None,
                id="fully_sampled_nan_into_integer_is_null",
            ),
            pytest.param(
                "DATE",
                _outside_the_sample(datetime.date(2024, 5, 6), float("nan")),
                datetime.date(2024, 5, 6),
                None,
                id="nan_into_date_is_null",
            ),
            pytest.param(
                "TIMESTAMP", _outside_the_sample(NAIVE, float("nan")), NAIVE, None, id="nan_into_timestamp_is_null"
            ),
            pytest.param("BOOLEAN", _outside_the_sample(True, float("nan")), True, None, id="nan_into_boolean_is_null"),
            pytest.param(
                "DOUBLE[]", _outside_the_sample([1.5], [float("nan")]), [1.5], [None], id="nan_in_a_list_is_null"
            ),
            pytest.param(
                "STRUCT(x DOUBLE)",
                _outside_the_sample({"x": 1.5}, {"x": float("nan")}),
                {"x": 1.5},
                {"x": None},
                id="nan_in_a_struct_is_null",
            ),
            pytest.param(
                "VARCHAR",
                _outside_the_sample("a", np.datetime64("NaT", "us")),
                "a",
                None,
                id="datetime64_nat_into_varchar_is_null",
            ),
        ],
    )
    def test_object_value_converts_into_the_table_column_type(
        self, duckdb_cursor, target, column, expected_first, expected_last
    ):
        duckdb_cursor.execute(f"CREATE TABLE t (a {target})")
        duckdb_cursor.append("t", pd.DataFrame({"a": column}))
        values = _column(duckdb_cursor)
        assert (values[0], values[-1]) == (expected_first, expected_last)

    @pytest.mark.parametrize(
        ("target", "column", "match"),
        [
            pytest.param(
                "JSON",
                _outside_the_sample(1, "x"),
                "Malformed JSON",
                id="text_that_is_not_json",
            ),
            pytest.param(
                "INTEGER",
                _outside_the_sample(1, "x"),
                "'x'",
                id="text_into_integer",
            ),
            pytest.param(
                "INTEGER",
                _outside_the_sample(1, 2**40),
                "1099511627776",
                id="int_out_of_range",
            ),
            pytest.param(
                "FLOAT",
                _outside_the_sample(1, 1e300),
                "out of range",
                id="float_out_of_range",
            ),
            pytest.param(
                "STRUCT(x INTEGER)",
                _outside_the_sample({"x": 1}, (2,)),
                "STRUCT",
                id="tuple_into_struct",
            ),
            pytest.param(
                "TIME",
                _outside_the_sample(datetime.time(1, 2, 3), AWARE),
                "TIME",
                id="aware_datetime_into_time",
            ),
            pytest.param("ENUM('a', 'b')", _outside_the_sample("a", "c"), "'c'", id="text_outside_the_enum"),
            pytest.param("INTEGER", pd.Series(["duck", "db"], dtype=object), "'duck'", id="text_column_into_integer"),
            pytest.param("UTINYINT", _outside_the_sample(1, -1), "out of range", id="negative_int_into_utinyint"),
            pytest.param("TINYINT", _outside_the_sample(1, 200), "out of range", id="int_out_of_tinyint_range"),
            pytest.param("DATE", _outside_the_sample(1, 2), "DATE", id="int_column_into_date"),
        ],
    )
    def test_object_value_the_table_column_type_refuses(self, duckdb_cursor, target, column, match):
        duckdb_cursor.execute(f"CREATE TABLE t (a {target})")
        frame = pd.DataFrame({"a": column})
        with pytest.raises(duckdb.ConversionException, match=match):
            duckdb_cursor.append("t", frame)

    def test_numpy_long_double_is_refused_like_the_appender(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a DOUBLE)")
        frame = pd.DataFrame({"a": _outside_the_sample(1.5, np.longdouble("2.5"))})
        with pytest.raises(duckdb.NotImplementedException, match="longdouble"):
            duckdb_cursor.append("t", frame)

    def test_keys_that_collapse_in_a_nested_map_are_refused(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a STRUCT(m MAP(INTEGER, INTEGER)))")
        # The same key set in every row, so one cast runs over the whole vector
        frame = pd.DataFrame({"a": _outside_the_sample({"m": {"01": 1, "1": 2}}, {"m": {"01": 10, "1": 20}})})
        with pytest.raises(duckdb.InvalidInputException, match=r"(?i)duplicat|unique"):
            duckdb_cursor.append("t", frame)

    def test_text_with_a_lone_surrogate_names_it(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a VARCHAR)")
        frame = pd.DataFrame({"a": _outside_the_sample("a", pathlib.PurePosixPath("/\udcff"))})
        with pytest.raises(duckdb.InvalidInputException, match="surrogate"):
            duckdb_cursor.append("t", frame)

    def test_text_column_mixing_str_with_other_cells_appends_each_as_text(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a VARCHAR)")
        cells = ["a", 5, 6, "b", None, 7, np.str_("c"), "\u00e9 \U0001f600"]
        duckdb_cursor.append("t", pd.DataFrame({"a": pd.Series(cells, dtype=object)}))
        assert duckdb_cursor.table("t").fetchall() == [
            ("a",),
            ("5",),
            ("6",),
            ("b",),
            (None,),
            ("7",),
            ("c",),
            ("\u00e9 \U0001f600",),
        ]

    def test_text_with_a_lone_surrogate_in_a_str_names_it(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a VARCHAR)")
        frame = pd.DataFrame({"a": pd.Series(["a", "b\udcff"], dtype=object)})
        with pytest.raises(duckdb.InvalidInputException, match="surrogate"):
            duckdb_cursor.append("t", frame)

    def test_null_row_among_struct_values_the_cast_converts(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a STRUCT(x INTEGER))")
        # Text fields cast into INTEGER, so the whole vector goes through one cast with the NULL row inside it
        values = [{"x": "1"}] * 1000 + [None] + [{"x": "2"}] * 1000
        duckdb_cursor.append("t", pd.DataFrame({"a": pd.Series(values, dtype=object)}))
        rows = duckdb_cursor.execute("SELECT a FROM t ORDER BY rowid OFFSET 999 LIMIT 3").fetchall()
        assert rows == [({"x": 1},), (None,), ({"x": 2},)]
        stats = duckdb_cursor.execute("SELECT stats(a) FROM t LIMIT 1").fetchall()[0][0]
        assert stats["child_stats"]["x"]["has_null"] is True

    def test_null_row_of_a_union_column_is_null(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a UNION(i INTEGER, s VARCHAR))")
        duckdb_cursor.append("t", pd.DataFrame({"a": _outside_the_sample(1, None)}))
        rows = duckdb_cursor.execute("SELECT a IS NULL, union_tag(a) FROM t ORDER BY rowid OFFSET 2998").fetchall()
        assert rows == [(False, "i"), (True, None)]

    def test_null_row_of_an_array_column_has_null_elements(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a INTEGER[3])")
        # A vector of rows before the NULL, so the NULL row's element slots hold values from an earlier chunk
        values = [[777, 777, 777]] * 2048 + [None] + [[5, 5, 5]] * 100
        duckdb_cursor.append("t", pd.DataFrame({"a": pd.Series(values, dtype=object)}))
        rows = duckdb_cursor.execute("SELECT a FROM t ORDER BY rowid OFFSET 2047 LIMIT 3").fetchall()
        assert rows == [((777, 777, 777),), (None,), ((5, 5, 5),)]
        stats = duckdb_cursor.execute("SELECT stats(a) FROM t LIMIT 1").fetchall()[0][0]
        assert stats["child_stats"]["has_null"] is True

    @pytest.mark.parametrize(
        ("zone", "expected"),
        [
            pytest.param(
                "UTC",
                datetime.datetime(2024, 5, 7, 5, 8, 9),
                id="UTC",
            ),
            pytest.param("Europe/Amsterdam", datetime.datetime(2024, 5, 7, 7, 8, 9), id="Europe_Amsterdam"),
            pytest.param(
                "America/New_York",
                datetime.datetime(2024, 5, 7, 1, 8, 9),
                id="America_New_York",
            ),
        ],
    )
    def test_aware_datetime_into_timestamp_reads_in_the_session_time_zone(self, duckdb_cursor, zone, expected):
        duckdb_cursor.execute(f"SET TimeZone = '{zone}'")
        duckdb_cursor.execute("CREATE TABLE t (a TIMESTAMP)")
        duckdb_cursor.append("t", pd.DataFrame({"a": _outside_the_sample(NAIVE, AWARE)}))
        values = _column(duckdb_cursor)
        assert (values[0], values[-1]) == (NAIVE, expected)

    @pytest.mark.parametrize("zone", ["UTC", "Europe/Amsterdam"], ids=["UTC", "Europe_Amsterdam"])
    @pytest.mark.parametrize("naive", [NAIVE, pd.Timestamp(NAIVE)], ids=["datetime", "Timestamp"])
    def test_naive_datetime_into_timestamptz_reads_in_the_session_time_zone(self, duckdb_cursor, zone, naive):
        duckdb_cursor.execute(f"SET TimeZone = '{zone}'")
        duckdb_cursor.execute("CREATE TABLE t (a TIMESTAMPTZ)")
        duckdb_cursor.append("t", pd.DataFrame({"a": _outside_the_sample(AWARE, naive)}))
        values = _column(duckdb_cursor)
        assert (values[0], values[-1]) == (AWARE, NAIVE.replace(tzinfo=zoneinfo.ZoneInfo(zone)))

    @pytest.mark.parametrize(
        "zone",
        [
            pytest.param("UTC", id="UTC"),
            pytest.param(
                "Europe/Amsterdam",
                id="Europe_Amsterdam",
            ),
            pytest.param(
                "America/New_York",
                id="America_New_York",
            ),
        ],
    )
    def test_date_into_timestamptz_reads_as_midnight_in_the_session_time_zone(self, duckdb_cursor, zone):
        duckdb_cursor.execute(f"SET TimeZone = '{zone}'")
        duckdb_cursor.execute("CREATE TABLE t (a TIMESTAMPTZ)")
        duckdb_cursor.append("t", pd.DataFrame({"a": _outside_the_sample(AWARE, datetime.date(2024, 5, 6))}))
        values = _column(duckdb_cursor)
        assert (values[0], values[-1]) == (AWARE, datetime.datetime(2024, 5, 6, tzinfo=zoneinfo.ZoneInfo(zone)))

    def test_by_name_declares_each_column_from_its_name(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a DOUBLE, b VARCHAR)")
        frame = pd.DataFrame({"b": _outside_the_sample(1, "x"), "A": _outside_the_sample(1, 2.5)})
        duckdb_cursor.append("t", frame, by_name=True)
        assert duckdb_cursor.execute("SELECT a, b FROM t ORDER BY rowid OFFSET 2999").fetchall() == [(2.5, "x")]

    def test_by_position_skips_generated_columns(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a DOUBLE, g DOUBLE AS (a * 2), b VARCHAR)")
        frame = pd.DataFrame({"x": _outside_the_sample(1, 2.5), "y": _outside_the_sample(1, "x")})
        duckdb_cursor.append("t", frame)
        assert duckdb_cursor.execute("SELECT a, g, b FROM t ORDER BY rowid OFFSET 2999").fetchall() == [(2.5, 5.0, "x")]

    def test_temporary_table_shadows_the_table_of_the_same_name(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE t (a INTEGER)")
        duckdb_cursor.execute("CREATE TEMPORARY TABLE t (a DOUBLE)")
        duckdb_cursor.append("t", pd.DataFrame({"a": _outside_the_sample(1, 2.5)}))
        written = duckdb_cursor.execute("SELECT a FROM temp.t ORDER BY rowid OFFSET 2999").fetchall()
        # main.t resolves to the temporary table too, so the persistent one is named by its catalog.
        untouched = duckdb_cursor.execute("SELECT count(*) FROM memory.main.t").fetchall()
        assert (written, untouched) == ([(2.5,)], [(0,)])

    def test_table_is_found_through_the_search_path(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE SCHEMA s")
        duckdb_cursor.execute("CREATE TABLE s.t (a DOUBLE)")
        duckdb_cursor.execute("CREATE TABLE main.t (a INTEGER)")
        duckdb_cursor.execute("SET search_path = 's,main'")
        duckdb_cursor.append("t", pd.DataFrame({"a": _outside_the_sample(1, 2.5)}))
        written = duckdb_cursor.execute("SELECT a FROM s.t ORDER BY rowid OFFSET 2999").fetchall()
        untouched = duckdb_cursor.execute("SELECT count(*) FROM memory.main.t").fetchall()
        assert (written, untouched) == ([(2.5,)], [(0,)])

    def test_table_name_with_a_dot_is_one_identifier(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE SCHEMA s")
        duckdb_cursor.execute('CREATE TABLE "s.t" (a DOUBLE)')
        duckdb_cursor.execute("CREATE TABLE s.t (a INTEGER)")
        duckdb_cursor.append("s.t", pd.DataFrame({"a": _outside_the_sample(1, 2.5)}))
        written = duckdb_cursor.execute('SELECT a FROM "s.t" ORDER BY rowid OFFSET 2999').fetchall()
        untouched = duckdb_cursor.execute("SELECT count(*) FROM s.t").fetchall()
        assert (written, untouched) == ([(2.5,)], [(0,)])

    @pytest.mark.parametrize(
        ("table", "columns"),
        [
            pytest.param("a DOUBLE, b DOUBLE", 1, id="fewer_columns"),
            pytest.param("a DOUBLE", 2, id="more_columns"),
        ],
    )
    def test_column_count_mismatch_is_refused_by_the_insert(self, duckdb_cursor, table, columns):
        duckdb_cursor.execute(f"CREATE TABLE t ({table})")
        frame = pd.DataFrame({f"c{i}": _outside_the_sample(1, 2.5) for i in range(columns)})
        with pytest.raises(duckdb.BinderException, match="values were supplied"):
            duckdb_cursor.append("t", frame)


class TestAppendMatchesTheAppender:
    @pytest.mark.parametrize(
        ("target", "column"),
        [
            pytest.param(
                "DOUBLE",
                _outside_the_sample(1, 2.5),
                id="int_sample_float_into_double",
            ),
            pytest.param(
                "DECIMAL(10,2)",
                _outside_the_sample(Decimal("1.5"), Decimal("1.25")),
                id="decimals",
            ),
            pytest.param(
                "VARCHAR",
                _outside_the_sample(1, 2.5),
                id="numbers_into_varchar",
            ),
            pytest.param(
                "TIMESTAMP",
                _outside_the_sample(NAIVE, AWARE),
                id="aware_datetime_into_timestamp",
            ),
            pytest.param(
                "TIMESTAMPTZ",
                _outside_the_sample(AWARE, NAIVE),
                id="naive_datetime_into_timestamptz",
            ),
            pytest.param(
                "UNION(i INTEGER, s VARCHAR)",
                _outside_the_sample(1, "x"),
                id="union",
            ),
            pytest.param(
                "INTEGER",
                _outside_the_sample(1, True),
                id="bool_into_integer",
            ),
            pytest.param("BOOLEAN", _outside_the_sample(True, 1), id="int_into_boolean"),
            pytest.param("INTEGER[3]", _outside_the_sample([1, 2, 3], [4, 5, 6]), id="lists_into_array"),
            pytest.param("DOUBLE", _outside_the_sample(1.5, float("nan")), id="nan_into_double"),
        ],
    )
    def test_append_writes_what_the_appender_writes(self, duckdb_cursor, target, column):
        duckdb_cursor.execute("SET TimeZone = 'UTC'")
        duckdb_cursor.execute(f"CREATE TABLE by_rows (a {target})")
        duckdb_cursor.execute(f"CREATE TABLE t (a {target})")
        _append_rows(duckdb_cursor, "by_rows", column)
        # The appender keeps NaN as a value; an object column reads it as NULL.
        expected = [None if isinstance(v, float) and math.isnan(v) else v for v in _column(duckdb_cursor, "by_rows")]
        duckdb_cursor.append("t", pd.DataFrame({"a": column}))
        assert _mismatches(_column(duckdb_cursor), expected) == []

    @pytest.mark.parametrize(
        ("target", "column"),
        [
            pytest.param(
                "JSON",
                _outside_the_sample(1, "x"),
                id="text_that_is_not_json",
            ),
            pytest.param(
                "INTEGER",
                _outside_the_sample(1, 2**40),
                id="int_out_of_range",
            ),
            pytest.param(
                "STRUCT(x INTEGER)",
                _outside_the_sample({"x": 1}, (2,)),
                id="tuple_into_struct",
            ),
            pytest.param("ENUM('a', 'b')", _outside_the_sample("a", "c"), id="text_outside_the_enum"),
        ],
    )
    def test_append_refuses_what_the_appender_refuses(self, duckdb_cursor, target, column):
        duckdb_cursor.execute(f"CREATE TABLE by_rows (a {target})")
        duckdb_cursor.execute(f"CREATE TABLE t (a {target})")
        with pytest.raises(duckdb.Error) as refused:
            _append_rows(duckdb_cursor, "by_rows", column)
        frame = pd.DataFrame({"a": column})
        with pytest.raises(type(refused.value)):
            duckdb_cursor.append("t", frame)


class TestRegisterDeclared:
    def test_declared_type_replaces_the_sampled_type(self, duckdb_cursor):
        frame = pd.DataFrame({"a": _outside_the_sample(1, 2.5), "b": _outside_the_sample("x", "y")})
        duckdb_cursor._register_declared("declared", frame, [duckdb.sqltypes.DOUBLE, None])
        rel = duckdb_cursor.sql("SELECT a, b FROM declared WHERE a = 2.5")
        assert rel.types == [duckdb.sqltypes.DOUBLE, duckdb.sqltypes.VARCHAR]
        assert rel.fetchall() == [(2.5, "y")]

    def test_one_type_per_column_or_refused(self, duckdb_cursor):
        frame = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(duckdb.InvalidInputException, match="1 types given for a DataFrame of 2 columns"):
            duckdb_cursor._register_declared("declared", frame, [duckdb.sqltypes.DOUBLE])

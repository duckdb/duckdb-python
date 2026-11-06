import datetime
import gc
import platform
import re
import sys
import threading
import time
import weakref
from io import BytesIO, StringIO
from pathlib import Path
from typing import NoReturn

import pytest

import duckdb
from duckdb import CSVLineTerminator


def TestFile(name):
    filename = Path(__file__).parent / ".." / "data" / name
    return str(filename)


@pytest.fixture
def create_temp_csv(tmp_path):
    # Create temporary CSV files
    file1_content = """1
2
3"""
    file2_content = """4
5
6"""
    file1_path = tmp_path / "file1.csv"
    file2_path = tmp_path / "file2.csv"

    file1_path.write_text(file1_content)
    file2_path.write_text(file2_content)

    return file1_path, file2_path


class TestReadCSV:
    def test_using_connection_wrapper(self):
        rel = duckdb.read_csv(TestFile("category.csv"))
        res = rel.fetchall()[0]
        print(res)
        assert res == (1, "Action", datetime.datetime(2006, 2, 15, 4, 46, 27))

    def test_using_connection_wrapper_with_keyword(self):
        rel = duckdb.read_csv(TestFile("category.csv"), dtype={"category_id": "string"})
        res = rel.fetchall()[0]
        print(res)
        assert res == ("1", "Action", datetime.datetime(2006, 2, 15, 4, 46, 27))

    def test_no_options(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"))
        res = rel.fetchall()[0]
        print(res)
        assert res == (1, "Action", datetime.datetime(2006, 2, 15, 4, 46, 27))

    def test_dtype(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"), dtype={"category_id": "string"})
        res = rel.fetchall()[0]
        print(res)
        assert res == ("1", "Action", datetime.datetime(2006, 2, 15, 4, 46, 27))

    def test_dtype_as_list(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"), dtype=["string"])
        res = rel.fetchall()[0]
        print(res)
        assert res == ("1", "Action", datetime.datetime(2006, 2, 15, 4, 46, 27))

        rel = duckdb_cursor.read_csv(TestFile("category.csv"), dtype=["double"])
        res = rel.fetchall()[0]
        print(res)
        assert res == (1.0, "Action", datetime.datetime(2006, 2, 15, 4, 46, 27))

    def test_sep(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"), sep=" ")
        res = rel.fetchall()[0]
        print(res)
        assert res == ("1|Action|2006-02-15", datetime.time(4, 46, 27))

    def test_delimiter(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"), delimiter=" ")
        res = rel.fetchall()[0]
        print(res)
        assert res == ("1|Action|2006-02-15", datetime.time(4, 46, 27))

    def test_delimiter_and_sep(self, duckdb_cursor):
        with pytest.raises(duckdb.InvalidInputException, match="read_csv takes either 'delimiter' or 'sep', not both"):
            duckdb_cursor.read_csv(TestFile("category.csv"), delimiter=" ", sep=" ")

    def test_header_true(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"))
        res = rel.fetchall()[0]
        print(res)
        assert res == (1, "Action", datetime.datetime(2006, 2, 15, 4, 46, 27))

    def test_header_false(self, duckdb_cursor):
        duckdb_cursor.read_csv(TestFile("category.csv"), header=False)

    def test_na_values(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"), na_values="Action")
        res = rel.fetchall()[0]
        print(res)
        assert res == (1, None, datetime.datetime(2006, 2, 15, 4, 46, 27))

    def test_na_values_list(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"), na_values=["Action", "Animation"])
        rows = rel.fetchall()
        assert rows[0] == (1, None, datetime.datetime(2006, 2, 15, 4, 46, 27))
        assert rows[1] == (2, None, datetime.datetime(2006, 2, 15, 4, 46, 27))

    def test_skiprows(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"), skiprows=1)
        res = rel.fetchall()[0]
        print(res)
        assert res == (1, "Action", datetime.datetime(2006, 2, 15, 4, 46, 27))

    # We want to detect this at bind time
    def test_compression_wrong(self, duckdb_cursor):
        with pytest.raises(duckdb.Error, match="Input is not a GZIP stream"):
            duckdb_cursor.read_csv(TestFile("category.csv"), compression="gzip")

    def test_quotechar(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("unquote_without_delimiter.csv"), quotechar="", header=False)
        res = rel.fetchall()[0]
        print(res)
        assert res == ('"AAA"BB',)

    def test_quote(self, duckdb_cursor):
        with pytest.raises(
            duckdb.Error, match='The methods read_csv and read_csv_auto do not have the "quote" argument'
        ):
            duckdb_cursor.read_csv(TestFile("unquote_without_delimiter.csv"), quote="", header=False)

    def test_escapechar(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("quote_escape.csv"), escapechar=";", header=False)
        res = rel.limit(1, 1).fetchall()[0]
        print(res)
        assert res == ("345", "TEST6", '"text""2""text"')

    def test_encoding_wrong(self, duckdb_cursor):
        with pytest.raises(
            duckdb.BinderException, match="Copy is only supported for UTF-8 encoded files, ENCODING 'UTF-8'"
        ):
            duckdb_cursor.read_csv(TestFile("quote_escape.csv"), encoding=";")

    def test_encoding_correct(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("quote_escape.csv"), encoding="UTF-8")
        res = rel.limit(1, 1).fetchall()[0]
        print(res)
        assert res == (345, "TEST6", 'text"2"text')

    def test_date_format_as_datetime(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("datetime.csv"))
        res = rel.fetchall()[0]
        print(res)
        assert res == (
            123,
            "TEST2",
            datetime.time(12, 12, 12),
            datetime.date(2000, 1, 1),
            datetime.datetime(2000, 1, 1, 12, 12),
        )

    def test_date_format_as_date(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("datetime.csv"), date_format="%Y-%m-%d")
        res = rel.fetchall()[0]
        print(res)
        assert res == (
            123,
            "TEST2",
            datetime.time(12, 12, 12),
            datetime.date(2000, 1, 1),
            datetime.datetime(2000, 1, 1, 12, 12),
        )

    def test_timestamp_format(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("datetime.csv"), timestamp_format="%Y-%m-%d %H:%M:%S")
        res = rel.fetchall()[0]
        assert res == (
            123,
            "TEST2",
            datetime.time(12, 12, 12),
            datetime.date(2000, 1, 1),
            datetime.datetime(2000, 1, 1, 12, 12),
        )

    def test_sample_size_correct(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("problematic.csv"), sample_size=-1)
        res = rel.fetchall()[0]
        print(res)
        assert res == ("1", "1", "1")

    def test_all_varchar(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"), all_varchar=True)
        res = rel.fetchall()[0]
        print(res)
        assert res == ("1", "Action", "2006-02-15 04:46:27")

    def test_null_padding(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("nullpadding.csv"), null_padding=False, header=False)
        res = rel.fetchall()
        assert res == [
            ("# this file has a bunch of gunk at the top",),
            ("one,two,three,four",),
            ("1,a,alice",),
            ("2,b,bob",),
        ]

        rel = duckdb_cursor.read_csv(TestFile("nullpadding.csv"), null_padding=True, header=False)
        res = rel.fetchall()
        assert res == [
            ("# this file has a bunch of gunk at the top", None, None, None),
            ("one", "two", "three", "four"),
            ("1", "a", "alice", None),
            ("2", "b", "bob", None),
        ]

        rel = duckdb.read_csv(TestFile("nullpadding.csv"), null_padding=False, header=False)
        res = rel.fetchall()
        assert res == [
            ("# this file has a bunch of gunk at the top",),
            ("one,two,three,four",),
            ("1,a,alice",),
            ("2,b,bob",),
        ]

        rel = duckdb.read_csv(TestFile("nullpadding.csv"), null_padding=True, header=False)
        res = rel.fetchall()
        assert res == [
            ("# this file has a bunch of gunk at the top", None, None, None),
            ("one", "two", "three", "four"),
            ("1", "a", "alice", None),
            ("2", "b", "bob", None),
        ]

        rel = duckdb_cursor.from_csv_auto(TestFile("nullpadding.csv"), null_padding=False, header=False)
        res = rel.fetchall()
        assert res == [
            ("# this file has a bunch of gunk at the top",),
            ("one,two,three,four",),
            ("1,a,alice",),
            ("2,b,bob",),
        ]

        rel = duckdb_cursor.from_csv_auto(TestFile("nullpadding.csv"), null_padding=True, header=False)
        res = rel.fetchall()
        assert res == [
            ("# this file has a bunch of gunk at the top", None, None, None),
            ("one", "two", "three", "four"),
            ("1", "a", "alice", None),
            ("2", "b", "bob", None),
        ]

    def test_normalize_names(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"), normalize_names=False)
        df = rel.df()
        column_names = list(df.columns.values)
        # The names are not normalized, so they are capitalized
        assert "CATEGORY_ID" in column_names

        rel = duckdb_cursor.read_csv(TestFile("category.csv"), normalize_names=True)
        df = rel.df()
        column_names = list(df.columns.values)
        # The capitalized names are normalized to lowercase instead
        assert "CATEGORY_ID" not in column_names

    def test_filename(self, duckdb_cursor):
        rel = duckdb_cursor.read_csv(TestFile("category.csv"), filename=False)
        df = rel.df()
        column_names = list(df.columns.values)
        # The filename is not included in the returned columns
        assert "filename" not in column_names

        rel = duckdb_cursor.read_csv(TestFile("category.csv"), filename=True)
        df = rel.df()
        column_names = list(df.columns.values)
        # The filename is included in the returned columns
        assert "filename" in column_names

    def test_read_pathlib_path(self, duckdb_cursor):
        pathlib = pytest.importorskip("pathlib")
        path = pathlib.Path(TestFile("category.csv"))
        rel = duckdb_cursor.read_csv(path)
        res = rel.fetchall()[0]
        print(res)
        assert res == (1, "Action", datetime.datetime(2006, 2, 15, 4, 46, 27))

    def test_read_filelike(self, duckdb_cursor):
        pytest.importorskip("fsspec")

        string = StringIO("c1,c2,c3\na,b,c")
        res = duckdb_cursor.read_csv(string).fetchall()
        assert res == [("a", "b", "c")]

    def test_read_filelike_rel_out_of_scope(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")

        def keep_in_scope():
            string = StringIO("c1,c2,c3\na,b,c")
            # Create a ReadCSVRelation on a file-like object
            # this will add the object to our internal object filesystem
            rel = duckdb_cursor.read_csv(string)
            # The file-like object will still exist, so we can execute this later
            return rel

        def close_scope():
            string = StringIO("c1,c2,c3\na,b,c")
            # Create a ReadCSVRelation on a file-like object
            # this will add the object to our internal object filesystem
            res = duckdb_cursor.read_csv(string).fetchall()
            # When the relation goes out of scope - we delete the file-like object from our filesystem
            return res

        relation = keep_in_scope()
        res = relation.fetchall()

        res2 = close_scope()
        assert res == res2

    def test_filelike_bytesio(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        string = BytesIO(b"c1,c2,c3\na,b,c")
        res = duckdb_cursor.read_csv(string).fetchall()
        assert res == [("a", "b", "c")]

    def test_filelike_exception(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")

        class ReadError:
            def __init__(self) -> None:
                pass

            def read(self, amount=-1) -> NoReturn:
                raise ValueError(amount)

            def seek(self, loc) -> int:
                return 0

        class SeekError:
            def __init__(self) -> None:
                pass

            def read(self, amount=-1) -> bytes:
                return b"test"

            def seek(self, loc) -> NoReturn:
                raise ValueError(loc)

        # The MemoryFileSystem reads the content into another object, so this fails instantly
        obj = ReadError()
        with pytest.raises(ValueError, match="-1"):
            duckdb_cursor.read_csv(obj).fetchall()

        # For that same reason, this will not error, because the data is retrieved with 'read' and then
        # SeekError is never used again
        obj = SeekError()
        duckdb_cursor.read_csv(obj).fetchall()

    def test_filelike_custom(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")

        class CustomIO:
            def __init__(self) -> None:
                self.loc = 0

            def seek(self, loc):
                self.loc = loc
                return loc

            def read(self, amount=-1):
                file = b"c1,c2,c3\na,b,c"
                if amount == -1:
                    return file
                out = file[self.loc : self.loc + amount : 1]
                self.loc += amount
                return out

        obj = CustomIO()
        res = duckdb_cursor.read_csv(obj).fetchall()
        assert res == [("a", "b", "c")]

    def test_filelike_non_readable(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        obj = 5
        with pytest.raises(TypeError, match="Can not read from a non file-like object"):
            duckdb_cursor.read_csv(obj).fetchall()

    def test_filelike_none(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        obj = None
        with pytest.raises(TypeError, match="Can not read from a non file-like object"):
            duckdb_cursor.read_csv(obj).fetchall()

    @pytest.mark.skip(reason="depends on garbage collector behaviour, and sporadically breaks in CI")
    def test_internal_object_filesystem_cleanup(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")

        class CountedObject(StringIO):
            instance_count = 0

            def __init__(self, str) -> None:
                CountedObject.instance_count += 1
                super().__init__(str)

            def __del__(self) -> None:
                CountedObject.instance_count -= 1

        def scoped_objects(duckdb_cursor) -> None:
            obj = CountedObject("a,b,c")
            rel1 = duckdb_cursor.read_csv(obj)
            assert rel1.fetchall() == [
                (
                    "a",
                    "b",
                    "c",
                )
            ]
            assert CountedObject.instance_count == 1

            obj = CountedObject("a,b,c")
            rel2 = duckdb_cursor.read_csv(obj)
            assert rel2.fetchall() == [
                (
                    "a",
                    "b",
                    "c",
                )
            ]
            assert CountedObject.instance_count == 2

            obj = CountedObject("a,b,c")
            rel3 = duckdb_cursor.read_csv(obj)
            assert rel3.fetchall() == [
                (
                    "a",
                    "b",
                    "c",
                )
            ]
            assert CountedObject.instance_count == 3

        assert CountedObject.instance_count == 0
        scoped_objects(duckdb_cursor)
        assert CountedObject.instance_count == 0

    def test_read_csv_glob(self, tmp_path, create_temp_csv):
        # Use the temporary file paths to read CSV files
        con = duckdb.connect()
        rel = con.read_csv(f"{tmp_path}/file*.csv")  # noqa: F841
        res = con.sql("select * from rel order by all").fetchall()
        assert res == [(1,), (2,), (3,), (4,), (5,), (6,)]

    @pytest.mark.xfail(condition=platform.system() == "Emscripten", reason="time zones not working")
    def test_read_csv_combined(self, duckdb_cursor):
        CSV_FILE = TestFile("stress_test.csv")
        COLUMNS = {
            "result": "VARCHAR",
            "table": "BIGINT",
            "_time": "TIMESTAMPTZ",
            "_measurement": "VARCHAR",
            "bench_test": "VARCHAR",
            "flight_id": "VARCHAR",
            "flight_status": "VARCHAR",
            "log_level": "VARCHAR",
            "sys_uuid": "VARCHAR",
            "message": "VARCHAR",
        }

        rel = duckdb.read_csv(CSV_FILE, skiprows=1, delimiter=",", quotechar='"', escapechar="\\", dtype=COLUMNS)
        res = rel.fetchall()

        rel2 = duckdb_cursor.sql(rel.sql_query())
        res2 = rel2.fetchall()

        # Assert that the results are the same
        assert res == res2

        # And assert that the columns and types of the relations are the same
        assert rel.columns == rel2.columns
        assert rel.types == rel2.types

    def test_read_csv_names(self, tmp_path):
        file = tmp_path / "file.csv"
        file.write_text("one,two,three,four\n1,2,3,4\n1,2,3,4\n1,2,3,4")

        con = duckdb.connect()
        rel = con.read_csv(str(file), names=["a", "b", "c"])
        assert rel.columns == ["a", "b", "c", "four"]

        with pytest.raises(duckdb.InvalidInputException, match="read_csv only accepts 'names' as a list of strings"):
            con.read_csv(file, names=True)

        with pytest.raises(duckdb.InvalidInputException, match="not possible to detect the CSV Header"):
            con.read_csv(file, names=["a", "b", "c", "d", "e"])

        # Duplicates are not okay
        with pytest.raises(duckdb.BinderException, match="must have unique values"):
            con.read_csv(file, names=["a", "b", "a", "b"])

    def test_read_csv_names_mixed_with_dtypes(self, tmp_path):
        file = tmp_path / "file.csv"
        file.write_text("one,two,three,four\n1,2,3,4\n1,2,3,4\n1,2,3,4")

        con = duckdb.connect()
        rel = con.read_csv(
            file,
            names=["a", "b", "c"],
            dtype={
                "a": int,
                "b": bool,
                "c": str,
            },
        )
        assert rel.columns == ["a", "b", "c", "four"]
        assert rel.types == ["BIGINT", "BOOLEAN", "VARCHAR", "BIGINT"]

        # dtypes and names dont match
        # TODO: seems the order columns are named in this error is non-deterministic  # noqa: TD002, TD003
        # so for now I'm excluding the list of columns from the expected error
        expected_error = """do not exist in the CSV File"""
        with pytest.raises(duckdb.BinderException, match=expected_error):
            rel = con.read_csv(
                file,
                names=["a", "b", "c"],
                dtype={
                    "d": int,
                    "e": bool,
                    "f": str,
                },
            )

    def test_read_csv_multi_file(self, tmp_path):
        file1 = tmp_path / "file1.csv"
        file1.write_text("one,two,three,four\n1,2,3,4\n1,2,3,4\n1,2,3,4")

        file2 = tmp_path / "file2.csv"
        file2.write_text("one,two,three,four\n5,6,7,8\n5,6,7,8\n5,6,7,8")

        file3 = tmp_path / "file3.csv"
        file3.write_text("one,two,three,four\n9,10,11,12\n9,10,11,12\n9,10,11,12")

        con = duckdb.connect()
        files = [str(file1), str(file2), str(file3)]
        rel = con.read_csv(files)
        res = rel.fetchall()
        assert res == [
            (1, 2, 3, 4),
            (1, 2, 3, 4),
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (5, 6, 7, 8),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (9, 10, 11, 12),
            (9, 10, 11, 12),
        ]

    def test_read_csv_empty_list(self):
        con = duckdb.connect()
        files = []
        with pytest.raises(
            duckdb.InvalidInputException, match="Please provide a non-empty list of paths or file-like objects"
        ):
            con.read_csv(files)

    def test_read_auto_detect(self, tmp_path):
        file1 = tmp_path / "file1.csv"
        file1.write_text("one|two|three|four\n1|2|3|4")

        con = duckdb.connect()
        rel = con.read_csv(str(file1), columns={"a": "VARCHAR"}, auto_detect=False, header=False)
        assert rel.fetchall() == [("one|two|three|four",), ("1|2|3|4",)]

    def test_read_csv_list_invalid_path(self, tmp_path):
        con = duckdb.connect()

        file1 = tmp_path / "file1.csv"
        file1.write_text("one,two,three,four\n1,2,3,4\n1,2,3,4\n1,2,3,4")

        file3 = tmp_path / "file3.csv"
        file3.write_text("one,two,three,four\n9,10,11,12\n9,10,11,12\n9,10,11,12")

        files = [str(file1), "not_valid_path", str(file3)]
        with pytest.raises(duckdb.IOException, match='No files found that match the pattern "not_valid_path"'):
            con.read_csv(files)

    @pytest.mark.parametrize(
        "options",
        [
            {"lineterminator": "\\n"},
            {"lineterminator": "LINE_FEED"},
            {"lineterminator": CSVLineTerminator.LINE_FEED},
            {"columns": {"id": "INTEGER", "name": "INTEGER", "c": "integer", "d": "INTEGER"}},
            {"auto_type_candidates": ["INTEGER", "INTEGER"]},
            {"max_line_size": 10000},
            {"ignore_errors": True},
            {"ignore_errors": False},
            {"store_rejects": True},
            {"store_rejects": False},
            {"rejects_table": "my_rejects_table"},
            {"rejects_scan": "my_rejects_scan"},
            {"rejects_table": "my_rejects_table", "rejects_limit": 50},
            {"force_not_null": ["one", "two"]},
            {"buffer_size": 2097153},
            {"decimal": "."},
            {"allow_quoted_nulls": True},
            {"allow_quoted_nulls": False},
            {"filename": True},
            {"filename": "test"},
            {"hive_partitioning": True},
            {"hive_partitioning": False},
            {"union_by_name": True},
            {"union_by_name": False},
            {"hive_types_autocast": False},
            {"hive_types_autocast": True},
            {"hive_types": {"one": "INTEGER", "two": "VARCHAR"}},
        ],
    )
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Skipping on Windows because of lineterminator option")
    def test_read_csv_options(self, duckdb_cursor, options, tmp_path):
        file = tmp_path / "file.csv"
        file.write_text("one,two,three,four\n1,2,3,4\n1,2,3,4\n1,2,3,4")
        print(options)
        if "hive_types" in options:
            with pytest.raises(duckdb.InvalidInputException, match=r"Unknown hive_type:"):
                rel = duckdb_cursor.read_csv(file, **options)
        else:
            rel = duckdb_cursor.read_csv(file, **options)
            rel.fetchall()

    def test_read_comment(self, tmp_path):
        file1 = tmp_path / "file1.csv"
        file1.write_text("one|two|three|four\n1|2|3|4#|5|6\n#bla\n1|2|3|4\n")

        con = duckdb.connect()
        rel = con.read_csv(str(file1), columns={"a": "VARCHAR"}, auto_detect=False, header=False, comment="#")
        assert rel.fetchall() == [("one|two|three|four",), ("1|2|3|4",), ("1|2|3|4",)]

    def test_read_enum(self, tmp_path):
        file1 = tmp_path / "file1.csv"
        file1.write_text("feelings\nhappy\nsad\nangry\nhappy\n")

        con = duckdb.connect()
        con.execute("CREATE TYPE mood AS ENUM ('happy', 'sad', 'angry')")

        rel = con.read_csv(str(file1), dtype=["mood"])
        assert rel.fetchall() == [("happy",), ("sad",), ("angry",), ("happy",)]

        rel = con.read_csv(str(file1), dtype={"feelings": "mood"})
        assert rel.fetchall() == [("happy",), ("sad",), ("angry",), ("happy",)]

        rel = con.read_csv(str(file1), columns={"feelings": "mood"})
        assert rel.fetchall() == [("happy",), ("sad",), ("angry",), ("happy",)]

        with pytest.raises(duckdb.CatalogException, match="Type with name mood_2 does not exist!"):
            rel = con.read_csv(str(file1), columns={"feelings": "mood_2"})

        with pytest.raises(duckdb.CatalogException, match="Type with name mood_2 does not exist!"):
            rel = con.read_csv(str(file1), dtype={"feelings": "mood_2"})

        with pytest.raises(duckdb.CatalogException, match="Type with name mood_2 does not exist!"):
            rel = con.read_csv(str(file1), dtype=["mood_2"])

    def test_strict_mode(self, tmp_path):
        file1 = tmp_path / "file1.csv"
        file1.write_text("one|two|three|four\n1|2|3|4\n1|2|3|4|5\n1|2|3|4\n")

        con = duckdb.connect()
        rel = con.read_csv(
            str(file1),
            header=True,
            delimiter="|",
            columns={"a": "INTEGER", "b": "INTEGER", "c": "INTEGER", "d": "INTEGER"},
            auto_detect=False,
        )
        with pytest.raises(duckdb.InvalidInputException, match="CSV Error on Line"):
            rel.fetchall()

        rel = con.read_csv(
            str(file1),
            header=True,
            delimiter="|",
            strict_mode=False,
            columns={"a": "INTEGER", "b": "INTEGER", "c": "INTEGER", "d": "INTEGER"},
            auto_detect=False,
        )
        assert rel.fetchall() == [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)]

    def test_union_by_name(self, tmp_path):
        file1 = tmp_path / "file1.csv"
        file1.write_text("one|two|three|four\n1|2|3|4")

        file1 = tmp_path / "file2.csv"
        file1.write_text("two|three|four|five\n2|3|4|5")

        con = duckdb.connect()

        file_path = tmp_path / "file*.csv"
        rel = con.read_csv(file_path, union_by_name=True)
        assert rel.columns == ["one", "two", "three", "four", "five"]
        assert rel.fetchall() == [(1, 2, 3, 4, None), (None, 2, 3, 4, 5)]

    def test_thousands_separator(self, tmp_path):
        file = tmp_path / "file_thousands.csv"
        file.write_text('money\n"10,000.23"\n"1,000,000,000.01"')

        con = duckdb.connect()
        rel = con.read_csv(file, thousands=",")
        assert rel.fetchall() == [(10000.23,), (1000000000.01,)]

        with pytest.raises(
            duckdb.BinderException, match="Unsupported parameter for THOUSANDS: should be max one character"
        ):
            con.read_csv(file, thousands=",,,")

    def test_skip_comment_option(self, tmp_path):
        file1 = tmp_path / "file1.csv"
        file1.write_text("skip this line\n# comment\nx,y,z\n1,2,3\n4,5,6")
        con = duckdb.connect()
        rel = con.read_csv(file1, comment="#", skiprows=1, all_varchar=True)
        assert rel.columns == ["x", "y", "z"]
        assert rel.fetchall() == [("1", "2", "3"), ("4", "5", "6")]

    def test_files_to_sniff_option(self, tmp_path):
        file1 = tmp_path / "file1.csv"
        file1.write_text("bar,baz\n2025-05-12,baz")
        file2 = tmp_path / "file2.csv"
        file2.write_text("bar,baz\nbar,baz")

        file_path = tmp_path / "file*.csv"
        con = duckdb.connect()
        rel = con.read_csv(file_path, files_to_sniff=1)
        with pytest.raises(duckdb.ConversionException, match="Conversion Error"):
            rel.fetchall()
        rel = con.read_csv(file_path, files_to_sniff=-1)
        assert rel.fetchall() == [("2025-05-12", "baz"), ("bar", "baz")]


def _object_store_path(relation):
    return re.search(r"DUCKDB_INTERNAL_OBJECTSTORE://[0-9a-f]+", relation.sql_query()).group(0)


def _one_reader(con):
    return con.read_csv(StringIO("c1\na"))


def _two_readers_in_a_union(con):
    return con.read_csv(StringIO("c1\na")).union(con.read_csv(StringIO("c1\nb")))


def _one_reader_of_two_filelikes(con):
    return con.read_csv([StringIO("c1\na"), StringIO("c1\nb")])


def _a_reader_beside_a_table(con):
    con.execute("CREATE TABLE t AS SELECT 1 AS k")
    return con.table("t").cross(con.read_csv(StringIO("c1\na")))


def _a_reader_read_by_sql(con):
    reader = con.read_csv(StringIO("c1\na"))  # noqa: F841, read by name below
    return con.sql("SELECT * FROM reader")


class TestFilelikeViews:
    # The object store entry of a file-like reader must live as long as a view that reads it, not as long as the
    # relation the view was made from.

    @pytest.mark.parametrize("make_view", ["to_view", "create_view"])
    @pytest.mark.parametrize("reader", ["read_csv", "read_json"])
    def test_view_outlives_the_relation(self, duckdb_cursor, reader, make_view):
        _ = pytest.importorskip("fsspec")
        source = StringIO('[{"c1": "a"}]') if reader == "read_json" else StringIO("c1\na")
        getattr(getattr(duckdb_cursor, reader)(source), make_view)("v")
        gc.collect()
        assert duckdb_cursor.sql("SELECT * FROM v").fetchall() == [("a",)]
        assert duckdb_cursor.sql("SELECT * FROM v").fetchall() == [("a",)]

    def test_view_over_a_derived_relation(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        duckdb_cursor.read_csv(StringIO("c1\n1\n2\n3")).filter("c1 > 1").project("c1 * 10 AS c").to_view("v")
        gc.collect()
        assert duckdb_cursor.sql("SELECT c FROM v ORDER BY c").fetchall() == [(20,), (30,)]

    def test_view_over_two_filelikes(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        left = duckdb_cursor.read_csv(StringIO("c1\n1"))
        right = duckdb_cursor.read_csv(StringIO("c1\n2"))
        left.union(right).to_view("v")
        del left, right
        gc.collect()
        assert duckdb_cursor.sql("SELECT c1 FROM v ORDER BY c1").fetchall() == [(1,), (2,)]

    def test_views_share_the_entry(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        rel = duckdb_cursor.read_csv(StringIO("c1\na"))
        path = _object_store_path(rel)
        rel.to_view("v1")
        rel.to_view("v2")
        del rel
        gc.collect()
        duckdb_cursor.execute("DROP VIEW v1")
        gc.collect()
        assert duckdb_cursor.sql("SELECT * FROM v2").fetchall() == [("a",)]
        duckdb_cursor.execute("DROP VIEW v2")
        gc.collect()
        with pytest.raises(duckdb.IOException, match="No files found"):
            duckdb_cursor.sql(f"SELECT * FROM read_csv('{path}')").fetchall()

    def test_replaced_view_releases_its_entry(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        first = duckdb_cursor.read_csv(StringIO("c1\na"))
        first_path = _object_store_path(first)
        first.to_view("v")
        del first
        duckdb_cursor.read_csv(StringIO("c1\nb")).to_view("v")
        gc.collect()
        assert duckdb_cursor.sql("SELECT * FROM v").fetchall() == [("b",)]
        with pytest.raises(duckdb.IOException, match="No files found"):
            duckdb_cursor.sql(f"SELECT * FROM read_csv('{first_path}')").fetchall()

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [
            pytest.param("frame_alone", [(1,)], id="frame_alone"),
            pytest.param("frame_union_filelike", [(1,), (2,)], id="frame_union_filelike"),
        ],
    )
    def test_view_over_a_registered_frame_releases_the_frame(self, duckdb_cursor, shape, expected):
        _ = pytest.importorskip("fsspec")
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame({"c1": [1]})
        alive = weakref.ref(frame)
        rel = duckdb_cursor.from_df(frame)
        if shape == "frame_union_filelike":
            rel = rel.union(duckdb_cursor.read_csv(StringIO("c1\n2")))
        # A frame alone puts the registry's own reference, with its shared dependency, in the view's FROM; the union
        # puts a CTE there. The registry entry must not come to hold the view's handles, or the frame outlives the view
        rel.to_view("v")
        del rel, frame
        gc.collect()
        assert duckdb_cursor.sql("SELECT c1 FROM v ORDER BY c1").fetchall() == expected
        duckdb_cursor.execute("DROP VIEW v")
        gc.collect()
        assert alive() is None

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            pytest.param(
                lambda con: con.read_csv(StringIO("a\n1")).cross(con.read_csv(StringIO("b\n2"))),
                [(1, 2)],
                id="join_of_two_filelikes",
            ),
            pytest.param(
                lambda con: con.sql("SELECT 1 AS a").cross(con.read_csv(StringIO("b\n2"))),
                [(1, 2)],
                id="join_with_a_constant_left_side",
            ),
            pytest.param(
                lambda con: con.sql("SELECT 1 AS c1").union(con.read_csv(StringIO("c1\n2"))),
                [(1,), (2,)],
                id="union_with_a_constant_left_side",
            ),
        ],
    )
    def test_renamed_view_keeps_its_sources(self, duckdb_cursor, build, expected):
        _ = pytest.importorskip("fsspec")
        # The engine copies the view to rename it, and the copy of a join reference drops what hangs on the join
        build(duckdb_cursor).to_view("v")
        gc.collect()
        duckdb_cursor.execute("ALTER VIEW v RENAME TO renamed")
        gc.collect()
        assert duckdb_cursor.sql("SELECT * FROM renamed ORDER BY ALL").fetchall() == expected

    def test_view_over_a_join_keeps_the_deduplicated_column_names(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        duckdb_cursor.read_csv(StringIO("a\n1")).cross(duckdb_cursor.read_csv(StringIO("a\n2"))).to_view("v")
        gc.collect()
        assert [row[0] for row in duckdb_cursor.execute("DESCRIBE v").fetchall()] == ["a", "a_1"]
        assert duckdb_cursor.sql("SELECT * FROM v").fetchall() == [(1, 2)]

    def test_rolled_back_create_releases_the_entry(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        rel = duckdb_cursor.read_csv(StringIO("c1\na"))
        path = _object_store_path(rel)
        duckdb_cursor.execute("BEGIN")
        rel.to_view("v")
        del rel
        gc.collect()
        assert duckdb_cursor.sql("SELECT * FROM v").fetchall() == [("a",)]
        duckdb_cursor.execute("ROLLBACK")
        gc.collect()
        assert duckdb_cursor.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall() == []
        with pytest.raises(duckdb.IOException, match="No files found"):
            duckdb_cursor.sql(f"SELECT * FROM read_csv('{path}')").fetchall()

    def test_rolled_back_drop_keeps_the_view(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        duckdb_cursor.read_csv(StringIO("c1\na")).to_view("v")
        gc.collect()
        duckdb_cursor.execute("BEGIN")
        duckdb_cursor.execute("DROP VIEW v")
        gc.collect()
        duckdb_cursor.execute("ROLLBACK")
        gc.collect()
        assert duckdb_cursor.sql("SELECT * FROM v").fetchall() == [("a",)]

    def test_rolled_back_replace_restores_the_view_and_releases_the_replacement(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        duckdb_cursor.read_csv(StringIO("c1\na")).to_view("v")
        gc.collect()
        duckdb_cursor.execute("BEGIN")
        replacement = duckdb_cursor.read_csv(StringIO("c1\nb"))
        path = _object_store_path(replacement)
        replacement.to_view("v")
        del replacement
        gc.collect()
        assert duckdb_cursor.sql("SELECT * FROM v").fetchall() == [("b",)]
        duckdb_cursor.execute("ROLLBACK")
        gc.collect()
        assert duckdb_cursor.sql("SELECT * FROM v").fetchall() == [("a",)]
        with pytest.raises(duckdb.IOException, match="No files found"):
            duckdb_cursor.sql(f"SELECT * FROM read_csv('{path}')").fetchall()

    def test_create_over_a_taken_name_without_replace_releases_the_entry(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        duckdb_cursor.read_csv(StringIO("c1\na")).to_view("v")
        gc.collect()
        second = duckdb_cursor.read_csv(StringIO("c1\nb"))
        path = _object_store_path(second)
        with pytest.raises(duckdb.CatalogException, match="already exists"):
            second.create_view("v", replace=False)
        del second
        gc.collect()
        assert duckdb_cursor.sql("SELECT * FROM v").fetchall() == [("a",)]
        with pytest.raises(duckdb.IOException, match="No files found"):
            duckdb_cursor.sql(f"SELECT * FROM read_csv('{path}')").fetchall()

    def test_view_reads_from_a_cursor(self, duckdb_cursor):
        _ = pytest.importorskip("fsspec")
        duckdb_cursor.read_csv(StringIO("c1\na")).to_view("v")
        gc.collect()
        assert duckdb_cursor.cursor().sql("SELECT * FROM v").fetchall() == [("a",)]

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            pytest.param(_one_reader, [("a",)], id="reader"),
            pytest.param(_one_reader_of_two_filelikes, [("a",), ("b",)], id="reader_of_two_filelikes"),
            pytest.param(_two_readers_in_a_union, [("a",), ("b",)], id="union_of_two_readers"),
            pytest.param(_a_reader_beside_a_table, [(1, "a")], id="reader_beside_a_table"),
            pytest.param(_a_reader_read_by_sql, [("a",)], id="reader_read_by_sql"),
        ],
    )
    def test_view_over_a_filelike_is_refused_in_a_database_file(self, tmp_path, build, expected):
        _ = pytest.importorskip("fsspec")
        # The object store is a file system in this process's memory, so a stored view would read nothing after a
        # reopen
        con = duckdb.connect(str(tmp_path / "saved.duckdb"))
        rel = build(con)
        match = r"cannot be stored in a database file\. Use register\(\) for a view that reads it in this session"
        with pytest.raises(duckdb.InvalidInputException, match=match):
            rel.to_view("v")
        assert con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall() == []
        con.register("v", rel)
        del rel
        gc.collect()
        assert con.sql("SELECT * FROM v ORDER BY ALL").fetchall() == expected
        con.close()

    def test_view_over_a_filelike_in_an_attached_memory_database_is_created(self, tmp_path):
        _ = pytest.importorskip("fsspec")
        con = duckdb.connect(str(tmp_path / "saved.duckdb"))
        con.execute("ATTACH ':memory:' AS scratch")
        con.execute("USE scratch")
        con.read_csv(StringIO("c1\na")).to_view("v")
        gc.collect()
        assert con.sql("SELECT * FROM scratch.main.v").fetchall() == [("a",)]
        con.close()


class TestObjectStoreAcrossConnections:
    @pytest.mark.parametrize("first", ["cursor", "parent"])
    def test_filelike_reads_on_cursor_and_parent(self, first):
        _ = pytest.importorskip("fsspec")
        con = duckdb.connect()
        cursor = con.cursor()
        order = [cursor, con] if first == "cursor" else [con, cursor]
        for number, connection in enumerate(order, start=1):
            assert connection.read_csv(StringIO(f"c1\n{number}")).fetchall() == [(number,)]
        assert con.cursor().read_csv(StringIO("c1\n3")).fetchall() == [(3,)]

    def test_database_closes_after_the_registering_cursor_is_gone(self):
        _ = pytest.importorskip("fsspec")
        con = duckdb.connect()
        cursor = con.cursor()
        assert cursor.read_csv(StringIO("c1\n1")).fetchall() == [(1,)]
        cursor.close()
        del cursor
        gc.collect()
        # The store's Python object dies with the database, dropped by a connection that never built it
        con.close()

    def test_two_connections_to_one_database_file(self, tmp_path):
        _ = pytest.importorskip("fsspec")
        path = str(tmp_path / "shared.db")
        first = duckdb.connect(path)
        second = duckdb.connect(path)
        assert first.read_csv(StringIO("c1\n1")).fetchall() == [(1,)]
        assert second.read_csv(StringIO("c1\n2")).fetchall() == [(2,)]
        first.close()
        assert second.read_csv(StringIO("c1\n3")).fetchall() == [(3,)]
        second.close()

    def test_another_connection_registers_inside_the_fsspec_import(self, monkeypatch):
        _ = pytest.importorskip("fsspec")
        import builtins

        import duckdb.filesystem

        con = duckdb.connect()
        sibling = con.cursor()
        original = builtins.__import__
        armed = [True]

        def reenter(name, *args, **kwargs):
            # The import the registration runs hands the GIL over; the sibling registers the store meanwhile
            if name == "fsspec" and armed[0]:
                armed[0] = False
                assert sibling.read_csv(StringIO("c1\n2")).fetchall() == [(2,)]
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", reenter)
        assert con.read_csv(StringIO("c1\n1")).fetchall() == [(1,)]

    @pytest.mark.timeout(20)
    def test_two_connections_build_the_store_at_once(self, monkeypatch):
        _ = pytest.importorskip("fsspec")
        from duckdb.filesystem import ModifiedMemoryFileSystem

        original = ModifiedMemoryFileSystem.__init__
        building = threading.Event()

        def slow_init(self, *args, **kwargs):
            # Hands the GIL to the other connection in the middle of building the store
            building.set()
            time.sleep(0.5)
            original(self, *args, **kwargs)

        # fsspec hands out a cached instance per thread without calling __init__; a fresh one per build is needed
        monkeypatch.setattr(ModifiedMemoryFileSystem, "cachable", False)
        monkeypatch.setattr(ModifiedMemoryFileSystem, "__init__", slow_init)
        con = duckdb.connect()
        cursor = con.cursor()
        results = []
        worker = threading.Thread(target=lambda: results.append(cursor.read_csv(StringIO("c1\n1")).fetchall()))
        worker.start()
        assert building.wait(10)
        results.append(con.read_csv(StringIO("c1\n2")).fetchall())
        worker.join()
        assert sorted(results) == [[(1,)], [(2,)]]

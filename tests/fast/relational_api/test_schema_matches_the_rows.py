"""A relation's declared schema is the truth about the rows it returns, whatever the source did since it was built.

A reader that sniffs its schema at construction has to carry that schema into the statement it runs. Otherwise the
file can change under the relation and the rows no longer answer to `columns` and `types`.
"""

import json
import re

import pytest

import duckdb

ROW_SHAPE = "a relation returns as many values as it declares columns"


@pytest.fixture
def con():
    connection = duckdb.connect()
    yield connection
    connection.close()


def reader(con, path, kind):
    return con.read_csv(str(path)) if kind == "csv" else con.read_json(str(path))


def written(path, kind, rows):
    if kind == "csv":
        path.write_text("".join(rows))
    else:
        path.write_text("".join(rows))


def built_over(con, tmp_path, kind, first):
    path = tmp_path / f"source.{kind}"
    written(path, kind, first)
    return reader(con, path, kind), path


FIRST = {
    "csv": ["id,amount\n", "1,10\n"],
    "json": ['{"id":1,"amount":10}\n'],
}
GAINED = {
    "csv": ["id,amount,currency\n", "1,10,eur\n"],
    "json": ['{"id":1,"amount":10,"currency":"eur"}\n'],
}
WIDENED = {
    "csv": ["id,amount\n", "1,20.5\n"],
    "json": ['{"id":1,"amount":20.5}\n'],
}
LOST = {
    "csv": ["id\n", "1\n"],
    "json": ['{"id":1}\n'],
}
KINDS = [pytest.param("csv", id="read_csv"), pytest.param("json", id="read_json")]


def pinned_twin(con, path, kind):
    """The rows the reader gives for this file when the bound schema is written into the call by hand."""
    columns = "{'id': 'BIGINT', 'amount': 'BIGINT'}"
    if kind == "csv":
        return con.execute(
            f"SELECT * FROM read_csv('{path}', columns := {columns}, auto_detect := false, header := true)"
        ).fetchall()
    return con.execute(f"SELECT * FROM read_json('{path}', columns := {columns})").fetchall()


def assert_the_rows_answer_to_the_schema(rel, rows):
    for row in rows:
        assert len(row) == len(rel.columns), ROW_SHAPE
        assert all(value is None or isinstance(value, int) for value in row), rel.types


class TestAReaderKeepsTheSchemaItBound:
    """Whatever happens to the file, the rows obey the declared columns and types; how each reader gets there differs.

    A CSV reader with its columns pinned and its sniffer off refuses a line that does not fit, as the shipping client
    did. A JSON reader keeps its format detection on, so with its columns pinned it reads only those fields:
    an extra field is left out and a missing one is NULL. Both keep the invariant, which is the rule.
    """

    def test_a_csv_file_that_gained_a_column_is_refused(self, con, tmp_path):
        rel, path = built_over(con, tmp_path, "csv", FIRST["csv"])
        written(path, "csv", GAINED["csv"])
        with pytest.raises(duckdb.InvalidInputException, match="CSV Error"):
            rel.fetchall()

    def test_a_json_file_that_gained_a_field_answers_the_declared_columns_only(self, con, tmp_path):
        rel, path = built_over(con, tmp_path, "json", FIRST["json"])
        written(path, "json", GAINED["json"])
        rows = rel.fetchall()
        assert_the_rows_answer_to_the_schema(rel, rows)
        assert rows == [(1, 10)]

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_file_whose_values_widened_still_answers_in_the_declared_type(self, con, tmp_path, kind):
        rel, path = built_over(con, tmp_path, kind, FIRST[kind])
        assert [str(declared) for declared in rel.types] == ["BIGINT", "BIGINT"]
        written(path, kind, WIDENED[kind])
        rows = rel.fetchall()
        assert_the_rows_answer_to_the_schema(rel, rows)
        assert rows == pinned_twin(con, path, kind)

    def test_a_csv_file_that_lost_a_column_is_refused(self, con, tmp_path):
        rel, path = built_over(con, tmp_path, "csv", FIRST["csv"])
        written(path, "csv", LOST["csv"])
        with pytest.raises(duckdb.InvalidInputException, match="CSV Error"):
            rel.fetchall()

    def test_a_json_file_that_lost_a_field_answers_null_for_it(self, con, tmp_path):
        rel, path = built_over(con, tmp_path, "json", FIRST["json"])
        written(path, "json", LOST["json"])
        rows = rel.fetchall()
        assert_the_rows_answer_to_the_schema(rel, rows)
        assert rows == [(1, None)]

    @pytest.mark.parametrize("kind", KINDS)
    def test_every_row_answers_to_the_declared_columns_and_types(self, con, tmp_path, kind):
        rel, path = built_over(con, tmp_path, kind, FIRST[kind])
        written(path, kind, FIRST[kind] + (["2,20\n"] if kind == "csv" else ['{"id":2,"amount":20}\n']))
        rows = rel.fetchall()
        assert rows == [(1, 10), (2, 20)]
        assert_the_rows_answer_to_the_schema(rel, rows)


class TestAReaderRendersTheSchemaItBound:
    """The cause, not the symptom: the text a reader renders carries its bound columns."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_text_names_every_bound_column_with_its_type(self, con, tmp_path, kind):
        rel, _ = built_over(con, tmp_path, kind, FIRST[kind])
        text = rel.sql_query()
        assert "columns" in text.lower()
        assert all(name in text for name in rel.columns)
        assert text.count("BIGINT") >= len(rel.columns)

    def test_the_csv_text_turns_the_sniffer_off(self, con, tmp_path):
        rel, _ = built_over(con, tmp_path, "csv", FIRST["csv"])
        assert re.search(r"auto_detect\s*(:?=)\s*false", rel.sql_query(), re.IGNORECASE)

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_text_reads_the_same_rows_on_the_same_connection(self, con, tmp_path, kind):
        rel, _ = built_over(con, tmp_path, kind, FIRST[kind])
        assert con.sql(rel.sql_query()).fetchall() == rel.fetchall() == [(1, 10)]


class TestAReaderWithoutASniffer:
    """Parquet carries its schema in the file, so there is nothing to sniff and nothing to render back."""

    def test_parquet_reads_its_schema_from_the_file(self, con, tmp_path):
        path = tmp_path / "source.parquet"
        con.execute(f"COPY (SELECT 1 AS id, 10 AS amount) TO '{path}' (FORMAT parquet)")
        rel = con.read_parquet(str(path))
        assert rel.columns == ["id", "amount"]
        assert rel.fetchall() == [(1, 10)]
        assert "auto_detect" not in rel.sql_query()


def the_paths(con, rel):
    """The rows a reader relation gives on every path a user reads it through."""
    return {
        "fetchall": rel.fetchall(),
        "text": con.sql(rel.sql_query()).fetchall(),
        "step": rel.filter("true").fetchall(),
    }


def sql_twin(con, sql):
    cursor = con.execute(sql)
    names = [column[0] for column in cursor.description]
    return names, cursor.fetchall()


FORMAT_FILES = [
    pytest.param("id,day\n1,31/12/2024\n2,01/02/2025\n", ["dateformat"], id="date"),
    pytest.param("id,at\n1,31/12/2024 13:45:00\n2,01/02/2025 08:00:00\n", ["timestampformat"], id="timestamp"),
    pytest.param(
        "id,day,at\n1,31/12/2024,31/12/2024 13:45:00\n", ["dateformat", "timestampformat"], id="date_and_timestamp"
    ),
]


class TestTheSniffedFormatsArePinned:
    """Pinning keeps every sniffed option, the date and timestamp formats included, so the rows stay the same."""

    @pytest.mark.parametrize(("text", "formats"), FORMAT_FILES)
    def test_every_path_reads_the_rows_the_sniffing_call_reads(self, con, tmp_path, text, formats):
        path = tmp_path / "formats.csv"
        path.write_text(text)
        names, expected = sql_twin(con, f"SELECT * FROM read_csv('{path}')")
        rel = con.read_csv(str(path))
        assert rel.columns == names
        for path_name, answered in the_paths(con, rel).items():
            assert answered == expected, path_name

    @pytest.mark.parametrize(("text", "formats"), FORMAT_FILES)
    def test_the_text_carries_the_sniffed_formats(self, con, tmp_path, text, formats):
        path = tmp_path / "formats.csv"
        path.write_text(text)
        rendered = con.read_csv(str(path)).sql_query().lower()
        assert all(name in rendered for name in formats)

    @pytest.mark.parametrize(("text", "formats"), FORMAT_FILES)
    def test_a_row_added_in_the_same_format_reads_too(self, con, tmp_path, text, formats):
        path = tmp_path / "formats.csv"
        path.write_text(text)
        rel = con.read_csv(str(path))
        first_row = text.splitlines()[1]
        path.write_text(text + first_row.replace("1,", "9,", 1) + "\n")
        _, expected = sql_twin(con, f"SELECT * FROM read_csv('{path}')")
        assert rel.fetchall() == expected


class TestGeneratedColumns:
    """A generated column is declared, returned and kept out of the pinned physical columns."""

    def one_file(self, tmp_path):
        path = tmp_path / "plain.csv"
        path.write_text("a\n1\n")
        return path

    def partitioned(self, tmp_path):
        part = tmp_path / "hive" / "part=x"
        part.mkdir(parents=True)
        (part / "f.csv").write_text("a\n1\n")
        return tmp_path / "hive" / "*" / "*.csv", part / "f.csv"

    def test_filename_is_declared_and_returned(self, con, tmp_path):
        path = self.one_file(tmp_path)
        rel = con.read_csv(str(path), filename=True)
        names, expected = sql_twin(con, f"SELECT * FROM read_csv('{path}', filename = true)")
        assert rel.columns == names == ["a", "filename"]
        assert rel.fetchall() == expected
        assert "auto_detect := false" in rel.sql_query()

    def test_a_hive_partition_is_declared_and_returned(self, con, tmp_path):
        pattern, _ = self.partitioned(tmp_path)
        rel = con.read_csv(str(pattern), hive_partitioning=True)
        names, expected = sql_twin(con, f"SELECT * FROM read_csv('{pattern}', hive_partitioning = true)")
        assert rel.columns == names == ["a", "part"]
        assert rel.fetchall() == expected
        assert "auto_detect := false" in rel.sql_query()

    def test_filename_and_a_hive_partition_together(self, con, tmp_path):
        pattern, _ = self.partitioned(tmp_path)
        rel = con.read_csv(str(pattern), hive_partitioning=True, filename=True)
        names, expected = sql_twin(
            con, f"SELECT * FROM read_csv('{pattern}', hive_partitioning = true, filename = true)"
        )
        assert rel.columns == names
        assert rel.fetchall() == expected
        assert "auto_detect := false" in rel.sql_query()

    @pytest.mark.parametrize(
        "options",
        [
            pytest.param({"filename": True}, id="filename"),
            pytest.param({"hive_partitioning": True}, id="hive"),
            pytest.param({"hive_partitioning": True, "filename": True}, id="both"),
        ],
    )
    def test_a_file_that_gained_a_column_is_refused_rather_than_answered(self, con, tmp_path, options):
        if options.get("hive_partitioning"):
            source, written_file = self.partitioned(tmp_path)
        else:
            source = written_file = self.one_file(tmp_path)
        rel = con.read_csv(str(source), **options)
        declared = rel.columns
        assert len(declared) == len(rel.fetchall()[0]), ROW_SHAPE
        written_file.write_text("a,b\n1,2\n")
        with pytest.raises(duckdb.Error, match="CSV Error"):
            rel.fetchall()


class TestEveryOptionKeepsThePin:
    """Whatever options the user passes, the reader pins the bound schema and answers what the same call answers."""

    def files(self, tmp_path):
        (tmp_path / "a.csv").write_text("A b,c\n1,x\n2,y\n")
        (tmp_path / "b.csv").write_text("c,A b\ny,3\n")
        return tmp_path

    @pytest.mark.parametrize(
        ("source", "options", "sql_options"),
        [
            pytest.param("a.csv", {"names": ["n1", "n2"]}, "names = ['n1', 'n2']", id="names"),
            pytest.param("a.csv", {"header": False}, "header = false", id="header_false"),
            pytest.param("a.csv", {"dtype": {"c": "VARCHAR"}}, "types = {'c': 'VARCHAR'}", id="dtype"),
            pytest.param("a.csv", {"normalize_names": True}, "normalize_names = true", id="normalize_names"),
            pytest.param("a.csv", {"all_varchar": True}, "all_varchar = true", id="all_varchar"),
        ],
    )
    def test_the_pinned_call_answers_what_the_sniffing_call_answers(self, con, tmp_path, source, options, sql_options):
        directory = self.files(tmp_path)
        path = directory / source
        names, expected = sql_twin(con, f"SELECT * FROM read_csv('{path}', {sql_options})")
        rel = con.read_csv(str(path), **options)
        assert rel.columns == names
        assert sorted(rel.fetchall(), key=repr) == sorted(expected, key=repr)
        assert "auto_detect := false" in rel.sql_query()

    def test_a_union_by_name_pins_the_bound_types_by_name(self, con, tmp_path):
        # Files of different shapes share no dialect or column list, so the union pins the types by name and
        # selects the bound names; the invariant is the rule, the sniffer flag is only the single file mechanism.
        directory = self.files(tmp_path)
        pattern = directory / "*.csv"
        names, expected = sql_twin(con, f"SELECT * FROM read_csv('{pattern}', union_by_name = true)")
        rel = con.read_csv(str(pattern), union_by_name=True)
        assert rel.columns == names
        assert sorted(rel.fetchall(), key=repr) == sorted(expected, key=repr)
        text = rel.sql_query()
        tree = json.loads(con.execute("SELECT json_serialize_sql(?)", [text]).fetchall()[0][0])
        select_list = tree["statements"][0]["node"]["select_list"]
        assert [item["column_names"] for item in select_list] == [[name] for name in rel.columns]
        assert "types :=" in text
        assert all(f"'{name}': '{kind}'" in text for name, kind in zip(rel.columns, rel.types, strict=True))

    def test_a_pinned_call_that_cannot_bind_raises_at_construction(self, con, tmp_path, monkeypatch):
        # The reader never falls back to the sniffing call; a pin it cannot keep is an error at the call.
        from duckdb import _seam

        path = tmp_path / "plain.csv"
        path.write_text("a\n1\n")
        bind = _seam.bind

        def refusing(connection, sql, *args, **kwargs):
            if "auto_detect := false" in sql:
                message = "the pinned call cannot bind"
                raise duckdb.BinderException(message)
            return bind(connection, sql, *args, **kwargs)

        monkeypatch.setattr(_seam, "bind", refusing)
        with pytest.raises(duckdb.BinderException, match="pinned call cannot bind"):
            con.read_csv(str(path))


class TestANameTheReaderMadeUpCannotBePinned:
    """JSON keys differing only in case are renamed by the reader, so the call refuses rather than reads NULL."""

    def test_keys_differing_only_in_case_are_refused_at_construction_naming_them(self, con, tmp_path):
        path = tmp_path / "keys.json"
        path.write_text('{"a": 1, "A": 2}\n')
        with pytest.raises(duckdb.InvalidInputException) as error:
            con.read_json(str(path))
        message = str(error.value)
        assert re.search(r"\ba\b", message)
        assert re.search(r"\bA\b", message)

    def test_keys_that_differ_otherwise_still_read(self, con, tmp_path):
        path = tmp_path / "keys.json"
        path.write_text('{"a": 1, "b": 2}\n')
        assert con.read_json(str(path)).fetchall() == [(1, 2)]

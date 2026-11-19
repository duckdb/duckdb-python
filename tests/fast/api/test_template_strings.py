import datetime
import decimal
import importlib.util
import uuid
from string.templatelib import Interpolation, Template

import pytest

import duckdb

needs_pandas = pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="pandas not installed")

STRING_LITERAL = (
    "is inside a string literal, where it would be text, not a parameter; make the whole literal the value, "
    "or join the value to the literal with ||"
)
GLUED = "touches the text next to it, so it is not a parameter of its own; put a space between them"
WITH_PARAMS = (
    "A template carries its own values, so params cannot be given with it; put the values in the template, "
    "or pass a str with params"
)
EXECUTEMANY = (
    "executemany() does not take a template string; call execute() with a template for each row, "
    "or pass a str with the parameter sets"
)


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    yield connection
    connection.close()


def excerpt(line: str, part: str, number: int = 1) -> str:
    prefix = f"LINE {number}: "
    return f"\n\n{prefix}{line}\n{' ' * (len(prefix) + line.index(part))}{'^' * len(part)}"


def refused(call, message: str) -> None:
    with pytest.raises(duckdb.InvalidInputException) as info:
        call()
    assert str(info.value) == f"Invalid Input Error: {message}"


ENTRY_POINTS = {
    "execute": lambda con, query: con.execute(query),
    "sql": lambda con, query: con.sql(query),
    "query": lambda con, query: con.query(query),
    "from_query": lambda con, query: con.from_query(query),
    "cursor execute": lambda con, query: con.cursor().execute(query),
    "module execute": lambda con, query: duckdb.execute(query, connection=con),
    "module sql": lambda con, query: duckdb.sql(query, connection=con),
    "module query": lambda con, query: duckdb.query(query, connection=con),
    "module from_query": lambda con, query: duckdb.from_query(query, connection=con),
}


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_entry_points_accept_template(con, entry):
    x = 41
    assert ENTRY_POINTS[entry](con, t"SELECT {x} + 1 AS y").fetchall() == [(42,)]


def test_default_connection_accepts_template():
    x = 41
    assert duckdb.sql(t"SELECT {x} + 1").fetchall() == [(42,)]
    assert duckdb.execute(t"SELECT {x} + 1").fetchall() == [(42,)]


VALUES = [
    1,
    -5,
    2**64 - 1,
    2**70,
    1.5,
    "O'Reilly",
    "",
    b"\x00\xff",
    True,
    None,
    [1, None, 3],
    {"a": 1, "b": [2, 3]},
    datetime.date(2024, 1, 2),
    datetime.datetime(2024, 1, 2, 3, 4, 5),
    decimal.Decimal("1.25"),
    uuid.UUID(int=1),
]


@pytest.mark.parametrize("value", VALUES, ids=repr)
def test_value_binds_as_params_would(con, value):
    expected = con.execute("SELECT $1 AS v, typeof($1) AS t", [value]).fetchall()
    assert con.execute(t"SELECT {value} AS v, typeof({value}) AS t").fetchall() == expected
    assert con.sql(t"SELECT {value} AS v, typeof({value}) AS t").fetchall() == expected


def test_quote_and_semicolon_stay_data(con):
    con.execute("CREATE TABLE people (name VARCHAR)")
    name = "x'); DROP TABLE people;"
    con.execute(t"INSERT INTO people VALUES ({name})")
    assert con.execute("SELECT name FROM people").fetchall() == [(name,)]


def test_none_binds_null(con):
    value = None
    assert con.execute(t"SELECT {value} IS NULL, {value} = {value}").fetchall() == [(True, None)]


def test_repeated_variable_is_two_parameters(con):
    x = 3
    relation = con.sql(t"SELECT {x} * {x} AS y")
    assert "$1 * $2" in relation.sql_query()
    assert relation.fetchall() == [(9,)]


def test_any_expression_is_a_value(con):
    bounds = {"low": 1, "high": 3}
    query = t"SELECT * FROM range({bounds['low']}, {bounds['high'] + 1})"
    assert con.execute(query).fetchall() == [(1,), (2,), (3,)]


def test_empty_template_is_the_empty_query(con):
    assert con.execute(t"") is con.execute("") is None
    assert con.sql(t"") is con.sql("") is None


def test_template_without_interpolations(con):
    assert con.execute(t"SELECT 1").fetchall() == [(1,)]
    assert con.sql(t"SELECT 1").fetchall() == [(1,)]


def test_refusal_caret_under_interpolation(con):
    name = "bob"
    refused(
        lambda: con.execute(t"SELECT 'bob' LIKE '%{name}%'"),
        f"{{name}} {STRING_LITERAL}\n\nLINE 1: SELECT 'bob' LIKE '%{{name}}%'\n                            ^^^^^^",
    )


@pytest.mark.parametrize(
    ("query", "line"),
    [
        (t"SELECT 'x{1}'", "SELECT 'x{1}'"),
        (t"SELECT $$ {1} $$", "SELECT $$ {1} $$"),
        (t"SELECT E'\\'{1}'", "SELECT E'\\'{1}'"),
    ],
)
def test_interpolation_inside_string_literal_refused(con, query, line):
    refused(lambda: con.execute(query), f"{{1}} {STRING_LITERAL}" + excerpt(line, "{1}"))
    refused(lambda: con.sql(query), f"{{1}} {STRING_LITERAL}" + excerpt(line, "{1}"))


def test_interpolation_inside_quoted_identifier_refused(con):
    column = "a"
    message = (
        "{column} is inside a quoted identifier, where it would be part of the name, not a parameter; write "
        "{column:i} for a name"
    )
    refused(lambda: con.execute(t'SELECT 1 AS "x{column}"'), message + excerpt('SELECT 1 AS "x{column}"', "{column}"))
    refused(lambda: con.execute(t"SELECT 1 AS `x{column}`"), message + excerpt("SELECT 1 AS `x{column}`", "{column}"))


@pytest.mark.parametrize(
    ("query", "line"),
    [(t"SELECT 1 /* {1} */", "SELECT 1 /* {1} */"), (t"SELECT 1 /*{1}*/", "SELECT 1 /*{1}*/")],
)
def test_interpolation_inside_comment_refused(con, query, line):
    refused(
        lambda: con.execute(query),
        "{1} is inside a comment, where it would be ignored; take it out of the comment" + excerpt(line, "{1}"),
    )


@pytest.mark.parametrize(
    ("query", "line"),
    [
        (t"SELECT a{1}", "SELECT a{1}"),
        (t"SELECT {1}a", "SELECT {1}a"),
        (t"SELECT {1}0", "SELECT {1}0"),
        (t"SELECT 1{1}", "SELECT 1{1}"),
        (t"SELECT 'a'{1}", "SELECT 'a'{1}"),
        (t'SELECT {1}"alias"', 'SELECT {1}"alias"'),
        (t"SELECT 1 LIMIT{1}", "SELECT 1 LIMIT{1}"),
        (t"SELECT {1}.0", "SELECT {1}.0"),
    ],
)
def test_interpolation_glued_to_identifier_refused(con, query, line):
    refused(lambda: con.execute(query), f"{{1}} {GLUED}" + excerpt(line, "{1}"))


def test_fields_touching_each_other_refused(con):
    x, y, name = 1, 2, "a"
    refused(lambda: con.execute(t"SELECT {x}{y}"), f"{{x}} {GLUED}" + excerpt("SELECT {x}{y}", "{x}"))
    refused(
        lambda: con.execute(t"SELECT {name:i}{x} FROM (SELECT 1 AS a)"),
        "{name:i} touches the text next to it, so it is not a name of its own; put a space between them"
        + excerpt("SELECT {name:i}{x} FROM (SELECT 1 AS a)", "{name:i}"),
    )


def test_punctuation_next_to_a_field_is_accepted(con):
    x, y, schema, table = 2, 3, "s", "t"
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.t AS SELECT 7 AS a")
    assert con.execute(t"SELECT ({x}),{y}, -{x}, {x}::VARCHAR, [{x},{y}]").fetchall() == [(2, 3, -2, "2", [2, 3])]
    assert con.execute(t"SELECT a FROM {schema:i}.{table:i}").fetchall() == [(7,)]


MARKER = "which nothing gives a value; write the value in its place, as {value}"


@pytest.mark.parametrize(
    ("query", "marker"),
    [
        (t"SELECT ?, {1}", "?"),
        (t"SELECT ?1, {1}", "?1"),
        (t"SELECT $1, {1}", "$1"),
        (t"SELECT $name, {1}", "$name"),
        (t"SELECT ?", "?"),
    ],
)
def test_marker_in_template_text_refused(con, query, marker):
    line = "".join(part if isinstance(part, str) else "{1}" for part in query)
    refused(
        lambda: con.execute(query),
        f"The template text holds the parameter marker {marker}, {MARKER}" + excerpt(line, marker),
    )


def test_markers_in_statements_that_bind_them_late_are_refused(con):
    size = 16384
    line = "ATTACH ':memory:' AS aux (BLOCK_SIZE {size}, ROW_GROUP_SIZE $1)"
    refused(
        lambda: con.execute(t"ATTACH ':memory:' AS aux (BLOCK_SIZE {size}, ROW_GROUP_SIZE $1)"),
        f"The template text holds the parameter marker $1, {MARKER}" + excerpt(line, "$1"),
    )
    refused(
        lambda: con.execute(t"PREPARE p AS SELECT $1::INTEGER"),
        f"The template text holds the parameter marker $1, {MARKER}" + excerpt("PREPARE p AS SELECT $1::INTEGER", "$1"),
    )


def test_question_mark_quantifier_reads_as_a_marker_as_in_sql(con):
    pattern = "MATCH_RECOGNIZE (ORDER BY i MEASURES FIRST(i) AS x PATTERN (A?) DEFINE A AS true)"
    query = Template(f"SELECT x FROM (VALUES (1)) AS t(i) {pattern}")
    with pytest.raises(duckdb.InvalidInputException, match=r"holds the parameter marker \?"):
        con.execute(query)
    with pytest.raises(duckdb.InvalidInputException, match="Values were not provided"):
        con.sql(f"SELECT x FROM (VALUES (1)) AS t(i) {pattern}")


def test_template_with_params_refused(con):
    x = 1
    refused(lambda: con.execute(t"SELECT {x}", [1]), WITH_PARAMS)
    refused(lambda: con.sql(t"SELECT {x}", params=[1]), WITH_PARAMS)
    refused(lambda: con.sql(t"SELECT {x}", params={"a": 1}), WITH_PARAMS)
    refused(lambda: duckdb.execute(t"SELECT {x}", [1], connection=con), WITH_PARAMS)
    refused(lambda: duckdb.sql(t"SELECT {x}", params=[1], connection=con), WITH_PARAMS)


def test_template_with_empty_params_accepted(con):
    x = 1
    assert con.execute(t"SELECT {x}", []).fetchall() == [(1,)]
    assert con.execute(t"SELECT {x}", ()).fetchall() == [(1,)]
    assert con.sql(t"SELECT {x}", params={}).fetchall() == [(1,)]


def test_executemany_refuses_template(con):
    con.execute("CREATE TABLE t (a INT)")
    x = 1
    refused(lambda: con.executemany(t"INSERT INTO t VALUES ({x})", [[1], [2]]), EXECUTEMANY)
    refused(lambda: duckdb.executemany(t"INSERT INTO t VALUES ({x})", [[1]], connection=con), EXECUTEMANY)
    assert con.execute("SELECT count(*) FROM t").fetchall() == [(0,)]


@pytest.mark.parametrize(
    ("query", "field"), [(t"SELECT {1!r}", "{1!r}"), (t"SELECT {1!s}", "{1!s}"), (t"SELECT {1!a}", "{1!a}")]
)
def test_conversion_refused(con, query, field):
    refused(
        lambda: con.execute(query),
        f"{field} converts its value, which a query does not do; write {{1}}, which takes the value as it is"
        + excerpt(f"SELECT {field}", field),
    )


def test_conversion_refused_before_spec(con):
    name = "t"
    refused(
        lambda: con.execute(t"SELECT * FROM {name!s:i}"),
        "{name!s:i} converts its value, which a query does not do; write {name:i}, which takes the value as it is"
        + excerpt("SELECT * FROM {name!s:i}", "{name!s:i}"),
    )


def test_identifier_spec_quotes_str(con):
    con.execute('CREATE TABLE "Mixed Case" AS SELECT 1 AS "Col"')
    table, column = "Mixed Case", "Col"
    relation = con.sql(t"SELECT {column:i} FROM {table:i}")
    assert relation.fetchall() == [(1,)]
    assert '"Col"' in relation.sql_query()
    assert '"Mixed Case"' in relation.sql_query()


def test_identifier_spec_quotes_embedded_quote_and_dot(con):
    name = 'we"ird.name'
    con.execute(t"CREATE TABLE {name:i} (a INT)")
    assert con.execute("SELECT table_name FROM duckdb_tables()").fetchall() == [(name,)]


def test_identifier_spec_cannot_escape_the_name(con):
    con.execute("CREATE TABLE t (a INT)")
    hostile = 'a" FROM t; DROP TABLE t;'
    with pytest.raises(duckdb.BinderException, match="not found"):
        con.execute(t"SELECT {hostile:i} FROM t")
    assert con.execute("SELECT count(*) FROM t").fetchall() == [(0,)]


def test_identifier_spec_tuple_is_qualified(con):
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.t AS SELECT 5 AS a")
    name = ("s", "t")
    full = ("memory", "s", "t")
    assert con.execute(t"SELECT a FROM {name:i}").fetchall() == [(5,)]
    assert con.execute(t"SELECT a FROM {full:i}").fetchall() == [(5,)]
    assert '"s"."t"' in con.sql(t"SELECT a FROM {name:i}").sql_query()


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (1, "int"),
        (None, "NoneType"),
        (b"t", "bytes"),
        (["s", "t"], "list"),
        (("s", 1), "a tuple holding int"),
        ((), "an empty tuple"),
        ("", "an empty str"),
        (("s", ""), "a tuple holding an empty str"),
    ],
)
def test_identifier_spec_refuses_non_str(con, value, kind):
    refused(
        lambda: con.execute(t"SELECT * FROM {value:i}"),
        f"{{value:i}} is a name, so its value must be a str or a tuple of str, not {kind}"
        + excerpt("SELECT * FROM {value:i}", "{value:i}"),
    )


NAME = "a"
NAME_GLUED = "touches the text next to it, so it is not a name of its own; put a space between them"


@pytest.mark.parametrize(
    ("query", "line", "message"),
    [
        (
            t"SELECT 'x{NAME:i}'",
            "SELECT 'x{NAME:i}'",
            "is inside a string literal, where it would be text, not a name; take it out of the literal",
        ),
        (
            t"SELECT 1 /* {NAME:i} */",
            "SELECT 1 /* {NAME:i} */",
            "is inside a comment, where it would be ignored; take it out of the comment",
        ),
        (
            t'SELECT 1 AS "{NAME:i}"',
            'SELECT 1 AS "{NAME:i}"',
            "is inside a quoted identifier, and quotes itself; take away the quotes around it",
        ),
        (t"SELECT {NAME:i}{NAME:i}", "SELECT {NAME:i}{NAME:i}", NAME_GLUED),
        (t"SELECT * FROM prefix_{NAME:i}", "SELECT * FROM prefix_{NAME:i}", NAME_GLUED),
        (t"SELECT {NAME:i}_total FROM t", "SELECT {NAME:i}_total FROM t", NAME_GLUED),
        (t"SELECT 1{NAME:i}", "SELECT 1{NAME:i}", NAME_GLUED),
        (t"SELECT 'x'{NAME:i}", "SELECT 'x'{NAME:i}", NAME_GLUED),
        (t"SELECT ${NAME:i}", "SELECT ${NAME:i}", NAME_GLUED),
    ],
)
def test_identifier_spec_must_stand_alone(con, query, line, message):
    refused(lambda: con.execute(query), f"{{NAME:i}} {message}" + excerpt(line, "{NAME:i}"))


@pytest.mark.parametrize(
    ("query", "field", "spec"),
    [
        (t"SELECT {1:d}", "{1:d}", "d"),
        (t"SELECT {1:s}", "{1:s}", "s"),
        (t"SELECT {1:l}", "{1:l}", "l"),
        (t"SELECT {1:>10}", "{1:>10}", ">10"),
    ],
)
def test_unknown_spec_refused(con, query, field, spec):
    refused(
        lambda: con.execute(query),
        f"{field} has the format spec '{spec}', which a query does not take; write {{1}} for a value, {{1:i}} for "
        "a name, or {1:q} for a template" + excerpt(f"SELECT {field}", field),
    )


def test_nested_template_splices_and_numbers_in_order(con):
    first, low, high = 0, 2, 4
    where = t"a >= {low} AND a < {high}"
    relation = con.sql(t"SELECT a + {first} AS a FROM range(10) t(a) WHERE {where:q} ORDER BY a")
    assert relation.fetchall() == [(2,), (3,)]
    assert "a + $1" in relation.sql_query()
    assert "a >= $2 AND a < $3" in relation.sql_query()


def test_nested_templates_compose_filters(con):
    low, high = 1, 5
    filters = [t"a > {low}", t"a < {high}"]
    joined = filters[0] + t" AND " + filters[1]
    assert con.execute(t"SELECT count(*) FROM range(10) t(a) WHERE {joined:q}").fetchall() == [(3,)]


def test_nested_template_is_spliced_inline(con):
    where, order = t"WHERE a > {1} ", t"ORDER BY a DESC"
    assert con.execute(t"SELECT a FROM range(4) t(a) {where:q}{order:q}").fetchall() == [(3,), (2,)]
    digit = t"1"
    assert con.execute(t"SELECT 2{digit:q}").fetchall() == [(21,)]


def test_nested_template_without_q_refused(con):
    where = t"a > {1}"
    line = "SELECT * FROM range(3) t(a) WHERE {where}"
    refused(
        lambda: con.execute(t"SELECT * FROM range(3) t(a) WHERE {where}"),
        "{where} is a template; write {where:q} to splice it into the query" + excerpt(line, "{where}"),
    )


def test_q_on_non_template_refused(con):
    where = "a > 1"
    line = "SELECT * FROM range(3) t(a) WHERE {where:q}"
    refused(
        lambda: con.execute(t"SELECT * FROM range(3) t(a) WHERE {where:q}"),
        "{where:q} splices a template, but its value is a str; write {where} for a value, or {where:i} for a name"
        + excerpt(line, "{where:q}"),
    )


def test_nested_template_markers_refused(con):
    where = t"a > ?"
    refused(
        lambda: con.execute(t"SELECT * FROM range(3) t(a) WHERE {where:q}"),
        "The template text of {where:q} holds the parameter marker ?, which nothing gives a value; write the value "
        "in its place, as {value}" + excerpt("a > ?", "?"),
    )


def test_refusal_in_nested_template_names_it(con):
    name = "x"
    inner = t"s = '{name}'"
    outer = t"{inner:q} AND true"
    refused(
        lambda: con.execute(t"SELECT * FROM (SELECT 'x' AS s) WHERE {outer:q}"),
        f"{{name}} in {{inner:q}} in {{outer:q}} {STRING_LITERAL}" + excerpt("s = '{name}'", "{name}"),
    )


def test_refusal_in_multiline_nested_template_names_its_line(con):
    name = "x"
    where = t"""a = 1
AND s = '{name}'"""
    refused(
        lambda: con.execute(t"SELECT * FROM (SELECT 1 AS a, 'x' AS s) WHERE {where:q}"),
        f"{{name}} in {{where:q}} {STRING_LITERAL}" + excerpt("AND s = '{name}'", "{name}", 2),
    )


def test_refusal_in_earlier_statement(con):
    x = 1
    refused(
        lambda: con.execute(t"SELECT '{x}'; SELECT 1"),
        f"{{x}} {STRING_LITERAL}" + excerpt("SELECT '{x}'; SELECT 1", "{x}"),
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_refusal_names_the_line_as_the_engine_does(con, newline):
    x = 1
    query = Template(f"SELECT 1;{newline}SELECT '", Interpolation(x, "x"), "'")
    refused(lambda: con.execute(query), f"{{x}} {STRING_LITERAL}" + excerpt("SELECT '{x}'", "{x}", 2))


def test_refusal_caret_counts_wide_characters_as_the_engine_does(con):
    x = 1
    refused(
        lambda: con.execute(t"SELECT '界', '{x}'"),
        f"{{x}} {STRING_LITERAL}\n\nLINE 1: SELECT '界', '{{x}}'\n{' ' * 22}^^^",
    )


@pytest.mark.parametrize(("text", "columns"), [("a\u200bb", 2), ("e\u0301", 1), ("a\u00adb", 3)])
def test_refusal_caret_counts_zero_width_characters_as_the_engine_does(con, text, columns):
    x = 1
    query = Template(f"SELECT '{text}', '", Interpolation(x, "x"), "'")
    line = f"SELECT '{text}', '{{x}}'"
    refused(lambda: con.execute(query), f"{{x}} {STRING_LITERAL}\n\nLINE 1: {line}\n{' ' * (8 + 12 + columns)}^^^")


def test_text_before_a_field_may_be_non_ascii(con):
    x = "é"
    assert con.execute(t"SELECT 'é' || 'ü', {x}").fetchall() == [("éü", "é")]
    refused(
        lambda: con.execute(t"SELECT 'éü' LIKE '%{x}%'"),
        f"{{x}} {STRING_LITERAL}" + excerpt("SELECT 'éü' LIKE '%{x}%'", "{x}"),
    )


@pytest.mark.parametrize("name", ["a\udcff", ("s", "\udcff")])
def test_surrogate_in_a_name_refused(con, name):
    with pytest.raises(duckdb.InvalidInputException, match="cannot encode str as UTF-8, surrogate"):
        con.execute(t"SELECT * FROM {name:i}")
    with pytest.raises(duckdb.InvalidInputException, match="cannot encode str as UTF-8, surrogate"):
        con.sql(t"SELECT * FROM {name:i}")


def test_surrogate_in_template_text_refused(con):
    query = Template("SELECT '\udcff', ", Interpolation(1, "1"))
    with pytest.raises(duckdb.InvalidInputException, match="cannot encode str as UTF-8, surrogate"):
        con.execute(query)


def test_refusal_multiline_names_line(con):
    name = "bob"
    query = t"""SELECT *
FROM range(3)
WHERE 'bob' = '{name}'"""
    refused(lambda: con.execute(query), f"{{name}} {STRING_LITERAL}" + excerpt("WHERE 'bob' = '{name}'", "{name}", 3))


RELATION = (
    "is a relation, which a template cannot take as a value; write the name of a variable that holds it in the "
    "query text"
)


def test_relation_value_refused(con):
    relation = con.sql("SELECT 1 AS a")
    line = "SELECT * FROM {relation}"
    refused(lambda: con.execute(t"SELECT * FROM {relation}"), f"{{relation}} {RELATION}" + excerpt(line, "{relation}"))
    refused(lambda: con.sql(t"SELECT * FROM {relation}"), f"{{relation}} {RELATION}" + excerpt(line, "{relation}"))
    relations = [relation]
    line = "SELECT * FROM {relations[0]}"
    refused(
        lambda: con.execute(t"SELECT * FROM {relations[0]}"),
        f"{{relations[0]}} {RELATION}" + excerpt(line, "{relations[0]}"),
    )


@needs_pandas
def test_dataframe_value_fails_as_a_parameter(con):
    import pandas as pd

    frame = pd.DataFrame({"a": [1]})
    with pytest.raises(duckdb.NotImplementedException, match="Unable to transform python value") as info:
        con.execute(t"SELECT {frame}")
    assert info.value.__notes__ == ["The statement came from a template; $1 is {frame}:\n    SELECT {frame}"]


def test_sql_template_statement_that_writes_runs_now(con):
    con.execute("CREATE TABLE t (a INT)")
    x = 5
    assert con.sql(t"INSERT INTO t VALUES ({x})").fetchall() == [(1,)]
    assert con.execute("SELECT a FROM t").fetchall() == [(5,)]
    name = "u"
    via_template = con.sql(t"CREATE TABLE {name:i} (b INT)")
    via_text = con.sql("CREATE TABLE v (b INT)")
    assert (via_template.columns, via_template.fetchall()) == (via_text.columns, via_text.fetchall())
    assert con.execute("SELECT table_name FROM duckdb_tables() ORDER BY 1").fetchall() == [("t",), ("u",), ("v",)]


def test_sql_template_is_lazy(con):
    con.execute("CREATE TABLE t (a INT)")
    x = 0
    relation = con.sql(t"SELECT count(*) FROM t WHERE a > {x}")
    con.execute("INSERT INTO t VALUES (1), (2)")
    assert relation.fetchall() == [(2,)]


def test_sql_template_values_copied_at_call(con):
    values = [1, 2]
    relation = con.sql(t"SELECT {values} AS v")
    values.append(3)
    assert relation.fetchall() == [([1, 2],)]


def test_sql_query_shows_markers(con):
    x, y = 1, "a"
    assert "SELECT $1 AS x, $2 AS y" in con.sql(t"SELECT {x} AS x, {y} AS y").sql_query()


def test_sql_template_view_is_the_view_params_make(con):
    value = "it's"
    con.sql(t"SELECT {value} AS v").create_view("from_template")
    con.sql("SELECT $1 AS v", params=[value]).create_view("from_params")
    assert con.sql("SELECT v FROM from_template").fetchall() == con.sql("SELECT v FROM from_params").fetchall()
    assert con.sql("SELECT v FROM from_template").fetchall() == [("it's",)]


def test_template_relations_join(con):
    x, y = 1, 2
    left = con.sql(t"SELECT {x} AS k, 'l' AS l")
    right = con.sql(t"SELECT {y} - 1 AS k, 'r' AS r")
    assert left.join(right, "k").fetchall() == [(1, "l", "r")]


def test_interpolation_in_earlier_statement(con):
    x = 1
    with pytest.raises(duckdb.NotImplementedException, match="only supported for the last statement"):
        con.execute(t"SELECT {x}; SELECT 2")
    with pytest.raises(duckdb.NotImplementedException, match="only supported for the last statement"):
        con.sql(t"SELECT {x}; SELECT 2")
    assert con.execute(t"CREATE TABLE s AS SELECT 1 AS v; SELECT v + {x} FROM s").fetchall() == [(2,)]


@needs_pandas
def test_text_reads_python_object_by_name(con):
    import pandas as pd

    frame = pd.DataFrame({"a": [1, 2, 3]})
    low = 1
    assert con.sql(t"SELECT a FROM frame WHERE a > {low} ORDER BY a").fetchall() == [(2,), (3,)]
    assert con.execute(t"SELECT a FROM frame WHERE a > {low} ORDER BY a").fetchall() == [(2,), (3,)]
    assert frame.shape == (3, 1)


def test_parser_error_keeps_engine_message_and_gets_note(con):
    name, user_id = "x", 1
    with pytest.raises(duckdb.ParserException) as direct:
        con.execute("SELECT $1 FORM users WHERE id = $2", [name, user_id])
    note = (
        "The statement came from a template; $1 is {name}, $2 is {user_id}:\n"
        "    SELECT {name} FORM users WHERE id = {user_id}"
    )
    for call in (con.execute, con.sql):
        with pytest.raises(duckdb.ParserException) as info:
            call(t"SELECT {name} FORM users WHERE id = {user_id}")
        assert str(info.value) == str(direct.value)
        assert info.value.__notes__ == [note]


def test_binder_error_gets_note(con):
    x = 1
    with pytest.raises(duckdb.BinderException, match='Referenced column "b" not found') as info:
        con.sql(t"SELECT b, {x} FROM range(1) t(a)")
    assert info.value.__notes__ == [
        "The statement came from a template; $1 is {x}:\n    SELECT b, {x} FROM range(1) t(a)"
    ]


def test_conversion_error_gets_note(con):
    x, thing = 1, object()
    with pytest.raises(duckdb.NotImplementedException, match="Unable to transform python value") as info:
        con.execute(t"SELECT {x}, {thing}")
    assert info.value.__notes__ == [
        "The statement came from a template; $1 is {x}, $2 is {thing}:\n    SELECT {x}, {thing}"
    ]


def test_note_names_nested_markers(con):
    low = 1
    where = t"a > {low} AND b"
    with pytest.raises(duckdb.BinderException) as info:
        con.execute(t"SELECT {low} FROM range(3) t(a) WHERE {where:q}")
    assert info.value.__notes__ == [
        "The statement came from a template; $1 is {low}, $2 is {low} in {where:q}:\n"
        "    SELECT {low} FROM range(3) t(a) WHERE {where:q}"
    ]


def test_note_without_parameters(con):
    table = "missing"
    with pytest.raises(duckdb.CatalogException) as info:
        con.execute(t"SELECT * FROM {table:i}")
    assert info.value.__notes__ == ["The statement came from a template:\n    SELECT * FROM {table:i}"]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_note_indents_every_line(con, newline):
    x = 1
    with pytest.raises(duckdb.ParserException) as info:
        con.execute(Template("SELECT ", Interpolation(x, "x"), f"{newline}FORM t"))
    assert info.value.__notes__ == ["The statement came from a template; $1 is {x}:\n    SELECT {x}\n    FORM t"]


def test_error_outside_duckdb_not_annotated(con):
    class Unsized(list):
        def __len__(self) -> int:
            message = "no length"
            raise RuntimeError(message)

    value = Unsized([1])
    with pytest.raises(RuntimeError, match="no length") as info:
        con.execute(t"SELECT {value}")
    assert not getattr(info.value, "__notes__", None)


def test_terminal_error_has_no_note(con):
    text = "abc"
    relation = con.sql(t"SELECT {text}::INT AS x")
    with pytest.raises(duckdb.ConversionException) as info:
        relation.fetchall()
    assert not getattr(info.value, "__notes__", None)


def test_template_built_without_literal_syntax(con):
    query = Template("SELECT ", Interpolation(5, "five"), " + 1")
    assert con.execute(query).fetchall() == [(6,)]

"""The constructors and methods of the earlier expression API build the same tree as `col`, `lit` and `fn`."""

from __future__ import annotations

import datetime
import decimal
import re

import pytest

import duckdb
from duckdb import (
    CaseExpression,
    CoalesceOperator,
    ColumnExpression,
    ConstantExpression,
    DefaultExpression,
    FunctionExpression,
    LambdaExpression,
    SQLExpression,
    StarExpression,
    col,
    fn,
    lit,
    when,
)
from duckdb import _expression as expression_module
from duckdb._expression import string_literals
from duckdb.sqltypes import BIGINT, DOUBLE, INTEGER, SQLNULL, TIMESTAMP, VARCHAR, DuckDBPyType
from duckdb.value.constant import (
    BinaryValue,
    BlobValue,
    DateValue,
    IntegerValue,
    ListValue,
    NullValue,
    StringValue,
    Value,
)


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    yield connection
    connection.close()


def value(con: duckdb.DuckDBPyConnection, expression: object, source: str = "") -> object:
    return con.execute(f"SELECT {expression}{source}").fetchall()[0][0]


class TestTheModuleExportsThePythonTree:
    def test_one_expression_class(self) -> None:
        assert duckdb.Expression is expression_module.Expression
        assert expression_module.Expr is duckdb.Expression
        for built in (
            ColumnExpression("a"),
            ConstantExpression(1),
            FunctionExpression("f"),
            CaseExpression(col("a") == 1, 2),
            CoalesceOperator(1),
            StarExpression(),
            SQLExpression("1"),
            LambdaExpression("x", col("x")),
            DefaultExpression(),
        ):
            assert isinstance(built, duckdb.Expression)

    def test_the_module_functions_are_exported(self) -> None:
        names = [
            "col",
            "lit",
            "fn",
            "sql_expr",
            "star",
            "when",
            "coalesce",
            "count_all",
            "row_number",
            "rank",
            "dense_rank",
            "ntile",
            "lag",
            "lead",
            "first_value",
            "last_value",
        ]
        for name in names:
            assert getattr(duckdb, name) is getattr(expression_module, name)
            assert name in duckdb.__all__

    def test_compat_and_new_constructors_combine(self, con: duckdb.DuckDBPyConnection) -> None:
        combined = (ColumnExpression("a") + lit(1)) * fn("abs", ConstantExpression(-2))
        assert str(combined) == '(("a" + 1) * abs(-2))'
        assert value(con, combined, " FROM (SELECT 2 AS a)") == 6


class TestColumnExpression:
    def test_parts_are_a_qualified_name(self) -> None:
        assert str(ColumnExpression("t", "a")) == '"t"."a"'
        assert str(ColumnExpression("c", "s", "t", "a", "field")) == '"c"."s"."t"."a"."field"'

    def test_a_name_is_never_split(self, con: duckdb.DuckDBPyConnection) -> None:
        assert str(ColumnExpression("a.b")) == '"a.b"'
        assert value(con, ColumnExpression("a.b"), " FROM (SELECT 1 AS \"a.b\", {'b': 2} AS a)") == 1
        assert value(con, ColumnExpression("a", "b"), " FROM (SELECT 1 AS \"a.b\", {'b': 2} AS a)") == 2

    def test_the_one_name_star_is_the_star(self, con: duckdb.DuckDBPyConnection) -> None:
        # ColumnExpression("*") is the star, as before.
        assert str(ColumnExpression("*")) == "*"
        assert con.execute(f"SELECT {ColumnExpression('*')} FROM (SELECT 1 AS a, 2 AS b)").fetchall() == [(1, 2)]
        assert str(ColumnExpression("t", "*")) == '"t"."*"'

    def test_the_same_as_col(self) -> None:
        assert str(ColumnExpression("x")) == str(col("x"))


class TestConstantExpression:
    # The text is the seam's (`_duckdb._value_literal`), so these pin what it means, not how it is spelled.
    @pytest.mark.parametrize(
        ("constant", "result"),
        [
            (Value(5, INTEGER), 5),
            (IntegerValue(1), 1),
            (BinaryValue(100), 100),
            (StringValue("it's"), "it's"),
            (BlobValue(b"\x01"), b"\x01"),
            (DateValue(datetime.date(2020, 1, 2)), datetime.date(2020, 1, 2)),
            (ListValue([1, 2], INTEGER), [1, 2]),
            (NullValue(), None),
            (Value("7", BIGINT), 7),
        ],
        ids=["value", "integer", "tinyint", "string", "blob", "date", "list", "null", "string as bigint"],
    )
    def test_a_value_is_a_cast_to_its_type(
        self, con: duckdb.DuckDBPyConnection, constant: Value, result: object
    ) -> None:
        expression = ConstantExpression(constant)
        assert value(con, expression) == result
        assert con.execute(f"SELECT typeof({expression})").fetchall()[0][0] == str(constant.type)

    def test_a_value_is_an_operand_too(self, con: duckdb.DuckDBPyConnection) -> None:
        overflow = ColumnExpression("s") * Value(100, duckdb.sqltypes.TINYINT)
        assert str(overflow).startswith('("s" * ')
        with pytest.raises(duckdb.OutOfRangeException, match="Overflow in multiplication of INT16"):
            value(con, overflow, " FROM (SELECT 3000::SMALLINT AS s)")

    def test_a_value_is_a_snapshot(self) -> None:
        items = [1]
        expression = ConstantExpression(Value(items, DuckDBPyType("INTEGER[]")))
        items.append(2)
        assert duckdb.connect().execute(f"SELECT {expression}").fetchall() == [([1],)]

    @pytest.mark.parametrize(
        ("target", "obj"),
        [
            ("STRUCT(a INTEGER, b INTEGER)", {"A": 1, "B": 2}),
            ("TUPLE(INTEGER, INTEGER)", {"k1": 1, "k2": 2}),
            ("MAP(INTEGER, VARCHAR)", {1: "x"}),
            ("VARCHAR[]", ["a'b", "c, d", None]),
            ("STRUCT(s VARCHAR)[]", [{"s": "a]b"}]),
            ("FLOAT", 0.1),
            ("TIMESTAMP WITH TIME ZONE", datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC)),
        ],
        ids=["struct keys in another case", "unnamed struct", "map", "texts", "nested text", "float", "timestamptz"],
    )
    def test_a_value_means_what_it_means_as_a_parameter(
        self, con: duckdb.DuckDBPyConnection, target: str, obj: object
    ) -> None:
        # The literal is the value converted to its type, as a Value parameter converts it.
        constant = Value(obj, con.type(target))
        assert value(con, ConstantExpression(constant)) == con.execute("SELECT ?", [constant]).fetchall()[0][0]

    def test_a_value_its_type_cannot_hold_is_refused_where_it_is_built(self) -> None:
        with pytest.raises(duckdb.Error):
            ConstantExpression(Value("not a number", INTEGER))

    @pytest.mark.parametrize(
        ("constant", "text"),
        [
            ("x", "'x'"),
            (5, "5"),
            (None, "NULL"),
            (1.5, "1.5::DOUBLE"),
            (decimal.Decimal("1.5"), "CAST('1.5' AS DECIMAL(2, 1))"),
            (datetime.date(2020, 1, 2), "'2020-01-02'::DATE"),
        ],
    )
    def test_a_python_value(self, constant: object, text: str) -> None:
        assert str(ConstantExpression(constant)) == text
        assert str(ConstantExpression(constant)) == str(lit(constant))

    def test_a_numpy_array_is_a_list(self, con: duckdb.DuckDBPyConnection) -> None:
        np = pytest.importorskip("numpy")
        assert str(ConstantExpression(np.array([1, 2]))) == "[1, 2]"
        assert str(ConstantExpression(np.array([[1.5], [2.5]]))) == "[[1.5::DOUBLE], [2.5::DOUBLE]]"
        assert value(con, ConstantExpression(np.array(["a", "b"]))) == ["a", "b"]

    @pytest.mark.parametrize("constant", [object(), col("a"), lambda x: x], ids=["object", "expression", "callable"])
    def test_what_is_no_value_is_refused(self, constant: object) -> None:
        with pytest.raises(TypeError, match="takes a value"):
            ConstantExpression(constant)


class TestFunctionExpression:
    def test_arguments_are_operands(self, con: duckdb.DuckDBPyConnection) -> None:
        call = FunctionExpression("concat", ColumnExpression("s"), "-", 1)
        assert str(call) == "concat(\"s\", '-', 1)"
        assert value(con, call, " FROM (SELECT 'a' AS s)") == "a-1"

    def test_an_operator_by_name(self, con: duckdb.DuckDBPyConnection) -> None:
        difference = FunctionExpression("-", ColumnExpression("b"), ColumnExpression("a"))
        assert value(con, difference, " FROM (SELECT 1 AS a, 5 AS b)") == 4
        assert value(con, FunctionExpression("~~", "abc", "a%")) is True

    def test_no_arguments(self, con: duckdb.DuckDBPyConnection) -> None:
        assert str(FunctionExpression("pi")) == "pi()"
        assert value(con, FunctionExpression("pi")) == pytest.approx(3.14159, rel=1e-5)


class TestCaseExpression:
    def test_unclosed_is_else_null(self, con: duckdb.DuckDBPyConnection) -> None:
        case = CaseExpression(ColumnExpression("a") > 1, 5)
        assert str(case) == '(CASE WHEN ("a" > 1) THEN 5 ELSE NULL END)'
        assert value(con, case, " FROM (SELECT 1 AS a)") is None

    def test_when_adds_a_branch_and_otherwise_closes(self, con: duckdb.DuckDBPyConnection) -> None:
        a = ColumnExpression("a")
        case = CaseExpression(a > 1, "many").when(a == 1, "one").otherwise("none")
        assert str(case) == "(CASE WHEN (\"a\" > 1) THEN 'many' WHEN (\"a\" = 1) THEN 'one' ELSE 'none' END)"
        assert [value(con, case, f" FROM (SELECT {n} AS a)") for n in (0, 1, 2)] == ["none", "one", "many"]

    def test_each_step_is_a_new_expression(self) -> None:
        first = CaseExpression(col("a") == 1, 1)
        second = first.when(col("a") == 2, 2)
        closed = second.otherwise(3)
        assert str(first) == '(CASE WHEN ("a" = 1) THEN 1 ELSE NULL END)'
        assert str(second) == '(CASE WHEN ("a" = 1) THEN 1 WHEN ("a" = 2) THEN 2 ELSE NULL END)'
        assert str(closed.otherwise(4)) == '(CASE WHEN ("a" = 1) THEN 1 WHEN ("a" = 2) THEN 2 ELSE 4 END)'

    def test_a_branch_after_otherwise_keeps_the_fallback(self) -> None:
        case = CaseExpression(col("a") == 1, 1).otherwise(0).when(col("a") == 2, 2)
        assert str(case) == '(CASE WHEN ("a" = 1) THEN 1 WHEN ("a" = 2) THEN 2 ELSE 0 END)'

    def test_the_methods_extend_a_case_from_when_too(self) -> None:
        case = when(col("a") == 1).then(1).end().when(col("a") == 2, 2)
        assert str(case) == '(CASE WHEN ("a" = 1) THEN 1 WHEN ("a" = 2) THEN 2 END)'

    def test_only_a_case_takes_when_and_otherwise(self) -> None:
        with pytest.raises(ValueError, match="CaseExpression"):
            col("a").when(col("b") == 1, 1)
        with pytest.raises(ValueError, match="CaseExpression"):
            col("a").otherwise(1)

    def test_a_bare_string_is_no_condition(self) -> None:
        with pytest.raises(TypeError, match="takes an expression"):
            CaseExpression("a", 1)

    @pytest.mark.parametrize("condition", [True, 1, 1.5], ids=["bool", "int", "float"])
    def test_a_value_condition_is_lifted_as_any_value_is(
        self, con: duckdb.DuckDBPyConnection, condition: object
    ) -> None:
        case = CaseExpression(col("a") == 9, 1).when(condition, 2)
        assert str(case) == str(CaseExpression(col("a") == 9, 1).when(lit(condition), 2))
        assert value(con, case.otherwise(3), " FROM (SELECT 2 AS a)") == 2


class TestCoalesceOperator:
    def test_the_first_value_that_is_not_null(self, con: duckdb.DuckDBPyConnection) -> None:
        assert str(CoalesceOperator(None, ColumnExpression("a"), "x")) == "COALESCE(NULL, \"a\", 'x')"
        assert value(con, CoalesceOperator(None, ColumnExpression("a"), 42), " FROM (SELECT NULL::INT AS a)") == 42

    def test_at_least_one_argument(self) -> None:
        with pytest.raises(duckdb.InvalidInputException, match="at least one argument"):
            CoalesceOperator()


class TestStarExpression:
    def test_names_and_qualified_columns(self) -> None:
        assert str(StarExpression()) == "*"
        assert str(StarExpression(exclude=[])) == "*"
        assert str(StarExpression(exclude="a")) == '* EXCLUDE ("a")'
        excluded = StarExpression(exclude=["a.b", ColumnExpression("t", "c"), ColumnExpression("d")])
        assert str(excluded) == '* EXCLUDE ("a.b", "t"."c", "d")'

    def test_a_qualified_exclusion_runs(self, con: duckdb.DuckDBPyConnection) -> None:
        source = "(SELECT 1 AS id, 2 AS x) a, (SELECT 3 AS id) b"
        assert con.execute(
            f"SELECT {StarExpression(exclude=[ColumnExpression('b', 'id')])} FROM {source}"
        ).fetchall() == [(1, 2)]

    def test_only_columns_are_excluded(self) -> None:
        with pytest.raises(ValueError, match="Only ColumnExpressions"):
            StarExpression(exclude=[ConstantExpression(1)])
        with pytest.raises(ValueError, match="Only ColumnExpressions"):
            StarExpression(exclude=[1])
        with pytest.raises(ValueError, match="1 to 4 name parts"):
            StarExpression(exclude=[ColumnExpression("a", "b", "c", "d", "e")])


class TestSQLExpression:
    @pytest.mark.parametrize(
        ("text", "name", "body", "alias"),
        [
            ("sum(x) AS total", "total", "sum(x)", 'AS "total"'),
            ('sum(x) AS "Total"', "Total", "sum(x)", 'AS "Total"'),
            ("x total", "total", "x", 'AS "total"'),
            ('x AS "we""ird"', 'we"ird', "x", 'AS "we""ird"'),
            ('x AS "x AS y"', "x AS y", "x", 'AS "x AS y"'),
            ("CAST(x AS INTEGER) AS n", "n", "CAST(x AS INTEGER)", 'AS "n"'),
            ("x + 1 AS n", "n", "x + 1", 'AS "n"'),
        ],
    )
    def test_an_alias_names_the_column(
        self, con: duckdb.DuckDBPyConnection, text: str, name: str, body: str, alias: str
    ) -> None:
        # The alias the parser finds is split off and rendered by the layer; the rest is the text as written.
        expression = SQLExpression(text)
        assert expression.get_name() == name
        assert str(expression) == body
        assert expression.as_select().endswith(alias)
        described = con.execute(f"SELECT {expression.as_select()} FROM (SELECT 2 AS x)")
        assert [d[0] for d in described.description] == [name]
        assert expression.alias("other").as_select().endswith(' AS "other"')

    @pytest.mark.parametrize(
        ("text", "parsed"),
        [
            ("CAST(x AS INTEGER)", "CAST(x AS INTEGER)"),
            ("x::INTEGER", "CAST(x AS INTEGER)"),
            ("'AS' || x", "('AS' || x)"),
        ],
    )
    def test_an_as_inside_is_no_alias(self, text: str, parsed: str) -> None:
        # The name is the engine's for the column, while the SQL stays as written.
        assert SQLExpression(text).get_name() == parsed
        assert str(SQLExpression(text)) == text

    @pytest.mark.parametrize(
        ("text", "written"),
        [
            ("a % 2 = 0 AND b LIKE '%o%'", "a % 2 = 0 AND b LIKE '%o%'"),
            ("UPPER(b)", "UPPER(b)"),
            ("1;", "1"),
        ],
    )
    def test_the_text_renders_as_written(self, text: str, written: str) -> None:
        # The text is checked by the parser and spliced unchanged; a trailing semicolon stays out of it.
        assert str(SQLExpression(text)) == written

    def test_the_connection_s_parser_settings_apply(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute("SET regex_match_operator_semantics = 'full'")
        assert con.sql("SELECT 'abc' AS s").project(SQLExpression("s ~ 'b'")).fetchall() == [(False,)]

    def test_a_parsed_condition_still_runs(self, con: duckdb.DuckDBPyConnection) -> None:
        expression = SQLExpression("a % 2 = 0 AND b LIKE '%o%'")
        assert value(con, expression, " FROM (SELECT 2 AS a, 'two' AS b)") is True

    @pytest.mark.parametrize("text", ["t.*", "#1", "COLUMNS('a')"])
    def test_star_like_forms_survive_the_parentheses(self, con: duckdb.DuckDBPyConnection, text: str) -> None:
        assert value(con, SQLExpression(text), " FROM (SELECT 7 AS a) t") == 7

    @pytest.mark.parametrize(
        ("text", "error", "message"),
        [
            ("", duckdb.ParserException, "SELECT clause without selection list"),
            ("SELECT *", duckdb.ParserException, 'syntax error at or near "SELECT"'),
            ("1, 2", duckdb.InvalidInputException, "found 2 expressions"),
        ],
    )
    def test_checked_by_the_parser_where_built(self, text: str, error: type[Exception], message: str) -> None:
        with pytest.raises(error, match=message):
            SQLExpression(text)

    # The seam's own refusal names the first clause the text holds, so each case holds only the clause it names.
    @pytest.mark.parametrize(
        ("text", "holds"),
        [
            ("1 LIMIT 2", "a modifier"),
            ("1 ORDER BY 1", "a modifier"),
            ("a FROM t", "a FROM clause"),
            ("a FROM t WHERE a > 1", "a FROM clause"),
            ("a WHERE a > 1", "a WHERE clause"),
            ("count(a) GROUP BY b", "a GROUP BY clause"),
            ("count(a) HAVING count(a) > 1", "a HAVING clause"),
            ("a QUALIFY row_number() OVER () = 1", "a QUALIFY clause"),
            ("a USING SAMPLE 10%", "a SAMPLE clause"),
            ("1 UNION ALL SELECT 2", "a set operation"),
            ("1; SELECT 2", "more than one statement"),
        ],
    )
    def test_a_clause_the_expression_list_cannot_hold_is_named(self, text: str, holds: str) -> None:
        message = re.escape(f'Parser Error: "{text}" is not an expression list: it holds {holds}')
        with pytest.raises(duckdb.ParserException, match=message):
            SQLExpression(text)

    def test_sql_expr_is_not_checked(self) -> None:
        assert str(duckdb.sql_expr("1, 2")) == "(1, 2)"

    def test_takes_text(self) -> None:
        with pytest.raises(TypeError, match="takes SQL as a string"):
            SQLExpression(1)


class TestLambdaExpression:
    def test_one_name_or_a_tuple(self, con: duckdb.DuckDBPyConnection) -> None:
        one = LambdaExpression("x", ColumnExpression("x") + 3)
        two = LambdaExpression(("x", "y"), ColumnExpression("x") + ColumnExpression("y"))
        assert str(one) == 'lambda "x": ("x" + 3)'
        assert str(two) == 'lambda "x", "y": ("x" + "y")'
        assert value(con, FunctionExpression("list_apply", [1, 2], one)) == [4, 5]
        assert value(con, FunctionExpression("list_reduce", [1, 2, 3], two)) == 6

    def test_column_expressions_as_names(self) -> None:
        assert str(LambdaExpression(ColumnExpression("x"), 1)) == 'lambda "x": 1'
        assert str(LambdaExpression((ColumnExpression("x"), "y"), 1)) == 'lambda "x", "y": 1'

    def test_a_body_value_is_a_constant(self) -> None:
        assert str(LambdaExpression("x", "s")) == "lambda \"x\": 's'"

    @pytest.mark.parametrize(
        ("lhs", "message"),
        [
            (42, "either a tuple containing strings, or a single string"),
            (ColumnExpression("t", "x"), "not of type ColumnExpression"),
            ((ConstantExpression(1),), "not of type ColumnExpression"),
            ((), "at least one parameter"),
        ],
        ids=["number", "qualified", "constant", "none"],
    )
    def test_what_is_no_parameter_is_refused(self, lhs: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            LambdaExpression(lhs, col("x"))


class TestDefaultExpression:
    def test_default_in_an_insert(self, con: duckdb.DuckDBPyConnection) -> None:
        assert str(DefaultExpression()) == "DEFAULT"
        con.execute("CREATE TABLE with_default (a INTEGER DEFAULT 7, b INTEGER)")
        con.execute(f"INSERT INTO with_default VALUES ({DefaultExpression()}, 1)")
        assert con.execute("SELECT * FROM with_default").fetchall() == [(7, 1)]


class TestCompatMethods:
    def test_null_tests(self, con: duckdb.DuckDBPyConnection) -> None:
        a = ColumnExpression("a")
        assert str(a.isnull()) == str(a.is_null()) == '("a" IS NULL)'
        assert str(a.isnotnull()) == str(a.is_not_null()) == '("a" IS NOT NULL)'
        assert value(con, a.isnull(), " FROM (SELECT NULL AS a)") is True

    @pytest.mark.parametrize(
        ("membership", "text"),
        [
            (col("a").isin(1, 2), '("a" IN (1, 2))'),
            (col("a").isin([1, 2]), '("a" IN ([1, 2]))'),
            (col("a").isin((1, 2)), '("a" IN ([1, 2]))'),
            (col("a").isin(1), '("a" IN (1))'),
            (col("a").isin("x", "y"), "(\"a\" IN ('x', 'y'))"),
            (col("a").isin(None, 1), '("a" IN (NULL, 1))'),
            (col("a").isin(col("b")), '("a" IN ("b"))'),
            (col("a").isin("x"), "(\"a\" IN ('x'))"),
            (col("a").isin(b"x"), "(\"a\" IN ('\\x78'::BLOB))"),
            (col("a").isin([]), '("a" IN ([]))'),
            (col("a").isnotin(1, 2), '(NOT ("a" IN (1, 2)))'),
            (col("a").isnotin([1, 2]), '(NOT ("a" IN ([1, 2])))'),
            (col("a").isnotin([]), '(NOT ("a" IN ([])))'),
        ],
        ids=[
            "values",
            "one list",
            "one tuple",
            "one value",
            "texts",
            "null",
            "one expression",
            "one text",
            "one blob",
            "one empty list",
            "not values",
            "not one list",
            "not one empty list",
        ],
    )
    def test_membership(self, membership: duckdb.Expression, text: str) -> None:
        assert str(membership) == text

    def test_membership_runs(self, con: duckdb.DuckDBPyConnection) -> None:
        source = " FROM (SELECT 2 AS a, [1, 2] AS xs)"
        assert value(con, col("a").isin(1, 2), source) is True
        assert value(con, col("a").isnotin(1, 2), source) is False
        assert value(con, col("a").isin(col("a")), source) is True
        assert value(con, col("xs").isin([1, 2]), source) is True
        assert value(con, col("xs").isnotin([1, 2]), source) is False

    def test_one_argument_is_one_value(self, con: duckdb.DuckDBPyConnection) -> None:
        # `isin([1, 2])` means the list itself, as `isin(lit([1, 2]))` does; the members are `isin(1, 2)`.
        assert str(col("a").isin([1, 2])) == str(col("a").isin(lit([1, 2])))
        assert str(col("a").isin([1, 2])) != str(col("a").isin(1, 2))
        assert value(con, col("a").isin(1, 2), " FROM (SELECT 2 AS a)") is True

    @pytest.mark.parametrize("method", ["isin", "isnotin"])
    def test_a_list_against_a_scalar_column_is_a_comparison(self, con: duckdb.DuckDBPyConnection, method: str) -> None:
        membership = getattr(col("a"), method)([1, 2])
        with pytest.raises(duckdb.ConversionException):
            value(con, membership, " FROM (SELECT 2 AS a)")

    @pytest.mark.parametrize("method", ["isin", "isnotin"])
    def test_membership_needs_a_value(self, method: str) -> None:
        with pytest.raises(duckdb.InvalidInputException, match=f"to '{method}', needs at least 1 parameter"):
            getattr(col("a"), method)()

    def test_collate(self, con: duckdb.DuckDBPyConnection) -> None:
        assert str(ConstantExpression("A").collate("NOCASE")) == "('A' COLLATE \"NOCASE\")"
        assert value(con, ConstantExpression("a") == ConstantExpression("A").collate("nocase")) is True
        assert value(con, ConstantExpression("á") == ConstantExpression("A").collate("nocase.noaccent")) is True
        with pytest.raises(duckdb.CatalogException, match="Collation with name non-existant does not exist"):
            value(con, ConstantExpression("a").collate("non-existant"))

    @pytest.mark.parametrize(
        ("target", "text"),
        [
            ("INTEGER", 'CAST("a" AS INTEGER)'),
            (INTEGER, 'CAST("a" AS INTEGER)'),
            (TIMESTAMP, 'CAST("a" AS TIMESTAMP)'),
            (int, 'CAST("a" AS BIGINT)'),
            (str, 'CAST("a" AS VARCHAR)'),
            (DuckDBPyType("DECIMAL(4, 1)"), 'CAST("a" AS DECIMAL(4,1))'),
        ],
        ids=["text", "integer", "timestamp", "int", "str", "decimal"],
    )
    def test_cast_takes_text_or_a_type(self, target: object, text: str) -> None:
        assert str(col("a").cast(target)) == text
        assert str(col("a").try_cast(target)) == text.replace("CAST", "TRY_CAST")

    def test_cast_refuses_what_is_no_type(self) -> None:
        with pytest.raises(TypeError, match="SQL type"):
            col("a").cast(object())

    def test_get_name(self) -> None:
        assert ColumnExpression("a").get_name() == "a"
        assert ColumnExpression("t", "a").get_name() == "a"
        assert ColumnExpression("a").alias("b").get_name() == "b"
        assert (ColumnExpression("a") + 1).get_name() == '("a" + 1)'
        assert ConstantExpression("x").get_name() == "'x'"
        assert (col("a") + 1).alias("total").get_name() == "total"

    def test_show_prints_the_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        (ColumnExpression("a") == "x").show()
        assert capsys.readouterr().out == "(\"a\" = 'x')\n"

    def test_str_and_repr_are_the_text_with_values_written_in(self) -> None:
        expression = (col("s") == "it's") & (col("d") > datetime.date(2020, 1, 2))
        text = "((\"s\" = 'it''s') AND (\"d\" > '2020-01-02'::DATE))"
        assert str(expression) == text
        assert repr(expression) == text

    def test_str_leaves_the_alias_and_order_out(self) -> None:
        assert str(col("a").alias("b").desc()) == '"a"'

    def test_reflected_operators(self) -> None:
        a = ColumnExpression("a")
        assert str(None & a) == '(NULL AND "a")'
        assert str(True | a) == '(TRUE OR "a")'
        assert str(7 // a) == '(7 // "a")'
        assert str(7 % a) == '(7 % "a")'
        assert str(2**a) == 'pow(2, "a")'


class TestTypesAreThoseOfTheValues:
    def test_typed_columns(self, con: duckdb.DuckDBPyConnection) -> None:
        typed = [
            (ConstantExpression(1), INTEGER),
            (ConstantExpression(2**40), BIGINT),
            (ConstantExpression(1.5), DOUBLE),
            (ConstantExpression("x"), VARCHAR),
            (ConstantExpression(None), SQLNULL),
        ]
        for expression, sql_type in typed:
            assert con.execute(f"SELECT typeof({expression})").fetchall()[0][0] == str(sql_type)


class TestDecisionsOnTheOldForms:
    def test_an_empty_alias_is_no_alias(self, con: duckdb.DuckDBPyConnection) -> None:
        # As the earlier API did, since the engine's parser refuses AS "".
        unnamed = ColumnExpression("a").alias("b").alias("")
        assert unnamed.as_select() == '"a"'
        assert unnamed.get_name() == "a"
        assert [d[0] for d in con.execute(f"SELECT {unnamed.as_select()} FROM (SELECT 1 AS a)").description] == ["a"]

    def test_an_aliased_argument_is_a_named_argument_where_names_are_read(self, con: duckdb.DuckDBPyConnection) -> None:
        # The functions struct_pack and struct_insert read argument names.
        packed = FunctionExpression("struct_pack", ColumnExpression("x").alias("name"), lit(2).alias("select"))
        assert str(packed) == 'struct_pack(name := "x", "select" := 2)'
        assert value(con, packed, " FROM (SELECT 1 AS x)") == {"name": 1, "select": 2}
        inserted = FunctionExpression("STRUCT_INSERT", ColumnExpression("s"), lit(3).alias("c"))
        assert str(inserted) == 'STRUCT_INSERT("s", c := 3)'
        assert value(con, inserted, " FROM (SELECT {'a': 1} AS s)") == {"a": 1, "c": 3}

    def test_an_aliased_argument_elsewhere_drops_its_alias(self, con: duckdb.DuckDBPyConnection) -> None:
        # Every other function ignored an argument's alias.
        call = FunctionExpression("epoch_us", ColumnExpression("t").alias("ts"))
        assert str(call) == 'epoch_us("t")'
        assert value(con, fn("greatest", col("a").alias("x"), 2), " FROM (SELECT 1 AS a)") == 2

    def test_a_method_call_drops_the_receivers_alias(self) -> None:
        assert str(col("x").alias("t").sum()) == 'sum("x")'
        assert str(col("s").alias("n").str().upper()) == 'upper("s")'

    @pytest.mark.parametrize("blob", [b"ab", bytearray(b"ab")], ids=["bytes", "bytearray"])
    def test_bytes_and_bytearray_are_blob_values(self, con: duckdb.DuckDBPyConnection, blob: object) -> None:
        assert str(ConstantExpression(blob)) == "'\\x61\\x62'::BLOB"
        assert str(col("b") == blob) == "(\"b\" = '\\x61\\x62'::BLOB)"
        assert value(con, ConstantExpression(blob)) == b"ab"

    def test_a_bytearray_is_a_snapshot(self) -> None:
        blob = bytearray(b"a")
        expression = ConstantExpression(blob)
        blob.extend(b"b")
        assert str(expression) == "'\\x61'::BLOB"


class TestTheExpressionConstructor:
    # Expression(x) is the column x for a string and the value x for anything else, as before.
    @pytest.mark.parametrize(
        ("argument", "text"),
        [
            ("x", '"x"'),
            ("a.b", '"a.b"'),
            ("*", "*"),
            (1, "1"),
            (None, "NULL"),
            (b"x", "'\\x78'::BLOB"),
            (1.5, "1.5::DOUBLE"),
            ([1, 2], "[1, 2]"),
        ],
        ids=["column", "dotted column", "star", "integer", "null", "blob", "double", "list"],
    )
    def test_a_name_or_a_value(self, argument: object, text: str) -> None:
        built = duckdb.Expression(argument)
        assert isinstance(built, duckdb.Expression)
        assert str(built) == text

    def test_a_duckdb_value(self, con: duckdb.DuckDBPyConnection) -> None:
        built = duckdb.Expression(Value(1, INTEGER))
        assert con.execute(f"SELECT {built}, typeof({built})").fetchall() == [(1, "INTEGER")]

    def test_it_runs(self, con: duckdb.DuckDBPyConnection) -> None:
        assert value(con, duckdb.Expression("x") + duckdb.Expression(1), " FROM (SELECT 2 AS x)") == 3

    @pytest.mark.parametrize(
        ("args", "error"),
        [((), TypeError), (("a", "b"), TypeError), ((col("a"),), TypeError), ((object(),), TypeError)],
        ids=["none", "two", "an expression", "an object"],
    )
    def test_what_it_refuses(self, args: tuple[object, ...], error: type[Exception]) -> None:
        with pytest.raises(error):
            duckdb.Expression(*args)

    def test_the_node_classes_are_untouched(self) -> None:
        assert str(expression_module.Col(("t", "a"))) == '"t"."a"'
        assert str(expression_module.Lit("x")) == "'x'"


class TestStringLiterals:
    def test_every_bare_string_in_tree_order(self) -> None:
        expression = (
            (col("a") == "x")
            & col("b").isin("y", 1, "z")
            & (CaseExpression(col("c") == "w", "v").otherwise(fn("f", "u", named="t")) == lit("x"))
            & col("d").like("p%")
        )
        # The second "x" is lit("x"), written on purpose, so it is not one.
        assert string_literals(expression) == ["x", "y", "z", "w", "v", "u", "t", "p%"]

    def test_an_explicit_constant_is_not_a_bare_string(self) -> None:
        explicit = (col("a") == ConstantExpression("x")) & (col("b") == lit("y")) & (col("c") == duckdb.Expression(1))
        assert string_literals(explicit) == []
        assert string_literals(col("a") == "x") == ["x"]

    def test_only_plain_string_literals_count(self) -> None:
        expression = (
            (col("a")["field"] == col("p"))
            & (duckdb.sql_expr("'s'") == lit(["in a list"]))
            & (ConstantExpression(Value("v", VARCHAR)) == col("b").cast("VARCHAR").alias("named"))
            & StarExpression(exclude=["c"]).is_null()
            & (LambdaExpression("x", col("x")) == b"bytes")
        )
        assert string_literals(expression) == []

    def test_a_container_of_expressions(self) -> None:
        assert string_literals([col("a") == "x", (col("b") == "y",)]) == ["x", "y"]

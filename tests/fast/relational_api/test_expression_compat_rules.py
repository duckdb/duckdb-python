"""Expressions built through the old constructor names, used in relation verbs and checked against direct SQL."""

import decimal
import math
import re

import pytest

import duckdb
from duckdb import CaseExpression, ColumnExpression, ConstantExpression, FunctionExpression


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES (1, '\\x00'::BLOB, 'p'), (2, '\\x01'::BLOB, 'q')) v(i, b, x)"
    )
    yield connection
    connection.close()


def described(con, sql, params=None):
    cursor = con.execute(sql, params)
    description = [(column[0], str(column[1])) for column in cursor.description]
    return cursor.fetchall(), description


class TestBinaryValues:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(
                b"\x00",
                id="bytes",
            ),
            pytest.param(bytearray(b"\x00"), id="bytearray"),
        ],
    )
    def test_a_binary_operand_compares_with_a_blob(self, con, value):
        rel = con.table("t").filter(ColumnExpression("b") == value)
        assert rel.fetchall() == con.execute("SELECT * FROM t WHERE b = '\\x00'::BLOB").fetchall()

    def test_a_memoryview_is_a_blob_value(self, con):
        rel = con.values([memoryview(b"abc")])
        assert [str(kind) for kind in rel.types] == ["BLOB"]
        assert rel.fetchall() == [(b"abc",)]

    def test_a_memoryview_beside_another_value(self, con):
        assert [str(kind) for kind in con.values([1, memoryview(b"abc")]).types] == ["INTEGER", "BLOB"]

    def test_a_memoryview_as_a_constant_and_as_an_operand(self, con):
        assert con.sql("SELECT 1").project(ConstantExpression(memoryview(b"abc"))).fetchall() == [(b"abc",)]
        blob = con.sql("SELECT 'abc'::BLOB AS b")
        assert blob.filter(ColumnExpression("b") == memoryview(b"abc")).fetchall() == [(b"abc",)]

    def test_a_memoryview_of_another_item_format_gives_its_raw_bytes(self, con):
        assert con.values([memoryview(bytearray(b"ab")).cast("H")]).fetchall() == [(b"ab",)]

    def test_a_strided_memoryview_gives_its_logical_bytes(self, con):
        # New behaviour: the old client read the raw buffer and raised BufferError for a view that is not contiguous.
        assert con.values([memoryview(b"abcdef")[::2]]).fetchall() == [(b"ace",)]

    def test_a_value_no_literal_can_hold_is_a_type_error_naming_its_type(self, con):
        with pytest.raises(TypeError, match="set"):
            con.values([{1, 2}])


class TestOldCallForms:
    def test_isin_with_one_expression_is_one_member(self, con):
        rel = con.table("t").filter(ColumnExpression("i").isin(ConstantExpression(2)))
        assert rel.fetchall() == con.execute("SELECT * FROM t WHERE i IN (2)").fetchall()

    def test_isin_with_a_column_is_one_member(self, con):
        rel = con.table("t").filter(ColumnExpression("i").isin(ColumnExpression("i")))
        assert rel.fetchall() == con.execute("SELECT * FROM t WHERE i IN (i)").fetchall()

    def test_a_column_named_star_is_the_star(self, con):
        rel = con.table("t").select(ColumnExpression("*"))
        assert rel.columns == ["i", "b", "x"]
        assert rel.fetchall() == con.execute("SELECT * FROM t").fetchall()

    def test_an_empty_alias_keeps_the_column_name(self, con):
        rel = con.table("t").select(ColumnExpression("i").alias(""))
        assert rel.columns == ["i"]
        assert rel.fetchall() == [(1,), (2,)]

    def test_struct_pack_keeps_the_field_name_of_an_alias(self, con):
        rel = con.table("t").select(FunctionExpression("struct_pack", ColumnExpression("x").alias("name")).alias("s"))
        rows, description = described(con, "SELECT struct_pack(name := x) AS s FROM t")
        assert [(name, str(kind)) for name, kind in zip(rel.columns, rel.types, strict=True)] == description
        assert rel.fetchall() == rows

    def test_an_alias_on_a_function_argument_is_dropped(self, con):
        rel = con.table("t").select(FunctionExpression("upper", ColumnExpression("x").alias("q")))
        rows, description = described(con, "SELECT upper(x) FROM t")
        assert [(name, str(kind)) for name, kind in zip(rel.columns, rel.types, strict=True)] == description
        assert rel.fetchall() == rows

    def test_a_column_without_a_name_is_refused(self):
        with pytest.raises(ValueError, match=r"(?i)name"):
            ColumnExpression()


class TestValueConstants:
    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            pytest.param(5, duckdb.sqltypes.TINYINT, id="tinyint"),
            pytest.param("2026-10-18", duckdb.sqltypes.DATE, id="date"),
            pytest.param(
                {"a": 1, "b": "x"},
                duckdb.struct_type({"a": duckdb.sqltypes.BIGINT, "b": duckdb.sqltypes.VARCHAR}),
                id="struct",
            ),
        ],
    )
    def test_a_value_constant_stores_what_binding_the_value_stores(self, con, value, kind):
        constant = ConstantExpression(duckdb.Value(value, kind)).alias("c")
        rel = con.sql("SELECT 1 AS z").select(constant)
        rows, description = described(con, "SELECT ? AS c", [duckdb.Value(value, kind)])
        assert [(name, str(k)) for name, k in zip(rel.columns, rel.types, strict=True)] == description
        assert rel.fetchall() == rows


def unfinished_case():
    return duckdb.when(duckdb.col("i") == 1).then(1)


# The exact binary value of the float 0.1, 55 digits long, which is the ordinary way to meet a wide Decimal.
DECIMAL_OF_A_FLOAT = decimal.Decimal(0.1)  # noqa: RUF032
UNFINISHED = r"\.otherwise\(|\.end\("


class TestAnUnfinishedCase:
    """A CASE without .otherwise() or .end() is no expression, so every use of it names the way out."""

    @pytest.mark.parametrize(
        "use",
        [
            pytest.param(lambda case: case == 1, id="equals"),
            pytest.param(lambda case: case != 1, id="not_equals"),
            pytest.param(lambda case: case < 1, id="less_than"),
            pytest.param(lambda case: case + 1, id="plus"),
            pytest.param(lambda case: 1 - case, id="reflected_minus"),
            pytest.param(bool, id="truth"),
        ],
    )
    def test_comparing_computing_or_testing_it_raises(self, use):
        with pytest.raises(TypeError, match=UNFINISHED):
            use(unfinished_case())

    def test_filter_over_a_comparison_with_it_raises_rather_than_returning_no_rows(self, con):
        # The comparison is an expression position, so the expression layer refuses it with TypeError.
        with pytest.raises(TypeError, match=UNFINISHED):
            con.table("t").filter(unfinished_case() == 1)

    @pytest.mark.parametrize(
        "verb",
        [
            pytest.param(lambda rel, case: rel.update({"i": case}), id="update_set"),
            pytest.param(lambda rel, case: rel.filter(case), id="filter"),
            pytest.param(lambda rel, case: rel.select(case), id="select"),
            pytest.param(lambda rel, case: rel.sort(case), id="sort"),
            pytest.param(lambda rel, case: rel.aggregate([case]), id="aggregate"),
            pytest.param(lambda rel, case: rel.join(rel.set_alias("o"), case), id="join_condition"),
        ],
    )
    def test_a_verb_given_it_raises_the_verb_argument_class_with_the_same_message(self, con, verb):
        # A verb refuses an argument it cannot take with InvalidInputException, as the shipping client did.
        with pytest.raises(duckdb.InvalidInputException, match=UNFINISHED):
            verb(con.table("t"), unfinished_case())

    def test_a_finished_case_compares_as_an_expression(self, con):
        finished = unfinished_case().otherwise(0)
        assert con.table("t").filter(finished == 1).project("i").fetchall() == [(1,)]


class TestDecimalsTheParameterPathHolds:
    """A Decimal the parameter path types, as DOUBLE when DECIMAL(38) cannot hold it, renders on every path."""

    def parameter_answer(self, con, value):
        return con.execute("SELECT ? AS d", [value]).fetchall()

    def test_decimal_of_a_float_in_a_filter(self, con):
        rel = con.sql("SELECT 0.05::DOUBLE AS x").filter(ColumnExpression("x") < DECIMAL_OF_A_FLOAT)
        assert rel.fetchall() == [(0.05,)]

    def test_decimal_of_a_float_in_values(self, con):
        value = DECIMAL_OF_A_FLOAT
        assert con.values([value]).fetchall() == self.parameter_answer(con, value)

    def test_decimal_of_a_float_in_a_projection(self, con):
        value = DECIMAL_OF_A_FLOAT
        rel = con.sql("SELECT 1 AS z").select(ConstantExpression(value).alias("d"))
        assert rel.fetchall() == self.parameter_answer(con, value)

    @pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity"])
    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(
                lambda con, v: con.sql("SELECT 1 AS z").select(ConstantExpression(v).alias("d")), id="projection"
            ),
            pytest.param(lambda con, v: con.values([v]), id="values"),
        ],
    )
    def test_a_value_that_is_not_finite(self, con, text, path):
        value = decimal.Decimal(text)
        [(answered,)] = path(con, value).fetchall()
        [(expected,)] = self.parameter_answer(con, value)
        assert (math.isnan(answered) and math.isnan(expected)) or answered == expected

    def test_a_value_that_is_not_finite_in_a_filter(self, con):
        rel = con.sql("SELECT 1.0::DOUBLE AS x").filter(ColumnExpression("x") < decimal.Decimal("Infinity"))
        assert rel.fetchall() == [(1.0,)]


class TestDecimalWidthAtTheEdges:
    """The width counts the leading zero of a fraction as the parameter path does, and past 38 the value is DOUBLE."""

    @pytest.mark.parametrize(
        ("text", "kind"),
        [
            pytest.param("1e-38", "DOUBLE", id="smallest_scale_38"),
            pytest.param("9" * 38, "DECIMAL(38,0)", id="largest_integral"),
            pytest.param("0." + "9" * 38, "DECIMAL(38,38)", id="largest_fraction"),
        ],
    )
    def test_a_decimal_at_the_edge_renders_at_that_width(self, con, text, kind):
        value = decimal.Decimal(text)
        rel = con.sql("SELECT 1 AS z").select(ConstantExpression(value).alias("d"))
        assert [str(t) for t in rel.types] == [kind] == [con.execute("SELECT typeof(?)", [value]).fetchall()[0][0]]
        assert rel.fetchall() == con.execute("SELECT ?", [value]).fetchall()


class TestKeywordForms:
    """The keyword names of the shipping client's constructors and methods are accepted."""

    def test_constant_takes_value(self, con):
        assert con.sql("SELECT 1 AS z").select(ConstantExpression(value=5).alias("v")).fetchall() == [(5,)]

    def test_cast_takes_type(self, con):
        rel = con.sql("SELECT 1 AS z").select(ColumnExpression("z").cast(type=duckdb.sqltypes.VARCHAR).alias("v"))
        assert rel.fetchall() == [("1",)]

    def test_try_cast_takes_type(self, con):
        rel = con.sql("SELECT 'x' AS z").select(ColumnExpression("z").try_cast(type=duckdb.sqltypes.INTEGER).alias("v"))
        assert rel.fetchall() == [(None,)]

    def test_between_takes_lower_and_upper(self, con):
        between = ColumnExpression("z").between(lower=ConstantExpression(1), upper=ConstantExpression(3))
        assert con.sql("SELECT 2 AS z").select(between.alias("v")).fetchall() == [(True,)]


class TestConditionValues:
    """A condition is any value but a string, so a Decimal is one and None, which no condition can be, is not."""

    def test_a_decimal_condition(self, con):
        case = CaseExpression(decimal.Decimal("1.5"), ConstantExpression(7)).otherwise(ConstantExpression(0))
        assert con.sql("SELECT 1 AS z").select(case.alias("v")).fetchall() == [(7,)]

    def test_none_is_no_condition(self):
        with pytest.raises(TypeError):
            CaseExpression(None, ConstantExpression(7))

    def test_where_takes_a_value_condition(self, con):
        # .where() takes any value but a string, as when() and filter() do.
        total = duckdb.col("i").sum().where(True).alias("s")
        assert "FILTER (WHERE TRUE)" in str(total)
        assert con.table("t").aggregate([total]).fetchall() == [(3,)]


class TestAMarkerInsideAValue:
    """A value holding `$1` stays a value; the text is assembled without a replacement pass."""

    @pytest.fixture
    def marked(self, con):
        return con.sql("SELECT * FROM (VALUES ('prefix$1suffix', 1), (' || version() || ', 2)) v(s, n)")

    def test_in_a_filter(self, marked):
        assert marked.filter(ColumnExpression("s") == ConstantExpression("prefix$1suffix")).fetchall() == [
            ("prefix$1suffix", 1)
        ]

    def test_in_a_function_beside_a_value_that_reads_like_sql(self, con):
        call = FunctionExpression(
            "concat", ConstantExpression(" || version() || "), ConstantExpression("prefix$1suffix")
        )
        assert con.sql("SELECT 1 AS z").select(call.alias("c")).fetchall() == [(" || version() || prefix$1suffix",)]

    def test_in_a_join_condition(self, con, marked):
        other = con.sql("SELECT * FROM (VALUES ('prefix$1suffix', 'x')) w(s, y)")
        condition = (ColumnExpression("a", "s") == ColumnExpression("b", "s")) & (
            ColumnExpression("b", "s") == ConstantExpression("prefix$1suffix")
        )
        joined = marked.set_alias("a").join(other.set_alias("b"), condition)
        assert joined.fetchall() == [("prefix$1suffix", 1, "prefix$1suffix", "x")]


class TestLiteralsOfSubclasses:
    """A literal is its base type's value; an overriding method of a subclass never becomes SQL."""

    def test_an_int_subclass(self, con):
        class Sly(int):
            def __str__(self) -> str:
                return "version()"

            __repr__ = __str__

        assert con.sql("SELECT 1").project(ConstantExpression(Sly(7))).fetchall() == [(7,)]

    def test_a_str_subclass(self, con):
        class Sly(str):
            __slots__ = ()

            def replace(self, *args):
                return "' || version() || '"

        assert con.sql("SELECT 1").project(ConstantExpression(Sly("safe"))).fetchall() == [("safe",)]

    def test_a_float_subclass(self, con):
        class Sly(float):
            def __repr__(self) -> str:
                return "42.0"

            __str__ = __repr__

        assert con.sql("SELECT 1").project(ConstantExpression(Sly(1.5))).fetchall() == [(1.5,)]


class TestLiteralsFollowTheParameterPath:
    """Where a literal could differ from the parameter path, it takes the parameter path's type and value."""

    def literal_and_bound(self, con, value):
        rel = con.sql("SELECT 1").select(ConstantExpression(value).alias("v"))
        literal = ([str(kind) for kind in rel.types], rel.fetchall())
        cursor = con.execute("SELECT ? AS v", [value])
        return literal, ([str(column[1]) for column in cursor.description], cursor.fetchall())

    def test_a_decimal_beside_an_unsigned_integer_in_a_struct_list_keeps_its_width(self, con):
        value = [{"d": decimal.Decimal("0.0001"), "u": 2**63}]
        literal, bound = self.literal_and_bound(con, value)
        assert bound[1] == [(value,)]
        assert literal == bound

    def test_a_null_in_a_list_does_not_decide_the_element_type(self, con):
        value = [1, 2**63, None]
        assert str(con.values([value]).types[0]) == con.execute("SELECT typeof(?)", [value]).fetchall()[0][0]

    def test_a_decimal_counts_the_leading_zero_of_its_fraction(self, con):
        assert [str(kind) for kind in con.values([decimal.Decimal("0.001")]).types] == ["DECIMAL(4,3)"]

    def test_an_empty_dict_is_an_empty_map(self, con):
        rel = con.values([{}])
        assert [str(kind) for kind in rel.types] == ['MAP("NULL", "NULL")']
        assert rel.fetchall() == [({},)]

    def test_a_mixed_list_takes_the_parameter_path_s_element_type(self, con):
        rel = con.values([[1, "a"]])
        assert [str(kind) for kind in rel.types] == ["VARCHAR[]"]
        assert rel.fetchall() == [(["1", "a"],)]

    def test_a_dict_with_int_keys_stays_a_map(self, con):
        assert [str(kind) for kind in con.values([{1: "a"}]).types] == ["MAP(INTEGER, VARCHAR)"]

    def test_a_typed_memoryview_value(self, con):
        value = duckdb.Value(memoryview(b"abc"), duckdb.sqltypes.BLOB)
        assert con.execute("SELECT ?", [value]).fetchall() == [(b"abc",)]
        assert con.values([value]).fetchall() == [(b"abc",)]

    def test_an_integer_past_the_string_limit_is_refused_as_a_parameter_is(self, con):
        value = 10**5000
        with pytest.raises(duckdb.InvalidInputException, match="too large for 128-bit integer type"):
            con.execute("SELECT ?", [value])
        with pytest.raises(duckdb.InvalidInputException, match="too large for 128-bit integer type"):
            con.values([value])


class TestIntegerLiteralTypes:
    """An int is typed as the parameter path types it: BIGINT, then UBIGINT, then HUGEINT, then refused."""

    def typeof(self, con, value):
        return con.sql("SELECT 1").project(FunctionExpression("typeof", ConstantExpression(value))).fetchall()

    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            pytest.param(2**63, "UBIGINT", id="past_int64"),
            pytest.param(2**64, "HUGEINT", id="past_uint64"),
            pytest.param(-(2**63) - 1, "HUGEINT", id="below_int64"),
        ],
    )
    def test_the_type_is_the_parameter_path_s(self, con, value, kind):
        assert self.typeof(con, value) == con.execute("SELECT typeof(?)", [value]).fetchall() == [(kind,)]

    def test_an_unsigned_sum_overflows_as_the_engine_says(self, con):
        with pytest.raises(duckdb.OutOfRangeException, match="Overflow"):
            con.sql("SELECT 1").project(ConstantExpression(2**63) + ConstantExpression(2**63)).fetchall()

    def test_a_value_past_int128_is_refused_as_a_parameter_is(self, con):
        with pytest.raises(duckdb.InvalidInputException):
            con.execute("SELECT ?", [2**128])
        with pytest.raises(duckdb.InvalidInputException):
            con.sql("SELECT 1").project(ConstantExpression(2**128)).fetchall()


class TestThePublicAnnotations:
    def test_mypy_accepts_the_supported_calls(self, tmp_path):
        api = pytest.importorskip("mypy.api")
        snippet = tmp_path / "calls.py"
        snippet.write_text(
            "from duckdb import CaseExpression, Expression, when\n\n"
            'Expression("x")\n'
            "CaseExpression(True, 1)\n"
            "when(True)\n"
        )
        dash = "-" * 2
        flags = [f"{dash}{flag}" for flag in ("strict", "no-incremental", "follow-imports=silent")]
        report, errors, status = api.run([str(snippet), *flags, f"{dash}cache-dir", str(tmp_path)])
        assert status == 0, report + errors


class TestRawTextInsideALambda:
    """A lambda variable written into raw SQL text names the variable the lambda finally binds."""

    def test_a_variable_in_raw_text_binds(self, con):
        # The variable is interpolated before its final name is chosen, and the text still reads it.
        incremented = duckdb.fn("list_transform", [1, 2], lambda x: duckdb.SQLExpression(f"{x} + 1"))
        assert con.sql("SELECT 1").select(incremented).fetchall() == [([2, 3],)]

    def test_a_nested_lambda_with_a_column_in_raw_text_binds(self, con):
        # The inner variable and the column k both appear in the raw text of the inner lambda.
        con.execute("CREATE TABLE nested AS SELECT [[1, 2], [3]] AS xss, 10 AS k")
        shifted = duckdb.fn(
            "list_transform",
            duckdb.col("xss"),
            lambda xs: duckdb.fn("list_transform", xs, lambda x: duckdb.SQLExpression(f"{x} + k")),
        )
        direct = "SELECT list_transform(xss, lambda xs: list_transform(xs, lambda x: x + k)) FROM nested"
        expected = con.execute(direct).fetchall()
        assert con.table("nested").select(shifted).fetchall() == expected == [([[11, 12], [13]],)]

    def test_an_aliased_lambda_variable_still_binds(self, con):
        renamed = duckdb.lit([1, 2]).list().transform(lambda x: x.alias("item"))
        assert con.sql("SELECT 1").project(renamed).fetchall() == [([1, 2],)]

    def test_an_aliased_lambda_variable_inside_a_function(self, con):
        packed = duckdb.lit([1, 2]).list().transform(lambda x: duckdb.fn("struct_pack", x.alias("item")))
        assert con.sql("SELECT 1").project(packed).fetchall() == [([{"item": 1}, {"item": 2}],)]

    def test_a_lambda_variable_is_named_after_its_parameter(self, con):
        rel = con.sql("SELECT 1").project(duckdb.lit([1, 2]).list().transform(lambda x: x + 1))
        assert re.search(r"\bx_[0-9a-f]{6}\b", rel.sql_query())

    def test_two_parameters_differing_only_in_case_are_two_variables(self, con):
        summed = duckdb.lit([10, 20]).list().transform(lambda x, X: x + X)
        assert con.sql("SELECT 1").project(summed).fetchall() == [([11, 22],)]

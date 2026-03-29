"""Exhaustive tests for template.py — the SQL template / t-string system."""

from __future__ import annotations

import pytest

from duckdb.template import (
    CompiledSql,
    IntoInterpolation,
    Param,
    ParamInterpolation,
    ResolvedSqlTemplate,
    SqlTemplate,
    SupportsDuckdbTemplate,
    compile,
    compile_parts,
    param,
    parse_parts,
    template,
)

# ── helpers ───────────────────────────────────────────────────────────────────


class FakeInterpolation:
    """Minimal object satisfying IntoInterpolation protocol."""

    def __init__(self, value, *, expression=None, conversion=None, format_spec="") -> None:
        self.value = value
        self.expression = expression
        self.conversion = conversion
        self.format_spec = format_spec


class SimpleRelation:
    """A minimal SupportsDuckdbTemplate implementation returning a string."""

    def __init__(self, sql: str) -> None:
        self._sql = sql

    def __duckdb_template__(self, **kwargs) -> str:
        return self._sql


class Cafe:
    """Test for ascii(obj) conversion."""

    def __repr__(self) -> str:
        return "Café"

    @classmethod
    def ascii(cls) -> str:
        return r"Caf\xe9"


# ═══════════════════════════════════════════════════════════════════════════════
# Param
# ═══════════════════════════════════════════════════════════════════════════════


class TestParam:
    def test_basic_creation(self):
        p = Param(value=42)
        assert p.value == 42
        assert p.name is None
        assert p.exact is False

    def test_named_param(self):
        p = Param(value="hello", name="greeting")
        assert p.name == "greeting"
        assert p.exact is False

    def test_exact_param_requires_name(self):
        with pytest.raises(ValueError, match="exact=True must have a name"):
            Param(value=1, exact=True)

    def test_exact_param_with_name(self):
        p = Param(value=1, name="x", exact=True)
        assert p.name == "x"
        assert p.exact is True

    def test_frozen(self):
        p = Param(value=1)
        with pytest.raises(AttributeError):
            p.value = 2  # ty:ignore[invalid-assignment]

    def test_param_helper_function(self):
        p = param(42, "answer", exact=True)
        assert isinstance(p, Param)
        assert p.value == 42
        assert p.name == "answer"
        assert p.exact is True

    def test_param_repr(self):
        p = Param(value=42, name="x")
        r = repr(p)
        assert "42" in r
        assert "x" in r

    def test_param_equality(self):
        """Frozen dataclasses support equality by default."""
        assert Param(value=1, name="x") == Param(value=1, name="x")
        assert Param(value=1) != Param(value=2)

    def test_param_various_value_types(self):
        """Params should accept any Python object as a value."""
        for val in [None, 3.14, True, [1, 2], {"a": 1}, b"bytes", object()]:
            p = Param(value=val)
            assert p.value is val


# ═══════════════════════════════════════════════════════════════════════════════
# ParamInterpolation
# ═══════════════════════════════════════════════════════════════════════════════


class TestParamInterpolation:
    def test_wraps_param(self):
        p = Param(value=42, name="x")
        pi = ParamInterpolation(p)
        assert pi.value is p
        assert pi.expression == "x"
        assert pi.conversion is None
        assert pi.format_spec == ""

    def test_unnamed_param_expression_is_none(self):
        p = Param(value=42)
        pi = ParamInterpolation(p)
        assert pi.expression is None

    def test_satisfies_into_interpolation_protocol(self):
        pi = ParamInterpolation(Param(value=1, name="x"))
        assert isinstance(pi, IntoInterpolation)


# ═══════════════════════════════════════════════════════════════════════════════
# parse_parts
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseParts:
    def test_all_strings(self):
        strings, others = parse_parts(["hello", " ", "world"])
        assert strings == ("hello world",)
        assert others == ()

    def test_all_others(self):
        a, b = object(), object()
        strings, others = parse_parts([a, b])
        # two others → need three string spacers: "", "", ""
        assert strings == ("", "", "")
        assert others == (a, b)

    def test_alternating(self):
        a = object()
        strings, others = parse_parts(["before", a, "after"])
        assert strings == ("before", "after")
        assert others == (a,)

    def test_string_then_other(self):
        a = object()
        strings, others = parse_parts(["sql", a])
        assert strings == ("sql", "")
        assert others == (a,)

    def test_other_then_string(self):
        a = object()
        strings, others = parse_parts([a, "sql"])
        assert strings == ("", "sql")
        assert others == (a,)

    def test_adjacent_strings_merged(self):
        strings, others = parse_parts(["a", "b", "c"])
        assert strings == ("abc",)
        assert others == ()

    def test_adjacent_others_get_empty_string_spacers(self):
        a, b, c = object(), object(), object()
        strings, others = parse_parts([a, b, c])
        assert strings == ("", "", "", "")
        assert others == (a, b, c)

    def test_invariant_strings_one_more_than_others(self):
        """The fundamental invariant: len(strings) == len(others) + 1."""
        cases = [
            ["a"],
            ["a", object()],
            [object(), "a"],
            [object(), object()],
            ["a", object(), "b", object(), "c"],
        ]
        for parts in cases:
            strings, others = parse_parts(parts)
            assert len(strings) == len(others) + 1, f"Failed for parts={parts}"

    def test_empty_input(self):
        """Empty input should return a single empty string and no others."""
        strings, others = parse_parts([])
        assert strings == ("",)
        assert others == ()

    def test_single_string(self):
        strings, others = parse_parts(["SELECT 1"])
        assert strings == ("SELECT 1",)
        assert others == ()

    def test_single_other(self):
        a = object()
        strings, others = parse_parts([a])
        assert strings == ("", "")
        assert others == (a,)


# ═══════════════════════════════════════════════════════════════════════════════
# SqlTemplate construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestSqlTemplateConstruction:
    def test_plain_string(self):
        t = SqlTemplate("SELECT 1")
        assert t.strings == ("SELECT 1",)
        assert t.interpolations == ()

    def test_multiple_strings_merged(self):
        t = SqlTemplate("SELECT ", "1")
        assert t.strings == ("SELECT 1",)
        assert t.interpolations == ()

    def test_with_interpolation(self):
        interp = FakeInterpolation(value=42, expression="x")
        t = SqlTemplate("a ", interp, " b")
        assert len(t.interpolations) == 1
        assert t.interpolations[0] is interp

    def test_bare_param_errors(self):
        p = Param(value=42)
        with pytest.raises(TypeError, match="Unexpected part type"):
            SqlTemplate("SELECT ", p)  # ty:ignore[invalid-argument-type]

    def test_wrapped_param(self):
        wrapped = ParamInterpolation(Param(value=42, name="x"))
        t = SqlTemplate("SELECT ", wrapped, " FROM t")
        assert len(t.interpolations) == 1

    def test_rejects_invalid_types(self):
        """Items that are not str, IntoInterpolation, or Param should raise TypeError."""
        with pytest.raises(TypeError, match="Unexpected part type"):
            SqlTemplate(42)  # ty:ignore[invalid-argument-type]

    def test_no_args(self):
        """Empty SqlTemplate should produce a single empty string."""
        t = SqlTemplate()
        assert t.strings == ("",)
        assert t.interpolations == ()


# ═══════════════════════════════════════════════════════════════════════════════
# SqlTemplate iteration and repr
# ═══════════════════════════════════════════════════════════════════════════════


class TestSqlTemplateIterRepr:
    def test_iter_plain_string(self):
        t = SqlTemplate("hello")
        parts = list(t)
        assert parts == ["hello"]

    def test_iter_with_interpolations(self):
        t = template("a ", param(1, "x"), " b")
        parts = list(t)
        assert len(parts) == 3
        assert parts[0] == "a "
        assert isinstance(parts[1], IntoInterpolation)
        assert parts[2] == " b"

    def test_str_raises(self):
        t = SqlTemplate("hello")
        with pytest.raises(NotImplementedError):
            str(t)

    def test_repr_plain_string(self):
        t = SqlTemplate("SELECT 1")
        r = repr(t)
        assert "SqlTemplate" in r
        assert "'SELECT 1'" in r


# ═══════════════════════════════════════════════════════════════════════════════
# template() factory — basic cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateFactory:
    def test_plain_string(self):
        t = template("SELECT 1")
        compiled = t.compile()
        expected = CompiledSql("SELECT 1", {})
        assert expected == compiled

    def test_param(self):
        p = Param(value=42, name="answer")
        t = template(p)
        compiled = t.compile()
        expected = CompiledSql("$p0_answer", {"p0_answer": 42})
        assert expected == compiled

    def test_into_interpolation(self):
        interp = FakeInterpolation(value=42, expression="x")
        t = template(interp)
        compiled = t.compile()
        expected = CompiledSql("$p0_x", {"p0_x": 42})
        assert expected == compiled

    def test_supports_duckdb_template_string(self):
        rel = SimpleRelation("SELECT 1")
        t = template(rel)
        compiled = t.compile()
        expected = CompiledSql("SELECT 1", {})
        assert expected == compiled

    def test_supports_duckdb_template_returns_iterable(self):
        class MultiPart:
            def __duckdb_template__(self, **kwargs) -> list[str | Param]:
                return ["SELECT * FROM ", param(42, "x")]

        t = template(MultiPart())
        compiled = t.compile()
        expected = CompiledSql("SELECT * FROM $p0_x", {"p0_x": 42})
        assert expected == compiled

    def test_supports_duckdb_template_returns_interpolation(self):
        class InterpReturner:
            def __duckdb_template__(self, **kwargs) -> FakeInterpolation:
                return FakeInterpolation(value="hello", expression="val")

        t = template(InterpReturner())
        compiled = t.compile()
        expected = CompiledSql("$p0_val", {"p0_val": "hello"})
        assert expected == compiled

    def test_iterable_of_strings(self):
        t = template("SELECT ", "1")
        compiled = t.compile()
        expected = CompiledSql("SELECT 1", {})
        assert expected == compiled

    def test_iterable_with_params(self):
        t = template("SELECT * FROM t WHERE id = ", Param(value=5, name="id"))
        compiled = t.compile()
        expected = CompiledSql("SELECT * FROM t WHERE id = $p0_id", {"p0_id": 5})
        assert expected == compiled

    def test_iterable_with_bare_values(self):
        """Bare values in an iterable should be treated as params."""
        t = template("SELECT * FROM t WHERE id = ", 42)
        compiled = t.compile()
        expected = CompiledSql("SELECT * FROM t WHERE id = $p0", {"p0": 42})
        assert expected == compiled

    def test_bare_value_at_top_level_becomes_param(self):
        """A bare value passed directly to template() becomes a param."""
        t = template(42)  # type: ignore[arg-type]
        compiled = t.compile()
        expected = CompiledSql("$p0", {"p0": 42})
        assert expected == compiled

    def test_bytes_not_treated_as_iterable(self):
        """Bytes should not be iterated — _is_iterable excludes bytes."""
        t = template(b"hello")  # type: ignore[arg-type]
        compiled = t.compile()
        expected = CompiledSql("$p0", params={"p0": b"hello"})
        assert compiled == expected


# ═══════════════════════════════════════════════════════════════════════════════
# template() factory — t-string integration (Python 3.14+)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateWithInterpolations:
    """Tests for template() with interpolations, using FakeInterpolation to simulate t-string behavior."""

    def test_simple_param(self):
        interp = FakeInterpolation(value=123, expression="user_id")
        compiled = compile("SELECT * FROM users WHERE id = ", interp, "")
        expected = CompiledSql("SELECT * FROM users WHERE id = $p0_user_id", {"p0_user_id": 123})
        assert expected == compiled

    def test_multiple_params(self):
        name_interp = FakeInterpolation(value="Alice", expression="name")
        age_interp = FakeInterpolation(value=30, expression="age")
        t = template("SELECT * FROM users WHERE name = ", name_interp, " AND age = ", age_interp, "")
        compiled = t.compile()
        expected = CompiledSql(
            "SELECT * FROM users WHERE name = $p0_name AND age = $p1_age",
            {"p0_name": "Alice", "p1_age": 30},
        )
        assert expected == compiled

    def test_no_params(self):
        compiled = compile("SELECT 1")
        expected = CompiledSql("SELECT 1")
        assert expected == compiled

    def test_string_conversion_s(self):
        """!s should inline the value as raw SQL (no param)."""
        interp = FakeInterpolation(value="users", expression="table", conversion="s")
        compiled = compile("SELECT * FROM ", interp, "")
        expected = CompiledSql("SELECT * FROM users")
        assert expected == compiled

    def test_repr_conversion_r(self):
        """!r should inline repr(value) as raw SQL."""
        interp = FakeInterpolation(value="hello", expression="val", conversion="r")
        compiled = compile("SELECT ", interp, "")
        # repr of "hello" is "'hello'"
        expected = CompiledSql("SELECT 'hello'")
        assert expected == compiled

    def test_ascii_conversion_a(self):
        interp = FakeInterpolation(value=Cafe(), expression="val", conversion="a")
        compiled = compile("SELECT ", interp)
        expected = CompiledSql("SELECT " + Cafe.ascii())
        assert expected == compiled

    def test_string_value_becomes_param_not_raw_sql(self):
        """A string interpolation WITHOUT conversion should be a param, not raw SQL."""
        interp = FakeInterpolation(value="Alice", expression="name")
        compiled = compile("SELECT * FROM users WHERE name = ", interp, "")
        expected = CompiledSql("SELECT * FROM users WHERE name = $p0_name", {"p0_name": "Alice"})
        assert expected == compiled

    def test_nested_template_via_supports_duckdb_template(self):
        """An interpolated SupportsDuckdbTemplate should be expanded inline."""
        inner = SimpleRelation("SELECT * FROM people")
        interp = FakeInterpolation(value=inner, expression="inner")
        compiled = compile("SELECT name FROM (", interp, ")")
        expected = CompiledSql("SELECT name FROM (SELECT * FROM people)", {})
        assert expected == compiled

    def test_nested_chaining(self):
        """Chaining templates: inner template params should propagate."""
        inner = template("SELECT * FROM people WHERE age >= ", 18)
        interp = FakeInterpolation(value=inner, expression="inner")
        t = template("SELECT name FROM (", interp, ")")
        compiled = t.compile()
        # Skipping CompiledSql equality: exact param name depends on counter offsets across nesting levels
        assert "SELECT * FROM people WHERE age >= $" in compiled.sql
        assert 18 in compiled.params.values()

    def test_param_name_derived_from_expression(self):
        """Param names should be based on the expression in the interpolation."""
        interp = FakeInterpolation(value=99, expression="my_value")
        t = template("SELECT ", interp, "")
        compiled = t.compile()
        expected = CompiledSql("SELECT $p0_my_value", {"p0_my_value": 99})
        assert expected == compiled

    def test_explicit_param_in_interpolation(self):
        """An explicit Param() used in an interpolation should be treated as a param."""
        p = param(42, "answer")
        interp = FakeInterpolation(value=p, expression="p")
        t = template("SELECT ", interp, "")
        compiled = t.compile()
        # Skipping exact equality: param name may come from Param.name or interpolation expression
        assert 42 in compiled.params.values()

    def test_format_spec_on_conversion(self):
        """Format spec combined with conversion: conversion first, then format."""
        interp = FakeInterpolation(value=3.14159, expression="val", conversion="s", format_spec=".5")
        t = template("SELECT ", interp, "")
        compiled = t.compile()
        # Python semantics: str(3.14159) = "3.14159", then format("3.14159", ".5") = "3.141"
        expected = CompiledSql("SELECT 3.141")
        assert expected == compiled

    def test_format_spec_without_conversion_is_ignored(self):
        """Format spec without conversion should ideally apply, but currently it's silently dropped."""
        interp = FakeInterpolation(value=3.14159, expression="val", format_spec=".2f")
        t = template("SELECT ", interp, "")
        compiled = t.compile()
        expected = CompiledSql("SELECT $p0_val", {"p0_val": 3.14159})
        assert expected == compiled


# ═══════════════════════════════════════════════════════════════════════════════
# resolve / _resolve_interpolation
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolve:
    def test_plain_string_resolves_to_itself(self):
        t = SqlTemplate("SELECT 1")
        resolved = t.resolve()
        assert list(resolved) == ["SELECT 1"]

    def test_conversion_s_resolves_to_string(self):
        interp = FakeInterpolation(value=42, expression="x", conversion="s")
        t = SqlTemplate("SELECT ", interp)
        resolved = t.compile()
        expected = CompiledSql("SELECT 42")
        assert resolved == expected

    def test_conversion_r_resolves_to_repr(self):
        interp = FakeInterpolation(value="hello", expression="x", conversion="r")
        t = SqlTemplate("SELECT ", interp)
        resolved = t.resolve()
        expected = CompiledSql("SELECT 'hello'")
        assert resolved == expected

    def test_conversion_a_resolves_to_ascii(self):
        interp = FakeInterpolation(value=Cafe(), expression="x", conversion="a")
        actual = compile("SELECT ", interp)
        expected = CompiledSql("SELECT " + Cafe.ascii())
        assert actual == expected

    def test_string_value_without_conversion_becomes_param(self):
        """A string value in an interpolation (no conversion) must become a param, not raw SQL."""
        interp = FakeInterpolation(value="DROP TABLE users", expression="val")
        t = SqlTemplate("SELECT ", interp)
        resolved = t.resolve()
        expected = CompiledSql("SELECT $p0_val", params={"p0_val": "DROP TABLE users"})
        assert resolved == expected

    def test_nested_supports_duckdb_template(self):
        rel = SimpleRelation("SELECT 1")
        interp = FakeInterpolation(value=rel, expression="rel")
        t = SqlTemplate("SELECT * FROM (", interp, ")")
        resolved = t.resolve()
        expected = CompiledSql("SELECT * FROM (SELECT 1)")
        assert resolved == expected

    def test_expression_name_preserved_for_simple_param(self):
        """When interpolation resolves to a single param, the expression name should be kept."""
        interp = FakeInterpolation(value=42, expression="my_age")
        t = SqlTemplate("age = ", interp)
        resolved = t.resolve()
        expected = CompiledSql("age = $p0_my_age", params={"p0_my_age": 42})
        assert resolved == expected


# ═══════════════════════════════════════════════════════════════════════════════
# ResolvedSqlTemplate
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolvedSqlTemplate:
    def test_basic(self):
        r = ResolvedSqlTemplate(["SELECT ", Param(value=42, name="x")])
        parts = list(r)
        assert len(parts) == 2

    def test_compile(self):
        r = ResolvedSqlTemplate(["SELECT ", Param(value=42, name="x")])
        compiled = r.compile()
        assert isinstance(compiled, CompiledSql)
        assert 42 in compiled.params.values()

    def test_str_raises(self):
        r = ResolvedSqlTemplate(["SELECT 1"])
        with pytest.raises(NotImplementedError):
            str(r)

    def test_repr(self):
        r = ResolvedSqlTemplate(["SELECT ", Param(value=42, name="x")])
        rep = repr(r)
        assert "ResolvedSqlTemplate" in rep
        assert "x=42" in rep

    def test_iter(self):
        parts_in = ["a", Param(value=1, name="x"), "b"]
        r = ResolvedSqlTemplate(parts_in)
        assert list(r) == parts_in


# ═══════════════════════════════════════════════════════════════════════════════
# compile_parts
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompileParts:
    def test_all_strings(self):
        result = compile_parts(["SELECT 1"])
        assert result == CompiledSql("SELECT 1")

    def test_single_unnamed_param(self):
        result = compile_parts(["SELECT ", Param(value=42)])
        expected = CompiledSql("SELECT $p0", params={"p0": 42})
        assert result == expected

    def test_single_named_param(self):
        result = compile_parts(["SELECT ", Param(value=42, name="x")])
        expected = CompiledSql("SELECT $p0_x", params={"p0_x": 42})
        assert result == expected

    def test_exact_param_uses_literal_name(self):
        result = compile_parts(["SELECT ", Param(value=42, name="my_param", exact=True)])
        expected = CompiledSql("SELECT $my_param", params={"my_param": 42})
        assert result == expected

    def test_multiple_params_numbered_sequentially(self):
        result = compile_parts(["SELECT * WHERE a = ", Param(value=1, name="a"), " AND b = ", Param(value=2, name="b")])
        expected = CompiledSql("SELECT * WHERE a = $p0_a AND b = $p1_b", params={"p0_a": 1, "p1_b": 2})
        assert result == expected

    def test_duplicate_param_names_raises(self):
        with pytest.raises(ValueError, match="Duplicate parameter names"):
            compile_parts(
                [
                    Param(value=1, name="x", exact=True),
                    Param(value=2, name="x", exact=True),
                ]
            )

    def test_unnamed_params_get_sequential_names(self):
        result = compile_parts(
            [
                "a = ",
                Param(value=1),
                " AND b = ",
                Param(value=2),
            ]
        )
        expected = CompiledSql("a = $p0 AND b = $p1", params={"p0": 1, "p1": 2})
        assert result == expected

    def test_exact_param_causes_counter_gap(self):
        """Known issue: exact params still increment the counter, causing non-sequential auto names."""
        result = compile_parts(
            [
                "a = ",
                Param(value=1, name="x"),  # → p0_x
                " AND b = ",
                Param(value=2, name="b", exact=True),  # → b (exact), but counter increments
                " AND c = ",
                Param(value=3, name="y"),  # → p2_y (not p1_y!)
            ]
        )
        expected = CompiledSql("a = $p0_x AND b = $b AND c = $p2_y", params={"p0_x": 1, "b": 2, "p2_y": 3})
        assert result == expected

    def test_empty_parts(self):
        result = compile_parts([])
        expected = CompiledSql("")
        assert result == expected

    def test_adjacent_strings(self):
        result = compile_parts(["SELECT ", "1"])
        expected = CompiledSql("SELECT 1")
        assert result == expected

    def test_param_with_none_value(self):
        result = compile_parts(["SELECT ", Param(value=None, name="x")])
        expected = CompiledSql("SELECT $p0_x", params={"p0_x": None})
        assert result == expected


# ═══════════════════════════════════════════════════════════════════════════════
# SupportsDuckdbTemplate protocol
# ═══════════════════════════════════════════════════════════════════════════════


class TestSupportsDuckdbTemplate:
    def test_protocol_check(self):
        rel = SimpleRelation("SELECT 1")
        assert isinstance(rel, SupportsDuckdbTemplate)

    def test_non_implementor(self):
        assert not isinstance("hello", SupportsDuckdbTemplate)
        assert not isinstance(42, SupportsDuckdbTemplate)

    def test_template_calls_dunder(self):
        class Tracking:
            def __init__(self) -> None:
                self.called = False

            def __duckdb_template__(self, **kwargs) -> str:
                self.called = True
                return "SELECT 1"

        obj = Tracking()
        template(obj)
        assert obj.called

    def test_future_kwargs_accepted(self):
        """Implementations should accept **kwargs for future extensibility."""

        class Strict:
            def __duckdb_template__(self, **kwargs) -> str:
                assert isinstance(kwargs, dict)
                return "SELECT 1"

        template(Strict())

    def test_returns_interpolation(self):
        """__duckdb_template__ can return an interpolation."""

        class InterpolationReturner:
            def __duckdb_template__(self, **kwargs) -> FakeInterpolation:
                return FakeInterpolation(value=42, expression="val")

        t = template(InterpolationReturner())
        compiled = t.compile()
        expected = CompiledSql("$p0_val", params={"p0_val": 42})
        assert compiled == expected

    def test_supports_duckdb_template_priority_over_iterable(self):
        class IterableWithTemplate:
            def __duckdb_template__(self, **kwargs) -> str:
                return "SELECT 1"

            def __iter__(self) -> list[str]:
                return ["SELECT 2"]

        result = template(IterableWithTemplate()).compile()
        expected = CompiledSql("SELECT 1")
        assert result == expected


# ═══════════════════════════════════════════════════════════════════════════════
# IntoInterpolation protocol
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntoInterpolation:
    def test_protocol_check_positive(self):
        interp = FakeInterpolation(value=1, expression="x")
        assert isinstance(interp, IntoInterpolation)

    def test_protocol_check_negative(self):
        assert not isinstance("hello", IntoInterpolation)
        assert not isinstance(42, IntoInterpolation)

    def test_param_interpolation_satisfies(self):
        pi = ParamInterpolation(Param(value=1))
        assert isinstance(pi, IntoInterpolation)


# ═══════════════════════════════════════════════════════════════════════════════
# CompiledSql
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompiledSql:
    def test_basic(self):
        c = CompiledSql("SELECT $p0", params={"p0": 42})
        assert c.sql == "SELECT $p0"
        assert c.params == {"p0": 42}

    def test_positional_params(self):
        c = CompiledSql("SELECT $p0", {"p0": 42})
        assert c.sql == "SELECT $p0"
        assert c.params == {"p0": 42}

    def test_empty_params(self):
        c = CompiledSql("SELECT 1")
        assert c.sql == "SELECT 1"
        assert c.params == {}

    def test_optional_params(self):
        c = CompiledSql("SELECT 1")
        assert c.sql == "SELECT 1"
        assert c.params == {}

    def test_frozen(self):
        c = CompiledSql("SELECT 1")
        with pytest.raises(AttributeError):
            c.sql = "SELECT 2"  # ty:ignore[invalid-assignment]

    def test_equality(self):
        a = CompiledSql("SELECT 1")
        b = CompiledSql("SELECT 1")
        assert a == b

    def test_repr(self):
        c = CompiledSql("SELECT 1")
        expected = "CompiledSql(sql='SELECT 1', params={})"
        assert repr(c) == expected

    def test_str(self):
        c = CompiledSql("SELECT 1")
        with pytest.raises(NotImplementedError):
            str(c)


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end: compile
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEndCompile:
    def test_plain_sql(self):
        result = compile("SELECT * FROM users")
        assert result == CompiledSql("SELECT * FROM users", {})

    def test_strings_joined(self):
        result = compile("SELECT * FROM ", "users")
        assert result == CompiledSql("SELECT * FROM users", {})

    def test_param_in_list(self):
        result = compile("SELECT * FROM users WHERE id = ", Param(value=5, name="id"))
        expected = CompiledSql("SELECT * FROM users WHERE id = $p0_id", {"p0_id": 5})
        assert expected == result

    def test_multiple_params_in_list(self):
        result = compile(
            "SELECT * FROM users WHERE name = ",
            Param(value="Alice", name="name"),
            " AND age > ",
            Param(value=18, name="age"),
        )
        expected = CompiledSql(
            "SELECT * FROM users WHERE name = $p0_name AND age > $p1_age",
            {"p0_name": "Alice", "p1_age": 18},
        )
        assert expected == result

    def test_supports_duckdb_template_end_to_end(self):
        class MyTable:
            def __duckdb_template__(self, **kwargs) -> str:
                return "SELECT * FROM my_table"

        result = compile(MyTable())
        expected = CompiledSql("SELECT * FROM my_table", {})
        assert expected == result

    def test_unparameterized_strings(self):
        result = compile("SELECT * FROM users WHERE id = ", 42, " AND name = ", "Alice")
        expected = CompiledSql("SELECT * FROM users WHERE id = $p0 AND name = Alice", {"p0": 42})
        assert expected == result

    def test_parameterized_strings(self):
        result = compile("SELECT * FROM users WHERE id = ", 42, " AND name = ", param("Alice"))
        expected = CompiledSql("SELECT * FROM users WHERE id = $p0 AND name = $p1", {"p0": 42, "p1": "Alice"})
        assert expected == result

    def test_nested_template_relations(self):
        """The key use case: chaining template queries."""
        inner = template("SELECT * FROM people WHERE age >= ", 18)
        outer = template("SELECT name FROM (", inner, ")")
        result = outer.compile()
        # Skipping exact equality: param name depends on counter offsets across nesting levels
        assert "SELECT * FROM people WHERE age >=" in result.sql
        assert "SELECT name FROM (" in result.sql
        assert 18 in result.params.values()
        assert len(result.params) == 1

    def test_deeply_nested_tstrings(self):
        """Three levels of nesting."""
        val = 100
        level1 = template("SELECT * FROM t WHERE x = ", val)
        level2 = template("SELECT * FROM (", level1, ") WHERE y = 1")
        level3 = template("SELECT count(*) FROM (", level2, ")")
        result = level3.compile()
        # Skipping exact equality: param name depends on counter offsets across nesting levels
        assert "SELECT * FROM t WHERE x = $" in result.sql
        assert 100 in result.params.values()
        assert len(result.params) == 1

    def test_multiple_nested_with_separate_params(self):
        """Two nested templates each with their own params."""
        a = 1
        b = 2
        t1 = template("SELECT * FROM t1 WHERE a = ", a)
        t2 = template("SELECT * FROM t2 WHERE b = ", b)
        outer = template("SELECT * FROM (", t1, ") JOIN (", t2, ")")
        result = outer.compile()
        # Skipping exact equality: param names depend on counter offsets across nesting levels
        assert 1 in result.params.values()
        assert 2 in result.params.values()
        assert len(result.params) == 2

    def test_sql_injection_prevented_by_default(self):
        """Without !s, even SQL-like strings become params, not raw SQL."""
        evil = "1; DROP TABLE users; --"
        interp = FakeInterpolation(value=evil, expression="evil")
        result = compile("SELECT * FROM users WHERE id = ", interp, "")
        expected = CompiledSql("SELECT * FROM users WHERE id = $p0_evil", {"p0_evil": evil})
        assert expected == result

    def test_sql_injection_possible_with_s_conversion(self):
        """With !s, values ARE inlined — this is intentional but dangerous."""
        evil = "1; DROP TABLE users; --"
        interp = FakeInterpolation(value=evil, expression="evil", conversion="s")
        result = compile("SELECT * FROM users WHERE id = ", interp, "")
        expected = CompiledSql("SELECT * FROM users WHERE id = 1; DROP TABLE users; --", {})
        assert expected == result

    def test_exact_param_name(self):
        result = compile("SELECT * FROM t WHERE id = ", param(value=42, name="my_id", exact=True))
        expected = CompiledSql("SELECT * FROM t WHERE id = $my_id", {"my_id": 42})
        assert expected == result


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases and known issues
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_string_template(self):
        result = compile("")
        expected = CompiledSql("")
        assert result == expected

    def test_param_with_none_value(self):
        result = compile(param(value=None, name="x"))
        expected = CompiledSql("$p0_x", params={"p0_x": None})
        assert result == expected

    def test_param_with_list_value(self):
        result = compile(param(value=[1, 2, 3], name="ids"))
        expected = CompiledSql("$p0_ids", params={"p0_ids": [1, 2, 3]})
        assert result == expected

    def test_param_with_dict_value(self):
        d = {"key": "value"}
        result = compile(param(value=d, name="data"))
        expected = CompiledSql("$p0_data", params={"p0_data": d})
        assert result == expected

    def test_bool_param(self):
        result = compile("SELECT * FROM t WHERE active = ", True)
        expected = CompiledSql("SELECT * FROM t WHERE active = $p0", params={"p0": True})
        assert result == expected

    def test_float_param(self):
        interp = FakeInterpolation(value=3.14, expression="threshold")
        result = template("SELECT * FROM t WHERE score > ", interp, "").compile()
        expected = CompiledSql("SELECT * FROM t WHERE score > $p0_threshold", params={"p0_threshold": 3.14})
        assert result == expected

    def test_none_param(self):
        interp = FakeInterpolation(value=None, expression="val")
        result = template("SELECT * FROM t WHERE x IS ", interp, "").compile()
        expected = CompiledSql("SELECT * FROM t WHERE x IS $p0_val", params={"p0_val": None})
        assert result == expected

    def test_param_object_in_interpolation_preserves_name(self):
        """An explicit Param used in an interpolation should keep its name."""
        p = Param(value=42, name="custom_name")
        interp = FakeInterpolation(value=p, expression="p")
        result = template("SELECT ", interp, "").compile()
        expected = CompiledSql("SELECT $custom_name", params={"custom_name": 42})
        assert result == expected

    def test_same_expression_used_twice(self):
        """Using the same expression twice should create two separate params."""
        interp1 = FakeInterpolation(value=42, expression="x")
        interp2 = FakeInterpolation(value=42, expression="x")
        result = template("SELECT * FROM t WHERE a = ", interp1, " AND b = ", interp2, "").compile()
        expected = CompiledSql("SELECT * FROM t WHERE a = $p0_x AND b = $p1_x", params={"p0_x": 42, "p1_x": 42})
        assert result == expected

    def test_mixed_conversion_and_param(self):
        """Mix of !s (inlined) and regular (parameterized) in same template."""
        table_interp = FakeInterpolation(value="users", expression="table", conversion="s")
        id_interp = FakeInterpolation(value=5, expression="user_id")
        result = template("SELECT * FROM ", table_interp, " WHERE id = ", id_interp, "").compile()
        expected = CompiledSql("SELECT * FROM users WHERE id = $p0_user_id", params={"p0_user_id": 5})
        assert result == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Conversion semantics — documenting current (potentially incorrect) behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestConversionSemantics:
    """Verify conversion + format_spec follows Python f-string semantics."""

    def test_s_conversion_on_int(self):
        interp = FakeInterpolation(value=42, expression="x", conversion="s")
        actual = compile(interp)
        expected = CompiledSql("42")
        assert actual == expected

    def test_r_conversion_on_string(self):
        """repr('hello') = "'hello'"."""
        interp = FakeInterpolation(value="hello", expression="x", conversion="r")
        actual = compile(interp)
        expected = CompiledSql("'hello'")
        assert actual == expected

    def test_r_conversion_on_int(self):
        """repr(42) = '42', no quotes."""
        interp = FakeInterpolation(value=42, expression="x", conversion="r")
        actual = compile(interp)
        expected = CompiledSql("42")
        assert actual == expected

    def test_s_conversion_with_format_spec(self):
        """Conversion first, then format_spec: str(3.14159) then format with '.5' truncates."""
        interp = FakeInterpolation(value=3.14159, expression="x", conversion="s", format_spec=".5")
        actual = compile(interp)
        expected = CompiledSql("3.141")
        assert actual == expected

    def test_r_conversion_with_format_spec(self):
        """Python semantics: repr first, then format."""
        interp = FakeInterpolation(value="hi", expression="x", conversion="r", format_spec=".4")
        actual = compile(interp)
        expected = CompiledSql("'hi'")
        assert actual == expected

    def test_format_spec_ignored_for_parameterized_values(self):
        """When no conversion is specified, format_spec is silently ignored — the value is parameterized as-is."""
        interp = FakeInterpolation(value=3.14159, expression="x", format_spec=".2f")
        actual = compile(interp)
        expected = CompiledSql("$p0_x", params={"p0_x": 3.14159})
        assert actual == expected

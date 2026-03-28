from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


class Param(TypedDict):
    """Represents a parameter to be passed to duckdb, with a name and a value."""

    value: object
    name: str


class CompiledSql(TypedDict):
    """Represents a compiled SQL statement, with the final SQL string and a list of Params to be passed to duckdb."""

    sql: str
    params: list[Param]


@runtime_checkable
class SupportsDuckdbTemplate(Protocol):
    def __duckdb_template__(
        self, /, **future_kwargs
    ) -> str | IntoInterpolation | Iterable[str | IntoInterpolation]: ...


def resolve_to_template(thing: object, /, **ignored_kwargs) -> SqlTemplate:
    if isinstance(thing, SupportsDuckdbTemplate):
        raw = thing.__duckdb_template__(**ignored_kwargs)
        return SqlTemplate.from_part_or_parts(raw)
    if isinstance(thing, IntoTemplate):
        return resolve_into_template(thing)
    if isinstance(thing, IntoInterpolation):
        return resolve_interpolation(thing)
    return SqlTemplate(OurInterpolation(thing))


def compile(thing: object) -> CompiledSql:
    resolved_template = resolve_to_template(thing)
    return compile_sql_template(resolved_template)


def resolve_into_template(template: IntoTemplate) -> SqlTemplate:
    """Resolve a Template, recursively resolving interpolations and flattening nested templates."""
    parts: list[str | IntoInterpolation] = []
    for part in template:
        if isinstance(part, str):
            parts.append(part)
        else:
            inner_parts = resolve_interpolation(part)
            parts.extend(inner_parts)
    return SqlTemplate(*parts)


def resolve_interpolation(interp: IntoInterpolation) -> SqlTemplate:
    # if conversion specified (!s, !r, !a), treat as raw sql, eg
    # t"SELECT {"mycol"!s} FROM foo" should be "SELECT mycol FROM foo
    if interp.conversion == "s":
        return SqlTemplate(str(interp.value))
    elif interp.conversion == "r":
        return SqlTemplate(repr(interp.value))
    elif interp.conversion == "a":
        return SqlTemplate(ascii(interp.value))

    templ = resolve_to_template(interp.value)
    # If the resolved inner is just a single Interpolation, then just return
    # the original value so that we preserve the expression name.
    # For example, if we have
    # ```python
    # age = 37
    # people = t"SELECT * FROM people WHERE age = {age}"
    # resolve_template(people)
    # ```
    # should resolve to:
    # "SELECT * FROM people WHERE age = $age", with a param $age=37,
    # eg with a friendly param name, rather than
    # "SELECT * FROM people WHERE age = $p0", with a param $p0=37
    if len(templ.strings) == 2 and templ.strings[0] == "" and templ.strings[1] == "" and len(templ.interpolations) == 1:
        return SqlTemplate(interp)
    else:
        # We got something nested, eg
        # age = 37
        # people = t"SELECT * FROM people WHERE age = {age}"
        # names = t"SELECT name FROM ({foo})"
        # names should resolve to:
        # "SELECT name FROM (SELECT * FROM people WHERE age = $age)", with a param $age=37
        return templ


class DuckdDbPyRelation:
    def __duckdb_template__(self, /, **future_kwargs) -> str:
        # this would just return the existing SQL.
        return "SELECT * FROM some_table"


class OurInterpolation:
    def __init__(
        self,
        value: object,
        conversion: Literal["s", "r", "a"] | None = None,
        expression: str | None = None,
    ):
        self.value = value
        self.conversion = conversion
        self.expression = expression


class SqlTemplate:
    def __init__(self, *parts: str | IntoInterpolation):
        strings, interpolations = [], []
        last_thing: Literal["string", "interpolation"] | None = None
        for part in parts:
            if isinstance(part, str):
                if last_thing == "string":
                    # Merge adjacent string parts for efficiency, since the template engine allows that
                    strings[-1] += part
                strings.append(part)
                last_thing = "string"
            else:
                if last_thing is None or last_thing == "interpolation":
                    # this is the first part,
                    # or there were two adjacent interpolations,
                    # so we need an empty string spacer
                    strings.append("")
                interpolations.append(OurInterpolation(part.value, part.conversion))
                last_thing = "interpolation"
        if last_thing == "interpolation":
            # If the last part was an interpolation, we need to end with an empty string
            strings.append("")
        assert len(strings) == len(interpolations) + 1
        self.strings = strings
        self.interpolations = interpolations

    def __iter__(self):
        for s, i in zip(self.strings, self.interpolations):
            yield s
            yield i
        yield self.strings[-1]

    @classmethod
    def from_part_or_parts(
        cls, part_or_parts: str | IntoInterpolation | Iterable[str | IntoInterpolation]
    ) -> SqlTemplate:
        if isinstance(part_or_parts, (str, IntoInterpolation)):
            return cls(part_or_parts)
        else:
            return cls(*part_or_parts)

    def __str__(self):
        msg = f"{self.__class__.__name__} cannot be directly converted to string. It needs to be processed by the SQL engine to produce the final SQL string."  # noqa: E501
        raise NotImplementedError(msg)


def compile_sql_template(template: SqlTemplate) -> CompiledSql:
    """Compile a resolved SqlTemplate into a final SQL string with named parameter placeholders, and a list of Params."""
    sql_parts: list[str] = []
    params: list[Param] = []
    for part in template:
        if isinstance(part, str):
            sql_parts.append(part)
        else:
            param_name = f"p{len(params)}"
            if part.expression is not None:
                param_name += f"_{part.expression}"
            assert_param_name_legal(param_name)
            sql_parts.append(f"${param_name}")
            params.append({"name": param_name, "value": part.value})
    return {
        "sql": "".join(sql_parts),
        "params": params,
    }


# from string.templatelib import Interpolation, Template
@runtime_checkable
class IntoInterpolation(Protocol):
    """Something that can be converted into a string.templatelib.Interpolation."""

    value: object
    conversion: Literal["s", "r", "a"] | None
    expression: str | None


@runtime_checkable
class IntoTemplate(Protocol):
    strings: Sequence[str]
    interpolations: Sequence[IntoInterpolation]

    def __iter__(self) -> Iterator[str | IntoInterpolation]: ...


def assert_param_name_legal(name: str) -> None:
    """Eg `$param_1` is legal, but `$1param`, `$param-1`, `$param 1`, and `$p ; DROP TABLE users` are not."""
    # not implemented yet
    # Not exactly sure what part of the stack this should get called in,
    # or perhaps we shouldn't even check, just pass it to duckdb and let it error if it's illegal

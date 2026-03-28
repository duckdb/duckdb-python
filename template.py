from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from typing_extensions import TypeIs

__all__ = [
    "CompiledSql",
    "IntoInterpolation",
    "IntoTemplate",
    "Param",
    "SupportsDuckdbTemplate",
    "compile",
    "template",
]


class CompiledSql(TypedDict):
    """Represents a compiled SQL statement, with the final SQL string and a list of Params to be passed to duckdb."""

    sql: str
    params: list[Param]


class Param(TypedDict):
    """Represents a parameter to be passed to duckdb, with a name and a value."""

    value: object
    name: str


# class IntoParam(TypedDict):
#     """A Param with a name that is None, which can be used as input to the template engine, which will assign it a name based on its position and optionally an expression."""

#     value: object
#     name: NotRequired[str | None]


# def is_into_param(thing: object) -> TypeIs[IntoParam]:
#     try:
#         value = thing["value"]  # ty:ignore[not-subscriptable]
#     except (TypeError, KeyError):
#         return False
#     try:
#         name = thing["name"]  # ty:ignore[not-subscriptable]
#     except KeyError:
#         name = None

#     return isinstance(value, object) and isinstance(name, (str, type(None)))


def is_into_interpolation(thing: object) -> TypeIs[IntoInterpolation]:
    return isinstance(thing, IntoInterpolation)


@runtime_checkable
class SupportsDuckdbTemplate(Protocol):
    """Something that can be converted into a Template by implementing the __duckdb_template__ method."""

    def __duckdb_template__(self, /, **future_kwargs) -> str | IntoInterpolation | Iterable[str | IntoInterpolation]:
        """Convert self into an IntoTemplate, by returning either a string, an IntoInterpolation, or an iterable of these.

        The future_kwargs are for future extensibility, in case duckdb wants
        to pass additional information in the future.
        To be future-proof, implementations should accept any additional kwargs,
        and ignore them at this point.

        Examples:
        A simple implementation might just return a string, eg
        ```python
        class MyRelation:
            def __duckdb_template__(self, **kwargs):
                return "SELECT * FROM my_table"
        ```

        A more complex implementation might return an iterable of strings and IntoParams.
        An IntoParam is a dict with a "value" key, and optionally a "name" key.

        For example:
        ```python
        class User:
            def __init__(self, user_id: int):
                self.user_id = user_id

            def __duckdb_template__(self, **kwargs):
                user_id = self.user_id
                return t"SELECT * FROM users WHERE id = {user_id}"
        ```

        This will resolve to the final SQL and params:
        ```python
        {
            "sql": "SELECT * FROM users WHERE id = $p0_user_id",
            "params": [{"name": "p0_user_id", "value": 123}],
        }
        ```
        """


def param(value: object, name: str | None = None) -> ParamInterpolation:
    """Helper function to create an IntoParam with an optional name."""
    if name is not None:
        assert_param_name_legal(name)
    return ParamInterpolation(value=value, expression=name, conversion=None)


def template(thing: object, /, **ignored_kwargs) -> OurTemplate:
    """Convert something to a Template-ish.

    The rules are:
    - If the thing has a __duckdb_template__ method, call it and convert the
        resuling strings and IntoParams into a SqlTemplate.
    - If the thing is a string, treat it as raw SQL and return a SqlTemplate with that string.
    - If the thing is an IntoTemplate, resolve it into a SqlTemplate by recursively resolving
        any inner IntoInterpolations and flattening any nested templates.
    - If the thing is an IntoInterpolation, resolve it into a SqlTemplate by recursively resolving
        its value, and if it has a conversion specified (!s, !r, !a), treat it as raw SQL.
    - Otherwise, treat the thing as a param.
    """
    if isinstance(thing, SupportsDuckdbTemplate):
        raw = thing.__duckdb_template__(**ignored_kwargs)
        parts = [raw] if isinstance(raw, str) or is_into_interpolation(raw) else list(raw)
        return OurTemplate(*parts)
    if isinstance(thing, str):
        return OurTemplate(thing)
    if isinstance(thing, IntoTemplate):
        return OurTemplate(*thing)
    if isinstance(thing, IntoInterpolation):
        return OurTemplate(thing)
    return OurTemplate(param(value=thing))


def compile(thing: object) -> CompiledSql:
    t = template(thing)
    resolved = resolve(t)
    return compile_parts(resolved)


def resolve(parts: Iterable[str | IntoInterpolation]) -> OurTemplate[str | IntoInterpolation]:
    """Resolve an OurTemplate by recursively resolving any inner templates and interpolations."""
    resolved: list[str | IntoInterpolation] = []
    for part in parts:
        if isinstance(part, str):
            resolved.append(part)
        else:
            resolved.extend(resolve_interpolation(part))
    return OurTemplate(*resolved)


def resolve_interpolation(interp: IntoInterpolation) -> Iterable[str | IntoInterpolation]:
    value = interp.value
    # if conversion specified (!s, !r, !a), treat as raw sql, eg
    # name = "Alice"
    # t"SELECT * FROM users  where name = '{name!s}'" should be
    # "SELECT * FROM users where name = 'Alice'", eg with no param,
    # since the user is explicitly asking for the value to be directly interpolated into the SQL string,
    # rather than passed as a param.
    # This is useful for cases where the value is not something that can be passed as a param,
    # eg an identifier like a table or column name, or a SQL expression like "CURRENT_DATE",
    # or if the user just wants to write raw SQL and doesn't care about safety
    #  Note that this is potentially unsafe if the value comes from an untrusted source,
    # since it could lead to SQL injection vulnerabilities, so it should be used with caution.
    if interp.conversion == "s":
        return OurTemplate(str(value))
    elif interp.conversion == "r":
        return OurTemplate(repr(value))
    elif interp.conversion == "a":
        return OurTemplate(ascii(value))

    if isinstance(value, str):
        # do NOT pass to template, since that would treat it as a raw SQL.
        return OurTemplate(param(value, name=interp.expression))
    templ = template(value)
    # resolved = resolve(templ)
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
        return OurTemplate(param(value=value, name=interp.expression))
    else:
        # We got something nested, eg
        # age = 37
        # people = t"SELECT * FROM people WHERE age = {age}"
        # names = t"SELECT name FROM ({foo})"
        # names should resolve to:
        # "SELECT name FROM (SELECT * FROM people WHERE age = $age)", with a param $age=37
        return templ


def compile_parts(parts: Iterable[str | IntoInterpolation], /) -> CompiledSql:
    """Compile a resolved SqlTemplate into a final SQL string with named parameter placeholders, and a list of Params."""
    sql_parts: list[str] = []
    params: list[Param] = []
    for part in parts:
        if isinstance(part, str):
            sql_parts.append(part)
        else:
            param_name = f"p{len(params)}"
            if passed_name := part.expression:
                param_name += f"_{passed_name}"
            sql_parts.append(f"${param_name}")
            params.append({"name": param_name, "value": part.value})
    return {
        "sql": "".join(sql_parts),
        "params": params,
    }


@runtime_checkable
class IntoInterpolation(Protocol):
    """Something that can be converted into a string.templatelib.Interpolation."""

    value: object
    conversion: Literal["s", "r", "a"] | None
    expression: str | None
    format_spec: str


@runtime_checkable
class IntoTemplate(Protocol):
    """Something that can be converted into string.templatelib.Template."""

    strings: Sequence[str]
    interpolations: Sequence[IntoInterpolation]

    def __iter__(self) -> Iterator[str | IntoInterpolation]:
        """Iterate over the strings and interpolations in order."""


def assert_param_name_legal(name: str) -> None:
    """Eg `$param_1` is legal, but `$1param`, `$param-1`, `$param 1`, and `$p ; DROP TABLE users` are not."""
    # not implemented yet
    # Not exactly sure what part of the stack this should get called in,
    # or perhaps we shouldn't even check, just pass it to duckdb and let it error if it's illegal


class ParamInterpolation:
    """A simple implementation of IntoInterpolation, for testing purposes."""

    def __init__(
        self, value: object, conversion: Literal["s", "r", "a"] | None = None, expression: str | None = None
    ) -> None:
        self.value = value
        self.conversion = conversion
        self.expression = expression
        self.format_spec = ""


class OurTemplate:
    """A simple implementation of IntoTemplate, for testing purposes."""

    def __init__(
        self,
        *parts: str | IntoInterpolation,
    ) -> None:
        self.strings, self.interpolations = parse_strings_and_params(parts)

    def __iter__(self) -> Iterator[str | IntoInterpolation]:
        """Iterate over the strings and interpolations in order."""
        for s, i in zip(self.strings, self.interpolations, strict=False):
            yield s
            yield i
        yield self.strings[-1]

    def resolve(self) -> OurTemplate:
        """Resolve any inner templates and interpolations, returning a new OurTemplate with only strings and ParamInterpolations."""
        return resolve(self)

    def compile(self) -> CompiledSql:
        """Compile this template into a final SQL string with named parameter placeholders, and a list of Params."""
        resolved = self.resolve()
        return compile_parts(resolved)


T = TypeVar("T")


def parse_strings_and_params(
    parts: Iterable[str | T],
) -> tuple[tuple[str, ...], tuple[T, ...]]:
    """Parse an iterable of strings and params into separate tuples of strings and params, merging adjacent strings and ensuring that the number of strings is one more than the number of params."""
    strings, params = [], []
    last_thing: Literal["string", "param"] | None = None
    for part in parts:
        if isinstance(part, str):
            if last_thing == "string":
                # Merge adjacent string parts for efficiency, since the template engine allows that
                strings[-1] += part
            strings.append(part)
            last_thing = "string"
        else:
            if last_thing is None or last_thing == "param":
                # this is the first part or there were two adjacent params,
                # so we need an empty string spacer
                strings.append("")
            params.append(part)
            last_thing = "param"
    if last_thing == "param":
        # If the last part was a param, we need to end with an empty string
        strings.append("")
    assert len(strings) == len(params) + 1
    return tuple(strings), tuple(params)

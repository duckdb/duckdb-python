"""Template system for duckdb SQL statements, based on Python's string.templatelib."""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, NoReturn, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "CompiledSql",
    "IntoInterpolation",
    "Param",
    "SupportsDuckdbTemplate",
    "compile",
    "param",
    "template",
]


@dataclasses.dataclass(frozen=True, slots=True)
class CompiledSql:
    """Represents a compiled SQL statement, with the final SQL string and a dict of params to be passed to duckdb.

    You will typically not create this directly, but will get it as the result
    of calling .compile() on a SqlTemplate or ResolvedSqlTemplate.

    Example:
    >>> age = 37
    >>> c = compile(t"SELECT * FROM users WHERE age >= {age}")
    >>> c
    >>> CompiledSql(sql="SELECT * FROM users WHERE age >= $p0_age", params={"p0_age": 37})
    duckdb.query(c.sql, c.params)
    """

    sql: str
    params: dict[str, object] = dataclasses.field(default_factory=dict)


@runtime_checkable
class SupportsDuckdbTemplate(Protocol):
    """Something that can be converted into a Template by implementing the __duckdb_template__ method."""

    def __duckdb_template__(
        self, /, **future_kwargs
    ) -> (
        str
        | IntoInterpolation
        | Param
        | SupportsDuckdbTemplate
        | object
        | Iterable[str | IntoInterpolation | Param | SupportsDuckdbTemplate | object]
    ):
        """Convert self into something that template() understands."""


@dataclasses.dataclass(frozen=True, slots=True)
class Param:
    """Represents a parameter to be passed to duckdb, with an optional name and an optional exact flag."""

    value: object
    name: str | None = None
    exact: bool = False

    def __post_init__(self) -> None:
        """Ensure passed args were valid."""
        if self.exact:
            if self.name is None:
                msg = "Param with exact=True must have a name."
                raise ValueError(msg)
            else:
                assert_param_name_legal(self.name)


def param(value: object, name: str | None = None, *, exact: bool = False) -> Param:
    """Helper function to create an Param with an optional name and an optional exact flag."""
    return Param(value=value, name=name, exact=exact)


def template(*parts: str | IntoInterpolation | Param | SupportsDuckdbTemplate | object) -> SqlTemplate:
    """Convert a sequence of things into a SqlTemplate.

    We go through the parts and convert it into a sequence of str and Interpolations,
    which we then hand off to SqlTemplate.
    - If the thing has a `.__duckdb_template__()` method, call it,
      and call template() recursively on the result.
    - If the thing is a `str`, treat it as raw SQL and return a SqlTemplate with that string.
    - If it's an Interpolation, leave it as is, treating it as an interpolation.
    - It it's a Param, wrap it in an Interpolation.
    - Otherwise, treat the thing as a param, and then wrap the Param in an Interpolation.

    Examples:
    A very simple example is just passing a string, which will be treated as raw SQL:

    >>> t = template("SELECT * FROM users WHERE id = 123")
    >>> repr(t)
    SqlTemplate('SELECT * FROM users WHERE id = 123')
    >>> t.compile()
    CompiledSql(sql='SELECT * FROM users WHERE id = 123', params={})

    In python 3.14+, [tstrings](https://docs.python.org/3/library/string.templatelib.html)
    are very useful here.
    Any interpolation inside a tstring will be richly interpreted,
    either treated as a param, or expanded as a subquery:

    >>> user_id = 123
    >>> t = template(t"SELECT * FROM users WHERE id = {user_id}")
    >>> repr(t)
    SqlTemplate('SELECT * FROM users WHERE id = ', Param(value=123, name='user_id'), '')
    >>> t.compile()
    CompiledSql(sql='SELECT * FROM users WHERE id = $p0_user_id', params={'p0_user_id': 123})

    This is very friendly with chaining relations:
    >>> all_people = duckdb.sql("SELECT * FROM people")
    >>> age = 18
    >>> adults = template(t"SELECT * FROM ({all_people}) WHERE age >= {age}")
    >>> names = template(t"SELECT name FROM ({adults})")
    >>> names.compile()
    CompiledSql(sql='SELECT name FROM (SELECT * FROM (SELECT * FROM people) WHERE age >= $p0_age)', params={'p0_age': 18})

    We also support iterables of strings and Interpolations/Params/etc,
    which will be joined together into a single template.
    This is very useful for versions of python before 3.14 that don't have tstrings,
    since it allows you to build up a template from smaller pieces:

    >>> t = template("SELECT * FROM (", all_people, ") WHERE age >= ", age)
    >>> t.compile()
    CompiledSql(sql='SELECT * FROM (SELECT * FROM people) WHERE age >= $p0_age', params={'p0_age': 18})

    You can define evaluation logic for your custom types by defining a `.__duckdb_template__()` method.
    If this method is defined, the result of that call will be used to create the template.

    >>> class Record:
    ...     def __init__(self, table_name: str, id: int):
    ...         self.table_name = table_name
    ...         self.id = id
    ...
    ...     def __duckdb_template__(self, **kwargs):
    ...         id = self.id
    ...         # note the use of !s to indicate that the table name should be treated as raw SQL
    ...         return t"SELECT * FROM {self.table_name!s} WHERE id = {id}"
    >>> t = template(Record("users", 123))
    >>> t.compile()
    CompiledSql(sql='SELECT * FROM users WHERE id = $p0_id', params={'p0_id': 123})
    """  # noqa: E501
    expanded = []
    for part in parts:
        expanded.extend(_expand_part(part))
    return SqlTemplate(*expanded)


def compile(*part: str | IntoInterpolation | Param | SupportsDuckdbTemplate | object) -> CompiledSql:
    """Compile a sequence of things into a final SQL string with named parameter placeholders, and a list of Params.

    This is a convenience function that combines template() and .compile() into one step.

    For more details and examples, see template().
    """
    t = template(*part)
    return t.compile()


def _expand_part(part: object) -> Iterable[str | IntoInterpolation]:
    if isinstance(part, SupportsDuckdbTemplate):
        raw = part.__duckdb_template__()
        if isinstance(raw, str):  # noqa: SIM114
            yield raw
        elif isinstance(raw, IntoInterpolation):
            yield raw
        elif isinstance(raw, Param):
            yield ParamInterpolation(raw)
        elif isinstance(raw, Iterable):
            yield from _expand_part(raw)
        else:
            p = param(value=raw)
            yield ParamInterpolation(p)
    elif isinstance(part, str):  # noqa: SIM114
        yield part
    elif isinstance(part, IntoInterpolation):
        yield part
    elif isinstance(part, Param):
        yield ParamInterpolation(part)
    else:
        p = param(value=part)
        yield ParamInterpolation(p)


class ParamInterpolation:
    """A simple wrapper that implements the IntoInterpolation protocol for a given Param."""

    def __init__(self, param: Param):  # noqa: ANN204
        self.value = param
        self.expression = param.name
        self.conversion = None
        self.format_spec = ""

    def __repr__(self) -> str:
        return repr(self.value)


def _resolve(parts: Iterable[str | IntoInterpolation]) -> ResolvedSqlTemplate:
    """Resolve a stream of strings and Interpolations, recursively resolving inner interpolations."""
    resolved: list[str | Param] = []
    for part in parts:
        if isinstance(part, str):
            resolved.append(part)
        else:
            resolved.extend(_resolve_interpolation(part))
    return ResolvedSqlTemplate(resolved)


def _resolve_interpolation(interp: IntoInterpolation) -> Iterable[str | Param]:
    value = interp.value
    if isinstance(value, Param):
        # If it's already a Param, we can skip the template resolution and just return it as a param.
        return (value,)

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
    #
    # Follow Python's f-string semantics: apply conversion first, then format_spec.
    # e.g. {x!r:.10} means format(repr(x), ".10")
    if interp.conversion == "s":
        converted = str(value)
    elif interp.conversion == "r":
        converted = repr(value)
    elif interp.conversion == "a":
        converted = ascii(value)
    else:
        converted = None

    if converted is not None:
        return (format(converted, interp.format_spec),)

    if isinstance(value, str):
        # do NOT pass to template, since that would treat it as a raw SQL.
        return (param(value, name=interp.expression),)
    templ = template(value)
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
        return (param(value, name=interp.expression),)
    else:
        # We got something nested, eg
        # age = 37
        # people = t"SELECT * FROM people WHERE age = {age}"
        # names = t"SELECT name FROM ({foo})"
        # names should resolve to:
        # "SELECT name FROM (SELECT * FROM people WHERE age = $age)", with a param $age=37
        return _resolve(templ)


@runtime_checkable
class IntoInterpolation(Protocol):
    """Something that can be converted into a string.templatelib.Interpolation."""

    value: object
    conversion: Literal["s", "r", "a"] | None
    expression: str | None
    format_spec: str


def assert_param_name_legal(name: str) -> None:
    """Eg `$param_1` is legal, but `$1param`, `$param-1`, `$param 1`, and `$p ; DROP TABLE users` are not."""
    # not implemented yet
    # Not exactly sure what part of the stack this should get called in,
    # or perhaps we shouldn't even check, just pass it to duckdb and let it error if it's illegal


class SqlTemplate:
    """A sequence of strings and Interpolations."""

    def __init__(self, *parts: str | IntoInterpolation) -> None:
        self.strings, self.interpolations = parse_parts(parts)

    def __iter__(self) -> Iterator[str | IntoInterpolation]:
        """Iterate over the strings and interpolations in order."""
        for s, i in zip(self.strings, self.interpolations, strict=False):
            yield s
            yield i
        yield self.strings[-1]

    def resolve(self) -> ResolvedSqlTemplate:
        """Recursively resolve Interpolations into Params, returning a ResolvedSqlTemplate."""
        return _resolve(self)

    def compile(self) -> CompiledSql:
        """Compile this template into a final SQL string with named parameter placeholders, and a list of Params."""
        resolved = self.resolve()
        return compile_parts(resolved)

    def __str__(self) -> NoReturn:
        msg = "SqlTemplate cannot be directly converted to a string, since it may contain unresolved interpolations. Please call .resolve() or .compile() first."  # noqa: E501
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        part_strings = [repr(part) for part in self]
        return f"SqlTemplate({', '.join(part_strings)})"


class ResolvedSqlTemplate:
    """A SqlTemplate that has been resolved to only strings and Params."""

    def __init__(self, parts: Iterable[str | Param]) -> None:
        self.parts = tuple(parts)

    def compile(self) -> CompiledSql:
        """Compile this template into a final SQL string with named parameter placeholders, and a list of Params."""
        return compile_parts(self.parts)

    def __str__(self) -> NoReturn:
        msg = "ResolvedSqlTemplate cannot be directly converted to a string, since it may contain unresolved interpolations. Please call .compile() first."  # noqa: E501
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        part_strings = []
        for part in self.parts:
            if isinstance(part, str):
                part_strings.append(repr(part))
            else:
                part_strings.append(f"{{{part.name}={part.value}}}")
        return f"ResolvedSqlTemplate({', '.join(part_strings)})"

    def __iter__(self) -> Iterator[str | Param]:
        yield from self.parts


T = TypeVar("T")


def parse_parts(parts: Iterable[str | T]) -> tuple[tuple[str, ...], tuple[T, ...]]:
    """Parse an iterable of strings and others into separate tuples of strings and others.

    This merges adjacent strings and ensuring that the number of strings is one more than the number of others.
    """
    strings, others = [], []
    last_thing: Literal["string", "other"] | None = None
    for part in parts:
        if isinstance(part, str):
            if last_thing == "string":
                # Merge adjacent string parts
                strings[-1] += part
            else:
                strings.append(part)
            last_thing = "string"
        else:
            if last_thing is None or last_thing == "other":
                # this is the first part or there were two adjacent others,
                # so we need an empty string spacer
                strings.append("")
            others.append(part)
            last_thing = "other"
    if last_thing is None:
        # Empty input — return a single empty string to maintain the invariant
        strings.append("")
    elif last_thing == "other":
        # If the last part was an other, we need to end with an empty string
        strings.append("")
    assert len(strings) == len(others) + 1
    return tuple(strings), tuple(others)


def compile_parts(parts: Iterable[str | Param], /) -> CompiledSql:
    """Compile parts into a final SQL string with named parameter placeholders, and a list of Params."""
    sql_parts: list[str] = []
    params_items = []

    def next_name(suffix: str | None = None) -> str:
        # still count exact params in the count, so we get p0, my_param, p2, p3, my_param_2, p5, etc
        base = f"p{len(params_items)}"
        if suffix is not None:
            return f"{base}_{suffix}"
        else:
            return base

    for part in parts:
        if isinstance(part, str):
            sql_parts.append(part)
        else:
            if passed_name := part.name:
                param_name = passed_name if part.exact else next_name(passed_name)
            else:
                param_name = next_name()
            assert_param_name_legal(param_name)
            sql_parts.append(f"${param_name}")
            params_items.append((param_name, part.value))
    param_name_counts = Counter(name for name, _ in params_items)
    dupes = [name for name, count in param_name_counts.items() if count > 1]
    if dupes:
        msg = f"Duplicate parameter names found: {dupes}. Please ensure all parameter names are unique."
        raise ValueError(msg)
    return CompiledSql(sql="".join(sql_parts), params=dict(params_items))

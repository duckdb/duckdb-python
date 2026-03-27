from collections.abc import Iterable
from typing import Literal, Protocol, TypedDict, runtime_checkable

from typing_extensions import Self


# from string.templatelib import Interpolation, Template
class PInterpolation(Protocol):
    value: object
    conversion: Literal["s", "r", "a"] | None


@runtime_checkable
class PTemplate(Protocol):
    strings: list[str]
    interpolations: list[PInterpolation]


class Param(TypedDict):
    name: str
    value: object


class ResolvedQuery(TypedDict):
    sql: str
    params: list[Param]


class PQueryContext(Protocol):
    def current_params(self) -> list[Param]:
        """Return the list of parameters currently in the context."""

    def resolve_next(
        self,
        thing: object,
        /,
    ) -> str:
        """Resolve the given thing as the next step in resolving a query."""

    def add_param(self, value: object, /, name: str | None = None) -> Param:
        """Create a new parameter with the given value and an optional name. Does NOT add it to the context."""


class ResolveOneFunc(Protocol):
    def __call__(self, thing: object, ctx: PQueryContext, /, **ignored_kwargs) -> str: ...


class QueryContext:
    def __init__(self, params: Iterable[Param] | None = None, resolver: ResolveOneFunc | None = None):
        self._params: list[Param] = list(params) if params is not None else []
        self._resolver: ResolveOneFunc = resolver if resolver is not None else resolve_one

    def current_params(self) -> list[Param]:
        # Full deep copy so the user can't mess up the internal state
        return [{"name": p["name"], "value": p["value"]} for p in self._params]

    def resolve_next(self, thing: object, /) -> str:
        return self._resolver(thing, self)

    def add_param(self, value: object, /, name: str | None = None) -> Param:
        param_name = (
            name
            if name is not None
            else generate_unique_param_name((p["name"] for p in self._params), template="param_{idx}")
        )
        p = Param(name=param_name, value=value)
        self._params.append(p)
        return p


def resolve_query(thing: object) -> ResolvedQuery:
    ctx = QueryContext()
    sql = resolve_one(thing, ctx)
    return {"sql": sql, "params": ctx.current_params()}


def resolve_one(thing: object, ctx: PQueryContext, /, **ignored_kwargs) -> str:
    if isinstance(thing, SupportsDuckdbResolve):
        return thing.__duckdb_resolve__(ctx, **ignored_kwargs)
    if isinstance(thing, PTemplate):
        return resolve_template(thing, ctx)
    param = ctx.add_param(thing)
    return f"${param['name']}"


def resolve_template(template: PTemplate, ctx: PQueryContext) -> str:
    """Resolve a Template, recursively resolving any interpolations and applying any specified conversions."""
    sql_parts = []

    for i, static_part in enumerate(template.strings):
        sql_parts.append(static_part)
        if i < len(template.interpolations):
            interp = template.interpolations[i]
            value = interp.value

            # Apply conversion if specified (!s, !r, !a)
            if interp.conversion == "s":
                value = str(value)
            elif interp.conversion == "r":
                value = repr(value)
            elif interp.conversion == "a":
                value = ascii(value)

            sql_parts.append(ctx.resolve_next(value))

    return "".join(sql_parts)


def generate_unique_param_name(existing_names: Iterable[str], *, template: str | None = None) -> str:
    """Generate a unique parameter name that does not conflict with existing_names.

    If template is provided, it should be a string with a single {idx} placeholder
    that will be replaced with an integer index to generate candidate names.
    If template is None, a default template of "param_{idx}" will be used.
    """
    if template is None:
        template = "param_{idx}"
    existing_names = set(existing_names)
    idx = 1
    while True:
        candidate = template.format(idx=idx)
        if candidate not in existing_names:
            return candidate
        idx += 1


@runtime_checkable
class SupportsDuckdbResolve(Protocol):
    def __duckdb_resolve__(self, ctx: PQueryContext, /, **future_kwargs) -> str: ...


class DuckdDbPyRelation:
    def __duckdb_resolve__(self, ctx: PQueryContext, /, **future_kwargs) -> str:
        # this would just return the existing SQL.
        return "SELECT * FROM some_table"

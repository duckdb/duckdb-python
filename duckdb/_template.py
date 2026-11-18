"""Template strings (PEP 750) as the query of `execute()` and `sql()`.

Every interpolated value is bound as a parameter, or quoted as a name with `:i`, and never written into the SQL text.
"""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
import re
import sys
import unicodedata
from typing import TYPE_CHECKING, Any

import _duckdb

from duckdb import _relation
from duckdb._expression import identifier, text_tokens
from duckdb._relation import DuckDBPyRelation, _encodable, _error, _has_values, _markers

if TYPE_CHECKING:
    from collections.abc import Iterator

if sys.version_info >= (3, 14):
    from string.templatelib import Template

    def is_template(query: object) -> bool:
        """Whether a query is a template string."""
        return isinstance(query, Template)

else:

    def is_template(query: object) -> bool:
        """Whether a query is a template string, which Python has from 3.14 on."""
        return False


WITH_PARAMS = (
    "A template carries its own values, so params cannot be given with it; put the values in the template, "
    "or pass a str with params"
)


@dataclasses.dataclass(frozen=True)
class _Field:
    """An interpolation as the user wrote it: its template's text, and where in that text it stands."""

    written: str
    start: int
    end: int
    expression: str
    #: The fields of the templates it is spliced into, innermost first.
    within: tuple[str, ...]

    @property
    def text(self) -> str:
        return self.written[self.start : self.end]

    def named(self) -> str:
        return " in ".join((self.text, *self.within))


@dataclasses.dataclass(frozen=True)
class _Slot:
    """An interpolation's place in the statement: a parameter marker, or a name when `number` is None."""

    start: int
    end: int
    number: int | None
    field: _Field


@dataclasses.dataclass(frozen=True)
class _Run:
    """A run of a template's own text, at `start` in the statement and at `offset` in the template."""

    start: int
    length: int
    written: str
    offset: int
    within: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _Compiled:
    text: str
    values: list[object]
    template: Any
    slots: list[_Slot]

    def note(self) -> str:
        """Which field each marker of the statement came from, with the template as written."""
        origins = ", ".join(f"${slot.number} is {slot.field.named()}" for slot in self.slots if slot.number is not None)
        head = "The statement came from a template" + (f"; {origins}" if origins else "") + ":"
        return head + "".join(f"\n    {line}" for line in _LINE_BREAK.split(_written(self.template)[0]))


def _field_text(expression: str, conversion: str | None, spec: str) -> str:
    return "{" + expression + (f"!{conversion}" if conversion else "") + (f":{spec}" if spec else "") + "}"


def _written(template: Any) -> tuple[str, list[int], list[tuple[int, int]]]:  # noqa: ANN401
    """The template as written, each field in braces; where each of its strings starts, and each field's span."""
    pieces: list[str] = []
    string_offsets: list[int] = []
    field_spans: list[tuple[int, int]] = []
    at = 0
    for index, string in enumerate(template.strings):
        string_offsets.append(at)
        pieces.append(string)
        at += len(string)
        if index < len(template.interpolations):
            item = template.interpolations[index]
            text = _field_text(item.expression, item.conversion, item.format_spec)
            field_spans.append((at, at + len(text)))
            pieces.append(text)
            at += len(text)
    return "".join(pieces), string_offsets, field_spans


_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def _width(char: str) -> int:
    """The columns a terminal gives a character: two when wide, none for a combining mark or a format character."""
    # The soft hyphen is a format character that terminals, and the engine, still give a column.
    if unicodedata.category(char) in ("Mn", "Me", "Cf") and char != "\u00ad":
        return 0
    return 2 if unicodedata.east_asian_width(char) in "WF" else 1


def _columns(text: str) -> int:
    return sum(map(_width, text))


def _excerpt(written: str, start: int, end: int) -> str:
    """The line of `written` holding `start`, with carets under `start` to `end`, as the engine shows a position."""
    breaks = [match.end() for match in _LINE_BREAK.finditer(written, 0, start)]
    line_start = breaks[-1] if breaks else 0
    following = _LINE_BREAK.search(written, start)
    line_end = following.start() if following else len(written)
    prefix = f"LINE {len(breaks) + 1}: "
    carets = "^" * max(1, _columns(written[start : min(end, line_end)]))
    indent = " " * (len(prefix) + _columns(written[line_start:start]))
    return f"\n\n{prefix}{written[line_start:line_end]}\n{indent}{carets}"


def _refusal(message: str) -> Exception:
    return _error(_duckdb.InvalidInputException, message)


def _refuse_at(field: _Field, message: str) -> Exception:
    return _refusal(message + _excerpt(field.written, field.start, field.end))


def _part_kind(part: object) -> str | None:
    """What is wrong with one part of a name, as `con.table()` refuses it; None for a part it takes."""
    if not isinstance(part, str):
        return type(part).__name__
    return "an empty str" if not part else None


def _quoted_name(value: object, field: _Field) -> str:
    if isinstance(value, tuple):
        kinds = [kind for kind in map(_part_kind, value) if kind is not None]
        kind = f"a tuple holding {kinds[0]}" if kinds else (None if value else "an empty tuple")
    else:
        kind = _part_kind(value)
    if kind is not None:
        message = f"{field.named()} is a name, so its value must be a str or a tuple of str, not {kind}"
        raise _refuse_at(field, message)
    parts = value if isinstance(value, tuple) else (value,)
    return identifier(tuple(str.__str__(part) for part in parts))


class _Builder:
    """The statement a template compiles to, built one template string and one field at a time."""

    def __init__(self) -> None:
        self.pieces: list[str] = []
        self.length = 0
        self.values: list[object] = []
        self.slots: list[_Slot] = []
        self.runs: list[_Run] = []

    def _append(self, text: str) -> int:
        start = self.length
        self.pieces.append(text)
        self.length += len(text)
        return start

    def _slot(self, text: str, number: int | None, field: _Field) -> None:
        start = self._append(text)
        self.slots.append(_Slot(start, self.length, number, field))

    def add(self, template: Any, within: tuple[str, ...]) -> None:  # noqa: ANN401
        written, string_offsets, field_spans = _written(template)
        for index, string in enumerate(template.strings):
            if string:
                self.runs.append(_Run(self._append(string), len(string), written, string_offsets[index], within))
            if index == len(template.interpolations):
                break
            item = template.interpolations[index]
            field = _Field(written, *field_spans[index], item.expression, within)
            expression, spec, value = item.expression, item.format_spec, item.value
            if item.conversion:
                message = (
                    f"{field.named()} converts its value, which a query does not do; write "
                    f"{_field_text(expression, None, spec)}, which takes the value as it is"
                )
                raise _refuse_at(field, message)
            if is_template(value):
                if spec != "q":
                    message = f"{field.named()} is a template; write {{{expression}:q}} to splice it into the query"
                    raise _refuse_at(field, message)
                self.add(value, (field.text, *within))
            elif spec == "q":
                message = (
                    f"{field.named()} splices a template, but its value is a {type(value).__name__}; write "
                    f"{{{expression}}} for a value, or {{{expression}:i}} for a name"
                )
                raise _refuse_at(field, message)
            elif spec == "i":
                self._slot(_quoted_name(value, field), None, field)
            elif spec:
                message = (
                    f"{field.named()} has the format spec '{spec}', which a query does not take; write "
                    f"{{{expression}}} for a value, {{{expression}:i}} for a name, or {{{expression}:q}} for a template"
                )
                raise _refuse_at(field, message)
            elif isinstance(value, DuckDBPyRelation):
                message = (
                    f"{field.named()} is a relation, which a template cannot take as a value; write the name of a "
                    "variable that holds it in the query text"
                )
                raise _refuse_at(field, message)
            else:
                self.values.append(value)
                self._slot(f"${len(self.values)}", len(self.values), field)

    def text(self) -> str:
        return "".join(self.pieces)


def _misplaced(text: str, tokens: list[tuple[int, _duckdb.token_type]], slot: _Slot) -> Exception:
    """Why a field does not stand as a token of its own, from the token it fell into."""
    starts = [offset for offset, _ in tokens]
    index = bisect.bisect_right(starts, slot.start) - 1
    offset, kind = tokens[index] if index >= 0 else (slot.start, None)
    what = "a parameter" if slot.number is not None else "a name"
    named = slot.field.named()
    expression = slot.field.expression
    if offset < slot.start and kind == _duckdb.token_type.string_const:
        advice = (
            "make the whole literal the value, or join the value to the literal with ||"
            if slot.number is not None
            else "take it out of the literal"
        )
        message = f"{named} is inside a string literal, where it would be text, not {what}; {advice}"
    elif offset < slot.start and kind == _duckdb.token_type.comment:
        message = f"{named} is inside a comment, where it would be ignored; take it out of the comment"
    elif offset < slot.start and kind == _duckdb.token_type.identifier and text[offset] in '"`':
        if slot.number is not None:
            message = (
                f"{named} is inside a quoted identifier, where it would be part of the name, not a parameter; "
                f"write {{{expression}:i}} for a name"
            )
        else:
            message = f"{named} is inside a quoted identifier, and quotes itself; take away the quotes around it"
    else:
        return _glued(slot)
    return _refuse_at(slot.field, message)


def _glued(slot: _Slot) -> Exception:
    what = "a parameter" if slot.number is not None else "a name"
    message = (
        f"{slot.field.named()} touches the text next to it, so it is not {what} of its own; put a space between them"
    )
    return _refuse_at(slot.field, message)


_WORDS = frozenset(
    {
        _duckdb.token_type.identifier,
        _duckdb.token_type.keyword,
        _duckdb.token_type.numeric_const,
        _duckdb.token_type.string_const,
    }
)


def _touches_word(text: str, tokens: list[tuple[int, _duckdb.token_type]], starts: list[int], slot: _Slot) -> bool:
    """Whether a word, a number, a string or a `$` touches the slot, which SQL would read as one token or an alias."""
    before = bisect.bisect_left(starts, slot.start) - 1
    # A `$` right before a field would make it a parameter's name.
    if (
        before >= 0
        and not text[slot.start - 1].isspace()
        and (tokens[before][1] in _WORDS or text[slot.start - 1] == "$")
    ):
        return True
    after = bisect.bisect_left(starts, slot.end)
    return after < len(starts) and starts[after] == slot.end and tokens[after][1] in _WORDS


def _stands_alone(text: str, starts: list[int], start: int, end: int) -> bool:
    """Whether the tokens from `start` to `end` are those of that text alone, so no neighbour merges into them."""
    first = bisect.bisect_left(starts, start)
    last = bisect.bisect_left(starts, end)
    own = [offset + start for offset, _ in text_tokens(text[start:end])]
    following = starts[last] if last < len(starts) else len(text)
    return starts[first:last] == own and not text[end:following].strip()


def _check_places(builder: _Builder, text: str) -> None:
    """Refuse a field that does not stand as its own marker or name, and a marker the template's own text holds."""
    tokens = text_tokens(text)
    starts = [offset for offset, _ in tokens]
    slot_starts = {slot.start for slot in builder.slots}
    slot_ends = {slot.end for slot in builder.slots}
    for slot in builder.slots:
        if not _stands_alone(text, starts, slot.start, slot.end):
            raise _misplaced(text, tokens, slot)
        if slot.start in slot_ends or slot.end in slot_starts or _touches_word(text, tokens, starts, slot):
            raise _glued(slot)
    taken = {slot.start for slot in builder.slots if slot.number is not None}
    for marker in _markers(text):
        if marker.start in taken:
            continue
        # Every slot stands as its own tokens by now, so a marker left over starts in the template's own text.
        run = next(run for run in builder.runs if run.start <= marker.start < run.start + run.length)
        of = " of " + " in ".join(run.within) if run.within else ""
        offset = run.offset + marker.start - run.start
        message = (
            f"The template text{of} holds the parameter marker {text[marker.start : marker.end]}, which nothing gives "
            "a value; write the value in its place, as {value}"
        )
        raise _refusal(message + _excerpt(run.written, offset, offset + marker.end - marker.start))


def _compiled(query: Any, params: object) -> _Compiled:  # noqa: ANN401
    if _has_values(params):
        raise _refusal(WITH_PARAMS)
    builder = _Builder()
    builder.add(query, ())
    text = _encodable(builder.text())
    _check_places(builder, text)
    return _Compiled(text, builder.values, query, builder.slots)


@contextlib.contextmanager
def _noted(compiled: _Compiled) -> Iterator[None]:
    """Add the template's note to a DuckDB error raised while its statement runs."""
    try:
        yield
    except _duckdb.Error as error:
        error.add_note(compiled.note())
        raise


def execute(con: _duckdb.DuckDBPyConnection, query: Any, params: object) -> _duckdb.DuckDBPyConnection:  # noqa: ANN401
    """Run a template string as `execute()` runs a str, its values bound as the statement's parameters."""
    compiled = _compiled(query, params)
    with _noted(compiled):
        return con.execute(compiled.text, compiled.values)


def sql(
    con: _duckdb.DuckDBPyConnection,
    query: Any,  # noqa: ANN401
    alias: str,
    params: object,
) -> DuckDBPyRelation | None:
    """The relation `sql()` makes of a template string, its values bound as the statement's parameters."""
    compiled = _compiled(query, params)
    with _noted(compiled):
        return _relation.sql(con, compiled.text, alias, compiled.values)

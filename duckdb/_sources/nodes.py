"""The predicate tree the Arrow scan offers a source, built in C++ from a filter the engine would apply.

A tree holds `Binary` for AND, OR and the comparisons = != < <= > >= of a `Col` (always on the left) with a
`Lit`, `Unary("NOT", predicate)`, `Postfix("IS NULL" or "IS NOT NULL", Col)` and `In(Col, [Lit, ...])`; BETWEEN
arrives as AND of two comparisons. `Col.parts` is the declared column name followed by the struct field names that
lead into it. `Lit.type` is the engine's name for the constant's type, which is exactly the column's type, such as
"INTEGER" or "DECIMAL(10,2)"; a "TIMESTAMP WITH TIME ZONE" value is the instant as an int count of microseconds
since the epoch, and every other value is the plain Python value.
"""

from __future__ import annotations

import datetime
import decimal
import math
import re
import struct
from dataclasses import dataclass


class Untranslatable(ValueError):
    """Raised by a translator for a node it cannot evaluate exactly as the engine would."""


class Node:
    """A node of the predicate tree."""

    def fragment(self) -> str:
        """The node as parenthesised SQL, the text EXPLAIN shows under Filters."""
        return fragment(self)

    def _children(self) -> tuple[Node, ...]:
        return ()

    def _render(self, children: list[str]) -> str:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class Col(Node):
    """A column, or a struct field inside one."""

    parts: tuple[str, ...]

    def _render(self, children: list[str]) -> str:
        return ".".join(_quote(part) for part in self.parts)


@dataclass(frozen=True, eq=False)
class Lit(Node):
    """A constant of the column's type."""

    value: object
    type: str

    def _render(self, children: list[str]) -> str:
        return _literal(self.value, self.type)


@dataclass(frozen=True, eq=False)
class Binary(Node):
    """A comparison of a column with a constant, or AND or OR of two predicates."""

    op: str
    left: Node
    right: Node

    def _children(self) -> tuple[Node, ...]:
        return (self.left, self.right)

    def _render(self, children: list[str]) -> str:
        return f"({children[0]} {self.op} {children[1]})"


@dataclass(frozen=True, eq=False)
class Unary(Node):
    """NOT of a predicate."""

    op: str
    operand: Node

    def _children(self) -> tuple[Node, ...]:
        return (self.operand,)

    def _render(self, children: list[str]) -> str:
        return f"({self.op} {children[0]})"


@dataclass(frozen=True, eq=False)
class Postfix(Node):
    """IS NULL or IS NOT NULL of a column."""

    op: str
    operand: Node

    def _children(self) -> tuple[Node, ...]:
        return (self.operand,)

    def _render(self, children: list[str]) -> str:
        return f"({children[0]} {self.op})"


@dataclass(frozen=True, eq=False)
class In(Node):
    """Membership of a column in a list of at least one constant."""

    operand: Node
    values: list[Node]

    def _children(self) -> tuple[Node, ...]:
        return (self.operand, *self.values)

    def _render(self, children: list[str]) -> str:
        if not self.values:
            message = "an IN needs at least one candidate"
            raise Untranslatable(message)
        return f"({children[0]} IN ({', '.join(children[1:])}))"


def fragment(node: Node) -> str:
    """The SQL text of `node`, built without recursion so that any depth of AND and OR renders."""
    texts: list[str] = []
    pending: list[tuple[Node, bool]] = [(node, False)]
    while pending:
        current, expanded = pending.pop()
        children = current._children()
        if children and not expanded:
            pending.append((current, True))
            pending.extend((child, False) for child in reversed(children))
            continue
        start = len(texts) - len(children)
        rendered = current._render(texts[start:])
        del texts[start:]
        texts.append(rendered)
    return texts[0]


_INTEGER_RANGES = {
    "TINYINT": (-(1 << 7), (1 << 7) - 1),
    "SMALLINT": (-(1 << 15), (1 << 15) - 1),
    "INTEGER": (-(1 << 31), (1 << 31) - 1),
    "BIGINT": (-(1 << 63), (1 << 63) - 1),
    "UTINYINT": (0, (1 << 8) - 1),
    "USMALLINT": (0, (1 << 16) - 1),
    "UINTEGER": (0, (1 << 32) - 1),
    "UBIGINT": (0, (1 << 64) - 1),
}

#: The step of each naive timestamp type, in microseconds.
_TIMESTAMP_STEPS = {"TIMESTAMP_S": 1_000_000, "TIMESTAMP_MS": 1_000, "TIMESTAMP": 1}

_DECIMAL = re.compile(r"DECIMAL\((\d+),(\d+)\)")


def _decimal_type(type_text: str) -> tuple[int, int] | None:
    """The precision and scale of a DECIMAL type text, None for any other type."""
    match = _DECIMAL.fullmatch(type_text)
    return (int(match[1]), int(match[2])) if match else None


def exact_value(constant: Lit) -> bool:
    """Whether the constant's value is exactly a value of its type, as the translators require."""
    value, type_text = constant.value, constant.type
    if type_text == "BOOLEAN":
        return isinstance(value, bool)
    if type_text in _INTEGER_RANGES or type_text == "TIMESTAMP WITH TIME ZONE":
        low, high = _INTEGER_RANGES.get(type_text, (-(1 << 63), (1 << 63) - 1))
        return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high
    if type_text in ("FLOAT", "DOUBLE"):
        if not isinstance(value, float) or not math.isfinite(value):
            return False
        try:
            return type_text == "DOUBLE" or struct.unpack("f", struct.pack("f", value))[0] == value
        except OverflowError:
            return False
    if type_text == "VARCHAR":
        return isinstance(value, str)
    if type_text == "BLOB":
        return isinstance(value, bytes)
    if type_text == "DATE":
        return isinstance(value, datetime.date) and not isinstance(value, datetime.datetime)
    if type_text == "TIME":
        return isinstance(value, datetime.time) and value.tzinfo is None
    if type_text in _TIMESTAMP_STEPS:
        return (
            isinstance(value, datetime.datetime)
            and value.tzinfo is None
            and value.microsecond % _TIMESTAMP_STEPS[type_text] == 0
        )
    shape = _decimal_type(type_text)
    if shape is not None and isinstance(value, decimal.Decimal):
        return _fits(value, *shape)
    return False


def _fits(value: decimal.Decimal, precision: int, scale: int) -> bool:
    if not value.is_finite():
        return False
    _, digits, exponent = value.as_tuple()
    unscaled = int("".join(map(str, digits)) or "0")
    shift = int(exponent) + scale
    if shift >= 0:
        unscaled *= 10**shift
    elif unscaled % 10**-shift == 0:
        unscaled //= 10**-shift
    else:
        return False
    return len(str(unscaled)) <= precision


def _quote(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _text(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _fraction(microseconds: int) -> str:
    return f".{microseconds:06d}".rstrip("0") if microseconds else ""


def _clock(value: datetime.time | datetime.datetime) -> str:
    return f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}{_fraction(value.microsecond)}"


def _literal(value: object, type_text: str) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if type_text == "TIMESTAMP WITH TIME ZONE" and isinstance(value, int):
        moment = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=value)
        return f"TIMESTAMPTZ '{moment.date().isoformat()} {_clock(moment)}+00'"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value!r}::DOUBLE"
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, bytes):
        # One escape per byte, since the engine reads exactly two hex digits after each.
        escaped = "".join(f"\\x{byte:02x}" for byte in value)
        return f"'{escaped}'::BLOB"
    if isinstance(value, decimal.Decimal) and value.is_finite():
        shape = _decimal_type(type_text)
        if shape is None:
            _, digits, exponent = value.as_tuple()
            scale = max(0, -int(exponent))
            shape = (max(len(digits) + max(int(exponent), 0), scale + 1), scale)
        return f"CAST('{value:f}' AS DECIMAL({shape[0]}, {shape[1]}))"
    if isinstance(value, datetime.datetime):
        return f"TIMESTAMP '{value.date().isoformat()} {_clock(value)}'"
    if isinstance(value, datetime.date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, datetime.time):
        return f"TIME '{_clock(value)}'"
    return f"{_text(str(value))}::{type_text}"

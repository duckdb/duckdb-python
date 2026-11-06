from __future__ import annotations  # noqa: D100

import datetime
import functools
import io
import json
import operator
import typing

import polars as pl
from polars.io.plugins import register_io_source

import duckdb

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import typing_extensions

_ExpressionTree: typing_extensions.TypeAlias = typing.Dict[str, typing.Union[str, int, "_ExpressionTree", typing.Any]]  # noqa: UP006


def _predicate_to_expression(predicate: pl.Expr) -> duckdb.Expression | None:
    """Convert a Polars predicate expression to a DuckDB-compatible SQL expression.

    Parameters:
        predicate (pl.Expr): A Polars expression (e.g., col("foo") > 5)

    Returns:
        SQLExpression: A DuckDB SQL expression string equivalent.
        None: If conversion fails.

    Example:
        >>> _predicate_to_expression(pl.col("foo") > 5)
        SQLExpression("(foo > 5)")
    """
    # Serialize the Polars expression tree to JSON
    tree = json.loads(predicate.meta.serialize(format="json"))
    return _tree_to_sql_expression(tree)


def _tree_to_sql_expression(
    tree: _ExpressionTree, schema: Mapping[str, pl.DataType] | None = None
) -> duckdb.Expression | None:
    """Convert an already-parsed Polars expression tree to a DuckDB expression.

    ``schema`` maps the columns of the scanned relation whose polars dtype stands for their engine type to that dtype;
    a comparison on any other column does not translate, and without a schema no cast of a column and no modulus does.
    Returns None if the tree contains a node we cannot translate to SQL.
    """
    try:
        return duckdb.SQLExpression(_predicate_to_sql(tree, schema))
    except Exception:
        # If the conversion fails, we return None
        return None


# Polars "dynamic predicates"
# ---------------------------
# When a slice / TOP-N sits above a scan, polars' optimizer pushes a *dynamic
# predicate* into the scan, AND-ed onto any real filter. It is an internal
# optimizer node, not a materializable expression: it serializes as a `Display`
# node with `fmt_str` "dynamic_pred: <uuid>", and feeding it back into
# `DataFrame.filter()` panics polars (`unreachable!` in `expr_to_ir`). It is only
# an early-pruning hint -- the limit above the scan still runs -- so we drop it
# and keep the real predicate, which we MUST still apply (polars trusts the
# source and does not re-filter above it). Reported as duckdb-python#460; for
# polars-side context see:
#   - https://github.com/pola-rs/polars/issues/21665  (why real + dynamic arrive AND-ed)
#   - https://github.com/pola-rs/polars/issues/22252  (filter() panicking on un-lowerable nodes)
def _is_dynamic_predicate_node(node: typing.Any) -> bool:  # noqa: ANN401
    """Return True if a serialized node is a dynamic predicate (see the note above).

    Detected by shape: a ``Display`` node whose ``fmt_str`` starts with ``dynamic_pred``.
    """
    if not isinstance(node, dict):
        return False
    display = node.get("Display")
    return (
        isinstance(display, dict)
        and isinstance(display.get("fmt_str"), str)
        and display["fmt_str"].startswith("dynamic_pred")
    )


def _tree_contains_dynamic_predicate(node: typing.Any) -> bool:  # noqa: ANN401
    """Return True if the serialized expression tree contains a dynamic predicate anywhere."""
    if _is_dynamic_predicate_node(node):
        return True
    if isinstance(node, dict):
        return any(_tree_contains_dynamic_predicate(child) for child in node.values())
    if isinstance(node, list):
        return any(_tree_contains_dynamic_predicate(child) for child in node)
    return False


def _strip_dynamic_predicates(tree: typing.Any) -> tuple[_ExpressionTree | None, bool]:  # noqa: ANN401
    """Remove dynamic-predicate conjuncts from a serialized predicate tree.

    See the note above for what a dynamic predicate is and why we drop it.

    Returns ``(stripped_tree, removed)``. ``stripped_tree`` is ``None`` when the
    predicate was purely dynamic. Raises ``NotImplementedError`` if a dynamic
    predicate appears anywhere other than a top-level ``And`` conjunct — a shape
    polars does not produce today, where the hint can neither be safely dropped
    nor applied.
    """
    if _is_dynamic_predicate_node(tree):
        return None, True
    if isinstance(tree, dict) and "BinaryExpr" in tree:
        bin_expr = tree["BinaryExpr"]
        if isinstance(bin_expr, dict) and bin_expr.get("op") == "And":
            left, left_removed = _strip_dynamic_predicates(bin_expr["left"])
            right, right_removed = _strip_dynamic_predicates(bin_expr["right"])
            removed = left_removed or right_removed
            if left is None:
                return right, removed
            if right is None:
                return left, removed
            return {"BinaryExpr": {**bin_expr, "left": left, "right": right}}, removed
    if _tree_contains_dynamic_predicate(tree):
        msg = "Cannot handle a polars dynamic predicate outside a top-level AND conjunct"
        raise NotImplementedError(msg)
    return tree, False


def _conjuncts(tree: _ExpressionTree) -> list[_ExpressionTree]:
    """The operands of a top level AND chain, nested on either side, or the tree itself."""
    bin_expr = tree.get("BinaryExpr")
    if isinstance(bin_expr, dict) and bin_expr.get("op") == "And":
        left, right = bin_expr.get("left"), bin_expr.get("right")
        if isinstance(left, dict) and isinstance(right, dict):
            return [*_conjuncts(left), *_conjuncts(right)]
    return [tree]


def _expression_from_tree(tree: _ExpressionTree) -> pl.Expr:
    """Rebuild a polars expression from a serialized tree (inverse of meta.serialize)."""
    return pl.Expr.deserialize(io.BytesIO(json.dumps(tree).encode()), format="json")


def _pl_operation_to_sql(op: str) -> str:
    """Map Polars comparison operation strings to SQL equivalents.

    And and Or are bitwise on integers, so only ``_predicate_to_sql`` translates them.

    Example:
        >>> _pl_operation_to_sql("Eq")
        '='
    """
    try:
        return {
            "Lt": "<",
            "LtEq": "<=",
            "Gt": ">",
            "GtEq": ">=",
            "Eq": "=",
        }[op]
    except KeyError:
        raise NotImplementedError(op)  # noqa: B904


def _escape_sql_identifier(identifier: str) -> str:
    """Escape SQL identifiers by doubling any double quotes and wrapping in double quotes.

    Example:
        >>> _escape_sql_identifier('column"name')
        '"column""name"'
    """
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


_INTEGER_RANGES = {
    f"{prefix}Int{bits}": (0, 2**bits - 1) if prefix else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    for prefix in ("", "U")
    for bits in (8, 16, 32, 64, 128)
}


_EXACT_IN_A_DOUBLE = frozenset({"Int8", "Int16", "Int32", "UInt8", "UInt16", "UInt32"})
# polars 1 casts a column compared with a Python float to the dynamic float type, which it computes as a Float64
_FLOAT64 = ("Float64", {"Unknown": "Float"})


def _compares_bare(column: pl.DataType | None, target: object) -> bool:
    """Whether a column of dtype ``column`` compares as its cast to the serialized dtype ``target`` does.

    An integer to an integer that holds it and a decimal to a decimal that keeps its integer and fractional digits keep
    every value. A cast to Float64 keeps every comparison with a double for an integer of 32 bits or fewer, which is a
    double exactly, and for a decimal of 15 digits or fewer, which converts to a double no other such decimal shares;
    the engine compares the bare column with a double's shortest text as an exact decimal, or as a double. An integer
    to a decimal keeps every value too, but the engine raises where the common decimal type of a comparison needs more
    than 38 digits, and polars casts a Date past 294247 or DATE 'infinity' to a NULL Datetime.
    """
    kind = target.get("Literal") if isinstance(target, dict) else None
    if kind in _FLOAT64:
        if isinstance(column, pl.Decimal):
            return column.precision is not None and column.precision <= 15
        return str(column) in _EXACT_IN_A_DOUBLE
    if isinstance(kind, str) and kind in _INTEGER_RANGES and str(column) in _INTEGER_RANGES:
        low, high = _INTEGER_RANGES[str(column)]
        target_low, target_high = _INTEGER_RANGES[kind]
        return target_low <= low and high <= target_high
    if isinstance(kind, dict) and isinstance(column, pl.Decimal) and column.precision is not None:
        precision_scale = kind.get("Decimal")
        if not (isinstance(precision_scale, list) and len(precision_scale) == 2):
            return False
        precision, scale = precision_scale
        if not (isinstance(precision, int) and isinstance(scale, int)):
            return False
        return scale >= column.scale and precision - scale >= column.precision - column.scale
    return False


def _integer_literal(tree: _ExpressionTree) -> int | None:
    """The value of a typed integer literal, as polars sends it in an optimized predicate, or None."""
    literal = tree.get("Literal")
    scalar = literal.get("Scalar") if isinstance(literal, dict) else None
    if isinstance(scalar, dict) and len(scalar) == 1:
        [(dtype, value)] = scalar.items()
        if dtype in _INTEGER_RANGES and isinstance(value, int):
            return value
    return None


def _is_integer(tree: _ExpressionTree, schema: Mapping[str, pl.DataType] | None) -> bool:
    """Whether polars types the node as an integer: a column of an integer dtype, or a cast to one."""
    if schema is None:
        return False
    column = tree.get("Column")
    if isinstance(column, str):
        return str(schema.get(column)) in _INTEGER_RANGES
    cast = tree.get("Cast")
    target = cast.get("dtype") if isinstance(cast, dict) else None
    kind = target.get("Literal") if isinstance(target, dict) else None
    return isinstance(kind, str) and kind in _INTEGER_RANGES


def _floored_modulus_to_sql(
    dividend: _ExpressionTree, divisor: _ExpressionTree, schema: Mapping[str, pl.DataType] | None
) -> str:
    """Translate polars' integer modulus, which takes the divisor's sign where the engine's ``%`` takes the dividend's.

    Only an integer dividend and a nonzero integer literal divisor translate: the engine raises on a zero divisor where
    polars gives NULL, and polars floors a float or decimal modulus that the engine truncates.
    """
    divisor_value = _integer_literal(divisor)
    if not divisor_value or not _is_integer(dividend, schema):
        msg = "Only an integer modulus by a nonzero integer literal can be pushed down"
        raise NotImplementedError(msg)
    remainder = f"({_pl_tree_to_sql(dividend, schema)} % {divisor_value})"
    # Adding the divisor moves a remainder of the other sign into the divisor's and stays inside the divisor's range
    other_sign = "<" if divisor_value > 0 else ">"
    return f"(CASE WHEN {remainder} {other_sign} 0 THEN {remainder} + {divisor_value} ELSE {remainder} END)"


def _decimal_to_sql(unscaled: int, scale: int) -> str:
    """A decimal literal at its exact value, typed with the digits it needs.

    The engine reads an integer exactly but a decimal text of more than 38 digits, a zero before the point included, as
    a DOUBLE, so a fraction is cast; its precision is the digits the value needs, so the literal never widens the
    common type of a comparison beyond what the column and the value need.
    """
    if scale == 0:
        return str(unscaled)
    digits = str(abs(unscaled)).rjust(scale + 1, "0")
    sign = "-" if unscaled < 0 else ""
    precision = max(len(str(abs(unscaled))), scale)
    return f"CAST('{sign}{digits[:-scale]}.{digits[-scale:]}' AS DECIMAL({precision}, {scale}))"


def _decimal_digits(tree: _ExpressionTree, schema: Mapping[str, pl.DataType]) -> tuple[int, int] | None:
    """The integer digits and the scale the engine compares a decimal operand with, or None for any other operand.

    A cast that reaches the engine is a cast pushed as its bare column, so it compares with the column's digits.
    """
    cast = tree.get("Cast")
    inner = cast.get("expr") if isinstance(cast, dict) else tree
    column = inner.get("Column") if isinstance(inner, dict) else None
    if isinstance(column, str):
        dtype = schema.get(column)
        if isinstance(dtype, pl.Decimal) and dtype.precision is not None:
            return dtype.precision - dtype.scale, dtype.scale
        return None
    literal = tree.get("Literal")
    scalar = literal.get("Scalar") if isinstance(literal, dict) else None
    value = scalar.get("Decimal") if isinstance(scalar, dict) else None
    if isinstance(value, list) and isinstance(value[0], int) and isinstance(value[-1], int):
        unscaled, scale = value[0], value[-1]
        return max(len(str(abs(unscaled))) - scale, 0), scale
    return None


def _operand_to_sql(tree: _ExpressionTree, schema: Mapping[str, pl.DataType]) -> str:
    """An operand of a comparison, a text column compared by its bytes as polars compares it.

    The engine applies the column's collation or the session's default collation to a text comparison, and an ENUM
    compares as its text; binary collation keeps the scan filter and its pruning.
    """
    sql = _pl_tree_to_sql(tree, schema)
    column = tree.get("Column")
    dtype = schema.get(column) if isinstance(column, str) else None
    if dtype == pl.String:
        return f"({sql} COLLATE C)"
    if isinstance(dtype, pl.Categorical):
        return f"(CAST({sql} AS VARCHAR) COLLATE C)"
    return sql


def _comparison_to_sql(
    lhs: _ExpressionTree, op: str, rhs: _ExpressionTree, schema: Mapping[str, pl.DataType] | None
) -> str:
    """A comparison the engine evaluates as polars does.

    A column the schema leaves out has an engine type its polars dtype does not stand for. Two decimals whose common
    type needs more than 38 digits stay with polars until the engine compares them: it clamps that type to 38 digits
    and raises for a value that does not fit, where polars compares.
    """
    sql_op = _pl_operation_to_sql(op)
    if schema is None:
        return f"({_pl_tree_to_sql(lhs, schema)} {sql_op} {_pl_tree_to_sql(rhs, schema)})"
    for side in (lhs, rhs):
        column = side.get("Column")
        if isinstance(column, str) and column not in schema:
            msg = f"The engine does not compare column {column!r} as its polars dtype"
            raise NotImplementedError(msg)
    left, right = _decimal_digits(lhs, schema), _decimal_digits(rhs, schema)
    if left is not None and right is not None and max(left[0], right[0]) + max(left[1], right[1]) > 38:
        msg = "The engine raises on a decimal comparison whose common type needs more than 38 digits"
        raise NotImplementedError(msg)
    return f"({_operand_to_sql(lhs, schema)} {sql_op} {_operand_to_sql(rhs, schema)})"


def _predicate_to_sql(tree: _ExpressionTree, schema: Mapping[str, pl.DataType] | None = None) -> str:
    """Convert a Boolean polars expression tree, a filter predicate or a conjunct of one, to a SQL string.

    polars applies And and Or bitwise to integers but types them Boolean only between Boolean operands, and a filter
    predicate is Boolean, so an And or Or is SQL's AND or OR here and in its operands, and nowhere below.
    """
    bin_expr = tree.get("BinaryExpr")
    if isinstance(bin_expr, dict) and bin_expr.get("op") in ("And", "Or"):
        lhs, op, rhs = bin_expr["left"], bin_expr["op"], bin_expr["right"]
        assert isinstance(lhs, dict), f"LHS of a BinaryExpr should be a dict but got {type(lhs)}"
        assert isinstance(op, str), f"The op of a BinaryExpr should be a str but got {type(op)}"
        assert isinstance(rhs, dict), f"RHS of a BinaryExpr should be a dict but got {type(rhs)}"
        return f"({_predicate_to_sql(lhs, schema)} {op.upper()} {_predicate_to_sql(rhs, schema)})"
    return _pl_tree_to_sql(tree, schema)


def _pl_tree_to_sql(tree: _ExpressionTree, schema: Mapping[str, pl.DataType] | None = None) -> str:
    """Recursively convert a Polars expression tree (as JSON) to a SQL string.

    Parameters:
        tree (dict): JSON-deserialized expression tree from Polars
        schema (Mapping): the polars dtypes of the scanned columns the engine compares as polars does, which decide
            whether a cast of a column compares as the bare column, whether a modulus is an integer's and how a
            comparison renders; without it no cast of a column and no modulus translates

    Returns:
        str: SQL expression string

    Example:
        Input tree:
        {
            "BinaryExpr": {
                "left": { "Column": "foo" },
                "op": "Gt",
                "right": { "Literal": { "Int": 5 } }
            }
        }
        Output: "(foo > 5)"
    """
    [node_type] = tree.keys()

    if node_type == "BinaryExpr":
        # Binary expressions: left OP right
        bin_expr_tree = tree[node_type]
        assert isinstance(bin_expr_tree, dict), f"A {node_type} should be a dict but got {type(bin_expr_tree)}"
        lhs, op, rhs = bin_expr_tree["left"], bin_expr_tree["op"], bin_expr_tree["right"]
        assert isinstance(lhs, dict), f"LHS of a {node_type} should be a dict but got {type(lhs)}"
        assert isinstance(op, str), f"The op of a {node_type} should be a str but got {type(op)}"
        assert isinstance(rhs, dict), f"RHS of a {node_type} should be a dict but got {type(rhs)}"
        if op == "Modulus":
            return _floored_modulus_to_sql(lhs, rhs, schema)
        return _comparison_to_sql(lhs, op, rhs, schema)
    if node_type == "Column":
        # A reference to a column name
        # Wrap in quotes to handle special characters
        col_name = tree[node_type]
        assert isinstance(col_name, str), f"The col name of a {node_type} should be a str but got {type(col_name)}"
        return _escape_sql_identifier(col_name)

    if node_type in ("Literal", "Dyn"):
        # Recursively process dynamic or literal values
        val_tree = tree[node_type]
        assert isinstance(val_tree, dict), f"A {node_type} should be a dict but got {type(val_tree)}"
        return _pl_tree_to_sql(val_tree, schema)

    if node_type == "Int":
        # Direct integer literals
        int_literal = tree[node_type]
        assert isinstance(int_literal, (int, str)), (
            f"The value of an Int should be an int or str but got {type(int_literal)}"
        )
        return str(int_literal)

    if node_type == "Float":
        # Direct float literals
        float_literal = tree[node_type]
        if not isinstance(float_literal, (float, int, str)):
            # JSON has no inf/nan, polars serializes them as null
            msg = f"The value of a Float should be a float, int or str but got {type(float_literal)}"
            raise NotImplementedError(msg)
        return str(float_literal)

    if node_type == "Function":
        # Handle boolean functions like IsNull, IsNotNull
        func_tree = tree[node_type]
        assert isinstance(func_tree, dict), f"A {node_type} should be a dict but got {type(func_tree)}"
        inputs = func_tree["input"]
        assert isinstance(inputs, list), f"A {node_type} should have a list of dicts as input but got {type(inputs)}"
        input_tree = inputs[0]
        assert isinstance(input_tree, dict), (
            f"A {node_type} should have a list of dicts as input but got {type(input_tree)}"
        )
        func_dict = func_tree["function"]
        assert isinstance(func_dict, dict), (
            f"A {node_type} should have a function dict as input but got {type(func_dict)}"
        )

        if "Boolean" in func_dict:
            func = func_dict["Boolean"]
            arg_sql = _pl_tree_to_sql(inputs[0], schema)

            if func == "IsNull":
                return f"({arg_sql} IS NULL)"
            if func == "IsNotNull":
                return f"({arg_sql} IS NOT NULL)"
            msg = f"Boolean function not supported: {func}"
            raise NotImplementedError(msg)

        msg = f"Unsupported function type: {func_dict}"
        raise NotImplementedError(msg)

    if node_type == "Cast":
        cast_tree = tree[node_type]
        assert isinstance(cast_tree, dict), f"A {node_type} should be a dict but got {type(cast_tree)}"
        cast_expr = cast_tree["expr"]
        assert isinstance(cast_expr, dict), f"A {node_type} should be a dict but got {type(cast_expr)}"
        column = cast_expr.get("Column")
        if (
            schema is not None
            and isinstance(column, str)
            and _compares_bare(schema.get(column), cast_tree.get("dtype"))
        ):
            # No cast option changes how such a cast compares, so the bare column compares the same
            return _pl_tree_to_sql(cast_expr, schema)
        # Any other cast of a column can truncate, turn an overflow into NULL or change the type the comparison runs
        # in, and the bare column carries none of that
        if "Literal" not in cast_expr:
            msg = "A cast of a column or an expression cannot be pushed down"
            raise NotImplementedError(msg)
        options = cast_tree.get("options")
        if options not in ("Strict", "NonStrict"):
            msg = f"Only NonStrict/Strict casts can be safely unwrapped, got {options!r}"
            raise NotImplementedError(msg)
        # polars folds a cast of a literal before a scan sees it, so only unoptimized trees such as those
        # _predicate_to_expression builds reach this, and the literal renders without the cast's target type
        return _pl_tree_to_sql(cast_expr, schema)

    if node_type == "Scalar":
        # Detect format: old style (dtype/value) or new style (direct type key)
        scalar_tree = tree[node_type]
        assert isinstance(scalar_tree, dict), f"A {node_type} should be a dict but got {type(scalar_tree)}"
        if "dtype" in scalar_tree and "value" in scalar_tree:
            dtype = str(scalar_tree["dtype"])
            value = scalar_tree["value"]
        else:
            # New style: dtype is the single key in the dict
            dtype = next(iter(scalar_tree.keys()))
            value = scalar_tree
        assert isinstance(dtype, str), f"A {node_type} should have a str dtype but got  {type(dtype)}"
        assert isinstance(value, dict), f"A {node_type} should have a dict value but got {type(value)}"

        # Decimal support
        if dtype.startswith("{'Decimal'") or dtype == "Decimal":
            decimal_value = value["Decimal"]
            assert isinstance(decimal_value, list), (
                f"A {dtype} should be a two or three member list but got {type(decimal_value)}"
            )
            assert 2 <= len(decimal_value) <= 3, (
                f"A {dtype} should be a two or three member list but got {len(decimal_value)} member list"
            )
            unscaled, scale = decimal_value[0], decimal_value[-1]
            assert isinstance(unscaled, int), f"A {dtype} value should be an int but got {type(unscaled)}"
            assert isinstance(scale, int), f"A {dtype} scale should be an int but got {type(scale)}"
            return _decimal_to_sql(unscaled, scale)

        # Datetime: [value since epoch, time unit, time zone]
        if dtype.startswith("{'Datetime'") or dtype == "Datetime":
            datetime_value = value["Datetime"]
            assert isinstance(datetime_value, list), f"A {dtype} should be a list but got {type(datetime_value)}"
            epoch_value = datetime_value[0]
            time_unit = datetime_value[1] if len(datetime_value) > 1 else "Microseconds"
            time_zone = datetime_value[2] if len(datetime_value) > 2 else None
            assert isinstance(epoch_value, int), f"A {dtype} value should be an int but got {type(epoch_value)}"
            if time_unit == "Milliseconds":
                micros = epoch_value * 1_000
            elif time_unit == "Microseconds":
                micros = epoch_value
            else:
                # A TIMESTAMP literal cannot hold nanoseconds, let polars apply the filter
                msg = f"Unsupported datetime time unit {time_unit!r}"
                raise NotImplementedError(msg)
            dt_timestamp = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=micros)
            if time_zone is None:
                return f"'{dt_timestamp!s}'::TIMESTAMP"
            # The value is UTC; compare as an instant so the session TimeZone does not shift it
            return f"'{dt_timestamp!s}+00'::TIMESTAMPTZ"

        # Match simple numeric/boolean types
        if dtype in (
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Int128",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "UInt128",
            "Float32",
            "Float64",
            "Boolean",
        ):
            if value[dtype] is None:
                # JSON has no inf/nan, polars serializes them as null
                msg = f"Non-finite {dtype} literals cannot be pushed down"
                raise NotImplementedError(msg)
            return str(value[dtype])

        # Time type
        if dtype == "Time":
            nanoseconds = value["Time"]
            assert isinstance(nanoseconds, int), f"A {dtype} should be an int but got {type(nanoseconds)}"
            if nanoseconds % 1_000:
                # A TIME literal cannot hold nanoseconds, let polars apply the filter
                msg = "A Time literal finer than a microsecond cannot be pushed down"
                raise NotImplementedError(msg)
            seconds = nanoseconds // 1_000_000_000
            microseconds = (nanoseconds % 1_000_000_000) // 1_000
            dt_time = (datetime.datetime.min + datetime.timedelta(seconds=seconds, microseconds=microseconds)).time()
            return f"'{dt_time}'::TIME"

        # Date type
        if dtype == "Date":
            days_since_epoch = value["Date"]
            assert isinstance(days_since_epoch, (float, int)), (
                f"A {dtype} should be a number but got {type(days_since_epoch)}"
            )
            date = datetime.date(1970, 1, 1) + datetime.timedelta(days=days_since_epoch)
            return f"'{date}'::DATE"

        # Binary type
        if dtype == "Binary":
            bin_value = value["Binary"]
            assert isinstance(bin_value, list), f"A {dtype} should be a list but got {type(bin_value)}"
            binary_data = bytes(bin_value)
            escaped = "".join(f"\\x{b:02x}" for b in binary_data)
            return f"'{escaped}'::BLOB"

        # String type
        if dtype == "String" or dtype == "StringOwned":
            # Some new formats may store directly under StringOwned
            string_val = value.get("StringOwned", value.get("String", None))
            # the string must be a string constant
            return str(duckdb.ConstantExpression(string_val))

        msg = f"Unsupported scalar type {dtype!s}, with value {value}"
        raise NotImplementedError(msg)

    msg = f"Node type: {node_type} is not implemented. {tree[node_type]}"
    raise NotImplementedError(msg)


_ENGINE_TYPES = {
    "Boolean": ("BOOLEAN",),
    "Int8": ("TINYINT",),
    "Int16": ("SMALLINT",),
    "Int32": ("INTEGER",),
    "Int64": ("BIGINT",),
    "UInt8": ("UTINYINT",),
    "UInt16": ("USMALLINT",),
    "UInt32": ("UINTEGER",),
    "UInt64": ("UBIGINT",),
    "Float32": ("FLOAT",),
    "Float64": ("DOUBLE",),
    "String": ("VARCHAR",),
    "Binary": ("BLOB",),
    "Date": ("DATE",),
    "Time": ("TIME",),
}
_DATETIME_ENGINE_TYPES = {"ms": ("TIMESTAMP_S", "TIMESTAMP_MS"), "us": ("TIMESTAMP",), "ns": ("TIMESTAMP_NS",)}


def _stands_for(dtype: pl.DataType, engine_type: str) -> bool:
    """Whether the polars dtype of a column stands for the column's engine type, so both compare it alike.

    UUID and JSON reach polars as a String, BIT as a Binary, TIME_NS and TIME WITH TIME ZONE as a Time, and HUGEINT
    and UHUGEINT, which hold 39 digits, as a Decimal(38, 0); the engine compares each by its own type.
    """
    if isinstance(dtype, pl.Decimal):
        return engine_type == f"DECIMAL({dtype.precision},{dtype.scale})"
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            return engine_type == "TIMESTAMP WITH TIME ZONE"
        return engine_type in _DATETIME_ENGINE_TYPES.get(dtype.time_unit, ())
    if isinstance(dtype, pl.Categorical):
        return engine_type.startswith("ENUM(")
    return engine_type in _ENGINE_TYPES.get(str(dtype), ())


def duckdb_source(relation: duckdb.DuckDBPyRelation, schema: pl.schema.Schema) -> pl.LazyFrame:
    """A polars IO plugin for DuckDB."""
    engine_types = dict(zip(relation.columns, map(str, relation.types), strict=True))
    comparable = {name: dtype for name, dtype in schema.items() if _stands_for(dtype, engine_types.get(name, ""))}

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        duck_predicate = None
        fallback_predicate = None
        relation_final = relation
        if with_columns is not None:
            cols = ",".join(map(_escape_sql_identifier, with_columns))
            relation_final = relation_final.project(cols)
        if n_rows is not None:
            relation_final = relation_final.limit(n_rows)
        if predicate is not None:
            # Strip any dynamic-predicate hint (see the dynamic-predicate note
            # above); the real predicate must still be applied.
            tree = json.loads(predicate.meta.serialize(format="json"))
            real_tree, had_dynamic = _strip_dynamic_predicates(tree)
            if real_tree is not None:
                # A row the predicate keeps satisfies every conjunct, so the engine may filter by the conjuncts that
                # translate as long as polars applies the whole predicate when one does not.
                conjuncts = _conjuncts(real_tree)
                pushed = [
                    expression
                    for expression in (_tree_to_sql_expression(conjunct, comparable) for conjunct in conjuncts)
                    if expression is not None
                ]
                if pushed:
                    duck_predicate = functools.reduce(operator.and_, pushed)
                if len(pushed) < len(conjuncts):
                    # Rebuild the expression from the stripped tree so we never hand polars the dynamic node it
                    # cannot lower.
                    fallback_predicate = _expression_from_tree(real_tree) if had_dynamic else predicate
        # Try to pushdown filter, if one exists
        if duck_predicate is not None:
            relation_final = relation_final.filter(duck_predicate)
        results = relation_final.to_arrow_reader() if batch_size is None else relation_final.to_arrow_reader(batch_size)

        # The reader holds the connection until it is read to the end or closed, so a source polars stops reading
        # early must close it, or the connection stays refused
        try:
            for record_batch in iter(results.read_next_batch, None):
                if fallback_predicate is not None:
                    # We have a predicate, but did not manage to push it down, we fallback here
                    yield pl.from_arrow(record_batch).filter(fallback_predicate)  # type: ignore[arg-type,misc,unused-ignore]
                else:
                    yield pl.from_arrow(record_batch)  # type: ignore[misc,unused-ignore]
        finally:
            results.close()

    return register_io_source(source_generator, schema=schema, is_pure=True)

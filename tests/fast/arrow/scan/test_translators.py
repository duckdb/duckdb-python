"""The predicate translators, evaluated on small pyarrow tables and polars frames.

Each case lists the row ids the engine keeps for the predicate, worked out by hand with its semantics: NULL is
three valued, NaN orders above every number and equals only itself, -0.0 equals 0.0, and a constant between two
points of a coarser column equals no row. A translator that cannot match the engine must refuse.
"""

from __future__ import annotations

import datetime
import decimal
import math
import uuid

import pytest

from duckdb._sources import to_arrow as to_arrow_module
from duckdb._sources.nodes import Binary, Col, In, Lit, Postfix, Unary, Untranslatable, fragment

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")

D = decimal.Decimal
NAN = float("nan")
NEW_YORK = "America/New_York"
# 2024-05-01 12:00:00 UTC in seconds, and half a second after it in microseconds.
BASE = 1714564800
HALF = BASE * 1_000_000 + 500_000
# Deeper than Python's recursion limit, as a left folded OR of that many predicates reaches.
DEEP = 1500
DICTIONARY_KERNELS = pytest.mark.skipif(
    int(pa.__version__.split(".", 1)[0]) < to_arrow_module.DICTIONARY_KERNELS_FIXED_IN,
    reason="pyarrow before 26 reads a valid index that points at a NULL dictionary entry as a value",
)
ENCODED = [pytest.param(False, id="plain"), pytest.param(True, id="dictionary", marks=DICTIONARY_KERNELS)]


def col(*parts):
    return Col(tuple(parts))


def cmp(op, name, value, type_text):
    return Binary(op, col(*name.split(".")), Lit(value, type_text))


def isin(name, type_text, *values):
    return In(col(*name.split(".")), [Lit(value, type_text) for value in values])


def both(left, right):
    return Binary("AND", left, right)


def either(left, right):
    return Binary("OR", left, right)


def negate(node):
    return Unary("NOT", node)


def null(name):
    return Postfix("IS NULL", col(*name.split(".")))


def not_null(name):
    return Postfix("IS NOT NULL", col(*name.split(".")))


def arrow_table(values, kind):
    return pa.table({"id": list(range(len(values))), "c": pa.array(values, kind)})


def arrow_ids(table, node):
    from duckdb._sources.to_arrow import to_arrow

    expression = to_arrow(node, table.schema)
    return sorted(ds.dataset(table).to_table(filter=expression).column("id").to_pylist())


def refused_by_arrow(schema, node):
    from duckdb._sources.to_arrow import to_arrow

    with pytest.raises(Untranslatable):
        to_arrow(node, schema)


INT_VALUES = [1, 5, 9, None]

SHAPES = [
    pytest.param(cmp("=", "c", 5, "INTEGER"), [1], id="equal"),
    pytest.param(cmp("!=", "c", 5, "INTEGER"), [0, 2], id="not-equal"),
    pytest.param(cmp("<", "c", 5, "INTEGER"), [0], id="less"),
    pytest.param(cmp("<=", "c", 5, "INTEGER"), [0, 1], id="less-or-equal"),
    pytest.param(cmp(">", "c", 5, "INTEGER"), [2], id="greater"),
    pytest.param(cmp(">=", "c", 5, "INTEGER"), [1, 2], id="greater-or-equal"),
    pytest.param(both(cmp(">", "c", 1, "INTEGER"), cmp("<", "c", 9, "INTEGER")), [1], id="and"),
    pytest.param(either(cmp("=", "c", 1, "INTEGER"), cmp("=", "c", 9, "INTEGER")), [0, 2], id="or"),
    pytest.param(negate(cmp("=", "c", 5, "INTEGER")), [0, 2], id="not"),
    pytest.param(negate(both(cmp(">", "c", 1, "INTEGER"), cmp("<", "c", 9, "INTEGER"))), [0, 2], id="not-and"),
    pytest.param(negate(either(cmp("=", "c", 1, "INTEGER"), null("c"))), [1, 2], id="not-or-is-null"),
    pytest.param(null("c"), [3], id="is-null"),
    pytest.param(not_null("c"), [0, 1, 2], id="is-not-null"),
    pytest.param(isin("c", "INTEGER", 1, 9), [0, 2], id="in"),
    pytest.param(negate(isin("c", "INTEGER", 1, 9)), [1], id="not-in-keeps-null-out"),
    pytest.param(both(cmp(">=", "c", 1, "INTEGER"), cmp("<=", "c", 5, "INTEGER")), [0, 1], id="between"),
]

INTEGERS = [
    ("int8", "TINYINT", -(1 << 7), (1 << 7) - 1),
    ("int16", "SMALLINT", -(1 << 15), (1 << 15) - 1),
    ("int32", "INTEGER", -(1 << 31), (1 << 31) - 1),
    ("int64", "BIGINT", -(1 << 63), (1 << 63) - 1),
    ("uint8", "UTINYINT", 0, (1 << 8) - 1),
    ("uint16", "USMALLINT", 0, (1 << 16) - 1),
    ("uint32", "UINTEGER", 0, (1 << 32) - 1),
    ("uint64", "UBIGINT", 0, (1 << 64) - 1),
]

FLOAT_VALUES = [-0.0, 0.0, 1.5, NAN, None, -2.5]

FLOAT_CASES = [
    pytest.param(">", 1.0, [2, 3], id="greater-keeps-nan"),
    pytest.param(">=", 1.5, [2, 3], id="greater-or-equal-keeps-nan"),
    pytest.param(">", -0.0, [2, 3], id="greater-than-negative-zero"),
    pytest.param("=", -0.0, [0, 1], id="negative-zero-equals-zero"),
    pytest.param("<", -0.0, [5], id="less-than-negative-zero"),
    pytest.param("<=", -0.0, [0, 1, 5], id="less-or-equal-negative-zero"),
    pytest.param("<", 1.5, [0, 1, 5], id="less-drops-nan"),
    pytest.param("<=", 1.5, [0, 1, 2, 5], id="less-or-equal-drops-nan"),
    pytest.param("=", 0.0, [0, 1], id="zero-equals-negative-zero"),
    pytest.param("!=", 0.0, [2, 3, 5], id="not-equal-keeps-nan"),
]

FLOAT_IN_CASES = [
    pytest.param(isin("c", "{t}", 0.0, 1.5), [0, 1, 2], id="in-zero-matches-negative-zero"),
    pytest.param(isin("c", "{t}", -0.0), [0, 1], id="in-negative-zero-matches-zero"),
    pytest.param(isin("c", "{t}", -0.0, 1.5), [0, 1, 2], id="in-negative-zero-and-another"),
    pytest.param(negate(isin("c", "{t}", -0.0, 1.5)), [3, 5], id="not-in-negative-zero-and-another"),
    pytest.param(isin("c", "{t}", 0.0), [0, 1], id="in-zero-alone"),
    pytest.param(isin("c", "{t}", 1.5, -2.5), [2, 5], id="in-without-zero"),
    pytest.param(negate(isin("c", "{t}", 0.0)), [2, 3, 5], id="not-in-zero-keeps-nan"),
    pytest.param(negate(cmp(">", "c", 1.0, "{t}")), [0, 1, 5], id="not-greater-drops-nan"),
]


def typed(node, type_text):
    """`node` with every "{t}" constant type replaced by `type_text`."""
    match node:
        case Binary(op=op, left=left, right=right):
            return Binary(op, typed(left, type_text), typed(right, type_text))
        case Unary(op=op, operand=operand):
            return Unary(op, typed(operand, type_text))
        case In(operand=operand, values=values):
            return In(operand, [typed(value, type_text) for value in values])
        case Lit(value=value, type=kind):
            return Lit(value, type_text if kind == "{t}" else kind)
    return node


def dictionary(values, kind):
    """A dictionary array holding `values`, its NULLs taken in turn from a NULL dictionary entry and a NULL index."""
    # Entries are told apart by their text, so -0.0 and 0.0 stay two entries.
    entries = list({repr(value): value for value in values if value is not None}.values())
    lookup = {repr(value): i for i, value in enumerate(entries)}
    indices = []
    nulls = 0
    for value in values:
        if value is None:
            indices.append(len(entries) if nulls % 2 == 0 else None)
            nulls += 1
        else:
            indices.append(lookup[repr(value)])
    return pa.DictionaryArray.from_arrays(pa.array(indices, pa.int32()), pa.array([*entries, None], kind))


class TestArrowShapes:
    @pytest.mark.parametrize(("node", "expected"), SHAPES)
    def test_every_node_shape(self, node, expected):
        assert arrow_ids(arrow_table(INT_VALUES, pa.int32()), node) == expected

    @pytest.mark.parametrize(
        "node",
        [
            pytest.param(Binary(">", Lit(1, "INTEGER"), col("c")), id="constant-on-the-left"),
            pytest.param(Binary("=", col("c"), col("c")), id="column-with-column"),
            pytest.param(Binary("LIKE", col("c"), Lit(1, "INTEGER")), id="unknown-operator"),
            pytest.param(Unary("-", cmp("=", "c", 1, "INTEGER")), id="unknown-prefix"),
            pytest.param(negate(col("c")), id="not-of-a-column"),
            pytest.param(Postfix("IS NULL", Lit(1, "INTEGER")), id="is-null-of-a-constant"),
            pytest.param(In(col("c"), [col("c")]), id="in-a-column"),
            pytest.param(In(col("c"), [Lit(1, "INTEGER"), Lit(None, "INTEGER")]), id="in-with-null"),
            pytest.param(In(col("c"), []), id="empty-in"),
            pytest.param(negate(In(col("c"), [])), id="not-empty-in"),
            pytest.param(cmp("=", "c", None, "INTEGER"), id="null-constant"),
            pytest.param(cmp("=", "missing", 1, "INTEGER"), id="missing-column"),
            pytest.param(Binary("=", Col(()), Lit(1, "INTEGER")), id="no-name"),
            pytest.param(both(cmp("=", "c", 1, "INTEGER"), cmp("=", "c", 1, "BIGINT")), id="and-with-a-refused-side"),
            pytest.param(
                either(cmp("=", "c", 1, "INTEGER"), cmp("=", "c", 1.5, "DOUBLE")), id="or-with-a-refused-side"
            ),
        ],
    )
    def test_a_shape_it_cannot_evaluate_is_refused(self, node):
        refused_by_arrow(arrow_table(INT_VALUES, pa.int32()).schema, node)

    def test_a_repeated_column_name_is_refused(self):
        table = pa.table([pa.array([1]), pa.array([2])], names=["c", "c"])
        refused_by_arrow(table.schema, cmp("=", "c", 1, "BIGINT"))


def chain(depth):
    node = cmp("=", "c", 0, "INTEGER")
    for i in range(1, depth):
        node = either(node, cmp("=", "c", i, "INTEGER"))
    return node


class TestDepth:
    def test_a_chain_deeper_than_python_recursion_is_refused(self, pl):
        from duckdb._sources.to_polars import to_polars

        refused_by_arrow(pa.schema([("c", pa.int32())]), chain(DEEP))
        with pytest.raises(Untranslatable):
            to_polars(chain(DEEP), pl.Schema({"c": pl.Int32}))

    def test_a_long_chain_within_it_translates(self, pl):
        assert arrow_ids(arrow_table(INT_VALUES, pa.int32()), chain(200)) == [0, 1, 2]
        assert polars_ids(polars_frame(pl, INT_VALUES, pl.Int32), chain(200)) == [0, 1, 2]


class TestSelectorNames:
    """Polars reads ^x$ and * as selectors in a column name, so columns are reached without their names."""

    ROWS = (
        (0, 0, 2, 5, {"^x$": 0, "x": 2, "*": 7}),
        (1, 2, 0, 6, {"^x$": 2, "x": 0, "*": None}),
        (2, None, None, None, None),
    )

    def native_ids(self, predicate):
        import duckdb

        with duckdb.connect() as con:
            con.execute(
                'CREATE TABLE t (id INTEGER, "^x$" BIGINT, x BIGINT, "*" BIGINT, '
                's STRUCT("^x$" BIGINT, x BIGINT, "*" BIGINT))'
            )
            con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?)", [list(row) for row in self.ROWS])
            return [row[0] for row in con.execute(f"SELECT id FROM t WHERE {predicate} ORDER BY id").fetchall()]

    def frame(self, pl):
        columns = list(zip(*self.ROWS, strict=True))
        return pl.DataFrame(
            {
                "id": list(columns[0]),
                "^x$": pl.Series(columns[1], dtype=pl.Int64),
                "x": pl.Series(columns[2], dtype=pl.Int64),
                "*": pl.Series(columns[3], dtype=pl.Int64),
                "s": list(columns[4]),
            }
        )

    @pytest.mark.parametrize(
        ("node", "predicate"),
        [
            pytest.param(cmp(">", "^x$", 1, "BIGINT"), '"^x$" > 1', id="regex-name"),
            pytest.param(cmp("<", "^x$", 1, "BIGINT"), '"^x$" < 1', id="regex-name-less"),
            pytest.param(cmp("=", "*", 6, "BIGINT"), '"*" = 6', id="wildcard-name"),
            pytest.param(isin("^x$", "BIGINT", 2), '"^x$" IN (2)', id="regex-name-in"),
            pytest.param(null("*"), '"*" IS NULL', id="wildcard-name-is-null"),
            pytest.param(cmp(">", "s.x", 1, "BIGINT"), "s.x > 1", id="plain-field-beside-a-regex-field"),
        ],
    )
    def test_rows_match_native_duckdb(self, pl, node, predicate):
        expected = self.native_ids(predicate)
        assert polars_ids(self.frame(pl), node) == expected
        table = self.frame(pl).to_arrow()
        assert arrow_ids(table, node) == expected

    @pytest.mark.parametrize(
        ("node", "predicate"),
        [
            pytest.param(Binary(">", Col(("s", "^x$")), Lit(1, "BIGINT")), 's."^x$" > 1', id="regex-field"),
            pytest.param(Postfix("IS NULL", Col(("s", "^x$"))), 's."^x$" IS NULL', id="regex-field-is-null"),
            pytest.param(Binary(">", Col(("s", "*")), Lit(6, "BIGINT")), 's."*" > 6', id="wildcard-field"),
            pytest.param(Postfix("IS NULL", Col(("s", "*"))), 's."*" IS NULL', id="wildcard-field-is-null"),
        ],
    )
    def test_a_struct_field_named_like_a_selector(self, pl, node, predicate):
        """Polars has no positional struct access it can plan, so it refuses; pyarrow names fields literally."""
        refused_by_polars(self.frame(pl).schema, node)
        assert arrow_ids(self.frame(pl).to_arrow(), node) == self.native_ids(predicate)


class TestPolarsCategories:
    """The engine reads a Categorical or Enum as VARCHAR; polars matches strings exactly but orders by category."""

    VALUES = ("a", "b", None, "a")

    def native_ids(self, predicate):
        import duckdb

        with duckdb.connect() as con:
            con.execute("CREATE TABLE t (id INTEGER, c VARCHAR)")
            con.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(self.VALUES)))
            return [row[0] for row in con.execute(f"SELECT id FROM t WHERE {predicate} ORDER BY id").fetchall()]

    def frame(self, pl, dtype):
        kinds = {"categorical": pl.Categorical(), "enum": pl.Enum(["b", "a", "unused"])}
        return pl.DataFrame({"id": [0, 1, 2, 3], "c": pl.Series(list(self.VALUES), dtype=kinds[dtype])})

    @pytest.mark.parametrize("dtype", ["categorical", "enum"])
    @pytest.mark.parametrize(
        ("node", "predicate", "expected"),
        [
            pytest.param(cmp("=", "c", "a", "VARCHAR"), "c = 'a'", [0, 3], id="equal"),
            pytest.param(cmp("!=", "c", "a", "VARCHAR"), "c <> 'a'", [1], id="not-equal"),
            pytest.param(cmp("=", "c", "z", "VARCHAR"), "c = 'z'", [], id="equal-to-a-missing-category"),
            pytest.param(cmp("!=", "c", "z", "VARCHAR"), "c <> 'z'", [0, 1, 3], id="not-equal-to-a-missing-category"),
            pytest.param(cmp("=", "c", "unused", "VARCHAR"), "c = 'unused'", [], id="equal-to-an-unused-category"),
            pytest.param(isin("c", "VARCHAR", "a", "z"), "c IN ('a', 'z')", [0, 3], id="in"),
            pytest.param(negate(isin("c", "VARCHAR", "a")), "NOT (c IN ('a'))", [1], id="not-in"),
            pytest.param(negate(isin("c", "VARCHAR", "z")), "NOT (c IN ('z'))", [0, 1, 3], id="not-in-missing"),
            pytest.param(null("c"), "c IS NULL", [2], id="is-null"),
        ],
    )
    def test_equality_and_membership_match_native_duckdb(self, pl, dtype, node, predicate, expected):
        assert self.native_ids(predicate) == expected
        assert polars_ids(self.frame(pl, dtype), node) == expected

    @pytest.mark.parametrize("dtype", ["categorical", "enum"])
    @pytest.mark.parametrize(
        ("op", "expected"), [("<", []), ("<=", [0, 3]), (">", [1]), (">=", [0, 1, 3])], ids=["<", "<=", ">", ">="]
    )
    def test_ordering_is_refused_and_left_to_the_engine(self, pl, dtype, op, expected):
        assert self.native_ids(f"c {op} 'a'") == expected
        refused_by_polars(self.frame(pl, dtype).schema, cmp(op, "c", "a", "VARCHAR"))

    def test_a_constant_of_another_type_is_refused(self, pl):
        refused_by_polars(self.frame(pl, "enum").schema, cmp("=", "c", b"a", "BLOB"))
        refused_by_polars(self.frame(pl, "categorical").schema, isin("c", "INTEGER", 1))


class TestDictionaryKernels:
    """A Dataset takes a predicate on a dictionary column only where pyarrow evaluates it exactly."""

    @pytest.mark.parametrize(("fixed_in", "accepted"), [(0, True), (10_000, False)], ids=["fixed", "defective"])
    def test_the_dataset_source_follows_the_pyarrow_version(self, monkeypatch, fixed_in, accepted):
        from duckdb._sources import adapt

        monkeypatch.setattr(to_arrow_module, "DICTIONARY_KERNELS_FIXED_IN", fixed_in)
        table = pa.table({"id": [0, 1, 2], "d": pa.array(["a", "b", "a"]).dictionary_encode(), "n": [1, 2, 3]})
        source = adapt(ds.dataset(table))
        assert source.accepts(cmp("=", "d", "a", "VARCHAR")) is accepted
        assert source.accepts(null("d")) is accepted
        assert source.accepts(isin("d", "VARCHAR", "b")) is accepted
        assert source.accepts(cmp(">", "n", 1, "BIGINT")) is True
        if accepted:
            stream, _ = source.stream(None, (cmp("=", "d", "a", "VARCHAR"),))
            reader = pa.RecordBatchReader._import_from_c_capsule(stream)
            assert reader.read_all().column("id").to_pylist() == [0, 2]


class TestArrowIntegers:
    @pytest.mark.parametrize(("arrow_name", "engine_name", "low", "high"), INTEGERS)
    def test_extremes(self, arrow_name, engine_name, low, high):
        table = arrow_table([low, 1, high, None], getattr(pa, arrow_name)())
        assert arrow_ids(table, cmp("=", "c", high, engine_name)) == [2]
        assert arrow_ids(table, cmp(">=", "c", low, engine_name)) == [0, 1, 2]
        assert arrow_ids(table, cmp("<", "c", high, engine_name)) == [0, 1]
        assert arrow_ids(table, isin("c", engine_name, low, high)) == [0, 2]
        assert arrow_ids(table, negate(isin("c", engine_name, high))) == [0, 1]
        refused_by_arrow(table.schema, cmp("=", "c", high + 1, engine_name))
        refused_by_arrow(table.schema, cmp("=", "c", low - 1, engine_name))

    def test_a_constant_of_another_width_is_refused(self):
        refused_by_arrow(arrow_table([1], pa.int16()).schema, cmp("=", "c", 1, "TINYINT"))

    def test_a_bool_is_not_an_integer(self):
        refused_by_arrow(arrow_table([1], pa.int32()).schema, cmp("=", "c", True, "INTEGER"))

    @pytest.mark.parametrize(
        "constant",
        [
            pytest.param(Lit(1, "HUGEINT"), id="hugeint"),
            pytest.param(Lit(uuid.UUID(int=1), "UUID"), id="uuid"),
            pytest.param(Lit("a", "ENUM('a')"), id="enum"),
        ],
    )
    def test_a_type_without_translation_is_refused(self, constant):
        refused_by_arrow(arrow_table([1], pa.int64()).schema, Binary("=", col("c"), constant))


class TestArrowFloats:
    @pytest.mark.parametrize(("kind", "type_text"), [(pa.float64(), "DOUBLE"), (pa.float32(), "FLOAT")])
    @pytest.mark.parametrize("encoded", ENCODED)
    @pytest.mark.parametrize(("op", "value", "expected"), FLOAT_CASES)
    def test_comparisons_order_nan_last_and_equate_zeros(self, kind, type_text, encoded, op, value, expected):
        array = dictionary(FLOAT_VALUES, kind) if encoded else pa.array(FLOAT_VALUES, kind)
        table = pa.table({"id": list(range(len(FLOAT_VALUES))), "c": array})
        assert arrow_ids(table, cmp(op, "c", value, type_text)) == expected

    @pytest.mark.parametrize(("kind", "type_text"), [(pa.float64(), "DOUBLE"), (pa.float32(), "FLOAT")])
    @pytest.mark.parametrize("encoded", ENCODED)
    @pytest.mark.parametrize(("node", "expected"), FLOAT_IN_CASES)
    def test_membership_equates_zeros(self, kind, type_text, encoded, node, expected):
        array = dictionary(FLOAT_VALUES, kind) if encoded else pa.array(FLOAT_VALUES, kind)
        table = pa.table({"id": list(range(len(FLOAT_VALUES))), "c": array})
        assert arrow_ids(table, typed(node, type_text)) == expected

    def test_a_float_constant_compares_in_its_own_width(self):
        table = arrow_table([0.1, 0.5, None], pa.float32())
        assert arrow_ids(table, cmp("=", "c", 0.10000000149011612, "FLOAT")) == [0]
        assert arrow_ids(table, isin("c", "FLOAT", 0.10000000149011612)) == [0]

    @pytest.mark.parametrize(
        ("kind", "constant"),
        [
            pytest.param(pa.float32(), Lit(0.1, "FLOAT"), id="float-not-held-by-32-bits"),
            pytest.param(pa.float32(), Lit(1e300, "FLOAT"), id="float-out-of-range"),
            pytest.param(pa.float64(), Lit(NAN, "DOUBLE"), id="nan"),
            pytest.param(pa.float64(), Lit(math.inf, "DOUBLE"), id="infinity"),
            pytest.param(pa.float64(), Lit(1, "DOUBLE"), id="an-int-for-a-double"),
            pytest.param(pa.float64(), Lit(1.5, "FLOAT"), id="float-for-a-double-column"),
            pytest.param(pa.float16(), Lit(1.5, "FLOAT"), id="half-float-column"),
        ],
    )
    def test_a_constant_that_is_not_exact_is_refused(self, kind, constant):
        refused_by_arrow(pa.schema([("c", kind)]), Binary(">", col("c"), constant))
        refused_by_arrow(pa.schema([("c", kind)]), In(col("c"), [constant]))


class TestArrowStrings:
    VALUES = ("a", "b", None, "c", None, "it's")

    @pytest.mark.parametrize("kind", [pa.string(), pa.large_string()])
    @pytest.mark.parametrize("encoded", ENCODED)
    def test_comparisons_membership_and_nulls(self, kind, encoded):
        array = dictionary(self.VALUES, kind) if encoded else pa.array(self.VALUES, kind)
        table = pa.table({"id": list(range(len(self.VALUES))), "c": array})
        assert arrow_ids(table, cmp("=", "c", "b", "VARCHAR")) == [1]
        assert arrow_ids(table, cmp(">", "c", "a", "VARCHAR")) == [1, 3, 5]
        assert arrow_ids(table, cmp("=", "c", "it's", "VARCHAR")) == [5]
        assert arrow_ids(table, isin("c", "VARCHAR", "a", "c")) == [0, 3]
        assert arrow_ids(table, negate(isin("c", "VARCHAR", "a", "c"))) == [1, 5]
        assert arrow_ids(table, null("c")) == [2, 4]
        assert arrow_ids(table, not_null("c")) == [0, 1, 3, 5]

    def test_a_blob_constant_on_a_string_column_is_refused(self):
        refused_by_arrow(arrow_table(["a"], pa.string()).schema, cmp("=", "c", b"a", "BLOB"))


class TestArrowBlobs:
    VALUES = (b"aa", b"bb", None)

    @pytest.mark.parametrize("kind", [pa.binary(), pa.large_binary(), pa.binary(2)])
    @pytest.mark.parametrize("encoded", ENCODED)
    def test_comparisons_and_membership(self, kind, encoded):
        array = dictionary(self.VALUES, kind) if encoded else pa.array(self.VALUES, kind)
        table = pa.table({"id": list(range(len(self.VALUES))), "c": array})
        assert arrow_ids(table, cmp("=", "c", b"bb", "BLOB")) == [1]
        assert arrow_ids(table, cmp(">", "c", b"a", "BLOB")) == [0, 1]
        assert arrow_ids(table, cmp("<", "c", b"bbb", "BLOB")) == [0, 1]
        # A candidate of another length is held by no row of a fixed size column.
        assert arrow_ids(table, isin("c", "BLOB", b"a", b"bb", b"bbb")) == [1]
        assert arrow_ids(table, negate(isin("c", "BLOB", b"a"))) == [0, 1]


class TestArrowTemporal:
    def test_dates(self):
        table = arrow_table([datetime.date(2020, 1, 1), datetime.date(2021, 6, 1), None], pa.date32())
        assert arrow_ids(table, cmp(">=", "c", datetime.date(2021, 1, 1), "DATE")) == [1]
        assert arrow_ids(table, isin("c", "DATE", datetime.date(2020, 1, 1))) == [0]
        refused_by_arrow(table.schema, cmp(">", "c", datetime.datetime(2021, 1, 1), "DATE"))

    def test_a_date64_column_takes_no_constant(self):
        # Half a day past the epoch, which the engine reads as 1970-01-01 and pyarrow compares in milliseconds.
        table = pa.table({"id": [0, 1], "c": pa.array([43_200_000, None], pa.date64())})
        refused_by_arrow(table.schema, cmp("=", "c", datetime.date(1970, 1, 1), "DATE"))
        refused_by_arrow(table.schema, isin("c", "DATE", datetime.date(1970, 1, 1)))
        assert arrow_ids(table, null("c")) == [1]

    @pytest.mark.parametrize("kind", [pa.time32("s"), pa.time32("ms"), pa.time64("us")])
    def test_a_time_between_two_points_of_the_column_equals_no_row(self, kind):
        table = arrow_table([datetime.time(0, 1), datetime.time(0, 1, 1), None], kind)
        between = datetime.time(0, 1, 0, 500)
        assert arrow_ids(table, cmp(">", "c", between, "TIME")) == [1]
        assert arrow_ids(table, cmp("<", "c", between, "TIME")) == [0]
        assert arrow_ids(table, cmp("=", "c", between, "TIME")) == []
        assert arrow_ids(table, isin("c", "TIME", between, datetime.time(0, 1, 1))) == [1]
        assert arrow_ids(table, isin("c", "TIME", datetime.time(0, 1, 1, 1))) == []
        assert arrow_ids(table, negate(isin("c", "TIME", between))) == [0, 1]

    def test_nanosecond_time_is_refused(self):
        refused_by_arrow(arrow_table([1], pa.time64("ns")).schema, cmp(">", "c", datetime.time(1), "TIME"))

    @pytest.mark.parametrize(("unit", "type_text"), [("s", "TIMESTAMP_S"), ("ms", "TIMESTAMP_MS"), ("us", "TIMESTAMP")])
    def test_naive_timestamps(self, unit, type_text):
        moments = [datetime.datetime(2024, 5, 1, 12), datetime.datetime(2024, 5, 1, 12, 0, 1), None]
        table = arrow_table(moments, pa.timestamp(unit))
        assert arrow_ids(table, cmp(">", "c", moments[0], type_text)) == [1]
        assert arrow_ids(table, cmp("=", "c", moments[1], type_text)) == [1]
        assert arrow_ids(table, isin("c", type_text, moments[0])) == [0]
        assert arrow_ids(table, negate(isin("c", type_text, moments[0]))) == [1]

    @pytest.mark.parametrize(
        ("kind", "constant"),
        [
            pytest.param(pa.timestamp("s"), Lit(datetime.datetime(2024, 1, 1, 0, 0, 0, 5), "TIMESTAMP_S"), id="s-off"),
            pytest.param(
                pa.timestamp("ms"), Lit(datetime.datetime(2024, 1, 1, 0, 0, 0, 5), "TIMESTAMP_MS"), id="ms-off"
            ),
            pytest.param(pa.timestamp("s"), Lit(datetime.datetime(2024, 1, 1), "TIMESTAMP"), id="another-unit"),
            pytest.param(pa.timestamp("ns"), Lit(datetime.datetime(2024, 1, 1), "TIMESTAMP"), id="nanoseconds"),
            pytest.param(pa.timestamp("us"), Lit(BASE * 1_000_000, "TIMESTAMP WITH TIME ZONE"), id="zoned-on-naive"),
            pytest.param(
                pa.timestamp("us"),
                Lit(datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc), "TIMESTAMP"),
                id="aware-value",
            ),
        ],
    )
    def test_a_timestamp_that_is_not_exact_is_refused(self, kind, constant):
        refused_by_arrow(arrow_table([0], kind).schema, Binary(">", col("c"), constant))

    @pytest.mark.parametrize("zone", [NEW_YORK, "UTC", "+05:30"])
    @pytest.mark.parametrize("unit", ["s", "ms", "us"])
    @pytest.mark.parametrize("encoded", ENCODED)
    def test_a_zoned_constant_keeps_its_microseconds(self, zone, unit, encoded):
        per_second = {"s": 1, "ms": 1_000, "us": 1_000_000}[unit]
        counts = [BASE * per_second, (BASE + 1) * per_second, None]
        kind = pa.timestamp(unit, tz=zone)
        array = dictionary(counts, kind) if encoded else pa.array(counts, kind)
        table = pa.table({"id": [0, 1, 2], "c": array})
        zoned = "TIMESTAMP WITH TIME ZONE"
        on_grid = (BASE + 1) * 1_000_000
        assert arrow_ids(table, cmp(">=", "c", HALF, zoned)) == [1]
        assert arrow_ids(table, cmp("<", "c", HALF, zoned)) == [0]
        assert arrow_ids(table, cmp("=", "c", HALF, zoned)) == []
        assert arrow_ids(table, cmp("=", "c", on_grid, zoned)) == [1]
        assert arrow_ids(table, isin("c", zoned, HALF, on_grid)) == [1]
        assert arrow_ids(table, negate(isin("c", zoned, HALF))) == [0, 1]
        # One microsecond after a row is no row, whatever the column's unit.
        assert arrow_ids(table, cmp("=", "c", on_grid + 1, zoned)) == []
        assert arrow_ids(table, cmp("<", "c", on_grid + 1, zoned)) == [0, 1]
        assert arrow_ids(table, isin("c", zoned, on_grid + 1)) == []

    def test_a_nanosecond_zoned_column_is_refused(self):
        kind = pa.timestamp("ns", tz="UTC")
        refused_by_arrow(arrow_table([0], kind).schema, cmp(">", "c", HALF, "TIMESTAMP WITH TIME ZONE"))


class TestArrowDecimals:
    VALUES = (D("1.25"), D("2.50"), None)

    @pytest.mark.parametrize("encoded", ENCODED)
    @pytest.mark.parametrize(("precision", "scale"), [(10, 2), (30, 2), (38, 10)])
    def test_comparisons_and_membership(self, encoded, precision, scale):
        kind = pa.decimal128(precision, scale)
        array = dictionary(self.VALUES, kind) if encoded else pa.array(self.VALUES, kind)
        table = pa.table({"id": [0, 1, 2], "c": array})
        type_text = f"DECIMAL({precision},{scale})"
        assert arrow_ids(table, cmp(">", "c", D("1.25"), type_text)) == [1]
        assert arrow_ids(table, cmp("=", "c", D("2.5"), type_text)) == [1]
        assert arrow_ids(table, isin("c", type_text, D("2.5000"))) == [1]
        assert arrow_ids(table, negate(isin("c", type_text, D("1.25")))) == [1]

    @pytest.mark.parametrize(
        "constant",
        [
            pytest.param(Lit(D("1.25"), "DECIMAL(10,3)"), id="another-scale"),
            pytest.param(Lit(D("1.25"), "DECIMAL(12,2)"), id="another-width"),
            pytest.param(Lit(D("1.255"), "DECIMAL(10,2)"), id="digits-beyond-the-scale"),
            pytest.param(Lit(D("123456789.00"), "DECIMAL(10,2)"), id="digits-beyond-the-width"),
            pytest.param(Lit(D("NaN"), "DECIMAL(10,2)"), id="nan"),
            pytest.param(Lit(1.25, "DECIMAL(10,2)"), id="a-float-for-a-decimal"),
        ],
    )
    def test_a_decimal_that_is_not_exact_is_refused(self, constant):
        refused_by_arrow(arrow_table(self.VALUES, pa.decimal128(10, 2)).schema, Binary(">", col("c"), constant))

    def test_the_widest_value_of_a_width_is_held(self):
        table = arrow_table([D("99999999.99"), None], pa.decimal128(10, 2))
        assert arrow_ids(table, cmp("=", "c", D("99999999.99"), "DECIMAL(10,2)")) == [0]

    @pytest.mark.parametrize(
        "kind", [pytest.param(pa.decimal32(5, 2), id="32-bit"), pytest.param(pa.decimal64(15, 2), id="64-bit")]
    )
    def test_narrow_decimals_are_refused(self, kind):
        refused_by_arrow(pa.schema([("c", kind)]), cmp(">", "c", D("1.25"), f"DECIMAL({kind.precision},2)"))


class TestArrowBooleans:
    def test_comparisons_and_membership(self):
        table = arrow_table([True, False, None], pa.bool_())
        assert arrow_ids(table, cmp("=", "c", True, "BOOLEAN")) == [0]
        assert arrow_ids(table, cmp(">", "c", False, "BOOLEAN")) == [0]
        assert arrow_ids(table, isin("c", "BOOLEAN", False)) == [1]
        assert arrow_ids(table, negate(isin("c", "BOOLEAN", False))) == [0]
        refused_by_arrow(table.schema, cmp("=", "c", 1, "BOOLEAN"))


class TestArrowStructs:
    KIND = pa.struct([("x", pa.int64()), ("y", pa.struct([("z", pa.string())]))])
    VALUES = ({"x": 1, "y": {"z": "a"}}, {"x": 2, "y": {"z": "b"}}, None, {"x": None, "y": None})

    def table(self):
        return pa.table({"id": [0, 1, 2, 3], "s": pa.array(self.VALUES, self.KIND)})

    def test_field_paths(self):
        table = self.table()
        assert arrow_ids(table, cmp(">", "s.x", 1, "BIGINT")) == [1]
        assert arrow_ids(table, cmp("=", "s.y.z", "a", "VARCHAR")) == [0]
        assert arrow_ids(table, isin("s.x", "BIGINT", 1, 2)) == [0, 1]
        assert arrow_ids(table, negate(isin("s.x", "BIGINT", 1))) == [1]

    def test_a_null_parent_makes_its_fields_null(self):
        table = self.table()
        assert arrow_ids(table, null("s.x")) == [2, 3]
        assert arrow_ids(table, null("s.y.z")) == [2, 3]
        assert arrow_ids(table, null("s.y")) == [2, 3]
        assert arrow_ids(table, null("s")) == [2]
        assert arrow_ids(table, not_null("s")) == [0, 1, 3]

    def test_a_dictionary_field_inside_a_struct_is_refused(self):
        inner = dictionary(["a", "b", None], pa.string())
        schema = pa.table({"id": [0, 1, 2], "s": pa.StructArray.from_arrays([inner], ["c"])}).schema
        refused_by_arrow(schema, cmp("=", "s.c", "b", "VARCHAR"))
        refused_by_arrow(schema, null("s.c"))

    @pytest.mark.parametrize(
        "node",
        [
            pytest.param(cmp("=", "s.missing", 1, "BIGINT"), id="missing-field"),
            pytest.param(cmp("=", "s.x.deeper", 1, "BIGINT"), id="path-through-an-integer"),
            pytest.param(Binary("=", col("s"), Lit({"x": 1}, "STRUCT(x BIGINT)")), id="whole-struct"),
            pytest.param(isin("s.y", "VARCHAR", "a"), id="struct-in"),
        ],
    )
    def test_a_path_it_cannot_resolve_is_refused(self, node):
        refused_by_arrow(self.table().schema, node)

    def test_a_path_through_a_dictionary_of_structs_is_refused(self):
        structs = pa.DictionaryArray.from_arrays(pa.array([0, 1], pa.int32()), pa.array([{"i": 1}, {"i": 2}]))
        refused_by_arrow(pa.table({"s": structs}).schema, cmp(">", "s.i", 1, "BIGINT"))

    def test_a_repeated_field_name_is_refused(self):
        kind = pa.struct([("x", pa.int64()), ("x", pa.int64())])
        refused_by_arrow(pa.schema([("s", kind)]), cmp("=", "s.x", 1, "BIGINT"))


class TestArrowSchemaGate:
    """pyarrow filters and takes rows across every column, so one column it cannot filter refuses every predicate."""

    @pytest.mark.parametrize(
        "kind",
        [
            pytest.param(pa.string_view(), id="string-view"),
            pytest.param(pa.binary_view(), id="binary-view"),
            pytest.param(pa.struct([("v", pa.string_view())]), id="view-in-a-struct"),
            pytest.param(pa.list_(pa.string_view()), id="view-in-a-list"),
            pytest.param(pa.large_list(pa.struct([("v", pa.binary_view())])), id="view-deep-in-a-list"),
            pytest.param(pa.map_(pa.string(), pa.string_view()), id="view-in-a-map"),
            pytest.param(pa.dictionary(pa.int32(), pa.struct([("v", pa.string_view())])), id="view-in-a-dictionary"),
            pytest.param(pa.list_view(pa.int64()), id="list-view"),
            pytest.param(pa.run_end_encoded(pa.int32(), pa.int64()), id="run-end-encoded"),
            pytest.param(pa.struct([("r", pa.run_end_encoded(pa.int32(), pa.int64()))]), id="run-end-in-a-struct"),
            pytest.param(pa.list_(pa.run_end_encoded(pa.int32(), pa.string())), id="run-end-in-a-list"),
            pytest.param(pa.struct([("d", pa.dictionary(pa.int32(), pa.string()))]), id="dictionary-in-a-struct"),
            pytest.param(pa.list_(pa.dictionary(pa.int8(), pa.string())), id="dictionary-in-a-list"),
            pytest.param(pa.map_(pa.string(), pa.dictionary(pa.int32(), pa.string())), id="dictionary-in-a-map"),
            pytest.param(
                pa.struct([("t", pa.struct([("d", pa.dictionary(pa.int32(), pa.int64()))]))]),
                id="dictionary-deep-in-a-struct",
            ),
        ],
    )
    def test_a_column_pyarrow_cannot_filter_refuses_a_predicate_on_another(self, kind):
        refused_by_arrow(pa.schema([("n", pa.int64()), ("other", kind)]), cmp(">", "n", 1, "BIGINT"))

    @DICTIONARY_KERNELS
    def test_a_dictionary_column_of_its_own_keeps_the_predicate(self):
        table = pa.table({"id": [0, 1], "d": pa.array(["a", "b"]).dictionary_encode()})
        assert arrow_ids(table, cmp(">", "id", 0, "BIGINT")) == [1]
        assert arrow_ids(table, cmp("=", "d", "a", "VARCHAR")) == [0]

    def test_nested_columns_pyarrow_can_filter_keep_the_predicate(self):
        table = pa.table(
            {
                "id": [0, 1],
                "l": pa.array([[1], [2]]),
                "m": pa.array([[("a", 1)], [("b", 2)]], pa.map_(pa.string(), pa.int64())),
                "s": pa.array([{"t": "a"}, {"t": "b"}]),
                "u": pa.array([uuid.UUID(int=1).bytes, uuid.UUID(int=2).bytes], pa.uuid()),
            }
        )
        assert arrow_ids(table, cmp(">", "id", 0, "BIGINT")) == [1]


NULLNESS_ONLY_ARROW = [
    pytest.param(pa.array([1, None], pa.time64("ns")), id="nanosecond-time"),
    pytest.param(pa.array([1, None], pa.timestamp("ns")), id="nanosecond-timestamp"),
    pytest.param(pa.array([1, None], pa.timestamp("ns", tz="UTC")), id="nanosecond-zoned"),
    pytest.param(pa.array([D("1.5"), None], pa.decimal32(5, 1)), id="decimal32"),
    pytest.param(pa.array([D("1.5"), None], pa.decimal64(12, 1)), id="decimal64"),
    pytest.param(pa.array([[1], None]), id="list"),
    pytest.param(pa.array([[1], None], pa.large_list(pa.int64())), id="large-list"),
    pytest.param(pa.array([[1], None], pa.list_(pa.int64(), 1)), id="fixed-size-list"),
    pytest.param(pa.array([[("a", 1)], None], pa.map_(pa.string(), pa.int64())), id="map"),
    pytest.param(pa.array([uuid.UUID(int=1).bytes, None], pa.uuid()), id="uuid"),
    pytest.param(pa.array([1, None], pa.duration("us")), id="duration"),
    pytest.param(pa.array([pa.MonthDayNano([1, 2, 3]), None], pa.month_day_nano_interval()), id="interval"),
    pytest.param(
        pa.DictionaryArray.from_arrays(pa.array([0, None], pa.int32()), pa.array([[1]])),
        id="dictionary-of-lists",
        marks=DICTIONARY_KERNELS,
    ),
]


class TestArrowNullness:
    """IS NULL and IS NOT NULL carry no constant, so they translate on a column whose constants do not."""

    @pytest.mark.parametrize("array", NULLNESS_ONLY_ARROW)
    def test_nullness_translates_where_comparisons_are_refused(self, array):
        table = pa.table({"id": [0, 1], "c": array})
        assert arrow_ids(table, null("c")) == [1]
        assert arrow_ids(table, not_null("c")) == [0]
        assert arrow_ids(table, negate(null("c"))) == [0]
        refused_by_arrow(table.schema, cmp("=", "c", 1, "BIGINT"))

    def test_a_column_of_nulls(self):
        table = pa.table({"id": [0, 1], "c": pa.nulls(2)})
        assert arrow_ids(table, null("c")) == [0, 1]
        assert arrow_ids(table, not_null("c")) == []

    @pytest.mark.parametrize(
        "array",
        [
            pytest.param(
                pa.UnionArray.from_sparse(pa.array([0, 1], pa.int8()), [pa.array([1, None]), pa.array(["a", "b"])]),
                id="sparse",
            ),
            pytest.param(
                pa.UnionArray.from_dense(
                    pa.array([0, 1], pa.int8()), pa.array([0, 0], pa.int32()), [pa.array([None]), pa.array(["a"])]
                ),
                id="dense",
            ),
        ],
    )
    def test_a_union_is_refused(self, array):
        refused_by_arrow(pa.table({"c": array}).schema, null("c"))


def polars_ids(frame, node):
    from duckdb._sources.to_polars import to_polars

    return sorted(frame.lazy().filter(to_polars(node, frame.schema)).collect()["id"].to_list())


def refused_by_polars(schema, node):
    from duckdb._sources.to_polars import to_polars

    with pytest.raises(Untranslatable):
        to_polars(node, schema)


@pytest.fixture
def pl():
    return pytest.importorskip("polars")


def polars_frame(pl, values, dtype):
    return pl.DataFrame({"id": list(range(len(values))), "c": pl.Series(values, dtype=dtype)})


class TestPolarsShapes:
    @pytest.mark.parametrize(("node", "expected"), SHAPES)
    def test_every_node_shape(self, pl, node, expected):
        assert polars_ids(polars_frame(pl, INT_VALUES, pl.Int32), node) == expected

    @pytest.mark.parametrize(
        "node",
        [
            pytest.param(Binary(">", Lit(1, "INTEGER"), col("c")), id="constant-on-the-left"),
            pytest.param(Binary("=", col("c"), col("c")), id="column-with-column"),
            pytest.param(negate(col("c")), id="not-of-a-column"),
            pytest.param(In(col("c"), [Lit(1, "INTEGER"), Lit(None, "INTEGER")]), id="in-with-null"),
            pytest.param(In(col("c"), []), id="empty-in"),
            pytest.param(negate(In(col("c"), [])), id="not-empty-in"),
            pytest.param(cmp("=", "missing", 1, "INTEGER"), id="missing-column"),
            pytest.param(cmp("=", "c", 1, "BIGINT"), id="another-width"),
            pytest.param(cmp("=", "c", 1 << 31, "INTEGER"), id="out-of-range"),
        ],
    )
    def test_a_shape_it_cannot_evaluate_is_refused(self, pl, node):
        refused_by_polars(polars_frame(pl, INT_VALUES, pl.Int32).schema, node)


class TestPolarsTypes:
    @pytest.mark.parametrize(("arrow_name", "engine_name", "low", "high"), INTEGERS)
    def test_integer_extremes(self, pl, arrow_name, engine_name, low, high):
        dtype = getattr(pl, arrow_name.replace("uint", "UInt").replace("int", "Int"))
        frame = polars_frame(pl, [low, 1, high, None], dtype)
        assert polars_ids(frame, cmp("=", "c", high, engine_name)) == [2]
        assert polars_ids(frame, cmp(">=", "c", low, engine_name)) == [0, 1, 2]
        assert polars_ids(frame, isin("c", engine_name, low, high)) == [0, 2]
        assert polars_ids(frame, negate(isin("c", engine_name, high))) == [0, 1]
        refused_by_polars(frame.schema, cmp("=", "c", high + 1, engine_name))

    @pytest.mark.parametrize(("dtype", "type_text"), [("Float64", "DOUBLE"), ("Float32", "FLOAT")])
    @pytest.mark.parametrize(("op", "value", "expected"), FLOAT_CASES)
    def test_float_comparisons(self, pl, dtype, type_text, op, value, expected):
        frame = polars_frame(pl, FLOAT_VALUES, getattr(pl, dtype))
        assert polars_ids(frame, cmp(op, "c", value, type_text)) == expected

    @pytest.mark.parametrize(("dtype", "type_text"), [("Float64", "DOUBLE"), ("Float32", "FLOAT")])
    @pytest.mark.parametrize(("node", "expected"), FLOAT_IN_CASES)
    def test_float_membership(self, pl, dtype, type_text, node, expected):
        frame = polars_frame(pl, FLOAT_VALUES, getattr(pl, dtype))
        assert polars_ids(frame, typed(node, type_text)) == expected

    def test_a_float_constant_compares_in_its_own_width(self, pl):
        frame = polars_frame(pl, [0.1, 0.5, None], pl.Float32)
        assert polars_ids(frame, cmp("=", "c", 0.10000000149011612, "FLOAT")) == [0]
        refused_by_polars(frame.schema, cmp("=", "c", 0.1, "FLOAT"))
        refused_by_polars(frame.schema, cmp("=", "c", NAN, "FLOAT"))

    def test_strings(self, pl):
        frame = polars_frame(pl, list(TestArrowStrings.VALUES), pl.String)
        assert polars_ids(frame, cmp(">", "c", "a", "VARCHAR")) == [1, 3, 5]
        assert polars_ids(frame, cmp("=", "c", "it's", "VARCHAR")) == [5]
        assert polars_ids(frame, isin("c", "VARCHAR", "a", "c")) == [0, 3]
        assert polars_ids(frame, negate(isin("c", "VARCHAR", "a", "c"))) == [1, 5]
        assert polars_ids(frame, null("c")) == [2, 4]

    def test_blobs(self, pl):
        frame = polars_frame(pl, list(TestArrowBlobs.VALUES), pl.Binary)
        assert polars_ids(frame, cmp(">", "c", b"a", "BLOB")) == [0, 1]
        assert polars_ids(frame, isin("c", "BLOB", b"a", b"bb")) == [1]

    def test_dates(self, pl):
        frame = polars_frame(pl, [datetime.date(2020, 1, 1), datetime.date(2021, 6, 1), None], pl.Date)
        assert polars_ids(frame, cmp(">=", "c", datetime.date(2021, 1, 1), "DATE")) == [1]
        assert polars_ids(frame, isin("c", "DATE", datetime.date(2020, 1, 1))) == [0]

    def test_booleans(self, pl):
        frame = polars_frame(pl, [True, False, None], pl.Boolean)
        assert polars_ids(frame, cmp(">", "c", False, "BOOLEAN")) == [0]
        assert polars_ids(frame, negate(isin("c", "BOOLEAN", False))) == [0]

    @pytest.mark.parametrize(("unit", "type_text"), [("ms", "TIMESTAMP_MS"), ("us", "TIMESTAMP")])
    def test_naive_datetimes(self, pl, unit, type_text):
        moments = [datetime.datetime(2024, 5, 1, 12), datetime.datetime(2024, 5, 1, 12, 0, 1), None]
        frame = polars_frame(pl, moments, pl.Datetime(unit))
        assert polars_ids(frame, cmp(">", "c", moments[0], type_text)) == [1]
        assert polars_ids(frame, isin("c", type_text, moments[0])) == [0]
        assert polars_ids(frame, negate(isin("c", type_text, moments[0]))) == [1]
        refused_by_polars(frame.schema, cmp(">", "c", datetime.datetime(2024, 1, 1, 0, 0, 0, 5), "TIMESTAMP_MS"))

    @pytest.mark.parametrize("zone", [NEW_YORK, "UTC"])
    @pytest.mark.parametrize("unit", ["ms", "us"])
    def test_a_zoned_constant_keeps_its_microseconds(self, pl, zone, unit):
        per_second = {"ms": 1_000, "us": 1_000_000}[unit]
        counts = pl.Series([BASE * per_second, (BASE + 1) * per_second, None], dtype=pl.Int64)
        frame = pl.DataFrame({"id": [0, 1, 2], "c": counts.cast(pl.Datetime(unit, zone))})
        zoned = "TIMESTAMP WITH TIME ZONE"
        on_grid = (BASE + 1) * 1_000_000
        assert polars_ids(frame, cmp(">=", "c", HALF, zoned)) == [1]
        assert polars_ids(frame, cmp("<", "c", HALF, zoned)) == [0]
        assert polars_ids(frame, cmp("=", "c", HALF, zoned)) == []
        assert polars_ids(frame, cmp("=", "c", on_grid, zoned)) == [1]
        assert polars_ids(frame, isin("c", zoned, HALF, on_grid)) == [1]
        assert polars_ids(frame, negate(isin("c", zoned, HALF))) == [0, 1]
        assert polars_ids(frame, cmp("=", "c", on_grid + 1, zoned)) == []
        assert polars_ids(frame, cmp("<", "c", on_grid + 1, zoned)) == [0, 1]
        assert polars_ids(frame, isin("c", zoned, on_grid + 1)) == []

    def test_decimals(self, pl):
        frame = polars_frame(pl, list(TestArrowDecimals.VALUES), pl.Decimal(10, 2))
        assert polars_ids(frame, cmp(">", "c", D("1.25"), "DECIMAL(10,2)")) == [1]
        assert polars_ids(frame, isin("c", "DECIMAL(10,2)", D("2.5"))) == [1]
        assert polars_ids(frame, negate(isin("c", "DECIMAL(10,2)", D("1.25")))) == [1]
        refused_by_polars(frame.schema, cmp(">", "c", D("1.255"), "DECIMAL(10,2)"))
        refused_by_polars(frame.schema, cmp(">", "c", D("1.25"), "DECIMAL(10,3)"))

    def test_struct_paths(self, pl):
        frame = pl.DataFrame({"id": [0, 1, 2, 3], "s": list(TestArrowStructs.VALUES)})
        assert polars_ids(frame, cmp(">", "s.x", 1, "BIGINT")) == [1]
        assert polars_ids(frame, cmp("=", "s.y.z", "a", "VARCHAR")) == [0]
        assert polars_ids(frame, null("s.x")) == [2, 3]
        assert polars_ids(frame, null("s")) == [2]
        refused_by_polars(frame.schema, cmp("=", "s.missing", 1, "BIGINT"))
        refused_by_polars(frame.schema, isin("s.y", "VARCHAR", "a"))

    @pytest.mark.parametrize(
        "dtype",
        ["Time", "Datetime-ns", "Datetime-ns-zoned", "Categorical", "Enum", "List", "Array", "Duration", "Int128"],
    )
    def test_nullness_translates_where_comparisons_are_refused(self, pl, dtype):
        values = {
            "Time": pl.Series([datetime.time(1), None], dtype=pl.Time),
            "Datetime-ns": pl.Series([datetime.datetime(2024, 1, 1), None], dtype=pl.Datetime("ns")),
            "Datetime-ns-zoned": pl.Series([datetime.datetime(2024, 1, 1), None], dtype=pl.Datetime("ns", "UTC")),
            "Categorical": pl.Series(["a", None], dtype=pl.Categorical()),
            "Enum": pl.Series(["a", None], dtype=pl.Enum(["a"])),
            "List": pl.Series([[1], None]),
            "Array": pl.Series([[1], None], dtype=pl.Array(pl.Int64, 1)),
            "Duration": pl.Series([datetime.timedelta(1), None]),
            "Int128": pl.Series([1, None], dtype=pl.Int128),
        }
        frame = pl.DataFrame({"id": [0, 1], "c": values[dtype]})
        assert polars_ids(frame, null("c")) == [1]
        assert polars_ids(frame, not_null("c")) == [0]
        assert polars_ids(frame, negate(not_null("c"))) == [1]
        refused_by_polars(frame.schema, cmp("=", "c", 1, "BIGINT"))

    def test_a_column_of_nulls(self, pl):
        frame = pl.DataFrame({"id": [0, 1], "c": pl.Series([None, None])})
        assert polars_ids(frame, null("c")) == [0, 1]
        assert polars_ids(frame, not_null("c")) == []

    @pytest.mark.parametrize(
        ("dtype", "constant"),
        [
            pytest.param("Time", Lit(datetime.time(1), "TIME"), id="nanosecond-time"),
            pytest.param("Datetime-ns", Lit(datetime.datetime(2024, 1, 1), "TIMESTAMP"), id="nanosecond-datetime"),
            pytest.param("Datetime-ns-zoned", Lit(HALF, "TIMESTAMP WITH TIME ZONE"), id="nanosecond-zoned"),
        ],
    )
    def test_a_dtype_it_cannot_match_is_refused(self, pl, dtype, constant):
        dtypes = {
            "Time": pl.Time,
            "Datetime-ns": pl.Datetime("ns"),
            "Datetime-ns-zoned": pl.Datetime("ns", "UTC"),
        }
        schema = pl.Schema({"c": dtypes[dtype]})
        refused_by_polars(schema, Binary(">", col("c"), constant))
        refused_by_polars(schema, In(col("c"), [constant]))


class TestFragments:
    @pytest.mark.parametrize(
        ("node", "text"),
        [
            pytest.param(cmp(">", "n", 7, "INTEGER"), '("n" > 7)', id="comparison"),
            pytest.param(cmp("!=", "s", "9", "VARCHAR"), "(\"s\" != '9')", id="string"),
            pytest.param(negate(isin("n", "INTEGER", 0, 1)), '(NOT ("n" IN (0, 1)))', id="not-in"),
            pytest.param(
                cmp(">=", "d", datetime.date(9999, 12, 31), "DATE"), "(\"d\" >= DATE '9999-12-31')", id="date"
            ),
            pytest.param(cmp(">", "", 1, "BIGINT"), '("" > 1)', id="nameless"),
            pytest.param(cmp("=", "s.x", 1, "BIGINT"), '("s"."x" = 1)', id="struct-path"),
            pytest.param(cmp("=", 'a"b', "it's", "VARCHAR"), "(\"a\"\"b\" = 'it''s')", id="quotes"),
            pytest.param(null("n"), '("n" IS NULL)', id="is-null"),
            pytest.param(not_null("n"), '("n" IS NOT NULL)', id="is-not-null"),
            pytest.param(
                both(cmp(">=", "n", 1, "INTEGER"), cmp("<=", "n", 5, "INTEGER")),
                '(("n" >= 1) AND ("n" <= 5))',
                id="between",
            ),
            pytest.param(
                either(cmp("=", "b", True, "BOOLEAN"), cmp("=", "b", False, "BOOLEAN")),
                '(("b" = TRUE) OR ("b" = FALSE))',
                id="booleans",
            ),
            pytest.param(cmp(">", "f", 1.5, "DOUBLE"), '("f" > 1.5::DOUBLE)', id="double"),
            pytest.param(
                cmp(">", "f", 0.10000000149011612, "FLOAT"), '("f" > 0.10000000149011612::DOUBLE)', id="float"
            ),
            pytest.param(
                cmp(">", "m", D("1.50"), "DECIMAL(10,2)"), "(\"m\" > CAST('1.50' AS DECIMAL(10, 2)))", id="decimal"
            ),
            pytest.param(
                cmp(">", "m", D("-12.0"), "DECIMAL(10,1)"), "(\"m\" > CAST('-12.0' AS DECIMAL(10, 1)))", id="negative"
            ),
            pytest.param(cmp("=", "r", b"a\x00", "BLOB"), "(\"r\" = '\\x61\\x00'::BLOB)", id="blob"),
            pytest.param(cmp("<", "t", datetime.time(4), "TIME"), "(\"t\" < TIME '04:00:00')", id="time"),
            pytest.param(
                cmp(">", "ts", datetime.datetime(2024, 5, 6, 7, 8, 9, 123456), "TIMESTAMP"),
                "(\"ts\" > TIMESTAMP '2024-05-06 07:08:09.123456')",
                id="timestamp",
            ),
            pytest.param(
                cmp(">", "ts", datetime.datetime(2024, 5, 6, 7, 8, 9, 123000), "TIMESTAMP_MS"),
                "(\"ts\" > TIMESTAMP '2024-05-06 07:08:09.123')",
                id="timestamp-ms",
            ),
            pytest.param(
                cmp(">", "z", HALF, "TIMESTAMP WITH TIME ZONE"),
                "(\"z\" > TIMESTAMPTZ '2024-05-01 12:00:00.5+00')",
                id="zoned-in-utc",
            ),
        ],
    )
    def test_text(self, node, text):
        assert node.fragment() == text
        assert fragment(node) == text

    @pytest.mark.parametrize(
        "constant",
        [
            pytest.param(Lit(True, "BOOLEAN"), id="boolean"),
            pytest.param(Lit(-(1 << 63), "BIGINT"), id="bigint"),
            pytest.param(Lit((1 << 64) - 1, "UBIGINT"), id="ubigint"),
            pytest.param(Lit(-2.5e-300, "DOUBLE"), id="double"),
            pytest.param(Lit("it's", "VARCHAR"), id="varchar"),
            pytest.param(Lit(b"a\x00\xff", "BLOB"), id="blob"),
            pytest.param(Lit(D("1.00"), "DECIMAL(3,2)"), id="decimal"),
            pytest.param(Lit(D("0." + "9" * 38), "DECIMAL(38,38)"), id="decimal-widest-fraction"),
            pytest.param(Lit(D("1E-38"), "DECIMAL(38,38)"), id="decimal-smallest"),
            pytest.param(Lit(D("-" + "9" * 38), "DECIMAL(38,0)"), id="decimal-widest-integer"),
            pytest.param(Lit(D("1.5"), "DECIMAL(10,4)"), id="decimal-below-its-scale"),
            pytest.param(Lit(datetime.date(1, 1, 1), "DATE"), id="date"),
            pytest.param(Lit(datetime.time(23, 59, 59, 999999), "TIME"), id="time"),
            pytest.param(Lit(datetime.datetime(2024, 5, 6, 7, 8, 9, 120000), "TIMESTAMP"), id="timestamp"),
            pytest.param(Lit(datetime.datetime(2024, 5, 6, 7, 8, 9), "TIMESTAMP_S"), id="timestamp-s"),
        ],
    )
    def test_every_literal_reads_back_as_its_value_in_duckdb(self, constant):
        import duckdb

        with duckdb.connect() as con:
            assert con.execute(f"SELECT {constant.fragment()}").fetchone()[0] == constant.value

    def test_a_zoned_literal_reads_back_as_its_instant_in_duckdb(self):
        import duckdb

        with duckdb.connect() as con:
            for micros in (HALF, -1, 0, 253402300799999999):
                constant = Lit(micros, "TIMESTAMP WITH TIME ZONE")
                assert con.execute(f"SELECT epoch_us({constant.fragment()})").fetchone()[0] == micros

    def test_an_empty_in_has_no_text(self):
        with pytest.raises(Untranslatable):
            In(col("n"), []).fragment()

    def test_any_depth_of_or_renders(self):
        node = cmp("=", "c", 0, "INTEGER")
        for i in range(1, DEEP):
            node = either(node, cmp("=", "c", i, "INTEGER"))
        text = fragment(node)
        assert text.startswith("(" * (DEEP - 1) + '("c" = 0) OR ("c" = 1))')
        assert text.endswith(f'OR ("c" = {DEEP - 1}))')
        assert text.count(" OR ") == DEEP - 1

"""The object families read through the Arrow scan: pyarrow datasets and scanners, polars frames, series.

The tests read through con.register and the replacement scan, or through con._register_arrow_source for a source a
test wraps, and compare rows with a native DuckDB table holding the same data. Nothing imports duckdb._sources at
module level.
"""

from __future__ import annotations

import datetime
import decimal
import math
import threading
import time

import pytest
from _pushdown_helpers import scan_of

import duckdb

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")
pq = pytest.importorskip("pyarrow.parquet")
pc = pytest.importorskip("pyarrow.compute")


@pytest.fixture
def con():
    with duckdb.connect() as connection:
        yield connection


@pytest.fixture
def sources():
    from duckdb import _sources

    return _sources


@pytest.fixture
def parquet_dir(tmp_path):
    for name, start in (("a", 0), ("b", 5)):
        pq.write_table(
            pa.table({"n": list(range(start, start + 5)), "s": [str(i) for i in range(start, start + 5)]}),
            tmp_path / f"{name}.parquet",
        )
    return tmp_path


def rows(con, query):
    return con.execute(query).fetchall()


def schema(con, name):
    return [(column, kind) for column, kind, *_ in con.execute(f'DESCRIBE "{name}"').fetchall()]


def register_and_read(con, name, obj):
    """Register `obj` and read it, for a refusal that may come from either step."""
    con.register(name, obj)
    return rows(con, f'SELECT * FROM "{name}"')


def native(con, name, arrow_table):
    """`arrow_table` copied into a DuckDB table called `name`, the reference a scan's rows are compared with."""
    con.register(f"{name}_arrow", arrow_table)
    con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "{name}_arrow"')
    con.unregister(f"{name}_arrow")


def columns_of(capsule):
    return list(pa.RecordBatchReader._import_from_c_capsule(capsule).read_all().column_names)


def fragments(predicates):
    from duckdb._sources.nodes import fragment

    return [fragment(predicate) for predicate in predicates]


def greater_than(name, value, type_text):
    from duckdb._sources.nodes import Binary, Col, Lit

    return Binary(">", Col((name,)), Lit(value, type_text))


def wrapped(mixin, obj):
    """`obj` adapted, with `mixin` layered over the source class the adapter picked for it."""
    from duckdb._sources import adapt

    base = type(adapt(obj))
    return type(mixin.__name__, (mixin, base), {})(obj)


def uneven_sizes(count):
    """`count` batch sizes cycling through a mix of tiny and large, some of which split across several chunks."""
    pattern = [1, 7, 100, 2048, 3000, 5000]
    return (pattern * (count // len(pattern) + 1))[:count]


def check_order_preserved(con, register, name, total):
    """A monotone `n` column over `name` survives a plain scan, a windowed rank, a LIMIT and a CREATE TABLE AS."""
    register()
    assert [row[0] for row in rows(con, f"SELECT n FROM {name}")] == list(range(total))
    register()
    ranked = rows(con, f"SELECT row_number() OVER () AS r, n FROM {name}")
    assert ranked == [(n + 1, n) for n in range(total)]
    register()
    assert rows(con, f"SELECT n FROM {name} LIMIT 5") == [(i,) for i in range(5)]
    register()
    con.execute(f'CREATE TABLE "{name}_copy" AS SELECT * FROM {name}')
    assert [row[0] for row in rows(con, f'SELECT n FROM "{name}_copy"')] == list(range(total))
    con.execute(f'DROP TABLE "{name}_copy"')


class TestClassification:
    def test_every_family_is_read_as_often_as_asked(self, con, parquet_dir):
        pl = pytest.importorskip("polars")
        pd = pytest.importorskip("pandas")
        frame = pl.DataFrame({"a": [1]})
        families = (
            (ds.dataset(parquet_dir), 10),
            (ds.dataset(parquet_dir).scanner(), 10),
            (frame, 1),
            (frame.lazy(), 1),
            (pd.DataFrame({"a": [1]}), 1),
        )
        for family, count in families:
            con.register("family", family)
            for _ in range(2):
                assert rows(con, "SELECT count(*) FROM family") == [(count,)], type(family).__name__

    def test_a_class_named_like_a_reader_still_needs_the_export(self, con):
        class RecordBatchReader:
            pass

        with pytest.raises(duckdb.InvalidInputException, match="not suitable for replacement scans"):
            con.register("fake", RecordBatchReader())

    def test_a_fragment_is_not_a_dataset(self, con, parquet_dir):
        fragment = next(iter(ds.dataset(parquet_dir).get_fragments()))
        with pytest.raises(duckdb.InvalidInputException, match="not suitable for replacement scans"):
            con.register("fragment", fragment)

    def test_a_dataset_shaped_class_that_is_not_a_dataset_is_refused(self, con, parquet_dir):
        inner = ds.dataset(parquet_dir)

        class Mine:
            schema = inner.schema

            def scanner(self, **kwargs) -> object:
                return inner.scanner(**kwargs)

        with pytest.raises(duckdb.InvalidInputException, match="not suitable for replacement scans"):
            con.register("mine", Mine())

    def test_an_export_attribute_that_raises_is_reported_on_every_entry_point(self, con):
        class Raising:
            @property
            def __arrow_c_stream__(self) -> object:
                message = "classification sentinel"
                raise ValueError(message)

        raising = Raising()
        refusal = "as Arrow data failed: ValueError: classification sentinel"
        with pytest.raises(duckdb.InvalidInputException, match=f"reading the Python object 'raising' {refusal}"):
            con.sql("SELECT * FROM raising")
        with pytest.raises(duckdb.InvalidInputException, match=f"reading the Python object 'raising' {refusal}"):
            con.execute("SELECT * FROM raising")
        with pytest.raises(duckdb.InvalidInputException, match=f"reading the object registered as 'raised' {refusal}"):
            con.register("raised", raising)
        with pytest.raises(duckdb.InvalidInputException, match=f"reading the Arrow source {refusal}"):
            con.from_arrow(raising)

    def test_a_dict_whose_attribute_lookup_raises_key_error_is_read_as_columns(self, con):
        np = pytest.importorskip("numpy")

        class AttrDict(dict):
            __getattr__ = dict.__getitem__

        columns = AttrDict(a=np.array([1, 2, 3]))
        con.register("registered", columns)
        assert rows(con, "SELECT sum(a) FROM registered") == [(6,)]
        assert con.sql("SELECT sum(a) FROM columns").fetchall() == [(6,)]

    def test_an_export_attribute_raising_attribute_error_is_no_export(self, con):
        class Missing:
            @property
            def __arrow_c_stream__(self) -> object:
                message = "__arrow_c_stream__"
                raise AttributeError(message)

        missing = Missing()
        with pytest.raises(duckdb.InvalidInputException, match="not suitable for replacement scans"):
            con.sql("SELECT * FROM missing")
        with pytest.raises(duckdb.InvalidInputException, match="not suitable for replacement scans"):
            con.register("absent", missing)
        with pytest.raises(duckdb.InvalidInputException, match="Python Object Type Missing is not an accepted Arrow"):
            con.from_arrow(missing)


class SchemaProbed:
    """A source of `obj` with no schema hook of its own, reading every column through `obj`'s stream."""

    one_shot = False
    pull_under_gil = True

    def __init__(self, obj) -> None:
        self.obj = obj

    def rows(self) -> None:
        return None

    def accepts(self, predicate) -> bool:
        return False

    def stream(self, columns, filters) -> tuple[object, bool]:
        return pa.table({"n": [1, 2, 3]}).__arrow_c_stream__(), False


class TestSchemaProbes:
    """Looking for a schema export reports an attribute that raises, and takes only AttributeError as no export."""

    def test_a_raising_schema_hook_is_reported(self, con):
        class Hooked(SchemaProbed):
            @property
            def __arrow_c_schema__(self) -> object:
                sentinel = "schema sentinel"
                raise ValueError(sentinel)

        refusal = "reading the schema of the object registered as 'hooked' failed: ValueError: schema sentinel"
        with pytest.raises(duckdb.InvalidInputException, match=refusal):
            con._register_arrow_source("hooked", Hooked(pa.table({"n": [1, 2, 3]})))

    def test_a_raising_schema_of_the_wrapped_object_is_reported(self, con):
        class Wrapped:
            @property
            def schema(self) -> object:
                sentinel = "wrapped schema sentinel"
                raise ValueError(sentinel)

        refusal = "reading the schema of the object registered as 'wrapped' failed: ValueError: wrapped schema sentinel"
        with pytest.raises(duckdb.InvalidInputException, match=refusal):
            con._register_arrow_source("wrapped", SchemaProbed(Wrapped()))

    def test_a_schema_hook_raising_attribute_error_is_no_hook(self, con):
        class Hooked(SchemaProbed):
            @property
            def __arrow_c_schema__(self) -> object:
                missing = "__arrow_c_schema__"
                raise AttributeError(missing)

        con._register_arrow_source("hooked", Hooked(pa.table({"n": [1, 2, 3]})))
        assert rows(con, "SELECT sum(n) FROM hooked") == [(6,)]


class TestDatasets:
    def test_a_dataset_over_files_is_read_repeatedly(self, con, parquet_dir):
        con.register("files", ds.dataset(parquet_dir))
        assert schema(con, "files") == [("n", "BIGINT"), ("s", "VARCHAR")]
        assert rows(con, "SELECT count(*), sum(n) FROM files") == [(10, 45)]
        assert rows(con, "SELECT count(*) FROM files a JOIN files b USING (n)") == [(10,)]

    def test_a_scanner_keeps_its_own_projection_and_filter(self, con, parquet_dir):
        scanner = ds.dataset(parquet_dir).scanner(columns=["n"], filter=ds.field("n") >= 7)
        con.register("part", scanner)
        assert schema(con, "part") == [("n", "BIGINT")]
        assert rows(con, "SELECT n FROM part ORDER BY n") == [(7,), (8,), (9,)]
        assert rows(con, "SELECT count(*) FROM part") == [(3,)]

    def test_a_partitioned_dataset_carries_its_partition_column(self, con, tmp_path):
        for year in (2020, 2021):
            (tmp_path / f"year={year}").mkdir()
            pq.write_table(pa.table({"n": [year - 2000]}), tmp_path / f"year={year}" / "part.parquet")
        con.register("years", ds.dataset(tmp_path, partitioning="hive"))
        assert schema(con, "years") == [("n", "BIGINT"), ("year", "INTEGER")]
        assert rows(con, "SELECT year, n FROM years ORDER BY year") == [(2020, 20), (2021, 21)]
        assert rows(con, "SELECT year FROM years ORDER BY year") == [(2020,), (2021,)]

    def test_a_partitioned_dataset_streams_only_the_partition_column(self, sources, tmp_path):
        for year in (2020, 2021):
            (tmp_path / f"year={year}").mkdir()
            pq.write_table(pa.table({"n": [year - 2000]}), tmp_path / f"year={year}" / "part.parquet")
        capsule, projected = sources.adapt(ds.dataset(tmp_path, partitioning="hive")).stream([1], ())
        assert projected
        assert columns_of(capsule) == ["year"]

    def test_a_dataset_is_scanned_per_query(self, con, parquet_dir):
        con.register("files", ds.dataset(parquet_dir))
        assert rows(con, "SELECT count(*) FROM files") == [(10,)]
        assert rows(con, "SELECT max(n) FROM files") == [(9,)]


class TestProjection:
    def test_a_dataset_scanner_is_asked_for_the_columns(self, con, parquet_dir):
        asked = []
        inner = ds.dataset(parquet_dir)

        class Mine(ds.Dataset):
            def __init__(self) -> None:
                pass

            @property
            def schema(self) -> object:
                return inner.schema

            def scanner(self, **kwargs) -> object:
                asked.append(kwargs.get("columns"))
                return inner.scanner(**kwargs)

        con.register("files", Mine())
        assert rows(con, "SELECT s FROM files ORDER BY s LIMIT 1") == [("0",)]
        assert rows(con, "SELECT count(*) FROM files") == [(10,)]
        assert asked[0] == ["s"]
        assert len(asked) == 2

    def test_a_lazy_frame_is_narrowed_before_it_is_collected(self, con, sources):
        pl = pytest.importorskip("polars")
        lazy = pl.DataFrame({"i": range(5), "s": ["a", "b", "c", "d", "e"], "unused": [0] * 5}).lazy()
        con.register("lf", lazy)
        assert rows(con, "SELECT s FROM lf WHERE i = 2") == [("c",)]
        capsule, projected = sources.adapt(lazy).stream([1], ())
        assert projected
        assert columns_of(capsule) == ["s"]

    def test_tables_project_and_scanners_do_not(self, sources, parquet_dir):
        capsule, projected = sources.adapt(pa.table({"a": [1], "b": [2], "c": [3]})).stream([2, 0], ())
        assert projected
        assert columns_of(capsule) == ["c", "a"]
        capsule, projected = sources.adapt(ds.dataset(parquet_dir).scanner(columns=["s", "n"])).stream([1], ())
        assert not projected
        assert columns_of(capsule) == ["s", "n"]


class TestCardinality:
    def test_in_memory_objects_know_their_length(self, sources):
        pl = pytest.importorskip("polars")
        assert sources.adapt(pl.DataFrame({"a": range(50)})).rows() == 50
        assert sources.adapt(pa.table({"a": range(77)})).rows() == 77
        assert sources.adapt(pa.array([1, 2, 3, 4])).rows() == 4
        assert sources.adapt(pa.chunked_array([[1, 2], [3]])).rows() == 3

    def test_plans_scanners_and_datasets_do_not(self, sources, parquet_dir):
        pl = pytest.importorskip("polars")
        assert sources.adapt(pl.DataFrame({"a": range(50)}).lazy()).rows() is None
        assert sources.adapt(ds.dataset(parquet_dir).scanner(filter=ds.field("n") > 3)).rows() is None
        assert sources.adapt(ds.dataset(parquet_dir)).rows() is None

    def test_a_filtered_dataset_reads_what_its_filter_keeps(self, con, sources, parquet_dir):
        filtered = ds.dataset(parquet_dir).filter(ds.field("n") >= 7)
        assert sources.adapt(filtered).rows() is None
        con.register("late", filtered)
        assert rows(con, "SELECT count(*) FROM late") == [(3,)]

    def test_a_csv_dataset_does_not_count_by_reading(self, sources, tmp_path):
        (tmp_path / "rows.csv").write_text("n\n1\n2\n3\n")
        assert sources.adapt(ds.dataset(tmp_path / "rows.csv", format="csv")).rows() is None


class TestSeries:
    def test_a_pandas_series_is_one_column(self, con):
        pd = pytest.importorskip("pandas")
        con.register("s", pd.Series([1.5, None], name="f"))
        assert schema(con, "s") == [("value", "DOUBLE")]
        assert rows(con, "SELECT value FROM s") == [(1.5,), (None,)]

    def test_a_polars_series_keeps_its_name(self, con):
        pl = pytest.importorskip("polars")
        con.register("s", pl.Series("p", ["x", None]))
        assert schema(con, "s") == [("p", "VARCHAR")]
        assert rows(con, "SELECT p FROM s") == [("x",), (None,)]


class TestLazyFrames:
    def test_a_lazy_frame_is_collected_per_query(self, con):
        pl = pytest.importorskip("polars")
        collected = 0

        def counting(batch):
            nonlocal collected
            collected += 1
            return batch

        lazy = pl.DataFrame({"i": range(10)}).lazy().map_batches(counting).filter(pl.col("i") >= 5)
        con.register("lf", lazy)
        assert schema(con, "lf") == [("i", "BIGINT")]
        assert collected == 0
        assert rows(con, "SELECT sum(i) FROM lf") == [(35,)]
        assert rows(con, "SELECT count(*) FROM lf") == [(5,)]
        assert collected == 2

    def test_a_lazy_frame_that_cannot_run_fails(self, con, tmp_path):
        pl = pytest.importorskip("polars")
        missing = tmp_path / "nope.parquet"
        with pytest.raises(duckdb.Error, match=r"nope\.parquet"):
            register_and_read(con, "missing", pl.scan_parquet(missing))

    def test_a_plan_whose_output_differs_from_its_declared_schema_fails_at_the_query(self, con):
        pl = pytest.importorskip("polars")
        # Without this polars itself refuses the frame at collect, before the scan sees the String column.
        lazy = (
            pl.DataFrame({"i": [1, 2]})
            .lazy()
            .map_batches(lambda b: b.with_columns(pl.col("i").cast(pl.Utf8)), validate_output_schema=False)
        )
        con.register("wrong", lazy)
        assert schema(con, "wrong") == [("i", "BIGINT")]
        expected = "answered with column 'i' of type VARCHAR where BIGINT was declared"
        with pytest.raises(duckdb.InvalidInputException, match=expected):
            rows(con, "SELECT * FROM wrong")
        con.register("wrong", pl.DataFrame({"i": [1, 2]}).lazy())
        assert rows(con, "SELECT sum(i) FROM wrong") == [(3,)]

    @pytest.mark.parametrize("dtype", ["String", "Categorical"])
    def test_string_columns_match_their_bind_type(self, con, dtype):
        pl = pytest.importorskip("polars")
        values = ["x", None, "y" * 40]
        lazy = pl.DataFrame({"s": values, "n": [1, 2, 3]}).lazy().with_columns(pl.col("s").cast(getattr(pl, dtype)))
        con.register("strings", lazy)
        assert schema(con, "strings") == [("s", "VARCHAR"), ("n", "BIGINT")]
        for _ in range(2):
            assert rows(con, "SELECT s, n FROM strings ORDER BY n") == list(zip(values, [1, 2, 3], strict=True))

    def test_categoricals_read_the_same_twice_through_a_lazy_frame(self, con):
        pl = pytest.importorskip("polars")
        lazy = pl.DataFrame({"c": ["x", "y", "x", None]}).lazy().with_columns(pl.col("c").cast(pl.Categorical))
        con.register("cats", lazy)
        for _ in range(2):
            assert rows(con, "SELECT c, count(*) FROM cats GROUP BY c ORDER BY c") == [("x", 2), ("y", 1), (None, 1)]


@pytest.fixture
def typed_dir(tmp_path):
    """Two parquet files under hive partitions, with a NULL and a NaN in every column that can hold one."""
    for year, start in ((2020, 0), (2021, 5)):
        (tmp_path / f"year={year}").mkdir()
        ns = [start, start + 1, None, start + 3, start + 4]

        def stamps(unit, zone=None, ns=ns, year=year):
            values = [None if n is None else datetime.datetime(year, 1, 1, n) for n in ns]
            return pa.array(values, pa.timestamp(unit, zone))

        pq.write_table(
            pa.table(
                {
                    "n": pa.array(ns, pa.int32()),
                    "s": [None if n is None else str(n) for n in ns],
                    "f": [float("nan") if n == start + 3 else (None if n is None else float(n)) for n in ns],
                    "flag": [None if n is None else n % 2 == 0 for n in ns],
                    "d": [None if n is None else datetime.date(year, 1, n + 1) for n in ns],
                    "ts": stamps("us"),
                    "ts_ns": stamps("ns"),
                    "ts_ms": stamps("ms"),
                    "tz": stamps("us", "UTC"),
                    "dec": pa.array([None if n is None else decimal.Decimal(n) / 2 for n in ns], pa.decimal128(10, 2)),
                    "st": pa.array([None if n is None else {"x": n} for n in ns], pa.struct([("x", pa.int32())])),
                    "raw": pa.array([None if n is None else str(n).encode() for n in ns], pa.binary()),
                }
            ),
            tmp_path / f"year={year}" / "part.parquet",
        )
    return tmp_path


class Pushing:
    """A source that remembers what it was offered and what each scan applied."""

    def __init__(self, obj) -> None:
        super().__init__(obj)
        self.offered = []
        self.applied = []

    def accepts(self, predicate) -> bool:
        self.offered.extend(fragments([predicate]))
        return super().accepts(predicate)

    def stream(self, columns, filters) -> tuple[object, bool]:
        self.applied.append(fragments(filters))
        return super().stream(columns, filters)


PUSHED = [
    "n > 6",
    "6 < n",
    "n >= 6",
    "n < 3",
    "n <= 3",
    "n = 3",
    "n <> 3",
    "n IN (1, 3, 5)",
    "n NOT IN (1, 3, 5)",
    "NOT (n IN (0, 5) OR s = '4')",
    "NOT (n > 6)",
    "n > 1 AND s <> '4'",
    "n < 2 OR n > 7",
    "n IS NULL",
    "s IS NOT NULL",
    "s = '5'",
    "s > '5'",
    "s IN ('1', '9')",
    "flag = true",
    "flag <> false",
    "f > 6.0",
    "f >= 8.0",
    "f < 3.0",
    "f <= 3.0",
    "f = 1.0",
    "f <> 1.0",
    "NOT (f > 6.0)",
    "d >= DATE '2021-01-02'",
    "d = DATE '2020-01-05'",
    "ts > TIMESTAMP '2021-01-01 04:00:00'",
    "ts IN (TIMESTAMP '2020-01-01 00:00:00', TIMESTAMP '2021-01-01 09:00:00')",
    "year = 2021",
    "year = 2021 AND n > 6",
    # The engine exposes a range's bounds and their inclusivity, so BETWEEN and a folded range are offered.
    "n BETWEEN 3 AND 6",
    "n > 1 AND n < 8",
    "CAST(ts AS DATE) = DATE '2021-01-01'",
    "tz > TIMESTAMPTZ '2021-01-01 04:00:00+00'",
    "tz IN (TIMESTAMPTZ '2020-01-01 00:00:00+00', TIMESTAMPTZ '2021-01-01 09:00:00+00')",
    "dec > 2.00",
    "dec IN (0.50, 4.50)",
    "st.x > 6",
    "st.x IS NULL",
    "raw = '5'::BLOB",
    "raw > '5'::BLOB",
]

KEPT = [
    "n > 1.5",
    "n > 3.5",
    "lower(s) = '4'",
    "s = n",
    "n IN (1, NULL)",
    "n IN (1, 3.5)",
    "flag",
    "ts_ns >= TIMESTAMP '2021-01-01 00:00:00'",
    "f = 'nan'::DOUBLE",
    "d >= DATE 'infinity'",
    "d = DATE '-infinity'",
    "ts < TIMESTAMP 'infinity'",
    "ts IN (TIMESTAMP '-infinity', TIMESTAMP '2021-01-01 09:00:00')",
    "n IS DISTINCT FROM 3",
    "coalesce(n, 0) > 6",
    "CASE WHEN n > 6 THEN true ELSE false END",
    "s LIKE '%5%'",
    "s LIKE '1%'",
    "s LIKE '1%1'",
    "CAST(ts AS DATE) > DATE '2020-01-03'",
    "CAST(tz AS DATE) = DATE '2021-01-01'",
    "CAST(ts_ms AS DATE) = DATE '2021-01-01'",
]

SELECTED = "n, s, f, flag, d, ts, tz, dec, st, raw, year"


def without_nan(result):
    """NaN compares unequal to itself, so rows are compared with it spelled out."""
    return [tuple("nan" if isinstance(v, float) and math.isnan(v) else v for v in row) for row in result]


def register_dataset_pair(con, typed_dir):
    source = wrapped(Pushing, ds.dataset(typed_dir, partitioning="hive"))
    con._register_arrow_source("pushed", source)
    native(con, "plain", ds.dataset(typed_dir, partitioning="hive").to_table())
    return source


def register_lazy_pair(con, typed_dir):
    pl = pytest.importorskip("polars")
    source = wrapped(Pushing, pl.scan_parquet(typed_dir, hive_partitioning=True))
    con._register_arrow_source("pushed", source)
    native(con, "plain", pl.scan_parquet(typed_dir, hive_partitioning=True).collect().to_arrow())
    return source


PAIRS = [pytest.param(register_dataset_pair, id="dataset"), pytest.param(register_lazy_pair, id="lazyframe")]

# A Dataset, and a LazyFrame over files, prune row groups by their minimum and maximum before filtering, and those
# leave NaN out, so on a float column they take only predicates that NaN, the greatest value to the engine, cannot
# satisfy.
NAN_MAY_SATISFY = ("f > 6.0", "f >= 8.0", "f <> 1.0")


def placement_cases(wheres, *, kept):
    """Each predicate with each source pair: the pushed ones, or the kept ones, after the float rule."""
    cases = [
        pytest.param(pair.values[0], where, id=f"{where}-{pair.id}")
        for where in wheres
        for pair in PAIRS
        if kept or where not in NAN_MAY_SATISFY
    ]
    if kept:
        cases += [
            pytest.param(pair.values[0], where, id=f"{where}-{pair.id}") for where in NAN_MAY_SATISFY for pair in PAIRS
        ]
    return cases


class TestPushdown:
    """A Dataset and a LazyFrame apply what they accept exactly, and anything else stays with the engine."""

    @pytest.mark.parametrize(("register_pair", "where"), placement_cases(PUSHED, kept=False))
    def test_a_pushed_predicate_selects_what_the_engine_selects(self, con, typed_dir, register_pair, where):
        source = register_pair(con, typed_dir)
        expected = rows(con, f"SELECT {SELECTED} FROM plain WHERE {where} ORDER BY year, n")
        pushed = rows(con, f"SELECT {SELECTED} FROM pushed WHERE {where} ORDER BY year, n")
        assert without_nan(pushed) == without_nan(expected)
        assert expected
        assert source.applied
        assert all(source.applied), where
        scan, filter_above = scan_of(con, f"SELECT n FROM pushed WHERE {where}")
        assert not filter_above, where
        assert scan["extra_info"].get("Filters"), where

    @pytest.mark.parametrize(("register_pair", "where"), placement_cases(KEPT, kept=True))
    def test_a_predicate_outside_the_set_stays_with_the_engine(self, con, typed_dir, register_pair, where):
        source = register_pair(con, typed_dir)
        expected = rows(con, f"SELECT n, year FROM plain WHERE {where} ORDER BY year, n")
        assert rows(con, f"SELECT n, year FROM pushed WHERE {where} ORDER BY year, n") == expected
        assert source.applied == [[]], where
        _, filter_above = scan_of(con, f"SELECT n FROM pushed WHERE {where}")
        assert filter_above, where

    @pytest.mark.parametrize("register_pair", PAIRS)
    def test_a_predicate_nothing_satisfies_yields_no_rows(self, con, typed_dir, register_pair):
        source = register_pair(con, typed_dir)
        assert rows(con, "SELECT n FROM pushed WHERE n > 100") == []
        assert rows(con, "SELECT count(*) FROM pushed WHERE s = 'none'") == [(0,)]
        assert source.applied == [['("n" > 100)'], ["(\"s\" = 'none')"]]

    @pytest.mark.parametrize("register_pair", PAIRS)
    def test_a_conjunction_is_pushed_conjunct_by_conjunct(self, con, typed_dir, register_pair):
        source = register_pair(con, typed_dir)
        where = "n > 1 AND n < 8 AND s <> '4'"
        expected = rows(con, f"SELECT n FROM plain WHERE {where} ORDER BY n")
        assert rows(con, f"SELECT n FROM pushed WHERE {where} ORDER BY n") == expected
        assert expected == [(3,), (5,), (6,)]
        both = sorted(['(("n" > 1) AND ("n" < 8))', "(\"s\" != '4')"])
        assert sorted(source.offered) == both
        assert [sorted(applied) for applied in source.applied] == [both]
        _, filter_above = scan_of(con, f"SELECT n FROM pushed WHERE {where}")
        assert not filter_above

    @pytest.mark.parametrize("register_pair", PAIRS)
    def test_a_conjunction_with_a_refused_part_keeps_only_that_part(self, con, typed_dir, register_pair):
        source = register_pair(con, typed_dir)
        where = "n > 1 AND s LIKE '%5%'"
        expected = rows(con, f"SELECT n FROM plain WHERE {where} ORDER BY n")
        assert rows(con, f"SELECT n FROM pushed WHERE {where} ORDER BY n") == expected
        assert source.applied == [['("n" > 1)']]
        _, filter_above = scan_of(con, f"SELECT n FROM pushed WHERE {where}")
        assert filter_above

    def test_a_partition_filter_prunes_before_reading(self, con, typed_dir):
        source = register_dataset_pair(con, typed_dir)
        assert rows(con, "SELECT count(*) FROM pushed WHERE year = 2021 AND n IS NOT NULL") == [(4,)]
        assert sorted(source.applied[0]) == ['("n" IS NOT NULL)', '("year" = 2021)']
        fragments_read = ds.dataset(typed_dir, partitioning="hive").get_fragments(filter=ds.field("year") == 2021)
        assert len(list(fragments_read)) == 1

    @pytest.mark.parametrize("register_pair", PAIRS)
    def test_only_the_columns_read_are_returned(self, con, typed_dir, register_pair):
        source = register_pair(con, typed_dir)
        capsule, projected = source.stream([1], (greater_than("n", 6, "INTEGER"),))
        assert projected
        assert columns_of(capsule) == ["s"]
        assert rows(con, "SELECT s FROM pushed WHERE n > 6 ORDER BY s") == [("8",), ("9",)]

    def test_a_scanner_keeps_refusing(self, con, sources, typed_dir):
        scanner = ds.dataset(typed_dir, partitioning="hive").scanner(columns=["n"], filter=ds.field("n") >= 7)
        assert sources.adapt(scanner).accepts(greater_than("n", 8, "INTEGER")) is False
        con.register("part", scanner)
        assert rows(con, "SELECT n FROM part WHERE n > 8") == [(9,)]

    def test_an_eager_frame_keeps_refusing(self, con, sources):
        pl = pytest.importorskip("polars")
        frame = pl.DataFrame({"n": [1, 2, 3]})
        assert sources.adapt(frame).accepts(greater_than("n", 1, "BIGINT")) is False
        con.register("eager", frame)
        assert rows(con, "SELECT n FROM eager WHERE n > 1 ORDER BY n") == [(2,), (3,)]


class TestLazyFramePushdownShapes:
    def test_a_decimal_pushes_and_a_nanosecond_time_stays(self, con):
        pl = pytest.importorskip("polars")
        frame = pl.DataFrame(
            {
                "money": pl.Series([decimal.Decimal("1.50"), decimal.Decimal("2.25"), None], dtype=pl.Decimal(10, 2)),
                "clock": [datetime.time(1, 2, 3), datetime.time(4, 5, 6), None],
            }
        )
        source = wrapped(Pushing, frame.lazy())
        con._register_arrow_source("late", source)
        native(con, "now", frame.to_arrow())
        assert schema(con, "late") == [("money", "DECIMAL(10,2)"), ("clock", "TIME_NS")]
        for where, pushed in (
            ("money > 1.50", True),
            ("money IN (1.50, 9.99)", True),
            ("money IS NULL", True),
            ("clock < '04:00:00'::TIME_NS", False),
            ("clock IS NULL", True),
        ):
            expected = rows(con, f"SELECT money FROM now WHERE {where}")
            assert rows(con, f"SELECT money FROM late WHERE {where}") == expected
            assert bool(source.applied[-1]) is pushed, where

    def test_a_float32_column_compares_in_its_own_width(self, con):
        pl = pytest.importorskip("polars")
        frame = pl.DataFrame({"f": pl.Series([0.1, 0.2, None], dtype=pl.Float32)})
        source = wrapped(Pushing, frame.lazy())
        con._register_arrow_source("late", source)
        native(con, "now", frame.to_arrow())
        # NaN could satisfy > and >=, so those stay above the scan; the rest are taken in the column's width.
        for where, taken in (
            ("f > 0.1", False),
            ("f = 0.1", True),
            ("f >= 0.1", False),
            ("f < 0.2", True),
            ("f IN (0.1, 0.3)", True),
        ):
            expected = rows(con, f"SELECT f FROM now WHERE {where}")
            assert rows(con, f"SELECT f FROM late WHERE {where}") == expected, where
            assert bool(source.applied[-1]) is taken, where
        assert rows(con, "SELECT count(*) FROM late WHERE f > 0.1") == [(1,)]

    def test_a_decimal_literal_beyond_the_scale_stays_with_the_engine(self, con):
        pl = pytest.importorskip("polars")
        money = pl.Series([decimal.Decimal("1.50"), decimal.Decimal("-1.51")], dtype=pl.Decimal(10, 2))
        frame = pl.DataFrame({"money": money})
        source = wrapped(Pushing, frame.lazy())
        con._register_arrow_source("late", source)
        native(con, "now", frame.to_arrow())
        for where, pushed in (
            ("money > 1.5", True),
            ("money = -1.51", True),
            ("money = 1.505", False),
            ("money > 1.499", False),
        ):
            expected = rows(con, f"SELECT money FROM now WHERE {where}")
            assert rows(con, f"SELECT money FROM late WHERE {where}") == expected, where
            assert bool(source.applied[-1]) is pushed, where


class TestPolarsShapes:
    def test_a_column_of_python_objects_is_refused(self, con):
        pl = pytest.importorskip("polars")
        frame = pl.DataFrame({"n": [1, 2], "o": pl.Series([object(), object()], dtype=pl.Object)})
        refusal = r"as Arrow data failed: TypeError: the polars column 'o' holds Python objects"
        with pytest.raises(duckdb.InvalidInputException, match=f"reading the object registered as 'eager' {refusal}"):
            con.register("eager", frame)
        with pytest.raises(duckdb.InvalidInputException, match=f"reading the Python object 'frame' {refusal}"):
            con.sql("SELECT * FROM frame")
        with pytest.raises(duckdb.InvalidInputException, match=f"reading the Python object 'frame' {refusal}"):
            con.execute("SELECT * FROM frame")
        with pytest.raises(duckdb.InvalidInputException, match=f"reading the Arrow source {refusal}"):
            con.from_arrow(frame)
        with pytest.raises(duckdb.InvalidInputException, match="the polars column 'o' holds Python objects"):
            register_and_read(con, "lazy", frame.lazy())
        con.register("fine", frame.select("n"))
        assert rows(con, "SELECT sum(n) FROM fine") == [(3,)]

    def test_a_frame_just_over_a_power_of_two_scans_whole(self, con):
        pl = pytest.importorskip("polars")
        n = 65_537
        holes = [None if i in (0, n - 2, n - 1) else str(i) for i in range(n)]
        con.register("src", pl.DataFrame({"n": range(n), "s": holes}))
        assert rows(con, "SELECT count(*), sum(n), count(s) FROM src") == [(n, sum(range(n)), n - 3)]
        last = rows(con, "SELECT n, s FROM src ORDER BY n DESC LIMIT 3")
        assert last == [(n - 1, None), (n - 2, None), (n - 3, str(n - 3))]
        assert rows(con, "SELECT n FROM src WHERE n IN (0, 65535, 65536) ORDER BY n") == [(0,), (65535,), (65536,)]

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    def test_an_empty_frame_gives_no_rows(self, con, lazy):
        pl = pytest.importorskip("polars")
        frame = pl.DataFrame({"n": pl.Series([], dtype=pl.Int64), "s": pl.Series([], dtype=pl.String)})
        con.register("empty", frame.lazy() if lazy else frame)
        assert rows(con, "SELECT * FROM empty") == []
        assert rows(con, "SELECT count(*), sum(n) FROM empty") == [(0, None)]

    def test_a_self_join_reads_a_chunked_frame_twice_at_once(self, con):
        pl = pytest.importorskip("polars")
        con.execute("SET threads = 4")
        parts = [pl.DataFrame({"n": range(start, start + 1000)}) for start in range(0, 5000, 1000)]
        frame = pl.concat(parts, rechunk=False)
        con.register("m", frame)
        assert rows(con, "SELECT count(*), sum(a.n) FROM m a JOIN m b USING (n)") == [(5000, sum(range(5000)))]
        assert frame.n_chunks() == 5

    def test_scanning_a_chunked_frame_leaves_its_chunk_count_unchanged(self, con):
        pl = pytest.importorskip("polars")
        frame = pl.concat([pl.DataFrame({"n": [1, 2]}), pl.DataFrame({"n": [3, 4, 5]})], rechunk=False)
        assert frame.n_chunks() == 2
        con.register("src", frame)
        assert rows(con, "SELECT n FROM src") == [(1,), (2,), (3,), (4,), (5,)]
        assert frame.n_chunks() == 2


NESTED_ENCODINGS = {
    "list_of_string_view": lambda: pa.array([["a"], None, ["c"]], pa.list_(pa.string_view())),
    "struct_of_binary_view": lambda: pa.array([{"v": b"a"}, None, {"v": b"c"}], pa.struct([("v", pa.binary_view())])),
    "struct_of_run_end_encoded": lambda: pa.StructArray.from_arrays(
        [pc.run_end_encode(pa.array([1, 1, 3]))], ["v"], mask=pa.array([False, True, False])
    ),
    "list_of_dictionary": lambda: pa.array([["a"], None, ["c"]]).cast(pa.list_(pa.dictionary(pa.int32(), pa.string()))),
    "struct_of_list_of_run_end_encoded": lambda: pa.StructArray.from_arrays(
        [pa.ListArray.from_arrays(pa.array([0, 1, 1, 2], pa.int32()), pc.run_end_encode(pa.array(["a", "c"])))],
        ["v"],
        mask=pa.array([False, True, False]),
    ),
}


class TestDatasetShapes:
    @pytest.mark.parametrize("binary", [False, True], ids=["string_view", "binary_view"])
    def test_a_view_column_keeps_every_filter_with_the_engine(self, con, binary):
        kind = pa.binary_view() if binary else pa.string_view()
        values = [b"abc", b"efg", None] if binary else ["abc", "efg", None]
        source = wrapped(Pushing, ds.dataset(pa.table({"v": pa.array(values, kind), "n": [1, 2, 3]})))
        con._register_arrow_source("views", source)
        assert rows(con, "SELECT n FROM views WHERE v IS NULL") == [(3,)]
        assert rows(con, "SELECT n FROM views WHERE n > 1") == [(2,), (3,)]
        assert rows(con, "SELECT v IS NULL FROM views WHERE n > 1") == [(False,), (True,)]
        assert source.applied == [[], [], []]

    @pytest.mark.parametrize("nested", list(NESTED_ENCODINGS))
    @pytest.mark.parametrize("query", ["SELECT n FROM views WHERE n > 1", "SELECT * FROM views WHERE n > 1"])
    def test_an_integer_filter_beside_a_nested_column_pyarrow_cannot_filter_stays_with_the_engine(
        self, con, nested, query
    ):
        table = pa.table({"x": NESTED_ENCODINGS[nested](), "n": [1, 2, 3]})
        source = wrapped(Pushing, ds.dataset(table))
        con._register_arrow_source("views", source)
        native(con, "plain", table)
        assert rows(con, query.replace("views", "plain") + " ORDER BY n") == rows(con, query + " ORDER BY n")
        assert source.applied == [[]]

    def test_a_view_column_stays_unfilterable_in_this_pyarrow(self):
        views = pa.table({"v": pa.array(["a", "b"], pa.string_view()), "n": [1, 2]})
        with pytest.raises(pa.ArrowNotImplementedError, match="array_filter"):
            ds.dataset(views).scanner(filter=ds.field("n") > 1).to_table()

    def test_a_top_n_over_a_dataset_and_a_plan(self, con, typed_dir):
        pl = pytest.importorskip("polars")
        con.register("files", ds.dataset(typed_dir, partitioning="hive"))
        con.register("plan", pl.scan_parquet(typed_dir, hive_partitioning=True))
        for name in ("files", "plan"):
            assert rows(con, f"SELECT n FROM {name} ORDER BY n DESC LIMIT 2") == [(9,), (8,)]
            assert rows(con, f"SELECT n FROM {name} ORDER BY n ASC NULLS FIRST LIMIT 3") == [(None,), (None,), (0,)]

    def test_not_in_and_a_long_in_list(self, con, typed_dir):
        source = wrapped(Pushing, ds.dataset(typed_dir, partitioning="hive"))
        con._register_arrow_source("files", source)
        assert rows(con, "SELECT n FROM files WHERE n NOT IN (0, 1, 3, 4, 5, 6) ORDER BY n") == [(8,), (9,)]
        assert source.applied[-1] == ['(NOT ("n" IN (0, 1, 3, 4, 5, 6)))']
        members = ", ".join(str(i) for i in range(1000, 6000)) + ", 9"
        assert rows(con, f"SELECT n FROM files WHERE n IN ({members})") == [(9,)]
        [applied] = source.applied[-1]
        assert applied.startswith('("n" IN (1000, 1001')


class TestParallelOrdering:
    """The scan's batch ordering survives four threads pulling from one chunked or batched source."""

    def test_order_preserved_over_a_record_batch_reader(self, con):
        con.execute("SET threads = 4")
        sizes = uneven_sizes(204)
        total = sum(sizes)
        arrow_schema = pa.schema([("n", pa.int64())])

        def batches():
            start = 0
            for size in sizes:
                yield pa.record_batch([pa.array(range(start, start + size))], schema=arrow_schema)
                start += size

        def register():
            con.register("src", pa.RecordBatchReader.from_batches(arrow_schema, batches()))

        check_order_preserved(con, register, "src", total)

    def test_order_preserved_over_a_table_with_many_chunks(self, con):
        con.execute("SET threads = 4")
        sizes = uneven_sizes(204)
        total = sum(sizes)
        arrow_schema = pa.schema([("n", pa.int64())])
        start = 0
        batches = []
        for size in sizes:
            batches.append(pa.record_batch([pa.array(range(start, start + size))], schema=arrow_schema))
            start += size
        table = pa.Table.from_batches(batches)
        assert table.column(0).num_chunks == len(sizes)
        con.register("src", table)
        check_order_preserved(con, lambda: None, "src", total)

    def test_order_preserved_over_a_polars_frame_with_many_chunks(self, con):
        pl = pytest.importorskip("polars")
        con.execute("SET threads = 4")
        sizes = uneven_sizes(204)
        total = sum(sizes)
        start = 0
        parts = []
        for size in sizes:
            parts.append(pl.DataFrame({"n": range(start, start + size)}))
            start += size
        frame = pl.concat(parts, rechunk=False)
        assert frame.n_chunks() == len(sizes)
        con.register("src", frame)
        check_order_preserved(con, lambda: None, "src", total)

    def test_order_preserved_over_a_one_chunk_polars_frame(self, con):
        pl = pytest.importorskip("polars")
        con.execute("SET threads = 4")
        total = sum(uneven_sizes(204))
        frame = pl.DataFrame({"n": range(total)})
        assert frame.n_chunks() == 1
        con.register("src", frame)
        check_order_preserved(con, lambda: None, "src", total)


class TestParallelResultsMatch:
    """Every source family gives the same answer scanned by one thread as by four."""

    def test_matches_the_single_threaded_scan(self, con):
        pl = pytest.importorskip("polars")
        pd = pytest.importorskip("pandas")
        n = 20_000
        chunk = 137
        arrow_schema = pa.schema([("n", pa.int64())])

        def table_batches():
            return [
                pa.record_batch([pa.array(range(start, min(start + chunk, n)))], schema=arrow_schema)
                for start in range(0, n, chunk)
            ]

        table = pa.Table.from_batches(table_batches())
        frame = pl.concat(
            [pl.DataFrame({"n": range(start, min(start + chunk, n))}) for start in range(0, n, chunk)],
            rechunk=False,
        )
        pandas_frame = pd.DataFrame({"n": range(n)})
        makers = [
            ("table", lambda: table),
            ("batch reader", lambda: pa.RecordBatchReader.from_batches(arrow_schema, table_batches())),
            ("polars frame", lambda: frame),
            ("lazy frame", lambda: frame.lazy()),
            ("pandas frame", lambda: pandas_frame),
            ("capsule", lambda: table.__arrow_c_stream__()),
        ]
        for name, maker in makers:
            for threads in (1, 4):
                con.execute(f"SET threads = {threads}")
                con.register("src", maker())
                assert rows(con, "SELECT count(*), sum(n) FROM src") == [(n, sum(range(n)))], (name, threads)

    def test_a_dataset_and_a_scanner_match_the_single_threaded_scan(self, con, tmp_path):
        n = 200_000
        files = 4
        for part in range(files):
            start = part * n // files
            pq.write_table(pa.table({"n": range(start, start + n // files)}), tmp_path / f"{part}.parquet")
        makers = [
            ("dataset", lambda: ds.dataset(tmp_path)),
            ("scanner", lambda: ds.dataset(tmp_path).scanner(batch_size=1000)),
        ]
        for name, maker in makers:
            for threads in (1, 4):
                con.execute(f"SET threads = {threads}")
                con.register("files", maker())
                assert rows(con, "SELECT count(*), sum(n) FROM files") == [(n, sum(range(n)))], (name, threads)

    def test_pulls_from_several_threads_agree_with_one(self, con):
        arrow_schema = pa.schema([("n", pa.int64())])
        pulling = {"now": 0, "overlaps": 0}

        def batches(pulled):
            for start in range(0, 100_000, 1000):
                if pulling["now"]:
                    pulling["overlaps"] += 1
                pulling["now"] += 1
                pulled.add(threading.get_ident())
                # Held a little, so that a pull another thread started meanwhile would be seen.
                time.sleep(0.001)
                pulling["now"] -= 1
                yield pa.record_batch([pa.array(range(start, start + 1000))], schema=arrow_schema)

        pullers = {}
        answers = {}
        for threads in (1, 4):
            con.execute(f"SET threads = {threads}")
            pullers[threads] = set()
            con.register("src", pa.RecordBatchReader.from_batches(arrow_schema, batches(pullers[threads])))
            answers[threads] = rows(con, "SELECT count(*), sum(n) FROM src")
        assert answers[1] == [(100_000, sum(range(100_000)))]
        assert answers[4] == answers[1]
        assert len(pullers[1]) == 1
        assert pulling["overlaps"] == 0


TRANSLATION_DATA = {
    "n": pa.array([1, 2, 5, 6, None], pa.int32()),
    "big": pa.array([1, 2, 5, 6, None], pa.int64()),
    "f": pa.array([0.5, 1.0, float("nan"), 7.0, None]),
    "s": pa.array(["a", "x", "y", None, "x"], pa.large_string()),
    "d": pa.array([datetime.date(2020, 1, d) for d in (1, 2, 3, 4)] + [None]),
    "ts": pa.array([datetime.datetime(2019, 12, 31), datetime.datetime(2020, 1, 2), None, None, None]),
    "zoned": pa.array(
        [datetime.datetime(2020, 1, 1, h, tzinfo=datetime.timezone.utc) for h in (0, 1, 2, 3)] + [None],
        pa.timestamp("us", tz="UTC"),
    ),
    "money": pa.array([decimal.Decimal(v) for v in ("1.00", "1.50", "1.51", "2.00")] + [None], pa.decimal128(10, 2)),
}

MIDNIGHT_MICROS = 1_577_836_800_000_000


def translation_cases():
    """(predicate node, the same predicate in SQL) pairs a translator must evaluate as the engine does."""
    from duckdb._sources.nodes import Binary, Col, In, Lit, Postfix, Unary

    def column(name):
        return Col((name,))

    return [
        (Binary(">", column("n"), Lit(5, "INTEGER")), "n > 5"),
        (
            Binary("AND", Binary(">=", column("n"), Lit(5, "INTEGER")), Binary("!=", column("s"), Lit("x", "VARCHAR"))),
            "n >= 5 AND s <> 'x'",
        ),
        (
            Binary("OR", Unary("NOT", Binary("<", column("n"), Lit(5, "INTEGER"))), Postfix("IS NULL", column("d"))),
            "NOT (n < 5) OR d IS NULL",
        ),
        (In(column("n"), [Lit(1, "INTEGER"), Lit(2, "INTEGER")]), "n IN (1, 2)"),
        (
            Binary("AND", Binary(">=", column("n"), Lit(1, "INTEGER")), Binary("<=", column("n"), Lit(2, "INTEGER"))),
            "n BETWEEN 1 AND 2",
        ),
        (Binary(">", column("ts"), Lit(datetime.datetime(2020, 1, 1), "TIMESTAMP")), "ts > TIMESTAMP '2020-01-01'"),
        (Binary(">", column("f"), Lit(1.0, "DOUBLE")), "f > 1.0"),
        (Binary(">=", column("f"), Lit(1.0, "DOUBLE")), "f >= 1.0"),
        (Binary("<", column("f"), Lit(1.0, "DOUBLE")), "f < 1.0"),
        (Binary("=", column("f"), Lit(1.0, "DOUBLE")), "f = 1.0"),
        (Binary(">", column("money"), Lit(decimal.Decimal("1.50"), "DECIMAL(10,2)")), "money > 1.50"),
        (
            Binary(">", column("zoned"), Lit(MIDNIGHT_MICROS + 3_600_000_000, "TIMESTAMP WITH TIME ZONE")),
            "zoned > TIMESTAMPTZ '2020-01-01 01:00:00+00'",
        ),
    ]


def refused_cases():
    """Predicates no translator may accept: the constant's type is not the column's, or the shape is not exact."""
    from duckdb._sources.nodes import Binary, Col, In, Lit

    def column(name):
        return Col((name,))

    return [
        Binary(">", column("n"), Lit(5.5, "DOUBLE")),
        Binary(">", column("n"), Lit("5", "VARCHAR")),
        Binary(">", column("s"), Lit(5, "INTEGER")),
        Binary(">", column("big"), Lit(True, "BOOLEAN")),
        Binary(">", column("n"), Lit(5_000_000_000, "BIGINT")),
        Binary(">", column("zoned"), Lit(datetime.datetime(2020, 1, 1), "TIMESTAMP")),
        Binary(">", column("ts"), Lit(MIDNIGHT_MICROS, "TIMESTAMP WITH TIME ZONE")),
        Binary(">", column("missing"), Lit(1, "INTEGER")),
        Binary(">", column("n"), column("big")),
        In(column("n"), [Lit(1, "INTEGER"), Lit(None, "INTEGER")]),
    ]


@pytest.fixture
def engine_rows(con):
    """The ids of TRANSLATION_DATA's rows a SQL predicate keeps, as the engine evaluates it."""
    table = pa.table({"id": list(range(5)), **TRANSLATION_DATA})
    native(con, "translated", table)

    def keep(where):
        return [row[0] for row in rows(con, f"SELECT id FROM translated WHERE {where} ORDER BY id")]

    return table, keep


class TestArrowTranslation:
    def test_each_form_selects_the_rows_the_engine_selects(self, sources, engine_rows):
        from duckdb._sources.to_arrow import to_arrow

        table, keep = engine_rows
        for node, where in translation_cases():
            kept = table.filter(to_arrow(node, table.schema)).column("id").to_pylist()
            assert kept == keep(where), where

    def test_the_rest_is_refused(self, sources):
        from duckdb._sources.nodes import Untranslatable
        from duckdb._sources.to_arrow import to_arrow

        arrow_schema = pa.table(TRANSLATION_DATA).schema
        for node in refused_cases():
            with pytest.raises(Untranslatable):
                to_arrow(node, arrow_schema)


class TestPolarsTranslation:
    def test_each_form_selects_the_rows_the_engine_selects(self, sources, engine_rows):
        pl = pytest.importorskip("polars")
        from duckdb._sources.to_polars import to_polars

        table, keep = engine_rows
        frame = pl.from_arrow(table)
        for node, where in translation_cases():
            kept = frame.filter(to_polars(node, frame.schema))["id"].to_list()
            assert kept == keep(where), where

    def test_the_rest_is_refused(self, sources):
        pl = pytest.importorskip("polars")
        from duckdb._sources.nodes import Untranslatable
        from duckdb._sources.to_polars import to_polars

        frame_schema = pl.from_arrow(pa.table(TRANSLATION_DATA)).schema
        for node in refused_cases():
            with pytest.raises(Untranslatable):
                to_polars(node, frame_schema)

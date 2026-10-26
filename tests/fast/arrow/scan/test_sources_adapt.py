"""The source adapters: which source `adapt` picks, and what each source answers and streams, without a scan."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from unittest import mock

import pytest

from duckdb._sources import ArrowScanSource, Source, adapt
from duckdb._sources.arrow import ArrowArraySource, ArrowCapsuleSource, ArrowStreamSource
from duckdb._sources.nodes import Binary, Col, In, Lit, Postfix, Unary

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")

from duckdb._sources.pyarrow import (  # noqa: E402
    DatasetSource,
    PyArrowArraySource,
    PyArrowDatasetSource,
    PyArrowReaderSource,
    PyArrowScannerSource,
    PyArrowStreamSource,
    PyArrowTableSource,
    ScannerSource,
)

GREATER = Binary(">", Col(("n",)), Lit(1, "BIGINT"))


@pytest.fixture
def numbers():
    return pa.table({"n": [0, 1, 2, 3], "s": ["a", "b", "c", "d"], "f": [0.5, 1.5, 2.5, 3.5]})


@pytest.fixture
def pl():
    return pytest.importorskip("polars")


@pytest.fixture
def pd():
    return pytest.importorskip("pandas")


def read(stream):
    """The table a stream capsule, or a (schema, array) capsule pair, holds."""
    if isinstance(stream, tuple):
        array = pa.Array._import_from_c_capsule(*stream)
        return pa.Table.from_struct_array(array) if pa.types.is_struct(array.type) else pa.table({"": array})
    return pa.RecordBatchReader._import_from_c_capsule(stream).read_all()


def schema_of(source):
    return pa.Schema._import_from_c_capsule(source.__arrow_c_schema__())


def capsule_name(capsule):
    return repr(capsule).split('"')[1]


class Exporter:
    """A third party object exporting a pyarrow table's stream."""

    def __init__(self, table) -> None:
        self.table = table

    def __arrow_c_stream__(self, requested_schema=None) -> object:
        return self.table.__arrow_c_stream__(requested_schema)


class DuckDataset:
    """A third party object with the interface of a pyarrow Dataset."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.schema = inner.schema

    def scanner(self, columns=None, filter=None):
        return self.inner.scanner(columns=columns, filter=filter)


class ExportOnlySchema:
    """A schema object offering only the older `_export_to_c`, as some Arrow libraries still do."""

    def __init__(self, schema) -> None:
        self.schema = schema

    def _export_to_c(self, address) -> None:
        self.schema._export_to_c(address)


class DuckScanner:
    """A third party object with the interface of a pyarrow Scanner."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.projected_schema = inner.projected_schema

    def to_reader(self):
        return self.inner.to_reader()


class TestClassification:
    def test_tables_and_batches_are_read_repeatedly(self, numbers):
        for obj in (numbers, numbers.to_batches()[0]):
            source = adapt(obj)
            assert type(source) is PyArrowTableSource
            assert source.obj is obj
            assert not source.one_shot

    def test_readers_and_capsules_are_streams(self, numbers):
        cases = (
            (pa.RecordBatchReader.from_stream(numbers), PyArrowReaderSource),
            (numbers.__arrow_c_stream__(), ArrowCapsuleSource),
        )
        for obj, kind in cases:
            source = adapt(obj)
            assert type(source) is kind
            assert source.obj is obj
            assert source.one_shot

    def test_every_capsule_goes_to_the_scan_which_checks_its_kind(self, numbers):
        for capsule in (numbers.schema.__arrow_c_schema__(), pa.array([1]).__arrow_c_array__()[1]):
            assert type(adapt(capsule)) is ArrowCapsuleSource

    def test_a_class_named_like_a_capsule_is_not_one(self, numbers):
        class PyCapsule(Exporter):
            pass

        source = adapt(PyCapsule(numbers))
        assert type(source) is ArrowStreamSource
        assert not source.one_shot

    def test_a_class_named_like_a_reader_is_not_one(self, numbers):
        class RecordBatchReader(Exporter):
            pass

        source = adapt(RecordBatchReader(numbers))
        assert type(source) is ArrowStreamSource
        assert not source.one_shot
        with pytest.raises(TypeError, match="RecordBatchReader is none of these"):
            adapt(type("RecordBatchReader", (), {})())

    def test_a_class_claiming_a_library_module_is_judged_by_identity(self, numbers):
        for module in ("pyarrow.lib", "polars"):
            forged = type("Table", (Exporter,), {"__module__": module})
            source = adapt(forged(numbers))
            assert type(source) is ArrowStreamSource
            assert source.schema_is_stable is False

    def test_arrays_and_chunked_arrays(self):
        array = pa.array([1, 2])
        assert type(adapt(array)) is PyArrowArraySource
        assert type(adapt(pa.array([{"a": 1}]))) is PyArrowArraySource
        assert type(adapt(pa.chunked_array([array, array]))) is PyArrowStreamSource

    def test_datasets_and_scanners(self, numbers):
        dataset = ds.dataset(numbers)
        assert type(adapt(dataset)) is PyArrowDatasetSource
        assert type(adapt(dataset.scanner())) is PyArrowScannerSource
        assert type(adapt(DuckDataset(dataset))) is DatasetSource
        assert type(adapt(DuckScanner(dataset.scanner()))) is ScannerSource

    def test_a_fragment_is_not_a_dataset(self, numbers, tmp_path):
        ds.write_dataset(numbers, tmp_path, format="parquet")
        fragment = next(iter(ds.dataset(tmp_path).get_fragments()))
        with pytest.raises(TypeError, match="none of these"):
            adapt(fragment)

    def test_polars_objects(self, pl, numbers):
        frame = pl.from_arrow(numbers)
        source = adapt(frame)
        assert type(source) is PyArrowTableSource
        assert isinstance(source.obj, pa.Table)
        assert source.obj.equals(frame.to_arrow())
        from duckdb._sources.polars import LazyFrameSource, PolarsStreamSource

        assert type(adapt(frame.lazy())) is LazyFrameSource
        assert type(adapt(frame["n"])) is PolarsStreamSource

    def test_a_subclass_of_a_polars_frame_is_a_frame(self, pl):
        class Mine(pl.DataFrame):
            pass

        assert type(adapt(Mine({"a": [1]}))) is PyArrowTableSource

    def test_a_polars_frame_of_python_objects_is_refused(self, pl):
        frame = pl.DataFrame({"o": pl.Series([object()], dtype=pl.Object)})
        with pytest.raises(TypeError, match="the polars column 'o' holds Python objects"):
            adapt(frame)
        with pytest.raises(TypeError, match="the polars column 'o' holds Python objects"):
            adapt(frame["o"])
        source = adapt(frame.lazy())
        with pytest.raises(TypeError, match="the polars column 'o' holds Python objects"):
            source.__arrow_c_schema__()

    def test_pandas_objects_export_through_the_interface(self, pd):
        for obj in (pd.Series([1, 2]), pd.DataFrame({"a": [1, 2]})):
            source = adapt(obj)
            assert type(source) is ArrowStreamSource

    def test_a_class_without_a_module_is_judged_by_its_exports(self, numbers):
        nameless = type("Nameless", (Exporter,), {"__module__": None})
        source = adapt(nameless(numbers))
        assert type(source) is ArrowStreamSource
        assert source.rows() is None

    def test_a_source_passes_through(self, numbers):
        source = ArrowScanSource(numbers)
        assert adapt(source) is source

    @pytest.mark.parametrize("obj", [[1, 2, 3], {"a": [1]}, "abc", object(), 42])
    def test_anything_else_is_refused(self, obj):
        with pytest.raises(TypeError, match=f"{type(obj).__name__} is none of these"):
            adapt(obj)

    def test_a_schema_alone_is_not_a_dataset(self):
        class OnlySchema:
            schema = pa.schema([("a", pa.int64())])

        with pytest.raises(TypeError, match="OnlySchema is none of these"):
            adapt(OnlySchema())


class TestOneShot:
    def test_only_readers_and_capsules_are_read_once(self, pl, numbers):
        dataset = ds.dataset(numbers)
        frame = pl.from_arrow(numbers)
        repeatable = (
            numbers,
            numbers.to_batches()[0],
            dataset,
            dataset.scanner(),
            pa.array([1]),
            pa.chunked_array([[1]]),
            frame,
            frame.lazy(),
            frame["n"],
            Exporter(numbers),
            DuckDataset(dataset),
            DuckScanner(dataset.scanner()),
        )
        for obj in repeatable:
            assert adapt(obj).one_shot is False, type(obj)
        for obj in (pa.RecordBatchReader.from_stream(numbers), numbers.__arrow_c_stream__()):
            assert adapt(obj).one_shot is True, type(obj)


class TestSchemaIsStable:
    def test_library_objects_datasets_scanners_and_readers_keep_their_schema(self, pl, numbers):
        dataset = ds.dataset(numbers)
        frame = pl.from_arrow(numbers)
        stable = (
            numbers,
            numbers.to_batches()[0],
            pa.RecordBatchReader.from_stream(numbers),
            numbers.__arrow_c_stream__(),
            dataset,
            dataset.scanner(),
            pa.array([1]),
            pa.chunked_array([[1]]),
            frame,
            frame.lazy(),
            frame["n"],
        )
        for obj in stable:
            assert adapt(obj).schema_is_stable is True, type(obj)

    def test_a_third_party_exporter_may_change_its_schema(self, pd, numbers):
        class Array:
            def __arrow_c_array__(self, requested_schema=None) -> object:
                return pa.array([1]).__arrow_c_array__()

        dataset = ds.dataset(numbers)
        third_party = (
            Exporter(numbers),
            Array(),
            pd.DataFrame({"a": [1]}),
            DuckDataset(dataset),
            DuckScanner(dataset.scanner()),
        )
        for obj in third_party:
            assert adapt(obj).schema_is_stable is False, type(obj)
        assert Source.schema_is_stable is False

    def test_a_duck_typed_dataset_is_read_anew(self, numbers):
        wrapper = DuckDataset(ds.dataset(numbers))
        source = adapt(wrapper)
        assert schema_of(source).equals(numbers.schema)
        other = pa.table({"y": ["a", "b"]})
        wrapper.inner = ds.dataset(other)
        wrapper.schema = other.schema
        assert schema_of(source).equals(other.schema)
        stream, projected = source.stream([0], ())
        assert projected
        assert read(stream).to_pydict() == {"y": ["a", "b"]}

    def test_a_pyarrow_dataset_keeps_its_first_schema(self, numbers):
        source = adapt(ds.dataset(numbers))
        assert source.declared_schema() is source.declared_schema()


class TestPullUnderGil:
    def test_no_source_asks_for_the_gil(self, pl, numbers):
        """A producer that reaches Python takes the GIL itself, so the scan pulls every stream without it."""
        dataset = ds.dataset(numbers)
        frame = pl.from_arrow(numbers)

        class Array:
            def __arrow_c_array__(self, requested_schema=None) -> object:
                return pa.array([1]).__arrow_c_array__()

        for obj in (
            numbers,
            numbers.to_batches()[0],
            pa.RecordBatchReader.from_stream(numbers),
            dataset,
            dataset.scanner(),
            pa.array([1]),
            pa.chunked_array([[1]]),
            frame,
            frame.lazy(),
            frame["n"],
            numbers.__arrow_c_stream__(),
            Exporter(numbers),
            Array(),
            DuckDataset(dataset),
            DuckScanner(dataset.scanner()),
        ):
            assert adapt(obj).pull_under_gil is False, type(obj)

    def test_a_source_may_opt_in(self, numbers):
        class Holding(ArrowScanSource):
            pull_under_gil = True

        assert Holding(numbers).pull_under_gil is True


class TestRows:
    def test_in_memory_objects_know_their_rows(self, pl, pd, numbers):
        assert adapt(numbers).rows() == 4
        assert adapt(numbers.to_batches()[0]).rows() == 4
        assert adapt(pa.array([1, 2, 3])).rows() == 3
        assert adapt(pa.chunked_array([[1, 2], [3]])).rows() == 3
        assert adapt(pl.from_arrow(numbers)).rows() == 4
        assert adapt(pl.Series([1, 2])).rows() == 2
        assert adapt(pd.Series([1, 2])).rows() == 2

    def test_streams_plans_and_datasets_do_not(self, pl, numbers):
        dataset = ds.dataset(numbers)
        for obj in (
            pa.RecordBatchReader.from_stream(numbers),
            numbers.__arrow_c_stream__(),
            dataset,
            dataset.scanner(),
            pl.from_arrow(numbers).lazy(),
        ):
            assert adapt(obj).rows() is None, type(obj)

    def test_a_third_party_length_is_not_trusted(self, numbers):
        class Sized(Exporter):
            def __len__(self) -> int:
                return 0

        assert adapt(Sized(numbers)).rows() is None
        for module in ("pyarrow.lib", "polars.series.series", "pandas.core.frame"):
            forged = type("Table", (Sized,), {"__module__": module})
            assert adapt(forged(numbers)).rows() is None, module

    def test_a_pandas_frame_knows_its_rows(self, pd):
        assert adapt(pd.DataFrame({"a": [1, 2, 3]})).rows() == 3


class TestStreams:
    def test_a_table_narrows_by_position(self, numbers):
        source = adapt(numbers)
        stream, projected = source.stream(None, ())
        assert capsule_name(stream) == "arrow_array_stream"
        assert not projected
        assert read(stream).equals(numbers)
        stream, projected = source.stream([2, 0], ())
        assert projected
        assert read(stream).column_names == ["f", "n"]
        stream, projected = source.stream([], ())
        assert projected
        assert read(stream).num_rows == 4

    def test_a_batch_narrows_by_position(self, numbers):
        stream, projected = adapt(numbers.to_batches()[0]).stream([1], ())
        assert projected
        assert read(stream).column("s").to_pylist() == ["a", "b", "c", "d"]

    def test_streams_and_arrays_return_every_column(self, numbers):
        cases = [
            pa.RecordBatchReader.from_stream(numbers),
            numbers.__arrow_c_stream__(),
            Exporter(numbers),
            pa.chunked_array([[1, 2]]),
        ]
        for obj in cases:
            stream, projected = adapt(obj).stream([0], ())
            assert not projected
            assert capsule_name(stream) == "arrow_array_stream"
        capsule = numbers.__arrow_c_stream__()
        assert adapt(capsule).stream(None, ())[0] is capsule

    def test_an_array_is_one_batch(self):
        struct = pa.array([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        pair, projected = adapt(struct).stream([1], ())
        assert not projected
        assert [capsule_name(capsule) for capsule in pair] == ["arrow_schema", "arrow_array"]
        assert read(pair).column_names == ["a", "b"]

    def test_a_dataset_narrows_by_name_and_applies_filters(self, numbers):
        source = adapt(ds.dataset(numbers))
        stream, projected = source.stream([1], (GREATER,))
        assert projected
        assert read(stream).to_pydict() == {"s": ["c", "d"]}
        stream, projected = source.stream(None, (GREATER, Binary("<", Col(("f",)), Lit(3.0, "DOUBLE"))))
        assert not projected
        assert read(stream).column("n").to_pylist() == [2]

    def test_a_dataset_is_scanned_through_its_type(self, numbers):
        calls = []

        class Counting(DuckDataset):
            def scanner(self, columns=None, filter=None):
                calls.append(columns)
                return super().scanner(columns=columns, filter=filter)

        dataset = Counting(ds.dataset(numbers))
        dataset.scanner = None
        stream, projected = adapt(dataset).stream([0], ())
        assert projected
        assert read(stream).column_names == ["n"]
        assert calls == [["n"]]

    def test_a_scanner_keeps_its_own_projection_and_filter(self, numbers):
        scanner = ds.dataset(numbers).scanner(columns=["s"], filter=ds.field("n") > 2)
        source = adapt(scanner)
        assert schema_of(source).names == ["s"]
        for _ in range(2):
            stream, projected = source.stream([0], ())
            assert not projected
            assert read(stream).to_pydict() == {"s": ["d"]}

    def test_a_lazy_frame_filters_narrows_and_collects(self, pl, numbers):
        source = adapt(pl.from_arrow(numbers).lazy())
        stream, projected = source.stream([1, 0], (GREATER,))
        assert projected
        assert read(stream).to_pydict() == {"s": ["c", "d"], "n": [2, 3]}
        stream, projected = source.stream(None, ())
        assert not projected
        assert read(stream).num_rows == 4

    def test_a_lazy_frame_narrows_by_position_whatever_the_names(self, pl):
        frame = pl.DataFrame({"id": [0, 1], "^x$": [0, 2], "x": [2, 0], "*": [5, 6]})
        source = adapt(frame.lazy())
        stream, projected = source.stream([3, 1], ())
        assert projected
        assert read(stream).to_pydict() == {"*": [5, 6], "^x$": [0, 2]}
        predicate = Binary(">", Col(("^x$",)), Lit(1, "BIGINT"))
        assert source.accepts(predicate)
        stream, projected = source.stream([0], (predicate,))
        assert projected
        assert read(stream).to_pydict() == {"id": [1]}

    def test_a_lazy_frame_asked_for_no_columns_keeps_its_rows(self, pl, numbers):
        stream, projected = adapt(pl.from_arrow(numbers).lazy()).stream([], (GREATER,))
        assert not projected
        assert read(stream).num_rows == 2


class TestSchemaHook:
    def test_a_chunked_array_says_its_type_without_a_stream(self):
        source = adapt(pa.chunked_array([[1, 2], [3]]))
        assert "__arrow_c_schema__" in source.__dict__
        assert pa.Field._import_from_c_capsule(source.__arrow_c_schema__()).type == pa.int64()

    def test_in_memory_objects_and_readers_say_their_schema_without_a_stream(self, numbers):
        for obj in (numbers, numbers.to_batches()[0], ds.dataset(numbers)):
            assert schema_of(adapt(obj)).equals(numbers.schema)
        field = pa.Field._import_from_c_capsule(adapt(pa.array([1.5])).__arrow_c_schema__())
        assert (field.name, field.type) == ("", pa.float64())

    def test_a_reader_schema_does_not_consume_it(self, numbers):
        reader = pa.RecordBatchReader.from_stream(numbers)
        source = adapt(reader)
        assert schema_of(source).equals(numbers.schema)
        stream, _ = source.stream(None, ())
        assert read(stream).num_rows == 4

    def test_a_third_party_object_offers_its_schema_hook(self, numbers):
        class WithHook(Exporter):
            def __arrow_c_schema__(self) -> object:
                return self.table.schema.__arrow_c_schema__()

        class WithSchema(Exporter):
            @property
            def schema(self):
                return self.table.schema

        for obj in (WithHook(numbers), WithSchema(numbers)):
            assert schema_of(adapt(obj)).equals(numbers.schema)
        assert not hasattr(adapt(Exporter(numbers)), "__arrow_c_schema__")
        assert not hasattr(adapt(numbers.__arrow_c_stream__()), "__arrow_c_schema__")

    def test_an_array_exporter_offers_its_own_schema_hook(self):
        class ArrayWithSchema:
            def __arrow_c_array__(self, requested_schema=None) -> object:
                raise AssertionError

            def __arrow_c_schema__(self) -> object:
                return pa.int64().__arrow_c_schema__()

        source = adapt(ArrayWithSchema())
        assert type(source) is ArrowArraySource
        field = pa.Field._import_from_c_capsule(source.__arrow_c_schema__())
        assert field.type == pa.int64()

    def test_a_schema_exporting_only_through_export_to_c_is_read(self, numbers):
        class Mine(ds.InMemoryDataset):
            @property
            def schema(self):
                return ExportOnlySchema(super().schema)

        source = adapt(Mine(numbers))
        assert type(source) is PyArrowDatasetSource
        assert schema_of(source).equals(numbers.schema)
        assert source.accepts(GREATER)
        stream, projected = source.stream([1], (GREATER,))
        assert projected
        assert read(stream).to_pydict() == {"s": ["c", "d"]}

        class WithSchema(Exporter):
            @property
            def schema(self):
                return ExportOnlySchema(self.table.schema)

        reader = pa.RecordBatchReader.from_stream(numbers)
        stream_source = adapt(WithSchema(reader))
        assert schema_of(stream_source).equals(numbers.schema)
        assert read(stream_source.stream(None, ())[0]).num_rows == 4

        class ExportOnlyScanner(DuckScanner):
            def __init__(self, inner) -> None:
                super().__init__(inner)
                self.projected_schema = ExportOnlySchema(inner.projected_schema)

        scanner = ds.dataset(numbers).scanner(columns=["f"])
        assert schema_of(adapt(ExportOnlyScanner(scanner))).names == ["f"]

    @pytest.mark.parametrize(
        "kind",
        [
            pytest.param(pa.int64(), id="plain"),
            pytest.param(pa.timestamp("us", tz="UTC"), id="parametrized"),
            pytest.param(pa.struct([("a", pa.int64()), ("b", pa.string())]), id="struct"),
        ],
    )
    def test_an_array_type_exporting_only_through_export_to_c(self, kind):
        class Array:
            type = ExportOnlySchema(kind)

            def __arrow_c_array__(self, requested_schema=None) -> object:
                raise AssertionError

        capsule = adapt(Array()).__arrow_c_schema__()
        if pa.types.is_struct(kind):
            assert pa.Schema._import_from_c_capsule(capsule) == pa.schema(list(kind))
        else:
            assert pa.Field._import_from_c_capsule(capsule).type == kind

    @pytest.mark.parametrize(
        "field",
        [
            pytest.param(pa.field("named", pa.int64(), nullable=False, metadata={b"k": b"v"}), id="plain"),
            pytest.param(
                pa.field("zoned", pa.timestamp("ms", tz="Europe/Amsterdam"), metadata={b"a": b"1"}), id="zoned"
            ),
        ],
    )
    def test_an_exported_field_keeps_its_name_nullability_and_metadata(self, field):
        class Array:
            type = ExportOnlySchema(field)

            def __arrow_c_array__(self, requested_schema=None) -> object:
                raise AssertionError

        read_back = pa.Field._import_from_c_capsule(adapt(Array()).__arrow_c_schema__())
        assert read_back.equals(field, check_metadata=True)
        assert (read_back.name, read_back.nullable, read_back.metadata) == (field.name, field.nullable, field.metadata)

    def test_an_own_hook_behind_a_property_is_read_at_every_call(self, numbers):
        class PropertyExporter:
            def __init__(self) -> None:
                self.table = numbers

            @property
            def __arrow_c_schema__(self) -> object:
                return self.table.schema.__arrow_c_schema__

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                return self.table.__arrow_c_stream__()

        exporter = PropertyExporter()
        source = adapt(exporter)
        assert schema_of(source).equals(numbers.schema)
        exporter.table = pa.table({"y": ["a"]})
        assert schema_of(source).names == ["y"]

    @pytest.mark.parametrize("wrap", [lambda schema: schema, ExportOnlySchema], ids=["capsule", "export-to-c"])
    def test_a_replaced_schema_attribute_is_read_at_every_call(self, numbers, wrap):
        class Changing:
            def __init__(self) -> None:
                self.schema = wrap(numbers.schema)

            def __arrow_c_stream__(self, requested_schema=None) -> object:
                raise AssertionError

        exporter = Changing()
        source = adapt(exporter)
        assert schema_of(source).equals(numbers.schema)
        exporter.schema = wrap(pa.schema([("other", pa.string())]))
        assert schema_of(source).equals(pa.schema([("other", pa.string())]))
        del exporter.schema
        with pytest.raises(AttributeError, match="schema"):
            source.__arrow_c_schema__()

    @pytest.mark.parametrize("wrap", [lambda kind: kind, ExportOnlySchema], ids=["capsule", "export-to-c"])
    def test_a_replaced_type_attribute_is_read_at_every_call(self, wrap):
        class Changing:
            def __init__(self) -> None:
                self.type = wrap(pa.int64())

            def __arrow_c_array__(self, requested_schema=None) -> object:
                raise AssertionError

        exporter = Changing()
        source = adapt(exporter)
        assert pa.Field._import_from_c_capsule(source.__arrow_c_schema__()).type == pa.int64()
        exporter.type = wrap(pa.string())
        assert pa.Field._import_from_c_capsule(source.__arrow_c_schema__()).type == pa.string()
        del exporter.type
        with pytest.raises(AttributeError, match="type"):
            source.__arrow_c_schema__()

    def test_a_lazy_frame_declares_the_arrow_types_it_collects_to(self, pl):
        frame = pl.DataFrame(
            {"s": ["a", None], "c": pl.Series(["x", "y"], dtype=pl.Categorical()), "d": [1.5, 2.5]}
        ).lazy()
        source = adapt(frame)
        declared = schema_of(source)
        stream, _ = source.stream(None, ())
        assert declared.equals(read(stream).schema)
        assert declared.field("s").type == pa.large_string()


def native_ids(create, rows, predicate):
    import duckdb

    with duckdb.connect() as con:
        con.execute(create)
        con.executemany(f"INSERT INTO t VALUES ({', '.join('?' * len(rows[0]))})", [list(row) for row in rows])
        return [row[0] for row in con.execute(f"SELECT id FROM t WHERE {predicate} ORDER BY id").fetchall()]


def f(op, value):
    return Binary(op, Col(("f",)), Lit(value, "DOUBLE"))


def i(op, value):
    return Binary(op, Col(("i",)), Lit(value, "INTEGER"))


NAN = float("nan")
PARQUET_ROWS = [(0, NAN, 1), (1, 1.0, 2), (2, 2.0, 3), (3, 3.0, 4), (4, None, 5), (5, NAN, 6)]


class TestParquetStatistics:
    """Row groups are pruned by minimum and maximum statistics, which leave NaN out, before any filter runs."""

    @pytest.fixture(params=["dataset", "scan_parquet", "scan_pyarrow_dataset"])
    def parquet(self, request, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        table = pa.table(
            {
                "id": pa.array([row[0] for row in PARQUET_ROWS], pa.int32()),
                "f": pa.array([row[1] for row in PARQUET_ROWS], pa.float64()),
                "i": pa.array([row[2] for row in PARQUET_ROWS], pa.int32()),
            }
        )
        path = tmp_path / "t.parquet"
        pq.write_table(table, path, row_group_size=2)
        if request.param == "dataset":
            return ds.dataset(path)
        pl = pytest.importorskip("polars")
        if request.param == "scan_parquet":
            return pl.scan_parquet(path)
        return pl.scan_pyarrow_dataset(ds.dataset(path))

    @pytest.mark.parametrize(
        ("node", "predicate", "accepted"),
        [
            pytest.param(f("=", 2.0), "f = 2.0", True, id="equal"),
            pytest.param(f("<", 2.0), "f < 2.0", True, id="less"),
            pytest.param(f("<=", 2.0), "f <= 2.0", True, id="less-or-equal"),
            pytest.param(In(Col(("f",)), [Lit(1.0, "DOUBLE"), Lit(3.0, "DOUBLE")]), "f IN (1.0, 3.0)", True, id="in"),
            pytest.param(Postfix("IS NULL", Col(("f",))), "f IS NULL", True, id="is-null"),
            pytest.param(Postfix("IS NOT NULL", Col(("f",))), "f IS NOT NULL", True, id="is-not-null"),
            pytest.param(Binary("AND", f("<", 3.0), i(">", 1)), "f < 3.0 AND i > 1", True, id="and"),
            pytest.param(Binary("OR", f("=", 1.0), f("<", 3.0)), "f = 1.0 OR f < 3.0", True, id="or"),
            pytest.param(Unary("NOT", i("=", 2)), "NOT (i = 2)", True, id="not-on-an-integer"),
            pytest.param(f(">", 1.5), "f > 1.5", False, id="greater"),
            pytest.param(f(">=", 2.0), "f >= 2.0", False, id="greater-or-equal"),
            pytest.param(f("!=", 2.0), "f <> 2.0", False, id="not-equal"),
            pytest.param(Unary("NOT", In(Col(("f",)), [Lit(1.0, "DOUBLE")])), "NOT (f IN (1.0))", False, id="not-in"),
            pytest.param(Unary("NOT", f("=", 1.0)), "NOT (f = 1.0)", False, id="not"),
            pytest.param(Unary("NOT", Postfix("IS NULL", Col(("f",)))), "NOT (f IS NULL)", False, id="not-is-null"),
            pytest.param(Binary("OR", f("=", 1.0), f(">", 2.0)), "f = 1.0 OR f > 2.0", False, id="or-with-greater"),
            pytest.param(Binary("AND", i(">", 0), f("!=", 2.0)), "i > 0 AND f <> 2.0", False, id="and-with-not-equal"),
        ],
    )
    def test_a_predicate_nan_may_satisfy_is_refused(self, parquet, node, predicate, accepted):
        expected = native_ids("CREATE TABLE t (id INTEGER, f DOUBLE, i INTEGER)", PARQUET_ROWS, predicate)
        source = adapt(parquet)
        assert source.accepts(node) is accepted
        if accepted:
            stream, _ = source.stream(None, (node,))
            assert sorted(read(stream).column("id").to_pylist()) == expected
        else:
            assert {0, 5} & set(expected), "a refused shape is one NaN rows satisfy"


class TestAccepts:
    def test_only_a_pyarrow_dataset_and_a_lazy_frame_accept(self, pl, numbers):
        dataset = ds.dataset(numbers)
        assert adapt(dataset).accepts(GREATER)
        assert adapt(pl.from_arrow(numbers).lazy()).accepts(GREATER)
        refusing = (
            numbers,
            numbers.to_batches()[0],
            pa.RecordBatchReader.from_stream(numbers),
            numbers.__arrow_c_stream__(),
            Exporter(numbers),
            dataset.scanner(),
            DuckDataset(dataset),
            DuckScanner(dataset.scanner()),
            pl.from_arrow(numbers),
        )
        for obj in refusing:
            assert adapt(obj).accepts(GREATER) is False, type(obj)

    def test_a_chain_deeper_than_python_recursion_is_refused(self, pl, numbers):
        node = Binary("=", Col(("n",)), Lit(0, "BIGINT"))
        for i in range(1, 1500):
            node = Binary("OR", node, Binary("=", Col(("n",)), Lit(i, "BIGINT")))
        assert adapt(ds.dataset(numbers)).accepts(node) is False
        assert adapt(pl.from_arrow(numbers).lazy()).accepts(node) is False

    def test_a_predicate_the_translator_refuses_is_refused(self, pl, numbers):
        wrong_type = Binary(">", Col(("n",)), Lit(1, "INTEGER"))
        assert not adapt(ds.dataset(numbers)).accepts(wrong_type)
        assert not adapt(pl.from_arrow(numbers).lazy()).accepts(wrong_type)

    @pytest.mark.parametrize(
        ("other", "accepted"),
        [
            pytest.param(pa.array([{"v": "a"}, {"v": "b"}], pa.struct([("v", pa.string_view())])), False, id="struct"),
            pytest.param(pa.array([[b"a"], [b"b"]], pa.list_(pa.binary_view())), False, id="list"),
            pytest.param(
                pa.StructArray.from_arrays([pa.array(["a", "b"]).dictionary_encode()], ["d"]), False, id="dictionary"
            ),
            pytest.param(pa.array(["a", "b"]).dictionary_encode(), True, id="dictionary-column-control"),
        ],
    )
    def test_a_dataset_beside_a_nested_column(self, other, accepted):
        """A column pyarrow cannot filter refuses every predicate, and the rows still stream either way."""
        source = adapt(ds.dataset(pa.table({"n": [1, 2], "other": other})))
        assert source.accepts(GREATER) is accepted
        stream, projected = source.stream([0], ())
        assert projected
        assert read(stream).to_pydict() == {"n": [1, 2]}
        stream, projected = source.stream(None, ())
        assert not projected
        assert read(stream).column("other").to_pylist() == other.to_pylist()


def test_importing_the_sources_imports_no_data_library():
    code = textwrap.dedent(
        """
        import sys

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] in {"pyarrow", "polars", "pandas"}:
                    raise ImportError(name)

        sys.meta_path.insert(0, Block())
        import duckdb._sources
        import duckdb._sources.arrow
        import duckdb._sources.nodes
        import duckdb._sources.polars
        import duckdb._sources.pyarrow
        import duckdb._sources.to_arrow
        import duckdb._sources.to_polars

        class Exporter:
            def __arrow_c_stream__(self, requested_schema=None):
                raise AssertionError

        assert type(duckdb._sources.adapt(Exporter())).__name__ == "ArrowStreamSource"
        print(sorted(name for name in sys.modules if name.partition(".")[0] in {"pyarrow", "polars", "pandas"}))
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_no_call_imports_a_duckdb_module(pl, numbers):
    """A call that imported a duckdb module while duckdb is out of sys.modules would load the extension again."""
    import duckdb._sources

    predicate = GREATER
    with mock.patch.dict(sys.modules):
        for name in [name for name in sys.modules if name == "duckdb" or name.startswith("duckdb.")]:
            del sys.modules[name]
        for obj in (numbers, ds.dataset(numbers), pl.from_arrow(numbers).lazy(), pl.from_arrow(numbers)):
            source = duckdb._sources.adapt(obj)
            source.rows()
            if source.accepts(predicate):
                read(source.stream([0], (predicate,))[0])
            read(source.stream(None, ())[0])
            assert source.__arrow_c_schema__()
        assert predicate.fragment() == '("n" > 1)'
        assert not [name for name in sys.modules if name == "duckdb" or name.startswith("duckdb.")]


def test_a_source_is_a_plain_python_object():
    assert issubclass(ArrowScanSource, Source)
    assert ArrowScanSource.pull_under_gil is False
    assert Source.one_shot is False
    assert ArrowScanSource(object()).accepts(GREATER) is False
    assert Source(object()).rows() is None
    assert issubclass(ArrowArraySource, ArrowScanSource)

#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb_python/arrow/polars_filter_pushdown.hpp"
#include "duckdb_python/arrow/pyarrow_filter_pushdown.hpp"

#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb/function/table/arrow.hpp"

#include "duckdb/common/assert.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/main/client_config.hpp"

namespace duckdb {

void TransformDuckToArrowChunk(ArrowSchema &arrow_schema, ArrowArray &data, py::list &batches) {
	py::gil_assert();
	auto pyarrow_lib_module = py::module::import("pyarrow").attr("lib");
	auto batch_import_func = pyarrow_lib_module.attr("RecordBatch").attr("_import_from_c");
	batches.append(batch_import_func(reinterpret_cast<uint64_t>(&data), reinterpret_cast<uint64_t>(&arrow_schema)));
}

void VerifyArrowDatasetLoaded() {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (!import_cache.pyarrow.dataset() || !ModuleIsLoaded<PyarrowDatasetCacheItem>()) {
		throw InvalidInputException("Optional module 'pyarrow.dataset' is required to perform this action");
	}
}

py::object PythonTableArrowArrayStreamFactory::ProduceScanner(py::object &arrow_scanner, py::handle &arrow_obj_handle,
                                                              ArrowStreamParameters &parameters,
                                                              const ClientProperties &client_properties) {
	D_ASSERT(!py::isinstance<py::capsule>(arrow_obj_handle));
	ArrowSchemaWrapper schema;
	PythonTableArrowArrayStreamFactory::GetSchemaInternal(arrow_obj_handle, schema);
	ArrowTableSchema arrow_table;
	ArrowTableFunction::PopulateArrowTableSchema(*client_properties.client_context.get_mutable(), arrow_table,
	                                             schema.arrow_schema);

	auto filters = parameters.filters;
	auto &column_list = parameters.projected_columns.columns;
	auto &filter_to_col = parameters.projected_columns.filter_to_col;
	py::list projection_list = py::cast(column_list);

	bool has_filter = filters && filters->HasFilters();
	py::dict kwargs;
	if (!column_list.empty()) {
		kwargs["columns"] = projection_list;
	}

	if (has_filter) {
		auto filter = PyArrowFilterPushdown::TransformFilter(*filters, parameters.projected_columns.projection_map,
		                                                     filter_to_col, client_properties, arrow_table);
		if (!filter.is(py::none())) {
			kwargs["filter"] = filter;
		}
	}
	return arrow_scanner(arrow_obj_handle, **kwargs);
}

unique_ptr<ArrowArrayStreamWrapper> PythonTableArrowArrayStreamFactory::Produce(uintptr_t factory_ptr,
                                                                                ArrowStreamParameters &parameters) {
	py::gil_scoped_acquire acquire;
	auto factory = static_cast<PythonTableArrowArrayStreamFactory *>(reinterpret_cast<void *>(factory_ptr)); // NOLINT
	D_ASSERT(factory->arrow_object);
	py::handle arrow_obj_handle(factory->arrow_object);
	auto arrow_object_type = factory->cached_arrow_type;

	if (arrow_object_type == PyArrowObjectType::PolarsLazyFrame) {
		py::object lf = py::reinterpret_borrow<py::object>(arrow_obj_handle);

		auto filters = parameters.filters;
		bool filters_pushed = false;

		// Translate DuckDB filters to Polars expressions and push into the lazy plan
		if (filters && filters->HasFilters()) {
			try {
				auto filter_expr = PolarsFilterPushdown::TransformFilter(
				    *filters, parameters.projected_columns.projection_map, parameters.projected_columns.filter_to_col,
				    factory->client_properties);
				if (!filter_expr.is(py::none())) {
					lf = lf.attr("filter")(filter_expr);
					filters_pushed = true;
				}
			} catch (...) {
				// Fallback: DuckDB handles filtering post-scan
			}
		}

		// If no filters were pushed and we have a cached Arrow table, reuse it. This avoids re-reading from source and
		// re-converting on repeated unfiltered scans.
		py::object arrow_table;
		if (!filters_pushed && factory->cached_arrow_table.ptr() != nullptr) {
			arrow_table = factory->cached_arrow_table;
		} else {
			arrow_table = lf.attr("collect")().attr("to_arrow")();
			// Cache only unfiltered results (filtered results are partial)
			if (!filters_pushed) {
				factory->cached_arrow_table = arrow_table;
			}
		}

		// Apply column projection
		auto &column_list = parameters.projected_columns.columns;
		if (!column_list.empty()) {
			arrow_table = arrow_table.attr("select")(py::cast(column_list));
		}

		auto capsule_obj = arrow_table.attr("__arrow_c_stream__")();
		auto capsule = py::reinterpret_borrow<py::capsule>(capsule_obj);
		auto stream = capsule.get_pointer<struct ArrowArrayStream>();
		auto res = make_uniq<ArrowArrayStreamWrapper>();
		res->arrow_array_stream = *stream;
		stream->release = nullptr;
		return res;
	}

	if (arrow_object_type == PyArrowObjectType::PyCapsuleInterface || arrow_object_type == PyArrowObjectType::Table) {
		py::object capsule_obj = arrow_obj_handle.attr("__arrow_c_stream__")();
		auto capsule = py::reinterpret_borrow<py::capsule>(capsule_obj);
		auto stream = capsule.get_pointer<struct ArrowArrayStream>();
		if (!stream->release) {
			throw InvalidInputException(
			    "The __arrow_c_stream__() method returned a released stream. "
			    "If this object is single-use, implement __arrow_c_schema__() or expose a .schema attribute "
			    "with _export_to_c() so that DuckDB can extract the schema without consuming the stream.");
		}

		auto &import_cache_check = *DuckDBPyConnection::ImportCache();
		if (import_cache_check.pyarrow.dataset()) {
			// Tier A: full pushdown via pyarrow.dataset
			// Import as RecordBatchReader, feed through Scanner.from_batches for projection/filter pushdown.
			auto pyarrow_lib_module = py::module::import("pyarrow").attr("lib");
			auto import_func = pyarrow_lib_module.attr("RecordBatchReader").attr("_import_from_c");
			py::object reader = import_func(reinterpret_cast<uint64_t>(stream));
			// _import_from_c takes ownership of the stream; null out to prevent capsule double-free
			stream->release = nullptr;
			auto &import_cache = *DuckDBPyConnection::ImportCache();
			py::object arrow_batch_scanner = import_cache.pyarrow.dataset.Scanner().attr("from_batches");
			py::handle reader_handle = reader;
			auto scanner = ProduceScanner(arrow_batch_scanner, reader_handle, parameters, factory->client_properties);
			auto record_batches = scanner.attr("to_reader")();
			auto res = make_uniq<ArrowArrayStreamWrapper>();
			auto export_to_c = record_batches.attr("_export_to_c");
			export_to_c(reinterpret_cast<uint64_t>(&res->arrow_array_stream));
			return res;
		} else {
			// Tier B: no pyarrow.dataset, return raw stream (no pushdown)
			// DuckDB applies projection/filter post-scan via arrow_scan_dumb
			auto res = make_uniq<ArrowArrayStreamWrapper>();
			res->arrow_array_stream = *stream;
			stream->release = nullptr;
			return res;
		}
	}

	if (arrow_object_type == PyArrowObjectType::PyCapsule) {
		auto res = make_uniq<ArrowArrayStreamWrapper>();
		auto capsule = py::reinterpret_borrow<py::capsule>(arrow_obj_handle);
		auto stream = capsule.get_pointer<struct ArrowArrayStream>();
		if (!stream->release) {
			throw InvalidInputException("This ArrowArrayStream has already been consumed and cannot be scanned again.");
		}
		res->arrow_array_stream = *stream;
		stream->release = nullptr;
		return res;
	}

	// Scanner and Dataset: require pyarrow.dataset for pushdown
	VerifyArrowDatasetLoaded();
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	py::object scanner;
	py::object arrow_batch_scanner = import_cache.pyarrow.dataset.Scanner().attr("from_batches");
	switch (arrow_object_type) {
	case PyArrowObjectType::Scanner: {
		// If it's a scanner we have to turn it to a record batch reader, and then a scanner again since we can't stack
		// scanners on arrow Otherwise pushed-down projections and filters will disappear like tears in the rain
		auto record_batches = arrow_obj_handle.attr("to_reader")();
		scanner = ProduceScanner(arrow_batch_scanner, record_batches, parameters, factory->client_properties);
		break;
	}
	case PyArrowObjectType::Dataset: {
		py::object arrow_scanner = arrow_obj_handle.attr("__class__").attr("scanner");
		scanner = ProduceScanner(arrow_scanner, arrow_obj_handle, parameters, factory->client_properties);
		break;
	}
	default: {
		auto py_object_type = string(py::str(py::type::of(arrow_obj_handle).attr("__name__")));
		throw InvalidInputException("Object of type '%s' is not a recognized Arrow object", py_object_type);
	}
	}

	auto record_batches = scanner.attr("to_reader")();
	auto res = make_uniq<ArrowArrayStreamWrapper>();
	auto export_to_c = record_batches.attr("_export_to_c");
	export_to_c(reinterpret_cast<uint64_t>(&res->arrow_array_stream));
	return res;
}

void PythonTableArrowArrayStreamFactory::GetSchemaInternal(py::handle arrow_obj_handle, ArrowSchemaWrapper &schema) {
	// PyCapsule (from bare capsule Produce path)
	if (py::isinstance<py::capsule>(arrow_obj_handle)) {
		auto capsule = py::reinterpret_borrow<py::capsule>(arrow_obj_handle);
		auto stream = capsule.get_pointer<struct ArrowArrayStream>();
		if (!stream->release) {
			throw InvalidInputException("This ArrowArrayStream has already been consumed and cannot be scanned again.");
		}
		if (stream->get_schema(stream, &schema.arrow_schema)) {
			throw InvalidInputException("Failed to get Arrow schema from stream: %s",
			                            stream->get_last_error ? stream->get_last_error(stream) : "unknown error");
		}
		return;
	}

	// Scanner: use projected_schema; everything else (RecordBatchReader, Dataset): use .schema
	VerifyArrowDatasetLoaded();
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (py::isinstance(arrow_obj_handle, import_cache.pyarrow.dataset.Scanner())) {
		auto obj_schema = arrow_obj_handle.attr("projected_schema");
		obj_schema.attr("_export_to_c")(reinterpret_cast<uint64_t>(&schema.arrow_schema));
	} else {
		auto obj_schema = arrow_obj_handle.attr("schema");
		obj_schema.attr("_export_to_c")(reinterpret_cast<uint64_t>(&schema.arrow_schema));
	}
}

void PythonTableArrowArrayStreamFactory::GetSchema(uintptr_t factory_ptr, ArrowSchemaWrapper &schema) {
	auto factory = static_cast<PythonTableArrowArrayStreamFactory *>(reinterpret_cast<void *>(factory_ptr)); // NOLINT

	// Fast path: return cached schema without GIL or Python calls
	if (factory->schema_cached) {
		schema.arrow_schema = factory->cached_schema; // struct copy
		schema.arrow_schema.release = nullptr;        // non-owning copy
		return;
	}

	py::gil_scoped_acquire acquire;
	D_ASSERT(factory->arrow_object);
	py::handle arrow_obj_handle(factory->arrow_object);

	auto type = factory->cached_arrow_type;
	if (type == PyArrowObjectType::PolarsLazyFrame) {
		// head(0).collect().to_arrow() gives the Arrow-exported schema (e.g. large_string) without materializing data.
		// collect_schema() would give Polars-native types (e.g. string_view) that don't match the actual export.
		const auto empty_arrow = arrow_obj_handle.attr("head")(0).attr("collect")().attr("to_arrow")();
		const auto schema_capsule = empty_arrow.attr("schema").attr("__arrow_c_schema__")();
		const auto capsule = py::reinterpret_borrow<py::capsule>(schema_capsule);
		const auto arrow_schema = capsule.get_pointer<struct ArrowSchema>();
		factory->cached_schema = *arrow_schema;
		arrow_schema->release = nullptr;
		factory->schema_cached = true;
		schema.arrow_schema = factory->cached_schema;
		schema.arrow_schema.release = nullptr;
		return;
	}
	if (type == PyArrowObjectType::PyCapsuleInterface || type == PyArrowObjectType::Table) {
		// Get __arrow_c_schema__ if it exists
		if (py::hasattr(arrow_obj_handle, "__arrow_c_schema__")) {
			auto schema_capsule = arrow_obj_handle.attr("__arrow_c_schema__")();
			auto capsule = py::reinterpret_borrow<py::capsule>(schema_capsule);
			auto arrow_schema = capsule.get_pointer<struct ArrowSchema>();
			factory->cached_schema = *arrow_schema; // factory takes ownership
			arrow_schema->release = nullptr;
			factory->schema_cached = true;
			schema.arrow_schema = factory->cached_schema; // non-owning copy
			schema.arrow_schema.release = nullptr;
			return;
		}
		// Otherwise try to use .schema with _export_to_c
		if (py::hasattr(arrow_obj_handle, "schema")) {
			auto obj_schema = arrow_obj_handle.attr("schema");
			if (py::hasattr(obj_schema, "_export_to_c")) {
				obj_schema.attr("_export_to_c")(reinterpret_cast<uint64_t>(&schema.arrow_schema));
				return;
			}
		}
		// Fallback: create a temporary stream just for the schema (consumes single-use streams!)
		auto stream_capsule = arrow_obj_handle.attr("__arrow_c_stream__")();
		auto capsule = py::reinterpret_borrow<py::capsule>(stream_capsule);
		auto stream = capsule.get_pointer<struct ArrowArrayStream>();
		if (stream->get_schema(stream, &schema.arrow_schema)) {
			throw InvalidInputException("Failed to get Arrow schema from stream: %s",
			                            stream->get_last_error ? stream->get_last_error(stream) : "unknown error");
		}
		return; // stream_capsule goes out of scope, stream released by capsule destructor
	}
	GetSchemaInternal(arrow_obj_handle, schema);

	// Cache for Table and Dataset (immutable schema)
	if (type == PyArrowObjectType::Table || type == PyArrowObjectType::Dataset) {
		factory->cached_schema = schema.arrow_schema; // factory takes ownership
		schema.arrow_schema.release = nullptr;        // caller gets non-owning copy
		factory->schema_cached = true;
	}
}

} // namespace duckdb

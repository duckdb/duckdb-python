#include "duckdb_python/arrow/arrow_array_stream.hpp"
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

py::object PythonTableArrowArrayStreamFactory::ProduceScanner(DBConfig &config, py::object &arrow_scanner,
                                                              py::handle &arrow_obj_handle,
                                                              ArrowStreamParameters &parameters,
                                                              const ClientProperties &client_properties) {
	D_ASSERT(!py::isinstance<py::capsule>(arrow_obj_handle));
	ArrowSchemaWrapper schema;
	PythonTableArrowArrayStreamFactory::GetSchemaInternal(arrow_obj_handle, schema);
	ArrowTableSchema arrow_table;
	ArrowTableFunction::PopulateArrowTableSchema(config, arrow_table, schema.arrow_schema);

	auto filters = parameters.filters;
	auto &column_list = parameters.projected_columns.columns;
	auto &filter_to_col = parameters.projected_columns.filter_to_col;
	py::list projection_list = py::cast(column_list);

	bool has_filter = filters && !filters->filters.empty();
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
	auto arrow_object_type = DuckDBPyConnection::GetArrowType(arrow_obj_handle);

	if (arrow_object_type == PyArrowObjectType::PyCapsule) {
		auto res = make_uniq<ArrowArrayStreamWrapper>();
		auto capsule = py::reinterpret_borrow<py::capsule>(arrow_obj_handle);
		auto stream = capsule.get_pointer<struct ArrowArrayStream>();
		if (!stream->release) {
			throw InternalException("ArrowArrayStream was released by another thread/library");
		}
		res->arrow_array_stream = *stream;
		stream->release = nullptr;
		return res;
	}

	auto &import_cache = *DuckDBPyConnection::ImportCache();
	py::object scanner;
	py::object arrow_batch_scanner = import_cache.pyarrow.dataset.Scanner().attr("from_batches");
	switch (arrow_object_type) {
	case PyArrowObjectType::Table: {
		auto arrow_dataset = import_cache.pyarrow.dataset().attr("dataset");
		auto dataset = arrow_dataset(arrow_obj_handle);
		py::object arrow_scanner = dataset.attr("__class__").attr("scanner");
		scanner = ProduceScanner(factory->config, arrow_scanner, dataset, parameters, factory->client_properties);
		break;
	}
	case PyArrowObjectType::RecordBatchReader: {
		scanner = ProduceScanner(factory->config, arrow_batch_scanner, arrow_obj_handle, parameters,
		                         factory->client_properties);
		break;
	}
	case PyArrowObjectType::Scanner: {
		// If it's a scanner we have to turn it to a record batch reader, and then a scanner again since we can't stack
		// scanners on arrow Otherwise pushed-down projections and filters will disappear like tears in the rain
		auto record_batches = arrow_obj_handle.attr("to_reader")();
		scanner = ProduceScanner(factory->config, arrow_batch_scanner, record_batches, parameters,
		                         factory->client_properties);
		break;
	}
	case PyArrowObjectType::Dataset: {
		py::object arrow_scanner = arrow_obj_handle.attr("__class__").attr("scanner");
		scanner =
		    ProduceScanner(factory->config, arrow_scanner, arrow_obj_handle, parameters, factory->client_properties);
		break;
	}
	default: {
		auto py_object_type = string(py::str(arrow_obj_handle.get_type().attr("__name__")));
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
	if (py::isinstance<py::capsule>(arrow_obj_handle)) {
		auto capsule = py::reinterpret_borrow<py::capsule>(arrow_obj_handle);
		auto stream = capsule.get_pointer<struct ArrowArrayStream>();
		if (!stream->release) {
			throw InternalException("ArrowArrayStream was released by another thread/library");
		}
		stream->get_schema(stream, &schema.arrow_schema);
		return;
	}

	auto table_class = py::module::import("pyarrow").attr("Table");
	if (py::isinstance(arrow_obj_handle, table_class)) {
		auto obj_schema = arrow_obj_handle.attr("schema");
		auto export_to_c = obj_schema.attr("_export_to_c");
		export_to_c(reinterpret_cast<uint64_t>(&schema.arrow_schema));
		return;
	}

	VerifyArrowDatasetLoaded();

	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto scanner_class = import_cache.pyarrow.dataset.Scanner();

	if (py::isinstance(arrow_obj_handle, scanner_class)) {
		auto obj_schema = arrow_obj_handle.attr("projected_schema");
		auto export_to_c = obj_schema.attr("_export_to_c");
		export_to_c(reinterpret_cast<uint64_t>(&schema));
	} else {
		auto obj_schema = arrow_obj_handle.attr("schema");
		auto export_to_c = obj_schema.attr("_export_to_c");
		export_to_c(reinterpret_cast<uint64_t>(&schema));
	}
}

void PythonTableArrowArrayStreamFactory::GetSchema(uintptr_t factory_ptr, ArrowSchemaWrapper &schema) {
	py::gil_scoped_acquire acquire;
	auto factory = static_cast<PythonTableArrowArrayStreamFactory *>(reinterpret_cast<void *>(factory_ptr)); // NOLINT
	D_ASSERT(factory->arrow_object);
	py::handle arrow_obj_handle(factory->arrow_object);
	GetSchemaInternal(arrow_obj_handle, schema);
}

} // namespace duckdb

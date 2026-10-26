#include "duckdb_python/arrow/arrow_export_utils.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb_python/pyutil.hpp"

namespace duckdb {

namespace pyarrow {

nb::object ToPyArrowSchema(const ArrowSchema &schema) {
	nb::gil_scoped_acquire acquire;

	auto pyarrow_lib_module = nb::module_::import_("pyarrow").attr("lib");
	auto schema_import_func = pyarrow_lib_module.attr("Schema").attr("_import_from_c");
	return schema_import_func(reinterpret_cast<uint64_t>(&schema));
}

nb::object ToArrowTable(const nb::list &batches, nb::object pyarrow_schema) {
	nb::gil_scoped_acquire acquire;

	auto pyarrow_lib_module = nb::module_::import_("pyarrow").attr("lib");
	auto from_batches_func = pyarrow_lib_module.attr("Table").attr("from_batches");

	return nb::cast<duckdb::pyarrow::Table>(from_batches_func(batches, pyarrow_schema));
}

nb::object ToArrowTable(const vector<LogicalType> &types, const vector<string> &names, const nb::list &batches,
                        ClientProperties &options) {
	ArrowSchema schema;
	ArrowConverter::ToArrowSchema(&schema, types, names, options);
	return ToArrowTable(batches, ToPyArrowSchema(schema));
}

} // namespace pyarrow

void TransformDuckToArrowChunk(nb::object pyarrow_schema, ArrowArray &data, nb::list &batches) {
	DUCKDB_PY_ASSERT_GIL_HELD();
	auto pyarrow_lib_module = nb::module_::import_("pyarrow").attr("lib");
	auto batch_import_func = pyarrow_lib_module.attr("RecordBatch").attr("_import_from_c");
	batches.append(batch_import_func(reinterpret_cast<uint64_t>(&data), pyarrow_schema));
}

} // namespace duckdb

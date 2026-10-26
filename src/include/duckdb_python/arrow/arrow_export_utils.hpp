#pragma once

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/main/client_properties.hpp"
#include "duckdb_python/nb/casters.hpp"

namespace duckdb {

namespace pyarrow {

class RecordBatchReader : public nb::object {
public:
	RecordBatchReader(const nb::object &o) : nb::object(o, nb::detail::borrow_t {}) {
	}
	using nb::object::object;

public:
	static bool check_(const nb::handle &object) {
		return !nb::none().is(object);
	}
};

class Table : public nb::object {
public:
	Table(const nb::object &o) : nb::object(o, nb::detail::borrow_t {}) {
	}
	using nb::object::object;

public:
	static bool check_(const nb::handle &object) {
		return !nb::none().is(object);
	}
};

nb::object ToPyArrowSchema(const ArrowSchema &schema);

nb::object ToArrowTable(const vector<LogicalType> &types, const vector<string> &names, const nb::list &batches,
                        ClientProperties &options);

nb::object ToArrowTable(const nb::list &batches, nb::object pyarrow_schema);

} // namespace pyarrow

void TransformDuckToArrowChunk(nb::object pyarrow_schema, ArrowArray &data, nb::list &batches);

} // namespace duckdb

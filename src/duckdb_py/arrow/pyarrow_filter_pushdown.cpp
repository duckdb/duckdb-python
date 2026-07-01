#include "duckdb_python/arrow/pyarrow_filter_pushdown.hpp"

#include "duckdb_python/arrow/filter_pushdown_visitor.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb/function/table/arrow.hpp"

namespace duckdb {

namespace {

string ConvertTimestampUnit(ArrowDateTimeType unit) {
	switch (unit) {
	case ArrowDateTimeType::MICROSECONDS:
		return "us";
	case ArrowDateTimeType::MILLISECONDS:
		return "ms";
	case ArrowDateTimeType::NANOSECONDS:
		return "ns";
	case ArrowDateTimeType::SECONDS:
		return "s";
	default:
		throw NotImplementedException("DatetimeType not recognized in ConvertTimestampUnit: %d",
		                              static_cast<int>(unit));
	}
}

int64_t ConvertTimestampTZValue(int64_t base_value, ArrowDateTimeType datetime_type) {
	auto input = timestamp_t(base_value);
	if (!Value::IsFinite(input)) {
		return base_value;
	}
	switch (datetime_type) {
	case ArrowDateTimeType::MICROSECONDS:
		return Timestamp::GetEpochMicroSeconds(input);
	case ArrowDateTimeType::MILLISECONDS:
		return Timestamp::GetEpochMs(input);
	case ArrowDateTimeType::NANOSECONDS:
		return Timestamp::GetEpochNanoSeconds(input);
	case ArrowDateTimeType::SECONDS:
		return Timestamp::GetEpochSeconds(input);
	default:
		throw NotImplementedException("DatetimeType not recognized in ConvertTimestampTZValue");
	}
}

// Build a pyarrow.dataset scalar matching the given DuckDB Value and (optionally) ArrowType.
// The ArrowType is needed for timestamp unit / decimal precision / blob-view disambiguation; the
// DuckDB Value alone is not sufficient.
py::object MakePyArrowScalar(const Value &constant, const string &timezone_config, const ArrowType *arrow_type) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto scalar = import_cache.pyarrow.scalar();
	py::handle dataset_scalar = import_cache.pyarrow.dataset().attr("scalar");

	switch (constant.type().id()) {
	case LogicalTypeId::BOOLEAN:
		return dataset_scalar(constant.GetValue<bool>());
	case LogicalTypeId::TINYINT:
		return dataset_scalar(constant.GetValue<int8_t>());
	case LogicalTypeId::SMALLINT:
		return dataset_scalar(constant.GetValue<int16_t>());
	case LogicalTypeId::INTEGER:
		return dataset_scalar(constant.GetValue<int32_t>());
	case LogicalTypeId::BIGINT:
		return dataset_scalar(constant.GetValue<int64_t>());
	case LogicalTypeId::DATE: {
		py::handle date_type = import_cache.pyarrow.date32();
		return dataset_scalar(scalar(constant.GetValue<int32_t>(), date_type()));
	}
	case LogicalTypeId::TIME: {
		py::handle date_type = import_cache.pyarrow.time64();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("us")));
	}
	case LogicalTypeId::TIME_NS: {
		// Polars TIME columns round-trip through arrow as time64("ns").
		// `Value::GetValue<int64_t>()` has a hand-rolled fast-path switch for TIME but not
		// TIME_NS — it falls through to GetValueInternal, which then tries
		// Cast::Operation<dtime_ns_t, int64_t> for which no specialization exists, and
		// throws "Unimplemented type for cast (INT64 -> INT64)". Use the type-strong
		// GetValueUnsafe<dtime_ns_t>() which reads `value_.time_ns` from the union
		// directly. The `dtime_ns_t.micros` field name is a misnomer — it actually holds
		// nanoseconds (see arrow_conversion.cpp:432).
		py::handle date_type = import_cache.pyarrow.time64();
		return dataset_scalar(scalar(constant.GetValueUnsafe<dtime_ns_t>().micros, date_type("ns")));
	}
	case LogicalTypeId::TIMESTAMP: {
		py::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("us")));
	}
	case LogicalTypeId::TIMESTAMP_MS: {
		py::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("ms")));
	}
	case LogicalTypeId::TIMESTAMP_NS: {
		py::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("ns")));
	}
	case LogicalTypeId::TIMESTAMP_SEC: {
		py::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("s")));
	}
	case LogicalTypeId::TIMESTAMP_TZ: {
		if (!arrow_type) {
			throw NotImplementedException("Cannot push down TIMESTAMP_TZ filter without an arrow type");
		}
		auto &datetime_info = arrow_type->GetTypeInfo<ArrowDateTimeInfo>();
		auto base_value = constant.GetValue<int64_t>();
		auto arrow_datetime_type = datetime_info.GetDateTimeType();
		auto time_unit_string = ConvertTimestampUnit(arrow_datetime_type);
		auto converted_value = ConvertTimestampTZValue(base_value, arrow_datetime_type);
		py::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(converted_value, date_type(time_unit_string, py::arg("tz") = timezone_config)));
	}
	case LogicalTypeId::TIMESTAMP_TZ_NS: {
		py::handle date_type = import_cache.pyarrow.timestamp();
		auto converted_value = Timestamp::GetEpochNanoSeconds(timestamp_t(constant.GetValue<int64_t>()));
		return dataset_scalar(scalar(converted_value, date_type("ns", py::arg("tz") = timezone_config)));
	}
	case LogicalTypeId::UTINYINT: {
		py::handle integer_type = import_cache.pyarrow.uint8();
		return dataset_scalar(scalar(constant.GetValue<uint8_t>(), integer_type()));
	}
	case LogicalTypeId::USMALLINT: {
		py::handle integer_type = import_cache.pyarrow.uint16();
		return dataset_scalar(scalar(constant.GetValue<uint16_t>(), integer_type()));
	}
	case LogicalTypeId::UINTEGER: {
		py::handle integer_type = import_cache.pyarrow.uint32();
		return dataset_scalar(scalar(constant.GetValue<uint32_t>(), integer_type()));
	}
	case LogicalTypeId::UBIGINT: {
		py::handle integer_type = import_cache.pyarrow.uint64();
		return dataset_scalar(scalar(constant.GetValue<uint64_t>(), integer_type()));
	}
	case LogicalTypeId::FLOAT:
		return dataset_scalar(constant.GetValue<float>());
	case LogicalTypeId::DOUBLE:
		return dataset_scalar(constant.GetValue<double>());
	case LogicalTypeId::VARCHAR:
		return dataset_scalar(constant.ToString());
	case LogicalTypeId::BLOB: {
		if (arrow_type && arrow_type->GetTypeInfo<ArrowStringInfo>().GetSizeType() == ArrowVariableSizeType::VIEW) {
			py::handle binary_view_type = import_cache.pyarrow.binary_view();
			return dataset_scalar(scalar(py::bytes(constant.GetValueUnsafe<string>()), binary_view_type()));
		}
		return dataset_scalar(py::bytes(constant.GetValueUnsafe<string>()));
	}
	case LogicalTypeId::DECIMAL: {
		if (!arrow_type) {
			throw NotImplementedException("Cannot push down DECIMAL filter without an arrow type");
		}
		py::handle decimal_type;
		auto &decimal_info = arrow_type->GetTypeInfo<ArrowDecimalInfo>();
		auto bit_width = decimal_info.GetBitWidth();
		switch (bit_width) {
		case DecimalBitWidth::DECIMAL_32:
			decimal_type = import_cache.pyarrow.decimal32();
			break;
		case DecimalBitWidth::DECIMAL_64:
			decimal_type = import_cache.pyarrow.decimal64();
			break;
		case DecimalBitWidth::DECIMAL_128:
			decimal_type = import_cache.pyarrow.decimal128();
			break;
		default:
			throw NotImplementedException("Unsupported precision for Arrow Decimal Type.");
		}

		uint8_t width;
		uint8_t scale;
		constant.type().GetDecimalProperties(width, scale);
		auto val = import_cache.decimal.Decimal()(constant.ToString());
		return dataset_scalar(
		    scalar(std::move(val), decimal_type(py::arg("precision") = width, py::arg("scale") = scale)));
	}
	default:
		throw NotImplementedException("Unimplemented type \"%s\" for Arrow Filter Pushdown",
		                              constant.type().ToString());
	}
}

struct PyArrowBackend : public FilterBackend {
	explicit PyArrowBackend(const ClientProperties &client_properties_p) : client_properties(client_properties_p) {
		auto &import_cache = *DuckDBPyConnection::ImportCache();
		field_factory = import_cache.pyarrow.dataset().attr("field");
		dataset_scalar = import_cache.pyarrow.dataset().attr("scalar");
	}

	py::object MakeColumnRef(const vector<Identifier> &path) override {
		vector<string> str_path;
		std::transform(path.begin(), path.end(), std::back_inserter(str_path),
		               [](const Identifier &segment) { return segment.GetIdentifierName(); });
		return field_factory(py::tuple(py::cast(str_path)));
	}

	py::object MakeScalar(const Value &v, const ArrowType *arrow_type, const string &timezone_config) override {
		return MakePyArrowScalar(v, timezone_config, arrow_type);
	}

	py::object Compare(ExpressionType op, py::object col, py::object scalar) override {
		switch (op) {
		case ExpressionType::COMPARE_EQUAL:
			return col.attr("__eq__")(scalar);
		case ExpressionType::COMPARE_NOTEQUAL:
			return col.attr("__ne__")(scalar);
		case ExpressionType::COMPARE_LESSTHAN:
			return col.attr("__lt__")(scalar);
		case ExpressionType::COMPARE_GREATERTHAN:
			return col.attr("__gt__")(scalar);
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			return col.attr("__le__")(scalar);
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return col.attr("__ge__")(scalar);
		default:
			throw NotImplementedException("Comparison Type %s can't be an Arrow Scan Pushdown Filter",
			                              ExpressionTypeToString(op));
		}
	}

	py::object NaNCompare(ExpressionType op, py::object col) override {
		switch (op) {
		case ExpressionType::COMPARE_EQUAL:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return col.attr("is_nan")();
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_NOTEQUAL:
			return col.attr("is_nan")().attr("__invert__")();
		case ExpressionType::COMPARE_GREATERTHAN:
			// Nothing is greater than NaN.
			return dataset_scalar(false);
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			// Everything is less than or equal to NaN.
			return dataset_scalar(true);
		default:
			throw NotImplementedException("Unsupported comparison type (%s) for NaN values",
			                              ExpressionTypeToString(op));
		}
	}

	py::object IsNull(py::object col) override {
		return col.attr("is_null")();
	}

	py::object IsNotNull(py::object col) override {
		return col.attr("is_valid")();
	}

	py::object IsIn(py::object col, const vector<Value> &values, const LogicalType &col_logical_type,
	                const string &timezone_config) override {
		// PyArrow accepts a plain Python list of Python-typed scalars; type
		// coercion happens inside the scanner. We don't need the column type.
		(void)col_logical_type;
		(void)timezone_config;
		py::list py_values;
		for (auto &val : values) {
			py_values.append(PythonObject::FromValue(val, val.type(), client_properties));
		}
		return col.attr("isin")(std::move(py_values));
	}

	py::object And(py::object a, py::object b) override {
		return a.attr("__and__")(b);
	}

	py::object Or(py::object a, py::object b) override {
		return a.attr("__or__")(b);
	}

private:
	const ClientProperties &client_properties;
	py::object field_factory;
	py::object dataset_scalar;
};

} // anonymous namespace

py::object PyArrowFilterPushdown::TransformFilter(TableFilterSet &filter_collection,
                                                  unordered_map<idx_t, string> &columns,
                                                  unordered_map<idx_t, idx_t> filter_to_col,
                                                  const ClientProperties &config, const ArrowTableSchema &arrow_table) {
	PyArrowBackend backend(config);
	py::object expression = py::none();
	for (auto &entry : filter_collection) {
		auto column_idx = entry.GetIndex();
		auto &column_name = columns[column_idx];
		D_ASSERT(columns.find(column_idx) != columns.end());

		vector<Identifier> column_path = {Identifier(column_name)};
		auto &arrow_type = arrow_table.GetColumns().at(filter_to_col.at(column_idx));
		py::object child_expression = duckdb::TransformFilter(entry.Filter(), std::move(column_path), backend,
		                                                      arrow_type.get(), config.time_zone);
		if (child_expression.is(py::none())) {
			continue;
		}
		if (expression.is(py::none())) {
			expression = std::move(child_expression);
		} else {
			expression = expression.attr("__and__")(child_expression);
		}
	}
	return expression;
}

} // namespace duckdb

#include "duckdb_python/arrow/arrow_importer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/function/table/arrow/arrow_type_info.hpp"

#include <cstring>

namespace duckdb {

namespace {

//! The rows the batch's own validity bitmap calls NULL, from `from` on; null when it has no bitmap or no NULL row.
unique_ptr<ValidityMask> BatchValidity(const ArrowArray &batch, idx_t from, idx_t rows) {
	if (batch.null_count == 0 || batch.n_buffers == 0 || !batch.buffers || !batch.buffers[0]) {
		return nullptr;
	}
	auto bits = static_cast<const uint8_t *>(batch.buffers[0]);
	auto mask = make_uniq<ValidityMask>(rows);
	for (idx_t row = 0; row < rows; row++) {
		auto index = NumericCast<idx_t>(batch.offset) + from + row;
		if (!(bits[index / 8] & (1 << (index % 8)))) {
			mask->SetInvalid(row);
		}
	}
	if (!mask->CanHaveNull()) {
		return nullptr;
	}
	return mask;
}

bool ValidRange(int64_t offset, int64_t length) {
	return offset >= 0 && length >= 0 && offset <= NumericLimits<int64_t>::Maximum() - length;
}

// Arrow buffers need not be aligned to their value type, so a scalar is copied out instead of read through a pointer.
template <class T>
T LoadAt(const void *buffer, int64_t index) {
	if (index < 0 || index > NumericLimits<int64_t>::Maximum() / static_cast<int64_t>(sizeof(T))) {
		throw InvalidInputException("an Arrow batch addresses row %d of a buffer, beyond what a buffer can hold",
		                            index);
	}
	T value;
	memcpy(&value, static_cast<const char *>(buffer) + index * static_cast<int64_t>(sizeof(T)), sizeof(T));
	return value;
}

void CheckNested(const ArrowArray &array, const ArrowType &type, idx_t column) {
	if (!ValidRange(array.offset, array.length)) {
		throw InvalidInputException("an Arrow batch carried column %d with a nested array of offset %d and length %d, "
		                            "which do not describe a range of rows",
		                            column, array.offset, array.length);
	}
	// The engine reads validity only through the bitmap, so NULL rows would read as values.
	if (array.null_count > 0 && array.n_buffers > 0 && (!array.buffers || !array.buffers[0])) {
		throw InvalidInputException(
		    "an Arrow batch carried column %d with a non-zero null count but no validity bitmap", column);
	}
	// The engine reads every child its format names, so a missing one would be read through a null pointer.
	auto child_of = [&](idx_t index) -> const ArrowArray & {
		if (!array.children || array.n_children <= NumericCast<int64_t>(index) || !array.children[index]) {
			throw InvalidInputException("an Arrow batch carried column %d with a nested array missing its child %d",
			                            column, index);
		}
		if (!array.children[index]->release) {
			throw InvalidInputException(
			    "an Arrow batch carried column %d with a nested array whose child %d is released already", column,
			    index);
		}
		return *array.children[index];
	};
	// Like the batch: the children of a struct or a sparse union cover its offset plus its length.
	auto check_covers = [&](const ArrowArray &child, idx_t factor) {
		const auto rows = array.offset + array.length;
		const auto width = NumericCast<int64_t>(factor);
		if (width > 0 && rows > NumericLimits<int64_t>::Maximum() / width) {
			throw InvalidInputException("an Arrow batch carried column %d with a nested array too long to address",
			                            column);
		}
		const auto needed = rows * width;
		if (child.length < needed) {
			throw InvalidInputException("Arrow array child length does not match the array length: column %d holds a "
			                            "nested array of %d rows where %d were expected",
			                            column, child.length, needed);
		}
	};
	switch (type.GetPhysicalType()) {
	case ArrowArrayPhysicalType::DICTIONARY_ENCODED:
		if (!array.dictionary) {
			throw InvalidInputException("an Arrow batch carried column %d with a dictionary-encoded array that has no "
			                            "dictionary",
			                            column);
		}
		if (!array.dictionary->release) {
			throw InvalidInputException(
			    "an Arrow batch carried column %d with a dictionary-encoded array whose dictionary is released already",
			    column);
		}
		CheckNested(*array.dictionary, type.GetDictionary(), column);
		return;
	case ArrowArrayPhysicalType::RUN_END_ENCODED: {
		auto &info = type.GetTypeInfo<ArrowStructInfo>();
		auto &run_ends = child_of(0);
		auto &values = child_of(1);
		if (run_ends.length != values.length) {
			throw InvalidInputException("an Arrow batch carried column %d with a run end encoded array of %d run ends "
			                            "and %d values",
			                            column, run_ends.length, values.length);
		}
		CheckNested(run_ends, info.GetChild(0), column);
		CheckNested(values, info.GetChild(1), column);
		// The engine decodes runs until they are exhausted and leaves the rows after the last run end unwritten.
		int64_t last_end = 0;
		if (run_ends.length > 0) {
			if (run_ends.n_buffers < 2 || !run_ends.buffers || !run_ends.buffers[1]) {
				throw InvalidInputException(
				    "an Arrow batch carried column %d with a run end encoded array whose run ends have no values",
				    column);
			}
			const auto index = run_ends.offset + run_ends.length - 1;
			switch (info.GetChild(0).GetDuckType().id()) {
			case LogicalTypeId::SMALLINT:
				last_end = LoadAt<int16_t>(run_ends.buffers[1], index);
				break;
			case LogicalTypeId::INTEGER:
				last_end = LoadAt<int32_t>(run_ends.buffers[1], index);
				break;
			case LogicalTypeId::BIGINT:
				last_end = LoadAt<int64_t>(run_ends.buffers[1], index);
				break;
			default:
				throw InvalidInputException(
				    "an Arrow batch carried column %d with a run end encoded array whose run ends "
				    "are %s, not SMALLINT, INTEGER or BIGINT",
				    column, info.GetChild(0).GetDuckType().ToString());
			}
		}
		if (array.length > 0 && last_end < array.offset + array.length) {
			throw InvalidInputException("an Arrow batch carried column %d with a run end encoded array whose runs end "
			                            "at %d before its %d rows",
			                            column, last_end, array.offset + array.length);
		}
		return;
	}
	default:
		break;
	}
	switch (type.GetDuckType().id()) {
	// A VARIANT is converted as the struct it is stored as.
	case LogicalTypeId::VARIANT:
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::TUPLE:
	case LogicalTypeId::UNION: {
		auto &info = type.GetTypeInfo<ArrowStructInfo>();
		// The engine converts every child the array has into vectors sized by the type.
		if (array.n_children != NumericCast<int64_t>(info.ChildCount())) {
			throw InvalidInputException("an Arrow batch carried column %d with a struct or union of %d child arrays "
			                            "where its type has %d",
			                            column, array.n_children, info.ChildCount());
		}
		for (idx_t i = 0; i < info.ChildCount(); i++) {
			auto &child = child_of(i);
			check_covers(child, 1);
			CheckNested(child, info.GetChild(i), column);
		}
		return;
	}
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP: {
		auto &info = type.GetTypeInfo<ArrowListInfo>();
		auto &child = child_of(0);
		auto value_at = [&](idx_t buffer, int64_t index) -> int64_t {
			if (info.GetSizeType() == ArrowVariableSizeType::SUPER_SIZE) {
				return LoadAt<int64_t>(array.buffers[buffer], index);
			}
			return LoadAt<int32_t>(array.buffers[buffer], index);
		};
		// The child's own range is checked first, so the offset checks below compare against a valid length.
		CheckNested(child, info.GetChild(), column);
		if (!info.IsView() && array.length > 0) {
			if (array.n_buffers < 2 || !array.buffers || !array.buffers[1]) {
				throw InvalidInputException("an Arrow batch carried column %d with a list that has no offsets", column);
			}
			// The engine takes the child rows to read from the first and the last offset.
			const auto first = value_at(1, array.offset);
			const auto last = value_at(1, array.offset + array.length);
			if (first < 0 || first > last || last > child.length) {
				throw InvalidInputException(
				    "an Arrow batch carried column %d with a list whose offsets %d to %d do not "
				    "fit its child of %d rows",
				    column, first, last, child.length);
			}
		} else if (info.IsView() && array.length > 0) {
			if (array.n_buffers < 3 || !array.buffers || !array.buffers[1] || !array.buffers[2]) {
				throw InvalidInputException(
				    "an Arrow batch carried column %d with a list view that has no offsets or sizes", column);
			}
			// The engine reads the child from the lowest offset to the highest end of every list with a size.
			for (int64_t row = array.offset; row < array.offset + array.length; row++) {
				const auto size = value_at(2, row);
				const auto start = value_at(1, row);
				if (size != 0 && (size < 0 || start < 0 || start > child.length - size)) {
					throw InvalidInputException(
					    "an Arrow batch carried column %d with a list view whose list at row %d "
					    "reaches past its child of %d rows",
					    column, row - array.offset, child.length);
				}
			}
		}
		return;
	}
	case LogicalTypeId::ARRAY: {
		auto &child = child_of(0);
		auto &info = type.GetTypeInfo<ArrowArrayInfo>();
		CheckNested(child, info.GetChild(), column);
		check_covers(child, info.FixedSize());
		return;
	}
	default:
		return;
	}
}

void ApplyBatchValidity(Vector &vector, const ValidityMask &batch_validity, idx_t rows) {
	auto &mask = FlatVector::ValidityMutable(vector);
	for (idx_t row = 0; row < rows; row++) {
		if (!batch_validity.RowIsValid(row)) {
			mask.SetInvalid(row);
		}
	}
}

} // namespace

void ReadArrowColumn(Vector &vector, ArrowArray &array, const ArrowType &arrow_type, idx_t position, idx_t rows,
                     ArrowArrayScanState &state, ValidityMask *row_validity) {
	switch (arrow_type.GetPhysicalType()) {
	case ArrowArrayPhysicalType::DICTIONARY_ENCODED:
		if (!array.dictionary) {
			throw InvalidInputException("a dictionary-encoded Arrow column has no dictionary");
		}
		ArrowToDuckDBConversion::ColumnArrowToDuckDBDictionary(vector, array, position, state, rows, arrow_type, -1,
		                                                       row_validity);
		break;
	case ArrowArrayPhysicalType::RUN_END_ENCODED:
		ArrowToDuckDBConversion::ColumnArrowToDuckDBRunEndEncoded(vector, array, position, state, rows, arrow_type, -1,
		                                                          row_validity);
		break;
	case ArrowArrayPhysicalType::DEFAULT:
		ArrowToDuckDBConversion::SetValidityMask(vector, array, position, rows, 0, -1);
		// Merged before the conversion so a struct or union column passes the NULL rows on to its children.
		if (row_validity) {
			ApplyBatchValidity(vector, *row_validity, rows);
		}
		ArrowToDuckDBConversion::ColumnArrowToDuckDB(vector, array, position, state, rows, arrow_type, -1,
		                                             row_validity);
		break;
	default:
		throw NotImplementedException("Only default, dictionary and run-end encoded Arrow layouts are supported");
	}
	FlatVector::SetSize(vector, rows);
	// A conversion may rebuild the validity mask from the column's own buffers.
	if (row_validity && vector.GetVectorType() == VectorType::FLAT_VECTOR) {
		ApplyBatchValidity(vector, *row_validity, rows);
	}
}

ArrowBatchImporter::ArrowBatchImporter(ClientContext &context, shared_ptr<ArrowTableSchema> schema_p,
                                       vector<idx_t> columns_p)
    : context(context), schema(std::move(schema_p)), arrow_columns(schema->GetColumns()),
      columns(std::move(columns_p)) {
}

shared_ptr<ArrowArrayWrapper> ArrowBatchImporter::Adopt(ArrowArray &incoming) const {
	const auto expected = arrow_columns.size();
	if (incoming.n_children < 0 || NumericCast<idx_t>(incoming.n_children) != expected) {
		throw InvalidInputException("an Arrow batch carried %d columns where %d were expected", incoming.n_children,
		                            expected);
	}
	if (!ValidRange(incoming.offset, incoming.length)) {
		throw InvalidInputException("an Arrow batch has offset %d and length %d, which do not describe a range of rows",
		                            incoming.offset, incoming.length);
	}
	if (expected > 0 && !incoming.children) {
		throw InvalidInputException("an Arrow batch has no column arrays");
	}
	// Columns of a sliced batch keep their full length; the batch offset applies to every one of them.
	const auto rows_needed = incoming.offset + incoming.length;
	for (idx_t i = 0; i < expected; i++) {
		auto child = incoming.children[i];
		if (!child || !child->release) {
			throw InvalidInputException("an Arrow batch carried column %d missing or released already", i);
		}
		if (!ValidRange(child->offset, child->length)) {
			throw InvalidInputException("an Arrow batch carried column %d with offset %d and length %d, which do not "
			                            "describe a range of rows",
			                            i, child->offset, child->length);
		}
		if (child->length < rows_needed) {
			throw InvalidInputException("Arrow array child length does not match the array length: column %d holds %d "
			                            "rows where %d were expected",
			                            i, child->length, rows_needed);
		}
	}
	if (incoming.null_count > 0 && (incoming.n_buffers == 0 || !incoming.buffers || !incoming.buffers[0])) {
		throw InvalidInputException("an Arrow batch has a non-zero null count but no validity bitmap");
	}
	for (auto column : columns) {
		CheckNested(*incoming.children[column], *arrow_columns.at(column), column);
	}
	auto owned = make_shared_ptr<ArrowArrayWrapper>();
	owned->arrow_array = incoming;
	incoming.release = nullptr;
	return owned;
}

void ArrowBatchImporter::Read(shared_ptr<ArrowArrayWrapper> batch_p, idx_t start, idx_t rows, bool last_p) {
	D_ASSERT(batch_p && start + rows <= NumericCast<idx_t>(batch_p->arrow_array.length));
	if (batch_p != batch) {
		states.clear();
		batch = std::move(batch_p);
	}
	offset = start;
	end = start + rows;
	last = last_p;
}

void ArrowBatchImporter::Append(ArrowArray &incoming) {
	auto adopted = Adopt(incoming);
	const auto rows = NumericCast<idx_t>(adopted->arrow_array.length);
	Read(std::move(adopted), 0, rows, true);
}

ArrowArrayScanState &ArrowBatchImporter::StateOf(idx_t column) {
	auto entry = states.find(column);
	if (entry != states.end()) {
		return *entry->second;
	}
	auto state = make_uniq<ArrowArrayScanState>(context);
	// Zero-copy vectors keep the batch alive through this.
	state->owned_data = batch;
	auto &result = *state;
	states.emplace(column, std::move(state));
	return result;
}

bool ArrowBatchImporter::Next(DataChunk &output) {
	if (!batch || offset >= end) {
		if (last) {
			states.clear();
			batch.reset();
		}
		return false;
	}
	D_ASSERT(output.ColumnCount() == columns.size());
	auto &array = batch->arrow_array;
	const auto rows = MinValue<idx_t>(STANDARD_VECTOR_SIZE, end - offset);
	// A batch offset shifts the rows the way the start of a row range does, so both go into the row position.
	const auto position = NumericCast<idx_t>(array.offset) + offset;
	output.SetChildCardinality(rows);
	auto batch_validity = BatchValidity(array, offset, rows);
	for (idx_t i = 0; i < columns.size(); i++) {
		const auto column = columns[i];
		ReadArrowColumn(output.data[i], *array.children[column], *arrow_columns.at(column), position, rows,
		                StateOf(column), batch_validity.get());
	}
	offset += rows;
	return true;
}

} // namespace duckdb

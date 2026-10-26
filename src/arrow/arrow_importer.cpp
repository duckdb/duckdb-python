#include "duckdb_python/arrow/arrow_importer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/function/table/arrow/arrow_type_info.hpp"

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

//! Checks the arrays below a requested column as the batch is checked, and refuses an offset the engine would drop: its
//! struct conversion hands its children only the struct's own offset and its list conversion none, so an offset met
//! further up would be lost there and the rows below read from the wrong places.
//! `received` is the offset the engine passes the array as its parent's, and `in_list` whether a list above it makes
//! the engine ignore parent offsets.
void CheckNested(const ArrowArray &array, const ArrowType &type, idx_t column, int64_t received, bool in_list) {
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
		// The dictionary is converted on its own, from its start.
		CheckNested(*array.dictionary, type.GetDictionary(), column, 0, false);
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
		// The run ends and the values are converted on their own, from their start.
		CheckNested(run_ends, info.GetChild(0), column, 0, false);
		CheckNested(values, info.GetChild(1), column, 0, false);
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
		const bool is_union = type.GetDuckType().id() == LogicalTypeId::UNION;
		// A union adds its own offset to the one it received, a struct passes on only its own.
		if ((in_list && array.offset != 0) || (!is_union && !in_list && received != 0)) {
			throw InvalidInputException(
			    "an Arrow batch carried column %d with a struct or union below an offset the conversion would drop, "
			    "which would read the wrong rows",
			    column);
		}
		const auto passed = is_union ? received + array.offset : array.offset;
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
			CheckNested(child, info.GetChild(i), column, passed, in_list);
		}
		return;
	}
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP: {
		auto &info = type.GetTypeInfo<ArrowListInfo>();
		auto &child = child_of(0);
		auto value_at = [&](idx_t buffer, int64_t index) -> int64_t {
			if (info.GetSizeType() == ArrowVariableSizeType::SUPER_SIZE) {
				return static_cast<const int64_t *>(array.buffers[buffer])[index];
			}
			return static_cast<const int32_t *>(array.buffers[buffer])[index];
		};
		// When the lists address no child values and start at offset 0, the engine reads the child as plain values,
		// which a dictionary-encoded child is not; a list view counts as starting at 0 whenever its lists are empty.
		const bool dictionary_child = info.GetChild().GetPhysicalType() == ArrowArrayPhysicalType::DICTIONARY_ENCODED;
		bool only_empty_from_zero = false;
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
			only_empty_from_zero = last == 0;
		} else if (info.IsView() && array.length > 0 && dictionary_child) {
			if (array.n_buffers < 3 || !array.buffers || !array.buffers[2]) {
				throw InvalidInputException("an Arrow batch carried column %d with a list view that has no sizes",
				                            column);
			}
			only_empty_from_zero = true;
			for (int64_t row = array.offset; row < array.offset + array.length && only_empty_from_zero; row++) {
				only_empty_from_zero = value_at(2, row) == 0;
			}
		}
		if (dictionary_child && only_empty_from_zero) {
			throw InvalidInputException("an Arrow batch carried column %d with only empty lists over a "
			                            "dictionary-encoded child, which the conversion cannot read",
			                            column);
		}
		CheckNested(child, info.GetChild(), column, 0, true);
		return;
	}
	case LogicalTypeId::ARRAY: {
		auto &child = child_of(0);
		auto &info = type.GetTypeInfo<ArrowArrayInfo>();
		// The engine's fixed size list conversion reads a run end encoded child as plain values.
		if (array.length > 0 && info.GetChild().GetPhysicalType() == ArrowArrayPhysicalType::RUN_END_ENCODED) {
			throw InvalidInputException("an Arrow batch carried column %d with a run end encoded array inside a fixed "
			                            "size list, which the conversion cannot read",
			                            column);
		}
		check_covers(child, info.FixedSize());
		CheckNested(child, info.GetChild(), column, 0, true);
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
		CheckNested(*incoming.children[column], *arrow_columns.at(column), column, 0, false);
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
	// The batch offset is folded into the row position, which the engine hands on to nested children, rather than
	// passed as the parent offset, which its struct conversion does not hand on.
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

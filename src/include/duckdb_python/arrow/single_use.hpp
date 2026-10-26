#pragma once

#include "duckdb_python/nb/casters.hpp"

namespace duckdb {

//! Whether the object is a pyarrow RecordBatchReader, which hands out its batches once however many streams it exports;
//! call with the GIL held
bool IsSingleUseArrowReader(nb::handle object);

//! Records a single-use reader as consumed, by identity, and takes the record back unless Keep() is called; construct
//! and destroy with the GIL held
class ConsumedReaderMark {
public:
	ConsumedReaderMark(nb::handle reader, bool single_use);
	~ConsumedReaderMark();
	ConsumedReaderMark(const ConsumedReaderMark &) = delete;
	ConsumedReaderMark &operator=(const ConsumedReaderMark &) = delete;

	void Keep() {
		kept = true;
	}

private:
	//! Set only once the reader is marked
	nb::object record;
	nb::object key;
	bool kept = false;
};

} // namespace duckdb

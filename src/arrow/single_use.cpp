#include "duckdb_python/arrow/single_use.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb_python/exceptions.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"

namespace duckdb {

static constexpr const char *CONSUMED_READER_ERROR =
    "This RecordBatchReader has already been consumed and cannot be scanned again. A RecordBatchReader can be read "
    "only once; to scan its data more than once, read it into a table first, for example with read_all().";

namespace {

//! The walk only compares names, so a missing or unreadable one is an empty text rather than an error
string TypeText(nb::object value) {
	if (!value.is_valid() || !nb::str::check_(value)) {
		PyErr_Clear();
		return string();
	}
	try {
		return ToUTF8(value);
	} catch (nb::python_error &) {
		return string();
	}
}

//! The fallback for a reader whose pyarrow left sys.modules: an extension type's module and name come from tp_name,
//! while a Python class can forge both through __module__ and __qualname__, so it is never the reader.
bool IsPyarrowReaderType(PyTypeObject *type) {
	if (PyType_GetFlags(type) & Py_TPFLAGS_HEAPTYPE) {
		return false;
	}
	auto name = TypeText(nb::steal(PyType_GetName(type)));
	auto module = TypeText(nb::steal(PyObject_GetAttrString(reinterpret_cast<PyObject *>(type), "__module__")));
	return name == "RecordBatchReader" && module == "pyarrow.lib";
}

//! Walks the bases the type was created with, not __mro__, which a metaclass can replace; each ancestor once, since
//! a diamond hierarchy repeats them
bool DerivesFromPyarrowReader(PyTypeObject *type) {
	vector<PyTypeObject *> pending {type};
	unordered_set<PyTypeObject *> seen;
	while (!pending.empty()) {
		auto current = pending.back();
		pending.pop_back();
		if (!seen.insert(current).second) {
			continue;
		}
		if (IsPyarrowReaderType(current)) {
			return true;
		}
		auto bases = static_cast<PyObject *>(PyType_GetSlot(current, Py_tp_bases));
		if (!bases || !PyTuple_Check(bases)) {
			PyErr_Clear();
			continue;
		}
		auto count = PyTuple_Size(bases);
		for (Py_ssize_t i = 0; i < count; i++) {
			pending.push_back(reinterpret_cast<PyTypeObject *>(PyTuple_GetItem(bases, i)));
		}
	}
	return false;
}

} // namespace

bool IsSingleUseArrowReader(nb::handle object) {
	if (ModuleIsLoaded<PyarrowCacheItem>()) {
		auto reader_type = DuckDBPyConnection::ImportCache()->pyarrow.RecordBatchReader();
		if (reader_type.ptr() && PyType_Check(reader_type.ptr())) {
			return PyType_IsSubtype(Py_TYPE(object.ptr()), reinterpret_cast<PyTypeObject *>(reader_type.ptr())) != 0;
		}
	}
	return DerivesFromPyarrowReader(Py_TYPE(object.ptr()));
}

ConsumedReaderMark::ConsumedReaderMark(nb::handle reader, bool single_use) {
	if (!single_use) {
		return;
	}
	auto readers = DuckDBPyConnection::ConsumedArrowReaders();
	key = nb::steal(PyLong_FromVoidPtr(reader.ptr()));
	if (!key) {
		throw nb::python_error();
	}
	auto forget = nb::module_::import_("functools").attr("partial")(readers.attr("pop"), key);
	auto entry = nb::steal(PyWeakref_NewRef(reader.ptr(), forget.ptr()));
	if (!entry) {
		throw nb::python_error();
	}
	// One setdefault tests and records at once; the entry leaves with the reader, before its id can be reused.
	auto recorded = readers.attr("setdefault")(key, entry);
	if (!recorded.is(entry)) {
		throw InvalidInputException(CONSUMED_READER_ERROR);
	}
	record = std::move(readers);
}

ConsumedReaderMark::~ConsumedReaderMark() {
	if (record && !kept && PyDict_DelItem(record.ptr(), key.ptr()) < 0) {
		PyErr_Clear();
	}
}

} // namespace duckdb

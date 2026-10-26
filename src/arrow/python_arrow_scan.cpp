#include "duckdb_python/arrow/python_arrow_scan.hpp"

#include "duckdb_python/arrow/arrow_importer.hpp"
#include "duckdb_python/arrow/predicate_translation.hpp"
#include "duckdb_python/arrow/single_use.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/external_dependencies.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/storage_info.hpp"
#include "duckdb_python/engine/connection.hpp"
#include "duckdb_python/exceptions.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_dependency.hpp"

#include <chrono>
#include <cstring>
#include <thread>
#include <type_traits>
#ifndef _WIN32
#include <pthread.h>
#endif

namespace duckdb {

namespace {

constexpr const char *STREAM_CAPSULE = "arrow_array_stream";
constexpr const char *SCHEMA_CAPSULE = "arrow_schema";
constexpr const char *ARRAY_CAPSULE = "arrow_array";
//! The engine clamps the cap to its scheduler's thread count, so a large one asks for every thread it has
constexpr idx_t STREAM_MAX_THREADS = 1024;

string StreamError(ArrowArrayStream &stream, int code) {
	const char *text = stream.get_last_error ? stream.get_last_error(&stream) : nullptr;
	if (!text || !*text) {
		return StringUtil::Format("no error message (error code %d)", code);
	}
	return text;
}

//! Set by an atexit hook: from then on no scan calls into a source, since a thread that waits for the GIL while the
//! interpreter finalizes never gets it, and the thread finalizing may itself be waiting for that scan to end. A scan
//! an exit handler starts after the hook is still read, as the interpreter only finalizes once every handler ran.
atomic<bool> interpreter_exiting {false};
//! Calls into a source that were let in and have not returned; the atexit hook waits for them with the GIL released.
atomic<idx_t> calls_in_flight {0};
//! The calls among them made on this thread, the only ones still in flight in a child this thread forks
thread_local idx_t calls_on_this_thread = 0;

class SourceCall {
public:
	explicit SourceCall(bool started_during_exit = false) {
		calls_in_flight++;
		admitted = !interpreter_exiting.load() || started_during_exit;
		if (!admitted) {
			calls_in_flight -= 1;
		} else {
			calls_on_this_thread++;
		}
	}
	~SourceCall() {
		if (admitted) {
			calls_on_this_thread -= 1;
			calls_in_flight -= 1;
		}
	}
	SourceCall(const SourceCall &) = delete;
	SourceCall &operator=(const SourceCall &) = delete;

	explicit operator bool() const {
		return admitted;
	}

private:
	bool admitted;
};

void OnInterpreterExit() {
	interpreter_exiting = true;
	nb::gil_scoped_release release;
	// Not bounded: a call let in may need the GIL again before it returns, which it never gets once the interpreter
	// finalizes, so a slow source delays the exit instead of hanging it.
	while (calls_in_flight.load() > 0) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

#ifndef _WIN32
//! A forked child keeps the exit flag, as it continues from the parent's place among the exit handlers.
void AfterForkInChild() {
	calls_in_flight = calls_on_this_thread;
}
#endif

//! Whether this call comes from an exit handler running after the exit hook, which the hook lets through; call with
//! the GIL held
bool StartedDuringExit() {
	return interpreter_exiting.load() && !nb::cast<bool>(nb::module_::import_("sys").attr("is_finalizing")());
}

//! Releases an Arrow struct taken from a source: under the GIL when the source asked for it, directly otherwise, and
//! not at all once the interpreter is exiting, since even pyarrow's own release can enter Python.
template <class T>
void ReleaseFromSource(T &value, bool under_gil, bool admitted_during_exit) noexcept {
	if (!value.release) {
		return;
	}
	if (SourceCall call {admitted_during_exit}) {
		if (!under_gil) {
			value.release(&value);
		} else if (nb::detail::cleanup_guard guard {}) {
			value.release(&value);
		}
	}
	value.release = nullptr;
}

template <class T>
struct ArrowOwned {
	ArrowOwned() {
		value.release = nullptr;
	}
	ArrowOwned(ArrowOwned &&other) noexcept
	    : value(other.value), release_under_gil(other.release_under_gil),
	      admitted_during_exit(other.admitted_during_exit) {
		other.value.release = nullptr;
	}
	ArrowOwned &operator=(ArrowOwned &&other) noexcept {
		if (this != &other) {
			Release();
			value = other.value;
			release_under_gil = other.release_under_gil;
			admitted_during_exit = other.admitted_during_exit;
			other.value.release = nullptr;
		}
		return *this;
	}
	ArrowOwned(const ArrowOwned &) = delete;
	ArrowOwned &operator=(const ArrowOwned &) = delete;
	~ArrowOwned() {
		Release();
	}

	void Release() noexcept {
		ReleaseFromSource(value, release_under_gil, admitted_during_exit);
	}
	explicit operator bool() const {
		return value.release != nullptr;
	}

	T value {};
	bool release_under_gil = true;
	bool admitted_during_exit = false;
};

struct RoutedRelease {
	void (*release)(ArrowArray *);
	void *private_data;
	bool under_gil;
	bool admitted_during_exit;
};

void ReleaseRouted(ArrowArray *array) {
	auto routed = static_cast<RoutedRelease *>(array->private_data);
	array->release = routed->release;
	array->private_data = routed->private_data;
	const bool under_gil = routed->under_gil;
	const bool admitted_during_exit = routed->admitted_during_exit;
	delete routed;
	ReleaseFromSource(*array, under_gil, admitted_during_exit);
}

//! The engine releases an adopted array wherever its last vector goes, on any thread and after the scan is gone, so
//! its release is routed through ReleaseFromSource here.
void RouteRelease(ArrowArray &array, bool under_gil, bool admitted_during_exit) {
	array.private_data = new RoutedRelease {array.release, array.private_data, under_gil, admitted_during_exit};
	array.release = &ReleaseRouted;
}

template <class T>
ArrowOwned<T> TakeFromCapsule(nb::handle capsule, const char *kind, const string &subject) {
	auto &held = InCapsule<T>(capsule, kind, subject);
	ArrowOwned<T> taken;
	taken.value = held;
	held.release = nullptr;
	return taken;
}

bool IsBatch(const ArrowSchema &schema) {
	return schema.format && std::strcmp(schema.format, "+s") == 0;
}

string NameOf(const ArrowSchema &schema, idx_t index) {
	auto child = schema.children[index];
	return child->name ? child->name : "";
}

//! A struct array with no validity buffer still declares one buffer slot, holding null.
const void *NO_BUFFERS[1] = {nullptr};

template <class T>
struct Wrapped {
	T *child;
	T **children;
};

template <class T>
void ReleaseWrapped(T *wrapper) {
	auto owned = static_cast<Wrapped<T> *>(wrapper->private_data);
	if (owned->child->release) {
		owned->child->release(owned->child);
	}
	delete owned->child;
	delete[] owned->children;
	delete owned;
	wrapper->release = nullptr;
}

void WrapAsBatch(ArrowSchema &schema) {
	auto child = new ArrowSchema(schema);
	auto children = new ArrowSchema *[1] {child};
	auto wrapped = new Wrapped<ArrowSchema> {child, children};
	schema = ArrowSchema {};
	schema.format = "+s";
	schema.name = "";
	schema.n_children = 1;
	schema.children = children;
	schema.release = &ReleaseWrapped<ArrowSchema>;
	schema.private_data = wrapped;
}

void WrapAsBatch(ArrowArray &array) {
	auto child = new ArrowArray(array);
	auto children = new ArrowArray *[1] {child};
	auto wrapped = new Wrapped<ArrowArray> {child, children};
	array = ArrowArray {};
	array.length = child->length;
	array.null_count = 0;
	array.offset = 0;
	array.n_buffers = 1;
	array.buffers = NO_BUFFERS;
	array.n_children = 1;
	array.children = children;
	array.release = &ReleaseWrapped<ArrowArray>;
	array.private_data = wrapped;
}

//! A predicate the source promised to apply, whole, since one may read several columns, and its text for the plan
struct AcceptedPredicate {
	shared_ptr<PyObjectHolder> node;
	string text;
};

//! What one bind of the scan knows; shared with copies of the plan, so nothing of one execution lives here
struct PythonArrowScanBindData : public TableFunctionData {
	explicit PythonArrowScanBindData(shared_ptr<PythonArrowSourceInfo> info_p) : info(std::move(info_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PythonArrowScanBindData>(*this);
	}

	shared_ptr<PythonArrowSourceInfo> info;
	//! The Arrow names of the declared columns, empty for a nameless array
	vector<string> raw_names;
	vector<LogicalType> types;
	ArrowTableSchema arrow_table;
	PredicateColumns predicate_columns;
	optional_idx rows;
	//! Accumulated across the optimizer's calls
	vector<AcceptedPredicate> predicates;
};

struct Exported {
	nb::object stream;
	nb::object schema;
	nb::object array;
	bool projected = false;

	bool IsArray() const {
		return array.is_valid();
	}
};

//! Call with the GIL held
Exported ExportStream(const PythonArrowSourceInfo &info, const vector<idx_t> *columns,
                      const vector<AcceptedPredicate> &predicates) {
	nb::object request = nb::none();
	if (columns) {
		nb::list wanted;
		for (auto column : *columns) {
			wanted.append(nb::int_(static_cast<uint64_t>(column)));
		}
		request = std::move(wanted);
	}
	nb::list promised;
	for (auto &predicate : predicates) {
		promised.append(predicate.node->obj);
	}
	auto answer = info.source.obj.attr("stream")(request, nb::tuple(promised));
	if (!PyTuple_Check(answer.ptr()) || PyTuple_Size(answer.ptr()) != 2 ||
	    !PyBool_Check(PyTuple_GetItem(answer.ptr(), 1))) {
		throw InvalidInputException("%s answered stream() with something other than a (data, projected) pair",
		                            Subject(info.name, info.kind, "source"));
	}
	auto data = nb::borrow(PyTuple_GetItem(answer.ptr(), 0));
	Exported exported;
	exported.projected = PyTuple_GetItem(answer.ptr(), 1) == Py_True;
	if (PyTuple_Check(data.ptr()) && PyTuple_Size(data.ptr()) == 2) {
		exported.schema = nb::borrow(PyTuple_GetItem(data.ptr(), 0));
		exported.array = nb::borrow(PyTuple_GetItem(data.ptr(), 1));
	} else {
		exported.stream = std::move(data);
	}
	return exported;
}

} // namespace

struct CachedArrowSchema {
	ArrowOwned<ArrowSchema> schema;
	bool wrapped = false;
};

namespace {

//! Resolves a batch schema whose columns may carry a NULL name, which the Arrow C data interface allows and the engine
//! cannot read: such a column resolves as one named "", through copies that leave the source's structs untouched.
void ResolveBatchSchema(ClientContext &context, ArrowTableSchema &table, const ArrowSchema &batch) {
	const auto count = NumericCast<idx_t>(batch.n_children);
	vector<ArrowSchema> columns(count);
	vector<ArrowSchema *> pointers(count);
	for (idx_t i = 0; i < count; i++) {
		columns[i] = *batch.children[i];
		if (!columns[i].name) {
			columns[i].name = "";
		}
		pointers[i] = &columns[i];
	}
	auto named = batch;
	named.children = pointers.data();
	ArrowTableFunction::PopulateArrowTableSchema(context, table, named);
}

//! The source's schema, read without consuming a stream: from a schema hook, asked once for the life of a source whose
//! schema is stable and at every bind otherwise, else from a stream that is only borrowed; call with the GIL held
shared_ptr<CachedArrowSchema> SchemaOf(PythonArrowSourceInfo &info) {
	if (info.schema) {
		return info.schema;
	}
	auto &source = info.source.obj;
	const auto subject = Subject(info.name, info.kind, "object");
	auto result = make_shared_ptr<CachedArrowSchema>();
	bool from_hook = true;
	auto underlying = nb::getattr(source, "obj", nb::none());
	if (PyUtil::HasAttribute(source, "__arrow_c_schema__")) {
		result->schema = TakeFromCapsule<ArrowSchema>(source.attr("__arrow_c_schema__")(), SCHEMA_CAPSULE, subject);
	} else if (!underlying.is_none() && PyUtil::HasAttribute(underlying, "schema") &&
	           PyUtil::HasAttribute(underlying.attr("schema"), "_export_to_c")) {
		underlying.attr("schema").attr("_export_to_c")(reinterpret_cast<uintptr_t>(&result->schema.value));
	} else {
		from_hook = false;
		auto exported = ExportStream(info, nullptr, {});
		if (exported.IsArray()) {
			// The array capsule is dropped unread, which releases it.
			result->schema = TakeFromCapsule<ArrowSchema>(exported.schema, SCHEMA_CAPSULE, subject);
		} else {
			// Only borrowed, so a stream the source keeps, such as a bare capsule, is still there for the scan.
			auto &stream = InCapsule<ArrowArrayStream>(exported.stream, STREAM_CAPSULE, subject);
			auto code = stream.get_schema(&stream, &result->schema.value);
			if (code != 0) {
				result->schema.value.release = nullptr;
				throw InvalidInputException("reading the schema of %s failed: %s",
				                            Subject(info.name, info.kind, "stream"), StreamError(stream, code));
			}
		}
	}
	if (!result->schema) {
		throw InvalidInputException("reading the schema of %s failed: it gave a released schema", subject);
	}
	result->schema.admitted_during_exit = StartedDuringExit();
	result->wrapped = !IsBatch(result->schema.value);
	if (result->wrapped) {
		WrapAsBatch(result->schema.value);
	}
	if (from_hook && nb::getattr(source, "schema_is_stable", nb::handle(Py_False)).ptr() == Py_True) {
		info.schema = result;
	}
	return result;
}

unique_ptr<FunctionData> PythonArrowScanBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<Identifier> &names) {
	if (!input.ref.bind_info) {
		throw BinderException("python_arrow_scan requires a Python Arrow source bind input");
	}
	DynamicCastCheck<PythonArrowSourceInfo>(input.ref.bind_info.get());
	auto info = shared_ptr_cast<TableFunctionInfo, PythonArrowSourceInfo>(input.ref.bind_info);
	auto result = make_uniq<PythonArrowScanBindData>(info);
	shared_ptr<CachedArrowSchema> schema;
	{
		nb::gil_scoped_acquire gil;
		engine::CallbackScope callback(context);
		try {
			schema = SchemaOf(*info);
		} catch (nb::python_error &error) {
			throw InvalidInputException("reading the schema of %s failed: %s",
			                            Subject(info->name, info->kind, "object"), FormatUDFPythonError(error));
		}
		try {
			auto rows = info->source.obj.attr("rows")();
			if (PyLong_Check(rows.ptr()) && !PyBool_Check(rows.ptr())) {
				auto count = PyLong_AsUnsignedLongLong(rows.ptr());
				if (PyErr_Occurred()) {
					PyErr_Clear();
				} else {
					result->rows = NumericCast<idx_t>(count);
				}
			}
		} catch (nb::python_error &error) {
			throw InvalidInputException("counting the rows of %s failed: %s", Subject(info->name, info->kind, "object"),
			                            FormatUDFPythonError(error));
		}
	}
	auto &arrow_schema = schema->schema.value;
	ResolveBatchSchema(context, result->arrow_table, arrow_schema);
	result->types = result->arrow_table.GetTypes();
	if (result->types.empty()) {
		throw InvalidInputException("Provided table/dataframe must have at least one column");
	}
	auto &resolved_names = result->arrow_table.GetNames();
	for (idx_t i = 0; i < result->types.size(); i++) {
		auto raw = NameOf(arrow_schema, i);
		// A plain array has no column name of its own; the engine would call it v0.
		names.emplace_back(schema->wrapped && raw.empty() ? string("value") : resolved_names[i]);
		result->raw_names.push_back(std::move(raw));
	}
	result->predicate_columns = PredicateColumns(arrow_schema, result->types);
	return_types = result->types;
	return std::move(result);
}

unique_ptr<NodeStatistics> PythonArrowScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<PythonArrowScanBindData>();
	if (!bind_data.rows.IsValid()) {
		return make_uniq<NodeStatistics>();
	}
	// An estimate only: an object can grow in place after bind, and a maximum would let a reused plan drop a LIMIT.
	return make_uniq<NodeStatistics>(bind_data.rows.GetIndex());
}

//! Offers each predicate the walker can carry exactly to the source; one it accepts leaves the plan, which is why only
//! a True answer accepts. The optimizer may call this more than once, each time with the predicates still in the plan.
void PythonArrowScanPushdown(ClientContext &context, LogicalGet &get, FunctionData *bind_data_p,
                             vector<unique_ptr<Expression>> &filters) {
	auto &bind_data = bind_data_p->Cast<PythonArrowScanBindData>();
	if (filters.empty()) {
		return;
	}
	auto &column_ids = get.GetColumnIds();
	nb::gil_scoped_acquire gil;
	engine::CallbackScope callback(context);
	try {
		nb::object fragment;
		for (idx_t i = 0; i < filters.size(); i++) {
			auto resolve = [&](idx_t binding_index) {
				if (binding_index >= column_ids.size()) {
					throw PredicateRefused();
				}
				auto &column = column_ids[binding_index];
				if (!column.HasPrimaryIndex() || column.IsVirtualColumn()) {
					throw PredicateRefused();
				}
				return bind_data.predicate_columns.Resolve(column.GetPrimaryIndex());
			};
			nb::object node;
			try {
				node = TranslatePredicate(*filters[i], resolve);
			} catch (const PredicateRefused &) {
				continue;
			}
			auto answer = bind_data.info->source.obj.attr("accepts")(node);
			if (answer.ptr() != Py_True) {
				continue;
			}
			if (!fragment.is_valid()) {
				fragment = DuckDBPyConnection::ArrowSourcesNodes().attr("fragment");
			}
			AcceptedPredicate accepted;
			accepted.text = ToUTF8(fragment(node));
			accepted.node = make_shared_ptr<PyObjectHolder>(std::move(node));
			bind_data.predicates.push_back(std::move(accepted));
			filters.erase_at(i);
			i -= 1;
		}
	} catch (nb::python_error &error) {
		throw InvalidInputException("offering a filter to %s failed: %s",
		                            Subject(bind_data.info->name, bind_data.info->kind, "source"),
		                            FormatUDFPythonError(error));
	}
}

InsertionOrderPreservingMap<string> PythonArrowScanToString(TableFunctionToStringInput &input) {
	InsertionOrderPreservingMap<string> result;
	result["Function"] = "PYTHON_ARROW_SCAN";
	if (!input.bind_data) {
		return result;
	}
	auto &bind_data = input.bind_data->Cast<PythonArrowScanBindData>();
	string filters;
	for (auto &predicate : bind_data.predicates) {
		if (!filters.empty()) {
			filters += "\n";
		}
		filters += predicate.text;
	}
	if (!filters.empty()) {
		result["Filters"] = filters;
	}
	return result;
}

struct PythonArrowScanGlobalState : public GlobalTableFunctionState {
	idx_t MaxThreads() const override {
		return max_threads;
	}

	string name;
	ArrowSourceKind kind = ArrowSourceKind::REGISTERED;
	ArrowOwned<ArrowArrayStream> stream;
	//! The one batch, until a thread claims it
	ArrowOwned<ArrowArray> single;
	ArrowOwned<ArrowSchema> schema;
	shared_ptr<ArrowTableSchema> arrow_table;
	//! The stream column each output column reads
	vector<idx_t> picks;
	//! Whether the stream's arrays are plain values, each to be read as a batch of one column
	bool wrap = false;
	//! Whether the source asked for the GIL around pulling the next array and releasing one
	bool python_callbacks = false;
	//! Started by an exit handler that runs after the exit hook, while other threads can still take the GIL
	bool started_during_exit = false;
	idx_t max_threads = 1;
	//! A pulled batch is read in ranges of this many rows from its start, the last one also taking a remainder shorter
	//! than a range
	idx_t range_rows = DEFAULT_ROW_GROUP_SIZE;

	//! Guards the pull, the claim of a range with its batch index, and the claim of the single batch
	mutex pull_lock;
	//! The batch whose ranges are being handed out, checked once by the thread that pulled it
	shared_ptr<ArrowArrayWrapper> current;
	//! The first row of current not yet handed out
	idx_t next_row = 0;
	idx_t next_batch = 0;
	bool exhausted = false;
};

//! Exports the stream, narrowed to the requested columns when the source can, and checks it against the declared
//! columns by width, name and type: the scan reads columns by position, so a mismatch is refused rather than read.
unique_ptr<GlobalTableFunctionState> PythonArrowScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<PythonArrowScanBindData>();
	auto &info = *bind_data.info;
	const auto declared = bind_data.types.size();
	vector<idx_t> requested;
	bool identity = input.column_ids.size() == declared;
	for (idx_t i = 0; i < input.column_ids.size(); i++) {
		auto column = input.column_ids[i];
		if (column >= declared) {
			throw InternalException("python_arrow_scan was asked for column %d, which it does not declare", column);
		}
		bind_data.arrow_table.GetColumns().at(column)->ThrowIfInvalid();
		requested.push_back(column);
		identity = identity && column == i;
	}
	auto result = make_uniq<PythonArrowScanGlobalState>();
	result->name = info.name;
	result->kind = info.kind;
	const auto subject = Subject(info.name, info.kind, "object");
	const auto source_subject = Subject(info.name, info.kind, "source");

	nb::gil_scoped_acquire gil;
	engine::CallbackScope callback(context);
	result->started_during_exit = StartedDuringExit();
	// Kept only once every check passed, so a scan refused before reading a row leaves a single use reader readable.
	unique_ptr<ConsumedReaderMark> mark;
	Exported exported;
	try {
		auto &source = info.source.obj;
		auto underlying = nb::getattr(source, "obj", source);
		// A producer that runs Python takes the GIL itself, as pyarrow's, ctypes' and Cython's callbacks do; holding it
		// across a pull instead deadlocks a producer whose own threads need Python, such as a reader over a Python
		// file.
		result->python_callbacks = nb::getattr(source, "pull_under_gil", nb::handle(Py_False)).ptr() == Py_True;
		mark = make_uniq<ConsumedReaderMark>(underlying, IsSingleUseArrowReader(underlying));
		exported = ExportStream(info, identity ? nullptr : &requested, bind_data.predicates);
	} catch (nb::python_error &error) {
		throw InvalidInputException("exporting a stream from %s failed: %s", subject, FormatUDFPythonError(error));
	}
	if (exported.IsArray()) {
		result->schema = TakeFromCapsule<ArrowSchema>(exported.schema, SCHEMA_CAPSULE, subject);
		auto single = TakeFromCapsule<ArrowArray>(exported.array, ARRAY_CAPSULE, subject);
		RouteRelease(single.value, result->python_callbacks, result->started_during_exit);
		single.release_under_gil = false;
		single.admitted_during_exit = result->started_during_exit;
		result->single = std::move(single);
	} else {
		// Only borrowed until every check passed, so a scan that fails here leaves a stream the source keeps in place.
		auto &borrowed = InCapsule<ArrowArrayStream>(exported.stream, STREAM_CAPSULE, subject);
		auto code = borrowed.get_schema(&borrowed, &result->schema.value);
		if (code != 0) {
			result->schema.value.release = nullptr;
			throw InvalidInputException("reading the schema of %s failed: %s", Subject(info.name, info.kind, "stream"),
			                            StreamError(borrowed, code));
		}
	}
	result->schema.release_under_gil = result->python_callbacks;
	result->schema.admitted_during_exit = result->started_during_exit;
	auto &schema = result->schema.value;
	result->wrap = !IsBatch(schema);
	if (result->wrap) {
		WrapAsBatch(schema);
		if (result->single) {
			WrapAsBatch(result->single.value);
		}
	}
	const auto fields = NumericCast<idx_t>(schema.n_children);
	if (exported.projected) {
		if (fields != requested.size()) {
			throw InvalidInputException("%s answered with %d columns where %d were requested", source_subject, fields,
			                            requested.size());
		}
		for (idx_t i = 0; i < fields; i++) {
			result->picks.push_back(i);
		}
	} else {
		if (fields != declared) {
			throw InvalidInputException("%s answered with %d columns where it declared %d", source_subject, fields,
			                            declared);
		}
		result->picks = requested;
	}
	result->arrow_table = make_shared_ptr<ArrowTableSchema>();
	ResolveBatchSchema(context, *result->arrow_table, schema);
	auto &types = result->arrow_table->GetTypes();
	for (idx_t i = 0; i < fields; i++) {
		const auto expected = exported.projected ? requested[i] : i;
		const auto name = NameOf(schema, i);
		if (name != bind_data.raw_names[expected]) {
			throw InvalidInputException("%s answered with column '%s' at position %d where '%s' was expected",
			                            source_subject, name, i, bind_data.raw_names[expected]);
		}
		if (types[i] != bind_data.types[expected]) {
			throw InvalidInputException("%s answered with column '%s' of type %s where %s was declared", source_subject,
			                            name, types[i].ToString(), bind_data.types[expected].ToString());
		}
	}
	// One vector ranges under verify_parallelism, as in the engine's own scans, so queries under it cross range ends.
	result->range_rows =
	    ClientConfig::GetConfig(context).verify_parallelism ? STANDARD_VECTOR_SIZE : DEFAULT_ROW_GROUP_SIZE;
	result->max_threads = STREAM_MAX_THREADS;
	if (result->single) {
		// Its number of ranges, as the last one takes a remainder shorter than a range
		const auto length = result->single.value.length;
		result->max_threads = MaxValue<idx_t>(1, length > 0 ? NumericCast<idx_t>(length) / result->range_rows : 0);
	}
	if (!exported.IsArray()) {
		auto stream = TakeFromCapsule<ArrowArrayStream>(exported.stream, STREAM_CAPSULE, subject);
		stream.release_under_gil = result->python_callbacks;
		stream.admitted_during_exit = result->started_during_exit;
		result->stream = std::move(stream);
	}
	mark->Keep();
	return std::move(result);
}

struct PythonArrowScanLocalState : public LocalTableFunctionState {
	PythonArrowScanLocalState(ClientContext &context, shared_ptr<ArrowTableSchema> schema, vector<idx_t> columns)
	    : importer(context, std::move(schema), std::move(columns)) {
	}

	ArrowBatchImporter importer;
	idx_t batch_index = 0;
};

unique_ptr<LocalTableFunctionState> PythonArrowScanInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                             GlobalTableFunctionState *global_state) {
	auto &global = global_state->Cast<PythonArrowScanGlobalState>();
	return make_uniq<PythonArrowScanLocalState>(context.client, global.arrow_table, global.picks);
}

void PullNext(ClientContext &context, PythonArrowScanGlobalState &global, ArrowArray &out) {
	auto &stream = global.stream.value;
	// A pyarrow reader over a generator runs that Python code on this thread, so the scope is set either way.
	engine::CallbackScope callback(context);
	SourceCall call(global.started_during_exit);
	if (!call) {
		throw InvalidInputException("reading %s failed: the Python interpreter is shutting down",
		                            Subject(global.name, global.kind, "stream"));
	}
	int code;
	string error;
	if (global.python_callbacks) {
		nb::gil_scoped_acquire gil;
		code = stream.get_next(&stream, &out);
		if (code != 0) {
			error = StreamError(stream, code);
		}
	} else {
		code = stream.get_next(&stream, &out);
		if (code != 0) {
			error = StreamError(stream, code);
		}
	}
	if (code != 0) {
		out.release = nullptr;
		throw InvalidInputException("reading %s failed: %s", Subject(global.name, global.kind, "stream"), error);
	}
	if (out.release) {
		RouteRelease(out, global.python_callbacks, global.started_during_exit);
	}
}

void PythonArrowScanExec(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &global = data.global_state->Cast<PythonArrowScanGlobalState>();
	auto &local = data.local_state->Cast<PythonArrowScanLocalState>();
	while (!local.importer.Next(output)) {
		// Declared before the lock, so a batch refused or used up here is released after it, as a release may take the
		// GIL.
		ArrowArrayWrapper pulled;
		shared_ptr<ArrowArrayWrapper> batch;
		idx_t start = 0;
		idx_t rows = 0;
		bool last = true;
		{
			lock_guard<mutex> guard(global.pull_lock);
			if (!global.current) {
				if (global.exhausted) {
					return;
				}
				if (!global.stream) {
					pulled.arrow_array = global.single.value;
					global.single.value.release = nullptr;
					global.exhausted = true;
				} else {
					try {
						PullNext(context, global, pulled.arrow_array);
					} catch (...) {
						// A stream that failed is not pulled again by another thread while the query is cancelled.
						global.exhausted = true;
						throw;
					}
					if (!pulled.arrow_array.release) {
						global.exhausted = true;
					} else if (global.wrap) {
						WrapAsBatch(pulled.arrow_array);
					}
				}
				if (!pulled.arrow_array.release) {
					return;
				}
				try {
					global.current = local.importer.Adopt(pulled.arrow_array);
				} catch (...) {
					global.exhausted = true;
					throw;
				}
				global.next_row = 0;
			}
			const auto length = NumericCast<idx_t>(global.current->arrow_array.length);
			start = global.next_row;
			const auto remaining = length - start;
			rows = remaining < 2 * global.range_rows ? remaining : global.range_rows;
			global.next_row += rows;
			last = global.next_row == length;
			batch = last ? std::move(global.current) : global.current;
			local.batch_index = global.next_batch++;
		}
		local.importer.Read(std::move(batch), start, rows, last);
	}
}

//! A thread claims increasing indexes from one counter, so its batch index never goes down, as the engine requires.
OperatorPartitionData PythonArrowScanPartitionData(ClientContext &context, TableFunctionGetPartitionInput &input) {
	if (input.partition_info.RequiresPartitionColumns()) {
		throw InternalException("python_arrow_scan: partition columns are not supported");
	}
	return OperatorPartitionData(input.local_state->Cast<PythonArrowScanLocalState>().batch_index);
}

} // namespace

void WatchInterpreterExit() {
#ifndef _WIN32
	pthread_atfork(nullptr, nullptr, &AfterForkInChild);
#endif
	nb::module_::import_("atexit").attr("register")(nb::cpp_function(&OnInterpreterExit));
}

void RegisterPythonArrowScan(DatabaseInstance &db) {
	TableFunction function("python_arrow_scan", {}, PythonArrowScanExec, PythonArrowScanBind, PythonArrowScanInitGlobal,
	                       PythonArrowScanInitLocal);
	function.cardinality = PythonArrowScanCardinality;
	function.pushdown_complex_filter = PythonArrowScanPushdown;
	function.to_string = PythonArrowScanToString;
	function.get_partition_data = PythonArrowScanPartitionData;
	function.projection_pushdown = true;
	function.filter_pushdown = false;
	function.filter_prune = false;

	TableFunctionSet set(function.name);
	set.AddFunction(std::move(function));
	CreateTableFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	auto &catalog = Catalog::GetSystemCatalog(db);
	auto transaction = CatalogTransaction::GetSystemTransaction(db);
	catalog.CreateFunction(transaction, info);
}

string Subject(const string &name, ArrowSourceKind kind, const char *noun) {
	if (name.empty() || kind == ArrowSourceKind::NAMELESS) {
		return "the Arrow source";
	}
	if (kind == ArrowSourceKind::VARIABLE) {
		if (std::strcmp(noun, "stream") == 0) {
			return StringUtil::Format("the stream of the Python object '%s'", name);
		}
		return StringUtil::Format("the Python object '%s'", name);
	}
	return StringUtil::Format("the %s registered as '%s'", noun, name);
}

unique_ptr<TableFunctionRef> CreatePythonArrowScanRef(nb::object source, const string &name, ArrowSourceKind kind) {
	auto ref = make_uniq<TableFunctionRef>();
	vector<unique_ptr<ParsedExpression>> children;
	ref->function = make_uniq<FunctionExpression>("python_arrow_scan", std::move(children));
	auto dependency = make_shared_ptr<ExternalDependency>();
	dependency->AddDependency("replacement_cache", PythonDependencyItem::Create(source));
	ref->external_dependency = std::move(dependency);
	ref->bind_info = make_shared_ptr<PythonArrowSourceInfo>(std::move(source), name, kind);
	return ref;
}

} // namespace duckdb

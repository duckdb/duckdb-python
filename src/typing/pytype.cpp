#include "duckdb_python/pytype.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

// NOLINTNEXTLINE(readability-identifier-naming)
bool PyGenericAlias::check_(const nb::handle &object) {
	if (!ModuleIsLoaded<TypesCacheItem>()) {
		return false;
	}
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	return duckdb::PyUtil::IsInstance(object, import_cache.types.GenericAlias());
}

// NOLINTNEXTLINE(readability-identifier-naming)
bool PyUnionType::check_(const nb::handle &object) {
	auto types_loaded = ModuleIsLoaded<TypesCacheItem>();
	auto &import_cache = *DuckDBPyConnection::ImportCache();

	// for >= py310: isinstance(object, types.UnionType)
	if (types_loaded && duckdb::PyUtil::IsInstance(object, import_cache.types.UnionType())) {
		return true;
	}
	// for all py3: typing.get_origin(object) is typing.Union
	auto get_origin_func = import_cache.typing.get_origin();
	auto origin = get_origin_func(object);
	if (origin.is(import_cache.typing.Union())) {
		return true;
	}
	return false;
}

DuckDBPyType::DuckDBPyType(LogicalType type) : type(std::move(type)) {
}

//! Heap-allocate an owned DuckDBPyType. Spelled std::unique_ptr (not duckdb::unique_ptr) so nanobind's
//! type_caster<std::unique_ptr<T>> transfers ownership to Python; lets call-sites embed a type in a tuple/attr
//! and lets the nb::new_ factories deduce the right return type.
static std::unique_ptr<DuckDBPyType> MakeType(LogicalType type) {
	return make_uniq<DuckDBPyType>(std::move(type));
}

bool DuckDBPyType::Equals(const DuckDBPyType &other) const {
	return type == other.Type();
}

bool DuckDBPyType::EqualsString(const string &type_str) const {
	return StringUtil::CIEquals(type.ToString(), type_str);
}

std::unique_ptr<DuckDBPyType> DuckDBPyType::GetAttribute(const string &name) const {
	auto name_identifier = Identifier(name);
	if (type.id() == LogicalTypeId::STRUCT || type.id() == LogicalTypeId::TUPLE || type.id() == LogicalTypeId::UNION) {
		auto &children = StructType::GetChildTypes(type);
		for (idx_t i = 0; i < children.size(); i++) {
			auto &child = children[i];
			if (child.first == name) {
				return MakeType(StructType::GetChildType(type, i));
			}
		}
	}
	if (type.id() == LogicalTypeId::LIST && StringUtil::CIEquals(name, "child")) {
		return MakeType(ListType::GetChildType(type));
	}
	if (type.id() == LogicalTypeId::MAP) {
		auto is_key = StringUtil::CIEquals(name, "key");
		auto is_value = StringUtil::CIEquals(name, "value");
		if (is_key) {
			return MakeType(MapType::KeyType(type));
		} else if (is_value) {
			return MakeType(MapType::ValueType(type));
		} else {
			throw nb::attribute_error(StringUtil::Format("Tried to get a child from a map by the name of '%s', but "
			                                             "this type only has 'key' and 'value' children",
			                                             name)
			                              .c_str());
		}
	}
	throw nb::attribute_error(
	    StringUtil::Format("Tried to get child type by the name of '%s', but this type either isn't nested, "
	                       "or it doesn't have a child by that name",
	                       name)
	        .c_str());
}

static LogicalType FromObject(const nb::object &object);

namespace {
enum class PythonTypeObject : uint8_t {
	INVALID,   // not convertible to our type
	BASE,      // 'builtin' type objects
	UNION,     // typing.UnionType
	COMPOSITE, // list|dict types
	STRUCT,    // dictionary
	STRING,    // string value
	TYPE       // duckdb pytype
};
}

static PythonTypeObject GetTypeObjectType(const nb::handle &type_object) {
	if (nb::isinstance<nb::type_object>(type_object)) {
		return PythonTypeObject::BASE;
	}
	if (nb::isinstance<nb::str>(type_object)) {
		return PythonTypeObject::STRING;
	}
	if (nb::isinstance<PyGenericAlias>(type_object)) {
		return PythonTypeObject::COMPOSITE;
	}
	if (nb::isinstance<nb::dict>(type_object)) {
		return PythonTypeObject::STRUCT;
	}
	if (nb::isinstance<PyUnionType>(type_object)) {
		return PythonTypeObject::UNION;
	}
	if (nb::isinstance<DuckDBPyType>(type_object)) {
		return PythonTypeObject::TYPE;
	}
	return PythonTypeObject::INVALID;
}

static LogicalType FromString(const string &type_str, std::shared_ptr<DuckDBPyConnection> pycon) {
	if (!pycon) {
		pycon = DuckDBPyConnection::DefaultConnection();
	}
	auto &connection = pycon->con.GetConnection();

	LogicalType type;
	connection.context->RunFunctionInTransaction(
	    [&]() { type = TransformStringToLogicalType(type_str, *connection.context); });
	return type;
}

static bool FromNumpyType(const nb::object &type, LogicalType &result) {
	// Since this is a type, we have to create an instance from it first.
	auto obj = type();
	// We convert these to string because the underlying physical
	// types of a numpy type aren't consistent on every platform
	if (!nb::hasattr(obj, "dtype")) {
		return false;
	}
	string type_str = nb::cast<std::string>(nb::str(nb::object(obj.attr("dtype"))));
	if (type_str == "bool") {
		result = LogicalType::BOOLEAN;
	} else if (type_str == "int8") {
		result = LogicalType::TINYINT;
	} else if (type_str == "uint8") {
		result = LogicalType::UTINYINT;
	} else if (type_str == "int16") {
		result = LogicalType::SMALLINT;
	} else if (type_str == "uint16") {
		result = LogicalType::USMALLINT;
	} else if (type_str == "int32") {
		result = LogicalType::INTEGER;
	} else if (type_str == "uint32") {
		result = LogicalType::UINTEGER;
	} else if (type_str == "int64") {
		result = LogicalType::BIGINT;
	} else if (type_str == "uint64") {
		result = LogicalType::UBIGINT;
	} else if (type_str == "float16") {
		// FIXME: should we even support this?
		result = LogicalType::FLOAT;
	} else if (type_str == "float32") {
		result = LogicalType::FLOAT;
	} else if (type_str == "float64") {
		result = LogicalType::DOUBLE;
	} else {
		return false;
	}
	return true;
}

static LogicalType FromType(const nb::type_object &obj) {
	nb::module_ builtins = nb::module_::import_("builtins");
	if (obj.is(builtins.attr("str"))) {
		return LogicalType::VARCHAR;
	}
	if (obj.is(builtins.attr("int"))) {
		return LogicalType::BIGINT;
	}
	if (obj.is(builtins.attr("bytearray"))) {
		return LogicalType::BLOB;
	}
	if (obj.is(builtins.attr("bytes"))) {
		return LogicalType::BLOB;
	}
	if (obj.is(builtins.attr("float"))) {
		return LogicalType::DOUBLE;
	}
	if (obj.is(builtins.attr("bool"))) {
		return LogicalType::BOOLEAN;
	}

	LogicalType result;
	if (FromNumpyType(obj, result)) {
		return result;
	}

	throw nb::type_error("Could not convert from unknown 'type' to DuckDBPyType");
}

static bool IsMapType(const nb::tuple &args) {
	if (args.size() != 2) {
		return false;
	}
	for (auto arg : args) {
		if (GetTypeObjectType(arg) == PythonTypeObject::INVALID) {
			return false;
		}
	}
	return true;
}

static nb::tuple FilterNones(const nb::tuple &args) {
	nb::list result;

	for (const auto &arg : args) {
		nb::object object = nb::borrow<nb::object>(arg);
		if (object.is((nb::none()).type())) {
			continue;
		}
		result.append(object);
	}
	return nb::tuple(result);
}

static LogicalType FromUnionTypeInternal(const nb::tuple &args) {
	idx_t index = 1;
	child_list_t<LogicalType> members;

	for (const auto &arg : args) {
		auto name = Identifier(StringUtil::Format("u%d", index++));
		nb::object object = nb::borrow<nb::object>(arg);
		members.push_back(make_pair(name, FromObject(object)));
	}

	return LogicalType::UNION(std::move(members));
}

static LogicalType FromUnionType(const nb::object &obj) {
	nb::tuple args = obj.attr("__args__");

	// Optional inserts NoneType into the Union
	// all types are nullable in DuckDB so we just filter the Nones
	auto filtered_args = FilterNones(args);
	if (filtered_args.size() == 1) {
		// If only a single type is left, dont construct a UNION
		return FromObject(filtered_args[0]);
	}
	return FromUnionTypeInternal(filtered_args);
};

static LogicalType FromGenericAlias(const nb::object &obj) {
	nb::module_ builtins = nb::module_::import_("builtins");
	nb::module_ types = nb::module_::import_("types");
	auto generic_alias = types.attr("GenericAlias");
	D_ASSERT(duckdb::PyUtil::IsInstance(obj, generic_alias));
	// nb::object (not auto, which deduces an accessor): nb::str(accessor) is an ambiguous overload on MSVC.
	nb::object origin = obj.attr("__origin__");
	nb::tuple args = obj.attr("__args__");

	if (origin.is(builtins.attr("list"))) {
		if (args.size() != 1) {
			throw NotImplementedException("Can only create a LIST from a single type");
		}
		return LogicalType::LIST(FromObject(args[0]));
	}
	if (origin.is(builtins.attr("dict"))) {
		if (IsMapType(args)) {
			return LogicalType::MAP(FromObject(args[0]), FromObject(args[1]));
		} else {
			throw NotImplementedException("Can only create a MAP from a dict if args is formed correctly");
		}
	}
	string origin_type = nb::cast<std::string>(nb::str(origin));
	throw InvalidInputException("Could not convert from '%s' to DuckDBPyType", origin_type);
}

static LogicalType FromDictionary(const nb::object &obj) {
	auto dict = nb::borrow<nb::dict>(obj);
	child_list_t<LogicalType> children;
	if (dict.size() == 0) {
		throw InvalidInputException("Could not convert empty dictionary to a duckdb STRUCT type");
	}
	children.reserve(dict.size());
	for (auto item : dict) {
		auto &name_p = item.first;
		auto type_p = nb::borrow<nb::object>(item.second);
		auto name = Identifier(duckdb::PyUtil::CastToString(name_p));
		auto type = FromObject(type_p);
		children.push_back(std::make_pair(name, std::move(type)));
	}
	return LogicalType::STRUCT(std::move(children));
}

static LogicalType FromObject(const nb::object &object) {
	auto object_type = GetTypeObjectType(object);
	switch (object_type) {
	case PythonTypeObject::BASE: {
		return FromType(nb::cast<nb::type_object>(object));
	}
	case PythonTypeObject::COMPOSITE: {
		return FromGenericAlias(object);
	}
	case PythonTypeObject::STRUCT: {
		return FromDictionary(object);
	}
	case PythonTypeObject::UNION: {
		return FromUnionType(object);
	}
	case PythonTypeObject::STRING: {
		auto string_value = nb::cast<std::string>(nb::str(object));
		return FromString(string_value, nullptr);
	}
	case PythonTypeObject::TYPE: {
		// GetTypeObjectType already established that `object` is a DuckDBPyType instance, so borrow a const ref
		// (no ownership extraction) and copy out its LogicalType.
		return nb::cast<const DuckDBPyType &>(object).Type();
	}
	default: {
		string actual_type = nb::cast<std::string>(nb::str((object).type()));
		throw NotImplementedException("Could not convert from object of type '%s' to DuckDBPyType", actual_type);
	}
	}
}

bool DuckDBPyType::TryConvert(const nb::object &object, std::unique_ptr<DuckDBPyType> &result) {
	if (nb::isinstance<DuckDBPyType>(object)) {
		// Copy the existing type into a fresh owned instance (value semantics; mirrors the old shared_ptr share).
		result = MakeType(nb::cast<const DuckDBPyType &>(object).Type());
		return true;
	}
	try {
		// Construct via the registered DuckDBPyType type (DuckDBPyType(object)); this hits the same factories
		// that drive the implicit conversion. The constructed Python object owns its DuckDBPyType, so copy its
		// LogicalType into our own owned instance before it goes out of scope.
		nb::object converted = nb::type<DuckDBPyType>()(object);
		result = MakeType(nb::cast<const DuckDBPyType &>(converted).Type());
		return true;
	} catch (...) {
		// A failed construction (e.g. an unannotated parameter) leaves the Python error indicator set; clear it
		// so the caller's subsequent Python operations don't trip on a stale error.
		PyErr_Clear();
		return false;
	}
}

void DuckDBPyType::Initialize(nb::handle &m) {
	// nanobind types aren't weak-referenceable by default.
	auto type_module = nb::class_<DuckDBPyType>(m, "DuckDBPyType", nb::is_weak_referenceable());

	type_module.def("__repr__", &DuckDBPyType::ToString, "Stringified representation of the type object");
	type_module.def("__duckdb_template__", &DuckDBPyType::ToString);
	type_module.def("__eq__", &DuckDBPyType::Equals, "Compare two types for equality", nb::arg("other"),
	                nb::is_operator());
	type_module.def("__eq__", &DuckDBPyType::EqualsString, "Compare two types for equality", nb::arg("other"),
	                nb::is_operator());
	type_module.def("__hash__", [](const DuckDBPyType &type) {
		auto s = type.ToString();
		return nb::hash(nb::str(s.c_str(), s.size()));
	});
	type_module.def_prop_ro("id", &DuckDBPyType::GetId);
	type_module.def_prop_ro("children", &DuckDBPyType::Children);
	type_module.def(nb::new_([](const string &type_str, std::shared_ptr<DuckDBPyConnection> connection) {
		                auto ltype = FromString(type_str, std::move(connection));
		                return MakeType(ltype);
	                }),
	                nb::arg("type_str"), nb::arg("connection").none() = nb::none());
	type_module.def(nb::new_([](const PyGenericAlias &obj) {
		auto ltype = FromGenericAlias(obj);
		return MakeType(ltype);
	}));
	type_module.def(nb::new_([](const PyUnionType &obj) {
		auto ltype = FromUnionType(obj);
		return MakeType(ltype);
	}));
	type_module.def(nb::new_([](const nb::object &obj) {
		auto ltype = FromObject(obj);
		return MakeType(ltype);
	}));
	type_module.def("__getattr__", &DuckDBPyType::GetAttribute, "Get the child type by 'name'", nb::arg("name"));
	// nanobind: nb::is_operator() implies operator-style argument handling and rejects the explicit nb::arg name
	type_module.def("__getitem__", &DuckDBPyType::GetAttribute, "Get the child type by 'name'", nb::is_operator());

	nb::implicitly_convertible<nb::object, DuckDBPyType>();
	nb::implicitly_convertible<nb::str, DuckDBPyType>();
	nb::implicitly_convertible<PyGenericAlias, DuckDBPyType>();
	nb::implicitly_convertible<PyUnionType, DuckDBPyType>();
}

string DuckDBPyType::ToString() const {
	return type.ToString();
}

nb::list DuckDBPyType::Children() const {

	switch (type.id()) {
	case LogicalTypeId::LIST:
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::TUPLE:
	case LogicalTypeId::UNION:
	case LogicalTypeId::MAP:
	case LogicalTypeId::ARRAY:
	case LogicalTypeId::ENUM:
	case LogicalTypeId::DECIMAL:
		break;
	default:
		throw InvalidInputException("This type is not nested so it doesn't have children");
	}

	nb::list children;
	auto id = type.id();
	if (id == LogicalTypeId::LIST) {
		children.append(nb::make_tuple("child", MakeType(ListType::GetChildType(type))));
		return children;
	}
	if (id == LogicalTypeId::ARRAY) {
		children.append(nb::make_tuple("child", MakeType(ArrayType::GetChildType(type))));
		children.append(nb::make_tuple("size", ArrayType::GetSize(type)));
		return children;
	}
	if (id == LogicalTypeId::ENUM) {
		auto &values_insert_order = EnumType::GetValuesInsertOrder(type);
		auto strings = FlatVector::GetData<string_t>(values_insert_order);
		nb::list strings_list;
		for (size_t i = 0; i < EnumType::GetSize(type); i++) {
			{
				auto sv = strings[i].GetString();
				strings_list.append(nb::str(sv.c_str(), sv.size()));
			}
		}
		children.append(nb::make_tuple("values", strings_list));
		return children;
	}
	if (id == LogicalTypeId::STRUCT || id == LogicalTypeId::TUPLE || id == LogicalTypeId::UNION) {
		auto &struct_children = StructType::GetChildTypes(type);
		for (idx_t i = 0; i < struct_children.size(); i++) {
			auto &child = struct_children[i];
			children.append(nb::make_tuple(child.first, MakeType(StructType::GetChildType(type, i))));
		}
		return children;
	}
	if (id == LogicalTypeId::MAP) {
		children.append(nb::make_tuple("key", MakeType(MapType::KeyType(type))));
		children.append(nb::make_tuple("value", MakeType(MapType::ValueType(type))));
		return children;
	}
	if (id == LogicalTypeId::DECIMAL) {
		children.append(nb::make_tuple("precision", DecimalType::GetWidth(type)));
		children.append(nb::make_tuple("scale", DecimalType::GetScale(type)));
		return children;
	}
	throw InternalException("Children is not implemented for this type");
}

string DuckDBPyType::GetId() const {
	return StringUtil::Lower(LogicalTypeIdToString(type.id()));
}

const LogicalType &DuckDBPyType::Type() const {
	return type;
}

} // namespace duckdb

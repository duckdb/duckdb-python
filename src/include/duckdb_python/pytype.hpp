#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb/common/types.hpp"

#include <memory>

namespace duckdb {

class ClientContext;
struct DuckDBPyConnection;

class PyGenericAlias : public nb::object {
public:
	using nb::object::object;

public:
	static bool check_(const nb::handle &object);
};

class PyUnionType : public nb::object {
public:
	using nb::object::object;

public:
	static bool check_(const nb::handle &object);
};

//! Value-semantic wrapper around a LogicalType. There is no shared ownership to model -- every factory returns a
//! brand-new, fully-owned type. Bound to Python by value (returned as std::unique_ptr); implicit
//! str/type-object/dict -> DuckDBPyType conversions are handled by nanobind's value caster + the registered
//! implicitly_convertible<>() rules (no custom shared_ptr caster).
class DuckDBPyType {
public:
	explicit DuckDBPyType(LogicalType type);

public:
	static void Initialize(nb::handle &m);

	//! Convert a Python object (an existing DuckDBPyType, a type string, a Python type object such as `int`, or a
	//! dict describing a struct) into an owned DuckDBPyType. An existing DuckDBPyType is copied (value semantics);
	//! anything else is routed through the registered Python constructor, which drives the same factories as the
	//! registered implicit conversions. Returns false (clearing any pending Python error) when the object can't be
	//! converted, so a caller can raise a context-specific message.
	static bool TryConvert(const nb::object &object, std::unique_ptr<DuckDBPyType> &result);
	//! Converts any form TryConvert accepts, resolving type names against the connection, or the default connection
	//! when it is null. Raises the conversion's own error instead of returning false.
	static LogicalType Resolve(const nb::object &object, const std::shared_ptr<DuckDBPyConnection> &connection);
	//! Parses a type name on the context. A name that does not parse raises its own error and leaves the caller's
	//! open transaction usable.
	static LogicalType ParseTypeName(ClientContext &context, const string &name);

public:
	bool Equals(const DuckDBPyType &other) const;
	bool EqualsString(const string &type_str) const;
	std::unique_ptr<DuckDBPyType> GetAttribute(const string &name) const;
	nb::list Children() const;
	string ToString() const;
	const LogicalType &Type() const;
	string GetId() const;

private:
private:
	LogicalType type;
};

} // namespace duckdb

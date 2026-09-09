#include "duckdb_python/exceptions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/list.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/nb/casters.hpp"

namespace nb = nanobind;

namespace duckdb {

class Warning : public std::exception {};

// This is the error structure defined in the DBAPI spec
// StandardError
// |__ Warning
// |__ Error
//    |__ InterfaceError
//    |__ DatabaseError
//       |__ DataError
//       |__ OperationalError
//       |__ IntegrityError
//       |__ InternalError
//       |__ ProgrammingError
//       |__ NotSupportedError
//===--------------------------------------------------------------------===//
// Base Error
//===--------------------------------------------------------------------===//
class PyError : public std::runtime_error {
public:
	explicit PyError(const string &err) : std::runtime_error(err) {
	}
};

class DatabaseError : public PyError {
public:
	explicit DatabaseError(const string &err) : PyError(err) {
	}
};

//===--------------------------------------------------------------------===//
// Unknown Errors
//===--------------------------------------------------------------------===//
class PyFatalException : public DatabaseError {
public:
	explicit PyFatalException(const string &err) : DatabaseError(err) {
	}
};

class PyInterruptException : public DatabaseError {
public:
	explicit PyInterruptException(const string &err) : DatabaseError(err) {
	}
};

class PyPermissionException : public DatabaseError {
public:
	explicit PyPermissionException(const string &err) : DatabaseError(err) {
	}
};

class PySequenceException : public DatabaseError {
public:
	explicit PySequenceException(const string &err) : DatabaseError(err) {
	}
};

class PyDependencyException : public DatabaseError {
public:
	explicit PyDependencyException(const string &err) : DatabaseError(err) {
	}
};

//===--------------------------------------------------------------------===//
// Data Error
//===--------------------------------------------------------------------===//
class DataError : public DatabaseError {
public:
	explicit DataError(const string &err) : DatabaseError(err) {
	}
};

class PyOutOfRangeException : public DataError {
public:
	explicit PyOutOfRangeException(const string &err) : DataError(err) {
	}
};

class PyConversionException : public DataError {
public:
	explicit PyConversionException(const string &err) : DataError(err) {
	}
};

class PyTypeMismatchException : public DataError {
public:
	explicit PyTypeMismatchException(const string &err) : DataError(err) {
	}
};

//===--------------------------------------------------------------------===//
// Operational Error
//===--------------------------------------------------------------------===//
class OperationalError : public DatabaseError {
public:
	explicit OperationalError(const string &err) : DatabaseError(err) {
	}
};

class PyTransactionException : public OperationalError {
public:
	explicit PyTransactionException(const string &err) : OperationalError(err) {
	}
};

class PyOutOfMemoryException : public OperationalError {
public:
	explicit PyOutOfMemoryException(const string &err) : OperationalError(err) {
	}
};

class PyConnectionException : public OperationalError {
public:
	explicit PyConnectionException(const string &err) : OperationalError(err) {
	}
};

class PySerializationException : public OperationalError {
public:
	explicit PySerializationException(const string &err) : OperationalError(err) {
	}
};

class PyIOException : public OperationalError {
public:
	explicit PyIOException(const string &err) : OperationalError(err) {
	}
};

class PyHTTPException : public PyIOException {
public:
	explicit PyHTTPException(const string &err) : PyIOException(err) {
	}
};

//===--------------------------------------------------------------------===//
// Integrity Error
//===--------------------------------------------------------------------===//
class IntegrityError : public DatabaseError {
public:
	explicit IntegrityError(const string &err) : DatabaseError(err) {
	}
};

class PyConstraintException : public IntegrityError {
public:
	explicit PyConstraintException(const string &err) : IntegrityError(err) {
	}
};

//===--------------------------------------------------------------------===//
// Internal Error
//===--------------------------------------------------------------------===//
class InternalError : public DatabaseError {
public:
	explicit InternalError(const string &err) : DatabaseError(err) {
	}
};

class PyInternalException : public InternalError {
public:
	explicit PyInternalException(const string &err) : InternalError(err) {
	}
};

//===--------------------------------------------------------------------===//
// Programming Error
//===--------------------------------------------------------------------===//
class ProgrammingError : public DatabaseError {
public:
	explicit ProgrammingError(const string &err) : DatabaseError(err) {
	}
};

class PyParserException : public ProgrammingError {
public:
	explicit PyParserException(const string &err) : ProgrammingError(err) {
	}
};

class PySyntaxException : public ProgrammingError {
public:
	explicit PySyntaxException(const string &err) : ProgrammingError(err) {
	}
};

class PyBinderException : public ProgrammingError {
public:
	explicit PyBinderException(const string &err) : ProgrammingError(err) {
	}
};

class PyInvalidInputException : public ProgrammingError {
public:
	explicit PyInvalidInputException(const string &err) : ProgrammingError(err) {
	}
};

class PyInvalidTypeException : public ProgrammingError {
public:
	explicit PyInvalidTypeException(const string &err) : ProgrammingError(err) {
	}
};

class PyCatalogException : public ProgrammingError {
public:
	explicit PyCatalogException(const string &err) : ProgrammingError(err) {
	}
};

//===--------------------------------------------------------------------===//
// Not Supported Error
//===--------------------------------------------------------------------===//
class NotSupportedError : public DatabaseError {
public:
	explicit NotSupportedError(const string &err) : DatabaseError(err) {
	}
};

class PyNotImplementedException : public NotSupportedError {
public:
	explicit PyNotImplementedException(const string &err) : NotSupportedError(err) {
	}
};

//===--------------------------------------------------------------------===//
// PyThrowException
//===--------------------------------------------------------------------===//
void PyThrowException(ErrorData &error, PyObject *http_exception) {
	switch (error.Type()) {
	case ExceptionType::HTTP: {
		// construct exception object
		auto exc_msg = error.Message();
		auto e = nb::handle(http_exception)(nb::str(exc_msg.c_str(), exc_msg.size()));

		auto headers = nb::dict();
		for (auto &entry : error.ExtraInfo()) {
			if (entry.first == "status_code") {
				e.attr("status_code") = std::stoi(entry.second);
			} else if (entry.first == "response_body" || entry.first == "body") {
				e.attr("body") = entry.second;
			} else if (entry.first == "reason") {
				e.attr("reason") = entry.second;
			} else if (StringUtil::StartsWith(entry.first, "header_")) {
				auto header_name = entry.first.substr(7);
				headers[nb::str(header_name.c_str(), header_name.size())] = entry.second;
			}
		}
		e.attr("headers") = std::move(headers);

		// "throw" exception object
		PyErr_SetObject(http_exception, e.ptr());
		break;
	}
	case ExceptionType::CATALOG:
		throw PyCatalogException(error.Message());
	case ExceptionType::FATAL:
		throw PyFatalException(error.Message());
	case ExceptionType::INTERRUPT:
		throw PyInterruptException(error.Message());
	case ExceptionType::PERMISSION:
		throw PyPermissionException(error.Message());
	case ExceptionType::SEQUENCE:
		throw PySequenceException(error.Message());
	case ExceptionType::DEPENDENCY:
		throw PyDependencyException(error.Message());
	case ExceptionType::OUT_OF_RANGE:
		throw PyOutOfRangeException(error.Message());
	case ExceptionType::CONVERSION:
		throw PyConversionException(error.Message());
	case ExceptionType::MISMATCH_TYPE:
		throw PyTypeMismatchException(error.Message());
	case ExceptionType::TRANSACTION:
		throw PyTransactionException(error.Message());
	case ExceptionType::OUT_OF_MEMORY:
		throw PyOutOfMemoryException(error.Message());
	case ExceptionType::CONNECTION:
		throw PyConnectionException(error.Message());
	case ExceptionType::SERIALIZATION:
		throw PySerializationException(error.Message());
	case ExceptionType::CONSTRAINT:
		throw PyConstraintException(error.Message());
	case ExceptionType::INTERNAL:
		throw PyInternalException(error.Message());
	case ExceptionType::PARSER:
		throw PyParserException(error.Message());
	case ExceptionType::SYNTAX:
		throw PySyntaxException(error.Message());
	case ExceptionType::IO:
		throw PyIOException(error.Message());
	case ExceptionType::BINDER:
		throw PyBinderException(error.Message());
	case ExceptionType::INVALID_INPUT:
		throw PyInvalidInputException(error.Message());
	case ExceptionType::INVALID_TYPE:
		throw PyInvalidTypeException(error.Message());
	case ExceptionType::NOT_IMPLEMENTED:
		throw PyNotImplementedException(error.Message());
	default:
		throw PyError(error.RawMessage());
	}
}

static void UnsetPythonException() {
	if (PyErr_Occurred()) {
		PyErr_Clear();
	}
}

/**
 * @see https://peps.python.org/pep-0249/#exceptions
 */
void RegisterExceptions(const nb::module_ &m) {
	// The base class is mapped to Error in python to somewhat match the DBAPI 2.0 specifications
	nb::exception<Warning>(m, "Warning");
	auto error = nb::exception<PyError>(m, "Error").ptr();
	auto db_error = nb::exception<DatabaseError>(m, "DatabaseError", error).ptr();

	// order of declaration matters, and this needs to be checked last
	// Unknown
	nb::exception<PyFatalException>(m, "FatalException", db_error);
	nb::exception<PyInterruptException>(m, "InterruptException", db_error);
	nb::exception<PyPermissionException>(m, "PermissionException", db_error);
	nb::exception<PySequenceException>(m, "SequenceException", db_error);
	nb::exception<PyDependencyException>(m, "DependencyException", db_error);

	// DataError
	auto data_error = nb::exception<DataError>(m, "DataError", db_error).ptr();
	nb::exception<PyOutOfRangeException>(m, "OutOfRangeException", data_error);
	nb::exception<PyConversionException>(m, "ConversionException", data_error);
	// no unknown type error, or decimal type
	nb::exception<PyTypeMismatchException>(m, "TypeMismatchException", data_error);

	// OperationalError
	auto operational_error = nb::exception<OperationalError>(m, "OperationalError", db_error).ptr();
	nb::exception<PyTransactionException>(m, "TransactionException", operational_error);
	nb::exception<PyOutOfMemoryException>(m, "OutOfMemoryException", operational_error);
	nb::exception<PyConnectionException>(m, "ConnectionException", operational_error);
	// no object size error
	// no null pointer errors
	auto io_exception = nb::exception<PyIOException>(m, "IOException", operational_error).ptr();
	nb::exception<PySerializationException>(m, "SerializationException", operational_error);

	// Use a raw pointer to avoid destructor running after Python finalization.
	// The module holds a reference to the exception type, keeping it alive.
	static PyObject *HTTP_EXCEPTION = nullptr;
	{
		auto http_exc = nb::exception<PyHTTPException>(m, "HTTPException", io_exception);
		HTTP_EXCEPTION = http_exc.ptr();
		const auto string_type = (nb::str("")).type();
		const auto Dict = nb::module_::import_("typing").attr("Dict");
		// nanobind nb::dict has no kwargs constructor; build the annotations dict explicitly.
		nb::dict annotations;
		annotations["status_code"] = (nb::int_(0)).type();
		annotations["body"] = string_type;
		annotations["reason"] = string_type;
		annotations["headers"] = Dict[nb::make_tuple(string_type, string_type)];
		http_exc.attr("__annotations__") = annotations;
		http_exc.doc() = "Thrown when an error occurs in the httpfs extension, or whilst downloading an extension.";
	}

	// IntegrityError
	auto integrity_error = nb::exception<IntegrityError>(m, "IntegrityError", db_error).ptr();
	nb::exception<PyConstraintException>(m, "ConstraintException", integrity_error);

	// InternalError
	auto internal_error = nb::exception<InternalError>(m, "InternalError", db_error).ptr();
	nb::exception<PyInternalException>(m, "InternalException", internal_error);

	//// ProgrammingError
	auto programming_error = nb::exception<ProgrammingError>(m, "ProgrammingError", db_error).ptr();
	nb::exception<PyParserException>(m, "ParserException", programming_error);
	nb::exception<PySyntaxException>(m, "SyntaxException", programming_error);
	nb::exception<PyBinderException>(m, "BinderException", programming_error);
	nb::exception<PyInvalidInputException>(m, "InvalidInputException", programming_error);
	nb::exception<PyInvalidTypeException>(m, "InvalidTypeException", programming_error);
	// no type for expression exceptions?
	nb::exception<PyCatalogException>(m, "CatalogException", programming_error);

	// NotSupportedError
	auto not_supported_error = nb::exception<NotSupportedError>(m, "NotSupportedError", db_error).ptr();
	nb::exception<PyNotImplementedException>(m, "NotImplementedException", not_supported_error);

	nb::register_exception_translator([](const std::exception_ptr &p, void *) {
		try {
			if (p) {
				std::rethrow_exception(p);
			}
		} catch (const duckdb::Exception &ex) {
			duckdb::ErrorData error(ex);
			UnsetPythonException();
			PyThrowException(error, HTTP_EXCEPTION);
		} catch (const nb::builtin_exception &ex) {
			// These represent Python exceptions, we don't want to catch these
			throw;
		} catch (const std::exception &ex) {
			duckdb::ErrorData error(ex);
			if (error.Type() == ExceptionType::INVALID) {
				// we need to pass non-DuckDB exceptions through as-is
				throw;
			}
			UnsetPythonException();
			PyThrowException(error, HTTP_EXCEPTION);
		}
	});
}
} // namespace duckdb

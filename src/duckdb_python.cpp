#include "duckdb_python/nb/casters.hpp"

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/parser/parser.hpp"

#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyconnection/seam.hpp"
#include "duckdb_python/pystatement.hpp"
#include "duckdb_python/exceptions.hpp"
#include "duckdb_python/typing.hpp"
#include "duckdb_python/functional.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb_python/nb/conversions/exception_handling_enum.hpp"
#include "duckdb_python/nb/conversions/python_udf_type_enum.hpp"
#include "duckdb_python/nb/conversions/python_csv_line_terminator_enum.hpp"
#include "duckdb/common/enums/statement_type.hpp"
#include "duckdb/common/adbc/adbc-init.hpp"
#include "duckdb_python/arrow/predicate_translation.hpp"
#include "duckdb_python/arrow/python_arrow_scan.hpp"

#ifndef DUCKDB_PYTHON_LIB_NAME
#define DUCKDB_PYTHON_LIB_NAME _duckdb
#endif

// Defined by the static extension loader generated at configure time from LINK_EXTENSIONS
// (see duckdb_python_link_extensions in cmake/duckdb_loader.cmake)
extern "C" int32_t duckdb_register_static_extensions(void);

namespace duckdb {

enum PySQLTokenType : uint8_t {
	PY_SQL_TOKEN_IDENTIFIER = 0,
	PY_SQL_TOKEN_NUMERIC_CONSTANT,
	PY_SQL_TOKEN_STRING_CONSTANT,
	PY_SQL_TOKEN_OPERATOR,
	PY_SQL_TOKEN_KEYWORD,
	PY_SQL_TOKEN_COMMENT
};

static nb::list PyTokenize(const string &query) {
	auto tokens = Parser::Tokenize(query);
	nb::list result;
	for (auto &token : tokens) {
		// nanobind tuples are immutable; compute the token type then build the 2-tuple with make_tuple
		PySQLTokenType token_type = PY_SQL_TOKEN_IDENTIFIER;
		switch (token.type) {
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_IDENTIFIER:
			token_type = PY_SQL_TOKEN_IDENTIFIER;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_NUMERIC_CONSTANT:
			token_type = PY_SQL_TOKEN_NUMERIC_CONSTANT;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_STRING_CONSTANT:
			token_type = PY_SQL_TOKEN_STRING_CONSTANT;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_OPERATOR:
			token_type = PY_SQL_TOKEN_OPERATOR;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_KEYWORD:
			token_type = PY_SQL_TOKEN_KEYWORD;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT:
			token_type = PY_SQL_TOKEN_COMMENT;
			break;
		default:
			break;
		}
		result.append(nb::make_tuple(token.start, token_type));
	}
	return result;
}

static void InitializeConnectionMethods(nb::module_ &m) {

	// START_OF_CONNECTION_METHODS
	m.def(
	    "cursor",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Cursor();
	    },
	    "Create a duplicate of the current connection", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "register_filesystem",
	    [](AbstractFileSystem filesystem, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->RegisterFilesystem(filesystem);
	    },
	    "Register a fsspec compliant filesystem", nb::arg("filesystem"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "unregister_filesystem",
	    [](const nb::str &name, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->UnregisterFilesystem(name);
	    },
	    "Unregister a filesystem", nb::arg("name"), nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "list_filesystems",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ListFilesystems();
	    },
	    "List registered filesystems, including builtin ones", nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "filesystem_is_registered",
	    [](const string &name, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FileSystemIsRegistered(name);
	    },
	    "Check if a filesystem with the provided name is currently registered", nb::arg("name"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "get_profiling_information",
	    [](const std::string &format, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetProfilingInformation(format);
	    },
	    "Get profiling information from a query", nb::kw_only(), nb::arg("format") = "json",
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "enable_profiling",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->EnableProfiling();
	    },
	    "Enable profiling for the current connection", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "disable_profiling",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->DisableProfiling();
	    },
	    "Disable profiling for the current connection", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "create_function",
	    [](const string &name, const nb::callable &udf, const nb::object &arguments = nb::none(),
	       const nb::object &return_type = nb::none(), PythonUDFType type = PythonUDFType::NATIVE,
	       FunctionNullHandling null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING,
	       PythonExceptionHandling exception_handling = PythonExceptionHandling::FORWARD_ERROR,
	       bool side_effects = false, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RegisterScalarUDF(name, udf, arguments, return_type, type, null_handling, exception_handling,
		                                   side_effects);
	    },
	    "Create a DuckDB function out of the passing in Python function so it can be used in queries", nb::arg("name"),
	    nb::arg("function"), nb::arg("parameters") = nb::none(), nb::arg("return_type").none() = nb::none(),
	    nb::kw_only(), nb::arg("type") = PythonUDFType::NATIVE,
	    nb::arg("null_handling") = FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    nb::arg("exception_handling") = PythonExceptionHandling::FORWARD_ERROR, nb::arg("side_effects") = false,
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "remove_function",
	    [](const string &name, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->UnregisterUDF(name);
	    },
	    "Remove a previously created function", nb::arg("name"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "sqltype",
	    [](const string &type_str, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Type(type_str);
	    },
	    "Create a type object by parsing the 'type_str' string", nb::arg("type_str"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "dtype",
	    [](const string &type_str, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Type(type_str);
	    },
	    "Create a type object by parsing the 'type_str' string", nb::arg("type_str"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "type",
	    [](const string &type_str, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Type(type_str);
	    },
	    "Create a type object by parsing the 'type_str' string", nb::arg("type_str"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "array_type",
	    [](const DuckDBPyType &type, idx_t size, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ArrayType(type, size);
	    },
	    "Create an array type object of 'type'", nb::arg("type"), nb::arg("size"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "list_type",
	    [](const DuckDBPyType &type, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ListType(type);
	    },
	    "Create a list type object of 'type'", nb::arg("type"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "union_type",
	    [](const nb::object &members, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->UnionType(members);
	    },
	    "Create a union type object from 'members'", nb::arg("members"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "string_type",
	    [](const string &collation = string(), std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->StringType(collation);
	    },
	    "Create a string type with an optional collation", nb::arg("collation") = "", nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "enum_type",
	    [](const string &name, const DuckDBPyType &type, const nb::list &values_p,
	       std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->EnumType(name, type, values_p);
	    },
	    "Create an enum type of underlying 'type', consisting of the list of 'values'", nb::arg("name"),
	    nb::arg("type"), nb::arg("values"), nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "decimal_type",
	    [](int width, int scale, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->DecimalType(width, scale);
	    },
	    "Create a decimal type with 'width' and 'scale'", nb::arg("width"), nb::arg("scale"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "struct_type",
	    [](const nb::object &fields, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->StructType(fields);
	    },
	    "Create a struct type object from 'fields'", nb::arg("fields"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "row_type",
	    [](const nb::object &fields, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->StructType(fields);
	    },
	    "Create a struct type object from 'fields'", nb::arg("fields"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "map_type",
	    [](const DuckDBPyType &key_type, const DuckDBPyType &value_type,
	       std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->MapType(key_type, value_type);
	    },
	    "Create a map type object from 'key_type' and 'value_type'", nb::arg("key"), nb::arg("value"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "duplicate",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Cursor();
	    },
	    "Create a duplicate of the current connection", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "execute",
	    [](const nb::object &query, nb::object params = nb::list(),
	       std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Execute(query, params);
	    },
	    "Execute the given SQL query, optionally using prepared statements with parameters set. A query runs until "
	    "its first rows are ready, so an error there raises here; every other statement runs to its end",
	    nb::arg("query"), nb::arg("parameters") = nb::none(), nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "executemany",
	    [](const nb::object &query, nb::object params = nb::list(),
	       std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ExecuteMany(query, params);
	    },
	    "Execute the given prepared statement multiple times using the list of parameter sets in parameters",
	    nb::arg("query"), nb::arg("parameters") = nb::none(), nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "close",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->Close();
	    },
	    "Close the connection", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "interrupt",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->Interrupt();
	    },
	    "Interrupt pending operations", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "complete",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->Complete();
	    },
	    "Run the result of execute() to its end without reading its rows", nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "abort",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->Abort();
	    },
	    "Cancel the rest of the result of execute(); a statement that may write fails an open transaction",
	    nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "query_progress",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->QueryProgress();
	    },
	    "Query progress of pending operation", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "fetchone",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchOne();
	    },
	    "Fetch a single row from a result following execute", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "fetchmany",
	    [](idx_t size, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchMany(size);
	    },
	    "Fetch the next set of rows from a result following execute", nb::arg("size") = 1, nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "fetchall",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchAll();
	    },
	    "Fetch all rows from a result following execute", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "fetchnumpy",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchNumpy();
	    },
	    "Fetch a result as list of NumPy arrays following execute", nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "fetchdf",
	    [](bool date_as_object, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", nb::kw_only(), nb::arg("date_as_object") = false,
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "fetch_df",
	    [](bool date_as_object, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", nb::kw_only(), nb::arg("date_as_object") = false,
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "df",
	    [](bool date_as_object, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", nb::kw_only(), nb::arg("date_as_object") = false,
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "fetch_df_chunk",
	    [](const idx_t vectors_per_chunk = 1, bool date_as_object = false,
	       std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDFChunk(vectors_per_chunk, date_as_object);
	    },
	    "Fetch a chunk of the result as DataFrame following execute()", nb::arg("vectors_per_chunk") = 1, nb::kw_only(),
	    nb::arg("date_as_object") = false, nb::arg("connection").none() = nb::none());
	m.def(
	    "torch",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchPyTorch();
	    },
	    "Fetch a result as dict of PyTorch Tensors following execute()", nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "tf",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchTF();
	    },
	    "Fetch a result as dict of TensorFlow Tensors following execute()", nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "begin",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Begin();
	    },
	    "Start a new transaction", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "commit",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Commit();
	    },
	    "Commit changes performed within a transaction", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "rollback",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Rollback();
	    },
	    "Roll back changes performed within a transaction", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "checkpoint",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Checkpoint();
	    },
	    "Synchronizes data in the write-ahead log (WAL) to the database data file (no-op for in-memory connections)",
	    nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "append",
	    [](const string &name, const PandasDataFrame &value, bool by_name,
	       std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Append(name, value, by_name);
	    },
	    "Append the passed DataFrame to the named table", nb::arg("table_name"), nb::arg("df"), nb::kw_only(),
	    nb::arg("by_name") = false, nb::arg("connection").none() = nb::none());
	m.def(
	    "appender",
	    [](const string &table, const std::optional<nb::list> &columns, const string &action,
	       const std::optional<nb::list> &key, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->CreateAppender(table, columns, action, key);
	    },
	    "Create an appender that buffers rows and inserts, upserts, updates or deletes them in the named table at each "
	    "flush",
	    nb::arg("table"), nb::kw_only(), nb::arg("columns").none() = nb::none(), nb::arg("action") = "insert",
	    nb::arg("key").none() = nb::none(), nb::arg("connection").none() = nb::none());
	m.def(
	    "query_appender",
	    [](const string &dml, const nb::dict &columns, const std::optional<string> &source,
	       std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->CreateQueryAppender(dml, columns, source);
	    },
	    "Create an appender that buffers rows of the declared columns and runs the DML over them at each flush",
	    nb::arg("dml"), nb::arg("columns"), nb::kw_only(), nb::arg("source").none() = nb::none(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "register",
	    [](const string &name, const nb::object &python_object, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RegisterPythonObject(name, python_object);
	    },
	    "Register the passed Python Object value for querying with a view", nb::arg("view_name"),
	    nb::arg("python_object"), nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "unregister",
	    [](const string &name, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->UnregisterPythonObject(name);
	    },
	    "Unregister the view name", nb::arg("view_name"), nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "extract_statements",
	    [](const string &query, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ExtractStatements(query);
	    },
	    "Parse the query string and extract the Statement object(s) produced", nb::arg("query"), nb::kw_only(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "get_table_names",
	    [](const string &query, bool qualified, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetTableNames(query, qualified);
	    },
	    "Extract the required table names from a query", nb::arg("query"), nb::kw_only(), nb::arg("qualified") = false,
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "install_extension",
	    [](const string &extension, bool force_install = false, const nb::object &repository = nb::none(),
	       const nb::object &repository_url = nb::none(), const nb::object &version = nb::none(),
	       std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->InstallExtension(extension, force_install, repository, repository_url, version);
	    },
	    "Install an extension by name, with an optional version and/or repository to get the extension from",
	    nb::arg("extension"), nb::kw_only(), nb::arg("force_install") = false, nb::arg("repository") = nb::none(),
	    nb::arg("repository_url") = nb::none(), nb::arg("version") = nb::none(),
	    nb::arg("connection").none() = nb::none());
	m.def(
	    "load_extension",
	    [](const string &extension, std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->LoadExtension(extension);
	    },
	    "Load an installed extension", nb::arg("extension"), nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "description",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetDescription();
	    },
	    "Get result set attributes, mainly column names", nb::kw_only(), nb::arg("connection").none() = nb::none());
	m.def(
	    "rowcount",
	    [](std::shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetRowcount();
	    },
	    "Get result set row count", nb::kw_only(), nb::arg("connection").none() = nb::none());
	// END_OF_CONNECTION_METHODS

	m.def(
	    "df",
	    [](bool date_as_object, std::shared_ptr<DuckDBPyConnection> conn) -> PandasDataFrame {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", nb::kw_only(), nb::arg("date_as_object") = false,
	    nb::arg("connection").none() = nb::none());
}

static void RegisterStatementType(nb::handle &m) {
	auto statement_type = nb::enum_<duckdb::StatementType>(m, "StatementType");
	static const duckdb::StatementType TYPES[] = {
	    duckdb::StatementType::INVALID_STATEMENT,       duckdb::StatementType::SELECT_STATEMENT,
	    duckdb::StatementType::INSERT_STATEMENT,        duckdb::StatementType::UPDATE_STATEMENT,
	    duckdb::StatementType::CREATE_STATEMENT,        duckdb::StatementType::DELETE_STATEMENT,
	    duckdb::StatementType::PREPARE_STATEMENT,       duckdb::StatementType::EXECUTE_STATEMENT,
	    duckdb::StatementType::ALTER_STATEMENT,         duckdb::StatementType::TRANSACTION_STATEMENT,
	    duckdb::StatementType::COPY_STATEMENT,          duckdb::StatementType::ANALYZE_STATEMENT,
	    duckdb::StatementType::VARIABLE_SET_STATEMENT,  duckdb::StatementType::CREATE_FUNC_STATEMENT,
	    duckdb::StatementType::EXPLAIN_STATEMENT,       duckdb::StatementType::DROP_STATEMENT,
	    duckdb::StatementType::EXPORT_STATEMENT,        duckdb::StatementType::PRAGMA_STATEMENT,
	    duckdb::StatementType::VACUUM_STATEMENT,        duckdb::StatementType::CALL_STATEMENT,
	    duckdb::StatementType::SET_STATEMENT,           duckdb::StatementType::LOAD_STATEMENT,
	    duckdb::StatementType::RELATION_STATEMENT,      duckdb::StatementType::EXTENSION_STATEMENT,
	    duckdb::StatementType::LOGICAL_PLAN_STATEMENT,  duckdb::StatementType::ATTACH_STATEMENT,
	    duckdb::StatementType::DETACH_STATEMENT,        duckdb::StatementType::MULTI_STATEMENT,
	    duckdb::StatementType::COPY_DATABASE_STATEMENT, duckdb::StatementType::MERGE_INTO_STATEMENT};
	static const idx_t AMOUNT = sizeof(TYPES) / sizeof(duckdb::StatementType);
	for (idx_t i = 0; i < AMOUNT; i++) {
		auto &type = TYPES[i];
		statement_type.value(StatementTypeToString(type).c_str(), type);
	}
	statement_type.export_values();
}

static void RegisterExpectedResultType(nb::handle &m) {
	auto expected_return_type = nb::enum_<duckdb::StatementReturnType>(m, "ExpectedResultType");
	static const duckdb::StatementReturnType TYPES[] = {duckdb::StatementReturnType::QUERY_RESULT,
	                                                    duckdb::StatementReturnType::CHANGED_ROWS,
	                                                    duckdb::StatementReturnType::NOTHING};
	static const idx_t AMOUNT = sizeof(TYPES) / sizeof(duckdb::StatementReturnType);
	for (idx_t i = 0; i < AMOUNT; i++) {
		auto &type = TYPES[i];
		expected_return_type.value(StatementReturnTypeToString(type).c_str(), type);
	}
	expected_return_type.export_values();
}

// ######################################################################
// Force inclusion of symbols that are not referenced by any code in the
// Python module but must be present in the shared object.
//
// duckdb_adbc_init: entrypoint for the ADBC driver. Not called from Python
//   code, but loaded by adbc_driver_manager via dlsym/GetProcAddress.
//
// Without this, the linker may strip these as dead code.
extern "C" {
NB_EXPORT void *_force_symbol_inclusion() {
	static void *symbols[] = {
	    (void *)&duckdb_adbc_init,
	};
	return (void *)symbols;
}
};

NB_MODULE(DUCKDB_PYTHON_LIB_NAME, m) { // NOLINT
	// DO NOT REMOVE: the below forces that we include all symbols we want to export
	volatile auto *keep_alive = _force_symbol_inclusion();
	(void)keep_alive;
	// END

	// A static initializer registered the linked extensions when the library was loaded. Repeating the call is a no-op
	// on a healthy build and makes a broken build fail at import instead of at the first connect.
	if (duckdb_register_static_extensions() != 0) {
		throw std::runtime_error("Failed to register the statically linked DuckDB extensions");
	}

	nb::enum_<duckdb::ExplainType>(m, "ExplainType")
	    .value("STANDARD", duckdb::ExplainType::EXPLAIN_STANDARD)
	    .value("ANALYZE", duckdb::ExplainType::EXPLAIN_ANALYZE)
	    .export_values();

	RegisterStatementType(m);
	RegisterExpectedResultType(m);

	nb::enum_<duckdb::PythonCSVLineTerminator::Type>(m, "CSVLineTerminator")
	    .value("LINE_FEED", duckdb::PythonCSVLineTerminator::Type::LINE_FEED)
	    .value("CARRIAGE_RETURN_LINE_FEED", duckdb::PythonCSVLineTerminator::Type::CARRIAGE_RETURN_LINE_FEED)
	    .export_values();

	nb::enum_<duckdb::PythonExceptionHandling>(m, "PythonExceptionHandling")
	    .value("DEFAULT", duckdb::PythonExceptionHandling::FORWARD_ERROR)
	    .value("RETURN_NULL", duckdb::PythonExceptionHandling::RETURN_NULL)
	    .export_values();

	nb::enum_<duckdb::RenderMode>(m, "RenderMode")
	    .value("ROWS", duckdb::RenderMode::ROWS)
	    .value("COLUMNS", duckdb::RenderMode::COLUMNS)
	    .export_values();

	DuckDBPyTyping::Initialize(m);
	DuckDBPyFunctional::Initialize(m);
	DuckDBPyStatement::Initialize(m);
	DuckDBPyAppender::Initialize(m);
	DuckDBPySeam::Initialize(m);
	DuckDBPyConnection::Initialize(m);

	m.doc() = "DuckDB is an embeddable SQL OLAP Database Management System";
	m.attr("__package__") = "duckdb";
	m.attr("__version__") = std::string(DuckDB::LibraryVersion()).substr(1);
	m.attr("__standard_vector_size__") = DuckDB::StandardVectorSize();
	m.attr("__git_revision__") = DuckDB::SourceID();
	m.attr("__interactive__") = DuckDBPyConnection::DetectAndGetEnvironment();
	m.attr("__jupyter__") = DuckDBPyConnection::IsJupyter();
	m.attr("__formatted_python_version__") = DuckDBPyConnection::FormattedPythonVersion();
	m.def("default_connection", &DuckDBPyConnection::DefaultConnection,
	      "Retrieve the connection currently registered as the default to be used by the module");
	m.def("set_default_connection", &DuckDBPyConnection::SetDefaultConnection,
	      "Register the provided connection as the default to be used by the module", nb::arg("connection"));
	m.attr("apilevel") = "2.0";
	m.attr("threadsafety") = 1;
	m.attr("paramstyle") = "qmark";

	InitializeConnectionMethods(m);

	RegisterExceptions(m);

	m.def("connect", &DuckDBPyConnection::Connect,
	      "Create a DuckDB database instance. Can take a database file name to read/write persistent data and a "
	      "read_only flag if no changes are desired",
	      nb::arg("database") = ":memory:", nb::arg("read_only") = false, nb::arg("config") = nb::dict());
	m.def("tokenize", PyTokenize,
	      "Tokenizes a SQL string, returning a list of (position, type) tuples that can be "
	      "used for e.g., syntax highlighting",
	      nb::arg("query"));
	m.def("_set_arrow_sources", &DuckDBPyConnection::SetArrowSources, nb::arg("sources"));
	m.def("_translate_predicate", &TranslatePredicateForTesting, nb::arg("connection"), nb::arg("predicate"),
	      nb::arg("columns").none() = nb::none(), nb::arg("schema").none() = nb::none(), nb::arg("select") = "*");
	nb::enum_<PySQLTokenType>(m, "token_type")
	    .value("identifier", PySQLTokenType::PY_SQL_TOKEN_IDENTIFIER)
	    .value("numeric_const", PySQLTokenType::PY_SQL_TOKEN_NUMERIC_CONSTANT)
	    .value("string_const", PySQLTokenType::PY_SQL_TOKEN_STRING_CONSTANT)
	    .value("operator", PySQLTokenType::PY_SQL_TOKEN_OPERATOR)
	    .value("keyword", PySQLTokenType::PY_SQL_TOKEN_KEYWORD)
	    .value("comment", PySQLTokenType::PY_SQL_TOKEN_COMMENT)
	    .export_values();

	// we need this because otherwise we try to remove registered_dfs on shutdown when python is already dead.
	// nanobind's capsule has no "callable destructor" ctor; use a non-null sentinel pointer + a cleanup callback
	// that runs when the capsule (held in the module dict) is destroyed at interpreter shutdown.
	static char clean_default_connection_sentinel;
	m.attr("_clean_default_connection") =
	    nb::capsule(&clean_default_connection_sentinel, [](void *) noexcept { DuckDBPyConnection::Cleanup(); });
	WatchInterpreterExit();
}

} // namespace duckdb

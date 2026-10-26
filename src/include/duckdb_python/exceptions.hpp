#pragma once

#include "duckdb_python/nb/casters.hpp"

namespace nb = nanobind;

namespace duckdb {

void RegisterExceptions(const nb::module_ &m);

//! Python text as UTF-8, a lone surrogate escaped the way Python prints it in a traceback; call with the GIL held
string ToUTF8(const nb::object &text);
//! "TypeName: message" for a caught Python error, without its traceback; call with the GIL held
string FormatUDFPythonError(nb::python_error &error);

} // namespace duckdb

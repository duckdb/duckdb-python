#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/nb/conversions/explain_enum.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb/main/relation/query_relation.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/main/relation/view_relation.hpp"
#include "duckdb/function/pragma/pragma_functions.hpp"
#include "duckdb/parser/statement/pragma_statement.hpp"

namespace duckdb {

static void InitializeReadOnlyProperties(nb::class_<DuckDBPyRelation> &m) {
	m.def_prop_ro("type", &DuckDBPyRelation::Type, "Get the type of the relation.")
	    .def_prop_ro("columns", &DuckDBPyRelation::Columns,
	                 "Return a list containing the names of the columns of the relation.")
	    .def_prop_ro("types", &DuckDBPyRelation::ColumnTypes,
	                 "Return a list containing the types of the columns of the relation.")
	    .def_prop_ro("dtypes", &DuckDBPyRelation::ColumnTypes,
	                 "Return a list containing the types of the columns of the relation.")
	    .def_prop_ro("description", &DuckDBPyRelation::Description, "Return the description of the result")
	    .def_prop_ro("alias", &DuckDBPyRelation::GetAlias, "Get the name of the current alias")
	    .def("__len__", &DuckDBPyRelation::Length, "Number of rows in relation.")
	    .def_prop_ro("shape", &DuckDBPyRelation::Shape, " Tuple of # of rows, # of columns in relation.");
}

static void InitializeConsumers(nb::class_<DuckDBPyRelation> &m) {
	// Execute() returns *this (DuckDBPyRelation&). Without reference_internal nanobind applies the default policy to
	// the reference return and *moves* the (move-only) relation into a fresh wrapper, leaving the original with a
	// null rel/result (so a subsequent fetch returns []). reference_internal returns the existing object instead.
	m.def("execute", &DuckDBPyRelation::Execute, nb::rv_policy::reference_internal,
	      "Transform the relation into a result set")
	    .def("close", &DuckDBPyRelation::Close, "Closes the result");

	DefineMethod({"to_parquet", "write_parquet"}, m, &DuckDBPyRelation::ToParquet,
	             "Write the relation object to a Parquet file in 'file_name'", nb::arg("file_name"), nb::kw_only(),
	             nb::arg("compression") = nb::none(), nb::arg("field_ids") = nb::none(),
	             nb::arg("row_group_size_bytes") = nb::none(), nb::arg("row_group_size") = nb::none(),
	             nb::arg("overwrite") = nb::none(), nb::arg("per_thread_output") = nb::none(),
	             nb::arg("use_tmp_file") = nb::none(), nb::arg("partition_by") = nb::none(),
	             nb::arg("write_partition_columns") = nb::none(), nb::arg("append") = nb::none(),
	             nb::arg("filename_pattern") = nb::none(), nb::arg("file_size_bytes") = nb::none());

	DefineMethod(
	    {"to_csv", "write_csv"}, m, &DuckDBPyRelation::ToCSV, "Write the relation object to a CSV file in 'file_name'",
	    nb::arg("file_name"), nb::kw_only(), nb::arg("sep") = nb::none(), nb::arg("na_rep") = nb::none(),
	    nb::arg("header") = nb::none(), nb::arg("quotechar") = nb::none(), nb::arg("escapechar") = nb::none(),
	    nb::arg("date_format") = nb::none(), nb::arg("timestamp_format") = nb::none(), nb::arg("quoting") = nb::none(),
	    nb::arg("encoding") = nb::none(), nb::arg("compression") = nb::none(), nb::arg("overwrite") = nb::none(),
	    nb::arg("per_thread_output") = nb::none(), nb::arg("use_tmp_file") = nb::none(),
	    nb::arg("partition_by") = nb::none(), nb::arg("write_partition_columns") = nb::none());

	m.def("fetchone", &DuckDBPyRelation::FetchOne, "Execute and fetch a single row as a tuple")
	    .def("fetchmany", &DuckDBPyRelation::FetchMany, "Execute and fetch the next set of rows as a list of tuples",
	         nb::arg("size") = 1)
	    .def("fetchall", &DuckDBPyRelation::FetchAll, "Execute and fetch all rows as a list of tuples")
	    .def("fetchnumpy", &DuckDBPyRelation::FetchNumpy,
	         "Execute and fetch all rows as a Python dict mapping each column to one numpy arrays")
	    .def("df", &DuckDBPyRelation::FetchDF, "Execute and fetch all rows as a pandas DataFrame", nb::kw_only(),
	         nb::arg("date_as_object") = false)
	    .def("fetchdf", &DuckDBPyRelation::FetchDF, "Execute and fetch all rows as a pandas DataFrame", nb::kw_only(),
	         nb::arg("date_as_object") = false)
	    .def("to_df", &DuckDBPyRelation::FetchDF, "Execute and fetch all rows as a pandas DataFrame", nb::kw_only(),
	         nb::arg("date_as_object") = false)
	    .def("fetch_df_chunk", &DuckDBPyRelation::FetchDFChunk, "Execute and fetch a chunk of the rows",
	         nb::arg("vectors_per_chunk") = 1, nb::kw_only(), nb::arg("date_as_object") = false)
	    .def("to_arrow_table", &DuckDBPyRelation::ToArrowTable, "Execute and fetch all rows as an Arrow Table",
	         nb::arg("batch_size") = 1000000)
	    .def("to_arrow_reader", &DuckDBPyRelation::ToRecordBatch,
	         "Execute and return an Arrow Record Batch Reader that yields all rows", nb::arg("batch_size") = 1000000)
	    .def("arrow", &DuckDBPyRelation::ToRecordBatch,
	         "Alias of to_arrow_reader(). We recommend using to_arrow_reader() instead.",
	         nb::arg("batch_size") = 1000000)
	    .def(
	        "fetch_arrow_table",
	        [](nb::object &self, idx_t batch_size) {
		        PyErr_WarnEx(PyExc_DeprecationWarning,
		                     "fetch_arrow_table() is deprecated, use to_arrow_table() instead.", 0);
		        return self.attr("to_arrow_table")(batch_size);
	        },
	        "Execute and fetch all rows as an Arrow Table", nb::arg("batch_size") = 1000000)
	    .def("pl", &DuckDBPyRelation::ToPolars, "Execute and fetch all rows as a Polars DataFrame",
	         nb::arg("batch_size") = 1000000, nb::kw_only(), nb::arg("lazy") = false)
	    .def("torch", &DuckDBPyRelation::FetchPyTorch, "Fetch a result as dict of PyTorch Tensors")
	    .def("tf", &DuckDBPyRelation::FetchTF, "Fetch a result as dict of TensorFlow Tensors");
	const char *capsule_docs = R"(
			Execute and return an ArrowArrayStream through the Arrow PyCapsule Interface.

			https://arrow.apache.org/docs/dev/format/CDataInterface/PyCapsuleInterface.html
		)";
	m.def("__arrow_c_stream__", &DuckDBPyRelation::ToArrowCapsule, capsule_docs,
	      nb::arg("requested_schema") = nb::none());
	m.def(
	     "fetch_record_batch",
	     [](nb::object &self, idx_t rows_per_batch) {
		     PyErr_WarnEx(PyExc_DeprecationWarning,
		                  "fetch_record_batch() is deprecated, use to_arrow_reader() instead.", 0);
		     return self.attr("to_arrow_reader")(rows_per_batch);
	     },
	     "Execute and return an Arrow Record Batch Reader that yields all rows", nb::arg("rows_per_batch") = 1000000)
	    .def(
	        "fetch_arrow_reader",
	        [](nb::object &self, idx_t batch_size) {
		        PyErr_WarnEx(PyExc_DeprecationWarning,
		                     "fetch_arrow_reader() is deprecated, use to_arrow_reader() instead.", 0);
		        if (PyErr_Occurred()) {
			        throw nb::python_error();
		        }
		        return self.attr("to_arrow_reader")(batch_size);
	        },
	        "Execute and return an Arrow Record Batch Reader that yields all rows", nb::arg("batch_size") = 1000000);
}

static void InitializeAggregates(nb::class_<DuckDBPyRelation> &m) {
	/* General aggregate functions */
	m.def("any_value", &DuckDBPyRelation::AnyValue, "Returns the first non-null value from a given expression",
	      nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "", nb::arg("projected_columns") = "")
	    .def("arg_max", &DuckDBPyRelation::ArgMax,
	         "Finds the row with the maximum value for a value column and returns the value of that row for an "
	         "argument column",
	         nb::arg("arg_column"), nb::arg("value_column"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("arg_min", &DuckDBPyRelation::ArgMin,
	         "Finds the row with the minimum value for a value column and returns the value of that row for an "
	         "argument column",
	         nb::arg("arg_column"), nb::arg("value_column"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "");
	DefineMethod({"avg", "mean"}, m, &DuckDBPyRelation::Avg, "Computes the average of a given expression",
	             nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	             nb::arg("projected_columns") = "");
	m.def("bit_and", &DuckDBPyRelation::BitAnd, "Computes the bitwise AND of all bits present in a given expression",
	      nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "", nb::arg("projected_columns") = "")
	    .def("bit_or", &DuckDBPyRelation::BitOr, "Computes the bitwise OR of all bits present in a given expression",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("bit_xor", &DuckDBPyRelation::BitXor, "Computes the bitwise XOR of all bits present in a given expression",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("bitstring_agg", &DuckDBPyRelation::BitStringAgg,
	         "Computes a bitstring with bits set for each distinct value in a given expression", nb::arg("expression"),
	         nb::arg("min") = nb::none(), nb::arg("max") = nb::none(), nb::arg("groups") = "",
	         nb::arg("window_spec") = "", nb::arg("projected_columns") = "")
	    .def("bool_and", &DuckDBPyRelation::BoolAnd,
	         "Computes the logical AND of all values present in a given expression", nb::arg("expression"),
	         nb::arg("groups") = "", nb::arg("window_spec") = "", nb::arg("projected_columns") = "")
	    .def("bool_or", &DuckDBPyRelation::BoolOr,
	         "Computes the logical OR of all values present in a given expression", nb::arg("expression"),
	         nb::arg("groups") = "", nb::arg("window_spec") = "", nb::arg("projected_columns") = "")
	    .def("count", &DuckDBPyRelation::Count, "Computes the number of elements present in a given expression",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("value_counts", &DuckDBPyRelation::ValueCounts,
	         "Computes the number of elements present in a given expression, also projecting the original expression",
	         nb::arg("expression"), nb::arg("groups") = "")
	    .def("favg", &DuckDBPyRelation::FAvg,
	         "Computes the average of all values present in a given expression using a more accurate floating point "
	         "summation (Kahan Sum)",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("first", &DuckDBPyRelation::First, "Returns the first value of a given expression", nb::arg("expression"),
	         nb::arg("groups") = "", nb::arg("projected_columns") = "")
	    .def("fsum", &DuckDBPyRelation::FSum,
	         "Computes the sum of all values present in a given expression using a more accurate floating point "
	         "summation (Kahan Sum)",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("geomean", &DuckDBPyRelation::GeoMean,
	         "Computes the geometric mean over all values present in a given expression", nb::arg("expression"),
	         nb::arg("groups") = "", nb::arg("projected_columns") = "")
	    .def("histogram", &DuckDBPyRelation::Histogram,
	         "Computes the histogram over all values present in a given expression", nb::arg("expression"),
	         nb::arg("groups") = "", nb::arg("window_spec") = "", nb::arg("projected_columns") = "")
	    .def("list", &DuckDBPyRelation::List, "Returns a list containing all values present in a given expression",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("last", &DuckDBPyRelation::Last, "Returns the last value of a given expression", nb::arg("expression"),
	         nb::arg("groups") = "", nb::arg("projected_columns") = "")
	    .def("max", &DuckDBPyRelation::Max, "Returns the maximum value present in a given expression",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("min", &DuckDBPyRelation::Min, "Returns the minimum value present in a given expression",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("product", &DuckDBPyRelation::Product, "Returns the product of all values present in a given expression",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("string_agg", &DuckDBPyRelation::StringAgg,
	         "Concatenates the values present in a given expression with a separator", nb::arg("expression"),
	         nb::arg("sep") = ",", nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("sum", &DuckDBPyRelation::Sum, "Computes the sum of all values present in a given expression",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("unique", &DuckDBPyRelation::Unique, "Returns the distinct values in a column.", nb::arg("unique_aggr"));
	/* TODO: Approximate aggregate functions */
	/* TODO: Statistical aggregate functions */
	m.def("median", &DuckDBPyRelation::Median, "Computes the median over all values present in a given expression",
	      nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "", nb::arg("projected_columns") = "")
	    .def("mode", &DuckDBPyRelation::Mode, "Computes the mode over all values present in a given expression",
	         nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "")
	    .def("quantile_cont", &DuckDBPyRelation::QuantileCont,
	         "Computes the interpolated quantile value for a given expression", nb::arg("expression"),
	         nb::arg("q") = 0.5, nb::arg("groups") = "", nb::arg("window_spec") = "",
	         nb::arg("projected_columns") = "");
	DefineMethod({"quantile_disc", "quantile"}, m, &DuckDBPyRelation::QuantileDisc,
	             "Computes the exact quantile value for a given expression", nb::arg("expression"), nb::arg("q") = 0.5,
	             nb::arg("groups") = "", nb::arg("window_spec") = "", nb::arg("projected_columns") = "");
	m.def("stddev_pop", &DuckDBPyRelation::StdPop, "Computes the population standard deviation for a given expression",
	      nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	      nb::arg("projected_columns") = "");
	DefineMethod({"stddev_samp", "stddev", "std"}, m, &DuckDBPyRelation::StdSamp,
	             "Computes the sample standard deviation for a given expression", nb::arg("expression"),
	             nb::arg("groups") = "", nb::arg("window_spec") = "", nb::arg("projected_columns") = "");
	m.def("var_pop", &DuckDBPyRelation::VarPop, "Computes the population variance for a given expression",
	      nb::arg("expression"), nb::arg("groups") = "", nb::arg("window_spec") = "",
	      nb::arg("projected_columns") = "");
	DefineMethod({"var_samp", "variance", "var"}, m, &DuckDBPyRelation::VarSamp,
	             "Computes the sample variance for a given expression", nb::arg("expression"), nb::arg("groups") = "",
	             nb::arg("window_spec") = "", nb::arg("projected_columns") = "");
}

static void InitializeWindowOperators(nb::class_<DuckDBPyRelation> &m) {
	m.def("row_number", &DuckDBPyRelation::RowNumber, "Computes the row number within the partition",
	      nb::arg("window_spec"), nb::arg("projected_columns") = "")
	    .def("rank", &DuckDBPyRelation::Rank, "Computes the rank within the partition", nb::arg("window_spec"),
	         nb::arg("projected_columns") = "");

	DefineMethod({"dense_rank", "rank_dense"}, m, &DuckDBPyRelation::DenseRank,
	             "Computes the dense rank within the partition", nb::arg("window_spec"),
	             nb::arg("projected_columns") = "");
	m.def("percent_rank", &DuckDBPyRelation::PercentRank, "Computes the relative rank within the partition",
	      nb::arg("window_spec"), nb::arg("projected_columns") = "")
	    .def("cume_dist", &DuckDBPyRelation::CumeDist, "Computes the cumulative distribution within the partition",
	         nb::arg("window_spec"), nb::arg("projected_columns") = "")
	    .def("first_value", &DuckDBPyRelation::FirstValue, "Computes the first value within the group or partition",
	         nb::arg("expression"), nb::arg("window_spec") = "", nb::arg("projected_columns") = "")
	    .def("n_tile", &DuckDBPyRelation::NTile, "Divides the partition as equally as possible into num_buckets",
	         nb::arg("window_spec"), nb::arg("num_buckets"), nb::arg("projected_columns") = "")
	    .def("lag", &DuckDBPyRelation::Lag, "Computes the lag within the partition", nb::arg("expression"),
	         nb::arg("window_spec"), nb::arg("offset") = 1, nb::arg("default_value") = "NULL",
	         nb::arg("ignore_nulls") = false, nb::arg("projected_columns") = "")
	    .def("last_value", &DuckDBPyRelation::LastValue, "Computes the last value within the group or partition",
	         nb::arg("expression"), nb::arg("window_spec") = "", nb::arg("projected_columns") = "")
	    .def("lead", &DuckDBPyRelation::Lead, "Computes the lead within the partition", nb::arg("expression"),
	         nb::arg("window_spec"), nb::arg("offset") = 1, nb::arg("default_value") = "NULL",
	         nb::arg("ignore_nulls") = false, nb::arg("projected_columns") = "")
	    .def("nth_value", &DuckDBPyRelation::NthValue, "Computes the nth value within the partition",
	         nb::arg("expression"), nb::arg("window_spec"), nb::arg("offset"), nb::arg("ignore_nulls") = false,
	         nb::arg("projected_columns") = "");
}

static void InitializeSetOperators(nb::class_<DuckDBPyRelation> &m) {
	m.def("union", &DuckDBPyRelation::Union, nb::arg("union_rel"),
	      "Create the set union of this relation object with another relation object in other_rel")
	    .def("except_", &DuckDBPyRelation::Except,
	         "Create the set except of this relation object with another relation object in other_rel",
	         nb::arg("other_rel"))
	    .def("intersect", &DuckDBPyRelation::Intersect,
	         "Create the set intersection of this relation object with another relation object in other_rel",
	         nb::arg("other_rel"));
}

static void InitializeMetaQueries(nb::class_<DuckDBPyRelation> &m) {
	m.def("describe", &DuckDBPyRelation::Describe,
	      "Gives basic statistics (e.g., min, max) and if NULL exists for each column of the relation.")
	    .def(
	        "explain",
	        [](DuckDBPyRelation &self, ExplainType type, const nb::object &format) {
		        // An omitted format (None) maps to "" = auto-select (default, or HTML under Jupyter).
		        string format_str = format.is_none() ? string() : nb::cast<std::string>(nb::str(format));
		        return self.Explain(type, format_str);
	        },
	        nb::arg("type") = ExplainType::EXPLAIN_STANDARD, nb::arg("format") = nb::none());
}

void DuckDBPyRelation::Initialize(nb::handle &m) {
	// nanobind types aren't weak-referenceable by default.
	auto relation_module = nb::class_<DuckDBPyRelation>(m, "DuckDBPyRelation", nb::is_weak_referenceable());
	InitializeReadOnlyProperties(relation_module);
	InitializeAggregates(relation_module);
	InitializeWindowOperators(relation_module);
	InitializeSetOperators(relation_module);
	InitializeMetaQueries(relation_module);
	InitializeConsumers(relation_module);

	relation_module.def("__getattr__", &DuckDBPyRelation::GetAttribute,
	                    "Get a projection relation created from this relation, on the provided column name",
	                    nb::arg("name"));
	relation_module.def("__getitem__", &DuckDBPyRelation::GetAttribute,
	                    "Get a projection relation created from this relation, on the provided column name",
	                    nb::arg("name"));

	relation_module.def("filter", &DuckDBPyRelation::Filter, "Filter the relation object by the filter in filter_expr",
	                    nb::arg("filter_expr"));
	// nanobind forbids a named typed parameter (groups) after nb::args; bind via a lambda that pulls the
	// keyword-only `groups` from **kwargs (preserving `rel.select(*exprs, groups=...)`).
	for (const char *alias : {"select", "project"}) {
		relation_module.def(
		    alias,
		    [](DuckDBPyRelation &self, const nb::args &expr, const nb::kwargs &kwargs) {
			    string groups = "";
			    if (kwargs.contains("groups") && !kwargs["groups"].is_none()) {
				    groups = nb::cast<std::string>(kwargs["groups"]);
			    }
			    return self.Project(expr, groups);
		    },
		    "Project the relation object by the projection in project_expr");
	}
	DefineMethod({"select_types", "select_dtypes"}, relation_module, &DuckDBPyRelation::ProjectFromTypes,
	             "Select columns from the relation, by filtering based on type(s)", nb::arg("types"));

	relation_module.def("__contains__", &DuckDBPyRelation::ContainsColumnByName, nb::arg("name"));

	relation_module
	    .def("set_alias", &DuckDBPyRelation::SetAlias, "Rename the relation object to new alias", nb::arg("alias"))
	    .def("order", &DuckDBPyRelation::Order, "Reorder the relation object by order_expr", nb::arg("order_expr"))
	    .def("sort", &DuckDBPyRelation::Sort, "Reorder the relation object by the provided expressions")
	    .def("aggregate", &DuckDBPyRelation::Aggregate,
	         "Compute the aggregate aggr_expr by the optional groups group_expr on the relation", nb::arg("aggr_expr"),
	         nb::arg("group_expr") = "")
	    .def("apply", &DuckDBPyRelation::GenericAggregator,
	         "Compute the function of a single column or a list of columns by the optional groups on the relation",
	         nb::arg("function_name"), nb::arg("function_aggr"), nb::arg("group_expr") = "",
	         nb::arg("function_parameter") = "", nb::arg("projected_columns") = "")

	    .def("join", &DuckDBPyRelation::Join,
	         "Join the relation object with another relation object in other_rel using the join condition expression "
	         "in join_condition. Types supported are 'inner', 'left', 'right', 'outer', 'semi' and 'anti'",
	         nb::arg("other_rel").none(), nb::arg("condition"), nb::arg("how") = "inner")
	    .def("cross", &DuckDBPyRelation::Cross, "Create cross/cartesian product of two relational objects",
	         nb::arg("other_rel"))

	    .def("distinct", &DuckDBPyRelation::Distinct, "Retrieve distinct rows from this relation object")
	    .def("limit", &DuckDBPyRelation::Limit,
	         "Only retrieve the first n rows from this relation object, starting at offset", nb::arg("n"),
	         nb::arg("offset") = 0)
	    .def("insert", &DuckDBPyRelation::Insert, "Inserts the given values into the relation", nb::arg("values"))
	    .def("update", &DuckDBPyRelation::Update, "Update the given relation with the provided expressions",
	         nb::arg("set"), nb::kw_only(), nb::arg("condition") = nb::none())

	    // This should be deprecated in favor of a replacement scan
	    .def("query", &DuckDBPyRelation::Query,
	         "Run the given SQL query in sql_query on the view named virtual_table_name that refers to the relation "
	         "object",
	         nb::arg("virtual_table_name"), nb::arg("sql_query"))

	    // Aren't these also technically consumers?
	    .def("insert_into", &DuckDBPyRelation::InsertInto,
	         "Inserts the relation object into an existing table named table_name", nb::arg("table_name"));

	DefineMethod({"create", "to_table"}, relation_module, &DuckDBPyRelation::Create,
	             "Creates a new table named table_name with the contents of the relation object",
	             nb::arg("table_name"));

	DefineMethod({"create_view", "to_view"}, relation_module, &DuckDBPyRelation::CreateView,
	             "Creates a view named view_name that refers to the relation object", nb::arg("view_name"),
	             nb::arg("replace") = true);

	relation_module
	    .def("map", &DuckDBPyRelation::Map, nb::arg("map_function"), nb::kw_only(), nb::arg("schema") = nb::none(),
	         "Calls the passed function on the relation")
	    .def("show", &DuckDBPyRelation::Print, "Display a summary of the data", nb::kw_only(),
	         nb::arg("max_width") = nb::none(), nb::arg("max_rows") = nb::none(), nb::arg("max_col_width") = nb::none(),
	         nb::arg("null_value") = nb::none(), nb::arg("render_mode") = nb::none())
	    .def("__duckdb_template__", &DuckDBPyRelation::ToSQL)
	    .def("__str__", &DuckDBPyRelation::ToString)
	    .def("__repr__", &DuckDBPyRelation::ToString);

	relation_module.def("sql_query", &DuckDBPyRelation::ToSQL, "Get the SQL query that is equivalent to the relation");
}

} // namespace duckdb

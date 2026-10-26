"""TPC-H and TPC-DS queries written with the relational API, as tests of how a chain of steps renders.

Each test builds one benchmark query from verbs and expressions and checks three things. The rows equal the query
text run on the same connection, and for TPC-H the shipped answer too. The rendered statement holds the number of
SELECTs the clause slot rules predict for the chain: a step appends to the open SELECT while its slot is empty and no
later slot is filled, and otherwise closes that SELECT as a common table expression. Every column qualified by a join
input's name sits in a SELECT whose FROM is that join, because once a SELECT closes only its output names remain.

The data comes from the directory named by DUCKDB_PYTHON_TPC_DATA, which holds `tpch/` and `tpcds/` with the tables
as Parquet under `sf0.01/` beside `queries.csv` and, for TPC-H, `answers_sf0.01.csv`. Without it the tpch and tpcds
extensions generate the data at scale factor 0.01, and the tests skip when they do not load.
"""

import csv
import dataclasses
import datetime
import decimal
import json
import math
import os
from pathlib import Path

import pytest

import duckdb
from duckdb import col, count_all, fn, rank, sql_expr, when

D = decimal.Decimal
DAYS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")


@dataclasses.dataclass
class Benchmark:
    con: duckdb.DuckDBPyConnection
    queries: dict[int, str]
    answers: dict[int, str]


def _read_csv(path, column):
    with path.open(newline="") as handle:
        return {int(row["query_nr"]): row[column] for row in csv.DictReader(handle)}


def _from_files(name, generator):
    root = os.environ.get("DUCKDB_PYTHON_TPC_DATA")
    con = duckdb.connect()
    if root:
        directory = Path(root).expanduser() / name
        tables = sorted((directory / "sf0.01").glob("*.parquet"))
        if not tables:
            pytest.skip(f"no {name} Parquet files under {directory}")
        for table in tables:
            path = str(table).replace("'", "''")
            con.execute(f"CREATE VIEW {table.stem} AS SELECT * FROM read_parquet('{path}')")
        queries = _read_csv(directory / "queries.csv", "query")
        answers_file = directory / "answers_sf0.01.csv"
        answers = _read_csv(answers_file, "answer") if name == "tpch" and answers_file.exists() else {}
        return Benchmark(con, queries, answers)
    try:
        con.execute(f"LOAD {name}")
    except duckdb.Error:
        con.close()
        pytest.skip(f"the {name} extension does not load")
    con.execute(f"CALL {generator}(sf=0.01)")
    queries = dict(con.execute(f"SELECT query_nr, query FROM {name}_queries()").fetchall())
    answers = {}
    if name == "tpch":
        answers = dict(con.execute("SELECT query_nr, answer FROM tpch_answers() WHERE scale_factor = 0.01").fetchall())
    return Benchmark(con, queries, answers)


@pytest.fixture(scope="module")
def tpch():
    benchmark = _from_files("tpch", "dbgen")
    yield benchmark
    benchmark.con.close()


@pytest.fixture(scope="module")
def tpcds():
    benchmark = _from_files("tpcds", "dsdgen")
    yield benchmark
    benchmark.con.close()


# Decimals and integers compare exactly; doubles to a relative 1e-9, because a parallel sum may add the same
# values in another order and the answers file prints a double with its shortest round trip digits.


def _close(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, float) or isinstance(expected, float):
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
    return actual == expected


def _sort_key(row):
    def cell(value):
        if value is None:
            return (0, "", "")
        if isinstance(value, (float, D)):
            return (1, "number", f"{float(value):.9g}")
        return (1, type(value).__name__, str(value))

    return tuple(cell(value) for value in row)


def assert_rows(actual, expected, *, ordered):
    assert len(actual) == len(expected)
    if not ordered:
        actual, expected = sorted(actual, key=_sort_key), sorted(expected, key=_sort_key)
    for position, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert len(got) == len(want), (position, got, want)
        assert all(_close(g, w) for g, w in zip(got, want, strict=True)), (position, got, want)


def _answer_cell(value, text):
    if value is None:
        return text in ("", "NULL")
    if isinstance(value, bool):
        return str(value).lower() == text.lower()
    if isinstance(value, int):
        return value == int(text)
    if isinstance(value, D):
        return value == D(text)
    if isinstance(value, float):
        return math.isclose(value, float(text), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(value, datetime.date):
        return value.isoformat() == text
    return str(value) == text


def assert_answer(actual, answer):
    lines = answer.split("\n")[1:]
    if lines and not lines[-1]:
        lines.pop()
    assert len(actual) == len(lines)
    for row, line in zip(actual, lines, strict=True):
        cells = line.split("|")
        assert len(row) == len(cells), (row, cells)
        assert all(_answer_cell(v, t) for v, t in zip(row, cells, strict=True)), (row, cells)


def _parsed(con, rel):
    [(text,)] = con.execute("SELECT json_serialize_sql(?)", [rel.sql_query()]).fetchall()
    tree = json.loads(text)
    assert not tree["error"], tree
    [statement] = tree["statements"]
    return statement["node"]


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def count_selects(con, rel):
    return sum(1 for part in _walk(_parsed(con, rel)) if part.get("type") == "SELECT_NODE")


def _join_inputs(table):
    if table.get("type") == "JOIN":
        return _join_inputs(table["left"]) | _join_inputs(table["right"])
    return {(table.get("alias") or table.get("table_name") or "").lower()}


def _misplaced_qualifiers(node, scopes, found):
    if isinstance(node, list):
        for value in node:
            _misplaced_qualifiers(value, scopes, found)
        return
    if not isinstance(node, dict):
        return
    if node.get("class") == "COLUMN_REF" and len(node["column_names"]) > 1:
        qualifier = node["column_names"][0].lower()
        if not any(qualifier in scope for scope in scopes):
            found.append(".".join(node["column_names"]))
        return
    if node.get("type") == "SELECT_NODE":
        table = node.get("from_table") or {}
        own = _join_inputs(table) if table.get("type") == "JOIN" else set()
        inner = [*scopes, own]
        for key, value in node.items():
            # A common table expression's body does not see the FROM of the SELECT it is attached to.
            _misplaced_qualifiers(value, scopes if key == "cte_map" else inner, found)
        return
    for value in node.values():
        _misplaced_qualifiers(value, scopes, found)


def misplaced_qualifiers(con, rel):
    found = []
    _misplaced_qualifiers(_parsed(con, rel), [], found)
    return found


def check(benchmark, number, build, *, ordered):
    con = benchmark.con
    rel, expected_selects = build(con)
    rows = rel.fetchall()
    assert_rows(rows, con.execute(benchmark.queries[number]).fetchall(), ordered=ordered)
    if benchmark.answers:
        assert_answer(rows, benchmark.answers[number])
    assert count_selects(con, rel) == expected_selects
    assert misplaced_qualifiers(con, rel) == []


TPCH = {}
TPCDS = {}


def query(registry, number, *, ordered):
    def register(build):
        registry[number] = (build, ordered)
        return build

    return register


def disc_price():
    return col("l_extendedprice") * (1 - col("l_discount"))


@query(TPCH, 1, ordered=True)
def h01(con):
    rel = (
        con.table("lineitem")
        .filter(col("l_shipdate") <= datetime.date(1998, 9, 2))
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            col("l_quantity").sum().alias("sum_qty"),
            col("l_extendedprice").sum().alias("sum_base_price"),
            disc_price().sum().alias("sum_disc_price"),
            (disc_price() * (1 + col("l_tax"))).sum().alias("sum_charge"),
            col("l_quantity").avg().alias("avg_qty"),
            col("l_extendedprice").avg().alias("avg_price"),
            col("l_discount").avg().alias("avg_disc"),
            count_all().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )
    # The table opens the SELECT; WHERE and GROUP BY append; ORDER BY after the list closes it.
    return rel, 2


@query(TPCH, 2, ordered=True)
def h02(con):
    europe = (
        con.table("partsupp")
        .join(con.table("supplier"), col("s_suppkey") == col("ps_suppkey"))
        .join(con.table("nation"), col("s_nationkey") == col("n_nationkey"))
        .join(con.table("region"), col("n_regionkey") == col("r_regionkey"))
        .filter(col("r_name") == "EUROPE")
    )
    cheapest = (
        europe.group_by(col("ps_partkey").alias("cheapest_partkey"))
        .agg(col("ps_supplycost").min().alias("min_cost"))
        .set_alias("cheapest")
    )
    rel = (
        europe.join(con.table("part"), col("p_partkey") == col("ps_partkey"))
        .join(cheapest, (col("cheapest_partkey") == col("p_partkey")) & (col("ps_supplycost") == col("min_cost")))
        .filter((col("p_size") == 15) & col("p_type").like("%BRASS"))
        .select("s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr", "s_address", "s_phone", "s_comment")
        .sort(col("s_acctbal").desc(), "n_name", "s_name", "p_partkey")
        .limit(100)
    )
    # europe closes as a join input; cheapest appends GROUP BY to its own copy; the outer join and its ORDER BY.
    return rel, 4


@query(TPCH, 3, ordered=True)
def h03(con):
    rel = (
        con.table("customer")
        .join(con.table("orders"), col("c_custkey") == col("o_custkey"))
        .join(con.table("lineitem"), col("l_orderkey") == col("o_orderkey"))
        .filter(
            (col("c_mktsegment") == "BUILDING")
            & (col("o_orderdate") < datetime.date(1995, 3, 15))
            & (col("l_shipdate") > datetime.date(1995, 3, 15))
        )
        .aggregate(
            [col("l_orderkey"), disc_price().sum().alias("revenue"), col("o_orderdate"), col("o_shippriority")],
            "l_orderkey, o_orderdate, o_shippriority",
        )
        .sort(col("revenue").desc(), "o_orderdate")
        .limit(10)
    )
    # Bare joins inline into one FROM; WHERE and GROUP BY append; ORDER BY after the list closes it.
    return rel, 2


@query(TPCH, 4, ordered=True)
def h04(con):
    rel = (
        con.table("orders")
        .filter((col("o_orderdate") >= datetime.date(1993, 7, 1)) & (col("o_orderdate") < datetime.date(1993, 10, 1)))
        .join(
            con.table("lineitem"),
            (col("l_orderkey") == col("o_orderkey")) & (col("l_commitdate") < col("l_receiptdate")),
            how="semi",
        )
        .group_by("o_orderpriority")
        .agg(count_all().alias("order_count"))
        .sort("o_orderpriority")
    )
    # The filtered orders close as a join input; the semi join's SELECT takes GROUP BY; ORDER BY one more.
    return rel, 3


@query(TPCH, 5, ordered=True)
def h05(con):
    rel = (
        con.table("customer")
        .join(con.table("orders"), col("c_custkey") == col("o_custkey"))
        .join(con.table("lineitem"), col("l_orderkey") == col("o_orderkey"))
        .join(
            con.table("supplier"),
            (col("l_suppkey") == col("s_suppkey")) & (col("c_nationkey") == col("s_nationkey")),
        )
        .join(con.table("nation"), col("s_nationkey") == col("n_nationkey"))
        .join(con.table("region"), col("n_regionkey") == col("r_regionkey"))
        .filter(
            (col("r_name") == "ASIA")
            & (col("o_orderdate") >= datetime.date(1994, 1, 1))
            & (col("o_orderdate") < datetime.date(1995, 1, 1))
        )
        .group_by("n_name")
        .agg(disc_price().sum().alias("revenue"))
        .sort(col("revenue").desc())
    )
    # Six bare joins inline into one FROM; WHERE and GROUP BY append; ORDER BY after the list closes it.
    return rel, 2


@query(TPCH, 6, ordered=False)
def h06(con):
    rel = (
        con.table("lineitem")
        .filter(
            (col("l_shipdate") >= datetime.date(1994, 1, 1))
            & (col("l_shipdate") < datetime.date(1995, 1, 1))
            & col("l_discount").between(D("0.05"), D("0.07"))
            & (col("l_quantity") < 24)
        )
        .aggregate([(col("l_extendedprice") * col("l_discount")).sum().alias("revenue")])
    )
    return rel, 1  # The table opens the SELECT; WHERE then the aggregate list append.


@query(TPCH, 7, ordered=True)
def h07(con):
    n1 = con.table("nation").set_alias("n1")
    n2 = con.table("nation").set_alias("n2")
    shipping = (
        con.table("supplier")
        .join(con.table("lineitem"), col("s_suppkey") == col("l_suppkey"))
        .join(con.table("orders"), col("o_orderkey") == col("l_orderkey"))
        .join(con.table("customer"), col("c_custkey") == col("o_custkey"))
        .join(n1, col("s_nationkey") == col("n1", "n_nationkey"))
        .join(n2, col("c_nationkey") == col("n2", "n_nationkey"))
        .filter(
            (
                ((col("n1", "n_name") == "FRANCE") & (col("n2", "n_name") == "GERMANY"))
                | ((col("n1", "n_name") == "GERMANY") & (col("n2", "n_name") == "FRANCE"))
            )
            & col("l_shipdate").between(datetime.date(1995, 1, 1), datetime.date(1996, 12, 31))
        )
        .select(
            col("n1", "n_name").alias("supp_nation"),
            col("n2", "n_name").alias("cust_nation"),
            fn("year", col("l_shipdate")).alias("l_year"),
            disc_price().alias("volume"),
        )
    )
    rel = (
        shipping.group_by("supp_nation", "cust_nation", "l_year")
        .agg(col("volume").sum().alias("revenue"))
        .sort("supp_nation", "cust_nation", "l_year")
    )
    # n1 and n2 read inline; the join one; GROUP BY after its select list one more; ORDER BY one more.
    return rel, 3


@query(TPCH, 8, ordered=True)
def h08(con):
    n1 = con.table("nation").set_alias("n1")
    n2 = con.table("nation").set_alias("n2")
    all_nations = (
        con.table("lineitem")
        .join(con.table("part"), col("p_partkey") == col("l_partkey"))
        .join(con.table("supplier"), col("s_suppkey") == col("l_suppkey"))
        .join(con.table("orders"), col("l_orderkey") == col("o_orderkey"))
        .join(con.table("customer"), col("o_custkey") == col("c_custkey"))
        .join(n1, col("c_nationkey") == col("n1", "n_nationkey"))
        .join(con.table("region"), col("n1", "n_regionkey") == col("r_regionkey"))
        .join(n2, col("s_nationkey") == col("n2", "n_nationkey"))
        .filter(
            (col("r_name") == "AMERICA")
            & col("o_orderdate").between(datetime.date(1995, 1, 1), datetime.date(1996, 12, 31))
            & (col("p_type") == "ECONOMY ANODIZED STEEL")
        )
        .select(
            fn("year", col("o_orderdate")).alias("o_year"),
            disc_price().alias("volume"),
            col("n2", "n_name").alias("nation"),
        )
    )
    brazil = when(col("nation") == "BRAZIL").then(col("volume")).otherwise(0)
    rel = all_nations.group_by("o_year").agg((brazil.sum() / col("volume").sum()).alias("mkt_share")).sort("o_year")
    # n1 and n2 read inline; the join one; GROUP BY after its select list one more; ORDER BY one more.
    return rel, 3


@query(TPCH, 9, ordered=True)
def h09(con):
    profit = (
        con.table("lineitem")
        .join(con.table("part"), col("p_partkey") == col("l_partkey"))
        .join(con.table("supplier"), col("s_suppkey") == col("l_suppkey"))
        .join(
            con.table("partsupp"),
            (col("ps_suppkey") == col("l_suppkey")) & (col("ps_partkey") == col("l_partkey")),
        )
        .join(con.table("orders"), col("o_orderkey") == col("l_orderkey"))
        .join(con.table("nation"), col("s_nationkey") == col("n_nationkey"))
        .filter(col("p_name").like("%green%"))
        .select(
            col("n_name").alias("nation"),
            fn("year", col("o_orderdate")).alias("o_year"),
            (disc_price() - col("ps_supplycost") * col("l_quantity")).alias("amount"),
        )
    )
    rel = (
        profit.group_by("nation", "o_year")
        .agg(col("amount").sum().alias("sum_profit"))
        .sort("nation", col("o_year").desc())
    )
    # The join takes WHERE and its select list; GROUP BY after it closes it; ORDER BY after GROUP BY too.
    return rel, 3


@query(TPCH, 10, ordered=True)
def h10(con):
    rel = (
        con.table("customer")
        .join(con.table("orders"), col("c_custkey") == col("o_custkey"))
        .join(con.table("lineitem"), col("l_orderkey") == col("o_orderkey"))
        .join(con.table("nation"), col("c_nationkey") == col("n_nationkey"))
        .filter(
            (col("o_orderdate") >= datetime.date(1993, 10, 1))
            & (col("o_orderdate") < datetime.date(1994, 1, 1))
            & (col("l_returnflag") == "R")
        )
        .aggregate(
            [
                col("c_custkey"),
                col("c_name"),
                disc_price().sum().alias("revenue"),
                col("c_acctbal"),
                col("n_name"),
                col("c_address"),
                col("c_phone"),
                col("c_comment"),
            ],
            "c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment",
        )
        .sort(col("revenue").desc())
        .limit(20)
    )
    # Bare joins inline into one FROM; WHERE and GROUP BY append; ORDER BY after the list closes it.
    return rel, 2


@query(TPCH, 11, ordered=True)
def h11(con):
    germany = (
        con.table("partsupp")
        .join(con.table("supplier"), col("ps_suppkey") == col("s_suppkey"))
        .join(con.table("nation"), col("s_nationkey") == col("n_nationkey"))
        .filter(col("n_name") == "GERMANY")
    )
    value = (col("ps_supplycost") * col("ps_availqty")).sum()
    per_part = germany.group_by("ps_partkey").agg(value.alias("value"))
    total = germany.aggregate([(value * D("0.0001000000")).alias("threshold")]).set_alias("germany_total")
    rel = (
        per_part.join(total, how="cross")
        .filter(col("value") > col("threshold"))
        .select("ps_partkey", "value")
        .sort(col("value").desc())
    )
    # per_part and total each append GROUP BY to the filtered join; the cross join one; its ORDER BY one.
    return rel, 4


@query(TPCH, 12, ordered=True)
def h12(con):
    urgent = (col("o_orderpriority") == "1-URGENT") | (col("o_orderpriority") == "2-HIGH")
    rel = (
        con.table("orders")
        .join(con.table("lineitem"), col("o_orderkey") == col("l_orderkey"))
        .filter(
            col("l_shipmode").isin("MAIL", "SHIP")
            & (col("l_commitdate") < col("l_receiptdate"))
            & (col("l_shipdate") < col("l_commitdate"))
            & (col("l_receiptdate") >= datetime.date(1994, 1, 1))
            & (col("l_receiptdate") < datetime.date(1995, 1, 1))
        )
        .group_by("l_shipmode")
        .agg(
            when(urgent).then(1).otherwise(0).sum().alias("high_line_count"),
            when(~urgent).then(1).otherwise(0).sum().alias("low_line_count"),
        )
        .sort("l_shipmode")
    )
    # The bare join opens the SELECT; WHERE and GROUP BY append; ORDER BY after the list closes it.
    return rel, 2


@query(TPCH, 13, ordered=True)
def h13(con):
    rel = (
        con.table("customer")
        .join(
            con.table("orders"),
            (col("c_custkey") == col("o_custkey")) & ~col("o_comment").like("%special%requests%"),
            how="left",
        )
        .group_by("c_custkey")
        .agg(col("o_orderkey").count().alias("c_count"))
        .group_by("c_count")
        .agg(count_all().alias("custdist"))
        .sort(col("custdist").desc(), col("c_count").desc())
    )
    # The join takes the first GROUP BY; the second closes it; ORDER BY after the second closes again.
    return rel, 3


@query(TPCH, 14, ordered=False)
def h14(con):
    promo = when(col("p_type").like("PROMO%")).then(disc_price()).otherwise(0)
    rel = (
        con.table("lineitem")
        .join(con.table("part"), col("l_partkey") == col("p_partkey"))
        .filter((col("l_shipdate") >= datetime.date(1995, 9, 1)) & (col("l_shipdate") < datetime.date(1995, 10, 1)))
        .aggregate([(D("100.00") * promo.sum() / disc_price().sum()).alias("promo_revenue")])
    )
    return rel, 1  # The bare join opens the SELECT; WHERE and the aggregate list append.


@query(TPCH, 15, ordered=True)
def h15(con):
    revenue = (
        con.table("lineitem")
        .filter((col("l_shipdate") >= datetime.date(1996, 1, 1)) & (col("l_shipdate") < datetime.date(1996, 4, 1)))
        .group_by(col("l_suppkey").alias("supplier_no"))
        .agg(disc_price().sum().alias("total_revenue"))
    )
    peak = revenue.aggregate([col("total_revenue").max().alias("max_revenue")]).set_alias("peak")
    rel = (
        con.table("supplier")
        .join(revenue, col("s_suppkey") == col("supplier_no"))
        .join(peak, col("total_revenue") == col("max_revenue"))
        .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .sort("s_suppkey")
    )
    # revenue is one SELECT read twice; peak's GROUP BY finds its slot filled; the join one; ORDER BY one.
    return rel, 4


@query(TPCH, 16, ordered=True)
def h16(con):
    complaints = con.table("supplier").filter(col("s_comment").like("%Customer%Complaints%"))
    rel = (
        con.table("partsupp")
        .join(con.table("part"), col("p_partkey") == col("ps_partkey"))
        .join(complaints, col("ps_suppkey") == col("s_suppkey"), how="anti")
        .filter(
            (col("p_brand") != "Brand#45")
            & ~col("p_type").like("MEDIUM POLISHED%")
            & col("p_size").isin(49, 14, 23, 45, 19, 3, 36, 9)
        )
        .group_by("p_brand", "p_type", "p_size")
        .agg(col("ps_suppkey").n_unique().alias("supplier_cnt"))
        .sort(col("supplier_cnt").desc(), "p_brand", "p_type", "p_size")
    )
    # complaints closes as a join input; the anti join's SELECT takes GROUP BY; ORDER BY one more.
    return rel, 3


@query(TPCH, 17, ordered=False)
def h17(con):
    part_average = (
        con.table("lineitem")
        .group_by(col("l_partkey").alias("average_partkey"))
        .agg((D("0.2") * col("l_quantity").avg()).alias("small_quantity"))
        .set_alias("part_average")
    )
    rel = (
        con.table("lineitem")
        .join(con.table("part"), col("p_partkey") == col("l_partkey"))
        .join(part_average, col("average_partkey") == col("p_partkey"))
        .filter(
            (col("p_brand") == "Brand#23")
            & (col("p_container") == "MED BOX")
            & (col("l_quantity") < col("small_quantity"))
        )
        .aggregate([(col("l_extendedprice").sum() / D("7.0")).alias("avg_yearly")])
    )
    return rel, 2  # part_average is one SELECT; the join takes WHERE and the aggregate list.


@query(TPCH, 18, ordered=True)
def h18(con):
    large = (
        con.table("lineitem")
        .group_by(col("l_orderkey").alias("large_orderkey"))
        .agg(col("l_quantity").sum().alias("large_quantity"))
        .filter(col("large_quantity") > 300)
    )
    rel = (
        con.table("customer")
        .join(con.table("orders"), col("c_custkey") == col("o_custkey"))
        .join(con.table("lineitem"), col("o_orderkey") == col("l_orderkey"))
        .join(large, col("o_orderkey") == col("large_orderkey"), how="semi")
        .group_by("c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice")
        .agg(col("l_quantity").sum())
        .sort(col("o_totalprice").desc(), "o_orderdate")
        .limit(100)
    )
    # large is two SELECTs; the inner join enters closed; the outer one; ORDER BY one.
    return rel, 5


@query(TPCH, 19, ordered=False)
def h19(con):
    def branch(brand, containers, low, size):
        return (
            (col("p_brand") == brand)
            & col("p_container").isin(*containers)
            & (col("l_quantity") >= low)
            & (col("l_quantity") <= low + 10)
            & col("p_size").between(1, size)
            & col("l_shipmode").isin("AIR", "AIR REG")
            & (col("l_shipinstruct") == "DELIVER IN PERSON")
        )

    rel = (
        con.table("lineitem")
        .join(con.table("part"), col("p_partkey") == col("l_partkey"))
        .filter(
            branch("Brand#12", ("SM CASE", "SM BOX", "SM PACK", "SM PKG"), 1, 5)
            | branch("Brand#23", ("MED BAG", "MED BOX", "MED PKG", "MED PACK"), 10, 10)
            | branch("Brand#34", ("LG CASE", "LG BOX", "LG PACK", "LG PKG"), 20, 15)
        )
        .aggregate([disc_price().sum().alias("revenue")])
    )
    return rel, 1  # The bare join opens the SELECT; WHERE and the aggregate list append.


@query(TPCH, 20, ordered=True)
def h20(con):
    shipped = (
        con.table("lineitem")
        .filter((col("l_shipdate") >= datetime.date(1994, 1, 1)) & (col("l_shipdate") < datetime.date(1995, 1, 1)))
        .group_by(col("l_partkey").alias("shipped_partkey"), col("l_suppkey").alias("shipped_suppkey"))
        .agg((D("0.5") * col("l_quantity").sum()).alias("half_quantity"))
    )
    forest = con.table("part").filter(col("p_name").like("forest%"))
    candidates = (
        con.table("partsupp")
        .join(forest, col("ps_partkey") == col("p_partkey"), how="semi")
        .join(
            shipped,
            (col("shipped_partkey") == col("ps_partkey")) & (col("shipped_suppkey") == col("ps_suppkey")),
        )
        .filter(col("ps_availqty") > col("half_quantity"))
    )
    rel = (
        con.table("supplier")
        .join(con.table("nation"), col("s_nationkey") == col("n_nationkey"))
        .join(candidates, col("s_suppkey") == col("ps_suppkey"), how="semi")
        .filter(col("n_name") == "CANADA")
        .select("s_name", "s_address")
        .sort("s_name")
    )
    # shipped, forest and the filtered candidates join are one closed SELECT each; the outer one; ORDER BY.
    return rel, 5


@query(TPCH, 21, ordered=True)
def h21(con):
    l1 = con.table("lineitem").set_alias("l1")
    l2 = con.table("lineitem").set_alias("l2")
    l3 = con.table("lineitem").set_alias("l3")
    rel = (
        l1.join(con.table("supplier"), col("s_suppkey") == col("l1", "l_suppkey"))
        .join(con.table("orders"), col("o_orderkey") == col("l1", "l_orderkey"))
        .join(con.table("nation"), col("s_nationkey") == col("n_nationkey"))
        .join(
            l2,
            (col("l2", "l_orderkey") == col("l1", "l_orderkey")) & (col("l2", "l_suppkey") != col("l1", "l_suppkey")),
            how="semi",
        )
        .join(
            l3,
            (col("l3", "l_orderkey") == col("l1", "l_orderkey"))
            & (col("l3", "l_suppkey") != col("l1", "l_suppkey"))
            & (col("l3", "l_receiptdate") > col("l3", "l_commitdate")),
            how="anti",
        )
        .filter(
            (col("o_orderstatus") == "F")
            & (col("l1", "l_receiptdate") > col("l1", "l_commitdate"))
            & (col("n_name") == "SAUDI ARABIA")
        )
        .group_by("s_name")
        .agg(count_all().alias("numwait"))
        .sort(col("numwait").desc(), "s_name")
        .limit(100)
    )
    # l1, l2 and l3 read inline; the one join SELECT takes WHERE and GROUP BY; ORDER BY closes it.
    return rel, 2


@query(TPCH, 22, ordered=True)
def h22(con):
    codes = ("13", "31", "23", "29", "30", "18", "17")
    country = fn("substring", col("c_phone"), 1, 2)
    average = (
        con.table("customer")
        .filter((col("c_acctbal") > D("0.00")) & country.isin(*codes))
        .aggregate([col("c_acctbal").avg().alias("average_balance")])
        .set_alias("average")
    )
    rel = (
        con.table("customer")
        .join(con.table("orders"), col("o_custkey") == col("c_custkey"), how="anti")
        .join(average, how="cross")
        .filter(country.isin(*codes) & (col("c_acctbal") > col("average_balance")))
        .select(country.alias("cntrycode"), "c_acctbal")
        .group_by("cntrycode")
        .agg(count_all().alias("numcust"), col("c_acctbal").sum().alias("totacctbal"))
        .sort("cntrycode")
    )
    # average is one SELECT; the joins take WHERE and a select list; GROUP BY closes; ORDER BY closes.
    return rel, 4


@query(TPCDS, 1, ordered=True)
def ds01(con):
    returns = (
        con.table("store_returns")
        .join(con.table("date_dim"), col("sr_returned_date_sk") == col("d_date_sk"))
        .filter(col("d_year") == 2000)
        .group_by(col("sr_customer_sk").alias("ctr_customer_sk"), col("sr_store_sk").alias("ctr_store_sk"))
        .agg(col("sr_return_amt").sum().alias("ctr_total_return"))
    )
    store_limit = (
        returns.group_by(col("ctr_store_sk").alias("limit_store_sk"))
        .agg((col("ctr_total_return").avg() * D("1.2")).alias("limit_return"))
        .set_alias("ctr2")
    )
    rel = (
        returns.join(con.table("store"), col("s_store_sk") == col("ctr_store_sk"))
        .join(con.table("customer"), col("ctr_customer_sk") == col("c_customer_sk"))
        .join(store_limit, col("limit_store_sk") == col("ctr_store_sk"))
        .filter((col("ctr_total_return") > col("limit_return")) & (col("s_state") == "TN"))
        .select("c_customer_id")
        .sort("c_customer_id")
        .limit(100)
    )
    # returns is one SELECT read twice; store_limit's GROUP BY finds its slot filled; the join; ORDER BY.
    return rel, 4


@query(TPCDS, 2, ordered=True)
def ds02(con):
    web = con.table("web_sales").select(
        col("ws_sold_date_sk").alias("sold_date_sk"), col("ws_ext_sales_price").alias("sales_price")
    )
    catalog = con.table("catalog_sales").select(
        col("cs_sold_date_sk").alias("sold_date_sk"), col("cs_ext_sales_price").alias("sales_price")
    )
    short = [day[:3].lower() for day in DAYS]
    weekly = (
        web.union(catalog)
        .join(con.table("date_dim"), col("d_date_sk") == col("sold_date_sk"))
        .group_by("d_week_seq")
        .agg(
            *(
                when(col("d_day_name") == day).then(col("sales_price")).end().sum().alias(f"{name}_sales")
                for day, name in zip(DAYS, short, strict=True)
            )
        )
        .set_alias("wswscs")
    )

    def year(number, suffix):
        return (
            weekly.join(con.table("date_dim"), col("date_dim", "d_week_seq") == col("wswscs", "d_week_seq"))
            .filter(col("d_year") == number)
            .select(
                col("wswscs", "d_week_seq").alias(f"d_week_seq{suffix}"),
                *(col(f"{name}_sales").alias(f"{name}_sales{suffix}") for name in short),
            )
        )

    rel = (
        year(2001, 1)
        .join(year(2002, 2), col("d_week_seq1") == col("d_week_seq2") - 53)
        .select(
            col("d_week_seq1"),
            *(
                fn("round", col(f"{name}_sales1") / col(f"{name}_sales2"), 2).alias(f"r{position}")
                for position, name in enumerate(short, start=1)
            ),
        )
        .sort(col("d_week_seq1").nulls_first())
    )
    # Two sides one each, the union two; the weekly join one; each year one; the outer join; ORDER BY.
    return rel, 9


@query(TPCDS, 6, ordered=True)
def ds06(con):
    target = (
        con.table("date_dim")
        .filter((col("d_year") == 2001) & (col("d_moy") == 1))
        .select(col("d_month_seq").alias("target_month_seq"))
        .distinct()
        .set_alias("target_month")
    )
    category_average = (
        con.table("item")
        .group_by(col("i_category").alias("average_category"))
        .agg(col("i_current_price").avg().alias("average_price"))
        .set_alias("category_average")
    )
    rel = (
        con.table("customer_address")
        .join(con.table("customer"), col("ca_address_sk") == col("c_current_addr_sk"))
        .join(con.table("store_sales"), col("c_customer_sk") == col("ss_customer_sk"))
        .join(con.table("date_dim"), col("ss_sold_date_sk") == col("d_date_sk"))
        .join(con.table("item"), col("ss_item_sk") == col("i_item_sk"))
        .join(target, col("d_month_seq") == col("target_month_seq"))
        .join(category_average, col("average_category") == col("i_category"))
        .filter(col("i_current_price") > D("1.2") * col("average_price"))
        .group_by(col("ca_state").alias("state"))
        .agg(count_all().alias("cnt"))
        .filter(col("cnt") >= 10)
        .sort(col("cnt").nulls_first(), col("state").nulls_first())
        .limit(100)
    )
    return rel, 4  # target and category_average one each; the join one, and its second filter closes it.


@query(TPCDS, 12, ordered=True)
def ds12(con):
    price = col("ws_ext_sales_price")
    rel = (
        con.table("web_sales")
        .join(con.table("item"), col("ws_item_sk") == col("i_item_sk"))
        .join(con.table("date_dim"), col("ws_sold_date_sk") == col("d_date_sk"))
        .filter(
            col("i_category").isin("Sports", "Books", "Home")
            & col("d_date").between(datetime.date(1999, 2, 22), datetime.date(1999, 3, 24))
        )
        .group_by("i_item_id", "i_item_desc", "i_category", "i_class", "i_current_price")
        .agg(
            price.sum().alias("itemrevenue"),
            (price.sum() * D("100.0000") / price.sum().sum().over(partition_by=col("i_class"))).alias("revenueratio"),
        )
        .sort("i_category", "i_class", "i_item_id", "i_item_desc", "revenueratio")
        .limit(100)
    )
    # The window sits in the aggregate list; WHERE and GROUP BY append; ORDER BY closes, LIMIT appends.
    return rel, 2


def _buyers(con, sales, date_key, customer_key):
    return (
        con.table(sales)
        .join(con.table("date_dim"), col(date_key) == col("d_date_sk"))
        .join(con.table("customer"), col(customer_key) == col("c_customer_sk"))
        .filter(col("d_month_seq").between(1200, 1200 + 11))
        .select("c_last_name", "c_first_name", "d_date")
        .distinct()
    )


@query(TPCDS, 38, ordered=False)
def ds38(con):
    hot = (
        _buyers(con, "store_sales", "ss_sold_date_sk", "ss_customer_sk")
        .intersect(_buyers(con, "catalog_sales", "cs_sold_date_sk", "cs_bill_customer_sk"))
        .intersect(_buyers(con, "web_sales", "ws_sold_date_sk", "ws_bill_customer_sk"))
    )
    rel = hot.aggregate([count_all()]).limit(100)
    return rel, 8  # Three distinct sides one each; each intersect two; the count over the set operation one.


@query(TPCDS, 43, ordered=True)
def ds43(con):
    names = [f"{day[:3].lower()}_sales" for day in DAYS]
    rel = (
        con.table("date_dim")
        .join(con.table("store_sales"), col("d_date_sk") == col("ss_sold_date_sk"))
        .join(con.table("store"), col("s_store_sk") == col("ss_store_sk"))
        .filter((col("s_gmt_offset") == -5) & (col("d_year") == 2000))
        .group_by("s_store_name", "s_store_id")
        .agg(
            *(
                when(col("d_day_name") == day).then(col("ss_sales_price")).end().sum().alias(name)
                for day, name in zip(DAYS, names, strict=True)
            )
        )
        .sort("s_store_name", "s_store_id", *names)
        .limit(100)
    )
    # The bare joins open the SELECT; WHERE and GROUP BY append; ORDER BY closes, LIMIT appends.
    return rel, 2


@query(TPCDS, 51, ordered=True)
def ds51(con):
    def cumulative(sales, item_key, date_key, price, alias):
        return (
            con.table(sales)
            .join(con.table("date_dim"), col(date_key) == col("d_date_sk"))
            .filter(col("d_month_seq").between(1200, 1200 + 11) & col(item_key).is_not_null())
            .group_by(col(item_key).alias("item_sk"), "d_date")
            .agg(
                col(price)
                .sum()
                .sum()
                .over(partition_by=col(item_key), order_by=col("d_date"), rows=(None, 0))
                .alias("cume_sales")
            )
            .set_alias(alias)
        )

    web = cumulative("web_sales", "ws_item_sk", "ws_sold_date_sk", "ws_sales_price", "web")
    store = cumulative("store_sales", "ss_item_sk", "ss_sold_date_sk", "ss_sales_price", "store")

    def either(name):
        return when(col("web", name).is_not_null()).then(col("web", name)).otherwise(col("store", name)).alias(name)

    frame = {"partition_by": col("item_sk"), "order_by": col("d_date"), "rows": (None, 0)}
    rel = (
        web.join(
            store,
            (col("web", "item_sk") == col("store", "item_sk")) & (col("web", "d_date") == col("store", "d_date")),
            how="outer",
        )
        .select(
            either("item_sk"),
            either("d_date"),
            col("web", "cume_sales").alias("web_sales"),
            col("store", "cume_sales").alias("store_sales"),
        )
        .with_columns(
            web_cumulative=col("web_sales").max().over(**frame),
            store_cumulative=col("store_sales").max().over(**frame),
        )
        .filter(col("web_cumulative") > col("store_cumulative"))
        .sort(col("item_sk").nulls_first(), col("d_date").nulls_first())
        .limit(100)
    )
    return rel, 5  # web and store one each; the join one; with_columns finds the list filled; WHERE after it too.


@query(TPCDS, 69, ordered=True)
def ds69(con):
    def buyers(sales, date_key):
        return (
            con.table(sales)
            .join(con.table("date_dim"), col(date_key) == col("d_date_sk"))
            .filter((col("d_year") == 2001) & col("d_moy").between(4, 4 + 2))
        )

    rel = (
        con.table("customer")
        .join(con.table("customer_address"), col("c_current_addr_sk") == col("ca_address_sk"))
        .join(con.table("customer_demographics"), col("cd_demo_sk") == col("c_current_cdemo_sk"))
        .join(buyers("store_sales", "ss_sold_date_sk"), col("c_customer_sk") == col("ss_customer_sk"), how="semi")
        .join(buyers("web_sales", "ws_sold_date_sk"), col("c_customer_sk") == col("ws_bill_customer_sk"), how="anti")
        .join(
            buyers("catalog_sales", "cs_sold_date_sk"), col("c_customer_sk") == col("cs_ship_customer_sk"), how="anti"
        )
        .filter(col("ca_state").isin("KY", "GA", "NM"))
        .aggregate(
            [
                col("cd_gender"),
                col("cd_marital_status"),
                col("cd_education_status"),
                count_all().alias("cnt1"),
                col("cd_purchase_estimate"),
                count_all().alias("cnt2"),
                col("cd_credit_rating"),
                count_all().alias("cnt3"),
            ],
            "cd_gender, cd_marital_status, cd_education_status, cd_purchase_estimate, cd_credit_rating",
        )
        .sort("cd_gender", "cd_marital_status", "cd_education_status", "cd_purchase_estimate", "cd_credit_rating")
        .limit(100)
    )
    # Three filtered buyer joins close as join inputs; the outer join one; its ORDER BY one more.
    return rel, 5


@query(TPCDS, 86, ordered=True)
def ds86(con):
    category, klass = sql_expr("grouping(i_category)"), sql_expr("grouping(i_class)")
    net_paid = col("ws_net_paid").sum()
    rel = (
        con.table("web_sales")
        .join(con.table("date_dim"), col("d_date_sk") == col("ws_sold_date_sk"))
        .join(con.table("item"), col("i_item_sk") == col("ws_item_sk"))
        .filter(col("d_month_seq").between(1200, 1200 + 11))
        .aggregate(
            [
                net_paid.alias("total_sum"),
                col("i_category"),
                col("i_class"),
                (category + klass).alias("lochierarchy"),
                rank()
                .over(
                    partition_by=[category + klass, when(klass == 0).then(col("i_category")).end()],
                    order_by=net_paid.desc(),
                )
                .alias("rank_within_parent"),
            ],
            "ROLLUP (i_category, i_class)",
        )
        .sort(
            col("lochierarchy").desc().nulls_first(),
            when(col("lochierarchy") == 0).then(col("i_category")).end().nulls_first(),
            col("rank_within_parent").nulls_first(),
        )
        .limit(100)
    )
    # The rollup, grouping and window sit in one aggregate list; ORDER BY closes it, LIMIT appends.
    return rel, 2


@query(TPCDS, 87, ordered=False)
def ds87(con):
    cool = (
        _buyers(con, "store_sales", "ss_sold_date_sk", "ss_customer_sk")
        .except_(_buyers(con, "catalog_sales", "cs_sold_date_sk", "cs_bill_customer_sk"))
        .except_(_buyers(con, "web_sales", "ws_sold_date_sk", "ws_bill_customer_sk"))
    )
    rel = cool.aggregate([count_all()])
    return rel, 8  # Three distinct sides one each; each except two; the count over the set operation one.


@pytest.mark.parametrize("number", sorted(TPCH), ids=lambda number: f"q{number:02d}")
def test_tpch(tpch, number):
    build, ordered = TPCH[number]
    check(tpch, number, build, ordered=ordered)


@pytest.mark.parametrize("number", sorted(TPCDS), ids=lambda number: f"q{number:02d}")
def test_tpcds(tpcds, number):
    build, ordered = TPCDS[number]
    check(tpcds, number, build, ordered=ordered)

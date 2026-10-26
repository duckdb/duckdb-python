"""The verbs the relational API gains: their rows, their schema, and the shapes that tell them apart."""

import pytest

import duckdb


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE orders AS SELECT * FROM (VALUES "
        "(1, 'ann', 'nl', 100), (2, 'bob', 'nl', 200), (3, 'ann', 'be', 300), (4, 'cid', 'be', 300)"
        ") v(id, who, land, total)"
    )
    connection.execute(
        "CREATE TABLE baskets AS SELECT * FROM (VALUES "
        "(1, [10, 20]), (2, []::INTEGER[]), (3, [30]), (4, NULL::INTEGER[])"
        ") v(id, items)"
    )
    connection.execute("CREATE TABLE quarters AS SELECT * FROM (VALUES ('nl', 1, 2), ('be', 3, NULL)) v(land, q1, q2)")
    yield connection
    connection.close()


def schema(rel):
    return [(name, str(kind)) for name, kind in zip(rel.columns, rel.types, strict=True)]


def executed(con, sql):
    cursor = con.execute(sql)
    description = [(column[0], str(column[1])) for column in cursor.description]
    return description, cursor.fetchall()


def same_as(con, rel, sql):
    description, rows = executed(con, sql)
    assert schema(rel) == description
    assert rel.fetchall() == rows
    return rows


class TestWithColumns:
    def test_a_new_column_goes_last(self, con):
        rel = con.table("orders").with_columns(half=duckdb.ColumnExpression("total") / 2)
        assert same_as(con, rel, "SELECT *, total / 2 AS half FROM orders")[0] == (1, "ann", "nl", 100, 50.0)

    def test_a_replaced_column_keeps_its_place(self, con):
        rel = con.table("orders").with_columns(total=duckdb.ColumnExpression("total") + 1)
        assert rel.columns == ["id", "who", "land", "total"]
        assert same_as(con, rel, "SELECT * REPLACE (total + 1 AS total) FROM orders")[0] == (1, "ann", "nl", 101)

    def test_every_expression_reads_the_columns_as_they_were(self, con):
        rel = con.table("orders").with_columns(
            total=duckdb.ColumnExpression("total") + 1, copy=duckdb.ColumnExpression("total")
        )
        assert rel.fetchall()[0] == (1, "ann", "nl", 101, 100)

    def test_without_a_column_it_is_refused(self, con):
        with pytest.raises(TypeError):
            con.table("orders").with_columns()


class TestDropAndRename:
    def test_drop_removes_the_named_columns(self, con):
        rel = con.table("orders").drop("who", "land")
        assert schema(rel) == [("id", "INTEGER"), ("total", "INTEGER")]
        same_as(con, rel, "SELECT * EXCLUDE (who, land) FROM orders")

    def test_rename_keeps_the_place_of_the_column(self, con):
        rel = con.table("orders").rename(who="customer")
        assert rel.columns == ["id", "customer", "land", "total"]
        same_as(con, rel, "SELECT * RENAME (who AS customer) FROM orders")

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda rel: rel.drop("nope"), id="drop"),
            pytest.param(lambda rel: rel.rename(nope="other"), id="rename"),
        ],
    )
    def test_a_column_that_is_not_there_is_refused(self, con, build):
        with pytest.raises(duckdb.Error, match="nope"):
            build(con.table("orders"))


class TestUnnestAndUnpivot:
    def test_unnest_gives_one_row_per_element(self, con):
        rel = con.table("baskets").unnest("items")
        same_as(con, rel, "SELECT * REPLACE (unnest(items) AS items) FROM baskets")
        assert rel.fetchall() == [(1, 10), (1, 20), (3, 30)]

    def test_unpivot_turns_columns_into_rows(self, con):
        rel = con.table("quarters").unpivot("q1", "q2", name="quarter", value="sales")
        assert rel.columns == ["land", "quarter", "sales"]
        assert sorted(rel.fetchall()) == [("be", "q1", 3), ("nl", "q1", 1), ("nl", "q2", 2)]

    def test_unpivot_without_a_column_is_refused(self, con):
        with pytest.raises(TypeError):
            con.table("quarters").unpivot()


class TestPivot:
    def test_explicit_values_become_columns(self, con):
        rel = con.table("orders").pivot("land", ["nl", "be"], "sum(total)")
        description, expected = executed(con, "PIVOT orders ON land IN ('nl', 'be') USING sum(total)")
        assert schema(rel) == description
        assert sorted(rel.fetchall()) == sorted(expected)
        assert sorted(rel.fetchall()) == [
            (1, "ann", 100, None),
            (2, "bob", 200, None),
            (3, "ann", None, 300),
            (4, "cid", None, 300),
        ]

    def test_a_pivot_without_values_is_refused(self, con):
        with pytest.raises(TypeError):
            con.table("orders").pivot("land", [], "sum(total)")

    def test_the_values_are_read_at_the_call_and_not_from_the_data(self, con):
        rel = con.table("orders").pivot("land", ["nl"], "sum(total)")
        con.execute("INSERT INTO orders VALUES (5, 'dan', 'de', 400)")
        assert "de" not in rel.columns
        assert rel.columns == con.sql("PIVOT orders ON land IN ('nl') USING sum(total)").columns


class TestHeadAndOffset:
    def test_head_keeps_the_first_rows(self, con):
        rel = con.table("orders").order("id").head(2)
        same_as(con, rel, "SELECT * FROM orders ORDER BY id LIMIT 2")
        assert [row[0] for row in rel.fetchall()] == [1, 2]

    def test_head_without_a_number_keeps_five(self, con):
        con.execute("INSERT INTO orders SELECT range + 10, 'x', 'nl', 1 FROM range(5)")
        assert len(con.table("orders").head().fetchall()) == 5

    def test_offset_skips_rows(self, con):
        rel = con.table("orders").order("id").offset(3)
        assert [row[0] for row in rel.fetchall()] == [4]


class TestUnionByName:
    def test_the_sides_line_up_by_name_not_by_position(self, con):
        left = con.sql("SELECT 1 AS a, 'x' AS b")
        right = con.sql("SELECT 'y' AS b, 2 AS a")
        rel = left.union_by_name(right)
        assert rel.columns == ["a", "b"]
        assert rel.fetchall() == [(1, "x"), (2, "y")]
        assert (
            rel.fetchall()
            == con.execute("SELECT 1 AS a, 'x' AS b UNION ALL BY NAME SELECT 'y' AS b, 2 AS a").fetchall()
        )

    def test_a_column_one_side_lacks_comes_back_null(self, con):
        left = con.sql("SELECT 1 AS a, 'x' AS b")
        right = con.sql("SELECT 2 AS a")
        assert left.union_by_name(right).fetchall() == [(1, "x"), (2, None)]

    def test_duplicates_are_kept_unless_asked(self, con):
        one = con.sql("SELECT 1 AS a")
        assert one.union_by_name(one).fetchall() == [(1,), (1,)]
        assert one.union_by_name(one, all=False).fetchall() == [(1,)]

    def test_union_by_position_still_reads_by_position(self, con):
        left = con.sql("SELECT 1 AS a, 'x' AS b")
        right = con.sql("SELECT 2 AS a, 'y' AS b")
        assert left.union(right).fetchall() == [(1, "x"), (2, "y")]


class TestDistinctOn:
    def test_one_row_per_key(self, con):
        rel = con.table("orders").order("total DESC, id").distinct(on="who").project("who, id").order("who")
        assert rel.fetchall() == [("ann", 3), ("bob", 2), ("cid", 4)]

    def test_several_keys(self, con):
        rel = con.table("orders").distinct(on=["who", "land"]).project("who, land").order("who, land")
        assert rel.fetchall() == [("ann", "be"), ("ann", "nl"), ("bob", "nl"), ("cid", "be")]

    def test_distinct_without_keys_still_reads_the_whole_row(self, con):
        rel = con.sql("SELECT 1 AS a, 2 AS b UNION ALL SELECT 1 AS a, 3 AS b").distinct()
        assert sorted(rel.fetchall()) == [(1, 2), (1, 3)]


class TestGroupByAgg:
    def test_the_keys_come_first(self, con):
        rel = con.table("orders").group_by("land").agg("sum(total) AS spent")
        assert rel.columns == ["land", "spent"]
        assert sorted(rel.fetchall()) == sorted(
            con.execute("SELECT land, sum(total) AS spent FROM orders GROUP BY land").fetchall()
        )

    def test_a_filter_after_it_reads_the_groups(self, con):
        rel = con.table("orders").group_by("land").agg("sum(total) AS spent").filter("spent > 400")
        assert rel.fetchall() == [("be", 600)]

    def test_a_null_key_is_a_group_of_its_own(self, con):
        con.execute("INSERT INTO orders VALUES (5, 'dan', NULL, 50)")
        rel = con.table("orders").group_by("land").agg("count(*) AS n")
        assert sorted(rel.fetchall(), key=repr) == sorted([("be", 2), ("nl", 2), (None, 1)], key=repr)

    def test_without_an_aggregate_it_is_refused(self, con):
        with pytest.raises(TypeError):
            con.table("orders").group_by("land").agg()


class TestRowsInPython:
    def test_to_dicts_keys_every_row_by_column_name(self, con):
        rel = con.table("orders").order("id").head(2).project("id, who")
        assert rel.to_dicts() == [{"id": 1, "who": "ann"}, {"id": 2, "who": "bob"}]

    def test_a_duplicate_name_keeps_both_values_in_the_row_and_one_in_the_dict(self, con):
        rel = con.sql("SELECT 1 AS a, 2 AS a")
        assert rel.fetchall() == [(1, 2)]
        assert rel.to_dicts() == [{"a": 2}]

    def test_iter_rows_gives_every_row_across_a_batch_boundary(self, con):
        rel = con.sql("SELECT CASE WHEN range % 4 = 0 THEN NULL ELSE range % 3 END AS v FROM range(10)").order(
            "v NULLS FIRST, v"
        )
        expected = rel.fetchall()
        assert len(expected) == 10
        assert list(rel.iter_rows(batch_size=3)) == expected
        assert list(rel.iter_rows(batch_size=1)) == expected
        assert sorted(rel.iter_rows(batch_size=4), key=repr) == sorted(expected, key=repr)

    def test_iter_rows_keeps_duplicate_rows_and_nulls(self, con):
        rel = con.sql("SELECT * FROM (VALUES (1, NULL), (1, NULL), (2, 'x')) t(a, b)")
        assert sorted(rel.iter_rows(batch_size=2), key=repr) == sorted(rel.fetchall(), key=repr)
        assert list(rel.iter_rows(batch_size=2)).count((1, None)) == 2

    def test_a_batch_size_of_zero_is_refused(self, con):
        with pytest.raises(ValueError, match="batch size"):
            list(con.table("orders").iter_rows(batch_size=0))


class TestTextForReading:
    def test_preview_draws_the_first_rows(self, con):
        text = con.table("orders").order("id").preview(limit=2)
        assert "who" in text
        assert "ann" in text
        assert "cid" not in text

    @pytest.mark.parametrize("limit", [1, 3])
    def test_preview_over_a_one_shot_reader_reads_it_once(self, con, limit):
        # One submission gives both the head and the footer, so a reader that can be read once still previews.
        pa = pytest.importorskip("pyarrow")
        table = pa.table({"i": list(range(25))})
        reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=5))
        text = con.from_arrow(reader).preview(limit)
        assert "25 rows" in text
        assert f"{limit} shown" in text
        assert all(f" {i} " in text for i in range(limit))
        assert f" {limit} " not in text

    def test_explain_analyze_runs_the_relation(self, con):
        rel = con.table("orders").filter("total > 100")
        plan = rel.explain(analyze=True)
        assert "orders" in plan.lower()
        assert plan != rel.explain()
        assert len(rel.fetchall()) == 3

import json

import duckdb


def parsed(sql):
    tree = json.loads(duckdb.connect().execute("SELECT json_serialize_sql(?)", [sql]).fetchall()[0][0])
    return [without_locations(statement) for statement in tree["statements"]]


def without_locations(node):
    if isinstance(node, dict):
        return {k: without_locations(v) for k, v in node.items() if not k.startswith("query_location")}
    if isinstance(node, list):
        return [without_locations(v) for v in node]
    return node


class TestRAPIFunctions:
    def test_rapi_str_print(self, duckdb_cursor, capsys):
        res = duckdb_cursor.query("select 42::INT AS a, 84::BIGINT AS b")
        drawn = str(res)
        assert [name for name in res.columns if name in drawn] == ["a", "b"]
        assert "42" in drawn
        assert "84" in drawn
        res.show()
        assert capsys.readouterr().out.strip() == drawn.strip()

    def test_rapi_relation_sql_query(self):
        res = duckdb.table_function("range", [10])
        assert parsed(res.sql_query()) == parsed("SELECT * FROM range(10)")
        assert duckdb.sql(res.sql_query()).fetchall() == res.fetchall()

import json
import re
import subprocess
import sys
import webbrowser
from html.parser import HTMLParser

import pytest

import duckdb
from duckdb.query_graph import ProfilingInfo
from duckdb.query_graph import main as query_graph_main


@pytest.fixture(scope="session")
def profiling_connection():
    con = duckdb.connect()
    con.enable_profiling()
    con.execute("SELECT 42;").fetchall()
    yield con
    con.close()


class TestProfiler:
    def test_profiler_matches_expected_format(self, profiling_connection, tmp_path_factory):
        # Test String returned
        profiling_info = ProfilingInfo(profiling_connection)
        profiling_info_json = profiling_info.to_json()
        assert isinstance(profiling_info_json, str)

        # Test expected metrics are there and profiling is json loadable. The profiling output is now grouped
        # into top-level sections (the flat per-metric keys moved underneath these, e.g. latency -> query.total_time,
        # total_bytes_read -> io.total_bytes_read, system_peak_buffer_memory -> system.peak_buffer_memory).
        profiling_dict = profiling_info.to_pydict()
        expected_keys = {
            "query",
            "system",
            "io",
            "operator",
            "optimizer",
            "physical_planner",
            "planner",
            # `parser` was dropped as a top-level profiling section in duckdb >= v1.6.0-dev10062.
        }
        assert expected_keys.issubset(profiling_dict.keys())

    def test_profiler_html_output(self, profiling_connection, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("profiler", numbered=True)
        profiling_info = ProfilingInfo(profiling_connection)
        page = profiling_info.to_html(output_file=f"{tmp_dir}/profiler_output.html")
        assert page == (tmp_dir / "profiler_output.html").read_text(encoding="utf8")
        assert "SELECT 42" in page


JOIN_SQL = "SELECT a.i, b.j\nFROM range(100) a(i) JOIN range(50) b(j) ON a.i = b.j\nWHERE a.i % 2 = 0\nORDER BY 1"

GRID_TITLES = (
    "Execution Time (s)",
    "CPU Time (s)",
    "Total GB Read",
    "Total GB Written",
    "Peak Memory (GB)",
    "Rows Scanned",
)


def engine_profile(sql: str, *settings: str) -> dict:
    """The JSON profile the running engine writes for one query, as a dict."""
    con = duckdb.connect()
    try:
        for setting in settings:
            con.execute(setting)
        con.execute("SET enable_profiling = 'no_output'")
        con.execute(sql).fetchall()
        return json.loads(con.get_profiling_information("json"))
    finally:
        con.close()


def walk(node: dict) -> list[tuple[dict, int]]:
    """The operator and its descendants with their depth, in the order the engine wrote them."""
    result = []
    stack = [(node, 0)]
    while stack:
        current, depth = stack.pop()
        result.append((current, depth))
        stack.extend((child, depth + 1) for child in reversed(current.get("children", [])))
    return result


def chain_profile(depth: int) -> dict:
    """A plan of PROJECTION operators nested depth deep, as a dict."""
    root: dict = {"type": "PROJECTION", "name": "PROJECTION", "timing": 0.001, "extra_info": {}}
    for _ in range(depth - 1):
        root = {"type": "PROJECTION", "name": "PROJECTION", "timing": 0.001, "extra_info": {}, "children": [root]}
    return {"query": {"total_time": 1.0, "cpu_time": 1.0, "sql": "deep"}, "operator": [root]}


def chain_profile_text(depth: int) -> str:
    """The JSON text of chain_profile(depth), built without recursive encoding so any depth can be written."""
    node = '{"type": "PROJECTION", "name": "PROJECTION", "timing": 0.001, "extra_info": {}'
    opening = (node + ', "children": [') * (depth - 1) + node + "}"
    return '{"query": {"total_time": 1.0, "cpu_time": 1.0, "sql": "deep"}, "operator": [' + opening + "]}" * depth


class RenderedTree(HTMLParser):
    """Rebuilds the operator tree of a page as a browser would nest it.

    Each node is a two element list [headline, children]. A browser closes an open li when the next li starts with
    no ul open inside it, and a new li belongs to the li that owns the innermost open ul; a parser that only follows
    li tags would nest B under A in <li>A<ul></ul><li>B</li></li>, where a browser shows B beside A. End tags follow
    the browser too: a </li> is ignored while a ul is still open inside the li (so <li>A<ul><li>B</li></li> leaves
    A's list open and the next li lands under A), and a </ul> is ignored when no ul is open.
    """

    def __init__(self) -> None:
        super().__init__()
        self.roots: list[list] = []
        self.stack: list[list | None] = []  # an open li as its node, an open ul as None
        self.in_headline = False

    def handle_starttag(self, tag, attrs):
        if tag == "ul":
            self.stack.append(None)
        elif tag == "li":
            if self.stack and self.stack[-1] is not None:
                self.stack.pop()
            innermost_ul = max((i for i, entry in enumerate(self.stack) if entry is None), default=-1)
            owner = next((entry for entry in reversed(self.stack[:innermost_ul]) if entry is not None), None)
            node: list = ["", []]
            (self.roots if owner is None else owner[1]).append(node)
            self.stack.append(node)
        elif tag == "b" and self.stack and self.stack[-1] is not None and not self.stack[-1][0]:
            self.in_headline = True

    def handle_data(self, data):
        if self.in_headline:
            self.stack[-1][0] += data

    def handle_endtag(self, tag):
        if tag == "b":
            self.in_headline = False
        elif tag == "li":
            if self.stack and self.stack[-1] is not None:
                self.stack.pop()
        elif tag == "ul" and None in self.stack:
            while self.stack.pop() is not None:
                pass


def rendered_tree(html: str) -> list:
    parser = RenderedTree()
    parser.feed(html)
    assert parser.stack == []
    return parser.roots


def expected_tree(node: dict) -> list:
    """The (headline, children) shape of a profile's operator, built without recursion."""
    shapes: dict[int, list] = {}
    for current, _ in reversed(walk(node)):
        shapes[id(current)] = [
            current.get("type", current.get("name", "OPERATOR")).replace("_", " "),
            [shapes[id(child)] for child in current.get("children", [])],
        ]
    return shapes[id(node)]


def operators(node: dict) -> list[dict]:
    return [op for op, _ in walk(node)]


def copy_of(profile: dict) -> dict:
    return json.loads(json.dumps(profile))


def render(profile: dict | str | bytes, tmp_path) -> str:
    """Render a profile (a dict, or the raw file content) through ProfilingInfo and return the HTML."""
    profile_file = tmp_path / "profile.json"
    if isinstance(profile, bytes):
        profile_file.write_bytes(profile)
    else:
        profile_file.write_text(profile if isinstance(profile, str) else json.dumps(profile), encoding="utf8")
    out_file = tmp_path / "profile.html"
    returned = ProfilingInfo(from_file=str(profile_file)).to_html(output_file=str(out_file))
    written = out_file.read_text(encoding="utf8")
    assert returned == written
    return written


def grid_value(html: str, title: str) -> str:
    """The value shown in the metric box with the given title."""
    pattern = rf'<div class="metric-title">{re.escape(title)}</div>\s*<div class="metric-value">([^<]*)</div>'
    matches = re.findall(pattern, html)
    assert len(matches) == 1, f"{title}: {matches}"
    return matches[0]


def headlines(html: str) -> list[str]:
    return re.findall(r"<p><b>([^<]+)</b></p>", html)


def node_shades(html: str) -> list[str]:
    return re.findall(r'<li><span class="tf-nc" style="background-color: (#[0-9a-f]{6});">', html)


def percentage_cells(html: str) -> list[str]:
    return re.findall(r"<td[^>]*>([^<]*%)</td>", html)


def sql_block(html: str) -> str:
    blocks = re.findall(r"<pre><code>(.*?)</code></pre>", html, flags=re.DOTALL)
    assert len(blocks) == 1
    return blocks[0]


@pytest.fixture(scope="module")
def join_profile():
    return engine_profile(JOIN_SQL)


class TestQueryGraphRenderer:
    def test_renders_engine_profile_with_query_metrics_and_operators(self, join_profile, tmp_path):
        html = render(join_profile, tmp_path)
        query = join_profile["query"]
        io = join_profile["io"]

        assert grid_value(html, "Execution Time (s)") == f"{query['total_time']:.4f}"
        assert grid_value(html, "CPU Time (s)") == f"{query['cpu_time']:.4f}"
        assert grid_value(html, "Total GB Read") == f"{io['total_bytes_read'] / 1024**3:.4f}"
        assert grid_value(html, "Total GB Written") == f"{io['total_bytes_written'] / 1024**3:.4f}"
        assert grid_value(html, "Peak Memory (GB)") == f"{join_profile['system']['peak_buffer_memory'] / 1024**3:.4f}"
        assert grid_value(html, "Rows Scanned") == f"{query['total_rows_scanned']:,}"
        assert JOIN_SQL in sql_block(html)
        ops = operators(join_profile["operator"][0])
        for op in ops:
            assert op["type"].replace("_", " ") in html
        hash_join = next(op for op in ops if op["type"] == "HASH_JOIN")
        assert f"<p>time: {hash_join['timing']:.4f}s</p>" in html
        assert f"<p>cardinality: {hash_join['intermediate_rows']}</p>" in html
        assert f"<p>estimate: {hash_join['extra_info']['Estimated Cardinality']}</p>" in html
        assert f"<p>result size: {hash_join['intermediate_size_bytes']} bytes</p>" in html
        assert f"Join Type: {hash_join['extra_info']['Join Type']} <br>" in html

    def test_tree_nests_children_in_engine_order(self, join_profile, tmp_path):
        html = render(join_profile, tmp_path)
        ops = operators(join_profile["operator"][0])

        assert html.count("<li>") == len(ops)
        assert headlines(html) == [op["type"].replace("_", " ") for op in ops]
        assert rendered_tree(html) == [expected_tree(join_profile["operator"][0])]
        # the two scans are leaves: each sits in its own nested list under the join
        assert html.count("<ul>") == 1 + sum(1 for op in ops if op.get("children"))
        assert html.count("</ul>") == html.count("<ul>")

    def test_name_shown_in_its_own_line_when_it_differs_from_type(self, join_profile, tmp_path):
        html = render(join_profile, tmp_path)
        ops = operators(join_profile["operator"][0])
        scan = next(op for op in ops if op["type"] == "TABLE_SCAN")
        hash_join = next(op for op in ops if op["type"] == "HASH_JOIN")

        assert scan["name"] == "RANGE"
        assert hash_join["name"] == hash_join["type"]
        assert f"<p>name: {scan['name']}</p>" in html
        assert html.count("<p>name: ") == sum(1 for op in ops if op["name"] != op["type"])

    def test_scan_counters_are_listed_in_the_details(self, join_profile, tmp_path):
        html = render(join_profile, tmp_path)
        scan = next(op for op in operators(join_profile["operator"][0]) if op["type"] == "TABLE_SCAN")

        for key in ("rows_scanned", "row_groups_scanned", "total_row_groups_to_scan"):
            assert f"{key}: {scan[key]} <br>" in html

    def test_phase_table_sums_timing_per_operator_type(self, join_profile, tmp_path):
        html = render(join_profile, tmp_path)
        entries = walk(join_profile["operator"][0])
        scans = [(op, depth) for op, depth in entries if op["type"] == "TABLE_SCAN"]
        total = sum(op["timing"] for op, _ in entries)

        assert len(scans) == 2
        assert "<td><b>Execution Time (CPU)</b></td>" in html
        assert f"<td>{round(total, 8)}</td>" in html
        assert f"<td>{round(sum(op['timing'] for op, _ in scans), 8)}</td>" in html
        assert percentage_cells(html)[0] == "100.0000%"
        assert f"Show {len(scans)} nodes" in html
        per_type: dict[str, list[dict]] = {}
        for op, _ in entries:
            per_type.setdefault(op["type"], []).append(op)
        assert html.count("↳ Depth") == sum(len(group) for group in per_type.values() if len(group) > 1)
        for _, depth in scans:
            assert f"↳ Depth {depth}</td>" in html

    def test_phase_percentages_have_fixed_decimals(self, join_profile, tmp_path):
        profile = copy_of(join_profile)
        ops = operators(profile["operator"][0])
        for op in ops:
            op["timing"] = 0.0
        next(op for op in ops if op["type"] == "HASH_JOIN")["timing"] = 0.9
        next(op for op in ops if op["type"] == "ORDER_BY")["timing"] = 0.1
        next(op for op in ops if op["type"] == "RESULT_COLLECTOR")["timing"] = 1e-9

        html = render(profile, tmp_path)

        cells = percentage_cells(html)
        assert "90.0000%" in cells
        assert "10.0000%" in cells
        assert "0.0000%" in cells
        assert all(re.fullmatch(r"\d+\.\d{4}%", cell) for cell in cells), cells

    def test_metric_grid_reads_grouped_io_and_system_metrics(self, join_profile, tmp_path):
        profile = copy_of(join_profile)
        profile["io"]["total_bytes_read"] = 3 * 1024**3
        profile["io"]["total_bytes_written"] = 1024**3 // 2
        profile["system"]["peak_buffer_memory"] = 2 * 1024**3
        profile["query"]["total_rows_scanned"] = 1234567

        html = render(profile, tmp_path)

        assert grid_value(html, "Total GB Read") == "3.0000"
        assert grid_value(html, "Total GB Written") == "0.5000"
        assert grid_value(html, "Peak Memory (GB)") == "2.0000"
        assert grid_value(html, "Rows Scanned") == "1,234,567"

    def test_node_shade_follows_the_share_of_cpu_time(self, join_profile, tmp_path):
        profile = copy_of(join_profile)
        profile["query"]["cpu_time"] = 1.0
        ops = operators(profile["operator"][0])
        for op in ops:
            op["timing"] = 0.0
        ops[0]["timing"] = 0.5

        shades = node_shades(render(profile, tmp_path))

        assert len(shades) == len(ops)
        assert shades[0] == "#dfffc3"
        assert shades[1] == "#f7fff0"

    def test_extra_info_list_values_render_each_item(self, tmp_path):
        profile = engine_profile("SELECT 1 AS x, 2 AS y")
        projection = next(op for op in operators(profile["operator"][0]) if op["type"] == "PROJECTION")
        assert projection["extra_info"]["Projections"] == ["x", "y"], "the engine writes multi line values as lists"

        html = render(profile, tmp_path)

        assert "Projections: x, y <br>" in html
        assert "['x'" not in html

    def test_user_names_in_the_details_are_shown_unchanged(self, join_profile, tmp_path):
        profile = copy_of(join_profile)
        projection = next(op for op in operators(profile["operator"][0]) if op["type"] == "PROJECTION")
        projection["extra_info"] = {
            "Projections": ["compress_integral", "__internal_z", "x,y", "__internal_compress_integral_utinyint(#0, 1)"]
        }

        html = render(profile, tmp_path)

        assert "Projections: compress_integral, __internal_z, x,y, __compress_utinyint(#0, 1) <br>" in html

    def test_profile_text_is_html_escaped(self, tmp_path):
        profile = engine_profile("SELECT '<b>&' AS x")
        assert "<b>&" in profile["query"]["sql"]
        ops = operators(profile["operator"][0])
        ops[1]["type"] = "<b>T</b>_X"
        ops[1]["name"] = "<i>n</i>"
        ops[1]["extra_info"] = {"<k>": "<v>"}

        html = render(profile, tmp_path)

        assert "'&lt;b&gt;&amp;'" in sql_block(html)
        assert "'<b>&'" not in html
        assert "<p><b>&lt;b&gt;T&lt;/b&gt; X</b></p>" in html
        assert "<p>name: &lt;i&gt;n&lt;/i&gt;</p>" in html
        assert "&lt;k&gt;: &lt;v&gt; <br>" in html
        for raw in ("<b>T</b>", "<i>n</i>", "<k>", "<v>"):
            assert raw not in html

    def test_sql_containing_template_markers_stays_text(self, tmp_path):
        profile = engine_profile("SELECT '${TREE}' AS a, '${TIMING_TABLE}' AS b, '{metric_grid}' AS c")

        html = render(profile, tmp_path)

        block = sql_block(html)
        assert "'${TREE}' AS a, '${TIMING_TABLE}' AS b, '{metric_grid}' AS c" in block
        assert "<li>" not in block
        assert "<table" not in block
        assert html.count("<li>") == len(operators(profile["operator"][0]))

    def test_deep_operator_chain_renders_without_recursion(self):
        # The dict goes straight to the renderer: Python 3.11's JSON decoder cannot read a chain this deep.
        depth = 1000
        profile = chain_profile(depth)

        html = ProfilingInfo()._render_page(profile)

        assert html.count("<li>") == depth
        assert f"Show {depth} nodes" in html
        assert f"↳ Depth {depth - 1}</td>" in html
        tree = rendered_tree(html)
        for _ in range(depth - 1):
            assert len(tree) == 1
            assert tree[0][0] == "PROJECTION"
            tree = tree[0][1]
        assert tree == [["PROJECTION", []]]

    def test_deep_operator_chain_file_renders(self, tmp_path):
        # 300 operators nest 600 JSON levels, well within what every supported Python's JSON decoder reads. The
        # tightest case is 3.11, whose limit of 1000 also counts pytest's own frames (about 944 levels remain inside
        # pytest with xdist); Windows ARM64 with 3.13 has the same 1000 for the decoder alone.
        depth = 300
        text = chain_profile_text(depth)
        assert json.loads(text) == chain_profile(depth)

        html = render(text, tmp_path)

        assert html.count("<li>") == depth
        tree = rendered_tree(html)
        for _ in range(depth - 1):
            assert len(tree) == 1
            tree = tree[0][1]
        assert tree == [["PROJECTION", []]]

    def test_profile_too_deep_for_the_json_decoder_raises_the_module_error(self, tmp_path, monkeypatch):
        # Python 3.11's decoder raises RecursionError from about 500 operators; from 3.12 on the bound is the
        # platform's C recursion limit (about 500 operators on Windows ARM64 with 3.13, about 5000 on Linux), and
        # that decoder does not follow sys.setrecursionlimit, so the failure is forced here on every version.
        def too_deep(text):
            msg = "maximum recursion depth exceeded while decoding a JSON object"
            raise RecursionError(msg)

        monkeypatch.setattr(json, "loads", too_deep)

        with pytest.raises(ValueError, match=r"profile\.json nests its operators deeper"):
            render(chain_profile_text(1000), tmp_path)

    def test_profile_a_thousand_deep_never_leaks_a_recursion_error(self, tmp_path):
        # How deep the JSON decoder reads differs by Python version and platform; the invariant is the outcome:
        # a rendered page or the module's error, never a RecursionError.
        depth = 1000
        try:
            html = render(chain_profile_text(depth), tmp_path)
        except ValueError as e:
            assert "nests its operators deeper" in str(e)  # noqa: PT017
        else:
            assert html.count("<li>") == depth

    def test_profile_without_timing_metrics_renders_tree(self, tmp_path):
        profile = engine_profile(JOIN_SQL, "SET tracked_metrics = ['operator.type', 'operator.name']")
        assert set(profile) == {"operator"}

        html = render(profile, tmp_path)

        assert headlines(html) == [op["type"].replace("_", " ") for op in operators(profile["operator"][0])]
        assert "time:" not in html
        assert "Execution Time (CPU)" not in html
        assert all(grid_value(html, title) == "N/A" for title in GRID_TITLES)

    def test_only_name_tracked_heads_the_node_with_the_name_once(self, tmp_path):
        profile = engine_profile(JOIN_SQL, "SET tracked_metrics = ['operator.name', 'operator.timing']")
        ops = operators(profile["operator"][0])
        assert "type" not in ops[0]

        html = render(profile, tmp_path)

        assert headlines(html) == [op["name"].replace("_", " ") for op in ops]
        assert "<p>name: " not in html

    def test_only_timing_tracked_heads_every_node_operator(self, tmp_path):
        profile = engine_profile(JOIN_SQL, "SET tracked_metrics = ['operator.timing']")
        ops = operators(profile["operator"][0])
        assert set(ops[0]) == {"timing", "children"}

        html = render(profile, tmp_path)

        assert headlines(html) == ["OPERATOR"] * len(ops)

    def test_profile_without_operator_metrics_renders_query_metrics(self, tmp_path):
        profile = engine_profile(JOIN_SQL, "SET tracked_metrics = ['query.*']")
        assert set(profile) == {"query"}

        html = render(profile, tmp_path)

        assert grid_value(html, "Execution Time (s)") == f"{profile['query']['total_time']:.4f}"
        assert JOIN_SQL in sql_block(html)
        assert "<li>" not in html
        assert "The profile holds no operator metrics" in html

    def test_profile_with_only_system_metrics_renders_its_boxes(self, tmp_path):
        profile = engine_profile(JOIN_SQL, "SET tracked_metrics = ['system.*']")
        assert set(profile) == {"system"}

        html = render(profile, tmp_path)

        assert grid_value(html, "Peak Memory (GB)") == f"{profile['system']['peak_buffer_memory'] / 1024**3:.4f}"
        assert grid_value(html, "Execution Time (s)") == "N/A"
        assert "<li>" not in html
        assert "Execution Time (CPU)" not in html

    def test_empty_profile_object_renders_without_metrics(self, tmp_path):
        html = render({}, tmp_path)

        assert all(grid_value(html, title) == "N/A" for title in GRID_TITLES)
        assert "<li>" not in html

    @pytest.mark.parametrize("result", ["disabled", "empty", "error"])
    def test_profile_without_query_raises(self, result, tmp_path):
        with pytest.raises(ValueError, match=result):
            render({"result": result}, tmp_path)

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("this is not json", id="not-json"),
            pytest.param("", id="empty-file"),
            pytest.param("[1, 2, 3]", id="json-list"),
            pytest.param('{"query": {}}'.encode("utf-16"), id="utf-16-file"),
        ],
    )
    def test_input_that_is_not_a_profile_raises(self, content, tmp_path):
        with pytest.raises(ValueError, match=r"profile\.json"):
            render(content, tmp_path)

    def test_missing_input_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProfilingInfo(from_file=str(tmp_path / "missing.json")).to_html(output_file=str(tmp_path / "out.html"))

    def test_profiling_info_without_connection_or_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="connection or a file"):
            ProfilingInfo().to_html(output_file=str(tmp_path / "out.html"))

    @pytest.mark.parametrize(
        ("settings", "present", "absent"),
        [
            pytest.param(("SET legacy_metrics_format = true",), ("latency", "children"), (), id="legacy-all-metrics"),
            pytest.param(
                ("SET legacy_metrics_format = true", "SET tracked_metrics = ['operator.*']"),
                ("children",),
                ("latency",),
                id="legacy-operator-metrics-only",
            ),
        ],
    )
    def test_legacy_flat_format_is_rejected_with_a_clear_message(self, settings, present, absent, tmp_path):
        profile = engine_profile(JOIN_SQL, *settings)
        for key in present:
            assert key in profile
        for key in absent:
            assert key not in profile

        with pytest.raises(ValueError, match=r"flat profiling format of DuckDB 1\.x.*legacy_metrics_format"):
            render(profile, tmp_path)

    def test_profile_written_by_duckdb_1x_is_rejected(self, tmp_path):
        # Shortened from what the DuckDB 1.5.6 wheel wrote for SELECT 42.
        profile_1x = {
            "query_name": "SELECT 42",
            "cpu_time": 0.000947039,
            "latency": 0.005966684,
            "children": [
                {
                    "operator_type": "PROJECTION",
                    "operator_name": "PROJECTION",
                    "operator_timing": 7.4e-06,
                    "operator_cardinality": 1,
                    "result_set_size": 4,
                    "cpu_time": 9.933e-06,
                    "extra_info": {"Projections": "42", "Estimated Cardinality": "1"},
                    "children": [
                        {
                            "operator_type": "DUMMY_SCAN",
                            "operator_name": "DUMMY_SCAN",
                            "operator_timing": 2.533e-06,
                            "operator_cardinality": 1,
                            "result_set_size": 4,
                            "cpu_time": 2.533e-06,
                            "extra_info": {},
                            "children": [],
                        }
                    ],
                }
            ],
        }

        with pytest.raises(ValueError, match=r"flat profiling format of DuckDB 1\.x"):
            render(profile_1x, tmp_path)


class TestQueryGraphCommandLine:
    @pytest.fixture
    def browser_calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(webbrowser, "open", lambda url, *args, **kwargs: calls.append(url))
        return calls

    @staticmethod
    def run_cli(monkeypatch, *args: str) -> None:
        monkeypatch.setattr(sys, "argv", ["query_graph", *args])
        query_graph_main()

    @staticmethod
    def write_profile(profile: dict, path) -> str:
        path.write_text(json.dumps(profile), encoding="utf8")
        return str(path)

    def test_cli_renders_profile_input_to_out(self, join_profile, tmp_path, monkeypatch, browser_calls, capsys):
        profile_file = self.write_profile(join_profile, tmp_path / "profile.json")
        out_file = tmp_path / "graph.html"

        self.run_cli(monkeypatch, "--profile_input", profile_file, "--out", str(out_file))

        html = out_file.read_text(encoding="utf8")
        assert JOIN_SQL in html
        assert "HASH JOIN" in html
        assert browser_calls == [out_file.resolve().as_uri()]
        assert str(out_file) in capsys.readouterr().out

    def test_cli_positional_profile_path_renders(self, join_profile, tmp_path, monkeypatch, browser_calls):
        profile_file = self.write_profile(join_profile, tmp_path / "profile.json")
        out_file = tmp_path / "graph.html"

        self.run_cli(monkeypatch, profile_file, "--out", str(out_file))

        assert JOIN_SQL in out_file.read_text(encoding="utf8")
        assert browser_calls == [out_file.resolve().as_uri()]

    def test_cli_default_output_path_replaces_json_suffix(self, join_profile, tmp_path, monkeypatch, browser_calls):
        # The directory name holds .json too; only the suffix of the file changes.
        directory = tmp_path / "runs.json"
        directory.mkdir()
        profile_file = self.write_profile(join_profile, directory / "profile.json")

        self.run_cli(monkeypatch, "--profile_input", profile_file)

        assert (directory / "profile.html").exists()
        assert not (tmp_path / "runs.html").exists()
        assert len(browser_calls) == 1

    def test_cli_refuses_a_profile_without_json_suffix_when_no_output_is_named(
        self, join_profile, tmp_path, monkeypatch, browser_calls, capsys
    ):
        profile_file = self.write_profile(join_profile, tmp_path / "profile.txt")

        with pytest.raises(SystemExit) as exc:
            self.run_cli(monkeypatch, profile_file, "--no-open")

        assert exc.value.code == 1
        assert ".json" in capsys.readouterr().err
        assert list(tmp_path.glob("*.html")) == []
        assert browser_calls == []

    def test_cli_positional_path_and_profile_input_must_agree(self, join_profile, tmp_path, monkeypatch, browser_calls):
        profile_file = self.write_profile(join_profile, tmp_path / "profile.json")
        other_file = self.write_profile(join_profile, tmp_path / "other.json")

        with pytest.raises(SystemExit) as exc:
            self.run_cli(monkeypatch, profile_file, "--profile_input", other_file, "--no-open")
        assert exc.value.code == 2

        self.run_cli(monkeypatch, profile_file, "--profile_input", profile_file, "--no-open")
        assert (tmp_path / "profile.html").exists()
        assert browser_calls == []

    def test_cli_no_open_writes_the_file_without_a_browser(self, join_profile, tmp_path, monkeypatch, browser_calls):
        profile_file = self.write_profile(join_profile, tmp_path / "profile.json")

        self.run_cli(monkeypatch, profile_file, "--no-open")

        assert (tmp_path / "profile.html").exists()
        assert browser_calls == []

    def test_cli_without_a_path_is_a_usage_error(self, monkeypatch, browser_calls):
        with pytest.raises(SystemExit) as exc:
            self.run_cli(monkeypatch, "--no-open")

        assert exc.value.code == 2
        assert browser_calls == []

    def test_cli_rejects_non_profile_with_exit_code_1(self, tmp_path, monkeypatch, browser_calls, capsys):
        profile_file = self.write_profile({"result": "empty"}, tmp_path / "profile.json")
        out_file = tmp_path / "graph.html"

        with pytest.raises(SystemExit) as exc:
            self.run_cli(monkeypatch, profile_file, "--out", str(out_file))

        assert exc.value.code == 1
        assert "empty" in capsys.readouterr().err
        assert not out_file.exists()
        assert browser_calls == []

    @pytest.mark.parametrize("out_name", ["graph.txt", "graph.html.txt"])
    def test_cli_rejects_output_without_html_suffix(self, out_name, join_profile, tmp_path, monkeypatch, browser_calls):
        profile_file = self.write_profile(join_profile, tmp_path / "profile.json")

        with pytest.raises(SystemExit) as exc:
            self.run_cli(monkeypatch, profile_file, "--out", str(tmp_path / out_name))

        assert exc.value.code == 1
        assert not (tmp_path / out_name).exists()
        assert browser_calls == []

    def test_cli_refuses_an_output_path_equal_to_the_input(
        self, join_profile, tmp_path, monkeypatch, browser_calls, capsys
    ):
        profile_file = self.write_profile(join_profile, tmp_path / "saved.html")

        with pytest.raises(SystemExit) as exc:
            self.run_cli(monkeypatch, profile_file, "--out", profile_file)

        assert exc.value.code == 1
        assert "input" in capsys.readouterr().err
        assert json.loads((tmp_path / "saved.html").read_text(encoding="utf8")) == join_profile
        assert browser_calls == []

    def test_cli_names_the_file_it_could_not_read(self, tmp_path, monkeypatch, browser_calls, capsys):
        directory = tmp_path / "dir.json"
        directory.mkdir()

        with pytest.raises(SystemExit) as exc:
            self.run_cli(monkeypatch, str(directory), "--no-open")

        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "read the profile" in err
        assert str(directory) in err

    def test_cli_names_the_file_it_could_not_write(self, join_profile, tmp_path, monkeypatch, browser_calls, capsys):
        # The output path is given relative and not in normal form, as a user types it.
        profile_file = self.write_profile(join_profile, tmp_path / "profile.json")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc:
            self.run_cli(monkeypatch, profile_file, "--out", "./missing/graph.html")

        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "write the output file ./missing/graph.html" in err
        assert "read the profile" not in err
        assert browser_calls == []

    def test_cli_module_run_does_not_warn_about_double_import(self, join_profile, tmp_path):
        profile_file = self.write_profile(join_profile, tmp_path / "profile.json")

        result = subprocess.run(
            [sys.executable, "-m", "duckdb.query_graph", profile_file, "--no-open"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "RuntimeWarning" not in result.stderr
        assert (tmp_path / "profile.html").exists()
        assert str(tmp_path / "profile.html") in result.stdout

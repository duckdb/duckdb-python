import json
import re
import subprocess
import sys
import webbrowser

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
        # Test HTML execution works, nothing to assert!
        profiling_info.to_html(output_file=f"{tmp_dir}/profiler_output.html")


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


def walk(node: dict, depth: int = 0) -> list[tuple[dict, int]]:
    """The operator and its descendants with their depth, in the order the engine wrote them."""
    return [(node, depth), *[entry for child in node.get("children", []) for entry in walk(child, depth + 1)]]


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

    def test_deep_operator_chain_renders(self, tmp_path):
        depth = 1000
        root: dict = {"type": "PROJECTION", "name": "PROJECTION", "timing": 0.001, "extra_info": {}}
        for _ in range(depth - 1):
            root = {"type": "PROJECTION", "name": "PROJECTION", "timing": 0.001, "extra_info": {}, "children": [root]}
        profile = {"query": {"total_time": 1.0, "cpu_time": 1.0, "sql": "deep"}, "operator": [root]}

        html = render(profile, tmp_path)

        assert html.count("<li>") == depth
        assert html.count("</li>") == depth
        assert f"Show {depth} nodes" in html
        assert f"↳ Depth {depth - 1}</td>" in html

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

    def test_legacy_flat_format_is_rejected_with_a_clear_message(self, tmp_path):
        profile = engine_profile(JOIN_SQL, "SET legacy_metrics_format = true")
        assert "latency" in profile
        assert "children" in profile

        with pytest.raises(ValueError, match="legacy_metrics_format"):
            render(profile, tmp_path)


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
        profile_file = self.write_profile(join_profile, tmp_path / "profile.json")

        self.run_cli(monkeypatch, "--profile_input", profile_file)

        assert (tmp_path / "profile.html").exists()
        assert len(browser_calls) == 1

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

    def test_cli_rejects_output_without_html_suffix(self, join_profile, tmp_path, monkeypatch, browser_calls):
        profile_file = self.write_profile(join_profile, tmp_path / "profile.json")

        with pytest.raises(SystemExit) as exc:
            self.run_cli(monkeypatch, profile_file, "--out", str(tmp_path / "graph.txt"))

        assert exc.value.code == 1
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
        profile_file = self.write_profile(join_profile, tmp_path / "profile.json")
        out_file = tmp_path / "missing" / "graph.html"

        with pytest.raises(SystemExit) as exc:
            self.run_cli(monkeypatch, profile_file, "--out", str(out_file))

        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "write the output file" in err
        assert str(out_file) in err
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

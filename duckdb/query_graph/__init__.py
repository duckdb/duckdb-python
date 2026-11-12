"""Render a DuckDB JSON profile as an HTML page with the query metrics and the operator tree.

The profile is the grouped document a DuckDB 2.x engine writes with ``SET enable_profiling = 'json'`` and
``SET profile_output = 'profile.json'``, or returns from ``connection.get_profiling_information("json")``.
"""

import argparse
import html
import json
import re
import sys
import webbrowser
from pathlib import Path
from typing import IO, Any, NoReturn

from duckdb import DuckDBPyConnection

__all__ = ["ProfilingInfo", "main"]

Profile = dict[str, Any]
Node = dict[str, Any]

qgraph_css = """
:root {
  --text-primary-color: #0d0d0d;
  --text-secondary-color: #444;
  --doc-codebox-border-color: #e6e6e6;
  --doc-codebox-background-color: #f7f7f7;
  --doc-scrollbar-bg: #e6e6e6;
  --doc-scrollbar-slider: #ccc;
  --duckdb-accent: #009982;
  --duckdb-accent-light: #00b89a;
  --card-bg: #fff;
  --border-radius: 8px;
  --shadow: 0 4px 14px rgba(0,0,0,0.05);
}

html, body {
  margin: 0;
  padding: 0;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--text-primary-color);
  background: #fafafa;
  line-height: 1.55;
}

.container {
  max-width: 1000px;
  margin: 40px auto;
  padding: 0 20px;
}

header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 5px;
}

header img {
  width: 100px;
  height: 100px;
}

header h1 {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
  color: var(--text-primary-color);
}

/* === Table Styling (DuckDB documentation style, flat header) === */
table {
  border-collapse: collapse;
  width: 100%;
  margin-bottom: 20px;
  text-align: left;
  font-variant-numeric: tabular-nums;
  border: 1px solid var(--doc-codebox-border-color);
  border-radius: var(--border-radius);
  overflow: hidden;
  box-shadow: var(--shadow);
  background: var(--card-bg);
}

thead {
  background-color: var(--duckdb-accent);
  color: white;
}

th, td {
  padding: 10px 12px;
  font-size: 14px;
  vertical-align: top;
}

th {
  font-weight: 700;
}

tbody tr {
  border-bottom: 1px solid var(--doc-codebox-border-color);
}

tbody tr:last-child td {
  border-bottom: none;
}

tbody tr:hover {
  background: var(--doc-codebox-border-color);
}

tbody tr.phase-details-row {
  border-bottom: none;
}

tbody tr.phase-details-row:hover {
  background: transparent;
}

tbody tr.phase-details-row details summary {
  font-size: 12px;
  padding: 4px 0;
}

tbody tr.phase-details-row details[open] summary {
  margin-bottom: 4px;
}

/* === Chart/Card Section === */
.chart {
  padding: 20px;
  border: 1px solid var(--doc-codebox-border-color);
  border-radius: var(--border-radius);
  background: var(--card-bg);
  box-shadow: var(--shadow);
  overflow: visible;
}

/* === Tree Layout Styling === */
.tf-tree {
  overflow-x: visible;
  overflow-y: visible;
  padding-top: 20px;
}

.tf-nc {
  background: var(--card-bg);
  border: 1px solid var(--doc-codebox-border-color);
  border-radius: var(--border-radius);
  padding: 6px;
  display: inline-block;
}

.node-body {
  font-size: 13px;
  text-align: left;
  padding: 10px;
  white-space: nowrap;
}

.node-body p {
  margin: 2px 0;
}

.node-details {
  white-space: nowrap;
  overflow: visible;
  display: inline-block;
}

/* === Metric Boxes === */
.chart .metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}

.chart .metric-box {
  background: var(--card-bg);
  border: 1px solid var(--doc-codebox-border-color);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow);
  padding: 12px 16px;
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.chart .metric-box:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
}

.chart .metric-title {
  font-size: 13px;
  color: var(--text-secondary-color);
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.chart .metric-value {
  font-size: 18px;
  font-weight: 600;
  color: var(--duckdb-accent);
}


/* === SQL Query Block === */
.chart.sql-block {
  background: var(--doc-codebox-background-color);
  border: 1px solid var(--doc-codebox-border-color);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow);
  padding: 16px;
  overflow-x: auto;
  margin-top: 20px;
}

.chart.sql-block pre {
  margin: 0;
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  font-size: 13.5px;
  line-height: 1.5;
  color: var(--text-primary-color);
  white-space: pre;
}

.chart.sql-block code {
  color: var(--duckdb-accent);
  font-weight: 500;
}


/* === Links, Typography, and Consistency === */
a {
  color: var(--duckdb-accent);
  text-decoration: underline;
  transition: color 0.3s;
}

a:hover {
  color: black;
}

strong {
  font-weight: 600;
}

/* === Dark Mode Support === */
@media (prefers-color-scheme: dark) {
  :root {
    --text-primary-color: #e6e6e6;
    --text-secondary-color: #b3b3b3;
    --doc-codebox-border-color: #2a2a2a;
    --doc-codebox-background-color: #1e1e1e;
    --card-bg: #111;
  }
  body {
    background: #0b0b0b;
  }
  thead {
    background-color: var(--duckdb-accent);
  }
  tbody tr:hover {
    background: #222;
  }
  
  /* Fix tree node text visibility in dark mode */
  .tf-nc .node-body,
  .tf-nc .node-body p,
  .tf-nc .node-details {
    color: #1a1a1a !important;
  }
  
  /* Fix metric title visibility in dark mode */
  .chart .metric-title {
    color: #b3b3b3;
  }
}
"""  # noqa: W293


def open_utf8(fpath: str, flags: str) -> IO[Any]:
    return Path(fpath).open(mode=flags, encoding="utf8")


def _escape(text: object) -> str:
    return html.escape(str(text), quote=False)


def _seconds(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _gigabytes(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value) / (1024**3):.4f}"


def _count(value: int | None) -> str:
    return "N/A" if value is None else f"{int(value):,}"


def _percent(time: float, total: float) -> str:
    return f"{time / total * 100 if total else 0.0:.4f}%"


def _group(profile: Profile, name: str) -> Profile:
    group = profile.get(name)
    return group if isinstance(group, dict) else {}


def _children(node: Node) -> list[Node]:
    children = node.get("children")
    return [child for child in children if isinstance(child, dict)] if isinstance(children, list) else []


def _operators(roots: list[Node]) -> list[tuple[Node, int]]:
    """Every operator with its depth, in the order the engine wrote them, without recursion."""
    result = []
    stack = [(root, 0) for root in reversed(roots)]
    while stack:
        node, depth = stack.pop()
        result.append((node, depth))
        stack.extend((child, depth + 1) for child in reversed(_children(node)))
    return result


def _headline(node: Node) -> str:
    label = node.get("type", node.get("name"))
    return "OPERATOR" if label is None else str(label).replace("_", " ")


# The compressed materialization optimizer names its functions __internal_compress_integral_<type>; a details line
# full of them reads better short, and no other text is touched.
_ENGINE_COMPRESS_FUNCTION = re.compile(r"\b__internal_((?:de)?compress)_integral_(\w+)\(")


def _detail_value(value: object) -> str:
    text = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
    return _ENGINE_COMPRESS_FUNCTION.sub(r"__\1_\2(", text)


SCAN_KEYS = ("rows_scanned", "row_groups_scanned", "total_row_groups_to_scan", "bytes_scanned")
# Every flat profile, whatever is tracked, has its operator tree under a top level "children" list, which alone
# decides; the other keys are the flat names of the first operator and of the query time, kept as a belt.
FLAT_FORMAT_KEYS = ("children", "latency", "operator_type", "operator_timing")


class ProfilingInfo:
    """The profile of the last query on a connection, or a profile read from a file."""

    def __init__(self, conn: DuckDBPyConnection | None = None, from_file: str | None = None) -> None:  # noqa: D107
        self.conn = conn
        self.from_file = from_file

    def to_json(self) -> str:  # noqa: D102
        if self.from_file is not None:
            with open_utf8(self.from_file, "r") as f:
                text: str = f.read()
                return text
        if self.conn is None:
            msg = "ProfilingInfo needs a connection or a file to read the profile from"
            raise ValueError(msg)
        profile: str = self.conn.get_profiling_information(format="json")
        return profile

    def to_pydict(self) -> Profile:  # noqa: D102
        profile: Profile = json.loads(self.to_json())
        return profile

    def to_html(self, output_file: str = "profile.html") -> str:
        """Write the profile as an HTML page to output_file and return the page text."""
        page = self._render_html()
        with open_utf8(output_file, "w") as f:
            f.write(page)
        return page

    def _render_html(self) -> str:
        try:
            text = self.to_json()
        except UnicodeDecodeError as e:
            msg = f"{self._source_name()} is not a UTF-8 text file: {e}"
            raise ValueError(msg) from e
        return self._render_page(self._parse_profile(text))

    def _source_name(self) -> str:
        return self.from_file if self.from_file is not None else "the connection's profiling output"

    def _parse_profile(self, text: str) -> Profile:
        source = self._source_name()
        try:
            profile = json.loads(text)
        except json.JSONDecodeError as e:
            msg = f"{source} is not a JSON profile: {e}"
            raise ValueError(msg) from e
        except RecursionError as e:
            # An operator nests two JSON levels. Python 3.11's decoder counts them against the recursion limit of
            # 1000, so about 500 operators; 3.12 and 3.13 against the platform's C recursion limit, about 5000
            # operators on Linux and macOS, about 1500 on Windows (also ARM64 with 3.12) and about 500 on Windows
            # ARM64 with 3.13; 3.14 is bounded by the thread's stack, 20000 and more in the main thread on Linux.
            msg = f"{source} nests its operators deeper than this Python's JSON decoder can read"
            raise ValueError(msg) from e
        if not isinstance(profile, dict):
            msg = f"{source} is not a profile: expected a JSON object, got {type(profile).__name__}"
            raise ValueError(msg)  # noqa: TRY004
        if any(key in profile for key in FLAT_FORMAT_KEYS):
            msg = (
                f"{source} is in the flat profiling format of DuckDB 1.x, which a 2.x engine writes only with "
                "legacy_metrics_format = true; profile the query with a 2.x engine's default format to render it"
            )
            raise ValueError(msg)
        if set(profile) == {"result"}:
            msg = f"{source} holds no query to render (result: {profile['result']})"
            raise ValueError(msg)
        return profile

    @staticmethod
    def _metric_grid_html(profile: Profile) -> str:
        query = _group(profile, "query")
        io = _group(profile, "io")
        system = _group(profile, "system")
        metrics = {
            "Execution Time (s)": _seconds(query.get("total_time")),
            "CPU Time (s)": _seconds(query.get("cpu_time")),
            "Total GB Read": _gigabytes(io.get("total_bytes_read")),
            "Total GB Written": _gigabytes(io.get("total_bytes_written")),
            "Peak Memory (GB)": _gigabytes(system.get("peak_buffer_memory")),
            "Rows Scanned": _count(query.get("total_rows_scanned")),
        }
        boxes = "".join(
            f"""
            <div class="metric-box">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
            """
            for title, value in metrics.items()
        )
        return f'<div class="metrics-grid">{boxes}</div>'

    @staticmethod
    def _sql_html(profile: Profile) -> str:
        sql = _group(profile, "query").get("sql")
        return f"""
        <details><summary><b>SQL Query</b></summary>
        <div class="chart sql-block">
            <pre><code>{_escape(sql) if sql is not None else "N/A"}</code></pre>
        </div>
        </details><br>
        """

    @staticmethod
    def _timing_table_html(nodes: list[tuple[Node, int]]) -> str:
        timed = [(node, depth) for node, depth in nodes if "timing" in node]
        if not timed:
            return ""
        total = sum(float(node["timing"]) for node, _ in timed)

        per_type: dict[str, list[tuple[float, int]]] = {}
        for node, depth in timed:
            per_type.setdefault(_headline(node), []).append((float(node["timing"]), depth))
        phases = sorted(per_type, key=lambda phase: sum(time for time, _ in per_type[phase]), reverse=True)

        rows = f"""
      <tr>
          <td><b>Execution Time (CPU)</b></td>
                <td>{round(total, 8)}</td>
                <td>{_percent(total, total)}</td>
        </tr>
    """
        for phase in phases:
            timings = per_type[phase]
            phase_time = sum(time for time, _ in timings)
            rows += f"""
      <tr>
          <td>{_escape(phase)}</td>
                <td>{round(phase_time, 8)}</td>
                <td>{_percent(phase_time, total)}</td>
        </tr>
    """
            if len(timings) > 1:
                node_rows = "".join(
                    f"""
                            <tr style="background: var(--doc-codebox-background-color);">
                                <td style="padding: 4px 12px; border: none;">{"&nbsp;" * (depth * 4)}↳ Depth {depth}</td>
                                <td style="padding: 4px 12px; border: none;">{round(time, 8)}</td>
                                <td style="padding: 4px 12px; border: none;">{_percent(time, total)}</td>
                            </tr>
    """  # noqa: E501
                    for time, depth in sorted(timings, key=lambda entry: entry[0], reverse=True)
                )
                rows += f"""
        <tr class="phase-details-row">
            <td colspan="3">
                <details>
                    <summary style="cursor: pointer; padding: 4px 0; color: var(--text-secondary-color);">
                        Show {len(timings)} nodes
                    </summary>
                    <table style="margin: 8px 0; width: 100%; border: none; box-shadow: none;">
                        <tbody>{node_rows}
                        </tbody>
                    </table>
                </details>
            </td>
        </tr>
    """
        return f"""
      <table>
        <thead>
          <tr>
            <th>Phase</th>
            <th>Time (s)</th>
            <th>Percentage</th>
          </tr>
        </thead><tbody>{rows}</tbody></table>"""

    @staticmethod
    def _shade_hex(fraction: float) -> str:
        """A shade from very light (#f7fff0) to a darker green-yellow for a fraction between 0 and 1."""
        fraction = max(0.0, min(1.0, fraction))
        light_color = (247, 255, 240)
        dark_color = (200, 255, 150)
        r, g, b = (int(light + (dark - light) * fraction) for light, dark in zip(light_color, dark_color, strict=True))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _node_html(self, node: Node, shade_total: float | None) -> str:
        extra_info = node.get("extra_info")
        extra_info = extra_info if isinstance(extra_info, dict) else {}
        timing = node.get("timing")
        fraction = float(timing) / shade_total if timing is not None and shade_total else 0.0

        body = f'<span class="tf-nc" style="background-color: {self._shade_hex(fraction)};">'
        body += '<div class="node-body">'
        body += f"<p><b>{_escape(_headline(node))}</b></p>"
        if "type" in node and "name" in node and node["name"] != node["type"]:
            body += f"<p>name: {_escape(node['name'])}</p>"
        if timing is not None:
            body += f"<p>time: {float(timing):.4f}s</p>"
        if "intermediate_rows" in node:
            body += f"<p>cardinality: {_escape(node['intermediate_rows'])}</p>"
        if "Estimated Cardinality" in extra_info:
            body += f"<p>estimate: {_escape(extra_info['Estimated Cardinality'])}</p>"
        if "intermediate_size_bytes" in node:
            body += f"<p>result size: {_escape(node['intermediate_size_bytes'])} bytes</p>"

        details = [(key, node[key]) for key in SCAN_KEYS if key in node]
        details += [(key, value) for key, value in extra_info.items() if key != "Estimated Cardinality"]
        lines = "".join(f"{_escape(key)}: {_escape(_detail_value(value))} <br>" for key, value in details)
        body += "<details>"
        body += "<summary>Extra info</summary>"
        body += '<div class="node-details">'
        body += f"<p>{lines}</p>"
        body += "</div>"
        body += "</details>"
        body += "</div>"
        body += "</span>"
        return body

    def _tree_html(self, profile: Profile, roots: list[Node], nodes: list[tuple[Node, int]]) -> str:
        if not roots:
            return (
                '<div class="chart"><p>The profile holds no operator metrics; '
                "set tracked_metrics to include operator.* to see the operator tree.</p></div>"
            )
        cpu_time = _group(profile, "query").get("cpu_time")
        if cpu_time:
            shade_total = float(cpu_time)
        else:
            shade_total = sum(float(node["timing"]) for node, _ in nodes if "timing" in node)

        # An explicit stack keeps a deep plan from hitting the recursion limit; a string entry closes a parent.
        parts = ['<div class="tf-tree tf-gap-sm"> \n <ul>']
        stack: list[Node | str] = list(reversed(roots))
        while stack:
            item = stack.pop()
            if isinstance(item, str):
                parts.append(item)
                continue
            parts.append("<li>" + self._node_html(item, shade_total or None))
            children = _children(item)
            if children:
                parts.append("<ul>")
                stack.append("</ul></li>")
                stack.extend(reversed(children))
            else:
                parts.append("</li>")
        parts.append("</ul> </div>")
        return "".join(parts)

    def _render_page(self, profile: Profile) -> str:
        operator = profile.get("operator")
        roots = [root for root in operator if isinstance(root, dict)] if isinstance(operator, list) else []
        nodes = _operators(roots)
        metric_grid = self._metric_grid_html(profile)
        sql_block = self._sql_html(profile)
        timing_table = self._timing_table_html(nodes)
        tree = self._tree_html(profile, roots, nodes)
        return f"""<!DOCTYPE html>
    <html>
      <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width">
      <title>Query Profile Graph for Query</title>
      <link rel="stylesheet" href="https://unpkg.com/treeflex/dist/css/treeflex.css">
      <style>
        {qgraph_css}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <img src="https://raw.githubusercontent.com/duckdb/duckdb/refs/heads/main/logo/DuckDB_Logo-horizontal.svg" alt="DuckDB Logo">
                <h1>Query Profile Graph</h1>
            </header>
        <div class="chart" id="query-overview">
            {metric_grid}
        </div>
      <div class="chart" id="query-profile">
            {sql_block}
            {timing_table}
      </div>
      {tree}
        </div>
    </body>
    </html>
    """  # noqa: E501


def _fail(message: str) -> NoReturn:
    print(f"query_graph: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    """Render a JSON profile file to HTML from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m duckdb.query_graph",
        description="Given a JSON profile written by DuckDB, generate an HTML file showing the query graph and the "
        "timings of its operators.",
    )
    parser.add_argument("profile", nargs="?", help="the JSON profile written with SET enable_profiling = 'json'")
    parser.add_argument("--profile_input", help="the same path as the positional argument, kept for older scripts")
    parser.add_argument("--out", help="the HTML file to write; defaults to the profile path with .html")
    parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open the HTML file in the browser once written (the default)",
    )
    args = parser.parse_args()

    if args.profile is not None and args.profile_input is not None and args.profile != args.profile_input:
        parser.error("the profile path was given twice with different values")
    profile_path: str | None = args.profile if args.profile is not None else args.profile_input
    if profile_path is None:
        parser.error("the path of a JSON profile is required")

    if args.out is not None:
        if not args.out.endswith(".html"):
            _fail("the output file must end in .html")
        output_path: str = args.out
    elif profile_path.endswith(".json"):
        output_path = profile_path[: -len(".json")] + ".html"
    else:
        _fail("the profile must be a .json file when --out is not given")

    if Path(output_path).resolve() == Path(profile_path).resolve():
        _fail(f"the output path {output_path} is the input profile; choose another --out")

    try:
        page = ProfilingInfo(from_file=profile_path)._render_html()
    except ValueError as e:
        _fail(str(e))
    except OSError as e:
        _fail(f"could not read the profile {profile_path}: {e.strerror or e}")
    try:
        with open_utf8(output_path, "w") as f:
            f.write(page)
    except OSError as e:
        _fail(f"could not write the output file {output_path}: {e.strerror or e}")

    print(f"query_graph: wrote {output_path}")
    if args.open:
        webbrowser.open(Path(output_path).resolve().as_uri(), new=2)

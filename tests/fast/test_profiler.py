import pytest

import duckdb
from duckdb.query_graph import ProfilingInfo


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

        # Test expected metrics are there and profiling is json loadable
        profiling_dict = profiling_info.to_pydict()
        expected_keys = {
            "query_name",
            "total_bytes_written",
            "total_bytes_read",
            "system_peak_temp_dir_size",
            "system_peak_buffer_memory",
            "rows_returned",
            "result_set_size",
            "latency",
            "cumulative_rows_scanned",
            "cumulative_cardinality",
            "cpu_time",
            "extra_info",
            "blocked_thread_time",
            "children",
        }
        assert expected_keys.issubset(profiling_dict.keys())

    def test_profiler_html_output(self, profiling_connection, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("profiler", numbered=True)
        profiling_info = ProfilingInfo(profiling_connection)
        # Test HTML execution works, nothing to assert!
        profiling_info.to_html(output_file=f"{tmp_dir}/profiler_output.html")

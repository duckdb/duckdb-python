Tests in this directory are intended to be run with [pytest-run-parallel](https://github.com/Quansight-Labs/pytest-run-parallel) to exercise thread safety. 

Example usage: `pytest --parallel-threads=10 --iterations=5 --verbose tests/fast/threading -n 4 --durations=5`

#### Thread Safety and DuckDB

Not all duckdb operations are thread safe - cursors are not thread safe, so some care must be considered to avoid running tests that concurrently hit the same tests.

Tests can be marked as single threaded with:
- `pytest.mark.thread_unsafe` or the equivalent `pytest.mark.parallel_threads(1)`

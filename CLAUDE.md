# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project overview

This is the **production** duckdb-python client — the `duckdb` package on PyPI. It provides Python bindings for [DuckDB](https://duckdb.org), an in-process OLAP database engine, via pybind11 and a custom scikit-build-core build backend.

- **Repository**: https://github.com/duckdb/duckdb-python
- **Package name**: `duckdb`
- **Bindings**: pybind11
- **Build backend**: `duckdb_packaging.build_backend` (custom wrapper around scikit-build-core)
- **Supported Python**: 3.10, 3.11, 3.12, 3.13, 3.14
- **Free-threaded Python**: not supported in this client. A separate prototype client based on DuckDB's C API targets free-threading, Stable ABI, and multi-interpreter support.

## IMPORTANT: build before running anything

**You MUST complete a full build before running tests, scripts, or `uv run` in a fresh worktree or after a clean slate.** `uv run pytest` triggers scikit-build-core's editable rebuild on import, which compiles 2000+ C++ files from scratch — this takes 5–10 minutes and will exceed the Bash tool's default timeout. Do not attempt to run tests, scripts, or `uv run python` until the two-step build below has completed successfully.

```bash
# Step 1: install build deps (~5 seconds)
uv sync --only-group build --no-install-project -p 3.13

# Step 2: build the extension (3–10 min cold, ~30s with sccache, use timeout: 600000)
uv sync --no-build-isolation -v --reinstall -p 3.13
```

After step 2 completes, `uv run pytest`, `uv run python`, and `.venv/bin/python` all work immediately. Subsequent C++ changes trigger fast incremental rebuilds (seconds with sccache), not full cold builds.

## Build system

### Editable install (standard development workflow)

The build uses `--no-build-isolation` for two reasons: (1) it makes incremental rebuilds fast by reusing the target venv, and (2) **it keeps pybind11 headers in the venv so CLion / other C++ IDEs can resolve them** — with build isolation, pybind11 is installed into a temp env that gets destroyed after the build, leaving `compile_commands.json` with stale include paths and CLion unable to navigate the bindings code.

`--no-build-isolation` requires that build dependencies are already installed. **This is a two-step process on a fresh venv:**

**Step 1 — install build deps** (fast, ~5 seconds):

```bash
uv sync --only-group build --no-install-project -p 3.13
```

**Step 2 — build the extension** (3–10 min cold, ~30 seconds with warm sccache):

```bash
uv sync --no-build-isolation -v --reinstall -p 3.13
```

This produces a debug editable install in `build/debug/` with `editable.mode = "redirect"`. Python code changes are picked up immediately; C++ changes require a rebuild.

### sccache (strongly recommended)

sccache caches compiled object files across builds and worktrees. Without it, cold builds take 5–10 minutes (2000+ C++ compilation units). With a warm cache, they take ~30 seconds.

```bash
# Export before any uv sync
export CMAKE_C_COMPILER_LAUNCHER="$(command -v sccache)"
export CMAKE_CXX_COMPILER_LAUNCHER="$(command -v sccache)"
```

Install with `brew install sccache` (macOS). Check cache state with `sccache -s`.

### Non-editable (release-style) install

```bash
uv sync --no-build-isolation --no-editable -v --reinstall -p 3.13
```

Produces a release build (no debug symbols, optimized).

### Wheel build

```bash
uv build --wheel
```

Produces a wheel in `dist/`. Uses cibuildwheel for CI — see `pyproject.toml` `[tool.cibuildwheel]` for the CI wheel matrix (macOS arm64/x86_64, Linux x86_64/aarch64, Windows AMD64/ARM64).

### sdist build

```bash
uv build --sdist
```

The sdist includes the DuckDB submodule source via the `[tool.scikit-build.sdist]` include list in `pyproject.toml`.

### Incremental rebuild shortcuts

After editing C++ source:

```bash
# Fastest: touch the __init__.py to trigger scikit-build-core's rebuild detection
touch duckdb/__init__.py && uv sync --no-build-isolation -v --reinstall
```

Rebuild only the duckdb package (useful when dependency lock hasn't changed):

```bash
uv sync --reinstall-package duckdb
```

### Clean slate

```bash
rm -rf build .venv uv.lock && uv cache clean --force
```

### Python version selection

Pass `-p <version>` to any `uv sync` command:

```bash
uv sync --no-build-isolation -v --reinstall -p 3.11
uv sync --no-build-isolation -v --reinstall -p 3.14
```

Supported: `3.10`, `3.11`, `3.12`, `3.13`, `3.14`. Do **not** use free-threaded variants (`3.13t`, `3.14t`) — the production client does not support them.

### Build configuration reference

Key `pyproject.toml` settings:

- `BUILD_EXTENSIONS = "core_functions;json;parquet;icu;jemalloc"` — extensions built into the wheel.
- Editable overrides: `build-dir = "build/debug/"`, `editable.rebuild = true`, `editable.mode = "redirect"`, `cmake.build-type = "Debug"`, `DISABLE_UNITY = "1"` (unity disabled for better debugging).
- Coverage overrides: `build-dir = "build/coverage/"`, `RelWithDebInfo`, `--coverage` flags. Activate with `COVERAGE=true uv sync ...`.

## Testing

### Test layout

```
tests/
├── fast/              # quick per-subsystem tests (seconds each)
│   ├── arrow/         # pyarrow integration
│   ├── pandas/        # pandas integration
│   ├── polars/        # polars integration
│   ├── spark/         # pyspark compatibility
│   ├── adbc/          # ADBC driver
│   ├── api/           # relational API
│   ├── dbapi/         # PEP 249 DB-API 2.0
│   ├── capi/          # C API bindings
│   ├── udf/           # Python UDFs
│   └── ...
└── slow/              # heavy integration/performance tests
```

### Common test commands

```bash
# Run one subsystem's fast tests
uv run pytest tests/fast/arrow/ -v

# Run a specific test by name
uv run pytest tests/fast/api/test_relation.py -k 'test_value_relation' -v

# Parallel execution (pytest-xdist)
uv run pytest tests/fast/ -n auto -r skip -vv

# Filter by markers
uv run pytest tests/fast/ -r skip -vv -m "not (performance and threading)"

# Slow tests (be patient)
uv run pytest -n 4 -r skip -vv tests/slow/test_h2oai_arrow.py
```

### Markers

`capi`, `dbapi`, `threading`, `performance`, `types_map`, `type_union`, and more. See `pyproject.toml` `[tool.pytest.ini_options]` for the canonical list and default addopts (`-ra --verbose`).

### Test dependencies

The test dependency group includes heavy packages with complex platform constraints (tensorflow, torch, pyspark, numpy version splits). Install test deps without building the project:

```bash
uv sync --only-group test --no-install-project -p 3.13
```

See `pyproject.toml` `[dependency-groups]` for the full dependency matrix and its platform-specific constraints.

## Linting and formatting

```bash
# Python (ruff — configured in pyproject.toml, 120-char line length, google docstrings)
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking (mypy — strict mode, see [tool.mypy] in pyproject.toml)
uv run mypy

# Pre-commit hooks (configured in .pre-commit-config.yaml)
uvx pre-commit run --all-files
```

## Debugging

### Quick interpreter check

```bash
.venv/bin/python -c 'import duckdb; print(duckdb.__version__); duckdb.sql("SELECT 42").show()'
```

### AddressSanitizer (macOS)

```bash
DYLD_INSERT_LIBRARIES=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/lib/darwin/libclang_rt.asan_osx_dynamic.dylib \
    .venv/bin/python repro.py
```

### Memory profiling with memray

```bash
.venv/bin/python -m memray run -o profile.bin repro.py
.venv/bin/python -m memray flamegraph profile.bin
open memray-flamegraph-profile.html
```

### Python memory debug allocator

```bash
PYTHONMALLOC=malloc_debug .venv/bin/python repro.py
```

### macOS deployment target inspection

```bash
.venv/bin/python -c "import sysconfig; print(sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET'))"
```

### Timezone isolation

```bash
TZ=not/existing .venv/bin/python -c 'import duckdb; ...'
```

### Bug investigation repro convention

When investigating a specific issue, write scripts into:

```
debugscripts/<issue_number>_repro/repro.py
debugscripts/<issue_number>_repro/generate_data.py   # if synthetic data needed
```

This convention keeps repro scripts organized by issue and makes them easy to find, share, or reference in bug reports.

## Project structure

```
├── duckdb/                    # Python package (pure Python + extension module)
│   ├── __init__.py
│   ├── experimental/          # pyspark compatibility layer
│   ├── filesystem.py          # fsspec integration
│   └── query_graph/           # (old, unmaintained)
├── adbc_driver_duckdb/        # ADBC driver package
├── _duckdb-stubs/             # type stubs (*.pyi)
├── src/                       # C++ extension source (pybind11)
├── external/duckdb/           # DuckDB submodule
├── duckdb_packaging/          # custom build backend
├── tests/                     # test suite (fast/ + slow/)
├── scripts/                   # maintenance scripts
├── debugscripts/              # issue repro scripts (convention: <issue>_repro/)
├── pyproject.toml             # build config, deps, linting, CI
└── CMakeLists.txt             # CMake build system
```

### DuckDB submodule

The DuckDB engine is included as a git submodule at `external/duckdb/`. After creating a worktree or switching branches:

```bash
git submodule update --init --recursive
```

No `--depth=1` — the build backend needs version detection from git history.

## Development workflows

### Worktrees

This repository supports git worktrees for parallel development. The recommended pattern is to keep worktrees as siblings of the main checkout:

```bash
# Create a feature branch worktree
git worktree add -b feature/my-feature ../feature_my_feature v1.5-variegata
cd ../feature_my_feature
git submodule update --init --recursive
uv sync --only-group build --no-install-project -p 3.13
uv sync --no-build-isolation -v --reinstall -p 3.13
```

### GitHub CLI patterns

```bash
# CI status
gh run list --repo duckdb/duckdb-python --workflow="Packaging" --limit 5
gh run list --workflow pypi_packaging.yml --json status,url -s failure

# Download CI artifacts
gh run download <run-id> -D ./artifacts/

# PR workflows
gh pr checkout 167
gh pr create -B v1.5-variegata
```

## Scope

This file covers the **Python extension layer** — the pybind11 bindings, the build system, the test suite, and the Python packaging. The DuckDB core engine source is included as a submodule at `external/duckdb/` and is compiled from source as part of the build; debugging may require navigating into the submodule's C++ code.

**Free-threaded Python** is not supported in this client. A separate prototype client based on DuckDB's C API exists for free-threading, Stable ABI, and multi-interpreter support.

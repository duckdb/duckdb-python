# AGENTS.md

Guidance for coding agents working in this repository.

## Project overview

This is the **production** duckdb-python client, the `duckdb` package on PyPI. It provides Python bindings for [DuckDB](https://duckdb.org), an in-process OLAP database engine, built with nanobind and a custom scikit-build-core build backend.

- **Repository**: https://github.com/duckdb/duckdb-python
- **Package name**: `duckdb`
- **Bindings**: nanobind (3.x). The C++ extension module is `_duckdb`; the `duckdb` package wraps it.
- **Build backend**: `duckdb_packaging.build_backend`, a wrapper around scikit-build-core
- **Supported Python**: 3.11, 3.12, 3.13, 3.14, 3.15
- **Wheels**: starting with v2.0 (branch `v2.0-cyanoptera`) each platform ships one version-specific wheel for 3.11 and one stable ABI wheel tagged `cp312-abi3` that serves 3.12 through 3.15 and later releases. See "Stable ABI" below.
- **Free-threaded Python**: not supported. CI builds and tests no free-threaded interpreter, and no wheel is shipped for one. A separate prototype client based on DuckDB's C API targets free-threading and multi-interpreter support.

### Branches and binding libraries

- `v2.0-cyanoptera` and later: nanobind, stable ABI wheels. This file describes these branches.
- `v1.5-variegata`, `v1.4-andium`, `v1.3-ossivalis` and older release branches: pybind11, one wheel per Python version. Their own `CLAUDE.md`, where present, describes them. Do not carry pybind11 assumptions into v2.0 work or nanobind assumptions into 1.x fixes.

## IMPORTANT: build before running anything

**You MUST complete a full build before running tests, scripts, or `uv run` in a fresh worktree or after a clean slate.** `uv run pytest` triggers scikit-build-core's editable rebuild on import, which compiles 2000+ C++ files from scratch. That takes 5 to 10 minutes and exceeds the Bash tool's default timeout. Do not run tests, scripts, or `uv run python` until the two-step build below has completed.

```bash
# Step 1: install build deps (~5 seconds)
uv sync --only-group build --no-install-project -p 3.13

# Step 2: build the extension (3 to 10 min cold, ~30s with a warm compiler cache, use timeout: 600000)
uv sync --no-build-isolation -v --reinstall -p 3.13
```

After step 2 completes, `uv run pytest`, `uv run python`, and `.venv/bin/python` all work immediately. Subsequent C++ changes trigger fast incremental rebuilds, not full cold builds.

## Build system

### Editable install (standard development workflow)

The build uses `--no-build-isolation` for two reasons: (1) it makes incremental rebuilds fast by reusing the target venv, and (2) **it keeps the nanobind headers and CMake config in the venv so CLion and other C++ IDEs can resolve them**. With build isolation, nanobind is installed into a temp env that is destroyed after the build, leaving `compile_commands.json` with stale include paths. `CMakeLists.txt` locates nanobind through `python -m nanobind --cmake_dir`, so the venv's interpreter must have it installed.

`--no-build-isolation` requires that build dependencies are already installed. **This is a two-step process on a fresh venv:**

**Step 1: install build deps** (fast, ~5 seconds):

```bash
uv sync --only-group build --no-install-project -p 3.13
```

**Step 2: build the extension** (3 to 10 min cold, ~30 seconds with a warm compiler cache):

```bash
uv sync --no-build-isolation -v --reinstall -p 3.13
```

This produces a Debug editable install in `build/debug/` with `editable.mode = "redirect"`. Python code changes are picked up immediately; C++ changes require a rebuild. Editable builds keep the full CPython API (`wheel.py-api` is cleared by an override), so the debug asserts on GIL state are active. Only wheel builds use the stable ABI.

### Compiler cache (strongly recommended)

`cmake/compiler_launcher.cmake` picks up a compiler launcher automatically when `CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER` are unset. It looks for `ccache` first and `sccache` second. To force sccache on a machine that has both, export the launcher before any `uv sync`:

```bash
export CMAKE_C_COMPILER_LAUNCHER="$(command -v sccache)"
export CMAKE_CXX_COMPILER_LAUNCHER="$(command -v sccache)"
```

Without a cache, cold builds take 5 to 10 minutes (2000+ C++ compilation units). With a warm cache, they take about 30 seconds. Check cache state with `sccache -s`.

### Non-editable (release-style) install

```bash
uv sync --no-build-isolation --no-editable -v --reinstall -p 3.13
```

Produces a Release build (scikit-build-core's default build type) in `build/<cache_tag>/`, for example `build/cpython-313/`.

### Wheel build

```bash
uv build --wheel
```

Produces a wheel in `dist/`. On CPython 3.12 and later the wheel is tagged `cp312-abi3`; on 3.11 it is version-specific. CI uses cibuildwheel; see `pyproject.toml` `[tool.cibuildwheel]` for the platform matrix (macOS arm64/x86_64, Linux x86_64/aarch64, Windows AMD64/ARM64) and `.github/workflows/packaging_wheels.yml` for the Python matrix. cibuildwheel builds use `build/shared` as build dir.

### Stable ABI

From v2.0 the release wheels for CPython 3.12 and later are built against the limited API and tagged abi3. How it fits together:

- `pyproject.toml` sets `wheel.py-api = "cp312"`. scikit-build-core tags the wheel abi3 on 3.12 and later and hands CMake the `Development.SABIModule` component; on 3.11 the same setting yields a version-specific wheel.
- `CMakeLists.txt` builds `nanobind-static-abi3` and passes `STABLE_ABI` to `nanobind_add_module` under the same condition nanobind applies: CPython 3.12 or later, not free-threaded. nanobind puts `Py_LIMITED_API` on its abi3 library as a public define, so the binding object libraries and the module compile against the same Python headers.
- The limited API has no way to ask whether the current thread holds the GIL. The two sites that release the GIL only when held use the scope guard `PyUtil::GilReleaseIfHeld`, which on 3.12+ relies on `PyThreadState_Swap(nullptr)`. GIL asserts go through macros that are empty in stable-ABI builds and active in editable (full API) builds.
- CI builds the cp312 wheel once per platform with no inline tests, runs `abi3audit --strict` on it, and installs it into every interpreter it serves to run the suite: 3.12 to 3.15 on Linux, the oldest and newest of those on Windows and macOS in pull request runs, all of them in release runs. cibuildwheel would test a wheel only on the interpreter that built it.

When adding C++ that touches the CPython API, use only functions in the limited API or guard the call with `Py_LIMITED_API` checks. abi3audit in CI is the backstop; do not rely on it instead of checking.

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

Any `uv run` also rebuilds on import when C++ changed. Do not start a `uv run` while a long test run uses the same venv: the rebuild replaces the module under the running suite.

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

Supported: `3.11`, `3.12`, `3.13`, `3.14`, `3.15`. Do **not** use free-threaded variants (`3.13t`, `3.14t`, `3.15t`): the build falls back to a version-specific nanobind on them, but the client does not support them.

### Build configuration reference

Key `pyproject.toml` settings:

- `[build-system].requires`: `scikit-build-core>=0.11.4`, `nanobind>=3.0`, `numpy>=2.0` (C API headers for the numpy result path; a numpy 2.x build is compatible with numpy 1.19+ at runtime).
- `BUILD_EXTENSIONS = "core_functions;json;parquet;icu"`: extensions built into the module. `LINK_EXTENSIONS = ""` means link all of them. jemalloc is part of DuckDB core and the engine enables it on 64-bit Linux that is not musl, BSD, Android, z/OS or wasm.
- Default build dir `build/{cache_tag}`; cibuildwheel override `build/shared`.
- Editable overrides: `build-dir = "build/debug/"`, `editable.rebuild = true`, `editable.mode = "redirect"`, `cmake.build-type = "Debug"`, `wheel.py-api = ""`. On macOS only, `DISABLE_UNITY = "1"` is added for better debugging.
- Coverage overrides: `build-dir = "build/coverage/"`, `RelWithDebInfo`, `--coverage -O0` flags. Activate with `COVERAGE=true uv sync ...`.
- `[tool.duckdb_packaging].release_version`: the version this branch releases next. Bump it right after every release from the branch.

CMake option `DUCKDB_PY_ASAN=ON` instruments only the binding objects with AddressSanitizer and leaves the engine uninstrumented, so the engine keeps hitting the compiler cache. Pass it through `SKBUILD_CMAKE_DEFINE`.

## Testing

### Test layout

```
tests/
├── conftest.py        # shared fixtures, --skiplist option
├── fast/              # quick per-subsystem tests (seconds each)
│   ├── adbc/          # ADBC driver
│   ├── api/           # connection and module API
│   ├── arrow/         # pyarrow integration
│   ├── numpy/         # numpy integration
│   ├── pandas/        # pandas integration
│   ├── relational_api/
│   ├── spark/         # pyspark compatibility layer
│   ├── sqlite/        # SQLite-derived SQL tests
│   ├── types/         # type system
│   ├── udf/           # Python UDFs
│   └── test_*.py      # cross-cutting tests, including test_nanobind_cutover_regressions.py
├── slow/              # heavy integration and shutdown tests
├── coverage/          # coverage-specific tests
├── extensions/        # extension loading
└── spark_namespace/   # spark namespace tests
benchmarks/            # CodSpeed benchmarks, see benchmarks/README.md
```

### Common test commands

```bash
# Run one subsystem's fast tests
uv run pytest tests/fast/arrow/ -v

# Run a specific test by name
uv run pytest tests/fast/api/test_relation.py -k 'test_value_relation' -v

# Parallel execution (pytest-xdist)
uv run pytest tests/fast/ -n auto -r skip -vv

# Skip named tests or classes
uv run pytest tests/fast/ --skiplist test_name_a test_name_b

# Slow tests (be patient)
uv run pytest -n 4 -r skip -vv tests/slow/test_h2oai_arrow.py
```

### Pytest configuration

`[tool.pytest.ini_options]` in `pyproject.toml` sets `addopts = "-ra --verbose"` and `filterwarnings = ["error", ...]`: any warning not on the ignore list fails the test. The test tree registers no custom markers; tests use the built-in `skipif`, `xfail`, `timeout` and `parametrize`. The benchmark suite registers its own `gate` and `informational` markers in `benchmarks/conftest.py`.

### Test dependencies

The test dependency group includes heavy packages with complex platform constraints (tensorflow, torch, pyspark, numpy version splits). Install test deps without building the project:

```bash
uv sync --only-group test --no-install-project -p 3.13
```

See `pyproject.toml` `[dependency-groups]` for the full dependency matrix and its platform-specific constraints.

## Linting and formatting

```bash
# Python (ruff, configured in pyproject.toml, 120-char line length, google docstrings)
uv run ruff check duckdb/ tests/
uv run ruff format duckdb/ tests/

# Type checking (mypy, strict mode, see [tool.mypy] in pyproject.toml)
uv run mypy

# Pre-commit hooks (ruff, mypy, clang-format, cmake-format, basic file checks).
# Install pinned to 3.12 (cmakelang crashes on 3.14; keeps hooks off the build interpreter):
uv tool install --python 3.12 pre-commit
pre-commit install          # git hook, runs on commit
pre-commit run --all-files  # run across the tree
```

C++ formatting is clang-format 21 through pre-commit; CMake files go through cmake-format.

## Type stubs

Stubs live in `_duckdb-stubs/` and are maintained by hand. There is no stub generator; the pybind11-era generator scripts were removed with the nanobind port.

## Debugging

### Quick interpreter check

```bash
.venv/bin/python -c 'import duckdb; print(duckdb.__version__); duckdb.sql("SELECT 42").show()'
```

### AddressSanitizer

Binding-only instrumentation on any platform: rebuild with `DUCKDB_PY_ASAN=ON` (see the build configuration reference). On macOS, preload the runtime:

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
├── duckdb/                    # Python package wrapping the _duckdb extension module
│   ├── __init__.py
│   ├── experimental/          # pyspark compatibility layer
│   ├── filesystem.py          # fsspec integration
│   ├── func/, sqltypes/, value/
│   ├── polars_io.py, udf.py
│   └── query_graph/           # (old, unmaintained)
├── adbc_driver_duckdb/        # ADBC driver package
├── _duckdb-stubs/             # type stubs (*.pyi)
├── src/                       # C++ extension source (nanobind), one object library per subdirectory
│   └── include/duckdb_python/ # headers
├── cmake/                     # compiler launcher detection, DuckDB engine loader
├── external/duckdb/           # DuckDB submodule
├── duckdb_packaging/          # custom build backend, versioning, release scripts
├── tests/                     # test suite (see Test layout)
├── benchmarks/                # CodSpeed benchmark suite
├── scripts/                   # import cache generator, maintenance scripts
├── debugscripts/              # issue repro scripts (convention: <issue>_repro/, create as needed)
├── pyproject.toml             # build config, deps, linting, CI
└── CMakeLists.txt             # CMake build system
```

### DuckDB submodule

The DuckDB engine is included as a git submodule at `external/duckdb/`. After creating a worktree or switching branches:

```bash
git submodule update --init --recursive
```

No `--depth=1`: the build backend needs version detection from git history.

## Development workflows

### Worktrees

This repository supports git worktrees for parallel development. The recommended pattern is to keep worktrees as siblings of the main checkout:

```bash
# Create a feature branch worktree
git worktree add -b feature/my-feature ../feature_my_feature v2.0-cyanoptera
cd ../feature_my_feature
git submodule update --init --recursive
uv sync --only-group build --no-install-project -p 3.13
uv sync --no-build-isolation -v --reinstall -p 3.13
```

### Continuous integration

One rule per event, all in `.github/workflows/`:

- **Push to a fork branch** (`on_push.yml`): code quality, then the full-API editable build with assertions (Release plus `FORCE_ASSERT`) on the fast suite (`assert_test.yml`). Feature branches live in forks; upstream holds only the protected branches and the merge queue refs. This is the only build with `D_ASSERT` and the GIL asserts live; every wheel is Release.
- **Ready pull request and merge queue** (`on_pr.yml`): identical runs. Code quality; the sdist built and smoke-imported; the cp312 abi3 wheel built once per platform with no inline tests; the cp311 wheel on Linux with its inline tests; the full-API Release build with assertions (the engine's relassert) on the fast suite; the abi3 wheel installed and tested per interpreter (`test_abi3_wheels`). A draft runs only code quality. `Summary` is the required check.
- **Push to a protected branch** (`cache_warm.yml`): the only writer of every ccache: the seed wheels per platform, the assertions build and the pre-commit cache, builds only. Saving is gated on `github.ref_protected`, so a new release branch gets caches the day it gets a ruleset. Everything else restores and never saves.
- **Release** (`release.yml`): the full matrix, every interpreter on every platform, the sdist with the suite, no cache writes.
- **Coverage** (`coverage.yml`) and **benchmarks** (`codspeed.yml`): dispatch and nightly, not gates.

### GitHub CLI patterns

```bash
# CI status
gh run list --repo duckdb/duckdb-python --workflow="Packaging" --limit 5
gh run list --workflow packaging.yml --json status,url -s failure

# Download CI artifacts
gh run download <run-id> -D ./artifacts/

# PR workflows
gh pr checkout 167
gh pr create -B v2.0-cyanoptera
```

## Scope

This file covers the **Python extension layer**: the nanobind bindings, the build system, the test suite, and the Python packaging. The DuckDB core engine source is included as a submodule at `external/duckdb/` and is compiled from source as part of the build; debugging may require navigating into the submodule's C++ code.

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/duckdb/duckdb/refs/heads/main/logo/DuckDB_Logo-horizontal.svg">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/duckdb/duckdb/refs/heads/main/logo/DuckDB_Logo-horizontal-dark-mode.svg">
    <img alt="DuckDB logo" src="https://raw.githubusercontent.com/duckdb/duckdb/refs/heads/main/logo/DuckDB_Logo-horizontal.svg" height="100">
  </picture>
</div>
<br />
<p align="center">
  <a href="https://discord.gg/tcvwpjfnZx"><img src="https://shields.io/discord/909674491309850675" alt="Discord" /></a>
  <a href="https://pypi.org/project/duckdb/"><img src="https://img.shields.io/pypi/v/duckdb.svg" alt="PyPI Latest Release"/></a>
</p>
<br />
<p align="center">
  <a href="https://duckdb.org">DuckDB.org</a>
  |
  <a href="https://duckdb.org/docs/stable/guides/python/install">User Guide (Python)</a>
  -
  <a href="https://duckdb.org/docs/stable/clients/python/overview">API Docs (Python)</a>
</p>

# DuckDB: A Fast, In-Process, Portable, Open Source, Analytical Database System

* **Simple**: DuckDB is easy to install and deploy. It has zero external dependencies and runs in-process in its host application or as a single binary.
* **Portable**: DuckDB runs on Linux, macOS, Windows, Android, iOS and all popular hardware architectures. It has idiomatic client APIs for major programming languages.
* **Feature-rich**: DuckDB offers a rich SQL dialect. It can read and write file formats such as CSV, Parquet, and JSON, to and from the local file system and remote endpoints such as S3 buckets.
* **Fast**: DuckDB runs analytical queries at blazing speed thanks to its columnar engine, which supports parallel execution and can process larger-than-memory workloads.
* **Extensible**: DuckDB is extensible by third-party features such as new data types, functions, file formats and new SQL syntax. User contributions are available as community extensions.
* **Free**: DuckDB and its core extensions are open-source under the permissive MIT License. The intellectual property of the project is held by the DuckDB Foundation.

## Installation

Install the latest release of DuckDB directly from [PyPI](https://pypi.org/project/duckdb/):

```bash
pip install duckdb
```

Install with all optional dependencies:

```bash
pip install 'duckdb[all]'
```

## Development

Start by <a href="https://github.com/duckdb/duckdb-python/fork"><svg height="16" viewBox="0 0 16 16" version="1.1" width="16">
    <path fill-rule="evenodd" d="M5 3.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm0 2.122a2.25 2.25 0 10-1.5 0v.878A2.25 2.25 0 005.75 8.5h1.5v2.128a2.251 2.251 0 101.5 0V8.5h1.5a2.25 2.25 0 002.25-2.25v-.878a2.25 2.25 0 10-1.5 0v.878a.75.75 0 01-.75.75h-4.5A.75.75 0 015 6.25v-.878z"></path>
</svg>forking duckdb-python</a>.

### Cloning

After forking the duckdb-python repo we recommend you clone your fork as follows:
```shell
git clone --recurse-submodules $REPO_URL
git remote add upstream https://github.com/duckdb/duckdb-python.git
git fetch --all
```

... or, if you have already cloned your fork:
```shell
git submodule update --init --recursive
git remote add upstream https://github.com/duckdb/duckdb-python.git
git fetch --all
```

### Submodule update hook

If you'll be switching between branches that are have the submodule set to different refs, then make your life 
easier and add the git hooks in the .githooks directory to your local config: 
```shell
git config --local core.hooksPath .githooks/
```


### Editable installs (general)

  It's good to be aware of the following when performing an editable install:
- `uv sync` or `uv run [tool]` perform an editable install by default. We have 
  configured the project so that scikit-build-core will use a persistent build-dir, but since the build itself 
  happens in an isolated, ephemeral environment, cmake's paths will point to non-existing directories. CMake itself 
  will be missing.
- You should install all development dependencies, and then build the project without build isolation, in two separate 
  steps. After this you can happily keep building and running, as long as you don't forget to pass in the 
  `--no-build-isolation` flag.

```bash
# install all dev dependencies without building the project (needed once)
uv sync -p 3.11 --no-install-project
# build and install without build isolation
uv sync --no-build-isolation
```

### Editable installs (IDEs)

  If you're using an IDE then life is a little simpler. You install build dependencies and the project in the two 
  steps outlined above, and from that point on you can rely on e.g. CLion's cmake capabilities to do incremental 
  compilation and editable rebuilds. This will skip scikit-build-core's build backend and all of uv's dependency 
  management, so for "real" builds you better revert to the CLI. However, this should work fine for coding and debugging.


### Cleaning

```shell
uv cache clean
rm -rf build .venv uv.lock
```


### Building wheels and sdists

To build a wheel and sdist for your system and the default Python version:
```bash
uv build
````

To build a wheel for a different Python version:
```bash
# E.g. for Python 3.9
uv build -p 3.9
```

### Running tests

  Run all pytests:
```bash
uv run --no-build-isolation pytest ./tests --verbose
```

  Exclude the test/slow directory:
```bash
uv run --no-build-isolation pytest ./tests --verbose --ignore=./tests/slow
```

### Test coverage

  Run with coverage (during development you probably want to specify which tests to run):
```bash
COVERAGE=1 uv run --no-build-isolation coverage run -m pytest ./tests --verbose
```

  The `COVERAGE` env var will compile the extension with `--coverage`, allowing us to collect coverage stats of C++ 
  code as well as Python code.

  Check coverage for Python code:
```bash
uvx coverage html -d htmlcov-python
uvx coverage report --format=markdown
```

  Check coverage for C++ code (note: this will clutter your project dir with html files, consider saving them in some 
  other place):
```bash
uvx gcovr \
  --gcov-ignore-errors all \
  --root "$PWD" \
  --filter "${PWD}/src/duckdb_py" \
  --exclude '.*/\.cache/.*' \
  --gcov-exclude '.*/\.cache/.*' \
  --gcov-exclude '.*/external/.*' \
  --gcov-exclude '.*/site-packages/.*' \
  --exclude-unreachable-branches \
  --exclude-throw-branches \
  --html --html-details -o coverage-cpp.html \
  build/coverage/src/duckdb_py \
  --print-summary
```

### Typechecking and linting

- We're not running any mypy typechecking tests at the moment
- We're not running any Ruff / linting / formatting at the moment

### Cibuildwheel

You can run cibuildwheel locally for Linux. E.g. limited to Python 3.9:
```bash
CIBW_BUILD='cp39-*' uvx cibuildwheel --platform linux .
```

### Code conventions

* Follow the [Google Python styleguide](https://google.github.io/styleguide/pyguide.html)
* See the section on [Comments and Docstrings](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)

### Tooling

This codebase is developed with the following tools:
- [Astral uv](https://docs.astral.sh/uv/) - for dependency management across all platforms we provide wheels for,
  and for Python environment management. It will be hard to work on this codebase without having UV installed.
- [Scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html) - the build backend for
  building the extension. On the background, scikit-build-core uses cmake and ninja for compilation.
- [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) - a bridge between C++ and Python.
- [CMake](https://cmake.org/) - the build system for both DuckDB itself and the DuckDB Python module.
- Cibuildwheel

### Merging changes to pythonpkg from duckdb main

1. Checkout main
2Identify the merge commits that brought in tags to main:
```bash
git log --graph --oneline --decorate main --simplify-by-decoration
```

3. Get the log of commits
```bash
git log --oneline 71c5c07cdd..c9254ecff2 -- tools/pythonpkg/
```

4. Checkout v1.3-ossivalis
5. Get the log of commits
```bash
git log --oneline v1.3.0..v1.3.1 -- tools/pythonpkg/
```
git diff --name-status 71c5c07cdd c9254ecff2 -- tools/pythonpkg/

```bash
git log --oneline 71c5c07cdd..c9254ecff2 -- tools/pythonpkg/
git diff --name-status <HASH_A> <HASH_B> -- tools/pythonpkg/
```


## Versioning and Releases

The DuckDB Python package versioning and release scheme follows that of DuckDB itself. This means that a `X.Y.Z[.
postN]` release of the Python package ships the DuckDB stable release `X.Y.Z`. The optional `.postN` releases ship the same stable release of DuckDB as their predecessors plus Python package-specific fixes and / or features.

| Types                                                                  | DuckDB Version | Resulting Python Extension Version |
|------------------------------------------------------------------------|----------------|------------------------------------|
| Stable release: DuckDB stable release                                  | `1.3.1`        | `1.3.1`                            |
| Stable post release: DuckDB stable release + Python fixes and features | `1.3.1`        | `1.3.1.postX`                      |
| Nightly micro: DuckDB next micro nightly + Python next micro nightly   | `1.3.2.devM`   | `1.3.2.devN`                       |
| Nightly minor: DuckDB next minor nightly + Python next minor nightly   | `1.4.0.devM`   | `1.4.0.devN`                       |

Note that we do not ship nightly post releases (e.g. we don't ship `1.3.1.post2.dev3`).

### Branch and Tag Strategy

We cut releases as follows:

| Type                 | Tag          | How                                                                             |
|----------------------|--------------|---------------------------------------------------------------------------------|
| Stable minor release | vX.Y.0       | Adding a tag on `main`                                                          |
| Stable micro release | vX.Y.Z       | Adding a tag on a minor release branch (e.g. `v1.3-ossivalis`)                  |
| Stable post release  | vX.Y.Z-postN | Adding a tag on a post release branch (e.g. `v1.3.1-post`)                      |
| Nightly micro        | _not tagged_ | Combining HEAD of the _micro_ release branches of DuckDB and the Python package |
| Nightly minor        | _not tagged_ | Combining HEAD of the _minor_ release branches of DuckDB and the Python package |

### Release Runbooks

We cut a new **stable minor release** with the following steps:
1. Create a PR on `main` to pin the DuckDB submodule to the tag of its current release.
1. Iff all tests pass in CI, merge the PR.
1. Manually start the release workflow with the hash of this commit, and the tag name.
1. Iff all goes well, create a new PR to let the submodule track DuckDB main.

We cut a new **stable micro release** with the following steps:
1. Create a PR on the minor release branch to pin the DuckDB submodule to the tag of its current release.
1. Iff all tests pass in CI, merge the PR.
1. Manually start the release workflow with the hash of this commit, and the tag name.
1. Iff all goes well, create a new PR to let the submodule track DuckDB's minor release branch.

We cut a new **stable post release** with the following steps:
1. Create a PR on the post release branch to pin the DuckDB submodule to the tag of its current release.
1. Iff all tests pass in CI, merge the PR.
1. Manually start the release workflow with the hash of this commit, and the tag name.
1. Iff all goes well, create a new PR to let the submodule track DuckDB's minor release branch.

### Dynamic Versioning Integration

The package uses `setuptools_scm` with `scikit-build` for automatic version determination, and implements a custom
versioning scheme.

- **pyproject.toml configuration**:
  ```toml
  [tool.scikit-build]
  metadata.version.provider = "scikit_build_core.metadata.setuptools_scm"
  
  [tool.setuptools_scm]
  version_scheme = "duckdb_packaging._setuptools_scm_version:version_scheme"
  ```

- **Environment variables**:
  - `MAIN_BRANCH_VERSIONING=0`: Use release branch versioning (patch increments)
  - `MAIN_BRANCH_VERSIONING=1`: Use main branch versioning (minor increments)
  - `OVERRIDE_GIT_DESCRIBE`: Override version detection

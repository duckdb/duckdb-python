# Changelog

## v1.4.1
**DuckDB Core**: v1.4.1

### Bug Fixes
- **ADBC Driver**: Fixed ADBC driver implementation (#81)
- **SQLAlchemy compatibility**: Added `__hash__` method overload (#61)
- **Error Handling**: Reset PyErr before throwing Python exceptions (#69)
- **Polars Lazyframes**: Fixed Polars expression pushdown (#102)

### Code Quality Improvements & Developer Experience
- **MyPy Support**: MyPy is functional again and better integrated with the dev workflow
- **Stubs**: Re-created and manually curated stubs for the binary extension
- **Type Shadowing**: Deprecated `typing` and `functional` modules
- **Linting & Formatting**: Comprehensive code quality improvements with Ruff
- **Type Annotations**: Added missing overloads and improved type coverage
- **Pre-commit Integration**: Added ruff, clang-format, cmake-format and mypy configs
- **CI/CD**: Added code quality workflow

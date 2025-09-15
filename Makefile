PYTHON ?= python3

format-main:
	$(PYTHON) external/duckdb/scripts/format.py main --fix --noconfirm
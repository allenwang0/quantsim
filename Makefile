.PHONY: setup browsers fixture test lint report bench

setup:  ## create .venv and install quantsim with every extra, pinned by uv.lock
	uv sync --all-extras

browsers:  ## Chromium for the Playwright report test (about 600 MB, kept in .playwright/)
	PLAYWRIGHT_BROWSERS_PATH=.playwright uv run python -m playwright install chromium

fixture:
	uv run python tests/fixtures/make_fixture.py

test:
	PLAYWRIGHT_BROWSERS_PATH=.playwright uv run pytest -m "not live" -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

report: fixture  ## rebuild results/demo (seeded) and the published page docs/index.html
	uv run quantsim report --prices tests/fixtures/synthetic_prices.csv --strategy sma momentum --out results/demo --card
	python3 scripts/build_report.py --readme README.md

bench:
	cd benchmarks && uv run python bench_engines.py

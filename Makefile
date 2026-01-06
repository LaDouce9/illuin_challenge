.PHONY: install test lint format run

install:
	uv sync

test:
	uv run pytest tests

lint:
	uv run ruff check .

format:
	uv run ruff format .

run:
	uv run python -m src.main

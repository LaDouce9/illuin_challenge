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

build:
	docker build -t illuin-challenge .

run-jupyter:
	docker run --name illuin-jupyter -p 8888:8888 -v "$(PWD):/app" -v /app/.venv illuin-challenge

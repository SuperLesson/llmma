all: deps tidy

deps:
	uv sync
	uv pip compile pyproject.toml -o requirements.txt

update:
	uv sync -U
	uv pip compile pyproject.toml -o requirements.txt

lint:
	uv run ruff check
	uv run pyright

tidy:
	uv run ruff check --fix --unsafe-fixes || true
	uv run ruff format

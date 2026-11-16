.PHONY: setup setup-models run

setup:
	uv sync --locked

setup-models:
	SAM2_BUILD_CUDA=0 uv sync --locked --extra models

run:
	uv run --locked --no-sync python -m data_generation

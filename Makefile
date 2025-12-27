.PHONY: install setup test lint format clean train infer download

install:
	uv sync || poetry install

setup: install
	pre-commit install

test:
	uv run --with pytest pytest tests/ -v

lint:
	uv run --with pre-commit pre-commit run --all-files

format:
	ruff check --fix .
	prettier --write "**/*.{md,json,yaml}"

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .ruff_cache

train:
	uv run main.py train

infer:
	uv run main.py infer --checkpoint MODEL.ckpt --input-csv input.csv --output-csv output.csv

test-model:
	uv run main.py test --checkpoint MODEL.ckpt

convert:
	uv run main.py convert --checkpoint MODEL.ckpt

infer-triton:
	uv run main.py infer-triton --input-csv input.csv --output-csv output.csv

download:
	./get_data.sh

debug-train:
	uv run main.py train train_config.data_config.max_samples=1500 train_config.num_epochs=2 train_config.accelerator=cpu train_config.precision=32

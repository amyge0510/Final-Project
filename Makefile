.PHONY: setup test lint format clean run-api

setup:
	python -m pip install -e ".[dev]"

test:
	pytest tests/

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

process-data:
	python src/data_processing/ingest.py

build-graph:
	python src/graph_construction/build_graph.py

help:
	@echo "Available commands:"
	@echo "  setup        - Install the package and development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Check code style"
	@echo "  format       - Format code"
	@echo "  clean        - Remove generated files"
	@echo "  run-api      - Start the API server"
	@echo "  process-data - Process raw data"
	@echo "  build-graph  - Build the knowledge graph" 
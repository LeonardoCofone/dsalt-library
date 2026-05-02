.PHONY: help install install-dev test test-cov lint format clean build publish docs

help:
	@echo "Available commands:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linters"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build distribution"
	@echo "  publish      Publish to PyPI"
	@echo "  docs         Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest --cov=dsalt --cov-report=html tests/

lint:
	flake8 dsalt tests
	mypy dsalt

format:
	black dsalt tests
	isort dsalt tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf dsalt/__pycache__/
	rm -rf tests/__pycache__/

build:
	python -m build

publish: clean build
	twine upload dist/*

docs:
	sphinx-build docs docs/_build/html
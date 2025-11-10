# Copyright 2025 Chaukas AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

.PHONY: help install install-dev test lint format type-check clean build publish test-install run-examples clean-all

# Default target - show help
help:
	@echo "Chaukas SDK - Available targets:"
	@echo ""
	@echo "  Development:"
	@echo "    install          - Install package in editable mode"
	@echo "    install-dev      - Install package with dev dependencies"
	@echo "    install-all      - Install with all optional dependencies"
	@echo ""
	@echo "  Testing & Quality:"
	@echo "    test             - Run tests with pytest"
	@echo "    test-cov         - Run tests with coverage report"
	@echo "    lint             - Run all linters (black, isort, mypy)"
	@echo "    format           - Format code with black and isort"
	@echo "    type-check       - Run mypy type checker"
	@echo ""
	@echo "  Building:"
	@echo "    build            - Build distribution packages"
	@echo "    clean            - Clean build artifacts"
	@echo "    clean-all        - Clean all generated files"
	@echo ""
	@echo "  Package Testing:"
	@echo "    test-install     - Build and test package installation"
	@echo ""
	@echo "  Examples:"
	@echo "    run-examples     - Run all examples"

# Install package in editable mode
install:
	pip install -e .

# Install with dev dependencies
install-dev:
	pip install -e ".[dev]"

# Install with all optional dependencies
install-all:
	pip install -e ".[dev,openai,google,crewai]"

# Run tests (OpenAI integration tests are skipped - they need updating)
test:
	pytest -v

# Run tests with coverage
test-cov:
	pytest -v --cov=src --cov-report=html --cov-report=term

# Format code
format:
	black .
	isort .
	@echo "✅ Code formatted successfully"

# Check formatting (for CI)
format-check:
	black --check .
	isort --check-only .

# Run type checker
type-check:
	mypy src/

# Run all linters
lint: format-check type-check
	@echo "✅ All linting checks passed"

# Build distribution packages
build:
	python -m build
	@echo "✅ Package built successfully!"
	@echo "📦 Available packages:"
	@ls -lh dist/

# Test package installation
test-install: build
	@echo "🧪 Testing package installation..."
	pip install --force-reinstall dist/chaukas_sdk-*.whl
	python -c "from chaukas.sdk import ChaukasClient; print('✅ Package installs and imports correctly')"
	pip uninstall -y chaukas-sdk
	@echo "✅ Package installation test passed!"

# Clean build artifacts
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "✅ Cleaned build artifacts"

# Clean all generated files including venv
clean-all: clean
	rm -rf .venv
	@echo "✅ Cleaned all generated files"

# Run examples (if they exist)
run-examples:
	@echo "Running examples..."
	@for example in examples/*/; do \
		if [ -f "$$example/main.py" ]; then \
			echo "Running $$example..."; \
			cd "$$example" && python main.py && cd ../..; \
		fi \
	done
	@echo "✅ All examples completed"

# Install dependencies from pyproject.toml
deps:
	pip install --upgrade pip
	pip install build twine pytest pytest-asyncio pytest-mock pytest-cov
	pip install black isort mypy

# Publish to PyPI (use with caution - prefer GitHub Actions)
publish: build
	@echo "⚠️  Publishing to PyPI..."
	@echo "⚠️  Make sure you have configured PyPI credentials!"
	twine upload dist/*

# Publish to Test PyPI
publish-test: build
	@echo "Publishing to Test PyPI..."
	twine upload --repository testpypi dist/*

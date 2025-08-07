# Makefile for Legal Contract Processing Pipeline
# ==============================================

.PHONY: help install install-dev test test-coverage lint format clean run example docker

# Default target
help:
	@echo "Legal Contract Processing Pipeline - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  setup        Complete setup including NLTK data"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  test         Run unit tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run linting (flake8)"
	@echo "  format       Format code with black"
	@echo "  check        Run all quality checks"
	@echo ""
	@echo "Execution:"
	@echo "  run          Run pipeline with default settings"
	@echo "  example      Run example usage script"
	@echo "  demo         Run pipeline with sample data"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean        Clean up generated files"
	@echo "  clean-all    Clean everything including dependencies"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run pipeline in Docker container"

# Installation targets
install:
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt
	@echo "✅ Production dependencies installed"

install-dev:
	@echo "📦 Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest black flake8 mypy pre-commit
	@echo "✅ Development dependencies installed"

setup: install
	@echo "🔧 Setting up NLTK data..."
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
	@echo "📁 Creating directories..."
	mkdir -p data outputs logs
	@echo "⚙️ Setup completed successfully!"

# Quality assurance targets
test:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ --cov=contract_processor --cov-report=html --cov-report=term-missing

lint:
	@echo "🔍 Running linting checks..."
	flake8 contract_processor.py tests/ --max-line-length=100 --ignore=E203,W503

format:
	@echo "🎨 Formatting code..."
	black contract_processor.py tests/ example_usage.py config.py --line-length=100

check: lint test
	@echo "✅ All quality checks passed!"

# Execution targets
run:
	@echo "🚀 Running pipeline with default settings..."
	@if [ ! -d "data" ]; then echo "❌ Data directory not found. Please create 'data' directory and add contract files."; exit 1; fi
	python contract_processor.py --data_dir ./data --output ./outputs/analysis.csv

example:
	@echo "📚 Running example usage script..."
	python example_usage.py

demo:
	@echo "🎯 Running pipeline demo..."
	@if [ ! -f ".env" ]; then echo "OPENAI_API_KEY=your-key-here" > .env.example; echo "⚠️  Please copy .env.example to .env and add your API key"; fi
	@if [ -f ".env" ]; then \
		echo "Loading environment variables..."; \
		export $$(cat .env | xargs) && python contract_processor.py --data_dir ./data --max_contracts 5; \
	else \
		echo "❌ Please create .env file with your OpenAI API key"; \
	fi

# Maintenance targets
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -f *.log
	@echo "✅ Cleanup completed"

clean-all: clean
	@echo "🧹 Deep cleaning..."
	rm -rf outputs/*
	rm -rf .coverage
	pip uninstall -y legal-contract-processor
	@echo "✅ Deep cleanup completed"

# Development workflow
dev-setup: install-dev setup
	@echo "🔧 Setting up pre-commit hooks..."
	pre-commit install
	@echo "✅ Development environment ready!"

# Docker targets
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t legal-contract-processor:latest .

docker-run:
	@echo "🐳 Running pipeline in Docker..."
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs \
		-e OPENAI_API_KEY=${OPENAI_API_KEY} \
		legal-contract-processor:latest

# Data management
download-cuad:
	@echo "📥 Downloading CUAD dataset sample..."
	@if command -v wget >/dev/null 2>&1; then \
		mkdir -p data/cuad_sample; \
		echo "Please download CUAD dataset manually from https://www.atticusprojectai.org/cuad"; \
	else \
		echo "❌ wget not found. Please download CUAD dataset manually"; \
	fi

# Performance testing
benchmark:
	@echo "⚡ Running performance benchmarks..."
	python -m pytest tests/test_contract_processor.py::run_performance_tests -v

# Configuration management
config-example:
	@echo "⚙️ Creating example configuration..."
	python config.py

# Documentation
docs:
	@echo "📖 Generating documentation..."
	@if command -v pandoc >/dev/null 2>&1; then \
		pandoc README.md -o README.pdf; \
		echo "✅ Documentation generated"; \
	else \
		echo "❌ pandoc not found. Install pandoc to generate PDF documentation"; \
	fi

# Deployment targets
package:
	@echo "📦 Creating distribution package..."
	python setup.py sdist bdist_wheel
	@echo "✅ Package created in dist/"

install-package: package
	@echo "📦 Installing package..."
	pip install dist/*.whl --force-reinstall

# Monitoring and logging
logs:
	@echo "📋 Viewing recent logs..."
	@if [ -f "contract_processing.log" ]; then \
		tail -50 contract_processing.log; \
	else \
		echo "No log file found"; \
	fi

monitor:
	@echo "👀 Monitoring pipeline execution..."
	@if [ -f "contract_processing.log" ]; then \
		tail -f contract_processing.log; \
	else \
		echo "No log file found. Run the pipeline first."; \
	fi

# Git workflow helpers
commit-check: format lint test
	@echo "✅ Pre-commit checks passed. Ready to commit!"

release-check: clean test-cov lint
	@echo "🚀 Release checks passed. Ready for deployment!"

# Environment validation
validate-env:
	@echo "🔍 Validating environment..."
	@python -c "import sys; print(f'Python version: {sys.version}')"
	@python -c "import nltk, pandas, openai, sentence_transformers; print('✅ All required packages available')" 2>/dev/null || echo "❌ Some packages missing"
	@if [ -z "${OPENAI_API_KEY}" ]; then echo "⚠️  OPENAI_API_KEY not set"; else echo "✅ OPENAI_API_KEY is set"; fi

# Quick start guide
quickstart:
	@echo "🚀 Quick Start Guide"
	@echo "==================="
	@echo "1. Install dependencies: make install"
	@echo "2. Set up environment: make setup"
	@echo "3. Add your OpenAI API key to .env file"
	@echo "4. Place contract files in ./data directory"
	@echo "5. Run pipeline: make run"
	@echo ""
	@echo "For development:"
	@echo "1. make dev-setup"
	@echo "2. make example"
	@echo ""
	@echo "For testing:"
	@echo "1. make test"
	@echo "2. make check"

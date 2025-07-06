# Marathon Time Predictor Makefile
# Common development tasks

.PHONY: help install install-dev test test-cov lint format clean run train docs

# Default target
help:
	@echo "Marathon Time Predictor - Available commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  run          Run the FastAPI server"
	@echo "  train        Train the model"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean up generated files"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo ""
	@echo "Quality:"
	@echo "  security     Run security checks"
	@echo "  type-check   Run type checking with mypy"
	@echo "  all-checks   Run all quality checks"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Development
run:
	python marathon_api.py

train:
	python -c "from marathon_prediction import MarathonPrediction; model = MarathonPrediction(); metrics = model.train_model(); print('Training completed!'); print(f'R² Score: {metrics[\"r2_score\"]:.3f}'); print(f'MAE: {metrics[\"mae_minutes\"]:.1f} minutes')"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=marathon_prediction --cov-report=html --cov-report=term-missing -v

# Code quality
lint:
	flake8 marathon_prediction.py marathon_api.py tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	black marathon_prediction.py marathon_api.py tests/
	isort marathon_prediction.py marathon_api.py tests/

type-check:
	mypy marathon_prediction.py marathon_api.py --ignore-missing-imports

security:
	bandit -r . -f json -o bandit-report.json
	safety check --json --output safety-report.json

all-checks: format lint type-check security test

# Documentation
docs:
	@echo "Building documentation..."
	@echo "Documentation is available in docs/ directory"
	@echo "API documentation: docs/api.md"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "bandit-report.json" -delete
	find . -type f -name "safety-report.json" -delete
	@echo "Cleanup completed!"

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup completed!"
	@echo "Run 'make run' to start the server"
	@echo "Run 'make test' to run tests"

# Quick start
quick-start: install-dev train run
	@echo "Quick start completed!"
	@echo "Server is running at http://localhost:8000"
	@echo "API docs available at http://localhost:8000/docs"

# Docker support (if needed)
docker-build:
	docker build -t marathon-predictor .

docker-run:
	docker run -p 8000:8000 marathon-predictor

# Release tasks
release-check: all-checks
	@echo "Running release checks..."
	@echo "All checks passed! Ready for release."

# Help for specific tasks
test-help:
	@echo "Test commands:"
	@echo "  make test         - Run all tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  pytest tests/ -k test_name  - Run specific test"
	@echo "  pytest tests/ -m unit        - Run only unit tests"
	@echo "  pytest tests/ -m integration - Run only integration tests"
	@echo "  pytest tests/ -v             - Verbose output"

format-help:
	@echo "Formatting commands:"
	@echo "  make format       - Format all Python files"
	@echo "  black .           - Format with black only"
	@echo "  isort .           - Sort imports only"
	@echo "  black --check .   - Check formatting without changing"
	@echo "  isort --check-only . - Check import sorting without changing" 
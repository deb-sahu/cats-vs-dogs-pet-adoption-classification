.PHONY: help install install-dev test test-cov lint format train run docker-build docker-run docker-test clean mlflow-ui

help:
	@echo "Available commands:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run linters"
	@echo "  make format        Format code"
	@echo "  make train         Train the model"
	@echo "  make run           Run the API server"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make docker-test   Test Docker container"
	@echo "  make mlflow-ui     Start MLflow UI"
	@echo "  make clean         Clean up artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

train:
	python scripts/train.py --create-sample --epochs 5

train-full:
	python scripts/train.py --epochs 10

run:
	uvicorn src.api.main:app --reload --port 8000

docker-build:
	docker build -t catdog-classifier:latest .

docker-run:
	docker run -d -p 8000:8000 --name catdog-api catdog-classifier:latest

docker-stop:
	docker stop catdog-api && docker rm catdog-api

docker-test:
	@echo "Testing health endpoint..."
	curl -s http://localhost:8000/health | python -m json.tool
	@echo "\nTesting docs endpoint..."
	curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

mlflow-ui:
	mlflow ui --port 5000

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf artifacts/*.png artifacts/*.json
	rm -rf mlruns mlartifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Kubernetes commands
k8s-deploy:
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete -f k8s/

k8s-logs:
	kubectl logs -l app=catdog-classifier -f

k8s-port-forward:
	kubectl port-forward svc/catdog-classifier-service 8080:80

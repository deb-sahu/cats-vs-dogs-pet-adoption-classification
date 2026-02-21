# Local Setup Guide

This guide provides step-by-step instructions to set up and run the Cats vs Dogs classifier locally.

---

## Prerequisites

- Python 3.10 or higher (recommended: 3.11)
- pip or conda
- Docker and Docker Compose (for containerized deployment)
- Minikube and kubectl (for Kubernetes deployment)
- Helm (for monitoring stack)

---

## 1. Environment Setup

### Clone and Create Virtual Environment

```bash
# Navigate to project directory
cd mlops2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development (includes testing tools):
pip install -r requirements-dev.txt
```

---

## 2. Data Preparation

### Option A: Download from Kaggle (Full Dataset)

Requires Kaggle API credentials in `~/.kaggle/kaggle.json`:

```bash
python -m src.data.download
```

### Option B: Create Sample Dataset (Quick Testing)

```bash
python -m src.data.download --sample --n-samples 100
```

This creates synthetic images for testing the pipeline without downloading the full dataset.

---

## 3. Model Training

### Train with MLflow Tracking

```bash
# Train with sample data
python scripts/train.py --create-sample --epochs 5

# Train with full dataset (if downloaded)
python scripts/train.py --epochs 10 --lr 0.001
```

### View MLflow Experiments

```bash
mlflow ui --port 5000
```

Open http://localhost:5000 in your browser.

**Logged Metrics:**
- Training/validation loss and accuracy
- Test accuracy, precision, recall, F1, ROC-AUC
- Confusion matrix and training curves (artifacts)

---

## 4. Run API Locally

### Start the FastAPI Server

```bash
# Using uvicorn directly
uvicorn src.api.main:app --reload --port 8000

# Or using Make
make run
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Swagger documentation
open http://localhost:8000/docs

# Test prediction (replace with actual image path)
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/cat_or_dog.jpg"
```

---

## 5. Docker Deployment

### Build and Run Container

```bash
# Build image
docker build -t catdog-classifier .

# Run container
docker run -d -p 8000:8000 --name catdog-api catdog-classifier

# Test
curl http://localhost:8000/health

# Stop and remove
docker stop catdog-api && docker rm catdog-api
```

### Run with Docker Compose (Full Stack)

```bash
# Start all services (API, MLflow, Prometheus, Grafana)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

**Services:**
| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | - |
| MLflow | http://localhost:5000 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin123 |

---

## 6. Kubernetes Deployment (Minikube)

### Start Minikube

```bash
# Install minikube (macOS)
brew install minikube

# Start cluster
minikube start

# Use Minikube's Docker daemon
eval $(minikube docker-env)
```

### Build and Deploy

```bash
# Build image in Minikube's Docker
docker build -t catdog-classifier:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for pods to be ready
kubectl -n catdog-classifier get pods -w

# Port forward to access API
kubectl -n catdog-classifier port-forward svc/catdog-classifier-service 8080:80
```

### Test Deployed API

```bash
# Health check
curl http://localhost:8080/health

# Make prediction
curl -X POST http://localhost:8080/predict \
  -F "file=@test_image.jpg"

# Check Prometheus metrics
curl http://localhost:8080/metrics
```

---

## 7. Monitoring Setup

### Install Prometheus & Grafana via Helm

```bash
# Run setup script
./scripts/setup_monitoring.sh

# Or manually:
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring --create-namespace \
  -f k8s/monitoring/prometheus-values.yaml

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword=admin123 \
  -f k8s/monitoring/grafana-values.yaml
```

### Access Monitoring Services

```bash
# Prometheus
kubectl -n monitoring port-forward svc/prometheus-server 9090:80 &

# Grafana
kubectl -n monitoring port-forward svc/grafana 3000:80 &
```

**Access URLs:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / admin123)

### Import Dashboard

1. Open Grafana
2. Go to Dashboards → Import
3. Upload `k8s/monitoring/grafana-dashboard.json`

---

## 8. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

## 9. Smoke Tests

After deployment, verify the service is working:

```bash
# Local API
python scripts/smoke_test.py --url http://localhost:8000

# Kubernetes deployment
python scripts/smoke_test.py --url http://localhost:8080
```

---

## 10. Simulate Traffic (for Monitoring)

Generate load to see metrics in Grafana:

```bash
python scripts/simulate_traffic.py \
  --url http://localhost:8000 \
  --rps 2 \
  --duration 60
```

---

## Troubleshooting

### Model Not Loading

```bash
# Check if model file exists
ls -la models/model.pt

# Train model if missing
python scripts/train.py --create-sample --epochs 5
```

### Docker Build Fails

```bash
# Clean Docker cache
docker system prune -af

# Rebuild
docker build --no-cache -t catdog-classifier .
```

### Kubernetes Pods Not Starting

```bash
# Check pod status
kubectl -n catdog-classifier describe pods

# Check logs
kubectl -n catdog-classifier logs -l app=catdog-classifier
```

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Train model | `python scripts/train.py --create-sample` |
| Run API | `uvicorn src.api.main:app --reload` |
| Run tests | `pytest tests/ -v` |
| Docker build | `docker build -t catdog-classifier .` |
| Docker run | `docker run -p 8000:8000 catdog-classifier` |
| K8s deploy | `kubectl apply -f k8s/` |
| Smoke test | `python scripts/smoke_test.py` |

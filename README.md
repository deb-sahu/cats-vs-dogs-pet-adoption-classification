# Cats vs Dogs Classifier - MLOps Project

[![CI/CD Pipeline](https://github.com/deb-sahu/mlops2/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/deb-sahu/mlops2/actions/workflows/ci-cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready MLOps pipeline for binary image classification (Cats vs Dogs) using PyTorch, MLflow, FastAPI, Docker, GitHub Actions, Kubernetes (Minikube), and Prometheus/Grafana monitoring.

---

> ### 📖 **For Evaluators/Instructors**
>
> | Document | Description |
> |----------|-------------|
> | **➡️ [Local Setup Guide](docs/SETUP.md)** | Step-by-step instructions to run everything locally |
> | **📄 [Final Report](Final%20Report.docx)** | Assignment report with screenshots |
>
> The setup guide covers: Model training, API testing, Kubernetes deployment, and Prometheus + Grafana monitoring.

---

## 📋 Table of Contents

- [**Setup Guide**](docs/SETUP.md) ⭐
- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Docker Deployment](#-docker-deployment)
- [Kubernetes Deployment](#-kubernetes-deployment-minikube)
- [Monitoring](#-monitoring-with-prometheus--grafana)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Testing](#-testing)
- [Model Information](#-model-information)
- [Architecture](#-architecture)

---

## 🎯 Overview

This project implements an end-to-end MLOps pipeline for classifying images of cats and dogs, designed for a pet adoption platform.

### Problem Statement

Build a machine learning classifier to identify cats vs dogs from images, and deploy the solution as a containerized, monitored API with CI/CD automation.

### Dataset

- **Source**: [Kaggle Cats and Dogs Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Preprocessing**: Resized to 224x224 RGB images
- **Split**: 80% train / 10% validation / 10% test
- **Augmentation**: Random flip, rotation, color jitter

---

## ✨ Features

- **Data Pipeline**: Automated dataset download, preprocessing, and versioning with DVC
- **Model Training**: PyTorch CNN with MLflow experiment tracking
- **REST API**: FastAPI-based inference endpoint with Prometheus metrics
- **Containerization**: Multi-stage Docker build for optimized images
- **CI/CD**: GitHub Actions pipeline with lint, test, build, and deploy stages
- **Kubernetes**: Deployment manifests for Minikube with HPA
- **Monitoring**: Prometheus + Grafana dashboards for real-time metrics

---

## 📁 Project Structure

```
mlops2/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # CI/CD: lint, test, train, build, deploy
├── data/
│   ├── raw/                       # Original dataset (DVC tracked)
│   └── processed/                 # Preprocessed images
├── docs/
│   ├── SETUP.md                   # Local setup guide
│   ├── ARCHITECTURE.md            # System architecture
│   └── DEPLOYMENT_GUIDE.md        # Deployment instructions
├── src/
│   ├── config.py                  # Configuration settings
│   ├── data/                      # Data loading & preprocessing
│   │   ├── download.py            # Kaggle dataset download
│   │   ├── preprocess.py          # Image preprocessing
│   │   └── dataset.py             # PyTorch Dataset class
│   ├── models/                    # Model definitions & training
│   │   ├── cnn.py                 # SimpleCNN & ResNet models
│   │   └── train.py               # Training with MLflow
│   ├── api/                       # FastAPI application
│   │   ├── main.py                # API endpoints
│   │   ├── schemas.py             # Pydantic models
│   │   └── predict.py             # Inference logic
│   └── utils/                     # Utilities
│       └── metrics.py             # Evaluation metrics
├── tests/                         # Unit & integration tests
│   ├── test_preprocess.py
│   ├── test_inference.py
│   └── test_api.py
├── k8s/                           # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── monitoring/                # Prometheus/Grafana configs
├── scripts/
│   ├── train.py                   # Training entrypoint
│   ├── smoke_test.py              # Post-deployment tests
│   ├── simulate_traffic.py        # Load testing
│   └── setup_monitoring.sh        # Helm installation
├── Dockerfile                     # Multi-stage build
├── docker-compose.yml             # Full stack deployment
├── Makefile                       # Convenience commands
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
├── dvc.yaml                       # DVC pipeline
├── pytest.ini                     # Test configuration
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (recommended: 3.11)
- Docker and Docker Compose
- Minikube and kubectl (for K8s)
- Helm (for monitoring)

### 1. Clone and Setup

```bash
cd mlops2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Create sample data and train
python scripts/train.py --create-sample --epochs 5

# View MLflow experiments
mlflow ui --port 5000
```

### 3. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000

# Test health check
curl http://localhost:8000/health

# Test prediction (with image file)
curl -X POST http://localhost:8000/predict -F "file=@image.jpg"
```

### 4. Docker Deployment

```bash
# Build and run
docker build -t catdog-classifier .
docker run -p 8000:8000 catdog-classifier

# Or full stack with docker-compose
docker-compose up
```

---

## 📦 Installation

### Using pip

```bash
# Production
pip install -r requirements.txt

# Development (includes testing tools)
pip install -r requirements-dev.txt
```

### Using Make

```bash
make install       # Production dependencies
make install-dev   # Development dependencies
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| torch, torchvision | Deep learning framework |
| fastapi, uvicorn | REST API |
| mlflow | Experiment tracking |
| prometheus-client | Metrics collection |
| dvc | Data versioning |
| pytest | Testing |

---

## 🔧 Usage

### Training Models

```bash
# Train with sample data
python scripts/train.py --create-sample --epochs 5

# Train with full dataset (requires Kaggle credentials)
python -m src.data.download
python scripts/train.py --epochs 10 --lr 0.001

# Using Make
make train
```

### Making Predictions

#### Python API

```python
from src.api.predict import Predictor
from PIL import Image

predictor = Predictor(model_path="models/model.pt")
image = Image.open("cat.jpg")
result = predictor.predict(image)

print(result)
# {'prediction': 0, 'label': 'cat', 'confidence': 0.95, 
#  'probability_cat': 0.95, 'probability_dog': 0.05}
```

#### REST API

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

---

## 📡 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check for K8s probes |
| `/predict` | POST | Image classification |
| `/metrics` | GET | Prometheus metrics |
| `/model/info` | GET | Model metadata |
| `/docs` | GET | Swagger UI |

### Interactive Documentation

Once the API is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Response

```json
{
  "prediction": 1,
  "label": "dog",
  "confidence": 0.87,
  "probability_cat": 0.13,
  "probability_dog": 0.87
}
```

---

## 📊 MLflow Experiment Tracking

### Starting MLflow UI

```bash
mlflow ui --port 5000
```

Visit http://localhost:5000 to view experiments.

### Tracked Metrics

- Training/validation loss and accuracy
- Test accuracy, precision, recall, F1, ROC-AUC

### Logged Artifacts

- Confusion matrix plots
- Training curves
- ROC curves
- Model checkpoints

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t catdog-classifier .

# Run container
docker run -d -p 8000:8000 --name catdog-api catdog-classifier

# Test
curl http://localhost:8000/health
```

### Docker Compose (Full Stack)

```bash
docker-compose up -d
```

**Services:**
| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI application |
| mlflow | 5000 | Experiment tracking |
| prometheus | 9090 | Metrics server |
| grafana | 3000 | Dashboards |

---

## ☸️ Kubernetes Deployment (Minikube)

### Quick Start with Minikube

```bash
# Start Minikube
minikube start

# Use Minikube's Docker daemon
eval $(minikube docker-env)

# Build image
docker build -t catdog-classifier:latest .

# Deploy
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Port forward
kubectl -n catdog-classifier port-forward svc/catdog-classifier-service 8080:80
```

### Test Deployed API

```bash
curl http://localhost:8080/health
curl http://localhost:8080/metrics
```

---

## 📊 Monitoring with Prometheus & Grafana

### Install Prometheus & Grafana

```bash
./scripts/setup_monitoring.sh

# Access services
kubectl -n monitoring port-forward svc/prometheus-server 9090:80 &
kubectl -n monitoring port-forward svc/grafana 3000:80 &
```

**Access URLs:**
| Service | URL | Credentials |
|---------|-----|-------------|
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin123 |

### Custom Metrics

- `prediction_requests_total` - Total predictions by status/class
- `prediction_latency_seconds` - Request latency percentiles
- `prediction_confidence` - Model confidence distribution
- `model_loaded` - Model status gauge

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow (ci-cd.yml)

| Stage | Description |
|-------|-------------|
| **Lint** | Code quality checks (flake8, black, isort) |
| **Test** | Unit tests with pytest and coverage |
| **Train** | Model training with sample data |
| **Docker Build** | Build and test Docker image |
| **Integration Test** | End-to-end API testing |
| **Push Image** | Push to GitHub Container Registry (main branch only) |

### Triggers

- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual trigger via workflow_dispatch

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Test Categories

- `test_preprocess.py` - Image preprocessing tests
- `test_inference.py` - Model inference tests
- `test_api.py` - API endpoint tests

---

## 🤖 Model Information

### Architecture

**SimpleCNN:**
- 4 convolutional blocks (Conv + BatchNorm + ReLU + MaxPool)
- Adaptive average pooling
- 2 fully connected layers with dropout
- Binary classification output (sigmoid)

### Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~85-90% |
| P95 Latency | <100ms |

### Input Requirements

- Image format: JPEG, PNG
- Resized to: 224x224
- Normalized: ImageNet mean/std

---

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CI/CD Pipeline                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │
│  │  Lint   │→ │  Test   │→ │  Build  │→ │  Push to GHCR       │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ Data Loader  │ → │ Preprocessor │ → │ CNN Training         │ │
│  │ (Kaggle)     │   │ (PyTorch)    │   │ (MLflow Tracking)    │ │
│  └──────────────┘   └──────────────┘   └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Serving Layer                               │
│  ┌──────────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ FastAPI Endpoint │ ← │ Model Loader │ ← │ Saved Model     │  │
│  │ /predict         │   │              │   │ (model.pt)      │  │
│  └──────────────────┘   └──────────────┘   └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Kubernetes Deployment (Minikube)                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Namespace: catdog-classifier                                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │ │
│  │  │ Deployment  │→ │   Service   │→ │  HPA (Auto-scale)  │  │ │
│  │  │ (2 replicas)│  │ (NodePort)  │  │  (2-5 replicas)    │  │ │
│  │  └─────────────┘  └─────────────┘  └────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Monitoring Stack (Helm)                        │
│  ┌──────────────────┐              ┌──────────────────────────┐ │
│  │ Prometheus       │  ──scrape──▶ │ /metrics endpoint        │ │
│  │ (metrics store)  │              │ (prediction_requests,    │ │
│  └────────┬─────────┘              │  latency, confidence)    │ │
│           │                        └──────────────────────────┘ │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Grafana          │                                           │
│  │ (dashboards)     │                                           │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| ⭐ **[docs/SETUP.md](docs/SETUP.md)** | **Complete local setup guide (start here!)** |
| 📄 **[Final Report.docx](Final%20Report.docx)** | **Assignment report with screenshots** |
| [README.md](README.md) | Project overview |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture diagrams |
| [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) | Kubernetes & monitoring details |

---

## 📄 License

This project is licensed under the MIT License.

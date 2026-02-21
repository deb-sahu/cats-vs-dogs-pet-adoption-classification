# System Architecture

This document describes the architecture of the Cats vs Dogs MLOps pipeline.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CI/CD Pipeline (GitHub Actions)             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────────┐  │
│  │   Lint   │ → │   Test   │ → │  Build   │ → │  Push to GHCR   │  │
│  │ (flake8) │   │ (pytest) │   │ (Docker) │   │                 │  │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Training Pipeline                              │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐   │
│  │    Data      │   │  PyTorch     │   │    MLflow              │   │
│  │   Loader     │ → │  Training    │ → │    Tracking            │   │
│  │ (Dataset)    │   │  (CNN/ResNet)│   │ (params, metrics, art) │   │
│  └──────────────┘   └──────────────┘   └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Serving Layer                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     FastAPI Application                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐  │   │
│  │  │   /health   │  │  /predict   │  │     /metrics         │  │   │
│  │  │   (GET)     │  │   (POST)    │  │  (Prometheus)        │  │   │
│  │  └─────────────┘  └─────────────┘  └──────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Kubernetes Deployment                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Minikube Cluster                          │    │
│  │  ┌─────────────────┐  ┌─────────────────────────────────┐   │    │
│  │  │   Deployment    │  │         Monitoring               │   │    │
│  │  │  (2 replicas)   │  │  ┌───────────┐  ┌───────────┐   │   │    │
│  │  │  + Service      │  │  │Prometheus │  │  Grafana  │   │   │    │
│  │  │  + HPA          │  │  └───────────┘  └───────────┘   │   │    │
│  │  └─────────────────┘  └─────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                        Data Pipeline                          │
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │   Kaggle    │    │  Preprocess  │    │   PyTorch      │   │
│  │  Download   │ →  │  (224x224)   │ →  │   Dataset      │   │
│  │             │    │  + Normalize │    │   + DataLoader │   │
│  └─────────────┘    └──────────────┘    └────────────────┘   │
│         │                  │                     │            │
│         ▼                  ▼                     ▼            │
│    data/raw/         data/processed/       train/val/test    │
│    (DVC tracked)     (224x224 images)      (80%/10%/10%)     │
└──────────────────────────────────────────────────────────────┘
```

**Components:**
- **Download**: Kaggle API or manual download
- **Preprocess**: Resize to 224x224, RGB conversion
- **Augmentation**: Random flip, rotation, color jitter (training only)
- **Normalization**: ImageNet mean/std

### 2. Model Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      SimpleCNN Model                          │
│                                                               │
│  Input: (batch, 3, 224, 224)                                  │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────────────────────────────────┐            │
│  │  Feature Extractor                            │            │
│  │  Conv(3→32) → BN → ReLU → MaxPool            │            │
│  │  Conv(32→64) → BN → ReLU → MaxPool           │            │
│  │  Conv(64→128) → BN → ReLU → MaxPool          │            │
│  │  Conv(128→256) → BN → ReLU → MaxPool         │            │
│  └──────────────────────────────────────────────┘            │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────────────────────────────────┐            │
│  │  Classifier                                   │            │
│  │  AdaptiveAvgPool → Flatten                   │            │
│  │  Linear(256*7*7 → 512) → ReLU → Dropout      │            │
│  │  Linear(512 → 128) → ReLU → Dropout          │            │
│  │  Linear(128 → 1)                              │            │
│  └──────────────────────────────────────────────┘            │
│         │                                                     │
│         ▼                                                     │
│  Output: (batch, 1) → Sigmoid → [0, 1] probability           │
│          0 = cat, 1 = dog                                     │
└──────────────────────────────────────────────────────────────┘
```

### 3. API Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI Application                       │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    Middleware                           │  │
│  │  - CORS (Cross-Origin Resource Sharing)                 │  │
│  │  - Request Logging                                      │  │
│  │  - Prometheus Metrics Collection                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    Endpoints                            │  │
│  │                                                         │  │
│  │  GET  /           → API info                           │  │
│  │  GET  /health     → Health check + model status        │  │
│  │  POST /predict    → Image classification               │  │
│  │  GET  /metrics    → Prometheus metrics                 │  │
│  │  GET  /model/info → Model metadata                     │  │
│  │  GET  /docs       → Swagger UI                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    Predictor                            │  │
│  │  - Load model on startup                               │  │
│  │  - Preprocess uploaded images                          │  │
│  │  - Run inference with PyTorch                          │  │
│  │  - Return predictions with confidence                  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 4. CI/CD Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                   GitHub Actions Workflow                     │
│                                                               │
│  Trigger: Push to main/develop, PR to main                   │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    CI Pipeline (ci.yml)                  │ │
│  │                                                          │ │
│  │  ┌────────┐   ┌────────┐   ┌────────────────────────┐   │ │
│  │  │  Lint  │ → │  Test  │ → │  Build & Push Image    │   │ │
│  │  │ flake8 │   │ pytest │   │  to ghcr.io            │   │ │
│  │  │ black  │   │ + cov  │   │                        │   │ │
│  │  └────────┘   └────────┘   └────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    CD Pipeline (cd.yml)                  │ │
│  │                                                          │ │
│  │  ┌────────────────┐   ┌────────────────────────────┐    │ │
│  │  │ Deploy to K8s  │ → │  Run Smoke Tests           │    │ │
│  │  │ (main branch)  │   │  (health + prediction)     │    │ │
│  │  └────────────────┘   └────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 5. Monitoring Stack

```
┌──────────────────────────────────────────────────────────────┐
│                   Monitoring Architecture                     │
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                   │
│  │   API Pod    │ ──────► │  Prometheus  │                   │
│  │  /metrics    │  scrape │              │                   │
│  └──────────────┘         └──────────────┘                   │
│                                  │                            │
│                                  │ query                      │
│                                  ▼                            │
│                           ┌──────────────┐                   │
│                           │   Grafana    │                   │
│                           │  Dashboard   │                   │
│                           └──────────────┘                   │
│                                                               │
│  Metrics Collected:                                           │
│  - prediction_requests_total (by status, prediction)         │
│  - prediction_latency_seconds (histogram)                    │
│  - prediction_confidence (histogram)                         │
│  - model_loaded (gauge)                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Training Flow

```
1. Download dataset from Kaggle
2. Preprocess images (resize, normalize)
3. Split into train/val/test (80/10/10)
4. Apply augmentation to training set
5. Train CNN with MLflow tracking
6. Log metrics and artifacts
7. Save best model checkpoint
```

### Inference Flow

```
1. Client uploads image to /predict
2. API validates file type
3. Image preprocessed (resize, normalize)
4. Model inference (forward pass)
5. Sigmoid activation for probability
6. Return prediction, label, confidence
7. Log metrics to Prometheus
```

---

## Security Considerations

- **Non-root container**: API runs as non-privileged user
- **Read-only model**: Model file mounted as read-only
- **Input validation**: Pydantic schemas validate requests
- **CORS**: Configurable cross-origin settings
- **Health checks**: Kubernetes probes for availability

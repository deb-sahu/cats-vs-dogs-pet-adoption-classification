# Deployment & Monitoring Guide

This guide covers deploying the Cats vs Dogs Classifier API to local Kubernetes and setting up monitoring with Prometheus and Grafana.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Kubernetes Deployment](#2-local-kubernetes-deployment)
3. [Prometheus & Grafana Setup](#3-prometheus--grafana-setup)
4. [Prometheus Metrics](#4-prometheus-metrics)
5. [Creating Dashboards](#5-creating-dashboards)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

### Required Software

| Tool | Purpose | Installation |
|------|---------|--------------|
| Docker | Container runtime | [Install Docker](https://docs.docker.com/get-docker/) |
| kubectl | Kubernetes CLI | [Install kubectl](https://kubernetes.io/docs/tasks/tools/) |
| Minikube OR Docker Desktop | Local Kubernetes | See below |
| Helm | Kubernetes package manager | `brew install helm` |

### Option A: Minikube (Recommended for Learning)

```bash
# macOS
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Windows (PowerShell as Admin)
choco install minikube
```

### Option B: Docker Desktop Kubernetes

1. Open Docker Desktop
2. Go to Settings → Kubernetes
3. Check "Enable Kubernetes"
4. Click "Apply & Restart"

---

## 2. Local Kubernetes Deployment

### Quick Deploy (Automated)

```bash
# Make the script executable
chmod +x scripts/deploy_k8s.sh

# Deploy to Minikube
./scripts/deploy_k8s.sh minikube

# OR deploy to Docker Desktop
./scripts/deploy_k8s.sh docker-desktop
```

### Manual Deployment Steps

#### Step 1: Start Kubernetes Cluster

**Minikube:**
```bash
# Start with adequate resources
minikube start --driver=docker --memory=4096 --cpus=2

# Enable addons
minikube addons enable metrics-server
minikube addons enable ingress

# Use Minikube's Docker daemon (so images are accessible)
eval $(minikube docker-env)
```

**Docker Desktop:**
```bash
# Verify Kubernetes is running
kubectl cluster-info
```

#### Step 2: Build Docker Image

```bash
# Build the image (use minikube's Docker daemon)
docker build -t catdog-classifier:latest .

# Verify image was created
docker images | grep catdog-classifier
```

> **Important**: Make sure to run `eval $(minikube docker-env)` before building so the image is available in minikube's Docker registry.

#### Step 3: Deploy to Kubernetes

```bash
# Apply all Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

#### Step 4: Verify Deployment

```bash
# Check pods are running
kubectl get pods -n catdog-classifier

# Expected output:
# NAME                                 READY   STATUS    RESTARTS   AGE
# catdog-classifier-xxxxxxxxxx-xxxxx   1/1     Running   0          1m
# catdog-classifier-xxxxxxxxxx-xxxxx   1/1     Running   0          1m

# Check services
kubectl get svc -n catdog-classifier

# Check deployment status
kubectl describe deployment catdog-classifier -n catdog-classifier
```

#### Step 5: Access the API

**Minikube:**
```bash
# Get service URL
minikube service catdog-classifier-service -n catdog-classifier --url

# OR use port-forward
kubectl port-forward svc/catdog-classifier-service 8080:80 -n catdog-classifier

# Access at http://localhost:8080
```

**Docker Desktop:**
```bash
# Use port-forward
kubectl port-forward svc/catdog-classifier-service 8080:80 -n catdog-classifier

# Access at http://localhost:8080
```

#### Step 6: Test the API

**Available Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check - returns model status |
| `/predict` | POST | Make cat/dog prediction (upload image) |
| `/metrics` | GET | Prometheus metrics for monitoring |
| `/docs` | GET | Interactive Swagger UI documentation |
| `/model/info` | GET | Model information |

**Test Commands:**

```bash
# 1. Health check
curl http://localhost:8080/health
# Response: {"status":"healthy","model_loaded":true,"version":"1.0.0"}

# 2. Make a prediction (upload an image)
curl -X POST http://localhost:8080/predict \
  -F "file=@path/to/cat_or_dog.jpg"
# Response: {"prediction":0,"label":"cat","confidence":0.85,"probability_cat":0.85,"probability_dog":0.15}

# 3. Check Prometheus metrics
curl http://localhost:8080/metrics | grep prediction
# Shows: prediction_requests_total, prediction_latency_seconds

# 4. Get model info
curl http://localhost:8080/model/info

# 5. Open Swagger UI in browser
open http://localhost:8080/docs
```

---

## 3. Prometheus & Grafana Setup

We deploy Prometheus and Grafana locally in Minikube using Helm charts. This provides a **completely free** monitoring solution.

### Quick Setup (Automated)

```bash
# Run the setup script
./scripts/setup_monitoring.sh
```

### Manual Setup Steps

#### Step 1: Add Helm Repositories

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

#### Step 2: Deploy Prometheus

```bash
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace \
  -f k8s/monitoring/prometheus-values.yaml
```

#### Step 3: Deploy Grafana

```bash
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set persistence.enabled=false \
  --set adminPassword=admin123 \
  -f k8s/monitoring/grafana-values.yaml
```

#### Step 4: Wait for Pods

```bash
kubectl -n monitoring wait --for=condition=ready pod --all --timeout=120s
kubectl -n monitoring get pods
```

Expected output:
```
NAME                                            READY   STATUS    RESTARTS   AGE
grafana-xxxxxxxxxx-xxxxx                        1/1     Running   0          1m
prometheus-server-xxxxxxxxxx-xxxxx              2/2     Running   0          1m
```

#### Step 5: Access the Services

```bash
# Port forward Prometheus (in background)
kubectl -n monitoring port-forward svc/prometheus-server 9090:80 &

# Port forward Grafana (in background)
kubectl -n monitoring port-forward svc/grafana 3000:80 &
```

**Access URLs:**

| Service | URL | Credentials |
|---------|-----|-------------|
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | `admin` / `admin123` |

#### Step 6: Configure Grafana Data Source

1. Open http://localhost:3000
2. Login with `admin` / `admin123`
3. Go to **Connections** → **Data Sources** → **Add data source**
4. Select **Prometheus**
5. Set URL to: `http://prometheus-server.monitoring.svc.cluster.local`
6. Click **Save & Test**

#### Step 7: Import Dashboard

1. Go to **Dashboards** → **Import**
2. Upload `k8s/monitoring/grafana-dashboard.json`
3. Select your Prometheus data source
4. Click **Import**

---

## 4. Prometheus Metrics

The API exposes custom metrics at `/metrics`:

### Prediction Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_requests_total` | Counter | Total predictions by status/class |
| `prediction_latency_seconds` | Histogram | Prediction request latency |
| `prediction_confidence` | Histogram | Prediction confidence distribution |
| `model_loaded` | Gauge | Whether model is loaded (1/0) |

### Example Prometheus Queries

```promql
# Prediction rate (per minute)
rate(prediction_requests_total[1m])

# P95 latency
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))

# Predictions by class
sum(prediction_requests_total) by (prediction)

# Model uptime
model_loaded
```

---

## 5. Creating Dashboards

### Pre-built Dashboard

Import `k8s/monitoring/grafana-dashboard.json` which includes:
- Predictions per minute (by class)
- P95 latency gauge
- Model status indicator
- Error rate panel

### Custom Panels

**Predictions Over Time:**
```promql
sum(rate(prediction_requests_total[5m])) by (prediction)
```

**Average Latency:**
```promql
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))
```

---

## 6. Troubleshooting

### Pod not starting

```bash
# Check pod status
kubectl describe pod -l app=catdog-classifier -n catdog-classifier

# Check logs
kubectl logs -l app=catdog-classifier -n catdog-classifier
```

### Image not found

```bash
# For Minikube, ensure you're using Minikube's Docker
eval $(minikube docker-env)
docker build -t catdog-classifier:latest .

# Verify imagePullPolicy is "IfNotPresent" in deployment.yaml
```

### Service not accessible

```bash
# Check service endpoints
kubectl get endpoints -n catdog-classifier

# Use port-forward as fallback
kubectl port-forward svc/catdog-classifier-service 8080:80 -n catdog-classifier
```

### Metrics not appearing in Grafana

```bash
# Verify metrics endpoint works
kubectl port-forward svc/catdog-classifier-service 8080:80 -n catdog-classifier
curl http://localhost:8080/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job, health}'
```

### Cleanup

```bash
# Delete the application
kubectl delete namespace catdog-classifier

# Delete monitoring
kubectl delete namespace monitoring

# Stop Minikube
minikube stop

# Delete Minikube cluster completely
minikube delete
```

---

## Quick Reference

### Commands

```bash
# Full deployment
./scripts/deploy_k8s.sh minikube

# Check status
kubectl get all -n catdog-classifier

# View logs
kubectl logs -f deployment/catdog-classifier -n catdog-classifier

# Scale
kubectl scale deployment catdog-classifier --replicas=3 -n catdog-classifier

# Delete
kubectl delete namespace catdog-classifier
```

### URLs (after port-forward)

| Service | URL |
|---------|-----|
| API | http://localhost:8080 |
| Health Check | http://localhost:8080/health |
| API Docs | http://localhost:8080/docs |
| Metrics | http://localhost:8080/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

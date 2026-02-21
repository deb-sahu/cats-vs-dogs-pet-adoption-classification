# Deployment Guide

This guide covers deploying the Cats vs Dogs classifier to various environments.

---

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [Kubernetes Deployment (Minikube)](#kubernetes-deployment-minikube)
3. [Monitoring Setup](#monitoring-setup)
4. [Production Considerations](#production-considerations)

---

## Docker Deployment

### Building the Image

```bash
# Standard build
docker build -t catdog-classifier:latest .

# With build arguments
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t catdog-classifier:latest .

# Multi-platform build
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t catdog-classifier:latest .
```

### Running the Container

```bash
# Basic run
docker run -d -p 8000:8000 --name catdog-api catdog-classifier:latest

# With model volume mount
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  --name catdog-api \
  catdog-classifier:latest

# With environment variables
docker run -d -p 8000:8000 \
  -e MODEL_PATH=/app/models/model.pt \
  -e LOG_LEVEL=DEBUG \
  --name catdog-api \
  catdog-classifier:latest
```

### Docker Compose

For local development with full stack:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale API
docker-compose up -d --scale api=3

# Cleanup
docker-compose down -v
```

**Services in docker-compose.yml:**
| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI application |
| mlflow | 5000 | Experiment tracking UI |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Dashboards |

---

## Kubernetes Deployment (Minikube)

### Prerequisites

```bash
# Install Minikube (macOS)
brew install minikube kubectl helm

# Start Minikube
minikube start --memory=4096 --cpus=2

# Enable addons
minikube addons enable metrics-server
minikube addons enable ingress
```

### Build Image in Minikube

```bash
# Point Docker to Minikube's daemon
eval $(minikube docker-env)

# Build image
docker build -t catdog-classifier:latest .

# Verify
docker images | grep catdog
```

### Deploy Application

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy configuration
kubectl apply -f k8s/configmap.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Optional: Enable autoscaling
kubectl apply -f k8s/hpa.yaml
```

### Verify Deployment

```bash
# Check pods
kubectl -n catdog-classifier get pods

# Check service
kubectl -n catdog-classifier get svc

# View logs
kubectl -n catdog-classifier logs -l app=catdog-classifier -f

# Describe deployment
kubectl -n catdog-classifier describe deployment catdog-classifier
```

### Access the API

```bash
# Option 1: Port forward
kubectl -n catdog-classifier port-forward svc/catdog-classifier-service 8080:80

# Option 2: Minikube service
minikube -n catdog-classifier service catdog-classifier-service

# Option 3: NodePort (if configured)
minikube ip  # Get Minikube IP
# Access at http://<minikube-ip>:30080
```

### Kubernetes Manifests Explained

**deployment.yaml:**
- 2 replicas for high availability
- Resource limits (512Mi memory, 500m CPU)
- Liveness/readiness probes on /health
- Rolling update strategy

**service.yaml:**
- NodePort type for Minikube access
- Exposes port 80 → container port 8000

**hpa.yaml:**
- Scales 2-5 replicas based on CPU/memory
- Target 70% CPU utilization

---

## Monitoring Setup

### Install Prometheus & Grafana

```bash
# Run the setup script
./scripts/setup_monitoring.sh

# Or manually:
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace \
  -f k8s/monitoring/prometheus-values.yaml

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword=admin123 \
  -f k8s/monitoring/grafana-values.yaml
```

### Access Monitoring

```bash
# Prometheus
kubectl -n monitoring port-forward svc/prometheus-server 9090:80 &

# Grafana
kubectl -n monitoring port-forward svc/grafana 3000:80 &
```

### Configure Grafana

1. Open http://localhost:3000
2. Login with admin / admin123
3. Add data source:
   - Type: Prometheus
   - URL: http://prometheus-server.monitoring.svc.cluster.local
4. Import dashboard:
   - Upload `k8s/monitoring/grafana-dashboard.json`

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_requests_total` | Counter | Total predictions by status/class |
| `prediction_latency_seconds` | Histogram | Request latency percentiles |
| `prediction_confidence` | Histogram | Model confidence distribution |
| `model_loaded` | Gauge | 1 if model loaded, 0 otherwise |

### Example Prometheus Queries

```promql
# Request rate (last 5 min)
sum(rate(prediction_requests_total[5m]))

# P95 latency
histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))

# Error rate
sum(rate(prediction_requests_total{status="error"}[5m])) / sum(rate(prediction_requests_total[5m]))

# Predictions per class
sum(prediction_requests_total) by (prediction)
```

---

## Production Considerations

### Security

1. **Use secrets for sensitive data:**
   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: api-secrets
   data:
     MODEL_PATH: <base64-encoded-path>
   ```

2. **Network policies:**
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: api-network-policy
   spec:
     podSelector:
       matchLabels:
         app: catdog-classifier
     ingress:
       - from:
           - namespaceSelector:
               matchLabels:
                 name: ingress
   ```

3. **Pod security:**
   ```yaml
   securityContext:
     runAsNonRoot: true
     runAsUser: 1000
     readOnlyRootFilesystem: true
   ```

### Scaling

1. **Horizontal Pod Autoscaler** (included in k8s/hpa.yaml)
2. **Cluster autoscaler** for node scaling
3. **Consider GPU nodes** for inference optimization

### Reliability

1. **Pod Disruption Budget:**
   ```yaml
   apiVersion: policy/v1
   kind: PodDisruptionBudget
   spec:
     minAvailable: 1
     selector:
       matchLabels:
         app: catdog-classifier
   ```

2. **Resource quotas** per namespace
3. **Health checks** with appropriate timeouts

### Observability

1. Enable structured JSON logging
2. Use distributed tracing (Jaeger/Zipkin)
3. Set up alerting in Grafana
4. Implement SLOs/SLIs

---

## Troubleshooting

### Pods Not Starting

```bash
# Check events
kubectl -n catdog-classifier get events --sort-by=.metadata.creationTimestamp

# Check pod status
kubectl -n catdog-classifier describe pod <pod-name>

# Check image pull
kubectl -n catdog-classifier get pods -o jsonpath='{.items[*].status.containerStatuses[*].state}'
```

### Model Not Loading

```bash
# Check logs
kubectl -n catdog-classifier logs -l app=catdog-classifier | grep -i model

# Verify model path
kubectl -n catdog-classifier exec -it <pod-name> -- ls -la /app/models/
```

### High Latency

```bash
# Check resource usage
kubectl -n catdog-classifier top pods

# Review Prometheus metrics
curl http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,prediction_latency_seconds_bucket)
```

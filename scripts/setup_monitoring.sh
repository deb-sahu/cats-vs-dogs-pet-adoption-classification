#!/bin/bash
# Setup Prometheus and Grafana monitoring in Minikube using Helm

set -e

echo "=========================================="
echo "Setting up Monitoring Stack in Minikube"
echo "=========================================="

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed"
    exit 1
fi

# Check if helm is available
if ! command -v helm &> /dev/null; then
    echo "Error: helm is not installed"
    echo "Install with: brew install helm"
    exit 1
fi

# Check if minikube is running
if ! minikube status &> /dev/null; then
    echo "Error: Minikube is not running"
    echo "Start with: minikube start"
    exit 1
fi

echo ""
echo "1. Adding Helm repositories..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

echo ""
echo "2. Creating monitoring namespace..."
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

echo ""
echo "3. Installing Prometheus..."
helm upgrade --install prometheus prometheus-community/prometheus \
    --namespace monitoring \
    --set server.persistentVolume.enabled=false \
    --set alertmanager.persistentVolume.enabled=false \
    --set server.service.type=NodePort \
    --set server.service.nodePort=30090 \
    -f k8s/monitoring/prometheus-values.yaml \
    --wait

echo ""
echo "4. Installing Grafana..."
helm upgrade --install grafana grafana/grafana \
    --namespace monitoring \
    --set persistence.enabled=false \
    --set adminPassword=admin123 \
    --set service.type=NodePort \
    --set service.nodePort=30030 \
    -f k8s/monitoring/grafana-values.yaml \
    --wait

echo ""
echo "5. Waiting for pods to be ready..."
kubectl -n monitoring wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus --timeout=120s || true
kubectl -n monitoring wait --for=condition=ready pod -l app.kubernetes.io/name=grafana --timeout=120s || true

echo ""
echo "=========================================="
echo "Monitoring Stack Setup Complete!"
echo "=========================================="
echo ""
echo "Access URLs (via port-forward):"
echo "  Prometheus: kubectl -n monitoring port-forward svc/prometheus-server 9090:80"
echo "  Grafana:    kubectl -n monitoring port-forward svc/grafana 3000:80"
echo ""
echo "Or via Minikube service:"
echo "  Prometheus: minikube service prometheus-server -n monitoring"
echo "  Grafana:    minikube service grafana -n monitoring"
echo ""
echo "Grafana Credentials:"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "Next steps:"
echo "  1. Open Grafana in browser"
echo "  2. Add Prometheus data source: http://prometheus-server.monitoring.svc.cluster.local"
echo "  3. Import the dashboard from k8s/monitoring/grafana-dashboard.json"
echo ""

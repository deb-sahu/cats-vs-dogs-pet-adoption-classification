#!/bin/bash
# Deploy Cats vs Dogs Classifier to Local Kubernetes
# Usage: ./scripts/deploy_k8s.sh [minikube|docker-desktop]

set -e

DRIVER="${1:-minikube}"
NAMESPACE="catdog-classifier"
IMAGE_NAME="catdog-classifier:latest"

echo "=========================================="
echo "Deploying Cats vs Dogs Classifier"
echo "Platform: $DRIVER"
echo "=========================================="

# Check prerequisites
check_prerequisites() {
    echo ""
    echo "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        echo "❌ kubectl not found. Install with: brew install kubectl"
        exit 1
    fi
    echo "✓ kubectl found"
    
    if ! command -v docker &> /dev/null; then
        echo "❌ docker not found. Install Docker Desktop or run: brew install --cask docker"
        exit 1
    fi
    echo "✓ docker found"
    
    if [ "$DRIVER" == "minikube" ]; then
        if ! command -v minikube &> /dev/null; then
            echo "❌ minikube not found. Install with: brew install minikube"
            exit 1
        fi
        echo "✓ minikube found"
    fi
}

# Start Kubernetes cluster
start_cluster() {
    echo ""
    echo "Starting Kubernetes cluster..."
    
    if [ "$DRIVER" == "minikube" ]; then
        # Check if minikube is running
        if ! minikube status &> /dev/null; then
            echo "Starting Minikube..."
            minikube start --driver=docker --memory=4096 --cpus=2
        else
            echo "Minikube is already running"
        fi
        
        # Enable addons
        minikube addons enable metrics-server
        minikube addons enable ingress
        
        # Point to Minikube's Docker daemon
        echo ""
        echo "Configuring Docker to use Minikube's daemon..."
        eval $(minikube docker-env)
        
    else
        # Docker Desktop - verify Kubernetes is enabled
        if ! kubectl cluster-info &> /dev/null; then
            echo "❌ Kubernetes not running. Enable it in Docker Desktop settings."
            exit 1
        fi
        echo "Docker Desktop Kubernetes is running"
    fi
}

# Build Docker image
build_image() {
    echo ""
    echo "Building Docker image..."
    
    # For Minikube, ensure we're using its Docker daemon
    if [ "$DRIVER" == "minikube" ]; then
        eval $(minikube docker-env)
    fi
    
    docker build -t $IMAGE_NAME .
    
    echo "✓ Image built: $IMAGE_NAME"
    docker images | grep catdog-classifier
}

# Deploy to Kubernetes
deploy_app() {
    echo ""
    echo "Deploying to Kubernetes..."
    
    # Apply manifests in order
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml
    
    echo "✓ Manifests applied"
}

# Wait for deployment
wait_for_deployment() {
    echo ""
    echo "Waiting for pods to be ready..."
    
    kubectl -n $NAMESPACE wait --for=condition=ready pod -l app=catdog-classifier --timeout=120s
    
    echo ""
    echo "Deployment status:"
    kubectl -n $NAMESPACE get pods
    kubectl -n $NAMESPACE get svc
}

# Run smoke tests
run_smoke_tests() {
    echo ""
    echo "Running smoke tests..."
    
    # Start port-forward in background
    kubectl -n $NAMESPACE port-forward svc/catdog-classifier-service 8080:80 &
    PF_PID=$!
    sleep 5
    
    # Run smoke test
    if [ -f "scripts/smoke_test.py" ]; then
        python scripts/smoke_test.py --url "http://localhost:8080" || true
    else
        # Basic health check
        curl -s http://localhost:8080/health | python -m json.tool
    fi
    
    # Kill port-forward
    kill $PF_PID 2>/dev/null || true
}

# Print access info
print_access_info() {
    echo ""
    echo "=========================================="
    echo "✓ Deployment Complete!"
    echo "=========================================="
    echo ""
    echo "Access the API:"
    echo ""
    
    if [ "$DRIVER" == "minikube" ]; then
        echo "  Option 1 - Minikube service:"
        echo "    minikube -n $NAMESPACE service catdog-classifier-service"
        echo ""
        echo "  Option 2 - Port forward:"
        echo "    kubectl -n $NAMESPACE port-forward svc/catdog-classifier-service 8080:80"
        echo "    curl http://localhost:8080/health"
    else
        echo "  Port forward:"
        echo "    kubectl -n $NAMESPACE port-forward svc/catdog-classifier-service 8080:80"
        echo "    curl http://localhost:8080/health"
    fi
    
    echo ""
    echo "API Endpoints:"
    echo "  - Health:  http://localhost:8080/health"
    echo "  - Predict: http://localhost:8080/predict (POST with image)"
    echo "  - Metrics: http://localhost:8080/metrics"
    echo "  - Docs:    http://localhost:8080/docs"
    echo ""
    echo "Next steps:"
    echo "  1. Setup monitoring: ./scripts/setup_monitoring.sh"
    echo "  2. Simulate traffic: python scripts/simulate_traffic.py --url http://localhost:8080"
    echo ""
}

# Main
main() {
    check_prerequisites
    start_cluster
    build_image
    deploy_app
    wait_for_deployment
    run_smoke_tests
    print_access_info
}

main

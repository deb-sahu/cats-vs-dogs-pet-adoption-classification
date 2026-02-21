"""Tests for FastAPI application."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import io
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
import torch


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    from src.models.cnn import SimpleCNN
    return SimpleCNN()


@pytest.fixture
def client(mock_model, tmp_path):
    """Create test client with mocked model."""
    model_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": mock_model.state_dict()}, model_path)
    
    import os
    os.environ["MODEL_PATH"] = str(model_path)
    
    from src.api.main import app
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_no_model():
    """Create test client without model."""
    import os
    os.environ["MODEL_PATH"] = "nonexistent_model.pt"
    
    from importlib import reload
    import src.api.main
    reload(src.api.main)
    
    with TestClient(src.api.main.app) as client:
        yield client


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_api_info(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "version" in data
    
    def test_health_check_format(self, client):
        """Test health check response format."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["version"], str)


class TestPredictEndpoint:
    """Tests for prediction endpoint."""
    
    def create_test_image(self) -> bytes:
        """Create a test image as bytes."""
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer.getvalue()
    
    def test_predict_success(self, client):
        """Test successful prediction."""
        image_bytes = self.create_test_image()
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "label" in data
        assert "confidence" in data
        assert data["prediction"] in [0, 1]
        assert data["label"] in ["cat", "dog"]
    
    def test_predict_response_structure(self, client):
        """Test prediction response has all required fields."""
        image_bytes = self.create_test_image()
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        required_fields = ["prediction", "label", "confidence", "probability_cat", "probability_dog"]
        for field in required_fields:
            assert field in data
    
    def test_predict_probabilities_sum_to_one(self, client):
        """Test that probabilities sum to approximately 1."""
        image_bytes = self.create_test_image()
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )
        
        data = response.json()
        total_prob = data["probability_cat"] + data["probability_dog"]
        assert abs(total_prob - 1.0) < 0.01
    
    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code == 400
    
    def test_predict_no_file(self, client):
        """Test prediction without file."""
        response = client.post("/predict")
        
        assert response.status_code == 422
    
    def test_predict_png_image(self, client):
        """Test prediction with PNG image."""
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        response = client.post(
            "/predict",
            files={"file": ("test.png", buffer.getvalue(), "image/png")}
        )
        
        assert response.status_code == 200


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""
    
    def test_metrics_returns_prometheus_format(self, client):
        """Test that metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"] or \
               "text/plain" in response.headers.get("content-type", "")
    
    def test_metrics_contains_custom_metrics(self, client):
        """Test that custom metrics are present."""
        client.post(
            "/predict",
            files={"file": ("test.jpg", self.create_test_image(), "image/jpeg")}
        )
        
        response = client.get("/metrics")
        content = response.text
        
        assert "prediction_requests_total" in content or "model_loaded" in content
    
    def create_test_image(self) -> bytes:
        """Create a test image as bytes."""
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer.getvalue()


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""
    
    def test_model_info_returns_details(self, client):
        """Test that model info endpoint returns model details."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "input_size" in data
        assert "classes" in data

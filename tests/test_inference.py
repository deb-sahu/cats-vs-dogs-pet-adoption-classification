"""Tests for model inference functions."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import io
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn import SimpleCNN, get_model, count_parameters
from src.api.predict import Predictor


class TestSimpleCNN:
    """Tests for SimpleCNN model."""
    
    def test_model_forward_pass(self):
        """Test that model produces output of correct shape."""
        model = SimpleCNN()
        x = torch.randn(1, 3, 224, 224)
        
        output = model(x)
        
        assert output.shape == (1, 1)
    
    def test_model_batch_forward(self):
        """Test model with batch input."""
        model = SimpleCNN()
        x = torch.randn(8, 3, 224, 224)
        
        output = model(x)
        
        assert output.shape == (8, 1)
    
    def test_model_predict_proba(self):
        """Test that predict_proba returns values in [0, 1]."""
        model = SimpleCNN()
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            probs = model.predict_proba(x)
        
        assert probs.shape == (4, 1)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0
    
    def test_model_predict_labels(self):
        """Test that predict returns binary labels."""
        model = SimpleCNN()
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            labels = model.predict(x)
        
        assert labels.shape == (4, 1)
        assert set(labels.flatten().tolist()).issubset({0, 1})
    
    def test_model_with_dropout(self):
        """Test that dropout is applied during training."""
        model = SimpleCNN(dropout=0.5)
        model.train()
        x = torch.randn(4, 3, 224, 224)
        
        out1 = model(x)
        out2 = model(x)
        
        assert not torch.allclose(out1, out2)
    
    def test_model_deterministic_eval(self):
        """Test that model is deterministic in eval mode."""
        model = SimpleCNN()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        assert torch.allclose(out1, out2)


class TestGetModel:
    """Tests for model factory function."""
    
    def test_get_simple_cnn(self):
        """Test getting SimpleCNN model."""
        model = get_model("simple_cnn")
        
        assert isinstance(model, SimpleCNN)
    
    def test_get_model_invalid_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            get_model("invalid_model")


class TestCountParameters:
    """Tests for count_parameters function."""
    
    def test_count_parameters_simple_cnn(self):
        """Test counting parameters for SimpleCNN."""
        model = SimpleCNN()
        
        total, trainable = count_parameters(model)
        
        assert total > 0
        assert trainable == total
        assert total > 1_000_000


class TestPredictor:
    """Tests for Predictor class."""
    
    def test_predictor_initialization(self):
        """Test predictor initialization without model."""
        predictor = Predictor()
        
        assert predictor.model is None
        assert not predictor.is_loaded()
    
    def test_predictor_preprocess_image(self):
        """Test image preprocessing."""
        predictor = Predictor()
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        
        tensor = predictor.preprocess_image(img)
        
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32
    
    def test_predictor_preprocess_grayscale(self):
        """Test preprocessing grayscale image."""
        predictor = Predictor()
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            mode="L"
        )
        
        tensor = predictor.preprocess_image(img)
        
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_predictor_predict_without_model(self):
        """Test that prediction without model raises error."""
        predictor = Predictor()
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            predictor.predict(img)
    
    def test_predictor_predict_with_model(self, tmp_path):
        """Test prediction with loaded model."""
        model = SimpleCNN()
        model_path = tmp_path / "model.pt"
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        
        predictor = Predictor(model_path=str(model_path))
        
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        result = predictor.predict(img)
        
        assert "prediction" in result
        assert "label" in result
        assert "confidence" in result
        assert result["prediction"] in [0, 1]
        assert result["label"] in ["cat", "dog"]
        assert 0 <= result["confidence"] <= 1
    
    def test_predictor_predict_from_bytes(self, tmp_path, sample_image_bytes):
        """Test prediction from image bytes."""
        model = SimpleCNN()
        model_path = tmp_path / "model.pt"
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        
        predictor = Predictor(model_path=str(model_path))
        result = predictor.predict_from_bytes(sample_image_bytes)
        
        assert "prediction" in result
        assert "label" in result
    
    def test_predictor_result_structure(self, tmp_path):
        """Test that prediction result has all required fields."""
        model = SimpleCNN()
        model_path = tmp_path / "model.pt"
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        
        predictor = Predictor(model_path=str(model_path))
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        result = predictor.predict(img)
        
        expected_keys = {"prediction", "label", "confidence", "probability_cat", "probability_dog"}
        assert set(result.keys()) == expected_keys
        
        assert abs(result["probability_cat"] + result["probability_dog"] - 1.0) < 0.001


class TestPredictorEdgeCases:
    """Edge case tests for Predictor."""
    
    def test_predictor_load_invalid_model(self, tmp_path):
        """Test loading invalid model file."""
        invalid_path = tmp_path / "invalid.pt"
        invalid_path.write_text("not a valid model")
        
        predictor = Predictor()
        success = predictor.load_model(str(invalid_path))
        
        assert not success
        assert not predictor.is_loaded()
    
    def test_predictor_load_nonexistent_model(self):
        """Test loading non-existent model file."""
        predictor = Predictor()
        success = predictor.load_model("nonexistent_model.pt")
        
        assert not success
        assert not predictor.is_loaded()

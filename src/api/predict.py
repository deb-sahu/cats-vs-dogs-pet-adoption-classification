"""Inference logic for the API."""

import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.cnn import SimpleCNN, get_model


class Predictor:
    """
    Model predictor for inference.
    
    Handles model loading and image prediction.
    """
    
    CLASSES = ["cat", "dog"]
    INPUT_SIZE = (224, 224)
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.model: Optional[torch.nn.Module] = None
        self.model_path = model_path
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.transform = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            True if model loaded successfully
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model = SimpleCNN()
            
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            self.model_path = model_path
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor ready for model
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        tensor = self.transform(image)
        return tensor.unsqueeze(0)
    
    def predict(self, image: Image.Image) -> dict:
        """
        Make prediction on an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob_dog = torch.sigmoid(output).item()
        
        prob_cat = 1.0 - prob_dog
        prediction = 1 if prob_dog >= 0.5 else 0
        label = self.CLASSES[prediction]
        confidence = prob_dog if prediction == 1 else prob_cat
        
        return {
            "prediction": prediction,
            "label": label,
            "confidence": round(confidence, 4),
            "probability_cat": round(prob_cat, 4),
            "probability_dog": round(prob_dog, 4),
        }
    
    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """
        Make prediction from image bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with prediction results
        """
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict(image)
    
    def predict_from_path(self, image_path: str) -> dict:
        """
        Make prediction from image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        image = Image.open(image_path)
        return self.predict(image)


_predictor: Optional[Predictor] = None


def get_predictor(model_path: str = None) -> Predictor:
    """
    Get or create global predictor instance.
    
    Args:
        model_path: Path to model checkpoint (only used on first call)
        
    Returns:
        Predictor instance
    """
    global _predictor
    
    if _predictor is None:
        if model_path is None:
            default_paths = [
                "models/model.pt",
                "/app/models/model.pt",
                Path(__file__).parent.parent.parent / "models" / "model.pt",
            ]
            for path in default_paths:
                if Path(path).exists():
                    model_path = str(path)
                    break
        
        _predictor = Predictor(model_path=model_path)
    
    return _predictor

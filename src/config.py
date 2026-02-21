"""Configuration settings for the Cats vs Dogs classifier."""

from pathlib import Path
from typing import Tuple
import os


class Config:
    """Application configuration."""
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    MODEL_PATH = os.environ.get("MODEL_PATH", str(MODELS_DIR / "model.pt"))
    
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 10
    DROPOUT: float = 0.5
    
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    CLASSES = ["cat", "dog"]
    NUM_CLASSES = 2
    
    MLFLOW_EXPERIMENT_NAME = "catdog-classifier"
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    
    API_HOST = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT = int(os.environ.get("API_PORT", "8000"))
    API_VERSION = "1.0.0"
    
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    
    KAGGLE_DATASET = "bhavikjikadara/dog-and-cat-classification-dataset"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.ARTIFACTS_DIR,
            cls.LOGS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


config = Config()

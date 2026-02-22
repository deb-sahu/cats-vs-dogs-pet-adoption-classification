"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def sample_image_path(tmp_path_factory):
    """Create a sample image file for testing."""
    tmp_dir = tmp_path_factory.mktemp("images")
    img_path = tmp_dir / "test_image.jpg"

    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(img_path)

    return img_path


@pytest.fixture(scope="session")
def sample_dataset(tmp_path_factory):
    """Create a minimal sample dataset for testing."""
    data_dir = tmp_path_factory.mktemp("dataset")

    for split in ["train", "val", "test"]:
        for cls in ["cats", "dogs"]:
            cls_dir = data_dir / split / cls
            cls_dir.mkdir(parents=True)

            n_images = 10 if split == "train" else 3
            for i in range(n_images):
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(cls_dir / f"{cls}_{i}.jpg")

    return data_dir


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for API testing."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    import io

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer.getvalue()


@pytest.fixture(scope="session")
def trained_model(sample_dataset):
    """Train a minimal model for testing."""
    import torch

    from src.models.cnn import SimpleCNN

    model = SimpleCNN()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "val_loss": 0.5,
            "val_accuracy": 0.75,
        },
        sample_dataset.parent / "model.pt",
    )

    return model, sample_dataset.parent / "model.pt"

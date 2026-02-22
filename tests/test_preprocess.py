"""Tests for data preprocessing functions."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    normalize_for_model,
    preprocess_image,
    preprocess_image_bytes,
)


class TestPreprocessImage:
    """Tests for preprocess_image function."""

    def test_preprocess_image_returns_correct_shape(self, sample_image_path):
        """Test that preprocessed image has correct shape."""
        result = preprocess_image(sample_image_path)

        assert result.shape == (224, 224, 3)

    def test_preprocess_image_normalizes_values(self, sample_image_path):
        """Test that pixel values are normalized to [0, 1]."""
        result = preprocess_image(sample_image_path, normalize=True)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_image_custom_size(self, sample_image_path):
        """Test preprocessing with custom target size."""
        result = preprocess_image(sample_image_path, target_size=(128, 128))

        assert result.shape == (128, 128, 3)

    def test_preprocess_image_no_normalization(self, sample_image_path):
        """Test preprocessing without normalization."""
        result = preprocess_image(sample_image_path, normalize=False)

        assert result.max() <= 255.0

    def test_preprocess_image_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            preprocess_image("nonexistent_image.jpg")

    def test_preprocess_image_converts_grayscale(self, tmp_path):
        """Test that grayscale images are converted to RGB."""
        gray_img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode="L")
        gray_path = tmp_path / "gray.png"
        gray_img.save(gray_path)

        result = preprocess_image(gray_path)

        assert result.shape == (224, 224, 3)

    def test_preprocess_image_handles_rgba(self, tmp_path):
        """Test that RGBA images are converted to RGB."""
        rgba_img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8), mode="RGBA"
        )
        rgba_path = tmp_path / "rgba.png"
        rgba_img.save(rgba_path)

        result = preprocess_image(rgba_path)

        assert result.shape == (224, 224, 3)


class TestPreprocessImageBytes:
    """Tests for preprocess_image_bytes function."""

    def test_preprocess_bytes_returns_correct_shape(self, sample_image_bytes):
        """Test that preprocessing from bytes returns correct shape."""
        result = preprocess_image_bytes(sample_image_bytes)

        assert result.shape == (224, 224, 3)

    def test_preprocess_bytes_normalizes_values(self, sample_image_bytes):
        """Test that pixel values are normalized."""
        result = preprocess_image_bytes(sample_image_bytes, normalize=True)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_bytes_custom_size(self, sample_image_bytes):
        """Test preprocessing bytes with custom size."""
        result = preprocess_image_bytes(sample_image_bytes, target_size=(128, 128))

        assert result.shape == (128, 128, 3)


class TestNormalizeForModel:
    """Tests for normalize_for_model function."""

    def test_normalize_uses_imagenet_defaults(self):
        """Test that ImageNet normalization is applied by default."""
        img = np.random.rand(224, 224, 3).astype(np.float32)

        result = normalize_for_model(img)

        expected_mean = np.array(IMAGENET_MEAN)
        expected_std = np.array(IMAGENET_STD)

        expected = (img - expected_mean) / expected_std
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_custom_mean_std(self):
        """Test normalization with custom mean and std."""
        img = np.ones((224, 224, 3), dtype=np.float32) * 0.5
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        result = normalize_for_model(img, mean=mean, std=std)

        expected = np.zeros((224, 224, 3), dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_output_dtype(self):
        """Test that output maintains float32 dtype."""
        img = np.random.rand(224, 224, 3).astype(np.float32)

        result = normalize_for_model(img)

        assert result.dtype == np.float32


class TestEdgeCases:
    """Tests for edge cases in preprocessing."""

    def test_very_small_image(self, tmp_path):
        """Test preprocessing a very small image."""
        small_img = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        small_path = tmp_path / "small.jpg"
        small_img.save(small_path)

        result = preprocess_image(small_path)

        assert result.shape == (224, 224, 3)

    def test_very_large_image(self, tmp_path):
        """Test preprocessing a large image."""
        large_img = Image.fromarray(np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8))
        large_path = tmp_path / "large.jpg"
        large_img.save(large_path)

        result = preprocess_image(large_path)

        assert result.shape == (224, 224, 3)

    def test_non_square_image(self, tmp_path):
        """Test preprocessing a non-square image."""
        rect_img = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
        rect_path = tmp_path / "rect.jpg"
        rect_img.save(rect_path)

        result = preprocess_image(rect_path)

        assert result.shape == (224, 224, 3)

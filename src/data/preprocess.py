"""Image preprocessing utilities for the Cats vs Dogs classifier."""

from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image


def preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> np.ndarray:
    """
    Load and preprocess a single image for CNN input.
    
    Args:
        image_path: Path to the image file
        target_size: Target (height, width) for resizing
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image as numpy array with shape (H, W, C)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    try:
        img = Image.open(path)
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    
    img_array = np.array(img, dtype=np.float32)
    
    if normalize:
        img_array = img_array / 255.0
    
    return img_array


def preprocess_image_bytes(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess image from bytes (e.g., from file upload).
    
    Args:
        image_bytes: Raw image bytes
        target_size: Target (height, width) for resizing
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image as numpy array with shape (H, W, C)
    """
    import io
    
    img = Image.open(io.BytesIO(image_bytes))
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    
    img_array = np.array(img, dtype=np.float32)
    
    if normalize:
        img_array = img_array / 255.0
    
    return img_array


def preprocess_dataset(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    target_size: Tuple[int, int] = (224, 224),
) -> dict:
    """
    Preprocess all images in a dataset directory.
    
    Args:
        input_dir: Directory containing images organized by class
        output_dir: Optional directory to save preprocessed images
        target_size: Target (height, width) for resizing
        
    Returns:
        Dictionary with processing statistics
    """
    from tqdm import tqdm
    
    input_path = Path(input_dir)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {"processed": 0, "failed": 0, "classes": {}}
    
    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        stats["classes"][class_name] = {"processed": 0, "failed": 0}
        
        if output_dir:
            (output_path / class_name).mkdir(parents=True, exist_ok=True)
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        
        for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
            try:
                img_array = preprocess_image(img_file, target_size, normalize=False)
                
                if output_dir:
                    output_file = output_path / class_name / img_file.name
                    Image.fromarray(img_array.astype(np.uint8)).save(output_file)
                
                stats["processed"] += 1
                stats["classes"][class_name]["processed"] += 1
            except Exception as e:
                print(f"Failed to process {img_file}: {e}")
                stats["failed"] += 1
                stats["classes"][class_name]["failed"] += 1
    
    return stats


def get_image_stats(data_dir: Union[str, Path]) -> dict:
    """
    Compute statistics of images in a dataset directory.
    
    Args:
        data_dir: Directory containing images
        
    Returns:
        Dictionary with image statistics (mean, std per channel)
    """
    from tqdm import tqdm
    
    data_path = Path(data_dir)
    
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count = 0
    
    image_files = list(data_path.rglob("*.jpg")) + list(data_path.rglob("*.jpeg")) + list(data_path.rglob("*.png"))
    
    for img_file in tqdm(image_files, desc="Computing stats"):
        try:
            img_array = preprocess_image(img_file, normalize=True)
            pixel_sum += img_array.mean(axis=(0, 1))
            pixel_sq_sum += (img_array ** 2).mean(axis=(0, 1))
            count += 1
        except Exception:
            continue
    
    if count == 0:
        return {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    
    mean = pixel_sum / count
    std = np.sqrt(pixel_sq_sum / count - mean ** 2)
    
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_images": count
    }


# ImageNet normalization constants (commonly used for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_for_model(img_array: np.ndarray, mean: list = None, std: list = None) -> np.ndarray:
    """
    Normalize image array for model input using channel-wise mean and std.
    
    Args:
        img_array: Image array with shape (H, W, C), values in [0, 1]
        mean: Per-channel mean values
        std: Per-channel std values
        
    Returns:
        Normalized image array
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    return (img_array - mean) / std

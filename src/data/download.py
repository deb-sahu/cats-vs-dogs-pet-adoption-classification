"""Dataset download utilities for Kaggle Cats vs Dogs dataset."""

import os
import zipfile
import shutil
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


def download_from_kaggle(
    dataset: str = "bhavikjikadara/dog-and-cat-classification-dataset",
    output_dir: str = "data/raw",
) -> Path:
    """
    Download dataset from Kaggle using kaggle API.
    
    Requires kaggle.json credentials in ~/.kaggle/kaggle.json
    
    Args:
        dataset: Kaggle dataset identifier
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded data directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset, path=str(output_path), unzip=True)
        print(f"Dataset downloaded successfully to {output_path}")
        return output_path
    except Exception as e:
        print(f"Kaggle API download failed: {e}")
        print("Please download manually from: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset")
        raise


def organize_dataset(raw_dir: str = "data/raw", output_dir: str = "data/processed") -> dict:
    """
    Organize the downloaded dataset into train/val/test splits.
    
    The Kaggle dataset has structure:
    - dataset/training_set/cats/*.jpg
    - dataset/training_set/dogs/*.jpg
    - dataset/test_set/cats/*.jpg
    - dataset/test_set/dogs/*.jpg
    
    Args:
        raw_dir: Directory containing raw downloaded data
        output_dir: Directory for organized data
        
    Returns:
        Dictionary with counts for each split
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    splits = ["train", "val", "test"]
    classes = ["cats", "dogs"]
    
    for split in splits:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    counts = {split: {cls: 0 for cls in classes} for split in splits}
    
    training_cats = list((raw_path / "dataset" / "training_set" / "cats").glob("*.jpg"))
    training_dogs = list((raw_path / "dataset" / "training_set" / "dogs").glob("*.jpg"))
    test_cats = list((raw_path / "dataset" / "test_set" / "cats").glob("*.jpg"))
    test_dogs = list((raw_path / "dataset" / "test_set" / "dogs").glob("*.jpg"))
    
    if not training_cats:
        training_cats = list((raw_path / "dataset" / "training_set" / "training_set" / "cats").glob("*.jpg"))
        training_dogs = list((raw_path / "dataset" / "training_set" / "training_set" / "dogs").glob("*.jpg"))
        test_cats = list((raw_path / "dataset" / "test_set" / "test_set" / "cats").glob("*.jpg"))
        test_dogs = list((raw_path / "dataset" / "test_set" / "test_set" / "dogs").glob("*.jpg"))
    
    def split_and_copy(files: list, class_name: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Split files into train/val/test and copy to respective directories."""
        import random
        random.seed(42)
        random.shuffle(files)
        
        n = len(files)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits_data = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }
        
        for split, split_files in splits_data.items():
            for f in split_files:
                dest = output_path / split / class_name / f.name
                shutil.copy2(f, dest)
                counts[split][class_name] += 1
    
    print("Organizing cats images...")
    all_cats = training_cats + test_cats
    split_and_copy(all_cats, "cats")
    
    print("Organizing dogs images...")
    all_dogs = training_dogs + test_dogs
    split_and_copy(all_dogs, "dogs")
    
    print("\nDataset organization complete:")
    for split in splits:
        total = sum(counts[split].values())
        print(f"  {split}: {total} images (cats: {counts[split]['cats']}, dogs: {counts[split]['dogs']})")
    
    return counts


def create_sample_dataset(output_dir: str = "data/processed", n_samples: int = 100) -> dict:
    """
    Create a small sample dataset for testing without downloading full dataset.
    
    Creates random noise images for quick testing of the pipeline.
    
    Args:
        output_dir: Directory for the sample data
        n_samples: Number of samples per class per split
        
    Returns:
        Dictionary with counts
    """
    from PIL import Image
    import numpy as np
    
    output_path = Path(output_dir)
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    classes = ["cats", "dogs"]
    
    for split in splits:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    counts = {split: {cls: 0 for cls in classes} for split in splits}
    
    np.random.seed(42)
    
    for cls_idx, cls in enumerate(classes):
        for split, ratio in splits.items():
            n = int(n_samples * ratio)
            for i in range(n):
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                if cls == "cats":
                    img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 30, 0, 255)
                else:
                    img_array[:, :, 2] = np.clip(img_array[:, :, 2] + 30, 0, 255)
                
                img = Image.fromarray(img_array)
                img.save(output_path / split / cls / f"{cls}_{split}_{i:04d}.jpg")
                counts[split][cls] += 1
    
    print("Sample dataset created:")
    for split in splits:
        total = sum(counts[split].values())
        print(f"  {split}: {total} images")
    
    return counts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and organize Cats vs Dogs dataset")
    parser.add_argument("--sample", action="store_true", help="Create sample dataset for testing")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples for sample dataset")
    args = parser.parse_args()
    
    if args.sample:
        create_sample_dataset(n_samples=args.n_samples)
    else:
        download_from_kaggle()
        organize_dataset()

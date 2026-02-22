"""PyTorch Dataset for Cats vs Dogs classification."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CatDogDataset(Dataset):
    """
    PyTorch Dataset for Cats vs Dogs binary classification.

    Attributes:
        root_dir: Root directory containing class subdirectories
        transform: Optional transforms to apply to images
        class_to_idx: Mapping from class name to index
        samples: List of (image_path, label) tuples
    """

    CLASSES = ["cats", "dogs"]

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize the dataset.

        Args:
            root_dir: Root directory with 'cats' and 'dogs' subdirectories
            transform: Optional torchvision transforms
            target_size: Image size (height, width)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

        if self.transform is None:
            self.transform = self._default_transform()

    def _load_samples(self) -> None:
        """Load all image paths and labels."""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, class_idx))

    def _default_transform(self) -> transforms.Compose:
        """Default image transform for inference."""
        return transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_name(self, idx: int) -> str:
        """Get class name from index."""
        return self.idx_to_class.get(idx, "unknown")


def get_train_transforms(target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get training transforms with data augmentation.

    Args:
        target_size: Target image size (height, width)

    Returns:
        Composed transforms for training
    """
    return transforms.Compose(
        [
            transforms.Resize((int(target_size[0] * 1.1), int(target_size[1] * 1.1))),
            transforms.RandomCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms(target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        target_size: Target image size (height, width)

    Returns:
        Composed transforms for validation/testing
    """
    return transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        data_dir: Root data directory with train/val/test subdirectories
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        target_size: Target image size

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_path = Path(data_dir)

    train_dataset = CatDogDataset(data_path / "train", transform=get_train_transforms(target_size))

    val_dataset = CatDogDataset(data_path / "val", transform=get_val_transforms(target_size))

    test_dataset = CatDogDataset(data_path / "test", transform=get_val_transforms(target_size))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader

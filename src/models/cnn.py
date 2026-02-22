"""Simple CNN model for Cats vs Dogs classification."""

from typing import Tuple

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for binary image classification.

    Architecture:
    - 3 convolutional blocks (conv + relu + maxpool)
    - 2 fully connected layers with dropout
    - Sigmoid output for binary classification

    Input: (batch_size, 3, 224, 224)
    Output: (batch_size, 1) - probability of being a dog
    """

    def __init__(self, dropout: float = 0.5):
        """
        Initialize the CNN.

        Args:
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x: Input tensor

        Returns:
            Probability tensor in range [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            x: Input tensor
            threshold: Classification threshold

        Returns:
            Predicted labels (0 = cat, 1 = dog)
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).long()


class ResNetTransfer(nn.Module):
    """
    Transfer learning model using pretrained ResNet18.

    Uses ResNet18 as feature extractor with custom classifier head.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        """
        Initialize the transfer learning model.

        Args:
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze the backbone weights
        """
        super().__init__()

        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict class labels."""
        probs = self.predict_proba(x)
        return (probs >= threshold).long()


def get_model(model_type: str = "simple_cnn", **kwargs) -> nn.Module:
    """
    Factory function to get a model by type.

    Args:
        model_type: One of "simple_cnn" or "resnet_transfer"
        **kwargs: Additional arguments passed to model constructor

    Returns:
        PyTorch model
    """
    models = {
        "simple_cnn": SimpleCNN,
        "resnet_transfer": ResNetTransfer,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](**kwargs)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

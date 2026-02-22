"""Training module with MLflow experiment tracking."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.cnn import count_parameters, get_model
from src.utils.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)


class Trainer:
    """
    Model trainer with MLflow integration.

    Handles training loop, validation, checkpointing, and experiment tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader = None,
        learning_rate: float = 0.001,
        device: str = None,
        experiment_name: str = "catdog-classifier",
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            learning_rate: Learning rate for optimizer
            device: Device to train on (auto-detected if None)
            experiment_name: MLflow experiment name
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5
        )

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        self.best_val_loss = float("inf")

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(
        self, loader: DataLoader = None
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate the model.

        Args:
            loader: DataLoader to validate on (uses val_loader if None)

        Returns:
            Tuple of (loss, accuracy, y_true, y_pred, y_prob)
        """
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(self.device)
            labels_tensor = labels.float().unsqueeze(1).to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels_tensor)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

        avg_loss = total_loss / len(loader.dataset)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        accuracy = (y_true == y_pred).mean()

        return avg_loss, accuracy, y_true, y_pred, y_prob

    def train(
        self,
        epochs: int = 10,
        save_dir: str = "models",
        artifacts_dir: str = "artifacts",
    ) -> Dict:
        """
        Full training loop with MLflow tracking.

        Args:
            epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            artifacts_dir: Directory to save training artifacts

        Returns:
            Dictionary with training results
        """
        save_path = Path(save_dir)
        artifacts_path = Path(artifacts_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        artifacts_path.mkdir(parents=True, exist_ok=True)

        total_params, trainable_params = count_parameters(self.model)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "learning_rate": self.learning_rate,
                    "epochs": epochs,
                    "batch_size": self.train_loader.batch_size,
                    "optimizer": "Adam",
                    "model_type": self.model.__class__.__name__,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "device": str(self.device),
                }
            )

            for epoch in range(1, epochs + 1):
                print(f"\nEpoch {epoch}/{epochs}")
                print("-" * 40)

                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc, _, _, _ = self.validate()

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accs.append(train_acc)
                self.val_accs.append(val_acc)

                self.scheduler.step(val_loss)

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )

                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    model_path = save_path / "model.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                            "val_accuracy": val_acc,
                        },
                        model_path,
                    )
                    print(f"Saved best model to {model_path}")

            test_loader = self.test_loader or self.val_loader
            _, test_acc, y_true, y_pred, y_prob = self.validate(test_loader)

            metrics = compute_metrics(y_true, y_pred, y_prob)
            print("\nTest Results:")
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")

            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

            cm_path = artifacts_path / "confusion_matrix.png"
            plot_confusion_matrix(y_true, y_pred, save_path=str(cm_path))
            mlflow.log_artifact(str(cm_path))

            curves_path = artifacts_path / "training_curves.png"
            plot_training_curves(
                self.train_losses,
                self.val_losses,
                self.train_accs,
                self.val_accs,
                save_path=str(curves_path),
            )
            mlflow.log_artifact(str(curves_path))

            if len(np.unique(y_true)) > 1:
                roc_path = artifacts_path / "roc_curve.png"
                plot_roc_curve(y_true, y_prob, save_path=str(roc_path))
                mlflow.log_artifact(str(roc_path))

            metrics_path = artifacts_path / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "test_metrics": metrics,
                        "best_val_loss": self.best_val_loss,
                        "final_train_loss": self.train_losses[-1],
                        "final_val_loss": self.val_losses[-1],
                    },
                    f,
                    indent=2,
                )
            mlflow.log_artifact(str(metrics_path))

            mlflow.pytorch.log_model(self.model, "model")

            run_id = mlflow.active_run().info.run_id
            print(f"\nMLflow run ID: {run_id}")

        return {
            "test_metrics": metrics,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "run_id": run_id,
        }


def train_model(
    data_dir: str = "data/processed",
    model_type: str = "simple_cnn",
    epochs: int = 10,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    save_dir: str = "models",
    artifacts_dir: str = "artifacts",
) -> Dict:
    """
    High-level function to train a model.

    Args:
        data_dir: Directory containing train/val/test splits
        model_type: Model type ("simple_cnn" or "resnet_transfer")
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        save_dir: Directory to save model
        artifacts_dir: Directory to save artifacts

    Returns:
        Dictionary with training results
    """
    print(f"Loading data from {data_dir}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, batch_size=batch_size, num_workers=0
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print(f"\nCreating {model_type} model...")
    model = get_model(model_type)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=learning_rate,
    )

    print(f"\nStarting training for {epochs} epochs...")
    results = trainer.train(epochs=epochs, save_dir=save_dir, artifacts_dir=artifacts_dir)

    return results

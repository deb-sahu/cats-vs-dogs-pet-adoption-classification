#!/usr/bin/env python
"""Training script entry point."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train import train_model
from src.data.download import create_sample_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs classifier")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing train/val/test data",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="simple_cnn",
        choices=["simple_cnn", "resnet_transfer"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=0.001,
        dest="learning_rate",
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample dataset if data doesn't exist",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per class for sample dataset",
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        if args.create_sample:
            print("Data not found. Creating sample dataset...")
            create_sample_dataset(output_dir=args.data_dir, n_samples=args.n_samples)
        else:
            print(f"Error: Data directory {args.data_dir} does not exist or is empty.")
            print("Run with --create-sample to create a sample dataset for testing.")
            sys.exit(1)
    
    results = train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Test metrics:")
    for metric, value in results['test_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nMLflow run ID: {results['run_id']}")
    print("View experiments: mlflow ui --port 5000")


if __name__ == "__main__":
    main()

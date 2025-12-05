#!/usr/bin/env python3
"""
Evaluation script for SITS-Former.

This script provides evaluation capabilities for trained SITS-Former models.

Usage:
    python -m sitsformer.scripts.evaluate --checkpoint checkpoints/best_model.pt --config configs/default.yaml
    python -m sitsformer.scripts.evaluate --checkpoint checkpoints/best_model.pt --test-data /path/to/test/data
"""

import argparse
from pathlib import Path

import torch

from ..data import DummySatelliteDataset, create_dataloader
from ..evaluation import ModelEvaluator
from ..models import create_sits_former
from ..utils import (
    get_device,
    get_system_info,
    load_config,
    set_seed,
    setup_logging,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SITS-Former model")

    # Model and checkpoint
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")

    # Data arguments
    parser.add_argument("--test-data", type=str, help="Path to test dataset")
    parser.add_argument(
        "--dummy-data", action="store_true", help="Use dummy dataset for testing"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples for dummy dataset",
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu/auto)")

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save model predictions"
    )

    return parser.parse_args()


def load_model_and_config(checkpoint_path, config_path=None):
    """Load model and configuration from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Use weights_only=True to prevent arbitrary code execution via pickle
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Try to get config from checkpoint first
    if "config" in checkpoint and config_path is None:
        config = checkpoint["config"]
        print("Using configuration from checkpoint")
    elif config_path:
        config = load_config(Path(config_path))
        print(f"Using configuration from: {config_path}")
    else:
        raise ValueError(
            "No configuration found. Please provide --config or use checkpoint with saved config."
        )

    # Create model
    model = create_sits_former(config["model"])

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded successfully")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "best_val_loss" in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

    return model, config


def create_test_dataset(config, args):
    """Create test dataset."""
    if args.dummy_data or not args.test_data:
        print("Using dummy dataset for evaluation...")

        test_dataset = DummySatelliteDataset(
            num_samples=args.num_samples,
            sequence_length=config["data"]["sequence_length"],
            image_size=config["data"]["image_size"],
            num_channels=config["model"]["in_channels"],
            num_classes=config["model"]["num_classes"],
        )
    else:
        # Load real test dataset (placeholder)
        raise NotImplementedError(
            "Real dataset loading not implemented yet. Use --dummy-data flag."
        )

    return test_dataset


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, "evaluation")

    logger.info("Starting SITS-Former evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {output_dir}")

    # Log system information
    system_info = get_system_info()
    logger.info(f"System: {system_info['platform']}")
    logger.info(f"PyTorch: {system_info['pytorch_version']}")

    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load model and configuration
    model, config = load_model_and_config(args.checkpoint, args.config)
    model.to(device)
    model.eval()

    # Set random seed for reproducibility
    set_seed(config.get("experiment", {}).get("seed", 42))

    # Create test dataset
    logger.info("Creating test dataset...")
    test_dataset = create_test_dataset(config, args)
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create data loader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create evaluator
    evaluator = ModelEvaluator(model, device)

    # Get class names
    class_names = config.get("evaluation", {}).get("class_names")
    if not class_names:
        class_names = [f"Class_{i}" for i in range(config["model"]["num_classes"])]

    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate(test_loader, class_names=class_names)

    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"  Number of samples: {results['num_samples']}")
    logger.info(f"  Number of classes: {results['num_classes']}")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  Precision (macro): {results['precision_macro']:.4f}")
    logger.info(f"  Recall (macro): {results['recall_macro']:.4f}")
    logger.info(f"  F1-score (macro): {results['f1_macro']:.4f}")

    if results["roc_auc"] is not None:
        logger.info(f"  ROC AUC: {results['roc_auc']:.4f}")

    # Per-class results
    logger.info("\\nPer-class results:")
    for i, class_name in enumerate(class_names):
        if i < len(results["precision_per_class"]):
            precision = results["precision_per_class"][i]
            recall = results["recall_per_class"][i]
            f1 = results["f1_per_class"][i]
            support = results["support_per_class"][i]

            logger.info(
                f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}"
            )

    # Save results
    evaluator.save_results(results, output_dir / "results.json")
    logger.info("Results saved to results.json")

    # Get predictions for visualization
    predictions, probabilities, true_labels = evaluator.predict(test_loader)

    # Save predictions if requested
    if args.save_predictions:
        import numpy as np

        np.save(output_dir / "predictions.npy", predictions)
        np.save(output_dir / "probabilities.npy", probabilities)
        np.save(output_dir / "true_labels.npy", true_labels)
        logger.info("Predictions saved")

    # Create visualizations
    logger.info("Creating visualizations...")

    # Confusion matrix
    fig = evaluator.plot_confusion_matrix(
        true_labels,
        predictions,
        class_names,
        save_path=output_dir / "confusion_matrix.png",
    )
    logger.info("Confusion matrix saved")

    # Per-class performance
    fig = evaluator.plot_class_performance(
        results, save_path=output_dir / "class_performance.png"
    )
    logger.info("Class performance plot saved")

    # Calculate additional metrics
    from sklearn.metrics import classification_report

    report = classification_report(
        true_labels, predictions, target_names=class_names, digits=4
    )

    # Save detailed report
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write("SITS-Former Evaluation Report\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"Checkpoint: {args.checkpoint}\\n")
        f.write(f"Test samples: {len(test_dataset)}\\n")
        f.write(f"Device: {device}\\n\\n")
        f.write("Classification Report:\\n")
        f.write("-" * 30 + "\\n")
        f.write(report)
        f.write("\\n\\nConfusion Matrix:\\n")
        f.write("-" * 20 + "\\n")
        f.write(str(results["confusion_matrix"]))

    logger.info("Detailed report saved to classification_report.txt")
    logger.info(f"Evaluation completed. All results saved to: {output_dir}")


if __name__ == "__main__":
    main()

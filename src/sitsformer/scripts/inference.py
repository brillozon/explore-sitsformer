#!/usr/bin/env python3
"""
Inference script for SITS-Former.

This script provides inference capabilities for trained SITS-Former models
on new satellite image time series data.

Usage:
    python -m sitsformer.scripts.inference --checkpoint checkpoints/best_model.pt --input /path/to/data
    python -m sitsformer.scripts.inference --checkpoint checkpoints/best_model.pt --dummy-data
"""

import argparse
import json
import time
from pathlib import Path

import torch

from ..data import DummySatelliteDataset, create_dataloader
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
    parser = argparse.ArgumentParser(description="Run inference with SITS-Former model")

    # Model and checkpoint
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")

    # Input data
    parser.add_argument("--input", type=str, help="Path to input data")
    parser.add_argument(
        "--dummy-data", action="store_true", help="Use dummy dataset for testing"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for dummy dataset",
    )

    # Processing arguments
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu/auto)")

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv", "numpy"],
        default="json",
        help="Output format for predictions",
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

    return model, config


def create_inference_dataset(config, args):
    """Create dataset for inference."""
    if args.dummy_data or not args.input:
        print("Using dummy dataset for inference...")

        dataset = DummySatelliteDataset(
            num_samples=args.num_samples,
            sequence_length=config["data"]["sequence_length"],
            image_size=config["data"]["image_size"],
            num_channels=config["model"]["in_channels"],
            num_classes=config["model"]["num_classes"],
        )
    else:
        # Load real dataset (placeholder)
        raise NotImplementedError(
            "Real dataset loading not implemented yet. Use --dummy-data flag."
        )

    return dataset


def run_inference(model, data_loader, device):
    """Run inference on the dataset."""
    model.eval()
    predictions = []
    probabilities = []
    sample_ids = []

    total_samples = 0
    start_time = time.time()

    print(f"Running inference on {len(data_loader)} batches...")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Store results
            predictions.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
            sample_ids.extend([total_samples + i for i in range(len(preds))])

            total_samples += len(preds)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f}s")
    print(f"Average time per sample: {inference_time / total_samples:.4f}s")

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "sample_ids": sample_ids,
        "total_samples": total_samples,
        "inference_time": inference_time,
    }


def save_results(results, output_dir, format_type, config):
    """Save inference results in specified format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get class names
    class_names = config.get("evaluation", {}).get("class_names")
    if not class_names:
        class_names = [f"Class_{i}" for i in range(config["model"]["num_classes"])]

    if format_type == "json":
        # Save as JSON
        output_data = {
            "metadata": {
                "total_samples": results["total_samples"],
                "inference_time": results["inference_time"],
                "class_names": class_names,
                "num_classes": len(class_names),
            },
            "predictions": [
                {
                    "sample_id": sample_id,
                    "predicted_class": pred,
                    "predicted_class_name": (
                        class_names[pred]
                        if pred < len(class_names)
                        else f"Unknown_{pred}"
                    ),
                    "confidence": max(probs),
                    "probabilities": {
                        class_names[i] if i < len(class_names) else f"Class_{i}": prob
                        for i, prob in enumerate(probs)
                    },
                }
                for sample_id, pred, probs in zip(
                    results["sample_ids"],
                    results["predictions"],
                    results["probabilities"],
                )
            ],
        }

        with open(output_dir / "predictions.json", "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {output_dir / 'predictions.json'}")

    elif format_type == "csv":
        # Save as CSV
        import pandas as pd

        data = []
        for sample_id, pred, probs in zip(
            results["sample_ids"], results["predictions"], results["probabilities"]
        ):
            row = {
                "sample_id": sample_id,
                "predicted_class": pred,
                "predicted_class_name": (
                    class_names[pred] if pred < len(class_names) else f"Unknown_{pred}"
                ),
                "confidence": max(probs),
            }

            # Add probability columns
            for i, prob in enumerate(probs):
                class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
                row[f"prob_{class_name}"] = prob

            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_dir / "predictions.csv", index=False)

        print(f"Results saved to {output_dir / 'predictions.csv'}")

    elif format_type == "numpy":
        # Save as NumPy arrays
        import numpy as np

        np.save(output_dir / "predictions.npy", results["predictions"])
        np.save(output_dir / "probabilities.npy", results["probabilities"])
        np.save(output_dir / "sample_ids.npy", results["sample_ids"])

        # Save metadata
        metadata = {
            "total_samples": results["total_samples"],
            "inference_time": results["inference_time"],
            "class_names": class_names,
            "num_classes": len(class_names),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Results saved to {output_dir}/ as numpy arrays")

    # Always save a summary
    summary = {
        "total_samples": results["total_samples"],
        "inference_time": results["inference_time"],
        "avg_time_per_sample": results["inference_time"] / results["total_samples"],
        "class_distribution": {},
    }

    # Calculate class distribution
    for pred in results["predictions"]:
        class_name = class_names[pred] if pred < len(class_names) else f"Class_{pred}"
        summary["class_distribution"][class_name] = (
            summary["class_distribution"].get(class_name, 0) + 1
        )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {output_dir / 'summary.json'}")


def main():
    """Main inference function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, "inference")

    logger.info("Starting SITS-Former inference")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output format: {args.output_format}")

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

    # Set random seed for reproducibility (if using dummy data)
    set_seed(config.get("experiment", {}).get("seed", 42))

    # Create inference dataset
    logger.info("Creating inference dataset...")
    dataset = create_inference_dataset(config, args)
    logger.info(f"Inference samples: {len(dataset)}")

    # Create data loader
    data_loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Run inference
    logger.info("Running inference...")
    results = run_inference(model, data_loader, device)

    # Save results
    logger.info(f"Saving results in {args.output_format} format...")
    save_results(results, output_dir, args.output_format, config)

    # Print summary
    logger.info("Inference Summary:")
    logger.info(f"  Total samples: {results['total_samples']}")
    logger.info(f"  Inference time: {results['inference_time']:.2f}s")
    logger.info(
        f"  Average time per sample: "
        f"{results['inference_time'] / results['total_samples']:.4f}s"
    )

    # Class distribution
    class_names = config.get("evaluation", {}).get(
        "class_names", [f"Class_{i}" for i in range(config["model"]["num_classes"])]
    )
    class_counts = {}
    for pred in results["predictions"]:
        class_name = class_names[pred] if pred < len(class_names) else f"Class_{pred}"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    logger.info("\\nClass distribution:")
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / results["total_samples"]) * 100
        logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")

    logger.info(f"\\nInference completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

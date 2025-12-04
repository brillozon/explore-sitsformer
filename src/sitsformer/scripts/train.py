#!/usr/bin/env python3
"""
Main training script for SITS-Former.

This script provides a command-line interface for training the SITS-Former model
with various configurations and datasets.

Usage:
    python -m sitsformer.scripts.train --config configs/default.yaml
    python -m sitsformer.scripts.train --config configs/small_model.yaml --data-dir /path/to/data
    python -m sitsformer.scripts.train --preset small --epochs 50
"""

import argparse
from pathlib import Path

import torch
import wandb

from sitsformer.data import DummySatelliteDataset, create_dataloader
from sitsformer.evaluation import ModelEvaluator
from sitsformer.models import create_sits_former
from sitsformer.training import (
    SITSFormerTrainer,
    create_criterion,
    create_optimizer,
    create_scheduler,
)
from sitsformer.utils import (
    backup_code,
    create_checkpoint_dir,
    create_experiment_name,
    get_config_summary,
    get_device,
    get_preset_config,
    get_system_info,
    load_config,
    merge_configs,
    set_seed,
    setup_logging,
    validate_config,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SITS-Former model")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["small", "medium", "large"],
        help="Use predefined configuration preset",
    )

    # Data arguments
    parser.add_argument("--data-dir", type=str, help="Path to dataset directory")
    parser.add_argument(
        "--dummy-data", action="store_true", help="Use dummy dataset for testing"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples for dummy dataset",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu/auto)")

    # Experiment arguments
    parser.add_argument("--experiment-name", type=str, help="Name of the experiment")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to save logs"
    )

    # Debugging
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run with only a few batches for testing",
    )

    # Logging
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="sits-former",
        help="Weights & Biases project name",
    )

    return parser.parse_args()


def load_and_merge_config(args):
    """Load and merge configuration from various sources."""
    # Start with default configuration
    if args.preset:
        config = get_preset_config(args.preset)
    elif args.config:
        config = load_config(Path(args.config))
    else:
        config = get_preset_config("small")  # Default to small for testing

    # Override with command line arguments
    overrides = {}

    if args.data_dir:
        overrides.setdefault("data", {})["data_dir"] = args.data_dir

    if args.epochs:
        overrides.setdefault("training", {})["epochs"] = args.epochs

    if args.batch_size:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size

    if args.learning_rate:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate

    if args.device:
        overrides.setdefault("hardware", {})["device"] = args.device

    if args.wandb:
        overrides.setdefault("logging", {})["use_wandb"] = True

    if args.wandb_project:
        overrides.setdefault("logging", {})["project_name"] = args.wandb_project

    if args.experiment_name:
        overrides.setdefault("logging", {})["experiment_name"] = args.experiment_name

    if args.debug:
        overrides.setdefault("experiment", {})["debug"] = True

    if args.fast_dev_run:
        overrides.setdefault("experiment", {})["fast_dev_run"] = True
        overrides.setdefault("training", {})["epochs"] = 1

    # Merge configurations
    if overrides:
        config = merge_configs(config, overrides)

    # Validate configuration
    config = validate_config(config)

    return config


def create_datasets(config, args):
    """Create training and validation datasets."""
    if args.dummy_data or not config.get("data", {}).get("data_dir"):
        print("Using dummy dataset for training...")

        # Create dummy dataset
        full_dataset = DummySatelliteDataset(
            num_samples=args.num_samples,
            sequence_length=config["data"]["sequence_length"],
            image_size=config["data"]["image_size"],
            num_channels=config["model"]["in_channels"],
            num_classes=config["model"]["num_classes"],
        )

        # Split dataset
        from torch.utils.data import random_split

        data_config = config["data"]
        train_size = int(len(full_dataset) * data_config["train_ratio"])
        val_size = int(len(full_dataset) * data_config["val_ratio"])
        test_size = len(full_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(data_config["random_seed"]),
        )

    else:
        # Load real dataset (placeholder)
        raise NotImplementedError(
            "Real dataset loading not implemented yet. Use --dummy-data flag."
        )

    return train_dataset, val_dataset, test_dataset


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_and_merge_config(args)

    # Set up experiment
    experiment_name = args.experiment_name or create_experiment_name(config)

    # Set random seed
    set_seed(config.get("experiment", {}).get("seed", 42))

    # Setup logging
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir, experiment_name)

    # Print configuration summary
    logger.info("Starting SITS-Former training")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(get_config_summary(config))

    # Log system information
    system_info = get_system_info()
    logger.info(f"System: {system_info['platform']}")
    logger.info(f"Python: {system_info['python_version']}")
    logger.info(f"PyTorch: {system_info['pytorch_version']}")
    if "gpu_names" in system_info:
        logger.info(f"GPUs: {system_info['gpu_names']}")

    # Get device
    device = get_device(config.get("hardware", {}).get("device"))
    logger.info(f"Using device: {device}")

    # Create checkpoint directory
    save_dir = create_checkpoint_dir(Path(args.save_dir), experiment_name)
    logger.info(f"Checkpoint directory: {save_dir}")

    # Backup source code
    try:
        backup_code(save_dir / "src_backup")
        logger.info("Source code backed up")
    except Exception as e:
        logger.warning(f"Could not backup source code: {e}")

    # Save configuration
    from utils import save_config

    save_config(config, save_dir / "config.yaml")

    # Initialize Weights & Biases
    if config.get("logging", {}).get("use_wandb", False):
        wandb.init(
            project=config["logging"]["project_name"],
            name=experiment_name,
            config=config,
            dir=log_dir,
        )
        logger.info("Weights & Biases initialized")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(config, args)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    dataloader_config = config.get("dataloader", {})
    training_config = config["training"]

    train_loader = create_dataloader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=dataloader_config.get("num_workers", 4),
        pin_memory=dataloader_config.get("pin_memory", True),
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=dataloader_config.get("num_workers", 4),
        pin_memory=dataloader_config.get("pin_memory", True),
    )

    # Create model
    logger.info("Creating model...")
    model = create_sits_former(config["model"])

    from utils import count_parameters

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer, scheduler, and criterion
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, {**training_config, **config})
    criterion = create_criterion(training_config)

    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")
    logger.info(f"Criterion: {criterion.__class__.__name__}")

    # Create trainer
    trainer_config = {
        **training_config,
        **config.get("logging", {}),
        "device": str(device),
    }

    trainer = SITSFormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=trainer_config,
    )

    # Fast dev run check
    if config.get("experiment", {}).get("fast_dev_run", False):
        logger.info("Running fast dev run (1 epoch, few batches)")
        epochs = 1
    else:
        epochs = training_config["epochs"]

    # Train model
    logger.info(f"Starting training for {epochs} epochs...")

    try:
        history = trainer.train(num_epochs=epochs, save_dir=save_dir)

        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {history['train_losses'][-1]:.4f}")

        if history["val_losses"]:
            logger.info(f"Final validation loss: {history['val_losses'][-1]:.4f}")
            logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

        if history["val_accuracies"]:
            logger.info(
                f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}"
            )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    # Evaluate model
    if len(val_dataset) > 0:
        logger.info("Evaluating model...")

        evaluator = ModelEvaluator(model, device)

        # Get class names from config
        class_names = config.get("evaluation", {}).get("class_names")
        if not class_names:
            class_names = [f"Class_{i}" for i in range(config["model"]["num_classes"])]

        results = evaluator.evaluate(val_loader, class_names=class_names)

        logger.info("Evaluation Results:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Precision (macro): {results['precision_macro']:.4f}")
        logger.info(f"  Recall (macro): {results['recall_macro']:.4f}")
        logger.info(f"  F1-score (macro): {results['f1_macro']:.4f}")

        # Save evaluation results
        evaluator.save_results(results, save_dir / "evaluation_results.json")

        # Create evaluation plots
        predictions, probabilities, true_labels = evaluator.predict(val_loader)

        evaluator.plot_confusion_matrix(
            true_labels,
            predictions,
            class_names,
            save_path=save_dir / "confusion_matrix.png",
        )

        evaluator.plot_class_performance(
            results, save_path=save_dir / "class_performance.png"
        )

        logger.info("Evaluation plots saved")

    # Log final metrics to wandb
    if config.get("logging", {}).get("use_wandb", False):
        final_metrics = {
            "final_train_loss": history["train_losses"][-1],
            "best_val_loss": trainer.best_val_loss,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

        if history["val_accuracies"]:
            final_metrics["final_val_accuracy"] = history["val_accuracies"][-1]

        if len(val_dataset) > 0:
            final_metrics.update(
                {
                    "eval_accuracy": results["accuracy"],
                    "eval_precision": results["precision_macro"],
                    "eval_recall": results["recall_macro"],
                    "eval_f1": results["f1_macro"],
                }
            )

        wandb.log(final_metrics)
        wandb.finish()

    logger.info(f"Experiment completed. Results saved to: {save_dir}")


if __name__ == "__main__":
    main()

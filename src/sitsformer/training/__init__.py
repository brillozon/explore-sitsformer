"""
Training utilities and trainer classes for SITS-Former models.

This module provides comprehensive training infrastructure for SITS-Former models,
including trainer classes, optimization utilities, learning rate schedulers, and
loss functions specifically designed for satellite image time series classification.

Key Components:
    SITSFormerTrainer: Main trainer class with support for distributed training,
        mixed precision, checkpointing, and experiment tracking
    create_optimizer: Factory function for creating optimizers (Adam, AdamW, SGD)
    create_scheduler: Factory function for learning rate scheduling strategies
    create_criterion: Factory function for loss functions and class weighting

Features:
    - Automatic mixed precision training for memory efficiency
    - Distributed training support for multi-GPU setups
    - Integration with Weights & Biases and TensorBoard
    - Flexible checkpoint saving and resuming
    - Early stopping with patience
    - Gradient clipping and accumulation
    - Custom learning rate scheduling

Example:
    Basic training setup::

        from ..training import SITSFormerTrainer, create_optimizer
        from ..models import SITSFormer

        # Create model and optimizer
        model = SITSFormer(num_classes=10, embed_dim=256)
        optimizer = create_optimizer(model.parameters(), 'adamw', lr=1e-4)

        # Create trainer
        trainer = SITSFormerTrainer(
            model=model,
            optimizer=optimizer,
            device='cuda',
            mixed_precision=True
        )

        # Train the model
        trainer.train(train_loader, val_loader, epochs=100)
"""

from .trainer import (
    SITSFormerTrainer,
    create_criterion,
    create_optimizer,
    create_scheduler,
)

__all__ = [
    "SITSFormerTrainer",
    "create_optimizer",
    "create_scheduler",
    "create_criterion",
]

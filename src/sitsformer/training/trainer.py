"""
Training utilities and trainer class for SITS-Former.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class SITSFormerTrainer:
    """Trainer class for SITS-Former model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            model: SITS-Former model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config or {}

        # Move model to device
        self.model.to(self.device)

        # Set up loss function
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get("learning_rate", 1e-4),
                weight_decay=self.config.get("weight_decay", 1e-2),
            )
        else:
            self.optimizer = optimizer

        # Set up scheduler
        self.scheduler = scheduler

        # Mixed precision training
        self.use_amp = self.config.get("use_amp", True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        # Logging
        self.logger = self._setup_logging()

        # Wandb logging
        self.use_wandb = self.config.get("use_wandb", False)
        if self.use_wandb:
            wandb.watch(self.model)

    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("SITSFormerTrainer")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)  # [B, T, C, H, W]
            labels = labels.to(self.device)  # [B]

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Update metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Update progress bar
            avg_loss = total_loss / total_samples
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log(
                    {
                        "train_loss_step": loss.item(),
                        "epoch": self.current_epoch,
                        "step": self.current_epoch * len(self.train_loader) + batch_idx,
                    }
                )

        avg_loss = total_loss / total_samples
        return {"train_loss": avg_loss}

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Update metrics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Calculate accuracy
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions == labels).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    def save_checkpoint(self, filepath: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)

        if is_best:
            best_path = filepath.parent / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filepath: Path) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.val_accuracies = checkpoint["val_accuracies"]

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint

    def train(
        self, num_epochs: int, save_dir: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs."""
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["train_loss"])

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics["val_loss"])
                self.val_accuracies.append(val_metrics["val_accuracy"])

                # Check for best model
                is_best = val_metrics["val_loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["val_loss"]

                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}, "
                    f"val_acc={val_metrics['val_accuracy']:.4f}"
                )
            else:
                is_best = False
                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}"
                )

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Log to wandb
            if self.use_wandb:
                log_dict = {"epoch": epoch, **train_metrics, **val_metrics}
                wandb.log(log_dict)

            # Save checkpoint
            if save_dir is not None and (epoch % 10 == 0 or is_best):
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(checkpoint_path, is_best)

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Create optimizer from configuration."""
    optimizer_type = config.get("optimizer", "adamw").lower()
    lr = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 1e-2)

    if optimizer_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        momentum = config.get("momentum", 0.9)
        return optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(optimizer: optim.Optimizer, config: Dict) -> Optional[Any]:
    """Create learning rate scheduler from configuration."""
    scheduler_type = config.get("scheduler", None)

    if scheduler_type is None:
        return None
    elif scheduler_type == "cosine":
        T_max = config.get("epochs", 100)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == "step":
        step_size = config.get("step_size", 30)
        gamma = config.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "plateau":
        patience = config.get("patience", 10)
        factor = config.get("factor", 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def create_criterion(config: Dict) -> nn.Module:
    """Create loss function from configuration."""
    criterion_type = config.get("criterion", "cross_entropy").lower()

    if criterion_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif criterion_type == "focal":
        # Placeholder for focal loss implementation
        alpha = config.get("focal_alpha", 1.0)
        gamma = config.get("focal_gamma", 2.0)
        return nn.CrossEntropyLoss()  # Replace with actual focal loss
    elif criterion_type == "label_smoothing":
        smoothing = config.get("label_smoothing", 0.1)
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    else:
        raise ValueError(f"Unknown criterion: {criterion_type}")

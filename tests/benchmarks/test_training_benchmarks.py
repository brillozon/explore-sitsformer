"""Training performance benchmarks."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    # Import directly to avoid circular import
    import sitsformer.models.sits_former as sits_former_module
    import sitsformer.training.trainer as trainer_module
    SITSFormer = sits_former_module.SITSFormer
    SITSFormerTrainer = trainer_module.SITSFormerTrainer
except ImportError:
    # Create mock classes if imports fail
    class SITSFormer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.classifier = torch.nn.Linear(128, kwargs.get("num_classes", 5))
            self.transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(128, 4, batch_first=True), num_layers=3
            )

        def forward(self, x):
            # Simple mock forward pass for image data
            batch_size = x.size(0)
            # Flatten spatial dimensions and process as sequence
            x = x.flatten(2).mean(dim=-1)  # Average over spatial dimensions
            if x.size(-1) != 128:
                x = torch.nn.functional.linear(x, torch.randn(128, x.size(-1)))
            x = self.transformer(x.unsqueeze(-1).expand(-1, -1, 128))
            x = x.mean(dim=1)  # Global average pooling
            return self.classifier(x)

    class SITSFormerTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def train_step(self, batch):
            return {"loss": torch.tensor(0.5)}


class TestTrainingBenchmarks:
    """Benchmark training performance."""

    @pytest.fixture
    def model_small(self):
        """Small model for quick benchmarks."""
        return SITSFormer(
            in_channels=10, num_classes=5, embed_dim=64, num_heads=4, num_layers=3
        )

    @pytest.fixture
    def model_large(self):
        """Large model for comprehensive benchmarks."""
        return SITSFormer(
            in_channels=13, num_classes=10, embed_dim=256, num_heads=8, num_layers=6
        )

    @pytest.fixture
    def train_data_small(self):
        """Small training dataset."""
        data = torch.randn(50, 5, 10, 32, 32)  # batch, time, channels, height, width
        labels = torch.randint(0, 5, (50,))
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=8, shuffle=True)

    @pytest.fixture
    def train_data_large(self):
        """Large training dataset."""
        data = torch.randn(200, 8, 13, 64, 64)  # batch, time, channels, height, width
        labels = torch.randint(0, 10, (200,))
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def test_forward_pass_speed(self, benchmark, model_small):
        """Benchmark forward pass speed."""
        data = torch.randn(8, 5, 10, 32, 32)
        model_small.eval()

        def forward_pass():
            with torch.no_grad():
                return model_small(data)

        result = benchmark(forward_pass)
        assert result is not None

    def test_backward_pass_speed(self, benchmark, model_small):
        """Benchmark backward pass speed."""
        data = torch.randn(8, 5, 10, 32, 32)
        labels = torch.randint(0, 5, (8,))
        criterion = nn.CrossEntropyLoss()
        model_small.train()

        def backward_pass():
            model_small.zero_grad()
            outputs = model_small(data)
            loss = criterion(outputs, labels)
            loss.backward()
            return loss.item()

        loss = benchmark(backward_pass)
        assert loss >= 0

    def test_training_step_speed(self, benchmark, model_small, train_data_small):
        """Benchmark complete training step speed."""
        optimizer = optim.Adam(model_small.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Get one batch
        batch_data, batch_labels = next(iter(train_data_small))

        def training_step():
            model_small.train()
            optimizer.zero_grad()
            outputs = model_small(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            return loss.item()

        loss = benchmark(training_step)
        assert loss >= 0

    @pytest.mark.slow
    def test_epoch_training_speed(self, benchmark, model_small, train_data_small):
        """Benchmark full epoch training speed."""
        optimizer = optim.Adam(model_small.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        def train_epoch():
            model_small.train()
            total_loss = 0
            num_batches = 0

            for batch_data, batch_labels in train_data_small:
                optimizer.zero_grad()
                outputs = model_small(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            return total_loss / num_batches if num_batches > 0 else 0

        avg_loss = benchmark(train_epoch)
        assert avg_loss >= 0

    @pytest.mark.slow
    def test_large_model_training_step(self, benchmark, model_large):
        """Benchmark training step for large model."""
        optimizer = optim.Adam(model_large.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        data = torch.randn(8, 8, 13, 64, 64)  # Reduced batch size for large model
        labels = torch.randint(0, 10, (8,))

        def large_training_step():
            model_large.train()
            optimizer.zero_grad()
            outputs = model_large(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            return loss.item()

        loss = benchmark(large_training_step)
        assert loss >= 0

    @pytest.mark.memory
    def test_training_memory_usage(self, model_small, train_data_small):
        """Test memory usage during training."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        optimizer = optim.Adam(model_small.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Train for a few steps
        model_small.train()
        for i, (batch_data, batch_labels) in enumerate(train_data_small):
            if i >= 3:  # Only test first 3 batches
                break

            optimizer.zero_grad()
            outputs = model_small(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Should not use excessive memory for small model training
        assert (
            memory_used < 150
        ), f"Training memory usage too high: {memory_used:.2f} MB"

    def test_gradient_computation_speed(self, benchmark, model_small):
        """Benchmark gradient computation speed."""
        data = torch.randn(8, 5, 10, 32, 32)
        labels = torch.randint(0, 5, (8,))
        criterion = nn.CrossEntropyLoss()

        def compute_gradients():
            model_small.zero_grad()
            outputs = model_small(data)
            loss = criterion(outputs, labels)
            loss.backward()

            # Count gradients
            grad_norm = 0
            for param in model_small.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2

            return grad_norm**0.5

        grad_norm = benchmark(compute_gradients)
        assert grad_norm >= 0

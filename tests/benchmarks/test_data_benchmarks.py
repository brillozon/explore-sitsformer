"""Data loading performance benchmarks."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import tempfile
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from sitsformer.data.dataset import SITSDataset
    from sitsformer.data.utils import normalize_time_series
except ImportError:
    # Create mock classes if imports fail
    class SITSDataset:
        def __init__(self, *args, **kwargs):
            self.data = torch.randn(100, 24, 10)
            self.labels = torch.randint(0, 5, (100,))

        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    def normalize_time_series(x):
        return (x - x.mean()) / (x.std() + 1e-8)


class TestDataLoadingBenchmarks:
    """Benchmark data loading performance."""

    @pytest.fixture
    def sample_dataset_small(self):
        """Create small synthetic dataset."""
        data = torch.randn(100, 24, 10)  # 100 samples, 24 timesteps, 10 features
        labels = torch.randint(0, 5, (100,))
        return TensorDataset(data, labels)

    @pytest.fixture
    def sample_dataset_large(self):
        """Create large synthetic dataset."""
        data = torch.randn(1000, 48, 13)  # 1000 samples, 48 timesteps, 13 features
        labels = torch.randint(0, 10, (1000,))
        return TensorDataset(data, labels)

    def test_dataloader_small_batch(self, benchmark, sample_dataset_small):
        """Benchmark small batch data loading."""
        dataloader = DataLoader(
            sample_dataset_small, batch_size=8, shuffle=True, num_workers=0
        )

        def load_one_epoch():
            batches = list(dataloader)
            return len(batches)

        num_batches = benchmark(load_one_epoch)
        assert num_batches > 0

    @pytest.mark.slow
    def test_dataloader_large_batch(self, benchmark, sample_dataset_large):
        """Benchmark large batch data loading."""
        dataloader = DataLoader(
            sample_dataset_large, batch_size=32, shuffle=True, num_workers=0
        )

        def load_one_epoch():
            batches = list(dataloader)
            return len(batches)

        num_batches = benchmark(load_one_epoch)
        assert num_batches > 0

    def test_multiprocessing_loading(self, benchmark, sample_dataset_small):
        """Benchmark multiprocessing data loading."""
        dataloader = DataLoader(
            sample_dataset_small, batch_size=8, shuffle=True, num_workers=2
        )

        def load_with_multiprocessing():
            batches = list(dataloader)
            return len(batches)

        num_batches = benchmark(load_with_multiprocessing)
        assert num_batches > 0

    def test_data_preprocessing_speed(self, benchmark):
        """Benchmark data preprocessing operations."""
        data = torch.randn(32, 24, 10)

        def preprocess_batch():
            # Simulate common preprocessing steps
            normalized = normalize_time_series(data)
            augmented = normalized + torch.randn_like(normalized) * 0.1
            return augmented

        result = benchmark(preprocess_batch)
        assert result.shape == data.shape

    @pytest.mark.slow
    def test_data_augmentation_speed(self, benchmark):
        """Benchmark data augmentation operations."""
        data = torch.randn(32, 48, 13)

        def augment_batch():
            # Simulate augmentation pipeline
            # Time shifting
            shift = torch.randint(-2, 3, (1,)).item()
            if shift != 0:
                data_shifted = torch.roll(data, shift, dims=1)
            else:
                data_shifted = data

            # Noise addition
            noise = torch.randn_like(data_shifted) * 0.05
            augmented = data_shifted + noise

            # Normalization
            normalized = normalize_time_series(augmented)

            return normalized

        result = benchmark(augment_batch)
        assert result.shape == data.shape

    @pytest.mark.memory
    def test_memory_efficient_loading(self, sample_dataset_large):
        """Test memory usage during data loading."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Load data in batches
        dataloader = DataLoader(sample_dataset_large, batch_size=64, num_workers=0)

        for i, (batch_data, batch_labels) in enumerate(dataloader):
            if i >= 5:  # Only test first 5 batches
                break

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Should not use excessive memory
        assert memory_used < 200, f"Memory usage too high: {memory_used:.2f} MB"

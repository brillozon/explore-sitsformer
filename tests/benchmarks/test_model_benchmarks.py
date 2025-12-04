"""Model performance benchmarks."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    # Import directly to avoid circular import
    import sitsformer.models.sits_former as sits_former_module
    SITSFormer = sits_former_module.SITSFormer
except ImportError:
    # Create mock classes if imports fail
    class SITSFormer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            embed_dim = kwargs.get("embed_dim", 128)
            num_classes = kwargs.get("num_classes", 5)
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(embed_dim, embed_dim) for _ in range(6)]
            )
            self.classifier = torch.nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            # Mock forward pass for image tensor
            batch_size = x.size(0)
            # Flatten and reduce to feature vector
            x = x.flatten(2).mean(dim=-1)  # Average spatial dimensions
            if x.size(-1) != self.layers[0].in_features:
                # Project to correct dimension
                linear = torch.nn.Linear(x.size(-1), self.layers[0].in_features)
                x = linear(x)

            for layer in self.layers:
                x = torch.relu(layer(x))
            x = x.mean(dim=1)  # Average over time dimension
            return self.classifier(x)


class TestModelInferenceBenchmarks:
    """Benchmark model inference performance."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for quick benchmarks."""
        return SITSFormer(
            in_channels=10,
            num_classes=5,
            embed_dim=64,
            num_heads=4,
            num_layers=3,
            mlp_ratio=2,
        )

    @pytest.fixture
    def large_model(self):
        """Create a larger model for comprehensive benchmarks."""
        return SITSFormer(
            in_channels=13,
            num_classes=10,
            embed_dim=256,
            num_heads=8,
            num_layers=8,
            mlp_ratio=4,
        )

    @pytest.fixture
    def sample_data_small(self):
        """Small batch for quick tests."""
        batch_size, seq_len, channels, height, width = 8, 5, 10, 32, 32
        return torch.randn(batch_size, seq_len, channels, height, width)

    @pytest.fixture
    def sample_data_large(self):
        """Larger batch for comprehensive tests."""
        batch_size, seq_len, channels, height, width = 16, 8, 13, 64, 64
        return torch.randn(batch_size, seq_len, channels, height, width)

    def test_small_model_inference_speed(
        self, benchmark, small_model, sample_data_small
    ):
        """Benchmark inference speed for small model (quick test)."""
        small_model.eval()
        with torch.no_grad():
            result = benchmark(small_model, sample_data_small)
        assert result is not None

    @pytest.mark.slow
    def test_large_model_inference_speed(
        self, benchmark, large_model, sample_data_large
    ):
        """Benchmark inference speed for large model (comprehensive test)."""
        large_model.eval()
        with torch.no_grad():
            result = benchmark(large_model, sample_data_large)
        assert result is not None

    def test_batch_size_scaling(self, benchmark, small_model):
        """Benchmark how inference scales with batch size."""

        def run_inference_with_batch_size(batch_size):
            data = torch.randn(batch_size, 5, 10, 32, 32)
            small_model.eval()
            with torch.no_grad():
                return small_model(data)

        # Test with batch size 4
        result = benchmark(run_inference_with_batch_size, 4)
        assert result is not None

    @pytest.mark.slow
    def test_sequence_length_scaling(self, benchmark, small_model):
        """Benchmark how inference scales with sequence length."""

        def run_inference_with_seq_len(seq_len):
            data = torch.randn(4, seq_len, 10, 32, 32)
            small_model.eval()
            with torch.no_grad():
                return small_model(data)

        # Test with sequence length 8
        result = benchmark(run_inference_with_seq_len, 8)
        assert result is not None

    @pytest.mark.memory
    def test_memory_usage_small(self, small_model, sample_data_small):
        """Test memory usage for small model."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        small_model.eval()
        with torch.no_grad():
            _ = small_model(sample_data_small)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Should use less than 100MB for small model
        assert memory_used < 100, f"Memory usage too high: {memory_used:.2f} MB"

    @pytest.mark.memory
    @pytest.mark.slow
    def test_memory_usage_large(self, large_model, sample_data_large):
        """Test memory usage for large model."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        large_model.eval()
        with torch.no_grad():
            _ = large_model(sample_data_large)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Should use less than 500MB for large model
        assert memory_used < 500, f"Memory usage too high: {memory_used:.2f} MB"

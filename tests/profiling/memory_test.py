#!/usr/bin/env python3
"""Memory profiling script for SITS-Former."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import gc
import psutil
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    # Import directly to avoid circular import
    import sitsformer.models.sits_former as sits_former_module

    SITSFormer = sits_former_module.SITSFormer
except ImportError:
    # Create mock class if import fails
    class SITSFormer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            embed_dim = kwargs.get("embed_dim", 256)
            self.layers = nn.ModuleList(
                [
                    nn.Linear(embed_dim, embed_dim)
                    for _ in range(kwargs.get("num_layers", 6))
                ]
            )
            self.classifier = nn.Linear(embed_dim, kwargs.get("num_classes", 10))

        def forward(self, x):
            # Simple mock forward pass for image data
            batch_size = x.size(0)
            x = x.flatten(2).mean(dim=-1)  # Average spatial dimensions
            if x.size(-1) != self.layers[0].in_features:
                linear = nn.Linear(x.size(-1), self.layers[0].in_features)
                x = linear(x)

            for layer in self.layers:
                x = torch.relu(layer(x))
            x = x.mean(dim=1)  # Average over time
            return self.classifier(x)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# Try to use memory_profiler if available, otherwise use dummy decorator
try:
    from memory_profiler import profile
except ImportError:

    def profile(func):
        """Dummy profile decorator when memory_profiler is not available."""
        return func


@profile
def create_and_run_model():
    """Profile model creation and inference."""
    print("Creating model...")
    memory_start = get_memory_usage()

    # Create model
    model = SITSFormer(
        in_channels=13,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4,
    )

    memory_after_model = get_memory_usage()
    print(f"Memory after model creation: {memory_after_model - memory_start:.2f} MB")

    # Create sample data
    print("Creating sample data...")
    batch_size = 8  # Reduced batch size
    sequence_length = 5
    channels = 13
    height = width = 64

    data = torch.randn(batch_size, sequence_length, channels, height, width)
    memory_after_data = get_memory_usage()
    print(
        f"Memory after data creation: {memory_after_data - memory_after_model:.2f} MB"
    )

    # Run inference
    print("Running inference...")
    model.eval()
    with torch.no_grad():
        outputs = model(data)

    memory_after_inference = get_memory_usage()
    print(
        f"Memory after inference: {memory_after_inference - memory_after_data:.2f} MB"
    )

    return outputs


@profile
def training_memory_profile():
    """Profile memory usage during training."""
    print("\nTraining memory profile...")
    memory_start = get_memory_usage()

    # Smaller model for training
    model = SITSFormer(
        in_channels=10, num_classes=5, embed_dim=128, num_heads=4, num_layers=4
    )

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    memory_after_setup = get_memory_usage()
    print(f"Memory after training setup: {memory_after_setup - memory_start:.2f} MB")

    # Training loop
    model.train()
    for step in range(10):
        # Create batch
        batch_data = torch.randn(4, 5, 10, 32, 32)  # Small batch for training
        batch_labels = torch.randint(0, 5, (4,))

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if step % 3 == 0:
            current_memory = get_memory_usage()
            print(f"Memory at step {step}: {current_memory - memory_start:.2f} MB")

    final_memory = get_memory_usage()
    print(f"Final memory usage: {final_memory - memory_start:.2f} MB")


@profile
def batch_size_scaling_profile():
    """Profile how memory scales with batch size."""
    print("\nBatch size scaling profile...")

    model = SITSFormer(
        in_channels=10, num_classes=5, embed_dim=128, num_heads=4, num_layers=3
    )

    model.eval()

    batch_sizes = [1, 4, 8, 16, 32, 64]

    for batch_size in batch_sizes:
        gc.collect()  # Clean up before each test
        memory_before = get_memory_usage()

        data = torch.randn(batch_size, 5, 10, 32, 32)  # time, channels, height, width

        with torch.no_grad():
            outputs = model(data)

        memory_after = get_memory_usage()
        memory_used = memory_after - memory_before

        print(f"Batch size {batch_size:2d}: {memory_used:.2f} MB")

        # Clean up
        del data, outputs


def main():
    """Main profiling function."""
    print("Starting memory profiling...")
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")

    # Run profiling functions
    try:
        outputs = create_and_run_model()
        print(f"Model output shape: {outputs.shape}")

        training_memory_profile()

        batch_size_scaling_profile()

    except Exception as e:
        print(f"Error during profiling: {e}")
        print(
            "This is expected if running without memory_profiler or model imports fail"
        )

    print(f"\nFinal memory usage: {get_memory_usage():.2f} MB")
    print("Memory profiling complete.")


if __name__ == "__main__":
    main()

"""Test SITS-Former model creation and basic functionality."""

import torch

from sitsformer.models import create_sits_former


def test_model_creation():
    """Test basic model creation."""
    config = {
        "in_channels": 10,
        "num_classes": 5,
        "patch_size": 16,
        "sequence_length": 24,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
    }

    model = create_sits_former(config)
    assert model is not None

    # Test forward pass
    batch_size = 2
    seq_len = config["sequence_length"]
    channels = config["in_channels"]
    height = width = 64

    x = torch.randn(batch_size, seq_len, channels, height, width)
    output = model(x)

    assert output.shape == (batch_size, config["num_classes"])


def test_model_different_configs():
    """Test model with different configurations."""
    configs = [
        {
            "in_channels": 13,
            "num_classes": 10,
            "patch_size": 8,
            "sequence_length": 12,
            "hidden_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "dropout": 0.0,
        },
        {
            "in_channels": 4,
            "num_classes": 3,
            "patch_size": 32,
            "sequence_length": 6,
            "hidden_dim": 64,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.2,
        },
    ]

    for config in configs:
        model = create_sits_former(config)
        assert model is not None

        # Test parameter count is reasonable
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

        # Test forward pass
        x = torch.randn(1, config["sequence_length"], config["in_channels"], 32, 32)
        output = model(x)
        assert output.shape == (1, config["num_classes"])


if __name__ == "__main__":
    test_model_creation()
    test_model_different_configs()
    print("All tests passed!")

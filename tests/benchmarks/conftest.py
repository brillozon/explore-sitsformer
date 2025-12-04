"""Configuration for benchmark tests."""

import pytest
import torch


def pytest_configure(config):
    """Configure pytest for benchmarks."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "memory: marks tests as memory profiling tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


@pytest.fixture(scope="session")
def device():
    """Determine device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def torch_seed():
    """Set random seed for reproducible benchmarks."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    return 42


@pytest.fixture(autouse=True)
def setup_benchmark_env(torch_seed):
    """Setup environment for each benchmark test."""
    # Ensure reproducible results
    torch.manual_seed(torch_seed)
    
    # Set deterministic behavior for benchmarks
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    yield
    
    # Cleanup after test
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

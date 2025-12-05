"""
SITS-Former: Satellite Image Time Series Transformer

A PyTorch implementation of transformer-based models for satellite image time series
analysis and classification. SITS-Former is designed for processing temporal sequences
of satellite imagery for various remote sensing applications including land cover
classification, crop monitoring, and change detection.

Key Features:
    - Transformer-based architecture for temporal sequence modeling
    - Support for multi-spectral satellite imagery
    - Configurable model architectures
    - Data augmentation pipelines for satellite imagery
    - Integration with experiment tracking tools
    - Comprehensive evaluation metrics

Example:
    Basic usage for land cover classification::

        from sitsformer.models import SITSFormer
        from sitsformer.data import SatelliteTimeSeriesDataset

        # Create model
        model = SITSFormer(
            input_dim=13,  # Sentinel-2 bands
            num_classes=10,  # Land cover classes
            d_model=128,
            nhead=8,
            num_layers=6
        )

        # Load data
        dataset = SatelliteTimeSeriesDataset(
            data_dir="path/to/data",
            metadata_file="metadata.csv"
        )
"""

# Use lazy imports to avoid circular import issues
# Import submodules only when they are accessed

def __getattr__(name):
    """Lazy import of submodules."""
    if name == "data":
        from . import data
        return data
    elif name == "evaluation":
        from . import evaluation
        return evaluation
    elif name == "models":
        from . import models
        return models
    elif name == "training":
        from . import training
        return training
    elif name == "utils":
        from . import utils
        return utils
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = "0.1.0"
__author__ = "Mike Martinez"
__email__ = "brillozon@gmail.com"

__all__ = ["models", "data", "training", "evaluation", "utils"]

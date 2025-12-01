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

from . import models
from . import data
from . import training
from . import evaluation
from . import utils

__version__ = "0.1.0"
__author__ = "Mike Martinez"
__email__ = "brillozon@gmail.com"

__all__ = ['models', 'data', 'training', 'evaluation', 'utils']
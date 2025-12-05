"""
Model architectures for satellite image time series analysis.

This module contains transformer-based model implementations specifically designed
for processing temporal sequences of satellite imagery. The models support various
remote sensing applications including land cover classification, crop monitoring,
and change detection.

Available Models:
    SITSFormer: Main transformer model for satellite image time series

Utilities:
    create_sits_former: Factory function for creating pre-configured models

Example:
    Creating and using a SITS-Former model::

        from ..models import SITSFormer, create_sits_former

        # Method 1: Direct instantiation
        model = SITSFormer(
            input_dim=13,      # Number of spectral bands
            num_classes=10,    # Number of output classes
            d_model=256,       # Model dimension
            nhead=8,          # Number of attention heads
            num_layers=6,     # Number of transformer layers
            dropout=0.1
        )

        # Method 2: Using factory function
        model = create_sits_former(
            model_type="base",
            input_dim=13,
            num_classes=10
        )
"""

from .sits_former import SITSFormer, create_sits_former

__all__ = ['SITSFormer', 'create_sits_former']

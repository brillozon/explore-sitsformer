Data Preparation for Satellite Image Time Series
=================================================

This tutorial covers the complete data preparation pipeline for satellite image time series analysis using SITS-Former. We'll walk through data organization, preprocessing, normalization, and dataset creation.

Overview
--------

Preparing satellite imagery data for deep learning involves several key steps:

1. **Data Organization** - Structuring files and metadata
2. **Quality Assessment** - Handling clouds, shadows, and missing data  
3. **Normalization** - Scaling pixel values for model training
4. **Temporal Alignment** - Synchronizing time series observations
5. **Dataset Creation** - PyTorch-compatible data loading

Prerequisites
-------------

Before starting, ensure you have:

- Satellite imagery (Sentinel-2, Landsat, etc.)
- Ground truth labels or reference data
- Sufficient storage space (satellite data can be large)
- Basic understanding of remote sensing concepts

Data Organization
-----------------

Recommended Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Organize your data following this structure for best results:

.. code-block:: text

    project_data/
    ├── raw_images/
    │   ├── sentinel2/
    │   │   ├── 2023/
    │   │   │   ├── site001_20230115_S2.tif
    │   │   │   ├── site001_20230130_S2.tif
    │   │   │   └── site001_20230214_S2.tif
    │   │   └── 2024/
    │   └── landsat8/
    ├── labels/
    │   ├── site_labels.csv
    │   └── class_definitions.json
    ├── metadata/
    │   ├── image_metadata.csv
    │   └── acquisition_log.json
    └── processed/
        ├── normalized/
        └── time_series/

Metadata Files
~~~~~~~~~~~~~~

Create a comprehensive metadata file (CSV format):

.. code-block:: python

    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Example metadata structure
    metadata_columns = [
        'image_id',           # Unique identifier
        'site_id',           # Spatial location identifier  
        'acquisition_date',   # Image acquisition date
        'sensor',            # Satellite sensor (S2, L8, etc.)
        'cloud_cover',       # Cloud coverage percentage
        'file_path',         # Relative path to image file
        'label',             # Ground truth class
        'quality_flag'       # Data quality indicator
    ]
    
    # Create sample metadata
    sample_metadata = pd.DataFrame({
        'image_id': ['S2_001_20230115', 'S2_001_20230130', 'S2_001_20230214'],
        'site_id': ['site_001', 'site_001', 'site_001'],
        'acquisition_date': ['2023-01-15', '2023-01-30', '2023-02-14'],
        'sensor': ['Sentinel-2', 'Sentinel-2', 'Sentinel-2'],
        'cloud_cover': [5.2, 12.8, 3.1],
        'file_path': [
            'sentinel2/2023/site001_20230115_S2.tif',
            'sentinel2/2023/site001_20230130_S2.tif', 
            'sentinel2/2023/site001_20230214_S2.tif'
        ],
        'label': ['Forest', 'Forest', 'Forest'],
        'quality_flag': ['Good', 'Fair', 'Good']
    })
    
    # Save metadata
    sample_metadata.to_csv('metadata/image_metadata.csv', index=False)
    print("Metadata file created successfully!")

Data Loading and Exploration
-----------------------------

Loading Satellite Images
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    def load_satellite_image(image_path):
        """Load satellite image with all bands."""
        with rasterio.open(image_path) as src:
            # Read all bands
            image = src.read()  # Shape: (bands, height, width)
            
            # Get metadata
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'band_count': src.count,
                'dtype': src.dtype
            }
            
        # Transpose to (height, width, bands) for easier handling
        image = np.transpose(image, (1, 2, 0))
        
        return image, metadata
    
    # Load example Sentinel-2 image
    image_path = "raw_images/sentinel2/2023/site001_20230115_S2.tif"
    image, meta = load_satellite_image(image_path)
    
    print(f"Image shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Value range: {image.min()} - {image.max()}")

Visualizing Satellite Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def plot_satellite_bands(image, bands_to_plot=None, figsize=(15, 10)):
        """Plot individual spectral bands."""
        if bands_to_plot is None:
            # Default Sentinel-2 bands for visualization
            bands_to_plot = {
                'Blue (B2)': 1, 'Green (B3)': 2, 'Red (B4)': 3,
                'NIR (B8)': 7, 'SWIR1 (B11)': 10, 'SWIR2 (B12)': 11
            }
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (name, band_idx) in enumerate(bands_to_plot.items()):
            if idx < len(axes) and band_idx < image.shape[2]:
                band_data = image[:, :, band_idx]
                
                # Apply percentile stretch for better visualization
                p2, p98 = np.percentile(band_data, (2, 98))
                band_stretched = np.clip((band_data - p2) / (p98 - p2), 0, 1)
                
                axes[idx].imshow(band_stretched, cmap='gray')
                axes[idx].set_title(f'{name}')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Visualize bands
    plot_satellite_bands(image)

Data Quality Assessment
-----------------------

Cloud and Shadow Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def assess_image_quality(image, cloud_threshold=0.15):
        """Assess image quality based on various criteria."""
        quality_metrics = {}
        
        # Cloud detection (simplified approach using blue band)
        blue_band = image[:, :, 1]  # Assuming Sentinel-2 band order
        bright_pixels = (blue_band > np.percentile(blue_band, 95)).sum()
        total_pixels = blue_band.size
        cloud_ratio = bright_pixels / total_pixels
        
        quality_metrics['cloud_ratio'] = cloud_ratio
        quality_metrics['cloud_contaminated'] = cloud_ratio > cloud_threshold
        
        # Check for missing data (NaN or extreme values)
        missing_data = np.isnan(image).any(axis=2).sum()
        quality_metrics['missing_pixels'] = missing_data
        
        # Dynamic range assessment
        for i in range(image.shape[2]):
            band_range = image[:, :, i].max() - image[:, :, i].min()
            quality_metrics[f'band_{i}_range'] = band_range
        
        # Overall quality flag
        if cloud_ratio < 0.05 and missing_data == 0:
            quality_metrics['overall_quality'] = 'Excellent'
        elif cloud_ratio < 0.15 and missing_data < 100:
            quality_metrics['overall_quality'] = 'Good'
        elif cloud_ratio < 0.30:
            quality_metrics['overall_quality'] = 'Fair'
        else:
            quality_metrics['overall_quality'] = 'Poor'
            
        return quality_metrics
    
    # Assess quality
    quality = assess_image_quality(image)
    print(f"Quality Assessment:")
    for key, value in quality.items():
        print(f"  {key}: {value}")

Data Preprocessing
------------------

Normalization and Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sitsformer.data.utils import calculate_band_statistics, normalize_image
    
    def preprocess_satellite_image(image, stats=None, clip_values=None):
        """Comprehensive preprocessing pipeline."""
        
        # Step 1: Handle invalid values
        image = np.where(np.isnan(image), 0, image)
        image = np.where(image < 0, 0, image)
        
        # Step 2: Optional clipping (remove outliers)
        if clip_values:
            for i in range(image.shape[2]):
                p_low, p_high = clip_values
                low_val = np.percentile(image[:, :, i], p_low)
                high_val = np.percentile(image[:, :, i], p_high)
                image[:, :, i] = np.clip(image[:, :, i], low_val, high_val)
        
        # Step 3: Normalization using precomputed statistics
        if stats is not None:
            image_normalized = normalize_image(image, stats['mean'], stats['std'])
        else:
            # Per-image normalization (not recommended for training)
            means = np.mean(image, axis=(0, 1))
            stds = np.std(image, axis=(0, 1))
            image_normalized = (image - means) / (stds + 1e-8)
        
        return image_normalized
    
    # Example: Calculate dataset statistics
    # In practice, you'd compute these from your training set
    stats = calculate_band_statistics('/path/to/training/data')
    
    # Preprocess image
    processed_image = preprocess_satellite_image(
        image, 
        stats=stats,
        clip_values=(1, 99)  # Remove 1% outliers on each end
    )
    
    print(f"Original range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"Normalized range: [{processed_image.min():.2f}, {processed_image.max():.2f}]")

Temporal Alignment
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def create_time_series(metadata_df, max_sequence_length=10, min_sequence_length=5):
        """Create time series from individual images."""
        time_series_data = []
        
        # Group by site
        for site_id in metadata_df['site_id'].unique():
            site_data = metadata_df[metadata_df['site_id'] == site_id].copy()
            
            # Sort by acquisition date
            site_data['acquisition_date'] = pd.to_datetime(site_data['acquisition_date'])
            site_data = site_data.sort_values('acquisition_date')
            
            # Filter by quality
            quality_data = site_data[site_data['quality_flag'].isin(['Good', 'Excellent'])]
            
            if len(quality_data) >= min_sequence_length:
                # Create sequences of max_sequence_length
                for start_idx in range(len(quality_data) - min_sequence_length + 1):
                    end_idx = min(start_idx + max_sequence_length, len(quality_data))
                    sequence = quality_data.iloc[start_idx:end_idx]
                    
                    time_series_data.append({
                        'site_id': site_id,
                        'sequence_length': len(sequence),
                        'start_date': sequence['acquisition_date'].iloc[0],
                        'end_date': sequence['acquisition_date'].iloc[-1],
                        'image_paths': sequence['file_path'].tolist(),
                        'labels': sequence['label'].iloc[0]  # Assuming consistent label
                    })
        
        return pd.DataFrame(time_series_data)
    
    # Create time series from metadata
    time_series_df = create_time_series(sample_metadata)
    print(f"Created {len(time_series_df)} time series sequences")

Dataset Creation
----------------

Custom Dataset Class
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from torch.utils.data import Dataset
    from sitsformer.data import SatelliteTimeSeriesDataset
    
    class CustomSITSDataset(Dataset):
        """Custom dataset for satellite image time series."""
        
        def __init__(self, time_series_df, data_root, transform=None, 
                     normalize_stats=None, max_sequence_length=10):
            self.time_series_df = time_series_df
            self.data_root = Path(data_root)
            self.transform = transform
            self.normalize_stats = normalize_stats
            self.max_sequence_length = max_sequence_length
            
            # Create label encoding
            unique_labels = sorted(time_series_df['labels'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            
        def __len__(self):
            return len(self.time_series_df)
        
        def __getitem__(self, idx):
            row = self.time_series_df.iloc[idx]
            
            # Load image sequence
            images = []
            for img_path in row['image_paths']:
                full_path = self.data_root / img_path
                image, _ = load_satellite_image(full_path)
                
                # Preprocess
                if self.normalize_stats:
                    image = preprocess_satellite_image(image, self.normalize_stats)
                
                images.append(image)
            
            # Convert to tensor and pad sequence
            sequence = torch.stack([torch.from_numpy(img).float() for img in images])
            
            # Pad sequence to max_sequence_length
            current_length = sequence.shape[0]
            if current_length < self.max_sequence_length:
                padding = torch.zeros(
                    self.max_sequence_length - current_length,
                    *sequence.shape[1:]
                )
                sequence = torch.cat([sequence, padding], dim=0)
                mask = torch.cat([
                    torch.ones(current_length),
                    torch.zeros(self.max_sequence_length - current_length)
                ])
            else:
                sequence = sequence[:self.max_sequence_length]
                mask = torch.ones(self.max_sequence_length)
            
            # Convert sequence to [T, H, W, C] format
            sequence = sequence.permute(0, 3, 1, 2)  # [T, C, H, W]
            
            # Get label
            label = self.label_to_idx[row['labels']]
            
            return {
                'sequence': sequence,
                'mask': mask,
                'label': torch.tensor(label, dtype=torch.long),
                'site_id': row['site_id']
            }

Data Splitting and Validation
------------------------------

Train/Validation/Test Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    def split_dataset(time_series_df, test_size=0.2, val_size=0.2, 
                      random_state=42, stratify_by='labels'):
        """Split dataset maintaining temporal and spatial balance."""
        
        # Ensure reproducibility
        np.random.seed(random_state)
        
        # Stratified split by labels
        if stratify_by:
            stratify_col = time_series_df[stratify_by]
        else:
            stratify_col = None
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            time_series_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        # Second split: train vs val
        if stratify_by:
            stratify_col_train = train_val_df[stratify_by]
        else:
            stratify_col_train = None
            
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size/(1-test_size),  # Adjust for reduced dataset
            random_state=random_state,
            stratify=stratify_col_train
        )
        
        print(f"Dataset split completed:")
        print(f"  Training: {len(train_df)} sequences ({len(train_df)/len(time_series_df)*100:.1f}%)")
        print(f"  Validation: {len(val_df)} sequences ({len(val_df)/len(time_series_df)*100:.1f}%)")
        print(f"  Test: {len(test_df)} sequences ({len(test_df)/len(time_series_df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    # Split the dataset
    train_df, val_df, test_df = split_dataset(time_series_df)

DataLoader Creation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torch.utils.data import DataLoader
    from sitsformer.data import create_dataloader
    
    def create_dataloaders(train_df, val_df, test_df, data_root, 
                          normalize_stats, batch_size=32):
        """Create PyTorch DataLoaders for training."""
        
        # Create datasets
        train_dataset = CustomSITSDataset(
            train_df, data_root, normalize_stats=normalize_stats
        )
        val_dataset = CustomSITSDataset(
            val_df, data_root, normalize_stats=normalize_stats
        )
        test_dataset = CustomSITSDataset(
            test_df, data_root, normalize_stats=normalize_stats
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        data_root='/path/to/project_data/raw_images',
        normalize_stats=stats,
        batch_size=16
    )
    
    print(f"Data loaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

Data Validation
---------------

.. code-block:: python

    def validate_dataset(dataloader, num_samples=5):
        """Validate dataset integrity and display samples."""
        
        print("Dataset Validation Report")
        print("=" * 50)
        
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            sequences = batch['sequence']
            masks = batch['mask']
            labels = batch['label']
            
            print(f"\\nBatch {i+1}:")
            print(f"  Sequence shape: {sequences.shape}")
            print(f"  Mask shape: {masks.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Sequence range: [{sequences.min():.3f}, {sequences.max():.3f}]")
            print(f"  Valid time steps: {masks.sum(dim=1).int().tolist()}")
            print(f"  Unique labels: {labels.unique().tolist()}")
            
            # Check for NaN or infinite values
            if torch.isnan(sequences).any():
                print("  WARNING: NaN values detected in sequences!")
            if torch.isinf(sequences).any():
                print("  WARNING: Infinite values detected in sequences!")
                
    # Validate the training data loader
    validate_dataset(train_loader)

Best Practices Summary
----------------------

Key Recommendations
~~~~~~~~~~~~~~~~~~~

1. **Data Organization**
   - Use consistent naming conventions
   - Maintain comprehensive metadata
   - Separate raw and processed data

2. **Quality Control**
   - Implement automated quality assessment
   - Filter out heavily cloud-contaminated images
   - Document data filtering decisions

3. **Preprocessing**
   - Compute normalization statistics on training data only
   - Handle missing values consistently
   - Consider sensor-specific preprocessing needs

4. **Temporal Considerations**
   - Align time series to meaningful temporal patterns
   - Consider seasonal variations in your application
   - Handle irregular sampling intervals appropriately

5. **Memory Management**
   - Use efficient data loading with multiple workers
   - Implement lazy loading for large datasets
   - Consider data compression for storage

Next Steps
----------

After completing data preparation:

1. Proceed to :doc:`model_training` tutorial
2. Explore :doc:`fine_tuning` for domain adaptation  
3. Check :doc:`../examples/land_cover_classification` for real-world applications

The quality of your data preparation directly impacts model performance. Take time to understand your data and implement appropriate preprocessing steps for your specific application.
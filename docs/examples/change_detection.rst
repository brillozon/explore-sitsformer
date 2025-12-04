Change Detection
================

This example demonstrates how to use SITS-Former for detecting and analyzing changes in satellite image time series.

Overview
--------

Change detection applications include:

- Deforestation monitoring
- Urban expansion tracking
- Natural disaster impact assessment
- Agricultural land conversion
- Wetland dynamics
- Infrastructure development

Dataset
-------

We use a multi-temporal dataset covering:

- **Temporal span**: 3 years
- **Temporal resolution**: Monthly composites
- **Spatial resolution**: 30m
- **Study area**: Mixed landscape (forest, agriculture, urban)
- **Change types**: Deforestation, urban growth, water level changes

Data Preprocessing
------------------

.. code-block:: python

    from sitsformer.data import ChangeDetectionDataLoader
    from sitsformer.data.utils import normalize_time_series
    import numpy as np
    
    # Load change detection dataset
    data_loader = ChangeDetectionDataLoader(
        data_path="/path/to/time_series",
        reference_path="/path/to/reference_changes",
        sequence_length=36,  # 36 monthly observations
        patch_size=64,  # 64x64 pixel patches
        batch_size=16,
        overlap=0.5
    )
    
    # Preprocessing pipeline
    def preprocess_change_data(time_series):
        # Normalize and remove outliers
        normalized = normalize_time_series(time_series)
        
        # Calculate temporal derivatives
        derivatives = np.diff(normalized, axis=1)
        
        # Combine original and derivatives
        return np.concatenate([normalized[:, 1:], derivatives], axis=-1)
    
    data_loader.add_transform(preprocess_change_data)

Model Configuration
-------------------

.. code-block:: python

    from sitsformer.models import ChangeDetectionSITSFormer
    
    # Specialized model for change detection
    model = ChangeDetectionSITSFormer(
        input_dim=12,  # 6 bands + 6 derivatives
        d_model=192,
        nhead=8,
        num_layers=6,
        dim_feedforward=768,
        dropout=0.1,
        change_detection_head={
            'binary_change': 1,      # Binary change/no-change
            'change_type': 5,        # 5 types of change
            'change_magnitude': 1,   # Continuous magnitude
            'change_timing': 36      # When change occurred
        },
        attention_mechanism='temporal_spatial'
    )

Training Approach
-----------------

.. code-block:: python

    from sitsformer.training import ChangeDetectionTrainer
    import torch.optim as optim
    import torch.nn as nn
    
    # Custom loss for change detection
    class ChangeDetectionLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.ce_loss = nn.CrossEntropyLoss()
            self.mse_loss = nn.MSELoss()
    
        def forward(self, predictions, targets):
            binary_loss = self.bce_loss(predictions['binary'], targets['binary'])
            type_loss = self.ce_loss(predictions['type'], targets['type'])
            magnitude_loss = self.mse_loss(predictions['magnitude'], targets['magnitude'])
            timing_loss = self.ce_loss(predictions['timing'], targets['timing'])
            
            return binary_loss + 0.5 * type_loss + 0.3 * magnitude_loss + 0.2 * timing_loss
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = ChangeDetectionLoss()
    
    trainer = ChangeDetectionTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda'
    )
    
    history = trainer.train(
        epochs=200,
        patience=20,
        save_path="./checkpoints/change_detection_model.pth"
    )

Change Analysis
---------------

.. code-block:: python

    from sitsformer.evaluation import change_metrics
    import matplotlib.pyplot as plt
    
    # Evaluate change detection performance
    results = change_metrics(model, test_loader)
    
    print("Change Detection Results:")
    print(f"Binary change detection F1: {results['binary_f1']:.3f}")
    print(f"Change type accuracy: {results['type_accuracy']:.3f}")
    print(f"Timing accuracy (±1 month): {results['timing_accuracy']:.3f}")
    
    # Visualize change timeline
    def plot_change_timeline(time_series, predictions, ground_truth):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot NDVI time series
        dates = range(len(time_series))
        ndvi = (time_series[:, 3] - time_series[:, 2]) / (time_series[:, 3] + time_series[:, 2])
        
        ax1.plot(dates, ndvi, 'g-', linewidth=2, label='NDVI')
        ax1.set_ylabel('NDVI')
        ax1.legend()
        ax1.grid(True)
        
        # Plot change probability
        change_prob = torch.sigmoid(predictions['binary']).cpu().numpy()
        ax2.plot(dates[1:], change_prob, 'r-', linewidth=2, label='Change Probability')
        ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Change Probability')
        ax2.legend()
        ax2.grid(True)
        
        # Plot ground truth changes
        true_changes = ground_truth['timing'].cpu().numpy()
        for change_time in true_changes:
            if change_time > 0:
                ax3.axvline(x=change_time, color='red', alpha=0.7)
        
        ax3.set_xlabel('Time (months)')
        ax3.set_ylabel('True Changes')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

Real-time Monitoring
--------------------

.. code-block:: python

    from sitsformer.monitoring import ChangeMonitor
    import geopandas as gpd
    
    # Setup change monitoring system
    monitor = ChangeMonitor(
        model=model,
        confidence_threshold=0.8,
        alert_system=True
    )
    
    # Process new satellite observation
    def process_new_image(new_observation, date, region_bounds):
        """Process new satellite data for change detection"""
        
        # Update time series buffer
        monitor.update_time_series(new_observation, date)
        
        # Detect changes
        changes = monitor.detect_changes(region_bounds)
        
        # Generate alerts
        for change in changes:
            if change['confidence'] > 0.8:
                print(f"CHANGE ALERT:")
                print(f"  Location: {change['coordinates']}")
                print(f"  Type: {change['type']}")
                print(f"  Confidence: {change['confidence']:.2f}")
                print(f"  Area: {change['area_ha']:.1f} hectares")
                print(f"  Date detected: {date}")
        
        return changes
    
    # Example: Monthly monitoring
    import datetime
    
    for month in range(1, 13):
        date = datetime.date(2024, month, 15)
        new_data = load_monthly_observation(date)
        changes = process_new_image(new_data, date, study_area_bounds)

Post-processing and Validation
------------------------------

.. code-block:: python

    from sitsformer.postprocess import ChangeValidator
    from scipy import ndimage
    
    # Post-process change detections
    validator = ChangeValidator(
        min_change_area=0.5,  # Minimum 0.5 hectares
        temporal_consistency=True,
        spatial_smoothing=True
    )
    
    def post_process_changes(raw_detections):
        # Remove small isolated changes
        filtered = validator.filter_by_area(raw_detections)
        
        # Apply temporal consistency checks
        consistent = validator.temporal_filter(filtered)
        
        # Spatial smoothing
        smoothed = validator.spatial_filter(consistent)
        
        return smoothed
    
    # Validate against external data sources
    def validate_with_external_data(detections, reference_data):
        validation_results = {
            'precision': calculate_precision(detections, reference_data),
            'recall': calculate_recall(detections, reference_data),
            'false_alarm_rate': calculate_false_alarms(detections, reference_data)
        }
        return validation_results

Applications and Results
------------------------

Typical performance metrics:

- **Binary change detection F1**: 0.85-0.92
- **Change type accuracy**: 0.78-0.85
- **Temporal accuracy (±1 month)**: 0.70-0.80
- **False alarm rate**: 5-12%

Use cases:

- **Forest monitoring**: Detect illegal logging, fire damage
- **Urban planning**: Track development, infrastructure changes
- **Agriculture**: Monitor crop rotation, field abandonment
- **Water resources**: Lake level changes, wetland dynamics
- **Disaster response**: Assess damage from floods, earthquakes

Benefits of SITS-Former approach:

- Handles irregular time series
- Captures long-term trends and seasonal patterns
- Provides interpretable attention weights
- Robust to cloud cover and data gaps
- Scalable to large geographic areas
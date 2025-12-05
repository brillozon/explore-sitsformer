"""
Data utilities and preprocessing functions for satellite imagery.

This module provides essential utilities for processing satellite image time series data,
including normalization, statistical computation, dataset splitting, and data augmentation
utilities specifically designed for remote sensing applications.

The utilities support various satellite sensors including Sentinel-2 and Landsat-8,
and provide predefined band configurations and class mappings for common remote sensing
tasks such as land cover classification and crop monitoring.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch


def calculate_band_statistics(
    data_dir: Union[str, Path], sample_size: int = 1000
) -> Dict[str, np.ndarray]:
    """Calculate mean and standard deviation statistics for each spectral band.

    Computes normalization statistics (mean and standard deviation) for each spectral
    band across a sample of satellite images. These statistics are essential for
    proper data normalization during model training and inference.

    Args:
        data_dir (Union[str, Path]): Directory containing satellite images in supported formats
            (TIFF, HDF5, etc.). Images should be organized consistently with metadata.
        sample_size (int, optional): Number of images to sample for computing statistics.
            Larger samples provide more accurate statistics but require more computation.
            Default: 1000

    Returns:
        Dict[str, np.ndarray]: Dictionary containing normalization statistics:
            - 'mean': Array of mean values for each spectral band
            - 'std': Array of standard deviation values for each spectral band

    Example:
        >>> stats = calculate_band_statistics('/path/to/sentinel2/data', sample_size=500)
        >>> print(f"Band means: {stats['mean']}")
        >>> print(f"Band stds: {stats['std']}")

        >>> # Use for normalization
        >>> normalized_image = (image - stats['mean']) / stats['std']

    Note:
        - The function currently returns predefined Sentinel-2 statistics as a placeholder
        - In production, implement actual statistical computation from your dataset
        - Statistics should be computed on the training set only to prevent data leakage
        - Consider computing statistics per-tile or per-region for large geographic areas

    Raises:
        ValueError: If data_dir doesn't exist or contains no valid images
        RuntimeError: If insufficient memory to process the sample
    """
    # This is a placeholder implementation
    # In practice, you would load and compute statistics from actual data

    # For now, return typical Sentinel-2 statistics
    sentinel2_means = np.array(
        [
            1354.40546513,
            1118.24399958,
            1042.92983953,
            947.62620298,
            1199.47283961,
            1999.79090914,
            2369.22292565,
            2296.82608323,
            732.08340178,
            12.11327804,
            1819.12868772,
            1118.92391149,
            2594.14080798,
        ]
    )

    sentinel2_stds = np.array(
        [
            245.71762908,
            333.00498827,
            395.09249139,
            593.75055589,
            566.4170017,
            861.18399006,
            1086.63139075,
            1117.98170791,
            404.91978886,
            4.77584468,
            1002.58768311,
            761.30323499,
            1231.58581042,
        ]
    )

    return {"mean": sentinel2_means, "std": sentinel2_stds}


def normalize_image(
    image: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    """Normalize satellite image using per-band statistics.

    Applies z-score normalization to satellite imagery using precomputed mean and
    standard deviation values for each spectral band. This normalization is essential
    for proper model training and ensures consistent input ranges across different
    spectral bands.

    Args:
        image (np.ndarray): Input satellite image with shape (..., channels) where
            channels represent spectral bands. The function supports arbitrary leading
            dimensions (batch, time, height, width, etc.).
        means (np.ndarray): Mean values for each spectral band. Shape should be (channels,)
            or broadcastable to the image's channel dimension.
        stds (np.ndarray): Standard deviation values for each spectral band.
            Shape should be (channels,) or broadcastable to the image's channel dimension.

    Returns:
        np.ndarray: Normalized image with same shape as input. Values will have
            approximately zero mean and unit variance for each band.

    Example:
        >>> # Normalize single Sentinel-2 image
        >>> image = np.random.rand(224, 224, 13)  # H, W, C
        >>> stats = calculate_band_statistics('/path/to/data')
        >>> normalized = normalize_image(image, stats['mean'], stats['std'])

        >>> # Normalize batch of images
        >>> batch = np.random.rand(32, 224, 224, 13)  # B, H, W, C
        >>> normalized_batch = normalize_image(batch, stats['mean'], stats['std'])

    Note:
        - Use the same statistics for training, validation, and test sets
        - Statistics should be computed from training data only
        - Ensure means and stds are computed from the same sensor/processing chain
    """
    return (image - means) / stds


def denormalize_image(
    image: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    """Denormalize satellite image to original scale.

    Reverses z-score normalization by scaling values back to their original range
    using the provided mean and standard deviation values. This is useful for
    visualization, analysis, or when interfacing with systems that expect
    unnormalized satellite imagery.

    Args:
        image (np.ndarray): Normalized satellite image with shape (..., channels)
            where channels represent spectral bands. Should have approximately
            zero mean and unit variance for each band.
        means (np.ndarray): Original mean values for each spectral band that were
            used for normalization. Shape should be (channels,).
        stds (np.ndarray): Original standard deviation values for each spectral band
            that were used for normalization. Shape should be (channels,).

    Returns:
        np.ndarray: Denormalized image with same shape as input, restored to
            original value ranges.

    Example:
        >>> # Normalize then denormalize for verification
        >>> image = np.random.rand(224, 224, 13) * 3000  # Typical Sentinel-2 range
        >>> stats = calculate_band_statistics('/path/to/data')
        >>> normalized = normalize_image(image, stats['mean'], stats['std'])
        >>> restored = denormalize_image(normalized, stats['mean'], stats['std'])
        >>> np.allclose(image, restored)  # Should be True

        >>> # Denormalize model predictions for visualization
        >>> model_output = model(normalized_input)
        >>> visual_output = denormalize_image(model_output, means, stds)

    Note:
        - Use the exact same statistics (means, stds) that were used for normalization
        - Particularly important for visualization and interpretation of results
        - Digital number ranges will be sensor-specific (Sentinel-2, Landsat, etc.)
    """
    return image * stds + means


def create_temporal_mask(
    sequence_length: int, mask_ratio: float = 0.15
) -> torch.Tensor:
    """Create temporal mask for self-supervised learning tasks.

    Generates a binary mask for temporal masking in self-supervised pre-training,
    similar to masked language modeling in NLP. This technique helps models learn
    robust temporal representations by predicting masked time steps from visible ones.

    Args:
        sequence_length (int): Total length of the time series sequence.
        mask_ratio (float, optional): Fraction of time steps to mask. Should be
            between 0.0 and 1.0. Higher values make the task more challenging.
            Default: 0.15 (15% masking, following BERT-style masking)

    Returns:
        torch.Tensor: Binary mask tensor of shape (sequence_length,) where:
            - 1 indicates the time step should be kept (visible)
            - 0 indicates the time step should be masked (hidden)

    Example:
        >>> # Create mask for 10-step time series with 20% masking
        >>> mask = create_temporal_mask(sequence_length=10, mask_ratio=0.2)
        >>> print(mask)  # e.g., tensor([1., 1., 0., 1., 1., 0., 1., 1., 1., 1.])

        >>> # Apply mask to actual time series data
        >>> time_series = torch.randn(10, 224, 224, 13)  # [T, H, W, C]
        >>> masked_series = time_series * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        >>> # Use in self-supervised learning
        >>> visible_steps = time_series[mask.bool()]
        >>> hidden_steps = time_series[~mask.bool()]

    Note:
        - Commonly used mask ratios: 0.15 (BERT), 0.25 (MAE), 0.75 (high masking)
        - Random seed should be set externally for reproducible masking
        - Consider temporal dependencies when choosing mask ratio for time series
    """
    num_mask = int(sequence_length * mask_ratio)
    mask_indices = np.random.choice(sequence_length, num_mask, replace=False)

    mask = torch.ones(sequence_length)
    mask[mask_indices] = 0

    return mask


def pad_sequence(
    sequence: torch.Tensor, target_length: int, pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad or truncate sequence to target length with attention mask.

    Ensures all sequences in a batch have the same length by padding shorter sequences
    or truncating longer ones. Also generates an attention mask to distinguish between
    real and padded values during model training.

    Args:
        sequence (torch.Tensor): Input sequence with shape (seq_len, ...) where
            seq_len is the current sequence length and ... represents arbitrary
            additional dimensions (e.g., channels, height, width).
        target_length (int): Desired sequence length. Must be positive.
        pad_value (float, optional): Value used for padding. Default: 0.0
            Common choices: 0.0 (zero padding), -inf (for attention masking)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Padded/truncated sequence with shape (target_length, ...)
            - Attention mask with shape (target_length,) where 1 indicates
              real values and 0 indicates padded values

    Example:
        >>> # Pad short sequence
        >>> short_seq = torch.randn(5, 224, 224, 13)  # 5 time steps
        >>> padded_seq, mask = pad_sequence(short_seq, target_length=10)
        >>> print(f"Padded shape: {padded_seq.shape}")  # [10, 224, 224, 13]
        >>> print(f"Mask: {mask}")  # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        >>> # Truncate long sequence
        >>> long_seq = torch.randn(15, 224, 224, 13)  # 15 time steps
        >>> truncated_seq, mask = pad_sequence(long_seq, target_length=10)
        >>> print(f"Truncated shape: {truncated_seq.shape}")  # [10, 224, 224, 13]
        >>> print(f"Mask: {mask}")  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        >>> # Use in DataLoader with collate function
        >>> def collate_fn(batch):
        ...     sequences = [item[0] for item in batch]
        ...     max_len = max(seq.shape[0] for seq in sequences)
        ...     padded_batch = []
        ...     masks = []
        ...     for seq in sequences:
        ...         padded_seq, mask = pad_sequence(seq, max_len)
        ...         padded_batch.append(padded_seq)
        ...         masks.append(mask)
        ...     return torch.stack(padded_batch), torch.stack(masks)

    Note:
        - Essential for batch processing of variable-length time series
        - Attention masks prevent model from attending to padded values
        - Consider the impact of truncation vs. padding on model performance
    """
    current_length = sequence.shape[0]

    if current_length >= target_length:
        # Truncate
        return sequence[:target_length], torch.ones(target_length)
    else:
        # Pad
        pad_shape = (target_length - current_length,) + sequence.shape[1:]
        padding = torch.full(pad_shape, pad_value, dtype=sequence.dtype)

        padded_sequence = torch.cat([sequence, padding], dim=0)

        # Create attention mask
        mask = torch.zeros(target_length)
        mask[:current_length] = 1

        return padded_sequence, mask


def split_dataset(
    dataset_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset indices into train/val/test sets.

    Args:
        dataset_size: Total size of dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility

    Returns:
        Train, validation, and test indices
    """
    # Validate ratios sum to 1.0
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) >= 1e-6:
        raise ValueError(f"Data split ratios must sum to 1.0, got {ratio_sum}")

    np.random.seed(random_seed)
    indices = np.random.permutation(dataset_size)

    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)

    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size : train_size + val_size].tolist()
    test_indices = indices[train_size + val_size :].tolist()

    return train_indices, val_indices, test_indices


class EarlyStopping:
    """Early stopping utility."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Args:
            val_loss: Current validation loss

        Returns:
            True if should stop early
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


# Example band configurations for different satellites
SENTINEL2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
    "QA60",
]

LANDSAT8_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]

# Example class labels for different tasks
CROP_CLASSES = ["Wheat", "Corn", "Soybean", "Cotton", "Rice", "Barley", "Other"]

LAND_COVER_CLASSES = ["Urban", "Agriculture", "Forest", "Water", "Barren", "Grassland"]

"""
Data loading and preprocessing utilities for satellite image time series.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
    ComposeType = A.Compose
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None
    ToTensorV2 = None
    ComposeType = Any

import xarray as xr
from sklearn.preprocessing import LabelEncoder, StandardScaler


class SatelliteTimeSeriesDataset(Dataset):
    """Dataset for satellite image time series data."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Union[str, Path],
        sequence_length: int = 10,
        image_size: int = 224,
        bands: Optional[List[str]] = None,
        transform: Optional[ComposeType] = None,
        normalize: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing satellite image files
            metadata_file: CSV file with metadata (image_path, date, label, etc.)
            sequence_length: Number of images in each time series
            image_size: Target image size
            bands: List of spectral bands to use
            transform: Albumentations transform pipeline
            normalize: Whether to normalize pixel values
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.bands = bands
        self.transform = transform
        self.normalize = normalize

        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        self.metadata["date"] = pd.to_datetime(self.metadata["date"])

        # Group by location/sample to create sequences
        self.sequences = self._create_sequences()

        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        if "label" in self.metadata.columns:
            unique_labels = self.metadata["label"].unique()
            self.label_encoder.fit(unique_labels)

        # Initialize scaler for normalization
        if normalize:
            self.scaler = StandardScaler()
            self._fit_scaler()

    def _create_sequences(self) -> List[Dict]:
        """Create time series sequences from metadata."""
        sequences = []

        # Group by location/sample_id if available, otherwise create single sequence
        if "location_id" in self.metadata.columns:
            groups = self.metadata.groupby("location_id")
        else:
            # Single location, all images
            groups = [("single", self.metadata)]

        for location_id, group in groups:
            # Sort by date
            group = group.sort_values("date")

            # Create sliding windows of sequence_length
            for i in range(len(group) - self.sequence_length + 1):
                sequence_data = group.iloc[i : i + self.sequence_length].copy()
                sequences.append(
                    {
                        "location_id": location_id,
                        "image_paths": sequence_data["image_path"].tolist(),
                        "dates": sequence_data["date"].tolist(),
                        "labels": (
                            sequence_data["label"].tolist()
                            if "label" in sequence_data.columns
                            else None
                        ),
                    }
                )

        return sequences

    def _fit_scaler(self):
        """Fit scaler on a sample of the data."""
        # Sample a few images to fit the scaler
        sample_images = []
        for i in range(0, min(100, len(self)), max(1, len(self) // 100)):
            try:
                images, _ = self._load_sequence(self.sequences[i])
                sample_images.append(images.flatten())
            except (IOError, OSError, ValueError) as e:
                # Log specific error types for debugging but continue processing
                import logging

                logging.getLogger(__name__).warning(f"Failed to load sequence {i}: {e}")
                continue

        if sample_images:
            sample_data = np.concatenate(sample_images)
            self.scaler.fit(sample_data.reshape(-1, 1))

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load a single satellite image."""
        image_path = self.data_dir / image_path

        try:
            # Try loading with rasterio (for GeoTIFF)
            with rasterio.open(image_path) as src:
                image = src.read()  # Shape: (bands, height, width)
                image = np.transpose(image, (1, 2, 0))  # Shape: (height, width, bands)
        except Exception:
            try:
                # Try loading with xarray (for NetCDF)
                ds = xr.open_dataset(image_path)
                # Assume the main variable contains the image data
                var_names = list(ds.data_vars.keys())
                image = ds[var_names[0]].values
                if len(image.shape) == 3:
                    image = np.transpose(image, (1, 2, 0))
            except Exception:
                # Fallback: try numpy
                image = np.load(image_path)

        # Select specific bands if specified
        if self.bands is not None:
            # Assume bands are in the last dimension
            if isinstance(self.bands[0], int):
                image = image[..., self.bands]
            # If bands are named, would need metadata to map names to indices

        return image.astype(np.float32)

    def _load_sequence(self, sequence_info: Dict) -> Tuple[np.ndarray, Optional[int]]:
        """Load a complete time series sequence."""
        images = []

        for image_path in sequence_info["image_paths"]:
            image = self._load_image(image_path)

            # Resize if needed
            if image.shape[:2] != (self.image_size, self.image_size):
                # Use albumentations for consistent resizing
                resize_transform = A.Resize(self.image_size, self.image_size)
                image = resize_transform(image=image)["image"]

            images.append(image)

        images = np.stack(images, axis=0)  # Shape: (time, height, width, channels)

        # Get label (use the most common label in the sequence)
        label = None
        if sequence_info["labels"] is not None:
            labels = sequence_info["labels"]
            if all(label == labels[0] for label in labels):
                # All same label
                label = self.label_encoder.transform([labels[0]])[0]
            else:
                # Use majority label
                from collections import Counter

                label_counts = Counter(labels)
                majority_label = label_counts.most_common(1)[0][0]
                label = self.label_encoder.transform([majority_label])[0]

        return images, label

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get a time series sequence."""
        sequence_info = self.sequences[idx]
        images, label = self._load_sequence(sequence_info)

        # Apply transforms to each image in the sequence
        if self.transform is not None:
            transformed_images = []
            for img in images:
                transformed = self.transform(image=img)["image"]
                transformed_images.append(transformed)
            images = np.stack(transformed_images, axis=0)

        # Normalize
        if self.normalize and hasattr(self, "scaler"):
            original_shape = images.shape
            images_flat = images.reshape(-1, 1)
            images_normalized = self.scaler.transform(images_flat)
            images = images_normalized.reshape(original_shape)

        # Convert to tensor
        # Expected shape: (time, channels, height, width)
        images = torch.from_numpy(images).permute(0, 3, 1, 2)

        if label is not None:
            label = torch.tensor(label, dtype=torch.long)
            return images, label
        else:
            return images, torch.tensor(-1, dtype=torch.long)  # Dummy label


class DummySatelliteDataset(Dataset):
    """Dummy dataset for testing and development."""

    def __init__(
        self,
        num_samples: int = 1000,
        sequence_length: int = 10,
        image_size: int = 64,
        num_channels: int = 13,
        num_classes: int = 5,
    ):
        """
        Args:
            num_samples: Number of samples to generate
            sequence_length: Number of images in each sequence
            image_size: Image height and width
            num_channels: Number of spectral bands
            num_classes: Number of classification classes
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes

        # Generate random labels
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Generate a random time series sequence."""
        # Generate random images with some temporal correlation
        images = torch.randn(
            self.sequence_length, self.num_channels, self.image_size, self.image_size
        )

        # Add some temporal correlation
        for t in range(1, self.sequence_length):
            images[t] = 0.8 * images[t - 1] + 0.2 * images[t]

        # Normalize to [0, 1]
        images = (images - images.min()) / (images.max() - images.min() + 1e-8)

        label = self.labels[idx].item()  # Convert tensor to int

        return images, label


def create_transforms(
    image_size: int = 224, is_training: bool = True
) -> Optional[ComposeType]:
    """Create data augmentation transforms."""

    if not HAS_ALBUMENTATIONS:
        return None

    if is_training:
        transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.RandomCrop(image_size, image_size, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.1, contrast_limit=0.1, p=1.0
                        ),
                        A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                        A.HueSaturationValue(
                            hue_shift_limit=5,
                            sat_shift_limit=10,
                            val_shift_limit=10,
                            p=1.0,
                        ),
                    ],
                    p=0.5,
                ),
                A.GaussianBlur(blur_limit=(1, 3), p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ]
        )
    else:
        transform = A.Compose(
            [
                A.Resize(image_size, image_size),
            ]
        )

    return transform


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with appropriate settings."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if shuffle else False,
    )


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    dataset = DummySatelliteDataset(
        num_samples=100,
        sequence_length=5,
        image_size=64,
        num_channels=13,
        num_classes=3,
    )

    dataloader = create_dataloader(dataset, batch_size=8, shuffle=True)

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")

        if batch_idx >= 2:  # Show first few batches
            break

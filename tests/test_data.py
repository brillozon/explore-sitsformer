"""Test data utilities."""

from sitsformer.data import DummySatelliteDataset, create_dataloader


def test_dummy_dataset_creation():
    """Test dummy dataset creation."""
    dataset = DummySatelliteDataset(
        num_samples=100,
        sequence_length=24,
        image_size=64,
        num_channels=10,
        num_classes=5,
    )

    assert len(dataset) == 100

    # Test sample retrieval
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2

    images, label = sample
    assert images.shape == (24, 10, 64, 64)  # seq_len, channels, height, width
    assert isinstance(label, int)
    assert 0 <= label < 5


def test_dataloader_creation():
    """Test dataloader creation."""
    dataset = DummySatelliteDataset(
        num_samples=50, sequence_length=12, image_size=32, num_channels=8, num_classes=3
    )

    dataloader = create_dataloader(
        dataset, batch_size=4, shuffle=True, num_workers=0  # Use 0 for testing
    )

    # Test batch retrieval
    batch = next(iter(dataloader))
    assert isinstance(batch, (tuple, list))  # DataLoader can return either
    assert len(batch) == 2

    images, labels = batch
    assert images.shape == (4, 12, 8, 32, 32)  # batch, seq_len, channels, height, width
    assert labels.shape == (4,)
    assert all(0 <= label < 3 for label in labels)


if __name__ == "__main__":
    test_dummy_dataset_creation()
    test_dataloader_creation()
    print("All data tests passed!")

# Exploring SITS-Former: Satellite Image Time Series Transformer

A PyTorch implementation of SITS-Former, a transformer-based model for satellite image time series analysis and classification.

## Overview

SITS-Former is designed to process and classify satellite image time series data using a transformer architecture. The model can handle temporal sequences of satellite imagery and perform various classification tasks such as land cover classification, crop type identification, and change detection.

The [International Journal of Applied Earth Observation and Geoinformation ](https://doi.org/10.1016/j.jag.2021.102651) paper describing the pre-trained model is also included in the `docs` directory [here](./docs/SITS-Former.pdf).

## Features

- **Transformer Architecture**: Multi-head attention mechanism for temporal modeling
- **Flexible Data Pipeline**: Support for various satellite imagery formats
- **Comprehensive Training Framework**: Full training pipeline with logging and checkpointing
- **Evaluation Tools**: Detailed metrics and visualization capabilities
- **Configuration System**: YAML-based configuration management
- **Command-line Interface**: Easy-to-use scripts for training, evaluation, and inference

## Installation

### Requirements

- Python 3.12+
- Poetry (for dependency management)
- PyTorch 2.0+

**Note**: If Python 3.12 is not available on your system, you can install it using:
- **macOS**: `brew install python@3.12` or download from [python.org](https://python.org)
- **Ubuntu/Debian**: `sudo apt install python3.12`
- **Using pyenv**: `pyenv install 3.12.7 && pyenv global 3.12.7`

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd sitsformer
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies and create virtual environment:
```bash
poetry install
```

4. Activate the Poetry shell:
```bash
poetry shell
```

## Project Structure

```
sitsformer/
├── src/sitsformer/        # Main package
│   ├── models/           # Model architectures
│   │   ├── __init__.py
│   │   └── sits_former.py
│   ├── data/            # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── training/        # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/      # Evaluation metrics and tools
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── utils/          # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── helpers.py
│   └── scripts/        # Training and evaluation scripts
│       ├── __init__.py
│       ├── train.py
│       ├── evaluate.py
│       └── inference.py
├── configs/            # Configuration files
│   ├── default.yaml
│   ├── small_model.yaml
│   └── large_model.yaml
├── notebooks/          # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb
│   ├── model_architecture.ipynb
│   ├── training_experiments.ipynb
│   └── results_analysis.ipynb
├── tests/             # Unit tests
├── pyproject.toml     # Poetry configuration
└── README.md
```

## Quick Start

### 1. Training with Dummy Data

Start with a small model using dummy data for testing:

```bash
poetry run python -m sitsformer.scripts.train --preset small --dummy-data --epochs 5
```

Or use the Poetry script:

```bash
poetry run sitsformer-train --preset small --dummy-data --epochs 5
```

### 2. Training with Custom Configuration

```bash
poetry run sitsformer-train --config configs/default.yaml --data-path /path/to/your/data
```

### 3. Evaluation

```bash
poetry run sitsformer-evaluate --checkpoint checkpoints/best_model.pt --dummy-data
```

### 4. Inference

```bash
poetry run sitsformer-inference --checkpoint checkpoints/best_model.pt --dummy-data
```

## Configuration

The project uses YAML configuration files for managing experiments. Key configuration sections include:

### Model Configuration
```yaml
model:
  in_channels: 10        # Number of input channels (bands)
  num_classes: 10        # Number of output classes
  patch_size: 16         # Patch size for image processing
  sequence_length: 24    # Length of time series
  hidden_dim: 256        # Hidden dimension
  num_heads: 8           # Number of attention heads
  num_layers: 6          # Number of transformer layers
  dropout: 0.1           # Dropout rate
```

### Training Configuration
```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  optimizer: adamw
  scheduler: cosine
  mixed_precision: true
```

### Data Configuration
```yaml
data:
  sequence_length: 24
  image_size: 64
  num_channels: 10
  num_classes: 10
  augmentation: true
```

## Model Architecture

SITS-Former consists of several key components:

1. **Patch Embedding**: Converts image patches to embeddings
2. **Positional Encoding**: Adds temporal and spatial position information
3. **Transformer Encoder**: Multi-layer transformer with self-attention
4. **Temporal Aggregation**: Aggregates temporal information
5. **Classification Head**: Final classification layer

### Key Features:
- Multi-head self-attention for temporal modeling
- Patch-based image processing
- Flexible architecture for different input sizes
- Support for variable sequence lengths

## Data Format

The model expects satellite image time series data in the following format:

- **Input Shape**: `(batch_size, sequence_length, channels, height, width)`
- **Target Shape**: `(batch_size,)` for classification

### Supported Data Sources:
- Sentinel-2 imagery
- Landsat imagery
- Custom satellite data
- Dummy data for testing

## Training

### Command Line Options

```bash
poetry run sitsformer-train [OPTIONS]
# or
poetry run python -m sitsformer.scripts.train [OPTIONS]

Options:
  --config PATH              Path to configuration file
  --preset {small,default,large}  Use predefined configuration
  --data-path PATH           Path to training data
  --dummy-data              Use dummy dataset for testing
  --output-dir PATH         Directory for outputs
  --device {auto,cuda,cpu}  Device to use for training
  --resume PATH             Resume from checkpoint
```

### Training Features:
- Automatic mixed precision training
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Progress tracking with tqdm
- Experiment logging with wandb (optional)

## Evaluation

### Metrics

The evaluation framework provides comprehensive metrics:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Per-class Metrics**: Individual class performance
- **Confusion Matrix**: Detailed confusion analysis
- **ROC Curves**: Receiver operating characteristic analysis

### Visualization

- Confusion matrix plots
- Per-class performance charts
- Training loss curves
- Learning rate schedules

## Notebooks

The project includes Jupyter notebooks for exploration and analysis:

1. **data_exploration.ipynb**: Explore and visualize satellite data
2. **model_architecture.ipynb**: Understand the transformer architecture
3. **training_experiments.ipynb**: Run and compare training experiments
4. **results_analysis.ipynb**: Analyze and visualize results

## Examples

### Example 1: Land Cover Classification

```python
from src.models import create_sits_former
from src.utils import load_config

# Load configuration
config = load_config("configs/default.yaml")

# Create model
model = create_sits_former(config['model'])

# Train model
# ... (see training script)
```

### Example 2: Custom Dataset

```python
from src.data import SatelliteTimeSeriesDataset

# Create custom dataset
dataset = SatelliteTimeSeriesDataset(
    data_path="path/to/data",
    sequence_length=24,
    transform=create_transforms(train=True)
)
```

## Advanced Usage

### Custom Model Configuration

Create custom model configurations by modifying the YAML files:

```yaml
model:
  in_channels: 13         # Sentinel-2 bands
  num_classes: 20         # Land cover classes
  patch_size: 8           # Smaller patches
  hidden_dim: 512         # Larger model
  num_layers: 12          # Deeper network
```

### Hyperparameter Tuning

Use the configuration system for systematic hyperparameter tuning:

```bash
# Create multiple configurations
poetry run sitsformer-train --config configs/experiment_1.yaml
poetry run sitsformer-train --config configs/experiment_2.yaml
```

### Multi-GPU Training

The training script supports multi-GPU training:

```bash
poetry run sitsformer-train --config configs/default.yaml --device cuda
```

## Performance

### Benchmarks

Model performance depends on configuration and data:

- **Small Model**: ~1M parameters, suitable for quick experiments
- **Default Model**: ~10M parameters, good balance of performance and speed
- **Large Model**: ~50M parameters, maximum performance

### Optimization Tips

1. **Use Mixed Precision**: Enable `mixed_precision: true` for faster training
2. **Batch Size**: Increase batch size for better GPU utilization
3. **Data Loading**: Use multiple workers for data loading
4. **Caching**: Enable data caching for repeated experiments

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Import Errors**: Ensure all dependencies are installed
3. **Data Loading**: Check data format and paths

### Debug Mode

Enable debug logging for detailed information:

```bash
poetry run sitsformer-train --config configs/default.yaml --debug
```

## Development

### Setting up Development Environment

1. Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd sitsformer
poetry install --with dev,experiment,docs
```

2. Install pre-commit hooks:

```bash
poetry run pre-commit install
```

3. Run tests:

```bash
poetry run pytest
```

4. Format code:

```bash
poetry run black .
poetry run isort .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{explore-sits-former,
  title={Exploring SITS-Former: Satellite Image Time Series Transformer},
  author={Mike Martinez},
  year={2025},
  url={[Repository URL]}
}

@article{yuan2022sits,
  title={SITS-Former: A pre-trained spatio-spectral-temporal representation model for Sentinel-2 time series classification},
  author={Yuan, Yuan and Lin, Lei and Liu, Qingshan and Hang, Renlong and Zhou, Zeng-Guang},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={106},
  pages={102651},
  year={2022},
  doi = {10.1016/j.jag.2021.102651},
  publisher={Elsevier}
}
```

## Acknowledgments

- Based on the original SITS-Former paper
- Built with PyTorch and modern deep learning practices
- Thanks to the satellite imagery and remote sensing community

## Contact

For questions and support, please create an issue.

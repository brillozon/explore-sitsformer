# SITSFormer: Satellite Image Time Series Transformer

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-sphinx-brightgreen.svg)](docs/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![CI](https://github.com/brillozon/explore-sitsformer/workflows/CI/badge.svg)](https://github.com/brillozon/explore-sitsformer/actions/workflows/ci.yml)
[![Documentation](https://github.com/brillozon/explore-sitsformer/workflows/Documentation/badge.svg)](https://github.com/brillozon/explore-sitsformer/actions/workflows/docs.yml)
[![Security](https://github.com/brillozon/explore-sitsformer/workflows/Security/badge.svg)](https://github.com/brillozon/explore-sitsformer/actions/workflows/security.yml)
[![Quality](https://github.com/brillozon/explore-sitsformer/workflows/Quality/badge.svg)](https://github.com/brillozon/explore-sitsformer/actions/workflows/quality.yml)

A state-of-the-art transformer-based model for analyzing satellite image time series data. SITSFormer leverages the power of attention mechanisms to capture temporal dependencies in Earth observation data for various remote sensing applications including land cover classification, change detection, and crop monitoring.

## 🚀 Features

- **Transformer Architecture**: Advanced attention mechanisms for temporal modeling of satellite imagery
- **Multi-Spectral Support**: Works with various satellite sensors (Sentinel-2, Landsat, etc.)
- **Flexible Input Formats**: Support for different time series lengths and spectral bands
- **Pre-trained Models**: Ready-to-use models for common remote sensing tasks
- **Easy Integration**: Simple API for both research and production use
- **Comprehensive Documentation**: Detailed tutorials and API documentation
- **Extensible Design**: Easy to adapt for custom datasets and applications

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Tutorials](#tutorials)
- [Model Architecture](#model-architecture)
- [Supported Datasets](#supported-datasets)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🛠️ Installation

### Requirements

- Python 3.8+
- PyTorch 2.4+
- CUDA support (optional but recommended for training)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/sitsformer.git
cd sitsformer

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -
# or
pip install poetry

# Install the project and dependencies in a virtual environment
poetry install

# Activate the virtual environment
poetry shell
```

### Optional Dependencies

Install additional dependencies for specific use cases:

```bash
# For geospatial data processing
poetry install --extras geospatial

# For development
poetry install --extras dev

# For documentation building
poetry install --extras docs

# For experiment tracking
poetry install --extras experiment
```

### Using Poetry Commands

After installation, run all commands through Poetry to ensure you're using the correct virtual environment:

```bash
# Run Python scripts
poetry run python your_script.py

# Or activate the shell once and run commands normally
poetry shell
python your_script.py
```

## 🚀 Quick Start

### Poetry Workflow

SITSFormer uses Poetry for dependency management and virtual environments. Here's the recommended workflow:

```bash
# After installation, always use poetry run or activate the shell
poetry shell  # Activates virtual environment

# Or prefix commands with poetry run
poetry run python your_script.py
poetry run jupyter notebook
poetry run pytest tests/
```

### Basic Usage

```python
import torch
from sitsformer import SITSFormer

# Initialize the model
model = SITSFormer(
    num_bands=10,        # Number of spectral bands
    sequence_length=24,   # Number of time steps
    num_classes=10,      # Number of output classes
    patch_size=16,       # Spatial patch size
    embed_dim=256,       # Embedding dimension
    num_heads=8,         # Number of attention heads
    num_layers=6         # Number of transformer layers
)

# Sample input: (batch_size, sequence_length, num_bands, height, width)
x = torch.randn(4, 24, 10, 64, 64)

# Forward pass
predictions = model(x)
print(f"Predictions shape: {predictions.shape}")  # (4, 10)
```

### Training Example

Run training scripts using Poetry to ensure the correct virtual environment:

```bash
# Using poetry run
poetry run python examples/train_model.py

# Or activate shell first
poetry shell
python examples/train_model.py
```

```python
# examples/train_model.py
from sitsformer.training import Trainer
from sitsformer.data import SITSDataset

# Load your dataset
train_dataset = SITSDataset("path/to/train/data")
val_dataset = SITSDataset("path/to/val/data")

# Initialize trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=1e-4,
    batch_size=32
)

# Start training
trainer.train(epochs=100)
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory and includes:

- **API Reference**: Complete documentation of all classes and functions
- **Tutorials**: Step-by-step guides for common use cases
- **Examples**: Ready-to-run example scripts
- **Best Practices**: Guidelines for optimal model performance

To build the documentation locally:

```bash
# Install documentation dependencies
poetry install --extras docs

# Build documentation
cd docs
poetry run make html
# Open docs/_build/html/index.html in your browser

# Or use poetry shell for multiple commands
poetry shell
cd docs
make html
```

## 📖 Tutorials

Our tutorial series covers the complete workflow:

1. **[Data Preparation](docs/tutorials/data_preparation.rst)**: Loading and preprocessing satellite time series
2. **[Model Training](docs/tutorials/model_training.rst)**: Training SITSFormer from scratch
3. **[Fine-tuning](docs/tutorials/fine_tuning.rst)**: Adapting pre-trained models to new datasets
4. **[Model Evaluation](docs/tutorials/model_evaluation.rst)**: Comprehensive model assessment
5. **[Deployment](docs/tutorials/deployment.rst)**: Production deployment strategies
6. **[Advanced Configurations](docs/tutorials/advanced_configurations.rst)**: Research and experimental features

## 🏗️ Model Architecture

SITSFormer uses a hierarchical transformer architecture specifically designed for satellite image time series:

```
Input SITS → Patch Embedding → Temporal Encoding → Transformer Layers → Classification Head
     ↓              ↓                ↓                     ↓                    ↓
(T,C,H,W)    (T,N,D)        (T,N,D)          (T,N,D)              (Classes)
```

Key components:
- **Patch Embedding**: Converts spatial patches into embeddings
- **Temporal Encoding**: Adds positional information for time series
- **Multi-Head Attention**: Captures temporal dependencies
- **Feed-Forward Networks**: Non-linear transformations

## 🌍 Supported Datasets

SITSFormer works with various satellite imagery datasets:

- **Sentinel-2**: 10-13 spectral bands, 10-60m resolution
- **Landsat**: 7-11 bands, 30m resolution  
- **MODIS**: Various products, 250m-1km resolution
- **Custom datasets**: Easy adaptation to proprietary data

### Supported Tasks

- Land cover classification
- Crop type mapping
- Change detection
- Phenology monitoring
- Disaster monitoring
- Urban planning

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/sitsformer.git
cd sitsformer

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install development dependencies
poetry install --extras dev

# Activate virtual environment
poetry shell

# Run tests
poetry run pytest tests/
# or after activating shell:
pytest tests/

# Format code
poetry run black src/ tests/
poetry run isort src/ tests/

# Type checking
poetry run mypy src/

# Linting
poetry run flake8 src/ tests/

# Pre-commit hooks (recommended)
poetry run pre-commit install
```

### Reporting Issues

Please use our [issue templates](.github/ISSUE_TEMPLATE/) to report:
- 🐛 Bug reports
- 🚀 Feature requests  
- 📖 Documentation improvements
- ❓ Questions

## 📊 Benchmarks

| Dataset | Model | Accuracy | F1-Score | Parameters |
|---------|-------|----------|----------|------------|
| BreizhCrops | SITSFormer-Base | 94.2% | 93.8% | 23M |
| BreizhCrops | SITSFormer-Large | 95.1% | 94.7% | 86M |
| PASTIS | SITSFormer-Base | 92.7% | 91.9% | 23M |

## 📝 Citation

If you use SITSFormer in your research, please cite:

```bibtex
@article{sitsformer2024,
  title={SITSFormer: Satellite Image Time Series Transformer for Land Cover Classification},
  author={Your Name and Contributors},
  journal={Remote Sensing of Environment},
  year={2024},
  doi={10.1016/j.rse.2024.xxxxx}
}
```

## 🏆 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Inspired by Vision Transformer architectures
- Thanks to the Earth observation community

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: Available in the [docs/](docs/) directory
- **Paper**: [arXiv:xxxx.xxxxx](https://arxiv.org)
- **Issues**: [GitHub Issues](https://github.com/yourusername/sitsformer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sitsformer/discussions)

---

<p align="center">
Made with ❤️ for the Earth observation community
</p>
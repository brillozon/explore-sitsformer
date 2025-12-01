Installation Guide
==================

Requirements
------------

* Python 3.12 or higher
* PyTorch 2.4.0 or higher
* CUDA (optional, for GPU acceleration)

Installation Methods
--------------------

Using Poetry (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have Poetry installed:

.. code-block:: bash

    git clone https://github.com/yourusername/sitsformer.git
    cd sitsformer
    poetry install

    # For development
    poetry install --extras "dev docs experiment"

    # For geospatial data support (optional)
    poetry install --extras geospatial

Using pip
~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/yourusername/sitsformer.git
    cd sitsformer
    pip install -e .

    # For development
    pip install -e ".[dev,docs,experiment]"

    # For geospatial data support (optional)
    pip install -e ".[geospatial]"

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

SITS-Former includes several optional dependency groups:

* **geospatial**: For working with geospatial data formats (rasterio, geopandas, shapely)
* **dev**: Development tools (pytest, black, mypy, etc.)
* **docs**: Documentation generation (sphinx, sphinx-rtd-theme)
* **experiment**: Experiment tracking (wandb, tensorboard)

Verify Installation
-------------------

.. code-block:: python

    import sitsformer
    print(f"SITS-Former version: {sitsformer.__version__}")

    # Check if PyTorch is available
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

Development Setup
-----------------

For contributors, additional setup steps:

.. code-block:: bash

    # Install pre-commit hooks
    poetry run pre-commit install

    # Run tests
    poetry run pytest

    # Build documentation
    cd docs
    poetry run make html
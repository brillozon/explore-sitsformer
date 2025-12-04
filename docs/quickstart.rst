Quick Start Guide
=================

This guide will help you get started with SITS-Former quickly.

Basic Usage
-----------

Loading and Processing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sitsformer.data import SITSDataLoader
    from sitsformer.models import SITSFormer
    
    # Load your satellite image time series data
    dataloader = SITSDataLoader(
        data_path="path/to/your/data",
        batch_size=32,
        sequence_length=10
    )
    
    # Create and configure the model
    model = SITSFormer(
        input_dim=13,  # Number of spectral bands
        num_classes=10,  # Number of land cover classes
        d_model=128,
        nhead=8,
        num_layers=6
    )

Training a Model
~~~~~~~~~~~~~~~~

.. code-block:: python

    from sitsformer.training import Trainer
    
    trainer = Trainer(
        model=model,
        train_loader=dataloader,
        val_loader=val_dataloader,
        learning_rate=1e-4,
        epochs=100
    )
    
    # Start training
    trainer.train()

Making Predictions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load trained model
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(input_sequences)

Command Line Interface
----------------------

SITS-Former provides convenient command-line tools:

Training
~~~~~~~~

.. code-block:: bash

    # Train with default configuration
    sitsformer-train --config configs/default.yaml

    # Train with custom parameters
    sitsformer-train --data_path /path/to/data --batch_size 64 --epochs 50

Evaluation
~~~~~~~~~~

.. code-block:: bash

    # Evaluate model performance
    sitsformer-evaluate --model_path checkpoints/best_model.pth --test_data /path/to/test

Inference
~~~~~~~~~

.. code-block:: bash

    # Run inference on new data
    sitsformer-inference --model_path checkpoints/best_model.pth --input_data /path/to/new/data

Configuration Files
-------------------

SITS-Former uses YAML configuration files for reproducible experiments:

.. code-block:: yaml

    # Example config.yaml
    model:
      input_dim: 13
      num_classes: 10
      d_model: 128
      nhead: 8
      num_layers: 6
    
    training:
      batch_size: 32
      learning_rate: 1e-4
      epochs: 100
      optimizer: "adam"
    
    data:
      sequence_length: 10
      normalize: true
      augment: false

Next Steps
----------

* Check out the :doc:`tutorials/index` for detailed examples
* Explore the :doc:`api/modules` for complete API reference
* See the :doc:`examples/index` for real-world use cases
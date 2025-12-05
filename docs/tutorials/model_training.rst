Model Training with SITS-Former
================================

This tutorial covers the complete training pipeline for SITS-Former models, from basic training setup to advanced techniques like mixed precision training, distributed training, and hyperparameter optimization.

Overview
--------

Training a SITS-Former model involves several key components:

1. **Model Configuration** - Setting up architecture parameters
2. **Training Setup** - Optimizer, scheduler, and loss function configuration
3. **Training Loop** - Core training and validation procedures
4. **Monitoring** - Tracking metrics and experiment logging
5. **Advanced Techniques** - Mixed precision, gradient clipping, etc.

Prerequisites
-------------

Before starting training, ensure you have:

- Prepared dataset (see :doc:`data_preparation`)
- GPU with sufficient memory (8GB+ recommended)
- Proper environment setup with required dependencies

Basic Training Setup
---------------------

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    from sitsformer.models import SITSFormer, create_sits_former
    from sitsformer.training import SITSFormerTrainer, create_optimizer, create_scheduler
    
    # Method 1: Direct model creation
    model = SITSFormer(
        img_size=224,           # Input image size
        patch_size=16,          # Patch size for vision transformer
        in_channels=13,         # Sentinel-2 spectral bands
        num_classes=10,         # Number of land cover classes
        embed_dim=256,          # Embedding dimension
        num_layers=8,           # Number of transformer layers
        num_heads=8,            # Number of attention heads
        mlp_ratio=4,            # MLP expansion ratio
        dropout=0.1,            # Dropout rate
        max_seq_len=20          # Maximum sequence length
    )
    
    # Method 2: Configuration-based creation
    model_config = {
        'img_size': 224,
        'patch_size': 16,
        'in_channels': 13,
        'num_classes': 10,
        'embed_dim': 256,
        'num_layers': 8,
        'num_heads': 8,
        'dropout': 0.1
    }
    model = create_sits_former(model_config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

Optimizer and Scheduler Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sitsformer.training import create_optimizer, create_scheduler
    
    # Create optimizer
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_type='adamw',      # AdamW optimizer
        lr=1e-4,                    # Learning rate
        weight_decay=1e-2,          # L2 regularization
        betas=(0.9, 0.999)          # Adam beta parameters
    )
    
    # Create learning rate scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type='cosine',     # Cosine annealing
        num_epochs=100,             # Total training epochs
        warmup_epochs=5,            # Warmup period
        min_lr=1e-6                 # Minimum learning rate
    )
    
    # Alternative: Step-based scheduler
    step_scheduler = create_scheduler(
        optimizer,
        scheduler_type='step',
        step_size=30,               # Reduce LR every 30 epochs
        gamma=0.1                   # Multiply LR by 0.1
    )
    
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Scheduler: {scheduler.__class__.__name__}")

Loss Function Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sitsformer.training import create_criterion
    import torch.nn as nn
    
    # Method 1: Standard cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    
    # Method 2: Weighted loss for imbalanced datasets
    # Compute class weights from training data
    def compute_class_weights(train_loader):
        \"\"\"Compute class weights for imbalanced datasets.\"\"\"
        class_counts = torch.zeros(num_classes)
        total_samples = 0
        
        for batch in train_loader:
            labels = batch['label']
            for label in labels:
                class_counts[label] += 1
                total_samples += 1
        
        # Inverse frequency weighting
        class_weights = total_samples / (num_classes * class_counts)
        return class_weights
    
    class_weights = compute_class_weights(train_loader)
    weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Method 3: Label smoothing for regularization
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
            self.confidence = 1. - smoothing
        
        def forward(self, x, target):
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
    
    smooth_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

Basic Training Loop
-------------------

Simple Training Function
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch.nn.functional as F
    from tqdm import tqdm
    import wandb  # For experiment tracking
    
    def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
        \"\"\"Train model for one epoch.\"\"\"
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            sequences = batch['sequence'].to(device)  # [B, T, C, H, W]
            masks = batch['mask'].to(device)          # [B, T]
            labels = batch['label'].to(device)        # [B]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences, masks)         # [B, num_classes]
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(model, val_loader, criterion, device):
        \"\"\"Validate model for one epoch.\"\"\"
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                sequences = batch['sequence'].to(device)
                masks = batch['mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(sequences, masks)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

Complete Training Script
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def train_sits_former(model, train_loader, val_loader, optimizer, 
                          scheduler, criterion, device, num_epochs=100,
                          save_dir='checkpoints', experiment_name='sits_training'):
        \"\"\"Complete training pipeline with logging and checkpointing.\"\"\"
        
        # Initialize experiment tracking
        wandb.init(project="sits-former", name=experiment_name, config={
            'model': 'SITSFormer',
            'epochs': num_epochs,
            'lr': optimizer.param_groups[0]['lr'],
            'batch_size': train_loader.batch_size
        })
        
        # Create save directory
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0
        best_epoch = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            
            # Validation
            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, device
            )
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_acc': best_val_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, os.path.join(save_dir, 'best_model.pth'))
                
                print(f"New best model saved! Val Acc: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
                break
                
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        wandb.finish()
        return best_val_acc, best_epoch

Advanced Training Techniques
-----------------------------

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torch.cuda.amp import autocast, GradScaler
    
    def train_epoch_mixed_precision(model, train_loader, optimizer, 
                                   criterion, device, scaler, epoch):
        \"\"\"Training with automatic mixed precision for faster training.\"\"\"
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} (Mixed Precision)')
        for batch_idx, batch in enumerate(pbar):
            sequences = batch['sequence'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                outputs = model(sequences, masks)
                loss = criterion(outputs, labels)
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(train_loader), 100. * correct / total
    
    # Usage with mixed precision
    scaler = GradScaler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch_mixed_precision(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )

Gradient Accumulation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def train_with_gradient_accumulation(model, train_loader, optimizer, 
                                        criterion, device, accumulation_steps=4):
        \"\"\"Training with gradient accumulation for larger effective batch size.\"\"\"
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            sequences = batch['sequence'].to(device)
            masks = batch['mask'].to(device)  
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(sequences, masks)
            loss = criterion(outputs, labels) / accumulation_steps  # Scale loss
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Final update if needed
        if (len(train_loader)) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        return total_loss / len(train_loader), 100. * correct / total

Hyperparameter Optimization
----------------------------

Using Weights & Biases Sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import wandb
    
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'embed_dim': {
                'values': [128, 256, 384, 512]
            },
            'num_layers': {
                'values': [4, 6, 8, 12]
            },
            'num_heads': {
                'values': [4, 8, 12, 16]
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.5
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-1
            }
        }
    }
    
    def train_with_sweep():
        \"\"\"Training function for hyperparameter sweep.\"\"\"
        wandb.init()
        config = wandb.config
        
        # Create model with sweep parameters
        model = SITSFormer(
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            **model_config  # Other fixed parameters
        )
        
        # Create optimizer with sweep parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Train model
        best_val_acc, _ = train_sits_former(
            model, train_loader, val_loader, optimizer,
            scheduler, criterion, device, num_epochs=50
        )
        
        return best_val_acc
    
    # Start sweep
    sweep_id = wandb.sweep(sweep_config, project="sits-former-optimization")
    wandb.agent(sweep_id, train_with_sweep, count=20)

Model Checkpointing and Resuming
---------------------------------

Saving Checkpoints
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, 
                       filepath, is_best=False):
        \"\"\"Save training checkpoint.\"\"\"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'accuracy': accuracy,
            'model_config': model_config,
            'timestamp': torch.tensor(time.time())
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

Loading and Resuming Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
        \"\"\"Load checkpoint and resume training.\"\"\"
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state  
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        best_acc = checkpoint.get('accuracy', 0)
        
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Previous best - Loss: {best_loss:.4f}, Acc: {best_acc:.2f}%")
        
        return start_epoch, best_loss, best_acc
    
    # Resume training example
    if resume_from_checkpoint:
        start_epoch, best_loss, best_acc = load_checkpoint(
            model, optimizer, scheduler, 'checkpoints/latest_checkpoint.pth', device
        )
    else:
        start_epoch = 1

Training Monitoring and Visualization
-------------------------------------

Real-time Monitoring
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    
    class TrainingMonitor:
        def __init__(self):
            self.train_losses = []
            self.val_losses = []
            self.train_accs = []
            self.val_accs = []
            self.learning_rates = []
        
        def update(self, train_loss, val_loss, train_acc, val_acc, lr):
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.learning_rates.append(lr)
        
        def plot_metrics(self):
            clear_output(wait=True)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(self.train_losses) + 1)
            
            # Loss plot
            ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
            ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy plot
            ax2.plot(epochs, self.train_accs, 'b-', label='Training Accuracy')
            ax2.plot(epochs, self.val_accs, 'r-', label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            # Learning rate plot
            ax3.plot(epochs, self.learning_rates, 'g-', label='Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True)
            
            # Training progress
            best_val_acc = max(self.val_accs)
            best_epoch = self.val_accs.index(best_val_acc) + 1
            ax4.text(0.1, 0.9, f'Best Val Acc: {best_val_acc:.2f}%', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.8, f'Best Epoch: {best_epoch}', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.7, f'Current Epoch: {len(epochs)}', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f'Current Val Acc: {self.val_accs[-1]:.2f}%', transform=ax4.transAxes, fontsize=12)
            ax4.set_xlim([0, 1])
            ax4.set_ylim([0, 1])
            ax4.axis('off')
            ax4.set_title('Training Progress')
            
            plt.tight_layout()
            plt.show()
    
    # Usage in training loop
    monitor = TrainingMonitor()
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(...)
        val_loss, val_acc = validate_epoch(...)
        
        current_lr = optimizer.param_groups[0]['lr']
        monitor.update(train_loss, val_loss, train_acc, val_acc, current_lr)
        
        # Plot every 5 epochs
        if epoch % 5 == 0:
            monitor.plot_metrics()

Complete Training Example
-------------------------

.. code-block:: python

    # Main training script
    def main():
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Model setup
        model = SITSFormer(
            img_size=224,
            patch_size=16,
            in_channels=13,
            num_classes=10,
            embed_dim=256,
            num_layers=8,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-2
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training
        best_val_acc, best_epoch = train_sits_former(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            num_epochs=100,
            save_dir='checkpoints',
            experiment_name='sits_base_model'
        )
        
        print(f"Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    if __name__ == "__main__":
        main()

Next Steps
----------

After training your model:

1. Proceed to model evaluation: :doc:`model_evaluation`
2. Learn about fine-tuning: :doc:`fine_tuning`
3. Explore example applications: :doc:`../examples/land_cover_classification`

Training Tips
-------------

1. **Start Simple**: Begin with a smaller model and dataset to verify your pipeline
2. **Monitor Closely**: Use experiment tracking and visualizations from the start
3. **Regularization**: Apply dropout, weight decay, and data augmentation appropriately
4. **Learning Rate**: Start with 1e-4 and adjust based on convergence behavior
5. **Batch Size**: Use the largest batch size that fits in memory (16-64 typical)
6. **Patience**: Allow sufficient epochs for convergence (50-200 depending on dataset size)

The training process is iterative - expect to experiment with different hyperparameters and techniques to achieve optimal results for your specific dataset and application.
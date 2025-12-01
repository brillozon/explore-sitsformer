Fine-Tuning SITS-Former Models
===============================

This tutorial covers fine-tuning pre-trained SITS-Former models for specific applications, including domain adaptation, transfer learning, and specialized classification tasks.

Overview
--------

Fine-tuning allows you to leverage pre-trained models and adapt them to new datasets or tasks with:

1. **Reduced Training Time** - Start from pre-trained weights rather than random initialization
2. **Better Performance** - Especially beneficial for smaller datasets
3. **Domain Adaptation** - Adapt models trained on one region/sensor to another
4. **Task Adaptation** - Modify models for different classification schemes

Prerequisites
-------------

Before fine-tuning, ensure you have:

- A pre-trained SITS-Former model or checkpoint
- Target dataset prepared (see :doc:`data_preparation`)
- Understanding of the differences between source and target domains

Types of Fine-Tuning
--------------------

1. **Full Fine-Tuning** - Update all model parameters
2. **Partial Fine-Tuning** - Freeze some layers, update others
3. **Linear Probing** - Only train the classification head
4. **Progressive Fine-Tuning** - Gradually unfreeze layers during training

Loading Pre-trained Models
---------------------------

From Checkpoint
~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from sitsformer.models import SITSFormer
    
    def load_pretrained_model(checkpoint_path, device='cpu'):
        """Load a pre-trained SITS-Former model from checkpoint."""
        print(f"Loading pre-trained model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model configuration if available
        model_config = checkpoint.get('model_config', {
            'img_size': 224,
            'patch_size': 16,
            'in_channels': 13,
            'num_classes': 10,  # Will be modified for new task
            'embed_dim': 256,
            'num_layers': 8,
            'num_heads': 8,
            'dropout': 0.1
        })
        
        # Create model with original configuration
        model = SITSFormer(**model_config)
        
        # Load pre-trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model, model_config
    
    # Load pre-trained model
    pretrained_model, config = load_pretrained_model('checkpoints/best_model.pth')

Adapting for New Tasks
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def adapt_model_for_new_task(pretrained_model, num_new_classes, freeze_backbone=False):
        """Adapt pre-trained model for new classification task."""
        
        # Get original number of classes
        original_num_classes = pretrained_model.head.out_features
        print(f"Original classes: {original_num_classes}, New classes: {num_new_classes}")
        
        # Replace classification head
        pretrained_model.head = torch.nn.Linear(
            pretrained_model.head.in_features, 
            num_new_classes
        )
        
        # Initialize new head with proper weights
        torch.nn.init.xavier_uniform_(pretrained_model.head.weight)
        torch.nn.init.zeros_(pretrained_model.head.bias)
        
        # Optionally freeze backbone for transfer learning
        if freeze_backbone:
            for name, param in pretrained_model.named_parameters():
                if 'head' not in name:  # Don't freeze the new classification head
                    param.requires_grad = False
            print("Backbone frozen - only training classification head")
        else:
            print("Full model will be fine-tuned")
        
        return pretrained_model
    
    # Adapt for new task (e.g., 5 classes instead of 10)
    model = adapt_model_for_new_task(
        pretrained_model, 
        num_new_classes=5, 
        freeze_backbone=False
    )

Domain Adaptation
-----------------

Spectral Band Adaptation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def adapt_spectral_bands(model, new_in_channels, adaptation_method='interpolation'):
        """Adapt model for different number of spectral bands."""
        
        # Get current patch embedding layer
        patch_embed = model.patch_embed
        old_in_channels = patch_embed.proj.in_channels
        
        print(f"Adapting from {old_in_channels} to {new_in_channels} channels")
        
        if new_in_channels == old_in_channels:
            return model
        
        # Save old weights
        old_weight = patch_embed.proj.weight.data.clone()  # [embed_dim, old_channels, patch_h, patch_w]
        old_bias = patch_embed.proj.bias.data.clone()
        
        # Create new projection layer
        new_proj = torch.nn.Conv2d(
            new_in_channels, 
            patch_embed.proj.out_channels,
            kernel_size=patch_embed.proj.kernel_size,
            stride=patch_embed.proj.stride,
            padding=patch_embed.proj.padding
        )
        
        if adaptation_method == 'interpolation':
            # Interpolate weights across spectral dimension
            if new_in_channels > old_in_channels:
                # Upsample: interpolate to more channels
                weight_interp = torch.nn.functional.interpolate(
                    old_weight, size=(new_in_channels, old_weight.size(-1)), 
                    mode='bilinear', align_corners=False
                )
                new_proj.weight.data = weight_interp
            else:
                # Downsample: select subset of channels
                indices = torch.linspace(0, old_in_channels-1, new_in_channels).long()
                new_proj.weight.data = old_weight[:, indices, :, :]
        
        elif adaptation_method == 'averaging':
            # Average existing channels and replicate
            if new_in_channels > old_in_channels:
                # Replicate averaged channels
                avg_weight = old_weight.mean(dim=1, keepdim=True)
                new_proj.weight.data = avg_weight.repeat(1, new_in_channels, 1, 1)
            else:
                # Group and average channels
                group_size = old_in_channels // new_in_channels
                new_weight = old_weight.view(
                    old_weight.size(0), new_in_channels, group_size, 
                    old_weight.size(-2), old_weight.size(-1)
                ).mean(dim=2)
                new_proj.weight.data = new_weight
        
        elif adaptation_method == 'random':
            # Initialize randomly (fallback)
            torch.nn.init.xavier_uniform_(new_proj.weight)
        
        # Copy bias
        new_proj.bias.data = old_bias
        
        # Replace the projection layer
        model.patch_embed.proj = new_proj
        
        return model
    
    # Example: Adapt from Sentinel-2 (13 bands) to Landsat-8 (8 bands)
    model_landsat = adapt_spectral_bands(
        model, 
        new_in_channels=8, 
        adaptation_method='interpolation'
    )

Spatial Resolution Adaptation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def adapt_spatial_resolution(model, new_img_size, new_patch_size=None):
        """Adapt model for different spatial resolution."""
        
        old_img_size = model.img_size
        old_patch_size = model.patch_embed.patch_size[0]
        
        if new_patch_size is None:
            new_patch_size = old_patch_size
        
        print(f"Adapting from {old_img_size}x{old_img_size} to {new_img_size}x{new_img_size}")
        print(f"Patch size: {old_patch_size} -> {new_patch_size}")
        
        # Update model attributes
        model.img_size = new_img_size
        
        # Adapt patch embedding if patch size changed
        if new_patch_size != old_patch_size:
            old_proj = model.patch_embed.proj
            new_proj = torch.nn.Conv2d(
                old_proj.in_channels,
                old_proj.out_channels,
                kernel_size=new_patch_size,
                stride=new_patch_size,
                padding=0
            )
            
            # Initialize with interpolated weights
            if new_patch_size > old_patch_size:
                # Upsample kernel
                weight_interp = torch.nn.functional.interpolate(
                    old_proj.weight, size=(new_patch_size, new_patch_size),
                    mode='bilinear', align_corners=False
                )
            else:
                # Downsample kernel  
                weight_interp = torch.nn.functional.adaptive_avg_pool2d(
                    old_proj.weight, (new_patch_size, new_patch_size)
                )
            
            new_proj.weight.data = weight_interp
            new_proj.bias.data = old_proj.bias.data.clone()
            
            model.patch_embed.proj = new_proj
            model.patch_embed.patch_size = (new_patch_size, new_patch_size)
        
        # Adapt positional embeddings
        old_num_patches = (old_img_size // old_patch_size) ** 2
        new_num_patches = (new_img_size // new_patch_size) ** 2
        
        if new_num_patches != old_num_patches:
            print(f"Adapting positional embeddings: {old_num_patches} -> {new_num_patches} patches")
            
            old_pos_embed = model.pos_embed.data  # [1, old_num_patches + 1, embed_dim]
            class_token = old_pos_embed[:, 0:1, :]  # Class token
            old_patch_embed = old_pos_embed[:, 1:, :]  # Patch embeddings
            
            # Reshape to 2D grid
            old_grid_size = int(old_num_patches ** 0.5)
            new_grid_size = int(new_num_patches ** 0.5)
            embed_dim = old_patch_embed.size(-1)
            
            old_patch_embed_2d = old_patch_embed.view(1, old_grid_size, old_grid_size, embed_dim)
            old_patch_embed_2d = old_patch_embed_2d.permute(0, 3, 1, 2)  # [1, embed_dim, H, W]
            
            # Interpolate to new size
            new_patch_embed_2d = torch.nn.functional.interpolate(
                old_patch_embed_2d, size=(new_grid_size, new_grid_size),
                mode='bilinear', align_corners=False
            )
            
            # Reshape back
            new_patch_embed_2d = new_patch_embed_2d.permute(0, 2, 3, 1)  # [1, H, W, embed_dim]
            new_patch_embed = new_patch_embed_2d.view(1, new_num_patches, embed_dim)
            
            # Combine class token and patch embeddings
            new_pos_embed = torch.cat([class_token, new_patch_embed], dim=1)
            
            # Update model
            model.pos_embed = torch.nn.Parameter(new_pos_embed)
        
        return model

Fine-Tuning Strategies
----------------------

Linear Probing
~~~~~~~~~~~~~~

.. code-block:: python

    def setup_linear_probing(model, learning_rate=1e-3):
        """Setup for linear probing - only train classification head."""
        
        # Freeze all parameters except classification head
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Create optimizer only for trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        
        print(f"Linear probing setup - training {sum(p.numel() for p in trainable_params):,} parameters")
        
        return optimizer
    
    # Linear probing optimizer
    linear_optimizer = setup_linear_probing(model, learning_rate=1e-3)

Progressive Fine-Tuning
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class ProgressiveFinetuner:
        """Progressive fine-tuning by gradually unfreezing layers."""
        
        def __init__(self, model, unfreeze_schedule):
            """
            Args:
                model: SITS-Former model
                unfreeze_schedule: Dict mapping epoch to layer names to unfreeze
                                 e.g., {0: ['head'], 10: ['blocks.7', 'blocks.6'], 20: 'all'}
            """
            self.model = model
            self.unfreeze_schedule = unfreeze_schedule
            
            # Initially freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
        
        def update_frozen_layers(self, epoch):
            """Update which layers are frozen based on epoch."""
            if epoch not in self.unfreeze_schedule:
                return
            
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            
            if layers_to_unfreeze == 'all':
                # Unfreeze all parameters
                for param in self.model.parameters():
                    param.requires_grad = True
                print(f"Epoch {epoch}: Unfroze all layers")
            else:
                # Unfreeze specific layers
                for layer_name in layers_to_unfreeze:
                    for name, param in self.model.named_parameters():
                        if layer_name in name:
                            param.requires_grad = True
                print(f"Epoch {epoch}: Unfroze layers {layers_to_unfreeze}")
        
        def get_trainable_params(self):
            """Get currently trainable parameters."""
            return [p for p in self.model.parameters() if p.requires_grad]
    
    # Progressive fine-tuning setup
    unfreeze_schedule = {
        0: ['head'],                    # Start with head only
        5: ['blocks.7', 'blocks.6'],    # Add top transformer layers
        10: ['blocks.5', 'blocks.4'],   # Add more layers
        15: ['blocks.3', 'blocks.2'],   # Add more layers
        20: 'all'                       # Unfreeze everything
    }
    
    progressive_ft = ProgressiveFinetuner(model, unfreeze_schedule)

Layer-wise Learning Rate Decay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def setup_layerwise_lr(model, base_lr=1e-4, decay_factor=0.8):
        """Setup different learning rates for different layers."""
        
        # Define layer groups (deeper layers get higher learning rates)
        param_groups = []
        
        # Classification head - highest learning rate
        head_params = []
        for name, param in model.named_parameters():
            if 'head' in name:
                head_params.append(param)
        param_groups.append({'params': head_params, 'lr': base_lr, 'name': 'head'})
        
        # Transformer blocks - decreasing learning rate with depth
        num_layers = len(model.blocks)
        for i, block in enumerate(model.blocks):
            layer_lr = base_lr * (decay_factor ** (num_layers - i - 1))
            block_params = list(block.parameters())
            param_groups.append({
                'params': block_params, 
                'lr': layer_lr, 
                'name': f'block_{i}'
            })
        
        # Patch embedding - lowest learning rate
        embed_params = []
        for name, param in model.named_parameters():
            if 'patch_embed' in name or 'pos_embed' in name:
                embed_params.append(param)
        embed_lr = base_lr * (decay_factor ** num_layers)
        param_groups.append({'params': embed_params, 'lr': embed_lr, 'name': 'embedding'})
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
        
        # Print learning rates
        for group in param_groups:
            print(f"Layer group '{group['name']}': lr={group['lr']:.2e}")
        
        return optimizer

Fine-Tuning Training Loop
-------------------------

.. code-block:: python

    def finetune_model(model, train_loader, val_loader, num_epochs=50,
                       base_lr=1e-5, warmup_epochs=3, progressive_ft=None,
                       save_dir='finetune_checkpoints'):
        """Complete fine-tuning training loop."""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizer (will be updated for progressive fine-tuning)
        if progressive_ft:
            progressive_ft.update_frozen_layers(0)
            optimizer = torch.optim.AdamW(
                progressive_ft.get_trainable_params(), 
                lr=base_lr, 
                weight_decay=1e-3
            )
        else:
            optimizer = setup_layerwise_lr(model, base_lr)
        
        # Learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs, eta_min=base_lr * 0.01
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        best_val_acc = 0
        patience_counter = 0
        patience = 10
        
        for epoch in range(num_epochs):
            print(f"\\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Progressive fine-tuning update
            if progressive_ft:
                progressive_ft.update_frozen_layers(epoch)
                
                # Update optimizer if trainable parameters changed
                new_trainable = progressive_ft.get_trainable_params()
                if len(new_trainable) != len(optimizer.param_groups[0]['params']):
                    optimizer = torch.optim.AdamW(
                        new_trainable, lr=base_lr, weight_decay=1e-3
                    )
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=num_epochs - epoch, eta_min=base_lr * 0.01
                    )
            
            # Training
            model.train()
            train_loss, train_acc = train_epoch_finetune(
                model, train_loader, optimizer, criterion, device, epoch
            )
            
            # Validation
            model.eval()
            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, device
            )
            
            # Learning rate scheduling (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': history
                }, os.path.join(save_dir, 'best_finetuned_model.pth'))
                
                print(f"New best model saved! Val Acc: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return model, history, best_val_acc

    def train_epoch_finetune(model, train_loader, optimizer, criterion, device, epoch):
        """Training epoch for fine-tuning with careful learning rate handling."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Lower learning rate for early epochs (warmup)
        if epoch < 3:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (epoch + 1) / 3
        
        pbar = tqdm(train_loader, desc=f'Fine-tune Epoch {epoch+1}')
        for batch in pbar:
            sequences = batch['sequence'].to(device)
            masks = batch['mask'].to(device) if 'mask' in batch else None
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if masks is not None:
                outputs = model(sequences, masks)
            else:
                outputs = model(sequences)
            
            loss = criterion(outputs, labels)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Lower than pre-training
            optimizer.step()
            
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

Specialized Fine-Tuning Scenarios
----------------------------------

Few-Shot Learning
~~~~~~~~~~~~~~~~~

.. code-block:: python

    def setup_few_shot_learning(model, support_loader, num_shots=5):
        """Setup for few-shot learning with prototypical networks approach."""
        
        # Extract features for support set
        model.eval()
        prototypes = {}
        
        with torch.no_grad():
            for batch in support_loader:
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device)
                
                # Get features before classification head
                features = model.forward_features(sequences)  # [B, embed_dim]
                
                # Compute prototypes for each class
                for label in labels.unique():
                    mask = labels == label
                    class_features = features[mask]
                    
                    if label.item() not in prototypes:
                        prototypes[label.item()] = []
                    prototypes[label.item()].append(class_features.mean(dim=0))
        
        # Average prototypes across shots
        for class_id in prototypes:
            prototypes[class_id] = torch.stack(prototypes[class_id]).mean(dim=0)
        
        return prototypes
    
    def few_shot_predict(model, query_batch, prototypes, device):
        """Make predictions using prototypical networks."""
        model.eval()
        
        with torch.no_grad():
            sequences = query_batch['sequence'].to(device)
            
            # Extract features
            query_features = model.forward_features(sequences)  # [B, embed_dim]
            
            # Compute distances to prototypes
            predictions = []
            for query_feat in query_features:
                distances = {}
                for class_id, prototype in prototypes.items():
                    dist = torch.dist(query_feat, prototype)
                    distances[class_id] = dist
                
                # Predict closest prototype
                pred_class = min(distances, key=distances.get)
                predictions.append(pred_class)
        
        return torch.tensor(predictions)

Domain-Specific Fine-Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def geographic_adaptation_loss(outputs, labels, source_domain_mask, target_domain_mask, alpha=0.1):
        """Custom loss for geographic domain adaptation."""
        
        # Standard classification loss
        ce_loss = torch.nn.functional.cross_entropy(outputs, labels)
        
        # Domain adversarial component (simplified)
        if source_domain_mask.sum() > 0 and target_domain_mask.sum() > 0:
            source_features = outputs[source_domain_mask]
            target_features = outputs[target_domain_mask]
            
            # Encourage similar feature distributions
            domain_loss = torch.nn.functional.mse_loss(
                source_features.mean(dim=0),
                target_features.mean(dim=0)
            )
            
            total_loss = ce_loss + alpha * domain_loss
        else:
            total_loss = ce_loss
        
        return total_loss

Multi-Task Fine-Tuning
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class MultiTaskSITSFormer(torch.nn.Module):
        """SITS-Former adapted for multiple tasks."""
        
        def __init__(self, base_model, task_configs):
            super().__init__()
            self.backbone = base_model
            
            # Remove original head
            self.backbone.head = torch.nn.Identity()
            
            # Create task-specific heads
            self.task_heads = torch.nn.ModuleDict()
            for task_name, config in task_configs.items():
                self.task_heads[task_name] = torch.nn.Sequential(
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(base_model.embed_dim, config['num_classes'])
                )
        
        def forward(self, x, task_name=None, masks=None):
            # Get backbone features
            features = self.backbone.forward_features(x, masks)
            
            if task_name:
                # Single task prediction
                return self.task_heads[task_name](features)
            else:
                # All tasks prediction
                outputs = {}
                for task in self.task_heads:
                    outputs[task] = self.task_heads[task](features)
                return outputs
    
    def multitask_training_step(model, batch, task_weights, criterion):
        """Training step for multi-task learning."""
        sequences = batch['sequence']
        masks = batch.get('mask', None)
        
        # Get predictions for all tasks
        outputs = model(sequences, masks=masks)
        
        total_loss = 0
        for task_name, task_output in outputs.items():
            if task_name in batch:  # Only compute loss if ground truth available
                task_loss = criterion(task_output, batch[task_name])
                weighted_loss = task_weights.get(task_name, 1.0) * task_loss
                total_loss += weighted_loss
        
        return total_loss, outputs

Evaluation and Analysis
-----------------------

Fine-Tuning Performance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_finetuning_performance(original_model, finetuned_model, 
                                      test_loader, device):
        """Compare performance before and after fine-tuning."""
        
        models = {
            'Original': original_model,
            'Fine-tuned': finetuned_model
        }
        
        results = {}
        
        for model_name, model in models.items():
            model.eval()
            correct = 0
            total = 0
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in test_loader:
                    sequences = batch['sequence'].to(device)
                    masks = batch.get('mask', None)
                    labels = batch['label'].to(device)
                    
                    if masks is not None:
                        outputs = model(sequences, masks)
                    else:
                        outputs = model(sequences)
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    predictions.extend(predicted.cpu().numpy())
                    targets.extend(labels.cpu().numpy())
            
            accuracy = 100. * correct / total
            
            # Compute additional metrics
            from sklearn.metrics import classification_report, confusion_matrix
            
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'targets': targets,
                'classification_report': classification_report(targets, predictions),
                'confusion_matrix': confusion_matrix(targets, predictions)
            }
            
            print(f"{model_name} Model Accuracy: {accuracy:.2f}%")
        
        # Performance improvement
        improvement = results['Fine-tuned']['accuracy'] - results['Original']['accuracy']
        print(f"\\nPerformance improvement: {improvement:.2f} percentage points")
        
        return results

Layer-wise Feature Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_layer_changes(original_model, finetuned_model, sample_input):
        """Analyze how features change through fine-tuning."""
        
        def get_layer_outputs(model, x):
            outputs = {}
            
            # Hook function to capture intermediate outputs
            def hook_fn(name):
                def hook(module, input, output):
                    outputs[name] = output.detach()
                return hook
            
            # Register hooks
            hooks = []
            for name, module in model.named_modules():
                if 'blocks' in name and 'attn' in name:
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            # Forward pass
            with torch.no_grad():
                _ = model(sample_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            return outputs
        
        # Get outputs from both models
        orig_outputs = get_layer_outputs(original_model, sample_input)
        ft_outputs = get_layer_outputs(finetuned_model, sample_input)
        
        # Compare feature similarity
        similarities = {}
        for layer_name in orig_outputs:
            if layer_name in ft_outputs:
                orig_feat = orig_outputs[layer_name].flatten()
                ft_feat = ft_outputs[layer_name].flatten()
                
                # Cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    orig_feat.unsqueeze(0), ft_feat.unsqueeze(0)
                ).item()
                
                similarities[layer_name] = similarity
        
        return similarities

Best Practices for Fine-Tuning
-------------------------------

1. **Start Conservative**
   - Begin with lower learning rates (1e-5 to 1e-4)
   - Use shorter training epochs initially
   - Monitor for overfitting carefully

2. **Progressive Approach**
   - Start with linear probing
   - Gradually unfreeze more layers
   - Use layer-wise learning rate decay

3. **Data Considerations**
   - Ensure target domain data quality
   - Consider data augmentation strategies
   - Balance between source and target domains

4. **Regularization**
   - Use dropout and weight decay
   - Apply gradient clipping
   - Monitor validation performance closely

5. **Experiment Tracking**
   - Compare multiple fine-tuning strategies
   - Track both performance and training dynamics
   - Save intermediate checkpoints

Next Steps
----------

After fine-tuning your model:

1. Evaluate thoroughly: :doc:`../examples/model_evaluation`
2. Deploy your model: :doc:`deployment`
3. Monitor performance in production

Fine-tuning is an iterative process - expect to experiment with different strategies to find what works best for your specific application and dataset.
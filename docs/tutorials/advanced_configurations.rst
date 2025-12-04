Advanced Configurations and Customization
==========================================

This tutorial covers advanced configuration options for SITS-Former, including custom architectures, experimental features, advanced data augmentation, custom loss functions, and specialized training techniques.

Overview
--------

Advanced configurations enable you to:

1. **Custom Model Architectures** - Modify network components and experiment with new designs
2. **Advanced Training Strategies** - Implement sophisticated training algorithms
3. **Custom Data Processing** - Create specialized preprocessing and augmentation pipelines
4. **Experimental Features** - Test cutting-edge techniques and research ideas
5. **Performance Optimization** - Fine-tune for specific hardware and use cases
6. **Multi-modal Integration** - Combine satellite imagery with other data sources

Custom Model Architectures
---------------------------

Modular Architecture Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from typing import Optional, Callable, Union, List
    import torch
    import torch.nn as nn
    from sitsformer.models.base import SITSFormerBase
    from sitsformer.models.components import (
        PatchEmbedding, PositionalEncoding, 
        TransformerBlock, AttentionBlock
    )
    
    class CustomSITSFormer(SITSFormerBase):
        """Highly customizable SITS-Former with modular components."""
        
        def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 13,
            num_classes: int = 10,
            embed_dim: int = 256,
            num_layers: int = 8,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
            # Advanced configuration options
            attention_type: str = 'standard',  # 'standard', 'sparse', 'linear', 'longformer'
            patch_embed_type: str = 'conv',    # 'conv', 'linear', 'depthwise'
            positional_encoding: str = 'learned',  # 'learned', 'sinusoidal', 'relative'
            activation_fn: str = 'gelu',       # 'gelu', 'relu', 'swish', 'mish'
            norm_layer: Optional[Callable] = None,
            custom_blocks: Optional[List[nn.Module]] = None,
            use_checkpoint: bool = False,      # Gradient checkpointing
            # Architectural modifications
            use_cls_token: bool = True,
            pool_type: str = 'cls',           # 'cls', 'mean', 'max', 'attention'
            stochastic_depth_prob: float = 0.0,  # Stochastic depth
            layer_scale_init: Optional[float] = None,  # Layer scale
            # Multi-scale features
            use_pyramid: bool = False,
            pyramid_levels: List[int] = [2, 4, 8],
            # Temporal modeling
            temporal_fusion: str = 'transformer',  # 'transformer', 'lstm', 'gru', 'tcn'
            temporal_aggregation: str = 'attention',  # 'attention', 'mean', 'max', 'last'
        ):
            super().__init__()
            
            self.img_size = img_size
            self.patch_size = patch_size
            self.num_classes = num_classes
            self.embed_dim = embed_dim
            self.use_cls_token = use_cls_token
            self.use_checkpoint = use_checkpoint
            
            # Patch embedding with different strategies
            self.patch_embed = self._create_patch_embedding(
                patch_embed_type, img_size, patch_size, in_channels, embed_dim
            )
            
            num_patches = self.patch_embed.num_patches
            self.seq_len = num_patches + (1 if use_cls_token else 0)
            
            # Positional encoding
            self.pos_embed = self._create_positional_encoding(
                positional_encoding, self.seq_len, embed_dim
            )
            
            # CLS token
            if use_cls_token:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                nn.init.trunc_normal_(self.cls_token, std=0.02)
            
            # Dropout
            self.pos_drop = nn.Dropout(dropout)
            
            # Transformer blocks
            if custom_blocks is not None:
                self.blocks = nn.ModuleList(custom_blocks)
            else:
                self.blocks = self._create_transformer_blocks(
                    num_layers, embed_dim, num_heads, mlp_ratio,
                    dropout, attention_type, activation_fn, norm_layer,
                    stochastic_depth_prob, layer_scale_init
                )
            
            # Multi-scale pyramid features
            if use_pyramid:
                self.pyramid_blocks = self._create_pyramid_blocks(
                    pyramid_levels, embed_dim
                )
                self.use_pyramid = True
            else:
                self.use_pyramid = False
            
            # Temporal modeling
            self.temporal_module = self._create_temporal_module(
                temporal_fusion, embed_dim, dropout
            )
            
            # Final normalization
            norm_layer = norm_layer or nn.LayerNorm
            self.norm = norm_layer(embed_dim)
            
            # Classification head with flexible pooling
            self.pool_type = pool_type
            if pool_type == 'attention':
                self.attention_pool = nn.MultiheadAttention(
                    embed_dim, num_heads, dropout=dropout, batch_first=True
                )
                self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
            
            self.head = nn.Linear(embed_dim, num_classes)
            
            # Initialize weights
            self.apply(self._init_weights)
        
        def _create_patch_embedding(self, embed_type: str, img_size: int, 
                                  patch_size: int, in_channels: int, embed_dim: int):
            """Create different types of patch embeddings."""
            
            if embed_type == 'conv':
                return PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
            
            elif embed_type == 'depthwise':
                return DepthwisePatchEmbedding(
                    img_size, patch_size, in_channels, embed_dim
                )
            
            elif embed_type == 'linear':
                return LinearPatchEmbedding(
                    img_size, patch_size, in_channels, embed_dim
                )
            
            else:
                raise ValueError(f"Unknown patch embedding type: {embed_type}")
        
        def _create_positional_encoding(self, pos_type: str, seq_len: int, embed_dim: int):
            """Create different types of positional encodings."""
            
            if pos_type == 'learned':
                pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
                nn.init.trunc_normal_(pos_embed, std=0.02)
                return pos_embed
            
            elif pos_type == 'sinusoidal':
                return SinusoidalPositionalEncoding(seq_len, embed_dim)
            
            elif pos_type == 'relative':
                return RelativePositionalEncoding(embed_dim)
            
            else:
                raise ValueError(f"Unknown positional encoding type: {pos_type}")
        
        def _create_transformer_blocks(self, num_layers: int, embed_dim: int,
                                     num_heads: int, mlp_ratio: float, dropout: float,
                                     attention_type: str, activation_fn: str,
                                     norm_layer: Optional[Callable],
                                     stochastic_depth_prob: float,
                                     layer_scale_init: Optional[float]):
            """Create transformer blocks with advanced features."""
            
            # Stochastic depth schedule
            dpr = [x.item() for x in torch.linspace(0, stochastic_depth_prob, num_layers)]
            
            blocks = []
            for i in range(num_layers):
                block = AdvancedTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_type=attention_type,
                    activation_fn=activation_fn,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    layer_scale_init=layer_scale_init
                )
                blocks.append(block)
            
            return nn.ModuleList(blocks)
        
        def _create_temporal_module(self, fusion_type: str, embed_dim: int, dropout: float):
            """Create temporal modeling modules."""
            
            if fusion_type == 'transformer':
                return TemporalTransformer(embed_dim, dropout)
            elif fusion_type == 'lstm':
                return TemporalLSTM(embed_dim, dropout)
            elif fusion_type == 'gru':
                return TemporalGRU(embed_dim, dropout)
            elif fusion_type == 'tcn':
                return TemporalCNN(embed_dim, dropout)
            else:
                return nn.Identity()

Advanced Attention Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class SparseAttention(nn.Module):
        """Sparse attention for handling long sequences efficiently."""
        
        def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1,
                     sparsity_pattern: str = 'local', window_size: int = 64):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.sparsity_pattern = sparsity_pattern
            self.window_size = window_size
            
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.attn_drop = nn.Dropout(dropout)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(dropout)
        
        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            if self.sparsity_pattern == 'local':
                attn_weights = self._local_attention(q, k, v, mask)
            elif self.sparsity_pattern == 'strided':
                attn_weights = self._strided_attention(q, k, v, mask)
            elif self.sparsity_pattern == 'random':
                attn_weights = self._random_attention(q, k, v, mask)
            else:
                # Standard full attention
                attn_weights = self._full_attention(q, k, v, mask)
            
            x = attn_weights.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
        def _local_attention(self, q, k, v, mask):
            """Local attention within sliding windows."""
            B, H, N, D = q.shape
            
            # Pad sequence for windowing
            pad_len = (self.window_size - N % self.window_size) % self.window_size
            if pad_len > 0:
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
                N_padded = N + pad_len
            else:
                N_padded = N
            
            # Reshape into windows
            num_windows = N_padded // self.window_size
            q = q.view(B, H, num_windows, self.window_size, D)
            k = k.view(B, H, num_windows, self.window_size, D)
            v = v.view(B, H, num_windows, self.window_size, D)
            
            # Compute attention within windows
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            
            out = (attn @ v).view(B, H, N_padded, D)
            
            # Remove padding
            if pad_len > 0:
                out = out[:, :, :N, :]
            
            return out
    
    class LinearAttention(nn.Module):
        """Linear attention with O(N) complexity."""
        
        def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.proj = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x: torch.Tensor):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            # Apply ELU + 1 to ensure non-negativity
            q = F.elu(q) + 1
            k = F.elu(k) + 1
            
            # Linear attention computation
            kv = k.transpose(-2, -1) @ v  # (B, H, D, D)
            k_sum = k.sum(dim=-2, keepdim=True)  # (B, H, 1, D)
            
            out = q @ kv / (q @ k_sum.transpose(-2, -1) + 1e-8)
            
            out = out.transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            out = self.dropout(out)
            
            return out

Custom Loss Functions
---------------------

Advanced Loss Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class FocalLoss(nn.Module):
        """Focal Loss for addressing class imbalance."""
        
        def __init__(self, alpha: Union[float, torch.Tensor] = 1.0, 
                     gamma: float = 2.0, reduction: str = 'mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
        
        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
    
    class DiceLoss(nn.Module):
        """Dice Loss for segmentation-like tasks."""
        
        def __init__(self, smooth: float = 1e-6):
            super().__init__()
            self.smooth = smooth
        
        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            # Convert to one-hot
            num_classes = inputs.shape[1]
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            
            # Softmax for inputs
            inputs_soft = F.softmax(inputs, dim=1)
            
            # Reshape for computation
            inputs_flat = inputs_soft.view(inputs_soft.shape[0], inputs_soft.shape[1], -1)
            targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
            
            # Dice coefficient per class
            intersection = (inputs_flat * targets_flat).sum(dim=-1)
            total = inputs_flat.sum(dim=-1) + targets_flat.sum(dim=-1)
            
            dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
            
            # Average over classes and batch
            return 1 - dice.mean()
    
    class ContrastiveLoss(nn.Module):
        """Contrastive loss for representation learning."""
        
        def __init__(self, temperature: float = 0.07, margin: float = 0.5):
            super().__init__()
            self.temperature = temperature
            self.margin = margin
        
        def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            # Normalize features
            features = F.normalize(features, p=2, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(features, features.T) / self.temperature
            
            # Create positive and negative masks
            labels = labels.unsqueeze(1)
            positive_mask = torch.eq(labels, labels.T).float()
            negative_mask = 1 - positive_mask
            
            # Remove self-similarity
            positive_mask.fill_diagonal_(0)
            
            # Compute contrastive loss
            exp_sim = torch.exp(similarity_matrix)
            
            # Positive pairs
            pos_sim = similarity_matrix * positive_mask
            pos_exp = exp_sim * positive_mask
            
            # Negative pairs
            neg_exp = exp_sim * negative_mask
            
            # Loss computation
            numerator = torch.sum(pos_exp, dim=1)
            denominator = torch.sum(neg_exp, dim=1) + numerator
            
            loss = -torch.log(numerator / (denominator + 1e-8))
            return loss.mean()
    
    class TripletLoss(nn.Module):
        """Triplet loss for metric learning."""
        
        def __init__(self, margin: float = 0.3, mining: str = 'hard'):
            super().__init__()
            self.margin = margin
            self.mining = mining
        
        def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            # Compute pairwise distances
            distances = self._pairwise_distance(embeddings)
            
            if self.mining == 'hard':
                return self._hard_triplet_loss(distances, labels)
            else:
                return self._batch_all_triplet_loss(distances, labels)
        
        def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
            """Compute pairwise Euclidean distances."""
            dot_product = torch.matmul(embeddings, embeddings.T)
            square_norm = torch.diagonal(dot_product)
            distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
            distances = torch.clamp(distances, min=0.0)
            return torch.sqrt(distances + 1e-8)
        
        def _hard_triplet_loss(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """Hard negative mining triplet loss."""
            positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            negative_mask = ~positive_mask
            
            # Remove diagonal
            positive_mask.fill_diagonal_(False)
            
            # Hard positive: furthest positive
            positive_distances = distances * positive_mask.float()
            positive_distances[~positive_mask] = float('-inf')
            hardest_positive = positive_distances.max(dim=1)[0]
            
            # Hard negative: closest negative
            negative_distances = distances * negative_mask.float()
            negative_distances[~negative_mask] = float('inf')
            hardest_negative = negative_distances.min(dim=1)[0]
            
            # Triplet loss
            triplet_loss = F.relu(hardest_positive - hardest_negative + self.margin)
            return triplet_loss.mean()

Advanced Data Augmentation
---------------------------

Spectral and Temporal Augmentations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import random
    import numpy as np
    from typing import Tuple, List, Optional
    
    class AdvancedSatelliteAugmentation:
        """Advanced augmentation techniques for satellite imagery."""
        
        def __init__(self, 
                     spectral_shift_prob: float = 0.3,
                     temporal_dropout_prob: float = 0.2,
                     cloud_injection_prob: float = 0.1,
                     noise_injection_prob: float = 0.4,
                     seasonal_shift_prob: float = 0.2):
            
            self.spectral_shift_prob = spectral_shift_prob
            self.temporal_dropout_prob = temporal_dropout_prob
            self.cloud_injection_prob = cloud_injection_prob
            self.noise_injection_prob = noise_injection_prob
            self.seasonal_shift_prob = seasonal_shift_prob
        
        def __call__(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            """Apply augmentations to satellite image time series."""
            # sequence shape: [T, C, H, W]
            
            if mask is None:
                mask = torch.ones(sequence.shape[0], dtype=torch.bool)
            
            # Apply various augmentations
            sequence, mask = self._spectral_shift(sequence, mask)
            sequence, mask = self._temporal_dropout(sequence, mask)
            sequence, mask = self._cloud_injection(sequence, mask)
            sequence, mask = self._noise_injection(sequence, mask)
            sequence, mask = self._seasonal_shift(sequence, mask)
            
            return sequence, mask
        
        def _spectral_shift(self, sequence: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Simulate spectral calibration differences between sensors."""
            if random.random() > self.spectral_shift_prob:
                return sequence, mask
            
            T, C, H, W = sequence.shape
            
            # Random spectral shifts per band
            spectral_shifts = torch.normal(0, 0.02, size=(C,))
            multiplicative_shifts = torch.normal(1, 0.05, size=(C,))
            
            # Apply shifts
            for c in range(C):
                sequence[:, c] = sequence[:, c] * multiplicative_shifts[c] + spectral_shifts[c]
            
            return torch.clamp(sequence, 0, 1), mask
        
        def _temporal_dropout(self, sequence: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Randomly drop temporal observations."""
            if random.random() > self.temporal_dropout_prob:
                return sequence, mask
            
            T = sequence.shape[0]
            
            # Keep at least 3 time steps
            min_keep = max(3, T // 3)
            num_keep = random.randint(min_keep, T)
            
            # Randomly select time steps to keep
            keep_indices = torch.randperm(T)[:num_keep].sort()[0]
            
            sequence_dropped = sequence[keep_indices]
            mask_dropped = mask[keep_indices]
            
            return sequence_dropped, mask_dropped
        
        def _cloud_injection(self, sequence: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Inject synthetic cloud cover."""
            if random.random() > self.cloud_injection_prob:
                return sequence, mask
            
            T, C, H, W = sequence.shape
            
            # Choose random time step for cloud injection
            cloud_time = random.randint(0, T - 1)
            
            # Generate cloud mask (smooth blobs)
            cloud_coverage = random.uniform(0.1, 0.4)  # 10-40% coverage
            cloud_mask = self._generate_cloud_mask(H, W, cloud_coverage)
            
            # Simulate cloud effects (higher reflectance, especially in visible bands)
            cloud_reflectance = torch.rand(C) * 0.3 + 0.7  # 0.7-1.0 reflectance
            
            # Apply cloud effects
            for c in range(C):
                sequence[cloud_time, c] = (
                    sequence[cloud_time, c] * (1 - cloud_mask) + 
                    cloud_reflectance[c] * cloud_mask
                )
            
            # Update mask to indicate cloud contamination
            if cloud_mask.sum() > 0.2 * H * W:  # If >20% cloudy, mark as invalid
                mask[cloud_time] = False
            
            return sequence, mask
        
        def _generate_cloud_mask(self, H: int, W: int, coverage: float) -> torch.Tensor:
            """Generate realistic cloud mask using Perlin noise-like approach."""
            import torch.nn.functional as F
            
            # Create random seeds
            num_seeds = int(coverage * 10)
            cloud_mask = torch.zeros(H, W)
            
            for _ in range(num_seeds):
                # Random center and size
                cx, cy = random.randint(0, W-1), random.randint(0, H-1)
                radius = random.randint(10, min(H, W) // 3)
                
                # Create Gaussian blob
                y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                distance = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                blob = torch.exp(-(distance ** 2) / (2 * (radius / 3) ** 2))
                
                cloud_mask = torch.maximum(cloud_mask, blob)
            
            # Smooth and threshold
            cloud_mask = F.gaussian_blur(cloud_mask.unsqueeze(0).unsqueeze(0), 
                                       kernel_size=5, sigma=2.0).squeeze()
            cloud_mask = (cloud_mask > 0.3).float()
            
            return cloud_mask
        
        def _noise_injection(self, sequence: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Add various types of noise to simulate sensor artifacts."""
            if random.random() > self.noise_injection_prob:
                return sequence, mask
            
            noise_type = random.choice(['gaussian', 'salt_pepper', 'striping'])
            
            if noise_type == 'gaussian':
                # Additive Gaussian noise
                noise_std = random.uniform(0.01, 0.05)
                noise = torch.normal(0, noise_std, size=sequence.shape)
                sequence = sequence + noise
                
            elif noise_type == 'salt_pepper':
                # Salt and pepper noise
                noise_prob = random.uniform(0.001, 0.01)
                salt_pepper = torch.rand(sequence.shape)
                sequence[salt_pepper < noise_prob / 2] = 0  # Pepper
                sequence[salt_pepper > 1 - noise_prob / 2] = 1  # Salt
                
            elif noise_type == 'striping':
                # Systematic striping artifacts
                T, C, H, W = sequence.shape
                stripe_intensity = random.uniform(0.02, 0.08)
                
                # Random stripes along one dimension
                if random.random() > 0.5:
                    # Vertical stripes
                    stripe_pattern = torch.sin(torch.arange(W) * 0.1) * stripe_intensity
                    sequence = sequence + stripe_pattern.view(1, 1, 1, -1)
                else:
                    # Horizontal stripes
                    stripe_pattern = torch.sin(torch.arange(H) * 0.1) * stripe_intensity
                    sequence = sequence + stripe_pattern.view(1, 1, -1, 1)
            
            return torch.clamp(sequence, 0, 1), mask
        
        def _seasonal_shift(self, sequence: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Simulate seasonal/phenological shifts."""
            if random.random() > self.seasonal_shift_prob:
                return sequence, mask
            
            T = sequence.shape[0]
            if T < 4:  # Need at least 4 time steps for meaningful shift
                return sequence, mask
            
            # Random circular shift
            shift_amount = random.randint(1, T - 1)
            sequence_shifted = torch.roll(sequence, shift_amount, dims=0)
            mask_shifted = torch.roll(mask, shift_amount, dims=0)
            
            return sequence_shifted, mask_shifted

Mixup and CutMix for Time Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class TemporalMixup:
        """Mixup augmentation adapted for time series data."""
        
        def __init__(self, alpha: float = 0.4, prob: float = 0.5):
            self.alpha = alpha
            self.prob = prob
        
        def __call__(self, sequences: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
            """Apply temporal mixup."""
            if random.random() > self.prob:
                return sequences, labels, 1.0
            
            batch_size = sequences.shape[0]
            
            # Sample mixing coefficient
            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1.0
            
            # Random shuffle for mixing pairs
            shuffle_idx = torch.randperm(batch_size)
            
            # Mix sequences
            mixed_sequences = lam * sequences + (1 - lam) * sequences[shuffle_idx]
            
            # Return mixed data with lambda for label mixing
            return mixed_sequences, labels[shuffle_idx], lam
    
    class TemporalCutMix:
        """CutMix augmentation for time series."""
        
        def __init__(self, alpha: float = 1.0, prob: float = 0.5):
            self.alpha = alpha
            self.prob = prob
        
        def __call__(self, sequences: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
            """Apply temporal cutmix."""
            if random.random() > self.prob:
                return sequences, labels, 1.0
            
            batch_size, T, C, H, W = sequences.shape
            
            # Sample mixing ratio
            lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
            
            # Random shuffle
            shuffle_idx = torch.randperm(batch_size)
            
            # Determine cut region (temporal dimension)
            cut_t = int(T * np.sqrt(1 - lam))
            cut_t = max(1, min(T - 1, cut_t))
            
            # Random temporal window
            t_start = random.randint(0, T - cut_t)
            t_end = t_start + cut_t
            
            # Apply cutmix
            mixed_sequences = sequences.clone()
            mixed_sequences[:, t_start:t_end] = sequences[shuffle_idx][:, t_start:t_end]
            
            # Adjust lambda based on actual cut ratio
            lam = 1 - (cut_t / T)
            
            return mixed_sequences, labels[shuffle_idx], lam

Multi-Modal Integration
-----------------------

Fusion with Auxiliary Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class MultiModalSITSFormer(nn.Module):
        """SITS-Former with multi-modal data integration."""
        
        def __init__(
            self,
            # Core SITS-Former parameters
            sits_config: dict,
            # Multi-modal configuration
            use_dem: bool = False,
            use_weather: bool = False,
            use_location: bool = False,
            use_metadata: bool = False,
            # Fusion strategy
            fusion_strategy: str = 'late',  # 'early', 'middle', 'late'
            fusion_dim: int = 128,
        ):
            super().__init__()
            
            # Core SITS-Former model
            self.sits_model = CustomSITSFormer(**sits_config)
            embed_dim = sits_config['embed_dim']
            
            # Auxiliary modality encoders
            self.modalities = {}
            aux_dim_total = 0
            
            if use_dem:
                self.dem_encoder = DEMEncoder(output_dim=64)
                self.modalities['dem'] = self.dem_encoder
                aux_dim_total += 64
            
            if use_weather:
                self.weather_encoder = WeatherEncoder(output_dim=32)
                self.modalities['weather'] = self.weather_encoder
                aux_dim_total += 32
            
            if use_location:
                self.location_encoder = LocationEncoder(output_dim=16)
                self.modalities['location'] = self.location_encoder
                aux_dim_total += 16
            
            if use_metadata:
                self.metadata_encoder = MetadataEncoder(output_dim=32)
                self.modalities['metadata'] = self.metadata_encoder
                aux_dim_total += 32
            
            # Fusion modules
            self.fusion_strategy = fusion_strategy
            
            if fusion_strategy == 'early':
                # Fuse at input level
                self.early_fusion = nn.Linear(aux_dim_total, embed_dim)
                
            elif fusion_strategy == 'middle':
                # Fuse at feature level
                self.middle_fusion = CrossModalAttention(embed_dim, aux_dim_total)
                
            elif fusion_strategy == 'late':
                # Fuse at decision level
                self.late_fusion = nn.Sequential(
                    nn.Linear(embed_dim + aux_dim_total, fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(fusion_dim, sits_config['num_classes'])
                )
                
            # Multi-modal attention for importance weighting
            self.modal_attention = ModalityAttention(embed_dim, len(self.modalities))
        
        def forward(self, sits_data: torch.Tensor, auxiliary_data: dict, 
                   masks: Optional[torch.Tensor] = None):
            """Forward pass with multi-modal fusion."""
            
            # Encode auxiliary modalities
            aux_features = []
            for modal_name, encoder in self.modalities.items():
                if modal_name in auxiliary_data:
                    aux_feat = encoder(auxiliary_data[modal_name])
                    aux_features.append(aux_feat)
            
            if not aux_features:
                # No auxiliary data, use standard SITS-Former
                return self.sits_model(sits_data, masks)
            
            # Concatenate auxiliary features
            aux_concat = torch.cat(aux_features, dim=-1)
            
            if self.fusion_strategy == 'early':
                # Early fusion: modify input
                aux_embedded = self.early_fusion(aux_concat)
                # Add to positional embeddings or patch embeddings
                sits_features = self.sits_model.forward_features(sits_data, masks)
                sits_features = sits_features + aux_embedded.unsqueeze(1)
                output = self.sits_model.head(sits_features.mean(dim=1))
                
            elif self.fusion_strategy == 'middle':
                # Middle fusion: cross-modal attention
                sits_features = self.sits_model.forward_features(sits_data, masks)
                fused_features = self.middle_fusion(sits_features, aux_concat)
                output = self.sits_model.head(fused_features.mean(dim=1))
                
            else:  # late fusion
                # Late fusion: concatenate final features
                sits_output = self.sits_model.forward_features(sits_data, masks)
                sits_pooled = sits_output.mean(dim=1)
                
                # Concatenate with auxiliary features
                combined_features = torch.cat([sits_pooled, aux_concat], dim=-1)
                output = self.late_fusion(combined_features)
            
            return output
    
    class CrossModalAttention(nn.Module):
        """Cross-modal attention for feature fusion."""
        
        def __init__(self, sits_dim: int, aux_dim: int, num_heads: int = 8):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                sits_dim, num_heads, batch_first=True
            )
            self.aux_proj = nn.Linear(aux_dim, sits_dim)
            self.norm = nn.LayerNorm(sits_dim)
        
        def forward(self, sits_features: torch.Tensor, aux_features: torch.Tensor):
            # Project auxiliary features to SITS dimension
            aux_proj = self.aux_proj(aux_features).unsqueeze(1)  # [B, 1, D]
            
            # Cross-attention: SITS queries auxiliary
            attended, _ = self.attention(sits_features, aux_proj, aux_proj)
            
            # Residual connection
            output = self.norm(sits_features + attended)
            return output

Experimental Features
---------------------

Self-Supervised Learning
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class SITSFormerSSL(nn.Module):
        """Self-supervised SITS-Former with various pretext tasks."""
        
        def __init__(self, base_model: nn.Module, ssl_tasks: List[str]):
            super().__init__()
            self.base_model = base_model
            self.ssl_tasks = ssl_tasks
            
            # Create heads for each SSL task
            self.ssl_heads = nn.ModuleDict()
            
            for task in ssl_tasks:
                if task == 'temporal_order':
                    self.ssl_heads[task] = nn.Linear(base_model.embed_dim, 2)  # Binary classification
                elif task == 'masked_reconstruction':
                    self.ssl_heads[task] = nn.Linear(base_model.embed_dim, base_model.embed_dim)
                elif task == 'rotation_prediction':
                    self.ssl_heads[task] = nn.Linear(base_model.embed_dim, 4)  # 4 rotations
                elif task == 'next_frame_prediction':
                    self.ssl_heads[task] = nn.Linear(base_model.embed_dim, base_model.embed_dim)
        
        def forward(self, x: torch.Tensor, ssl_task: str = None, **kwargs):
            """Forward pass for SSL training."""
            
            # Extract features
            features = self.base_model.forward_features(x)
            
            if ssl_task is None:
                # Standard classification
                return self.base_model.head(features.mean(dim=1))
            
            # SSL task-specific processing
            if ssl_task == 'temporal_order':
                return self._temporal_order_prediction(features, **kwargs)
            elif ssl_task == 'masked_reconstruction':
                return self._masked_reconstruction(features, **kwargs)
            elif ssl_task == 'rotation_prediction':
                return self._rotation_prediction(features)
            elif ssl_task == 'next_frame_prediction':
                return self._next_frame_prediction(features, **kwargs)
        
        def _temporal_order_prediction(self, features: torch.Tensor, 
                                     shuffled_indices: torch.Tensor):
            """Predict if temporal sequence is in correct order."""
            pooled_features = features.mean(dim=1)
            return self.ssl_heads['temporal_order'](pooled_features)
        
        def _masked_reconstruction(self, features: torch.Tensor, 
                                 mask_indices: torch.Tensor):
            """Reconstruct masked temporal segments."""
            masked_features = features[mask_indices]
            return self.ssl_heads['masked_reconstruction'](masked_features)
    
    class SSLDataAugmentation:
        """Data augmentation for self-supervised learning."""
        
        def temporal_shuffle(self, sequence: torch.Tensor, prob: float = 0.5):
            """Create temporally shuffled sequences for order prediction."""
            if random.random() > prob:
                return sequence, torch.tensor(0)  # Original order
            
            T = sequence.shape[0]
            shuffle_idx = torch.randperm(T)
            shuffled_sequence = sequence[shuffle_idx]
            
            return shuffled_sequence, torch.tensor(1)  # Shuffled
        
        def mask_temporal_segments(self, sequence: torch.Tensor, 
                                 mask_ratio: float = 0.3):
            """Mask random temporal segments."""
            T = sequence.shape[0]
            num_mask = int(T * mask_ratio)
            
            mask_indices = torch.randperm(T)[:num_mask]
            masked_sequence = sequence.clone()
            
            # Replace with learnable mask token
            masked_sequence[mask_indices] = 0  # Or use learnable mask token
            
            return masked_sequence, mask_indices, sequence[mask_indices]

Neural Architecture Search (NAS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class SITSFormerNAS(nn.Module):
        """Differentiable Neural Architecture Search for SITS-Former."""
        
        def __init__(self, config: dict):
            super().__init__()
            self.config = config
            
            # Define search spaces
            self.attention_choices = nn.ModuleList([
                StandardAttention(config['embed_dim'], config['num_heads']),
                LinearAttention(config['embed_dim'], config['num_heads']),
                SparseAttention(config['embed_dim'], config['num_heads'])
            ])
            
            self.mlp_choices = nn.ModuleList([
                StandardMLP(config['embed_dim'], config['mlp_ratio']),
                ConvMLP(config['embed_dim'], config['mlp_ratio']),
                GatedMLP(config['embed_dim'], config['mlp_ratio'])
            ])
            
            # Architecture weights (learnable)
            self.attention_weights = nn.Parameter(torch.randn(len(self.attention_choices)))
            self.mlp_weights = nn.Parameter(torch.randn(len(self.mlp_choices)))
            
            # Temperature for Gumbel softmax
            self.temperature = 1.0
        
        def forward(self, x: torch.Tensor):
            """Forward pass with architecture sampling."""
            
            # Sample architecture using Gumbel softmax
            attention_probs = F.gumbel_softmax(
                self.attention_weights, tau=self.temperature, hard=False
            )
            mlp_probs = F.gumbel_softmax(
                self.mlp_weights, tau=self.temperature, hard=False
            )
            
            # Weighted combination of architectural choices
            attention_out = sum(
                prob * choice(x) 
                for prob, choice in zip(attention_probs, self.attention_choices)
            )
            
            mlp_out = sum(
                prob * choice(attention_out)
                for prob, choice in zip(mlp_probs, self.mlp_choices)
            )
            
            return mlp_out
        
        def get_architecture(self):
            """Get the current best architecture."""
            attention_idx = self.attention_weights.argmax().item()
            mlp_idx = self.mlp_weights.argmax().item()
            
            return {
                'attention': attention_idx,
                'mlp': mlp_idx,
                'attention_weights': F.softmax(self.attention_weights, dim=0),
                'mlp_weights': F.softmax(self.mlp_weights, dim=0)
            }

Performance Optimization
------------------------

Memory Efficient Training
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class MemoryEfficientSITSFormer(nn.Module):
        """Memory-efficient SITS-Former with gradient checkpointing."""
        
        def __init__(self, config: dict, use_checkpointing: bool = True,
                     use_8bit_optimizer: bool = False):
            super().__init__()
            
            self.use_checkpointing = use_checkpointing
            self.base_model = CustomSITSFormer(**config)
            
            # Enable gradient checkpointing
            if use_checkpointing:
                self._enable_gradient_checkpointing()
        
        def _enable_gradient_checkpointing(self):
            """Enable gradient checkpointing for memory efficiency."""
            def checkpoint_wrapper(module):
                def forward(*args, **kwargs):
                    return checkpoint.checkpoint(module, *args, **kwargs)
                return forward
            
            # Apply checkpointing to transformer blocks
            for block in self.base_model.blocks:
                block.forward = checkpoint_wrapper(block.forward)
        
        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
            return self.base_model(x, mask)
    
    class AdaptiveBatchSize:
        """Automatically adjust batch size based on memory usage."""
        
        def __init__(self, initial_batch_size: int = 32, memory_threshold: float = 0.9):
            self.initial_batch_size = initial_batch_size
            self.current_batch_size = initial_batch_size
            self.memory_threshold = memory_threshold
            
        def adjust_batch_size(self):
            """Adjust batch size based on GPU memory usage."""
            if not torch.cuda.is_available():
                return self.current_batch_size
            
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            if memory_usage > self.memory_threshold and self.current_batch_size > 1:
                self.current_batch_size = max(1, self.current_batch_size // 2)
                torch.cuda.empty_cache()
                print(f"Reduced batch size to {self.current_batch_size}")
                
            elif memory_usage < 0.7 and self.current_batch_size < self.initial_batch_size:
                self.current_batch_size = min(self.initial_batch_size, self.current_batch_size * 2)
                print(f"Increased batch size to {self.current_batch_size}")
            
            return self.current_batch_size

Configuration Management
------------------------

YAML Configuration System
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # config_manager.py
    
    import yaml
    import os
    from typing import Dict, Any, Optional
    from dataclasses import dataclass, asdict
    from pathlib import Path
    
    @dataclass
    class ModelConfig:
        # Architecture
        img_size: int = 224
        patch_size: int = 16
        in_channels: int = 13
        num_classes: int = 10
        embed_dim: int = 256
        num_layers: int = 8
        num_heads: int = 8
        mlp_ratio: float = 4.0
        dropout: float = 0.1
        
        # Advanced features
        attention_type: str = 'standard'
        use_checkpointing: bool = False
        use_flash_attention: bool = False
    
    @dataclass
    class TrainingConfig:
        # Basic training
        batch_size: int = 32
        num_epochs: int = 100
        learning_rate: float = 1e-4
        weight_decay: float = 1e-2
        
        # Advanced training
        use_mixed_precision: bool = True
        gradient_clip_norm: float = 1.0
        warmup_epochs: int = 5
        scheduler_type: str = 'cosine'
        
        # Augmentation
        use_advanced_augmentation: bool = True
        mixup_alpha: float = 0.4
        cutmix_alpha: float = 1.0
    
    @dataclass
    class DataConfig:
        # Data paths
        train_data_path: str = "data/train"
        val_data_path: str = "data/val"
        test_data_path: str = "data/test"
        
        # Data processing
        normalize: bool = True
        mean: List[float] = None
        std: List[float] = None
        max_sequence_length: int = 20
    
    class ConfigManager:
        """Manage experiment configurations."""
        
        def __init__(self, config_path: Optional[str] = None):
            self.config_path = config_path
            self.model_config = ModelConfig()
            self.training_config = TrainingConfig()
            self.data_config = DataConfig()
        
        @classmethod
        def from_yaml(cls, config_path: str) -> 'ConfigManager':
            """Load configuration from YAML file."""
            manager = cls(config_path)
            manager.load_yaml(config_path)
            return manager
        
        def load_yaml(self, config_path: str):
            """Load configuration from YAML."""
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Update configurations
            if 'model' in config_dict:
                self.model_config = ModelConfig(**config_dict['model'])
            if 'training' in config_dict:
                self.training_config = TrainingConfig(**config_dict['training'])
            if 'data' in config_dict:
                self.data_config = DataConfig(**config_dict['data'])
        
        def save_yaml(self, save_path: str):
            """Save configuration to YAML."""
            config_dict = {
                'model': asdict(self.model_config),
                'training': asdict(self.training_config),
                'data': asdict(self.data_config)
            }
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        def update_config(self, config_updates: Dict[str, Any]):
            """Update configuration with new values."""
            for section, updates in config_updates.items():
                if section == 'model':
                    for key, value in updates.items():
                        setattr(self.model_config, key, value)
                elif section == 'training':
                    for key, value in updates.items():
                        setattr(self.training_config, key, value)
                elif section == 'data':
                    for key, value in updates.items():
                        setattr(self.data_config, key, value)
    
    # Example usage:
    # config = ConfigManager.from_yaml('configs/experiment1.yaml')
    # model = CustomSITSFormer(**asdict(config.model_config))

Example Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # configs/base_config.yaml
    
    model:
      img_size: 224
      patch_size: 16
      in_channels: 13
      num_classes: 10
      embed_dim: 256
      num_layers: 8
      num_heads: 8
      mlp_ratio: 4.0
      dropout: 0.1
      attention_type: "standard"
      use_checkpointing: false
      use_flash_attention: false
    
    training:
      batch_size: 32
      num_epochs: 100
      learning_rate: 1e-4
      weight_decay: 1e-2
      use_mixed_precision: true
      gradient_clip_norm: 1.0
      warmup_epochs: 5
      scheduler_type: "cosine"
      use_advanced_augmentation: true
      mixup_alpha: 0.4
      cutmix_alpha: 1.0
    
    data:
      train_data_path: "data/train"
      val_data_path: "data/val"
      test_data_path: "data/test"
      normalize: true
      max_sequence_length: 20

.. code-block:: yaml

    # configs/large_model.yaml
    
    model:
      img_size: 224
      patch_size: 16
      in_channels: 13
      num_classes: 10
      embed_dim: 384
      num_layers: 12
      num_heads: 12
      mlp_ratio: 4.0
      dropout: 0.1
      attention_type: "sparse"
      use_checkpointing: true
      use_flash_attention: true
    
    training:
      batch_size: 16  # Smaller due to larger model
      num_epochs: 150
      learning_rate: 8e-5
      weight_decay: 1e-2
      use_mixed_precision: true
      gradient_clip_norm: 0.5
      warmup_epochs: 10
      scheduler_type: "cosine"

This advanced configurations tutorial provides comprehensive customization options for SITS-Former, enabling researchers and practitioners to experiment with cutting-edge techniques, optimize for specific hardware constraints, and adapt the model for specialized applications. The modular design allows for easy experimentation and configuration management.
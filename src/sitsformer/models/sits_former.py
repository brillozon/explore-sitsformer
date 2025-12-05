"""
SITS-Former: Satellite Image Time Series Transformer

This module implements the core SITS-Former architecture for processing
satellite image time series data using transformer mechanisms.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """Patch embedding layer for satellite images.

    Converts input satellite images into patch embeddings suitable for transformer processing.
    Each image is divided into non-overlapping patches which are then linearly projected to
    create embeddings.

    Args:
        img_size (int): Size of input images (assumed square). Default: 224
        patch_size (int): Size of each patch (assumed square). Default: 16
        in_channels (int): Number of input channels/spectral bands. Default: 3
        embed_dim (int): Dimension of output embeddings. Default: 768

    Attributes:
        img_size (int): Input image size
        patch_size (int): Patch size
        num_patches (int): Total number of patches per image
        projection (nn.Conv2d): Convolution layer for patch projection

    Example:
        >>> patch_embed = PatchEmbedding(img_size=224, patch_size=16,
        ...                              in_channels=13, embed_dim=256)
        >>> x = torch.randn(4, 13, 224, 224)  # Batch of Sentinel-2 images
        >>> patches = patch_embed(x)  # Shape: [4, 196, 256]
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input images
        Returns:
            patches: [B, num_patches, embed_dim] patch embeddings
        """
        B, C, H, W = x.shape
        x = self.projection(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models.

    Adds positional information to input embeddings using sinusoidal functions.
    This allows the model to understand the relative or absolute position of
    tokens in the sequence.

    Args:
        embed_dim (int): Dimension of the embeddings
        max_len (int): Maximum sequence length. Default: 5000

    Attributes:
        pe (torch.Tensor): Precomputed positional encodings [max_len, embed_dim]

    Note:
        The positional encoding uses sine and cosine functions of different frequencies:
        PE(pos, 2i) = sin(pos / 10000^(2i/embed_dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))

    Example:
        >>> pos_enc = PositionalEncoding(embed_dim=256, max_len=100)
        >>> x = torch.randn(4, 50, 256)  # [batch, seq_len, embed_dim]
        >>> x_with_pos = pos_enc(x)
    """

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, embed_dim]
        Returns:
            x with positional encoding added
        """
        return x + self.pe[: x.size(0), :]


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for time series analysis.

    Multi-head self-attention specifically designed for processing temporal sequences
    of satellite imagery. This module captures temporal dependencies and patterns
    across time steps in the satellite image time series.

    Args:
        embed_dim (int): Dimension of input embeddings
        num_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout probability. Default: 0.1

    Attributes:
        embed_dim (int): Input embedding dimension
        num_heads (int): Number of attention heads
        head_dim (int): Dimension per attention head
        q_linear (nn.Linear): Linear projection for queries
        k_linear (nn.Linear): Linear projection for keys
        v_linear (nn.Linear): Linear projection for values
        out_linear (nn.Linear): Output linear projection
        dropout (nn.Dropout): Dropout layer

    Example:
        >>> temp_attn = TemporalAttention(embed_dim=256, num_heads=8)
        >>> x = torch.randn(4, 10, 256)  # [batch, seq_len, embed_dim]
        >>> output = temp_attn(x)  # Same shape as input
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, embed_dim] input sequence
            mask: [B, seq_len] attention mask
        Returns:
            output: [B, seq_len, embed_dim] attended sequence
        """
        B, seq_len, embed_dim = x.shape

        # Linear projections
        Q = self.q_linear(x)  # [B, seq_len, embed_dim]
        K = self.k_linear(x)  # [B, seq_len, embed_dim]
        V = self.v_linear(x)  # [B, seq_len, embed_dim]

        # Reshape for multi-head attention
        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, embed_dim)

        return self.out_linear(out)


class SITSFormerBlock(nn.Module):
    """Single SITS-Former transformer block.

    A complete transformer block consisting of temporal multi-head self-attention
    followed by a feed-forward network (MLP). Both sub-layers use residual connections
    and layer normalization.

    The block follows the standard transformer architecture:
    1. Layer normalization → Multi-head attention → Residual connection
    2. Layer normalization → MLP → Residual connection

    Args:
        embed_dim (int): Dimension of input embeddings
        num_heads (int): Number of attention heads. Default: 8
        mlp_ratio (int): Ratio for MLP hidden dimension (hidden_dim = embed_dim * mlp_ratio). Default: 4
        dropout (float): Dropout probability. Default: 0.1

    Attributes:
        norm1 (nn.LayerNorm): Layer normalization before attention
        temporal_attn (TemporalAttention): Multi-head self-attention module
        norm2 (nn.LayerNorm): Layer normalization before MLP
        mlp (nn.Sequential): Feed-forward network

    Example:
        >>> block = SITSFormerBlock(embed_dim=256, num_heads=8, mlp_ratio=4)
        >>> x = torch.randn(4, 10, 256)  # [batch, seq_len, embed_dim]
        >>> output = block(x)  # Same shape as input
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, embed_dim] input sequence
            mask: [B, seq_len] attention mask
        Returns:
            output: [B, seq_len, embed_dim] transformed sequence
        """
        # Temporal attention with residual connection
        x = x + self.temporal_attn(self.norm1(x), mask)

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class SITSFormer(nn.Module):
    """SITS-Former: Satellite Image Time Series Transformer.

    A transformer-based neural network for processing and classifying satellite image
    time series (SITS) data. The model processes temporal sequences of satellite images
    by converting them to patch embeddings and applying transformer attention mechanisms
    to capture both spatial and temporal patterns.

    Architecture Overview:
        1. Patch Embedding: Convert satellite images to patch embeddings
        2. Positional Encoding: Add positional information to embeddings
        3. Temporal Token: Special token for temporal aggregation
        4. Transformer Blocks: Stack of self-attention and feed-forward layers
        5. Classification Head: Final linear layer for class prediction

    Args:
        img_size (int): Size of input images (assumed square). Default: 224
        patch_size (int): Size of patches (assumed square). Default: 16
        in_channels (int): Number of spectral bands in satellite imagery. Default: 13 (Sentinel-2)
        num_classes (int): Number of output classes for classification. Default: 10
        embed_dim (int): Dimension of patch embeddings. Default: 768
        num_layers (int): Number of transformer blocks. Default: 12
        num_heads (int): Number of attention heads in each block. Default: 12
        mlp_ratio (int): Ratio for MLP hidden dimension. Default: 4
        dropout (float): Dropout probability. Default: 0.1
        max_seq_len (int): Maximum sequence length (number of time steps). Default: 100

    Attributes:
        embed_dim (int): Embedding dimension
        num_patches (int): Number of patches per image
        patch_embed (PatchEmbedding): Patch embedding layer
        pos_encoding (PositionalEncoding): Positional encoding layer
        temporal_token (nn.Parameter): Learnable temporal aggregation token
        blocks (nn.ModuleList): List of transformer blocks
        norm (nn.LayerNorm): Final layer normalization
        classifier (nn.Linear): Classification head
        dropout (nn.Dropout): Dropout layer

    Example:
        Basic usage for land cover classification::

            import torch
            from ..models import SITSFormer

            # Create model for Sentinel-2 data with 10 land cover classes
            model = SITSFormer(
                img_size=224,
                patch_size=16,
                in_channels=13,    # Sentinel-2 spectral bands
                num_classes=10,    # Land cover classes
                embed_dim=256,
                num_layers=6,
                num_heads=8
            )

            # Input: batch of time series [batch, time_steps, channels, height, width]
            batch_size, time_steps = 4, 10
            x = torch.randn(batch_size, time_steps, 13, 224, 224)

            # Forward pass
            logits = model(x)  # Shape: [batch_size, num_classes]

            # Get predictions
            predictions = torch.argmax(logits, dim=1)

    Note:
        - Input images are expected to be normalized appropriately for the satellite sensor
        - The model expects time series as input with shape [B, T, C, H, W]
        - For single-image classification, use time_steps=1
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 13,  # Sentinel-2 bands
        num_classes: int = 10,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            embed_dim, max_seq_len * self.num_patches
        )

        # Temporal token for aggregating temporal information
        self.temporal_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                SITSFormerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the SITS-Former model.

        Processes a batch of satellite image time series through the complete model pipeline:
        1. Convert images to patch embeddings
        2. Add temporal token for sequence aggregation
        3. Apply positional encoding
        4. Process through transformer blocks
        5. Extract temporal representation and classify

        Args:
            x (torch.Tensor): Input satellite image time series with shape [B, T, C, H, W]
                - B: Batch size
                - T: Number of time steps in the series
                - C: Number of spectral bands (channels)
                - H, W: Spatial dimensions (height, width)
            mask (torch.Tensor, optional): Attention mask with shape [B, T].
                Values of 1 indicate valid time steps, 0 indicates padding.
                If None, all time steps are considered valid.

        Returns:
            torch.Tensor: Classification logits with shape [B, num_classes]

        Example:
            >>> model = SITSFormer(in_channels=13, num_classes=10)
            >>> # Batch of 4 time series, each with 8 time steps
            >>> x = torch.randn(4, 8, 13, 224, 224)
            >>> mask = torch.ones(4, 8)  # All time steps valid
            >>> logits = model(x, mask)  # Shape: [4, 10]
            >>> probs = torch.softmax(logits, dim=1)  # Convert to probabilities

        Note:
            The temporal token aggregates information across all time steps and spatial
            patches before final classification. This design allows the model to capture
            complex spatio-temporal patterns in satellite imagery.
        """
        B, T, C, H, W = x.shape

        # Reshape to process all images together
        x = rearrange(x, "b t c h w -> (b t) c h w")

        # Extract patches
        patches = self.patch_embed(x)  # [(B*T), num_patches, embed_dim]

        # Reshape back to time series format
        patches = rearrange(patches, "(b t) n d -> b (t n) d", b=B, t=T)

        # Add temporal token
        temporal_tokens = repeat(self.temporal_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([temporal_tokens, patches], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        # Create attention mask including temporal token
        if mask is not None:
            # Repeat mask for each patch
            patch_mask = repeat(mask, "b t -> b (t n)", n=self.num_patches)
            # Add mask for temporal token (always attend)
            temporal_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            full_mask = torch.cat([temporal_mask, patch_mask], dim=1)
        else:
            full_mask = None

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, full_mask)

        # Extract temporal token and classify
        temporal_repr = x[:, 0]  # [B, embed_dim]
        temporal_repr = self.norm(temporal_repr)
        logits = self.classifier(temporal_repr)

        return logits


def create_sits_former(config: dict) -> SITSFormer:
    """Create SITS-Former model from configuration dictionary.

    Factory function to instantiate a SITS-Former model using a configuration dictionary.
    This provides a convenient way to create models with predefined configurations or
    load models from saved configuration files.

    Args:
        config (dict): Configuration dictionary containing model parameters.
            Supported keys:
            - 'img_size' (int): Input image size. Default: 224
            - 'patch_size' (int): Patch size for patch embedding. Default: 16
            - 'in_channels' (int): Number of input spectral bands. Default: 13
            - 'num_classes' (int): Number of output classes. Default: 10
            - 'embed_dim' (int): Embedding dimension. Default: 768
            - 'num_layers' (int): Number of transformer layers. Default: 12
            - 'num_heads' (int): Number of attention heads. Default: 12
            - 'mlp_ratio' (int): MLP expansion ratio. Default: 4
            - 'dropout' (float): Dropout probability. Default: 0.1
            - 'max_seq_len' (int): Maximum sequence length. Default: 100

    Returns:
        SITSFormer: Configured SITS-Former model instance

    Example:
        >>> # Define model configuration
        >>> config = {
        ...     'img_size': 224,
        ...     'patch_size': 16,
        ...     'in_channels': 13,  # Sentinel-2 bands
        ...     'num_classes': 7,   # CORINE land cover classes
        ...     'embed_dim': 256,
        ...     'num_layers': 8,
        ...     'num_heads': 8,
        ...     'dropout': 0.1
        ... }
        >>> model = create_sits_former(config)

        >>> # Or load from YAML configuration file
        >>> import yaml
        >>> with open('model_config.yaml', 'r') as f:
        ...     config = yaml.safe_load(f)
        >>> model = create_sits_former(config['model'])

    Note:
        This function is particularly useful when working with configuration files
        or when you need to create multiple model variants programmatically.
    """
    return SITSFormer(
        img_size=config.get("img_size", 224),
        patch_size=config.get("patch_size", 16),
        in_channels=config.get("in_channels", 13),
        num_classes=config.get("num_classes", 10),
        embed_dim=config.get("embed_dim", 768),
        num_layers=config.get("num_layers", 12),
        num_heads=config.get("num_heads", 12),
        mlp_ratio=config.get("mlp_ratio", 4),
        dropout=config.get("dropout", 0.1),
        max_seq_len=config.get("max_seq_len", 100),
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the model
    model = SITSFormer(
        img_size=64,
        patch_size=16,
        in_channels=13,
        num_classes=5,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        max_seq_len=10,
    )

    # Create dummy input: batch_size=2, time_steps=5, channels=13, height=64, width=64
    x = torch.randn(2, 5, 13, 64, 64)
    mask = torch.ones(2, 5)  # No masking

    # Forward pass
    with torch.no_grad():
        output = model(x, mask)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

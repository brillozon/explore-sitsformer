"""
Configuration utilities for SITS-Former project.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 13
    num_classes: int = 10
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout: float = 0.1
    max_seq_len: int = 50


@dataclass 
class TrainingConfig:
    """Training configuration dataclass."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    optimizer: str = "adamw"
    scheduler: Optional[str] = "cosine"
    criterion: str = "cross_entropy"
    use_amp: bool = True
    gradient_clip_val: Optional[float] = 1.0


@dataclass
class DataConfig:
    """Data configuration dataclass."""
    data_dir: str = "data/raw"
    metadata_file: str = "data/metadata.csv"
    sequence_length: int = 10
    image_size: int = 224
    normalize: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: Path):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, indent=2, default_flow_style=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries recursively.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and set default values for configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration
    """
    # Set device if auto
    if config.get('hardware', {}).get('device') == 'auto':
        if 'hardware' not in config:
            config['hardware'] = {}
        config['hardware']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Validate model configuration
    model_config = config.get('model', {})
    assert model_config.get('embed_dim', 768) % model_config.get('num_heads', 12) == 0, \
        "embed_dim must be divisible by num_heads"
    
    # Validate data splitting ratios
    data_config = config.get('data', {})
    ratios = [
        data_config.get('train_ratio', 0.7),
        data_config.get('val_ratio', 0.15), 
        data_config.get('test_ratio', 0.15)
    ]
    assert abs(sum(ratios) - 1.0) < 1e-6, "Data split ratios must sum to 1.0"
    
    # Validate training configuration
    training_config = config.get('training', {})
    assert training_config.get('batch_size', 32) > 0, "batch_size must be positive"
    assert training_config.get('learning_rate', 1e-4) > 0, "learning_rate must be positive"
    
    return config


def create_experiment_name(config: Dict[str, Any]) -> str:
    """
    Create experiment name from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Experiment name string
    """
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    
    name_parts = [
        f"embed{model_cfg.get('embed_dim', 768)}",
        f"layers{model_cfg.get('num_layers', 12)}",
        f"heads{model_cfg.get('num_heads', 12)}",
        f"bs{training_cfg.get('batch_size', 32)}",
        f"lr{training_cfg.get('learning_rate', 1e-4):.0e}"
    ]
    
    return "_".join(name_parts)


def get_config_summary(config: Dict[str, Any]) -> str:
    """
    Get a summary string of the configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration summary string
    """
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    
    summary_lines = [
        "Configuration Summary:",
        f"Model: {model_cfg.get('embed_dim', 768)}d, {model_cfg.get('num_layers', 12)} layers, {model_cfg.get('num_heads', 12)} heads",
        f"Data: {data_cfg.get('sequence_length', 10)} timesteps, {data_cfg.get('image_size', 224)}px images",
        f"Training: {training_cfg.get('batch_size', 32)} batch size, {training_cfg.get('learning_rate', 1e-4):.0e} LR, {training_cfg.get('epochs', 100)} epochs",
        f"Classes: {model_cfg.get('num_classes', 10)}"
    ]
    
    return "\n".join(summary_lines)


# Predefined configurations
SMALL_CONFIG = {
    'model': {
        'img_size': 64,
        'embed_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'max_seq_len': 10
    },
    'training': {
        'epochs': 20,
        'batch_size': 16,
        'learning_rate': 5e-4
    },
    'data': {
        'sequence_length': 5,
        'image_size': 64
    }
}

MEDIUM_CONFIG = {
    'model': {
        'img_size': 128,
        'embed_dim': 512,
        'num_layers': 8,
        'num_heads': 8,
        'max_seq_len': 25
    },
    'training': {
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 1e-4
    },
    'data': {
        'sequence_length': 10,
        'image_size': 128
    }
}

LARGE_CONFIG = {
    'model': {
        'img_size': 256,
        'embed_dim': 1024,
        'num_layers': 16,
        'num_heads': 16,
        'max_seq_len': 100
    },
    'training': {
        'epochs': 200,
        'batch_size': 8,
        'learning_rate': 1e-4
    },
    'data': {
        'sequence_length': 20,
        'image_size': 256
    }
}


def get_preset_config(preset: str) -> Dict[str, Any]:
    """
    Get a predefined configuration preset.
    
    Args:
        preset: Preset name ('small', 'medium', 'large')
        
    Returns:
        Configuration dictionary
    """
    presets = {
        'small': SMALL_CONFIG,
        'medium': MEDIUM_CONFIG,
        'large': LARGE_CONFIG
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset].copy()


# Example usage
if __name__ == "__main__":
    # Load default config
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        config = load_config(config_path)
        print(get_config_summary(config))
        print(f"Experiment name: {create_experiment_name(config)}")
    else:
        print("Default config not found, using small preset")
        config = get_preset_config('small')
        print(get_config_summary(config))
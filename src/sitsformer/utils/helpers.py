"""
Utility functions for SITS-Former project.
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: Path, experiment_name: str) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_dir: Directory to save log files
        experiment_name: Name of the experiment

    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger("SITSFormer")
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get torch device based on availability and preference.

    Args:
        device_str: Device string ('cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        Torch device
    """
    if device_str is None or device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU.")
        device = torch.device("cpu")

    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def save_metrics(metrics: Dict[str, Any], save_path: Path):
    """
    Save metrics dictionary to JSON file.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save metrics
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, np.generic):
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = value

    with open(save_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)


def load_metrics(load_path: Path) -> Dict[str, Any]:
    """
    Load metrics dictionary from JSON file.

    Args:
        load_path: Path to load metrics from

    Returns:
        Dictionary of metrics
    """
    with open(load_path, "r") as f:
        metrics = json.load(f)
    return metrics


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        print(f"{self.name} took {format_time(self.elapsed_time)}")


class ProgressTracker:
    """Track training progress and estimate remaining time."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []

    def update(self, step: int):
        """Update current step."""
        self.current_step = step
        current_time = time.time()
        if self.current_step > 0:
            self.step_times.append(current_time - self.start_time)

    def get_eta(self) -> str:
        """Get estimated time to completion."""
        if len(self.step_times) == 0:
            return "Unknown"

        avg_step_time = np.mean(self.step_times[-10:])  # Use last 10 steps
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = remaining_steps * avg_step_time

        return format_time(eta_seconds)

    def get_progress(self) -> str:
        """Get progress string."""
        progress = self.current_step / self.total_steps * 100
        return f"{progress:.1f}% ({self.current_step}/{self.total_steps})"


def create_checkpoint_dir(base_dir: Path, experiment_name: str) -> Path:
    """
    Create checkpoint directory with timestamp.

    Args:
        base_dir: Base directory for checkpoints
        experiment_name: Name of the experiment

    Returns:
        Path to checkpoint directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = base_dir / f"{experiment_name}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.

    Returns:
        Dictionary with memory usage information
    """
    memory_info = {}

    # CPU memory
    import psutil

    process = psutil.Process(os.getpid())
    memory_info["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024

    # GPU memory
    if torch.cuda.is_available():
        memory_info["gpu_memory_allocated_mb"] = (
            torch.cuda.memory_allocated() / 1024 / 1024
        )
        memory_info["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        memory_info["gpu_memory_max_mb"] = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
        )

    return memory_info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def get_git_info() -> Dict[str, str]:
    """
    Get git repository information.

    Returns:
        Dictionary with git information
    """
    import subprocess  # nosec B404 - subprocess needed for git operations
    import shutil

    git_info = {}

    # First check if git is available
    git_executable = shutil.which("git")
    if not git_executable:
        git_info["error"] = "git executable not found"
        return git_info

    try:
        # Get current commit hash
        git_info["commit_hash"] = (
            subprocess.check_output(  # nosec B603 - git commands are safe
                [git_executable, "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL,
                timeout=10  # Add timeout for security
            )
            .decode("utf-8")
            .strip()
        )

        # Get current branch
        git_info["branch"] = (
            subprocess.check_output(  # nosec B603 - git commands are safe
                [git_executable, "rev-parse", "--abbrev-ref", "HEAD"], 
                stderr=subprocess.DEVNULL,
                timeout=10  # Add timeout for security
            )
            .decode("utf-8")
            .strip()
        )

        # Check if repository is dirty
        try:
            subprocess.check_output(  # nosec B603 - git commands are safe
                [git_executable, "diff-index", "--quiet", "HEAD", "--"],
                stderr=subprocess.DEVNULL,
                timeout=10  # Add timeout for security
            )
            git_info["dirty"] = False
        except subprocess.CalledProcessError:
            git_info["dirty"] = True

    except (subprocess.TimeoutExpired, OSError, subprocess.CalledProcessError) as e:
        git_info["error"] = str(e)

    return git_info


def backup_code(backup_dir: Path, exclude_patterns: Optional[List[str]] = None):
    """
    Backup source code to directory.

    Args:
        backup_dir: Directory to backup code to
        exclude_patterns: Patterns to exclude from backup
    """
    import fnmatch
    import shutil

    if exclude_patterns is None:
        exclude_patterns = [
            "*.pyc",
            "__pycache__",
            ".git",
            "*.log",
            "checkpoints",
            "logs",
            ".pytest_cache",
            ".vscode",
            ".DS_Store",
        ]

    def should_exclude(path: str) -> bool:
        return any(fnmatch.fnmatch(path, pattern) for pattern in exclude_patterns)

    src_dir = Path.cwd()
    backup_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.rglob("*"):
        if should_exclude(str(item.relative_to(src_dir))):
            continue

        if item.is_file():
            rel_path = item.relative_to(src_dir)
            dest_path = backup_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_path)


# System information
def get_system_info() -> Dict[str, Any]:
    """
    Get system information.

    Returns:
        Dictionary with system information
    """
    import platform

    import psutil

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / 1024**3,
        "pytorch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

    return info


# Example usage
if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    print("Random seed set to 42")

    device = get_device()
    print(f"Using device: {device}")

    # Test timer
    with Timer("Test operation"):
        time.sleep(1)

    # Test memory usage
    memory = get_memory_usage()
    print(f"Memory usage: {memory}")

    # Test system info
    system_info = get_system_info()
    print(f"System info: {system_info}")

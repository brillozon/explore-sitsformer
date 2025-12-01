"""
Evaluation utilities and metrics for SITS-Former.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json


class ModelEvaluator:
    """Evaluation class for SITS-Former model."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Args:
            model: Trained SITS-Former model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions from the model.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        probabilities = np.concatenate(all_probabilities)
        true_labels = np.concatenate(all_labels)
        
        return predictions, probabilities, true_labels
    
    def evaluate(self, dataloader: DataLoader, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            class_names: List of class names for reporting
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions, probabilities, true_labels = self.predict(dataloader)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # Macro and micro averages
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        precision_micro = accuracy  # For multiclass, micro avg precision = accuracy
        recall_micro = accuracy     # For multiclass, micro avg recall = accuracy
        f1_micro = accuracy        # For multiclass, micro avg F1 = accuracy
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # ROC AUC (if binary or multiclass with probabilities)
        try:
            if len(np.unique(true_labels)) == 2:
                # Binary classification
                roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
            else:
                # Multiclass classification
                roc_auc = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='macro')
        except Exception:
            roc_auc = None
        
        results = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'num_samples': len(true_labels),
            'num_classes': len(np.unique(true_labels))
        }
        
        if class_names is not None:
            results['class_names'] = class_names
        
        return results
    
    def plot_confusion_matrix(self, 
                            true_labels: np.ndarray, 
                            predictions: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            save_path: Optional[Path] = None,
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            true_labels: True labels
            predictions: Predicted labels
            class_names: List of class names
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(true_labels, predictions)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names or range(cm.shape[1]),
            yticklabels=class_names or range(cm.shape[0]),
            ax=ax
        )
        
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_class_performance(self,
                             results: Dict[str, Any],
                             save_path: Optional[Path] = None,
                             figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot per-class performance metrics.
        
        Args:
            results: Results from evaluate method
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        precision = results['precision_per_class']
        recall = results['recall_per_class']
        f1 = results['f1_per_class']
        
        class_names = results.get('class_names', [f'Class {i}' for i in range(len(precision))])
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_results(self, results: Dict[str, Any], save_path: Path):
        """Save evaluation results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.generic):
                serializable_results[key] = value.item()
            else:
                serializable_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)


def calculate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Calculate FLOPs for the model (simplified estimation).
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        
    Returns:
        Estimated number of FLOPs
    """
    # This is a very simplified FLOP calculation
    # For more accurate results, use libraries like thop or fvcore
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Rough estimation: 2 FLOPs per parameter per forward pass
    estimated_flops = total_params * 2
    
    return estimated_flops


def model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Get model summary statistics.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB (assuming float32)
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    # Estimate FLOPs
    flops = calculate_flops(model, input_shape)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'estimated_flops': flops,
        'input_shape': input_shape
    }
    
    return summary


def compare_models(results_list: List[Dict[str, Any]], 
                  model_names: List[str],
                  save_path: Optional[Path] = None) -> plt.Figure:
    """
    Compare multiple models' performance.
    
    Args:
        results_list: List of evaluation results
        model_names: List of model names
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[metric] for results in results_list]
        
        ax = axes[i]
        bars = ax.bar(model_names, values, alpha=0.7)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(max(model_names, key=len)) > 10:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Example usage
if __name__ == "__main__":
    # This would typically be used with a trained model
    print("Evaluation utilities loaded successfully!")
    print("Usage example:")
    print("evaluator = ModelEvaluator(model)")
    print("results = evaluator.evaluate(test_loader)")
    print("evaluator.plot_confusion_matrix(true_labels, predictions)")
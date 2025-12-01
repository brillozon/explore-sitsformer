"""
Evaluation utilities and metrics for SITS-Former models.

This module provides comprehensive evaluation tools for assessing model performance
on satellite image time series classification tasks. It includes standard classification
metrics, remote sensing specific evaluations, model complexity analysis, and 
visualization utilities.

Key Components:
    ModelEvaluator: Main evaluation class with support for multiple metrics
    calculate_flops: Model complexity analysis (FLOPs, parameters, memory)
    model_summary: Detailed model architecture and statistics summary
    compare_models: Side-by-side model comparison utilities

Supported Metrics:
    - Classification: Accuracy, Precision, Recall, F1-Score
    - Multi-class: Macro/Micro averages, Per-class metrics
    - Remote Sensing: Overall Accuracy (OA), Kappa coefficient
    - Confusion matrices with class-wise analysis
    - ROC curves and AUC for binary/multi-class problems

Features:
    - Batch-wise evaluation for memory efficiency
    - Support for class imbalanced datasets
    - Temporal analysis for time series predictions
    - Visualization of results and confusion matrices
    - Statistical significance testing
    - Cross-validation support

Example:
    Comprehensive model evaluation::

        from sitsformer.evaluation import ModelEvaluator
        from sitsformer.models import SITSFormer
        
        # Create evaluator
        evaluator = ModelEvaluator(
            model=model,
            device='cuda',
            class_names=['Forest', 'Urban', 'Water', 'Agriculture']
        )
        
        # Evaluate on test set
        results = evaluator.evaluate(test_loader)
        
        # Print detailed results
        evaluator.print_results(results)
        
        # Generate visualizations
        evaluator.plot_confusion_matrix(results)
        evaluator.plot_classification_report(results)
        
        # Model complexity analysis
        flops, params = calculate_flops(model, input_shape=(10, 13, 224, 224))
        summary = model_summary(model)
"""

from .metrics import ModelEvaluator, calculate_flops, model_summary, compare_models

__all__ = ['ModelEvaluator', 'calculate_flops', 'model_summary', 'compare_models']
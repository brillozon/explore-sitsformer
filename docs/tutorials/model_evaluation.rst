Model Evaluation and Analysis
==============================

This tutorial covers comprehensive evaluation of SITS-Former models, including performance metrics, visualization techniques, error analysis, and model interpretation for satellite image time series classification.

Overview
--------

Model evaluation for satellite imagery involves several key aspects:

1. **Performance Metrics** - Accuracy, precision, recall, F1-score, confusion matrices
2. **Spatial Analysis** - Geographic performance patterns and spatial autocorrelation
3. **Temporal Analysis** - Performance across different time periods and seasons
4. **Class-specific Evaluation** - Per-class metrics and imbalanced dataset handling
5. **Model Interpretation** - Attention visualization and feature importance
6. **Robustness Testing** - Performance under different conditions

Prerequisites
-------------

Before evaluating your model, ensure you have:

- A trained SITS-Former model
- Test dataset with ground truth labels
- Evaluation environment setup

.. code-block:: python

    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report,
        roc_auc_score, average_precision_score
    )
    from sitsformer.models import SITSFormer
    from sitsformer.evaluation import ModelEvaluator, AttentionVisualizer

Basic Model Evaluation
-----------------------

Loading Model for Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def load_model_for_evaluation(checkpoint_path, device='cpu'):
        """Load trained model for evaluation."""
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = checkpoint.get('model_config', {})
        
        # Create model
        model = SITSFormer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        return model

    # Load your trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_for_evaluation('checkpoints/best_model.pth', device)

Prediction Generation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def generate_predictions(model, test_loader, device, return_probabilities=False):
        """Generate predictions for test dataset."""
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_features = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Generating predictions'):
                sequences = batch['sequence'].to(device)
                masks = batch.get('mask', None)
                if masks is not None:
                    masks = masks.to(device)
                labels = batch['label']
                
                # Forward pass
                if masks is not None:
                    outputs = model(sequences, masks)
                else:
                    outputs = model(sequences)
                
                # Get probabilities and predictions
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(labels.numpy())
                
                # Optionally extract features for analysis
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(sequences, masks)
                    all_features.extend(features.cpu().numpy())
        
        results = {
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities)
        }
        
        if all_features:
            results['features'] = np.array(all_features)
        
        return results

    # Generate predictions
    test_results = generate_predictions(model, test_loader, device, return_probabilities=True)

Performance Metrics
-------------------

Basic Classification Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def compute_classification_metrics(targets, predictions, probabilities, class_names=None):
        """Compute comprehensive classification metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Macro and micro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            targets, predictions, average='micro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        metrics_df = pd.DataFrame({
            'Class': class_names or [f'Class_{i}' for i in range(len(precision))],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Overall metrics
        overall_metrics = {
            'Overall Accuracy': accuracy,
            'Macro Precision': macro_precision,
            'Macro Recall': macro_recall,
            'Macro F1-Score': macro_f1,
            'Micro Precision': micro_precision,
            'Micro Recall': micro_recall,
            'Micro F1-Score': micro_f1,
            'Weighted Precision': weighted_precision,
            'Weighted Recall': weighted_recall,
            'Weighted F1-Score': weighted_f1
        }
        
        # Multi-class ROC AUC (if probabilities available)
        if probabilities is not None:
            try:
                # One-vs-Rest ROC AUC
                roc_auc_ovr = roc_auc_score(targets, probabilities, multi_class='ovr', average='macro')
                roc_auc_ovo = roc_auc_score(targets, probabilities, multi_class='ovo', average='macro')
                overall_metrics['ROC AUC (OvR)'] = roc_auc_ovr
                overall_metrics['ROC AUC (OvO)'] = roc_auc_ovo
            except ValueError:
                # Handle case with only one class in targets
                pass
        
        return metrics_df, overall_metrics

    # Compute metrics
    class_names = ['Water', 'Forest', 'Grassland', 'Cropland', 'Urban', 
                   'Bare Soil', 'Snow/Ice', 'Cloud', 'Shadow', 'Wetland']
    
    per_class_metrics, overall_metrics = compute_classification_metrics(
        test_results['targets'], 
        test_results['predictions'],
        test_results['probabilities'],
        class_names
    )
    
    print("Per-class Metrics:")
    print(per_class_metrics.round(3))
    print("\nOverall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

Confusion Matrix Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def plot_confusion_matrix(targets, predictions, class_names, normalize=True, 
                             figsize=(12, 10)):
        """Plot detailed confusion matrix with analysis."""
        
        # Compute confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_norm
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm_display = cm
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot confusion matrix
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(title)
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Plot raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Counts)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.show()
        
        # Analyze confusion patterns
        print("Confusion Matrix Analysis:")
        print("-" * 40)
        
        if normalize:
            # Find most confused classes
            np.fill_diagonal(cm_norm, 0)  # Remove diagonal
            max_confusion = np.unravel_index(cm_norm.argmax(), cm_norm.shape)
            print(f"Most confused classes: {class_names[max_confusion[0]]} -> {class_names[max_confusion[1]]} ({cm_norm[max_confusion]:.3f})")
            
            # Find classes with lowest recall (diagonal values in original normalized matrix)
            cm_norm_diag = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            worst_recall_idx = np.diagonal(cm_norm_diag).argmin()
            print(f"Lowest recall class: {class_names[worst_recall_idx]} ({np.diagonal(cm_norm_diag)[worst_recall_idx]:.3f})")
            
            # Find classes with lowest precision
            precision_per_class = np.diagonal(cm) / cm.sum(axis=0)
            worst_precision_idx = precision_per_class.argmin()
            print(f"Lowest precision class: {class_names[worst_precision_idx]} ({precision_per_class[worst_precision_idx]:.3f})")
        
        return cm

    # Plot confusion matrix
    confusion_matrix_result = plot_confusion_matrix(
        test_results['targets'], 
        test_results['predictions'], 
        class_names
    )

Temporal and Spatial Analysis
-----------------------------

Temporal Performance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_temporal_performance(test_results, temporal_metadata, class_names):
        """Analyze model performance across different temporal periods."""
        
        # Extract temporal information (month, season, year)
        # Assuming temporal_metadata contains 'acquisition_date' for each sample
        
        results_df = pd.DataFrame({
            'target': test_results['targets'],
            'prediction': test_results['predictions'],
            'correct': test_results['targets'] == test_results['predictions']
        })
        
        # Add temporal metadata
        if 'acquisition_date' in temporal_metadata:
            dates = pd.to_datetime(temporal_metadata['acquisition_date'])
            results_df['month'] = dates.dt.month
            results_df['season'] = dates.dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            results_df['year'] = dates.dt.year
            
            # Monthly performance
            monthly_accuracy = results_df.groupby('month')['correct'].mean()
            
            # Seasonal performance
            seasonal_accuracy = results_df.groupby('season')['correct'].mean()
            
            # Plot temporal trends
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Monthly accuracy
            monthly_accuracy.plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Monthly Accuracy')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Accuracy')
            ax1.set_xticklabels([f'M{i}' for i in range(1, 13)], rotation=45)
            
            # Seasonal accuracy
            seasonal_accuracy.plot(kind='bar', ax=ax2, color='lightgreen')
            ax2.set_title('Seasonal Accuracy')
            ax2.set_xlabel('Season')
            ax2.set_ylabel('Accuracy')
            ax2.set_xticklabels(seasonal_accuracy.index, rotation=45)
            
            # Class performance by season
            class_seasonal = results_df.groupby(['season', 'target'])['correct'].mean().unstack()
            sns.heatmap(class_seasonal.T, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax3)
            ax3.set_title('Class Accuracy by Season')
            ax3.set_ylabel('Class')
            
            # Temporal confusion patterns
            if len(results_df['year'].unique()) > 1:
                yearly_accuracy = results_df.groupby('year')['correct'].mean()
                yearly_accuracy.plot(kind='line', marker='o', ax=ax4, color='red')
                ax4.set_title('Yearly Accuracy Trend')
                ax4.set_xlabel('Year')
                ax4.set_ylabel('Accuracy')
            else:
                ax4.text(0.5, 0.5, 'Single year dataset', ha='center', va='center')
                ax4.set_title('Yearly Trend (N/A)')
            
            plt.tight_layout()
            plt.show()
            
            return {
                'monthly_accuracy': monthly_accuracy,
                'seasonal_accuracy': seasonal_accuracy,
                'class_seasonal_performance': class_seasonal
            }

Spatial Performance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_spatial_performance(test_results, spatial_metadata, plot_maps=True):
        """Analyze model performance across geographic regions."""
        
        # Assuming spatial_metadata contains 'latitude', 'longitude', 'region'
        results_df = pd.DataFrame({
            'target': test_results['targets'],
            'prediction': test_results['predictions'],
            'correct': test_results['targets'] == test_results['predictions'],
            'latitude': spatial_metadata.get('latitude', []),
            'longitude': spatial_metadata.get('longitude', []),
            'region': spatial_metadata.get('region', [])
        })
        
        spatial_analysis = {}
        
        # Regional performance
        if 'region' in results_df.columns and results_df['region'].notna().any():
            regional_accuracy = results_df.groupby('region')['correct'].mean()
            spatial_analysis['regional_accuracy'] = regional_accuracy
            
            print("Regional Performance:")
            for region, accuracy in regional_accuracy.items():
                print(f"  {region}: {accuracy:.3f}")
        
        # Spatial autocorrelation analysis
        if 'latitude' in results_df.columns and 'longitude' in results_df.columns:
            # Simple spatial binning
            lat_bins = pd.cut(results_df['latitude'], bins=5, labels=['S', 'S-C', 'C', 'N-C', 'N'])
            lon_bins = pd.cut(results_df['longitude'], bins=5, labels=['W', 'W-C', 'C', 'E-C', 'E'])
            
            results_df['lat_bin'] = lat_bins
            results_df['lon_bin'] = lon_bins
            results_df['spatial_bin'] = lat_bins.astype(str) + '-' + lon_bins.astype(str)
            
            spatial_accuracy = results_df.groupby('spatial_bin')['correct'].mean()
            spatial_analysis['spatial_binned_accuracy'] = spatial_accuracy
            
            if plot_maps:
                # Plot spatial accuracy map
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Accuracy by spatial bin
                spatial_pivot = results_df.groupby(['lat_bin', 'lon_bin'])['correct'].mean().unstack()
                sns.heatmap(spatial_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax1)
                ax1.set_title('Accuracy by Geographic Region')
                ax1.set_xlabel('Longitude Bin')
                ax1.set_ylabel('Latitude Bin')
                
                # Scatter plot of predictions
                colors = ['red' if not correct else 'green' for correct in results_df['correct']]
                ax2.scatter(results_df['longitude'], results_df['latitude'], 
                           c=colors, alpha=0.5, s=1)
                ax2.set_title('Prediction Accuracy (Red=Error, Green=Correct)')
                ax2.set_xlabel('Longitude')
                ax2.set_ylabel('Latitude')
                
                plt.tight_layout()
                plt.show()
        
        return spatial_analysis

Model Interpretation and Visualization
---------------------------------------

Attention Visualization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class AttentionVisualizer:
        """Visualize attention patterns in SITS-Former."""
        
        def __init__(self, model, device='cpu'):
            self.model = model
            self.device = device
            self.attention_maps = {}
            
            # Register hooks to capture attention
            self.hooks = []
            self._register_hooks()
        
        def _register_hooks(self):
            """Register forward hooks to capture attention weights."""
            
            def get_attention_hook(name):
                def hook(module, input, output):
                    # output is typically (attention_weights, attention_output)
                    if isinstance(output, tuple) and len(output) >= 2:
                        self.attention_maps[name] = output[0].detach()
                return hook
            
            # Register hooks on attention modules
            for name, module in self.model.named_modules():
                if 'attn' in name and hasattr(module, 'attention'):
                    hook = module.register_forward_hook(get_attention_hook(name))
                    self.hooks.append(hook)
        
        def get_attention_maps(self, sequence, mask=None):
            """Get attention maps for a sequence."""
            self.attention_maps = {}
            
            with torch.no_grad():
                if mask is not None:
                    _ = self.model(sequence.unsqueeze(0), mask.unsqueeze(0))
                else:
                    _ = self.model(sequence.unsqueeze(0))
            
            return self.attention_maps
        
        def visualize_temporal_attention(self, sequence, mask=None, layer_idx=-1):
            """Visualize temporal attention patterns."""
            
            attention_maps = self.get_attention_maps(sequence, mask)
            
            if not attention_maps:
                print("No attention maps captured. Check model architecture.")
                return
            
            # Get attention from specified layer
            layer_names = list(attention_maps.keys())
            if layer_idx < len(layer_names):
                layer_name = layer_names[layer_idx]
                attention = attention_maps[layer_name]  # [batch, heads, seq_len, seq_len]
                
                # Average over heads and batch
                attention_avg = attention.mean(dim=(0, 1)).cpu().numpy()
                
                # Plot attention matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(attention_avg, cmap='Blues', cbar=True)
                plt.title(f'Temporal Attention Pattern - {layer_name}')
                plt.xlabel('Key Position (Time Step)')
                plt.ylabel('Query Position (Time Step)')
                plt.show()
                
                return attention_avg
        
        def visualize_spatial_attention(self, sequence, mask=None, time_step=0, layer_idx=-1):
            """Visualize spatial attention for a specific time step."""
            
            attention_maps = self.get_attention_maps(sequence, mask)
            
            if not attention_maps:
                return
            
            layer_names = list(attention_maps.keys())
            if layer_idx < len(layer_names):
                layer_name = layer_names[layer_idx]
                attention = attention_maps[layer_name]
                
                # Focus on specific time step
                if attention.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    time_attention = attention[0, :, time_step, :].mean(dim=0)  # Average over heads
                    
                    # Reshape to spatial dimensions (assuming square patches)
                    spatial_size = int(np.sqrt(len(time_attention) - 1))  # -1 for CLS token
                    if spatial_size * spatial_size == len(time_attention) - 1:
                        spatial_attention = time_attention[1:].view(spatial_size, spatial_size)
                        
                        plt.figure(figsize=(8, 6))
                        plt.imshow(spatial_attention.cpu().numpy(), cmap='hot', interpolation='nearest')
                        plt.colorbar(label='Attention Weight')
                        plt.title(f'Spatial Attention - Time Step {time_step} - {layer_name}')
                        plt.show()
                        
                        return spatial_attention

    # Example usage
    visualizer = AttentionVisualizer(model, device)

Feature Importance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_feature_importance(model, test_loader, device, num_samples=100):
        """Analyze feature importance using gradient-based methods."""
        
        model.eval()
        model.requires_grad_(True)
        
        feature_importances = []
        
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            sequences = batch['sequence'].to(device)
            masks = batch.get('mask', None)
            if masks is not None:
                masks = masks.to(device)
            labels = batch['label'].to(device)
            
            sequences.requires_grad_(True)
            
            # Forward pass
            if masks is not None:
                outputs = model(sequences, masks)
            else:
                outputs = model(sequences)
            
            # Compute gradients for predicted class
            predicted_class = outputs.argmax(dim=1)
            loss = outputs[range(len(predicted_class)), predicted_class].sum()
            loss.backward()
            
            # Get gradients
            gradients = sequences.grad.data
            
            # Compute importance (gradient * input)
            importance = gradients * sequences
            importance = importance.abs().mean(dim=(2, 3, 4))  # Average over spatial and spectral
            
            feature_importances.append(importance.cpu().numpy())
        
        # Aggregate importance scores
        feature_importance = np.concatenate(feature_importances, axis=0)
        temporal_importance = feature_importance.mean(axis=0)
        
        # Plot temporal importance
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(temporal_importance)), temporal_importance, marker='o')
        plt.title('Temporal Feature Importance')
        plt.xlabel('Time Step')
        plt.ylabel('Importance Score')
        plt.grid(True)
        plt.show()
        
        return temporal_importance

Spectral Band Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_spectral_importance(model, test_sample, band_names=None, device='cpu'):
        """Analyze importance of different spectral bands."""
        
        if band_names is None:
            band_names = [f'Band_{i}' for i in range(test_sample.size(1))]
        
        model.eval()
        model.requires_grad_(True)
        
        # Baseline prediction
        test_input = test_sample.unsqueeze(0).to(device)
        test_input.requires_grad_(True)
        
        baseline_output = model(test_input)
        baseline_pred = baseline_output.argmax(dim=1).item()
        baseline_conf = torch.nn.functional.softmax(baseline_output, dim=1).max().item()
        
        # Test each band's importance by masking
        band_importance = []
        
        for band_idx in range(len(band_names)):
            # Mask out this band
            masked_input = test_input.clone()
            masked_input[:, :, band_idx, :, :] = 0
            
            masked_output = model(masked_input)
            masked_pred = masked_output.argmax(dim=1).item()
            masked_conf = torch.nn.functional.softmax(masked_output, dim=1).max().item()
            
            # Importance = change in confidence when band is removed
            importance = baseline_conf - masked_conf
            band_importance.append(importance)
        
        # Plot band importance
        plt.figure(figsize=(12, 6))
        bars = plt.bar(band_names, band_importance)
        plt.title('Spectral Band Importance')
        plt.xlabel('Spectral Band')
        plt.ylabel('Importance (Confidence Drop)')
        plt.xticks(rotation=45)
        
        # Color bars by importance
        max_importance = max(band_importance)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            bar.set_color(plt.cm.viridis(height / max_importance))
        
        plt.tight_layout()
        plt.show()
        
        return dict(zip(band_names, band_importance))

Error Analysis and Debugging
-----------------------------

Hard Example Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_hard_examples(test_results, test_loader, model, device, top_k=10):
        """Analyze the most challenging examples for the model."""
        
        predictions = test_results['predictions']
        targets = test_results['targets']
        probabilities = test_results['probabilities']
        
        # Find examples with lowest confidence in correct prediction
        correct_mask = predictions == targets
        correct_confidences = probabilities[correct_mask, targets[correct_mask]]
        
        # Find examples with highest confidence in wrong prediction
        wrong_mask = predictions != targets
        wrong_confidences = probabilities[wrong_mask, predictions[wrong_mask]]
        
        # Get indices of hard examples
        correct_indices = np.where(correct_mask)[0]
        wrong_indices = np.where(wrong_mask)[0]
        
        # Sort by difficulty
        hard_correct_idx = correct_indices[np.argsort(correct_confidences)[:top_k]]
        confident_wrong_idx = wrong_indices[np.argsort(wrong_confidences)[-top_k:]]
        
        print(f"Top {top_k} Hard Correct Examples (Low Confidence):")
        for i, idx in enumerate(hard_correct_idx):
            conf = correct_confidences[np.where(correct_indices == idx)[0][0]]
            print(f"  {i+1}. Index {idx}: True={targets[idx]}, Pred={predictions[idx]}, Conf={conf:.3f}")
        
        print(f"\\nTop {top_k} Confident Wrong Examples:")
        for i, idx in enumerate(confident_wrong_idx):
            conf = wrong_confidences[np.where(wrong_indices == idx)[0][0]]
            print(f"  {i+1}. Index {idx}: True={targets[idx]}, Pred={predictions[idx]}, Conf={conf:.3f}")
        
        return {
            'hard_correct_indices': hard_correct_idx,
            'confident_wrong_indices': confident_wrong_idx
        }

Class Imbalance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_class_imbalance_impact(test_results, class_names):
        """Analyze how class imbalance affects model performance."""
        
        targets = test_results['targets']
        predictions = test_results['predictions']
        
        # Class distribution
        unique_targets, target_counts = np.unique(targets, return_counts=True)
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        
        # Create distribution comparison
        class_stats = pd.DataFrame({
            'Class': [class_names[i] for i in unique_targets],
            'True_Count': target_counts,
            'Pred_Count': [pred_counts[np.where(unique_preds == i)[0][0]] if i in unique_preds else 0 
                          for i in unique_targets],
        })
        
        class_stats['True_Frequency'] = class_stats['True_Count'] / class_stats['True_Count'].sum()
        class_stats['Pred_Frequency'] = class_stats['Pred_Count'] / class_stats['Pred_Count'].sum()
        class_stats['Frequency_Diff'] = class_stats['Pred_Frequency'] - class_stats['True_Frequency']
        
        # Compute per-class accuracy
        class_accuracy = []
        for class_idx in unique_targets:
            class_mask = targets == class_idx
            class_correct = (predictions[class_mask] == class_idx).sum()
            class_total = class_mask.sum()
            accuracy = class_correct / class_total if class_total > 0 else 0
            class_accuracy.append(accuracy)
        
        class_stats['Accuracy'] = class_accuracy
        
        # Plot analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        x = np.arange(len(class_stats))
        width = 0.35
        ax1.bar(x - width/2, class_stats['True_Count'], width, label='True', alpha=0.7)
        ax1.bar(x + width/2, class_stats['Pred_Count'], width, label='Predicted', alpha=0.7)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution: True vs Predicted')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_stats['Class'], rotation=45)
        ax1.legend()
        
        # Accuracy vs frequency
        ax2.scatter(class_stats['True_Frequency'], class_stats['Accuracy'], alpha=0.7)
        ax2.set_xlabel('True Class Frequency')
        ax2.set_ylabel('Class Accuracy')
        ax2.set_title('Accuracy vs Class Frequency')
        
        # Add trend line
        z = np.polyfit(class_stats['True_Frequency'], class_stats['Accuracy'], 1)
        p = np.poly1d(z)
        ax2.plot(class_stats['True_Frequency'], p(class_stats['True_Frequency']), "r--", alpha=0.8)
        
        # Frequency difference
        colors = ['red' if diff < 0 else 'green' for diff in class_stats['Frequency_Diff']]
        ax3.bar(range(len(class_stats)), class_stats['Frequency_Diff'], color=colors, alpha=0.7)
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Frequency Difference (Pred - True)')
        ax3.set_title('Prediction Bias by Class')
        ax3.set_xticks(range(len(class_stats)))
        ax3.set_xticklabels(class_stats['Class'], rotation=45)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Per-class accuracy
        ax4.bar(range(len(class_stats)), class_stats['Accuracy'], color='skyblue', alpha=0.7)
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Per-Class Accuracy')
        ax4.set_xticks(range(len(class_stats)))
        ax4.set_xticklabels(class_stats['Class'], rotation=45)
        ax4.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
        
        return class_stats

Model Robustness Testing
-------------------------

Noise Robustness
~~~~~~~~~~~~~~~~

.. code-block:: python

    def test_noise_robustness(model, test_sample, device, noise_levels=[0.01, 0.05, 0.1, 0.2]):
        """Test model robustness to input noise."""
        
        model.eval()
        
        original_input = test_sample.unsqueeze(0).to(device)
        
        with torch.no_grad():
            original_output = model(original_input)
            original_pred = original_output.argmax(dim=1).item()
            original_conf = torch.nn.functional.softmax(original_output, dim=1).max().item()
        
        robustness_results = []
        
        for noise_level in noise_levels:
            # Add Gaussian noise
            noise = torch.randn_like(original_input) * noise_level
            noisy_input = original_input + noise
            
            with torch.no_grad():
                noisy_output = model(noisy_input)
                noisy_pred = noisy_output.argmax(dim=1).item()
                noisy_conf = torch.nn.functional.softmax(noisy_output, dim=1).max().item()
            
            prediction_changed = original_pred != noisy_pred
            confidence_drop = original_conf - noisy_conf
            
            robustness_results.append({
                'noise_level': noise_level,
                'prediction_changed': prediction_changed,
                'confidence_drop': confidence_drop,
                'original_conf': original_conf,
                'noisy_conf': noisy_conf
            })
        
        # Plot robustness
        noise_levels_plot = [r['noise_level'] for r in robustness_results]
        conf_drops = [r['confidence_drop'] for r in robustness_results]
        pred_changes = [r['prediction_changed'] for r in robustness_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence drop
        ax1.plot(noise_levels_plot, conf_drops, 'o-', color='blue')
        ax1.set_xlabel('Noise Level (σ)')
        ax1.set_ylabel('Confidence Drop')
        ax1.set_title('Model Confidence vs Noise Level')
        ax1.grid(True)
        
        # Prediction changes
        pred_change_binary = [1 if pc else 0 for pc in pred_changes]
        ax2.plot(noise_levels_plot, pred_change_binary, 'o-', color='red')
        ax2.set_xlabel('Noise Level (σ)')
        ax2.set_ylabel('Prediction Changed')
        ax2.set_title('Prediction Stability vs Noise Level')
        ax2.set_ylim([-0.1, 1.1])
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return robustness_results

Comprehensive Evaluation Report
-------------------------------

.. code-block:: python

    class ModelEvaluationReport:
        """Generate comprehensive evaluation report."""
        
        def __init__(self, model, test_loader, class_names, device='cpu'):
            self.model = model
            self.test_loader = test_loader
            self.class_names = class_names
            self.device = device
            self.results = None
            
        def generate_full_report(self, save_path='model_evaluation_report.html'):
            """Generate complete evaluation report."""
            
            print("Generating comprehensive evaluation report...")
            
            # Generate predictions
            self.results = generate_predictions(self.model, self.test_loader, self.device)
            
            # Compute all metrics
            per_class_metrics, overall_metrics = compute_classification_metrics(
                self.results['targets'], 
                self.results['predictions'],
                self.results['probabilities'],
                self.class_names
            )
            
            # Create HTML report
            html_content = f"""
            <html>
            <head>
                <title>SITS-Former Model Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ margin: 10px 0; }}
                    .section {{ margin: 30px 0; border-left: 3px solid #4CAF50; padding-left: 15px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>SITS-Former Model Evaluation Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Overall Performance</h2>
                    <div class="metric">Total Samples: {len(self.results['targets'])}</div>
                    <div class="metric">Number of Classes: {len(self.class_names)}</div>
            """
            
            for metric, value in overall_metrics.items():
                html_content += f'<div class="metric">{metric}: {value:.4f}</div>\\n'
            
            html_content += f"""
                </div>
                
                <div class="section">
                    <h2>Per-Class Performance</h2>
                    {per_class_metrics.to_html(classes='table-striped')}
                </div>
                
                <div class="section">
                    <h2>Model Summary</h2>
                    <div class="metric">Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}</div>
                    <div class="metric">Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}</div>
                </div>
                
            </body>
            </html>
            """
            
            # Save report
            with open(save_path, 'w') as f:
                f.write(html_content)
            
            print(f"Evaluation report saved to: {save_path}")
            
            return {
                'per_class_metrics': per_class_metrics,
                'overall_metrics': overall_metrics,
                'predictions': self.results
            }

# Example usage
evaluator = ModelEvaluationReport(model, test_loader, class_names, device)
evaluation_results = evaluator.generate_full_report()

Best Practices for Model Evaluation
-----------------------------------

1. **Comprehensive Testing**
   - Test on truly held-out data
   - Include diverse geographic and temporal conditions
   - Test edge cases and challenging scenarios

2. **Multiple Metrics**
   - Don't rely solely on accuracy
   - Consider precision, recall, and F1-score
   - Use confusion matrices for detailed analysis

3. **Domain-Specific Analysis**
   - Analyze temporal and spatial patterns
   - Consider seasonal variations
   - Test across different sensor configurations

4. **Error Analysis**
   - Identify systematic error patterns
   - Analyze class-specific performance
   - Investigate challenging examples

5. **Robustness Testing**
   - Test with noisy inputs
   - Evaluate under different conditions
   - Check for domain transfer capabilities

Next Steps
----------

After comprehensive evaluation:

1. Identify areas for improvement: :doc:`fine_tuning`
2. Deploy your model: :doc:`deployment`
3. Monitor production performance: :doc:`monitoring`

Remember that evaluation is an iterative process - use insights from analysis to guide model improvements and retraining strategies.